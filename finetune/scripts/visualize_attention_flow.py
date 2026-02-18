import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from transformers import set_seed

import decord

from finetune.utils.RAFT.raft_bi import RAFT_bi
from finetune.utils.RAFT.utils.flow_viz import flow_to_image


decord.bridge.set_bridge("torch")


def preprocess_video_match(video_path: Path | str, is_match: bool = True) -> Tuple[torch.Tensor, int, int, int, Tuple[int, int, int, int]]:
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    f_count, height, width, channels = frames.shape
    original_shape = (f_count, height, width, channels)

    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (f_count - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        pad_h = (16 - height % 16) % 16
        pad_w = (16 - width % 16) % 16
        if pad_h > 0 or pad_w > 0:
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))

    frames = frames.float().permute(0, 3, 1, 2).contiguous()
    return frames, pad_f, pad_h, pad_w, original_shape


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    if transformer_config.patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )
    return freqs_cos, freqs_sin


def _reshape_heads(attn_module, tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(attn_module, "head_to_batch_dim"):
        batch = tensor.shape[0]
        head_batch = attn_module.head_to_batch_dim(tensor)
        heads = head_batch.shape[0] // batch
        return head_batch.view(batch, heads, head_batch.shape[1], head_batch.shape[2])

    heads = int(getattr(attn_module, "heads", 1))
    dim = tensor.shape[-1] // heads
    return tensor.view(tensor.shape[0], tensor.shape[1], heads, dim).permute(0, 2, 1, 3)


def _as_uint8_rgb(frame_chw: torch.Tensor) -> np.ndarray:
    frame = frame_chw.detach().cpu().permute(1, 2, 0).numpy()
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def _flow_to_image_rgb(flow_hw2: np.ndarray) -> np.ndarray:
    return flow_to_image(flow_hw2)


def _epe_colormap(epe_map: np.ndarray) -> np.ndarray:
    max_val = float(np.percentile(epe_map, 99.0))
    max_val = max(max_val, 1e-6)
    norm = np.clip(epe_map / max_val, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stack_panels(panels: List[np.ndarray]) -> np.ndarray:
    height = min(panel.shape[0] for panel in panels)
    resized = []
    for panel in panels:
        if panel.shape[0] != height:
            scale = height / panel.shape[0]
            width = int(round(panel.shape[1] * scale))
            panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_LINEAR)
        resized.append(panel)
    return np.concatenate(resized, axis=1)


class AttentionCapture:
    def __init__(
        self,
        transformer,
        tokens_per_frame: int,
        num_temporal_tokens: int,
        src_frame_idx: int,
        dst_frame_idx: int,
    ):
        self.transformer = transformer
        self.tokens_per_frame = tokens_per_frame
        self.num_temporal_tokens = num_temporal_tokens
        self.src_frame_idx = src_frame_idx
        self.dst_frame_idx = dst_frame_idx
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.affinity_by_layer: Dict[str, torch.Tensor] = {}
        self.layer_order: List[str] = []
        self.video_tokens = tokens_per_frame * num_temporal_tokens

    def _q_hook(self, layer_name: str):
        def _hook(_module, _inputs, output):
            self.cache.setdefault(layer_name, {})["q"] = output
        return _hook

    def _k_hook(self, layer_name: str):
        def _hook(_module, _inputs, output):
            self.cache.setdefault(layer_name, {})["k"] = output
        return _hook

    def _attn_hook(self, layer_name: str, attn_module):
        def _hook(_module, _inputs, _output):
            layer_cache = self.cache.get(layer_name, {})
            q = layer_cache.get("q")
            k = layer_cache.get("k")
            if q is None or k is None:
                return

            with torch.no_grad():
                qh = _reshape_heads(attn_module, q)
                kh = _reshape_heads(attn_module, k)

                query_len = qh.shape[2]
                key_len = kh.shape[2]
                if query_len < self.video_tokens or key_len < self.video_tokens:
                    return

                q_video = qh[:, :, -self.video_tokens:, :]
                k_video = kh[:, :, -self.video_tokens:, :]

                src_start = self.src_frame_idx * self.tokens_per_frame
                src_end = src_start + self.tokens_per_frame
                dst_start = self.dst_frame_idx * self.tokens_per_frame
                dst_end = dst_start + self.tokens_per_frame

                if src_end > q_video.shape[2] or dst_end > k_video.shape[2]:
                    return

                q_src = q_video[:, :, src_start:src_end, :].float()
                k_dst = k_video[:, :, dst_start:dst_end, :].float()

                scale = 1.0 / math.sqrt(q_src.shape[-1])
                affinity = torch.matmul(q_src, k_dst.transpose(-1, -2)) * scale
                affinity = affinity.mean(dim=1).squeeze(0).detach().cpu()

                self.affinity_by_layer[layer_name] = affinity

            self.cache[layer_name] = {}
        return _hook

    def register(self):
        for layer_name, module in self.transformer.named_modules():
            if not (hasattr(module, "to_q") and hasattr(module, "to_k")):
                continue
            if "attention" not in module.__class__.__name__.lower() and "attn" not in layer_name.lower():
                continue

            self.layer_order.append(layer_name)
            self.handles.append(module.to_q.register_forward_hook(self._q_hook(layer_name)))
            self.handles.append(module.to_k.register_forward_hook(self._k_hook(layer_name)))
            self.handles.append(module.register_forward_hook(self._attn_hook(layer_name, module)))

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _compute_layer_correspondence(
    affinity: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = torch.softmax(affinity, dim=-1)
    confidence, best_idx = probs.max(dim=-1)

    src_y = torch.arange(grid_h, dtype=torch.float32).view(grid_h, 1).repeat(1, grid_w)
    src_x = torch.arange(grid_w, dtype=torch.float32).view(1, grid_w).repeat(grid_h, 1)
    src_x = src_x.reshape(-1)
    src_y = src_y.reshape(-1)

    dst_y = torch.div(best_idx, grid_w, rounding_mode="floor").float()
    dst_x = (best_idx % grid_w).float()

    dx = (dst_x - src_x).view(grid_h, grid_w).cpu().numpy()
    dy = (dst_y - src_y).view(grid_h, grid_w).cpu().numpy()
    conf = confidence.view(grid_h, grid_w).cpu().numpy()

    disp = np.stack([dx, dy], axis=-1)
    return disp, conf


def _flow_token_grid(flow_px_2hw: torch.Tensor, grid_h: int, grid_w: int, stride_px: int) -> np.ndarray:
    flow_1 = flow_px_2hw.unsqueeze(0)
    flow_resized = F.interpolate(flow_1, size=(grid_h, grid_w), mode="bilinear", align_corners=False)[0]
    flow_token = (flow_resized / float(stride_px)).permute(1, 2, 0).detach().cpu().numpy()
    return flow_token


def _token_flow_to_pixel_flow(flow_token_hw2: np.ndarray, out_h: int, out_w: int, stride_px: int) -> np.ndarray:
    flow_px_grid = flow_token_hw2 * float(stride_px)
    flow_px_full = cv2.resize(flow_px_grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return flow_px_full


def _run_single_forward(
    pipe: CogVideoXPipeline,
    video_input_1cfhw: torch.Tensor,
    src_prompt: str,
    sr_noise_step: int,
    noise_step: int,
):
    video_for_vae = video_input_1cfhw.to(pipe.vae.device, dtype=pipe.vae.dtype)
    latent = pipe.vae.encode(video_for_vae).latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size_t = pipe.transformer.config.patch_size_t
    ncopy = 0
    if patch_size_t is not None:
        ncopy = latent.shape[2] % patch_size_t
        if ncopy > 0:
            first_frame = latent[:, :, :1, :, :]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)

    batch_size, _, num_frames, latent_h, latent_w = latent.shape

    tokens = pipe.tokenizer(
        src_prompt,
        padding="max_length",
        max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids
    prompt_embedding = pipe.text_encoder(tokens.to(latent.device))[0]
    prompt_embedding = prompt_embedding.view(batch_size, prompt_embedding.shape[1], -1).to(dtype=latent.dtype)

    latent_input = latent.permute(0, 2, 1, 3, 4)

    if noise_step != 0:
        noise = torch.randn_like(latent_input)
        add_timesteps = torch.full((batch_size,), fill_value=noise_step, dtype=torch.long, device=latent_input.device)
        latent_input = pipe.scheduler.add_noise(latent_input, noise, add_timesteps)

    timesteps = torch.full((batch_size,), fill_value=sr_noise_step, dtype=torch.long, device=latent_input.device)

    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=latent_h * vae_scale_factor_spatial,
            width=latent_w * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=pipe.transformer.config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=latent_input.device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    _ = pipe.transformer(
        hidden_states=latent_input,
        encoder_hidden_states=prompt_embedding,
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        return_dict=False,
    )[0]

    grid_h = latent_h // pipe.transformer.config.patch_size
    grid_w = latent_w // pipe.transformer.config.patch_size

    if patch_size_t is None:
        temporal_tokens = num_frames
        temporal_patch = 1
    else:
        temporal_tokens = (num_frames + patch_size_t - 1) // patch_size_t
        temporal_patch = patch_size_t

    return {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "temporal_tokens": temporal_tokens,
        "temporal_patch": temporal_patch,
        "num_frames_latent": num_frames,
        "ncopy": ncopy,
        "vae_scale": vae_scale_factor_spatial,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize per-layer transformer attention correspondence against optical flow")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained DOVE model")
    parser.add_argument("--output_dir", type=str, default="./attn_flow_vis", help="Directory to save outputs")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--upscale_mode", type=str, default="bilinear")
    parser.add_argument("--src_frame", type=int, default=0, help="Source frame index in temporal tokens")
    parser.add_argument("--dst_frame", type=int, default=1, help="Destination frame index in temporal tokens")
    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)
    parser.add_argument("--raft_iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    frames_fchw, _, _, _, _ = preprocess_video_match(args.input, is_match=True)
    frames_fchw = F.interpolate(frames_fchw, scale_factor=args.upscale, mode=args.upscale_mode, align_corners=False)
    frames_norm = frames_fchw / 255.0
    model_video = (frames_norm * 2.0 - 1.0).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(args.device)
    pipe.vae.eval()
    pipe.transformer.eval()
    pipe.text_encoder.eval()

    meta_dummy = _run_single_forward(
        pipe=pipe,
        video_input_1cfhw=model_video,
        src_prompt=args.prompt,
        sr_noise_step=args.sr_noise_step,
        noise_step=args.noise_step,
    )

    grid_h = meta_dummy["grid_h"]
    grid_w = meta_dummy["grid_w"]
    temporal_tokens = meta_dummy["temporal_tokens"]
    src_idx = max(0, min(args.src_frame, temporal_tokens - 1))
    dst_idx = max(0, min(args.dst_frame, temporal_tokens - 1))
    if src_idx == dst_idx:
        raise ValueError("src_frame and dst_frame must be different")

    capture = AttentionCapture(
        transformer=pipe.transformer,
        tokens_per_frame=grid_h * grid_w,
        num_temporal_tokens=temporal_tokens,
        src_frame_idx=src_idx,
        dst_frame_idx=dst_idx,
    )
    capture.register()

    _ = _run_single_forward(
        pipe=pipe,
        video_input_1cfhw=model_video,
        src_prompt=args.prompt,
        sr_noise_step=args.sr_noise_step,
        noise_step=args.noise_step,
    )
    capture.clear()

    if not capture.affinity_by_layer:
        raise RuntimeError("No attention affinity was captured. Check diffusers version compatibility.")

    flow_model = RAFT_bi(model_path="./finetune/utils/RAFT/raft-things.pth", device=args.device)
    flow_f, _ = flow_model.forward_slicing(frames_norm.unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device), iters=args.raft_iters)

    temporal_patch = meta_dummy["temporal_patch"]
    src_raw_frame = src_idx * temporal_patch
    dst_raw_frame = dst_idx * temporal_patch
    if dst_raw_frame <= src_raw_frame:
        src_raw_frame, dst_raw_frame = dst_raw_frame, src_raw_frame

    frame_step = dst_raw_frame - src_raw_frame
    if frame_step != 1:
        raise ValueError(f"This script currently supports adjacent frame pairs only. Got step={frame_step}.")

    flow_pair = flow_f[0, :, src_raw_frame].detach().float().cpu()

    stride_px = meta_dummy["vae_scale"] * pipe.transformer.config.patch_size
    flow_token = _flow_token_grid(flow_pair, grid_h, grid_w, stride_px=stride_px)
    flow_pair_px = flow_pair.permute(1, 2, 0).numpy()

    src_frame_rgb = _as_uint8_rgb(frames_norm[src_raw_frame])
    flow_rgb = _flow_to_image_rgb(flow_pair_px)
    full_h, full_w = src_frame_rgb.shape[:2]

    results = {
        "meta": {
            "input": args.input,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "temporal_tokens": temporal_tokens,
            "src_token_idx": src_idx,
            "dst_token_idx": dst_idx,
            "src_raw_frame": src_raw_frame,
            "dst_raw_frame": dst_raw_frame,
            "stride_px": stride_px,
        },
        "layers": {},
    }

    layer_names = [name for name in capture.layer_order if name in capture.affinity_by_layer]
    for layer_i, layer_name in enumerate(layer_names):
        affinity = capture.affinity_by_layer[layer_name]
        attn_disp_token, conf = _compute_layer_correspondence(affinity, grid_h, grid_w)
        epe_token = np.sqrt(np.sum((attn_disp_token - flow_token) ** 2, axis=-1))

        attn_disp_px = _token_flow_to_pixel_flow(attn_disp_token, out_h=full_h, out_w=full_w, stride_px=stride_px)
        epe_px = np.sqrt(np.sum((attn_disp_px - flow_pair_px) ** 2, axis=-1))

        attn_flow_rgb = _flow_to_image_rgb(attn_disp_px)
        epe_rgb = _epe_colormap(epe_px)

        panel = _stack_panels([src_frame_rgb, flow_rgb, attn_flow_rgb, epe_rgb])
        panel_path = os.path.join(args.output_dir, f"layer_{layer_i:03d}.png")
        cv2.imwrite(panel_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        conf_mean = float(conf.mean())
        epe_token_mean = float(epe_token.mean())
        epe_token_median = float(np.median(epe_token))
        epe_px_mean = float(epe_px.mean())
        epe_px_median = float(np.median(epe_px))
        results["layers"][layer_name] = {
            "index": layer_i,
            "panel_path": panel_path,
            "confidence_mean": conf_mean,
            "epe_mean_token": epe_token_mean,
            "epe_median_token": epe_token_median,
            "epe_mean_pixel": epe_px_mean,
            "epe_median_pixel": epe_px_median,
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    print(f"Saved {len(layer_names)} layer panels to: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()