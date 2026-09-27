#!/usr/bin/env python3
"""Train the static PointForward-style decoder with frozen Turbo RGB features."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DISTT_SCRIPTS = REPO_ROOT / "third_party" / "distt" / "scripts"
COGVIDEOX_SCALING_FACTOR = 1.15258426
VIEW_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--turbo-repo-root", type=Path, required=True)
    parser.add_argument("--turbo-config", type=Path, default=None)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--turbo-feature-key", default="up_block_2")
    parser.add_argument("--latent-scale", type=float, default=1.0 / COGVIDEOX_SCALING_FACTOR)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--target-frames", type=int, nargs="+", default=[0, 6, 12, 16])
    parser.add_argument("--target-views", nargs="+", default=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"])
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument(
        "--targets-per-step", type=int, default=0,
        help="Random target images rendered per step; 0 uses every selected target.",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--fusion-type", choices=["mlp", "xattn"], default="mlp")
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--use-projection-valid-mask", action="store_true")
    parser.add_argument("--learned-local-sampling", action="store_true")
    parser.add_argument("--local-sampling-radius-px", type=float, default=4.0)
    parser.add_argument("--local-offset-reg-weight", type=float, default=0.0)
    parser.add_argument("--iterative-refinement-layers", type=int, default=0)
    parser.add_argument("--refinement-hidden-dim", type=int, default=64)
    parser.add_argument("--refinement-fusion-type", choices=["mlp", "xattn"], default="mlp")
    parser.add_argument("--refinement-attention-heads", type=int, default=8)
    parser.add_argument(
        "--refinement-sample-raw-rgb",
        action="store_true",
        help="Sample the packaged full-resolution RGB target directly instead of a learned shallow RGB feature map.",
    )
    parser.add_argument(
        "--refinement-turbo-only",
        action="store_true",
        help="Use only the sampled Turbo VAE feature in refinement, without an RGB feature branch.",
    )
    parser.add_argument("--fullres-feature-dim", type=int, default=16)
    parser.add_argument("--refinement-step-m", type=float, default=0.5)
    parser.add_argument(
        "--flow-track-refinement",
        action="store_true",
        help="Refine the Stage1 flow-RGB initialized per-frame 3D track instead of one static anchor.",
    )
    parser.add_argument("--gaussians-per-query", type=int, default=1)
    parser.add_argument("--appearance-mode", choices=["direct_rgb", "feature_unet"], default="direct_rgb")
    parser.add_argument("--color-feature-dim", type=int, default=32)
    parser.add_argument("--unet-base-channels", type=int, default=32)
    parser.add_argument(
        "--target-camera-conditioning",
        action="store_true",
        help="Condition the shared feature-render UNet on the target physical camera.",
    )
    parser.add_argument("--camera-embedding-dim", type=int, default=16)
    parser.add_argument("--min-scale-m", type=float, default=0.03)
    parser.add_argument("--max-scale-m", type=float, default=1.2)
    parser.add_argument("--max-delta-m", type=float, default=0.8)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--rgb-loss", choices=["l1", "charbonnier"], default="charbonnier")
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--lpips-module-root", type=Path, default=None)
    parser.add_argument("--predict-lifespan", action="store_true")
    parser.add_argument("--opacity-reg-weight", type=float, default=0.001)
    parser.add_argument("--scale-reg-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--height", type=int, default=424)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def read_manifest_row(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} does not contain row {index}")


def view_to_index(view: str) -> int:
    return int(view) if view.isdigit() else VIEW_NAMES.index(view)


def as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating):
        values = rgb * 255.0 if float(np.nanmax(rgb)) <= 1.5 else rgb
        return np.floor(np.clip(values, 0.0, 255.0) + 0.5).astype(np.uint8)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def load_stage1(stage1_dir: Path, device: torch.device) -> dict:
    query_npz = np.load(stage1_dir / "stage1_static_queries.npz", allow_pickle=True)
    obs_npz = np.load(stage1_dir / "stage1_observations.npz", allow_pickle=True)
    columns = [str(item) for item in query_npz["columns"].tolist()]
    return {
        "col": {name: idx for idx, name in enumerate(columns)},
        "queries": torch.from_numpy(query_npz["queries"]).float().to(device),
        "observations": {key: torch.from_numpy(obs_npz[key]).to(device) for key in obs_npz.files},
    }


def build_turbo_decoder(args: argparse.Namespace, device: torch.device):
    if str(DISTT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(DISTT_SCRIPTS))
    from compare_turbo_vaed_cog_decoder import build_turbo_decoder as build_decoder

    decoder_args = argparse.Namespace(
        turbo_repo_root=str(args.turbo_repo_root),
        turbo_config=str(args.turbo_config) if args.turbo_config is not None else None,
        turbo_checkpoint=str(args.turbo_checkpoint),
        turbo_decoder_device=None,
        turbo_decoder_dtype="fp16",
        enable_slicing=False,
        enable_tiling=False,
        framewise_decoding=False,
    )
    return build_decoder(decoder_args, device)


@torch.no_grad()
def decode_context_features(
    clip: dict,
    context_frames: torch.Tensor,
    context_views: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    decoder, load_info = build_turbo_decoder(args, device)
    unique_views = sorted(set(context_views.tolist()))
    latent = clip["rgb_latent"][unique_views].to(device=device, dtype=torch.float16) * args.latent_scale
    _, features = decoder.decode(latent, feature_enabled=True)
    if args.turbo_feature_key not in features:
        raise KeyError(f"Turbo feature {args.turbo_feature_key!r} is unavailable: {sorted(features)}")
    feature_video = features[args.turbo_feature_key]
    view_slot = {view: slot for slot, view in enumerate(unique_views)}
    selected = [
        feature_video[view_slot[int(view)], :, int(frame)]
        for frame, view in zip(context_frames.tolist(), context_views.tolist(), strict=True)
    ]
    result = torch.stack(selected).float()
    feature_shape = list(result.shape)
    del decoder, latent, features, feature_video
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, {"load_info": load_info, "feature_shape": feature_shape, "unique_views": unique_views}


def sample_context_features(
    feature_maps: torch.Tensor,
    obs_uv: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    sampled_contexts = []
    for context_index, feature in enumerate(feature_maps):
        x_norm = (obs_uv[:, context_index, 0] / max(width - 1, 1)) * 2.0 - 1.0
        y_norm = (obs_uv[:, context_index, 1] / max(height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            feature[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled_contexts.append(sampled[0, :, :, 0].transpose(0, 1))
    return torch.stack(sampled_contexts, dim=1)


class LearnedLocalFeatureSampler(nn.Module):
    """Predict bounded image-space offsets and bilinearly sample local feature patches."""

    def __init__(self, query_dim: int, feature_dim: int, hidden_dim: int, radius_px: float) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.radius_px = float(radius_px)
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    def forward(
        self,
        feature_maps: torch.Tensor,
        center_features: torch.Tensor,
        obs_uv: torch.Tensor,
        query_features: torch.Tensor,
        depth_difference: torch.Tensor,
        same_time: torch.Tensor,
        query_time: torch.Tensor,
        context_time: torch.Tensor,
        context_view: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_encoder(query_features)
        sampled_contexts, offsets = [], []
        for context_index, feature_map in enumerate(feature_maps):
            cues = torch.cat(
                [
                    depth_difference[:, context_index],
                    same_time[:, context_index],
                    query_time,
                    context_time[:, context_index],
                    context_view[:, context_index],
                ],
                dim=-1,
            )
            center = self.feature_encoder(center_features[:, context_index])
            offset = torch.tanh(self.offset_mlp(torch.cat([query, center, cues], dim=-1))) * self.radius_px
            shifted_uv = obs_uv[:, context_index] + offset
            shifted_uv = torch.stack(
                [shifted_uv[:, 0].clamp(0.0, width - 1.0), shifted_uv[:, 1].clamp(0.0, height - 1.0)], dim=-1
            )
            x_norm = (shifted_uv[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
            y_norm = (shifted_uv[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
            grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                feature_map[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            sampled_contexts.append(sampled[0, :, :, 0].transpose(0, 1))
            offsets.append(offset)
        return torch.stack(sampled_contexts, dim=1), torch.stack(offsets, dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(self.norm(features))


class PointQueryRefinementBlock(nn.Module):
    """Fuse reprojected image observations and update query state and position."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        fusion_type: str = "mlp",
        attention_heads: int = 8,
        trajectory_refinement: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.trajectory_refinement = bool(trajectory_refinement)
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.temporal_encoder = nn.Linear(2, hidden_dim) if fusion_type == "xattn" else None
        self.cross_attention = (
            nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
            if fusion_type == "xattn"
            else None
        )
        self.feature_update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
        )
        self.position_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 2) if self.trajectory_refinement else nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * 2 + 2 if self.trajectory_refinement else hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        observations: torch.Tensor,
        observation_valid: torch.Tensor,
        same_time: torch.Tensor,
        query_time: torch.Tensor,
        step_m: float,
        context_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.observation_encoder(observations)
        temporal_value = (
            context_time
            if self.trajectory_refinement and context_time is not None
            else query_time[:, None].expand(-1, spatial.shape[1], -1)
        )
        temporal = torch.cat([same_time, temporal_value], dim=-1)
        if self.fusion_type == "xattn":
            key_value = spatial + self.temporal_encoder(temporal)
            fused, weights = self.cross_attention(
                query[:, None],
                key_value,
                key_value,
                key_padding_mask=~observation_valid,
                need_weights=True,
            )
            fused = fused[:, 0]
            weights = weights[:, 0]
        else:
            logits = self.weight_mlp(
                torch.cat([query[:, None] * spatial, temporal], dim=-1)
            ).squeeze(-1)
            logits = logits.masked_fill(~observation_valid, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
            fused = (weights[..., None] * spatial).sum(dim=1)
        query = query + self.feature_update(torch.cat([query, fused, query_time], dim=-1))
        if self.trajectory_refinement:
            position_features = torch.cat(
                [query[:, None].expand(-1, spatial.shape[1], -1), spatial, temporal], dim=-1
            )
            delta = torch.tanh(self.position_head(position_features)) * step_m
        else:
            delta = torch.tanh(self.position_head(query)) * step_m
        return query, delta, weights


class FullResolutionRGBEncoder(nn.Module):
    """Shallow pixel-aligned encoder that preserves RGB resolution."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def sample_reprojected_features(
    points_ref: torch.Tensor,
    turbo_feature_maps: torch.Tensor,
    fullres_feature_maps: torch.Tensor | None,
    context_viewmats: torch.Tensor,
    context_intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((points_ref.shape[0], 1), dtype=points_ref.dtype, device=points_ref.device)
    homogeneous = torch.cat([points_ref, ones], dim=-1)
    sampled_contexts, valid_contexts = [], []
    for context_index in range(context_viewmats.shape[0]):
        camera_points = homogeneous @ context_viewmats[context_index].transpose(0, 1)
        z = camera_points[:, 2]
        intrinsics = context_intrinsics[context_index]
        u = camera_points[:, 0] / z.clamp_min(1.0e-6) * intrinsics[0, 0] + intrinsics[0, 2]
        v = camera_points[:, 1] / z.clamp_min(1.0e-6) * intrinsics[1, 1] + intrinsics[1, 2]
        valid = (z > 0.1) & (u >= 0.0) & (u <= width - 1.0) & (v >= 0.0) & (v <= height - 1.0)
        grid = torch.stack(
            [(u / max(width - 1, 1)) * 2.0 - 1.0, (v / max(height - 1, 1)) * 2.0 - 1.0],
            dim=-1,
        ).view(1, -1, 1, 2)
        turbo = F.grid_sample(
            turbo_feature_maps[context_index : context_index + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[0, :, :, 0].transpose(0, 1)
        if fullres_feature_maps is None:
            sampled_contexts.append(turbo)
        else:
            fullres = F.grid_sample(
                fullres_feature_maps[context_index : context_index + 1], grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )[0, :, :, 0].transpose(0, 1)
            sampled_contexts.append(torch.cat([turbo, fullres], dim=-1))
        valid_contexts.append(valid)
    observation_valid = torch.stack(valid_contexts, dim=1)
    no_valid = ~observation_valid.any(dim=1)
    observation_valid[no_valid, 0] = True
    return torch.stack(sampled_contexts, dim=1), observation_valid


def sample_tracked_features(
    points_ref_by_context: torch.Tensor,
    turbo_feature_maps: torch.Tensor,
    fullres_feature_maps: torch.Tensor | None,
    context_viewmats: torch.Tensor,
    context_intrinsics: torch.Tensor,
    width: int,
    height: int,
    track_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample context c at the query's flow-initialized/refined position for context c."""
    sampled_contexts, valid_contexts = [], []
    for context_index in range(context_viewmats.shape[0]):
        points_ref = points_ref_by_context[:, context_index]
        ones = torch.ones((points_ref.shape[0], 1), dtype=points_ref.dtype, device=points_ref.device)
        camera_points = torch.cat([points_ref, ones], dim=-1) @ context_viewmats[context_index].transpose(0, 1)
        z = camera_points[:, 2]
        intrinsics = context_intrinsics[context_index]
        u = camera_points[:, 0] / z.clamp_min(1.0e-6) * intrinsics[0, 0] + intrinsics[0, 2]
        v = camera_points[:, 1] / z.clamp_min(1.0e-6) * intrinsics[1, 1] + intrinsics[1, 2]
        valid = (z > 0.1) & (u >= 0.0) & (u <= width - 1.0) & (v >= 0.0) & (v <= height - 1.0)
        if track_valid is not None:
            valid = valid & track_valid[:, context_index]
        grid = torch.stack(
            [(u / max(width - 1, 1)) * 2.0 - 1.0, (v / max(height - 1, 1)) * 2.0 - 1.0], dim=-1
        ).view(1, -1, 1, 2)
        turbo = F.grid_sample(
            turbo_feature_maps[context_index : context_index + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[0, :, :, 0].transpose(0, 1)
        if fullres_feature_maps is None:
            sampled_contexts.append(turbo)
        else:
            fullres = F.grid_sample(
                fullres_feature_maps[context_index : context_index + 1], grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )[0, :, :, 0].transpose(0, 1)
            sampled_contexts.append(torch.cat([turbo, fullres], dim=-1))
        valid_contexts.append(valid)
    observation_valid = torch.stack(valid_contexts, dim=1)
    no_valid = ~observation_valid.any(dim=1)
    observation_valid[no_valid, 0] = True
    return torch.stack(sampled_contexts, dim=1), observation_valid


def average_context_deltas_by_frame(
    deltas: torch.Tensor, context_frames: torch.Tensor
) -> torch.Tensor:
    """Force the three camera observations of one frame to update one shared 3D track point."""
    averaged = torch.empty_like(deltas)
    for frame in torch.unique(context_frames).tolist():
        mask = context_frames == int(frame)
        averaged[:, mask] = deltas[:, mask].mean(dim=1, keepdim=True)
    return averaged


class ObjectSlotModule(nn.Module):
    """Group dynamic queries into K object slots with instance-id assignment.

    Each query (after fusion) is softly assigned to one of K learnable slots
    via attention; the slot carries a canonical feature plus a per-frame SE(3)
    trajectory shared by all Gaussians of that object. The per-query Gaussian
    then offsets from the slot's canonical position instead of drifting on its
    own per-point track, giving object-level coherence.
    """

    def __init__(self, hidden_dim: int, num_slots: int, num_frames: int, num_heads: int = 4):
        super().__init__()
        self.num_slots = int(num_slots)
        self.num_frames = int(num_frames)
        self.slot_tokens = nn.Parameter(torch.empty(num_slots, hidden_dim))
        nn.init.normal_(self.slot_tokens, std=0.02)
        # Per-slot canonical center offset + per-frame trajectory deltas (SE(3)-lite:
        # per-frame 3D center; rotation is deferred to a later upgrade).
        self.slot_center = nn.Parameter(torch.zeros(num_slots, 3))
        self.traj_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.slot_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.id_embed = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        fused_query: torch.Tensor,          # (N, H)
        dynamic_probability: torch.Tensor,  # (N,) in [0,1]
        time_features: torch.Tensor,        # (N, C_t)
        num_queries: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (slot_features, slot_ids, slot_weights).

        slot_features: (N, H) per-query feature augmented with its slot id embedding.
        slot_ids:      (N,) long in [0, K)  (hard assignment, for rendering ids)
        slot_weights:  (N, K) soft assignment for the consistency loss.
        """
        # Soft assignment over slots (all queries, but dynamic ones dominate the
        # assignment because only their trajectory loss is active).
        tokens = self.slot_tokens[None].expand(fused_query.shape[0], -1, -1)
        attn_out, attn_w = self.slot_attn(
            fused_query[:, None], tokens, tokens, need_weights=True,
        )
        slot_weights = attn_w[:, 0]                       # (N, K) softmax over slots
        slot_ids = slot_weights.argmax(dim=-1)            # (N,)
        # Canonical slot feature per query = weighted mean of slot tokens.
        slot_feature = (slot_weights[..., None] * tokens).sum(dim=1)  # (N, H)
        augmented = fused_query + slot_feature + self.id_embed(slot_feature)
        return augmented, slot_ids, slot_weights


class StaticPointForwardModel(nn.Module):
    """PointForward view weighting followed by learnable Gaussian slots per query."""

    def __init__(
        self,
        query_dim: int,
        rgb_feature_dim: int,
        hidden_dim: int,
        gaussians_per_query: int,
        min_scale_m: float,
        max_scale_m: float,
        max_delta_m: float,
        fusion_type: str = "mlp",
        fusion_depth: int = 2,
        attention_heads: int = 8,
        appearance_mode: str = "direct_rgb",
        color_feature_dim: int = 32,
        dedicated_rgb_head: bool = False,
        predict_lifespan: bool = False,
        iterative_refinement_layers: int = 0,
        refinement_hidden_dim: int = 64,
        refinement_fusion_type: str = "mlp",
        refinement_attention_heads: int = 8,
        refinement_sample_raw_rgb: bool = False,
        refinement_turbo_only: bool = False,
        fullres_feature_dim: int = 16,
        refinement_step_m: float = 0.5,
        flow_track_refinement: bool = False,
        learned_velocity: bool = False,
        dynamic_velocity_only: bool = False,
        learned_dynamic_separation: bool = False,
        max_velocity_m: float = 3.0,
        object_slots: int = 0,
        object_slot_frames: int = 4,
    ):
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.rgb_encoder = nn.Sequential(nn.Linear(rgb_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.depth_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.observation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.temporal_encoder = nn.Linear(2, hidden_dim)
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True
        ) if fusion_type == "xattn" else None
        self.fusion_input = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.fusion_blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(fusion_depth)])
        self.appearance_mode = appearance_mode
        self.appearance_dim = 3 if appearance_mode == "direct_rgb" else int(color_feature_dim)
        self.dedicated_rgb_head = bool(dedicated_rgb_head and appearance_mode == "direct_rgb")
        self.head_appearance_dim = 0 if self.dedicated_rgb_head else self.appearance_dim
        if self.dedicated_rgb_head:
            self.rgb_head = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
        self.predict_lifespan = bool(predict_lifespan)
        self.iterative_refinement_layers = int(iterative_refinement_layers)
        self.refinement_step_m = float(refinement_step_m)
        self.refinement_sample_raw_rgb = bool(refinement_sample_raw_rgb)
        self.refinement_turbo_only = bool(refinement_turbo_only)
        self.flow_track_refinement = bool(flow_track_refinement)
        self.learned_velocity = bool(learned_velocity)
        self.dynamic_velocity_only = bool(dynamic_velocity_only)
        self.learned_dynamic_separation = bool(learned_dynamic_separation)
        self.max_velocity_m = float(max_velocity_m)
        self.object_slots = int(object_slots)
        if self.object_slots > 0:
            self.object_slot_module = ObjectSlotModule(
                hidden_dim, self.object_slots, int(object_slot_frames)
            )
        if self.refinement_sample_raw_rgb and self.refinement_turbo_only:
            raise ValueError("Raw-RGB and Turbo-only refinement modes are mutually exclusive")
        if self.iterative_refinement_layers > 0:
            self.refinement_query_encoder = nn.Sequential(
                nn.Linear(query_dim, refinement_hidden_dim), nn.LayerNorm(refinement_hidden_dim), nn.SiLU()
            )
            self.fullres_rgb_encoder = (
                None
                if self.refinement_sample_raw_rgb or self.refinement_turbo_only
                else FullResolutionRGBEncoder(fullres_feature_dim)
            )
            sampled_fullres_dim = (
                0 if self.refinement_turbo_only
                else 3 if self.refinement_sample_raw_rgb
                else fullres_feature_dim
            )
            self.refinement_blocks = nn.ModuleList(
                PointQueryRefinementBlock(
                    rgb_feature_dim + sampled_fullres_dim,
                    refinement_hidden_dim,
                    refinement_fusion_type,
                    refinement_attention_heads,
                    self.flow_track_refinement,
                )
                for _ in range(self.iterative_refinement_layers)
            )
            self.refinement_output = nn.Sequential(
                nn.Linear(refinement_hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
            )
        self.gaussian_dim = 11 + self.head_appearance_dim + int(self.predict_lifespan) + max(self.object_slots, 0)
        if gaussians_per_query == 1:
            self.gaussian_slots = None
            head_output_dim = self.gaussian_dim
        else:
            self.gaussian_slots = nn.Parameter(torch.empty(gaussians_per_query, hidden_dim))
            nn.init.normal_(self.gaussian_slots, std=0.02)
            head_output_dim = self.gaussian_dim
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, head_output_dim),
        )
        self.velocity_head = (
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
            if self.learned_velocity else None
        )
        if self.velocity_head is not None:
            nn.init.zeros_(self.velocity_head[-1].weight)
            nn.init.zeros_(self.velocity_head[-1].bias)
        self.dynamic_residual_head = (
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            if self.learned_dynamic_separation else None
        )
        if self.dynamic_residual_head is not None:
            nn.init.zeros_(self.dynamic_residual_head[-1].weight)
            nn.init.zeros_(self.dynamic_residual_head[-1].bias)
        self.fusion_type = fusion_type
        self.gaussians_per_query = gaussians_per_query
        self.min_scale_m = float(min_scale_m)
        self.max_scale_m = float(max_scale_m)
        self.max_delta_m = float(max_delta_m)
        self.reset_head()

    def reset_head(self) -> None:
        output_head = self.head[-1]
        if self.gaussian_slots is None:
            nn.init.zeros_(output_head.weight)
        else:
            nn.init.normal_(output_head.weight, std=1.0e-4)
        nn.init.zeros_(output_head.bias)
        scale_unit = (0.12 - self.min_scale_m) / max(self.max_scale_m - self.min_scale_m, 1.0e-6)
        scale_bias = math.log(max(scale_unit, 1.0e-4) / max(1.0 - scale_unit, 1.0e-4))
        with torch.no_grad():
            output_head.bias[3:6].fill_(scale_bias)
            base_opacity = torch.sigmoid(torch.tensor(-1.5)).item()
            child_opacity = 1.0 - (1.0 - base_opacity) ** (1.0 / self.gaussians_per_query)
            output_head.bias[10].fill_(math.log(child_opacity / (1.0 - child_opacity)))

    def forward(
        self,
        query_features: torch.Tensor,
        anchors_ref: torch.Tensor,
        sampled_rgb_features: torch.Tensor,
        depth_difference: torch.Tensor,
        same_time: torch.Tensor,
        query_time: torch.Tensor,
        observation_valid: torch.Tensor,
        context_feature_maps: torch.Tensor | None = None,
        context_rgb_images: torch.Tensor | None = None,
        context_viewmats: torch.Tensor | None = None,
        context_intrinsics: torch.Tensor | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        track_anchors_ref: torch.Tensor | None = None,
        track_valid: torch.Tensor | None = None,
        context_frames: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
        flow_displacement_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        refinement_deltas = []
        refined_anchors = anchors_ref
        refined_tracks = None
        if self.iterative_refinement_layers > 0:
            required = (context_feature_maps, context_viewmats, context_intrinsics, image_width, image_height)
            if not self.refinement_turbo_only:
                required += (context_rgb_images,)
            if any(value is None for value in required):
                raise ValueError("Iterative refinement requires context feature maps, RGB images, and cameras")
            query = self.refinement_query_encoder(query_features)
            if self.refinement_turbo_only:
                fullres_feature_maps = None
            elif self.refinement_sample_raw_rgb:
                fullres_feature_maps = context_rgb_images
            else:
                fullres_feature_maps = self.fullres_rgb_encoder(context_rgb_images)
            if self.flow_track_refinement:
                if track_anchors_ref is None or context_frames is None:
                    raise ValueError("Flow-track refinement requires per-context track anchors and context frames")
                refined_tracks = track_anchors_ref
            for block in self.refinement_blocks:
                if self.flow_track_refinement:
                    observations, current_valid = sample_tracked_features(
                        refined_tracks, context_feature_maps, fullres_feature_maps,
                        context_viewmats, context_intrinsics, image_width, image_height, track_valid,
                    )
                else:
                    observations, current_valid = sample_reprojected_features(
                        refined_anchors, context_feature_maps, fullres_feature_maps,
                        context_viewmats, context_intrinsics, image_width, image_height,
                    )
                query, refinement_delta, weights = block(
                    query, observations, current_valid, same_time, query_time, self.refinement_step_m,
                    context_time=context_time,
                )
                if self.flow_track_refinement:
                    refinement_delta = average_context_deltas_by_frame(refinement_delta, context_frames)
                    refined_tracks = refined_tracks + refinement_delta
                else:
                    refined_anchors = refined_anchors + refinement_delta
                refinement_deltas.append(refinement_delta)
            fused_query = self.refinement_output(query)
        else:
            query = self.query_encoder(query_features)
            rgb = self.rgb_encoder(sampled_rgb_features)
            depth = self.depth_encoder(depth_difference)
            spatial = self.observation_encoder(torch.cat([rgb, depth], dim=-1))
            temporal = torch.cat(
                [same_time, query_time[:, None].expand(-1, spatial.shape[1], -1)], dim=-1
            )
            if self.fusion_type == "xattn":
                observations = spatial + self.temporal_encoder(temporal)
                fused, weights = self.cross_attention(
                    query[:, None], observations, observations,
                    key_padding_mask=~observation_valid, need_weights=True,
                )
                fused = fused[:, 0]
                weights = weights[:, 0]
            else:
                weight_input = torch.cat([query[:, None] * spatial, temporal], dim=-1)
                logits = self.weight_mlp(weight_input).squeeze(-1)
                logits = logits.masked_fill(~observation_valid, torch.finfo(logits.dtype).min)
                weights = torch.softmax(logits, dim=1)
                fused = (weights[..., None] * spatial).sum(dim=1)
            fused_query = self.fusion_blocks(self.fusion_input(torch.cat([query, fused, query_time], dim=-1)))

        if self.gaussian_slots is None:
            slot_features = fused_query[:, None]
        else:
            slot_features = fused_query[:, None] + self.gaussian_slots[None]
        raw = self.head(slot_features)
        delta = torch.tanh(raw[..., 0:3]) * self.max_delta_m
        scales = self.min_scale_m + (self.max_scale_m - self.min_scale_m) * torch.sigmoid(raw[..., 3:6])
        quat_raw = raw[..., 6:10].clone()
        quat_raw[..., 0] += 1.0
        quats = F.normalize(quat_raw, dim=-1)
        if self.dedicated_rgb_head:
            appearance = torch.sigmoid(self.rgb_head(slot_features))
        else:
            appearance = raw[..., 11 : 11 + self.appearance_dim]
            appearance = torch.sigmoid(appearance) if self.appearance_mode == "direct_rgb" else torch.tanh(appearance)
        gaussian_times = query_time[:, None].expand(-1, self.gaussians_per_query, -1)
        # ---- object-slot aggregation (optional) ----
        slot_ids = None
        slot_weights = None
        slot_trajectories = None
        if self.object_slots > 0:
            # dynamic_probability is the last query feature column when present.
            dynamic_probability = query_features[..., -1].clamp(0.0, 1.0)
            time_features = query_time
            fused_query, slot_ids, slot_weights = self.object_slot_module(
                fused_query, dynamic_probability, time_features, query_features.shape[0]
            )
            slot_features = fused_query[:, None]
            raw = self.head(slot_features)
            delta = torch.tanh(raw[..., 0:3]) * self.max_delta_m
            scales = self.min_scale_m + (self.max_scale_m - self.min_scale_m) * torch.sigmoid(raw[..., 3:6])
            quat_raw = raw[..., 6:10].clone()
            quat_raw[..., 0] += 1.0
            quats = F.normalize(quat_raw, dim=-1)
            if self.dedicated_rgb_head:
                appearance = torch.sigmoid(self.rgb_head(slot_features))
            else:
                appearance = raw[..., 11 : 11 + self.appearance_dim]
                appearance = (
                    torch.sigmoid(appearance)
                    if self.appearance_mode == "direct_rgb"
                    else torch.tanh(appearance)
                )
        if refined_tracks is None:
            means = (refined_anchors[:, None] + delta).reshape(-1, 3)
            means_by_context = None
        else:
            means_by_context = refined_tracks[:, :, None, :] + delta[:, None, :, :]
            means = means_by_context[:, 0].reshape(-1, 3)
            refined_anchors = refined_tracks[:, 0]
        velocity = None
        base_means = means
        dynamic_probability = query_features[..., -1].clamp(0.0, 1.0)
        if self.dynamic_residual_head is not None:
            # Start from flow's soft dynamic estimate, then learn a residual
            # from held-out-frame rendering. Zero initialization preserves the
            # flow-track solution at the beginning of the new experiment.
            flow_logit = torch.logit(dynamic_probability.clamp(1.0e-4, 1.0 - 1.0e-4))
            residual_logit = self.dynamic_residual_head(slot_features).squeeze(-1)
            if residual_logit.ndim > flow_logit.ndim:
                flow_logit = flow_logit[..., None]
            dynamic_probability = torch.sigmoid(
                flow_logit + residual_logit
            )
        if self.learned_velocity:
            residual_velocity = torch.tanh(self.velocity_head(slot_features)) * self.max_velocity_m
            dynamic_gate = (
                dynamic_probability[..., None]
                if dynamic_probability.ndim == 2
                else dynamic_probability[:, None, None]
            )
            if flow_displacement_prior is not None:
                velocity = dynamic_gate * (
                    flow_displacement_prior[:, None, :] + residual_velocity
                )
            else:
                velocity = residual_velocity * dynamic_gate
            means = base_means.reshape(-1, 3)
        result = {
            "means": means,
            "scales": scales.reshape(-1, 3),
            "quats": quats.reshape(-1, 4),
            "opacities": torch.sigmoid(raw[..., 10:11]).reshape(-1),
            "appearance_features": appearance.reshape(-1, self.appearance_dim),
            "gaussian_times": gaussian_times.reshape(-1),
            "raw_scales": scales,
            "delta": delta,
            "view_weights": weights,
            "refined_anchors": refined_anchors,
            "refinement_deltas": refinement_deltas,
            "dynamic_probability_pred": dynamic_probability.reshape(-1),
        }
        if velocity is not None:
            result["base_means"] = base_means.reshape(-1, 3)
            result["velocity"] = velocity.reshape(-1, 3)
            base_time = (
                context_time[:, 0, 0]
                if context_time is not None
                else query_time.reshape(-1)
            )
            result["trajectory_base_time"] = base_time.detach()
            result["trajectory_time_origin"] = query_time.new_tensor(float(getattr(self, "time_origin", 0.0)))
            result["trajectory_time_denominator"] = query_time.new_tensor(float(getattr(self, "time_denominator", 1.0)))
        if means_by_context is not None:
            result["means_by_context"] = means_by_context
            result["refined_tracks"] = refined_tracks
            result["context_frames"] = context_frames
        if self.appearance_mode == "direct_rgb":
            result["colors"] = result["appearance_features"]
        if self.predict_lifespan:
            lifespan_raw = raw[..., 11 + self.head_appearance_dim]
            result["lifespans"] = F.softplus(lifespan_raw).reshape(-1)
        if self.object_slots > 0:
            id_raw = raw[..., 11 + self.head_appearance_dim + int(self.predict_lifespan) :]
            result["instance_id_logits"] = id_raw.reshape(-1, self.object_slots)
            if slot_ids is not None:
                result["instance_ids"] = slot_ids
            if slot_weights is not None:
                result["instance_weights"] = slot_weights
        return result


class PerCameraAffine(nn.Module):
    """Per-camera affine color correction applied after rendering per-Gaussian RGB.

    The appearance head of a direct_rgb model predicts camera-agnostic Gaussian
    colors. Real multi-camera rigs expose each sensor with different gains; this
    module learns one 3x3 matrix (plus bias) per camera so the rendered image can
    be corrected per sensor while the exported per-Gaussian RGB stays clean.
    """

    def __init__(self, num_cameras: int) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.affine = nn.Parameter(torch.zeros(self.num_cameras, 3, 4))
        with torch.no_grad():
            self.affine[:, :, :3] = torch.eye(3)

    def forward(self, rgb: torch.Tensor, camera_index: int) -> torch.Tensor:
        matrix = self.affine[camera_index]  # (3, 4)
        color, offset = matrix[:, :3], matrix[:, 3]
        return (torch.einsum("dc,chw->dhw", color, rgb) + offset[:, None, None]).clamp(0.0, 1.0)


class FeatureRenderUNet(nn.Module):
    """Lightweight image-space decoder for rendered Gaussian appearance features."""

    def __init__(
        self,
        feature_dim: int,
        base_channels: int,
        num_cameras: int = 0,
        camera_embedding_dim: int = 16,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"FeatureRenderUNet depth must be >= 1, got {depth}")
        self.input_block = nn.Sequential(
            nn.Conv2d(feature_dim, base_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.SiLU(),
        )
        self.down_block = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.SiLU(),
        )
        self.output_block = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )
        self.refine_blocks = nn.Sequential(
            *[
                layer
                for _ in range(depth - 1)
                for layer in (
                    nn.Conv2d(base_channels * 3, base_channels * 3, 3, padding=1),
                    nn.SiLU(),
                )
            ]
        )
        self.camera_embedding = (
            nn.Embedding(num_cameras, camera_embedding_dim) if num_cameras > 0 else None
        )
        self.camera_input_film = (
            nn.Linear(camera_embedding_dim, base_channels * 2)
            if self.camera_embedding is not None else None
        )
        self.camera_low_film = (
            nn.Linear(camera_embedding_dim, base_channels * 4)
            if self.camera_embedding is not None else None
        )
        if self.camera_embedding is not None:
            nn.init.zeros_(self.camera_input_film.weight)
            nn.init.zeros_(self.camera_input_film.bias)
            nn.init.zeros_(self.camera_low_film.weight)
            nn.init.zeros_(self.camera_low_film.bias)

    @staticmethod
    def apply_film(features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        scale, bias = parameters.chunk(2, dim=-1)
        return features * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]

    def forward(
        self, features: torch.Tensor, camera_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        skip = self.input_block(features)
        camera_embedding = None
        if self.camera_embedding is not None:
            if camera_indices is None:
                raise ValueError("camera_indices are required for target-camera conditioning")
            camera_embedding = self.camera_embedding(camera_indices.to(features.device).long())
            skip = self.apply_film(skip, self.camera_input_film(camera_embedding))
        low = self.down_block(skip)
        if camera_embedding is not None:
            low = self.apply_film(low, self.camera_low_film(camera_embedding))
        up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.refine_blocks(torch.cat([skip, up], dim=1))
        return torch.sigmoid(self.output_block(decoded))


@dataclass
class Stage2Batch:
    query_features: torch.Tensor
    anchors_ref: torch.Tensor
    sampled_rgb_features: torch.Tensor
    depth_difference: torch.Tensor
    same_time: torch.Tensor
    query_time: torch.Tensor
    observation_valid: torch.Tensor
    obs_uv: torch.Tensor
    context_time: torch.Tensor
    context_view: torch.Tensor
    context_feature_maps: torch.Tensor | None = None
    track_anchors_ref: torch.Tensor | None = None
    track_valid: torch.Tensor | None = None


def build_stage2_batch(stage1: dict, context_features: torch.Tensor, args: argparse.Namespace) -> Stage2Batch:
    obs = stage1["observations"]
    obs_uv = torch.stack([obs["u"].float(), obs["v"].float()], dim=-1)
    sampled = sample_context_features(context_features, obs_uv, args.width, args.height)
    learned_local_sampling = getattr(args, "learned_local_sampling", False)
    iterative_refinement = getattr(args, "iterative_refinement_layers", 0) > 0
    batch = build_stage2_batch_from_sampled(
        stage1,
        sampled,
        getattr(args, "use_projection_valid_mask", False),
        time_origin=float(getattr(args, "time_origin", 0.0)),
        time_denominator=float(getattr(args, "time_denominator", 16.0)),
    )
    batch.context_feature_maps = context_features if learned_local_sampling or iterative_refinement else None
    return batch


def load_refinement_context(
    stage1_dir: Path,
    clip: dict,
    observations: dict[str, torch.Tensor],
    device: torch.device,
    sample_raw_rgb: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_raw_rgb:
        rgb_video = torch.from_numpy(as_uint8_rgb(clip["rgb_target"].numpy())).permute(1, 4, 0, 2, 3).contiguous()
    else:
        decoded = torch.load(stage1_dir / "stage1_decoded_rgbd.pt", map_location="cpu")
        rgb_video = decoded["rgb_u8"]
    context_frames = observations["context_frame"].long().tolist()
    context_views = observations["context_view"].long().tolist()
    images, viewmats, intrinsics = [], [], []
    for frame, view in zip(context_frames, context_views, strict=True):
        images.append(rgb_video[view, :, frame].float() / 255.0)
        frame_to_ref = clip["frame_to_ref_lidar"][frame].float()
        viewmats.append(clip["lidar2camera"][frame, view].float() @ torch.linalg.inv(frame_to_ref))
        intrinsics.append(clip["camera_intrinsics"][frame, view].float())
    return (
        torch.stack(images).to(device),
        torch.stack(viewmats).to(device),
        torch.stack(intrinsics).to(device),
    )


def build_stage2_batch_from_sampled(
    stage1: dict,
    sampled: torch.Tensor,
    use_projection_valid_mask: bool = False,
    time_origin: float = 0.0,
    time_denominator: float = 16.0,
) -> Stage2Batch:
    q = stage1["queries"]
    col = stage1["col"]
    obs = stage1["observations"]
    obs_uv = torch.stack([obs["u"].float(), obs["v"].float()], dim=-1)
    anchors = q[:, [col["x_ref"], col["y_ref"], col["z_ref"]]]
    xyz_norm = (anchors - anchors.mean(dim=0, keepdim=True)) / anchors.std(dim=0, keepdim=True).clamp_min(1.0)
    source_rgb = q[:, [col["rgb_r"], col["rgb_g"], col["rgb_b"]]]
    ray_dir = q[:, [col["ray_dir_x_ref"], col["ray_dir_y_ref"], col["ray_dir_z_ref"]]]
    ray_moment = q[:, [col["ray_moment_x_ref"], col["ray_moment_y_ref"], col["ray_moment_z_ref"]]]
    query_parts = [xyz_norm, source_rgb, ray_dir, ray_moment]
    if "dynamic_probability" in col:
        query_parts.append(q[:, col["dynamic_probability"] : col["dynamic_probability"] + 1])
    query_features = torch.cat(query_parts, dim=1)
    depth_difference = torch.log1p(obs["depth_abs_diff_m"].float().clamp_min(0.0))[..., None] / math.log(101.0)
    source_frame = q[:, col["source_frame"]].long()
    context_frame = obs["context_frame"].long()
    same_time = (source_frame[:, None] == context_frame[None]).float()[..., None]
    time_denominator = max(float(time_denominator), 1.0)
    query_time = ((source_frame.float() - float(time_origin)) / time_denominator)[:, None]
    observation_valid = obs["in_bounds"].bool().clone() if use_projection_valid_mask else torch.ones_like(
        obs["in_bounds"], dtype=torch.bool
    )
    # MultiheadAttention and masked softmax require at least one key. The rare
    # queries outside every context receive one zero-padded fallback observation.
    no_valid = ~observation_valid.any(dim=1)
    observation_valid[no_valid, 0] = True
    context_time = (
        (context_frame.float() - float(time_origin)) / time_denominator
    )[None, :, None].expand(q.shape[0], -1, -1)
    context_view = (obs["context_view"].float() / max(len(VIEW_NAMES) - 1, 1))[None, :, None].expand(q.shape[0], -1, -1)
    batch = Stage2Batch(
        query_features, anchors, sampled, depth_difference, same_time, query_time, observation_valid,
        obs_uv, context_time, context_view,
    )
    if "track_points_ref" in obs:
        batch.track_anchors_ref = obs["track_points_ref"].float()
        batch.track_valid = obs.get("track_valid", torch.ones_like(obs["in_bounds"], dtype=torch.bool)).bool()
    return batch


def camera_for_target(clip, frame_index, view_index, render_width, render_height, args, device):
    intrinsics = clip["camera_intrinsics"][frame_index, view_index].float().to(device).clone()
    intrinsics[0, [0, 2]] *= render_width / float(args.width)
    intrinsics[1, [1, 2]] *= render_height / float(args.height)
    ref_to_current = torch.linalg.inv(clip["frame_to_ref_lidar"][frame_index].float().to(device))
    lidar2camera = clip["lidar2camera"][frame_index, view_index].float().to(device).clone()
    # Camera x points to image-right. Moving the camera center by +x therefore
    # subtracts the same offset from the world-to-camera translation.
    lidar2camera[0, 3] -= float(getattr(args, "camera_lateral_offset_m", 0.0))
    return (lidar2camera @ ref_to_current)[None], intrinsics[None]


def opacities_at_frame(
    gaussian: dict[str, torch.Tensor], frame_index: int, args: argparse.Namespace | None = None
) -> torch.Tensor:
    if "lifespans" not in gaussian:
        return gaussian["opacities"]
    time_origin = float(getattr(args, "time_origin", 0.0)) if args is not None else 0.0
    time_denominator = max(
        float(getattr(args, "time_denominator", 16.0)) if args is not None else 16.0,
        1.0,
    )
    target_time = (float(frame_index) - time_origin) / time_denominator
    normalized_offset = (target_time - gaussian["gaussian_times"]) / (gaussian["lifespans"] + 1.0)
    return gaussian["opacities"] * torch.exp(-0.5 * normalized_offset.square())


def means_at_frame(gaussian: dict[str, torch.Tensor], frame_index: int) -> torch.Tensor:
    if "velocity" in gaussian and "base_means" in gaussian:
        origin = float(gaussian["trajectory_time_origin"].item())
        denominator = max(float(gaussian["trajectory_time_denominator"].item()), 1.0)
        target_time = (float(frame_index) - origin) / denominator
        delta_time = target_time - gaussian["trajectory_base_time"]
        return gaussian["base_means"] + gaussian["velocity"] * delta_time[:, None]
    if "means_by_context" not in gaussian:
        return gaussian["means"]
    context_frames = gaussian["context_frames"]
    matches = torch.nonzero(context_frames == int(frame_index), as_tuple=False).flatten()
    if matches.numel() == 0:
        unique_frames = torch.unique(context_frames, sorted=True)
        lower = unique_frames[unique_frames < int(frame_index)]
        upper = unique_frames[unique_frames > int(frame_index)]
        if lower.numel() == 0:
            nearest = int(upper[0])
            index = torch.nonzero(context_frames == nearest, as_tuple=False).flatten()[0]
            return gaussian["means_by_context"][:, int(index)].reshape(-1, 3)
        if upper.numel() == 0:
            nearest = int(lower[-1])
            index = torch.nonzero(context_frames == nearest, as_tuple=False).flatten()[0]
            return gaussian["means_by_context"][:, int(index)].reshape(-1, 3)
        lower_frame, upper_frame = int(lower[-1]), int(upper[0])
        lower_index = torch.nonzero(context_frames == lower_frame, as_tuple=False).flatten()[0]
        upper_index = torch.nonzero(context_frames == upper_frame, as_tuple=False).flatten()[0]
        weight = (float(frame_index) - lower_frame) / float(upper_frame - lower_frame)
        lower_means = gaussian["means_by_context"][:, int(lower_index)]
        upper_means = gaussian["means_by_context"][:, int(upper_index)]
        return torch.lerp(lower_means, upper_means, weight).reshape(-1, 3)
    return gaussian["means_by_context"][:, int(matches[0])].reshape(-1, 3)


def render_gsplat(gaussian, clip, frame_index, view_index, render_height, render_width, args, device):
    from gsplat import rasterization

    viewmats, ks = camera_for_target(clip, frame_index, view_index, render_width, render_height, args, device)
    # gsplat 1.5.x uses packed rasterization by default, so backgrounds are
    # channel-only rather than batched as [1, C].
    backgrounds = torch.full((3,), float(args.background), dtype=torch.float32, device=device)
    colors, _, _ = rasterization(
        means=means_at_frame(gaussian, frame_index), quats=gaussian["quats"], scales=gaussian["scales"],
        opacities=opacities_at_frame(gaussian, frame_index, args), colors=gaussian["colors"], viewmats=viewmats, Ks=ks,
        width=render_width, height=render_height, near_plane=0.1, far_plane=200.0,
        backgrounds=backgrounds, render_mode="RGB",
    )
    return colors[0].permute(2, 0, 1).clamp(0.0, 1.0)


def render_appearance_features(gaussian, clip, frame_index, view_index, render_height, render_width, args, device):
    from gsplat import rasterization

    viewmats, ks = camera_for_target(clip, frame_index, view_index, render_width, render_height, args, device)
    feature_dim = gaussian["appearance_features"].shape[-1]
    backgrounds = torch.zeros((feature_dim,), dtype=torch.float32, device=device)
    features, _, _ = rasterization(
        means=means_at_frame(gaussian, frame_index), quats=gaussian["quats"], scales=gaussian["scales"],
        opacities=opacities_at_frame(gaussian, frame_index, args), colors=gaussian["appearance_features"],
        viewmats=viewmats, Ks=ks, width=render_width, height=render_height,
        near_plane=0.1, far_plane=200.0, backgrounds=backgrounds, render_mode="RGB",
    )
    return features[0].permute(2, 0, 1)


def render_output(gaussian, appearance_decoder, clip, frame, view, height, width, args, device, camera_affine=None):
    if appearance_decoder is None:
        rendered = render_gsplat(gaussian, clip, frame, view, height, width, args, device)
    else:
        feature_map = render_appearance_features(gaussian, clip, frame, view, height, width, args, device)
        camera_indices = (
            torch.tensor([view], device=device)
            if getattr(args, "target_camera_conditioning", False) else None
        )
        rendered = appearance_decoder(feature_map[None], camera_indices)[0]
    if camera_affine is not None:
        rendered = camera_affine(rendered, view)
    return rendered


def resize_target(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate((rgb.float() / 255.0)[None], size=(height, width), mode="area")[0]


def rgb_loss(rendered: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "l1":
        return (rendered - target).abs().mean()
    return torch.sqrt((rendered - target).pow(2).sum(dim=0) + 1.0e-6).mean()


def psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(rendered, target).detach().clamp_min(1.0e-10)
    return float((-10.0 * torch.log10(mse)).item())


def save_panel(path: Path, rendered: torch.Tensor, target: torch.Tensor) -> None:
    def to_u8(image):
        array = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return np.floor(array * 255.0 + 0.5).astype(np.uint8)

    rendered_u8, target_u8 = to_u8(rendered), to_u8(target)
    error = np.abs(rendered_u8.astype(np.float32) - target_u8.astype(np.float32)).mean(axis=2)
    heat = cv2.applyColorMap(np.clip(error * 4.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panel = np.concatenate([cv2.cvtColor(target_u8, cv2.COLOR_RGB2BGR), cv2.cvtColor(rendered_u8, cv2.COLOR_RGB2BGR), heat], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def run_git(args: list[str]) -> str:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def write_run_files(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    if args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty checkpoint directory: {args.checkpoint_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "command.sh").write_text(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n", encoding="utf-8")
    git_text = f"HEAD: {run_git(['git', 'rev-parse', 'HEAD'])}\nSTATUS:\n{run_git(['git', 'status', '--short'])}\n"
    (args.output_dir / "git.txt").write_text(git_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.fast_dev_run:
        args.iterations = 1
        args.target_frames = args.target_frames[:1]
        args.target_views = args.target_views[:1]
        args.save_every = 1
        args.log_every = 1
    write_run_files(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    lpips_model = None
    if args.lpips_weight > 0:
        if args.lpips_module_root is not None and str(args.lpips_module_root) not in sys.path:
            sys.path.append(str(args.lpips_module_root))
        from lpips import LPIPS
        lpips_model = LPIPS(net="alex").to(device).eval().requires_grad_(False)
    row = read_manifest_row(args.manifest, args.clip_index)
    clip = torch.load(row["clip_pt"], map_location="cpu")
    stage1 = load_stage1(args.stage1_dir, device)
    obs = stage1["observations"]
    context_features, turbo_info = decode_context_features(
        clip, obs["context_frame"].long(), obs["context_view"].long(), args, device
    )
    batch = build_stage2_batch(stage1, context_features, args)
    context_rgb_images = context_viewmats = context_intrinsics = None
    if args.iterative_refinement_layers > 0:
        context_rgb_images, context_viewmats, context_intrinsics = load_refinement_context(
            args.stage1_dir, clip, obs, device, args.refinement_sample_raw_rgb
        )
    if not args.learned_local_sampling and args.iterative_refinement_layers <= 0:
        del context_features

    render_height = int(round(args.height * args.render_scale))
    render_width = int(round(args.width * args.render_scale))
    video_length = int(clip.get("video_length", row.get("video_length", 17)))
    target_frames = [frame for frame in args.target_frames if frame < video_length]
    target_views = [view_to_index(view) for view in args.target_views]
    rgb_target = as_uint8_rgb(clip["rgb_target"].numpy())
    targets = [
        (frame, view, resize_target(torch.from_numpy(rgb_target[frame, view]).permute(2, 0, 1).to(device), render_height, render_width))
        for frame in target_frames for view in target_views
    ]
    if not targets:
        raise RuntimeError("No render targets were selected")
    target_rng = np.random.default_rng(args.seed)

    model = StaticPointForwardModel(
        query_dim=batch.query_features.shape[-1], rgb_feature_dim=batch.sampled_rgb_features.shape[-1],
        hidden_dim=args.hidden_dim, gaussians_per_query=args.gaussians_per_query,
        min_scale_m=args.min_scale_m, max_scale_m=args.max_scale_m, max_delta_m=args.max_delta_m,
        fusion_type=args.fusion_type, fusion_depth=args.fusion_depth, attention_heads=args.attention_heads,
        appearance_mode=args.appearance_mode, color_feature_dim=args.color_feature_dim,
        predict_lifespan=args.predict_lifespan,
        iterative_refinement_layers=args.iterative_refinement_layers,
        refinement_hidden_dim=args.refinement_hidden_dim,
        refinement_fusion_type=args.refinement_fusion_type,
        refinement_attention_heads=args.refinement_attention_heads,
        refinement_sample_raw_rgb=args.refinement_sample_raw_rgb,
        refinement_turbo_only=args.refinement_turbo_only,
        fullres_feature_dim=args.fullres_feature_dim,
        refinement_step_m=args.refinement_step_m,
        flow_track_refinement=args.flow_track_refinement,
    ).to(device)
    appearance_decoder = (
        FeatureRenderUNet(
            args.color_feature_dim,
            args.unet_base_channels,
            num_cameras=len(VIEW_NAMES) if args.target_camera_conditioning else 0,
            camera_embedding_dim=args.camera_embedding_dim,
        ).to(device)
        if args.appearance_mode == "feature_unet" else None
    )
    local_feature_sampler = (
        LearnedLocalFeatureSampler(
            batch.query_features.shape[-1], batch.sampled_rgb_features.shape[-1],
            min(args.hidden_dim, 64), args.local_sampling_radius_px,
        ).to(device)
        if args.learned_local_sampling else None
    )
    trainable_parameters = list(model.parameters())
    if appearance_decoder is not None:
        trainable_parameters += list(appearance_decoder.parameters())
    if local_feature_sampler is not None:
        trainable_parameters += list(local_feature_sampler.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    optimizer_resumed = False
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if appearance_decoder is not None:
            appearance_decoder.load_state_dict(checkpoint["appearance_decoder"])
        if local_feature_sampler is not None:
            local_feature_sampler.load_state_dict(checkpoint["local_feature_sampler"])
        start_step = int(checkpoint["step"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_resumed = True
        if "target_rng_state" in checkpoint:
            target_rng.bit_generator.state = checkpoint["target_rng_state"]
        elif 0 < args.targets_per_step < len(targets):
            for _ in range(start_step):
                target_rng.choice(len(targets), size=args.targets_per_step, replace=False)
        if args.iterations <= start_step:
            raise ValueError(
                f"--iterations ({args.iterations}) must exceed resumed step ({start_step})"
            )
        print(json.dumps({
            "resume_checkpoint": str(args.resume_checkpoint),
            "resume_step": start_step,
            "optimizer_resumed": optimizer_resumed,
        }), flush=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_path = args.checkpoint_dir / "best_static_pointforward_stage2.pt"
    start = time.perf_counter()
    best_psnr = -1.0

    for step in range(start_step + 1, args.iterations + 1):
        model.train()
        if appearance_decoder is not None:
            appearance_decoder.train()
        sampled_rgb_features = batch.sampled_rgb_features
        local_offsets = None
        if local_feature_sampler is not None:
            local_feature_sampler.train()
            sampled_rgb_features, local_offsets = local_feature_sampler(
                batch.context_feature_maps, batch.sampled_rgb_features, batch.obs_uv, batch.query_features,
                batch.depth_difference, batch.same_time, batch.query_time, batch.context_time,
                batch.context_view, args.width, args.height,
            )
        gaussian = model(
            batch.query_features, batch.anchors_ref, sampled_rgb_features, batch.depth_difference,
            batch.same_time, batch.query_time, batch.observation_valid,
            context_feature_maps=batch.context_feature_maps,
            context_rgb_images=context_rgb_images,
            context_viewmats=context_viewmats,
            context_intrinsics=context_intrinsics,
            image_width=args.width,
            image_height=args.height,
            track_anchors_ref=batch.track_anchors_ref,
            track_valid=batch.track_valid,
            context_frames=obs["context_frame"].long(),
            context_time=batch.context_time,
        )
        l1_losses, lpips_losses, psnrs, rendered_cache = [], [], [], {}
        if 0 < args.targets_per_step < len(targets):
            target_indices = target_rng.choice(len(targets), size=args.targets_per_step, replace=False)
            step_targets = [targets[int(index)] for index in target_indices]
        else:
            step_targets = targets
        for frame, view, target in step_targets:
            rendered = render_output(
                gaussian, appearance_decoder, clip, frame, view,
                render_height, render_width, args, device,
            )
            l1_losses.append(rgb_loss(rendered, target, args.rgb_loss))
            if lpips_model is not None:
                lpips_losses.append(
                    lpips_model(rendered[None] * 2.0 - 1.0, target[None] * 2.0 - 1.0).mean()
                )
            psnrs.append(psnr(rendered, target))
            rendered_cache[(frame, view)] = (rendered.detach(), target)
        pixel_loss = torch.stack(l1_losses).mean()
        lpips_loss = torch.stack(lpips_losses).mean() if lpips_losses else pixel_loss.new_zeros(())
        image_loss = pixel_loss + args.lpips_weight * lpips_loss
        opacity_reg = gaussian["opacities"].mean()
        scale_reg = gaussian["raw_scales"].mean()
        local_offset_reg = (
            (local_offsets / max(args.local_sampling_radius_px, 1.0e-6)).square().mean()
            if local_offsets is not None else image_loss.new_zeros(())
        )
        loss = (
            image_loss + args.opacity_reg_weight * opacity_reg + args.scale_reg_weight * scale_reg
            + args.local_offset_reg_weight * local_offset_reg
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()

        view_weights = gaussian["view_weights"].detach().clamp_min(1.0e-8)
        record = {
            "step": step, "loss": float(loss.detach()), "image_loss": float(image_loss.detach()),
            "pixel_loss": float(pixel_loss.detach()), "lpips_loss": float(lpips_loss.detach()),
            "psnr": float(np.mean(psnrs)), "opacity_mean": float(gaussian["opacities"].detach().mean()),
            "scale_mean_m": float(gaussian["raw_scales"].detach().mean()),
            "delta_mean_m": float(gaussian["delta"].detach().norm(dim=-1).mean()),
            "view_weight_max_mean": float(view_weights.max(dim=1).values.mean()),
            "view_weight_entropy": float((-(view_weights * view_weights.log()).sum(dim=1)).mean()),
            "elapsed_sec": time.perf_counter() - start,
        }
        if "lifespans" in gaussian:
            record["lifespan_mean"] = float(gaussian["lifespans"].detach().mean())
        if gaussian["refinement_deltas"]:
            record["refinement_delta_mean_m"] = [
                float(value.detach().norm(dim=-1).mean()) for value in gaussian["refinement_deltas"]
            ]
            record["refinement_total_delta_mean_m"] = float(
                (gaussian["refined_anchors"].detach() - batch.anchors_ref).norm(dim=-1).mean()
            )
        if local_offsets is not None:
            record["local_offset_reg"] = float(local_offset_reg.detach())
            record["local_offset_mean_px"] = float(local_offsets.detach().norm(dim=-1).mean())
            record["local_offset_max_px"] = float(local_offsets.detach().norm(dim=-1).max())
        if device.type == "cuda":
            record["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            record["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024**3)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if record["psnr"] > best_psnr:
            best_psnr = record["psnr"]
            payload = {
                "model": model.state_dict(), "config": vars(args), "step": step, "psnr": best_psnr,
                "optimizer": optimizer.state_dict(), "target_rng_state": target_rng.bit_generator.state,
            }
            if appearance_decoder is not None:
                payload["appearance_decoder"] = appearance_decoder.state_dict()
            if local_feature_sampler is not None:
                payload["local_feature_sampler"] = local_feature_sampler.state_dict()
            torch.save(payload, best_path)
        if step == start_step + 1 or step % args.save_every == 0 or step == args.iterations:
            for (frame, view), (rendered, target) in rendered_cache.items():
                save_panel(args.output_dir / "visuals" / f"step{step:06d}_f{frame:02d}_{VIEW_NAMES[view]}.jpg", rendered, target)
        if step == start_step + 1 or step % args.log_every == 0 or step == args.iterations:
            print(json.dumps(record), flush=True)

    summary = {
        "output_dir": str(args.output_dir), "checkpoint_dir": str(args.checkpoint_dir),
        "stage1_dir": str(args.stage1_dir), "num_queries": int(batch.anchors_ref.shape[0]),
        "num_contexts": int(batch.sampled_rgb_features.shape[1]), "num_targets": len(targets),
        "turbo_feature_key": args.turbo_feature_key, "turbo": turbo_info,
        "appearance_mode": args.appearance_mode,
        "color_feature_dim": args.color_feature_dim if appearance_decoder is not None else 3,
        "valid_observations_per_query_mean": float(batch.observation_valid.sum(dim=1).float().mean()),
        "all_invalid_query_count": int((~stage1["observations"]["in_bounds"].any(dim=1)).sum()),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "resume_step": start_step, "optimizer_resumed": optimizer_resumed,
        "best_psnr": best_psnr, "best_checkpoint": str(best_path), "metrics_jsonl": str(metrics_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
