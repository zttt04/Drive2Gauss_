#!/usr/bin/env python3
"""Inspect PointForward-style static queries and cross-view observations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
TURBO_ROOT_DEFAULT = Path(
    os.environ.get("TURBO_VAED_ROOT", REPO_ROOT / "third_party" / "Turbo-VAED")
)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--query-frames", type=int, nargs="+", default=[0, 6, 12, 16])
    parser.add_argument("--query-views", nargs="+", default=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"])
    parser.add_argument("--context-frames", type=int, nargs="+", default=None)
    parser.add_argument("--context-views", nargs="+", default=None)
    parser.add_argument("--num-queries-per-frame-view", type=int, default=2048)
    parser.add_argument("--max-total-queries", type=int, default=12288)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument(
        "--dense-pixel-queries",
        action="store_true",
        help="Use every valid source pixel as a query without spatial sampling.",
    )
    parser.add_argument("--static-mask-source", choices=["static_mask_target", "valid_depth"], default="static_mask_target")
    parser.add_argument("--depth-max-m", type=float, default=100.0)
    parser.add_argument("--depth-min-valid-m", type=float, default=0.1)
    parser.add_argument("--depth-max-valid-m", type=float, default=99.5)
    parser.add_argument("--exclude-sky-by-depth", action="store_true")
    parser.add_argument("--sky-depth-threshold-m", type=float, default=90.0)
    parser.add_argument("--depth-abs-threshold-m", type=float, default=3.0)
    parser.add_argument("--depth-rel-threshold", type=float, default=0.20)
    parser.add_argument(
        "--flow-rgb-track-init",
        action="store_true",
        help="Initialize a per-frame 3D query track by approximately inverting packaged flow RGB.",
    )
    parser.add_argument("--flow-scale-px", type=float, default=64.0)
    parser.add_argument("--flow-white-threshold", type=float, default=0.02)
    parser.add_argument("--flow-white-temperature", type=float, default=0.01)
    parser.add_argument("--flow-inverse-iterations", type=int, default=3)
    parser.add_argument(
        "--flow-track-max-step-m",
        type=float,
        default=3.0,
        help="Maximum initialized 3D displacement per frame; prevents weak-depth outliers from leaving the image.",
    )
    parser.add_argument("--projection-dot-limit", type=int, default=6000)
    parser.add_argument(
        "--save-decoded-rgbd",
        action="store_true",
        help="Save Turbo decoded RGB/depth tensors for Stage 2 training without importing Turbo.",
    )
    parser.add_argument(
        "--save-stage2-features",
        action="store_true",
        help="Save sampled frozen Turbo features for multi-scene Stage 2 training.",
    )
    parser.add_argument("--turbo-feature-key", default="up_block_2")
    parser.add_argument("--turbo-repo-root", type=Path, default=TURBO_ROOT_DEFAULT)
    parser.add_argument("--turbo-config", type=Path, default=None)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-scale", type=float, default=1.0 / COGVIDEOX_SCALING_FACTOR)
    parser.add_argument("--height", type=int, default=424)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def read_manifest_row(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} does not contain row {index}")


def view_to_index(view: str) -> int:
    return int(view) if view.isdigit() else VIEW_NAMES.index(view)


def build_turbo_decoder(args: argparse.Namespace, device: torch.device):
    from drive2gauss.models.turbo_decoder import build_turbo_decoder as build_decoder

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


def decoded_to_rgb(decoded: torch.Tensor) -> torch.Tensor:
    return ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).byte()


def decoded_to_metric_depth(decoded: torch.Tensor, depth_max_m: float) -> torch.Tensor:
    depth_norm = decoded.float().mean(dim=1).clamp(-1.0, 1.0)
    return ((depth_norm + 1.0) * 0.5) * float(depth_max_m)


def transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=1)
    return (matrix.unsqueeze(0) @ homogeneous[:, :, None])[:, :3, 0]


def transform_vectors(vectors: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return (matrix[:3, :3].unsqueeze(0) @ vectors[:, :, None])[:, :, 0]


def unproject_pixels_to_ref(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera2lidar: torch.Tensor,
    frame_to_ref_lidar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fx = intrinsics[0, 0].clamp_min(1.0e-6)
    fy = intrinsics[1, 1].clamp_min(1.0e-6)
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    camera_points = torch.stack([x_cam, y_cam, depth], dim=1)
    current_lidar = transform_points(camera_points, camera2lidar)
    points_ref = transform_points(current_lidar, frame_to_ref_lidar)

    ray_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=1)
    ray_lidar = transform_vectors(ray_cam, camera2lidar)
    ray_ref = F.normalize(transform_vectors(ray_lidar, frame_to_ref_lidar), dim=1, eps=1.0e-6)
    camera_origin_ref = transform_points(torch.zeros((1, 3), dtype=u.dtype, device=u.device), frame_to_ref_lidar @ camera2lidar)
    moment_ref = torch.cross(camera_origin_ref.expand_as(ray_ref), ray_ref, dim=1)
    return points_ref, ray_ref, moment_ref


def project_ref_points_to_camera(
    points_ref: torch.Tensor,
    intrinsics: torch.Tensor,
    lidar2camera: torch.Tensor,
    frame_to_ref_lidar: torch.Tensor,
    width: int,
    height: int,
    min_depth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_to_current = torch.linalg.inv(frame_to_ref_lidar)
    current_lidar = transform_points(points_ref, ref_to_current)
    camera_points = transform_points(current_lidar, lidar2camera)
    z = camera_points[:, 2]
    u = camera_points[:, 0] / z.clamp_min(1.0e-6) * intrinsics[0, 0] + intrinsics[0, 2]
    v = camera_points[:, 1] / z.clamp_min(1.0e-6) * intrinsics[1, 1] + intrinsics[1, 2]
    in_bounds = (z > min_depth) & (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
    return u, v, z, in_bounds


def sample_image_at_uv(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor, width: int, height: int) -> torch.Tensor:
    x_norm = (u / max(width - 1, 1)) * 2.0 - 1.0
    y_norm = (v / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image.unsqueeze(0).float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def make_flow_colorwheel() -> np.ndarray:
    """Return the exact RAFT/Middlebury color wheel used by the flow RGB cache."""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    colorwheel = np.zeros((ry + yg + gc + cb + bm + mr, 3), dtype=np.float32)
    col = 0
    colorwheel[col : col + ry, 0] = 255
    colorwheel[col : col + ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg
    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb
    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255
    return colorwheel


def build_flow_rgb_inverse_lut(device: torch.device, directions: int = 720) -> tuple[torch.Tensor, torch.Tensor]:
    """Build unit-flow colors and directions under the original fixed-scale encoder."""
    theta = np.linspace(-np.pi, np.pi, directions, endpoint=False, dtype=np.float32)
    unit_uv = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    u, v = unit_uv[:, 0], unit_uv[:, 1]
    angle = np.arctan2(-v, -u) / np.pi
    colorwheel = make_flow_colorwheel()
    fk = (angle + 1.0) * 0.5 * (colorwheel.shape[0] - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % colorwheel.shape[0]
    fraction = (fk - k0)[:, None]
    colors = ((1.0 - fraction) * colorwheel[k0] + fraction * colorwheel[k1]) / 255.0
    return (
        torch.from_numpy(colors.astype(np.float32)).to(device=device),
        torch.from_numpy(unit_uv.astype(np.float32)).to(device=device),
    )


def invert_flow_rgb_samples(
    rgb: torch.Tensor,
    flow_scale_px: float,
    color_lut: torch.Tensor,
    direction_lut: torch.Tensor,
    white_threshold: float,
    white_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximately invert quantized fixed-scale flow RGB into uv and soft dynamic probability."""
    color = rgb.float().clamp(0.0, 255.0) / 255.0
    radius = (1.0 - color.amin(dim=-1)).clamp(0.0, 1.0)
    saturated = (color.amax(dim=-1) < 0.80) & (color.amin(dim=-1) < 0.05)
    safe_radius = radius.clamp_min(1.0 / 255.0)
    inside_base = 1.0 - (1.0 - color) / safe_radius[:, None]
    saturated_base = color / 0.75
    base_color = torch.where(saturated[:, None], saturated_base, inside_base).clamp(0.0, 1.0)

    nearest_parts = []
    lut_norm = color_lut.square().sum(dim=-1)[None]
    for start in range(0, base_color.shape[0], 65536):
        part = base_color[start : start + 65536]
        distance = part.square().sum(dim=-1, keepdim=True) + lut_norm - 2.0 * part @ color_lut.transpose(0, 1)
        nearest_parts.append(distance.argmin(dim=-1))
    nearest = torch.cat(nearest_parts) if nearest_parts else torch.empty((0,), dtype=torch.long, device=rgb.device)
    flow_uv = direction_lut[nearest] * (radius * float(flow_scale_px))[:, None]
    probability = torch.sigmoid(
        (radius - float(white_threshold)) / max(float(white_temperature), 1.0e-6)
    )
    probability = torch.where(radius <= (1.0 / 255.0), torch.zeros_like(probability), probability)
    return flow_uv, probability, radius


def sample_inverted_flow_rgb(
    flow_rgb: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
    color_lut: torch.Tensor,
    direction_lut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = sample_image_at_uv(flow_rgb.permute(2, 0, 1), u, v, args.width, args.height)
    return invert_flow_rgb_samples(
        rgb,
        args.flow_scale_px,
        color_lut,
        direction_lut,
        args.flow_white_threshold,
        args.flow_white_temperature,
    )


def initialize_flow_rgb_tracks(
    clip: dict,
    depth_m: torch.Tensor,
    queries: dict[str, torch.Tensor],
    track_frames: list[int],
    args: argparse.Namespace,
    device: torch.device,
    view_slots: dict[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Follow approximate forward/inverse RGB flow and lift the paths into frame-0 lidar coordinates."""
    if "flow_rgb_target" not in clip:
        raise KeyError("--flow-rgb-track-init requires flow_rgb_target in the clip")
    flow_rgb = clip["flow_rgb_target"].to(device=device)
    flow_valid = clip.get("flow_rgb_valid_target")
    if flow_valid is None:
        flow_valid = torch.ones(flow_rgb.shape[:-1], dtype=torch.uint8)
    flow_valid = flow_valid.to(device=device)
    color_lut, direction_lut = build_flow_rgb_inverse_lut(device)
    unique_frames = sorted(set(int(frame) for frame in track_frames))
    frame_to_output = {frame: index for index, frame in enumerate(unique_frames)}
    num_queries = int(queries["points_ref"].shape[0])
    tracks = queries["points_ref"][:, None].expand(-1, len(unique_frames), -1).clone()
    raw_tracks = tracks.clone()
    track_valid = torch.ones((num_queries, len(unique_frames)), dtype=torch.bool, device=device)
    dynamic_probability = torch.zeros((num_queries,), dtype=torch.float32, device=device)
    decoded_radii = []
    clipped_track_steps = []
    dynamic_track_steps = []

    source_frames = queries["source_frame"].long()
    source_views = queries["source_view"].long()
    view_slots = view_slots or {}
    for source_frame in torch.unique(source_frames).tolist():
        for source_view in torch.unique(source_views[source_frames == source_frame]).tolist():
            indices = torch.nonzero(
                (source_frames == source_frame) & (source_views == source_view), as_tuple=False
            ).flatten()
            if indices.numel() == 0:
                continue
            source_u = queries["source_u"][indices]
            source_v = queries["source_v"][indices]
            uv_by_frame = {int(source_frame): torch.stack([source_u, source_v], dim=-1)}
            edge_probability: list[torch.Tensor] = []

            current_uv = uv_by_frame[int(source_frame)]
            for frame in range(int(source_frame), max(unique_frames)):
                uv_flow, probability, radius = sample_inverted_flow_rgb(
                    flow_rgb[frame, source_view], current_uv[:, 0], current_uv[:, 1],
                    args, color_lut, direction_lut,
                )
                edge_exists = sample_image_at_uv(
                    flow_valid[frame, source_view][None].float(), current_uv[:, 0], current_uv[:, 1],
                    args.width, args.height,
                )[:, 0] > 0.5
                probability = probability * edge_exists.float()
                edge_probability.append(probability)
                decoded_radii.append(radius.detach())
                current_uv = current_uv + uv_flow * edge_exists[:, None]
                uv_by_frame[frame + 1] = current_uv

            current_uv = uv_by_frame[int(source_frame)]
            for frame in range(int(source_frame) - 1, min(unique_frames) - 1, -1):
                target_uv = current_uv
                previous_uv = target_uv.clone()
                probability = torch.zeros((indices.numel(),), device=device)
                radius = torch.zeros_like(probability)
                edge_exists = torch.ones_like(probability, dtype=torch.bool)
                for _ in range(max(int(args.flow_inverse_iterations), 1)):
                    uv_flow, probability, radius = sample_inverted_flow_rgb(
                        flow_rgb[frame, source_view], previous_uv[:, 0], previous_uv[:, 1],
                        args, color_lut, direction_lut,
                    )
                    edge_exists = sample_image_at_uv(
                        flow_valid[frame, source_view][None].float(), previous_uv[:, 0], previous_uv[:, 1],
                        args.width, args.height,
                    )[:, 0] > 0.5
                    previous_uv = target_uv - uv_flow * edge_exists[:, None]
                probability = probability * edge_exists.float()
                edge_probability.append(probability)
                decoded_radii.append(radius.detach())
                current_uv = previous_uv
                uv_by_frame[frame] = current_uv

            group_probability = (
                torch.stack(edge_probability, dim=1).amax(dim=1)
                if edge_probability else torch.zeros((indices.numel(),), device=device)
            )
            dynamic_probability[indices] = group_probability
            source_anchor = queries["points_ref"][indices]
            for frame in unique_frames:
                if frame == int(source_frame):
                    continue
                uv = uv_by_frame[frame]
                depth = sample_image_at_uv(
                    depth_m[view_slots.get(int(source_view), int(source_view)), frame][None],
                    uv[:, 0],
                    uv[:, 1],
                    args.width,
                    args.height,
                )[:, 0]
                valid = (
                    (uv[:, 0] >= 0.0) & (uv[:, 0] <= args.width - 1.0)
                    & (uv[:, 1] >= 0.0) & (uv[:, 1] <= args.height - 1.0)
                    & (depth > args.depth_min_valid_m) & (depth < args.depth_max_valid_m)
                )
                candidate, _, _ = unproject_pixels_to_ref(
                    uv[:, 0], uv[:, 1], depth,
                    clip["camera_intrinsics"][frame, source_view].to(device=device).float(),
                    clip["camera2lidar"][frame, source_view].to(device=device).float(),
                    clip["frame_to_ref_lidar"][frame].to(device=device).float(),
                )
                displacement = candidate - source_anchor
                max_displacement = float(args.flow_track_max_step_m) * abs(frame - int(source_frame))
                displacement_norm = displacement.norm(dim=-1, keepdim=True)
                clipped_track_steps.append((displacement_norm[:, 0] > max_displacement).detach())
                dynamic_track_steps.append((group_probability > 0.5).detach())
                displacement = displacement * torch.clamp(
                    max_displacement / displacement_norm.clamp_min(1.0e-6), max=1.0
                )
                probability = group_probability[:, None]
                initialized = source_anchor + probability * displacement
                raw_initialized = source_anchor + displacement
                output_index = frame_to_output[frame]
                tracks[indices, output_index] = torch.where(valid[:, None], initialized, source_anchor)
                raw_tracks[indices, output_index] = torch.where(
                    valid[:, None], raw_initialized, source_anchor
                )
                track_valid[indices, output_index] = valid

    radii = torch.cat(decoded_radii) if decoded_radii else torch.zeros((1,), device=device)
    radius_quantile_values = radii.float()
    if radius_quantile_values.numel() > 1_000_000:
        stride = math.ceil(radius_quantile_values.numel() / 1_000_000)
        radius_quantile_values = radius_quantile_values[::stride]
    stats = {
        "flow_scale_px": float(args.flow_scale_px),
        "dynamic_probability_mean": float(dynamic_probability.mean()),
        "dynamic_probability_gt_half_ratio": float((dynamic_probability > 0.5).float().mean()),
        "decoded_radius_mean": float(radii.mean()),
        "decoded_radius_p99": float(torch.quantile(radius_quantile_values, 0.99)),
        "track_valid_ratio": float(track_valid.float().mean()),
        "dynamic_track_displacement_clipped_ratio": float(
            torch.cat(clipped_track_steps)[torch.cat(dynamic_track_steps)].float().mean()
        ) if dynamic_track_steps and bool(torch.cat(dynamic_track_steps).any()) else 0.0,
        "flow_track_max_step_m": float(args.flow_track_max_step_m),
    }
    return tracks, raw_tracks, track_valid, dynamic_probability, stats


def compute_edge_mask(
    rgb_u8: torch.Tensor,
    depth_m: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    kernel_size: int = 3,
    magnitude_mult: float = 8.0,
    vertical_bias: float = 1.6,
) -> torch.Tensor:
    """Full-resolution structural edge mask for thin VERTICAL structures.

    Targets upright thin structures (poles, sign posts, traffic-light masts):
    Sobel over RGB luminance + depth, robust median-normalized magnitude
    threshold, then a directional gate keeping only edges whose horizontal
    gradient dominates (i.e. near-vertical edges). Horizontal edges (lane
    lines, building cornices, horizon) are excluded, which keeps the pool
    focused on pole-like structures instead of 15% of the frame.
    """
    if rgb_u8.dim() == 4:  # (C, H, W) single frame/view slice
        rgb = rgb_u8.float()
    elif rgb_u8.dim() == 3 and rgb_u8.shape[0] in (1, 3):
        rgb = rgb_u8.float()
    else:
        raise ValueError(f"unexpected rgb shape {tuple(rgb_u8.shape)}")
    luminance = rgb.mean(dim=0, keepdim=True)  # (1, H, W)
    stack = torch.cat([luminance, depth_m.float().clamp(0.0, 200.0)[None]], dim=0)  # (2, H, W)
    pad = kernel_size // 2
    stack = F.pad(stack[None], (pad, pad, pad, pad), mode="reflect")[0]  # (2, H+2p, W+2p)
    gy = stack[:, 2:, 1:-1] - stack[:, :-2, 1:-1]  # vertical gradient (d/dy)
    gx = stack[:, 1:-1, 2:] - stack[:, 1:-1, :-2]  # horizontal gradient (d/dx)
    mag = (gx.square() + gy.square()).sqrt().mean(dim=0)  # (H, W)
    # Directional gate: a vertical edge has large |gx| (intensity changes along x)
    # and small |gy|. Require |gx| >= vertical_bias * |gy| (per channel, then
    # averaged), so near-horizontal edges are rejected.
    gx_m = gx.abs().mean(dim=0)
    gy_m = gy.abs().mean(dim=0)
    vertical = (gx_m >= vertical_bias * gy_m) & (gx_m > 1.0e-3)
    # Robust magnitude threshold: keep only strong structural gradients.
    flat = mag.reshape(-1)
    med = flat.median().clamp_min(1.0e-3)
    strong = mag > magnitude_mult * med
    edge = (strong & vertical).to(device)
    # Keep only edges with valid finite depth (structure, not sky boundary).
    valid_depth = (depth_m > 0.05) & (depth_m < 200.0)
    edge = edge & valid_depth
    return edge


def spatial_sample_mask(mask: torch.Tensor, count: int, cell_size: int, generator: torch.Generator) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=mask.device)
    height, width = mask.shape
    flat_candidate = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if flat_candidate.numel() == 0:
        return flat_candidate
    y = flat_candidate // width
    x = flat_candidate % width
    cells_x = math.ceil(width / cell_size)
    cell_id = (y // cell_size) * cells_x + (x // cell_size)
    active_cells = torch.unique(cell_id)
    n_cells = int(active_cells.numel())
    per_cell_quota = max(1, math.ceil(count / max(n_cells, 1)))
    # Vectorized cell-bucketed sampling: globally shuffle, then stable-sort by cell
    # so candidates are grouped by cell (ascending) with a random order inside each
    # cell. Equivalent distribution to the previous per-cell Python loop.
    perm = torch.randperm(flat_candidate.numel(), device=mask.device, generator=generator)
    cell_perm = cell_id[perm]
    order = torch.argsort(cell_perm, stable=True)
    sorted_candidates = flat_candidate[perm][order]
    sorted_cells = cell_perm[order]
    n = sorted_candidates.numel()
    counts = torch.bincount(sorted_cells, minlength=int(cell_id.max()) + 1)
    cell_start = torch.cumsum(counts, dim=0) - counts
    group_rank = torch.arange(n, device=mask.device) - cell_start[sorted_cells]
    keep = group_rank < per_cell_quota
    candidates = sorted_candidates[keep]
    if candidates.numel() >= count:
        return candidates[:count].long()
    # Top-up from the remainder with uniform random sampling (same as before).
    selected_mask = torch.zeros((height * width,), dtype=torch.bool, device=mask.device)
    selected_mask[candidates] = True
    rest = flat_candidate[~selected_mask[flat_candidate]]
    if rest.numel() > 0:
        need = min(count - int(candidates.numel()), int(rest.numel()))
        perm_rest = torch.randperm(rest.numel(), device=mask.device, generator=generator)[:need]
        candidates = torch.cat([candidates, rest[perm_rest]])
    return candidates[:count].long()


def add_label(image: Image.Image, text: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 24), fill=(255, 255, 255))
    draw.text((6, 5), text, fill=(0, 0, 0))
    return output


def heat_image(values: torch.Tensor, vmax: float) -> Image.Image:
    values_np = values.detach().float().cpu().numpy()
    values_np = np.nan_to_num(values_np, nan=0.0, posinf=0.0, neginf=0.0)
    values_np = np.clip(values_np / max(float(vmax), 1.0e-6), 0.0, 1.0)
    return Image.fromarray((values_np * 255.0).astype(np.uint8), mode="L").convert("RGB")


def overlay_source_queries(rgb_image: Image.Image, indices: torch.Tensor, width: int) -> Image.Image:
    output = rgb_image.copy()
    draw = ImageDraw.Draw(output)
    for index in indices.detach().cpu().tolist():
        y, x = divmod(int(index), width)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 220, 255), outline=(0, 0, 0))
    return output


def overlay_projected_queries(
    rgb_image: Image.Image,
    u: np.ndarray,
    v: np.ndarray,
    in_bounds: np.ndarray,
    strict_valid: np.ndarray,
    dot_limit: int,
) -> Image.Image:
    output = rgb_image.copy()
    draw = ImageDraw.Draw(output)
    candidate = np.flatnonzero(in_bounds)
    if candidate.size > dot_limit:
        candidate = candidate[np.linspace(0, candidate.size - 1, dot_limit).round().astype(np.int64)]
    for idx in candidate.tolist():
        x = float(u[idx])
        y = float(v[idx])
        fill = (0, 230, 90) if bool(strict_valid[idx]) else (255, 195, 0)
        draw.ellipse((x - 1.8, y - 1.8, x + 1.8, y + 1.8), fill=fill)
    return output


def make_source_panel(
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    mask: torch.Tensor,
    selected: torch.Tensor,
    title: str,
    width: int,
) -> Image.Image:
    rgb_image = Image.fromarray(rgb.permute(1, 2, 0).detach().cpu().numpy())
    mask_image = Image.fromarray((mask.detach().cpu().numpy().astype(np.uint8) * 255), mode="L").convert("RGB")
    query_image = overlay_source_queries(rgb_image, selected, width)
    panels = [
        add_label(rgb_image, title),
        add_label(heat_image(depth_m, 100.0), "Turbo metric depth"),
        add_label(mask_image, "static candidate mask"),
        add_label(query_image, "sampled static queries"),
    ]
    panel = Image.new("RGB", (panels[0].width * 2, panels[0].height * 2), "white")
    for index, item in enumerate(panels):
        panel.paste(item, ((index % 2) * item.width, (index // 2) * item.height))
    return panel


def make_histogram_image(values: np.ndarray, title: str, max_x: int | None = None) -> Image.Image:
    values = values.astype(np.float32)
    if values.size == 0:
        values = np.zeros((1,), dtype=np.float32)
    if max_x is None:
        max_x = max(int(values.max()), 1)
    bins = np.arange(max_x + 2) - 0.5
    hist, _ = np.histogram(np.clip(values, 0, max_x), bins=bins)
    width, height = 760, 300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), title, fill=(0, 0, 0))
    plot_left, plot_top, plot_right, plot_bottom = 48, 44, width - 20, height - 36
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(0, 0, 0))
    denom = max(int(hist.max()), 1)
    for idx, count in enumerate(hist):
        x0 = plot_left + idx * (plot_right - plot_left) / len(hist)
        x1 = plot_left + (idx + 1) * (plot_right - plot_left) / len(hist) - 2
        bar_h = (plot_bottom - plot_top) * float(count) / float(denom)
        draw.rectangle((x0, plot_bottom - bar_h, x1, plot_bottom), fill=(70, 130, 180))
        if idx % max(1, len(hist) // 12) == 0:
            draw.text((x0, plot_bottom + 6), str(idx), fill=(0, 0, 0))
    return image


def write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {points.shape[0]}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            stream.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def append_query_batch(
    clip: dict,
    rgb_u8: torch.Tensor,
    depth_m: torch.Tensor,
    args: argparse.Namespace,
    frame_index: int,
    view_index: int,
    generator: torch.Generator,
    device: torch.device,
    view_slot: int | None = None,
    make_panel: bool = True,
    extra_edge_queries: int = 0,
) -> tuple[dict[str, torch.Tensor], dict, Image.Image | None]:
    source_slot = view_index if view_slot is None else int(view_slot)
    depth = depth_m[source_slot, frame_index]
    depth_upper = min(args.depth_max_valid_m, args.sky_depth_threshold_m) if args.exclude_sky_by_depth else args.depth_max_valid_m
    valid_depth = (depth > args.depth_min_valid_m) & (depth < depth_upper)
    if args.static_mask_source == "static_mask_target" and "static_mask_target" in clip:
        static_mask = clip["static_mask_target"][frame_index, view_index].to(device=device).bool() & valid_depth
    else:
        static_mask = valid_depth
    if args.dense_pixel_queries:
        selected = torch.nonzero(static_mask.reshape(-1), as_tuple=False).flatten()
    else:
        selected = spatial_sample_mask(static_mask, args.num_queries_per_frame_view, args.cell_size, generator)
    # Structural (edge) query pool: extra thin-structure pixels on top of the
    # base uniform pool. Sampled from a full-resolution Sobel edge mask over
    # RGB luminance + depth, restricted to valid static depth.
    edge_selected = torch.empty((0,), dtype=torch.long, device=device)
    if extra_edge_queries > 0:
        rgb_frame = rgb_u8[source_slot, :, frame_index]  # (3, H, W)
        edge_mask = compute_edge_mask(
            rgb_frame, depth, args.width, args.height, device
        ) & static_mask
        edge_selected = spatial_sample_mask(
            edge_mask, extra_edge_queries, max(1, args.cell_size // 2), generator
        )
        selected = torch.cat([selected, edge_selected])
    y = selected // args.width
    x = selected % args.width
    u = x.float()
    v = y.float()
    selected_depth = depth.reshape(-1)[selected]
    points_ref, ray_dir_ref, ray_moment_ref = unproject_pixels_to_ref(
        u,
        v,
        selected_depth,
        clip["camera_intrinsics"][frame_index, view_index].to(device=device).float(),
        clip["camera2lidar"][frame_index, view_index].to(device=device).float(),
        clip["frame_to_ref_lidar"][frame_index].to(device=device).float(),
    )
    colors = rgb_u8[source_slot, :, frame_index].permute(1, 2, 0).reshape(-1, 3)[selected]
    batch = {
        "source_frame": torch.full((selected.numel(),), frame_index, dtype=torch.long, device=device),
        "source_view": torch.full((selected.numel(),), view_index, dtype=torch.long, device=device),
        "source_u": u,
        "source_v": v,
        "source_depth_m": selected_depth,
        "points_ref": points_ref,
        "ray_dir_ref": ray_dir_ref,
        "ray_moment_ref": ray_moment_ref,
        "colors": colors.float() / 255.0,
        "colors_u8": colors,
    }
    item = {
        "frame": int(frame_index),
        "view": VIEW_NAMES[view_index],
        "static_candidate_ratio": float(static_mask.float().mean().item()),
        "queries": int(selected.numel()),
        "depth_mean_m": float(selected_depth.mean().item()) if selected.numel() else 0.0,
        "depth_valid_upper_m": float(depth_upper),
    }
    panel = (
        make_source_panel(
            rgb_u8[source_slot, :, frame_index],
            depth,
            static_mask,
            selected,
            f"source {VIEW_NAMES[view_index]} f{frame_index}",
            args.width,
        )
        if make_panel
        else None
    )
    return batch, item, panel


def concatenate_query_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in batches[0]}


def limit_queries(queries: dict[str, torch.Tensor], max_total: int, generator: torch.Generator) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    total = int(queries["points_ref"].shape[0])
    if max_total <= 0 or total <= max_total:
        keep = torch.arange(total, device=queries["points_ref"].device)
        return queries, keep
    keep = torch.randperm(total, device=queries["points_ref"].device, generator=generator)[:max_total].sort().values
    return {key: value[keep] for key, value in queries.items()}, keep


def collect_observations(
    clip: dict,
    rgb_u8: torch.Tensor,
    depth_m: torch.Tensor,
    queries: dict[str, torch.Tensor],
    context_frames: list[int],
    context_views: list[int],
    args: argparse.Namespace,
    device: torch.device,
    track_points_ref: torch.Tensor | None = None,
    track_valid: torch.Tensor | None = None,
    track_frames: list[int] | None = None,
    view_slots: dict[int, int] | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    num_queries = int(queries["points_ref"].shape[0])
    num_contexts = len(context_frames) * len(context_views)
    observation = {
        "context_frame": torch.empty((num_contexts,), dtype=torch.long, device=device),
        "context_view": torch.empty((num_contexts,), dtype=torch.long, device=device),
        "u": torch.empty((num_queries, num_contexts), dtype=torch.float32, device=device),
        "v": torch.empty((num_queries, num_contexts), dtype=torch.float32, device=device),
        "projected_depth_m": torch.empty((num_queries, num_contexts), dtype=torch.float32, device=device),
        "target_depth_m": torch.empty((num_queries, num_contexts), dtype=torch.float32, device=device),
        "depth_abs_diff_m": torch.empty((num_queries, num_contexts), dtype=torch.float32, device=device),
        "depth_rel_diff": torch.empty((num_queries, num_contexts), dtype=torch.float32, device=device),
        "in_bounds": torch.empty((num_queries, num_contexts), dtype=torch.bool, device=device),
        "target_depth_valid": torch.empty((num_queries, num_contexts), dtype=torch.bool, device=device),
        "depth_consistent": torch.empty((num_queries, num_contexts), dtype=torch.bool, device=device),
        "rgb": torch.empty((num_queries, num_contexts, 3), dtype=torch.float32, device=device),
    }
    if track_points_ref is not None:
        observation["track_points_ref"] = torch.empty(
            (num_queries, num_contexts, 3), dtype=torch.float32, device=device
        )
        observation["track_valid"] = torch.empty(
            (num_queries, num_contexts), dtype=torch.bool, device=device
        )
        if track_frames is None or track_valid is None:
            raise ValueError("track_frames and track_valid are required with track_points_ref")
        track_frame_to_index = {int(frame): index for index, frame in enumerate(track_frames)}
    view_slots = view_slots or {}
    context_items = []
    context_index = 0
    for frame_index in context_frames:
        for view_index in context_views:
            projected_points = (
                track_points_ref[:, track_frame_to_index[frame_index]]
                if track_points_ref is not None else queries["points_ref"]
            )
            intrinsics = clip["camera_intrinsics"][frame_index, view_index].to(device=device).float()
            lidar2camera = clip["lidar2camera"][frame_index, view_index].to(device=device).float()
            frame_to_ref = clip["frame_to_ref_lidar"][frame_index].to(device=device).float()
            u, v, projected_depth, in_bounds = project_ref_points_to_camera(
                projected_points,
                intrinsics,
                lidar2camera,
                frame_to_ref,
                args.width,
                args.height,
                args.depth_min_valid_m,
            )
            view_slot = view_slots.get(int(view_index), int(view_index))
            target_depth = sample_image_at_uv(
                depth_m[view_slot, frame_index].unsqueeze(0), u, v, args.width, args.height
            )[:, 0]
            target_rgb = sample_image_at_uv(
                rgb_u8[view_slot, :, frame_index].float() / 255.0,
                u,
                v,
                args.width,
                args.height,
            )
            depth_upper = min(args.depth_max_valid_m, args.sky_depth_threshold_m) if args.exclude_sky_by_depth else args.depth_max_valid_m
            target_depth_valid = in_bounds & (target_depth > args.depth_min_valid_m) & (target_depth < depth_upper)
            abs_diff = (projected_depth - target_depth).abs()
            rel_diff = abs_diff / target_depth.clamp_min(args.depth_min_valid_m)
            depth_consistent = target_depth_valid & (abs_diff <= args.depth_abs_threshold_m) & (rel_diff <= args.depth_rel_threshold)

            observation["context_frame"][context_index] = frame_index
            observation["context_view"][context_index] = view_index
            observation["u"][:, context_index] = u
            observation["v"][:, context_index] = v
            observation["projected_depth_m"][:, context_index] = projected_depth
            observation["target_depth_m"][:, context_index] = target_depth
            observation["depth_abs_diff_m"][:, context_index] = abs_diff
            observation["depth_rel_diff"][:, context_index] = rel_diff
            observation["in_bounds"][:, context_index] = in_bounds
            observation["target_depth_valid"][:, context_index] = target_depth_valid
            observation["depth_consistent"][:, context_index] = depth_consistent
            observation["rgb"][:, context_index] = target_rgb
            if track_points_ref is not None:
                track_index = track_frame_to_index[frame_index]
                observation["track_points_ref"][:, context_index] = projected_points
                observation["track_valid"][:, context_index] = track_valid[:, track_index]

            context_items.append(
                {
                    "frame": int(frame_index),
                    "view": VIEW_NAMES[view_index],
                    "in_bounds_ratio": float(in_bounds.float().mean().item()),
                    "target_depth_valid_ratio": float(target_depth_valid.float().mean().item()),
                    "depth_consistent_ratio": float(depth_consistent.float().mean().item()),
                    "depth_abs_diff_m_mean": float(abs_diff[target_depth_valid].mean().item())
                    if bool(target_depth_valid.any())
                    else None,
                    "depth_abs_diff_m_median": float(abs_diff[target_depth_valid].median().item())
                    if bool(target_depth_valid.any())
                    else None,
                }
            )
            context_index += 1
    return observation, context_items


def tensor_to_numpy_dict(values: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def make_projection_panels(
    output_dir: Path,
    clip_index: int,
    rgb_u8: torch.Tensor,
    observation: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[str]:
    panel_paths = []
    obs_np = tensor_to_numpy_dict(observation)
    for context_index, (frame_index, view_index) in enumerate(zip(obs_np["context_frame"], obs_np["context_view"], strict=True)):
        frame = int(frame_index)
        view = int(view_index)
        rgb_image = Image.fromarray(rgb_u8[view, :, frame].permute(1, 2, 0).detach().cpu().numpy())
        overlay = overlay_projected_queries(
            rgb_image,
            obs_np["u"][:, context_index],
            obs_np["v"][:, context_index],
            obs_np["in_bounds"][:, context_index],
            obs_np["depth_consistent"][:, context_index],
            args.projection_dot_limit,
        )
        strict = obs_np["depth_consistent"][:, context_index]
        target_valid = obs_np["target_depth_valid"][:, context_index]
        title = (
            f"projection {VIEW_NAMES[view]} f{frame} | "
            f"in {obs_np['in_bounds'][:, context_index].mean():.2f} "
            f"depth-valid {target_valid.mean():.2f} strict {strict.mean():.2f}"
        )
        panel = add_label(overlay, title)
        path = output_dir / f"clip{clip_index:04d}_{VIEW_NAMES[view]}_f{frame:02d}_projection_observations.jpg"
        panel.save(path, quality=94)
        panel_paths.append(str(path))
    return panel_paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    row = read_manifest_row(args.manifest, args.clip_index)
    clip = torch.load(row["clip_pt"], map_location="cpu")
    video_length = int(clip.get("video_length", row.get("video_length", 17)))
    query_frames = [frame for frame in args.query_frames if frame < video_length]
    query_views = [view_to_index(view) for view in args.query_views]
    context_frames = [frame for frame in (args.context_frames or args.query_frames) if frame < video_length]
    context_views = [view_to_index(view) for view in (args.context_views or args.query_views)]

    turbo_decoder, turbo_info = build_turbo_decoder(args, device)
    rgb_latent = clip["rgb_latent"].to(device=device, dtype=torch.float16) * args.latent_scale
    depth_latent = clip["depth_latent"].to(device=device, dtype=torch.float16) * args.latent_scale
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        if args.save_stage2_features:
            decoded_rgb, rgb_feature_dict = turbo_decoder.decode(rgb_latent, feature_enabled=True)
            decoded_rgb = decoded_rgb[:, :, :video_length]
        else:
            decoded_rgb = turbo_decoder.decode(rgb_latent, return_dict=False)[0][:, :, :video_length]
            rgb_feature_dict = None
        decoded_depth = turbo_decoder.decode(depth_latent, return_dict=False)[0][:, :, :video_length]
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    decode_seconds = time.perf_counter() - start
    rgb_u8 = decoded_to_rgb(decoded_rgb)
    depth_m = decoded_to_metric_depth(decoded_depth, args.depth_max_m)

    query_batches = []
    source_items = []
    source_panel_paths = []
    for frame_index in query_frames:
        for view_index in query_views:
            batch, item, panel = append_query_batch(
                clip,
                rgb_u8,
                depth_m,
                args,
                frame_index,
                view_index,
                generator,
                device,
            )
            panel_path = args.output_dir / f"clip{args.clip_index:04d}_{VIEW_NAMES[view_index]}_f{frame_index:02d}_source_queries.jpg"
            panel.save(panel_path, quality=94)
            item["panel"] = str(panel_path)
            query_batches.append(batch)
            source_items.append(item)
            source_panel_paths.append(str(panel_path))
    if not query_batches:
        raise RuntimeError("No query batches were produced")

    raw_query_count = int(sum(item["queries"] for item in source_items))
    queries = concatenate_query_batches(query_batches)
    queries, keep_indices = limit_queries(queries, args.max_total_queries, generator)
    flow_track_stats = None
    track_points_ref = track_valid = None
    track_frames = sorted(set(context_frames))
    if args.flow_rgb_track_init:
        track_points_ref, _, track_valid, dynamic_probability, flow_track_stats = initialize_flow_rgb_tracks(
            clip, depth_m, queries, track_frames, args, device
        )
        queries["dynamic_probability"] = dynamic_probability
    else:
        queries["dynamic_probability"] = torch.zeros(
            (queries["points_ref"].shape[0],), dtype=torch.float32, device=device
        )
    observation, context_items = collect_observations(
        clip,
        rgb_u8,
        depth_m,
        queries,
        context_frames,
        context_views,
        args,
        device,
        track_points_ref=track_points_ref,
        track_valid=track_valid,
        track_frames=track_frames,
    )
    stage2_features_path = None
    if args.save_stage2_features:
        if args.turbo_feature_key not in rgb_feature_dict:
            raise KeyError(f"Unavailable Turbo feature {args.turbo_feature_key!r}: {sorted(rgb_feature_dict)}")
        feature_video = rgb_feature_dict[args.turbo_feature_key]
        sampled_contexts = []
        context_index = 0
        for frame_index in context_frames:
            for view_index in context_views:
                sampled_contexts.append(
                    sample_image_at_uv(
                        feature_video[view_index, :, frame_index],
                        observation["u"][:, context_index],
                        observation["v"][:, context_index],
                        args.width,
                        args.height,
                    )
                )
                context_index += 1
        sampled_features = torch.stack(sampled_contexts, dim=1).half().cpu()
        stage2_features_path = args.output_dir / f"stage2_{args.turbo_feature_key}_features.pt"
        torch.save(
            {
                "features": sampled_features,
                "feature_key": args.turbo_feature_key,
                "shape": list(sampled_features.shape),
                "context_frames": context_frames,
                "context_views": context_views,
            },
            stage2_features_path,
        )

    points_np = queries["points_ref"].detach().float().cpu().numpy()
    colors_np = (queries["colors"].detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    source_rgb_np = queries["colors"].detach().cpu().numpy().astype(np.float32)
    query_np = np.concatenate(
        [
            np.arange(points_np.shape[0], dtype=np.float32)[:, None],
            queries["source_frame"].detach().float().cpu().numpy()[:, None],
            queries["source_view"].detach().float().cpu().numpy()[:, None],
            queries["source_u"].detach().float().cpu().numpy()[:, None],
            queries["source_v"].detach().float().cpu().numpy()[:, None],
            queries["source_depth_m"].detach().float().cpu().numpy()[:, None],
            points_np,
            source_rgb_np,
            queries["ray_dir_ref"].detach().float().cpu().numpy(),
            queries["ray_moment_ref"].detach().float().cpu().numpy(),
            queries["dynamic_probability"].detach().float().cpu().numpy()[:, None],
        ],
        axis=1,
    )
    query_columns = np.asarray(
        [
            "query_id",
            "source_frame",
            "source_view",
            "source_u",
            "source_v",
            "source_depth_m",
            "x_ref",
            "y_ref",
            "z_ref",
            "rgb_r",
            "rgb_g",
            "rgb_b",
            "ray_dir_x_ref",
            "ray_dir_y_ref",
            "ray_dir_z_ref",
            "ray_moment_x_ref",
            "ray_moment_y_ref",
            "ray_moment_z_ref",
            "dynamic_probability",
        ]
    )
    np.savez_compressed(args.output_dir / "stage1_static_queries.npz", queries=query_np, columns=query_columns)
    write_point_ply(args.output_dir / "stage1_static_queries.ply", points_np, colors_np)

    observation_np = tensor_to_numpy_dict(observation)
    np.savez_compressed(args.output_dir / "stage1_observations.npz", **observation_np)
    decoded_rgbd_path = None
    if args.save_decoded_rgbd:
        decoded_rgbd_path = args.output_dir / "stage1_decoded_rgbd.pt"
        torch.save(
            {
                "rgb_u8": rgb_u8.detach().cpu(),
                "depth_m": depth_m.detach().float().cpu(),
                "shape_note": "rgb_u8 [view, channel, frame, height, width], depth_m [view, frame, height, width]",
            },
            decoded_rgbd_path,
        )

    valid_count = observation["depth_consistent"].sum(dim=1).detach().cpu().numpy()
    in_bounds_count = observation["in_bounds"].sum(dim=1).detach().cpu().numpy()
    valid_hist_path = args.output_dir / "valid_observation_count_hist.jpg"
    in_bounds_hist_path = args.output_dir / "in_bounds_observation_count_hist.jpg"
    make_histogram_image(valid_count, "depth-consistent observation count", len(context_frames) * len(context_views)).save(
        valid_hist_path,
        quality=94,
    )
    make_histogram_image(in_bounds_count, "in-bounds observation count", len(context_frames) * len(context_views)).save(
        in_bounds_hist_path,
        quality=94,
    )
    projection_panel_paths = make_projection_panels(args.output_dir, args.clip_index, rgb_u8, observation, args)

    summary = {
        "manifest": str(args.manifest),
        "clip_index": int(args.clip_index),
        "clip_pt": row["clip_pt"],
        "output_dir": str(args.output_dir),
        "turbo": turbo_info,
        "decode_seconds": float(decode_seconds),
        "query_frames": query_frames,
        "query_views": [VIEW_NAMES[index] for index in query_views],
        "context_frames": context_frames,
        "context_views": [VIEW_NAMES[index] for index in context_views],
        "raw_queries_before_limit": raw_query_count,
        "kept_query_indices": int(keep_indices.shape[0]),
        "num_queries": int(points_np.shape[0]),
        "num_contexts": int(len(context_frames) * len(context_views)),
        "static_mask_source": args.static_mask_source,
        "dense_pixel_queries": bool(args.dense_pixel_queries),
        "flow_rgb_track_init": bool(args.flow_rgb_track_init),
        "flow_rgb_track_stats": flow_track_stats,
        "exclude_sky_by_depth": bool(args.exclude_sky_by_depth),
        "depth_thresholds": {
            "min_valid_m": float(args.depth_min_valid_m),
            "max_valid_m": float(args.depth_max_valid_m),
            "sky_depth_threshold_m": float(args.sky_depth_threshold_m),
            "abs_consistent_m": float(args.depth_abs_threshold_m),
            "rel_consistent": float(args.depth_rel_threshold),
        },
        "source_items": source_items,
        "context_items": context_items,
        "observation_stats": {
            "mean_in_bounds_count": float(np.mean(in_bounds_count)) if in_bounds_count.size else 0.0,
            "mean_depth_consistent_count": float(np.mean(valid_count)) if valid_count.size else 0.0,
            "ratio_ge_1_consistent": float(np.mean(valid_count >= 1)) if valid_count.size else 0.0,
            "ratio_ge_2_consistent": float(np.mean(valid_count >= 2)) if valid_count.size else 0.0,
            "ratio_ge_4_consistent": float(np.mean(valid_count >= 4)) if valid_count.size else 0.0,
        },
        "artifacts": {
            "queries_npz": str(args.output_dir / "stage1_static_queries.npz"),
            "queries_ply": str(args.output_dir / "stage1_static_queries.ply"),
            "observations_npz": str(args.output_dir / "stage1_observations.npz"),
            "decoded_rgbd_pt": str(decoded_rgbd_path) if decoded_rgbd_path is not None else None,
            "stage2_features_pt": str(stage2_features_path) if stage2_features_path is not None else None,
            "source_panels": source_panel_paths,
            "projection_panels": projection_panel_paths,
            "valid_hist": str(valid_hist_path),
            "in_bounds_hist": str(in_bounds_hist_path),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_queries": int(points_np.shape[0]),
                "num_contexts": int(len(context_frames) * len(context_views)),
                "mean_depth_consistent_count": summary["observation_stats"]["mean_depth_consistent_count"],
                "ratio_ge_2_consistent": summary["observation_stats"]["ratio_ge_2_consistent"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
