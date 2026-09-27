#!/usr/bin/env python3
"""Render PointForward flow-track Gaussians with a freely controllable camera.

Reuses the exact training render pipeline: gsplat rasterization of C128 appearance
features + target-camera-conditioned UNet decode (model.appearance_decoder).

Camera controls (applied on top of the original clip camera when --view is a real
camera, or fully free in --look-at mode):

  --raise-m        raise the camera along the reference-frame up (+z) by N meters
  --pitch-deg      rotate about the camera x axis (positive = look DOWN)
  --yaw-deg        rotate about the camera y axis (positive = pan)
  --roll-deg       rotate about the camera z axis
  --lateral-offset-m  lateral shift in meters (same convention as training)
  --look-at        free camera: space-separated "eye_x eye_y eye_z look_x look_y look_z up_x up_y up_z"
                   positions are in reference (frame0-lidar) coordinates; the tool
                   prints the scene gaussian center so you know where to aim.

Output: a panel image (GT | render) when GT exists, otherwise render-only.
With --novel-lateral-offset-m, static mode writes an origin | novel render pair.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
for p in (TOOLS, TOOLS / "inspect"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_static_pointforward_flowtrack_multiscene as ms  # noqa: E402
import train_static_pointforward_stage2 as pointforward  # noqa: E402
import prepare_query_static_mini_dataset as query_data  # noqa: E402
from gsplat import rasterization  # noqa: E402

VIEWS6 = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--ann-file", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--depth-root", type=Path, default=None)
    p.add_argument("--depth-map-json", type=Path, default=None)
    p.add_argument("--masked-flow-rgb-root", type=Path, default=None)
    p.add_argument("--masked-flow-index", type=Path, default=None)
    p.add_argument("--allow-missing-flow-rgb", action="store_true")
    p.add_argument(
        "--generated-flow-rgb-root",
        type=Path,
        default=None,
        help=(
            "Optional decoded generated-flow root. Accepts either gen_flow/<token> "
            "or gen_flow; replaces matched masked-flow RGB for the requested views."
        ),
    )
    p.add_argument("--clip-index", type=int, default=0)
    p.add_argument("--window-start", type=int, default=0, choices=[0, 4, 8, 12, 13])
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--view", type=str, default="CAM_FRONT_LEFT", choices=VIEWS6)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, nargs=2, default=[800, 424], help="width height")
    p.add_argument("--show-gt", action="store_true", default=True)
    p.add_argument("--no-gt", action="store_true")
    p.add_argument(
        "--gt-project-colors",
        action="store_true",
        help="Sample GT RGB at each Gaussian's original-camera projection and render with those colors.",
    )
    # camera controls
    p.add_argument("--raise-m", type=float, default=0.0)
    p.add_argument("--pitch-deg", type=float, default=0.0)
    p.add_argument("--yaw-deg", type=float, default=0.0)
    p.add_argument("--roll-deg", type=float, default=0.0)
    p.add_argument("--lateral-offset-m", type=float, default=0.0)
    p.add_argument(
        "--novel-lateral-offset-m",
        type=float,
        default=None,
        help="Render origin and a laterally shifted camera side by side in static mode.",
    )
    p.add_argument("--lateral-video-offsets-m", type=float, nargs="+", default=[])
    p.add_argument(
        "--motion-flow-overlay",
        action="store_true",
        help="Overlay projected per-Gaussian dynamic motion arrows; forces origin-only video output.",
    )
    p.add_argument(
        "--motion-flow-threshold",
        type=float,
        default=0.08,
        help="Minimum dynamic probability for motion-flow arrows (top-scoring fallback keeps the overlay visible).",
    )
    p.add_argument(
        "--motion-flow-max-arrows",
        type=int,
        default=1500,
        help="Maximum number of projected motion arrows per frame.",
    )
    p.add_argument(
        "--motion-flow-scale",
        type=float,
        default=8.0,
        help="Display-only scale for projected motion arrows; does not change Gaussian motion.",
    )
    p.add_argument(
        "--dynamic-gaussian-overlay",
        action="store_true",
        help="Mark final dynamic Gaussians on the origin-camera render.",
    )
    p.add_argument(
        "--dynamic-gaussian-threshold",
        type=float,
        default=0.08,
        help="Minimum final dynamic probability for Gaussian markers.",
    )
    p.add_argument(
        "--dynamic-gaussian-max-points",
        type=int,
        default=1200,
        help="Maximum dynamic Gaussian markers per frame.",
    )
    p.add_argument("--video-fps", type=float, default=4.0)
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip lateral MP4 writing while still exporting raw PNG frames.",
    )
    p.add_argument(
        "--raw-frame-output-dir",
        type=Path,
        default=None,
        help="Optional directory for lossless render frames saved before visualization overlays.",
    )
    p.add_argument(
        "--context-frame-count",
        type=int,
        default=4,
        help="Number of consecutive context frames used to predict one Gaussian set.",
    )
    p.add_argument(
        "--context-frames",
        type=int,
        nargs="+",
        default=None,
        help="Explicit frames used in one Gaussian prediction, overriding consecutive windows.",
    )
    p.add_argument(
        "--render-frames",
        type=int,
        nargs="+",
        default=None,
        help="Frames rendered from an explicit context; requires --context-frames.",
    )
    p.add_argument(
        "--single-view-context",
        action="store_true",
        help="Use CAM_FRONT only when constructing the PointForward context.",
    )
    p.add_argument(
        "--video-window-starts",
        type=int,
        nargs="+",
        default=None,
        help="Explicit context-window starts for video export; defaults to the four training windows.",
    )
    p.add_argument("--summary-output", type=Path, default=None)
    p.add_argument("--look-at", type=float, nargs=9, default=None,
                   metavar=("eye_x", "eye_y", "eye_z", "look_x", "look_y", "look_z", "up_x", "up_y", "up_z"))
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_model(args: argparse.Namespace, device: torch.device):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # build train args from a clean argv (the checkpoint config overrides everything anyway)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--manifest", "/tmp/placeholder", "--cache-root", "/tmp/placeholder",
                "--output-dir", "/tmp/placeholder", "--checkpoint-dir", "/tmp/placeholder"]
    try:
        train_args = ms.parse_args()
    finally:
        sys.argv = saved_argv
    for k, v in ckpt["config"].items():
        if hasattr(train_args, k):
            setattr(train_args, k, v)
    model = ms.FlowTrackRenderModel(train_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[renderer] ckpt {args.checkpoint.name} step={ckpt.get('step')} loaded", flush=True)
    return model, train_args, ckpt


def rot_x(a: float, device) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    R = torch.eye(4, device=device)
    R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    return R


def rot_y(a: float, device) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    R = torch.eye(4, device=device)
    R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    return R


def rot_z(a: float, device) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    R = torch.eye(4, device=device)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def look_at_gsplat(eye, center, up, device):
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)
    f = F.normalize(center - eye, dim=0)
    s = F.normalize(torch.cross(f, up), dim=0)
    u = torch.cross(s, f)
    R = torch.stack([s, u, -f], dim=1)
    t = -R @ eye
    m = torch.eye(4, device=device)
    m[:3, :3] = R
    m[:3, 3] = t
    flip = torch.eye(4, device=device)
    flip[2, 2] = -1.0
    return (flip @ m)[None]


def camera_payload(viewmat: torch.Tensor, intrinsics: torch.Tensor) -> dict:
    """Return JSON-friendly camera matrices for reproducible novel renders."""
    vm = viewmat[0].detach().cpu().float()
    k = intrinsics[0].detach().cpu().float()
    c2w = torch.linalg.inv(vm)
    return {
        "viewmat_world_to_camera": vm.tolist(),
        "camera_to_world": c2w.tolist(),
        "intrinsics": k.tolist(),
    }


def shifted_camera(
    clip,
    frame: int,
    view: int,
    width: int,
    height: int,
    train_args,
    device: torch.device,
    lateral_offset_m: float,
    raise_m: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a render camera from the clip camera and controlled offsets."""
    render_args = argparse.Namespace(**vars(train_args))
    render_args.camera_lateral_offset_m = float(lateral_offset_m)
    vm, K = pointforward.camera_for_target(
        clip, frame, view, width, height, render_args, device
    )
    if raise_m != 0.0:
        delta = torch.tensor([0.0, 0.0, raise_m], device=device)
        vm[0, :3, 3] = vm[0, :3, 3] - vm[0, :3, :3] @ delta
    rotation = torch.eye(4, device=device)
    if pitch_deg != 0.0:
        rotation = rot_x(math.radians(pitch_deg), device) @ rotation
    if yaw_deg != 0.0:
        rotation = rot_y(math.radians(yaw_deg), device) @ rotation
    if roll_deg != 0.0:
        rotation = rot_z(math.radians(roll_deg), device) @ rotation
    if not torch.equal(rotation, torch.eye(4, device=device)):
        vm = rotation @ vm
    return vm, K


def project_gt_colors(
    gaussian, frame, color_vm, color_K, gt_rgb, args, invalid_color=0.0,
    return_valid=False,
):
    """Attach GT colors to Gaussian points using the original camera projection."""
    means = pointforward.means_at_frame(gaussian, frame)
    homogeneous = torch.cat(
        [means, torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)], dim=-1
    )
    camera_points = homogeneous @ color_vm[0].transpose(0, 1)
    depth = camera_points[:, 2]
    pixels = camera_points[:, :3] @ color_K[0].transpose(0, 1)
    pixels = pixels[:, :2] / pixels[:, 2:3].clamp_min(1.0e-6)
    gt_height, gt_width = gt_rgb.shape[-2:]
    grid = torch.stack(
        [2.0 * (pixels[:, 0] + 0.5) / gt_width - 1.0,
         2.0 * (pixels[:, 1] + 0.5) / gt_height - 1.0],
        dim=-1,
    )
    sampled = F.grid_sample(
        gt_rgb[None], grid.reshape(1, -1, 1, 2), mode="bilinear",
        padding_mode="zeros", align_corners=False,
    )[0, :, :, 0].transpose(0, 1)
    valid = (
        (depth > 0.1)
        & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < gt_width)
        & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < gt_height)
    )
    colors = sampled.masked_fill(~valid[:, None], float(invalid_color))
    if return_valid:
        return colors, valid
    return colors


@torch.no_grad()
def draw_dynamic_gaussian_overlay(
    image_bgr: np.ndarray,
    gaussian: dict[str, torch.Tensor],
    sample,
    frame: int,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    threshold: float,
    max_points: int,
) -> tuple[np.ndarray, int]:
    """Mark dynamic query/Gaussian points projected into the render camera."""
    if max_points <= 0:
        return image_bgr, 0
    if sample is not None:
        means = sample.batch.anchors_ref
        dynamic_probability = sample.batch.query_features[:, -1].reshape(-1).clamp(0.0, 1.0)
    else:
        means = pointforward.means_at_frame(gaussian, frame)
        dynamic_probability = gaussian["dynamic_probability_pred"].reshape(-1).clamp(0.0, 1.0)
    homogeneous_now = torch.cat(
        [means, torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)],
        dim=-1,
    )
    camera_now = homogeneous_now @ viewmat[0].transpose(0, 1)
    depth_now = camera_now[:, 2]
    pixels_now = camera_now[:, :3] @ intrinsics[0].transpose(0, 1)
    pixels_now = pixels_now[:, :2] / pixels_now[:, 2:3].clamp_min(1.0e-6)
    valid = (
        torch.isfinite(pixels_now).all(dim=-1)
        & (depth_now > 0.1)
        & (pixels_now[:, 0] >= 0.0) & (pixels_now[:, 0] < width)
        & (pixels_now[:, 1] >= 0.0) & (pixels_now[:, 1] < height)
    )
    selected_indices = torch.nonzero(valid & (dynamic_probability >= float(threshold)), as_tuple=False).flatten()
    if selected_indices.numel() > max_points:
        top = torch.topk(dynamic_probability[selected_indices], int(max_points)).indices
        selected_indices = selected_indices[top]
    if selected_indices.numel() == 0:
        return image_bgr, 0
    starts = pixels_now[selected_indices].round().to(torch.int32).cpu().numpy()
    probabilities = dynamic_probability[selected_indices].cpu().numpy()
    for start, probability in zip(starts, probabilities, strict=True):
        strength = float(np.clip(probability, 0.0, 1.0))
        color = (0, int(200.0 * (1.0 - strength)), 255)
        cv2.circle(image_bgr, (int(start[0]), int(start[1])), 2, color, -1, cv2.LINE_AA)
    return image_bgr, int(selected_indices.numel())


def render_with_vm(
    gaussian, frame, vm, K, width, height, cam_idx, appearance_decoder, args, device,
    camera_affine=None, gt_rgb=None, gt_color_vm=None, gt_color_K=None,
):
    # Some cached clips carry an extra singleton batch dimension on camera data;
    # gsplat expects one camera dimension here.
    if vm.ndim == 4 and vm.shape[0] == 1:
        vm = vm[0]
    if K.ndim == 4 and K.shape[0] == 1:
        K = K[0]
    if vm.ndim == 2:
        vm = vm[None]
    if K.ndim == 2:
        K = K[None]
    if gt_rgb is not None:
        if gt_color_vm is None or gt_color_K is None:
            raise ValueError("GT color projection requires the original camera matrices")
        colors = project_gt_colors(gaussian, frame, gt_color_vm, gt_color_K, gt_rgb, args)
        rendered, _, _ = rasterization(
            means=pointforward.means_at_frame(gaussian, frame),
            quats=gaussian["quats"], scales=gaussian["scales"],
            opacities=pointforward.opacities_at_frame(gaussian, frame, args), colors=colors,
            viewmats=vm, Ks=K, width=width, height=height, near_plane=0.1, far_plane=300.0,
            backgrounds=None, render_mode="RGB",
        )
        return rendered[0].permute(2, 0, 1).clamp(0.0, 1.0)
    if appearance_decoder is None:
        # direct_rgb: rasterize per-Gaussian RGB directly, then optional affine
        colors, _, _ = rasterization(
            means=pointforward.means_at_frame(gaussian, frame), quats=gaussian["quats"], scales=gaussian["scales"],
            opacities=pointforward.opacities_at_frame(gaussian, frame, args), colors=gaussian["colors"],
            viewmats=vm, Ks=K, width=width, height=height, near_plane=0.1, far_plane=300.0,
            backgrounds=None, render_mode="RGB",
        )
        rgb = colors[0].permute(2, 0, 1).clamp(0.0, 1.0)
        if camera_affine is not None:
            rgb = camera_affine(rgb, cam_idx)
        return rgb
    means = pointforward.means_at_frame(gaussian, frame)
    feats, _, _ = rasterization(
        means=means, quats=gaussian["quats"], scales=gaussian["scales"],
        opacities=pointforward.opacities_at_frame(gaussian, frame, args),
        colors=gaussian["appearance_features"], viewmats=vm, Ks=K,
        width=width, height=height, near_plane=0.1, far_plane=300.0,
        backgrounds=None,
        render_mode="RGB", channel_chunk=gaussian["appearance_features"].shape[-1],
    )
    fm = feats[0].permute(2, 0, 1)
    ci = torch.tensor([cam_idx], device=device) if getattr(args, "target_camera_conditioning", False) else None
    return appearance_decoder(fm[None], ci)[0].clamp(0, 1)


def render_depth_alpha_with_vm(
    gaussian,
    frame: int,
    vm: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render metric depth and alpha for camera reprojection diagnostics."""
    depth_alpha, alpha, _ = rasterization(
        means=pointforward.means_at_frame(gaussian, frame),
        quats=gaussian["quats"],
        scales=gaussian["scales"],
        opacities=pointforward.opacities_at_frame(gaussian, frame, args),
        colors=torch.zeros((gaussian["means"].shape[0], 1), device=device),
        viewmats=vm,
        Ks=K,
        width=width,
        height=height,
        near_plane=0.1,
        far_plane=300.0,
        backgrounds=torch.zeros((1, 1), device=device),
        render_mode="ED",
    )
    return depth_alpha[0, ..., 0], alpha[0, ..., 0]


def depth_reprojection_metrics(
    origin_depth: torch.Tensor,
    origin_alpha: torch.Tensor,
    novel_depth: torch.Tensor,
    novel_alpha: torch.Tensor,
    origin_vm: torch.Tensor,
    novel_vm: torch.Tensor,
    K: torch.Tensor,
    alpha_threshold: float = 0.01,
) -> dict[str, float]:
    """Measure origin-depth points after projection into a novel camera."""
    device = origin_depth.device
    height, width = origin_depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=origin_depth.dtype),
        torch.arange(width, device=device, dtype=origin_depth.dtype),
        indexing="ij",
    )
    fx, fy, cx, cy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
    valid = (origin_depth > 0.1) & (origin_alpha > alpha_threshold)
    x_cam = (xx - cx) * origin_depth / fx.clamp_min(1.0e-6)
    y_cam = (yy - cy) * origin_depth / fy.clamp_min(1.0e-6)
    camera_points = torch.stack([x_cam, y_cam, origin_depth], dim=-1).reshape(-1, 3)
    origin_c2w = torch.linalg.inv(origin_vm[0])
    world_points = camera_points @ origin_c2w[:3, :3].transpose(0, 1) + origin_c2w[:3, 3]
    novel_camera_points = world_points @ novel_vm[0, :3, :3].transpose(0, 1) + novel_vm[0, :3, 3]
    novel_z = novel_camera_points[:, 2].reshape(height, width)
    novel_u = (novel_camera_points[:, 0] / novel_camera_points[:, 2].clamp_min(1.0e-6) * fx + cx).reshape(height, width)
    novel_v = (novel_camera_points[:, 1] / novel_camera_points[:, 2].clamp_min(1.0e-6) * fy + cy).reshape(height, width)
    in_bounds = (novel_z > 0.1) & (novel_u >= 0.0) & (novel_u <= width - 1.0) & (novel_v >= 0.0) & (novel_v <= height - 1.0)
    valid = valid & in_bounds
    if not bool(valid.any()):
        return {"valid_ratio": 0.0, "novel_hole_ratio": 1.0, "depth_abs_error_m": None}
    grid = torch.stack(
        [2.0 * (novel_u + 0.5) / width - 1.0, 2.0 * (novel_v + 0.5) / height - 1.0], dim=-1
    )
    sampled_depth = F.grid_sample(
        novel_depth[None, None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    sampled_alpha = F.grid_sample(
        novel_alpha[None, None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    projected_count = valid.sum().float()
    visible = valid & (sampled_alpha > alpha_threshold)
    depth_error = (sampled_depth - novel_z).abs()[visible]
    return {
        "valid_ratio": float(projected_count.item() / max(height * width, 1)),
        "novel_hole_ratio": float((valid & ~visible).sum().item() / max(projected_count.item(), 1.0)),
        "depth_abs_error_m": float(depth_error.mean().item()) if depth_error.numel() else None,
        "visible_ratio": float(visible.sum().item() / max(projected_count.item(), 1.0)),
    }


def write_lateral_video(cli, model, train_args, ckpt, row, cache, clip, view, device) -> None:
    width, height = cli.size
    if cli.context_frames is not None:
        context_specs = [(int(cli.context_frames[0]), list(cli.context_frames))]
    else:
        window_starts = cli.video_window_starts or ms.WINDOW_STARTS
        context_specs = [(int(window_start), None) for window_start in window_starts]
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not cli.no_video:
        writer = cv2.VideoWriter(
            str(cli.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cli.video_fps,
            (width * len(cli.lateral_video_offsets_m), height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {cli.output}")
    if cli.raw_frame_output_dir is not None:
        cli.raw_frame_output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_frames = 0
    motion_arrow_counts = []
    dynamic_marker_counts = []
    psnr_values = []
    raw_frame_paths = []
    window_summaries = []
    inference_start = time.perf_counter()
    try:
        with torch.no_grad():
            for window_start, context_frames in context_specs:
                build_start = time.perf_counter()
                sample = ms.build_window_sample(
                    row,
                    window_start,
                    cache,
                    clip,
                    train_args,
                    device,
                    context_frames=context_frames,
                )
                build_seconds = time.perf_counter() - build_start
                predict_start = time.perf_counter()
                gaussian = model.predict_gaussians(sample, train_args)
                predict_seconds = time.perf_counter() - predict_start
                window_render_start = time.perf_counter()
                frames_to_render = cli.render_frames or sample.frames
                video_length = int(
                    clip.get("video_length", clip["rgb_target"].shape[0])
                )
                if any(frame < 0 or frame >= video_length for frame in frames_to_render):
                    raise ValueError(
                        f"Render frames {frames_to_render} exceed video length {video_length}"
                    )
                for frame in frames_to_render:
                    rendered_views = []
                    for offset_index, offset_m in enumerate(cli.lateral_video_offsets_m):
                        train_args.camera_lateral_offset_m = offset_m
                        vm, K = pointforward.camera_for_target(
                            clip, frame, view, width, height, train_args, device
                        )
                        if cli.raise_m != 0.0:
                            delta = torch.tensor([0.0, 0.0, cli.raise_m], device=device)
                            vm[0, :3, 3] = vm[0, :3, 3] - vm[0, :3, :3] @ delta
                        R = torch.eye(4, device=device)
                        if cli.pitch_deg != 0.0:
                            R = rot_x(math.radians(cli.pitch_deg), device) @ R
                        if cli.yaw_deg != 0.0:
                            R = rot_y(math.radians(cli.yaw_deg), device) @ R
                        if cli.roll_deg != 0.0:
                            R = rot_z(math.radians(cli.roll_deg), device) @ R
                        if not torch.equal(R, torch.eye(4, device=device)):
                            vm = R @ vm
                        rendered = render_with_vm(
                            gaussian,
                            frame,
                            vm,
                            K,
                            width,
                            height,
                            view,
                            model.appearance_decoder,
                            train_args,
                            device,
                            camera_affine=getattr(model, "camera_affine", None),
                        )
                        if (
                            abs(offset_m) < 1.0e-8
                            and cli.raise_m == 0.0
                            and cli.pitch_deg == 0.0
                            and cli.yaw_deg == 0.0
                            and cli.roll_deg == 0.0
                        ):
                            target = ms.target_image(
                                sample, ms.PackagedQuerySource(), frame, view, train_args, device
                            )
                            mse = float((rendered - target).square().mean())
                            psnr_values.append(10.0 * math.log10(1.0 / max(mse, 1.0e-12)))
                        image = (
                            rendered.detach()
                            .clamp(0, 1)
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                        image = cv2.cvtColor(
                            np.floor(image * 255.0 + 0.5).astype(np.uint8),
                            cv2.COLOR_RGB2BGR,
                        )
                        if cli.raw_frame_output_dir is not None:
                            raw_frame_path = cli.raw_frame_output_dir / (
                                f"frame{total_frames:04d}_offset{offset_index:02d}.png"
                            )
                            if not cv2.imwrite(str(raw_frame_path), image):
                                raise RuntimeError(f"Failed to save raw render frame {raw_frame_path}")
                            raw_frame_paths.append(str(raw_frame_path))
                        arrow_count = 0
                        dynamic_marker_count = 0
                        if cli.motion_flow_overlay and frame < video_length - 1:
                            image, arrow_count = draw_motion_flow_overlay(
                                image,
                                gaussian,
                                frame,
                                vm,
                                K,
                                width,
                                height,
                                cli.motion_flow_threshold,
                                cli.motion_flow_max_arrows,
                                cli.motion_flow_scale,
                            )
                        if cli.dynamic_gaussian_overlay:
                            image, dynamic_marker_count = draw_dynamic_gaussian_overlay(
                                image,
                                gaussian,
                                sample,
                                frame,
                                vm,
                                K,
                                width,
                                height,
                                cli.dynamic_gaussian_threshold,
                                cli.dynamic_gaussian_max_points,
                            )
                        label = f"{offset_m:+.1f} m"
                        cv2.rectangle(image, (12, 10), (145, 48), (0, 0, 0), -1)
                        cv2.putText(
                            image,
                            label,
                            (22, 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        rendered_views.append(image)
                    if cli.motion_flow_overlay:
                        motion_arrow_counts.append(arrow_count)
                    if cli.dynamic_gaussian_overlay:
                        dynamic_marker_counts.append(dynamic_marker_count)
                    if writer is not None:
                        writer.write(np.concatenate(rendered_views, axis=1))
                    total_frames += 1
                window_summaries.append(
                    {
                        "window_start": int(window_start),
                        "frames": sample.frames,
                        "context_frames": sample.frames,
                        "render_frames": list(frames_to_render),
                        "views": [pointforward.VIEW_NAMES[index] for index in sample.views],
                        "gaussians": int(gaussian["means"].shape[0]),
                        "dynamic_probability_mean": float(sample.dynamic_probability_mean),
                        "build_seconds": build_seconds,
                        "predict_seconds": predict_seconds,
                        "render_seconds": time.perf_counter() - window_render_start,
                    }
                )
                del sample, gaussian
    finally:
        train_args.camera_lateral_offset_m = 0.0
        if writer is not None:
            writer.release()
    elapsed_seconds = time.perf_counter() - inference_start
    summary = {
        "checkpoint": str(cli.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "manifest_index": int(row["manifest_index"]),
        "context_frame_count": int(train_args.context_frame_count),
        "context_frames": cli.context_frames,
        "render_frames": cli.render_frames,
        "single_view_context": bool(cli.single_view_context),
        "output_view": pointforward.VIEW_NAMES[view],
        "flow_rgb_source": clip.get("_flow_rgb_source", "packaged"),
        "lateral_offsets_m": cli.lateral_video_offsets_m,
        "total_frames": total_frames,
        "zero_offset_psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "zero_offset_psnr_per_frame": psnr_values,
        "raw_frame_output_dir": (
            str(cli.raw_frame_output_dir) if cli.raw_frame_output_dir is not None else None
        ),
        "raw_frame_paths": raw_frame_paths,
        "motion_flow_overlay": bool(cli.motion_flow_overlay),
        "motion_flow_threshold": float(cli.motion_flow_threshold),
        "motion_flow_max_arrows": int(cli.motion_flow_max_arrows),
        "motion_flow_scale": float(cli.motion_flow_scale),
        "motion_arrow_counts": motion_arrow_counts,
        "dynamic_gaussian_overlay": bool(cli.dynamic_gaussian_overlay),
        "dynamic_gaussian_threshold": float(cli.dynamic_gaussian_threshold),
        "dynamic_gaussian_max_points": int(cli.dynamic_gaussian_max_points),
        "dynamic_marker_counts": dynamic_marker_counts,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "windows": window_summaries,
    }
    if cli.summary_output is not None:
        cli.summary_output.parent.mkdir(parents=True, exist_ok=True)
        cli.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[renderer] saved {total_frames}-frame lateral video {cli.output} "
        f"PSNR={summary['zero_offset_psnr_mean']} elapsed={elapsed_seconds:.2f}s",
        flush=True,
    )


def replace_with_generated_flow_rgb(
    clip: dict,
    row: dict,
    root: Path,
    view_indices: list[int],
    width: int,
    height: int,
) -> None:
    token = str(row["token"])
    token_root = root / token if (root / token).is_dir() else root
    video_length = int(clip.get("video_length", row.get("video_length", 17)))
    flow_rgb = torch.full(
        (video_length, len(VIEWS6), height, width, 3),
        255,
        dtype=torch.uint8,
    )
    flow_valid = torch.zeros(
        (video_length, len(VIEWS6), height, width),
        dtype=torch.uint8,
    )
    missing = []
    for frame in range(video_length):
        for view in view_indices:
            image_path = token_root / VIEWS6[view] / f"{frame}.jpg"
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                missing.append(str(image_path))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            flow_rgb[frame, view] = torch.from_numpy(rgb.copy())
            flow_valid[frame, view] = 1
    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} generated flow RGB frames under {token_root}. "
            f"First missing paths:\n{preview}"
        )
    clip["flow_rgb_target"] = flow_rgb
    clip["flow_rgb_valid_target"] = flow_valid
    clip["_flow_rgb_source"] = f"generated_flow_latent_decode:{token_root}"


def load_clip(cli, row, train_args, view_indices=None):
    requested_views = list(view_indices or range(len(VIEWS6)))
    payload = torch.load(
        ms.feature_cache.source_path(row), map_location="cpu", weights_only=False
    )
    if "camera_intrinsics" in payload:
        clip = payload
        clip.setdefault("_flow_rgb_source", "packaged")
    elif cli.ann_file is not None and cli.data_root is not None and cli.masked_flow_index is not None:
        query_source = ms.OnlineQuerySource(
            ann_file=cli.ann_file,
            data_root=cli.data_root,
            flow_rgb_root=cli.masked_flow_rgb_root,
            flow_index_path=cli.masked_flow_index,
            width=train_args.width,
            height=train_args.height,
            allow_missing_flow_rgb=cli.allow_missing_flow_rgb,
            zero_flow_input=False,
        )
        clip = query_source.load_clip(row, requested_views)
        rgb_target = np.empty(
            (int(row.get("video_length", 17)), len(VIEWS6), train_args.height, train_args.width, 3),
            dtype=np.uint8,
        )
        for frame_index in range(rgb_target.shape[0]):
            for view_index in range(rgb_target.shape[1]):
                rgb_target[frame_index, view_index] = query_source.target_rgb(
                    clip, frame_index, view_index
                )
        clip["rgb_target"] = torch.from_numpy(rgb_target)
        clip["_flow_rgb_source"] = "matched_masked_flow_rgb"
    else:
        required = {
            "ann_file": cli.ann_file,
            "data_root": cli.data_root,
            "depth_root": cli.depth_root,
            "depth_map_json": cli.depth_map_json,
            "masked_flow_rgb_root": cli.masked_flow_rgb_root,
            "masked_flow_index": cli.masked_flow_index,
        }
        missing = [name.replace("_", "-") for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "The selected clip requires reconstruction arguments: "
                + ", ".join(f"--{name}" for name in missing)
            )
        annotations = query_data.load_ann(cli.ann_file)
        depth_map = query_data.load_json(cli.depth_map_json)
        masked_flow_index = query_data.load_json(cli.masked_flow_index)
        payload["rdepth_root"] = str(cli.depth_root)
        payload["depth_map_json"] = str(cli.depth_map_json)
        build_args = argparse.Namespace(
            ann_file=cli.ann_file,
            data_root=cli.data_root,
            height=train_args.height,
            width=train_args.width,
            low_height=53,
            low_width=100,
            masked_flow_rgb_root=cli.masked_flow_rgb_root,
            masked_flow_index=cli.masked_flow_index,
            allow_missing_flow_rgb=cli.allow_missing_flow_rgb,
        )
        clip = query_data.build_clip(
            row, payload, annotations["infos"], depth_map, build_args, masked_flow_index
        )["tensors"]
        clip["_flow_rgb_source"] = "matched_masked_flow_rgb"
    if getattr(cli, "generated_flow_rgb_root", None) is not None:
        replace_with_generated_flow_rgb(
            clip,
            row,
            cli.generated_flow_rgb_root,
            requested_views,
            train_args.width,
            train_args.height,
        )
    return clip


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    model, train_args, ckpt = load_model(cli, device)
    if cli.context_frame_count <= 0:
        raise ValueError("--context-frame-count must be positive")
    if cli.render_frames is not None and cli.context_frames is None:
        raise ValueError("--render-frames requires --context-frames")
    if cli.motion_flow_overlay:
        if cli.motion_flow_threshold < 0.0 or cli.motion_flow_threshold > 1.0:
            raise ValueError("--motion-flow-threshold must be in [0, 1]")
        if cli.motion_flow_max_arrows <= 0:
            raise ValueError("--motion-flow-max-arrows must be positive")
        if cli.motion_flow_scale <= 0.0:
            raise ValueError("--motion-flow-scale must be positive")
        if cli.lateral_video_offsets_m and any(
            abs(float(offset)) > 1.0e-8 for offset in cli.lateral_video_offsets_m
        ):
            raise ValueError("--motion-flow-overlay only supports the origin camera")
        if not cli.lateral_video_offsets_m:
            cli.lateral_video_offsets_m = [0.0]
    if cli.dynamic_gaussian_overlay:
        if cli.dynamic_gaussian_threshold < 0.0 or cli.dynamic_gaussian_threshold > 1.0:
            raise ValueError("--dynamic-gaussian-threshold must be in [0, 1]")
        if cli.dynamic_gaussian_max_points <= 0:
            raise ValueError("--dynamic-gaussian-max-points must be positive")
        if cli.lateral_video_offsets_m and any(
            abs(float(offset)) > 1.0e-8 for offset in cli.lateral_video_offsets_m
        ):
            raise ValueError("--dynamic-gaussian-overlay only supports the origin camera")
        if not cli.lateral_video_offsets_m:
            cli.lateral_video_offsets_m = [0.0]
    if cli.context_frames is not None:
        if len(cli.context_frames) != len(set(cli.context_frames)):
            raise ValueError("--context-frames must contain distinct frames")
        train_args.context_frame_count = len(cli.context_frames)
    else:
        train_args.context_frame_count = cli.context_frame_count
    if cli.single_view_context:
        train_args.single_view_train = True

    rows = ms.read_manifest(cli.manifest)
    row = rows[cli.clip_index]
    cache = torch.load(ms.cache_path_for(row, cli.cache_root), map_location="cpu", weights_only=False)
    clip = load_clip(cli, row, train_args, cache.get("view_indices"))
    if cli.view not in VIEWS6:
        raise ValueError(f"view {cli.view} not in {VIEWS6}")
    view = VIEWS6.index(cli.view)
    if cli.lateral_video_offsets_m:
        if cli.look_at is not None:
            raise ValueError("--look-at cannot be combined with --lateral-video-offsets-m")
        write_lateral_video(cli, model, train_args, ckpt, row, cache, clip, view, device)
        return
    if cli.novel_lateral_offset_m is not None and cli.look_at is not None:
        raise ValueError("--look-at cannot be combined with --novel-lateral-offset-m")
    if cli.novel_lateral_offset_m is not None and cli.gt_project_colors:
        raise ValueError("--gt-project-colors is only valid for a single render")
    with torch.no_grad():
        sample = ms.build_window_sample(
            row,
            cli.window_start,
            cache,
            clip,
            train_args,
            device,
            context_frames=cli.context_frames,
        )
        gaussian = model.predict_gaussians(sample, train_args)
    if cli.frame not in sample.frames:
        raise ValueError(f"frame {cli.frame} not in window frames {sample.frames}")
    frame = cli.frame
    means = pointforward.means_at_frame(gaussian, frame)
    center = means.mean(dim=0)
    print(f"[renderer] scene{row['scene_index']} start{row['scene_frame_start']} window={cli.window_start} "
          f"frame={frame} view={cli.view} queries={means.shape[0]}", flush=True)
    print(f"[renderer] gaussian center (ref coords) = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) "
          f"| spans x[{means[:, 0].min():.0f},{means[:, 0].max():.0f}] y[{means[:, 1].min():.0f},{means[:, 1].max():.0f}] "
          f"z[{means[:, 2].min():.0f},{means[:, 2].max():.0f}]", flush=True)

    width, height = cli.size
    with torch.no_grad():
        if cli.look_at is not None:
            ex, ey, ez, lx, ly, lz, ux, uy, uz = cli.look_at
            vm = look_at_gsplat([ex, ey, ez], [lx, ly, lz], [ux, uy, uz], device)
            K = torch.tensor(
                [[[width * 0.5, 0, width * 0.5], [0, width * 0.5, height * 0.5], [0, 0, 1]]],
                device=device,
                dtype=torch.float32,
            )
            rgb = render_with_vm(
                gaussian, frame, vm, K, width, height, view, model.appearance_decoder,
                train_args, device, camera_affine=getattr(model, "camera_affine", None),
            )
            panels = [rgb]
            camera_records = {"look_at": camera_payload(vm, K)}
        else:
            offsets = (
                [0.0, float(cli.novel_lateral_offset_m)]
                if cli.novel_lateral_offset_m is not None
                else [float(cli.lateral_offset_m)]
            )
            panels = []
            camera_records = {}
            vm_records = {}
            depth_alpha_records = []
            for offset in offsets:
                vm, K = shifted_camera(
                    clip,
                    frame,
                    view,
                    width,
                    height,
                    train_args,
                    device,
                    lateral_offset_m=offset,
                    raise_m=cli.raise_m,
                    pitch_deg=cli.pitch_deg,
                    yaw_deg=cli.yaw_deg,
                    roll_deg=cli.roll_deg,
                )
                panels.append(
                    render_with_vm(
                        gaussian,
                        frame,
                        vm,
                        K,
                        width,
                        height,
                        view,
                        model.appearance_decoder,
                        train_args,
                        device,
                        camera_affine=getattr(model, "camera_affine", None),
                    )
                )
                if cli.novel_lateral_offset_m is not None:
                    depth_alpha_records.append(
                        render_depth_alpha_with_vm(
                            gaussian, frame, vm, K, width, height, train_args, device
                        )
                    )
                key = "origin" if len(offsets) == 2 and offset == 0.0 else f"offset_{offset:+g}m"
                camera_records[key] = camera_payload(vm, K)
                vm_records[key] = (vm, K)

            if cli.novel_lateral_offset_m is not None:
                origin_depth, origin_alpha = depth_alpha_records[0]
                novel_depth, novel_alpha = depth_alpha_records[1]
                novel_key = f"offset_{float(cli.novel_lateral_offset_m):+g}m"
                origin_vm, origin_K = vm_records["origin"]
                novel_vm, novel_K = vm_records[novel_key]
                reprojection = depth_reprojection_metrics(
                    origin_depth,
                    origin_alpha,
                    novel_depth,
                    novel_alpha,
                    origin_vm,
                    novel_vm,
                    origin_K,
                )

        if cli.novel_lateral_offset_m is None and not cli.no_gt and cli.look_at is None:
            target = ms.target_image(
                sample, ms.PackagedQuerySource(), frame, view, train_args, device
            )
            mse = ((panels[0] - target) ** 2).mean().item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
            print(f"[renderer] PSNR vs GT = {psnr:.2f}", flush=True)
            panels = [target, panels[0]]

        arrs = [
            (im.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            for im in panels
        ]
        if cli.novel_lateral_offset_m is not None:
            labels = ["origin", f"novel {cli.novel_lateral_offset_m:+g}m"]
            for image, label in zip(arrs, labels, strict=True):
                cv2.rectangle(image, (12, 10), (190, 48), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    label,
                    (22, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(arrs, axis=1)).save(cli.output)
        if cli.novel_lateral_offset_m is not None:
            camera_path = cli.output.with_name(f"{cli.output.stem}_cameras.json")
            camera_path.write_text(
                json.dumps(
                    {
                        "frame": int(frame),
                        "view": cli.view,
                        "novel_lateral_offset_m": float(cli.novel_lateral_offset_m),
                        "cameras": camera_records,
                        "depth_reprojection": reprojection,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[renderer] saved camera metadata {camera_path}", flush=True)
        print(f"[renderer] saved {cli.output}", flush=True)


if __name__ == "__main__":
    main()
