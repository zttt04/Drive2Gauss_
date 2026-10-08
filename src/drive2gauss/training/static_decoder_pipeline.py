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
import torch.nn.functional as F

from drive2gauss.data import manifest as dataset_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]

from drive2gauss.models.gaussian_modules import (  # noqa: E402
    FeatureRenderUNet,
    LearnedLocalFeatureSampler,
    StaticPointForwardModel,
    camera_for_target,
    means_at_frame,
    opacities_at_frame,
    render_appearance_features,
    render_gsplat,
    render_output,
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
    rows = dataset_manifest.read_jsonl(path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest {path} does not contain row {index}")
    return rows[index]


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
