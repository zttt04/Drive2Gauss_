"""gsplat rendering and time-dependent Gaussian evaluation."""

from __future__ import annotations

import argparse

import torch

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
    # gsplat's packed rasterizer expects one shared background vector.  This
    # renderer submits one camera per call, so a shared vector is sufficient
    # and remains compatible with current gsplat releases.
    backgrounds = torch.full(
        (3,), float(args.background), dtype=torch.float32, device=device
    )
    colors, _, _ = rasterization(
        means=means_at_frame(gaussian, frame_index), quats=gaussian["quats"], scales=gaussian["scales"],
        opacities=opacities_at_frame(gaussian, frame_index, args), colors=gaussian["colors"], viewmats=viewmats, Ks=ks,
        width=render_width, height=render_height, near_plane=0.1, far_plane=200.0,
        backgrounds=backgrounds, render_mode="RGB", packed=True,
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
        packed=True,
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
