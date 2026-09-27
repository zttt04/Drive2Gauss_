"""Stage4-style RGB FD utilities for the DiST CogVideoX trainer.

This module keeps the old online RGB FD path untouched.  The Stage4 path uses
three explicit boundaries:

* fixed, precomputed GT moments (one population per camera);
* a detached generated-feature queue populated from the frozen step-3600 model;
* a latent gradient boundary so the frozen VAE/feature graph is released before
  the diffusion model backward.

The custom decoder VJP mirrors ``AutoencoderKLCogVideoX.tiled_decode``.  Its
forward pass stores decoded tiles, while backward replays one spatial/temporal
tile at a time and accumulates the exact gradient into the input latent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .frechet_rgb_loss import _sqrt_psd, differentiable_all_gather


def _temporal_ranges(num_frames: int, frame_batch_size: int) -> tuple[tuple[int, int], ...]:
    ranges = []
    for index in range(max(num_frames // frame_batch_size, 1)):
        remaining = num_frames % frame_batch_size
        start = frame_batch_size * index + (0 if index == 0 else remaining)
        end = frame_batch_size * (index + 1) + remaining
        ranges.append((start, end))
    return tuple(ranges)


def _blend_vertical(above: torch.Tensor, current: torch.Tensor, extent: int) -> torch.Tensor:
    extent = min(int(above.shape[-2]), int(current.shape[-2]), int(extent))
    result = current.clone()
    for offset in range(extent):
        result[..., offset, :] = (
            above[..., -extent + offset, :] * (1.0 - offset / extent)
            + current[..., offset, :] * (offset / extent)
        )
    return result


def _blend_horizontal(left: torch.Tensor, current: torch.Tensor, extent: int) -> torch.Tensor:
    extent = min(int(left.shape[-1]), int(current.shape[-1]), int(extent))
    result = current.clone()
    for offset in range(extent):
        result[..., offset] = (
            left[..., -extent + offset] * (1.0 - offset / extent)
            + current[..., offset] * (offset / extent)
        )
    return result


def _stitch_tiles(
    tiles: tuple[torch.Tensor, ...],
    rows: int,
    columns: int,
    blend_extent_height: int,
    blend_extent_width: int,
    row_limit_height: int,
    row_limit_width: int,
) -> torch.Tensor:
    if len(tiles) != rows * columns:
        raise ValueError("decoded tile count does not match tiling geometry")
    processed_rows = []
    output_rows = []
    index = 0
    for row_index in range(rows):
        processed_row = []
        output_row = []
        for column_index in range(columns):
            tile = tiles[index]
            index += 1
            if row_index:
                tile = _blend_vertical(
                    processed_rows[row_index - 1][column_index],
                    tile,
                    blend_extent_height,
                )
            if column_index:
                tile = _blend_horizontal(processed_row[column_index - 1], tile, blend_extent_width)
            processed_row.append(tile)
            output_row.append(tile[..., :row_limit_height, :row_limit_width])
        processed_rows.append(processed_row)
        output_rows.append(torch.cat(output_row, dim=-1))
    return torch.cat(output_rows, dim=-2)


def _tile_geometry(module: torch.nn.Module, latent: torch.Tensor):
    overlap_height = int(module.tile_latent_min_height * (1 - module.tile_overlap_factor_height))
    overlap_width = int(module.tile_latent_min_width * (1 - module.tile_overlap_factor_width))
    blend_extent_height = int(module.tile_sample_min_height * module.tile_overlap_factor_height)
    blend_extent_width = int(module.tile_sample_min_width * module.tile_overlap_factor_width)
    row_limit_height = module.tile_sample_min_height - blend_extent_height
    row_limit_width = module.tile_sample_min_width - blend_extent_width
    if min(overlap_height, overlap_width, blend_extent_height, blend_extent_width) <= 0:
        raise ValueError("invalid CogVideoX VAE tiling geometry")
    row_starts = tuple(range(0, int(latent.shape[-2]), overlap_height))
    column_starts = tuple(range(0, int(latent.shape[-1]), overlap_width))
    positions = tuple((row, column) for row in row_starts for column in column_starts)
    return (
        blend_extent_height,
        blend_extent_width,
        row_limit_height,
        row_limit_width,
        positions,
        len(row_starts),
        len(column_starts),
    )


def _decode_chunk(module: torch.nn.Module, latent_chunk: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    decoded_latent = latent_chunk / float(scaling_factor)
    decoder_parameter = next(module.decoder.parameters(), None)
    if decoder_parameter is not None and decoded_latent.dtype != decoder_parameter.dtype:
        decoded_latent = decoded_latent.to(dtype=decoder_parameter.dtype)
    if module.post_quant_conv is not None:
        decoded_latent = module.post_quant_conv(decoded_latent)
    return module.decoder(decoded_latent)


class _CogVideoXTiledVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent: torch.Tensor, vae_wrapper: torch.nn.Module) -> torch.Tensor:
        module = vae_wrapper.module
        frame_ranges = _temporal_ranges(int(latent.shape[2]), int(module.num_latent_frames_batch_size))
        geometry = _tile_geometry(module, latent)
        blend_h, blend_w, limit_h, limit_w, positions, rows, columns = geometry
        raw_tiles = []
        for row_start, column_start in positions:
            chunks = []
            for start, end in frame_ranges:
                tile_latent = latent[
                    :, :, start:end,
                    row_start:row_start + module.tile_latent_min_height,
                    column_start:column_start + module.tile_latent_min_width,
                ]
                chunks.append(_decode_chunk(module, tile_latent, vae_wrapper.scaling_factor))
            module._clear_fake_context_parallel_cache()
            raw_tiles.append(torch.cat(chunks, dim=2))
        output = _stitch_tiles(
            tuple(raw_tiles), rows, columns, blend_h, blend_w, limit_h, limit_w
        )
        ctx.save_for_backward(latent)
        ctx.vae_wrapper = vae_wrapper
        ctx.raw_tiles = tuple(raw_tiles)
        ctx.geometry = geometry
        ctx.frame_ranges = frame_ranges
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        (latent,) = ctx.saved_tensors
        vae_wrapper = ctx.vae_wrapper
        module = vae_wrapper.module
        blend_h, blend_w, limit_h, limit_w, positions, rows, columns = ctx.geometry
        raw_tiles = ctx.raw_tiles
        with torch.enable_grad():
            tile_leaves = tuple(tile.detach().requires_grad_(True) for tile in raw_tiles)
            stitched = _stitch_tiles(
                tile_leaves, rows, columns, blend_h, blend_w, limit_h, limit_w
            )
            tile_grads = torch.autograd.grad(
                stitched,
                tile_leaves,
                grad_outputs=output_gradient,
                retain_graph=False,
                create_graph=False,
            )
            latent_gradient = torch.zeros_like(latent)
            for (row_start, column_start), tile_gradient in zip(positions, tile_grads):
                for (start, end), gradient_chunk in zip(
                    ctx.frame_ranges, tile_gradient.split([end - start for start, end in ctx.frame_ranges], dim=2)
                ):
                    tile_latent = latent[
                        :, :, start:end,
                        row_start:row_start + module.tile_latent_min_height,
                        column_start:column_start + module.tile_latent_min_width,
                    ].detach().requires_grad_(True)
                    decoded = _decode_chunk(module, tile_latent, vae_wrapper.scaling_factor)
                    tile_latent_gradient = torch.autograd.grad(
                        decoded,
                        tile_latent,
                        grad_outputs=gradient_chunk,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    latent_gradient[
                        :, :, start:end,
                        row_start:row_start + module.tile_latent_min_height,
                        column_start:column_start + module.tile_latent_min_width,
                    ].add_(tile_latent_gradient)
                    module._clear_fake_context_parallel_cache()
        ctx.vae_wrapper = None
        ctx.raw_tiles = ()
        ctx.geometry = None
        return latent_gradient, None


def memory_bounded_cogvideox_decode(vae_wrapper: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Decode a frozen, spatially tiled CogVideoX VAE with a bounded VJP."""
    if not hasattr(vae_wrapper, "module"):
        raise TypeError("expected VideoAutoencoderKLCogVideoX wrapper")
    module = vae_wrapper.module
    if not bool(getattr(module, "use_tiling", False)):
        raise ValueError("Stage4 bounded decode requires vae.module.enable_tiling()")
    if any(parameter.requires_grad for parameter in vae_wrapper.parameters()):
        raise ValueError("Stage4 bounded decode requires a frozen VAE")
    return _CogVideoXTiledVJP.apply(latent, vae_wrapper)


def decode_rgb_stage4(vae_wrapper, rgb_latent: torch.Tensor, *, use_bounded_vjp: bool = True) -> torch.Tensor:
    """Decode ``[N,C,T,H,W]`` RGB latents, choosing the bounded path when needed."""
    if use_bounded_vjp and torch.is_grad_enabled() and rgb_latent.requires_grad:
        return memory_bounded_cogvideox_decode(vae_wrapper, rgb_latent)
    return vae_wrapper.decode(rgb_latent)


def decode_rgb_stage4_turbo(
    turbo_decoder: torch.nn.Module,
    rgb_latent: torch.Tensor,
    *,
    latent_scale: float,
    checkpoint_decode: bool = False,
) -> torch.Tensor:
    """Decode CogVideoX RGB latents with the frozen Turbo-VAED-Cog decoder."""
    decoder_parameter = next(turbo_decoder.parameters(), None)
    decode_dtype = decoder_parameter.dtype if decoder_parameter is not None else rgb_latent.dtype
    scaled_latent = (rgb_latent * float(latent_scale)).to(dtype=decode_dtype)

    def decode(latent: torch.Tensor) -> torch.Tensor:
        return turbo_decoder.decode(latent, return_dict=False)[0]

    if checkpoint_decode and torch.is_grad_enabled() and scaled_latent.requires_grad:
        from torch.utils.checkpoint import checkpoint as activation_checkpoint

        return activation_checkpoint(decode, scaled_latent, use_reentrant=False)
    return decode(scaled_latent)


def select_single_camera_latent(
    latent: torch.Tensor,
    batch_size: int,
    num_cameras: int,
    camera_index: int,
) -> torch.Tensor:
    """Select one camera from a ``[B*NC,C,T,H,W]`` camera-major latent batch."""
    batch_size = int(batch_size)
    num_cameras = int(num_cameras)
    camera_index = int(camera_index)
    if latent.shape[0] != batch_size * num_cameras:
        raise ValueError(
            f"expected first latent dimension B*NC={batch_size * num_cameras}, got {latent.shape[0]}"
        )
    if not 0 <= camera_index < num_cameras:
        raise ValueError(f"camera_index must be in [0,{num_cameras}), got {camera_index}")
    indices = torch.arange(batch_size, device=latent.device) * num_cameras + camera_index
    return latent.index_select(0, indices)


def build_per_view_rgb_features(decoded: torch.Tensor, batch_size: int, num_cameras: int, pool_size: int) -> torch.Tensor:
    """Return ``[B, NC, D]`` features; cameras remain separate populations."""
    decoded = decoded.float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
    pooled = F.adaptive_avg_pool3d(decoded, (1, int(pool_size), int(pool_size)))
    return pooled.flatten(1).view(batch_size, num_cameras, -1)


class StyleGANVI3DFeatureExtractor(torch.nn.Module):
    """Frozen StyleGAN-V I3D feature extractor used by the historical FVD code.

    The metric loader converts RGB frames to ``[0, 1]`` and resizes each frame
    to 224x224 before calling the TorchScript model with ``rescale=False`` and
    ``resize=False``.  This class keeps that contract while preserving the
    gradient with respect to the decoded input for the Stage4 VJP.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        clip_length: int = 16,
        resolution: int = 224,
        batch_size: int = 1,
        device=None,
    ):
        super().__init__()
        if int(clip_length) != 16:
            raise ValueError("the historical StyleGAN-V I3D checkpoint expects 16 frames")
        self.clip_length = int(clip_length)
        self.resolution = int(resolution)
        self.batch_size = max(int(batch_size), 1)
        self.model = torch.jit.load(str(checkpoint), map_location=device or "cpu").eval()
        if device is not None:
            self.model = self.model.to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _prepare_video(self, decoded: torch.Tensor) -> torch.Tensor:
        if decoded.ndim != 5 or decoded.shape[1] != 3:
            raise ValueError(f"decoded RGB must be [N,3,T,H,W], got {tuple(decoded.shape)}")
        video = decoded.float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
        num_frames = int(video.shape[2])
        # The 17-frame metric clips drop the first context frame (f01), while
        # the 9-frame training bucket is temporally resized to the same 16-frame
        # contract used by the original FVD implementation.
        if num_frames == 17:
            video = video[:, :, 1:17]
        elif num_frames != self.clip_length:
            video = F.interpolate(
                video,
                size=(self.clip_length, int(video.shape[-2]), int(video.shape[-1])),
                mode="trilinear",
                align_corners=False,
            )
        if video.shape[-2:] != (self.resolution, self.resolution):
            frames = video.permute(0, 2, 1, 3, 4).reshape(
                -1, 3, video.shape[-2], video.shape[-1]
            )
            frames = F.interpolate(
                frames,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
            video = frames.view(video.shape[0], self.clip_length, 3, self.resolution, self.resolution)
            video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    def forward(self, decoded: torch.Tensor) -> torch.Tensor:
        video = self._prepare_video(decoded)
        outputs = []
        for start in range(0, int(video.shape[0]), self.batch_size):
            chunk = video[start : start + self.batch_size]
            output = self.model(chunk, rescale=False, resize=False, return_features=True)
            if isinstance(output, (tuple, list)):
                output = output[0]
            elif isinstance(output, dict):
                output = output.get("features", output.get("x", None))
            if output is None:
                raise TypeError("StyleGAN-V I3D returned no feature tensor")
            if output.ndim > 2:
                output = output.flatten(1)
            outputs.append(output.float())
        result = torch.cat(outputs, dim=0)
        if result.ndim != 2 or result.shape[1] != 400:
            raise ValueError(f"expected StyleGAN-V I3D features [N,400], got {tuple(result.shape)}")
        return result


def load_stage4_reference_stats(path: str | Path, *, device=None) -> dict[str, Any]:
    """Load per-view GT moments saved as ``.pt`` or ``.npz``."""
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location=device or "cpu")
        if not isinstance(payload, dict):
            raise TypeError("Stage4 reference stats must be a mapping")
        result = dict(payload)
    else:
        with np.load(path, allow_pickle=False) as data:
            result = {key: torch.from_numpy(np.array(data[key])) for key in data.files}
    mu = result.get("mu")
    cov = result.get("cov", result.get("sigma"))
    if not isinstance(mu, torch.Tensor) or not isinstance(cov, torch.Tensor):
        raise KeyError("Stage4 reference stats require tensor/array keys 'mu' and 'cov' or 'sigma'")
    if mu.ndim != 2 or cov.ndim != 3 or cov.shape[:2] != (mu.shape[0], mu.shape[1]):
        raise ValueError(f"per-view stats must be mu [NC,D], cov [NC,D,D], got {tuple(mu.shape)}, {tuple(cov.shape)}")
    result["mu"] = mu.to(device=device, dtype=torch.float32)
    result["cov"] = cov.to(device=device, dtype=torch.float32)
    result["view_names"] = result.get("view_names", tuple(f"view_{i}" for i in range(mu.shape[0])))
    return result


def save_stage4_reference_stats(path: str | Path, mu: torch.Tensor, cov: torch.Tensor, *, view_names=None, metadata=None) -> None:
    """Save fixed per-view GT moments without touching training checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mu": mu.detach().cpu().float(),
        "cov": cov.detach().cpu().float(),
        "view_names": tuple(view_names or [f"view_{i}" for i in range(mu.shape[0])]),
        "metadata": dict(metadata or {}),
    }, path)


def _frechet_from_stats(fake_mu, fake_cov, ref_mu, ref_cov, eps, ref_sqrt=None):
    feature_dim = fake_mu.shape[-1]
    eye = torch.eye(feature_dim, device=fake_cov.device, dtype=fake_cov.dtype)
    fake_cov = 0.5 * (fake_cov + fake_cov.transpose(-1, -2)) + eps * eye
    ref_cov = 0.5 * (ref_cov + ref_cov.transpose(-1, -2)) + eps * eye
    if ref_sqrt is None:
        ref_sqrt = torch.stack([_sqrt_psd(matrix, eps) for matrix in ref_cov])
    else:
        ref_sqrt = ref_sqrt.to(device=fake_cov.device, dtype=fake_cov.dtype)
    middle = ref_sqrt.matmul(fake_cov).matmul(ref_sqrt)
    eig = torch.linalg.eigvalsh(0.5 * (middle + middle.transpose(-1, -2))).clamp_min(0).sqrt().sum(-1)
    mean_term = (fake_mu - ref_mu).square().sum(-1)
    return mean_term + fake_cov.diagonal(dim1=-2, dim2=-1).sum(-1) + ref_cov.diagonal(dim1=-2, dim2=-1).sum(-1) - 2.0 * eig


class Stage4PerViewFrechetRGBLoss:
    """Fixed GT moments plus a detached generated queue, independently per view."""

    def __init__(self, reference_mu, reference_cov, queue_size=128, min_population=8, eps=1e-4):
        reference_mu = torch.as_tensor(reference_mu, dtype=torch.float32)
        reference_cov = torch.as_tensor(reference_cov, dtype=torch.float32)
        if reference_mu.ndim != 2 or reference_cov.shape != (reference_mu.shape[0], reference_mu.shape[1], reference_mu.shape[1]):
            raise ValueError("reference_mu must be [NC,D] and reference_cov must be [NC,D,D]")
        self.reference_mu = reference_mu
        self.reference_cov = reference_cov
        self.queue_size = int(queue_size)
        self.min_population = int(min_population)
        self.eps = float(eps)
        feature_dim = int(reference_mu.shape[1])
        eye = torch.eye(feature_dim, device=reference_cov.device, dtype=reference_cov.dtype)
        regularized_reference_cov = (
            0.5 * (reference_cov + reference_cov.transpose(-1, -2)) + self.eps * eye
        )
        with torch.no_grad():
            self.reference_cov_sqrt = torch.stack(
                [_sqrt_psd(matrix, self.eps) for matrix in regularized_reference_cov]
            )
        self.fake_queue = None

    @property
    def num_views(self):
        return int(self.reference_mu.shape[0])

    @torch.no_grad()
    def prefill(self, features: torch.Tensor) -> None:
        # Every rank loads the same immutable queue artifact. Do not all-gather it
        # here, otherwise initialization communicates world_size duplicate copies.
        self.fake_queue = self._append(features.detach(), gather=False)

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        self.fake_queue = self._append(features.detach(), gather=True)

    def _append(self, features, *, gather: bool):
        features = features.detach().float()
        if features.ndim != 3 or features.shape[1:] != self.reference_mu.shape:
            raise ValueError(f"features must be [N,{self.num_views},{self.reference_mu.shape[1]}], got {tuple(features.shape)}")
        if gather and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, features.contiguous())
            features = torch.cat(gathered, dim=0)
        if self.fake_queue is not None:
            features = torch.cat([self.fake_queue.to(features), features], dim=0)
        return features[-self.queue_size:] if self.queue_size > 0 else features[:0]

    def __call__(self, features: torch.Tensor):
        if features.ndim != 3 or features.shape[1:] != self.reference_mu.shape:
            raise ValueError(f"features must be [N,{self.num_views},{self.reference_mu.shape[1]}], got {tuple(features.shape)}")
        gathered = differentiable_all_gather(features.reshape(features.shape[0], -1))
        current = gathered.view(-1, features.shape[1], features.shape[2])
        population = current if self.fake_queue is None else torch.cat([self.fake_queue.to(current), current], dim=0)
        if population.shape[0] < self.min_population:
            return features.sum() * 0.0, False
        mu = population.float().mean(dim=0)
        centered = population.float() - mu
        cov = torch.einsum("nvd,nve->vde", centered, centered) / max(population.shape[0] - 1, 1)
        ref_mu = self.reference_mu.to(device=features.device)
        ref_cov = self.reference_cov.to(device=features.device)
        value = _frechet_from_stats(
            mu,
            cov,
            ref_mu,
            ref_cov,
            self.eps,
            ref_sqrt=self.reference_cov_sqrt,
        ).mean()
        return value, True


class Stage4PerViewFrechetRGBEMALoss:
    """Per-view Fréchet loss with detached historical EMA moments.

    The current global batch keeps its gradient, while all history is stored as
    detached first and second moments.  Each camera is normalized independently
    before the camera losses are averaged.
    """

    def __init__(self, reference_mu, reference_cov, decay=0.999, eps=1e-4, normalization_eps=0.01):
        reference_mu = torch.as_tensor(reference_mu, dtype=torch.float32)
        reference_cov = torch.as_tensor(reference_cov, dtype=torch.float32)
        expected_cov_shape = (reference_mu.shape[0], reference_mu.shape[1], reference_mu.shape[1])
        if reference_mu.ndim != 2 or reference_cov.shape != expected_cov_shape:
            raise ValueError("reference_mu must be [NC,D] and reference_cov must be [NC,D,D]")
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.reference_mu = reference_mu
        self.reference_cov = reference_cov
        self.decay = float(decay)
        self.eps = float(eps)
        self.normalization_eps = float(normalization_eps)
        feature_dim = int(reference_mu.shape[1])
        eye = torch.eye(feature_dim, device=reference_cov.device, dtype=reference_cov.dtype)
        regularized_reference_cov = (
            0.5 * (reference_cov + reference_cov.transpose(-1, -2)) + self.eps * eye
        )
        with torch.no_grad():
            self.reference_cov_sqrt = torch.stack(
                [_sqrt_psd(matrix, self.eps) for matrix in regularized_reference_cov]
            )
        self.fake_mu = None
        self.fake_second_moment = None

    @property
    def num_views(self):
        return int(self.reference_mu.shape[0])

    @torch.no_grad()
    def prefill(self, features: torch.Tensor) -> None:
        features = self._validate_features(features).detach().float()
        self.fake_mu = features.mean(dim=0)
        self.fake_second_moment = torch.einsum("nvd,nve->vde", features, features) / features.shape[0]

    def _validate_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != self.reference_mu.shape:
            raise ValueError(
                f"features must be [N,{self.num_views},{self.reference_mu.shape[1]}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] < 1:
            raise ValueError("features must contain at least one sample")
        return features

    def _current_moments(self, features: torch.Tensor):
        gathered = differentiable_all_gather(features.reshape(features.shape[0], -1))
        current = gathered.view(-1, features.shape[1], features.shape[2]).float()
        batch_mu = current.mean(dim=0)
        batch_second_moment = torch.einsum("nvd,nve->vde", current, current) / current.shape[0]
        if self.fake_mu is None:
            return batch_mu, batch_second_moment
        history_mu = self.fake_mu.to(batch_mu).detach()
        history_second_moment = self.fake_second_moment.to(batch_second_moment).detach()
        mu = self.decay * history_mu + (1.0 - self.decay) * batch_mu
        second_moment = (
            self.decay * history_second_moment
            + (1.0 - self.decay) * batch_second_moment
        )
        return mu, second_moment

    def __call__(self, features: torch.Tensor):
        features = self._validate_features(features)
        mu, second_moment = self._current_moments(features)
        cov = second_moment - torch.einsum("vd,ve->vde", mu, mu)
        raw_per_view = _frechet_from_stats(
            mu,
            cov,
            self.reference_mu.to(device=features.device),
            self.reference_cov.to(device=features.device),
            self.eps,
            ref_sqrt=self.reference_cov_sqrt,
        )
        normalized_per_view = raw_per_view / (raw_per_view.detach() + self.normalization_eps)
        return normalized_per_view.mean(), raw_per_view, normalized_per_view

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        features = self._validate_features(features)
        mu, second_moment = self._current_moments(features)
        self.fake_mu = mu.detach()
        self.fake_second_moment = second_moment.detach()

    def state_dict(self):
        if self.fake_mu is None:
            return {}
        return {
            "fake_mu": self.fake_mu.detach().cpu(),
            "fake_second_moment": self.fake_second_moment.detach().cpu(),
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict) -> None:
        fake_mu = torch.as_tensor(state_dict["fake_mu"], dtype=torch.float32)
        fake_second_moment = torch.as_tensor(state_dict["fake_second_moment"], dtype=torch.float32)
        expected_second_shape = (
            self.reference_mu.shape[0],
            self.reference_mu.shape[1],
            self.reference_mu.shape[1],
        )
        if fake_mu.shape != self.reference_mu.shape or fake_second_moment.shape != expected_second_shape:
            raise ValueError("EMA state shape does not match reference statistics")
        self.fake_mu = fake_mu.to(self.reference_mu.device)
        self.fake_second_moment = fake_second_moment.to(self.reference_mu.device)


def prediction_gradient_surrogate(prediction: torch.Tensor, fd_gradient: torch.Tensor) -> torch.Tensor:
    """Inject a detached latent-space gradient into the original prediction graph."""
    if prediction.shape != fd_gradient.shape:
        raise ValueError("prediction and fd_gradient must have identical shapes")
    return (prediction.float() * fd_gradient.detach().float()).sum()
