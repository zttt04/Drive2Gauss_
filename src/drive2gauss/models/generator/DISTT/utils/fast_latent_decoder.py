from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def _group_count(channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _run_with_optional_checkpoint(module: nn.Module, x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and torch.is_grad_enabled() and x.requires_grad:
        return activation_checkpoint(module, x, use_reentrant=False)
    return module(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _group_count(channels, groups)
        self.net = nn.Sequential(
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _group_count(channels, groups)
        self.net = nn.Sequential(
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialUpsample3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        residual_blocks: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            nn.SiLU(),
        ]
        layers.extend(ResidualBlock3D(out_channels, groups=groups) for _ in range(residual_blocks))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames, height, width = x.shape[-3:]
        x = F.interpolate(
            x,
            size=(frames, height * 2, width * 2),
            mode="trilinear",
            align_corners=False,
        )
        return self.proj(x)


class FastLatentVideoDecoder(nn.Module):
    """Decode CogVideoX video latents into dense supervision maps.

    The module keeps temporal reasoning at latent resolution and only uses 2D
    layers after folding frames into the batch dimension. This is intended for
    differentiable geometry losses, not high-fidelity visual reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 1,
        hidden_channels: int = 128,
        temporal_blocks: int = 3,
        spatial_channels: tuple[int, ...] = (96, 64, 32),
        spatial_frame_chunk_size: int = 16,
        groups: int = 8,
        final_tanh: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.final_tanh = final_tanh
        self.spatial_frame_chunk_size = spatial_frame_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.in_proj = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.temporal_net = nn.Sequential(
            *[ResidualBlock3D(hidden_channels, groups=groups) for _ in range(temporal_blocks)]
        )

        up_layers: list[nn.Module] = []
        current_channels = hidden_channels
        for next_channels in spatial_channels:
            up_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_group_count(next_channels, groups), next_channels),
                    nn.SiLU(),
                    ResidualBlock2D(next_channels, groups=groups),
                ]
            )
            current_channels = next_channels
        self.spatial_net = nn.Sequential(*up_layers)
        self.out_proj = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z: torch.Tensor,
        out_frames: int = 17,
        out_size: tuple[int, int] = (424, 800),
    ) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(f"Expected latent shape B,C,T,H,W, got {tuple(z.shape)}")

        x = self.in_proj(z)
        x = _run_with_optional_checkpoint(self.temporal_net, x, self.gradient_checkpointing)
        x = F.interpolate(
            x,
            size=(out_frames, z.shape[-2], z.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )

        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        if self.spatial_frame_chunk_size is None or self.spatial_frame_chunk_size <= 0:
            chunks = [x]
        else:
            chunks = list(x.split(self.spatial_frame_chunk_size, dim=0))
        decoded_chunks = []
        for chunk in chunks:
            chunk = _run_with_optional_checkpoint(self.spatial_net, chunk, self.gradient_checkpointing)
            if chunk.shape[-2:] != out_size:
                chunk = F.interpolate(chunk, size=out_size, mode="bilinear", align_corners=False)
            decoded_chunks.append(self.out_proj(chunk))
        x = torch.cat(decoded_chunks, dim=0)
        if self.final_tanh:
            x = torch.tanh(x)
        return x.reshape(batch, frames, -1, out_size[0], out_size[1]).permute(0, 2, 1, 3, 4)


class TemporalIntervalFastLatentVideoDecoder(nn.Module):
    """A faster decoder with learnable interval-wise temporal expansion.

    CogVideoX latents use a temporal compression ratio of 4, so Tz latent
    frames represent `(Tz - 1) * 4 + 1` video frames. This decoder explicitly
    predicts the four phases between adjacent latent frames instead of using a
    single trilinear temporal interpolation.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 1,
        hidden_channels: int = 128,
        temporal_blocks: int = 3,
        spatial_channels: tuple[int, ...] = (96, 64, 32),
        spatial_frame_chunk_size: int = 16,
        groups: int = 8,
        final_tanh: bool = True,
        temporal_upsample_factor: int = 4,
        temporal_refine_blocks: int = 2,
        spatial_3d_blocks: int = 1,
        spatial_2d_blocks: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if temporal_upsample_factor < 1:
            raise ValueError("temporal_upsample_factor must be >= 1")
        self.final_tanh = final_tanh
        self.spatial_frame_chunk_size = spatial_frame_chunk_size
        self.temporal_upsample_factor = temporal_upsample_factor
        self.gradient_checkpointing = gradient_checkpointing

        self.in_proj = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.latent_temporal_net = nn.Sequential(
            *[ResidualBlock3D(hidden_channels, groups=groups) for _ in range(temporal_blocks)]
        )
        self.phase_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=1),
                    nn.GroupNorm(_group_count(hidden_channels, groups), hidden_channels),
                    nn.SiLU(),
                    ResidualBlock3D(hidden_channels, groups=groups),
                )
                for _ in range(temporal_upsample_factor)
            ]
        )
        self.phase_embeddings = nn.Parameter(torch.zeros(temporal_upsample_factor, hidden_channels))
        self.final_frame_proj = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(_group_count(hidden_channels, groups), hidden_channels),
            nn.SiLU(),
            ResidualBlock3D(hidden_channels, groups=groups),
        )
        self.temporal_refine = nn.Sequential(
            *[ResidualBlock3D(hidden_channels, groups=groups) for _ in range(temporal_refine_blocks)]
        )

        spatial3d_channels = spatial_channels[:2]
        spatial2d_channels = spatial_channels[2:] or spatial_channels[-1:]
        current_channels = hidden_channels
        self.spatial3d_net = nn.ModuleList()
        for next_channels in spatial3d_channels:
            self.spatial3d_net.append(
                SpatialUpsample3DBlock(
                    current_channels,
                    next_channels,
                    groups=groups,
                    residual_blocks=spatial_3d_blocks,
                )
            )
            current_channels = next_channels

        up_layers: list[nn.Module] = []
        for next_channels in spatial2d_channels:
            up_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_group_count(next_channels, groups), next_channels),
                    nn.SiLU(),
                ]
            )
            up_layers.extend(ResidualBlock2D(next_channels, groups=groups) for _ in range(spatial_2d_blocks))
            current_channels = next_channels
        self.spatial2d_net = nn.Sequential(*up_layers)
        self.out_proj = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def _temporal_expand(self, x: torch.Tensor, out_frames: int) -> torch.Tensor:
        batch, channels, latent_frames, height, width = x.shape
        if latent_frames == 1:
            x = self.final_frame_proj(x)
            if out_frames != 1:
                x = x.expand(batch, channels, out_frames, height, width)
            return x

        left = x[:, :, :-1]
        right = x[:, :, 1:]
        pair = torch.cat([left, right], dim=1)
        phase_features = []
        for phase_idx, phase_mixer in enumerate(self.phase_mixers):
            phase_feature = _run_with_optional_checkpoint(phase_mixer, pair, self.gradient_checkpointing)
            phase_embedding = self.phase_embeddings[phase_idx].view(1, channels, 1, 1, 1)
            phase_features.append(phase_feature + phase_embedding)
        x = torch.stack(phase_features, dim=3)
        x = x.reshape(batch, channels, (latent_frames - 1) * self.temporal_upsample_factor, height, width)
        final_frame = self.final_frame_proj(right[:, :, -1:])
        x = torch.cat([x, final_frame], dim=2)
        if x.shape[2] != out_frames:
            x = F.interpolate(
                x,
                size=(out_frames, height, width),
                mode="nearest",
            )
        return x

    def forward(
        self,
        z: torch.Tensor,
        out_frames: int = 17,
        out_size: tuple[int, int] = (424, 800),
    ) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(f"Expected latent shape B,C,T,H,W, got {tuple(z.shape)}")

        x = self.in_proj(z)
        x = _run_with_optional_checkpoint(self.latent_temporal_net, x, self.gradient_checkpointing)
        x = self._temporal_expand(x, out_frames)
        x = _run_with_optional_checkpoint(self.temporal_refine, x, self.gradient_checkpointing)

        for block in self.spatial3d_net:
            x = _run_with_optional_checkpoint(block, x, self.gradient_checkpointing)

        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        if self.spatial_frame_chunk_size is None or self.spatial_frame_chunk_size <= 0:
            chunks = [x]
        else:
            chunks = list(x.split(self.spatial_frame_chunk_size, dim=0))
        decoded_chunks = []
        for chunk in chunks:
            chunk = _run_with_optional_checkpoint(self.spatial2d_net, chunk, self.gradient_checkpointing)
            if chunk.shape[-2:] != out_size:
                chunk = F.interpolate(chunk, size=out_size, mode="bilinear", align_corners=False)
            decoded_chunks.append(self.out_proj(chunk))
        x = torch.cat(decoded_chunks, dim=0)
        if self.final_tanh:
            x = torch.tanh(x)
        return x.reshape(batch, frames, -1, out_size[0], out_size[1]).permute(0, 2, 1, 3, 4)


def _decoder_kwargs(kwargs: dict) -> tuple[str, dict]:
    kwargs = dict(kwargs)
    architecture = kwargs.pop("decoder_architecture", "framewise_2d")
    if architecture in ("framewise", "legacy"):
        architecture = "framewise_2d"
    return architecture, kwargs


def build_fast_depth_decoder(**kwargs) -> nn.Module:
    architecture, kwargs = _decoder_kwargs(kwargs)
    if architecture == "framewise_2d":
        kwargs.pop("temporal_upsample_factor", None)
        kwargs.pop("temporal_refine_blocks", None)
        kwargs.pop("spatial_3d_blocks", None)
        kwargs.pop("spatial_2d_blocks", None)
        return FastLatentVideoDecoder(out_channels=1, **kwargs)
    if architecture == "temporal_interval":
        return TemporalIntervalFastLatentVideoDecoder(out_channels=1, **kwargs)
    raise ValueError(f"Unknown decoder architecture: {architecture}")


def build_fast_flow_rgb_decoder(**kwargs) -> nn.Module:
    architecture, kwargs = _decoder_kwargs(kwargs)
    if architecture == "framewise_2d":
        kwargs.pop("temporal_upsample_factor", None)
        kwargs.pop("temporal_refine_blocks", None)
        kwargs.pop("spatial_3d_blocks", None)
        kwargs.pop("spatial_2d_blocks", None)
        return FastLatentVideoDecoder(out_channels=3, **kwargs)
    if architecture == "temporal_interval":
        return TemporalIntervalFastLatentVideoDecoder(out_channels=3, **kwargs)
    raise ValueError(f"Unknown decoder architecture: {architecture}")
