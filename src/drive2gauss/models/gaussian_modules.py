"""Drive2Gauss PointForward Gaussian decoder architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VIEW_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

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


from drive2gauss.rendering.gsplat_renderer import (
    camera_for_target,
    means_at_frame,
    opacities_at_frame,
    render_appearance_features,
    render_gsplat,
    render_output,
)
