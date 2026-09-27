"""Top-level flow-track Gaussian decoder used by Drive2Gauss."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from drive2gauss.models import gaussian_modules as pointforward


class Drive2GaussGaussianDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.appearance_mode = args.appearance_mode
        self.point_model = pointforward.StaticPointForwardModel(
            query_dim=13,
            rgb_feature_dim=32,
            hidden_dim=args.hidden_dim,
            gaussians_per_query=1,
            min_scale_m=args.min_scale_m,
            max_scale_m=args.max_scale_m,
            max_delta_m=args.max_delta_m,
            fusion_type="mlp",
            fusion_depth=args.fusion_depth,
            attention_heads=args.attention_heads,
            appearance_mode=args.appearance_mode,
            color_feature_dim=args.color_feature_dim,
            dedicated_rgb_head=getattr(args, "rgb_head_only", False),
            predict_lifespan=True,
            iterative_refinement_layers=2,
            refinement_hidden_dim=args.refinement_hidden_dim,
            refinement_fusion_type="xattn",
            refinement_attention_heads=args.refinement_attention_heads,
            refinement_turbo_only=True,
            refinement_step_m=args.refinement_step_m,
            flow_track_refinement=True,
            learned_velocity=(
                getattr(args, "learned_velocity", False)
                or getattr(args, "learned_dynamic_velocity", False)
            ),
            dynamic_velocity_only=getattr(args, "learned_dynamic_velocity", False),
            learned_dynamic_separation=(
                getattr(args, "learned_dynamic_separation", False)
                or getattr(args, "learned_dynamic_velocity", False)
            ),
            max_velocity_m=getattr(args, "max_velocity_m", 3.0),
            object_slots=getattr(args, "object_slots", 0),
            object_slot_frames=getattr(args, "object_slot_frames", 4),
        )
        if args.appearance_mode == "feature_unet":
            self.appearance_decoder = pointforward.FeatureRenderUNet(
                args.color_feature_dim,
                args.unet_base_channels,
                num_cameras=len(pointforward.VIEW_NAMES),
                camera_embedding_dim=args.camera_embedding_dim,
                depth=getattr(args, "unet_depth", 1),
            )
        else:
            self.appearance_decoder = None
            if args.single_view_train:
                # Single-camera training needs no per-camera affine: every Gaussian
                # is supervised by exactly one camera, so its color is already in
                # that camera's photometry.
                self.camera_affine = None
            else:
                self.camera_affine = pointforward.PerCameraAffine(len(pointforward.VIEW_NAMES))
        inactive_modules = [
            self.point_model.query_encoder,
            self.point_model.rgb_encoder,
            self.point_model.depth_encoder,
            self.point_model.observation_encoder,
            self.point_model.temporal_encoder,
            self.point_model.weight_mlp,
            self.point_model.fusion_input,
            self.point_model.fusion_blocks,
        ]
        self.backbone_modules = tuple(inactive_modules)
        for refinement_block in self.point_model.refinement_blocks:
            if refinement_block.cross_attention is not None:
                inactive_modules.append(refinement_block.weight_mlp)
        if args.single_view_train and args.init_checkpoint is None:
            # Fresh single-view run: train everything from scratch so the RGB head
            # shows the representation's ceiling on a tiny overfit set.
            for module in inactive_modules:
                module.requires_grad_(True)
        else:
            for module in inactive_modules:
                module.requires_grad_(False)
        if getattr(args, "unfreeze_backbone", False):
            if args.init_checkpoint is None:
                raise ValueError("--unfreeze-backbone requires --init-checkpoint")
            for module in self.backbone_modules:
                module.requires_grad_(True)
        if getattr(args, "rgb_head_only", False):
            if args.appearance_mode != "direct_rgb":
                raise ValueError("--rgb-head-only requires --appearance-mode direct_rgb")
            for parameter in self.point_model.parameters():
                parameter.requires_grad_(False)
            if not self.point_model.dedicated_rgb_head:
                raise RuntimeError("Dedicated RGB head was not constructed")
            for parameter in self.point_model.rgb_head.parameters():
                parameter.requires_grad_(True)

    def predict_gaussians(self, sample: WindowSample, args: argparse.Namespace) -> dict:
        batch = sample.batch
        self.point_model.time_origin = float(getattr(args, "time_origin", 0.0))
        self.point_model.time_denominator = float(getattr(args, "time_denominator", 1.0))
        return self.point_model(
            batch.query_features,
            batch.anchors_ref,
            batch.sampled_rgb_features,
            batch.depth_difference,
            batch.same_time,
            batch.query_time,
            batch.observation_valid,
            context_feature_maps=batch.context_feature_maps,
            context_viewmats=sample.context_viewmats,
            context_intrinsics=sample.context_intrinsics,
            image_width=args.width,
            image_height=args.height,
            track_anchors_ref=batch.track_anchors_ref,
            track_valid=batch.track_valid,
            context_frames=sample.context_frames,
            context_time=batch.context_time,
            flow_displacement_prior=sample.flow_displacement_prior,
        )

    def init_rgb_bias_from_observations(self, sample: WindowSample) -> None:
        """Bias the direct_rgb appearance head toward the observed query RGB mean.

        The direct_rgb head emits sigmoid(RGB); starting its bias at logit(mean
        observed color) makes the initial render close to the data so the fine
        colors refine from a sensible base instead of mid-gray.
        """
        if self.appearance_mode != "direct_rgb":
            return
        observed = sample.batch.query_features[:, 3:6].clamp(1.0e-3, 1.0 - 1.0e-3).mean(dim=0)
        logit = torch.log(observed / (1.0 - observed))
        with torch.no_grad():
            if self.point_model.dedicated_rgb_head:
                self.point_model.rgb_head[-1].bias.copy_(logit)
            else:
                self.point_model.head[-1].bias[11:14].copy_(logit)

    def forward(
        self,
        sample: WindowSample,
        target_frame: int,
        target_view: int,
        args: argparse.Namespace,
    ) -> tuple[torch.Tensor, dict]:
        gaussian = self.predict_gaussians(sample, args)
        rendered = pointforward.render_output(
            gaussian,
            self.appearance_decoder,
            sample.clip,
            target_frame,
            target_view,
            args.height,
            args.width,
            args,
            sample.batch.anchors_ref.device,
            camera_affine=getattr(self, "camera_affine", None),
        )
        return rendered, gaussian


# Backward-compatible class name used by existing checkpoints and scripts.
FlowTrackRenderModel = Drive2GaussGaussianDecoder
