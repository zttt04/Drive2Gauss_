#!/usr/bin/env python3
"""Train the validated flow-track PointForward model on 4-frame multiscene windows with DDP."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from drive2gauss.data import feature_cache
from drive2gauss.data import manifest as dataset_manifest
from drive2gauss.data import motion_release
from drive2gauss.data import query_dataset as query_data
from drive2gauss.data import query_sampling as stage1
from drive2gauss.training import static_decoder_pipeline as pointforward


REPO_ROOT = Path(__file__).resolve().parents[3]
from drive2gauss.models.gaussian_decoder import Drive2GaussGaussianDecoder  # noqa: E402

FlowTrackRenderModel = Drive2GaussGaussianDecoder


WINDOW_STARTS = [0, 4, 8, 12]
DEFAULT_VAL_SCENES = [211, 434, 528, 656, 699]
TEMPORAL_CONTEXT_RECORDS_PER_CLIP = {
    "contiguous": len(WINDOW_STARTS),
    "fixed_window_manifest": 1,
    "mixed_sparse2_contiguous1": 3,
    "contiguous4_neighbor2_sparse4_stride2": 2,
    "contiguous4_local8": 1,
    "sparse4_stride2_local8": 1,
    "sparse4_stride4_supervise8": 1,
    "sparse_stride2_fullclip": 1,
    "alternating_sparse4_stride4_sparse6_stride2": 1,
}


@dataclass(frozen=True)
class WindowRecord:
    row: dict
    window_start: int
    context_frames: tuple[int, ...] | None = None
    context_mode: str = "contiguous"
    target_frame: int | None = None
    target_scope: tuple[int, ...] | None = None
    target_weight: float = 1.0


@dataclass
class WindowSample:
    batch: pointforward.Stage2Batch
    clip: dict
    context_frames: torch.Tensor
    context_viewmats: torch.Tensor
    context_intrinsics: torch.Tensor
    frames: list[int]
    views: list[int]
    dynamic_probability_mean: float
    flow_displacement_prior: torch.Tensor | None = None


class PackagedQuerySource:
    def load_clip(self, row: dict, view_indices: list[int]) -> dict:
        del view_indices
        return torch.load(feature_cache.source_path(row), map_location="cpu", weights_only=False)

    @staticmethod
    def target_rgb(clip: dict, frame: int, view: int) -> np.ndarray:
        return pointforward.as_uint8_rgb(clip["rgb_target"][frame, view].numpy())


class OnlineQuerySource:
    """Build the small Query payload from annotations and compressed flow on demand."""

    def __init__(
        self,
        ann_file: Path,
        data_root: Path,
        flow_rgb_root: Path | None,
        flow_index_path: Path | None,
        width: int,
        height: int,
        allow_missing_flow_rgb: bool,
        zero_flow_input: bool,
        motion_data: motion_release.MotionRelease | None = None,
    ) -> None:
        annotation = query_data.load_ann(ann_file)
        self.infos = annotation["infos"] if isinstance(annotation, dict) else annotation
        self.token_to_info_index = {
            str(info["token"]): index for index, info in enumerate(self.infos)
        }
        self.data_root = data_root
        self.flow_rgb_root = flow_rgb_root
        self.width = width
        self.height = height
        self.allow_missing_flow_rgb = allow_missing_flow_rgb
        self.zero_flow_input = zero_flow_input
        self.motion_data = motion_data
        if motion_data is None:
            if flow_index_path is None or flow_rgb_root is None:
                raise ValueError("Legacy flow loading requires both flow root and flow index")
            flow_index = query_data.load_json(flow_index_path)
            self.flow_scenes = flow_index["scenes"]
            first_view = query_data.VIEW_ORDER[0]
            self.token_to_flow_scene = {
                str(token): str(scene_index)
                for scene_index, scene in self.flow_scenes.items()
                for token in scene.get(first_view, {})
            }
        else:
            self.flow_scenes = {}
            self.token_to_flow_scene = {}

    def _frame_infos(self, row: dict) -> list[dict]:
        token = str(row["token"])
        if token not in self.token_to_info_index:
            raise KeyError(f"Could not find clip token {token} in annotation infos")
        start = self.token_to_info_index[token]
        video_length = int(row.get("video_length", 17))
        stop = start + video_length
        if stop > len(self.infos):
            raise IndexError(
                f"Annotation frame range {start}..{stop - 1} exceeds {len(self.infos)} infos"
            )
        frame_infos = self.infos[start:stop]
        if str(frame_infos[0]["token"]) != token:
            raise RuntimeError(f"Annotation clip starts at {frame_infos[0]['token']}, expected {token}")
        return frame_infos

    def _read_flow_rgb(
        self,
        row: dict,
        clip_tokens: list[str],
        view_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count = len(clip_tokens)
        flow_rgb = np.full(
            (frame_count, len(query_data.VIEW_ORDER), self.height, self.width, 3),
            255,
            dtype=np.uint8,
        )
        flow_valid = np.zeros(
            (frame_count, len(query_data.VIEW_ORDER), self.height, self.width),
            dtype=np.uint8,
        )
        generated_root_value = row.get("generated_flow_rgb_root")
        if generated_root_value is not None:
            generated_root = Path(generated_root_value)
            token_root = (
                generated_root / str(row["token"])
                if (generated_root / str(row["token"])).is_dir()
                else generated_root
            )
            for frame_index in range(frame_count):
                for view_index in view_indices:
                    camera = query_data.VIEW_ORDER[view_index]
                    image_path = token_root / camera / f"{frame_index}.jpg"
                    bgr = query_data.cv2.imread(str(image_path), query_data.cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise FileNotFoundError(image_path)
                    rgb = query_data.cv2.cvtColor(bgr, query_data.cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (self.height, self.width):
                        rgb = query_data.cv2.resize(
                            rgb,
                            (self.width, self.height),
                            interpolation=query_data.cv2.INTER_AREA,
                        )
                    flow_rgb[frame_index, view_index] = rgb
                    flow_valid[frame_index, view_index] = 1
            return torch.from_numpy(flow_rgb), torch.from_numpy(flow_valid)
        if self.motion_data is not None:
            for frame_index, (source_token, target_token) in enumerate(
                zip(clip_tokens[:-1], clip_tokens[1:])
            ):
                for view_index in view_indices:
                    camera = query_data.VIEW_ORDER[view_index]
                    try:
                        rgb, valid = self.motion_data.load_masked_flow_rgb(
                            source_token, target_token, camera
                        )
                    except (KeyError, FileNotFoundError):
                        if self.allow_missing_flow_rgb:
                            continue
                        raise
                    if rgb.shape[:2] != (self.height, self.width):
                        rgb = query_data.cv2.resize(
                            rgb, (self.width, self.height), interpolation=query_data.cv2.INTER_AREA
                        )
                        valid = query_data.cv2.resize(
                            valid, (self.width, self.height), interpolation=query_data.cv2.INTER_NEAREST
                        )
                    flow_rgb[frame_index, view_index] = rgb
                    flow_valid[frame_index, view_index] = valid
            return torch.from_numpy(flow_rgb), torch.from_numpy(flow_valid)
        flow_scene_index = self.token_to_flow_scene.get(clip_tokens[0])
        if flow_scene_index is None:
            if self.allow_missing_flow_rgb:
                return torch.from_numpy(flow_rgb), torch.from_numpy(flow_valid)
            raise KeyError(f"No masked-flow scene contains clip token {clip_tokens[0]}")
        scene = self.flow_scenes[flow_scene_index]
        for frame_index, (source_token, target_token) in enumerate(
            zip(clip_tokens[:-1], clip_tokens[1:])
        ):
            for view_index in view_indices:
                camera = query_data.VIEW_ORDER[view_index]
                edge = scene.get(camera, {}).get(source_token)
                if edge is None:
                    if self.allow_missing_flow_rgb:
                        continue
                    raise KeyError(
                        f"Missing masked-flow edge for token={source_token}, camera={camera}"
                    )
                flow_frame_index, indexed_target_token = int(edge[0]), str(edge[1])
                if indexed_target_token != target_token:
                    raise RuntimeError(
                        f"Masked-flow target mismatch for source={source_token}, camera={camera}: "
                        f"index target={indexed_target_token}, expected={target_token}"
                    )
                image_path = (
                    self.flow_rgb_root
                    / f"{flow_scene_index}_scene"
                    / camera
                    / "dynamic_flow_gray_bg"
                    / f"dynamic_flow_gray_bg_{flow_frame_index:06d}.png"
                )
                bgr = query_data.cv2.imread(str(image_path), query_data.cv2.IMREAD_COLOR)
                if bgr is None:
                    if self.allow_missing_flow_rgb:
                        continue
                    raise FileNotFoundError(image_path)
                rgb = query_data.cv2.cvtColor(bgr, query_data.cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (self.height, self.width):
                    rgb = query_data.cv2.resize(
                        rgb, (self.width, self.height), interpolation=query_data.cv2.INTER_AREA
                    )
                flow_rgb[frame_index, view_index] = rgb
                flow_valid[frame_index, view_index] = 1
        return torch.from_numpy(flow_rgb), torch.from_numpy(flow_valid)

    def load_clip(self, row: dict, view_indices: list[int]) -> dict:
        frame_infos = self._frame_infos(row)
        frame_count = len(frame_infos)
        view_count = len(query_data.VIEW_ORDER)
        camera_intrinsics = np.empty((frame_count, view_count, 3, 3), dtype=np.float32)
        camera2lidar = np.empty((frame_count, view_count, 4, 4), dtype=np.float32)
        lidar2camera = np.empty_like(camera2lidar)
        lidar2global = np.empty((frame_count, 4, 4), dtype=np.float32)
        rgb_paths: list[list[str]] = []
        for frame_index, frame_info in enumerate(frame_infos):
            lidar2global[frame_index] = query_data.lidar_to_global(frame_info)
            frame_rgb_paths = []
            for view_index, camera in enumerate(query_data.VIEW_ORDER):
                camera_info = frame_info["cams"][camera]
                camera_intrinsics[frame_index, view_index] = query_data.resized_intrinsics(
                    camera_info, self.width, self.height
                )
                camera2lidar[frame_index, view_index] = query_data.camera_to_lidar(camera_info)
                lidar2camera[frame_index, view_index] = np.linalg.inv(
                    camera2lidar[frame_index, view_index]
                ).astype(np.float32)
                frame_rgb_paths.append(
                    str(query_data.resolve_data_path(camera_info["data_path"], self.data_root))
                )
            rgb_paths.append(frame_rgb_paths)
        ref_from_global = np.linalg.inv(lidar2global[0]).astype(np.float32)
        frame_to_ref_lidar = np.einsum(
            "ij,tjk->tik", ref_from_global, lidar2global
        ).astype(np.float32)
        clip_tokens = [str(info["token"]) for info in frame_infos]
        flow_rgb_target, flow_rgb_valid_target = self._read_flow_rgb(
            row, clip_tokens, view_indices
        )
        if self.zero_flow_input:
            # The RAFT/Middlebury encoding represents zero flow as white.
            # Keep the original validity mask; only replace the model input.
            flow_rgb_target = torch.full_like(flow_rgb_target, 255)
        return {
            "video_length": frame_count,
            "camera_intrinsics": torch.from_numpy(camera_intrinsics),
            "camera2lidar": torch.from_numpy(camera2lidar),
            "lidar2camera": torch.from_numpy(lidar2camera),
            "frame_to_ref_lidar": torch.from_numpy(frame_to_ref_lidar),
            "flow_rgb_target": flow_rgb_target,
            "flow_rgb_valid_target": flow_rgb_valid_target,
            "_rgb_paths": rgb_paths,
            "_rgb_cache": {},
            "_token": str(row["token"]),
            "_generated_rgb_root": row.get("generated_rgb_root"),
            "_target_rgb_source": (
                "generated_rgb" if row.get("generated_rgb_root") is not None else "ground_truth_rgb"
            ),
        }

    def target_rgb(self, clip: dict, frame: int, view: int) -> np.ndarray:
        key = (frame, view)
        if key not in clip["_rgb_cache"]:
            generated_root_value = clip.get("_generated_rgb_root")
            if generated_root_value is None:
                image_path = Path(clip["_rgb_paths"][frame][view])
            else:
                generated_root = Path(generated_root_value)
                token_root = (
                    generated_root / clip["_token"]
                    if (generated_root / clip["_token"]).is_dir()
                    else generated_root
                )
                image_path = token_root / query_data.VIEW_ORDER[view] / f"{frame}.jpg"
            bgr = query_data.cv2.imread(str(image_path), query_data.cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            rgb = query_data.cv2.cvtColor(bgr, query_data.cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (self.height, self.width):
                rgb = query_data.cv2.resize(
                    rgb, (self.width, self.height), interpolation=query_data.cv2.INTER_AREA
                )
            clip["_rgb_cache"][key] = rgb.astype(np.uint8)
        return clip["_rgb_cache"][key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Legacy packaged Query manifest containing both train and validation scenes.",
    )
    parser.add_argument("--cache-root", type=Path, default=None, help="Legacy shared Turbo cache root.")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--train-cache-root", type=Path, default=None)
    parser.add_argument("--val-cache-root", type=Path, default=None)
    parser.add_argument(
        "--include-val-in-train",
        action="store_true",
        help="Intentionally add the validation manifest to the training schedule. "
        "Rows keep their validation cache and online-Query source; validation is "
        "still reported as a deliberate leakage/overfit measurement.",
    )
    parser.add_argument(
        "--online-query",
        action="store_true",
        help="Build camera/flow Query payloads online instead of loading packaged clip.pt files.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-ann-file", type=Path, default=None)
    parser.add_argument("--val-ann-file", type=Path, default=None)
    parser.add_argument("--train-masked-flow-rgb-root", type=Path, default=None)
    parser.add_argument("--val-masked-flow-rgb-root", type=Path, default=None)
    parser.add_argument("--train-masked-flow-index", type=Path, default=None)
    parser.add_argument("--val-masked-flow-index", type=Path, default=None)
    parser.add_argument(
        "--motion-release-root",
        type=Path,
        default=None,
        help="Portable release root containing manifest.jsonl plus relative flow/mask paths.",
    )
    parser.add_argument("--motion-release-manifest", type=Path, default=None)
    parser.add_argument("--train-motion-release-root", type=Path, default=None)
    parser.add_argument("--val-motion-release-root", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-flow-rgb",
        action="store_true",
        help="Keep missing flow edges invalid and continue instead of failing.",
    )
    parser.add_argument(
        "--zero-flow-input",
        action="store_true",
        help="Replace flow RGB with white zero-flow after loading it, while keeping "
        "the original flow-validity mask and all downstream processing unchanged.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model tensors whose names and shapes match, then create a "
        "new optimizer. A matching feature_unet checkpoint loads the complete model; "
        "shape-mismatched heads are skipped for architecture changes.",
    )
    parser.add_argument(
        "--reset-motion-residual-heads-on-init",
        action="store_true",
        help="After loading --init-checkpoint, zero the learned dynamic-logit and "
        "velocity residual output layers. This starts the gated-motion fine-tune "
        "from the flow trajectory while preserving the loaded backbone.",
    )
    parser.add_argument("--appearance-mode", choices=["feature_unet", "direct_rgb"], default="feature_unet")
    parser.add_argument(
        "--rgb-head-only",
        action="store_true",
        help="For direct_rgb, freeze the loaded PointForward geometry/backbone and train only the dedicated per-Gaussian RGB head.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Jointly fine-tune the query/observation/temporal/fusion backbone from --init-checkpoint.",
    )
    parser.add_argument(
        "--single-view-train",
        action="store_true",
        help="Use one camera for both query construction and RGB supervision.",
    )
    parser.add_argument(
        "--single-view-name",
        choices=pointforward.VIEW_NAMES,
        default="CAM_FRONT",
        help="Camera used when --single-view-train is enabled.",
    )
    parser.add_argument(
        "--context-frame-count",
        type=int,
        default=4,
        help="Number of consecutive input frames in each prediction window.",
    )
    parser.add_argument(
        "--val-scenes",
        type=int,
        nargs="+",
        default=DEFAULT_VAL_SCENES,
        help="Held-out scenes for legacy combined manifests; separate val manifests use every row.",
    )
    parser.add_argument(
        "--train-scenes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit train scene list (subset of manifest scenes excluding val scenes).",
    )
    parser.add_argument(
        "--limit-train-scenes",
        type=int,
        default=0,
        help="Keep only the first N train scenes (all their clips). 0 keeps all train scenes.",
    )
    parser.add_argument(
        "--fixed-val-manifest-index",
        type=int,
        default=None,
        help="Legacy zero-based position of one fixed validation row.",
    )
    parser.add_argument(
        "--fixed-val-manifest-indices",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based positions of fixed validation rows evaluated every validation epoch.",
    )
    parser.add_argument("--fixed-val-window-start", type=int, default=12)
    parser.add_argument(
        "--temporal-context-mode",
        choices=[
            "contiguous",
            "fixed_window_manifest",
            "mixed_sparse2_contiguous1",
            "contiguous4_neighbor2_sparse4_stride2",
            "contiguous4_local8",
            "sparse4_stride2_local8",
            "sparse4_stride4_supervise8",
            "sparse_stride2_fullclip",
            "alternating_sparse4_stride4_sparse6_stride2",
        ],
        default="contiguous",
        help="Training context schedule. mixed_sparse2_contiguous1 emits exactly two "
        "four-frame sparse contexts and one contiguous four-frame context per clip/epoch. "
        "contiguous4_neighbor2_sparse4_stride2 emits one contiguous and one stride-2 "
        "four-frame context per clip/epoch, with local eight-frame supervision. "
        "contiguous4_local8 emits one contiguous four-frame context inside an exact "
        "eight-frame target span per clip/epoch. sparse4_stride2_local8 emits four "
        "inputs at stride 2 inside the same eight-frame target span. "
        "sparse4_stride4_supervise8 emits inputs at stride 4 and eight targets "
        "sampled every two frames. sparse_stride2_fullclip emits a configurable "
        "stride-2 context and supervises targets from the entire clip. "
        "alternating_sparse4_stride4_sparse6_stride2 alternates four stride-4 "
        "inputs with six stride-2 inputs; the six-frame step samples targets "
        "from its local continuous span.",
    )
    parser.add_argument(
        "--sparse-context-frame-count",
        type=int,
        default=6,
        help="Input-frame count for sparse_stride2_fullclip; frames use stride 2.",
    )
    parser.add_argument(
        "--full-clip-targets-per-step",
        type=int,
        default=0,
        help="Targets drawn from a deterministically shuffled full target scope each step; "
        "blocks cover the scope before reshuffling. 0 supervises the entire scope.",
    )
    parser.add_argument(
        "--neighbor-supervision-weight",
        type=float,
        default=0.25,
        help="Image-loss multiplier for the two neighboring frames around a contiguous context.",
    )
    parser.add_argument(
        "--full-clip-target-supervision",
        action="store_true",
        help="Draw each training target from the full clip rather than only the four "
        "context frames. Targets rotate deterministically over the mode-specific target scope.",
    )
    parser.add_argument(
        "--limit-train-clips",
        type=int,
        default=0,
        help="Bound train clips for infrastructure checks; 0 uses all 585 train clips.",
    )
    parser.add_argument(
        "--train-manifest-indices",
        type=int,
        nargs="+",
        default=None,
        help="Use explicit train manifest rows for a reproducible bounded smoke test.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Optional bounded step override for infrastructure checks; 0 trains by --epochs.",
    )
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1.0e-5,
        help="Learning rate for the unfrozen query/observation/temporal/fusion backbone.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lpips-weight", type=float, default=0.2)
    parser.add_argument("--lpips-module-root", type=Path, default=None)
    parser.add_argument("--opacity-reg-weight", type=float, default=0.001)
    parser.add_argument("--scale-reg-weight", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--refinement-hidden-dim", type=int, default=64)
    parser.add_argument("--refinement-attention-heads", type=int, default=8)
    parser.add_argument("--refinement-step-m", type=float, default=0.5)
    parser.add_argument(
        "--learned-velocity", action="store_true",
        help="Predict one learned 3D velocity per Gaussian; flow tracks only initialize the base position.",
    )
    parser.add_argument(
        "--learned-dynamic-velocity", action="store_true",
        help="Enable learned velocity only for queries marked dynamic by the flow-derived probability; implies --learned-velocity.",
    )
    parser.add_argument(
        "--learned-dynamic-separation", action="store_true",
        help="Learn a residual dynamic probability on top of the flow probability.",
    )
    parser.add_argument("--max-velocity-m", type=float, default=3.0)
    parser.add_argument(
        "--object-slots",
        type=int,
        default=0,
        help="Enable object-centric dynamic head: group dynamic queries into this many "
        "instance-id slots (0 = disabled, current per-point flow-track behavior).",
    )
    parser.add_argument("--object-slot-frames", type=int, default=4)
    parser.add_argument(
        "--extra-edge-queries-per-frame-view",
        type=int,
        default=0,
        help="Add this many structural (Sobel edge) queries per frame-view on top "
        "of the base uniform pool, targeting thin structures (poles, signs). "
        "The total per-sample query budget becomes "
        "base (num_queries_per_frame_view * frame_views) + "
        "extra_edge_queries_per_frame_view * frame_views.",
    )
    parser.add_argument("--color-feature-dim", type=int, default=128)
    parser.add_argument("--unet-base-channels", type=int, default=32)
    parser.add_argument(
        "--unet-depth",
        type=int,
        default=1,
        help="Number of decoder refinement blocks in FeatureRenderUNet; 1 preserves old checkpoints.",
    )
    parser.add_argument("--camera-embedding-dim", type=int, default=16)
    parser.add_argument("--min-scale-m", type=float, default=0.005)
    parser.add_argument("--max-scale-m", type=float, default=0.1)
    parser.add_argument("--max-delta-m", type=float, default=0.1)
    parser.add_argument("--num-queries-per-frame-view", type=int, default=80000)
    parser.add_argument("--max-total-queries", type=int, default=960000)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument("--depth-min-valid-m", type=float, default=0.1)
    parser.add_argument("--depth-max-valid-m", type=float, default=99.5)
    parser.add_argument("--depth-abs-threshold-m", type=float, default=3.0)
    parser.add_argument("--depth-rel-threshold", type=float, default=0.2)
    parser.add_argument("--flow-scale-px", type=float, default=64.0)
    parser.add_argument("--flow-white-threshold", type=float, default=0.02)
    parser.add_argument("--flow-white-temperature", type=float, default=0.01)
    parser.add_argument("--flow-inverse-iterations", type=int, default=3)
    parser.add_argument("--flow-track-max-step-m", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=424)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.set_defaults(
        use_projection_valid_mask=True,
        learned_local_sampling=False,
        iterative_refinement_layers=2,
        target_camera_conditioning=True,
        time_origin=0.0,
        time_denominator=3.0,
    )
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def read_manifest(path: Path) -> list[dict]:
    return dataset_manifest.read_jsonl(path)


def split_rows(
    rows: list[dict], val_scenes: list[int], train_scenes: list[int] | None, limit_train_scenes: int
) -> tuple[list[dict], list[dict]]:
    available = {int(row["scene_index"]) for row in rows}
    val_set = set(val_scenes)
    missing = val_set - available
    if missing:
        raise ValueError(f"Validation scenes absent from manifest: {sorted(missing)}")
    train_rows = [row for row in rows if int(row["scene_index"]) not in val_set]
    val_rows = [row for row in rows if int(row["scene_index"]) in val_set]
    if train_scenes is not None:
        wanted = set(train_scenes)
        overlap = wanted & val_set
        if overlap:
            raise ValueError(f"Train scenes overlap validation scenes: {sorted(overlap)}")
        train_rows = [row for row in train_rows if int(row["scene_index"]) in wanted]
    elif limit_train_scenes > 0:
        kept_scenes = sorted({int(row["scene_index"]) for row in train_rows})[:limit_train_scenes]
        train_rows = [row for row in train_rows if int(row["scene_index"]) in set(kept_scenes)]
    return train_rows, val_rows


def filter_train_rows(
    rows: list[dict], train_scenes: list[int] | None, limit_train_scenes: int
) -> list[dict]:
    if train_scenes is not None:
        wanted = set(train_scenes)
        available = {int(row["scene_index"]) for row in rows}
        missing = wanted - available
        if missing:
            raise ValueError(f"Train scenes absent from manifest: {sorted(missing)}")
        return [row for row in rows if int(row["scene_index"]) in wanted]
    if limit_train_scenes > 0:
        kept = sorted({int(row["scene_index"]) for row in rows})[:limit_train_scenes]
        kept_set = set(kept)
        return [row for row in rows if int(row["scene_index"]) in kept_set]
    return rows


def resolve_manifests_and_caches(
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], Path, Path]:
    separate_manifests = args.train_manifest is not None or args.val_manifest is not None
    if separate_manifests:
        if args.train_manifest is None or args.val_manifest is None:
            raise ValueError("--train-manifest and --val-manifest must be provided together")
        train_rows = filter_train_rows(
            read_manifest(args.train_manifest), args.train_scenes, args.limit_train_scenes
        )
        val_rows = read_manifest(args.val_manifest)
        train_cache_root = args.train_cache_root or args.cache_root
        val_cache_root = args.val_cache_root or args.cache_root
    else:
        if args.manifest is None:
            raise ValueError("Provide --manifest or both --train-manifest and --val-manifest")
        rows = read_manifest(args.manifest)
        train_rows, val_rows = split_rows(
            rows, args.val_scenes, args.train_scenes, args.limit_train_scenes
        )
        train_cache_root = val_cache_root = args.cache_root
    if train_cache_root is None or val_cache_root is None:
        raise ValueError("Provide cache roots for both train and validation data")
    if not train_rows:
        raise ValueError("Training manifest selection is empty")
    if not val_rows:
        raise ValueError("Validation manifest selection is empty")
    return train_rows, val_rows, train_cache_root, val_cache_root


def build_query_source(args: argparse.Namespace, split: str):
    if not args.online_query:
        return PackagedQuerySource()
    names = {
        "ann_file": getattr(args, f"{split}_ann_file"),
        "flow_rgb_root": getattr(args, f"{split}_masked_flow_rgb_root"),
        "flow_index_path": getattr(args, f"{split}_masked_flow_index"),
        "data_root": args.data_root,
    }
    option_names = {
        "ann_file": f"--{split}-ann-file",
        "flow_rgb_root": f"--{split}-masked-flow-rgb-root",
        "flow_index_path": f"--{split}-masked-flow-index",
        "data_root": "--data-root",
    }
    motion_release_root = (
        getattr(args, f"{split}_motion_release_root", None)
        or getattr(args, "motion_release_root", None)
    )
    motion_release_manifest = getattr(args, "motion_release_manifest", None)
    use_motion_release = motion_release_root is not None
    required_names = ("ann_file", "data_root") if use_motion_release else tuple(names)
    missing = [option_names[name] for name in required_names if names[name] is None]
    if missing:
        raise ValueError(f"--online-query requires: {', '.join(missing)}")
    return OnlineQuerySource(
        **names,
        width=args.width,
        height=args.height,
        allow_missing_flow_rgb=args.allow_missing_flow_rgb,
        zero_flow_input=args.zero_flow_input,
        motion_data=(
            motion_release.MotionRelease(
                motion_release_root,
                motion_release_manifest,
            )
            if use_motion_release
            else None
        ),
    )


def cache_path_for(row: dict, cache_root: Path) -> Path:
    cache_row = row
    if "source_manifest_index" in row:
        cache_row = {**row, "manifest_index": int(row["source_manifest_index"])}
    return feature_cache.cache_path(cache_root, cache_row)


def cache_manifest_index(row: dict) -> int:
    return int(row.get("source_manifest_index", row["manifest_index"]))


def row_window_start(row: dict, default: int) -> int:
    return int(row.get("window_start", default))


def resolve_fixed_val_rows(args: argparse.Namespace, val_rows: list[dict]) -> list[dict]:
    if args.fixed_val_manifest_index is not None and args.fixed_val_manifest_indices is not None:
        raise ValueError(
            "Use either --fixed-val-manifest-index or --fixed-val-manifest-indices, not both"
        )
    if args.fixed_val_manifest_indices is not None:
        positions = list(args.fixed_val_manifest_indices)
    elif args.fixed_val_manifest_index is not None:
        positions = [args.fixed_val_manifest_index]
    else:
        front_positions = [
            position
            for position, row in enumerate(val_rows)
            if feature_cache.clip_views(cache_manifest_index(row))[0] == "front"
        ]
        positions = front_positions[:1] or [0]
    if not positions:
        raise ValueError("At least one fixed validation row is required")
    if len(set(positions)) != len(positions):
        raise ValueError(f"Fixed validation row positions contain duplicates: {positions}")
    invalid = [position for position in positions if position < 0 or position >= len(val_rows)]
    if invalid:
        raise IndexError(
            f"Fixed validation row positions out of range: {invalid}; "
            f"validation rows={len(val_rows)}"
        )
    fixed_rows = [val_rows[position] for position in positions]
    for row in fixed_rows:
        if feature_cache.clip_views(cache_manifest_index(row))[0] != "front":
            raise ValueError(
                "Every fixed validation row must use the front view triplet: "
                f"manifest_index={row['manifest_index']}"
            )
    return fixed_rows


def validate_cache_paths(rows: list[dict], cache_root: Path) -> None:
    missing = [str(cache_path_for(row, cache_root)) for row in rows if not cache_path_for(row, cache_root).exists()]
    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} clip caches. First missing paths:\n{preview}")


def ssim_value(first: torch.Tensor, second: torch.Tensor) -> float:
    """Compute the project-standard Gaussian-window SSIM for one BCHW image pair."""
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError(f"Expected matching BCHW tensors, got {first.shape} and {second.shape}")
    window_size = 11
    sigma = 1.5
    coordinates = torch.arange(window_size, device=first.device, dtype=first.dtype)
    coordinates -= (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(first.shape[1], 1, window_size, window_size)
    mean_first = F.conv2d(first, kernel, groups=first.shape[1])
    mean_second = F.conv2d(second, kernel, groups=second.shape[1])
    mean_first_sq = mean_first.square()
    mean_second_sq = mean_second.square()
    mean_product = mean_first * mean_second
    variance_first = F.conv2d(first.square(), kernel, groups=first.shape[1]) - mean_first_sq
    variance_second = F.conv2d(second.square(), kernel, groups=second.shape[1]) - mean_second_sq
    covariance = F.conv2d(first * second, kernel, groups=first.shape[1]) - mean_product
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = (
        (2.0 * mean_product + c1)
        * (2.0 * covariance + c2)
        / ((mean_first_sq + mean_second_sq + c1) * (variance_first + variance_second + c2))
    )
    return float(ssim_map.mean().item())


def training_source(row: dict) -> str:
    return str(row.get("_training_source", "train"))


def training_cache_path(row: dict, train_cache_root: Path, val_cache_root: Path) -> Path:
    cache_root = val_cache_root if training_source(row) == "val" else train_cache_root
    return cache_path_for(row, cache_root)


def stage1_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        width=args.width,
        height=args.height,
        depth_min_valid_m=args.depth_min_valid_m,
        depth_max_valid_m=args.depth_max_valid_m,
        depth_abs_threshold_m=args.depth_abs_threshold_m,
        depth_rel_threshold=args.depth_rel_threshold,
        exclude_sky_by_depth=False,
        sky_depth_threshold_m=90.0,
        static_mask_source="valid_depth",
        dense_pixel_queries=False,
        num_queries_per_frame_view=args.num_queries_per_frame_view,
        max_total_queries=args.max_total_queries,
        cell_size=args.cell_size,
        flow_scale_px=args.flow_scale_px,
        flow_white_threshold=args.flow_white_threshold,
        flow_white_temperature=args.flow_white_temperature,
        flow_inverse_iterations=args.flow_inverse_iterations,
        flow_track_max_step_m=args.flow_track_max_step_m,
        extra_edge_queries_per_frame_view=getattr(args, "extra_edge_queries_per_frame_view", 0),
    )


def pack_stage1_queries(queries: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, int]]:
    points = queries["points_ref"]
    values = torch.cat(
        [
            torch.arange(points.shape[0], device=points.device, dtype=torch.float32)[:, None],
            queries["source_frame"].float()[:, None],
            queries["source_view"].float()[:, None],
            queries["source_u"][:, None],
            queries["source_v"][:, None],
            queries["source_depth_m"][:, None],
            points,
            queries["colors"],
            queries["ray_dir_ref"],
            queries["ray_moment_ref"],
            queries["dynamic_probability"][:, None],
        ],
        dim=1,
    )
    names = [
        "query_id", "source_frame", "source_view", "source_u", "source_v", "source_depth_m",
        "x_ref", "y_ref", "z_ref", "rgb_r", "rgb_g", "rgb_b",
        "ray_dir_x_ref", "ray_dir_y_ref", "ray_dir_z_ref",
        "ray_moment_x_ref", "ray_moment_y_ref", "ray_moment_z_ref", "dynamic_probability",
    ]
    return values, {name: index for index, name in enumerate(names)}


def build_window_sample(
    row: dict,
    window_start: int,
    cache_payload: dict,
    clip: dict,
    args: argparse.Namespace,
    device: torch.device,
    context_frames: tuple[int, ...] | list[int] | None = None,
) -> WindowSample:
    configured_context_frame_count = int(getattr(args, "context_frame_count", 4))
    if configured_context_frame_count <= 0:
        raise ValueError(
            "context_frame_count must be positive, "
            f"found {configured_context_frame_count}"
        )
    video_length = int(clip["video_length"] if "video_length" in clip else clip["rgb_target"].shape[0])
    if context_frames is None:
        if window_start < 0 or window_start + configured_context_frame_count > video_length:
            raise ValueError(
                f"Context window [{window_start}, "
                f"{window_start + configured_context_frame_count}) "
                f"exceeds video length {video_length}"
            )
        frames = list(range(window_start, window_start + configured_context_frame_count))
    else:
        frames = [int(frame) for frame in context_frames]
        if not frames or len(set(frames)) != len(frames):
            raise ValueError(
                f"Expected distinct explicit context frames, found {frames}"
            )
        if frames != sorted(frames) or frames[0] < 0 or frames[-1] >= video_length:
            raise ValueError(f"Invalid context frames {frames} for video length {video_length}")
    views = [int(value) for value in cache_payload["view_indices"]]
    if args.single_view_train:
        # Single-view direct-RGB training uses one camera consistently for query
        # construction and supervision so each Gaussian has one photometric target.
        single_view_name = getattr(args, "single_view_name", "CAM_FRONT")
        single_view = pointforward.VIEW_NAMES.index(single_view_name)
        if single_view not in views:
            raise ValueError(f"{single_view_name} not in cache views {views}")
        views = [single_view]
    view_slots = {view: slot for slot, view in enumerate(views)}
    all_views = [int(value) for value in cache_payload["view_indices"]]
    cache_slots = [all_views.index(view) for view in views]
    rgb_u8 = cache_payload["rgb_u8"][cache_slots].to(device=device, non_blocking=True)
    depth_m = cache_payload["depth_m"][cache_slots].to(device=device, dtype=torch.float32, non_blocking=True)
    feature_video = cache_payload["feature_maps"][cache_slots].to(device=device, dtype=torch.float32, non_blocking=True)
    query_generator = torch.Generator(device=device)
    context_seed = sum((index + 1) * frame for index, frame in enumerate(frames))
    query_generator.manual_seed(
        args.seed + int(row["manifest_index"]) * 1009 + context_seed * 9176
    )
    s1_args = stage1_args(args)
    if args.temporal_context_mode == "alternating_sparse4_stride4_sparse6_stride2":
        s1_args.num_queries_per_frame_view = min(
            s1_args.num_queries_per_frame_view,
            args.max_total_queries // (len(frames) * len(views)),
        )
    query_batches = []
    for frame in frames:
        for view in views:
            query_batch, _, _ = stage1.append_query_batch(
                clip,
                rgb_u8,
                depth_m,
                s1_args,
                frame,
                view,
                query_generator,
                device,
                view_slot=view_slots[view],
                make_panel=False,
                extra_edge_queries=getattr(args, "extra_edge_queries_per_frame_view", 0),
            )
            query_batches.append(query_batch)
    queries = stage1.concatenate_query_batches(query_batches)
    extra_edge = int(getattr(args, "extra_edge_queries_per_frame_view", 0))
    if extra_edge > 0:
        # Total budget grows with the structural pool: base cap is per frame-view,
        # and edge queries are appended on top (not carved out of the base pool).
        args.max_total_queries = max(
            args.max_total_queries,
            len(frames) * len(views) * (args.num_queries_per_frame_view + extra_edge),
        )
    queries, _ = stage1.limit_queries(queries, args.max_total_queries, query_generator)
    track_points, raw_track_points, track_valid, dynamic_probability, _ = stage1.initialize_flow_rgb_tracks(
        clip,
        depth_m,
        queries,
        frames,
        s1_args,
        device,
        view_slots=view_slots,
    )
    queries["dynamic_probability"] = dynamic_probability
    observations, _ = stage1.collect_observations(
        clip,
        rgb_u8,
        depth_m,
        queries,
        frames,
        views,
        s1_args,
        device,
        track_points_ref=track_points,
        track_valid=track_valid,
        track_frames=frames,
        view_slots=view_slots,
    )
    packed_queries, columns = pack_stage1_queries(queries)
    stage1_payload = {"queries": packed_queries, "col": columns, "observations": observations}
    context_features = torch.stack(
        [feature_video[view_slots[view], :, frame] for frame in frames for view in views]
    )
    # Normalize time by the selected context span. This preserves the legacy
    # contiguous-window 0..1 convention for checkpoint compatibility, while a
    # sparse context retains its real temporal spacing. Full-clip targets may
    # legitimately fall just outside 0..1 when the context omits a clip edge.
    args.time_origin = float(frames[0])
    args.time_denominator = float(max(frames[-1] - frames[0], 1))
    batch = pointforward.build_stage2_batch(stage1_payload, context_features, args)
    batch.obs_uv = torch.empty((0,), device=device)
    context_viewmats = []
    context_intrinsics = []
    for frame in frames:
        for view in views:
            frame_to_ref = clip["frame_to_ref_lidar"][frame].to(device=device).float()
            context_viewmats.append(
                clip["lidar2camera"][frame, view].to(device=device).float() @ torch.linalg.inv(frame_to_ref)
            )
            context_intrinsics.append(clip["camera_intrinsics"][frame, view].to(device=device).float())
    sample = WindowSample(
        batch=batch,
        clip=clip,
        context_frames=observations["context_frame"],
        context_viewmats=torch.stack(context_viewmats),
        context_intrinsics=torch.stack(context_intrinsics),
        frames=frames,
        views=views,
        dynamic_probability_mean=float(dynamic_probability.mean()),
        flow_displacement_prior=(
            raw_track_points[:, -1] - raw_track_points[:, 0]
            if raw_track_points is not None and len(frames) > 1
            else None
        ),
    )
    del rgb_u8, depth_m, feature_video, query_batches, queries, track_points, raw_track_points, track_valid
    del observations, packed_queries, stage1_payload, context_features
    # NOTE: intentionally no torch.cuda.empty_cache() here. The per-step cache
    # release caused GPU memory to thrash between ~3GB and ~92GB every step
    # (cudaMalloc/cudaFree each step). Letting the PyTorch caching allocator
    # keep the blocks resident avoids the repeated allocation overhead.
    return sample


def target_image(
    sample: WindowSample,
    query_source,
    frame: int,
    view: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    rgb = query_source.target_rgb(sample.clip, frame, view)
    return pointforward.resize_target(torch.from_numpy(rgb).permute(2, 0, 1).to(device), args.height, args.width)


def select_full_clip_targets(
    target_scope: tuple[int, ...],
    targets_per_step: int,
    seed: int,
    epoch: int,
    manifest_index: int,
) -> tuple[int, ...]:
    if targets_per_step <= 0 or targets_per_step >= len(target_scope):
        return target_scope
    blocks_per_cycle = math.ceil(len(target_scope) / targets_per_step)
    cycle = epoch // blocks_per_cycle
    block = epoch % blocks_per_cycle
    shuffled = list(target_scope)
    random.Random(seed + manifest_index * 9176 + cycle * 1000003).shuffle(shuffled)
    offset = block * targets_per_step
    selected = [
        shuffled[(offset + index) % len(shuffled)]
        for index in range(targets_per_step)
    ]
    return tuple(sorted(selected))


def train_schedule(
    train_rows: list[dict], epoch: int, rank: int, world_size: int, seed: int,
    temporal_context_mode: str = "contiguous",
    neighbor_supervision_weight: float = 0.25,
    sparse_context_frame_count: int = 6,
) -> list[WindowRecord]:
    clip_order = train_rows.copy()
    random.Random(seed + epoch).shuffle(clip_order)
    padded_count = math.ceil(len(clip_order) / world_size) * world_size
    repeats = math.ceil(padded_count / len(clip_order))
    clip_order = (clip_order * repeats)[:padded_count]
    rank_clips = clip_order[rank::world_size]
    schedule = []
    for clip_offset, row in enumerate(rank_clips):
        if temporal_context_mode == "mixed_sparse2_contiguous1":
            video_length = int(row.get("video_length", 17))
            if video_length != 17:
                raise ValueError(
                    "mixed_sparse2_contiguous1 currently requires 17-frame clips, "
                    f"found video_length={video_length}"
                )
            rng = random.Random(seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset)
            sparse_patterns = [
                (0, 4, 10, 16),
                (0, 6, 12, 16),
                (1, 5, 11, 15),
                (2, 7, 12, 16),
            ]
            selected_sparse = rng.sample(sparse_patterns, 2)
            contiguous_start = rng.randrange(video_length - 4 + 1)
            context_records = [
                (selected_sparse[0], "sparse"),
                (selected_sparse[1], "sparse"),
                (tuple(range(contiguous_start, contiguous_start + 4)), "contiguous"),
            ]
            rng.shuffle(context_records)
            # Two sparse records per epoch cover two new full-clip targets;
            # across nine epochs this rotation visits every one of 17 frames.
            target_base = (epoch * 2 + int(row["manifest_index"]) * 7) % video_length
            sparse_offset = 0
            for record_offset, (frames, mode) in enumerate(context_records):
                if mode == "sparse":
                    target_scope = tuple(range(video_length))
                    target_frame = (target_base + sparse_offset) % video_length
                    sparse_offset += 1
                else:
                    local_start = max(0, int(frames[0]) - 2)
                    local_end = min(video_length - 1, int(frames[-1]) + 2)
                    target_scope = tuple(range(local_start, local_end + 1))
                    local_offset = (epoch + int(row["manifest_index"]) + record_offset) % len(target_scope)
                    target_frame = target_scope[local_offset]
                schedule.append(
                    WindowRecord(
                        row=row,
                        window_start=int(frames[0]),
                        context_frames=frames,
                        context_mode=mode,
                        target_frame=target_frame,
                        target_scope=target_scope,
                        target_weight=1.0,
                    )
                )
            continue
        if temporal_context_mode == "fixed_window_manifest":
            video_length = int(row.get("video_length", 17))
            start = row_window_start(row, -1)
            if start < 0 or start + 4 > video_length:
                raise ValueError(
                    "fixed_window_manifest requires a valid four-frame window_start, "
                    f"found start={start}, video_length={video_length}, "
                    f"manifest_index={row['manifest_index']}"
                )
            schedule.append(WindowRecord(row=row, window_start=start))
            continue
        if temporal_context_mode == "contiguous4_neighbor2_sparse4_stride2":
            video_length = int(row.get("video_length", 17))
            if video_length < 8:
                raise ValueError(
                    "contiguous4_neighbor2_sparse4_stride2 requires at least 8-frame clips, "
                    f"found video_length={video_length}"
                )
            rng = random.Random(seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset)
            contiguous_start = rng.randrange(video_length - 4 + 1)
            sparse_start = rng.randrange(video_length - 8 + 1)
            context_records = [
                (tuple(range(contiguous_start, contiguous_start + 4)), "contiguous"),
                (tuple(sparse_start + 2 * offset for offset in range(4)), "sparse_stride2"),
            ]
            rng.shuffle(context_records)
            for record_offset, (frames, mode) in enumerate(context_records):
                if mode == "contiguous":
                    local_start = max(0, int(frames[0]) - 2)
                    local_end = min(video_length - 1, int(frames[-1]) + 2)
                    target_scope = tuple(range(local_start, local_end + 1))
                    target_offset = (epoch + int(row["manifest_index"]) + record_offset) % len(target_scope)
                    target_frame = target_scope[target_offset]
                    target_weight = neighbor_supervision_weight if target_frame not in frames else 1.0
                else:
                    target_scope = tuple(range(sparse_start, min(video_length, sparse_start + 8)))
                    target_offset = (epoch + int(row["manifest_index"]) + record_offset) % len(target_scope)
                    target_frame = target_scope[target_offset]
                    target_weight = 1.0
                schedule.append(
                    WindowRecord(
                        row=row,
                        window_start=int(frames[0]),
                        context_frames=frames,
                        context_mode=mode,
                        target_frame=target_frame,
                        target_scope=target_scope,
                        target_weight=target_weight,
                    )
                )
            continue
        if temporal_context_mode == "contiguous4_local8":
            video_length = int(row.get("video_length", 17))
            if video_length < 8:
                raise ValueError(
                    "contiguous4_local8 requires at least 8-frame clips, "
                    f"found video_length={video_length}"
                )
            # Keep four contiguous context frames inside an exact eight-frame
            # target span, with two supervised neighbors on each side.
            start_min = 2
            start_max_exclusive = video_length - 8 + 1
            if start_min >= start_max_exclusive:
                raise ValueError(
                    "contiguous4_local8 cannot place a centered eight-frame span "
                    f"inside video_length={video_length}"
                )
            start = random.Random(
                seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset
            ).randrange(start_min, start_max_exclusive)
            frames = tuple(range(start, start + 4))
            target_scope = tuple(range(start - 2, start + 6))
            target_offset = (epoch + int(row["manifest_index"])) % len(target_scope)
            schedule.append(
                WindowRecord(
                    row=row,
                    window_start=start,
                    context_frames=frames,
                    context_mode="contiguous4_local8",
                    target_frame=target_scope[target_offset],
                    target_scope=target_scope,
                    target_weight=1.0,
                )
            )
            continue
        if temporal_context_mode == "sparse4_stride2_local8":
            video_length = int(row.get("video_length", 17))
            if video_length < 8:
                raise ValueError(
                    "sparse4_stride2_local8 requires at least 8-frame clips, "
                    f"found video_length={video_length}"
                )
            start_max_exclusive = video_length - 6
            if start_max_exclusive <= 0:
                raise ValueError(
                    "sparse4_stride2_local8 cannot place an eight-frame target span "
                    f"inside video_length={video_length}"
                )
            start = random.Random(
                seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset
            ).randrange(0, start_max_exclusive)
            frames = tuple(start + 2 * offset for offset in range(4))
            target_scope = tuple(range(start, start + 8))
            target_offset = (epoch + int(row["manifest_index"])) % len(target_scope)
            schedule.append(
                WindowRecord(
                    row=row,
                    window_start=start,
                    context_frames=frames,
                    context_mode="sparse4_stride2_local8",
                    target_frame=target_scope[target_offset],
                    target_scope=target_scope,
                    target_weight=1.0,
                )
            )
            continue
        if temporal_context_mode == "sparse_stride2_fullclip":
            video_length = int(row.get("video_length", 17))
            context_span = 2 * (sparse_context_frame_count - 1) + 1
            if context_span > video_length:
                raise ValueError(
                    "sparse_stride2_fullclip cannot fit "
                    f"{sparse_context_frame_count} stride-2 frames inside "
                    f"video_length={video_length}"
                )
            start = random.Random(
                seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset
            ).randrange(0, video_length - context_span + 1)
            frames = tuple(start + 2 * offset for offset in range(sparse_context_frame_count))
            target_scope = tuple(range(video_length))
            schedule.append(
                WindowRecord(
                    row=row,
                    window_start=start,
                    context_frames=frames,
                    context_mode="sparse_stride2_fullclip",
                    target_frame=target_scope[0],
                    target_scope=target_scope,
                    target_weight=1.0,
                )
            )
            continue
        if temporal_context_mode == "alternating_sparse4_stride4_sparse6_stride2":
            video_length = int(row.get("video_length", 17))
            alternating_index = epoch * len(rank_clips) + clip_offset
            use_sparse6 = alternating_index % 2 == 1
            rng = random.Random(
                seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset
            )
            if use_sparse6:
                context_span = 11
                if video_length < context_span:
                    raise ValueError(
                        "Six stride-2 context frames require an 11-frame local span, "
                        f"found video_length={video_length}"
                    )
                start = rng.randrange(0, video_length - context_span + 1)
                frames = tuple(start + 2 * offset for offset in range(6))
                target_scope = tuple(range(start, start + context_span))
                context_mode = "sparse6_stride2_local11"
            else:
                context_span = 15
                if video_length < context_span:
                    raise ValueError(
                        "Four stride-4 inputs with eight stride-2 targets require "
                        f"a 15-frame span, found video_length={video_length}"
                    )
                start = rng.randrange(0, video_length - context_span + 1)
                frames = tuple(start + 4 * offset for offset in range(4))
                target_scope = tuple(start + 2 * offset for offset in range(8))
                context_mode = "sparse4_stride4_supervise8"
            schedule.append(
                WindowRecord(
                    row=row,
                    window_start=start,
                    context_frames=frames,
                    context_mode=context_mode,
                    target_frame=target_scope[0],
                    target_scope=target_scope,
                    target_weight=1.0,
                )
            )
            continue
        if temporal_context_mode == "sparse4_stride4_supervise8":
            video_length = int(row.get("video_length", 17))
            if video_length < 15:
                raise ValueError(
                    "sparse4_stride4_supervise8 requires at least 15-frame clips, "
                    f"found video_length={video_length}"
                )
            start_max_exclusive = video_length - 15 + 1
            start = random.Random(
                seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset
            ).randrange(0, start_max_exclusive)
            frames = tuple(start + 4 * offset for offset in range(4))
            target_scope = tuple(start + 2 * offset for offset in range(8))
            schedule.append(
                WindowRecord(
                    row=row,
                    window_start=start,
                    context_frames=frames,
                    context_mode="sparse4_stride4_supervise8",
                    target_frame=target_scope[0],
                    target_scope=target_scope,
                    target_weight=1.0,
                )
            )
            continue
        starts = WINDOW_STARTS.copy()
        random.Random(seed + epoch * 1000003 + int(row["manifest_index"]) + clip_offset).shuffle(starts)
        schedule.extend(WindowRecord(row=row, window_start=start) for start in starts)
    return schedule


def records_per_clip(temporal_context_mode: str) -> int:
    try:
        return TEMPORAL_CONTEXT_RECORDS_PER_CLIP[temporal_context_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown temporal context mode: {temporal_context_mode}") from exc


def unwrap(model: nn.Module) -> FlowTrackRenderModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_metrics(values: list[float], device: torch.device, world_size: int) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.cpu().tolist()


def reduce_optional_mean(
    values: list[float], device: torch.device, world_size: int
) -> float | None:
    total_and_count = torch.tensor(
        [sum(values), len(values)], dtype=torch.float64, device=device
    )
    if world_size > 1:
        dist.all_reduce(total_and_count, op=dist.ReduceOp.SUM)
    total, count = total_and_count.cpu().tolist()
    return total / count if count > 0 else None


@torch.no_grad()
def validate_fixed_window(
    model: FlowTrackRenderModel,
    row: dict,
    cache_payload: dict,
    clip: dict,
    query_source,
    lpips_model,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
) -> dict:
    model.eval()
    inference_start = time.perf_counter()
    context_frames = None
    validation_frames = None
    validation_window_start = row_window_start(row, args.fixed_val_window_start)
    if args.temporal_context_mode in {
        "sparse_stride2_fullclip",
        "alternating_sparse4_stride4_sparse6_stride2",
    }:
        video_length = int(row.get("video_length", 17))
        context_span = 2 * (args.sparse_context_frame_count - 1) + 1
        validation_window_start = min(
            max(validation_window_start, 0), video_length - context_span
        )
        context_frames = tuple(
            validation_window_start + 2 * offset
            for offset in range(args.sparse_context_frame_count)
        )
        if args.temporal_context_mode == "sparse_stride2_fullclip":
            validation_frames = tuple(range(video_length))
        else:
            validation_frames = tuple(
                range(validation_window_start, validation_window_start + context_span)
            )
    sample = build_window_sample(
        row,
        validation_window_start,
        cache_payload,
        clip,
        args,
        device,
        context_frames=context_frames,
    )
    gaussian = model.predict_gaussians(sample, args)
    sums = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0, "ssim": 0.0}
    visual_dir = (
        args.output_dir
        / "val_visuals"
        / f"step{step:06d}"
        / f"manifest_{int(row['manifest_index']):06d}"
    )
    frames_to_render = validation_frames or tuple(sample.frames)
    for frame in frames_to_render:
        for view in sample.views:
            rendered = pointforward.render_output(
                gaussian,
                model.appearance_decoder,
                sample.clip,
                frame,
                view,
                args.height,
                args.width,
                args,
                device,
                camera_affine=getattr(model, "camera_affine", None),
            )
            target = target_image(sample, query_source, frame, view, args, device)
            sums["l1"] += float((rendered - target).abs().mean())
            sums["lpips"] += float(
                lpips_model(rendered[None] * 2.0 - 1.0, target[None] * 2.0 - 1.0).mean()
            )
            sums["psnr"] += pointforward.psnr(rendered, target)
            sums["ssim"] += ssim_value(
                rendered.detach().float().clamp(0.0, 1.0)[None],
                target.detach().float().clamp(0.0, 1.0)[None],
            )
            pointforward.save_panel(
                visual_dir / f"f{frame:02d}_{pointforward.VIEW_NAMES[view]}.jpg",
                rendered,
                target,
            )
    count = len(frames_to_render) * len(sample.views)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time_sec = time.perf_counter() - inference_start
    record = {
        "step": step,
        "manifest_index": int(row["manifest_index"]),
        "scene_index": int(row["scene_index"]),
        "scene_frame_start": int(row["scene_frame_start"]),
        "window_start": validation_window_start,
        "context_frames": sample.frames,
        "frames": list(frames_to_render),
        "views": [pointforward.VIEW_NAMES[view] for view in sample.views],
        "dynamic_probability_mean": sample.dynamic_probability_mean,
        "target_rgb_source": sample.clip.get("_target_rgb_source", "packaged_rgb"),
        **{key: value / count for key, value in sums.items()},
        "target_count": count,
        "inference_time_sec": inference_time_sec,
        "inference_time_sec_per_target": inference_time_sec / count,
    }
    del gaussian, sample
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def validate_fixed_windows(
    model: FlowTrackRenderModel,
    rows: list[dict],
    val_cache_root: Path,
    query_source,
    lpips_model,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
) -> dict:
    validation_start = time.perf_counter()
    sample_records = []
    for row in rows:
        cache_payload = torch.load(
            cache_path_for(row, val_cache_root), map_location="cpu", weights_only=False
        )
        clip = query_source.load_clip(row, cache_payload["view_indices"])
        sample_records.append(
            validate_fixed_window(
                model,
                row,
                cache_payload,
                clip,
                query_source,
                lpips_model,
                args,
                device,
                step,
            )
        )
        del cache_payload, clip
    target_count = sum(int(record["target_count"]) for record in sample_records)
    metric_names = ("l1", "lpips", "psnr", "ssim")
    aggregate = {
        "step": step,
        "sample_count": len(sample_records),
        "target_count": target_count,
        "samples": sample_records,
        **{
            name: sum(float(record[name]) * int(record["target_count"]) for record in sample_records)
            / target_count
            for name in metric_names
        },
        "inference_time_sec": sum(
            float(record["inference_time_sec"]) for record in sample_records
        ),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    aggregate["validation_wall_time_sec"] = time.perf_counter() - validation_start
    aggregate["inference_time_sec_per_sample"] = (
        aggregate["inference_time_sec"] / len(sample_records)
    )
    return aggregate


def write_run_files(
    args: argparse.Namespace,
    train_rows: list[dict],
    val_rows: list[dict],
    fixed_val_rows: list[dict],
    world_size: int,
) -> None:
    if args.resume_checkpoint is not None:
        if not args.output_dir.exists() or not args.checkpoint_dir.exists():
            raise FileNotFoundError("Resume requires the existing output and checkpoint directories")
        with (args.output_dir / "resume_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "resume_checkpoint": str(args.resume_checkpoint),
                        "command": " ".join(shlex.quote(value) for value in sys.argv),
                    }
                )
                + "\n"
            )
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    if args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty checkpoint directory: {args.checkpoint_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["world_size"] = world_size
    (args.output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "command.sh").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
    )
    split = {
        "train_scenes": sorted({int(row["scene_index"]) for row in train_rows}),
        "val_scenes": sorted({int(row["scene_index"]) for row in val_rows}),
        "train_clips": len(train_rows),
        "train_source_clips": {
            source: sum(training_source(row) == source for row in train_rows)
            for source in ("train", "val")
        },
        "generated_rgb_target_rows": sum(
            row.get("generated_rgb_root") is not None for row in train_rows
        ),
        "val_clips": len(val_rows),
        "train_records_per_clip": records_per_clip(args.temporal_context_mode),
        "train_windows_per_epoch": len(train_rows) * records_per_clip(args.temporal_context_mode),
        "val_windows": (
            len(val_rows)
            if args.temporal_context_mode == "fixed_window_manifest"
            else len(val_rows) * len(WINDOW_STARTS)
        ),
        "window_starts": WINDOW_STARTS,
        "fixed_val_samples": [
            {
                "manifest_index": int(row["manifest_index"]),
                "scene_index": int(row["scene_index"]),
                "scene_frame_start": int(row["scene_frame_start"]),
                "window_start": row_window_start(row, args.fixed_val_window_start),
            }
            for row in fixed_val_rows
        ],
    }
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    git_text = result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"
    (args.output_dir / "git.txt").write_text(git_text + "\n", encoding="utf-8")


def save_checkpoint(path: Path, model: nn.Module, optimizer, step: int, validation: dict | None, args) -> None:
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "validation": validation,
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if (args.height, args.width) != (424, 800):
        raise ValueError("This trainer requires full-resolution 424x800 supervision")
    rank, world_size, local_rank, device = distributed_context()
    if world_size < 1:
        raise ValueError(f"Expected at least one DDP rank, found {world_size}")
    train_rows, val_rows, train_cache_root, val_cache_root = resolve_manifests_and_caches(args)
    train_rows = [{**row, "_training_source": "train"} for row in train_rows]
    if args.include_val_in_train:
        train_rows.extend({**row, "_training_source": "val"} for row in val_rows)
    if args.train_manifest_indices is not None:
        selected = set(args.train_manifest_indices)
        train_rows = [row for row in train_rows if int(row["manifest_index"]) in selected]
        found = {int(row["manifest_index"]) for row in train_rows}
        if found != selected:
            raise ValueError(f"Train manifest indices absent from selection: {sorted(selected - found)}")
    fixed_val_rows = resolve_fixed_val_rows(args, val_rows)
    if args.train_manifest is None:
        val_scene_set = set(args.val_scenes)
        invalid_scenes = [
            int(row["scene_index"])
            for row in fixed_val_rows
            if int(row["scene_index"]) not in val_scene_set
        ]
        if invalid_scenes:
            raise ValueError(
                "Every fixed validation window must belong to a held-out validation scene: "
                f"{invalid_scenes}"
            )
    if args.limit_train_clips > 0:
        limited_rows = []
        limited_scenes = set()
        for row in train_rows:
            scene_key = (training_source(row), int(row["scene_index"]))
            if scene_key in limited_scenes:
                continue
            limited_rows.append(row)
            limited_scenes.add(scene_key)
            if len(limited_rows) == args.limit_train_clips:
                break
        train_rows = limited_rows
    if args.single_view_train:
        # Single-view training needs front-view caches; drop clips cached as rear.
        front_rows = [
            row for row in train_rows
            if feature_cache.clip_views(cache_manifest_index(row))[0] == "front"
        ]
        if not front_rows:
            raise ValueError("single_view_train found no front-view train clips")
        train_rows = front_rows
    if args.fast_dev_run:
        args.iterations = 1
        args.val_every_epochs = 1
        args.log_every = 1
        args.checkpoint_every_epochs = 1
        args.num_queries_per_frame_view = min(args.num_queries_per_frame_view, 1024)
        args.max_total_queries = min(args.max_total_queries, 12288)
        if args.limit_train_clips == 0:
            smoke_rows = []
            smoke_scenes = set()
            if args.include_val_in_train:
                source_rows = {
                    source: [row for row in train_rows if training_source(row) == source]
                    for source in ("train", "val")
                }
                smoke_candidates = []
                for offset in range(max(len(rows) for rows in source_rows.values())):
                    for source in ("train", "val"):
                        if offset < len(source_rows[source]):
                            smoke_candidates.append(source_rows[source][offset])
            else:
                smoke_candidates = train_rows
            for row in smoke_candidates:
                scene_key = (training_source(row), int(row["scene_index"]))
                if scene_key in smoke_scenes:
                    continue
                smoke_rows.append(row)
                smoke_scenes.add(scene_key)
                if len(smoke_rows) == world_size:
                    break
            train_rows = smoke_rows
        else:
            train_rows = train_rows[:world_size]
    train_source_rows = [row for row in train_rows if training_source(row) == "train"]
    val_source_rows = [row for row in train_rows if training_source(row) == "val"]
    validate_cache_paths(train_source_rows, train_cache_root)
    validate_cache_paths(val_source_rows, val_cache_root)
    validate_cache_paths(fixed_val_rows, val_cache_root)
    records_this_clip = records_per_clip(args.temporal_context_mode)
    if not 0.0 < args.neighbor_supervision_weight <= 1.0:
        raise ValueError("--neighbor-supervision-weight must be in (0, 1]")
    if args.sparse_context_frame_count < 2:
        raise ValueError("--sparse-context-frame-count must be >= 2")
    if 2 * (args.sparse_context_frame_count - 1) + 1 > 17:
        raise ValueError("--sparse-context-frame-count does not fit a 17-frame clip")
    if (
        args.temporal_context_mode == "alternating_sparse4_stride4_sparse6_stride2"
        and args.sparse_context_frame_count != 6
    ):
        raise ValueError(
            "alternating_sparse4_stride4_sparse6_stride2 requires "
            "--sparse-context-frame-count 6"
        )
    if args.full_clip_targets_per_step < 0:
        raise ValueError("--full-clip-targets-per-step must be >= 0")
    if args.temporal_context_mode == "sparse_stride2_fullclip":
        args.context_frame_count = args.sparse_context_frame_count
    if args.unet_depth < 1:
        raise ValueError("--unet-depth must be >= 1")
    if args.learned_dynamic_separation and not (
        args.learned_velocity or args.learned_dynamic_velocity
    ):
        raise ValueError(
            "--learned-dynamic-separation requires --learned-velocity or "
            "--learned-dynamic-velocity"
        )
    if args.rgb_head_only and args.init_checkpoint is None:
        raise ValueError("--rgb-head-only requires --init-checkpoint")
    if args.unfreeze_backbone and args.rgb_head_only:
        raise ValueError("--unfreeze-backbone cannot be combined with --rgb-head-only")
    if args.backbone_lr <= 0.0:
        raise ValueError("--backbone-lr must be positive")
    if args.single_view_train and args.single_view_name not in pointforward.VIEW_NAMES:
        raise ValueError(f"Unknown --single-view-name {args.single_view_name!r}")
    if args.context_frame_count != 4:
        raise ValueError("The multiscene trainer currently requires --context-frame-count 4")
    if args.reset_motion_residual_heads_on_init and args.init_checkpoint is None:
        raise ValueError("--reset-motion-residual-heads-on-init requires --init-checkpoint")
    if args.temporal_context_mode in {
        "mixed_sparse2_contiguous1",
        "contiguous4_neighbor2_sparse4_stride2",
        "contiguous4_local8",
        "sparse4_stride2_local8",
        "sparse4_stride4_supervise8",
        "sparse_stride2_fullclip",
        "alternating_sparse4_stride4_sparse6_stride2",
    } and not args.full_clip_target_supervision:
        raise ValueError(
            f"{args.temporal_context_mode} requires --full-clip-target-supervision so "
            "scheduled target frames are used"
        )
    steps_per_epoch = math.ceil(len(train_rows) / world_size) * records_this_clip
    args.steps_per_epoch = steps_per_epoch
    args.total_steps = args.iterations if args.iterations > 0 else args.epochs * steps_per_epoch
    train_query_source = build_query_source(args, "train")
    val_query_source = (
        build_query_source(args, "val")
        if rank == 0 or args.include_val_in_train
        else None
    )
    train_query_sources = {"train": train_query_source, "val": val_query_source}
    if rank == 0:
        write_run_files(args, train_rows, val_rows, fixed_val_rows, world_size)
    if world_size > 1:
        dist.barrier()

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    model: nn.Module = FlowTrackRenderModel(args).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=args.unfreeze_backbone,
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying head-freeze options")
    backbone_parameters = []
    if args.unfreeze_backbone:
        backbone_ids = {
            id(parameter)
            for module in unwrap(model).backbone_modules
            for parameter in module.parameters()
        }
        backbone_parameters = [parameter for parameter in trainable_parameters if id(parameter) in backbone_ids]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [parameter for parameter in trainable_parameters if id(parameter) not in backbone_ids]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    if args.lpips_module_root is not None and str(args.lpips_module_root) not in sys.path:
        sys.path.append(str(args.lpips_module_root))
    from lpips import LPIPS

    lpips_model = LPIPS(net="alex").to(device).eval().requires_grad_(False)
    metrics_path = args.output_dir / "metrics.jsonl"
    val_path = args.output_dir / "val_metrics.jsonl"
    best_val_psnr = -math.inf
    best_checkpoint = None
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as error:
            saved_groups = len(checkpoint["optimizer"].get("param_groups", ()))
            current_groups = len(optimizer.param_groups)
            raise ValueError(
                "The checkpoint optimizer layout does not match this training "
                f"configuration (checkpoint groups={saved_groups}, current "
                f"groups={current_groups}). --resume-checkpoint is only for an "
                "exact continuation with the same trainable modules and parameter "
                "groups. To fine-tune released model weights on generated latents, "
                "use --init-checkpoint instead; it loads the model and creates a "
                "new optimizer."
            ) from error
        start_step = int(checkpoint["step"])
        best_record_path = args.checkpoint_dir / "best.json"
        if best_record_path.exists():
            best_record = json.loads(best_record_path.read_text(encoding="utf-8"))
            best_val_psnr = float(best_record["psnr"])
            best_checkpoint = Path(best_record["checkpoint"])
        elif checkpoint.get("validation") is not None:
            best_val_psnr = float(checkpoint["validation"]["psnr"])
        if start_step >= args.total_steps:
            raise ValueError(
                f"Resume step {start_step} has already reached total_steps={args.total_steps}"
            )
    elif args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        target_state = unwrap(model).state_dict()
        loaded, partial_loaded, skipped = 0, 0, 0
        for name, value in checkpoint["model"].items():
            if name not in target_state:
                skipped += 1
                continue
            if tuple(target_state[name].shape) != tuple(value.shape):
                if (
                    getattr(args, "rgb_head_only", False)
                    and name in {"point_model.head.3.weight", "point_model.head.3.bias"}
                    and tuple(target_state[name].shape[:1]) == (12,)
                    and tuple(value.shape[:1]) == (140,)
                ):
                    target_state[name][:11].copy_(value[:11])
                    target_state[name][11].copy_(value[-1])
                    partial_loaded += 1
                    continue
                skipped += 1
                continue
            target_state[name].copy_(value)
            loaded += 1
        unwrap(model).load_state_dict(target_state)
        if args.reset_motion_residual_heads_on_init:
            point_model = unwrap(model).point_model
            residual_heads = (
                point_model.dynamic_residual_head,
                point_model.velocity_head,
            )
            if any(head is None for head in residual_heads):
                raise ValueError(
                    "--reset-motion-residual-heads-on-init requires learned dynamic "
                    "separation and learned velocity"
                )
            for head in residual_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        print(
            f"[rank {rank}] init_checkpoint: loaded {loaded} tensors "
            f"(+{partial_loaded} partial geometry heads), "
            f"skipped {skipped} shape-mismatched or absent, "
            f"reset_motion_residual_heads={args.reset_motion_residual_heads_on_init}",
            flush=True,
        )
    start_time = time.perf_counter()
    active_row_key = None
    active_clip = active_cache = None
    current_epoch = -1
    rank_schedule: list[WindowRecord] = []
    step_time_sum_sec = 0.0
    step_time_min_sec = math.inf
    step_time_max_sec = 0.0
    measured_step_count = 0

    for step in range(start_step + 1, args.total_steps + 1):
        step_start = time.perf_counter()
        epoch = (step - 1) // steps_per_epoch
        if not rank_schedule or epoch != current_epoch:
            current_epoch = epoch
            rank_schedule = train_schedule(
                train_rows,
                current_epoch,
                rank,
                world_size,
                args.seed,
                temporal_context_mode=args.temporal_context_mode,
                neighbor_supervision_weight=args.neighbor_supervision_weight,
                sparse_context_frame_count=args.sparse_context_frame_count,
            )
            if len(rank_schedule) != steps_per_epoch:
                raise RuntimeError(
                    f"Rank {rank} schedule has {len(rank_schedule)} steps, expected {steps_per_epoch}"
                )
        record = rank_schedule[(step - 1) % steps_per_epoch]
        row = record.row
        manifest_index = int(row["manifest_index"])
        row_source = training_source(row)
        row_key = (row_source, manifest_index)
        query_source = train_query_sources[row_source]
        if active_row_key != row_key:
            active_cache = torch.load(
                training_cache_path(row, train_cache_root, val_cache_root),
                map_location="cpu",
                weights_only=False,
            )
            active_clip = query_source.load_clip(row, active_cache["view_indices"])
            active_row_key = row_key
        sample = build_window_sample(
            row,
            record.window_start,
            active_cache,
            active_clip,
            args,
            device,
            context_frames=record.context_frames,
        )
        multi_target_supervision = (
            args.full_clip_target_supervision
            and args.temporal_context_mode in {
                "sparse4_stride4_supervise8",
                "sparse_stride2_fullclip",
                "alternating_sparse4_stride4_sparse6_stride2",
            }
        )
        if args.full_clip_target_supervision:
            if multi_target_supervision:
                if not record.target_scope:
                    raise RuntimeError("Multi-target supervision requires a target scope")
                target_frames = tuple(int(frame) for frame in record.target_scope)
                targets_per_step = args.full_clip_targets_per_step
                if (
                    args.temporal_context_mode
                    == "alternating_sparse4_stride4_sparse6_stride2"
                    and record.context_mode == "sparse4_stride4_supervise8"
                ):
                    targets_per_step = 0
                if 0 < targets_per_step < len(target_frames):
                    target_frames = select_full_clip_targets(
                        target_frames,
                        targets_per_step,
                        args.seed,
                        current_epoch,
                        manifest_index,
                    )
            else:
                if record.target_frame is None:
                    raise RuntimeError("Full-clip supervision requires a scheduled target frame")
                target_frames = (int(record.target_frame),)
            target_view = sample.views[(step - 1 + rank) % len(sample.views)]
        else:
            target_index = (step - 1 + rank) % (len(sample.frames) * len(sample.views))
            target_frames = (sample.frames[target_index // len(sample.views)],)
            target_view = sample.views[target_index % len(sample.views)]
        if args.single_view_train:
            target_view = sample.views[
                sample.views.index(pointforward.VIEW_NAMES.index(args.single_view_name))
            ]
        model.train()
        # Only seed the direct_rgb output bias on a fresh start. A --resume-checkpoint
        # run already carries a trained RGB bias in its optimizer/model state and must
        # not have it overwritten at the first resumed step.
        if start_step == 0 and step == start_step + 1 and args.appearance_mode == "direct_rgb":
            unwrap(model).init_rgb_bias_from_observations(sample)
        optimizer.zero_grad(set_to_none=True)
        target_count = len(target_frames)
        loss_value = pixel_value = lpips_value = psnr_value = 0.0
        scale_value = lifespan_value = slot_value = 0.0
        dynamic_pred_value = trajectory_displacement_value = 0.0
        target_psnr_values = []
        target_weight = float(record.target_weight)
        # Dynamic target frames have different Gaussian means, so this gsplat
        # version cannot batch them; keep one render graph per target.
        target_batch_size = 1
        for batch_start in range(0, target_count, target_batch_size):
            batch_frames = target_frames[batch_start : batch_start + target_batch_size]
            batch_entries = []
            batch_loss = None
            for batch_offset, target_frame in enumerate(batch_frames):
                target = target_image(
                    sample, query_source, target_frame, target_view, args, device
                )
                rendered, gaussian = model(sample, target_frame, target_view, args)
                pixel_loss = (rendered - target).abs().mean()
                lpips_loss = lpips_model(
                    rendered[None] * 2.0 - 1.0, target[None] * 2.0 - 1.0
                ).mean()
                image_loss = target_weight * (pixel_loss + args.lpips_weight * lpips_loss)
                opacity_reg = gaussian["opacities"].mean()
                scale_reg = gaussian["raw_scales"].mean()
                loss = image_loss + args.opacity_reg_weight * opacity_reg + args.scale_reg_weight * scale_reg
                slot_reg = None
                if args.object_slots > 0 and "instance_ids" in gaussian:
                    # Object-slot coherence: within each slot, per-query Gaussians should
                    # share one trajectory. Penalize the per-frame variance of the
                    # per-query track deltas (refinement deltas) inside each slot, gated by
                    # dynamic probability so static queries are not forced into slots.
                    dynamic_prob = sample.batch.query_features[:, -1].clamp(0.0, 1.0)
                    if "means_by_context" in gaussian:
                        # (N, T, 1, 3) -> (N, T, 3)
                        tracks = gaussian["means_by_context"][..., 0, :]
                        ids = gaussian["instance_ids"]
                        num_slots = int(args.object_slots)
                        per_slot_var = torch.zeros((num_slots,), device=ids.device)
                        counts = torch.zeros((num_slots,), device=ids.device)
                        for slot in range(num_slots):
                            member = (ids == slot) & (dynamic_prob > 0.2)
                            n_member = int(member.sum())
                            if n_member > 1:
                                centered = tracks[member] - tracks[member].mean(dim=0, keepdim=True)
                                per_slot_var[slot] = centered.square().mean()
                                counts[slot] = n_member
                        active = counts > 1
                        if bool(active.any()):
                            slot_reg = (per_slot_var[active] * counts[active]).sum() / counts[active].sum()
                            loss = loss + 0.01 * slot_reg
                batch_entries.append((target, rendered, gaussian, loss, pixel_loss, lpips_loss, slot_reg))
                batch_loss = loss if batch_loss is None else batch_loss + loss
            # Accumulate target gradients and synchronize DDP on the final target.
            sync_context = (
                model.no_sync()
                if batch_start + len(batch_frames) < target_count
                and isinstance(model, DistributedDataParallel)
                else nullcontext()
            )
            with sync_context:
                (batch_loss / target_count).backward()
            for target, rendered, gaussian, loss, pixel_loss, lpips_loss, slot_reg in batch_entries:
                loss_value += float(loss.detach()) / target_count
                pixel_value += float(pixel_loss.detach()) / target_count
                lpips_value += float(lpips_loss.detach()) / target_count
                target_psnr = pointforward.psnr(rendered, target)
                target_psnr_values.append(target_psnr)
                psnr_value += target_psnr / target_count
                scale_value += float(gaussian["raw_scales"].detach().mean()) / target_count
                lifespan_value += float(gaussian["lifespans"].detach().mean()) / target_count
                slot_value += float(slot_reg.detach()) / target_count if slot_reg is not None else 0.0
                dynamic_pred_value += float(
                    gaussian["dynamic_probability_pred"].detach().mean()
                ) / target_count
                if "velocity" in gaussian:
                    trajectory_displacement_value += float(
                        gaussian["velocity"].detach().norm(dim=-1).mean()
                    ) / target_count
                del target, rendered, gaussian, loss, pixel_loss, lpips_loss
            del batch_entries, batch_loss
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time_sec = time.perf_counter() - step_start
        if world_size > 1:
            step_time_tensor = torch.tensor(step_time_sec, dtype=torch.float64, device=device)
            dist.all_reduce(step_time_tensor, op=dist.ReduceOp.MAX)
            step_time_sec = float(step_time_tensor.item())
        step_time_sum_sec += step_time_sec
        step_time_min_sec = min(step_time_min_sec, step_time_sec)
        step_time_max_sec = max(step_time_max_sec, step_time_sec)
        measured_step_count += 1
        local_values = [
            loss_value,
            pixel_value,
            lpips_value,
            psnr_value,
            scale_value,
            lifespan_value,
            sample.dynamic_probability_mean,
            slot_value,
            dynamic_pred_value,
            trajectory_displacement_value,
        ]
        mean_values = reduce_metrics(local_values, device, world_size)
        mean_target_psnr_values = reduce_metrics(target_psnr_values, device, world_size)
        input_frame_psnr = reduce_optional_mean(
            [
                value
                for frame, value in zip(target_frames, target_psnr_values, strict=True)
                if frame in sample.frames
            ],
            device,
            world_size,
        )
        heldout_frame_psnr = reduce_optional_mean(
            [
                value
                for frame, value in zip(target_frames, target_psnr_values, strict=True)
                if frame not in sample.frames
            ],
            device,
            world_size,
        )
        if rank == 0:
            metric = {
                "step": step,
                "epoch": current_epoch + 1,
                "loss": mean_values[0],
                "pixel_loss": mean_values[1],
                "lpips_loss": mean_values[2],
                "target_weight": target_weight,
                "psnr": mean_values[3],
                "psnr_by_target_position": mean_target_psnr_values,
                "input_frame_psnr": input_frame_psnr,
                "heldout_frame_psnr": heldout_frame_psnr,
                "scale_mean_m": mean_values[4],
                "lifespan_mean": mean_values[5],
                "flow_dynamic_probability_mean": mean_values[6],
                "slot_reg": mean_values[7],
                "dynamic_probability_pred_mean": mean_values[8],
                "trajectory_displacement_mean_m": mean_values[9],
                "rank0_manifest_index": manifest_index,
                "rank0_scene_index": int(row["scene_index"]),
                "rank0_window_start": int(record.window_start),
                "rank0_context_mode": record.context_mode,
                "rank0_context_frames": list(record.context_frames or tuple(sample.frames)),
                "rank0_query_count": int(sample.batch.query_features.shape[0]),
                "rank0_target_frame": int(target_frames[0]),
                "rank0_target_frames": list(target_frames),
                "rank0_target_view": pointforward.VIEW_NAMES[target_view],
                "rank0_target_scope": list(record.target_scope or tuple(sample.frames)),
                "elapsed_sec": time.perf_counter() - start_time,
                "step_time_sec": step_time_sec,
                "step_time_ms": step_time_sec * 1000.0,
                "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metric) + "\n")
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(metric), flush=True)
        del sample
        # NOTE: intentionally no torch.cuda.empty_cache() here either (see the
        # stage1 note above). Keep the caching allocator resident across steps.

        epoch_number = current_epoch + 1
        epoch_finished = step % steps_per_epoch == 0
        run_validation = (
            args.val_every_epochs > 0
            and (
                (args.fast_dev_run and step == 1)
                or (epoch_finished and epoch_number % args.val_every_epochs == 0)
                or step == args.total_steps
            )
        )
        if run_validation:
            if world_size > 1:
                dist.barrier()
            validation = None
            if rank == 0:
                validation = validate_fixed_windows(
                    unwrap(model),
                    fixed_val_rows,
                    val_cache_root,
                    val_query_source,
                    lpips_model,
                    args,
                    device,
                    step,
                )
                with val_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(validation) + "\n")
                print(json.dumps({"validation": validation}), flush=True)
                if validation["psnr"] > best_val_psnr:
                    best_val_psnr = validation["psnr"]
                    best_checkpoint = args.checkpoint_dir / (
                        f"best_val_psnr_epoch_{epoch_number:04d}_step_{step:06d}.pt"
                    )
                    save_checkpoint(best_checkpoint, model, optimizer, step, validation, args)
                    (args.checkpoint_dir / "best.json").write_text(
                        json.dumps(
                            {
                                "checkpoint": str(best_checkpoint),
                                "epoch": epoch_number,
                                "step": step,
                                "psnr": best_val_psnr,
                                "validation": validation,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            if world_size > 1:
                dist.barrier()
        save_epoch = (
            epoch_finished
            and args.checkpoint_every_epochs > 0
            and epoch_number % args.checkpoint_every_epochs == 0
        )
        if rank == 0 and (save_epoch or step == args.total_steps):
            checkpoint_path = (
                args.checkpoint_dir / f"epoch_{epoch_number:04d}_step_{step:06d}.pt"
                if epoch_finished
                else args.checkpoint_dir / f"step_{step:06d}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                validation if run_validation else None,
                args,
            )

    if rank == 0:
        summary = {
            "epochs_requested": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": args.total_steps,
            "epochs_completed": args.total_steps / steps_per_epoch,
            "world_size": world_size,
            "train_scenes": len({int(row["scene_index"]) for row in train_rows}),
            "val_scenes": len({int(row["scene_index"]) for row in val_rows}),
            "train_clips": len(train_rows),
            "train_windows_per_epoch": len(train_rows) * records_this_clip,
            "queries_per_sample": args.max_total_queries,
            "train_step_time_sec": {
                "mean": step_time_sum_sec / measured_step_count
                if measured_step_count > 0
                else None,
                "min": step_time_min_sec if measured_step_count > 0 else None,
                "max": step_time_max_sec if measured_step_count > 0 else None,
                "count": measured_step_count,
            },
            "best_val_psnr": best_val_psnr,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
