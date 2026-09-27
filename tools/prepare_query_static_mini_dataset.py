#!/usr/bin/env python3
"""Build a tiny RGB-D-flow latent package for static-query experiments."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VIEW_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)
ORIGINAL_IMAGE_SIZE = (1600, 900)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--ann-file", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--clips-per-scene", type=int, default=1)
    parser.add_argument("--video-length", type=int, default=17)
    parser.add_argument("--scene-indices", default=None)
    parser.add_argument("--height", type=int, default=424)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--low-height", type=int, default=53)
    parser.add_argument("--low-width", type=int, default=100)
    parser.add_argument(
        "--masked-flow-rgb-root",
        type=Path,
        default=None,
        help="Optional root containing {scene}_scene/{camera}/dynamic_flow_gray_bg/*.png flow GT images.",
    )
    parser.add_argument(
        "--masked-flow-index",
        type=Path,
        default=None,
        help="Optional masked-flow index JSON. When provided, flow RGB is looked up by clip token instead of scene_frame_start.",
    )
    parser.add_argument(
        "--repair-existing-mini-dir",
        type=Path,
        default=None,
        help="Optional existing mini dataset. When set, copy clips and rebuild only flow_rgb_target/valid by token index.",
    )
    parser.add_argument(
        "--rebuild-existing-mini-dir",
        type=Path,
        default=None,
        help="Optional existing mini dataset whose latent tensors are reused while RGB-D targets and cameras are rebuilt.",
    )
    parser.add_argument(
        "--allow-missing-flow-rgb",
        action="store_true",
        help="Fill missing flow RGB edges with invalid white frames instead of failing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-existing-depth-target",
        action="store_true",
        help="In rebuild mode, preserve the packaged weak depth target when its external depth files are unavailable.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    num_scenes: int,
    clips_per_scene: int,
    video_length: int,
    scene_indices: set[int] | None,
) -> list[dict[str, Any]]:
    selected = []
    scene_counts: dict[int, int] = {}
    for row in rows:
        if int(row.get("video_length", -1)) != video_length:
            continue
        scene_index = int(row["scene_index"])
        if scene_indices is not None and scene_index not in scene_indices:
            continue
        if scene_index not in scene_counts and len(scene_counts) >= num_scenes:
            continue
        if scene_counts.get(scene_index, 0) >= clips_per_scene:
            continue
        selected.append(row)
        scene_counts[scene_index] = scene_counts.get(scene_index, 0) + 1
        if len(scene_counts) >= num_scenes and all(
            count >= clips_per_scene for count in scene_counts.values()
        ):
            break
    if scene_indices is not None:
        # Explicit scene list: keep every available clip for the requested scenes
        # (clips_per_scene acts as a per-scene cap, not a required exact count).
        expected_clips = len(scene_indices)
        if len(scene_counts) < expected_clips:
            raise RuntimeError(
                f"Only found {len(scene_counts)} requested scenes out of {len(scene_indices)} "
                f"for video_length={video_length}; scene_counts={scene_counts}."
            )
    else:
        expected_clips = num_scenes * clips_per_scene
        if len(selected) < expected_clips:
            raise RuntimeError(
                f"Only found {len(selected)} clips for num_scenes={num_scenes}, "
                f"clips_per_scene={clips_per_scene}, video_length={video_length}, "
                f"scene_indices={scene_indices}, scene_counts={scene_counts}."
            )
    return selected


def load_ann(path: Path) -> dict[str, Any]:
    try:
        import mmcv

        return mmcv.load(str(path))
    except Exception:
        with path.open("rb") as file:
            return pickle.load(file)


def quaternion_to_rotation_matrix(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_matrix(rotation: Any, translation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quaternion_to_rotation_matrix(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def camera_to_global(camera_info: dict[str, Any]) -> np.ndarray:
    camera_to_ego = pose_matrix(
        camera_info["sensor2ego_rotation"],
        camera_info["sensor2ego_translation"],
    )
    ego_to_global = pose_matrix(
        camera_info["ego2global_rotation"],
        camera_info["ego2global_translation"],
    )
    return ego_to_global @ camera_to_ego


def lidar_to_global(frame_info: dict[str, Any]) -> np.ndarray:
    lidar_to_ego = pose_matrix(
        frame_info["lidar2ego_rotation"],
        frame_info["lidar2ego_translation"],
    )
    ego_to_global = pose_matrix(
        frame_info["ego2global_rotation"],
        frame_info["ego2global_translation"],
    )
    return ego_to_global @ lidar_to_ego


def camera_to_lidar(camera_info: dict[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(camera_info["sensor2lidar_rotation"], dtype=np.float32)
    matrix[:3, 3] = np.asarray(camera_info["sensor2lidar_translation"], dtype=np.float32)
    return matrix


def resized_intrinsics(camera_info: dict[str, Any], width: int, height: int) -> np.ndarray:
    intrinsics = np.asarray(camera_info["camera_intrinsics"], dtype=np.float32).copy()
    intrinsics[0] *= width / ORIGINAL_IMAGE_SIZE[0]
    intrinsics[1] *= height / ORIGINAL_IMAGE_SIZE[1]
    return intrinsics


def resolve_data_path(path: str, data_root: Path) -> Path:
    prefixes = ("../data/nuscenes/", "data/nuscenes/", "nuscenes/")
    for prefix in prefixes:
        if path.startswith(prefix):
            return data_root / path[len(prefix) :]
    candidate = Path(path)
    return candidate if candidate.is_absolute() else data_root / candidate


def read_rgb(camera_info: dict[str, Any], data_root: Path, width: int, height: int) -> np.ndarray:
    image_path = resolve_data_path(camera_info["data_path"], data_root)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)


def read_depth(
    depth_root: Path,
    depth_map: dict[str, str],
    token: str,
    camera: str,
    width: int,
    height: int,
) -> np.ndarray:
    depth_entry = Path(depth_map[token])
    sample_root = depth_entry if depth_entry.is_absolute() else depth_root / str(depth_entry).lstrip("/")
    depth_path = sample_root / camera / "refined_depth.npz"
    data = np.load(depth_path)
    depth = data["depth_pred"].astype(np.float32)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
    depth = np.where(np.isfinite(depth), np.maximum(depth, 0.0), 0.0)
    return depth.astype(np.float32)


def read_masked_flow_rgb(
    root: Path,
    scene_index: int,
    scene_frame_start: int,
    frame_count: int,
    width: int,
    height: int,
    clip_tokens: list[str] | None = None,
    masked_flow_index: dict[str, Any] | None = None,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    flow_rgb = np.full((frame_count, len(VIEW_ORDER), height, width, 3), 255, dtype=np.uint8)
    flow_valid = np.zeros((frame_count, len(VIEW_ORDER), height, width), dtype=np.uint8)
    flow_edges: list[dict[str, Any]] = []
    for frame_index in range(frame_count - 1):
        for view_index, camera in enumerate(VIEW_ORDER):
            if masked_flow_index is None:
                flow_scene_index = scene_index
                flow_frame_index = scene_frame_start + frame_index
            else:
                if clip_tokens is None:
                    raise ValueError("clip_tokens is required with masked_flow_index")
                token = clip_tokens[frame_index]
                expected_target_token = clip_tokens[frame_index + 1]
                hits = []
                for flow_scene_key, scene in masked_flow_index["scenes"].items():
                    view_edges = scene.get(camera, {})
                    if token in view_edges:
                        edge = view_edges[token]
                        hits.append((int(flow_scene_key), int(edge[0]), str(edge[1])))
                if not hits:
                    if allow_missing:
                        continue
                    raise RuntimeError(
                        f"Missing masked-flow index edge for scene={scene_index} "
                        f"start={scene_frame_start} frame={frame_index} view={camera} token={token}"
                    )
                if len(hits) > 1:
                    raise RuntimeError(
                        f"Ambiguous masked-flow index edge for scene={scene_index} "
                        f"start={scene_frame_start} frame={frame_index} view={camera} token={token}: {hits}"
                    )
                flow_scene_index, flow_frame_index, target_token = hits[0]
                if target_token != expected_target_token:
                    raise RuntimeError(
                        f"Masked-flow target mismatch for source={token} view={camera}: "
                        f"index target={target_token}, expected={expected_target_token}"
                    )
            image_path = (
                root
                / f"{flow_scene_index}_scene"
                / camera
                / "dynamic_flow_gray_bg"
                / f"dynamic_flow_gray_bg_{flow_frame_index:06d}.png"
            )
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                if allow_missing:
                    continue
                raise FileNotFoundError(image_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            flow_rgb[frame_index, view_index] = rgb.astype(np.uint8)
            flow_valid[frame_index, view_index] = 1
            flow_edges.append(
                {
                    "frame": int(frame_index),
                    "view": camera,
                    "source_token": str(clip_tokens[frame_index]) if clip_tokens is not None else None,
                    "target_token": str(clip_tokens[frame_index + 1]) if clip_tokens is not None else None,
                    "flow_scene": int(flow_scene_index),
                    "flow_frame": int(flow_frame_index),
                    "image_path": str(image_path),
                }
            )
    return flow_rgb, flow_valid, flow_edges


def compute_edge_low(rgb_target: np.ndarray, depth_target: np.ndarray, low_width: int, low_height: int) -> np.ndarray:
    frame_count, view_count = rgb_target.shape[:2]
    edge_low = np.empty((frame_count, view_count, low_height, low_width), dtype=np.float32)
    for frame_index in range(frame_count):
        for view_index in range(view_count):
            gray = cv2.cvtColor(rgb_target[frame_index, view_index], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            log_depth = np.log1p(depth_target[frame_index, view_index].astype(np.float32))
            rgb_dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            rgb_dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            depth_dx = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
            depth_dy = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.sqrt(rgb_dx * rgb_dx + rgb_dy * rgb_dy) + 0.25 * np.sqrt(
                depth_dx * depth_dx + depth_dy * depth_dy
            )
            edge = cv2.resize(edge, (low_width, low_height), interpolation=cv2.INTER_AREA)
            max_value = float(np.percentile(edge, 99.5))
            if max_value > 1.0e-6:
                edge = np.clip(edge / max_value, 0.0, 1.0)
            edge_low[frame_index, view_index] = edge
    return edge_low


def resize_depth_low(depth_target: np.ndarray, low_width: int, low_height: int) -> np.ndarray:
    frame_count, view_count = depth_target.shape[:2]
    depth_low = np.empty((frame_count, view_count, low_height, low_width), dtype=np.float32)
    for frame_index in range(frame_count):
        for view_index in range(view_count):
            depth_low[frame_index, view_index] = cv2.resize(
                depth_target[frame_index, view_index],
                (low_width, low_height),
                interpolation=cv2.INTER_AREA,
            )
    return depth_low


def upsample_dynamic_mask(flow_loss_mask: torch.Tensor, video_length: int, height: int, width: int) -> np.ndarray:
    mask = flow_loss_mask.float()
    if mask.ndim != 5:
        raise ValueError(f"Expected flow_loss_mask [V,1,Tz,H,W], got {tuple(mask.shape)}")
    mask = F.interpolate(
        mask,
        size=(video_length, height, width),
        mode="trilinear",
        align_corners=False,
    )
    mask = mask[:, 0].permute(1, 0, 2, 3).contiguous()
    return mask.cpu().numpy().astype(np.float32)


def cog_down_frame_groups(groups: list[list[int]]) -> list[list[int]]:
    if len(groups) % 2 == 1:
        return [groups[0]] + [groups[index] + groups[index + 1] for index in range(1, len(groups), 2)]
    return [groups[index] + groups[index + 1] for index in range(0, len(groups), 2)]


def latent_time_frame_groups(video_length: int, latent_time_length: int) -> list[list[int]]:
    groups = [[index] for index in range(video_length)]
    while len(groups) > latent_time_length:
        groups = cog_down_frame_groups(groups)
    if len(groups) != latent_time_length:
        raise RuntimeError(
            f"Could not map video_length={video_length} to latent_time_length={latent_time_length}; "
            f"ended with {len(groups)} groups."
        )
    return groups


def build_clip(
    row: dict[str, Any],
    payload: dict[str, Any],
    infos: list[dict[str, Any]],
    depth_map: dict[str, str],
    args: argparse.Namespace,
    masked_flow_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_token = str(row["token"])
    payload_token = str(payload.get("token", row_token))
    if payload_token != row_token:
        raise RuntimeError(
            f"Latent payload token {payload_token} does not match manifest row token {row_token}"
        )
    frame_indices = [int(index) for index in payload.get("ann_frame_indices", [])]
    frame_indices_match_row = (
        bool(frame_indices)
        and 0 <= frame_indices[0] < len(infos)
        and str(infos[frame_indices[0]]["token"]) == str(row["token"])
    )
    if not frame_indices_match_row:
        start = next(
            (
                index
                for index, info in enumerate(infos)
                if str(info["token"]) == str(row["token"])
            ),
            None,
        )
        if start is None:
            raise KeyError(f"Could not find clip token {row['token']} in annotation infos")
        frame_indices = list(range(start, start + int(row["video_length"])))
    if frame_indices[-1] >= len(infos):
        raise IndexError(
            f"Annotation frame range {frame_indices[0]}..{frame_indices[-1]} exceeds "
            f"the {len(infos)} available infos for token {row['token']}"
        )
    frame_infos = [infos[index] for index in frame_indices]
    clip_tokens = [info["token"] for info in frame_infos]
    if str(clip_tokens[0]) != row_token:
        raise RuntimeError(
            f"Annotation clip starts at {clip_tokens[0]}, expected latent row token {row_token}"
        )

    frame_count = len(frame_infos)
    view_count = len(VIEW_ORDER)
    rgb_target = np.empty((frame_count, view_count, args.height, args.width, 3), dtype=np.uint8)
    depth_target = np.empty((frame_count, view_count, args.height, args.width), dtype=np.float32)
    camera_param = np.empty((frame_count, view_count, 3, 7), dtype=np.float32)
    camera_intrinsics = np.empty((frame_count, view_count, 3, 3), dtype=np.float32)
    camera2lidar = np.empty((frame_count, view_count, 4, 4), dtype=np.float32)
    lidar2camera = np.empty((frame_count, view_count, 4, 4), dtype=np.float32)
    camera2global = np.empty((frame_count, view_count, 4, 4), dtype=np.float32)
    lidar2global = np.empty((frame_count, 4, 4), dtype=np.float32)

    depth_root = Path(payload["rdepth_root"])
    for frame_index, frame_info in enumerate(frame_infos):
        lidar2global[frame_index] = lidar_to_global(frame_info)
        for view_index, camera in enumerate(VIEW_ORDER):
            camera_info = frame_info["cams"][camera]
            intrinsics = resized_intrinsics(camera_info, args.width, args.height)
            c2l = camera_to_lidar(camera_info)
            camera_intrinsics[frame_index, view_index] = intrinsics
            camera2lidar[frame_index, view_index] = c2l
            lidar2camera[frame_index, view_index] = np.linalg.inv(c2l).astype(np.float32)
            camera2global[frame_index, view_index] = camera_to_global(camera_info)
            camera_param[frame_index, view_index] = np.concatenate([intrinsics, c2l[:3]], axis=-1)
            rgb_target[frame_index, view_index] = read_rgb(camera_info, args.data_root, args.width, args.height)
            if "depth_target_override" not in payload:
                depth_target[frame_index, view_index] = read_depth(
                    depth_root,
                    depth_map,
                    frame_info["token"],
                    camera,
                    args.width,
                    args.height,
                )

    if "depth_target_override" in payload:
        depth_override = torch.as_tensor(payload["depth_target_override"]).float().cpu().numpy()
        if depth_override.shape != depth_target.shape:
            raise ValueError(
                f"depth_target_override has shape {depth_override.shape}, expected {depth_target.shape}"
            )
        depth_target[...] = depth_override

    ref_from_global = np.linalg.inv(lidar2global[0]).astype(np.float32)
    frame_to_ref_lidar = np.einsum("ij,tjk->tik", ref_from_global, lidar2global).astype(np.float32)
    dynamic_mask_target = upsample_dynamic_mask(
        payload["flow_loss_mask"],
        frame_count,
        args.height,
        args.width,
    )
    static_mask_target = (dynamic_mask_target < 0.5).astype(np.uint8)
    dynamic_mask_low = np.empty((frame_count, view_count, args.low_height, args.low_width), dtype=np.float32)
    static_mask_low = np.empty_like(dynamic_mask_low)
    for frame_index in range(frame_count):
        for view_index in range(view_count):
            low = cv2.resize(
                dynamic_mask_target[frame_index, view_index],
                (args.low_width, args.low_height),
                interpolation=cv2.INTER_NEAREST,
            )
            dynamic_mask_low[frame_index, view_index] = low
            static_mask_low[frame_index, view_index] = (low < 0.5).astype(np.float32)

    depth_low = resize_depth_low(depth_target, args.low_width, args.low_height)
    edge_low = compute_edge_low(rgb_target, depth_target, args.low_width, args.low_height)
    valid_depth_low = (
        np.isfinite(depth_low)
        & (depth_low > 0.1)
        & (depth_low < 99.5)
    ).astype(np.uint8)
    flow_rgb_target = None
    flow_rgb_valid_target = None
    flow_rgb_edges: list[dict[str, Any]] = []
    if args.masked_flow_rgb_root is not None:
        flow_rgb_target, flow_rgb_valid_target, flow_rgb_edges = read_masked_flow_rgb(
            args.masked_flow_rgb_root,
            int(row["scene_index"]),
            int(row.get("scene_frame_start", 0)),
            frame_count,
            args.width,
            args.height,
            clip_tokens=clip_tokens,
            masked_flow_index=masked_flow_index,
            allow_missing=args.allow_missing_flow_rgb,
        )

    tensors = {
        "latent": payload["latent"].half().cpu(),
        "rgb_latent": payload["rgb_latent"].half().cpu(),
        "depth_latent": payload["depth_latent"].half().cpu(),
        "flow_latent": payload["flow_latent"].half().cpu(),
        "flow_loss_mask_latent": payload["flow_loss_mask"].half().cpu(),
        "rgb_target": torch.from_numpy(rgb_target),
        "depth_target": torch.from_numpy(depth_target).half(),
        "depth_low": torch.from_numpy(depth_low).half(),
        "edge_low": torch.from_numpy(edge_low).half(),
        "dynamic_mask_low": torch.from_numpy(dynamic_mask_low).half(),
        "static_mask_low": torch.from_numpy(static_mask_low.astype(np.uint8)),
        "static_mask_target": torch.from_numpy(static_mask_target),
        "valid_depth_low": torch.from_numpy(valid_depth_low),
        "camera_param": torch.from_numpy(camera_param),
        "camera_intrinsics": torch.from_numpy(camera_intrinsics),
        "camera2lidar": torch.from_numpy(camera2lidar),
        "lidar2camera": torch.from_numpy(lidar2camera),
        "camera2global": torch.from_numpy(camera2global),
        "lidar2global": torch.from_numpy(lidar2global),
        "frame_to_ref_lidar": torch.from_numpy(frame_to_ref_lidar),
    }
    if flow_rgb_target is not None and flow_rgb_valid_target is not None:
        tensors["flow_rgb_target"] = torch.from_numpy(flow_rgb_target)
        tensors["flow_rgb_valid_target"] = torch.from_numpy(flow_rgb_valid_target)
    meta = {
        "dataset_index": int(row["dataset_index"]),
        "scene_index": int(row["scene_index"]),
        "scene_frame_start": int(row.get("scene_frame_start", 0)),
        "token": row["token"],
        "clip_tokens": clip_tokens,
        "ann_frame_indices": frame_indices,
        "video_length": frame_count,
        "views": list(VIEW_ORDER),
        "image_hw": [args.height, args.width],
        "low_hw": [args.low_height, args.low_width],
        "latent_shape": list(payload["latent"].shape),
        "latent_time_frame_groups": latent_time_frame_groups(
            frame_count,
            int(payload["latent"].shape[2]),
        ),
        "reference_frame": 0,
        "reference_coordinate": "frame0 lidar",
        "camera_param_format": "3x3 resized intrinsics concatenated with current-frame camera2lidar 3x4",
        "frame_to_ref_lidar_format": "column-vector transform from current frame lidar coordinates to frame0 lidar coordinates",
        "static_mask_source": "flow_loss_mask upsampled from latent grid; static is mask < 0.5",
        "edge_low_source": "Sobel RGB edge plus 0.25 * Sobel log-depth edge, normalized per frame/view",
        "flow_rgb_target_source": (
            str(args.masked_flow_rgb_root)
            if args.masked_flow_rgb_root is not None
            else None
        ),
        "flow_rgb_target_format": (
            "masked optical-flow RGB image, white background, frame t means token[t]->token[t+1], last frame invalid"
            if args.masked_flow_rgb_root is not None
            else None
        ),
        "flow_rgb_target_index": str(args.masked_flow_index) if args.masked_flow_index is not None else None,
        "flow_rgb_edges": flow_rgb_edges,
        "source_identity_audit": {
            "manifest_token": row_token,
            "latent_payload_token": payload_token,
            "annotation_first_token": str(clip_tokens[0]),
            "tokens_match": True,
            "flow_edge_count": len(flow_rgb_edges),
            "expected_flow_edge_count": (frame_count - 1) * len(VIEW_ORDER),
            "flow_edges_complete": len(flow_rgb_edges) == (frame_count - 1) * len(VIEW_ORDER),
        },
        "source_cache_path": row["path"],
        "source_rgbd_latent_path": row.get("source_rgbd_latent_path"),
        "source_flow_latent_path": row.get("source_flow_latent_path"),
        "ann_file": str(args.ann_file),
        "depth_map_json": payload.get("depth_map_json"),
        "rdepth_root": payload.get("rdepth_root"),
        "depth_target_reused": "depth_target_override" in payload,
    }
    return {"tensors": tensors, "meta": meta}


def repair_existing_mini_flow_rgb(args: argparse.Namespace) -> None:
    if args.repair_existing_mini_dir is None:
        raise ValueError("--repair-existing-mini-dir is required for repair mode")
    if args.masked_flow_rgb_root is None:
        raise ValueError("--masked-flow-rgb-root is required for repair mode")
    if args.masked_flow_index is None:
        raise ValueError("--masked-flow-index is required for repair mode")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    source_dir = args.repair_existing_mini_dir
    source_manifest = source_dir / "manifest.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    rows = read_manifest(source_manifest)
    masked_flow_index = load_json(args.masked_flow_index)

    manifest_rows = []
    total_valid = 0
    total_possible = 0
    clips_summary = []
    for row in rows:
        source_clip = Path(row["clip_pt"])
        source_meta = Path(row["meta_json"])
        clip = torch.load(source_clip, map_location="cpu")
        meta = load_json(source_meta)
        frame_count = int(meta["video_length"])
        height, width = [int(value) for value in meta["image_hw"]]
        flow_rgb, flow_valid, flow_edges = read_masked_flow_rgb(
            args.masked_flow_rgb_root,
            int(meta["scene_index"]),
            int(meta.get("scene_frame_start", 0)),
            frame_count,
            width,
            height,
            clip_tokens=[str(token) for token in meta["clip_tokens"]],
            masked_flow_index=masked_flow_index,
            allow_missing=args.allow_missing_flow_rgb,
        )
        clip["flow_rgb_target"] = torch.from_numpy(flow_rgb)
        clip["flow_rgb_valid_target"] = torch.from_numpy(flow_valid)

        relative_name = Path(row["clip_dir"]).name
        clip_dir = args.output_dir / relative_name
        clip_dir.mkdir(parents=True)
        torch.save(clip, clip_dir / "clip.pt")

        possible = (frame_count - 1) * len(VIEW_ORDER)
        valid_edges = int(flow_valid[: frame_count - 1].reshape(frame_count - 1, len(VIEW_ORDER), -1).max(axis=2).sum())
        total_valid += valid_edges
        total_possible += possible

        meta["flow_rgb_target_source"] = str(args.masked_flow_rgb_root)
        meta["flow_rgb_target_index"] = str(args.masked_flow_index)
        meta["flow_rgb_target_format"] = "masked optical-flow RGB image, white background, frame t means token[t]->token[t+1], last frame invalid"
        meta["flow_rgb_target_repaired_from"] = str(source_dir)
        meta["flow_rgb_edges"] = flow_edges
        meta["flow_rgb_valid_edges"] = valid_edges
        meta["flow_rgb_possible_edges"] = possible
        dump_json(clip_dir / "meta.json", meta)

        manifest_row = dict(row)
        manifest_row["clip_dir"] = str(clip_dir)
        manifest_row["clip_pt"] = str(clip_dir / "clip.pt")
        manifest_row["meta_json"] = str(clip_dir / "meta.json")
        manifest_rows.append(manifest_row)
        clips_summary.append(
            {
                "clip_dir": str(clip_dir),
                "scene_index": int(meta["scene_index"]),
                "scene_frame_start": int(meta.get("scene_frame_start", 0)),
                "flow_rgb_valid_edges": valid_edges,
                "flow_rgb_possible_edges": possible,
            }
        )

    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for row in manifest_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_summary_path = source_dir / "summary.json"
    summary = load_json(source_summary_path) if source_summary_path.exists() else {}
    summary.update(
        {
            "output_dir": str(args.output_dir),
            "repair_existing_mini_dir": str(source_dir),
            "flow_rgb_target_source": str(args.masked_flow_rgb_root),
            "flow_rgb_target_index": str(args.masked_flow_index),
            "flow_rgb_target_format": "masked optical-flow RGB image, white background, frame t means token[t]->token[t+1], last frame invalid",
            "num_clips": len(manifest_rows),
            "flow_rgb_valid_edges": total_valid,
            "flow_rgb_possible_edges": total_possible,
            "clips": manifest_rows,
            "repair_clips": clips_summary,
        }
    )
    dump_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def rebuild_existing_mini(args: argparse.Namespace) -> None:
    """Rebuild selected clips while preserving their already-computed latent tensors."""
    source_dir = args.rebuild_existing_mini_dir
    if source_dir is None:
        raise ValueError("--rebuild-existing-mini-dir is required for rebuild mode")
    if args.ann_file is None or args.data_root is None:
        raise ValueError("--ann-file and --data-root are required for rebuild mode")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    source_manifest = source_dir / "manifest.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    source_rows = read_manifest(source_manifest)
    enriched_rows = []
    for source_row in source_rows:
        meta = load_json(Path(source_row["meta_json"]))
        enriched_row = dict(source_row)
        enriched_row.update(
            {
                "dataset_index": int(meta["dataset_index"]),
                "scene_index": int(meta["scene_index"]),
                "scene_frame_start": int(meta.get("scene_frame_start", 0)),
                "token": str(meta["token"]),
                "video_length": int(meta["video_length"]),
                "source_meta": meta,
            }
        )
        enriched_rows.append(enriched_row)

    scene_indices = None
    if args.scene_indices:
        scene_indices = {int(value.strip()) for value in args.scene_indices.split(",") if value.strip()}
    selected_rows = select_rows(
        enriched_rows,
        args.num_scenes,
        args.clips_per_scene,
        args.video_length,
        scene_indices,
    )
    infos = load_ann(args.ann_file)["infos"]
    masked_flow_index = load_json(args.masked_flow_index) if args.masked_flow_index is not None else None

    manifest_rows = []
    for mini_index, source_row in enumerate(selected_rows):
        source_clip = torch.load(Path(source_row["clip_pt"]), map_location="cpu")
        source_meta = source_row["source_meta"]
        payload = {
            "latent": source_clip["latent"],
            "rgb_latent": source_clip["rgb_latent"],
            "depth_latent": source_clip["depth_latent"],
            "flow_latent": source_clip["flow_latent"],
            "flow_loss_mask": source_clip["flow_loss_mask_latent"],
            "ann_frame_indices": source_meta.get("ann_frame_indices", []),
            "depth_map_json": source_meta["depth_map_json"],
            "rdepth_root": source_meta["rdepth_root"],
        }
        if args.reuse_existing_depth_target:
            payload["depth_target_override"] = source_clip["depth_target"]
        row = {
            "dataset_index": int(source_row["dataset_index"]),
            "scene_index": int(source_row["scene_index"]),
            "scene_frame_start": int(source_row.get("scene_frame_start", 0)),
            "token": str(source_row["token"]),
            "video_length": int(source_row["video_length"]),
            "path": str(source_meta.get("source_cache_path", source_row["clip_pt"])),
            "source_rgbd_latent_path": source_meta.get("source_rgbd_latent_path"),
            "source_flow_latent_path": source_meta.get("source_flow_latent_path"),
        }
        depth_map = (
            {}
            if args.reuse_existing_depth_target
            else load_json(Path(payload["depth_map_json"]))
        )
        rebuilt = build_clip(row, payload, infos, depth_map, args, masked_flow_index)
        rebuilt["meta"]["rebuilt_from"] = str(source_row["clip_pt"])

        clip_name = f"scene{row['scene_index']:04d}_start{row['scene_frame_start']:04d}"
        clip_dir = args.output_dir / clip_name
        clip_dir.mkdir(parents=True)
        torch.save(rebuilt["tensors"], clip_dir / "clip.pt")
        dump_json(clip_dir / "meta.json", rebuilt["meta"])
        manifest_rows.append(
            {
                "mini_index": mini_index,
                "clip_dir": str(clip_dir),
                "clip_pt": str(clip_dir / "clip.pt"),
                "meta_json": str(clip_dir / "meta.json"),
                "scene_index": row["scene_index"],
                "scene_frame_start": row["scene_frame_start"],
                "token": row["token"],
                "video_length": row["video_length"],
                "latent_shape": list(rebuilt["tensors"]["latent"].shape),
                "image_hw": [args.height, args.width],
                "low_hw": [args.low_height, args.low_width],
            }
        )

    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for row in manifest_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "output_dir": str(args.output_dir),
        "rebuild_existing_mini_dir": str(source_dir),
        "ann_file": str(args.ann_file),
        "data_root": str(args.data_root),
        "num_clips": len(manifest_rows),
        "clips": manifest_rows,
    }
    dump_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.rebuild_existing_mini_dir is not None:
        rebuild_existing_mini(args)
        return
    if args.repair_existing_mini_dir is not None:
        repair_existing_mini_flow_rgb(args)
        return
    if args.manifest is None or args.ann_file is None or args.data_root is None:
        raise ValueError("--manifest, --ann-file, and --data-root are required outside repair mode")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    scene_indices = None
    if args.scene_indices:
        scene_indices = {int(value.strip()) for value in args.scene_indices.split(",") if value.strip()}

    rows = read_manifest(args.manifest)
    selected_rows = select_rows(
        rows,
        args.num_scenes,
        args.clips_per_scene,
        args.video_length,
        scene_indices,
    )
    ann = load_ann(args.ann_file)
    infos = ann["infos"]
    masked_flow_index = load_json(args.masked_flow_index) if args.masked_flow_index is not None else None

    manifest_rows = []
    for mini_index, row in enumerate(selected_rows):
        payload = torch.load(row["path"], map_location="cpu")
        depth_map = load_json(Path(payload["depth_map_json"]))
        clip = build_clip(row, payload, infos, depth_map, args, masked_flow_index)
        clip_name = f"scene{int(row['scene_index']):04d}_start{int(row.get('scene_frame_start', 0)):04d}"
        clip_dir = args.output_dir / clip_name
        clip_dir.mkdir(parents=True)
        torch.save(clip["tensors"], clip_dir / "clip.pt")
        dump_json(clip_dir / "meta.json", clip["meta"])
        manifest_row = {
            "mini_index": mini_index,
            "clip_dir": str(clip_dir),
            "clip_pt": str(clip_dir / "clip.pt"),
            "meta_json": str(clip_dir / "meta.json"),
            "scene_index": int(row["scene_index"]),
            "scene_frame_start": int(row.get("scene_frame_start", 0)),
            "token": row["token"],
            "video_length": int(row["video_length"]),
            "latent_shape": list(clip["tensors"]["latent"].shape),
            "image_hw": [args.height, args.width],
            "low_hw": [args.low_height, args.low_width],
        }
        manifest_rows.append(manifest_row)

    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for row in manifest_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "output_dir": str(args.output_dir),
        "manifest": str(args.manifest),
        "ann_file": str(args.ann_file),
        "data_root": str(args.data_root),
        "num_clips": len(manifest_rows),
        "clips": manifest_rows,
        "fields": {
            "latent": "[6,48,Tz,Hz,Wz] clean RGB-D-flow latent",
            "depth_low": "[T,6,low_h,low_w] anchor depth",
            "edge_low": "[T,6,low_h,low_w] anchor edge score",
            "static_mask_low": "[T,6,low_h,low_w] anchor static mask",
            "rgb_target": "[T,6,H,W,3] uint8 RGB render target",
            "depth_target": "[T,6,H,W] fp16 weak depth target",
            "flow_rgb_target": "[T,6,H,W,3] uint8 masked optical-flow RGB target, optional",
            "flow_rgb_valid_target": "[T,6,H,W] uint8 valid mask for forward-flow target, optional",
            "camera_param": "[T,6,3,7] resized intrinsics + camera2lidar",
            "frame_to_ref_lidar": "[T,4,4] current lidar to frame0 lidar",
        },
    }
    dump_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
