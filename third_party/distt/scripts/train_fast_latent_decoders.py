from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import sys

sys.path.append(".")

from DISTT.utils.fast_latent_decoder import (  # noqa: E402
    FastLatentVideoDecoder,
    build_fast_depth_decoder,
    build_fast_flow_rgb_decoder,
)


VIEW_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)
VAE_OUT_CHANNELS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fast depth/flow RGB decoders from cached CogVideoX latents."
    )
    parser.add_argument("--manifest", action="append", required=True, help="Latent cache manifest jsonl.")
    parser.add_argument("--ann-file", required=True, help="NuScenes annotation pkl with scene_tokens.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--modalities", nargs="+", choices=("depth", "flow"), default=["depth", "flow"])
    parser.add_argument(
        "--depth-root-json",
        default=os.environ.get("DEPTH_ROOT_JSON", "data/nus_sampletoken2depthroot.json"),
    )
    parser.add_argument("--rdepth-root", default=os.environ.get("RDEPTH_ROOT", "data/nus_Rdepth"))
    parser.add_argument("--flow-root", default=None, help="Raw optical-flow array root. Required for raw flow training.")
    parser.add_argument(
        "--flow-mask-root",
        default=None,
        help="Dynamic mask root for raw flow training. Defaults to --flow-root when omitted.",
    )
    parser.add_argument(
        "--flow-target-source",
        choices=("raw", "vae_decode", "flow_rgb_image"),
        default="raw",
        help="Use raw optical-flow files, CogVideoX VAE-decoded flow latents, or pre-rendered flow RGB PNGs as supervision.",
    )
    parser.add_argument(
        "--flow-image-root",
        default=None,
        help="Pre-rendered white-background flow RGB PNG root. Required for --flow-target-source flow_rgb_image.",
    )
    parser.add_argument(
        "--flow-image-subdir",
        default="dynamic_flow_gray_bg",
        help="Per-camera subdirectory containing flow RGB PNGs.",
    )
    parser.add_argument(
        "--flow-image-prefix",
        default="dynamic_flow_gray_bg",
        help="Flow RGB PNG filename prefix before the zero-padded frame index.",
    )
    parser.add_argument(
        "--drop-missing-flow-images",
        action="store_true",
        help="Drop clips whose pre-rendered flow RGB PNG supervision is incomplete before building the dataloader.",
    )
    parser.add_argument(
        "--vae-pretrained",
        default=str(Path(os.environ.get("DRIVE2GAUSS_PRETRAINED_ROOT", "pretrained")) / "CogVideoX-2b"),
    )
    parser.add_argument("--vae-subfolder", default="vae")
    parser.add_argument(
        "--flow-white-threshold",
        type=float,
        default=0.9,
        help="Pixels with all decoded flow RGB channels above this threshold are treated as static.",
    )
    parser.add_argument("--height", type=int, default=424)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--flow-scale-px", type=float, default=64.0)
    parser.add_argument("--use-pixels-per-second", action="store_true")
    parser.add_argument("--allow-missing-flow", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Clips per GPU.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--decoder-architecture",
        choices=("framewise_2d", "temporal_interval"),
        default="framewise_2d",
        help="Fast decoder backbone. temporal_interval uses learnable interval-wise temporal expansion.",
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--temporal-blocks", type=int, default=3)
    parser.add_argument("--temporal-upsample-factor", type=int, default=4)
    parser.add_argument("--temporal-refine-blocks", type=int, default=2)
    parser.add_argument("--spatial-3d-blocks", type=int, default=1)
    parser.add_argument("--spatial-2d-blocks", type=int, default=1)
    parser.add_argument("--spatial-frame-chunk-size", type=int, default=16)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute decoder blocks during backward to reduce activation memory.",
    )
    parser.add_argument("--depth-gradient-weight", type=float, default=0.1)
    parser.add_argument("--flow-dynamic-weight", type=float, default=2.0)
    parser.add_argument(
        "--flow-temporal-gradient-weight",
        type=float,
        default=0.0,
        help="Weight for temporal-gradient L1 loss on flow RGB predictions.",
    )
    parser.add_argument(
        "--flow-edge-weight",
        type=float,
        default=0.0,
        help="Weight for dynamic-mask edge L1 loss on flow RGB predictions.",
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load one batch and run one forward/backward step.")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def read_manifest(path: str | os.PathLike) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(manifest_paths: list[str], limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for manifest_path in manifest_paths:
        rows.extend(read_manifest(manifest_path))
    rows.sort(key=lambda row: (int(row["video_length"]), int(row["dataset_index"])))
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_scene_tokens(ann_file: str) -> list[list[str]]:
    data = mmcv.load(ann_file)
    return data["scene_tokens"]


def clip_tokens_for_row(row: dict, scene_tokens: list[list[str]]) -> list[str]:
    scene_index = int(row["scene_index"])
    start = int(row["scene_frame_start"])
    video_length = int(row["video_length"])
    tokens = scene_tokens[scene_index][start : start + video_length]
    if len(tokens) != video_length:
        raise RuntimeError(
            f"Row dataset_index={row.get('dataset_index')} needs {video_length} frames, "
            f"but scene {scene_index}:{start} only returns {len(tokens)}."
        )
    return tokens


def make_colorwheel() -> np.ndarray:
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
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


def flow_to_image_fixed_scale(flow: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), 1.0e-6)
    u = flow[:, :, 0] / scale
    v = flow[:, :, 1] / scale
    rad = np.sqrt(np.square(u) + np.square(v))
    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    angle = np.arctan2(-v, -u) / np.pi
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    for channel in range(3):
        color0 = colorwheel[k0, channel] / 255.0
        color1 = colorwheel[k1, channel] / 255.0
        color = (1.0 - f) * color0 + f * color1
        inside = rad <= 1.0
        color[inside] = 1.0 - rad[inside] * (1.0 - color[inside])
        color[~inside] *= 0.75
        image[:, :, channel] = np.floor(255.0 * color)
    return image


def flow_from_array(data: np.lib.npyio.NpzFile, use_pixels_per_second: bool) -> np.ndarray:
    if use_pixels_per_second and "optical_flow_per_second" in data.files:
        return data["optical_flow_per_second"].astype(np.float32)
    if use_pixels_per_second and "dt_seconds" in data.files:
        dt_seconds = max(float(data["dt_seconds"]), 1.0e-6)
        return data["optical_flow"].astype(np.float32) / dt_seconds
    return data["optical_flow"].astype(np.float32)


def build_teacher_vae(args: argparse.Namespace, device: torch.device):
    try:
        import transformers

        if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
            transformers.EncoderDecoderCache = transformers.DynamicCache
    except Exception:
        pass

    vae_path = Path(__file__).resolve().parents[1] / "DISTT" / "models" / "vae" / "vae_cogvideox.py"
    spec = importlib.util.spec_from_file_location("vae_cogvideox_teacher", vae_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load VAE module from {vae_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vae = module.VideoAutoencoderKLCogVideoX(
        from_pretrained=args.vae_pretrained,
        subfolder=args.vae_subfolder,
        micro_frame_size=None,
        micro_batch_size=1,
    )
    return vae.to(device, torch.float16).eval()


def build_flow_index(flow_root: str | os.PathLike) -> dict:
    flow_root = Path(flow_root)
    scenes = {}
    first_token_to_flow_scene = {}
    scene_dirs = sorted(flow_root.glob("*_scene"), key=lambda path: int(path.name.split("_")[0]))
    for scene_dir in scene_dirs:
        flow_scene = int(scene_dir.name.split("_")[0])
        scene_payload = {}
        for view in VIEW_ORDER:
            flow_dir = scene_dir / view / "flow" / "flow_arrays"
            view_payload = {}
            for flow_path in sorted(flow_dir.glob("flow_*.npz")):
                data = np.load(flow_path)
                source_token = str(data["source_token"].item())
                target_token = str(data["target_token"].item())
                flow_idx = int(flow_path.stem.split("_")[-1])
                view_payload[source_token] = [flow_idx, target_token]
                if view == "CAM_FRONT" and flow_idx == 0:
                    first_token_to_flow_scene[source_token] = flow_scene
            scene_payload[view] = view_payload
        scenes[str(flow_scene)] = scene_payload
    return {"scenes": scenes, "first_token_to_flow_scene": first_token_to_flow_scene}


def build_scene_to_flow_scene(scene_tokens: list[list[str]], flow_index: dict) -> dict[int, int]:
    mapping = {}
    first_token_to_flow_scene = flow_index["first_token_to_flow_scene"]
    for scene_index, tokens in enumerate(scene_tokens):
        flow_scene = first_token_to_flow_scene.get(tokens[0])
        if flow_scene is not None:
            mapping[scene_index] = int(flow_scene)
    return mapping


def build_scene_to_flow_image_scene(scene_tokens: list[list[str]], flow_image_root: str | os.PathLike) -> dict[int, int]:
    scene_dirs = sorted(Path(flow_image_root).glob("*_scene"), key=lambda path: int(path.name.split("_")[0]))
    if len(scene_dirs) < len(scene_tokens):
        raise RuntimeError(
            f"Flow RGB image root has {len(scene_dirs)} scene dirs, but ann-file has {len(scene_tokens)} scenes."
        )
    return {scene_index: int(scene_dir.name.split("_")[0]) for scene_index, scene_dir in enumerate(scene_dirs)}


def flow_image_path(
    flow_image_root: str | os.PathLike,
    flow_image_subdir: str,
    flow_image_prefix: str,
    scene_index: int,
    view: str,
    frame_index: int,
) -> Path:
    return (
        Path(flow_image_root)
        / f"{scene_index}_scene"
        / view
        / flow_image_subdir
        / f"{flow_image_prefix}_{frame_index:06d}.png"
    )


def build_flow_image_frame_sets(
    flow_image_root: str | os.PathLike,
    flow_image_subdir: str,
    flow_image_prefix: str,
) -> dict[int, dict[str, set[int]]]:
    frame_sets: dict[int, dict[str, set[int]]] = {}
    scene_dirs = sorted(Path(flow_image_root).glob("*_scene"), key=lambda path: int(path.name.split("_")[0]))
    for scene_dir in scene_dirs:
        scene_id = int(scene_dir.name.split("_")[0])
        view_sets = {}
        for view in VIEW_ORDER:
            image_dir = scene_dir / view / flow_image_subdir
            view_sets[view] = {
                int(path.stem.split("_")[-1])
                for path in image_dir.glob(f"{flow_image_prefix}_*.png")
            }
        frame_sets[scene_id] = view_sets
    return frame_sets


def flow_image_row_is_complete(
    row: dict,
    scene_to_flow_scene: dict[int, int],
    frame_sets: dict[int, dict[str, set[int]]],
) -> bool:
    scene_index = int(row["scene_index"])
    scene_id = scene_to_flow_scene.get(scene_index, scene_index)
    start = int(row["scene_frame_start"])
    video_length = int(row["video_length"])
    if scene_id not in frame_sets:
        return False
    for view in VIEW_ORDER:
        view_frames = frame_sets[scene_id].get(view)
        if not view_frames:
            return False
        for frame_index in range(start, start + video_length - 1):
            if frame_index not in view_frames:
                return False
    return True


def filter_rows_with_flow_images(
    rows: list[dict],
    scene_to_flow_scene: dict[int, int],
    flow_image_root: str | os.PathLike,
    flow_image_subdir: str,
    flow_image_prefix: str,
) -> tuple[list[int], dict]:
    frame_sets = build_flow_image_frame_sets(flow_image_root, flow_image_subdir, flow_image_prefix)
    valid_indices = [
        index
        for index, row in enumerate(rows)
        if flow_image_row_is_complete(row, scene_to_flow_scene, frame_sets)
    ]
    return valid_indices, {
        "total_rows": len(rows),
        "kept_rows": len(valid_indices),
        "dropped_rows": len(rows) - len(valid_indices),
        "kept_ratio": len(valid_indices) / max(len(rows), 1),
    }


def resize_map(array: np.ndarray, size: tuple[int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(array).float()[None, None]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=size, mode=mode)
    else:
        resized = F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return resized[0, 0]


class FastLatentDecoderDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        scene_tokens: list[list[str]],
        modalities: tuple[str, ...],
        height: int,
        width: int,
        depth_root_json: str,
        rdepth_root: str,
        flow_root: str | None,
        flow_mask_root: str | None,
        flow_image_root: str | None,
        flow_image_subdir: str,
        flow_image_prefix: str,
        flow_index: dict | None,
        scene_to_flow_scene: dict[int, int] | None,
        flow_scale_px: float,
        use_pixels_per_second: bool,
        allow_missing_flow: bool,
        flow_target_source: str,
        flow_white_threshold: float,
    ) -> None:
        self.rows = rows
        self.scene_tokens = scene_tokens
        self.modalities = modalities
        self.height = height
        self.width = width
        self.rdepth_root = Path(rdepth_root)
        self.flow_root = Path(flow_root) if flow_root is not None else None
        self.flow_mask_root = Path(flow_mask_root) if flow_mask_root is not None else self.flow_root
        self.flow_image_root = Path(flow_image_root) if flow_image_root is not None else None
        self.flow_image_subdir = flow_image_subdir
        self.flow_image_prefix = flow_image_prefix
        self.flow_index = flow_index
        self.scene_to_flow_scene = scene_to_flow_scene or {}
        self.flow_scale_px = flow_scale_px
        self.use_pixels_per_second = use_pixels_per_second
        self.allow_missing_flow = allow_missing_flow
        self.flow_target_source = flow_target_source
        self.flow_white_threshold = flow_white_threshold
        with open(depth_root_json, "r", encoding="utf-8") as file:
            self.sample_token_to_depth = json.load(file)

    def __len__(self) -> int:
        return len(self.rows)

    def _payload_latent(self, payload: dict, modality: str) -> torch.Tensor:
        if modality == "depth":
            latent = payload.get("depth_latent")
            if latent is None:
                latent = payload["latent"][:, VAE_OUT_CHANNELS : VAE_OUT_CHANNELS * 2]
        elif modality == "flow":
            latent = payload.get("flow_latent")
            if latent is None:
                latent = payload["latent"][:, VAE_OUT_CHANNELS * 2 : VAE_OUT_CHANNELS * 3]
        else:
            raise ValueError(modality)
        return latent.float()

    def _load_depth_target(self, clip_tokens: list[str]) -> torch.Tensor:
        depth_video = torch.empty(len(VIEW_ORDER), 1, len(clip_tokens), self.height, self.width)
        for frame_idx, token in enumerate(clip_tokens):
            sample_root = self.rdepth_root / self.sample_token_to_depth[token]
            for view_idx, view in enumerate(VIEW_ORDER):
                depth_path = sample_root / view / "refined_depth.npz"
                semantic_path = sample_root / view / "semantic_oneformer.npz"
                depth = np.load(depth_path)["depth_pred"].astype(np.float32)
                semantic = np.load(semantic_path)["sem"]
                depth_tensor = resize_map(depth, (self.height, self.width), "bicubic")
                semantic_tensor = resize_map(semantic.astype(np.float32), (self.height, self.width), "nearest")
                depth_tensor[semantic_tensor == 27] = 100.0
                depth_video[view_idx, 0, frame_idx] = (2.0 * (depth_tensor.clamp(0.0, 100.0) / 100.0)) - 1.0
        return depth_video

    def _flow_paths(self, flow_scene: int, view: str, flow_idx: int) -> tuple[Path, Path]:
        assert self.flow_root is not None
        assert self.flow_mask_root is not None
        flow_scene_dir = self.flow_root / f"{flow_scene}_scene"
        mask_scene_dir = self.flow_mask_root / f"{flow_scene}_scene"
        return (
            flow_scene_dir / view / "flow" / "flow_arrays" / f"flow_{flow_idx:06d}.npz",
            mask_scene_dir / view / "sam" / "masks" / f"dynamic_object_mask_{flow_idx:06d}.png",
        )

    def _load_flow_target(self, row: dict, clip_tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flow_index is None or self.flow_root is None:
            raise RuntimeError("--flow-root is required when training flow.")
        scene_index = int(row["scene_index"])
        if scene_index not in self.scene_to_flow_scene:
            if not self.allow_missing_flow:
                raise RuntimeError(f"Missing flow scene for scene_index={scene_index}")
            flow_video = torch.ones(len(VIEW_ORDER), 3, len(clip_tokens), self.height, self.width)
            dynamic_mask = torch.zeros(len(VIEW_ORDER), 1, len(clip_tokens), self.height, self.width)
            return flow_video, dynamic_mask

        flow_scene = self.scene_to_flow_scene[scene_index]
        scene_index_payload = self.flow_index["scenes"][str(flow_scene)]
        flow_video = torch.ones(len(VIEW_ORDER), 3, len(clip_tokens), self.height, self.width)
        dynamic_mask = torch.zeros(len(VIEW_ORDER), 1, len(clip_tokens), self.height, self.width)
        for view_idx, view in enumerate(VIEW_ORDER):
            view_index = scene_index_payload[view]
            for frame_idx, source_token in enumerate(clip_tokens[:-1]):
                item = view_index.get(source_token)
                if item is None or item[1] != clip_tokens[frame_idx + 1]:
                    if self.allow_missing_flow:
                        continue
                    raise RuntimeError(
                        f"Missing flow edge scene={scene_index} view={view} token={source_token}"
                    )
                flow_path, mask_path = self._flow_paths(flow_scene, view, item[0])
                data = np.load(flow_path)
                flow = np.nan_to_num(flow_from_array(data, self.use_pixels_per_second))
                valid = data["valid"].astype(bool)
                mask = np.asarray(Image.open(mask_path).convert("L")) > 127
                mask = np.logical_and(mask, valid)
                if flow.shape[:2] != (self.height, self.width):
                    raise RuntimeError(f"Unexpected flow shape {flow.shape} in {flow_path}")
                if mask.shape != (self.height, self.width):
                    raise RuntimeError(f"Unexpected mask shape {mask.shape} in {mask_path}")
                masked_flow = flow * mask[..., None]
                flow_rgb = flow_to_image_fixed_scale(masked_flow, self.flow_scale_px)
                flow_video[view_idx, :, frame_idx] = (
                    torch.from_numpy(flow_rgb.astype(np.float32).transpose(2, 0, 1)) / 127.5
                ) - 1.0
                dynamic_mask[view_idx, 0, frame_idx] = torch.from_numpy(mask.astype(np.float32))
        return flow_video, dynamic_mask

    def _flow_image_path(self, scene_index: int, view: str, frame_index: int) -> Path:
        assert self.flow_image_root is not None
        return flow_image_path(
            self.flow_image_root,
            self.flow_image_subdir,
            self.flow_image_prefix,
            scene_index,
            view,
            frame_index,
        )

    def _load_flow_image_target(self, row: dict, clip_tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flow_image_root is None:
            raise RuntimeError("--flow-image-root is required for flow_rgb_image supervision.")
        scene_index = int(row["scene_index"])
        flow_scene = self.scene_to_flow_scene.get(scene_index, scene_index)
        scene_frame_start = int(row["scene_frame_start"])
        flow_video = torch.ones(len(VIEW_ORDER), 3, len(clip_tokens), self.height, self.width)
        dynamic_mask = torch.zeros(len(VIEW_ORDER), 1, len(clip_tokens), self.height, self.width)
        for view_idx, view in enumerate(VIEW_ORDER):
            for frame_idx in range(len(clip_tokens) - 1):
                image_path = self._flow_image_path(flow_scene, view, scene_frame_start + frame_idx)
                if not image_path.exists():
                    if self.allow_missing_flow:
                        continue
                    raise RuntimeError(f"Missing flow RGB image: {image_path}")
                image = np.asarray(Image.open(image_path).convert("RGB"))
                if image.shape[:2] != (self.height, self.width):
                    raise RuntimeError(f"Unexpected flow RGB image shape {image.shape} in {image_path}")
                target = (torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)) / 127.5) - 1.0
                flow_video[view_idx, :, frame_idx] = target
                dynamic_mask[view_idx, 0, frame_idx] = (target.amin(dim=0) < self.flow_white_threshold).float()
        return flow_video, dynamic_mask

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        payload = torch.load(row["path"], map_location="cpu")
        clip_tokens = clip_tokens_for_row(row, self.scene_tokens)
        item: dict[str, torch.Tensor | int | str] = {
            "dataset_index": int(row["dataset_index"]),
            "video_length": int(row["video_length"]),
            "token": row["token"],
        }
        if "depth" in self.modalities:
            item["depth_latent"] = self._payload_latent(payload, "depth")
            item["depth_target"] = self._load_depth_target(clip_tokens)
        if "flow" in self.modalities:
            item["flow_latent"] = self._payload_latent(payload, "flow")
            if self.flow_target_source == "raw":
                flow_target, flow_dynamic_mask = self._load_flow_target(row, clip_tokens)
                item["flow_target"] = flow_target
                item["flow_dynamic_mask"] = flow_dynamic_mask
            elif self.flow_target_source == "flow_rgb_image":
                flow_target, flow_dynamic_mask = self._load_flow_image_target(row, clip_tokens)
                item["flow_target"] = flow_target
                item["flow_dynamic_mask"] = flow_dynamic_mask
        return item


def flatten_views(tensor: torch.Tensor) -> torch.Tensor:
    batch, views = tensor.shape[:2]
    return tensor.reshape(batch * views, *tensor.shape[2:])


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand(-1, pred.shape[1], -1, -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return l1_loss(pred_dx, target_dx) + l1_loss(pred_dy, target_dy)


def temporal_gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] <= 1:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])


def dynamic_edge_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 5:
        raise ValueError(f"Expected mask shape B,1,T,H,W, got {tuple(mask.shape)}")
    mask = (mask > 0.5).to(dtype=mask.dtype)
    batch, channels, frames, height, width = mask.shape
    mask_2d = mask.reshape(batch * frames, channels, height, width)
    dilated = F.max_pool2d(mask_2d, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask_2d, kernel_size=3, stride=1, padding=1)
    edge = (dilated - eroded).clamp(0.0, 1.0)
    return edge.reshape(batch, channels, frames, height, width)


def autocast_context(precision: str):
    if precision == "fp32":
        return torch.amp.autocast("cuda", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def build_decoders(args: argparse.Namespace, modalities: tuple[str, ...]) -> dict[str, FastLatentVideoDecoder]:
    common = dict(
        decoder_architecture=args.decoder_architecture,
        hidden_channels=args.hidden_channels,
        temporal_blocks=args.temporal_blocks,
        temporal_upsample_factor=args.temporal_upsample_factor,
        temporal_refine_blocks=args.temporal_refine_blocks,
        spatial_3d_blocks=args.spatial_3d_blocks,
        spatial_2d_blocks=args.spatial_2d_blocks,
        spatial_frame_chunk_size=args.spatial_frame_chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    decoders = {}
    if "depth" in modalities:
        decoders["depth"] = build_fast_depth_decoder(**common)
    if "flow" in modalities:
        decoders["flow"] = build_fast_flow_rgb_decoder(**common)
    return decoders


def module_state(module: torch.nn.Module) -> dict:
    return module.module.state_dict() if isinstance(module, DistributedDataParallel) else module.state_dict()


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    decoders: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "args": vars(args),
        "decoder_config": {
            "decoder_architecture": args.decoder_architecture,
            "hidden_channels": args.hidden_channels,
            "temporal_blocks": args.temporal_blocks,
            "temporal_upsample_factor": args.temporal_upsample_factor,
            "temporal_refine_blocks": args.temporal_refine_blocks,
            "spatial_3d_blocks": args.spatial_3d_blocks,
            "spatial_2d_blocks": args.spatial_2d_blocks,
            "spatial_frame_chunk_size": args.spatial_frame_chunk_size,
            "gradient_checkpointing": args.gradient_checkpointing,
            "height": args.height,
            "width": args.width,
        },
        "decoders": {name: module_state(decoder) for name, decoder in decoders.items()},
        "optimizer": optimizer.state_dict(),
    }
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    decoders: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    payload = torch.load(path, map_location=device)
    for name, state in payload["decoders"].items():
        if name in decoders:
            target = decoders[name].module if isinstance(decoders[name], DistributedDataParallel) else decoders[name]
            target.load_state_dict(state)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"]), int(payload["epoch"])


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size
        reduced[key] = float(tensor.item())
    return reduced


def prepare_output_dir(path: Path, args: argparse.Namespace, rank: int) -> None:
    if not is_main_process(rank):
        return
    if path.exists() and any(path.iterdir()) and args.resume is None and not args.overwrite_output_dir:
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")


def train_step(
    batch: dict,
    decoders: dict[str, torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    teacher_vae=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_loss = torch.zeros((), device=device)
    metrics: dict[str, torch.Tensor] = {}
    out_size = (args.height, args.width)

    if "depth" in decoders:
        depth_latent = flatten_views(batch["depth_latent"]).to(device, non_blocking=True)
        depth_target = flatten_views(batch["depth_target"]).to(device, non_blocking=True)
        with autocast_context(args.precision):
            depth_pred = decoders["depth"](depth_latent, out_frames=depth_target.shape[2], out_size=out_size)
            depth_l1 = l1_loss(depth_pred, depth_target)
            depth_grad = gradient_l1_loss(depth_pred, depth_target)
            depth_loss = depth_l1 + args.depth_gradient_weight * depth_grad
        total_loss = total_loss + depth_loss
        metrics["loss_depth_l1"] = depth_l1
        metrics["loss_depth_grad"] = depth_grad
        metrics["loss_depth"] = depth_loss

    if "flow" in decoders:
        flow_latent = flatten_views(batch["flow_latent"]).to(device, non_blocking=True)
        if args.flow_target_source == "vae_decode":
            if teacher_vae is None:
                raise RuntimeError("teacher_vae is required for --flow-target-source vae_decode.")
            with torch.inference_mode():
                flow_target = teacher_vae.decode(flow_latent.to(torch.float16)).detach()
            flow_dynamic_mask = (flow_target.amin(dim=1, keepdim=True) < args.flow_white_threshold).float()
        else:
            flow_target = flatten_views(batch["flow_target"]).to(device, non_blocking=True)
            flow_dynamic_mask = flatten_views(batch["flow_dynamic_mask"]).to(device, non_blocking=True)
        with autocast_context(args.precision):
            flow_target = flow_target.to(dtype=flow_latent.dtype if args.precision == "fp32" else flow_target.dtype)
            flow_pred = decoders["flow"](flow_latent, out_frames=flow_target.shape[2], out_size=out_size)
            flow_target = flow_target.to(dtype=flow_pred.dtype)
            flow_dynamic_mask = flow_dynamic_mask.to(dtype=flow_pred.dtype)
            flow_l1 = l1_loss(flow_pred, flow_target)
            flow_dynamic = masked_l1_loss(flow_pred, flow_target, flow_dynamic_mask)
            flow_temporal_grad = temporal_gradient_l1_loss(flow_pred, flow_target)
            flow_edge_mask = dynamic_edge_mask(flow_dynamic_mask)
            flow_edge = masked_l1_loss(flow_pred, flow_target, flow_edge_mask)
            flow_loss = (
                flow_l1
                + args.flow_dynamic_weight * flow_dynamic
                + args.flow_temporal_gradient_weight * flow_temporal_grad
                + args.flow_edge_weight * flow_edge
            )
        total_loss = total_loss + flow_loss
        metrics["loss_flow_l1"] = flow_l1
        metrics["loss_flow_dynamic"] = flow_dynamic
        metrics["loss_flow_temporal_grad"] = flow_temporal_grad
        metrics["loss_flow_edge"] = flow_edge
        metrics["loss_flow"] = flow_loss

    metrics["loss"] = total_loss
    return total_loss, metrics


def main() -> None:
    args = parse_args()
    modalities = tuple(args.modalities)
    if "flow" in modalities and args.flow_target_source == "raw" and args.flow_root is None:
        raise RuntimeError("--flow-root is required when --modalities includes flow.")
    if "flow" in modalities and args.flow_target_source == "flow_rgb_image" and args.flow_image_root is None:
        raise RuntimeError("--flow-image-root is required when --flow-target-source flow_rgb_image.")

    rank, world_size, local_rank, device = init_distributed()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args, rank)
    barrier()

    rows = load_rows(args.manifest, args.limit)
    scene_tokens = load_scene_tokens(args.ann_file)
    flow_index = None
    scene_to_flow_scene = None
    if "flow" in modalities and args.flow_target_source == "raw":
        if is_main_process(rank):
            print(f"Building flow index from {args.flow_root}", flush=True)
            flow_index = build_flow_index(args.flow_root)
        if world_size > 1:
            flow_index_payload = [flow_index]
            dist.broadcast_object_list(flow_index_payload, src=0)
            flow_index = flow_index_payload[0]
        else:
            assert flow_index is not None
        scene_to_flow_scene = build_scene_to_flow_scene(scene_tokens, flow_index)
    elif "flow" in modalities and args.flow_target_source == "flow_rgb_image":
        if is_main_process(rank):
            print(f"Building flow RGB image scene mapping from {args.flow_image_root}", flush=True)
            scene_to_flow_scene = build_scene_to_flow_image_scene(scene_tokens, args.flow_image_root)
        if world_size > 1:
            scene_map_payload = [scene_to_flow_scene]
            dist.broadcast_object_list(scene_map_payload, src=0)
            scene_to_flow_scene = scene_map_payload[0]
        else:
            assert scene_to_flow_scene is not None
        if args.drop_missing_flow_images:
            if is_main_process(rank):
                print("Filtering rows with incomplete flow RGB image supervision", flush=True)
                valid_indices, filter_stats = filter_rows_with_flow_images(
                    rows,
                    scene_to_flow_scene,
                    args.flow_image_root,
                    args.flow_image_subdir,
                    args.flow_image_prefix,
                )
            else:
                valid_indices, filter_stats = None, None
            if world_size > 1:
                filter_payload = [valid_indices, filter_stats]
                dist.broadcast_object_list(filter_payload, src=0)
                valid_indices, filter_stats = filter_payload
            assert valid_indices is not None and filter_stats is not None
            kept_rows = [rows[index] for index in valid_indices]
            if is_main_process(rank):
                filter_report = dict(filter_stats)
                filter_report["kept_row_indices"] = [int(index) for index in valid_indices]
                filter_report["kept_dataset_indices"] = [int(row["dataset_index"]) for row in kept_rows]
                filter_report["kept_tokens"] = [str(row["token"]) for row in kept_rows]
                (output_dir / "flow_image_valid_rows.json").write_text(
                    json.dumps(filter_report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"Filtered flow RGB image rows: kept={filter_stats['kept_rows']} "
                    f"dropped={filter_stats['dropped_rows']} total={filter_stats['total_rows']}",
                    flush=True,
                )
            rows = kept_rows
    if is_main_process(rank):
        print(
            f"Loaded {len(rows)} rows, modalities={modalities}, "
            f"flow_target_source={args.flow_target_source}, world_size={world_size}",
            flush=True,
        )

    dataset = FastLatentDecoderDataset(
        rows=rows,
        scene_tokens=scene_tokens,
        modalities=modalities,
        height=args.height,
        width=args.width,
        depth_root_json=args.depth_root_json,
        rdepth_root=args.rdepth_root,
        flow_root=args.flow_root,
        flow_mask_root=args.flow_mask_root,
        flow_image_root=args.flow_image_root,
        flow_image_subdir=args.flow_image_subdir,
        flow_image_prefix=args.flow_image_prefix,
        flow_index=flow_index,
        scene_to_flow_scene=scene_to_flow_scene,
        flow_scale_px=args.flow_scale_px,
        use_pixels_per_second=args.use_pixels_per_second,
        allow_missing_flow=args.allow_missing_flow,
        flow_target_source=args.flow_target_source,
        flow_white_threshold=args.flow_white_threshold,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    decoders = build_decoders(args, modalities)
    for name, decoder in decoders.items():
        decoder.to(device)
        if world_size > 1:
            decoders[name] = DistributedDataParallel(decoder, device_ids=[local_rank])
    params = [param for decoder in decoders.values() for param in decoder.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    teacher_vae = None
    if "flow" in modalities and args.flow_target_source == "vae_decode":
        teacher_vae = build_teacher_vae(args, device)
        for param in teacher_vae.parameters():
            param.requires_grad_(False)

    global_step = 0
    start_epoch = 0
    if args.resume is not None:
        global_step, start_epoch = load_checkpoint(args.resume, decoders, optimizer, device)
        if is_main_process(rank):
            print(f"Resumed from {args.resume} at step={global_step} epoch={start_epoch}", flush=True)

    last_log = time.time()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = train_step(batch, decoders, args, device, teacher_vae=teacher_vae)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            should_log = global_step == 1 or global_step % args.log_every == 0
            if should_log:
                reduced = reduce_metrics(metrics, world_size)
                if is_main_process(rank):
                    elapsed = max(time.time() - last_log, 1.0e-6)
                    last_log = time.time()
                    metric_text = " ".join(f"{key}={value:.5f}" for key, value in sorted(reduced.items()))
                    print(
                        f"step={global_step} epoch={epoch} lr={optimizer.param_groups[0]['lr']:.3e} "
                        f"sec_per_log={elapsed:.2f} {metric_text}",
                        flush=True,
                    )

            if is_main_process(rank) and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(output_dir / f"checkpoint_step{global_step:06d}.pt", args, decoders, optimizer, global_step, epoch)
                save_checkpoint(output_dir / "checkpoint_latest.pt", args, decoders, optimizer, global_step, epoch)

            if args.dry_run or (args.max_steps is not None and global_step >= args.max_steps):
                break
        if args.dry_run or (args.max_steps is not None and global_step >= args.max_steps):
            break

    if is_main_process(rank):
        save_checkpoint(output_dir / "checkpoint_latest.pt", args, decoders, optimizer, global_step, epoch)
        print(f"Finished fast latent decoder training at step={global_step}", flush=True)
    barrier()


if __name__ == "__main__":
    main()
