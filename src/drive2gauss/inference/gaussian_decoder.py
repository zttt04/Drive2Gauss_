#!/usr/bin/env python3
"""Render every generated-latent FlowTrack clip and evaluate infer/GT fidelity."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from drive2gauss.data import manifest as dataset_manifest


WINDOW_STARTS = (0, 4, 8, 12)
METRIC_NAMES = (
    "psnr_infer",
    "lpips_infer_alex",
    "lpips_gt_alex",
    "ssim_infer",
)


def checkpoint_train_args(config: dict[str, Any]) -> argparse.Namespace:
    """Restore training arguments used by released checkpoints.

    ``zero_flow_input`` was added after the released feature-UNet checkpoint.
    Its normal RGB-D-flow behavior is False, so old checkpoints may safely omit
    the key without modifying either the checkpoint or its flow inputs.
    """
    restored = dict(config)
    restored.setdefault("zero_flow_input", False)
    return argparse.Namespace(**restored)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache-root", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lpips-module-root", type=Path, default=None)
    parser.add_argument("--torch-home", type=Path, default=None)
    parser.add_argument("--torch-extensions-root", type=Path, default=None)
    parser.add_argument("--lpips-batch-size", type=int, default=12)
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-ann-file", type=Path, default=None)
    parser.add_argument("--val-ann-file", type=Path, default=None)
    parser.add_argument("--motion-release-root", type=Path, default=None)
    parser.add_argument("--motion-release-manifest", type=Path, default=None)
    parser.add_argument("--train-motion-release-root", type=Path, default=None)
    parser.add_argument("--val-motion-release-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
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


def read_front_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = dataset_manifest.read_jsonl(path)
    selected = []
    for row in rows:
        if row.get("query_view_group") != "front":
            continue
        selected.append({**row, "dataset_split": split})
    return selected


def render_image_path(
    output_dir: Path,
    split: str,
    token: str,
    window_start: int,
    frame: int,
    view_name: str,
) -> Path:
    return (
        output_dir
        / "renders"
        / split
        / token
        / f"window_{window_start:02d}"
        / f"frame_{frame:02d}_{view_name}.png"
    )


def window_images_exist(
    output_dir: Path,
    row: dict[str, Any],
    window_start: int,
    view_names: list[str],
) -> bool:
    return all(
        render_image_path(
            output_dir,
            str(row["dataset_split"]),
            str(row["token"]),
            window_start,
            frame,
            view_name,
        ).is_file()
        for frame in range(window_start, window_start + 4)
        for view_name in view_names
    )


def write_png_atomic(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(temporary), bgr):
        raise RuntimeError(f"Failed to write render image: {temporary}")
    os.replace(temporary, path)


def load_completed(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            completed.add(
                (
                    str(record["dataset_split"]),
                    int(record["manifest_index"]),
                    int(record["window_start"]),
                )
            )
    return completed


def metric_values(
    rendered: torch.Tensor,
    infer: torch.Tensor,
    gt: torch.Tensor,
    lpips_model: torch.nn.Module,
    lpips_batch_size: int,
) -> dict[str, torch.Tensor]:
    infer_mse = (rendered - infer).square().flatten(1).mean(dim=1)
    values = {
        "psnr_infer": -10.0 * torch.log10(infer_mse.clamp_min(1.0e-12)),
        "ssim_infer": ssim_per_image(rendered, infer),
    }
    infer_parts = []
    gt_parts = []
    for start in range(0, rendered.shape[0], lpips_batch_size):
        end = start + lpips_batch_size
        render_part = rendered[start:end] * 2.0 - 1.0
        infer_parts.append(
            lpips_model(render_part, infer[start:end] * 2.0 - 1.0).flatten()
        )
        gt_parts.append(lpips_model(render_part, gt[start:end] * 2.0 - 1.0).flatten())
    values["lpips_infer_alex"] = torch.cat(infer_parts)
    values["lpips_gt_alex"] = torch.cat(gt_parts)
    return values


def ssim_per_image(
    first: torch.Tensor,
    second: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError(f"Expected matching BCHW tensors, got {first.shape} and {second.shape}")
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
    return ssim_map.flatten(1).mean(dim=1)


def raw_gt_image(
    clip: dict[str, Any],
    frame: int,
    view: int,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    path = Path(clip["_rgb_paths"][frame][view])
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32).div_(255.0)


def evaluate_window(
    model,
    row: dict[str, Any],
    cache_payload: dict[str, Any],
    clip: dict[str, Any],
    generated_source,
    train_args: argparse.Namespace,
    lpips_model: torch.nn.Module,
    lpips_batch_size: int,
    output_dir: Path,
    window_start: int,
    pointforward,
    trainer,
    device: torch.device,
) -> dict[str, Any]:
    sample = trainer.build_window_sample(
        row, window_start, cache_payload, clip, train_args, device
    )
    gaussian = model.predict_gaussians(sample, train_args)
    rendered_images = []
    infer_images = []
    gt_images = []
    image_keys = []
    for frame in sample.frames:
        for view in sample.views:
            rendered = pointforward.render_output(
                gaussian,
                model.appearance_decoder,
                sample.clip,
                frame,
                view,
                train_args.height,
                train_args.width,
                train_args,
                device,
                camera_affine=getattr(model, "camera_affine", None),
            )
            infer = trainer.target_image(
                sample, generated_source, frame, view, train_args, device
            )
            gt = raw_gt_image(
                clip, frame, view, train_args.width, train_args.height, device
            )
            render_u8 = (
                rendered.detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .add(0.5)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            rendered_images.append(
                torch.from_numpy(render_u8)
                .permute(2, 0, 1)
                .to(device=device, dtype=torch.float32)
                .div_(255.0)
            )
            infer_images.append(infer)
            gt_images.append(gt)
            image_keys.append((frame, view))
            write_png_atomic(
                render_image_path(
                    output_dir,
                    str(row["dataset_split"]),
                    str(row["token"]),
                    window_start,
                    frame,
                    pointforward.VIEW_NAMES[view],
                ),
                render_u8,
            )

    rendered_batch = torch.stack(rendered_images)
    infer_batch = torch.stack(infer_images)
    gt_batch = torch.stack(gt_images)
    values = metric_values(
        rendered_batch, infer_batch, gt_batch, lpips_model, lpips_batch_size
    )
    images = [
        {
            "frame": int(frame),
            "view": pointforward.VIEW_NAMES[view],
            **{name: float(values[name][index]) for name in METRIC_NAMES},
        }
        for index, (frame, view) in enumerate(image_keys)
    ]
    result = {
        "dataset_split": str(row["dataset_split"]),
        "manifest_index": int(row["manifest_index"]),
        "scene_index": int(row["scene_index"]),
        "scene_frame_start": int(row["scene_frame_start"]),
        "token": str(row["token"]),
        "window_start": int(window_start),
        "frames": sample.frames,
        "views": [pointforward.VIEW_NAMES[view] for view in sample.views],
        "num_queries": int(gaussian["means"].shape[0]),
        "metrics": {name: float(values[name].mean()) for name in METRIC_NAMES},
        "images": images,
    }
    del sample, gaussian, rendered_batch, infer_batch, gt_batch, values
    return result


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan, "ci95": math.nan}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(array.size)),
    }


def summarize(records: list[dict[str, Any]], checkpoint_step: int) -> dict[str, Any]:
    images = [image for record in records for image in record["images"]]

    def image_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            name: distribution([float(image[name]) for image in selected])
            for name in METRIC_NAMES
        }

    by_clip: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_clip[(record["dataset_split"], int(record["manifest_index"]))].extend(
            record["images"]
        )
    return {
        "checkpoint_step": checkpoint_step,
        "protocol": {
            "clips": len(by_clip),
            "windows": len(records),
            "images": len(images),
            "window_starts": list(WINDOW_STARTS),
            "frames_per_window": 4,
            "views": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            "render_target": "generated RGB decoded from the same generated latent",
            "gt_target": "raw nuScenes RGB",
        },
        "image_level": image_summary(images),
        "clip_level": {
            name: distribution(
                [
                    float(np.mean([float(image[name]) for image in clip_images]))
                    for clip_images in by_clip.values()
                ]
            )
            for name in METRIC_NAMES
        },
        "by_split": {
            split: image_summary(
                [
                    image
                    for record in records
                    if record["dataset_split"] == split
                    for image in record["images"]
                ]
            )
            for split in ("train", "heldout")
        },
        "by_camera": {
            view: image_summary([image for image in images if image["view"] == view])
            for view in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
        },
        "by_window_start": {
            str(window_start): image_summary(
                [
                    image
                    for record in records
                    if int(record["window_start"]) == window_start
                    for image in record["images"]
                ]
            )
            for window_start in WINDOW_STARTS
        },
    }


def write_run_files(
    args: argparse.Namespace,
    world_size: int,
    checkpoint_step: int,
    clip_count: int,
) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        {
            "world_size": world_size,
            "checkpoint_step": checkpoint_step,
            "clip_count": clip_count,
            "window_starts": list(WINDOW_STARTS),
            "view_group": "front",
        }
    )
    (args.output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "command.sh").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    (args.output_dir / "git.txt").write_text(revision.stdout + status.stdout)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.lpips_batch_size <= 0:
        raise ValueError("--lpips-batch-size must be positive")
    rank, world_size, local_rank, device = distributed_context()
    if args.torch_extensions_root is not None:
        rank_extension_root = args.torch_extensions_root / f"rank_{local_rank:02d}"
        rank_extension_root.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(rank_extension_root)

    from drive2gauss.training import gaussian_decoder as trainer
    from drive2gauss.training import static_decoder_pipeline as pointforward

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(checkpoint["step"])
    train_args = checkpoint_train_args(checkpoint["config"])
    for name in (
        "data_root",
        "train_ann_file",
        "val_ann_file",
        "motion_release_root",
        "motion_release_manifest",
        "train_motion_release_root",
        "val_motion_release_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(train_args, name, value)
    train_args.context_frame_count = 4
    train_args.single_view_train = False
    rows = read_front_rows(args.train_manifest, "train")
    rows.extend(read_front_rows(args.val_manifest, "heldout"))
    if args.limit_clips > 0:
        rows = rows[: args.limit_clips]
    if rank == 0:
        write_run_files(args, world_size, checkpoint_step, len(rows))
    if world_size > 1:
        dist.barrier()

    cache_roots = {
        "train": args.train_cache_root,
        "heldout": args.val_cache_root,
    }
    sources = {
        "train": trainer.build_query_source(train_args, "train"),
        "heldout": trainer.build_query_source(train_args, "val"),
    }
    model = trainer.FlowTrackRenderModel(train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    lpips_root = args.lpips_module_root or getattr(train_args, "lpips_module_root", None)
    if lpips_root is not None and str(lpips_root) not in sys.path:
        sys.path.append(str(lpips_root))
    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = str(args.torch_home)
    from lpips import LPIPS

    lpips_model = LPIPS(net="alex").to(device).eval().requires_grad_(False)
    random.seed(int(train_args.seed) + rank)
    np.random.seed(int(train_args.seed) + rank)
    torch.manual_seed(int(train_args.seed) + rank)

    assigned_rows = rows[rank::world_size]
    metrics_path = args.output_dir / f"metrics_rank{rank:02d}.jsonl"
    completed = load_completed(metrics_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    start_time = time.perf_counter()
    completed_windows = 0
    with metrics_path.open(mode, encoding="utf-8") as stream:
        for row in assigned_rows:
            split = str(row["dataset_split"])
            cache_payload = torch.load(
                trainer.cache_path_for(row, cache_roots[split]),
                map_location="cpu",
                weights_only=False,
            )
            source = sources[split]
            clip = source.load_clip(row, [int(view) for view in cache_payload["view_indices"]])
            view_names = [pointforward.VIEW_NAMES[int(view)] for view in cache_payload["view_indices"]]
            for window_start in WINDOW_STARTS:
                key = (split, int(row["manifest_index"]), window_start)
                if key in completed and window_images_exist(
                    args.output_dir, row, window_start, view_names
                ):
                    continue
                record = evaluate_window(
                    model,
                    row,
                    cache_payload,
                    clip,
                    source,
                    train_args,
                    lpips_model,
                    args.lpips_batch_size,
                    args.output_dir,
                    window_start,
                    pointforward,
                    trainer,
                    device,
                )
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                completed_windows += 1
                elapsed = time.perf_counter() - start_time
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "local_rank": local_rank,
                            "dataset_split": split,
                            "manifest_index": int(row["manifest_index"]),
                            "window_start": window_start,
                            "completed_windows": completed_windows,
                            "elapsed_sec": elapsed,
                            "windows_per_sec": completed_windows / max(elapsed, 1.0e-6),
                            **record["metrics"],
                        }
                    ),
                    flush=True,
                )
            del cache_payload, clip

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        records = []
        for other_rank in range(world_size):
            path = args.output_dir / f"metrics_rank{other_rank:02d}.jsonl"
            records.extend(
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            )
        expected_windows = len(rows) * len(WINDOW_STARTS)
        keys = {
            (
                record["dataset_split"],
                int(record["manifest_index"]),
                int(record["window_start"]),
            )
            for record in records
        }
        if len(records) != expected_windows or len(keys) != expected_windows:
            raise RuntimeError(
                f"Expected {expected_windows} unique records, found "
                f"{len(records)} records and {len(keys)} unique keys"
            )
        summary = summarize(records, checkpoint_step)
        summary.update(
            {
                "checkpoint": str(args.checkpoint),
                "world_size": world_size,
                "elapsed_sec": time.perf_counter() - start_time,
            }
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
