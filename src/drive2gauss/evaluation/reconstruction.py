#!/usr/bin/env python3
"""Evaluate a flow-track PointForward checkpoint on every 4-frame validation window."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist

from drive2gauss.data import query_dataset as query_data
from drive2gauss.data import manifest as dataset_manifest
from drive2gauss.training import gaussian_decoder as trainer
from drive2gauss.training import static_decoder_pipeline as pointforward


METRIC_NAMES = ("psnr", "ssim", "lpips", "l1")
PREBUILT_CLIP_KEYS = {
    "rgb_target",
    "depth_target",
    "flow_rgb_target",
    "flow_rgb_valid_target",
    "camera_intrinsics",
    "lidar2camera",
    "frame_to_ref_lidar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--depth-map-json", type=Path, required=True)
    parser.add_argument("--masked-flow-rgb-root", type=Path, required=True)
    parser.add_argument("--masked-flow-index", type=Path, required=True)
    parser.add_argument("--allow-missing-flow-rgb", action="store_true")
    parser.add_argument("--lpips-module-root", type=Path, default=None)
    parser.add_argument("--torch-home", type=Path, default=None)
    parser.add_argument("--lpips-batch-size", type=int, default=3)
    parser.add_argument(
        "--distt-video-dir",
        type=Path,
        default=None,
        help="Optional directory of DiST-T RGB videos used as a second reference.",
    )
    parser.add_argument(
        "--distt-video-pattern",
        default="{token}_{view}.mp4",
        help="Filename pattern within --distt-video-dir.",
    )
    parser.add_argument(
        "--cache-view-group",
        choices=("all", "front", "rear"),
        default="all",
        help="Evaluate only clips whose cached features belong to this view group.",
    )
    parser.add_argument(
        "--single-view-name",
        choices=pointforward.VIEW_NAMES,
        default="CAM_FRONT",
        help="Camera used by a checkpoint configured for single-view training.",
    )
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument("--visualize-clips-per-rank", type=int, default=0)
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


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return dataset_manifest.read_jsonl(path)


def write_run_files(
    args: argparse.Namespace,
    world_size: int,
    checkpoint_step: int,
    views_per_clip: int,
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
            "window_starts": trainer.WINDOW_STARTS,
            "frames_per_window": 4,
            "views_per_clip": views_per_clip,
        }
    )
    (args.output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "command.sh").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n"
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    (args.output_dir / "git.txt").write_text(git.stdout + status.stdout)


def load_completed(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
        completed.add((int(record["manifest_index"]), int(record["window_start"])))
    return completed


def metric_values(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lpips_model,
    lpips_batch_size: int,
) -> dict[str, torch.Tensor]:
    from pytorch_msssim import ssim

    difference = rendered - target
    mse = difference.square().flatten(1).mean(dim=1)
    values = {
        "psnr": -10.0 * torch.log10(mse.clamp_min(1.0e-12)),
        "ssim": ssim(rendered, target, data_range=1.0, size_average=False),
        "l1": difference.abs().flatten(1).mean(dim=1),
    }
    lpips_parts = []
    for start in range(0, rendered.shape[0], lpips_batch_size):
        end = start + lpips_batch_size
        lpips_parts.append(
            lpips_model(
                rendered[start:end] * 2.0 - 1.0,
                target[start:end] * 2.0 - 1.0,
            ).flatten()
        )
    values["lpips"] = torch.cat(lpips_parts)
    return values


def read_distt_video(
    path: Path,
    frame_count: int,
    width: int,
    height: int,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open DiST-T reference video: {path}")
    frames = []
    try:
        for frame_index in range(frame_count):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"DiST-T reference video {path} ended before frame {frame_index}"
                )
            if frame_bgr.shape[:2] != (height, width):
                raise ValueError(
                    f"DiST-T reference video {path} has frame size "
                    f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}, expected {width}x{height}"
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
    finally:
        capture.release()
    return torch.stack(frames).float().div_(255.0)


def load_distt_references(
    video_dir: Path,
    video_pattern: str,
    token: str,
    views: list[int],
    frame_count: int,
    width: int,
    height: int,
) -> dict[int, torch.Tensor]:
    references = {}
    for view in views:
        view_name = pointforward.VIEW_NAMES[view]
        path = video_dir / video_pattern.format(token=token, view=view_name)
        references[view] = read_distt_video(path, frame_count, width, height)
    return references


def evaluate_window(
    model: trainer.FlowTrackRenderModel,
    row: dict[str, Any],
    cache_payload: dict[str, Any],
    clip: dict[str, torch.Tensor],
    query_source: trainer.PackagedQuerySource,
    train_args: argparse.Namespace,
    lpips_model,
    lpips_batch_size: int,
    device: torch.device,
    panel_dir: Path | None,
    distt_references: dict[int, torch.Tensor] | None,
) -> dict[str, Any]:
    window_start = int(train_args.time_origin)
    sample = trainer.build_window_sample(
        row, window_start, cache_payload, clip, train_args, device
    )
    gaussian = model.predict_gaussians(sample, train_args)
    rendered_images = []
    target_images = []
    distt_images = []
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
            target = trainer.target_image(
                sample, query_source, frame, view, train_args, device
            )
            rendered_images.append(rendered)
            target_images.append(target)
            if distt_references is not None:
                distt_images.append(distt_references[view][frame].to(device))
            image_keys.append((frame, view))
            if panel_dir is not None:
                pointforward.save_panel(
                    panel_dir / f"f{frame:02d}_{pointforward.VIEW_NAMES[view]}.jpg",
                    rendered,
                    target,
                )
    rendered_batch = torch.stack(rendered_images)
    target_batch = torch.stack(target_images)
    values = metric_values(
        rendered_batch, target_batch, lpips_model, lpips_batch_size
    )
    distt_values = None
    if distt_references is not None:
        distt_batch = torch.stack(distt_images)
        distt_values = metric_values(
            rendered_batch, distt_batch, lpips_model, lpips_batch_size
        )
    images = []
    for index, (frame, view) in enumerate(image_keys):
        image = {
            "frame": int(frame),
            "view": pointforward.VIEW_NAMES[view],
            **{name: float(values[name][index]) for name in METRIC_NAMES},
        }
        if distt_values is not None:
            image["vs_distt"] = {
                name: float(distt_values[name][index]) for name in METRIC_NAMES
            }
        images.append(image)
    result = {
        "manifest_index": int(row["manifest_index"]),
        "scene_index": int(row["scene_index"]),
        "scene_frame_start": int(row["scene_frame_start"]),
        "token": str(row["token"]),
        "view_group": str(cache_payload["view_group"]),
        "window_start": window_start,
        "frames": sample.frames,
        "views": [pointforward.VIEW_NAMES[view] for view in sample.views],
        "dynamic_probability_mean": sample.dynamic_probability_mean,
        "num_queries": int(gaussian["means"].shape[0]),
        "metrics": {
            name: float(values[name].mean()) for name in METRIC_NAMES
        },
        "images": images,
    }
    if distt_values is not None:
        result["metrics_vs_distt"] = {
            name: float(distt_values[name].mean()) for name in METRIC_NAMES
        }
    del sample, gaussian, rendered_images, target_images, rendered_batch, target_batch, values
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
    clip_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for image in record["images"]:
            for name in METRIC_NAMES:
                clip_values[int(record["manifest_index"])][name].append(float(image[name]))

    def image_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            name: distribution([float(image[name]) for image in selected])
            for name in METRIC_NAMES
        }

    clip_summary = {
        name: distribution(
            [float(np.mean(values[name])) for values in clip_values.values()]
        )
        for name in METRIC_NAMES
    }
    by_camera = {
        view: image_summary([image for image in images if image["view"] == view])
        for view in pointforward.VIEW_NAMES
    }
    by_group = {
        group: image_summary(
            [
                image
                for record in records
                if record["view_group"] == group
                for image in record["images"]
            ]
        )
        for group in ("front", "rear")
    }
    by_window = {
        str(window_start): image_summary(
            [
                image
                for record in records
                if int(record["window_start"]) == window_start
                for image in record["images"]
            ]
        )
        for window_start in trainer.WINDOW_STARTS
    }
    return {
        "checkpoint_step": checkpoint_step,
        "protocol": {
            "clips": len(clip_values),
            "windows": len(records),
            "images": len(images),
            "window_starts": trainer.WINDOW_STARTS,
            "frames_per_window": 4,
            "views_per_clip": len(records[0]["views"]),
        },
        "clip_level": clip_summary,
        "image_level": image_summary(images),
        "by_camera": by_camera,
        "by_view_group": by_group,
        "by_window_start": by_window,
    }


def remap_distt_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remapped = []
    for record in records:
        remapped.append(
            {
                "manifest_index": record["manifest_index"],
                "view_group": record["view_group"],
                "window_start": record["window_start"],
                "images": [
                    {
                        "frame": image["frame"],
                        "view": image["view"],
                        **image["vs_distt"],
                    }
                    for image in record["images"]
                ],
            }
        )
    return remapped


@torch.no_grad()
def main() -> None:
    cli = parse_args()
    if cli.lpips_batch_size <= 0:
        raise ValueError("--lpips-batch-size must be positive")
    rank, world_size, local_rank, device = distributed_context()
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(checkpoint["step"])
    checkpoint_config = checkpoint["config"]
    views_per_clip = 1 if checkpoint_config.get("single_view_train", False) else 3
    if rank == 0:
        write_run_files(cli, world_size, checkpoint_step, views_per_clip)
    if world_size > 1:
        dist.barrier()

    rows = read_manifest(cli.manifest)
    if cli.cache_view_group != "all":
        rows = [
            row
            for row in rows
            if trainer.feature_cache.clip_views(int(row["manifest_index"]))[0]
            == cli.cache_view_group
        ]
    if cli.limit_clips > 0:
        rows = rows[: cli.limit_clips]
    trainer.validate_cache_paths(rows, cli.cache_root)
    assigned_rows = rows[rank::world_size]

    train_args = argparse.Namespace(**checkpoint_config)
    train_args.single_view_name = cli.single_view_name
    train_args.manifest = cli.manifest
    train_args.cache_root = cli.cache_root
    train_args.output_dir = cli.output_dir
    random.seed(train_args.seed + rank)
    np.random.seed(train_args.seed + rank)
    torch.manual_seed(train_args.seed + rank)

    model = trainer.FlowTrackRenderModel(train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    lpips_root = cli.lpips_module_root or getattr(train_args, "lpips_module_root", None)
    if lpips_root is not None and str(lpips_root) not in sys.path:
        sys.path.append(str(lpips_root))
    if cli.torch_home is not None:
        os.environ["TORCH_HOME"] = str(cli.torch_home)
    from lpips import LPIPS

    lpips_model = LPIPS(net="alex").to(device).eval().requires_grad_(False)
    infos = None
    depth_map = None
    masked_flow_index = None
    build_args = argparse.Namespace(
        ann_file=cli.ann_file,
        data_root=cli.data_root,
        height=train_args.height,
        width=train_args.width,
        low_height=53,
        low_width=100,
        masked_flow_rgb_root=cli.masked_flow_rgb_root,
        masked_flow_index=cli.masked_flow_index,
        allow_missing_flow_rgb=cli.allow_missing_flow_rgb,
    )
    query_source = trainer.PackagedQuerySource()

    metrics_path = cli.output_dir / f"metrics_rank{rank:02d}.jsonl"
    completed = load_completed(metrics_path) if cli.resume else set()
    mode = "a" if cli.resume else "w"
    start_time = time.perf_counter()
    completed_windows = 0
    with metrics_path.open(mode, encoding="utf-8") as stream:
        for clip_offset, row in enumerate(assigned_rows):
            payload = torch.load(row["clip_pt"], map_location="cpu", weights_only=False)
            if PREBUILT_CLIP_KEYS.issubset(payload):
                clip = payload
            else:
                if infos is None:
                    infos = query_data.load_ann(cli.ann_file)["infos"]
                    depth_map = query_data.load_json(cli.depth_map_json)
                    masked_flow_index = query_data.load_json(cli.masked_flow_index)
                payload["rdepth_root"] = str(cli.depth_root)
                payload["depth_map_json"] = str(cli.depth_map_json)
                clip = query_data.build_clip(
                    row, payload, infos, depth_map, build_args, masked_flow_index
                )["tensors"]
            cache_payload = torch.load(
                trainer.cache_path_for(row, cli.cache_root),
                map_location="cpu",
                weights_only=False,
            )
            distt_references = None
            if cli.distt_video_dir is not None:
                distt_references = load_distt_references(
                    cli.distt_video_dir,
                    cli.distt_video_pattern,
                    str(row["token"]),
                    [int(view) for view in cache_payload["view_indices"]],
                    int(clip.get("video_length", clip["rgb_target"].shape[0])),
                    train_args.width,
                    train_args.height,
                )
            for window_start in trainer.WINDOW_STARTS:
                key = (int(row["manifest_index"]), window_start)
                if key in completed:
                    continue
                train_args.time_origin = float(window_start)
                panel_dir = None
                if clip_offset < cli.visualize_clips_per_rank:
                    panel_dir = (
                        cli.output_dir
                        / "visuals"
                        / f"rank{rank:02d}_clip{int(row['manifest_index']):04d}_w{window_start:02d}"
                    )
                record = evaluate_window(
                    model,
                    row,
                    cache_payload,
                    clip,
                    query_source,
                    train_args,
                    lpips_model,
                    cli.lpips_batch_size,
                    device,
                    panel_dir,
                    distt_references,
                )
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                completed_windows += 1
                elapsed = time.perf_counter() - start_time
                progress_metrics = dict(record["metrics"])
                if "metrics_vs_distt" in record:
                    progress_metrics.update(
                        {
                            f"{name}_vs_distt": value
                            for name, value in record["metrics_vs_distt"].items()
                        }
                    )
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "local_rank": local_rank,
                            "manifest_index": int(row["manifest_index"]),
                            "window_start": window_start,
                            "completed_windows": completed_windows,
                            "elapsed_sec": elapsed,
                            "windows_per_sec": completed_windows / max(elapsed, 1.0e-6),
                            **progress_metrics,
                        }
                    ),
                    flush=True,
                )
            del payload, clip, cache_payload, distt_references

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        records = []
        for other_rank in range(world_size):
            path = cli.output_dir / f"metrics_rank{other_rank:02d}.jsonl"
            records.extend(
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            )
        records.sort(key=lambda record: (record["manifest_index"], record["window_start"]))
        expected_windows = len(rows) * len(trainer.WINDOW_STARTS)
        keys = {(record["manifest_index"], record["window_start"]) for record in records}
        if len(records) != expected_windows or len(keys) != expected_windows:
            raise RuntimeError(
                f"Expected {expected_windows} unique window records, found "
                f"{len(records)} records and {len(keys)} unique keys"
            )
        summary = summarize(records, checkpoint_step)
        if cli.distt_video_dir is not None:
            summary["comparisons"] = {
                "gaussian_vs_gt": {
                    key: summary[key]
                    for key in (
                        "clip_level",
                        "image_level",
                        "by_camera",
                        "by_view_group",
                        "by_window_start",
                    )
                },
                "gaussian_vs_distt": {
                    key: value
                    for key, value in summarize(
                        remap_distt_reference_records(records), checkpoint_step
                    ).items()
                    if key
                    in (
                        "clip_level",
                        "image_level",
                        "by_camera",
                        "by_view_group",
                        "by_window_start",
                    )
                },
            }
            summary["distt_video_dir"] = str(cli.distt_video_dir)
            summary["distt_video_pattern"] = cli.distt_video_pattern
        summary["checkpoint"] = str(cli.checkpoint)
        summary["elapsed_sec"] = time.perf_counter() - start_time
        summary["world_size"] = world_size
        (cli.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
