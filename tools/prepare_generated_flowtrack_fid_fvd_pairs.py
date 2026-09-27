#!/usr/bin/env python3
"""Prepare matched flat render/raw-GT trees for FlowTrack FID and FVD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Any


VIEW_NAMES = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
FRAME_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_front_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        {**row, "dataset_split": split}
        for row in rows
        if row.get("query_view_group") == "front"
    ]


def load_infos(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise TypeError(f"Expected an annotation info list, found {type(infos).__name__}")
    return infos


def resolve_data_path(path: str, data_root: Path) -> Path:
    prefixes = ("../data/nuscenes/", "data/nuscenes/", "nuscenes/")
    for prefix in prefixes:
        if path.startswith(prefix):
            return data_root / path[len(prefix) :]
    candidate = Path(path)
    return candidate if candidate.is_absolute() else data_root / candidate


def flat_name(dataset_index: int, token: str, frame: int, view_name: str) -> str:
    return f"{dataset_index:08d}_{token}_f{frame:02d}_{view_name}"


def render_source_path(
    render_root: Path,
    split: str,
    token: str,
    frame: int,
    view_name: str,
) -> Path:
    window_start = (frame // 4) * 4
    return (
        render_root
        / "renders"
        / split
        / token
        / f"window_{window_start:02d}"
        / f"frame_{frame:02d}_{view_name}.png"
    )


def ensure_symlink(source: Path, destination: Path, resume: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.is_symlink() or destination.exists():
        if resume and destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing pair entry: {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    gt_dir = args.output_dir / "gt"
    render_dir = args.output_dir / "render"
    gt_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    rows = read_front_rows(args.train_manifest, "train")
    rows.extend(read_front_rows(args.val_manifest, "heldout"))
    infos = load_infos(args.ann_file)
    token_to_index = {str(info["token"]): index for index, info in enumerate(infos)}
    expected_links = len(rows) * FRAME_COUNT * len(VIEW_NAMES)
    linked = 0
    for dataset_index, row in enumerate(rows):
        token = str(row["token"])
        start = token_to_index.get(token)
        if start is None:
            raise KeyError(f"Manifest token absent from annotation: {token}")
        if start + FRAME_COUNT > len(infos):
            raise IndexError(f"Clip {token} exceeds annotation length at frame {FRAME_COUNT - 1}")
        for frame in range(FRAME_COUNT):
            info = infos[start + frame]
            for view_name in VIEW_NAMES:
                stem = flat_name(dataset_index, token, frame, view_name)
                gt_source = resolve_data_path(str(info["cams"][view_name]["data_path"]), args.data_root)
                render_source = render_source_path(
                    args.render_root,
                    str(row["dataset_split"]),
                    token,
                    frame,
                    view_name,
                )
                ensure_symlink(gt_source, gt_dir / f"{stem}{gt_source.suffix.lower()}", args.resume)
                ensure_symlink(render_source, render_dir / f"{stem}.png", args.resume)
                linked += 1
        if (dataset_index + 1) % 100 == 0 or dataset_index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "clips": dataset_index + 1,
                        "total_clips": len(rows),
                        "paired_images": linked,
                    }
                ),
                flush=True,
            )

    summary = {
        "status": "complete",
        "clips": len(rows),
        "train_clips": sum(row["dataset_split"] == "train" for row in rows),
        "heldout_clips": sum(row["dataset_split"] == "heldout" for row in rows),
        "frames_per_clip": FRAME_COUNT,
        "views": list(VIEW_NAMES),
        "paired_images": linked,
        "expected_paired_images": expected_links,
        "gt_dir": str(gt_dir),
        "render_dir": str(render_dir),
        "render_root": str(args.render_root),
        "annotation": str(args.ann_file),
        "data_root": str(args.data_root),
        "pair_type": "absolute_symlink",
    }
    (args.output_dir / "pair_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
