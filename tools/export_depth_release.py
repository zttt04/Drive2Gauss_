#!/usr/bin/env python3
"""Export flow-aligned train700 depth as sky-aware uint16 PNG files."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdepth-root", type=Path, required=True)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--png-compression", type=int, default=6)
    parser.add_argument("--sky-class-id", type=int, default=27)
    parser.add_argument("--sky-depth-meters", type=float, default=100.0)
    return parser.parse_args()


def export_camera(job):
    summary_path, rdepth_root, output_root, compression, sky_class, sky_depth = job
    summary_path = Path(summary_path)
    rdepth_root = Path(rdepth_root)
    output_root = Path(output_root)
    scene = summary_path.parents[2].name
    camera = summary_path.parents[1].name
    output_dir = output_root / scene / camera
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = json.loads(summary_path.read_text(encoding="utf-8"))["frame_summaries"]
    counts = {"expected": len(frame_rows), "exported": 0, "skipped": 0, "missing_depth": 0, "missing_semantic": 0}
    for row in frame_rows:
        index = int(row["frame"])
        output_path = output_dir / f"depth_{index:06d}.png"
        if output_path.is_file():
            counts["skipped"] += 1
            continue
        sample_root = rdepth_root / scene / row["source_token"] / camera
        depth_path = sample_root / "refined_depth.npz"
        semantic_path = sample_root / "semantic_oneformer.npz"
        if not depth_path.is_file():
            counts["missing_depth"] += 1
            continue
        if not semantic_path.is_file():
            counts["missing_semantic"] += 1
            continue
        depth = np.load(depth_path)["depth_pred"].astype(np.float32)
        semantic = np.load(semantic_path)["sem"]
        depth = cv2.resize(depth, (800, 424), interpolation=cv2.INTER_CUBIC)
        semantic = cv2.resize(semantic.astype(np.int32), (800, 424), interpolation=cv2.INTER_NEAREST)
        depth[semantic == sky_class] = sky_depth
        valid = np.isfinite(depth) & (depth > 0.0)
        encoded = np.zeros(depth.shape, dtype=np.uint16)
        encoded[valid] = np.rint(np.clip(depth[valid], 0.0, 65535.0 / 256.0) * 256.0).astype(np.uint16)
        if not cv2.imwrite(str(output_path), encoded, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
            raise OSError(f"Failed to write {output_path}")
        counts["exported"] += 1
    return scene, camera, counts


def main():
    args = parse_args()
    summaries = sorted(args.alignment_root.glob("*_scene/CAM_*/sam/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No alignment summaries under {args.alignment_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(path), str(args.rdepth_root), str(args.output_root), args.png_compression, args.sky_class_id, args.sky_depth_meters)
        for path in summaries
    ]
    totals = {"expected": 0, "exported": 0, "skipped": 0, "missing_depth": 0, "missing_semantic": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _, _, counts in executor.map(export_camera, jobs, chunksize=1):
            for key in totals:
                totals[key] += counts[key]
    summary = {
        **totals,
        "scene_camera_pairs": len(summaries),
        "height": 424,
        "width": 800,
        "meters_per_unit": 1.0 / 256.0,
        "zero_is_invalid": True,
        "sky_class_id": args.sky_class_id,
        "sky_depth_meters": args.sky_depth_meters,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
