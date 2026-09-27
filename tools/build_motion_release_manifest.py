#!/usr/bin/env python3
"""Build a path-portable manifest for Drive2Gauss flow and motion masks."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--flow-tree", default="flow_source")
    parser.add_argument("--mask-tree", default="masks")
    parser.add_argument("--output", default="manifest.jsonl")
    parser.add_argument("--summary", default="manifest_summary.json")
    parser.add_argument("--require-files", action="store_true")
    return parser.parse_args()


def relative_sample_paths(scene, camera, pair_index, flow_tree, mask_tree):
    flow_path = Path(flow_tree) / scene / camera / "flow" / "flow_arrays" / f"flow_{pair_index:06d}.png"
    mask_path = Path(mask_tree) / scene / camera / "sam" / "masks" / f"dynamic_object_mask_{pair_index:06d}.png"
    return flow_path, mask_path


def main():
    args = parse_args()
    release_root = args.release_root.resolve()
    flow_root = release_root / args.flow_tree
    metrics_paths = sorted(flow_root.glob("*_scene/CAM_*/flow/metrics.csv"))
    if not metrics_paths:
        raise FileNotFoundError(f"No camera metrics found under {flow_root}")

    output_path = release_root / args.output
    summary_path = release_root / args.summary
    missing_flow = []
    missing_mask = []
    record_count = 0
    scene_camera_pairs = set()

    with output_path.open("w", encoding="utf-8") as output_file:
        for metrics_path in metrics_paths:
            camera = metrics_path.parent.parent.name
            scene = metrics_path.parent.parent.parent.name
            scene_camera_pairs.add((scene, camera))
            with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
                for row in csv.DictReader(metrics_file):
                    pair_index = int(row["pair_index"])
                    flow_path, mask_path = relative_sample_paths(
                        scene, camera, pair_index, args.flow_tree, args.mask_tree
                    )
                    if args.require_files:
                        if not (release_root / flow_path).is_file():
                            missing_flow.append(flow_path.as_posix())
                        if not (release_root / mask_path).is_file():
                            missing_mask.append(mask_path.as_posix())
                    record = {
                        "scene": scene,
                        "camera": camera,
                        "pair_index": pair_index,
                        "source_token": row["source_token"],
                        "target_token": row["target_token"],
                        "source_filename": row["source_filename"],
                        "target_filename": row["target_filename"],
                        "dt_seconds": float(row["dt_seconds"]),
                        "height": int(row["height"]),
                        "width": int(row["width"]),
                        "flow": flow_path.as_posix(),
                        "dynamic_mask": mask_path.as_posix(),
                    }
                    output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    record_count += 1

    summary = {
        "records": record_count,
        "scene_camera_pairs": len(scene_camera_pairs),
        "scenes": len({scene for scene, _ in scene_camera_pairs}),
        "cameras": sorted({camera for _, camera in scene_camera_pairs}),
        "files_checked": bool(args.require_files),
        "missing_flow": len(missing_flow),
        "missing_mask": len(missing_mask),
        "missing_flow_examples": missing_flow[:20],
        "missing_mask_examples": missing_mask[:20],
        "manifest": args.output,
        "paths_are_relative_to": ".",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.require_files and (missing_flow or missing_mask):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
