#!/usr/bin/env python3
"""Export DGGT sky-mask scene archives from existing OneFormer semantics."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-scene DGGT sky-mask tar archives from OneFormer NPZ files."
    )
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--token-map", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sky-class-id", type=int, default=27)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--start-scene", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=None)
    return parser.parse_args()


def load_scene_frames(metadata_root: Path) -> list[dict]:
    with (metadata_root / "scene.json").open() as fp:
        scenes = json.load(fp)
    with (metadata_root / "sample.json").open() as fp:
        samples = json.load(fp)
    sample_by_token = {row["token"]: row for row in samples}

    scene_frames = []
    for scene_index, scene in enumerate(scenes):
        frames = []
        token = scene["first_sample_token"]
        while token:
            sample = sample_by_token[token]
            frames.append(
                {
                    "token": token,
                    "timestamp": int(sample["timestamp"]),
                }
            )
            token = sample.get("next", "")
        scene_frames.append(
            {
                "scene_index": scene_index,
                "scene_token": scene["token"],
                "frames": frames,
            }
        )
    return scene_frames


def semantic_paths_for_frame(
    token: str,
    token_to_depth: dict[str, str],
    semantic_root: Path,
) -> list[Path] | None:
    relative_root = token_to_depth.get(token)
    if relative_root is None:
        return None
    frame_root = semantic_root / relative_root.lstrip("/")
    paths = [frame_root / camera / "semantic_oneformer.npz" for camera in CAMERAS]
    return paths if all(path.is_file() for path in paths) else None


def export_scene(
    scene: dict,
    token_to_depth: dict[str, str],
    semantic_root: str,
    output_dir: str,
    sky_class_id: int,
) -> dict:
    scene_index = int(scene["scene_index"])
    scene_name = f"{scene_index:03d}"
    output_root = Path(output_dir)
    archive_path = output_root / "scene_archives" / f"{scene_name}.tar"
    if archive_path.is_file():
        return {
            "scene_index": scene_index,
            "scene_token": scene["scene_token"],
            "status": "existing",
            "archive": str(archive_path),
        }

    semantic_root_path = Path(semantic_root)
    valid_frames = []
    missing_frames = []
    with tempfile.TemporaryDirectory(
        prefix=f"sky_{scene_name}_", dir=output_root / "work"
    ) as temporary_dir:
        scene_dir = Path(temporary_dir) / scene_name
        mask_dir = scene_dir / "sky_masks"
        mask_dir.mkdir(parents=True)

        for frame_index, frame in enumerate(scene["frames"]):
            token = frame["token"]
            semantic_paths = semantic_paths_for_frame(
                token, token_to_depth, semantic_root_path
            )
            if semantic_paths is None:
                missing_frames.append(
                    {
                        "frame_index": frame_index,
                        "token": token,
                        "timestamp": frame["timestamp"],
                    }
                )
                continue

            masks = []
            try:
                for semantic_path in semantic_paths:
                    semantic = np.load(semantic_path)["sem"]
                    masks.append((semantic == sky_class_id).astype(np.uint8) * 255)
            except (KeyError, OSError, ValueError) as error:
                missing_frames.append(
                    {
                        "frame_index": frame_index,
                        "token": token,
                        "timestamp": frame["timestamp"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            for camera_index, mask in enumerate(masks):
                mask_path = mask_dir / f"{frame_index:03d}_{camera_index}.png"
                if not cv2.imwrite(
                    str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                ):
                    raise OSError(f"Failed to write {mask_path}")
            valid_frames.append(
                {
                    "frame_index": frame_index,
                    "token": token,
                    "timestamp": frame["timestamp"],
                }
            )

        scene_manifest = {
            "scene_index": scene_index,
            "scene_token": scene["scene_token"],
            "frame_count": len(scene["frames"]),
            "valid_frame_count": len(valid_frames),
            "missing_frame_count": len(missing_frames),
            "valid_frames": valid_frames,
            "missing_frames": missing_frames,
            "sky_class_id": sky_class_id,
        }
        with (scene_dir / "sky_mask_frames.json").open("w") as fp:
            json.dump(scene_manifest, fp, indent=2)

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_archive = archive_path.with_suffix(".tar.partial")
        with tarfile.open(temporary_archive, "w") as archive:
            archive.add(scene_dir, arcname=scene_name)
        os.replace(temporary_archive, archive_path)

    return {
        **scene_manifest,
        "status": "exported",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "work").mkdir(exist_ok=True)
    scene_frames = load_scene_frames(args.metadata_root)
    end_scene = (
        len(scene_frames)
        if args.num_scenes is None
        else min(len(scene_frames), args.start_scene + args.num_scenes)
    )
    selected_scenes = scene_frames[args.start_scene:end_scene]
    with args.token_map.open() as fp:
        token_to_depth = json.load(fp)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                export_scene,
                scene,
                token_to_depth,
                str(args.semantic_root),
                str(args.output_dir),
                args.sky_class_id,
            )
            for scene in selected_scenes
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(futures)}] scene={result['scene_index']:03d} "
                f"status={result['status']} valid={result.get('valid_frame_count', '-')} "
                f"missing={result.get('missing_frame_count', '-')}",
                flush=True,
            )

    results.sort(key=lambda row: row["scene_index"])
    summary = {
        "metadata_root": str(args.metadata_root),
        "token_map": str(args.token_map),
        "semantic_root": str(args.semantic_root),
        "sky_class_id": args.sky_class_id,
        "scene_count": len(results),
        "exported_scene_count": sum(row["status"] == "exported" for row in results),
        "valid_frame_count": sum(row.get("valid_frame_count", 0) for row in results),
        "missing_frame_count": sum(row.get("missing_frame_count", 0) for row in results),
        "scenes": results,
    }
    with (args.output_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
