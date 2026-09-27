from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


VIEW_ORDER = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
)
GT_RE = re.compile(r"^\d+_(?P<token>[0-9a-f]+)_f(?P<frame>\d+)_(?P<camera>CAM_.+)\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare matched frame trees for RGB decoder FID/FVD variants.")
    parser.add_argument("--variant", action="append", required=True, help="LABEL=VIDEO_DIR")
    parser.add_argument("--gt-flat-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def video_token(path: Path) -> str:
    return path.name.split("_", 1)[0]


def decode_task(task: tuple[str, int, str, str, int, int]) -> int:
    video_text, clip_index, token, output_text, frames, start_frame = task
    capture = cv2.VideoCapture(video_text)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {video_text}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output = Path(output_text)
    try:
        for frame_idx in range(1, frames + 1):
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"failed reading frame {start_frame + frame_idx - 1} from {video_text}")
            height, width = image.shape[:2]
            if height % 2 or width % 3:
                raise RuntimeError(f"invalid mosaic shape {width}x{height}: {video_text}")
            tile_h, tile_w = height // 2, width // 3
            for view_idx, camera in enumerate(VIEW_ORDER):
                row, column = (0, view_idx) if view_idx < 3 else (1, view_idx - 3)
                tile = image[row * tile_h:(row + 1) * tile_h, column * tile_w:(column + 1) * tile_w]
                target = output / f"{clip_index:08d}_{token}_f{frame_idx:02d}_{camera}.jpg"
                if not cv2.imwrite(str(target), tile, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed writing {target}")
    finally:
        capture.release()
    return frames * len(VIEW_ORDER)


def main() -> None:
    args = parse_args()
    variants = {}
    for value in args.variant:
        label, path = value.split("=", 1)
        variants[label] = Path(path)
    video_maps = {
        label: {video_token(path): path for path in sorted(directory.glob("*.mp4"))}
        for label, directory in variants.items()
    }
    token_sets = [set(mapping) for mapping in video_maps.values()]
    if not token_sets or any(tokens != token_sets[0] for tokens in token_sets[1:]):
        raise RuntimeError("variant token sets differ")
    tokens = sorted(token_sets[0])

    gt_index = {}
    for path in Path(args.gt_flat_dir).glob("*.jpg"):
        match = GT_RE.match(path.name)
        if match and match.group("token") in token_sets[0]:
            gt_index[(match.group("token"), int(match.group("frame")), match.group("camera"))] = path

    expected = len(tokens) * args.frames * len(VIEW_ORDER)
    output_root = Path(args.output_root)
    tasks = []
    for label, mapping in video_maps.items():
        generated = output_root / label / "generated"
        gt = output_root / label / "gt"
        generated.mkdir(parents=True, exist_ok=False)
        gt.mkdir(parents=True, exist_ok=False)
        for clip_index, token in enumerate(tokens):
            tasks.append((str(mapping[token]), clip_index, token, str(generated), args.frames, args.start_frame))
            for frame in range(1, args.frames + 1):
                for camera in VIEW_ORDER:
                    source = gt_index.get((token, frame, camera))
                    if source is None:
                        raise FileNotFoundError(f"missing GT: {token} frame={frame} camera={camera}")
                    os.symlink(source, gt / f"{clip_index:08d}_{token}_f{frame:02d}_{camera}.jpg")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        decoded = sum(executor.map(decode_task, tasks))
    summary = {
        "tokens": len(tokens), "frames": args.frames, "views": len(VIEW_ORDER),
        "expected_images_per_side": expected, "decoded_images_total": decoded,
        "variants": sorted(variants),
    }
    for label in variants:
        for side in ("gt", "generated"):
            count = len(list((output_root / label / side).glob("*.jpg")))
            if count != expected:
                raise RuntimeError(f"{label}/{side}: expected {expected}, got {count}")
    (output_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
