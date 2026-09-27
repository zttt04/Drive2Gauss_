#!/usr/bin/env python3
"""Build train/validation manifests pairing generated latents with their decoded targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decode_generated_rgbd_flow_latents as generated_decode


DEFAULT_HELDOUT_SCENES = (31, 45, 54, 67, 78, 81, 102, 145)
FLOW_REPRESENTATION = {
    "name": "masked_optical_rgb_raft_whitebg",
    "channels": ["flow_color_r", "flow_color_g", "flow_color_b"],
    "colorwheel": "RAFT/Middlebury fixed-scale",
    "zero_flow": "white",
    "flow_scale_px": 64.0,
    "use_pixels_per_second": False,
    "last_frame": "zero_flow_white",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--decode-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-manifest-dir",
        type=Path,
        default=None,
        help="Optional prior train/val manifests whose token order should be preserved for cache reuse.",
    )
    parser.add_argument("--heldout-scenes", type=int, nargs="+", default=list(DEFAULT_HELDOUT_SCENES))
    parser.add_argument("--smoke-clips", type=int, default=8)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build from the currently complete subset; omit for formal manifests.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def paired_rows(reference: dict, latent_path: Path, decode_root: Path) -> list[dict]:
    token = str(reference["token"])
    base = {
        **reference,
        "path": str(latent_path),
        "clip_pt": str(latent_path),
        "shape_latent": [6, 48, 5, 53, 100],
        "flow_latent_shape": [6, 16, 5, 53, 100],
        "latent_mode": "rgb_plus_depth_flow_sequence",
        "source_rgbd_latent_path": str(reference["path"]),
        "flow_loss_mask": True,
        "flow_representation": FLOW_REPRESENTATION,
        "generated_latent_source": str(latent_path),
        "generated_rgb_root": str(decode_root / "gen_video"),
        "generated_flow_rgb_root": str(decode_root / "gen_flow"),
        "latent_domain": "step3600_generated",
    }
    return [{**base, "query_view_group": group} for group in ("front", "rear")]


def front_token_order(path: Path) -> list[str]:
    rows = read_jsonl(path)
    tokens = [str(row["token"]) for row in rows if row.get("query_view_group") == "front"]
    if len(tokens) * 2 != len(rows) or len(set(tokens)) != len(tokens):
        raise ValueError(f"Expected paired front/rear rows with unique clip tokens in {path}")
    return tokens


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latent_paths = generated_decode.discover_latents(args.latent_root)
    latent_by_token = {path.name.removesuffix("_latent.pt"): path for path in latent_paths}
    reference_rows = read_jsonl(args.reference_manifest)
    reference_by_token = {str(row["token"]): row for row in reference_rows}
    if len(reference_by_token) != len(reference_rows):
        raise ValueError(f"Duplicate tokens in reference manifest: {args.reference_manifest}")

    heldout_scenes = set(args.heldout_scenes)
    incomplete_tokens: list[str] = []
    absent_reference_tokens: list[str] = []
    complete_references: dict[str, dict] = {}
    for token, latent_path in latent_by_token.items():
        reference = reference_by_token.get(token)
        if reference is None:
            absent_reference_tokens.append(token)
            continue
        if not generated_decode.clip_is_complete(args.decode_root, token):
            incomplete_tokens.append(token)
            continue
        complete_references[token] = reference

    complete_clip_count = len(complete_references)

    if absent_reference_tokens:
        raise RuntimeError(f"Generated tokens absent from reference manifest: {absent_reference_tokens[:8]}")
    if incomplete_tokens and not args.allow_incomplete:
        raise RuntimeError(
            f"Decode is incomplete for {len(incomplete_tokens)} tokens; first={incomplete_tokens[:8]}"
        )
    if not args.allow_incomplete and complete_clip_count != len(latent_by_token):
        raise RuntimeError(f"Expected {len(latent_by_token)} complete clips, got {complete_clip_count}")

    reference_order = [str(row["token"]) for row in reference_rows if str(row["token"]) in complete_references]
    split_tokens = {
        "train": [
            token for token in reference_order
            if int(complete_references[token]["scene_index"]) not in heldout_scenes
        ],
        "val": [
            token for token in reference_order
            if int(complete_references[token]["scene_index"]) in heldout_scenes
        ],
    }
    if args.base_manifest_dir is not None:
        for split in ("train", "val"):
            base_tokens = front_token_order(args.base_manifest_dir / f"manifest_{split}.jsonl")
            missing_base = [token for token in base_tokens if token not in split_tokens[split]]
            if missing_base:
                raise RuntimeError(f"Base {split} tokens absent from complete split: {missing_base[:8]}")
            base_set = set(base_tokens)
            split_tokens[split] = base_tokens + [
                token for token in split_tokens[split] if token not in base_set
            ]

    train_rows = [
        row
        for token in split_tokens["train"]
        for row in paired_rows(complete_references[token], latent_by_token[token], args.decode_root)
    ]
    val_rows = [
        row
        for token in split_tokens["val"]
        for row in paired_rows(complete_references[token], latent_by_token[token], args.decode_root)
    ]

    smoke_rows = []
    seen_scenes = set()
    for index in range(0, len(train_rows), 2):
        front, rear = train_rows[index : index + 2]
        scene = int(front["scene_index"])
        if scene in seen_scenes:
            continue
        smoke_rows.extend([front, rear])
        seen_scenes.add(scene)
        if len(seen_scenes) >= args.smoke_clips:
            break

    write_jsonl(args.output_dir / "manifest_train.jsonl", train_rows)
    write_jsonl(args.output_dir / "manifest_val.jsonl", val_rows)
    write_jsonl(args.output_dir / "manifest_smoke.jsonl", smoke_rows)
    summary = {
        "reference_manifest": str(args.reference_manifest),
        "latent_root": str(args.latent_root),
        "decode_root": str(args.decode_root),
        "base_manifest_dir": None if args.base_manifest_dir is None else str(args.base_manifest_dir),
        "latent_candidates": len(latent_by_token),
        "complete_generated_clips": complete_clip_count,
        "incomplete_generated_clips": len(incomplete_tokens),
        "train_clips": len(train_rows) // 2,
        "train_manifest_rows": len(train_rows),
        "val_clips": len(val_rows) // 2,
        "val_manifest_rows": len(val_rows),
        "train_scenes": len({int(row["scene_index"]) for row in train_rows}),
        "heldout_scene_indices": sorted(heldout_scenes),
        "train_excludes_all_heldout_scenes": not any(
            int(row["scene_index"]) in heldout_scenes for row in train_rows
        ),
        "required_pairing": "same-run generated latent + decoded RGB/depth/flow, 17 frames x 6 views",
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
