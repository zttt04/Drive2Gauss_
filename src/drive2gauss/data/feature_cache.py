#!/usr/bin/env python3
"""Cache frozen Turbo outputs once per 17-frame clip for multiscene PointForward training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from drive2gauss.training import static_decoder_pipeline as pointforward
from drive2gauss.data import manifest as dataset_manifest


FRONT_VIEWS = [0, 1, 2]
REAR_VIEWS = [3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--turbo-repo-root", type=Path, required=True)
    parser.add_argument("--turbo-config", type=Path, default=None)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--turbo-feature-key", default="up_block_2")
    parser.add_argument("--latent-scale", type=float, default=1.0 / pointforward.COGVIDEOX_SCALING_FACTOR)
    parser.add_argument("--depth-max-m", type=float, default=100.0)
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument(
        "--manifest-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit manifest rows to cache, useful for a bounded smoke test.",
    )
    parser.add_argument("--fast-dev-run", action="store_true")
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


def read_manifest(path: Path) -> list[dict]:
    return dataset_manifest.read_jsonl(path)


def clip_views(manifest_index: int) -> tuple[str, list[int]]:
    return ("front", FRONT_VIEWS) if manifest_index % 2 == 0 else ("rear", REAR_VIEWS)


def cache_path(cache_root: Path, row: dict) -> Path:
    group, _ = clip_views(int(row["manifest_index"]))
    return cache_root / (
        f"scene{int(row['scene_index']):04d}_start{int(row['scene_frame_start']):04d}_{group}.pt"
    )


def source_path(row: dict) -> Path:
    path = row.get("clip_pt", row.get("path"))
    if path is None:
        raise KeyError("Manifest row requires either 'clip_pt' or 'path'")
    return Path(path)


def cache_is_valid(path: Path, row: dict, feature_key: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    _, expected_views = clip_views(int(row["manifest_index"]))
    return (
        payload.get("manifest_index") == int(row["manifest_index"])
        and payload.get("feature_key") == feature_key
        and payload.get("view_indices") == expected_views
        and tuple(payload["rgb_u8"].shape[:3]) == (3, 3, 17)
        and tuple(payload["depth_m"].shape[:2]) == (3, 17)
        and tuple(payload["feature_maps"].shape[:3]) == (3, 32, 17)
    )


def decoded_to_metric_depth(decoded: torch.Tensor, depth_max_m: float) -> torch.Tensor:
    depth_norm = decoded.float().mean(dim=1).clamp(-1.0, 1.0)
    return ((depth_norm + 1.0) * 0.5) * float(depth_max_m)


def write_cache(
    row: dict,
    clip: dict,
    decoder,
    load_info: dict,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> dict:
    group, view_indices = clip_views(int(row["manifest_index"]))
    rgb_latent = clip["rgb_latent"][view_indices].to(device=device, dtype=torch.float16) * args.latent_scale
    depth_latent = clip["depth_latent"][view_indices].to(device=device, dtype=torch.float16) * args.latent_scale
    with torch.no_grad():
        decoded_rgb, feature_dict = decoder.decode(rgb_latent, feature_enabled=True)
        decoded_depth = decoder.decode(depth_latent, return_dict=False)[0]
    if args.turbo_feature_key not in feature_dict:
        raise KeyError(
            f"Unavailable Turbo feature {args.turbo_feature_key!r}: {sorted(feature_dict)}"
        )
    rgb_u8 = ((decoded_rgb[:, :, :17].float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte().cpu()
    depth_m = decoded_to_metric_depth(decoded_depth[:, :, :17], args.depth_max_m).half().cpu()
    feature_maps = feature_dict[args.turbo_feature_key][:, :, :17].half().cpu()
    if not bool(torch.isfinite(depth_m.float()).all()) or not bool(torch.isfinite(feature_maps.float()).all()):
        raise ValueError(f"Non-finite cached tensor for manifest index {row['manifest_index']}")
    payload = {
        "manifest_index": int(row["manifest_index"]),
        "scene_index": int(row["scene_index"]),
        "scene_frame_start": int(row["scene_frame_start"]),
        "source_path": str(source_path(row)),
        "view_group": group,
        "view_indices": view_indices,
        "feature_key": args.turbo_feature_key,
        "rgb_u8": rgb_u8,
        "depth_m": depth_m,
        "feature_maps": feature_maps,
        "decoder_load_info": load_info,
    }
    path = cache_path(args.cache_root, row)
    temporary = path.with_suffix(f".rank{rank}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "manifest_index": int(row["manifest_index"]),
        "path": str(path),
        "group": group,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = distributed_context()
    args.cache_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        config["world_size"] = world_size
        (args.cache_root / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        (args.cache_root / "command.sh").write_text(
            " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
        )
    rows = read_manifest(args.manifest)
    if args.manifest_indices is not None:
        selected = set(args.manifest_indices)
        rows = [row for row in rows if int(row["manifest_index"]) in selected]
        found = {int(row["manifest_index"]) for row in rows}
        if found != selected:
            raise ValueError(f"Manifest indices absent from dataset: {sorted(selected - found)}")
    if args.limit_clips > 0:
        rows = rows[: args.limit_clips]
    if args.fast_dev_run:
        rows = rows[: max(world_size, 1)]
    assigned = rows[rank::world_size]
    decoder, load_info = pointforward.build_turbo_decoder(args, device)
    start = time.perf_counter()
    built = skipped = 0
    for row in assigned:
        path = cache_path(args.cache_root, row)
        if cache_is_valid(path, row, args.turbo_feature_key):
            skipped += 1
            continue
        clip = torch.load(source_path(row), map_location="cpu", weights_only=False)
        record = write_cache(row, clip, decoder, load_info, args, device, rank)
        built += 1
        print(json.dumps({"rank": rank, "local_rank": local_rank, **record}), flush=True)
        del clip
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        records = []
        missing = []
        for row in rows:
            path = cache_path(args.cache_root, row)
            if cache_is_valid(path, row, args.turbo_feature_key):
                group, _ = clip_views(int(row["manifest_index"]))
                records.append({
                    "manifest_index": int(row["manifest_index"]),
                    "scene_index": int(row["scene_index"]),
                    "scene_frame_start": int(row["scene_frame_start"]),
                    "view_group": group,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                })
            else:
                missing.append(int(row["manifest_index"]))
        summary = {
            "manifest": str(args.manifest),
            "feature_key": args.turbo_feature_key,
            "num_expected": len(rows),
            "num_valid": len(records),
            "missing_manifest_indices": missing,
            "total_bytes": sum(record["bytes"] for record in records),
            "elapsed_sec": time.perf_counter() - start,
            "records": records,
        }
        (args.cache_root / "cache_index.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
