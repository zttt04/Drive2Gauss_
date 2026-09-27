import argparse
import json
import os
from pathlib import Path

import torch


VAE_OUT_CHANNELS = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge RGB-D latent cache payloads with flow-only latent cache payloads."
    )
    parser.add_argument("--rgbd-manifest", required=True, help="RGB-D cache manifest jsonl.")
    parser.add_argument("--flow-manifest", required=True, help="Flow-only cache manifest jsonl.")
    parser.add_argument("--output-dir", required=True, help="Merged RGB-D-flow cache directory.")
    parser.add_argument(
        "--flow-cache-dir",
        default=None,
        help="Optional local flow cache directory used when flow manifest paths are from another machine.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cache_key(row):
    return (
        int(row["video_length"]),
        str(row["token"]),
    )


def output_path_for_row(output_dir, row):
    dataset_index = int(row["dataset_index"])
    token = row["token"]
    shard_dir = Path(output_dir) / "clips" / f"{dataset_index // 1000:04d}"
    return shard_dir / f"{dataset_index:08d}_{token}_rgbd_flow_latent.pt"


def flow_path_for_row(row, flow_cache_dir):
    if flow_cache_dir is None:
        return row["path"]
    dataset_index = int(row["dataset_index"])
    token = row["token"]
    shard_dir = Path(flow_cache_dir) / "clips" / f"{dataset_index // 1000:04d}"
    return str(shard_dir / f"{dataset_index:08d}_{token}_flow_latent.pt")


def modality_latents(payload, source_path):
    latent = payload.get("latent")
    rgb_latent = payload.get("rgb_latent")
    depth_latent = payload.get("depth_latent")
    if rgb_latent is None:
        if latent is None:
            raise KeyError(f"Missing rgb_latent and latent in {source_path}")
        rgb_latent = latent[:, :VAE_OUT_CHANNELS]
    if depth_latent is None:
        if latent is None:
            raise KeyError(f"Missing depth_latent and latent in {source_path}")
        depth_latent = latent[:, VAE_OUT_CHANNELS : VAE_OUT_CHANNELS * 2]
    return rgb_latent.half(), depth_latent.half()


def validate_shapes(rgb_latent, depth_latent, flow_latent, flow_loss_mask, rgbd_path, flow_path):
    if rgb_latent.shape != depth_latent.shape:
        raise ValueError(
            f"RGB/depth latent shape mismatch in {rgbd_path}: "
            f"{tuple(rgb_latent.shape)} vs {tuple(depth_latent.shape)}"
        )
    expected_flow = (
        rgb_latent.shape[0],
        VAE_OUT_CHANNELS,
        rgb_latent.shape[2],
        rgb_latent.shape[3],
        rgb_latent.shape[4],
    )
    if tuple(flow_latent.shape) != tuple(expected_flow):
        raise ValueError(
            f"Flow latent shape mismatch for {flow_path}: "
            f"expected {expected_flow}, got {tuple(flow_latent.shape)}"
        )
    expected_mask = (
        rgb_latent.shape[0],
        1,
        rgb_latent.shape[2],
        rgb_latent.shape[3],
        rgb_latent.shape[4],
    )
    if flow_loss_mask is not None and tuple(flow_loss_mask.shape) != tuple(expected_mask):
        raise ValueError(
            f"Flow loss mask shape mismatch for {flow_path}: "
            f"expected {expected_mask}, got {tuple(flow_loss_mask.shape)}"
        )


def merge_payload(rgbd_row, flow_row, output_path, flow_cache_dir):
    rgbd_path = rgbd_row["path"]
    flow_path = flow_path_for_row(flow_row, flow_cache_dir)
    rgbd_payload = torch.load(rgbd_path, map_location="cpu")
    flow_payload = torch.load(flow_path, map_location="cpu")

    rgb_latent, depth_latent = modality_latents(rgbd_payload, rgbd_path)
    flow_latent = flow_payload.get("flow_latent")
    if flow_latent is None:
        raise KeyError(f"Missing flow_latent in {flow_path}")
    flow_latent = flow_latent.half()
    flow_loss_mask = flow_payload.get("flow_loss_mask")
    if flow_loss_mask is not None:
        flow_loss_mask = flow_loss_mask.half()

    validate_shapes(
        rgb_latent,
        depth_latent,
        flow_latent,
        flow_loss_mask,
        rgbd_path,
        flow_path,
    )

    payload = dict(rgbd_payload)
    payload["rgb_latent"] = rgb_latent
    payload["depth_latent"] = depth_latent
    payload["flow_latent"] = flow_latent
    if flow_loss_mask is not None:
        payload["flow_loss_mask"] = flow_loss_mask
        payload["flow_loss_mask_source"] = flow_payload.get("flow_loss_mask_source")
        payload["flow_loss_mask_shape"] = list(flow_loss_mask.shape)
    payload["latent"] = torch.cat([rgb_latent, depth_latent, flow_latent], dim=1)
    payload["latent_mode"] = "rgb_depth_masked_flow_sequence"
    payload["source_rgbd_latent_path"] = rgbd_path
    payload["source_flow_latent_path"] = flow_path
    payload["flow_representation"] = flow_payload.get("flow_representation")
    payload["flow_only_source"] = flow_payload.get("flow_only_source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, output_path)
    return payload


def main():
    args = parse_args()
    rgbd_rows = read_manifest(args.rgbd_manifest)
    flow_rows = read_manifest(args.flow_manifest)
    flow_by_key = {cache_key(row): row for row in flow_rows}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_train.jsonl"
    manifest_tmp = manifest_path.with_suffix(".jsonl.tmp")
    summary_path = output_dir / "summary.json"

    missing_flow = 0
    skipped_existing = 0
    merged_rows = 0
    shape_latent = None
    flow_latent_shape = None

    with open(manifest_tmp, "w", encoding="utf-8") as manifest:
        for rgbd_row in rgbd_rows:
            if args.limit is not None and merged_rows >= args.limit:
                break
            key = cache_key(rgbd_row)
            flow_row = flow_by_key.get(key)
            if flow_row is None:
                missing_flow += 1
                continue

            output_path = output_path_for_row(output_dir, rgbd_row)
            if args.skip_existing and output_path.exists():
                payload = torch.load(output_path, map_location="cpu")
                skipped_existing += 1
            else:
                payload = merge_payload(rgbd_row, flow_row, output_path, args.flow_cache_dir)

            shape_latent = list(payload["latent"].shape)
            flow_latent_shape = list(payload["flow_latent"].shape)
            new_row = dict(rgbd_row)
            new_row["path"] = str(output_path)
            new_row["shape_latent"] = shape_latent
            new_row["flow_latent_shape"] = flow_latent_shape
            new_row["latent_mode"] = "rgb_depth_masked_flow_sequence"
            new_row["source_rgbd_latent_path"] = rgbd_row["path"]
            new_row["source_flow_latent_path"] = flow_path_for_row(flow_row, args.flow_cache_dir)
            new_row["flow_loss_mask"] = "flow_loss_mask" in payload
            new_row["flow_representation"] = payload.get("flow_representation")
            manifest.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            merged_rows += 1

    os.replace(manifest_tmp, manifest_path)
    summary = {
        "rgbd_manifest": args.rgbd_manifest,
        "flow_manifest": args.flow_manifest,
        "flow_cache_dir": args.flow_cache_dir,
        "output_dir": str(output_dir),
        "input_rgbd_rows": len(rgbd_rows),
        "input_flow_rows": len(flow_rows),
        "output_rows": merged_rows,
        "missing_flow_rows": missing_flow,
        "skipped_existing_rows": skipped_existing,
        "shape_latent": shape_latent,
        "flow_latent_shape": flow_latent_shape,
        "latent_mode": "rgb_depth_masked_flow_sequence",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
