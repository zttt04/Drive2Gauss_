"""Compute original StyleGAN-V I3D features for the 3600/GT metric pair.

The historical FVD command consumes 16 RGB frames at 224x224 and returns a
400-dimensional feature.  This tool applies that exact preprocessing to the
0805 full-validation flat JPG directories, keeping the six cameras separate,
then writes immutable GT moments and a detached step-3600 queue for Stage4 FD.
It supports ``torchrun`` so the expensive feature extraction can use all eight
H20 GPUs without changing the source metric directories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from DISTT.utils.frechet_rgb_stage4 import (
    StyleGANVI3DFeatureExtractor,
    save_stage4_reference_stats,
)


CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)
_IMAGE_RE = re.compile(r"^(?P<token>.+)_f(?P<frame>[0-9]{2})_(?P<camera>CAM_[A-Z_]+)\.jpg$")


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _init_distributed() -> None:
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def _index_directory(root: Path) -> dict[str, dict[str, dict[int, Path]]]:
    index: dict[str, dict[str, dict[int, Path]]] = {}
    for path in sorted(root.glob("*.jpg")):
        match = _IMAGE_RE.match(path.name)
        if match is None:
            continue
        token = match.group("token")
        camera = match.group("camera")
        if camera not in CAMERAS:
            continue
        index.setdefault(token, {}).setdefault(camera, {})[int(match.group("frame"))] = path
    return index


def _valid_tokens(
    gt_index: dict[str, dict[str, dict[int, Path]]],
    generated_index: dict[str, dict[str, dict[int, Path]]],
    clip_length: int,
) -> list[str]:
    expected = set(range(1, clip_length + 1))
    tokens = []
    for token in sorted(set(gt_index).intersection(generated_index)):
        if all(
            set(gt_index[token].get(camera, {})) == expected
            and set(generated_index[token].get(camera, {})) == expected
            for camera in CAMERAS
        ):
            tokens.append(token)
    return tokens


def _load_clip(index: dict[str, dict[int, Path]], token: str, camera: str, clip_length: int, resolution: int) -> torch.Tensor:
    frames = []
    for frame in range(1, clip_length + 1):
        with Image.open(index[token][camera][frame]) as image:
            image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(image, dtype="uint8").copy()).permute(2, 0, 1).float().div(255.0))
    return torch.stack(frames, dim=1)


def _extract_kind(
    index: dict[str, dict[str, dict[int, Path]]],
    tokens: list[str],
    extractor: StyleGANVI3DFeatureExtractor,
    *,
    clip_length: int,
    resolution: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    results = []
    for start in range(0, len(tokens), batch_size):
        token_batch = tokens[start : start + batch_size]
        clips = torch.stack(
            [_load_clip(index, token, camera, clip_length, resolution) for token in token_batch for camera in CAMERAS],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            features = extractor.model(clips, rescale=False, resize=False, return_features=True)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            features = features.get("features", features.get("x"))
        if features is None:
            raise TypeError("StyleGAN-V I3D returned no features")
        if features.ndim > 2:
            features = features.flatten(1)
        if features.shape != (len(token_batch) * len(CAMERAS), 400):
            raise ValueError(f"expected I3D output [{len(token_batch) * len(CAMERAS)},400], got {tuple(features.shape)}")
        results.append(features.cpu().view(len(token_batch), len(CAMERAS), 400))
    return torch.cat(results, dim=0) if results else torch.empty((0, len(CAMERAS), 400))


def _per_view_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = torch.einsum("nvd,nve->vde", centered, centered) / max(features.shape[0] - 1, 1)
    return mean, 0.5 * (covariance + covariance.transpose(-1, -2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--i3d-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="clips per rank; each item contains six cameras")
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    _init_distributed()
    rank = _rank()
    world_size = _world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("I3D extraction requires CUDA")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    if args.clip_length != 16:
        raise ValueError("the original StyleGAN-V FVD feature contract is exactly 16 frames")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    gt_index = _index_directory(args.gt_dir)
    generated_index = _index_directory(args.generated_dir)
    tokens = _valid_tokens(gt_index, generated_index, args.clip_length)
    if not tokens:
        raise RuntimeError("no complete GT/generated token pairs found")
    local_tokens = tokens[rank::world_size]
    extractor = StyleGANVI3DFeatureExtractor(
        args.i3d_checkpoint,
        clip_length=args.clip_length,
        resolution=args.resolution,
        batch_size=max(args.batch_size * len(CAMERAS), 1),
        device=device,
    )
    for name, index in (("gt", gt_index), ("generated", generated_index)):
        features = _extract_kind(
            index,
            local_tokens,
            extractor,
            clip_length=args.clip_length,
            resolution=args.resolution,
            batch_size=max(args.batch_size, 1),
            device=device,
        )
        torch.save({"tokens": local_tokens, "features": features}, args.output_dir / f"{name}_rank{rank:02d}.pt")
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged = {}
        for name in ("gt", "generated"):
            records = []
            for shard_rank in range(world_size):
                payload = torch.load(args.output_dir / f"{name}_rank{shard_rank:02d}.pt", map_location="cpu")
                records.extend(zip(payload["tokens"], payload["features"]))
            feature_by_token = dict(records)
            merged[name] = torch.stack([feature_by_token[token] for token in tokens], dim=0)
            torch.save({"tokens": tokens, "features": merged[name]}, args.output_dir / f"{name}_features.pt")
        mu, cov = _per_view_moments(merged["gt"])
        metadata = {
            "feature_extractor": "StyleGAN-V I3D TorchScript",
            "feature_dim": 400,
            "clip_length": args.clip_length,
            "resolution": args.resolution,
            "cameras": list(CAMERAS),
            "num_tokens": len(tokens),
            "gt_dir": str(args.gt_dir),
            "generated_dir": str(args.generated_dir),
            "i3d_checkpoint": str(args.i3d_checkpoint),
        }
        save_stage4_reference_stats(
            args.output_dir / "stage4_rgb_i3d_gt_reference.pt",
            mu,
            cov,
            view_names=CAMERAS,
            metadata=metadata,
        )
        torch.save(
            {"features": merged["generated"], "view_names": CAMERAS, "metadata": metadata},
            args.output_dir / "stage4_rgb_i3d_step3600_queue.pt",
        )
        (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"saved {len(tokens)} token pairs: GT reference={args.output_dir / 'stage4_rgb_i3d_gt_reference.pt'} "
            f"queue={args.output_dir / 'stage4_rgb_i3d_step3600_queue.pt'}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
