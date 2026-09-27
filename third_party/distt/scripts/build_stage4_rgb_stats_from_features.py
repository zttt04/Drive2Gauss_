"""Build fixed per-view GT moments and an optional step-3600 feature queue.

The feature extractor/decode job can write tensors with shape ``[N, NC, D]``.
This small packaging step keeps the immutable GT reference statistics separate
from detached generated features used only to warm the online population.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from DISTT.utils.frechet_rgb_stage4 import save_stage4_reference_stats


def _load_features(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("features", payload.get("rgb_features"))
    features = torch.as_tensor(payload, dtype=torch.float32)
    if features.ndim != 3:
        raise ValueError(f"expected [N, NC, D] features, got {tuple(features.shape)} from {path}")
    if features.shape[0] < 2:
        raise ValueError("at least two feature rows are required for covariance")
    if not torch.isfinite(features).all():
        raise ValueError(f"non-finite feature value in {path}")
    return features


def _per_view_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = torch.einsum("nvd,nve->vde", centered, centered) / (features.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    return mean, covariance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-features", type=Path, required=True, help="GT RGB feature tensor [N, NC, D]")
    parser.add_argument("--output", type=Path, required=True, help="output .pt reference-stat file")
    parser.add_argument("--queue-features", type=Path, default=None, help="optional step-3600 features [N, NC, D]")
    parser.add_argument("--view-name", action="append", default=None, help="repeat once per camera")
    args = parser.parse_args()

    gt_features = _load_features(args.gt_features)
    mu, cov = _per_view_moments(gt_features)
    view_names = args.view_name or [f"view_{index}" for index in range(gt_features.shape[1])]
    if len(view_names) != gt_features.shape[1]:
        raise ValueError("number of --view-name values must match NC")
    save_stage4_reference_stats(
        args.output,
        mu,
        cov,
        view_names=view_names,
        metadata={
            "feature_shape": list(gt_features.shape),
            "reference_source": "gt_rgb",
            "queue_source": "step3600_rgb" if args.queue_features else None,
        },
    )
    print(f"saved GT reference stats: rows={gt_features.shape[0]} views={gt_features.shape[1]} dim={gt_features.shape[2]} output={args.output}")
    if args.queue_features is not None:
        queue = _load_features(args.queue_features)
        if queue.shape[1:] != gt_features.shape[1:]:
            raise ValueError(f"queue shape {tuple(queue.shape)} does not match GT feature shape {tuple(gt_features.shape)}")
        queue_path = args.output.with_name(args.output.stem + "_queue.pt")
        torch.save({"features": queue, "source": "step3600_rgb", "view_names": view_names}, queue_path)
        print(f"saved detached generated queue: rows={queue.shape[0]} output={queue_path}")


if __name__ == "__main__":
    main()
