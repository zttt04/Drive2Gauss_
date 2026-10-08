"""Reader for the portable Drive2Gauss uint16 flow and motion-mask release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def make_colorwheel() -> np.ndarray:
    transitions = (15, 6, 4, 11, 13, 6)
    wheel = np.zeros((sum(transitions), 3), dtype=np.float32)
    column = 0
    ry, yg, gc, cb, bm, mr = transitions
    wheel[column : column + ry, 0] = 255
    wheel[column : column + ry, 1] = np.floor(255 * np.arange(ry) / ry)
    column += ry
    wheel[column : column + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    wheel[column : column + yg, 1] = 255
    column += yg
    wheel[column : column + gc, 1] = 255
    wheel[column : column + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    column += gc
    wheel[column : column + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    wheel[column : column + cb, 2] = 255
    column += cb
    wheel[column : column + bm, 2] = 255
    wheel[column : column + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    column += bm
    wheel[column : column + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    wheel[column : column + mr, 0] = 255
    return wheel


def flow_to_image_fixed_scale(flow: np.ndarray, scale: float = 64.0) -> np.ndarray:
    """Match the white-background RAFT/Middlebury encoding used for training."""
    normalized = flow.astype(np.float32) / float(scale)
    u, v = normalized[..., 0], normalized[..., 1]
    radius = np.sqrt(u * u + v * v)
    angle = np.arctan2(-v, -u) / np.pi
    position = (angle + 1.0) * 0.5 * (len(make_colorwheel()) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = (lower + 1) % len(make_colorwheel())
    fraction = position - lower
    wheel = make_colorwheel()
    image = np.empty((*flow.shape[:2], 3), dtype=np.uint8)
    for channel in range(3):
        color = (1.0 - fraction) * wheel[lower, channel] / 255.0
        color += fraction * wheel[upper, channel] / 255.0
        color = np.where(radius <= 1.0, 1.0 - radius * (1.0 - color), color * 0.75)
        image[..., channel] = np.floor(255.0 * np.clip(color, 0.0, 1.0)).astype(np.uint8)
    return image


class MotionRelease:
    """Token-indexed access to a path-portable motion release."""

    def __init__(self, root: Path, manifest: Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = Path(manifest or self.root / "manifest.jsonl").expanduser().resolve()
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        with self.manifest.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["source_token"]), str(row["camera"]))
                if key in self.records:
                    raise ValueError(f"Duplicate motion-release key: {key}")
                self.records[key] = row

    def load_masked_flow_rgb(
        self, source_token: str, target_token: str, camera: str, scale: float = 64.0
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (str(source_token), str(camera))
        if key not in self.records:
            raise KeyError(f"Motion release has no edge for token={source_token}, camera={camera}")
        row = self.records[key]
        if str(row["target_token"]) != str(target_token):
            raise RuntimeError(
                f"Motion target mismatch for token={source_token}, camera={camera}: "
                f"release={row['target_token']}, expected={target_token}"
            )
        encoded = cv2.imread(str(self.root / row["flow"]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(self.root / row["dynamic_mask"]), cv2.IMREAD_GRAYSCALE)
        if encoded is None:
            raise FileNotFoundError(self.root / row["flow"])
        if mask is None:
            raise FileNotFoundError(self.root / row["dynamic_mask"])
        if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
            raise ValueError(f"Expected uint16 HxWx3 flow PNG, got {encoded.dtype} {encoded.shape}")
        # OpenCV reads the release's logical RGB channels [U, V, valid] as BGR.
        flow = np.stack(
            (encoded[..., 2].astype(np.float32) - 32768.0, encoded[..., 1].astype(np.float32) - 32768.0),
            axis=-1,
        ) / 64.0
        flow_valid = encoded[..., 0] > 0
        dynamic_valid = flow_valid & (mask > 0)
        flow[~dynamic_valid] = 0.0
        return flow_to_image_fixed_scale(flow, scale), flow_valid.astype(np.uint8)
