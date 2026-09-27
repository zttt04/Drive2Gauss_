import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "src"
    / "drive2gauss"
    / "inference"
    / "gaussian_decoder.py"
)
SPEC = importlib.util.spec_from_file_location("drive2gauss.inference.gaussian_decoder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_read_front_rows_keeps_original_manifest_indices(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"token": "a", "query_view_group": "front"},
        {"token": "a", "query_view_group": "rear"},
        {"token": "b", "query_view_group": "front"},
        {"token": "b", "query_view_group": "rear"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = MODULE.read_front_rows(manifest, "heldout")

    assert [row["token"] for row in selected] == ["a", "b"]
    assert [row["manifest_index"] for row in selected] == [0, 2]
    assert {row["dataset_split"] for row in selected} == {"heldout"}


def test_render_paths_are_unique_across_splits_windows_frames_and_views(tmp_path):
    first = MODULE.render_image_path(
        tmp_path, "train", "token", 0, 0, "CAM_FRONT"
    )
    second = MODULE.render_image_path(
        tmp_path, "heldout", "token", 4, 4, "CAM_FRONT_RIGHT"
    )

    assert first != second
    assert first.name == "frame_00_CAM_FRONT.png"
    assert second.name == "frame_04_CAM_FRONT_RIGHT.png"


def test_summary_averages_requested_metrics_per_image():
    metrics_a = {
        "psnr_infer": 20.0,
        "lpips_infer_alex": 0.2,
        "lpips_gt_alex": 0.4,
        "ssim_infer": 0.8,
    }
    metrics_b = {
        "psnr_infer": 30.0,
        "lpips_infer_alex": 0.1,
        "lpips_gt_alex": 0.3,
        "ssim_infer": 0.9,
    }
    records = [
        {
            "dataset_split": "train",
            "manifest_index": 0,
            "window_start": 0,
            "images": [
                {"frame": 0, "view": "CAM_FRONT", **metrics_a},
                {"frame": 1, "view": "CAM_FRONT", **metrics_b},
            ],
        }
    ]

    summary = MODULE.summarize(records, checkpoint_step=42)

    assert summary["protocol"]["images"] == 2
    assert np.isclose(summary["image_level"]["psnr_infer"]["mean"], 25.0)
    assert np.isclose(summary["image_level"]["lpips_gt_alex"]["mean"], 0.35)


def test_ssim_is_one_for_identical_images():
    image = torch.rand(2, 3, 32, 48)

    values = MODULE.ssim_per_image(image, image)

    assert torch.allclose(values, torch.ones_like(values), atol=1.0e-5)


def test_released_checkpoint_defaults_to_normal_flow_input():
    args = MODULE.checkpoint_train_args({"appearance_mode": "feature_unet"})

    assert args.appearance_mode == "feature_unet"
    assert args.zero_flow_input is False


def test_checkpoint_preserves_explicit_flow_setting():
    args = MODULE.checkpoint_train_args(
        {"appearance_mode": "feature_unet", "zero_flow_input": True}
    )

    assert args.zero_flow_input is True
