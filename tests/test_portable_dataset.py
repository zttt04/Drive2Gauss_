import json

import cv2
import numpy as np

from drive2gauss.data import manifest
from drive2gauss.data.motion_release import MotionRelease


def test_manifest_resolves_relative_paths_and_preserves_absolute_paths(tmp_path):
    absolute = tmp_path / "legacy.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "path_base": ".",
                "clip_pt": "latents/sample.pt",
                "generated_rgb_root": "decoded/rgb",
                "source_rgbd_latent_path": str(absolute),
            }
        )
        + "\n"
    )

    row = manifest.read_jsonl(manifest_path)[0]

    assert row["clip_pt"] == str((tmp_path / "latents/sample.pt").resolve())
    assert row["generated_rgb_root"] == str((tmp_path / "decoded/rgb").resolve())
    assert row["source_rgbd_latent_path"] == str(absolute)
    assert row["manifest_index"] == 0


def test_motion_release_decodes_uint16_flow_and_applies_dynamic_mask(tmp_path):
    flow_path = tmp_path / "flow.png"
    mask_path = tmp_path / "mask.png"
    encoded = np.zeros((2, 2, 3), dtype=np.uint16)
    encoded[..., 0] = 1
    encoded[..., 1] = 32768
    encoded[..., 2] = 32768 + 64
    mask = np.asarray([[255, 0], [255, 255]], dtype=np.uint8)
    assert cv2.imwrite(str(flow_path), encoded)
    assert cv2.imwrite(str(mask_path), mask)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "source_token": "source",
                "target_token": "target",
                "camera": "CAM_FRONT",
                "flow": "flow.png",
                "dynamic_mask": "mask.png",
            }
        )
        + "\n"
    )

    rgb, valid = MotionRelease(tmp_path).load_masked_flow_rgb(
        "source", "target", "CAM_FRONT"
    )

    assert rgb.shape == (2, 2, 3)
    assert valid.tolist() == [[1, 1], [1, 1]]
    assert rgb[0, 1].tolist() == [255, 255, 255]
    assert rgb[0, 0].tolist() != [255, 255, 255]
