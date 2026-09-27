# Data and release tools

These scripts prepare data or export intermediate artifacts. They are not the
model API and are intentionally kept outside `src/drive2gauss/`.

| Group | Scripts | Purpose |
| --- | --- | --- |
| Latent manifests | `build_generated_rgbd_flow_training_manifests.py`, `build_motion_release_manifest.py` | Index generated RGB-D-flow latents and motion metadata |
| Latent decoding | `decode_generated_rgbd_flow_latents.py` | Decode generator latents into RGB/depth/flow artifacts |
| Flow and masks | `compute_nuscenes_flow.py`, `run_nuscenes_dynamic_mask_batch.py`, `segment_dynamic_objects_with_sam.py`, `filter_sam_mask_with_nuscenes_velocity.py` | Build optical-flow and dynamic-object preprocessing outputs |
| Release exports | `export_depth_release.py`, `export_nuscenes_sky_masks.py`, `prepare_generated_flowtrack_fid_fvd_pairs.py` | Export reproducible evaluation or release artifacts |

Use each script's `--help` for its required paths and output contract. Generated
manifests, decoded data, and evaluation outputs belong under the configured data
or output roots, not in this directory.
