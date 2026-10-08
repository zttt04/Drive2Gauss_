# Drive2Gauss dataset release

The Drive2Gauss dataset folder is hosted on
[Google Drive](https://drive.google.com/drive/folders/1XB83zwYeLrJ6WnIZ51adw_7Q9esjggSJ).
The folder contains a release `README.md` and `manifest_summary.json` alongside
the uploaded shards. Google Drive access and sharing settings apply.

## Available flow and motion-mask data

The completed flow/mask release supplements nuScenes training scenes with
optical flow and dynamic-object masks. It does **not** redistribute the source
nuScenes RGB images.

- 700 training scenes, six cameras per scene
- 958,912 aligned flow/mask pairs
- 700 scene tar files, 233,189,048,320 bytes total
- Resolution: 424 x 800; source camera streams: interpolated 12 Hz

Each flow sample describes motion from frame `index` to `index + 1`. The
dynamic-object mask is aligned to the source frame. The scene, camera, and
frame indices align between flow and mask trees.

```text
flow_source/
  <scene>_scene/<CAMERA>/flow/flow_arrays/flow_<index>.png
masks/
  <scene>_scene/<CAMERA>/sam/masks/dynamic_object_mask_<index>.png
```

Flow files are three-channel uint16 PNGs. In logical RGB order:

```text
flow_x = (U - 32768) / 64
flow_y = (V - 32768) / 64
flow_valid = valid > 0
```

OpenCV reads PNG channels in BGR order, so its unchanged buffer is `[valid, V,
U]`. Masks are binary uint8 PNGs (`0` background, `255` selected moving
objects), generated from SAM instances filtered by projected nuScenes moving
boxes. Vehicles use a 0.5 m/s speed threshold, other dynamic classes use
0.1 m/s, and projected boxes are dilated by 12 pixels.

Keep flow validity and dynamic-object masks as separate signals. For dynamic
motion supervision, use `flow_valid & (dynamic_object_mask > 0)`.

## Depth status

A separate depth-shard upload was started, but its completion and exact remote
coverage have not been verified. Do not assume the Drive folder contains a
complete depth release until a complete shard listing and manifest are
confirmed. The published 700-shard count and byte total above refer only to
flow and masks.

## Mini-sample status

An input-only, one-clip sample has been prepared and tested locally on the
server. It contains one 17-frame, six-camera nuScenes clip, RGB/depth/flow/mask
inputs, minimal annotations and map cache, and a relative-path manifest; it
does not include generated latents or runtime caches. The archive is not
attached to this repository yet because it contains nuScenes RGB and derived
data. Public redistribution must comply with the
[nuScenes terms of use](https://www.nuscenes.org/terms-of-use), including the
applicable non-commercial and attribution/share-alike requirements. Obtain
nuScenes through its authorized distribution if the sample is not made
available here.

## License and citation

The repository's BSD-3-Clause license covers original code only. nuScenes data
and derived artifacts remain subject to the nuScenes terms; this page does not
grant rights to nuScenes. Users must obtain the source dataset through its
authorized channel and comply with its terms. See
[`third_party_notices.md`](third_party_notices.md).
