# Drive2Gauss

Drive2Gauss generates six-view RGB-D-flow video latents and reconstructs them
as a feed-forward dynamic Gaussian scene:

```text
nuScenes context -> DiST-T RGB-D-flow latent -> decoded RGB-D-flow
                 -> feature-UNet Gaussian decoder -> rendered views
```

The two core model stages live together under `src/drive2gauss/models/`:
`generator/` contains the integrated RGB-D-flow video generator, while the
Gaussian decoder is defined in the adjacent Python modules. Training,
inference, rendering, data loading, and evaluation live in their corresponding
`src/drive2gauss/` packages. The top-level `scripts/` directory contains only
thin user entry points, while `tools/` is reserved for preprocessing and
artifact export.

The Gaussian decoder architecture is defined independently from its training
loop:

```text
src/drive2gauss/models/gaussian_modules.py
    StaticPointForwardModel, FeatureRenderUNet, and Gaussian rendering primitives

src/drive2gauss/models/gaussian_decoder.py
    Drive2GaussGaussianDecoder

src/drive2gauss/models/generator/
    RGB-D-flow video generator
```

The remaining source layout is:

```text
src/drive2gauss/data/          datasets, latent caches, and query sampling
src/drive2gauss/training/      generator and Gaussian-decoder training loops
src/drive2gauss/inference/     video generation, reconstruction, and novel views
src/drive2gauss/rendering/     time-dependent gsplat rendering
src/drive2gauss/evaluation/    reconstruction metrics and full-set evaluation
```

The official Gaussian decoder consumes generated flow normally. Do not pass
`--zero-flow-input`; that flag exists only for ablations. Direct-RGB
experiments are not part of the released result.

See [CHECKPOINTS.md](CHECKPOINTS.md) for checkpoint identity and metrics and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing code,
weights, or data.

## Environment

The two stages use separate environments because their CUDA extension stacks
are not interchangeable:

| Stage | Python | PyTorch stack | Important compiled dependencies |
| --- | --- | --- | --- |
| RGB-D-flow generator | 3.10 | 2.4.0 + CUDA 11.8 reference environment | MagicDrive-compatible ColossalAI, FlashAttention, MMCV/MMDetection |
| Gaussian decoder | 3.10 | 2.4.1 + CUDA 12.1 | torchvision 0.19.1, gsplat 1.5.3 |

The Gaussian decoder was validated on NVIDIA H20 GPUs. Full 960k-query
decoder training peaks near 81 GB.

```bash
conda create -n drive2gauss-gaussian python=3.10 -y
conda activate drive2gauss-gaussian
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install gsplat==1.5.3
pip install --no-deps -e .
python -m compileall -q src/drive2gauss tools
pytest -q tests
```

DiST-T additionally needs its MagicDriveDiT-compatible ColossalAI fork and a
FlashAttention build matching the local CUDA/PyTorch ABI. CogVideoX, T5,
Turbo-VAED, SEA-RAFT, Grounded-SAM 2, and nuScenes are separately licensed.

Do not install `src/drive2gauss/models/generator/requirement/distt.txt` directly: it is an
environment capture containing machine-local Conda URLs and mutually
exclusive MMCV variants. Use it only as a version reference when recreating
the DiST-T environment. In particular, the captured file contains both
`mmcv==2.2.0` and `mmcv-full==1.7.2`; that list is provenance, not a clean pip
lock file.

## Quick start

Copy the path template, edit it, and export it before using the launchers:

```bash
cp configs/paths.env.example configs/paths.env
set -a
source configs/paths.env
set +a
```

The four official entry points are:

```bash
# Train and infer the RGB-D-flow video generator.
bash scripts/train_generator.sh
bash scripts/infer_generator.sh

# Train and infer the feed-forward Gaussian decoder.
bash scripts/train_gaussian_decoder.sh
bash scripts/infer_gaussian_decoder.sh
```

Extra arguments are forwarded to the underlying Python entry point. For a
bounded decoder inference check, for example:

```bash
bash scripts/infer_gaussian_decoder.sh --limit-clips 2
```

## Files

```text
pretrained/CogVideoX-2b/
pretrained/t5-v1_1-xxl/
pretrained/Turbo-VAED-Cog.pth
checkpoints/drive2gauss_distt_step3600/
checkpoints/drive2gauss_feature_unet_step3744.pt
data/manifest_train.jsonl
data/manifest_val.jsonl
data/generated_latent_rgb_4f3v_train/
data/generated_latent_rgbd_flow_full_val/
```

The generator config reads `NUSCENES_ROOT`, `DEPTH_ROOT_JSON`, `RDEPTH_ROOT`,
`DRIVE2GAUSS_PRETRAINED_ROOT`, `DRIVE2GAUSS_LATENT_MANIFESTS`, and
`DRIVE2GAUSS_OUTPUT_ROOT`. Set `DRIVE2GAUSS_RESUME_CHECKPOINT` when continuing
training from a released checkpoint.

## Generate RGB-D-flow latents

The recommended entry point is:

```bash
bash scripts/infer_generator.sh
```

Keep 17 frames, six views, RGB-D-flow decoding, and 30 sampling steps for the
released protocol. When resuming diffusion training with a changed GPU count,
use `reset_sampler=True`; preserve sampler state for the same world size.
`max_train_steps` is an absolute global-step ceiling after resume.

## Decode and index generated latents

```bash
torchrun --nproc_per_node=8 tools/decode_generated_rgbd_flow_latents.py \
  --latent-root /path/to/generated_latents \
  --output-dir /path/to/decoded_generated_latents \
  --vae-pretrained /path/to/pretrained/CogVideoX-2b

python tools/build_generated_rgbd_flow_training_manifests.py \
  --reference-manifest /path/to/reference_manifest.jsonl \
  --latent-root /path/to/generated_latents \
  --decode-root /path/to/decoded_generated_latents \
  --output-dir /path/to/generated_dataset
```

Use each command's `--help` for naming variants emitted by a particular
generator run.

## Reconstruct and evaluate generated latents

This is the canonical decoder command. It does not enable zero-flow or the
direct-RGB experimental head:

```bash
bash scripts/infer_gaussian_decoder.sh
```

The released checkpoint predates the optional `zero_flow_input` config field.
The loader supplies the normal value `False` in memory. It neither edits the
checkpoint nor changes its generated flow input. Add `--limit-clips 2` for a
bounded infrastructure check.

Full-set infer-reference results over 1,976 clips / 94,848 images are PSNR
29.90838, SSIM 0.883813, and LPIPS-Alex 0.101202. They compare the Gaussian
render with RGB decoded from the same generated latent, not raw nuScenes RGB.

## Fine-tune the Gaussian decoder

The published checkpoint is already trained for generated latents. For an
intentional fine-tune, load its model and create a fresh optimizer:

```bash
export GAUSSIAN_INIT_CHECKPOINT="${GAUSSIAN_CHECKPOINT}"
bash scripts/train_gaussian_decoder.sh \
  --unfreeze-backbone --lr 1e-4 --backbone-lr 1e-4
```

Gaussian decoder training constructs camera geometry and flow queries online.
In addition to the manifests and feature caches, set the `GAUSSIAN_DATA_ROOT`,
`GAUSSIAN_{TRAIN,VAL}_ANN_FILE`, `GAUSSIAN_{TRAIN,VAL}_FLOW_RGB_ROOT`, and
`GAUSSIAN_{TRAIN,VAL}_FLOW_INDEX` values shown in `configs/paths.env.example`.
The annotation split must match the source split of each manifest.

`--resume-checkpoint` is only for an exact continuation with identical
architecture, trainable modules, and optimizer groups. It is not
interchangeable with `--init-checkpoint`.

## Encodings

Flow PNGs are uint16 with `U=round(flow_x*64+32768)`,
`V=round(flow_y*64+32768)`, and validity in channel three. Metric depth PNGs
use `round(depth_m*256)`, zero for invalid pixels, and 100 m for sky.
