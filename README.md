# Drive2Gauss

[![Checkpoints](https://img.shields.io/badge/Checkpoints-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1qGS1ix6Krm3LLeDMmzKqd4sgKRCewm-c)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-34A853?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1XB83zwYeLrJ6WnIZ51adw_7Q9esjggSJ)

Drive2Gauss is a two-stage driving-scene pipeline. It generates six-view
RGB-D-flow video latents from nuScenes context, then decodes them as a
feed-forward dynamic Gaussian scene for reconstruction and novel-view
rendering:

```text
nuScenes context -> DiST-T RGB-D-flow latent -> decoded RGB-D-flow
                 -> feature-UNet Gaussian decoder -> rendered views
```

The released protocol uses 17 frames, six cameras, 424x800 resolution, and
RGB, metric depth, and masked optical-flow modalities. The two core model
stages live together under `src/drive2gauss/models/`:
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

See [docs/checkpoints.md](docs/checkpoints.md) for checkpoint identity and metrics,
[docs/dataset.md](docs/dataset.md) for dataset contents and access status, and
[docs/third_party_notices.md](docs/third_party_notices.md) before redistributing
code, weights, or data.

## Released resources

- **Checkpoints:** [Google Drive folder](https://drive.google.com/drive/folders/1qGS1ix6Krm3LLeDMmzKqd4sgKRCewm-c). Google sign-in may be required; see [checkpoint details](docs/checkpoints.md).
- **Dataset:** [Google Drive folder](https://drive.google.com/drive/folders/1XB83zwYeLrJ6WnIZ51adw_7Q9esjggSJ). The 700-scene flow/mask shards are available; depth-shard completeness is not yet verified. See [dataset details](docs/dataset.md).
- **Mini-sample:** [Download the one-clip input sample](https://github.com/zttt04/Drive2Gauss_/releases/tag/data-sample-v1) (78 MB; nuScenes terms apply).

## Installation

The Gaussian decoder uses Python 3.10, PyTorch 2.4.1, CUDA 12.1, and gsplat
1.5.3. The generator uses a separate MagicDrive/DiST-T-compatible environment.

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

The generator requirement file is an environment reference, not a portable
lock file; install its compiled dependencies according to the target server.

## Quick start

Copy the path template, edit it, and export it before using the launchers:

```bash
cp configs/paths.env.example configs/paths.env
set -a
source configs/paths.env
set +a
```

Edit the template with your local nuScenes, pretrained model, checkpoint,
dataset, annotation, and output paths. Set `DRIVE2GAUSS_DATASET_ROOT` to the
downloaded data release; its JSONL manifests use paths relative to this root.
Legacy absolute-path manifests and pre-rendered flow-index datasets remain
supported.

The launchers set `PYTHONPATH` and call the reorganized `src/` entry points.
Extra arguments are forwarded to the underlying Python entry points.

## Required files

```text
pretrained/CogVideoX-2b/
pretrained/t5-v1_1-xxl/
pretrained/Turbo-VAED-Cog.pth
checkpoints/drive2gauss_step3600/ema.pt
checkpoints/drive2gauss_feature_unet_step3744.pt
data/manifest_train.jsonl
data/manifest_val.jsonl
data/generated_latent_rgb_4f3v_train/
data/generated_latent_rgbd_flow_full_val/
```

For the one-clip mini-sample, download and extract the
[release archive](https://github.com/zttt04/Drive2Gauss_/releases/tag/data-sample-v1),
then point the two data roots at the extracted tree:

```bash
export DRIVE2GAUSS_DATASET_ROOT=/path/to/drive2gauss_sample
export NUSCENES_ROOT="${DRIVE2GAUSS_DATASET_ROOT}/nuscenes"
export DRIVE2GAUSS_LATENT_MANIFESTS=
```

This input-only archive contains RGB/depth/flow/mask data, annotations, and map
caches, but no generated latents or runtime feature caches. The sample reader
supports its relative paths and uint16 depth/flow files. See
[`docs/dataset.md`](docs/dataset.md) for contents and nuScenes terms.

With the released generator checkpoint and pretrained CogVideoX/T5 weights
configured in `configs/paths.env`, generate one 17-frame sample and decode its
RGB/depth/flow outputs:

```bash
GPUS=1 bash scripts/infer_generator.sh --cfg-options 'validation_index=[0]' num_frames=17
python tools/decode_generated_rgbd_flow_latents.py \
  --latent-root "${DRIVE2GAUSS_INFERENCE_ROOT}" \
  --output-dir "${DRIVE2GAUSS_DATASET_ROOT}/runtime/decoded" \
  --vae-pretrained "${DRIVE2GAUSS_PRETRAINED_ROOT}/CogVideoX-2b" \
  --max-latents 1 --fast-dev-run
```

This is an inference smoke test; the archive intentionally does not bundle
generated latents or Gaussian feature caches. Gaussian-decoder evaluation
requires those outputs plus the separately released Gaussian checkpoint.

The portable preprocessing release is organized as:

```text
data/
  manifest.jsonl
  depth/<scene>/<camera>/depth_<frame>.png
  flow_source/<scene>/<camera>/flow/flow_arrays/flow_<frame>.png
  masks/<scene>/<camera>/sam/masks/dynamic_object_mask_<frame>.png
```

## Commands

```bash
# Generate RGB-D-flow latents
bash scripts/infer_generator.sh

# Train or resume the generator
bash scripts/train_generator.sh

# Train the Gaussian decoder
bash scripts/train_gaussian_decoder.sh

# Reconstruct and evaluate generated latents
bash scripts/infer_gaussian_decoder.sh --limit-clips 2
```

The Gaussian decoder uses generated flow by default; `--zero-flow-input` is
only for ablations. Preprocessing and latent-manifest utilities are listed in
[`tools/README.md`](tools/README.md).

## Notes

- The two stages require separate environments and separately licensed model
  weights/data. nuScenes data and derived artifacts are subject to the
  [nuScenes terms](https://www.nuscenes.org/terms-of-use).
- Do not commit checkpoints, generated data, logs, or videos.
- See [`docs/checkpoints.md`](docs/checkpoints.md) and
  [`docs/third_party_notices.md`](docs/third_party_notices.md) for release and
  redistribution details.
