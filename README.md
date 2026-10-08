# Drive2Gauss

> **TL;DR:** Drive2Gauss converts jointly generated multiview RGB, metric depth, and dynamic flow into a dynamic Gaussian scene in one feed-forward pass, enabling reconstruction and novel-view rendering.

[Project Page](https://zttt04.github.io/Drive2Gauss_page/) | [Code](https://github.com/zttt04/Drive2Gauss_) | [Checkpoints](https://drive.google.com/drive/folders/1qGS1ix6Krm3LLeDMmzKqd4sgKRCewm-c) | [Dataset](https://drive.google.com/drive/folders/1XB83zwYeLrJ6WnIZ51adw_7Q9esjggSJ) | [Mini-sample](https://github.com/zttt04/Drive2Gauss_/releases/tag/data-sample-v1)

<p align="center">
Tong Zhao<sup>1,2</sup>, Zhiyuan Han<sup>2</sup>, Lin Chen<sup>3</sup>, Bowen Xie<sup>4</sup>,<br>
Yunxi Qiao<sup>5</sup>, Bohan Li<sup>6</sup>, Weiqing Xiao<sup>7</sup>, Gen Li<sup>8</sup>,<br>
Shu Han<sup>9</sup>, Xuyang Dai<sup>10</sup>, Cheng Bi<sup>10</sup>, Hao Zhao<sup>1</sup>,<br>
Chaojian Li<sup>11,†</sup>
</p>

<p align="center">
<sup>1</sup> AIR, Tsinghua University · <sup>2</sup> Tongji University · <sup>3</sup> Beijing Technology and Business University · <sup>4</sup> Fuzhou University<br>
<sup>5</sup> Tsinghua University · <sup>6</sup> Shanghai Jiao Tong University · <sup>7</sup> Nanjing University · <sup>8</sup> Zhejiang University<br>
<sup>9</sup> University of Wisconsin–Madison · <sup>10</sup> Great Wall Motor · <sup>11</sup> The Hong Kong University of Science and Technology<br>
<sup>†</sup> Corresponding author
</p>

```text
nuScenes context -> DiST-T RGB-D-flow video latents -> decoded RGB-D-flow
                 -> feature-UNet Gaussian decoder -> rendered views
```

The released protocol uses 17 frames, six cameras, 424 × 800 resolution, and
RGB, metric depth, and masked optical flow. The Gaussian decoder normally uses
the generated flow; `--zero-flow-input` is only for ablations. Direct-RGB
experiments are not part of the released result.

## Downloads

| Resource | Contents | Download / details |
| --- | --- | --- |
| Checkpoints | Generator EMA and Gaussian decoder weights | [Google Drive](https://drive.google.com/drive/folders/1qGS1ix6Krm3LLeDMmzKqd4sgKRCewm-c) · [files, hashes, and metrics](docs/checkpoints.md) |
| Dataset | 700-scene flow/mask release; depth-shard completeness is unverified | [Google Drive](https://drive.google.com/drive/folders/1XB83zwYeLrJ6WnIZ51adw_7Q9esjggSJ) · [contents and status](docs/dataset.md) |
| Mini-sample | One-clip, input-only sample (78 MB) | [GitHub Release](https://github.com/zttt04/Drive2Gauss_/releases/tag/data-sample-v1) · nuScenes terms apply |

Google sign-in may be required to access the Drive folders. The GitHub release
contains one 17-frame, six-camera clip with RGB, depth, flow, masks, minimal
annotations, map cache, and a relative-path manifest. It does not include
generated latents or runtime feature caches.

## Environment setup

The Gaussian decoder uses Python 3.10, PyTorch 2.4.1, CUDA 12.1, and gsplat
1.5.3. The generator requires a separate MagicDrive/DiST-T-compatible
environment.

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
lock file. Install compiled dependencies according to the target server.

## Quick start

Copy the path template, edit it for your machine, then export its values:

```bash
cp configs/paths.env.example configs/paths.env
set -a
source configs/paths.env
set +a
```

Set the local nuScenes, pretrained-model, checkpoint, dataset, annotation, and
output paths in `configs/paths.env`. Set `DRIVE2GAUSS_DATASET_ROOT` to the
downloaded data release; its JSONL manifests use paths relative to this root.
Legacy absolute-path manifests and pre-rendered flow-index datasets are also
supported. The launchers set `PYTHONPATH` and forward extra arguments to their
Python entry points.

### Run the mini-sample

Download and extract the [one-clip sample](https://github.com/zttt04/Drive2Gauss_/releases/tag/data-sample-v1),
then point the data roots at the extracted directory:

```bash
export DRIVE2GAUSS_DATASET_ROOT=/path/to/drive2gauss_sample
export NUSCENES_ROOT="${DRIVE2GAUSS_DATASET_ROOT}/nuscenes"
export DRIVE2GAUSS_LATENT_MANIFESTS=
```

With the released generator checkpoint and pretrained CogVideoX/T5 weights
configured in `configs/paths.env`, generate and decode one 17-frame sample:

```bash
GPUS=1 bash scripts/infer_generator.sh --cfg-options 'validation_index=[0]' num_frames=17
python tools/decode_generated_rgbd_flow_latents.py \
  --latent-root "${DRIVE2GAUSS_INFERENCE_ROOT}" \
  --output-dir "${DRIVE2GAUSS_DATASET_ROOT}/runtime/decoded" \
  --vae-pretrained "${DRIVE2GAUSS_PRETRAINED_ROOT}/CogVideoX-2b" \
  --max-latents 1 --fast-dev-run
```

This is an inference smoke test. The archive does not bundle generated
latents or Gaussian feature caches; evaluating the Gaussian decoder requires
those outputs and the separately released Gaussian checkpoint. The sample
reader supports relative paths and uint16 depth/flow files. See
[`docs/dataset.md`](docs/dataset.md) for its contents and nuScenes terms.

## Training and inference

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

The decoder uses generated flow by default; `--zero-flow-input` is only for
ablations. Preprocessing and latent-manifest utilities are documented in
[`tools/README.md`](tools/README.md).

## Data and checkpoint layout

For the full pipeline, configure these files and directories in
`configs/paths.env`:

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

The portable preprocessing release is organized as:

```text
data/
  manifest.jsonl
  depth/<scene>/<camera>/depth_<frame>.png
  flow_source/<scene>/<camera>/flow/flow_arrays/flow_<frame>.png
  masks/<scene>/<camera>/sam/masks/dynamic_object_mask_<frame>.png
```

The Gaussian decoder architecture is separate from its training loop:

```text
src/drive2gauss/models/gaussian_modules.py
    StaticPointForwardModel, FeatureRenderUNet, and Gaussian rendering primitives
src/drive2gauss/models/gaussian_decoder.py
    Drive2GaussGaussianDecoder
src/drive2gauss/models/generator/
    RGB-D-flow video generator
```

The remaining packages contain data loading and query sampling, training,
inference, time-dependent gsplat rendering, and evaluation. See
[`docs/checkpoints.md`](docs/checkpoints.md) for checkpoint hashes and
reference metrics, and [`docs/dataset.md`](docs/dataset.md) for the data
inventory and release status.

## License and terms

The repository's BSD-3-Clause license covers original code only. Model
weights, external software, nuScenes data, and derived artifacts remain
subject to their respective terms. nuScenes data and the mini-sample are
subject to the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).
Read [`docs/third_party_notices.md`](docs/third_party_notices.md) before
redistributing code, weights, or data. Do not commit checkpoints, generated
data, logs, or videos.
