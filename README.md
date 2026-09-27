# Drive2Gauss

Drive2Gauss generates six-view RGB-D-flow video latents and reconstructs them
as a feed-forward dynamic Gaussian scene:

```text
nuScenes context -> DiST-T RGB-D-flow latent -> decoded RGB-D-flow
                 -> feature-UNet Gaussian decoder -> rendered views
```

The official Gaussian decoder consumes generated flow normally. Do not pass
`--zero-flow-input`; that flag exists only for ablations. Direct-RGB
experiments are not part of the released result.

See [CHECKPOINTS.md](CHECKPOINTS.md) for checkpoint identity and metrics and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing code,
weights, or data.

## Environment

Validated: Linux, Python 3.10, PyTorch 2.4.1+cu121, torchvision 0.19.1, and
gsplat 1.5.3 on NVIDIA H20 GPUs. Full 960k-query decoder training peaks near
81 GB.

```bash
conda create -n drive2gauss python=3.10 -y
conda activate drive2gauss
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install gsplat==1.5.3
python -m compileall -q tools third_party/distt/DISTT
pytest -q tests
```

DiST-T additionally needs its MagicDriveDiT-compatible ColossalAI fork and a
FlashAttention build matching the local CUDA/PyTorch ABI. CogVideoX, T5,
Turbo-VAED, SEA-RAFT, Grounded-SAM 2, and nuScenes are separately licensed.

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

The generator config reads `NUSCENES_ROOT`, `DRIVE2GAUSS_PRETRAINED_ROOT`,
`DRIVE2GAUSS_LATENT_MANIFESTS`, and `DRIVE2GAUSS_OUTPUT_ROOT`.

## Generate RGB-D-flow latents

Run from `third_party/distt`:

```bash
export NUSCENES_ROOT=/path/to/nuscenes
export DRIVE2GAUSS_PRETRAINED_ROOT=/path/to/pretrained

torchrun --nproc_per_node=8 scripts/infer_dist_dataset_full_onlyRGB.py \
  configs/magicdrive/train/train_9-17x424x800_rgbd_flow_bbox_instance80_vehicle05_dilate12_train700.py \
  --ckpt-path /path/to/checkpoints/drive2gauss_distt_step3600

cd ../..
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
torchrun --nproc_per_node=8 tools/render_generated_latent_flowtrack_dataset.py \
  --train-manifest /path/to/manifest_train.jsonl \
  --train-cache-root /path/to/generated_latent_rgb_4f3v_train \
  --val-manifest /path/to/manifest_val.jsonl \
  --val-cache-root /path/to/generated_latent_rgbd_flow_full_val \
  --checkpoint /path/to/checkpoints/drive2gauss_feature_unet_step3744.pt \
  --output-dir outputs/feature_unet_step3744_eval \
  --torch-extensions-root /tmp/drive2gauss_gsplat_extensions
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
torchrun --nproc_per_node=8 tools/train_static_pointforward_flowtrack_multiscene.py \
  --train-manifest /path/to/manifest_train.jsonl \
  --train-cache-root /path/to/generated_latent_rgb_4f3v_train \
  --val-manifest /path/to/manifest_val.jsonl \
  --val-cache-root /path/to/generated_latent_rgbd_flow_full_val \
  --init-checkpoint /path/to/checkpoints/drive2gauss_feature_unet_step3744.pt \
  --appearance-mode feature_unet --unfreeze-backbone \
  --num-queries-per-frame-view 80000 --max-total-queries 960000 \
  --lr 1e-4 --backbone-lr 1e-4 \
  --output-dir outputs/feature_unet_generated_latent_finetune \
  --checkpoint-dir checkpoints/feature_unet_generated_latent_finetune
```

`--resume-checkpoint` is only for an exact continuation with identical
architecture, trainable modules, and optimizer groups. It is not
interchangeable with `--init-checkpoint`.

## Encodings

Flow PNGs are uint16 with `U=round(flow_x*64+32768)`,
`V=round(flow_y*64+32768)`, and validity in channel three. Metric depth PNGs
use `round(depth_m*256)`, zero for invalid pixels, and 100 m for sky.
