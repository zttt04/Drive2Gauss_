#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
distt_root="${repo_root}/third_party/distt"
config="${GENERATOR_TRAIN_CONFIG:-configs/magicdrive/train/train_9-17x424x800_rgbd_flow_bbox_instance80_vehicle05_dilate12_train700.py}"
gpus="${GPUS:-8}"
master_port="${MASTER_PORT:-29500}"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset root}"
: "${DRIVE2GAUSS_PRETRAINED_ROOT:?Set DRIVE2GAUSS_PRETRAINED_ROOT}"
: "${DRIVE2GAUSS_LATENT_MANIFESTS:?Set DRIVE2GAUSS_LATENT_MANIFESTS}"
: "${DRIVE2GAUSS_OUTPUT_ROOT:?Set DRIVE2GAUSS_OUTPUT_ROOT}"

cd "${distt_root}"
exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${gpus}" \
  --master_port "${master_port}" \
  scripts/train_dist_mm_onlyRGBcondition.py \
  "${config}" \
  --cfg-options num_workers=1 prefetch_factor=1 pin_memory=False "$@"
