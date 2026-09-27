#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
distt_root="${repo_root}/third_party/distt"
config="${GENERATOR_INFER_CONFIG:-configs/magicdrive/test/infer_drive2gauss.py}"
gpus="${GPUS:-8}"
master_port="${MASTER_PORT:-29501}"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset root}"
: "${DRIVE2GAUSS_PRETRAINED_ROOT:?Set DRIVE2GAUSS_PRETRAINED_ROOT}"
: "${DRIVE2GAUSS_CHECKPOINT:?Set DRIVE2GAUSS_CHECKPOINT to the DiST-T checkpoint}"
: "${DRIVE2GAUSS_INFERENCE_ROOT:?Set DRIVE2GAUSS_INFERENCE_ROOT}"

cd "${distt_root}"
exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${gpus}" \
  --master_port "${master_port}" \
  scripts/infer_dist_dataset_full_onlyRGB.py \
  "${config}" "$@"
