#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generator_root="${repo_root}/src/drive2gauss/models/generator"
entrypoint="${repo_root}/src/drive2gauss/training/video_generator.py"
config="${GENERATOR_TRAIN_CONFIG:-configs/magicdrive/train/train_9-17x424x800_rgbd_flow_bbox_instance80_vehicle05_dilate12_train700.py}"
gpus="${GPUS:-8}"
master_port="${MASTER_PORT:-29500}"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset root}"
: "${DRIVE2GAUSS_PRETRAINED_ROOT:?Set DRIVE2GAUSS_PRETRAINED_ROOT}"
: "${DRIVE2GAUSS_OUTPUT_ROOT:?Set DRIVE2GAUSS_OUTPUT_ROOT}"

cd "${generator_root}"
export PYTHONPATH="${repo_root}/src:${generator_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${gpus}" \
  --master_port "${master_port}" \
  "${entrypoint}" \
  "${config}" \
  --cfg-options num_workers=1 prefetch_factor=1 pin_memory=False "$@"
