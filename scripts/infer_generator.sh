#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generator_root="${repo_root}/src/drive2gauss/models/generator"
entrypoint="${repo_root}/src/drive2gauss/inference/video_generator.py"
config="${GENERATOR_INFER_CONFIG:-configs/magicdrive/test/infer_drive2gauss.py}"
gpus="${GPUS:-8}"
master_port="${MASTER_PORT:-29501}"

: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset root}"
: "${DRIVE2GAUSS_PRETRAINED_ROOT:?Set DRIVE2GAUSS_PRETRAINED_ROOT}"
: "${DRIVE2GAUSS_CHECKPOINT:?Set DRIVE2GAUSS_CHECKPOINT to the DiST-T checkpoint}"
: "${DRIVE2GAUSS_INFERENCE_ROOT:?Set DRIVE2GAUSS_INFERENCE_ROOT}"

cd "${generator_root}"
export PYTHONPATH="${repo_root}/src:${generator_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${gpus}" \
  --master_port "${master_port}" \
  "${entrypoint}" \
  "${config}" "$@"
