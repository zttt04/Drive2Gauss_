#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
entrypoint="${repo_root}/src/drive2gauss/training/gaussian_decoder.py"
gpus="${GPUS:-8}"
master_port="${MASTER_PORT:-29502}"

: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST}"
: "${TRAIN_CACHE_ROOT:?Set TRAIN_CACHE_ROOT}"
: "${VAL_MANIFEST:?Set VAL_MANIFEST}"
: "${VAL_CACHE_ROOT:?Set VAL_CACHE_ROOT}"
: "${GAUSSIAN_OUTPUT_ROOT:?Set GAUSSIAN_OUTPUT_ROOT}"
: "${GAUSSIAN_CHECKPOINT_ROOT:?Set GAUSSIAN_CHECKPOINT_ROOT}"
: "${GAUSSIAN_DATA_ROOT:?Set GAUSSIAN_DATA_ROOT to the nuScenes root}"
: "${GAUSSIAN_TRAIN_ANN_FILE:?Set GAUSSIAN_TRAIN_ANN_FILE}"
: "${GAUSSIAN_VAL_ANN_FILE:?Set GAUSSIAN_VAL_ANN_FILE}"
if [[ -z "${DRIVE2GAUSS_MOTION_ROOT:-}" && -z "${GAUSSIAN_TRAIN_MOTION_ROOT:-}" ]]; then
  : "${GAUSSIAN_TRAIN_FLOW_RGB_ROOT:?Set GAUSSIAN_TRAIN_FLOW_RGB_ROOT or DRIVE2GAUSS_MOTION_ROOT}"
  : "${GAUSSIAN_VAL_FLOW_RGB_ROOT:?Set GAUSSIAN_VAL_FLOW_RGB_ROOT or DRIVE2GAUSS_MOTION_ROOT}"
  : "${GAUSSIAN_TRAIN_FLOW_INDEX:?Set GAUSSIAN_TRAIN_FLOW_INDEX or DRIVE2GAUSS_MOTION_ROOT}"
  : "${GAUSSIAN_VAL_FLOW_INDEX:?Set GAUSSIAN_VAL_FLOW_INDEX or DRIVE2GAUSS_MOTION_ROOT}"
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  python -m torch.distributed.run
  --standalone
  --nproc_per_node "${gpus}"
  --master_port "${master_port}"
  "${entrypoint}"
  --train-manifest "${TRAIN_MANIFEST}"
  --train-cache-root "${TRAIN_CACHE_ROOT}"
  --val-manifest "${VAL_MANIFEST}"
  --val-cache-root "${VAL_CACHE_ROOT}"
  --online-query
  --data-root "${GAUSSIAN_DATA_ROOT}"
  --train-ann-file "${GAUSSIAN_TRAIN_ANN_FILE}"
  --val-ann-file "${GAUSSIAN_VAL_ANN_FILE}"
  --appearance-mode feature_unet
  --num-queries-per-frame-view 80000
  --max-total-queries 960000
  --output-dir "${GAUSSIAN_OUTPUT_ROOT}"
  --checkpoint-dir "${GAUSSIAN_CHECKPOINT_ROOT}"
)
if [[ -n "${GAUSSIAN_TRAIN_MOTION_ROOT:-}" || -n "${GAUSSIAN_VAL_MOTION_ROOT:-}" ]]; then
  : "${GAUSSIAN_TRAIN_MOTION_ROOT:?Set GAUSSIAN_TRAIN_MOTION_ROOT}"
  : "${GAUSSIAN_VAL_MOTION_ROOT:?Set GAUSSIAN_VAL_MOTION_ROOT}"
  command+=(
    --train-motion-release-root "${GAUSSIAN_TRAIN_MOTION_ROOT}"
    --val-motion-release-root "${GAUSSIAN_VAL_MOTION_ROOT}"
  )
elif [[ -n "${DRIVE2GAUSS_MOTION_ROOT:-}" ]]; then
  command+=(--motion-release-root "${DRIVE2GAUSS_MOTION_ROOT}")
  if [[ -n "${DRIVE2GAUSS_MOTION_MANIFEST:-}" ]]; then
    command+=(--motion-release-manifest "${DRIVE2GAUSS_MOTION_MANIFEST}")
  fi
else
  command+=(
    --train-masked-flow-rgb-root "${GAUSSIAN_TRAIN_FLOW_RGB_ROOT}"
    --val-masked-flow-rgb-root "${GAUSSIAN_VAL_FLOW_RGB_ROOT}"
    --train-masked-flow-index "${GAUSSIAN_TRAIN_FLOW_INDEX}"
    --val-masked-flow-index "${GAUSSIAN_VAL_FLOW_INDEX}"
  )
fi
if [[ "${GAUSSIAN_ALLOW_MISSING_FLOW_RGB:-0}" == "1" ]]; then
  command+=(--allow-missing-flow-rgb)
fi
if [[ -n "${GAUSSIAN_INIT_CHECKPOINT:-}" ]]; then
  command+=(--init-checkpoint "${GAUSSIAN_INIT_CHECKPOINT}")
fi
exec "${command[@]}" "$@"
