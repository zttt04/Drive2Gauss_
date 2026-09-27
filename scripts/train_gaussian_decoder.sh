#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpus="${GPUS:-8}"
master_port="${MASTER_PORT:-29502}"

: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST}"
: "${TRAIN_CACHE_ROOT:?Set TRAIN_CACHE_ROOT}"
: "${VAL_MANIFEST:?Set VAL_MANIFEST}"
: "${VAL_CACHE_ROOT:?Set VAL_CACHE_ROOT}"
: "${GAUSSIAN_OUTPUT_ROOT:?Set GAUSSIAN_OUTPUT_ROOT}"
: "${GAUSSIAN_CHECKPOINT_ROOT:?Set GAUSSIAN_CHECKPOINT_ROOT}"

cd "${repo_root}"
command=(
  python -m torch.distributed.run
  --standalone
  --nproc_per_node "${gpus}"
  --master_port "${master_port}"
  tools/train_static_pointforward_flowtrack_multiscene.py
  --train-manifest "${TRAIN_MANIFEST}"
  --train-cache-root "${TRAIN_CACHE_ROOT}"
  --val-manifest "${VAL_MANIFEST}"
  --val-cache-root "${VAL_CACHE_ROOT}"
  --appearance-mode feature_unet
  --num-queries-per-frame-view 80000
  --max-total-queries 960000
  --output-dir "${GAUSSIAN_OUTPUT_ROOT}"
  --checkpoint-dir "${GAUSSIAN_CHECKPOINT_ROOT}"
)
if [[ -n "${GAUSSIAN_INIT_CHECKPOINT:-}" ]]; then
  command+=(--init-checkpoint "${GAUSSIAN_INIT_CHECKPOINT}")
fi
exec "${command[@]}" "$@"
