#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST}"
: "${TRAIN_CACHE_ROOT:?Set TRAIN_CACHE_ROOT}"
: "${VAL_MANIFEST:?Set VAL_MANIFEST}"
: "${VAL_CACHE_ROOT:?Set VAL_CACHE_ROOT}"
: "${GAUSSIAN_CHECKPOINT:?Set GAUSSIAN_CHECKPOINT}"
: "${GAUSSIAN_OUTPUT_ROOT:?Set GAUSSIAN_OUTPUT_ROOT}"

cd "${repo_root}"
exec python tools/render_generated_latent_flowtrack_dataset.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --train-cache-root "${TRAIN_CACHE_ROOT}" \
  --val-manifest "${VAL_MANIFEST}" \
  --val-cache-root "${VAL_CACHE_ROOT}" \
  --checkpoint "${GAUSSIAN_CHECKPOINT}" \
  --output-dir "${GAUSSIAN_OUTPUT_ROOT}" \
  --torch-extensions-root "${TORCH_EXTENSIONS_ROOT:-/tmp/drive2gauss_gsplat_extensions}" \
  "$@"
