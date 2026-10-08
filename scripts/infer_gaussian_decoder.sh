#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
entrypoint="${repo_root}/src/drive2gauss/inference/gaussian_decoder.py"

: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST}"
: "${TRAIN_CACHE_ROOT:?Set TRAIN_CACHE_ROOT}"
: "${VAL_MANIFEST:?Set VAL_MANIFEST}"
: "${VAL_CACHE_ROOT:?Set VAL_CACHE_ROOT}"
: "${GAUSSIAN_CHECKPOINT:?Set GAUSSIAN_CHECKPOINT}"
: "${GAUSSIAN_OUTPUT_ROOT:?Set GAUSSIAN_OUTPUT_ROOT}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(python "${entrypoint}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --train-cache-root "${TRAIN_CACHE_ROOT}" \
  --val-manifest "${VAL_MANIFEST}" \
  --val-cache-root "${VAL_CACHE_ROOT}" \
  --checkpoint "${GAUSSIAN_CHECKPOINT}" \
  --output-dir "${GAUSSIAN_OUTPUT_ROOT}" \
  --torch-extensions-root "${TORCH_EXTENSIONS_ROOT:-/tmp/drive2gauss_gsplat_extensions}")
if [[ -n "${GAUSSIAN_DATA_ROOT:-}" ]]; then
  command+=(--data-root "${GAUSSIAN_DATA_ROOT}")
fi
if [[ -n "${GAUSSIAN_TRAIN_ANN_FILE:-}" ]]; then
  command+=(--train-ann-file "${GAUSSIAN_TRAIN_ANN_FILE}")
fi
if [[ -n "${GAUSSIAN_VAL_ANN_FILE:-}" ]]; then
  command+=(--val-ann-file "${GAUSSIAN_VAL_ANN_FILE}")
fi
if [[ -n "${DRIVE2GAUSS_MOTION_ROOT:-}" ]]; then
  command+=(--motion-release-root "${DRIVE2GAUSS_MOTION_ROOT}")
fi
if [[ -n "${DRIVE2GAUSS_MOTION_MANIFEST:-}" ]]; then
  command+=(--motion-release-manifest "${DRIVE2GAUSS_MOTION_MANIFEST}")
fi
if [[ -n "${GAUSSIAN_TRAIN_MOTION_ROOT:-}" ]]; then
  command+=(--train-motion-release-root "${GAUSSIAN_TRAIN_MOTION_ROOT}")
fi
if [[ -n "${GAUSSIAN_VAL_MOTION_ROOT:-}" ]]; then
  command+=(--val-motion-release-root "${GAUSSIAN_VAL_MOTION_ROOT}")
fi
exec "${command[@]}" "$@"
