#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python experiments/pfd_stackcube/run_pfd_stackcube_pipeline.py \
  --variant pfd \
  --amp \
  "$@"
