#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python experiments/pickcube_fastwam/run_pickcube_fastwam_pipeline.py \
  --amp \
  "$@"
