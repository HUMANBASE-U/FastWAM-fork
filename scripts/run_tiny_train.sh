#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_DIR="${RUN_DIR:-runs/demo_tiny_fastwam/tiny_train}"
LOG_PATH="${LOG_PATH:-runs/demo_tiny_fastwam/tiny_train_log.txt}"
GPU_PROFILE_PATH="${GPU_PROFILE_PATH:-runs/demo_tiny_fastwam/gpu_profile.csv}"
MAX_STEPS="${MAX_STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEMORY="${PIN_MEMORY:-false}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
EXTRA_ARGS=("$@")

# RTX 40-series cards can trip Accelerate/NCCL P2P checks even for tiny local
# runs. These defaults keep the demo on the safe single-node path; override them
# explicitly if you know your interconnect supports P2P/IB.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RUN_DIR}"

GPU_PROFILE_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv \
    -l 1 > "${GPU_PROFILE_PATH}" 2>&1 &
  GPU_PROFILE_PID="$!"
  cleanup() {
    if [[ -n "${GPU_PROFILE_PID}" ]] && kill -0 "${GPU_PROFILE_PID}" >/dev/null 2>&1; then
      kill "${GPU_PROFILE_PID}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT
else
  echo "nvidia-smi not found; GPU profile unavailable" > "${GPU_PROFILE_PATH}"
fi

python scripts/train.py \
  task=demo_tiny_fastwam \
  "output_dir=${RUN_DIR}" \
  "max_steps=${MAX_STEPS}" \
  "batch_size=${BATCH_SIZE}" \
  "num_workers=${NUM_WORKERS}" \
  "pin_memory=${PIN_MEMORY}" \
  "persistent_workers=${PERSISTENT_WORKERS}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"
