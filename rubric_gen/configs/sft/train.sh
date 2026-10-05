#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO}"
if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-verl}"
fi
GEN=${GEN:-1}; TAG=${TAG:-sft}
RUN_ID=${RUN_ID:-sftv${GEN}_$(date +%Y%m%d_%H%M%S)_${TAG}}
RUN_DIR=${REPO}/output/sft/${RUN_ID}
mkdir -p logs/triton_cache "${RUN_DIR}/tensorboard" "${RUN_DIR}/checkpoints"
LOG_FILE="${RUN_DIR}/train.log"
export TRITON_CACHE_DIR=${REPO}/logs/triton_cache
export TENSORBOARD_LOGGING_DIR=${RUN_DIR}/tensorboard
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
# Required: set MODEL_PATH to a Hugging Face model ID or a local base-model/checkpoint directory.

torchrun --nproc_per_node=2 scripts/train_sft.py \
  --model "${MODEL_PATH:?Set MODEL_PATH to the base model or checkpoint}" \
  --train-file ${REPO}/data/sft/train.jsonl \
  --val-file ${REPO}/data/sft/val.jsonl \
  --output-dir ${RUN_DIR}/checkpoints \
  --deepspeed ${REPO}/configs/ds_zero3.json \
  --max-seq-len 5120 \
  --lr 5e-6 \
  --epochs 10 \
  --per-device-bs 2 \
  --grad-accum 32 \
  --warmup-ratio 0.03 \
  --logging-steps 1 \
  --tensorboard-dir ${RUN_DIR}/tensorboard \
  --seed 42 \
  2>&1 | tee "${LOG_FILE}"
