#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-verl}"
fi

export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
export ONLINE_RUBRIC_ROOT=${ONLINE_RUBRIC_ROOT:-${PROJECT_ROOT}/online_rl}
export ONLINE_VERL_CODE_ROOT=${ONLINE_VERL_CODE_ROOT:-${ONLINE_RUBRIC_ROOT}/vendor/verl_code_qwen3}
export ONLINE_RUBRIC_PROFILE_DATA=${ONLINE_RUBRIC_PROFILE_DATA:-${ONLINE_RUBRIC_ROOT}/data/train_profile.jsonl}
export ONLINE_RUBRIC_TEST_PROFILE_DATA=${ONLINE_RUBRIC_TEST_PROFILE_DATA:-${ONLINE_RUBRIC_ROOT}/data/test_profile.jsonl}
# Required: set POLICY_MODEL to a Hugging Face model ID or a local model directory.
export POLICY_MODEL=${POLICY_MODEL:-""}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${ONLINE_RUBRIC_ROOT}/outputs}
export LOG_DIR=${LOG_DIR:-${ONLINE_RUBRIC_ROOT}/logs}
export PROJECT_NAME=${PROJECT_NAME:-online_rubric_rl_rlver}
export LOCAL_CACHE_BASE=${LOCAL_CACHE_BASE:-"/tmp/online_rubric_rl_rlver_${USER:-user}"}

_TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
export TOTAL_GPUS=${TOTAL_GPUS:-${_TOTAL_GPUS:-4}}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-${TOTAL_GPUS}}
unset _TOTAL_GPUS

export RUN_ID=${RUN_ID:-online_rubric_step100_continue_$(date +%Y%m%d_%H%M%S)}
export RUN_DIR=${RUN_DIR:-${OUTPUT_ROOT}/${RUN_ID}}
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

cd "${PROJECT_ROOT}"

# Create rubric.env, judge.env, and/or wandb.env locally when their services are used.
# Keep credentials out of version control; set -a exports variables sourced from these files.
set -a
[[ -f "${SCRIPT_DIR}/rubric.env" ]] && source "${SCRIPT_DIR}/rubric.env"
[[ -f "${SCRIPT_DIR}/judge.env" ]] && source "${SCRIPT_DIR}/judge.env"
[[ -f "${SCRIPT_DIR}/wandb.env" ]] && source "${SCRIPT_DIR}/wandb.env"
set +a

export PYTHONPATH="${PROJECT_ROOT}/src:${ONLINE_RUBRIC_ROOT}/src:${ONLINE_VERL_CODE_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ -z "${CUDA_HOME:-}" || ! -f "${CUDA_HOME}/include/cuda_runtime.h" ]]; then
  for cuda_root in /usr/local/cuda-12.9 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -f "${cuda_root}/include/cuda_runtime.h" ]]; then
      export CUDA_HOME="${cuda_root}"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"${LOCAL_CACHE_BASE}/vllm_cache"}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"${LOCAL_CACHE_BASE}/flashinfer_workspace"}
mkdir -p "${VLLM_CACHE_ROOT}" "${FLASHINFER_WORKSPACE_BASE}"

export ONLINE_REWARD_FAIL_FAST=${ONLINE_REWARD_FAIL_FAST:-0}
export ONLINE_RUBRIC_PROFILE_DATA
export ONLINE_RUBRIC_SIMULATOR_DIR="${RUN_DIR}/simulator_logs"
export RESPONSE_LENGTH_PENALTY_MAX_TOKENS=${RESPONSE_LENGTH_PENALTY_MAX_TOKENS:-200}
export RESPONSE_LENGTH_PENALTY_WEIGHT=${RESPONSE_LENGTH_PENALTY_WEIGHT:-1.0}
export RESPONSE_LENGTH_PENALTY_MAX_PENALTY=${RESPONSE_LENGTH_PENALTY_MAX_PENALTY:-2.0}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export SAVE_CRITIC=${SAVE_CRITIC:-False}
export SAVE_FSDP_CHECKPOINT=${SAVE_FSDP_CHECKPOINT:-0}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
export ONLINE_RUBRIC_ACTOR_FSDP_CPU_OFFLOAD=${ONLINE_RUBRIC_ACTOR_FSDP_CPU_OFFLOAD:-0}

ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-False}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.000}
ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-low_var_kl}
KL_COEF=${KL_COEF:-0.000}

if [[ "${SMOKE:-0}" == "1" ]]; then
  export MOCK_RUBRIC=${MOCK_RUBRIC:-1}
  export MOCK_JUDGE=${MOCK_JUDGE:-1}
  export SIMULATOR_MOCK=${SIMULATOR_MOCK:-1}
  export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console']"}

  VIRTUAL_DATASET_SIZE=${VIRTUAL_DATASET_SIZE:-${NGPUS_PER_NODE}}
  VAL_VIRTUAL_DATASET_SIZE=${VAL_VIRTUAL_DATASET_SIZE:-${NGPUS_PER_NODE}}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${NGPUS_PER_NODE}}
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

  ROLLOUT_N=${ROLLOUT_N:-1}
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
  PER_TURN_LENGTH=${PER_TURN_LENGTH:-128}
  MAX_TURNS=${MAX_TURNS:-1}
  ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}
  PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
else
  VIRTUAL_DATASET_SIZE=${VIRTUAL_DATASET_SIZE:-1600}
  VAL_VIRTUAL_DATASET_SIZE=${VAL_VIRTUAL_DATASET_SIZE:-128}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}

  ROLLOUT_N=${ROLLOUT_N:-6}
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-15240}
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-5120}
  PER_TURN_LENGTH=${PER_TURN_LENGTH:-1024}
  MAX_TURNS=${MAX_TURNS:-8}
  PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24000}
fi

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-True}

cd "${ONLINE_VERL_CODE_ROOT}"

python3 -m verl.trainer.main_ppo \
  +data.virtual_dataset_size="${VIRTUAL_DATASET_SIZE}" \
  +data.val_virtual_dataset_size="${VAL_VIRTUAL_DATASET_SIZE}" \
  data.prompt_key=prompt \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-5000}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-10000}" \
  data.return_raw_chat=True \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
  actor_rollout_ref.thinking="${IF_THINK:-True}" \
  actor_rollout_ref.model.path="${POLICY_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-2e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-48000}" \
  actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}" \
  +actor_rollout_ref.actor.use_loss_generation_mask=True \
  actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
  +actor_rollout_ref.rollout.environment.name=online_rubric_environment \
  +actor_rollout_ref.rollout.environment.per_turn_length="${PER_TURN_LENGTH:-5000}" \
  +actor_rollout_ref.rollout.environment.max_turns="${MAX_TURNS:-8}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${TOP_P:-1.0}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.disable_log_stats="${ROLLOUT_DISABLE_LOG_STATS:-True}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-48000}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-True}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE:-True}" \
  +actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-True}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-True}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-48000}" \
  critic.optim.lr="${CRITIC_LR:-2e-7}" \
  critic.model.path="${CRITIC_MODEL:-${POLICY_MODEL}}" \
  critic.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}" \
  critic.model.use_remove_padding=True \
  critic.model.fsdp_config.optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD:-True}" \
  critic.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-48000}" \
  critic.forward_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-48000}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${RUN_ID}" \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
  +trainer.val_before_train=False \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE:-4}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.save_rollout="${SAVE_ROLLOUT:-True}" \
  +trainer.save_critic="${SAVE_CRITIC:-False}" \
  trainer.test_freq="${TEST_FREQ:-999999}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  "$@" 2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
