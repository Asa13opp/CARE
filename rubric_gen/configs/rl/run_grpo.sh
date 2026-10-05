#!/usr/bin/env bash

set -xeuo pipefail

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-verl}"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${SCRIPT_DIR}/../.." && pwd)
export PYTHONPATH=${REPO}:${PYTHONPATH:-}  # so rl/reward_fn.py can import rl.*

# Required: create judge.env locally with the judge-service configuration and credentials.
# Do not echo secrets while loading local credential files.
{ set +x; } 2>/dev/null
source ${REPO}/configs/rl/judge.env
[[ -f ${REPO}/configs/rl/wandb.env ]] && source ${REPO}/configs/rl/wandb.env   # WANDB_API_KEY (gitignored), optional
set -x
source ${REPO}/configs/rl/env_setup.sh  # local caches + CUDA + vLLM stability

LOGGER='["console","tensorboard"]'
[[ -n "${WANDB_API_KEY:-}" ]] && LOGGER='["console","tensorboard","wandb"]'

# Required Stage-1 initializer; set MODEL_PATH to the checkpoint for this run.
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Stage-1 checkpoint}

NGPUS=${NGPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-8}  # GRPO group size

PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPT_OFFLOAD=${OPT_OFFLOAD:-True}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

EXEC_COEF=${EXEC_COEF:-1.0}
EXEC_TAU=${EXEC_TAU:-0.7}
LAMBDA_TRANS=${LAMBDA_TRANS:-0.1}
GAMMA_FORMAT=${GAMMA_FORMAT:-0.3}
BETA_CONSISTENCY=${BETA_CONSISTENCY:-0.0}
CHOSEN_WIN_BONUS=${CHOSEN_WIN_BONUS:-0.5}
PARSE_FAILURE_PENALTY=${PARSE_FAILURE_PENALTY:-0.5}
LENGTH_PENALTY_COEF=${LENGTH_PENALTY_COEF:-0.0}
LENGTH_TARGET=${LENGTH_TARGET:-6500}
LENGTH_MIN=${LENGTH_MIN:-4500}
LENGTH_MAX=${LENGTH_MAX:-9500}
LENGTH_PENALTY_MAX=${LENGTH_PENALTY_MAX:-1.0}

JUDGE_FAILURE_WINDOW=${JUDGE_FAILURE_WINDOW:-32}
JUDGE_FAILURE_RATE=${JUDGE_FAILURE_RATE:-1.0}

if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_BS=8
  MINI_BS=8
  ROLLOUT_N=4
  MAXSTEPS=2
  TEST_FREQ=0
  SAVE_FREQ=0

  RUN_ID=smoke
  RUN_DIR=${REPO}/output/_smoke
  CKPT_DIR=${RUN_DIR}
  TB_DIR=${RUN_DIR}/tensorboard
  ROLLOUT_DATA_DIR=null
  LOG_FILE=${RUN_DIR}/train.log
  mkdir -p "${TB_DIR}"
else
  TRAIN_BS=${TRAIN_BS:-32}
  MINI_BS=${MINI_BS:-16}
  MAXSTEPS=${MAXSTEPS:-200}

  [[ "${MAXSTEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: MAXSTEPS must be a positive integer; got ${MAXSTEPS@Q}" >&2
    exit 2
  }

  TEST_FREQ=${TEST_FREQ:-10}
  SAVE_FREQ=${SAVE_FREQ:-10}

  GEN=${GEN:-6}
  TAG=${TAG:-}
  RUN_ID=${RUN_ID:-grpov${GEN}_$(date +%Y%m%d_%H%M%S)${TAG:+_${TAG}}}

  RUN_DIR=${REPO}/output/rl/${RUN_ID}
  CKPT_DIR=${RUN_DIR}/checkpoints
  ROLLOUT_DATA_DIR=${RUN_DIR}/rollout_data
  TB_DIR=${RUN_DIR}/tensorboard
  LOG_FILE=${RUN_DIR}/train.log
  mkdir -p "${CKPT_DIR}" "${ROLLOUT_DATA_DIR}" "${TB_DIR}"
fi

export TENSORBOARD_DIR=${TB_DIR}

if [[ "${SMOKE:-0}" != "1" ]]; then
  cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "run_id": "${RUN_ID}", "algo": "grpo", "gen": ${GEN}, "tag": "${TAG}",
  "date": "$(date -Iseconds)",
  "base_model": "${MODEL_PATH}",
  "steps": ${MAXSTEPS}, "save_freq": ${SAVE_FREQ}, "test_freq": ${TEST_FREQ},
  "train_batch_size": ${TRAIN_BS}, "mini_batch_size": ${MINI_BS}, "rollout_n": ${ROLLOUT_N},
  "param_offload": "${PARAM_OFFLOAD}", "opt_offload": "${OPT_OFFLOAD}", "gpu_mem_util": ${GPU_MEM_UTIL},
  "checkpoint_save_contents": ["hf_model"],
  "reward_kwargs": {"exec_coef": ${EXEC_COEF}, "exec_tau": ${EXEC_TAU}, "lambda_trans": ${LAMBDA_TRANS}, "gamma_format": ${GAMMA_FORMAT}, "beta_consistency": ${BETA_CONSISTENCY}, "chosen_win_bonus": ${CHOSEN_WIN_BONUS}, "parse_failure_penalty": ${PARSE_FAILURE_PENALTY}, "length_penalty_coef": ${LENGTH_PENALTY_COEF}, "length_target": ${LENGTH_TARGET}, "length_min": ${LENGTH_MIN}, "length_max": ${LENGTH_MAX}, "length_penalty_max": ${LENGTH_PENALTY_MAX}, "judge_failure_window": ${JUDGE_FAILURE_WINDOW}, "judge_failure_rate": ${JUDGE_FAILURE_RATE}}
}
EOF
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${REPO}/data/rl/train.parquet \
  data.val_files=${REPO}/data/rl/val.parquet \
  data.train_batch_size=${TRAIN_BS} \
  data.max_prompt_length=5120 \
  data.max_response_length=5120 \
  data.truncation=left \
  data.filter_overlong_prompts=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
  actor_rollout_ref.actor.checkpoint.load_contents="['hf_model']" \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path=${REPO}/rl/reward_fn.py \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.exec_coef=${EXEC_COEF} \
  +reward.custom_reward_function.reward_kwargs.exec_tau=${EXEC_TAU} \
  +reward.custom_reward_function.reward_kwargs.lambda_trans=${LAMBDA_TRANS} \
  +reward.custom_reward_function.reward_kwargs.beta_consistency=${BETA_CONSISTENCY} \
  +reward.custom_reward_function.reward_kwargs.gamma_format=${GAMMA_FORMAT} \
  +reward.custom_reward_function.reward_kwargs.chosen_win_bonus=${CHOSEN_WIN_BONUS} \
  +reward.custom_reward_function.reward_kwargs.parse_failure_penalty=${PARSE_FAILURE_PENALTY} \
  +reward.custom_reward_function.reward_kwargs.length_penalty_coef=${LENGTH_PENALTY_COEF} \
  +reward.custom_reward_function.reward_kwargs.length_target=${LENGTH_TARGET} \
  +reward.custom_reward_function.reward_kwargs.length_min=${LENGTH_MIN} \
  +reward.custom_reward_function.reward_kwargs.length_max=${LENGTH_MAX} \
  +reward.custom_reward_function.reward_kwargs.length_penalty_max=${LENGTH_PENALTY_MAX} \
  +reward.custom_reward_function.reward_kwargs.judge_failure_window=${JUDGE_FAILURE_WINDOW} \
  +reward.custom_reward_function.reward_kwargs.judge_failure_rate=${JUDGE_FAILURE_RATE} \
  trainer.logger="${LOGGER}" \
  trainer.project_name=rubric_rl \
  trainer.experiment_name=${RUN_ID} \
  trainer.n_gpus_per_node=${NGPUS} \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.total_training_steps=${MAXSTEPS} \
  trainer.default_local_dir=${CKPT_DIR} \
  trainer.rollout_data_dir=${ROLLOUT_DATA_DIR:-null} \
  hydra.run.dir=${RUN_DIR} \
  "$@" 2>&1 | tee "${LOG_FILE}"

if [[ "${SMOKE:-0}" != "1" ]]; then
  find "${CKPT_DIR}" -type f \( -name data.pt -o -name latest_checkpointed_iteration.txt \) -delete
fi
