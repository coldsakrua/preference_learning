#!/bin/bash
#SBATCH -o logs/onesided_v3_4b_1gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13,gpua800n04

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning

# conda activate/deactivate hooks may reference unset vars; avoid nounset here.
source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/hendrycks_math/aggregated_l3plus/train.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-base}

seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-0}
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-64}
online_steps=${ONLINE_STEPS:-30}
online_objectives_per_step=${ONLINE_OBJECTIVES_PER_STEP:-8}
online_save_every_updates=${ONLINE_SAVE_EVERY_UPDATES:-16}
rollout_n=${ROLLOUT_N:-8}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.8}
top_k=${TOP_K:-20}
min_p=${MIN_P:-0.0}
presence_penalty=${PRESENCE_PENALTY:-0.0}
max_new_tokens=${MAX_NEW_TOKENS:-2048}
learning_rate=${LEARNING_RATE:-1e-6}

# v3: correct responses use token-weighted MLE; wrong final-answer clusters
# enter an answer-level contrastive denominator against the gold answer.
lambda_mle=${LAMBDA_MLE:-0.3}
lambda_answer=${LAMBDA_ANSWER:-1.0}
answer_tau=${ANSWER_TAU:-0.7}
answer_cluster_weight=${ANSWER_CLUSTER_WEIGHT:-sqrt_count}
answer_min_cluster_count=${ANSWER_MIN_CLUSTER_COUNT:-1}
answer_max_clusters=${ANSWER_MAX_CLUSTERS:-0}
answer_use_normalized=${ANSWER_USE_NORMALIZED:-true}

# Prompt rarity weighting: kept at the conservative legacy defaults so the
# pipeline does not assume any external difficulty signal beyond the
# rollout-derived correct rate rho_hat. Set PROMPT_WEIGHT_GAMMA=0 to disable
# rarity reweighting entirely (uniform prompt weights, capped at 1.0).
prompt_smoothing_alpha=${PROMPT_SMOOTHING_ALPHA:-1.0}
prompt_smoothing_beta=${PROMPT_SMOOTHING_BETA:-1.0}
prompt_weight_gamma=${PROMPT_WEIGHT_GAMMA:-1.0}
prompt_weight_min=${PROMPT_WEIGHT_MIN:-0.1}
prompt_weight_max=${PROMPT_WEIGHT_MAX:-1.0}

# Token-weighted MLE on correct responses.
token_weight_type=${TOKEN_WEIGHT_TYPE:-entropy}
token_weight_alpha=${TOKEN_WEIGHT_ALPHA:-1.0}
token_weight_topk_pct=${TOKEN_WEIGHT_TOPK_PCT:-0.2}
max_grad_norm=${MAX_GRAD_NORM:-10.0}
online_hard_grad_norm_cap=${ONLINE_HARD_GRAD_NORM_CAP:-10.0}

tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
vllm_dtype=${VLLM_DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-4096}
max_length=${MAX_LENGTH:-${rollout_max_model_len}}
online_vllm_enforce_eager=${ONLINE_VLLM_ENFORCE_EAGER:-true}

use_lora=${USE_LORA:-true}
lora_r=${LORA_R:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.0}
vllm_max_lora_rank=${VLLM_MAX_LORA_RANK:-64}
log_rollout_text=${LOG_ROLLOUT_TEXT:-false}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/onesided_v3_4b_1gpu/${run_name}}
train_out="${run_root}/train"

world_size=${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')}
if [[ -z "${world_size}" || "${world_size}" -lt 1 ]]; then
  world_size=1
fi
if [[ "${world_size}" -ne 1 ]]; then
  echo "[ONESIDED-V3][warn] expected 1 GPU for this script, got world_size=${world_size}"
fi
if (( rollout_batch_size % world_size != 0 )); then
  echo "[ONESIDED-V3][error] rollout_batch_size=${rollout_batch_size} must be divisible by world_size=${world_size}."
  exit 1
fi

mkdir -p "${run_root}" "${train_out}"

echo "[ONESIDED-V3] run_root=${run_root}"
echo "[ONESIDED-V3] model_path=${model_path}"
echo "[ONESIDED-V3] rollout_batch_size=${rollout_batch_size} rollout_n=${rollout_n}"
echo "[ONESIDED-V3] lambda_mle=${lambda_mle} lambda_answer=${lambda_answer} answer_tau=${answer_tau} answer_cluster_weight=${answer_cluster_weight}"
echo "[ONESIDED-V3] token_weight_type=${token_weight_type} alpha=${token_weight_alpha} topk_pct=${token_weight_topk_pct}"
echo "[ONESIDED-V3] prompt_weight_gamma=${prompt_weight_gamma}"
echo "[ONESIDED-V3] answer_min_cluster_count=${answer_min_cluster_count} answer_max_clusters=${answer_max_clusters} answer_use_normalized=${answer_use_normalized}"
echo "[ONESIDED-V3] max_grad_norm=${max_grad_norm}"
echo "[ONESIDED-V3] online_hard_grad_norm_cap=${online_hard_grad_norm_cap}"

python train_onesided_v3.py \
  --seed "${seed}" \
  --dataset_path "${dataset_path}" \
  --model_path "${model_path}" \
  --output_dir "${train_out}" \
  --online_rollout_backend vllm \
  --tensor_parallel_size "${tensor_parallel_size}" \
  --vllm_dtype "${vllm_dtype}" \
  --gpu_memory_utilization "${gpu_memory_utilization}" \
  --rollout_max_model_len "${rollout_max_model_len}" \
  --prompt_mode none \
  --max_source_samples "${max_source_samples}" \
  --online_steps "${online_steps}" \
  --online-pairs-per-step "${online_objectives_per_step}" \
  --online_save_every_updates "${online_save_every_updates}" \
  --rollout_n "${rollout_n}" \
  --rollout_batch_size "${rollout_batch_size}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --top_k "${top_k}" \
  --min_p "${min_p}" \
  --presence_penalty "${presence_penalty}" \
  --max_new_tokens "${max_new_tokens}" \
  --learning_rate "${learning_rate}" \
  --lambda_mle "${lambda_mle}" \
  --lambda_answer "${lambda_answer}" \
  --prompt_smoothing_alpha "${prompt_smoothing_alpha}" \
  --prompt_smoothing_beta "${prompt_smoothing_beta}" \
  --prompt_weight_gamma "${prompt_weight_gamma}" \
  --prompt_weight_min "${prompt_weight_min}" \
  --prompt_weight_max "${prompt_weight_max}" \
  --token_weight_type "${token_weight_type}" \
  --token_weight_alpha "${token_weight_alpha}" \
  --token_weight_topk_pct "${token_weight_topk_pct}" \
  --max_grad_norm "${max_grad_norm}" \
  --online_hard_grad_norm_cap "${online_hard_grad_norm_cap}" \
  --answer_tau "${answer_tau}" \
  --answer_cluster_weight "${answer_cluster_weight}" \
  --answer_min_cluster_count "${answer_min_cluster_count}" \
  --answer_max_clusters "${answer_max_clusters}" \
  --answer_use_normalized "${answer_use_normalized}" \
  --max_length "${max_length}" \
  --use_lora "${use_lora}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --vllm_max_lora_rank "${vllm_max_lora_rank}" \
  --online_vllm_enforce_eager "${online_vllm_enforce_eager}" \
  --gradient_checkpointing true \
  --enable_thinking false \
  --log_rollout_text "${log_rollout_text}"

echo "[ONESIDED-V3] done"
echo "train_output=${train_out}"
