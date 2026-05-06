#!/bin/bash
#SBATCH -o logs/privileged_hidden_opsd_4b_1gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13,gpua800n04

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning/privileged_hidden_opsd

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
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}

seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-0}
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-32}
online_steps=${ONLINE_STEPS:-80}
online_objectives_per_step=${ONLINE_OBJECTIVES_PER_STEP:-8}
online_save_every_updates=${ONLINE_SAVE_EVERY_UPDATES:-8}
rollout_n=${ROLLOUT_N:-8}
temperature=${TEMPERATURE:-1.1}
top_p=${TOP_P:-0.8}
top_k=${TOP_K:-20}
min_p=${MIN_P:-0.0}
presence_penalty=${PRESENCE_PENALTY:-0.0}
max_new_tokens=${MAX_NEW_TOKENS:-2048}
learning_rate=${LEARNING_RATE:-1e-6}
lambda_mle=${LAMBDA_MLE:-1.0}
lambda_priv=${LAMBDA_PRIV:-5.0}
lambda_gt=${LAMBDA_GT:-5.0}
privileged_jsd_beta=${PRIVILEGED_JSD_BETA:--1.0}
privileged_distill_temperature=${PRIVILEGED_DISTILL_TEMPERATURE:-1.0}
privileged_pointwise_kl_clip=${PRIVILEGED_POINTWISE_KL_CLIP:-0.05}
privileged_logit_clip_abs=${PRIVILEGED_LOGIT_CLIP_ABS:-80.0}
privileged_trace_max_chars=${PRIVILEGED_TRACE_MAX_CHARS:-0}
hidden_layer_offset=${HIDDEN_LAYER_OFFSET:-4}
rollout_feature_micro_batch_size=${ROLLOUT_FEATURE_MICRO_BATCH_SIZE:-4}
logprob_micro_batch_size=${LOGPROB_MICRO_BATCH_SIZE:-2}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
vllm_dtype=${VLLM_DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.60}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-4096}
max_length=${MAX_LENGTH:-${rollout_max_model_len}}
online_vllm_enforce_eager=${ONLINE_VLLM_ENFORCE_EAGER:-true}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}

use_lora=${USE_LORA:-true}
lora_r=${LORA_R:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.05}
vllm_max_lora_rank=${VLLM_MAX_LORA_RANK:-64}
log_rollout_text=${LOG_ROLLOUT_TEXT:-false}
use_all_wrong_gt_preference=${USE_ALL_WRONG_GT_PREFERENCE:-true}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/privileged_hidden_opsd_4b_1gpu/${run_name}}
train_out="${run_root}/train"

world_size=${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')}
if [[ -z "${world_size}" || "${world_size}" -lt 1 ]]; then
  world_size=1
fi
if [[ "${world_size}" -ne 1 ]]; then
  echo "[PRIV-HIDDEN-OPSD][warn] expected 1 GPU for this script, got world_size=${world_size}"
fi
if (( rollout_batch_size % world_size != 0 )); then
  echo "[PRIV-HIDDEN-OPSD][error] rollout_batch_size=${rollout_batch_size} must be divisible by world_size=${world_size}."
  exit 1
fi
if (( logprob_micro_batch_size > 0 )) && (( logprob_micro_batch_size % world_size != 0 )); then
  echo "[PRIV-HIDDEN-OPSD][error] logprob_micro_batch_size=${logprob_micro_batch_size} must be divisible by world_size=${world_size}."
  exit 1
fi

mkdir -p "${run_root}" "${train_out}"

echo "[PRIV-HIDDEN-OPSD] run_root=${run_root}"
echo "[PRIV-HIDDEN-OPSD] model_path=${model_path}"
echo "[PRIV-HIDDEN-OPSD] rollout_batch_size=${rollout_batch_size} rollout_n=${rollout_n}"
echo "[PRIV-HIDDEN-OPSD] lambda_mle=${lambda_mle} lambda_priv=${lambda_priv} lambda_gt=${lambda_gt}"
echo "[PRIV-HIDDEN-OPSD] hidden_layer_offset=${hidden_layer_offset} jsd_beta=${privileged_jsd_beta}"
echo "[PRIV-HIDDEN-OPSD] use_all_wrong_gt_preference=${use_all_wrong_gt_preference} (false => skip all-wrong samples)"
echo "[PRIV-HIDDEN-OPSD] attn_implementation=${attn_implementation}"

python train_privileged_hidden_opsd.py \
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
  --lambda_priv "${lambda_priv}" \
  --lambda_gt "${lambda_gt}" \
  --privileged_jsd_beta "${privileged_jsd_beta}" \
  --privileged_distill_temperature "${privileged_distill_temperature}" \
  --privileged_pointwise_kl_clip "${privileged_pointwise_kl_clip}" \
  --privileged_logit_clip_abs "${privileged_logit_clip_abs}" \
  --privileged_trace_max_chars "${privileged_trace_max_chars}" \
  --hidden_layer_offset "${hidden_layer_offset}" \
  --rollout_feature_micro_batch_size "${rollout_feature_micro_batch_size}" \
  --max_length "${max_length}" \
  --logprob_micro_batch_size "${logprob_micro_batch_size}" \
  --use_lora "${use_lora}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --vllm_max_lora_rank "${vllm_max_lora_rank}" \
  --online_vllm_enforce_eager "${online_vllm_enforce_eager}" \
  --attn_implementation "${attn_implementation}" \
  --gradient_checkpointing true \
  --enable_thinking false \
  --use_all_wrong_gt_preference "${use_all_wrong_gt_preference}" \
  --log_rollout_text "${log_rollout_text}"

echo "[PRIV-HIDDEN-OPSD] done"
echo "train_output=${train_out}"
