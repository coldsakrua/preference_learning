#!/bin/bash
#SBATCH -o logs/group_mle_4b_1gpu.%j.out
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
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-32}
online_steps=${ONLINE_STEPS:-80}
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
lambda_mle=${LAMBDA_MLE:-1.0}
lambda_group=${LAMBDA_GROUP:-0.25}
group_tau=${GROUP_TAU:-0.5}
group_score_norm=${GROUP_SCORE_NORM:-none}
group_score_std_floor=${GROUP_SCORE_STD_FLOOR:-0.05}
group_score_clip_abs=${GROUP_SCORE_CLIP_ABS:-0.0}
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

run_root=${RUN_ROOT:-outputs/group_mle_4b_1gpu/${run_name}}
train_out="${run_root}/train"

world_size=${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')}
if [[ -z "${world_size}" || "${world_size}" -lt 1 ]]; then
  world_size=1
fi
if [[ "${world_size}" -ne 1 ]]; then
  echo "[GROUP-MLE][warn] expected 1 GPU for this script, got world_size=${world_size}"
fi
if (( rollout_batch_size % world_size != 0 )); then
  echo "[GROUP-MLE][error] rollout_batch_size=${rollout_batch_size} must be divisible by world_size=${world_size}."
  exit 1
fi

mkdir -p "${run_root}" "${train_out}"

echo "[GROUP-MLE] run_root=${run_root}"
echo "[GROUP-MLE] model_path=${model_path}"
echo "[GROUP-MLE] rollout_batch_size=${rollout_batch_size} rollout_n=${rollout_n}"
echo "[GROUP-MLE] lambda_mle=${lambda_mle} lambda_group=${lambda_group} group_tau=${group_tau}"
python train_group_mle.py \
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
  --lambda_group "${lambda_group}" \
  --group_tau "${group_tau}" \
  --group_score_norm "${group_score_norm}" \
  --group_score_std_floor "${group_score_std_floor}" \
  --group_score_clip_abs "${group_score_clip_abs}" \
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

echo "[GROUP-MLE] done"
echo "train_output=${train_out}"
