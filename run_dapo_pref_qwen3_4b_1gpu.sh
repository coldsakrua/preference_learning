#!/bin/bash
#SBATCH -o logs/dapo_pref_4b_1gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13

set -eo pipefail
nvidia-smi

project_root=${PROJECT_ROOT:-/gpfs/share/home/2501210611/preference-learning}
cd "${project_root}"

# conda activate/deactivate hooks may reference unset vars; avoid nounset here.
source activate anchor
set -u
mkdir -p logs

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset_path=${DATASET_PATH:-/path/to/dapo-math-17k.parquet}
model_path=${MODEL_PATH:-/path/to/qwen3-4b}
prompt_file=${PROMPT_FILE:-config/prompt_candidates_en.txt}

seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-17000}
target_pairs=${TARGET_PAIRS:-5000}
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-128}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.95}
max_new_tokens=${MAX_NEW_TOKENS:-1024}
vllm_dtype=${VLLM_DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-8192}
enable_thinking=${ENABLE_THINKING:-true}

num_epochs=${NUM_EPOCHS:-1}
train_batch_size=${TRAIN_BATCH_SIZE:-2}
grad_accum=${GRAD_ACCUM:-8}
learning_rate=${LEARNING_RATE:-2e-6}
beta=${BETA:-0.1}
max_length=${MAX_LENGTH:-4096}
length_average=${LENGTH_AVERAGE:-true}
torch_dtype=${TORCH_DTYPE:-bfloat16}
attn_impl=${ATTN_IMPLEMENTATION:-flash_attention_2}
save_every_epoch=${SAVE_EVERY_EPOCH:-true}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/dapo_pref_4b_1gpu/${run_name}}
data_dir="${run_root}/data"
train_out="${run_root}/train"
pairs_file="${data_dir}/dapo_pref_pairs.jsonl"

mkdir -p "${run_root}" "${data_dir}" "${train_out}"

echo "[DAPO-PREF] run_root=${run_root}"
echo "[DAPO-PREF] step 1/2: rollout + preference pair mining"
python train_dapo_preference.py \
  --stage generate \
  --seed "${seed}" \
  --dataset_path "${dataset_path}" \
  --rollout_model_path "${model_path}" \
  --preference_pairs_path "${pairs_file}" \
  --max_source_samples "${max_source_samples}" \
  --target_pairs "${target_pairs}" \
  --rollout_batch_size "${rollout_batch_size}" \
  --tensor_parallel_size "${tensor_parallel_size}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --max_new_tokens "${max_new_tokens}" \
  --vllm_dtype "${vllm_dtype}" \
  --gpu_memory_utilization "${gpu_memory_utilization}" \
  --rollout_max_model_len "${rollout_max_model_len}" \
  --enable_thinking "${enable_thinking}" \
  --prompt_mode random \
  --prompt_candidates_file "${prompt_file}"

if [[ ! -s "${pairs_file}" ]]; then
  echo "[DAPO-PREF] ERROR: preference pair file is empty: ${pairs_file}"
  echo "[DAPO-PREF] Try increasing MAX_SOURCE_SAMPLES or lowering TARGET_PAIRS."
  exit 1
fi

echo "[DAPO-PREF] step 2/2: preference training"
python train_dapo_preference.py \
  --stage train \
  --seed "${seed}" \
  --train_model_path "${model_path}" \
  --preference_pairs_path "${pairs_file}" \
  --output_dir "${train_out}" \
  --num_epochs "${num_epochs}" \
  --train_batch_size "${train_batch_size}" \
  --gradient_accumulation_steps "${grad_accum}" \
  --learning_rate "${learning_rate}" \
  --beta "${beta}" \
  --max_length "${max_length}" \
  --length_average "${length_average}" \
  --torch_dtype "${torch_dtype}" \
  --attn_implementation "${attn_impl}" \
  --save_every_epoch "${save_every_epoch}" \
  --enable_thinking "${enable_thinking}" \
  --prompt_mode none

echo "[DAPO-PREF] done"
echo "pairs_file=${pairs_file}"
echo "train_output=${train_out}"
