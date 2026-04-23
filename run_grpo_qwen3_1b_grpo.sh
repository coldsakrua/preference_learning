#!/bin/bash
#SBATCH -o logs/grpo_qwen3_4b_1gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/hendrycks_math/aggregated_l3plus/train.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}

seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-17000}
learning_rate=${LEARNING_RATE:-1e-6}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
max_steps=${MAX_STEPS:--1}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_completion_length=${MAX_COMPLETION_LENGTH:-1024}
num_generations=${NUM_GENERATIONS:-8}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.95}
beta=${BETA:-0.04}
save_steps=${SAVE_STEPS:-50}
logging_steps=${LOGGING_STEPS:-5}

use_lora=${USE_LORA:-true}
lora_r=${LORA_R:-32}
lora_alpha=${LORA_ALPHA:-64}
lora_dropout=${LORA_DROPOUT:-0.05}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/grpo_qwen3_4b_1gpu/${run_name}}
train_out="${run_root}/train"
mkdir -p "${run_root}" "${train_out}"

echo "[GRPO] run_root=${run_root}"
echo "[GRPO] model_path=${model_path}"
echo "[GRPO] dataset_path=${dataset_path}"

python train_grpo_preference.py \
  --seed "${seed}" \
  --dataset_path "${dataset_path}" \
  --dataset_layout auto \
  --model_path "${model_path}" \
  --output_dir "${train_out}" \
  --max_source_samples "${max_source_samples}" \
  --learning_rate "${learning_rate}" \
  --num_train_epochs "${num_train_epochs}" \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size "${per_device_train_batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --max_prompt_length "${max_prompt_length}" \
  --max_completion_length "${max_completion_length}" \
  --num_generations "${num_generations}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --beta "${beta}" \
  --logging_steps "${logging_steps}" \
  --save_steps "${save_steps}" \
  --save_total_limit 3 \
  --gradient_checkpointing true \
  --bf16 true \
  --fp16 false \
  --report_to none \
  --run_name "grpo_qwen3_4b_${run_name}" \
  --use_lora "${use_lora}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}"

echo "[GRPO] done"
echo "train_output=${train_out}"
