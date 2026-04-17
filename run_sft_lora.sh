#!/bin/bash
#SBATCH -o logs/sft_lora.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning

# conda activate/deactivate hooks may reference unset vars; avoid nounset here.
source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs outputs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/dapo-math-17k.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}

seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-0}
scan_batch_size=${SCAN_BATCH_SIZE:-1024}
system_prompt=${SYSTEM_PROMPT:-}
enable_thinking=${ENABLE_THINKING:-true}
answer_prefix=${ANSWER_PREFIX:-Answer: }

max_length=${MAX_LENGTH:-2048}
eval_ratio=${EVAL_RATIO:-0.02}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-8}
learning_rate=${LEARNING_RATE:-2e-4}
weight_decay=${WEIGHT_DECAY:-0.0}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1}
warmup_ratio=${WARMUP_RATIO:-0.03}
logging_steps=${LOGGING_STEPS:-10}
save_steps=${SAVE_STEPS:-200}
eval_steps=${EVAL_STEPS:-200}

torch_dtype=${TORCH_DTYPE:-bfloat16}
attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-true}

lora_r=${LORA_R:-16}
lora_alpha=${LORA_ALPHA:-32}
lora_dropout=${LORA_DROPOUT:-0.05}
lora_target_modules=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/sft_lora/${run_name}}
train_out="${run_root}/train"
mkdir -p "${train_out}"

echo "[SFT-LoRA] run_root=${run_root}"
echo "[SFT-LoRA] dataset_path=${dataset_path}"
echo "[SFT-LoRA] model_path=${model_path}"
echo "[SFT-LoRA] lora_r=${lora_r} lora_alpha=${lora_alpha} lora_dropout=${lora_dropout}"

python train_sft_lora.py \
  --dataset_path "${dataset_path}" \
  --model_path "${model_path}" \
  --output_dir "${train_out}" \
  --seed "${seed}" \
  --scan_batch_size "${scan_batch_size}" \
  --max_source_samples "${max_source_samples}" \
  --system_prompt "${system_prompt}" \
  --enable_thinking "${enable_thinking}" \
  --answer_prefix "${answer_prefix}" \
  --max_length "${max_length}" \
  --eval_ratio "${eval_ratio}" \
  --per_device_train_batch_size "${per_device_train_batch_size}" \
  --per_device_eval_batch_size "${per_device_eval_batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --learning_rate "${learning_rate}" \
  --weight_decay "${weight_decay}" \
  --num_train_epochs "${num_train_epochs}" \
  --warmup_ratio "${warmup_ratio}" \
  --logging_steps "${logging_steps}" \
  --save_steps "${save_steps}" \
  --eval_steps "${eval_steps}" \
  --torch_dtype "${torch_dtype}" \
  --attn_implementation "${attn_implementation}" \
  --gradient_checkpointing "${gradient_checkpointing}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --lora_target_modules "${lora_target_modules}"

echo "[SFT-LoRA] done"
echo "train_output=${train_out}"
