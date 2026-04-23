#!/bin/bash
#SBATCH -o logs/sft_lora_llama3_2_1b.%j.out
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

# conda activate/deactivate hooks may reference unset vars; avoid nounset here.
source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs outputs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/hendrycks_math/aggregated_l3plus/train.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/llama3.2-1b}
seed=${SEED:-42}
system_prompt=${SYSTEM_PROMPT:-}
enable_thinking=${ENABLE_THINKING:-false}
answer_prefix=${ANSWER_PREFIX:-Answer: }

rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-4096}
max_length=${MAX_LENGTH:-${rollout_max_model_len}}
eval_ratio=${EVAL_RATIO:-0}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-32}
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"
max_steps="${MAX_STEPS:-640}"
learning_rate=${LEARNING_RATE:-2e-6}
weight_decay=${WEIGHT_DECAY:-0.0}
warmup_ratio=${WARMUP_RATIO:-0.03}
logging_steps=${LOGGING_STEPS:-1}
save_steps=${SAVE_STEPS:-80}
eval_steps=${EVAL_STEPS:-200}

torch_dtype=${TORCH_DTYPE:-bfloat16}
attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-true}

lora_r=${LORA_R:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.05}
lora_target_modules=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/sft_lora_llama3_2_1b/${run_name}}
train_out="${run_root}/train"
mkdir -p "${train_out}"

effective_batch=$(( per_device_train_batch_size * gradient_accumulation_steps ))

echo "[SFT-LLAMA] run_root=${run_root}"
echo "[SFT-LLAMA] dataset_path=${dataset_path}"
echo "[SFT-LLAMA] model_path=${model_path}"
echo "[SFT-LLAMA] num_train_epochs=${num_train_epochs} max_steps=${max_steps}"
echo "[SFT-LLAMA] per_device_train_batch_size=${per_device_train_batch_size} gradient_accumulation_steps=${gradient_accumulation_steps} effective_batch_per_gpu=${effective_batch} eval_ratio=${eval_ratio}"
echo "[SFT-LLAMA] max_length=${max_length} learning_rate=${learning_rate}"
echo "[SFT-LLAMA] lora_r=${lora_r} lora_alpha=${lora_alpha} lora_dropout=${lora_dropout}"

python run_sft_lora.py \
  --dataset_path "${dataset_path}" \
  --model_path "${model_path}" \
  --output_dir "${train_out}" \
  --seed "${seed}" \
  --system_prompt "${system_prompt}" \
  --enable_thinking "${enable_thinking}" \
  --answer_prefix "${answer_prefix}" \
  --max_length "${max_length}" \
  --eval_ratio "${eval_ratio}" \
  --per_device_train_batch_size "${per_device_train_batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --num_train_epochs "${num_train_epochs}" \
  --max_steps "${max_steps}" \
  --learning_rate "${learning_rate}" \
  --weight_decay "${weight_decay}" \
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

echo "[SFT-LLAMA] done"
echo "train_output=${train_out}"
