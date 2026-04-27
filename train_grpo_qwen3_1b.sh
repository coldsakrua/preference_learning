#!/bin/bash
#SBATCH -o logs/grpo_qwen3_4b_2gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n01,gpua800n11

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/hendrycks_math/aggregated_l3plus/train.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}

seed=${SEED:-42}
learning_rate=${LEARNING_RATE:-1e-6}
train_steps=${TRAIN_STEPS:-200}
global_prompts_per_step=${GLOBAL_PROMPTS_PER_STEP:-4}
rollouts_per_prompt=${ROLLOUTS_PER_PROMPT:-4}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_completion_length=${MAX_COMPLETION_LENGTH:-4096}
temperature=${TEMPERATURE:-0.6}
top_p=${TOP_P:-0.95}
beta=${BETA:-0.04}
save_steps=${SAVE_STEPS:-20}
logging_steps=${LOGGING_STEPS:-5}

use_lora=${USE_LORA:-1}
if [[ "${use_lora}" == "1" ]]; then
  use_lora_bool=true
elif [[ "${use_lora}" == "0" ]]; then
  use_lora_bool=false
else
  echo "[GRPO] USE_LORA must be 0 or 1, got: ${use_lora}"
  exit 1
fi
lora_r=${LORA_R:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.05}
lora_target_modules=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
lora_path=${LORA_PATH-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/outputs/pref_qwen3_1b_mixed_diff_1gpu/20260425_022124_job1453738/train/final}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/grpo_qwen3_4b_2gpu/${run_name}}
train_out="${run_root}/train"
mkdir -p "${run_root}" "${train_out}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  world_size="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${SLURM_NTASKS_PER_NODE:-}" && "${SLURM_NTASKS_PER_NODE}" -gt 1 ]]; then
  world_size="${SLURM_NTASKS_PER_NODE}"
else
  world_size=2
fi

denom=$(( per_device_train_batch_size * world_size ))
if (( denom <= 0 )); then
  echo "[GRPO] invalid batch config: per_device_train_batch_size=${per_device_train_batch_size}, world_size=${world_size}"
  exit 1
fi
if (( global_prompts_per_step % denom != 0 )); then
  echo "[GRPO] GLOBAL_PROMPTS_PER_STEP(${global_prompts_per_step}) must be divisible by PER_DEVICE_TRAIN_BATCH_SIZE(${per_device_train_batch_size}) * WORLD_SIZE(${world_size})"
  exit 1
fi
if (( rollouts_per_prompt <= 0 )); then
  echo "[GRPO] ROLLOUTS_PER_PROMPT must be > 0, got ${rollouts_per_prompt}"
  exit 1
fi
if (( global_prompts_per_step % rollouts_per_prompt != 0 )); then
  echo "[GRPO] GLOBAL_PROMPTS_PER_STEP(${global_prompts_per_step}) must be divisible by ROLLOUTS_PER_PROMPT(${rollouts_per_prompt})"
  exit 1
fi
gradient_accumulation_steps=$(( global_prompts_per_step / denom ))
if (( gradient_accumulation_steps <= 0 )); then
  echo "[GRPO] computed GRADIENT_ACCUMULATION_STEPS=${gradient_accumulation_steps} is invalid"
  exit 1
fi

echo "[GRPO] run_root=${run_root}"
echo "[GRPO] model_path=${model_path}"
echo "[GRPO] dataset_path=${dataset_path}"
echo "[GRPO] train_steps=${train_steps}, global_prompts_per_step=${global_prompts_per_step}, rollouts_per_prompt=${rollouts_per_prompt}"
echo "[GRPO] world_size=${world_size}, per_device_train_batch_size=${per_device_train_batch_size}, gradient_accumulation_steps=${gradient_accumulation_steps}"
echo "[GRPO] generation_batch_size=${global_prompts_per_step} (must be divisible by rollouts_per_prompt)"
echo "[GRPO] use_lora=${use_lora} (bool=${use_lora_bool})"
if [[ -n "${lora_path}" ]]; then
  echo "[GRPO] lora_path=${lora_path} (will load adapter if use_lora=1)"
else
  echo "[GRPO] lora_path is empty: will not load a pretrained LoRA (use_lora=1 -> train LoRA from scratch on base; use_lora=0 -> full model)"
fi
echo "[GRPO] note: current TRL GRPO setup uses integrated generation+KL in each rank; cannot pin one GPU only for rollout and the other only for KL."

torchrun --nproc_per_node="${world_size}" --master_port="${MASTER_PORT:-29501}" train_grpo_dapo_preference.py \
  --seed "${seed}" \
  --dataset_path "${dataset_path}" \
  --dataset_layout auto \
  --model_path "${model_path}" \
  --output_dir "${train_out}" \
  --learning_rate "${learning_rate}" \
  --max_steps "${train_steps}" \
  --per_device_train_batch_size "${per_device_train_batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --max_prompt_length "${max_prompt_length}" \
  --max_completion_length "${max_completion_length}" \
  --num_generations "${rollouts_per_prompt}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --beta "${beta}" \
  --logging_steps "${logging_steps}" \
  --save_steps "${save_steps}" \
  --save_total_limit 100 \
  --gradient_checkpointing true \
  --bf16 true \
  --fp16 false \
  --report_to none \
  --run_name "grpo_qwen3_4b_${run_name}" \
  --use_lora "${use_lora_bool}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --lora_target_modules "${lora_target_modules}" \
  --lora_path "${lora_path}"

echo "[GRPO] done"
echo "train_output=${train_out}"
