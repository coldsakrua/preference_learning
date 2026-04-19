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

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/dapo-math-17k.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-instruct}

seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-0}
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-512}
online_steps=${ONLINE_STEPS:-30}
online_pairs_per_step=${ONLINE_PAIRS_PER_STEP:-16}
online_save_every_updates=${ONLINE_SAVE_EVERY_UPDATES:-5}
rollout_n=${ROLLOUT_N:-8}
temperature=${TEMPERATURE:-0.6}
top_p=${TOP_P:-0.95}
max_new_tokens=${MAX_NEW_TOKENS:-2048}
learning_rate=${LEARNING_RATE:-2e-6}
beta=${BETA:-0.1}
max_length=${MAX_LENGTH:-8192}
logprob_micro_batch_size=${LOGPROB_MICRO_BATCH_SIZE:-0}
online_gap_clip_abs=${ONLINE_GAP_CLIP_ABS:-1.0}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
vllm_dtype=${VLLM_DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
# vLLM 单轮：需 >= 题干 token + max_new_tokens；默认 8192 省 KV（超长题可 export ROLLOUT_MAX_MODEL_LEN）
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-8192}
online_vllm_enforce_eager=${ONLINE_VLLM_ENFORCE_EAGER:-true}

# PEFT LoRA (anchor env includes peft). Set USE_LORA=false for full fine-tuning.
use_lora=${USE_LORA:-true}
lora_r=${LORA_R:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.05}
vllm_max_lora_rank=${VLLM_MAX_LORA_RANK:-64}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/dapo_pref_4b_1gpu/${run_name}}
train_out="${run_root}/train"

mkdir -p "${run_root}" "${train_out}"

echo "[DAPO-PREF] run_root=${run_root}"
echo "[DAPO-PREF] use_lora=${use_lora} lora_r=${lora_r} lora_alpha=${lora_alpha}"
echo "[DAPO-PREF] online mode: vLLM rollout + HF preference update"
python train_dapo_preference.py \
  --seed "${seed}" \
  --dataset_path "${dataset_path}" \
  --model_path "${model_path}" \
  --output_dir "${train_out}" \
  --require_gold_rationale_for_all_wrong true \
  --online_rollout_backend vllm \
  --tensor_parallel_size "${tensor_parallel_size}" \
  --vllm_dtype "${vllm_dtype}" \
  --gpu_memory_utilization "${gpu_memory_utilization}" \
  --rollout_max_model_len "${rollout_max_model_len}" \
  --prompt_mode none \
  --max_source_samples "${max_source_samples}" \
  --online_steps "${online_steps}" \
  --online-pairs-per-step "${online_pairs_per_step}" \
  --online_save_every_updates "${online_save_every_updates}" \
  --rollout_n "${rollout_n}" \
  --rollout_batch_size "${rollout_batch_size}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --max_new_tokens "${max_new_tokens}" \
  --learning_rate "${learning_rate}" \
  --beta "${beta}" \
  --max_length "${max_length}" \
  --logprob_micro_batch_size "${logprob_micro_batch_size}" \
  --online_gap_clip_abs "${online_gap_clip_abs}" \
  --use_lora "${use_lora}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --vllm_max_lora_rank "${vllm_max_lora_rank}" \
  --online_vllm_enforce_eager "${online_vllm_enforce_eager}" \
  --enable_thinking false

echo "[DAPO-PREF] done"
echo "train_output=${train_out}"
