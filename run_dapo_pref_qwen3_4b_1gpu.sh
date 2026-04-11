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
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
prompt_file=${PROMPT_FILE:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/config/prompt_candidates_en.txt}

seed=${SEED:-42}
# 从 parquet 最多扫描多少条题目；0 表示不设上限。要跑满 N 个 step 至少需要 N * ROLLOUT_BATCH_SIZE 条有效题。
max_source_samples=${MAX_SOURCE_SAMPLES:-0}
# 每个 step：顺序取 ROLLOUT_BATCH_SIZE 道题，每题当前模型采样 2 次，筛出「一对一错」的偏好对；若有则做一次 optimizer.step。
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-256}
# 跑多少个上述 step 后结束（与某 step 内是否凑到有效对无关，都会计作一步）。
online_steps=${ONLINE_STEPS:-10}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.95}
max_new_tokens=${MAX_NEW_TOKENS:-2048}
learning_rate=${LEARNING_RATE:-1e-6}
beta=${BETA:-0.1}
max_length=${MAX_LENGTH:-4096}

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
echo "[DAPO-PREF] online mode: rollout-train loop"
python train_dapo_preference.py \
  --stage online \
  --seed "${seed}" \
  --dataset_path "${dataset_path}" \
  --online_init_model_path "${model_path}" \
  --rollout_model_path "${model_path}" \
  --train_model_path "${model_path}" \
  --output_dir "${train_out}" \
  --prompt_mode random \
  --prompt_candidates_file "${prompt_file}" \
  --max_source_samples "${max_source_samples}" \
  --online_steps "${online_steps}" \
  --rollout_batch_size "${rollout_batch_size}" \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --max_new_tokens "${max_new_tokens}" \
  --learning_rate "${learning_rate}" \
  --beta "${beta}" \
  --max_length "${max_length}" \
  --enable_thinking true \
  --save_every_epoch false

echo "[DAPO-PREF] done"
echo "train_output=${train_out}"
