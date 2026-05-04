#!/bin/bash
#SBATCH -o logs/onesided_v2_4b_1gpu.%j.out
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
online_steps=${ONLINE_STEPS:-40}
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

# v2 defaults: bumped lambda_group (0.25 -> 1.0), bumped prompt_weight_gamma (1.0 -> 2.0).
lambda_mle=${LAMBDA_MLE:-1.0}
lambda_group=${LAMBDA_GROUP:-1.0}
group_tau=${GROUP_TAU:-0.5}
group_score_norm=${GROUP_SCORE_NORM:-none}
group_score_std_floor=${GROUP_SCORE_STD_FLOOR:-0.05}
group_score_clip_abs=${GROUP_SCORE_CLIP_ABS:-0.0}
one_sided_weight_type=${ONE_SIDED_WEIGHT_TYPE:-logsigmoid}
group_margin=${GROUP_MARGIN:-0.0}
hard_weight_tau=${HARD_WEIGHT_TAU:-0.5}
hard_weight_min=${HARD_WEIGHT_MIN:-0.0}
hard_weight_max=${HARD_WEIGHT_MAX:-2.0}

# Prompt rarity weighting: kept at the conservative legacy defaults so the
# pipeline does not assume any external difficulty signal beyond the
# rollout-derived correct rate ρ̂. Set PROMPT_WEIGHT_GAMMA=0 to disable
# rarity reweighting entirely (uniform prompt weights, capped at 1.0).
prompt_smoothing_alpha=${PROMPT_SMOOTHING_ALPHA:-1.0}
prompt_smoothing_beta=${PROMPT_SMOOTHING_BETA:-1.0}
prompt_weight_gamma=${PROMPT_WEIGHT_GAMMA:-1.0}
prompt_weight_min=${PROMPT_WEIGHT_MIN:-0.1}
prompt_weight_max=${PROMPT_WEIGHT_MAX:-1.0}

# v2 token-weighted MLE on correct.
token_weight_type=${TOKEN_WEIGHT_TYPE:-entropy}
token_weight_alpha=${TOKEN_WEIGHT_ALPHA:-1.0}
token_weight_topk_pct=${TOKEN_WEIGHT_TOPK_PCT:-0.2}

# v2 mode-collapse aware hard negatives.
mode_min_cluster=${MODE_MIN_CLUSTER:-2}

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

run_root=${RUN_ROOT:-outputs/onesided_v2_4b_1gpu/${run_name}}
train_out="${run_root}/train"

world_size=${NPROC_PER_NODE:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')}
if [[ -z "${world_size}" || "${world_size}" -lt 1 ]]; then
  world_size=1
fi
if [[ "${world_size}" -ne 1 ]]; then
  echo "[ONESIDED-V2][warn] expected 1 GPU for this script, got world_size=${world_size}"
fi
if (( rollout_batch_size % world_size != 0 )); then
  echo "[ONESIDED-V2][error] rollout_batch_size=${rollout_batch_size} must be divisible by world_size=${world_size}."
  exit 1
fi

mkdir -p "${run_root}" "${train_out}"

echo "[ONESIDED-V2] run_root=${run_root}"
echo "[ONESIDED-V2] model_path=${model_path}"
echo "[ONESIDED-V2] rollout_batch_size=${rollout_batch_size} rollout_n=${rollout_n}"
echo "[ONESIDED-V2] lambda_mle=${lambda_mle} lambda_group=${lambda_group} group_tau=${group_tau}"
echo "[ONESIDED-V2] token_weight_type=${token_weight_type} alpha=${token_weight_alpha} topk_pct=${token_weight_topk_pct}"
echo "[ONESIDED-V2] mode_min_cluster=${mode_min_cluster}  prompt_weight_gamma=${prompt_weight_gamma}"
echo "[ONESIDED-V2] weight_type=${one_sided_weight_type} hard_weight_tau=${hard_weight_tau} hard_weight_max=${hard_weight_max}"

python train_onesided_v2.py \
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
  --one_sided_weight_type "${one_sided_weight_type}" \
  --group_margin "${group_margin}" \
  --hard_weight_tau "${hard_weight_tau}" \
  --hard_weight_min "${hard_weight_min}" \
  --hard_weight_max "${hard_weight_max}" \
  --prompt_smoothing_alpha "${prompt_smoothing_alpha}" \
  --prompt_smoothing_beta "${prompt_smoothing_beta}" \
  --prompt_weight_gamma "${prompt_weight_gamma}" \
  --prompt_weight_min "${prompt_weight_min}" \
  --prompt_weight_max "${prompt_weight_max}" \
  --token_weight_type "${token_weight_type}" \
  --token_weight_alpha "${token_weight_alpha}" \
  --token_weight_topk_pct "${token_weight_topk_pct}" \
  --mode_min_cluster "${mode_min_cluster}" \
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

echo "[ONESIDED-V2] done"
echo "train_output=${train_out}"
