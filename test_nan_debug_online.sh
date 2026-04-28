#!/bin/bash
#SBATCH -o logs/pref_debug.%j.out
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

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/hendrycks_math/aggregated_l3plus/train.parquet}
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}

# Keep NaN-debug parameters aligned with 1b preference training defaults.
seed=${SEED:-42}
max_source_samples=${MAX_SOURCE_SAMPLES:-1000000}
rollout_batch_size=${ROLLOUT_BATCH_SIZE:-64}
online_steps=${ONLINE_STEPS:-8}
online_pairs_per_step=${ONLINE_PAIRS_PER_STEP:-16}
online_save_every_updates=${ONLINE_SAVE_EVERY_UPDATES:-8}
rollout_n=${ROLLOUT_N:-8}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.8}
top_k=${TOP_K:-20}
min_p=${MIN_P:-0.0}
presence_penalty=${PRESENCE_PENALTY:-0.0}
max_new_tokens=${MAX_NEW_TOKENS:-3072}
learning_rate=${LEARNING_RATE:-1e-6}
beta=${BETA:-0.3}
logprob_micro_batch_size=${LOGPROB_MICRO_BATCH_SIZE:-8}
online_gap_clip_abs=${ONLINE_GAP_CLIP_ABS:-1.0}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
vllm_dtype=${VLLM_DTYPE:-bfloat16}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.95}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-4096}
max_length=${MAX_LENGTH:-${rollout_max_model_len}}
online_vllm_enforce_eager=${ONLINE_VLLM_ENFORCE_EAGER:-true}
online_hard_grad_norm_cap=${ONLINE_HARD_GRAD_NORM_CAP:-1.5}

use_lora=${USE_LORA:-true}
lora_r=${LORA_R:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.05}
vllm_max_lora_rank=${VLLM_MAX_LORA_RANK:-64}
use_deepspeed=${USE_DEEPSPEED:-true}
deepspeed_config_path=${DEEPSPEED_CONFIG_PATH:-}
deepspeed_zero_stage=${DEEPSPEED_ZERO_STAGE:-2}
deepspeed_offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER:-false}
deepspeed_offload_param=${DEEPSPEED_OFFLOAD_PARAM:-false}
deepspeed_reduce_bucket_size=${DEEPSPEED_REDUCE_BUCKET_SIZE:-50000000}
deepspeed_allgather_bucket_size=${DEEPSPEED_ALLGATHER_BUCKET_SIZE:-50000000}
deepspeed_stage3_param_persistence_threshold=${DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD:-100000}
deepspeed_stage3_prefetch_bucket_size=${DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE:-50000000}

stamp=$(date -u +%Y%m%d_%H%M%S)
root_out=${RUN_ROOT:-outputs/nan_debug/${stamp}}
mkdir -p "${root_out}"

run_case() {
  local case_name="$1"
  local online_pref_loss_only="$2"
  local online_mle_on_correct_only="$3"
  local lambda_mle="$4"
  local lambda_pref="$5"
  local lambda_gt="$6"

  local case_dir="${root_out}/${case_name}"
  local train_out="${case_dir}/train"
  local log_file="logs/nan_debug_${case_name}.${stamp}.out"
  mkdir -p "${case_dir}" "${train_out}"

  echo "[NAN-DEBUG] case=${case_name} -> ${log_file}"
  echo "[NAN-DEBUG] use_deepspeed=${use_deepspeed} zero_stage=${deepspeed_zero_stage} offload_opt=${deepspeed_offload_optimizer}"

  if [[ "${use_deepspeed}" == "true" ]]; then
    launcher=(deepspeed --num_gpus=1)
  else
    launcher=(python)
  fi

  "${launcher[@]}" train_preference.py \
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
    --online-pairs-per-step "${online_pairs_per_step}" \
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
    --beta "${beta}" \
    --max_length "${max_length}" \
    --logprob_micro_batch_size "${logprob_micro_batch_size}" \
    --online_gap_clip_abs "${online_gap_clip_abs}" \
    --online_hard_grad_norm_cap "${online_hard_grad_norm_cap}" \
    --online_skip_nonfinite_loss true \
    --online_abort_on_lora_nan true \
    --use_lora "${use_lora}" \
    --lora_r "${lora_r}" \
    --lora_alpha "${lora_alpha}" \
    --lora_dropout "${lora_dropout}" \
    --vllm_max_lora_rank "${vllm_max_lora_rank}" \
    --use_deepspeed "${use_deepspeed}" \
    --deepspeed_config_path "${deepspeed_config_path}" \
    --deepspeed_zero_stage "${deepspeed_zero_stage}" \
    --deepspeed_offload_optimizer "${deepspeed_offload_optimizer}" \
    --deepspeed_offload_param "${deepspeed_offload_param}" \
    --deepspeed_reduce_bucket_size "${deepspeed_reduce_bucket_size}" \
    --deepspeed_allgather_bucket_size "${deepspeed_allgather_bucket_size}" \
    --deepspeed_stage3_param_persistence_threshold "${deepspeed_stage3_param_persistence_threshold}" \
    --deepspeed_stage3_prefetch_bucket_size "${deepspeed_stage3_prefetch_bucket_size}" \
    --online_vllm_enforce_eager "${online_vllm_enforce_eager}" \
    --enable_thinking false \
    --use_all_wrong_gt_preference true \
    --online_pref_min_avg_logprob_chosen -3 \
    --online_pref_min_avg_logprob_rejected -3 \
    --online_pref_loss_only "${online_pref_loss_only}" \
    --online_mle_on_correct_only "${online_mle_on_correct_only}" \
    --lambda_mle "${lambda_mle}" \
    --lambda_pref "${lambda_pref}" \
    --lambda_gt "${lambda_gt}" \
    --use_all_wrong_gt_preference false \
    > "${log_file}" 2>&1
}

# 1) Full mixed objective
# run_case "mixed" "false" "false" "1.0" "0.25" "0.5"

# 2) Preference-only branch (disable MLE/GT branch)
run_case "pref_only" "true" "false" "0.0" "0.25" "0.0"

# 3) MLE-only-on-correct branch
# run_case "mle_only" "false" "true" "1.0" "0.0" "0.0"

echo
echo "[NAN-DEBUG] ===== summary ====="
python - <<'PY'
from collections import Counter
from pathlib import Path
import re

log_dir = Path("logs")
logs = sorted(log_dir.glob("nan_debug_*.out"))
if not logs:
    raise SystemExit("No nan_debug logs found.")

for p in logs:
    c = Counter()
    optimizer_steps = 0
    no_update_rollouts = 0
    entropy_nan = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "optimizer_step=" in line:
            optimizer_steps += 1
        if "no optimizer update applied" in line:
            no_update_rollouts += 1
        if "entropy_overall_mean=nan" in line:
            entropy_nan += 1
        m = re.search(r"skip optimizer update reason=([^\n]+)", line)
        if m:
            c[m.group(1).strip()] += 1

    print(f"\n[{p.name}]")
    print(f"optimizer_steps={optimizer_steps}, no_update_rollouts={no_update_rollouts}, entropy_nan_steps={entropy_nan}")
    if not c:
        print("skip_reasons: none")
        continue
    print("skip_reasons:")
    for k, v in c.most_common():
        print(f"  {v:>3}  {k}")
PY

echo
echo "[NAN-DEBUG] outputs saved under: ${root_out}"
