#!/bin/bash
#SBATCH -o logs/hidden_state_divergence.%j.out
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

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset_path=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/hendrycks_math/aggregated_l3plus/train.parquet}
opsd_dataset_path_default=/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/OPSD/train-00000-of-00002.parquet
if [[ -e "${opsd_dataset_path_default}" ]]; then
  dataset_paths_csv=${DATASET_PATHS:-${dataset_path},${opsd_dataset_path_default}}
else
  dataset_paths_csv=${DATASET_PATHS:-}
fi
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}
inference_backend=${INFERENCE_BACKEND:-vllm}

seed=${SEED:-42}
max_samples=${MAX_SAMPLES:-0}
rollout_n=${ROLLOUT_N:-8}
problems_per_batch=${PROBLEMS_PER_BATCH:-128}
rollout_rounds=${ROLLOUT_ROUNDS:-8}
scan_batch_size=${SCAN_BATCH_SIZE:-256}
gen_batch_size=${GEN_BATCH_SIZE:-128}
max_prompt_tokens=${MAX_PROMPT_TOKENS:-1024}
max_reference_tokens=${MAX_REFERENCE_TOKENS:-4096}
max_new_tokens=${MAX_NEW_TOKENS:-4096}
bootstrap_max_tokens=${BOOTSTRAP_MAX_TOKENS:-${max_new_tokens}}
device=${DEVICE:-auto}
dtype=${DTYPE:-auto}
system_prompt=${SYSTEM_PROMPT:-}

inspect_only=${INSPECT_ONLY:-false}
do_sample=${DO_SAMPLE:-true}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.95}
skip_plot=${SKIP_PLOT:-false}

vllm_tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE:-1}
vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
vllm_max_model_len=${VLLM_MAX_MODEL_LEN:-4096}
vllm_dtype=${VLLM_DTYPE:-bfloat16}
vllm_enforce_eager=${VLLM_ENFORCE_EAGER:-false}

python_bin=${PYTHON_BIN:-python}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_name="${stamp}_job${SLURM_JOB_ID}"
else
  run_name="${stamp}"
fi

run_root=${RUN_ROOT:-outputs/hidden_state_distribution/${run_name}}
mkdir -p "${run_root}"

echo "[hidden-div] run_root=${run_root}"
echo "[hidden-div] model_path=${model_path}"
echo "[hidden-div] inference_backend=${inference_backend}"
echo "[hidden-div] max_samples=${max_samples} rollout_n=${rollout_n} problems_per_batch=${problems_per_batch} rollout_rounds=${rollout_rounds}"
echo "[hidden-div] scan_batch_size=${scan_batch_size} gen_batch_size=${gen_batch_size} max_new_tokens=${max_new_tokens}"

dataset_paths=()
if [[ -n "${dataset_paths_csv}" ]]; then
  IFS=',' read -r -a _dataset_raw <<< "${dataset_paths_csv}"
  for _p in "${_dataset_raw[@]}"; do
    _trimmed="$(echo "${_p}" | xargs)"
    if [[ -n "${_trimmed}" ]]; then
      dataset_paths+=("${_trimmed}")
    fi
  done
else
  dataset_paths=("${dataset_path}")
fi

if [[ "${#dataset_paths[@]}" -eq 0 ]]; then
  echo "[hidden-div] ERROR: no dataset paths resolved." >&2
  exit 1
fi

echo "[hidden-div] datasets_count=${#dataset_paths[@]}"
for ds in "${dataset_paths[@]}"; do
  echo "[hidden-div] dataset=${ds}"
done

args=(
  --dataset_paths "${dataset_paths_csv}"
  --model_path "${model_path}"
  --inference_backend "${inference_backend}"
  --max_samples "${max_samples}"
  --rollout_n "${rollout_n}"
  --problems_per_batch "${problems_per_batch}"
  --rollout_rounds "${rollout_rounds}"
  --scan_batch_size "${scan_batch_size}"
  --gen_batch_size "${gen_batch_size}"
  --max_prompt_tokens "${max_prompt_tokens}"
  --max_reference_tokens "${max_reference_tokens}"
  --max_new_tokens "${max_new_tokens}"
  --bootstrap_max_tokens "${bootstrap_max_tokens}"
  --device "${device}"
  --dtype "${dtype}"
  --seed "${seed}"
  --vllm_tensor_parallel_size "${vllm_tensor_parallel_size}"
  --vllm_gpu_memory_utilization "${vllm_gpu_memory_utilization}"
  --vllm_max_model_len "${vllm_max_model_len}"
  --vllm_dtype "${vllm_dtype}"
)

if [[ -n "${system_prompt}" ]]; then
  args+=(--system_prompt "${system_prompt}")
fi
if [[ "${inspect_only}" == "true" ]]; then
  args+=(--inspect_only)
fi
if [[ "${do_sample}" == "true" ]]; then
  args+=(--do_sample --temperature "${temperature}" --top_p "${top_p}")
fi
if [[ "${skip_plot}" == "true" ]]; then
  args+=(--skip_plot)
fi
if [[ "${vllm_enforce_eager}" == "true" ]]; then
  args+=(--vllm_enforce_eager)
fi

echo "[hidden-div] start joint analysis"
"${python_bin}" analyze_reasoning_hidden_distribution.py \
  --dataset_path "${dataset_path}" \
  --output_dir "${run_root}" \
  "${args[@]}" \
  "$@"

echo "[hidden-div] done"
echo "output_dir=${run_root}"
