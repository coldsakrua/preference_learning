#!/bin/bash
#SBATCH -o logs/eval_math_local.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=24:00:00

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs outputs

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
# 双卡下 custom all-reduce 在部分环境会触发 invalid argument，默认关闭更稳。
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"

model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}

# 1=关闭 Qwen thinking/CoT（--no-thinking）；0=开启 CoT（与 eval 默认一致）
NO_THINKING=${NO_THINKING:-1}
datasets_csv=${DATASETS:-math500,aime24,aime25,aime26,hmmt25}
data_format=${DATA_FORMAT:-auto}
# LoRA: set CHECKPOINT_DIR or LORA_PATH (adapter dir, or .../train — eval picks final/ or lora_adapter/).
checkpoint_dir=${CHECKPOINT_DIR:-${LORA_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/outputs/dapo_pref_4b_1gpu/20260413_172617_job1374902/train/final}}
max_lora_rank=${MAX_LORA_RANK:-${VLLM_MAX_LORA_RANK:-64}}
use_lora=${USE_LORA:-1}
num_samples=${NUM_SAMPLES:-0}
val_n=${VAL_N:-16}
pass_at_k=${PASS_AT_K:-1,4,8,16}
max_new_tokens=${MAX_NEW_TOKENS:-4096}
temperature=${TEMPERATURE:-0.6}
top_p=${TOP_P:-0.95}
seed=${SEED:-42}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-2}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
max_model_len=${MAX_MODEL_LEN:-0}
generate_batch_size=${GENERATE_BATCH_SIZE:-16}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_tag="${stamp}_job${SLURM_JOB_ID}"
else
  run_tag="${stamp}"
fi
if [[ "${NO_THINKING}" == "1" ]]; then
  _eval_cot_dir=no_cot
else
  _eval_cot_dir=cot
fi
output_json=${OUTPUT_JSON:-outputs/eval_math_local/${_eval_cot_dir}/eval_${run_tag}.json}

mkdir -p "$(dirname "${output_json}")"

echo "[EVAL-2GPU] model_path=${model_path}"
echo "[EVAL-2GPU] checkpoint_dir=${checkpoint_dir:-<none>}"
echo "[EVAL-2GPU] USE_LORA=${use_lora} (1=use LoRA, 0=disable LoRA)"
echo "[EVAL-2GPU] DATASETS=${datasets_csv}"
echo "[EVAL-2GPU] NO_THINKING=${NO_THINKING} (1=no CoT, 0=CoT) -> subdir=${_eval_cot_dir}"
echo "[EVAL-2GPU] TP=${tensor_parallel_size}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[EVAL-2GPU] output_json=${output_json}"

cmd=(
  python eval_math_vllm_local.py
  --model-path "${model_path}"
  --data-format "${data_format}"
  --output-json "${output_json}"
  --num-samples "${num_samples}"
  --val-n "${val_n}"
  --pass-at-k "${pass_at_k}"
  --generate-batch-size "${generate_batch_size}"
  --max-new-tokens "${max_new_tokens}"
  --temperature "${temperature}"
  --top-p "${top_p}"
  --seed "${seed}"
  --tensor-parallel-size "${tensor_parallel_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
)

IFS=',' read -ra _ds <<< "${datasets_csv}"
for _n in "${_ds[@]}"; do
  _n="${_n#"${_n%%[![:space:]]*}"}"
  _n="${_n%"${_n##*[![:space:]]}"}"
  [[ -z "${_n}" ]] && continue
  cmd+=(--dataset="${_n}")
done

if [[ "${use_lora}" == "1" ]]; then
  if [[ -n "${checkpoint_dir}" ]]; then
    cmd+=(--lora-path "${checkpoint_dir}")
  fi
  if [[ -n "${checkpoint_dir}" && -n "${max_lora_rank}" && "${max_lora_rank}" != "0" ]]; then
    cmd+=(--max-lora-rank "${max_lora_rank}")
  fi
fi

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ "${NO_THINKING}" == "1" ]]; then
  cmd+=(--no-thinking)
fi

"${cmd[@]}"

echo "[EVAL-2GPU] done -> ${output_json}"

