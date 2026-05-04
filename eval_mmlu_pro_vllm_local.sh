#!/bin/bash
#SBATCH -o eval_logs/eval_mmlu_pro_local.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/preference_learning

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p eval_logs outputs

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-base}

NO_THINKING=${NO_THINKING:-1}
dataset_name=${DATASET_NAME:-mmlu-pro}
data_format=${DATA_FORMAT:-mmlu_pro_hf}
mmlu_pro_config=${MMLU_PRO_CONFIG:-default}
mmlu_pro_split=${MMLU_PRO_SPLIT:-test}
checkpoint_dir=${CHECKPOINT_DIR:-${LORA_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/outputs/sft_lora_qwen3_4b/20260428_105621_job1479722/train/checkpoint-640}}
max_lora_rank=${MAX_LORA_RANK:-${VLLM_MAX_LORA_RANK:-64}}
use_lora=${USE_LORA:-1}
num_samples=${NUM_SAMPLES:-0}
val_n=${VAL_N:-16}
pass_at_k=${PASS_AT_K:-1,4,8,16}
max_new_tokens=${MAX_NEW_TOKENS:-4096}
max_model_len=${MAX_MODEL_LEN:-4096}
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-0.8}
top_k=${TOP_K:-20}
min_p=${MIN_P:-0.0}
presence_penalty=${PRESENCE_PENALTY:-0.0}
seed=${SEED:-42}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
generate_batch_size=${GENERATE_BATCH_SIZE:-64}
force_base_tokenizer=${FORCE_BASE_TOKENIZER:-1}

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
output_json=${OUTPUT_JSON:-eval_outputs/eval_mmlu_pro_local/${_eval_cot_dir}/eval_${run_tag}.json}

mkdir -p "$(dirname "${output_json}")"
echo "[EVAL-MMLU-PRO] model_path=${model_path}"
echo "[EVAL-MMLU-PRO] checkpoint_dir=${checkpoint_dir:-<none>}"
echo "[EVAL-MMLU-PRO] USE_LORA=${use_lora} (1=use LoRA, 0=disable LoRA)"
echo "[EVAL-MMLU-PRO] dataset=${dataset_name} format=${data_format} config=${mmlu_pro_config} split=${mmlu_pro_split}"
echo "[EVAL-MMLU-PRO] NO_THINKING=${NO_THINKING} (1=no CoT, 0=CoT) -> subdir=${_eval_cot_dir}"
echo "[EVAL-MMLU-PRO] FORCE_BASE_TOKENIZER=${force_base_tokenizer} (1=base tokenizer/chat_template)"
echo "[EVAL-MMLU-PRO] output_json=${output_json}"

cmd=(
  python eval_math_vllm_local.py
  --model-path "${model_path}"
  --dataset "${dataset_name}"
  --data-format "${data_format}"
  --mmlu-pro-config "${mmlu_pro_config}"
  --mmlu-pro-split "${mmlu_pro_split}"
  --output-json "${output_json}"
  --num-samples "${num_samples}"
  --val-n "${val_n}"
  --pass-at-k "${pass_at_k}"
  --generate-batch-size "${generate_batch_size}"
  --max-new-tokens "${max_new_tokens}"
  --temperature "${temperature}"
  --top-p "${top_p}"
  --top-k "${top_k}"
  --min-p "${min_p}"
  --presence-penalty "${presence_penalty}"
  --seed "${seed}"
  --tensor-parallel-size "${tensor_parallel_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
)

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

if [[ "${force_base_tokenizer}" == "1" ]]; then
  cmd+=(--force-base-tokenizer)
fi

"${cmd[@]}"

echo "[EVAL-MMLU-PRO] done -> ${output_json}"
