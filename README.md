# DAPO Preference Learning (Qwen / Llama + vLLM)

This repository now uses an online-only (on-policy) training pipeline:

1. Rollout with `vLLM` (`n=8` per sample by default, via `--rollout_n`) on current policy.
2. For each prompt:
   - Mixed correct/wrong rollouts: use hidden-state nearest-neighbor pair mining (each wrong sample pairs with its nearest correct sample).
   - All rollout samples correct: use weighted MLE on all correct sampled responses.
   - All rollout samples wrong: use GT-positive preference when valid gold rationale exists; otherwise skip.
3. Prompt-level rarity weighting uses smoothed correctness:

```text
rho_hat = (n_correct + alpha) / (n_total + alpha + beta)
prompt_weight = clip((1 - rho_hat)^gamma, w_min, w_max)
```

4. Update current policy immediately in the same rollout step (no cross-step cache).

Length-normalized log-prob is always enabled.
All wrong-format / no-final-answer / long responses are retained in training and treated as wrong when verifier says incorrect.

Main script: `train_preference.py`

Tested model families in this repo:

- Qwen3 (1B/4B style chat models)
- Llama 3.2 (e.g. `llama3.2-1b`)

## Versions

- `vllm==0.8.5`
- `flash-attn==2.7.3.post1`

## Data Fields

The script is already adapted to `dapo-math-17k.parquet`:

- `prompt`: list of chat messages (uses the last user message as question)
- `reward_model.ground_truth`: gold answer
- `extra_info.index`: sample id (optional)

## System Prompt Options (English)

You can use:

- one fixed system prompt
- multiple candidate prompts
- random selection per sample

Supported arguments:

- `--prompt_mode {none,fixed,random}`
- `--system_prompt "..."` (single prompt)
- `--prompt_candidate "..."` (repeatable)
- `--prompt_candidates_file config/prompt_candidates_en.txt`
- `--use_default_prompt_candidates true`
- `--prompt_fixed_index 0`
- `--prompt_smoothing_alpha 1.0`
- `--prompt_smoothing_beta 1.0`
- `--prompt_weight_gamma 1.0`

Answer parsing in sampling follows DAPO-style last-line extraction:

- only the last non-empty line is used
- it must start with `Answer:` (case-insensitive, `：` also accepted)
- both `Answer: 10` and `Answer: $10` are accepted

In online mode, sampled pairs are logged to `<output_dir>/online_pairs.jsonl`.

## Usage

### Online training (only supported mode)

```bash
python train_preference.py \
  --dataset_path /path/to/dapo-math-17k.parquet \
  --model_path /path/to/Qwen3-4B \
  --output_dir /path/to/outputs/qwen3-4b-pref-online \
  --max_source_samples 17000 \
  --online_steps 30 \
  --rollout_batch_size 256 \
  --rollout_n 8 \
  --online-pairs-per-step 32 \
  --tensor_parallel_size 1 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --min_p 0 \
  --max_new_tokens 3072 \
  --max_length 4096 \
  --beta 0.1 \
  --prompt_smoothing_alpha 1.0 \
  --prompt_smoothing_beta 1.0 \
  --prompt_weight_gamma 1.0 \
  --prompt_mode random \
  --prompt_candidates_file config/prompt_candidates_en.txt
```

## SLURM Script

Use these ready-to-run examples:

- `run_pref_qwen3_4b.sh`
- `run_pref_qwen3_1b.sh`
- `run_pref_llama3_2_1b.sh`

They run online rollout + preference updates and write outputs under
`outputs/pref_*/<timestamp>_job<id>/`.

The SLURM defaults (training scripts) are:

- `temperature=0.7` (non-thinking) / `0.6` (thinking)
- `top_p=0.8` (non-thinking) / `0.95` (thinking)
- `top_k=20`, `min_p=0`, `presence_penalty=0`
- `max_new_tokens=3072`
- `max_length=4096` (`rollout_max_model_len=4096`)

## GRPO RL (trl==0.22.1)

You can further run GRPO RL on top of a preference-tuned checkpoint:

```bash
python train_grpo_dapo_preference.py \
  --dataset_path /path/to/dapo-math-17k.parquet \
  --model_path /path/to/outputs/.../train/final \
  --output_dir /path/to/outputs/grpo_run \
  --num_generations 8 \
  --max_completion_length 1024 \
  --beta 0.04
```

SLURM launcher: `run_grpo_qwen3_1b_grpo.sh`

## Llama 3.2 (1B) Quick Start

Assuming your base model is already local at `.../model/llama3.2-1b`:

```bash
# online preference training (vLLM rollout + HF update)
bash run_pref_llama3_2_1b.sh

# LoRA SFT
bash run_sft_lora_llama3_2_1b.sh

# local eval with vLLM 0.8.5
bash eval_math_vllm_local_llama3_2_1b.sh
```

Common overrides:

- `MODEL_PATH=/path/to/llama3.2-1b`
- `VLLM_DTYPE=bfloat16`
- `USE_LORA=1 CHECKPOINT_DIR=/path/to/train/final`
