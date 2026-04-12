# DAPO Preference Learning (Qwen3-4B + vLLM)

This repository provides a two-stage training pipeline:

1. Rollout with `vLLM` (`n=2` per sample), then keep pairs with exactly one correct and one wrong answer.
2. Preference training with:

```text
L = -log(sigmoid(beta * (logpi(y+|x) - logpi(y-|x))))
```

Length-normalized log-prob is supported (`--length_average true`).

Main script: `train_dapo_preference.py`

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

When generating preference pairs, the selected `system_prompt` is saved into each JSONL row and reused during training.

## Usage

### 1) Generate preference pairs only

```bash
python train_dapo_preference.py \
  --stage generate \
  --dataset_path /path/to/dapo-math-17k.parquet \
  --rollout_model_path /path/to/Qwen3-4B \
  --preference_pairs_path /path/to/outputs/dapo_pref_pairs.jsonl \
  --max_source_samples 17000 \
  --target_pairs 5000 \
  --rollout_batch_size 128 \
  --tensor_parallel_size 1 \
  --prompt_mode random \
  --prompt_candidates_file config/prompt_candidates_en.txt
```

### 2) Train only

```bash
python train_dapo_preference.py \
  --stage train \
  --train_model_path /path/to/Qwen3-4B \
  --preference_pairs_path /path/to/outputs/dapo_pref_pairs.jsonl \
  --output_dir /path/to/outputs/qwen3-4b-pref \
  --num_epochs 1 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --beta 0.1 \
  --length_average true \
  --record_train_samples true
```

When `--record_train_samples true`, sampled records are saved to:

- default: `<output_dir>/train_sampled_pairs.jsonl`
- custom path: `--train_sample_log_path /path/to/sampled.jsonl`
- optional cap: `--train_sample_log_max_records 100000`

### 3) Run all stages

```bash
python train_dapo_preference.py \
  --stage all \
  --dataset_path /path/to/dapo-math-17k.parquet \
  --rollout_model_path /path/to/Qwen3-4B \
  --train_model_path /path/to/Qwen3-4B \
  --preference_pairs_path /path/to/outputs/dapo_pref_pairs.jsonl \
  --output_dir /path/to/outputs/qwen3-4b-pref
```

## SLURM Script

Use this ready-to-run example:

- `run_dapo_pref_qwen3_4b_1gpu.sh`

It does:

1. rollout + pair mining (with `--prompt_mode random`)
2. preference training

and writes outputs under `outputs/dapo_pref_4b_1gpu/<timestamp>_job<id>/`.
