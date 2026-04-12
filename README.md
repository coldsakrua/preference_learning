# DAPO Preference Learning (Qwen3-4B + vLLM)

This repository now uses an online-only (on-policy) training pipeline:

1. Rollout with `vLLM` (`n=2` per sample) on current policy.
2. Keep samples with exactly one correct and one wrong answer.
3. Skip pairs where two rollouts share the same normalized final answer.
4. Update current policy immediately with:

```text
L = -log(sigmoid(beta * (logpi(y+|x) - logpi(y-|x)))) + lambda * CE(chosen)
```

Default `lambda` is `0.02` (`--chosen_ce_weight 0.02`).
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
- `--sample_rejected_requires_final_answer true` (default): reject truncated/no-final-answer negatives at sampling time
- `--sample_chosen_requires_final_answer false` (optional stricter filter)
- Same-answer dedup is enabled in sampling: if two rollouts have the same normalized final answer, the pair is skipped.

Answer parsing in sampling is robust to markdown variants, for example:

- `Answer: 10`
- `**Answer:** 10`
- `**Answer: 10**`

In online mode, sampled pairs are logged to `<output_dir>/online_pairs.jsonl`.

## Usage

### Online training (only supported mode)

```bash
python train_dapo_preference.py \
  --dataset_path /path/to/dapo-math-17k.parquet \
  --online_init_model_path /path/to/Qwen3-4B \
  --rollout_model_path /path/to/Qwen3-4B \
  --output_dir /path/to/outputs/qwen3-4b-pref-online \
  --max_source_samples 17000 \
  --online_steps 30 \
  --rollout_batch_size 256 \
  --online-pairs-per-step 32 \
  --tensor_parallel_size 1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_new_tokens 8192 \
  --beta 0.1 \
  --chosen_ce_weight 0.02 \
  --prompt_mode random \
  --prompt_candidates_file config/prompt_candidates_en.txt
```

## SLURM Script

Use this ready-to-run example:

- `run_dapo_pref_qwen3_4b_1gpu.sh`

It runs online rollout + preference updates and writes outputs under
`outputs/dapo_pref_4b_1gpu/<timestamp>_job<id>/`.

The SLURM defaults are aligned with local eval decoding (`eval_math_vllm_local.sh`):

- `temperature=0.6`
- `top_p=0.95`
- `max_new_tokens=8192`
- `enable_thinking=false`
