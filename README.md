# DAPO Preference Learning (Qwen3-4B + vLLM)

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
python train_dapo_preference.py \
  --dataset_path /path/to/dapo-math-17k.parquet \
  --model_path /path/to/Qwen3-4B \
  --output_dir /path/to/outputs/qwen3-4b-pref-online \
  --max_source_samples 17000 \
  --online_steps 30 \
  --rollout_batch_size 256 \
  --rollout_n 8 \
  --online-pairs-per-step 32 \
  --tensor_parallel_size 1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_new_tokens 8192 \
  --max_length 8192 \
  --beta 0.1 \
  --prompt_smoothing_alpha 1.0 \
  --prompt_smoothing_beta 1.0 \
  --prompt_weight_gamma 1.0 \
  --prompt_mode random \
  --prompt_candidates_file config/prompt_candidates_en.txt
```

## SLURM Script

Use this ready-to-run example:

- `run_dapo_pref_qwen3_4b.sh`
- `run_dapo_pref_qwen3_1b.sh`

It runs online rollout + preference updates and writes outputs under
`outputs/dapo_pref_4b_1gpu/<timestamp>_job<id>/`.

The SLURM defaults are aligned with local eval decoding (`eval_math_vllm_local.sh`):

- `temperature=0.6`
- `top_p=0.95`
- `max_new_tokens=8192`
- `enable_thinking=false`
