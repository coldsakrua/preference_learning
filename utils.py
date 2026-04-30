from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import torch


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    value = v.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DapoSample:
    prompt: str
    ground_truth: str
    gold_rationale: str
    sample_id: str


DEFAULT_GOLD_RATIONALE_KEY_PATHS: Tuple[str, ...] = (
    "reward_model.gold_rationale",
    "reward_model.rationale",
    "reward_model.solution",
    "reward_model.reference_solution",
    "extra_info.gold_rationale",
)

DEFAULT_MATH_HF_USER_CONTENT_SUFFIX = (
    "\n\nPlease solve the problem with step-by-step reasoning. "
    "End your full response with exactly one final line of the form:\n"
    "Answer: <your final answer>."
)


def extract_user_prompt(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    user_contents: List[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                user_contents.append(content)
    if user_contents:
        return user_contents[-1]
    for message in reversed(messages):
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            if content:
                return content
    return ""


def _get_nested_field(obj: object, dotted_path: str) -> str:
    current = obj
    for key in dotted_path.split("."):
        if not isinstance(current, dict):
            return ""
        if key not in current:
            return ""
        current = current.get(key)
    if current is None:
        return ""
    return str(current).strip()


def strip_prompt_prefix_from_text(prompt: str, text: str) -> str:
    p = str(prompt).strip()
    t = str(text).strip()
    if not p or not t:
        return t
    if t.startswith(p):
        return t[len(p) :].lstrip()
    return t


def _extract_gold_rationale_text(
    prompt_text: str,
    reward_obj: object,
    extra_obj: object,
    key_paths: Sequence[str],
) -> str:
    for path in key_paths:
        path = str(path).strip()
        if not path:
            continue
        if path.startswith("reward_model."):
            text = _get_nested_field(reward_obj, path[len("reward_model.") :])
        elif path.startswith("extra_info."):
            text = _get_nested_field(extra_obj, path[len("extra_info.") :])
        else:
            text = _get_nested_field(reward_obj, path)
        if not text:
            continue
        text = strip_prompt_prefix_from_text(prompt_text, text)
        if text:
            return text
    return ""


def iter_dapo_samples(
    parquet_path: str,
    scan_batch_size: int,
    max_source_samples: Optional[int],
    gold_rationale_key_paths: Sequence[str],
    require_gold_rationale: bool,
) -> Iterator[DapoSample]:
    parquet_file = pq.ParquetFile(parquet_path)
    yielded = 0
    columns = ["prompt", "reward_model", "extra_info"]
    for record_batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=columns):
        prompt_col = record_batch.column("prompt").to_pylist()
        reward_col = record_batch.column("reward_model").to_pylist()
        extra_col = record_batch.column("extra_info").to_pylist()
        for prompt_obj, reward_obj, extra_obj in zip(prompt_col, reward_col, extra_col):
            prompt_text = extract_user_prompt(prompt_obj)
            if not prompt_text:
                continue
            ground_truth = ""
            if isinstance(reward_obj, dict):
                ground_truth = str(reward_obj.get("ground_truth", "")).strip()
            if not ground_truth:
                continue
            gold_rationale = _extract_gold_rationale_text(
                prompt_text=prompt_text,
                reward_obj=reward_obj,
                extra_obj=extra_obj,
                key_paths=gold_rationale_key_paths,
            )
            if require_gold_rationale and not gold_rationale:
                continue
            sample_id = ""
            if isinstance(extra_obj, dict):
                sample_id = str(extra_obj.get("index", "")).strip()
            if not sample_id:
                sample_id = f"row-{yielded}"
            yield DapoSample(
                prompt=prompt_text,
                ground_truth=ground_truth,
                gold_rationale=gold_rationale,
                sample_id=sample_id,
            )
            yielded += 1
            if max_source_samples is not None and yielded >= max_source_samples:
                return


def detect_parquet_dataset_layout(parquet_path: str) -> str:
    """Return ``dapo`` or ``math_hf`` based on column names."""
    schema = pq.ParquetFile(parquet_path).schema_arrow
    names = set(schema.names)
    if "prompt" in names and "reward_model" in names:
        return "dapo"
    if "problem" in names and "solution" in names:
        return "math_hf"
    raise ValueError(
        f"Unsupported parquet layout in {parquet_path!r}; columns={sorted(names)}. "
        "Expected DAPO (prompt+reward_model) or MATH HF (problem+solution)."
    )


def extract_boxed_answer_last(text: str) -> str:
    """Return inner text of the last ``\\boxed{...}`` region, or empty string."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""
    i = idx
    num_left_braces = 0
    right_brace_idx: Optional[int] = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return ""
    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return ""


def ground_truth_from_math_solution(solution: str) -> str:
    if not solution:
        return ""
    boxed = extract_boxed_answer_last(solution)
    if boxed:
        return boxed
    return extract_final_answer_from_any_line(solution).strip()


def iter_math_hf_samples(
    parquet_path: str,
    scan_batch_size: int,
    max_source_samples: Optional[int],
    gold_rationale_key_paths: Sequence[str],
    require_gold_rationale: bool,
) -> Iterator[DapoSample]:
    del gold_rationale_key_paths  # API symmetry with ``iter_dapo_samples``; unused here.
    del require_gold_rationale  # API symmetry with ``iter_dapo_samples``; unused here.
    parquet_file = pq.ParquetFile(parquet_path)
    yielded = 0
    cols = ["problem", "solution"]
    for record_batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=cols):
        problems = record_batch.column("problem").to_pylist()
        solutions = record_batch.column("solution").to_pylist()
        for problem_obj, solution_obj in zip(problems, solutions):
            prompt_text = str(problem_obj or "").strip()
            solution_text = str(solution_obj or "").strip()
            if not prompt_text or not solution_text:
                continue
            ground_truth = ground_truth_from_math_solution(solution_text)
            if not ground_truth:
                continue
            sample_id = f"math-{yielded}"
            yield DapoSample(
                prompt=prompt_text,
                ground_truth=ground_truth,
                gold_rationale=solution_text,
                sample_id=sample_id,
            )
            yielded += 1
            if max_source_samples is not None and yielded >= max_source_samples:
                return


_ANSWER_LINE_CANONICAL = re.compile(r"^\s*(?:final\s+)?answer\s*[:\uFF1A]\s*(.+?)\s*$", flags=re.IGNORECASE)
_ANSWER_LINE_PARQUET = re.compile(r"^\s*answer\s*[:\uFF1A]\s*(.+?)\s*$", flags=re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_LATEX_FRAC = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")


def strip_outer_formatting(text: str) -> str:
    s = text.strip()
    wrappers = [
        ("**", "**"),
        ("*", "*"),
        ("`", "`"),
        ("$", "$"),
        ("\\(", "\\)"),
        ("\\[", "\\]"),
    ]
    changed = True
    while changed and s:
        changed = False
        for left, right in wrappers:
            if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
                s = s[len(left) : -len(right)].strip()
                changed = True
    return s


def normalize_answer_line_for_parse(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^[>\-\*\#\s]+", "", s)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_answer_from_line(line: str) -> Optional[str]:
    normalized_line = normalize_answer_line_for_parse(line)
    match = _ANSWER_LINE_CANONICAL.match(normalized_line)
    if not match:
        return None
    answer = strip_outer_formatting(match.group(1))
    if not answer:
        return None
    return answer


def parse_answer_from_line_parquet(line: str) -> Optional[str]:
    raw_line = line.strip()
    match = _ANSWER_LINE_PARQUET.match(raw_line)
    if not match:
        return None
    answer = match.group(1).strip()
    if answer.startswith("$"):
        answer = answer[1:].strip()
    answer = strip_outer_formatting(answer)
    if answer.startswith("$"):
        answer = answer[1:].strip()
    if not answer:
        return None
    return answer


def extract_final_answer_if_last_line(text: str) -> tuple[bool, str]:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, ""
    answer = parse_answer_from_line_parquet(lines[-1])
    if answer is None:
        return False, ""
    return True, answer


def extract_rollout_scored_answer(text: str) -> tuple[bool, str]:
    """Parse a model rollout for grading: last-line ``Answer:`` first, else last ``\\boxed{}``."""
    has_last, ans = extract_final_answer_if_last_line(text)
    if has_last and ans.strip():
        return True, ans.strip()
    boxed = extract_boxed_answer_last(text)
    if boxed.strip():
        return True, boxed.strip()
    return False, ""


def extract_final_answer_from_any_line(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        parsed = parse_answer_from_line(line)
        if parsed:
            return parsed
        parsed = parse_answer_from_line_parquet(line)
        if parsed:
            return parsed
    return ""


def extract_reference_answer_for_verifier(text: str) -> str:
    """Ground-truth text from a reference solution (e.g. MATH): ``Answer:`` lines or ``\\boxed{}``."""
    s = extract_final_answer_from_any_line(text)
    if s.strip():
        return s.strip()
    boxed = extract_boxed_answer_last(text)
    return boxed.strip()


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = _BOXED.sub(r"\1", answer)
    answer = _LATEX_FRAC.sub(r"\1/\2", answer)
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")
    answer = answer.replace("\u3002", "")
    answer = re.sub(r"\s+", "", answer)
    answer = answer.rstrip(".")
    answer = answer.strip("()[]{}")
    return answer.lower()


def to_number_if_simple(answer: str) -> Optional[float]:
    if re.fullmatch(r"-?\d+(\.\d+)?", answer):
        return float(answer)
    if re.fullmatch(r"-?\d+/-?\d+", answer):
        numerator, denominator = answer.split("/")
        denominator_value = float(denominator)
        if abs(denominator_value) < 1e-12:
            return None
        return float(numerator) / denominator_value
    return None


def answer_text_matches(predicted_answer: str, ground_truth: str) -> bool:
    predicted = normalize_answer(predicted_answer)
    target = normalize_answer(ground_truth)
    if predicted and predicted == target:
        return True
    predicted_num = to_number_if_simple(predicted)
    target_num = to_number_if_simple(target)
    if predicted_num is not None and target_num is not None:
        return abs(predicted_num - target_num) <= 1e-6
    return False


def compute_smoothed_correct_rate(
    r_cnt: int,
    total: int,
    alpha: float,
    beta: float,
) -> float:
    denom = float(total + alpha + beta)
    if denom <= 0:
        return 0.0
    return float((r_cnt + alpha) / denom)


def compute_prompt_rarity_weight(
    rho_hat: float,
    gamma: float,
    w_min: float,
    w_max: float,
) -> float:
    raw = max(0.0, 1.0 - float(rho_hat)) ** float(gamma)
    return float(min(max(raw, w_min), w_max))


def build_parser(default_system_prompt: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DAPO preference training with vLLM rollout.")
    # Compatibility arg injected by DeepSpeed/Torch distributed launchers.
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default="/path/to/dapo-math-17k.parquet")
    parser.add_argument(
        "--dataset_layout",
        type=str,
        default="auto",
        choices=["auto", "dapo", "math_hf"],
        help=(
            "Parquet layout: ``dapo`` (prompt+reward_model+extra_info), "
            "``math_hf`` (problem+solution, e.g. HF MATH train parquet). "
            "``auto`` picks by column names."
        ),
    )
    parser.add_argument(
        "--user_content_suffix",
        type=str,
        default="",
        help="Appended to each raw user prompt before chat templating (e.g. output format for HF MATH).",
    )
    parser.add_argument(
        "--auto_math_hf_user_suffix",
        type=str2bool,
        default=True,
        help="If true and layout is math_hf and --user_content_suffix is empty, append DEFAULT_MATH_HF suffix.",
    )
    parser.add_argument(
        "--gold_rationale_key",
        action="append",
        default=list(DEFAULT_GOLD_RATIONALE_KEY_PATHS),
        help=(
            "Dotted key path for gold rationale text. "
            "Repeat this arg for fallback lookup order, e.g. --gold_rationale_key reward_model.solution."
        ),
    )
    parser.add_argument(
        "--require_gold_rationale_for_all_wrong",
        type=str2bool,
        default=False,
        help=(
            "If true, source samples without gold rationale are skipped at loading time "
            "so all-wrong can always use GT preference."
        ),
    )
    parser.add_argument("--model_path", type=str, default="/path/to/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-4b-pref")
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument(
        "--max_source_samples",
        type=int,
        default=17000,
        help="Maximum source prompts scanned for online objective mining.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Optional anti-repetition penalty (recommended range: 0~2).",
    )
    parser.add_argument("--rollout_n", type=int, default=8, help="Number of sampled responses per prompt during rollout.")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--rollout_max_model_len", type=int, default=32768)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="none",
        choices=["none", "fixed", "random"],
        help="System prompt strategy. 'random' picks one prompt per sample from pool.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=default_system_prompt,
        help="Single English system prompt. Used directly when prompt_mode=fixed (unless index points elsewhere).",
    )
    parser.add_argument(
        "--prompt_candidate",
        action="append",
        default=[],
        help="Add one candidate system prompt. Repeat this argument for multiple candidates.",
    )
    parser.add_argument("--prompt_candidates_file", type=str, default="", help="TXT/JSON file containing candidate English system prompts.")
    parser.add_argument("--use_default_prompt_candidates", type=str2bool, default=False, help="Append built-in English prompt candidates to the prompt pool.")
    parser.add_argument("--prompt_fixed_index", type=int, default=0, help="Prompt index used when prompt_mode=fixed.")
    parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        default=True,
        help=(
            "Pass enable_thinking to tokenizer chat template when supported. "
            "Ignored automatically for tokenizers that do not accept this argument."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--lambda_mle", type=float, default=1.0, help="Weight for L_mle.")
    parser.add_argument("--lambda_pref", type=float, default=0.25, help="Weight for mixed-prompt L_pref.")
    parser.add_argument("--lambda_gt", type=float, default=0.5, help="Weight for all-wrong GT preference loss.")
    parser.add_argument(
        "--online_mle_on_correct_only",
        type=str2bool,
        default=False,
        help=(
            "Online only: use only correct rollout trajectories for MLE updates. "
            "Disable mixed/all-wrong preference losses."
        ),
    )
    parser.add_argument(
        "--online_pref_loss_only",
        type=str2bool,
        default=False,
        help=(
            "Online only: use only mixed-prompt preference loss (L_pref). "
            "Disable MLE and all-wrong GT preference losses."
        ),
    )
    parser.add_argument("--prompt_smoothing_alpha", type=float, default=1.0, help="Alpha in rho_hat=(r+alpha)/(n+alpha+beta).")
    parser.add_argument("--prompt_smoothing_beta", type=float, default=1.0, help="Beta in rho_hat=(r+alpha)/(n+alpha+beta).")
    parser.add_argument("--prompt_weight_gamma", type=float, default=1.0, help="Gamma in prompt rarity weighting clip((1-rho_hat)^gamma, w_min, w_max).")
    parser.add_argument("--prompt_weight_min", type=float, default=0.1, help="Min clip for prompt rarity weight.")
    parser.add_argument("--prompt_weight_max", type=float, default=1.0, help="Max clip for prompt rarity weight.")
    parser.add_argument(
        "--positive_weight_mode",
        type=str,
        default="nll_softmax",
        choices=["nll_softmax", "uniform"],
        help="Correct-trajectory weighting mode for MLE term.",
    )
    parser.add_argument("--positive_weight_tau", type=float, default=1.0, help="Tau in softmax(tau * avg_nll) for correct trajectories.")
    parser.add_argument("--hidden_layer_offset", type=int, default=4, help="Use hidden_states[-hidden_layer_offset] for hidden-state pooling.")
    parser.add_argument("--rollout_feature_micro_batch_size", type=int, default=8, help="Micro-batch size for rollout feature extraction (avg_logprob + hidden_vec).")
    parser.add_argument("--use_all_wrong_gt_preference", type=str2bool, default=True, help="Enable GT-positive preference branch for all-wrong prompts.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--logprob_micro_batch_size",
        type=int,
        default=0,
        help=(
            "Online only: max sequences per HF forward when computing logp for preference/MLE. "
            "0 = no chunking (one forward pass per branch over the whole in-step batch; matches "
            "on-policy training step best). "
            "Set >0 only to reduce peak VRAM on long completions; multiple backward() chunks "
            "accumulate grads equivalent to one full-batch loss (same total gradient if stable)."
        ),
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--online_hard_grad_norm_cap", type=float, default=5.0, help="Online only: hard cap for pre-step grad norm. If grad_norm > cap, skip this optimizer update. Use <=0 to disable.")
    parser.add_argument("--online_loss_value_cap", type=float, default=20.0, help="Online only: hard cap for each micro-batch loss_chunk absolute value. If exceeded, skip this optimizer update. Use <=0 to disable.")
    parser.add_argument(
        "--online_gap_clip_abs",
        type=float,
        default=0.0,
        help=(
            "Online only: if >0, clamp preference gap (chosen_logp - rejected_logp) to "
            "[-online_gap_clip_abs, +online_gap_clip_abs] before preference loss."
        ),
    )
    parser.add_argument(
        "--online_pref_min_avg_logprob_chosen",
        type=float,
        default=None,
        help=(
            "Online: require chosen (correct or GT-positive) trajectory avg_logprob >= this value. "
            "More negative = stricter (e.g. -5 drops very low-confidence chosen). "
            "Omit to disable. Note: -1 is usually too aggressive for real text."
        ),
    )
    parser.add_argument(
        "--online_pref_min_avg_logprob_rejected",
        type=float,
        default=None,
        help=(
            "Online: require rejected (wrong) trajectory avg_logprob >= this value. "
            "Drops pairs where the model is extremely surprised by the completion (often "
            "correlates with grad NaN in bf16). Typical try: -4 to -6. Omit to disable."
        ),
    )
    parser.add_argument(
        "--online_mle_min_avg_logprob",
        type=float,
        default=None,
        help=(
            "Online: require MLE/correct trajectory avg_logprob >= this value before "
            "autograd. This keeps low-confidence correct samples from creating bf16 "
            "non-finite gradients. Omit to disable."
        ),
    )
    parser.add_argument("--online_skip_nonfinite_loss", type=str2bool, default=True, help="Online only: skip update when loss chunk or grad norm is non-finite.")
    parser.add_argument("--online_abort_on_lora_nan", type=str2bool, default=True, help="Online only: immediately stop if LoRA params become NaN after optimizer step.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    # Hidden compatibility switches: older debug scripts may still pass them.
    parser.add_argument("--rollout_compute_entropy", type=str2bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--hf_data_parallel", type=str2bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--use_lora", type=str2bool, default=False, help="Train with PEFT LoRA (HF forward + preference loss); vLLM rollout loads base + adapter.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA (Qwen/Llama-style attention/MLP).",
    )
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64, help="vLLM max_lora_rank when use_lora=true; must be >= lora_r.")
    parser.add_argument(
        "--online_steps",
        type=int,
        default=0,
        help=(
            "Online only: number of rollout batches to run. Each batch rolls out "
            "rollout_batch_size prompts with --rollout_n samples each; prompt-level "
            "objectives (mixed-pair preference + full-correct SFT) are consumed "
            "immediately in this rollout step (no cross-step cache). "
            "Use 0 for no limit (until source samples exhausted)."
        ),
    )
    parser.add_argument(
        "--online-pairs-per-step",
        type=int,
        default=16,
        dest="online_pairs_per_step",
        help=(
            "Online only: number of prompt-level objectives per optimizer chunk inside one rollout step. "
            "If a rollout yields fewer objectives, it still updates once with all available objectives."
        ),
    )
    parser.add_argument("--online_save_every_updates", type=int, default=0, help="Save checkpoint every N online updates. Use 0 to disable periodic checkpoints.")
    parser.add_argument(
        "--online_pairs_include_dense_rollouts",
        type=str2bool,
        default=False,
        help="If true, online_pairs.jsonl stores token_ids and hidden_vec per rollout (very large).",
    )
    parser.add_argument("--online_rollout_backend", type=str, default="vllm", choices=["vllm", "hf"], help="Online: rollout engine - vLLM (default) or Hugging Face generate.")
    parser.add_argument("--online_vllm_use_tqdm", type=str2bool, default=True, help="Online + vLLM: show tqdm progress in llm.generate.")
    parser.add_argument(
        "--online_vllm_enforce_eager",
        type=str2bool,
        default=True,
        help=(
            "Online + vLLM: use eager mode to reduce per-rollout engine init overhead "
            "when the engine is recreated frequently."
        ),
    )
    return parser
