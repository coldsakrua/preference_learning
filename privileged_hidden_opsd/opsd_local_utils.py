#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import torch

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise math reasoning assistant. "
    "Solve the problem step by step, then end with exactly one final line in the format: "
    "Answer: $<final_answer>."
)

DEFAULT_PROMPT_CANDIDATES = [
    "You are a careful math tutor. Show concise but correct reasoning and finish with: Answer: $<final_answer>.",
    "Solve the math problem with rigorous steps. Keep reasoning structured and end with: Answer: $<final_answer>.",
    "You are an expert competition-math assistant. Verify key steps and finish with: Answer: $<final_answer>.",
    "Reason clearly and avoid arithmetic mistakes. The last line must be: Answer: $<final_answer>.",
    "Produce a correct step-by-step solution, then output one final line: Answer: $<final_answer>.",
]

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

_EMPTY_LORA_HEALTH = {
    "lora_mean_abs": 0.0,
    "lora_max_abs": 0.0,
    "lora_nan_ratio": 0.0,
    "lora_inf_ratio": 0.0,
}
_WARNED_MISSING_CHAT_TEMPLATE = False


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
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


@dataclass
class RolloutTrajectory:
    response_text: str
    token_ids: List[int]
    is_correct: bool
    fail_type: str
    has_final_answer_line: bool
    final_answer: str
    avg_logprob: float
    avg_nll: float
    avg_entropy: float
    hidden_vec: List[float]


@dataclass
class RolloutCandidateSplit:
    responses_has_final_answer_line: List[bool]
    responses_final_answers: List[str]
    responses_correct: List[bool]
    responses_fail_type: List[str]
    correct_kept_indices: List[int]
    wrong_kept_indices: List[int]
    correct_kept: List[str]
    wrong_kept: List[str]


def _empty_lora_health() -> Dict[str, float]:
    return dict(_EMPTY_LORA_HEALTH)


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
    del gold_rationale_key_paths
    del require_gold_rationale
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
    has_last, ans = extract_final_answer_if_last_line(text)
    if has_last and ans.strip():
        return True, ans.strip()
    boxed = extract_boxed_answer_last(text)
    if boxed.strip():
        return True, boxed.strip()
    return False, ""


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


def split_rollout_candidates_for_training(
    candidates: Sequence[str],
    ground_truth: str,
) -> RolloutCandidateSplit:
    responses_has_final_answer_line: List[bool] = []
    responses_final_answers: List[str] = []
    responses_correct: List[bool] = []
    responses_fail_type: List[str] = []
    correct_kept_indices: List[int] = []
    wrong_kept_indices: List[int] = []
    correct_kept: List[str] = []
    wrong_kept: List[str] = []
    for idx, candidate in enumerate(candidates):
        has_final_answer_line, parsed_last_answer = extract_rollout_scored_answer(candidate)
        parsed_answer = parsed_last_answer if has_final_answer_line else ""
        is_correct = answer_text_matches(parsed_answer, ground_truth)
        responses_has_final_answer_line.append(has_final_answer_line)
        responses_final_answers.append(parsed_answer)
        responses_correct.append(is_correct)
        if is_correct:
            responses_fail_type.append("correct")
        elif not has_final_answer_line:
            responses_fail_type.append("no_final_answer")
        elif not parsed_answer:
            responses_fail_type.append("empty_final_answer")
        else:
            responses_fail_type.append("wrong_answer")
        if is_correct:
            correct_kept_indices.append(idx)
            correct_kept.append(str(candidate))
        else:
            wrong_kept_indices.append(idx)
            wrong_kept.append(str(candidate))
    return RolloutCandidateSplit(
        responses_has_final_answer_line=responses_has_final_answer_line,
        responses_final_answers=responses_final_answers,
        responses_correct=responses_correct,
        responses_fail_type=responses_fail_type,
        correct_kept_indices=correct_kept_indices,
        wrong_kept_indices=wrong_kept_indices,
        correct_kept=correct_kept,
        wrong_kept=wrong_kept,
    )


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


def compute_correct_trajectory_weights(
    correct_trajs: Sequence[RolloutTrajectory],
    mode: str,
    tau: float,
) -> List[float]:
    count = len(correct_trajs)
    if count == 0:
        return []
    if mode == "uniform":
        return [1.0 / count for _ in range(count)]
    if mode != "nll_softmax":
        raise ValueError(f"Unsupported positive weight mode: {mode}")
    nll = torch.tensor([float(t.avg_nll) for t in correct_trajs], dtype=torch.float32)
    logits = float(tau) * nll
    weights = torch.softmax(logits, dim=0)
    return [float(x) for x in weights.tolist()]


def rollout_trajectory_to_json(traj: RolloutTrajectory, *, include_dense: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "response_text": traj.response_text,
        "is_correct": bool(traj.is_correct),
        "fail_type": traj.fail_type,
        "has_final_answer_line": bool(traj.has_final_answer_line),
        "final_answer": traj.final_answer,
        "avg_logprob": float(traj.avg_logprob),
        "avg_nll": float(traj.avg_nll),
        "avg_entropy": float(traj.avg_entropy),
    }
    if include_dense:
        out["token_ids"] = [int(t) for t in traj.token_ids]
        out["hidden_vec"] = [float(v) for v in traj.hidden_vec]
    return out


def load_prompt_candidates_from_file(prompt_file: str) -> List[str]:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt candidate file not found: {prompt_file}")
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            prompts = content.get("prompts", [])
        else:
            prompts = content
        if not isinstance(prompts, list):
            raise ValueError("Prompt JSON must be a list of strings or {'prompts': [...]} format.")
        return [str(p).strip() for p in prompts if str(p).strip()]
    prompts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        prompts.append(text)
    return prompts


def deduplicate_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def build_prompt_pool(args: argparse.Namespace) -> List[str]:
    pool: List[str] = []
    if args.system_prompt.strip():
        pool.append(args.system_prompt.strip())
    for prompt in args.prompt_candidate:
        if prompt.strip():
            pool.append(prompt.strip())
    if args.prompt_candidates_file.strip():
        pool.extend(load_prompt_candidates_from_file(args.prompt_candidates_file.strip()))
    if args.use_default_prompt_candidates:
        pool.extend(DEFAULT_PROMPT_CANDIDATES)
    return deduplicate_keep_order(pool)


def choose_system_prompt(
    prompt_pool: Sequence[str],
    prompt_mode: str,
    prompt_fixed_index: int,
    rng: random.Random,
    explicit_prompt: Optional[str] = None,
) -> str:
    if explicit_prompt is not None and str(explicit_prompt).strip():
        return str(explicit_prompt).strip()
    if prompt_mode == "none" or not prompt_pool:
        return ""
    if prompt_mode == "fixed":
        return prompt_pool[prompt_fixed_index % len(prompt_pool)]
    if prompt_mode == "random":
        return prompt_pool[rng.randrange(len(prompt_pool))]
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def apply_qwen_chat_template(
    tokenizer: object,
    prompt: str,
    enable_thinking: bool,
    system_prompt: str = "",
) -> str:
    global _WARNED_MISSING_CHAT_TEMPLATE
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except ValueError as e:
        if "tokenizer.chat_template is not set" not in str(e):
            raise
        if not _WARNED_MISSING_CHAT_TEMPLATE:
            print(
                "[warn] tokenizer.chat_template is missing; fallback to plain text prompts "
                "for online rollout/training."
            )
            _WARNED_MISSING_CHAT_TEMPLATE = True
        return "\n\n".join(m.get("content", "") for m in messages if m.get("content"))


def wrap_model_with_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    if not targets:
        raise ValueError("--lora-target-modules must list at least one module name.")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    return get_peft_model(model, lora_config)


def ensure_input_require_grads_for_checkpointing(model: Any) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    if not hasattr(model, "get_input_embeddings"):
        return
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        return
    if getattr(model, "_pref_input_require_grads_hook", None) is not None:
        return

    def _make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        elif isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor):
                    item.requires_grad_(True)
        return output

    hook = embeddings.register_forward_hook(_make_inputs_require_grad)
    setattr(model, "_pref_input_require_grads_hook", hook)


def unwrap_model_for_save(model: object) -> object:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _compute_lora_param_health(model: object) -> Dict[str, float]:
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return _empty_lora_health()

    total_numel = 0
    nan_numel = 0
    inf_numel = 0
    abs_sum = 0.0
    abs_max = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" not in name:
                continue
            if param is None:
                continue
            tensor = param.detach()
            if tensor.numel() == 0:
                continue
            total_numel += tensor.numel()
            nan_numel += int(torch.isnan(tensor).sum().item())
            inf_numel += int(torch.isinf(tensor).sum().item())
            abs_tensor = torch.abs(tensor)
            abs_sum += float(abs_tensor.sum().item())
            abs_max = max(abs_max, float(abs_tensor.max().item()))

    if total_numel == 0:
        return _empty_lora_health()
    return {
        "lora_mean_abs": abs_sum / total_numel,
        "lora_max_abs": abs_max,
        "lora_nan_ratio": nan_numel / total_numel,
        "lora_inf_ratio": inf_numel / total_numel,
    }


def online_rollout_completions_flat_vllm(
    args: argparse.Namespace,
    *,
    model: object,
    tokenizer: object,
    device: torch.device,
    prompt_texts: List[str],
    rollout_steps: int,
    total_steps_str: str,
    init_model_path: str,
    vllm_staging_dir: Path,
    hf_updates_so_far: int,
) -> List[str]:
    from vllm import LLM, SamplingParams

    use_lora = bool(getattr(args, "use_lora", False))
    lora_request = None
    if use_lora:
        vllm_staging_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = vllm_staging_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        unwrap_model_for_save(model).save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        ckpt = init_model_path
        try:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("online_lora", 1, str(adapter_dir.resolve()))
        except Exception as e:
            raise RuntimeError(
                "use_lora=true requires vLLM LoRA support and a successful LoRARequest; "
                f"got: {e}"
            ) from e
        print(
            f"[online] vLLM+LoRA rollout_step={rollout_steps}/{total_steps_str} "
            f"base={ckpt} adapter={adapter_dir}",
            flush=True,
        )
        llm_kw: Dict[str, Any] = {
            "model": ckpt,
            "tokenizer": ckpt,
            "trust_remote_code": True,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.vllm_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.rollout_max_model_len,
            "enforce_eager": args.online_vllm_enforce_eager,
            "enable_lora": True,
            "max_lora_rank": args.vllm_max_lora_rank,
            "max_loras": 1,
            "max_cpu_loras": 1,
        }
    elif hf_updates_so_far > 0:
        vllm_staging_dir.mkdir(parents=True, exist_ok=True)
        unwrap_model_for_save(model).save_pretrained(vllm_staging_dir)
        tokenizer.save_pretrained(vllm_staging_dir)
        ckpt = str(vllm_staging_dir)
        print(f"[online] vLLM loading rollout_step={rollout_steps}/{total_steps_str} ckpt={ckpt}", flush=True)
        llm_kw = {
            "model": ckpt,
            "tokenizer": ckpt,
            "trust_remote_code": True,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.vllm_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.rollout_max_model_len,
            "enforce_eager": args.online_vllm_enforce_eager,
        }
    else:
        ckpt = init_model_path
        print(f"[online] vLLM loading rollout_step={rollout_steps}/{total_steps_str} ckpt={ckpt}", flush=True)
        llm_kw = {
            "model": ckpt,
            "tokenizer": ckpt,
            "trust_remote_code": True,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.vllm_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.rollout_max_model_len,
            "enforce_eager": args.online_vllm_enforce_eager,
        }

    model.eval()
    model.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    llm = LLM(**llm_kw)
    sampling_params = SamplingParams(
        n=args.rollout_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed + rollout_steps * 100003,
    )
    gen_kw: Dict[str, Any] = {"use_tqdm": args.online_vllm_use_tqdm}
    if lora_request is not None:
        gen_kw["lora_request"] = lora_request
    outputs = llm.generate(prompt_texts, sampling_params, **gen_kw)
    completion_flat: List[str] = []
    for output in outputs:
        for cand in output.outputs:
            completion_flat.append(cand.text)

    del llm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.to(device)
    print(f"[online] vLLM finished rollout_step={rollout_steps}/{total_steps_str}", flush=True)
    return completion_flat


def online_rollout_completions_flat_hf(
    model: object,
    tokenizer: object,
    device: torch.device,
    prompt_texts: List[str],
    args: argparse.Namespace,
) -> List[str]:
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.rollout_max_model_len,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        expanded_input_ids = input_ids.repeat_interleave(args.rollout_n, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(args.rollout_n, dim=0)
        generated = model.generate(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_lens = expanded_attention_mask.sum(dim=1).tolist()
    out: List[str] = []
    for i, prompt_len in enumerate(prompt_lens):
        completion_ids = generated[i, int(prompt_len) :]
        out.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return out


def build_parser(default_system_prompt: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online privileged OPSD training.")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, default="/path/to/dapo-math-17k.parquet")
    parser.add_argument("--dataset_layout", type=str, default="auto", choices=["auto", "dapo", "math_hf"])
    parser.add_argument("--user_content_suffix", type=str, default="")
    parser.add_argument("--auto_math_hf_user_suffix", type=str2bool, default=True)
    parser.add_argument("--gold_rationale_key", action="append", default=list(DEFAULT_GOLD_RATIONALE_KEY_PATHS))
    parser.add_argument("--require_gold_rationale_for_all_wrong", type=str2bool, default=False)
    parser.add_argument("--model_path", type=str, default="/path/to/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-4b-pref")
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument("--max_source_samples", type=int, default=17000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--rollout_max_model_len", type=int, default=32768)
    parser.add_argument("--prompt_mode", type=str, default="none", choices=["none", "fixed", "random"])
    parser.add_argument("--system_prompt", type=str, default=default_system_prompt)
    parser.add_argument("--prompt_candidate", action="append", default=[])
    parser.add_argument("--prompt_candidates_file", type=str, default="")
    parser.add_argument("--use_default_prompt_candidates", type=str2bool, default=False)
    parser.add_argument("--prompt_fixed_index", type=int, default=0)
    parser.add_argument("--enable_thinking", type=str2bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--lambda_mle", type=float, default=1.0)
    parser.add_argument("--lambda_gt", type=float, default=0.5)
    parser.add_argument("--prompt_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--prompt_smoothing_beta", type=float, default=1.0)
    parser.add_argument("--prompt_weight_gamma", type=float, default=1.0)
    parser.add_argument("--prompt_weight_min", type=float, default=0.1)
    parser.add_argument("--prompt_weight_max", type=float, default=1.0)
    parser.add_argument("--positive_weight_mode", type=str, default="nll_softmax", choices=["nll_softmax", "uniform"])
    parser.add_argument("--positive_weight_tau", type=float, default=1.0)
    parser.add_argument("--hidden_layer_offset", type=int, default=4)
    parser.add_argument("--rollout_feature_micro_batch_size", type=int, default=8)
    parser.add_argument("--use_all_wrong_gt_preference", type=str2bool, default=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--logprob_micro_batch_size", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--online_hard_grad_norm_cap", type=float, default=5.0)
    parser.add_argument("--online_loss_value_cap", type=float, default=20.0)
    parser.add_argument("--online_skip_nonfinite_loss", type=str2bool, default=True)
    parser.add_argument("--online_abort_on_lora_nan", type=str2bool, default=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--rollout_compute_entropy", type=str2bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--hf_data_parallel", type=str2bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--use_lora", type=str2bool, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)
    parser.add_argument("--online_steps", type=int, default=0)
    parser.add_argument("--online-pairs-per-step", type=int, default=16, dest="online_pairs_per_step")
    parser.add_argument("--online_save_every_updates", type=int, default=0)
    parser.add_argument("--online_rollout_backend", type=str, default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--online_vllm_use_tqdm", type=str2bool, default=True)
    parser.add_argument("--online_vllm_enforce_eager", type=str2bool, default=True)
    return parser
