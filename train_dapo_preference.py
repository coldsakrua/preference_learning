#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Online (on-policy) preference training for DAPO math data.

For each rollout step:
  - Use vLLM/HF to sample N responses per prompt (--rollout_n, default 8).
  - If mixed correct/wrong: use all (correct x wrong) pairs with pairwise mean loss.
  - If all correct: use full SFT loss on all sampled responses.
  - If all wrong: skip for now.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise math reasoning assistant. "
    "Solve the problem step by step, then end with exactly one final line in the format: "
    "Answer: <final_answer>."
)

DEFAULT_PROMPT_CANDIDATES = [
    "You are a careful math tutor. Show concise but correct reasoning and finish with: Answer: <final_answer>.",
    "Solve the math problem with rigorous steps. Keep reasoning structured and end with: Answer: <final_answer>.",
    "You are an expert competition-math assistant. Verify key steps and finish with: Answer: <final_answer>.",
    "Reason clearly and avoid arithmetic mistakes. The last line must be: Answer: <final_answer>.",
    "Produce a correct step-by-step solution, then output one final line: Answer: <final_answer>.",
]


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
    sample_id: str


@dataclass
class OnlinePendingObjective:
    """One prompt-level objective queued for immediate rollout-step updates."""

    train_prompt: str
    correct: List[str]
    wrong: List[str]
    objective_type: str  # "pref" or "sft"
    pair_weight: float


@dataclass
class OnlineStepLossStats:
    total_loss: float
    pref_loss: float
    sft_loss: float
    mean_gap: float
    pref_pairs_used: int
    sft_samples_used: int


@dataclass
class RolloutCandidateSplit:
    responses_has_final_answer_line: List[bool]
    responses_final_answers: List[str]
    responses_correct: List[bool]
    correct_kept: List[str]
    wrong_kept: List[str]


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


def iter_dapo_samples(
    parquet_path: str,
    scan_batch_size: int,
    max_source_samples: Optional[int],
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
            sample_id = ""
            if isinstance(extra_obj, dict):
                sample_id = str(extra_obj.get("index", "")).strip()
            if not sample_id:
                sample_id = f"row-{yielded}"
            yield DapoSample(prompt=prompt_text, ground_truth=ground_truth, sample_id=sample_id)
            yielded += 1
            if max_source_samples is not None and yielded >= max_source_samples:
                return


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
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


_ANSWER_LINE = re.compile(r"answer\s*[:\uFF1A]\s*(.+)", flags=re.IGNORECASE)
_ANSWER_LINE_CANONICAL = re.compile(r"^\s*(?:final\s+)?answer\s*[:\uFF1A]\s*(.+?)\s*$", flags=re.IGNORECASE)
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
    # Remove common markdown quote/list prefixes.
    s = re.sub(r"^[>\-\*\#\s]+", "", s)
    # Remove common markdown emphasis wrappers.
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


def extract_final_answer_if_last_line(text: str) -> tuple[bool, str]:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, ""
    answer = parse_answer_from_line(lines[-1])
    if answer is None:
        return False, ""
    return True, answer


def extract_final_answer_robust(text: str) -> str:
    has_last, answer_last = extract_final_answer_if_last_line(text)
    if has_last:
        return answer_last

    if "</think>" in text:
        text = text.split("</think>")[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        answer = parse_answer_from_line(line)
        if answer:
            return answer
        match = _ANSWER_LINE.search(normalize_answer_line_for_parse(line))
        if match:
            fallback = strip_outer_formatting(match.group(1))
            if fallback:
                return fallback

    boxed_matches = _BOXED.findall(text)
    if boxed_matches:
        answer = strip_outer_formatting(boxed_matches[-1])
        if answer:
            return answer

    if lines:
        return strip_outer_formatting(lines[-1])
    return text.strip()


def extract_final_answer(text: str) -> str:
    return extract_final_answer_robust(text)


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


def answers_match(predicted_text: str, ground_truth: str) -> bool:
    predicted = normalize_answer(extract_final_answer_robust(predicted_text))
    target = normalize_answer(ground_truth)
    if predicted and predicted == target:
        return True
    predicted_num = to_number_if_simple(predicted)
    target_num = to_number_if_simple(target)
    if predicted_num is not None and target_num is not None:
        return abs(predicted_num - target_num) <= 1e-6
    return False


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


def extract_normalized_final_answer(text: str) -> str:
    return normalize_answer(extract_final_answer_robust(text))

def candidates_share_same_final_answer(candidates: Sequence[str]) -> bool:
    normalized_answers = [extract_normalized_final_answer(candidate) for candidate in candidates]
    normalized_answers = [answer for answer in normalized_answers if answer]
    if len(normalized_answers) < 2:
        return False
    return len(set(normalized_answers)) == 1

def choose_preference_pair(
    candidates: Sequence[str],
    ground_truth: str,
    require_rejected_final_answer: bool,
    require_chosen_final_answer: bool,
) -> Optional[Dict[str, str]]:
    # Skip pairs with identical final answers (different reasoning but same answer => no preference signal).
    if candidates_share_same_final_answer(candidates):
        return None

    chosen = None
    chosen_final_answer = ""
    rejected = None
    rejected_final_answer = ""
    for candidate in candidates:
        has_final_answer_line, parsed_last_answer = extract_final_answer_if_last_line(candidate)
        parsed_answer = parsed_last_answer if has_final_answer_line else extract_final_answer_robust(candidate)
        is_correct = answer_text_matches(parsed_answer, ground_truth)

        if is_correct:
            if require_chosen_final_answer and not has_final_answer_line:
                continue
            if chosen is None:
                chosen = candidate
                chosen_final_answer = parsed_answer
        else:
            if require_rejected_final_answer and not has_final_answer_line:
                continue
            if rejected is None:
                rejected = candidate
                rejected_final_answer = parsed_last_answer if has_final_answer_line else parsed_answer
    if chosen is None or rejected is None:
        return None
    return {
        "chosen": chosen,
        "rejected": rejected,
        "chosen_final_answer": chosen_final_answer,
        "rejected_final_answer": rejected_final_answer,
    }


def build_preference_jsonl_record(
    sample_id: str,
    prompt: str,
    system_prompt: str,
    ground_truth: str,
    candidates: Sequence[str],
    pair: Dict[str, str],
) -> Dict[str, Any]:
    """JSONL row: chosen/rejected plus raw rollouts (n=rollout_n) and per-rollout correctness."""
    responses_final_answers = [extract_final_answer_robust(c) for c in candidates]
    responses_has_final_answer_line = [extract_final_answer_if_last_line(c)[0] for c in candidates]
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "ground_truth": ground_truth,
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
        "chosen_final_answer": pair.get("chosen_final_answer", ""),
        "rejected_final_answer": pair.get("rejected_final_answer", ""),
        "responses": [str(c) for c in candidates],
        "responses_final_answers": responses_final_answers,
        "responses_has_final_answer_line": responses_has_final_answer_line,
        "responses_correct": [answers_match(c, ground_truth) for c in candidates],
    }


def compute_entropy_rarity_weight(
    n_correct: int,
    n_total: int,
    rarity_floor: float,
    eps: float,
) -> tuple[float, float, float, float]:
    if n_total <= 0:
        return 0.0, 0.0, rarity_floor, 0.0
    r = n_correct / n_total
    entropy = -(r * math.log(r + eps) + (1.0 - r) * math.log(1.0 - r + eps))
    rarity_bonus = max(1.0 - r, rarity_floor)
    return r, entropy, rarity_bonus, entropy * rarity_bonus


def split_rollout_candidates_for_training(
    candidates: Sequence[str],
    ground_truth: str,
    require_rejected_final_answer: bool,
    require_chosen_final_answer: bool,
) -> RolloutCandidateSplit:
    responses_has_final_answer_line: List[bool] = []
    responses_final_answers: List[str] = []
    responses_correct: List[bool] = []
    correct_kept: List[str] = []
    wrong_kept: List[str] = []
    for candidate in candidates:
        has_final_answer_line, parsed_last_answer = extract_final_answer_if_last_line(candidate)
        parsed_answer = parsed_last_answer if has_final_answer_line else extract_final_answer_robust(candidate)
        is_correct = answer_text_matches(parsed_answer, ground_truth)
        responses_has_final_answer_line.append(has_final_answer_line)
        responses_final_answers.append(parsed_answer)
        responses_correct.append(is_correct)

        if is_correct:
            if require_chosen_final_answer and not has_final_answer_line:
                continue
            correct_kept.append(str(candidate))
        else:
            if require_rejected_final_answer and not has_final_answer_line:
                continue
            wrong_kept.append(str(candidate))
    return RolloutCandidateSplit(
        responses_has_final_answer_line=responses_has_final_answer_line,
        responses_final_answers=responses_final_answers,
        responses_correct=responses_correct,
        correct_kept=correct_kept,
        wrong_kept=wrong_kept,
    )


def build_online_objective_jsonl_record(
    sample_id: str,
    prompt: str,
    system_prompt: str,
    ground_truth: str,
    candidates: Sequence[str],
    split: RolloutCandidateSplit,
    objective_type: str,
    r: float,
    entropy: float,
    rarity_bonus: float,
    weight: float,
) -> Dict[str, Any]:
    chosen_preview = str(split.correct_kept[0]) if split.correct_kept else ""
    rejected_preview = str(split.wrong_kept[0]) if split.wrong_kept else ""
    n_correct_total = int(sum(1 for x in split.responses_correct if x))
    n_total = len(candidates)
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "ground_truth": ground_truth,
        "chosen": chosen_preview,
        "rejected": rejected_preview,
        "responses": [str(c) for c in candidates],
        "responses_final_answers": [str(a) for a in split.responses_final_answers],
        "responses_has_final_answer_line": [bool(v) for v in split.responses_has_final_answer_line],
        "responses_correct": [bool(v) for v in split.responses_correct],
        "correct_kept": [str(c) for c in split.correct_kept],
        "wrong_kept": [str(w) for w in split.wrong_kept],
        "n_total": n_total,
        "n_correct_total": n_correct_total,
        "n_wrong_total": n_total - n_correct_total,
        "n_correct_kept": len(split.correct_kept),
        "n_wrong_kept": len(split.wrong_kept),
        "r": float(r),
        "entropy": float(entropy),
        "rarity_bonus": float(rarity_bonus),
        "weight": float(weight),
        "objective_type": objective_type,
    }


def generate_preference_pairs(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    output_path = Path(args.preference_pairs_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.rollout_model_path, trust_remote_code=True)

    llm = LLM(
        model=args.rollout_model_path,
        tokenizer=args.rollout_model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.rollout_max_model_len,
    )
    sampling_params = SamplingParams(
        n=args.rollout_n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260410)
    if args.prompt_mode != "none" and not prompt_pool:
        raise ValueError(
            "Prompt mode is not 'none' but prompt pool is empty. "
            "Set --system_prompt or --prompt_candidate/--prompt_candidates_file, "
            "or use --use_default_prompt_candidates true."
        )
    print(f"[generate] prompt_mode={args.prompt_mode}, prompt_pool_size={len(prompt_pool)}")

    processed = 0
    saved_pairs = 0
    same_answer_skipped = 0
    buffer: List[DapoSample] = []

    source_iter = iter_dapo_samples(
        parquet_path=args.dataset_path,
        scan_batch_size=args.scan_batch_size,
        max_source_samples=args.max_source_samples,
    )

    with output_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=args.target_pairs, desc="collect preference pairs", dynamic_ncols=True)
        for sample in source_iter:
            buffer.append(sample)
            processed += 1

            should_flush = len(buffer) >= args.rollout_batch_size
            if not should_flush:
                if args.target_pairs is not None and saved_pairs >= args.target_pairs:
                    break
                continue

            system_prompts = [
                choose_system_prompt(
                    prompt_pool=prompt_pool,
                    prompt_mode=args.prompt_mode,
                    prompt_fixed_index=args.prompt_fixed_index,
                    rng=prompt_rng,
                )
                for _ in buffer
            ]
            prompts = [
                apply_qwen_chat_template(
                    tokenizer,
                    s.prompt,
                    enable_thinking=args.enable_thinking,
                    system_prompt=sp,
                )
                for s, sp in zip(buffer, system_prompts)
            ]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            for sample_obj, output, system_prompt in zip(buffer, outputs, system_prompts):
                candidates = [candidate.text for candidate in output.outputs]
                pair = choose_preference_pair(
                    candidates,
                    sample_obj.ground_truth,
                    require_rejected_final_answer=args.sample_rejected_requires_final_answer,
                    require_chosen_final_answer=args.sample_chosen_requires_final_answer,
                )
                if pair is None:
                    if candidates_share_same_final_answer(candidates):
                        same_answer_skipped += 1
                    continue
                record = build_preference_jsonl_record(
                    sample_obj.sample_id,
                    sample_obj.prompt,
                    system_prompt,
                    sample_obj.ground_truth,
                    candidates,
                    pair,
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_pairs += 1
                pbar.update(1)
                if args.target_pairs is not None and saved_pairs >= args.target_pairs:
                    break

            buffer = []
            if args.target_pairs is not None and saved_pairs >= args.target_pairs:
                break

        if buffer and (args.target_pairs is None or saved_pairs < args.target_pairs):
            system_prompts = [
                choose_system_prompt(
                    prompt_pool=prompt_pool,
                    prompt_mode=args.prompt_mode,
                    prompt_fixed_index=args.prompt_fixed_index,
                    rng=prompt_rng,
                )
                for _ in buffer
            ]
            prompts = [
                apply_qwen_chat_template(
                    tokenizer,
                    s.prompt,
                    enable_thinking=args.enable_thinking,
                    system_prompt=sp,
                )
                for s, sp in zip(buffer, system_prompts)
            ]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            for sample_obj, output, system_prompt in zip(buffer, outputs, system_prompts):
                candidates = [candidate.text for candidate in output.outputs]
                pair = choose_preference_pair(
                    candidates,
                    sample_obj.ground_truth,
                    require_rejected_final_answer=args.sample_rejected_requires_final_answer,
                    require_chosen_final_answer=args.sample_chosen_requires_final_answer,
                )
                if pair is None:
                    if candidates_share_same_final_answer(candidates):
                        same_answer_skipped += 1
                    continue
                record = build_preference_jsonl_record(
                    sample_obj.sample_id,
                    sample_obj.prompt,
                    system_prompt,
                    sample_obj.ground_truth,
                    candidates,
                    pair,
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_pairs += 1
                pbar.update(1)
                if args.target_pairs is not None and saved_pairs >= args.target_pairs:
                    break
        pbar.close()

    kept_ratio = (saved_pairs / processed) if processed else 0.0
    print(
        f"[generate] processed={processed}, saved_pairs={saved_pairs}, "
        f"same_answer_skipped={same_answer_skipped}, keep_ratio={kept_ratio:.4f}, output={output_path}"
    )
    return saved_pairs


class PreferenceDataset(Dataset):
    def __init__(self, jsonl_path: str, max_pairs: Optional[int] = None) -> None:
        self.items: List[Dict[str, str]] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "prompt" not in item or "chosen" not in item or "rejected" not in item:
                    continue
                self.items.append(item)
                if max_pairs is not None and len(self.items) >= max_pairs:
                    break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


def collate_batch(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        "sample_id": [str(item.get("sample_id", "")).strip() for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "system_prompt": [str(item.get("system_prompt", "")).strip() for item in batch],
        "chosen": [item["chosen"] for item in batch],
        "rejected": [item["rejected"] for item in batch],
    }


def _compute_sequence_logps_batch(
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
    length_average: bool,
) -> torch.Tensor:
    full_texts = [prompt + completion for prompt, completion in zip(prompt_texts, completion_texts)]
    prompt_ids = tokenizer(
        list(prompt_texts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    encoded = tokenizer(
        full_texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    labels = torch.full_like(input_ids, -100)
    seq_lens = attention_mask.sum(dim=1).tolist()

    for i, seq_len in enumerate(seq_lens):
        prompt_len = len(prompt_ids[i])
        start = min(prompt_len, seq_len)
        if start < seq_len:
            labels[i, start:seq_len] = input_ids[i, start:seq_len]

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    valid_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    seq_logps = (token_logps * valid_mask).sum(dim=-1)

    if length_average:
        token_counts = valid_mask.sum(dim=-1).clamp_min(1)
        seq_logps = seq_logps / token_counts
    return seq_logps


def compute_sequence_logps(
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
    length_average: bool,
) -> torch.Tensor:
    prompts = list(prompt_texts)
    completions = list(completion_texts)
    if len(prompts) == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    return _compute_sequence_logps_batch(
        model,
        tokenizer,
        prompts,
        completions,
        max_length,
        device,
        length_average,
    )


def resolve_train_sample_log_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if str(args.train_sample_log_path).strip():
        return Path(args.train_sample_log_path)
    return output_dir / "train_sampled_pairs.jsonl"


def train_with_preference_loss(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    dataset = PreferenceDataset(args.preference_pairs_path, max_pairs=args.max_train_pairs)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No preference pairs found in {args.preference_pairs_path}. "
            "Run generation first or check pair filtering."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.train_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260411)
    if args.prompt_mode != "none" and len(prompt_pool) == 0:
        print(
            "[train] prompt_mode is not 'none' but prompt pool is empty; "
            "falling back to no system prompt."
        )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {args.torch_dtype}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[args.torch_dtype],
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.train_model_path, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_train_samples = bool(args.record_train_samples)
    train_sample_log_path = resolve_train_sample_log_path(args, output_dir)
    train_sample_log_path.parent.mkdir(parents=True, exist_ok=True)
    train_sample_log_fout = None
    logged_records = 0
    max_log_records = args.train_sample_log_max_records

    global_step = 0
    running_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    try:
        if record_train_samples:
            train_sample_log_fout = train_sample_log_path.open("w", encoding="utf-8")

        for epoch in range(args.num_epochs):
            epoch_iterator = tqdm(
                dataloader,
                desc=f"train epoch {epoch + 1}/{args.num_epochs}",
                dynamic_ncols=True,
            )
            for step, batch in enumerate(epoch_iterator, start=1):
                sample_ids = batch["sample_id"]
                raw_prompts = batch["prompt"]
                raw_system_prompts = batch["system_prompt"]
                chosen = batch["chosen"]
                rejected = batch["rejected"]

                if record_train_samples and train_sample_log_fout is not None:
                    if max_log_records is None or logged_records < max_log_records:
                        for idx, (sample_id, raw_prompt, raw_system_prompt, chosen_text, rejected_text) in enumerate(
                            zip(sample_ids, raw_prompts, raw_system_prompts, chosen, rejected)
                        ):
                            if max_log_records is not None and logged_records >= max_log_records:
                                break
                            record = {
                                "epoch": epoch + 1,
                                "step_in_epoch": step,
                                "global_optimizer_step": global_step,
                                "sample_index_in_batch": idx,
                                "sample_id": sample_id,
                                "prompt": raw_prompt,
                                "system_prompt": raw_system_prompt,
                                "chosen": chosen_text,
                                "rejected": rejected_text,
                            }
                            train_sample_log_fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            logged_records += 1

                prompt_texts = [
                    apply_qwen_chat_template(
                        tokenizer,
                        prompt,
                        enable_thinking=args.enable_thinking,
                        system_prompt=choose_system_prompt(
                            prompt_pool=prompt_pool,
                            prompt_mode=args.prompt_mode,
                            prompt_fixed_index=args.prompt_fixed_index,
                            rng=prompt_rng,
                            explicit_prompt=record_system_prompt,
                        ),
                    )
                    for prompt, record_system_prompt in zip(raw_prompts, raw_system_prompts)
                ]

                all_prompts = prompt_texts + prompt_texts
                all_completions = chosen + rejected
                all_logps = compute_sequence_logps(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=all_prompts,
                    completion_texts=all_completions,
                    max_length=args.max_length,
                    device=device,
                    length_average=args.length_average,
                )

                batch_size = len(prompt_texts)
                chosen_logps = all_logps[:batch_size]
                rejected_logps = all_logps[batch_size:]
                preference_gap = chosen_logps - rejected_logps
                pref_loss = -F.logsigmoid(args.beta * preference_gap).mean()
                chosen_ce_loss = (-chosen_logps).mean()
                loss = pref_loss + args.chosen_ce_weight * chosen_ce_loss

                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                running_loss += loss.item()

                should_update = (step % args.gradient_accumulation_steps == 0) or (step == len(dataloader))
                if should_update:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = running_loss / max(1, args.gradient_accumulation_steps)
                    epoch_iterator.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        pref=f"{pref_loss.item():.4f}",
                        ce=f"{chosen_ce_loss.item():.4f}",
                        gap=f"{preference_gap.mean().item():.4f}",
                        step=global_step,
                    )
                    running_loss = 0.0

            if args.save_every_epoch:
                epoch_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(epoch_dir)
                tokenizer.save_pretrained(epoch_dir)
                print(f"[train] saved checkpoint to {epoch_dir}")
    finally:
        if train_sample_log_fout is not None:
            train_sample_log_fout.close()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] finished. final model saved to {final_dir}")
    if record_train_samples:
        suffix = ""
        if max_log_records is not None:
            suffix = f" (max={max_log_records})"
        print(
            f"[train] sampled training records saved: {logged_records} -> "
            f"{train_sample_log_path}{suffix}"
        )


def wrap_model_with_lora(model: Any, args: argparse.Namespace) -> Any:
    """Attach LoRA adapters (PEFT). Base weights stay frozen; only adapters train."""
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
    """
    Make input embeddings require grad when gradient checkpointing is enabled.
    This is required for PEFT/LoRA; otherwise autograd can see no grad_fn.
    """
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


def _online_rollout_completions_flat_vllm(
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
        model.save_pretrained(adapter_dir)
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
        model.save_pretrained(vllm_staging_dir)
        tokenizer.save_pretrained(vllm_staging_dir)
        ckpt = str(vllm_staging_dir)
        print(
            f"[online] vLLM loading rollout_step={rollout_steps}/{total_steps_str} ckpt={ckpt}",
            flush=True,
        )
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
        print(
            f"[online] vLLM loading rollout_step={rollout_steps}/{total_steps_str} ckpt={ckpt}",
            flush=True,
        )
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
        max_tokens=args.max_new_tokens,
        seed=args.seed + rollout_steps * 100003,
    )
    gen_kw: Dict[str, Any] = {"use_tqdm": args.online_vllm_use_tqdm}
    if lora_request is not None:
        gen_kw["lora_request"] = lora_request
    outputs = llm.generate(
        prompt_texts,
        sampling_params,
        **gen_kw,
    )
    completion_flat: List[str] = []
    for output in outputs:
        for cand in output.outputs:
            completion_flat.append(cand.text)

    del llm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.to(device)
    print(
        f"[online] vLLM finished rollout_step={rollout_steps}/{total_steps_str}",
        flush=True,
    )
    return completion_flat


def _online_rollout_completions_flat_hf(
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


@dataclass
class PairTruncationStats:
    total_pairs: int
    kept_pairs: int
    dropped_pairs: int
    dropped_prompt_too_long: int
    dropped_chosen_too_long: int
    dropped_rejected_too_long: int


def filter_pairs_without_truncation(
    tokenizer: object,
    train_prompts: Sequence[str],
    chosen: Sequence[str],
    rejected: Sequence[str],
    max_length: int,
) -> tuple[List[str], List[str], List[str], PairTruncationStats]:
    """Drop pairs that would be truncated by max_length in log-prob computation."""
    total = len(train_prompts)
    if total == 0:
        stats = PairTruncationStats(
            total_pairs=0,
            kept_pairs=0,
            dropped_pairs=0,
            dropped_prompt_too_long=0,
            dropped_chosen_too_long=0,
            dropped_rejected_too_long=0,
        )
        return [], [], [], stats

    prompt_ids = tokenizer(
        list(train_prompts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    chosen_ids = tokenizer(
        list(chosen),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    rejected_ids = tokenizer(
        list(rejected),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]

    keep_prompts: List[str] = []
    keep_chosen: List[str] = []
    keep_rejected: List[str] = []
    dropped_prompt_too_long = 0
    dropped_chosen_too_long = 0
    dropped_rejected_too_long = 0

    for prompt, ch, rj, p_ids, ch_ids, rj_ids in zip(
        train_prompts, chosen, rejected, prompt_ids, chosen_ids, rejected_ids
    ):
        prompt_len = len(p_ids)
        budget = max_length - prompt_len
        if budget <= 0:
            dropped_prompt_too_long += 1
            continue
        chosen_ok = len(ch_ids) <= budget
        rejected_ok = len(rj_ids) <= budget
        if not chosen_ok:
            dropped_chosen_too_long += 1
        if not rejected_ok:
            dropped_rejected_too_long += 1
        if not chosen_ok or not rejected_ok:
            continue
        keep_prompts.append(str(prompt))
        keep_chosen.append(str(ch))
        keep_rejected.append(str(rj))

    kept = len(keep_prompts)
    dropped = total - kept
    stats = PairTruncationStats(
        total_pairs=total,
        kept_pairs=kept,
        dropped_pairs=dropped,
        dropped_prompt_too_long=dropped_prompt_too_long,
        dropped_chosen_too_long=dropped_chosen_too_long,
        dropped_rejected_too_long=dropped_rejected_too_long,
    )
    return keep_prompts, keep_chosen, keep_rejected, stats


@dataclass
class SftTruncationStats:
    total_samples: int
    kept_samples: int
    dropped_samples: int
    dropped_prompt_too_long: int
    dropped_completion_too_long: int


def filter_weighted_pairs_without_truncation(
    tokenizer: object,
    train_prompts: Sequence[str],
    chosen: Sequence[str],
    rejected: Sequence[str],
    weights: Sequence[float],
    max_length: int,
) -> tuple[List[str], List[str], List[str], List[float], PairTruncationStats]:
    total = len(train_prompts)
    if total == 0:
        stats = PairTruncationStats(
            total_pairs=0,
            kept_pairs=0,
            dropped_pairs=0,
            dropped_prompt_too_long=0,
            dropped_chosen_too_long=0,
            dropped_rejected_too_long=0,
        )
        return [], [], [], [], stats

    prompt_ids = tokenizer(
        list(train_prompts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    chosen_ids = tokenizer(
        list(chosen),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    rejected_ids = tokenizer(
        list(rejected),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    keep_prompts: List[str] = []
    keep_chosen: List[str] = []
    keep_rejected: List[str] = []
    keep_weights: List[float] = []
    dropped_prompt_too_long = 0
    dropped_chosen_too_long = 0
    dropped_rejected_too_long = 0
    for prompt, ch, rj, w, p_ids, ch_ids, rj_ids in zip(
        train_prompts, chosen, rejected, weights, prompt_ids, chosen_ids, rejected_ids
    ):
        budget = max_length - len(p_ids)
        if budget <= 0:
            dropped_prompt_too_long += 1
            continue
        chosen_ok = len(ch_ids) <= budget
        rejected_ok = len(rj_ids) <= budget
        if not chosen_ok:
            dropped_chosen_too_long += 1
        if not rejected_ok:
            dropped_rejected_too_long += 1
        if not chosen_ok or not rejected_ok:
            continue
        keep_prompts.append(str(prompt))
        keep_chosen.append(str(ch))
        keep_rejected.append(str(rj))
        keep_weights.append(float(w))
    kept = len(keep_prompts)
    stats = PairTruncationStats(
        total_pairs=total,
        kept_pairs=kept,
        dropped_pairs=total - kept,
        dropped_prompt_too_long=dropped_prompt_too_long,
        dropped_chosen_too_long=dropped_chosen_too_long,
        dropped_rejected_too_long=dropped_rejected_too_long,
    )
    return keep_prompts, keep_chosen, keep_rejected, keep_weights, stats


def filter_weighted_sft_without_truncation(
    tokenizer: object,
    train_prompts: Sequence[str],
    completions: Sequence[str],
    weights: Sequence[float],
    max_length: int,
) -> tuple[List[str], List[str], List[float], SftTruncationStats]:
    total = len(train_prompts)
    if total == 0:
        stats = SftTruncationStats(
            total_samples=0,
            kept_samples=0,
            dropped_samples=0,
            dropped_prompt_too_long=0,
            dropped_completion_too_long=0,
        )
        return [], [], [], stats

    prompt_ids = tokenizer(
        list(train_prompts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    completion_ids = tokenizer(
        list(completions),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]

    keep_prompts: List[str] = []
    keep_completions: List[str] = []
    keep_weights: List[float] = []
    dropped_prompt_too_long = 0
    dropped_completion_too_long = 0
    for prompt, completion, w, p_ids, c_ids in zip(
        train_prompts, completions, weights, prompt_ids, completion_ids
    ):
        budget = max_length - len(p_ids)
        if budget <= 0:
            dropped_prompt_too_long += 1
            continue
        if len(c_ids) > budget:
            dropped_completion_too_long += 1
            continue
        keep_prompts.append(str(prompt))
        keep_completions.append(str(completion))
        keep_weights.append(float(w))

    kept = len(keep_prompts)
    stats = SftTruncationStats(
        total_samples=total,
        kept_samples=kept,
        dropped_samples=total - kept,
        dropped_prompt_too_long=dropped_prompt_too_long,
        dropped_completion_too_long=dropped_completion_too_long,
    )
    return keep_prompts, keep_completions, keep_weights, stats


def _online_run_preference_optimizer_step(
    model: object,
    tokenizer: object,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
    pref_train_prompts: List[str],
    pref_chosen: List[str],
    pref_rejected: List[str],
    pref_weights: List[float],
    sft_train_prompts: List[str],
    sft_completions: List[str],
    sft_weights: List[float],
) -> OnlineStepLossStats:
    """Single optimizer.step() on weighted preference and SFT samples."""
    pref_batch = len(pref_train_prompts)
    sft_batch = len(sft_train_prompts)
    total_weight = float(sum(pref_weights) + sum(sft_weights))
    if total_weight <= 0:
        return OnlineStepLossStats(
            total_loss=0.0,
            pref_loss=0.0,
            sft_loss=0.0,
            mean_gap=0.0,
            pref_pairs_used=0,
            sft_samples_used=0,
        )
    mb_pref = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(pref_batch, 1)
    mb_sft = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(sft_batch, 1)

    optimizer.zero_grad(set_to_none=True)
    pref_loss_weighted_sum = 0.0
    sft_loss_weighted_sum = 0.0
    gap_weighted_sum = 0.0
    pref_weight_sum = 0.0

    if pref_batch > 0:
        for start in range(0, pref_batch, mb_pref):
            end = min(start + mb_pref, pref_batch)
            tp = pref_train_prompts[start:end]
            ch = pref_chosen[start:end]
            rj = pref_rejected[start:end]
            w = torch.tensor(pref_weights[start:end], device=device, dtype=torch.float32)
            chosen_logps = _compute_sequence_logps_batch(
                model,
                tokenizer,
                tp,
                ch,
                args.max_length,
                device,
                args.length_average,
            )
            rejected_logps = _compute_sequence_logps_batch(
                model,
                tokenizer,
                tp,
                rj,
                args.max_length,
                device,
                args.length_average,
            )
            preference_gap = chosen_logps - rejected_logps
            pref_loss_vec = -F.logsigmoid(args.beta * preference_gap)
            loss_chunk = (pref_loss_vec * w).sum() / total_weight
            if not loss_chunk.requires_grad:
                raise RuntimeError(
                    "Online preference loss has no grad_fn. "
                    "If use_lora=true with gradient_checkpointing=true, "
                    "ensure input grads are enabled for checkpointing."
                )
            loss_chunk.backward()
            pref_loss_weighted_sum += (pref_loss_vec * w).sum().item()
            gap_weighted_sum += (preference_gap * w).sum().item()
            pref_weight_sum += w.sum().item()

    if sft_batch > 0:
        for start in range(0, sft_batch, mb_sft):
            end = min(start + mb_sft, sft_batch)
            tp = sft_train_prompts[start:end]
            cp = sft_completions[start:end]
            w = torch.tensor(sft_weights[start:end], device=device, dtype=torch.float32)
            logps = _compute_sequence_logps_batch(
                model,
                tokenizer,
                tp,
                cp,
                args.max_length,
                device,
                args.length_average,
            )
            sft_loss_vec = -logps
            loss_chunk = (sft_loss_vec * w).sum() / total_weight
            if not loss_chunk.requires_grad:
                raise RuntimeError(
                    "Online SFT loss has no grad_fn. "
                    "If use_lora=true with gradient_checkpointing=true, "
                    "ensure input grads are enabled for checkpointing."
                )
            loss_chunk.backward()
            sft_loss_weighted_sum += (sft_loss_vec * w).sum().item()

    if args.max_grad_norm > 0:
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
    optimizer.step()

    mean_gap = gap_weighted_sum / pref_weight_sum if pref_weight_sum > 0 else 0.0
    pref_loss = pref_loss_weighted_sum / pref_weight_sum if pref_weight_sum > 0 else 0.0
    sft_weight_sum = float(sum(sft_weights))
    sft_loss = sft_loss_weighted_sum / sft_weight_sum if sft_weight_sum > 0 else 0.0
    total_loss = (pref_loss_weighted_sum + sft_loss_weighted_sum) / total_weight
    return OnlineStepLossStats(
        total_loss=total_loss,
        pref_loss=pref_loss,
        sft_loss=sft_loss,
        mean_gap=mean_gap,
        pref_pairs_used=pref_batch,
        sft_samples_used=sft_batch,
    )


def run_online_preference_training(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    online_pairs_path = output_root / "online_pairs.jsonl"

    model_path = args.online_init_model_path.strip()
    if not model_path:
        model_path = args.train_model_path.strip() or args.rollout_model_path.strip()
    if not model_path:
        raise ValueError("Online mode requires a valid initial model path.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batched decoder-only generation expects left padding so the last token is real text.
    tokenizer.padding_side = "left"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {args.torch_dtype}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[args.torch_dtype],
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if args.use_lora:
        if args.online_rollout_backend == "vllm" and args.lora_r > args.vllm_max_lora_rank:
            raise ValueError(
                "For vLLM LoRA rollout, --lora-r must be <= --vllm-max-lora-rank "
                f"(got lora_r={args.lora_r}, vllm_max_lora_rank={args.vllm_max_lora_rank})."
            )
        model = wrap_model_with_lora(model, args)
        model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if args.use_lora:
            ensure_input_require_grads_for_checkpointing(model)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260412)

    source_iter = iter_dapo_samples(
        parquet_path=args.dataset_path,
        scan_batch_size=args.scan_batch_size,
        max_source_samples=args.max_source_samples,
    )

    updates = 0
    rollout_steps = 0
    scanned = 0
    kept_pref_pairs = 0
    kept_sft_samples = 0
    skipped_all_wrong = 0
    skipped_mixed_after_filter = 0
    logged_pref_objectives = 0
    logged_sft_objectives = 0
    buffer: List[DapoSample] = []
    k = args.online_pairs_per_step

    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"
    print(
        f"[online] rollout_backend={args.online_rollout_backend}, "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"({args.rollout_n} samples per prompt via n={args.rollout_n}), "
        f"online_pairs_per_step={k} (chunk size within one rollout step; no cross-step cache), "
        f"online_steps={total_steps_str}, max_source_samples={args.max_source_samples}, "
        f"vllm_enforce_eager={args.online_vllm_enforce_eager}, "
        f"rarity_floor={args.pref_weight_rarity_floor}"
    )
    if args.online_rollout_backend == "vllm" and device.type != "cuda":
        raise RuntimeError("online_rollout_backend=vllm requires a CUDA device.")

    # Persist sampled objectives immediately so online_pairs.jsonl is visible while the job runs.
    with online_pairs_path.open("w", encoding="utf-8", buffering=1) as fout:
        for sample in source_iter:
            buffer.append(sample)
            scanned += 1
            if len(buffer) < args.rollout_batch_size:
                continue

            rollout_steps += 1
            system_prompts = [
                choose_system_prompt(
                    prompt_pool=prompt_pool,
                    prompt_mode=args.prompt_mode,
                    prompt_fixed_index=args.prompt_fixed_index,
                    rng=prompt_rng,
                )
                for _ in buffer
            ]
            prompt_texts = [
                apply_qwen_chat_template(
                    tokenizer,
                    s.prompt,
                    enable_thinking=args.enable_thinking,
                    system_prompt=sp,
                )
                for s, sp in zip(buffer, system_prompts)
            ]

            vllm_staging_dir = output_root / "vllm_rollout_ckpt"
            if args.online_rollout_backend == "vllm":
                completion_flat = _online_rollout_completions_flat_vllm(
                    args,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_texts=prompt_texts,
                    rollout_steps=rollout_steps,
                    total_steps_str=total_steps_str,
                    init_model_path=model_path,
                    vllm_staging_dir=vllm_staging_dir,
                    hf_updates_so_far=updates,
                )
            else:
                with torch.no_grad():
                    completion_flat = _online_rollout_completions_flat_hf(
                        model, tokenizer, device, prompt_texts, args
                    )

            model.train()

            rollout_objectives: List[OnlinePendingObjective] = []
            pref_objectives_in_rollout = 0
            sft_objectives_in_rollout = 0
            skipped_all_wrong_in_rollout = 0
            skipped_mixed_after_filter_in_rollout = 0
            for idx, sample_obj in enumerate(buffer):
                start = idx * args.rollout_n
                end = start + args.rollout_n
                candidates = completion_flat[start:end]
                if len(candidates) != args.rollout_n:
                    raise RuntimeError(
                        f"Rollout candidate count mismatch at sample {idx}: "
                        f"expected {args.rollout_n}, got {len(candidates)}"
                    )

                split = split_rollout_candidates_for_training(
                    candidates,
                    sample_obj.ground_truth,
                    require_rejected_final_answer=args.sample_rejected_requires_final_answer,
                    require_chosen_final_answer=args.sample_chosen_requires_final_answer,
                )
                n_total = len(candidates)
                n_correct_total = sum(1 for x in split.responses_correct if x)
                r, entropy, rarity_bonus, weight = compute_entropy_rarity_weight(
                    n_correct=n_correct_total,
                    n_total=n_total,
                    rarity_floor=args.pref_weight_rarity_floor,
                    eps=args.pref_weight_eps,
                )

                objective_type = "skip"
                if n_correct_total == 0:
                    skipped_all_wrong += 1
                    skipped_all_wrong_in_rollout += 1
                elif n_correct_total == n_total:
                    if split.correct_kept:
                        objective_type = "sft"
                        rollout_objectives.append(
                            OnlinePendingObjective(
                                train_prompt=prompt_texts[idx],
                                correct=split.correct_kept,
                                wrong=[],
                                objective_type=objective_type,
                                pair_weight=args.full_correct_sft_weight,
                            )
                        )
                        sft_objectives_in_rollout += 1
                        logged_sft_objectives += 1
                    else:
                        skipped_mixed_after_filter += 1
                        skipped_mixed_after_filter_in_rollout += 1
                elif split.correct_kept and split.wrong_kept:
                    objective_type = "pref"
                    rollout_objectives.append(
                        OnlinePendingObjective(
                            train_prompt=prompt_texts[idx],
                            correct=split.correct_kept,
                            wrong=split.wrong_kept,
                            objective_type=objective_type,
                            pair_weight=weight,
                        )
                    )
                    pref_objectives_in_rollout += 1
                    logged_pref_objectives += 1
                else:
                    skipped_mixed_after_filter += 1
                    skipped_mixed_after_filter_in_rollout += 1

                if objective_type != "skip":
                    record = build_online_objective_jsonl_record(
                        sample_id=sample_obj.sample_id,
                        prompt=sample_obj.prompt,
                        system_prompt=system_prompts[idx],
                        ground_truth=sample_obj.ground_truth,
                        candidates=candidates,
                        split=split,
                        objective_type=objective_type,
                        r=r,
                        entropy=entropy,
                        rarity_bonus=rarity_bonus,
                        weight=(args.full_correct_sft_weight if objective_type == "sft" else weight),
                    )
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if rollout_objectives:
                fout.flush()

            print(
                f"[online] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"pref_objectives_in_rollout={pref_objectives_in_rollout} "
                f"sft_objectives_in_rollout={sft_objectives_in_rollout} "
                f"skipped_all_wrong_in_rollout={skipped_all_wrong_in_rollout} "
                f"skipped_after_filter_in_rollout={skipped_mixed_after_filter_in_rollout} "
                f"objectives_ready_for_update={len(rollout_objectives)}"
            )

            updates_in_rollout = 0
            consumed_pref_pairs_in_rollout = 0
            consumed_sft_samples_in_rollout = 0
            dropped_pref_pairs_by_truncation_in_rollout = 0
            dropped_sft_by_truncation_in_rollout = 0
            if rollout_objectives:
                for chunk_start in range(0, len(rollout_objectives), k):
                    chunk = rollout_objectives[chunk_start : chunk_start + k]
                    pref_train_prompts_raw: List[str] = []
                    pref_chosen_raw: List[str] = []
                    pref_rejected_raw: List[str] = []
                    pref_weights_raw: List[float] = []
                    sft_train_prompts_raw: List[str] = []
                    sft_completions_raw: List[str] = []
                    sft_weights_raw: List[float] = []

                    for objective in chunk:
                        if objective.objective_type == "pref":
                            pair_count = len(objective.correct) * len(objective.wrong)
                            if pair_count <= 0:
                                continue
                            pair_weight = objective.pair_weight / pair_count
                            for c in objective.correct:
                                for w in objective.wrong:
                                    pref_train_prompts_raw.append(objective.train_prompt)
                                    pref_chosen_raw.append(c)
                                    pref_rejected_raw.append(w)
                                    pref_weights_raw.append(pair_weight)
                        elif objective.objective_type == "sft":
                            sample_count = len(objective.correct)
                            if sample_count <= 0:
                                continue
                            sample_weight = objective.pair_weight / sample_count
                            for c in objective.correct:
                                sft_train_prompts_raw.append(objective.train_prompt)
                                sft_completions_raw.append(c)
                                sft_weights_raw.append(sample_weight)

                    (
                        pref_train_prompts,
                        pref_chosen,
                        pref_rejected,
                        pref_weights,
                        pref_trunc_stats,
                    ) = filter_weighted_pairs_without_truncation(
                        tokenizer=tokenizer,
                        train_prompts=pref_train_prompts_raw,
                        chosen=pref_chosen_raw,
                        rejected=pref_rejected_raw,
                        weights=pref_weights_raw,
                        max_length=args.max_length,
                    )
                    dropped_pref_pairs_by_truncation_in_rollout += pref_trunc_stats.dropped_pairs
                    if pref_trunc_stats.dropped_pairs > 0:
                        print(
                            f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                            f"truncation_filter_pref dropped_pairs={pref_trunc_stats.dropped_pairs}/{pref_trunc_stats.total_pairs} "
                            f"(prompt_too_long={pref_trunc_stats.dropped_prompt_too_long}, "
                            f"chosen_too_long={pref_trunc_stats.dropped_chosen_too_long}, "
                            f"rejected_too_long={pref_trunc_stats.dropped_rejected_too_long})"
                        )

                    (
                        sft_train_prompts,
                        sft_completions,
                        sft_weights,
                        sft_trunc_stats,
                    ) = filter_weighted_sft_without_truncation(
                        tokenizer=tokenizer,
                        train_prompts=sft_train_prompts_raw,
                        completions=sft_completions_raw,
                        weights=sft_weights_raw,
                        max_length=args.max_length,
                    )
                    dropped_sft_by_truncation_in_rollout += sft_trunc_stats.dropped_samples
                    if sft_trunc_stats.dropped_samples > 0:
                        print(
                            f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                            f"truncation_filter_sft dropped_samples={sft_trunc_stats.dropped_samples}/{sft_trunc_stats.total_samples} "
                            f"(prompt_too_long={sft_trunc_stats.dropped_prompt_too_long}, "
                            f"completion_too_long={sft_trunc_stats.dropped_completion_too_long})"
                        )

                    if not pref_train_prompts and not sft_train_prompts:
                        continue
                    if (sum(pref_weights) + sum(sft_weights)) <= 0:
                        continue

                    loss_stats = _online_run_preference_optimizer_step(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        device=device,
                        args=args,
                        pref_train_prompts=pref_train_prompts,
                        pref_chosen=pref_chosen,
                        pref_rejected=pref_rejected,
                        pref_weights=pref_weights,
                        sft_train_prompts=sft_train_prompts,
                        sft_completions=sft_completions,
                        sft_weights=sft_weights,
                    )
                    updates += 1
                    updates_in_rollout += 1
                    consumed_pref_pairs_in_rollout += loss_stats.pref_pairs_used
                    consumed_sft_samples_in_rollout += loss_stats.sft_samples_used
                    kept_pref_pairs += loss_stats.pref_pairs_used
                    kept_sft_samples += loss_stats.sft_samples_used
                    print(
                        f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                        f"optimizer_step={updates} pref_pairs={loss_stats.pref_pairs_used} "
                        f"sft_samples={loss_stats.sft_samples_used} "
                        f"loss={loss_stats.total_loss:.4f} "
                        f"pref_loss={loss_stats.pref_loss:.4f} "
                        f"sft_loss={loss_stats.sft_loss:.4f} "
                        f"gap={loss_stats.mean_gap:.4f}"
                    )

                    if args.online_save_every_updates > 0 and updates % args.online_save_every_updates == 0:
                        ckpt_dir = output_root / f"checkpoint-update-{updates}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"[online] saved checkpoint to {ckpt_dir}")

            if rollout_objectives and updates_in_rollout == 0:
                print(
                    f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                    "no optimizer update (all mixed/sft samples filtered out)"
                )
            elif rollout_objectives:
                print(
                    f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                    f"updates_in_rollout={updates_in_rollout} "
                    f"consumed_pref_pairs_in_rollout={consumed_pref_pairs_in_rollout} "
                    f"consumed_sft_samples_in_rollout={consumed_sft_samples_in_rollout} "
                    f"dropped_pref_pairs_by_truncation_in_rollout={dropped_pref_pairs_by_truncation_in_rollout} "
                    f"dropped_sft_by_truncation_in_rollout={dropped_sft_by_truncation_in_rollout}"
                )

            buffer = []
            if args.online_steps is not None and rollout_steps >= args.online_steps:
                break

        if buffer and (args.online_steps is None or rollout_steps < args.online_steps):
            print("[online] remaining tail batch ignored to keep fixed rollout_batch_size behavior")

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(
        f"[online] finished. rollout_steps={rollout_steps}, optimizer_steps={updates}, "
        f"scanned={scanned}, kept_pref_pairs={kept_pref_pairs}, kept_sft_samples={kept_sft_samples}, "
        f"logged_pref_objectives={logged_pref_objectives}, logged_sft_objectives={logged_sft_objectives}, "
        f"skipped_all_wrong={skipped_all_wrong}, skipped_after_filter={skipped_mixed_after_filter}, "
        f"objectives_log={online_pairs_path}, final_model={final_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DAPO preference training with vLLM rollout.")

    # pipeline control
    parser.add_argument(
        "--stage",
        type=str,
        default="online",
        choices=["online"],
        help="Only online mode is supported.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # io
    parser.add_argument("--dataset_path", type=str, default="/path/to/dapo-math-17k.parquet")
    parser.add_argument("--rollout_model_path", type=str, default="/path/to/Qwen3-4B")
    parser.add_argument("--train_model_path", type=str, default="/path/to/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-4b-pref")

    # online rollout
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
    parser.add_argument(
        "--rollout_n",
        type=int,
        default=8,
        help="Number of sampled responses per prompt during rollout.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--rollout_max_model_len", type=int, default=32768)
    parser.add_argument(
        "--sample_rejected_requires_final_answer",
        type=str2bool,
        default=True,
        help=(
            "Sampling filter: rejected response must have a parseable final answer on LAST line "
            "(supports markdown variants like '**Answer:** 10'). Truncated/no-final-answer rejected "
            "responses are dropped."
        ),
    )
    parser.add_argument(
        "--sample_chosen_requires_final_answer",
        type=str2bool,
        default=False,
        help="Sampling filter: if true, chosen response also needs a parseable final answer on LAST line.",
    )

    # shared chat format
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="fixed",
        choices=["none", "fixed", "random"],
        help="System prompt strategy. 'random' picks one prompt per sample from pool.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Single English system prompt. Used directly when prompt_mode=fixed (unless index points elsewhere).",
    )
    parser.add_argument(
        "--prompt_candidate",
        action="append",
        default=[],
        help="Add one candidate system prompt. Repeat this argument for multiple candidates.",
    )
    parser.add_argument(
        "--prompt_candidates_file",
        type=str,
        default="",
        help="TXT/JSON file containing candidate English system prompts.",
    )
    parser.add_argument(
        "--use_default_prompt_candidates",
        type=str2bool,
        default=False,
        help="Append built-in English prompt candidates to the prompt pool.",
    )
    parser.add_argument(
        "--prompt_fixed_index",
        type=int,
        default=0,
        help="Prompt index used when prompt_mode=fixed.",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        default=True,
        help="Whether to enable Qwen thinking mode when building chat template.",
    )

    # online optimization
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--chosen_ce_weight",
        type=float,
        default=0.02,
        help=(
            "Deprecated compatibility flag (kept for old run scripts). "
            "The online objective now uses weighted all-pairs preference + full-correct SFT."
        ),
    )
    parser.add_argument(
        "--pref_weight_rarity_floor",
        type=float,
        default=0.25,
        help="Rarity floor alpha in weight = entropy(r) * max(1-r, alpha) for mixed (correct/wrong) prompts.",
    )
    parser.add_argument(
        "--pref_weight_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon used in entropy(r) for mixed-prompt weighting.",
    )
    parser.add_argument(
        "--full_correct_sft_weight",
        type=float,
        default=1.0,
        help="Per-prompt objective weight used when all rollout samples are correct (full SFT branch).",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--logprob_micro_batch_size",
        type=int,
        default=8,
        help=(
            "Online only: number of preference pairs per step for logp computation. "
            "Each step runs chosen then rejected LM forward on the same prompts; "
            "uses scaled backward() per chunk to cap peak VRAM (avoids huge logits). "
            "Use 0 to process all pairs in one go (may OOM on long completions)."
        ),
    )
    parser.add_argument("--length_average", type=str2bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument(
        "--use_lora",
        type=str2bool,
        default=False,
        help="Train with PEFT LoRA (HF forward + preference loss); vLLM rollout loads base + adapter.",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA (Qwen/Llama-style attention/MLP).",
    )
    parser.add_argument(
        "--vllm_max_lora_rank",
        type=int,
        default=64,
        help="vLLM max_lora_rank when use_lora=true; must be >= lora_r.",
    )
    parser.add_argument(
        "--online_init_model_path",
        type=str,
        default="",
        help="Initial model path for online training. Defaults to train/rollout model path.",
    )
    parser.add_argument(
        "--online_steps",
        type=int,
        default=0,
        help=(
            "Online only: number of rollout batches to run. Each batch rolls out "
            "rollout_batch_size prompts with --rollout_n samples each; prompt-level "
            "objectives (weighted mixed all-pairs + full-correct SFT) are consumed "
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
    parser.add_argument(
        "--online_save_every_updates",
        type=int,
        default=0,
        help="Save checkpoint every N online updates. Use 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--online_rollout_backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
        help="Online: rollout engine - vLLM (default) or Hugging Face generate.",
    )
    parser.add_argument(
        "--online_vllm_use_tqdm",
        type=str2bool,
        default=True,
        help="Online + vLLM: show tqdm progress in llm.generate.",
    )
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.max_source_samples == 0:
        args.max_source_samples = None
    if args.online_steps == 0:
        args.online_steps = None

    if args.online_pairs_per_step < 1:
        raise SystemExit("error: --online-pairs-per-step must be >= 1")
    if args.rollout_n < 2:
        raise SystemExit("error: --rollout_n must be >= 2")
    if args.pref_weight_rarity_floor < 0:
        raise SystemExit("error: --pref_weight_rarity_floor must be >= 0")
    if args.pref_weight_eps <= 0:
        raise SystemExit("error: --pref_weight_eps must be > 0")
    if args.full_correct_sft_weight < 0:
        raise SystemExit("error: --full_correct_sft_weight must be >= 0")
    run_online_preference_training(args)


if __name__ == "__main__":
    main()


