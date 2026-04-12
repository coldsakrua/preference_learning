#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Two-stage preference training for DAPO math data.

Stage 1 (generate):
  - Use vLLM to rollout each prompt twice.
  - Keep samples where exactly one rollout is correct and the other is wrong.
  - Save (prompt, system_prompt, chosen, rejected, ground_truth) into JSONL.

Stage 2 (train):
  - Optimize preference objective:
      L = -log(sigmoid(beta * (logpi(y+|x) - logpi(y-|x))))
  - Supports optional length normalization for logpi.
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
from typing import Dict, Iterator, List, Optional, Sequence

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


_ANSWER_LINE = re.compile(r"answer\s*:\s*(.+)", flags=re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_LATEX_FRAC = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")


def extract_final_answer(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_LINE.search(line)
        if match:
            return match.group(1).strip()
    if lines:
        return lines[-1]
    return text


def extract_final_answer_if_last_line(text: str) -> tuple[bool, str]:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, ""
    last_line = lines[-1]
    match = re.fullmatch(r"answer\s*:\s*(.+)", last_line, flags=re.IGNORECASE)
    if not match:
        return False, ""
    answer = match.group(1).strip()
    if not answer:
        return False, ""
    return True, answer


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
    predicted = normalize_answer(extract_final_answer(predicted_text))
    target = normalize_answer(ground_truth)
    if predicted and predicted == target:
        return True
    predicted_num = to_number_if_simple(predicted)
    target_num = to_number_if_simple(target)
    if predicted_num is not None and target_num is not None:
        return abs(predicted_num - target_num) <= 1e-6
    return False


def answer_text_matches(answer_text: str, ground_truth: str) -> bool:
    predicted = normalize_answer(answer_text)
    target = normalize_answer(ground_truth)
    if predicted and predicted == target:
        return True
    predicted_num = to_number_if_simple(predicted)
    target_num = to_number_if_simple(target)
    if predicted_num is not None and target_num is not None:
        return abs(predicted_num - target_num) <= 1e-6
    return False


def choose_preference_pair(
    candidates: Sequence[str],
    ground_truth: str,
    require_rejected_final_answer: bool = True,
    require_chosen_final_answer: bool = False,
) -> Optional[Dict[str, str]]:
    chosen = None
    chosen_final_answer = ""
    rejected = None
    rejected_final_answer = ""
    for candidate in candidates:
        has_final_line, final_answer = extract_final_answer_if_last_line(candidate)
        if has_final_line:
            is_correct = answer_text_matches(final_answer, ground_truth)
        else:
            is_correct = answers_match(candidate, ground_truth)

        if is_correct:
            if require_chosen_final_answer and (not has_final_line):
                continue
            if chosen is None:
                chosen = candidate
                chosen_final_answer = final_answer if has_final_line else ""
        else:
            if require_rejected_final_answer and (not has_final_line):
                continue
            if rejected is None:
                rejected = candidate
                rejected_final_answer = final_answer if has_final_line else ""
    if chosen is None or rejected is None:
        return None
    return {
        "chosen": chosen,
        "rejected": rejected,
        "chosen_final_answer": chosen_final_answer,
        "rejected_final_answer": rejected_final_answer,
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
        n=2,
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
                    continue
                record = {
                    "sample_id": sample_obj.sample_id,
                    "prompt": sample_obj.prompt,
                    "system_prompt": system_prompt,
                    "ground_truth": sample_obj.ground_truth,
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                    "chosen_final_answer": pair["chosen_final_answer"],
                    "rejected_final_answer": pair["rejected_final_answer"],
                }
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
                    continue
                record = {
                    "sample_id": sample_obj.sample_id,
                    "prompt": sample_obj.prompt,
                    "system_prompt": system_prompt,
                    "ground_truth": sample_obj.ground_truth,
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                    "chosen_final_answer": pair["chosen_final_answer"],
                    "rejected_final_answer": pair["rejected_final_answer"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved_pairs += 1
                pbar.update(1)
                if args.target_pairs is not None and saved_pairs >= args.target_pairs:
                    break
        pbar.close()

    kept_ratio = (saved_pairs / processed) if processed else 0.0
    print(
        f"[generate] processed={processed}, saved_pairs={saved_pairs}, "
        f"keep_ratio={kept_ratio:.4f}, output={output_path}"
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
                loss = -F.logsigmoid(args.beta * preference_gap).mean()

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

    if hf_updates_so_far > 0:
        vllm_staging_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(vllm_staging_dir)
        tokenizer.save_pretrained(vllm_staging_dir)
        ckpt = str(vllm_staging_dir)
    else:
        ckpt = init_model_path

    model.eval()
    model.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[online] vLLM loading rollout_step={rollout_steps}/{total_steps_str} ckpt={ckpt}",
        flush=True,
    )
    llm = LLM(
        model=ckpt,
        tokenizer=ckpt,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.rollout_max_model_len,
    )
    sampling_params = SamplingParams(
        n=2,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed + rollout_steps * 100003,
    )
    outputs = llm.generate(
        prompt_texts,
        sampling_params,
        use_tqdm=args.online_vllm_use_tqdm,
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
        expanded_input_ids = input_ids.repeat_interleave(2, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(2, dim=0)
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


def run_online_preference_training(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_root = Path(args.output_dir)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
    kept_pairs = 0
    buffer: List[DapoSample] = []

    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"
    print(
        f"[online] rollout_backend={args.online_rollout_backend}, "
        f"rollout_batch_size={args.rollout_batch_size} (2 samples per prompt via n=2), "
        f"online_steps={total_steps_str}, max_source_samples={args.max_source_samples}"
    )
    if args.online_rollout_backend == "vllm" and device.type != "cuda":
        raise RuntimeError("online_rollout_backend=vllm requires a CUDA device.")

    with online_pairs_path.open("w", encoding="utf-8") as fout:
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

            train_prompts: List[str] = []
            chosen: List[str] = []
            rejected: List[str] = []
            for idx, sample_obj in enumerate(buffer):
                candidates = completion_flat[2 * idx : 2 * idx + 2]
                pair = choose_preference_pair(
                    candidates,
                    sample_obj.ground_truth,
                    require_rejected_final_answer=args.sample_rejected_requires_final_answer,
                    require_chosen_final_answer=args.sample_chosen_requires_final_answer,
                )
                if pair is None:
                    continue
                train_prompts.append(prompt_texts[idx])
                chosen.append(pair["chosen"])
                rejected.append(pair["rejected"])
                record = {
                    "sample_id": sample_obj.sample_id,
                    "prompt": sample_obj.prompt,
                    "system_prompt": system_prompts[idx],
                    "ground_truth": sample_obj.ground_truth,
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                    "chosen_final_answer": pair["chosen_final_answer"],
                    "rejected_final_answer": pair["rejected_final_answer"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if train_prompts:
                batch_size = len(train_prompts)
                mb = (
                    args.logprob_micro_batch_size
                    if args.logprob_micro_batch_size > 0
                    else batch_size
                )

                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                total_gap_weighted = 0.0
                for start in range(0, batch_size, mb):
                    end = min(start + mb, batch_size)
                    tp = train_prompts[start:end]
                    ch = chosen[start:end]
                    rj = rejected[start:end]
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
                    loss_c = -F.logsigmoid(args.beta * preference_gap).mean()
                    scale = (end - start) / batch_size
                    (loss_c * scale).backward()
                    weighted_loss += loss_c.item() * scale
                    total_gap_weighted += preference_gap.sum().item()

                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                updates += 1
                kept_pairs += batch_size
                mean_gap = total_gap_weighted / batch_size
                print(
                    f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                    f"optimizer_step={updates} scanned={scanned} "
                    f"kept_in_batch={batch_size} loss={weighted_loss:.4f} "
                    f"gap={mean_gap:.4f}"
                )

                if args.online_save_every_updates > 0 and updates % args.online_save_every_updates == 0:
                    ckpt_dir = output_root / f"checkpoint-update-{updates}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[online] saved checkpoint to {ckpt_dir}")
            else:
                print(
                    f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                    f"scanned={scanned} kept_in_batch=0 skip_optimizer"
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
        f"scanned={scanned}, kept_pairs={kept_pairs}, "
        f"pairs_log={online_pairs_path}, final_model={final_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAPO preference training with vLLM rollout + DPO-style loss.")

    # pipeline control
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "generate", "train", "online"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # io
    parser.add_argument("--dataset_path", type=str, default="/path/to/dapo-math-17k.parquet")
    parser.add_argument("--rollout_model_path", type=str, default="/path/to/Qwen3-4B")
    parser.add_argument("--train_model_path", type=str, default="/path/to/Qwen3-4B")
    parser.add_argument("--preference_pairs_path", type=str, default="./outputs/dapo_pref_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3-4b-pref")

    # generate stage
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument(
        "--max_source_samples",
        type=int,
        default=17000,
        help="Maximum source prompts scanned for pair mining.",
    )
    parser.add_argument(
        "--target_pairs",
        type=int,
        default=5000,
        help="Stop generating when this many preference pairs are collected. Use 0 for no limit.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--rollout_max_model_len", type=int, default=8192)
    parser.add_argument(
        "--sample_rejected_requires_final_answer",
        type=str2bool,
        default=True,
        help=(
            "Pair mining: keep a rejected sample only if its LAST non-empty line is "
            "'Answer: ...' and the parsed answer is wrong. Truncated/no-final-answer rejected "
            "samples are dropped."
        ),
    )
    parser.add_argument(
        "--sample_chosen_requires_final_answer",
        type=str2bool,
        default=False,
        help=(
            "Pair mining: optionally require chosen sample to also have LAST line "
            "'Answer: ...'."
        ),
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

    # train stage
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.1)
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
    parser.add_argument("--max_train_pairs", type=int, default=0)
    parser.add_argument("--save_every_epoch", type=str2bool, default=True)
    parser.add_argument(
        "--record_train_samples",
        type=str2bool,
        default=False,
        help="Train only: save actually sampled training pairs per step to JSONL.",
    )
    parser.add_argument(
        "--train_sample_log_path",
        type=str,
        default="",
        help="Train only: JSONL path for sampled training records. Default: <output_dir>/train_sampled_pairs.jsonl",
    )
    parser.add_argument(
        "--train_sample_log_max_records",
        type=int,
        default=0,
        help="Train only: maximum sampled records to log. Use 0 for no limit.",
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
            "rollout_batch_size prompts with 2 samples each, keeps one-right-one-wrong pairs, "
            "then optimizer.step() if any. Use 0 for no limit (until source samples exhausted)."
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
        help="Online: rollout engine — vLLM (default) or Hugging Face generate.",
    )
    parser.add_argument(
        "--online_vllm_use_tqdm",
        type=str2bool,
        default=True,
        help="Online + vLLM: show tqdm progress in llm.generate.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.target_pairs == 0:
        args.target_pairs = None
    if args.max_source_samples == 0:
        args.max_source_samples = None
    if args.max_train_pairs == 0:
        args.max_train_pairs = None
    if args.train_sample_log_max_records == 0:
        args.train_sample_log_max_records = None
    if args.online_steps == 0:
        args.online_steps = None

    if args.stage == "online":
        run_online_preference_training(args)
        return

    if args.stage in {"all", "generate"}:
        generate_preference_pairs(args)
    if args.stage in {"all", "train"}:
        train_with_preference_loss(args)


if __name__ == "__main__":
    main()

