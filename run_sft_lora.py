#!/usr/bin/env python3
"""
LoRA SFT training on the same DAPO-style parquet dataset.

Dataset schema (same as train_dapo_preference.py):
  - prompt: list[{"role": "...", "content": "..."}]
  - reward_model: {"ground_truth": "..."}
  - extra_info: {"index": "..."} (optional)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class DapoSample:
    prompt: str
    ground_truth: str
    sample_id: str


def str2bool(v: str) -> bool:
    value = v.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


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


def build_target_text(ground_truth: str, answer_prefix: str) -> str:
    return f"{answer_prefix}{ground_truth}".strip()


def build_records(args: argparse.Namespace, tokenizer: object) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for sample in iter_dapo_samples(
        parquet_path=args.dataset_path,
        scan_batch_size=args.scan_batch_size,
        max_source_samples=args.max_source_samples,
    ):
        prompt_text = apply_qwen_chat_template(
            tokenizer=tokenizer,
            prompt=sample.prompt,
            enable_thinking=args.enable_thinking,
            system_prompt=args.system_prompt,
        )
        completion_text = build_target_text(sample.ground_truth, args.answer_prefix)
        records.append(
            {
                "sample_id": sample.sample_id,
                "prompt_text": prompt_text,
                "completion_text": completion_text,
            }
        )
    return records


class SFTPromptMaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        tokenizer: object,
        max_length: int,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _encode_prompt_completion(self, prompt_text: str, completion_text: str) -> Optional[Dict[str, List[int]]]:
        """Tokenize prompt and completion separately, concat, then truncate.

        Separate calls on `prompt` vs `prompt+completion` break subword alignment and
        can yield labels that are all -100 after truncation, which makes loss nan/inf.
        """
        tok = self.tokenizer
        prompt_ids: List[int] = tok(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        completion_ids: List[int] = tok(
            completion_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if not completion_ids:
            return None
        max_len = self.max_length
        while len(prompt_ids) + len(completion_ids) > max_len:
            if len(prompt_ids) > 0:
                prompt_ids = prompt_ids[1:]
            else:
                if len(completion_ids) <= 1:
                    break
                completion_ids = completion_ids[:-1]
        if not completion_ids:
            return None
        full_ids = prompt_ids + completion_ids
        prompt_len = len(prompt_ids)
        if prompt_len > len(full_ids):
            prompt_len = 0
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]
        supervised = sum(1 for x in labels if x != -100)
        if supervised == 0:
            return None
        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        for k in range(len(self.records)):
            j = (idx + k) % len(self.records)
            r = self.records[j]
            out = self._encode_prompt_completion(r["prompt_text"], r["completion_text"])
            if out is not None:
                return out
        raise RuntimeError("No sample with non-empty supervised labels.")


class DataCollatorForPromptMaskedSFT:
    def __init__(self, tokenizer: object, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lb in labels:
            padded_labels.append(lb + [self.label_pad_token_id] * (max_len - len(lb)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LoRA SFT on DAPO parquet dataset")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument(
        "--max_source_samples",
        type=int,
        default=0,
        help="Cap parquet rows loaded (0: if max_steps>0, use max_steps*per_device_batch*grad_accum; else load all).",
    )
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--enable_thinking", type=str2bool, default=True)
    parser.add_argument("--answer_prefix", type=str, default="Answer: ")

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0, run exactly this many optimizer steps (overrides num_train_epochs).",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.max_source_samples == 0:
        if args.max_steps > 0:
            args.max_source_samples = (
                args.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps
            )
        else:
            args.max_source_samples = None

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = build_records(args, tokenizer)
    if not records:
        raise RuntimeError("No valid samples loaded from dataset.")

    random.shuffle(records)
    eval_size = int(len(records) * args.eval_ratio) if args.eval_ratio > 0 else 0
    if eval_size >= len(records):
        eval_size = max(0, len(records) - 1)
    eval_records = records[:eval_size]
    train_records = records[eval_size:]

    print(
        f"[sft] loaded={len(records)} train={len(train_records)} eval={len(eval_records)} "
        f"answer_prefix={json.dumps(args.answer_prefix)}"
    )

    train_dataset = SFTPromptMaskDataset(
        records=train_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    eval_dataset = (
        SFTPromptMaskDataset(
            records=eval_records,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        if eval_records
        else None
    )

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[args.torch_dtype],
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    target_modules = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
    )
    if args.max_steps > 0:
        training_args_kwargs["max_steps"] = args.max_steps
    if args.gradient_checkpointing:
        training_args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    try:
        training_args = TrainingArguments(**training_args_kwargs)
    except TypeError as exc:
        # Compatibility for some transformers versions that use `eval_strategy`.
        if "evaluation_strategy" not in str(exc):
            raise
        training_args_kwargs["eval_strategy"] = training_args_kwargs.pop("evaluation_strategy")
        training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForPromptMaskedSFT(tokenizer),
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] done -> {final_dir}")


if __name__ == "__main__":
    main()
