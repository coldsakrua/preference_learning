#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from utils import (
    DEFAULT_GOLD_RATIONALE_KEY_PATHS,
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    answer_text_matches,
    detect_parquet_dataset_layout,
    extract_rollout_scored_answer,
    iter_dapo_samples,
    iter_math_hf_samples,
    set_seed,
    str2bool,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise math reasoning assistant. "
    "Solve the problem step by step, then end with exactly one final line in the format: "
    "Answer: $<final_answer>."
)


class JsonlMetricsCallback(TrainerCallback):
    """Persist every trainer log event to a standalone JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep a compact, consistent subset of key RL metrics for easy plotting.
        self._tracked_keywords = (
            "reward",
            "entropy",
            "kl",
            "loss",
            "adv",
            "return",
        )

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        record: Dict[str, Any] = {
            "event": "trainer_log",
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        for k, v in logs.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                record[k] = v
            else:
                record[k] = str(v)
        tracked_metrics: Dict[str, float] = {}
        for k, v in logs.items():
            if not isinstance(v, (int, float)):
                continue
            key_lower = str(k).lower()
            if any(token in key_lower for token in self._tracked_keywords):
                tracked_metrics[str(k)] = float(v)
        if tracked_metrics:
            record["tracked_metrics"] = tracked_metrics
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO RL training on top of a preference-tuned model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--dataset_layout",
        type=str,
        default="auto",
        choices=["auto", "dapo", "math_hf"],
    )
    parser.add_argument("--model_path", type=str, required=True, help="Base/fine-tuned checkpoint path.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument("--max_source_samples", type=int, default=20000)
    parser.add_argument("--user_content_suffix", type=str, default="")
    parser.add_argument("--auto_math_hf_user_suffix", type=str2bool, default=True)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt_mode", type=str, default="fixed", choices=["none", "fixed"])
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default="grpo-dapo-preference")
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--lora_path",
        "--lora-path",
        dest="lora_path",
        type=str,
        default="",
        help="Existing LoRA adapter path to continue GRPO fine-tuning from.",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    return parser


def _ensure_trl_version() -> None:
    version = importlib.metadata.version("trl")
    if version != "0.22.1":
        print(f"[warning] expected trl==0.22.1, but found trl=={version}")


def _build_prompt_messages(prompt: str, *, system_prompt: str, prompt_mode: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if prompt_mode == "fixed" and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": str(prompt).strip()})
    return messages


def _resolve_dataset_layout(dataset_layout: str, dataset_path: str) -> str:
    return detect_parquet_dataset_layout(dataset_path) if dataset_layout == "auto" else dataset_layout


def build_grpo_dataset(args: argparse.Namespace) -> Dataset:
    layout = _resolve_dataset_layout(args.dataset_layout, args.dataset_path)
    max_samples = None if args.max_source_samples == 0 else args.max_source_samples
    user_suffix = str(args.user_content_suffix or "")
    if not user_suffix and layout == "math_hf" and args.auto_math_hf_user_suffix:
        user_suffix = DEFAULT_MATH_HF_USER_CONTENT_SUFFIX

    rows: List[Dict[str, Any]] = []
    if layout == "dapo":
        source_iter = iter_dapo_samples(
            parquet_path=args.dataset_path,
            scan_batch_size=args.scan_batch_size,
            max_source_samples=max_samples,
            gold_rationale_key_paths=list(DEFAULT_GOLD_RATIONALE_KEY_PATHS),
            require_gold_rationale=False,
        )
    elif layout == "math_hf":
        source_iter = iter_math_hf_samples(
            parquet_path=args.dataset_path,
            scan_batch_size=args.scan_batch_size,
            max_source_samples=max_samples,
            gold_rationale_key_paths=[],
            require_gold_rationale=False,
        )
    else:
        raise ValueError(f"Unsupported dataset_layout: {layout}")

    for sample in source_iter:
        prompt_text = sample.prompt + user_suffix
        rows.append(
            {
                "prompt": _build_prompt_messages(
                    prompt_text, system_prompt=args.system_prompt, prompt_mode=args.prompt_mode
                ),
                "ground_truth": sample.ground_truth,
                "sample_id": sample.sample_id,
            }
        )

    if not rows:
        raise RuntimeError("No training samples were loaded from dataset.")
    print(f"[grpo] loaded {len(rows)} samples (layout={layout})")
    return Dataset.from_list(rows)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return str(content) if content is not None else ""
    if isinstance(completion, list):
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if content is not None:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(completion)


def build_reward_funcs() -> List[Any]:
    def answer_accuracy_reward(completions: List[Any], ground_truth: List[str], **_: Any) -> List[float]:
        rewards: List[float] = []
        for completion, target in zip(completions, ground_truth):
            text = _completion_to_text(completion)
            has_final, parsed = extract_rollout_scored_answer(text)
            pred = parsed if has_final else ""
            rewards.append(1.0 if answer_text_matches(pred, str(target)) else -0.2)
        return rewards

    def answer_format_reward(completions: List[Any], **_: Any) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            has_final, _ = extract_rollout_scored_answer(_completion_to_text(completion))
            rewards.append(0.1 if has_final else -0.1)
        return rewards

    return [answer_accuracy_reward, answer_format_reward]


def _build_peft_config(args: argparse.Namespace) -> Optional[Any]:
    if not args.use_lora:
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError("use_lora=true requires peft to be installed.") from exc

    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not targets:
        raise ValueError("--lora_target_modules resolves to an empty module list.")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )


def _ensure_input_require_grads_for_checkpointing(model: Any) -> None:
    """Make sure checkpointed LoRA training has grad-carrying inputs."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    if not hasattr(model, "get_input_embeddings"):
        return
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        return
    if getattr(model, "_grpo_input_require_grads_hook", None) is not None:
        return

    def _make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            if output and hasattr(output[0], "requires_grad_"):
                output[0].requires_grad_(True)
            return output
        if hasattr(output, "requires_grad_"):
            output.requires_grad_(True)
        return output

    hook = embeddings.register_forward_hook(_make_inputs_require_grad)
    setattr(model, "_grpo_input_require_grads_hook", hook)


def _adapter_dir_has_weights(d: Path) -> bool:
    return (d / "adapter_model.safetensors").is_file() or (d / "adapter_model.bin").is_file()


def _resolve_lora_adapter_dir(user_path: str) -> Optional[Path]:
    raw = user_path.strip()
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LoRA path not found: {root}")
    if _adapter_dir_has_weights(root):
        return root
    for cand in (root / "final", root / "lora_adapter"):
        if _adapter_dir_has_weights(cand):
            return cand
    raise FileNotFoundError(
        f"No adapter weights found under {root} (or its final/ / lora_adapter/ subdirs)."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ensure_trl_version()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl_path = output_dir / "training_metrics.jsonl"
    train_dataset = build_grpo_dataset(args)
    reward_funcs = build_reward_funcs()

    report_to = [] if args.report_to.lower() == "none" else [args.report_to]
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_dir = _resolve_lora_adapter_dir(args.lora_path) if args.use_lora else None
    if not args.use_lora and args.lora_path.strip():
        print("[grpo] use_lora=false: ignore --lora_path and train base model directly.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_for_trainer: Any = args.model_path
    peft_config = _build_peft_config(args)
    if args.use_lora and lora_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("--lora_path requires peft to be installed.") from exc
        base_model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        model_for_trainer = PeftModel.from_pretrained(base_model, str(lora_dir), is_trainable=True)
        peft_config = None
        print(f"[grpo] loaded existing LoRA adapter: {lora_dir}")
    if args.use_lora and args.gradient_checkpointing:
        _ensure_input_require_grads_for_checkpointing(model_for_trainer)

    trainer = GRPOTrainer(
        model=model_for_trainer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(JsonlMetricsCallback(metrics_jsonl_path))
    with metrics_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": "run_start",
                    "model_path": args.model_path,
                    "dataset_path": args.dataset_path,
                    "output_dir": str(output_dir),
                    "metrics_jsonl": str(metrics_jsonl_path),
                    "use_lora": bool(args.use_lora),
                    "lora_path": str(args.lora_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    resume_ckpt = args.resume_from_checkpoint.strip() or None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if args.use_lora:
        model_to_save = trainer.accelerator.unwrap_model(trainer.model)
        model_to_save.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with metrics_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": "run_end",
                    "final_model": str(final_dir),
                    "metrics_jsonl": str(metrics_jsonl_path),
                    "global_step": int(getattr(trainer.state, "global_step", 0)),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"[grpo] finished. final_model={final_dir}")


if __name__ == "__main__":
    main()
