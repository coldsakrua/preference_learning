#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, Iterator, List, Set

import pyarrow as pa
import pyarrow.parquet as pq

from train_preference import DEFAULT_SYSTEM_PROMPT, run_online_preference_training
from utils import (
    build_parser as build_cli_parser,
    extract_final_answer_from_any_line,
    extract_user_prompt,
    str2bool,
)


_THINK_BEGIN_OPSD = "<|begin_of_thought|>"
_THINK_END_OPSD = "<|end_of_thought|>"
_SOLUTION_BEGIN_OPSD = "<|begin_of_solution|>"
_SOLUTION_END_OPSD = "<|end_of_solution|>"
_WS_RE = re.compile(r"\s+")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+)?answer\s*[:\uFF1A]\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


def _norm_for_dedup(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip()).lower()


def _clean_answer_text(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""
    match = _ANSWER_PREFIX_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text.strip().strip("$").strip()


def _extract_between(text: str, start_token: str, end_token: str) -> str:
    if not text:
        return ""
    start = text.find(start_token)
    if start < 0:
        return ""
    start += len(start_token)
    end = text.find(end_token, start)
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()


def _extract_conversation_assistant_text(conversations: object) -> str:
    if not isinstance(conversations, list):
        return ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", "")).strip().lower()
        if role == "assistant":
            return str(message.get("value", "")).strip()
    return ""


def _extract_conversation_user_text(conversations: object) -> str:
    if not isinstance(conversations, list):
        return ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from", "")).strip().lower()
        if role == "user":
            return str(message.get("value", "")).strip()
    return ""


def _sanitize_thought_text(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return ""
    for token in (_THINK_BEGIN_OPSD, _THINK_END_OPSD, _SOLUTION_BEGIN_OPSD, _SOLUTION_END_OPSD):
        out = out.replace(token, "")
    return out.strip()


def _extract_opsd_prompt(record: Dict[str, object]) -> str:
    problem = str(record.get("problem", "") or "").strip()
    if problem:
        return problem

    question = str(record.get("Question", "") or "").strip()
    if question:
        return question

    prompt_from_messages = extract_user_prompt(record.get("messages"))
    if prompt_from_messages:
        return prompt_from_messages

    prompt_from_conv = _extract_conversation_user_text(record.get("conversations"))
    if prompt_from_conv:
        return prompt_from_conv

    return ""


def _extract_opsd_answer(record: Dict[str, object]) -> str:
    answer = _clean_answer_text(str(record.get("Answer", "") or ""))
    if answer:
        return answer

    solution = str(record.get("solution", "") or "").strip()
    parsed = extract_final_answer_from_any_line(solution)
    return _clean_answer_text(parsed)


def _extract_opsd_thought(record: Dict[str, object]) -> str:
    assistant_text = _extract_conversation_assistant_text(record.get("conversations"))
    if assistant_text:
        thought = _extract_between(assistant_text, _THINK_BEGIN_OPSD, _THINK_END_OPSD)
        if thought:
            return _sanitize_thought_text(thought)

    cot_reason = str(record.get("COT_Reason", "") or "").strip()
    if cot_reason:
        return _sanitize_thought_text(cot_reason)

    solution_text = str(record.get("solution", "") or "").strip()
    if solution_text:
        return _sanitize_thought_text(solution_text)

    return ""


def _format_solution_with_thinking(
    *,
    thought: str,
    answer: str,
    think_token_format: str,
) -> str:
    clean_answer = _clean_answer_text(answer)
    if think_token_format == "opsd":
        if thought:
            return (
                f"{_THINK_BEGIN_OPSD}\n"
                f"{thought.strip()}\n"
                f"{_THINK_END_OPSD}\n\n"
                f"Answer: {clean_answer}"
            ).strip()
        return f"Answer: {clean_answer}".strip()

    # qwen format
    if thought:
        return f"<think>\n{thought.strip()}\n</think>\n\nAnswer: {clean_answer}".strip()
    return f"Answer: {clean_answer}".strip()


def _iter_parquet_paths(dataset_path: str) -> List[Path]:
    raw = str(dataset_path).strip()
    if not raw:
        return []

    candidate = Path(raw)
    if candidate.is_file():
        return [candidate.resolve()]
    if candidate.is_dir():
        return sorted(p.resolve() for p in candidate.glob("*.parquet"))

    matched = [Path(p).resolve() for p in glob.glob(raw, recursive=True)]
    return sorted(p for p in matched if p.is_file() and p.suffix.lower() == ".parquet")


def _iter_reference_prompts_from_parquet(parquet_path: Path, scan_batch_size: int) -> Iterator[str]:
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)

    if "problem" in schema_names:
        columns = ["problem"]
        field_name = "problem"
    elif "prompt" in schema_names:
        columns = ["prompt"]
        field_name = "prompt"
    elif "Question" in schema_names:
        columns = ["Question"]
        field_name = "Question"
    elif "messages" in schema_names:
        columns = ["messages"]
        field_name = "messages"
    else:
        return

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=columns):
        values = batch.column(field_name).to_pylist()
        for value in values:
            if field_name in {"prompt", "messages"} and isinstance(value, list):
                text = extract_user_prompt(value)
            else:
                text = str(value or "").strip()
            if text:
                yield text


def _build_reference_prompt_set(
    *,
    dedup_root: Path,
    excluded_parquet_paths: Set[Path],
    scan_batch_size: int,
) -> Set[str]:
    if not dedup_root.exists():
        return set()

    out: Set[str] = set()
    for parquet_file in sorted(dedup_root.rglob("*.parquet")):
        resolved = parquet_file.resolve()
        if resolved in excluded_parquet_paths:
            continue
        for prompt in _iter_reference_prompts_from_parquet(resolved, scan_batch_size):
            normalized = _norm_for_dedup(prompt)
            if normalized:
                out.add(normalized)
    return out


def build_opsd_thinking_parquet(args: argparse.Namespace) -> Path:
    source_paths = _iter_parquet_paths(args.dataset_path)
    if not source_paths:
        raise FileNotFoundError(
            f"No parquet files found from --dataset_path={args.dataset_path!r}; "
            "pass OPSD dir/file/glob."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.opsd_processed_parquet.strip():
        processed_path = Path(args.opsd_processed_parquet).resolve()
    else:
        processed_path = output_dir / "opsd_thinking_math_hf_dedup.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if processed_path.exists() and not args.opsd_overwrite_processed:
        print(f"[opsd] reuse existing processed parquet: {processed_path}")
        return processed_path

    reference_set: Set[str] = set()
    if args.opsd_enable_cross_dataset_dedup:
        dedup_root = Path(args.opsd_cross_dedup_root).resolve()
        excluded = {p.resolve() for p in source_paths}
        reference_set = _build_reference_prompt_set(
            dedup_root=dedup_root,
            excluded_parquet_paths=excluded,
            scan_batch_size=args.scan_batch_size,
        )
        print(
            "[opsd] built reference dedup set: "
            f"root={dedup_root} prompts={len(reference_set)}"
        )

    rows: List[Dict[str, str]] = []
    seen_opsd_prompts: Set[str] = set()
    total_read = 0
    dropped_missing_prompt_or_answer = 0
    dropped_intra_opsd_dup = 0
    dropped_cross_dataset_dup = 0

    for parquet_path in source_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        wanted_cols = [
            "problem",
            "Question",
            "Answer",
            "solution",
            "COT_Reason",
            "messages",
            "conversations",
        ]
        available_cols = set(parquet_file.schema_arrow.names)
        columns = [c for c in wanted_cols if c in available_cols]
        for batch in parquet_file.iter_batches(batch_size=args.scan_batch_size, columns=columns):
            for record in batch.to_pylist():
                total_read += 1
                prompt = _extract_opsd_prompt(record)
                answer = _extract_opsd_answer(record)
                if not prompt or not answer:
                    dropped_missing_prompt_or_answer += 1
                    continue

                norm_prompt = _norm_for_dedup(prompt)
                if not norm_prompt:
                    dropped_missing_prompt_or_answer += 1
                    continue

                if args.opsd_dedup_within_dataset and norm_prompt in seen_opsd_prompts:
                    dropped_intra_opsd_dup += 1
                    continue
                seen_opsd_prompts.add(norm_prompt)

                if reference_set and norm_prompt in reference_set:
                    dropped_cross_dataset_dup += 1
                    continue

                thought = _extract_opsd_thought(record)
                solution = _format_solution_with_thinking(
                    thought=thought,
                    answer=answer,
                    think_token_format=args.opsd_think_token_format,
                )
                rows.append({"problem": prompt, "solution": solution})

    if not rows:
        raise RuntimeError(
            "No samples left after OPSD preprocessing/dedup. "
            "Try disabling cross-dataset dedup or checking dataset path."
        )

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, processed_path, compression="zstd")
    print(
        "[opsd] wrote processed parquet: "
        f"path={processed_path} kept={len(rows)} total_read={total_read} "
        f"dropped_missing={dropped_missing_prompt_or_answer} "
        f"dropped_intra_opsd_dup={dropped_intra_opsd_dup} "
        f"dropped_cross_dataset_dup={dropped_cross_dataset_dup}"
    )
    return processed_path


def _validate_training_args(args: argparse.Namespace) -> None:
    if args.online_mle_on_correct_only and args.online_pref_loss_only:
        raise SystemExit(
            "error: --online_mle_on_correct_only and --online_pref_loss_only cannot both be true"
        )
    if args.online_mle_on_correct_only and (args.lambda_pref != 0 or args.lambda_gt != 0):
        print(
            "[online] online_mle_on_correct_only=true: preference branches are disabled; "
            "lambda_pref/lambda_gt will not be used."
        )
    if args.online_pref_loss_only and (args.lambda_mle != 0 or args.lambda_gt != 0):
        print(
            "[online] online_pref_loss_only=true: MLE/all-wrong GT branches are disabled; "
            "lambda_mle/lambda_gt will not be used."
        )

    if args.max_source_samples == 0:
        args.max_source_samples = None
    if args.online_steps == 0:
        args.online_steps = None

    validations = [
        (args.online_pairs_per_step < 1, "error: --online-pairs-per-step must be >= 1"),
        (args.rollout_n < 2, "error: --rollout_n must be >= 2"),
        (args.beta <= 0, "error: --beta must be > 0"),
        (
            args.lambda_mle < 0 or args.lambda_pref < 0 or args.lambda_gt < 0,
            "error: --lambda_mle/--lambda_pref/--lambda_gt must be >= 0",
        ),
        (
            args.prompt_smoothing_alpha < 0 or args.prompt_smoothing_beta < 0,
            "error: --prompt_smoothing_alpha/--prompt_smoothing_beta must be >= 0",
        ),
        (args.prompt_weight_gamma < 0, "error: --prompt_weight_gamma must be >= 0"),
        (
            args.prompt_weight_min < 0 or args.prompt_weight_max <= 0,
            "error: --prompt_weight_min must be >=0 and --prompt_weight_max must be >0",
        ),
        (
            args.prompt_weight_min > args.prompt_weight_max,
            "error: --prompt_weight_min must be <= --prompt_weight_max",
        ),
        (args.hidden_layer_offset < 1, "error: --hidden_layer_offset must be >= 1"),
        (
            args.rollout_feature_micro_batch_size < 0,
            "error: --rollout_feature_micro_batch_size must be >= 0",
        ),
    ]
    for failed, message in validations:
        if failed:
            raise SystemExit(message)


def build_parser() -> argparse.ArgumentParser:
    parser = build_cli_parser(DEFAULT_SYSTEM_PROMPT)
    parser.description = (
        "OPSD thinking-mode wrapper over train_preference.py "
        "(same online pipeline + same losses)."
    )
    parser.add_argument(
        "--opsd_processed_parquet",
        type=str,
        default="",
        help=(
            "Optional path to save the processed OPSD parquet (problem+solution). "
            "Default: <output_dir>/opsd_thinking_math_hf_dedup.parquet"
        ),
    )
    parser.add_argument(
        "--opsd_overwrite_processed",
        type=str2bool,
        default=False,
        help="If true, rebuild processed parquet even when it already exists.",
    )
    parser.add_argument(
        "--opsd_dedup_within_dataset",
        type=str2bool,
        default=True,
        help="Simple dedup inside OPSD by normalized prompt text.",
    )
    parser.add_argument(
        "--opsd_enable_cross_dataset_dedup",
        type=str2bool,
        default=True,
        help="Simple dedup against other parquet datasets under --opsd_cross_dedup_root.",
    )
    parser.add_argument(
        "--opsd_cross_dedup_root",
        type=str,
        default="data",
        help="Directory scanned recursively for reference parquet prompts to dedup against.",
    )
    parser.add_argument(
        "--opsd_think_token_format",
        type=str,
        default="qwen",
        choices=["qwen", "opsd"],
        help=(
            "Thinking tokens in generated solution text. "
            "'qwen' => <think>...</think>; 'opsd' => <|begin_of_thought|>...<|end_of_thought|>."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    processed_parquet_path = build_opsd_thinking_parquet(args)

    args.dataset_path = str(processed_parquet_path)
    args.dataset_layout = "math_hf"
    args.enable_thinking = True

    _validate_training_args(args)
    run_online_preference_training(args)


if __name__ == "__main__":
    main()
