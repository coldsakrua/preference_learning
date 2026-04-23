#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-specific launcher for local math eval.

This file does not replace `eval_math_vllm_local.py`.
It only injects Llama-friendly defaults, while still delegating
the real evaluation logic to the shared evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import eval_math_vllm_local as base_eval


DEFAULT_MODEL_PATH = "/gpfs/share/home/2501210611/labShare/2501210611/model/llama-3.2-1b"
DEFAULT_DATASETS = ["math500", "aime24", "aime25", "aime26"]
DEFAULT_OUTPUT_ROOT = "outputs/eval_math_llama3_2_1b_local/no_cot"


def _has_flag(argv: List[str], flag: str) -> bool:
    return flag in argv


def _has_prefixed(argv: List[str], prefix: str) -> bool:
    return any(a.startswith(prefix) for a in argv)


def _inject_kv(argv: List[str], key: str, value: str) -> List[str]:
    if _has_flag(argv, key) or _has_prefixed(argv, f"{key}="):
        return argv
    return [*argv, key, value]


def _inject_flag(argv: List[str], flag: str) -> List[str]:
    if _has_flag(argv, flag):
        return argv
    return [*argv, flag]


def _inject_default_datasets(argv: List[str]) -> List[str]:
    has_dataset = _has_flag(argv, "--dataset") or _has_prefixed(argv, "--dataset=")
    has_data_path = _has_flag(argv, "--data-path") or _has_prefixed(argv, "--data-path=")
    if has_dataset or has_data_path:
        return argv
    out = list(argv)
    for ds in DEFAULT_DATASETS:
        out.extend(["--dataset", ds])
    return out


def _default_output_json() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_id = Path("/proc/self").exists()  # always true on Linux; keep deterministic suffix style simple.
    _ = job_id
    return f"{DEFAULT_OUTPUT_ROOT}/eval_{stamp}.json"


def build_llama_args(raw_argv: List[str]) -> List[str]:
    argv = list(raw_argv)
    argv = _inject_kv(argv, "--model-path", DEFAULT_MODEL_PATH)
    argv = _inject_default_datasets(argv)

    # Llama-base friendly defaults.
    argv = _inject_flag(argv, "--no-thinking")
    argv = _inject_flag(argv, "--force-base-tokenizer")
    argv = _inject_kv(argv, "--temperature", "0.3")
    argv = _inject_kv(argv, "--top-p", "0.9")
    argv = _inject_kv(argv, "--val-n", "16")
    argv = _inject_kv(argv, "--pass-at-k", "1,4,8,16")
    argv = _inject_kv(argv, "--max-new-tokens", "4096")
    argv = _inject_kv(argv, "--generate-batch-size", "16")
    argv = _inject_kv(argv, "--tensor-parallel-size", "1")
    argv = _inject_kv(argv, "--gpu-memory-utilization", "0.9")
    argv = _inject_kv(argv, "--output-json", _default_output_json())
    return argv


def main() -> None:
    forwarded = build_llama_args(sys.argv[1:])
    try:
        sys.argv = [sys.argv[0], *forwarded]
        base_eval.main()
        return
    except ValueError as e:
        msg = str(e)
        if "tokenizer.chat_template is not set" not in msg:
            raise
        print("[llama-eval] vLLM path failed due to missing chat_template, fallback to HF generate().", flush=True)
        run_hf_fallback_eval(forwarded)


def _parse_forwarded_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--data-path", action="append", default=[])
    p.add_argument("--data-format", type=str, default="auto")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--val-n", type=int, default=16)
    p.add_argument("--pass-at-k", type=str, default="1,4,8,16")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_known_args(argv)


def run_hf_fallback_eval(argv: List[str]) -> None:
    args, unknown = _parse_forwarded_args(argv)
    if unknown:
        print(f"[eval-hf] ignore unsupported forwarded args: {' '.join(unknown)}", flush=True)
    data_root = base_eval.default_data_root()
    limit = args.num_samples if args.num_samples > 0 else None

    load_queue: List[tuple[str, str | None]] = []
    for dn in args.dataset:
        p = base_eval.resolve_dataset_path(dn, data_root)
        load_queue.append((str(p), base_eval.normalize_dataset_key(dn)))
    for raw in args.data_path:
        load_queue.append((raw, None))
    if not load_queue:
        raise SystemExit("No dataset/data-path provided.")

    examples: List[Dict[str, Any]] = []
    resolved_paths: List[Path] = []
    tag_counts: Dict[str, int] = {}
    for raw, tag_override in load_queue:
        data_path = Path(raw).expanduser().resolve()
        resolved_paths.append(data_path)
        batch = base_eval.load_examples(data_path, args.data_format, limit)
        base_tag = tag_override if tag_override is not None else data_path.stem
        tag_counts[base_tag] = tag_counts.get(base_tag, 0) + 1
        tag = base_tag if tag_counts[base_tag] == 1 else f"{base_tag}_{tag_counts[base_tag]}"
        for ex in batch:
            ex["dataset_tag"] = tag
            ex["dataset_path"] = str(data_path)
        examples.extend(batch)
        print(f"[eval-hf] +{len(batch)} problems from {data_path} (tag={tag})", flush=True)

    if not examples:
        raise RuntimeError("No examples loaded for fallback eval.")

    pass_at_k_list = base_eval.parse_pass_at_k(args.pass_at_k)
    gen_n = max(args.val_n, max(pass_at_k_list))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    user_suffix = (
        "\n\nSolve the problem."
        "\nReturn exactly one final answer."
        "\nThe last line must be exactly: Final answer: \\boxed{...}"
    )
    prompts = [ex["problem"] + user_suffix for ex in examples]

    results: List[Dict[str, Any]] = []
    pass_at_k_counts: Dict[int, int] = {k: 0 for k in pass_at_k_list}
    total_correct = 0
    total_solutions = 0
    formatted_total = 0
    majority_correct = 0

    for ex, prompt in tqdm(
        zip(examples, prompts),
        total=len(examples),
        desc="hf_prompt_eval",
        dynamic_ncols=True,
    ):
        generations: List[str] = []
        preds: List[str] = []
        correct_flags: List[bool] = []
        formatted_flags: List[bool] = []
        enc = tokenizer([prompt] * gen_n, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for i in range(gen_n):
            gen = tokenizer.decode(out[i, int(prompt_lens[i]):], skip_special_tokens=True)
            pred = base_eval.extract_boxed_answer(gen)
            formatted = pred is not None
            p = pred if pred is not None else "[no boxed]"
            ok = base_eval.grade_answer(pred, ex["ground_truth"])
            generations.append(gen)
            preds.append(p)
            correct_flags.append(ok)
            formatted_flags.append(formatted)
            total_solutions += 1
            total_correct += int(ok)
            formatted_total += int(formatted)

        pass_at_k_problem: Dict[str, bool] = {}
        for k in pass_at_k_list:
            ok_k = any(correct_flags[:k])
            pass_at_k_problem[str(k)] = ok_k
            if ok_k:
                pass_at_k_counts[k] += 1

        maj_ok = False
        fpreds = [p for p, f in zip(preds, formatted_flags) if f]
        if fpreds:
            top = max(set(fpreds), key=fpreds.count)
            maj_ok = base_eval.grade_answer(top, ex["ground_truth"])
        majority_correct += int(maj_ok)

        results.append(
            {
                "dataset_tag": ex.get("dataset_tag", ""),
                "dataset_path": ex.get("dataset_path", ""),
                "problem_id": ex["id"],
                "problem": ex["problem"],
                "ground_truth": ex["ground_truth"],
                "gen_n": gen_n,
                "pass_at_k": pass_at_k_problem,
                "generations": [
                    {
                        "predicted_answer": p,
                        "full_generation": g,
                        "correct": c,
                        "formatted": f,
                    }
                    for p, g, c, f in zip(preds, generations, correct_flags, formatted_flags)
                ],
                "num_correct": sum(correct_flags),
                "pass_at_gen_n": bool(any(correct_flags)),
                "majority_vote_correct": maj_ok,
                "predicted_answer": preds[0],
                "full_generation": generations[0],
                "correct": correct_flags[0],
                "formatted": formatted_flags[0],
            }
        )

    n = len(results)
    pass_at_k_summary: Dict[str, Dict[str, Any]] = {}
    for k in pass_at_k_list:
        c = pass_at_k_counts[k]
        pass_at_k_summary[str(k)] = {
            "count": c,
            "total": n,
            "pct": 100.0 * c / n if n else 0.0,
        }
    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_tag.setdefault(str(r.get("dataset_tag", "")), []).append(r)
    metrics_by_dataset = {}
    for tag, sub in sorted(by_tag.items(), key=lambda x: x[0]):
        path0 = sub[0].get("dataset_path", "") if sub else ""
        metrics_by_dataset[tag] = {"dataset_path": path0, **base_eval.summarize_result_subset(sub, pass_at_k_list, gen_n)}

    summary = {
        "model_path": args.model_path,
        "backend": "hf_generate_fallback",
        "data_root": str(data_root),
        "data_paths": [str(p) for p in resolved_paths],
        "dataset_args": list(args.dataset),
        "data_format": args.data_format,
        "enable_thinking": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "val_n_requested": args.val_n,
        "gen_n": gen_n,
        "pass_at_k_list": pass_at_k_list,
        "pass_at_k": pass_at_k_summary,
        "metrics_by_dataset": metrics_by_dataset,
        "num_problems": n,
        "num_problems_total": n,
        "total_solutions": total_solutions,
        "average_pass1_over_gen_n_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
        "average_correct_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
        "majority_vote_pct": 100.0 * majority_correct / n if n else 0.0,
        "format_rate_pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
        "math_verify": True,
        "streaming_write": False,
        "results": results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[eval-hf] wrote final json -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
