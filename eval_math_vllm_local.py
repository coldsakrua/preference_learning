#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local math eval with vLLM + Qwen3 chat template + \\boxed{} extraction + math_verify grading.

Data under ``preference_learning/data/`` can be selected by **name** (``--dataset``), no full path needed.
Use ``--list-datasets`` to print known names. Custom files still use ``--data-path``.

File formats (auto by suffix if ``--data-format auto``):
  - *.jsonl: {"problem", "answer", "id"(optional)}
  - *.parquet: detected from schema —
    DAPO: prompt (chat messages), reward_model.ground_truth, extra_info;
    AMO-Bench-like: prompt (plain string), answer, question_id (optional);
    CMIMC/HMMT/BRUMO-like: problem, answer, problem_idx (optional)

Summary JSON includes ``pass_at_k`` (e.g. pass@1/4/8/16): each k counts problems where
at least one of the first k samples is graded correct.

Example:
  python eval_math_vllm_local.py \\
    --model-path /path/to/qwen3-4b \\
    --data-path data/AIME26/test.parquet \\
    --val-n 16 --pass-at-k 1,4,8,16 \\
    --output-json outputs/aime26_eval.json

  python eval_math_vllm_local.py \\
    --model-path /path/to/qwen3-4b \\
    --data-path data/AIME26/aime2026.jsonl \\
    --output-json outputs/aime26_eval.json

  # Multiple local datasets in one vLLM run (JSON has ``metrics_by_dataset``):
  python eval_math_vllm_local.py \\
    --model-path /path/to/qwen3-4b \\
    --data-path data/AIME26/test.parquet \\
    --data-path data/other/test.parquet \\
    --output-json outputs/multi_eval.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


def extract_boxed_answer(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left_braces = 0
    right_brace_idx = None
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
        return None
    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def grade_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    if _HAS_MATH_VERIFY:
        try:
            pred_w = predicted if "$" in predicted else f"${predicted}$"
            gt_w = ground_truth if "$" in ground_truth else f"${ground_truth}$"
            pred_parsed = parse(pred_w, fallback_mode="no_fallback")
            gt_parsed = parse(gt_w, fallback_mode="no_fallback")
            return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
        except Exception:
            pass
    pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
    gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
    return pred_norm == gt_norm


def load_jsonl_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(
                {
                    "id": o.get("id", len(rows)),
                    "problem": str(o["problem"]).strip(),
                    "ground_truth": str(o["answer"]).strip(),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _parquet_loader_kind(path: Path) -> str:
    """Return 'dapo', 'amo_qa', or 'problem_answer' based on Parquet schema."""
    schema = pq.ParquetFile(path).schema_arrow
    names = set(schema.names)
    if "reward_model" in names:
        return "dapo"
    if "problem" in names and "answer" in names:
        return "problem_answer"
    if "prompt" in names and "answer" in names:
        pt = schema.field("prompt").type
        if pa.types.is_string(pt) or pa.types.is_large_string(pt):
            return "amo_qa"
    raise ValueError(
        f"Unsupported parquet schema in {path}; "
        f"need DAPO (reward_model+…), or string prompt+answer, or problem+answer. "
        f"Columns: {sorted(names)}"
    )


def load_amo_qa_parquet_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """AMO-Bench style: prompt (string), answer, optional question_id."""
    rows: List[Dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = ["prompt", "answer"]
    if "question_id" in names:
        cols = ["question_id", "prompt", "answer"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        if "question_id" in cols:
            qids = batch.column("question_id").to_pylist()
        else:
            qids = None
        prompts = batch.column("prompt").to_pylist()
        answers = batch.column("answer").to_pylist()
        for i, (pr, ans) in enumerate(zip(prompts, answers)):
            problem = str(pr).strip() if pr is not None else ""
            gt = str(ans).strip() if ans is not None else ""
            if not problem or not gt:
                continue
            if qids is not None:
                sid = str(qids[i]).strip() if qids[i] is not None else str(len(rows))
            else:
                sid = str(len(rows))
            rows.append({"id": sid, "problem": problem, "ground_truth": gt})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_problem_answer_parquet_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """CMIMC / HMMT / BRUMO style: problem, answer; id from problem_idx or id if present."""
    rows: List[Dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    id_col: Optional[str] = None
    if "problem_idx" in names:
        id_col = "problem_idx"
    elif "id" in names:
        id_col = "id"
    cols = ["problem", "answer"]
    if id_col:
        cols = [id_col, "problem", "answer"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        if id_col:
            ids = batch.column(id_col).to_pylist()
        else:
            ids = None
        problems = batch.column("problem").to_pylist()
        answers = batch.column("answer").to_pylist()
        for i, (pr, ans) in enumerate(zip(problems, answers)):
            problem = str(pr).strip() if pr is not None else ""
            gt = str(ans).strip() if ans is not None else ""
            if not problem or not gt:
                continue
            if ids is not None:
                raw_id = ids[i]
                sid = str(raw_id).strip() if raw_id is not None else str(len(rows))
            else:
                sid = str(len(rows))
            rows.append({"id": sid, "problem": problem, "ground_truth": gt})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_dapo_parquet_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    from train_dapo_preference import extract_user_prompt

    rows: List[Dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    cols = ["prompt", "reward_model", "extra_info"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        prompts = batch.column("prompt").to_pylist()
        rewards = batch.column("reward_model").to_pylist()
        extras = batch.column("extra_info").to_pylist()
        for prompt_obj, reward_obj, extra_obj in zip(prompts, rewards, extras):
            problem = extract_user_prompt(prompt_obj)
            if not problem:
                continue
            gt = ""
            if isinstance(reward_obj, dict):
                gt = str(reward_obj.get("ground_truth", "")).strip()
            if not gt:
                continue
            sid = ""
            if isinstance(extra_obj, dict):
                sid = str(extra_obj.get("index", "")).strip()
            if not sid:
                sid = str(len(rows))
            rows.append({"id": sid, "problem": problem, "ground_truth": gt})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_examples(path: Path, fmt: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    if fmt == "auto":
        suf = path.suffix.lower()
        if suf == ".jsonl":
            fmt = "jsonl"
        elif suf == ".parquet":
            kind = _parquet_loader_kind(path)
            fmt = {
                "dapo": "dapo_parquet",
                "amo_qa": "amo_qa_parquet",
                "problem_answer": "problem_answer_parquet",
            }[kind]
        else:
            raise ValueError(f"Cannot auto-detect format for suffix {suf}; set --data-format")
    if fmt == "jsonl":
        return load_jsonl_examples(path, limit)
    if fmt == "dapo_parquet":
        return load_dapo_parquet_examples(path, limit)
    if fmt == "amo_qa_parquet":
        return load_amo_qa_parquet_examples(path, limit)
    if fmt == "problem_answer_parquet":
        return load_problem_answer_parquet_examples(path, limit)
    raise ValueError(f"Unknown --data-format: {fmt}")


def build_llm(
    model_path: str,
    lora_path: Optional[str],
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
) -> Any:
    from vllm import LLM

    cfg: Dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    if enforce_eager:
        cfg["enforce_eager"] = True
    if lora_path:
        adapter_st = Path(lora_path) / "adapter_model.safetensors"
        adapter_bin = Path(lora_path) / "adapter_model.bin"
        if adapter_st.is_file() or adapter_bin.is_file():
            cfg["enable_lora"] = True
            cfg["max_lora_rank"] = 64
            cfg["max_loras"] = 1
            cfg["max_cpu_loras"] = 1
        else:
            print(f"[warn] No adapter weights under {lora_path}; running without LoRA flags.")
            lora_path = None
    return LLM(**cfg)


def default_data_root() -> Path:
    return Path(__file__).resolve().parent / "data"


# Aliases (lowercase, hyphen) -> path relative to data root
_DATASET_REL_PATH: Dict[str, str] = {}
for _aliases, _rel in (
    (("aime24",), "AIME24/test.parquet"),
    (("aime25",), "AIME25/test.parquet"),
    (("aime26",), "AIME26/test.parquet"),
    (("aime26-jsonl", "aime26-json"), "AIME26/aime2026.jsonl"),
    (("amc23",), "AMC23/test.parquet"),
    (("amo-bench", "amo_bench", "amobench"), "AMO-Bench/test.parquet"),
    (("brumo25",), "BRUMO25/test.parquet"),
    (("cmimc25",), "CMIMC25/test.parquet"),
    (("dapo", "dapo-math", "dapo-math-17k", "dapo17k"), "dapo-math-17k.parquet"),
    (("hmmt25", "hmmt-25"), "HMMT25/test.parquet"),
    (("math500", "math-500"), "MATH-500/test.parquet"),
    (("minerva",), "Minerva/test.parquet"),
    (("olympiad", "olympiad-bench", "olympiad_bench"), "Olympiad-Bench/test.parquet"),
):
    for _a in _aliases:
        _DATASET_REL_PATH[_a] = _rel


def normalize_dataset_key(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def resolve_dataset_path(name: str, data_root: Path) -> Path:
    key = normalize_dataset_key(name)
    rel = _DATASET_REL_PATH.get(key)
    if rel is None:
        known = ", ".join(sorted(set(_DATASET_REL_PATH.keys())))
        raise SystemExit(f"Unknown dataset {name!r}. Known aliases: {known}\n  (--data-root={data_root})")
    p = (data_root / rel).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Dataset {name!r} -> expected file missing: {p}")
    return p


def summarize_result_subset(
    rows: List[Dict[str, Any]],
    pass_at_k_list: List[int],
    gen_n: int,
) -> Dict[str, Any]:
    n_d = len(rows)
    pass_at_k: Dict[str, Dict[str, Any]] = {}
    for k in pass_at_k_list:
        c = sum(1 for r in rows if r.get("pass_at_k", {}).get(str(k)))
        pass_at_k[str(k)] = {
            "count": c,
            "total": n_d,
            "pct": 100.0 * c / n_d if n_d else 0.0,
        }
    maj = sum(1 for r in rows if r.get("majority_vote_correct"))
    total_sol = n_d * gen_n
    fmt = sum(sum(1 for g in r.get("generations", []) if g.get("formatted")) for r in rows)
    tot_correct = sum(r.get("num_correct", 0) for r in rows)
    return {
        "num_problems": n_d,
        "pass_at_k": pass_at_k,
        "majority_vote_pct": 100.0 * maj / n_d if n_d else 0.0,
        "average_correct_pct": 100.0 * tot_correct / total_sol if total_sol else 0.0,
        "format_rate_pct": 100.0 * fmt / total_sol if total_sol else 0.0,
    }


def parse_pass_at_k(s: str) -> List[int]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not parts:
        return [1]
    out = sorted(set(parts))
    if any(k < 1 for k in out):
        raise ValueError("--pass-at-k values must be positive integers")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Local math eval (vLLM + math_verify + boxed)")
    parser.add_argument("--model-path", type=str, default="", help="Base model dir (not needed for --list-datasets).")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Dataset name under --data-root (default: ./data). Repeat for multiple. "
            "Run with --list-datasets to see names (e.g. aime26, math500, dapo)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Root directory for --dataset (default: <this_repo>/preference_learning/data).",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        metavar="PATH",
        help="Explicit file path; repeat for multiple. Combined with resolved --dataset entries.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print known --dataset names and exit.",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="auto",
        choices=["auto", "jsonl", "dapo_parquet", "amo_qa_parquet", "problem_answer_parquet"],
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="LoRA adapter directory (optional)")
    parser.add_argument("--output-json", type=str, default="", help="Summary JSON path (not needed for --list-datasets).")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = use all rows")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--val-n",
        type=int,
        default=1,
        help="Samples per problem (vLLM n). Raised automatically to max(pass-at-k) if smaller.",
    )
    parser.add_argument(
        "--pass-at-k",
        type=str,
        default="1,4,8,16",
        help="Comma-separated k for pass@k (any correct in first k samples). Written to output JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="0 = auto (40960 if thinking else 32768)",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=0,
        help=(
            "Number of **problems** (prompts) per llm.generate() call. "
            "0 = one call with all prompts. "
            "Use 8–32 to cap concurrent prefill/decode batches (vLLM still schedules internally)."
        ),
    )
    args = parser.parse_args()

    if args.list_datasets:
        root = Path(args.data_root).expanduser().resolve() if args.data_root.strip() else default_data_root()
        print(f"data_root: {root}\n")
        by_rel: Dict[str, List[str]] = {}
        for alias, rel in _DATASET_REL_PATH.items():
            by_rel.setdefault(rel, []).append(alias)
        for rel in sorted(by_rel.keys()):
            aliases = ", ".join(sorted(set(by_rel[rel])))
            p = root / rel
            ok = "ok" if p.is_file() else "MISSING"
            print(f"  [{ok}] {rel}")
            print(f"       names: {aliases}")
        raise SystemExit(0)

    if not args.model_path.strip():
        raise SystemExit("error: --model-path is required (unless using --list-datasets)")
    if not args.output_json.strip():
        raise SystemExit("error: --output-json is required (unless using --list-datasets)")

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root.strip()
        else default_data_root()
    )

    load_queue: List[tuple[str, Optional[str]]] = []
    if args.dataset:
        for dn in args.dataset:
            p = resolve_dataset_path(dn, data_root)
            load_queue.append((str(p), normalize_dataset_key(dn)))
    if args.data_path:
        for raw in args.data_path:
            load_queue.append((raw, None))
    if not load_queue:
        raise SystemExit(
            "error: provide at least one --dataset and/or --data-path "
            "(or run with --list-datasets)"
        )

    pass_at_k_list = parse_pass_at_k(args.pass_at_k)
    max_k = max(pass_at_k_list)
    gen_n = max(args.val_n, max_k)
    if gen_n != args.val_n:
        print(f"[eval] val-n {args.val_n} < max(pass-at-k)={max_k}; generating n={gen_n} samples per problem.")

    if not _HAS_MATH_VERIFY:
        print("[warn] math_verify not installed; grading falls back to normalized string equality.")
        print("       pip install math-verify")

    limit = args.num_samples if args.num_samples > 0 else None
    resolved_paths: List[Path] = []
    examples: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = {}
    for raw, tag_override in load_queue:
        data_path = Path(raw).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        resolved_paths.append(data_path)
        batch = load_examples(data_path, args.data_format, limit)
        base_tag = tag_override if tag_override is not None else data_path.stem
        tag_counts[base_tag] = tag_counts.get(base_tag, 0) + 1
        tag = base_tag if tag_counts[base_tag] == 1 else f"{base_tag}_{tag_counts[base_tag]}"
        for ex in batch:
            ex["dataset_tag"] = tag
            ex["dataset_path"] = str(data_path)
        examples.extend(batch)
        print(f"[eval] +{len(batch)} problems from {data_path} (tag={tag})")

    if not examples:
        raise RuntimeError("No examples loaded; check --data-path and format.")

    max_model_len = args.max_model_len
    if max_model_len <= 0:
        max_model_len = 40960 if args.enable_thinking else 32768

    print(f"[eval] total {len(examples)} problems from {len(resolved_paths)} file(s)")
    print(f"[eval] math_verify={'yes' if _HAS_MATH_VERIFY else 'no'}, thinking={args.enable_thinking}")

    llm = build_llm(
        args.model_path,
        args.checkpoint_dir,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        max_model_len,
        args.enforce_eager,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    lora_request = None
    if args.checkpoint_dir:
        p = Path(args.checkpoint_dir)
        if (p / "adapter_model.safetensors").is_file() or (p / "adapter_model.bin").is_file():
            try:
                from vllm.lora.request import LoRARequest

                lora_request = LoRARequest("eval_lora", 1, str(p.resolve()))
                print(f"[eval] LoRARequest -> {p}")
            except Exception as e:
                print(f"[warn] LoRA disabled: {e}")

    user_suffix = (
        "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    all_prompts: List[str] = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["problem"] + user_suffix}]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            kwargs["enable_thinking"] = args.enable_thinking
            text = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            text = tokenizer.apply_chat_template(messages, **kwargs)
        all_prompts.append(text)

    sp_kw: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "max_tokens": args.max_new_tokens,
        "n": gen_n,
        "seed": args.seed,
    }
    if args.top_k > 0:
        sp_kw["top_k"] = args.top_k
    if args.presence_penalty != 0.0:
        sp_kw["presence_penalty"] = args.presence_penalty

    from vllm import SamplingParams

    sampling_params = SamplingParams(**sp_kw)

    n_prompts = len(all_prompts)
    gbs = args.generate_batch_size
    if gbs <= 0:
        gbs = n_prompts
    print(
        f"[eval] generating {n_prompts} prompts x n={gen_n} "
        f"(pass-at-k={pass_at_k_list}, generate_batch_size={gbs}) ..."
    )
    outputs: List[Any] = []
    use_inner_tqdm = n_prompts <= gbs
    for start in tqdm(
        range(0, n_prompts, gbs),
        desc="prompt_batches",
        dynamic_ncols=True,
        disable=n_prompts <= gbs,
    ):
        end = min(start + gbs, n_prompts)
        chunk = all_prompts[start:end]
        if lora_request is not None:
            part = llm.generate(
                chunk,
                sampling_params,
                lora_request=lora_request,
                use_tqdm=use_inner_tqdm,
            )
        else:
            part = llm.generate(chunk, sampling_params, use_tqdm=use_inner_tqdm)
        outputs.extend(part)
    if len(outputs) != n_prompts:
        raise RuntimeError(f"expected {n_prompts} vLLM outputs, got {len(outputs)}")

    results: List[Dict[str, Any]] = []
    pass_at_k_counts: Dict[int, int] = {k: 0 for k in pass_at_k_list}
    formatted_total = 0
    total_solutions = 0
    total_correct = 0
    majority_correct = 0

    for ex, output in tqdm(
        zip(examples, outputs),
        total=len(examples),
        desc="grade",
        dynamic_ncols=True,
    ):
        gt = ex["ground_truth"]
        generations: List[str] = []
        preds: List[str] = []
        correct_flags: List[bool] = []
        formatted_flags: List[bool] = []

        for o in output.outputs:
            gen = o.text
            generations.append(gen)
            pred = extract_boxed_answer(gen)
            formatted = pred is not None
            if pred is None:
                preds.append("[no boxed]")
            else:
                preds.append(pred)
            ok = grade_answer(pred, gt)
            correct_flags.append(ok)
            formatted_flags.append(formatted)
            total_solutions += 1
            if formatted:
                formatted_total += 1
            if ok:
                total_correct += 1

        pass_at_k_problem: Dict[str, bool] = {}
        for k in pass_at_k_list:
            ok_k = any(correct_flags[:k])
            pass_at_k_problem[str(k)] = ok_k
            if ok_k:
                pass_at_k_counts[k] += 1

        maj_ok = False
        fpreds = [p for p, f in zip(preds, formatted_flags) if f]
        if fpreds:
            top = Counter(fpreds).most_common(1)[0][0]
            maj_ok = grade_answer(top, gt)
        if maj_ok:
            majority_correct += 1

        results.append(
            {
                "dataset_tag": ex.get("dataset_tag", ""),
                "dataset_path": ex.get("dataset_path", ""),
                "problem_id": ex["id"],
                "problem": ex["problem"],
                "ground_truth": gt,
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

    n = len(examples)
    pass_at_k_summary: Dict[str, Dict[str, Any]] = {}
    for k in pass_at_k_list:
        c = pass_at_k_counts[k]
        pass_at_k_summary[str(k)] = {
            "count": c,
            "total": n,
            "pct": 100.0 * c / n if n else 0.0,
        }

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_tag[str(r.get("dataset_tag", ""))].append(r)

    metrics_by_dataset: Dict[str, Any] = {}
    for tag, sub in sorted(by_tag.items(), key=lambda x: x[0]):
        path0 = sub[0].get("dataset_path", "") if sub else ""
        metrics_by_dataset[tag] = {
            "dataset_path": path0,
            **summarize_result_subset(sub, pass_at_k_list, gen_n),
        }

    summary = {
        "model_path": args.model_path,
        "checkpoint_dir": args.checkpoint_dir,
        "data_root": str(data_root),
        "data_paths": [str(p) for p in resolved_paths],
        "dataset_args": list(args.dataset) if args.dataset else [],
        "data_format": args.data_format,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "val_n_requested": args.val_n,
        "gen_n": gen_n,
        "pass_at_k_list": pass_at_k_list,
        "pass_at_k": pass_at_k_summary,
        "metrics_by_dataset": metrics_by_dataset,
        "num_problems": n,
        "total_solutions": total_solutions,
        "average_correct_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
        "majority_vote_pct": 100.0 * majority_correct / n if n else 0.0,
        "format_rate_pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
        "math_verify": _HAS_MATH_VERIFY,
        "generate_batch_size": gbs if args.generate_batch_size > 0 else n_prompts,
        "generate_batch_size_requested": args.generate_batch_size,
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("[ALL] combined")
    for k in pass_at_k_list:
        s = pass_at_k_summary[str(k)]
        print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{n})")
    for tag, m in metrics_by_dataset.items():
        print(f"[{tag}] n={m['num_problems']}")
        for k in pass_at_k_list:
            s = m["pass_at_k"][str(k)]
            print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{m['num_problems']})")
    print(f"Avg correct / sample: {summary['average_correct_pct']:.2f}%")
    print(f"Majority vote: {summary['majority_vote_pct']:.2f}%")
    print(f"Boxed format rate: {summary['format_rate_pct']:.2f}%")
    print(f"Wrote {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
