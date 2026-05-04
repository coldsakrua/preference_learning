#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from utils import extract_user_prompt

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


def max_seq_len_from_model_config(model_path: str) -> Optional[int]:
    """Return max context from config.json (vLLM refuses max_model_len above this unless env override)."""
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        m = getattr(cfg, "max_position_embeddings", None)
        if m is None:
            m = getattr(cfg, "model_max_length", None)
        return int(m) if m is not None else None
    except Exception:
        return None


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


def _extract_gsm8k_final_answer(answer_text: str) -> str:
    """
    Extract the final GSM8K answer.
    Typical format ends with: '#### 72'
    """
    text = str(answer_text or "").strip()
    if not text:
        return ""
    m = re.search(r"####\s*(.+?)\s*$", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: use last non-empty line if no #### marker exists.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def load_gsm8k_hf_examples(
    path: Path,
    limit: Optional[int],
    gsm8k_config: str = "main",
    gsm8k_split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load GSM8K from a HuggingFace-datasets style directory.
    Expected fields: question, answer
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Loading GSM8K directory requires `datasets`. Install with: pip install datasets"
        ) from e

    ds = load_dataset(str(path), gsm8k_config, split=gsm8k_split)
    rows: List[Dict[str, Any]] = []
    for i, o in enumerate(ds):
        problem = str(o.get("question", "")).strip()
        gt = _extract_gsm8k_final_answer(str(o.get("answer", "")))
        if not problem or not gt:
            continue
        sid = str(o.get("id", i)).strip()
        rows.append({"id": sid, "problem": problem, "ground_truth": gt})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _choice_label(idx: int) -> str:
    return chr(ord("A") + idx)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def load_mmlu_pro_hf_examples(
    path: Path,
    limit: Optional[int],
    mmlu_pro_config: str = "default",
    mmlu_pro_split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load MMLU-Pro from a HuggingFace-datasets style directory.
    Expected fields: question, options, answer_index (and/or answer)
    """
    rows: List[Dict[str, Any]] = []

    # 1) Prefer direct local parquet loading to avoid dataset_infos.json incompatibilities.
    split_key = mmlu_pro_split.strip()
    parquet_patterns = [
        f"{split_key}-*.parquet",
        f"{split_key}*.parquet",
    ]
    parquet_files: List[Path] = []
    for base in (path / "data", path):
        if not base.exists():
            continue
        for pat in parquet_patterns:
            parquet_files.extend(sorted(base.glob(pat)))
    if parquet_files:
        print(
            f"[eval] loading local MMLU-Pro split={split_key} from parquet files: {len(parquet_files)}",
            flush=True,
        )
        for pf_path in parquet_files:
            pf = pq.ParquetFile(pf_path)
            cols = ["question", "options", "answer", "answer_index", "category", "question_id"]
            existing_cols = [c for c in cols if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=512, columns=existing_cols):
                pyd = batch.to_pydict()
                n = len(next(iter(pyd.values()))) if pyd else 0
                for i in range(n):
                    o = {k: pyd[k][i] for k in pyd}
                    question = str(o.get("question", "")).strip()
                    raw_options = o.get("options", [])
                    if not isinstance(raw_options, list):
                        raw_options = list(raw_options) if raw_options is not None else []
                    options = [str(x).strip() for x in raw_options if str(x).strip()]
                    if not question or not options:
                        continue

                    answer_index_raw = o.get("answer_index", None)
                    answer_text_raw = str(o.get("answer", "")).strip()
                    gt_idx: Optional[int] = None
                    try:
                        if answer_index_raw is not None:
                            gt_idx = int(answer_index_raw)
                    except Exception:
                        gt_idx = None
                    if gt_idx is None and answer_text_raw:
                        ans_norm = _normalize_text(answer_text_raw)
                        for oi, opt in enumerate(options):
                            if _normalize_text(opt) == ans_norm:
                                gt_idx = oi
                                break
                    if gt_idx is None or gt_idx < 0 or gt_idx >= len(options):
                        continue

                    gt_letter = _choice_label(gt_idx)
                    option_lines = [f"{_choice_label(oi)}. {opt}" for oi, opt in enumerate(options)]
                    prompt_text = question + "\n\nOptions:\n" + "\n".join(option_lines)
                    sid = str(o.get("question_id", len(rows))).strip()
                    category = str(o.get("category", "")).strip()
                    rows.append(
                        {
                            "id": sid,
                            "problem": prompt_text,
                            "ground_truth": gt_letter,
                            "ground_truth_choice": gt_letter,
                            "ground_truth_text": options[gt_idx],
                            "options": options,
                            "category": category,
                            "eval_type": "mcq",
                        }
                    )
                    if limit is not None and len(rows) >= limit:
                        return rows
        if rows:
            return rows

    # 2) Fallback to datasets local directory/hub.
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Loading MMLU-Pro directory requires `datasets`. Install with: pip install datasets"
        ) from e

    ds = None
    load_err: Optional[Exception] = None
    try:
        ds = load_dataset(str(path), mmlu_pro_config, split=mmlu_pro_split)
    except Exception as e:
        load_err = e
        print(
            f"[warn] Failed to load local MMLU-Pro via datasets from {path}: {e}\n"
            "[warn] Falling back to HuggingFace hub dataset: TIGER-Lab/MMLU-Pro",
            flush=True,
        )
        ds = load_dataset("TIGER-Lab/MMLU-Pro", mmlu_pro_config, split=mmlu_pro_split)
    if ds is None:
        raise RuntimeError(
            f"Unable to load MMLU-Pro split={mmlu_pro_split} config={mmlu_pro_config} "
            f"from local path {path} or hub fallback."
        ) from load_err
    for i, o in enumerate(ds):
        question = str(o.get("question", "")).strip()
        raw_options = o.get("options", [])
        if not isinstance(raw_options, list):
            raw_options = list(raw_options) if raw_options is not None else []
        options = [str(x).strip() for x in raw_options if str(x).strip()]
        if not question or not options:
            continue

        answer_index_raw = o.get("answer_index", None)
        answer_text_raw = str(o.get("answer", "")).strip()

        gt_idx: Optional[int] = None
        try:
            if answer_index_raw is not None:
                gt_idx = int(answer_index_raw)
        except Exception:
            gt_idx = None
        if gt_idx is None and answer_text_raw:
            ans_norm = _normalize_text(answer_text_raw)
            for oi, opt in enumerate(options):
                if _normalize_text(opt) == ans_norm:
                    gt_idx = oi
                    break
        if gt_idx is None or gt_idx < 0 or gt_idx >= len(options):
            continue

        gt_letter = _choice_label(gt_idx)
        option_lines = [f"{_choice_label(oi)}. {opt}" for oi, opt in enumerate(options)]
        prompt_text = question + "\n\nOptions:\n" + "\n".join(option_lines)

        sid = str(o.get("question_id", i)).strip()
        category = str(o.get("category", "")).strip()
        rows.append(
            {
                "id": sid,
                "problem": prompt_text,
                "ground_truth": gt_letter,
                "ground_truth_choice": gt_letter,
                "ground_truth_text": options[gt_idx],
                "options": options,
                "category": category,
                "eval_type": "mcq",
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
    if "question" in names and "answer" in names:
        # Keep GSM8K parquet files compatible with the generic problem+answer loader.
        return "problem_answer"
    if "prompt" in names and "answer" in names:
        pt = schema.field("prompt").type
        if pa.types.is_string(pt) or pa.types.is_large_string(pt):
            return "amo_qa"
    raise ValueError(
        f"Unsupported parquet schema in {path}; "
        f"need DAPO (reward_model+…), or string prompt+answer, or problem/question+answer. "
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
    text_col = "problem" if "problem" in names else "question" if "question" in names else None
    if text_col is None:
        raise ValueError(f"{path} has no 'problem' or 'question' column.")
    id_col: Optional[str] = None
    if "problem_idx" in names:
        id_col = "problem_idx"
    elif "id" in names:
        id_col = "id"
    cols = [text_col, "answer"]
    if id_col:
        cols = [id_col, text_col, "answer"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        if id_col:
            ids = batch.column(id_col).to_pylist()
        else:
            ids = None
        problems = batch.column(text_col).to_pylist()
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


def load_examples(
    path: Path,
    fmt: str,
    limit: Optional[int],
    gsm8k_config: str = "main",
    gsm8k_split: str = "test",
    mmlu_pro_config: str = "default",
    mmlu_pro_split: str = "test",
) -> List[Dict[str, Any]]:
    if fmt == "auto" and path.is_dir():
        has_hf_meta = (path / "dataset_infos.json").is_file() and (path / "README.md").is_file()
        has_gsm8k_meta = has_hf_meta and (path / "dataset_info.json").is_file()
        has_mmlu_pro_meta = has_hf_meta and "mmlu-pro" in path.name.lower()
        if has_gsm8k_meta:
            fmt = "gsm8k_hf"
        elif has_mmlu_pro_meta:
            fmt = "mmlu_pro_hf"
        else:
            raise ValueError(
                f"Cannot auto-detect format for directory {path}; set --data-format explicitly."
            )

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
    if fmt == "gsm8k_hf":
        return load_gsm8k_hf_examples(path, limit, gsm8k_config=gsm8k_config, gsm8k_split=gsm8k_split)
    if fmt == "mmlu_pro_hf":
        return load_mmlu_pro_hf_examples(
            path,
            limit,
            mmlu_pro_config=mmlu_pro_config,
            mmlu_pro_split=mmlu_pro_split,
        )
    raise ValueError(f"Unknown --data-format: {fmt}")


def _adapter_dir_has_weights(d: Path) -> bool:
    return (d / "adapter_model.safetensors").is_file() or (d / "adapter_model.bin").is_file()


def _adapter_config_base_model(adapter_dir: Path) -> Optional[str]:
    p = adapter_dir / "adapter_config.json"
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta.get("base_model_name_or_path") or meta.get("base_model_name")


def _infer_max_lora_rank_from_adapter(adapter_dir: Path, fallback: int) -> int:
    p = adapter_dir / "adapter_config.json"
    if not p.is_file():
        return fallback
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        r = int(meta.get("r", fallback))
        return max(r, 1)
    except Exception:
        return fallback


def _is_peft_adapter_dir(d: Path) -> bool:
    return _adapter_dir_has_weights(d) and (d / "adapter_config.json").is_file()


def resolve_user_lora_dir(raw: Optional[str]) -> Optional[Path]:
    """
    Turn CLI LoRA path into the directory that actually holds adapter_config + weights.

    If ``raw`` is a training output parent (e.g. .../train) without weights at the root,
    use ``final/`` or ``lora_adapter/`` when present (matches train_preference.py layout).
    """
    if raw is None or not str(raw).strip():
        return None
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        return p
    if _is_peft_adapter_dir(p):
        return p
    for sub in ("final", "lora_adapter"):
        c = p / sub
        if _is_peft_adapter_dir(c):
            print(f"[eval] LoRA path {p} -> using adapter at {c}")
            return c
    return p


def resolve_vllm_base_and_lora(
    model_path: str,
    checkpoint_dir: Optional[str],
) -> tuple[str, Optional[Path]]:
    """
    Returns (vLLM base model path, optional LoRA adapter directory).

    If --checkpoint-dir / --lora-path is set, --model-path is the base model and that option is the adapter
    (after resolve_user_lora_dir, e.g. .../train -> .../train/final).
    Else if --model-path points to a PEFT adapter (adapter_config + adapter weights), base is read from
    adapter_config.json and the same path is the LoRA directory.
    """
    mp = Path(model_path).expanduser().resolve()
    ckpt_raw = checkpoint_dir.strip() if checkpoint_dir and str(checkpoint_dir).strip() else ""
    if ckpt_raw:
        return str(mp), Path(ckpt_raw).expanduser().resolve()
    if _is_peft_adapter_dir(mp):
        base_raw = _adapter_config_base_model(mp)
        if not base_raw:
            raise SystemExit(
                f"error: {mp} looks like a LoRA adapter but adapter_config.json "
                "has no base_model_name_or_path"
            )
        bpath = Path(base_raw)
        if bpath.exists():
            base_resolved = str(bpath.expanduser().resolve())
        else:
            base_resolved = base_raw
        return base_resolved, mp
    return str(mp), None


def build_llm(
    model_path: str,
    lora_path: Optional[str],
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    disable_custom_all_reduce: bool,
    max_lora_rank: int,
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
        "disable_custom_all_reduce": disable_custom_all_reduce,
    }
    if enforce_eager:
        cfg["enforce_eager"] = True
    if lora_path:
        adapter_st = Path(lora_path) / "adapter_model.safetensors"
        adapter_bin = Path(lora_path) / "adapter_model.bin"
        if adapter_st.is_file() or adapter_bin.is_file():
            cfg["enable_lora"] = True
            cfg["max_lora_rank"] = max_lora_rank
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
    (("gsm8k",), "gsm8k/socratic/test-00000-of-00001.parquet"),
    (("mmlu-pro", "mmlu_pro", "mmlupro"), "mmlu-pro"),
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
    if not p.exists():
        raise FileNotFoundError(f"Dataset {name!r} -> expected path missing: {p}")
    return p


def extract_mcq_answer(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed:
        m = re.search(r"\b([A-J])\b", boxed.upper())
        if m:
            return m.group(1)

    patterns = [
        r"(?:final answer|answer|correct option|option)\s*[:：]\s*\(?\s*([A-J])\s*\)?",
        r"\b([A-J])\b(?=\s*(?:\.|,|:|$))",
        r"\(([A-J])\)",
    ]
    up = text.upper()
    for pat in patterns:
        m = re.search(pat, up)
        if m:
            return m.group(1)
    return None


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
    avg1_pct = pass_at_k.get("1", {}).get("pct", 0.0)
    avg16_pct = 100.0 * tot_correct / total_sol if total_sol else 0.0
    return {
        "num_problems": n_d,
        "pass_at_k": pass_at_k,
        "avg1_pct": avg1_pct,
        "avg16_pct": avg16_pct,
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
        choices=[
            "auto",
            "jsonl",
            "dapo_parquet",
            "amo_qa_parquet",
            "problem_answer_parquet",
            "gsm8k_hf",
            "mmlu_pro_hf",
        ],
    )
    parser.add_argument(
        "--gsm8k-config",
        type=str,
        default="main",
        choices=["main", "socratic"],
        help="Used when --data-format=gsm8k_hf (or auto-detected gsm8k directory).",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="test",
        help="Used when --data-format=gsm8k_hf (default: test).",
    )
    parser.add_argument(
        "--mmlu-pro-config",
        type=str,
        default="default",
        help="Used when --data-format=mmlu_pro_hf (default: default).",
    )
    parser.add_argument(
        "--mmlu-pro-split",
        type=str,
        default="test",
        help="Used when --data-format=mmlu_pro_hf (default: test).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--lora-path",
        dest="lora_arg",
        type=str,
        default=None,
        help=(
            "PEFT LoRA adapter directory (optional). Alias: --lora-path. "
            "May be train output root (.../train); then uses final/ or lora_adapter/ if weights are there."
        ),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=0,
        help=(
            "vLLM max_lora_rank when using a LoRA adapter. "
            "0 = use r from adapter_config.json (fallback 64). "
            "If set below adapter r, it is raised automatically."
        ),
    )
    parser.add_argument("--output-json", type=str, default="", help="Summary JSON path (not needed for --list-datasets).")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = use all rows")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 = auto by mode (thinking=38912, non-thinking=32768)")
    parser.add_argument("--temperature", type=float, default=-1.0, help="<0 = auto by mode (thinking=0.6, non-thinking=0.7)")
    parser.add_argument("--top-p", type=float, default=-1.0, help="<0 = auto by mode (thinking=0.95, non-thinking=0.8)")
    parser.add_argument("--top-k", type=int, default=20, help="Set to 20 per Qwen3 official recommendation.")
    parser.add_argument("--min-p", type=float, default=0.0, help="Set to 0 per Qwen3 official recommendation.")
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
        "--disable-custom-all-reduce",
        action="store_true",
        default=False,
        help="Disable vLLM custom all-reduce and fall back to NCCL for tensor parallel inference.",
    )
    parser.add_argument(
        "--force-base-tokenizer",
        action="store_true",
        default=False,
        help=(
            "Always load tokenizer/chat template from --model-path (vLLM base), "
            "even when LoRA adapter dir contains tokenizer files."
        ),
    )
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
            ok = "ok" if p.exists() else "MISSING"
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
        if args.data_format in {"gsm8k_hf", "mmlu_pro_hf"}:
            if not data_path.exists():
                raise FileNotFoundError(data_path)
        else:
            if not data_path.is_file():
                raise FileNotFoundError(data_path)
        resolved_paths.append(data_path)
        if args.data_format == "gsm8k_hf":
            batch = load_gsm8k_hf_examples(data_path, limit, gsm8k_config=args.gsm8k_config, gsm8k_split=args.gsm8k_split)
        elif args.data_format == "mmlu_pro_hf":
            batch = load_mmlu_pro_hf_examples(
                data_path,
                limit,
                mmlu_pro_config=args.mmlu_pro_config,
                mmlu_pro_split=args.mmlu_pro_split,
            )
        else:
            batch = load_examples(
                data_path,
                args.data_format,
                limit,
                gsm8k_config=args.gsm8k_config,
                gsm8k_split=args.gsm8k_split,
                mmlu_pro_config=args.mmlu_pro_config,
                mmlu_pro_split=args.mmlu_pro_split,
            )
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
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else (38912 if args.enable_thinking else 32768)
    temperature = args.temperature if args.temperature >= 0 else (0.6 if args.enable_thinking else 0.7)
    top_p = args.top_p if args.top_p >= 0 else (0.95 if args.enable_thinking else 0.8)
    top_k = max(args.top_k, 0)
    min_p = max(args.min_p, 0.0)
    presence_penalty = args.presence_penalty

    print(f"[eval] total {len(examples)} problems from {len(resolved_paths)} file(s)")
    print(f"[eval] math_verify={'yes' if _HAS_MATH_VERIFY else 'no'}, thinking={args.enable_thinking}")

    lora_dir_cli = resolve_user_lora_dir(args.lora_arg)
    vllm_model_path, lora_dir = resolve_vllm_base_and_lora(
        args.model_path,
        str(lora_dir_cli) if lora_dir_cli is not None else None,
    )
    lora_dir_str = str(lora_dir) if lora_dir is not None else None

    max_lora_rank_cli = args.max_lora_rank
    if lora_dir is not None and _adapter_dir_has_weights(lora_dir):
        inferred_r = _infer_max_lora_rank_from_adapter(lora_dir, 64)
        if max_lora_rank_cli <= 0:
            max_lora_rank = inferred_r
        else:
            max_lora_rank = max(max_lora_rank_cli, inferred_r)
            if max_lora_rank_cli < inferred_r:
                print(
                    f"[warn] --max-lora-rank {max_lora_rank_cli} < adapter r={inferred_r}; "
                    f"using {max_lora_rank} for vLLM."
                )
        print(f"[eval] vLLM base={vllm_model_path} LoRA={lora_dir} max_lora_rank={max_lora_rank}")
    else:
        max_lora_rank = max_lora_rank_cli if max_lora_rank_cli > 0 else 64
        if lora_dir is not None and not _adapter_dir_has_weights(lora_dir):
            print(f"[warn] LoRA dir {lora_dir} has no adapter weights; eval runs base model only.")
        print(f"[eval] vLLM model={vllm_model_path}")

    cfg_max = max_seq_len_from_model_config(vllm_model_path)
    if cfg_max is not None and max_model_len > cfg_max:
        print(f"[eval] capping max_model_len {max_model_len} -> {cfg_max} (base model max_position_embeddings)")
        max_model_len = cfg_max

    llm = build_llm(
        vllm_model_path,
        lora_dir_str,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        max_model_len,
        args.enforce_eager,
        args.disable_custom_all_reduce,
        max_lora_rank,
    )

    tokenizer_src = vllm_model_path
    if (
        not args.force_base_tokenizer
        and lora_dir is not None
        and (lora_dir / "tokenizer_config.json").is_file()
    ):
        tokenizer_src = str(lora_dir.resolve())
    print(f"[eval] tokenizer_source={tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

    lora_request = None
    if lora_dir is not None and _adapter_dir_has_weights(lora_dir):
        try:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("eval_lora", 1, str(lora_dir.resolve()))
            print(f"[eval] LoRARequest -> {lora_dir}")
        except Exception as e:
            print(f"[warn] LoRA disabled: {e}")

    all_prompts: List[str] = []
    for ex in examples:
        eval_type = str(ex.get("eval_type", "boxed_math"))
        if eval_type == "mcq":
            user_suffix = (
                "\n\nPlease reason step by step and provide the final answer as a single capital letter "
                "(A, B, C, ...), wrapped in \\boxed{}."
            )
        else:
            user_suffix = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
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
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "max_tokens": max_new_tokens,
        "n": gen_n,
        "seed": args.seed,
    }
    if top_k > 0:
        sp_kw["top_k"] = top_k
    if presence_penalty != 0.0:
        sp_kw["presence_penalty"] = presence_penalty

    from vllm import SamplingParams

    sampling_params = SamplingParams(**sp_kw)

    n_prompts = len(all_prompts)
    gbs = args.generate_batch_size
    if gbs <= 0:
        gbs = n_prompts
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval] generating {n_prompts} prompts x n={gen_n} "
        f"(pass-at-k={pass_at_k_list}, generate_batch_size={gbs}) ..."
    )

    results: List[Dict[str, Any]] = []
    pass_at_k_counts: Dict[int, int] = {k: 0 for k in pass_at_k_list}
    formatted_total = 0
    total_solutions = 0
    total_correct = 0
    majority_correct = 0
    processed = 0

    use_inner_tqdm = n_prompts <= gbs
    for start in tqdm(
        range(0, n_prompts, gbs),
        desc="prompt_batches",
        dynamic_ncols=True,
        disable=n_prompts <= gbs,
    ):
        end = min(start + gbs, n_prompts)
        chunk_prompts = all_prompts[start:end]
        chunk_examples = examples[start:end]
        if lora_request is not None:
            chunk_outputs = llm.generate(
                chunk_prompts,
                sampling_params,
                lora_request=lora_request,
                use_tqdm=use_inner_tqdm,
            )
        else:
            chunk_outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=use_inner_tqdm)
        if len(chunk_outputs) != len(chunk_examples):
            raise RuntimeError(
                f"expected {len(chunk_examples)} vLLM outputs in batch, got {len(chunk_outputs)}"
            )

        for ex, output in zip(chunk_examples, chunk_outputs):
            gt = ex["ground_truth"]
            generations: List[str] = []
            preds: List[str] = []
            correct_flags: List[bool] = []
            formatted_flags: List[bool] = []

            for o in output.outputs:
                gen = o.text
                generations.append(gen)
                eval_type = str(ex.get("eval_type", "boxed_math"))
                if eval_type == "mcq":
                    pred = extract_mcq_answer(gen)
                    formatted = pred is not None
                    gt_choice = str(ex.get("ground_truth_choice", ex["ground_truth"])).upper().strip()
                    ok = bool(pred is not None and pred.upper() == gt_choice)
                else:
                    pred = extract_boxed_answer(gen)
                    formatted = pred is not None
                    ok = grade_answer(pred, gt)
                if pred is None:
                    preds.append("[no boxed]")
                else:
                    preds.append(pred)
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
                    "category": ex.get("category", ""),
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

        processed = len(results)
        pass_at_k_summary: Dict[str, Dict[str, Any]] = {}
        for k in pass_at_k_list:
            c = pass_at_k_counts[k]
            pass_at_k_summary[str(k)] = {
                "count": c,
                "total": processed,
                "pct": 100.0 * c / processed if processed else 0.0,
            }
        avg1_pct = pass_at_k_summary.get("1", {}).get("pct", 0.0)
        avg16_pct = 100.0 * total_correct / total_solutions if total_solutions else 0.0

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

        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            cat = str(r.get("category", "")).strip()
            if not cat:
                cat = "__uncategorized__"
            by_category[cat].append(r)
        metrics_by_category: Dict[str, Any] = {}
        for cat, sub in sorted(by_category.items(), key=lambda x: x[0]):
            metrics_by_category[cat] = summarize_result_subset(sub, pass_at_k_list, gen_n)

        summary = {
            "model_path": args.model_path,
            "vllm_base_model_path": vllm_model_path,
            "checkpoint_dir": args.lora_arg,
            "lora_adapter_dir": lora_dir_str,
            "max_lora_rank": max_lora_rank,
            "data_root": str(data_root),
            "data_paths": [str(p) for p in resolved_paths],
            "dataset_args": list(args.dataset) if args.dataset else [],
            "data_format": args.data_format,
            "enable_thinking": args.enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n_requested": args.val_n,
            "gen_n": gen_n,
            "pass_at_k_list": pass_at_k_list,
            "pass_at_k": pass_at_k_summary,
            "avg1_pct": avg1_pct,
            "avg16_pct": avg16_pct,
            "metrics_by_dataset": metrics_by_dataset,
            "metrics_by_category": metrics_by_category,
            "num_problems": processed,
            "num_problems_total": n_prompts,
            "total_solutions": total_solutions,
            "average_correct_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            "majority_vote_pct": 100.0 * majority_correct / processed if processed else 0.0,
            "format_rate_pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
            "math_verify": _HAS_MATH_VERIFY,
            "generate_batch_size": gbs if args.generate_batch_size > 0 else n_prompts,
            "generate_batch_size_requested": args.generate_batch_size,
            "streaming_write": True,
            "results": results,
        }
        # Avoid rewriting the full generations blob every batch (O(n²) I/O, huge last write).
        # Checkpoint only scalars + metrics; full `results` is written once after the loop.
        disk_summary = {k: v for k, v in summary.items() if k != "results"}
        disk_summary["partial_only"] = True
        disk_summary["results_count"] = len(results)
        out_path.write_text(json.dumps(disk_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] wrote partial metrics json: {processed}/{n_prompts}", flush=True)

    n = len(results)
    pass_at_k_summary = summary["pass_at_k"]
    metrics_by_dataset = summary["metrics_by_dataset"]
    metrics_by_category = summary["metrics_by_category"]

    print(
        "[eval] writing final JSON with all generations (can take minutes on large n × problems) ...",
        flush=True,
    )
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[eval] wrote final json -> {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("[ALL] combined", flush=True)
    for k in pass_at_k_list:
        s = pass_at_k_summary[str(k)]
        print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{n})", flush=True)
    print(f"  Avg1(one-shot hit rate): {summary['avg1_pct']:.2f}%", flush=True)
    print(f"  Avg16(overall correctness): {summary['avg16_pct']:.2f}%", flush=True)
    for tag, m in metrics_by_dataset.items():
        print(f"[{tag}] n={m['num_problems']}", flush=True)
        for k in pass_at_k_list:
            s = m["pass_at_k"][str(k)]
            print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{m['num_problems']})", flush=True)
        print(f"  Avg1(one-shot hit rate): {m['avg1_pct']:.2f}%", flush=True)
        print(f"  Avg16(overall correctness): {m['avg16_pct']:.2f}%", flush=True)
    print("[BY_CATEGORY]", flush=True)
    for cat, m in metrics_by_category.items():
        print(f"[{cat}] n={m['num_problems']}", flush=True)
        for k in pass_at_k_list:
            s = m["pass_at_k"][str(k)]
            print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{m['num_problems']})", flush=True)
        print(f"  Avg1(one-shot hit rate): {m['avg1_pct']:.2f}%", flush=True)
        print(f"  Avg16(overall correctness): {m['avg16_pct']:.2f}%", flush=True)
    print(f"Avg correct / sample: {summary['average_correct_pct']:.2f}%", flush=True)
    print(f"Majority vote: {summary['majority_vote_pct']:.2f}%", flush=True)
    print(f"Boxed format rate: {summary['format_rate_pct']:.2f}%", flush=True)
    print(f"Wrote {out_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
