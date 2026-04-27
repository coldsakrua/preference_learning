#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False


@dataclass
class MathSample:
    sample_id: str
    problem: str
    solution: str
    ground_truth: str
    level: str
    problem_type: str
    subject: str
    dataset_name: str


@dataclass
class GenerationResult:
    text: str
    token_ids: Optional[List[int]]


def batched_list(items: List[str], batch_size: int) -> Iterable[List[str]]:
    size = max(int(batch_size), 1)
    for i in range(0, len(items), size):
        yield items[i : i + size]


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*(?:$|\n)", re.IGNORECASE)


def extract_boxed_answer_last(text: str) -> str:
    matches = list(_BOXED_RE.finditer(text or ""))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def extract_answer_candidate(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    boxed = extract_boxed_answer_last(s)
    if boxed:
        return boxed
    matches = list(_ANSWER_LINE_RE.finditer(s))
    if matches:
        return matches[-1].group(1).strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1].strip().strip("$")


def normalize_answer(text: str) -> str:
    return str(text or "").replace("$", "").replace(" ", "").strip().lower()


def is_correct_answer(predicted: str, ground_truth: str) -> bool:
    p = str(predicted or "").strip()
    g = str(ground_truth or "").strip()
    if not p or not g:
        return False
    if _HAS_MATH_VERIFY:
        try:
            p_in = p if "$" in p else f"${p}$"
            g_in = g if "$" in g else f"${g}$"
            p_parsed = parse(p_in, fallback_mode="no_fallback")
            g_parsed = parse(g_in, fallback_mode="no_fallback")
            return bool(verify(g_parsed, p_parsed, timeout_seconds=5))
        except Exception:
            pass
    return normalize_answer(p) == normalize_answer(g)


class RunningDiagStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.numel() == 0:
            return
        x = x.float().cpu()
        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.mean is not None and self.m2 is not None
        delta = batch_mean - self.mean
        total_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / total_n)
        self.m2 = self.m2 + batch_m2 + delta.pow(2) * (self.n * batch_n / total_n)
        self.n = total_n

    def finalize(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.n == 0 or self.mean is None or self.m2 is None:
            return None, None
        var = self.m2 / max(self.n, 1)
        return self.mean, var


def load_math_hf_samples(
    parquet_path: Path,
    max_samples: Optional[int],
    scan_batch_size: int,
    dataset_name: str,
) -> List[MathSample]:
    parquet_file = pq.ParquetFile(parquet_path)
    required = {"problem", "solution"}
    names = set(parquet_file.schema_arrow.names)
    if not required.issubset(names):
        raise ValueError(
            f"{parquet_path} is missing required fields {sorted(required)}; "
            f"available fields: {sorted(names)}"
        )

    cols = ["problem", "solution", "level", "type", "subject"]
    use_cols = [c for c in cols if c in names]
    samples: List[MathSample] = []
    for batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=use_cols):
        problems = batch.column("problem").to_pylist()
        solutions = batch.column("solution").to_pylist()
        levels = batch.column("level").to_pylist() if "level" in use_cols else ["" for _ in problems]
        types = batch.column("type").to_pylist() if "type" in use_cols else ["" for _ in problems]
        subjects = batch.column("subject").to_pylist() if "subject" in use_cols else ["" for _ in problems]
        for p, s, lv, tp, sub in zip(problems, solutions, levels, types, subjects):
            problem = str(p or "").strip()
            solution = str(s or "").strip()
            if not problem or not solution:
                continue
            ground_truth = extract_answer_candidate(solution)
            if not ground_truth:
                continue
            sample = MathSample(
                sample_id=f"math-{len(samples)}",
                problem=problem,
                solution=solution,
                ground_truth=ground_truth,
                level=str(lv or "").strip(),
                problem_type=str(tp or "").strip(),
                subject=str(sub or "").strip(),
                dataset_name=dataset_name,
            )
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def _extract_opsd_prompt(record: Dict[str, object]) -> str:
    problem = str(record.get("problem", "") or "").strip()
    if problem:
        return problem
    question = str(record.get("Question", "") or "").strip()
    if question:
        return question
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            if str(item.get("from", "")).strip().lower() == "user":
                text = str(item.get("value", "")).strip()
                if text:
                    return text
    return ""


def _extract_opsd_answer(record: Dict[str, object]) -> str:
    answer = str(record.get("Answer", "") or "").strip()
    if answer:
        return extract_answer_candidate(answer)
    solution = str(record.get("solution", "") or "").strip()
    return extract_answer_candidate(solution)


def _extract_opsd_solution(record: Dict[str, object]) -> str:
    solution = str(record.get("solution", "") or "").strip()
    if solution:
        return solution
    thought = str(record.get("COT_Reason", "") or "").strip()
    answer = str(record.get("Answer", "") or "").strip()
    if thought and answer:
        return f"<think>\n{thought}\n</think>\n\nAnswer: {answer}".strip()
    if answer:
        return f"Answer: {answer}".strip()
    return thought


def _iter_records_from_jsonlike(path: Path) -> Iterator[Dict[str, object]]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        return
    loaded = json.loads(text)
    if isinstance(loaded, list):
        for item in loaded:
            if isinstance(item, dict):
                yield item
    elif isinstance(loaded, dict):
        candidates = loaded.get("data")
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict):
                    yield item


def load_opsd_samples_from_records(
    records: Iterator[Dict[str, object]],
    max_samples: Optional[int],
    dataset_name: str,
) -> List[MathSample]:
    samples: List[MathSample] = []
    for record in records:
        problem = _extract_opsd_prompt(record)
        solution = _extract_opsd_solution(record)
        gt = _extract_opsd_answer(record)
        if not problem or not gt:
            continue
        if not solution:
            solution = f"Answer: {gt}"
        sample = MathSample(
            sample_id=f"{dataset_name}-{len(samples)}",
            problem=problem,
            solution=solution,
            ground_truth=gt,
            level="",
            problem_type="opsd",
            subject="",
            dataset_name=dataset_name,
        )
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _load_samples_for_single_file(
    dataset_file: Path,
    max_samples: Optional[int],
    scan_batch_size: int,
    dataset_name: str,
) -> List[MathSample]:
    suffix = dataset_file.suffix.lower()
    if suffix == ".parquet":
        parquet_file = pq.ParquetFile(dataset_file)
        names = set(parquet_file.schema_arrow.names)
        if {"problem", "solution"}.issubset(names):
            return load_math_hf_samples(
                parquet_path=dataset_file,
                max_samples=max_samples,
                scan_batch_size=scan_batch_size,
                dataset_name=dataset_name,
            )
        rows: List[Dict[str, object]] = []
        use_cols = [c for c in ["problem", "Question", "Answer", "solution", "COT_Reason", "conversations"] if c in names]
        for batch in parquet_file.iter_batches(batch_size=scan_batch_size, columns=use_cols):
            for rec in batch.to_pylist():
                if isinstance(rec, dict):
                    rows.append(rec)
                if max_samples is not None and len(rows) >= max_samples:
                    break
            if max_samples is not None and len(rows) >= max_samples:
                break
        return load_opsd_samples_from_records(iter(rows), max_samples=max_samples, dataset_name=dataset_name)
    if suffix in {".jsonl", ".json"}:
        return load_opsd_samples_from_records(
            _iter_records_from_jsonlike(dataset_file),
            max_samples=max_samples,
            dataset_name=dataset_name,
        )
    return []


def load_samples_for_dataset_path(
    dataset_path: Path,
    max_samples: Optional[int],
    scan_batch_size: int,
) -> List[MathSample]:
    if dataset_path.is_dir():
        dataset_name = dataset_path.name
        files = sorted(
            [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".jsonl", ".json"}]
        )
        merged: List[MathSample] = []
        for fp in files:
            part = _load_samples_for_single_file(
                fp,
                max_samples=None,
                scan_batch_size=scan_batch_size,
                dataset_name=dataset_name,
            )
            if part:
                merged.extend(part)
            if max_samples is not None and len(merged) >= max_samples:
                return merged[:max_samples]
        return merged
    dataset_name = dataset_path.stem
    return _load_samples_for_single_file(
        dataset_path,
        max_samples=max_samples,
        scan_batch_size=scan_batch_size,
        dataset_name=dataset_name,
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_prompt_text(tokenizer, problem: str, system_prompt: str) -> str:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": problem})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    if system_prompt.strip():
        return f"System: {system_prompt.strip()}\nUser: {problem}\nAssistant:"
    return f"User: {problem}\nAssistant:"


def encode_text(tokenizer, text: str, max_length: int) -> torch.Tensor:
    out = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    return out.input_ids


def sample_rows(x: torch.Tensor, max_rows: int, generator: torch.Generator) -> torch.Tensor:
    if x.shape[0] <= max_rows:
        return x
    idx = torch.randperm(x.shape[0], generator=generator)[:max_rows]
    return x[idx]


def get_last_hidden_suffix(
    model: Any,
    input_ids: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden = outputs.hidden_states[-1][0]
    if prefix_len >= hidden.shape[0]:
        return hidden.new_zeros((0, hidden.shape[1]))
    return hidden[prefix_len:].detach().float().cpu()


def trajectory_embedding_from_ids(
    model: Any,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    if suffix_ids.numel() == 0:
        return None
    full = torch.cat([prompt_ids[0], suffix_ids], dim=0).unsqueeze(0)
    hidden = get_last_hidden_suffix(model, full, prefix_len=int(prompt_ids.shape[1]))
    if hidden.shape[0] == 0:
        return None
    mu = hidden.mean(dim=0)
    end = hidden[-1]
    mx = hidden.max(dim=0).values
    return torch.cat([mu, end, mx], dim=0)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    return float(1.0 - sim)


def mean_pairwise_distance(vectors: List[torch.Tensor]) -> Tuple[float, int]:
    if len(vectors) < 2:
        return float("nan"), 0
    vals: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vals.append(cosine_distance(vectors[i], vectors[j]))
    if not vals:
        return float("nan"), 0
    return float(sum(vals) / len(vals)), len(vals)


def mean_cross_distance(a_vectors: List[torch.Tensor], b_vectors: List[torch.Tensor]) -> Tuple[float, int]:
    if not a_vectors or not b_vectors:
        return float("nan"), 0
    vals: List[float] = []
    for a in a_vectors:
        for b in b_vectors:
            vals.append(cosine_distance(a, b))
    if not vals:
        return float("nan"), 0
    return float(sum(vals) / len(vals)), len(vals)


def mean_to_standard_distance(vectors: List[torch.Tensor], standard: Optional[torch.Tensor]) -> Tuple[float, int]:
    if standard is None or not vectors:
        return float("nan"), 0
    vals = [cosine_distance(v, standard) for v in vectors]
    return float(sum(vals) / len(vals)), len(vals)


def load_hf_model_for_hidden(
    model_path: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> Any:
    from transformers import AutoModelForCausalLM

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    return model


def generate_reasoning_with_hf(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: List[str],
    rollout_n: int,
    gen_batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[List[GenerationResult]]:
    model_device = next(model.parameters()).device
    all_results: List[List[GenerationResult]] = []
    for prompt_batch in tqdm(
        list(batched_list(prompt_texts, gen_batch_size)),
        desc="Generate(HF)",
        ncols=100,
    ):
        batch_results: List[List[GenerationResult]] = [[] for _ in range(len(prompt_batch))]
        for _ in range(max(int(rollout_n), 1)):
            encoded = tokenizer(
                prompt_batch,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            )
            input_ids = encoded["input_ids"].to(model_device)
            attention_mask = encoded["attention_mask"].to(model_device)
            if input_ids.shape[1] == 0:
                for i in range(len(prompt_batch)):
                    batch_results[i].append(GenerationResult(text="", token_ids=[]))
                continue
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            prompt_lens = attention_mask.sum(dim=1).tolist()
            for row_i, prompt_len in enumerate(prompt_lens):
                gen_ids = generated[row_i][int(prompt_len) :]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids.numel() > 0 else ""
                batch_results[row_i].append(
                    GenerationResult(
                        text=text,
                        token_ids=[int(x) for x in gen_ids.tolist()],
                    )
                )
        all_results.extend(batch_results)
    return all_results


def generate_reasoning_with_vllm(
    *,
    model_path: str,
    prompt_texts: List[str],
    rollout_n: int,
    gen_batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    enforce_eager: bool,
) -> List[List[GenerationResult]]:
    from vllm import LLM, SamplingParams

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    n = max(int(rollout_n), 1)
    # vLLM forbids greedy decoding with n > 1. Auto-enable sampling in that case.
    effective_do_sample = bool(do_sample or n > 1)
    if n > 1 and not do_sample:
        print(
            f"[warn] rollout_n={n} with do_sample=False is incompatible with greedy decoding in vLLM; "
            "auto-switching to sampling mode."
        )
    effective_temperature = float(temperature) if effective_do_sample else 0.0
    if effective_do_sample and effective_temperature <= 0.0:
        effective_temperature = 1.0
    sp_kwargs = dict(
        n=n,
        temperature=effective_temperature,
        top_p=top_p if effective_do_sample else 1.0,
        max_tokens=max_new_tokens,
    )

    all_results: List[List[GenerationResult]] = []
    prompt_batches = list(batched_list(prompt_texts, gen_batch_size))
    for batch_idx, prompt_batch in enumerate(tqdm(prompt_batches, desc="Generate(vLLM)", ncols=100)):
        # Keep seeds stable across batches.
        sampling_params = SamplingParams(seed=seed + batch_idx, **sp_kwargs)
        outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
        batch_results: List[List[GenerationResult]] = []
        for out in outputs:
            one_prompt_results: List[GenerationResult] = []
            if not out.outputs:
                for _ in range(max(int(rollout_n), 1)):
                    one_prompt_results.append(GenerationResult(text="", token_ids=[]))
                batch_results.append(one_prompt_results)
                continue
            for cand in out.outputs:
                token_ids_obj = getattr(cand, "token_ids", None)
                token_ids = [int(x) for x in token_ids_obj] if token_ids_obj is not None else None
                one_prompt_results.append(
                    GenerationResult(
                        text=str(getattr(cand, "text", "")).strip(),
                        token_ids=token_ids,
                    )
                )
            while len(one_prompt_results) < max(int(rollout_n), 1):
                one_prompt_results.append(GenerationResult(text="", token_ids=[]))
            batch_results.append(one_prompt_results)
        all_results.extend(batch_results)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_results


def diag_gaussian_stats(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False) + eps
    return mu, var


def symmetric_kl_diag_gaussian(x: torch.Tensor, y: torch.Tensor) -> float:
    mu_x, var_x = diag_gaussian_stats(x)
    mu_y, var_y = diag_gaussian_stats(y)
    kl_xy = 0.5 * torch.sum(torch.log(var_y / var_x) + (var_x + (mu_x - mu_y) ** 2) / var_y - 1.0)
    kl_yx = 0.5 * torch.sum(torch.log(var_x / var_y) + (var_y + (mu_y - mu_x) ** 2) / var_x - 1.0)
    dim = float(mu_x.numel())
    return float((0.5 * (kl_xy + kl_yx) / max(dim, 1.0)).item())


def cosine_of_means(x: torch.Tensor, y: torch.Tensor) -> float:
    mu_x = x.mean(dim=0, keepdim=True)
    mu_y = y.mean(dim=0, keepdim=True)
    cos = F.cosine_similarity(mu_x, mu_y, dim=1).item()
    return float(cos)


def l2_of_means(x: torch.Tensor, y: torch.Tensor) -> float:
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    return float(torch.norm(mu_x - mu_y, p=2).item() / math.sqrt(max(mu_x.numel(), 1)))


def _rbf_kernel_matrix(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True)
    sqdist = x_norm + y_norm.t() - 2.0 * (x @ y.t())
    sqdist = torch.clamp(sqdist, min=0.0)
    return torch.exp(-gamma * sqdist)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, generator: torch.Generator, max_tokens: int = 256) -> float:
    x_s = sample_rows(x, max_tokens, generator)
    y_s = sample_rows(y, max_tokens, generator)
    if x_s.shape[0] == 0 or y_s.shape[0] == 0:
        return float("nan")

    z = torch.cat([x_s, y_s], dim=0)
    if z.shape[0] <= 1:
        return 0.0
    with torch.no_grad():
        sq = torch.cdist(z, z, p=2).pow(2)
        vals = sq[sq > 0]
        sigma2 = torch.median(vals).item() if vals.numel() > 0 else 1.0
        sigma2 = max(float(sigma2), 1e-6)
        gamma = 1.0 / (2.0 * sigma2)
        k_xx = _rbf_kernel_matrix(x_s, x_s, gamma).mean()
        k_yy = _rbf_kernel_matrix(y_s, y_s, gamma).mean()
        k_xy = _rbf_kernel_matrix(x_s, y_s, gamma).mean()
        mmd2 = (k_xx + k_yy - 2.0 * k_xy).item()
    return float(max(mmd2, 0.0))


def linear_cka(x: torch.Tensor, y: torch.Tensor, generator: torch.Generator, max_tokens: int = 256) -> float:
    x_s = sample_rows(x, max_tokens, generator)
    y_s = sample_rows(y, max_tokens, generator)
    n = min(x_s.shape[0], y_s.shape[0])
    if n < 2:
        return float("nan")
    x_s = x_s[:n] - x_s[:n].mean(dim=0, keepdim=True)
    y_s = y_s[:n] - y_s[:n].mean(dim=0, keepdim=True)
    xty = x_s.t() @ y_s
    hsic = torch.norm(xty, p="fro").pow(2)
    xx = x_s.t() @ x_s
    yy = y_s.t() @ y_s
    denom = torch.norm(xx, p="fro") * torch.norm(yy, p="fro")
    if denom.item() <= 0:
        return float("nan")
    return float((hsic / denom).item())


def mean_without_nan(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def std_without_nan(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return float(math.sqrt(max(var, 0.0)))


def ratio_or_nan(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def bootstrap_split_distance(
    bank: torch.Tensor,
    generator: torch.Generator,
    distance_fn: Any,
    repeats: int = 100,
) -> float:
    if bank.ndim != 2 or bank.shape[0] < 4:
        return float("nan")
    n = int(bank.shape[0])
    vals: List[float] = []
    for _ in range(max(int(repeats), 1)):
        perm = torch.randperm(n, generator=generator)
        half = n // 2
        if half <= 0 or (n - half) <= 0:
            continue
        a = bank[perm[:half]]
        b = bank[perm[half:]]
        vals.append(float(distance_fn(a, b)))
    return mean_without_nan(vals)


def weighted_mean_from_pairs(rows: List[Dict[str, object]], mean_key: str, count_key: str) -> float:
    num = 0.0
    den = 0
    for r in rows:
        c = int(r.get(count_key, 0) or 0)
        m = float(r.get(mean_key, float("nan")))
        if c <= 0 or math.isnan(m):
            continue
        num += m * c
        den += c
    if den == 0:
        return float("nan")
    return float(num / den)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def maybe_make_plot(per_problem_rows: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    x = list(range(len(per_problem_rows)))
    cc = [float(r["cos_dist_correct_correct_mean"]) for r in per_problem_rows]
    cw = [float(r["cos_dist_correct_wrong_mean"]) for r in per_problem_rows]
    cs = [float(r["cos_dist_correct_standard_mean"]) for r in per_problem_rows]
    ws = [float(r["cos_dist_wrong_standard_mean"]) for r in per_problem_rows]
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.plot(x, cc, label="Correct-Correct")
    ax.plot(x, cw, label="Correct-Wrong")
    ax.plot(x, cs, label="Correct-Standard")
    ax.plot(x, ws, label="Wrong-Standard")
    ax.set_ylabel("Cosine Distance (1 - cos)")
    ax.set_xlabel("Problem Index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "cosine_distance_by_problem.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hidden-state distributions between model-generated reasoning "
            "and reference reasoning (solution), then report divergence metrics."
        )
    )
    parser.add_argument("--dataset_path", type=str, default="logs/train.parquet")
    parser.add_argument(
        "--dataset_paths",
        type=str,
        default="",
        help="Optional comma-separated dataset paths for joint multi-dataset analysis.",
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/hidden_state_distribution")
    parser.add_argument("--inference_backend", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no extra cap; use rounds*problems_per_batch.")
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--problems_per_batch", type=int, default=128)
    parser.add_argument("--rollout_rounds", type=int, default=3)
    parser.add_argument("--scan_batch_size", type=int, default=256)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_reference_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--system_prompt", type=str, default="")
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--max_global_tokens", type=int, default=4096)
    parser.add_argument("--max_tokens_per_sample_for_global", type=int, default=128)
    parser.add_argument("--bootstrap_repeats", type=int, default=100)
    parser.add_argument("--bootstrap_max_tokens", type=int, default=1024)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument(
        "--vllm_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_paths_raw = [x.strip() for x in str(args.dataset_paths).split(",") if x.strip()]
    if dataset_paths_raw:
        dataset_paths = [Path(p).expanduser().resolve() for p in dataset_paths_raw]
    else:
        dataset_paths = [Path(args.dataset_path).expanduser().resolve()]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for dp in dataset_paths:
        if not dp.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dp}")

    per_dataset_loaded: Dict[str, int] = {}
    per_dataset_selected: Dict[str, int] = {}
    samples: List[MathSample] = []
    samples_all_count = 0
    requested_total = max(int(args.problems_per_batch), 1) * max(int(args.rollout_rounds), 1)
    if args.max_samples > 0:
        requested_total = min(requested_total, int(args.max_samples))
    per_dataset_cap = max(1, int(math.ceil(requested_total / max(len(dataset_paths), 1))))
    for dataset_path in dataset_paths:
        loaded = load_samples_for_dataset_path(
            dataset_path=dataset_path,
            max_samples=None,
            scan_batch_size=args.scan_batch_size,
        )
        if not loaded:
            continue
        per_dataset_loaded[dataset_path.stem] = len(loaded)
        selected = loaded[: min(len(loaded), per_dataset_cap)]
        per_dataset_selected[dataset_path.stem] = len(selected)
        samples.extend(selected)
        samples_all_count += len(loaded)
    if not samples:
        raise RuntimeError("No samples were selected after parsing all dataset paths.")
    if len(samples) > requested_total:
        samples = samples[:requested_total]
    effective_rounds = int(math.ceil(len(samples) / max(int(args.problems_per_batch), 1))) if samples else 0

    preview = {
        "dataset_paths": [str(p) for p in dataset_paths],
        "num_loaded_samples_total": samples_all_count,
        "num_loaded_samples_by_dataset": per_dataset_loaded,
        "num_selected_samples_by_dataset": per_dataset_selected,
        "num_selected_samples": len(samples),
        "requested_total_samples": requested_total,
        "problems_per_batch": int(args.problems_per_batch),
        "rollout_rounds": int(args.rollout_rounds),
        "effective_rounds": effective_rounds,
        "rollout_n": int(args.rollout_n),
        "first_sample": {
            "sample_id": samples[0].sample_id,
            "dataset_name": samples[0].dataset_name,
            "problem_preview": samples[0].problem[:240],
            "solution_preview": samples[0].solution[:240],
            "ground_truth": samples[0].ground_truth,
            "level": samples[0].level,
            "type": samples[0].problem_type,
            "subject": samples[0].subject,
        },
    }
    (output_dir / "dataset_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.inspect_only:
        print("[inspect_only] Dataset parsing finished. Model loading was skipped.")
        return

    if not args.model_path.strip():
        raise ValueError("Please provide --model_path (local path or HF model id/path).")

    from transformers import AutoTokenizer

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_texts = [build_prompt_text(tokenizer, s.problem, args.system_prompt) for s in samples]

    model: Any
    generation_results: List[List[GenerationResult]]
    if args.inference_backend == "vllm":
        generation_results = []
        for round_idx in range(effective_rounds):
            st = round_idx * int(args.problems_per_batch)
            ed = min((round_idx + 1) * int(args.problems_per_batch), len(prompt_texts))
            if st >= ed:
                break
            round_prompts = prompt_texts[st:ed]
            round_results = generate_reasoning_with_vllm(
                model_path=args.model_path,
                prompt_texts=round_prompts,
                rollout_n=args.rollout_n,
                gen_batch_size=args.gen_batch_size,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + round_idx * 10007,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                dtype=args.vllm_dtype,
                enforce_eager=args.vllm_enforce_eager,
            )
            generation_results.extend(round_results)
        model = load_hf_model_for_hidden(
            args.model_path,
            device=device,
            dtype=dtype,
        )
    else:
        model = load_hf_model_for_hidden(
            args.model_path,
            device=device,
            dtype=dtype,
        )
        generation_results = generate_reasoning_with_hf(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            rollout_n=args.rollout_n,
            gen_batch_size=args.gen_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    if len(generation_results) != len(samples):
        raise RuntimeError(
            f"Generation result size mismatch: got {len(generation_results)}, expected {len(samples)}."
        )

    model_device = next(model.parameters()).device

    teacher_bank: List[torch.Tensor] = []
    rollout_bank: List[torch.Tensor] = []
    correct_bank: List[torch.Tensor] = []
    wrong_bank: List[torch.Tensor] = []
    teacher_bank_by_dataset: Dict[str, List[torch.Tensor]] = {}
    rollout_bank_by_dataset: Dict[str, List[torch.Tensor]] = {}
    correct_counts: List[int] = []
    wrong_counts: List[int] = []
    for sample, prompt_text, one_problem_rollouts in tqdm(
        zip(samples, prompt_texts, generation_results),
        desc="Analyze",
        ncols=100,
        total=len(samples),
    ):
        prompt_ids = encode_text(tokenizer, prompt_text, max_length=args.max_prompt_tokens)
        if prompt_ids.shape[1] == 0:
            continue
        prompt_ids = prompt_ids.to(model_device)

        ref_ids = encode_text(tokenizer, sample.solution, max_length=args.max_reference_tokens)[0]
        if ref_ids.numel() == 0:
            continue
        standard_emb = trajectory_embedding_from_ids(model, prompt_ids, ref_ids.to(model_device))
        if standard_emb is not None:
            teacher_bank.append(standard_emb)
            teacher_bank_by_dataset.setdefault(sample.dataset_name, []).append(standard_emb)

        one_correct = 0
        one_wrong = 0
        for gen_res in one_problem_rollouts[: max(int(args.rollout_n), 1)]:
            gen_ids: torch.Tensor
            if gen_res.token_ids is not None:
                gen_ids = torch.tensor(gen_res.token_ids, device=model_device, dtype=torch.long)
            else:
                gen_ids = encode_text(
                    tokenizer,
                    gen_res.text,
                    max_length=args.max_new_tokens,
                )[0].to(model_device)
            if gen_ids.numel() == 0:
                continue
            if args.max_new_tokens > 0 and gen_ids.numel() > args.max_new_tokens:
                gen_ids = gen_ids[: args.max_new_tokens]
            emb = trajectory_embedding_from_ids(model, prompt_ids, gen_ids)
            if emb is None:
                continue
            rollout_bank.append(emb)
            rollout_bank_by_dataset.setdefault(sample.dataset_name, []).append(emb)
            pred_answer = extract_answer_candidate(gen_res.text)
            if is_correct_answer(pred_answer, sample.ground_truth):
                correct_bank.append(emb)
                one_correct += 1
            else:
                wrong_bank.append(emb)
                one_wrong += 1
        correct_counts.append(one_correct)
        wrong_counts.append(one_wrong)

    if not teacher_bank or not rollout_bank:
        raise RuntimeError("No usable embeddings were collected for teacher/rollout banks.")

    teacher_tensor = torch.stack(teacher_bank, dim=0).float()
    rollout_tensor = torch.stack(rollout_bank, dim=0).float()
    correct_tensor = torch.stack(correct_bank, dim=0).float() if correct_bank else None
    wrong_tensor = torch.stack(wrong_bank, dim=0).float() if wrong_bank else None

    metric_generator = torch.Generator().manual_seed(args.seed + 17)
    bootstrap_generator = torch.Generator().manual_seed(args.seed + 29)

    def mmd_with_limit(a: torch.Tensor, b: torch.Tensor) -> float:
        return mmd_rbf(
            a,
            b,
            generator=metric_generator,
            max_tokens=max(int(args.bootstrap_max_tokens), 32),
        )

    teacher_rollout_mmd = mmd_with_limit(teacher_tensor, rollout_tensor)
    rollout_internal_mmd = bootstrap_split_distance(
        rollout_tensor,
        generator=bootstrap_generator,
        distance_fn=mmd_with_limit,
        repeats=int(args.bootstrap_repeats),
    )
    teacher_rollout_ratio = ratio_or_nan(teacher_rollout_mmd, rollout_internal_mmd)

    teacher_rollout_cos_mean = cosine_of_means(teacher_tensor, rollout_tensor)
    teacher_rollout_l2_mean = l2_of_means(teacher_tensor, rollout_tensor)

    correct_wrong_mmd = float("nan")
    correct_wrong_ratio = float("nan")
    if correct_tensor is not None and wrong_tensor is not None:
        correct_wrong_mmd = mmd_with_limit(correct_tensor, wrong_tensor)
        correct_wrong_ratio = ratio_or_nan(correct_wrong_mmd, rollout_internal_mmd)

    per_dataset_margin: Dict[str, Dict[str, float]] = {}
    for ds_name, t_list in teacher_bank_by_dataset.items():
        r_list = rollout_bank_by_dataset.get(ds_name, [])
        if not t_list or not r_list:
            continue
        t_tensor = torch.stack(t_list, dim=0).float()
        r_tensor = torch.stack(r_list, dim=0).float()
        ds_tr = mmd_with_limit(t_tensor, r_tensor)
        ds_rr = bootstrap_split_distance(
            r_tensor,
            generator=bootstrap_generator,
            distance_fn=mmd_with_limit,
            repeats=int(args.bootstrap_repeats),
        )
        per_dataset_margin[ds_name] = {
            "teacher_rollout_mmd": ds_tr,
            "rollout_internal_mmd_bootstrap": ds_rr,
            "teacher_rollout_over_internal_ratio": ratio_or_nan(ds_tr, ds_rr),
            "teacher_count": float(t_tensor.shape[0]),
            "rollout_count": float(r_tensor.shape[0]),
        }

    dataset_pairwise_teacher_mmd: List[Dict[str, object]] = []
    ds_names = sorted([k for k, v in teacher_bank_by_dataset.items() if len(v) > 0])
    for i in range(len(ds_names)):
        for j in range(i + 1, len(ds_names)):
            a_name = ds_names[i]
            b_name = ds_names[j]
            a_tensor = torch.stack(teacher_bank_by_dataset[a_name], dim=0).float()
            b_tensor = torch.stack(teacher_bank_by_dataset[b_name], dim=0).float()
            dataset_pairwise_teacher_mmd.append(
                {
                    "dataset_a": a_name,
                    "dataset_b": b_name,
                    "teacher_teacher_mmd": mmd_with_limit(a_tensor, b_tensor),
                    "teacher_teacher_cosine_of_means": cosine_of_means(a_tensor, b_tensor),
                    "teacher_teacher_l2_of_means_normed": l2_of_means(a_tensor, b_tensor),
                }
            )

    summary = {
        "dataset_paths": [str(p) for p in dataset_paths],
        "model_path": args.model_path,
        "samples_total_available": samples_all_count,
        "samples_selected": len(samples),
        "samples_used": len(teacher_bank),
        "settings": {
            "inference_backend": args.inference_backend,
            "rollout_n": args.rollout_n,
            "problems_per_batch": args.problems_per_batch,
            "rollout_rounds": args.rollout_rounds,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_reference_tokens": args.max_reference_tokens,
            "max_new_tokens": args.max_new_tokens,
            "gen_batch_size": args.gen_batch_size,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "system_prompt": args.system_prompt,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_dtype": args.vllm_dtype,
            "vllm_enforce_eager": args.vllm_enforce_eager,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_max_tokens": args.bootstrap_max_tokens,
        },
        "bank_sizes": {
            "teacher_bank_size": int(teacher_tensor.shape[0]),
            "rollout_bank_size": int(rollout_tensor.shape[0]),
            "correct_bank_size": int(0 if correct_tensor is None else correct_tensor.shape[0]),
            "wrong_bank_size": int(0 if wrong_tensor is None else wrong_tensor.shape[0]),
            "embedding_dim": int(teacher_tensor.shape[1]),
        },
        "correctness_counts": {
            "mean_num_correct_per_problem": float(sum(correct_counts) / len(correct_counts)),
            "mean_num_wrong_per_problem": float(sum(wrong_counts) / len(wrong_counts)),
            "total_correct_trajectories": int(sum(correct_counts)),
            "total_wrong_trajectories": int(sum(wrong_counts)),
        },
        "margin_hypothesis_metrics": {
            "teacher_rollout_mmd": teacher_rollout_mmd,
            "rollout_internal_mmd_bootstrap": rollout_internal_mmd,
            "teacher_rollout_over_internal_ratio": teacher_rollout_ratio,
            "teacher_rollout_cosine_of_means": teacher_rollout_cos_mean,
            "teacher_rollout_l2_of_means_normed": teacher_rollout_l2_mean,
            "correct_wrong_mmd": correct_wrong_mmd,
            "correct_wrong_over_internal_ratio": correct_wrong_ratio,
        },
        "per_dataset_margin_metrics": per_dataset_margin,
        "dataset_pairwise_teacher_metrics": dataset_pairwise_teacher_mmd,
        "files": {
            "dataset_preview": str(output_dir / "dataset_preview.json"),
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
