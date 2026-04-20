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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
            )
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


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
    return hidden.mean(dim=0)


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
    sp_kwargs = dict(
        n=max(int(rollout_n), 1),
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
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

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_all = load_math_hf_samples(
        parquet_path=dataset_path,
        max_samples=None,
        scan_batch_size=args.scan_batch_size,
    )
    if not samples_all:
        raise RuntimeError(f"No valid samples were loaded from {dataset_path}.")

    requested_total = max(int(args.problems_per_batch), 1) * max(int(args.rollout_rounds), 1)
    if args.max_samples > 0:
        requested_total = min(requested_total, int(args.max_samples))
    requested_total = min(requested_total, len(samples_all))
    samples = samples_all[:requested_total]
    if not samples:
        raise RuntimeError("No samples were selected after applying round/batch/sample limits.")
    effective_rounds = int(math.ceil(len(samples) / max(int(args.problems_per_batch), 1))) if samples else 0

    preview = {
        "dataset_path": str(dataset_path),
        "num_loaded_samples": len(samples_all),
        "num_selected_samples": len(samples),
        "requested_total_samples": requested_total,
        "problems_per_batch": int(args.problems_per_batch),
        "rollout_rounds": int(args.rollout_rounds),
        "effective_rounds": effective_rounds,
        "rollout_n": int(args.rollout_n),
        "columns": list(pq.ParquetFile(dataset_path).schema_arrow.names),
        "first_sample": {
            "sample_id": samples[0].sample_id,
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

    per_problem_rows: List[Dict[str, object]] = []
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

        correct_vectors: List[torch.Tensor] = []
        wrong_vectors: List[torch.Tensor] = []
        first_correct_preview = ""
        first_wrong_preview = ""
        valid_rollouts = 0
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
            valid_rollouts += 1
            pred_answer = extract_answer_candidate(gen_res.text)
            if is_correct_answer(pred_answer, sample.ground_truth):
                correct_vectors.append(emb)
                if not first_correct_preview:
                    first_correct_preview = gen_res.text[:180].replace("\n", " ")
            else:
                wrong_vectors.append(emb)
                if not first_wrong_preview:
                    first_wrong_preview = gen_res.text[:180].replace("\n", " ")

        cc_mean, cc_count = mean_pairwise_distance(correct_vectors)
        cw_mean, cw_count = mean_cross_distance(correct_vectors, wrong_vectors)
        cs_mean, cs_count = mean_to_standard_distance(correct_vectors, standard_emb)
        ws_mean, ws_count = mean_to_standard_distance(wrong_vectors, standard_emb)

        row = {
            "sample_id": sample.sample_id,
            "level": sample.level,
            "type": sample.problem_type,
            "subject": sample.subject,
            "rollout_n": int(args.rollout_n),
            "valid_rollouts": int(valid_rollouts),
            "num_correct": len(correct_vectors),
            "num_wrong": len(wrong_vectors),
            "ground_truth": sample.ground_truth,
            "correct_correct_pairs": int(cc_count),
            "correct_wrong_pairs": int(cw_count),
            "correct_standard_pairs": int(cs_count),
            "wrong_standard_pairs": int(ws_count),
            "cos_dist_correct_correct_mean": cc_mean,
            "cos_dist_correct_wrong_mean": cw_mean,
            "cos_dist_correct_standard_mean": cs_mean,
            "cos_dist_wrong_standard_mean": ws_mean,
            "problem_preview": sample.problem[:120].replace("\n", " "),
            "first_correct_preview": first_correct_preview,
            "first_wrong_preview": first_wrong_preview,
            "solution_preview": sample.solution[:160].replace("\n", " "),
        }
        per_problem_rows.append(row)

    if not per_problem_rows:
        raise RuntimeError("No usable per-problem rows were produced. Check model outputs and token limits.")

    save_csv(output_dir / "per_problem_cosine_metrics.csv", per_problem_rows)

    metric_defs = [
        ("cos_dist_correct_correct_mean", "correct_correct_pairs"),
        ("cos_dist_correct_wrong_mean", "correct_wrong_pairs"),
        ("cos_dist_correct_standard_mean", "correct_standard_pairs"),
        ("cos_dist_wrong_standard_mean", "wrong_standard_pairs"),
    ]
    per_metric: Dict[str, Dict[str, float]] = {}
    for mean_key, count_key in metric_defs:
        vals = [float(r[mean_key]) for r in per_problem_rows]
        valid_vals = [v for v in vals if not math.isnan(v)]
        per_metric[mean_key] = {
            "mean_over_problems": mean_without_nan(vals),
            "std_over_problems": std_without_nan(vals),
            "min_over_problems": min(valid_vals) if valid_vals else float("nan"),
            "max_over_problems": max(valid_vals) if valid_vals else float("nan"),
            "weighted_mean_over_pairs": weighted_mean_from_pairs(per_problem_rows, mean_key, count_key),
            "total_pair_count": float(sum(int(r[count_key]) for r in per_problem_rows)),
            "valid_problem_count": float(sum(0 if math.isnan(float(r[mean_key])) else 1 for r in per_problem_rows)),
        }

    correct_counts = [int(r["num_correct"]) for r in per_problem_rows]
    wrong_counts = [int(r["num_wrong"]) for r in per_problem_rows]

    summary = {
        "dataset_path": str(dataset_path),
        "model_path": args.model_path,
        "samples_total_available": len(samples_all),
        "samples_selected": len(samples),
        "samples_used": len(per_problem_rows),
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
        },
        "correctness_counts": {
            "mean_num_correct_per_problem": float(sum(correct_counts) / len(correct_counts)),
            "mean_num_wrong_per_problem": float(sum(wrong_counts) / len(wrong_counts)),
            "total_correct_trajectories": int(sum(correct_counts)),
            "total_wrong_trajectories": int(sum(wrong_counts)),
        },
        "cosine_distance_stats": per_metric,
        "files": {
            "dataset_preview": str(output_dir / "dataset_preview.json"),
            "per_problem_cosine_metrics_csv": str(output_dir / "per_problem_cosine_metrics.csv"),
        },
    }

    plot_path = None
    if not args.skip_plot:
        plot_path = maybe_make_plot(per_problem_rows, output_dir)
        if plot_path is not None:
            summary["files"]["plot"] = str(plot_path)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
