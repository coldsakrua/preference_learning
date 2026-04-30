#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from utils import (
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    DapoSample,
    answer_text_matches,
    build_parser as build_cli_parser,
    compute_prompt_rarity_weight,
    compute_smoothed_correct_rate,
    detect_parquet_dataset_layout,
    extract_reference_answer_for_verifier,
    extract_rollout_scored_answer,
    iter_dapo_samples,
    iter_math_hf_samples,
    set_seed,
    strip_prompt_prefix_from_text,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise math reasoning assistant. "
    "Solve the problem step by step, then end with exactly one final line in the format: "
    "Answer: $<final_answer>."
)

DEFAULT_PROMPT_CANDIDATES = [
    "You are a careful math tutor. Show concise but correct reasoning and finish with: Answer: $<final_answer>.",
    "Solve the math problem with rigorous steps. Keep reasoning structured and end with: Answer: $<final_answer>.",
    "You are an expert competition-math assistant. Verify key steps and finish with: Answer: $<final_answer>.",
    "Reason clearly and avoid arithmetic mistakes. The last line must be: Answer: $<final_answer>.",
    "Produce a correct step-by-step solution, then output one final line: Answer: $<final_answer>.",
]


_EMPTY_LORA_HEALTH = {
    "lora_mean_abs": 0.0,
    "lora_max_abs": 0.0,
    "lora_nan_ratio": 0.0,
    "lora_inf_ratio": 0.0,
}

_WARNED_MISSING_CHAT_TEMPLATE = False


def _empty_lora_health() -> Dict[str, float]:
    return dict(_EMPTY_LORA_HEALTH)



@dataclass
class OnlinePendingObjective:
    """One prompt-level objective queued for immediate rollout-step update."""

    sample_id: str
    ground_truth: str
    train_prompt: str
    objective_type: str  # "mixed" | "all_correct" | "all_wrong" | "correct_only_mle" | "mixed_pref_only"
    rho_hat: float
    prompt_weight: float
    correct: List["RolloutTrajectory"]
    wrong: List["RolloutTrajectory"]
    correct_traj_weights: List[float]
    mixed_pref_pairs: List[Tuple[int, int]]  # (correct_idx, wrong_idx)
    gt_positive: Optional["RolloutTrajectory"]


@dataclass
class RolloutTrajectory:
    response_text: str
    token_ids: List[int]
    is_correct: bool
    fail_type: str
    has_final_answer_line: bool
    final_answer: str
    avg_logprob: float
    avg_nll: float
    avg_entropy: float
    hidden_vec: List[float]


@dataclass
class OnlineStepLossStats:
    total_loss: float
    mle_loss: float
    pref_loss: float
    gt_pref_loss: float
    mean_gap: float
    pref_pairs_used: int
    gt_pref_pairs_used: int
    mle_samples_used: int
    lora_mean_abs: float
    lora_max_abs: float
    lora_nan_ratio: float
    lora_inf_ratio: float
    grad_norm: float
    update_applied: bool
    skip_reason: str


def _compute_lora_param_health(model: object) -> Dict[str, float]:
    """Compute lightweight LoRA parameter health metrics after each update."""
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return _empty_lora_health()

    total_numel = 0
    nan_numel = 0
    inf_numel = 0
    abs_sum = 0.0
    abs_max = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" not in name:
                continue
            if param is None:
                continue
            tensor = param.detach()
            if tensor.numel() == 0:
                continue
            total_numel += tensor.numel()
            nan_numel += int(torch.isnan(tensor).sum().item())
            inf_numel += int(torch.isinf(tensor).sum().item())
            abs_tensor = torch.abs(tensor)
            abs_sum += float(abs_tensor.sum().item())
            abs_max = max(abs_max, float(abs_tensor.max().item()))

    if total_numel == 0:
        return _empty_lora_health()
    return {
        "lora_mean_abs": abs_sum / total_numel,
        "lora_max_abs": abs_max,
        "lora_nan_ratio": nan_numel / total_numel,
        "lora_inf_ratio": inf_numel / total_numel,
    }


@dataclass
class RolloutCandidateSplit:
    responses_has_final_answer_line: List[bool]
    responses_final_answers: List[str]
    responses_correct: List[bool]
    responses_fail_type: List[str]
    correct_kept_indices: List[int]
    wrong_kept_indices: List[int]
    correct_kept: List[str]
    wrong_kept: List[str]


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
    global _WARNED_MISSING_CHAT_TEMPLATE
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
    except ValueError as e:
        if "tokenizer.chat_template is not set" not in str(e):
            raise
        if not _WARNED_MISSING_CHAT_TEMPLATE:
            print(
                "[warn] tokenizer.chat_template is missing; fallback to plain text prompts "
                "for online rollout/training."
            )
            _WARNED_MISSING_CHAT_TEMPLATE = True
        return "\n\n".join(m.get("content", "") for m in messages if m.get("content"))


def split_rollout_candidates_for_training(
    candidates: Sequence[str],
    ground_truth: str,
) -> RolloutCandidateSplit:
    responses_has_final_answer_line: List[bool] = []
    responses_final_answers: List[str] = []
    responses_correct: List[bool] = []
    responses_fail_type: List[str] = []
    correct_kept_indices: List[int] = []
    wrong_kept_indices: List[int] = []
    correct_kept: List[str] = []
    wrong_kept: List[str] = []
    for idx, candidate in enumerate(candidates):
        has_final_answer_line, parsed_last_answer = extract_rollout_scored_answer(candidate)
        parsed_answer = parsed_last_answer if has_final_answer_line else ""
        is_correct = answer_text_matches(parsed_answer, ground_truth)
        responses_has_final_answer_line.append(has_final_answer_line)
        responses_final_answers.append(parsed_answer)
        responses_correct.append(is_correct)
        if is_correct:
            responses_fail_type.append("correct")
        elif not has_final_answer_line:
            responses_fail_type.append("no_final_answer")
        elif not parsed_answer:
            responses_fail_type.append("empty_final_answer")
        else:
            responses_fail_type.append("wrong_answer")

        if is_correct:
            correct_kept_indices.append(idx)
            correct_kept.append(str(candidate))
        else:
            wrong_kept_indices.append(idx)
            wrong_kept.append(str(candidate))
    return RolloutCandidateSplit(
        responses_has_final_answer_line=responses_has_final_answer_line,
        responses_final_answers=responses_final_answers,
        responses_correct=responses_correct,
        responses_fail_type=responses_fail_type,
        correct_kept_indices=correct_kept_indices,
        wrong_kept_indices=wrong_kept_indices,
        correct_kept=correct_kept,
        wrong_kept=wrong_kept,
    )


def compute_correct_trajectory_weights(
    correct_trajs: Sequence[RolloutTrajectory],
    mode: str,
    tau: float,
) -> List[float]:
    count = len(correct_trajs)
    if count == 0:
        return []
    if mode == "uniform":
        return [1.0 / count for _ in range(count)]
    if mode != "nll_softmax":
        raise ValueError(f"Unsupported positive weight mode: {mode}")
    nll = torch.tensor([float(t.avg_nll) for t in correct_trajs], dtype=torch.float32)
    logits = float(tau) * nll
    weights = torch.softmax(logits, dim=0)
    return [float(x) for x in weights.tolist()]


def build_hidden_nn_pairs(
    correct_trajs: Sequence[RolloutTrajectory],
    wrong_trajs: Sequence[RolloutTrajectory],
) -> List[Tuple[int, int]]:
    if not correct_trajs or not wrong_trajs:
        return []
    correct_h = torch.tensor([t.hidden_vec for t in correct_trajs], dtype=torch.float32)
    wrong_h = torch.tensor([t.hidden_vec for t in wrong_trajs], dtype=torch.float32)
    # hidden_vec has already been L2-normalized; dot product equals cosine.
    sim = wrong_h @ correct_h.transpose(0, 1)
    nn_correct_idx = sim.argmax(dim=1).tolist()
    return [(int(c_idx), int(w_idx)) for w_idx, c_idx in enumerate(nn_correct_idx)]


def _pref_pair_passes_avg_logprob_floor(
    chosen_avg_logprob: float,
    rejected_avg_logprob: float,
    min_chosen: Optional[float],
    min_rejected: Optional[float],
) -> bool:
    """Drop preference pairs whose rollout-time avg sequence logprob is too negative."""
    if min_chosen is not None and chosen_avg_logprob < min_chosen:
        return False
    if min_rejected is not None and rejected_avg_logprob < min_rejected:
        return False
    return True


def filter_mixed_pref_pairs_by_avg_logprob(
    mixed_pairs: Sequence[Tuple[int, int]],
    correct_trajs: Sequence[RolloutTrajectory],
    wrong_trajs: Sequence[RolloutTrajectory],
    min_chosen: Optional[float],
    min_rejected: Optional[float],
) -> List[Tuple[int, int]]:
    if min_chosen is None and min_rejected is None:
        return [(int(a), int(b)) for a, b in mixed_pairs]
    out: List[Tuple[int, int]] = []
    for c_idx, w_idx in mixed_pairs:
        c_lp = correct_trajs[c_idx].avg_logprob
        w_lp = wrong_trajs[w_idx].avg_logprob
        if _pref_pair_passes_avg_logprob_floor(c_lp, w_lp, min_chosen, min_rejected):
            out.append((int(c_idx), int(w_idx)))
    return out


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(float(v) for v in values) / len(values))


def rollout_trajectory_to_json(traj: RolloutTrajectory, *, include_dense: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "response_text": traj.response_text,
        "is_correct": bool(traj.is_correct),
        "fail_type": traj.fail_type,
        "has_final_answer_line": bool(traj.has_final_answer_line),
        "final_answer": traj.final_answer,
        "avg_logprob": float(traj.avg_logprob),
        "avg_nll": float(traj.avg_nll),
        "avg_entropy": float(traj.avg_entropy),
    }
    if include_dense:
        out["token_ids"] = [int(t) for t in traj.token_ids]
        out["hidden_vec"] = [float(v) for v in traj.hidden_vec]
    return out


def build_online_bootstrap_jsonl_record(
    *,
    sample_id: str,
    prompt: str,
    prompt_user_effective: str,
    system_prompt: str,
    ground_truth: str,
    candidates: Sequence[str],
    split: RolloutCandidateSplit,
    objective: Optional[OnlinePendingObjective],
    prompt_weight: float,
    rho_hat: float,
    all_trajectories: Sequence[RolloutTrajectory],
    include_dense_rollouts: bool,
) -> Dict[str, Any]:
    n_correct_total = int(sum(1 for x in split.responses_correct if x))
    n_total = len(candidates)
    n_wrong_total = n_total - n_correct_total
    objective_type = "skip" if objective is None else objective.objective_type
    record: Dict[str, Any] = {
        "sample_id": sample_id,
        "prompt": prompt,
        "prompt_user_effective": prompt_user_effective,
        "system_prompt": system_prompt,
        "ground_truth": ground_truth,
        "responses": [str(c) for c in candidates],
        "responses_final_answers": [str(a) for a in split.responses_final_answers],
        "responses_has_final_answer_line": [bool(v) for v in split.responses_has_final_answer_line],
        "responses_correct": [bool(v) for v in split.responses_correct],
        "responses_fail_type": [str(v) for v in split.responses_fail_type],
        "correct_kept_indices": [int(i) for i in split.correct_kept_indices],
        "wrong_kept_indices": [int(i) for i in split.wrong_kept_indices],
        "n_total": n_total,
        "n_correct_total": n_correct_total,
        "n_wrong_total": n_wrong_total,
        "n_correct_kept": len(split.correct_kept),
        "n_wrong_kept": len(split.wrong_kept),
        "rho_hat": float(rho_hat),
        "prompt_weight": float(prompt_weight),
        "objective_type": objective_type,
        "rollouts": [rollout_trajectory_to_json(t, include_dense=include_dense_rollouts) for t in all_trajectories],
    }
    if objective is not None:
        record["correct_traj_weights"] = [float(v) for v in objective.correct_traj_weights]
        record["mixed_pref_pairs"] = [
            {"correct_idx": int(c_idx), "wrong_idx": int(w_idx)}
            for c_idx, w_idx in objective.mixed_pref_pairs
        ]
        if objective.gt_positive is not None:
            record["gt_positive"] = rollout_trajectory_to_json(
                objective.gt_positive, include_dense=include_dense_rollouts
            )
    return record


def _labeled_batch_tensors(
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts = list(prompt_texts)
    completions = list(completion_texts)
    full_texts = [p + c for p, c in zip(prompts, completions)]
    prompt_ids = tokenizer(
        prompts,
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
        seq_len = int(seq_len)
        if seq_len <= 0:
            continue
        prompt_len = len(prompt_ids[i])
        non_pad_positions = attention_mask[i].nonzero(as_tuple=False)
        if non_pad_positions.numel() == 0:
            continue
        content_start = int(non_pad_positions[0].item())
        content_end = content_start + seq_len
        completion_start = min(content_start + prompt_len, content_end)
        if completion_start < content_end:
            labels[i, completion_start:content_end] = input_ids[i, completion_start:content_end]
    return input_ids, attention_mask, labels


def _seq_logps_from_logits_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    # Use fp32 for vocab normalization, but avoid materializing full-vocab log_softmax.
    shifted_logits_f = shifted_logits.float()
    target_logits = shifted_logits_f.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logps = target_logits - torch.logsumexp(shifted_logits_f, dim=-1)
    token_logps = torch.nan_to_num(token_logps, nan=-20.0, neginf=-20.0, posinf=0.0).clamp_min(-20.0)
    seq_logps = (token_logps * valid_mask).sum(dim=-1)
    seq_logps = seq_logps / valid_mask.sum(dim=-1).clamp_min(1)
    return seq_logps


def _seq_entropy_from_logits_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100).to(shifted_logits.dtype)
    token_log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
    token_log_probs = torch.nan_to_num(token_log_probs, nan=-20.0, neginf=-20.0, posinf=0.0).clamp_min(-20.0)
    token_probs = token_log_probs.exp()
    token_entropy = -(token_probs * token_log_probs).sum(dim=-1)
    seq_entropy = (token_entropy * valid_mask).sum(dim=-1)
    seq_entropy = seq_entropy / valid_mask.sum(dim=-1).clamp_min(1.0)
    return seq_entropy


def _compute_sequence_logps_batch(
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention_mask, labels = _labeled_batch_tensors(
        tokenizer, prompt_texts, completion_texts, max_length, device
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return _seq_logps_from_logits_labels(logits, labels)


def _compute_sequence_logps_and_hidden_batch(
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
    hidden_layer_offset: int,
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, labels = _labeled_batch_tensors(
        tokenizer, prompt_texts, completion_texts, max_length, device
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    seq_logps = _seq_logps_from_logits_labels(outputs.logits, labels)
    if compute_entropy:
        seq_entropy = _seq_entropy_from_logits_labels(outputs.logits, labels)
    else:
        seq_entropy = torch.full_like(seq_logps, float("nan"))
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("Model did not return hidden_states; cannot run hidden-state pair mining.")
    layer_idx = len(hidden_states) - int(hidden_layer_offset)
    layer_idx = max(0, min(layer_idx, len(hidden_states) - 1))
    layer_hidden = hidden_states[layer_idx]
    completion_mask = labels.ne(-100).to(layer_hidden.dtype)
    denom = completion_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (layer_hidden * completion_mask.unsqueeze(-1)).sum(dim=1) / denom
    hidden_vec = F.normalize(pooled, p=2, dim=-1, eps=1e-12)
    return seq_logps, seq_entropy, hidden_vec


def build_rollout_trajectories_for_prompt(
    model: object,
    tokenizer: object,
    device: torch.device,
    train_prompt: str,
    candidates: Sequence[str],
    split: RolloutCandidateSplit,
    args: argparse.Namespace,
) -> List[RolloutTrajectory]:
    if not candidates:
        return []

    token_ids_all = tokenizer(
        list(candidates),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    prompt_texts = [train_prompt for _ in candidates]
    avg_logprobs: List[float] = []
    avg_entropies: List[float] = []
    hidden_vecs: List[List[float]] = []
    mb = args.rollout_feature_micro_batch_size if args.rollout_feature_micro_batch_size > 0 else len(candidates)

    was_training = bool(getattr(model, "training", False))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), mb):
            end = min(start + mb, len(candidates))
            batch_prompts = prompt_texts[start:end]
            batch_completions = list(candidates[start:end])
            seq_logps, seq_entropy, batch_hidden = _compute_sequence_logps_and_hidden_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=batch_prompts,
                completion_texts=batch_completions,
                max_length=args.max_length,
                device=device,
                hidden_layer_offset=args.hidden_layer_offset,
                compute_entropy=bool(getattr(args, "rollout_compute_entropy", True)),
            )
            avg_logprobs.extend([float(v) for v in seq_logps.detach().cpu().tolist()])
            avg_entropies.extend([float(v) for v in seq_entropy.detach().cpu().tolist()])
            hidden_vecs.extend([[float(x) for x in row] for row in batch_hidden.detach().cpu().tolist()])
    if was_training:
        model.train()

    trajectories: List[RolloutTrajectory] = []
    for idx, response_text in enumerate(candidates):
        avg_logprob = float(avg_logprobs[idx])
        trajectories.append(
            RolloutTrajectory(
                response_text=str(response_text),
                token_ids=[int(tid) for tid in token_ids_all[idx]],
                is_correct=bool(split.responses_correct[idx]),
                fail_type=str(split.responses_fail_type[idx]),
                has_final_answer_line=bool(split.responses_has_final_answer_line[idx]),
                final_answer=str(split.responses_final_answers[idx]),
                avg_logprob=avg_logprob,
                avg_nll=-avg_logprob,
                avg_entropy=float(avg_entropies[idx]),
                hidden_vec=list(hidden_vecs[idx]),
            )
        )
    return trajectories


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


def unwrap_model_for_save(model: object) -> object:
    """Return the underlying HF/PEFT model when wrapped by DataParallel."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


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
        unwrap_model_for_save(model).save_pretrained(adapter_dir)
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
        unwrap_model_for_save(model).save_pretrained(vllm_staging_dir)
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
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
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
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
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
    keep_prompts: List[str] = [str(prompt) for prompt in train_prompts]
    keep_chosen: List[str] = [str(ch) for ch in chosen]
    keep_rejected: List[str] = [str(rj) for rj in rejected]
    keep_weights: List[float] = [float(w) for w in weights]
    kept = total
    stats = PairTruncationStats(
        total_pairs=total,
        kept_pairs=kept,
        dropped_pairs=0,
        dropped_prompt_too_long=0,
        dropped_chosen_too_long=0,
        dropped_rejected_too_long=0,
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
    keep_prompts: List[str] = [str(prompt) for prompt in train_prompts]
    keep_completions: List[str] = [str(completion) for completion in completions]
    keep_weights: List[float] = [float(w) for w in weights]
    kept = total
    stats = SftTruncationStats(
        total_samples=total,
        kept_samples=kept,
        dropped_samples=0,
        dropped_prompt_too_long=0,
        dropped_completion_too_long=0,
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
    gt_pref_train_prompts: List[str],
    gt_pref_chosen: List[str],
    gt_pref_rejected: List[str],
    gt_pref_weights: List[float],
    mle_train_prompts: List[str],
    mle_completions: List[str],
    mle_weights: List[float],
) -> OnlineStepLossStats:
    """Single optimizer.step() on weighted mixed-pref + gt-pref + MLE samples."""
    pref_batch = len(pref_train_prompts)
    gt_pref_batch = len(gt_pref_train_prompts)
    mle_batch = len(mle_train_prompts)
    total_weight = float(sum(pref_weights) + sum(gt_pref_weights) + sum(mle_weights))

    def _build_zero_stats(reason: str, grad_norm_value: float = 0.0) -> OnlineStepLossStats:
        return OnlineStepLossStats(
            total_loss=0.0,
            mle_loss=0.0,
            pref_loss=0.0,
            gt_pref_loss=0.0,
            mean_gap=0.0,
            pref_pairs_used=0,
            gt_pref_pairs_used=0,
            mle_samples_used=0,
            lora_mean_abs=0.0,
            lora_max_abs=0.0,
            lora_nan_ratio=0.0,
            lora_inf_ratio=0.0,
            grad_norm=grad_norm_value,
            update_applied=False,
            skip_reason=reason,
        )

    if total_weight <= 0:
        return _build_zero_stats("zero_total_weight")
    mb_pref = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(pref_batch, 1)
    mb_gt_pref = (
        args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(gt_pref_batch, 1)
    )
    mb_mle = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(mle_batch, 1)

    optimizer.zero_grad(set_to_none=True)
    pref_loss_weighted_sum = 0.0
    gt_pref_loss_weighted_sum = 0.0
    mle_loss_weighted_sum = 0.0
    gap_weighted_sum = 0.0
    pref_weight_sum = 0.0
    gt_pref_weight_sum = 0.0
    mle_weight_sum_used = 0.0
    gap_weight_sum = 0.0
    pref_pairs_used = 0
    gt_pref_pairs_used = 0
    mle_samples_used = 0

    def _check_chunk_and_backward(
        loss_chunk: torch.Tensor,
        loss_chunk_val: float,
        skip_prefix: str,
        start: int,
        end: int,
    ) -> Optional[OnlineStepLossStats]:
        if not loss_chunk.requires_grad:
            raise RuntimeError(
                "Online loss has no grad_fn. If use_lora=true with gradient_checkpointing=true, "
                "ensure input grads are enabled for checkpointing."
            )
        if args.online_skip_nonfinite_loss and not torch.isfinite(loss_chunk.detach()):
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_{skip_prefix}_loss_chunk(start={start},end={end})")
        if args.online_loss_value_cap > 0 and abs(loss_chunk_val) > args.online_loss_value_cap:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                f"{skip_prefix}_loss_chunk_too_large(value={loss_chunk_val:.4f},"
                f"cap={args.online_loss_value_cap:.4f})"
            )
        loss_chunk.backward()
        return None

    def _filter_list_by_mask(items: List[str], keep_mask: torch.Tensor) -> List[str]:
        keep = [bool(x) for x in keep_mask.detach().cpu().tolist()]
        return [item for item, should_keep in zip(items, keep) if should_keep]

    def _prefilter_pref_chunk_before_autograd(
        branch: str,
        start: int,
        end: int,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        weights: torch.Tensor,
    ) -> Tuple[List[str], List[str], List[str], torch.Tensor, int]:
        if (
            args.online_pref_min_avg_logprob_chosen is None
            and args.online_pref_min_avg_logprob_rejected is None
            and args.online_gap_clip_abs <= 0
        ):
            return prompts, chosen, rejected, weights, int(weights.numel())

        was_training = bool(getattr(model, "training", False))
        if was_training:
            model.eval()
        try:
            with torch.no_grad():
                chosen_logps = _compute_sequence_logps_batch(
                    model,
                    tokenizer,
                    prompts,
                    chosen,
                    args.max_length,
                    device,
                )
                rejected_logps = _compute_sequence_logps_batch(
                    model,
                    tokenizer,
                    prompts,
                    rejected,
                    args.max_length,
                    device,
                )
        finally:
            if was_training:
                model.train()

        raw_gap = chosen_logps - rejected_logps
        keep_mask = torch.isfinite(chosen_logps) & torch.isfinite(rejected_logps) & torch.isfinite(raw_gap)
        finite_mask = keep_mask.clone()
        if args.online_pref_min_avg_logprob_chosen is not None:
            keep_mask = keep_mask & (chosen_logps >= float(args.online_pref_min_avg_logprob_chosen))
        if args.online_pref_min_avg_logprob_rejected is not None:
            keep_mask = keep_mask & (rejected_logps >= float(args.online_pref_min_avg_logprob_rejected))
        if args.online_gap_clip_abs > 0:
            keep_mask = keep_mask & (torch.abs(raw_gap) <= float(args.online_gap_clip_abs))

        kept = int(keep_mask.sum().item())
        total = int(keep_mask.numel())
        if kept < total:
            chosen_floor_drop = 0
            rejected_floor_drop = 0
            gap_cap_drop = 0
            if args.online_pref_min_avg_logprob_chosen is not None:
                chosen_floor_drop = int(
                    (finite_mask & (chosen_logps < float(args.online_pref_min_avg_logprob_chosen))).sum().item()
                )
            if args.online_pref_min_avg_logprob_rejected is not None:
                rejected_floor_drop = int(
                    (finite_mask & (rejected_logps < float(args.online_pref_min_avg_logprob_rejected))).sum().item()
                )
            if args.online_gap_clip_abs > 0:
                gap_cap_drop = int((finite_mask & (torch.abs(raw_gap) > float(args.online_gap_clip_abs))).sum().item())
            print(
                f"[online] {branch} train_prefilter chunk=[{start},{end}) "
                f"kept={kept}/{total} dropped_before_autograd={total - kept} "
                f"drop_nonfinite={int((~finite_mask).sum().item())} "
                f"drop_chosen_floor={chosen_floor_drop} "
                f"drop_rejected_floor={rejected_floor_drop} "
                f"drop_raw_gap_cap={gap_cap_drop}",
                flush=True,
            )
        if kept <= 0:
            return [], [], [], weights[:0], 0

        keep_mask = keep_mask.detach()
        return (
            _filter_list_by_mask(prompts, keep_mask),
            _filter_list_by_mask(chosen, keep_mask),
            _filter_list_by_mask(rejected, keep_mask),
            weights[keep_mask.to(device=weights.device)],
            kept,
        )

    def _run_pref_like_branch(
        branch: str,
        train_prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        weights: List[float],
        micro_batch: int,
    ) -> Optional[OnlineStepLossStats]:
        nonlocal pref_loss_weighted_sum
        nonlocal gt_pref_loss_weighted_sum
        nonlocal gap_weighted_sum
        nonlocal pref_weight_sum
        nonlocal gt_pref_weight_sum
        nonlocal gap_weight_sum
        nonlocal pref_pairs_used
        nonlocal gt_pref_pairs_used

        branch_batch = len(train_prompts)
        if branch_batch <= 0:
            return None

        for start in range(0, branch_batch, micro_batch):
            end = min(start + micro_batch, branch_batch)
            tp = train_prompts[start:end]
            ch = chosen[start:end]
            rj = rejected[start:end]
            w = torch.tensor(weights[start:end], device=device, dtype=torch.float32)
            tp, ch, rj, w, chunk_pairs_used = _prefilter_pref_chunk_before_autograd(
                branch,
                start,
                end,
                tp,
                ch,
                rj,
                w,
            )
            if chunk_pairs_used <= 0:
                continue
            chosen_logps = _compute_sequence_logps_batch(
                model,
                tokenizer,
                tp,
                ch,
                args.max_length,
                device,
            )
            rejected_logps = _compute_sequence_logps_batch(
                model,
                tokenizer,
                tp,
                rj,
                args.max_length,
                device,
            )
            preference_gap = chosen_logps - rejected_logps
            if args.online_gap_clip_abs > 0:
                preference_gap = preference_gap.clamp(-args.online_gap_clip_abs, args.online_gap_clip_abs)
            pref_like_loss_vec = -F.logsigmoid(args.beta * preference_gap)
            loss_chunk = (pref_like_loss_vec * w).sum() / total_weight
            loss_chunk_val = float(loss_chunk.detach().item())
            failed = _check_chunk_and_backward(
                loss_chunk=loss_chunk,
                loss_chunk_val=loss_chunk_val,
                skip_prefix=branch,
                start=start,
                end=end,
            )
            if failed is not None:
                return failed
            loss_sum = (pref_like_loss_vec * w).sum().item()
            gap_sum = (preference_gap * w).sum().item()
            w_sum = w.sum().item()

            if branch == "pref":
                pref_loss_weighted_sum += loss_sum
                pref_weight_sum += w_sum
                pref_pairs_used += chunk_pairs_used
            else:
                gt_pref_loss_weighted_sum += loss_sum
                gt_pref_weight_sum += w_sum
                gt_pref_pairs_used += chunk_pairs_used

            gap_weighted_sum += gap_sum
            gap_weight_sum += w_sum
        return None

    failed = _run_pref_like_branch(
        branch="pref",
        train_prompts=pref_train_prompts,
        chosen=pref_chosen,
        rejected=pref_rejected,
        weights=pref_weights,
        micro_batch=mb_pref,
    )
    if failed is not None:
        return failed

    failed = _run_pref_like_branch(
        branch="gt_pref",
        train_prompts=gt_pref_train_prompts,
        chosen=gt_pref_chosen,
        rejected=gt_pref_rejected,
        weights=gt_pref_weights,
        micro_batch=mb_gt_pref,
    )
    if failed is not None:
        return failed

    if mle_batch > 0:
        for start in range(0, mle_batch, mb_mle):
            end = min(start + mb_mle, mle_batch)
            tp = mle_train_prompts[start:end]
            cp = mle_completions[start:end]
            w = torch.tensor(mle_weights[start:end], device=device, dtype=torch.float32)
            logps = _compute_sequence_logps_batch(
                model,
                tokenizer,
                tp,
                cp,
                args.max_length,
                device,
            )
            mle_loss_vec = -logps
            loss_chunk = (mle_loss_vec * w).sum() / total_weight
            loss_chunk_val = float(loss_chunk.detach().item())
            failed = _check_chunk_and_backward(
                loss_chunk=loss_chunk,
                loss_chunk_val=loss_chunk_val,
                skip_prefix="mle",
                start=start,
                end=end,
            )
            if failed is not None:
                return failed
            mle_loss_weighted_sum += (mle_loss_vec * w).sum().item()
            mle_weight_sum_used += w.sum().item()
            mle_samples_used += int(w.numel())

    if pref_weight_sum + gt_pref_weight_sum + mle_weight_sum_used <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("all_train_samples_filtered_before_autograd")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.max_grad_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
    else:
        grad_norm_parts = [torch.linalg.vector_norm(p.grad.detach(), ord=2) for p in trainable if p.grad is not None]
        if grad_norm_parts:
            total_norm = torch.linalg.vector_norm(torch.stack(grad_norm_parts), ord=2)
        else:
            total_norm = torch.tensor(0.0, device=device)
    grad_norm = float(total_norm.item()) if isinstance(total_norm, torch.Tensor) else float(total_norm)
    if args.online_skip_nonfinite_loss and not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("nonfinite_grad_norm", grad_norm_value=grad_norm)
    if args.online_hard_grad_norm_cap > 0 and grad_norm > args.online_hard_grad_norm_cap:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats(
            (
                f"grad_norm_too_large(value={grad_norm:.4f},"
                f"cap={args.online_hard_grad_norm_cap:.4f})"
            ),
            grad_norm_value=grad_norm,
        )
    optimizer.step()

    mean_gap = gap_weighted_sum / gap_weight_sum if gap_weight_sum > 0 else 0.0
    pref_loss = pref_loss_weighted_sum / pref_weight_sum if pref_weight_sum > 0 else 0.0
    gt_pref_loss = gt_pref_loss_weighted_sum / gt_pref_weight_sum if gt_pref_weight_sum > 0 else 0.0
    mle_weight_sum = float(sum(mle_weights))
    mle_loss = mle_loss_weighted_sum / mle_weight_sum if mle_weight_sum > 0 else 0.0
    total_loss = (pref_loss_weighted_sum + gt_pref_loss_weighted_sum + mle_loss_weighted_sum) / total_weight
    lora_health = _compute_lora_param_health(model)
    if args.online_abort_on_lora_nan and lora_health["lora_nan_ratio"] > 0:
        raise RuntimeError(
            "Detected NaN in LoRA params after optimizer.step: "
            f"lora_nan_ratio={lora_health['lora_nan_ratio']:.6f}"
        )
    return OnlineStepLossStats(
        total_loss=total_loss,
        mle_loss=mle_loss,
        pref_loss=pref_loss,
        gt_pref_loss=gt_pref_loss,
        mean_gap=mean_gap,
        pref_pairs_used=pref_pairs_used,
        gt_pref_pairs_used=gt_pref_pairs_used,
        mle_samples_used=mle_samples_used,
        lora_mean_abs=lora_health["lora_mean_abs"],
        lora_max_abs=lora_health["lora_max_abs"],
        lora_nan_ratio=lora_health["lora_nan_ratio"],
        lora_inf_ratio=lora_health["lora_inf_ratio"],
        grad_norm=grad_norm,
        update_applied=True,
        skip_reason="",
    )


def run_online_preference_training(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    online_pairs_path = output_root / "online_pairs.jsonl"
    metrics_jsonl_path = output_root / "training_metrics.jsonl"

    def _write_metric(event: str, payload: Dict[str, Any]) -> None:
        rec = {"event": event, **payload}
        with metrics_jsonl_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    model_path = args.model_path.strip()
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
    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if bool(getattr(args, "hf_data_parallel", True)) and device.type == "cuda" and cuda_device_count > 1:
        model = torch.nn.DataParallel(model)
        print(f"[online] enabled DataParallel for training across {cuda_device_count} GPUs")
    if args.gradient_checkpointing:
        base_model = unwrap_model_for_save(model)
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False
        if args.use_lora:
            ensure_input_require_grads_for_checkpointing(base_model)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260412)

    layout = args.dataset_layout
    if layout == "auto":
        layout = detect_parquet_dataset_layout(args.dataset_path)
    if layout == "dapo":
        source_iter = iter_dapo_samples(
            parquet_path=args.dataset_path,
            scan_batch_size=args.scan_batch_size,
            max_source_samples=args.max_source_samples,
            gold_rationale_key_paths=args.gold_rationale_key,
            require_gold_rationale=args.require_gold_rationale_for_all_wrong,
        )
    elif layout == "math_hf":
        source_iter = iter_math_hf_samples(
            parquet_path=args.dataset_path,
            scan_batch_size=args.scan_batch_size,
            max_source_samples=args.max_source_samples,
            gold_rationale_key_paths=(),
            require_gold_rationale=args.require_gold_rationale_for_all_wrong,
        )
    else:
        raise ValueError(f"Unsupported --dataset_layout: {layout}")

    rollout_user_suffix = str(args.user_content_suffix or "")
    if (
        args.auto_math_hf_user_suffix
        and layout == "math_hf"
        and not rollout_user_suffix.strip()
    ):
        rollout_user_suffix = DEFAULT_MATH_HF_USER_CONTENT_SUFFIX

    updates = 0
    rollout_steps = 0
    scanned = 0
    kept_pref_pairs = 0
    kept_gt_pref_pairs = 0
    kept_mle_samples = 0
    skipped_all_wrong = 0
    skipped_after_filter = 0
    logged_mixed_objectives = 0
    logged_all_correct_objectives = 0
    logged_all_wrong_objectives = 0
    buffer: List[DapoSample] = []
    k = args.online_pairs_per_step

    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"
    print(
        f"[online] dataset_layout={layout}, "
        f"user_content_suffix_chars={len(rollout_user_suffix)}, "
        f"rollout_backend={args.online_rollout_backend}, "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"({args.rollout_n} samples per prompt via n={args.rollout_n}), "
        f"online_pairs_per_step={k} (chunk size within one rollout step), "
        f"online_steps={total_steps_str}, max_source_samples={args.max_source_samples}, "
        f"vllm_enforce_eager={args.online_vllm_enforce_eager}, "
        f"prompt_smoothing=({args.prompt_smoothing_alpha},{args.prompt_smoothing_beta}), "
        f"prompt_gamma={args.prompt_weight_gamma}, "
        f"prompt_weight_clip=[{args.prompt_weight_min},{args.prompt_weight_max}], "
        f"pos_weight_mode={args.positive_weight_mode}, "
        f"online_mle_on_correct_only={args.online_mle_on_correct_only}, "
        f"online_pref_loss_only={args.online_pref_loss_only}, "
        f"lambda_mle={args.lambda_mle}, lambda_pref={args.lambda_pref}, lambda_gt={args.lambda_gt}, "
        f"gap_clip_abs={args.online_gap_clip_abs}, "
        f"pref_min_avg_logprob_chosen={args.online_pref_min_avg_logprob_chosen}, "
        f"pref_min_avg_logprob_rejected={args.online_pref_min_avg_logprob_rejected}"
    )
    _write_metric(
        "run_start",
        {
            "output_dir": str(output_root),
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_path),
            "online_steps": args.online_steps,
            "online_pairs_per_step": args.online_pairs_per_step,
            "rollout_n": args.rollout_n,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "lambda_mle": args.lambda_mle,
            "lambda_pref": args.lambda_pref,
            "lambda_gt": args.lambda_gt,
            "metrics_jsonl": str(metrics_jsonl_path),
        },
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
                    s.prompt + rollout_user_suffix,
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
            mixed_objectives_in_rollout = 0
            all_correct_objectives_in_rollout = 0
            all_wrong_objectives_in_rollout = 0
            skipped_all_wrong_in_rollout = 0
            skipped_after_filter_in_rollout = 0
            sampled_correct_total_in_rollout = 0
            sampled_candidates_total_in_rollout = 0
            rollout_all_entropy_values: List[float] = []
            rollout_correct_entropy_values: List[float] = []
            rollout_wrong_entropy_values: List[float] = []
            rollout_gt_entropy_values: List[float] = []
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
                )
                n_total = len(candidates)
                n_correct_total = sum(1 for x in split.responses_correct if x)
                sampled_correct_total_in_rollout += int(n_correct_total)
                sampled_candidates_total_in_rollout += int(n_total)
                rho_hat = compute_smoothed_correct_rate(
                    r_cnt=n_correct_total,
                    total=n_total,
                    alpha=args.prompt_smoothing_alpha,
                    beta=args.prompt_smoothing_beta,
                )
                prompt_weight = compute_prompt_rarity_weight(
                    rho_hat=rho_hat,
                    gamma=args.prompt_weight_gamma,
                    w_min=args.prompt_weight_min,
                    w_max=args.prompt_weight_max,
                )

                if n_correct_total == 0 and (
                    not args.use_all_wrong_gt_preference or float(args.lambda_gt) <= 0.0
                ):
                    skipped_all_wrong += 1
                    skipped_all_wrong_in_rollout += 1
                    record = build_online_bootstrap_jsonl_record(
                        sample_id=sample_obj.sample_id,
                        prompt=sample_obj.prompt,
                        prompt_user_effective=sample_obj.prompt + rollout_user_suffix,
                        system_prompt=system_prompts[idx],
                        ground_truth=sample_obj.ground_truth,
                        candidates=candidates,
                        split=split,
                        objective=None,
                        prompt_weight=prompt_weight,
                        rho_hat=rho_hat,
                        all_trajectories=[],
                        include_dense_rollouts=False,
                    )
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                trajectories = build_rollout_trajectories_for_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    train_prompt=prompt_texts[idx],
                    candidates=candidates,
                    split=split,
                    args=args,
                )
                rollout_all_entropy_values.extend(
                    [float(t.avg_entropy) for t in trajectories if math.isfinite(float(t.avg_entropy))]
                )
                correct_trajs = [trajectories[i] for i in split.correct_kept_indices]
                wrong_trajs = [trajectories[i] for i in split.wrong_kept_indices]
                rollout_correct_entropy_values.extend(
                    [float(t.avg_entropy) for t in correct_trajs if math.isfinite(float(t.avg_entropy))]
                )
                rollout_wrong_entropy_values.extend(
                    [float(t.avg_entropy) for t in wrong_trajs if math.isfinite(float(t.avg_entropy))]
                )
                objective: Optional[OnlinePendingObjective] = None

                if args.online_pref_loss_only:
                    if n_correct_total > 0 and n_correct_total < n_total and correct_trajs and wrong_trajs:
                        mixed_pairs = build_hidden_nn_pairs(correct_trajs, wrong_trajs)
                        mixed_pairs = filter_mixed_pref_pairs_by_avg_logprob(
                            mixed_pairs,
                            correct_trajs,
                            wrong_trajs,
                            args.online_pref_min_avg_logprob_chosen,
                            args.online_pref_min_avg_logprob_rejected,
                        )
                        if mixed_pairs:
                            objective = OnlinePendingObjective(
                                sample_id=sample_obj.sample_id,
                                ground_truth=sample_obj.ground_truth,
                                train_prompt=prompt_texts[idx],
                                objective_type="mixed_pref_only",
                                rho_hat=rho_hat,
                                prompt_weight=prompt_weight,
                                correct=correct_trajs,
                                wrong=wrong_trajs,
                                correct_traj_weights=[],
                                mixed_pref_pairs=mixed_pairs,
                                gt_positive=None,
                            )
                            rollout_objectives.append(objective)
                            mixed_objectives_in_rollout += 1
                            logged_mixed_objectives += 1
                        else:
                            skipped_after_filter += 1
                            skipped_after_filter_in_rollout += 1
                    else:
                        skipped_after_filter += 1
                        skipped_after_filter_in_rollout += 1
                elif args.online_mle_on_correct_only:
                    if correct_trajs:
                        correct_weights = compute_correct_trajectory_weights(
                            correct_trajs=correct_trajs,
                            mode=args.positive_weight_mode,
                            tau=args.positive_weight_tau,
                        )
                        objective_type = "all_correct" if n_correct_total == n_total else "correct_only_mle"
                        objective = OnlinePendingObjective(
                            sample_id=sample_obj.sample_id,
                            ground_truth=sample_obj.ground_truth,
                            train_prompt=prompt_texts[idx],
                            objective_type=objective_type,
                            rho_hat=rho_hat,
                            prompt_weight=prompt_weight,
                            correct=correct_trajs,
                            wrong=[],
                            correct_traj_weights=correct_weights,
                            mixed_pref_pairs=[],
                            gt_positive=None,
                        )
                        rollout_objectives.append(objective)
                        if n_correct_total == n_total:
                            all_correct_objectives_in_rollout += 1
                            logged_all_correct_objectives += 1
                        else:
                            mixed_objectives_in_rollout += 1
                            logged_mixed_objectives += 1
                    else:
                        skipped_after_filter += 1
                        skipped_after_filter_in_rollout += 1
                elif n_correct_total > 0 and n_correct_total < n_total:
                    if correct_trajs and wrong_trajs:
                        correct_weights = compute_correct_trajectory_weights(
                            correct_trajs=correct_trajs,
                            mode=args.positive_weight_mode,
                            tau=args.positive_weight_tau,
                        )
                        mixed_pairs = build_hidden_nn_pairs(correct_trajs, wrong_trajs)
                        mixed_pairs = filter_mixed_pref_pairs_by_avg_logprob(
                            mixed_pairs,
                            correct_trajs,
                            wrong_trajs,
                            args.online_pref_min_avg_logprob_chosen,
                            args.online_pref_min_avg_logprob_rejected,
                        )
                        if mixed_pairs:
                            objective = OnlinePendingObjective(
                                sample_id=sample_obj.sample_id,
                                ground_truth=sample_obj.ground_truth,
                                train_prompt=prompt_texts[idx],
                                objective_type="mixed",
                                rho_hat=rho_hat,
                                prompt_weight=prompt_weight,
                                correct=correct_trajs,
                                wrong=wrong_trajs,
                                correct_traj_weights=correct_weights,
                                mixed_pref_pairs=mixed_pairs,
                                gt_positive=None,
                            )
                            rollout_objectives.append(objective)
                            mixed_objectives_in_rollout += 1
                            logged_mixed_objectives += 1
                        else:
                            skipped_after_filter += 1
                            skipped_after_filter_in_rollout += 1
                    else:
                        skipped_after_filter += 1
                        skipped_after_filter_in_rollout += 1
                elif n_correct_total == n_total:
                    if correct_trajs:
                        correct_weights = compute_correct_trajectory_weights(
                            correct_trajs=correct_trajs,
                            mode=args.positive_weight_mode,
                            tau=args.positive_weight_tau,
                        )
                        objective = OnlinePendingObjective(
                            sample_id=sample_obj.sample_id,
                            ground_truth=sample_obj.ground_truth,
                            train_prompt=prompt_texts[idx],
                            objective_type="all_correct",
                            rho_hat=rho_hat,
                            prompt_weight=prompt_weight,
                            correct=correct_trajs,
                            wrong=[],
                            correct_traj_weights=correct_weights,
                            mixed_pref_pairs=[],
                            gt_positive=None,
                        )
                        rollout_objectives.append(objective)
                        all_correct_objectives_in_rollout += 1
                        logged_all_correct_objectives += 1
                    else:
                        skipped_after_filter += 1
                        skipped_after_filter_in_rollout += 1
                else:
                    gt_positive: Optional[RolloutTrajectory] = None
                    if args.use_all_wrong_gt_preference and wrong_trajs:
                        gt_text = strip_prompt_prefix_from_text(
                            sample_obj.prompt,
                            sample_obj.gold_rationale,
                        )

                        if not gt_text and sample_obj.ground_truth:
                            gt_text = f"Answer: {sample_obj.ground_truth}"
                        if gt_text:
                            gt_final_answer = extract_reference_answer_for_verifier(gt_text)
                            gt_has_final = bool(gt_final_answer)
                            gt_is_correct = answer_text_matches(
                                gt_final_answer if gt_has_final else "",
                                sample_obj.ground_truth,
                            )
                            gt_split = RolloutCandidateSplit(
                                responses_has_final_answer_line=[gt_has_final],
                                responses_final_answers=[gt_final_answer if gt_has_final else ""],
                                responses_correct=[gt_is_correct],
                                responses_fail_type=["correct" if gt_is_correct else "wrong_answer"],
                                correct_kept_indices=[0] if gt_is_correct else [],
                                wrong_kept_indices=[] if gt_is_correct else [0],
                                correct_kept=[gt_text] if gt_is_correct else [],
                                wrong_kept=[] if gt_is_correct else [gt_text],
                            )
                            gt_trajectories = build_rollout_trajectories_for_prompt(
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                train_prompt=prompt_texts[idx],
                                candidates=[gt_text],
                                split=gt_split,
                                args=args,
                            )
                            if gt_trajectories and gt_is_correct:
                                gt_positive = gt_trajectories[0]

                    if gt_positive is not None and wrong_trajs:
                        if math.isfinite(float(gt_positive.avg_entropy)):
                            rollout_all_entropy_values.append(float(gt_positive.avg_entropy))
                            rollout_gt_entropy_values.append(float(gt_positive.avg_entropy))
                        objective = OnlinePendingObjective(
                            sample_id=sample_obj.sample_id,
                            ground_truth=sample_obj.ground_truth,
                            train_prompt=prompt_texts[idx],
                            objective_type="all_wrong",
                            rho_hat=rho_hat,
                            prompt_weight=prompt_weight,
                            correct=[],
                            wrong=wrong_trajs,
                            correct_traj_weights=[],
                            mixed_pref_pairs=[],
                            gt_positive=gt_positive,
                        )
                        rollout_objectives.append(objective)
                        all_wrong_objectives_in_rollout += 1
                        logged_all_wrong_objectives += 1
                    else:
                        skipped_all_wrong += 1
                        skipped_all_wrong_in_rollout += 1

                record = build_online_bootstrap_jsonl_record(
                    sample_id=sample_obj.sample_id,
                    prompt=sample_obj.prompt,
                    prompt_user_effective=sample_obj.prompt + rollout_user_suffix,
                    system_prompt=system_prompts[idx],
                    ground_truth=sample_obj.ground_truth,
                    candidates=candidates,
                    split=split,
                    objective=objective,
                    prompt_weight=prompt_weight,
                    rho_hat=rho_hat,
                    all_trajectories=trajectories,
                    include_dense_rollouts=args.online_pairs_include_dense_rollouts,
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            fout.flush()

            print(
                f"[online] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"mixed_in_rollout={mixed_objectives_in_rollout} "
                f"all_correct_in_rollout={all_correct_objectives_in_rollout} "
                f"all_wrong_in_rollout={all_wrong_objectives_in_rollout} "
                f"skipped_all_wrong_in_rollout={skipped_all_wrong_in_rollout} "
                f"skipped_after_filter_in_rollout={skipped_after_filter_in_rollout} "
                f"objectives_ready_for_update={len(rollout_objectives)}"
            )
            ent_overall = _mean_or_nan(rollout_all_entropy_values)
            ent_correct = _mean_or_nan(rollout_correct_entropy_values)
            ent_wrong = _mean_or_nan(rollout_wrong_entropy_values)
            ent_gt = _mean_or_nan(rollout_gt_entropy_values)
            ent_gap_wrong_minus_correct = (
                float(ent_wrong - ent_correct)
                if not math.isnan(ent_wrong) and not math.isnan(ent_correct)
                else float("nan")
            )
            print(
                f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                f"entropy_overall_mean={ent_overall:.4f} "
                f"entropy_overall_count={len(rollout_all_entropy_values)} "
                f"entropy_correct_mean={ent_correct:.4f} "
                f"entropy_wrong_mean={ent_wrong:.4f} "
                f"entropy_gt_ref_mean={ent_gt:.4f} "
                f"entropy_gap_wrong_minus_correct={ent_gap_wrong_minus_correct:.4f}"
            )
            sampled_correct_rate = (
                float(sampled_correct_total_in_rollout) / float(sampled_candidates_total_in_rollout)
                if sampled_candidates_total_in_rollout > 0
                else 0.0
            )
            _write_metric(
                "rollout_summary",
                {
                    "rollout_step": int(rollout_steps),
                    "scanned": int(scanned),
                    "mixed_in_rollout": int(mixed_objectives_in_rollout),
                    "all_correct_in_rollout": int(all_correct_objectives_in_rollout),
                    "all_wrong_in_rollout": int(all_wrong_objectives_in_rollout),
                    "skipped_all_wrong_in_rollout": int(skipped_all_wrong_in_rollout),
                    "skipped_after_filter_in_rollout": int(skipped_after_filter_in_rollout),
                    "objectives_ready_for_update": int(len(rollout_objectives)),
                    "sampled_correct_total": int(sampled_correct_total_in_rollout),
                    "sampled_candidates_total": int(sampled_candidates_total_in_rollout),
                    "sampled_correct_rate": float(sampled_correct_rate),
                    "entropy_overall_mean": float(ent_overall),
                    "entropy_overall_count": int(len(rollout_all_entropy_values)),
                    "entropy_correct_mean": float(ent_correct),
                    "entropy_wrong_mean": float(ent_wrong),
                    "entropy_gt_ref_mean": float(ent_gt),
                    "entropy_gap_wrong_minus_correct": float(ent_gap_wrong_minus_correct),
                },
            )

            updates_in_rollout = 0
            consumed_pref_pairs_in_rollout = 0
            consumed_gt_pref_pairs_in_rollout = 0
            consumed_mle_samples_in_rollout = 0
            dropped_pref_pairs_by_truncation_in_rollout = 0
            dropped_gt_pref_pairs_by_truncation_in_rollout = 0
            dropped_mle_by_truncation_in_rollout = 0
            last_optimizer_skip_reason: Optional[str] = None
            if rollout_objectives:
                for chunk_start in range(0, len(rollout_objectives), k):
                    chunk = rollout_objectives[chunk_start : chunk_start + k]
                    pref_train_prompts_raw: List[str] = []
                    pref_chosen_raw: List[str] = []
                    pref_rejected_raw: List[str] = []
                    pref_weights_raw: List[float] = []
                    gt_pref_train_prompts_raw: List[str] = []
                    gt_pref_chosen_raw: List[str] = []
                    gt_pref_rejected_raw: List[str] = []
                    gt_pref_weights_raw: List[float] = []
                    mle_train_prompts_raw: List[str] = []
                    mle_completions_raw: List[str] = []
                    mle_weights_raw: List[float] = []

                    for objective in chunk:
                        if objective.objective_type == "mixed":
                            for traj, traj_weight in zip(objective.correct, objective.correct_traj_weights):
                                mle_train_prompts_raw.append(objective.train_prompt)
                                mle_completions_raw.append(traj.response_text)
                                mle_weights_raw.append(
                                    float(args.lambda_mle) * float(objective.prompt_weight) * float(traj_weight)
                                )
                            if objective.mixed_pref_pairs:
                                pair_weight = float(args.lambda_pref) / float(len(objective.mixed_pref_pairs))
                                for c_idx, w_idx in objective.mixed_pref_pairs:
                                    pref_train_prompts_raw.append(objective.train_prompt)
                                    pref_chosen_raw.append(objective.correct[c_idx].response_text)
                                    pref_rejected_raw.append(objective.wrong[w_idx].response_text)
                                    pref_weights_raw.append(pair_weight)
                        elif objective.objective_type in {"all_correct", "correct_only_mle"}:
                            for traj, traj_weight in zip(objective.correct, objective.correct_traj_weights):
                                mle_train_prompts_raw.append(objective.train_prompt)
                                mle_completions_raw.append(traj.response_text)
                                mle_weights_raw.append(
                                    float(args.lambda_mle) * float(objective.prompt_weight) * float(traj_weight)
                                )
                        elif objective.objective_type == "mixed_pref_only":
                            if objective.mixed_pref_pairs:
                                pair_weight = float(args.lambda_pref) / float(len(objective.mixed_pref_pairs))
                                for c_idx, w_idx in objective.mixed_pref_pairs:
                                    pref_train_prompts_raw.append(objective.train_prompt)
                                    pref_chosen_raw.append(objective.correct[c_idx].response_text)
                                    pref_rejected_raw.append(objective.wrong[w_idx].response_text)
                                    pref_weights_raw.append(pair_weight)
                        elif objective.objective_type == "all_wrong":
                            if objective.gt_positive is None or not objective.wrong:
                                continue
                            gt_wrong_kept = [
                                w
                                for w in objective.wrong
                                if _pref_pair_passes_avg_logprob_floor(
                                    objective.gt_positive.avg_logprob,
                                    w.avg_logprob,
                                    args.online_pref_min_avg_logprob_chosen,
                                    args.online_pref_min_avg_logprob_rejected,
                                )
                            ]
                            if not gt_wrong_kept:
                                continue
                            pair_weight = float(args.lambda_gt) / float(len(gt_wrong_kept))
                            for wrong_traj in gt_wrong_kept:
                                gt_pref_train_prompts_raw.append(objective.train_prompt)
                                gt_pref_chosen_raw.append(objective.gt_positive.response_text)
                                gt_pref_rejected_raw.append(wrong_traj.response_text)
                                gt_pref_weights_raw.append(pair_weight)

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

                    (
                        gt_pref_train_prompts,
                        gt_pref_chosen,
                        gt_pref_rejected,
                        gt_pref_weights,
                        gt_pref_trunc_stats,
                    ) = filter_weighted_pairs_without_truncation(
                        tokenizer=tokenizer,
                        train_prompts=gt_pref_train_prompts_raw,
                        chosen=gt_pref_chosen_raw,
                        rejected=gt_pref_rejected_raw,
                        weights=gt_pref_weights_raw,
                        max_length=args.max_length,
                    )
                    dropped_gt_pref_pairs_by_truncation_in_rollout += gt_pref_trunc_stats.dropped_pairs

                    (
                        mle_train_prompts,
                        mle_completions,
                        mle_weights,
                        mle_trunc_stats,
                    ) = filter_weighted_sft_without_truncation(
                        tokenizer=tokenizer,
                        train_prompts=mle_train_prompts_raw,
                        completions=mle_completions_raw,
                        weights=mle_weights_raw,
                        max_length=args.max_length,
                    )
                    dropped_mle_by_truncation_in_rollout += mle_trunc_stats.dropped_samples

                    if not pref_train_prompts and not gt_pref_train_prompts and not mle_train_prompts:
                        continue
                    if (sum(pref_weights) + sum(gt_pref_weights) + sum(mle_weights)) <= 0:
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
                        gt_pref_train_prompts=gt_pref_train_prompts,
                        gt_pref_chosen=gt_pref_chosen,
                        gt_pref_rejected=gt_pref_rejected,
                        gt_pref_weights=gt_pref_weights,
                        mle_train_prompts=mle_train_prompts,
                        mle_completions=mle_completions,
                        mle_weights=mle_weights,
                    )
                    if not loss_stats.update_applied:
                        last_optimizer_skip_reason = loss_stats.skip_reason
                        print(
                            f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                            f"skip optimizer update reason={loss_stats.skip_reason} "
                            f"grad_norm={loss_stats.grad_norm:.4f}"
                        )
                        _write_metric(
                            "optimizer_step_skipped",
                            {
                                "rollout_step": int(rollout_steps),
                                "optimizer_step": int(updates),
                                "skip_reason": str(loss_stats.skip_reason),
                                "grad_norm": float(loss_stats.grad_norm),
                            },
                        )
                        continue
                    updates += 1
                    updates_in_rollout += 1
                    consumed_pref_pairs_in_rollout += loss_stats.pref_pairs_used
                    consumed_gt_pref_pairs_in_rollout += loss_stats.gt_pref_pairs_used
                    consumed_mle_samples_in_rollout += loss_stats.mle_samples_used
                    kept_pref_pairs += loss_stats.pref_pairs_used
                    kept_gt_pref_pairs += loss_stats.gt_pref_pairs_used
                    kept_mle_samples += loss_stats.mle_samples_used
                    print(
                        f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                        f"optimizer_step={updates} "
                        f"pref_loss={loss_stats.pref_loss:.6f} "
                        f"mean_gap={loss_stats.mean_gap:.6f} "
                        f"gt_pref_loss={loss_stats.gt_pref_loss:.6f} "
                        f"mle_loss={loss_stats.mle_loss:.6f} "
                        f"total_loss={loss_stats.total_loss:.6f} "
                        f"grad_norm={loss_stats.grad_norm:.6f}",
                        flush=True,
                    )
                    _write_metric(
                        "optimizer_step",
                        {
                            "rollout_step": int(rollout_steps),
                            "optimizer_step": int(updates),
                            "pref_loss": float(loss_stats.pref_loss),
                            "mean_gap": float(loss_stats.mean_gap),
                            "gt_pref_loss": float(loss_stats.gt_pref_loss),
                            "mle_loss": float(loss_stats.mle_loss),
                            "total_loss": float(loss_stats.total_loss),
                            "grad_norm": float(loss_stats.grad_norm),
                            "pref_pairs_used": int(loss_stats.pref_pairs_used),
                            "gt_pref_pairs_used": int(loss_stats.gt_pref_pairs_used),
                            "mle_samples_used": int(loss_stats.mle_samples_used),
                            "lora_mean_abs": float(loss_stats.lora_mean_abs),
                            "lora_max_abs": float(loss_stats.lora_max_abs),
                            "lora_nan_ratio": float(loss_stats.lora_nan_ratio),
                            "lora_inf_ratio": float(loss_stats.lora_inf_ratio),
                        },
                    )

                    if args.online_save_every_updates > 0 and updates % args.online_save_every_updates == 0:
                        ckpt_dir = output_root / f"checkpoint-update-{updates}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        unwrap_model_for_save(model).save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"[online] saved checkpoint to {ckpt_dir}")

            if rollout_objectives and updates_in_rollout == 0:
                hint = (
                    f"last_skip_reason={last_optimizer_skip_reason!r}"
                    if last_optimizer_skip_reason
                    else "no_chunk_reached_optimizer"
                )
                print(
                    f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                    f"no optimizer update applied ({hint}; see skip lines above or empty batch)"
                )
            elif rollout_objectives:
                print(
                    f"[online] rollout_step={rollout_steps}/{total_steps_str} "
                    f"updates_in_rollout={updates_in_rollout} "
                    f"consumed_pref_pairs_in_rollout={consumed_pref_pairs_in_rollout} "
                    f"consumed_gt_pref_pairs_in_rollout={consumed_gt_pref_pairs_in_rollout} "
                    f"consumed_mle_samples_in_rollout={consumed_mle_samples_in_rollout} "
                    f"dropped_pref_pairs_by_truncation_in_rollout={dropped_pref_pairs_by_truncation_in_rollout} "
                    f"dropped_gt_pref_pairs_by_truncation_in_rollout={dropped_gt_pref_pairs_by_truncation_in_rollout} "
                    f"dropped_mle_by_truncation_in_rollout={dropped_mle_by_truncation_in_rollout}"
                )

            buffer = []
            if args.online_steps is not None and rollout_steps >= args.online_steps:
                break

        if buffer and (args.online_steps is None or rollout_steps < args.online_steps):
            print("[online] remaining tail batch ignored to keep fixed rollout_batch_size behavior")

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model_for_save(model).save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(
        f"[online] finished. rollout_steps={rollout_steps}, optimizer_steps={updates}, "
        f"scanned={scanned}, kept_pref_pairs={kept_pref_pairs}, "
        f"kept_gt_pref_pairs={kept_gt_pref_pairs}, kept_mle_samples={kept_mle_samples}, "
        f"logged_mixed_objectives={logged_mixed_objectives}, "
        f"logged_all_correct_objectives={logged_all_correct_objectives}, "
        f"logged_all_wrong_objectives={logged_all_wrong_objectives}, "
        f"skipped_all_wrong={skipped_all_wrong}, skipped_after_filter={skipped_after_filter}, "
        f"objectives_log={online_pairs_path}, final_model={final_dir}"
    )
    _write_metric(
        "run_end",
        {
            "rollout_steps": int(rollout_steps),
            "optimizer_steps": int(updates),
            "scanned": int(scanned),
            "kept_pref_pairs": int(kept_pref_pairs),
            "kept_gt_pref_pairs": int(kept_gt_pref_pairs),
            "kept_mle_samples": int(kept_mle_samples),
            "logged_mixed_objectives": int(logged_mixed_objectives),
            "logged_all_correct_objectives": int(logged_all_correct_objectives),
            "logged_all_wrong_objectives": int(logged_all_wrong_objectives),
            "skipped_all_wrong": int(skipped_all_wrong),
            "skipped_after_filter": int(skipped_after_filter),
            "objectives_log": str(online_pairs_path),
            "final_model": str(final_dir),
            "metrics_jsonl": str(metrics_jsonl_path),
        },
    )


def main() -> None:
    parser = build_cli_parser(DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()
    set_seed(args.seed)

    launcher_world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if launcher_world_size > 1:
        raise SystemExit(
            "error: train_preference.py is a single-process online trainer. "
            "Do not launch it with torchrun/DDP; use one Python process with "
            "CUDA_VISIBLE_DEVICES=0,1 and --tensor_parallel_size 2 for dual-GPU vLLM rollout."
        )

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
    run_online_preference_training(args)


if __name__ == "__main__":
    main()


