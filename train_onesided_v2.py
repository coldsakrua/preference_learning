#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online One-Sided Group MLE — v2.

Compared to ``train_onesided_group_mle.py``, this file adds three method-side
changes targeted at the empirical findings on the qwen3-4b/MATH rollouts:

  1. Token-entropy-weighted MLE on the *correct* responses (``--token_weight_type``)
     - low-entropy boilerplate tokens contribute almost no gradient, the model
       focuses on decision-point tokens. Defaults to ``entropy`` weighting.

  2. Mode-collapse-aware hard-negative filtering before computing the
     one-sided ``hard_weight`` (``--mode_min_cluster``)
     - wrong rollouts are clustered by their normalized final answer; only
       clusters with at least ``--mode_min_cluster`` rollouts are used as
       negatives, the rest are silently ignored. This keeps the negative
       signal restricted to systematic mistakes (~58% of wrongs in our data)
       and discards heterogeneous noise.

  3. Robust verifier normalization to reduce false negatives
     - strips LaTeX inline-math wrappers ``\\(..\\)`` / ``\\[..\\]``,
       ``\\text{...}``, ``\\quad`` etc. before answer comparison. Recovers
       roughly a quarter of ``wrong_answer`` rollouts that are actually
       correct but rejected by exact-match.

Default ``--lambda_group`` is bumped from 0.25 to 1.0 so the group/preference
branch contributes a meaningful share of the loss. ``--prompt_weight_gamma``
is left at the legacy default (1.0) so that the only difficulty signal used
is the on-policy rollout correct rate ρ̂; no external difficulty prior is
assumed about the dataset.

This file does NOT modify ``train_onesided_group_mle.py`` or ``utils.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from train_onesided_group_mle import (
    GroupObjective,
    _load_source_iter,
    _normalize_group_scores,
    _objective_potential_weight,
    _soft_anchor,
    _zero_lora_health,
)
from train_preference import (
    DEFAULT_SYSTEM_PROMPT,
    _compute_lora_param_health,
    _labeled_batch_tensors,
    _online_rollout_completions_flat_hf,
    _online_rollout_completions_flat_vllm,
    _seq_logps_from_logits_labels,
    apply_qwen_chat_template,
    build_prompt_pool,
    choose_system_prompt,
    ensure_input_require_grads_for_checkpointing,
    unwrap_model_for_save,
    wrap_model_with_lora,
)
from utils import (
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    DapoSample,
    answer_text_matches,
    compute_prompt_rarity_weight,
    compute_smoothed_correct_rate,
    extract_rollout_scored_answer,
    normalize_answer,
    set_seed,
    str2bool,
    to_number_if_simple,
)


# =====================================================================
#  Modification 3: robustified verifier (LaTeX wrapper stripping)
# =====================================================================
#
# Empirical observation: ~24.5% of "wrong_answer" rollouts on MATH have a
# parsed final answer that is *textually equivalent* to the gold answer
# but wrapped in ``\( ... \)`` or ``\[ ... \]`` and possibly with a trailing
# ``.`` or surrounding ``\text{...}``. The default ``utils.normalize_answer``
# does not strip these, so the rollouts are scored as wrong and end up in
# the negative pool, contaminating the preference signal.

_LATEX_INLINE_OPEN = re.compile(r"\\[\(\[]")
_LATEX_INLINE_CLOSE = re.compile(r"\\[\)\]]")
_LATEX_TEXT_BLOCK = re.compile(r"\\text\s*\{([^{}]*)\}")
_LATEX_MATHRM_BLOCK = re.compile(r"\\mathrm\s*\{([^{}]*)\}")
_LATEX_SPACING = re.compile(r"\\(?:quad|qquad|,|;|:|!)")
_LATEX_DEGREE = re.compile(r"\\(?:degree|circ)\b")
_LATEX_LEFT_RIGHT = re.compile(r"\\(?:left|right)\b")


def normalize_answer_robust(answer: str) -> str:
    """Stronger wrapper-stripping normalizer; falls through to ``normalize_answer``."""
    if answer is None:
        return ""
    s = str(answer).strip()
    if not s:
        return ""
    for _ in range(3):
        prev = s
        s = _LATEX_INLINE_OPEN.sub("", s)
        s = _LATEX_INLINE_CLOSE.sub("", s)
        s = _LATEX_TEXT_BLOCK.sub(r"\1", s)
        s = _LATEX_MATHRM_BLOCK.sub(r"\1", s)
        s = _LATEX_SPACING.sub("", s)
        s = _LATEX_LEFT_RIGHT.sub("", s)
        s = _LATEX_DEGREE.sub("", s)
        s = s.strip()
        if s == prev:
            break
    return normalize_answer(s)


def answer_text_matches_robust(predicted: str, ground_truth: str) -> bool:
    """First try the canonical matcher; if it fails, retry under robust normalization."""
    if answer_text_matches(predicted, ground_truth):
        return True
    p = normalize_answer_robust(predicted)
    g = normalize_answer_robust(ground_truth)
    if p and p == g:
        return True
    pn = to_number_if_simple(p) if p else None
    gn = to_number_if_simple(g) if g else None
    if pn is not None and gn is not None and abs(pn - gn) <= 1e-6:
        return True
    return False


@dataclass
class RolloutCandidateSplitV2:
    responses_has_final_answer_line: List[bool]
    responses_final_answers: List[str]
    responses_correct: List[bool]
    responses_fail_type: List[str]
    correct_kept_indices: List[int]
    wrong_kept_indices: List[int]
    correct_kept: List[str]
    wrong_kept: List[str]
    n_recovered_by_v2: int


def split_rollout_candidates_v2(
    candidates: Sequence[str], ground_truth: str
) -> RolloutCandidateSplitV2:
    rh: List[bool] = []
    rfa: List[str] = []
    rc: List[bool] = []
    rft: List[str] = []
    cki: List[int] = []
    wki: List[int] = []
    ck: List[str] = []
    wk: List[str] = []
    n_recovered = 0
    for idx, candidate in enumerate(candidates):
        has_final, parsed = extract_rollout_scored_answer(candidate)
        ans = parsed if has_final else ""
        is_correct = False
        if has_final and ans:
            if answer_text_matches(ans, ground_truth):
                is_correct = True
            elif answer_text_matches_robust(ans, ground_truth):
                is_correct = True
                n_recovered += 1
        rh.append(has_final)
        rfa.append(ans)
        rc.append(is_correct)
        if is_correct:
            rft.append("correct")
            cki.append(idx)
            ck.append(str(candidate))
        else:
            if not has_final:
                rft.append("no_final_answer")
            elif not ans:
                rft.append("empty_final_answer")
            else:
                rft.append("wrong_answer")
            wki.append(idx)
            wk.append(str(candidate))
    return RolloutCandidateSplitV2(
        responses_has_final_answer_line=rh,
        responses_final_answers=rfa,
        responses_correct=rc,
        responses_fail_type=rft,
        correct_kept_indices=cki,
        wrong_kept_indices=wki,
        correct_kept=ck,
        wrong_kept=wk,
        n_recovered_by_v2=n_recovered,
    )


# =====================================================================
#  Modification 1: token-level entropy-weighted MLE on correct
# =====================================================================
#
# We compute per-sequence weighted -mean(log_p) where weights are
# detached and derived from token-level entropy / surprise.
# The gradient-flowing path uses logsumexp+gather (no full softmax
# materialization), and the weight signal is computed under no_grad to
# keep activation memory comparable to the legacy uniform path.


def _token_logps_with_grad(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (token_logps[B,T-1], log_norm[B,T-1], valid_mask[B,T-1]). Gradient flows."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    shifted_logits_f = shifted_logits.float()
    log_norm = torch.logsumexp(shifted_logits_f, dim=-1)
    target_logits = shifted_logits_f.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_logps = target_logits - log_norm
    token_logps = torch.nan_to_num(
        token_logps, nan=-20.0, neginf=-20.0, posinf=0.0
    ).clamp_min(-20.0)
    return token_logps, log_norm, valid_mask


def _seq_logps_token_weighted_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight_type: str,
    weight_alpha: float,
    weight_topk_pct: float,
) -> torch.Tensor:
    """Per-sequence weighted mean log_p; gradient flows through token_logps only."""
    token_logps, log_norm, valid_mask = _token_logps_with_grad(logits, labels)
    valid_f = valid_mask.to(token_logps.dtype)
    if weight_type == "uniform":
        denom = valid_f.sum(dim=-1).clamp_min(1.0)
        return (token_logps * valid_f).sum(dim=-1) / denom

    with torch.no_grad():
        if weight_type in ("entropy", "topk_entropy"):
            shifted_logits_f = logits[:, :-1, :].float()
            log_probs = shifted_logits_f - log_norm.unsqueeze(-1)
            log_probs = torch.nan_to_num(
                log_probs, nan=-20.0, neginf=-20.0, posinf=0.0
            ).clamp_min(-20.0)
            signal = -(log_probs.exp() * log_probs).sum(dim=-1)
        elif weight_type == "surprise":
            signal = -token_logps.detach()
        else:
            raise ValueError(f"Unsupported --token_weight_type: {weight_type}")

        signal = signal * valid_f

        if weight_type == "topk_entropy":
            keep_pct = float(weight_topk_pct)
            keep_pct = max(0.0, min(1.0, keep_pct))
            weights = torch.zeros_like(signal)
            for i in range(signal.shape[0]):
                vp = valid_mask[i].nonzero(as_tuple=False).squeeze(-1)
                if vp.numel() == 0:
                    continue
                n_keep = max(1, int(math.ceil(vp.numel() * keep_pct)))
                n_keep = min(n_keep, int(vp.numel()))
                sig_i = signal[i, vp]
                top_idx = torch.topk(sig_i, k=n_keep, dim=0).indices
                weights[i, vp[top_idx]] = 1.0
        else:
            sig_sum = signal.sum(dim=-1, keepdim=True)
            denom_v = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_sig = (sig_sum / denom_v).clamp_min(1e-6)
            scale = signal / mean_sig
            cap = 1.0 + float(weight_alpha) * 8.0
            weights = (1.0 + float(weight_alpha) * scale) * valid_f
            weights = weights.clamp(min=0.0, max=cap)

    weight_sum = weights.sum(dim=-1).clamp_min(1e-6)
    return (token_logps * weights).sum(dim=-1) / weight_sum


def _compute_seq_logps_token_weighted(
    *,
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
    weight_type: str,
    weight_alpha: float,
    weight_topk_pct: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, labels = _labeled_batch_tensors(
        tokenizer, prompt_texts, completion_texts, max_length, device
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    seq_logps = _seq_logps_token_weighted_from_logits(
        logits,
        labels,
        weight_type=weight_type,
        weight_alpha=weight_alpha,
        weight_topk_pct=weight_topk_pct,
    )
    valid_counts = labels[:, 1:].ne(-100).sum(dim=-1)
    return seq_logps, valid_counts


def _compute_seq_logps_no_grad(
    *,
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        input_ids, attention_mask, labels = _labeled_batch_tensors(
            tokenizer, prompt_texts, completion_texts, max_length, device
        )
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        seq_logps = _seq_logps_from_logits_labels(logits, labels)
        valid_counts = labels[:, 1:].ne(-100).sum(dim=-1)
        return seq_logps, valid_counts


# =====================================================================
#  Modification 2: mode-collapse aware filtering of wrong rollouts
# =====================================================================
#
# A wrong rollout enters the negative pool only if at least
# ``--mode_min_cluster`` other wrong rollouts on the same prompt agree on
# the same normalized final answer. This restricts negative gradient to
# systematic misconceptions and discards heterogeneous noise.


def _select_mode_cluster_wrong_indices(
    wrong_responses: Sequence[str],
    *,
    min_cluster: int,
) -> Tuple[List[int], int]:
    """Return (indices into wrong_responses kept, number of distinct clusters)."""
    if min_cluster <= 1:
        keys: List[str] = []
        for resp in wrong_responses:
            has_final, parsed = extract_rollout_scored_answer(resp or "")
            if has_final and parsed:
                keys.append(normalize_answer_robust(parsed))
            else:
                keys.append("")
        keep = [i for i, k in enumerate(keys) if k]
        return keep, len(set(k for k in keys if k))

    keys2: List[Optional[str]] = []
    for resp in wrong_responses:
        has_final, parsed = extract_rollout_scored_answer(resp or "")
        if has_final and parsed:
            k = normalize_answer_robust(parsed)
            keys2.append(k or None)
        else:
            keys2.append(None)
    counts: Counter = Counter(k for k in keys2 if k)
    keep_idx: List[int] = []
    cluster_keys: set = set()
    for i, k in enumerate(keys2):
        if k is None:
            continue
        if counts.get(k, 0) >= int(min_cluster):
            keep_idx.append(i)
            cluster_keys.add(k)
    return keep_idx, len(cluster_keys)


# =====================================================================
#  Optimizer step (v2)
# =====================================================================


@dataclass
class OptimizerStepStatsV2:
    total_loss: float
    mle_loss: float
    group_loss: float
    mean_gap: float
    group_correct_mass: float
    hard_weight: float
    groups_used: int
    mle_samples_used: int
    mode_clusters_used: int
    mode_wrongs_kept: int
    mode_wrongs_rejected: int
    grad_norm: float
    update_applied: bool
    skip_reason: str
    lora_health: Dict[str, float]


def _zero_stats_v2(reason: str = "", grad_norm: float = 0.0) -> OptimizerStepStatsV2:
    return OptimizerStepStatsV2(
        total_loss=0.0,
        mle_loss=0.0,
        group_loss=0.0,
        mean_gap=0.0,
        group_correct_mass=0.0,
        hard_weight=0.0,
        groups_used=0,
        mle_samples_used=0,
        mode_clusters_used=0,
        mode_wrongs_kept=0,
        mode_wrongs_rejected=0,
        grad_norm=grad_norm,
        update_applied=False,
        skip_reason=reason,
        lora_health=_zero_lora_health(),
    )


def _detached_hard_weight(
    pos_scores: torch.Tensor,
    wrong_scores: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        gp = pos_scores.detach()
        gw = wrong_scores.detach()
        all_s = torch.cat([gp, gw], dim=0)
        norm = _normalize_group_scores(all_s, args)
        pos_anchor = _soft_anchor(norm[: gp.shape[0]], float(args.group_tau))
        wrong_anchor = _soft_anchor(norm[gp.shape[0] :], float(args.group_tau))
        gap = pos_anchor - wrong_anchor
        viol = (
            (wrong_anchor - pos_anchor + float(args.group_margin))
            / float(args.hard_weight_tau)
        )
        if args.one_sided_weight_type == "logsigmoid":
            hw = F.softplus(viol)
        elif args.one_sided_weight_type == "hinge":
            hw = F.relu(viol)
        else:
            raise ValueError(f"Unsupported one_sided_weight_type: {args.one_sided_weight_type}")
        if args.hard_weight_min > 0:
            hw = hw.clamp_min(float(args.hard_weight_min))
        if args.hard_weight_max > 0:
            hw = hw.clamp_max(float(args.hard_weight_max))
        conf = torch.sigmoid(
            (pos_anchor - wrong_anchor - float(args.group_margin))
            / float(args.hard_weight_tau)
        )
        return hw.detach(), float(gap.item()), float(conf.item())


def run_optimizer_step_v2(
    *,
    model: object,
    tokenizer: object,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
    objectives: Sequence[GroupObjective],
) -> OptimizerStepStatsV2:
    total_weight = sum(_objective_potential_weight(o, args) for o in objectives)
    if total_weight <= 0:
        return _zero_stats_v2("zero_total_weight")

    optimizer.zero_grad(set_to_none=True)

    total_loss_w = 0.0
    mle_loss_w_sum = 0.0
    group_loss_w_sum = 0.0
    gap_w_sum = 0.0
    conf_w_sum = 0.0
    hw_w_sum = 0.0
    mle_w_total = 0.0
    group_w_total = 0.0
    groups_used = 0
    mle_samples_used = 0
    mode_clusters_used = 0
    mode_wrongs_kept = 0
    mode_wrongs_rejected = 0

    for obj_idx, obj in enumerate(objectives):
        if not obj.correct:
            continue

        pos_prompts = [obj.train_prompt for _ in obj.correct]
        pos_scores, pos_valid_counts = _compute_seq_logps_token_weighted(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=pos_prompts,
            completion_texts=obj.correct,
            max_length=args.max_length,
            device=device,
            weight_type=str(args.token_weight_type),
            weight_alpha=float(args.token_weight_alpha),
            weight_topk_pct=float(args.token_weight_topk_pct),
        )
        pos_valid = torch.isfinite(pos_scores) & pos_valid_counts.gt(0)

        loss_terms: List[torch.Tensor] = []
        obj_mle_val = 0.0
        obj_group_val = 0.0
        obj_group_used = False

        n_pos = int(pos_valid.sum().item())
        if n_pos > 0 and args.lambda_mle > 0:
            mle_loss = -pos_scores[pos_valid].mean()
            mle_wt = float(args.lambda_mle) * float(obj.prompt_weight)
            loss_terms.append(mle_loss * mle_wt)
            obj_mle_val = float(mle_loss.detach().item())
            mle_loss_w_sum += obj_mle_val * mle_wt
            mle_w_total += mle_wt
            mle_samples_used += n_pos

        if obj.is_mixed and args.lambda_group > 0:
            keep_idx, n_clusters = _select_mode_cluster_wrong_indices(
                obj.wrong, min_cluster=int(args.mode_min_cluster)
            )
            mode_wrongs_kept += len(keep_idx)
            mode_wrongs_rejected += max(0, len(obj.wrong) - len(keep_idx))

            if keep_idx:
                wrong_subset = [obj.wrong[i] for i in keep_idx]
                wrong_prompts = [obj.train_prompt for _ in wrong_subset]
                wrong_scores, wrong_valid_counts = _compute_seq_logps_no_grad(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=wrong_prompts,
                    completion_texts=wrong_subset,
                    max_length=args.max_length,
                    device=device,
                )
                wrong_valid = torch.isfinite(wrong_scores) & wrong_valid_counts.gt(0)
                kc = int(pos_valid.sum().item())
                kw = int(wrong_valid.sum().item())
                if kc > 0 and kw > 0:
                    gp = pos_scores[pos_valid]
                    gw = wrong_scores[wrong_valid]
                    hw, gap_v, conf_v = _detached_hard_weight(gp, gw, args)
                    group_pos_ce = -gp.mean()
                    group_loss = hw.to(device=device, dtype=group_pos_ce.dtype) * group_pos_ce
                    group_wt = float(args.lambda_group) * float(obj.prompt_weight)
                    loss_terms.append(group_loss * group_wt)

                    obj_group_used = True
                    obj_group_val = float(group_loss.detach().item())
                    group_loss_w_sum += obj_group_val * group_wt
                    gap_w_sum += gap_v * group_wt
                    conf_w_sum += conf_v * group_wt
                    hw_w_sum += float(hw.item()) * group_wt
                    group_w_total += group_wt
                    groups_used += 1
                    mode_clusters_used += n_clusters

        if not loss_terms:
            continue

        loss_unscaled = torch.stack(loss_terms).sum()
        loss_chunk = loss_unscaled / float(total_weight)
        loss_chunk_val = float(loss_chunk.detach().item())
        if args.online_skip_nonfinite_loss and not torch.isfinite(loss_chunk.detach()):
            optimizer.zero_grad(set_to_none=True)
            return _zero_stats_v2(f"nonfinite_loss(obj={obj_idx})")
        if args.online_loss_value_cap > 0 and abs(loss_chunk_val) > args.online_loss_value_cap:
            optimizer.zero_grad(set_to_none=True)
            return _zero_stats_v2(
                f"loss_too_large(obj={obj_idx},value={loss_chunk_val:.4f})"
            )
        loss_chunk.backward()

        total_loss_w += obj_mle_val * (
            float(args.lambda_mle) * float(obj.prompt_weight) if n_pos > 0 else 0.0
        )
        total_loss_w += obj_group_val * (
            float(args.lambda_group) * float(obj.prompt_weight) if obj_group_used else 0.0
        )

    if mle_w_total + group_w_total <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _zero_stats_v2("all_objectives_filtered")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.max_grad_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
    else:
        parts = [
            torch.linalg.vector_norm(p.grad.detach(), ord=2)
            for p in trainable
            if p.grad is not None
        ]
        total_norm = (
            torch.linalg.vector_norm(torch.stack(parts), ord=2)
            if parts
            else torch.tensor(0.0, device=device)
        )
    grad_norm = float(total_norm.item()) if isinstance(total_norm, torch.Tensor) else float(total_norm)
    if args.online_skip_nonfinite_loss and not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return _zero_stats_v2("nonfinite_grad_norm", grad_norm=grad_norm)
    if args.online_hard_grad_norm_cap > 0 and grad_norm > args.online_hard_grad_norm_cap:
        optimizer.zero_grad(set_to_none=True)
        return _zero_stats_v2(
            f"grad_norm_too_large(value={grad_norm:.4f})", grad_norm=grad_norm
        )

    optimizer.step()
    lora_health = _compute_lora_param_health(model)
    if args.online_abort_on_lora_nan and lora_health["lora_nan_ratio"] > 0:
        raise RuntimeError(
            f"Detected NaN in LoRA params: lora_nan_ratio={lora_health['lora_nan_ratio']:.6f}"
        )

    return OptimizerStepStatsV2(
        total_loss=total_loss_w / total_weight,
        mle_loss=mle_loss_w_sum / mle_w_total if mle_w_total > 0 else 0.0,
        group_loss=group_loss_w_sum / group_w_total if group_w_total > 0 else 0.0,
        mean_gap=gap_w_sum / group_w_total if group_w_total > 0 else 0.0,
        group_correct_mass=conf_w_sum / group_w_total if group_w_total > 0 else 0.0,
        hard_weight=hw_w_sum / group_w_total if group_w_total > 0 else 0.0,
        groups_used=groups_used,
        mle_samples_used=mle_samples_used,
        mode_clusters_used=mode_clusters_used,
        mode_wrongs_kept=mode_wrongs_kept,
        mode_wrongs_rejected=mode_wrongs_rejected,
        grad_norm=grad_norm,
        update_applied=True,
        skip_reason="",
        lora_health=lora_health,
    )


# =====================================================================
#  Rollout record (v2 with recovered count)
# =====================================================================


def _rollout_record_v2(
    *,
    sample: DapoSample,
    split: RolloutCandidateSplitV2,
    objective: Optional[GroupObjective],
    candidates: Sequence[str],
    log_text: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "ground_truth": sample.ground_truth,
        "objective_type": (
            "skip"
            if objective is None
            else ("mixed_group" if objective.is_mixed else "all_correct")
        ),
        "n_total": len(candidates),
        "n_correct": int(sum(1 for x in split.responses_correct if x)),
        "n_wrong": int(sum(1 for x in split.responses_correct if not x)),
        "n_recovered_by_v2": int(split.n_recovered_by_v2),
        "responses_correct": [bool(x) for x in split.responses_correct],
        "responses_fail_type": [str(x) for x in split.responses_fail_type],
        "responses_final_answers": [str(x) for x in split.responses_final_answers],
    }
    if objective is not None:
        record["rho_hat"] = float(objective.rho_hat)
        record["prompt_weight"] = float(objective.prompt_weight)
    if log_text:
        record["prompt"] = sample.prompt
        record["responses"] = [str(x) for x in candidates]
    return record


# =====================================================================
#  Training driver
# =====================================================================


def run_training(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "training_metrics.jsonl"
    rollout_log_path = output_root / "rollout_records.jsonl"

    def write_metric(event: str, payload: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported --torch_dtype: {args.torch_dtype}")
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[args.torch_dtype],
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.use_lora:
        if (
            args.online_rollout_backend == "vllm"
            and args.lora_r > args.vllm_max_lora_rank
        ):
            raise ValueError(
                f"--lora_r must be <= --vllm_max_lora_rank for vLLM LoRA rollout "
                f"(got {args.lora_r} > {args.vllm_max_lora_rank})"
            )
        model = wrap_model_with_lora(model, args)
        model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    if args.hf_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(
            f"[onesided_v2] enabled DataParallel over {torch.cuda.device_count()} GPUs",
            flush=True,
        )
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

    if args.max_source_samples == 0:
        args.max_source_samples = None
    if args.online_steps == 0:
        args.online_steps = None
    layout, source_iter = _load_source_iter(args)
    rollout_user_suffix = str(args.user_content_suffix or "")
    if (
        args.auto_math_hf_user_suffix
        and layout == "math_hf"
        and not rollout_user_suffix.strip()
    ):
        rollout_user_suffix = DEFAULT_MATH_HF_USER_CONTENT_SUFFIX

    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260412)
    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"
    print(
        f"[onesided_v2] dataset_layout={layout} rollout_backend={args.online_rollout_backend} "
        f"rollout_batch_size={args.rollout_batch_size} rollout_n={args.rollout_n} "
        f"objectives_per_update={args.online_pairs_per_step} online_steps={total_steps_str} "
        f"lambda_mle={args.lambda_mle} lambda_group={args.lambda_group} "
        f"token_weight_type={args.token_weight_type} token_weight_alpha={args.token_weight_alpha} "
        f"token_weight_topk_pct={args.token_weight_topk_pct} "
        f"mode_min_cluster={args.mode_min_cluster} "
        f"prompt_weight_gamma={args.prompt_weight_gamma}",
        flush=True,
    )
    write_metric(
        "run_start",
        {
            "output_dir": str(output_root),
            "model_path": str(args.model_path),
            "dataset_path": str(args.dataset_path),
            "dataset_layout": str(layout),
            "lambda_mle": float(args.lambda_mle),
            "lambda_group": float(args.lambda_group),
            "token_weight_type": str(args.token_weight_type),
            "token_weight_alpha": float(args.token_weight_alpha),
            "token_weight_topk_pct": float(args.token_weight_topk_pct),
            "mode_min_cluster": int(args.mode_min_cluster),
            "prompt_weight_gamma": float(args.prompt_weight_gamma),
            "prompt_weight_min": float(args.prompt_weight_min),
            "prompt_weight_max": float(args.prompt_weight_max),
            "rollout_n": int(args.rollout_n),
            "group_tau": float(args.group_tau),
            "hard_weight_tau": float(args.hard_weight_tau),
            "hard_weight_min": float(args.hard_weight_min),
            "hard_weight_max": float(args.hard_weight_max),
        },
    )

    updates = 0
    rollout_steps = 0
    scanned = 0
    kept_groups = 0
    kept_mle_samples = 0
    skipped_all_wrong = 0
    logged_mixed = 0
    logged_all_correct = 0
    total_recovered_by_v2 = 0
    buffer: List[DapoSample] = []

    with rollout_log_path.open("w", encoding="utf-8", buffering=1) as rollout_log:
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

            if args.online_rollout_backend == "vllm":
                if device.type != "cuda":
                    raise RuntimeError("online_rollout_backend=vllm requires CUDA.")
                completion_flat = _online_rollout_completions_flat_vllm(
                    args,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_texts=prompt_texts,
                    rollout_steps=rollout_steps,
                    total_steps_str=total_steps_str,
                    init_model_path=args.model_path,
                    vllm_staging_dir=output_root / "vllm_rollout_ckpt",
                    hf_updates_so_far=updates,
                )
            else:
                completion_flat = _online_rollout_completions_flat_hf(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_texts=prompt_texts,
                    args=args,
                )
            model.train()

            rollout_objectives: List[GroupObjective] = []
            sampled_correct_total = 0
            sampled_total = 0
            mixed_in_rollout = 0
            all_correct_in_rollout = 0
            skipped_all_wrong_in_rollout = 0
            recovered_in_rollout = 0

            for idx, sample_obj in enumerate(buffer):
                start = idx * args.rollout_n
                end = start + args.rollout_n
                candidates = completion_flat[start:end]
                if len(candidates) != args.rollout_n:
                    raise RuntimeError(
                        f"Rollout candidate count mismatch at sample {idx}: "
                        f"expected {args.rollout_n}, got {len(candidates)}"
                    )
                split = split_rollout_candidates_v2(candidates, sample_obj.ground_truth)
                recovered_in_rollout += split.n_recovered_by_v2
                total_recovered_by_v2 += split.n_recovered_by_v2
                n_total = len(candidates)
                n_correct = int(sum(1 for x in split.responses_correct if x))
                sampled_correct_total += n_correct
                sampled_total += n_total
                objective: Optional[GroupObjective] = None
                if n_correct <= 0:
                    skipped_all_wrong += 1
                    skipped_all_wrong_in_rollout += 1
                else:
                    rho_hat = compute_smoothed_correct_rate(
                        r_cnt=n_correct,
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
                    objective = GroupObjective(
                        sample_id=sample_obj.sample_id,
                        ground_truth=sample_obj.ground_truth,
                        train_prompt=prompt_texts[idx],
                        correct=[str(x) for x in split.correct_kept],
                        wrong=[str(x) for x in split.wrong_kept],
                        rho_hat=rho_hat,
                        prompt_weight=prompt_weight,
                    )
                    rollout_objectives.append(objective)
                    if objective.is_mixed:
                        mixed_in_rollout += 1
                        logged_mixed += 1
                    else:
                        all_correct_in_rollout += 1
                        logged_all_correct += 1

                rollout_log.write(
                    json.dumps(
                        _rollout_record_v2(
                            sample=sample_obj,
                            split=split,
                            objective=objective,
                            candidates=candidates,
                            log_text=args.log_rollout_text,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            sampled_correct_rate = (
                sampled_correct_total / sampled_total if sampled_total > 0 else 0.0
            )
            print(
                f"[onesided_v2] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"mixed={mixed_in_rollout} all_correct={all_correct_in_rollout} "
                f"skipped_all_wrong={skipped_all_wrong_in_rollout} "
                f"sampled_correct_rate={sampled_correct_rate:.4f} "
                f"objectives={len(rollout_objectives)} "
                f"recovered_by_v2={recovered_in_rollout} (cum={total_recovered_by_v2})",
                flush=True,
            )
            write_metric(
                "rollout_summary",
                {
                    "rollout_step": int(rollout_steps),
                    "scanned": int(scanned),
                    "mixed": int(mixed_in_rollout),
                    "all_correct": int(all_correct_in_rollout),
                    "skipped_all_wrong": int(skipped_all_wrong_in_rollout),
                    "sampled_correct_rate": float(sampled_correct_rate),
                    "objectives": int(len(rollout_objectives)),
                    "recovered_by_v2": int(recovered_in_rollout),
                    "total_recovered_by_v2": int(total_recovered_by_v2),
                },
            )

            updates_in_rollout = 0
            for chunk_start in range(0, len(rollout_objectives), args.online_pairs_per_step):
                chunk = rollout_objectives[chunk_start : chunk_start + args.online_pairs_per_step]
                if not chunk:
                    continue
                stats = run_optimizer_step_v2(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                    objectives=chunk,
                )
                if not stats.update_applied:
                    print(
                        f"[onesided_v2] rollout_step={rollout_steps}/{total_steps_str} "
                        f"skip optimizer update reason={stats.skip_reason} "
                        f"grad_norm={stats.grad_norm:.4f}",
                        flush=True,
                    )
                    write_metric(
                        "optimizer_step_skipped",
                        {
                            "rollout_step": int(rollout_steps),
                            "optimizer_step": int(updates),
                            "skip_reason": stats.skip_reason,
                            "grad_norm": float(stats.grad_norm),
                        },
                    )
                    continue
                updates += 1
                updates_in_rollout += 1
                kept_groups += stats.groups_used
                kept_mle_samples += stats.mle_samples_used
                print(
                    f"[onesided_v2] rollout_step={rollout_steps}/{total_steps_str} "
                    f"optimizer_step={updates} mle_loss={stats.mle_loss:.6f} "
                    f"group_loss={stats.group_loss:.6f} mean_gap={stats.mean_gap:.6f} "
                    f"pref_conf={stats.group_correct_mass:.6f} hw={stats.hard_weight:.6f} "
                    f"mode_clusters={stats.mode_clusters_used} "
                    f"mode_kept/total={stats.mode_wrongs_kept}/"
                    f"{stats.mode_wrongs_kept + stats.mode_wrongs_rejected} "
                    f"total_loss={stats.total_loss:.6f} grad_norm={stats.grad_norm:.6f}",
                    flush=True,
                )
                write_metric(
                    "optimizer_step",
                    {
                        "rollout_step": int(rollout_steps),
                        "optimizer_step": int(updates),
                        "mle_loss": float(stats.mle_loss),
                        "group_loss": float(stats.group_loss),
                        "mean_gap": float(stats.mean_gap),
                        "group_correct_mass": float(stats.group_correct_mass),
                        "pref_confidence": float(stats.group_correct_mass),
                        "hard_weight": float(stats.hard_weight),
                        "total_loss": float(stats.total_loss),
                        "grad_norm": float(stats.grad_norm),
                        "groups_used": int(stats.groups_used),
                        "mle_samples_used": int(stats.mle_samples_used),
                        "mode_clusters_used": int(stats.mode_clusters_used),
                        "mode_wrongs_kept": int(stats.mode_wrongs_kept),
                        "mode_wrongs_rejected": int(stats.mode_wrongs_rejected),
                        **{k: float(v) for k, v in stats.lora_health.items()},
                    },
                )
                if (
                    args.online_save_every_updates > 0
                    and updates % args.online_save_every_updates == 0
                ):
                    ckpt_dir = output_root / f"checkpoint-update-{updates}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrap_model_for_save(model).save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[onesided_v2] saved checkpoint to {ckpt_dir}", flush=True)

            print(
                f"[onesided_v2] rollout_step={rollout_steps}/{total_steps_str} "
                f"updates_in_rollout={updates_in_rollout}",
                flush=True,
            )
            buffer = []
            if args.online_steps is not None and rollout_steps >= args.online_steps:
                break

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model_for_save(model).save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(
        f"[onesided_v2] finished rollout_steps={rollout_steps} optimizer_steps={updates} "
        f"scanned={scanned} kept_groups={kept_groups} kept_mle_samples={kept_mle_samples} "
        f"logged_mixed={logged_mixed} logged_all_correct={logged_all_correct} "
        f"skipped_all_wrong={skipped_all_wrong} "
        f"total_recovered_by_v2={total_recovered_by_v2} final_model={final_dir}",
        flush=True,
    )
    write_metric(
        "run_end",
        {
            "rollout_steps": int(rollout_steps),
            "optimizer_steps": int(updates),
            "scanned": int(scanned),
            "kept_groups": int(kept_groups),
            "kept_mle_samples": int(kept_mle_samples),
            "logged_mixed": int(logged_mixed),
            "logged_all_correct": int(logged_all_correct),
            "skipped_all_wrong": int(skipped_all_wrong),
            "total_recovered_by_v2": int(total_recovered_by_v2),
            "final_model": str(final_dir),
            "metrics_jsonl": str(metrics_path),
            "rollout_log": str(rollout_log_path),
        },
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
#  CLI
# =====================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Online MLE + token-entropy weighted positive CE + "
            "mode-collapse aware one-sided hard-negative weighting."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--dataset_layout", type=str, default="auto", choices=["auto", "dapo", "math_hf"]
    )
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument("--max_source_samples", type=int, default=0)
    parser.add_argument("--user_content_suffix", type=str, default="")
    parser.add_argument("--auto_math_hf_user_suffix", type=str2bool, default=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--prompt_mode", type=str, default="none", choices=["none", "fixed", "random"]
    )
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt_candidate", action="append", default=[])
    parser.add_argument("--prompt_candidates_file", type=str, default="")
    parser.add_argument("--use_default_prompt_candidates", type=str2bool, default=False)
    parser.add_argument("--prompt_fixed_index", type=int, default=0)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)

    parser.add_argument(
        "--online_rollout_backend", type=str, default="vllm", choices=["vllm", "hf"]
    )
    parser.add_argument("--online_vllm_use_tqdm", type=str2bool, default=True)
    parser.add_argument("--online_vllm_enforce_eager", type=str2bool, default=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--rollout_max_model_len", type=int, default=4096)
    parser.add_argument("--rollout_batch_size", type=int, default=32)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    parser.add_argument("--online_steps", type=int, default=0)
    parser.add_argument(
        "--online-pairs-per-step",
        dest="online_pairs_per_step",
        type=int,
        default=8,
    )
    parser.add_argument("--online_save_every_updates", type=int, default=0)
    parser.add_argument("--log_rollout_text", type=str2bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_mle", type=float, default=1.0)
    parser.add_argument("--lambda_group", type=float, default=1.0)
    parser.add_argument("--group_tau", type=float, default=0.5)
    parser.add_argument(
        "--group_score_norm", type=str, default="none", choices=["none", "zscore"]
    )
    parser.add_argument("--group_score_std_floor", type=float, default=0.05)
    parser.add_argument("--group_score_clip_abs", type=float, default=0.0)
    parser.add_argument(
        "--one_sided_weight_type",
        type=str,
        default="logsigmoid",
        choices=["logsigmoid", "hinge"],
    )
    parser.add_argument("--group_margin", type=float, default=0.0)
    parser.add_argument("--hard_weight_tau", type=float, default=0.5)
    parser.add_argument("--hard_weight_min", type=float, default=0.0)
    parser.add_argument("--hard_weight_max", type=float, default=2.0)
    parser.add_argument("--prompt_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--prompt_smoothing_beta", type=float, default=1.0)
    parser.add_argument("--prompt_weight_gamma", type=float, default=1.0)
    parser.add_argument("--prompt_weight_min", type=float, default=0.1)
    parser.add_argument("--prompt_weight_max", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--online_hard_grad_norm_cap", type=float, default=5.0)
    parser.add_argument("--online_loss_value_cap", type=float, default=20.0)
    parser.add_argument("--online_skip_nonfinite_loss", type=str2bool, default=True)
    parser.add_argument("--online_abort_on_lora_nan", type=str2bool, default=True)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--hf_data_parallel", type=str2bool, default=True)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)

    parser.add_argument(
        "--token_weight_type",
        type=str,
        default="entropy",
        choices=["uniform", "entropy", "surprise", "topk_entropy"],
        help=(
            "Per-token weighting in MLE on correct. "
            "'entropy': w_t = 1 + alpha * (H_t / mean_H_per_seq) "
            "(default; emphasizes decision-point tokens, suppresses boilerplate). "
            "'surprise': w_t uses -log_pi(y_t) instead of entropy. "
            "'topk_entropy': only top --token_weight_topk_pct tokens by entropy "
            "receive weight 1, the rest get 0. "
            "'uniform': legacy plain mean."
        ),
    )
    parser.add_argument(
        "--token_weight_alpha",
        type=float,
        default=1.0,
        help="Strength of entropy/surprise weighting; 0 reduces to uniform.",
    )
    parser.add_argument(
        "--token_weight_topk_pct",
        type=float,
        default=0.2,
        help=(
            "Used only with --token_weight_type=topk_entropy: "
            "fraction (0,1] of highest-entropy positions to keep."
        ),
    )

    parser.add_argument(
        "--mode_min_cluster",
        type=int,
        default=2,
        help=(
            "Minimum number of wrong rollouts agreeing on the same normalized "
            "final answer for them to be used as one-sided hard negatives. "
            "Wrongs that do not belong to such a cluster are silently dropped "
            "from the group loss."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validations = [
        (args.rollout_n < 2, "error: --rollout_n must be >= 2"),
        (args.rollout_batch_size < 1, "error: --rollout_batch_size must be >= 1"),
        (args.online_pairs_per_step < 1, "error: --online-pairs-per-step must be >= 1"),
        (
            args.lambda_mle < 0 or args.lambda_group < 0,
            "error: --lambda_mle/--lambda_group must be >= 0",
        ),
        (
            args.lambda_mle == 0 and args.lambda_group == 0,
            "error: at least one of --lambda_mle/--lambda_group must be > 0",
        ),
        (args.group_tau <= 0, "error: --group_tau must be > 0"),
        (args.hard_weight_tau <= 0, "error: --hard_weight_tau must be > 0"),
        (args.token_weight_alpha < 0, "error: --token_weight_alpha must be >= 0"),
        (
            args.token_weight_topk_pct <= 0 or args.token_weight_topk_pct > 1.0,
            "error: --token_weight_topk_pct must be in (0, 1]",
        ),
        (args.mode_min_cluster < 1, "error: --mode_min_cluster must be >= 1"),
        (args.group_score_std_floor <= 0, "error: --group_score_std_floor must be > 0"),
        (args.group_score_clip_abs < 0, "error: --group_score_clip_abs must be >= 0"),
        (args.hard_weight_min < 0, "error: --hard_weight_min must be >= 0"),
        (args.hard_weight_max < 0, "error: --hard_weight_max must be >= 0"),
        (
            args.hard_weight_max > 0 and args.hard_weight_max < args.hard_weight_min,
            "error: --hard_weight_max must be 0 or >= --hard_weight_min",
        ),
        (args.max_length < 1, "error: --max_length must be >= 1"),
    ]
    for failed, message in validations:
        if failed:
            raise SystemExit(message)
    launcher_world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if launcher_world_size > 1:
        raise SystemExit(
            "error: train_onesided_v2.py is single-process. Use CUDA_VISIBLE_DEVICES "
            "plus --tensor_parallel_size for vLLM rollout, not torchrun/DDP."
        )
    set_seed(args.seed)
    run_training(args)


if __name__ == "__main__":
    main()
