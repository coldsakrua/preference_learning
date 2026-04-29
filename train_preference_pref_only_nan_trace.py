#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import train_preference as base
from utils import set_seed, str2bool


def _first_nonfinite_grad(model: object) -> tuple[str, int, int]:
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return "", 0, 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None or (not getattr(p, "requires_grad", False)):
                continue
            if p.grad is None:
                continue
            g = p.grad.detach()
            nan_cnt = int(torch.isnan(g).sum().item())
            inf_cnt = int(torch.isinf(g).sum().item())
            if nan_cnt > 0 or inf_cnt > 0:
                return str(name), nan_cnt, inf_cnt
    return "", 0, 0


def _first_nonfinite_tensor(t: torch.Tensor, name: str) -> Optional[str]:
    bad = ~torch.isfinite(t)
    bad_count = int(bad.sum().item())
    if bad_count <= 0:
        return None
    finite = t[torch.isfinite(t)]
    if finite.numel() > 0:
        fmin = float(finite.min().item())
        fmax = float(finite.max().item())
        return f"{name}: nonfinite={bad_count}, finite_min={fmin:.6f}, finite_max={fmax:.6f}"
    return f"{name}: nonfinite={bad_count}, no finite values"


def _build_zero_stats(reason: str, grad_norm_value: float = 0.0) -> base.OnlineStepLossStats:
    return base.OnlineStepLossStats(
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


def _trace(msg: str, args: argparse.Namespace) -> None:
    if getattr(args, "nan_trace_verbose", True):
        print(f"[nan-trace] {msg}", flush=True)


def _online_run_pref_only_nan_trace_step(
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
) -> base.OnlineStepLossStats:
    pref_batch = len(pref_train_prompts)
    if gt_pref_train_prompts or gt_pref_chosen or gt_pref_rejected or gt_pref_weights:
        return _build_zero_stats("debug_runner_expected_pref_only_but_gt_branch_nonempty")
    if mle_train_prompts or mle_completions or mle_weights:
        return _build_zero_stats("debug_runner_expected_pref_only_but_mle_branch_nonempty")

    total_weight = float(sum(pref_weights))
    if pref_batch <= 0:
        return _build_zero_stats("empty_pref_batch")
    if total_weight <= 0:
        return _build_zero_stats("zero_total_weight")

    mb_pref = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(pref_batch, 1)

    optimizer.zero_grad(set_to_none=True)
    pref_loss_weighted_sum = 0.0
    gap_weighted_sum = 0.0
    pref_weight_sum = 0.0

    _trace(f"optimizer_step_start pref_batch={pref_batch} total_weight={total_weight:.6f} mb_pref={mb_pref}", args)

    for start in range(0, pref_batch, mb_pref):
        end = min(start + mb_pref, pref_batch)
        tp = pref_train_prompts[start:end]
        ch = pref_chosen[start:end]
        rj = pref_rejected[start:end]
        w = torch.tensor(pref_weights[start:end], device=device, dtype=torch.float32)

        _trace(f"chunk[{start}:{end}] forward/chosen_logps", args)
        chosen_logps = base._compute_sequence_logps_batch(model, tokenizer, tp, ch, args.max_length, device)
        bad_msg = _first_nonfinite_tensor(chosen_logps, f"pref.chosen_logps[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_before_gap({bad_msg})")

        _trace(f"chunk[{start}:{end}] forward/rejected_logps", args)
        rejected_logps = base._compute_sequence_logps_batch(model, tokenizer, tp, rj, args.max_length, device)
        bad_msg = _first_nonfinite_tensor(rejected_logps, f"pref.rejected_logps[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_before_gap({bad_msg})")

        preference_gap = chosen_logps - rejected_logps
        if args.online_gap_clip_abs > 0:
            preference_gap = preference_gap.clamp(-args.online_gap_clip_abs, args.online_gap_clip_abs)
        bad_msg = _first_nonfinite_tensor(preference_gap, f"pref.gap[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_in_gap({bad_msg})")

        pref_loss_vec = -F.logsigmoid(args.beta * preference_gap)
        bad_msg = _first_nonfinite_tensor(pref_loss_vec, f"pref.loss_vec[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_in_pref_loss({bad_msg})")

        loss_chunk = (pref_loss_vec * w).sum() / total_weight
        loss_chunk_val = float(loss_chunk.detach().item())
        if args.online_skip_nonfinite_loss and not torch.isfinite(loss_chunk.detach()):
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_pref_loss_chunk(start={start},end={end})")
        if args.online_loss_value_cap > 0 and abs(loss_chunk_val) > args.online_loss_value_cap:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                f"pref_loss_chunk_too_large(value={loss_chunk_val:.4f},cap={args.online_loss_value_cap:.4f})"
            )

        _trace(f"chunk[{start}:{end}] backward(loss={loss_chunk_val:.6f})", args)
        loss_chunk.backward()

        bad_param, nan_cnt, inf_cnt = _first_nonfinite_grad(model)
        if bad_param:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                "nonfinite_grad_after_backward("
                f"branch=pref,chunk=[{start},{end}),param={bad_param},nan={nan_cnt},inf={inf_cnt})"
            )

        pref_loss_weighted_sum += (pref_loss_vec * w).sum().item()
        gap_weighted_sum += (preference_gap * w).sum().item()
        pref_weight_sum += w.sum().item()

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
    _trace(f"pre_step grad_norm={grad_norm:.6f}", args)

    if args.online_skip_nonfinite_loss and not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("nonfinite_grad_norm", grad_norm_value=grad_norm)
    if args.online_hard_grad_norm_cap > 0 and grad_norm > args.online_hard_grad_norm_cap:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats(
            f"grad_norm_too_large(value={grad_norm:.4f},cap={args.online_hard_grad_norm_cap:.4f})",
            grad_norm_value=grad_norm,
        )

    optimizer.step()

    lora_health = base._compute_lora_param_health(model)
    if args.online_abort_on_lora_nan and lora_health["lora_nan_ratio"] > 0:
        raise RuntimeError(
            "Detected NaN in LoRA params after optimizer.step: "
            f"lora_nan_ratio={lora_health['lora_nan_ratio']:.6f}"
        )

    mean_gap = gap_weighted_sum / pref_weight_sum if pref_weight_sum > 0 else 0.0
    pref_loss = pref_loss_weighted_sum / pref_weight_sum if pref_weight_sum > 0 else 0.0

    _trace(
        f"optimizer_step_ok pref_loss={pref_loss:.6f} mean_gap={mean_gap:.6f} "
        f"lora_nan_ratio={lora_health['lora_nan_ratio']:.6f}",
        args,
    )

    return base.OnlineStepLossStats(
        total_loss=pref_loss,
        mle_loss=0.0,
        pref_loss=pref_loss,
        gt_pref_loss=0.0,
        mean_gap=mean_gap,
        pref_pairs_used=pref_batch,
        gt_pref_pairs_used=0,
        mle_samples_used=0,
        lora_mean_abs=lora_health["lora_mean_abs"],
        lora_max_abs=lora_health["lora_max_abs"],
        lora_nan_ratio=lora_health["lora_nan_ratio"],
        lora_inf_ratio=lora_health["lora_inf_ratio"],
        grad_norm=grad_norm,
        update_applied=True,
        skip_reason="",
    )


def main() -> None:
    parser = base.build_cli_parser(base.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--nan_trace_verbose",
        type=str2bool,
        default=True,
        help="Print detailed step-by-step NaN trace logs.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    # Force pref-only debug mode.
    args.online_pref_loss_only = True
    args.online_mle_on_correct_only = False
    args.lambda_mle = 0.0
    args.lambda_gt = 0.0
    if args.lambda_pref <= 0:
        args.lambda_pref = 1.0
    args.use_all_wrong_gt_preference = False

    # Keep the rollout focused on pairs that are less likely to be pathological.
    if args.online_pref_min_avg_logprob_chosen is None:
        args.online_pref_min_avg_logprob_chosen = -3.0
    if args.online_pref_min_avg_logprob_rejected is None:
        args.online_pref_min_avg_logprob_rejected = -3.0

    print(
        "[nan-trace] mode=pref-only "
        f"lambda_pref={args.lambda_pref} "
        f"logprob_micro_batch_size={args.logprob_micro_batch_size} "
        f"gradient_checkpointing={args.gradient_checkpointing}",
        flush=True,
    )

    # Monkey-patch the optimizer step with detailed trace version.
    base._online_run_preference_optimizer_step = _online_run_pref_only_nan_trace_step

    base.run_online_preference_training(args)


if __name__ == "__main__":
    main()
