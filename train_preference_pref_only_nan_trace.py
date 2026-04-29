#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import train_preference as base
from utils import set_seed, str2bool

_ORIG_WRAP_MODEL_WITH_LORA = base.wrap_model_with_lora


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


def _collect_nonfinite_grad_params(model: object, max_items: int) -> List[Dict[str, Any]]:
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return []
    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None or (not getattr(p, "requires_grad", False)) or p.grad is None:
                continue
            g = p.grad.detach()
            nan_cnt = int(torch.isnan(g).sum().item())
            inf_cnt = int(torch.isinf(g).sum().item())
            if nan_cnt <= 0 and inf_cnt <= 0:
                continue
            finite_mask = torch.isfinite(g)
            finite_cnt = int(finite_mask.sum().item())
            max_abs_finite = 0.0
            if finite_cnt > 0:
                max_abs_finite = float(g[finite_mask].abs().max().item())
            out.append(
                {
                    "name": str(name),
                    "shape": list(g.shape),
                    "numel": int(g.numel()),
                    "nan_cnt": nan_cnt,
                    "inf_cnt": inf_cnt,
                    "finite_cnt": finite_cnt,
                    "max_abs_finite": max_abs_finite,
                }
            )
            if len(out) >= max_items:
                break
    return out


def _tensor_stats(t: torch.Tensor, name: str) -> Dict[str, Any]:
    x = t.detach()
    numel = int(x.numel())
    finite_mask = torch.isfinite(x)
    finite_cnt = int(finite_mask.sum().item())
    nan_cnt = int(torch.isnan(x).sum().item())
    inf_cnt = int(torch.isinf(x).sum().item())
    out: Dict[str, Any] = {
        "name": name,
        "shape": list(x.shape),
        "numel": numel,
        "finite_cnt": finite_cnt,
        "nan_cnt": nan_cnt,
        "inf_cnt": inf_cnt,
    }
    if finite_cnt > 0:
        f = x[finite_mask]
        out.update(
            {
                "min": float(f.min().item()),
                "max": float(f.max().item()),
                "mean": float(f.mean().item()),
                "std": float(f.std(unbiased=False).item()) if finite_cnt > 1 else 0.0,
                "max_abs": float(f.abs().max().item()),
            }
        )
    return out


def _stats_line(stats: Dict[str, Any]) -> str:
    base_line = (
        f"{stats['name']} shape={stats['shape']} numel={stats['numel']} "
        f"finite={stats['finite_cnt']} nan={stats['nan_cnt']} inf={stats['inf_cnt']}"
    )
    if "min" not in stats:
        return base_line
    return (
        base_line
        + f" min={stats['min']:.6f} max={stats['max']:.6f}"
        + f" mean={stats['mean']:.6f} std={stats['std']:.6f} max_abs={stats['max_abs']:.6f}"
    )


def _grad_overview(model: object) -> Dict[str, Any]:
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return {}
    total_numel = 0
    finite_cnt = 0
    nan_cnt = 0
    inf_cnt = 0
    grad_param_cnt = 0
    nonfinite_param_cnt = 0
    max_abs_finite = 0.0
    with torch.no_grad():
        for _name, p in model.named_parameters():
            if p is None or (not getattr(p, "requires_grad", False)) or p.grad is None:
                continue
            grad_param_cnt += 1
            g = p.grad.detach()
            total_numel += int(g.numel())
            finite_mask = torch.isfinite(g)
            cur_finite = int(finite_mask.sum().item())
            cur_nan = int(torch.isnan(g).sum().item())
            cur_inf = int(torch.isinf(g).sum().item())
            finite_cnt += cur_finite
            nan_cnt += cur_nan
            inf_cnt += cur_inf
            if cur_nan > 0 or cur_inf > 0:
                nonfinite_param_cnt += 1
            if cur_finite > 0:
                max_abs_finite = max(max_abs_finite, float(g[finite_mask].abs().max().item()))
    return {
        "grad_param_cnt": grad_param_cnt,
        "nonfinite_param_cnt": nonfinite_param_cnt,
        "total_numel": total_numel,
        "finite_cnt": finite_cnt,
        "nan_cnt": nan_cnt,
        "inf_cnt": inf_cnt,
        "max_abs_finite": max_abs_finite,
    }


def _lora_param_init_stats(model: object, max_items: int) -> Dict[str, Any]:
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return {}
    total_numel = 0
    finite_cnt = 0
    nan_cnt = 0
    inf_cnt = 0
    max_abs = 0.0
    items: List[Dict[str, Any]] = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" not in name:
                continue
            if p is None:
                continue
            x = p.detach().float()
            n = int(x.numel())
            total_numel += n
            cur_nan = int(torch.isnan(x).sum().item())
            cur_inf = int(torch.isinf(x).sum().item())
            cur_finite = int(torch.isfinite(x).sum().item())
            nan_cnt += cur_nan
            inf_cnt += cur_inf
            finite_cnt += cur_finite
            cur_max_abs = 0.0
            cur_mean = 0.0
            cur_std = 0.0
            finite_mask = torch.isfinite(x)
            if cur_finite > 0:
                xf = x[finite_mask]
                cur_max_abs = float(xf.abs().max().item())
                cur_mean = float(xf.mean().item())
                cur_std = float(xf.std(unbiased=False).item()) if cur_finite > 1 else 0.0
                max_abs = max(max_abs, cur_max_abs)
            if len(items) < max_items:
                items.append(
                    {
                        "name": str(name),
                        "shape": list(x.shape),
                        "numel": n,
                        "nan_cnt": cur_nan,
                        "inf_cnt": cur_inf,
                        "finite_cnt": cur_finite,
                        "mean": cur_mean,
                        "std": cur_std,
                        "max_abs": cur_max_abs,
                    }
                )
    return {
        "lora_param_cnt_shown": len(items),
        "lora_total_numel": total_numel,
        "lora_finite_cnt": finite_cnt,
        "lora_nan_cnt": nan_cnt,
        "lora_inf_cnt": inf_cnt,
        "lora_global_max_abs": max_abs,
        "items": items,
    }


def _wrap_model_with_lora_and_log(model: Any, args: argparse.Namespace) -> Any:
    wrapped = _ORIG_WRAP_MODEL_WITH_LORA(model, args)
    stats = _lora_param_init_stats(wrapped, max_items=args.nan_trace_max_bad_params)
    if stats:
        _trace(
            "lora_init_overview "
            f"total_numel={stats.get('lora_total_numel', 0)} "
            f"finite={stats.get('lora_finite_cnt', 0)} "
            f"nan={stats.get('lora_nan_cnt', 0)} inf={stats.get('lora_inf_cnt', 0)} "
            f"global_max_abs={stats.get('lora_global_max_abs', 0.0):.6f}",
            args,
        )
        for item in stats.get("items", []):
            _trace(
                "lora_init_param "
                f"name={item['name']} shape={item['shape']} numel={item['numel']} "
                f"nan={item['nan_cnt']} inf={item['inf_cnt']} finite={item['finite_cnt']} "
                f"mean={item['mean']:.6f} std={item['std']:.6f} max_abs={item['max_abs']:.6f}",
                args,
            )
        _trace_event(args, "lora_init_stats", stats)
    return wrapped


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


def _trace_event(args: argparse.Namespace, event: str, payload: Dict[str, Any]) -> None:
    if not getattr(args, "nan_trace_write_jsonl", True):
        return
    path = getattr(args, "_nan_trace_jsonl_path", "")
    if not path:
        return
    rec = {"event": event, **payload}
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _second_filter_pref_pairs(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    raw_gap = chosen_logps - rejected_logps
    finite_mask = torch.isfinite(chosen_logps) & torch.isfinite(rejected_logps) & torch.isfinite(raw_gap)
    chosen_ok = chosen_logps >= float(args.train_pref_second_filter_min_chosen_logp)
    rejected_ok = rejected_logps >= float(args.train_pref_second_filter_min_rejected_logp)
    gap_ok = torch.abs(raw_gap) <= float(args.train_pref_second_filter_max_abs_raw_gap)
    keep_mask = finite_mask & chosen_ok & rejected_ok & gap_ok

    return {
        "keep_mask": keep_mask,
        "raw_gap": raw_gap,
        "drop_nonfinite": int((~finite_mask).sum().item()),
        "drop_chosen_floor": int((finite_mask & (~chosen_ok)).sum().item()),
        "drop_rejected_floor": int((finite_mask & (~rejected_ok)).sum().item()),
        "drop_raw_gap_cap": int((finite_mask & (~gap_ok)).sum().item()),
        "kept": int(keep_mask.sum().item()),
        "total": int(keep_mask.numel()),
    }


def _trace_grad_of_tensor(t: torch.Tensor, name: str, args: argparse.Namespace) -> None:
    grad = getattr(t, "grad", None)
    if grad is None:
        _trace(f"{name}.grad is None", args)
        _trace_event(args, "tensor_grad_missing", {"name": name})
        return
    stats = _tensor_stats(grad, f"{name}.grad")
    _trace(_stats_line(stats), args)
    _trace_event(args, "tensor_grad_stats", stats)


def _backward_with_optional_anomaly(
    loss: torch.Tensor,
    args: argparse.Namespace,
    *,
    tag: str,
) -> Optional[str]:
    try:
        if getattr(args, "nan_trace_detect_anomaly", False):
            with torch.autograd.detect_anomaly(check_nan=True):
                loss.backward()
        else:
            loss.backward()
        return None
    except RuntimeError as e:
        return f"{tag}: backward_exception={type(e).__name__}: {e}"


def _run_chunk_backward_probes(
    model: object,
    tokenizer: object,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
    tp: List[str],
    ch: List[str],
    rj: List[str],
    w: torch.Tensor,
    total_weight: float,
    keep_mask: Optional[torch.Tensor],
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    probe_results: List[Dict[str, Any]] = []
    denom = float(total_weight) if float(total_weight) > 0 else 1.0
    probe_defs = ["chosen_linear", "rejected_linear", "gap_linear", "pref_logsigmoid"]

    def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        return x[mask.to(device=x.device)]

    for probe_name in probe_defs:
        optimizer.zero_grad(set_to_none=True)
        chosen_lp = base._compute_sequence_logps_batch(model, tokenizer, tp, ch, args.max_length, device)
        rejected_lp = base._compute_sequence_logps_batch(model, tokenizer, tp, rj, args.max_length, device)
        w_local = w

        if keep_mask is not None:
            chosen_lp = _apply_mask(chosen_lp, keep_mask)
            rejected_lp = _apply_mask(rejected_lp, keep_mask)
            w_local = _apply_mask(w_local, keep_mask)

        num_pairs = int(chosen_lp.numel())
        if num_pairs <= 0:
            probe_results.append(
                {
                    "probe": probe_name,
                    "status": "skipped_empty_after_mask",
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                }
            )
            continue

        raw_gap = chosen_lp - rejected_lp
        gap_for_pref = raw_gap
        if args.online_gap_clip_abs > 0:
            gap_for_pref = gap_for_pref.clamp(-args.online_gap_clip_abs, args.online_gap_clip_abs)

        if probe_name == "chosen_linear":
            loss_probe = -((chosen_lp * w_local).sum() / denom)
        elif probe_name == "rejected_linear":
            loss_probe = (rejected_lp * w_local).sum() / denom
        elif probe_name == "gap_linear":
            loss_probe = -((raw_gap * w_local).sum() / denom)
        else:
            loss_probe = (-F.logsigmoid(args.beta * gap_for_pref) * w_local).sum() / denom

        result: Dict[str, Any] = {
            "probe": probe_name,
            "chunk_start": int(start),
            "chunk_end": int(end),
            "num_pairs": num_pairs,
            "loss": float(loss_probe.detach().item()),
        }
        if not torch.isfinite(loss_probe.detach()):
            result["status"] = "nonfinite_loss"
            probe_results.append(result)
            continue

        backward_err = _backward_with_optional_anomaly(loss_probe, args, tag=f"probe={probe_name}")
        if backward_err is not None:
            result["status"] = "backward_exception"
            result["error"] = backward_err
            probe_results.append(result)
            continue

        bad_param, nan_cnt, inf_cnt = _first_nonfinite_grad(model)
        overview = _grad_overview(model)
        result["grad_nonfinite_param_cnt"] = int(overview.get("nonfinite_param_cnt", 0))
        result["grad_max_abs_finite"] = float(overview.get("max_abs_finite", 0.0))
        if bad_param:
            result["status"] = "nonfinite_grad"
            result["bad_param"] = bad_param
            result["nan_cnt"] = int(nan_cnt)
            result["inf_cnt"] = int(inf_cnt)
        else:
            result["status"] = "ok"
        probe_results.append(result)

    optimizer.zero_grad(set_to_none=True)
    return probe_results


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
    effective_pairs_used = 0
    dropped_by_second_filter = 0

    _trace(f"optimizer_step_start pref_batch={pref_batch} total_weight={total_weight:.6f} mb_pref={mb_pref}", args)
    _trace_event(
        args,
        "optimizer_step_start",
        {"pref_batch": int(pref_batch), "total_weight": float(total_weight), "mb_pref": int(mb_pref)},
    )

    for start in range(0, pref_batch, mb_pref):
        end = min(start + mb_pref, pref_batch)
        tp = pref_train_prompts[start:end]
        ch = pref_chosen[start:end]
        rj = pref_rejected[start:end]
        w = torch.tensor(pref_weights[start:end], device=device, dtype=torch.float32)
        w_raw_for_probe = w.detach().clone()

        probe_keep_mask: Optional[torch.Tensor] = None

        _trace(f"chunk[{start}:{end}] forward/chosen_logps", args)
        chosen_logps = base._compute_sequence_logps_batch(model, tokenizer, tp, ch, args.max_length, device)
        if args.nan_trace_log_tensor_stats:
            stats = _tensor_stats(chosen_logps, f"pref.chosen_logps[{start}:{end}]")
            _trace(_stats_line(stats), args)
            _trace_event(args, "tensor_stats", stats)
        bad_msg = _first_nonfinite_tensor(chosen_logps, f"pref.chosen_logps[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_before_gap({bad_msg})")

        _trace(f"chunk[{start}:{end}] forward/rejected_logps", args)
        rejected_logps = base._compute_sequence_logps_batch(model, tokenizer, tp, rj, args.max_length, device)
        if args.nan_trace_log_tensor_stats:
            stats = _tensor_stats(rejected_logps, f"pref.rejected_logps[{start}:{end}]")
            _trace(_stats_line(stats), args)
            _trace_event(args, "tensor_stats", stats)
        bad_msg = _first_nonfinite_tensor(rejected_logps, f"pref.rejected_logps[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_before_gap({bad_msg})")

        raw_gap = chosen_logps - rejected_logps
        if args.train_pref_second_filter_enabled:
            second_filter = _second_filter_pref_pairs(chosen_logps, rejected_logps, args)
            raw_gap = second_filter["raw_gap"]
            keep_mask = second_filter["keep_mask"]
            keep_cnt = int(second_filter["kept"])
            total_cnt = int(second_filter["total"])
            drop_cnt = total_cnt - keep_cnt
            dropped_by_second_filter += drop_cnt
            _trace(
                f"chunk[{start}:{end}] second_filter kept={keep_cnt}/{total_cnt} "
                f"drop_nonfinite={second_filter['drop_nonfinite']} "
                f"drop_chosen_floor={second_filter['drop_chosen_floor']} "
                f"drop_rejected_floor={second_filter['drop_rejected_floor']} "
                f"drop_raw_gap_cap={second_filter['drop_raw_gap_cap']}",
                args,
            )
            _trace_event(
                args,
                "second_filter",
                {
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                    "kept": keep_cnt,
                    "total": total_cnt,
                    "drop_nonfinite": int(second_filter["drop_nonfinite"]),
                    "drop_chosen_floor": int(second_filter["drop_chosen_floor"]),
                    "drop_rejected_floor": int(second_filter["drop_rejected_floor"]),
                    "drop_raw_gap_cap": int(second_filter["drop_raw_gap_cap"]),
                },
            )
            if keep_cnt < int(args.train_pref_second_filter_min_pairs_per_chunk):
                _trace(
                    f"chunk[{start}:{end}] skipped_by_second_filter "
                    f"(kept={keep_cnt} < min_pairs_per_chunk={args.train_pref_second_filter_min_pairs_per_chunk})",
                    args,
                )
                continue

            chosen_logps = chosen_logps[keep_mask]
            rejected_logps = rejected_logps[keep_mask]
            raw_gap = raw_gap[keep_mask]
            w = w[keep_mask]
            probe_keep_mask = keep_mask.detach().cpu()
            effective_pairs_used += keep_cnt
        else:
            effective_pairs_used += int(raw_gap.numel())

        preference_gap = raw_gap
        if args.online_gap_clip_abs > 0:
            preference_gap = preference_gap.clamp(-args.online_gap_clip_abs, args.online_gap_clip_abs)
        if args.nan_trace_log_tensor_stats:
            stats = _tensor_stats(preference_gap, f"pref.gap[{start}:{end}]")
            _trace(_stats_line(stats), args)
            _trace_event(args, "tensor_stats", stats)
        bad_msg = _first_nonfinite_tensor(preference_gap, f"pref.gap[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_in_gap({bad_msg})")

        pref_loss_vec = -F.logsigmoid(args.beta * preference_gap)
        if args.nan_trace_log_tensor_stats:
            stats = _tensor_stats(pref_loss_vec, f"pref.loss_vec[{start}:{end}]")
            _trace(_stats_line(stats), args)
            _trace_event(args, "tensor_stats", stats)
        bad_msg = _first_nonfinite_tensor(pref_loss_vec, f"pref.loss_vec[{start}:{end}]")
        if bad_msg is not None:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_detected_in_pref_loss({bad_msg})")

        if args.nan_trace_log_chain_grads:
            chosen_logps.retain_grad()
            rejected_logps.retain_grad()
            preference_gap.retain_grad()
            pref_loss_vec.retain_grad()

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
        backward_err = _backward_with_optional_anomaly(loss_chunk, args, tag=f"chunk=[{start},{end})")
        if backward_err is not None:
            _trace(backward_err, args)
            _trace_event(
                args,
                "backward_exception",
                {"chunk_start": int(start), "chunk_end": int(end), "error": backward_err},
            )
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"backward_exception(branch=pref,chunk=[{start},{end}))")

        if args.nan_trace_log_chain_grads:
            _trace_grad_of_tensor(pref_loss_vec, f"pref.loss_vec[{start}:{end}]", args)
            _trace_grad_of_tensor(preference_gap, f"pref.gap[{start}:{end}]", args)
            _trace_grad_of_tensor(chosen_logps, f"pref.chosen_logps[{start}:{end}]", args)
            _trace_grad_of_tensor(rejected_logps, f"pref.rejected_logps[{start}:{end}]", args)

        if args.nan_trace_log_grad_overview:
            overview = _grad_overview(model)
            _trace(
                "grad_overview "
                f"chunk=[{start},{end}) grad_param_cnt={overview.get('grad_param_cnt', 0)} "
                f"nonfinite_param_cnt={overview.get('nonfinite_param_cnt', 0)} "
                f"nan_cnt={overview.get('nan_cnt', 0)} inf_cnt={overview.get('inf_cnt', 0)} "
                f"max_abs_finite={overview.get('max_abs_finite', 0.0):.6f}",
                args,
            )
            _trace_event(args, "grad_overview", {"chunk_start": int(start), "chunk_end": int(end), **overview})

        bad_param, nan_cnt, inf_cnt = _first_nonfinite_grad(model)
        if bad_param:
            bad_list = _collect_nonfinite_grad_params(model, max_items=args.nan_trace_max_bad_params)
            _trace(
                f"nonfinite_grad_params_detected count={len(bad_list)} "
                f"(showing up to {args.nan_trace_max_bad_params})",
                args,
            )
            for item in bad_list:
                _trace(
                    "bad_grad "
                    f"name={item['name']} shape={item['shape']} numel={item['numel']} "
                    f"nan={item['nan_cnt']} inf={item['inf_cnt']} finite={item['finite_cnt']} "
                    f"max_abs_finite={item['max_abs_finite']:.6f}",
                    args,
                )
            _trace_event(
                args,
                "bad_grad_list",
                {"chunk_start": int(start), "chunk_end": int(end), "items": bad_list},
            )
            if args.nan_trace_backward_probe_on_failure:
                _trace(f"chunk[{start}:{end}] running_backward_probes_on_failure", args)
                probe_results = _run_chunk_backward_probes(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                    tp=tp,
                    ch=ch,
                    rj=rj,
                    w=w_raw_for_probe,
                    total_weight=total_weight,
                    keep_mask=probe_keep_mask,
                    start=start,
                    end=end,
                )
                for pr in probe_results:
                    _trace(
                        "probe_result "
                        f"chunk=[{pr.get('chunk_start')},{pr.get('chunk_end')}) "
                        f"name={pr.get('probe')} status={pr.get('status')} "
                        f"loss={pr.get('loss', 0.0):.6f} "
                        f"grad_nonfinite_param_cnt={pr.get('grad_nonfinite_param_cnt', 0)} "
                        f"grad_max_abs_finite={pr.get('grad_max_abs_finite', 0.0):.6f} "
                        f"bad_param={pr.get('bad_param', '')}",
                        args,
                    )
                _trace_event(
                    args,
                    "backward_probe_results",
                    {"chunk_start": int(start), "chunk_end": int(end), "results": probe_results},
                )
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                "nonfinite_grad_after_backward("
                f"branch=pref,chunk=[{start},{end}),param={bad_param},nan={nan_cnt},inf={inf_cnt})"
            )

        pref_loss_weighted_sum += (pref_loss_vec * w).sum().item()
        gap_weighted_sum += (preference_gap * w).sum().item()
        pref_weight_sum += w.sum().item()

    if effective_pairs_used <= 0 or pref_weight_sum <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats(
            f"all_pairs_filtered_in_step(dropped={dropped_by_second_filter},total={pref_batch})"
        )

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
        pref_pairs_used=effective_pairs_used,
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
    parser.add_argument(
        "--nan_trace_log_tensor_stats",
        type=str2bool,
        default=True,
        help="Print and record tensor stats for chosen/rejected/gap/loss_vec on every chunk.",
    )
    parser.add_argument(
        "--nan_trace_log_grad_overview",
        type=str2bool,
        default=True,
        help="Print and record grad aggregate stats after each backward.",
    )
    parser.add_argument(
        "--nan_trace_max_bad_params",
        type=int,
        default=20,
        help="Max number of nonfinite-gradient parameters to print when detected.",
    )
    parser.add_argument(
        "--nan_trace_write_jsonl",
        type=str2bool,
        default=True,
        help="Write structured debug records to output_dir/nan_trace_debug.jsonl.",
    )
    parser.add_argument(
        "--nan_trace_log_chain_grads",
        type=str2bool,
        default=True,
        help="Log grad stats for pref_loss_vec/gap/chosen_logps/rejected_logps after each backward.",
    )
    parser.add_argument(
        "--nan_trace_detect_anomaly",
        type=str2bool,
        default=False,
        help="Enable torch autograd anomaly detection to pinpoint backward op failures.",
    )
    parser.add_argument(
        "--nan_trace_backward_probe_on_failure",
        type=str2bool,
        default=True,
        help="When nonfinite grad appears, rerun chosen/rejected/gap/pref probe backward passes on same chunk.",
    )
    parser.add_argument(
        "--train_pref_second_filter_enabled",
        type=str2bool,
        default=True,
        help="Enable per-chunk second-stage pair filtering before backward.",
    )
    parser.add_argument(
        "--train_pref_second_filter_min_chosen_logp",
        type=float,
        default=-3.0,
        help="Second filter: keep only pairs with chosen_logp >= this threshold.",
    )
    parser.add_argument(
        "--train_pref_second_filter_min_rejected_logp",
        type=float,
        default=-4.5,
        help="Second filter: keep only pairs with rejected_logp >= this threshold.",
    )
    parser.add_argument(
        "--train_pref_second_filter_max_abs_raw_gap",
        type=float,
        default=2.0,
        help="Second filter: keep only pairs with abs(raw_gap) <= this threshold before gap clip.",
    )
    parser.add_argument(
        "--train_pref_second_filter_min_pairs_per_chunk",
        type=int,
        default=1,
        help="If kept pairs in chunk is below this number, skip this chunk.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    args._nan_trace_jsonl_path = str(Path(args.output_dir).expanduser().resolve() / "nan_trace_debug.jsonl")

    # Keep semantics aligned with base main(): 0 means "no limit".
    if args.max_source_samples == 0:
        args.max_source_samples = None
    if args.online_steps == 0:
        args.online_steps = None

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
        f"gradient_checkpointing={args.gradient_checkpointing} "
        f"second_filter={args.train_pref_second_filter_enabled} "
        f"chosen_floor={args.train_pref_second_filter_min_chosen_logp} "
        f"rejected_floor={args.train_pref_second_filter_min_rejected_logp} "
        f"raw_gap_cap={args.train_pref_second_filter_max_abs_raw_gap} "
        f"jsonl={args._nan_trace_jsonl_path}",
        flush=True,
    )

    # Monkey-patch LoRA wrapping to emit initialization stats.
    base.wrap_model_with_lora = _wrap_model_with_lora_and_log

    # Monkey-patch the optimizer step with detailed trace version.
    base._online_run_preference_optimizer_step = _online_run_pref_only_nan_trace_step

    base.run_online_preference_training(args)


if __name__ == "__main__":
    main()
