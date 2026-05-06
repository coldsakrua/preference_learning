#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from privileged_hidden_opsd.opsd_non_training import (
    PrivilegedObjective,
    build_gt_privileged_examples,
    build_mixed_privileged_examples,
    build_privileged_distill_batch,
    build_rollout_record,
    build_rollout_trajectories_with_hidden,
    compute_sequence_logps_batch_local,
    extract_completion_logits,
    load_source_iter,
    opsd_generalized_jsd_row_loss_and_gap,
)
from privileged_hidden_opsd.opsd_local_utils import (  # noqa: E402
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    DEFAULT_SYSTEM_PROMPT,
    DapoSample,
    _compute_lora_param_health,
    apply_qwen_chat_template,
    build_parser as build_cli_parser,
    build_prompt_pool,
    compute_correct_trajectory_weights,
    choose_system_prompt,
    ensure_input_require_grads_for_checkpointing,
    online_rollout_completions_flat_hf as _online_rollout_completions_flat_hf,
    online_rollout_completions_flat_vllm as _online_rollout_completions_flat_vllm,
    set_seed,
    split_rollout_candidates_for_training,
    str2bool,
    unwrap_model_for_save,
    wrap_model_with_lora,
)


@dataclass
class PrivilegedStepStats:
    total_loss: float
    mle_loss: float
    privileged_loss: float
    gt_privileged_loss: float
    mean_privileged_advantage: float
    mean_gt_privileged_advantage: float
    privileged_pairs_used: int
    gt_privileged_pairs_used: int
    mle_samples_used: int
    grad_norm: float
    update_applied: bool
    skip_reason: str
    lora_health: Dict[str, float]


def _zero_lora_health() -> Dict[str, float]:
    return {
        "lora_mean_abs": 0.0,
        "lora_max_abs": 0.0,
        "lora_nan_ratio": 0.0,
        "lora_inf_ratio": 0.0,
    }


def _build_zero_stats(reason: str, grad_norm: float = 0.0) -> PrivilegedStepStats:
    return PrivilegedStepStats(
        total_loss=0.0,
        mle_loss=0.0,
        privileged_loss=0.0,
        gt_privileged_loss=0.0,
        mean_privileged_advantage=0.0,
        mean_gt_privileged_advantage=0.0,
        privileged_pairs_used=0,
        gt_privileged_pairs_used=0,
        mle_samples_used=0,
        grad_norm=grad_norm,
        update_applied=False,
        skip_reason=reason,
        lora_health=_zero_lora_health(),
    )


def _teacher_adapter_context(model: object, args: argparse.Namespace):
    """OPSD fixed-teacher mode: disable LoRA adapters during teacher forward."""
    if not bool(getattr(args, "fixed_teacher", False)):
        return nullcontext()
    if not bool(getattr(args, "use_lora", False)):
        return nullcontext()
    unwrapped = unwrap_model_for_save(model)
    disable_adapter = getattr(unwrapped, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    print(
        "[privileged_hidden_opsd] fixed_teacher=true but model has no disable_adapter(); "
        "teacher will use current weights.",
        flush=True,
    )
    return nullcontext()


def _first_nonfinite_grad_names(model: object, limit: int = 3) -> List[str]:
    out: List[str] = []
    named_params = getattr(model, "named_parameters", None)
    if not callable(named_params):
        return out
    for name, param in model.named_parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if not torch.isfinite(grad.detach()).all():
            out.append(str(name))
            if len(out) >= limit:
                break
    return out


def _sanitize_nonfinite_grads_(
    trainable: Sequence[torch.nn.Parameter],
    *,
    element_clip_abs: float,
) -> tuple[int, int]:
    touched_tensors = 0
    nonfinite_values = 0
    for param in trainable:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if grad.is_sparse:
            grad = grad.coalesce()
            param.grad = grad
            values = grad._values()
            finite_mask = torch.isfinite(values)
            if not bool(finite_mask.all()):
                nonfinite_values += int((~finite_mask).sum().item())
                touched_tensors += 1
                values.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if element_clip_abs > 0:
                values.clamp_(min=-element_clip_abs, max=element_clip_abs)
            continue
        g = grad.detach()
        finite_mask = torch.isfinite(g)
        if not bool(finite_mask.all()):
            nonfinite_values += int((~finite_mask).sum().item())
            touched_tensors += 1
            g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if element_clip_abs > 0:
            g.clamp_(min=-element_clip_abs, max=element_clip_abs)
    return touched_tensors, nonfinite_values


def _compute_grad_norm_and_clip_(
    trainable: Sequence[torch.nn.Parameter],
    *,
    max_grad_norm: float,
    device: torch.device,
) -> float:
    total_sq = torch.zeros((), dtype=torch.float32, device=device)
    for param in trainable:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if grad.is_sparse:
            grad = grad.coalesce()
            param.grad = grad
            vals = grad._values().float()
            total_sq = total_sq + (vals * vals).sum()
        else:
            g = grad.detach().float()
            total_sq = total_sq + (g * g).sum()
    total_norm = torch.sqrt(total_sq)
    grad_norm = float(total_norm.item())
    if max_grad_norm > 0 and math.isfinite(grad_norm) and grad_norm > max_grad_norm:
        scale = float(max_grad_norm) / (grad_norm + 1e-12)
        for param in trainable:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce()
                param.grad = grad
                grad._values().mul_(scale)
            else:
                grad.mul_(scale)
    return grad_norm


def _run_optimizer_step(
    *,
    model: object,
    tokenizer: object,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
    mle_train_prompts: List[str],
    mle_completions: List[str],
    mle_weights: List[float],
    priv_train_prompts: List[str],
    priv_prompts: List[str],
    priv_wrong: List[str],
    priv_weights: List[float],
    gt_priv_train_prompts: List[str],
    gt_priv_prompts: List[str],
    gt_priv_wrong: List[str],
    gt_priv_weights: List[float],
) -> PrivilegedStepStats:
    original_total_weight = float(sum(mle_weights) + sum(priv_weights) + sum(gt_priv_weights))
    if original_total_weight <= 0:
        return _build_zero_stats("zero_total_weight")

    mb = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(
        len(mle_train_prompts),
        len(priv_train_prompts),
        len(gt_priv_train_prompts),
        1,
    )

    optimizer.zero_grad(set_to_none=True)
    mle_loss_weighted_sum = 0.0
    priv_loss_weighted_sum = 0.0
    gt_priv_loss_weighted_sum = 0.0
    priv_adv_weighted_sum = 0.0
    gt_priv_adv_weighted_sum = 0.0
    priv_weight_sum_used = 0.0
    gt_priv_weight_sum_used = 0.0
    mle_weight_sum_used = 0.0
    mle_samples_used = 0
    priv_pairs_used = 0
    gt_priv_pairs_used = 0

    def _check_and_backward(
        loss_chunk: torch.Tensor,
        loss_chunk_val: float,
        skip_prefix: str,
        start: int,
        end: int,
    ) -> Optional[PrivilegedStepStats]:
        if not loss_chunk.requires_grad:
            raise RuntimeError(
                "Loss has no grad_fn. If use_lora=true with gradient_checkpointing=true, "
                "ensure input grads are enabled for checkpointing."
            )
        if args.online_skip_nonfinite_loss and not torch.isfinite(loss_chunk.detach()):
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_{skip_prefix}_loss_chunk(start={start},end={end})")
        if args.online_loss_value_cap > 0 and abs(loss_chunk_val) > args.online_loss_value_cap:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                f"{skip_prefix}_loss_chunk_too_large(value={loss_chunk_val:.4f},cap={args.online_loss_value_cap:.4f})"
            )
        loss_chunk.backward()
        bad_grad_names = _first_nonfinite_grad_names(model)
        if bad_grad_names:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                f"nonfinite_{skip_prefix}_grad_after_backward(start={start},end={end},"
                f"params={','.join(bad_grad_names)})"
            )
        return None

    def _run_privileged_branch(
        *,
        branch: str,
        train_prompts: List[str],
        privileged_prompts: List[str],
        wrong_completions: List[str],
        weights: List[float],
    ) -> Optional[PrivilegedStepStats]:
        nonlocal priv_loss_weighted_sum
        nonlocal gt_priv_loss_weighted_sum
        nonlocal priv_adv_weighted_sum
        nonlocal gt_priv_adv_weighted_sum
        nonlocal priv_weight_sum_used
        nonlocal gt_priv_weight_sum_used
        nonlocal priv_pairs_used
        nonlocal gt_priv_pairs_used

        branch_batch = len(train_prompts)
        if branch_batch <= 0:
            return None
        for start in range(0, branch_batch, mb):
            end = min(start + mb, branch_batch)
            tp = train_prompts[start:end]
            pp = privileged_prompts[start:end]
            wc = wrong_completions[start:end]
            distill_batch = build_privileged_distill_batch(
                tokenizer=tokenizer,
                student_prompts=tp,
                teacher_prompts=pp,
                completions=wc,
                weights=weights[start:end],
                args=args,
                device=device,
            )
            if distill_batch is None:
                print(
                    f"[privileged_hidden_opsd] {branch} distill chunk=[{start},{end}) kept=0/{len(tp)} "
                    "after token-length filtering",
                    flush=True,
                )
                continue
            if distill_batch.kept_count < distill_batch.total_count:
                print(
                    f"[privileged_hidden_opsd] {branch} distill chunk=[{start},{end}) "
                    f"kept={distill_batch.kept_count}/{distill_batch.total_count} after token-length filtering",
                    flush=True,
                )

            was_training = bool(getattr(model, "training", False))
            model.eval()
            try:
                with torch.no_grad(), _teacher_adapter_context(model, args):
                    teacher_outputs = model(
                        input_ids=distill_batch.teacher_input_ids,
                        attention_mask=distill_batch.teacher_attention_mask,
                        use_cache=False,
                    )
                    teacher_logits = extract_completion_logits(
                        teacher_outputs.logits,
                        distill_batch.teacher_prompt_lens,
                        distill_batch.token_mask,
                    ).detach()
            finally:
                if was_training:
                    model.train()

            student_outputs = model(
                input_ids=distill_batch.student_input_ids,
                attention_mask=distill_batch.student_attention_mask,
                use_cache=False,
            )
            student_logits = extract_completion_logits(
                student_outputs.logits,
                distill_batch.student_prompt_lens,
                distill_batch.token_mask,
            )
            opsd_outputs = opsd_generalized_jsd_row_loss_and_gap(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                target_ids=distill_batch.target_ids,
                token_mask=distill_batch.token_mask,
                args=args,
            )
            row_loss = opsd_outputs[0]
            raw_gap = opsd_outputs[1]

            keep_mask = torch.isfinite(row_loss) & torch.isfinite(raw_gap)
            kept = int(keep_mask.sum().item())
            total = int(keep_mask.numel())
            if kept <= 0:
                print(
                    f"[privileged_hidden_opsd] {branch} prefilter chunk=[{start},{end}) kept=0/{total}",
                    flush=True,
                )
                continue
            if kept < total:
                print(
                    f"[privileged_hidden_opsd] {branch} prefilter chunk=[{start},{end}) kept={kept}/{total}",
                    flush=True,
                )

            keep_mask = keep_mask.to(device=device)
            w_kept = distill_batch.weights[keep_mask]
            loss_vec = row_loss[keep_mask]
            gap_vec = raw_gap.to(device=device)[keep_mask]
            loss_chunk = (loss_vec * w_kept).sum() / original_total_weight
            loss_chunk_val = float(loss_chunk.detach().item())
            failed = _check_and_backward(
                loss_chunk=loss_chunk,
                loss_chunk_val=loss_chunk_val,
                skip_prefix=branch,
                start=start,
                end=end,
            )
            if failed is not None:
                return failed

            loss_sum = float((loss_vec.detach() * w_kept).sum().item())
            adv_sum = float((gap_vec.detach() * w_kept).sum().item())
            w_sum = float(w_kept.sum().item())
            if branch == "privileged":
                priv_loss_weighted_sum += loss_sum
                priv_adv_weighted_sum += adv_sum
                priv_weight_sum_used += w_sum
                priv_pairs_used += kept
            else:
                gt_priv_loss_weighted_sum += loss_sum
                gt_priv_adv_weighted_sum += adv_sum
                gt_priv_weight_sum_used += w_sum
                gt_priv_pairs_used += kept
        return None

    failed = _run_privileged_branch(
        branch="privileged",
        train_prompts=priv_train_prompts,
        privileged_prompts=priv_prompts,
        wrong_completions=priv_wrong,
        weights=priv_weights,
    )
    if failed is not None:
        return failed

    failed = _run_privileged_branch(
        branch="gt_privileged",
        train_prompts=gt_priv_train_prompts,
        privileged_prompts=gt_priv_prompts,
        wrong_completions=gt_priv_wrong,
        weights=gt_priv_weights,
    )
    if failed is not None:
        return failed

    if mle_train_prompts:
        for start in range(0, len(mle_train_prompts), mb):
            end = min(start + mb, len(mle_train_prompts))
            tp = mle_train_prompts[start:end]
            cp = mle_completions[start:end]
            w = torch.tensor(mle_weights[start:end], device=device, dtype=torch.float32)
            logps = compute_sequence_logps_batch_local(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=tp,
                completion_texts=cp,
                max_length=args.max_length,
                device=device,
            )
            keep_mask = torch.isfinite(logps)
            if int(keep_mask.sum().item()) <= 0:
                continue
            logps = logps[keep_mask]
            w = w[keep_mask.to(device=w.device)]
            mle_loss_vec = -logps
            loss_chunk = (mle_loss_vec * w).sum() / original_total_weight
            loss_chunk_val = float(loss_chunk.detach().item())
            failed = _check_and_backward(
                loss_chunk=loss_chunk,
                loss_chunk_val=loss_chunk_val,
                skip_prefix="mle",
                start=start,
                end=end,
            )
            if failed is not None:
                return failed
            mle_loss_weighted_sum += float((mle_loss_vec.detach() * w).sum().item())
            mle_weight_sum_used += float(w.sum().item())
            mle_samples_used += int(w.numel())

    used_weight = mle_weight_sum_used + priv_weight_sum_used + gt_priv_weight_sum_used
    if used_weight <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("all_train_samples_filtered_before_autograd")

    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_rescale = float(original_total_weight) / float(used_weight)
    if not math.isfinite(grad_rescale) or grad_rescale <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats(
            f"invalid_grad_rescale(original_total_weight={original_total_weight:.6f},used_weight={used_weight:.6f})"
        )
    # Keep backward micro-batch accumulation memory behavior, but align final gradient scale
    # with the filtered effective weight used by metrics.
    if abs(grad_rescale - 1.0) > 1e-12:
        for param in trainable:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            grad.mul_(grad_rescale)

    if bool(getattr(args, "online_sanitize_gradients", True)):
        touched, nonfinite_count = _sanitize_nonfinite_grads_(
            trainable,
            element_clip_abs=float(getattr(args, "online_grad_element_clip_abs", 0.0)),
        )
        if nonfinite_count > 0:
            print(
                "[privileged_hidden_opsd] sanitized nonfinite gradients "
                f"tensors={touched} values={nonfinite_count}",
                flush=True,
            )
    grad_norm = _compute_grad_norm_and_clip_(
        trainable,
        max_grad_norm=float(args.max_grad_norm),
        device=device,
    )
    if args.online_skip_nonfinite_loss and not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("nonfinite_grad_norm", grad_norm=grad_norm)
    if args.online_hard_grad_norm_cap > 0 and grad_norm > args.online_hard_grad_norm_cap:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats(
            f"grad_norm_too_large(value={grad_norm:.4f},cap={args.online_hard_grad_norm_cap:.4f})",
            grad_norm=grad_norm,
        )
    optimizer.step()

    lora_health = _compute_lora_param_health(model)
    if args.online_abort_on_lora_nan and lora_health["lora_nan_ratio"] > 0:
        raise RuntimeError(f"Detected NaN in LoRA params: lora_nan_ratio={lora_health['lora_nan_ratio']:.6f}")

    return PrivilegedStepStats(
        total_loss=(mle_loss_weighted_sum + priv_loss_weighted_sum + gt_priv_loss_weighted_sum) / used_weight,
        mle_loss=mle_loss_weighted_sum / mle_weight_sum_used if mle_weight_sum_used > 0 else 0.0,
        privileged_loss=priv_loss_weighted_sum / priv_weight_sum_used if priv_weight_sum_used > 0 else 0.0,
        gt_privileged_loss=(
            gt_priv_loss_weighted_sum / gt_priv_weight_sum_used if gt_priv_weight_sum_used > 0 else 0.0
        ),
        mean_privileged_advantage=priv_adv_weighted_sum / priv_weight_sum_used if priv_weight_sum_used > 0 else 0.0,
        mean_gt_privileged_advantage=(
            gt_priv_adv_weighted_sum / gt_priv_weight_sum_used if gt_priv_weight_sum_used > 0 else 0.0
        ),
        privileged_pairs_used=priv_pairs_used,
        gt_privileged_pairs_used=gt_priv_pairs_used,
        mle_samples_used=mle_samples_used,
        grad_norm=grad_norm,
        update_applied=True,
        skip_reason="",
        lora_health=lora_health,
    )


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

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if args.torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported --torch_dtype: {args.torch_dtype}")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype_map[args.torch_dtype]}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.use_lora:
        if args.online_rollout_backend == "vllm" and args.lora_r > args.vllm_max_lora_rank:
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
        print(f"[privileged_hidden_opsd] enabled DataParallel over {torch.cuda.device_count()} GPUs", flush=True)
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
    layout, source_iter = load_source_iter(args)
    rollout_user_suffix = str(args.user_content_suffix or "")
    if args.auto_math_hf_user_suffix and layout == "math_hf" and not rollout_user_suffix.strip():
        rollout_user_suffix = DEFAULT_MATH_HF_USER_CONTENT_SUFFIX

    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260412)
    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"

    print(
        f"[privileged_hidden_opsd] dataset_layout={layout} rollout_backend={args.online_rollout_backend} "
        f"rollout_batch_size={args.rollout_batch_size} rollout_n={args.rollout_n} "
        f"objectives_per_update={args.online_pairs_per_step} online_steps={total_steps_str} "
        f"lambda_mle={args.lambda_mle} lambda_priv={args.lambda_priv} lambda_gt={args.lambda_gt} "
        f"hidden_layer_offset={args.hidden_layer_offset} "
        f"jsd_beta={args.privileged_jsd_beta if args.privileged_jsd_beta >= 0 else args.beta} "
        f"pointwise_kl_clip={args.privileged_pointwise_kl_clip} fixed_teacher={args.fixed_teacher}",
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
            "lambda_priv": float(args.lambda_priv),
            "lambda_gt": float(args.lambda_gt),
            "rollout_n": int(args.rollout_n),
            "hidden_layer_offset": int(args.hidden_layer_offset),
            "privileged_jsd_beta": float(args.privileged_jsd_beta if args.privileged_jsd_beta >= 0 else args.beta),
            "privileged_distill_temperature": float(args.privileged_distill_temperature),
            "privileged_pointwise_kl_clip": float(args.privileged_pointwise_kl_clip),
            "fixed_teacher": bool(args.fixed_teacher),
            "mle_include_mixed_correct": bool(args.mle_include_mixed_correct),
            "online_sanitize_gradients": bool(args.online_sanitize_gradients),
            "online_grad_element_clip_abs": float(args.online_grad_element_clip_abs),
        },
    )

    if args.online_rollout_backend == "vllm" and device.type != "cuda":
        raise RuntimeError("online_rollout_backend=vllm requires CUDA.")

    updates = 0
    rollout_steps = 0
    scanned = 0
    kept_mle_samples = 0
    kept_priv_pairs = 0
    kept_gt_priv_pairs = 0
    logged_mixed = 0
    logged_mixed_mle_only = 0
    logged_all_correct = 0
    logged_all_wrong = 0
    skipped_all_wrong = 0
    skipped_after_filter = 0
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
            prompt_user_effective = [s.prompt + rollout_user_suffix for s in buffer]
            prompt_texts = [
                apply_qwen_chat_template(
                    tokenizer,
                    user_prompt,
                    enable_thinking=args.enable_thinking,
                    system_prompt=sp,
                )
                for user_prompt, sp in zip(prompt_user_effective, system_prompts)
            ]

            if args.online_rollout_backend == "vllm":
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

            rollout_objectives: List[PrivilegedObjective] = []
            sampled_correct_total = 0
            sampled_total = 0
            mixed_in_rollout = 0
            mixed_mle_only_in_rollout = 0
            all_correct_in_rollout = 0
            all_wrong_in_rollout = 0
            skipped_all_wrong_in_rollout = 0
            skipped_after_filter_in_rollout = 0
            priv_examples_in_rollout = 0
            gt_priv_examples_in_rollout = 0

            for idx, sample_obj in enumerate(buffer):
                start = idx * args.rollout_n
                end = start + args.rollout_n
                candidates = completion_flat[start:end]
                if len(candidates) != args.rollout_n:
                    raise RuntimeError(
                        f"Rollout candidate count mismatch at sample {idx}: expected {args.rollout_n}, got {len(candidates)}"
                    )
                split = split_rollout_candidates_for_training(candidates, sample_obj.ground_truth)
                n_total = len(candidates)
                n_correct = int(sum(1 for x in split.responses_correct if x))
                sampled_correct_total += n_correct
                sampled_total += n_total

                objective: Optional[PrivilegedObjective] = None
                trajectories = build_rollout_trajectories_with_hidden(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    train_prompt=prompt_texts[idx],
                    candidates=candidates,
                    split=split,
                    args=args,
                )
                correct_trajs = [trajectories[i] for i in split.correct_kept_indices]
                wrong_trajs = [trajectories[i] for i in split.wrong_kept_indices]

                if n_correct == n_total and correct_trajs:
                    uniform_weight = 1.0 / float(len(correct_trajs))
                    correct_weights = [uniform_weight for _ in correct_trajs]
                    objective = PrivilegedObjective(
                        sample_id=sample_obj.sample_id,
                        ground_truth=sample_obj.ground_truth,
                        train_prompt=prompt_texts[idx],
                        objective_type="all_correct",
                        correct=correct_trajs,
                        wrong=[],
                        correct_weights=correct_weights,
                        privileged_wrong_examples=[],
                    )
                    rollout_objectives.append(objective)
                    all_correct_in_rollout += 1
                    logged_all_correct += 1
                elif n_correct > 0 and correct_trajs and wrong_trajs:
                    correct_weights = compute_correct_trajectory_weights(
                        correct_trajs,
                        mode=str(args.positive_weight_mode),
                        tau=float(args.positive_weight_tau),
                    )
                    privileged_examples = build_mixed_privileged_examples(
                        tokenizer=tokenizer,
                        prompt_user_effective=prompt_user_effective[idx],
                        system_prompt=system_prompts[idx],
                        train_prompt=prompt_texts[idx],
                        correct_trajs=correct_trajs,
                        wrong_trajs=wrong_trajs,
                        args=args,
                    )
                    objective = PrivilegedObjective(
                        sample_id=sample_obj.sample_id,
                        ground_truth=sample_obj.ground_truth,
                        train_prompt=prompt_texts[idx],
                        objective_type="mixed_hidden_opsd",
                        correct=correct_trajs,
                        wrong=wrong_trajs,
                        correct_weights=correct_weights,
                        privileged_wrong_examples=privileged_examples,
                    )
                    if privileged_examples:
                        rollout_objectives.append(objective)
                        mixed_in_rollout += 1
                        logged_mixed += 1
                        priv_examples_in_rollout += len(privileged_examples)
                    elif args.lambda_mle > 0 and args.mle_include_mixed_correct:
                        objective.objective_type = "mixed_mle_only"
                        rollout_objectives.append(objective)
                        mixed_mle_only_in_rollout += 1
                        logged_mixed_mle_only += 1
                    else:
                        skipped_after_filter += 1
                        skipped_after_filter_in_rollout += 1
                elif n_correct <= 0 and wrong_trajs:
                    gt_examples = build_gt_privileged_examples(
                        tokenizer=tokenizer,
                        sample=sample_obj,
                        prompt_user_effective=prompt_user_effective[idx],
                        system_prompt=system_prompts[idx],
                        train_prompt=prompt_texts[idx],
                        wrong_trajs=wrong_trajs,
                        args=args,
                    )
                    if gt_examples:
                        objective = PrivilegedObjective(
                            sample_id=sample_obj.sample_id,
                            ground_truth=sample_obj.ground_truth,
                            train_prompt=prompt_texts[idx],
                            objective_type="all_wrong_gt_privileged",
                            correct=[],
                            wrong=wrong_trajs,
                            correct_weights=[],
                            privileged_wrong_examples=gt_examples,
                        )
                        rollout_objectives.append(objective)
                        all_wrong_in_rollout += 1
                        logged_all_wrong += 1
                        gt_priv_examples_in_rollout += len(gt_examples)
                    else:
                        skipped_all_wrong += 1
                        skipped_all_wrong_in_rollout += 1
                else:
                    skipped_after_filter += 1
                    skipped_after_filter_in_rollout += 1

                rollout_log.write(
                    json.dumps(
                        build_rollout_record(
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

            sampled_correct_rate = sampled_correct_total / sampled_total if sampled_total > 0 else 0.0
            print(
                f"[privileged_hidden_opsd] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"mixed={mixed_in_rollout} mixed_mle_only={mixed_mle_only_in_rollout} "
                f"all_correct={all_correct_in_rollout} all_wrong_gt={all_wrong_in_rollout} "
                f"skipped_all_wrong={skipped_all_wrong_in_rollout} skipped_after_filter={skipped_after_filter_in_rollout} "
                f"sampled_correct_rate={sampled_correct_rate:.4f} objectives={len(rollout_objectives)} "
                f"priv_examples={priv_examples_in_rollout} gt_priv_examples={gt_priv_examples_in_rollout}",
                flush=True,
            )
            write_metric(
                "rollout_summary",
                {
                    "rollout_step": int(rollout_steps),
                    "scanned": int(scanned),
                    "mixed": int(mixed_in_rollout),
                    "mixed_mle_only": int(mixed_mle_only_in_rollout),
                    "all_correct": int(all_correct_in_rollout),
                    "all_wrong_gt": int(all_wrong_in_rollout),
                    "skipped_all_wrong": int(skipped_all_wrong_in_rollout),
                    "skipped_after_filter": int(skipped_after_filter_in_rollout),
                    "sampled_correct_rate": float(sampled_correct_rate),
                    "objectives": int(len(rollout_objectives)),
                    "privileged_examples": int(priv_examples_in_rollout),
                    "gt_privileged_examples": int(gt_priv_examples_in_rollout),
                },
            )

            updates_in_rollout = 0
            for chunk_start in range(0, len(rollout_objectives), args.online_pairs_per_step):
                chunk = rollout_objectives[chunk_start : chunk_start + args.online_pairs_per_step]
                if not chunk:
                    continue

                mle_train_prompts: List[str] = []
                mle_completions: List[str] = []
                mle_weights: List[float] = []
                priv_train_prompts: List[str] = []
                priv_prompts: List[str] = []
                priv_wrong: List[str] = []
                priv_weights: List[float] = []
                gt_priv_train_prompts: List[str] = []
                gt_priv_prompts: List[str] = []
                gt_priv_wrong: List[str] = []
                gt_priv_weights: List[float] = []

                for obj in chunk:
                    if args.lambda_mle > 0 and obj.correct:
                        if obj.correct_weights and len(obj.correct_weights) == len(obj.correct):
                            current_correct_weights = obj.correct_weights
                        else:
                            uniform_weight = 1.0 / float(len(obj.correct))
                            current_correct_weights = [uniform_weight for _ in obj.correct]
                        for traj, traj_weight in zip(obj.correct, current_correct_weights):
                            mle_train_prompts.append(obj.train_prompt)
                            mle_completions.append(traj.response_text)
                            mle_weights.append(float(args.lambda_mle) * float(traj_weight))
                    for ex in obj.privileged_wrong_examples:
                        if ex.source_type == "gt_rationale":
                            gt_priv_train_prompts.append(ex.train_prompt)
                            gt_priv_prompts.append(ex.privileged_prompt)
                            gt_priv_wrong.append(ex.wrong_response)
                            gt_priv_weights.append(float(ex.weight))
                        else:
                            priv_train_prompts.append(ex.train_prompt)
                            priv_prompts.append(ex.privileged_prompt)
                            priv_wrong.append(ex.wrong_response)
                            priv_weights.append(float(ex.weight))

                if not mle_train_prompts and not priv_train_prompts and not gt_priv_train_prompts:
                    continue
                stats = _run_optimizer_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                    mle_train_prompts=mle_train_prompts,
                    mle_completions=mle_completions,
                    mle_weights=mle_weights,
                    priv_train_prompts=priv_train_prompts,
                    priv_prompts=priv_prompts,
                    priv_wrong=priv_wrong,
                    priv_weights=priv_weights,
                    gt_priv_train_prompts=gt_priv_train_prompts,
                    gt_priv_prompts=gt_priv_prompts,
                    gt_priv_wrong=gt_priv_wrong,
                    gt_priv_weights=gt_priv_weights,
                )
                if not stats.update_applied:
                    print(
                        f"[privileged_hidden_opsd] rollout_step={rollout_steps}/{total_steps_str} "
                        f"skip optimizer update reason={stats.skip_reason} grad_norm={stats.grad_norm:.4f}",
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
                kept_mle_samples += stats.mle_samples_used
                kept_priv_pairs += stats.privileged_pairs_used
                kept_gt_priv_pairs += stats.gt_privileged_pairs_used
                print(
                    f"[privileged_hidden_opsd] rollout_step={rollout_steps}/{total_steps_str} optimizer_step={updates} "
                    f"mle_loss={stats.mle_loss:.6f} priv_loss={stats.privileged_loss:.6f} "
                    f"gt_priv_loss={stats.gt_privileged_loss:.6f} "
                    f"priv_adv={stats.mean_privileged_advantage:.6f} "
                    f"gt_priv_adv={stats.mean_gt_privileged_advantage:.6f} "
                    f"total_loss={stats.total_loss:.6f} grad_norm={stats.grad_norm:.6f} "
                    f"priv_pairs={stats.privileged_pairs_used} gt_priv_pairs={stats.gt_privileged_pairs_used} "
                    f"mle_samples={stats.mle_samples_used}",
                    flush=True,
                )
                write_metric(
                    "optimizer_step",
                    {
                        "rollout_step": int(rollout_steps),
                        "optimizer_step": int(updates),
                        "mle_loss": float(stats.mle_loss),
                        "privileged_loss": float(stats.privileged_loss),
                        "gt_privileged_loss": float(stats.gt_privileged_loss),
                        "mean_privileged_advantage": float(stats.mean_privileged_advantage),
                        "mean_gt_privileged_advantage": float(stats.mean_gt_privileged_advantage),
                        "total_loss": float(stats.total_loss),
                        "grad_norm": float(stats.grad_norm),
                        "privileged_pairs_used": int(stats.privileged_pairs_used),
                        "gt_privileged_pairs_used": int(stats.gt_privileged_pairs_used),
                        "mle_samples_used": int(stats.mle_samples_used),
                        **{k: float(v) for k, v in stats.lora_health.items()},
                    },
                )
                if args.online_save_every_updates > 0 and updates % args.online_save_every_updates == 0:
                    ckpt_dir = output_root / f"checkpoint-update-{updates}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrap_model_for_save(model).save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[privileged_hidden_opsd] saved checkpoint to {ckpt_dir}", flush=True)

            print(
                f"[privileged_hidden_opsd] rollout_step={rollout_steps}/{total_steps_str} "
                f"updates_in_rollout={updates_in_rollout}",
                flush=True,
            )
            buffer = []
            if args.online_steps is not None and rollout_steps >= args.online_steps:
                break

        if buffer and (args.online_steps is None or rollout_steps < args.online_steps):
            print("[privileged_hidden_opsd] remaining tail batch ignored to keep fixed rollout_batch_size behavior")

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model_for_save(model).save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(
        f"[privileged_hidden_opsd] finished rollout_steps={rollout_steps} optimizer_steps={updates} scanned={scanned} "
        f"kept_mle_samples={kept_mle_samples} kept_priv_pairs={kept_priv_pairs} "
        f"kept_gt_priv_pairs={kept_gt_priv_pairs} logged_mixed={logged_mixed} "
        f"logged_mixed_mle_only={logged_mixed_mle_only} "
        f"logged_all_correct={logged_all_correct} logged_all_wrong={logged_all_wrong} "
        f"skipped_all_wrong={skipped_all_wrong} skipped_after_filter={skipped_after_filter} final_model={final_dir}",
        flush=True,
    )
    write_metric(
        "run_end",
        {
            "rollout_steps": int(rollout_steps),
            "optimizer_steps": int(updates),
            "scanned": int(scanned),
            "kept_mle_samples": int(kept_mle_samples),
            "kept_priv_pairs": int(kept_priv_pairs),
            "kept_gt_priv_pairs": int(kept_gt_priv_pairs),
            "logged_mixed": int(logged_mixed),
            "logged_mixed_mle_only": int(logged_mixed_mle_only),
            "logged_all_correct": int(logged_all_correct),
            "logged_all_wrong": int(logged_all_wrong),
            "skipped_all_wrong": int(skipped_all_wrong),
            "skipped_after_filter": int(skipped_after_filter),
            "final_model": str(final_dir),
            "metrics_jsonl": str(metrics_path),
            "rollout_log": str(rollout_log_path),
        },
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = build_cli_parser(DEFAULT_SYSTEM_PROMPT)
    parser.description = "Online MLE + hidden-state-routed privileged OPSD-style scoring."
    parser.add_argument(
        "--lambda_priv",
        type=float,
        default=0.25,
        help="Weight for mixed prompts: privileged-context OPSD distillation on wrong trajectories.",
    )
    parser.add_argument(
        "--privileged_jsd_beta",
        type=float,
        default=-1.0,
        help="Beta for generalized JSD. Negative means reuse --beta.",
    )
    parser.add_argument(
        "--privileged_distill_temperature",
        type=float,
        default=1.0,
        help="Temperature applied to student/teacher logits before OPSD JSD.",
    )
    parser.add_argument(
        "--privileged_pointwise_kl_clip",
        type=float,
        default=0.05,
        help="If >0, clamp each vocab-entry contribution before summing token JSD, OPSD-style.",
    )
    parser.add_argument(
        "--privileged_logit_clip_abs",
        type=float,
        default=80.0,
        help="Clip fp32 student/teacher logits inside privileged JSD. Use <=0 to disable.",
    )
    parser.add_argument(
        "--privileged_prompt_style",
        type=str,
        default="compact",
        choices=["compact"],
        help="Template style for teacher prompts with private reference traces.",
    )
    parser.add_argument(
        "--privileged_trace_max_chars",
        type=int,
        default=0,
        help="Optional character cap for private reference traces. 0 keeps the full trace.",
    )
    parser.add_argument(
        "--fixed_teacher",
        type=str2bool,
        default=False,
        help="If true with LoRA, disable adapters for teacher forward, matching OPSD fixed-teacher main setting.",
    )
    parser.add_argument(
        "--log_rollout_text",
        type=str2bool,
        default=False,
        help="If true, rollout_records.jsonl stores full response and privileged prompt text.",
    )
    parser.add_argument(
        "--mle_include_mixed_correct",
        type=str2bool,
        default=False,
        help=(
            "If true, include correct trajectories from mixed samples into MLE updates even when privileged "
            "examples are disabled."
        ),
    )
    parser.add_argument(
        "--online_sanitize_gradients",
        type=str2bool,
        default=True,
        help="If true, replace non-finite gradients with 0 and optionally clamp gradient elements before norm/clipping.",
    )
    parser.add_argument(
        "--online_grad_element_clip_abs",
        type=float,
        default=100.0,
        help="If >0 with --online_sanitize_gradients, clamp each gradient element into [-clip_abs, clip_abs].",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validations = [
        (args.rollout_n < 2, "error: --rollout_n must be >= 2"),
        (args.rollout_batch_size < 1, "error: --rollout_batch_size must be >= 1"),
        (args.online_pairs_per_step < 1, "error: --online-pairs-per-step must be >= 1"),
        (
            args.lambda_mle < 0 or args.lambda_priv < 0 or args.lambda_gt < 0,
            "error: --lambda_mle/--lambda_priv/--lambda_gt must be >= 0",
        ),
        (
            args.lambda_mle == 0 and args.lambda_priv == 0 and args.lambda_gt == 0,
            "error: at least one loss weight must be > 0",
        ),
        (args.hidden_layer_offset < 1, "error: --hidden_layer_offset must be >= 1"),
        (
            args.rollout_feature_micro_batch_size < 0,
            "error: --rollout_feature_micro_batch_size must be >= 0",
        ),
        (args.logprob_micro_batch_size < 0, "error: --logprob_micro_batch_size must be >= 0"),
        (args.max_length < 1, "error: --max_length must be >= 1"),
        (args.privileged_distill_temperature <= 0, "error: --privileged_distill_temperature must be > 0"),
        (args.privileged_jsd_beta > 0 and args.privileged_jsd_beta >= 1, "error: --privileged_jsd_beta must be < 1"),
        (args.privileged_pointwise_kl_clip < 0, "error: --privileged_pointwise_kl_clip must be >= 0"),
        (args.privileged_logit_clip_abs < 0, "error: --privileged_logit_clip_abs must be >= 0"),
        (args.privileged_trace_max_chars < 0, "error: --privileged_trace_max_chars must be >= 0"),
        (args.online_grad_element_clip_abs < 0, "error: --online_grad_element_clip_abs must be >= 0"),
    ]
    for failed, message in validations:
        if failed:
            raise SystemExit(message)

    launcher_world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if launcher_world_size > 1:
        raise SystemExit(
            "error: train_privileged_hidden_opsd.py is single-process. Use CUDA_VISIBLE_DEVICES plus "
            "--tensor_parallel_size for vLLM rollout, not torchrun/DDP."
        )
    set_seed(args.seed)
    run_training(args)


if __name__ == "__main__":
    main()
