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
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW

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
    split_rollout_candidates_for_training,
    unwrap_model_for_save,
    wrap_model_with_lora,
)
from utils import (
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    DapoSample,
    compute_prompt_rarity_weight,
    compute_smoothed_correct_rate,
    detect_parquet_dataset_layout,
    iter_dapo_samples,
    iter_math_hf_samples,
    set_seed,
    str2bool,
)


@dataclass
class GroupObjective:
    sample_id: str
    ground_truth: str
    train_prompt: str
    correct: List[str]
    wrong: List[str]
    rho_hat: float
    prompt_weight: float

    @property
    def is_mixed(self) -> bool:
        return bool(self.correct) and bool(self.wrong)


@dataclass
class OptimizerStepStats:
    total_loss: float
    mle_loss: float
    group_loss: float
    mean_gap: float
    group_correct_mass: float
    hard_weight: float
    groups_used: int
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


def _load_source_iter(args: argparse.Namespace):
    layout = args.dataset_layout
    if layout == "auto":
        layout = detect_parquet_dataset_layout(args.dataset_path)
    if layout == "dapo":
        source_iter = iter_dapo_samples(
            parquet_path=args.dataset_path,
            scan_batch_size=args.scan_batch_size,
            max_source_samples=args.max_source_samples,
            gold_rationale_key_paths=(),
            require_gold_rationale=False,
        )
    elif layout == "math_hf":
        source_iter = iter_math_hf_samples(
            parquet_path=args.dataset_path,
            scan_batch_size=args.scan_batch_size,
            max_source_samples=args.max_source_samples,
            gold_rationale_key_paths=(),
            require_gold_rationale=False,
        )
    else:
        raise ValueError(f"Unsupported --dataset_layout: {layout}")
    return layout, source_iter


def _compute_sequence_logps_and_counts(
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, labels = _labeled_batch_tensors(
        tokenizer, prompt_texts, completion_texts, max_length, device
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    seq_logps = _seq_logps_from_logits_labels(logits, labels)
    valid_counts = labels[:, 1:].ne(-100).sum(dim=-1)
    return seq_logps, valid_counts


def _normalize_group_scores(scores: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mode = str(args.group_score_norm)
    if mode == "none":
        out = scores
    elif mode == "zscore":
        detached = scores.detach()
        center = detached.mean()
        scale = detached.std(unbiased=False).clamp_min(float(args.group_score_std_floor))
        out = (scores - center) / scale
    else:
        raise ValueError(f"Unsupported --group_score_norm: {mode}")
    if args.group_score_clip_abs > 0:
        out = out.clamp(-float(args.group_score_clip_abs), float(args.group_score_clip_abs))
    return out


def _soft_anchor(scores: torch.Tensor, tau: float) -> torch.Tensor:
    return float(tau) * torch.logsumexp(scores / float(tau), dim=0)


def _detached_hard_weight_from_group(
    pos_scores: torch.Tensor,
    wrong_scores: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Return a detached hard-negative weight and diagnostics.

    The weight is computed from positive-vs-wrong set scores, but it is used only
    as a scalar multiplier on positive CE. Rejected/wrong responses never receive
    gradient in this objective.
    """
    with torch.no_grad():
        group_scores = torch.cat([pos_scores.detach(), wrong_scores.detach()], dim=0)
        pos_mask = torch.zeros(group_scores.shape[0], dtype=torch.bool, device=group_scores.device)
        pos_mask[: pos_scores.shape[0]] = True
        norm_scores = _normalize_group_scores(group_scores, args)
        pos_anchor = _soft_anchor(norm_scores[pos_mask], float(args.group_tau))
        wrong_anchor = _soft_anchor(norm_scores[~pos_mask], float(args.group_tau))
        gap = pos_anchor - wrong_anchor
        violation = (wrong_anchor - pos_anchor + float(args.group_margin)) / float(args.hard_weight_tau)
        if args.one_sided_weight_type == "logsigmoid":
            hard_weight = F.softplus(violation)
        elif args.one_sided_weight_type == "hinge":
            hard_weight = F.relu(violation)
        else:
            raise ValueError(f"Unsupported --one_sided_weight_type: {args.one_sided_weight_type}")
        if args.hard_weight_min > 0:
            hard_weight = hard_weight.clamp_min(float(args.hard_weight_min))
        if args.hard_weight_max > 0:
            hard_weight = hard_weight.clamp_max(float(args.hard_weight_max))
        confidence = torch.sigmoid((pos_anchor - wrong_anchor - float(args.group_margin)) / float(args.hard_weight_tau))
        return (
            hard_weight.detach(),
            float(gap.item()),
            float(confidence.item()),
            float(pos_anchor.item()),
            float(wrong_anchor.item()),
        )


def _build_zero_stats(reason: str, grad_norm: float = 0.0) -> OptimizerStepStats:
    return OptimizerStepStats(
        total_loss=0.0,
        mle_loss=0.0,
        group_loss=0.0,
        mean_gap=0.0,
        group_correct_mass=0.0,
        hard_weight=0.0,
        groups_used=0,
        mle_samples_used=0,
        grad_norm=grad_norm,
        update_applied=False,
        skip_reason=reason,
        lora_health=_zero_lora_health(),
    )


def _objective_potential_weight(obj: GroupObjective, args: argparse.Namespace) -> float:
    if not obj.correct:
        return 0.0
    if args.lambda_mle > 0 or (obj.is_mixed and args.lambda_group > 0):
        return float(obj.prompt_weight)
    return 0.0


def run_optimizer_step(
    *,
    model: object,
    tokenizer: object,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
    objectives: Sequence[GroupObjective],
) -> OptimizerStepStats:
    total_weight = sum(_objective_potential_weight(obj, args) for obj in objectives)
    if total_weight <= 0:
        return _build_zero_stats("zero_total_weight")

    optimizer.zero_grad(set_to_none=True)
    total_loss_weighted_sum = 0.0
    mle_loss_weighted_sum = 0.0
    group_loss_weighted_sum = 0.0
    gap_weighted_sum = 0.0
    mass_weighted_sum = 0.0
    hard_weight_weighted_sum = 0.0
    mle_weight_sum = 0.0
    group_weight_sum = 0.0
    groups_used = 0
    mle_samples_used = 0

    for obj_idx, obj in enumerate(objectives):
        if not obj.correct:
            continue

        pos_prompts = [obj.train_prompt for _ in obj.correct]
        pos_scores, pos_valid_counts = _compute_sequence_logps_and_counts(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=pos_prompts,
            completion_texts=obj.correct,
            max_length=args.max_length,
            device=device,
        )
        pos_valid_mask = torch.isfinite(pos_scores) & pos_valid_counts.gt(0)

        loss_terms: List[torch.Tensor] = []
        obj_mle_loss_value = 0.0
        obj_group_loss_value = 0.0
        obj_group_used = False

        mle_count = int(pos_valid_mask.sum().item())
        if mle_count > 0 and args.lambda_mle > 0:
            mle_loss = -pos_scores[pos_valid_mask].mean()
            mle_weight = float(args.lambda_mle) * float(obj.prompt_weight)
            loss_terms.append(mle_loss * mle_weight)
            obj_mle_loss_value = float(mle_loss.detach().item())
            mle_loss_weighted_sum += obj_mle_loss_value * mle_weight
            mle_weight_sum += mle_weight
            mle_samples_used += mle_count

        if obj.is_mixed and args.lambda_group > 0:
            group_pos_mask = pos_valid_mask.clone()
            if args.min_correct_avg_logprob is not None:
                group_pos_mask = group_pos_mask & (pos_scores >= float(args.min_correct_avg_logprob))

            with torch.no_grad():
                wrong_prompts = [obj.train_prompt for _ in obj.wrong]
                wrong_scores, wrong_valid_counts = _compute_sequence_logps_and_counts(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=wrong_prompts,
                    completion_texts=obj.wrong,
                    max_length=args.max_length,
                    device=device,
                )
                wrong_valid_mask = torch.isfinite(wrong_scores) & wrong_valid_counts.gt(0)
                if args.min_wrong_avg_logprob is not None:
                    wrong_valid_mask = wrong_valid_mask & (wrong_scores >= float(args.min_wrong_avg_logprob))

            kept_correct = int(group_pos_mask.sum().item())
            kept_wrong = int(wrong_valid_mask.sum().item())
            if kept_correct > 0 and kept_wrong > 0:
                group_pos_scores = pos_scores[group_pos_mask]
                group_wrong_scores = wrong_scores[wrong_valid_mask]
                hard_weight, gap, confidence, _pos_anchor, _wrong_anchor = _detached_hard_weight_from_group(
                    group_pos_scores,
                    group_wrong_scores,
                    args,
                )
                group_pos_ce = -group_pos_scores.mean()
                group_loss = hard_weight.to(device=device, dtype=group_pos_ce.dtype) * group_pos_ce
                group_weight = float(args.lambda_group) * float(obj.prompt_weight)
                loss_terms.append(group_loss * group_weight)

                obj_group_used = True
                obj_group_loss_value = float(group_loss.detach().item())
                group_loss_weighted_sum += obj_group_loss_value * group_weight
                gap_weighted_sum += gap * group_weight
                mass_weighted_sum += confidence * group_weight
                hard_weight_weighted_sum += float(hard_weight.item()) * group_weight
                group_weight_sum += group_weight
                groups_used += 1
            else:
                print(
                    f"[one_sided_group_mle] group skipped after filters obj={obj_idx} "
                    f"kept_correct={kept_correct} kept_wrong={kept_wrong}",
                    flush=True,
                )

        if not loss_terms:
            continue
        loss_unscaled = torch.stack(loss_terms).sum()
        loss_chunk = loss_unscaled / float(total_weight)
        loss_chunk_val = float(loss_chunk.detach().item())
        if args.online_skip_nonfinite_loss and not torch.isfinite(loss_chunk.detach()):
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(f"nonfinite_loss(obj={obj_idx})")
        if args.online_loss_value_cap > 0 and abs(loss_chunk_val) > args.online_loss_value_cap:
            optimizer.zero_grad(set_to_none=True)
            return _build_zero_stats(
                f"loss_too_large(obj={obj_idx},value={loss_chunk_val:.4f},cap={args.online_loss_value_cap:.4f})"
            )
        loss_chunk.backward()
        total_loss_weighted_sum += obj_mle_loss_value * (
            float(args.lambda_mle) * float(obj.prompt_weight) if mle_count > 0 else 0.0
        )
        total_loss_weighted_sum += obj_group_loss_value * (
            float(args.lambda_group) * float(obj.prompt_weight) if obj_group_used else 0.0
        )

    if mle_weight_sum + group_weight_sum <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("all_objectives_filtered")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.max_grad_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
    else:
        parts = [torch.linalg.vector_norm(p.grad.detach(), ord=2) for p in trainable if p.grad is not None]
        total_norm = torch.linalg.vector_norm(torch.stack(parts), ord=2) if parts else torch.tensor(0.0, device=device)
    grad_norm = float(total_norm.item()) if isinstance(total_norm, torch.Tensor) else float(total_norm)
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

    return OptimizerStepStats(
        total_loss=total_loss_weighted_sum / total_weight,
        mle_loss=mle_loss_weighted_sum / mle_weight_sum if mle_weight_sum > 0 else 0.0,
        group_loss=group_loss_weighted_sum / group_weight_sum if group_weight_sum > 0 else 0.0,
        mean_gap=gap_weighted_sum / group_weight_sum if group_weight_sum > 0 else 0.0,
        group_correct_mass=mass_weighted_sum / group_weight_sum if group_weight_sum > 0 else 0.0,
        hard_weight=hard_weight_weighted_sum / group_weight_sum if group_weight_sum > 0 else 0.0,
        groups_used=groups_used,
        mle_samples_used=mle_samples_used,
        grad_norm=grad_norm,
        update_applied=True,
        skip_reason="",
        lora_health=lora_health,
    )


def _rollout_record(
    *,
    sample: DapoSample,
    split: Any,
    objective: Optional[GroupObjective],
    candidates: Sequence[str],
    log_text: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "ground_truth": sample.ground_truth,
        "objective_type": "skip" if objective is None else ("mixed_group" if objective.is_mixed else "all_correct"),
        "n_total": len(candidates),
        "n_correct": int(sum(1 for x in split.responses_correct if x)),
        "n_wrong": int(sum(1 for x in split.responses_correct if not x)),
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
        print(f"[one_sided_group_mle] enabled DataParallel over {torch.cuda.device_count()} GPUs", flush=True)
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
    if args.auto_math_hf_user_suffix and layout == "math_hf" and not rollout_user_suffix.strip():
        rollout_user_suffix = DEFAULT_MATH_HF_USER_CONTENT_SUFFIX

    prompt_pool = build_prompt_pool(args)
    prompt_rng = random.Random(args.seed + 20260412)
    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"
    print(
        f"[one_sided_group_mle] dataset_layout={layout} rollout_backend={args.online_rollout_backend} "
        f"rollout_batch_size={args.rollout_batch_size} rollout_n={args.rollout_n} "
        f"objectives_per_update={args.online_pairs_per_step} online_steps={total_steps_str} "
        f"lambda_mle={args.lambda_mle} lambda_group={args.lambda_group} "
        f"group_tau={args.group_tau} group_score_norm={args.group_score_norm} "
        f"one_sided_weight_type={args.one_sided_weight_type} hard_weight_tau={args.hard_weight_tau} "
        f"group_margin={args.group_margin} hard_weight_max={args.hard_weight_max}",
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
            "group_tau": float(args.group_tau),
            "group_score_norm": str(args.group_score_norm),
            "one_sided_weight_type": str(args.one_sided_weight_type),
            "group_margin": float(args.group_margin),
            "hard_weight_tau": float(args.hard_weight_tau),
            "hard_weight_min": float(args.hard_weight_min),
            "hard_weight_max": float(args.hard_weight_max),
            "rollout_n": int(args.rollout_n),
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
                        _rollout_record(
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
                f"[one_sided_group_mle] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"mixed={mixed_in_rollout} all_correct={all_correct_in_rollout} "
                f"skipped_all_wrong={skipped_all_wrong_in_rollout} sampled_correct_rate={sampled_correct_rate:.4f} "
                f"objectives={len(rollout_objectives)}",
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
                },
            )

            updates_in_rollout = 0
            for chunk_start in range(0, len(rollout_objectives), args.online_pairs_per_step):
                chunk = rollout_objectives[chunk_start : chunk_start + args.online_pairs_per_step]
                if not chunk:
                    continue
                stats = run_optimizer_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                    objectives=chunk,
                )
                if not stats.update_applied:
                    print(
                        f"[one_sided_group_mle] rollout_step={rollout_steps}/{total_steps_str} "
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
                kept_groups += stats.groups_used
                kept_mle_samples += stats.mle_samples_used
                print(
                    f"[one_sided_group_mle] rollout_step={rollout_steps}/{total_steps_str} optimizer_step={updates} "
                    f"mle_loss={stats.mle_loss:.6f} group_loss={stats.group_loss:.6f} "
                    f"mean_gap={stats.mean_gap:.6f} pref_confidence={stats.group_correct_mass:.6f} "
                    f"hard_weight={stats.hard_weight:.6f} "
                    f"total_loss={stats.total_loss:.6f} grad_norm={stats.grad_norm:.6f} "
                    f"groups_used={stats.groups_used} mle_samples_used={stats.mle_samples_used}",
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
                        **{k: float(v) for k, v in stats.lora_health.items()},
                    },
                )
                if args.online_save_every_updates > 0 and updates % args.online_save_every_updates == 0:
                    ckpt_dir = output_root / f"checkpoint-update-{updates}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unwrap_model_for_save(model).save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[one_sided_group_mle] saved checkpoint to {ckpt_dir}", flush=True)

            print(
                f"[one_sided_group_mle] rollout_step={rollout_steps}/{total_steps_str} updates_in_rollout={updates_in_rollout}",
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
        f"[one_sided_group_mle] finished rollout_steps={rollout_steps} optimizer_steps={updates} scanned={scanned} "
        f"kept_groups={kept_groups} kept_mle_samples={kept_mle_samples} "
        f"logged_mixed={logged_mixed} logged_all_correct={logged_all_correct} "
        f"skipped_all_wrong={skipped_all_wrong} final_model={final_dir}",
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
            "final_model": str(final_dir),
            "metrics_jsonl": str(metrics_path),
            "rollout_log": str(rollout_log_path),
        },
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online MLE + one-sided hard-negative-weighted positive MLE.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_layout", type=str, default="auto", choices=["auto", "dapo", "math_hf"])
    parser.add_argument("--scan_batch_size", type=int, default=1024)
    parser.add_argument("--max_source_samples", type=int, default=0)
    parser.add_argument("--user_content_suffix", type=str, default="")
    parser.add_argument("--auto_math_hf_user_suffix", type=str2bool, default=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--prompt_mode", type=str, default="none", choices=["none", "fixed", "random"])
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt_candidate", action="append", default=[])
    parser.add_argument("--prompt_candidates_file", type=str, default="")
    parser.add_argument("--use_default_prompt_candidates", type=str2bool, default=False)
    parser.add_argument("--prompt_fixed_index", type=int, default=0)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)

    parser.add_argument("--online_rollout_backend", type=str, default="vllm", choices=["vllm", "hf"])
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
    parser.add_argument("--online-pairs-per-step", dest="online_pairs_per_step", type=int, default=8)
    parser.add_argument("--online_save_every_updates", type=int, default=0)
    parser.add_argument("--log_rollout_text", type=str2bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_mle", type=float, default=1.0)
    parser.add_argument("--lambda_group", type=float, default=0.25)
    parser.add_argument("--group_tau", type=float, default=0.5)
    parser.add_argument("--group_score_norm", type=str, default="none", choices=["none", "zscore"])
    parser.add_argument("--group_score_std_floor", type=float, default=0.05)
    parser.add_argument("--group_score_clip_abs", type=float, default=0.0)
    parser.add_argument("--one_sided_weight_type", type=str, default="logsigmoid", choices=["logsigmoid", "hinge"])
    parser.add_argument("--group_margin", type=float, default=0.0)
    parser.add_argument("--hard_weight_tau", type=float, default=0.5)
    parser.add_argument("--hard_weight_min", type=float, default=0.0)
    parser.add_argument("--hard_weight_max", type=float, default=2.0)
    parser.add_argument("--min_correct_avg_logprob", type=float, default=None)
    parser.add_argument("--min_wrong_avg_logprob", type=float, default=None)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validations = [
        (args.rollout_n < 2, "error: --rollout_n must be >= 2"),
        (args.rollout_batch_size < 1, "error: --rollout_batch_size must be >= 1"),
        (args.online_pairs_per_step < 1, "error: --online-pairs-per-step must be >= 1"),
        (args.lambda_mle < 0 or args.lambda_group < 0, "error: --lambda_mle/--lambda_group must be >= 0"),
        (args.lambda_mle == 0 and args.lambda_group == 0, "error: at least one training loss weight must be > 0"),
        (args.group_tau <= 0, "error: --group_tau must be > 0"),
        (args.hard_weight_tau <= 0, "error: --hard_weight_tau must be > 0"),
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
            "error: train_onesided_group_mle.py is single-process. Use CUDA_VISIBLE_DEVICES plus "
            "--tensor_parallel_size for vLLM rollout, not torchrun/DDP."
        )
    set_seed(args.seed)
    run_training(args)


if __name__ == "__main__":
    main()
