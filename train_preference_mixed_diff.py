#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from train_preference import (
    DEFAULT_SYSTEM_PROMPT,
    OnlinePendingObjective,
    OnlineStepLossStats,
    RolloutTrajectory,
    _compute_lora_param_health,
    _compute_sequence_logps_batch,
    _mean_or_nan,
    _online_rollout_completions_flat_hf,
    _online_rollout_completions_flat_vllm,
    apply_qwen_chat_template,
    build_online_bootstrap_jsonl_record,
    build_prompt_pool,
    build_rollout_trajectories_for_prompt,
    choose_system_prompt,
    compute_correct_trajectory_weights,
    ensure_input_require_grads_for_checkpointing,
    filter_weighted_sft_without_truncation,
    split_rollout_candidates_for_training,
    wrap_model_with_lora,
)
from utils import (
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    DapoSample,
    build_parser as build_cli_parser,
    compute_prompt_rarity_weight,
    compute_smoothed_correct_rate,
    detect_parquet_dataset_layout,
    iter_dapo_samples,
    iter_math_hf_samples,
    set_seed,
)


def _filter_trajectories_by_avg_logprob(
    trajectories: Sequence[RolloutTrajectory],
    min_avg_logprob: Optional[float],
) -> List[RolloutTrajectory]:
    if min_avg_logprob is None:
        return list(trajectories)
    return [traj for traj in trajectories if float(traj.avg_logprob) >= float(min_avg_logprob)]


def _mean_sequence_logp_for_prompt(
    *,
    model: object,
    tokenizer: object,
    device: torch.device,
    prompt_text: str,
    completion_texts: Sequence[str],
    max_length: int,
    micro_batch_size: int,
) -> torch.Tensor:
    completions = [str(x) for x in completion_texts]
    if not completions:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    mb = micro_batch_size if micro_batch_size > 0 else len(completions)
    sum_logp = torch.tensor(0.0, device=device, dtype=torch.float32)
    count = 0
    for start in range(0, len(completions), mb):
        end = min(start + mb, len(completions))
        batch_completions = completions[start:end]
        batch_prompts = [prompt_text] * len(batch_completions)
        batch_logps = _compute_sequence_logps_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=batch_prompts,
            completion_texts=batch_completions,
            max_length=max_length,
            device=device,
        )
        sum_logp = sum_logp + batch_logps.float().sum()
        count += int(batch_logps.numel())
    if count <= 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    return sum_logp / float(count)


def _online_run_mixed_diff_optimizer_step(
    model: object,
    tokenizer: object,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
    mixed_objectives: Sequence[OnlinePendingObjective],
    mle_train_prompts: Sequence[str],
    mle_completions: Sequence[str],
    mle_weights: Sequence[float],
) -> OnlineStepLossStats:
    """Single optimizer.step() on mixed-diff objectives + all-correct MLE."""

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

    effective_mixed: List[Tuple[OnlinePendingObjective, List[RolloutTrajectory], List[RolloutTrajectory], float]] = []
    for objective in mixed_objectives:
        if objective.objective_type != "mixed_diff":
            continue
        raw_weight = float(args.lambda_pref) * float(objective.prompt_weight)
        if raw_weight <= 0:
            continue
        correct_kept = _filter_trajectories_by_avg_logprob(
            objective.correct,
            args.online_pref_min_avg_logprob_chosen,
        )
        wrong_kept = _filter_trajectories_by_avg_logprob(
            objective.wrong,
            args.online_pref_min_avg_logprob_rejected,
        )
        if not correct_kept or not wrong_kept:
            continue
        effective_mixed.append((objective, correct_kept, wrong_kept, raw_weight))

    mle_prompt_list = [str(x) for x in mle_train_prompts]
    mle_completion_list = [str(x) for x in mle_completions]
    mle_weight_list = [float(x) for x in mle_weights]

    total_weight = float(sum(w for _, _, _, w in effective_mixed) + sum(mle_weight_list))
    if total_weight <= 0:
        return _build_zero_stats("zero_total_weight")

    mixed_weighted_sum = 0.0
    mle_weighted_sum = 0.0
    gap_weighted_sum = 0.0
    mixed_weight_sum = 0.0
    used_mixed_objectives = 0
    used_mle_samples = 0
    backward_calls = 0

    optimizer.zero_grad(set_to_none=True)

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

    for idx, (objective, correct_kept, wrong_kept, prompt_weight) in enumerate(effective_mixed):
        mean_correct_logp = _mean_sequence_logp_for_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=objective.train_prompt,
            completion_texts=[traj.response_text for traj in correct_kept],
            max_length=args.max_length,
            micro_batch_size=args.logprob_micro_batch_size,
        )
        mean_wrong_logp = _mean_sequence_logp_for_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=objective.train_prompt,
            completion_texts=[traj.response_text for traj in wrong_kept],
            max_length=args.max_length,
            micro_batch_size=args.logprob_micro_batch_size,
        )
        gap = mean_correct_logp - mean_wrong_logp
        if args.online_gap_clip_abs > 0:
            gap = gap.clamp(-args.online_gap_clip_abs, args.online_gap_clip_abs)
        mixed_loss = -F.logsigmoid(float(args.beta) * gap)
        loss_chunk = mixed_loss * (prompt_weight / total_weight)
        loss_chunk_val = float(loss_chunk.detach().item())
        failed = _check_chunk_and_backward(
            loss_chunk=loss_chunk,
            loss_chunk_val=loss_chunk_val,
            skip_prefix="mixed_diff",
            start=idx,
            end=idx + 1,
        )
        if failed is not None:
            return failed
        backward_calls += 1
        used_mixed_objectives += 1
        mixed_weighted_sum += float(mixed_loss.detach().item()) * prompt_weight
        gap_weighted_sum += float(gap.detach().item()) * prompt_weight
        mixed_weight_sum += prompt_weight

    mle_batch = len(mle_prompt_list)
    mb_mle = args.logprob_micro_batch_size if args.logprob_micro_batch_size > 0 else max(mle_batch, 1)
    if mle_batch > 0:
        for start in range(0, mle_batch, mb_mle):
            end = min(start + mb_mle, mle_batch)
            tp = mle_prompt_list[start:end]
            cp = mle_completion_list[start:end]
            w = torch.tensor(mle_weight_list[start:end], device=device, dtype=torch.float32)
            if float(w.sum().item()) <= 0:
                continue
            logps = _compute_sequence_logps_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=tp,
                completion_texts=cp,
                max_length=args.max_length,
                device=device,
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
            backward_calls += 1
            used_mle_samples += (end - start)
            mle_weighted_sum += float((mle_loss_vec.detach() * w).sum().item())

    if backward_calls == 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("no_effective_samples")

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

    mean_gap = gap_weighted_sum / mixed_weight_sum if mixed_weight_sum > 0 else 0.0
    mixed_loss = mixed_weighted_sum / mixed_weight_sum if mixed_weight_sum > 0 else 0.0
    mle_weight_sum = float(sum(mle_weight_list))
    mle_loss = mle_weighted_sum / mle_weight_sum if mle_weight_sum > 0 else 0.0
    total_loss = (mixed_weighted_sum + mle_weighted_sum) / total_weight
    lora_health = _compute_lora_param_health(model)
    if args.online_abort_on_lora_nan and lora_health["lora_nan_ratio"] > 0:
        raise RuntimeError(
            "Detected NaN in LoRA params after optimizer.step: "
            f"lora_nan_ratio={lora_health['lora_nan_ratio']:.6f}"
        )
    return OnlineStepLossStats(
        total_loss=total_loss,
        mle_loss=mle_loss,
        pref_loss=mixed_loss,
        gt_pref_loss=0.0,
        mean_gap=mean_gap,
        pref_pairs_used=used_mixed_objectives,
        gt_pref_pairs_used=0,
        mle_samples_used=used_mle_samples,
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
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if args.use_lora:
            ensure_input_require_grads_for_checkpointing(model)

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
    kept_mixed_objectives = 0
    kept_mle_samples = 0
    skipped_all_wrong = 0
    skipped_after_filter = 0
    logged_mixed_objectives = 0
    logged_all_correct_objectives = 0
    buffer: List[DapoSample] = []
    k = args.online_pairs_per_step

    total_steps_str = str(args.online_steps) if args.online_steps is not None else "inf"
    print(
        f"[online-mixed-diff] dataset_layout={layout}, "
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
        f"lambda_mle={args.lambda_mle}, lambda_pref={args.lambda_pref}, "
        f"beta={args.beta}, gap_clip_abs={args.online_gap_clip_abs}, "
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
            "metrics_jsonl": str(metrics_jsonl_path),
        },
    )
    if args.online_rollout_backend == "vllm" and device.type != "cuda":
        raise RuntimeError("online_rollout_backend=vllm requires a CUDA device.")

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
                        model,
                        tokenizer,
                        device,
                        prompt_texts,
                        args,
                    )

            model.train()

            rollout_objectives: List[OnlinePendingObjective] = []
            mixed_objectives_in_rollout = 0
            all_correct_objectives_in_rollout = 0
            skipped_all_wrong_in_rollout = 0
            skipped_after_filter_in_rollout = 0
            sampled_correct_total_in_rollout = 0
            sampled_candidates_total_in_rollout = 0
            rollout_all_entropy_values: List[float] = []
            rollout_correct_entropy_values: List[float] = []
            rollout_wrong_entropy_values: List[float] = []
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

                trajectories = build_rollout_trajectories_for_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    train_prompt=prompt_texts[idx],
                    candidates=candidates,
                    split=split,
                    args=args,
                )
                rollout_all_entropy_values.extend([float(t.avg_entropy) for t in trajectories])
                correct_trajs = [trajectories[i] for i in split.correct_kept_indices]
                wrong_trajs = [trajectories[i] for i in split.wrong_kept_indices]
                rollout_correct_entropy_values.extend([float(t.avg_entropy) for t in correct_trajs])
                rollout_wrong_entropy_values.extend([float(t.avg_entropy) for t in wrong_trajs])
                objective: Optional[OnlinePendingObjective] = None

                if n_correct_total > 0 and n_correct_total < n_total:
                    if correct_trajs and wrong_trajs:
                        objective = OnlinePendingObjective(
                            sample_id=sample_obj.sample_id,
                            ground_truth=sample_obj.ground_truth,
                            train_prompt=prompt_texts[idx],
                            objective_type="mixed_diff",
                            rho_hat=rho_hat,
                            prompt_weight=prompt_weight,
                            correct=correct_trajs,
                            wrong=wrong_trajs,
                            correct_traj_weights=[],
                            mixed_pref_pairs=[],
                            gt_positive=None,
                        )
                        rollout_objectives.append(objective)
                        mixed_objectives_in_rollout += 1
                        logged_mixed_objectives += 1
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
                f"[online-mixed-diff] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"mixed_in_rollout={mixed_objectives_in_rollout} "
                f"all_correct_in_rollout={all_correct_objectives_in_rollout} "
                f"skipped_all_wrong_in_rollout={skipped_all_wrong_in_rollout} "
                f"skipped_after_filter_in_rollout={skipped_after_filter_in_rollout} "
                f"objectives_ready_for_update={len(rollout_objectives)}"
            )
            ent_overall = _mean_or_nan(rollout_all_entropy_values)
            ent_correct = _mean_or_nan(rollout_correct_entropy_values)
            ent_wrong = _mean_or_nan(rollout_wrong_entropy_values)
            ent_gap_wrong_minus_correct = (
                float(ent_wrong - ent_correct)
                if not math.isnan(ent_wrong) and not math.isnan(ent_correct)
                else float("nan")
            )
            print(
                f"[online-mixed-diff] rollout_step={rollout_steps}/{total_steps_str} "
                f"entropy_overall_mean={ent_overall:.4f} "
                f"entropy_overall_count={len(rollout_all_entropy_values)} "
                f"entropy_correct_mean={ent_correct:.4f} "
                f"entropy_wrong_mean={ent_wrong:.4f} "
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
                    "entropy_gap_wrong_minus_correct": float(ent_gap_wrong_minus_correct),
                },
            )

            updates_in_rollout = 0
            consumed_mixed_objectives_in_rollout = 0
            consumed_mle_samples_in_rollout = 0
            dropped_mle_by_truncation_in_rollout = 0
            last_optimizer_skip_reason: Optional[str] = None
            if rollout_objectives:
                for chunk_start in range(0, len(rollout_objectives), k):
                    chunk = rollout_objectives[chunk_start : chunk_start + k]
                    mixed_objectives_chunk: List[OnlinePendingObjective] = []
                    mle_train_prompts_raw: List[str] = []
                    mle_completions_raw: List[str] = []
                    mle_weights_raw: List[float] = []

                    for objective in chunk:
                        if objective.objective_type == "mixed_diff":
                            mixed_objectives_chunk.append(objective)
                        elif objective.objective_type == "all_correct":
                            for traj, traj_weight in zip(objective.correct, objective.correct_traj_weights):
                                mle_train_prompts_raw.append(objective.train_prompt)
                                mle_completions_raw.append(traj.response_text)
                                mle_weights_raw.append(
                                    float(args.lambda_mle) * float(objective.prompt_weight) * float(traj_weight)
                                )

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

                    if not mixed_objectives_chunk and not mle_train_prompts:
                        continue

                    loss_stats = _online_run_mixed_diff_optimizer_step(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        device=device,
                        args=args,
                        mixed_objectives=mixed_objectives_chunk,
                        mle_train_prompts=mle_train_prompts,
                        mle_completions=mle_completions,
                        mle_weights=mle_weights,
                    )
                    if not loss_stats.update_applied:
                        last_optimizer_skip_reason = loss_stats.skip_reason
                        print(
                            f"[online-mixed-diff] rollout_step={rollout_steps}/{total_steps_str} "
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
                    consumed_mixed_objectives_in_rollout += loss_stats.pref_pairs_used
                    consumed_mle_samples_in_rollout += loss_stats.mle_samples_used
                    kept_mixed_objectives += loss_stats.pref_pairs_used
                    kept_mle_samples += loss_stats.mle_samples_used
                    print(
                        f"[online-mixed-diff] rollout_step={rollout_steps}/{total_steps_str} "
                        f"optimizer_step={updates} "
                        f"mixed_loss={loss_stats.pref_loss:.6f} "
                        f"mean_gap={loss_stats.mean_gap:.6f} "
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
                            "mixed_loss": float(loss_stats.pref_loss),
                            "mean_gap": float(loss_stats.mean_gap),
                            "mle_loss": float(loss_stats.mle_loss),
                            "total_loss": float(loss_stats.total_loss),
                            "grad_norm": float(loss_stats.grad_norm),
                            "mixed_objectives_used": int(loss_stats.pref_pairs_used),
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
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        print(f"[online-mixed-diff] saved checkpoint to {ckpt_dir}")

            if rollout_objectives and updates_in_rollout == 0:
                hint = (
                    f"last_skip_reason={last_optimizer_skip_reason!r}"
                    if last_optimizer_skip_reason
                    else "no_chunk_reached_optimizer"
                )
                print(
                    f"[online-mixed-diff] rollout_step={rollout_steps}/{total_steps_str} "
                    f"no optimizer update applied ({hint}; see skip lines above or empty batch)"
                )
            elif rollout_objectives:
                print(
                    f"[online-mixed-diff] rollout_step={rollout_steps}/{total_steps_str} "
                    f"updates_in_rollout={updates_in_rollout} "
                    f"consumed_mixed_objectives_in_rollout={consumed_mixed_objectives_in_rollout} "
                    f"consumed_mle_samples_in_rollout={consumed_mle_samples_in_rollout} "
                    f"dropped_mle_by_truncation_in_rollout={dropped_mle_by_truncation_in_rollout}"
                )

            buffer = []
            if args.online_steps is not None and rollout_steps >= args.online_steps:
                break

        if buffer and (args.online_steps is None or rollout_steps < args.online_steps):
            print("[online-mixed-diff] remaining tail batch ignored to keep fixed rollout_batch_size behavior")

    final_dir = output_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(
        f"[online-mixed-diff] finished. rollout_steps={rollout_steps}, optimizer_steps={updates}, "
        f"scanned={scanned}, kept_mixed_objectives={kept_mixed_objectives}, "
        f"kept_mle_samples={kept_mle_samples}, "
        f"logged_mixed_objectives={logged_mixed_objectives}, "
        f"logged_all_correct_objectives={logged_all_correct_objectives}, "
        f"skipped_all_wrong={skipped_all_wrong}, skipped_after_filter={skipped_after_filter}, "
        f"objectives_log={online_pairs_path}, final_model={final_dir}"
    )
    _write_metric(
        "run_end",
        {
            "rollout_steps": int(rollout_steps),
            "optimizer_steps": int(updates),
            "scanned": int(scanned),
            "kept_mixed_objectives": int(kept_mixed_objectives),
            "kept_mle_samples": int(kept_mle_samples),
            "logged_mixed_objectives": int(logged_mixed_objectives),
            "logged_all_correct_objectives": int(logged_all_correct_objectives),
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

    if args.online_mle_on_correct_only:
        print("[online-mixed-diff] ignore --online_mle_on_correct_only; behavior is fixed in this script.")
    if args.online_pref_loss_only:
        print("[online-mixed-diff] ignore --online_pref_loss_only; behavior is fixed in this script.")
    if args.lambda_gt != 0:
        print("[online-mixed-diff] ignore --lambda_gt; all-wrong branch is disabled in this script.")

    if args.max_source_samples == 0:
        args.max_source_samples = None
    if args.online_steps == 0:
        args.online_steps = None

    validations = [
        (args.online_pairs_per_step < 1, "error: --online-pairs-per-step must be >= 1"),
        (args.rollout_n < 2, "error: --rollout_n must be >= 2"),
        (args.beta <= 0, "error: --beta must be > 0"),
        (
            args.lambda_mle < 0 or args.lambda_pref < 0,
            "error: --lambda_mle/--lambda_pref must be >= 0",
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

