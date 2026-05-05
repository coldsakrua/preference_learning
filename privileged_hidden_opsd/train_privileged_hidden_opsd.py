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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_preference import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    RolloutTrajectory,
    _compute_lora_param_health,
    _compute_sequence_logps_and_hidden_batch,
    _compute_sequence_logps_batch,
    _online_rollout_completions_flat_hf,
    _online_rollout_completions_flat_vllm,
    apply_qwen_chat_template,
    build_prompt_pool,
    choose_system_prompt,
    compute_correct_trajectory_weights,
    ensure_input_require_grads_for_checkpointing,
    rollout_trajectory_to_json,
    split_rollout_candidates_for_training,
    unwrap_model_for_save,
    wrap_model_with_lora,
)
from utils import (  # noqa: E402
    DEFAULT_MATH_HF_USER_CONTENT_SUFFIX,
    DapoSample,
    build_parser as build_cli_parser,
    compute_prompt_rarity_weight,
    compute_smoothed_correct_rate,
    detect_parquet_dataset_layout,
    iter_dapo_samples,
    iter_math_hf_samples,
    set_seed,
    strip_prompt_prefix_from_text,
    str2bool,
)


@dataclass
class PrivilegedWrongExample:
    train_prompt: str
    privileged_prompt: str
    wrong_response: str
    source_type: str  # "nearest_correct" | "gt_rationale"
    source_trace: str
    weight: float
    match_correct_index: int
    match_similarity: float
    wrong_avg_logprob: float


@dataclass
class PrivilegedObjective:
    sample_id: str
    ground_truth: str
    train_prompt: str
    objective_type: str  # "all_correct" | "mixed_privileged" | "all_wrong_gt_privileged"
    rho_hat: float
    prompt_weight: float
    correct: List[RolloutTrajectory]
    wrong: List[RolloutTrajectory]
    correct_weights: List[float]
    privileged_wrong_examples: List[PrivilegedWrongExample]


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


@dataclass
class DistillBatchTensors:
    student_input_ids: torch.Tensor
    student_attention_mask: torch.Tensor
    teacher_input_ids: torch.Tensor
    teacher_attention_mask: torch.Tensor
    student_prompt_lens: torch.Tensor
    teacher_prompt_lens: torch.Tensor
    target_ids: torch.Tensor
    token_mask: torch.Tensor
    weights: torch.Tensor
    kept_count: int
    total_count: int


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
    return layout, source_iter


def _truncate_privileged_trace(trace: str, max_chars: int) -> str:
    text = str(trace or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _format_privileged_user_prompt(
    *,
    prompt_user_effective: str,
    privileged_trace: str,
    source_type: str,
    args: argparse.Namespace,
) -> str:
    trace = _truncate_privileged_trace(
        privileged_trace,
        int(getattr(args, "privileged_trace_max_chars", 0)),
    )
    label = "ground-truth reference reasoning" if source_type == "gt_rationale" else "successful reference reasoning"
    if args.privileged_prompt_style == "compact":
        return (
            f"{prompt_user_effective}\n\n"
            f"Private {label} for the same problem:\n"
            f"{trace}\n\n"
            "Use the private reference only as hidden context, then solve the original problem."
        ).strip()
    raise ValueError(f"Unsupported --privileged_prompt_style: {args.privileged_prompt_style}")


def _build_privileged_prompt(
    tokenizer: object,
    *,
    prompt_user_effective: str,
    privileged_trace: str,
    source_type: str,
    system_prompt: str,
    args: argparse.Namespace,
) -> str:
    user_prompt = _format_privileged_user_prompt(
        prompt_user_effective=prompt_user_effective,
        privileged_trace=privileged_trace,
        source_type=source_type,
        args=args,
    )
    return apply_qwen_chat_template(
        tokenizer,
        user_prompt,
        enable_thinking=args.enable_thinking,
        system_prompt=system_prompt,
    )


def _gt_privileged_trace(sample: DapoSample) -> str:
    gt_text = strip_prompt_prefix_from_text(sample.prompt, sample.gold_rationale)
    if not gt_text and sample.ground_truth:
        gt_text = f"Answer: {sample.ground_truth}"
    return str(gt_text or "").strip()


def _build_rollout_trajectories_with_hidden(
    *,
    model: object,
    tokenizer: object,
    device: torch.device,
    train_prompt: str,
    candidates: Sequence[str],
    split: Any,
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
            seq_logps, seq_entropy, batch_hidden = _compute_sequence_logps_and_hidden_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts[start:end],
                completion_texts=list(candidates[start:end]),
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


def _nearest_correct_by_hidden(
    correct_trajs: Sequence[RolloutTrajectory],
    wrong_traj: RolloutTrajectory,
) -> tuple[int, float]:
    if not correct_trajs:
        return -1, float("nan")
    if wrong_traj.hidden_vec and all(t.hidden_vec for t in correct_trajs):
        wrong_vec = torch.tensor(wrong_traj.hidden_vec, dtype=torch.float32)
        correct_mat = torch.tensor([t.hidden_vec for t in correct_trajs], dtype=torch.float32)
        if correct_mat.ndim == 2 and correct_mat.shape[1] == wrong_vec.numel():
            sims = correct_mat @ wrong_vec
            best_idx = int(torch.argmax(sims).item())
            return best_idx, float(sims[best_idx].item())

    best_idx = min(range(len(correct_trajs)), key=lambda i: float(correct_trajs[i].avg_nll))
    return int(best_idx), float("nan")


def _build_mixed_privileged_examples(
    *,
    tokenizer: object,
    prompt_user_effective: str,
    system_prompt: str,
    train_prompt: str,
    correct_trajs: Sequence[RolloutTrajectory],
    wrong_trajs: Sequence[RolloutTrajectory],
    prompt_weight: float,
    args: argparse.Namespace,
) -> List[PrivilegedWrongExample]:
    if not correct_trajs or not wrong_trajs or args.lambda_priv <= 0:
        return []
    per_wrong_weight = float(args.lambda_priv) * float(prompt_weight) / float(len(wrong_trajs))
    examples: List[PrivilegedWrongExample] = []
    for wrong_traj in wrong_trajs:
        correct_idx, sim = _nearest_correct_by_hidden(correct_trajs, wrong_traj)
        if correct_idx < 0:
            continue
        source_trace = correct_trajs[correct_idx].response_text
        privileged_prompt = _build_privileged_prompt(
            tokenizer,
            prompt_user_effective=prompt_user_effective,
            privileged_trace=source_trace,
            source_type="nearest_correct",
            system_prompt=system_prompt,
            args=args,
        )
        examples.append(
            PrivilegedWrongExample(
                train_prompt=train_prompt,
                privileged_prompt=privileged_prompt,
                wrong_response=wrong_traj.response_text,
                source_type="nearest_correct",
                source_trace=source_trace,
                weight=per_wrong_weight,
                match_correct_index=int(correct_idx),
                match_similarity=float(sim),
                wrong_avg_logprob=float(wrong_traj.avg_logprob),
            )
        )
    return examples


def _build_gt_privileged_examples(
    *,
    tokenizer: object,
    sample: DapoSample,
    prompt_user_effective: str,
    system_prompt: str,
    train_prompt: str,
    wrong_trajs: Sequence[RolloutTrajectory],
    prompt_weight: float,
    args: argparse.Namespace,
) -> List[PrivilegedWrongExample]:
    if not wrong_trajs or not args.use_all_wrong_gt_preference or args.lambda_gt <= 0:
        return []
    source_trace = _gt_privileged_trace(sample)
    if not source_trace:
        return []
    privileged_prompt = _build_privileged_prompt(
        tokenizer,
        prompt_user_effective=prompt_user_effective,
        privileged_trace=source_trace,
        source_type="gt_rationale",
        system_prompt=system_prompt,
        args=args,
    )
    per_wrong_weight = float(args.lambda_gt) * float(prompt_weight) / float(len(wrong_trajs))
    return [
        PrivilegedWrongExample(
            train_prompt=train_prompt,
            privileged_prompt=privileged_prompt,
            wrong_response=wrong_traj.response_text,
            source_type="gt_rationale",
            source_trace=source_trace,
            weight=per_wrong_weight,
            match_correct_index=-1,
            match_similarity=float("nan"),
            wrong_avg_logprob=float(wrong_traj.avg_logprob),
        )
        for wrong_traj in wrong_trajs
    ]


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


def _safe_pad_token_id(tokenizer: object) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0


def _pad_id_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, empty
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        values = torch.tensor(list(seq), dtype=torch.long, device=device)
        input_ids[row, : values.numel()] = values
        attention_mask[row, : values.numel()] = 1
    return input_ids, attention_mask


def _build_privileged_distill_batch(
    *,
    tokenizer: object,
    student_prompts: Sequence[str],
    teacher_prompts: Sequence[str],
    completions: Sequence[str],
    weights: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[DistillBatchTensors]:
    """Build paired student/teacher inputs that share the exact wrong rollout tokens.

    OPSD distills the distribution on student on-policy tokens. Tokenizing the
    completion separately gives both prompts the same supervised token ids, even
    though the privileged prompt is longer.
    """
    if not student_prompts:
        return None

    student_prompt_ids = tokenizer(
        list(student_prompts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    teacher_prompt_ids = tokenizer(
        list(teacher_prompts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    completion_ids = tokenizer(
        list(completions),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]

    student_sequences: List[List[int]] = []
    teacher_sequences: List[List[int]] = []
    student_lens: List[int] = []
    teacher_lens: List[int] = []
    target_rows: List[List[int]] = []
    kept_weights: List[float] = []
    for sp_ids, tp_ids, comp_ids, weight in zip(
        student_prompt_ids,
        teacher_prompt_ids,
        completion_ids,
        weights,
    ):
        if float(weight) <= 0:
            continue
        sp = [int(x) for x in sp_ids]
        tp = [int(x) for x in tp_ids]
        cp = [int(x) for x in comp_ids]
        if not sp or not tp or not cp:
            continue
        max_completion_len = min(
            len(cp),
            int(args.max_length) - len(sp),
            int(args.max_length) - len(tp),
        )
        if max_completion_len <= 0:
            continue
        cp = cp[:max_completion_len]
        student_sequences.append(sp + cp)
        teacher_sequences.append(tp + cp)
        student_lens.append(len(sp))
        teacher_lens.append(len(tp))
        target_rows.append(cp)
        kept_weights.append(float(weight))

    kept = len(student_sequences)
    total = len(student_prompts)
    if kept <= 0:
        return None

    pad_token_id = _safe_pad_token_id(tokenizer)
    student_input_ids, student_attention_mask = _pad_id_sequences(
        student_sequences,
        pad_token_id=pad_token_id,
        device=device,
    )
    teacher_input_ids, teacher_attention_mask = _pad_id_sequences(
        teacher_sequences,
        pad_token_id=pad_token_id,
        device=device,
    )
    max_target_len = max(len(row) for row in target_rows)
    target_ids = torch.zeros((kept, max_target_len), dtype=torch.long, device=device)
    token_mask = torch.zeros((kept, max_target_len), dtype=torch.bool, device=device)
    for row_idx, row in enumerate(target_rows):
        row_tensor = torch.tensor(row, dtype=torch.long, device=device)
        target_ids[row_idx, : row_tensor.numel()] = row_tensor
        token_mask[row_idx, : row_tensor.numel()] = True

    return DistillBatchTensors(
        student_input_ids=student_input_ids,
        student_attention_mask=student_attention_mask,
        teacher_input_ids=teacher_input_ids,
        teacher_attention_mask=teacher_attention_mask,
        student_prompt_lens=torch.tensor(student_lens, dtype=torch.long, device=device),
        teacher_prompt_lens=torch.tensor(teacher_lens, dtype=torch.long, device=device),
        target_ids=target_ids,
        token_mask=token_mask,
        weights=torch.tensor(kept_weights, dtype=torch.float32, device=device),
        kept_count=kept,
        total_count=total,
    )


def _extract_completion_logits(
    logits: torch.Tensor,
    prompt_lens: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_completion_len = token_mask.shape
    out = logits.new_zeros((batch_size, max_completion_len, logits.shape[-1]))
    comp_lens = token_mask.sum(dim=1).tolist()
    for row, comp_len_obj in enumerate(comp_lens):
        comp_len = int(comp_len_obj)
        if comp_len <= 0:
            continue
        start = int(prompt_lens[row].item()) - 1
        if start < 0:
            continue
        out[row, :comp_len] = logits[row, start : start + comp_len]
    return out


def _opsd_generalized_jsd_row_loss_and_gap(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = float(getattr(args, "privileged_distill_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("--privileged_distill_temperature must be > 0")
    beta_value = float(getattr(args, "privileged_jsd_beta", -1.0))
    if beta_value < 0:
        beta_value = float(args.beta)
    beta_value = min(max(beta_value, 1e-6), 1.0 - 1e-6)
    beta = torch.tensor(beta_value, device=student_logits.device, dtype=torch.float32)

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-beta),
                teacher_log_probs + torch.log(beta),
            ]
        ),
        dim=0,
    )
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
    pointwise_jsd = beta * kl_teacher + (1.0 - beta) * kl_student
    clip_max = float(getattr(args, "privileged_pointwise_kl_clip", 0.0))
    if clip_max > 0:
        pointwise_jsd = pointwise_jsd.clamp(max=clip_max)

    mask_f = token_mask.to(dtype=pointwise_jsd.dtype)
    token_jsd = pointwise_jsd.sum(dim=-1)
    row_loss = (token_jsd * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)

    safe_targets = target_ids.clamp_min(0)
    student_target_logps = student_log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    teacher_target_logps = teacher_log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    student_avg_logps = (student_target_logps * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
    teacher_avg_logps = (teacher_target_logps * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
    sampled_token_gap = teacher_avg_logps.detach() - student_avg_logps.detach()
    return row_loss, sampled_token_gap, teacher_avg_logps.detach(), student_avg_logps.detach()


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
    total_weight = float(sum(mle_weights) + sum(priv_weights) + sum(gt_priv_weights))
    if total_weight <= 0:
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
            distill_batch = _build_privileged_distill_batch(
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
                with torch.no_grad():
                    teacher_outputs = model(
                        input_ids=distill_batch.teacher_input_ids,
                        attention_mask=distill_batch.teacher_attention_mask,
                        use_cache=False,
                    )
                    teacher_logits = _extract_completion_logits(
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
            student_logits = _extract_completion_logits(
                student_outputs.logits,
                distill_batch.student_prompt_lens,
                distill_batch.token_mask,
            )
            row_loss, raw_gap, teacher_avg_logps, student_avg_logps = _opsd_generalized_jsd_row_loss_and_gap(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                target_ids=distill_batch.target_ids,
                token_mask=distill_batch.token_mask,
                args=args,
            )

            keep_mask = torch.isfinite(row_loss) & torch.isfinite(raw_gap)
            if args.online_pref_min_avg_logprob_chosen is not None:
                keep_mask = keep_mask & (teacher_avg_logps >= float(args.online_pref_min_avg_logprob_chosen))
            if args.online_pref_min_avg_logprob_rejected is not None:
                keep_mask = keep_mask & (student_avg_logps >= float(args.online_pref_min_avg_logprob_rejected))
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
            loss_chunk = (loss_vec * w_kept).sum() / total_weight
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

    def _run_privileged_sampled_branch(
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
            w = torch.tensor(weights[start:end], device=device, dtype=torch.float32)
            was_training = bool(getattr(model, "training", False))
            model.eval()
            try:
                with torch.no_grad():
                    teacher_logps = _compute_sequence_logps_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_texts=pp,
                        completion_texts=wc,
                        max_length=args.max_length,
                        device=device,
                    )
            finally:
                if was_training:
                    model.train()
            student_logps = _compute_sequence_logps_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=tp,
                completion_texts=wc,
                max_length=args.max_length,
                device=device,
            )
            raw_adv = teacher_logps.detach() - student_logps.detach()
            keep_mask = torch.isfinite(teacher_logps) & torch.isfinite(student_logps) & torch.isfinite(raw_adv)
            if args.online_pref_min_avg_logprob_chosen is not None:
                keep_mask = keep_mask & (teacher_logps >= float(args.online_pref_min_avg_logprob_chosen))
            if args.online_pref_min_avg_logprob_rejected is not None:
                keep_mask = keep_mask & (student_logps.detach() >= float(args.online_pref_min_avg_logprob_rejected))
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
            keep_mask = keep_mask.to(device=student_logps.device)
            student_kept = student_logps[keep_mask]
            adv = raw_adv.to(device=student_logps.device)[keep_mask]
            if args.privileged_advantage_clip_abs > 0:
                adv = adv.clamp(
                    -float(args.privileged_advantage_clip_abs),
                    float(args.privileged_advantage_clip_abs),
                )
            w_kept = w[keep_mask]
            loss_vec = -adv * student_kept
            loss_chunk = (loss_vec * w_kept).sum() / total_weight
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
            adv_sum = float((adv.detach() * w_kept).sum().item())
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

    branch_runner = (
        _run_privileged_sampled_branch
        if args.privileged_distill_loss == "sampled_pg"
        else _run_privileged_branch
    )

    failed = branch_runner(
        branch="privileged",
        train_prompts=priv_train_prompts,
        privileged_prompts=priv_prompts,
        wrong_completions=priv_wrong,
        weights=priv_weights,
    )
    if failed is not None:
        return failed

    failed = branch_runner(
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
            logps = _compute_sequence_logps_batch(
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
            loss_chunk = (mle_loss_vec * w).sum() / total_weight
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

    if mle_weight_sum_used + priv_weight_sum_used + gt_priv_weight_sum_used <= 0:
        optimizer.zero_grad(set_to_none=True)
        return _build_zero_stats("all_train_samples_filtered_before_autograd")

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

    used_weight = mle_weight_sum_used + priv_weight_sum_used + gt_priv_weight_sum_used
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


def _rollout_record(
    *,
    sample: DapoSample,
    split: Any,
    objective: Optional[PrivilegedObjective],
    candidates: Sequence[str],
    log_text: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "ground_truth": sample.ground_truth,
        "objective_type": "skip" if objective is None else objective.objective_type,
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
        record["n_privileged_wrong_examples"] = len(objective.privileged_wrong_examples)
        record["privileged_matches"] = [
            {
                "source_type": ex.source_type,
                "match_correct_index": int(ex.match_correct_index),
                "match_similarity": float(ex.match_similarity),
                "wrong_avg_logprob": float(ex.wrong_avg_logprob),
                "weight": float(ex.weight),
            }
            for ex in objective.privileged_wrong_examples
        ]
        if log_text:
            record["rollouts"] = [
                rollout_trajectory_to_json(t, include_dense=bool(False))
                for t in [*objective.correct, *objective.wrong]
            ]
            record["privileged_examples"] = [
                {
                    "source_type": ex.source_type,
                    "source_trace": ex.source_trace,
                    "wrong_response": ex.wrong_response,
                    "privileged_prompt": ex.privileged_prompt,
                    "weight": float(ex.weight),
                }
                for ex in objective.privileged_wrong_examples
            ]
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
    layout, source_iter = _load_source_iter(args)
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
        f"hidden_layer_offset={args.hidden_layer_offset} distill_loss={args.privileged_distill_loss} "
        f"jsd_beta={args.privileged_jsd_beta if args.privileged_jsd_beta >= 0 else args.beta} "
        f"pointwise_kl_clip={args.privileged_pointwise_kl_clip}",
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
            "privileged_distill_loss": str(args.privileged_distill_loss),
            "privileged_jsd_beta": float(args.privileged_jsd_beta if args.privileged_jsd_beta >= 0 else args.beta),
            "privileged_distill_temperature": float(args.privileged_distill_temperature),
            "privileged_pointwise_kl_clip": float(args.privileged_pointwise_kl_clip),
            "privileged_advantage_clip_abs": float(args.privileged_advantage_clip_abs),
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

                objective: Optional[PrivilegedObjective] = None
                trajectories = _build_rollout_trajectories_with_hidden(
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
                    correct_weights = compute_correct_trajectory_weights(
                        correct_trajs=correct_trajs,
                        mode=args.positive_weight_mode,
                        tau=args.positive_weight_tau,
                    )
                    objective = PrivilegedObjective(
                        sample_id=sample_obj.sample_id,
                        ground_truth=sample_obj.ground_truth,
                        train_prompt=prompt_texts[idx],
                        objective_type="all_correct",
                        rho_hat=rho_hat,
                        prompt_weight=prompt_weight,
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
                        correct_trajs=correct_trajs,
                        mode=args.positive_weight_mode,
                        tau=args.positive_weight_tau,
                    )
                    privileged_examples = _build_mixed_privileged_examples(
                        tokenizer=tokenizer,
                        prompt_user_effective=prompt_user_effective[idx],
                        system_prompt=system_prompts[idx],
                        train_prompt=prompt_texts[idx],
                        correct_trajs=correct_trajs,
                        wrong_trajs=wrong_trajs,
                        prompt_weight=prompt_weight,
                        args=args,
                    )
                    objective = PrivilegedObjective(
                        sample_id=sample_obj.sample_id,
                        ground_truth=sample_obj.ground_truth,
                        train_prompt=prompt_texts[idx],
                        objective_type="mixed_privileged",
                        rho_hat=rho_hat,
                        prompt_weight=prompt_weight,
                        correct=correct_trajs,
                        wrong=wrong_trajs,
                        correct_weights=correct_weights,
                        privileged_wrong_examples=privileged_examples,
                    )
                    if correct_trajs or privileged_examples:
                        rollout_objectives.append(objective)
                        mixed_in_rollout += 1
                        logged_mixed += 1
                        priv_examples_in_rollout += len(privileged_examples)
                    else:
                        skipped_after_filter += 1
                        skipped_after_filter_in_rollout += 1
                elif n_correct <= 0 and wrong_trajs:
                    gt_examples = _build_gt_privileged_examples(
                        tokenizer=tokenizer,
                        sample=sample_obj,
                        prompt_user_effective=prompt_user_effective[idx],
                        system_prompt=system_prompts[idx],
                        train_prompt=prompt_texts[idx],
                        wrong_trajs=wrong_trajs,
                        prompt_weight=prompt_weight,
                        args=args,
                    )
                    if gt_examples:
                        objective = PrivilegedObjective(
                            sample_id=sample_obj.sample_id,
                            ground_truth=sample_obj.ground_truth,
                            train_prompt=prompt_texts[idx],
                            objective_type="all_wrong_gt_privileged",
                            rho_hat=rho_hat,
                            prompt_weight=prompt_weight,
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
                f"[privileged_hidden_opsd] rollout_step={rollout_steps}/{total_steps_str} scanned={scanned} "
                f"mixed={mixed_in_rollout} all_correct={all_correct_in_rollout} all_wrong_gt={all_wrong_in_rollout} "
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
                        for traj, traj_weight in zip(obj.correct, obj.correct_weights):
                            mle_train_prompts.append(obj.train_prompt)
                            mle_completions.append(traj.response_text)
                            mle_weights.append(
                                float(args.lambda_mle) * float(obj.prompt_weight) * float(traj_weight)
                            )
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
        "--privileged_distill_loss",
        type=str,
        default="jsd",
        choices=["jsd", "sampled_pg"],
        help=(
            "Privileged branch loss. 'jsd' is OPSD-style full-vocab generalized JSD on the wrong "
            "rollout tokens. 'sampled_pg' keeps the earlier sampled-token logprob-advantage fallback."
        ),
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
        default=0.0,
        help="If >0, clamp each vocab-entry contribution before summing token JSD, OPSD-style.",
    )
    parser.add_argument(
        "--privileged_advantage_clip_abs",
        type=float,
        default=1.0,
        help="Only used by --privileged_distill_loss sampled_pg. Clip teacher-minus-student avg-logprob advantage.",
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
        "--log_rollout_text",
        type=str2bool,
        default=False,
        help="If true, rollout_records.jsonl stores full response and privileged prompt text.",
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
        (
            args.privileged_advantage_clip_abs < 0,
            "error: --privileged_advantage_clip_abs must be >= 0",
        ),
        (args.privileged_trace_max_chars < 0, "error: --privileged_trace_max_chars must be >= 0"),
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
