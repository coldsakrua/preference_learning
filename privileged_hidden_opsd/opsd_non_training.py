#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from privileged_hidden_opsd.opsd_local_utils import (
    DapoSample,
    RolloutTrajectory,
    apply_qwen_chat_template,
    detect_parquet_dataset_layout,
    iter_dapo_samples,
    iter_math_hf_samples,
    rollout_trajectory_to_json,
    strip_prompt_prefix_from_text,
    unwrap_model_for_save,
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
    objective_type: str  # "all_correct" | "mixed_hidden_opsd" | "all_wrong_gt_privileged"
    correct: List[RolloutTrajectory]
    wrong: List[RolloutTrajectory]
    correct_weights: List[float]
    privileged_wrong_examples: List[PrivilegedWrongExample]


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


@dataclass
class CompletionBatchTensors:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lens: torch.Tensor
    target_ids: torch.Tensor
    token_mask: torch.Tensor
    completion_position_mask: torch.Tensor
    kept_indices: List[int]
    total_count: int


def load_source_iter(args: argparse.Namespace):
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


def build_completion_supervision_batch(
    *,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> Optional[CompletionBatchTensors]:
    if not prompt_texts:
        return None

    prompt_ids = tokenizer(
        list(prompt_texts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    completion_ids = tokenizer(
        list(completion_texts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]

    sequences: List[List[int]] = []
    prompt_lens: List[int] = []
    target_rows: List[List[int]] = []
    kept_indices: List[int] = []
    for idx, (sp_ids, comp_ids) in enumerate(zip(prompt_ids, completion_ids)):
        sp = [int(x) for x in sp_ids]
        cp = [int(x) for x in comp_ids]
        if int(max_length) <= 0:
            continue
        if len(sp) > int(max_length):
            sp = sp[: int(max_length)]
        remain = max(0, int(max_length) - len(sp))
        cp = cp[:remain]
        seq = sp + cp
        if not seq:
            continue
        sequences.append(seq)
        prompt_lens.append(len(sp))
        target_rows.append(cp)
        kept_indices.append(int(idx))

    if not sequences:
        return None

    pad_token_id = _safe_pad_token_id(tokenizer)
    input_ids, attention_mask = _pad_id_sequences(
        sequences,
        pad_token_id=pad_token_id,
        device=device,
    )

    max_target_len = max((len(row) for row in target_rows), default=0)
    target_ids = torch.zeros((len(sequences), max_target_len), dtype=torch.long, device=device)
    token_mask = torch.zeros((len(sequences), max_target_len), dtype=torch.bool, device=device)
    completion_position_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    for row_idx, (row, p_len) in enumerate(zip(target_rows, prompt_lens)):
        if row:
            row_tensor = torch.tensor(row, dtype=torch.long, device=device)
            target_ids[row_idx, : row_tensor.numel()] = row_tensor
            token_mask[row_idx, : row_tensor.numel()] = True
            completion_position_mask[row_idx, p_len : p_len + row_tensor.numel()] = True

    return CompletionBatchTensors(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.long, device=device),
        target_ids=target_ids,
        token_mask=token_mask,
        completion_position_mask=completion_position_mask,
        kept_indices=kept_indices,
        total_count=len(prompt_texts),
    )


def average_masked(values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask_f = token_mask.to(dtype=values.dtype)
    denom = mask_f.sum(dim=-1).clamp_min(1.0)
    averaged = (values * mask_f).sum(dim=-1) / denom
    return torch.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0)


def completion_target_logps(
    *,
    completion_logits: torch.Tensor,
    target_ids: torch.Tensor,
    logit_clip_abs: float,
) -> torch.Tensor:
    logits_f = torch.nan_to_num(
        completion_logits.float(),
        nan=0.0,
        posinf=logit_clip_abs if logit_clip_abs > 0 else 80.0,
        neginf=-(logit_clip_abs if logit_clip_abs > 0 else 80.0),
    )
    if logit_clip_abs > 0:
        logits_f = logits_f.clamp(-logit_clip_abs, logit_clip_abs)
    target_logps = F.log_softmax(logits_f, dim=-1).gather(
        dim=-1,
        index=target_ids.clamp_min(0).unsqueeze(-1),
    ).squeeze(-1)
    target_logps = torch.nan_to_num(target_logps, nan=-20.0, posinf=0.0, neginf=-20.0)
    return target_logps.clamp(min=-20.0, max=0.0)


def extract_completion_logits(
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


def compute_sequence_logps_batch_local(
    *,
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    out = torch.full((len(prompt_texts),), float("nan"), dtype=torch.float32, device=device)
    batch = build_completion_supervision_batch(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        completion_texts=completion_texts,
        max_length=max_length,
        device=device,
    )
    if batch is None:
        return out

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    completion_logits = extract_completion_logits(
        outputs.logits,
        batch.prompt_lens,
        batch.token_mask,
    )
    target_logps = completion_target_logps(
        completion_logits=completion_logits,
        target_ids=batch.target_ids,
        logit_clip_abs=80.0,
    )
    seq_logps = average_masked(target_logps, batch.token_mask)
    has_tokens = batch.token_mask.any(dim=1)
    seq_logps = torch.where(has_tokens, seq_logps, torch.full_like(seq_logps, float("nan")))
    index = torch.tensor(batch.kept_indices, dtype=torch.long, device=device)
    out[index] = seq_logps
    return out


def compute_sequence_logps_and_hidden_batch_local(
    *,
    model: object,
    tokenizer: object,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    max_length: int,
    device: torch.device,
    hidden_layer_offset: int,
    compute_entropy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_logps = torch.full((len(prompt_texts),), float("nan"), dtype=torch.float32, device=device)
    out_entropy = torch.full((len(prompt_texts),), float("nan"), dtype=torch.float32, device=device)

    batch = build_completion_supervision_batch(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        completion_texts=completion_texts,
        max_length=max_length,
        device=device,
    )
    if batch is None:
        hidden_size = int(getattr(getattr(unwrap_model_for_save(model), "config", object()), "hidden_size", 1))
        out_hidden = torch.zeros((len(prompt_texts), hidden_size), dtype=torch.float32, device=device)
        return out_logps, out_entropy, out_hidden

    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    completion_logits = extract_completion_logits(
        outputs.logits,
        batch.prompt_lens,
        batch.token_mask,
    )
    target_logps = completion_target_logps(
        completion_logits=completion_logits,
        target_ids=batch.target_ids,
        logit_clip_abs=80.0,
    )
    seq_logps = average_masked(target_logps, batch.token_mask)
    has_tokens = batch.token_mask.any(dim=1)
    seq_logps = torch.where(has_tokens, seq_logps, torch.full_like(seq_logps, float("nan")))

    if compute_entropy:
        logits_f = torch.nan_to_num(completion_logits.float(), nan=0.0, posinf=80.0, neginf=-80.0).clamp(-80.0, 80.0)
        token_log_probs = F.log_softmax(logits_f, dim=-1)
        token_log_probs = torch.nan_to_num(token_log_probs, nan=-20.0, posinf=0.0, neginf=-20.0).clamp_min(-20.0)
        token_probs = token_log_probs.exp()
        token_entropy = -(token_probs * token_log_probs).sum(dim=-1)
        seq_entropy = average_masked(token_entropy, batch.token_mask)
        seq_entropy = torch.where(has_tokens, seq_entropy, torch.full_like(seq_entropy, float("nan")))
    else:
        seq_entropy = torch.full_like(seq_logps, float("nan"))

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("Model did not return hidden_states; cannot run hidden-state pair mining.")
    layer_idx = len(hidden_states) - int(hidden_layer_offset)
    layer_idx = max(0, min(layer_idx, len(hidden_states) - 1))
    layer_hidden = hidden_states[layer_idx]
    completion_mask = batch.completion_position_mask.to(dtype=layer_hidden.dtype)
    denom = completion_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (layer_hidden * completion_mask.unsqueeze(-1)).sum(dim=1) / denom
    hidden_vec = F.normalize(torch.nan_to_num(pooled.float(), nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1, eps=1e-12)

    out_hidden = torch.zeros((len(prompt_texts), hidden_vec.shape[-1]), dtype=torch.float32, device=device)
    index = torch.tensor(batch.kept_indices, dtype=torch.long, device=device)
    out_logps[index] = seq_logps
    out_entropy[index] = seq_entropy
    out_hidden[index] = hidden_vec
    return out_logps, out_entropy, out_hidden


def build_rollout_trajectories_with_hidden(
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
            seq_logps, seq_entropy, batch_hidden = compute_sequence_logps_and_hidden_batch_local(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts[start:end],
                completion_texts=list(candidates[start:end]),
                max_length=args.max_length,
                device=device,
                hidden_layer_offset=args.hidden_layer_offset,
                compute_entropy=bool(getattr(args, "rollout_compute_entropy", True)),
            )
            for v in seq_logps.detach().cpu().tolist():
                fv = float(v)
                avg_logprobs.append(fv if math.isfinite(fv) else -20.0)
            for v in seq_entropy.detach().cpu().tolist():
                fv = float(v)
                avg_entropies.append(fv if math.isfinite(fv) else 0.0)
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


def build_mixed_privileged_examples(
    *,
    tokenizer: object,
    prompt_user_effective: str,
    system_prompt: str,
    train_prompt: str,
    correct_trajs: Sequence[RolloutTrajectory],
    wrong_trajs: Sequence[RolloutTrajectory],
    args: argparse.Namespace,
) -> List[PrivilegedWrongExample]:
    if not correct_trajs or not wrong_trajs or args.lambda_priv <= 0:
        return []
    per_wrong_weight = float(args.lambda_priv) / float(len(wrong_trajs))
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


def build_gt_privileged_examples(
    *,
    tokenizer: object,
    sample: DapoSample,
    prompt_user_effective: str,
    system_prompt: str,
    train_prompt: str,
    wrong_trajs: Sequence[RolloutTrajectory],
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
    per_wrong_weight = float(args.lambda_gt) / float(len(wrong_trajs))
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


def build_privileged_distill_batch(
    *,
    tokenizer: object,
    student_prompts: Sequence[str],
    teacher_prompts: Sequence[str],
    completions: Sequence[str],
    weights: Sequence[float],
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[DistillBatchTensors]:
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


def opsd_generalized_jsd_row_loss_and_gap(
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

    logit_clip_abs = float(getattr(args, "privileged_logit_clip_abs", 80.0))
    student_logits_f = torch.nan_to_num(
        student_logits.float(),
        nan=0.0,
        posinf=logit_clip_abs,
        neginf=-logit_clip_abs,
    )
    teacher_logits_f = torch.nan_to_num(
        teacher_logits.float(),
        nan=0.0,
        posinf=logit_clip_abs,
        neginf=-logit_clip_abs,
    )
    if logit_clip_abs > 0:
        student_logits_f = student_logits_f.clamp(-logit_clip_abs, logit_clip_abs)
        teacher_logits_f = teacher_logits_f.clamp(-logit_clip_abs, logit_clip_abs)

    student_log_probs = F.log_softmax(student_logits_f / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_f / temperature, dim=-1)
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
    pointwise_jsd = torch.nan_to_num(pointwise_jsd, nan=0.0, posinf=0.0, neginf=0.0)
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


def build_rollout_record(
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
