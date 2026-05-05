# Privileged Hidden OPSD

This folder contains a local experiment for hidden-state-routed privileged
self-distillation.

The intended behavior is:

1. Sample `rollout_n` on-policy responses from the normal student prompt.
2. Grade the responses.
3. If all responses are correct, train with MLE on the correct trajectories.
4. If a prompt has both correct and wrong responses, still MLE-train the correct
   trajectories. For each wrong trajectory, choose the correct trajectory with
   the nearest pooled hidden state and use that correct trajectory as private
   teacher context.
5. If all responses are wrong, use the dataset ground-truth reasoning trace as
   private teacher context.
6. Batch all privileged teacher-context scoring passes before the update. This
   avoids one extra model I/O round trip per wrong trajectory.

The privileged objective follows the OPSD-style distribution matching view:

- student context: original problem only
- teacher context: original problem plus private reference trace
- supervised tokens: the wrong on-policy rollout tokens

For each wrong trajectory, the script runs teacher and student forward passes on
the same wrong rollout token ids, then minimizes a full-vocab generalized JSD on
those token positions. `sampled_pg` is kept as a fallback loss, but the default is
`jsd`.

Run the 4B single-GPU script from the repo root:

```bash
sbatch privileged_hidden_opsd/train_privileged_hidden_opsd_qwen3_4b_1gpu.sh
```

Important environment knobs:

- `LAMBDA_MLE`: MLE weight for correct trajectories.
- `LAMBDA_PRIV`: mixed-prompt privileged wrong-trajectory weight.
- `LAMBDA_GT`: all-wrong GT-privileged wrong-trajectory weight.
- `PRIVILEGED_DISTILL_LOSS`: `jsd` by default, or `sampled_pg` fallback.
- `PRIVILEGED_JSD_BETA`: beta for generalized JSD. `-1` reuses `BETA`.
- `PRIVILEGED_POINTWISE_KL_CLIP`: OPSD-style per-entry clip before summing token
  JSD. Default `0.05`.
- `PRIVILEGED_LOGIT_CLIP_ABS`: fp32 logit clip inside privileged JSD for
  numerical stability. Default `80.0`.
- `PRIVILEGED_ADVANTAGE_CLIP_ABS`: only used by `sampled_pg`.
- `LOGPROB_MICRO_BATCH_SIZE`: chunks batched privileged scoring and MLE forwards.
- `ROLLOUT_FEATURE_MICRO_BATCH_SIZE`: chunks rollout feature extraction
  including hidden states.
