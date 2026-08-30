# Task: compute consistency constraints and AUROC each against outcome

Pure post-hoc analysis on the stored corpus. No model loading, no forward passes,
no git commands. Surface launch commands as copyable blocks; do not run long jobs.

## 0. Inspect and verify hidden-state consistency
Read `probe_handoff.md` and `collection_handoff.md`. Confirm the on-disk layout:
per-rollout hidden states `[T, 33, 4096]` fp16, per-rollout actions `[T, 7]`
(de-tokenised continuous: xyz, rotation, gripper), task_id, outcome.
Print shapes for one rollout before writing anything.

Definition used throughout: h^ℓ_t for ℓ = 1..32 is the raw residual stream at
the output of decoder block ℓ, i.e. BEFORE `model.norm` and BEFORE `lm_head`,
read at the 7 action-token positions and mean-pooled. ℓ = 0 is the embedding
output. All reads are pre-action-decoding; no layer is post-norm or post-logit.

Verify index 32 satisfies this. Check the capture code: if it used forward
hooks on `model.language_model.model.layers[i]`, index 32 is raw — fine. If it
used `output_hidden_states=True`, the last tuple entry is post-`model.norm`
and is NOT consistent with 1..31. Empirical check regardless: print the
per-timestep L2 norm distribution of index 31 vs 32 for one rollout. Raw
residual norms grow smoothly with depth; a post-norm read has a norm that is
near-constant across timesteps and off the depth trend.

If index 32 is post-norm: do not recapture. Anchor `xl_final` to layer 31
instead, exclude index 32 from `xl_adj` and `emb_temp`, and record this in the
checkpoint report so it propagates to the writeup. Stop and report before
proceeding to step 1.

## 1. Constraints (one scalar per timestep per rollout)
Compute in fp32. Timestep 0 of any temporal constraint is NaN and excluded.

a) Action temporal, split into two series (do not combine):
   - `act_mag[t]  = ||a_t[0:6] - a_{t-1}[0:6]||_2`   (exclude gripper)
   - `act_dir[t]  = 1 - cos(Δa_t[0:6], Δa_{t-1}[0:6])`, Δa_t = a_t - a_{t-1};
     NaN if either delta norm < 1e-6.
   - `grip_flip[t] = 1[sign(a_t[6]) != sign(a_{t-1}[6])]` reported separately.

b) Embedding temporal, per layer L in 0..32:
   - `emb_temp[L][t] = 1 - cos(h_t^L, h_{t-1}^L)`  → 33 series.

c) Cross-layer, per timestep, cosine distance (not L2 — residual norms grow
   with depth):
   - adjacent: `xl_adj[ℓ][t] = 1 - cos(h_t^ℓ, h_t^{ℓ+1})` for ℓ in 0..31
   - anchored: `xl_final[ℓ][t] = 1 - cos(h_t^ℓ, h_t^32)` for ℓ in 0..31
   - summary: `xl_spread[t] = mean over ℓ of xl_adj[ℓ][t]`

Save all series to `constraints/<rollout_id>.npz` plus a long-form parquet
`constraints/all.parquet` with columns
[rollout_id, task_id, outcome, t, T, t_norm, constraint_name, value].

## 2. AUROC each constraint independently
Use the existing evaluation utilities where they exist. For every constraint
series, report on the matched 40-rollout non-degenerate corpus (tasks 2 and 5
dropped), and also on the full 300:
   - Scheme A: all timesteps, broadcast outcome label
   - Scheme B: t/T >= 0.8 only
   - Rollout-level: aggregate each series to one scalar per rollout by
     {max, mean} over the Scheme-B window, AUROC against outcome.
Convention: report `max(auc, 1-auc)` and the sign (+ if higher value ⇒ failure,
− if anti-correlated). No training — these are raw scores ranked directly.
Baselines in the same table: timestep index `t` alone, and `t_norm` alone.
Permutation nulls: 1000 label shuffles, both global and within-task, report
null mean and 95th percentile.
Output `results/constraint_auroc.csv` with columns
[constraint, layer, scheme, corpus, n_rollouts, auroc, sign, null_global_mean,
null_global_p95, null_task_mean, null_task_p95]
and a one-line-per-layer plot for emb_temp and xl_adj / xl_final (AUROC vs layer).

## 3. Checkpoints
Stop and report after step 0 (shapes + index-32 verification), after step 1
(summary stats of each series; sanity: act_mag should be O(1e-2..1e-1) in
normalised action units), and after step 2. Propose a commit message at each
checkpoint; I will commit.