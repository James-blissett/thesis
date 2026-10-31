# B label-free combiner handoff

**Handoff version 1 — 19 Oct 2026.** If the copy in the repo doesn't say version 1 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box.*
*Builds on `abc-implementation-handoff.md` (version 10) and the Stage 5 run. Everything not listed under "Changed for this run" is unchanged from that handoff.*

---

## 0. Working rules (unchanged)

- **No git commands that write.** Propose a commit message at the checkpoint; James commits.
- **Don't run long jobs yourself.** Anything over about a minute: print the launch command for James to run in tmux.
- **Always run from the repo root** after `source env.sh`.
- **No model loading and no forward passes.** Post-hoc on stored data and the feature cache.
- **Don't edit existing files.** `analysis/abc/` and everything older are frozen. New code goes in `analysis/b_label_free/`, results in `results/b_label_free/`. Import from `analysis/abc/` where it helps.
- **Stop at the checkpoint** (⏸) and report.
- **Label convention:** 1 = failure, 0 = success. Outcome is `success_ever`.

---

## 1. Purpose

B's 124 grid probes are trained without outcome labels, but B's final detector (the combiner) is a logistic regression trained on outcome. This run asks: **does B still detect failure if the combiner is replaced by a fixed rule with no trained parameters?**

If yes, B's ranking needs no outcome-trained component at all.

---

## 2. Changed for this run

### 2.1 The label-free combiners (pre-registered; no selection between them)

At each kept row, take the grid's predicted probabilities at one H (the refit-on-all-train probes, exactly as Stage 5 used for eval-seen and unseen).

| Name | Rule | Inputs |
|---|---|---|
| **LF-all** (primary) | unweighted mean of the probabilities | all grid probes used in Stage 5 (124) |
| **LF-emb** (secondary) | unweighted mean of the probabilities | the `emb_temp[ℓ]` probes only (30) |

- No fitted weights, no scaler, no `c`, no choice made on eval-seen.
- Score over time: **running sum** of the per-row value, `s_t = Σ_{τ ≤ t} m_τ`, as for every trained model in A, B and C.
- Report **both** variants at **all three H** (0, 10, 25). Do not pick a best H or variant: every row goes in the table.

### 2.2 Probe outputs

- First check whether Stage 5 saved per-row grid probe outputs for eval-seen and unseen (search `results/abc/` and `setup_b.py`). If yes, reuse them.
- If not, refit the grid exactly as Stage 5 did (same splits from `results/abc/splits/`, same target preparation, same `c`, balanced class weights, GPU fit). **Only the refit on all of train is needed**: the label-free combiners are never trained, so the 5-fold out-of-fold pass is skipped.
- **Sanity check before scoring:** the refitted probes' own-target AUROC on unseen must match `results/abc/b_grid.csv` within 0.005 for every (layer, target, H, seed). If not, stop and report.

### 2.3 Evaluation (same as Stage 5 and 6)

For each (variant, H, seed):
1. M1-per-task on eval-seen and unseen (the number claims rest on), pooled M1, and the timestep-only floor.
2. Cut-variant conformal prediction at α = 0.1 (the version that stops at each task's shortest success, as reported in Stage 6): balanced accuracy, share of failures caught, false-alarm rate, mean detection time. Calibrated on successful eval-seen rollouts, as before.
3. Mean score against `t` on unseen failures (length check), one line per variant at each H.

### 2.4 Comparison

- Same table rows for B's trained combiner (from `results/abc/b_final.csv` and Stage 6 outputs). Do not recompute them.
- Paired difference per seed, LF-all minus trained B, for unseen M1-per-task, at each H. Report mean ± std. With 3 seeds, state that this cannot establish a difference; it can only show whether the two are level.

---

## 3. Outcome information still used (report this verbatim in the run log)

Neither variant trains anything on outcome, but two preparation steps upstream of the probes used outcome on train/eval-seen:
- **Orientation:** each constraint's sign was set by its train-split M1-per-task (one bit per constraint).
- **Smoothing β:** chosen per constraint family on eval-seen M1 in Stage 2 (`emb_temp`: none).

The conformal band is calibrated on successful rollouts, which needs to know which rollouts succeeded.

---

## 4. Outputs

- `results/b_label_free/lf_final.csv`: rows = (variant, H); columns as `results/abc/b_final.csv` (mean, std, per-seed values) plus the α = 0.1 alarm columns.
- `results/b_label_free/lf_vs_trained.csv`: the paired differences.
- `results/b_label_free/score_vs_t.png`
- `results/b_label_free/scores/lf_{variant}_h{H}_seed{s}.npz`
- `results/b_label_free/RUNLOG.md`: any deviation from this document, with the reason.

## 5. Checkpoint ⏸

Report: whether probe outputs were reused or refitted (and the sanity check), the full `lf_final.csv` alongside trained B, the paired differences, and the length-check figure. Propose a commit message. Stop.
