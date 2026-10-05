# A / B / C implementation handoff

**Handoff version 7 — 5 Oct 2026, 17:40 AEDT.** If the copy in the repo doesn't say version 7 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box.*
*Method and reasoning: `label-alignment-and-probe-protocol.md` (spec v2.5). This document says how to build it. Where the two disagree, this one wins, because it was written against the repo (commit `3c66a27`, 5 Oct 2026).*

---

## 0. Working rules

- **No git commands that write.** Never run `git add`, `git commit` or `git push`. Propose a commit message at each checkpoint; James commits.
- **Don't run long jobs yourself.** Anything over about a minute: print the launch command as a copyable block for James to run in tmux.
- **Always run from the repo root** after `source env.sh`. Results paths are relative to the working directory.
- **No model loading and no forward passes.** Everything here is post-hoc on stored data.
- **Don't edit existing files.** `collect/` and the current `analysis/*.py` are frozen. New code goes in `analysis/abc/`. Import from the existing modules where it helps.
- **Stop at every checkpoint** (marked ⏸) and report before going on.
- **Label convention, everywhere:** 1 = failure, 0 = success, matching `probe_layer.py`. Outcome is `success_ever` (137 successes, 163 failures).

---

## 1. What exists already (verified in the repo)

**Corpus:** `/data/rollouts_v2/<rollout_id>/`, 300 rollouts, 30 per task, 10 tasks, 147 GB.

| File | Contents |
|---|---|
| `hidden.pt` | `(520, 33, P, 4096)` fp16. P = 3 for 270 rollouts: position 0 is the mean over the 7 decision states. P = 7 for 30 rollouts (trials 0–2 of each task): take the mean yourself. |
| `actions.npz` | `action_token_ids` `(520, 7)` int32, plus `executed`, `action_raw`, `success_now` (the last three have 530 rows: 10 wait steps first). |
| `logits.pt` | `(520, 7, 256)` fp16, bins in ascending action value. |
| `frames/` | JPEGs of every env step (useful only for the optional onset labelling). |

**Index:** `corpus_v2_index.json` → per rollout: `rollout_id`, `task_idx`, `success_ever`, `success_final`, `t_success` (in policy steps, i.e. hidden-state rows; `null` for failures).

**Constraints, already computed:** `constraints/<rollout_id>.npz`, all `(520, …)` float32, NaN at `t = 0` for temporal ones:

| Key | Shape | Meaning |
|---|---|---|
| `act_mag` | `(520,)` | size of the change in action (6 dims, gripper excluded) |
| `act_dir` | `(520,)` | change in direction of the action delta; NaN also where a delta is ~0 |
| `grip_flip` | `(520,)` | gripper sign changed |
| `emb_temp` | `(520, 32)` | per layer 0–31: cosine distance between this step's and the previous step's hidden state |
| `xl_adj` | `(520, 31)` | per ℓ 0–30: cosine distance between layers ℓ and ℓ+1 at the same step |
| `xl_final` | `(520, 31)` | per ℓ 0–30: cosine distance between layer ℓ and layer 31 |
| `xl_spread` | `(520,)` | mean of `xl_adj` over layers |
| `xl_final_hN` | `(520, 32)` | single-position variant anchored to block 32's direction |
| `*_nosink` | same | the same series with dimension 1512 dropped |

`act_rep` isn't stored. Derive it as in `constraint_auroc.add_derived`: `act_mag <= ACT_REP_EPS`.

**Facts that change the spec's numbers:**
- **Index 32 is post-norm, not raw.** Raw layers are 0–31. All hidden-state constraints cover 0–31 only. So C has 32 per-layer probes, not 33, and B's grid has 32 layers.
- **The stored hidden state already includes the last prompt token's state** (it's the first of the 7 decision states). That closes the spec's open item on token positions.
- **The spec's `hl_*` names are `xl_*` here.** Use the repo's names.

**Existing code to reuse, not rewrite:**
- `compute_constraints.pooled_and_hN`: the correct way to read a rollout's pooled hidden states for both P = 3 and P = 7.
- `compute_constraints.normalised_actions`: token IDs → normalised actions.
- `control_diagnostic.global_permutation`, `within_task_permutation`: rollout-level label shuffles.

`probe_layer.load_features` reads the old 50-rollout corpus (`/data/corpus`, 8 positions). Don't use it for this work.

**Existing work to be aware of, not to build on:**
- `analysis/smooth_constraints.py` added `act_mag_w{w}`, `act_rep_w{w}`, `act_dir_w{w}` (w = 5, 11, 21, 51) to `constraints/all.parquet`. These are **centred** rolling means, so each value uses future timesteps. Don't use them as inputs or targets here; the smoothing in §2 is causal. Read constraints from the per-rollout `.npz` files, not the parquet.
- `analysis/completion_confound.py` verified the time base: `t_success` is a policy-step index with no offset, the same `t` as every constraint series. Its pre/post numbers (38,618 rows before success at 0.168 repeat rate; 32,485 after at 0.476) are the reason for the cut.

---

## 2. Definitions to implement exactly

**Kept rows.** Successes: `t < t_success`. Failures: all 520. Drop `t = 0` everywhere (temporal constraints are NaN there). Build one boolean mask `(300, 520)` and use it everywhere. (The step at `t_success` itself is still in progress, since success is checked after its action executes. It's excluded anyway, to match `completion_confound.py`.)

**Splits, per seed `s ∈ {0, 1, 2}`.**
- Unseen tasks: 3 of the 10, drawn with `np.random.default_rng(s)`.
- Within the 7 seen tasks: split whole rollouts 60/40 into train / eval-seen, stratified by task and outcome.
- Save each split to `results/abc/splits/seed{s}.json` and load it from there afterwards, so every stage uses identical splits.

**Constraint preparation** (fitted on train rows only, applied to all rows; redo per seed):
1. Replace NaN with the rollout's previous valid value (forward fill). Rows before a series' first valid value stay missing: exclude them from M1's maximum, and give them the neutral value (0 after scaling) as model inputs. A fill value must never be able to become a rollout's maximum. *(Revised after Stage 2: the earlier zero-fill for `act_dir` made every rollout tie at 0.500.)*
2. Smooth: exponential moving average, `E[t] = β·E[t−1] + (1−β)·x[t]`, started at the first valid step. β ∈ {none, 0.83, 0.91, 0.96}, where "none" means no smoothing. *(Stage 2 showed `emb_temp` is best unsmoothed.)*
3. Scale: subtract the mean and divide by the standard deviation of *successful train kept rows*.
4. Orient: compute the series' **M1-per-task on the train split** (the within-task ROC-AUC averaged over train tasks); multiply by −1 if it's below 0.5. Save the sign per constraint and per seed. *(Revised twice after Stage 2: row means disagreed with M1's max-over-time, and pooled train M1 carried the cross-task effect, which flipped `act_mag` in seed 2.)*
5. Sign stability: a constraint whose sign differs between seeds is marked **unstable** in every table. Unstable constraints are still fed to C's probes and final detector (those learn their own signs), but are not used as targets in B.

**Choosing β:** one β per constraint family (action; `emb_temp`; cross-layer), each chosen by that family's best eval-seen M1 in Stage 2's table. Fix those three for everything else and report the full grid.

**Use the primary (with-sink) series by default.** Report the `_nosink` variants as a side table in Stage 2 only.

**The probe.** `StandardScaler` → `LogisticRegression(C=c, max_iter=2000)`, as in `probe_layer.py`, with `c` chosen from {1, 0.1, 0.01, 0.001} on eval-seen M1-per-task (SAFE App. B.2 picks its L2 strength by grid search the same way). One `c` per setup for the probes and one for the final detector, not one per probe. Report the full grid. For B's grid, use a GPU implementation instead (§4, Stage 5).

**Row weights for anything trained on outcome** (A's probes, C's probes, all final detectors): weight each row by `1 / n_class(t)`, where `n_class(t)` is the number of train rollouts of that row's class still kept at step `t`. Only train on timesteps where **at least 5 train successes are still kept**; drop later rows. Rescale the weights to average 1 over the remaining train rows, so `C = 1.0` means what it does in `probe_layer.py`. Print per seed: the cut-off `t`, max weight ÷ mean weight, the share of total weight in the top 1% of rows, and the effective sample size (Σw)² ÷ Σw².

**C's per-layer probe inputs.** For layer ℓ in 0–31: `emb_temp[ℓ]`, `xl_adj[min(ℓ, 30)]`, `xl_final[min(ℓ, 30)]`. Action probe: `act_mag`, `act_dir`, `grip_flip`, `act_rep`.

**C's final detector inputs.** All prepared constraint series at once: `emb_temp` (32), `xl_adj` (31), `xl_final` (31), `xl_spread`, `act_mag`, `act_dir`, `grip_flip`, `act_rep`. For `emb_temp` and `act_mag`, also include the absolute value of the scaled score.

**A's inputs.** One probe per stored index 0–32 on the pooled hidden state (flag index 32 as post-norm in every table), plus one on `action_token_ids` cast to float.

**B's grid.** For each layer ℓ in 0–31 and each of four targets (`act_rep`, `act_mag`, `emb_temp[ℓ]`, `xl_final[min(ℓ, 30)]`). *(`xl_final` replaces `xl_adj` as the cross-layer target: Stage 2 found `xl_adj` flips sign between seeds in 5–9 layers. Stage 2 also found the cross-layer family is near chance per task, so treat this target as an expected-null control. Skip any (layer, target) whose target is marked unstable.)*
- Take the prepared target series.
- Look ahead: `y[t] = max(target[t : t+H+1])`, never reading past the rollout's last kept row. H ∈ {0, 10, 25}.
- Binarise at the split point that minimises within-group squared error of the train values (two groups, low and high).
- Input: layer ℓ's pooled hidden state at step `t`. No outcome labels, no row weights; use balanced class weights.

**Scoring over time (version 7; applies to every trained model in A, B and C).** A model's score at step `t` is the **running sum of its predicted failure probabilities** up to `t`: `s_t = Σ_{τ ≤ t} p_τ`. This is SAFE-MLP's score (SAFE §4.2). Stage 3 showed why: models are fitted on single rows, but M1 took each rollout's maximum, which rewards one outlier row; the running sum rewards the same thing the fit does. With a running sum, M1's "maximum over kept rows" is just the value at the cut. Training is unchanged (still logistic regression on rows). The training-free single constraints in Stage 2 keep their maximum.

**Selecting anything on eval-seen** (β, `c`, A's best probe) now uses **M1-per-task**, not pooled M1. A running sum grows with rollout length, so pooled M1 picks up more of the cross-task effect; still report it, with the floor recomputed for a running sum of a constant.

**B's final detector.** Inputs are the grid's predicted probabilities at one H (128 numbers per row). It must be trained on **out-of-fold** probe outputs: within train, 5-fold by rollout (`GroupKFold`), each fold's probes predicting the held-out fold. Then refit the probes on all of train to produce outputs for eval-seen and unseen.

**M1 (trajectory ROC-AUC, SAFE's protocol).** For an evaluation set: for each task, `T_task` = the shortest kept length among that set's rollouts in that task (520 if it has no successes). Score each rollout by its maximum score over kept rows with `t < T_task`. One ROC-AUC over all rollouts in the set. Report for eval-seen and unseen separately.

**M1 is not immune to task differences.** Within a task every rollout is cut at the same length, but the single ROC-AUC spans tasks, and tasks differ in both cut length and failure rate. Stage 2 measured a timestep-only score at 0.610 (eval-seen) and 0.547 (unseen). So:
- **Report the timestep-only M1 as a floor** in every table that reports M1.
- **Also report M1-per-task, and treat it as the number claims rest on:** the ROC-AUC computed within each task (same cut, same max-over-time), averaged over tasks that have at least 2 rollouts of each class in the set. State how many tasks were averaged. A timestep-only score gets exactly 0.5 on this by construction; check that it does.

**M4 (conformal prediction).** See Stage 6.

---

## 3. Compute plan

- **Feature cache first.** One pass over the corpus (about 10 minutes; the disk is the limit, as `compute_constraints.py` found). Write one fp16 memmap per stored index: `/data/tmp/abc_features/layer{L:02d}.npy`, shape `(300, 520, 4096)`, about 1.3 GB each, 42 GB total. Also `action_tokens.npy` `(300, 520, 7)`. Check free space on `/data/tmp` before starting.
- **A and C** use scikit-learn on CPU; each fit is small.
- **B is the heavy part, and runs on the GPU (confirmed with James):** 32 layers × 4 targets × 3 values of H × 3 seeds × 6 fits (5 folds + 1 refit). Fit all 12 targets for one layer in one go on the GPU: a single linear layer with 12 outputs, L2 penalty matching `C = 1.0`, standardised inputs, LBFGS. Before trusting it, check it against scikit-learn on one (layer, target): test AUROC within 0.005.

---

## 4. Stages

Each stage is one script in `analysis/abc/`, writing to `results/abc/`.

**Stage 0 — audit.** ⏸
- Confirm `/data/rollouts_v2` exists with 300 complete rollouts (James reports it does). **If it's gone, stop and say so:** Stages 2 and 3 can still run from `constraints/` alone, but A and B cannot.
- Print shapes for one P = 3 and one P = 7 rollout, the kept-row count per class, and `git status`.
- Look for SAFE's conformal prediction code in the OpenVLA fork (`/ephemeral/code/openvla`): search for "conformal". Report the file and function, or that it isn't there.

**Stage 1 — data module and cache.** ⏸
- `analysis/abc/data.py`: index loading, kept mask, splits, labels, weights.
- `analysis/abc/build_cache.py`: the feature cache.
- Checks: cached layer 15 for one rollout equals `pooled_and_hN` output exactly; kept-row totals after dropping `t = 0` are 38,618 success rows (matching `results/completion_confound.json`) and 163 × 519 = 84,597 failure rows; no rollout appears in two splits.

**Stage 2 — constraints on their own.** ⏸
- Prepare the constraints (§2) for each seed and β.
- Table: every constraint's M1 on eval-seen and unseen, with its sign, mean ± std over seeds. Include the `_nosink` side table.
- Baselines in the same table: token max-probability and mean entropy from `logits.pt`; a timestep-only score.
- FailureSpot's signal: `s = c + EMA(c) + r + EMA(r)` with `c = ‖a_t − a_{t−1}‖₂`, `r = ‖a_t‖₂` over all 7 normalised dims, β = 0.8. Report its M1 and the M1 of `−s`.
- Output: `results/abc/constraint_m1.csv`.

**Stage 3 — C.** ⏸
- 32 per-layer probes + 1 action probe, then the final detector, then the three leave-one-family-out ablations.
- Choose β here.
- Outputs: `c_probes.csv` (M1 per probe), `c_final.csv`, `c_final_weights.csv` (one weight per input), per-row scores saved as `.npz` for Stage 6.

**Stage 4 — A.** ⏸
- 34 probes. Final detector = the probe with the best eval-seen M1-per-task, reported on unseen.
- Shuffled-label control per layer (labels permuted across rollouts); report real minus shuffled.
- Sanity checks that must hold: layer 0 scores below the best layer; shuffled-label M1 within 0.40–0.60.
- Figure: A's and C's per-layer M1 against layer, on one axis.
- Outputs: `a_probes.csv`, `a_vs_c_by_layer.png`, scores `.npz`.

**Stage 5 — B.** ⏸
- The grid, then the final detector, for each H.
- Report per probe: AUROC against its own constraint target on unseen tasks (this is the layer × constraint map), and the final detector's M1.
- Ablation: final detector on each target's 32 probes alone.
- Outputs: `b_grid.csv`, `b_map_H{H}.png` (heatmap, layers × targets), `b_final.csv`, scores `.npz`.

**Stage 6 — conformal prediction and final tables.** ⏸
- Apply SAFE's functional CP to the final detector scores of A, B (each H) and C. Calibrate on successful eval-seen rollouts, evaluate on unseen. Use SAFE's code if Stage 0 found it; otherwise implement the one-sided band from Xu et al. (FAIL-Detect), Appendix B, adaptive modulation. Past the end of the calibration successes, hold the band at its last value unless SAFE's code does otherwise; report which.
- α: 15 levels from 0.02 to 0.9.
- Per α: balanced accuracy and detection time (first alarm step ÷ rollout kept length, 1 if never; mean over failed rollouts).
- Length check: mean final-detector score against `t` on unseen failures, one line per setup.
- Outputs: `final_table.csv` (rows A, C, B at each H; columns M1 seen, M1 unseen, balanced accuracy and detection time at α = 0.1; mean ± std over seeds), `m4_curves.png`, `score_vs_t.png`.

**Stage 7 — MLP ablation (only if asked).** SAFE-MLP's shape (2 hidden layers of 256) as the final detector for C and B.

---

## 5. Known data quirks

- `task05_trial27` has one physically implausible step (a simulator instability). It's kept; note it in reports.
- 28 successes later lose the success state (`success_final` false). They're successes here and are cut at `t_success` like the rest.
- Success counts per task run from 5 (task 3) to 24 (task 1), so some task splits will be unbalanced. Report class counts per split.
- The shortest success per task (the M1 cut when the whole task is in one set) is: task 0: 261, 1: 228, 2: 259, 3: 241, 4: 201, 5: 158, 6: 203, 7: 216, 8: 355, 9: 244.
- Per-timestep AUROCs on the *uncut* corpus are already in `results/constraint_auroc*.csv`. They aren't comparable to this work (no cut, different metric); don't mix them into the new tables.

---

## 6. Stage 2 findings (final, closed)

Unseen tasks, mean over 3 seeds. β and layer chosen on eval-seen pooled M1. Orientation by per-task train M1.

| Signal | Smoothing | Sign | M1 pooled | M1 per task |
|---|---|---|---|---|
| `emb_temp[28]` | none | − | 0.719 | 0.725 |
| `act_rep` | β 0.96 | + | 0.629 | 0.613 |
| `act_dir` | β 0.83 | + | 0.623 | 0.615 |
| `act_mag` | β 0.83 | − | 0.601 | 0.569 |
| `xl_final_hN[26]` | β 0.91 | − | 0.638 | 0.567 |
| `xl_adj[2]` | β 0.91 | − | 0.593 | 0.550 |
| `xl_final[2]` | β 0.96 | − | 0.593 | 0.503 |
| `grip_flip` | β 0.96 | + | 0.583 | 0.576 |
| `xl_spread` | β 0.96 | − | 0.525 | 0.573 |
| Timestep only (floor) | — | — | 0.547 | 0.500 |
| FailureSpot `s` / `−s` | β 0.8 | fixed | 0.491 / 0.574 | 0.477 / 0.542 |
| Best token baseline (mean entropy) | β 0.91 | + | 0.574 | 0.550 |

- `emb_temp` is the real signal and holds per task. The action constraints hold modestly (0.57–0.62 per task). The cross-layer family mostly doesn't: its pooled score is largely the cross-task effect.
- β per family going into Stage 3: `emb_temp` none; action constraints and cross-layer chosen on eval-seen as usual (expect 0.83–0.96).
- Unstable (sign differs between seeds): `emb_temp` layers 1 and 15 unsmoothed (layers 11–15 when smoothed); 10 of 31 `xl_adj` layers; 3 of 31 `xl_final`; 5 of 32 `xl_final_hN`; `grip_flip` at β 0.83 and 0.91. Read the flags from `results/abc/constraint_m1.csv`, don't hard-code them.
- No-sink variants are no better; leave them out of Stages 3–6.