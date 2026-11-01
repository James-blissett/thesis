# Task-resampling handoff

**Handoff version 4 — 1 Nov 2026.** If the copy in the repo doesn't say version 4 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box. Builds on `results_followup_handoff.md` (version 4) and its finished run in `results/thesis_results_final/`. Every definition and plain-word rule there is unchanged. In all outputs, call a seed a **task split** (it sets which 3 tasks are held out and how the rest divide into train and eval-seen); "seed" is the repo's word only. New code in `analysis/thesis_results/task_resampling/`, outputs in `results/thesis_results_final/task_resampling/`.*

---

## 0. Why

Every uncertainty range in the thesis comes from reshuffling rollouts **within** the 3 unseen tasks of each task split. It never asks what would happen if different tasks had been held out, which is the biggest source of uncertainty (only 3 test tasks per split, from 10). This run recomputes the ranges so that they also reshuffle **which tasks are included**.

**No refits.** Detector scores, training splits and conformal thresholds stay exactly as saved. Only the way test rollouts are drawn for the ranges changes.

---

## 1. The resampling (one round)

For each task split (repo: seed) separately:

1. Draw 3 tasks **with replacement** from that split's 3 unseen tasks. A task drawn twice counts twice.
2. For each drawn task, take all of its task-and-perturbation groups. Within each group, draw its rollouts with replacement, as the existing ranges do.
3. Compute the metric per split on this drawn set: per-task AUROC averaged over the drawn groups (a group drawn twice counts twice; skip a group if the draw leaves it without at least 2 successes and 2 failures), and balanced accuracy with each split's saved conformal threshold held fixed.

Average the three splits' values. For a comparison between two detectors, use **the same drawn tasks and rollouts for both** in each round, and record the difference.

Repeat 2,000 rounds with a fixed random seed. The 95% range is the 2.5th to 97.5th percentile of the recorded values. Record how many rounds were skipped or had groups dropped.

**Self-check before the full run:** with step 1 switched off (tasks kept as they are), the code must reproduce the existing ranges in `tables/paired_comparisons.csv` within ±0.005.

---

## 2. What to recompute

All on the headline setup (full training set, unseen tasks including LIBERO-PRO), B at H = 10 unless stated.

1. **Headline comparisons:** B mean − A mean, B mean − C mean, A mean − C mean; per-task AUROC and balanced accuracy at α = 0.1.
2. **Within each detector:** unweighted mean − single best probe, − trained combiner, − 2-layer MLP, for A, B and C (per-task AUROC).
3. **B's embedding-temporal mean** − A mean, − C mean, − untrained `emb_temp`, at H = 0, 10 and 25.
4. **Per perturbation type:** B mean − A mean on unperturbed, object-perturbation and instruction-paraphrase groups separately.
5. **Training data:** full minus original training set, for A mean, C mean and B's mean on the 113 shared probes. Both training sets are scored on the same drawn rollouts in each round.
6. **Smoothing:** each setting minus the selected setting, for C mean and B mean.
7. **Comparison with published detectors:** B mean − A mean on the original training set, tested on the unseen tasks' original rollouts only (the setting closest to SAFE's), per-task and pooled AUROC.
8. **B against the untrained constraint:** B mean − untrained `emb_temp` at layer 26 (the layer chosen on eval-seen); per-task AUROC and balanced accuracy at α = 0.1. Also B mean on the original training set, tested on the original rollouts, minus the same with the full training set.

---

## 3. Output for the ranges

`task_resampling/ranges.csv`, one row per comparison, columns: comparison, metric, difference, 95% range, whether the range excludes zero, rounds used. Report only these ranges; do not present them against the earlier rollouts-only ranges (those were used only for the self-check in §1).

---

## 4. False alarms against the target rate

The conformal threshold at level α is calibrated on seen-task successes so that at most a share α of successes raise a false alarm. Check whether that holds on the unseen tasks.

From the saved alarm results (headline setup), for A's, B's and C's means, their trained combiners, the untrained `emb_temp` constraint and the token and FailureSpot baselines: the false-alarm rate on unseen successes at each of the 15 α levels, mean ± std over splits. Output `task_resampling/false_alarms_vs_alpha.csv` and one figure (false-alarm rate against α, with the line false-alarm rate = α), PDF and PNG, same style as the existing figures.

---

## 5. Which failures are caught

Every result so far is an aggregate. This section asks which kinds of failure the detectors catch, using the original LIBERO-10 rollouts, the only ones with saved frames.

**Rollouts:** every failed rollout among the original LIBERO-10 rollouts of each split's unseen tasks (headline setup, full training set), each counted once per task split it is tested in.

**Failure type, from the logged actions** (before the task's cut only), in this order, first match wins:
1. **Froze:** the longest run of consecutive steps with a repeated action (the `act_rep` definition) is 20 steps or more. Record the step the run starts.
2. **Never grasped:** the gripper is never commanded closed.
3. **Other:** everything else (moves and grasps, but still fails).

**Check by eye:** for 20 failed rollouts (spread across types and tasks, fixed random seed), open the frames at the start, at one third and two thirds of the way, and at the cut, and write one plain sentence per rollout on what went wrong. Report how many of the 20 the action-based type matches, and adjust the 20-step threshold only if most "froze" labels are clearly wrong (log any change).

**Report, per failure type:** number of rollouts; share caught before the cut at α = 0.1 by A's, B's and C's means and the untrained `emb_temp` constraint; and for "froze" only, the share of caught failures whose alarm comes **before** the freeze starts, and the median gap in steps between freeze start and alarm.

Outputs: `task_resampling/failure_types.csv` (one row per rollout and split: rollout ID, task, type, freeze start, each detector's alarm step), `task_resampling/failure_types_summary.csv`, and the 20 written descriptions in `task_resampling/failure_descriptions.md`.

---

## 6. Random-direction control for B

**Why.** Every one of B's layer-0 probes detects failure about as well as A's layer-0 probe (0.74 to 0.78), whatever its target, even though the layer-0 targets carry no failure signal themselves. The likely reason is that any reading of the layer-0 hidden state picks up input-level failure cues. This control tests that: if random directions through the hidden state detect failure as well as B's probes, B's training isn't what makes them work at that layer.

**No refits.** Uses saved hidden states and existing scoring code only.

**At every layer B has probes for** (headline setup, H = 10, each task split separately):
1. Prepare the hidden states **exactly as B's probes' inputs are prepared** (same pooling over action positions, same normalisation fitted on that split's training rollouts). Read this from the code and state it in the report.
2. Draw 200 random directions (standard normal, scaled to length 1; fixed random seed; the same 200 directions for all three splits).
3. Score every rollout with each direction exactly as B's probes are scored: per-step score = prepared hidden state · direction, running sum to the cut.
4. **Sign:** compute each direction's per-task AUROC on eval-seen; if below 0.5, flip the direction. This gives random directions a small help from outcome labels that B never gets, so it is a tough test for B. Also keep the unflipped values.
5. Per-task AUROC on the unseen tasks, mean over the three splits, per direction.

**Output** `task_resampling/random_directions.csv`, one row per layer: median and 5th/95th percentiles of the 200 random directions (sign set on eval-seen, and unflipped); B's probe AUROC for each target at that layer; share of random directions each B probe beats.

**Two small extra checks:**
- **B without layer 0:** B's unweighted mean over all probes except the layer-0 ones, per-task AUROC, and its difference from the full mean with a range from §1. Same for A's unweighted mean without its layer-0 probe. Output `task_resampling/b_without_layer0.csv`.
- **Agreement with A at layer 0:** for each of B's layer-0 probes, the rank correlation (Spearman) between its rollout scores and A's layer-0 probe's rollout scores, computed within each task-and-perturbation group and averaged. Repeat at layers 1, 10 and 26 for contrast. Add as extra columns in `random_directions.csv` or a small separate table.

**Figure:** redraw F8 with the random-direction band (5th to 95th percentile, sign set on eval-seen) as a light grey band behind B's probe lines. Same file name, PDF and PNG; keep the old version as `F8_per_probe_detector_b_noband.*`.

---

## 7. Report

`task_resampling/REPORT.md`, plain words:

1. **Headline, five sentences:** which differences exclude zero; the largest amount by which B could trail A that the range still allows; whether the false-alarm rate stays at or below α on unseen tasks; which failure types each detector catches and misses; and how B's probes compare with random directions at layer 0 and from layer 1 onward.
2. The tables and figures from §3–6.
3. Anything that needed a decision, and one proposed commit message.

Stop after the report.
