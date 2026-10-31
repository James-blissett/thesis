# Results collection handoff

**Handoff version 4 — 31 Oct 2026.** If the copy in the repo doesn't say version 4 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box.*
*Builds on `abc-implementation-handoff.md` (v10), `b-label-free-handoff.md` (v1), and the finished more-data and deep-MLP runs (`results/abc2/`, `results/abc3/`). Every definition there is unchanged unless this document says otherwise: kept rows, splits, constraint preparation, row weights, running-sum scoring, per-task and pooled rollout-level AUROC, the cut-variant conformal alarm, and the paired bootstrap.*

---

## 0. Plain words only

The thesis uses no internal shorthand. Every figure label, table header, CSV column, file name you create, and sentence in the report uses the plain names below. The repo keys are only for finding existing files.

| Plain name (use this) | Repo key (read files only) |
|---|---|
| original training set (300 LIBERO-10 rollouts) | `D0`, `ori` |
| full training set (1,000 rollouts, including LIBERO-PRO object and instruction perturbations) | `D2` |
| 500-rollout training set (original + 200 unperturbed) | `D1` |
| unseen tasks, original rollouts | `T0` |
| unseen tasks, all rollouts incl. LIBERO-PRO object and instruction perturbations | `T2` |
| unseen tasks, original + new unperturbed rollouts | `T1` |
| position-swap and task-perturbation stress set | `Stress` |
| per-task AUROC / pooled AUROC (rollout-level) | `M1-per-task` / `M1` |
| unweighted mean of B's probes | `LF-all` |
| unweighted mean of B's embedding-temporal probes | `LF-emb` |
| Detector A / B / C | A / B / C (these stay) |
| eval-seen, train, unseen | unchanged (defined in the thesis) |

---

## 1. Working rules

- **No git commands that write.** One proposed commit message per checkpoint; James commits.
- **Main run:** stop at each checkpoint (⏸) and report. **Background run (§6):** unattended, in tmux, logs on disk, resumable.
- Anything over about a minute: print the tmux launch command for James, except in the background run, where you launch and poll it yourself (at most every 10 minutes).
- Run from the repo root after `source env.sh`.
- New code in `analysis/thesis_results/`. Outputs in `results/thesis_results/` (main) and `results/thesis_results_full/` (background). Don't edit anything older; import from it.
- Never delete or overwrite `/data/rollouts_v2/`, `/data/rollouts_v3/`, `constraints*/`, `results/abc*/`, `results/b_label_free/`, or any feature cache.
- **Reuse before recompute.** If a number already exists, read it and say which file it came from. Recompute only what §4 marks NEW, or what fails a reproduction check.
- Log every deviation and every reused file in `results/thesis_results/RUNLOG.md`.
- Label convention: 1 = failure, 0 = success.

---

## 2. Setup that is fixed for the whole run

- **Training set:** the original training set, seeds 0, 1, 2, with the existing splits (`results/abc/splits/`, identical to the original-set splits in `results/abc2/splits/`).
- **Test sets, all three for every detector:** unseen tasks (original rollouts) is the headline; unseen tasks incl. LIBERO-PRO; the stress set.
- **B's headline look-ahead:** H = 10 (the pre-registered default). Every B number is also reported at H = 0 and 25; nothing is selected across H.
- **Selection:** everything chosen on eval-seen per-task AUROC, as before. Unseen and stress rollouts never choose anything.
- **Report:** mean ± std over the three seeds everywhere, plus per-seed values in the CSVs.
- **Everything fitted on the original training set is frozen and applied unchanged to every test set:** constraint smoothing β, the success-row means and standard deviations used for scaling, orientation signs, unstable flags, B's binarisation thresholds, each probe's input scaler and weights, every combiner, and the conformal band. Nothing is re-fitted or re-normalised on LIBERO-PRO rollouts. Constraints for the new rollouts come from `constraints_v3/`.
- **Per-task AUROC on the LIBERO-PRO test set** is computed per (task, perturbation type) cell, as in the more-data run. Call it "per-task-and-perturbation AUROC" in labels and state the number of cells averaged.
- **Stress set:** nearly every rollout fails, so no AUROC is reported there; only the share alarmed before the cut at α = 0.1.
- **Detection time:** first alarm step ÷ that rollout's cut length, set to 1 if no alarm, averaged over failed rollouts. If the existing code divides by something else, use the existing definition for every row, report which, and flag it, because the thesis Methods still has this marked as unresolved.

---

## 3. Stage 0 — storage and inventory audit ⏸

1. Check these still exist and report sizes: `/data/rollouts_v2/`, `/data/rollouts_v3/`, the original-set feature cache (`/data/tmp/abc_features/`), the full-set feature cache (`/data/tmp/abc2_features/`), `constraints/`, `constraints_v3/`, and every `results/` directory named above. Report free disk space.
   - If the full-set feature cache is gone, the two LIBERO-PRO test sets can't be scored for hidden-state detectors. Say so and say whether it can be rebuilt from `/data/rollouts_v3/` and how long that would take. Don't rebuild without asking.
2. **Inventory of saved per-row outputs.** For each item below, report whether per-row probabilities or scores are saved for eval-seen, unseen and (where applicable) the LIBERO-PRO test sets, and the file:
   - A's 34 probes (not just the selected one)
   - C's 33 per-layer and action probes
   - B's 124 grid probes at each H
   - C's 132-input logistic regression and 2-layer MLP; A's 2-layer MLP; the deep MLPs
   - the trained combiner for B; the unweighted means of B
3. **Timing.** Time one A probe fit, one B grid fit (one layer, 12 outputs), and one 2-layer MLP fit on the original training set. Use these to estimate the main run and the background run in hours.
4. Report, propose nothing to commit, stop.

---

## 4. Main run — what to collect

### 4.1 Final detectors (the core table)

Every row is scored by the running sum and evaluated on all three test sets (stress set: alarm share only, §2): per-task AUROC, pooled AUROC (with the timestep-only floor), and the alarm at all 15 α (balanced accuracy, share of failures caught, false-alarm rate, mean normalised detection time).

| Detector | Variant | Status |
|---|---|---|
| A | single best probe (of 34) | exists; re-score on the LIBERO-PRO test sets if missing |
| A | **unweighted mean of 34 probe outputs** (primary) | NEW |
| A | trained combiner: logistic regression over the 34 probe outputs, on outcome, row weights, out-of-fold | NEW |
| A | 2-layer MLP on the best probe's hidden state | exists (more-data run, original training set) |
| C | single best probe (of 33) | NEW selection from existing probe scores |
| C | **unweighted mean of 33 probe outputs** (primary) | NEW |
| C | trained combiner over the 33 probe outputs, out-of-fold | NEW |
| C | 2-layer MLP on the best probe's inputs (3 constraints, or the 4 action constraints) | NEW |
| C (exploratory) | logistic regression on all 132 constraints | exists |
| C (exploratory) | 2-layer MLP on all 132 constraints | exists |
| B, each H | single best probe of 124, chosen on eval-seen **per-task AUROC against outcome** | NEW |
| B, each H | **unweighted mean of 124 probe outputs** (primary) | exists (label-free run); re-score on LIBERO-PRO test sets |
| B, each H | trained combiner over 124, out-of-fold | exists; re-score on LIBERO-PRO test sets |
| B, H = 10 | 2-layer MLP on the best probe's layer hidden state, **trained on that probe's constraint target** | NEW (see 4.4) |

**Out-of-fold, for the new A and C combiners:** exactly B's procedure. Within train, 5-fold `GroupKFold` by rollout; each fold's probes predict the held-out fold; the combiner is fitted on those outputs; then the probes refit on all train produce outputs for eval-seen and test sets. Combiner `c` chosen on eval-seen per-task AUROC from the usual six values. The fold refits are about 500 probe fits for A; if scikit-learn on CPU would take more than about 2 hours, use B's GPU logistic-regression fit instead, after checking it matches scikit-learn on one layer within 0.005 own-split AUROC.

**2-layer MLPs:** the plain architecture and recipe of the more-data run (standardised input → 256 → 256 → 1, ReLU, its optimiser, weight-decay grid, early stopping on eval-seen per-task AUROC). Same for the new C and B MLPs, so all three are like for like. Label these "2-layer MLP" everywhere. The capacity ablation's depth-2 model is a different (residual, LayerNorm) network: label it "2-layer residual MLP" so the two are never confused.

### 4.2 Baselines (one row each in the same table)

Rankings exist from the constraint stage; re-score on the LIBERO-PRO test sets if missing. **The conformal alarm (all 15 α) is probably NEW for the token and FailureSpot baselines and the timestep-only score**: the earlier alarm runs covered only the detectors and the best untrained constraint. Check, and compute where missing. Baselines: timestep-only; best single untrained constraint (`emb_temp` at the layer chosen on eval-seen **per-task** AUROC, running maximum; the earlier layer-28 choice used pooled AUROC, so report if the per-task choice differs); token mean entropy and max-probability; FailureSpot's signal in both orientations. Shuffled-label control for A's probes (exists).

### 4.3 Per-probe outcome profiles (for F6–F8 and F10)

For each seed, on eval-seen and every test set, per-task AUROC against outcome (running-sum score, same cut) for:
- every A probe (34; exists, plus the shuffled-label value per layer)
- every C probe (33; exists)
- every B probe (124 at each H) — NEW
- **B per layer**: unweighted mean of that layer's probes (up to 4) — NEW
- every untrained constraint per layer, **two columns**:
  - `running maximum` (exists): the Methods' scoring for untrained constraints. Used by F5, T3 and the best-untrained-constraint baseline; unchanged.
  - `running sum` — NEW: the same prepared constraint series, scored by its running sum at the cut, exactly as probes are scored. No fitting, re-scoring only. Used only for F7's comparison line. For C's per-layer comparison, also give the running sum of the mean of that layer's three prepared constraints, alongside the best of the three.

### 4.4 B's constraint-prediction results

- Row-level AUROC of every grid probe against its own target, per H, on eval-seen and unseen (exists in `b_grid.csv`; add the LIBERO-PRO test set if features allow).
- **Persistence baseline — NEW:** for each target and H, the target's current prepared value used as the prediction of its look-ahead label, row-level AUROC on the same rows. Also the timestep-only score against the look-ahead label.
- **B's MLP — NEW:** for each seed, take B's single best probe at H = 10 (from 4.1). Train the 2-layer MLP on that probe's layer's pooled hidden state, against that probe's binarised look-ahead target, balanced class weights, no outcome labels, regularisation and early stopping chosen on eval-seen **row-level AUROC against the target** (not outcome). Report: row-level AUROC against the target (vs the linear probe and persistence), and per-task AUROC against outcome plus the alarm when its probabilities are run-summed (vs the linear probe).
- Paired per-seed differences, unweighted mean minus trained combiner, per H (exist; copy).

### 4.5 Paired comparisons (headline test set)

Paired bootstrap as in the more-data run (1,000 resamples within task within seed), per-task AUROC and balanced accuracy at α = 0.1, 95% intervals:
- B mean vs A mean; B mean vs C mean; A mean vs C mean
- each detector's mean vs its own trained combiner, single best probe and MLP
- B's MLP vs B's best linear probe

### 4.6 Ablations (mostly exist; collect into tables)

- **Constraint families:** C's leave-one-family-out (exists, refit for the mean-based C if cheap; otherwise report the existing version and say which). B's per-target trained combiner and per-target unweighted mean, each H.
- **Look-ahead H:** B rows at 0/10/25 above.
- **Capacity:** depth 0 (linear), 2, 6, 12 for A and C, train / eval-seen / unseen per-task AUROC, both training sets (exists in `results/abc3/`).
- **Training data:** original, 500-rollout and full training sets × the three unseen test sets, for A, C and the best untrained constraint (exists in `results/abc2/`). B joins this table when the background run finishes.

### 4.7 Corpus and confounds

- Composition of all 1,000 rollouts: per source (LIBERO-10 original, LIBERO-10 new unperturbed, LIBERO-PRO object, instruction, position-swap, task) × task × outcome. From the more-data run.
- Kept-row curves for the original training set: number of kept rollouts per class against t (pooled over tasks), and the completion-time distribution of successes.
- Action-repeat rate before vs after first success (exists in `completion_confound.json`; recompute on the full corpus as well).
- **Length check — NEW for the new rows:** mean running-sum score against t on unseen failures and on unseen successes (kept rows only), one line per primary detector (A, C, B means) plus the timestep-only score.

### 4.8 Example rollout (for the alarm figure)

Seed 0, one unseen task. Pick one failure that B's mean alarms on before the cut at α = 0.1, and one success from the same task that it doesn't alarm on. Prefer a task with a clear stall. Save B's mean score trace, the conformal band, the cut, the alarm step, A's and C's mean traces, and 4 frames per rollout (start, mid, alarm step, cut) from `frames/`. Report the rollout IDs and why they were chosen.

---

## 5. Checkpoints and outputs

**Checkpoint 1 ⏸** after 4.1–4.2 on the headline test set only: the core table, with reproduction checks — existing rows must match their source files within ± 0.005 (AUROC) and ± 0.01 (balanced accuracy). If a new mean or combiner disagrees sharply with expectation (for example A's mean below its own single best probe by more than 0.1), say so before going on.

**Checkpoint 2 ⏸** after everything else in §4, all test sets.

**Outputs** in `results/thesis_results/`:
- `tables/` — one CSV per thesis table (§7), tidy, plain-word headers, mean, std and per-seed columns.
- `figures/` — one PDF and one PNG per figure (§7), plus the CSV that draws it (`figures/data/`). A figure must be redrawable from its CSV alone.
- `scores/` — per-row scores for every new detector, every test set and seed.
- `REPORT.md` — the headline in three sentences, then each table, each figure, deviations, and one proposed commit message.

**Figure style (all figures):**
- Vector PDF, 16 cm wide (full text width) unless stated; 9 pt sans-serif text at final size.
- Fixed colours across all figures: A, C, B one colour each; baselines grey; chance and floor dashed black. Every series also differs by line style or marker, so it survives greyscale printing.
- Shaded band = ± 1 std over seeds. Axes labelled in words with units; no titles (captions go in LaTeX).
- No shorthand anywhere in labels (§0).

---

## 6. Background run — the full 1,000-rollout training set

After **Checkpoint 2** is approved, launch in tmux and leave it unattended (not earlier: both runs read the same multi-GB layer files and share the GPU, which would slow the main run that the writing waits on):

- Repeat §4.1–4.6 with the full training set in place of the original, every selection re-made on that set's eval-seen, same seeds, same test sets. B's grid at all three H (5-fold out-of-fold plus refit), then everything downstream of it.
- Write to `results/thesis_results_full/` with the same file names and plain headers.
- Add a comparison table: each primary detector and baseline, original vs full training set, per-task AUROC and balanced accuracy at α = 0.1 on all three test sets, with the paired bootstrap. This becomes the training-data ablation.
- Resumable: `bash analysis/thesis_results/run_full.sh` skips any finished fit that passes its check.
- If B's grid fails twice, log it and finish everything else.
- Report what the selections chose (β per family, `c`, unstable constraints, B's probe count) next to the original-set values, because the thesis Methods quotes the original-set values.

---

## 7. What the figures and tables are for

Thesis Results chapter, about 14 pages, at most one figure per page. Order follows the chapter.

**Over budget by design.** 11 figures and 6 tables won't fit in 14 pages at one figure per page; expect 8–9 figures in the chapter. Produce all of them anyway (they're cheap and the writing decides). Likely cuts, in order: F10 (overlay), F1 (example rollout) or F4 (length check, to the appendix), T3 merged into F5's caption or T1.

| # | Section | Figure / table | Source |
|---|---|---|---|
| T1 | Runtime alarms | Core table: every row of §4.1–4.2, headline test set; per-task and pooled AUROC, timestep floor, α = 0.1 alarm columns, stress-set alarm share | 4.1, 4.2 |
| F1 | Runtime alarms | Example failure and success: B's mean score with band, cut and alarm step; frame strip above each | 4.8 |
| F2 | Runtime alarms | Balanced accuracy against normalised detection time over 15 α. Panels: (a) unseen tasks, original rollouts; (b) unseen tasks incl. LIBERO-PRO. Lines: A, C, B means; best untrained constraint; token entropy; FailureSpot | 4.1, 4.2 |
| T2 | Corpus | Composition of the 1,000 rollouts by source and outcome | 4.7 |
| F3 | Corpus | (a) Kept rollouts per class against t; (b) action-repeat rate before vs after first success | 4.7 |
| F4 | Corpus (or end of the B section, if the writing prefers detectors to be introduced first) | Length check: mean score against t, failures solid and successes dashed, A/C/B means and timestep-only | 4.7 |
| F5 | Constraints | Heatmap: untrained per-task AUROC, rows = `emb_temp`, `xl_adj`, `xl_final`, columns = layer 0–31; action constraints as a separate strip; sign-unstable cells hatched | exists |
| T3 | Constraints | Best per family with sign, smoothing and per-task AUROC; token, FailureSpot and timestep baselines | exists |
| F6 | Detector A | **Per-probe figure** (format below): 34 probes. Extra lines: shuffled-label control | 4.3 |
| F7 | Detector C | **Per-probe figure**: 33 probes. Extra line: the untrained constraints of that layer, **running-sum column**, best of the three per layer (dotted), so the line differs from the probes only in whether training happened | 4.3 |
| F8 | Detector B | **Per-probe figure**: 124 probes at H = 10, one line per target (`act_rep`, `act_mag`, `emb_temp`, `xl_final`); sign-unstable probes left as gaps | 4.3 |
| F9 | Detector B | Heatmap, layer × target, H = 10, unseen: row-level AUROC of each probe against **its own constraint target** (B only, since only B's probes have a non-outcome target) | 4.4 |
| T4 | Detector B | Per H: best probe, mean, trained combiner, paired difference; B's MLP vs best linear probe vs persistence | 4.4 |
| F10 | Where the signal is | Overlay of F6–F8 on one axis, unseen only: A per layer, C per layer, B per-layer mean of its probes. **Cut this first if over length**, since F6–F8 already carry it | 4.3 |
| — | Exploratory (appendix or cut) | Weights of C's 132-input logistic regression | exists |
| F11 | Ablations | Capacity: per-task AUROC against depth (linear, 2, 6, 12), panels A and C, train / eval-seen / unseen lines, original solid and full training set dashed | 4.6 |
| T5 | Ablations | Families (C leave-one-out; B per target) and look-ahead H | 4.6 |
| T6 | Ablations | Training data: original / 500 / full × three unseen test sets | 4.6, §6 |

**Per-probe figure format (F6, F7, F8 identical):**
- One point per probe: per-task AUROC **against episode outcome** on the headline test set, the probe's running-sum score, mean ± 1 std over seeds (shaded band).
- x-axis: layer 0–31. A adds index 32 at the right edge, marked "post-norm". A probe that doesn't read a layer (A's action-token probe, C's action-constraint probe) is a separate point with error bar to the right of a gap, labelled in words.
- Lines on every panel: unseen (solid); eval-seen (thin, faded); chance 0.5 (dashed black); the detector's own **unweighted mean** final detector (horizontal dash-dot, labelled; it is a separate model, not the average of the plotted AUROCs, so a probe can sit above it); the timestep-only floor if it isn't 0.5.
- **Same y-limits on all three figures**, chosen to fit every series in F6–F8. Same width and height, so the three sit at the same scale on their pages.
- One CSV behind all three: `figures/data/per_probe_outcome_auroc.csv`, columns: detector, probe, layer (blank for non-layer probes), target (B only), H (B only), test set, seed, per-task AUROC.
