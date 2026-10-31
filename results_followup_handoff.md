# Results follow-up handoff

**Handoff version 4 — 31 Oct 2026.** If the copy in the repo doesn't say version 4 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box. Builds on `results-collection-handoff.md` (version 4) and its finished runs in `results/thesis_results/` (original training set) and `results/thesis_results_full/` (full training set). Every definition, plain-word rule (§0), working rule (§1) and fixed setup (§2) there is unchanged unless this document says otherwise. Version 4 replaces versions 1–3: the headline setup has changed, and §5 adds a smoothing sensitivity check.*

---

## 0. The headline setup has changed

The thesis now presents and discusses one setup:

- **Training data: the full training set** (all 900 trainable rollouts: original, new unperturbed, LIBERO-PRO object and instruction perturbations; seen tasks only). Selections are the full set's own, as made in the background run (`selections_report.md`).
- **Test data: unseen tasks incl. LIBERO-PRO** (the 3 unseen tasks' original, new unperturbed and perturbed rollouts), scored by per-task-and-perturbation AUROC. The stress set is reported as before (alarm share only).

The original training set and the original-rollout test set become the **training-data ablation** and a breakdown, not the headline.

**No refits, except in Check 4 (§5), which refits B's grid and C's probes and may use the GPU.** Everything else comes from per-row scores already saved by the two runs. If a needed score isn't saved, stop and say which.

- Code: `analysis/thesis_results/final/`. Outputs: `results/thesis_results_final/`, same layout as `results/thesis_results/` (`tables/`, `figures/`, `figures/data/`, `REPORT.md`, `RUNLOG.md`).
- Paired bootstrap exactly as in the main run (1,000 resamples within task-and-perturbation cell within seed; 95% intervals; "improvement" only if the interval excludes zero).
- One checkpoint at the end (⏸). Propose one commit message.

---

## 1. Regenerate the tables and figures for the headline setup

Same definitions, styles and file names as `results/thesis_results/`, but every headline number on the full training set, tested on unseen tasks incl. LIBERO-PRO.

| # | What | Change from the main run |
|---|---|---|
| T1 | Core table, every detector and baseline row | Headline setup. Keep the stress-set column. Second block: the same rows on the original-rollout test set (full training set), for reference |
| Paired | All comparisons of the main run's §4.5 list | On the headline setup |
| F1 | Example failure and success | From the full training set, a seed with a qualifying pair on the headline test set; report rollout IDs, task, perturbation type and why chosen |
| F2 | Alarm trade-off over 15 α | Single panel, headline setup |
| F4 | Length check | Headline setup |
| F5 | Untrained constraint heatmap | Headline test set, full-set signs and unstable flags |
| F6–F8, F10 | Per-probe figures (same format rules: identical y-limits, unweighted-mean line, eval-seen faded) | Headline setup. B has 116 probes per H |
| T3 | Constraints and baselines | Full-set selections |
| T4 | B per look-ahead H: best probe, mean, embedding-temporal mean, trained combiner | Headline setup. Drop the persistence and own-target rows from T4; keep them in an appendix CSV only |
| T5 | Families (C leave-one-out; B per target) | Headline setup |
| F3, T2, F11 | Corpus, composition, capacity | Unchanged; copy |

F9 (B's own-target heatmap) and the persistence table go to `appendix/` only.

---

## 2. Check 1 — B's embedding-temporal mean against A and C

B's embedding-temporal mean was pre-registered as B's secondary detector (label-free run, 19 Oct). On the headline setup, each H:

- B's embedding-temporal mean vs A's mean, vs C's mean, vs B's mean of all grid probes, vs the best untrained constraint.

**Exploratory row, labelled as such:** B's mean of every grid probe except the `xl_final` probes (the expected-null control target in `abc-implementation-handoff.md` v10). Same comparisons. Label it "exploratory (excludes the cross-layer target)" everywhere.

Metrics: per-task-and-perturbation AUROC, balanced accuracy at α = 0.1. Output: `tables/b_secondary_paired.csv`, `tables/b_secondary_rows.csv`.

---

## 3. Check 2 — training-data ablation on the headline test set

Original vs full training set, both tested on unseen tasks incl. LIBERO-PRO, full minus original, paired bootstrap:

1. A's mean, C's mean, the best untrained constraint, the baselines (as in the existing `table6_training_data_paired.csv`).
2. **B on a common probe set:** the full set dropped 11 sign-unstable targets (7 of them `xl_final`), so B averages 116 probes there and 124 on the original set. Restrict both to the (layer, target) pairs present in both grids; report the count and what is excluded from each side. Then the same comparison per target (`act_rep`, `act_mag`, `emb_temp`, `xl_final`), common probes only.

Output: `tables/table6_training_data_headline.csv`.

---

## 4. Check 3 — where the detection on LIBERO-PRO comes from

On the original training set, B's and C's means rose about +0.08 from the original rollouts to the LIBERO-PRO test set, while A's rose +0.009. On the headline setup (full training set), for A's mean, C's mean, B's mean (each H), B's embedding-temporal mean, the best untrained constraint and the timestep-only score, report per-task AUROC on:

1. the original rollouts of the unseen tasks;
2. the original + new unperturbed rollouts;
3. each cell family of the LIBERO-PRO test set separately: unperturbed, object perturbation, instruction perturbation (number of cells and class counts in each).

Paired bootstrap: B mean vs A mean within each of these. Output: `tables/test_set_breakdown.csv`.

---

## 5. Check 4 — smoothing sensitivity for B and C

Smoothing β is chosen once per family on eval-seen. In the headline setup: `emb_temp` none, action family 0.83, cross-layer family 0.96. The smoothed series are C's inputs and B's training targets, so both detectors depend on the choice. This check shows how much.

**Settings** (headline setup only: full training set, tested on unseen tasks incl. LIBERO-PRO, 3 seeds):

| Setting | emb_temp | action family | cross-layer family |
|---|---|---|---|
| current (selected) | none | 0.83 | 0.96 |
| no smoothing | none | none | none |
| β = 0.83 everywhere | 0.83 | 0.83 | 0.83 |
| β = 0.96 everywhere | 0.96 | 0.96 | 0.96 |

**For each setting**, redo constraint preparation from the raw `.npz` series exactly as the Methods describe (fill, smooth, scale on successful train kept rows, orient by train-split per-task AUROC), so that signs, sign-unstable flags and B's binarisation thresholds follow the setting. **Hold fixed:** splits, seeds, every `c` at its current headline value (B's grid c = 1e-4; C's probes as selected), the conformal calibration procedure.

- **C:** refit the 33 probes on the setting's prepared inputs. Report the single best probe (chosen on eval-seen as before) and the unweighted mean.
- **B:** refit the grid at H = 0, 10 and 25 on the setting's targets, skipping that setting's sign-unstable targets. Report the single best probe, the unweighted mean (primary), the embedding-temporal mean (secondary) and the probe count per H. No trained combiner.
- **Reference row:** the best untrained constraint under each setting (layer re-chosen on eval-seen per-task AUROC).
- **Sanity check:** the "current" setting must reproduce the headline rows within ± 0.005 AUROC. If not, stop and find out why.

**Compute:** load each layer's features once and fit every setting's targets from that load; fits are about 0.5 s each (`logs/audit_timing.json`). Launch in tmux, resumable.

**Report**, per detector: per-task-and-perturbation AUROC (mean ± std over seeds), balanced accuracy at α = 0.1, the paired difference from "current" with its 95% interval, and B's probe count. Outputs: `tables/smoothing_sensitivity_b.csv`, `tables/smoothing_sensitivity_c.csv`, `tables/smoothing_sensitivity_signs.csv` (sign and unstable flag per constraint per setting).

---

## 6. Methods values

`tables/methods_values.csv`: every number the Methods chapter quotes, as used in the headline setup — smoothing β per family, `c` for every probe set and combiner, sign-unstable constraints, B's probe count and skipped targets, the best untrained constraint's layer, MLP weight decays, calibration rollouts per seed — with the original-set value beside each.

---

## 7. Report ⏸

`REPORT.md`, plain words:

1. **Headline**, at most five sentences: A, C and B on the headline setup and which gaps are separable; whether B's embedding-temporal mean is level with A; whether the training data changes anything once B is compared on common probes; where the LIBERO-PRO detection comes from; how sensitive B and C are to smoothing.
2. Every table and figure from §1–6.
3. Anything that needed a decision, and one proposed commit message.
