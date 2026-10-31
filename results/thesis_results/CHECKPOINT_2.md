# Checkpoint 2: everything in §4, all test sets

*results_collection_handoff.md (version 4), §5 Checkpoint 2. Written 2026-10-30 ~14:25 UTC by the unattended run. The full write-up is `REPORT.md`, which has the headline, every table and figure, the reproduction checks, the deviations and a proposed commit message.*

**Decision on §6: the background run (full training set) was started and finished at 15:04 UTC.** Its outputs are `results/thesis_results_full/REPORT.md`, `selections_report.md` and `tables/training_set_comparison.csv`. With the full training set, no primary detector's ranking changes significantly on unseen tasks (original rollouts). See the full-set REPORT.md. The brief's condition holds. Every reproduction check passed except 2 of 603 per-probe values. Those two were investigated and logged in RUNLOG.md as harmless: fp16 storage rounding of A's action-token probe, eval-seen, 0.0053 against a 0.005 tolerance; float32 reproduces the source exactly. The background run writes only to `results/thesis_results_full/` and `/data/tmp/thesis_results_work/full/`. It touches nothing from the main run, so it can be discarded if James decides against it.

**Questions for James (nothing blocks):**
1. **Example rollout (F1)** uses seed 1, not seed 0. In seed 0, B's mean at α = 0.1 alarms on no unseen failure before the cut, so no seed-0 pair exists. Alternatives would be another α, or a different detector for the example.
2. **B's mean vs persistence.** For every smoothed target, the target's current value predicts its look-ahead label better than any of B's linear probes (`tables/table4_b_linear_vs_persistence.csv`). This affects how the B section frames "predicting constraints ahead".
3. **Token max-probability** is read as the top-bin probability averaged over the 7 action tokens (`token_prob_mean`). If the Methods mean the maximum over tokens (`token_prob_max`), that row needs swapping. The AUROC already exists in `results/abc2/constraint_m1_summary.csv`; only the alarm would be new.

## What was collected (§4)

| § | Item | Status | Where |
|---|---|---|---|
| 4.1 | Core table, every detector row, three test sets | done | `tables/table1_core_detectors*.csv`, `table1_stress_set.csv` |
| 4.2 | Baselines with the conformal alarm (timestep, best untrained constraint by the per-task rule = emb_temp[26] and the earlier emb_temp[28], token entropy and max-probability, FailureSpot both ways) | done (alarms new) | same |
| 4.3 | Per-probe outcome profiles: A 34 + shuffled-label control per probe (new), C 33, B 124 × 3 H (new), B per-layer mean (new), untrained constraints by running maximum (read) and running sum (new), per-layer mean of three, best of three | done | `figures/data/per_probe_outcome_auroc.csv` (F6–F8, F10) |
| 4.4 | B's own-target AUROC on all three sets; persistence and timestep-only baselines (new); B's 2-layer MLP (new); mean minus combiner per H | done | `tables/data/b_target_auroc.csv`, `table4_*.csv`, `tables/data/b_mean_minus_combiner.csv` |
| 4.5 | Paired bootstraps, headline test set (13 comparisons × per-task AUROC, balanced accuracy, failures caught) | done | `tables/paired_comparisons.csv` |
| 4.6 | Families: C's mean without each family (new); C's 132-input logistic regression without each family (existing version); B per target, mean (new) and trained combiner (12 rows, equal to `results/abc/b_final.csv`). Look-ahead H. Capacity (`results/abc3`). Training data (`results/abc2`; B joins from §6) | done | `table5_*.csv`, `table6_training_data.csv`, F11 |
| 4.7 | Corpus composition, kept-row curves, completion times, action-repeat rate (original: equals `completion_confound.json`; every source: new), length check | done | `table2_corpus_composition.csv`, `figures/data/*.csv`, F3, F4 |
| 4.8 | Example failure and success | done, **seed 1** (see question 1) | `figures/data/example_rollouts.{csv,json}`, `figures/example_frames/`, F1 |

**Reproduction checks: 3,134 in all, 2 failed (explained).**
- 292 detector rows against six source files: all pass, max diff 0.0001.
- 603 per-probe values: 601 pass.
- 2,232 B own-target AUROCs: all pass, max 0.0013.
- 7 corpus counts: all pass.
- The bootstrap identity check passes (5e-7).

## Headline numbers

| Test set | A mean | C mean | B mean (H = 10) | Best untrained constraint |
|---|---|---|---|---|
| Unseen tasks, original rollouts: per-task AUROC | 0.745 ± 0.037 | 0.684 ± 0.066 | 0.678 ± 0.058 | 0.725 ± 0.061 |
| Unseen tasks incl. LIBERO-PRO: per-task-and-perturbation AUROC (8 / 7 / 9 cells) | 0.754 ± 0.027 | 0.760 ± 0.026 | 0.758 ± 0.014 | 0.740 ± 0.019 |
| Unseen tasks, original rollouts: balanced accuracy at α = 0.1 | 0.536 | 0.502 | 0.510 | 0.553 |
| Stress set: share alarmed before the cut at α = 0.1 | 0.207 | 0.183 | 0.313 | 0.290 |

The reading notes in REPORT.md cover the rest:
- combiners make no separable difference to ranking;
- the xl_final probes pull B's mean down;
- B rises with H;
- running-sum scores mostly count elapsed steps (F4), which is why the per-task cut matters.

## Proposed commit message (main run)

See `commit_message.txt` (repeated at the end of REPORT.md).
