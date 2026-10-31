# RUNLOG: results follow-up (results_followup_handoff.md, version 4)

Run in-session on 2026-10-31 (UTC), on the Brev box. Code is in `analysis/thesis_results/final/`; `run_final.sh` reruns everything (resumable). Work files are in `/data/tmp/thesis_results_work/final/{full,original}/` (5.0 GB, scratch). No git writes.

## Timeline (UTC)

- 04:2x: handoff saved as `results_followup_handoff.md`.
- 04:27–04:42: Check 4 in tmux `tf-smoothing` (`run_smoothing.sh`).
  - signs: 4 settings × 99 constraints × 3 seeds, 396 entries, all equal to the Stage 2 table.
  - C refits: 4 × 3 × 33 probes, about 40 s.
  - B refits: 32 layers × 4 settings × 3 seeds, about 11 min, GPU.
  - eval, sanity check and bootstrap: about 2 min.
- 04:31–04:33: §1 (`headline.py`): tables, paired comparisons (identity check max |diff| 5e-7), example, figures, appendix.
- 04:35–04:37: Checks 1–3 and §6 (`checks.py`). Check 2 runs its original-set side in a subprocess with `THESIS_POOL=D0`.
- 04:42: `scores.py` wrote 66 full-set and 15 original-set detectors.
- 04:43: `report.py` wrote `REPORT.md`.

## Reused (read, not recomputed)

- `results/thesis_results_full/tables/table1_core_detectors_all_test_sets.csv`, `table1_stress_set.csv`, `table4_b_by_lookahead.csv` and `table5_ablations_families_and_lookahead.csv`: T1, T4 and T5, re-cut to the headline test set.
- `results/thesis_results_full/tables/training_set_comparison.csv`: Check 2, part 1.
- `results/thesis_results_full/tables/selections_comparison.csv`: the Methods values.
- `results/thesis_results_full/tables/data/detector_{auroc,alarm,stress}_per_seed.csv`: F2, the Methods values and the sanity check.
- `results/thesis_results_full/figures/data/per_probe_outcome_auroc.csv` and `untrained_constraint_heatmap.csv`: F5–F8 and F10.
- `results/thesis_results_full/tables/table4_b_*persistence.csv`, `data/b_target_auroc.csv` and `figures/data/b_target_auroc_heatmap.csv`: appendix.
- `results/thesis_results/tables/table2_corpus_composition.csv`, plus F3, F11 and their CSVs: copied unchanged.
- `results/abc2/constraint_m1_summary.csv`: T3 (full training set rows), and the cross-check of the recomputed signs.
- `/data/tmp/thesis_results_work/full/{detectors,eval,npz,b_probes}/`: per-row scores and bands of every existing detector, and B's grid outputs (Checks 1–3, paired comparisons, F1, F4).
- `/data/tmp/thesis_results_work/original/b_probes/`: B's original-set grid outputs (Check 2, part 2).

## New computation

- Paired bootstrap on the headline test set: §1 list, Check 1, Check 2 part 2, Check 3, Check 4.
- Unweighted means of saved grid outputs:
  - B without the xl_final probes (93 per H);
  - B on the 113 common probes, overall and per target, for both training sets.
- Check 4: constraint preparation from the raw series, then C and B refits, for 4 smoothing settings.
- F1 example selection, and the F4 length-check curves on the headline test set.

## Checks

- Paired bootstrap identity check: passed (5.0e-7).
- Smoothing sanity check: "current" reproduces all 12 headline rows (difference of means 0.0000; `logs/smoothing_sanity_check.json`).
- Recomputed signs and unstable flags equal the Stage 2 table for all 4 settings.
- Common probe set: 113 per H (original grid 124, full grid 116).
  - Only in the original grid: L11–13 emb_temp; L19, 20, 24, 26, 28–31 xl_final.
  - Only in the full grid: L01 emb_temp, L15 emb_temp, L17 xl_final.

## Deviations

See REPORT.md, "Decisions and deviations". Key items:
- F1 uses original LIBERO-10 rollouts, because the new rollouts saved no frames.
- Check 3's items 2 and 3a are identical by construction.
- Joint-fit shifts of ≤ 0.002 in unchanged B targets.
