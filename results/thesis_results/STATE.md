RUN COMPLETE (2026-10-30 ~15:10 UTC). Main run: results/thesis_results/REPORT.md (+ CHECKPOINT_0_audit.md, CHECKPOINT_1.md, CHECKPOINT_2.md).
Background run (full training set, §6): results/thesis_results_full/REPORT.md, selections_report.md, tables/training_set_comparison.csv.
Reproduction checks: all pass except 3 per-probe values (A's action-token probe), each explained as harmless fp16 storage rounding.
Decisions for James (nothing blocked): F1 example uses seed 1 (seed 0 had no qualifying pair); B's probes lose to persistence for smoothed targets; token max-probability definition.
Commit messages: results/thesis_results/commit_message.txt (main) and results/thesis_results_full/commit_message.txt (background); James commits.

## Details
- nothing running; RUN_DONE written ("complete")
- work files /data/tmp/thesis_results_work/{original,full} (7.6 GB, deletable; re-running any step rebuilds them)
- rerun anything: bash analysis/thesis_results/run_main.sh (main) / bash analysis/thesis_results/run_full.sh (background); both skip finished work
- RUNLOG: results/thesis_results/RUNLOG.md (both runs)
