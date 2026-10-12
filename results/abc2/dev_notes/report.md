# Dev note: `report` (Step 9 paired comparison + section 5 report)

Saved by the lead from the report agent's hand-back (the agent could not write it).

- `analysis/abc2/bootstrap2.py`: Step 9 paired bootstrap. Resample with replacement within each cell within each seed's test set; same draws for both conditions; per-seed metric then mean over seeds; cut recomputed on the resample; alarms from stored band (alpha 0.1). Outputs paired_<T>.csv, paired_<T>_per_seed.csv, paired_<T>_check.json (identity check vs final_table.csv / cp_curves.csv). RNG_SEED 20261011.
  API: paired_bootstrap(files_x, key_x, files_y, key_y, test_set="T0", n_boot=1000, rng_seed=20261011, alpha=0.1, demotions=(), comparison=None) -> {rows, per_seed, identity}. Key K needs K_m1, K_cp, K_mu, K_xi, K_h.
  CLI generic: --pair "path{seed}.npz:KEY" "path{seed}.npz:KEY" --check-table T.csv --out-name NAME.csv
- `analysis/abc2/report2.py`: REPORT.md (sec. 5 items 1-11), report_tables/*.csv, report_figures/{m1_vs_pool,balacc_vs_alpha}.png. Optional MLP/B inputs: mlp_final_table.csv, mlp_paired_T0.csv, b_final_table.csv, {mlp,b}_status.json. Section 10 = RUNLOG bullets matching DEVIATION|Decision|FAIL|judgement|skipped|demot|FINDING.
- Commands: python analysis/abc2/bootstrap2.py ; python analysis/abc2/bootstrap2.py --test-set T1 T2 ; python analysis/abc2/report2.py
- Tests (dev run of pipeline D0/D1/D2 x 3 seeds with relabelled v2 rollouts, 3 A probes): identity check max |diff| 5e-7 vs rounded CSVs, 2.2e-16 vs unrounded; 1,080 random resamples matched data.m1 / evaluate_cp to 2.2e-16; self-comparison gives 0; runtime 2 s at N=1000.
- Lead fix: report2 pool_sizes glob tightened to `<cond>_seed?.npz` (B writes `<cond>_seed<s>_b10.npz`).
