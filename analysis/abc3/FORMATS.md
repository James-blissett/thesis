# abc3 file formats (deep-MLP run, deep_mlp_handoff.md v1)

All paths relative to the repo root. `source env.sh` first (Python venv). Code in
`analysis/abc3/`, results in `results/abc3/`. abc2 code is imported, never edited.

## Pipeline

| Step | Script | Reads | Writes |
|---|---|---|---|
| fits | `train3.py` (done by the lead) | abc2 cache, splits, selections | `/data/tmp/abc3_work/<pool>/<input>_d<depth>_seed<s>_p<p>_wd<wd>.pkl` (format: `common3.py` docstring) |
| evaluation | `eval3.py` | work files, `results/abc2/scores/*.npz` | everything under "eval3 outputs" |
| paired bootstrap | `paired3.py` | `results/abc3/scores/*_deep.npz`, `results/abc2/scores/<pool>_seed<s>.npz`, eval3 tables | `results/abc3/paired_T0*.{csv,json}` |
| report | `report3.py` | all of the above, `results/abc3/RUNLOG.md` | `results/abc3/REPORT.md`, `report_tables/`, `report_figures/` |

## Detector names (column `detector`)

| detector | model | input | depth | source |
|---|---|---|---|---|
| `A-d2`, `A-d6`, `A-d12` | `resmlp` (common3.DeepMLP) | A | 2 / 6 / 12 | abc3 fits |
| `C-d2`, `C-d6`, `C-d12` | `resmlp` | C | 2 / 6 / 12 | abc3 fits |
| `A-linear`, `C-linear` | `linear` (abc2 final detectors) | A / C | 0 | key `A` / `C` in `results/abc2/scores/<pool>_seed<s>.npz` |
| `A-mlp-abc2`, `C-mlp-abc2` | `plain-mlp-abc2` (abc2 Step 10, 256-256 plain) | A / C | 2 | key `A-MLP` / `C-MLP` in `results/abc2/scores/<pool>_seed<s>_mlp.npz` |

Pools (`cond`): `D0`, `D2`. Seeds 0, 1, 2. Test sets (`test_set`): `train` (the pool's
train rollouts; M1 only), `eval_seen` (optimistic: picked the epoch and the
regularisation), `T0`, `T1`, `T2`. Stress sets: `stress_seen`, `stress_unseen`.

## eval3 outputs (results/abc3/)

Column sets copy abc2's files of the same name, plus the extra columns listed.

- `scores/<pool>_seed<s>_deep.npz` — exactly stage10_mlp's npz format
  (`results/abc2/scores/D0_seed0_mlp.npz`), with detector keys `A-d2 ... C-d12`: `K_m1`,
  `K_cp` (both the running sum, float32 (1000, 520)), `K_p`, `K_mu`, `K_xi`, `K_h` (16: the
  15 grid alphas then 0.1), `K_band01`, `K_R`, `K_t_end`, `K_n_d1`, `K_n_d2`; common arrays
  `rollout_ids, cond, seed, alphas, kept_len, y, cell, base_task, block, cal_rids,
  train_rids, eval_seen_rids, stress_seen_cut, stress_unseen_cut`, and `selections`
  (JSON string). A group whose four fits all failed has no keys; it is listed in the JSON
  string `failed`. These npz files feed `bootstrap2.paired_bootstrap` directly.
- `final_m1.csv` — abc2 `final_m1.csv` columns (`cond,seed,detector,detail,test_set,
  m1_pooled,m1_per_task,floor,n_cells,n,n_fail`) + `input,depth,model`. One row per
  (pool, seed, detector, test set) for all 10 detectors and all 5 test sets; reference
  rows recomputed from the abc2 npz score arrays (their eval_seen / T0 / T1 / T2 values
  must equal abc2's `final_m1.csv` / `mlp_final_m1.csv`; eval3 checks this).
- `cp_curves.csv` — abc2 `cp_curves.csv` columns + `input,depth,model`; deep detectors
  computed here (eval_seen, T0, T1, T2; 16 alphas; cut variant); reference rows copied
  from abc2's `cp_curves.csv` / `mlp_cp_curves.csv` with the detector renamed.
- `stress.csv` — abc2 `stress.csv` columns + `input,depth,model`; references copied likewise.
- `final_table.csv` — abc2 `final_table.csv` columns + `input,depth,model,n_params`
  (mean over seeds of the chosen fits' parameter counts; linear = d + 1). One row per
  (pool, test set, detector), test sets train / eval_seen / T0 / T1 / T2 (alarm columns
  empty for `train`). Seed mean / std as pipe2_common.mean_std (ddof 1). This is the
  `--check-table` for bootstrap2's identity check.
- `selections.csv` — one row per (pool, seed, input, depth): `cond,seed,input,depth,
  detector,a_layer` (A's abc2 layer; empty for C), `p,wd` (chosen), `best_epoch,
  epochs_run,eval_seen_best` (best-epoch eval_seen M1-per-task), `lr` (1e-3, or 3e-4 after
  a retry), `status` (ok / failed), `n_params`.
- `grid.csv` — one row per fit (144): `cond,seed,input,depth,detector,p,wd,chosen,status,
  diverged,lr,n_attempts,best_epoch,epochs_run,stopped_early,seconds,n_params,
  n_train_rows` + `train_m1_per_task, eval_seen_m1_per_task, T0_m1_per_task,
  T1_m1_per_task, T2_m1_per_task, T0_m1_pooled` at the fit's best epoch.
- `train_log.csv` — one row per (fit, epoch): `cond,seed,input,depth,detector,p,wd,
  chosen,epoch,train_loss,lr_end,eval_seen_m1_per_task,eval_seen_m1_pooled,
  train_m1_per_task,train_m1_pooled`.
- `eval_check.json` — reference-row identity check (max |diff| vs abc2 tables, passed).
- `eval_status.json` — `{"status": "done"|"failed", "reason", "groups_failed": [...], ...}`.

## paired3 outputs (results/abc3/)

- `paired_T0.csv` — bootstrap2's `paired_<test>.csv` columns (test_set, comparison,
  cond_x, cond_y, detector, detector_x, detector_y, metric, n_seeds, n_boot, n_valid,
  rng_seed, x_point, y_point, diff_point, diff_mean, diff_ci_lo, diff_ci_hi, x_boot_mean,
  y_boot_mean, improvement, significant_worsening, verdict, x_point_per_seed,
  y_point_per_seed) + `family` (`vs_linear`, `vs_depth2`, `D2_vs_D0`), `input`, `depth`,
  `pool` (empty for D2_vs_D0). `detector_x` / `detector_y` use the names above (e.g.
  `C-d6` vs `C-linear`). Metrics: `m1_per_task`, `bal_acc`, `tpr`.
- `paired_T0_per_seed.csv`, `paired_T0_check.json` (identity check of every side
  against `final_m1.csv` / `cp_curves.csv` / `final_table.csv` above).
