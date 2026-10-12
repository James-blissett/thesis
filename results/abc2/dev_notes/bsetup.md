# bsetup: Stage 5 (B) at H = 10 for D0 and D2 (Step 11)

Agent `bsetup`, 2026-10-11.

**Status:** code is ready and tested. Downstream of the grid, abc's D0 B H=10 numbers are reproduced exactly. The real D0/D2 runs are not done yet; they need the pipeline's real outputs and a free GPU.

## Files written
- **`analysis/abc2/stage5_b.py`**: the only code file. Its docstring covers what it does, the outputs and usage.
- **This note.**
- **Nothing under `results/abc2/` apart from this note.** All tests wrote to `/data/tmp/abc2_dev/bsetup/`, which has been deleted. All of my tmux sessions have exited.

## What it does

### Imports instead of copies
- **From `setup_b`:** `lookahead`, `two_means_threshold`, `balanced_weights`, `fit_multi` (via `fit_grid_block`), `predict_block`, `target_auroc`, `target_name`, `probe_name`.
- **From `setup_c`:** `fit_prob`, `running_sum`, `C_GRID`.
- **From `cp_eval`:** `fit_band`, `band_h`, `evaluate_cp`.
- **From `stage6_cp`:** `ALPHAS`, `CAL_SPLIT_SEED`, `stress_cuts`, `alarms`.
- **From `stage4_a`:** `gpu_setup`, `load_probe_input`, `check_cache`.
- **From `pipe2_common`:** `Ctx`, `m1_safe`, `m1_row`, IO.

### Copied
- **`setup_b.prepared_targets`:** re-implemented for an Index2, a condition's train set, and the pipeline's betas.
- **The single-detector part of `stage6_cp.run_seed`.**
- **`pipeline.assemble`'s final-table aggregation.**

### Inputs taken from the pipeline, per condition
- **β per family:** from `<pipe>/selections.csv` (`beta_action`, `beta_emb_temp`, `beta_cross_layer`).
- **Unstable flags:** from `<pipe>/tables/<cond>/constraint_m1_summary.csv`: primary series flagged `unstable=True` at the family's chosen β. This is setup_b's rule.
- **Example:** on D0 this skips `emb_temp[1]`, `emb_temp[15]`, `xl_final[17]` and `xl_final[18]`, leaving 124 probes, as in abc.
- **If they are missing:** a condition without these files is skipped, and the reason goes to `b_status.json`.

### The grid
- **Pass 1:** refit on all kept train rows at every c (warm-started). Then own-target AUROC on eval_seen, T0, T1 and T2.
- **Grid c:** the best mean over seeds of the probe-mean eval_seen AUROC.
- **Pass 2:** at the grid c, out-of-fold outputs with `GroupKFold(5)`, plus a refit for every other rollout the condition uses.
- **Layer reads:** each layer is read once per pass for all conditions together.

### Final detector
- **Inputs:** the 124 out-of-fold probabilities (stored as fp16, as abc did).
- **Training:** `train_rows_and_weights`, with the running sum as the score.
- **c:** the best mean eval_seen M1-per-task.
- **Ablations:** per-target ablations at that c.

### Conformal prediction and stress
- **CP:** cut variant only, 15 α + 0.1, calibrated on the condition's eval_seen successes.
- **Stress:** at α = 0.1, using stage6's cut rule.

## Outputs
All of these go to `<out>`, which defaults to `results/abc2`.

- **Merging:** in every CSV, rows of other conditions already in the file are kept, and rows of the conditions just run are replaced.
- **`b_final_table.csv`:** same columns as the pipeline's `final_table.csv`, with detector `B H=10`.
- **Per-condition tables:** `b_final_m1.csv`, `b_cp_curves.csv`, `b_stress.csv` and `b_selections.csv`, in the formats of the pipeline's tables with the same names.
- **`b_grid.csv`:** long format. Columns are cond, seed, layer, target, target_series, H, c, chosen_c, test_set (eval_seen/T0/T1/T2), auroc and threshold.
- **`b_final_grid.csv`:** the final detector at every c.
- **`b_ablation.csv`:** the final detector on each target's probes alone.
- **Score files:** `scores/<cond>_seed<s>_b10.npz`. Same keys as the pipeline's npz, with detector key `B`: `B_m1`, `B_cp` (both the running sum), `B_p`, `B_mu`, `B_xi`, `B_h`, `B_band01`, `B_R`, `B_t_end`, `B_n_d1`, `B_n_d2`. Also the shared keys, `stress_*_cut`, and `selections` as JSON.
  - **Glob hazard:** the pattern `D0_seed*.npz` also matches these files. Consumers should glob `<cond>_seed?.npz`.
- **`b_status.json`:** contains `status` (done | failed | skipped), `reason`, `d0_check`, `conds_done`, `conds_skipped`, `full_run`, `seconds` and `selections`.
  - **Partly skipped runs:** if one condition was skipped (its pipeline outputs were missing) and the other ran, the status is "done" and the reason names the skipped condition.
- **Work files:** in `/data/tmp/abc2_scores/b/<cond>/`. These are pass-1 pkls, pass-2 fp16 npz files (only the rollouts the condition uses) and `b_final.pkl`.
  - **Resuming:** the run resumes from these files. Each is redone if β, the target set or the grid c changed.
  - **Size:** about 0.6 GB at N≈1000 for D0+D2. They can be deleted after the run.

## Commands for run_all.sh
Run these after `pipeline.py --cond D0 D1 D2` has finished, so that `results/abc2/selections.csv` and `tables/D0` and `tables/D2` exist. They must not block the report.
```bash
ABC2_GPU_FRAC=1.0 OMP_NUM_THREADS=4 python analysis/abc2/stage5_b.py --cond D0 D2 --n-jobs 4 \
    > results/abc2/logs/stage5_b.log 2>&1 \
  || python analysis/abc2/stage5_b.py --mark-failed "stage5_b.py exited non-zero (see logs/stage5_b.log)"
python analysis/abc2/stage5_b.py --check-d0 || true    # prints and records the D0 comparison
# if there is no time for B:  python analysis/abc2/stage5_b.py --mark-skipped "time"
```
- **Exceptions:** the script catches its own exceptions, writes `failed` with the reason, and exits 1. The `--mark-failed` fallback covers kills and OOM, and keeps a more specific reason if one was already written.
- **Re-running:** the same command resumes.
- **D0 check:** after a full D0 run (32 layers, seeds 0–2) the check runs automatically. `--check-d0` repeats it from `b_final_table.csv` and exits 1 if it fails.

## Tests
All tests used a v2-only index, `make_split2` splits (identical to `results/abc/splits`), the old cache, and the pipeline's D0 Stages 2–3 run into scratch. That run reproduced abc's selections: betas 0.96 / none / 0.91, and the 4 unstable targets.

- **(a) GPU vs scikit-learn (`--check 1e-4`):** layer 16, `emb_temp[16]`, H=10, seed 0.
  - T0 target AUROC is 0.7504 on the GPU and 0.7504 with scikit-learn (diff 0.0000). **PASSED.**
  - That value equals `results/abc/b_grid.csv` seed 0 unseen.
- **(b) Grid against `results/abc/b_grid.csv`:** layers 0, 16, 28 and 31 × seeds 0–2 × 6 c, H=10.
  - **At c = 1e-4 (abc's grid c):** max |diff| is 0.0004 on eval_seen and 0.0013 on T0.
  - **At c ≤ 1e-2:** max 0.0015, mean 1e-4.
  - **At c = 1 (never chosen):** up to 0.013. LBFGS hits `max_iter` at the weak penalty, and abc fitted 12 outputs (3 H) jointly where this fits 4, so the two stop at different points.
- **(b') Pass 2 against abc's stored per-row outputs:** layer 16, c = 1e-4, all seeds. Max |Δp| is 0.004 and the mean is 2–4e-4, on both out-of-fold train rows and refit rows.
- **(b'') Final detector, CP and tables on abc's own H=10 probe outputs** (`/data/tmp/abc_scores/b_probes_h10_seed*.npz`):
  - final c = 1e-3, as abc.
  - eval_seen M1-per-task is 0.7816 (abc 0.7816).
  - **D0 check:** T0 M1-per-task is 0.708642 against a reference of 0.7086. **PASSED.**
  - The CP cut curves (16 α × 3 seeds: BA, TPR, TNR, detection time) match abc's `cp_curves.csv` "B H=10" cut within 5e-5 (abc rounds to 4 d.p.). BA at α=0.1 is 0.5171, the same as abc.
- **Stress path:** tested with fake stress sets. Alarms recomputed from the npz alone equal `b_stress`.
- **Resume, skip and fail:** re-running does no GPU work. A condition with no pipeline outputs becomes `skipped`. `--mark-failed` works.
- **Not tested:** a full 32-layer D0 run end to end. That is the real run's D0 check.
  - **Expected result:** grid c should again be 1e-4. abc's H=10-only grid scores are 0.8632 at 1e-4 against 0.8598 at 1e-5, a margin far larger than the 1e-3 refit drift.
  - **Expected D0 check:** given (b'), the D0 check should pass well inside ±0.005.

## Runtime estimate (D0 + D2, N≈1000, free 46 GB GPU)
- **What I measured here (GPU shared with collection, capped at 6 GB, N=300, D0, 3 seeds):** pass 1 took 32–430 s per layer and pass 2 about 60 s per layer. abc's free-GPU log shows 26 s and 8.5 s per layer. So the slowdown is about 7–16×, and it varies with the collection workers' load. The final detector took 26 s for 18 fits with 124 inputs.
- **Scaling:** D0's train set is unchanged (about 53k rows), so its fits cost about the same as abc's. D2 has about 3× the rows (about 170k).
- **Per layer, both conditions, 3 seeds:**
  - pass 1: about 3 × (8 + 25) ≈ 100 s;
  - pass 2: about 3 × (3 + 9) ≈ 36 s;
  - two reads of a 4.3 GB layer: about 10–40 s.
- **Grid total:** 32 layers come to about 75–95 min.
- **Final detectors:** about 1 min for D0 and about 5–10 min for D2 (sklearn, 3–4 jobs). CP takes seconds.
- **Total: about 1.5 h (range 1–2.5 h).** If the GPU is still shared, multiply by about 10.
- **GPU memory:** about 10 GB peak (a 4.3 GB layer, plus D2's train rows in fp32 twice during standardisation). **Use `ABC2_GPU_FRAC=1.0`.**
- **RAM:** under 6 GB.

## Unresolved / for the lead
- **Fitting H=10 alone vs all three H jointly:** H=10 alone (as asked) shifts probe AUROCs by up to about 1e-3 compared with abc's joint 3-H fit. Downstream code is exact. If the D0 check ever fails, this is the first suspect. The fix would be to fit the H=0/25 targets alongside and discard them; that is not implemented.
- **Unchanged abc warnings:** cp_eval prints "Mean of empty slice" and "Degrees of freedom" warnings, as abc did.
