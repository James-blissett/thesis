# mlp: Step 10, the MLP ablation (A-MLP, C-MLP) for D0 and D2

Agent `mlp`, 2026-10-11. **Status:** the code is ready and was tested end to end in scratch on the old cache (D0 for real; D2 through the relabelling trick). The real run has not been done; it needs the pipeline's real outputs.

## Files written
- **`analysis/abc2/stage10_mlp.py`:** the only code file. Its docstring covers what it does, the outputs and usage.
- **This note.**
- **Nothing under `results/abc2/` apart from this note.** All tests wrote to `/data/tmp/abc2_dev/mlp/`, which has been deleted. All of my tmux sessions (`md-agent-mlp-pipe`, `md-agent-mlp-pipe2`) have exited.

## Hyperparameters
SAFE's training code is **not** in the fork. In `/ephemeral/code/openvla`, the "add safe changes" commit only touches the rollout/eval scripts, `unc_utils.py` (token uncertainty) and `viz_utils.py`; there is no failure-detector or MLP training anywhere. So these are Step 10's fallback hyperparameters:

- **Architecture:** standardised input (unweighted mean/sd over the train rows; sd 0 → 1, as `setup_a.fit_and_score`) → Linear(d, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1), with PyTorch's default init.
- **Loss:** batch mean of w·BCE, where w are the existing row weights. These come from `data.train_rows_and_weights`, so the rows are the same as the linear models' (≥ 5-successes cut, 1/n_class(t), mean 1).
- **Optimiser:** AdamW, lr 1e-3, weight decay wd on every parameter, batch 4096, rows reshuffled every epoch.
- **Epochs and early stopping:** at most 50 epochs. After every epoch the script computes the eval_seen M1-per-task of the running-sum score. Training stops after 5 epochs without a strict improvement, and the best epoch's weights are kept.
- **Seeding:** `torch.manual_seed(split seed)` before the model is built, plus a CPU generator seeded with the split seed for the shuffles.
- **Weight decay:** wd ∈ (1e-4, 1e-2, 1e-1, 1). One per (condition, detector): the best mean over seeds of the best-epoch eval_seen M1-per-task. This is the plain argmax of the c rule, with ties going to the first value in grid order, so the smallest wd.

## Inputs
- **A-MLP:** the pooled hidden state at the pipeline's `A_probe` for that (condition, seed), read from `selections.csv` and loaded by `stage4_a.load_probe_input`. If A ever picked `action_tokens`, those 7 floats would be used instead.
  - Each layer is read once for every (condition, seed) that needs it.
  - The layer stays on the GPU in fp16, and batches are standardised on the fly.
- **C-MLP:** Stage 3's own code path. That is `stage3_c.smoothed_families` at the condition's betas (from `selections.csv`), then `setup_c.seed_features` on the seed's train set: 132 inputs in Stage 3's order.

## Scoring and evaluation
Scoring, conformal prediction and stress are exactly those of `stage6_cp` / `stage5_b`:
- the running sum of P(failure) (kept as float32, with no fp16 rounding);
- `m1_safe` on eval_seen, T0, T1 and T2;
- `cp_eval.fit_band` on the condition's successful eval_seen rollouts, with `default_rng(CAL_SPLIT_SEED + seed)`;
- 15 α + 0.1, cut variant;
- stress via `stage6_cp.stress_cuts` / `alarms`.

## Outputs
All go to `--out-dir`, default `results/abc2`. Rows of conditions not re-run are kept.

- **`mlp_final_table.csv`:** exactly `final_table.csv`'s columns, with detectors `A-MLP` and `C-MLP`.
- **`mlp_final_m1.csv`, `mlp_cp_curves.csv`, `mlp_stress.csv`:** the pipeline's formats.
- **`mlp_selections.csv`:** `selections.csv` format, with rows `A-MLP_input`, `wd_A-MLP`, `A-MLP_best_epoch`, `A-MLP_epochs_run`, and the same for C-MLP.
- **`mlp_grid.csv`:** every wd, with best epoch, epochs run and M1 on all sets.
- **`mlp_train_log.csv`:** per epoch, the train loss and eval_seen M1.
- **`scores/<cond>_seed<s>_mlp.npz`:** keys `A-MLP_{m1,cp,p,mu,xi,h,band01,R,t_end,n_d1,n_d2}`, the same for `C-MLP`, plus the common arrays, the `stress_*_cut` arrays and `selections` as JSON. The pattern `<cond>_seed?.npz` does not match these files.
- **Paired bootstrap:**
  - `mlp_paired_T0.csv`, `mlp_paired_T0_per_seed.csv`, `mlp_paired_T0_check.json`. These concatenate `bootstrap2 --pair` runs (A-MLP vs A, C-MLP vs C, per condition) and use the same resamples as `paired_T0.csv`.
  - The per-pair files are in `mlp_paired/`.
  - `report2` reads `mlp_final_table.csv`, `mlp_paired_T0.csv` and `mlp_paired_T0_check.json`, with `mlp_status.json` as the fallback.
- **`mlp_status.json`:** `status` (done/failed/skipped) and `reason`, plus `conds_done`, `wd`, `paired_T0`, `hyperparameters` and `safe_code_found: false`.
- **Work files:** `/data/tmp/abc2_scores/mlp/<cond>/mlp_<det>_seed<s>.pkl`.
  - **Resuming:** per (cond, seed, detector), and per wd inside each file. A file is redone if its input layer, betas, train set, eval_seen set or hyperparameters changed.
  - **Size:** about 100 MB at N≈1000. They can be deleted after the run.

## Commands for run_all.sh
Run these after `pipeline.py --cond D0 D1 D2` has finished.
```bash
ABC2_GPU_FRAC=1.0 OMP_NUM_THREADS=3 python analysis/abc2/stage10_mlp.py --cond D0 D2 \
    > results/abc2/logs/stage10_mlp.log 2>&1 \
  || python analysis/abc2/stage10_mlp.py --mark-failed "stage10_mlp.py exited non-zero (see logs/stage10_mlp.log)"
# no time for it:  python analysis/abc2/stage10_mlp.py --mark-skipped "time"
# redo only the bootstrap:  python analysis/abc2/stage10_mlp.py --cond D0 D2 --paired-only
```
- **Exceptions:** the script catches its own exceptions, writes `failed` with the reason, and exits 1. The `--mark-failed` fallback covers kills and OOM, and keeps an existing, more specific `failed` reason.
- **Missing prerequisites:** a condition without pipeline selections is skipped, and the reason goes to the status file.
- **Re-running:** the same command resumes and does no training.

## Tests (scratch; the GPU was shared with collection at 100% use, capped at 6 GB)
### Setup
- **Index and splits:** a v2-only index plus `make_split2` splits.
- **Cache:** the old cache.
- **Pipeline run:** `pipeline.py --cond D0` with `ABC2_PROBES=layer01,layer10,layer02`.
  - It reproduced the regression-gate numbers on T0: A 0.7163, C 0.7169, emb 0.7250 M1-per-task, with BA 0.5368 / 0.5009 / 0.5559.
  - A's probes were layer01, layer10 and layer02.

### Checks
- **Training:**
  - The train loss falls every epoch. Examples: A-MLP seed 2 went 0.554 → 0.055 over 13 epochs; C-MLP seed 0 went 0.614 → 0.258.
  - Eval_seen M1-per-task peaks early and then falls (overfitting), so early stopping picks epochs 1–8 and stops at epochs 6–13.
  - Every score and P is finite.
  - A full recompute gives a bit-identical `mlp_final_table.csv` (deterministic).
  - A resume does no training.
- **C inputs are Stage 3's:** a linear `fit_prob` on my inputs at D0's c_final (1e-5) reproduces the pipeline npz `C_p` with max |diff| 0 for all 3 seeds.
  - On D2 (c_final = 1, almost unregularised) it differs by 1e-2 in p. Its eval_seen M1 is identical, the gap is 0 at c = 1e-5, and 6e-8 at c = 1e-2.
  - So this is lbfgs numerics (a joblib worker vs in-process), not an input difference.
- **A-MLP input:** it equals the npz's `a_probe` for every seed.
- **bootstrap2 identity check:** PASSED against `mlp_final_table.csv` (seed means) and the pipeline's `final_m1` / `cp_curves` (linear side, per seed). Max |diff| was 4.7e-7 on D0 and 8.6e-7 on D2, against a tolerance of 1e-6.
- **Stress:** recomputing from the MLP npz alone (the pipeline docstring's recipe) reproduces `mlp_stress.csv` on 24 rows, max |diff| 4.3e-7 (CSV rounding). This was on the relabelled D2, which has stress sets.
- **D2 path:**
  - **Index:** the v2 index relabelled by position within each task (k mod 6: ori, ori, ori2, obj, lan, swap/task), keeping source v2.
  - **Run:** `pipeline.py --cond D2 --seeds 0 1` with 2 A probes, then stage10 `--seeds 0 1`.
  - **Results:** T1/T2 cells (T2 has 6/5 cells), stress, CP and the paired bootstrap all work.
- **Status paths:** skipped (no pipeline selections), failed (missing cache → exit 1 with the reason), `--mark-failed` (keeps a specific earlier failure) and `--mark-skipped` all work.

### D0 sanity numbers on T0
Dev run on the old cache, mean ± std over 3 seeds. The linear rows are from the same scratch pipeline run.

| Detector | M1-per-task | Pooled M1 | BA@0.1 | TPR | FAR | wd | Best epochs (s0/s1/s2) |
|---|---|---|---|---|---|---|---|
| A (linear) | 0.7163 ± 0.0320 | 0.7271 | 0.5368 | 0.086 | 0.012 | | |
| A-MLP | 0.6855 ± 0.0349 | 0.7152 | 0.5164 | 0.033 | 0.000 | 1 | 2 / 1 / 8 |
| C (linear) | 0.7169 ± 0.0489 | 0.7041 | 0.5009 | 0.027 | 0.025 | | |
| C-MLP | 0.7175 ± 0.0346 | 0.7476 | 0.5908 | 0.207 | 0.025 | 1e-4 | 5 / 1 / 1 |

- **Floor:** 0.5466; 37 calibration rollouts.
- **Paired T0 (dev):**
  - A-MLP − A: M1-per-task −0.028 [−0.096, +0.036], BA −0.020 [−0.048, +0.008]. No significant change.
  - C-MLP − C: M1-per-task +0.001 [−0.043, +0.039]; BA +0.085 [+0.048, +0.117] and TPR +0.174 [+0.115, +0.227], which count as an "improvement".
- **Not degenerate.** These are dev numbers only; the real run replaces them.

## Runtime (D0 + D2, N≈1000, free GPU)
- **Measured here (N=300, shared GPU):**
  - D0, 3 seeds, 24 fits: 23 s; about 1 s per fit, with 43–47k train rows and 6–13 epochs.
  - Relabelled D2, 2 seeds: 14 s.
  - The paired bootstrap took under 1 s.
  - Peak GPU memory 1.9 GB; RAM 3.5 GB.
- **At N≈1000:**
  - D2 has about 3× the train and eval_seen rows. Expect 3–5 s per fit, so 48 fits take about 2–4 min.
  - Reading up to 6 distinct 4.3 GB layers takes about 1–3 min.
  - Loading C's series for 1000 rollouts takes under 1 min.
  - **Total: about 5–10 min (allow 20).** If the GPU is still shared, multiply training by about 2–3.
- **Resources:**
  - **GPU:** peak is about 5–5.5 GB (a 4.3 GB layer plus about 0.6 GB). That is borderline under the 6/46 default, so **use `ABC2_GPU_FRAC=1.0`**.
  - **RAM:** about 8 GB.
  - **Disk:** about 55 MB of npz in `results/abc2/scores` (about 9 MB each) and about 100 MB of work files.

## Unresolved / for the lead
- **The wd grid is nearly inert.** AdamW's decoupled decay shrinks weights by lr·wd ≤ 1e-3 per step. With about 11–40 steps per epoch and early stopping at a few epochs, wd ≤ 0.1 barely changes anything; on D0 C-MLP all four wd tied, and the tie went to 1e-4. Only wd = 1 differs a little. This follows from Step 10's spec, so I left it as specified. Say so in the report if wd matters to the reader.
- **Fast overfitting.** The best epoch is often 1–2, especially for A-MLP (4096 inputs, about 45k highly correlated rows). Early stopping also selects on eval_seen, which is what wd is chosen on, so eval_seen numbers are optimistic for the MLP. Only T0/T1/T2 are clean.
- **A wide band on one seed.** On D0 seed 1, A-MLP's band has h(0.1) = 346, against 2–4 elsewhere, so it barely alarms. This is a property of the CP band on a near-binary P (p reaches 0.99999), not a bug, but it pulls A-MLP's TPR down.
- **Weight decay on biases.** It is applied to every parameter, biases included (PyTorch's default param group).
