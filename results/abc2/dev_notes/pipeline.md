# pipeline (Stages 2, 3, 4, 6 port + Step 8 regression gate)

Agent `pipeline`, 2026-10-11. **Regression gate: PASSED. All six numbers match results/abc to 4 d.p. (diff 0.0000).**

## Files written
- `analysis/abc2/pipeline.py`: the entrypoint. Its docstring is the reference for the output formats, including the npz.
- `analysis/abc2/pipe2_common.py`: run context (paths, Index2, kept mask, splits, the condition's sets and the test sets), constraint/baseline/FailureSpot loading with per-rollout paths, `m1_safe`, IO.
- `analysis/abc2/stage2_constraints.py`: constraints alone, plus the beta and emb-layer selections.
- `analysis/abc2/stage3_c.py`: C.
- `analysis/abc2/stage4_a.py`: A.
- `analysis/abc2/stage6_cp.py`: conformal prediction (cut variant), final M1, stress.
- `analysis/abc2/regression_gate.py`: the Step 8 gate.
- `results/abc2/regression_gate.json` and `results/abc2/regression_gate.md`.

Nothing else was written under results/abc2. Scratch (`/data/tmp/abc2_dev/pipeline`, peak ~60 MB) has been deleted. `data2.py` was not edited. One workaround: `data2.load_split2` reads the fixed `results/abc2/splits/`, so `pipe2_common.load_split_from(splits_dir, ...)` does the same thing with the directory as a parameter.

## What is imported vs copied
**Imported unchanged:**
- `prep`: ffill, ema, smooth, fit_scale_orient, apply_prep, model_input, BETAS.
- `setup_c`: fit_prob, running_sum, seed_features, C_GRID, FAMILIES, base_of.
- `setup_a`: fit_and_score, fit_logreg (via fit_and_score), PROBES, N_PERM, PERM_SEED.
- `cp_eval`: fit_band, band_h, evaluate_cp, ALPHAS.
- `data` (via data2): m1, m1_cut, kept_mask, train_rows_and_weights, train_kept, open_layer(cache_dir).

**Copied with per-rollout paths:**
- `prep.load_series`, `constraint_m1.baseline_series` and `failurespot_series`.
- `setup_a.load_probe_input`, with the cache dir as a parameter.

**Selection logic re-implemented verbatim** (it read results/abc CSVs or hard-coded the choice).

## Selection rules found in analysis/abc and applied per condition (eval_seen only)
- **beta per family:** `setup_c.choose_betas`. For each family and beta, take the best series by eval_seen M1-per-task (mean over seeds, rounded to 4 d.p. because abc read it back from the CSV). The family's beta is the one whose best series scores highest.
- **emb_temp comparison layer (how emb_temp[28] arose):** `constraint_m1.summarise` took the layer with the best eval_seen **pooled** M1 (mean over seeds) at each beta. setup_c and cp_eval then hard-coded that layer at the chosen emb_temp beta ("none"). Applied as: argmax over layers of mean eval_seen pooled M1 at the chosen emb_temp beta.
  - NB: by eval_seen per-task M1, D0's choice would be layer 26. Handoff v7's "select on per-task" rule post-dates this choice. I followed the code that produced 28.
- **c for C's probes, c for C's final and c_A:** the best mean over seeds of eval_seen M1-per-task. For the probe c's this is the mean over 33 or 34 probes. Ties go to the first c in C_GRID.
- **A's probe:** chosen per seed, as the best eval_seen M1-per-task at c_A.
- **One choice per condition:** beta, the emb layer and the c's are chosen once per condition over the seeds in `--seeds`, as abc did across seeds. A's probe is the only per-seed choice.
- **Unstable flags:** a constraint is unstable when its sign differs across those seeds.
- **Where they are recorded:** all choices go to `selections.csv` (cond, seed, selection, value, criterion, score_eval_seen) and into each npz.

## Scope and deviations
- **Dropped series:** xl_final_hN and the _nosink variants are not in Stage 2. Step 6 does not recompute them for v3, and they never enter a selection.
- **Dropped C models:** no leave-one-family-out ablations and no emb_temp[28] 1-input probe.
- **Cut variant only** for conformal prediction.
- **Calibration rollouts on eval_seen:** conformal prediction is also reported on eval_seen. Its successes include the calibration rollouts.
- **Per-probe probabilities are not stored** (to save /data disk). Every probe's metrics on every set are stored. A's chosen probe is refitted with the same call and the same warm-start order. Refit drift against the grid was 0.0 in every test.
- **Shuffled control:** reproduces abc's rng stream. Earlier seeds' permutations are burned first.
- **m1_safe:** if M1 raises (no valid row under a cut, or no qualifying cell), it records NaN with an `error` and does not crash. No errors occurred on D0.
- **Stress cut:** the base task's unperturbed-cell cut from the same seed. It comes from the condition's eval_seen unperturbed rollouts for seen base tasks, and from T1 for the split's unseen tasks. The output is the share of stress rollouts with an alarm at a kept row t < cut, at alpha 0.1, given per stress set and per block (all / swap / task / demoted).

## Regression gate (D0 on T0, mean over seeds; the references come from results/abc/final_table.csv and cp_curves.csv, which match the spec table)
| Quantity | Reference | Port | Tolerance |
|---|---|---|---|
| emb_temp[28] M1-per-task | 0.7250 | 0.7250 | ±0.005 |
| C final M1-per-task | 0.7169 | 0.7169 | ±0.005 |
| A final M1-per-task | 0.7163 | 0.7163 | ±0.005 |
| emb_temp[28] balanced accuracy, α=0.1, cut | 0.5559 | 0.5559 | ±0.01 |
| A balanced accuracy, α=0.1, cut | 0.5368 | 0.5368 | ±0.01 |
| C balanced accuracy, α=0.1, cut | 0.5009 | 0.5009 | ±0.01 |

**Inputs:**
- A v2-only index (`{"rollouts": data2.v2_entries()}`).
- `data2.make_split2` splits. These are identical to results/abc/splits for all 3 seeds (train, eval_seen, unseen, unseen_tasks), and stress is empty.
- The old cache /data/tmp/abc_features.

**Every selection matches abc:**
- betas: action 0.96, emb none, cross 0.91.
- emb layer: emb_temp[28].
- c: C probes 1, C final 1e-5, A 1e-5.
- A probes: layer01, layer10, layer02.

**Per-seed comparisons (all within abc's 4-d.p. rounding):**
- Constraint M1: 2508 series×β×seed rows, max |diff| 5e-5, 0 sign strings differ.
- C probes: max |diff| 5e-5.
- A's 34 probes: max |diff| 5e-5.
- CP balanced accuracy at all 16 α × 3 seeds × 3 detectors: max |diff| 0.0000.

## Multi-condition functional smoke test (not a result)
**Setup:**
- 120 of the 300 v2 rollouts were relabelled as ori2/obj/lan/swap/task (marked v3, with symlinked constraints and rollouts), so the cache row order was kept.
- Ran D0, D1 and D2 × seeds 0 and 1, with A restricted to 3 probes via the test-only env var `ABC2_PROBES`.

**Results:**
- Every stage ran.
- T1/T2 cells and stress cuts work.
- Recomputing the stress alarms from the npz alone (recipe in the docstring) reproduces stress.csv exactly.

## Commands for run_all.sh
```bash
# Step 8: already PASSED on the old cache. Re-running it needs /data/tmp/abc_features, so
# skip it once results/abc2/regression_gate.json has "passed": true.
python analysis/abc2/regression_gate.py
# Step 9 (after corpus_v3_index.json, results/abc2/splits/, constraints_v3/ and
# /data/tmp/abc2_features/ exist). Resumable; re-run the same command after a crash.
ABC2_GPU_FRAC=1.0 OMP_NUM_THREADS=4 python analysis/abc2/pipeline.py --cond D0 D1 D2 --n-jobs 4
```
- Defaults: `--index corpus_v3_index.json --cache-dir /data/tmp/abc2_features --splits-dir results/abc2/splits --out-dir results/abc2 --work-dir /data/tmp/abc2_scores/work --demotions results/abc2/demotions.json --seeds 0 1 2`.
- `--until 2|3|4|6` stops early.
- If the cache dir has `rollout_ids.json` (list, or {"rollout_ids": [...]}), its order is checked against the index.

## Outputs (all per condition under results/abc2/tables/<cond>/, concatenated into results/abc2/<name>.csv)
- **Headline tables:** `final_table.csv` (cond × test_set × detector: M1 pooled and per-task mean±std, floor, n_cells per seed, and at α=0.1 bal_acc, TPR, FAR and detection time mean±std, plus n_cal), `final_m1.csv`, `cp_curves.csv` (16 α, cut), `stress.csv` and `stress_summary.csv`, `selections.csv`.
- **Stage tables:** constraint_m1(+_summary), c_betas, c_grid, c_grid_choice, c_probes, c_final, c_final_weights, a_grid, a_grid_choice, a_probes, a_final, a_shuffled.
- **Long format:** these use cond, seed and test_set columns, where test_set ∈ {eval_seen, T0, T1, T2}.
- **Detectors** are `A`, `C` and `emb` (the condition's chosen emb_temp layer). The `detail` column names the probe or layer.
- `timings.csv`.
- **`scores/<cond>_seed<s>.npz` (for the bootstrap, report and MLP agents).** Contents:
  - rollout_ids in index order; y; kept_len; cell; base_task; block; train/eval_seen/cal rids.
  - alphas: the 15 grid values, then 0.1.
  - For D in A, C, emb:
    - `D_m1`: the per-row score M1 maximises.
    - `D_cp`: the CP series (= D_m1 for A and C; the running max for emb).
    - `D_p`: P(failure), A and C only.
    - Band and calibration: `D_mu`, `D_xi`, `D_h[16]` (inf = never alarms), `D_band01`, `D_R`, `D_t_end`, `D_n_d1`, `D_n_d2`.
  - stress cuts; `selections` (JSON).
- **Alarm recipe:** `alarm = (1<=t<kept_len) & (t<cut) & (D_cp > D_mu + D_h[j]*D_xi)`. The cut is the minimum kept_len in the rollout's cell within the evaluated set. ~3 MB per file at N=300, so ~10 MB at N=1000.

## Runtimes
**Measured, D0, N=300, 3 seeds.** The GPU was shared with both collection workers at 100% utilisation and capped at 6 GB.

| Stage | Time | Notes |
|---|---|---|
| Stage 2 | 30 s | |
| Stage 3 | 72 s | 4 CPU jobs |
| Stage 4 grid | 3797 s | ~112 s/layer, 171 s of that reading the cache |
| Stage 4 finals and shuffled control | 260 s | |
| Stage 6 | 1.5 s | |

abc's own Stage 4 on a free GPU took 383 s in total, so the contention cost about 10×.

**Extrapolation for 3 conditions × 3 seeds, N≈1000, free GPU.** Fit cost scales with train rows: D0 1×, D1 ~1.7×, D2 ~3×. Each layer (4.3 GB) is read once for all 9 (cond, seed).

| Stage | Estimate |
|---|---|
| Stage 2 | ~5 min |
| Stage 3 | ~15 min |
| Stage 4 grid | ~45 min (~80 s/layer) |
| Stage 4 finals | ~12 min |
| Stage 6 | <1 min |
| **Total** | **~1.3 h (±50%)** |

If collection is still using the GPU, multiply Stage 4 by about 10.

**Resource needs:**
- **GPU:** at N=1000 a layer on the GPU plus D2's ~170k train rows needs about 10–12 GB peak. **Run with ABC2_GPU_FRAC=1.0, not the 6 GB default.**
- **RAM:** under 5 GB.
- **Disk:** about 15 MB per (cond, seed) of work files in /data/tmp/abc2_scores/work (deletable after the run, but then a re-run recomputes), and about 100 MB of npz in results/abc2/scores.

## Unresolved / for the lead
- **The emb-layer rule** (pooled, as the code did, vs per-task, as handoff v7 says) is the one judgement call. It reproduces 28, and it is logged in `selections.csv`.
- **Unchecked failure mode on real v3 data:** a cell with very short successes could leave act_dir with no valid row before the cut. `m1_safe` would then give NaN for that series and set the error field, rather than crash. This path is untested on real v3 data.
- **Unchanged abc warnings:** cp_eval's fit_band prints "Mean of empty slice" and "Degrees of freedom" warnings (columns before 2 calibration rollouts are alive). abc printed the same.
