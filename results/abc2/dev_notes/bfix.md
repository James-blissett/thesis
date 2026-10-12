# bfix: joint-H fit for Stage 5 (B), D0 check retry

Agent `bfix`, 2026-10-12.

## Why
The first real run (logs/stage5_b.log) failed the Step 11 D0 check. It reproduced a T0 M1-per-task of 0.6968 against abc's 0.7086 (tolerance ±0.005). The final c came out as 1e-6, where abc chose 1e-3.

The cause: stage5_b fitted only the 4 H=10 outputs per layer, but abc's setup_b fitted 12 outputs per layer (4 targets × H ∈ {0, 10, 25}) in one LBFGS problem. LBFGS stops on the whole problem (max_iter / tolerance), so the two fits stop at different points.

## Change (analysis/abc2/stage5_b.py only)
- **New flag `--joint-h`** (default on; `--no-joint-h` restores the old behaviour).
  - Each layer's block is fitted exactly as setup_b did: 12 outputs in setup_b's order (target-major, then H 0/10/25), the same warm start over sorted C_GRID, `fit_grid_block` / `fit_multi` unchanged.
  - Only the H=10 columns are used: AUROCs, thresholds, pass-2 probe outputs and everything downstream.
  - The H=0 and H=25 outputs are never saved or reported.
- **Grid c** is still chosen on the H=10 probes only.
- **Resume safety:** work pkl/npz files and b_final.pkl now record `fit_hs`. Old H=10-only work files are never reused, and the old ones were deleted before the run.
- **b_status.json** gains `fit` (joint_h, fit_block_H, kept_H, note), and `d0_check.fit` repeats it.

## Verification (D0, seed 0, layers 0/16/28/31, every c; vs results/abc/b_grid.csv, which is rounded to 4 d.p.)
| Grid | eval_seen max abs diff | T0 max abs diff |
|---|---|---|
| Old H=10-only, c ≤ 1e-2 | 1.7e-4 | 6.3e-4 |
| Old H=10-only, c = 1 | 4.0e-3 | 1.25e-2 |
| Joint, c ≤ 1e-2 | ≤ 5e-5 (rounding) | ≤ 1.9e-4 |
| Joint, c = 1 | 5e-5 | 1.1e-4 |

## Full run
**Command:**
```bash
ABC2_GPU_FRAC=0.95 OMP_NUM_THREADS=6 python analysis/abc2/stage5_b.py --cond D0 D2 --n-jobs 6
```
- **Log:** `logs/stage5_b_joint.log`.
- **Runtime:** 2965 s (49 min) on the free GPU.
  - Pass 1 took about 74 s per layer.
  - Pass 2 took 14 min.
- **Grid c:** 1e-4 for both D0 and D2.
  - D0 eval_seen target AUROC: 0.8632 at 1e-4 and 0.8598 at 1e-5, the same as abc.
- **Final c:**
  - D0: 1e-3, eval_seen M1-per-task 0.7816, both as abc.
  - D2: 1e-2.

**D0 check: PASSED.** T0 M1-per-task:

| | Mean ± std | Pooled |
|---|---|---|
| Reproduced | 0.708642 ± 0.0467 | 0.7278 |
| Reference | 0.7086 ± 0.0467 | 0.7278 |

- **Diff:** +4e-5.
- **BA at α = 0.1:** 0.5171, the same as abc's CP curve.
- **Record:** the result is written to `b_status.json["d0_check"]`, together with the `fit` note.

**D2 on T0:**
- **M1-per-task:** 0.7055 ± 0.0175.
- **BA at α = 0.1:** 0.5251 ± 0.0325.
- **TPR:** 0.149.
- **FAR:** 0.099.

**Rewritten outputs:** `b_*` and `scores/*_b10.npz`.

**Cleanup:**
- The work files in `/data/tmp/abc2_scores/b/{D0,D2}` (0.59 GB) were deleted after the run, so a re-run recomputes everything (about 50 min).
- The scratch directory was deleted.

**For run_all.sh:** the same command as before. `--joint-h` is the default.
