# Report notes (hand-written; report.py puts these sections into REPORT.md)

## Headline

1. On the headline setup (full training set, unseen tasks incl. LIBERO-PRO), per-task-and-perturbation AUROC is 0.780 ± 0.028 for A's mean, 0.749 ± 0.026 for C's mean and 0.748 ± 0.017 for B's mean (H = 10). No AUROC gap between the three is separable (B − A −0.032 [−0.065, +0.003]; A − C +0.031 [−0.011, +0.073]; B − C −0.001 [−0.038, +0.033]). On balanced accuracy at α = 0.1 all three gaps are separable: C's mean leads (0.607), then A's (0.573), then B's (0.555).
2. B's pre-registered embedding-temporal mean (0.753 ± 0.017 at H = 10) is level with A's mean on AUROC at every H (−0.024 to −0.027, every interval includes zero). It is below both A's and C's means on balanced accuracy at every H. Dropping the xl_final probes (exploratory) does not raise B (0.744 at H = 10).
3. With B compared on the 113 (layer, target) probes common to both grids, the full training set changes B's mean by −0.005 to −0.021 AUROC, and no interval excludes zero. Of the primary detectors, only A's mean rises separably with more training data (+0.026 [+0.001, +0.060]).
4. The LIBERO-PRO detection comes mostly from the instruction-perturbation cells, where A, C and B reach 0.82–0.83 against 0.70–0.75 on the unperturbed cells. The object-perturbation cells are few and noisy (2–3 cells, 6–27 successes per seed), and there B's mean is below A's (−0.081 [−0.179, −0.010] at H = 10).
5. B is robust to smoothing: across the four settings its mean moves by at most 0.024 and its embedding-temporal mean by at most 0.014, with no AUROC interval excluding zero, although its probe count ranges from 74 to 116. C's mean falls separably once emb_temp is smoothed (−0.029 at β = 0.83, −0.059 at β = 0.96 everywhere), as does the best untrained constraint (−0.043, −0.112), so the sensitivity runs through the emb_temp family, which the selected setting leaves unsmoothed.

## Decisions and deviations

Decisions for James (nothing blocks):

1. **F1 uses only rollouts that have frames.** The new rollouts (new unperturbed and LIBERO-PRO, `/data/rollouts_v3/`) saved only their first camera frame, and `obs.npz` holds no images. A frame strip is therefore possible only for the original LIBERO-10 rollouts of the headline test set. The example is seed 0, task 9's unperturbed cell: failure `task09_trial18` (B alarms at step 38, A at 52, C at 5, cut 244) and success `task09_trial26` (no detector alarms). The cut and band are the headline set's, from the full training set. It was the only qualifying framed pair in seed 0, and its stall is short (9 repeated actions before B's alarm). The alternative is to re-render frames for a LIBERO-PRO rollout by replaying its saved actions from its initial state in the LIBERO-PRO simulator. That is new collection work, so I didn't do it. The rule and candidates are in `figures/data/example_rollouts.json`.
2. **Check 3, items 2 and 3a are the same rollouts.** On the LIBERO-PRO test set the unperturbed cells are exactly the original + new unperturbed rollouts, with the same cells, so the two columns are identical by construction. Both are reported.
3. **Smoothing check, small shifts in unchanged targets.** Where a setting leaves a target's series unchanged (emb_temp under "no smoothing"), B's probe outputs still shift slightly (|AUROC difference| ≤ 0.002). B's GPU fit optimises a layer's outputs jointly, and dropping act_mag changes the joint stopping point. "Current" uses the original output sets and reproduces every headline row exactly.
4. **β = 0.83 everywhere makes 18 of B's emb_temp targets sign-unstable** (layers 10–27). In that setting B's embedding-temporal mean averages 14 probes, not 29. Under "no smoothing" and "β = 0.96 everywhere", act_mag is sign-unstable and leaves the grid at every layer, giving 93 and 74 probes.

Deviations and notes:

- **No refits outside Check 4.** T1, T4 and T5 are the full training set's run tables (`results/thesis_results_full/tables/`) re-cut to the headline test set. T3 rereads the Stage 2 table, adding the pooled AUROC and floor on the headline test set. The paired comparisons, Checks 1–3 and F1 recompute only from saved per-row scores and grid outputs.
- **New detectors** are unweighted means of saved probe outputs: B without xl_final (Check 1), and B on the common probe set, overall and per target, for both training sets (Check 2). Their per-row scores are in `scores/`.
- **F1 frame order:** frames are now placed left to right in time; the failure's alarm (step 38) comes before its mid frame. The main run's F1 was already in order.
- **Figure relabelling:** F5–F8 and F10 reuse `analysis/thesis_results/figures.py` with the headline test set. Their labels are rewritten to "per-task-and-perturbation AUROC" and "unseen tasks incl. LIBERO-PRO" before saving. F6–F8 keep one shared y-range, now fitted to the headline and eval-seen values.
- **F3, F11 and T2 are copied unchanged** from `results/thesis_results/`. F3(a) shows the original training set's kept rows, as specified there; the full-set version is in `results/thesis_results_full/figures/`.
- **"900 trainable rollouts"** is the corpus without the 100 stress rollouts. Each seed trains on 369–390 seen-task rollouts and calibrates on 84–101 successful eval-seen rollouts (`tables/methods_values.csv`).
- **File name in the handoff:** it cites `results-collection-handoff.md`; the repo file is `results_collection_handoff.md`.
- **Disk:** `/data` is 96% full (23 GB free). The follow-up's work files (`/data/tmp/thesis_results_work/final/`, 5.0 GB) can be deleted once this is committed; re-running rebuilds them.

## Proposed commit message

```
results follow-up: headline setup (full training set, LIBERO-PRO test set), checks 1-4, methods values
```
