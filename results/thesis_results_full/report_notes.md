## Headline

Training on the full training set (1,000 rollouts, including LIBERO-PRO object and instruction perturbations) instead of the original 300 doesn't change ranking on unseen tasks (original rollouts) for any primary detector. The paired differences, full minus original, in per-task AUROC are: A's mean +0.009 [−0.037, +0.052], C's mean −0.017 [−0.045, +0.012], and B's mean at H = 10 +0.041 [−0.001, +0.077]. On unseen tasks incl. LIBERO-PRO, A's mean improves slightly (+0.026 [+0.001, +0.060]) while C's and B's are unchanged. The alarm at α = 0.1 moves in both directions: C's mean improves (+0.049 balanced accuracy on the headline set), while A's mean (−0.035) and the untrained constraint (−0.072, whose scores don't depend on training at all) get worse, which shows these alarm verdicts mostly follow the recalibrated band.

## Reading notes

- **This is the training-data ablation** (handoff §6): `tables/training_set_comparison.csv`, merged into the main run's T6 (`results/thesis_results/tables/table6_training_data*.csv`). It uses the same paired bootstrap as the main run: 1,000 resamples within task (or task and perturbation cell) within seed, with each seed's conformal band held fixed. The alarm intervals therefore understate the uncertainty.
- **Untrained rows are a check on the alarm verdicts.** The untrained constraint and FailureSpot have identical scores in both training sets: their per-task AUROC difference is exactly 0. Their alarm changes come only from the band, which the full set calibrates on its own 92 eval-seen successes. A significant alarm change for a trained detector should be read against that.
- **Selections changed on the full set** (`selections_report.md`). β changed for the action family (0.96 → 0.83) and the cross-layer family (0.91 → 0.96). Every c changed except B's grid c (1e-4). 20 constraints are sign-unstable instead of 14, so B has 116 probes per H instead of 124 (29 embedding-temporal). The best untrained constraint is emb_temp[26] under the per-task rule on both training sets.
- **Full-set core table** (`tables/table1_core_detectors.csv`), unseen tasks, original rollouts, per-task AUROC:
  - A's mean 0.754, A's combiner 0.751, A's MLP 0.642;
  - C's mean 0.667, C's 132-input MLP 0.732;
  - B's mean at H = 10 0.719, B's embedding-temporal mean 0.741, B's combiner 0.706.
- **Reproduction checks.** 120 existing full-set rows (`results/abc2` D2 tables) reproduce exactly. One per-probe value misses: A's action-token probe, eval-seen, seed 2, 0.0118. Same cause as the main run: fp16 storage, with probabilities spanning only 0.484–0.515 at c = 1e-6. A float32 refit reproduces the source value 0.4823 exactly. B's grid has no earlier full-set reference at H = 0 and 25. Its H = 10 trained combiner row is the more-data run's, which this run's refit is listed beside.
- **F1 (example rollout) is a main-run figure only**, so the full-set figures folder has no F1.
