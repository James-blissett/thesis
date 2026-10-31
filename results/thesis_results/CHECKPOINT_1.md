# Checkpoint 1: core table on the headline test set

*results_collection_handoff.md (version 4), §5 Checkpoint 1 (§4.1–4.2, unseen tasks, original rollouts). Written 2026-10-30 ~14:20 UTC by the unattended run, which then carried on (brief: no stop). All three test sets are already computed; they're reported at Checkpoint 2.*

**Nothing needs a decision before going on.** Every existing row reproduces its source file (largest difference 0.0001). No new mean or combiner sits sharply below its own single best probe. Two things to know:

1. **Two per-probe values differ by 0.0053 from `results/abc2/a_probes.csv`**, against a 0.005 tolerance: A's action-token probe, eval-seen, seeds 1 and 2. I investigated, and the cause is harmless. The refitted probe outputs are stored as fp16, exactly as `setup_a` stored them in `/data/tmp/abc_scores/`. abc2's per-probe table was computed from float32. Refitting that probe and scoring it in float32 gives 0.6824 and 0.5758, equal to the source. In fp16 it gives 0.6877 and 0.5811. The detector rows (A's single best probe, A's mean) match their sources exactly.
2. **The example rollout (§4.8) can't come from seed 0.** At α = 0.1, B's unweighted mean catches no unseen failure before the cut in seed 0 (its share of failures caught is 0.000 in seed 0 on unseen tasks, original rollouts, as in `results/b_label_free/lf_final.csv`). I used the next seed that has a candidate, seed 1 (task 3), and marked this as a fallback in the figure data and in RUNLOG. James may prefer a different choice, e.g. a different α.

## Core table (T1), unseen tasks, original rollouts

Mean ± std over 3 seeds (per-seed values in `tables/table1_core_detectors.csv`). Per-task AUROC averages 3 tasks per seed. Floor = pooled AUROC of the timestep-only score (its per-task AUROC is 0.5 by construction). Alarm columns: cut-variant conformal alarm at α = 0.1, calibrated on the training set's successful eval-seen rollouts (33–43 per seed). Stress set: share of the 100 position-swap and task-perturbed rollouts alarmed before the cut at α = 0.1.

| Detector | Variant | H | per-task AUROC | pooled AUROC | floor | bal. acc. α 0.1 | failures caught | false alarms | detection time | stress set alarmed |
|---|---|---|---|---|---|---|---|---|---|---|
| A | single best probe (of 34) |  | 0.716 ± 0.032 | 0.727 ± 0.014 | 0.5466 | 0.537 ± 0.033 | 0.086 ± 0.081 | 0.012 ± 0.021 | 0.963 ± 0.045 | 0.207 ± 0.181 |
| A | unweighted mean of 34 probe outputs (primary) |  | 0.745 ± 0.037 | 0.735 ± 0.026 | 0.5466 | 0.536 ± 0.043 | 0.072 ± 0.086 | 0.000 ± 0.000 | 0.983 ± 0.015 | 0.207 ± 0.122 |
| A | trained combiner over 34 probe outputs (out-of-fold) |  | 0.769 ± 0.016 | 0.753 ± 0.013 | 0.5466 | 0.520 ± 0.034 | 0.040 ± 0.069 | 0.000 ± 0.000 | 0.993 ± 0.012 | 0.140 ± 0.185 |
| A | 2-layer MLP on the best probe's hidden state |  | 0.685 ± 0.035 | 0.715 ± 0.032 | 0.5466 | 0.516 ± 0.018 | 0.033 ± 0.036 | 0.000 ± 0.000 | 0.999 ± 0.002 | 0.170 ± 0.070 |
| C | single best probe (of 33) |  | 0.687 ± 0.062 | 0.639 ± 0.114 | 0.5466 | 0.504 ± 0.007 | 0.008 ± 0.014 | 0.000 ± 0.000 | 0.994 ± 0.011 | 0.103 ± 0.097 |
| C | unweighted mean of 33 probe outputs (primary) |  | 0.684 ± 0.066 | 0.680 ± 0.090 | 0.5466 | 0.502 ± 0.003 | 0.054 ± 0.094 | 0.050 ± 0.087 | 0.972 ± 0.049 | 0.183 ± 0.170 |
| C | trained combiner over 33 probe outputs (out-of-fold) |  | 0.701 ± 0.039 | 0.719 ± 0.059 | 0.5466 | 0.517 ± 0.018 | 0.047 ± 0.056 | 0.013 ± 0.022 | 0.978 ± 0.032 | 0.207 ± 0.081 |
| C | 2-layer MLP on the best probe's inputs |  | 0.707 ± 0.073 | 0.676 ± 0.102 | 0.5466 | 0.504 ± 0.007 | 0.008 ± 0.014 | 0.000 ± 0.000 | 0.994 ± 0.010 | 0.093 ± 0.107 |
| C (exploratory) | logistic regression on all 132 constraints |  | 0.717 ± 0.049 | 0.704 ± 0.070 | 0.5466 | 0.501 ± 0.002 | 0.027 ± 0.047 | 0.025 ± 0.044 | 0.979 ± 0.036 | 0.200 ± 0.125 |
| C (exploratory) | 2-layer MLP on all 132 constraints |  | 0.718 ± 0.035 | 0.748 ± 0.043 | 0.5466 | 0.591 ± 0.108 | 0.207 ± 0.226 | 0.025 ± 0.009 | 0.893 ± 0.124 | 0.360 ± 0.144 |
| B | single best probe (of 124), chosen on eval-seen per-task AUROC against outcome | 0 | 0.667 ± 0.054 | 0.669 ± 0.046 | 0.5466 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.107 ± 0.096 |
| B | unweighted mean of 124 probe outputs (primary) | 0 | 0.662 ± 0.053 | 0.697 ± 0.055 | 0.5466 | 0.510 ± 0.017 | 0.043 ± 0.074 | 0.024 ± 0.041 | 0.966 ± 0.058 | 0.227 ± 0.146 |
| B | unweighted mean of the 30 embedding-temporal probe outputs | 0 | 0.685 ± 0.043 | 0.703 ± 0.054 | 0.5466 | 0.556 ± 0.065 | 0.159 ± 0.211 | 0.048 ± 0.083 | 0.921 ± 0.122 | 0.283 ± 0.169 |
| B | trained combiner over 124 probe outputs (out-of-fold) | 0 | 0.693 ± 0.030 | 0.728 ± 0.061 | 0.5466 | 0.516 ± 0.020 | 0.064 ± 0.045 | 0.031 ± 0.029 | 0.962 ± 0.035 | 0.280 ± 0.053 |
| B | single best probe (of 124), chosen on eval-seen per-task AUROC against outcome | 10 | 0.658 ± 0.070 | 0.672 ± 0.037 | 0.5466 | 0.500 ± 0.001 | 0.018 ± 0.031 | 0.019 ± 0.033 | 0.987 ± 0.022 | 0.177 ± 0.076 |
| B | unweighted mean of 124 probe outputs (primary) | 10 | 0.678 ± 0.058 | 0.705 ± 0.058 | 0.5466 | 0.510 ± 0.045 | 0.093 ± 0.118 | 0.073 ± 0.064 | 0.935 ± 0.074 | 0.313 ± 0.081 |
| B | unweighted mean of the 30 embedding-temporal probe outputs | 10 | 0.731 ± 0.020 | 0.723 ± 0.048 | 0.5466 | 0.519 ± 0.035 | 0.093 ± 0.118 | 0.055 ± 0.054 | 0.934 ± 0.080 | 0.293 ± 0.091 |
| B | trained combiner over 124 probe outputs (out-of-fold) | 10 | 0.709 ± 0.047 | 0.728 ± 0.045 | 0.5466 | 0.517 ± 0.034 | 0.110 ± 0.108 | 0.075 ± 0.131 | 0.922 ± 0.098 | 0.393 ± 0.121 |
| B | single best probe (of 124), chosen on eval-seen per-task AUROC against outcome | 25 | 0.720 ± 0.021 | 0.705 ± 0.011 | 0.5466 | 0.521 ± 0.020 | 0.043 ± 0.041 | 0.000 ± 0.000 | 0.988 ± 0.017 | 0.220 ± 0.113 |
| B | unweighted mean of 124 probe outputs (primary) | 25 | 0.698 ± 0.056 | 0.714 ± 0.059 | 0.5466 | 0.510 ± 0.030 | 0.082 ± 0.050 | 0.062 ± 0.079 | 0.939 ± 0.054 | 0.323 ± 0.070 |
| B | unweighted mean of the 30 embedding-temporal probe outputs | 25 | 0.742 ± 0.036 | 0.721 ± 0.056 | 0.5466 | 0.515 ± 0.028 | 0.050 ± 0.048 | 0.019 ± 0.033 | 0.981 ± 0.024 | 0.243 ± 0.134 |
| B | trained combiner over 124 probe outputs (out-of-fold) | 25 | 0.731 ± 0.030 | 0.715 ± 0.058 | 0.5466 | 0.526 ± 0.039 | 0.084 ± 0.074 | 0.031 ± 0.054 | 0.968 ± 0.044 | 0.280 ± 0.123 |
| B | 2-layer MLP on the best probe's layer, trained on that probe's constraint target | 10 | 0.675 ± 0.070 | 0.682 ± 0.017 | 0.5466 | 0.514 ± 0.012 | 0.034 ± 0.029 | 0.006 ± 0.011 | 0.988 ± 0.021 | 0.160 ± 0.066 |
| Baseline | timestep only |  | 0.500 ± 0.000 | 0.547 ± 0.090 | 0.5466 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Baseline | best single untrained constraint (layer chosen on eval-seen per-task AUROC), running maximum |  | 0.725 ± 0.061 | 0.715 ± 0.039 | 0.5466 | 0.553 ± 0.101 | 0.220 ± 0.382 | 0.114 ± 0.180 | 0.861 ± 0.241 | 0.290 ± 0.331 |
| Baseline | untrained constraint at the earlier layer choice (eval-seen pooled AUROC), running maximum |  | 0.725 ± 0.055 | 0.719 ± 0.036 | 0.5466 | 0.556 ± 0.106 | 0.226 ± 0.391 | 0.114 ± 0.180 | 0.852 ± 0.257 | 0.297 ± 0.348 |
| Baseline | token mean entropy, running maximum |  | 0.539 ± 0.052 | 0.577 ± 0.060 | 0.5466 | 0.523 ± 0.028 | 0.141 ± 0.122 | 0.094 ± 0.090 | 0.926 ± 0.067 | 0.163 ± 0.160 |
| Baseline | token max-probability, running maximum |  | 0.475 ± 0.053 | 0.511 ± 0.032 | 0.5466 | 0.490 ± 0.020 | 0.048 ± 0.048 | 0.067 ± 0.066 | 0.958 ± 0.040 | 0.110 ± 0.139 |
| Baseline | FailureSpot signal s, running maximum |  | 0.477 ± 0.061 | 0.491 ± 0.125 | 0.5466 | 0.518 ± 0.018 | 0.128 ± 0.120 | 0.091 ± 0.085 | 0.912 ± 0.077 | 0.093 ± 0.081 |
| Baseline | FailureSpot signal, opposite orientation (-s), running maximum |  | 0.542 ± 0.052 | 0.574 ± 0.053 | 0.5466 | 0.528 ± 0.039 | 0.147 ± 0.138 | 0.090 ± 0.139 | 0.883 ± 0.109 | 0.013 ± 0.015 |

**Reading.**
- **Ranking.** A's unweighted mean (0.745 ± 0.037) is above its own single best probe (0.716) and below its trained combiner (0.769). C's mean (0.684) is level with its best probe (0.687) and combiner (0.701). B's mean at H = 10 (0.678) is level with its best probe (0.658), its combiner (0.709) and its MLP (0.675). The best untrained constraint alone reaches 0.725.
- **Paired bootstrap** (`tables/paired_comparisons.csv`, 95% intervals, 1,000 resamples):
  - B mean minus A mean, per-task AUROC: −0.068 [−0.126, −0.011]. A ranks better.
  - A mean minus C mean, balanced accuracy: +0.034 [+0.001, +0.071].
  - A mean minus its trained combiner, balanced accuracy: +0.016 [+0.003, +0.043].
  - B's MLP minus B's best linear probe, balanced accuracy: +0.014 [+0.003, +0.029].
  - Every other comparison's interval includes zero.
- **Alarm.** At α = 0.1 every detector is weak, at a balanced accuracy of 0.50–0.59. The best are C's 2-layer MLP on 132 inputs (0.591) and the untrained constraint (0.553). Most trained detectors catch under 10 % of failures before the cut.
- **Sharp disagreements:** none of the kind the handoff names. No mean is more than 0.1 below its own single best probe. The largest gaps are A's best probe 0.029 *below* A's mean, and B's mean at H = 25, 0.022 below its best probe. Worth noting: B's mean over all 124 probes (0.678 at H = 10) is below the mean of its 30 embedding-temporal probes (0.731). The xl_final probes sit near or below chance against outcome (F8), which pulls the unweighted mean down.
- **Changed choices:**
  - The best untrained constraint, chosen by the per-task rule, is emb_temp[26] (the pooled rule gave 28). Both score 0.725.
  - B's trained combiner at H = 0 and 25 was refitted on the refitted grid outputs. The refit equals `results/abc/final_table.csv` exactly. So does the H = 10 refit, kept as a check row.

## Reproduction checks

244 detector-row checks against six source files. **All pass, largest difference 0.0001** (`tables/data/reproduction_checks.csv`):
- `results/abc/final_table.csv`: A, C, B at H = 0/10/25, emb_temp[28]. Eval-seen and unseen, per-task and pooled AUROC, α = 0.1 balanced accuracy, failures caught, detection time.
- `results/abc2/final_table.csv`, `mlp_final_table.csv`, `b_final_table.csv`: A, C, emb_temp[28], both 2-layer MLPs, B H = 10. All three test sets, AUROC and every α = 0.1 alarm column.
- `results/abc2/stress_summary.csv`, `mlp_stress.csv`, `b_stress.csv`: stress-set share alarmed, seen and unseen base tasks.
- `results/b_label_free/lf_final.csv`: both unweighted means of B at each H.
- `results/abc2/constraint_m1_summary.csv`: timestep, emb_temp[26], token baselines, FailureSpot both ways, AUROC on all three sets.

Also checked:
- Probe refits against the saved outputs: A exactly, C exactly, B within fp16 rounding (0.0005).
- B's grid: 2,232 own-target AUROCs against `results/abc/b_grid.csv`, all within 0.005 (max 0.0013).
- A and C per-probe AUROCs against `results/abc2/{a,c}_probes.csv`: 601 of 603 pass, with the 2 fp16 cases above.
- The GPU fit against scikit-learn for A's out-of-fold probes: max difference 0.00006.
- The bootstrap identity check: max difference 5e-7.

## Proposed commit message

```
Thesis results, main run (original training set): core detector table

Collect every row of the thesis core table on the original training set:
A, C and B as single best probe, unweighted mean of probe outputs, trained
combiner (out-of-fold) and 2-layer MLP, plus baselines with conformal alarms,
on unseen tasks (original and incl. LIBERO-PRO) and the stress set. All
existing rows reproduce their source files (244 checks, max diff 0.0001).

New:
- analysis/thesis_results/: probe refits (A, B on GPU; C on CPU), detectors,
  evaluation, reproduction checks, paired bootstrap, profiles, tables, figures
- results/thesis_results/: CHECKPOINT_0_audit.md, CHECKPOINT_1.md, RUNLOG.md,
  STATE.md, tables/, figures/, scores/

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
