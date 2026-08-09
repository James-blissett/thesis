"""
analyse_control.py

Computes the valid significance statistics over the distributions that
control_diagnostic.py persists. Kept separate because the tests are cheap and the
distributions are expensive: re-analysing never needs a refit.

Why this exists. control_diagnostic.py's `p_null_ge_true_split_mean_INVALID` compares
individual null draws (std ~0.16) against the *mean* of the true-label splits (SE
~0.03) -- a draw against an average, which reads small whether or not an effect is
present. The two tests here compare like with like:

  1. Bootstrap-of-means. The observed 0.646 is a mean of n_splits draws, so the null it
     must be judged against is the null distribution *of that same statistic*: resample
     n_splits matched-null draws, take the mean, repeat. Locate 0.646 in that.
  2. Rank-sum. Distribution-free cross-check on the two sets of raw AUROCs, assuming
     nothing about shape.

Resolution floor: with N matched-null draws the empirical p cannot meaningfully go
below ~1/N however many bootstrap resamples are taken -- the resamples reuse the same N
base values. Report p < 1/N rather than the literal 0.0000 the bootstrap prints.

WHAT NEITHER TEST CONTROLS FOR: both permute the rollout->label mapping globally, which
destroys the task->outcome correlation along with the failure signal. Per-task success
rates range 40-100% in this corpus, so a probe that decoded only *task identity* would
beat this null comfortably. A significant result here is therefore consistent with
failure-relevant signal AND with task decoding, and does not separate them. Use
--within-task for the null that holds task identity fixed.

Usage:
    source env.sh
    python analyse_control.py
    python analyse_control.py --n-boot 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

RESULTS_DIR = Path("results/probe_pilot")
RANDOM_STATE = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    with open(results_dir / "control_diagnostic.json") as f:
        d = json.load(f)

    true = np.array([r["timestep_auroc"] for r in d["split_sensitivity"]["per_split"]])
    null = np.array(d["matched_null"]["values"], dtype=float)
    ctrl = np.array(d["control_distribution"]["values"], dtype=float)
    primary = d["locked_result"]["timestep_auroc"]

    if null.size == 0:
        raise SystemExit("no matched-null draws; rerun control_diagnostic.py --n-matched N")

    rng = np.random.RandomState(RANDOM_STATE)
    k = len(true)
    means = np.array([rng.choice(null, size=k, replace=False).mean()
                      for _ in range(args.n_boot)])
    p_boot = float((means >= true.mean()).mean())
    floor = 1.0 / len(null)
    u, p_rank = mannwhitneyu(true, null, alternative="greater")

    # The locked single-split result, judged against its own fixed-split null.
    p_primary = float((ctrl >= primary).mean())

    # The mandated [0.40, 0.60] gate, scored against the empirical null it is meant to
    # accept: coverage well under ~95% means the gate rejects clean runs.
    lo, hi = d["locked_result"]["control_acceptable_range"]
    coverage = float(((ctrl >= lo) & (ctrl <= hi)).mean())

    out = {
        "single_split_result": {
            "timestep_auroc": primary,
            "permutation_p": p_primary,
            "n_permutations": int(len(ctrl)),
            "verdict": "not significant" if p_primary > 0.05 else "significant",
        },
        "aggregate_result": {
            "true_label_split_mean": float(true.mean()),
            "true_label_split_std": float(true.std(ddof=1)),
            "n_splits": k,
            "n_above_chance": int((true > 0.5).sum()),
            "matched_null_mean": float(null.mean()),
            "matched_null_std": float(null.std(ddof=1)),
            "null_mean_of_k_std": float(means.std()),
            "sd_above_null": float((true.mean() - means.mean()) / means.std()),
            "p_bootstrap_of_means": p_boot,
            "p_bootstrap_reportable": f"< {floor:.4f}" if p_boot < floor else f"{p_boot:.4f}",
            "p_ranksum": float(p_rank),
        },
        "gate_calibration": {
            "mandated_range": [lo, hi],
            "empirical_null_mean": float(ctrl.mean()),
            "empirical_null_std": float(ctrl.std(ddof=1)),
            "fraction_of_clean_runs_inside_mandated_range": coverage,
            "two_sd_range": [float(ctrl.mean() - 2 * ctrl.std(ddof=1)),
                             float(ctrl.mean() + 2 * ctrl.std(ddof=1))],
        },
        "uncontrolled_confound": (
            "Both nulls permute rollout->label globally, destroying the task->outcome "
            "correlation. Per-task success is 40-100% in this corpus, so task decoding "
            "alone would beat these nulls. Significance here does NOT establish "
            "failure-relevant signal specifically."
        ),
    }

    path = results_dir / "control_analysis.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    a = out["aggregate_result"]
    print(f"single split : AUROC {primary:.4f}  perm p {p_primary:.3f} "
          f"({int((ctrl >= primary).sum())}/{len(ctrl)})  -> "
          f"{out['single_split_result']['verdict']}")
    print(f"aggregate    : true {a['true_label_split_mean']:.4f} (n={k}, "
          f"{a['n_above_chance']} above .5) vs null {a['matched_null_mean']:.4f}")
    print(f"               {a['sd_above_null']:.1f} SD above the null mean-of-{k}; "
          f"p {a['p_bootstrap_reportable']}, ranksum p {a['p_ranksum']:.4f}")
    g = out["gate_calibration"]
    print(f"gate         : mandated {g['mandated_range']} admits only "
          f"{100*g['fraction_of_clean_runs_inside_mandated_range']:.0f}% of clean runs; "
          f"+-2SD would be [{g['two_sd_range'][0]:.2f}, {g['two_sd_range'][1]:.2f}]")
    print(f"CONFOUND     : {out['uncontrolled_confound']}")
    print(f"=== wrote {path} ===")


if __name__ == "__main__":
    main()
