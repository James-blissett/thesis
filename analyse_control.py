"""
analyse_control.py

Computes the valid significance statistics over the distributions that
control_diagnostic.py persists. Kept separate because the tests are cheap and the
distributions are expensive: re-analysing never needs a refit.

Why this exists. control_diagnostic.py's `p_null_ge_true_split_mean_INVALID` compares
individual null draws (std ~0.16) against the *mean* of the true-label splits (SE
~0.03) -- a draw against an average, which reads small whether or not an effect is
present. The two tests here compare like with like:

  1. Bootstrap-of-means. The observed split mean is a mean of n_splits draws, so the null
     it must be judged against is the null distribution *of that same statistic*:
     resample n_splits null draws, take the mean, repeat. Locate the true mean in that.
  2. Rank-sum. Distribution-free cross-check on the two sets of raw AUROCs, assuming
     nothing about shape.

Resolution floor: with N null draws the empirical p cannot meaningfully go below ~1/N
however many bootstrap resamples are taken -- the resamples reuse the same N base
values. Report p < 1/N rather than the literal 0.0000 the bootstrap prints.

Two nulls, and only one of them answers the question of interest:

  * GLOBAL null -- permutes the rollout->label mapping across the whole corpus, so it
    destroys the task->outcome correlation along with the failure signal. Per-task
    success runs 40-100% here, so a probe that decoded only *task identity* beats this
    null comfortably. Significance against it does NOT establish failure-specific signal.
  * WITHIN-TASK null -- permutes labels only among rollouts sharing a task_id, leaving
    task identity untouched. A gap here is failure-specific by construction. This is the
    one that gates any claim about failure signal.

Run the within-task comparison on the non-degenerate subset (control_diagnostic.py
--exclude-degenerate-tasks): tasks whose rollouts are all one outcome cannot be permuted
within task, so they contribute identical rows to both sides and only dilute the test.

Usage:
    source env.sh
    python analyse_control.py                                  # full-corpus run
    python analyse_control.py --in control_diagnostic_nondegenerate.json \
                              --out control_analysis_nondegenerate.json
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


def aggregate_test(true: np.ndarray, null: np.ndarray, n_boot: int,
                   seed: int = RANDOM_STATE) -> dict:
    """Judge the mean of `true` against the null distribution OF THAT SAME STATISTIC.

    Resamples len(true) null draws without replacement, takes their mean, repeats. The
    rank-sum on the raw values is a distribution-free cross-check.
    """
    k = len(true)
    if null.size < k:
        raise ValueError(f"need at least {k} null draws, got {null.size}")
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(null, size=k, replace=False).mean()
                      for _ in range(n_boot)])
    p_boot = float((means >= true.mean()).mean())
    floor = 1.0 / null.size
    _, p_rank = mannwhitneyu(true, null, alternative="greater")
    return {
        "true_label_split_mean": float(true.mean()),
        "true_label_split_std": float(true.std(ddof=1)),
        "n_splits": int(k),
        "n_above_chance": int((true > 0.5).sum()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "n_null_draws": int(null.size),
        "null_mean_of_k_std": float(means.std()),
        # Degenerate when the null has barely more draws than k (resampling without
        # replacement then returns the same mean every time); the p-value still stands.
        "sd_above_null": (float((true.mean() - means.mean()) / means.std())
                          if means.std() > 0 else None),
        "p_bootstrap_of_means": p_boot,
        "p_bootstrap_reportable": f"< {floor:.4f}" if p_boot < floor else f"{p_boot:.4f}",
        "p_ranksum": float(p_rank),
        "verdict": "significant" if p_boot < 0.05 else "not significant",
    }


def single_split_test(primary: float, ctrl: np.ndarray) -> dict:
    """The locked single-split AUROC judged against its own fixed-split null."""
    p = float((ctrl >= primary).mean())
    return {
        "timestep_auroc": primary,
        "permutation_p": p,
        "n_permutations": int(ctrl.size),
        "n_null_at_or_above": int((ctrl >= primary).sum()),
        "verdict": "not significant" if p > 0.05 else "significant",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--in", dest="in_name", type=str, default="control_diagnostic.json")
    ap.add_argument("--out", dest="out_name", type=str, default=None,
                    help="defaults to the input name with 'diagnostic' -> 'analysis'")
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    with open(results_dir / args.in_name) as f:
        d = json.load(f)

    true = np.array([r["timestep_auroc"] for r in d["split_sensitivity"]["per_split"]])
    null = np.array(d["matched_null"]["values"], dtype=float)
    null_wt = np.array(d.get("matched_null_within_task", {}).get("values", []), dtype=float)
    ctrl = np.array(d["control_distribution"]["values"], dtype=float)
    ctrl_wt = np.array(
        d.get("control_distribution_within_task", {}).get("values", []), dtype=float)
    primary = d["locked_result"]["timestep_auroc"]
    subset = d.get("subset", {})
    excluded = subset.get("excluded_task_ids", [])

    out = {
        "subset": {
            "exclude_degenerate_tasks": subset.get("exclude_degenerate_tasks", False),
            "excluded_task_ids": excluded,
            "analysed_composition": subset.get("analysed_composition"),
            "note": "true and null statistics are computed on these rollouts only",
        },
        "single_split_result": {},
        "aggregate_result": {},
    }

    # --- Single locked split, against each available null ------------------------------
    if ctrl.size:
        out["single_split_result"]["vs_global_null"] = single_split_test(primary, ctrl)
    if ctrl_wt.size:
        out["single_split_result"]["vs_within_task_null"] = single_split_test(
            primary, ctrl_wt)

    # --- Aggregate over splits, against each available null ----------------------------
    if null.size:
        out["aggregate_result"]["vs_global_null"] = aggregate_test(true, null, args.n_boot)
    if null_wt.size:
        out["aggregate_result"]["vs_within_task_null"] = aggregate_test(
            true, null_wt, args.n_boot)

    if not out["aggregate_result"]:
        raise SystemExit("no matched-null draws; rerun control_diagnostic.py --n-matched N")

    # --- The mandated [0.40, 0.60] gate, scored against the empirical null it is meant
    # to accept: coverage well under ~95% means the gate rejects clean runs.
    if ctrl.size:
        lo, hi = d["locked_result"]["control_acceptable_range"]
        out["gate_calibration"] = {
            "mandated_range": [lo, hi],
            "empirical_null_mean": float(ctrl.mean()),
            "empirical_null_std": float(ctrl.std(ddof=1)),
            "fraction_of_clean_runs_inside_mandated_range":
                float(((ctrl >= lo) & (ctrl <= hi)).mean()),
            "two_sd_range": [float(ctrl.mean() - 2 * ctrl.std(ddof=1)),
                             float(ctrl.mean() + 2 * ctrl.std(ddof=1))],
        }

    # --- What each null does and does not license -------------------------------------
    wt = out["aggregate_result"].get("vs_within_task_null")
    out["interpretation"] = {
        "global_null_caveat":
            "The global null permutes rollout->label across the whole corpus, destroying "
            "the task->outcome correlation. Per-task success is 40-100% in this corpus, "
            "so task decoding alone would beat it. Significance against the global null "
            "does NOT establish failure-relevant signal specifically.",
        "within_task_null_meaning":
            "The within-task null permutes labels only among rollouts sharing a task_id, "
            "so task identity is intact in both true and null fits. A gap is "
            "failure-specific by construction.",
    }
    if wt is None:
        out["interpretation"]["claim_reached"] = (
            "signal exists vs. global null -- within-task null not run, so failure "
            "specificity is untested")
    elif not excluded:
        out["interpretation"]["claim_reached"] = (
            "within-task null run on ALL tasks, including degenerate ones that are inert "
            "under it; see the --exclude-degenerate-tasks run for the primary verdict")
    elif wt["p_bootstrap_of_means"] < 0.05:
        out["interpretation"]["claim_reached"] = (
            "signal survives the within-task null on the non-degenerate subset: "
            "directional failure-specific signal at layer 15, pilot-scale")
    else:
        out["interpretation"]["claim_reached"] = (
            "signal does NOT survive the within-task null on the non-degenerate subset: "
            "layer 15 decodes task identity; failure signal not yet demonstrated at this "
            "corpus size")

    out_name = args.out_name or args.in_name.replace("diagnostic", "analysis")
    path = results_dir / out_name
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # --- Report -----------------------------------------------------------------------
    comp = subset.get("analysed_composition") or {}
    print(f"subset       : {'degenerate tasks excluded ' + str(excluded) if excluded else 'all tasks'}"
          f" | {comp.get('n_rollouts', '?')} rollouts "
          f"({comp.get('n_failure_rollouts', '?')} fail / "
          f"{comp.get('n_success_rollouts', '?')} succ), "
          f"{comp.get('n_timesteps', '?')} timesteps")
    for label, key in (("global", "vs_global_null"), ("within-task", "vs_within_task_null")):
        s = out["single_split_result"].get(key)
        if s:
            print(f"single split : AUROC {s['timestep_auroc']:.4f}  vs {label} null: "
                  f"perm p {s['permutation_p']:.3f} "
                  f"({s['n_null_at_or_above']}/{s['n_permutations']}) -> {s['verdict']}")
    for label, key in (("global", "vs_global_null"), ("within-task", "vs_within_task_null")):
        a = out["aggregate_result"].get(key)
        if a:
            print(f"aggregate    : true {a['true_label_split_mean']:.4f} "
                  f"(n={a['n_splits']}, {a['n_above_chance']} above .5) vs {label} null "
                  f"{a['null_mean']:.4f} (n={a['n_null_draws']})")
            sd = ("n/a" if a["sd_above_null"] is None
                  else f"{a['sd_above_null']:.1f}")
            print(f"               {sd} SD above the null "
                  f"mean-of-{a['n_splits']}; p {a['p_bootstrap_reportable']}, "
                  f"ranksum p {a['p_ranksum']:.4f} -> {a['verdict']}")
    g = out.get("gate_calibration")
    if g:
        print(f"gate         : mandated {g['mandated_range']} admits only "
              f"{100*g['fraction_of_clean_runs_inside_mandated_range']:.0f}% of clean runs; "
              f"+-2SD would be [{g['two_sd_range'][0]:.2f}, {g['two_sd_range'][1]:.2f}]")
    print(f"CLAIM        : {out['interpretation']['claim_reached']}")
    print(f"=== wrote {path} ===")


if __name__ == "__main__":
    main()
