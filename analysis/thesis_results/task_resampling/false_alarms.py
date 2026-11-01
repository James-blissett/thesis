"""
false_alarms.py

§4 of task_resampling_handoff.md: does the conformal threshold keep the false-alarm rate
at or below alpha on the unseen tasks? Read from the full training set's saved alarm
results (results/thesis_results_full/tables/data/detector_alarm_per_seed.csv; cut variant,
far = 1 - TNR over the test set's successes), on the 15 alpha levels of the grid, headline
test set (unseen tasks incl. LIBERO-PRO). Eval-seen is given beside it for reference only:
its successes include the calibration rollouts.

Outputs: task_resampling/false_alarms_vs_alpha.csv, task_resampling/F12_false_alarms_vs_alpha.{pdf,png}
Usage: source env.sh; python analysis/thesis_results/task_resampling/false_alarms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr  # noqa: E402
from tr import OUT, SEEDS, common, log, read_csv, write_csv  # noqa: E402

SERIES = [("A_mean", "A, unweighted mean", "A", "-", True),
          ("B_mean_h10", "B, unweighted mean (H = 10)", "B", "-", True),
          ("C_mean", "C, unweighted mean", "C", "-", True),
          ("A_combiner", "A, trained combiner", "A", "--", False),
          ("B_combiner_h10", "B, trained combiner (H = 10)", "B", "--", False),
          ("C_combiner", "C, trained combiner", "C", "--", False),
          ("emb_best", "untrained emb_temp[26]", None, "-", True),
          ("token_entropy", "token mean entropy", None, "--", False),
          ("token_maxprob", "token max-probability", None, ":", False),
          ("failurespot", "FailureSpot s", None, "-.", False),
          ("failurespot_neg", "FailureSpot -s", None, (0, (1, 1)), False)]


def main() -> None:
    src = tr.fc.FULL / "tables" / "data" / "detector_alarm_per_seed.csv"
    keys = {k for k, *_ in SERIES}
    by = {}
    for r in read_csv(src):
        if r["key"] in keys and r["on_alpha_grid"] == "True":
            by[(r["key"], r["test_set"], round(float(r["alpha"]), 6), int(r["seed"]))] = r
    head, seen = common.TEST_SETS["T2"], common.TEST_SETS["eval_seen"]
    alphas = sorted({a for (_, ts, a, _) in by if ts == head})
    rows = []
    for k, name, *_ in SERIES:
        for a in alphas:
            far = [float(by[(k, head, a, s)]["far"]) for s in SEEDS]
            fs = [float(by[(k, seen, a, s)]["far"]) for s in SEEDS]
            n_succ = [int(by[(k, head, a, s)]["n"]) - int(by[(k, head, a, s)]["n_fail"]) for s in SEEDS]
            rows.append({"detector": name, "alpha": round(a, 4),
                         "false-alarm rate on unseen successes, mean over splits": round(float(np.mean(far)), 4),
                         "std over splits": round(float(np.std(far, ddof=1)), 4),
                         "per split": " ".join(f"{v:.4f}" for v in far),
                         "mean above alpha": bool(np.mean(far) > a),
                         "splits above alpha": int(sum(v > a for v in far)),
                         "eval-seen false-alarm rate, mean (reference; includes calibration rollouts)": round(float(np.mean(fs)), 4),
                         "unseen successes per split": " ".join(map(str, n_succ)),
                         "row key": k})
    write_csv(OUT / "false_alarms_vs_alpha.csv", rows)
    figure(rows, alphas)
    n_above = sum(r["mean above alpha"] for r in rows)
    log(f"[far] {len(rows)} rows; mean above alpha in {n_above}; source {src}")


def figure(rows, alphas) -> None:
    import figures as fg
    plt = fg.plt
    fig, axes = plt.subplots(1, 2, figsize=(fg.W, 10.5 * fg.CM), sharey=True)
    greys = iter([fg.GREY, "#4d4d49", fg.GREY2, "#8f8e88", "#3a3a37"])
    marks = iter(["D", "v", "x", "P", "*"])
    for k, name, d, ls, filled in SERIES:
        ax = axes[0] if d is not None else axes[1]
        R = [r for r in rows if r["row key"] == k]
        x = [r["alpha"] for r in R]
        y = [r["false-alarm rate on unseen successes, mean over splits"] for r in R]
        if d is not None:
            c, m = fg.COL[d], fg.MARK[d]
        else:
            c, m = (fg.COL["C"], "o") if k == "emb_best" else (next(greys), next(marks))
        ax.plot(x, y, color=c, ls=ls, marker=m, ms=3.5, lw=1.2, mfc=c if filled else "white",
                label=name.replace(" (H = 10)", ""))
    for ax, pl in zip(axes, ("(a) detectors A, B, C", "(b) untrained constraint and baselines")):
        ax.plot([0, 1], [0, 1], color="black", ls=(0, (4, 3)), lw=0.8, zorder=1, label="false-alarm rate = alpha")
        ax.set_xlim(0, 0.92)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.text(0.0, 1.02, pl, transform=ax.transAxes, fontsize=8, va="bottom")
        ax.set_xlabel("alpha (target false-alarm rate)")
        ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    axes[0].set_ylabel("false-alarm rate on unseen successes\n(mean over task splits)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F12_false_alarms_vs_alpha.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
