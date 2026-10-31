"""
paired.py

Paired comparisons on the headline test set (unseen tasks, original rollouts)
(results_collection_handoff.md §4.5): the more-data run's paired bootstrap
(analysis/abc2/bootstrap2.py, imported: 1,000 resamples of rollouts within each task
within each seed, both sides recomputed on the same resample, seed means, 95% percentile
interval; the conformal band of each seed held fixed) on the per-seed npz files written
by evaluate.py. Metrics: per-task AUROC, balanced accuracy at alpha 0.1 (and share of
failures caught). One set of resamples is shared by every comparison.

Comparisons (x - y), B at the headline H = 10:
  B mean - A mean;  B mean - C mean;  A mean - C mean
  each detector's mean - its trained combiner, its single best probe, its 2-layer MLP
  B's MLP - B's single best probe (the linear probe it was built on)

Identity check: on the original sample the recomputed per-seed values must equal
evaluate.py's per-seed tables (|diff| <= 1e-6).

Outputs: tables/paired_comparisons.csv (plain headers), tables/data/paired_comparisons_per_seed.csv,
logs/paired_identity_check.json.

Usage: source env.sh; python analysis/thesis_results/paired.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import evaluate as ev  # noqa: E402
from common import SEEDS, read_csv, write_csv  # noqa: E402

COMPARISONS = [
    ("B_mean_h10", "A_mean", "B mean vs A mean"),
    ("B_mean_h10", "C_mean", "B mean vs C mean"),
    ("A_mean", "C_mean", "A mean vs C mean"),
    ("A_mean", "A_combiner", "A mean vs A trained combiner"),
    ("A_mean", "A_best", "A mean vs A single best probe"),
    ("A_mean", "A_mlp", "A mean vs A 2-layer MLP"),
    ("C_mean", "C_combiner", "C mean vs C trained combiner"),
    ("C_mean", "C_best", "C mean vs C single best probe"),
    ("C_mean", "C_mlp", "C mean vs C 2-layer MLP"),
    ("B_mean_h10", "B_combiner_h10", "B mean vs B trained combiner"),
    ("B_mean_h10", "B_best_h10", "B mean vs B single best probe"),
    ("B_mean_h10", "B_mlp_h10", "B mean vs B 2-layer MLP"),
    ("B_mlp_h10", "B_best_h10", "B 2-layer MLP vs B single best linear probe"),
]
METRIC_NAME = {"m1_per_task": "per-task AUROC", "bal_acc": "balanced accuracy at alpha 0.1",
               "tpr": "share of failures caught at alpha 0.1"}


def label(key):
    d, v, _, H = ev.LABEL[key]
    return f"{d}: {v}" + (f", H = {H}" if H != "" else "")


def run(test_set: str = "T0", comparisons=COMPARISONS, files_x=None, files_y=None, name="paired_comparisons"):
    import bootstrap2 as b2
    files = [ev.NPZ / f"seed{s}.npz" for s in SEEDS]
    fx, fy = files_x or files, files_y or files
    sides = {}

    def side(files_, key):
        k = (tuple(map(str, files_)), key)
        if k not in sides:
            sides[k] = b2.load_side(files_, key.replace(":", "__"), test_set)
        return sides[k]

    first = side(fy, comparisons[0][1])
    draws = b2.make_draws(first, b2.N_BOOT, b2.RNG_SEED)
    boots = {}
    rows, per_seed, ident = [], [], {}
    for kx, ky, title in comparisons:
        x, y = side(fx, kx), side(fy, ky)
        for k, s_ in ((kx, x), (ky, y)):
            if (id(s_), k) not in boots:
                boots[(id(s_), k)] = b2.side_boot(s_, draws)
        r = b2.compare(x, y, draws, test_set=test_set, n_boot=b2.N_BOOT, rng_seed=b2.RNG_SEED,
                       comparison=title, x_boot=boots[(id(x), kx)], y_boot=boots[(id(y), ky)])
        for row in r["rows"]:
            rows.append({"comparison (x - y)": title, "x": label(kx), "y": label(ky),
                         "test set": common.TEST_SETS[test_set], "metric": METRIC_NAME[row["metric"]],
                         "x value": round(row["x_point"], 4), "y value": round(row["y_point"], 4),
                         "difference": round(row["diff_point"], 4),
                         "bootstrap mean difference": round(row["diff_mean"], 4),
                         "95% interval low": round(row["diff_ci_lo"], 4),
                         "95% interval high": round(row["diff_ci_hi"], 4),
                         "verdict": row["verdict"].replace("improvement", "x higher (interval above zero)")
                                                  .replace("significant worsening", "x lower (interval below zero)")
                                                  .replace("no significant change", "interval includes zero"),
                         "resamples": row["n_boot"], "valid resamples": row["n_valid"],
                         "x per seed": row["x_point_per_seed"], "y per seed": row["y_point_per_seed"],
                         "x key": kx, "y key": ky})
        per_seed += [{**p, "comparison": title} for p in r["per_seed"]]
        ident[kx] = r["identity"]["x"]
        ident[ky] = r["identity"]["y"]
    return rows, per_seed, ident


def identity_check(ident, test_set="T0") -> dict:
    d = common.OUT_DIR / "tables" / "data"
    ts = common.TEST_SETS[test_set]
    au = {(r["key"], int(r["seed"])): float(r["m1_per_task"]) for r in read_csv(d / "detector_auroc_per_seed.csv")
          if r["test_set"] == ts}
    ba = {(r["key"], int(r["seed"])): (float(r["bal_acc"]), float(r["tpr"]))
          for r in read_csv(d / "detector_alarm_per_seed.csv") if r["test_set"] == ts and r["on_alpha_grid"] == "False"}
    worst = 0.0
    for k, v in ident.items():
        for j, s in enumerate(SEEDS):
            worst = max(worst, abs(v["m1_per_task"][j] - au[(k, s)]), abs(v["bal_acc"][j] - ba[(k, s)][0]),
                        abs(v["tpr"][j] - ba[(k, s)][1]))
    return {"max_abs_diff": worst, "passed": worst <= 1e-6, "keys": sorted(ident)}


def main() -> None:
    rows, per_seed, ident = run("T0")
    chk = identity_check(ident)
    common.write_json(common.OUT_DIR / "logs" / "paired_identity_check.json", chk)
    write_csv(common.OUT_DIR / "tables" / "paired_comparisons.csv", rows)
    write_csv(common.OUT_DIR / "tables" / "data" / "paired_comparisons_per_seed.csv", per_seed)
    print(f"{len(rows)} rows; identity check max |diff| {chk['max_abs_diff']:.2e} "
          f"({'passed' if chk['passed'] else 'FAILED'})")


if __name__ == "__main__":
    main()
