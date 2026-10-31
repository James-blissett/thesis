"""
checks.py

Reproduction checks (results_collection_handoff.md §5, Checkpoint 1): every row that
already exists must match its source file within 0.005 (AUROC) and 0.01 (balanced
accuracy and the other alarm shares). Compares the seed means computed by evaluate.py
(tables/data/detector_*_per_seed.csv) with the earlier runs' tables.

Writes tables/data/reproduction_checks.csv: check, row key, quantity, test set, source
file, source value, this run, |difference|, tolerance, passed.

Usage: source env.sh; python analysis/thesis_results/checks.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import REPO, read_csv, write_csv  # noqa: E402

TOL_AUROC, TOL_ALARM = 0.005, 0.01
PLAIN = {v: k for k, v in common.TEST_SETS.items()}


def this_run():
    d = common.OUT_DIR / "tables" / "data"
    au = defaultdict(list)
    for r in read_csv(d / "detector_auroc_per_seed.csv"):
        ts = PLAIN[r["test_set"]]
        au[(r["key"], ts, "per_task")].append(float(r["m1_per_task"]))
        au[(r["key"], ts, "pooled")].append(float(r["m1_pooled"]))
    for r in read_csv(d / "detector_alarm_per_seed.csv"):
        if r["on_alpha_grid"] == "False":
            ts = PLAIN[r["test_set"]]
            for k in ("bal_acc", "tpr", "far", "det_time"):
                au[(r["key"], ts, k)].append(float(r[k]))
    for r in read_csv(d / "detector_stress_per_seed.csv"):
        if r["block"] == "all":
            au[(r["key"], r["stress_set"], "stress_share")].append(float(r["share_alarm_before_cut"]))
    return {k: float(np.mean(v)) for k, v in au.items()}


def checks() -> list:
    me = this_run()
    out = []

    def add(name, key, q, ts, src, ref):
        if (key, ts, q) not in me or ref in ("", None):
            return
        ref = float(ref)
        got = me[(key, ts, q)]
        tol = TOL_AUROC if q in ("per_task", "pooled") else TOL_ALARM
        out.append({"check": name, "row_key": key, "quantity": q,
                    "test_set": common.TEST_SETS.get(ts, ts), "source_file": src,
                    "source_value": round(ref, 4), "this_run": round(got, 4),
                    "abs_difference": round(abs(got - ref), 4), "tolerance": tol,
                    "passed": abs(got - ref) <= tol + 1e-9})

    pool = common.POOL
    # results/abc2 tables (this training set): A, C, emb_temp (pooled rule), MLPs, B H=10.
    for fname, mp in (("results/abc2/final_table.csv", {"A": "A_best", "C": "C_lr132", "emb": "emb_pooled"}),
                      ("results/abc2/mlp_final_table.csv", {"A-MLP": "A_mlp", "C-MLP": "C_mlp132"}),
                      ("results/abc2/b_final_table.csv", {"B H=10": "B_combiner_h10"})):
        for r in read_csv(REPO / fname):
            if r["cond"] != pool or r["detector"] not in mp or r["test_set"] not in common.AUROC_SETS:
                continue
            key = mp[r["detector"]]
            for q, col in (("per_task", "m1_per_task_mean"), ("pooled", "m1_pooled_mean"),
                           ("bal_acc", "a01_bal_acc_mean"), ("tpr", "a01_tpr_mean"),
                           ("far", "a01_far_mean"), ("det_time", "a01_det_time_mean")):
                add("existing row", key, q, r["test_set"], fname, r[col])
    for fname, mp in (("results/abc2/stress_summary.csv", {"A": "A_best", "C": "C_lr132", "emb": "emb_pooled"}),):
        for r in read_csv(REPO / fname):
            if r["cond"] == pool and r["detector"] in mp and r["block"] == "all":
                add("existing row", mp[r["detector"]], "stress_share", r["stress_set"], fname,
                    r["share_alarm_before_cut_mean"])
    for fname, mp in (("results/abc2/mlp_stress.csv", {"A-MLP": "A_mlp", "C-MLP": "C_mlp132"}),
                      ("results/abc2/b_stress.csv", {"B H=10": "B_combiner_h10"})):
        acc = defaultdict(list)
        for r in read_csv(REPO / fname):
            if r["cond"] == pool and r["detector"] in mp and r["block"] == "all":
                acc[(mp[r["detector"]], r["stress_set"])].append(float(r["share_alarm_before_cut"]))
        for (key, ss), v in acc.items():
            add("existing row", key, "stress_share", ss, fname, np.mean(v))
    if not common.MAIN:
        return out
    # results/abc (original training set only): A, C, B at each H, emb_temp[28].
    for r in read_csv(REPO / "results/abc/final_table.csv"):
        key = {"A": "A_best", "C": "C_lr132", "emb_temp[28] alone": "emb_pooled",
               "B H=0": "B_combiner_h0", "B H=10": "B_combiner_h10", "B H=25": "B_combiner_h25"}.get(r["detector"])
        if not key:
            continue
        for lab, ts in (("seen", "eval_seen"), ("unseen", "T0")):
            add("existing row", key, "per_task", ts, "results/abc/final_table.csv", r[f"m1_per_task_{lab}_mean"])
            add("existing row", key, "pooled", ts, "results/abc/final_table.csv", r[f"m1_pooled_{lab}_mean"])
        for q in ("bal_acc", "tpr", "det_time"):
            add("existing row", key, q, "T0", "results/abc/final_table.csv", r[f"a01_cut_{q}_mean"])
        if key == "B_combiner_h10":
            for lab, ts in (("seen", "eval_seen"), ("unseen", "T0")):
                add("refit check", "B_combiner_h10_refit", "per_task", ts, "results/abc/final_table.csv",
                    r[f"m1_per_task_{lab}_mean"])
            for q in ("bal_acc", "tpr"):
                add("refit check", "B_combiner_h10_refit", q, "T0", "results/abc/final_table.csv", r[f"a01_cut_{q}_mean"])
    # results/abc/b_final.csv: B's per-target trained combiners (families ablation).
    for r in read_csv(REPO / "results/abc/b_final.csv"):
        if r["model"].startswith("only_"):
            key = f"B_bytarget_h{r['H']}:combiner_{r['model'][5:]}"
            for lab, ts in (("seen", "eval_seen"), ("unseen", "T0")):
                add("existing row", key, "per_task", ts, "results/abc/b_final.csv", r[f"per_task_{lab}_mean"])
                add("existing row", key, "pooled", ts, "results/abc/b_final.csv", r[f"pooled_{lab}_mean"])
    # results/abc/c_final.csv: C's 132-input leave-one-family-out exists (not refitted here).
    # results/b_label_free (original training set): unweighted means of B.
    for r in read_csv(REPO / "results/b_label_free/lf_final.csv"):
        key = {"LF-all": "B_mean_h", "LF-emb": "B_meanemb_h"}[r["variant"]] + r["H"]
        for lab, ts in (("seen", "eval_seen"), ("unseen", "T0")):
            add("existing row", key, "per_task", ts, "results/b_label_free/lf_final.csv", r[f"per_task_{lab}_mean"])
            add("existing row", key, "pooled", ts, "results/b_label_free/lf_final.csv", r[f"pooled_{lab}_mean"])
        for q, col in (("bal_acc", "a01_cut_bal_acc_mean"), ("tpr", "a01_cut_tpr_mean"), ("det_time", "a01_cut_det_time_mean")):
            add("existing row", key, q, "T0", "results/b_label_free/lf_final.csv", r[col])
    # Baselines: AUROC exists in the Stage 2 table of this training set.
    sel = {}
    import detectors as det
    for key in ("emb_best", "token_entropy", "token_maxprob"):
        if det.done(key):
            sel[key] = det.load(key)["selection"]
    want = {("timestep", "none"): "timestep", ("failurespot", "0.8"): "failurespot",
            ("failurespot_neg", "0.8"): "failurespot_neg"}
    if "emb_best" in sel:
        want[("emb_temp", sel["emb_best"]["beta"], sel["emb_best"]["layer"])] = "emb_best"
    for k in ("token_entropy", "token_maxprob"):
        if k in sel:
            want[(sel[k]["series"], sel[k]["beta"])] = k
    for r in read_csv(REPO / "results/abc2/constraint_m1_summary.csv"):
        if r["cond"] != pool or r["test_set"] not in common.AUROC_SETS:
            continue
        key = want.get((r["constraint"], r["beta"], int(r["layer"]))) if r["constraint"] == "emb_temp" \
            else want.get((r["constraint"], r["beta"]))
        if r["constraint"] == "emb_temp" and r["variant"] != "primary":
            key = None
        if key:
            add("existing row", key, "per_task", r["test_set"], "results/abc2/constraint_m1_summary.csv", r["m1_per_task_mean"])
            add("existing row", key, "pooled", r["test_set"], "results/abc2/constraint_m1_summary.csv", r["m1_pooled_mean"])
    return out


def main() -> None:
    rows = checks()
    write_csv(common.OUT_DIR / "tables" / "data" / "reproduction_checks.csv", rows)
    bad = [r for r in rows if not r["passed"]]
    print(f"{len(rows)} checks, {len(bad)} failed")
    for r in bad:
        print("  FAIL", r)


if __name__ == "__main__":
    main()
