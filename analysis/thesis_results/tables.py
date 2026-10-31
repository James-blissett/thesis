"""
tables.py

Thesis tables (results_collection_handoff.md §5, §7) as tidy CSVs with plain-word headers,
mean, std (over seeds, ddof 1) and per-seed columns. Reads the per-seed tables written by
evaluate.py and the other stages; nothing is recomputed here.

    tables/table1_core_detectors.csv              T1, headline test set (unseen tasks,
                                                  original rollouts) + stress-set share
    tables/table1_core_detectors_all_test_sets.csv  the same for every test set (long)
    (T2-T6 are added by later stages)

Usage: source env.sh; python analysis/thesis_results/tables.py [t1 ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import SEEDS, read_csv, write_csv  # noqa: E402

D = common.OUT_DIR / "tables"
DATA = D / "data"
PLAIN = {v: k for k, v in common.TEST_SETS.items()}


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan


def r4(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def add_stat(row, name, per_seed: dict):
    m, s = ms([per_seed.get(k) for k in SEEDS])
    row[f"{name} mean"] = r4(m)
    row[f"{name} std"] = r4(s)
    for k in SEEDS:
        row[f"{name} seed {k}"] = r4(per_seed.get(k, np.nan))


def detector_rows():
    """Per (key, test set, seed) metrics from evaluate.py's per-seed tables."""
    import evaluate as ev
    au = read_csv(DATA / "detector_auroc_per_seed.csv")
    al = [r for r in read_csv(DATA / "detector_alarm_per_seed.csv") if r["on_alpha_grid"] == "False"]
    st = [r for r in read_csv(DATA / "detector_stress_per_seed.csv") if r["block"] == "all"]
    M = defaultdict(dict)
    for r in au:
        k = (r["key"], PLAIN[r["test_set"]])
        s = int(r["seed"])
        M[k].setdefault("per_task", {})[s] = float(r["m1_per_task"])
        M[k].setdefault("pooled", {})[s] = float(r["m1_pooled"])
        M[k].setdefault("floor", {})[s] = float(r["floor"])
        M[k].setdefault("n_cells", {})[s] = int(r["n_cells"])
        M[k].setdefault("detail", {})[s] = r["detail"]
    for r in al:
        k = (r["key"], PLAIN[r["test_set"]])
        s = int(r["seed"])
        for q in ("bal_acc", "tpr", "far", "det_time"):
            M[k].setdefault(q, {})[s] = float(r[q])
        M[k].setdefault("n_cal", {})[s] = int(r["n_cal"])
    for r in st:
        k = (r["key"], r["stress_set"])
        M[k].setdefault("share", {})[int(r["seed"])] = float(r["share_alarm_before_cut"])
        M[k].setdefault("n", {})[int(r["seed"])] = int(r["n"])
    return M, ev


def source_of(key: str) -> str:
    import detectors as det
    base = key.split(":")[0]
    return det.load(base)["source"] if det.done(base) else ""


def core_row(M, ev, key, ts, with_stress=True):
    lab = ev.LABEL[key]
    m = M.get((key, ts))
    if not m:
        return None
    row = {"detector": lab[0], "variant": lab[1], "look-ahead H": lab[3], "test set": common.TEST_SETS[ts]}
    add_stat(row, common.PER_TASK_LABEL[ts], m["per_task"])
    row[f"{common.PER_TASK_LABEL[ts]}: cells averaged per seed"] = " ".join(str(m["n_cells"][s]) for s in SEEDS)
    add_stat(row, "pooled AUROC", m["pooled"])
    row["timestep-only floor (pooled AUROC)"] = r4(ms(m["floor"].values())[0])
    add_stat(row, "balanced accuracy at alpha 0.1", m["bal_acc"])
    add_stat(row, "share of failures caught at alpha 0.1", m["tpr"])
    add_stat(row, "false-alarm rate at alpha 0.1", m["far"])
    add_stat(row, "mean normalised detection time at alpha 0.1", m["det_time"])
    row["calibration rollouts"] = " ".join(str(m["n_cal"][s]) for s in SEEDS)
    if with_stress:
        sm = M.get((key, "stress"), {})
        add_stat(row, "stress set: share alarmed before the cut at alpha 0.1", sm.get("share", {}))
        if sm:
            row["stress set rollouts"] = sm["n"][SEEDS[0]]
    row["choice per seed"] = " | ".join(m["detail"][s] for s in SEEDS)
    row["source"] = source_of(key)
    row["row key"] = key
    return row


def t1() -> None:
    M, ev = detector_rows()
    head = [r for r in (core_row(M, ev, k, "T0") for k in [x[2] for x in ev.ROWS]) if r]
    write_csv(D / "table1_core_detectors.csv", head)
    allrows = []
    for ts in common.AUROC_SETS:
        allrows += [r for r in (core_row(M, ev, k, ts) for k in [x[2] for x in ev.ROWS]) if r]
    write_csv(D / "table1_core_detectors_all_test_sets.csv", allrows)
    # stress set, all subsets
    rows = []
    for k in [x[2] for x in ev.ROWS]:
        lab = ev.LABEL[k]
        for ss, nm in (("stress", "all position-swap and task-perturbed rollouts"),
                       ("stress_seen", "base task seen in training"), ("stress_unseen", "base task unseen")):
            sm = M.get((k, ss))
            if not sm:
                continue
            row = {"detector": lab[0], "variant": lab[1], "look-ahead H": lab[3], "stress subset": nm,
                   "rollouts": sm["n"][SEEDS[0]]}
            add_stat(row, "share alarmed before the cut at alpha 0.1", sm["share"])
            row["row key"] = k
            rows.append(row)
    write_csv(D / "table1_stress_set.csv", rows)
    print(f"T1: {len(head)} rows on the headline test set")




# --- T3: constraints --------------------------------------------------------------------------
def t3() -> None:
    """Best per family with sign, smoothing and per-task AUROC, plus the token, FailureSpot
    and timestep baselines. Read from this training set's Stage 2 table
    (results/abc2/constraint_m1_summary.csv); layered families pick their layer on
    eval-seen per-task AUROC (mean over seeds) at the family's chosen beta."""
    import detectors as det
    sel = common.selections()["betas"]
    fam = {"act_mag": "action", "act_dir": "action", "grip_flip": "action", "act_rep": "action",
           "emb_temp": "emb_temp", "xl_adj": "cross_layer", "xl_final": "cross_layer", "xl_spread": "cross_layer"}
    S = read_csv(common.REPO / "results/abc2/constraint_m1_summary.csv")
    S = [r for r in S if r["cond"] == common.POOL]
    by = defaultdict(dict)
    for r in S:
        by[(r["variant"], r["constraint"], r["layer"], r["beta"])][r["test_set"]] = r
    rows = []

    def emit(kind, name, key, note=""):
        d = by[key]
        row = {"kind": kind, "signal": name, "smoothing beta": key[3],
               "sign per seed (+: higher means failure)": d["eval_seen"]["signs_all_seeds"],
               "sign-unstable": d["eval_seen"]["unstable"] == "True"}
        for ts in common.AUROC_SETS:
            row[f"{common.PER_TASK_LABEL[ts]}, {common.TEST_SETS[ts]}, mean"] = r4(float(d[ts]["m1_per_task_mean"]))
            row[f"{common.PER_TASK_LABEL[ts]}, {common.TEST_SETS[ts]}, std"] = r4(float(d[ts]["m1_per_task_std"]))
        row[f"pooled AUROC, {HEAD_NAME}, mean"] = r4(float(d["T0"]["m1_pooled_mean"]))
        row[f"timestep-only floor, {HEAD_NAME}"] = r4(float(d["T0"]["floor_mean"]))
        row["note"] = note
        row["source"] = "results/abc2/constraint_m1_summary.csv"
        rows.append(row)

    for c in ("emb_temp", "xl_adj", "xl_final"):
        cand = [k for k in by if k[0] == "primary" and k[1] == c and k[3] == sel[fam[c]]]
        best = max(cand, key=lambda k: float(by[k]["eval_seen"]["m1_per_task_mean"]))
        emit("untrained constraint", f"{c}[{best[2]}]", best, "layer chosen on eval-seen per-task AUROC")
    for c in ("act_rep", "act_dir", "act_mag", "grip_flip", "xl_spread"):
        emit("untrained constraint", c, ("primary", c, "-1", sel[fam[c]]))
    for k, lab in (("token_entropy", "token mean entropy"), ("token_maxprob", "token max-probability")):
        s_ = det.load(k)["selection"]
        emit("baseline", lab, ("baseline", s_["series"], "-1", s_["beta"]), "beta chosen on eval-seen per-task AUROC")
    emit("baseline", "FailureSpot signal s", ("failurespot", "failurespot", "-1", "0.8"), "fixed orientation")
    emit("baseline", "FailureSpot signal -s", ("failurespot", "failurespot_neg", "-1", "0.8"), "fixed orientation")
    emit("baseline", "timestep only", ("baseline", "timestep", "-1", "none"), "0.5 per task by construction")
    write_csv(D / "table3_constraints.csv", rows)


HEAD_NAME = common.TEST_SETS["T0"]


def _au(M, key, ts):
    m = M.get((key, ts))
    return m["per_task"] if m else {}


# --- T4: B per H, and B's MLP --------------------------------------------------------------
def t4() -> None:
    M, ev = detector_rows()
    rows = []
    for H in common.HS:
        for key, lab in ((f"B_best_h{H}", "single best probe"), (f"B_mean_h{H}", "unweighted mean of all grid probes (primary)"),
                         (f"B_meanemb_h{H}", "unweighted mean of the embedding-temporal probes"),
                         (f"B_combiner_h{H}", "trained combiner over all grid probes")):
            for ts in ("T0", "T2"):
                if (key, ts) not in M:
                    continue
                row = {"look-ahead H": H, "row": lab, "test set": common.TEST_SETS[ts]}
                add_stat(row, common.PER_TASK_LABEL[ts], M[(key, ts)]["per_task"])
                add_stat(row, "balanced accuracy at alpha 0.1", M[(key, ts)]["bal_acc"])
                row["choice per seed"] = " | ".join(M[(key, ts)]["detail"][s] for s in SEEDS)
                rows.append(row)
        for ts in ("T0", "T2"):
            a, b = _au(M, f"B_mean_h{H}", ts), _au(M, f"B_combiner_h{H}", ts)
            if a and b:
                row = {"look-ahead H": H, "row": "paired difference, unweighted mean minus trained combiner",
                       "test set": common.TEST_SETS[ts]}
                add_stat(row, common.PER_TASK_LABEL[ts], {s: a[s] - b[s] for s in SEEDS})
                rows.append(row)
    write_csv(D / "table4_b_by_lookahead.csv", rows)
    # B's MLP vs best linear probe vs persistence
    mt = read_csv(DATA / "b_mlp_target.csv")
    out = []
    for ts in common.AUROC_SETS:
        tsn = common.TEST_SETS[ts]
        R = {int(r["seed"]): r for r in mt if r["test set"] == tsn}
        if not R:
            continue
        for col in ("2-layer MLP", "linear probe", "persistence (current value)"):
            row = {"comparison": "row-level AUROC against the probe's own target (H = 10)", "model": col, "test set": tsn}
            add_stat(row, "AUROC", {s: float(R[s][col]) for s in R})
            row["target per seed"] = " | ".join(R[s]["target"] for s in sorted(R))
            out.append(row)
    for ts in ("T0", "T2"):
        for key, lab in (("B_mlp_h10", "2-layer MLP"), ("B_best_h10", "linear probe")):
            if (key, ts) in M:
                row = {"comparison": "per-task AUROC against outcome (running sum), and alarm", "model": lab,
                       "test set": common.TEST_SETS[ts]}
                add_stat(row, "AUROC", M[(key, ts)]["per_task"])
                add_stat(row, "balanced accuracy at alpha 0.1", M[(key, ts)]["bal_acc"])
                out.append(row)
    write_csv(D / "table4_b_mlp_vs_linear_vs_persistence.csv", out)


# --- T5: families and look-ahead ------------------------------------------------------------
def t5() -> None:
    M, ev = detector_rows()
    import evaluate
    rows = []
    for key in [k for k in evaluate.available_keys() if k.startswith("C_mean_lofo") or k.startswith("B_bytarget")] \
            + ["C_mean"] + [f"B_mean_h{H}" for H in common.HS] + [f"B_combiner_h{H}" for H in common.HS]:
        lab = evaluate.sub_label(key)
        for ts in ("T0", "T2"):
            if (key, ts) not in M:
                continue
            row = {"ablation": "constraint families" if ":" in key or key == "C_mean" else "look-ahead H",
                   "detector": lab["detector"], "variant": lab["variant"], "look-ahead H": lab["H"],
                   "test set": common.TEST_SETS[ts], "source": "this run"}
            add_stat(row, common.PER_TASK_LABEL[ts], M[(key, ts)]["per_task"])
            add_stat(row, "balanced accuracy at alpha 0.1", M[(key, ts)]["bal_acc"])
            rows.append(row)
    if common.MAIN:
        for r in read_csv(common.REPO / "results/abc/c_final.csv"):
            row = {"ablation": "constraint families", "detector": "C (exploratory)",
                   "variant": "logistic regression on all 132 constraints" + ("" if r["model"] == "full" else
                              f", without the {r['model'][3:].replace('_', '-')} family"),
                   "look-ahead H": "", "test set": HEAD_NAME, "source": "results/abc/c_final.csv (existing version)"}
            add_stat(row, "per-task AUROC", {s: float(r[f"per_task_unseen_s{s}"]) for s in SEEDS})
            rows.append(row)
    write_csv(D / "table5_ablations_families_and_lookahead.csv", rows)


# --- T6: training data ------------------------------------------------------------------------
def t6() -> None:
    """Original / 500-rollout / full training sets x the three unseen test sets, for A
    (single best probe), C (132-input logistic regression) and the best untrained
    constraint, copied from results/abc2/report_tables; B joins from the background run."""
    names = {"A": "A, single best probe", "C": "C (exploratory), logistic regression on all 132 constraints",
             "emb": "best untrained constraint (emb_temp, pooled layer choice)"}
    pools = {"D0": "original training set", "D1": "500-rollout training set", "D2": "full training set"}
    sets_ = {"T0": "unseen tasks, original rollouts", "T1": "unseen tasks, original + new unperturbed rollouts",
             "T2": "unseen tasks incl. LIBERO-PRO"}
    al = {(r["test_set"], r["detector"], r["cond"]): r for r in read_csv(common.REPO / "results/abc2/report_tables/alarm.csv")}
    rows = []
    for r in read_csv(common.REPO / "results/abc2/report_tables/ranking.csv"):
        for ts, tn in sets_.items():
            a = al.get((ts, r["detector"], r["cond"]), {})
            rows.append({"detector": names[r["detector"]], "training set": pools[r["cond"]], "test set": tn,
                         "per-task AUROC mean": r4(f(r[f"{ts}_m1_per_task_mean"])),
                         "per-task AUROC std": r4(f(r[f"{ts}_m1_per_task_std"])),
                         "balanced accuracy at alpha 0.1 mean": r4(f(a.get("a01_bal_acc_mean"))),
                         "balanced accuracy at alpha 0.1 std": r4(f(a.get("a01_bal_acc_std"))),
                         "source": "results/abc2/report_tables/ranking.csv, alarm.csv"})
    # Unweighted means (and B), original vs full training set, from each run's own core table.
    for tag, base in (("original training set", common.REPO / "results/thesis_results"),
                      ("full training set", common.REPO / "results/thesis_results_full")):
        f_ = base / "tables" / "table1_core_detectors_all_test_sets.csv"
        if not f_.exists():
            continue
        for r in read_csv(f_):
            if r["row key"] not in ("A_mean", "C_mean", "B_mean_h0", "B_mean_h10", "B_mean_h25", "emb_best"):
                continue
            if r["test set"] == common.TEST_SETS["eval_seen"]:
                continue
            pt = [c for c in r if c.endswith("AUROC mean") and "pooled" not in c][0][:-5]
            lab = f"{r['detector']}, {r['variant']}" + (f", H = {r['look-ahead H']}" if r["look-ahead H"] else "")
            rows.append({"detector": lab, "training set": tag, "test set": r["test set"],
                         "per-task AUROC mean": r[f"{pt} mean"], "per-task AUROC std": r[f"{pt} std"],
                         "balanced accuracy at alpha 0.1 mean": r["balanced accuracy at alpha 0.1 mean"],
                         "balanced accuracy at alpha 0.1 std": r["balanced accuracy at alpha 0.1 std"],
                         "source": str(f_.relative_to(common.REPO))})
    full = common.REPO / "results/thesis_results_full/tables/training_set_comparison.csv"
    if full.exists():
        write_csv(D / "table6_training_data_paired.csv", read_csv(full))
    write_csv(D / "table6_training_data.csv", rows)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def all_tables() -> None:
    for fn in (t1, t3, t4, t5, t6):
        try:
            fn()
            print(f"{fn.__name__} ok")
        except Exception as e:
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["all_tables"]
    for p in parts:
        globals()[p]()
