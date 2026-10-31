"""
b_constraint.py

B's constraint-prediction results (results_collection_handoff.md §4.4; F9, T4):

  grid       row-level AUROC of every grid probe against its own binarised look-ahead
             target, per H, on eval-seen, unseen tasks (original rollouts) and unseen
             tasks incl. LIBERO-PRO (probes_b.py, refit thresholds), checked against
             results/abc/b_grid.csv where it exists (original training set, chosen c).
  persistence  new: for each target and H, the target's current prepared value used as
             the prediction of its look-ahead label, row-level AUROC on the same rows.
  timestep   new: the timestep t as the prediction of the look-ahead label.
  mlp        B's 2-layer MLP at H = 10 (mlp_new.py b): row-level AUROC against its
             target, beside the linear probe and persistence on the same target.
  lf vs trained  paired per-seed differences, unweighted mean minus trained combiner,
             per H: copied from results/b_label_free/lf_vs_trained.csv (unseen
             original rollouts, original training set), and computed here from
             evaluate.py's tables for every test set.

Outputs: tables/data/b_target_auroc.csv (long: probe or baseline x H x test set x seed),
figures/data/b_target_auroc_heatmap.csv (F9), tables/data/b_mean_minus_combiner.csv,
tables/data/b_mlp_target.csv, tables/data/reproduction_checks_b_grid.csv.

Usage: source env.sh; python analysis/thesis_results/b_constraint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import HS, REPO, SEEDS, T, load_pkl, log, read_csv, write_csv  # noqa: E402


def grid_rows() -> list:
    rows = []
    for L in range(32):
        z = np.load(common.WORK_DIR / "b_probes" / f"L{L:02d}.npz")
        au = json.loads(str(z["auroc"]))
        hs = [int(h) for h in z["hs"]]
        for i, kd in enumerate(z["kinds"]):
            import setup_b as abc_b
            for hh in hs:
                col = i * len(hs) + hs.index(hh)
                for s in SEEDS:
                    for ts in common.AUROC_SETS:
                        rows.append({"model": "linear probe", "layer": L, "target kind": str(kd),
                                     "target": abc_b.target_name(str(kd), L), "H": hh,
                                     "test set": common.TEST_SETS[ts], "seed": s,
                                     "row-level AUROC against the target": round(au[str(s)][ts][col], 4),
                                     "threshold": round(float(z["thr"][list(SEEDS).index(s), col]), 4)})
    return rows


def baseline_rows(grid) -> list:
    """Persistence and timestep-only against each distinct target's look-ahead label."""
    import setup_b as abc_b
    ctx = common.ctx()
    kept = ctx.kept
    thr = {(r["target"], r["H"], r["seed"]): r["threshold"] for r in grid if r["test set"] == common.TEST_SETS["T0"]}
    targets = sorted({(r["target"], r["target kind"]) for r in grid})
    tt = np.broadcast_to(np.arange(T, dtype=np.float32), kept.shape)
    rows = []
    for s in SEEDS:
        P = common.b_targets(s)
        st = common.sets(s)
        for name, kd in targets:
            for H in HS:
                y = abc_b.lookahead(P[name], H)
                th = float(next(r["threshold"] for r in grid if r["target"] == name and r["H"] == H
                                and r["seed"] == s))
                # the threshold read back is rounded; use the stored full-precision one
                L = next(r["layer"] for r in grid if r["target"] == name)
                z = np.load(common.WORK_DIR / "b_probes" / f"L{L:02d}.npz")
                hs = [int(h) for h in z["hs"]]
                col = list(z["kinds"]).index(kd) * len(hs) + hs.index(H)
                th = float(z["thr"][list(SEEDS).index(s), col])
                for ts in common.AUROC_SETS:
                    pos = st[ts]
                    for model, score in (("persistence (current value)", P[name]), ("timestep only", tt)):
                        rows.append({"model": model, "layer": "", "target kind": kd, "target": name, "H": H,
                                     "test set": common.TEST_SETS[ts], "seed": s,
                                     "row-level AUROC against the target":
                                         round(abc_b.target_auroc(score, y, th, kept, pos), 4),
                                     "threshold": round(th, 4)})
    return rows


def check_rows(grid) -> list:
    if not common.MAIN:
        return []
    me = {(r["layer"], r["target kind"], r["H"], r["seed"], r["test set"]): r["row-level AUROC against the target"]
          for r in grid}
    out = []
    for r in read_csv(REPO / "results/abc/b_grid.csv"):
        if r["chosen_c"] != "True":
            continue
        for lab, ts in (("seen", "eval_seen"), ("unseen", "T0")):
            for s in SEEDS:
                k = (int(r["layer"]), r["target"], int(r["H"]), s, common.TEST_SETS[ts])
                if k in me:
                    ref = float(r[f"auroc_{lab}_s{s}"])
                    d = abs(me[k] - ref)
                    out.append({"layer": k[0], "target kind": k[1], "H": k[2], "seed": s, "test set": k[4],
                                "source_file": "results/abc/b_grid.csv", "source_value": ref,
                                "this_run": me[k], "abs_difference": round(d, 4), "tolerance": 0.005,
                                "passed": d <= 0.005 + 1e-4})
    return out


def heatmap(grid) -> list:
    """F9: layer x target kind, H = 10, mean over seeds, per test set."""
    out = []
    for ts in common.AUROC_SETS:
        for L in range(32):
            for kd in ("act_rep", "act_mag", "emb_temp", "xl_final"):
                v = [r["row-level AUROC against the target"] for r in grid
                     if r["layer"] == L and r["target kind"] == kd and r["H"] == 10 and r["test set"] == common.TEST_SETS[ts]]
                out.append({"layer": L, "target kind": kd, "H": 10, "test set": common.TEST_SETS[ts],
                            "row-level AUROC mean": round(float(np.mean(v)), 4) if v else "",
                            "row-level AUROC std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else "",
                            "sign-unstable target (no probe)": not v})
    return out


def mlp_rows(grid, base) -> list:
    p = common.WORK_DIR / "mlp" / "B_mlp_h10.pkl"
    if not p.exists():
        return []
    w = load_pkl(p)["selection"]
    out = []
    for s in SEEDS:
        info = w["per_seed"][s]
        name, L = info["target"], info["layer"]
        kd = info["probe"].split(":")[1]
        for ts in common.AUROC_SETS:
            tsn = common.TEST_SETS[ts]
            pers = next(r["row-level AUROC against the target"] for r in base if r["target"] == name and r["H"] == 10
                        and r["seed"] == s and r["test set"] == tsn and r["model"].startswith("persistence"))
            lin = next(r["row-level AUROC against the target"] for r in grid if r["layer"] == L and r["target kind"] == kd
                       and r["H"] == 10 and r["seed"] == s and r["test set"] == tsn)
            out.append({"seed": s, "probe": info["probe"], "target": name, "H": 10, "test set": tsn,
                        "2-layer MLP": round(info["mlp_target_auroc"][ts], 4), "linear probe": lin,
                        "persistence (current value)": pers, "weight decay": w["wd"],
                        "best epoch": info["best_epoch"], "epochs run": info["epochs_run"],
                        "train rows": info["n_train_rows"], "train positive share": round(info["train_positive_share"], 4)})
    return out


def lf_vs_trained() -> list:
    out = []
    if common.MAIN:
        for r in read_csv(REPO / "results/b_label_free/lf_vs_trained.csv"):
            if r["variant"] != "LF-all":
                continue
            out.append({"H": int(r["H"]), "test set": common.TEST_SETS["T0"], "source": "results/b_label_free/lf_vs_trained.csv",
                        "mean minus combiner, mean": float(r["diff_mean"]), "std": float(r["diff_std"]),
                        **{f"seed {s}": float(r[f"diff_s{s}"]) for s in SEEDS}})
    au = {(r["key"], r["test_set"], int(r["seed"])): float(r["m1_per_task"])
          for r in read_csv(common.OUT_DIR / "tables" / "data" / "detector_auroc_per_seed.csv")}
    for H in HS:
        for ts in common.AUROC_SETS:
            tsn = common.TEST_SETS[ts]
            d = [au[(f"B_mean_h{H}", tsn, s)] - au[(f"B_combiner_h{H}", tsn, s)] for s in SEEDS
                 if (f"B_combiner_h{H}", tsn, s) in au and (f"B_mean_h{H}", tsn, s) in au]
            if len(d) == len(SEEDS):
                out.append({"H": H, "test set": tsn, "source": "this run (evaluate.py)",
                            "mean minus combiner, mean": round(float(np.mean(d)), 4),
                            "std": round(float(np.std(d, ddof=1)), 4),
                            **{f"seed {s}": round(v, 4) for s, v in zip(SEEDS, d)}})
    return out


def persistence_summary(grid, base) -> list:
    """Per target kind, H and test set: mean row-level AUROC of the linear probes, of
    persistence and of the timestep score on the same targets, and the share of
    (probe, seed) pairs where the probe beats persistence. At H = 0 the label is the
    binarised current value, so persistence is 1 by construction."""
    per = {(r["target"], r["H"], r["test set"], r["seed"]): r["row-level AUROC against the target"]
           for r in base if r["model"].startswith("persistence")}
    tim = {(r["target"], r["H"], r["test set"], r["seed"]): r["row-level AUROC against the target"]
           for r in base if r["model"] == "timestep only"}
    acc = {}
    for r in grid:
        k = (r["target"], r["H"], r["test set"], r["seed"])
        acc.setdefault((r["target kind"], r["H"], r["test set"]), []).append(
            (r["row-level AUROC against the target"], per[k], tim[k]))
    out = []
    for (kd, H, ts), v in sorted(acc.items()):
        a = np.array(v, float)
        out.append({"target kind": kd, "H": H, "test set": ts, "probe-seed pairs": len(a),
                    "linear probes, mean row-level AUROC": round(float(a[:, 0].mean()), 4),
                    "persistence, mean row-level AUROC": round(float(a[:, 1].mean()), 4),
                    "timestep only, mean row-level AUROC": round(float(a[:, 2].mean()), 4),
                    "share of probes above persistence": round(float((a[:, 0] > a[:, 1]).mean()), 3)})
    return out


def main() -> None:
    grid = grid_rows()
    base = baseline_rows(grid)
    write_csv(common.OUT_DIR / "tables" / "data" / "b_target_auroc.csv", grid + base)
    write_csv(common.OUT_DIR / "figures" / "data" / "b_target_auroc_heatmap.csv", heatmap(grid))
    chk = check_rows(grid)
    write_csv(common.OUT_DIR / "tables" / "data" / "reproduction_checks_b_grid.csv", chk)
    write_csv(common.OUT_DIR / "tables" / "data" / "b_mlp_target.csv", mlp_rows(grid, base))
    write_csv(common.OUT_DIR / "tables" / "table4_b_linear_vs_persistence.csv", persistence_summary(grid, base))
    write_csv(common.OUT_DIR / "tables" / "data" / "b_mean_minus_combiner.csv", lf_vs_trained())
    bad = [c for c in chk if not c["passed"]]
    log(f"[b_constraint] grid rows {len(grid)}, baseline rows {len(base)}; grid checks {len(chk)}, "
        f"failed {len(bad)}, max diff {max([c['abs_difference'] for c in chk] or [0])}")


if __name__ == "__main__":
    main()
