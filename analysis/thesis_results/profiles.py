"""
profiles.py

Per-probe outcome profiles (results_collection_handoff.md §4.3; figures F6, F7, F8, F10):
per seed, on eval-seen and both unseen test sets, per-task AUROC against episode outcome of
the running-sum score (same cut as every detector), for

  A     every probe (34; refitted, probes_a.py) and its shuffled-label control (mean over
        5 permutations of the train labels, per probe and seed)
  C     every probe (33; refitted, probes_c.py)
  B     every probe (124 at each H; probes_b.py) and B per layer: the unweighted mean of
        that layer's probes (up to 4)
  untrained constraints per layer (C's inputs: emb_temp[l], xl_adj[l], xl_final[l]; the
        action constraints), at each family's chosen beta, two scorings:
          running maximum  the Methods' scoring for untrained constraints (exists:
                           results/abc2/constraint_m1.csv, read, not recomputed)
          running sum      new: the same prepared series (missing rows 0, as model
                           inputs), summed over kept rows, AUROC of the value at the cut
        plus, per layer, the running sum of the mean of that layer's three prepared
        constraints (and of the 4 action constraints), and the "best of the three" per
        layer, chosen on mean eval-seen per-task AUROC (never on unseen rollouts).
  each detector's unweighted mean (the horizontal line in F6-F8).

Output: figures/data/per_probe_outcome_auroc.csv with columns
  detector, series, probe, layer (blank if the probe reads no layer), target (B), H (B),
  scoring, test set, seed, per-task AUROC, cells averaged, unstable
and checks against the earlier per-probe tables (results/abc2/a_probes.csv and
c_probes.csv, this training set) in tables/data/reproduction_checks_probes.csv.

Usage: source env.sh; python analysis/thesis_results/profiles.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import detectors as det  # noqa: E402
from common import HS, REPO, SEEDS, T, log, m1_safe, read_csv, write_csv  # noqa: E402

OUT = common.OUT_DIR / "figures" / "data" / "per_probe_outcome_auroc.csv"


def stacked_m1(scores, s):
    """{test set key: (per_task (k,), n_cells)} for stacked (k, N, T) scores."""
    ctx = common.ctx()
    st = common.sets(s)
    out = {}
    for ts in common.AUROC_SETS:
        m = m1_safe(scores, ctx.kept, ctx.idx, st[ts])
        out[ts] = (np.atleast_1d(np.asarray(m["per_task"], float)), int(m["n_tasks"]))
    return out


def rsum(P):
    kept = common.ctx().kept
    return np.cumsum(np.where(kept[None], P, 0.0), axis=-1, dtype=np.float32)


def rows_for(detector, series, names, layers, targets, hs, scoring, res, s, unstable=None):
    out = []
    for ts, (pt, nc) in res.items():
        for k, nm in enumerate(names):
            out.append({"detector": detector, "series": series, "probe": nm,
                        "layer": "" if layers[k] is None else layers[k],
                        "target": targets[k] if targets else "", "H": hs[k] if hs else "",
                        "scoring": scoring, "test set": common.TEST_SETS[ts], "seed": s,
                        "per-task AUROC": round(float(pt[k]), 4), "cells averaged": nc,
                        "unstable": "" if unstable is None else bool(unstable[k])})
    return out


def a_rows() -> list:
    import setup_a as abc_a
    rows = []
    names = list(abc_a.PROBES)
    layers = [abc_a.probe_layer(n) for n in names]
    for s in SEEDS:
        _, ref, _ = det.a_outputs(s)
        res = stacked_m1(rsum(ref.astype(np.float32)), s)
        rows += rows_for("A", "probe", names, layers, None, None, "running sum", res, s)
        sh = {ts: (np.zeros(len(names)), 0) for ts in common.AUROC_SETS}
        for k, n in enumerate(names):
            z = np.load(common.WORK_DIR / "a_probes" / f"{n}.npz")
            d = json.loads(str(z["shuffled"]))[str(s)]
            for ts in common.AUROC_SETS:
                sh[ts][0][k] = np.mean([r[ts]["m1_per_task"] for r in d])
                sh[ts] = (sh[ts][0], d[0][ts]["n_cells"])
        rows += rows_for("A", "shuffled-label control (mean of 5 permutations)", names, layers, None, None,
                         "running sum", sh, s)
    return rows


def c_rows() -> list:
    rows = []
    for s in SEEDS:
        names, ref, _, _ = det.c_outputs(s)
        layers = [int(n[5:]) if n.startswith("layer") else None for n in names]
        rows += rows_for("C", "probe", names, layers, None, None, "running sum", stacked_m1(rsum(ref), s), s)
    return rows


def b_rows() -> list:
    rows = []
    meta = json.loads((common.WORK_DIR / "b_probes" / "meta.json").read_text())
    for H in HS:
        for s in SEEDS:
            names, P = det.b_outputs(H, s)
            layers = [int(n[1:3]) for n in names]
            kinds = [n.split(":")[1] for n in names]
            import setup_b as abc_b
            targets = [abc_b.target_name(kd, L) for kd, L in zip(kinds, layers)]
            rows += rows_for("B", "probe", names, layers, targets, [H] * len(names), "running sum",
                             stacked_m1(rsum(P), s), s)
            Lm = sorted(set(layers))
            M = np.stack([P[[k for k, L in enumerate(layers) if L == LL]].mean(0) for LL in Lm])
            rows += rows_for("B", "per-layer mean of B's probes", [f"layer{L:02d} mean" for L in Lm], Lm,
                             None, [H] * len(Lm), "running sum", stacked_m1(rsum(M), s), s)
            del P, M
    return rows


def constraint_rows() -> list:
    """Untrained constraints: running maximum (read) and running sum (new)."""
    import prep
    import stage3_c
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    sel = common.selections()
    rows = []
    # running maximum: read from the Stage 2 table of this training set
    fam_beta = {"act_mag": "action", "act_dir": "action", "grip_flip": "action", "act_rep": "action",
                "emb_temp": "emb_temp", "xl_adj": "cross_layer", "xl_final": "cross_layer"}
    for r in read_csv(REPO / "results/abc2/constraint_m1.csv"):
        if (r["cond"] != common.POOL or r["variant"] != "primary" or r["constraint"] not in fam_beta
                or r["beta"] != sel["betas"][fam_beta[r["constraint"]]] or r["test_set"] not in common.AUROC_SETS):
            continue
        L = int(r["layer"])
        rows.append({"detector": "untrained constraint", "series": r["constraint"],
                     "probe": f"{r['constraint']}[{L}]" if L >= 0 else r["constraint"],
                     "layer": "" if L < 0 else L, "target": "", "H": "", "scoring": "running maximum",
                     "test set": common.TEST_SETS[r["test_set"]], "seed": int(r["seed"]),
                     "per-task AUROC": round(float(r["m1_per_task"]), 4), "cells averaged": int(r["n_cells"]),
                     "unstable": r["unstable"] == "True"})
    # running sum: new
    fams = stage3_c.smoothed_families(ctx, sel["betas"])
    unst = {(r["constraint"], int(r["layer"])) for r in read_csv(REPO / "results/abc2/constraint_m1_summary.csv")
            if r["cond"] == common.POOL and r["variant"] == "primary" and r["test_set"] == "eval_seen"
            and r["unstable"] == "True" and r["constraint"] in fam_beta
            and r["beta"] == sel["betas"][fam_beta[r["constraint"]]]}
    for s in SEEDS:
        feats, _ = common.abc_c.seed_features(fams, idx, kept, common.sets(s)["train"])
        names = [n for n in feats if not n.startswith("|") and n != "xl_spread"]
        base = [common.abc_c.base_of(n) for n in names]
        P = np.stack([feats[n] for n in names])
        res = stacked_m1(rsum(P), s)
        rows += rows_for("untrained constraint", "", names, [L if L >= 0 else None for _, L in base], None, None,
                         "running sum", res, s, [b in unst for b in base])
        for r in rows[-len(names) * len(common.AUROC_SETS):]:
            r["series"] = common.abc_c.base_of(r["probe"])[0]
        # per-layer mean of the three prepared constraints, and of the 4 action constraints
        groups = {f"layer{L:02d} mean of 3": [f"emb_temp[{L}]", f"xl_adj[{min(L, 30)}]", f"xl_final[{min(L, 30)}]"]
                  for L in range(32)}
        groups["action mean of 4"] = ["act_mag", "act_dir", "grip_flip", "act_rep"]
        G = np.stack([np.mean([feats[n] for n in g], 0) for g in groups.values()])
        gl = [int(k[5:7]) if k.startswith("layer") else None for k in groups]
        rows += rows_for("untrained constraint", "mean of the layer's prepared constraints", list(groups), gl,
                         None, None, "running sum", stacked_m1(rsum(G), s), s)
        del P, G, feats
    # best of the three per layer (running sum), chosen on mean eval-seen per-task AUROC
    rs = [r for r in rows if r["detector"] == "untrained constraint" and r["scoring"] == "running sum"
          and r["series"] in ("emb_temp", "xl_adj", "xl_final")]
    seen = {}
    for r in rs:
        if r["test set"] == common.TEST_SETS["eval_seen"]:
            seen.setdefault((r["probe"]), []).append(r["per-task AUROC"])
    best = []
    for L in range(32):
        cands = [f"emb_temp[{L}]", f"xl_adj[{min(L, 30)}]", f"xl_final[{min(L, 30)}]"]
        b = max(cands, key=lambda c: np.mean(seen[c]))
        for r in rs:
            if r["probe"] == b:
                best.append({**r, "series": "best of the layer's three (chosen on eval-seen)", "layer": L,
                             "probe": f"layer{L:02d}: {b}"})
    return rows + best


def mean_rows() -> list:
    """Each detector's unweighted mean (F6-F8's horizontal line), from evaluate.py."""
    rows = []
    want = {"A_mean": ("A", ""), "C_mean": ("C", "")}
    want.update({f"B_mean_h{H}": ("B", H) for H in HS})
    for r in read_csv(common.OUT_DIR / "tables" / "data" / "detector_auroc_per_seed.csv"):
        if r["key"] in want:
            d, H = want[r["key"]]
            rows.append({"detector": d, "series": "unweighted mean (final detector)", "probe": "unweighted mean",
                         "layer": "", "target": "", "H": H, "scoring": "running sum", "test set": r["test_set"],
                         "seed": int(r["seed"]), "per-task AUROC": round(float(r["m1_per_task"]), 4),
                         "cells averaged": int(r["n_cells"]), "unstable": ""})
    return rows


def check_rows(rows) -> list:
    """Refitted A and C probes against results/abc2/{a,c}_probes.csv (this training set)."""
    me = {(r["detector"], r["probe"], r["test set"], r["seed"]): r["per-task AUROC"]
          for r in rows if r["series"] == "probe" and r["detector"] in ("A", "C")}
    out = []
    for det_, f in (("A", "results/abc2/a_probes.csv"), ("C", "results/abc2/c_probes.csv")):
        for r in read_csv(REPO / f):
            if r["cond"] != common.POOL or r["test_set"] not in common.AUROC_SETS:
                continue
            k = (det_, r["probe"], common.TEST_SETS[r["test_set"]], int(r["seed"]))
            if k in me:
                d = abs(me[k] - float(r["m1_per_task"]))
                out.append({"detector": det_, "probe": r["probe"], "test set": k[2], "seed": k[3],
                            "source_file": f, "source_value": round(float(r["m1_per_task"]), 4),
                            "this_run": me[k], "abs_difference": round(d, 4), "tolerance": 0.005,
                            "passed": d <= 0.005 + 1e-4})
    return out


def main() -> None:
    t0 = time.time()
    rows = []
    for name, fn in (("A", a_rows), ("C", c_rows), ("B", b_rows), ("constraints", constraint_rows),
                     ("means", mean_rows)):
        rows += fn()
        log(f"[profiles] {name}: {len(rows)} rows ({time.time() - t0:.0f} s)")
    write_csv(OUT, rows)
    chk = check_rows(rows)
    write_csv(common.OUT_DIR / "tables" / "data" / "reproduction_checks_probes.csv", chk)
    bad = [c for c in chk if not c["passed"]]
    log(f"[profiles] {len(rows)} rows -> {OUT}; probe checks {len(chk)}, failed {len(bad)}; "
        f"max diff {max([c['abs_difference'] for c in chk] or [0])}")


if __name__ == "__main__":
    main()
