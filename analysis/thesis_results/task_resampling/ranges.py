"""
ranges.py

§1-§3 of task_resampling_handoff.md: the 95% ranges that also redraw which unseen tasks
are included (engine: tr.py), for the comparison list of §2, plus §6's "without layer 0"
rows.

  check    self-check with step 1 switched off, against tables/paired_comparisons.csv:
           (a) with bootstrap2's own rollout draws (1,000, its seed): must agree to the
               CSV's 4 decimals (same draws, same computation);
           (b) with this run's draws (2,000 rounds, seed 20261101): within +-0.005.
           -> logs/self_check.json, logs/self_check.csv
  ranges   §2 -> ranges.csv (and logs/ranges_full.csv with the extra columns)
  layer0   §6 extra check (needs layer0.py's npz) -> b_without_layer0.csv

Usage: source env.sh; python analysis/thesis_results/task_resampling/ranges.py [check ranges layer0]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr  # noqa: E402
from tr import (NPZ_COMMON_FULL, NPZ_COMMON_ORIG, NPZ_FULL, NPZ_LAYER0, NPZ_ORIG, NPZ_SMOOTH, OUT,  # noqa: E402
                Comparison, Engine, common, log, read_csv, write_csv)

TOL = 0.005
LOGS = OUT / "logs"


# --- self-check -------------------------------------------------------------------------
def check() -> None:
    tr.quiet()
    mkey = {"per-task-and-perturbation AUROC": "m1_per_task", "balanced accuracy at alpha 0.1": "bal_acc"}
    ref = [r for r in read_csv(tr.fc.OUT / "tables" / "paired_comparisons.csv") if r["metric"] in mkey]
    comps, seen = [], set()
    for r in ref:
        k = (r["x key"], r["y key"])
        if k not in seen:
            seen.add(k)
            comps.append(Comparison("check", r["comparison (x - y)"], (NPZ_FULL, r["x key"]), (NPZ_FULL, r["y key"])))
    eng_a = Engine(plan_fn=lambda sides: tr.plan_from_b2(sides))
    eng_b = Engine(resample_tasks=False)
    got = {}
    for c in comps:
        for tag, eng in (("a", eng_a), ("b", eng_b)):
            for row in eng.run(c):
                got[(tag, c.x[1], c.y[1], row["metric"])] = row
    rows, worst = [], {"a": 0.0, "b": 0.0}
    for r in ref:
        m = {"m1_per_task": "per-task AUROC", "bal_acc": "balanced accuracy at alpha 0.1"}[mkey[r["metric"]]]
        out = {"comparison": r["comparison (x - y)"], "metric": r["metric"],
               "existing difference": float(r["difference"]),
               "existing range low": float(r["95% interval low"]), "existing range high": float(r["95% interval high"])}
        for tag in ("a", "b"):
            g = got[(tag, r["x key"], r["y key"], m)]
            d = [abs(g["difference"] - out["existing difference"]), abs(g["95% range low"] - out["existing range low"]),
                 abs(g["95% range high"] - out["existing range high"])]
            worst[tag] = max(worst[tag], max(d))
            name = {"a": "bootstrap2's draws", "b": "this run's draws, 2,000 rounds"}[tag]
            out[f"{name}: difference"] = round(g["difference"], 4)
            out[f"{name}: range low"] = round(g["95% range low"], 4)
            out[f"{name}: range high"] = round(g["95% range high"], 4)
            out[f"{name}: max |change|"] = round(max(d), 4)
            out[f"{name}: rounds used"] = g["rounds used"]
        rows.append(out)
    res = {"rows": len(rows),
           "a_bootstrap2_draws": {"max_abs_diff": worst["a"], "tolerance": 1e-4 * 0.5 + 1e-9,
                                  "passed": worst["a"] <= 0.5e-4 + 1e-9,
                                  "note": "same draws as tables/paired_comparisons.csv; CSV values are rounded to 4 d.p."},
           "b_fresh_draws": {"max_abs_diff": worst["b"], "tolerance": TOL, "passed": worst["b"] <= TOL,
                             "rounds": tr.N_ROUNDS, "rng_seed": tr.RNG_SEED}}
    LOGS.mkdir(parents=True, exist_ok=True)
    write_csv(LOGS / "self_check.csv", rows)
    common.write_json(LOGS / "self_check.json", res)
    log(f"[ranges] self-check (a) bootstrap2 draws max |diff| {worst['a']:.2e} "
        f"({'passed' if res['a_bootstrap2_draws']['passed'] else 'FAILED'}); "
        f"(b) fresh draws max |diff| {worst['b']:.4f} ({'passed' if res['b_fresh_draws']['passed'] else 'FAILED'})")


# --- §2 ---------------------------------------------------------------------------------
def comparisons() -> list:
    F, O = NPZ_FULL, NPZ_ORIG
    AU, BOTH = ("m1_per_task",), ("m1_per_task", "bal_acc")
    c = []
    s = "1. headline"
    c += [Comparison(s, "B mean - A mean", (F, "B_mean_h10"), (F, "A_mean")),
          Comparison(s, "B mean - C mean", (F, "B_mean_h10"), (F, "C_mean")),
          Comparison(s, "A mean - C mean", (F, "A_mean"), (F, "C_mean"))]
    s = "2. within each detector"
    for d, sfx in (("A", ""), ("B", "_h10"), ("C", "")):
        for k, name in (("best", "single best probe"), ("combiner", "trained combiner"), ("mlp", "2-layer MLP")):
            c.append(Comparison(s, f"{d} mean - {d} {name}", (F, f"{d}_mean{sfx}"), (F, f"{d}_{k}{sfx}"), metrics=AU))
    s = "3. B embedding-temporal mean"
    for H in (0, 10, 25):
        for y, name in (("A_mean", "A mean"), ("C_mean", "C mean"), ("emb_best", "untrained emb_temp[26]")):
            c.append(Comparison(s, f"B embedding-temporal mean (H = {H}) - {name}", (F, f"B_meanemb_h{H}"), (F, y)))
    s = "4. per perturbation type"
    for fam, name in tr.FAMILY.items():
        c.append(Comparison(s, f"B mean - A mean, {name} groups", (F, "B_mean_h10"), (F, "A_mean"), family=fam))
    s = "5. training data (full - original)"
    c += [Comparison(s, "A mean: full - original training set", (F, "A_mean"), (O, "A_mean")),
          Comparison(s, "C mean: full - original training set", (F, "C_mean"), (O, "C_mean")),
          Comparison(s, "B mean on the 113 shared probes: full - original training set",
                     (NPZ_COMMON_FULL, "F_B_common_h10"), (NPZ_COMMON_ORIG, "F_B_common_h10"))]
    s = "6. smoothing (setting - selected)"
    for st, name in (("none", "no smoothing"), ("b083", "beta = 0.83 everywhere"), ("b096", "beta = 0.96 everywhere")):
        for d, k in (("C mean", "C_mean"), ("B mean", "B_mean_h10")):
            c.append(Comparison(s, f"{d}: {name} - selected", (NPZ_SMOOTH, f"SM_{st}__{k}"), (NPZ_SMOOTH, f"SM_current__{k}")))
    s = "7. setting closest to SAFE's"
    c.append(Comparison(s, "B mean - A mean, original training set, unseen tasks' original rollouts",
                        (O, "B_mean_h10"), (O, "A_mean"), test_set="T0", metrics=("m1_per_task", "m1_pooled")))
    s = "8. B against the untrained constraint"
    c.append(Comparison(s, "B mean - untrained emb_temp[26]", (F, "B_mean_h10"), (F, "emb_best")))
    c.append(Comparison(s, "B mean, unseen tasks' original rollouts: original - full training set",
                        (O, "B_mean_h10"), (F, "B_mean_h10"), test_set="T0"))
    return c


SHOW = ["section", "comparison", "metric", "difference", "95% range low", "95% range high", "range excludes zero",
        "rounds used"]


def tidy(rows):
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 4) if np.isfinite(v) else ""
    return rows


def ranges() -> None:
    tr.quiet()
    eng = Engine()
    rows = []
    for c in comparisons():
        rows += eng.run(c)
        log(f"[ranges] {c.section}: {c.title}")
    tidy(rows)
    write_csv(LOGS / "ranges_full.csv", rows)
    write_csv(OUT / "ranges.csv", [{**{k: r[k] for k in SHOW}, "rounds skipped": r["rounds skipped"],
                                    "rounds with a group dropped": r["rounds with a group dropped"],
                                    "test set": r["test set"]} for r in rows])
    log(f"[ranges] {len(rows)} rows -> {OUT / 'ranges.csv'}")


def layer0() -> None:
    tr.quiet()
    if not all(p.exists() for p in NPZ_LAYER0):
        raise SystemExit("run layer0.py first (npz of the without-layer-0 means)")
    L = NPZ_LAYER0
    eng = Engine()
    s = "6. without layer 0"
    comps = [Comparison(s, "B mean without layer 0 - B mean (H = 10)", (L, "TR_B_nolayer0_h10"), (L, "B_mean_h10"),
                        metrics=("m1_per_task",)),
             Comparison(s, "A mean without layer 0 - A mean", (L, "TR_A_nolayer0"), (L, "A_mean"),
                        metrics=("m1_per_task",))]
    rows = []
    for c in comps:
        for r in eng.run(c):
            sd = eng.side(*c.x, c.test_set, c.family)[1]
            per = tr.side_rounds(sd, tr.identity_plan(sd))[1]["m1_per_task"][:, 0]
            r["without layer 0, per-task AUROC, mean over splits"] = r["x value"]
            r["without layer 0, std over splits"] = float(np.std(per, ddof=1))
            r["full mean, per-task AUROC"] = r["y value"]
            rows.append(r)
    tidy(rows)
    write_csv(OUT / "b_without_layer0.csv",
              [{"row": r["comparison"].split(" - ")[0], "per-task AUROC (mean over splits)": r["x value"],
                "std over splits": r["without layer 0, std over splits"], "full mean": r["y value"],
                "difference (without layer 0 - full)": r["difference"], "95% range low": r["95% range low"],
                "95% range high": r["95% range high"], "range excludes zero": r["range excludes zero"],
                "rounds used": r["rounds used"], "x key": r["x key"]} for r in rows])
    log(f"[ranges] layer0: {len(rows)} rows")


if __name__ == "__main__":
    for step in (sys.argv[1:] or ["check", "ranges"]):
        globals()[step]()
