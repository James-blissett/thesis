"""
stage2_constraints.py

Stage 2 of the port (analysis/abc/constraint_m1.py, generalised to an index, a condition
and the Step 7 test sets): every constraint scored against episode outcome on its own,
per seed and smoothing level, plus the token baselines, the timestep-only score and
FailureSpot. Then the condition-level selections that analysis/abc took from Stage 2's
table, made on THIS condition's eval_seen set:

  * beta per family (setup_c.choose_betas, verbatim rule): for each family (action;
    emb_temp; cross_layer) and beta, take the series with the best eval_seen
    M1-per-task (mean over seeds, rounded to 4 decimals as abc read it back from
    constraint_m1.csv); the family's beta is the one whose best series is highest.
    Ties go to the first series / first beta, as max() did.
  * the comparison emb_temp layer (how emb_temp[28] was found): constraint_m1.summarise
    picks, at each beta, the layer with the best eval_seen POOLED M1 (mean over seeds);
    setup_c / cp_eval then used that layer at the chosen emb_temp beta. Applied here as:
    argmax over layers of mean eval_seen pooled M1 at the chosen emb_temp beta.
    (By eval_seen M1-per-task the D0 answer would be layer 26, not 28.)
  * unstable flags: sign differs across the seeds run (at every beta; C carries the
    flags at its chosen betas).

Preparation (prep.py, imported): ffill -> EMA(beta) -> scale on the condition's train
successes -> orient by per-cell train M1-per-task. Evaluated on eval_seen, T0, T1, T2.

Work files (resumable per (cond, seed)):  <work>/<cond>/s2_seed{s}.pkl
Tables (per condition):  <out>/tables/<cond>/constraint_m1.csv (long, one row per
(seed, test_set, series, beta)), constraint_m1_summary.csv (mean +- std over seeds),
c_betas.csv (the family x beta grid behind the beta choice).
"""

from __future__ import annotations

import time

import numpy as np

import prep
from pipe2_common import (EVAL_SETS, FAILURESPOT_BETA, Ctx, PRIMARY2, log, load_pkl,  # noqa: F401
                          m1_row, m1_safe, mean_std, record_timing, save_pkl, write_csv)
from data2 import train_kept

C_FAMILIES = {"action": ["act_mag", "act_dir", "grip_flip", "act_rep"],
              "emb_temp": ["emb_temp"],
              "cross_layer": ["xl_adj", "xl_final", "xl_spread"]}   # = setup_c.FAMILIES
METRIC_KEYS = ("m1_pooled", "m1_per_task", "floor", "n_cells", "n", "n_fail")


def series_name(base: str, layer: int) -> str:
    return f"{base}[{layer}]" if layer >= 0 else base


def _score_block(S, names, variant, ctx, todo, sets, out):
    """Prepare and score a stack of series at every beta for the seeds in `todo`."""
    idx, kept = ctx.idx, ctx.kept
    for beta in prep.BETAS:
        E = prep.smooth(S, names, beta)
        for s in todo:
            mu, sd, sign, train_m1 = prep.fit_scale_orient(E, idx, kept, sets[s]["train"])
            P = prep.apply_prep(E, mu, sd, sign)
            ev = {name: m1_safe(P, kept, idx, sets[s][name]) for name in EVAL_SETS
                  if len(sets[s][name])}
            for k, (base, L) in enumerate(names):
                out[s].append({
                    "variant": variant,
                    "family": prep.family(base) if variant != "baseline" else "baseline",
                    "constraint": base, "layer": L, "beta": prep.beta_label(beta),
                    "sign": "+" if sign[k] > 0 else "-", "train_m1_task": float(train_m1[k]),
                    "metrics": {name: m1_row(m, k) for name, m in ev.items()},
                    "errors": {name: m["error"] for name, m in ev.items() if m["error"]}})
            del P


def run_seeds(ctx: Ctx, cond: str) -> None:
    """Per-seed work: every series x beta, scored on every evaluation set."""
    todo = [s for s in ctx.seeds if not ctx.wpath(cond, f"s2_seed{s}.pkl").exists()]
    if not todo:
        return
    t0 = time.time()
    idx, kept = ctx.idx, ctx.kept
    sets = {s: ctx.sets(cond, s) for s in todo}
    out = {s: [] for s in todo}
    S, names = ctx.series("primary")
    _score_block(S, names, "primary", ctx, todo, sets, out)
    B, bnames = ctx.series("baseline")
    _score_block(B, bnames, "baseline", ctx, todo, sets, out)
    fs = ctx.series("failurespot")
    fs = np.where(np.isnan(fs), 0.0, fs)
    for s in todo:
        for label, sgn, score in (("failurespot", "+", fs), ("failurespot_neg", "-", -fs)):
            ev = {name: m1_safe(score, kept, idx, sets[s][name]) for name in EVAL_SETS
                  if len(sets[s][name])}
            tk = train_kept(idx, kept, sets[s]["train"])
            out[s].append({"variant": "failurespot", "family": "failurespot",
                           "constraint": label, "layer": -1, "beta": f"{FAILURESPOT_BETA}",
                           "sign": sgn,
                           "train_m1_task": float(m1_safe(score, tk, idx, sets[s]["train"])["per_task"]),
                           "metrics": {name: m1_row(m) for name, m in ev.items()},
                           "errors": {name: m["error"] for name, m in ev.items() if m["error"]}})
    for s in todo:
        tp = [r for r in out[s] if r["constraint"] == "timestep"]
        for r in tp:
            for name, m in r["metrics"].items():
                if np.isfinite(m["m1_per_task"]):
                    assert np.isclose(m["m1_pooled"], m["floor"]), "timestep pooled != floor"
                    assert np.isclose(m["m1_per_task"], 0.5), "timestep per-task != 0.5"
        errs = sorted({e for r in out[s] for e in r["errors"].values()})
        if errs:
            log(f"[s2 {cond} seed {s}] M1 errors (NaN in tables): {errs[:5]}")
        save_pkl(ctx.wpath(cond, f"s2_seed{s}.pkl"), {"cond": cond, "seed": s, "rows": out[s]})
    record_timing(ctx, "stage2_seeds", cond, time.time() - t0, f"seeds {todo}")
    log(f"[s2 {cond}] seeds {todo} scored ({time.time() - t0:.0f} s)")


def summarise(ctx: Ctx, cond: str) -> dict:
    """Condition-level tables and selections from the per-seed work files."""
    seeds = list(ctx.seeds)
    per_seed = {s: load_pkl(ctx.wpath(cond, f"s2_seed{s}.pkl"))["rows"] for s in seeds}
    keyf = lambda r: (r["variant"], r["constraint"], r["layer"], r["beta"])  # noqa: E731
    by_key = {s: {keyf(r): r for r in per_seed[s]} for s in seeds}
    keys = [keyf(r) for r in per_seed[seeds[0]]]

    long_rows, summ_rows, summ = [], [], {}
    for key in keys:
        rs = [by_key[s][key] for s in seeds]
        signs = "".join(r["sign"] for r in rs)
        unstable = len(set(signs)) > 1
        head = {"cond": cond, "variant": key[0], "family": rs[0]["family"],
                "constraint": key[1], "layer": key[2], "beta": key[3]}
        for s, r in zip(seeds, rs):
            for name, m in r["metrics"].items():
                long_rows.append({**head, "seed": s, "test_set": name, "sign": r["sign"],
                                  "signs_all_seeds": signs, "unstable": unstable,
                                  "train_m1_task": r["train_m1_task"], **m})
        d = {}
        for name in rs[0]["metrics"]:
            row = {**head, "test_set": name, "signs_all_seeds": signs, "unstable": unstable,
                   "n_seeds": len(seeds)}
            for met in ("m1_pooled", "m1_per_task"):
                mu, sd = mean_std([r["metrics"][name][met] for r in rs])
                row[f"{met}_mean"], row[f"{met}_std"] = mu, sd
                d[(name, met)] = mu
            row["floor_mean"] = mean_std([r["metrics"][name]["floor"] for r in rs])[0]
            row["n_cells"] = " ".join(str(r["metrics"][name]["n_cells"]) for r in rs)
            summ_rows.append(row)
        summ[key] = {"metrics": d, "unstable": unstable, "signs": signs}

    tdir = ctx.tables_dir / cond
    write_csv(tdir / "constraint_m1.csv", long_rows)
    write_csv(tdir / "constraint_m1_summary.csv", summ_rows)

    # --- beta per family: setup_c.choose_betas on the rounded eval_seen means --------------
    prim = [k for k in keys if k[0] == "primary"]
    grid, chosen = [], {}
    for fam, bases in C_FAMILIES.items():
        best_by_beta = {}
        for beta in map(prep.beta_label, prep.BETAS):
            cand = [k for k in prim if k[1] in bases and k[3] == beta]
            score = {k: round(summ[k]["metrics"][("eval_seen", "m1_per_task")], 4) for k in cand}
            best = max(cand, key=lambda k: score[k])
            best_by_beta[beta] = (best, score[best])
            grid.append({"cond": cond, "family": fam, "beta": beta,
                         "best_series": series_name(best[1], best[2]),
                         "best_per_task_seen": score[best],
                         "best_pooled_seen": summ[best]["metrics"][("eval_seen", "m1_pooled")],
                         "median_per_task_seen": float(np.median(list(score.values()))),
                         "n_unstable": sum(summ[k]["unstable"] for k in cand),
                         "n_series": len(cand)})
        chosen[fam] = max(best_by_beta, key=lambda b: best_by_beta[b][1])
    for g in grid:
        g["chosen"] = g["beta"] == chosen[g["family"]]
    write_csv(tdir / "c_betas.csv", grid)

    # --- comparison emb_temp layer: best eval_seen pooled M1 at the chosen beta ------------
    emb_keys = [k for k in prim if k[1] == "emb_temp" and k[3] == chosen["emb_temp"]]
    emb_best = max(emb_keys, key=lambda k: summ[k]["metrics"][("eval_seen", "m1_pooled")])
    unstable = sorted((k[1], k[2]) for k in prim
                      if summ[k]["unstable"] and k[3] == chosen[_fam_of(k[1])]
                      and _fam_of(k[1]) is not None)
    sel = {"betas": chosen,
           "beta_scores": {f: best_by_beta_score(grid, f, chosen[f]) for f in C_FAMILIES},
           "emb_layer": int(emb_best[2]), "emb_beta": chosen["emb_temp"],
           "emb_score": summ[emb_best]["metrics"][("eval_seen", "m1_pooled")],
           "unstable": unstable, "seeds": seeds}
    log(f"[s2 {cond}] betas {chosen}; emb layer emb_temp[{sel['emb_layer']}] "
        f"(eval_seen pooled {sel['emb_score']:.4f}); {len(unstable)} unstable C series")
    return sel


def _fam_of(base: str):
    return next((f for f, b in C_FAMILIES.items() if base in b), None)


def best_by_beta_score(grid, fam, beta) -> float:
    return next(g["best_per_task_seen"] for g in grid if g["family"] == fam and g["beta"] == beta)


def run(ctx: Ctx, cond: str) -> dict:
    run_seeds(ctx, cond)
    return summarise(ctx, cond)
