"""
stage6_cp.py

Stage 6 of the port (analysis/abc/cp_eval.py, generalised; CUT VARIANT ONLY, Step 9.4):
conformal alarms for A, C and the condition's comparison emb_temp layer, plus the final
M1 table and the stress test.

Detectors per (cond, seed):
  A    A's final probe (stage4_a.final_scores): running sum of fp16-rounded P(failure)
  C    C's final detector at its c (stage3_c.final_scores): running sum of P(failure)
  emb  emb_temp[<layer>] at the emb_temp beta, both chosen on this condition's eval_seen
       (stage2_constraints.summarise); M1 score = its prepared z-score (NaN where
       missing), CP score = running maximum over kept rows (as cp_eval).
Band: cp_eval.fit_band (imported): calibration = the condition's successful eval_seen
rollouts, halves D1 / D2 drawn with default_rng(CAL_SPLIT_SEED + seed); h per alpha by
cp_eval.band_h; alarm at the first kept row t < cut with score(t) > mu(t) + h xi(t).
Evaluation (cp_eval.evaluate_cp with cut = data.m1_cut on the evaluated set, i.e. the
shortest kept length in the rollout's CELL within that set): eval_seen (its successes
include the calibration rollouts), T0, T1, T2; alphas = cp_eval.ALPHAS (15) and 0.1.
Stress (alpha = 0.1): a stress rollout's cut is its base task's unperturbed-cell cut from
the same seed: min kept length over the condition's eval_seen unperturbed rollouts of
that base task (seen base tasks) or over T1's rollouts of that base task (unseen base
tasks); share of stress rollouts with an alarm at a kept row t < cut.

Outputs: <out>/scores/<cond>_seed<s>.npz (format in pipeline.py's docstring), work file
<work>/<cond>/cp_seed{s}.pkl (table rows), and per-condition tables
<out>/tables/<cond>/{final_m1,cp_curves,stress}.csv.
"""

from __future__ import annotations

import json
import time

import numpy as np

import cp_eval as abc_cp
import prep
import stage3_c
import stage4_a
from data2 import T, m1_cut
from pipe2_common import (DETECTORS, EVAL_SETS, STRESS_SETS, Ctx, log, load_pkl, m1_row,
                          m1_safe, record_timing, save_pkl, write_csv)

ALPHAS = list(abc_cp.ALPHAS) + [abc_cp.ALPHA_TABLE]       # 15 grid levels, then 0.1
ALPHA_TABLE = abc_cp.ALPHA_TABLE
CAL_SPLIT_SEED = abc_cp.CAL_SPLIT_SEED


def emb_scores(ctx: Ctx, train_pos, layer: int, beta_label: str):
    """(x, run): prepared z-score of emb_temp[layer] and its running max over kept rows
    (NaN before the first kept row), as cp_eval.load_scores."""
    S, names = ctx.series("primary")
    k = names.index(("emb_temp", layer))
    beta = 0.0 if beta_label == "none" else float(beta_label)
    E = prep.smooth(S[k:k + 1], names[k:k + 1], beta)
    mu, sd, sign, _ = prep.fit_scale_orient(E, ctx.idx, ctx.kept, train_pos)
    x = prep.apply_prep(E, mu, sd, sign)[0]
    xk = np.where(ctx.kept, x, -np.inf)
    run = np.maximum.accumulate(np.nan_to_num(xk, nan=-np.inf), axis=1)
    return x, np.where(np.isfinite(run), run, np.nan), float(sign[0])


def stress_cuts(ctx: Ctx, seed: int, sets: dict, pos: np.ndarray) -> np.ndarray:
    idx = ctx.idx
    es = sets["eval_seen"]
    es_unp = es[idx.family[es] == "unperturbed"]
    t1 = sets["T1"]
    unseen_tasks = set(int(t) for t in ctx.split(seed)["unseen_tasks"])
    cut = np.empty(len(pos), dtype=np.int64)
    for j, r in enumerate(pos):
        b = int(idx.base_task[r])
        ref = t1[idx.base_task[t1] == b] if b in unseen_tasks else es_unp[idx.base_task[es_unp] == b]
        cut[j] = idx.kept_len[ref].min() if len(ref) else T
    return cut


def alarms(score, kept, pos, mu, xi, h, cut) -> np.ndarray:
    b = mu + h * xi
    rows = kept[pos] & (np.arange(T)[None, :] < cut[:, None])
    with np.errstate(invalid="ignore"):
        return (rows & (score[pos] > b[None, :])).any(1)


def run_seed(ctx: Ctx, cond: str, seed: int, sel: dict) -> None:
    idx, kept = ctx.idx, ctx.kept
    sets = ctx.sets(cond, seed)
    t0 = time.time()
    pA, sA = stage4_a.final_scores(ctx, cond, seed)
    pC, sC = stage3_c.final_scores(ctx, cond, seed, sel["c_final"])
    x, run, emb_sign = emb_scores(ctx, sets["train"], sel["emb_layer"], sel["emb_beta"])
    dets = {"A": {"m1": sA, "cp": sA, "p": pA},
            "C": {"m1": sC, "cp": sC, "p": pC},
            "emb": {"m1": x, "cp": run}}
    detail = {"A": f"{sel['a_probe'][seed]} c={sel['c_A']:g}",
              "C": f"final c={sel['c_final']:g}",
              "emb": f"emb_temp[{sel['emb_layer']}] beta={sel['emb_beta']}"}

    m1_rows, cp_rows, stress_rows = [], [], []
    evals = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
    cuts = {k: m1_cut(idx, pos) for k, pos in evals.items()}
    cal = sets["cal"]
    npz = {"rollout_ids": idx.rids, "cond": cond, "seed": seed,
           "alphas": np.array(ALPHAS), "kept_len": idx.kept_len, "y": idx.y,
           "cell": idx.task, "base_task": idx.base_task, "block": idx.block,
           "cal_rids": idx.rids[cal], "train_rids": idx.rids[sets["train"]],
           "eval_seen_rids": idx.rids[sets["eval_seen"]],
           "emb_sign": emb_sign}
    for d in DETECTORS:
        sc = dets[d]["cp"]
        mu, xi, R, t_end, n1, n2 = abc_cp.fit_band(sc, kept, cal,
                                                  np.random.default_rng(CAL_SPLIT_SEED + seed))
        hs = np.array([abc_cp.band_h(R, a) for a in ALPHAS])
        npz.update({f"{d}_m1": dets[d]["m1"], f"{d}_cp": sc, f"{d}_mu": mu, f"{d}_xi": xi,
                    f"{d}_h": hs, f"{d}_band01": mu + hs[-1] * xi, f"{d}_R": R,
                    f"{d}_t_end": t_end, f"{d}_n_d1": n1, f"{d}_n_d2": n2})
        if "p" in dets[d]:
            npz[f"{d}_p"] = dets[d]["p"]
        head = {"cond": cond, "seed": seed, "detector": d, "detail": detail[d]}
        for ts, pos in evals.items():
            m1_rows.append({**head, "test_set": ts, **m1_row(m1_safe(dets[d]["m1"], kept, idx, pos))})
            for a, h in zip(ALPHAS, hs):
                r = abc_cp.evaluate_cp(sc, kept, idx, pos, mu, xi, h, cuts[ts])
                cp_rows.append({**head, "test_set": ts, "alpha": float(a), "variant": "cut",
                                "on_alpha_grid": bool(a in abc_cp.ALPHAS), "h": h,
                                "bal_acc": r["bal_acc"], "tpr": r["tpr"], "tnr": r["tnr"],
                                "far": 1 - r["tnr"], "det_time": r["det_time"],
                                "n": int(len(pos)), "n_fail": int(idx.y[pos].sum()),
                                "n_cal": int(len(cal)), "n_d1": n1, "n_d2": n2, "t_end": t_end})
        h01 = hs[-1]
        for ss in STRESS_SETS:
            pos = sets[ss]
            if not len(pos):
                continue
            cut = stress_cuts(ctx, seed, sets, pos)
            fired = alarms(sc, kept, pos, mu, xi, h01, cut)
            npz[f"{ss}_cut"] = cut
            for blk in ["all"] + sorted(set(idx.block[pos].tolist())):
                m = np.ones(len(pos), bool) if blk == "all" else idx.block[pos] == blk
                stress_rows.append({**head, "stress_set": ss, "block": blk, "alpha": ALPHA_TABLE,
                                    "n": int(m.sum()), "n_fail": int(idx.y[pos][m].sum()),
                                    "share_alarm_before_cut": float(fired[m].mean()),
                                    "mean_cut": float(cut[m].mean())})
    npz["selections"] = json.dumps(_sel_for_seed(sel, seed))
    out = ctx.scores_dir / f"{cond}_seed{seed}.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    # emb_cp holds -1.8e308 (cp_eval's nan_to_num of -inf) before the first kept row; it is
    # never compared (t = 0 is not kept), and is stored as NaN rather than overflowing.
    npz["emb_cp"] = np.where(npz["emb_cp"] < -1e300, np.nan, npz["emb_cp"])
    np.savez_compressed(tmp, **{k: (v.astype(np.float32) if isinstance(v, np.ndarray)
                                    and v.dtype == np.float64 and v.ndim == 2 else v)
                                for k, v in npz.items()})
    tmp.replace(out)
    save_pkl(ctx.wpath(cond, f"cp_seed{seed}.pkl"),
             {"sel": _sel_for_seed(sel, seed), "m1": m1_rows, "cp": cp_rows, "stress": stress_rows})
    log(f"[s6 {cond} seed {seed}] CP done -> {out} ({time.time() - t0:.0f} s)")


def _sel_for_seed(sel: dict, seed: int) -> dict:
    d = {k: v for k, v in sel.items() if k not in ("a_probe", "a_probe_score", "unstable")}
    d["a_probe"] = sel["a_probe"][seed]
    d["a_probe_score"] = sel["a_probe_score"][seed]
    return d


def run(ctx: Ctx, cond: str, sel: dict) -> None:
    t0 = time.time()
    for s in ctx.seeds:
        w = ctx.wpath(cond, f"cp_seed{s}.pkl")
        out = ctx.scores_dir / f"{cond}_seed{s}.npz"
        if w.exists() and out.exists() and load_pkl(w)["sel"] == _sel_for_seed(sel, s):
            continue
        run_seed(ctx, cond, s, sel)
    W = [load_pkl(ctx.wpath(cond, f"cp_seed{s}.pkl")) for s in ctx.seeds]
    tdir = ctx.tables_dir / cond
    write_csv(tdir / "final_m1.csv", [r for w in W for r in w["m1"]])
    write_csv(tdir / "cp_curves.csv", [r for w in W for r in w["cp"]])
    write_csv(tdir / "stress.csv", [r for w in W for r in w["stress"]])
    record_timing(ctx, "stage6", cond, time.time() - t0)
