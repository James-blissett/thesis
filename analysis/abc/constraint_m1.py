"""
constraint_m1.py

Stage 2: every constraint scored against episode outcome on its own (spec section 2,
"first check"), per seed and smoothing level, on eval_seen and unseen.

Metrics, from data.m1:
  pooled     SAFE's M1 (one ROC-AUC across all tasks in the set)
  per_task   M1 within each task, averaged over tasks; a timestep score gets 0.5
  floor      pooled M1 of a timestep-only score on the same set (the length floor)

Rows of the output tables:
  * every prepared constraint series (primary, with-sink) at beta in {none, 0.83, 0.91,
    0.96}, oriented by its per-task train M1 (prep.py); "unstable" = sign differs
    across seeds.
  * baselines through the same preparation: per-step token probability and entropy from
    logits.pt (mean and max over the 7 action tokens, as SAFE's unc_utils defines them),
    and the timestep-only score itself (pooled must equal the floor, per_task 0.5).
  * FailureSpot's signal s = c + EMA(c) + r + EMA(r), c = ||a_t - a_{t-1}||_2,
    r = ||a_t||_2 over all 7 normalised dims, beta = 0.8, with its fixed sign: s and -s.
  * side table: the *_nosink series.

Per-layer families are summarised by the layer with the best eval_seen pooled M1, and
that layer's unseen numbers are reported: selection never looks at unseen.

Std is over the 3 task-split seeds (ddof = 1).

Outputs:
    results/abc/constraint_m1.csv          primary + baselines + FailureSpot
    results/abc/constraint_m1_nosink.csv   *_nosink series

Usage (from the repo root; under a minute):
    source env.sh
    python analysis/abc/constraint_m1.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))   # analysis/, for frozen modules
from compute_constraints import normalised_actions          # noqa: E402

from data import (CORPUS_ROOT, MIN_TRAIN_SUCCESSES, RESULTS_DIR, SEEDS, T,  # noqa: E402
                  kept_mask, load_index, load_split, m1, train_cut, train_kept,
                  train_rows_and_weights, weight_stats)
from prep import (BETAS, NOSINK, PRIMARY, apply_prep, beta_label, ema,  # noqa: E402
                  family, fit_scale_orient, load_series, smooth)

OUT = RESULTS_DIR / "constraint_m1.csv"
OUT_NOSINK = RESULTS_DIR / "constraint_m1_nosink.csv"
FAILURESPOT_BETA = 0.8
SETS = (("seen", "eval_seen"), ("unseen", "unseen"))
METRICS = ("pooled", "per_task")


def baseline_series(idx) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Token probability / entropy per step from logits.pt, plus t. (n, 300, 520)."""
    names = [("token_prob_mean", -1), ("token_prob_max", -1),
             ("token_entropy_mean", -1), ("token_entropy_max", -1), ("timestep", -1)]
    S = np.empty((len(names), idx.n, T))
    for i, rid in enumerate(idx.rids):
        lg = torch.load(str(CORPUS_ROOT / rid / "logits.pt"), map_location="cpu").float()
        p = torch.softmax(lg, -1)                                      # (520, 7, 256)
        sel = p.max(-1).values                                         # (520, 7)
        ent = -(p * torch.log_softmax(lg, -1)).sum(-1)                 # (520, 7)
        S[0, i] = sel.mean(-1).numpy()
        S[1, i] = sel.max(-1).values.numpy()
        S[2, i] = ent.mean(-1).numpy()
        S[3, i] = ent.max(-1).values.numpy()
        S[4, i] = np.arange(T)
    return S, names


def failurespot_series(idx) -> np.ndarray:
    """(300, 520) FailureSpot score; NaN at t = 0 only."""
    out = np.empty((idx.n, T))
    for i, rid in enumerate(idx.rids):
        tok = np.load(CORPUS_ROOT / rid / "actions.npz")["action_token_ids"]
        a = normalised_actions(tok).astype(np.float64)                 # (520, 7)
        c = np.full(T, np.nan)
        c[1:] = np.linalg.norm(a[1:] - a[:-1], axis=1)
        r = np.linalg.norm(a, axis=1)
        out[i] = c + ema(c, FAILURESPOT_BETA) + r + ema(r, FAILURESPOT_BETA)
    return out


def score_series(S, names, idx, kept, splits, variant):
    """All metrics for every series at every beta and seed -> list of CSV rows."""
    rows = []
    for beta in BETAS:
        E = smooth(S, names, beta)
        res = {}   # seed -> (sign, {set: m1 dict}, train per-task M1 before orienting)
        for seed in SEEDS:
            sp = splits[seed]
            mu, sd, sign, train_m1 = fit_scale_orient(E, idx, kept, sp["pos"]["train"])
            P = apply_prep(E, mu, sd, sign)
            res[seed] = (sign, {lab: m1(P, kept, idx, sp["pos"][key]) for lab, key in SETS},
                         train_m1)
        for k, (base, L) in enumerate(names):
            vals = {(lab, met): np.array([res[s][1][lab][met][k] for s in SEEDS])
                    for lab, _ in SETS for met in METRICS}
            floors = {lab: np.array([res[s][1][lab]["floor"] for s in SEEDS]) for lab, _ in SETS}
            rows.append(row(variant, family(base) if variant != "baseline" else "baseline",
                            base, L, beta_label(beta),
                            "".join("+" if res[s][0][k] > 0 else "-" for s in SEEDS),
                            vals, floors, np.array([res[s][2][k] for s in SEEDS])))
    return rows


def row(variant, fam, name, layer, beta, sign, vals, floors, train_m1) -> dict:
    """train_m1: (3,) unoriented per-task train M1 per seed, which set the sign."""
    d = {"variant": variant, "family": fam, "constraint": name, "layer": layer,
         "beta": beta, "sign": sign, "unstable": len(set(sign)) > 1}
    for s, v in zip(SEEDS, train_m1):
        d[f"train_m1_task_s{s}"] = float(v)
    for lab, _ in SETS:
        for met in METRICS:
            v = vals[(lab, met)]
            d[f"{met}_{lab}_mean"] = float(v.mean())
            d[f"{met}_{lab}_std"] = float(v.std(ddof=1))
        d[f"floor_{lab}"] = float(floors[lab].mean())
    for lab, _ in SETS:
        for met in METRICS:
            for s, v in zip(SEEDS, vals[(lab, met)]):
                d[f"{met}_{lab}_s{s}"] = float(v)
    return d


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def fmt(r) -> str:
    lay = f"[{r['layer']:2d}]" if r["layer"] >= 0 else "    "
    flag = "UNSTABLE" if r["unstable"] else "        "
    return (f"{r['constraint']:20s}{lay} {r['beta']:4s} {r['sign']} {flag} "
            f"pooled {r['pooled_seen_mean']:.3f} | {r['pooled_unseen_mean']:.3f}"
            f"±{r['pooled_unseen_std']:.3f}   "
            f"per-task {r['per_task_seen_mean']:.3f} | {r['per_task_unseen_mean']:.3f}"
            f"±{r['per_task_unseen_std']:.3f}")


def summarise(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===  (seen | unseen; floor seen {rows[0]['floor_seen']:.3f}, "
          f"unseen {rows[0]['floor_unseen']:.3f}; per-task floor 0.5)")
    bases = list(dict.fromkeys(r["constraint"] for r in rows))
    for base in bases:
        for beta in dict.fromkeys(r["beta"] for r in rows if r["constraint"] == base):
            lr = [r for r in rows if r["constraint"] == base and r["beta"] == beta]
            if lr[0]["layer"] < 0:
                print("   " + fmt(lr[0]))
                continue
            best = max(lr, key=lambda r: r["pooled_seen_mean"])
            seen = np.array([r["pooled_seen_mean"] for r in lr])
            unstable = [r["layer"] for r in lr if r["unstable"]]
            print("   " + fmt(best) + f"   [best of {len(lr)} on seen; layer median seen "
                  f"{np.median(seen):.3f}; unstable layers {unstable or 'none'}]")


def weight_diagnostics(idx, kept, splits) -> None:
    """The >= 5-successes training cut and the outcome-model row weights, per seed,
    before (>= 1 success, the original cut) and after."""
    print("\n=== training cut and row weights (outcome-trained models) ===")
    for seed in SEEDS:
        tr = splits[seed]["pos"]["train"]
        for label, k in (("before, >= 1 success", 1), (f"after,  >= {MIN_TRAIN_SUCCESSES} successes",
                                                        MIN_TRAIN_SUCCESSES)):
            r, t, y, w = train_rows_and_weights(idx, kept, tr, min_success=k)
            s = weight_stats(w)
            print(f"   seed {seed} {label}: t <= {train_cut(idx, kept, tr, k)}, rows {s['n_rows']} "
                  f"(succ {int((y == 0).sum())}, fail {int((y == 1).sum())}); "
                  f"max/mean {s['max_over_mean']:.1f}, top 1% share {s['top1pct_share']:.3f}, "
                  f"ESS {s['ess']:.0f} ({s['ess'] / s['n_rows']:.2f} of rows)")


def main() -> None:
    t0 = time.time()
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}

    S, names = load_series(idx, PRIMARY)
    rows = score_series(S, names, idx, kept, splits, "primary")
    del S
    print(f"[*] {len(names)} primary series scored ({time.time() - t0:.0f} s)")

    B, bnames = baseline_series(idx)
    brows = score_series(B, bnames, idx, kept, splits, "baseline")
    for r in brows:
        if r["constraint"] == "timestep":
            assert r["sign"] == "+++"
            assert np.isclose(r["pooled_seen_mean"], r["floor_seen"])
            assert np.isclose(r["pooled_unseen_mean"], r["floor_unseen"])
            assert np.isclose(r["per_task_seen_mean"], 0.5)
            assert np.isclose(r["per_task_unseen_mean"], 0.5)
    rows += brows

    fs = failurespot_series(idx)
    fs = np.where(np.isnan(fs), 0.0, fs)               # t = 0 only, never kept
    for label, sgn, score in (("failurespot", "+++", fs), ("failurespot_neg", "---", -fs)):
        res = {s: {lab: m1(score, kept, idx, splits[s]["pos"][key]) for lab, key in SETS}
               for s in SEEDS}
        vals = {(lab, met): np.array([res[s][lab][met] for s in SEEDS])
                for lab, _ in SETS for met in METRICS}
        floors = {lab: np.array([res[s][lab]["floor"] for s in SEEDS]) for lab, _ in SETS}
        train_m1 = np.array([m1(score, train_kept(idx, kept, splits[s]["pos"]["train"]), idx,
                                splits[s]["pos"]["train"])["per_task"] for s in SEEDS])
        rows.append(row("failurespot", "failurespot", label, -1, f"{FAILURESPOT_BETA}",
                        sgn, vals, floors, train_m1))
    write_csv(OUT, rows)

    N, nnames = load_series(idx, NOSINK)
    nrows = score_series(N, nnames, idx, kept, splits, "nosink")
    write_csv(OUT_NOSINK, nrows)

    summarise([r for r in rows if r["variant"] == "primary"], "primary constraints")
    summarise([r for r in rows if r["variant"] in ("baseline", "failurespot")], "baselines")
    summarise(nrows, "nosink side table")
    weight_diagnostics(idx, kept, splits)
    n_tasks = {lab: [m1(np.zeros((idx.n, T)), kept, idx, splits[s]["pos"][key])["n_tasks"]
                     for s in SEEDS] for lab, key in SETS}
    print(f"\n[*] tasks entering per-task M1 per seed: {n_tasks}")
    print(f"[*] {len(rows)} rows -> {OUT}; {len(nrows)} rows -> {OUT_NOSINK} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
