"""
probes_b.py

B's grid probes at H = 0, 10, 25 on the training set (original by default; THESIS_POOL=D2
for the full set), with per-row outputs for every rollout the evaluation uses
(results_collection_handoff.md §4.1, §4.3, §4.4, §6).

Everything is analysis/abc/setup_b's own code, imported: prepared targets at this pool's
beta (stage5_b.prepared_targets), look-ahead, two-means binarisation, balanced class
weights, the joint GPU fit of one layer's targets x H (fit_grid_block: 12 outputs per
layer, target-major then H, as setup_b fitted them), predict_block, target_auroc.
Unstable targets (sign differs between seeds at the chosen beta) are skipped, read from
the pool's Stage 2 table (stage5_b.pipeline_inputs on results/abc2).

Per layer and seed:
  pass 1   (only if the grid's c is not given) refit on all train rows at every grid c,
           warm-started; own-target AUROC on eval-seen. Grid c = best mean over seeds of
           the mean over probes and H of eval-seen own-target AUROC (setup_b's rule).
           Original training set: c = 1e-4, read from results/abc/b_grid_c.csv (the
           Stage 5 selection), so pass 1 is skipped.
  oof      at the grid c, 5-fold GroupKFold by rollout within train; thresholds and
           standardisation refitted per fold (setup_b pass 2).
  refit    at the grid c on all train rows; scores every other rollout used.
  auroc    each probe's row-level AUROC against its own binarised target (refit
           thresholds) on eval-seen, unseen original, unseen incl. LIBERO-PRO.

Work files (resumable, one per layer): <WORK_DIR>/b_probes/L<ll>.npz:
    p (3 seeds, k, n_used, 520) fp16 [train rows out-of-fold, others refit], pos
    (n_used,), kinds (per-layer targets), hs, thr (3, k) refit thresholds,
    thr_oof absent, auroc JSON {seed: {set: [k]}}, c, pass1 JSON (if run).
    k = len(kinds) * 3, ordered (kind, H) target-major.

Usage: source env.sh; python analysis/thesis_results/probes_b.py [--layers ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import C_GRID, HS, REPO, SEEDS, T, log  # noqa: E402

N_FOLDS = 5
LAYERS = list(range(32))


def work_path(name: str) -> Path:
    d = common.WORK_DIR / "b_probes"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def grid_inputs() -> dict:
    """betas, unstable targets and per-layer target kinds for this pool."""
    import stage5_b
    inp = stage5_b.pipeline_inputs(REPO / "results/abc2", common.POOL)
    inp["kinds"] = stage5_b.layer_targets(inp["unstable"])
    return inp


def given_c():
    """Grid c already chosen for the original training set (Stage 5); None otherwise."""
    if not common.MAIN:
        return None
    rows = list(csv.DictReader((REPO / "results/abc/b_grid_c.csv").open()))
    return float(next(r["c"] for r in rows if r["chosen"] == "True"))


def used_positions(s: int) -> np.ndarray:
    """Rollouts seed s fits on or evaluates (its unseen tasks differ from other seeds')."""
    st = common.sets(s)
    return np.unique(np.concatenate([st[k] for k in ("train", "eval_seen", "T0", "T1", "T2", "stress")]))


def ys_for(targets: dict, kinds, L: int) -> list:
    import setup_b as abc_b
    return [abc_b.lookahead(targets[abc_b.target_name(kd, L)], H) for kd in kinds for H in HS]


def pass1_layer(L: int, X, kinds) -> dict:
    """Own-target eval-seen AUROC at every c, per seed: {seed: {c: [k]}}."""
    import setup_b as abc_b
    ctx = common.ctx()
    kept = ctx.kept
    out = {}
    for s in SEEDS:
        st = common.sets(s)
        tr = st["train"]
        rr = np.nonzero(kept[tr])
        ys = ys_for(common.b_targets(s), kinds, L)
        fits, thr, mu, sd = abc_b.fit_grid_block(X, (tr[rr[0]], rr[1]), ys, list(C_GRID))
        es = st["eval_seen"]
        out[s] = {}
        for c in C_GRID:
            full = np.zeros((len(ys), ctx.idx.n, T), np.float32)
            full[:, es] = abc_b.predict_block(X, es, *fits[c], mu, sd)
            out[s][c] = [abc_b.target_auroc(full[j], ys[j], thr[j], kept, es) for j in range(len(ys))]
    return out


def choose_c(inp) -> tuple[float, dict]:
    path = work_path("grid_c.json")
    if path.exists():
        d = json.loads(path.read_text())
        return float(d["c"]), d
    W = {L: json.loads(np.load(work_path(f"P1_L{L:02d}.npz"))["auroc"].item()) for L in LAYERS}
    score = {}
    for c in C_GRID:
        per_seed = [np.nanmean([a for L in LAYERS for a in W[L][str(s)][repr(c)]]) for s in SEEDS]
        score[c] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)))
    c_best = max(C_GRID, key=lambda c: score[c][0])
    d = {"c": c_best, "score": {repr(c): v for c, v in score.items()},
         "rule": "best mean over seeds of the mean over probes and H of eval-seen own-target AUROC"}
    common.write_json(path, d)
    return c_best, d


def run_layer(L: int, X, c: float, inp) -> None:
    import setup_b as abc_b
    from sklearn.model_selection import GroupKFold
    out = work_path(f"L{L:02d}.npz")
    kinds = inp["kinds"][L]
    if out.exists():
        z = np.load(out)
        if float(z["c"]) == c and list(z["kinds"]) == kinds:
            return
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    pos = np.unique(np.concatenate([used_positions(s) for s in SEEDS]))   # union over seeds
    k = len(kinds) * len(HS)
    P = np.zeros((len(SEEDS), k, len(pos), T), np.float16)
    THR = np.zeros((len(SEEDS), k))
    auroc = {}
    for j, s in enumerate(SEEDS):
        st = common.sets(s)
        tr = st["train"]
        ys = ys_for(common.b_targets(s), kinds, L)
        full = np.zeros((k, idx.n, T), np.float32)
        for f_tr, f_te in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr):
            ftr = tr[f_tr]
            rr = np.nonzero(kept[ftr])
            fits, thr, mu, sd = abc_b.fit_grid_block(X, (ftr[rr[0]], rr[1]), ys, [c])
            full[:, tr[f_te]] = abc_b.predict_block(X, tr[f_te], *fits[c], mu, sd)
        rr = np.nonzero(kept[tr])
        fits, thr, mu, sd = abc_b.fit_grid_block(X, (tr[rr[0]], rr[1]), ys, [c])
        rest = np.setdiff1d(pos, tr)
        full[:, rest] = abc_b.predict_block(X, rest, *fits[c], mu, sd)
        THR[j] = thr
        auroc[s] = {key: [abc_b.target_auroc(full[i], ys[i], thr[i], kept, st[key]) for i in range(k)]
                    for key in common.AUROC_SETS}
        P[j] = full[:, pos].astype(np.float16)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, p=P, pos=pos, kinds=np.array(kinds), hs=np.array(HS), thr=THR, c=c,
             auroc=json.dumps(auroc), seeds=np.array(SEEDS),
             betas=json.dumps(inp["betas"], sort_keys=True))
    tmp.replace(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    a = ap.parse_args()
    import torch
    layers = a.layers if a.layers else LAYERS
    inp = grid_inputs()
    n_probes = sum(len(v) for v in inp["kinds"].values())
    log(f"pool {common.POOL}; betas {inp['betas']}; skipped unstable {inp['unstable']}; "
        f"{n_probes} probes per H")
    common.b_targets_all()
    c = given_c()
    t0 = time.time()
    if c is None:
        for L in layers:
            p1 = work_path(f"P1_L{L:02d}.npz")
            if p1.exists():
                continue
            X = common.load_layer(L)
            res = pass1_layer(L, X, inp["kinds"][L])
            np.savez(p1, auroc=json.dumps({str(s): {repr(cc): v for cc, v in d.items()}
                                           for s, d in res.items()}))
            del X
            torch.cuda.empty_cache()
            log(f"[B pass 1 {L + 1:2d}/32] layer {L} ({time.time() - t0:.0f} s)")
        c, d = choose_c(inp)
        log(f"grid c = {c:g} ({d['score']})")
    else:
        common.write_json(work_path("grid_c.json"), {"c": c, "rule": "given: results/abc/b_grid_c.csv (Stage 5)"})
        log(f"grid c = {c:g} (Stage 5's selection, results/abc/b_grid_c.csv)")
    for L in layers:
        if work_path(f"L{L:02d}.npz").exists():
            z = np.load(work_path(f"L{L:02d}.npz"))
            if float(z["c"]) == c and list(z["kinds"]) == inp["kinds"][L]:
                continue
        X = common.load_layer(L)
        run_layer(L, X, c, inp)
        del X
        torch.cuda.empty_cache()
        log(f"[B {L + 1:2d}/32] layer {L} done ({time.time() - t0:.0f} s)")
    common.write_json(work_path("meta.json"), {"c": c, "betas": inp["betas"], "unstable": inp["unstable"],
                                               "kinds": {str(L): v for L, v in inp["kinds"].items()},
                                               "n_probes_per_H": n_probes})
    log("B probes done")


if __name__ == "__main__":
    main()
