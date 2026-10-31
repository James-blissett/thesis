"""
probes_a.py

A's 34 probes on the training set (original by default; THESIS_POOL=D2 for the full
set), refitted so their per-row outputs cover every rollout, plus the out-of-fold outputs
A's trained combiner needs and a shuffled-label control per probe
(results_collection_handoff.md §4.1, §4.3).

Per probe (pooled hidden state at stored index 0..32, 32 = post-norm; action tokens) and
seed, with the existing GPU fit (analysis/abc/setup_a.fit_and_score: sklearn's weighted
L2 logistic-regression objective, standardised on train rows, L-BFGS, warm-started from
the strongest penalty) at this pool's c for A (results/abc2/selections.csv, c_A):
  refit     fit on all train rows (data.train_rows_and_weights) and score all rollouts.
            The warm-start path is every grid c <= c_A, ascending, exactly the start of
            the path setup_a / stage4_a used, so the c_A fit is the same fit.
  oof       5-fold GroupKFold by rollout within train (as B's grid, setup_b): each
            fold's probe (rows and weights recomputed on the fold's train rollouts)
            scores the held-out fold's train rollouts.
  shuffled  N_PERM permutations of the train-rollout labels (rng default_rng(PERM_SEED +
            1000 * seed + probe index)), refit, scored on real labels: M1 on eval-seen,
            unseen original and unseen incl. LIBERO-PRO.

Work files (resumable, one per probe): <WORK_DIR>/a_probes/<probe>.npz with
    p_refit (3, N, 520) fp16, p_oof (3, N, 520) fp16 (zeros off train rollouts),
    shuffled metrics (JSON), c, seeds.
--check: GPU fit vs scikit-learn on one probe (layer 16, seed 0, c_A): unseen row AUROC
and M1 must agree within 0.005 (handoff §4.1's condition for using the GPU fit).

Usage: source env.sh; python analysis/thesis_results/probes_a.py [--check] [--probes ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import C_GRID, SEEDS, T, log, m1_row, m1_safe  # noqa: E402

N_PERM = 5
PERM_SEED = 12345
N_FOLDS = 5
CHECK_TOL = 0.005


def probe_names():
    import setup_a as abc_a
    return list(abc_a.PROBES)


def work_path(probe: str) -> Path:
    d = common.WORK_DIR / "a_probes"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{probe}.npz"


def c_path(c_a: float) -> list:
    return sorted(c for c in C_GRID if c <= c_a)


def run_probe(probe: str, k: int, c_a: float) -> None:
    import setup_a as abc_a
    from sklearn.model_selection import GroupKFold
    import torch
    out = work_path(probe)
    if out.exists():
        z = np.load(out)
        if float(z["c"]) == c_a and list(z["seeds"]) == list(SEEDS):
            return
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    t0 = time.time()
    X = common.load_layer(probe if probe == "action_tokens" else int(probe[5:]))
    t_load = time.time() - t0
    cs = c_path(c_a)
    p_refit = np.zeros((len(SEEDS), idx.n, T), np.float16)
    p_oof = np.zeros((len(SEEDS), idx.n, T), np.float16)
    shuffled = {}
    for j, s in enumerate(SEEDS):
        sets = common.sets(s)
        tr = sets["train"]
        train = common.train_rows_and_weights(idx, kept, tr)
        p_refit[j] = abc_a.fit_and_score(X, train, cs)[c_a].astype(np.float16)
        for f_tr, f_te in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr):
            ftrain = common.train_rows_and_weights(idx, kept, tr[f_tr])
            p = abc_a.fit_and_score(X, ftrain, cs)[c_a]
            p_oof[j, tr[f_te]] = p[tr[f_te]].astype(np.float16)
        rng = np.random.default_rng(PERM_SEED + 1000 * s + k)
        ev = {key: sets[key] for key in common.AUROC_SETS}
        res = []
        for _ in range(N_PERM):
            y_perm = idx.y.copy()
            y_perm[tr] = rng.permutation(idx.y[tr])
            idx_perm = dataclasses.replace(idx, y=y_perm)
            p = abc_a.fit_and_score(X, common.train_rows_and_weights(idx_perm, kept, tr), cs)[c_a]
            sc = common.abc_c.running_sum(p, kept)
            res.append({key: m1_row(m1_safe(sc, kept, idx, pos)) for key, pos in ev.items()})
        shuffled[s] = res
    del X
    torch.cuda.empty_cache()
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, p_refit=p_refit, p_oof=p_oof, c=c_a, seeds=np.array(SEEDS),
             shuffled=json.dumps(shuffled), probe=probe)
    tmp.replace(out)
    log(f"[A {k + 1:2d}/34] {probe}: done ({time.time() - t0:.0f} s, load {t_load:.0f} s)")


def check(c_a: float) -> dict:
    """GPU vs scikit-learn on layer 16, seed 0, at c_A (handoff §4.1)."""
    import setup_a as abc_a
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    s = 0
    sets = common.sets(s)
    train = common.train_rows_and_weights(idx, kept, sets["train"])
    X = common.load_layer(16)
    p_gpu = abc_a.fit_and_score(X, train, c_path(c_a))[c_a]
    r, t, y, w = train
    Xtr = X[torch.from_numpy(r).cuda(), torch.from_numpy(t).cuda()].float().cpu().numpy()
    t0 = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_a, max_iter=2000))
    model.fit(Xtr, y, logisticregression__sample_weight=w)
    t_sk = time.time() - t0
    del Xtr
    need = np.concatenate([sets[k] for k in common.AUROC_SETS])
    need = np.unique(need)
    p_sk = np.zeros((idx.n, T), np.float32)
    for i in range(0, len(need), 30):
        q = need[i:i + 30]
        xb = X[torch.from_numpy(q).cuda()].float().cpu().numpy().reshape(-1, X.shape[-1])
        p_sk[q] = model.predict_proba(xb)[:, 1].reshape(len(q), T)
    res = {"probe": "layer16", "seed": s, "c": c_a, "sklearn_seconds": t_sk}
    worst = 0.0
    for key in common.AUROC_SETS:
        pos = sets[key]
        rr, tt = np.nonzero(kept[pos])
        yy = idx.y[pos][rr]
        a_g = roc_auc_score(yy, p_gpu[pos][rr, tt])
        a_s = roc_auc_score(yy, p_sk[pos][rr, tt])
        mg = m1_safe(common.abc_c.running_sum(p_gpu, kept), kept, idx, pos)
        ms = m1_safe(common.abc_c.running_sum(p_sk, kept), kept, idx, pos)
        res[key] = {"row_auroc_gpu": a_g, "row_auroc_sklearn": a_s,
                    "per_task_gpu": float(mg["per_task"]), "per_task_sklearn": float(ms["per_task"]),
                    "pooled_gpu": float(mg["pooled"]), "pooled_sklearn": float(ms["pooled"])}
        worst = max(worst, abs(a_g - a_s), abs(mg["per_task"] - ms["per_task"]),
                    abs(mg["pooled"] - ms["pooled"]))
    res["max_abs_diff"] = worst
    res["passed"] = bool(worst <= CHECK_TOL)
    common.write_json(common.OUT_DIR / "logs" / "check_a_gpu_vs_sklearn.json", res)
    log(f"[A check] max |GPU - sklearn| {worst:.4f} -> {'PASSED' if res['passed'] else 'FAILED'} "
        f"(sklearn {t_sk:.0f} s)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--probes", nargs="*", default=None)
    a = ap.parse_args()
    c_a = common.selections()["c_A"]
    log(f"pool {common.POOL} ({common.POOL_NAME}); c_A = {c_a:g}; path {c_path(c_a)}")
    if a.check:
        check(c_a)
        return
    names = probe_names()
    for k, p in enumerate(names):
        if a.probes and p not in a.probes:
            continue
        run_probe(p, k, c_a)
    log("A probes done")


if __name__ == "__main__":
    main()
