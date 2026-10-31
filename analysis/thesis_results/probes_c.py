"""
probes_c.py

C's 33 probes on the training set (original by default; THESIS_POOL=D2 for the full set),
refitted so their per-row outputs cover every rollout, plus out-of-fold outputs for C's
trained combiner and the leave-one-family-out probe sets for C's unweighted mean
(results_collection_handoff.md §4.1, §4.6).

Features are Stage 3's (stage3_c.smoothed_families at this pool's beta per family, then
setup_c.seed_features: scaled on the train successes and oriented on train, per seed).
Every probe is setup_c.fit_prob (StandardScaler -> LogisticRegression(C=c,
max_iter=2000), row weights from data.train_rows_and_weights) at this pool's c for C's
probes (results/abc2/selections.csv, c_C_probes).

  refit   the 33 probes (32 per-layer [emb_temp[l], xl_adj[min(l,30)],
          xl_final[min(l,30)]] + the action probe) fitted on all train rows.
  oof     5-fold GroupKFold by rollout within train (B's procedure); each fold's probe is
          refitted on the fold's train rollouts (rows, weights and the probe's own
          standardisation recomputed; the constraint scaling and orientation stay the
          ones fitted on the whole train split, as B's targets do in its folds).
  lofo    leave-one-family-out probe sets for the mean: without emb_temp, the per-layer
          probes on [xl_adj, xl_final]; without the cross-layer family, the per-layer
          probes on [emb_temp] alone; without the action family, the 32 per-layer probes
          (no refit). Refit on all train only.

Work file (resumable, one per seed): <WORK_DIR>/c_probes/seed<s>.npz with names (33),
p_refit (33, N, 520) float32, p_oof (33, N, 520) fp16 (zeros off train rollouts),
lofo_names, p_lofo (n, N, 520) float32, inputs JSON, c.

Usage: source env.sh; python analysis/thesis_results/probes_c.py [--n-jobs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import SEEDS, T, log  # noqa: E402

N_FOLDS = 5


def probe_sets() -> tuple[dict, dict]:
    import stage3_c
    probes = dict(stage3_c.PROBES)
    lofo = {}
    for L in range(32):
        lofo[f"no_emb_temp:layer{L:02d}"] = [f"xl_adj[{min(L, 30)}]", f"xl_final[{min(L, 30)}]"]
        lofo[f"no_cross_layer:layer{L:02d}"] = [f"emb_temp[{L}]"]
    return probes, lofo


def work_path(s: int) -> Path:
    d = common.WORK_DIR / "c_probes"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"seed{s}.npz"


def job(name, inputs, feats, folds, train_full, fold_trains, c, n):
    """One probe: refit p (float32) and, if folds given, out-of-fold p on train rollouts."""
    _, p, ok = common.abc_c.fit_prob(inputs, feats, train_full, c)
    oof = None
    if folds is not None:
        oof = np.zeros((n, T), np.float16)
        for (f_tr, f_te), ftrain in zip(folds, fold_trains):
            _, q, _ = common.abc_c.fit_prob(inputs, feats, ftrain, c)
            oof[f_te] = q[f_te].astype(np.float16)
    return name, p, oof, ok


def run_seed(s: int, n_jobs: int) -> None:
    import stage3_c
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold
    out = work_path(s)
    sel = common.selections()
    c = sel["c_C_probes"]
    if out.exists():
        z = np.load(out)
        if float(z["c"]) == c:
            return
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    t0 = time.time()
    fams = stage3_c.smoothed_families(ctx, sel["betas"])
    st = common.sets(s)
    tr = st["train"]
    feats, signs = common.abc_c.seed_features(fams, idx, kept, tr)
    del fams
    train_full = common.train_rows_and_weights(idx, kept, tr)
    folds = [(tr[a], tr[b]) for a, b in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr)]
    fold_trains = [common.train_rows_and_weights(idx, kept, a) for a, _ in folds]
    probes, lofo = probe_sets()
    jobs = [(nm, inp, True) for nm, inp in probes.items()] + [(nm, inp, False) for nm, inp in lofo.items()]
    res = Parallel(n_jobs=n_jobs)(
        delayed(job)(nm, inp, {k: feats[k] for k in inp}, folds if do_oof else None,
                     train_full, fold_trains, c, idx.n)
        for nm, inp, do_oof in jobs)
    got = {r[0]: r for r in res}
    names = list(probes)
    lofo_names = list(lofo)
    unconv = [r[0] for r in res if not r[3]]
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, names=np.array(names), p_refit=np.stack([got[n][1] for n in names]),
             p_oof=np.stack([got[n][2] for n in names]), lofo_names=np.array(lofo_names),
             p_lofo=np.stack([got[n][1] for n in lofo_names]),
             inputs=json.dumps({**probes, **lofo}), c=c, unconverged=json.dumps(unconv),
             signs=json.dumps(signs))
    tmp.replace(out)
    log(f"[C seed {s}] {len(names)} probes + {len(lofo_names)} leave-one-family-out probes, "
        f"c = {c:g}; unconverged {unconv} ({time.time() - t0:.0f} s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    for s in SEEDS:
        run_seed(s, a.n_jobs)
    log("C probes done")


if __name__ == "__main__":
    main()
