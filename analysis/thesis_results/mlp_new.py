"""
mlp_new.py

The two new 2-layer MLPs of results_collection_handoff.md §4.1 / §4.4, with the more-data
run's plain recipe (analysis/abc2/stage10_mlp.py, imported): standardised input -> 256 ->
256 -> 1, ReLU; AdamW lr 1e-3, batch 4096, at most 50 epochs, early stopping with
patience 5 keeping the best epoch; weight decay from stage10_mlp.WD_GRID chosen by the
best mean over seeds of the best-epoch eval-seen score; torch seed = split seed.
"2-layer MLP" everywhere (not the capacity ablation's residual network).

  c   C's MLP on the best C probe's inputs (per seed: the 3 constraints of that layer, or
      the 4 action constraints), trained on outcome with the row weights, early stopping
      on eval-seen per-task AUROC: stage10_mlp.train_one unchanged.
  b   B's MLP at H = 10: per seed, B's single best probe at H = 10 (detectors.py, chosen
      on eval-seen per-task AUROC against outcome); input = that probe's layer's pooled
      hidden state; target = that probe's binarised look-ahead constraint (refit
      threshold of the linear probe, probes_b.py); rows = all kept train rows with a
      target (as the grid); balanced class weights, no outcome labels; early stopping
      and weight decay on eval-seen ROW-LEVEL AUROC AGAINST THE TARGET.

Outputs: <WORK_DIR>/mlp/{C_mlp,B_mlp_h10}.pkl with p (per seed, (N, 520) float32),
detail, source, selection (chosen wd, epochs, scores per wd), and for B the row-level
target AUROC on eval-seen / unseen original / unseen incl. LIBERO-PRO.

Usage: source env.sh; python analysis/thesis_results/mlp_new.py c|b
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import SEEDS, T, load_pkl, log, mean_std, save_pkl  # noqa: E402

OUT = common.WORK_DIR / "mlp"


def run_c() -> None:
    import torch
    import stage3_c
    import stage10_mlp as m
    path = OUT / "C_mlp.pkl"
    if path.exists():
        return
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    sel_c = load_pkl(common.WORK_DIR / "detectors" / "C_best.pkl")["selection"]
    fams = stage3_c.smoothed_families(ctx, common.selections()["betas"])
    res = {}
    for s in SEEDS:
        st = common.sets(s)
        feats, _ = common.abc_c.seed_features(fams, idx, kept, st["train"])
        inputs = sel_c[s]["inputs"]
        X = torch.from_numpy(np.stack([feats[n] for n in inputs], -1)).to(m.DEV)
        train = common.train_rows_and_weights(idx, kept, st["train"])
        inp = m.Inputs(X, train[0], train[1])
        res[s] = {}
        for wd in m.WD_GRID:
            r = m.train_one(inp, train, wd, s, ctx, st["eval_seen"])
            res[s][wd] = {k: r[k] for k in ("best_epoch", "epochs_run", "best_score", "p", "history")}
            log(f"[C mlp seed {s}] {sel_c[s]['probe']} ({len(inputs)} inputs) wd {wd:g}: epoch "
                f"{r['best_epoch']}/{r['epochs_run']}, eval-seen per-task {r['best_score']:.4f}")
    score = {wd: mean_std([res[s][wd]["best_score"] for s in SEEDS]) for wd in m.WD_GRID}
    wd = max(m.WD_GRID, key=lambda w: score[w][0])
    OUT.mkdir(parents=True, exist_ok=True)
    save_pkl(path, {
        "p": {s: res[s][wd]["p"] for s in SEEDS},
        "detail": {s: f"{sel_c[s]['probe']} inputs wd={wd:g} epoch={res[s][wd]['best_epoch']}" for s in SEEDS},
        "source": "new: 2-layer MLP (stage10_mlp recipe) on the best C probe's inputs",
        "selection": {"wd": wd, "wd_scores": {repr(k): v for k, v in score.items()},
                      "per_seed": {s: {"probe": sel_c[s]["probe"], "inputs": sel_c[s]["inputs"],
                                       "best_epoch": res[s][wd]["best_epoch"],
                                       "epochs_run": res[s][wd]["epochs_run"]} for s in SEEDS},
                      "grid": {s: {repr(w): {k: res[s][w][k] for k in ("best_epoch", "epochs_run", "best_score")}
                                   for w in m.WD_GRID} for s in SEEDS}}})
    log(f"[C mlp] wd {wd:g}; saved {path}")


def target_rows(y, kept, pos):
    """Kept rows of rollouts `pos` with a finite target: (r, t)."""
    m = np.zeros_like(kept)
    m[pos] = kept[pos]
    m &= np.isfinite(y)
    return np.nonzero(m)


def train_b(inp, rr, tt, lab, w, wd, seed, y, thr, kept, es) -> dict:
    """stage10_mlp.train_one with the target as label, balanced class weights and early
    stopping on eval-seen row-level AUROC against the target."""
    import torch
    import torch.nn.functional as F
    import setup_b as abc_b
    import stage10_mlp as m
    dev = m.DEV
    n = len(rr)
    rrt = torch.from_numpy(rr).to(dev)
    ttt = torch.from_numpy(tt).to(dev)
    yy = torch.from_numpy(lab.astype(np.float32)).to(dev)
    ww = torch.from_numpy(w.astype(np.float32)).to(dev)
    torch.manual_seed(seed)
    model = m.make_mlp(inp.d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=m.HP["lr"], weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    hist, since = [], 0
    for ep in range(1, m.HP["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        tot = 0.0
        for i in range(0, n, m.HP["batch"]):
            b = perm[i:i + m.HP["batch"]]
            z = model(inp.rows(rrt[b], ttt[b])).squeeze(-1)
            loss = (ww[b] * F.binary_cross_entropy_with_logits(z, yy[b], reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        p = m.predict(model, inp, es)
        sc = abc_b.target_auroc(p, y, thr, kept, es)
        hist.append({"epoch": ep, "train_loss": tot / n, "eval_seen_target_auroc": sc})
        if np.isfinite(sc) and sc > best["score"]:
            best = {"score": sc, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
            since = 0
        else:
            since += 1
            if since >= m.HP["patience"]:
                break
    model.load_state_dict(best["state"])
    p_all = m.predict(model, inp, np.arange(inp.X.shape[0]))
    return {"best_epoch": best["epoch"], "epochs_run": len(hist), "best_score": best["score"],
            "history": hist, "p": p_all}


def run_b() -> None:
    import torch
    import setup_b as abc_b
    import stage10_mlp as m
    path = OUT / "B_mlp_h10.pkl"
    if path.exists():
        return
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    sel_b = load_pkl(common.WORK_DIR / "detectors" / "B_best_h10.pkl")["selection"]
    res, info = {}, {}
    for s in SEEDS:
        st = common.sets(s)
        name = sel_b[s]["probe"]                          # L<ll>:<kind>:H10
        L, kind = int(name[1:3]), name.split(":")[1]
        z = np.load(common.WORK_DIR / "b_probes" / f"L{L:02d}.npz")
        hs = [int(h) for h in z["hs"]]
        col = list(z["kinds"]).index(kind) * len(hs) + hs.index(10)
        thr = float(z["thr"][list(SEEDS).index(s), col])
        lin_auroc = {k: v[col] for k, v in json.loads(str(z["auroc"]))[str(s)].items()}
        y = abc_b.lookahead(common.b_targets(s)[abc_b.target_name(kind, L)], 10)
        rr, tt = target_rows(y, kept, st["train"])
        lab = (y[rr, tt] > thr).astype(np.float32)
        n1 = lab.sum()
        n0 = len(lab) - n1
        w = np.where(lab == 1, len(lab) / (2 * n1), len(lab) / (2 * n0)).astype(np.float32)
        X = common.load_layer(L)
        inp = m.Inputs(X, rr, tt)
        res[s] = {}
        for wd in m.WD_GRID:
            t0 = time.time()
            r = train_b(inp, rr, tt, lab, w, wd, s, y, thr, kept, st["eval_seen"])
            r["auroc"] = {k: abc_b.target_auroc(r["p"], y, thr, kept, st[k]) for k in common.AUROC_SETS}
            res[s][wd] = r
            log(f"[B mlp seed {s}] {name} wd {wd:g}: epoch {r['best_epoch']}/{r['epochs_run']}, eval-seen "
                f"target AUROC {r['best_score']:.4f} (linear {lin_auroc['eval_seen']:.4f}) ({time.time() - t0:.0f} s)")
        info[s] = {"probe": name, "layer": L, "target": abc_b.target_name(kind, L), "threshold": thr,
                   "n_train_rows": int(len(rr)), "train_positive_share": float(n1 / len(lab)),
                   "linear_target_auroc": lin_auroc}
        del inp, X
        torch.cuda.empty_cache()
    score = {wd: mean_std([res[s][wd]["best_score"] for s in SEEDS]) for wd in m.WD_GRID}
    wd = max(m.WD_GRID, key=lambda w_: score[w_][0])
    for s in SEEDS:
        info[s].update({"best_epoch": res[s][wd]["best_epoch"], "epochs_run": res[s][wd]["epochs_run"],
                        "mlp_target_auroc": res[s][wd]["auroc"]})
    OUT.mkdir(parents=True, exist_ok=True)
    save_pkl(path, {
        "p": {s: res[s][wd]["p"] for s in SEEDS},
        "detail": {s: f"{info[s]['probe']} target {info[s]['target']} wd={wd:g} epoch={info[s]['best_epoch']}"
                   for s in SEEDS},
        "source": "new: 2-layer MLP on the best B probe's layer, trained on its constraint target",
        "selection": {"wd": wd, "wd_scores": {repr(k): v for k, v in score.items()}, "per_seed": info,
                      "grid": {s: {repr(w_): {k: res[s][w_][k] for k in ("best_epoch", "epochs_run", "best_score", "auroc")}
                                   for w_ in m.WD_GRID} for s in SEEDS},
                      "history": {s: res[s][wd]["history"] for s in SEEDS}}})
    log(f"[B mlp] wd {wd:g}; saved {path}")


if __name__ == "__main__":
    {"c": run_c, "b": run_b}[sys.argv[1]]()
