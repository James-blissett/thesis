"""
setup_a.py

Stage 4: setup A, raw inputs -> episode outcome (handoff v9, sections 2 and 4).

  * 34 probes: the pooled hidden state at each stored index 0..32 (index 32 is
    post-norm and flagged), plus action_token_ids cast to float (7 inputs).
  * every probe: StandardScaler -> L2 logistic regression, trained on outcome with
    data.train_rows_and_weights (>= 5-successes cut, 1/n_class(t) weights rescaled to
    mean 1), exactly as in C. Score = running sum of predicted failure probability.
  * c gridded over C_GRID (as C); one c for all 34 probes, the best mean eval_seen
    M1-per-task (mean over probes and seeds).
  * final detector: per seed, the probe with the best eval_seen M1-per-task at that c.
  * control: shuffled-label probe on each seed's chosen probe (train labels permuted
    across train rollouts, N_PERM times; evaluated on real labels).
  * figure: A's and C's probes' M1-per-task against layer, on one axis.

The fits run on the GPU: the same objective as scikit-learn's lbfgs logistic regression
(sum_i s_i * logloss_i + ||w||^2 / (2c), intercept unpenalised) on standardised inputs,
minimised with full-batch L-BFGS in float32. `--check` compares it with scikit-learn on
one probe: test AUROC (rows) and M1 must agree within CHECK_TOL.

Each layer's (300, 520, 4096) fp16 cache file is loaded once and moved to the GPU; all
seeds and every c are fitted from it, the c values warm-started from strongest to
weakest penalty.

Outputs (results/abc/):
    a_probes.csv          one row per probe at the chosen c
    a_grid.csv            every probe at every c
    a_final.csv           per seed: the chosen probe and its scores; plus the summary
    a_shuffled.csv        the shuffled-label control
    a_vs_c_by_layer.png   A's and C's per-layer M1-per-task, eval_seen and unseen
    scores/a_seed{s}.npz  the chosen probe's per-row probability p and running sum s
and /data/tmp/abc_scores/a_probes_seed{s}.npz, all 34 probes' probabilities at the
chosen c (fp16).

Usage (from the repo root):
    source env.sh
    python analysis/abc/setup_a.py --check 1   # GPU vs scikit-learn on one probe, c = 1
    python analysis/abc/setup_a.py             # the full stage (run in tmux)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import (N_STORED, POST_NORM_INDEX, RESULTS_DIR, SEEDS, T, kept_mask, load_index,
                  load_split, m1, open_action_tokens, open_layer, train_rows_and_weights)
from setup_c import C_GRID, SETS, METRICS, running_sum, summary_row, write_csv

SCORES_DIR = RESULTS_DIR / "scores"
PROBE_SCORES_DIR = Path("/data/tmp/abc_scores")
N_PERM = 5
PERM_SEED = 12345
CHECK_TOL = 0.005
CHECK_PROBE, CHECK_SEED = "layer16", 0
MAX_ITER = 1000
CHUNK = 30                                    # rollouts per scoring chunk
DEV = torch.device("cuda")

PROBES = [f"layer{L:02d}" for L in range(N_STORED)] + ["action_tokens"]


def probe_layer(name: str) -> int | None:
    return int(name[5:]) if name.startswith("layer") else None


def load_probe_input(name: str) -> torch.Tensor:
    """(300, 520, d) on the GPU: fp16 hidden state, or the action tokens as float32."""
    if name == "action_tokens":
        return torch.from_numpy(np.asarray(open_action_tokens()).astype(np.float32)).to(DEV)
    return torch.from_numpy(np.array(open_layer(probe_layer(name)))).to(DEV)


# --- Weighted L2 logistic regression on the GPU ----------------------------------------
def fit_logreg(X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, cs) -> dict:
    """{c: (coef (d,), intercept)} for standardised X (n, d) float32. Same minimiser as
    sklearn LogisticRegression(C=c) with sample_weight=w; the loss is divided by n for
    conditioning, which does not move the minimum. Warm-started from large to small
    penalty (small c first)."""
    n, d = X.shape
    theta = torch.zeros(d + 1, device=DEV)
    out = {}
    for c in sorted(cs):
        p = theta.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=MAX_ITER, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = X @ p[:d] + p[d]
            loss = ((w * F.binary_cross_entropy_with_logits(z, y, reduction="none")).sum()
                    + (p[:d] ** 2).sum() / (2.0 * c)) / n
            loss.backward()
            return loss

        opt.step(closure)
        theta = p.detach()
        out[c] = (theta[:d].clone(), theta[d].clone())
    return out


def fit_and_score(Xall: torch.Tensor, train, cs) -> dict:
    """Standardise on the train rows, fit every c, and return {c: (300, 520) P(failure)}."""
    r, t, y, w = train
    Xtr = Xall[torch.from_numpy(r).to(DEV), torch.from_numpy(t).to(DEV)].float()
    mu = Xtr.mean(0)
    sd = Xtr.std(0, unbiased=False)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    Xtr = (Xtr - mu) / sd
    fits = fit_logreg(Xtr, torch.from_numpy(y.astype(np.float32)).to(DEV),
                      torch.from_numpy(w.astype(np.float32)).to(DEV), cs)
    del Xtr
    W = torch.stack([fits[c][0] for c in cs], 1)                      # (d, n_c)
    b = torch.stack([fits[c][1] for c in cs])                         # (n_c,)
    P = torch.empty((len(cs), Xall.shape[0], T), device=DEV)
    for i in range(0, Xall.shape[0], CHUNK):
        Z = ((Xall[i:i + CHUNK].float() - mu) / sd) @ W + b          # (chunk, T, n_c)
        P[:, i:i + CHUNK] = torch.sigmoid(Z).permute(2, 0, 1)
    P = P.cpu().numpy()
    return {c: P[k] for k, c in enumerate(cs)}


def evaluate(p, idx, kept, split) -> dict:
    s = running_sum(p, kept)
    return {lab: m1(s, kept, idx, split["pos"][key]) for lab, key in SETS}


# --- --check: GPU vs scikit-learn on one probe ---------------------------------------
def check(idx, kept, splits, CHECK_C: float) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sp = splits[CHECK_SEED]
    train = train_rows_and_weights(idx, kept, sp["pos"]["train"])
    Xall = load_probe_input(CHECK_PROBE)
    t0 = time.time()
    p_gpu = fit_and_score(Xall, train, [CHECK_C])[CHECK_C]
    t_gpu = time.time() - t0

    r, t, y, w = train
    Xtr = Xall[torch.from_numpy(r).to(DEV), torch.from_numpy(t).to(DEV)].float().cpu().numpy()
    t0 = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=CHECK_C, max_iter=2000))
    model.fit(Xtr, y, logisticregression__sample_weight=w)
    t_sk = time.time() - t0
    Xnp = Xall.cpu().numpy()
    p_sk = np.empty((idx.n, T), dtype=np.float32)
    for i in range(0, idx.n, CHUNK):
        p_sk[i:i + CHUNK] = model.predict_proba(
            Xnp[i:i + CHUNK].reshape(-1, Xnp.shape[-1]).astype(np.float32))[:, 1].reshape(-1, T)

    up = sp["pos"]["unseen"]
    rr, tt = np.nonzero(kept[up])
    yy = idx.y[up][rr]
    au_gpu = roc_auc_score(yy, p_gpu[up][rr, tt])
    au_sk = roc_auc_score(yy, p_sk[up][rr, tt])
    e_gpu, e_sk = evaluate(p_gpu, idx, kept, sp), evaluate(p_sk, idx, kept, sp)
    print(f"[check] {CHECK_PROBE}, seed {CHECK_SEED}, c = {CHECK_C:g}; GPU {t_gpu:.1f} s, "
          f"scikit-learn {t_sk:.1f} s")
    print(f"   unseen row AUROC   GPU {au_gpu:.4f}  sklearn {au_sk:.4f}  diff {abs(au_gpu - au_sk):.4f}")
    worst = abs(au_gpu - au_sk)
    for lab, _ in SETS:
        for met in METRICS:
            a, b = e_gpu[lab][met], e_sk[lab][met]
            worst = max(worst, abs(a - b))
            print(f"   M1 {met:8s} {lab:6s} GPU {a:.4f}  sklearn {b:.4f}  diff {abs(a - b):.4f}")
    print(f"   max |p_gpu - p_sk| over kept rows {np.abs(p_gpu - p_sk)[kept].max():.4f}")
    print(f"[check] {'PASSED' if worst <= CHECK_TOL else 'FAILED'} (tolerance {CHECK_TOL})")


# --- Figure --------------------------------------------------------------------------
def figure(a_rows, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_rows = list(csv.DictReader((RESULTS_DIR / "c_probes.csv").open()))
    ink, muted, grid, surface = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
    col = {"A": "#2a78d6", "C": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=surface)
    for ax, (lab, title) in zip(axes, (("seen", "Eval-seen tasks"), ("unseen", "Unseen tasks"))):
        ax.set_facecolor(surface)
        for setup, rows, key in (("A", [r for r in a_rows if r["probe"].startswith("layer")], "probe"),
                                 ("C", [r for r in c_rows if r["probe"].startswith("layer")], "probe")):
            L = np.array([int(r[key][5:]) for r in rows])
            m = np.array([float(r[f"per_task_{lab}_mean"]) for r in rows])
            s = np.array([float(r[f"per_task_{lab}_std"]) for r in rows])
            ax.fill_between(L, m - s, m + s, color=col[setup], alpha=0.12, linewidth=0)
            ax.plot(L, m, color=col[setup], linewidth=2, marker="o", markersize=4,
                    label=f"{setup}: {'hidden state' if setup == 'A' else 'its constraints'}")
            ax.annotate(setup, (L[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        color=ink, fontsize=10, va="center")
        ax.axhline(0.5, color=muted, linewidth=1, linestyle="--")
        ax.text(0.3, 0.505, "floor (0.5)", color=muted, fontsize=8, va="bottom")
        ax.axvline(POST_NORM_INDEX, color=grid, linewidth=6, zorder=0)
        ax.text(POST_NORM_INDEX, 0.405, "32 =\npost-norm", color=muted, fontsize=7, ha="center")
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("Layer (stored index)", color=muted)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.tick_params(colors=muted)
        ax.set_xlim(-1, 34)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("M1 per task (mean ± std, 3 seeds)", color=muted)
    axes[1].legend(frameon=False, loc="upper right", labelcolor=ink)
    fig.suptitle("Failure detection by layer: raw hidden state (A) vs that layer's constraints (C)",
                 color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=float, default=None, metavar="C",
                    help="GPU vs scikit-learn on one probe at this c")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")

    t0 = time.time()
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}
    if args.check is not None:
        check(idx, kept, splits, args.check)
        return
    trains = {s: train_rows_and_weights(idx, kept, splits[s]["pos"]["train"]) for s in SEEDS}

    metrics = {}                       # (probe, seed, c) -> {set: m1 dict}
    probs = {}                         # (probe, seed, c) -> (300, 520) fp16
    for k, probe in enumerate(PROBES):
        Xall = load_probe_input(probe)
        for s in SEEDS:
            P = fit_and_score(Xall, trains[s], list(C_GRID))
            for c, p in P.items():
                metrics[(probe, s, c)] = evaluate(p, idx, kept, splits[s])
                probs[(probe, s, c)] = p.astype(np.float16)
        del Xall
        torch.cuda.empty_cache()
        print(f"[{k + 1:2d}/{len(PROBES)}] {probe} ({time.time() - t0:.0f} s)", flush=True)

    # c: best mean eval_seen M1-per-task over probes and seeds.
    def seen_by_seed(c):
        return np.array([np.mean([metrics[(p, s, c)]["seen"]["per_task"] for p in PROBES])
                         for s in SEEDS])
    grid_score = {c: (float(seen_by_seed(c).mean()), float(seen_by_seed(c).std(ddof=1)))
                  for c in C_GRID}
    c_a = max(C_GRID, key=lambda c: grid_score[c][0])

    def head(p):
        L = probe_layer(p)
        return {"probe": p, "layer": -1 if L is None else L,
                "post_norm": L == POST_NORM_INDEX,
                "inputs": "action_token_ids (7)" if L is None else "pooled hidden state (4096)"}

    grid_rows = [summary_row({**head(p), "c": c}, [metrics[(p, s, c)] for s in SEEDS])
                 for c in C_GRID for p in PROBES]
    write_csv(RESULTS_DIR / "a_grid.csv", grid_rows)
    probe_rows = []
    for p in PROBES:
        row = summary_row({**head(p), "c": c_a}, [metrics[(p, s, c_a)] for s in SEEDS])
        for lab, _ in SETS:
            row[f"floor_{lab}"] = float(np.mean([metrics[(p, s, c_a)][lab]["floor"] for s in SEEDS]))
        probe_rows.append(row)
    write_csv(RESULTS_DIR / "a_probes.csv", probe_rows)

    # Final detector: best eval_seen M1-per-task probe, per seed.
    chosen = {s: max(PROBES, key=lambda p: metrics[(p, s, c_a)]["seen"]["per_task"]) for s in SEEDS}
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = []
    for s in SEEDS:
        p = probs[(chosen[s], s, c_a)].astype(np.float32)
        np.savez_compressed(SCORES_DIR / f"a_seed{s}.npz", rollout_ids=idx.rids, c=c_a,
                            probe=chosen[s], p=p, s=running_sum(p, kept).astype(np.float32))
        np.savez_compressed(PROBE_SCORES_DIR / f"a_probes_seed{s}.npz", c=c_a,
                            probes=np.array(PROBES), rollout_ids=idx.rids,
                            p=np.stack([probs[(q, s, c_a)] for q in PROBES]))
        m = metrics[(chosen[s], s, c_a)]
        final_rows.append({"seed": s, "probe": chosen[s], "c": c_a,
                           **{f"{met}_{lab}": m[lab][met] for lab, _ in SETS for met in METRICS},
                           **{f"floor_{lab}": m[lab]["floor"] for lab, _ in SETS}})
    write_csv(RESULTS_DIR / "a_final.csv", final_rows)

    # Shuffled-label control on each seed's chosen probe.
    rng = np.random.default_rng(PERM_SEED)
    shuf_rows = []
    for s in SEEDS:
        tr = splits[s]["pos"]["train"]
        Xall = load_probe_input(chosen[s])
        for k in range(N_PERM):
            y_perm = idx.y.copy()
            y_perm[tr] = rng.permutation(idx.y[tr])
            idx_perm = dataclasses.replace(idx, y=y_perm)
            # Training rows and weights follow the shuffled labels; the kept mask is the
            # real one (it is a property of the rollout, not of its label).
            train = train_rows_and_weights(idx_perm, kept, tr)
            p = fit_and_score(Xall, train, [c_a])[c_a]
            e = evaluate(p, idx, kept, splits[s])
            real = metrics[(chosen[s], s, c_a)]
            shuf_rows.append({"seed": s, "probe": chosen[s], "perm": k, "c": c_a,
                              **{f"shuffled_{met}_{lab}": e[lab][met] for lab, _ in SETS for met in METRICS},
                              **{f"real_{met}_{lab}": real[lab][met] for lab, _ in SETS for met in METRICS}})
        del Xall
        torch.cuda.empty_cache()
    write_csv(RESULTS_DIR / "a_shuffled.csv", shuf_rows)

    figure(probe_rows, RESULTS_DIR / "a_vs_c_by_layer.png")

    # --- Report ----------------------------------------------------------------------
    print("\n=== c grid (eval_seen M1-per-task, mean of 34 probes, mean ± std over seeds) ===")
    for c in C_GRID:
        print(f"   {c:<8g} {grid_score[c][0]:.3f}±{grid_score[c][1]:.3f}{'  <- chosen' if c == c_a else ''}")
    r0 = probe_rows[0]
    print(f"\n=== A probes at c = {c_a:g}, running-sum score (eval_seen | unseen; floor pooled "
          f"{r0['floor_seen']:.3f} | {r0['floor_unseen']:.3f}, per-task 0.5; tasks "
          f"{r0['n_tasks_seen']} | {r0['n_tasks_unseen']}) ===")
    for r in probe_rows:
        print(f"   {r['probe']:13s} pooled {r['pooled_seen_mean']:.3f}±{r['pooled_seen_std']:.3f} | "
              f"{r['pooled_unseen_mean']:.3f}±{r['pooled_unseen_std']:.3f}   per-task "
              f"{r['per_task_seen_mean']:.3f}±{r['per_task_seen_std']:.3f} | "
              f"{r['per_task_unseen_mean']:.3f}±{r['per_task_unseen_std']:.3f}"
              f"{'  (post-norm)' if r['post_norm'] else ''}")
    print("\n=== final detector: best eval_seen M1-per-task probe per seed ===")
    for r in final_rows:
        print(f"   seed {r['seed']}: {r['probe']:13s} seen per-task {r['per_task_seen']:.3f}  unseen "
              f"per-task {r['per_task_unseen']:.3f}, pooled {r['pooled_unseen']:.3f}")
    u = np.array([r["per_task_unseen"] for r in final_rows])
    up = np.array([r["pooled_unseen"] for r in final_rows])
    print(f"   unseen per-task {u.mean():.3f}±{u.std(ddof=1):.3f}, pooled {up.mean():.3f}±{up.std(ddof=1):.3f}")

    print(f"\n=== shuffled-label control ({N_PERM} permutations per seed, chosen probe) ===")
    for s in SEEDS:
        rs = [r for r in shuf_rows if r["seed"] == s]
        for lab, _ in SETS:
            sh = np.array([r[f"shuffled_per_task_{lab}"] for r in rs])
            shp = np.array([r[f"shuffled_pooled_{lab}"] for r in rs])
            real = rs[0][f"real_per_task_{lab}"]
            print(f"   seed {s} {chosen[s]:13s} {lab:6s} per-task real {real:.3f}, shuffled "
                  f"{sh.mean():.3f} (range {sh.min():.3f}-{sh.max():.3f}), real - shuffled "
                  f"{real - sh.mean():+.3f}; pooled shuffled {shp.mean():.3f}")
    all_sh = np.array([r[f"shuffled_{met}_{lab}"] for r in shuf_rows for lab, _ in SETS
                       for met in METRICS])
    print(f"   all shuffled M1 within 0.40-0.60: {bool(np.all((all_sh >= 0.4) & (all_sh <= 0.6)))} "
          f"(range {all_sh.min():.3f}-{all_sh.max():.3f})")

    l0 = next(r for r in probe_rows if r["probe"] == "layer00")
    best = max(probe_rows, key=lambda r: r["per_task_seen_mean"])
    print(f"\n=== layer 0 vs best ===\n   layer00 per-task seen {l0['per_task_seen_mean']:.3f}, "
          f"unseen {l0['per_task_unseen_mean']:.3f}; best on seen: {best['probe']} seen "
          f"{best['per_task_seen_mean']:.3f}, unseen {best['per_task_unseen_mean']:.3f}; layer 0 "
          f"below best on seen: {l0['per_task_seen_mean'] < best['per_task_seen_mean']}, on unseen: "
          f"{l0['per_task_unseen_mean'] < best['per_task_unseen_mean']}")
    print(f"\n[*] done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
