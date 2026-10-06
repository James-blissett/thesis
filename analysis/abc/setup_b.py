"""
setup_b.py

Stage 5: setup B, hidden states -> constraints H steps ahead, then a final detector on
outcome (handoff v10, sections 2 and 4).

Grid (no outcome labels)
  * For each raw layer l in 0..31 and target in [act_rep, act_mag, emb_temp[l],
    xl_final[min(l,30)]], at H in {0, 10, 25}: the prepared target series (prep.py, at
    Stage 3's beta per family, scaled and oriented on that seed's train split),
    looked ahead as y[t] = max(target[t : t+H+1]) without reading past the rollout's
    last kept row, then binarised at the split that minimises within-group squared
    error of the training values (two groups). Targets flagged unstable in Stage 2 at
    that beta are skipped (read from results/abc/constraint_m1.csv).
  * Input: layer l's pooled hidden state at t. Balanced class weights, no row weights.
  * All of one layer's targets x H are fitted together on the GPU (one linear layer, one
    output per probe; the outputs' objectives are independent), the same objective as
    scikit-learn's LogisticRegression(C=c, class_weight="balanced"). `--check` compares
    one (layer, target, H) with scikit-learn: test AUROC within CHECK_TOL.
  * Rows: kept train rows (all of them: the >= 5-successes cut is for outcome models).
  * One c for the whole grid: the best mean eval_seen AUROC of the probes at predicting
    their own binarised targets (mean over probes, H and seeds).  [pass 1]
  * At that c, out-of-fold outputs on train (GroupKFold(5) by rollout; thresholds and
    standardisation refitted inside each fold) and a refit on all of train for every
    other rollout.  [pass 2; probe outputs saved to /data/tmp/abc_scores]

Final detector (one per H; CPU)
  * Inputs: the grid's probabilities at that H (out-of-fold on train rollouts).
  * Trained on outcome exactly as A and C: data.train_rows_and_weights, running-sum
    score, c chosen on eval_seen M1-per-task (best mean over seeds).
  * Ablation: the same detector on each target's probes alone.

Outputs (results/abc/):
    b_grid.csv           every probe at every c: target AUROC on eval_seen and unseen
    b_grid_c.csv         the grid's c choice
    b_map_H{H}.png       layers x targets heatmap of unseen target AUROC at the chosen c
    b_final.csv          final detector and per-target ablations per H, at its c
    b_final_grid.csv     final detector per H at every c, with weight sign stability
    b_final_weights.csv  final detector coefficients per H at its c
    scores/b_h{H}_seed{s}.npz  final detector p and running sum s (and ablations)

Usage (from the repo root):
    source env.sh
    python analysis/abc/setup_b.py --check 1e-3   # GPU vs scikit-learn on one probe
    python analysis/abc/setup_b.py                # the full stage (run in tmux)
    python analysis/abc/setup_b.py --final-only   # redo the final detectors from saved
                                                  # probe outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from data import (N_RAW, RESULTS_DIR, SEEDS, T, kept_mask, load_index, load_split,
                  train_rows_and_weights)
from prep import PRIMARY, apply_prep, fit_scale_orient, load_series, smooth
from setup_a import CHUNK, DEV, MAX_ITER, load_probe_input
from setup_c import (C_GRID, FAMILIES, METRICS, SETS, choose_betas, evaluate, fam_of,
                     fit_prob, read_stage2, running_sum, series_name, summary_row, write_csv)

HS = (0, 10, 25)
N_FOLDS = 5
TARGET_KINDS = ("act_rep", "act_mag", "emb_temp", "xl_final")
OUT = RESULTS_DIR
SCORES_DIR = RESULTS_DIR / "scores"
PROBE_DIR = Path("/data/tmp/abc_scores")
LAYERS = list(range(N_RAW))
CHECK_TOL = 0.005
CHECK = {"layer": 16, "target": "emb_temp[16]", "H": 10, "seed": 0}


def target_name(kind: str, layer: int) -> str:
    if kind in ("act_rep", "act_mag"):
        return kind
    return f"{kind}[{layer if kind == 'emb_temp' else min(layer, 30)}]"


def probe_name(layer: int, kind: str, H: int) -> str:
    return f"L{layer:02d}:{kind}:H{H}"


# --- Targets -------------------------------------------------------------------------
def prepared_targets(idx, kept, splits, chosen) -> dict:
    """{(seed, target): (300, 520) prepared series, NaN off kept rows}."""
    out = {}
    for fam in ("action", "emb_temp", "cross_layer"):
        bases = {"action": ["act_rep", "act_mag"], "emb_temp": ["emb_temp"],
                 "cross_layer": ["xl_final"]}[fam]
        S, names = load_series(idx, [(b, l) for b, l in PRIMARY if b in bases])
        beta = 0.0 if chosen[fam] == "none" else float(chosen[fam])
        E = smooth(S, names, beta)
        for s in SEEDS:
            mu, sd, sign, _ = fit_scale_orient(E, idx, kept, splits[s]["pos"]["train"])
            P = apply_prep(E, mu, sd, sign).astype(np.float32)
            for k, (b, L) in enumerate(names):
                out[(s, series_name(b, L))] = np.where(kept, P[k], np.nan)
    return out


def lookahead(P: np.ndarray, H: int) -> np.ndarray:
    """y[t] = max(P[t : t+H+1]); P is NaN off kept rows, so nothing past the last kept
    row is read. NaN stays NaN off kept rows."""
    y = P.copy()
    for k in range(1, H + 1):
        sh = np.full_like(P, np.nan)
        sh[:, :-k] = P[:, k:]
        y = np.fmax(y, sh)
    y[np.isnan(P)] = np.nan
    return y


def two_means_threshold(v: np.ndarray) -> float:
    """Split point minimising the within-group squared error of v (two groups)."""
    v = np.sort(v.astype(np.float64))
    n = len(v)
    c1 = np.cumsum(v)
    c2 = np.cumsum(v * v)
    i = np.arange(1, n)                                     # left group = v[:i]
    left = c2[i - 1] - c1[i - 1] ** 2 / i
    right = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / (n - i)
    sse = left + right
    sse[v[i - 1] == v[i]] = np.inf                           # only split between distinct values
    j = int(np.argmin(sse))
    if not np.isfinite(sse[j]):
        raise ValueError("constant target")
    return float((v[j] + v[j + 1]) / 2)


# --- GPU: many balanced logistic regressions sharing one input -----------------------
def fit_multi(X, Y, M, cs) -> dict:
    """{c: (W (d, k), b (k,))}. X (n, d) standardised; Y (n, k) in {0, 1}; M (n, k)
    per-row weights (balanced class weights, 0 where the target is missing). Each
    output's objective is sklearn's sum_i m_i logloss_i + ||w||^2 / (2c); all are
    minimised together (they are independent), divided by n for conditioning."""
    n, d = X.shape
    k = Y.shape[1]
    theta = torch.zeros(d + 1, k, device=DEV)
    out = {}
    for c in sorted(cs):
        p = theta.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([p], lr=1.0, max_iter=MAX_ITER, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Z = X @ p[:d] + p[d]
            loss = ((M * F.binary_cross_entropy_with_logits(Z, Y, reduction="none")).sum()
                    + (p[:d] ** 2).sum() / (2.0 * c)) / n
            loss.backward()
            return loss

        opt.step(closure)
        theta = p.detach()
        out[c] = (theta[:d].clone(), theta[d].clone())
    return out


def balanced_weights(Y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """sklearn's class_weight='balanced' per output column: n / (2 n_class)."""
    M = np.zeros_like(Y, dtype=np.float32)
    for j in range(Y.shape[1]):
        v = valid[:, j]
        n, n1 = v.sum(), (Y[v, j] == 1).sum()
        n0 = n - n1
        if n0 == 0 or n1 == 0:
            continue
        M[v & (Y[:, j] == 1), j] = n / (2 * n1)
        M[v & (Y[:, j] == 0), j] = n / (2 * n0)
    return M


def fit_grid_block(Xall, rows, targets_y, cs):
    """Fit one layer's probes on `rows` = (r, t). targets_y: list of (300, 520) look-ahead
    series, one per probe. Thresholds and standardisation come from these rows.
    Returns ({c: (W, b)}, thresholds, mu, sd)."""
    r, t = rows
    vals = np.stack([y[r, t] for y in targets_y], 1)                 # (n, k)
    valid = np.isfinite(vals)
    thr = np.array([two_means_threshold(vals[valid[:, j], j]) for j in range(vals.shape[1])])
    Y = (vals > thr).astype(np.float32)
    M = balanced_weights(Y, valid)
    rt = (torch.from_numpy(r).to(DEV), torch.from_numpy(t).to(DEV))
    X = Xall[rt].float()
    mu = X.mean(0)
    sd = X.std(0, unbiased=False)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    X = (X - mu) / sd
    fits = fit_multi(X, torch.from_numpy(Y).to(DEV), torch.from_numpy(M).to(DEV), cs)
    del X
    return fits, thr, mu, sd


def predict_block(Xall, which, W, b, mu, sd) -> np.ndarray:
    """(k, len(which), 520) probabilities for whole rollouts `which`."""
    out = torch.empty((W.shape[1], len(which), T), device=DEV)
    wi = torch.from_numpy(np.asarray(which)).to(DEV)
    for i in range(0, len(which), CHUNK):
        Z = ((Xall[wi[i:i + CHUNK]].float() - mu) / sd) @ W + b      # (chunk, T, k)
        out[:, i:i + CHUNK] = torch.sigmoid(Z).permute(2, 0, 1)
    return out.cpu().numpy()


def target_auroc(p, y, thr, kept, pos) -> float:
    """Row AUROC of one probe against its own binarised target on kept rows of `pos`."""
    m = kept[pos] & np.isfinite(y[pos])
    lab = (y[pos][m] > thr).astype(int)
    if lab.min() == lab.max():
        return float("nan")
    return float(roc_auc_score(lab, p[pos][m]))


# --- Grid: both passes ---------------------------------------------------------------
def run_grid(idx, kept, splits, layer_targets, targets) -> tuple:
    """Pass 1 (every c, refit on all train, target AUROC) then pass 2 (chosen c: OOF on
    train + refit for the rest). Saves probe outputs per (H, seed)."""
    t0 = time.time()
    auroc = {}                                     # (layer, kind, H, seed, c) -> {set: auc}
    for L in LAYERS:
        Xall = load_probe_input(f"layer{L:02d}")
        for s in SEEDS:
            sp = splits[s]
            tr = sp["pos"]["train"]
            rows = np.nonzero(kept[tr])
            rows = (tr[rows[0]], rows[1])
            ys = [lookahead(targets[(s, target_name(kd, L))], H)
                  for kd in layer_targets[L] for H in HS]
            fits, thr, mu, sd = fit_grid_block(Xall, rows, ys, list(C_GRID))
            ev = np.concatenate([sp["pos"]["eval_seen"], sp["pos"]["unseen"]])
            for c in C_GRID:
                P = predict_block(Xall, ev, *fits[c], mu, sd)
                full = np.zeros((len(ys), idx.n, T), dtype=np.float32)
                full[:, ev] = P
                j = 0
                for kd in layer_targets[L]:
                    for H in HS:
                        auroc[(L, kd, H, s, c)] = {
                            lab: target_auroc(full[j], ys[j], thr[j], kept, sp["pos"][key])
                            for lab, key in SETS}
                        j += 1
        del Xall
        torch.cuda.empty_cache()
        print(f"[pass 1 {L + 1:2d}/{len(LAYERS)}] layer {L} ({time.time() - t0:.0f} s)", flush=True)

    probes = [(L, kd, H) for L in LAYERS for kd in layer_targets[L] for H in HS]
    c_score = {c: np.array([np.nanmean([auroc[(L, kd, H, s, c)]["seen"] for L, kd, H in probes])
                            for s in SEEDS]) for c in C_GRID}
    c_grid = max(C_GRID, key=lambda c: c_score[c].mean())
    print(f"[*] grid c = {c_grid:g}", flush=True)

    outputs = {(H, s): {} for H in HS for s in SEEDS}               # name -> (300, 520) fp16
    for L in LAYERS:
        Xall = load_probe_input(f"layer{L:02d}")
        for s in SEEDS:
            sp = splits[s]
            tr = sp["pos"]["train"]
            ys = [lookahead(targets[(s, target_name(kd, L))], H)
                  for kd in layer_targets[L] for H in HS]
            names = [probe_name(L, kd, H) for kd in layer_targets[L] for H in HS]
            full = np.zeros((len(ys), idx.n, T), dtype=np.float32)
            # Out-of-fold on train rollouts.
            for f_tr, f_te in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr):
                ftr = tr[f_tr]
                rr = np.nonzero(kept[ftr])
                fits, thr, mu, sd = fit_grid_block(Xall, (ftr[rr[0]], rr[1]), ys, [c_grid])
                full[:, tr[f_te]] = predict_block(Xall, tr[f_te], *fits[c_grid], mu, sd)
            # Refit on all of train for everything else.
            rr = np.nonzero(kept[tr])
            fits, thr, mu, sd = fit_grid_block(Xall, (tr[rr[0]], rr[1]), ys, [c_grid])
            rest = np.setdiff1d(np.arange(idx.n), tr)
            full[:, rest] = predict_block(Xall, rest, *fits[c_grid], mu, sd)
            for j, nm in enumerate(names):
                H = int(nm.split(":H")[1])
                outputs[(H, s)][nm] = full[j].astype(np.float16)
        del Xall
        torch.cuda.empty_cache()
        print(f"[pass 2 {L + 1:2d}/{len(LAYERS)}] layer {L} ({time.time() - t0:.0f} s)", flush=True)

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for (H, s), d in outputs.items():
        np.savez(PROBE_DIR / f"b_probes_h{H}_seed{s}.npz", names=np.array(list(d)),
                 p=np.stack(list(d.values())), c=c_grid, rollout_ids=idx.rids)
    return auroc, probes, c_grid, c_score


# --- Final detector ------------------------------------------------------------------
def final_job(H, s, c, names, feats, idx, kept, split, train):
    model, p, ok = fit_prob(names, feats, train, c)
    return {"H": H, "seed": s, "c": c, "names": names, "metrics": evaluate(running_sum(p, kept), idx, kept, split),
            "coef": model[-1].coef_[0].tolist(), "p": p, "converged": ok}


def run_final(idx, kept, splits, n_jobs) -> None:
    trains = {s: train_rows_and_weights(idx, kept, splits[s]["pos"]["train"]) for s in SEEDS}
    feats = {}
    for H in HS:
        for s in SEEDS:
            z = np.load(PROBE_DIR / f"b_probes_h{H}_seed{s}.npz")
            feats[(H, s)] = {str(n): z["p"][k].astype(np.float32) for k, n in enumerate(z["names"])}
    jobs = Parallel(n_jobs=n_jobs)(
        delayed(final_job)(H, s, c, list(feats[(H, s)]), feats[(H, s)], idx, kept, splits[s], trains[s])
        for H in HS for s in SEEDS for c in C_GRID)
    res = {(j["H"], j["seed"], j["c"]): j for j in jobs}

    grid_rows, chosen_c = [], {}
    for H in HS:
        def seen(c):
            return np.array([res[(H, s, c)]["metrics"]["seen"]["per_task"] for s in SEEDS])
        chosen_c[H] = max(C_GRID, key=lambda c: seen(c).mean())
        for c in C_GRID:
            names = res[(H, SEEDS[0], c)]["names"]
            cs = np.array([res[(H, s, c)]["coef"] for s in SEEDS])
            n_cons = int(np.sum(np.all(cs > 0, 0) | np.all(cs < 0, 0)))
            grid_rows.append(summary_row({"H": H, "c": c, "chosen": c == chosen_c[H],
                                          "n_inputs": len(names), "sign_consistent": n_cons},
                                         [res[(H, s, c)]["metrics"] for s in SEEDS]))
    write_csv(OUT / "b_final_grid.csv", grid_rows)

    # Per-target ablations at the chosen c.
    abl = Parallel(n_jobs=n_jobs)(
        delayed(final_job)(H, s, chosen_c[H], [n for n in feats[(H, s)] if n.split(":")[1] == kd],
                           feats[(H, s)], idx, kept, splits[s], trains[s])
        for H in HS for s in SEEDS for kd in TARGET_KINDS)
    abl_res = {(j["H"], j["seed"], j["names"][0].split(":")[1]): j for j in abl}

    final_rows, weight_rows = [], []
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for H in HS:
        c = chosen_c[H]
        names = res[(H, SEEDS[0], c)]["names"]
        cs = np.array([res[(H, s, c)]["coef"] for s in SEEDS])
        final_rows.append(summary_row({"H": H, "model": "full", "c": c, "n_inputs": len(names),
                                       "sign_consistent": int(np.sum(np.all(cs > 0, 0) | np.all(cs < 0, 0)))},
                                      [res[(H, s, c)]["metrics"] for s in SEEDS]))
        for kd in TARGET_KINDS:
            n_in = len(abl_res[(H, SEEDS[0], kd)]["names"])
            final_rows.append(summary_row({"H": H, "model": f"only_{kd}", "c": c, "n_inputs": n_in,
                                           "sign_consistent": ""},
                                          [abl_res[(H, s, kd)]["metrics"] for s in SEEDS]))
        for k, n in enumerate(names):
            weight_rows.append({"H": H, "input": n, "c": c, "coef_mean": float(cs[:, k].mean()),
                                "coef_std": float(cs[:, k].std(ddof=1)),
                                "sign_consistent": bool(np.all(cs[:, k] > 0) or np.all(cs[:, k] < 0)),
                                **{f"coef_s{s}": float(cs[i, k]) for i, s in enumerate(SEEDS)}})
        for s in SEEDS:
            p = res[(H, s, c)]["p"]
            extra = {f"p_only_{kd}": abl_res[(H, s, kd)]["p"] for kd in TARGET_KINDS}
            np.savez_compressed(SCORES_DIR / f"b_h{H}_seed{s}.npz", rollout_ids=idx.rids, c=c, H=H,
                                p=p, s=running_sum(p, kept).astype(np.float32), **extra)
    write_csv(OUT / "b_final.csv", final_rows)
    write_csv(OUT / "b_final_weights.csv", weight_rows)
    unconv = sorted({(j["H"], j["c"]) for j in list(jobs) + list(abl) if not j["converged"]})
    if unconv:
        print(f"[warn] final detectors not converged at (H, c): {unconv}")

    floor = res[(HS[0], SEEDS[0], C_GRID[0])]["metrics"]
    fl = {lab: np.mean([res[(HS[0], s, C_GRID[0])]["metrics"][lab]["floor"] for s in SEEDS]) for lab, _ in SETS}
    print(f"\n=== B final detectors (running sum; eval_seen | unseen; floor pooled {fl['seen']:.3f} | "
          f"{fl['unseen']:.3f}, per-task 0.5; tasks {floor['seen']['n_tasks']} | {floor['unseen']['n_tasks']}) ===")
    print("   c grid, eval_seen M1-per-task (mean ± std) and weights keeping their sign:")
    for H in HS:
        line = "   ".join(f"{r['c']:g}: {r['per_task_seen_mean']:.3f}±{r['per_task_seen_std']:.3f} "
                           f"[{r['sign_consistent']}/{r['n_inputs']}]{'*' if r['chosen'] else ''}"
                           for r in grid_rows if r["H"] == H)
        print(f"   H={H:2d}: {line}")
    for r in final_rows:
        print(f"   H={r['H']:2d} {r['model']:16s} c={r['c']:<6g} ({r['n_inputs']:3d} in) "
              f"pooled {r['pooled_seen_mean']:.3f}±{r['pooled_seen_std']:.3f} | "
              f"{r['pooled_unseen_mean']:.3f}±{r['pooled_unseen_std']:.3f}   per-task "
              f"{r['per_task_seen_mean']:.3f}±{r['per_task_seen_std']:.3f} | "
              f"{r['per_task_unseen_mean']:.3f}±{r['per_task_unseen_std']:.3f}"
              f"{'   sign consistent ' + str(r['sign_consistent']) + '/' + str(r['n_inputs']) if r['model'] == 'full' else ''}")


# --- Map -----------------------------------------------------------------------------
def write_map(auroc, probes, c, layer_targets) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rows = []
    for (L, kd, H) in probes:
        for cc in C_GRID:
            v = {lab: np.array([auroc[(L, kd, H, s, cc)][lab] for s in SEEDS]) for lab, _ in SETS}
            rows.append({"layer": L, "target": kd, "target_series": target_name(kd, L), "H": H,
                         "c": cc, "chosen_c": cc == c,
                         **{f"auroc_{lab}_mean": float(np.nanmean(v[lab])) for lab, _ in SETS},
                         **{f"auroc_{lab}_std": float(np.nanstd(v[lab], ddof=1)) for lab, _ in SETS},
                         **{f"auroc_{lab}_s{s}": float(v[lab][i]) for lab, _ in SETS
                            for i, s in enumerate(SEEDS)}})
    write_csv(OUT / "b_grid.csv", rows)

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue", ramp)
    cmap.set_bad("#efeee9")
    ink, muted, surface = "#1f1f1e", "#6b6a63", "#fcfcfb"
    print(f"\n=== layer x target map: unseen AUROC at predicting the probe's own target (c = {c:g}) ===")
    for H in HS:
        M = np.full((max(LAYERS) + 1, len(TARGET_KINDS)), np.nan)
        for r in rows:
            if r["H"] == H and r["chosen_c"]:
                M[r["layer"], TARGET_KINDS.index(r["target"])] = r["auroc_unseen_mean"]
        fig, ax = plt.subplots(figsize=(5.2, 9.5), facecolor=surface)
        ax.set_facecolor(surface)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
        for L in LAYERS:
            for j in range(len(TARGET_KINDS)):
                v = M[L, j]
                ax.text(j, L, "unstable" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color=muted if np.isnan(v) else ("#ffffff" if v > 0.78 else ink))
        ax.set_xticks(range(len(TARGET_KINDS)),
                      ["act_rep", "act_mag", "emb_temp[ℓ]", "xl_final[min(ℓ,30)]"], fontsize=8, color=ink)
        ax.set_yticks(range(0, len(M), 2), [str(L) for L in range(0, len(M), 2)], fontsize=8, color=muted)
        ax.set_ylabel("Layer ℓ (probe input)", color=muted)
        ax.xaxis.tick_top()
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Unseen-task AUROC (≤ 0.5 shown as 0.5)", color=muted, fontsize=8)
        cb.ax.tick_params(colors=muted, labelsize=7)
        cb.outline.set_visible(False)
        ax.set_title(f"B: predicting each constraint {H} steps ahead", color=ink, fontsize=10,
                     loc="left", pad=28)
        fig.tight_layout()
        fig.savefig(OUT / f"b_map_H{H}.png", dpi=160, facecolor=surface)
        plt.close(fig)
        print(f"   H = {H}:  layer  " + "  ".join(f"{k:>9s}" for k in TARGET_KINDS))
        for L in LAYERS:
            print("              " + f"{L:5d}  " + "  ".join("   unstbl" if np.isnan(v) else f"{v:9.3f}"
                                                         for v in M[L]))


# --- --check -------------------------------------------------------------------------
def check(idx, kept, splits, layer_targets, targets, c) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    L, s, H, tgt = CHECK["layer"], CHECK["seed"], CHECK["H"], CHECK["target"]
    sp = splits[s]
    tr = sp["pos"]["train"]
    rr = np.nonzero(kept[tr])
    rows = (tr[rr[0]], rr[1])
    ys = [lookahead(targets[(s, target_name(kd, L))], hh) for kd in layer_targets[L] for hh in HS]
    names = [probe_name(L, kd, hh) for kd in layer_targets[L] for hh in HS]
    j = names.index(probe_name(L, "emb_temp", H))
    Xall = load_probe_input(f"layer{L:02d}")
    t0 = time.time()
    fits, thr, mu, sd = fit_grid_block(Xall, rows, ys, [c])
    up = sp["pos"]["unseen"]
    full = np.zeros((idx.n, T), dtype=np.float32)
    full[up] = predict_block(Xall, up, *fits[c], mu, sd)[j]
    t_gpu = time.time() - t0
    a_gpu = target_auroc(full, ys[j], thr[j], kept, up)

    y = (ys[j][rows] > thr[j]).astype(int)
    Xtr = Xall[torch.from_numpy(rows[0]).to(DEV), torch.from_numpy(rows[1]).to(DEV)].float().cpu().numpy()
    t0 = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
    model.fit(Xtr, y)
    t_sk = time.time() - t0
    Xu = Xall[torch.from_numpy(up).to(DEV)].float().cpu().numpy()
    psk = np.zeros((idx.n, T), dtype=np.float32)
    psk[up] = model.predict_proba(Xu.reshape(-1, Xu.shape[-1]))[:, 1].reshape(len(up), T)
    a_sk = target_auroc(psk, ys[j], thr[j], kept, up)
    print(f"[check] layer {L}, target {tgt}, H = {H}, seed {s}, c = {c:g} "
          f"(threshold {thr[j]:.3f}, train positives {y.mean():.3f}); GPU {t_gpu:.1f} s "
          f"({len(names)} probes jointly), scikit-learn {t_sk:.1f} s (one probe)")
    print(f"   unseen target AUROC  GPU {a_gpu:.4f}  sklearn {a_sk:.4f}  diff {abs(a_gpu - a_sk):.4f}")
    print(f"[check] {'PASSED' if abs(a_gpu - a_sk) <= CHECK_TOL else 'FAILED'} (tolerance {CHECK_TOL})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=float, default=None, metavar="C")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", type=Path, default=None, metavar="DIR",
                    help="layers 0-1 only, every output under DIR: checks the pipeline end to end")
    args = ap.parse_args()
    global OUT, SCORES_DIR, PROBE_DIR, LAYERS
    if args.smoke is not None:
        OUT, SCORES_DIR, PROBE_DIR, LAYERS = args.smoke, args.smoke / "scores", args.smoke, [0, 1]
        OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}

    if args.final_only:
        run_final(idx, kept, splits, args.n_jobs)
        print(f"\n[*] done ({time.time() - t0:.0f} s)")
        return

    s2 = read_stage2()
    chosen, _ = choose_betas(s2)
    unstable = {series_name(r["constraint"], r["layer"]) for r in s2
                if r["unstable"] and r["beta"] == chosen[fam_of(r["constraint"])]}
    layer_targets = {L: [kd for kd in TARGET_KINDS if target_name(kd, L) not in unstable]
                     for L in LAYERS}
    skipped = sorted({target_name(kd, L) for L in LAYERS for kd in TARGET_KINDS
                      if kd not in layer_targets[L]})
    n_probes = sum(len(layer_targets[L]) for L in LAYERS)
    print(f"[*] beta per family {chosen}; skipped unstable targets {skipped}; "
          f"{n_probes} probes per H, {n_probes * len(HS)} in all")
    targets = prepared_targets(idx, kept, splits, chosen)

    if args.check is not None:
        check(idx, kept, splits, layer_targets, targets, args.check)
        return

    auroc, probes, c_grid, c_score = run_grid(idx, kept, splits, layer_targets, targets)
    write_csv(OUT / "b_grid_c.csv",
              [{"c": c, "seen_target_auroc_mean": float(c_score[c].mean()),
                "seen_target_auroc_std": float(c_score[c].std(ddof=1)), "chosen": c == c_grid}
               for c in C_GRID])
    print("\n=== grid c: mean eval_seen AUROC at predicting own targets (mean ± std over seeds) ===")
    for c in C_GRID:
        print(f"   {c:<8g} {c_score[c].mean():.3f}±{c_score[c].std(ddof=1):.3f}"
              f"{'  <- chosen' if c == c_grid else ''}")
    write_map(auroc, probes, c_grid, layer_targets)
    (OUT / "b_meta.json").write_text(json.dumps(
        {"beta": chosen, "skipped_unstable_targets": skipped, "grid_c": c_grid,
         "n_probes_per_H": n_probes}, indent=1))
    run_final(idx, kept, splits, args.n_jobs)
    print(f"\n[*] done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
