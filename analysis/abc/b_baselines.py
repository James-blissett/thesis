"""
b_baselines.py

Baselines for B's layer x target maps: how well each grid target (binarised, H steps
ahead) can be predicted WITHOUT a probe.

  * persistence: the target's own current prepared value at t (no look-ahead). At H = 0
    this is the target itself, so its AUROC is 1 by construction.
  * timestep-only: t.

Targets, look-ahead and thresholds are exactly setup_b's refit: prepared series at
Stage 3's beta per family, y[t] = max(target[t : t+H+1]) within kept rows, threshold
by two-group squared-error split on the seed's kept train rows. AUROC on kept rows of
eval_seen and unseen, mean +- std over seeds. act_rep and act_mag do not depend on the
layer, so their baselines are one number per H; emb_temp[l] and xl_final[min(l,30)]
get one per layer.

Outputs (results/abc/):
    b_baselines.csv                 persistence and timestep AUROC per target and H
    b_map_baselines_H{H}.png        probe | persistence | timestep, layers x targets

Usage (from the repo root; about a minute, CPU only):
    source env.sh
    python analysis/abc/b_baselines.py
"""

from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import roc_auc_score

from data import N_RAW, RESULTS_DIR, SEEDS, T, kept_mask, load_index, load_split
from setup_b import (HS, TARGET_KINDS, lookahead, prepared_targets, target_name,
                     two_means_threshold)
from setup_c import SETS, choose_betas, fam_of, read_stage2, series_name, write_csv


def auroc_rows(score, y, thr, kept, pos) -> float:
    m = kept[pos] & np.isfinite(y[pos])
    lab = (y[pos][m] > thr).astype(int)
    if lab.min() == lab.max():
        return float("nan")
    return float(roc_auc_score(lab, score[pos][m]))


def main() -> None:
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}
    s2 = read_stage2()
    chosen, _ = choose_betas(s2)
    unstable = {series_name(r["constraint"], r["layer"]) for r in s2
                if r["unstable"] and r["beta"] == chosen[fam_of(r["constraint"])]}
    targets = prepared_targets(idx, kept, splits, chosen)
    tgrid = np.broadcast_to(np.arange(T, dtype=np.float32), (idx.n, T))

    series = sorted({target_name(kd, L) for L in range(N_RAW) for kd in TARGET_KINDS} - unstable)
    rows = []
    for name in series:
        for H in HS:
            res = {("persistence", lab): [] for lab, _ in SETS}
            res.update({("timestep", lab): [] for lab, _ in SETS})
            for s in SEEDS:
                P = targets[(s, name)]
                y = lookahead(P, H)
                tr = splits[s]["pos"]["train"]
                thr = two_means_threshold(y[tr][kept[tr] & np.isfinite(y[tr])])
                for lab, key in SETS:
                    pos = splits[s]["pos"][key]
                    res[("persistence", lab)].append(auroc_rows(P, y, thr, kept, pos))
                    res[("timestep", lab)].append(auroc_rows(tgrid, y, thr, kept, pos))
            row = {"target_series": name, "H": H}
            for (b, lab), v in res.items():
                v = np.array(v)
                row[f"{b}_{lab}_mean"] = float(np.nanmean(v))
                row[f"{b}_{lab}_std"] = float(np.nanstd(v, ddof=1))
            rows.append(row)
    write_csv(RESULTS_DIR / "b_baselines.csv", rows)
    base = {(r["target_series"], r["H"]): r for r in rows}

    grid = [r for r in csv.DictReader((RESULTS_DIR / "b_grid.csv").open()) if r["chosen_c"] == "True"]
    probe = {(int(r["layer"]), r["target"], int(r["H"])): float(r["auroc_unseen_mean"]) for r in grid}

    print("=== B targets: unseen AUROC of the probe vs persistence vs timestep-only ===")
    for H in HS:
        print(f"-- H = {H}")
        for kd in TARGET_KINDS:
            layers = [L for L in range(N_RAW) if (L, kd, H) in probe]
            pr = np.array([probe[(L, kd, H)] for L in layers])
            pe = np.array([base[(target_name(kd, L), H)]["persistence_unseen_mean"] for L in layers])
            ts = np.array([base[(target_name(kd, L), H)]["timestep_unseen_mean"] for L in layers])
            best = layers[int(np.argmax(pr))]
            print(f"   {kd:9s} probe median {np.median(pr):.3f} (best L{best} {pr.max():.3f})   "
                  f"persistence median {np.median(pe):.3f}   timestep median {np.median(ts):.3f}   "
                  f"probe > persistence in {int((pr > pe).sum())}/{len(layers)} layers")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue", ramp)
    cmap.set_bad("#efeee9")
    ink, muted, surface = "#1f1f1e", "#6b6a63", "#fcfcfb"
    labels = ["act_rep", "act_mag", "emb_temp[ℓ]", "xl_final[ℓ*]"]
    for H in HS:
        mats = {}
        for title, get in (("Probe (hidden state at t)", lambda L, kd: probe.get((L, kd, H), np.nan)),
                           ("Persistence (target now)",
                            lambda L, kd: base.get((target_name(kd, L), H), {}).get("persistence_unseen_mean", np.nan)
                            if (L, kd, H) in probe else np.nan),
                           ("Timestep only",
                            lambda L, kd: base.get((target_name(kd, L), H), {}).get("timestep_unseen_mean", np.nan)
                            if (L, kd, H) in probe else np.nan)):
            mats[title] = np.array([[get(L, kd) for kd in TARGET_KINDS] for L in range(N_RAW)])
        fig, axes = plt.subplots(1, 3, figsize=(12, 9.5), facecolor=surface, sharey=True)
        for ax, (title, M) in zip(axes, mats.items()):
            ax.set_facecolor(surface)
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
            for L in range(N_RAW):
                for j in range(len(TARGET_KINDS)):
                    v = M[L, j]
                    ax.text(j, L, "unstable" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=muted if np.isnan(v) else ("#ffffff" if v > 0.78 else ink))
            ax.set_xticks(range(len(TARGET_KINDS)), labels, fontsize=7.5, color=ink)
            ax.xaxis.tick_top()
            ax.set_title(title, color=ink, fontsize=10, loc="left", pad=22)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.tick_params(length=0)
        axes[0].set_yticks(range(0, N_RAW, 2), [str(L) for L in range(0, N_RAW, 2)], fontsize=8, color=muted)
        axes[0].set_ylabel("Layer ℓ (probe input)", color=muted)
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label("Unseen-task AUROC at the binarised target (≤ 0.5 shown as 0.5)", color=muted, fontsize=8)
        cb.ax.tick_params(colors=muted, labelsize=7)
        cb.outline.set_visible(False)
        fig.suptitle(f"B, H = {H}: predicting each constraint {H} steps ahead, probe vs baselines"
                     f"   (ℓ* = min(ℓ, 30))", color=ink, fontsize=11, x=0.01, ha="left")
        fig.savefig(RESULTS_DIR / f"b_map_baselines_H{H}.png", dpi=160, facecolor=surface,
                    bbox_inches="tight")
        plt.close(fig)
    print("[*] results/abc/b_baselines.csv, b_map_baselines_H{0,10,25}.png")


if __name__ == "__main__":
    main()
