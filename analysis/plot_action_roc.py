"""
plot_action_roc.py

ROC curves for the two action-temporal constraints, act_mag and act_dir.

WHY THIS EXISTS SEPARATELY FROM plot_constraint_auroc.py. That figure plots AUROC --
one number per cell -- read from results/constraint_auroc.csv. A ROC curve needs the
underlying scores and labels, which the CSV does not carry, so this reads the
per-rollout series in constraints/*.npz and rebuilds the ranking from scratch. It
reuses constraint_auroc.py's window and floor constants so the curve it draws is the
same ranking the CSV scored; the printed AUC is cross-checked against that CSV on every
run and the script fails loudly if they disagree.

CURVES ARE DRAWN IN THEIR RAW ORIENTATION, NOT FOLDED. constraint_auroc.csv reports
max(auc, 1-auc) with a separate sign column, which is the right summary but the wrong
thing to draw: folding would silently mirror act_mag's curve and hide that it runs
BELOW the diagonal. act_mag is anti-correlated with failure -- less movement predicts
failure, the stuck-policy signature -- and a curve that dips under the diagonal is the
honest picture of that. Both AUCs are annotated raw, with the folded value beside them.

Failure is the positive class (label 1 = NOT success), matching probe_layer.py.

Usage:
    source env.sh
    python analysis/plot_action_roc.py
    python analysis/plot_action_roc.py --corpus success_final --dark
"""

from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from constraint_auroc import B_FRACTION, MIN_DEFINED
from plot_auroc_by_layer import palette, style_axes

CONSTRAINTS_DIR = Path("constraints")
INDEX_JSON = Path("corpus_v2_index.json")
AUROC_CSV = Path("results/constraint_auroc.csv")
OUT_DIR = Path("results")

SERIES = [("act_mag", "series1", "-"), ("act_dir", "series2", (0, (5, 2)))]
SCHEMES = [("A", "Scheme A — every timestep"), ("B", "Scheme B — t/T ≥ 0.7 only")]
MAX_CURVE_POINTS = 3000     # thin for file size; does not change the AUC


def load_series(corpus: str):
    """Per-rollout act_mag / act_dir plus the failure label. NaN kept for masking."""
    index = json.loads(INDEX_JSON.read_text())["rollouts"]
    rid_ok, labels, data = [], [], {n: [] for n, _, _ in SERIES}
    for ent in index:
        rid = ent["rollout_id"]
        p = CONSTRAINTS_DIR / f"{rid}.npz"
        if not p.exists():
            raise SystemExit(f"{p} missing -- run analysis/compute_constraints.py first")
        if corpus == "success_final":
            fail = int(not ent["success_final"])
        elif corpus == "success_ever_strict":
            if ent["success_ever"] and not ent["success_final"]:
                continue                      # drop the latched-then-lost rollouts
            fail = int(not ent["success_ever"])
        else:
            fail = int(not ent["success_ever"])
        z = np.load(p)
        for n, _, _ in SERIES:
            data[n].append(z[n])
        rid_ok.append(rid)
        labels.append(fail)
    return ({n: np.stack(v) for n, v in data.items()},
            np.array(labels, dtype=int), rid_ok)


def scores_for(mat: np.ndarray, labels: np.ndarray, t_lo: int | None):
    """Flatten to (score, label) pairs under one scheme, applying the same
    defined-timestep floor step 2 uses so the curve matches the reported AUROC."""
    w = mat if t_lo is None else mat[:, t_lo:]
    ok = np.isfinite(w)
    use = ok.sum(1) >= MIN_DEFINED
    rr, tt = np.nonzero(ok & use[:, None])
    return w[rr, tt], labels[rr], int(use.sum()), int((~use).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="success_ever")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    data, labels, rids = load_series(args.corpus)
    T = next(iter(data.values())).shape[1]
    t_lo = int(ceil(B_FRACTION * T))

    reported = {}
    if AUROC_CSV.exists():
        for r in csv.DictReader(open(AUROC_CSV)):
            if r["corpus"] == args.corpus and r["constraint"] in dict(
                    (n, 1) for n, _, _ in SERIES):
                reported[(r["constraint"], r["scheme"])] = (
                    float(r["auroc_raw"]), float(r["auroc"]), r["sign"])

    c = palette(args.dark)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 11.0))
    fig.patch.set_facecolor(c["surface"])

    for ax, (scheme, title) in zip(axes, SCHEMES):
        style_axes(ax, c)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8, alpha=0.9)
        ax.set_title(title, color=c["text"], fontsize=10.5, loc="left", pad=8)
        ax.plot([0, 1], [0, 1], color=c["muted"], linewidth=1.0,
                linestyle=(0, (4, 3)), alpha=0.55, zorder=1)

        for name, slot, dash in SERIES:
            s, y, n_use, n_drop = scores_for(
                data[name], labels, None if scheme == "A" else t_lo)
            auc = float(roc_auc_score(y, s))
            fpr, tpr, _ = roc_curve(y, s)
            if fpr.size > MAX_CURVE_POINTS:                # thin for drawing only
                k = np.linspace(0, fpr.size - 1, MAX_CURVE_POINTS).astype(int)
                fpr, tpr = fpr[k], tpr[k]

            key = (name, scheme)
            if key in reported and abs(auc - reported[key][0]) > 1e-6:
                raise SystemExit(
                    f"ASSERTION FAILED: {name}/{scheme} recomputed AUC {auc:.6f} != "
                    f"{reported[key][0]:.6f} in {AUROC_CSV}")
            folded = max(auc, 1 - auc)
            lbl = (f"{name}   AUC {auc:.3f}"
                   f"{'  (below chance; folded ' + format(folded, '.3f') + ')' if auc < 0.5 else ''}")
            ax.plot(fpr, tpr, color=c[slot], linewidth=2.0, linestyle=dash,
                    label=lbl, zorder=3)
            print(f"  {name:<9} {scheme:<2} AUC {auc:.4f}  folded {folded:.4f}  "
                  f"n_rollouts {n_use} (dropped {n_drop})  n_points {len(s)}")

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("false positive rate", color=c["muted"], fontsize=10)
        ax.set_ylabel("true positive rate", color=c["muted"], fontsize=10)
        leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5,
                        handlelength=2.2)
        for t in leg.get_texts():
            t.set_color(c["muted"])

    fig.suptitle(f"Action-constraint ROC  ·  corpus {args.corpus}, "
                 f"n = {len(rids)} rollouts",
                 color=c["text"], fontsize=12.5, x=0.02, ha="left", y=0.985)
    fig.text(0.5, 0.012,
             "Positive class = failure. Curves are RAW, not folded: act_mag runs below "
             "the diagonal because\nless movement predicts failure. Each point is one "
             "timestep, with its rollout's outcome broadcast.",
             color=c["muted"], fontsize=7.5, va="bottom", ha="center", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.972))

    out = Path(args.out_dir) / (f"action_roc_{args.corpus}"
                                + ("_dark" if args.dark else "") + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
