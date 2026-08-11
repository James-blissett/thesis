"""
plot_auroc_by_layer.py

AUROC vs layer for the scheme-B sweep: reads every results/probe_all_layers/layerNN/
metrics.json written by run_all_layers.sh and draws one figure plus a tidy summary table.

WHAT THE FIGURE SHOWS, AND WHY IT SHOWS THE NULL. A bare AUROC-vs-layer curve invites
the reader to treat the peak as "the layer where failure is represented". Two things
stop that reading and both are drawn:

  * the within-task null band. Per-task success runs 40-100%, so a probe decoding only
    task identity scores well above 0.5. The null is the floor that matters, not 0.5,
    and it is layer-dependent -- so it is plotted per layer, not as one line.
  * marker fill. Filled = the 20-seed mean clears the within-task null at p < 0.05
    (aggregate_vs_within_task_null); hollow = it does not. Significance is never
    encoded by colour alone.

SELECTION EFFECT -- READ BEFORE QUOTING A PEAK. These are 33 dependent tests. The
best-scoring layer is a maximum over 33 draws and its AUROC is biased upward; the
per-layer p-values are uncorrected. The figure annotates the locked pilot layer (15)
because that one was chosen before seeing the sweep. Any other layer quoted from this
plot is a post-hoc selection and needs to say so.

The elapsed-time confound documented at the top of probe_late_window.py applies at
every layer unchanged: within a task the failures are still the long rollouts, so a
within-task-significant point licenses "failure-specific signal", not "a representation
of failure as such".

Usage:
    source env.sh
    python plot_auroc_by_layer.py
    python plot_auroc_by_layer.py --results-root results/probe_all_layers --dark
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path("results/probe_all_layers")
LOCKED_LAYER = 15

# Slots 1 and 2 of the validated categorical palette, in fixed order, plus the
# reference greys. The null is a reference, not a competing entity, so it wears grey.
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#dcdbd6",
    "series1": "#2a78d6",   # subset (primary)
    "series2": "#eb6834",   # all tasks (continuity)
    "null": "#8a8880",
}
DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#3a3a38",
    "series1": "#3987e5",
    "series2": "#d95926",
    "null": "#8a8880",
}


def collect(results_root: Path) -> list[dict]:
    """One row per layer that produced a metrics.json, sorted by layer."""
    rows = []
    for mp in sorted(results_root.glob("layer*/metrics.json")):
        with open(mp) as f:
            m = json.load(f)
        sub = m["non_degenerate_subset"]
        sweep = sub["split_sweep"]["stats"] or {}
        null = (sub["within_task_null_resampled_split"]["stats"] or {})
        test = sub["tests"].get("aggregate_vs_within_task_null") or {}
        all_sweep = (m["all_tasks"]["split_sweep"]["stats"] or {})
        rows.append({
            "layer": int(m["layer"]),
            "subset_sweep_mean": sweep.get("mean"),
            "subset_sweep_std": sweep.get("std"),
            "subset_locked": sub["locked_split_timestep_auroc"],
            "all_tasks_sweep_mean": all_sweep.get("mean"),
            "all_tasks_locked": m["all_tasks"]["locked_split_timestep_auroc"],
            "wt_null_mean": null.get("mean"),
            "wt_null_std": null.get("std"),
            "sd_above_wt_null": test.get("sd_above_null"),
            "p_wt": test.get("p_bootstrap_of_means"),
            "p_wt_reportable": test.get("p_bootstrap_reportable"),
            "significant_wt": (test.get("p_bootstrap_of_means") is not None
                               and test["p_bootstrap_of_means"] < 0.05),
            "runtime_seconds": m.get("runtime_seconds"),
            "source": str(mp),
        })
    return sorted(rows, key=lambda r: r["layer"])


def draw(rows: list[dict], out_png: Path, dark: bool) -> None:
    c = DARK if dark else LIGHT
    L = np.array([r["layer"] for r in rows])
    sub = np.array([r["subset_sweep_mean"] for r in rows], dtype=float)
    sub_sd = np.array([r["subset_sweep_std"] or 0.0 for r in rows], dtype=float)
    allt = np.array([r["all_tasks_sweep_mean"] for r in rows], dtype=float)
    nul = np.array([r["wt_null_mean"] if r["wt_null_mean"] is not None else np.nan
                    for r in rows], dtype=float)
    nul_sd = np.array([r["wt_null_std"] or 0.0 for r in rows], dtype=float)
    sig = np.array([r["significant_wt"] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    # Recessive frame: horizontal grid only, two spines.
    ax.grid(axis="y", color=c["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])

    # Chance, and the null that actually matters.
    ax.axhline(0.5, color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.55)
    ax.annotate("chance", xy=(L[-1], 0.5), xytext=(3, 2), textcoords="offset points",
                color=c["muted"], fontsize=8, va="bottom")
    ax.fill_between(L, nul - nul_sd, nul + nul_sd, color=c["null"], alpha=0.22,
                    linewidth=0, label="within-task null, ±1 SD")
    ax.plot(L, nul, color=c["null"], linewidth=1.5, alpha=0.9)

    # Continuity series: all tasks, degenerate ones included. Thinner, no markers.
    ax.plot(L, allt, color=c["series2"], linewidth=1.6, alpha=0.85,
            label="all tasks (incl. degenerate)")

    # Primary series: non-degenerate subset, 20-seed mean ±1 SD.
    ax.fill_between(L, sub - sub_sd, sub + sub_sd, color=c["series1"], alpha=0.18,
                    linewidth=0)
    ax.plot(L, sub, color=c["series1"], linewidth=2.0,
            label="non-degenerate subset, 20-seed mean ±1 SD")
    # Marker fill carries significance, so it is never colour-alone.
    ax.plot(L[sig], sub[sig], linestyle="none", marker="o", markersize=4.5,
            color=c["series1"], markeredgecolor=c["surface"], markeredgewidth=1.0,
            label="clears within-task null (p < 0.05)")
    ax.plot(L[~sig], sub[~sig], linestyle="none", marker="o", markersize=4.5,
            markerfacecolor=c["surface"], markeredgecolor=c["series1"],
            markeredgewidth=1.4, label="does not clear it")

    # The one layer that was not selected post-hoc.
    if LOCKED_LAYER in set(L.tolist()):
        ax.axvline(LOCKED_LAYER, color=c["muted"], linewidth=1.0, alpha=0.45,
                   linestyle=(0, (2, 3)))
        ax.annotate(f"locked pilot layer {LOCKED_LAYER}",
                    xy=(LOCKED_LAYER, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", rotation=90, color=c["muted"],
                    fontsize=8, va="bottom")

    # One direct label: the peak, flagged as a post-hoc maximum. The curve shape is
    # not known in advance, so the label goes in the top corner furthest from the peak
    # -- where a peaked curve is lowest -- with a leader line back to the point.
    if np.isfinite(sub).any():
        k = int(np.nanargmax(sub))
        left = L[k] > (L.min() + L.max()) / 2
        ax.annotate(f"peak {sub[k]:.3f} @ layer {L[k]}\n(post-hoc max)",
                    xy=(L[k], sub[k] + sub_sd[k]),
                    xytext=(0.02 if left else 0.98, 0.97), textcoords=ax.transAxes,
                    ha="left" if left else "right", va="top",
                    color=c["text"], fontsize=9, linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", color=c["muted"], linewidth=0.8,
                                    alpha=0.7, shrinkA=4, shrinkB=3))

    ax.set_xlabel("hidden-state layer (0 = embeddings)", color=c["muted"], fontsize=10)
    ax.set_ylabel("timestep AUROC", color=c["muted"], fontsize=10)
    ax.set_title("Scheme B late-window probe: AUROC by layer",
                 color=c["text"], fontsize=12, loc="left", pad=14)
    ax.set_xlim(L.min() - 0.5, L.max() + 1.5)
    lo = float(np.nanmin([np.nanmin(nul - nul_sd), np.nanmin(sub - sub_sd), 0.45]))
    ax.set_ylim(max(0.0, lo - 0.05), 1.02)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)
    ax.set_xticks(np.arange(L.min(), L.max() + 1, 2))

    # Legend below the axes: with an unknown curve shape, no in-axes corner is
    # reliably empty, and overlapping the null band is worse than the extra height.
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.105),
                     frameon=False, fontsize=8.5, ncol=3, columnspacing=1.6,
                     handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(c["muted"])

    fig.text(0.5, 0.015,
             "33 dependent tests, uncorrected p-values; the peak is a maximum over "
             "layers and is biased upward.\nWithin-task significance licenses "
             "'failure-specific signal', not 'a representation of failure as such'.",
             color=c["muted"], fontsize=7, va="bottom", ha="center", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(out_png, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--out", type=str, default=None,
                    help="PNG path; default <results-root>/auroc_by_layer.png")
    ap.add_argument("--dark", action="store_true", help="dark-surface version")
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = collect(root)
    if not rows:
        raise SystemExit(f"no layer*/metrics.json under {root} -- run run_all_layers.sh")

    expected = set(range(33))
    missing = sorted(expected - {r["layer"] for r in rows})
    if missing:
        print(f"[!] no metrics for layers {missing} -- plotting the {len(rows)} present",
              flush=True)

    out_png = Path(args.out) if args.out else root / (
        "auroc_by_layer_dark.png" if args.dark else "auroc_by_layer.png")
    draw(rows, out_png, args.dark)

    fields = [k for k in rows[0] if k != "source"]
    with open(root / "auroc_by_layer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(root / "auroc_by_layer.json", "w") as f:
        json.dump({"n_layers": len(rows), "missing_layers": missing, "rows": rows},
                  f, indent=2)

    print(f"\n{'layer':>5} {'subset mean':>12} {'wt null':>9} {'SD above':>9} {'p':>10}")
    for r in rows:
        sd = "n/a" if r["sd_above_wt_null"] is None else f"{r['sd_above_wt_null']:.1f}"
        print(f"{r['layer']:>5} {r['subset_sweep_mean']:>12.4f} "
              f"{(r['wt_null_mean'] or float('nan')):>9.4f} {sd:>9} "
              f"{str(r['p_wt_reportable']):>10}")
    best = max(rows, key=lambda r: r["subset_sweep_mean"])
    print(f"\npeak: layer {best['layer']} at {best['subset_sweep_mean']:.4f} "
          f"(post-hoc max over {len(rows)} layers -- biased upward, p uncorrected)")
    print(f"wrote {out_png}, auroc_by_layer.csv, auroc_by_layer.json")


if __name__ == "__main__":
    main()
