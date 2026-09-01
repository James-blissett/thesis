"""
plot_constraint_auroc.py

Draws the step-2 constraint sweep: reads results/constraint_auroc.csv and writes two
figures plus a tidy summary. Kept separate from the script that computes the AUROCs, so
the figures can be redrawn without repeating the permutation nulls -- the same split as
run_all_layers.sh / plot_auroc_by_layer.py.

WHY THIS IS NOT plot_auroc_by_layer.py. That script reads results/probe_all_layers/
layerNN/metrics.json, one directory per layer, and its whole row schema is specific to
the *trained* probe: split sweeps, 20-seed means, bootstrap p-values. The constraint
sweep trains nothing -- the raw series are ranked directly -- so there is no split, no
seed variance and no error bar. It also needs several constraint families overlaid on
one layer axis rather than a single curve. The palette and the recessive frame are
imported from it; nothing else was reusable.

INPUT CONTRACT -- results/constraint_auroc.csv, one row per (constraint, layer, scheme,
corpus), with the columns the brief fixes:

    constraint  layer  scheme  corpus  n_rollouts  auroc  sign
    null_global_mean  null_global_p95  null_task_mean  null_task_p95

`layer` is -1 for the series that have no layer axis. `auroc` is already folded as
max(auc, 1-auc), so every value is >= 0.5 and `sign` ('+' or '-') carries the direction.
`scheme` is one of A, B, rollout_max, rollout_mean; `corpus` one of success_ever,
success_final.

WHAT THE FIGURES SHOW, AND WHY THEY SHOW THE NULL. Reading a peak off a bare AUROC
curve is the failure mode this project keeps guarding against. Both figures draw the
within-task null, because per-task success runs 17-80% here and anything tracking task
identity alone scores well above 0.5. The null is the floor that matters, not 0.5, and
it is per-layer, so it is drawn per-layer. Marker fill encodes clearing that floor, so
significance is never carried by colour alone.

SELECTION EFFECT -- READ BEFORE QUOTING A PEAK. The sweep is ~130 series x 4 schemes of
dependent tests with uncorrected nulls. Any best-scoring cell is a maximum over that
grid and is biased upward. Unlike the probe sweep there is no pre-registered layer here,
so *every* peak in these figures is a post-hoc selection and has to be reported as one.

The elapsed-time confound documented at the top of probe_late_window.py applies
unchanged: within a task the failures are still the long rollouts. That is why the `t`
and `t_norm` baselines are drawn alongside -- a constraint that does not beat a clock is
measuring the clock. Both baselines are degenerate at rollout level (T == 520 for all
300 rollouts, so they are constant per rollout) and are labelled as such.

Usage:
    source env.sh
    python analysis/plot_constraint_auroc.py
    python analysis/plot_constraint_auroc.py --corpus success_final --dark
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette and recessive frame are shared with the probe figure so the two never drift
# apart. Flat sibling import, matching how control_diagnostic.py pulls from probe_layer.
from plot_auroc_by_layer import palette, style_axes

RESULTS_CSV = Path("results/constraint_auroc.csv")
OUT_DIR = Path("results")

# Facet order and display names. Step 2 must emit exactly these `scheme` values.
SCHEMES = [
    # (value emitted by step 2, title for the wide facet, title for the narrow facet)
    ("A", "Scheme A — all timesteps", "Scheme A — all t"),
    ("B", "Scheme B — t/T ≥ 0.7", "Scheme B — t/T ≥ 0.7"),
    ("rollout_max", "Rollout level — max over the B window", "Rollout — max"),
    ("rollout_mean", "Rollout level — mean over the B window", "Rollout — mean"),
]

# Categorical slots 1-4 of the validated order, assigned to entities in fixed order and
# never cycled. Four overlaid lines sit on the *adjacent* pairlist (lines), which this
# order passes in both modes; identity is additionally carried by line style and a
# direct end-label, so it is never colour-alone. Slots 3 and 4 fall below 3:1 contrast
# on the light surface, which is what obliges those direct labels and the CSV table.
LAYERED = [
    ("emb_temp", "series1", "-"),
    ("xl_adj", "series2", (0, (5, 2))),
    ("xl_final", "series3", (0, (1.5, 1.5))),
    ("xl_final_hN", "series4", (0, (6, 2, 1.5, 2))),
]

# act_rep is derived in step 2 (1[act_mag <= 1e-6]); it has no layer axis.
SCALAR = ["act_mag", "act_dir", "act_rep", "grip_flip", "xl_spread"]
BASELINES = ["t", "t_norm"]

CAPTION = (
    "AUROC is folded as max(auc, 1-auc); marker shape carries the sign. ~130 series "
    "x 4 schemes of dependent tests, uncorrected nulls — every peak is a post-hoc\n"
    "maximum and is biased upward. A constraint that does not clear the t / t_norm "
    "baselines is measuring elapsed time, not failure."
)


def _f(v) -> float:
    """CSV cell -> float, with the empty string and the NaN spellings as NaN."""
    if v is None or str(v).strip() in ("", "nan", "NaN", "None"):
        return float("nan")
    return float(v)


def load(path: Path, corpus: str) -> list[dict]:
    """Rows for one corpus. Fails loudly rather than drawing an empty figure."""
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the step-2 AUROC sweep first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")

    seen = sorted({r["corpus"] for r in rows})
    kept = [r for r in rows if r["corpus"] == corpus]
    if not kept:
        raise SystemExit(f"no rows for corpus {corpus!r}; the file has {seen}")

    for r in kept:
        r["layer"] = int(r["layer"])
        r["n_rollouts"] = int(r["n_rollouts"])
        for k in ("auroc", "null_global_mean", "null_global_p95",
                  "null_task_mean", "null_task_p95"):
            r[k] = _f(r[k])
    return kept


def index(rows: list[dict]) -> dict:
    """(scheme, constraint) -> rows sorted by layer."""
    out = defaultdict(list)
    for r in rows:
        out[(r["scheme"], r["constraint"])].append(r)
    for k in out:
        out[k].sort(key=lambda r: r["layer"])
    return out


def place_end_labels(ax, items: list, c: dict, x_col: float,
                     min_gap_frac: float = 0.05) -> None:
    """Direct end-labels, pushed apart so they cannot overplot each other.

    Endpoints of neighbouring families land at almost the same AUROC often enough that
    unadjusted labels collide. Labels are stacked in a fixed column at x_col with a
    minimum vertical gap, and any label displaced from its line gets a hairline leader
    back to the endpoint. Text wears the muted ink token -- the marker it points at
    carries the hue -- so identity is never colour-alone.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda z: z[1])
    ys = [z[1] for z in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    over = ys[-1] - (hi - gap * 0.5)
    if over > 0:
        ys = [y - over for y in ys]

    for (x0, y0, name), y in zip(items, ys):
        displaced = abs(y - y0) > gap * 0.3
        ax.annotate(
            name, xy=(x0, y0), xytext=(x_col, y), textcoords="data",
            color=c["muted"], fontsize=7.5, va="center", ha="left", zorder=5,
            arrowprops=dict(arrowstyle="-", color=c["muted"], linewidth=0.7,
                            alpha=0.55, shrinkA=2, shrinkB=3) if displaced else None,
        )


def draw_null(ax, c: dict, L, task_mean, task_p95, glob_p95, label: bool) -> None:
    """The two nulls, both in reference grey -- they are a floor, not competing series.

    The within-task band is the bar that matters; the global null is drawn thinner
    because shuffling across the corpus also destroys the task->outcome correlation and
    therefore sits lower than the honest floor.
    """
    ax.fill_between(L, task_mean, task_p95, color=c["null"], alpha=0.22, linewidth=0,
                    label="within-task null, mean→p95" if label else None)
    ax.plot(L, task_p95, color=c["null"], linewidth=1.4, alpha=0.95,
            label="within-task null p95" if label else None)
    ax.plot(L, glob_p95, color=c["null"], linewidth=1.0, alpha=0.55,
            linestyle=(0, (1, 2)), label="global null p95" if label else None)


def draw_by_layer(idx: dict, corpus: str, n_roll: int, out_png: Path, dark: bool) -> None:
    c = palette(dark)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(c["surface"])

    for ax, (scheme, title, _short) in zip(axes.ravel(), SCHEMES):
        style_axes(ax, c)
        ax.set_title(title, color=c["text"], fontsize=10, loc="left", pad=8)
        # Set before any label placement: de-collision needs the final y range.
        ax.set_ylim(0.44, 1.02)
        ax.set_xlim(-1.0, 36.5)

        # Chance, and then the null that actually matters.
        ax.axhline(0.5, color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)),
                   alpha=0.55)

        anchor = idx.get((scheme, "emb_temp")) or idx.get((scheme, "xl_adj"))
        first = ax is axes.ravel()[0]
        end_labels: list = []
        if anchor:
            L = np.array([r["layer"] for r in anchor], dtype=float)
            draw_null(ax, c,
                      L,
                      np.array([r["null_task_mean"] for r in anchor]),
                      np.array([r["null_task_p95"] for r in anchor]),
                      np.array([r["null_global_p95"] for r in anchor]),
                      label=first)

        for name, slot, dash in LAYERED:
            rows = idx.get((scheme, name))
            if not rows:
                continue
            L = np.array([r["layer"] for r in rows], dtype=float)
            a = np.array([r["auroc"] for r in rows], dtype=float)
            p95 = np.array([r["null_task_p95"] for r in rows], dtype=float)
            neg = np.array([r["sign"].strip() == "-" for r in rows], dtype=bool)
            clears = a > p95

            ax.plot(L, a, color=c[slot], linewidth=1.9, linestyle=dash,
                    label=name if first else None, zorder=3)

            # Marker fill = clears the within-task null; marker shape = sign. Two
            # binary encodings, neither of them colour.
            for is_neg, marker in ((False, "o"), (True, "v")):
                for ok in (True, False):
                    m = (neg == is_neg) & (clears == ok)
                    if not m.any():
                        continue
                    ax.plot(L[m], a[m], linestyle="none", marker=marker, markersize=5.0,
                            color=c[slot],
                            markerfacecolor=c[slot] if ok else c["surface"],
                            markeredgecolor=c[slot] if not ok else c["surface"],
                            markeredgewidth=1.3 if not ok else 1.0, zorder=4)

            # Selective direct label at the endpoint, collected and de-collided below.
            if np.isfinite(a).any():
                k = int(np.where(np.isfinite(a))[0][-1])
                end_labels.append((L[k], a[k], name))

        place_end_labels(ax, end_labels, c, x_col=32.8)

    for ax in axes[-1]:
        ax.set_xlabel("layer (0 = embeddings)", color=c["muted"], fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC, max(auc, 1-auc)", color=c["muted"], fontsize=10)
    axes[0, 0].set_xticks(np.arange(0, 33, 4))

    fig.suptitle(f"Consistency constraints, AUROC by layer  ·  corpus "
                 f"{corpus}, n = {n_roll} rollouts",
                 color=c["text"], fontsize=12.5, x=0.012, ha="left", y=0.985)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.088),
                     frameon=False, fontsize=8.5, ncol=4, columnspacing=1.6,
                     handlelength=2.2)
    for t in leg.get_texts():
        t.set_color(c["muted"])

    fig.text(0.5, 0.012, CAPTION, color=c["muted"], fontsize=7, va="bottom",
             ha="center", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.155, 1, 0.965))
    fig.savefig(out_png, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def draw_scalar(idx: dict, corpus: str, n_roll: int, out_png: Path, dark: bool) -> None:
    """The series with no layer axis, plus the two clock baselines.

    Faceted by scheme like the layer figure, one series per panel -- so no legend box
    is needed here; the panel title names it.
    """
    c = palette(dark)
    names = SCALAR + BASELINES
    y = np.arange(len(names))[::-1]

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.5), sharey=True, sharex=True)
    fig.patch.set_facecolor(c["surface"])

    for ax, (scheme, _title, short) in zip(axes, SCHEMES):
        style_axes(ax, c)
        ax.set_title(short, color=c["text"], fontsize=9.5, loc="left", pad=8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8, alpha=0.9)
        ax.axvline(0.5, color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)),
                   alpha=0.55)

        for yi, name in zip(y, names):
            rows = idx.get((scheme, name))
            if not rows:
                continue
            r = rows[0]
            # The within-task floor for this cell, as a grey tick behind the point.
            if np.isfinite(r["null_task_p95"]):
                ax.plot([r["null_task_mean"], r["null_task_p95"]], [yi, yi],
                        color=c["null"], linewidth=3.0, alpha=0.35, solid_capstyle="butt",
                        zorder=2)
            clears = r["auroc"] > r["null_task_p95"]
            marker = "v" if r["sign"].strip() == "-" else "o"
            ax.plot([r["auroc"]], [yi], linestyle="none", marker=marker, markersize=6.0,
                    color=c["series1"],
                    markerfacecolor=c["series1"] if clears else c["surface"],
                    markeredgecolor=c["series1"], markeredgewidth=1.3, zorder=4)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [n + ("  (baseline)" if n in BASELINES else "") for n in names])
    axes[0].set_xlim(0.44, 1.02)
    for ax in axes:
        ax.set_xlabel("AUROC", color=c["muted"], fontsize=9.5)

    fig.suptitle(f"Constraints without a layer axis, and the clock baselines  ·  "
                 f"corpus {corpus}, n = {n_roll}",
                 color=c["text"], fontsize=12, x=0.008, ha="left", y=0.98)
    fig.text(0.5, 0.015,
             "Filled = clears the within-task null p95 (grey bar = null mean→p95). "
             "Marker shape carries the sign. t and t_norm are constant per rollout "
             "(T = 520 for all 300),\nso both are degenerate at rollout level "
             "and their per-timestep Scheme-A value must be exactly 0.5.",
             color=c["muted"], fontsize=7, va="bottom", ha="center", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    fig.savefig(out_png, dpi=200, facecolor=c["surface"])
    plt.close(fig)


def write_summary(rows: list[dict], out_csv: Path) -> list[dict]:
    """Best cell per (scheme, constraint family), with the margin over the honest floor.

    Sorted by that margin, because 'how far clear of the within-task null' is the
    question, not 'what is the biggest AUROC'.
    """
    best: dict = {}
    for r in rows:
        k = (r["scheme"], r["constraint"])
        margin = r["auroc"] - r["null_task_p95"]
        if k not in best or margin > best[k]["margin_over_task_p95"]:
            best[k] = {
                "scheme": r["scheme"], "constraint": r["constraint"],
                "layer": r["layer"], "auroc": r["auroc"], "sign": r["sign"],
                "null_task_p95": r["null_task_p95"],
                "margin_over_task_p95": margin,
                "clears_task_null": r["auroc"] > r["null_task_p95"],
                "n_rollouts": r["n_rollouts"],
            }
    out = sorted(best.values(),
                 key=lambda d: (-d["margin_over_task_p95"]
                                if np.isfinite(d["margin_over_task_p95"]) else 0.0))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", type=str, default=str(RESULTS_CSV))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--corpus", type=str, default="success_ever",
                    help="success_ever (primary) or success_final")
    ap.add_argument("--dark", action="store_true", help="dark-surface version")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load(Path(args.results_csv), args.corpus)
    idx = index(rows)
    n_roll = max(r["n_rollouts"] for r in rows)
    suffix = f"_{args.corpus}" + ("_dark" if args.dark else "")

    f1 = out_dir / f"constraint_auroc_by_layer{suffix}.png"
    f2 = out_dir / f"constraint_auroc_scalar{suffix}.png"
    draw_by_layer(idx, args.corpus, n_roll, f1, args.dark)
    draw_scalar(idx, args.corpus, n_roll, f2, args.dark)
    summary = write_summary(rows, out_dir / f"constraint_auroc_summary{suffix}.csv")

    print(f"corpus {args.corpus}, n = {n_roll}, {len(rows)} rows\n")
    print(f"{'scheme':<14}{'constraint':<16}{'layer':>6}{'auroc':>8}{'sign':>6}"
          f"{'wt p95':>9}{'margin':>9}")
    for d in summary[:12]:
        print(f"{d['scheme']:<14}{d['constraint']:<16}{d['layer']:>6}"
              f"{d['auroc']:>8.4f}{d['sign']:>6}{d['null_task_p95']:>9.4f}"
              f"{d['margin_over_task_p95']:>+9.4f}")
    n_clear = sum(1 for d in summary if d["clears_task_null"])
    print(f"\n{n_clear}/{len(summary)} (scheme, constraint) cells clear the within-task "
          f"null p95 -- uncorrected, and each is a max over layers")
    print(f"wrote {f1.name}, {f2.name}, constraint_auroc_summary{suffix}.csv")


if __name__ == "__main__":
    main()
