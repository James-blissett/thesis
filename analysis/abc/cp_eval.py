"""
cp_eval.py

Stage 6: conformal prediction (M4) and the final tables (handoff v10, section 4).

Detectors (per seed, scores from Stages 2-5):
    A        best single hidden-state probe (scores/a_seed{s}.npz, running sum)
    C        132-input constraint detector (scores/c_seed{s}.npz, s_full)
    B H=0/10/25  final detectors on the grid (scores/b_h{H}_seed{s}.npz)
    emb_temp[28] alone, no training: running maximum of its prepared z-score (Stage 2)

Band: SAFE's one-sided functional CP (Xu et al., FAIL-Detect, App. B, "adaptive
modulation"); SAFE's own code is not on this box (Stage 0), so it is implemented here.
  * Calibration set: the seed's successful eval_seen rollouts, over their kept rows.
    Split at random (CAL_SPLIT_SEED) into halves D1 and D2.
  * D1 gives mu(t) and the modulation xi(t) = std of the scores at t, over D1 rollouts
    still running at t.
  * D2 gives the nonconformity R_i = max over its kept rows of (s_i(t) - mu(t)) / xi(t);
    h(alpha) = the ceil((n2 + 1)(1 - alpha))-th smallest R_i (infinite if that rank
    exceeds n2: the band never alarms).
  * Band b(t) = mu(t) + h xi(t). Past T_END, the last t with >= 2 D1 rollouts still
    running, mu and xi are HELD at their T_END values (handoff default; SAFE's code is
    not available to check what it does).
  * Alarm at the first kept row with s(t) > b(t).
Evaluation on unseen rollouts, over their kept rows (successes to t_success - 1,
failures to 519):
  * balanced accuracy = (TPR on failures + TNR on successes) / 2
  * detection time = first alarm step / kept length (1 if never), mean over failures.
Side variant ("cut"): the same, evaluated only up to each task's M1 cut T_task (the
shortest unseen success in that task), detection time / T_task. It removes the length
advantage failures have over successes.

alpha: 15 levels, linspace(0.02, 0.9); the alpha = 0.1 table is computed separately
(0.1 is not on that grid).

Outputs (results/abc/):
    cp_curves.csv     detector x seed x alpha: balanced accuracy, TPR, TNR, detection time
    final_table.csv   per detector: M1 pooled / per-task / floor on eval_seen and unseen,
                      and M4 at alpha = 0.1 (full and cut), mean +- std over seeds
    m4_curves.png     balanced accuracy vs detection time
    score_vs_t.png    length check: mean per-step score against t on unseen rollouts

Usage (from the repo root; under a minute):
    source env.sh
    python analysis/abc/cp_eval.py
"""

from __future__ import annotations

import numpy as np

from data import RESULTS_DIR, SEEDS, T, kept_mask, load_index, load_split, m1, m1_cut
from prep import PRIMARY, apply_prep, fit_scale_orient, load_series, smooth
from setup_c import SETS, write_csv

SCORES = RESULTS_DIR / "scores"
ALPHAS = np.linspace(0.02, 0.9, 15)
ALPHA_TABLE = 0.1
CAL_SPLIT_SEED = 0
MIN_ALIVE = 2
COMPARE = ("emb_temp", 28)

DETECTORS = ["A", "C", "B H=0", "B H=10", "B H=25", "emb_temp[28] alone"]
COLOURS = {"A": "#2a78d6", "C": "#eb6834", "B H=0": "#1baf7a", "B H=10": "#eda100",
           "B H=25": "#e87ba4", "emb_temp[28] alone": "#008300"}


# --- Scores --------------------------------------------------------------------------
def load_scores(idx, kept, splits) -> dict:
    """{(detector, seed): (s (300, 520) score over time, x (300, 520) per-step score)}."""
    out = {}
    for s in SEEDS:
        z = np.load(SCORES / f"a_seed{s}.npz")
        out[("A", s)] = (z["s"], z["p"])
        z = np.load(SCORES / f"c_seed{s}.npz")
        out[("C", s)] = (z["s_full"], z["p_full"])
        for H in (0, 10, 25):
            z = np.load(SCORES / f"b_h{H}_seed{s}.npz")
            out[(f"B H={H}", s)] = (z["s"], z["p"])
    S, names = load_series(idx, [(b, l) for b, l in PRIMARY if b == COMPARE[0]])
    k = names.index(COMPARE)
    E = smooth(S[k:k + 1], names[k:k + 1], 0.0)
    for s in SEEDS:
        mu, sd, sign, _ = fit_scale_orient(E, idx, kept, splits[s]["pos"]["train"])
        x = apply_prep(E, mu, sd, sign)[0]
        xk = np.where(kept, x, -np.inf)
        run = np.maximum.accumulate(np.nan_to_num(xk, nan=-np.inf), axis=1)
        out[("emb_temp[28] alone", s)] = (np.where(np.isfinite(run), run, np.nan), x)
    return out


# --- Functional CP band --------------------------------------------------------------
def fit_band(s, kept, cal_pos, rng):
    """mu(t), xi(t), the D2 nonconformity scores, and T_END."""
    perm = rng.permutation(cal_pos)
    d1, d2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
    alive = kept[d1]
    n_alive = alive.sum(0)
    t_end = int(np.nonzero(n_alive >= MIN_ALIVE)[0].max())
    v = np.where(alive, s[d1], np.nan)
    mu = np.nanmean(v[:, : t_end + 1], 0)
    xi = np.nanstd(v[:, : t_end + 1], 0)
    first = int(np.nonzero(n_alive >= MIN_ALIVE)[0].min())
    mu_full = np.full(T, np.nan)
    xi_full = np.full(T, np.nan)
    mu_full[first: t_end + 1] = mu[first:]
    xi_full[first: t_end + 1] = xi[first:]
    mu_full[t_end + 1:] = mu[t_end]
    xi_full[t_end + 1:] = xi[t_end]
    pos_xi = xi_full[np.isfinite(xi_full) & (xi_full > 0)]
    xi_full = np.where(np.isfinite(xi_full), np.maximum(xi_full, 1e-6 * np.median(pos_xi)), np.nan)
    z = (s[d2] - mu_full) / xi_full
    R = np.nanmax(np.where(kept[d2], z, np.nan), 1)
    return mu_full, xi_full, np.sort(R), t_end, len(d1), len(d2)


def band_h(R, alpha) -> float:
    n = len(R)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(R[k - 1]) if k <= n else float("inf")


def evaluate_cp(s, kept, idx, pos, mu, xi, h, cut=None) -> dict:
    """Balanced accuracy, TPR, TNR and mean detection time on rollouts `pos`."""
    b = mu + h * xi
    rows = kept[pos].copy()
    denom = idx.kept_len[pos].astype(float)
    if cut is not None:
        rows &= np.arange(T)[None, :] < cut[:, None]
        denom = cut.astype(float)
    with np.errstate(invalid="ignore"):
        alarm = rows & (s[pos] > b[None, :])
    fired = alarm.any(1)
    first = np.where(fired, alarm.argmax(1), np.nan)
    y = idx.y[pos]
    tpr = fired[y == 1].mean()
    tnr = 1 - fired[y == 0].mean()
    det = np.where(fired, first / denom, 1.0)[y == 1].mean()
    return {"bal_acc": float((tpr + tnr) / 2), "tpr": float(tpr), "tnr": float(tnr),
            "det_time": float(det)}


# --- Figures -------------------------------------------------------------------------
def style(ax, ink, muted, grid, surface):
    ax.set_facecolor(surface)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(colors=muted, length=0)


def figures(curves, scores, idx, kept, splits) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=surface, sharey=True)
    for ax, variant, title in ((axes[0], "full", "Over each rollout's kept steps (protocol)"),
                               (axes[1], "cut", "Only up to the task's M1 cut (no length advantage)")):
        style(ax, ink, muted, grid, surface)
        for d in DETECTORS:
            ba = np.array([[curves[(d, s, a, variant)]["bal_acc"] for a in ALPHAS] for s in SEEDS]).mean(0)
            dt = np.array([[curves[(d, s, a, variant)]["det_time"] for a in ALPHAS] for s in SEEDS]).mean(0)
            ax.plot(dt, ba, color=COLOURS[d], linewidth=2, marker="o", markersize=4, label=d,
                    markeredgecolor=surface, markeredgewidth=1)
        ax.axhline(0.5, color=muted, linewidth=1, linestyle="--")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Detection time (first alarm ÷ length; 1 = never), failures", color=muted)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
    axes[0].set_ylabel("Balanced accuracy, unseen tasks", color=muted)
    axes[0].text(0.01, 0.505, "chance", color=muted, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(DETECTORS),
               labelcolor=ink, fontsize=9)
    fig.suptitle("M4: conformal alarms over α = 0.02–0.9 (mean of 3 seeds; top-left is best)",
                 color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS_DIR / "m4_curves.png", dpi=160, facecolor=surface)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), facecolor=surface, sharex=True)
    t = np.arange(T)
    for ax, d in zip(axes.ravel(), DETECTORS):
        style(ax, ink, muted, grid, surface)
        for lab, ysel, colour in (("failures", 1, COLOURS[d]), ("successes", 0, muted)):
            curves_t = []
            for s in SEEDS:
                up = splits[s]["pos"]["unseen"]
                sel = up[idx.y[up] == ysel]
                x = np.where(kept[sel], scores[(d, s)][1][sel], np.nan)
                n = np.sum(np.isfinite(x), 0)
                m = np.where(n >= 5, np.nanmean(np.where(np.isfinite(x), x, np.nan), 0), np.nan)
                curves_t.append(m)
            ax.plot(t, np.nanmean(curves_t, 0), color=colour, linewidth=2 if ysel else 1.5,
                    label=f"unseen {lab}")
        ax.set_title(d, color=ink, fontsize=10, loc="left")
        ax.set_ylabel("z-score" if "alone" in d else "P(failure) per step", color=muted, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Policy step t", color=muted)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle("Length check: mean per-step score against t (successes shown while ≥ 5 still running)",
                 color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "score_vs_t.png", dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> None:
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}
    scores = load_scores(idx, kept, splits)

    curves, rows, bands = {}, [], {}
    for d in DETECTORS:
        for s in SEEDS:
            sp = splits[s]
            es = sp["pos"]["eval_seen"]
            cal = es[idx.y[es] == 0]
            sc = scores[(d, s)][0]
            mu, xi, R, t_end, n1, n2 = fit_band(sc, kept, cal, np.random.default_rng(CAL_SPLIT_SEED + s))
            bands[(d, s)] = (t_end, n1, n2)
            up = sp["pos"]["unseen"]
            cut = m1_cut(idx, up)
            for a in list(ALPHAS) + [ALPHA_TABLE]:
                h = band_h(R, a)
                for variant, c in (("full", None), ("cut", cut)):
                    r = evaluate_cp(sc, kept, idx, up, mu, xi, h, c)
                    curves[(d, s, a, variant)] = r
                    if a in ALPHAS or a == ALPHA_TABLE:
                        rows.append({"detector": d, "seed": s, "alpha": float(a), "variant": variant,
                                     "h": h, "t_end": t_end, "n_d1": n1, "n_d2": n2, **r})
    write_csv(RESULTS_DIR / "cp_curves.csv", rows)

    final = []
    for d in DETECTORS:
        row = {"detector": d}
        for lab, key in SETS:
            ms = [m1(scores[(d, s)][0], kept, idx, splits[s]["pos"][key]) for s in SEEDS]
            for met in ("pooled", "per_task"):
                v = np.array([m[met] for m in ms])
                row[f"m1_{met}_{lab}_mean"] = float(v.mean())
                row[f"m1_{met}_{lab}_std"] = float(v.std(ddof=1))
            row[f"floor_pooled_{lab}"] = float(np.mean([m["floor"] for m in ms]))
            row[f"n_tasks_{lab}"] = ms[0]["n_tasks"]
        for variant in ("full", "cut"):
            for k in ("bal_acc", "det_time", "tpr", "tnr"):
                v = np.array([curves[(d, s, ALPHA_TABLE, variant)][k] for s in SEEDS])
                row[f"a01_{variant}_{k}_mean"] = float(v.mean())
                row[f"a01_{variant}_{k}_std"] = float(v.std(ddof=1))
        final.append(row)
    write_csv(RESULTS_DIR / "final_table.csv", final)
    figures(curves, scores, idx, kept, splits)

    # --- Report ----------------------------------------------------------------------
    print("=== calibration: eval_seen successes split D1/D2; T_END = last t with >= 2 D1 rollouts running ===")
    for s in SEEDS:
        t_end, n1, n2 = bands[(DETECTORS[0], s)]
        a_min = 1 - n2 / (n2 + 1)
        print(f"   seed {s}: D1 {n1}, D2 {n2}, T_END {t_end}; smallest alpha with a finite band "
              f"{a_min:.3f} (alpha = 0.02 never alarms)")
    print("   band past T_END: mu and xi held at their T_END values")

    print("\n=== M4 curves (mean of 3 seeds): balanced accuracy / detection time per alpha ===")
    for variant in ("full", "cut"):
        print(f"-- {variant}")
        print("   alpha      " + "  ".join(f"{a:5.3f}" for a in ALPHAS))
        for d in DETECTORS:
            ba = [np.mean([curves[(d, s, a, variant)]["bal_acc"] for s in SEEDS]) for a in ALPHAS]
            dt = [np.mean([curves[(d, s, a, variant)]["det_time"] for s in SEEDS]) for a in ALPHAS]
            print(f"   {d:18s} BA " + "  ".join(f"{v:5.3f}" for v in ba))
            print(f"   {'':18s} DT " + "  ".join(f"{v:5.3f}" for v in dt))

    print(f"\n=== alpha = {ALPHA_TABLE} (unseen; mean ± std over seeds) ===")
    for variant in ("full", "cut"):
        print(f"-- {variant}")
        for r in final:
            print(f"   {r['detector']:18s} bal.acc {r[f'a01_{variant}_bal_acc_mean']:.3f}±{r[f'a01_{variant}_bal_acc_std']:.3f}"
                  f"  det.time {r[f'a01_{variant}_det_time_mean']:.3f}±{r[f'a01_{variant}_det_time_std']:.3f}"
                  f"  TPR {r[f'a01_{variant}_tpr_mean']:.3f}  TNR {r[f'a01_{variant}_tnr_mean']:.3f}")

    print("\n=== final summary: M1 (mean ± std over seeds) ===")
    for r in final:
        print(f"   {r['detector']:18s} unseen pooled {r['m1_pooled_unseen_mean']:.3f}±{r['m1_pooled_unseen_std']:.3f}"
              f"  per-task {r['m1_per_task_unseen_mean']:.3f}±{r['m1_per_task_unseen_std']:.3f}"
              f"  floor {r['floor_pooled_unseen']:.3f} | seen pooled {r['m1_pooled_seen_mean']:.3f}"
              f"  per-task {r['m1_per_task_seen_mean']:.3f}  floor {r['floor_pooled_seen']:.3f}")
    print("[*] results/abc/cp_curves.csv, final_table.csv, m4_curves.png, score_vs_t.png")


if __name__ == "__main__":
    main()
