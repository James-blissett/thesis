"""
figures.py

Thesis figures F1-F11 (results_collection_handoff.md §5, §7). Each figure is drawn ONLY
from its CSV in figures/data/ (written here or by an earlier stage), so it can be
redrawn from that CSV alone; one PDF (vector) and one PNG per figure.

Style (all figures): 16 cm wide, 9 pt sans-serif at final size, no titles; fixed colours
A blue, C orange, B aqua (the dataviz skill's validated categorical slots 1-3),
baselines grey, chance and floor dashed black; every series also differs by line style
or marker; shaded band = +-1 std over seeds; no shorthand in labels.

    F1  example_failure_and_success           data/example_rollouts.csv (+ example_frames/)
    F2  alarm_tradeoff                        data/alarm_tradeoff.csv
    F3  corpus_kept_rollouts_and_action_repeat data/kept_rollouts_against_t.csv, data/action_repeat_before_after_success.csv
    F4  length_check                          data/length_check.csv
    F5  untrained_constraint_heatmap          data/untrained_constraint_heatmap.csv
    F6  per_probe_detector_a                  data/per_probe_outcome_auroc.csv
    F7  per_probe_detector_c                  data/per_probe_outcome_auroc.csv
    F8  per_probe_detector_b                  data/per_probe_outcome_auroc.csv
    F9  b_target_auroc_heatmap                data/b_target_auroc_heatmap.csv
    F10 per_probe_overlay                     data/per_probe_outcome_auroc.csv
    F11 capacity                              data/capacity.csv

Usage: source env.sh; python analysis/thesis_results/figures.py [data] [F1 F2 ...]
       ("data" first writes the CSVs this module owns: F2, F5, F11)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import REPO, SEEDS, read_csv, write_csv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = common.OUT_DIR / "figures"
DATA = FIG / "data"
CM = 1 / 2.54
W = 16 * CM
COL = {"A": "#2a78d6", "C": "#eb6834", "B": "#1baf7a"}
GREY, GREY2, INK = "#7a7a75", "#b0afa8", "#1f1f1e"
MARK = {"A": "o", "C": "s", "B": "^"}
LS = {"A": "-", "C": "--", "B": "-."}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555550", "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "savefig.dpi": 200, "lines.linewidth": 1.5})
HEAD = common.TEST_SETS["T0"]
PRO = common.TEST_SETS["T2"]
SEEN = common.TEST_SETS["eval_seen"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def chance(ax, y=0.5, label="chance"):
    ax.axhline(y, color="black", ls=(0, (4, 3)), lw=0.8, zorder=1)


# --- data writers ------------------------------------------------------------------------
def data_f2():
    """Balanced accuracy and detection time over the 15 alphas, per detector and seed."""
    keys = {"A_mean": ("A", "A, unweighted mean"), "C_mean": ("C", "C, unweighted mean"),
            "B_mean_h10": ("B", "B, unweighted mean (H = 10)"),
            "emb_best": ("baseline", "best untrained constraint"),
            "token_entropy": ("baseline", "token mean entropy"),
            "failurespot_neg": ("baseline", "FailureSpot (-s)")}
    rows = []
    for r in read_csv(common.OUT_DIR / "tables" / "data" / "detector_alarm_per_seed.csv"):
        if r["key"] in keys and r["on_alpha_grid"] == "True" and r["test_set"] in (HEAD, PRO):
            rows.append({"series": keys[r["key"]][1], "group": keys[r["key"]][0], "test set": r["test_set"],
                         "alpha": round(float(r["alpha"]), 4), "seed": r["seed"],
                         "balanced accuracy": round(float(r["bal_acc"]), 4),
                         "mean normalised detection time": round(float(r["det_time"]), 4)})
    write_csv(DATA / "alarm_tradeoff.csv", rows)


def data_f5():
    """Untrained per-task AUROC (running maximum, unseen original rollouts), mean over seeds."""
    sel = common.selections()["betas"]
    fam = {"emb_temp": "emb_temp", "xl_adj": "cross_layer", "xl_final": "cross_layer",
           "act_mag": "action", "act_dir": "action", "grip_flip": "action", "act_rep": "action"}
    rows = []
    for r in read_csv(REPO / "results/abc2/constraint_m1_summary.csv"):
        if (r["cond"] == common.POOL and r["variant"] == "primary" and r["constraint"] in fam
                and r["beta"] == sel[fam[r["constraint"]]] and r["test_set"] in ("T0", "eval_seen", "T2")):
            rows.append({"constraint": r["constraint"], "layer": int(r["layer"]) if int(r["layer"]) >= 0 else "",
                         "smoothing beta": r["beta"], "test set": common.TEST_SETS[r["test_set"]],
                         "per-task AUROC mean": round(float(r["m1_per_task_mean"]), 4),
                         "per-task AUROC std": round(float(r["m1_per_task_std"]), 4),
                         "sign per seed": r["signs_all_seeds"], "sign-unstable": r["unstable"] == "True",
                         "source": "results/abc2/constraint_m1_summary.csv"})
    write_csv(DATA / "untrained_constraint_heatmap.csv", rows)


def data_f11():
    rows = []
    for r in read_csv(REPO / "results/abc3/report_tables/depth.csv"):
        if r["model"] not in ("linear", "resmlp"):
            continue
        for ts, lab in (("train", "train"), ("eval_seen", "eval-seen (selected the epoch; optimistic)"), ("T0", HEAD)):
            rows.append({"detector": r["input"], "training set": common.TEST_SETS.get("", "") or
                         {"D0": "original training set", "D2": "full training set"}[r["pool"]],
                         "model": "linear" if r["model"] == "linear" else f"{r['depth']}-layer residual MLP",
                         "depth (hidden layers; 0 = linear)": int(r["depth"]), "evaluated on": lab,
                         "per-task AUROC mean": round(float(r[f"{ts}_m1_per_task_mean"]), 4),
                         "per-task AUROC std": round(float(r[f"{ts}_m1_per_task_std"]), 4),
                         "source": "results/abc3/report_tables/depth.csv"})
    write_csv(DATA / "capacity.csv", rows)


# --- figures -------------------------------------------------------------------------------
def F1():
    import json
    import matplotlib.image as mpimg
    rows = read_csv(DATA / "example_rollouts.csv")
    info = json.loads((DATA / "example_rollouts.json").read_text())
    if not rows:
        return
    fig = plt.figure(figsize=(W, 14 * CM))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1.8, 1, 1.8], hspace=0.55, wspace=0.05)
    for j, role in enumerate(("failure", "success")):
        ri = info["rollouts"][role]
        frs = list(ri["frames"].items())
        for k, (nm, fr) in enumerate(frs[:4]):
            ax = fig.add_subplot(gs[2 * j, k])
            ax.imshow(np.flipud(mpimg.imread(common.OUT_DIR / fr["file"])))
            ax.set_xticks([]), ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_xlabel(f"{nm}, step {fr['t']}", fontsize=7, labelpad=2)
        ax = fig.add_subplot(gs[2 * j + 1, :])
        xmax = int(ri["cut"] * 1.12)
        R = [r for r in rows if r["outcome"] == role and r["kept"] == "True" and int(r["t"]) <= xmax]
        t = np.array([int(r["t"]) for r in R])
        band = np.array([f(r["B mean band (alpha 0.1)"]) for r in R])
        ax.fill_between(t, 0, band, color=COL["B"], alpha=0.12, lw=0, label="B's conformal band (alpha 0.1)")
        ax.plot(t, band, color=COL["B"], lw=0.6, alpha=0.7)
        for key, lab, c, ls in (("B mean", "B, unweighted mean (H = 10)", COL["B"], "-"),
                                ("A mean", "A, unweighted mean", COL["A"], "--"),
                                ("C mean", "C, unweighted mean", COL["C"], ":")):
            sc = np.array([f(r[f"{key} score (running sum{', H = 10' if key == 'B mean' else ''})"]) for r in R])
            ax.plot(t, sc, color=c, ls=ls, lw=1.6 if key == "B mean" else 1.0, label=f"{lab} score")
        ax.axvline(ri["cut"], color="black", ls=(0, (4, 3)), lw=0.8, label="cut")
        if ri["B mean alarm step"] is not None:
            ax.axvline(ri["B mean alarm step"], color=COL["B"], lw=1.0, ls=(0, (1, 1)), label="B's alarm step")
        ax.set_ylabel("score (running sum\nof P(failure))")
        ax.set_xlabel("policy step t")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, None)
        ax.text(0.005, 1.03, f"{role.capitalize()}: {ri['rollout']} (task {ri['task']}), seed {info['seed']}",
                transform=ax.transAxes, fontsize=8, va="bottom")
        if j == 0:
            h, l_ = ax.get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.13)
    save(fig, "F1_example_failure_and_success")


def F2():
    rows = read_csv(DATA / "alarm_tradeoff.csv")
    if not rows:
        return
    series = list(dict.fromkeys(r["series"] for r in rows))
    style = {s: (COL.get(s[0], GREY) if s[0] in COL and s[1] == "," else [GREY, "#4d4d49", GREY2][i % 3],
                 MARK.get(s[0], ["D", "v", "x"][i % 3]) if s[1] == "," else ["D", "v", "x"][i % 3],
                 LS.get(s[0], ":") if s[1] == "," else [":", (0, (1, 1)), (0, (5, 1, 1, 1))][i % 3])
             for i, s in enumerate(series)}
    fig, axes = plt.subplots(1, 2, figsize=(W, 7 * CM), sharey=True)
    for ax, ts, pl in zip(axes, (HEAD, PRO), ("(a)", "(b)")):
        for s in series:
            R = [r for r in rows if r["series"] == s and r["test set"] == ts]
            al = sorted({float(r["alpha"]) for r in R})
            ba = [np.mean([float(r["balanced accuracy"]) for r in R if float(r["alpha"]) == a]) for a in al]
            dt = [np.mean([float(r["mean normalised detection time"]) for r in R if float(r["alpha"]) == a]) for a in al]
            c, m, ls = style[s]
            ax.plot(dt, ba, color=c, marker=m, ms=4, ls=ls, lw=1.2, label=s, mfc="white" if s[1] != "," else c)
        chance(ax)
        ax.set_xlim(-0.02, 1.02)
        ax.text(0.0, 1.02, f"{pl} {ts}", transform=ax.transAxes, fontsize=8, va="bottom")
    axes[0].set_ylabel("balanced accuracy")
    fig.supxlabel("mean normalised detection time over failures (first alarm step / cut; 1 = no alarm)", fontsize=9, y=0.06)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.24)
    save(fig, "F2_alarm_tradeoff")


def F3():
    kr = read_csv(DATA / "kept_rollouts_against_t.csv")
    ar = read_csv(DATA / "action_repeat_before_after_success.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 6 * CM), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(wspace=0.75)
    ax = axes[0]
    t = np.array([int(r["t"]) for r in kr])
    for cls, ls, lab in (("failures", "-", "failures"), ("successes", "--", "successes")):
        ax.plot(t, [int(r[f"original rollouts, {cls} still kept"]) for r in kr], color=INK, ls=ls,
                label=f"{lab}, all 300 original rollouts")
        tr = np.array([[int(r[f"training split seed {s}, {cls} still kept"]) for r in kr] for s in SEEDS])
        ax.plot(t, tr.mean(0), color=GREY, ls=ls, lw=1.0, label=f"{lab}, training split (mean of 3 seeds)")
    ax.set_xlabel("policy step t")
    ax.set_ylabel("rollouts still kept")
    ax.set_ylim(0, 175)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.25))
    ax.text(0, 1.02, "(a)", transform=ax.transAxes, fontsize=8)
    ax = axes[1]
    lab = [r["successful rollouts"] for r in ar]
    x = np.arange(len(lab))
    pre = [f(r["action-repeat rate before first success"]) for r in ar]
    post = [f(r["action-repeat rate from first success on"]) for r in ar]
    ax.barh(x - 0.2, pre, 0.38, color=GREY2, label="before first success", edgecolor="white")
    ax.barh(x + 0.2, post, 0.38, color=INK, label="from first success on", edgecolor="white", hatch="///")
    ax.set_yticks(x, [s.replace("LIBERO-PRO ", "LIBERO-PRO, ").replace(" perturbation", "") for s in lab], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("action-repeat rate,\nsuccessful rollouts")
    ax.set_xlim(0, 0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.3, 1.04))
    ax.text(0, 1.02, "(b)", transform=ax.transAxes, fontsize=8)
    save(fig, "F3_corpus_kept_rollouts_and_action_repeat")


def F4():
    rows = read_csv(DATA / "length_check.csv")
    labs = [("A, unweighted mean (running sum)", "A"), ("C, unweighted mean (running sum)", "C"),
            ("B, unweighted mean, H = 10 (running sum)", "B"), ("timestep only (t)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(W, 5.5 * CM))
    t = np.array([int(r["t"]) for r in rows])
    for ax, (lab, d) in zip(axes, labs):
        c = COL.get(d, GREY)
        for cls, ls in (("failures", "-"), ("successes", "--")):
            m = np.array([f(r[f"{lab}, {cls}, mean over seeds"]) for r in rows])
            ps = np.array([[f(r[f"{lab}, {cls}, seed {s}"]) for r in rows] for s in SEEDS])
            with np.errstate(invalid="ignore"):
                sd = np.nanstd(ps, 0, ddof=1)
            ax.plot(t, m, color=c, ls=ls, label=f"unseen {cls}")
            ax.fill_between(t, m - sd, m + sd, color=c, alpha=0.15, lw=0)
        ax.set_xlabel("policy step t")
        ax.text(0, 1.02, lab.replace(" (running sum)", ""), transform=ax.transAxes, fontsize=7)
    axes[0].set_ylabel("mean score (running sum)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=INK, ls="-", label="unseen failures"),
                        Line2D([], [], color=INK, ls="--", label="unseen successes (while at least 5 are kept)")],
               loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "F4_length_check")


def F5():
    rows = [r for r in read_csv(DATA / "untrained_constraint_heatmap.csv") if r["test set"] == HEAD]
    from matplotlib.colors import TwoSlopeNorm
    fams = ["emb_temp", "xl_adj", "xl_final"]
    M = np.full((3, 32), np.nan)
    U = np.zeros((3, 32), bool)
    for r in rows:
        if r["constraint"] in fams:
            i, L = fams.index(r["constraint"]), int(r["layer"])
            M[i, L] = float(r["per-task AUROC mean"])
            U[i, L] = r["sign-unstable"] == "True"
    act = ["act_mag", "act_dir", "grip_flip", "act_rep"]
    A = np.full((1, 4), np.nan)
    UA = np.zeros((1, 4), bool)
    for r in rows:
        if r["constraint"] in act:
            A[0, act.index(r["constraint"])] = float(r["per-task AUROC mean"])
            UA[0, act.index(r["constraint"])] = r["sign-unstable"] == "True"
    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=0.8)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#c2410c", "#f2f1ec", "#1c5cab"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 4.6 * CM), gridspec_kw={"width_ratios": [32, 5.5]})
    for ax, X, Ux, xt, yl in ((axes[0], M, U, list(range(0, 32, 2)), fams), (axes[1], A, UA, None, None)):
        im = ax.imshow(X, cmap=cmap, norm=norm, aspect="auto")
        for (i, j), u in np.ndenumerate(Ux):
            if u:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", lw=0, edgecolor=INK))
        if xt is not None:
            ax.set_xticks(xt)
            ax.set_xlabel("layer")
            ax.set_yticks(range(3), yl)
        else:
            ax.set_xticks(range(4), act, rotation=40, ha="right", fontsize=7)
            ax.set_yticks([])
            ax.set_xlabel("action constraints")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("per-task AUROC\n(unseen tasks, original rollouts)", fontsize=7)
    save(fig, "F5_untrained_constraint_heatmap")


def per_probe_ylim():
    rows = read_csv(DATA / "per_probe_outcome_auroc.csv")
    v = [float(r["per-task AUROC"]) for r in rows if r["test set"] in (HEAD, SEEN)
         and (r["detector"] in ("A", "C") or (r["detector"] == "B" and r["H"] == "10")
              or r["scoring"] == "running sum")]
    return rows, (max(0.0, np.floor(min(v) * 20) / 20 - 0.02), min(1.0, np.ceil(max(v) * 20) / 20 + 0.02))


def agg(rows, **cond):
    """{probe: (layer, mean, std)} over seeds for rows matching cond."""
    by = {}
    for r in rows:
        if all(r[k] == v for k, v in cond.items()):
            by.setdefault(r["probe"], (r["layer"], []))[1].append(float(r["per-task AUROC"]))
    return {k: (int(L) if L != "" else None, float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for k, (L, v) in by.items()}


def per_probe_panel(ax, rows, d, extra, ylim, H=None):
    c = COL[d]
    cond = {"detector": d, "series": "probe"}
    if H is not None:
        cond["H"] = str(H)
    if d == "B":
        kinds = ["act_rep", "act_mag", "emb_temp", "xl_final"]
        lsk = {"act_rep": "-", "act_mag": "--", "emb_temp": "-.", "xl_final": ":"}
        mk = {"act_rep": "o", "act_mag": "s", "emb_temp": "^", "xl_final": "D"}
        for kd in kinds:
            for ts, alpha_, lw in ((SEEN, 0.35, 0.8), (HEAD, 1.0, 1.3)):
                a = agg([r for r in rows if r["probe"].split(":")[1:2] == [kd]], **cond, **{"test set": ts})
                Mn, Sd = np.full(32, np.nan), np.full(32, np.nan)
                for k, (ll, mn, sd) in a.items():
                    Mn[ll], Sd[ll] = mn, sd
                x = np.arange(32)
                ax.plot(x, Mn, color=c, ls=lsk[kd], marker=mk[kd] if ts == HEAD else None, ms=3, lw=lw,
                        alpha=alpha_, label=f"target {kd}" if ts == HEAD else None, mfc="white")
                if ts == HEAD:
                    ax.fill_between(x, Mn - Sd, Mn + Sd, color=c, alpha=0.08, lw=0)
    else:
        for ts, alpha_, lw in ((SEEN, 0.35, 0.8), (HEAD, 1.0, 1.5)):
            a = agg(rows, **cond, **{"test set": ts})
            lay = sorted((v[0], k) for k, v in a.items() if v[0] is not None)
            x = np.array([L for L, _ in lay])
            mn = np.array([a[k][1] for _, k in lay])
            sd = np.array([a[k][2] for _, k in lay])
            ax.plot(x, mn, color=c, marker=MARK[d] if ts == HEAD else None, ms=3, lw=lw, alpha=alpha_,
                    label=f"probes, {'unseen tasks' if ts == HEAD else 'eval-seen'}")
            if ts == HEAD:
                ax.fill_between(x, mn - sd, mn + sd, color=c, alpha=0.15, lw=0)
                nl = [(k, v) for k, v in a.items() if v[0] is None]
                for k, (_, m_, s_) in nl:
                    xx = 34.5
                    ax.errorbar([xx], [m_], yerr=[s_], color=c, marker=MARK[d], ms=4, capsize=2, lw=1)
                    ax.text(xx, m_ + s_ + 0.01, {"action_tokens": "action-token\nprobe", "action": "action-constraint\nprobe"}[k],
                            ha="center", fontsize=6.5, va="bottom")
    for e in extra:
        e(ax)
    mrow = agg(rows, detector=d, series="unweighted mean (final detector)", **{"test set": HEAD},
               **({"H": str(H)} if H is not None else {}))
    if mrow:
        mv = list(mrow.values())[0][1]
        ax.axhline(mv, color=c, ls=(0, (6, 2, 1, 2)), lw=1.0)
        ax.text(31.5, mv + 0.005, "unweighted mean (final detector)", ha="right", va="bottom", fontsize=6.5)
    chance(ax)
    ax.set_ylim(*ylim)
    ax.set_xlim(-1, 36.5 if d in ("A", "C") else 32)
    xt = list(range(0, 32, 4)) + ([32] if d == "A" else [])
    ax.set_xticks(xt, [str(v) if v != 32 else "32\npost-norm" for v in xt])
    ax.set_xlabel("layer")
    ax.set_ylabel(f"per-task AUROC against outcome\n({HEAD})")


def F6():
    rows, ylim = per_probe_ylim()

    def shuffled(ax):
        a = agg(rows, detector="A", series="shuffled-label control (mean of 5 permutations)", **{"test set": HEAD})
        lay = sorted((v[0], k) for k, v in a.items() if v[0] is not None)
        ax.plot([L for L, _ in lay], [a[k][1] for _, k in lay], color=GREY, ls=":", lw=1.0, marker="x", ms=3,
                label="shuffled-label control, unseen tasks")
    fig, ax = plt.subplots(figsize=(W, 7 * CM))
    per_probe_panel(ax, rows, "A", [shuffled], ylim)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28))
    save(fig, "F6_per_probe_detector_a")


def F7():
    rows, ylim = per_probe_ylim()

    def untrained(ax):
        a = agg(rows, detector="untrained constraint", series="best of the layer's three (chosen on eval-seen)",
                **{"test set": HEAD})
        lay = sorted((v[0], k) for k, v in a.items())
        ax.plot([L for L, _ in lay], [a[k][1] for _, k in lay], color=COL["C"], ls=":", lw=1.2, marker=".", ms=3,
                label="untrained constraints of the layer (best of three, running sum), unseen tasks")
    fig, ax = plt.subplots(figsize=(W, 7 * CM))
    per_probe_panel(ax, rows, "C", [untrained], ylim)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    save(fig, "F7_per_probe_detector_c")


def F8():
    rows, ylim = per_probe_ylim()
    fig, ax = plt.subplots(figsize=(W, 7 * CM))
    per_probe_panel(ax, rows, "B", [], ylim, H=10)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.28))
    save(fig, "F8_per_probe_detector_b")


def F9():
    rows = [r for r in read_csv(DATA / "b_target_auroc_heatmap.csv") if r["test set"] == HEAD]
    kinds = ["act_rep", "act_mag", "emb_temp", "xl_final"]
    M = np.full((32, 4), np.nan)
    for r in rows:
        if r["row-level AUROC mean"] != "":
            M[int(r["layer"]), kinds.index(r["target kind"])] = float(r["row-level AUROC mean"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])
    cmap.set_bad("#efeee9")
    fig, ax = plt.subplots(figsize=(W, 6 * CM))
    im = ax.imshow(np.ma.masked_invalid(M.T), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(M.T):
        if not np.isfinite(v):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", lw=0, edgecolor=GREY))
    ax.set_yticks(range(4), ["act_rep", "act_mag", "emb_temp[layer]", "xl_final[layer]"])
    ax.set_xticks(range(0, 32, 2))
    ax.set_xlabel("layer of the probe's input (hatched: sign-unstable target, no probe)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"row-level AUROC against\nthe probe's own target, H = 10\n({HEAD})", fontsize=7)
    save(fig, "F9_b_target_auroc_heatmap")


def F10():
    rows, ylim = per_probe_ylim()
    fig, ax = plt.subplots(figsize=(W, 7 * CM))
    for d, series, cond in (("A", "probe", {}), ("C", "probe", {}), ("B", "per-layer mean of B's probes", {"H": "10"})):
        a = agg(rows, detector=d, series=series, **{"test set": HEAD}, **cond)
        lay = sorted((v[0], k) for k, v in a.items() if v[0] is not None and v[0] < 32)
        x = np.array([L for L, _ in lay])
        mn = np.array([a[k][1] for _, k in lay])
        sd = np.array([a[k][2] for _, k in lay])
        lab = {"A": "A, probe per layer", "C": "C, probe per layer", "B": "B, mean of the layer's probes (H = 10)"}[d]
        ax.plot(x, mn, color=COL[d], ls=LS[d], marker=MARK[d], ms=3, label=lab)
        ax.fill_between(x, mn - sd, mn + sd, color=COL[d], alpha=0.12, lw=0)
    chance(ax)
    ax.set_ylim(*ylim)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"per-task AUROC against outcome\n({HEAD})")
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=3)
    save(fig, "F10_per_probe_overlay")


def F11():
    rows = read_csv(DATA / "capacity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 6.5 * CM), sharey=True)
    ev_style = {"train": ("o", INK), "eval-seen (selected the epoch; optimistic)": ("s", GREY), HEAD: ("^", None)}
    for ax, d in zip(axes, ("A", "C")):
        for pool, ls in (("original training set", "-"), ("full training set", "--")):
            for ev_, (mk, c) in ev_style.items():
                R = sorted([r for r in rows if r["detector"] == d and r["training set"] == pool and r["evaluated on"] == ev_],
                           key=lambda r: int(r["depth (hidden layers; 0 = linear)"]))
                x = [int(r["depth (hidden layers; 0 = linear)"]) for r in R]
                m = np.array([float(r["per-task AUROC mean"]) for r in R])
                s = np.array([float(r["per-task AUROC std"]) for r in R])
                col = c or COL[d]
                ax.plot(x, m, color=col, ls=ls, marker=mk, ms=4, label=f"{ev_.split(' (')[0]}, {pool}")
                ax.fill_between(x, m - s, m + s, color=col, alpha=0.08, lw=0)
        chance(ax)
        ax.set_xticks([0, 2, 6, 12], ["linear", "2", "6", "12"])
        ax.set_xlabel("hidden layers (residual MLP)")
        ax.text(0, 1.02, f"Detector {d}", transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("per-task AUROC")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, marker="o", ls="none", label="train"),
         Line2D([], [], color=GREY, marker="s", ls="none", label="eval-seen (selected the epoch; optimistic)"),
         Line2D([], [], color=GREY, marker="^", ls="none", mfc="white", label=f"{HEAD} (detector colour)"),
         Line2D([], [], color=INK, ls="-", label="original training set"),
         Line2D([], [], color=INK, ls="--", label="full training set")]
    fig.legend(handles=h, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save(fig, "F11_capacity")


def main() -> None:
    args = sys.argv[1:] or ["data", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"]
    if "data" in args:
        data_f2()
        data_f5()
        data_f11()
    for a in args:
        if a.startswith("F"):
            try:
                globals()[a]()
                print(f"{a} ok")
            except Exception as e:  # report and carry on; a missing CSV is logged by the caller
                print(f"{a} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
