"""
lf_combiner.py

B with a label-free combiner (b_label_free_handoff.md, version 1).

Replaces B's trained final detector (a logistic regression on outcome) with a fixed
rule that has no fitted parameters, on the same grid probe outputs Stage 5 used:

    LF-all  unweighted mean of every grid probe's probability at one H (124 probes)
    LF-emb  unweighted mean of the emb_temp[l] probes only (30)

Score over time: running sum of the per-row mean, as for every trained model. Both
variants at H = 0, 10, 25; nothing is chosen between them.

Probe outputs: reused from Stage 5 (/data/tmp/abc_scores/b_probes_h{H}_seed{s}.npz;
eval_seen and unseen rows there are the refit-on-all-train probes). Before scoring,
each probe's own-target AUROC on eval_seen and unseen is recomputed from those outputs
and checked against results/abc/b_grid.csv at the grid's c (tolerance 0.005); any
miss stops the run.

Evaluation (as Stages 5 and 6, with their code): M1 pooled / per-task / floor on
eval_seen and unseen; conformal band calibrated on successful eval_seen rollouts
(cp_eval.fit_band, same calibration split seed), alpha = 0.1, cut variant.
Trained B is read from results/abc/b_final.csv and cp_curves.csv, not recomputed.

Outputs (results/b_label_free/):
    lf_final.csv          (variant, H): M1 mean, std, per seed; floor; alpha = 0.1 cut alarms
    lf_vs_trained.csv     per-seed unseen M1-per-task, LF minus trained B, per H
    score_vs_t.png        mean per-step score against t on unseen failures
    scores/lf_{variant}_h{H}_seed{s}.npz

Usage (from the repo root; about a minute, CPU only):
    source env.sh
    python analysis/b_label_free/lf_combiner.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abc"))

from cp_eval import ALPHA_TABLE, CAL_SPLIT_SEED, band_h, evaluate_cp, fit_band  # noqa: E402
from data import RESULTS_DIR, SEEDS, T, kept_mask, load_index, load_split, m1_cut  # noqa: E402
from setup_b import (HS, PROBE_DIR, lookahead, prepared_targets, target_auroc,  # noqa: E402
                     target_name, two_means_threshold)
from setup_c import SETS, evaluate, running_sum, summary_row, write_csv  # noqa: E402

OUT = Path("results/b_label_free")
SCORES_DIR = OUT / "scores"
VARIANTS = {"LF-all": lambda kd: True, "LF-emb": lambda kd: kd == "emb_temp"}
SLUG = {"LF-all": "all", "LF-emb": "emb"}
CHECK_TOL = 0.005
ALARM_KEYS = ("bal_acc", "tpr", "tnr", "det_time")
COLOURS = {"LF-all": "#2a78d6", "LF-emb": "#eb6834"}


def parse(name: str) -> tuple[int, str, int]:
    """'L07:emb_temp:H10' -> (7, 'emb_temp', 10)."""
    L, kd, H = name.split(":")
    return int(L[1:]), kd, int(H[1:])


def load_probes() -> dict:
    """{(H, seed): (names, (k, 300, 520) float32 probabilities)}."""
    out = {}
    for H in HS:
        for s in SEEDS:
            z = np.load(PROBE_DIR / f"b_probes_h{H}_seed{s}.npz")
            out[(H, s)] = ([str(n) for n in z["names"]], z["p"].astype(np.float32))
    return out


# --- Sanity check: reused outputs reproduce Stage 5's own-target AUROC -------------------
def sanity_check(probes, idx, kept, splits) -> list[dict]:
    meta = json.loads((RESULTS_DIR / "b_meta.json").read_text())
    c_grid = float(meta["grid_c"])
    chosen = meta["beta"]
    ref = {}
    with (RESULTS_DIR / "b_grid.csv").open() as f:
        for r in csv.DictReader(f):
            if r["chosen_c"] == "True":
                for s in SEEDS:
                    for lab, _ in SETS:
                        ref[(int(r["layer"]), r["target"], int(r["H"]), s, lab)] = float(r[f"auroc_{lab}_s{s}"])
    targets = prepared_targets(idx, kept, splits, chosen)
    rows = []
    for (H, s), (names, P) in probes.items():
        sp = splits[s]
        tr = sp["pos"]["train"]
        rr = np.nonzero(kept[tr])
        r_tr, t_tr = tr[rr[0]], rr[1]
        for j, nm in enumerate(names):
            L, kd, h = parse(nm)
            assert h == H
            y = lookahead(targets[(s, target_name(kd, L))], H)
            v = y[r_tr, t_tr]
            thr = two_means_threshold(v[np.isfinite(v)])     # refit-on-all-train threshold
            for lab, key in SETS:
                a = target_auroc(P[j], y, thr, kept, sp["pos"][key])
                b = ref[(L, kd, H, s, lab)]
                rows.append({"layer": L, "target": kd, "H": H, "seed": s, "set": lab,
                             "auroc_reused": a, "auroc_b_grid": b, "abs_diff": abs(a - b)})
    print(f"[check] grid c {c_grid:g}; {len(rows)} (probe, H, seed, set) AUROCs recomputed")
    return rows


# --- Scores and evaluation ---------------------------------------------------------------
def lf_scores(probes, kept) -> dict:
    """{(variant, H, seed): (names used, per-row mean p (300, 520), running sum)}."""
    out = {}
    for (H, s), (names, P) in probes.items():
        for v, keep in VARIANTS.items():
            sel = [j for j, nm in enumerate(names) if keep(parse(nm)[1])]
            p = P[sel].mean(0)
            out[(v, H, s)] = ([names[j] for j in sel], p, running_sum(p, kept))
    return out


def alarms(score, idx, kept, sp, s) -> dict:
    """alpha = 0.1, cut variant, on unseen (cp_eval's band and evaluation)."""
    es = sp["pos"]["eval_seen"]
    cal = es[idx.y[es] == 0]
    mu, xi, R, _, _, _ = fit_band(score, kept, cal, np.random.default_rng(CAL_SPLIT_SEED + s))
    up = sp["pos"]["unseen"]
    return evaluate_cp(score, kept, idx, up, mu, xi, band_h(R, ALPHA_TABLE), m1_cut(idx, up))


def table_row(head, metrics, alarm_list) -> dict:
    d = summary_row(head, metrics)
    for lab, _ in SETS:
        d[f"floor_pooled_{lab}"] = float(np.mean([m[lab]["floor"] for m in metrics]))
    for k in ALARM_KEYS:
        v = np.array([a[k] for a in alarm_list])
        d[f"a01_cut_{k}_mean"] = float(v.mean())
        d[f"a01_cut_{k}_std"] = float(v.std(ddof=1))
        for s, x in zip(SEEDS, v):
            d[f"a01_cut_{k}_s{s}"] = float(x)
    return d


def trained_b() -> dict:
    """{H: row} for B's trained combiner: b_final.csv (model = full) + cp_curves.csv."""
    out = {}
    with (RESULTS_DIR / "b_final.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] == "full":
                out[int(r["H"])] = dict(r)
    with (RESULTS_DIR / "cp_curves.csv").open() as f:
        for r in csv.DictReader(f):
            if (r["detector"].startswith("B H=") and r["variant"] == "cut"
                    and abs(float(r["alpha"]) - ALPHA_TABLE) < 1e-9):
                H, s = int(r["detector"][4:]), int(r["seed"])
                for k in ALARM_KEYS:
                    out[H][f"a01_cut_{k}_s{s}"] = float(r[k])
    for H, r in out.items():
        for k in ALARM_KEYS:
            v = np.array([r[f"a01_cut_{k}_s{s}"] for s in SEEDS])
            r[f"a01_cut_{k}_mean"], r[f"a01_cut_{k}_std"] = float(v.mean()), float(v.std(ddof=1))
    return out


# --- Figure ------------------------------------------------------------------------------
def figure(lf, idx, kept, splits) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
    fig, axes = plt.subplots(1, len(HS), figsize=(13, 4.2), facecolor=surface, sharey=True)
    t = np.arange(T)
    for ax, H in zip(axes, HS):
        ax.set_facecolor(surface)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.tick_params(colors=muted, length=0)
        for v in VARIANTS:
            curves = []
            for s in SEEDS:
                up = splits[s]["pos"]["unseen"]
                sel = up[idx.y[up] == 1]
                x = np.where(kept[sel], lf[(v, H, s)][1][sel], np.nan)
                n = np.sum(np.isfinite(x), 0)
                curves.append(np.where(n >= 5, np.nanmean(x, 0), np.nan))
            m = np.nanmean(curves, 0)
            ax.plot(t, m, color=COLOURS[v], linewidth=2)
            last = int(np.nonzero(np.isfinite(m))[0].max())
            ax.annotate(v, (t[last], m[last]), xytext=(4, 0), textcoords="offset points",
                        color=ink, fontsize=8, va="center")
        ax.set_xlim(0, T + 45)
        ax.set_title(f"H = {H}", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("Policy step t", color=muted)
    axes[0].set_ylabel("Mean probe P per step (per-row score)", color=muted)
    fig.suptitle("Length check: label-free B combiners on unseen failures "
                 "(mean of 3 seeds; steps with ≥ 5 failures still running)",
                 color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_t.png", dpi=160, facecolor=surface)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}
    probes = load_probes()
    counts = {H: len(probes[(H, SEEDS[0])][0]) for H in HS}
    print(f"[*] reused Stage 5 probe outputs from {PROBE_DIR}; probes per H {counts}")

    chk = sanity_check(probes, idx, kept, splits)
    write_csv(OUT / "sanity_check.csv", chk)
    worst = max(chk, key=lambda r: r["abs_diff"])
    n_bad = sum(r["abs_diff"] > CHECK_TOL for r in chk)
    print(f"[check] max |diff| {worst['abs_diff']:.4f} at L{worst['layer']} {worst['target']} "
          f"H{worst['H']} seed {worst['seed']} {worst['set']}; {n_bad} over {CHECK_TOL}")
    if n_bad:
        print("[check] FAILED: stopping before scoring (see sanity_check.csv)")
        sys.exit(1)
    print("[check] PASSED")

    lf = lf_scores(probes, kept)
    rows, per_seed = [], {}
    for v in VARIANTS:
        for H in HS:
            mets, als = [], []
            for s in SEEDS:
                names, p, sc = lf[(v, H, s)]
                mets.append(evaluate(sc, idx, kept, splits[s]))
                als.append(alarms(sc, idx, kept, splits[s], s))
                np.savez_compressed(SCORES_DIR / f"lf_{SLUG[v]}_h{H}_seed{s}.npz",
                                    rollout_ids=idx.rids, H=H, names=np.array(names),
                                    p=p.astype(np.float32), s=sc.astype(np.float32))
            per_seed[(v, H)] = [m["unseen"]["per_task"] for m in mets]
            rows.append(table_row({"variant": v, "H": H, "n_inputs": len(lf[(v, H, SEEDS[0])][0])},
                                  mets, als))
    write_csv(OUT / "lf_final.csv", rows)

    tb = trained_b()
    diffs = []
    for v in VARIANTS:
        for H in HS:
            d = np.array(per_seed[(v, H)]) - np.array([float(tb[H][f"per_task_unseen_s{s}"]) for s in SEEDS])
            diffs.append({"variant": v, "H": H, "metric": "m1_per_task_unseen",
                          "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
                          **{f"diff_s{s}": float(x) for s, x in zip(SEEDS, d)}})
    write_csv(OUT / "lf_vs_trained.csv", diffs)
    figure(lf, idx, kept, splits)

    # --- Report ------------------------------------------------------------------------
    print("\n=== M1 (mean ± std over 3 seeds) and alpha = 0.1 cut alarms on unseen ===")
    print(f"   {'detector':16s} {'n':>4s}  seen pooled  seen per-task  unseen pooled  unseen per-task"
          f"   bal.acc  caught  false-al  det.time")
    show = [(f"{r['variant']} H={r['H']}", r) for r in rows] + [(f"trained B H={H}", tb[H]) for H in HS]
    for lab, r in show:
        g = lambda k: float(r[k])  # noqa: E731
        print(f"   {lab:16s} {int(r['n_inputs']):4d}  {g('pooled_seen_mean'):.3f}±{g('pooled_seen_std'):.3f}"
              f"  {g('per_task_seen_mean'):.3f}±{g('per_task_seen_std'):.3f}"
              f"   {g('pooled_unseen_mean'):.3f}±{g('pooled_unseen_std'):.3f}"
              f"    {g('per_task_unseen_mean'):.3f}±{g('per_task_unseen_std'):.3f}"
              f"     {g('a01_cut_bal_acc_mean'):.3f}   {g('a01_cut_tpr_mean'):.3f}"
              f"    {1 - g('a01_cut_tnr_mean'):.3f}     {g('a01_cut_det_time_mean'):.3f}")
    fl = rows[0]
    print(f"   floor (timestep only, pooled): seen {fl['floor_pooled_seen']:.3f}  unseen "
          f"{fl['floor_pooled_unseen']:.3f}; per-task floor 0.5; tasks {fl['n_tasks_seen']} | "
          f"{fl['n_tasks_unseen']}")
    print("\n=== unseen M1-per-task, label-free minus trained B (per seed; 3 seeds) ===")
    for d in diffs:
        print(f"   {d['variant']} H={d['H']:2d}: {d['diff_mean']:+.4f} ± {d['diff_std']:.4f}   "
              + "  ".join(f"s{s} {d[f'diff_s{s}']:+.4f}" for s in SEEDS))
    print(f"[*] {OUT}/lf_final.csv, lf_vs_trained.csv, sanity_check.csv, score_vs_t.png, scores/")


if __name__ == "__main__":
    main()
