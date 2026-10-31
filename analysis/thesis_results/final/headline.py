"""
headline.py

§1 of results_followup_handoff.md: the thesis tables and figures regenerated for the
headline setup (full training set, tested on unseen tasks incl. LIBERO-PRO). No refits:
every number is read from the full training set's run (results/thesis_results_full/ and
its work files), re-cut to the headline test set; the paired bootstrap and the example
rollout are recomputed on that test set from saved per-row scores.

    T1      tables/table1_core_detectors.csv (headline block, then the original-rollout
            block for reference), table1_core_detectors_all_test_sets.csv, table1_stress_set.csv
    Paired  tables/paired_comparisons.csv (the main run's §4.5 list, headline test set)
    T3      tables/table3_constraints.csv (full-set selections; pooled AUROC and floor on
            the headline test set)
    T4      tables/table4_b_by_lookahead.csv (best probe, mean, embedding-temporal mean,
            trained combiner, paired difference mean minus combiner; headline test set)
    T5      tables/table5_ablations_families_and_lookahead.csv (headline test set)
    T2, F3, F11   copied unchanged from results/thesis_results/
    F1      example failure and success on the headline test set (full training set)
    F2      single panel, headline test set
    F4      length check on the headline test set
    F5, F6-F8, F10   headline test set (F6-F8 share one y-range, as before)
    appendix/   F9 (B's own-target heatmap, headline test set), the persistence and
                own-target tables

Usage: source env.sh; python analysis/thesis_results/final/headline.py [tables paired example figures appendix]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc  # noqa: E402
from fc import HEAD, HEAD_NAME, REF_NAME, SEEDS, T, common, det, ev, log, read_csv, write_csv  # noqa: E402

TAB = fc.OUT / "tables"
FIG = fc.OUT / "figures"
DATA = FIG / "data"
APP = fc.OUT / "appendix"
FT = fc.FULL / "tables"
PT = common.PER_TASK_LABEL[HEAD]       # per-task-and-perturbation AUROC


def drop_empty(rows: list) -> list:
    cols = [c for c in rows[0] if any(r.get(c, "") not in ("", None) for r in rows)] if rows else []
    return [{c: r.get(c, "") for c in cols} for r in rows]


# --- tables -----------------------------------------------------------------------------
def t1() -> None:
    allr = read_csv(FT / "table1_core_detectors_all_test_sets.csv")
    out = []
    for ts, block in ((HEAD_NAME, "headline: full training set, unseen tasks incl. LIBERO-PRO"),
                      (REF_NAME, "reference: full training set, unseen tasks, original rollouts")):
        for r in allr:
            if r["test set"] == ts:
                pt = common.PER_TASK_LABEL["T2" if ts == HEAD_NAME else "T0"]
                rr = {"block": block}
                for c, v in r.items():
                    rr[c.replace(pt, "per-task AUROC (per task-and-perturbation cell on the headline test set)")
                      if c.startswith(pt) else c] = v
                out.append(rr)
    write_csv(TAB / "table1_core_detectors.csv", out)
    write_csv(TAB / "table1_core_detectors_all_test_sets.csv", allr)
    shutil.copyfile(FT / "table1_stress_set.csv", TAB / "table1_stress_set.csv")
    log(f"[headline] T1: {sum(r['block'].startswith('headline') for r in out)} headline rows, "
        f"{sum(r['block'].startswith('reference') for r in out)} reference rows")


def t3() -> None:
    """tables.t3 with the pooled AUROC and floor read on the headline test set."""
    from collections import defaultdict
    sel = common.selections()["betas"]
    fam = {"act_mag": "action", "act_dir": "action", "grip_flip": "action", "act_rep": "action",
           "emb_temp": "emb_temp", "xl_adj": "cross_layer", "xl_final": "cross_layer", "xl_spread": "cross_layer"}
    S = [r for r in read_csv(fc.REPO / "results/abc2/constraint_m1_summary.csv") if r["cond"] == common.POOL]
    by = defaultdict(dict)
    for r in S:
        by[(r["variant"], r["constraint"], r["layer"], r["beta"])][r["test_set"]] = r
    rows = []

    def emit(kind, name, key, note=""):
        d = by[key]
        row = {"kind": kind, "signal": name, "smoothing beta": key[3],
               "sign per seed (+: higher means failure)": d["eval_seen"]["signs_all_seeds"],
               "sign-unstable": d["eval_seen"]["unstable"] == "True"}
        for ts in (HEAD, "eval_seen", "T0"):
            lab = common.PER_TASK_LABEL[ts]
            row[f"{lab}, {common.TEST_SETS[ts]}, mean"] = fc.r4(float(d[ts]["m1_per_task_mean"]))
            row[f"{lab}, {common.TEST_SETS[ts]}, std"] = fc.r4(float(d[ts]["m1_per_task_std"]))
        row[f"pooled AUROC, {HEAD_NAME}, mean"] = fc.r4(float(d[HEAD]["m1_pooled_mean"]))
        row[f"timestep-only floor (pooled AUROC), {HEAD_NAME}"] = fc.r4(float(d[HEAD]["floor_mean"]))
        row["note"] = note
        row["source"] = "results/abc2/constraint_m1_summary.csv (full training set rows)"
        rows.append(row)

    for c in ("emb_temp", "xl_adj", "xl_final"):
        cand = [k for k in by if k[0] == "primary" and k[1] == c and k[3] == sel[fam[c]]]
        best = max(cand, key=lambda k: float(by[k]["eval_seen"]["m1_per_task_mean"]))
        emit("untrained constraint", f"{c}[{best[2]}]", best, "layer chosen on eval-seen per-task AUROC")
    for c in ("act_rep", "act_dir", "act_mag", "grip_flip", "xl_spread"):
        emit("untrained constraint", c, ("primary", c, "-1", sel[fam[c]]))
    for k, lab in (("token_entropy", "token mean entropy"), ("token_maxprob", "token max-probability")):
        s_ = det.load(k)["selection"]
        emit("baseline", lab, ("baseline", s_["series"], "-1", s_["beta"]), "beta chosen on eval-seen per-task AUROC")
    emit("baseline", "FailureSpot signal s", ("failurespot", "failurespot", "-1", "0.8"), "fixed orientation")
    emit("baseline", "FailureSpot signal -s", ("failurespot", "failurespot_neg", "-1", "0.8"), "fixed orientation")
    emit("baseline", "timestep only", ("baseline", "timestep", "-1", "none"), "0.5 per cell by construction")
    write_csv(TAB / "table3_constraints.csv", rows)


def t4_t5() -> None:
    rows = [r for r in read_csv(FT / "table4_b_by_lookahead.csv") if r["test set"] == HEAD_NAME]
    write_csv(TAB / "table4_b_by_lookahead.csv", drop_empty(rows))
    rows = [r for r in read_csv(FT / "table5_ablations_families_and_lookahead.csv") if r["test set"] == HEAD_NAME]
    write_csv(TAB / "table5_ablations_families_and_lookahead.csv", drop_empty(rows))


def copies() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fc.ORIG / "tables/table2_corpus_composition.csv", TAB / "table2_corpus_composition.csv")
    for f in ("kept_rollouts_against_t.csv", "action_repeat_before_after_success.csv", "success_completion_times.csv",
              "capacity.csv"):
        shutil.copyfile(fc.ORIG / "figures/data" / f, DATA / f)
    for f in ("F3_corpus_kept_rollouts_and_action_repeat", "F11_capacity"):
        for ext in ("pdf", "png"):
            shutil.copyfile(fc.ORIG / "figures" / f"{f}.{ext}", FIG / f"{f}.{ext}")


def tables() -> None:
    copies()
    t1()
    t3()
    t4_t5()


# --- paired comparisons -----------------------------------------------------------------
def paired() -> None:
    import paired as pr
    rows, per_seed, ident = pr.run(HEAD)
    for r in rows:
        if r["metric"] == "per-task AUROC":
            r["metric"] = PT
    chk = pr.identity_check(ident, HEAD)
    common.write_json(fc.OUT / "logs" / "paired_identity_check.json", chk)
    write_csv(TAB / "paired_comparisons.csv", rows)
    write_csv(TAB / "data" / "paired_comparisons_per_seed.csv", per_seed)
    log(f"[headline] paired: {len(rows)} rows; identity check max |diff| {chk['max_abs_diff']:.2e} "
        f"({'passed' if chk['passed'] else 'FAILED'})")


# --- F1: example rollouts ---------------------------------------------------------------
def example() -> None:
    """example.py's rule on the headline test set: candidates are failures that B's
    unweighted mean (H = 10) alarms on before the cut at alpha 0.1 and successes of the
    same task-and-perturbation cell (same cut) that it does not alarm on; among cells with
    both, the one whose alarmed failure has the longest run of repeated actions before the
    alarm (the clearest stall), with that cell's unalarmed success of the longest kept
    length. Seed 0 first, then 1, 2. Only rollouts with saved frames are candidates: the
    new rollouts (new unperturbed and LIBERO-PRO) saved only their first frame, so the
    pair comes from the original LIBERO-10 rollouts of the headline test set; cuts and
    alarms are still those of the full headline test set (the cut is the cell's)."""
    import example as ex
    from data2 import cell_name
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    for seed in SEEDS:
        pos = common.sets(seed)["T2"]
        cut = dict(zip(pos.tolist(), common.m1_cut(idx, pos).tolist()))
        E = {k: det.load(k)["seeds"][seed] for k in ("B_mean_h10", "A_mean", "C_mean")}
        band = {k: (lambda b: b["mu"] + b["h"][-1] * b["xi"])(fc.evaluate(k)["bands"][seed]) for k in E}
        al = {i: ex.first_alarm(E["B_mean_h10"]["cp"][i], band["B_mean_h10"], kept[i], cut[i]) for i in pos}
        choice = []
        framed = {i: len(list((idx.rollout_dir(i) / "frames").glob("agentview_*.jpg"))) > 1 for i in pos}
        for c in sorted(set(idx.task[pos].tolist())):
            fail = [i for i in pos if idx.task[i] == c and idx.y[i] == 1 and al[i] is not None and framed[i]]
            succ = [i for i in pos if idx.task[i] == c and idx.y[i] == 0 and al[i] is None and framed[i]]
            if fail and succ:
                stall = {i: ex.longest_repeat_before(i, al[i]) for i in fail}
                f = max(fail, key=lambda i: (stall[i], -al[i]))
                s = max(succ, key=lambda i: idx.kept_len[i])
                choice.append((stall[f], c, f, s))
        if choice:
            break
        log(f"[headline] example: no qualifying pair in seed {seed}")
    else:
        log("[headline] example: no seed has a qualifying pair")
        return
    choice.sort(reverse=True)
    stall, cell, f, s = choice[0]
    pert = {"ori": "original", "ori2": "new unperturbed", "obj": "object perturbation", "lan": "instruction perturbation"}
    info = {"seed": seed, "seed note": "seed 0" if seed == 0 else f"seed {seed}: seeds before it have no qualifying pair",
            "test set": HEAD_NAME, "training set": "full training set", "cell": cell_name(cell),
            "rule": example.__doc__.strip().replace("\n    ", " "),
            "candidates (longest repeat run before alarm, cell, failure, success)":
                [(int(a), cell_name(int(b)), str(idx.rids[c_]), str(idx.rids[d])) for a, b, c_, d in choice],
            "rollouts": {}}
    rows = []
    fdir = FIG / "example_frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    for i, role in ((f, "failure"), (s, "success")):
        c_, a = cut[i], al[i]
        steps = {"start": 1, "mid": int(c_ // 2),
                 "alarm" if a is not None else "mid-way to cut": a if a is not None else int(3 * c_ // 4), "cut": int(c_ - 1)}
        steps = dict(sorted(steps.items(), key=lambda kv: kv[1]))      # frames left to right in time
        frames = {}
        for nm, tstep in steps.items():
            src = idx.rollout_dir(i) / "frames" / f"agentview_{tstep + ex.WAIT:04d}.jpg"
            dst = fdir / f"{role}_{idx.rids[i]}_{nm.replace(' ', '_')}_t{tstep:03d}.jpg"
            if src.exists():
                shutil.copyfile(src, dst)
                frames[nm] = {"t": tstep, "file": str(dst.relative_to(fc.OUT))}
        info["rollouts"][role] = {"rollout": str(idx.rids[i]),
                                  "task": f"{int(idx.base_task[i])}, {pert[str(idx.block[i])]}",
                                  "perturbation type": pert[str(idx.block[i])], "outcome": role,
                                  "kept length": int(idx.kept_len[i]), "cut": int(c_), "B mean alarm step": a,
                                  "A mean alarm step": ex.first_alarm(E["A_mean"]["cp"][i], band["A_mean"], kept[i], c_),
                                  "C mean alarm step": ex.first_alarm(E["C_mean"]["cp"][i], band["C_mean"], kept[i], c_),
                                  "longest action-repeat run before the alarm": stall if role == "failure" else None,
                                  "frames": frames}
        for t in range(T):
            def v(x):
                return round(float(x), 4) if np.isfinite(x) else ""
            rows.append({"seed": seed, "rollout": str(idx.rids[i]), "outcome": role, "t": t, "kept": bool(kept[i, t]),
                         "before cut": t < c_,
                         "B mean score (running sum, H = 10)": v(E["B_mean_h10"]["cp"][i, t]),
                         "B mean band (alpha 0.1)": v(band["B_mean_h10"][t]),
                         "A mean score (running sum)": v(E["A_mean"]["cp"][i, t]), "A mean band (alpha 0.1)": v(band["A_mean"][t]),
                         "C mean score (running sum)": v(E["C_mean"]["cp"][i, t]), "C mean band (alpha 0.1)": v(band["C_mean"][t]),
                         "cut": int(c_), "B mean alarm step": "" if a is None else a})
    info["why"] = (f"cell {cell_name(cell)}: the alarmed failure with the longest run of repeated actions before B's alarm "
                   f"({stall} steps), i.e. the clearest stall; the success of the same cell (same cut) with the longest "
                   f"kept length that B does not alarm on")
    write_csv(DATA / "example_rollouts.csv", rows)
    common.write_json(DATA / "example_rollouts.json", info)
    log(f"[headline] example: seed {seed}, {cell_name(cell)}: failure {idx.rids[f]} (alarm {al[f]}, stall {stall}), "
        f"success {idx.rids[s]}")


# --- figure data ------------------------------------------------------------------------
def data_f2() -> None:
    keys = {"A_mean": ("A", "A, unweighted mean"), "C_mean": ("C", "C, unweighted mean"),
            "B_mean_h10": ("B", "B, unweighted mean (H = 10)"),
            "emb_best": ("baseline", "best untrained constraint"),
            "token_entropy": ("baseline", "token mean entropy"),
            "failurespot_neg": ("baseline", "FailureSpot (-s)")}
    rows = []
    for r in read_csv(FT / "data" / "detector_alarm_per_seed.csv"):
        if r["key"] in keys and r["on_alpha_grid"] == "True" and r["test_set"] == HEAD_NAME:
            rows.append({"series": keys[r["key"]][1], "group": keys[r["key"]][0], "test set": r["test_set"],
                         "alpha": round(float(r["alpha"]), 4), "seed": r["seed"],
                         "balanced accuracy": round(float(r["bal_acc"]), 4),
                         "mean normalised detection time": round(float(r["det_time"]), 4)})
    write_csv(DATA / "alarm_tradeoff.csv", rows)


def data_f4() -> None:
    """corpus.length_check on the headline test set (kept rows; a curve point needs >= 5
    rollouts of the class still kept)."""
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    keys = {"A_mean": "A, unweighted mean (running sum)", "C_mean": "C, unweighted mean (running sum)",
            "B_mean_h10": "B, unweighted mean, H = 10 (running sum)", "timestep": "timestep only (t)"}
    curves = {}
    for k, lab in keys.items():
        E = det.load(k)["seeds"]
        for cls, yv in (("failures", 1), ("successes", 0)):
            per = []
            for s in SEEDS:
                pos = common.sets(s)[HEAD]
                sel = pos[idx.y[pos] == yv]
                x = np.where(kept[sel], E[s]["m1"][sel], np.nan)
                n = np.isfinite(x).sum(0)
                with np.errstate(invalid="ignore"):
                    per.append(np.where(n >= 5, np.nanmean(x, 0), np.nan))
            curves[(lab, cls)] = np.array(per)
    rows = []
    for t in range(T):
        r = {"t": t, "test set": HEAD_NAME}
        for (lab, cls), v in curves.items():
            with np.errstate(invalid="ignore"):
                r[f"{lab}, {cls}, mean over seeds"] = round(float(np.nanmean(v[:, t])), 5) if np.isfinite(v[:, t]).any() else ""
            for j, s in enumerate(SEEDS):
                r[f"{lab}, {cls}, seed {s}"] = round(float(v[j, t]), 5) if np.isfinite(v[j, t]) else ""
        rows.append(r)
    write_csv(DATA / "length_check.csv", rows)


def figure_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    data_f2()
    data_f4()
    for f in ("untrained_constraint_heatmap.csv", "per_probe_outcome_auroc.csv"):
        shutil.copyfile(fc.FULL / "figures/data" / f, DATA / f)


# --- figures ----------------------------------------------------------------------------
def figures_mod():
    """analysis/thesis_results/figures.py pointed at this run's folders and the headline
    test set; each saved figure is relabelled (per-task -> per-task-and-perturbation AUROC,
    test-set names) before saving."""
    import figures as fg
    common.OUT_DIR = fc.OUT                      # F1 reads its frames relative to OUT_DIR
    fg.FIG, fg.DATA, fg.HEAD = FIG, DATA, HEAD_NAME
    orig_save = fg.save

    def relabel(fig):
        def fix(s):
            return (s.replace("per-task AUROC against outcome\n", f"{PT}\nagainst outcome\n")
                    .replace("per-task AUROC", PT).replace(REF_NAME, HEAD_NAME)
                    .replace("unseen tasks incl. LIBERO-PRO", "\0").replace("unseen tasks", HEAD_NAME).replace("\0", HEAD_NAME))
        for ax in fig.axes:
            ax.set_ylabel(fix(ax.get_ylabel()))
            ax.set_xlabel(fix(ax.get_xlabel()))
            for t in ax.texts:
                t.set_text(fix(t.get_text()))
            lg = ax.get_legend()
            if lg:
                for t in lg.get_texts():
                    t.set_text(fix(t.get_text()))
        for lg in fig.legends:
            for t in lg.get_texts():
                t.set_text(fix(t.get_text()))

    def save(fig, name):
        relabel(fig)
        orig_save(fig, name)
    fg.save = save
    return fg


def F2(fg) -> None:
    """Single panel of figures.F2 on the headline test set."""
    plt = fg.plt
    rows = read_csv(DATA / "alarm_tradeoff.csv")
    series = list(dict.fromkeys(r["series"] for r in rows))
    style = {s: (fg.COL.get(s[0], fg.GREY) if s[0] in fg.COL and s[1] == "," else [fg.GREY, "#4d4d49", fg.GREY2][i % 3],
                 fg.MARK.get(s[0], "D") if s[1] == "," else ["D", "v", "x"][i % 3],
                 fg.LS.get(s[0], ":") if s[1] == "," else [":", (0, (1, 1)), (0, (5, 1, 1, 1))][i % 3])
             for i, s in enumerate(series)}
    fig, ax = plt.subplots(figsize=(fg.W, 8 * fg.CM))
    for s in series:
        R = [r for r in rows if r["series"] == s]
        al = sorted({float(r["alpha"]) for r in R})
        ba = np.array([[float(r["balanced accuracy"]) for r in R if float(r["alpha"]) == a] for a in al])
        dt = np.array([[float(r["mean normalised detection time"]) for r in R if float(r["alpha"]) == a] for a in al])
        c, m, ls = style[s]
        ax.plot(dt.mean(1), ba.mean(1), color=c, marker=m, ms=4, ls=ls, lw=1.2, label=s, mfc="white" if s[1] != "," else c)
        if s[1] == ",":
            sd = ba.std(1, ddof=1)
            ax.fill_between(dt.mean(1), ba.mean(1) - sd, ba.mean(1) + sd, color=c, alpha=0.08, lw=0)
    fg.chance(ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylabel("balanced accuracy")
    ax.set_xlabel("mean normalised detection time over failures\n(first alarm step / cut; 1 = no alarm)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fg.save(fig, "F2_alarm_tradeoff")


def figures() -> None:
    fg = figures_mod()
    for name in ("F1", "F2", "F4", "F5", "F6", "F7", "F8", "F10"):
        try:
            F2(fg) if name == "F2" else getattr(fg, name)()
            log(f"[headline] {name} ok")
        except Exception as e:
            log(f"[headline] {name} FAILED: {type(e).__name__}: {e}")
            raise


# --- appendix ---------------------------------------------------------------------------
def appendix() -> None:
    APP.mkdir(parents=True, exist_ok=True)
    (APP / "data").mkdir(exist_ok=True)
    for f in ("table4_b_linear_vs_persistence.csv", "table4_b_mlp_vs_linear_vs_persistence.csv"):
        shutil.copyfile(FT / f, APP / f)
    shutil.copyfile(FT / "data" / "b_target_auroc.csv", APP / "data" / "b_target_auroc.csv")
    shutil.copyfile(fc.FULL / "figures/data/b_target_auroc_heatmap.csv", APP / "data" / "b_target_auroc_heatmap.csv")
    fg = figures_mod()
    fg.FIG, fg.DATA = APP, APP / "data"
    fg.F9()
    log("[headline] appendix written")


def main() -> None:
    fc.quiet()
    args = sys.argv[1:] or ["tables", "paired", "example", "figure_data", "figures", "appendix"]
    for a in args:
        globals()[a]()
        log(f"[headline] {a} ok")


if __name__ == "__main__":
    main()
