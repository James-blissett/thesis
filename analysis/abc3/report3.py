"""
report3.py

Writes the deep-MLP run's report, results/abc3/REPORT.md (deep_mlp_handoff.md, handoff
version 1, section 4), the CSV behind every table and the two figures. Reads only
finished outputs (formats: analysis/abc3/FORMATS.md); nothing is recomputed except
tables reshaped for reading (means / std over seeds of stress.csv, counts over grid.csv).
Every conclusion is generated from the numbers; the headline comes from paired_T0.csv
with the abc2 rule: an improvement only if the 95% interval lies entirely above zero
(column `improvement`).

Inputs (<in> = --in-dir, default results/abc3; a missing file gives a "(missing: ...)"
note in its section, never a crash):
  final_table.csv, selections.csv, grid.csv, train_log.csv, stress.csv,
  eval_status.json, eval_check.json                                    (eval3.py)
  paired_T0.csv, paired_T0_check.json                                  (paired3.py)
  sanity_check.json, precision.json, RUNLOG.md, lead_notes.md (opt.)   (the lead)

Outputs (<out> = --out-dir, default <in>):
  REPORT.md
  report_tables/*.csv                 one per table in the report (named under each table)
  report_figures/m1_vs_depth.png      M1-per-task against depth (linear at 0), panels A / C,
                                      train / eval-seen / T0, D0 and D2, error bars = std
                                      over seeds; abc2 plain MLP as hollow markers
  report_figures/training_curves.png  C input, D2, seed 0, each depth's chosen fit: train
                                      loss and M1-per-task (eval-seen, train) per epoch

Usage (repo root, CPU, a few seconds):
    source env.sh
    python analysis/abc3/report3.py
    python analysis/abc3/report3.py --in-dir /data/tmp/x/in --out-dir /data/tmp/x/out
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "abc2"))
# Small helpers are imported from the abc2 report (unchanged): csv / json readers that return
# empty on a missing file, number formatting, markdown tables, RUNLOG bullet extraction.
from report2 import (CO_AUTHOR, RUNLOG_PAT, f, md_table, n3, pm, read_csv, read_json,  # noqa: E402,F401
                     runlog_entries, write_csv)

INPUTS = ("A", "C")
POOLS = ("D0", "D2")
SEEDS = ("0", "1", "2")
DEPTHS = (2, 6, 12)
MODELS = ("linear", "mlp-abc2", "d2", "d6", "d12")       # depth-table row order within (input, pool)
REG_ORDER = ((0.0, 0.01), (0.0, 0.1), (0.3, 0.01), (0.3, 0.1))
SETS = ("train", "eval_seen", "T0", "T1", "T2")
SET_LABEL = {"train": "Train", "eval_seen": "Eval-seen (optimistic)", "T0": "T0", "T1": "T1", "T2": "T2"}
METRIC_LABEL = {"m1_per_task": "M1-per-task", "bal_acc": "balanced accuracy", "tpr": "failures caught"}
FAMILIES = (("vs_linear", "Each depth against the linear detector (same input and pool)"),
            ("vs_depth2", "Depth 6 and 12 against depth 2 (same input and pool)"),
            ("D2_vs_D0", "D2 against D0 (same detector)"))
# Colour-blind-safe categorical slots 1-3 (dataviz reference palette, validated pair blue/orange).
POOL_COLOUR = {"D0": "#2a78d6", "D2": "#eb6834"}
DEPTH_COLOUR = {2: "#2a78d6", 6: "#eb6834", 12: "#1baf7a"}
INK, MUTED, GRIDC, WHITE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#ffffff"
SET_STYLE = {"train": (":", "^"), "eval_seen": ("--", "s"), "T0": ("-", "o")}


# --- helpers ---------------------------------------------------------------------------------
def det_name(inp: str, model: str) -> str:
    return {"linear": f"{inp}-linear", "mlp-abc2": f"{inp}-mlp-abc2"}.get(model, f"{inp}-{model}")


def det_label(det: str) -> str:
    """Readable model name for a detector key (A-linear, A-mlp-abc2, A-d6, ...)."""
    _, _, m = det.partition("-")
    if m == "linear":
        return "linear (abc2)"
    if m == "mlp-abc2":
        return "abc2 plain MLP, 2 layers (ref.)"
    if m.startswith("d") and m[1:].isdigit():
        return f"residual MLP, depth {m[1:]}"
    return det


def short(det: str) -> str:
    _, _, m = det.partition("-")
    return {"linear": "linear", "mlp-abc2": "abc2 MLP"}.get(m, m.replace("d", "depth ") if m[:1] == "d" else m)


def num(v) -> str:
    """Compact number for p / wd: 0, 0.3, 0.01."""
    x = f(v)
    return "?" if not np.isfinite(x) else f"{x:g}"


def is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def as_int(v, default=None):
    x = f(v)
    return int(round(x)) if np.isfinite(x) else default


def missing(*names: str) -> str:
    return "*(missing: " + ", ".join(f"`{n}`" for n in names) + ")*"


def ci(r: dict) -> str:
    return f"{n3(r['diff_mean'], True)} [{n3(r['diff_ci_lo'], True)}, {n3(r['diff_ci_hi'], True)}]"


def depth_of(r: dict):
    return as_int(r.get("depth"))


class Inputs:
    def __init__(self, d: Path):
        self.dir = d
        self.final = read_csv(d / "final_table.csv")
        self.sel = read_csv(d / "selections.csv")
        self.grid = read_csv(d / "grid.csv")
        self.log = read_csv(d / "train_log.csv")
        self.stress = read_csv(d / "stress.csv")
        self.paired = read_csv(d / "paired_T0.csv")
        self.paired_check = read_json(d / "paired_T0_check.json")
        self.eval_status = read_json(d / "eval_status.json")
        self.eval_check = read_json(d / "eval_check.json")
        self.sanity = read_json(d / "sanity_check.json")
        self.precision = read_json(d / "precision.json")
        self.runlog = (d / "RUNLOG.md").read_text() if (d / "RUNLOG.md").exists() else ""
        self.lead = (d / "lead_notes.md").read_text().strip() if (d / "lead_notes.md").exists() else ""
        self._ft = {(r["cond"], r["test_set"], r["detector"]): r for r in self.final}

    def ft(self, cond, ts, det):
        return self._ft.get((cond, ts, det))

    def sel_rows(self, cond, det):
        rs = [r for r in self.sel if r["cond"] == cond and r["detector"] == det]
        return sorted(rs, key=lambda r: as_int(r["seed"], 0))

    def failed_seeds(self, cond, det) -> list[str]:
        return [r["seed"] for r in self.sel_rows(cond, det) if r.get("status", "ok") != "ok"]

    def rows_order(self):
        """(input, pool, detector) in the report's row order, only those present anywhere."""
        out = []
        for i in INPUTS:
            for c in POOLS:
                for m in MODELS:
                    d = det_name(i, m)
                    if any(k[0] == c and k[2] == d for k in self._ft) or self.sel_rows(c, d):
                        out.append((i, c, d))
        return out


# --- 0. warnings at the top --------------------------------------------------------------------
def top_warnings(inp: Inputs) -> list[str]:
    out = []
    es = inp.eval_status
    if es is None:
        out += [f"**Evaluation status unknown** {missing('eval_status.json')}.", ""]
    elif es.get("status") != "done":
        out += [f"**EVALUATION NOT COMPLETE** (`eval_status.json` status `{es.get('status')}`): "
                f"{es.get('reason', '')}", ""]
    ec = inp.eval_check
    if ec is not None and not ec.get("passed", True):
        out += [f"**REFERENCE-ROW CHECK FAILED** (`eval_check.json`, max |diff| {ec.get('max_abs_diff')}): the "
                "recomputed linear / abc2-MLP rows do not reproduce abc2's tables.", ""]
    pc = inp.paired_check
    if pc is not None and not pc.get("passed", True):
        out += [f"**BOOTSTRAP IDENTITY CHECK FAILED** (`paired_T0_check.json`, max |diff| "
                f"{pc.get('max_abs_diff')}): treat section 4 and the headline with caution.", ""]
    return out


# --- 1. headline -----------------------------------------------------------------------------
MET_SHORT = {"m1_per_task": "M1-per-task", "bal_acc": "bal. acc.", "tpr": "failures caught"}


def lab_row(r: dict) -> str:
    """Compact label: 'C-d6 vs linear, D0, bal. acc. +0.094 [+0.045, +0.143]'."""
    if r.get("family") == "D2_vs_D0":
        what = f"{r['detector_x']} D2 vs D0"
    else:
        what = f"{r['detector_x']} vs {short(r['detector_y'])}, {r.get('pool') or r['cond_x']}"
    return f"{what}, {MET_SHORT.get(r['metric'], r['metric'])} {ci(r)}"


def listing(rows: list[dict]) -> str:
    return " (" + "; ".join(lab_row(r) for r in rows) + ")" if rows else ""


def rng_txt(rows: list[dict]) -> str:
    d = [f(r["diff_point"]) for r in rows]
    d = [x for x in d if np.isfinite(x)]
    return f"{min(d):+.3f} to {max(d):+.3f}" if d else "n/a"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def matched(k: int, n: int) -> str:
    return f"all {n} of their" if k == n else ("none of their " + str(n) if k == 0 else f"{k} of their {n}")


def counts(rows: list[dict]) -> tuple[list, list]:
    return ([r for r in rows if is_true(r["improvement"])],
            [r for r in rows if is_true(r["significant_worsening"])])


def headline(inp: Inputs) -> tuple[list[str], dict]:
    """Generated from paired_T0.csv. The question (handoff §0): does depth 6 or 12 beat linear or
    depth 2 on T0, for ranking (M1-per-task) or alarm (balanced accuracy, failures caught)?"""
    out = ["## 1. Headline", ""]
    rows = inp.paired
    if not rows:
        return out + ["The paired comparison on T0 is not available " + missing("paired_T0.csv")
                      + ", so no claim about depth can be made.", ""], {}
    key = [r for r in rows if r.get("family") in ("vs_linear", "vs_depth2") and depth_of(r) in (6, 12)]
    imp, wor = counts(key)
    n_boot = rows[0].get("n_boot", "?")
    sel = lambda fam, alarm: [r for r in key if r["family"] == fam  # noqa: E731
                              and (r["metric"] != "m1_per_task") == alarm]
    rl, rd, al, ad = sel("vs_linear", False), sel("vs_depth2", False), sel("vs_linear", True), sel("vs_depth2", True)
    (rli, rlw), (rdi, rdw), (ali, alw), (adi, adw) = counts(rl), counts(rd), counts(al), counts(ad)
    # does the 2-layer residual MLP show the same improvement over linear (same input, pool, metric)?
    d2lin = {(r.get("input"), r.get("pool"), r["metric"]): r for r in rows
             if r.get("family") == "vs_linear" and depth_of(r) == 2}
    shared = [r for r in ali if is_true(d2lin.get((r.get("input"), r.get("pool"), r["metric"]), {})
                                        .get("improvement"))]
    deep_vs_d2_imp, deep_vs_d2_wor = len(rdi) + len(adi), len(rdw) + len(adw)

    if not imp:
        s1 = ("**No: neither 6 nor 12 layers beats the linear detector or the 2-layer residual MLP on T0, for "
              f"ranking or for alarm.** None of the {len(key)} paired comparisons has a 95% interval entirely "
              f"above zero; {len(wor)} {'is a significant worsening' if len(wor) == 1 else 'are significant worsenings'}"
              f"{listing(wor)}.")
    elif not deep_vs_d2_imp:
        s1 = ("**Not from depth: 6 and 12 layers never beat the 2-layer residual MLP on T0"
              + ("" if rli else ", and never beat linear on ranking") + ".**"
              + (f" {matched(len(shared), len(ali)).capitalize()} alarm gains over linear are matched by the "
                 "2-layer MLP." if ali else ""))
    else:
        tail = (("" if rli else "; against linear they never improve ranking")
                + (f", and {matched(len(shared), len(ali))} alarm gains over linear are matched by the 2-layer MLP"
                   if ali else "") + ".")
        if deep_vs_d2_imp <= deep_vs_d2_wor:
            s1 = (f"**Depth beyond 2 layers does not reliably help on T0.** Against the 2-layer residual MLP, 6 and "
                  f"12 layers improve in {deep_vs_d2_imp} and are significantly worse in {deep_vs_d2_wor} of "
                  f"{len(rd) + len(ad)} comparisons" + tail)
        else:
            s1 = (f"**Partly: against the 2-layer residual MLP, 6 and 12 layers improve in {deep_vs_d2_imp} of "
                  f"{len(rd) + len(ad)} comparisons on T0** ({deep_vs_d2_wor} significantly worse)" + tail)
    s2 = (f"Ranking (M1-per-task on T0): depth 6 / 12 improve on linear in {len(rli)} of {len(rl)} and on depth 2 in "
          f"{len(rdi)} of {len(rd)} input × pool comparisons, and are significantly worse in {len(rlw) + len(rdw)}"
          f"{listing(rlw + rdw)}; point differences {rng_txt(rl)} against linear, {rng_txt(rd)} against depth 2.")
    s3 = (f"Alarm at α = 0.1 (balanced accuracy, failures caught): against linear, {len(ali)} of {len(al)} "
          f"improvements{listing(ali)}"
          + ((f", {len(shared)} of which the 2-layer MLP also achieves (same input, pool and metric), so that "
              "part of the gain comes from using an MLP, not from depth" if shared else
              ", none of which the 2-layer MLP also achieves") if ali else "")
          + f"; {len(alw)} significantly worse{listing(alw)}. Against depth 2: {len(adi)} of {len(ad)} improvements"
          f"{listing(adi)}, {len(adw)} significantly worse{listing(adw)}.")
    other = []
    for fam, txt in (("vs_linear", "Depth 2 against linear"), ("D2_vs_D0", "D2 against D0")):
        rs = [r for r in rows if r.get("family") == fam and (fam != "vs_linear" or depth_of(r) == 2)]
        if rs:
            ni, nw = counts(rs)
            other.append(f"{txt}: {plural(len(ni), 'improvement')} and {plural(len(nw), 'significant worsening')} "
                         f"of {len(rs)}")
    all_i, all_w = counts(rows)
    s4 = ("Other comparisons (section 4): " + "; ".join(other) + f". Over all {len(rows)} rows of "
          f"`paired_T0.csv`: {plural(len(all_i), 'improvement')}, {plural(len(all_w), 'significant worsening')}.")
    out += [s1, "", s2, "", s3, "", s4, "",
            f"*Rule: an improvement only if the paired-bootstrap 95% interval ({n_boot} resamples) lies entirely "
            "above zero. Differences are x − y (positive = the deeper model, or D2, is better): bootstrap mean with "
            "the 95% percentile interval. Alarm intervals hold each seed's conformal band fixed, so they "
            "understate the uncertainty (section 4).*", ""]
    if inp.lead:
        out += ["## Lead's notes", "", inp.lead, ""]
    return out, {"key": key, "imp": imp, "wor": wor, "all": rows,
                 "deep_vs_d2": (deep_vs_d2_imp, deep_vs_d2_wor, len(rd) + len(ad)),
                 "deep_vs_lin": (len(rli) + len(ali), len(rlw) + len(alw), len(rl) + len(al))}


# --- 2. depth table ----------------------------------------------------------------------------
def per_seed_sel(inp: Inputs, cond: str, det: str):
    rs = inp.sel_rows(cond, det)
    if not rs:
        return "—", "—", []
    reg, ep = [], []
    for r in rs:
        if r.get("status", "ok") != "ok":
            reg.append("failed")
            ep.append("failed")
        else:
            reg.append(f"{num(r['p'])}/{num(r['wd'])}")
            ep.append(str(as_int(r.get("best_epoch"), "?")))
    return ", ".join(reg), ", ".join(ep), rs


def params_txt(v) -> str:
    x = f(v)
    return f"{int(round(x)):,}" if np.isfinite(x) else "–"


def depth_table(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 2. Depth table: M1-per-task (mean ± std over seeds)", ""]
    if not inp.final:
        return out + [missing("final_table.csv"), ""]
    rows, csv_rows = [], []
    for i, c, d in inp.rows_order():
        reg, ep, _ = per_seed_sel(inp, c, d) if d.split("-")[1].startswith("d") else ("—", "—", [])
        fs = inp.failed_seeds(c, d)
        label = det_label(d)
        anyrow = next((inp.ft(c, ts, d) for ts in SETS if inp.ft(c, ts, d)), None)
        nseeds = as_int((anyrow or {}).get("n_seeds"), 0)
        if anyrow is None:
            label += " — **FAILED (all seeds)**"
        elif fs:
            label += f" — **failed seed {', '.join(fs)}** (mean over {nseeds})"
        line = [i, c, label, params_txt((anyrow or {}).get("n_params"))]
        rec = {"input": i, "pool": c, "detector": d, "model": (anyrow or {}).get("model", ""),
               "depth": (anyrow or {}).get("depth", ""), "n_params": (anyrow or {}).get("n_params", ""),
               "n_seeds": nseeds, "failed_seeds": " ".join(fs)}
        for ts in SETS:
            r = inp.ft(c, ts, d)
            line.append(pm(r["m1_per_task_mean"], r["m1_per_task_std"]) if r else "–")
            for k in ("m1_per_task_mean", "m1_per_task_std", "m1_pooled_mean", "m1_pooled_std"):
                rec[f"{ts}_{k}"] = r.get(k, "") if r else ""
        line += [reg, ep]
        rec["p_wd_per_seed"], rec["best_epoch_per_seed"] = reg, ep
        rows.append(line)
        csv_rows.append(rec)
    write_csv(tdir / "depth.csv", csv_rows)
    sel_note = "" if inp.sel else " " + missing("selections.csv") + " (dropout / weight decay not shown)."
    return out + [
        "Rows per input and pool: the linear detector (abc2's final detector, plotted as depth 0), abc2's plain "
        "256-256 MLP (reference only), then the pre-norm residual MLP at depth 2, 6 and 12 (width 256). "
        "**Eval-seen is optimistic**: it chose the epoch and the regularisation. Train = the pool's train "
        "rollouts, same scoring. Parameters: mean over seeds of the chosen fits (A's input layer can differ "
        "between seeds); linear = d + 1. *Dropout / wd*: chosen dropout p / weight decay per seed (seeds 0, 1, "
        "2); *best epoch*: early-stopping epoch of the chosen fit per seed. Pooled M1 for every set is in the "
        "csv. Table: `report_tables/depth.csv`." + sel_note, "",
        md_table(["Input", "Pool", "Model", "Params"] + [SET_LABEL[s] for s in SETS]
                 + ["Dropout / wd per seed", "Best epoch per seed"], rows), ""]


# --- 3. alarm ------------------------------------------------------------------------------------
def alarm_table(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 3. Alarm at α = 0.1 on T0 (cut variant, mean ± std over seeds)", ""]
    if not inp.final:
        return out + [missing("final_table.csv"), ""]
    rows, csv_rows = [], []
    for i, c, d in inp.rows_order():
        r = inp.ft(c, "T0", d)
        if not r:
            rows.append([i, c, det_label(d) + " — **FAILED**"] + ["–"] * 6)
            continue
        fs = inp.failed_seeds(c, d)
        rows.append([i, c, det_label(d) + (f" — **failed seed {', '.join(fs)}**" if fs else ""),
                     pm(r.get("a01_bal_acc_mean"), r.get("a01_bal_acc_std")),
                     pm(r.get("a01_tpr_mean"), r.get("a01_tpr_std")),
                     pm(r.get("a01_far_mean"), r.get("a01_far_std")),
                     pm(r.get("a01_det_time_mean"), r.get("a01_det_time_std")),
                     f"{f(r.get('n_cal_mean')):.1f}" if np.isfinite(f(r.get("n_cal_mean"))) else "–",
                     f"{f(r.get('n_mean')):.0f}" if np.isfinite(f(r.get("n_mean"))) else "–"])
        csv_rows.append({"input": i, "pool": c, "detector": d, "test_set": "T0",
                         **{k: v for k, v in r.items() if k.startswith("a01_") or k in ("n_cal_mean", "n_mean", "n_seeds")}})
    write_csv(tdir / "alarm_T0.csv", csv_rows)
    return out + [
        "Balanced accuracy = (failures caught + successes not flagged) / 2; failures caught = share of failed "
        "rollouts flagged before the cut (tpr); false alarms = share of successful rollouts flagged (far); "
        "detection time = first alarm ÷ cut, 1 if never (mean over failures). Calibration rollouts = the "
        "pool's successful eval-seen rollouts (mean over seeds); test rollouts = T0 rollouts. Table: "
        "`report_tables/alarm_T0.csv`.", "",
        md_table(["Input", "Pool", "Model", "Balanced acc.", "Failures caught", "False alarms (succ.)",
                  "Detection time", "Calib. rollouts", "Test rollouts"], rows), ""]


# --- 4. paired --------------------------------------------------------------------------------------
PAIRED_COLS = ("family", "input", "pool", "depth", "cond_x", "cond_y", "detector_x", "detector_y", "metric",
               "n_seeds", "n_boot", "n_valid", "y_point", "x_point", "diff_point", "diff_mean", "diff_ci_lo",
               "diff_ci_hi", "improvement", "significant_worsening", "verdict", "x_point_per_seed",
               "y_point_per_seed")
MET_ORDER = {"m1_per_task": 0, "bal_acc": 1, "tpr": 2}


def paired(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 4. Paired differences on T0 (95% intervals)", ""]
    rows = inp.paired
    if not rows:
        return out + [missing("paired_T0.csv"), ""]
    r0 = rows[0]
    out += [f"Paired bootstrap as in abc2: {r0.get('n_boot')} resamples of T0 rollouts within each cell within "
            f"each seed (rng seed {r0.get('rng_seed')}); both sides recomputed on the same resample, then "
            "averaged over seeds. Difference = x − y; shown as the bootstrap mean with its 95% percentile "
            "interval (the point difference on the original sample is in the csv). **Improvement** only if the "
            "whole interval is above zero; an interval entirely below zero is a **significant worsening**."]
    pc = inp.paired_check
    if pc is not None:
        out.append(f"Identity check (`paired_T0_check.json`): on the identity resample the recomputation "
                   f"{'reproduces' if pc.get('passed') else '**does NOT reproduce**'} the eval3 T0 numbers "
                   f"(max |diff| {f(pc.get('max_abs_diff')):.1e}).")
    else:
        out.append(missing("paired_T0_check.json"))
    out.append("")
    key = lambda r: (r.get("input", ""), r.get("pool") or "", depth_of(r) or 0, MET_ORDER.get(r["metric"], 9))  # noqa: E731
    ordered = []
    for fam, title in FAMILIES:
        rs = sorted([r for r in rows if r.get("family") == fam], key=key)
        ordered += rs
        out += [f"**{title}** (`family = {fam}`)", ""]
        if not rs:
            out += ["*no rows*", ""]
            continue
        tab = []
        for r in rs:
            v = r["verdict"]
            v = f"**{v}**" if v in ("improvement", "significant worsening") else v
            pool = r.get("pool") or f"{r['cond_x']} − {r['cond_y']}"
            xy = (f"{r['detector_x']}: {r['cond_x']} − {r['cond_y']}" if r["detector_x"] == r["detector_y"]
                  else f"{r['detector_x']} − {r['detector_y']}")
            tab.append([r.get("input", ""), pool, xy,
                        METRIC_LABEL.get(r["metric"], r["metric"]), n3(r["y_point"]), n3(r["x_point"]), ci(r), v])
        out += [md_table(["Input", "Pool", "x − y", "Metric", "y", "x", "Diff [95% interval]", "Verdict"], tab), ""]
    ordered += [r for r in rows if r.get("family") not in dict(FAMILIES)]
    write_csv(tdir / "paired_T0.csv", [{k: r.get(k, "") for k in PAIRED_COLS} for r in ordered])
    out += ["**The alarm intervals (balanced accuracy, failures caught) hold each seed's conformal band fixed: "
            "only the test rollouts are resampled, not the training or calibration sets, so they understate "
            "the uncertainty.** Seed-to-seed spread of the alarm metrics (section 3) is the better guide to "
            "how much they move. Table: `report_tables/paired_T0.csv`.", ""]
    return out


# --- 5. figures ------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9})
    return plt


def _style(ax):
    ax.set_facecolor(WHITE)
    ax.grid(color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=MUTED, length=0)


def fig_m1_vs_depth(inp: Inputs, path: Path) -> bool:
    if not inp.final:
        return False
    plt = _plt()
    from matplotlib.lines import Line2D
    inputs = [i for i in INPUTS if any(r["detector"].startswith(i + "-") for r in inp.final)]
    fig, axes = plt.subplots(1, len(inputs), figsize=(6.0 * len(inputs), 4.8), facecolor=WHITE, sharey=True,
                             squeeze=False)
    xs = {"linear": 0, "d2": 2, "d6": 6, "d12": 12}
    off = {"D0": -0.18, "D2": 0.18}
    for ax, i in zip(axes[0], inputs):
        _style(ax)
        for c in POOLS:
            for ts in ("train", "eval_seen", "T0"):
                ls, mk = SET_STYLE[ts]
                pts = [(xs[m] + off[c], inp.ft(c, ts, det_name(i, m))) for m in xs]
                pts = [(x, r) for x, r in pts if r and np.isfinite(f(r["m1_per_task_mean"]))]
                if not pts:
                    continue
                x = [p[0] for p in pts]
                m = [f(p[1]["m1_per_task_mean"]) for p in pts]
                s = np.nan_to_num([f(p[1]["m1_per_task_std"]) for p in pts])
                ax.errorbar(x, m, yerr=s, color=POOL_COLOUR[c], linestyle=ls, linewidth=2, marker=mk, markersize=7,
                            markeredgecolor=WHITE, markeredgewidth=1.2, capsize=3, elinewidth=1.1)
            r = inp.ft(c, "T0", det_name(i, "mlp-abc2"))
            if r and np.isfinite(f(r["m1_per_task_mean"])):
                ax.errorbar([2 + off[c] + (0.55 if c == "D2" else -0.55)], [f(r["m1_per_task_mean"])],
                            yerr=[np.nan_to_num(f(r["m1_per_task_std"]))], color=POOL_COLOUR[c], marker="o",
                            markersize=8, markerfacecolor=WHITE, markeredgewidth=1.8, linestyle="none",
                            capsize=3, elinewidth=1.1)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
        ax.set_xticks([0, 2, 6, 12])
        ax.set_xticklabels(["linear", "2", "6", "12"], color=MUTED)
        ax.set_xlim(-1.2, 13.0)
        ax.set_xlabel("Depth (hidden layers of the residual MLP; linear at 0)", color=MUTED)
        ax.set_title(f"{i} input", color=INK, loc="left")
    axes[0][0].set_ylabel("M1-per-task (mean ± std over 3 seeds)", color=MUTED)
    lo = axes[0][0].get_ylim()[0]
    if lo < 0.5:
        axes[0][0].text(-1.1, 0.505, "chance", color=MUTED, fontsize=8.5)
    h = [Line2D([], [], color=POOL_COLOUR[c], linewidth=6, label=f"pool {c}") for c in POOLS]
    h += [Line2D([], [], color=INK, linestyle=SET_STYLE[ts][0], marker=SET_STYLE[ts][1], linewidth=1.6,
                 label={"train": "train", "eval_seen": "eval-seen (optimistic)", "T0": "T0 (headline)"}[ts])
          for ts in ("train", "eval_seen", "T0")]
    h.append(Line2D([], [], color=INK, marker="o", markerfacecolor=WHITE, linestyle="none", markersize=8,
                    markeredgewidth=1.6, label="abc2 plain 2-layer MLP, T0"))
    fig.legend(handles=h, frameon=False, loc="lower center", ncol=len(h), labelcolor=INK)
    fig.suptitle("M1-per-task against depth: does depth overfit? (train rising while T0 stays flat or falls)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=WHITE)
    plt.close(fig)
    return True


def curve_rows(inp: Inputs, inp_name="C", cond="D2", seed="0") -> dict:
    out = {}
    for d in DEPTHS:
        rs = [r for r in inp.log if r["input"] == inp_name and r["cond"] == cond and str(as_int(r["seed"])) == seed
              and as_int(r["depth"]) == d and is_true(r["chosen"])]
        if rs:
            out[d] = sorted(rs, key=lambda r: as_int(r["epoch"], 0))
    return out


def fig_training_curves(inp: Inputs, path: Path, tdir: Path) -> tuple[bool, dict]:
    curves = curve_rows(inp)
    if not curves:
        return False, {}
    write_csv(tdir / "training_curves.csv", [r for d in curves for r in curves[d]])
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=WHITE)
    best = {}
    for d, rs in curves.items():
        e = np.array([as_int(r["epoch"]) for r in rs])
        loss = np.array([f(r["train_loss"]) for r in rs])
        ev = np.array([f(r["eval_seen_m1_per_task"]) for r in rs])
        tr = np.array([f(r.get("train_m1_per_task")) for r in rs])
        col = DEPTH_COLOUR[d]
        axes[0].plot(e, loss, color=col, linewidth=2, marker="o", markersize=4)
        axes[1].plot(e, ev, color=col, linewidth=2, marker="o", markersize=4)
        if np.isfinite(tr).any():
            axes[1].plot(e, tr, color=col, linewidth=1.6, linestyle=":")
        sr = [r for r in inp.sel_rows("D2", f"C-d{d}") if r["seed"] == "0"]
        be = as_int(sr[0]["best_epoch"]) if sr else (int(e[np.nanargmax(ev)]) if np.isfinite(ev).any() else None)
        if be is not None and be in e:
            k = int(np.where(e == be)[0][0])
            axes[1].plot([be], [ev[k]], marker="*", markersize=15, color=col, markeredgecolor=INK,
                         markeredgewidth=0.8, linestyle="none", zorder=5)
            best[d] = (be, len(e))
    for ax in axes:
        _style(ax)
        ax.set_xlabel("Epoch", color=MUTED)
    axes[0].set_ylabel("Train loss (row-weighted BCE)", color=MUTED)
    axes[0].set_title("Train loss", color=INK, loc="left")
    axes[1].set_ylabel("M1-per-task", color=MUTED)
    axes[1].set_title("M1-per-task: eval-seen (solid) and train (dotted)", color=INK, loc="left")
    regs = {}
    for d in curves:
        r0 = curves[d][0]
        regs[d] = f"p {num(r0['p'])}, wd {num(r0['wd'])}"
    h = [Line2D([], [], color=DEPTH_COLOUR[d], linewidth=2.5, label=f"depth {d} ({regs[d]})") for d in curves]
    h += [Line2D([], [], color=INK, linewidth=2, label="eval-seen"),
          Line2D([], [], color=INK, linewidth=1.6, linestyle=":", label="train"),
          Line2D([], [], color=INK, marker="*", markersize=12, linestyle="none", label="best epoch (kept)")]
    fig.legend(handles=h, frameon=False, loc="lower center", ncol=len(h), labelcolor=INK)
    fig.suptitle("Training curves of the chosen fits: C input, pool D2, seed 0", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=WHITE)
    plt.close(fig)
    return True, best


def figures(inp: Inputs, out_dir: Path, tdir: Path) -> list[str]:
    fdir = out_dir / "report_figures"
    fdir.mkdir(parents=True, exist_ok=True)
    out = ["## 5. Figures", ""]
    if fig_m1_vs_depth(inp, fdir / "m1_vs_depth.png"):
        out += ["**M1-per-task against depth** (linear at depth 0; D0 blue, D2 orange; line style = test set; "
                "hollow circles = abc2's plain 2-layer MLP on T0, offset to the side of depth 2; error bars = std "
                "over seeds). Depth overfits if train rises with depth while T0 stays flat or falls. Data: "
                "`report_tables/depth.csv`.", "",
                "![M1-per-task against depth](report_figures/m1_vs_depth.png)", ""]
    else:
        out += ["*M1-per-task against depth: not drawn* " + missing("final_table.csv"), ""]
    ok, best = fig_training_curves(inp, fdir / "training_curves.png", tdir)
    if ok:
        bt = ", ".join(f"depth {d}: best epoch {b} of {n} run" for d, (b, n) in best.items())
        out += ["**Training curves**, C input, D2, seed 0, the chosen regularisation for each depth (legend). "
                f"Early stopping on eval-seen M1-per-task, patience 5 ({bt}). Data: "
                "`report_tables/training_curves.csv`.", "",
                "![Training curves](report_figures/training_curves.png)", ""]
    else:
        out += ["*Training curves: not drawn* " + missing("train_log.csv") + " (or no chosen C / D2 / seed-0 fit).",
                ""]
    return out


# --- 6. stress -------------------------------------------------------------------------------------
def stress(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 6. Stress set: share flagged before the cut (α = 0.1, mean ± std over seeds)", ""]
    rs = [r for r in inp.stress if abs(f(r.get("alpha", 0.1)) - 0.1) < 1e-9]
    if not rs:
        return out + [missing("stress.csv"), ""]
    cols = list(dict.fromkeys((r["stress_set"], r["block"]) for r in rs))
    cols.sort(key=lambda k: (k[0], k[1] != "all", k[1]))
    agg = {}
    for r in rs:
        agg.setdefault((r["cond"], r["detector"], r["stress_set"], r["block"]), []).append(r)
    rows, csv_rows = [], []
    order = [(i, c, d) for i in INPUTS for c in POOLS for d in (det_name(i, m) for m in MODELS)]
    for i, c, d in order:
        if not any(k[0] == c and k[1] == d for k in agg):
            continue
        line = [i, c, det_label(d)]
        for k in cols:
            g = agg.get((c, d) + k, [])
            v = np.array([f(q["share_alarm_before_cut"]) for q in g])
            v = v[np.isfinite(v)]
            if len(v) == 0:
                line.append("–")
                continue
            m, s = v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))
            line.append(pm(m, s))
            csv_rows.append({"input": i, "pool": c, "detector": d, "stress_set": k[0], "block": k[1], "alpha": 0.1,
                             "n_seeds": len(v), "n_mean": np.mean([f(q["n"]) for q in g]),
                             "n_fail_mean": np.mean([f(q["n_fail"]) for q in g]),
                             "share_alarm_before_cut_mean": m, "share_alarm_before_cut_std": s,
                             "per_seed": " ".join(f"{x:.6g}" for x in v)})
        rows.append(line)
    nmean = {k: np.mean([f(r["n"]) for r in rs if (r["stress_set"], r["block"]) == k]) for k in cols}
    head = [f"{s.replace('stress_', '')} · {b} (n ≈ {nmean[(s, b)]:.0f})" for s, b in cols]
    write_csv(tdir / "stress.csv", csv_rows)
    return out + [
        "`seen` / `unseen`: stress rollouts whose base task is seen / unseen in that seed's split; `all` = every "
        "stress block, then per block. A rollout's cut is its base task's unperturbed-cell cut. Nearly all stress "
        "rollouts fail, so this is close to the share of failures caught in time. Table: `report_tables/stress.csv`.",
        "", md_table(["Input", "Pool", "Model"] + head, rows), ""]


# --- 7. checks, grid health, regularisation ---------------------------------------------------------
def med_range(v: list) -> str:
    v = [x for x in v if x is not None]
    return f"{int(np.median(v))} [{min(v)}–{max(v)}]" if v else "–"


WARMUP_EPOCHS = 0.05 * 50   # common3.HP: warm-up = 5% of the steps of the 50-epoch budget (2.5 epochs)
EPOCH_BINS = ((1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50))


def checks(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 7. Checks, grid health and the regularisation chosen", ""]
    # sanity (§3)
    s = inp.sanity
    if s is None:
        out += ["**§3 sanity check:** " + missing("sanity_check.json"), ""]
    else:
        out += [f"**§3 sanity check: {'PASSED' if s.get('passed') else 'FAILED'}.** {s.get('check', '')}, wd "
                f"{s.get('wd')}: T0 M1-per-task {n3(s.get('T0_m1_per_task'))} against abc2's "
                f"{n3(s.get('abc2_T0_m1_per_task'))} (diff {f(s.get('diff')):+.1e}, tolerance ± {s.get('tolerance')}); "
                f"best epoch {s.get('best_epoch')} (abc2 {s.get('abc2_best_epoch')}).", ""]
    p = inp.precision
    if p is None:
        out += ["**Precision (bf16 check, §2):** " + missing("precision.json"), ""]
    else:
        fp, bf = p.get("fp32", {}), p.get("bf16", {})
        out += [f"**Precision (§2): bf16 check {'passed' if p.get('passed') else 'FAILED'} → grid run in "
                f"{p.get('precision')}.** {p.get('check', '')}: eval-seen M1-per-task fp32 {n3(fp.get('best_score'))} "
                f"(best epoch {fp.get('best_epoch')}) vs bf16 {n3(bf.get('best_score'))} (best epoch "
                f"{bf.get('best_epoch')}); difference {f(p.get('diff_eval_seen')):+.3f}, tolerance "
                f"{p.get('tolerance')}.", ""]
    ec = inp.eval_check
    if ec is not None:
        out += [f"**Reference rows (`eval_check.json`):** {'passed' if ec.get('passed') else '**FAILED**'} — linear "
                f"and abc2-MLP rows recomputed from abc2's score arrays match abc2's tables (max |diff| "
                f"{f(ec.get('max_abs_diff')):.1e}).", ""]
    es = inp.eval_status
    if es is not None and es.get("groups_failed"):
        out += [f"**Groups failed** (all four regularisation fits failed, `eval_status.json`): "
                f"{', '.join(map(str, es['groups_failed']))}.", ""]

    # grid health
    g = inp.grid
    out += ["### Grid health", ""]
    if not g:
        out += [missing("grid.csv"), ""]
    else:
        def summ(rs, i, d):
            be = [as_int(r["best_epoch"]) for r in rs if r.get("status") == "ok"]
            bc = [as_int(r["best_epoch"]) for r in rs if r.get("status") == "ok" and is_true(r["chosen"])]
            er = [as_int(r["epochs_run"]) for r in rs if r.get("status") == "ok"]
            retried = [r for r in rs if (as_int(r.get("n_attempts"), 1) or 1) > 1 or abs(f(r.get("lr")) - 3e-4) < 1e-9]
            se = [is_true(r["stopped_early"]) for r in rs if r.get("status") == "ok"]
            return {"input": i, "depth": d, "fits": len(rs), "ok": sum(r.get("status") == "ok" for r in rs),
                    "failed": sum(r.get("status") != "ok" for r in rs),
                    "diverged": sum(is_true(r.get("diverged")) for r in rs),
                    "retried_lr3e-4": len(retried),
                    "retried_ok": sum(r.get("status") == "ok" for r in retried),
                    "stopped_early_share": (np.mean(se) if se else float("nan")),
                    "best_epoch_median": np.median(be) if be else "", "best_epoch_min": min(be) if be else "",
                    "best_epoch_max": max(be) if be else "",
                    "chosen_best_epoch_median": np.median(bc) if bc else "",
                    "chosen_best_epoch_min": min(bc) if bc else "", "chosen_best_epoch_max": max(bc) if bc else "",
                    "epochs_run_median": np.median(er) if er else "",
                    "hit_epoch_cap": sum(1 for x in er if x is not None and x >= 50),
                    "seconds_mean": np.nanmean([f(r.get("seconds")) for r in rs]) if rs else "",
                    "_be": be, "_bc": bc, "_er": er}
        recs = []
        for i in INPUTS:
            for d in DEPTHS:
                rs = [r for r in g if r["input"] == i and as_int(r["depth"]) == d]
                if rs:
                    recs.append(summ(rs, i, d))
        recs.append(summ(g, "all", "all"))
        rows = [[r["input"], r["depth"], r["fits"], r["ok"], r["failed"], r["diverged"],
                 f"{r['retried_lr3e-4']} ({r['retried_ok']} ok)", n3(r["stopped_early_share"]),
                 med_range(r["_be"]), med_range(r["_bc"]), med_range(r["_er"]),
                 f"{f(r['seconds_mean']):.1f}"] for r in recs]
        write_csv(tdir / "grid_health.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs])
        hist = []
        for lo, hi in EPOCH_BINS:
            lab = str(lo) if lo == hi else f"{lo}–{hi}"
            allb = [as_int(r["best_epoch"]) for r in g if r.get("status") == "ok"]
            chb = [as_int(r["best_epoch"]) for r in g if r.get("status") == "ok" and is_true(r["chosen"])]
            row = {"best_epoch": lab, "all_fits": sum(lo <= x <= hi for x in allb if x is not None),
                   "chosen_fits": sum(lo <= x <= hi for x in chb if x is not None)}
            for d in DEPTHS:
                row[f"chosen_depth{d}"] = sum(1 for r in g if r.get("status") == "ok" and is_true(r["chosen"])
                                              and as_int(r["depth"]) == d and lo <= (as_int(r["best_epoch"]) or 0) <= hi)
            hist.append(row)
        write_csv(tdir / "best_epoch_hist.csv", hist)
        tot = recs[-1]
        n_ch = len(tot["_bc"])
        e1 = sum(1 for x in tot["_bc"] if x == 1)
        e_warm = sum(1 for x in tot["_bc"] if x is not None and x <= WARMUP_EPOCHS)
        out += [f"{tot['fits']} fits: {tot['ok']} ok, {tot['failed']} failed; {tot['diverged']} flagged as diverged "
                f"(loss NaN or eval-seen M1 below 0.5 at every epoch); {tot['retried_lr3e-4']} retried at learning "
                f"rate 3e-4 ({tot['retried_ok']} of them then ok). Share stopped early (patience 5, before the "
                f"50-epoch cap): {n3(tot['stopped_early_share'])}; fits that ran to the cap: {tot['hit_epoch_cap']}. "
                f"Best epoch as median [min–max]. Table: `report_tables/grid_health.csv`.", "",
                md_table(["Input", "Depth", "Fits", "OK", "Failed", "Diverged", "Retried at lr 3e-4",
                          "Stopped early", "Best epoch (all fits)", "Best epoch (chosen)", "Epochs run",
                          "Seconds / fit"], rows), "",
                f"Best-epoch distribution (number of fits; {e1} of the {n_ch} chosen fits peaked at epoch 1"
                + (f"; {e_warm} of {n_ch} at epoch ≤ {int(WARMUP_EPOCHS)}, i.e. inside the learning-rate warm-up, "
                   f"which lasts the first {WARMUP_EPOCHS:g} epochs (5% of the 50-epoch budget)"
                   if n_ch and e_warm > n_ch / 2 else "")
                + "). Table: `report_tables/best_epoch_hist.csv`.", "",
                md_table(["Best epoch", "All fits", "Chosen fits"] + [f"Chosen, depth {d}" for d in DEPTHS],
                         [[h["best_epoch"], h["all_fits"], h["chosen_fits"]] + [h[f"chosen_depth{d}"] for d in DEPTHS]
                          for h in hist]), ""]

    # regularisation chosen
    out += ["### Regularisation chosen (eval-seen M1-per-task, per input × depth × pool × seed)", ""]
    sel = [r for r in inp.sel if r.get("status", "ok") == "ok"]
    if not inp.sel:
        out += [missing("selections.csv"), ""]
    else:
        def key(r):
            return (round(f(r["p"]), 3), round(f(r["wd"]), 4))
        labs = [f"p {p:g} / wd {wd:g}" for p, wd in REG_ORDER]
        rows, csv_rows = [], []
        groups = [(i, d) for i in INPUTS for d in DEPTHS] + [("all", "all")]
        for i, d in groups:
            rs = [r for r in sel if (i == "all" or (r["input"] == i and as_int(r["depth"]) == d))]
            if not rs:
                continue
            cnt = [sum(key(r) == (round(p, 3), round(wd, 4)) for r in rs) for p, wd in REG_ORDER]
            other = len(rs) - sum(cnt)
            rows.append([i, d] + cnt + ([other] if other else []) + [len(rs)])
            csv_rows.append({"input": i, "depth": d, **dict(zip(labs, cnt)), "other": other, "groups": len(rs)})
        any_other = any(r["other"] for r in csv_rows)
        write_csv(tdir / "regularisation.csv", csv_rows)
        nfail = len(inp.sel) - len(sel)
        out += ["Number of groups (input × depth × pool × seed) whose best fit used each setting; ties go to the "
                "first in grid order." + (f" {nfail} failed group(s) are excluded." if nfail else "")
                + " Table: `report_tables/regularisation.csv`.", "",
                md_table(["Input", "Depth"] + labs + (["other"] if any_other else []) + ["Groups"], rows), ""]
    return out


# --- 8. deviations ---------------------------------------------------------------------------------
def deviations(inp: Inputs) -> list[str]:
    out = ["## 8. Deviations and failures", ""]
    if not inp.runlog:
        return out + [missing("RUNLOG.md"), ""]
    ent = runlog_entries(inp.runlog)
    out += [f"RUNLOG.md entries mentioning a deviation, decision, failure, finding, judgement or skip "
            f"({len(ent)}, verbatim):", ""]
    out += ent if ent else ["*none found*"]
    return out + [""]


# --- 9. commit message -----------------------------------------------------------------------------
def new_paths(repo: Path) -> list[str]:
    """Untracked / modified paths under analysis/abc3/, results/abc3/ and deep_mlp_handoff.md, from a
    read-only `git status --porcelain`; files below the first directory level are collapsed to that
    directory with a file count."""
    try:
        res = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "analysis/abc3",
                              "results/abc3", "deep_mlp_handoff.md"], cwd=repo, capture_output=True, text=True,
                             timeout=60)
        lines = res.stdout.splitlines() if res.returncode == 0 else []
    except Exception:
        lines = []
    paths = [ln[3:].strip().strip('"') for ln in lines if len(ln) > 3]
    paths = [p.split(" -> ")[-1] for p in paths if "__pycache__" not in p]
    out, dirs = [], {}
    for p in paths:
        parts = p.split("/")
        if p == "deep_mlp_handoff.md" or len(parts) <= 3:
            out.append(p)
        else:
            d = "/".join(parts[:3]) + "/"
            dirs[d] = dirs.get(d, 0) + 1
    out += [f"{d} ({n} files)" for d, n in dirs.items()]
    # this script's own outputs (absent from git status on a first run)
    for own in ("results/abc3/REPORT.md", "results/abc3/report_tables/", "results/abc3/report_figures/"):
        if not any(p == own or p.startswith(own) for p in out):
            out.append(own)
    return sorted(set(out), key=lambda p: (not p.startswith("deep"), not p.startswith("analysis"), p))


def commit_message(inp: Inputs, hl: dict, repo: Path) -> list[str]:
    body = ["Deep-MLP run: residual MLPs of depth 2/6/12 as A and C detectors on D0/D2", ""]
    body += ["Trained pre-norm residual MLPs (width 256; dropout x weight decay grid chosen on",
             "eval-seen) on A's selected hidden-state layer and on C's final-detector inputs, pools",
             "D0 and D2, three seeds, and compared them with abc2's linear detectors on T0.", ""]
    if inp.final:
        for i in INPUTS:
            parts = []
            for c in POOLS:
                vals = []
                for m in ("linear", "d2", "d6", "d12"):
                    r = inp.ft(c, "T0", det_name(i, m))
                    vals.append(f"{short(det_name(i, m)).replace('depth ', 'd')} {n3(r['m1_per_task_mean'])}" if r
                                else f"{m} n/a")
                parts.append(f"{c}: " + ", ".join(vals))
            body.append(f"T0 M1-per-task, {i}: " + "; ".join(parts))
    if hl.get("deep_vs_d2") is not None:
        (di, dw, dn), (li, lw, ln) = hl["deep_vs_d2"], hl["deep_vs_lin"]
        body.append(f"Paired T0 comparisons, depth 6/12 vs depth 2: {di} of {dn} improve (95% CI above 0), "
                    f"{dw} worse; vs linear: {li} of {ln} improve, {lw} worse.")
    else:
        body.append("Paired comparison not available.")
    body += ["", "New:"]
    np_ = new_paths(repo)
    body += [f"- {p}" for p in np_] if np_ else ["- (git status listed no new paths)"]
    body += ["", CO_AUTHOR]
    return ["## 9. Proposed commit message", "",
            "Nothing has been committed; the user makes the commit. Proposed message:", "",
            "```", *body, "```", ""]


# --- main --------------------------------------------------------------------------------------------
INPUT_FILES = ("final_table.csv", "selections.csv", "grid.csv", "train_log.csv", "stress.csv", "eval_status.json",
               "eval_check.json", "paired_T0.csv", "paired_T0_check.json", "sanity_check.json", "precision.json",
               "RUNLOG.md")


def build(in_dir: Path, out_dir: Path, repo: Path = REPO) -> Path:
    inp = Inputs(in_dir)
    tdir = out_dir / "report_tables"
    tdir.mkdir(parents=True, exist_ok=True)
    miss = [n for n in INPUT_FILES if not (in_dir / n).exists()]
    try:
        src = in_dir.resolve().relative_to(repo)
    except ValueError:
        src = in_dir
    md = ["# Deep-MLP run: does depth help A and C?", "",
          "Residual MLPs with 2, 6 and 12 hidden layers (width 256) trained as A's and C's detectors on pools D0 "
          "and D2, three seeds, compared on T0 with abc2's linear detectors and with each other.", "",
          f"*Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `analysis/abc3/report3.py` from "
          f"`{src}`. Spec: deep_mlp_handoff.md (handoff version 1), section 4; definitions unchanged from abc2. "
          "Numbers to 3 d.p.; ± is the std over seeds.*"
          + (" *Missing inputs: " + ", ".join(f"`{m}`" for m in miss) + ".*" if miss else ""), ""]
    md += top_warnings(inp)
    hl_md, hl = headline(inp)
    md += hl_md
    md += depth_table(inp, tdir)
    md += alarm_table(inp, tdir)
    md += paired(inp, tdir)
    md += figures(inp, out_dir, tdir)
    md += stress(inp, tdir)
    md += checks(inp, tdir)
    md += deviations(inp)
    md += commit_message(inp, hl, repo)
    path = out_dir / "REPORT.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(md))
    shutil.move(tmp, path)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", type=Path, default=REPO / "results" / "abc3")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --in-dir")
    args = ap.parse_args(argv)
    out = args.out_dir or args.in_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = build(args.in_dir, out)
    print(f"[report3] {path} ({path.stat().st_size} bytes, {time.time() - t0:.1f} s); tables in "
          f"{out / 'report_tables'}, figures in {out / 'report_figures'}")


if __name__ == "__main__":
    main()
