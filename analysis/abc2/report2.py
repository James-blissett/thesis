"""
report2.py

Writes the more-data run's report, results/abc2/REPORT.md (more_data_collection.md,
handoff v1, section 5, items 1-11), the CSV behind every table and the figures. Reads
only finished outputs; nothing is recomputed except tables reshaped for reading. Every
conclusion is generated from the numbers (the headline from paired_T0.csv with Step 9's
rule: an improvement only if the 95% interval lies above zero); nothing is hard-coded.

Inputs (<in> = --in-dir, default results/abc2; a missing file is reported, not fatal):
  final_table.csv, cp_curves.csv, stress_summary.csv, selections.csv  (pipeline.py)
  paired_T0.csv (+ paired_T0_check.json, paired_T1/T2.csv if run)     (bootstrap2.py)
  audit_block.csv, audit_block_task.csv, audit.md, demotions.json,
  lan_neardup.json                                                    (build_index.py)
  regression_gate.json (regression_gate.py), RUNLOG.md, HARD_STOP.md (if any),
  perturbation_check/summary.csv and sheet.png, scores/<cond>_seed<s>.npz (pool sizes)
  optional, section 7: mlp_final_table.csv, mlp_paired_T0.csv, b_final_table.csv
  (final_table.csv / paired_T0.csv columns); if absent, {mlp,b}_status.json
  ({"status": ..., "reason": ...}) gives the reason shown.

Outputs (<out> = --out-dir, default <in>):
  REPORT.md
  report_tables/*.csv       one per table in the report (named in the report)
  report_figures/m1_vs_pool.png       M1-per-task against training-pool size, one line
                                      per detector, error bars = std over seeds
                                      (panels T0, T1, T2)
  report_figures/balacc_vs_alpha.png  balanced accuracy against alpha per condition,
                                      cut variant, T0, mean +- std over seeds
REPORT.md links the figures and results/abc2/perturbation_check/sheet.png by relative
path (out-dir must be <in> or contain them for the links to resolve).

Order at the top of REPORT.md: HARD_STOP.md's content if the file exists; then
**REGRESSION GATE FAILED** with the numbers if regression_gate.json does not say passed;
then a warning if bootstrap2's identity check failed; then items 1-11.

Usage (repo root, CPU, seconds):
    source env.sh
    python analysis/abc2/report2.py
    python analysis/abc2/report2.py --in-dir /data/tmp/x/in --out-dir /data/tmp/x/out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2  # noqa: E402

CONDS = ("D0", "D1", "D2")
DETS = ("emb", "A", "C")
DET_LABEL = {"emb": "emb_temp alone", "A": "A", "C": "C"}
TEST_SETS = ("T0", "T1", "T2")
COLOURS = {"A": "#2a78d6", "C": "#eb6834", "emb": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
METRIC_LABEL = {"m1_per_task": "M1-per-task", "bal_acc": "balanced accuracy, α 0.1 (cut)",
                "tpr": "share of failures caught, α 0.1 (cut)"}
# The words the brief names, in their case (lower-case "decision" would match "decision
# states"); FINDING catches the Step 1.4 finding that changes the lan near-duplicate check.
RUNLOG_PAT = re.compile(r"DEVIATION|Decision|FAIL|[Jj]udge?ment|[Ss]kipped|[Dd]emot|FINDING")
CO_AUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


# --- small helpers ----------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def n3(v, signed=False) -> str:
    x = f(v)
    if not np.isfinite(x):
        return "–"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def pm(mean, std) -> str:
    m, s = f(mean), f(std)
    if not np.isfinite(m):
        return "–"
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def missing(name: str) -> str:
    return f"*Not available: `{name}` is missing.*"


class Inputs:
    def __init__(self, in_dir: Path):
        self.dir = in_dir
        self.final = read_csv(in_dir / "final_table.csv")
        self.cp = read_csv(in_dir / "cp_curves.csv")
        self.stress = read_csv(in_dir / "stress_summary.csv")
        self.sel = read_csv(in_dir / "selections.csv")
        self.paired = {ts: read_csv(in_dir / f"paired_{ts}.csv") for ts in TEST_SETS}
        self.paired_check = read_json(in_dir / "paired_T0_check.json")
        self.audit_block = read_csv(in_dir / "audit_block.csv")
        self.audit_cell = read_csv(in_dir / "audit_block_task.csv")
        self.audit_md = (in_dir / "audit.md").read_text() if (in_dir / "audit.md").exists() else ""
        self.demotions = read_json(in_dir / "demotions.json")
        self.neardup = read_json(in_dir / "lan_neardup.json")
        self.gate = read_json(in_dir / "regression_gate.json")
        self.runlog = (in_dir / "RUNLOG.md").read_text() if (in_dir / "RUNLOG.md").exists() else ""
        self.hard_stop = (in_dir / "HARD_STOP.md").read_text() if (in_dir / "HARD_STOP.md").exists() else ""
        self.pert = read_csv(in_dir / "perturbation_check" / "summary.csv")
        self.conds = [c for c in CONDS if any(r["cond"] == c for r in self.final)]

    def frow(self, cond, ts, det, table=None):
        for r in (table if table is not None else self.final):
            if r["cond"] == cond and r["test_set"] == ts and r["detector"] == det:
                return r
        return None

    def emb_layer(self, cond) -> str:
        v = {r["value"] for r in self.sel if r["cond"] == cond and r["selection"] == "emb_layer"}
        return ", ".join(sorted(v)) if v else ""


# --- pool sizes (from the data) -----------------------------------------------------------
def pool_sizes(inp: Inputs) -> dict:
    """{cond: {'pool': rollouts in the condition's blocks, 'train': mean train size,
    'blocks': [...]}} read from scores/<cond>_seed<s>.npz (blocks = those in train or
    eval_seen). Falls back to the demotions.json counts and data2.CONDITIONS."""
    out = {}
    for cond in inp.conds:
        files = sorted((inp.dir / "scores").glob(f"{cond}_seed?.npz"))
        if files:
            pools, trains, blocks = [], [], set()
            for p in files:
                z = np.load(p, allow_pickle=False)
                rids, blk = z["rollout_ids"].astype(str), z["block"].astype(str)
                used = np.isin(rids, np.concatenate([z["train_rids"].astype(str),
                                                     z["eval_seen_rids"].astype(str)]))
                b = sorted(set(blk[used]))
                blocks |= set(b)
                pools.append(int(np.isin(blk, b).sum()))
                trains.append(len(z["train_rids"]))
            out[cond] = {"pool": int(np.median(pools)), "train": float(np.mean(trains)),
                         "blocks": [b for b in data2.BLOCKS if b in blocks]}
        else:
            dem = tuple((inp.demotions or {}).get("demoted", []))
            counts = (inp.demotions or {}).get("counts", {})
            bl = [b for b in data2.CONDITIONS[cond] if b not in dem]
            out[cond] = {"pool": sum(counts.get(b, {}).get("n", 0) for b in bl) or None,
                         "train": float("nan"), "blocks": bl}
    return out


# --- 0. top warnings ----------------------------------------------------------------------
def top_warnings(inp: Inputs) -> list[str]:
    out = []
    if inp.hard_stop:
        out += ["## HARD STOP", "", inp.hard_stop.strip(), ""]
    g = inp.gate
    if g is None:
        out += ["**REGRESSION GATE RESULT MISSING** (`regression_gate.json` not found): the D0/T0 "
                "reproduction of results/abc was not confirmed.", ""]
    elif not g.get("passed"):
        out += ["**REGRESSION GATE FAILED**", "", gate_table(g), ""]
    pc = inp.paired_check
    if pc is not None and not pc.get("passed"):
        out += [f"**BOOTSTRAP IDENTITY CHECK FAILED** (max |diff| {pc.get('max_abs_diff')}): the paired "
                "comparison's recomputation does not reproduce the pipeline's T0 numbers; treat "
                "section 5 with caution.", ""]
    return out


def gate_table(g: dict) -> str:
    rows = [[r["quantity"], n3(r.get("reference")), n3(r.get("got")), n3(r.get("diff"), True),
             f"± {r.get('tolerance')}", "pass" if r.get("pass") else "**FAIL**"] for r in g.get("gate", [])]
    return md_table(["Quantity (D0, T0, mean over seeds)", "Reference", "Got", "Diff", "Tolerance", ""], rows)


# --- 1. headline --------------------------------------------------------------------------
def describe(rows: list[dict]) -> str:
    """One clause for a set of paired rows (one comparison, one or more metrics)."""
    imp = [r for r in rows if r["improvement"] == "True"]
    wor = [r for r in rows if r["significant_worsening"] == "True"]
    lab = lambda r: (f"{DET_LABEL.get(r['detector'], r['detector'])} "  # noqa: E731
                     f"({n3(r['diff_point'], True)} [{n3(r['diff_ci_lo'], True)}, {n3(r['diff_ci_hi'], True)}])")
    parts = []
    if imp:
        parts.append("improved for " + ", ".join(lab(r) for r in imp))
    if wor:
        parts.append("was significantly worse for " + ", ".join(lab(r) for r in wor))
    if not parts:
        d = [f(r["diff_point"]) for r in rows]
        d = [x for x in d if np.isfinite(x)]
        rng = f" (point differences {min(d):+.3f} to {max(d):+.3f})" if d else ""
        parts.append("showed no significant change for any detector" + rng)
    return "; ".join(parts)


def n_worse(k: int) -> str:
    return f"{k} {'is a significant worsening' if k == 1 else 'are significant worsenings'}"


def headline(inp: Inputs, pools: dict) -> list[str]:
    rows = inp.paired["T0"]
    out = ["## 1. Headline", ""]
    if not rows:
        return out + ["The paired comparison on T0 is not available (`paired_T0.csv` is missing), so "
                      "no claim about D1 or D2 against D0 can be made.", ""]
    comps = list(dict.fromkeys(r["comparison"] for r in rows))
    n_boot = rows[0]["n_boot"]
    s1 = "Ranking (M1-per-task on T0): " + "; ".join(
        f"{c} {describe([r for r in rows if r['comparison'] == c and r['metric'] == 'm1_per_task'])}"
        for c in comps) + "."
    alarm = []
    for c in comps:
        for m in ("bal_acc", "tpr"):
            alarm.append(f"{c} {METRIC_LABEL[m].split(',')[0]} "
                         + describe([r for r in rows if r["comparison"] == c and r["metric"] == m]))
    s2 = "Alarm at α = 0.1 (cut variant): " + "; ".join(alarm) + "."
    imp = [r for r in rows if r["improvement"] == "True"]
    wor = [r for r in rows if r["significant_worsening"] == "True"]
    conds_cmp = sorted({r["cond_x"] for r in rows})
    known = [c for c in ["D0"] + conds_cmp if pools.get(c, {}).get("pool")]
    pools_txt = ("training pools of " + ", ".join(f"{c} {pools[c]['pool']}" for c in known) + " rollouts"
                 if known else "training pools " + ", ".join(["D0"] + conds_cmp))
    if not imp and not wor:
        s3 = (f"**Null result:** by Step 9's rule (an improvement only if the 95% interval of the paired "
              f"difference excludes zero; {n_boot} resamples), more data ({pools_txt}) did not improve "
              f"any detector on T0 in any of the {len(rows)} comparisons, and made none significantly worse.")
    elif not imp:
        s3 = (f"**No improvement:** by Step 9's rule, none of the {len(rows)} comparisons on T0 "
              f"({pools_txt}) is an improvement; {n_worse(len(wor))}.")
    else:
        s3 = (f"By Step 9's rule, {len(imp)} of {len(rows)} comparisons on T0 ({pools_txt}) "
              f"{'is an improvement' if len(imp) == 1 else 'are improvements'} (" + ", ".join(f"{r['comparison']} {DET_LABEL.get(r['detector'], r['detector'])} "
                                            f"{METRIC_LABEL[r['metric']].split(',')[0]}" for r in imp)
              + f"){'' if not wor else ' and ' + n_worse(len(wor))}; "
              f"the rest show no significant change.")
    return out + [s3, "", s1, "", s2, "",
                  "*Differences are x − y (positive = the larger pool is better); "
                  "point difference on the original sample with the bootstrap 95% percentile interval.*", ""] \
        + lead_notes(inp)


def lead_notes(inp: Inputs) -> list[str]:
    """The lead's reading notes (results/abc2/lead_notes.md), placed under the headline."""
    p = inp.dir / "lead_notes.md"
    return [p.read_text().rstrip(), ""] if p.exists() else []


# --- 2. corpus ----------------------------------------------------------------------------
def audit_timing(md: str) -> dict:
    t = {}
    m = re.search(r"started (\S+), last rollout finished (\S+)", md)
    if m:
        t["started"], t["last"] = m.group(1), m.group(2)
    m = re.search(r"- hours (\S+), rollouts finished in that window (\S+) \(\+ (\S+) smoke-test "
                  r"rollouts before the start\), rollouts/hour (\S+)", md)
    if m:
        t["hours"], t["n_window"], t["n_smoke"], t["per_hour"] = m.groups()
    m = re.search(r"COLLECT_HOURS_MAX (\S+) h; cap reached[^:]*: (\w+)", md)
    if m:
        t["cap"], t["cap_reached"] = m.groups()
    m = re.search(r"collection stops logged \(disk headroom\): (\d+)", md)
    if m:
        t["disk_stops"] = m.group(1)
    m = re.search(r"failures.jsonl entries: (\d+)", md)
    if m:
        t["failures_logged"] = m.group(1)
    return t


def corpus(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 2. Rollouts collected", ""]
    if not inp.audit_block:
        return out + [missing("audit_block.csv"), ""]
    dem = inp.demotions or {}
    demoted = dem.get("demoted", [])
    rows, csv_rows = [], []
    for r in inp.audit_block:
        role = r["role"] + (" → stress (demoted)" if r["block"] in demoted else "")
        rows.append([r["block"], role, r["planned"], r["collected"], f"{r['successes']}/{r['collected']}",
                     n3(r["success_rate"]), r["failed_rollouts"], r["not_run"], r["cells_skipped"]])
        csv_rows.append({**r, "demoted": int(r["block"] in demoted)})
    write_csv(tdir / "corpus_block.csv", csv_rows)
    out += ["Per block (block `ori` = the existing 300 rollouts). `failed`: raised twice and logged; "
            "`not run`: planned but never attempted. Table: `report_tables/corpus_block.csv`.", "",
            md_table(["Block", "Role", "Planned", "Collected", "Successes / total", "Success rate",
                      "Failed", "Not run", "Cells skipped"], rows), ""]

    blocks = [b for b in data2.BLOCKS if any(r["block"] == b for r in inp.audit_cell)]
    grid = []
    for t in range(10):
        line = [str(t)]
        for b in blocks:
            r = next((q for q in inp.audit_cell if q["block"] == b and int(q["task"]) == t), None)
            if r is None:
                line.append("–")
                continue
            cell = f"{r['successes']}/{r['collected']}"
            if r["collected"] != r["planned"]:
                cell += f" (of {r['planned']})"
            if r["cell_skipped"] == "1":
                cell += " **skipped**"
            line.append(cell)
        grid.append(line)
    write_csv(tdir / "corpus_block_task.csv", inp.audit_cell)
    out += ["Per (block, task): successes / collected, with the plan where it was not met. "
            "Table: `report_tables/corpus_block_task.csv`.", "",
            md_table(["Task"] + blocks, grid), ""]

    t = audit_timing(inp.audit_md)
    skipped = [f"({r['block']}, task {r['task']})" for r in inp.audit_cell if r.get("cell_skipped") == "1"]
    rates = dem.get("rates", {})
    out += ["**Collection.** "
            + (f"Started {t.get('started')}, last rollout finished {t.get('last')}: **{t.get('hours')} h** "
               f"({t.get('n_window')} rollouts in that window, {t.get('per_hour')}/h; plus {t.get('n_smoke')} "
               f"smoke-test rollouts before the start). COLLECT_HOURS_MAX {t.get('cap')} h, reached: "
               f"{t.get('cap_reached')}. Disk-headroom stops: {t.get('disk_stops')}. Failures logged: "
               f"{t.get('failures_logged')}." if t else "*Hours not available (`audit.md` missing).*"),
            "",
            f"**Skipped cells** (3 failures in a row): {', '.join(skipped) if skipped else 'none'}.", "",
            "**Demotions** (Step 5: `obj` or `lan` with overall success below 10% or above 90% becomes "
            f"stress-test only): {', '.join(demoted) if demoted else 'none'}"
            + (f" (success rates: obj {n3(rates.get('obj'))}, lan {n3(rates.get('lan'))})." if rates else "."),
            ""]
    nd = inp.neardup
    if nd:
        p, ref, nonp = nd.get("lan_vs_partner", {}), nd.get("reference", {}), nd.get("lan_vs_nonpartner", {})
        out += [f"**`lan` near-duplicate check.** {nd.get('n_with_partner')} of {nd.get('n_lan')} `lan` rollouts "
                f"have an unperturbed partner (same base task and init-state index). Share of the 520 steps "
                f"with all 7 action tokens identical: **mean {n3(p.get('mean'))}, median {n3(p.get('median'))}** "
                f"against the partner; mean {n3(nonp.get('mean'))} against the same task's unperturbed rollouts "
                f"at other init states. **Reference value** (all pairs of unperturbed rollouts of one task at "
                f"different init states, {ref.get('n')} pairs): mean {n3(ref.get('mean'))}, median "
                f"{n3(ref.get('median'))}.", "",
                f"> **Caveat.** {nd.get('caveat', '')}", ""]
        write_csv(tdir / "lan_neardup_summary.csv", [
            {"what": k, **{q: v.get(q) for q in ("n", "mean", "median", "p10", "p90")}}
            for k, v in (("lan_vs_partner", p), ("lan_vs_nonpartner", nonp), ("reference", ref))])
    else:
        out += [missing("lan_neardup.json"), ""]
    return out


# --- 3 / 4. ranking and alarm tables -------------------------------------------------------
def det_name(inp: Inputs, det: str, cond: str) -> str:
    if det == "emb":
        lay = inp.emb_layer(cond)
        return f"emb_temp alone ({lay})" if lay else "emb_temp alone"
    return DET_LABEL.get(det, det)


def ranking(inp: Inputs, tdir: Path, table=None, dets=DETS, csv_name="ranking.csv", title=None) -> list[str]:
    table = inp.final if table is None else table
    out = [title or "## 3. Ranking (M1, mean ± std over seeds)", ""]
    if not table:
        return out + [missing("final_table.csv"), ""]
    conds = [c for c in CONDS if any(r["cond"] == c for r in table)]
    dets = [d for d in dets if any(r["detector"] == d for r in table)]
    rows, csv_rows = [], []
    for d in dets:
        for c in conds:
            line = [det_name(inp, d, c) if table is inp.final else d, c]
            rec = {"detector": d, "cond": c}
            for ts in TEST_SETS:
                r = inp.frow(c, ts, d, table)
                line.append(pm(r["m1_per_task_mean"], r["m1_per_task_std"]) if r else "–")
                for k in ("m1_per_task_mean", "m1_per_task_std", "m1_pooled_mean", "m1_pooled_std",
                          "floor_mean", "n_cells", "n_mean"):
                    rec[f"{ts}_{k}"] = r.get(k) if r else ""
            r0 = inp.frow(c, "T0", d, table)
            line += [pm(r0["m1_pooled_mean"], r0["m1_pooled_std"]) if r0 else "–",
                     n3(r0["floor_mean"]) if r0 else "–"]
            line.append(" / ".join((inp.frow(c, ts, d, table) or {}).get("n_cells", "–") or "–"
                                   for ts in TEST_SETS))
            rows.append(line)
            csv_rows.append(rec)
    write_csv(tdir / csv_name, csv_rows)
    return out + [
        f"M1-per-task = within-cell ROC-AUC averaged over cells with ≥ 2 rollouts of each class (the "
        f"number claims rest on). Pooled M1 and the timestep-only floor on T0. Cells averaged per seed "
        f"for T0 / T1 / T2. Table: `report_tables/{csv_name}`.", "",
        md_table(["Detector", "Pool", "M1/task T0", "M1/task T1", "M1/task T2", "M1 pooled T0",
                  "Floor T0", "Cells per seed (T0 / T1 / T2)"], rows), ""]


def alarm(inp: Inputs, tdir: Path, table=None, dets=DETS, csv_name="alarm.csv", title=None) -> list[str]:
    table = inp.final if table is None else table
    out = [title or "## 4. Alarm at α = 0.1 (cut variant, mean ± std over seeds)", ""]
    if not table:
        return out + [missing("final_table.csv"), ""]
    conds = [c for c in CONDS if any(r["cond"] == c for r in table)]
    dets = [d for d in dets if any(r["detector"] == d for r in table)]
    csv_rows = []
    out.append("Balanced accuracy = (share of failures caught + share of successes not flagged) / 2; "
               "false-alarm rate on successes; detection time = first alarm ÷ cut, 1 if never (mean over "
               f"failures); calibration rollouts = the condition's successful eval-seen rollouts. Table: "
               f"`report_tables/{csv_name}`.")
    out.append("")
    for ts in TEST_SETS:
        rows = []
        for d in dets:
            for c in conds:
                r = inp.frow(c, ts, d, table)
                if not r:
                    continue
                rows.append([det_name(inp, d, c) if table is inp.final else d, c,
                             pm(r["a01_bal_acc_mean"], r["a01_bal_acc_std"]),
                             pm(r["a01_tpr_mean"], r["a01_tpr_std"]), pm(r["a01_far_mean"], r["a01_far_std"]),
                             pm(r["a01_det_time_mean"], r["a01_det_time_std"]), f"{f(r['n_cal_mean']):.1f}",
                             f"{f(r.get('n_mean')):.0f}" if r.get("n_mean") else "–"])
                csv_rows.append({"test_set": ts, "detector": d, "cond": c,
                                 **{k: r.get(k) for k in r if k.startswith("a01_") or k in ("n_cal_mean", "n_mean")}})
        if rows:
            out += [f"**{ts}**", "", md_table(["Detector", "Pool", "Balanced acc.", "Failures caught",
                                               "False alarms (succ.)", "Detection time", "Calib. rollouts",
                                               "Test rollouts"], rows), ""]
    write_csv(tdir / csv_name, csv_rows)
    return out


# --- 5. paired ----------------------------------------------------------------------------
def paired_section(inp: Inputs, tdir: Path, rows_by_ts: dict, title: str, csv_prefix: str,
                   check=None) -> list[str]:
    out = [title, ""]
    if not any(rows_by_ts.values()):
        return out + [missing(f"{csv_prefix}_T0.csv"), ""]
    for ts, rows in rows_by_ts.items():
        if not rows:
            continue
        r0 = rows[0]
        if ts == "T0":
            out.append(f"Paired bootstrap ({r0['n_boot']} resamples of rollouts within each cell within each "
                       f"seed, rng seed {r0['rng_seed']}; both sides recomputed on the same resample, then "
                       f"averaged over seeds). Difference = x − y. Verdict by Step 9's rule: improvement only "
                       f"if the 95% interval lies above zero; an interval below zero is a significant "
                       f"worsening.")
            if check is not None:
                out.append(f"Correctness check: on the identity resample the recomputation "
                           f"{'reproduces' if check.get('passed') else '**does NOT reproduce**'} the "
                           f"pipeline's per-seed and mean T0 numbers (max |diff| "
                           f"{f(check.get('max_abs_diff')):.1e}, tolerance {check.get('tolerance')}).")
            out.append("")
        table = []
        for r in rows:
            v = r["verdict"]
            v = f"**{v}**" if v in ("improvement", "significant worsening") else v
            table.append([r["comparison"], DET_LABEL.get(r["detector"], r["detector"]),
                          METRIC_LABEL.get(r["metric"], r["metric"]), n3(r["y_point"]), n3(r["x_point"]),
                          n3(r["diff_point"], True), n3(r["diff_mean"], True),
                          f"[{n3(r['diff_ci_lo'], True)}, {n3(r['diff_ci_hi'], True)}]", v])
        name = f"{csv_prefix}_{ts}.csv"
        write_csv(tdir / name, rows)
        out += [f"**{ts}** (`report_tables/{name}`)", "",
                md_table(["x − y", "Detector", "Metric", "y", "x", "Point diff", "Boot. mean diff",
                          "95% interval", "Verdict"], table), ""]
    return out


# --- 6. stress ----------------------------------------------------------------------------
def stress(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 6. Stress test (share flagged before the cut, α = 0.1, mean ± std over seeds)", ""]
    if not inp.stress:
        return out + [missing("stress_summary.csv") + " (or no stress rollouts)", ""]
    cols = list(dict.fromkeys((r["stress_set"], r["block"]) for r in inp.stress))
    cols.sort(key=lambda k: (k[0], k[1] != "all", k[1]))
    rows, csv_rows = [], []
    for d in DETS:
        for c in CONDS:
            rs = {(r["stress_set"], r["block"]): r for r in inp.stress if r["detector"] == d and r["cond"] == c}
            if not rs:
                continue
            rows.append([det_name(inp, d, c), c] + [
                pm(rs[k]["share_alarm_before_cut_mean"], rs[k]["share_alarm_before_cut_std"]) if k in rs else "–"
                for k in cols])
            csv_rows += [dict(r) for r in rs.values()]
    nmean = {k: next((r["n_mean"] for r in inp.stress if (r["stress_set"], r["block"]) == k), "")
             for k in cols}
    head = [f"{s.replace('stress_', '')} {b} (n {f(nmean[(s, b)]):.0f})" for s, b in cols]
    write_csv(tdir / "stress.csv", csv_rows)
    dem = (inp.demotions or {}).get("demoted", [])
    return out + [
        "`seen` / `unseen`: stress rollouts whose base task is seen / unseen in that seed's split. "
        "A rollout's cut is its base task's unperturbed-cell cut (eval-seen for seen tasks, T1 for unseen). "
        "Nearly all stress rollouts fail, so this is the share of (mostly) failures flagged in time"
        + (f"; demoted blocks ({', '.join(dem)}) are reported here too" if dem else "")
        + ". Table: `report_tables/stress.csv`.", "",
        md_table(["Detector", "Pool"] + head, rows), ""]


# --- 7. MLP and B ---------------------------------------------------------------------------
def optional_section(inp: Inputs, tdir: Path, tag: str, title: str) -> list[str]:
    out = [title, ""]
    ft = read_csv(inp.dir / f"{tag}_final_table.csv")
    if not ft:
        st = read_json(inp.dir / f"{tag}_status.json") or {}
        why = st.get("reason") or "no output and no status file was found"
        return out + [f"**Not run** ({st.get('status', 'no output')}): {why}.", ""]
    dets = list(dict.fromkeys(r["detector"] for r in ft))
    out += ranking(inp, tdir, table=ft, dets=dets, csv_name=f"{tag}_ranking.csv",
                   title=f"**{tag.upper()}: ranking (M1)**")
    out += alarm(inp, tdir, table=ft, dets=dets, csv_name=f"{tag}_alarm.csv",
                 title=f"**{tag.upper()}: alarm at α = 0.1 (cut)**")
    pr = {ts: read_csv(inp.dir / f"{tag}_paired_{ts}.csv") for ts in TEST_SETS}
    if any(pr.values()):
        out += paired_section(inp, tdir, pr, f"**{tag.upper()}: paired differences**", f"{tag}_paired",
                              read_json(inp.dir / f"{tag}_paired_T0_check.json"))
    return out


# --- 8. figures -------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def fig_m1_vs_pool(inp: Inputs, pools: dict, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in inp.conds if pools.get(c, {}).get("pool")]
    if not conds:
        return False
    sets = [ts for ts in TEST_SETS if any(r["test_set"] == ts for r in inp.final)]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.4 * len(sets), 4.2), facecolor=SURFACE, sharey=True,
                             squeeze=False)
    xs = [pools[c]["pool"] for c in conds]
    span = max(xs) - min(xs) or 1
    for ax, ts in zip(axes[0], sets):
        _style(ax)
        for k, d in enumerate(DETS):
            pts = [(pools[c]["pool"], inp.frow(c, ts, d)) for c in conds]
            pts = [(x, r) for x, r in pts if r]
            if not pts:
                continue
            off = (k - 1) * 0.012 * span           # small dodge so error bars don't overlap
            x = np.array([p[0] for p in pts]) + off
            m = np.array([f(p[1]["m1_per_task_mean"]) for p in pts])
            s = np.nan_to_num(np.array([f(p[1]["m1_per_task_std"]) for p in pts]))
            ax.errorbar(x, m, yerr=s, color=COLOURS[d], linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3, elinewidth=1.2,
                        label=DET_LABEL[d])
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c}\n{pools[c]['pool']}" for c in conds], color=MUTED)
        ax.set_title(f"{ts}" + (" (headline: original unseen rollouts)" if ts == "T0" else ""),
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Training pool (rollouts in the condition's blocks)", color=MUTED, fontsize=9)
    axes[0][0].set_ylabel("M1-per-task, unseen tasks (mean ± std, 3 seeds)", color=MUTED, fontsize=9)
    axes[0][0].text(axes[0][0].get_xlim()[0], 0.503, " chance", color=MUTED, fontsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=len(lab), labelcolor=INK, fontsize=9)
    fig.suptitle("M1-per-task against training-pool size", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_balacc_alpha(inp: Inputs, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in inp.cp if r["test_set"] == "T0" and r.get("variant", "cut") == "cut"]
    conds = [c for c in inp.conds if any(r["cond"] == c for r in rows)]
    if not conds:
        return False
    fig, axes = plt.subplots(1, len(conds), figsize=(4.4 * len(conds), 4.2), facecolor=SURFACE, sharey=True,
                             squeeze=False)
    for ax, c in zip(axes[0], conds):
        _style(ax)
        for d in DETS:
            rs = [r for r in rows if r["cond"] == c and r["detector"] == d]
            if not rs:
                continue
            al = sorted({f(r["alpha"]) for r in rs})
            M = np.array([[f(r["bal_acc"]) for r in rs if f(r["alpha"]) == a] for a in al], dtype=object)
            m = np.array([np.nanmean(v) for v in M])
            s = np.array([np.nanstd(v, ddof=1) if len(v) > 1 else 0 for v in M])
            ax.fill_between(al, m - s, m + s, color=COLOURS[d], alpha=0.12, linewidth=0)
            ax.plot(al, m, color=COLOURS[d], linewidth=2, label=DET_LABEL[d])
            j = int(np.argmin(np.abs(np.array(al) - 0.1)))
            ax.plot([al[j]], [m[j]], marker="o", markersize=7, color=COLOURS[d], markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
        ax.axvline(0.1, color=GRID, linewidth=1.5, zorder=0)
        ax.set_xlim(0, 0.92)
        ax.set_title(f"{c}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("α (marker at the headline α = 0.1)", color=MUTED, fontsize=9)
    axes[0][0].set_ylabel("Balanced accuracy on T0, cut variant (mean ± std)", color=MUTED, fontsize=9)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=len(lab), labelcolor=INK, fontsize=9)
    fig.suptitle("Conformal alarm: balanced accuracy against α, per training pool", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def figures(inp: Inputs, pools: dict, out_dir: Path, tdir: Path) -> list[str]:
    fdir = out_dir / "report_figures"
    fdir.mkdir(parents=True, exist_ok=True)
    out = ["## 8. Figures", ""]
    write_csv(tdir / "pool_sizes.csv", [{"cond": c, "pool": v["pool"], "train_mean": v["train"],
                                          "blocks": " ".join(v["blocks"])} for c, v in pools.items()])
    if inp.final and fig_m1_vs_pool(inp, pools, fdir / "m1_vs_pool.png"):
        sizes = ", ".join(f"{c} {v['pool']} (train ≈ {v['train']:.0f})" for c, v in pools.items())
        out += ["**M1-per-task against training-pool size** (pool = rollouts in the condition's blocks: "
                f"{sizes}; data: `report_tables/ranking.csv`, `report_tables/pool_sizes.csv`).", "",
                "![M1-per-task against training-pool size](report_figures/m1_vs_pool.png)", ""]
    else:
        out += ["*M1 against pool size: not drawn (final_table.csv or pool sizes missing).*", ""]
    if inp.cp and fig_balacc_alpha(inp, fdir / "balacc_vs_alpha.png"):
        out += ["**Balanced accuracy against α**, per condition, T0, cut variant (data: `cp_curves.csv`).", "",
                "![Balanced accuracy against alpha](report_figures/balacc_vs_alpha.png)", ""]
    else:
        out += ["*Balanced accuracy against α: not drawn (cp_curves.csv missing).*", ""]
    sheet = inp.dir / "perturbation_check" / "sheet.png"
    rel = os.path.relpath(sheet, out_dir)
    pert = ""
    if inp.pert:
        by = {}
        for r in inp.pert:
            by.setdefault(r["block"], []).append(r)
        pert = " Automatic criterion passed: " + ", ".join(
            f"{b} {sum(q['pass'] == 'True' for q in rs)}/{len(rs)}" for b, rs in by.items()) + \
            " (how obj / swap were judged: RUNLOG's Step 3.3 entry in section 10)."
        write_csv(tdir / "perturbation_check.csv", inp.pert)
    out += [f"**Perturbation check sheet** (Step 3.3): [{rel}]({rel})" + ("" if sheet.exists() else " (missing)")
            + "." + pert, "", f"![Perturbation check sheet]({rel})", ""]
    return out


# --- 9. selections -----------------------------------------------------------------------------
SEL_ORDER = ("beta_action", "beta_emb_temp", "beta_cross_layer", "emb_layer", "c_C_probes", "c_C_final",
             "c_A", "A_probe")


def selections(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 9. Selections made (on each condition's eval-seen set)", ""]
    if not inp.sel:
        return out + [missing("selections.csv"), ""]
    keys = [k for k in SEL_ORDER if any(r["selection"] == k for r in inp.sel)]
    keys += [k for k in dict.fromkeys(r["selection"] for r in inp.sel) if k not in keys]
    rows, csv_rows = [], []
    for c in CONDS:
        seeds = sorted({r["seed"] for r in inp.sel if r["cond"] == c}, key=int)
        if not seeds:
            continue
        per = {s: {r["selection"]: r["value"] for r in inp.sel if r["cond"] == c and r["seed"] == s}
               for s in seeds}
        csv_rows += [{"cond": c, "seed": s, **{k: per[s].get(k, "") for k in keys}} for s in seeds]
        # one row per condition; a selection that differs between seeds is shown per seed
        line = [c, " ".join(seeds)]
        for k in keys:
            vals = [per[s].get(k, "–") for s in seeds]
            line.append(vals[0] if len(set(vals)) == 1 else " / ".join(vals))
        rows.append(line)
    write_csv(tdir / "selections.csv", csv_rows)
    crit = {r["selection"]: r["criterion"] for r in inp.sel}
    return out + ["β per family, the emb_temp layer and the c values are chosen once per condition over the "
                  "seeds; A's probe per seed (values separated by / are per seed, in seed order). Table: "
                  "`report_tables/selections.csv` (one row per seed).", "",
                  md_table(["Pool", "Seeds"] + keys, rows), "",
                  "Criteria: " + "; ".join(f"`{k}`: {crit[k]}" for k in keys if k in crit) + ".", ""]


# --- 10. deviations -----------------------------------------------------------------------------
def runlog_entries(text: str) -> list[str]:
    """RUNLOG bullets (with continuation lines) that match RUNLOG_PAT, verbatim."""
    bullets, cur = [], None
    for line in text.splitlines():
        if line.startswith("- "):
            if cur:
                bullets.append(cur)
            cur = line
        elif cur is not None and line.strip() and (line.startswith(" ") or line.startswith("\t")):
            cur += "\n" + line
        else:
            if cur:
                bullets.append(cur)
            cur = None
    if cur:
        bullets.append(cur)
    return [b for b in bullets if RUNLOG_PAT.search(b)]


def deviations(inp: Inputs, tdir: Path) -> list[str]:
    out = ["## 10. Deviations and failures", ""]
    g = inp.gate
    if g is None:
        out += ["**Regression gate (Step 8): result missing.**", ""]
    else:
        out += [f"**Regression gate (Step 8): {'PASSED' if g.get('passed') else 'FAILED'}** "
                f"(D0 on T0 against results/abc; run {g.get('utc', '?')}).", "", gate_table(g), ""]
        write_csv(tdir / "regression_gate.csv", [{k: v for k, v in r.items() if not isinstance(v, list)}
                                                  for r in g.get("gate", [])])
    pc = inp.paired_check
    if pc is not None:
        out += [f"**Bootstrap identity check:** {'passed' if pc.get('passed') else 'FAILED'} "
                f"(max |diff| {f(pc.get('max_abs_diff')):.1e}).", ""]
    ent = runlog_entries(inp.runlog)
    out += [f"RUNLOG.md entries mentioning a deviation, decision, judgement, failure, skip or demotion "
            f"({len(ent)}, verbatim):", ""]
    out += ent if ent else ["*none found*" if inp.runlog else missing("RUNLOG.md")]
    return out + [""]


# --- 11. commit message ------------------------------------------------------------------------
def commit_message(inp: Inputs, pools: dict, repo: Path) -> list[str]:
    rows = inp.paired["T0"]
    if rows:
        imp = sum(r["improvement"] == "True" for r in rows)
        wor = sum(r["significant_worsening"] == "True" for r in rows)
        res = (f"no T0 improvement from more data ({len(rows)} paired comparisons, none excludes zero)"
               if imp == 0 and wor == 0 else
               f"T0: {imp} of {len(rows)} paired comparisons improve, {wor} worsen")
    else:
        res = "paired comparison not available"
    gate = "passed" if (inp.gate or {}).get("passed") else "FAILED or missing"
    n_new = next((r["collected"] for r in inp.audit_block if r["block"] == "all_new"), "?")
    pool_txt = ", ".join(f"{c} {v['pool']}" for c, v in pools.items() if v.get("pool")) or "sizes n/a"
    body = [f"More-data run: {n_new} new LIBERO-10 / LIBERO-PRO rollouts, A/C/CP on D0-D2",
            "",
            "Collected new unperturbed and LIBERO-PRO-perturbed rollouts and re-ran the frozen",
            f"A / C / constraint / conformal pipeline on nested training pools ({pool_txt}).",
            f"Regression gate {gate}; result: {res}.",
            "",
            "New:",
            "- collect_pro/: LIBERO-PRO capture code (suite and init state as arguments)",
            "- analysis/abc2/: index, constraints, feature cache, pipeline port, bootstrap, report",
            "- configs/: more_data.yaml (run knobs)",
            "- results/abc2/: RUNLOG, audit, tables, scores, figures, REPORT.md",
            "- corpus_v3_index.json: combined v2 + v3 rollout index"]
    for extra, what in (("run_all.sh", "resumable entrypoint"), ("status.sh", "collection status poller")):
        if (repo / extra).exists():
            body.append(f"- {extra}: {what}")
    body += ["", CO_AUTHOR]
    return ["## 11. Proposed commit message", "", "```", *body, "```", ""]


# --- main ---------------------------------------------------------------------------------------
def build(in_dir: Path, out_dir: Path, repo: Path) -> Path:
    inp = Inputs(in_dir)
    tdir = out_dir / "report_tables"
    tdir.mkdir(parents=True, exist_ok=True)
    pools = pool_sizes(inp)
    have = [n for n in ("final_table.csv", "cp_curves.csv", "stress_summary.csv", "selections.csv",
                        "paired_T0.csv", "audit_block.csv", "demotions.json", "lan_neardup.json",
                        "regression_gate.json", "RUNLOG.md") if (in_dir / n).exists()]
    miss = [n for n in ("final_table.csv", "cp_curves.csv", "stress_summary.csv", "selections.csv",
                        "paired_T0.csv", "audit_block.csv", "demotions.json", "lan_neardup.json",
                        "regression_gate.json", "RUNLOG.md") if n not in have]
    md = ["# More-data run: does more data improve the detectors?", "",
          f"*Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `analysis/abc2/report2.py` "
          f"from `{in_dir}`. Spec: more_data_collection.md (handoff v1), section 5. "
          f"Numbers to 3 d.p.; ± is the std over seeds.*"
          + (f" *Missing inputs: {', '.join(miss)}.*" if miss else ""), ""]
    md += top_warnings(inp)
    md += headline(inp, pools)
    md += corpus(inp, tdir)
    md += ranking(inp, tdir)
    md += alarm(inp, tdir)
    md += paired_section(inp, tdir, inp.paired, "## 5. Paired differences (95% intervals)", "paired",
                         inp.paired_check)
    md += stress(inp, tdir)
    md += ["## 7. MLP ablation and B (H = 10)", ""]
    md += optional_section(inp, tdir, "mlp", "### MLP ablation (Step 10)")
    md += optional_section(inp, tdir, "b", "### B at H = 10 (Step 11)")
    md += figures(inp, pools, out_dir, tdir)
    md += selections(inp, tdir)
    md += deviations(inp, tdir)
    md += commit_message(inp, pools, repo)
    path = out_dir / "REPORT.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(md))
    shutil.move(tmp, path)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", type=Path, default=data2.RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --in-dir")
    args = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    out = args.out_dir or args.in_dir
    out.mkdir(parents=True, exist_ok=True)
    path = build(args.in_dir, out, repo)
    print(f"[report] {path} ({path.stat().st_size} bytes); tables in {out / 'report_tables'}, "
          f"figures in {out / 'report_figures'}")


if __name__ == "__main__":
    main()
