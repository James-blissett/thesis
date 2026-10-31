"""
report.py

results/thesis_results_final/REPORT.md (results_followup_handoff.md §7). The headline,
the decisions and the proposed commit message are written by hand in
results/thesis_results_final/report_notes.md (sections "## Headline", "## Decisions and
deviations", "## Proposed commit message"); every table here is rendered from its CSV.

Usage: source env.sh; python analysis/thesis_results/final/report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc  # noqa: E402
from fc import HEAD_NAME, REF_NAME, read_csv  # noqa: E402

T = fc.OUT / "tables"
PT = "per-task-and-perturbation AUROC"


def md(rows, cols, heads=None) -> str:
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in rows:
        out.append("| " + " | ".join(str(c(r) if callable(c) else r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def pm(r, name, nd=3):
    return fc.pm(r.get(f"{name} mean", ""), r.get(f"{name} std", ""), nd)


def ci(r, lo="95% interval low", hi="95% interval high", d="difference"):
    if r.get(d, "") == "":
        return ""
    return f"{float(r[d]):+.3f} [{float(r[lo]):+.3f}, {float(r[hi]):+.3f}]"


def short_verdict(v: str) -> str:
    return {"interval includes zero": "includes 0"}.get(v, v.replace(" (interval above zero)", "").replace(" (interval below zero)", ""))


def notes() -> dict:
    p = fc.OUT / "report_notes.md"
    txt = p.read_text() if p.exists() else ""
    parts = re.split(r"^## ", txt, flags=re.M)
    return {s.split("\n", 1)[0].strip(): s.split("\n", 1)[1].strip() for s in parts[1:]}


def section_t1() -> str:
    rows = read_csv(T / "table1_core_detectors.csv")
    col = "per-task AUROC (per task-and-perturbation cell on the headline test set)"
    head = [r for r in rows if r["block"].startswith("headline")]
    ref = [r for r in rows if r["block"].startswith("reference")]
    var = lambda r: r["variant"] + (f", H = {r['look-ahead H']}" if r["look-ahead H"] else "")  # noqa: E731
    a = md(head, ["detector", var, lambda r: pm(r, col), lambda r: pm(r, "pooled AUROC"),
                  "timestep-only floor (pooled AUROC)", lambda r: pm(r, "balanced accuracy at alpha 0.1"),
                  lambda r: pm(r, "share of failures caught at alpha 0.1"), lambda r: pm(r, "false-alarm rate at alpha 0.1"),
                  lambda r: pm(r, "mean normalised detection time at alpha 0.1"),
                  lambda r: pm(r, "stress set: share alarmed before the cut at alpha 0.1")],
           ["detector", "variant", PT, "pooled AUROC", "floor", "bal. acc. α 0.1", "failures caught", "false alarms",
            "detection time", "stress set alarmed"])
    b = md(ref, ["detector", var, lambda r: pm(r, col), lambda r: pm(r, "balanced accuracy at alpha 0.1")],
           ["detector", "variant", "per-task AUROC", "bal. acc. α 0.1"])
    cells = head[0][f"{col}: cells averaged per seed"]
    return (f"## T1. Core table\n\nFile: `tables/table1_core_detectors.csv` (headline block, then the reference block); "
            f"all test sets incl. eval-seen: `table1_core_detectors_all_test_sets.csv`; stress subsets: `table1_stress_set.csv`. "
            f"Source: the full training set's run (`results/thesis_results_full/tables/table1_core_detectors_all_test_sets.csv`), "
            f"re-cut, nothing recomputed. {PT} averages {cells} cells per seed (cells need at least 2 rollouts of each class). "
            f"Floor = pooled AUROC of the timestep-only score. Alarm columns: cut-variant conformal alarm at α = 0.1.\n\n"
            f"### Headline: full training set, {HEAD_NAME}\n\n{a}\n\n### Reference: full training set, {REF_NAME}\n\n{b}\n")


def section_paired() -> str:
    rows = read_csv(T / "paired_comparisons.csv")
    rows = [r for r in rows if not r["metric"].startswith("share")]
    return ("## Paired comparisons (headline setup)\n\nFile: `tables/paired_comparisons.csv` (also share of failures caught; "
            "per seed in `tables/data/`). Bootstrap identity check: `logs/paired_identity_check.json`. B at H = 10. "
            "Difference x − y with 95% interval; \"x higher\" / \"x lower\" only if the interval excludes zero.\n\n"
            + md(rows, ["comparison (x - y)", "metric", "x value", "y value", ci, lambda r: short_verdict(r["verdict"])],
                 ["comparison (x − y)", "metric", "x", "y", "difference [95% interval]", "verdict"]) + "\n")


def section_t3() -> str:
    rows = read_csv(T / "table3_constraints.csv")
    return ("## T3. Constraints and baselines (full-set selections)\n\nFile: `tables/table3_constraints.csv`.\n\n"
            + md(rows, ["kind", "signal", "smoothing beta", "sign per seed (+: higher means failure)",
                        lambda r: pm(r, f"{PT}, {HEAD_NAME},"), lambda r: pm(r, "per-task AUROC, eval-seen,"),
                        f"pooled AUROC, {HEAD_NAME}, mean"],
                 ["kind", "signal", "β", "sign per seed", f"{PT} (headline)", "per-task AUROC (eval-seen)", "pooled AUROC (headline)"])
            + "\n")


def section_t4() -> str:
    rows = read_csv(T / "table4_b_by_lookahead.csv")
    return ("## T4. B per look-ahead H (headline setup)\n\nFile: `tables/table4_b_by_lookahead.csv`. The persistence and "
            "own-target rows are in `appendix/` only.\n\n"
            + md(rows, ["look-ahead H", "row", lambda r: pm(r, PT), lambda r: pm(r, "balanced accuracy at alpha 0.1"),
                        "choice per seed"],
                 ["H", "row", PT, "bal. acc. α 0.1", "choice per seed"]) + "\n")


def section_t5() -> str:
    rows = read_csv(T / "table5_ablations_families_and_lookahead.csv")
    return ("## T5. Families and look-ahead (headline setup)\n\nFile: `tables/table5_ablations_families_and_lookahead.csv`.\n\n"
            + md(rows, ["ablation", "detector", "variant", "look-ahead H", lambda r: pm(r, PT),
                        lambda r: pm(r, "balanced accuracy at alpha 0.1")],
                 ["ablation", "detector", "variant", "H", PT, "bal. acc. α 0.1"]) + "\n")


def section_check1() -> str:
    rows = read_csv(T / "b_secondary_rows.csv")
    pr = read_csv(T / "b_secondary_paired.csv")
    lab = lambda s: s.replace("B: unweighted mean of the 29 embedding-temporal probe outputs", "B embedding-temporal mean") \
        .replace("B: unweighted mean of every grid probe except the xl_final probes, exploratory (excludes the cross-layer target)",
                 "B mean without xl_final (exploratory)").replace("B: unweighted mean of 116 probe outputs (primary)", "B mean of all grid probes") \
        .replace("A: unweighted mean of 34 probe outputs (primary)", "A mean").replace("C: unweighted mean of 33 probe outputs (primary)", "C mean") \
        .replace("Baseline: best single untrained constraint (layer chosen on eval-seen per-task AUROC), running maximum", "best untrained constraint")  # noqa: E731
    a = md(rows, [lambda r: lab(r["row"]), "status", lambda r: pm(r, PT), lambda r: pm(r, "balanced accuracy at alpha 0.1")],
           ["row", "status", PT, "bal. acc. α 0.1"])
    b = md(pr, [lambda r: lab(r["comparison (x - y)"]), "metric", ci, lambda r: short_verdict(r["verdict"])],
           ["comparison (x − y)", "metric", "difference [95% interval]", "verdict"])
    return ("## Check 1. B's embedding-temporal mean against A and C\n\nFiles: `tables/b_secondary_rows.csv`, "
            "`tables/b_secondary_paired.csv`. The exploratory row (every grid probe except the xl_final probes, 93 per H) "
            "is new: an unweighted mean of the full run's saved grid outputs.\n\n" + a + "\n\n" + b + "\n")


def section_check2() -> str:
    rows = read_csv(T / "table6_training_data_headline.csv")
    p1 = [r for r in rows if r["part"].startswith("1") and r["test set"] == HEAD_NAME]
    p2 = [r for r in rows if r["part"].startswith("2")]
    st = [r for r in rows if r["part"].startswith("1") and r["test set"] == "stress set"]
    cols = ["detector", "metric", "original training set", "full training set",
            lambda r: ci(r, d="difference (full minus original)"), lambda r: short_verdict(r["verdict"])]
    heads = ["row", "metric", "original", "full", "full − original [95% interval]", "verdict"]
    x = p2[0]
    return ("## Check 2. Training-data ablation on the headline test set\n\nFile: `tables/table6_training_data_headline.csv`. "
            "Part 1 is read from `results/thesis_results_full/tables/training_set_comparison.csv` (same bootstrap). "
            f"Part 2 is new. Common probe set: {x['common probes per H']} (layer, target) pairs per H present in both grids "
            f"(original grid {x['probes per H, original grid / full grid'].split(' / ')[0]}, full grid "
            f"{x['probes per H, original grid / full grid'].split(' / ')[1]}). Excluded from the original grid: "
            f"{x['excluded from the original grid (not in the full grid)']}. Excluded from the full grid: "
            f"{x['excluded from the full grid (not in the original grid)']}.\n\n"
            "### Part 1: existing comparison\n\n" + md(p1, cols, heads)
            + "\n\nStress set (share alarmed before the cut at α = 0.1; no bootstrap):\n\n"
            + md(st, ["detector", "original training set", "full training set", "difference (full minus original)"],
                 ["row", "original", "full", "full − original"])
            + "\n\n### Part 2: B on the common probe set\n\n" + md(p2, cols + ["common probes per H"], heads + ["probes"]) + "\n")


def section_check3() -> str:
    rows = read_csv(T / "test_set_breakdown.csv")
    pr = read_csv(T / "test_set_breakdown_paired.csv")
    name = "per-task AUROC (per task-and-perturbation cell)"
    keys = list(dict.fromkeys(r["row"] for r in rows))
    subsets = list(dict.fromkeys(r["rollouts"] for r in rows))
    by = {(r["row"], r["rollouts"]): r for r in rows}
    short = {s: s.replace("LIBERO-PRO test set, ", "").replace("unseen tasks, ", "") for s in subsets}
    tab = ["| row | " + " | ".join(short[s] for s in subsets) + " |", "|" + "---|" * (len(subsets) + 1)]
    for k in keys:
        lab = k.replace(" probe outputs", "").replace("unweighted mean of", "mean of")
        tab.append(f"| {lab} | " + " | ".join(pm(by[(k, s)], name) for s in subsets) + " |")
    cnt = ["| rollouts | cells per seed | failures per seed | successes per seed |", "|---|---|---|---|"]
    for s in subsets:
        r = by[(keys[0], s)]
        cnt.append(f"| {s} | {r['cells averaged per seed']} | {r['failures per seed']} | {r['successes per seed']} |")
    b = md(pr, ["comparison (x - y)", "test set", "metric", ci, lambda r: short_verdict(r["verdict"])],
           ["comparison (x − y)", "rollouts", "metric", "difference [95% interval]", "verdict"])
    return ("## Check 3. Where the detection on LIBERO-PRO comes from\n\nFiles: `tables/test_set_breakdown.csv`, "
            "`tables/test_set_breakdown_paired.csv`. Full training set. Per-task AUROC is computed per task-and-perturbation "
            "cell; on the original rollouts and on original + new unperturbed rollouts, cells are tasks. The unperturbed "
            "cells of the LIBERO-PRO test set are exactly the original + new unperturbed rollouts (same rollouts, same "
            "cells), so those two columns are identical by construction.\n\n" + "\n".join(tab) + "\n\n" + "\n".join(cnt)
            + "\n\n" + b + "\n")


def section_check4() -> str:
    b = read_csv(T / "smoothing_sensitivity_b.csv")
    c = read_csv(T / "smoothing_sensitivity_c.csv")
    sg = read_csv(T / "smoothing_sensitivity_signs.csv")
    import json
    san = json.loads((fc.OUT / "logs" / "smoothing_sanity_check.json").read_text())
    setting = lambda r: {"current (selected): emb_temp none, action family 0.83, cross-layer family 0.96": "current"}.get(r["setting"], r["setting"])  # noqa: E731
    d = lambda nm: (lambda r: (f"{float(r[f'difference from current, {nm}']):+.3f} "  # noqa: E731
                               f"[{float(r[f'difference from current, {nm}, 95% interval low']):+.3f}, "
                               f"{float(r[f'difference from current, {nm}, 95% interval high']):+.3f}]")
                    if r[f"difference from current, {nm}, 95% interval low"] != "" else "")
    cols = [setting, lambda r: r["variant"].split(",")[0].replace(" (primary)", " (primary)") + (f", H = {r['look-ahead H']}" if r["look-ahead H"] else ""),
            lambda r: pm(r, PT), d("AUROC"), lambda r: pm(r, "balanced accuracy at alpha 0.1"), d("balanced accuracy")]
    heads = ["setting", "row", PT, "AUROC − current [95%]", "bal. acc. α 0.1", "bal. acc. − current [95%]"]
    bb = [r for r in b if r["detector"] == "B"]
    probes = {}
    for r in bb:
        probes[setting(r)] = (r["B probes per look-ahead H"], r["B targets skipped (sign-unstable)"])
    ptab = md([{"s": k, "n": v[0], "u": v[1]} for k, v in probes.items()], ["s", "n", "u"],
              ["setting", "B probes per H", "B targets skipped (sign-unstable)"])
    nun = {}
    for r in sg:
        nun.setdefault(r["setting"], [0, 0, True])
        nun[r["setting"]][0] += r["sign-unstable"] == "True"
        nun[r["setting"]][2] &= r["agrees with the Stage 2 table"] == "True"
    utab = md([{"s": setting({"setting": k}), "n": v[0], "a": "yes" if v[2] else "NO"} for k, v in nun.items()],
              ["s", "n", "a"], ["setting", "sign-unstable constraints (of 99)", "agrees with the Stage 2 table"])
    sanity = md(san["rows"], ["row", lambda r: f"{r['current setting (this check), mean']:.4f}",
                              lambda r: f"{r['headline row, mean']:.4f}", lambda r: f"{r['difference of means']:+.4f}",
                              lambda r: "yes" if r["passed (|difference of means| <= 0.005)"] else "NO"],
                ["row", "current setting (refit here)", "headline row", "difference", "within ± 0.005"])
    return ("## Check 4. Smoothing sensitivity for B and C\n\nFiles: `tables/smoothing_sensitivity_b.csv`, "
            "`tables/smoothing_sensitivity_c.csv`, `tables/smoothing_sensitivity_signs.csv`; per seed and the bootstrap rows "
            "in `tables/data/`. Headline setup only. Constraint preparation redone from the raw series per setting; "
            "C's 33 probes and B's grid (H = 0, 10, 25) refitted on all train rows at the headline c values (C 1e-3, "
            "B 1e-4); no trained combiner. The best untrained constraint re-chooses its emb_temp layer per setting. "
            "Paired bootstrap against \"current\" (same procedure as above).\n\n"
            f"**Sanity check ({'passed' if san['passed'] else 'FAILED'}):** \"current\" against the headline rows.\n\n" + sanity
            + "\n\n**Signs per setting** (recomputed per seed; cross-checked against the Stage 2 table, which holds the same "
            "quantities for every β):\n\n" + utab + "\n\n" + ptab
            + "\n\n### C\n\n" + md(c, cols, heads) + "\n\n### B (and the reference row)\n\n" + md(b, cols, heads) + "\n")


def section_methods() -> str:
    rows = read_csv(T / "methods_values.csv")
    return ("## Methods values\n\nFile: `tables/methods_values.csv` (sources per row).\n\n"
            + md(rows, ["value", "full training set (headline)", "original training set"],
                 ["value", "full training set (headline)", "original training set"]) + "\n")


FIGS = [("F1", "F1_example_failure_and_success", "example failure and success (full training set, headline test set, seed 0, task 9 unperturbed cell); data: figures/data/example_rollouts.{csv,json}"),
        ("F2", "F2_alarm_tradeoff", "balanced accuracy against normalised detection time over 15 α, single panel; data: figures/data/alarm_tradeoff.csv"),
        ("F3", "F3_corpus_kept_rollouts_and_action_repeat", "copied unchanged from results/thesis_results/"),
        ("F4", "F4_length_check", "length check on the headline test set; data: figures/data/length_check.csv"),
        ("F5", "F5_untrained_constraint_heatmap", "untrained per-task-and-perturbation AUROC, full-set signs and unstable flags; data: figures/data/untrained_constraint_heatmap.csv"),
        ("F6", "F6_per_probe_detector_a", "A's 34 probes; data: figures/data/per_probe_outcome_auroc.csv"),
        ("F7", "F7_per_probe_detector_c", "C's 33 probes; same CSV"),
        ("F8", "F8_per_probe_detector_b", "B's 116 probes at H = 10; same CSV"),
        ("F10", "F10_per_probe_overlay", "overlay of F6-F8; same CSV"),
        ("F11", "F11_capacity", "copied unchanged from results/thesis_results/"),
        ("F9 (appendix)", "../appendix/F9_b_target_auroc_heatmap", "B's own-target row-level AUROC, headline test set; data: appendix/data/b_target_auroc_heatmap.csv")]


def main() -> None:
    n = notes()
    out = ["# Thesis results: headline setup (full training set, unseen tasks incl. LIBERO-PRO)", "",
           "*results_followup_handoff.md (version 4), §7. Generated by `analysis/thesis_results/final/report.py` from "
           "`results/thesis_results_final/`; the headline, decisions and commit message are in `report_notes.md`. Numbers "
           "to 3 d.p.; ± is the std over the 3 seeds (ddof 1); per-seed values are in the CSVs. Every interval is the "
           "paired bootstrap of the main run (1,000 resamples within task-and-perturbation cell within seed, 95%, each "
           "seed's conformal band held fixed); \"higher\" / \"lower\" only if the interval excludes zero.*", "",
           "## Headline", "", n.get("Headline", "(missing)"), ""]
    for f in (section_t1, section_paired, section_t3, section_t4, section_t5, section_check1, section_check2,
              section_check3, section_check4, section_methods):
        try:
            out.append(f())
        except FileNotFoundError as e:
            out.append(f"## {f.__name__}\n\n(missing: {e})\n")
    out.append("## Figures\n\nEach figure has a PDF and a PNG in `figures/` and is drawn only from its CSV in `figures/data/`. "
               "T2 (corpus composition) is `tables/table2_corpus_composition.csv`, copied unchanged.\n\n"
               + "\n".join(f"- **{a}** `figures/{b}.pdf`: {c}" for a, b, c in FIGS) + "\n")
    out.append("## Decisions and deviations\n\n" + n.get("Decisions and deviations", "(missing)") + "\n")
    out.append("## Proposed commit message\n\n" + n.get("Proposed commit message", "(missing)") + "\n")
    (fc.OUT / "REPORT.md").write_text("\n".join(out))
    fc.log(f"[report] REPORT.md written ({sum(len(x) for x in out)} characters)")


if __name__ == "__main__":
    main()
