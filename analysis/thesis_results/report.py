"""
report.py

Assembles results/thesis_results/REPORT.md (results_collection_handoff.md §5): the
headline and reading notes (hand-written in report_notes.md, checked against the
tables), then every table (rendered from tables/*.csv) and figure (from figures/), the
reproduction checks, the deviations (RUNLOG.md entries that record a decision, deviation,
finding or failure) and one proposed commit message (commit_message.txt).

Usage: source env.sh; python analysis/thesis_results/report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import read_csv  # noqa: E402

OUT = common.OUT_DIR
T = OUT / "tables"


def pm(row, name, d=3):
    m, s = row.get(f"{name} mean", ""), row.get(f"{name} std", "")
    if m in ("", None):
        return ""
    return f"{float(m):.{d}f} ± {float(s):.{d}f}" if s not in ("", None) else f"{float(m):.{d}f}"


def md(rows, cols) -> str:
    """cols: list of (header, fn(row) -> str)."""
    out = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(fn(r)) for _, fn in cols) + " |")
    return "\n".join(out)


def t1_md(path: Path, per_task="per-task AUROC") -> str:
    rows = read_csv(path)
    return md(rows, [("Detector", lambda r: r["detector"]), ("Variant", lambda r: r["variant"]),
                     ("H", lambda r: r["look-ahead H"]),
                     (per_task, lambda r: pm(r, per_task)),
                     ("pooled AUROC", lambda r: pm(r, "pooled AUROC")),
                     ("floor", lambda r: r["timestep-only floor (pooled AUROC)"]),
                     ("bal. acc. α 0.1", lambda r: pm(r, "balanced accuracy at alpha 0.1")),
                     ("failures caught", lambda r: pm(r, "share of failures caught at alpha 0.1")),
                     ("false alarms", lambda r: pm(r, "false-alarm rate at alpha 0.1")),
                     ("detection time", lambda r: pm(r, "mean normalised detection time at alpha 0.1")),
                     ("stress set alarmed", lambda r: pm(r, "stress set: share alarmed before the cut at alpha 0.1"))])


def generic_md(path: Path, keep=None, drop_seed=True) -> str:
    rows = read_csv(path)
    if not rows:
        return "*(empty)*"
    cols = [c for c in rows[0] if not (drop_seed and re.search(r"seed \d$", c)) and (keep is None or c in keep)]
    merged, used = [], set()
    for c in cols:
        if c in used:
            continue
        if c.endswith(" mean") and c[:-5] + " std" in cols:
            base = c[:-5]
            merged.append((base, (lambda b: lambda r: pm(r, b))(base)))
            used |= {c, base + " std"}
        elif not c.endswith(" std") or c[:-4] + " mean" not in cols:
            merged.append((c, (lambda k: lambda r: r[k])(c)))
            used.add(c)
    return md(rows, merged)


def deviations() -> str:
    txt = (OUT / "RUNLOG.md").read_text().splitlines()
    keep = [ln for ln in txt if ln.startswith("- **") and re.search(
        r"(?i)deviation|decision|finding|fail|skip|question|mismatch|differs|not install", ln)]
    return "\n".join(keep) if keep else "*(none)*"


FIGS = [("F1", "F1_example_failure_and_success", "Example failure and success (seed 1, one unseen task; seed 0 has no unseen failure that B alarms on at α = 0.1): B's unweighted-mean score (H = 10) with its α = 0.1 conformal band, the cut and B's alarm step; A's and C's mean scores for comparison; frames at start, middle, alarm (or three-quarters of the cut for the success) and cut."),
        ("F2", "F2_alarm_tradeoff", "Balanced accuracy against mean normalised detection time over the 15 α levels, mean of 3 seeds: (a) unseen tasks, original rollouts; (b) unseen tasks incl. LIBERO-PRO."),
        ("F3", "F3_corpus_kept_rollouts_and_action_repeat", "(a) Kept rollouts per class against t (original rollouts; training split mean of 3 seeds); (b) action-repeat rate before vs from the first success, successful rollouts, per source."),
        ("F4", "F4_length_check", "Length check: mean score against t on unseen failures (solid) and unseen successes (dashed), unseen tasks, original rollouts; ± 1 std over seeds; drawn while ≥ 5 rollouts of the class are kept."),
        ("F5", "F5_untrained_constraint_heatmap", "Untrained per-task AUROC (running maximum; unseen tasks, original rollouts; mean of 3 seeds) per layer for emb_temp, xl_adj, xl_final, and the action constraints; hatched = sign-unstable."),
        ("F6", "F6_per_probe_detector_a", "Detector A: per-task AUROC against outcome of each of the 34 probes (running sum), unseen (solid, ± 1 std) and eval-seen (faded); the action-token probe to the right; shuffled-label control; A's unweighted mean (dash-dot)."),
        ("F7", "F7_per_probe_detector_c", "Detector C: the same for its 33 probes; dotted: the layer's untrained constraints scored by their running sum (best of three, chosen on eval-seen)."),
        ("F8", "F8_per_probe_detector_b", "Detector B: the same for its grid probes at H = 10 (124 on the original training set, 116 on the full one), one line per target; gaps are sign-unstable targets (no probe)."),
        ("F9", "F9_b_target_auroc_heatmap", "B's probes: row-level AUROC against each probe's own constraint target, H = 10, unseen tasks (original rollouts), mean of 3 seeds."),
        ("F10", "F10_per_probe_overlay", "Overlay: A per layer, C per layer, B per-layer mean of its probes (H = 10), unseen tasks, original rollouts."),
        ("F11", "F11_capacity", "Capacity: per-task AUROC against depth (linear, 2, 6, 12 hidden layers, residual MLP), A and C; train, eval-seen (optimistic) and unseen tasks; original (solid) and full (dashed) training set.")]


def main() -> None:
    notes = (OUT / "report_notes.md").read_text() if (OUT / "report_notes.md").exists() else "*(notes pending)*"
    msg = (OUT / "commit_message.txt").read_text() if (OUT / "commit_message.txt").exists() else "*(pending)*"
    chk = read_csv(T / "data" / "reproduction_checks.csv")
    chk_p = read_csv(T / "data" / "reproduction_checks_probes.csv")
    chk_b = read_csv(T / "data" / "reproduction_checks_b_grid.csv")
    chk_c = read_csv(T / "data" / "reproduction_checks_corpus.csv")
    nfail = sum(r["passed"] != "True" for r in chk + chk_p + chk_b + chk_c)
    maxd = max([float(r["abs_difference"]) for r in chk + chk_p + chk_b] or [0])
    parts = [f"# Thesis results: {common.POOL_NAME}", "",
             "*results_collection_handoff.md (version 4), §5. Generated by `analysis/thesis_results/report.py` from "
             "`results/thesis_results/`. Numbers to 3 d.p.; ± is the std over the 3 seeds (ddof 1); per-seed values are in "
             "the CSVs. Plain names throughout (handoff §0).*", "", notes, "",
             "## Reproduction checks", "",
             f"{len(chk) + len(chk_p) + len(chk_b) + len(chk_c)} checks against the earlier runs' files "
             f"({len(chk)} detector rows, {len(chk_p)} per-probe values, {len(chk_b)} B grid values, {len(chk_c)} corpus "
             f"counts); **{nfail} failed**; largest difference {maxd:.4f} (tolerance 0.005 AUROC, 0.01 balanced accuracy). "
             "Files: `tables/data/reproduction_checks*.csv`.", "",
             "## T1. Core table (unseen tasks, original rollouts)", "",
             "File: `tables/table1_core_detectors.csv` (all three test sets: `table1_core_detectors_all_test_sets.csv`; "
             "stress subsets: `table1_stress_set.csv`). Floor = pooled AUROC of the timestep-only score; per-task AUROC of the "
             "timestep score is 0.5 by construction. Alarm columns: cut-variant conformal alarm at α = 0.1.", "",
             t1_md(T / "table1_core_detectors.csv"), ""]
    allr = read_csv(T / "table1_core_detectors_all_test_sets.csv")
    pro = [r for r in allr if r["test set"] == common.TEST_SETS["T2"]]
    if pro:
        import csv as _csv
        tmp = T / "data" / "_t1_pro.csv"
        with tmp.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(pro[0]))
            w.writeheader()
            w.writerows(pro)
        parts += ["### T1, unseen tasks incl. LIBERO-PRO (per-task-and-perturbation AUROC)", "",
                  t1_md(tmp, "per-task-and-perturbation AUROC"), ""]
        tmp.unlink()
    for title, f in (("T2. Corpus composition", "table2_corpus_composition.csv"),
                     ("T3. Constraints: best per family and baselines", "table3_constraints.csv"),
                     ("T4. B per look-ahead H", "table4_b_by_lookahead.csv"),
                     ("T4 (continued). B's 2-layer MLP vs its linear probe vs persistence", "table4_b_mlp_vs_linear_vs_persistence.csv"),
                     ("T4 (continued). B's linear probes vs persistence, every target and H", "table4_b_linear_vs_persistence.csv"),
                     ("T5. Ablations: constraint families and look-ahead", "table5_ablations_families_and_lookahead.csv"),
                     ("T6. Training data", "table6_training_data.csv"),
                     ("T6 (continued). Original vs full training set, paired bootstrap (full minus original)", "table6_training_data_paired.csv"),
                     ("Paired comparisons (unseen tasks, original rollouts)", "paired_comparisons.csv")):
        p = T / f
        if p.exists():
            keep = None
            if f == "table6_training_data_paired.csv":
                keep = ["detector", "test set", "metric", "original training set", "full training set",
                        "difference (full minus original)", "95% interval low", "95% interval high", "verdict"]
            if f == "paired_comparisons.csv":
                keep = ["comparison (x - y)", "metric", "x value", "y value", "difference", "95% interval low",
                        "95% interval high", "verdict"]
            parts += [f"## {title}", "", f"File: `tables/{f}`.", "", generic_md(p, keep), ""]
    parts += ["## Figures", "", "Each figure: `figures/<name>.pdf` (vector) and `.png`; data in `figures/data/` "
              "(`per_probe_outcome_auroc.csv` behind F6–F8 and F10).", ""]
    for num, name, cap in FIGS:
        if (OUT / "figures" / f"{name}.png").exists():
            parts += [f"**{num}.** {cap}", "", f"![{num}](figures/{name}.png)", ""]
        else:
            parts += [f"**{num}.** *not produced* — see deviations.", ""]
    parts += ["## Deviations, decisions and findings (from RUNLOG.md)", "", deviations(), "",
              "## Proposed commit message", "", "```", msg.strip(), "```", ""]
    (OUT / "REPORT.md").write_text("\n".join(parts))
    print(f"REPORT.md written ({len(parts)} blocks)")


if __name__ == "__main__":
    main()
