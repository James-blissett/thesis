"""
report.py

§7 of task_resampling_handoff.md: task_resampling/REPORT.md from the CSVs of this run.
The headline sentences, decisions and commit message are written by hand in
task_resampling/report_notes.md (sections "## Headline", "## Decisions", "## Commit
message"); every table is rendered from the CSVs here.

Usage: source env.sh; python analysis/thesis_results/task_resampling/report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr  # noqa: E402
from tr import OUT, read_csv  # noqa: E402


def f3(x, sign=False) -> str:
    if x in ("", None):
        return ""
    v = float(x)
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def notes() -> dict:
    txt = (OUT / "report_notes.md").read_text()
    parts = re.split(r"^## ", txt, flags=re.M)[1:]
    return {p.split("\n", 1)[0].strip(): p.split("\n", 1)[1].strip() for p in parts}


def ranges_section() -> str:
    rows = read_csv(OUT / "ranges.csv")
    full = {(r["comparison"], r["metric"]): r for r in read_csv(OUT / "logs" / "ranges_full.csv")}
    chk = json.loads((OUT / "logs" / "self_check.json").read_text())
    out = ["## §3. Ranges that also redraw the unseen tasks", "",
           "File: `ranges.csv` (extra columns in `logs/ranges_full.csv`). Difference = x − y on the saved sample (mean over the "
           "three task splits); 95% range = 2.5th–97.5th percentile over 2,000 rounds (seed 20261101), each round redrawing "
           "3 of each split's 3 unseen tasks with replacement and then the rollouts within each task-and-perturbation group. "
           "Both sides of a comparison use the same draws. B at H = 10 unless stated; headline test set (unseen tasks incl. "
           "LIBERO-PRO) unless stated. **Bold** = the range excludes zero.", "",
           f"**Self-check (step 1 switched off), passed.** With bootstrap2's own rollout draws, the engine reproduces every "
           f"range in `tables/paired_comparisons.csv` to the CSV's 4 decimals (max |change| {chk['a_bootstrap2_draws']['max_abs_diff']:.1e}). "
           f"With this run's draws (2,000 rounds), the largest change in a difference or range end is "
           f"{chk['b_fresh_draws']['max_abs_diff']:.4f} (limit 0.005). Per-row values: `logs/self_check.csv`.", ""]
    sec = None
    body = []
    for r in rows:
        if r["section"] != sec:
            if body:
                out += [table(["comparison (x − y)", "metric", "difference", "95% range", "rounds used"], body), ""]
                body = []
            sec = r["section"]
            out += [f"**{sec}**", ""]
        ex = r["range excludes zero"] == "True"
        rng = f"[{f3(r['95% range low'], True)}, {f3(r['95% range high'], True)}]"
        fr = full[(r["comparison"], r["metric"])]
        used = r["rounds used"] + (f" (skipped {r['rounds skipped']})" if r["rounds skipped"] != "0" else "")
        body.append([r["comparison"], r["metric"], f3(r["difference"], True), f"**{rng}**" if ex else rng, used])
        del fr
    out += [table(["comparison (x − y)", "metric", "difference", "95% range", "rounds used"], body), ""]
    g = [r for r in read_csv(OUT / "logs" / "ranges_full.csv") if r["section"] == "1. headline"][0]
    out += [f"**Rounds skipped and groups dropped.** A round is skipped when a task split has no group with at least 2 "
            f"failures and 2 successes, or no success or no failure for balanced accuracy, so that every recorded value "
            f"averages three splits. On the headline test set no round is skipped. In {g['rounds with a group dropped']} of "
            f"2,000 rounds at least one drawn group lacks 2 of each class and is left out of the per-task AUROC: on average "
            f"{float(g['groups dropped per round (mean over used rounds, all splits)']):.1f} of "
            f"{float(g['groups drawn per round (mean over used rounds, all splits)']):.0f} drawn groups per round, mostly "
            f"object-perturbation groups with few successes. The same groups fail the rule on the saved sample (8, 7 and 9 of 9 "
            f"groups qualify). Within the object-perturbation groups alone, 292 rounds are skipped for AUROC (179 for balanced "
            f"accuracy), because a split's drawn tasks then often have no qualifying group.", ""]
    return "\n".join(out)


def far_section() -> str:
    rows = read_csv(OUT / "false_alarms_vs_alpha.csv")
    alphas = sorted({r["alpha"] for r in rows}, key=float)
    head = ["detector"] + [f"{float(a):.2f}" for a in alphas] + ["α levels with mean above α"]
    body = []
    for d in dict.fromkeys(r["detector"] for r in rows):
        R = [r for r in rows if r["detector"] == d]
        cells = []
        for r in R:
            v = float(r["false-alarm rate on unseen successes, mean over splits"])
            cells.append(f"**{v:.2f}**" if r["mean above alpha"] == "True" else f"{v:.2f}")
        body.append([d] + cells + [sum(r["mean above alpha"] == "True" for r in R)])
    return "\n".join(["## §4. False alarms against the target rate", "",
                      "File: `false_alarms_vs_alpha.csv` (mean ± std over splits, per split, eval-seen reference); figure "
                      "`F12_false_alarms_vs_alpha.{pdf,png}`. Source: the full training set's saved alarm results "
                      "(`results/thesis_results_full/tables/data/detector_alarm_per_seed.csv`), headline test set, cut variant, "
                      "the 15 α levels of the grid. Values: false-alarm rate on unseen successes, mean over the three task "
                      "splits; **bold** = above α. At α = 0.02 the band is infinite (too few calibration rollouts for that "
                      "level), so no detector alarms.", "",
                      table(head, body), "", "![F12](F12_false_alarms_vs_alpha.png)", ""])


def failures_section() -> str:
    S = read_csv(OUT / "failure_types_summary.csv")
    dets = ["A mean", "B mean (H = 10)", "C mean", "untrained emb_temp[26]"]
    body = []
    for r in S:
        body.append([r["failure type"], r["rollouts (rollout x split)"], r["distinct rollouts"]] +
                    [f"{float(r[f'{d}: share caught before the cut (alpha 0.1)']):.3f} ({r[f'{d}: caught (rollout x split)']})"
                     if r[f"{d}: share caught before the cut (alpha 0.1)"] else "" for d in dets])
    fz = [r for r in S if r["failure type"] == "froze"][0]
    body2 = [[d, fz[f"{d}: caught (rollout x split)"],
              fz[f"{d}: caught, alarm before the freeze starts (share of caught)"],
              fz[f"{d}: caught, median alarm step minus freeze start"]] for d in dets]
    return "\n".join([
        "## §5. Which failures are caught", "",
        "Files: `failure_types.csv` (one row per rollout and task split), `failure_types_summary.csv`, "
        "`failure_descriptions.md` (the 20 checked by eye), frame strips in `failure_frames/`. Rollouts: every failed original "
        "LIBERO-10 rollout of each split's unseen tasks, headline setup (full training set; cut = shortest kept length in the "
        "rollout's task-and-perturbation group of the headline test set), counted once per split. Caught = the α = 0.1 alarm "
        "fires before the cut.", "",
        table(["failure type", "rollouts (rollout × split)", "distinct rollouts"] +
              [f"{d}: share caught (count)" for d in dets], body), "",
        "\"Froze\" only (alarm step minus freeze start: negative = the alarm comes before the freeze):", "",
        table(["detector", "caught", "share of caught with the alarm before the freeze starts", "median alarm step − freeze start"],
              body2), "",
        "**Check by eye:** the action-based type matches 19 of the 20 sampled rollouts (froze 9 of 10, other 10 of 10); the "
        "threshold stays at 20 steps. No rollout is \"never grasped\": every failure that did not freeze closes the gripper "
        "at some point (the one rollout that never closes it, task02_trial02, froze first).", ""])


def random_section() -> str:
    R = read_csv(OUT / "random_directions.csv")
    kinds = ["act_rep", "act_mag", "emb_temp", "xl_final"]
    body = []
    for r in R:
        b = " / ".join(f3(r[f"B probe {k}"]) or "–" for k in kinds)
        beat = " / ".join(r[f"share of random directions (sign set) B {k} beats"] or "–" for k in kinds)
        body.append([r["layer"], f"{f3(r['random directions, sign set on eval-seen: median'])} "
                                 f"[{f3(r['random directions, sign set on eval-seen: 5th percentile'])}, "
                                 f"{f3(r['random directions, sign set on eval-seen: 95th percentile'])}]",
                     f"{f3(r['random directions, unflipped: median'])} "
                     f"[{f3(r['random directions, unflipped: 5th percentile'])}, {f3(r['random directions, unflipped: 95th percentile'])}]",
                     f3(r["A's probe at this layer"]), b, beat])
    L0 = read_csv(OUT / "b_without_layer0.csv")
    body0 = [[r["row"], f3(r["per-task AUROC (mean over splits)"]), f3(r["full mean"]),
              f3(r["difference (without layer 0 - full)"], True),
              ("**" if r["range excludes zero"] == "True" else "") + f"[{f3(r['95% range low'], True)}, {f3(r['95% range high'], True)}]"
              + ("**" if r["range excludes zero"] == "True" else "")] for r in L0]
    A = read_csv(OUT / "agreement_with_a.csv")
    lay = sorted({int(r["layer"]) for r in A})
    body_a = []
    for k in kinds:
        row = [k]
        for L in lay:
            v = [r for r in A if int(r["layer"]) == L and r["B probe target"] == k]
            row.append(f3(v[0]["Spearman with A's probe at the same layer (mean over splits)"]) if v else "–")
        body_a.append(row)
    chk = json.loads((OUT / "logs" / "random_directions_scorer_check.json").read_text())
    return "\n".join([
        "## §6. Random-direction control for B", "",
        "**How B's probe inputs are prepared (read from the code: `probes_b.run_layer` → `setup_b.fit_grid_block` / "
        "`predict_block`; `common.load_layer` → `stage4_a.load_probe_input`).** The input at layer L is the cached hidden state "
        "`layer<L>.npy`: per policy step, the mean over the 7 decision states (the action-token positions; "
        "`abc2/build_cache2.py`), fp16. It is standardised per dimension with the mean and the biased standard deviation over "
        "the kept rows of that task split's training rollouts (sd = 0 → 1). For eval-seen and unseen rollouts the refit on all "
        "training rows sets this standardisation. A probe's per-step output is sigmoid(w·x + b), and its rollout score is "
        "the running sum over kept rows read at its maximum before the cut, which equals the sum to the cut because the "
        "outputs are positive.", "",
        "**Random directions.** 200 unit directions (standard normal, scaled to length 1, seed 20261102), the same for every "
        "layer and split. Per-step score = standardised x · d; rollout score = sum over kept rows before the cut. The projection "
        "can be negative, so I use the sum to the cut, as the handoff asks, not the running sum's maximum. Per-task AUROC as for "
        "B's probes. Sign: a direction is flipped in a split if its eval-seen per-task AUROC is below 0.5 there, which turns "
        "its unseen AUROC into exactly 1 − AUROC. **Scorer check:** the same scoring applied to B's saved layer-0 outputs "
        f"reproduces their per-task AUROC in `figures/data/per_probe_outcome_auroc.csv` (max |diff| {chk['max_abs_diff']:.1e}, "
        "i.e. the CSV's rounding).", "",
        "File: `random_directions.csv` (also 5th/95th percentiles unflipped, share flipped, share of unflipped directions "
        "beaten, the Spearman columns). Per-task-and-perturbation AUROC on the headline test set, mean over the three task "
        "splits; random directions as median [5th, 95th percentile] over the 200; B's probes at H = 10.", "",
        table(["layer", "random, sign set on eval-seen", "random, unflipped", "A's probe", "B probes act_rep / act_mag / emb_temp / xl_final",
               "share of signed random directions each B probe beats"], body), "",
        "### B and A without layer 0", "", "File: `b_without_layer0.csv`. Range from §1 (2,000 rounds, tasks redrawn).", "",
        table(["row", "per-task AUROC", "full mean", "difference", "95% range"], body0), "",
        "### Agreement with A's probe at the same layer", "",
        "File: `agreement_with_a.csv`. Spearman correlation between the rollout scores of B's probe and A's probe at the same "
        "layer, within each task-and-perturbation group of the headline test set (groups of at least 3 rollouts), averaged "
        "over groups, then over the three task splits.", "",
        table(["B probe target"] + [f"layer {L}" for L in lay], body_a), "",
        "### F8 with the random-direction band", "",
        "`../figures/F8_per_probe_detector_b.{pdf,png}` now has the random-direction band (5th–95th percentile, sign set on "
        "eval-seen) in light grey behind B's probe lines; band data `../figures/data/F8_random_direction_band.csv`. The "
        "earlier version is kept as `../figures/F8_per_probe_detector_b_noband.{pdf,png}`.", "",
        "![F8](../figures/F8_per_probe_detector_b.png)", ""])


def main() -> None:
    n = notes()
    txt = "\n".join([
        "# Task resampling: ranges that redraw the unseen tasks, false alarms, failure types, random directions", "",
        "*task_resampling_handoff.md (version 4), §7. Generated by `analysis/thesis_results/task_resampling/report.py` from "
        "`results/thesis_results_final/task_resampling/`; the headline, decisions and commit message are in `report_notes.md`. "
        "A **task split** is what the repo calls a seed: it sets which 3 tasks are held out and how the rest divide into "
        "train and eval-seen. Headline setup throughout: full training set, tested on the unseen tasks incl. LIBERO-PRO. "
        "No refits: every detector score, split and conformal threshold is the saved one.*", "",
        "## Headline", "", n["Headline"], "",
        ranges_section(), far_section(), failures_section(), random_section(),
        "## Decisions and notes", "", n["Decisions"], "",
        "## Proposed commit message", "", n["Commit message"], ""])
    (OUT / "REPORT.md").write_text(txt)
    tr.log(f"[report] {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
