"""
full_compare.py

Background run (results_collection_handoff.md §6), after both training sets are done:

  comparison  each primary detector (A's, C's and B's unweighted means; B at every H) and
              each baseline, original vs full training set: per-task AUROC and balanced
              accuracy at alpha 0.1 on both unseen test sets, with the paired bootstrap
              (bootstrap2, same procedure as paired.py: full minus original, 1,000
              resamples within task (or task and perturbation cell) within seed, each
              seed's conformal band held fixed), plus the stress-set share alarmed.
              -> results/thesis_results_full/tables/training_set_comparison.csv
              (this becomes the training-data ablation; tables.py t6 appends it to T6)
  selections  what the selections chose on each training set's eval-seen, side by side:
              beta per family, c for every model, the best-constraint layer, unstable
              constraints, B's probe count and grid c, A's / C's / B's best probe per seed,
              MLP weight decay.
              -> results/thesis_results_full/tables/selections_comparison.csv and
                 results/thesis_results_full/selections_report.md

Usage: THESIS_POOL=D2 python analysis/thesis_results/full_compare.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["THESIS_POOL"] = "D2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import REPO, SEEDS, load_pkl, read_csv, write_csv  # noqa: E402

ORIG = REPO / "results/thesis_results"
FULL = REPO / "results/thesis_results_full"
W_ORIG = Path("/data/tmp/thesis_results_work/original")
W_FULL = Path("/data/tmp/thesis_results_work/full")
KEYS = [("A_mean", "A, unweighted mean of 34 probe outputs"), ("C_mean", "C, unweighted mean of 33 probe outputs"),
        ("B_mean_h0", "B, unweighted mean of all grid probe outputs (124 original / 116 full), H = 0"),
        ("B_mean_h10", "B, unweighted mean of all grid probe outputs (124 original / 116 full), H = 10"),
        ("B_mean_h25", "B, unweighted mean of all grid probe outputs (124 original / 116 full), H = 25"),
        ("emb_best", "best single untrained constraint (running maximum)"),
        ("token_entropy", "token mean entropy"), ("token_maxprob", "token max-probability"),
        ("failurespot", "FailureSpot signal s"), ("failurespot_neg", "FailureSpot signal -s")]


def comparison() -> list:
    import bootstrap2 as b2
    rows = []
    fo = [W_ORIG / "npz" / f"seed{s}.npz" for s in SEEDS]
    ff = [W_FULL / "npz" / f"seed{s}.npz" for s in SEEDS]
    stress = {}
    for tag, base in (("original", ORIG), ("full", FULL)):
        for r in read_csv(base / "tables" / "data" / "detector_stress_per_seed.csv"):
            if r["stress_set"] == "stress" and r["block"] == "all":
                stress.setdefault((tag, r["key"]), []).append(float(r["share_alarm_before_cut"]))
    for ts in ("T0", "T2"):
        y0 = None
        for key, lab in KEYS:
            nkey = None
            try:
                x = b2.load_side(ff, key, ts)
                y = b2.load_side(fo, key, ts)
            except SystemExit as e:
                rows.append({"detector": lab, "test set": common.TEST_SETS[ts], "metric": "", "note": f"missing: {e}"})
                continue
            if y0 is None:
                y0 = y
                draws = b2.make_draws(y0, b2.N_BOOT, b2.RNG_SEED)
            r = b2.compare(x, y, draws, test_set=ts, n_boot=b2.N_BOOT, rng_seed=b2.RNG_SEED,
                           comparison="full minus original training set")
            for m in r["rows"]:
                if m["metric"] not in ("m1_per_task", "bal_acc"):
                    continue
                name = {"m1_per_task": common.PER_TASK_LABEL[ts], "bal_acc": "balanced accuracy at alpha 0.1"}[m["metric"]]
                rows.append({"detector": lab, "test set": common.TEST_SETS[ts], "metric": name,
                             "original training set": round(m["y_point"], 4), "full training set": round(m["x_point"], 4),
                             "difference (full minus original)": round(m["diff_point"], 4),
                             "95% interval low": round(m["diff_ci_lo"], 4), "95% interval high": round(m["diff_ci_hi"], 4),
                             "verdict": m["verdict"].replace("improvement", "full higher (interval above zero)")
                                         .replace("significant worsening", "full lower (interval below zero)")
                                         .replace("no significant change", "interval includes zero"),
                             "original per seed": m["y_point_per_seed"], "full per seed": m["x_point_per_seed"],
                             "row key": key})
    for key, lab in KEYS:
        o, f_ = stress.get(("original", key)), stress.get(("full", key))
        if o and f_:
            rows.append({"detector": lab, "test set": "stress set", "metric": "share alarmed before the cut at alpha 0.1",
                         "original training set": round(float(np.mean(o)), 4), "full training set": round(float(np.mean(f_)), 4),
                         "difference (full minus original)": round(float(np.mean(f_) - np.mean(o)), 4),
                         "verdict": "no bootstrap (nearly every stress rollout fails)",
                         "original per seed": " ".join(f"{v:.4f}" for v in o), "full per seed": " ".join(f"{v:.4f}" for v in f_),
                         "row key": key})
    return rows


def sel_for(work: Path, pool: str) -> dict:
    out = {}
    rows = [r for r in read_csv(REPO / "results/abc2/selections.csv") if r["cond"] == pool]
    for r in rows:
        if r["selection"].startswith("beta_") or r["selection"] in ("c_C_probes", "c_C_final", "c_A", "emb_layer"):
            out[{"beta_action": "smoothing beta, action family", "beta_emb_temp": "smoothing beta, emb_temp family",
                 "beta_cross_layer": "smoothing beta, cross-layer family", "c_C_probes": "c, C's probes",
                 "c_C_final": "c, C's 132-input logistic regression", "c_A": "c, A's probes",
                 "emb_layer": "untrained constraint layer, pooled rule (earlier)"}[r["selection"]]] = r["value"]
        if r["selection"] == "A_probe":
            out.setdefault("A's single best probe per seed", []).append(r["value"])
    out["A's single best probe per seed"] = " / ".join(out.get("A's single best probe per seed", []))
    s2 = read_csv(REPO / "results/abc2/constraint_m1_summary.csv")
    betas = {k.split(", ")[0]: v for k, v in out.items()}
    fam = {"act_mag": "action", "act_dir": "action", "grip_flip": "action", "act_rep": "action", "emb_temp": "emb_temp",
           "xl_adj": "cross-layer", "xl_final": "cross-layer", "xl_spread": "cross-layer"}
    unst = sorted({f"{r['constraint']}[{r['layer']}]" if r["layer"] != "-1" else r["constraint"] for r in s2
                   if r["cond"] == pool and r["variant"] == "primary" and r["constraint"] in fam and r["test_set"] == "eval_seen"
                   and r["unstable"] == "True" and r["beta"] == out[f"smoothing beta, {fam[r['constraint']]} family"]})
    out["sign-unstable constraints at the chosen beta"] = f"{len(unst)}: " + " ".join(unst)

    def det(k):
        p = work / "detectors" / f"{k}.pkl"
        return load_pkl(p) if p.exists() else None
    for k, lab in (("A_combiner", "c, A's trained combiner"), ("C_combiner", "c, C's trained combiner"),
                   ("B_combiner_h0", "c, B's trained combiner, H = 0"), ("B_combiner_h25", "c, B's trained combiner, H = 25"),
                   ("B_combiner_h10_refit", "c, B's trained combiner, H = 10 (refit)")):
        d = det(k)
        out[lab] = d["selection"]["c"] if d else ""
    for k, lab in (("C_best", "C's single best probe per seed"), ("B_best_h10", "B's single best probe per seed, H = 10"),
                   ("B_best_h0", "B's single best probe per seed, H = 0"), ("B_best_h25", "B's single best probe per seed, H = 25")):
        d = det(k)
        out[lab] = " / ".join(d["selection"][s]["probe"] for s in SEEDS) if d else ""
    d = det("emb_best")
    out["best untrained constraint, per-task rule"] = f"emb_temp[{d['selection']['layer']}]" if d else ""
    for k, lab in (("token_entropy", "smoothing beta, token mean entropy"), ("token_maxprob", "smoothing beta, token max-probability")):
        d = det(k)
        out[lab] = d["selection"]["beta"] if d else ""
    m = work / "b_probes" / "meta.json"
    if m.exists():
        mm = json.loads(m.read_text())
        out["B's grid c"] = mm["c"]
        out["B's probe count per H"] = mm["n_probes_per_H"]
        out["B's skipped (sign-unstable) targets"] = " ".join(mm["unstable"])
    for k, lab in (("C_mlp", "weight decay, C's 2-layer MLP"), ("B_mlp_h10", "weight decay, B's 2-layer MLP")):
        p = work / "mlp" / f"{k}.pkl"
        out[lab] = load_pkl(p)["selection"]["wd"] if p.exists() else ""
    return out


def selections() -> list:
    o, f_ = sel_for(W_ORIG, "D0"), sel_for(W_FULL, "D2")
    return [{"selection": k, "original training set": o.get(k, ""), "full training set": f_.get(k, ""),
             "same": str(o.get(k, "")) == str(f_.get(k, ""))} for k in dict.fromkeys(list(o) + list(f_))]


def main() -> None:
    rows = comparison()
    write_csv(FULL / "tables" / "training_set_comparison.csv", rows)
    sel = selections()
    write_csv(FULL / "tables" / "selections_comparison.csv", sel)
    lines = ["# Selections: original vs full training set", "",
             "*Background run (results_collection_handoff.md §6). Every selection was re-made on each training set's own "
             "eval-seen rollouts by the rules the Methods describe; unseen and stress rollouts chose nothing. The thesis "
             "Methods quotes the original-set values. Source: `tables/selections_comparison.csv`.*", "",
             "| Selection | Original training set | Full training set | Same |", "|---|---|---|---|"]
    for r in sel:
        lines.append(f"| {r['selection']} | {r['original training set']} | {r['full training set']} | "
                     f"{'yes' if r['same'] else 'no'} |")
    (FULL / "selections_report.md").write_text("\n".join(lines) + "\n")
    print(f"comparison rows {len(rows)}, selections {len(sel)}")


if __name__ == "__main__":
    main()
