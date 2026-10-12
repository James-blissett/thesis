"""
regression_gate.py

Step 8 regression gate: the ported pipeline (pipeline.py), run for D0 on a v2-only index
with the OLD feature cache, must reproduce results/abc on T0 (= abc's unseen set), mean
over seeds:

    emb_temp[28] alone, M1-per-task          final_table.csv           +- 0.005
    C final detector, M1-per-task            final_table.csv           +- 0.005
    A final detector, M1-per-task            final_table.csv           +- 0.005
    emb_temp[28], balanced accuracy, a = 0.1, cut variant   cp_curves.csv  +- 0.01
    A, the same                                              cp_curves.csv  +- 0.01
    C, the same                                              cp_curves.csv  +- 0.01

Steps: write a v2-only index ({"rollouts": data2.v2_entries()}) into the scratch dir;
make splits for it with data2.make_split2 into scratch and check they equal
results/abc/splits; run pipeline.main for D0 into scratch (never into results/abc2);
compare. Also reported (informational, not gated): the selections against abc's
(betas, emb layer, c's, A's probe per seed), and per-seed differences of every
constraint's unseen M1, C's probes, A's probes, and the CP curves at all alphas.

Outputs: results/abc2/regression_gate.json and regression_gate.md.

Usage (repo root; ~10 min with a 6 GB GPU share):
    source env.sh
    python analysis/abc2/regression_gate.py [--scratch /data/tmp/abc2_dev/pipeline] [--skip-run]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2  # noqa: E402

ABC = Path("results/abc")
OUT_JSON = Path("results/abc2/regression_gate.json")
OUT_MD = Path("results/abc2/regression_gate.md")
SEEDS = (0, 1, 2)


def rcsv(p):
    return list(csv.DictReader(Path(p).open()))


def prepare(scratch: Path) -> dict:
    idx_path = scratch / "idx_v2.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps({"rollouts": data2.v2_entries()}))
    idx = data2.load_index2(idx_path)
    sdir = scratch / "splits"
    sdir.mkdir(parents=True, exist_ok=True)
    checks = {}
    for s in SEEDS:
        sp = data2.make_split2(s, idx)
        old = json.loads((ABC / "splits" / f"seed{s}.json").read_text())
        same = {n: sorted(sp[n]) == sorted(old[n]) for n in ("train", "eval_seen", "unseen")}
        same["unseen_tasks"] = sp["unseen_tasks"] == old["unseen_tasks"]
        same["stress_empty"] = len(sp["stress"]) == 0
        checks[f"seed{s}"] = same
        (sdir / f"seed{s}.json").write_text(json.dumps(sp, indent=1))
    ok = all(all(v.values()) for v in checks.values())
    print(f"[gate] v2-only index {idx.n} rollouts; splits equal old splits: {ok} {checks}")
    if not ok:
        raise SystemExit("make_split2 on the v2-only index does not reproduce results/abc/splits")
    return {"index": str(idx_path), "splits_dir": str(sdir), "split_checks": checks}


def compare(out: Path) -> dict:
    t = out / "tables" / "D0"
    final = {r["detector"]: r for r in rcsv(ABC / "final_table.csv")}
    cp_ref = rcsv(ABC / "cp_curves.csv")
    m1 = rcsv(t / "final_m1.csv")
    cp = rcsv(t / "cp_curves.csv")

    def mine_m1(det):
        v = [float(r["m1_per_task"]) for r in m1 if r["detector"] == det and r["test_set"] == "T0"]
        return float(np.mean(v)), v

    def mine_ba(det, alpha=0.1, seed=None):
        v = [float(r["bal_acc"]) for r in cp if r["detector"] == det and r["test_set"] == "T0"
             and np.isclose(float(r["alpha"]), alpha) and (seed is None or int(r["seed"]) == seed)]
        return float(np.mean(v)), v

    def ref_ba(det, alpha=0.1, seed=None):
        v = [float(r["bal_acc"]) for r in cp_ref if r["detector"] == det and r["variant"] == "cut"
             and np.isclose(float(r["alpha"]), alpha) and (seed is None or int(r["seed"]) == seed)]
        # cp_curves.csv holds alpha = 0.1 once per (detector, seed, variant)
        return float(np.mean(v)), v

    gate = []
    for label, det, ref_det, tol in (("emb_temp[28] alone, M1-per-task", "emb", "emb_temp[28] alone", 0.005),
                                     ("C final, M1-per-task", "C", "C", 0.005),
                                     ("A final, M1-per-task", "A", "A", 0.005)):
        got, per = mine_m1(det)
        ref = float(final[ref_det]["m1_per_task_unseen_mean"])
        gate.append({"quantity": label, "reference": ref, "got": got, "per_seed": per,
                     "diff": got - ref, "tolerance": tol, "pass": abs(got - ref) <= tol,
                     "source": "results/abc/final_table.csv m1_per_task_unseen_mean"})
    for label, det, ref_det, tol in (("emb_temp[28], balanced accuracy, alpha 0.1, cut", "emb", "emb_temp[28] alone", 0.01),
                                     ("A, balanced accuracy, alpha 0.1, cut", "A", "A", 0.01),
                                     ("C, balanced accuracy, alpha 0.1, cut", "C", "C", 0.01)):
        got, per = mine_ba(det)
        ref, ref_per = ref_ba(ref_det)
        ref_ft = float(final[ref_det]["a01_cut_bal_acc_mean"])
        gate.append({"quantity": label, "reference": ref, "reference_final_table": ref_ft,
                     "got": got, "per_seed": per, "reference_per_seed": ref_per,
                     "diff": got - ref, "tolerance": tol, "pass": abs(got - ref) <= tol,
                     "source": "results/abc/cp_curves.csv (variant cut, alpha 0.1), mean over seeds"})

    # --- informational parity ----------------------------------------------------------
    info = {}
    sel = {(r["selection"], int(r["seed"])): r["value"] for r in rcsv(t / "selections.csv")}
    betas = {r["family"]: r["beta"] for r in rcsv(ABC / "c_betas.csv") if r["chosen"] == "True"}
    cgc = rcsv(ABC / "c_grid_choice.csv")
    a_final = {int(r["seed"]): r for r in rcsv(ABC / "a_final.csv")}
    ref_sel = {**{f"beta_{f}": b for f, b in betas.items()}, "emb_layer": "emb_temp[28]",
               "c_C_probes": next(r["c"] for r in cgc if r["chosen_probes"] == "True"),
               "c_C_final": next(r["c"] for r in cgc if r["chosen_final"] == "True"),
               "c_A": a_final[0]["c"]}
    sel_cmp = {}
    for k, v in ref_sel.items():
        got = sel[(k, 0)]
        same = (float(got) == float(v)) if k.startswith("c_") else got == v
        sel_cmp[k] = {"abc": v, "port": got, "same": same}
    for s in SEEDS:
        sel_cmp[f"A_probe_seed{s}"] = {"abc": a_final[s]["probe"], "port": sel[("A_probe", s)],
                                       "same": a_final[s]["probe"] == sel[("A_probe", s)]}
    info["selections"] = sel_cmp

    # constraint M1 per seed on unseen: abc constraint_m1.csv vs port D0 T0
    ref_c = {(r["variant"], r["constraint"], int(r["layer"]), r["beta"]): r
             for r in rcsv(ABC / "constraint_m1.csv")}
    diffs, signs_diff, n = [], 0, 0
    for r in rcsv(t / "constraint_m1.csv"):
        if r["test_set"] not in ("T0", "eval_seen"):
            continue
        key = (r["variant"], r["constraint"], int(r["layer"]), r["beta"])
        if key not in ref_c:
            continue
        lab = "unseen" if r["test_set"] == "T0" else "seen"
        for met in ("pooled", "per_task"):
            a = float(ref_c[key][f"{met}_{lab}_s{r['seed']}"])
            diffs.append(abs(float(r[f"m1_{met}"]) - a))
        n += 1
        if r["test_set"] == "T0" and r["signs_all_seeds"] != ref_c[key]["sign"]:
            signs_diff += 1
    info["constraint_m1"] = {"rows_compared": n, "max_abs_diff_per_seed": float(max(diffs)),
                             "note": "abc's CSV is rounded to 4 d.p.",
                             "n_sign_strings_differing": signs_diff // len(SEEDS)}

    # C probes per seed (unseen per-task) at the chosen c
    ref_p = {r["probe"]: r for r in rcsv(ABC / "c_probes.csv")}
    d = [abs(float(r["m1_per_task"]) - float(ref_p[r["probe"]][f"per_task_unseen_s{r['seed']}"]))
         for r in rcsv(t / "c_probes.csv") if r["test_set"] == "T0"]
    info["c_probes_max_abs_diff_unseen_per_task"] = float(max(d))
    ref_a = {r["probe"]: r for r in rcsv(ABC / "a_probes.csv")}
    d = [abs(float(r["m1_per_task"]) - float(ref_a[r["probe"]][f"per_task_unseen_s{r['seed']}"]))
         for r in rcsv(t / "a_probes.csv") if r["test_set"] == "T0"]
    info["a_probes_max_abs_diff_unseen_per_task"] = float(max(d))
    info["a_probes_mean_abs_diff_unseen_per_task"] = float(np.mean(d))

    # CP curves at every alpha and seed
    cpd = {}
    for det, ref_det in (("A", "A"), ("C", "C"), ("emb", "emb_temp[28] alone")):
        dd = []
        for r in cp:
            if r["detector"] != det or r["test_set"] != "T0":
                continue
            ref = [float(q["bal_acc"]) for q in cp_ref if q["detector"] == ref_det
                   and q["variant"] == "cut" and int(q["seed"]) == int(r["seed"])
                   and abs(float(q["alpha"]) - float(r["alpha"])) < 1e-3]
            if ref:
                dd.append(abs(float(r["bal_acc"]) - ref[0]))
        cpd[det] = {"max_abs_diff_bal_acc_all_alpha_seed": float(max(dd)), "n": len(dd)}
    info["cp_curves"] = cpd
    return {"gate": gate, "passed": all(g["pass"] for g in gate), "parity": info}


def write_md(res: dict) -> None:
    lines = ["# Step 8 regression gate (D0 on T0 vs results/abc)", "",
             f"**{'PASSED' if res['passed'] else 'REGRESSION GATE FAILED'}** — run "
             f"{res['utc']}. Stage wall times (timings.csv, all runs; a re-run skips finished work): "
             + ", ".join(f"{t['stage']} {float(t['seconds']):.0f} s" for t in res.get("timings", [])
                         if float(t["seconds"]) >= 1) + ".", "",
             "Unseen = T0, mean over seeds 0-2. Port: analysis/abc2/pipeline.py on a v2-only "
             "index, old cache /data/tmp/abc_features, splits from data2.make_split2 (equal "
             "to results/abc/splits).", "",
             "| Quantity | Reference | Port | Diff | Tolerance | Pass | Port per seed |",
             "|---|---|---|---|---|---|---|"]
    for g in res["gate"]:
        lines.append(f"| {g['quantity']} | {g['reference']:.4f} | {g['got']:.4f} | "
                     f"{g['diff']:+.4f} | ±{g['tolerance']} | {'yes' if g['pass'] else 'NO'} | "
                     f"{', '.join(f'{v:.4f}' for v in g['per_seed'])} |")
    p = res["parity"]
    lines += ["", "## Informational parity (not gated)", "",
              "| Selection | abc | port | same |", "|---|---|---|---|"]
    for k, v in p["selections"].items():
        lines.append(f"| {k} | {v['abc']} | {v['port']} | {v['same']} |")
    lines += ["",
              f"- Constraint M1 per seed (eval_seen and unseen, pooled and per-task): "
              f"{p['constraint_m1']['rows_compared']} rows, max |diff| "
              f"{p['constraint_m1']['max_abs_diff_per_seed']:.5f} (abc CSV rounded to 4 d.p.); "
              f"sign strings differing: {p['constraint_m1']['n_sign_strings_differing']}.",
              f"- C probes, unseen per-task per seed: max |diff| {p['c_probes_max_abs_diff_unseen_per_task']:.5f}.",
              f"- A probes, unseen per-task per seed: max |diff| {p['a_probes_max_abs_diff_unseen_per_task']:.5f} "
              f"(mean {p['a_probes_mean_abs_diff_unseen_per_task']:.5f}; GPU L-BFGS).",
              "- CP balanced accuracy, every alpha and seed (cut): " + "; ".join(
                  f"{d} max |diff| {v['max_abs_diff_bal_acc_all_alpha_seed']:.4f} (n={v['n']})"
                  for d, v in p["cp_curves"].items()) + ".",
              "", f"Split check: {json.dumps(res['prep']['split_checks'])}"]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=Path("/data/tmp/abc2_dev/pipeline"))
    ap.add_argument("--cache-dir", type=Path, default=data2.OLD_CACHE_DIR)
    ap.add_argument("--skip-run", action="store_true", help="only compare existing outputs")
    args = ap.parse_args()

    prep = prepare(args.scratch)
    out = args.scratch / "out"
    t0 = time.time()
    if not args.skip_run:
        import pipeline
        pipeline.main(["--cond", "D0", "--seeds", *map(str, SEEDS), "--index", prep["index"],
                       "--cache-dir", str(args.cache_dir), "--splits-dir", prep["splits_dir"],
                       "--out-dir", str(out), "--work-dir", str(args.scratch / "work"),
                       "--demotions", str(args.scratch / "no_demotions.json")])
    res = compare(out)
    res.update({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prep": prep,
                "out_dir": str(out)})
    if not args.skip_run:
        res["runtime_s"] = time.time() - t0
    tim = out / "timings.csv"
    if tim.exists():
        res["timings"] = rcsv(tim)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    write_md(res)
    for g in res["gate"]:
        print(f"   {g['quantity']:52s} ref {g['reference']:.4f} got {g['got']:.4f} "
              f"diff {g['diff']:+.4f} {'PASS' if g['pass'] else 'FAIL'}")
    print(f"[gate] {'PASSED' if res['passed'] else 'FAILED'} -> {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
