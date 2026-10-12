"""
pipeline.py

The A / C / constraint / conformal pipeline of analysis/abc (Stages 2, 3, 4, 6), ported
to the combined corpus and run per condition and seed (more_data_collection.md Steps 8
and 9). The pipeline is frozen: scoring, features, smoothing, the probe, the c grid, row
weights, running sums, M1, the conformal band and every selection rule are analysis/abc's
own functions, imported (prep, setup_c, setup_a, cp_eval, data via data2). Only what
hard-coded the 300-rollout corpus or results/abc paths is re-implemented
(pipe2_common.py lists those). With an Index2, data.m1 / m1_cut group by CELL
(base task x family), the Step 7 "per cell" rule; on T0 it is the old per-task M1.

Per condition (D0, D1, D2) and seed: train / eval_seen / calibration sets from
data2.condition_sets, test sets from data2.test_sets (T0, T1, T2, stress_seen,
stress_unseen). Every selection is made on the condition's eval_seen set by abc's rule:
    beta per family    stage2_constraints (setup_c.choose_betas: best series' eval_seen
                       M1-per-task, mean over seeds, 4-decimal rounding as abc)
    emb layer          stage2_constraints (constraint_m1.summarise: best eval_seen POOLED
                       M1, mean over seeds, at the chosen emb_temp beta; gave 28 in abc)
    c for C's probes   stage3_c (best mean over seeds of the 33-probe mean, eval_seen M1-per-task)
    c for C's final    stage3_c (best mean over seeds, eval_seen M1-per-task)
    c for A            stage4_a (best mean over seeds of the 34-probe mean)
    A's probe          stage4_a (per seed, best eval_seen M1-per-task at c_A)
Selections that abc made once across seeds (beta, emb layer, the c's) are made once per
condition over the seeds given to --seeds; A's probe is per seed. Unseen rollouts never
choose anything.

Stages and resumability (each skips work whose file exists; cond-level summaries and
tables are cheap and are always rebuilt from the work files):
    2  stage2_constraints  <work>/<cond>/s2_seed{s}.pkl
    3  stage3_c            <work>/<cond>/c_grid_seed{s}.pkl
    4  stage4_a            <work>/<cond>/a_grid_seed{s}/<probe>.pkl  (layer-outer over all
                           requested conditions: each cache file is read once), then
                           <work>/<cond>/a_final_seed{s}.pkl (refit + shuffled control)
    6  stage6_cp           <out>/scores/<cond>_seed{s}.npz and <work>/<cond>/cp_seed{s}.pkl
Work files are redone automatically when an upstream selection they depend on changed.

Outputs (<out> = --out-dir, default results/abc2):
  tables/<cond>/*.csv     per-condition tables (below); concatenated over every condition
                          present into <out>/<name>.csv after each run.
  selections.csv          cond, seed, selection, value, criterion, score_eval_seen
  final_table.csv         cond, test_set, detector, n_seeds, m1_pooled_mean/std,
                          m1_per_task_mean/std, floor_mean, n_cells (per seed, space-sep),
                          a01_bal_acc_mean/std, a01_tpr_mean/std, a01_far_mean/std,
                          a01_det_time_mean/std, n_cal_mean   (alpha = 0.1, cut variant)
  final_m1.csv            cond, seed, detector, detail, test_set, m1_pooled, m1_per_task,
                          floor (pooled M1 of the timestep-only score), n_cells (cells in
                          M1-per-task), n, n_fail
  cp_curves.csv           cond, seed, detector, detail, test_set, alpha, variant (= cut),
                          on_alpha_grid (False for the extra 0.1), h, bal_acc, tpr, tnr,
                          far (= 1 - tnr), det_time (mean over failures of first alarm /
                          cut, 1 if none), n, n_fail, n_cal, n_d1, n_d2, t_end
  stress.csv              cond, seed, detector, detail, stress_set, block (all / swap /
                          task / demoted), alpha (0.1), n, n_fail, share_alarm_before_cut,
                          mean_cut
  stress_summary.csv      mean +- std over seeds of share_alarm_before_cut
  constraint_m1.csv       cond, variant, family, constraint, layer, beta, seed, test_set,
                          sign, signs_all_seeds, unstable, train_m1_task, m1_pooled,
                          m1_per_task, floor, n_cells, n, n_fail
  constraint_m1_summary.csv  the same, mean +- std over seeds
  c_betas.csv, c_grid.csv, c_grid_choice.csv, c_probes.csv, c_final.csv,
  c_final_weights.csv, a_grid.csv, a_grid_choice.csv, a_probes.csv, a_final.csv,
  a_shuffled.csv          as analysis/abc's, in long form with cond, seed, test_set columns
  timings.csv             wall time per stage and condition
"test_set" is one of eval_seen, T0, T1, T2 (stress is in stress.csv). Empty test sets are
skipped (e.g. on a v2-only index T1 = T2 = T0).

scores/<cond>_seed<s>.npz (one per condition and seed; N = rollouts in the index):
  rollout_ids (N,) str, index order; cond, seed; y (N,) 1 = failure; kept_len (N,)
      (kept rows are 1 <= t < kept_len); cell (N,) cell id; base_task (N,); block (N,)
  train_rids, eval_seen_rids, cal_rids: the condition's sets for this seed
  alphas (16,): cp_eval.ALPHAS (15, linspace(0.02, 0.9)) followed by 0.1
  for D in (A, C, emb):
      D_m1   (N, 520) float32  score M1 maximises over kept rows t < cut
                               (A, C: running sum of P(failure); emb: prepared z-score,
                               NaN where missing)
      D_cp   (N, 520) float32  CP score series (A, C: = D_m1; emb: running max of D_m1
                               over kept rows, NaN at t = 0)
      D_p    (N, 520) float32  per-step P(failure) (A, C only; A's is fp16-rounded)
      D_mu, D_xi (520,) float64  band centre and modulation
      D_h    (16,)  h per alpha (inf = band never alarms); D_band01 (520,) = mu + h(0.1) xi
      D_R    (n_d2,) sorted nonconformity scores; D_t_end, D_n_d1, D_n_d2
  stress_seen_cut, stress_unseen_cut: per-stress-rollout cut (only if stress rollouts exist)
  emb_sign: orientation of the emb series; selections: JSON string of the choices
  Alarm recomputation for any subset `pos` at alpha index j with cuts `cut`:
      rows  = (1 <= t < kept_len[pos]) & (t < cut)
      alarm = rows & (D_cp[pos] > D_mu + D_h[j] * D_xi)      (NaN compares False)
  and cut for M1 / CP on a set = min kept_len over that set's rollouts in the same cell.

Large per-probe arrays are not stored (disk): every probe's metrics are, and the final
detectors' per-row scores are in the npz above. Work files live in --work-dir (default
/data/tmp/abc2_scores/work, about 15 MB per (cond, seed) at N = 1000; deletable after the
run, but then a re-run recomputes).

Usage (repo root):
    source env.sh
    ABC2_GPU_FRAC=1.0 OMP_NUM_THREADS=4 python analysis/abc2/pipeline.py --cond D0 D1 D2
    python analysis/abc2/pipeline.py --cond D0 --seeds 0 --until 3     # stages 2-3 only
Paths: --index (default corpus_v3_index.json), --cache-dir (/data/tmp/abc2_features),
--splits-dir (results/abc2/splits), --out-dir (results/abc2), --work-dir.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2  # noqa: E402
from pipe2_common import Ctx, log, mean_std, read_csv, record_timing, write_csv  # noqa: E402

TABLES = ("constraint_m1", "constraint_m1_summary", "c_betas", "c_grid", "c_grid_choice",
          "c_probes", "c_final", "c_final_weights", "a_grid", "a_grid_choice", "a_probes",
          "a_final", "a_shuffled", "final_m1", "cp_curves", "stress", "selections")


def selection_rows(cond: str, seeds, sel: dict) -> list[dict]:
    rows = []
    for s in seeds:
        def add(name, value, crit, score):
            rows.append({"cond": cond, "seed": s, "selection": name, "value": value,
                         "criterion": crit, "score_eval_seen": score})
        for fam, b in sel["betas"].items():
            add(f"beta_{fam}", b, "best series' eval_seen M1-per-task, mean over seeds (4 d.p.)",
                sel["beta_scores"][fam])
        add("emb_layer", f"emb_temp[{sel['emb_layer']}]",
            "best eval_seen pooled M1, mean over seeds, at the chosen emb_temp beta", sel["emb_score"])
        if "c_probes" in sel:
            add("c_C_probes", sel["c_probes"], "mean over seeds of 33-probe mean eval_seen M1-per-task",
                sel["c_probes_score"])
            add("c_C_final", sel["c_final"], "mean over seeds of eval_seen M1-per-task",
                sel["c_final_score"])
        if "c_A" in sel:
            add("c_A", sel["c_A"], "mean over seeds of 34-probe mean eval_seen M1-per-task",
                sel["c_A_score"])
            add("A_probe", sel["a_probe"][s], "best eval_seen M1-per-task at c_A (this seed)",
                sel["a_probe_score"][s])
    return rows


def assemble(ctx: Ctx) -> None:
    """Concatenate tables/<cond>/<name>.csv over all conditions; summary tables."""
    conds = sorted(p.name for p in ctx.tables_dir.iterdir() if p.is_dir())
    for name in TABLES:
        rows = [r for c in conds for r in read_csv(ctx.tables_dir / c / f"{name}.csv")]
        if rows:
            write_csv(ctx.out_dir / f"{name}.csv", rows)

    m1 = read_csv(ctx.out_dir / "final_m1.csv")
    cp = [r for r in read_csv(ctx.out_dir / "cp_curves.csv") if r["on_alpha_grid"] == "False"]
    final = []
    keys = list(dict.fromkeys((r["cond"], r["test_set"], r["detector"]) for r in m1))
    for cond, ts, det in keys:
        mr = [r for r in m1 if (r["cond"], r["test_set"], r["detector"]) == (cond, ts, det)]
        cr = [r for r in cp if (r["cond"], r["test_set"], r["detector"]) == (cond, ts, det)]
        row = {"cond": cond, "test_set": ts, "detector": det, "n_seeds": len(mr)}
        for k in ("m1_pooled", "m1_per_task"):
            row[f"{k}_mean"], row[f"{k}_std"] = mean_std([float(r[k]) for r in mr])
        row["floor_mean"] = mean_std([float(r["floor"]) for r in mr])[0]
        row["n_cells"] = " ".join(r["n_cells"] for r in mr)
        for k in ("bal_acc", "tpr", "far", "det_time"):
            row[f"a01_{k}_mean"], row[f"a01_{k}_std"] = mean_std([float(r[k]) for r in cr])
        row["n_cal_mean"] = mean_std([float(r["n_cal"]) for r in cr])[0]
        row["n_mean"] = mean_std([float(r["n"]) for r in mr])[0]
        final.append(row)
    if final:
        write_csv(ctx.out_dir / "final_table.csv", final)

    st = read_csv(ctx.out_dir / "stress.csv")
    summ = []
    for key in dict.fromkeys((r["cond"], r["detector"], r["stress_set"], r["block"]) for r in st):
        rs = [r for r in st if (r["cond"], r["detector"], r["stress_set"], r["block"]) == key]
        mu, sd = mean_std([float(r["share_alarm_before_cut"]) for r in rs])
        summ.append({"cond": key[0], "detector": key[1], "stress_set": key[2], "block": key[3],
                     "alpha": rs[0]["alpha"], "n_seeds": len(rs), "n_mean": mean_std([float(r["n"]) for r in rs])[0],
                     "share_alarm_before_cut_mean": mu, "share_alarm_before_cut_std": sd})
    if summ:
        write_csv(ctx.out_dir / "stress_summary.csv", summ)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cond", nargs="+", default=["D0", "D1", "D2"], choices=list(data2.CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(data2.SEEDS))
    ap.add_argument("--index", type=Path, default=data2.INDEX_V3)
    ap.add_argument("--cache-dir", type=Path, default=data2.CACHE_DIR)
    ap.add_argument("--splits-dir", type=Path, default=data2.SPLITS_DIR)
    ap.add_argument("--out-dir", type=Path, default=data2.RESULTS_DIR)
    ap.add_argument("--work-dir", type=Path, default=Path("/data/tmp/abc2_scores/work"))
    ap.add_argument("--demotions", type=Path, default=data2.DEMOTIONS)
    ap.add_argument("--until", type=int, default=6, choices=(2, 3, 4, 6),
                    help="last stage to run (earlier ones are skipped where done)")
    ap.add_argument("--n-jobs", type=int, default=4, help="joblib processes for stage 3")
    args = ap.parse_args(argv)

    import stage2_constraints
    import stage3_c
    import stage4_a
    import stage6_cp

    t_all = time.time()
    ctx = Ctx(index_path=args.index, cache_dir=args.cache_dir, splits_dir=args.splits_dir,
              out_dir=args.out_dir, work_dir=args.work_dir, demotions_path=args.demotions,
              seeds=tuple(args.seeds), n_jobs=args.n_jobs)
    log(f"index {args.index}: {ctx.idx.n} rollouts; conds {args.cond}; seeds {ctx.seeds}; "
        f"demoted {ctx.demotions or 'none'}; out {args.out_dir}; work {args.work_dir}")
    (ctx.out_dir / "run_meta.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=1))

    sel = {}
    for cond in args.cond:
        t0 = time.time()
        sel[cond] = stage2_constraints.run(ctx, cond)
        record_timing(ctx, "stage2_total", cond, time.time() - t0)
        if args.until >= 3:
            t0 = time.time()
            sel[cond].update(stage3_c.run(ctx, cond, sel[cond]))
            record_timing(ctx, "stage3_total", cond, time.time() - t0)
    if args.until >= 4:
        t0 = time.time()
        stage4_a.grid(ctx, args.cond)
        for cond in args.cond:
            sel[cond].update(stage4_a.run(ctx, cond))
        record_timing(ctx, "stage4_total", "+".join(args.cond), time.time() - t0)
    if args.until >= 6:
        for cond in args.cond:
            stage6_cp.run(ctx, cond, sel[cond])
    for cond in args.cond:
        write_csv(ctx.tables_dir / cond / "selections.csv", selection_rows(cond, ctx.seeds, sel[cond]))
    assemble(ctx)
    record_timing(ctx, "all", "+".join(args.cond), time.time() - t_all)
    log(f"done ({time.time() - t_all:.0f} s); tables in {ctx.out_dir}")


if __name__ == "__main__":
    main()
