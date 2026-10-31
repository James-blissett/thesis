"""
eval3.py

Evaluation of the deep-MLP grid (deep_mlp_handoff.md v1, §3; output formats in
analysis/abc3/FORMATS.md, "eval3 outputs"). CPU only, no training.

For every group (input A/C, depth 2/6/12, pool D0/D2, seed 0/1/2) the regularisation
setting is chosen from the four work files of train3.py (common3.REG_GRID): the "ok" fit
with the highest best_score (best-epoch eval_seen M1-per-task); ties go to the first in
REG_GRID order. A group whose four fits all failed is reported as failed (no npz keys,
listed in the npz's `failed` JSON, eval_status.json groups_failed).

The chosen fit's stored best-epoch P(failure) is scored exactly as abc2's
stage10_mlp.Run.outputs: score = setup_c.running_sum(p, kept) (float32); M1 rows by
pipe2_common.m1_safe / m1_row on train (extra here: the pool's train rollouts, M1 only),
eval_seen, T0, T1, T2; conformal band cp_eval.fit_band calibrated on the pool's successful
eval_seen rollouts with default_rng(stage6_cp.CAL_SPLIT_SEED + seed), h for the 16 alphas of
stage6_cp.ALPHAS (15 grid levels, then 0.1), cp_eval.evaluate_cp with the cut variant
(data2.m1_cut per test set); stress at alpha 0.1 with stage6_cp.stress_cuts / alarms.

Reference detectors (no refitting): A-linear / C-linear = keys A / C of
results/abc2/scores/<pool>_seed<s>.npz, A-mlp-abc2 / C-mlp-abc2 = keys A-MLP / C-MLP of
results/abc2/scores/<pool>_seed<s>_mlp.npz. Their M1 rows are recomputed from the npz
<key>_m1 arrays for all five test sets (their eval_seen / T0 / T1 / T2 values are checked
against abc2's final_m1.csv / mlp_final_m1.csv, |diff| <= 1e-6, -> eval_check.json); CP
and stress rows are copied from abc2's cp_curves.csv / mlp_cp_curves.csv and stress.csv /
mlp_stress.csv with the detector renamed.

Checks (eval_check.json): the reference identity check above, and bootstrap2's own
identity check (load_side on T0 + check_identity against this run's final_m1.csv,
cp_curves.csv and final_table.csv) for every deep detector and every reference detector,
so the npz files are known to feed bootstrap2.paired_bootstrap correctly.

Outputs (<out> = --out-dir, default results/abc3; every file is rewritten on each run):
    scores/<pool>_seed<s>_deep.npz, final_m1.csv, cp_curves.csv, stress.csv,
    final_table.csv, selections.csv, grid.csv, train_log.csv, eval_check.json,
    eval_status.json   (formats: FORMATS.md)

Usage (repo root):
    source env.sh
    python analysis/abc3/eval3.py [--work-dir /data/tmp/abc3_work] [--out-dir results/abc3]
Exit status 2 if a work file of the 144-fit grid is missing or stale (spec mismatch),
1 if a check fails or anything else goes wrong (reason in eval_status.json), else 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only: never touch the GPU

import numpy as np  # noqa: E402

import common3 as c3  # noqa: E402
from common3 import DEPTHS, INPUTS, POOLS, REG_GRID, SEEDS, det_key, reg_tag  # noqa: E402
import train3  # noqa: E402

import bootstrap2  # noqa: E402
import cp_eval as abc_cp  # noqa: E402
import data2  # noqa: E402
import setup_c as abc_c  # noqa: E402
import stage6_cp  # noqa: E402
from data2 import m1_cut  # noqa: E402
from pipe2_common import (EVAL_SETS, STRESS_SETS, Ctx, load_pkl, log, m1_row, m1_safe,  # noqa: E402
                          mean_std, read_csv, write_csv)

ALPHAS = stage6_cp.ALPHAS                    # 15 grid levels, then 0.1
TEST_SETS = ("train",) + tuple(EVAL_SETS)    # train: M1 only
CHECK_TOL = 1e-6
DEEP = [det_key(i, d) for i in INPUTS for d in DEPTHS]
# reference detector -> (input, depth, model, abc2 key, abc2 table prefix, abc2 npz suffix)
REFS = {"A-linear": ("A", 0, "linear", c3.LINEAR["A"], "", ""),
        "C-linear": ("C", 0, "linear", c3.LINEAR["C"], "", ""),
        "A-mlp-abc2": ("A", 2, "plain-mlp-abc2", c3.ABC2_MLP["A"], "mlp_", "_mlp"),
        "C-mlp-abc2": ("C", 2, "plain-mlp-abc2", c3.ABC2_MLP["C"], "mlp_", "_mlp")}
DET_ORDER = [d for i in INPUTS for d in (f"{i}-linear", f"{i}-mlp-abc2", *(det_key(i, k) for k in DEPTHS))]
D_IN = {"A": 4096, "C": 132}                 # input widths (checked against the work files)
M1_COLS = ["cond", "seed", "detector", "detail", "test_set", "m1_pooled", "m1_per_task", "floor",
           "n_cells", "n", "n_fail"]
CP_COLS = ["cond", "seed", "detector", "detail", "test_set", "alpha", "variant", "on_alpha_grid", "h",
           "bal_acc", "tpr", "tnr", "far", "det_time", "n", "n_fail", "n_cal", "n_d1", "n_d2", "t_end"]
ST_COLS = ["cond", "seed", "detector", "detail", "stress_set", "block", "alpha", "n", "n_fail",
           "share_alarm_before_cut", "mean_cut"]
EXTRA = ["input", "depth", "model"]


class GridIncomplete(Exception):
    """A work file of the grid is missing or has a stale spec."""


def det_info(det: str) -> dict:
    if det in REFS:
        i, d, m = REFS[det][:3]
        return {"input": i, "depth": d, "model": m}
    i, d = det.split("-d")
    return {"input": i, "depth": int(d), "model": "resmlp"}


def ref_n_params(det: str) -> int:
    i, _, model = REFS[det][:3]
    d = D_IN[i]
    return d + 1 if model == "linear" else d * 256 + 256 + 256 * 256 + 256 + 256 + 1


def make_ctx(work_dir: Path, out_dir: Path) -> Ctx:
    """train3.make_ctx with this run's --out-dir."""
    return Ctx(index_path=c3.REPO / data2.INDEX_V3, cache_dir=data2.CACHE_DIR,
               splits_dir=c3.ABC2_RESULTS / "splits", out_dir=out_dir,
               work_dir=work_dir / "ctx", demotions_path=c3.ABC2_RESULTS / "demotions.json",
               seeds=SEEDS, n_jobs=1)


def ordered(row: dict, cols: list) -> dict:
    return {k: row.get(k, "") for k in cols}


def score_of(p: np.ndarray, kept: np.ndarray) -> np.ndarray:
    return abc_c.running_sum(p, kept).astype(np.float32)


# --- Loading and selection ---------------------------------------------------------------------
def load_grid(grid: train3.Grid) -> dict:
    """{fit tuple: work dict (without the state_dict)}; raises GridIncomplete if any is missing."""
    bad = []
    for f in c3.all_fits():
        if not grid.done(*f):
            i, d, c, s, p, wd = f
            path = grid.path(c, i, d, s, p, wd)
            bad.append(f"{path} ({'missing' if not path.exists() else 'stale spec / unreadable / bad p'})")
    if bad:
        raise GridIncomplete(f"{len(bad)} of {len(c3.all_fits())} work files not done:\n  " + "\n  ".join(bad))
    W = {}
    for f in c3.all_fits():
        i, d, c, s, p, wd = f
        w = load_pkl(grid.path(c, i, d, s, p, wd))
        w.pop("state", None)
        W[f] = w
    return W


def choose(W: dict) -> dict:
    """{(inp, depth, pool, seed): (p, wd) or None}: highest best_score among ok fits, ties to
    the first in REG_GRID order; None if every fit of the group failed."""
    out = {}
    for i in INPUTS:
        for c in POOLS:
            for s in SEEDS:
                for d in DEPTHS:
                    best, best_sc = None, -np.inf
                    for (p, wd) in REG_GRID:
                        w = W[(i, d, c, s, p, wd)]
                        if w["status"] != "ok" or w["p"] is None:
                            continue
                        sc = float(w["best_score"])
                        if best is None or (np.isfinite(sc) and sc > best_sc):
                            best, best_sc = (p, wd), sc
                    out[(i, d, c, s)] = best
    return out


# --- Main computation ----------------------------------------------------------------------------
class Eval:
    def __init__(self, ctx: Ctx, grid: train3.Grid, W: dict, out_dir: Path):
        self.ctx, self.grid, self.W, self.out_dir = ctx, grid, W, out_dir
        self.sel = choose(W)
        self.failed = [k for k, v in self.sel.items() if v is None]
        self._fit_m1 = {}

    def a_layer(self, pool: str, seed: int) -> str:
        return self.grid.sel[pool]["a_probe"][seed]

    def input_label(self, inp: str, pool: str, seed: int) -> str:
        return self.a_layer(pool, seed) if inp == "A" else "C final inputs"

    def fit_m1(self, f: tuple) -> dict:
        """{test set: m1_safe dict} at the fit's best epoch, for the five test sets (cached)."""
        if f not in self._fit_m1:
            i, d, c, s, p, wd = f
            w = self.W[f]
            if w["status"] != "ok" or w["p"] is None:
                self._fit_m1[f] = {}
            else:
                sets = self.grid.sets[(c, s)]
                sc = score_of(w["p"], self.ctx.kept)
                self._fit_m1[f] = {ts: m1_safe(sc, self.ctx.kept, self.ctx.idx, sets[ts])
                                   for ts in TEST_SETS if len(sets[ts])}
        return self._fit_m1[f]

    # deep detectors, one (pool, seed): npz + rows
    def deep_seed(self, pool: str, s: int) -> dict:
        idx, kept, ctx = self.ctx.idx, self.ctx.kept, self.ctx
        sets = self.grid.sets[(pool, s)]
        evals = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
        cuts = {k: m1_cut(idx, pos) for k, pos in evals.items()}
        cal = sets["cal"]
        npz = {"rollout_ids": idx.rids, "cond": pool, "seed": s,
               "alphas": np.array(ALPHAS), "kept_len": idx.kept_len, "y": idx.y,
               "cell": idx.task, "base_task": idx.base_task, "block": idx.block,
               "cal_rids": idx.rids[cal], "train_rids": idx.rids[sets["train"]],
               "eval_seen_rids": idx.rids[sets["eval_seen"]]}
        stress_cut = {}
        for ss in STRESS_SETS:
            if len(sets[ss]):
                stress_cut[ss] = stage6_cp.stress_cuts(ctx, s, sets, sets[ss])
                npz[f"{ss}_cut"] = stress_cut[ss]
        m1_rows, cp_rows, st_rows = [], [], []
        selj = {"hp": c3.HP, "reg_grid": [list(r) for r in REG_GRID],
                "selection_rule": "per (input, depth, pool, seed): highest best-epoch eval_seen M1-per-task "
                                  "among ok fits, ties to the first in REG_GRID order",
                "betas": self.grid.sel[pool]["betas"], "a_probe": self.a_layer(pool, s),
                "precision": self.grid.precision}
        failed = []
        for i in INPUTS:
            for d in DEPTHS:
                det = det_key(i, d)
                ch = self.sel[(i, d, pool, s)]
                if ch is None:
                    failed.append(det)
                    continue
                p_, wd = ch
                f = (i, d, pool, s, p_, wd)
                w = self.W[f]
                p = w["p"]
                sc = score_of(p, kept)
                with warnings.catch_warnings():       # empty-slice warnings past t_end (as abc2)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mu, xi, R, t_end, n1, n2 = abc_cp.fit_band(
                        sc, kept, cal, np.random.default_rng(stage6_cp.CAL_SPLIT_SEED + s))
                hs = np.array([abc_cp.band_h(R, a) for a in ALPHAS])
                detail = (f"{self.input_label(i, pool, s)} p={p_:g} wd={wd:g} epoch={w['best_epoch']}"
                          + (f" lr={w['lr']:g}" if w["lr"] != c3.HP["lr"] else ""))
                head = {"cond": pool, "seed": s, "detector": det, "detail": detail}
                extra = det_info(det)
                fm = self.fit_m1(f)
                for ts in TEST_SETS:
                    if ts in fm:
                        m1_rows.append(ordered({**head, "test_set": ts, **m1_row(fm[ts]), **extra},
                                               M1_COLS + EXTRA))
                for ts, pos in evals.items():
                    for a, h in zip(ALPHAS, hs):
                        e = abc_cp.evaluate_cp(sc, kept, idx, pos, mu, xi, h, cuts[ts])
                        cp_rows.append(ordered({**head, "test_set": ts, "alpha": float(a), "variant": "cut",
                                                "on_alpha_grid": bool(a in abc_cp.ALPHAS), "h": h,
                                                "bal_acc": e["bal_acc"], "tpr": e["tpr"], "tnr": e["tnr"],
                                                "far": 1 - e["tnr"], "det_time": e["det_time"],
                                                "n": int(len(pos)), "n_fail": int(idx.y[pos].sum()),
                                                "n_cal": int(len(cal)), "n_d1": n1, "n_d2": n2,
                                                "t_end": t_end, **extra}, CP_COLS + EXTRA))
                npz.update({f"{det}_m1": sc, f"{det}_cp": sc, f"{det}_p": p.astype(np.float32),
                            f"{det}_mu": mu, f"{det}_xi": xi, f"{det}_h": hs, f"{det}_band01": mu + hs[-1] * xi,
                            f"{det}_R": R, f"{det}_t_end": t_end, f"{det}_n_d1": n1, f"{det}_n_d2": n2})
                for ss, cut in stress_cut.items():
                    pos = sets[ss]
                    fired = stage6_cp.alarms(sc, kept, pos, mu, xi, hs[-1], cut)
                    for blk in ["all"] + sorted(set(idx.block[pos].tolist())):
                        m = np.ones(len(pos), bool) if blk == "all" else idx.block[pos] == blk
                        st_rows.append(ordered({**head, "stress_set": ss, "block": blk,
                                                "alpha": stage6_cp.ALPHA_TABLE, "n": int(m.sum()),
                                                "n_fail": int(idx.y[pos][m].sum()),
                                                "share_alarm_before_cut": float(fired[m].mean()),
                                                "mean_cut": float(cut[m].mean()), **extra}, ST_COLS + EXTRA))
                selj[det] = {"input": i, "depth": d, "input_desc": self.input_label(i, pool, s),
                             "p": p_, "wd": wd, "best_epoch": w["best_epoch"], "epochs_run": w["epochs_run"],
                             "best_score": w["best_score"], "lr": w["lr"], "n_params": w["n_params"],
                             "n_train_rows": w["n_train_rows"], "d": w["d"],
                             "work_file": str(self.grid.path(pool, i, d, s, p_, wd))}
        npz["selections"] = json.dumps(selj, default=float)
        npz["failed"] = json.dumps(failed)
        out = self.out_dir / "scores" / f"{pool}_seed{s}_deep.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **npz)
        tmp.replace(out)
        return {"final_m1": m1_rows, "cp": cp_rows, "stress": st_rows}

    # reference detectors: M1 rows recomputed from the abc2 npz score arrays
    def ref_seed(self, pool: str, s: int, abc2_m1: dict) -> list:
        idx, kept = self.ctx.idx, self.ctx.kept
        sets = self.grid.sets[(pool, s)]
        rows = []
        for det, (i, d, model, key, _, suffix) in REFS.items():
            path = c3.ABC2_RESULTS / "scores" / f"{pool}_seed{s}{suffix}.npz"
            z = np.load(path, allow_pickle=False)
            for name, ref in (("rollout_ids", idx.rids), ("train_rids", idx.rids[sets["train"]]),
                              ("eval_seen_rids", idx.rids[sets["eval_seen"]])):
                if not np.array_equal(z[name].astype(str), np.asarray(ref).astype(str)):
                    raise RuntimeError(f"{path}: {name} differs from this run's index / split ({pool} seed {s})")
            sc = z[f"{key}_m1"]
            ar = abc2_m1.get((pool, s, key), {})
            detail = next(iter(ar.values()))["detail"] if ar else ""
            head = {"cond": pool, "seed": s, "detector": det, "detail": detail}
            for ts in TEST_SETS:
                pos = sets[ts]
                if not len(pos):
                    continue
                rows.append(ordered({**head, "test_set": ts, **m1_row(m1_safe(sc, kept, idx, pos)),
                                     **det_info(det)}, M1_COLS + EXTRA))
        return rows


def abc2_tables() -> dict:
    """abc2's final_m1 / cp_curves / stress rows of the reference keys, pools D0 / D2."""
    out = {"m1": {}, "cp": [], "stress": []}
    for det, (i, d, model, key, pre, _) in REFS.items():
        for r in read_csv(c3.ABC2_RESULTS / f"{pre}final_m1.csv"):
            if r["cond"] in POOLS and r["detector"] == key:
                out["m1"].setdefault((r["cond"], int(r["seed"]), key), {})[r["test_set"]] = r
        for tab, cols in (("cp", CP_COLS), ("stress", ST_COLS)):
            fname = "cp_curves.csv" if tab == "cp" else "stress.csv"
            rows = [r for r in read_csv(c3.ABC2_RESULTS / f"{pre}{fname}")
                    if r["cond"] in POOLS and r["detector"] == key]
            if not rows:
                raise RuntimeError(f"no {key} rows for {POOLS} in {c3.ABC2_RESULTS / (pre + fname)}")
            out[tab] += [ordered({**r, "detector": det, **det_info(det)}, cols + EXTRA) for r in rows]
    return out


def reference_check(ref_rows: list, abc2_m1: dict) -> dict:
    """Recomputed reference M1 rows (eval_seen / T0 / T1 / T2) vs abc2's tables."""
    comps, problems = [], []
    for r in ref_rows:
        if r["test_set"] == "train":
            continue
        key = REFS[r["detector"]][3]
        a = abc2_m1.get((r["cond"], r["seed"], key), {}).get(r["test_set"])
        if a is None:
            problems.append(f"{r['cond']} seed {r['seed']} {r['detector']} {r['test_set']}: no abc2 row")
            continue
        for col in ("m1_pooled", "m1_per_task", "floor", "n_cells", "n", "n_fail"):
            got, ref = float(r[col]), float(a[col])
            diff = abs(got - ref) if (np.isfinite(got) and np.isfinite(ref)) else (
                0.0 if (np.isnan(got) and np.isnan(ref)) else float("inf"))
            comps.append(diff)
            if diff > CHECK_TOL:
                problems.append(f"{r['cond']} seed {r['seed']} {r['detector']} {r['test_set']} {col}: "
                                f"got {got!r}, abc2 {ref!r}")
    n_expected = len(POOLS) * len(SEEDS) * len(REFS) * len(EVAL_SETS)
    n_rows = sum(r["test_set"] != "train" for r in ref_rows)
    if n_rows != n_expected:
        problems.append(f"{n_rows} reference rows checked, expected {n_expected}")
    return {"tolerance": CHECK_TOL, "n_rows": n_rows, "n_compared": len(comps),
            "max_abs_diff": max(comps) if comps else None,
            "columns": ["m1_pooled", "m1_per_task", "floor", "n_cells", "n", "n_fail"],
            "reference": {"linear": str(c3.ABC2_RESULTS / "final_m1.csv"),
                          "plain-mlp-abc2": str(c3.ABC2_RESULTS / "mlp_final_m1.csv")},
            "passed": bool(comps) and not problems, "problems": problems[:50]}


def final_table(m1: list, cp: list, n_params: dict) -> list:
    """stage10_mlp.final_table (pipeline.assemble's final_table.csv) on the CSV rows read
    back, + input, depth, model, n_params; alarm columns empty for the train set."""
    cp = [r for r in cp if r["on_alpha_grid"] == "False"]
    final = []
    cpd, m1d = {}, {}
    for r in cp:
        cpd.setdefault((r["cond"], r["test_set"], r["detector"]), []).append(r)
    for r in m1:
        m1d.setdefault((r["cond"], r["test_set"], r["detector"]), []).append(r)
    for (cond, ts, det), mr in m1d.items():
        cr = cpd.get((cond, ts, det), [])
        row = {"cond": cond, "test_set": ts, "detector": det, "n_seeds": len(mr)}
        for k in ("m1_pooled", "m1_per_task"):
            row[f"{k}_mean"], row[f"{k}_std"] = mean_std([float(r[k]) for r in mr])
        row["floor_mean"] = mean_std([float(r["floor"]) for r in mr])[0]
        row["n_cells"] = " ".join(r["n_cells"] for r in mr)
        for k in ("bal_acc", "tpr", "far", "det_time"):
            row[f"a01_{k}_mean"], row[f"a01_{k}_std"] = (mean_std([float(r[k]) for r in cr]) if cr else ("", ""))
        row["n_cal_mean"] = mean_std([float(r["n_cal"]) for r in cr])[0] if cr else ""
        row["n_mean"] = mean_std([float(r["n"]) for r in mr])[0]
        info = det_info(det)
        row.update(info)
        row["n_params"] = n_params.get((cond, det), "")
        final.append(row)
    return final


def bootstrap_check(out_dir: Path, ev: Eval, demotions) -> dict:
    """bootstrap2.load_side on T0 + check_identity against this run's tables, every detector."""
    sides = []
    for pool in POOLS:
        for det in DET_ORDER:
            if det in REFS:
                key, suffix = REFS[det][3], REFS[det][5]
                files = [c3.ABC2_RESULTS / "scores" / f"{pool}_seed{s}{suffix}.npz" for s in SEEDS]
            else:
                i, d = det.split("-d")
                key = det
                files = [out_dir / "scores" / f"{pool}_seed{s}_deep.npz" for s in SEEDS
                         if ev.sel[(i, int(d), pool, s)] is not None]
            if not files:
                continue
            side = bootstrap2.load_side(files, key, "T0", demotions)
            res = bootstrap2.check_identity(side, "T0", out_dir, out_dir / "final_table.csv", detector_name=det)
            res["files"] = [str(f) for f in files]
            res["key"] = key
            sides.append(res)
    diffs = [c["max_abs_diff"] for c in sides if c["max_abs_diff"] is not None]
    return {"test_set": "T0", "tolerance": bootstrap2.CHECK_TOL, "n_sides": len(sides),
            "n_compared": sum(c["n_compared"] for c in sides),
            "max_abs_diff": max(diffs) if diffs else None,
            "passed": bool(sides) and all(c["passed"] for c in sides), "sides": sides}


def write_status(dest: Path, status: str, reason: str, **extra) -> None:
    d = {"status": status, "reason": reason, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         **extra}
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "eval_status.json.tmp"
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(dest / "eval_status.json")


def run(args) -> int:
    t0 = time.time()
    out_dir = args.out_dir
    ctx = make_ctx(args.work_dir, out_dir)
    grid = train3.Grid(ctx, train3.read_precision(None), args.work_dir)
    W = load_grid(grid)
    log(f"[eval3] {len(W)} work files loaded from {args.work_dir} ({time.time() - t0:.0f} s); precision "
        f"{grid.precision}")
    for (i, d, c, s, p, wd), w in W.items():
        if w["d"] != D_IN[i]:
            raise RuntimeError(f"{i} input width {w['d']} != {D_IN[i]} ({c} seed {s} d{d})")
    ev = Eval(ctx, grid, W, out_dir)
    for g in ev.failed:
        log(f"[eval3] group FAILED (all four fits failed): input {g[0]} depth {g[1]} {g[2]} seed {g[3]}")

    # deep detectors + references
    abc2 = abc2_tables()
    m1_rows, cp_rows, st_rows, ref_m1 = [], [], [], []
    for c in POOLS:
        for s in SEEDS:
            r = ev.deep_seed(c, s)
            m1_rows += r["final_m1"]
            cp_rows += r["cp"]
            st_rows += r["stress"]
            ref_m1 += ev.ref_seed(c, s, abc2["m1"])
    log(f"[eval3] deep detectors scored, npz written ({time.time() - t0:.0f} s)")
    check = reference_check(ref_m1, abc2["m1"])
    log(f"[eval3] reference check {'PASSED' if check['passed'] else 'FAILED'}: {check['n_rows']} rows, "
        f"max |diff| {check['max_abs_diff']}")

    def sort_key(r):
        return (POOLS.index(r["cond"]), int(r["seed"]), DET_ORDER.index(r["detector"]))
    m1_all = sorted(m1_rows + ref_m1, key=lambda r: (*sort_key(r), TEST_SETS.index(r["test_set"])))
    cp_all = sorted(cp_rows + abc2["cp"], key=lambda r: (*sort_key(r), TEST_SETS.index(r["test_set"])))
    st_all = sorted(st_rows + abc2["stress"], key=lambda r: (*sort_key(r), STRESS_SETS.index(r["stress_set"])))
    write_csv(out_dir / "final_m1.csv", m1_all)
    write_csv(out_dir / "cp_curves.csv", cp_all)
    write_csv(out_dir / "stress.csv", st_all)

    # parameter counts (mean over seeds of the chosen fits; references by formula)
    n_params = {}
    for c in POOLS:
        for det in DET_ORDER:
            if det in REFS:
                n_params[(c, det)] = ref_n_params(det)
            else:
                i, d = det.split("-d")
                v = [W[(i, int(d), c, s, *ev.sel[(i, int(d), c, s)])]["n_params"] for s in SEEDS
                     if ev.sel[(i, int(d), c, s)] is not None]
                if v:
                    n_params[(c, det)] = float(np.mean(v)) if len(set(v)) > 1 else int(v[0])
    ft = final_table(read_csv(out_dir / "final_m1.csv"), read_csv(out_dir / "cp_curves.csv"), n_params)
    ft.sort(key=lambda r: (POOLS.index(r["cond"]), TEST_SETS.index(r["test_set"]), DET_ORDER.index(r["detector"])))
    write_csv(out_dir / "final_table.csv", ft)

    # selections, grid, training log
    sel_rows, grid_rows, log_rows, es_diffs = [], [], [], []
    for c in POOLS:
        for s in SEEDS:
            for i in INPUTS:
                for d in DEPTHS:
                    det = det_key(i, d)
                    ch = ev.sel[(i, d, c, s)]
                    head = {"cond": c, "seed": s, "input": i, "depth": d, "detector": det}
                    a_layer = ev.a_layer(c, s) if i == "A" else ""
                    if ch is None:
                        nps = {W[(i, d, c, s, p, wd)]["n_params"] for (p, wd) in REG_GRID}
                        sel_rows.append({**head, "a_layer": a_layer, "p": "", "wd": "", "best_epoch": "",
                                         "epochs_run": "", "eval_seen_best": "", "lr": "", "status": "failed",
                                         "n_params": nps.pop() if len(nps) == 1 else ""})
                    else:
                        w = W[(i, d, c, s, *ch)]
                        sel_rows.append({**head, "a_layer": a_layer, "p": ch[0], "wd": ch[1],
                                         "best_epoch": w["best_epoch"], "epochs_run": w["epochs_run"],
                                         "eval_seen_best": w["best_score"], "lr": w["lr"], "status": "ok",
                                         "n_params": w["n_params"]})
                    for (p, wd) in REG_GRID:
                        f = (i, d, c, s, p, wd)
                        w = W[f]
                        chosen = ch == (p, wd)
                        fm = ev.fit_m1(f)
                        mets = {f"{ts}_m1_per_task": (float(fm[ts]["per_task"]) if ts in fm else "")
                                for ts in TEST_SETS}
                        mets["T0_m1_pooled"] = float(fm["T0"]["pooled"]) if "T0" in fm else ""
                        if "eval_seen" in fm and np.isfinite(w["best_score"]):
                            es_diffs.append(abs(float(fm["eval_seen"]["per_task"]) - float(w["best_score"])))
                        grid_rows.append({**head, "p": p, "wd": wd, "chosen": chosen, "status": w["status"],
                                          "diverged": bool(w["diverged"]), "lr": w["lr"],
                                          "n_attempts": len(w["attempts"]), "best_epoch": w["best_epoch"],
                                          "epochs_run": w["epochs_run"], "stopped_early": bool(w["stopped_early"]),
                                          "seconds": w["seconds"], "n_params": w["n_params"],
                                          "n_train_rows": w["n_train_rows"], **mets})
                        for h in w["history"]:
                            log_rows.append(ordered({**head, "p": p, "wd": wd, "chosen": chosen, **h},
                                                    ["cond", "seed", "input", "depth", "detector", "p", "wd",
                                                     "chosen", "epoch", "train_loss", "lr_end",
                                                     "eval_seen_m1_per_task", "eval_seen_m1_pooled",
                                                     "train_m1_per_task", "train_m1_pooled"]))
    write_csv(out_dir / "selections.csv", sel_rows)
    write_csv(out_dir / "grid.csv", grid_rows)
    write_csv(out_dir / "train_log.csv", log_rows)
    es_max = max(es_diffs) if es_diffs else None
    log(f"[eval3] grid / selections / train log written; stored-p eval_seen M1-per-task vs work-file "
        f"best_score: max |diff| {es_max}")

    # bootstrap2 load + identity check on T0
    bchk = bootstrap_check(out_dir, ev, ctx.demotions)
    log(f"[eval3] bootstrap2 identity check (T0, {bchk['n_sides']} sides, {bchk['n_compared']} values) "
        f"{'PASSED' if bchk['passed'] else 'FAILED'}: max |diff| {bchk['max_abs_diff']}")
    passed = check["passed"] and bchk["passed"]
    (out_dir / "eval_check.json").write_text(json.dumps(
        {"passed": passed, "max_abs_diff": check["max_abs_diff"], "tolerance": CHECK_TOL,
         "reference_check": check, "bootstrap_identity": bchk,
         "stored_p_vs_best_score_eval_seen_max_abs_diff": es_max}, indent=1, default=float))

    groups_failed = [{"cond": g[2], "seed": g[3], "input": g[0], "depth": g[1], "detector": det_key(g[0], g[1])}
                     for g in ev.failed]
    n_fail_fits = sum(w["status"] == "failed" for w in W.values())
    extra = {"groups_failed": groups_failed, "n_groups": len(ev.sel), "n_fits": len(W),
             "n_fits_ok": len(W) - n_fail_fits, "n_fits_failed": n_fail_fits,
             "n_fits_diverged_flagged": sum(bool(w["diverged"]) for w in W.values()),
             "n_fits_retried": sum(len(w["attempts"]) > 1 for w in W.values()),
             "eval_check_passed": check["passed"], "eval_check_max_abs_diff": check["max_abs_diff"],
             "bootstrap_identity_passed": bchk["passed"], "bootstrap_identity_max_abs_diff": bchk["max_abs_diff"],
             "precision": grid.precision, "work_dir": str(args.work_dir), "out_dir": str(out_dir),
             "detectors": DET_ORDER, "seconds": round(time.time() - t0, 1)}
    if not passed:
        why = []
        if not check["passed"]:
            why.append("reference rows differ from abc2's tables: " + "; ".join(check["problems"][:5]))
        if not bchk["passed"]:
            why.append("bootstrap2 identity check failed: " + ", ".join(
                f"{x['cond']} {x['detector']}" for x in bchk["sides"] if not x["passed"]))
        write_status(out_dir, "failed", " | ".join(why), **extra)
        log("[eval3] FAILED: " + " | ".join(why))
        return 1
    write_status(out_dir, "done", "" if not groups_failed else f"{len(groups_failed)} group(s) failed", **extra)
    summary(ft)
    log(f"[eval3] done ({time.time() - t0:.0f} s) -> {out_dir}")
    return 0


def summary(ft: list) -> None:
    def f(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "  -   "
    by = {(r["cond"], r["test_set"], r["detector"]): r for r in ft}
    print("\nT0 summary (mean ± std over seeds; train / eval_seen M1-per-task means for reference)")
    print(f"{'pool':4s} {'detector':11s} {'n_params':>9s}  {'T0 M1/task':>17s}  {'BA@0.1':>15s}  "
          f"{'TPR@0.1':>7s}  {'FAR@0.1':>7s}  {'train':>6s}  {'e_seen':>6s}")
    for c in POOLS:
        for det in DET_ORDER:
            r = by.get((c, "T0", det))
            if r is None:
                print(f"{c:4s} {det:11s}  (no rows: failed)")
                continue
            tr = by.get((c, "train", det), {})
            es = by.get((c, "eval_seen", det), {})
            npar = r["n_params"]
            print(f"{c:4s} {det:11s} {str(int(float(npar))) if npar != '' else '':>9s}  "
                  f"{f(r['m1_per_task_mean'])} ± {f(r['m1_per_task_std'])}  "
                  f"{f(r['a01_bal_acc_mean'])} ± {f(r['a01_bal_acc_std'])}  "
                  f"{f(r['a01_tpr_mean'], 3):>7s}  {f(r['a01_far_mean'], 3):>7s}  "
                  f"{f(tr.get('m1_per_task_mean'), 3):>6s}  {f(es.get('m1_per_task_mean'), 3):>6s}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work-dir", type=Path, default=c3.WORK_DIR)
    ap.add_argument("--out-dir", type=Path, default=c3.RESULTS)
    args = ap.parse_args(argv)
    args.work_dir = args.work_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    os.chdir(c3.REPO)
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 3)))
    except Exception:
        pass
    try:
        return run(args)
    except GridIncomplete as e:
        msg = f"grid incomplete: {e}"
        print(f"[eval3] ERROR: {msg}", file=sys.stderr)
        lines = msg.splitlines()
        write_status(args.out_dir, "failed", " ".join(x.strip() for x in lines[:2]) + (" ..." if len(lines) > 2 else ""))
        return 2
    except Exception as e:
        traceback.print_exc()
        write_status(args.out_dir, "failed", f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
