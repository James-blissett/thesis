"""
sanity3.py

The two checks deep_mlp_handoff.md asks for before the full grid.

--plain   §3 sanity check: depth 2, C input, D0, seed 0, with a plain (non-residual, no
          LayerNorm) MLP and abc2's settings (stage10_mlp: 256-256 ReLU, AdamW lr 1e-3
          constant, no clipping, fp32, abc2's chosen weight decay for C-MLP on D0) trained by
          train3.train_fit(plain=True). Its T0 M1-per-task must equal abc2's C-MLP D0 seed-0
          value (results/abc2/mlp_final_m1.csv) within +- 0.01.
          -> results/abc3/sanity_check.json
--bf16    §2 precision check: one fit of the §2 recipe (A input, depth 12, D2, seed 0,
          p 0, wd 1e-2; the widest input and the deepest model) in fp32 and in bf16
          autocast. bf16 is used for the grid if its best eval_seen M1-per-task is within
          0.005 of fp32's. -> results/abc3/precision.json (read by train3.py)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import common3 as c3
import torch

import setup_c as abc_c
import stage10_mlp
import stage4_a
import train3
from data2 import train_rows_and_weights
from pipe2_common import log, m1_safe, read_csv

TOL_PLAIN = 0.01
TOL_BF16 = 0.005


def plain_check(grid: train3.Grid) -> dict:
    ctx = grid.ctx
    ref = [r for r in read_csv(c3.ABC2_RESULTS / "mlp_final_m1.csv")
           if r["cond"] == "D0" and r["seed"] == "0" and r["detector"] == "C-MLP" and r["test_set"] == "T0"]
    sel = [r for r in read_csv(c3.ABC2_RESULTS / "mlp_selections.csv")
           if r["cond"] == "D0" and r["seed"] == "0" and r["selection"] == "wd_C-MLP"]
    ref_m1, wd = float(ref[0]["m1_per_task"]), float(sel[0]["value"])
    sets = grid.sets[("D0", 0)]
    X = grid.c_features("D0", 0)
    train = train_rows_and_weights(ctx.idx, ctx.kept, sets["train"])
    inp = stage10_mlp.Inputs(X, train[0], train[1])
    t0 = time.time()
    res = train3.train_fit(inp, train, 0, ctx, sets["eval_seen"], sets["train"], depth=2, p=0.0, wd=wd,
                           lr=stage10_mlp.HP["lr"], precision="fp32", plain=True, track_train=False)
    sc = abc_c.running_sum(res["p"], ctx.kept)
    m = m1_safe(sc, ctx.kept, ctx.idx, sets["T0"])
    got = float(m["per_task"])
    out = {"check": "plain 256-256 MLP, C input, D0, seed 0, abc2 settings, via train3.train_fit(plain=True)",
           "wd": wd, "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
           "eval_seen_m1_per_task": res["best_score"], "T0_m1_per_task": got, "abc2_T0_m1_per_task": ref_m1,
           "abc2_best_epoch": int([r for r in read_csv(c3.ABC2_RESULTS / "mlp_selections.csv")
                                   if r["cond"] == "D0" and r["seed"] == "0"
                                   and r["selection"] == "C-MLP_best_epoch"][0]["value"]),
           "diff": got - ref_m1, "tolerance": TOL_PLAIN, "passed": bool(abs(got - ref_m1) <= TOL_PLAIN),
           "seconds": time.time() - t0}
    (c3.RESULTS / "sanity_check.json").write_text(json.dumps(out, indent=1))
    log(f"[sanity plain] T0 M1-per-task {got:.6f} vs abc2 {ref_m1:.6f} (diff {got - ref_m1:+.6f}; "
        f"best epoch {res['best_epoch']} vs abc2 {out['abc2_best_epoch']}) -> "
        f"{'PASSED' if out['passed'] else 'FAILED'}")
    return out


def bf16_check(grid: train3.Grid) -> dict:
    ctx = grid.ctx
    pool, seed, depth, p, wd = "D2", 0, 12, 0.0, 1e-2
    sets = grid.sets[(pool, seed)]
    probe = grid.sel[pool]["a_probe"][seed]
    X = stage4_a.load_probe_input(probe, ctx.cache_dir).to(train3.DEV)
    train = train_rows_and_weights(ctx.idx, ctx.kept, sets["train"])
    inp = stage10_mlp.Inputs(X, train[0], train[1])
    res = {}
    for prec in ("fp32", "bf16"):
        t0 = time.time()
        r = train3.train_fit(inp, train, seed, ctx, sets["eval_seen"], sets["train"], depth=depth, p=p, wd=wd,
                             lr=c3.HP["lr"], precision=prec)
        res[prec] = {"best_score": r["best_score"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                     "seconds": time.time() - t0, "diverged": r["diverged"],
                     "T0_m1_per_task": float(m1_safe(abc_c.running_sum(r["p"], ctx.kept), ctx.kept, ctx.idx,
                                                     sets["T0"])["per_task"])}
        log(f"[sanity bf16] {prec}: eval_seen M1/task {r['best_score']:.4f} (epoch {r['best_epoch']}/"
            f"{r['epochs_run']}), {res[prec]['seconds']:.0f} s")
    diff = res["bf16"]["best_score"] - res["fp32"]["best_score"]
    ok = bool(abs(diff) <= TOL_BF16)
    out = {"check": f"A input ({probe}), depth {depth}, {pool}, seed {seed}, p {p}, wd {wd}",
           **res, "diff_eval_seen": diff, "tolerance": TOL_BF16, "passed": ok,
           "precision": "bf16" if ok else "fp32"}
    (c3.RESULTS / "precision.json").write_text(json.dumps(out, indent=1))
    log(f"[sanity bf16] diff {diff:+.4f} -> precision {out['precision']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 3)))
    os.chdir(c3.REPO)
    ctx = train3.make_ctx(c3.WORK_DIR)
    grid = train3.Grid(ctx, "fp32", c3.WORK_DIR)
    rc = 0
    if args.plain:
        rc |= 0 if plain_check(grid)["passed"] else 2
    if args.bf16:
        bf16_check(grid)
    return rc


if __name__ == "__main__":
    sys.exit(main())
