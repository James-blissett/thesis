"""
stage4_a.py

Stage 4 of the port (analysis/abc/setup_a.py, generalised to an index, a feature cache,
conditions and the Step 7 test sets): setup A, raw inputs -> episode outcome.

  * 34 probes: pooled hidden state at stored index 0..32 (32 = post-norm) and the 7
    action token ids as floats, each fitted with setup_a.fit_and_score (GPU L-BFGS,
    sklearn's objective; standardised on train rows; all of C_GRID warm-started from
    strong to weak penalty), on the condition's train rows / weights.
  * score = running sum of P(failure) over kept rows.
  * c_A: best mean over seeds of the mean over the 34 probes of eval_seen M1-per-task.
  * final detector, per seed: the probe with the best eval_seen M1-per-task at c_A.
    Its probabilities are rounded to fp16 and back, as setup_a stored them and cp_eval
    read them (scores/a_seed{s}.npz).
  * shuffled-label control on each seed's chosen probe: N_PERM permutations of the train
    labels, rng default_rng(PERM_SEED) consumed in seed order exactly as setup_a did
    (earlier seeds' permutations are burned first, so a single (cond, seed) is
    reproducible on its own).

Grid pass (stage4_a.grid): layer-outer, so each cache file is read once for every
(cond, seed) still missing it. Work: <work>/<cond>/a_grid_seed{s}/<probe>.pkl (metrics at
every c on every evaluation set). The chosen probe is then refitted (same call, same
warm-start order) in the final pass: <work>/<cond>/a_final_seed{s}.pkl. Per-probe
probabilities are not stored (disk); every probe's metrics are.

Tables (<out>/tables/<cond>/): a_grid.csv, a_grid_choice.csv, a_probes.csv (at c_A),
a_final.csv, a_shuffled.csv.

GPU: torch.cuda.set_per_process_memory_fraction(ABC2_GPU_FRAC), default 6/46.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np
import torch

import setup_a as abc_a
from data import open_action_tokens, open_layer
from data2 import POST_NORM_INDEX, train_rows_and_weights
from pipe2_common import (EVAL_SETS, Ctx, log, load_pkl, m1_row, m1_safe, mean_std,
                          record_timing, save_pkl, write_csv)
from setup_c import C_GRID, running_sum

PROBES = abc_a.PROBES
if os.environ.get("ABC2_PROBES"):          # testing only: restrict the probe list
    PROBES = [p for p in PROBES if p in os.environ["ABC2_PROBES"].split(",")]
N_PERM = abc_a.N_PERM
PERM_SEED = abc_a.PERM_SEED
_GPU_READY = False


def gpu_setup() -> None:
    global _GPU_READY
    if _GPU_READY:
        return
    if not torch.cuda.is_available():
        raise SystemExit("stage 4 needs a GPU (setup_a.fit_and_score runs on cuda)")
    frac = float(os.environ.get("ABC2_GPU_FRAC", 6 / 46))
    torch.cuda.set_per_process_memory_fraction(frac)
    log(f"[s4] GPU memory fraction {frac:.3f}")
    _GPU_READY = True


def check_cache(ctx: Ctx) -> None:
    """The cache's row order must be the index order."""
    p = ctx.cache_dir / "rollout_ids.json"
    if p.exists():
        rids = json.loads(p.read_text())
        rids = rids["rollout_ids"] if isinstance(rids, dict) else rids
        if list(rids) != ctx.idx.rids.tolist():
            raise SystemExit(f"{p}: rollout order differs from the index")
    n = open_layer(0, ctx.cache_dir).shape[0]
    if n != ctx.idx.n:
        raise SystemExit(f"cache has {n} rows, index {ctx.idx.n}")


def load_probe_input(name: str, cache_dir) -> torch.Tensor:
    """setup_a.load_probe_input with the cache directory as a parameter."""
    if name == "action_tokens":
        return torch.from_numpy(np.asarray(open_action_tokens(cache_dir)).astype(np.float32)).to(abc_a.DEV)
    return torch.from_numpy(np.array(open_layer(abc_a.probe_layer(name), cache_dir))).to(abc_a.DEV)


def evaluate(p, ctx: Ctx, eval_pos: dict) -> dict:
    s = running_sum(p, ctx.kept)
    return {k: m1_row(m1_safe(s, ctx.kept, ctx.idx, pos)) for k, pos in eval_pos.items()}


def _gdir(ctx, cond, s):
    d = ctx.wpath(cond, f"a_grid_seed{s}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def grid(ctx: Ctx, conds) -> None:
    """Fit every probe at every c for every (cond, seed) not yet done."""
    combos = [(c, s) for c in conds for s in ctx.seeds]
    missing = {p: [cs for cs in combos if not (_gdir(ctx, *cs) / f"{p}.pkl").exists()] for p in PROBES}
    if not any(missing.values()):
        return
    gpu_setup()
    check_cache(ctx)
    t0 = time.time()
    trains, evals = {}, {}
    for cond, s in combos:
        sets = ctx.sets(cond, s)
        trains[(cond, s)] = train_rows_and_weights(ctx.idx, ctx.kept, sets["train"])
        evals[(cond, s)] = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
    t_load = 0.0
    for k, probe in enumerate(PROBES):
        todo = missing[probe]
        if not todo:
            continue
        tl = time.time()
        X = load_probe_input(probe, ctx.cache_dir)
        t_load += time.time() - tl
        for cond, s in todo:
            P = abc_a.fit_and_score(X, trains[(cond, s)], list(C_GRID))
            res = {c: evaluate(P[c], ctx, evals[(cond, s)]) for c in C_GRID}
            save_pkl(_gdir(ctx, cond, s) / f"{probe}.pkl",
                     {"probe": probe, "cond": cond, "seed": s, "metrics": res})
        del X
        torch.cuda.empty_cache()
        log(f"[s4 grid {k + 1:2d}/{len(PROBES)}] {probe}: {len(todo)} (cond, seed) "
            f"({time.time() - t0:.0f} s, of which loading {t_load:.0f} s)")
    record_timing(ctx, "stage4_grid", "+".join(conds), time.time() - t0,
                  f"loading {t_load:.0f} s")


def perm_rng(ctx: Ctx, cond: str, seed: int) -> np.random.Generator:
    """default_rng(PERM_SEED) advanced past the permutations setup_a drew for earlier
    seeds (N_PERM per seed, each a permutation of that seed's train labels)."""
    rng = np.random.default_rng(PERM_SEED)
    for s in sorted(ctx.seeds):
        if s >= seed:
            break
        n = len(ctx.sets(cond, s)["train"])
        for _ in range(N_PERM):
            rng.permutation(n)
    return rng


def summarise_grid(ctx: Ctx, cond: str) -> tuple[dict, dict]:
    seeds = list(ctx.seeds)
    M = {(p, s): load_pkl(_gdir(ctx, cond, s) / f"{p}.pkl")["metrics"] for p in PROBES for s in seeds}

    def seen(p, s, c):
        return M[(p, s)][c]["eval_seen"]["m1_per_task"]

    score = {c: mean_std([np.mean([seen(p, s, c) for p in PROBES]) for s in seeds]) for c in C_GRID}
    c_a = max(C_GRID, key=lambda c: score[c][0])
    chosen = {s: max(PROBES, key=lambda p: seen(p, s, c_a)) for s in seeds}

    def head(p):
        L = abc_a.probe_layer(p)
        return {"probe": p, "layer": -1 if L is None else L, "post_norm": L == POST_NORM_INDEX,
                "inputs": "action_token_ids (7)" if L is None else "pooled hidden state (4096)"}

    tdir = ctx.tables_dir / cond
    rows = [{"cond": cond, "seed": s, **head(p), "c": c, "test_set": ts, **m}
            for c in C_GRID for p in PROBES for s in seeds for ts, m in M[(p, s)][c].items()]
    write_csv(tdir / "a_grid.csv", rows)
    write_csv(tdir / "a_probes.csv", [r for r in rows if r["c"] == c_a])
    write_csv(tdir / "a_grid_choice.csv", [
        {"cond": cond, "c": c, "probes_seen_task_mean": score[c][0],
         "probes_seen_task_std": score[c][1], "chosen": c == c_a} for c in C_GRID])
    sel = {"c_A": c_a, "c_A_score": score[c_a][0], "a_probe": chosen,
           "a_probe_score": {s: seen(chosen[s], s, c_a) for s in seeds}}
    log(f"[s4 {cond}] c_A {c_a:g} ({score[c_a][0]:.4f}); probes {chosen}")
    return sel, M


def final(ctx: Ctx, cond: str, sel4: dict, M: dict) -> None:
    """Refit the chosen probe per seed (final detector) and run the shuffled control."""
    c_a = sel4["c_A"]
    for s in ctx.seeds:
        path = ctx.wpath(cond, f"a_final_seed{s}.pkl")
        probe = sel4["a_probe"][s]
        if path.exists():
            w = load_pkl(path)
            if w["probe"] == probe and w["c"] == c_a:
                continue
        gpu_setup()
        check_cache(ctx)
        t0 = time.time()
        sets = ctx.sets(cond, s)
        evals = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
        train = train_rows_and_weights(ctx.idx, ctx.kept, sets["train"])
        X = load_probe_input(probe, ctx.cache_dir)
        p32 = abc_a.fit_and_score(X, train, list(C_GRID))[c_a]
        refit = evaluate(p32, ctx, evals)
        grid_seen = M[(probe, s)][c_a]["eval_seen"]["m1_per_task"]
        drift = abs(refit["eval_seen"]["m1_per_task"] - grid_seen)
        p16 = p32.astype(np.float16)

        rng = perm_rng(ctx, cond, s)
        tr = sets["train"]
        shuf = []
        for k in range(N_PERM):
            y_perm = ctx.idx.y.copy()
            y_perm[tr] = rng.permutation(ctx.idx.y[tr])
            idx_perm = dataclasses.replace(ctx.idx, y=y_perm)
            tr_rows = train_rows_and_weights(idx_perm, ctx.kept, tr)
            p = abc_a.fit_and_score(X, tr_rows, [c_a])[c_a]
            shuf.append(evaluate(p, ctx, evals))
        del X
        torch.cuda.empty_cache()
        save_pkl(path, {"cond": cond, "seed": s, "probe": probe, "c": c_a, "p16": p16,
                        "refit_metrics": refit, "refit_drift_eval_seen": drift,
                        "shuffled": shuf})
        record_timing(ctx, "stage4_final", cond, time.time() - t0, f"seed {s} {probe}")
        log(f"[s4 {cond} seed {s}] final {probe} refitted (eval_seen drift vs grid {drift:.2e}); "
            f"shuffled control done ({time.time() - t0:.0f} s)")


def final_scores(ctx: Ctx, cond: str, seed: int):
    """(p, s) of A's final detector, float32: p = fp16-rounded probability (as setup_a
    stored it), s = its running sum (as cp_eval used it)."""
    w = load_pkl(ctx.wpath(cond, f"a_final_seed{seed}.pkl"))
    p = w["p16"].astype(np.float32)
    return p, running_sum(p, ctx.kept).astype(np.float32)


def tables_final(ctx: Ctx, cond: str, sel4: dict, M: dict) -> None:
    tdir = ctx.tables_dir / cond
    rows, shuf_rows = [], []
    for s in ctx.seeds:
        w = load_pkl(ctx.wpath(cond, f"a_final_seed{s}.pkl"))
        for ts, m in M[(w["probe"], s)][w["c"]].items():
            rows.append({"cond": cond, "seed": s, "probe": w["probe"], "c": w["c"],
                         "test_set": ts, "refit_drift_eval_seen": w["refit_drift_eval_seen"], **m})
        for k, e in enumerate(w["shuffled"]):
            for ts, m in e.items():
                real = M[(w["probe"], s)][w["c"]][ts]
                shuf_rows.append({"cond": cond, "seed": s, "probe": w["probe"], "perm": k,
                                  "c": w["c"], "test_set": ts,
                                  "shuffled_m1_pooled": m["m1_pooled"],
                                  "shuffled_m1_per_task": m["m1_per_task"],
                                  "real_m1_pooled": real["m1_pooled"],
                                  "real_m1_per_task": real["m1_per_task"]})
    write_csv(tdir / "a_final.csv", rows)
    write_csv(tdir / "a_shuffled.csv", shuf_rows)


def run(ctx: Ctx, cond: str) -> dict:
    """Grid must already be done for `cond` (grid() is called across conditions)."""
    grid(ctx, [cond])
    sel4, M = summarise_grid(ctx, cond)
    final(ctx, cond, sel4, M)
    tables_final(ctx, cond, sel4, M)
    return sel4

