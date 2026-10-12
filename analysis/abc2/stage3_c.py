"""
stage3_c.py

Stage 3 of the port (analysis/abc/setup_c.py, generalised to an index, a condition and
the Step 7 test sets): setup C, constraints -> episode outcome.

  * betas: per family, from this condition's Stage 2 table (stage2_constraints.summarise).
  * 33 probes (32 per-layer [emb_temp[l], xl_adj[min(l,30)], xl_final[min(l,30)]] + the
    action probe) and the final detector (132 inputs incl. |emb_temp| and |act_mag|),
    each StandardScaler -> LogisticRegression(C=c, max_iter=2000) via setup_c.fit_prob,
    trained with data.train_rows_and_weights on the condition's train set; features via
    setup_c.seed_features (scale/orient on the condition's train set). No
    leave-one-family-out ablations and no emb_temp[28] 1-input probe (not asked for in
    Step 9; they never fed a selection).
  * score = running sum of P(failure) over kept rows (setup_c.running_sum).
  * c: one for the 33 probes (best mean over seeds of the mean over probes of eval_seen
    M1-per-task) and one for the final detector (best mean over seeds of its eval_seen
    M1-per-task), over C_GRID; ties to the first (largest) c, as max() did.

Work (resumable per (cond, seed)): <work>/<cond>/c_grid_seed{s}.pkl: metrics of every
model at every c on every evaluation set, final-detector coefficients, orientation signs,
and the final detector's per-row P(failure) at every c (float32).
Tables (<out>/tables/<cond>/): c_grid.csv, c_grid_choice.csv, c_probes.csv (at the
probes' c), c_final.csv (at the final c), c_final_weights.csv.
"""

from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed

import setup_c as abc_c
from data2 import train_rows_and_weights
from pipe2_common import (EVAL_SETS, Ctx, log, load_pkl, m1_row, m1_safe, mean_std,
                          record_timing, save_pkl, write_csv)

C_GRID = abc_c.C_GRID
FAMILIES = abc_c.FAMILIES
N_RAW = 32
PROBES = {**{f"layer{L:02d}": [f"emb_temp[{L}]", f"xl_adj[{min(L, 30)}]", f"xl_final[{min(L, 30)}]"]
             for L in range(N_RAW)}, "action": list(FAMILIES["action"])}


def smoothed_families(ctx: Ctx, betas: dict) -> dict:
    """setup_c.smoothed_families from the cached primary series (same order)."""
    S, names = ctx.series("primary")
    out = {}
    for fam, bases in FAMILIES.items():
        ks = [k for k, (b, _) in enumerate(names) if b in bases]
        beta = 0.0 if betas[fam] == "none" else float(betas[fam])
        out[fam] = (abc_c.smooth(S[ks], [names[k] for k in ks], beta), [names[k] for k in ks])
    return out


def run_job(seed, c, feats, all_inputs, idx, kept, eval_pos, train):
    """All C models for one (seed, c): metrics on every evaluation set, final coef and p."""
    out = {"seed": seed, "c": c, "metrics": {}, "unconverged": []}
    models = dict(PROBES)
    models["final"] = all_inputs
    for name, inputs in models.items():
        model, p, ok = abc_c.fit_prob(inputs, feats, train, c)
        if not ok:
            out["unconverged"].append(name)
        s = abc_c.running_sum(p, kept)
        out["metrics"][name] = {k: m1_row(m1_safe(s, kept, idx, pos)) for k, pos in eval_pos.items()}
        if name == "final":
            out["coef"] = dict(zip(inputs, model[-1].coef_[0].tolist()))
            out["p_final"] = p
    return out


def run_seeds(ctx: Ctx, cond: str, sel2: dict) -> None:
    idx, kept = ctx.idx, ctx.kept
    todo = []
    for s in ctx.seeds:
        p = ctx.wpath(cond, f"c_grid_seed{s}.pkl")
        if p.exists() and load_pkl(p)["betas"] == sel2["betas"]:
            continue
        todo.append(s)
    if not todo:
        return
    t0 = time.time()
    fams = smoothed_families(ctx, sel2["betas"])
    feats, signs, trains, evals = {}, {}, {}, {}
    for s in todo:
        sets = ctx.sets(cond, s)
        feats[s], signs[s] = abc_c.seed_features(fams, idx, kept, sets["train"])
        trains[s] = train_rows_and_weights(idx, kept, sets["train"])
        evals[s] = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
    del fams
    all_inputs = list(feats[todo[0]])
    log(f"[s3 {cond}] features ready ({time.time() - t0:.0f} s); fitting "
        f"{len(todo) * len(C_GRID)} (seed, c) jobs, n_jobs={ctx.n_jobs}")
    jobs = Parallel(n_jobs=ctx.n_jobs, verbose=0)(
        delayed(run_job)(s, c, feats[s], all_inputs, idx, kept, evals[s], trains[s])
        for s in todo for c in C_GRID)
    for s in todo:
        js = {j["c"]: j for j in jobs if j["seed"] == s}
        unconv = sorted({(c, n) for c, j in js.items() for n in j["unconverged"]})
        if unconv:
            log(f"[s3 {cond} seed {s}] not converged (c, model): {unconv}")
        save_pkl(ctx.wpath(cond, f"c_grid_seed{s}.pkl"), {
            "cond": cond, "seed": s, "betas": sel2["betas"], "inputs": all_inputs,
            "signs": signs[s], "unconverged": unconv,
            "metrics": {c: js[c]["metrics"] for c in C_GRID},
            "coef": {c: js[c]["coef"] for c in C_GRID},
            "p_final": {c: js[c]["p_final"].astype(np.float32) for c in C_GRID}})
    record_timing(ctx, "stage3_grid", cond, time.time() - t0, f"seeds {todo}")
    log(f"[s3 {cond}] grid done for seeds {todo} ({time.time() - t0:.0f} s)")


def summarise(ctx: Ctx, cond: str, sel2: dict) -> dict:
    seeds = list(ctx.seeds)
    W = {s: load_pkl(ctx.wpath(cond, f"c_grid_seed{s}.pkl")) for s in seeds}
    probes = list(PROBES)

    def seen(s, c, name):
        return W[s]["metrics"][c][name]["eval_seen"]["m1_per_task"]

    probe_score = {c: mean_std([np.mean([seen(s, c, p) for p in probes]) for s in seeds])
                   for c in C_GRID}
    final_score = {c: mean_std([seen(s, c, "final") for s in seeds]) for c in C_GRID}
    c_probe = max(C_GRID, key=lambda c: probe_score[c][0])
    c_final = max(C_GRID, key=lambda c: final_score[c][0])
    unstable = {tuple(u) for u in sel2["unstable"]}

    tdir = ctx.tables_dir / cond
    grid_rows = []
    for c in C_GRID:
        for s in seeds:
            for name in probes + ["final"]:
                for ts, m in W[s]["metrics"][c][name].items():
                    grid_rows.append({"cond": cond, "seed": s, "model": name, "c": c,
                                      "test_set": ts, **m})
    write_csv(tdir / "c_grid.csv", grid_rows)

    inputs = W[seeds[0]]["inputs"]
    n_consistent = {}
    for c in C_GRID:
        cs = np.array([[W[s]["coef"][c][n] for s in seeds] for n in inputs])
        n_consistent[c] = int(np.sum(np.all(cs > 0, 1) | np.all(cs < 0, 1)))
    write_csv(tdir / "c_grid_choice.csv", [
        {"cond": cond, "c": c, "probes_seen_task_mean": probe_score[c][0],
         "probes_seen_task_std": probe_score[c][1], "final_seen_task_mean": final_score[c][0],
         "final_seen_task_std": final_score[c][1],
         "final_weights_sign_consistent": n_consistent[c], "final_n_weights": len(inputs),
         "chosen_probes": c == c_probe, "chosen_final": c == c_final} for c in C_GRID])

    def base_of(n):
        return abc_c.base_of(n)

    probe_rows = []
    for p, pin in PROBES.items():
        unst = [n for n in pin if base_of(n) in unstable]
        for s in seeds:
            for ts, m in W[s]["metrics"][c_probe][p].items():
                probe_rows.append({"cond": cond, "seed": s, "probe": p, "test_set": ts,
                                   "inputs": " ".join(pin), "c": c_probe,
                                   "unstable_inputs": " ".join(unst), **m})
    write_csv(tdir / "c_probes.csv", probe_rows)

    final_rows = []
    for s in seeds:
        for ts, m in W[s]["metrics"][c_final]["final"].items():
            final_rows.append({"cond": cond, "seed": s, "model": "final", "test_set": ts,
                               "c": c_final, "n_inputs": len(inputs),
                               "n_unstable_inputs": sum(base_of(n) in unstable for n in inputs),
                               **{f"beta_{f}": sel2["betas"][f] for f in FAMILIES}, **m})
    write_csv(tdir / "c_final.csv", final_rows)

    weight_rows = []
    for n in inputs:
        b, L = base_of(n)
        cs = np.array([W[s]["coef"][c_final][n] for s in seeds])
        weight_rows.append({
            "cond": cond, "input": n, "family": abc_c.fam_of(b), "is_abs": n.startswith("|"),
            "beta": sel2["betas"][abc_c.fam_of(b)], "c": c_final,
            "orientation": "".join("+" if W[s]["signs"][n] > 0 else "-" for s in seeds),
            "unstable": (b, L) in unstable, "coef_mean": float(cs.mean()),
            "coef_std": float(cs.std(ddof=1)) if len(cs) > 1 else float("nan"),
            "coef_sign_consistent": bool(np.all(cs > 0) or np.all(cs < 0)),
            **{f"coef_s{s}": float(v) for s, v in zip(seeds, cs)}})
    write_csv(tdir / "c_final_weights.csv", weight_rows)

    sel = {"c_probes": c_probe, "c_probes_score": probe_score[c_probe][0],
           "c_final": c_final, "c_final_score": final_score[c_final][0]}
    log(f"[s3 {cond}] c probes {c_probe:g} ({probe_score[c_probe][0]:.4f}), c final "
        f"{c_final:g} ({final_score[c_final][0]:.4f})")
    return sel


def final_scores(ctx: Ctx, cond: str, seed: int, c_final: float):
    """(p, s): the final detector's per-row P(failure) and running sum at the chosen c,
    both float32, exactly as setup_c saved scores/c_seed{s}.npz (p_full, s_full)."""
    p = load_pkl(ctx.wpath(cond, f"c_grid_seed{seed}.pkl"))["p_final"][c_final]
    return p, abc_c.running_sum(p, ctx.kept).astype(np.float32)


def run(ctx: Ctx, cond: str, sel2: dict) -> dict:
    run_seeds(ctx, cond, sel2)
    return summarise(ctx, cond, sel2)
