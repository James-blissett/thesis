"""
detectors.py

Per-row scores of every row of the core table (results_collection_handoff.md §4.1, §4.2)
for one training set (THESIS_POOL; original by default), per seed.

Each detector gives, per seed, (N, 520) arrays:
    p    the per-step value (P(failure) for trained models and probe means; the prepared
         z-score for untrained constraints and baselines; t for the timestep score)
    m1   the score AUROC takes the maximum of under the cut (running sum of p for every
         trained model and mean, as the earlier runs; p itself for untrained series)
    cp   the score the conformal band is fitted to (= m1 for running sums; the running
         maximum over kept rows for untrained series, as cp_eval / stage6_cp)
plus `detail` (choice made, per seed) and `source` (file read, or "new").

Rows (key: plain label):
  A_best        A, single best probe (of 34)      reused: results/abc2/scores/D0_seed<s>.npz key A
  A_mean        A, unweighted mean of 34 probes    new: refits from probes_a.py
  A_combiner    A, trained combiner (out-of-fold)  new
  A_mlp         A, 2-layer MLP                     reused: ..._mlp.npz key A-MLP
  C_best        C, single best probe (of 33)       new selection: eval-seen per-task, per seed
  C_mean        C, unweighted mean of 33 probes    new
  C_combiner    C, trained combiner (out-of-fold)  new
  C_mlp         C, 2-layer MLP on the best probe's inputs   new (mlp_c.py)
  C_lr132       C (exploratory), logistic regression on 132 inputs   reused key C
  C_mlp132      C (exploratory), 2-layer MLP on 132 inputs           reused key C-MLP
  B_best_h<H>   B, single best probe (of 124) against outcome        new
  B_mean_h<H>   B, unweighted mean of 124 probes (primary)           new (label-free run: original rollouts only)
  B_meanemb_h<H> B, unweighted mean of the embedding-temporal probes  new (as above)
  B_combiner_h<H> B, trained combiner over 124 (out-of-fold)         H = 10 reused (..._b10.npz key B); H = 0, 25 refitted here
  B_mlp_h10     B, 2-layer MLP on the best probe's layer, trained on its constraint target   new (mlp_b.py)
  timestep      timestep only
  emb_best      best single untrained constraint: emb_temp at the layer chosen on eval-seen per-task AUROC, running maximum
  emb_pooled    emb_temp at the layer chosen on eval-seen pooled AUROC (the earlier choice), reused key emb
  token_entropy token mean entropy (beta chosen on eval-seen per-task), running maximum
  token_maxprob token max-probability: per step, the top-bin probability averaged over the 7 action tokens
  failurespot   FailureSpot's signal s;  failurespot_neg  its negation -s

Work file: <WORK_DIR>/detectors/<key>.pkl  {seed: {p, m1, cp, detail}, source, selection}
(resumable; delete one to rebuild it).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import C_GRID, HS, REPO, SEEDS, T, load_pkl, log, mean_std, m1_safe, save_pkl  # noqa: E402

WD = common.WORK_DIR / "detectors"
ABC2_SCORES = REPO / "results/abc2/scores"


# --- helpers --------------------------------------------------------------------------
def running_sum(p):
    return common.abc_c.running_sum(p, common.ctx().kept).astype(np.float32)


def running_max(x):
    """Running maximum over kept rows; NaN before the first valid kept row (cp_eval)."""
    kept = common.ctx().kept
    xk = np.where(kept, x, -np.inf)
    run = np.maximum.accumulate(np.nan_to_num(xk, nan=-np.inf), axis=1)
    # nan_to_num turns -inf into -1.8e308 (as cp_eval); those rows are before the first
    # kept row and never compared: store NaN, as stage6_cp does.
    return np.where(np.isfinite(run) & (run > -1e300), run, np.nan).astype(np.float32)


def seen_per_task(score, s) -> float:
    ctx = common.ctx()
    return float(m1_safe(score, ctx.kept, ctx.idx, common.sets(s)["eval_seen"])["per_task"])


def stacked_seen_per_task(scores, s) -> np.ndarray:
    """eval-seen per-task AUROC of each (k, N, T) score."""
    ctx = common.ctx()
    return np.asarray(m1_safe(scores, ctx.kept, ctx.idx, common.sets(s)["eval_seen"])["per_task"])


def abc2_npz(s: int, suffix: str = ""):
    return np.load(ABC2_SCORES / f"{common.POOL}_seed{s}{suffix}.npz")


def train_mask(s: int) -> np.ndarray:
    m = np.zeros(common.ctx().idx.n, bool)
    m[common.sets(s)["train"]] = True
    return m


def done(key: str) -> bool:
    return (WD / f"{key}.pkl").exists()


def save(key: str, per_seed: dict, source: str, selection: dict | None = None) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    save_pkl(WD / f"{key}.pkl", {"key": key, "seeds": per_seed, "source": source,
                                 "selection": selection or {}})
    log(f"[det] {key}: saved ({source})")


def load(key: str) -> dict:
    return load_pkl(WD / f"{key}.pkl")


def entry(p, detail, kind="sum"):
    p = np.asarray(p, np.float32)
    if kind == "sum":
        sc = running_sum(p)
        return {"p": p, "m1": sc, "cp": sc, "detail": detail}
    return {"p": p, "m1": p, "cp": running_max(p), "detail": detail}


# --- probe outputs --------------------------------------------------------------------
def a_outputs(s: int):
    import setup_a as abc_a
    names = list(abc_a.PROBES)
    j = list(SEEDS).index(s)
    ref, oof = [], []
    for n in names:
        z = np.load(common.WORK_DIR / "a_probes" / f"{n}.npz")
        ref.append(z["p_refit"][j])
        oof.append(z["p_oof"][j])
    return names, np.stack(ref), np.stack(oof)


def c_outputs(s: int):
    z = np.load(common.WORK_DIR / "c_probes" / f"seed{s}.npz")
    return [str(n) for n in z["names"]], z["p_refit"], z["p_oof"], z


def b_outputs(H: int, s: int):
    """names 'L<ll>:<kind>:H<H>' and (k, N, 520) float32: train rollouts out-of-fold,
    the rest refit; rollouts no seed uses stay 0."""
    n = common.ctx().idx.n
    j = list(SEEDS).index(s)
    names, arrs = [], []
    for L in range(32):
        z = np.load(common.WORK_DIR / "b_probes" / f"L{L:02d}.npz")
        hs = [int(h) for h in z["hs"]]
        pos = z["pos"]
        for i, kd in enumerate(z["kinds"]):
            col = i * len(hs) + hs.index(H)
            f = np.zeros((n, T), np.float32)
            f[pos] = z["p"][j, col].astype(np.float32)
            names.append(f"L{L:02d}:{kd}:H{H}")
            arrs.append(f)
    return names, np.stack(arrs)


# --- combiner (B's final-detector procedure) -------------------------------------------
def combiner(names, feats_by_seed: dict, n_jobs: int = 6):
    """Logistic regression over probe outputs (out-of-fold on train rows), on outcome,
    with row weights, running-sum score; c = best mean over seeds of eval-seen
    per-task AUROC (stage5_b / setup_b final detector)."""
    from joblib import Parallel, delayed
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept

    def job(s, c, feats, train, es):
        model, p, ok = common.abc_c.fit_prob(names, feats, train, c)
        sc = common.abc_c.running_sum(p, kept)
        return s, c, p, float(m1_safe(sc, kept, idx, es)["per_task"]), ok, model[-1].coef_[0].tolist()

    trains = {s: common.train_rows_and_weights(idx, kept, common.sets(s)["train"]) for s in SEEDS}
    res = Parallel(n_jobs=n_jobs)(
        delayed(job)(s, c, feats_by_seed[s], trains[s], common.sets(s)["eval_seen"])
        for s in SEEDS for c in C_GRID)
    R = {(r[0], r[1]): r for r in res}
    score = {c: mean_std([R[(s, c)][3] for s in SEEDS]) for c in C_GRID}
    c_best = max(C_GRID, key=lambda c: score[c][0])
    grid = [{"c": c, "eval_seen_per_task_mean": score[c][0], "eval_seen_per_task_std": score[c][1],
             "chosen": c == c_best, "converged": all(R[(s, c)][4] for s in SEEDS)} for c in C_GRID]
    return {s: R[(s, c_best)][2] for s in SEEDS}, c_best, grid, {s: R[(s, c_best)][5] for s in SEEDS}


# --- builders -------------------------------------------------------------------------
def build_reused() -> None:
    specs = {"A_best": ("", "A", "A"), "C_lr132": ("", "C", "C"), "emb_pooled": ("", "emb", "emb"),
             "A_mlp": ("_mlp", "A-MLP", "A-MLP"), "C_mlp132": ("_mlp", "C-MLP", "C-MLP"),
             "B_combiner_h10": ("_b10", "B", "B")}
    for key, (suf, k, _) in specs.items():
        if done(key):
            continue
        per = {}
        for s in SEEDS:
            z = abc2_npz(s, suf)
            sel = json.loads(str(z["selections"]))
            if key == "emb_pooled":
                p = z["emb_m1"]
                per[s] = {"p": p, "m1": p, "cp": z["emb_cp"],
                          "detail": f"emb_temp[{sel['emb_layer']}] beta={sel['emb_beta']}"}
            else:
                p = z[f"{k}_p"].astype(np.float32)
                detail = {"A_best": lambda: f"{sel['a_probe']} c={sel['c_A']:g}",
                          "C_lr132": lambda: f"c={sel['c_final']:g}",
                          "A_mlp": lambda: f"{sel['A-MLP']['input'].split(' (')[0]} wd={sel['A-MLP']['wd']:g}",
                          "C_mlp132": lambda: f"wd={sel['C-MLP']['wd']:g}",
                          "B_combiner_h10": lambda: f"grid c={sel['c_grid']:g} final c={sel['c_final']:g}"}[key]()
                per[s] = {"p": p, "m1": z[f"{k}_m1"], "cp": z[f"{k}_cp"], "detail": detail}
        save(key, per, f"results/abc2/scores/{common.POOL}_seed<s>{suf}.npz key {k}")


def build_a() -> None:
    if done("A_mean") and done("A_combiner"):
        return
    feats, mean = {}, {}
    for s in SEEDS:
        names, ref, oof = a_outputs(s)
        ref = ref.astype(np.float32)
        mean[s] = entry(ref.mean(0), "34 probes")
        tm = train_mask(s)
        feats[s] = {n: np.where(tm[:, None], oof[k].astype(np.float32), ref[k]) for k, n in enumerate(names)}
    save("A_mean", mean, "new: mean of the 34 refitted probes (probes_a.py)")
    p, c, grid, coef = combiner(names, feats)
    save("A_combiner", {s: entry(p[s], f"c={c:g}") for s in SEEDS}, "new: logistic regression over 34 out-of-fold probe outputs",
         {"c": c, "grid": grid, "coef": coef, "inputs": names})


def build_c() -> None:
    if all(done(k) for k in ("C_best", "C_mean", "C_combiner", "C_mean_lofo")):
        return
    best, mean, feats, lofo = {}, {}, {}, {}
    sel = {}
    for s in SEEDS:
        names, ref, oof, z = c_outputs(s)
        sums = np.stack([common.abc_c.running_sum(ref[k], common.ctx().kept) for k in range(len(names))])
        seen = stacked_seen_per_task(sums, s)
        kb = int(np.argmax(seen))
        sel[s] = {"probe": names[kb], "eval_seen_per_task": float(seen[kb]),
                  "inputs": json.loads(str(z["inputs"]))[names[kb]]}
        best[s] = entry(ref[kb], f"{names[kb]} c={float(z['c']):g}")
        mean[s] = entry(ref.mean(0), "33 probes")
        tm = train_mask(s)
        feats[s] = {n: np.where(tm[:, None], oof[k].astype(np.float32), ref[k]) for k, n in enumerate(names)}
        ln = [str(n) for n in z["lofo_names"]]
        pl = z["p_lofo"]
        per_layer = ref[[k for k, n in enumerate(names) if n.startswith("layer")]]
        action = ref[names.index("action")]
        lofo[s] = {
            "no_action": per_layer.mean(0),
            "no_emb_temp": np.concatenate([pl[[k for k, n in enumerate(ln) if n.startswith("no_emb_temp")]],
                                           action[None]]).mean(0),
            "no_cross_layer": np.concatenate([pl[[k for k, n in enumerate(ln) if n.startswith("no_cross_layer")]],
                                              action[None]]).mean(0)}
    save("C_best", best, "new selection: best of C's 33 probes on eval-seen per-task AUROC (per seed)", sel)
    save("C_mean", mean, "new: mean of the 33 refitted probes (probes_c.py)")
    save("C_mean_lofo", {s: {k: entry(v, k) for k, v in lofo[s].items()} for s in SEEDS},
         "new: C's unweighted mean without one constraint family (probes_c.py)")
    p, c, grid, coef = combiner(names, feats)
    save("C_combiner", {s: entry(p[s], f"c={c:g}") for s in SEEDS}, "new: logistic regression over 33 out-of-fold probe outputs",
         {"c": c, "grid": grid, "coef": coef, "inputs": names})


def build_b() -> None:
    for H in HS:
        keys = [f"B_best_h{H}", f"B_mean_h{H}", f"B_meanemb_h{H}", f"B_combiner_h{H}", f"B_bylayer_h{H}"]
        if all(done(k) for k in keys) or (H == 10 and all(done(k) for k in keys if k != "B_combiner_h10")
                                          and done("B_combiner_h10_refit")):
            continue
        best, mean, memb, feats, sel, bylayer = {}, {}, {}, {}, {}, {}
        kept = common.ctx().kept
        for s in SEEDS:
            names, P = b_outputs(H, s)
            sums = np.cumsum(np.where(kept[None], P, 0.0), axis=2, dtype=np.float32)
            seen = stacked_seen_per_task(sums, s)
            del sums
            kb = int(np.nanargmax(seen))
            sel[s] = {"probe": names[kb], "eval_seen_per_task": float(seen[kb])}
            best[s] = entry(P[kb], names[kb])
            mean[s] = entry(P.mean(0), f"{len(names)} probes")
            emb = [k for k, n in enumerate(names) if n.split(":")[1] == "emb_temp"]
            memb[s] = entry(P[emb].mean(0), f"{len(emb)} embedding-temporal probes")
            feats[s] = {n: P[k] for k, n in enumerate(names)}
            bylayer[s] = {L: P[[k for k, n in enumerate(names) if n.startswith(f"L{L:02d}:")]].mean(0)
                          for L in range(32)}
        save(f"B_best_h{H}", best, "new selection: best of B's probes on eval-seen per-task AUROC against outcome", sel)
        save(f"B_mean_h{H}", mean, "new: mean of the refitted grid probes (probes_b.py)")
        save(f"B_meanemb_h{H}", memb, "new: mean of the refitted embedding-temporal grid probes")
        save(f"B_bylayer_h{H}", {s: {L: {"p": v} for L, v in bylayer[s].items()} for s in SEEDS},
             "new: per-layer mean of B's probes (profiles only)")
        p, c, grid, coef = combiner(names, feats)
        key = f"B_combiner_h{H}" if H != 10 else "B_combiner_h10_refit"
        save(key, {s: entry(p[s], f"grid c=1e-4 final c={c:g}") for s in SEEDS},
             "new: B's trained combiner refitted on the refitted grid outputs",
             {"c": c, "grid": grid, "coef": coef, "inputs": names})
        del feats


def build_b_targets() -> None:
    """Families ablation for B (handoff §4.6): per target kind, the unweighted mean of that
    kind's probes and a trained combiner over them alone, at the full combiner's c (as
    setup_b's per-target ablation). Original training set, H = 10: the trained combiner's
    c is the more-data run's (1e-3, results/abc2 B H = 10); H = 0 / 25 and the full
    training set: the c chosen by build_b's refit."""
    from joblib import Parallel, delayed
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    kinds = ("act_rep", "act_mag", "emb_temp", "xl_final")
    for H in HS:
        key = f"B_bytarget_h{H}"
        if done(key):
            continue
        ck = f"B_combiner_h{H}" if H != 10 else "B_combiner_h10_refit"
        c = load(ck)["selection"]["c"]
        per = {s: {} for s in SEEDS}
        trains = {s: common.train_rows_and_weights(idx, kept, common.sets(s)["train"]) for s in SEEDS}
        for s in SEEDS:
            names, P = b_outputs(H, s)
            jobs = []
            for kd in kinds:
                ks = [k for k, n in enumerate(names) if n.split(":")[1] == kd]
                per[s][f"mean_{kd}"] = entry(P[ks].mean(0), f"{len(ks)} {kd} probes")
                jobs.append((kd, [names[k] for k in ks], {names[k]: P[k] for k in ks}))
            res = Parallel(n_jobs=4)(delayed(common.abc_c.fit_prob)(nm, ft, trains[s], c) for _, nm, ft in jobs)
            for (kd, nm, _), (_, p, _) in zip(jobs, res):
                per[s][f"combiner_{kd}"] = entry(p, f"{len(nm)} {kd} probes, c={c:g}")
            del P
        save(key, per, "new: B per target kind (unweighted mean; trained combiner at the full combiner's c)",
             {"c": c})


def emb_best_layer() -> tuple[int, str, dict]:
    """emb_temp layer chosen on eval-seen per-task AUROC (mean over seeds) at this pool's
    emb_temp beta, from the pool's Stage 2 table (results/abc2/constraint_m1_summary.csv)."""
    beta = common.selections()["betas"]["emb_temp"]
    rows = [r for r in common.read_csv(REPO / "results/abc2/constraint_m1_summary.csv")
            if r["cond"] == common.POOL and r["variant"] == "primary" and r["constraint"] == "emb_temp"
            and r["beta"] == beta and r["test_set"] == "eval_seen"]
    best = max(rows, key=lambda r: float(r["m1_per_task_mean"]))
    return int(best["layer"]), beta, {"layer": int(best["layer"]), "beta": beta,
                                      "eval_seen_per_task_mean": float(best["m1_per_task_mean"]),
                                      "rule": "best eval-seen per-task AUROC, mean over seeds"}


def baseline_beta(name: str) -> tuple[str, float]:
    rows = [r for r in common.read_csv(REPO / "results/abc2/constraint_m1_summary.csv")
            if r["cond"] == common.POOL and r["variant"] == "baseline" and r["constraint"] == name
            and r["test_set"] == "eval_seen"]
    best = max(rows, key=lambda r: float(r["m1_per_task_mean"]))
    return best["beta"], float(best["m1_per_task_mean"])


def build_baselines() -> None:
    import prep
    import stage6_cp
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    if not done("timestep"):
        t = np.broadcast_to(np.arange(T, dtype=np.float32), (idx.n, T)).copy()
        save("timestep", {s: {"p": t, "m1": t, "cp": np.where(kept, t, np.nan).astype(np.float32),
                              "detail": "t"} for s in SEEDS}, "new (alarm); the score is t itself")
    if not done("emb_best"):
        L, beta, sel = emb_best_layer()
        per = {}
        for s in SEEDS:
            x, run, sign = stage6_cp.emb_scores(ctx, common.sets(s)["train"], L, beta)
            run = np.where(run > -1e300, run, np.nan)
            per[s] = {"p": x.astype(np.float32), "m1": x.astype(np.float32), "cp": run.astype(np.float32),
                      "detail": f"emb_temp[{L}] beta={beta} sign={'+' if sign > 0 else '-'}"}
        save("emb_best", per, "new: emb_temp at the layer chosen on eval-seen per-task AUROC "
             "(results/abc2/constraint_m1_summary.csv), running maximum", sel)
    S, names = ctx.series("baseline")
    for key, name in (("token_entropy", "token_entropy_mean"), ("token_maxprob", "token_prob_mean")):
        if done(key):
            continue
        beta, score = baseline_beta(name)
        k = names.index((name, -1))
        E = prep.smooth(S[k:k + 1], names[k:k + 1], 0.0 if beta == "none" else float(beta))
        per = {}
        for s in SEEDS:
            mu, sd, sign, _ = prep.fit_scale_orient(E, idx, kept, common.sets(s)["train"])
            x = prep.apply_prep(E, mu, sd, sign)[0].astype(np.float32)
            per[s] = {"p": x, "m1": x, "cp": running_max(x),
                      "detail": f"{name} beta={beta} sign={'+' if sign[0] > 0 else '-'}"}
        save(key, per, "new (alarm): prepared as every constraint, running maximum",
             {"series": name, "beta": beta, "eval_seen_per_task_mean": score,
              "rule": "beta with the best eval-seen per-task AUROC, mean over seeds"})
    if not (done("failurespot") and done("failurespot_neg")):
        fs = ctx.series("failurespot")
        fs = np.where(np.isnan(fs), 0.0, fs).astype(np.float32)
        for key, sgn in (("failurespot", 1.0), ("failurespot_neg", -1.0)):
            x = sgn * fs
            save(key, {s: {"p": x, "m1": x, "cp": running_max(x), "detail": f"{'+' if sgn > 0 else '-'}s beta=0.8"}
                       for s in SEEDS}, "new (alarm): FailureSpot's signal, fixed orientation, running maximum")


def build_mlps() -> None:
    """C's and B's new MLPs are trained by mlp_c.py / mlp_b.py; here they are read."""
    for key, path in (("C_mlp", common.WORK_DIR / "mlp" / "C_mlp.pkl"),
                      ("B_mlp_h10", common.WORK_DIR / "mlp" / "B_mlp_h10.pkl")):
        if done(key) or not path.exists():
            continue
        w = load_pkl(path)
        save(key, {s: entry(w["p"][s], w["detail"][s]) for s in SEEDS}, w["source"], w.get("selection"))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    t0 = time.time()
    steps = {"reused": build_reused, "a": build_a, "c": build_c, "b": build_b, "b_targets": build_b_targets,
             "baselines": build_baselines, "mlps": build_mlps}
    for name, fn in steps.items():
        if a.only and name not in a.only:
            continue
        fn()
        log(f"[det] step {name} ok ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
