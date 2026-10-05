"""
setup_c.py

Stage 3: setup C, constraints -> episode outcome (handoff v9, sections 2 and 4).

  * beta: one per family (action; emb_temp; cross-layer), each the beta whose best
    series in that family has the highest eval_seen M1-per-task in Stage 2's table
    (results/abc/constraint_m1.csv), restricted to the series C uses. The full
    family x beta grid is printed and saved.
  * 33 probes: per raw layer l in 0..31, [emb_temp[l], xl_adj[min(l,30)],
    xl_final[min(l,30)]]; one action probe [act_mag, act_dir, grip_flip, act_rep].
  * final detector: every prepared series C uses (emb_temp 32, xl_adj 31, xl_final 31,
    xl_spread, act_mag, act_dir, grip_flip, act_rep) plus |emb_temp| and |act_mag|,
    132 inputs. Ablations: the same minus one family at a time.
  * every model: StandardScaler -> LogisticRegression(C=c, max_iter=2000), trained on
    outcome with data.train_rows_and_weights (>= 5-successes cut, 1/n_class(t) weights
    rescaled to mean 1). c is gridded over C_GRID and chosen by the best mean eval_seen
    M1-per-task over seeds (plain grid search, as SAFE). One c for all 33 probes (scored
    by the mean over probes), one for the final detector; the ablations use the final's c.
  * score over time: the running sum of predicted failure probabilities over kept rows,
    s_t = sum_{tau <= t} p_tau, so M1's maximum is the value at the cut.
  * floor: M1 of a running sum of a constant (recomputed; it orders rollouts like the
    timestep score, so it must equal data.m1's timestep floor).
  * comparison: a 1-input probe on emb_temp[28] at the probes' c, scored by its maximum
    and by its running sum, beside Stage 2's raw maximum.
  * unstable flags (sign differs across seeds) are read from Stage 2's CSV at the
    chosen beta and carried into every table. Unstable inputs are still used.
  * metrics per model: pooled M1, M1-per-task and the floor, eval_seen and unseen,
    mean +- std over seeds (ddof = 1).

The 12 (seed, c) combinations run in parallel processes.

Outputs (results/abc/):
    c_probes.csv          one row per probe, at the chosen probe c
    c_final.csv           final detector and the three ablations, at the chosen final c
    c_final_weights.csv   one row per final-detector input: coefficient per seed
    c_grid.csv            every model at every c (the full grid)
    c_betas.csv           the family x beta grid behind the beta choice
    scores/c_seed{s}.npz  final detector and ablations at the chosen c: per-row
                          probability p_<model> and running sum s_<model> (Stage 6)
and /data/tmp/abc_scores/c_probes_seed{s}.npz, per-row probabilities of the 33 probes
at the chosen c (too large to keep in the repo).

Usage (from the repo root):
    source env.sh
    python analysis/abc/setup_c.py
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from constraint_m1 import weight_diagnostics
from data import (RESULTS_DIR, SEEDS, T, kept_mask, load_index, load_split, m1,
                  train_rows_and_weights)
from prep import (BETAS, PRIMARY, apply_prep, beta_label, fit_scale_orient, load_series,
                  model_input, smooth)

STAGE2_CSV = RESULTS_DIR / "constraint_m1.csv"
SCORES_DIR = RESULTS_DIR / "scores"
PROBE_SCORES_DIR = Path("/data/tmp/abc_scores")

FAMILIES = {"action": ["act_mag", "act_dir", "grip_flip", "act_rep"],
            "emb_temp": ["emb_temp"],
            "cross_layer": ["xl_adj", "xl_final", "xl_spread"]}
ABS_INPUTS = ("emb_temp", "act_mag")
C_GRID = (1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
COMPARE = "emb_temp[28]"
N_RAW = 32
SETS = (("seen", "eval_seen"), ("unseen", "unseen"))
METRICS = ("pooled", "per_task")


def fam_of(base: str) -> str:
    return next(f for f, bases in FAMILIES.items() if base in bases)


def series_name(base: str, layer: int) -> str:
    return f"{base}[{layer}]" if layer >= 0 else base


def base_of(feature: str) -> tuple[str, int]:
    nm = feature.strip("|")
    if "[" in nm:
        b, L = nm[:-1].split("[")
        return b, int(L)
    return nm, -1


# --- Stage 2 table: beta choice and unstable flags -------------------------------------
def read_stage2() -> list[dict]:
    rows = list(csv.DictReader(STAGE2_CSV.open()))
    c_series = {b for bases in FAMILIES.values() for b in bases}
    out = []
    for r in rows:
        if r["variant"] != "primary" or r["constraint"] not in c_series:
            continue
        r["layer"] = int(r["layer"])
        r["unstable"] = r["unstable"] == "True"
        for k in ("pooled_seen_mean", "per_task_seen_mean", "pooled_unseen_mean",
                  "per_task_unseen_mean"):
            r[k] = float(r[k])
        out.append(r)
    return out


def choose_betas(s2: list[dict]) -> tuple[dict, list[dict]]:
    """Per family, the beta whose best series has the highest eval_seen M1-per-task."""
    grid, chosen = [], {}
    for fam, bases in FAMILIES.items():
        best_by_beta = {}
        for beta in map(beta_label, BETAS):
            cand = [r for r in s2 if r["constraint"] in bases and r["beta"] == beta]
            best = max(cand, key=lambda r: r["per_task_seen_mean"])
            best_by_beta[beta] = best
            grid.append({"family": fam, "beta": beta,
                         "best_series": series_name(best["constraint"], best["layer"]),
                         "best_per_task_seen": best["per_task_seen_mean"],
                         "best_pooled_seen": best["pooled_seen_mean"],
                         "median_per_task_seen": float(np.median([r["per_task_seen_mean"] for r in cand])),
                         "n_unstable": sum(r["unstable"] for r in cand), "n_series": len(cand)})
        chosen[fam] = max(best_by_beta, key=lambda b: best_by_beta[b]["per_task_seen_mean"])
    for g in grid:
        g["chosen"] = g["beta"] == chosen[g["family"]]
    return chosen, grid


# --- Features ------------------------------------------------------------------------
def smoothed_families(idx, chosen):
    """{family: (E (n, 300, 520), names)} at that family's chosen beta."""
    out = {}
    for fam, bases in FAMILIES.items():
        spec = [(b, layered) for b, layered in PRIMARY if b in bases]
        S, names = load_series(idx, spec)
        beta = 0.0 if chosen[fam] == "none" else float(chosen[fam])
        out[fam] = (smooth(S, names, beta), names)
    return out


def seed_features(fams, idx, kept, train_pos):
    """Model inputs for one seed: {name: (300, 520) float32}, plus orientation signs."""
    feats, signs = {}, {}
    for fam, (E, names) in fams.items():
        mu, sd, sign, _ = fit_scale_orient(E, idx, kept, train_pos)
        P = model_input(apply_prep(E, mu, sd, sign)).astype(np.float32)
        for k, (base, L) in enumerate(names):
            nm = series_name(base, L)
            feats[nm] = P[k]
            signs[nm] = float(sign[k])
            if base in ABS_INPUTS:
                feats[f"|{nm}|"] = np.abs(P[k])
                signs[f"|{nm}|"] = float(sign[k])
    return feats, signs


# --- Models --------------------------------------------------------------------------
def running_sum(p: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """s_t = sum of p over kept rows tau <= t. (300, 520)."""
    return np.cumsum(np.where(kept, p, 0.0), axis=1)


def fit_prob(inputs, feats, train, c):
    """Fit on the weighted train rows; return (model, (300, 520) P(failure), converged)."""
    r, t, y, w = train
    X = np.stack([feats[n][r, t] for n in inputs], 1)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y, logisticregression__sample_weight=w)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in caught)
    Xall = np.stack([feats[n].reshape(-1) for n in inputs], 1)
    p = model.predict_proba(Xall)[:, 1].reshape(-1, T).astype(np.float32)
    return model, p, converged


def evaluate(score, idx, kept, split) -> dict:
    return {lab: m1(score, kept, idx, split["pos"][key]) for lab, key in SETS}


def run_job(seed, c, feats, probes, finals, idx, kept, split, train):
    """Every C model for one (seed, c). Returns metrics, coefficients and probabilities."""
    out = {"seed": seed, "c": c, "metrics": {}, "p": {}, "unconverged": []}
    models = dict(probes)
    models.update({f"final:{k}": v for k, v in finals.items()})
    models["compare"] = [COMPARE]
    for name, inputs in models.items():
        model, p, ok = fit_prob(inputs, feats, train, c)
        if not ok:
            out["unconverged"].append(name)
        s = running_sum(p, kept)
        out["metrics"][name] = evaluate(s, idx, kept, split)
        if name == "compare":
            out["metrics"]["compare_max"] = evaluate(p, idx, kept, split)
        out["p"][name] = p
        if name == "final:full":
            out["coef"] = dict(zip(inputs, model[-1].coef_[0].tolist()))
    return out


# --- Tables --------------------------------------------------------------------------
def summary_row(head: dict, per_seed: list[dict]) -> dict:
    d = dict(head)
    for lab, _ in SETS:
        for met in METRICS:
            v = np.array([ps[lab][met] for ps in per_seed])
            d[f"{met}_{lab}_mean"] = float(v.mean())
            d[f"{met}_{lab}_std"] = float(v.std(ddof=1))
        d[f"n_tasks_{lab}"] = per_seed[0][lab]["n_tasks"]
    for lab, _ in SETS:
        for met in METRICS:
            for s, ps in zip(SEEDS, per_seed):
                d[f"{met}_{lab}_s{s}"] = float(ps[lab][met])
    return d


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:g}" if k == "c" else f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def fmt(r: dict) -> str:
    return (f"pooled {r['pooled_seen_mean']:.3f}±{r['pooled_seen_std']:.3f} | "
            f"{r['pooled_unseen_mean']:.3f}±{r['pooled_unseen_std']:.3f}   "
            f"per-task {r['per_task_seen_mean']:.3f}±{r['per_task_seen_std']:.3f} | "
            f"{r['per_task_unseen_mean']:.3f}±{r['per_task_unseen_std']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="seed 0, c = 1 only, two probes: checks the pipeline, writes nothing")
    args = ap.parse_args()

    t0 = time.time()
    idx = load_index()
    kept = kept_mask(idx)
    splits = {s: load_split(s, idx) for s in SEEDS}
    seeds = (0,) if args.smoke else SEEDS
    c_grid = (1.0,) if args.smoke else C_GRID

    # 1. Row-weight diagnostics first.
    weight_diagnostics(idx, kept, splits)

    # Floor: running sum of a constant, which must order rollouts like the timestep score.
    const = running_sum(np.ones((idx.n, T), dtype=np.float32), kept)
    floor = {}
    for s in SEEDS:
        res = evaluate(const, idx, kept, splits[s])
        for lab, _ in SETS:
            assert np.isclose(res[lab]["pooled"], res[lab]["floor"]), "constant-sum floor"
            assert np.isclose(res[lab]["per_task"], 0.5), "constant-sum per-task floor"
        floor[s] = res
    floor_mean = {(lab, met): float(np.mean([floor[s][lab][met] for s in SEEDS]))
                  for lab, _ in SETS for met in METRICS}
    print(f"\n[floor] running sum of a constant: pooled seen {floor_mean[('seen', 'pooled')]:.3f}, "
          f"unseen {floor_mean[('unseen', 'pooled')]:.3f}; per-task "
          f"{floor_mean[('seen', 'per_task')]:.3f} | {floor_mean[('unseen', 'per_task')]:.3f} "
          f"(equals the timestep floor, as it must)")

    # beta per family, and unstable flags at that beta.
    s2 = read_stage2()
    chosen, beta_grid = choose_betas(s2)
    print("\n=== beta per family (best series' eval_seen M1-per-task in Stage 2) ===")
    for g in beta_grid:
        print(f"   {g['family']:12s} beta {g['beta']:4s}  best {g['best_series']:14s} "
              f"per-task {g['best_per_task_seen']:.3f} (pooled {g['best_pooled_seen']:.3f})  "
              f"family median per-task {g['median_per_task_seen']:.3f}  unstable "
              f"{g['n_unstable']}/{g['n_series']}{'   <- chosen' if g['chosen'] else ''}")
    unstable = {(r["constraint"], r["layer"]) for r in s2
                if r["unstable"] and r["beta"] == chosen[fam_of(r["constraint"])]}
    s2_compare = next(r for r in s2 if series_name(r["constraint"], r["layer"]) == COMPARE
                      and r["beta"] == chosen["emb_temp"])

    fams = smoothed_families(idx, chosen)
    probes = {f"layer{L:02d}": [f"emb_temp[{L}]", f"xl_adj[{min(L, 30)}]",
                                f"xl_final[{min(L, 30)}]"] for L in range(N_RAW)}
    probes["action"] = list(FAMILIES["action"])
    if args.smoke:
        probes = {k: probes[k] for k in ("layer28", "action")}

    feats_by_seed, signs_by_seed, trains = {}, {}, {}
    for s in seeds:
        feats_by_seed[s], signs_by_seed[s] = seed_features(fams, idx, kept, splits[s]["pos"]["train"])
        trains[s] = train_rows_and_weights(idx, kept, splits[s]["pos"]["train"])
    all_inputs = list(feats_by_seed[seeds[0]])
    finals = {"full": all_inputs}
    for fam in FAMILIES:
        finals[f"no_{fam}"] = [n for n in all_inputs if fam_of(base_of(n)[0]) != fam]
    print(f"\n[*] features ready ({time.time() - t0:.0f} s); fitting "
          f"{len(seeds) * len(c_grid)} (seed, c) jobs")

    jobs = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(run_job)(s, c, feats_by_seed[s], probes, finals, idx, kept, splits[s], trains[s])
        for s in seeds for c in c_grid)
    res = {(j["seed"], j["c"]): j for j in jobs}
    print(f"[*] fits done ({time.time() - t0:.0f} s)")
    unconv = sorted({(j["c"], n) for j in jobs for n in j["unconverged"]})
    if unconv:
        print(f"[warn] not converged (c, model): {unconv}")

    if args.smoke:
        for name in list(probes) + ["final:full", "compare", "compare_max"]:
            m = res[(0, 1.0)]["metrics"][name]
            print(f"   [smoke] {name:12s} seen pooled/task {m['seen']['pooled']:.3f}/"
                  f"{m['seen']['per_task']:.3f}  unseen {m['unseen']['pooled']:.3f}/"
                  f"{m['unseen']['per_task']:.3f}")
        print(f"[*] smoke done ({time.time() - t0:.0f} s); nothing written")
        return

    # --- Choose c --------------------------------------------------------------------
    def per_seed(name, c):
        return [res[(s, c)]["metrics"][name] for s in SEEDS]

    grid_rows = []
    for c in C_GRID:
        for name in list(probes) + [f"final:{k}" for k in finals] + ["compare", "compare_max"]:
            grid_rows.append(summary_row({"model": name, "c": c}, per_seed(name, c)))
    write_csv(RESULTS_DIR / "c_grid.csv", grid_rows)

    def seen_task_by_seed(names, c):
        """(3,) eval_seen M1-per-task per seed, averaged over `names`."""
        return np.array([np.mean([res[(s, c)]["metrics"][n]["seen"]["per_task"] for n in names])
                         for s in SEEDS])

    def best_c(names):
        """{c: (mean, std)} and the c with the best mean."""
        sc = {c: (float(v.mean()), float(v.std(ddof=1)))
              for c, v in ((c, seen_task_by_seed(names, c)) for c in C_GRID)}
        return sc, max(C_GRID, key=lambda c: sc[c][0])

    probe_score, c_probe = best_c(list(probes))
    final_score, c_final = best_c(["final:full"])
    n_consistent = {}
    for c in C_GRID:
        cs = np.array([[res[(s, c)]["coef"][n] for s in SEEDS] for n in all_inputs])
        n_consistent[c] = int(np.sum(np.all(cs > 0, 1) | np.all(cs < 0, 1)))
    write_csv(RESULTS_DIR / "c_grid_choice.csv", [
        {"c": c, "probes_seen_task_mean": probe_score[c][0], "probes_seen_task_std": probe_score[c][1],
         "final_seen_task_mean": final_score[c][0], "final_seen_task_std": final_score[c][1],
         "final_weights_sign_consistent": n_consistent[c], "final_n_weights": len(all_inputs),
         "chosen_probes": c == c_probe, "chosen_final": c == c_final} for c in C_GRID])

    # --- Tables at the chosen c ------------------------------------------------------
    probe_rows = []
    for p, inputs in probes.items():
        unst = [n for n in inputs if base_of(n) in unstable]
        probe_rows.append(summary_row(
            {"probe": p, "inputs": " ".join(inputs), "c": c_probe,
             "betas": " ".join(f"{f}={chosen[f]}"
                               for f in dict.fromkeys(fam_of(base_of(n)[0]) for n in inputs)),
             "unstable_inputs": " ".join(unst), "has_unstable": bool(unst)},
            per_seed(p, c_probe)))
    write_csv(RESULTS_DIR / "c_probes.csv", probe_rows)

    final_rows = []
    for name, inputs in finals.items():
        final_rows.append(summary_row(
            {"model": name, "c": c_final, "n_inputs": len(inputs),
             "n_unstable_inputs": sum(base_of(n) in unstable for n in inputs),
             "beta_action": chosen["action"], "beta_emb_temp": chosen["emb_temp"],
             "beta_cross_layer": chosen["cross_layer"]}, per_seed(f"final:{name}", c_final)))
    write_csv(RESULTS_DIR / "c_final.csv", final_rows)

    weight_rows = []
    for n in all_inputs:
        b, L = base_of(n)
        cs = np.array([res[(s, c_final)]["coef"][n] for s in SEEDS])
        weight_rows.append({
            "input": n, "family": fam_of(b), "is_abs": n.startswith("|"),
            "beta": chosen[fam_of(b)], "c": c_final,
            "orientation": "".join("+" if signs_by_seed[s][n] > 0 else "-" for s in SEEDS),
            "unstable": (b, L) in unstable,
            "coef_mean": float(cs.mean()), "coef_std": float(cs.std(ddof=1)),
            "coef_sign_consistent": bool(np.all(cs > 0) or np.all(cs < 0)),
            **{f"coef_s{s}": float(v) for s, v in zip(SEEDS, cs)}})
    write_csv(RESULTS_DIR / "c_final_weights.csv", weight_rows)
    write_csv(RESULTS_DIR / "c_betas.csv", beta_grid)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        fp = {k.split(":")[1]: v for k, v in res[(s, c_final)]["p"].items() if k.startswith("final:")}
        np.savez_compressed(SCORES_DIR / f"c_seed{s}.npz", rollout_ids=idx.rids, c=c_final,
                            **{f"p_{k}": v for k, v in fp.items()},
                            **{f"s_{k}": running_sum(v, kept).astype(np.float32)
                               for k, v in fp.items()})
        np.savez_compressed(PROBE_SCORES_DIR / f"c_probes_seed{s}.npz", c=c_probe,
                            p=np.stack([res[(s, c_probe)]["p"][p] for p in probes]),
                            probes=np.array(list(probes)), rollout_ids=idx.rids)

    # --- Report ----------------------------------------------------------------------
    print("\n=== c grid (eval_seen M1-per-task, mean ± std over seeds; best mean chosen) ===")
    print("   c        probes (mean of 33)      final:full            final weights sign-consistent")
    for c in C_GRID:
        tag_p = "chosen" if c == c_probe else ""
        tag_f = "chosen" if c == c_final else ""
        print(f"   {c:<8g} {probe_score[c][0]:.3f}±{probe_score[c][1]:.3f} {tag_p:7s}  "
              f"{final_score[c][0]:.3f}±{final_score[c][1]:.3f} {tag_f:7s}  "
              f"{n_consistent[c]}/{len(all_inputs)}")
    print(f"   probes: c = {c_probe:g} ({probe_score[c_probe][0]:.4f});  final: c = {c_final:g} "
          f"({final_score[c_final][0]:.4f})")
    print("   full grid, every model and both sets: results/abc/c_grid.csv")

    print(f"\n=== C probes at c = {c_probe:g}, running-sum score (eval_seen | unseen; floor "
          f"pooled {floor_mean[('seen', 'pooled')]:.3f} | {floor_mean[('unseen', 'pooled')]:.3f}, "
          f"per-task 0.5; tasks {probe_rows[0]['n_tasks_seen']} | {probe_rows[0]['n_tasks_unseen']}) ===")
    for r in probe_rows:
        flag = f"  UNSTABLE: {r['unstable_inputs']}" if r["has_unstable"] else ""
        print(f"   {r['probe']:8s} {fmt(r)}{flag}")

    print(f"\n=== C final detector and leave-one-family-out at c = {c_final:g} ===")
    for r in final_rows:
        print(f"   {r['model']:15s} ({r['n_inputs']:3d} in, {r['n_unstable_inputs']:2d} unstable) {fmt(r)}")

    n_cons = sum(w["coef_sign_consistent"] for w in weight_rows)
    print(f"\n=== final detector weights at c = {c_final:g}: sign consistent across seeds in "
          f"{n_cons}/{len(weight_rows)} inputs ===")
    for fam in FAMILIES:
        fw = [w for w in weight_rows if w["family"] == fam]
        print(f"   {fam:12s} consistent {sum(w['coef_sign_consistent'] for w in fw)}/{len(fw)}, "
              f"sum |mean coef| {sum(abs(w['coef_mean']) for w in fw):.3f}")
    for w in sorted(weight_rows, key=lambda w: -abs(w["coef_mean"]))[:10]:
        print(f"   {w['input']:16s} {w['coef_mean']:+.3f}±{w['coef_std']:.3f}  orient "
              f"{w['orientation']}  {'consistent' if w['coef_sign_consistent'] else 'SIGN VARIES'}"
              f"{'  UNSTABLE' if w['unstable'] else ''}")

    print(f"\n=== {COMPARE} alone ===")
    print(f"   raw z-score, maximum (Stage 2, no training): pooled {s2_compare['pooled_seen_mean']:.3f} | "
          f"{s2_compare['pooled_unseen_mean']:.3f}   per-task {s2_compare['per_task_seen_mean']:.3f} | "
          f"{s2_compare['per_task_unseen_mean']:.3f}")
    for name, label in (("compare_max", "1-input probe, maximum of p"),
                        ("compare", "1-input probe, running sum of p")):
        print(f"   {label:30s} c = {c_probe:g}: {fmt(summary_row({}, per_seed(name, c_probe)))}")
    print(f"\n[*] done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
