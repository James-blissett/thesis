"""
evaluate.py

Evaluates every detector built by detectors.py on every test set, per seed, with the
existing code only (results_collection_handoff.md §2, §4.1, §4.2):

  AUROC      pipe2_common.m1_safe (data.m1): per-task AUROC (per task-and-perturbation
             cell on the LIBERO-PRO test set; cells with >= 2 rollouts of each class),
             pooled AUROC, and the timestep-only floor; on eval-seen, unseen tasks
             (original rollouts) and unseen tasks incl. LIBERO-PRO.
  alarm      cut-variant functional conformal band (cp_eval.fit_band, band_h,
             evaluate_cp): calibrated on the training set's successful eval-seen
             rollouts, halves drawn with default_rng(CAL_SPLIT_SEED + seed); 15 alphas
             (linspace(0.02, 0.9)) and 0.1; alarm at the first kept row before the cut
             (shortest kept length in the cell within the evaluated set) whose score
             exceeds the band. Balanced accuracy, share of failures caught (tpr),
             false-alarm rate on successes (1 - tnr), mean normalised detection time
             (first alarm step / cut, 1 if none, over failures).
  stress     share of stress-set rollouts alarmed before the cut at alpha 0.1, with
             stage6_cp's cut rule (base task's unperturbed cut from the same seed);
             all stress rollouts, and seen / unseen base tasks separately.

Outputs (results/thesis_results/ or results/thesis_results_full/):
  tables/data/detector_auroc_per_seed.csv    one row per (detector, seed, test set)
  tables/data/detector_alarm_per_seed.csv    one row per (detector, seed, test set, alpha)
  tables/data/detector_stress_per_seed.csv   one row per (detector, seed, stress subset)
  scores/<training set>_seed<s>.npz          per-row scores of the new detectors (format
                                             in write_scores' docstring)
and work files <WORK_DIR>/eval/<key>.pkl and <WORK_DIR>/npz/seed<s>.npz (the
bootstrap2 format: every detector's m1 / cp / band, for paired.py).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import detectors as det  # noqa: E402
from common import SEEDS, T, load_pkl, log, m1_cut, m1_row, m1_safe, save_pkl, write_csv  # noqa: E402

EV = common.WORK_DIR / "eval"
NPZ = common.WORK_DIR / "npz"

# (detector, variant, key, H) in table order; plain names (handoff section 0).
ROWS = [
    ("A", "single best probe (of 34)", "A_best", ""),
    ("A", "unweighted mean of 34 probe outputs (primary)", "A_mean", ""),
    ("A", "trained combiner over 34 probe outputs (out-of-fold)", "A_combiner", ""),
    ("A", "2-layer MLP on the best probe's hidden state", "A_mlp", ""),
    ("C", "single best probe (of 33)", "C_best", ""),
    ("C", "unweighted mean of 33 probe outputs (primary)", "C_mean", ""),
    ("C", "trained combiner over 33 probe outputs (out-of-fold)", "C_combiner", ""),
    ("C", "2-layer MLP on the best probe's inputs", "C_mlp", ""),
    ("C (exploratory)", "logistic regression on all 132 constraints", "C_lr132", ""),
    ("C (exploratory)", "2-layer MLP on all 132 constraints", "C_mlp132", ""),
]
for _H in common.HS:
    ROWS += [
        ("B", "single best probe (of 124), chosen on eval-seen per-task AUROC against outcome", f"B_best_h{_H}", _H),
        ("B", "unweighted mean of 124 probe outputs (primary)", f"B_mean_h{_H}", _H),
        ("B", "unweighted mean of the 30 embedding-temporal probe outputs", f"B_meanemb_h{_H}", _H),
        ("B", "trained combiner over 124 probe outputs (out-of-fold)", f"B_combiner_h{_H}", _H),
    ]
ROWS += [
    ("B", "2-layer MLP on the best probe's layer, trained on that probe's constraint target", "B_mlp_h10", 10),
    ("Baseline", "timestep only", "timestep", ""),
    ("Baseline", "best single untrained constraint (layer chosen on eval-seen per-task AUROC), running maximum", "emb_best", ""),
    ("Baseline", "untrained constraint at the earlier layer choice (eval-seen pooled AUROC), running maximum", "emb_pooled", ""),
    ("Baseline", "token mean entropy, running maximum", "token_entropy", ""),
    ("Baseline", "token max-probability, running maximum", "token_maxprob", ""),
    ("Baseline", "FailureSpot signal s, running maximum", "failurespot", ""),
    ("Baseline", "FailureSpot signal, opposite orientation (-s), running maximum", "failurespot_neg", ""),
]
def _b_counts():
    """B's probe counts per H for this training set (124 / 30 on the original set)."""
    m = common.WORK_DIR / "b_probes" / "meta.json"
    if not m.exists():
        return 124, 30
    d = json.loads(m.read_text())
    return int(d["n_probes_per_H"]), sum("emb_temp" in v for v in d["kinds"].values())


_NB, _NE = _b_counts()
ROWS = [(d, v.replace("124", str(_NB)).replace("the 30 embedding", f"the {_NE} embedding") if d == "B" else v, k, H)
        for d, v, k, H in ROWS]
EXTRA = [("B", "trained combiner over 124 probe outputs, refitted here (check)", "B_combiner_h10_refit", 10)]
ROW_KEYS = [r[2] for r in ROWS] + [r[2] for r in EXTRA]
LABEL = {r[2]: r for r in ROWS + EXTRA}
STRESS_SUBSETS = ("stress", "stress_seen", "stress_unseen")


def entries(key: str) -> dict:
    """{seed: entry} for a detector key; 'C_mean_lofo:<variant>' picks a sub-entry."""
    if ":" in key:
        base, sub = key.split(":")
        w = det.load(base)
        return {s: w["seeds"][s][sub] for s in SEEDS}
    return det.load(key)["seeds"]


def sub_label(key: str) -> dict:
    """detector / variant / H for a table row key, including the ablation sub-keys."""
    if ":" not in key:
        lab = LABEL[key]
        return {"detector": lab[0], "variant": lab[1], "H": lab[3]}
    base, sub = key.split(":")
    if base == "C_mean_lofo":
        fam = {"no_action": "action", "no_emb_temp": "emb_temp", "no_cross_layer": "cross-layer"}[sub]
        return {"detector": "C", "variant": f"unweighted mean of C's probes without the {fam} family", "H": ""}
    H = int(base.split("_h")[1])
    m, kd = sub.split("_", 1)
    v = "unweighted mean" if m == "mean" else "trained combiner"
    return {"detector": "B", "variant": f"{v} of the {kd} probes only", "H": H}


def eval_key(key: str) -> dict:
    import cp_eval as abc_cp
    import stage6_cp
    path = EV / f"{key.replace(':', '__')}.pkl"
    if path.exists():
        return load_pkl(path)
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    E = entries(key)
    auroc, alarm, stress, bands = [], [], [], {}
    for s in SEEDS:
        e = E[s]
        st = common.sets(s)
        head = {"key": key, "seed": s, "detail": e["detail"]}
        for ts in common.AUROC_SETS:
            auroc.append({**head, "test_set": ts, **m1_row(m1_safe(e["m1"], kept, idx, st[ts]))})
        cal = st["cal"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu, xi, R, t_end, n1, n2 = abc_cp.fit_band(e["cp"], kept, cal,
                                                      np.random.default_rng(abc_cp.CAL_SPLIT_SEED + s))
            hs = np.array([abc_cp.band_h(R, a) for a in stage6_cp.ALPHAS])
            for ts in common.AUROC_SETS:
                pos = st[ts]
                cut = m1_cut(idx, pos)
                for a, h in zip(stage6_cp.ALPHAS, hs):
                    r = abc_cp.evaluate_cp(e["cp"], kept, idx, pos, mu, xi, h, cut)
                    alarm.append({**head, "test_set": ts, "alpha": float(a),
                                  "on_alpha_grid": bool(a in abc_cp.ALPHAS), "h": float(h),
                                  "bal_acc": r["bal_acc"], "tpr": r["tpr"], "far": 1 - r["tnr"],
                                  "det_time": r["det_time"], "n": int(len(pos)),
                                  "n_fail": int(idx.y[pos].sum()), "n_cal": int(len(cal)),
                                  "n_d1": n1, "n_d2": n2, "t_end": t_end})
            for ss in STRESS_SUBSETS:
                pos = st[ss]
                cut = stage6_cp.stress_cuts(ctx, s, st, pos)
                fired = stage6_cp.alarms(e["cp"], kept, pos, mu, xi, hs[-1], cut)
                for blk in ("all", "swap", "task"):
                    m = np.ones(len(pos), bool) if blk == "all" else idx.block[pos] == blk
                    stress.append({**head, "stress_set": ss, "block": blk, "alpha": 0.1,
                                   "n": int(m.sum()), "n_fail": int(idx.y[pos][m].sum()),
                                   "share_alarm_before_cut": float(fired[m].mean()) if m.any() else np.nan,
                                   "mean_cut": float(cut[m].mean()) if m.any() else np.nan})
        bands[s] = {"mu": mu, "xi": xi, "h": hs, "R": R, "t_end": t_end, "n_d1": n1, "n_d2": n2}
    out = {"auroc": auroc, "alarm": alarm, "stress": stress, "bands": bands}
    EV.mkdir(parents=True, exist_ok=True)
    save_pkl(path, out)
    return out


def available_keys() -> list:
    keys = [k for k in ROW_KEYS if det.done(k)]
    if det.done("C_mean_lofo"):
        keys += [f"C_mean_lofo:{v}" for v in ("no_action", "no_emb_temp", "no_cross_layer")]
    for H in common.HS:
        if det.done(f"B_bytarget_h{H}"):
            keys += [f"B_bytarget_h{H}:{m}_{kd}" for m in ("mean", "combiner")
                     for kd in ("act_rep", "act_mag", "emb_temp", "xl_final")]
    return keys


def write_npz(keys) -> None:
    """bootstrap2-format npz per seed with every key's m1 / cp / band."""
    import stage6_cp
    ctx = common.ctx()
    idx = ctx.idx
    NPZ.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        st = common.sets(s)
        z = {"rollout_ids": idx.rids, "cond": common.POOL, "seed": s,
             "alphas": np.array(stage6_cp.ALPHAS), "kept_len": idx.kept_len, "y": idx.y,
             "cell": idx.task, "base_task": idx.base_task, "block": idx.block,
             "cal_rids": idx.rids[st["cal"]], "train_rids": idx.rids[st["train"]],
             "eval_seen_rids": idx.rids[st["eval_seen"]]}
        for k in keys:
            e = entries(k)[s]
            b = eval_key(k)["bands"][s]
            kk = k.replace(":", "__")
            z.update({f"{kk}_m1": np.asarray(e["m1"], np.float32), f"{kk}_cp": np.asarray(e["cp"], np.float32),
                      f"{kk}_mu": b["mu"], f"{kk}_xi": b["xi"], f"{kk}_h": b["h"]})
        tmp = NPZ / f"seed{s}.tmp.npz"
        np.savez(tmp, **z)
        tmp.replace(NPZ / f"seed{s}.npz")


def write_scores(keys) -> None:
    """results scores/<training set>_seed<s>.npz: for every NEW detector key K,
    K__step (N, 520) float16 = the per-step value (P(failure), or the prepared z-score
    for untrained series; NaN where missing), K__band (520,) float32 = the alpha 0.1
    band mu + h xi, K__band_mu / K__band_xi (520,), K__h (16,) for the alphas in `alphas`;
    common arrays rollout_ids, y (1 = failure), kept_len (kept rows 1 <= t < kept_len),
    cell, block, train_rids, eval_seen_rids, cal_rids. The AUROC score is the running sum
    of K__step over kept rows (trained models and probe means) or K__step itself
    (untrained series); the alarm score is that running sum, or the running maximum."""
    import stage6_cp
    ctx = common.ctx()
    idx = ctx.idx
    d = common.OUT_DIR / "scores"
    d.mkdir(parents=True, exist_ok=True)
    new = [k for k in keys if not det.load(k.split(":")[0])["source"].startswith("results/abc2")]
    for s in SEEDS:
        st = common.sets(s)
        z = {"rollout_ids": idx.rids, "y": idx.y, "kept_len": idx.kept_len, "cell": idx.task,
             "block": idx.block, "train_rids": idx.rids[st["train"]],
             "eval_seen_rids": idx.rids[st["eval_seen"]], "cal_rids": idx.rids[st["cal"]],
             "alphas": np.array(stage6_cp.ALPHAS), "training_set": common.POOL_NAME, "seed": s}
        for k in new:
            e = entries(k)[s]
            b = eval_key(k)["bands"][s]
            kk = k.replace(":", "__")
            z[f"{kk}__step"] = np.asarray(e["p"], np.float32).astype(np.float16)
            z[f"{kk}__band_mu"] = np.asarray(b["mu"], np.float32)
            z[f"{kk}__band_xi"] = np.asarray(b["xi"], np.float32)
            z[f"{kk}__h"] = np.asarray(b["h"], np.float32)
            z[f"{kk}__band"] = np.asarray(b["mu"] + b["h"][-1] * b["xi"], np.float32)
        out = d / f"{common.POOL_SLUG}_seed{s}.npz"
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **z)
        tmp.replace(out)


def main() -> None:
    t0 = time.time()
    keys = available_keys()
    rows = {"auroc": [], "alarm": [], "stress": []}
    for k in keys:
        r = eval_key(k)
        for part in rows:
            rows[part] += r[part]
        log(f"[eval] {k} ({time.time() - t0:.0f} s)")
    d = common.OUT_DIR / "tables" / "data"
    pl = {k: v for k, v in common.TEST_SETS.items()}
    for part, fname in (("auroc", "detector_auroc_per_seed.csv"), ("alarm", "detector_alarm_per_seed.csv"),
                        ("stress", "detector_stress_per_seed.csv")):
        out = []
        for r in rows[part]:
            rr = {**sub_label(r["key"]), **r}
            if "test_set" in rr:
                rr["test_set"] = pl[rr["test_set"]]
            out.append(rr)
        write_csv(d / fname, out)
    write_npz(keys)
    write_scores(keys)
    log(f"[eval] done: {len(keys)} detectors ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
