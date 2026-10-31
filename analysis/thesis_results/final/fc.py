"""
fc.py

Shared plumbing for the results follow-up (results_followup_handoff.md, version 4).

The headline setup is now the full training set (repo key D2), tested on unseen tasks
incl. LIBERO-PRO (T2). Everything here is a thin layer over analysis/thesis_results
(imported, never edited), with two redirections:

  * new detector work files go to /data/tmp/thesis_results_work/final/<original|full>/
    (detectors.WD and evaluate.EV are pointed there); a key that is not there is read
    from the earlier run's work files (/data/tmp/thesis_results_work/<original|full>/),
    so every existing detector is reused, never refitted;
  * outputs go to results/thesis_results_final/.

The training set is THESIS_POOL (D2 by default; D0 only for the original-set side of the
training-data ablation, run as a separate process).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("THESIS_POOL", "D2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import detectors as det  # noqa: E402
import evaluate as ev  # noqa: E402
from common import REPO, SEEDS, T, load_pkl, log, m1_row, m1_safe, read_csv, save_pkl, write_csv  # noqa: E402,F401

OUT = REPO / "results/thesis_results_final"
ORIG = REPO / "results/thesis_results"
FULL = REPO / "results/thesis_results_full"
WORK_ROOT = Path("/data/tmp/thesis_results_work")
FWD = WORK_ROOT / "final" / ("original" if common.MAIN else "full")
HEAD = "T2"
HEAD_NAME = common.TEST_SETS[HEAD]
REF = "T0"
REF_NAME = common.TEST_SETS[REF]

# --- redirect detector / evaluation work files ------------------------------------------
EARLIER_DET = det.WD
EARLIER_EV = ev.EV
det.WD = FWD / "detectors"
ev.EV = FWD / "eval"


def _load(key: str) -> dict:
    p = det.WD / f"{key}.pkl"
    return load_pkl(p if p.exists() else EARLIER_DET / f"{key}.pkl")


def _done(key: str) -> bool:
    return (det.WD / f"{key}.pkl").exists() or (EARLIER_DET / f"{key}.pkl").exists()


det.load = _load
det.done = _done


def is_new(key: str) -> bool:
    return (det.WD / f"{key.split(':')[0]}.pkl").exists()


def evaluate(key: str) -> dict:
    """evaluate.eval_key's result for a key: the earlier run's file for existing keys,
    computed (and cached under FWD/eval) for keys made here."""
    old = EARLIER_EV / f"{key.replace(':', '__')}.pkl"
    if not is_new(key) and old.exists():
        return load_pkl(old)
    return ev.eval_key(key)


def npz_dir(tag: str) -> Path:
    return FWD / "npz" / tag


def write_npz(keys, tag: str) -> list:
    """bootstrap2-format npz per seed (as evaluate.write_npz) holding `keys`, under
    FWD/npz/<tag>/. Returns the file list (seed order)."""
    import stage6_cp
    ctx = common.ctx()
    idx = ctx.idx
    d = npz_dir(tag)
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for s in SEEDS:
        st = common.sets(s)
        z = {"rollout_ids": idx.rids, "cond": common.POOL, "seed": s,
             "alphas": np.array(stage6_cp.ALPHAS), "kept_len": idx.kept_len, "y": idx.y,
             "cell": idx.task, "base_task": idx.base_task, "block": idx.block,
             "cal_rids": idx.rids[st["cal"]], "train_rids": idx.rids[st["train"]],
             "eval_seen_rids": idx.rids[st["eval_seen"]]}
        for k in keys:
            e = ev.entries(k)[s]
            b = evaluate(k)["bands"][s]
            kk = k.replace(":", "__")
            z.update({f"{kk}_m1": np.asarray(e["m1"], np.float32), f"{kk}_cp": np.asarray(e["cp"], np.float32),
                      f"{kk}_mu": b["mu"], f"{kk}_xi": b["xi"], f"{kk}_h": b["h"]})
        out = d / f"seed{s}.npz"
        tmp = d / f"seed{s}.tmp.npz"
        np.savez(tmp, **z)
        tmp.replace(out)
        files.append(out)
    return files


def per_seed_rows(keys, labels: dict | None = None) -> dict:
    """auroc / alarm / stress per-seed rows (evaluate.py's format) for `keys`."""
    rows = {"auroc": [], "alarm": [], "stress": []}
    for k in keys:
        r = evaluate(k)
        lab = (labels or {}).get(k, {})
        for part in rows:
            for x in r[part]:
                xx = {**lab, **x}
                if "test_set" in xx:
                    xx["test_set"] = common.TEST_SETS[xx["test_set"]]
                rows[part].append(xx)
    return rows


# --- statistics -------------------------------------------------------------------------
def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else np.nan)


def r4(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def add_stat(row: dict, name: str, per_seed: dict) -> None:
    m, s = ms([per_seed.get(k) for k in SEEDS])
    row[f"{name} mean"] = r4(m)
    row[f"{name} std"] = r4(s)
    for k in SEEDS:
        row[f"{name} seed {k}"] = r4(per_seed.get(k, np.nan))


def pm(m, s, nd=3) -> str:
    if m == "" or m is None or (isinstance(m, float) and not np.isfinite(m)):
        return ""
    return f"{float(m):.{nd}f} ± {float(s):.{nd}f}" if s not in ("", None) and np.isfinite(float(s)) else f"{float(m):.{nd}f}"


VERDICT = {"improvement": "x higher (interval above zero)",
           "significant worsening": "x lower (interval below zero)",
           "no significant change": "interval includes zero"}
METRIC_NAME = {"m1_per_task": "per-task AUROC", "bal_acc": "balanced accuracy at alpha 0.1",
               "tpr": "share of failures caught at alpha 0.1"}


def bootstrap_rows(pairs, test_set: str = HEAD, metrics=("m1_per_task", "bal_acc"), subset=None,
                   test_label: str | None = None) -> list:
    """Paired bootstrap (analysis/abc2/bootstrap2, exactly as paired.py: 1,000 resamples
    within cell within seed, seed means, 95% percentile interval, each seed's band held
    fixed). pairs: [(files_x, key_x, files_y, key_y, title, label_x, label_y)].
    subset: optional fn(SeedEval, npz path) -> bool mask restricting the test set.
    One set of resamples per call (drawn on the first pair's y side)."""
    import bootstrap2 as b2
    cache, boots = {}, {}

    def side(files, key):
        k = (tuple(map(str, files)), key)
        if k not in cache:
            sd = b2.load_side(files, key.replace(":", "__"), test_set)
            if subset is not None:
                sd = [restrict(x, subset(x, f)) for x, f in zip(sd, files)]
            cache[k] = sd
        return cache[k]

    draws = None
    rows = []
    pt_label = common.PER_TASK_LABEL.get(test_set, "per-task AUROC")
    for fx, kx, fy, ky, title, lx, ly in pairs:
        x, y = side(fx, kx), side(fy, ky)
        if draws is None:
            draws = b2.make_draws(y, b2.N_BOOT, b2.RNG_SEED)
        for k_, s_ in ((kx, x), (ky, y)):
            if (id(s_), k_) not in boots:
                boots[(id(s_), k_)] = b2.side_boot(s_, draws)
        r = b2.compare(x, y, draws, test_set=test_set, n_boot=b2.N_BOOT, rng_seed=b2.RNG_SEED, comparison=title,
                       x_boot=boots[(id(x), kx)], y_boot=boots[(id(y), ky)])
        for m in r["rows"]:
            if m["metric"] not in metrics:
                continue
            rows.append({"comparison (x - y)": title, "x": lx, "y": ly,
                         "test set": test_label or common.TEST_SETS.get(test_set, test_set),
                         "metric": pt_label if m["metric"] == "m1_per_task" else METRIC_NAME[m["metric"]],
                         "x value": r4(m["x_point"]), "y value": r4(m["y_point"]),
                         "difference": r4(m["diff_point"]), "bootstrap mean difference": r4(m["diff_mean"]),
                         "95% interval low": r4(m["diff_ci_lo"]), "95% interval high": r4(m["diff_ci_hi"]),
                         "verdict": VERDICT.get(m["verdict"], m["verdict"]),
                         "resamples": m["n_boot"], "valid resamples": m["n_valid"],
                         "x per seed": m["x_point_per_seed"], "y per seed": m["y_point_per_seed"],
                         "x key": kx, "y key": ky})
    return rows


def restrict(side, mask):
    """A bootstrap2.SeedEval restricted to the rollouts where mask is True."""
    from dataclasses import replace
    m = np.asarray(mask, bool)
    return replace(side, rids=side.rids[m], cell=side.cell[m], y=side.y[m], kept_len=side.kept_len[m],
                   runmax=side.runmax[m], first_alarm=side.first_alarm[m])


def quiet():
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def write_scores(keys) -> Path:
    """results/thesis_results_final/scores/<training set>_seed<s>.npz for the detector keys
    made here, in evaluate.write_scores' format: K__step (N, 520) float16 per-step value
    (P(failure), or the prepared z-score for untrained series), K__band_mu / K__band_xi
    (520,), K__h (16,) for `alphas`, K__band = mu + h(alpha 0.1) xi; common arrays
    rollout_ids, y (1 = failure), kept_len, cell, block, train_rids, eval_seen_rids, cal_rids."""
    import stage6_cp
    idx = common.ctx().idx
    d = OUT / "scores"
    d.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        st = common.sets(s)
        z = {"rollout_ids": idx.rids, "y": idx.y, "kept_len": idx.kept_len, "cell": idx.task, "block": idx.block,
             "train_rids": idx.rids[st["train"]], "eval_seen_rids": idx.rids[st["eval_seen"]],
             "cal_rids": idx.rids[st["cal"]], "alphas": np.array(stage6_cp.ALPHAS),
             "training_set": common.POOL_NAME, "seed": s}
        for k in keys:
            e = ev.entries(k)[s]
            b = evaluate(k)["bands"][s]
            z[f"{k}__step"] = np.asarray(e["p"], np.float32).astype(np.float16)
            z[f"{k}__band_mu"] = np.asarray(b["mu"], np.float32)
            z[f"{k}__band_xi"] = np.asarray(b["xi"], np.float32)
            z[f"{k}__h"] = np.asarray(b["h"], np.float32)
            z[f"{k}__band"] = np.asarray(b["mu"] + b["h"][-1] * b["xi"], np.float32)
        out = d / f"{common.POOL_SLUG}_seed{s}.npz"
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **z)
        tmp.replace(out)
    return d


def new_keys() -> list:
    return sorted(p.stem for p in (FWD / "detectors").glob("*.pkl"))
