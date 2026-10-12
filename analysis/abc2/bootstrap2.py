"""
bootstrap2.py

Step 9's paired comparison (more_data_collection.md, handoff v1): does a training pool
(or a model) change a headline number on the same test rollouts? For each pair of
conditions (default D1 - D0 and D2 - D0), each detector (default A, C, emb) and each
headline number on T0 (by default):

    m1_per_task   M1-per-task (cells with >= 2 rollouts of each class)
    bal_acc       balanced accuracy of the conformal alarm at alpha = 0.1, cut variant
    tpr           share of failures caught by that alarm (alpha = 0.1, cut variant)

it bootstraps N_BOOT = 1000 times, resampling rollouts with replacement within each cell
within each seed's test set, recomputes the seed-averaged metric for BOTH sides on the
SAME resample, and reports the mean difference (x - y, e.g. D1 - D0), its 95% percentile
interval, and the point difference on the original sample. Rule (Step 9): a change is an
`improvement` only if the interval lies entirely above zero; an interval entirely below
zero is a significant WORSENING (`significant_worsening`); otherwise no significant change
(`verdict` says which).

Recomputation on a resample is the pipeline's own computation (analysis/abc/data.py m1 /
m1_cut, analysis/abc/cp_eval.py evaluate_cp), on the stored per-row scores of
results/abc2/scores/<cond>_seed<s>.npz (format: analysis/abc2/pipeline.py docstring):
  * cut: per cell, the shortest kept length among the RESAMPLED rollouts of that cell
    (duplicates count once each; a resample can only lengthen the cut);
  * M1-per-task: a rollout's score is its maximum of D_m1 over kept rows t < cut (NaN
    rows left out); within-cell ROC-AUC (Mann-Whitney with average ranks = sklearn's
    roc_auc_score, duplicates included) averaged over cells with >= 2 failures and >= 2
    successes in the resample. NaN if no cell qualifies or a rollout has no valid row
    under its cut (data.m1 raises there; the pipeline records NaN);
  * alarm (stored band, no re-calibration): fired = some kept row t < cut has
    D_cp(t) > D_mu(t) + D_h[j] D_xi(t), j = the stored alpha equal to 0.1 (NaN compares
    False); TPR over the resample's failures, TNR over its successes, balanced accuracy
    (TPR + TNR) / 2, pooled over the seed's cells as cp_eval.evaluate_cp does.
  * seed average: mean over seeds of the finite per-seed values (pipe2_common.mean_std).
Correctness test (always run, `<out>/paired_<test>_check.json`): with the identity
resample the recomputed per-seed and seed-mean values must equal the pipeline's own
(final_m1.csv, cp_curves.csv at alpha 0.1 off-grid, final_table.csv) to CSV precision
(|diff| <= 1e-6).

Resampling: one numpy Generator, default_rng(RNG_SEED = 20261011). For each seed (in the
order given) and each cell of that seed's test set (ascending cell id), one draw
rng.integers(0, n_cell, size=(n_boot, n_cell)). The same draws are used for every
detector and every comparison of one invocation, and are reproduced exactly by any later
call with the same test sets, seeds, n_boot and rng seed, so D1 - D0 and D2 - D0 (and the
MLP agent's A-MLP - A) share resamples.

Test sets are read from the npz alone (no split files): T0 = block ori rollouts in
neither train_rids nor eval_seen_rids (the split's unseen ori rollouts); unseen base
tasks = T0's base tasks; T1 = ori + ori2 in those tasks; T2 = every non-demoted training
block (ori, ori2, obj, lan minus demotions.json) in those tasks. Both sides of a
comparison must have the same test-set rollout ids in the same order (checked).

Python API (generic: any two lists of per-seed npz files with a detector key each; a key
K needs arrays K_m1, K_cp, K_mu, K_xi, K_h and the npz's common arrays rollout_ids, y,
kept_len, cell, base_task, block, train_rids, eval_seen_rids, alphas):

    import bootstrap2 as b2
    res = b2.paired_bootstrap(files_x, "A-MLP", files_y, "A", test_set="T0")
        # files_x / files_y: lists of npz paths, one per seed, in the same seed order
        # (e.g. [f"results/abc2/mlp_scores/D2_seed{s}.npz" for s in (0, 1, 2)])
        # optional: n_boot=1000, rng_seed=RNG_SEED, alpha=0.1, demotions=()
    res["rows"]          list of dicts, one per metric, the paired_<test>.csv columns
    res["per_seed"]      list of dicts, one per (seed, metric) (per-seed differences)
    res["identity"]      identity-resample values per side and seed (for the check)
Lower-level pieces: load_side(files, key, test_set, demotions) -> list[SeedEval];
make_draws(sides, n_boot, rng_seed); boot_metrics(side, draws) -> {metric: (n_boot,)};
identity_metrics(side); check_identity(...) against the pipeline tables.

Output (<out> = --out-dir, default results/abc2):
  paired_<test>.csv   one row per (comparison, detector, metric): test_set, comparison
                      ("D1 - D0"), cond_x, cond_y, detector (key, or "x vs y"),
                      detector_x, detector_y, metric, n_seeds, n_boot, n_valid, rng_seed,
                      x_point, y_point, diff_point (original sample, seed means),
                      diff_mean, diff_ci_lo, diff_ci_hi (2.5 / 97.5 percentiles),
                      x_boot_mean, y_boot_mean, improvement (ci_lo > 0),
                      significant_worsening (ci_hi < 0), verdict
                      (improvement / significant worsening / no significant change),
                      x_point_per_seed, y_point_per_seed (space-separated)
  paired_<test>_per_seed.csv  the same per seed (transparency; per-seed difference, its
                      percentile interval over the same resamples)
  paired_<test>_check.json    identity-resample check (passed, max |diff|, every value)

Usage (repo root, CPU only, a few seconds; real run 1000 reps x 3 detectors x 2
differences x 3 metrics on T0):
    source env.sh
    python analysis/abc2/bootstrap2.py                                 # T0, D1-D0, D2-D0
    python analysis/abc2/bootstrap2.py --test-set T0 T1 T2             # also T1 / T2
    python analysis/abc2/bootstrap2.py --in-dir /data/tmp/x --out-dir /data/tmp/x
    # generic pair (e.g. MLP vs linear in D2); {seed} is filled in per seed:
    python analysis/abc2/bootstrap2.py --pair \
        "results/abc2/mlp_scores/D2_seed{seed}.npz:A-MLP" "results/abc2/scores/D2_seed{seed}.npz:A" \
        --check-table results/abc2/mlp_final_table.csv --out-name mlp_paired_T0.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2  # noqa: E402
from data2 import T, TRAINING_BLOCKS  # noqa: E402
from pipe2_common import read_csv, write_csv  # noqa: E402

RNG_SEED = 20261011
N_BOOT = 1000
ALPHA = 0.1
CI = (2.5, 97.5)
MIN_PER_CLASS = 2                     # data.M1_TASK_MIN_PER_CLASS
METRICS = ("m1_per_task", "bal_acc", "tpr")
METRIC_LABEL = {"m1_per_task": "M1-per-task", "bal_acc": "balanced accuracy (alpha 0.1, cut)",
                "tpr": "share of failures caught (alpha 0.1, cut)"}
CHECK_TOL = 1e-6
NO_HIT = T + 1                        # "never alarms"


# --- One side (npz + detector), restricted to a test set --------------------------------
@dataclass
class SeedEval:
    path: str
    key: str
    cond: str
    seed: int
    rids: np.ndarray        # (n,) test-set rollout ids, index order
    cell: np.ndarray        # (n,)
    y: np.ndarray           # (n,) 1 = failure
    kept_len: np.ndarray    # (n,)
    runmax: np.ndarray      # (n, T) running max of the M1 score over valid kept rows (-inf none)
    first_alarm: np.ndarray  # (n,) first kept row with an alarm at alpha (NO_HIT if none)


def test_positions(z, test_set: str, demotions=()) -> np.ndarray:
    """Positions (index order) of a test set, from the npz's own arrays."""
    rids, block, base = z["rollout_ids"].astype(str), z["block"].astype(str), z["base_task"]
    used = np.isin(rids, np.concatenate([z["train_rids"].astype(str), z["eval_seen_rids"].astype(str)]))
    t0 = (block == "ori") & ~used
    unseen = np.unique(base[t0])
    if np.isin(base[used], unseen).any():
        raise SystemExit("unseen base tasks also appear in train / eval_seen: npz inconsistent")
    if test_set == "T0":
        m = t0
    elif test_set == "T1":
        m = np.isin(block, ["ori", "ori2"]) & np.isin(base, unseen)
    elif test_set == "T2":
        m = np.isin(block, [b for b in TRAINING_BLOCKS if b not in demotions]) & np.isin(base, unseen)
    else:
        raise ValueError(f"unknown test set {test_set}")
    return np.nonzero(m)[0]


def load_seed(path, key: str, test_set: str = "T0", demotions=(), alpha: float = ALPHA) -> SeedEval:
    z = np.load(path, allow_pickle=False)
    need = [f"{key}_{k}" for k in ("m1", "cp", "mu", "xi", "h")]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"{path}: missing {miss} (detector key {key!r})")
    pos = test_positions(z, test_set, demotions)
    if not len(pos):
        raise SystemExit(f"{path}: empty test set {test_set}")
    alphas = np.asarray(z["alphas"], float)
    js = np.nonzero(np.isclose(alphas, alpha))[0]
    if not len(js):
        raise SystemExit(f"{path}: alpha {alpha} not stored")
    j = int(js[-1])
    kept_len = z["kept_len"][pos].astype(np.int64)
    t = np.arange(T)[None, :]
    kept = (t >= 1) & (t < kept_len[:, None])
    s = z[f"{key}_m1"][pos].astype(np.float64)
    runmax = np.maximum.accumulate(np.where(kept & ~np.isnan(s), s, -np.inf), axis=1)
    band = z[f"{key}_mu"].astype(np.float64) + float(z[f"{key}_h"][j]) * z[f"{key}_xi"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        alarm = kept & (z[f"{key}_cp"][pos].astype(np.float64) > band[None, :])
    first = np.where(alarm.any(1), alarm.argmax(1), NO_HIT)
    return SeedEval(path=str(path), key=key, cond=str(z["cond"]), seed=int(z["seed"]),
                    rids=z["rollout_ids"][pos].astype(str), cell=z["cell"][pos].astype(np.int64),
                    y=z["y"][pos].astype(np.int64), kept_len=kept_len, runmax=runmax,
                    first_alarm=first)


def load_side(files, key: str, test_set: str = "T0", demotions=(), alpha: float = ALPHA) -> list[SeedEval]:
    return [load_seed(f, key, test_set, demotions, alpha) for f in files]


def check_same_test_set(a: list[SeedEval], b: list[SeedEval]) -> None:
    if len(a) != len(b):
        raise SystemExit(f"different number of seeds: {len(a)} vs {len(b)}")
    for x, y in zip(a, b):
        if x.seed != y.seed:
            raise SystemExit(f"seed order differs: {x.path} vs {y.path}")
        if not (np.array_equal(x.rids, y.rids) and np.array_equal(x.y, y.y)
                and np.array_equal(x.kept_len, y.kept_len) and np.array_equal(x.cell, y.cell)):
            raise SystemExit(f"test sets differ between {x.path} and {y.path}")


# --- Resamples --------------------------------------------------------------------------
def cells_of(side: SeedEval) -> list[tuple[int, np.ndarray]]:
    return [(int(c), np.nonzero(side.cell == c)[0]) for c in np.unique(side.cell)]


def make_draws(sides: list[SeedEval], n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> list[list]:
    """draws[k] (seed k) = [(members (n_c,), I (n_boot, n_c) local indices)] per cell."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for side in sides:
        out.append([(mem, rng.integers(0, len(mem), size=(n_boot, len(mem))))
                    for _, mem in cells_of(side)])
    return out


def identity_draws(sides: list[SeedEval]) -> list[list]:
    return [[(mem, np.arange(len(mem))[None, :]) for _, mem in cells_of(side)] for side in sides]


# --- Metrics on resamples ---------------------------------------------------------------
def boot_metrics(side: SeedEval, cell_draws: list) -> dict:
    """{metric: (n_boot,)} for one seed, recomputed on every resample of `cell_draws`."""
    n_boot = cell_draws[0][1].shape[0]
    aucs, invalid = [], np.zeros(n_boot, bool)
    tp = nf = fp = ns = np.zeros(n_boot)
    for mem, I in cell_draws:
        P = mem[I]                                            # (n_boot, n_c) positions
        cut = side.kept_len[P].min(1)                         # shortest kept length in the cell
        top = side.runmax[P, (cut - 1)[:, None]]              # max over kept rows t < cut
        invalid |= np.isinf(top).any(1)
        yb = side.y[P]
        n1 = yb.sum(1)
        n0 = yb.shape[1] - n1
        ok = np.minimum(n0, n1) >= MIN_PER_CLASS
        r = rankdata(np.where(np.isinf(top), -1e300, top), axis=1, method="average")
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = ((r * yb).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        aucs.append(np.where(ok, auc, np.nan))
        fired = side.first_alarm[P] < cut[:, None]
        tp = tp + (fired & (yb == 1)).sum(1)
        nf = nf + n1
        fp = fp + (fired & (yb == 0)).sum(1)
        ns = ns + n0
    A = np.stack(aucs, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.isfinite(A).sum(1)
        m1pt = np.where(q > 0, np.nansum(np.nan_to_num(A, nan=0.0), 1) / np.maximum(q, 1), np.nan)
        tpr = tp / nf
        tnr = 1 - fp / ns
    m1pt = np.where(invalid, np.nan, m1pt)
    return {"m1_per_task": m1pt, "bal_acc": (tpr + tnr) / 2, "tpr": tpr, "n_cells": q}


def side_boot(sides: list[SeedEval], draws: list) -> tuple[dict, dict]:
    """Seed means {metric: (n_boot,)} and per-seed {metric: (n_seeds, n_boot)}."""
    per = [boot_metrics(s, d) for s, d in zip(sides, draws)]
    stacked = {m: np.stack([p[m] for p in per]) for m in METRICS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns -> NaN
        means = {m: np.nanmean(v, 0) for m, v in stacked.items()}
    stacked["n_cells"] = np.stack([p["n_cells"] for p in per])
    return means, stacked


def identity_metrics(sides: list[SeedEval]) -> dict:
    """Per-seed values on the original sample: {metric: [v_seed...]} plus n_cells."""
    _, st = side_boot(sides, identity_draws(sides))
    out = {m: st[m][:, 0].tolist() for m in METRICS}
    out["n_cells"] = st["n_cells"][:, 0].astype(int).tolist()
    return out


def _mean(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


# --- The paired comparison --------------------------------------------------------------
def verdict(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if lo > 0:
        return "improvement"
    if hi < 0:
        return "significant worsening"
    return "no significant change"


def compare(x: list[SeedEval], y: list[SeedEval], draws: list, *, test_set: str, n_boot: int,
            rng_seed: int, comparison: str | None = None, x_boot=None, y_boot=None) -> dict:
    check_same_test_set(x, y)
    xb, xs = x_boot or side_boot(x, draws)
    yb, ys = y_boot or side_boot(y, draws)
    xi, yi = identity_metrics(x), identity_metrics(y)
    cx, cy = x[0].cond, y[0].cond
    kx, ky = x[0].key, y[0].key
    comp = comparison or (f"{cx} - {cy}" if kx == ky else f"{cx}:{kx} - {cy}:{ky}")
    head = {"test_set": test_set, "comparison": comp, "cond_x": cx, "cond_y": cy,
            "detector": kx if kx == ky else f"{kx} vs {ky}", "detector_x": kx, "detector_y": ky}
    rows, per_seed = [], []
    for m in METRICS:
        d = xb[m] - yb[m]
        ok = np.isfinite(d)
        lo, hi = (np.percentile(d[ok], CI) if ok.any() else (np.nan, np.nan))
        xp, yp = _mean(xi[m]), _mean(yi[m])
        rows.append({**head, "metric": m, "n_seeds": len(x), "n_boot": n_boot, "n_valid": int(ok.sum()),
                     "rng_seed": rng_seed, "x_point": xp, "y_point": yp, "diff_point": xp - yp,
                     "diff_mean": float(d[ok].mean()) if ok.any() else np.nan,
                     "diff_ci_lo": float(lo), "diff_ci_hi": float(hi),
                     "x_boot_mean": _mean(xb[m]), "y_boot_mean": _mean(yb[m]),
                     "improvement": bool(np.isfinite(lo) and lo > 0),
                     "significant_worsening": bool(np.isfinite(hi) and hi < 0),
                     "verdict": verdict(lo, hi),
                     "x_point_per_seed": " ".join(f"{v:.6g}" for v in xi[m]),
                     "y_point_per_seed": " ".join(f"{v:.6g}" for v in yi[m])})
        for k, (sx, sy) in enumerate(zip(x, y)):
            ds = xs[m][k] - ys[m][k]
            okk = np.isfinite(ds)
            lo_s, hi_s = (np.percentile(ds[okk], CI) if okk.any() else (np.nan, np.nan))
            per_seed.append({**head, "seed": sx.seed, "metric": m, "n_test": len(sx.rids),
                             "n_cells_x": xi["n_cells"][k], "n_cells_y": yi["n_cells"][k],
                             "x_point": xi[m][k], "y_point": yi[m][k], "diff_point": xi[m][k] - yi[m][k],
                             "diff_mean": float(ds[okk].mean()) if okk.any() else np.nan,
                             "diff_ci_lo": float(lo_s), "diff_ci_hi": float(hi_s),
                             "n_valid": int(okk.sum()), "verdict": verdict(lo_s, hi_s)})
    return {"rows": rows, "per_seed": per_seed, "identity": {"x": xi, "y": yi}}


def paired_bootstrap(files_x, key_x: str, files_y, key_y: str, test_set: str = "T0",
                     n_boot: int = N_BOOT, rng_seed: int = RNG_SEED, alpha: float = ALPHA,
                     demotions=(), comparison: str | None = None) -> dict:
    """Generic paired bootstrap of x - y (see the module docstring)."""
    x = load_side(files_x, key_x, test_set, demotions, alpha)
    y = load_side(files_y, key_y, test_set, demotions, alpha)
    check_same_test_set(x, y)
    draws = make_draws(y, n_boot, rng_seed)
    return compare(x, y, draws, test_set=test_set, n_boot=n_boot, rng_seed=rng_seed,
                   comparison=comparison)


# --- Identity check against the pipeline's tables ----------------------------------------
def check_identity(side: list[SeedEval], test_set: str, table_dir: Path, check_table: Path,
                   detector_name: str | None = None) -> dict:
    """Compare identity-resample values with final_m1.csv / cp_curves.csv (per seed, if in
    table_dir) and check_table (seed means; final_table.csv format)."""
    det = detector_name or side[0].key
    cond = side[0].cond
    vals = identity_metrics(side)
    res = {"cond": cond, "detector": det, "test_set": test_set, "comparisons": []}

    def add(what, got, ref):
        if ref is None or ref == "":
            return
        ref = float(ref)
        diff = abs(got - ref) if np.isfinite(got) and np.isfinite(ref) else (
            0.0 if (np.isnan(got) and np.isnan(ref)) else float("inf"))
        res["comparisons"].append({"what": what, "got": got, "ref": ref, "abs_diff": diff})

    m1 = read_csv(table_dir / "final_m1.csv")
    cp = read_csv(table_dir / "cp_curves.csv")
    for k, s in enumerate(side):
        r = [q for q in m1 if q["cond"] == cond and q["detector"] == det and q["test_set"] == test_set
             and int(q["seed"]) == s.seed]
        if r:
            add(f"seed {s.seed} m1_per_task", vals["m1_per_task"][k], r[0]["m1_per_task"])
            add(f"seed {s.seed} n_cells", float(vals["n_cells"][k]), r[0]["n_cells"])
        r = [q for q in cp if q["cond"] == cond and q["detector"] == det and q["test_set"] == test_set
             and int(q["seed"]) == s.seed and q["on_alpha_grid"] == "False"
             and np.isclose(float(q["alpha"]), ALPHA)]
        if r:
            add(f"seed {s.seed} bal_acc", vals["bal_acc"][k], r[0]["bal_acc"])
            add(f"seed {s.seed} tpr", vals["tpr"][k], r[0]["tpr"])
    ft = [q for q in read_csv(check_table) if q.get("cond") == cond and q.get("detector") == det
          and q.get("test_set") == test_set]
    if ft:
        add("mean m1_per_task", _mean(vals["m1_per_task"]), ft[0]["m1_per_task_mean"])
        add("mean bal_acc", _mean(vals["bal_acc"]), ft[0]["a01_bal_acc_mean"])
        add("mean tpr", _mean(vals["tpr"]), ft[0]["a01_tpr_mean"])
    diffs = [c["abs_diff"] for c in res["comparisons"]]
    res["n_compared"] = len(diffs)
    res["max_abs_diff"] = max(diffs) if diffs else None
    res["passed"] = bool(diffs) and max(diffs) <= CHECK_TOL
    if not ft:
        res["note"] = f"no row for ({cond}, {det}, {test_set}) in {check_table}"
    return res


# --- CLI --------------------------------------------------------------------------------
def parse_spec(spec: str, seeds) -> tuple[list[Path], str]:
    tmpl, key = spec.rsplit(":", 1)
    return [Path(tmpl.format(seed=s)) for s in seeds], key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", type=Path, default=data2.RESULTS_DIR,
                    help="pipeline output dir (scores/, final_table.csv, final_m1.csv, cp_curves.csv)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --in-dir")
    ap.add_argument("--test-set", nargs="+", default=["T0"], choices=["T0", "T1", "T2"])
    ap.add_argument("--pairs", nargs="+", default=["D1:D0", "D2:D0"], help="X:Y condition pairs (X - Y)")
    ap.add_argument("--detectors", nargs="+", default=["A", "C", "emb"])
    ap.add_argument("--pair", nargs=2, metavar=("X_SPEC", "Y_SPEC"), default=None,
                    help="generic comparison instead of --pairs/--detectors: 'path{seed}.npz:KEY' each")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(data2.SEEDS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--demotions", type=Path, default=None, help="default: <in-dir>/demotions.json")
    ap.add_argument("--check-table", type=Path, default=None, help="default: <in-dir>/final_table.csv")
    ap.add_argument("--out-name", default=None, help="default paired_<test>.csv")
    ap.add_argument("--strict", action="store_true", help="exit 1 if the identity check fails")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dem = data2.demoted(args.demotions or args.in_dir / "demotions.json")
    check_table = args.check_table or args.in_dir / "final_table.csv"
    scores = args.in_dir / "scores"
    all_ok = True

    for ts in args.test_set:
        t0 = time.time()
        if args.pair:
            fx, kx = parse_spec(args.pair[0], args.seeds)
            fy, ky = parse_spec(args.pair[1], args.seeds)
            jobs = [(fx, kx, fy, ky)]
        else:
            jobs = []
            for pr in args.pairs:
                cx, cy = pr.split(":")
                for d in args.detectors:
                    jobs.append(([scores / f"{cx}_seed{s}.npz" for s in args.seeds], d,
                                 [scores / f"{cy}_seed{s}.npz" for s in args.seeds], d))
            have = [j for j in jobs if all(Path(f).exists() for f in j[0] + j[2])]
            for j in jobs:
                if j not in have:
                    print(f"[skip] missing npz for {j[0][0].name} / {j[2][0].name} ({j[1]})")
            jobs = have
        if not jobs:
            print(f"[{ts}] nothing to compare")
            continue

        sides, boots = {}, {}
        for fx, kx, fy, ky in jobs:
            for f, k in ((fx, kx), (fy, ky)):
                key = (tuple(map(str, f)), k)
                if key not in sides:
                    sides[key] = load_side(f, k, ts, dem)
        first = next(iter(sides.values()))
        for s in sides.values():
            check_same_test_set(first, s)
        draws = make_draws(first, args.n_boot, args.rng_seed)
        t_load = time.time() - t0
        for key, s in sides.items():
            boots[key] = side_boot(s, draws)

        rows, per_seed = [], []
        for fx, kx, fy, ky in jobs:
            kx_, ky_ = (tuple(map(str, fx)), kx), (tuple(map(str, fy)), ky)
            r = compare(sides[kx_], sides[ky_], draws, test_set=ts, n_boot=args.n_boot,
                        rng_seed=args.rng_seed, x_boot=boots[kx_], y_boot=boots[ky_])
            rows += r["rows"]
            per_seed += r["per_seed"]

        checks = [check_identity(s, ts, args.in_dir, check_table) for s in sides.values()]
        passed = all(c["passed"] for c in checks)
        all_ok &= passed
        name = args.out_name or f"paired_{ts}.csv"
        if args.out_name and len(args.test_set) > 1:
            name = name.replace(".csv", f"_{ts}.csv")
        write_csv(out_dir / name, rows)
        write_csv(out_dir / name.replace(".csv", "_per_seed.csv"), per_seed)
        (out_dir / name.replace(".csv", "_check.json")).write_text(json.dumps(
            {"test_set": ts, "passed": passed, "tolerance": CHECK_TOL, "rng_seed": args.rng_seed,
             "n_boot": args.n_boot, "check_table": str(check_table),
             "max_abs_diff": max((c["max_abs_diff"] or 0) for c in checks), "sides": checks},
            indent=1, default=float))
        dt = time.time() - t0
        print(f"[{ts}] {len(jobs)} comparisons x {len(METRICS)} metrics, {args.n_boot} reps, "
              f"{len(sides)} sides: {dt:.1f} s (loading {t_load:.1f} s) -> {out_dir / name}")
        print(f"[{ts}] identity check {'PASSED' if passed else 'FAILED'}: "
              + "; ".join(f"{c['cond']}/{c['detector']} n={c['n_compared']} max|d|={c['max_abs_diff']}"
                          for c in checks))
        for r in rows:
            print(f"    {r['comparison']:9s} {r['detector']:5s} {r['metric']:12s} "
                  f"point {r['diff_point']:+.3f}  mean {r['diff_mean']:+.3f} "
                  f"[{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]  {r['verdict']}")
    if not all_ok:
        print("[!] IDENTITY CHECK FAILED: the recomputation does not reproduce the pipeline's numbers")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
