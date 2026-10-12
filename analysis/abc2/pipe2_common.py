"""
pipe2_common.py

Shared plumbing for the more-data pipeline port (more_data_collection.md Steps 8-9):
the run context (paths, index, kept mask, splits, a condition's train / eval_seen /
calibration sets and the test sets), constraint-series loading for the combined corpus,
a fault-tolerant M1 wrapper, and CSV / pickle helpers. Used by stage2_constraints.py,
stage3_c.py, stage4_a.py, stage6_cp.py and pipeline.py (the entrypoint; see its
docstring for usage and the output formats).

Frozen definitions are imported from analysis/abc (prep, data via data2); only what
hard-codes the 300-rollout corpus is copied here:
  * load_series2         = prep.load_series, but reads idx.constraints_path(i)
                           (constraints/ for v2, constraints_v3/ for v3)
  * baseline_series2     = constraint_m1.baseline_series, reads idx.rollout_dir(i)
  * failurespot_series2  = constraint_m1.failurespot_series, reads idx.rollout_dir(i)
  * load_split_from      = data2.load_split2 with the splits directory as a parameter
                           (data2.load_split2 reads the fixed results/abc2/splits/)
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data2  # noqa: E402  (puts analysis/abc and analysis/ on sys.path)
from data2 import SEEDS, T, Index2, kept_mask, load_index2, m1  # noqa: E402,F401

import prep  # noqa: E402  (analysis/abc/prep.py, frozen)
from constraint_auroc import ACT_REP_EPS  # noqa: E402

# Stages 3-6 never used xl_final_hN or the _nosink variants (Step 6 does not recompute
# them for v3), so the port's Stage 2 drops them. Neither enters any selection: beta is
# chosen over C's series only, the emb_temp layer over emb_temp only.
PRIMARY2 = [(b, lay) for b, lay in prep.PRIMARY if b != "xl_final_hN"]
EVAL_SETS = ("eval_seen", "T0", "T1", "T2")
STRESS_SETS = ("stress_seen", "stress_unseen")
DETECTORS = ("A", "C", "emb")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Splits ------------------------------------------------------------------------------
def load_split_from(splits_dir: Path, seed: int, idx: Index2) -> dict:
    """data2.load_split2 with a configurable directory: the saved split, positions under
    `pos`; rollout ids not in `idx` are dropped."""
    split = json.loads((Path(splits_dir) / f"seed{seed}.json").read_text())
    have = set(idx.rids.tolist())
    split["pos"] = {name: idx.pos([r for r in split.get(name, []) if r in have])
                    for name in ("train", "eval_seen", "unseen", "stress")}
    return split


# --- Context -----------------------------------------------------------------------------
@dataclass
class Ctx:
    index_path: Path
    cache_dir: Path
    splits_dir: Path
    out_dir: Path
    work_dir: Path
    demotions_path: Path
    seeds: tuple
    n_jobs: int = 4
    idx: Index2 = field(init=False)
    kept: np.ndarray = field(init=False)
    demotions: tuple = field(init=False)
    _splits: dict = field(default_factory=dict, init=False)
    _series: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.idx = load_index2(self.index_path)
        self.kept = kept_mask(self.idx)
        self.demotions = data2.demoted(self.demotions_path)
        for d in (self.out_dir, self.work_dir, self.tables_dir, self.scores_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def tables_dir(self) -> Path:
        return self.out_dir / "tables"

    @property
    def scores_dir(self) -> Path:
        return self.out_dir / "scores"

    def wpath(self, cond: str, name: str) -> Path:
        d = self.work_dir / cond
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def split(self, seed: int) -> dict:
        if seed not in self._splits:
            self._splits[seed] = load_split_from(self.splits_dir, seed, self.idx)
        return self._splits[seed]

    def sets(self, cond: str, seed: int) -> dict:
        """Positions: train, eval_seen, cal (condition_sets) and T0, T1, T2,
        stress_seen, stress_unseen (test_sets)."""
        sp = self.split(seed)
        out = dict(data2.condition_sets(sp, self.idx, cond, self.demotions))
        out.update(data2.test_sets(sp, self.idx, self.demotions))
        return out

    def series(self, spec_key: str):
        """Raw series cached per process: 'primary' (PRIMARY2) or 'baseline'."""
        if spec_key not in self._series:
            t0 = time.time()
            if spec_key == "primary":
                self._series[spec_key] = load_series2(self.idx, PRIMARY2)
            elif spec_key == "baseline":
                self._series[spec_key] = baseline_series2(self.idx)
            elif spec_key == "failurespot":
                self._series[spec_key] = failurespot_series2(self.idx)
            else:
                raise KeyError(spec_key)
            log(f"loaded {spec_key} series for {self.idx.n} rollouts ({time.time() - t0:.0f} s)")
        return self._series[spec_key]


# --- Series (copies of prep.load_series / constraint_m1 baselines, per-rollout paths) ----
def load_series2(idx: Index2, spec) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """prep.load_series for an Index2: raw series -> (n_series, N, 520) float64."""
    names: list[tuple[str, int]] = []
    first = np.load(idx.constraints_path(0))
    for base, layered in spec:
        if base == "act_rep":
            names.append((base, -1))
            continue
        width = first[base].shape[1] if layered else None
        names += [(base, L) for L in range(width)] if layered else [(base, -1)]

    S = np.empty((len(names), idx.n, T), dtype=np.float64)
    for i in range(idx.n):
        z = np.load(idx.constraints_path(i))
        cache = {}
        for k, (base, L) in enumerate(names):
            if base == "act_rep":
                mag = z["act_mag"]
                S[k, i] = np.where(np.isnan(mag), np.nan, (mag <= ACT_REP_EPS).astype(float))
                continue
            if base not in cache:
                cache[base] = z[base]
            arr = cache[base]
            if arr.shape[0] != T:
                raise SystemExit(f"{idx.rids[i]}:{base} has {arr.shape[0]} rows")
            S[k, i] = arr[:, L] if L >= 0 else arr
    return S, names


def baseline_series2(idx: Index2):
    """constraint_m1.baseline_series for an Index2 (logits.pt per rollout)."""
    import torch
    names = [("token_prob_mean", -1), ("token_prob_max", -1),
             ("token_entropy_mean", -1), ("token_entropy_max", -1), ("timestep", -1)]
    S = np.empty((len(names), idx.n, T))
    for i in range(idx.n):
        lg = torch.load(str(idx.rollout_dir(i) / "logits.pt"), map_location="cpu").float()
        p = torch.softmax(lg, -1)
        sel = p.max(-1).values
        ent = -(p * torch.log_softmax(lg, -1)).sum(-1)
        S[0, i] = sel.mean(-1).numpy()
        S[1, i] = sel.max(-1).values.numpy()
        S[2, i] = ent.mean(-1).numpy()
        S[3, i] = ent.max(-1).values.numpy()
        S[4, i] = np.arange(T)
    return S, names


FAILURESPOT_BETA = 0.8


def failurespot_series2(idx: Index2) -> np.ndarray:
    """constraint_m1.failurespot_series for an Index2. (N, 520); NaN at t = 0 only."""
    from compute_constraints import normalised_actions
    out = np.empty((idx.n, T))
    for i in range(idx.n):
        tok = np.load(idx.rollout_dir(i) / "actions.npz")["action_token_ids"]
        a = normalised_actions(tok).astype(np.float64)
        c = np.full(T, np.nan)
        c[1:] = np.linalg.norm(a[1:] - a[:-1], axis=1)
        r = np.linalg.norm(a, axis=1)
        out[i] = c + prep.ema(c, FAILURESPOT_BETA) + r + prep.ema(r, FAILURESPOT_BETA)
    return out


# --- M1 ----------------------------------------------------------------------------------
def m1_safe(score, kept, idx, pos) -> dict:
    """data.m1 plus n / n_fail; never raises. If the stacked call fails (a rollout with no
    valid row under its cut, or no cell with 2 of each class), series are scored one by
    one and the failing ones get NaN; `error` says why."""
    base = {"n": int(len(pos)), "n_fail": int(idx.y[pos].sum()) if len(pos) else 0}
    if len(pos) == 0:
        return {**base, "pooled": np.nan, "per_task": np.nan, "floor": np.nan, "n_tasks": 0,
                "error": "empty set"}
    try:
        return {**base, **m1(score, kept, idx, pos), "error": ""}
    except ValueError as e:
        score = np.asarray(score)
        if score.ndim == 2:
            return {**base, "pooled": np.nan, "per_task": np.nan, "floor": np.nan, "n_tasks": 0,
                    "error": str(e)}
        res = [m1_safe(score[k], kept, idx, pos) for k in range(score.shape[0])]
        ok = [r for r in res if not r["error"]]
        return {**base, "pooled": np.array([r["pooled"] for r in res]),
                "per_task": np.array([r["per_task"] for r in res]),
                "floor": ok[0]["floor"] if ok else np.nan,
                "n_tasks": ok[0]["n_tasks"] if ok else 0,
                "error": f"{len(res) - len(ok)} series failed: {e}"}


def m1_row(m: dict, k: int | None = None) -> dict:
    """Flatten an m1_safe dict (element k for stacked input) into table columns."""
    pick = (lambda v: float(np.asarray(v)[k])) if k is not None else float
    return {"m1_pooled": pick(m["pooled"]), "m1_per_task": pick(m["per_task"]),
            "floor": float(m["floor"]), "n_cells": int(m["n_tasks"]), "n": m["n"],
            "n_fail": m["n_fail"]}


# --- IO ----------------------------------------------------------------------------------
def save_pkl(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_pkl(path: Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _fmt(k, v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:g}" if k in ("c", "alpha", "beta") else f"{v:.6g}"
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(k, v) for k, v in r.items()})
    tmp.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    return list(csv.DictReader(Path(path).open()))


def mean_std(v) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def record_timing(ctx: Ctx, stage: str, cond: str, seconds: float, note: str = "") -> None:
    path = ctx.out_dir / "timings.csv"
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("utc,stage,cond,seeds,n_rollouts,seconds,note\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},{stage},{cond},"
                f"{' '.join(map(str, ctx.seeds))},{ctx.idx.n},{seconds:.1f},{note}\n")
