"""
prep.py

Constraint series and their preparation (handoff section 2), shared by Stages 2, 3 and 5.

Series are held as one stacked array S of shape (n_series, 300, 520) with a parallel
list of names (base name, layer), layer = -1 for scalar series. Rollout order is
data.load_index()'s.

Preparation, in order:
  1. Forward-fill NaN within each rollout, every series alike (act_dir included: no
     zeros). Leading NaN stays NaN.
  2. Causal EMA, E[t] = beta * E[t-1] + (1 - beta) * x[t], initialised E = x at the
     first valid step. beta = 0 ("none") leaves the series unsmoothed.
  3. Scale by mean and std of successful TRAIN rows (NaN ignored).
  4. Orient: sign -1 if the scaled series' per-task M1 on the TRAIN split is below 0.5.
     A constraint whose sign differs across seeds is flagged unstable downstream.
  Rows before a series' first valid value stay NaN: m1 leaves them out of the maximum,
  and model_input() gives them 0 (neutral, the train-success mean) as model inputs.
Steps 1-2 depend only on beta; steps 3-4 are refitted per seed on that seed's train
split, using data.train_kept: kept train rows with t <= the >= 5-successes cut.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))   # analysis/, for frozen modules
from constraint_auroc import ACT_REP_EPS                    # noqa: E402

from data import CONSTRAINTS_DIR, T, Index, m1, train_kept  # noqa: E402

BETAS = (0.0, 0.83, 0.91, 0.96)          # 0.0 = no smoothing


def beta_label(beta: float) -> str:
    return "none" if beta == 0 else f"{beta:.2f}"

# (base name, layered) in table order. Families per spec section 2.
PRIMARY = [("act_mag", False), ("act_dir", False), ("grip_flip", False), ("act_rep", False),
           ("emb_temp", True),
           ("xl_adj", True), ("xl_final", True), ("xl_spread", False), ("xl_final_hN", True)]
NOSINK = [("emb_temp_nosink", True), ("xl_adj_nosink", True), ("xl_final_nosink", True),
          ("xl_spread_nosink", False), ("xl_final_hN_nosink", True)]
FAMILY = {"act_mag": "action", "act_dir": "action", "grip_flip": "action",
          "act_rep": "action", "emb_temp": "hidden_time", "xl_adj": "hidden_layers",
          "xl_final": "hidden_layers", "xl_spread": "hidden_layers",
          "xl_final_hN": "hidden_layers"}


def family(name: str) -> str:
    return FAMILY.get(name.removesuffix("_nosink"), "other")


def load_series(idx: Index, spec=PRIMARY) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Raw constraint series from constraints/<rid>.npz -> (n_series, 300, 520) float64."""
    names: list[tuple[str, int]] = []
    for base, layered in spec:
        if base == "act_rep":
            names.append((base, -1))
            continue
        width = None
        if layered:
            width = np.load(CONSTRAINTS_DIR / f"{idx.rids[0]}.npz")[base].shape[1]
        names += [(base, L) for L in range(width)] if layered else [(base, -1)]

    S = np.empty((len(names), idx.n, T), dtype=np.float64)
    for i, rid in enumerate(idx.rids):
        z = np.load(CONSTRAINTS_DIR / f"{rid}.npz")
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
                raise SystemExit(f"{rid}:{base} has {arr.shape[0]} rows")
            S[k, i] = arr[:, L] if L >= 0 else arr
    return S, names


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along the last axis. Leading NaN stays NaN."""
    valid = ~np.isnan(x)
    j = np.where(valid, np.arange(x.shape[-1]), 0)
    np.maximum.accumulate(j, axis=-1, out=j)
    out = np.take_along_axis(x, j, axis=-1)
    out[~np.take_along_axis(valid, j, axis=-1)] = np.nan
    return out


def ema(x: np.ndarray, beta: float) -> np.ndarray:
    """Causal EMA along the last axis, started (E = x) at the first non-NaN step.
    Expects NaN only as a leading run, as ffill leaves it."""
    if beta == 0:
        return x.copy()
    out = np.empty_like(x)
    prev = np.full(x.shape[:-1], np.nan)
    for t in range(x.shape[-1]):
        cur = x[..., t]
        prev = np.where(np.isnan(prev), cur, beta * prev + (1 - beta) * cur)
        out[..., t] = prev
    return out


def smooth(S: np.ndarray, names: list[tuple[str, int]], beta: float) -> np.ndarray:
    """Steps 1-2: fill then EMA. Depends on beta only, not on the split."""
    return ema(ffill(S), beta)


def fit_scale_orient(E: np.ndarray, idx: Index, kept: np.ndarray, train_pos: np.ndarray):
    """Steps 3-4 fitted on train. Returns (mu, sd, sign, train_m1_task), each
    (n_series,); train_m1_task is the unoriented per-task train M1 behind the sign."""
    tk = train_kept(idx, kept, train_pos)
    y = idx.y[train_pos]
    succ = train_pos[y == 0]
    vs = E[:, succ][:, tk[succ]]                       # (n_series, n_succ_rows)
    mu = np.nanmean(vs, 1)
    sd = np.nanstd(vs, 1)
    sd = np.where(sd > 0, sd, 1.0)
    ones = np.ones_like(mu)
    train_m1 = m1(apply_prep(E, mu, sd, ones), tk, idx, train_pos)["per_task"]
    sign = np.where(train_m1 < 0.5, -1.0, 1.0)
    return mu, sd, sign, train_m1


def apply_prep(E: np.ndarray, mu, sd, sign) -> np.ndarray:
    """Scale and orient. Missing rows stay NaN, which m1 leaves out of the maximum;
    use model_input() before feeding a model."""
    return ((E - mu[:, None, None]) / sd[:, None, None]) * sign[:, None, None]


def model_input(P: np.ndarray) -> np.ndarray:
    """Missing rows -> 0, the neutral value (the train-success mean after scaling)."""
    return np.nan_to_num(P, nan=0.0)
