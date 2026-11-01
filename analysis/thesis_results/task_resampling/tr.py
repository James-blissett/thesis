"""
tr.py

Shared plumbing for the task-resampling run (task_resampling_handoff.md, version 4):
the ranges engine of §1 and the paths every script here uses.

One round, for each task split (repo: seed) separately:
  1. draw 3 tasks with replacement from the split's 3 unseen tasks (a task drawn twice
     counts twice);
  2. for each drawn copy of a task, every task-and-perturbation group (repo: cell) of that
     task, its rollouts redrawn with replacement (each copy gets its own draw);
  3. the metric on the drawn set, computed exactly as analysis/abc2/bootstrap2.boot_metrics
     does for one group (cut = shortest kept length among the drawn rollouts; rollout score =
     running maximum of the M1 score at the cut; alarm = first stored-band crossing before
     the cut, the split's band held fixed):
       per-task AUROC   mean over the drawn group copies with >= 2 failures and >= 2
                        successes (a group drawn twice counts twice; others are dropped)
       balanced acc.    TPR over the drawn failures, TNR over the drawn successes, pooled
                        over the drawn group copies
       pooled AUROC     one AUROC over every drawn rollout (copies included), each scored
                        under its own group's cut (data.m1's pooled M1)
Then the mean over the three splits; a round where any split's value is undefined is
skipped (so every recorded value averages three splits). A comparison x - y uses the same
draws for both sides.

Implementation: every group gets up to 3 independent rollout draws per round (one per
possible copy); copy j of a group counts in a round only if its task was drawn more than
j times. With step 1 switched off every task counts once (copy 0 only), which is the
existing rollouts-only bootstrap; `plan_from_b2` reuses bootstrap2's own draws, so that
case can be checked to reproduce tables/paired_comparisons.csv exactly.

Rollout and test-set handling is bootstrap2's (load_side / test_positions; T2 = the
unseen tasks incl. LIBERO-PRO, T0 = their original rollouts). The task of a group is
cell % 10 (= base_task, checked).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

os.environ.setdefault("THESIS_POOL", "D2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "final"))
import fc  # noqa: E402
import bootstrap2 as b2  # noqa: E402
from fc import REPO, SEEDS, common, log, read_csv, write_csv  # noqa: E402,F401

OUT = fc.OUT / "task_resampling"
WORK = fc.WORK_ROOT / "task_resampling"
N_ROUNDS = 2000
RNG_SEED = 20261101
CI = (2.5, 97.5)
MIN_PER_CLASS = b2.MIN_PER_CLASS

# Saved per-split npz files (bootstrap2 format) of the earlier runs.
NPZ_FULL = [fc.WORK_ROOT / "full" / "npz" / f"seed{s}.npz" for s in SEEDS]
NPZ_ORIG = [fc.WORK_ROOT / "original" / "npz" / f"seed{s}.npz" for s in SEEDS]
NPZ_COMMON_FULL = [fc.WORK_ROOT / "final" / "full" / "npz" / "check2" / f"seed{s}.npz" for s in SEEDS]
NPZ_COMMON_ORIG = [fc.WORK_ROOT / "final" / "original" / "npz" / "check2" / f"seed{s}.npz" for s in SEEDS]
NPZ_SMOOTH = [fc.WORK_ROOT / "final" / "full" / "npz" / "smoothing" / f"seed{s}.npz" for s in SEEDS]
NPZ_LAYER0 = [WORK / "npz" / "layer0" / f"seed{s}.npz" for s in SEEDS]

FAMILY = {0: "unperturbed", 1: "object perturbation", 2: "instruction paraphrase"}


# --- sides ------------------------------------------------------------------------------
def load_side(files, key: str, test_set: str = "T2", family: int | None = None) -> list:
    """bootstrap2 SeedEvals per split; family restricts to one perturbation type."""
    sd = b2.load_side(files, key.replace(":", "__"), test_set)
    for s in sd:
        z = np.load(s.path)
        base = dict(zip(z["rollout_ids"].astype(str), z["base_task"]))
        if not np.array_equal(s.cell % 10, np.asarray([base[r] for r in s.rids])):
            raise SystemExit(f"{s.path}: cell % 10 is not the rollout's base task")
    if family is not None:
        sd = [fc.restrict(s, s.cell // 10 == family) for s in sd]
    return sd


# --- draws ------------------------------------------------------------------------------
@dataclass
class SplitPlan:
    tasks: np.ndarray          # (n_tasks,) task ids
    cells: list                # [(cell id, members (n_c,) local positions, task index)]
    mult: np.ndarray           # (R, n_tasks) times each task is drawn
    draws: list                # per cell: (copies, R, n_c) local indices


def make_plan(sides: list, n_rounds: int = N_ROUNDS, rng_seed: int = RNG_SEED, resample_tasks: bool = True) -> list:
    """Draws for every split, in split order: the task draw first, then each group's
    rollout draws (ascending group id), all from one Generator."""
    rng = np.random.default_rng(rng_seed)
    plans = []
    for side in sides:
        tasks = np.unique(side.cell % 10)
        nt = len(tasks)
        if resample_tasks:
            pick = rng.integers(0, nt, size=(n_rounds, nt))
            mult = np.stack([(pick == k).sum(1) for k in range(nt)], 1)
            copies = nt
        else:
            mult = np.ones((n_rounds, nt), np.int64)
            copies = 1
        cells, draws = [], []
        for c in np.unique(side.cell):
            mem = np.nonzero(side.cell == c)[0]
            cells.append((int(c), mem, int(np.searchsorted(tasks, c % 10))))
            draws.append(rng.integers(0, len(mem), size=(copies, n_rounds, len(mem))))
        plans.append(SplitPlan(tasks, cells, mult, draws))
    return plans


def plan_from_b2(sides: list, n_boot: int = b2.N_BOOT, rng_seed: int = b2.RNG_SEED) -> list:
    """Step 1 off, with bootstrap2's own rollout draws (exact reproduction check)."""
    plans = []
    for side, cd in zip(sides, b2.make_draws(sides, n_boot, rng_seed)):
        tasks = np.unique(side.cell % 10)
        cells = [(int(side.cell[mem[0]]), mem, int(np.searchsorted(tasks, side.cell[mem[0]] % 10))) for mem, _ in cd]
        plans.append(SplitPlan(tasks, cells, np.ones((n_boot, len(tasks)), np.int64), [I[None] for _, I in cd]))
    return plans


def identity_plan(sides: list) -> list:
    """The original sample: every task once, every rollout once."""
    plans = []
    for side in sides:
        tasks = np.unique(side.cell % 10)
        cells, draws = [], []
        for c in np.unique(side.cell):
            mem = np.nonzero(side.cell == c)[0]
            cells.append((int(c), mem, int(np.searchsorted(tasks, c % 10))))
            draws.append(np.arange(len(mem))[None, None, :])
        plans.append(SplitPlan(tasks, cells, np.ones((1, len(tasks)), np.int64), draws))
    return plans


# --- metrics ----------------------------------------------------------------------------
def split_metrics(side, plan: SplitPlan, pooled: bool = False) -> dict:
    """{metric: (R,)} for one split on every round of `plan`, plus drop counts."""
    R = plan.mult.shape[0]
    num, den, dropped, drawn = np.zeros(R), np.zeros(R), np.zeros(R), np.zeros(R)
    tp, nf, fp, ns = np.zeros(R), np.zeros(R), np.zeros(R), np.zeros(R)
    invalid = np.zeros(R, bool)
    P_tops, P_ys, P_w = [], [], []
    for (_, mem, ti), I in zip(plan.cells, plan.draws):
        for j in range(I.shape[0]):
            w = plan.mult[:, ti] > j                                  # this copy is in the round
            if not w.any():
                continue
            P = mem[I[j]]                                             # (R, n_c) positions
            cut = side.kept_len[P].min(1)
            top = side.runmax[P, (cut - 1)[:, None]]
            yb = side.y[P]
            n1 = yb.sum(1)
            n0 = yb.shape[1] - n1
            ok = np.minimum(n0, n1) >= MIN_PER_CLASS
            r = rankdata(np.where(np.isinf(top), -1e300, top), axis=1, method="average")
            with np.errstate(invalid="ignore", divide="ignore"):
                auc = ((r * yb).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
            num += np.where(w & ok, np.nan_to_num(auc), 0.0)
            den += w & ok
            dropped += w & ~ok
            drawn += w
            invalid |= w & np.isinf(top).any(1)
            fired = side.first_alarm[P] < cut[:, None]
            tp += w * (fired & (yb == 1)).sum(1)
            nf += w * n1
            fp += w * (fired & (yb == 0)).sum(1)
            ns += w * n0
            if pooled:
                P_tops.append(np.where(np.isinf(top), -1e300, top))
                P_ys.append(yb)
                P_w.append(np.repeat(w[:, None], yb.shape[1], 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        tpr = tp / nf
        tnr = 1 - fp / ns
    m1[invalid] = np.nan
    out = {"m1_per_task": m1, "bal_acc": (tpr + tnr) / 2, "tpr": tpr, "far": 1 - tnr,
           "groups_used": den, "groups_dropped": dropped, "groups_drawn": drawn}
    if pooled:
        top, yb, w = np.concatenate(P_tops, 1), np.concatenate(P_ys, 1), np.concatenate(P_w, 1)
        # Left-out copies rank below every drawn rollout, so a drawn rollout's rank among
        # the drawn ones is its overall rank minus the number left out.
        r = rankdata(np.where(w, top, -np.inf), axis=1, method="average") - (~w).sum(1, keepdims=True)
        n1 = (yb * w).sum(1)
        n0 = w.sum(1) - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            pa = ((r * yb * w).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        pa[invalid] = np.nan
        out["m1_pooled"] = pa
    return out


def side_rounds(sides: list, plans: list, pooled: bool = False) -> tuple[dict, dict]:
    """Split means per round (NaN if any split is undefined) and per-split values."""
    per = [split_metrics(s, p, pooled) for s, p in zip(sides, plans)]
    keys = [k for k in per[0] if not k.startswith("groups")]
    st = {k: np.stack([p[k] for p in per]) for k in per[0]}
    means = {k: st[k].mean(0) for k in keys}                         # NaN propagates: round skipped
    return means, st


@dataclass
class Comparison:
    section: str
    title: str
    x: tuple                   # (files, key)
    y: tuple
    test_set: str = "T2"
    family: int | None = None
    metrics: tuple = ("m1_per_task", "bal_acc")
    note: str = ""


METRIC_NAME = {"m1_per_task": "per-task AUROC", "bal_acc": "balanced accuracy at alpha 0.1",
               "m1_pooled": "pooled AUROC", "tpr": "share of failures caught at alpha 0.1"}


class Engine:
    """Caches sides, plans (one per test set / family) and per-side round values."""

    def __init__(self, n_rounds=N_ROUNDS, rng_seed=RNG_SEED, resample_tasks=True, plan_fn=None):
        self.n_rounds, self.rng_seed, self.resample_tasks = n_rounds, rng_seed, resample_tasks
        self.plan_fn = plan_fn
        self.sides, self.plans, self.vals = {}, {}, {}

    def side(self, files, key, test_set, family):
        k = (tuple(map(str, files)), key, test_set, family)
        if k not in self.sides:
            self.sides[k] = load_side(files, key, test_set, family)
        return k, self.sides[k]

    def plan(self, sides, test_set, family):
        k = (test_set, family)
        if k not in self.plans:
            self.plans[k] = (self.plan_fn(sides) if self.plan_fn else
                             make_plan(sides, self.n_rounds, self.rng_seed, self.resample_tasks))
            self.plans[(k, "id")] = identity_plan(sides)
        return self.plans[k], self.plans[(k, "id")]

    def values(self, sk, sides, plans, pooled, tag):
        k = (sk, pooled, tag)
        if k not in self.vals:
            self.vals[k] = side_rounds(sides, plans, pooled)
        return self.vals[k]

    def run(self, c: Comparison) -> list:
        kx, x = self.side(*c.x, c.test_set, c.family)
        ky, y = self.side(*c.y, c.test_set, c.family)
        b2.check_same_test_set(x, y)
        plans, ident = self.plan(y, c.test_set, c.family)
        pooled = "m1_pooled" in c.metrics
        xb, xs = self.values(kx, x, plans, pooled, "boot")
        yb, ys = self.values(ky, y, plans, pooled, "boot")
        xi, _ = self.values(kx, x, ident, pooled, "id")
        yi, _ = self.values(ky, y, ident, pooled, "id")
        rows = []
        for m in c.metrics:
            d = xb[m] - yb[m]
            ok = np.isfinite(d)
            lo, hi = np.percentile(d[ok], CI) if ok.any() else (np.nan, np.nan)
            any_drop = (xs["groups_dropped"] > 0).any(0)
            rows.append({"section": c.section, "comparison": c.title, "metric": METRIC_NAME[m],
                         "difference": float(xi[m][0] - yi[m][0]),
                         "95% range low": float(lo), "95% range high": float(hi),
                         "range excludes zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                         "rounds used": int(ok.sum()), "rounds skipped": int((~ok).sum()),
                         "rounds with a group dropped": int(any_drop[ok].sum()),
                         "groups dropped per round (mean over used rounds, all splits)":
                             float(xs["groups_dropped"].sum(0)[ok].mean()) if ok.any() else np.nan,
                         "groups drawn per round (mean over used rounds, all splits)":
                             float(xs["groups_drawn"].sum(0)[ok].mean()) if ok.any() else np.nan,
                         "x value": float(xi[m][0]), "y value": float(yi[m][0]),
                         "bootstrap mean difference": float(d[ok].mean()) if ok.any() else np.nan,
                         "test set": common.TEST_SETS[c.test_set] + (f", {FAMILY[c.family]} groups" if c.family is not None else ""),
                         "x key": c.x[1], "y key": c.y[1], "x files": str(Path(c.x[0][0]).parent),
                         "y files": str(Path(c.y[0][0]).parent), "note": c.note})
        return rows


def quiet():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
