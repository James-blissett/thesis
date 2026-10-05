"""
data.py

Shared data layer for the A/B/C build (handoff section 2): the rollout index, the kept
mask, the task splits, labels, row weights, and read access to the feature cache.

Every stage imports from here, so every stage sees the same rollout order, the same
kept rows and the same splits.

Conventions
-----------
* Rollout order is the order of corpus_v2_index.json. Arrays shaped (300, ...) and the
  feature cache's first axis both follow it.
* Labels: 1 = failure, 0 = success, from success_ever (matching probe_layer.py).
* Kept rows: successes keep 1 <= t < t_success, failures keep 1 <= t < 520. t = 0 is
  dropped everywhere because the temporal constraints are NaN there.

Usage (from the repo root), writes results/abc/splits/seed{0,1,2}.json and runs the
Stage 1 checks that don't need the cache:
    source env.sh
    python analysis/abc/data.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORPUS_ROOT = Path("/data/rollouts_v2")
INDEX_JSON = Path("corpus_v2_index.json")
CONSTRAINTS_DIR = Path("constraints")
CACHE_DIR = Path("/data/tmp/abc_features")
RESULTS_DIR = Path("results/abc")
SPLITS_DIR = RESULTS_DIR / "splits"

T = 520
N_STORED = 33          # stored indices 0..32; 32 is post-norm
N_RAW = 32             # raw layers 0..31
POST_NORM_INDEX = 32
D_MODEL = 4096
N_TASKS = 10
N_UNSEEN = 3
TRAIN_FRAC = 0.6
SEEDS = (0, 1, 2)
SPLIT_NAMES = ("train", "eval_seen", "unseen")

# Expected kept-row totals after dropping t = 0 (handoff Stage 1).
EXPECTED_SUCCESS_ROWS = 38_618     # results/completion_confound.json pre_rows
EXPECTED_FAILURE_ROWS = 163 * 519


# --- Index ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Index:
    rids: np.ndarray          # (300,) str
    task: np.ndarray          # (300,) int
    y: np.ndarray             # (300,) int8, 1 = failure
    t_success: np.ndarray     # (300,) float, NaN for failures
    n_positions: np.ndarray   # (300,) int, 3 or 7

    @property
    def n(self) -> int:
        return len(self.rids)

    @property
    def kept_len(self) -> np.ndarray:
        """(300,) one past the last kept t: t_success for successes, T for failures."""
        return np.where(self.y == 1, T, np.nan_to_num(self.t_success)).astype(np.int64)

    def pos(self, rids) -> np.ndarray:
        """Row positions of the given rollout ids."""
        lookup = {r: i for i, r in enumerate(self.rids)}
        return np.array([lookup[r] for r in rids], dtype=np.int64)


def load_index() -> Index:
    entries = json.loads(INDEX_JSON.read_text())["rollouts"]
    succ = np.array([bool(e["success_ever"]) for e in entries])
    tfs = np.array([np.nan if e["t_success"] is None else float(e["t_success"])
                    for e in entries])
    if not np.array_equal(np.isfinite(tfs), succ):
        raise SystemExit("t_success should be defined exactly on the success_ever rollouts")
    return Index(
        rids=np.array([e["rollout_id"] for e in entries]),
        task=np.array([int(e["task_idx"]) for e in entries]),
        y=(~succ).astype(np.int8),
        t_success=tfs,
        n_positions=np.array([int(e["n_positions"]) for e in entries]),
    )


# --- Kept rows ------------------------------------------------------------------------
def kept_mask(idx: Index) -> np.ndarray:
    """(300, 520) bool. Successes: 1 <= t < t_success. Failures: 1 <= t < 520."""
    t = np.arange(T)[None, :]
    return (t >= 1) & (t < idx.kept_len[:, None])


def rows(mask: np.ndarray, which: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(r, t) index arrays of the True cells of `mask`, restricted to rollout positions
    `which` if given. Row order is rollout-major, then t."""
    if which is not None:
        sub = np.zeros_like(mask)
        sub[which] = mask[which]
        mask = sub
    r, t = np.nonzero(mask)
    return r, t


# --- Splits ---------------------------------------------------------------------------
def make_split(seed: int, idx: Index) -> dict:
    """3 unseen tasks, then a 60/40 whole-rollout split of the seen tasks into train and
    eval_seen, stratified by (task, outcome). One rng, default_rng(seed), drives both."""
    rng = np.random.default_rng(seed)
    unseen_tasks = sorted(int(t) for t in rng.choice(N_TASKS, N_UNSEEN, replace=False))
    seen_tasks = [t for t in range(N_TASKS) if t not in unseen_tasks]

    train, eval_seen = [], []
    for task in seen_tasks:
        for label in (0, 1):
            members = sorted(idx.rids[(idx.task == task) & (idx.y == label)])
            if not members:
                continue
            perm = list(rng.permutation(members))
            k = int(round(TRAIN_FRAC * len(perm)))
            train += perm[:k]
            eval_seen += perm[k:]
    unseen = sorted(idx.rids[np.isin(idx.task, unseen_tasks)])

    split = {"seed": seed, "unseen_tasks": unseen_tasks, "seen_tasks": seen_tasks,
             "train": sorted(train), "eval_seen": sorted(eval_seen), "unseen": unseen}
    split["counts"] = split_counts(split, idx)
    return split


def split_counts(split: dict, idx: Index) -> dict:
    """Per split: rollouts per class, and per task as [successes, failures]."""
    out = {}
    for name in SPLIT_NAMES:
        p = idx.pos(split[name])
        per_task = {}
        for task in sorted(set(idx.task[p].tolist())):
            m = idx.task[p] == task
            per_task[str(task)] = [int((idx.y[p][m] == 0).sum()), int((idx.y[p][m] == 1).sum())]
        out[name] = {"n": int(len(p)), "success": int((idx.y[p] == 0).sum()),
                     "failure": int((idx.y[p] == 1).sum()), "per_task": per_task}
    return out


def split_path(seed: int) -> Path:
    return SPLITS_DIR / f"seed{seed}.json"


def save_split(split: dict) -> Path:
    path = split_path(split["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return path


def load_split(seed: int, idx: Index) -> dict:
    """The saved split, with rollout positions added under `pos`: {name: (n,) int}."""
    split = json.loads(split_path(seed).read_text())
    split["pos"] = {name: idx.pos(split[name]) for name in SPLIT_NAMES}
    return split


# --- Labels and weights ---------------------------------------------------------------
def row_labels(idx: Index, r: np.ndarray) -> np.ndarray:
    """Episode outcome given to every kept row (SAFE's labelling). 1 = failure."""
    return idx.y[r]


MIN_TRAIN_SUCCESSES = 5


def train_class_counts(idx: Index, kept: np.ndarray, train_pos: np.ndarray) -> np.ndarray:
    """(2, T): number of train rollouts of each class (0 = success) still kept at t."""
    y_tr = idx.y[train_pos]
    return np.stack([kept[train_pos[y_tr == label]].sum(0) for label in (0, 1)])


def train_cut(idx: Index, kept: np.ndarray, train_pos: np.ndarray,
              min_success: int = MIN_TRAIN_SUCCESSES) -> int:
    """Last t at which at least `min_success` train successes are still kept. Successes
    only drop out as t grows, so every t <= t_cut qualifies."""
    ok = train_class_counts(idx, kept, train_pos)[0] >= min_success
    if not ok.any():
        raise ValueError(f"no timestep has {min_success} train successes kept")
    return int(np.nonzero(ok)[0].max())


def train_kept(idx: Index, kept: np.ndarray, train_pos: np.ndarray,
               min_success: int = MIN_TRAIN_SUCCESSES) -> np.ndarray:
    """(300, 520) kept mask restricted to t <= train_cut: the train rows every fitted
    quantity uses (scaling, orientation, outcome-trained models)."""
    return kept & (np.arange(T)[None, :] <= train_cut(idx, kept, train_pos, min_success))


def train_rows_and_weights(idx: Index, kept: np.ndarray, train_pos: np.ndarray,
                           min_success: int = MIN_TRAIN_SUCCESSES):
    """Training rows and weights for every model trained on outcome.

    Rows: kept rows of train rollouts at timesteps where at least `min_success` train
    successes are still kept. Successes only drop out as t grows, so this is a prefix
    t <= t_cut. (min_success = 1 gives the handoff's original "last train success"
    cut-off.) Weights: 1 / n_class(t), n_class(t) being the number of train rollouts of
    that row's class still kept at t, so each class carries equal total weight at every
    t. The weights are then rescaled to mean 1 over the training rows, so the L2 penalty
    at C = 1.0 has the same strength relative to the data term as in an unweighted fit
    on the same rows.

    Returns (r, t, y, w), each (n_rows,).
    """
    n_class = train_class_counts(idx, kept, train_pos)
    t_cut = train_cut(idx, kept, train_pos, min_success)

    r, t = rows(kept, train_pos)
    keep = t <= t_cut
    r, t = r[keep], t[keep]
    y = idx.y[r]

    w = 1.0 / n_class[y, t]
    w *= len(w) / w.sum()
    return r, t, y, w


def weight_stats(w: np.ndarray) -> dict:
    """Concentration of a weight vector: max/mean, share of total weight in the top 1%
    of rows, and Kish's effective sample size (sum w)^2 / sum w^2."""
    top = np.sort(w)[::-1][: max(1, int(np.ceil(0.01 * len(w))))]
    return {"max_over_mean": float(w.max() / w.mean()),
            "top1pct_share": float(top.sum() / w.sum()),
            "ess": float(w.sum() ** 2 / (w ** 2).sum()),
            "n_rows": int(len(w))}


# --- M1: trajectory ROC-AUC (SAFE's protocol) -----------------------------------------
M1_TASK_MIN_PER_CLASS = 2



def m1_cut(idx: Index, pos: np.ndarray) -> np.ndarray:
    """(len(pos),) per-rollout cut T_task: the shortest kept length among this set's
    rollouts in the same task (T if the set has no success in that task)."""
    cut = np.empty(len(pos), dtype=np.int64)
    for task in np.unique(idx.task[pos]):
        m = idx.task[pos] == task
        cut[m] = idx.kept_len[pos][m].min()
    return cut


def m1(score: np.ndarray, kept: np.ndarray, idx: Index, pos: np.ndarray) -> dict:
    """Trajectory ROC-AUC over the rollouts at positions `pos`, three ways.

    score: (300, 520) or (n_series, 300, 520) per-row failure score, higher = failure.
    Each rollout is scored by its maximum over kept rows with t < T_task.

      pooled    SAFE's M1: one ROC-AUC over all rollouts in the set.
      per_task  the same within each task, averaged over tasks with at least
                M1_TASK_MIN_PER_CLASS rollouts of each class in the set.
                Length-immune: a timestep-only score gets exactly 0.5.
      floor     pooled M1 of the timestep-only score on this set. Not exactly 0.5,
                because tasks differ in cut length and failure rate.
      n_tasks   tasks entering per_task.

    NaN scores mark missing rows (before a series' first valid value); they are left out
    of the maximum, so no fill value can become a rollout's score. A rollout with no
    valid row under its cut is an error.

    pooled / per_task are floats, or (n_series,) arrays for stacked input.
    """
    from sklearn.metrics import roc_auc_score

    cut = m1_cut(idx, pos)
    mask = kept[pos] & (np.arange(T)[None, :] < cut[:, None])          # (n, T)
    s = np.asarray(score)[..., pos, :]
    top = np.where(mask & ~np.isnan(s), s, -np.inf).max(-1)            # (..., n)
    if np.isinf(top).any():
        raise ValueError("a rollout has no valid score under its M1 cut")
    y = idx.y[pos]
    task = idx.task[pos]
    tasks = [k for k in np.unique(task)
             if min((y[task == k] == 0).sum(), (y[task == k] == 1).sum()) >= M1_TASK_MIN_PER_CLASS]
    if not tasks:
        raise ValueError("no task qualifies for per-task M1")

    flat = top.reshape(-1, top.shape[-1])
    pooled = np.array([roc_auc_score(y, row) for row in flat])
    per_task = np.array([np.mean([roc_auc_score(y[task == k], row[task == k]) for k in tasks])
                         for row in flat])
    t_last = np.where(mask, np.arange(T)[None, :], -1).max(-1)
    floor = float(roc_auc_score(y, t_last))
    if top.ndim == 1:
        pooled, per_task = float(pooled[0]), float(per_task[0])
    return {"pooled": pooled, "per_task": per_task, "floor": floor, "n_tasks": len(tasks)}


# --- Feature cache --------------------------------------------------------------------
def cache_layer_path(layer: int, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"layer{layer:02d}.npy"


def open_layer(layer: int, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """(300, 520, 4096) fp16 memmap of pooled hidden state at stored index `layer`.
    Index 32 is post-norm."""
    return np.load(cache_layer_path(layer, cache_dir), mmap_mode="r")


def open_action_tokens(cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """(300, 520, 7) int32 action_token_ids."""
    return np.load(cache_dir / "action_tokens.npy", mmap_mode="r")


# --- Stage 1 checks that need no cache ------------------------------------------------
def main() -> None:
    idx = load_index()
    kept = kept_mask(idx)
    n_s = int(kept[idx.y == 0].sum())
    n_f = int(kept[idx.y == 1].sum())
    print(f"[kept] success rows {n_s} (expect {EXPECTED_SUCCESS_ROWS}), "
          f"failure rows {n_f} (expect {EXPECTED_FAILURE_ROWS})")
    assert n_s == EXPECTED_SUCCESS_ROWS and n_f == EXPECTED_FAILURE_ROWS
    assert not kept[:, 0].any()

    # Per-task minimum success length must match the handoff's quirks table.
    expect_min = {0: 261, 1: 228, 2: 259, 3: 241, 4: 201, 5: 158, 6: 203, 7: 216, 8: 355, 9: 244}
    got_min = {t: int(idx.kept_len[(idx.task == t) & (idx.y == 0)].min()) for t in range(N_TASKS)}
    print(f"[kept] shortest success per task {got_min}")
    assert got_min == expect_min

    for seed in SEEDS:
        split = make_split(seed, idx)
        sets = [set(split[n]) for n in SPLIT_NAMES]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), \
            "a rollout appears in two splits"
        assert len(set.union(*sets)) == idx.n
        unseen_task_set = set(idx.task[idx.pos(split["unseen"])].tolist())
        assert unseen_task_set == set(split["unseen_tasks"])
        assert not (set(idx.task[idx.pos(split["train"])].tolist()) & unseen_task_set)
        path = save_split(split)

        # Reload from disk: what later stages will see.
        split = load_split(seed, idx)
        c = split["counts"]
        print(f"[split seed {seed}] unseen tasks {split['unseen_tasks']} -> {path}")
        for name in SPLIT_NAMES:
            pt = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in c[name]["per_task"].items())
            print(f"    {name:9s} n={c[name]['n']:3d}  succ {c[name]['success']:3d}  "
                  f"fail {c[name]['failure']:3d}   per task (succ/fail) {pt}")

        for label, k in (("before (>= 1 success)", 1),
                         (f"after  (>= {MIN_TRAIN_SUCCESSES} successes)", MIN_TRAIN_SUCCESSES)):
            r, t, y, w = train_rows_and_weights(idx, kept, split["pos"]["train"], min_success=k)
            per_t = Counter()
            for c in (0, 1):
                per_t[c] = np.bincount(t[y == c], weights=w[y == c], minlength=T)
            both = (per_t[0] > 0) & (per_t[1] > 0)
            assert np.allclose(per_t[0][both], per_t[1][both]), "classes not balanced per t"
            s = weight_stats(w)
            print(f"    {label}: t <= {t.max()}, rows {s['n_rows']} "
                  f"(succ {int((y == 0).sum())}, fail {int((y == 1).sum())}); "
                  f"max/mean {s['max_over_mean']:.1f}, top 1% share {s['top1pct_share']:.3f}, "
                  f"ESS {s['ess']:.0f} ({s['ess'] / s['n_rows']:.2f} of rows)")

    print("[*] data checks passed")


if __name__ == "__main__":
    main()
