"""
data2.py

Shared data layer for the more-data run (more_data_collection.md, handoff v1, Steps 5-9).
Extends analysis/abc/data.py (imported, not copied) from the 300-rollout v2 corpus to
the combined v2 + v3 corpus, and adds blocks, cells, conditions and test sets.

Conventions (everything else as analysis/abc/data.py)
-----------------------------------------------------
* Rollout order is the order of corpus_v3_index.json: the 300 v2 rollouts first, in
  corpus_v2_index.json order, then the v3 rollouts. Every (N, ...) array and the feature
  cache's first axis follow it.
* Labels: 1 = failure, 0 = success, from success_ever.
* Kept rows: as abc (successes 1 <= t < t_success, failures 1 <= t < 520).
* Blocks: ori (v2), ori2, obj, lan (training role), swap, task (stress only).
  Families: unperturbed = {ori, ori2}; obj; lan; swap; task.
* `Index2.task` is the CELL id, cell = base_task + 10 * family_code. analysis/abc/data.py's
  m1(), m1_cut() and kept_mask() group by `idx.task`, so passing an Index2 makes "per
  task" mean "per cell" (Step 7), with no copy of those functions. For the unperturbed
  family the cell id equals the base task, so D0 on T0 is the old computation exactly.
  Use `idx.base_task` wherever the base task is meant (splits, stress cuts).

Splits (Step 7): results/abc2/splits/seed{s}.json, made by make_split2 (`python
analysis/abc2/data2.py --splits`). Lists of rollout ids:
    train, eval_seen   training-role blocks (ori, ori2, obj, lan) in seen base tasks
    unseen             training-role blocks in the split's 3 unseen base tasks
    stress             swap and task rollouts, all base tasks
Demoted blocks (Step 5; results/abc2/demotions.json) keep their sides in the file; the
condition and test-set helpers below move them out of training and into Stress.

Conditions (training pools) and test sets: condition_sets() and test_sets().
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ABC = Path(__file__).resolve().parents[1] / "abc"
sys.path.insert(0, str(ABC))
sys.path.insert(1, str(ABC.parent))
import data as abc_data  # noqa: E402
from data import (D_MODEL, MIN_TRAIN_SUCCESSES, N_RAW, N_STORED, POST_NORM_INDEX,  # noqa: E402,F401
                  SEEDS, T, kept_mask, m1, m1_cut, rows, train_class_counts, train_cut,
                  train_kept, train_rows_and_weights, weight_stats)

INDEX_V2 = Path("corpus_v2_index.json")
INDEX_V3 = Path("corpus_v3_index.json")
ROLLOUTS_V2 = Path("/data/rollouts_v2")
ROLLOUTS_V3 = Path("/data/rollouts_v3")
CONSTRAINTS_V2 = Path("constraints")
CONSTRAINTS_V3 = Path("constraints_v3")
CACHE_DIR = Path("/data/tmp/abc2_features")
OLD_CACHE_DIR = Path("/data/tmp/abc_features")
RESULTS_DIR = Path("results/abc2")
SPLITS_DIR = RESULTS_DIR / "splits"
OLD_SPLITS_DIR = Path("results/abc/splits")
DEMOTIONS = RESULTS_DIR / "demotions.json"

BLOCKS = ("ori", "ori2", "obj", "lan", "swap", "task")
TRAINING_BLOCKS = ("ori", "ori2", "obj", "lan")
STRESS_BLOCKS = ("swap", "task")
FAMILY_OF = {"ori": "unperturbed", "ori2": "unperturbed", "obj": "obj", "lan": "lan",
             "swap": "swap", "task": "task"}
FAMILY_CODE = {"unperturbed": 0, "obj": 1, "lan": 2, "swap": 3, "task": 4}
CONDITIONS = {"D0": ("ori",), "D1": ("ori", "ori2"), "D2": ("ori", "ori2", "obj", "lan")}
TEST_SETS = ("T0", "T1", "T2", "stress_seen", "stress_unseen")
TRAIN_FRAC = 0.6


def cell_id(base_task: int, family: str) -> int:
    return int(base_task) + 10 * FAMILY_CODE[family]


def cell_name(cell: int) -> str:
    fam = {v: k for k, v in FAMILY_CODE.items()}[cell // 10]
    return f"task{cell % 10}/{fam}"


# --- Index ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Index2:
    rids: np.ndarray           # (N,) str
    task: np.ndarray           # (N,) int CELL id (see module docstring)
    base_task: np.ndarray      # (N,) int 0..9
    block: np.ndarray          # (N,) str
    family: np.ndarray         # (N,) str
    source: np.ndarray         # (N,) str, v2 or v3
    init_state_index: np.ndarray
    y: np.ndarray              # (N,) int8, 1 = failure
    t_success: np.ndarray      # (N,) float, NaN for failures
    n_positions: np.ndarray    # (N,) int: 3 or 7 for v2, 1 for v3 (pooled only)

    n = abc_data.Index.n
    kept_len = abc_data.Index.kept_len
    pos = abc_data.Index.pos

    def blocks_mask(self, blocks) -> np.ndarray:
        return np.isin(self.block, list(blocks))

    def constraints_path(self, i: int) -> Path:
        d = CONSTRAINTS_V2 if self.source[i] == "v2" else CONSTRAINTS_V3
        return d / f"{self.rids[i]}.npz"

    def rollout_dir(self, i: int) -> Path:
        d = ROLLOUTS_V2 if self.source[i] == "v2" else ROLLOUTS_V3
        return d / str(self.rids[i])


def index_from_entries(entries: list[dict]) -> Index2:
    succ = np.array([bool(e["success_ever"]) for e in entries])
    tfs = np.array([np.nan if e["t_success"] is None else float(e["t_success"]) for e in entries])
    if not np.array_equal(np.isfinite(tfs), succ):
        raise SystemExit("t_success should be defined exactly on the success_ever rollouts")
    block = np.array([e["block"] for e in entries])
    fam = np.array([FAMILY_OF[b] for b in block])
    base = np.array([int(e["base_task_idx"]) for e in entries])
    return Index2(
        rids=np.array([e["rollout_id"] for e in entries]),
        task=np.array([cell_id(b, f) for b, f in zip(base, fam)]),
        base_task=base, block=block, family=fam,
        source=np.array([e["source"] for e in entries]),
        init_state_index=np.array([int(e["init_state_index"]) for e in entries]),
        y=(~succ).astype(np.int8), t_success=tfs,
        n_positions=np.array([int(e.get("n_positions", 1)) for e in entries]),
    )


def load_index2(path: Path = INDEX_V3) -> Index2:
    return index_from_entries(json.loads(Path(path).read_text())["rollouts"])


def v2_entries() -> list[dict]:
    """The 300 v2 rollouts in corpus_v2_index.json order, with the v3 index fields. The
    instruction is the manifest's task_description (corpus_v2_index.json has none)."""
    out = []
    for e in json.loads(INDEX_V2.read_text())["rollouts"]:
        e = dict(e)
        instr = e.get("task_description")
        if instr is None:
            man = json.loads((ROLLOUTS_V2 / e["rollout_id"] / "manifest.json").read_text())
            instr = man["task_description"]
        e.update(block="ori", base_task_idx=int(e["task_idx"]), source="v2",
                 suite="libero_10", instruction=instr)
        out.append(e)
    return out


def v3_entries(root: Path = ROLLOUTS_V3) -> list[dict]:
    """Complete v3 rollouts (meta.json with complete: true), in job order."""
    metas = []
    for d in sorted(p for p in Path(root).glob("*_task*_trial*") if not p.name.startswith(".")):
        try:
            m = json.loads((d / "meta.json").read_text())
        except Exception:
            continue
        if not m.get("complete") or not all((d / f).exists()
                                            for f in ("hidden.pt", "logits.pt", "actions.npz")):
            continue
        metas.append(m)
    order = {b: i for i, b in enumerate(BLOCKS)}
    metas.sort(key=lambda m: (m["trial_idx"], order[m["block"]], m["base_task_idx"]))
    out = []
    for m in metas:
        out.append({
            "rollout_id": m["rollout_id"], "task_idx": int(m["base_task_idx"]),
            "task_name": m.get("task_name"), "trial_idx": int(m["trial_idx"]),
            "init_state_index": int(m["init_state_index"]), "n_positions": 1,
            "T_total": int(m["T_total"]), "success_ever": bool(m["success_ever"]),
            "success_final": bool(m["success_final"]), "t_success": m["t_success"],
            "size_bytes": int(m["bytes"]), "model_checkpoint": m["model_checkpoint"],
            "unnorm_key": m["unnorm_key"], "block": m["block"], "suite": m["suite"],
            "base_task_idx": int(m["base_task_idx"]), "instruction": m["instruction"],
            "source": "v3",
        })
    return out


def write_index(path: Path = INDEX_V3, root: Path = ROLLOUTS_V3) -> dict:
    """corpus_v3_index.json (Step 5): v2 first in their existing order, then v3."""
    entries = v2_entries() + v3_entries(root)
    doc = {"schema": "corpus_v3_index/v1", "n_rollouts": len(entries),
           "n_v2": sum(e["source"] == "v2" for e in entries),
           "n_v3": sum(e["source"] == "v3" for e in entries),
           "rollouts": entries}
    if Path(path).resolve() == INDEX_V2.resolve():
        raise SystemExit("refusing to overwrite corpus_v2_index.json")
    Path(path).write_text(json.dumps(doc, indent=1))
    return doc


# --- Demotions (Step 5) ---------------------------------------------------------------
def demoted(path: Path = DEMOTIONS) -> tuple[str, ...]:
    """Blocks demoted to stress-test only (obj / lan outside 10-90% success)."""
    if not Path(path).exists():
        return ()
    return tuple(json.loads(Path(path).read_text()).get("demoted", []))


def condition_blocks(cond: str, demotions: tuple[str, ...] | None = None) -> tuple[str, ...]:
    dem = demoted() if demotions is None else demotions
    return tuple(b for b in CONDITIONS[cond] if b not in dem)


def stress_blocks(demotions: tuple[str, ...] | None = None) -> tuple[str, ...]:
    dem = demoted() if demotions is None else demotions
    return STRESS_BLOCKS + tuple(b for b in TRAINING_BLOCKS if b in dem)


# --- Splits (Step 7) ------------------------------------------------------------------
def make_split2(seed: int, idx: Index2) -> dict:
    """Same unseen base tasks as results/abc/splits/seed{s}.json; v2 rollouts keep their
    old side; new training-role rollouts in seen tasks are grouped by (base task,
    init-state index): a group containing a v2 rollout inherits its side, the rest are
    split 60/40 train / eval_seen per task with default_rng(seed) (one rng, tasks in
    ascending order, groups in ascending init index, first round(0.6 n) of a permutation
    to train). Stress blocks (swap, task) go to `stress` whatever their base task."""
    old = json.loads((OLD_SPLITS_DIR / f"seed{seed}.json").read_text())
    unseen_tasks = [int(t) for t in old["unseen_tasks"]]
    seen_tasks = [t for t in range(10) if t not in unseen_tasks]
    side = {}
    for name in ("train", "eval_seen", "unseen"):
        for rid in old[name]:
            side[rid] = name
    v2_side_by_group = {}
    for i in np.nonzero(idx.source == "v2")[0]:
        if str(idx.rids[i]) not in side:
            raise SystemExit(f"v2 rollout {idx.rids[i]} missing from old split seed {seed}")
        g = (int(idx.base_task[i]), int(idx.init_state_index[i]))
        v2_side_by_group[g] = side[str(idx.rids[i])]

    rng = np.random.default_rng(seed)
    new = (idx.source == "v3") & np.isin(idx.block, TRAINING_BLOCKS)
    for i in np.nonzero(new & np.isin(idx.base_task, unseen_tasks))[0]:
        side[str(idx.rids[i])] = "unseen"
    inherited, drawn = 0, 0
    for task in seen_tasks:
        members = np.nonzero(new & (idx.base_task == task))[0]
        groups = sorted({int(idx.init_state_index[i]) for i in members})
        free = []
        for g in groups:
            if (task, g) in v2_side_by_group:
                for i in members[idx.init_state_index[members] == g]:
                    side[str(idx.rids[i])] = v2_side_by_group[(task, g)]
                    inherited += 1
            else:
                free.append(g)
        perm = list(rng.permutation(free)) if free else []
        k = int(round(TRAIN_FRAC * len(perm)))
        for j, g in enumerate(perm):
            for i in members[idx.init_state_index[members] == g]:
                side[str(idx.rids[i])] = "train" if j < k else "eval_seen"
                drawn += 1
    for i in np.nonzero(np.isin(idx.block, STRESS_BLOCKS))[0]:
        side[str(idx.rids[i])] = "stress"

    split = {"seed": seed, "unseen_tasks": unseen_tasks, "seen_tasks": seen_tasks,
             "rule": "make_split2 (handoff v1 Step 7)",
             "n_new_inherited_side": inherited, "n_new_drawn_side": drawn}
    have = set(idx.rids.tolist())
    for name in ("train", "eval_seen", "unseen", "stress"):
        split[name] = sorted(r for r, s in side.items() if s == name and r in have)
    if sum(len(split[n]) for n in ("train", "eval_seen", "unseen", "stress")) != idx.n:
        raise SystemExit("split does not cover every rollout exactly once")
    split["counts"] = split_counts2(split, idx)
    return split


def split_counts2(split: dict, idx: Index2) -> dict:
    """Per split and block: [successes, failures]."""
    out = {}
    for name in ("train", "eval_seen", "unseen", "stress"):
        p = idx.pos(split[name]) if split[name] else np.array([], dtype=np.int64)
        out[name] = {b: [int(((idx.block[p] == b) & (idx.y[p] == 0)).sum()),
                         int(((idx.block[p] == b) & (idx.y[p] == 1)).sum())]
                     for b in BLOCKS if (idx.block[p] == b).any()}
    return out


def split_path(seed: int) -> Path:
    return SPLITS_DIR / f"seed{seed}.json"


def load_split2(seed: int, idx: Index2, splits_dir: Path | None = None) -> dict:
    """The saved split with positions under `pos`. Rollout ids not in `idx` are dropped
    (so a v2-only development index works with the full split file). `splits_dir`
    (default results/abc2/splits) lets dev runs read splits written with --splits-dir."""
    path = Path(splits_dir) / f"seed{seed}.json" if splits_dir else split_path(seed)
    split = json.loads(path.read_text())
    have = set(idx.rids.tolist())
    split["pos"] = {name: idx.pos([r for r in split[name] if r in have])
                    for name in ("train", "eval_seen", "unseen", "stress")}
    return split


def condition_sets(split: dict, idx: Index2, cond: str, demotions=None) -> dict:
    """{train, eval_seen, cal} positions for a condition: the split restricted to the
    condition's (non-demoted) blocks. cal = successful eval_seen rollouts (Stage 6)."""
    m = idx.blocks_mask(condition_blocks(cond, demotions))
    out = {name: split["pos"][name][m[split["pos"][name]]] for name in ("train", "eval_seen")}
    out["cal"] = out["eval_seen"][idx.y[out["eval_seen"]] == 0]
    return out


def test_sets(split: dict, idx: Index2, demotions=None) -> dict:
    """T0: unseen ori. T1: unseen ori + ori2. T2: unseen, every non-demoted training
    block. stress_seen / stress_unseen: swap, task and demoted blocks, by whether the
    base task is one of the split's unseen tasks. Identical for every condition."""
    dem = demoted() if demotions is None else demotions
    u = split["pos"]["unseen"]
    t2_blocks = [b for b in TRAINING_BLOCKS if b not in dem]
    sb = stress_blocks(dem)
    allpos = np.arange(idx.n)
    stress = allpos[idx.blocks_mask(sb)]
    unseen_task = np.isin(idx.base_task[stress], split["unseen_tasks"])
    return {"T0": u[idx.block[u] == "ori"],
            "T1": u[np.isin(idx.block[u], ["ori", "ori2"])],
            "T2": u[np.isin(idx.block[u], t2_blocks)],
            "stress_seen": stress[~unseen_task],
            "stress_unseen": stress[unseen_task]}


# --- CLI ------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX_V3)
    ap.add_argument("--splits", action="store_true", help="write results/abc2/splits/seed{s}.json")
    ap.add_argument("--splits-dir", type=Path, default=None)
    args = ap.parse_args()
    idx = load_index2(args.index)
    print(f"[index] {idx.n} rollouts: " + ", ".join(
        f"{b} {int((idx.block == b).sum())}" for b in BLOCKS if (idx.block == b).any()))
    if not args.splits:
        return
    out_dir = args.splits_dir or SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        split = make_split2(seed, idx)
        # v2 membership must be exactly the old split's
        old = json.loads((OLD_SPLITS_DIR / f"seed{seed}.json").read_text())
        for name in ("train", "eval_seen", "unseen"):
            v2_now = sorted(r for r in split[name] if r in set(old["train"] + old["eval_seen"] + old["unseen"]))
            assert v2_now == sorted(old[name]), f"seed {seed} {name}: v2 membership changed"
        path = out_dir / f"seed{seed}.json"
        path.write_text(json.dumps(split, indent=1))
        print(f"[split seed {seed}] unseen tasks {split['unseen_tasks']} -> {path} "
              f"(new rollouts: {split['n_new_inherited_side']} inherited a v2 side, "
              f"{split['n_new_drawn_side']} drawn)")
        for name, per_block in split["counts"].items():
            print(f"    {name:9s} " + "  ".join(f"{b}: {s}/{f}" for b, (s, f) in per_block.items())
                  + "   (succ/fail)")


if __name__ == "__main__":
    main()
