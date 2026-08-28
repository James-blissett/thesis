"""
make_init_assignment.py  --  Rollout Collection v2, Change 6

Draws the 30 initial-state indices per task that the v2 corpus will consume, and writes
`init_state_assignment.json`.

Why this file exists at all
---------------------------
Decoding is greedy and MuJoCo is deterministic, so the *only* source of variation between
two rollouts of the same task is the initial state. Re-running a consumed init-state index
does not produce a new sample: it produces the same rollout again. The set of consumed
indices is therefore part of the corpus definition, not an implementation detail, and
because the selection is randomised rather than formulaic it cannot be recovered from the
data. It must be in git before collection starts.

`unconsumed` is written per task so a future extension batch can draw from it without
colliding with this one.

Usage:
    source env.sh
    python collect/make_init_assignment.py                       # writes ./init_state_assignment.json
    python collect/make_init_assignment.py --dry-run             # print, write nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OPENVLA_ROOT = Path("/ephemeral/code/openvla")
sys.path.insert(0, str(OPENVLA_ROOT))

from libero.libero import benchmark  # noqa: E402

TASK_SUITE = "libero_10"
N_TRIALS_PER_TASK = 30
RNG_SEED = 20260817

# Trials that store all 7 decision states unpooled instead of the 3-position summary.
# Three per task rather than 30 on one task, so the per-DoF position ablation is not
# confounded with task identity.
SUBSET_TRIALS_PER_TASK = (0, 1, 2)

# Init-state indices consumed by the existing 50-rollout corpus (gen_rollouts.py used
# `initial_states[ep]` for ep in 0..4). Recorded for transparency, not excluded: v2 is
# meant to be a strict superset of v1, so overlap is harmless and mildly useful.
V1_CONSUMED = tuple(range(5))

OUT_PATH = Path(__file__).resolve().parent / "init_state_assignment.json"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def build(seed: int = RNG_SEED, n_trials: int = N_TRIALS_PER_TASK) -> dict:
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    rng = np.random.default_rng(seed)

    assignment = {}
    for task_id in range(suite.n_tasks):
        n_avail = len(suite.get_task_init_states(task_id))
        if n_trials > n_avail:
            raise ValueError(
                f"task {task_id}: asked for {n_trials} trials but only {n_avail} distinct "
                f"init states exist; distinct rollouts are capped at {n_avail}"
            )
        # Permutation, not sampling with replacement: a repeated index is a duplicate
        # rollout, not a second sample.
        perm = rng.permutation(n_avail)
        chosen = [int(i) for i in perm[:n_trials]]
        assignment[str(task_id)] = {
            "task_name": suite.get_task(task_id).name,
            "task_description": suite.get_task(task_id).language,
            "n_init_states_available": int(n_avail),
            "init_state_indices": chosen,
            "unconsumed": sorted(int(i) for i in perm[n_trials:]),
            "n_positions_by_trial": [
                7 if trial in SUBSET_TRIALS_PER_TASK else 3 for trial in range(n_trials)
            ],
            "also_in_v1_corpus": sorted(set(chosen) & set(V1_CONSUMED)),
        }

    return {
        "schema": "init_state_assignment/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "task_suite": TASK_SUITE,
        "rng_seed": seed,
        "rng": "numpy.random.default_rng(seed).permutation(n_avail)[:n_trials], per task in task order",
        "n_trials_per_task": n_trials,
        "n_positions_default": 3,
        "n_positions_subset": 7,
        "subset_trials_per_task": list(SUBSET_TRIALS_PER_TASK),
        "v1_consumed_indices": list(V1_CONSUMED),
        "assignment": assignment,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_PER_TASK)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = build(args.seed, args.n_trials)

    total = sum(len(v["init_state_indices"]) for v in doc["assignment"].values())
    n_subset = len(doc["subset_trials_per_task"]) * len(doc["assignment"])
    print(f"task_suite={doc['task_suite']} seed={doc['rng_seed']}")
    for task_id, v in doc["assignment"].items():
        idx = v["init_state_indices"]
        print(
            f"  task {int(task_id):02d}  n={len(idx):2d}/{v['n_init_states_available']}  "
            f"first5={idx[:5]}  unconsumed={len(v['unconsumed']):2d}  "
            f"v1_overlap={v['also_in_v1_corpus']}"
        )
    print(f"total rollouts={total}  P=7 subset={n_subset}  P=3={total - n_subset}")

    for task_id, v in doc["assignment"].items():
        assert len(set(v["init_state_indices"])) == len(v["init_state_indices"]), task_id
        assert not (set(v["init_state_indices"]) & set(v["unconsumed"])), task_id

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return

    out = Path(args.out)
    if out.exists():
        raise SystemExit(
            f"{out} already exists. It defines the corpus; refusing to overwrite. "
            "Move it aside deliberately if you really mean to redraw."
        )
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
