"""
common.py

Shared plumbing for the thesis results collection (results_collection_handoff.md,
version 4). Everything here is a thin layer over analysis/abc and analysis/abc2, which
are imported, never copied or edited:

  * one combined-corpus context (pipe2_common.Ctx on corpus_v3_index.json, splits from
    results/abc2/splits, which equal results/abc/splits on the original rollouts);
  * the training set ("pool"): the original training set (repo key D0) for the main run,
    the full training set (D2) for the background run (env THESIS_POOL=D2);
  * that pool's selections from results/abc2 (beta per family, c values, A's probe), so
    every definition stays the one the earlier runs used;
  * plain names for test sets (handoff section 0) and for files.

Rollout order everywhere is corpus_v3_index.json's: the 300 original rollouts first.
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "abc2"))
sys.path.insert(1, str(REPO / "analysis" / "abc"))
sys.path.insert(2, str(REPO / "analysis"))
import data2  # noqa: E402
import pipe2_common as p2  # noqa: E402
from data2 import T, m1_cut, train_rows_and_weights  # noqa: E402,F401
from pipe2_common import m1_row, m1_safe, mean_std, read_csv, write_csv, save_pkl, load_pkl  # noqa: E402,F401
import setup_c as abc_c  # noqa: E402

POOL = os.environ.get("THESIS_POOL", "D0")
if POOL not in ("D0", "D2"):
    raise SystemExit(f"THESIS_POOL must be D0 or D2, not {POOL}")
MAIN = POOL == "D0"
OUT_DIR = REPO / ("results/thesis_results" if MAIN else "results/thesis_results_full")
WORK_DIR = Path("/data/tmp/thesis_results_work") / ("original" if MAIN else "full")
CACHE_DIR = data2.CACHE_DIR                      # /data/tmp/abc2_features
SEEDS = (0, 1, 2)
C_GRID = abc_c.C_GRID
HS = (0, 10, 25)
HEADLINE_H = 10
ALPHA_TABLE = 0.1

POOL_NAME = {"D0": "original training set", "D2": "full training set"}[POOL]
POOL_SLUG = {"D0": "original_training_set", "D2": "full_training_set"}[POOL]

# Test sets: repo key -> plain name (handoff section 0).
TEST_SETS = {"eval_seen": "eval-seen",
             "T0": "unseen tasks, original rollouts",
             "T2": "unseen tasks incl. LIBERO-PRO",
             "stress": "stress set"}
TEST_SLUG = {"eval_seen": "eval_seen", "T0": "unseen_original", "T2": "unseen_incl_libero_pro",
             "stress": "stress", "train": "train"}
AUROC_SETS = ("eval_seen", "T0", "T2")           # sets with an AUROC
PER_TASK_LABEL = {"eval_seen": "per-task AUROC", "T0": "per-task AUROC",
                  "T2": "per-task-and-perturbation AUROC"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@lru_cache(maxsize=1)
def ctx() -> p2.Ctx:
    """The combined-corpus context (Ctx makes its out/work dirs; both are scratch)."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return p2.Ctx(index_path=REPO / data2.INDEX_V3, cache_dir=CACHE_DIR,
                  splits_dir=REPO / data2.SPLITS_DIR, out_dir=WORK_DIR / "ctx_out",
                  work_dir=WORK_DIR / "ctx_work", demotions_path=REPO / data2.DEMOTIONS,
                  seeds=SEEDS)


def sets(seed: int) -> dict:
    """Positions: train, eval_seen, cal, T0, T1, T2, stress_seen, stress_unseen, plus
    stress = stress_seen + stress_unseen (every position- and task-perturbed rollout)."""
    s = dict(ctx().sets(POOL, seed))
    s["stress"] = np.sort(np.concatenate([s["stress_seen"], s["stress_unseen"]]))
    return s


@lru_cache(maxsize=1)
def selections() -> dict:
    """This pool's selections from results/abc2 (the more-data run): beta per family,
    the emb_temp comparison layer (pooled rule), c for C's probes / C's final / A, and
    A's probe per seed. These are the definitions the earlier runs used."""
    rows = [r for r in read_csv(REPO / "results/abc2/selections.csv") if r["cond"] == POOL]
    out = {"betas": {}, "a_probe": {}}
    for r in rows:
        k, v = r["selection"], r["value"]
        if k.startswith("beta_"):
            out["betas"][k[5:]] = v
        elif k == "emb_layer":
            out["emb_layer_pooled_rule"] = int(v.split("[")[1].rstrip("]"))
        elif k in ("c_C_probes", "c_C_final", "c_A"):
            out[k] = float(v)
        elif k == "A_probe":
            out["a_probe"][int(r["seed"])] = v
    return out


def load_layer(L):
    """(N, 520, 4096) fp16 hidden state at stored index L on the GPU (or the action
    tokens as float32 for L = 'action_tokens'), via stage4_a's loader."""
    import stage4_a
    name = L if isinstance(L, str) else f"layer{int(L):02d}"
    return stage4_a.load_probe_input(name, CACHE_DIR)


@lru_cache(maxsize=None)
def b_targets_all() -> dict:
    """B's prepared target series for every seed (stage5_b.prepared_targets at this
    pool's betas): {(seed, series): (N, 520) float32, NaN off kept rows}."""
    import stage5_b
    return stage5_b.prepared_targets(ctx(), POOL, SEEDS, selections()["betas"])


def b_targets(seed: int) -> dict:
    return {k[1]: v for k, v in b_targets_all().items() if k[0] == seed}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)
