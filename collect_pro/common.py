"""
common.py  --  shared config, job list and paths for the more-data collection
(more_data_collection.md, handoff v1, Step 2).

Nothing here imports LIBERO or the model, so the analysis side and status.sh can use it.

Job list
--------
One job per (block, base task, trial). Sorted by (trial, block in the order
ori2, obj, lan, swap, task, task index), so a run cut short is balanced across blocks
and tasks. Worker k takes jobs with job_number % N_WORKERS == k.

Rollout ids: <block>_task<TT>_trial<NN>.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "more_data.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def block_number(cfg: dict, block: str) -> int:
    return cfg["blocks"].index(block)


def per_task(cfg: dict, block: str) -> int:
    return int(cfg[f"PER_TASK_{block.upper()}"])


def rollout_id(block: str, task: int, trial: int) -> str:
    return f"{block}_task{task:02d}_trial{trial:02d}"


def parse_rid(rid: str) -> tuple[str, int, int]:
    m = re.fullmatch(r"([a-z0-9]+)_task(\d\d)_trial(\d\d)", rid)
    return m.group(1), int(m.group(2)), int(m.group(3))


def out_dir(cfg: dict) -> Path:
    return Path(cfg["out_dir"])


def assignment_path(cfg: dict) -> Path:
    return REPO / cfg["assignment_v3"]


def load_assignment(cfg: dict) -> dict:
    return json.loads(assignment_path(cfg).read_text())


def jobs(cfg: dict, assignment: dict) -> list[dict]:
    """Every planned rollout, in job order. Trials beyond a cell's assigned init states
    (a suite with fewer states than asked for) are simply absent."""
    out = []
    for block in cfg["blocks"]:
        for task in range(10):
            idxs = assignment["assignment"][block][str(task)]["init_state_indices"]
            for trial in range(min(per_task(cfg, block), len(idxs))):
                out.append({"block": block, "task": task, "trial": trial,
                            "init_state_index": int(idxs[trial]),
                            "suite": cfg["suites"][block],
                            "rollout_id": rollout_id(block, task, trial)})
    order = {b: i for i, b in enumerate(cfg["blocks"])}
    out.sort(key=lambda j: (j["trial"], order[j["block"]], j["task"]))
    for n, j in enumerate(out):
        j["job_number"] = n
    return out


def is_complete(d: Path) -> bool:
    try:
        m = json.loads((d / "meta.json").read_text())
    except Exception:
        return False
    return bool(m.get("complete")) and all((d / f).exists()
                                           for f in ("hidden.pt", "logits.pt", "actions.npz"))


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def bddl_language(bddl_path: Path) -> str:
    """The instruction in a bddl's (:language ...) line. LIBERO's Task.language is built
    from the file NAME, which for libero_10_lan / libero_10_task is the original
    instruction, so the bddl text is the source of truth."""
    m = re.search(r"\(:language\s+([^)]*)\)", Path(bddl_path).read_text())
    if not m:
        raise ValueError(f"no :language in {bddl_path}")
    return " ".join(m.group(1).split())


def make_assignment(cfg: dict) -> dict:
    """ori2: the 20 indices per task not in collect/init_state_assignment.json, ascending.
    Other blocks: default_rng(1000 + block_number).choice(50, k, replace=False), one rng
    per block, drawn per task in task order (as the v2 assignment did)."""
    v2 = json.loads((REPO / cfg["assignment_v2"]).read_text())
    n_states = 50
    out = {"schema": "init_state_assignment/v3",
           "rule_ori2": "indices of libero_10 init states not used by the v2 assignment, ascending",
           "rule_other": cfg["init_rng"] + ", per task in task order, one rng per block",
           "block_numbers": {b: i for i, b in enumerate(cfg["blocks"])},
           "n_states_per_task": n_states, "assignment": {}}
    for block in cfg["blocks"]:
        k = per_task(cfg, block)
        out["assignment"][block] = {}
        rng = np.random.default_rng(1000 + block_number(cfg, block))
        for task in range(10):
            if block == "ori2":
                used = set(v2["assignment"][str(task)]["init_state_indices"])
                idxs = [i for i in range(n_states) if i not in used][:k]
            else:
                idxs = [int(i) for i in rng.choice(n_states, min(k, n_states), replace=False)]
            out["assignment"][block][str(task)] = {"suite": cfg["suites"][block],
                                                   "init_state_indices": idxs}
    return out
