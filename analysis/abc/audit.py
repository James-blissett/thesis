"""
audit.py

Stage 0 of the A/B/C build: confirm the corpus is there and complete, print the stored
shapes for one P = 3 and one P = 7 rollout, the kept-row counts per class, git status,
and whether SAFE's conformal-prediction code is in the OpenVLA fork.

Reads only manifests, the index, and two hidden.pt files (memory-mapped). No model.

There is deliberately no __init__.py in analysis/abc/: a regular package called `abc`
would shadow the standard library's `abc` module.

Usage (from the repo root):
    source env.sh
    python analysis/abc/audit.py
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CORPUS_ROOT = Path("/data/rollouts_v2")
INDEX_JSON = Path("corpus_v2_index.json")
OPENVLA_FORK = Path("/ephemeral/code/openvla")
OUT = Path("results/abc/audit.json")

T = 520
N_STORED = 33
D_MODEL = 4096
N_EXPECTED = 300


def corpus_check(entries: list[dict]) -> dict:
    """Every indexed rollout has a complete manifest and every data file at the right size."""
    problems = []
    p_counts = Counter()
    for e in entries:
        rid = e["rollout_id"]
        d = CORPUS_ROOT / rid
        man_path = d / "manifest.json"
        if not man_path.exists():
            problems.append(f"{rid}: no manifest")
            continue
        man = json.loads(man_path.read_text())
        if not man.get("complete"):
            problems.append(f"{rid}: manifest not complete")
        if man["n_positions"] != e["n_positions"]:
            problems.append(f"{rid}: n_positions manifest {man['n_positions']} != index {e['n_positions']}")
        if man["hidden_shape"] != [T, N_STORED, man["n_positions"], D_MODEL]:
            problems.append(f"{rid}: hidden_shape {man['hidden_shape']}")
        for fname in ("hidden.pt", "actions.npz", "logits.pt"):
            f = d / fname
            if not f.exists():
                problems.append(f"{rid}: missing {fname}")
            elif f.stat().st_size != man["file_sizes_bytes"][fname]:
                problems.append(f"{rid}: {fname} is {f.stat().st_size} B, manifest says "
                                f"{man['file_sizes_bytes'][fname]}")
        if not (d / "frames").is_dir():
            problems.append(f"{rid}: no frames/")
        p_counts[man["n_positions"]] += 1
    dirs = sorted(p.name for p in CORPUS_ROOT.iterdir() if p.is_dir())
    return {
        "n_dirs": len(dirs),
        "n_indexed": len(entries),
        "unindexed_dirs": sorted(set(dirs) - {e["rollout_id"] for e in entries}),
        "positions": {str(k): v for k, v in sorted(p_counts.items())},
        "problems": problems,
    }


def shapes(rid: str) -> dict:
    d = CORPUS_ROOT / rid
    x = torch.load(str(d / "hidden.pt"), map_location="cpu", mmap=True)
    lg = torch.load(str(d / "logits.pt"), map_location="cpu", mmap=True)
    a = np.load(d / "actions.npz")
    return {
        "hidden.pt": [list(x.shape), str(x.dtype)],
        "logits.pt": [list(lg.shape), str(lg.dtype)],
        **{f"actions.npz:{k}": [list(a[k].shape), str(a[k].dtype)] for k in a.files},
    }


def kept_counts(entries: list[dict]) -> dict:
    """Kept rows per class, before and after dropping t = 0.

    Successes keep t < t_success; failures keep all T steps (handoff section 2).
    """
    succ = [e for e in entries if e["success_ever"]]
    fail = [e for e in entries if not e["success_ever"]]
    s_rows = sum(int(e["t_success"]) for e in succ)
    f_rows = len(fail) * T
    return {
        "n_success": len(succ), "n_failure": len(fail),
        "success_rows_incl_t0": s_rows, "failure_rows_incl_t0": f_rows,
        "success_rows": s_rows - len(succ), "failure_rows": f_rows - len(fail),
    }


def git_status() -> str:
    return subprocess.run(["git", "status", "--short", "--branch"], capture_output=True,
                          text=True).stdout.strip()


def conformal_search() -> dict:
    """grep -ril 'conformal' over the fork, .git excluded."""
    r = subprocess.run(["grep", "-rIil", "--exclude-dir=.git", "conformal", str(OPENVLA_FORK)],
                       capture_output=True, text=True)
    head = subprocess.run(["git", "-C", str(OPENVLA_FORK), "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout.strip()
    return {"fork": str(OPENVLA_FORK), "fork_head": head,
            "files_matching_conformal": [l for l in r.stdout.splitlines() if l]}


def main() -> None:
    entries = json.loads(INDEX_JSON.read_text())["rollouts"]

    if not CORPUS_ROOT.is_dir():
        raise SystemExit(f"{CORPUS_ROOT} is gone. Stages 2 and 3 can still run from "
                         "constraints/ alone; A and B cannot.")

    corpus = corpus_check(entries)
    print(f"[corpus] {corpus['n_dirs']} dirs, {corpus['n_indexed']} indexed, "
          f"positions {corpus['positions']}, unindexed dirs {corpus['unindexed_dirs']}")
    print(f"[corpus] problems: {corpus['problems'] if corpus['problems'] else 'none'}")
    ok = corpus["n_indexed"] == N_EXPECTED and not corpus["problems"]

    p3 = next(e["rollout_id"] for e in entries if e["n_positions"] == 3)
    p7 = next(e["rollout_id"] for e in entries if e["n_positions"] == 7)
    sh = {p3: shapes(p3), p7: shapes(p7)}
    for rid, s in sh.items():
        print(f"[shapes] {rid}")
        for k, v in s.items():
            print(f"    {k:28s} {v[0]} {v[1]}")

    kc = kept_counts(entries)
    print(f"[kept] {kc['n_success']} successes: {kc['success_rows']} rows "
          f"({kc['success_rows_incl_t0']} incl. t=0)")
    print(f"[kept] {kc['n_failure']} failures:  {kc['failure_rows']} rows "
          f"({kc['failure_rows_incl_t0']} incl. t=0)")

    gs = git_status()
    print(f"[git]\n{gs}")

    cp = conformal_search()
    print(f"[conformal] {cp['fork']} @ {cp['fork_head']}: "
          f"{cp['files_matching_conformal'] or 'no file mentions conformal'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"corpus_ok": ok, "corpus": corpus, "shapes": sh,
                               "kept": kc, "git_status": gs, "conformal": cp}, indent=1))
    print(f"[*] corpus {'OK' if ok else 'NOT OK'} -> {OUT}")


if __name__ == "__main__":
    main()
