"""
check_parity.py  --  Step 3.1 environment parity (more_data_collection.md, handoff v1)

Compares a re-run of a v2 rollout through collect_pro/collect.py --parity with the stored
v2 rollout:
  * action tokens identical for the first 50 policy steps, and in >= 95% of all steps
    (a step counts as identical when all 7 tokens match);
  * pooled hidden state at t = 0: cosine > 0.999 at every one of the 33 stored indices
    (v2 pooled read with compute_constraints.pooled_and_hN, for P = 3 and P = 7 alike).

Usage:
    python collect_pro/check_parity.py task00_trial00 [--new-dir /data/tmp/abc2_parity]
Prints a JSON report and exits 1 if parity fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from compute_constraints import pooled_and_hN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("--new-dir", default="/data/tmp/abc2_parity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old = Path("/data/rollouts_v2") / args.rid
    new = Path(args.new_dir) / args.rid
    m_old = json.loads((old / "manifest.json").read_text())
    m_new = json.loads((new / "meta.json").read_text())

    tok_old = np.load(old / "actions.npz")["action_token_ids"]
    tok_new = np.load(new / "actions.npz")["action_token_ids"]
    same = (tok_old == tok_new).all(1)
    first_diff = int(np.argmin(same)) if not same.all() else None

    h_old_raw = torch.load(str(old / "hidden.pt"), map_location="cpu", mmap=True)
    pooled_old, _ = pooled_and_hN(h_old_raw[:1], int(m_old["n_positions"]))  # (1, 33, 4096)
    h_new = torch.load(str(new / "hidden.pt"), map_location="cpu", mmap=True)
    a, b = pooled_old[0].float(), h_new[0].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).numpy()

    rep = {
        "rid": args.rid,
        "steps_identical_first50": int(same[:50].sum()),
        "share_steps_identical": float(same.mean()),
        "first_differing_step": first_diff,
        "cos_t0_min": float(cos.min()),
        "cos_t0_per_layer": [round(float(c), 6) for c in cos],
        "max_abs_diff_t0": float((a - b).abs().max()),
        "bitwise_equal_t0": bool(torch.equal(pooled_old[0].half(), h_new[0])),
        "success_ever_old_new": [m_old["success_ever"], m_new["success_ever"]],
        "t_success_old_new": [m_old["t_success"], m_new["t_success"]],
        "instruction_new": m_new["instruction"], "instruction_old": m_old["task_description"],
    }
    rep["pass"] = bool(same[:50].all() and same.mean() >= 0.95 and cos.min() > 0.999)
    txt = json.dumps(rep, indent=1)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
