"""
check_rollout.py  --  Step 3.2 structural check of v3 rollouts.

Per rollout: hidden.pt (520, 33, 4096) fp16, finite, no all-zero layer; logits.pt
(520, 7, 256) fp16 finite; action_token_ids (520, 7) int32; success_now 530 rows; 520
policy steps even if it succeeded early; frames open and are not black (mean pixel of
agentview frame 0 > 5); meta fields present.

Usage: python collect_pro/check_rollout.py RID [RID ...] [--root /data/rollouts_v3]
       python collect_pro/check_rollout.py --all
Exits 1 if any rollout fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

META_FIELDS = ("rollout_id", "block", "suite", "base_task_idx", "init_state_index", "instruction",
               "success_ever", "success_final", "t_success", "wall_clock_seconds", "bytes")


def check(d: Path) -> list[str]:
    errs = []
    m = json.loads((d / "meta.json").read_text())
    errs += [f"meta missing {k}" for k in META_FIELDS if k not in m]
    h = torch.load(str(d / "hidden.pt"), map_location="cpu", mmap=True)
    if tuple(h.shape) != (520, 33, 4096) or h.dtype != torch.float16:
        errs.append(f"hidden {tuple(h.shape)} {h.dtype}")
    else:
        absmax = h.float().abs().amax(dim=(0, 2))
        dead = [i for i, v in enumerate(absmax.tolist()) if v == 0.0]
        if dead:
            errs.append(f"all-zero layers {dead}")
        if not torch.isfinite(h).all():
            errs.append("hidden non-finite")
    lg = torch.load(str(d / "logits.pt"), map_location="cpu")
    if tuple(lg.shape) != (520, 7, 256) or lg.dtype != torch.float16 or not torch.isfinite(lg).all():
        errs.append(f"logits {tuple(lg.shape)} {lg.dtype}")
    a = np.load(d / "actions.npz")
    if a["action_token_ids"].shape != (520, 7) or a["action_token_ids"].dtype != np.int32:
        errs.append(f"action_token_ids {a['action_token_ids'].shape}")
    if a["success_now"].shape != (530,) or a["executed"].shape != (530, 7):
        errs.append("env-step arrays not 530 rows")
    if m.get("T_total") != 520:
        errs.append(f"T_total {m.get('T_total')}")
    if m["success_ever"] != bool(a["success_now"].any()):
        errs.append("success_ever inconsistent with success_now")
    frames = sorted((d / "frames").glob("*.jpg"))
    want = 2 * 530 if m.get("frames") == "all" else 2
    if len(frames) != want:
        errs.append(f"{len(frames)} frames, expected {want}")
    f0 = d / "frames" / "agentview_0000.jpg"
    try:
        mean = float(np.asarray(Image.open(f0)).mean())
        if mean <= 5:
            errs.append(f"frame 0 black (mean {mean:.1f})")
    except Exception as e:
        errs.append(f"frame 0 unreadable: {e!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rids", nargs="*")
    ap.add_argument("--root", default="/data/rollouts_v3")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    rids = [p.name for p in sorted(root.glob("*_task*_trial*")) if not p.name.startswith(".")] if args.all else args.rids
    bad = 0
    for rid in rids:
        errs = check(root / rid)
        m = json.loads((root / rid / "meta.json").read_text())
        print(f"{rid}: {'OK' if not errs else 'FAIL ' + '; '.join(errs)} | success_ever={m['success_ever']} "
              f"instruction={m['instruction']!r}")
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
