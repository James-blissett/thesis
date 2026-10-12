"""
perturbation_check.py  --  Step 3.3: is each perturbation really applied?

Render-only (no model). For every base task and perturbed block, frame 0 (the
observation right after env.reset(); env.set_init_state(s), i.e. what collect.py stores
as agentview_0000.jpg) at the block's first assigned init state, against libero_10's
frame 0 at the first two ori2 init states of the same task.

  * lan, task: the instruction (bddl :language) must differ from the original task's.
  * obj, swap: mean absolute pixel difference to the unperturbed frame (d_pert, mean of
    the distances to both reference states) must be well above the difference between
    two unperturbed initial states of that task (d_base). "Well above" is taken as
    d_pert >= RATIO_MIN * d_base; all numbers are reported.

Outputs (results/abc2/perturbation_check/):
    <block>_task<TT>.png   side by side: unperturbed | perturbed (upright, agentview)
    sheet.png              all (block, task) pairs on one page
    summary.csv / summary.json
Usage (env from collect_pro/env_pro.sh):  python collect_pro/perturbation_check.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from collect import EnvCache  # noqa: E402

OUT = common.REPO / "results" / "abc2" / "perturbation_check"
RATIO_MIN = 1.5
BLOCKS = ("obj", "lan", "swap", "task")


def frame0(envs: EnvCache, suite: str, task: int, init_idx: int) -> tuple[np.ndarray, str]:
    env = envs.get(suite, task)
    env.reset()
    obs = env.set_init_state(envs.init_states[init_idx])
    return np.asarray(obs["agentview_image"])[::-1, ::-1].copy(), envs.instruction


def label(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(img).resize((256, 256))
    canvas = Image.new("RGB", (256, 276), "white")
    canvas.paste(im, (0, 20))
    ImageDraw.Draw(canvas).text((4, 4), text[:42], fill="black")
    return canvas


def main():
    cfg = common.load_config()
    asg = common.load_assignment(cfg)["assignment"]
    OUT.mkdir(parents=True, exist_ok=True)
    envs = EnvCache()
    rows, tiles = [], []
    for task in range(10):
        ori_idx = asg["ori2"][str(task)]["init_state_indices"][:2]
        ref_a, instr0 = frame0(envs, "libero_10", task, ori_idx[0])
        ref_b, _ = frame0(envs, "libero_10", task, ori_idx[1])
        d_base = float(np.abs(ref_a.astype(float) - ref_b).mean())
        for block in BLOCKS:
            idx = asg[block][str(task)]["init_state_indices"][0]
            img, instr = frame0(envs, cfg["suites"][block], task, idx)
            d_pert = float(np.mean([np.abs(img.astype(float) - r).mean() for r in (ref_a, ref_b)]))
            instr_diff = instr != instr0
            if block in ("lan", "task"):
                ok = instr_diff
            else:
                ok = d_pert >= RATIO_MIN * d_base
            rows.append({"block": block, "task": task, "init_state_index": idx,
                         "d_pert": round(d_pert, 3), "d_base": round(d_base, 3),
                         "ratio": round(d_pert / max(d_base, 1e-9), 2),
                         "instruction_original": instr0, "instruction_block": instr,
                         "instruction_differs": instr_diff, "pass": bool(ok)})
            pair = Image.new("RGB", (512, 276), "white")
            pair.paste(label(ref_a, f"libero_10 t{task} init {ori_idx[0]}"), (0, 0))
            pair.paste(label(img, f"{block} t{task} init {idx} r={d_pert / max(d_base, 1e-9):.1f}"), (256, 0))
            pair.save(OUT / f"{block}_task{task:02d}.png")
            tiles.append(pair)
            print(f"task {task} {block:5s} d_pert {d_pert:6.2f} d_base {d_base:6.2f} "
                  f"instr_diff {instr_diff} pass {ok}", flush=True)
    envs.close()

    # contact sheet: rows = tasks, columns = blocks, each tile half size
    w, h = 256, 138
    sheet = Image.new("RGB", (w * len(BLOCKS), h * 10), "white")
    for i, tile in enumerate(tiles):
        r, c = divmod(i, len(BLOCKS))
        sheet.paste(tile.resize((w, h)), (c * w, r * h))
    sheet.save(OUT / "sheet.png")

    with open(OUT / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    n_fail = sum(not r["pass"] for r in rows)
    print(f"[perturbation_check] {len(rows) - n_fail}/{len(rows)} pass")


if __name__ == "__main__":
    main()
