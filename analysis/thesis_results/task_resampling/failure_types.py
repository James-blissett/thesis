"""
failure_types.py

§5 of task_resampling_handoff.md: which kinds of failure the detectors catch.

Rollouts: every failed original LIBERO-10 rollout among each task split's unseen tasks
(headline setup: full training set, tested on the unseen tasks incl. LIBERO-PRO, so the
cut is the shortest kept length in the rollout's task-and-perturbation group of that test
set), once per split it is tested in.

Failure type from the logged actions, rows t < cut only, first match wins:
  froze           longest run of consecutive act_rep steps >= FREEZE_MIN (act_rep at t:
                  act_mag[t] = |a_t - a_{t-1}| over the 6 normalised arm dims <= 1e-6, the
                  constraint_auroc.ACT_REP_EPS definition; NaN at t = 0 counts as no repeat);
                  freeze start = first step of that run
  never grasped   the gripper is never commanded closed (executed action dim 6 never +1:
                  +1 closes after invert_gripper_action; wait steps send -1 = open)
  other           everything else
Alarm: first kept row with the alpha 0.1 alarm (stored band, bootstrap2.load_seed), counted
only before the cut.

Check by eye: 20 distinct rollouts (seed 20261103), spread over types and tasks; frame
strips (agentview, rotated upright, policy steps 0, cut/3, 2 cut/3, cut - 1) written to
task_resampling/failure_frames/ for reading.

Outputs: failure_types.csv, failure_types_summary.csv, failure_frames/*.jpg,
logs/failure_sample.json (the 20, with the facts the descriptions are written against).
Usage: source env.sh; THESIS_POOL=D2 python analysis/thesis_results/task_resampling/failure_types.py [FREEZE_MIN]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr  # noqa: E402
from tr import NPZ_FULL, OUT, SEEDS, common, log, write_csv  # noqa: E402

FREEZE_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ACT_REP_EPS = 1e-6
WAIT = 10
N_SAMPLE = 20
SAMPLE_SEED = 20261103
DETECTORS = [("A_mean", "A mean"), ("B_mean_h10", "B mean (H = 10)"), ("C_mean", "C mean"),
             ("emb_best", "untrained emb_temp[26]")]
TYPES = ("froze", "never grasped", "other")
FRAMES = OUT / "failure_frames"


def longest_run(flag: np.ndarray) -> tuple[int, int | None]:
    best, start, run = 0, None, 0
    for t, v in enumerate(flag):
        run = run + 1 if v else 0
        if run > best:
            best, start = run, t - run + 1
    return best, start


def classify(i: int, cut: int) -> dict:
    idx = common.ctx().idx
    mag = np.load(idx.constraints_path(i))["act_mag"][:cut]
    rep = np.nan_to_num(mag, nan=np.inf) <= ACT_REP_EPS
    run, start = longest_run(rep)
    a = np.load(idx.rollout_dir(i) / "actions.npz")
    env_t = a["policy_step_env_t"][:cut]
    grip = a["executed"][env_t, 6]
    closed = np.nonzero(grip > 0)[0]
    typ = "froze" if run >= FREEZE_MIN else ("never grasped" if len(closed) == 0 else "other")
    return {"type": typ, "longest repeated-action run": run, "freeze start": start if typ == "froze" else "",
            "longest run start": start if run else "", "gripper ever closed": bool(len(closed)),
            "first close step": int(closed[0]) if len(closed) else ""}


def main() -> None:
    import bootstrap2 as b2
    ctx = common.ctx()
    idx = ctx.idx
    rows = []
    for k, s in enumerate(SEEDS):
        pos = common.sets(s)["T2"]
        cut = dict(zip(pos.tolist(), common.m1_cut(idx, pos).tolist()))
        alarm = {}
        for key, _ in DETECTORS:
            se = b2.load_seed(NPZ_FULL[k], key, "T2")
            alarm[key] = dict(zip(se.rids.tolist(), se.first_alarm.tolist()))
        for i in pos:
            if idx.block[i] != "ori" or idx.y[i] != 1:
                continue
            rid, c = str(idx.rids[i]), int(cut[i])
            row = {"task split": s, "rollout": rid, "task": int(idx.base_task[i]),
                   "task name": json.loads((idx.rollout_dir(i) / "manifest.json").read_text())["task_description"],
                   "cut": c, "kept length": int(idx.kept_len[i]), **classify(i, c)}
            for key, name in DETECTORS:
                a = alarm[key][rid]
                row[f"{name}: alarm step before the cut"] = int(a) if a < c else ""
            rows.append(row)
    write_csv(OUT / "failure_types.csv", rows)
    summary(rows)
    sample(rows)
    log(f"[failures] {len(rows)} rollout-split rows ({len({r['rollout'] for r in rows})} rollouts), "
        f"freeze threshold {FREEZE_MIN}")


def summary(rows) -> None:
    out = []
    for typ in TYPES + ("all failures",):
        R = [r for r in rows if typ in (r["type"], "all failures")]
        row = {"failure type": typ, "rollouts (rollout x split)": len(R),
               "distinct rollouts": len({r["rollout"] for r in R}),
               "rollouts per split": " ".join(str(sum(r["task split"] == s for r in R)) for s in SEEDS)}
        for _, name in DETECTORS:
            col = f"{name}: alarm step before the cut"
            caught = [r for r in R if r[col] != ""]
            row[f"{name}: share caught before the cut (alpha 0.1)"] = round(len(caught) / len(R), 3) if R else ""
            row[f"{name}: caught (rollout x split)"] = len(caught)
            if typ == "froze":
                gaps = [r[col] - r["freeze start"] for r in caught]
                row[f"{name}: caught, alarm before the freeze starts (share of caught)"] = \
                    round(float(np.mean([g < 0 for g in gaps])), 3) if gaps else ""
                row[f"{name}: caught, median alarm step minus freeze start"] = float(np.median(gaps)) if gaps else ""
        out.append(row)
    write_csv(OUT / "failure_types_summary.csv", out)


def sample(rows) -> None:
    """20 distinct rollouts, spread over types (as even as availability allows) and, within
    a type, over tasks (round-robin in a random task order)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(SAMPLE_SEED)
    first = {}
    for r in rows:
        first.setdefault(r["rollout"], r)
    by = {t: [r for r in first.values() if r["type"] == t] for t in TYPES}
    want = {t: 0 for t in TYPES}
    left = N_SAMPLE
    while left:
        open_ = [t for t in TYPES if want[t] < len(by[t])]
        if not open_:
            break
        for t in open_:
            if left and want[t] < len(by[t]):
                want[t] += 1
                left -= 1
    pick = []
    for t in TYPES:
        tasks = {}
        for r in by[t]:
            tasks.setdefault(r["task"], []).append(r)
        order = list(tasks)
        rng.shuffle(order)
        for v in tasks.values():
            rng.shuffle(v)
        got = []
        while len(got) < want[t]:
            for task in order:
                if tasks[task] and len(got) < want[t]:
                    got.append(tasks[task].pop())
        pick += got
    FRAMES.mkdir(parents=True, exist_ok=True)
    idx = common.ctx().idx
    where = {str(r): i for i, r in enumerate(idx.rids)}
    info = []
    for r in pick:
        i = where[r["rollout"]]
        c = r["cut"]
        steps = [0, c // 3, 2 * c // 3, c - 1]
        ims = []
        for t in steps:
            im = Image.open(idx.rollout_dir(i) / "frames" / f"agentview_{t + WAIT:04d}.jpg").rotate(180)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 92, 14], fill=(0, 0, 0))
            d.text((3, 2), f"t={t} / cut {c}", fill=(255, 255, 255))
            ims.append(im)
        w, h = ims[0].size
        strip = Image.new("RGB", (w * 4, h))
        for k, im in enumerate(ims):
            strip.paste(im, (k * w, 0))
        f = FRAMES / f"{r['rollout']}.jpg"
        strip.save(f, quality=90)
        info.append({**{k: r[k] for k in ("rollout", "task", "task name", "type", "cut", "longest repeated-action run",
                                          "longest run start", "gripper ever closed", "first close step")},
                     "frame steps": steps, "strip": str(f.relative_to(tr.fc.OUT))})
    common.write_json(OUT / "logs" / "failure_sample.json",
                      {"seed": SAMPLE_SEED, "per type": want, "freeze threshold": FREEZE_MIN, "rollouts": info})


if __name__ == "__main__":
    main()
