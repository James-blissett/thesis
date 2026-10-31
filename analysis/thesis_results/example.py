"""
example.py

Example rollouts for the alarm figure (results_collection_handoff.md §4.8; F1).

Seed 0, one unseen task (original rollouts). If seed 0 has no unseen task with both an
alarmed failure and an unalarmed success (B's mean catches no unseen failure in seed 0
at alpha 0.1), the next seed that has one is used and the fallback is recorded. Candidates: failures that B's unweighted
mean (H = 10) alarms on before the cut at alpha 0.1, and successes from the same task it
does not alarm on. Rule (fixed before looking at frames): among unseen tasks with at
least one such failure and one such success, prefer the task whose alarmed failures
show the clearest stall, measured as the longest run of repeated actions (act_rep)
before the alarm; take that failure, and the success of the same task with the
latest first-success step (so its trace runs as long as possible before the cut).

Saved (figures/data/example_rollouts.csv, one row per (rollout, t)): B's mean score
(running sum), the alpha 0.1 band, A's and C's mean scores (running sums) and their own
bands, the kept flag, the cut, the alarm step; plus figures/data/example_rollouts.json
(ids, reasons, cut, alarm steps, frame paths) and 4 frames per rollout (start, mid, alarm
step or the cut's midpoint for the success, cut) copied to figures/example_frames/.
Frame file for policy step t is frames/agentview_{t + 10:04d}.jpg (10 wait steps first).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import detectors as det  # noqa: E402
import evaluate as ev  # noqa: E402
from common import T, log, m1_cut, write_csv  # noqa: E402

SEED = 0
WAIT = 10


def first_alarm(score, band, kept_row, cut):
    rows = kept_row & (np.arange(T) < cut)
    with np.errstate(invalid="ignore"):
        a = rows & (score > band)
    return int(np.argmax(a)) if a.any() else None


def longest_repeat_before(i, upto) -> int:
    from constraint_auroc import ACT_REP_EPS
    idx = common.ctx().idx
    mag = np.load(idx.constraints_path(i))["act_mag"][:upto]
    rep = np.nan_to_num(mag, nan=np.inf) <= ACT_REP_EPS
    best = run = 0
    for v in rep:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def main() -> None:
    for seed in common.SEEDS:
        if pick(seed):
            return
    log("[example] no seed has a candidate")


def pick(SEED) -> bool:
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    st = common.sets(SEED)
    pos = st["T0"]
    cut = dict(zip(pos.tolist(), m1_cut(idx, pos).tolist()))
    E = {k: det.load(k)["seeds"][SEED] for k in ("B_mean_h10", "A_mean", "C_mean")}
    bands = {k: ev.eval_key(k)["bands"][SEED] for k in E}
    band = {k: b["mu"] + b["h"][-1] * b["xi"] for k, b in bands.items()}
    cand = {}
    for i in pos:
        a = first_alarm(E["B_mean_h10"]["cp"][i], band["B_mean_h10"], kept[i], cut[i])
        cand[i] = a
    tasks = sorted(set(idx.base_task[pos].tolist()))
    choice = []
    for t in tasks:
        fail = [i for i in pos if idx.base_task[i] == t and idx.y[i] == 1 and cand[i] is not None]
        succ = [i for i in pos if idx.base_task[i] == t and idx.y[i] == 0 and cand[i] is None]
        if fail and succ:
            stall = {i: longest_repeat_before(i, cand[i]) for i in fail}
            f = max(fail, key=lambda i: (stall[i], -cand[i]))
            s = max(succ, key=lambda i: idx.kept_len[i])
            choice.append((stall[f], t, f, s))
    if not choice:
        log(f"[example] no unseen task in seed {SEED} has both an alarmed failure and an unalarmed success")
        return False
    choice.sort(reverse=True)
    stall, task, f, s = choice[0]
    info = {"seed": SEED, "seed note": "seed 0 as specified" if SEED == 0 else
            f"FALLBACK: seed 0 has no unseen failure that B's mean alarms on before the cut at alpha 0.1; seed {SEED} used",
            "task": task, "rule": __doc__.split("Saved")[0].strip().split("Rule")[1].strip(),
            "candidates (longest repeat run before alarm, task, failure, success)":
                [(int(a), int(b), str(idx.rids[c]), str(idx.rids[d])) for a, b, c, d in choice],
            "rollouts": {}}
    rows = []
    fdir = common.OUT_DIR / "figures" / "example_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, role in ((f, "failure"), (s, "success")):
        c = cut[i]
        a = cand[i]
        steps = {"start": 1, "mid": int(c // 2), "alarm" if a is not None else "mid-way to cut": a if a is not None else int(3 * c // 4),
                 "cut": int(c - 1)}
        frames = {}
        for nm, tstep in steps.items():
            src = idx.rollout_dir(i) / "frames" / f"agentview_{tstep + WAIT:04d}.jpg"
            dst = fdir / f"{role}_{idx.rids[i]}_{nm.replace(' ', '_')}_t{tstep:03d}.jpg"
            if src.exists():
                shutil.copyfile(src, dst)
                frames[nm] = {"t": tstep, "file": str(dst.relative_to(common.OUT_DIR))}
        info["rollouts"][role] = {"rollout": str(idx.rids[i]), "task": task, "outcome": role,
                                  "kept length": int(idx.kept_len[i]), "cut": int(c), "B mean alarm step": a,
                                  "A mean alarm step": first_alarm(E["A_mean"]["cp"][i], band["A_mean"], kept[i], c),
                                  "C mean alarm step": first_alarm(E["C_mean"]["cp"][i], band["C_mean"], kept[i], c),
                                  "longest action-repeat run before the alarm": stall if role == "failure" else None,
                                  "frames": frames}
        for t in range(T):
            rows.append({"seed": SEED, "rollout": str(idx.rids[i]), "outcome": role, "t": t, "kept": bool(kept[i, t]),
                         "before cut": t < c,
                         "B mean score (running sum, H = 10)": round(float(E["B_mean_h10"]["cp"][i, t]), 4),
                         "B mean band (alpha 0.1)": round(float(band["B_mean_h10"][t]), 4) if np.isfinite(band["B_mean_h10"][t]) else "",
                         "A mean score (running sum)": round(float(E["A_mean"]["cp"][i, t]), 4),
                         "A mean band (alpha 0.1)": round(float(band["A_mean"][t]), 4) if np.isfinite(band["A_mean"][t]) else "",
                         "C mean score (running sum)": round(float(E["C_mean"]["cp"][i, t]), 4),
                         "C mean band (alpha 0.1)": round(float(band["C_mean"][t]), 4) if np.isfinite(band["C_mean"][t]) else "",
                         "cut": int(c), "B mean alarm step": "" if a is None else a})
    info["why"] = (f"task {task}: the alarmed failure with the longest run of repeated actions before B's alarm "
                   f"({stall} steps), i.e. the clearest stall; the success of the same task with the latest "
                   f"first-success step that B does not alarm on")
    write_csv(common.OUT_DIR / "figures" / "data" / "example_rollouts.csv", rows)
    common.write_json(common.OUT_DIR / "figures" / "data" / "example_rollouts.json", info)
    log(f"[example] seed {SEED} task {task}: failure {idx.rids[f]} (alarm {cand[f]}, stall {stall}), success {idx.rids[s]}")
    return True


if __name__ == "__main__":
    main()
