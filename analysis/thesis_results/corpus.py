"""
corpus.py

Corpus and confounds (results_collection_handoff.md §4.7; T2, F3, F4):

  composition   all 1,000 rollouts by source x task x outcome (from corpus_v3_index.json;
                checked against results/abc2/report_tables/corpus_block.csv)
  kept rows     original rollouts: number of kept rollouts per class against t (all 300
                original rollouts, and the training split averaged over seeds), and the
                completion-time distribution of successes
  action repeat share of steps with a repeated action (act_rep: act_mag <= ACT_REP_EPS)
                before vs after the first success, on successful rollouts; original
                rollouts (must equal results/completion_confound.json) and every source
  length check  mean score against t on unseen failures and unseen successes (kept rows,
                unseen tasks, original rollouts), per seed then averaged, for A's, C's and
                B's (H = 10) unweighted means (running sums) and the timestep-only score;
                a point is drawn while at least 5 rollouts of that class are still kept

Outputs: tables/table2_corpus_composition.csv, tables/data/corpus_by_source_task.csv,
figures/data/kept_rollouts_against_t.csv, figures/data/success_completion_times.csv,
figures/data/action_repeat_before_after_success.csv, figures/data/length_check.csv,
tables/data/reproduction_checks_corpus.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import detectors as det  # noqa: E402
from common import REPO, SEEDS, T, log, read_csv, write_csv  # noqa: E402

SOURCE = {"ori": "LIBERO-10, original", "ori2": "LIBERO-10, new unperturbed",
          "obj": "LIBERO-PRO object perturbation", "lan": "LIBERO-PRO instruction perturbation",
          "swap": "LIBERO-PRO position swap", "task": "LIBERO-PRO task perturbation"}
ROLE = {"ori": "training role", "ori2": "training role", "obj": "training role", "lan": "training role",
        "swap": "stress set only", "task": "stress set only"}
FIG = common.OUT_DIR / "figures" / "data"


def composition():
    idx = common.ctx().idx
    rows, long = [], []
    for b in ("ori", "ori2", "obj", "lan", "swap", "task"):
        m = idx.block == b
        row = {"source": SOURCE[b], "role": ROLE[b], "rollouts": int(m.sum()),
               "successes": int((m & (idx.y == 0)).sum()), "failures": int((m & (idx.y == 1)).sum()),
               "success rate": round(float((idx.y[m] == 0).mean()), 3)}
        for t in range(10):
            mt = m & (idx.base_task == t)
            row[f"task {t} successes / rollouts"] = f"{int((mt & (idx.y == 0)).sum())}/{int(mt.sum())}"
            long.append({"source": SOURCE[b], "task": t, "rollouts": int(mt.sum()),
                         "successes": int((mt & (idx.y == 0)).sum()), "failures": int((mt & (idx.y == 1)).sum())})
        rows.append(row)
    m = np.ones(idx.n, bool)
    rows.append({"source": "all", "role": "", "rollouts": idx.n, "successes": int((idx.y == 0).sum()),
                 "failures": int((idx.y == 1).sum()), "success rate": round(float((idx.y == 0).mean()), 3),
                 **{f"task {t} successes / rollouts": f"{int(((idx.base_task == t) & (idx.y == 0)).sum())}/{int((idx.base_task == t).sum())}"
                    for t in range(10)}})
    write_csv(common.OUT_DIR / "tables" / "table2_corpus_composition.csv", rows)
    write_csv(common.OUT_DIR / "tables" / "data" / "corpus_by_source_task.csv", long)
    ref = {r["block"]: r for r in read_csv(REPO / "results/abc2/report_tables/corpus_block.csv")}
    chk = []
    for b, r in zip(("ori", "ori2", "obj", "lan", "swap", "task"), rows):
        chk.append({"check": f"composition {SOURCE[b]}", "source_file": "results/abc2/report_tables/corpus_block.csv",
                    "source_value": f"{ref[b]['successes']}/{ref[b]['collected']}",
                    "this_run": f"{r['successes']}/{r['rollouts']}",
                    "passed": int(ref[b]["successes"]) == r["successes"] and int(ref[b]["collected"]) == r["rollouts"]})
    return chk


def kept_curves():
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    ori = np.nonzero(idx.block == "ori")[0]
    rows = []
    for t in range(T):
        r = {"t": t,
             "original rollouts, successes still kept": int(kept[ori[idx.y[ori] == 0], t].sum()),
             "original rollouts, failures still kept": int(kept[ori[idx.y[ori] == 1], t].sum())}
        for s in SEEDS:
            tr = common.sets(s)["train"]
            r[f"training split seed {s}, successes still kept"] = int(kept[tr[idx.y[tr] == 0], t].sum())
            r[f"training split seed {s}, failures still kept"] = int(kept[tr[idx.y[tr] == 1], t].sum())
        rows.append(r)
    write_csv(FIG / "kept_rollouts_against_t.csv", rows)
    succ = ori[idx.y[ori] == 0]
    write_csv(FIG / "success_completion_times.csv",
              [{"rollout": str(idx.rids[i]), "task": int(idx.base_task[i]), "first success step": int(idx.t_success[i])}
               for i in succ])


def action_repeat():
    from constraint_auroc import ACT_REP_EPS
    idx = common.ctx().idx
    rows, chk = [], []
    groups = {SOURCE[b]: np.nonzero((idx.block == b) & (idx.y == 0))[0] for b in ("ori", "ori2", "obj", "lan", "task")}
    groups["all sources"] = np.nonzero(idx.y == 0)[0]
    for name, pos in groups.items():
        pre, post = [], []
        for i in pos:
            mag = np.load(idx.constraints_path(i))["act_mag"]
            rep = np.where(np.isnan(mag), np.nan, (mag <= ACT_REP_EPS).astype(float))
            ts = int(idx.t_success[i])
            pre.append(rep[1:ts])
            post.append(rep[ts:])
        pre = np.concatenate(pre) if pre else np.array([])
        post = np.concatenate(post) if post else np.array([])
        pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
        rows.append({"successful rollouts": name, "rollouts": len(pos),
                     "steps before first success": len(pre),
                     "action-repeat rate before first success": round(float(pre.mean()), 4) if len(pre) else "",
                     "steps from first success on": len(post),
                     "action-repeat rate from first success on": round(float(post.mean()), 4) if len(post) else ""})
    ref = json.loads((REPO / "results/completion_confound.json").read_text())["pre_post"]["act_rep"]
    r0 = rows[0]
    chk.append({"check": "action-repeat rate, original rollouts", "source_file": "results/completion_confound.json",
                "source_value": f"{ref['pre_mean']:.4f} ({ref['pre_rows']} rows) / {ref['post_mean']:.4f} ({ref['post_rows']} rows)",
                "this_run": f"{r0['action-repeat rate before first success']} ({r0['steps before first success']} rows) / "
                            f"{r0['action-repeat rate from first success on']} ({r0['steps from first success on']} rows)",
                "passed": abs(r0["action-repeat rate before first success"] - ref["pre_mean"]) < 1e-4
                and r0["steps before first success"] == ref["pre_rows"] and r0["steps from first success on"] == ref["post_rows"]})
    write_csv(FIG / "action_repeat_before_after_success.csv", rows)
    return chk


def length_check():
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    keys = {"A_mean": "A, unweighted mean (running sum)", "C_mean": "C, unweighted mean (running sum)",
            "B_mean_h10": "B, unweighted mean, H = 10 (running sum)", "timestep": "timestep only (t)"}
    rows = []
    curves = {}
    for k, lab in keys.items():
        E = det.load(k)["seeds"]
        for cls, yv in (("failures", 1), ("successes", 0)):
            per = []
            for s in SEEDS:
                pos = common.sets(s)["T0"]
                sel = pos[idx.y[pos] == yv]
                x = np.where(kept[sel], E[s]["m1"][sel], np.nan)
                n = np.isfinite(x).sum(0)
                with np.errstate(invalid="ignore"):
                    per.append(np.where(n >= 5, np.nanmean(x, 0), np.nan))
            curves[(lab, cls)] = np.array(per)
    for t in range(T):
        r = {"t": t}
        for (lab, cls), v in curves.items():
            r[f"{lab}, {cls}, mean over seeds"] = round(float(np.nanmean(v[:, t])), 5) if np.isfinite(v[:, t]).any() else ""
            for j, s in enumerate(SEEDS):
                r[f"{lab}, {cls}, seed {s}"] = round(float(v[j, t]), 5) if np.isfinite(v[j, t]) else ""
        rows.append(r)
    write_csv(FIG / "length_check.csv", rows)


def main() -> None:
    chk = composition()
    kept_curves()
    chk += action_repeat()
    length_check()
    write_csv(common.OUT_DIR / "tables" / "data" / "reproduction_checks_corpus.csv", chk)
    log(f"[corpus] done; checks {sum(c['passed'] for c in chk)}/{len(chk)} passed")


if __name__ == "__main__":
    main()
