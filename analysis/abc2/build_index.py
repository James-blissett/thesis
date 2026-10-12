"""
build_index.py

Step 5 of the more-data run (more_data_collection.md, handoff v1): the combined rollout
index, the corpus audit, the obj / lan demotion rule and the lan near-duplicate check.
Reads only meta.json / actions.npz files and the collection logs; no model, no GPU.

Outputs
-------
  corpus_v3_index.json   (repo root; --index-out)  via data2.write_index: the 300 v2
                         rollouts first in corpus_v2_index.json order, then every finished
                         v3 rollout in job order.
  under results/abc2/ (--out-dir):
  audit_block_task.csv   per (block, task): planned, collected, successes, failures,
                         success rate, failed (raised twice, logged in failures.jsonl),
                         not_run, and whether the cell was skipped by the 3-failures rule
  audit_block.csv        the same per block (block ori = the 300 v2 rollouts)
  audit.md               both tables plus collection hours, rollouts/hour, failures
  demotions.json         {"demoted": [...], "rates": {...}, "counts": {...}, "rule": ...};
                         obj / lan outside [10%, 90%] overall success become stress-only.
                         Always written (empty list if nothing is demoted).
  lan_neardup.csv        one row per lan rollout: its unperturbed partner (ori or ori2, same
                         base task and init_state_index) and the share of the 520 policy
                         steps whose 7 action tokens are all identical
  lan_neardup.json       n, mean / median share, and the reference: the same statistic for
                         every pair of unperturbed rollouts of one base task at different
                         init states (LIBERO-PRO's lan init states are fresh draws, not the
                         libero_10 scenes; see RUNLOG Step 1.4, so the pairs are not the
                         same scene and the figure must be read against that reference)

Usage (from the repo root; the script also chdirs there):
    source env.sh
    python analysis/abc2/build_index.py                        # real outputs
    python analysis/abc2/build_index.py --index-out /data/tmp/x/idx.json --out-dir /data/tmp/x
    python analysis/abc2/build_index.py --selftest             # writes nothing; checks
        data2's index / split / condition / test-set code on the real v2 + finished v3
        rollouts and on a synthetic full-size v3 corpus (every planned job)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "abc2"))
sys.path.insert(1, str(REPO))
import data2  # noqa: E402
from collect_pro import common  # noqa: E402

DEMOTE_LOW, DEMOTE_HIGH = 0.10, 0.90
DEMOTABLE = ("obj", "lan")
UNPERTURBED = ("ori", "ori2")
NEARDUP_CAVEAT = (
    "LIBERO-PRO's libero_10_lan init states are fresh draws, not the libero_10 states "
    "(RUNLOG Step 1.4: no row of any perturbed suite's init file matches a libero_10 "
    "state). A lan rollout and its 'partner' (same base task, same init_state_index) "
    "therefore do NOT share a scene; the handoff's near-duplicate premise does not hold. "
    "Read the pair figure against `reference` (unperturbed pairs of one task at different "
    "init states) and `lan_vs_nonpartner`: if they are about equal, the lan block is not "
    "a near-duplicate of the unperturbed rollouts.")


# --- helpers --------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def cell_skipped(progress: list[dict], failures: list[dict], block: str, task: int) -> bool:
    """collect_pro/collect.py's rule: a cell whose three most recent events (finished
    rollouts in progress.jsonl, give-ups in failures.jsonl) are all failures is skipped."""
    ev = [(r.get("finished_utc", ""), True) for r in progress
          if r.get("block") == block and r.get("base_task_idx") == task]
    ev += [(r.get("finished_utc", ""), False) for r in failures
           if r.get("block") == block and r.get("base_task_idx") == task]
    n = 0
    for _, ok in sorted(ev, reverse=True):
        if ok:
            break
        n += 1
    return n >= 3


def planned_cells() -> dict:
    """{(block, task): planned n}; ori from collect/init_state_assignment.json."""
    cfg = common.load_config()
    plan = Counter((j["block"], j["task"]) for j in common.jobs(cfg, common.load_assignment(cfg)))
    v2 = json.loads((REPO / cfg["assignment_v2"]).read_text())["assignment"]
    for t in range(10):
        plan[("ori", t)] = len(v2[str(t)]["init_state_indices"])
    return dict(plan)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def rate(s: int, n: int):
    return round(s / n, 4) if n else None


# --- audit ----------------------------------------------------------------------------
def audit(idx: data2.Index2, out_dir: Path, rollouts_v3: Path) -> dict:
    plan = planned_cells()
    progress = read_jsonl(rollouts_v3 / "progress.jsonl")
    failures = read_jsonl(rollouts_v3 / "failures.jsonl")
    have = set(idx.rids.tolist())
    failed_rids = defaultdict(set)          # gave up after two attempts, never completed
    stops = []
    for r in failures:
        if r.get("fatal"):
            stops.append(r)
            continue
        if r.get("rollout_id") not in have:
            failed_rids[(r.get("block"), r.get("base_task_idx"))].add(r.get("rollout_id"))

    cell_rows = []
    for block in data2.BLOCKS:
        for task in range(10):
            m = (idx.block == block) & (idx.base_task == task)
            n, s = int(m.sum()), int((m & (idx.y == 0)).sum())
            p = plan.get((block, task), 0)
            nf = len(failed_rids[(block, task)])
            skipped = block != "ori" and n < p and cell_skipped(progress, failures, block, task)
            cell_rows.append({"block": block, "task": task, "planned": p, "collected": n,
                              "successes": s, "failures": n - s, "success_rate": rate(s, n),
                              "failed_rollouts": nf, "not_run": max(p - n - nf, 0),
                              "cell_skipped": int(skipped)})
    block_rows = []
    for block in data2.BLOCKS:
        rs = [r for r in cell_rows if r["block"] == block]
        n, s = sum(r["collected"] for r in rs), sum(r["successes"] for r in rs)
        block_rows.append({"block": block, "role": "stress" if block in data2.STRESS_BLOCKS else "training",
                           "planned": sum(r["planned"] for r in rs), "collected": n,
                           "successes": s, "failures": n - s, "success_rate": rate(s, n),
                           "failed_rollouts": sum(r["failed_rollouts"] for r in rs),
                           "not_run": sum(r["not_run"] for r in rs),
                           "cells_skipped": sum(r["cell_skipped"] for r in rs)})
    new = [r for r in block_rows if r["block"] != "ori"]
    block_rows.append({"block": "all_new", "role": "", "planned": sum(r["planned"] for r in new),
                       "collected": sum(r["collected"] for r in new),
                       "successes": sum(r["successes"] for r in new),
                       "failures": sum(r["failures"] for r in new),
                       "success_rate": rate(sum(r["successes"] for r in new), sum(r["collected"] for r in new)),
                       "failed_rollouts": sum(r["failed_rollouts"] for r in new),
                       "not_run": sum(r["not_run"] for r in new),
                       "cells_skipped": sum(r["cells_skipped"] for r in new)})

    # Collection time: collect_started_utc -> last finished_utc in progress.jsonl.
    timing = {"collect_started_utc": None, "last_finished_utc": None, "hours": None,
              "n_finished_in_window": None, "rollouts_per_hour": None,
              "n_finished_before_start": None,
              "collect_hours_max": common.load_config()["COLLECT_HOURS_MAX"]}
    start_p = rollouts_v3 / "collect_started_utc"
    fin = {r["rollout_id"]: r["finished_utc"] for r in progress if r.get("finished_utc")}
    if start_p.exists() and fin:
        start = parse_utc(start_p.read_text())
        last = max(parse_utc(v) for v in fin.values())
        hours = (last - start).total_seconds() / 3600
        n_in = sum(parse_utc(v) >= start for v in fin.values())
        timing.update(collect_started_utc=start.isoformat(), last_finished_utc=last.isoformat(),
                      hours=round(hours, 3), n_finished_in_window=n_in,
                      n_finished_before_start=len(fin) - n_in,
                      rollouts_per_hour=round(n_in / hours, 2) if hours > 0 else None)

    write_csv(out_dir / "audit_block_task.csv", cell_rows)
    write_csv(out_dir / "audit_block.csv", block_rows)
    fail_lines = [f"- `{r.get('rollout_id')}` ({r.get('block')}, task {r.get('base_task_idx')}): "
                  f"{str(r.get('error', ''))[:160]}" for r in failures]
    skipped = [f"({r['block']}, task {r['task']})" for r in cell_rows if r["cell_skipped"]]
    md = ["# Corpus audit (more_data_collection.md Step 5)", "",
          "Success = `success_ever`. Block `ori` = the 300 v2 rollouts. `failed_rollouts`: raised "
          "twice and logged in failures.jsonl; `not_run`: planned but never attempted (cell "
          "skipped, COLLECT_HOURS_MAX or disk stop).", "",
          "## Per block", "", md_table(block_rows), "",
          "## Collection", "",
          f"- started {timing['collect_started_utc']}, last rollout finished {timing['last_finished_utc']}",
          f"- hours {timing['hours']}, rollouts finished in that window {timing['n_finished_in_window']} "
          f"(+ {timing['n_finished_before_start']} smoke-test rollouts before the start), "
          f"rollouts/hour {timing['rollouts_per_hour']}",
          f"- COLLECT_HOURS_MAX {timing['collect_hours_max']} h; cap reached (last finish at or after the cap): "
          f"{'yes' if timing['hours'] is not None and timing['hours'] >= timing['collect_hours_max'] else 'no'}",
          f"- cells skipped (3 failures in a row): {', '.join(skipped) if skipped else 'none'}",
          f"- collection stops logged (disk headroom): {len(stops)}",
          f"- failures.jsonl entries: {len(failures)}" + (":" if failures else ""), *fail_lines, "",
          "## Per (block, task)", "", md_table(cell_rows), ""]
    (out_dir / "audit.md").write_text("\n".join(md))
    return {"cells": cell_rows, "blocks": block_rows, "timing": timing,
            "n_failures_logged": len(failures), "skipped_cells": skipped}


# --- demotion -------------------------------------------------------------------------
def demotions(idx: data2.Index2, out_dir: Path) -> dict:
    rates, counts, demoted = {}, {}, []
    for b in data2.BLOCKS:
        m = idx.block == b
        n, s = int(m.sum()), int((m & (idx.y == 0)).sum())
        rates[b], counts[b] = rate(s, n), {"successes": s, "n": n}
    for b in DEMOTABLE:
        if rates[b] is not None and (rates[b] < DEMOTE_LOW or rates[b] > DEMOTE_HIGH):
            demoted.append(b)
    doc = {"demoted": demoted, "rates": rates, "counts": counts,
           "rule": (f"obj or lan with overall success rate (success_ever) < {DEMOTE_LOW:.0%} or "
                    f"> {DEMOTE_HIGH:.0%} becomes stress-test only (handoff v1 Step 5); a block "
                    "with no rollouts is not evaluated")}
    (out_dir / "demotions.json").write_text(json.dumps(doc, indent=1))
    return doc


# --- lan near-duplicate check ---------------------------------------------------------
def action_tokens(idx: data2.Index2, positions) -> np.ndarray:
    return np.stack([np.load(idx.rollout_dir(int(i)) / "actions.npz")["action_token_ids"]
                     for i in positions]).astype(np.int32)        # (n, 520, 7)


def same_step_share(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of policy steps whose 7 action tokens are all identical. Broadcasts over
    leading axes; a, b (..., 520, 7)."""
    return (a == b).all(-1).mean(-1)


def summary_stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 4),
            "median": round(float(np.median(x)), 4),
            "p10": round(float(np.percentile(x, 10)), 4), "p90": round(float(np.percentile(x, 90)), 4)}


def lan_neardup(idx: data2.Index2, out_dir: Path) -> dict:
    unp = np.nonzero(idx.blocks_mask(UNPERTURBED))[0]
    lan = np.nonzero(idx.block == "lan")[0]
    tok_u = action_tokens(idx, unp) if len(unp) else np.zeros((0, data2.T, 7), np.int32)
    tok_l = action_tokens(idx, lan) if len(lan) else np.zeros((0, data2.T, 7), np.int32)

    rows, pair_share, nonpartner, multi = [], [], [], 0
    for j, i in enumerate(lan):
        same_task = idx.base_task[unp] == idx.base_task[i]
        partner = np.nonzero(same_task & (idx.init_state_index[unp] == idx.init_state_index[i]))[0]
        others = np.nonzero(same_task & (idx.init_state_index[unp] != idx.init_state_index[i]))[0]
        if len(others):
            nonpartner.append(float(same_step_share(tok_l[j][None], tok_u[others]).mean()))
        row = {"lan_rollout": str(idx.rids[i]), "base_task": int(idx.base_task[i]),
               "init_state_index": int(idx.init_state_index[i]), "lan_success": int(idx.y[i] == 0),
               "partner": "", "partner_block": "", "partner_success": "",
               "share_identical_steps": "", "share_identical_tokens": "", "first_diff_step": "",
               "mean_share_vs_other_init_states": round(nonpartner[-1], 4) if len(others) else ""}
        if len(partner):
            multi += len(partner) > 1
            k = partner[0]
            a, b = tok_l[j], tok_u[k]
            eq_step = (a == b).all(-1)
            share = float(eq_step.mean())
            pair_share.append(share)
            row.update(partner=str(idx.rids[unp[k]]), partner_block=str(idx.block[unp[k]]),
                       partner_success=int(idx.y[unp[k]] == 0), share_identical_steps=round(share, 4),
                       share_identical_tokens=round(float((a == b).mean()), 4),
                       first_diff_step=int(np.argmin(eq_step)) if not eq_step.all() else data2.T)
        rows.append(row)

    # Reference: every pair of unperturbed rollouts of one base task at different init states.
    ref = []
    for task in range(10):
        p = np.nonzero(idx.base_task[unp] == task)[0]
        if len(p) < 2:
            continue
        S = same_step_share(tok_u[p][:, None], tok_u[p][None])     # (n, n)
        diff_init = idx.init_state_index[unp][p][:, None] != idx.init_state_index[unp][p][None]
        iu = np.triu_indices(len(p), 1)
        ref += S[iu][diff_init[iu]].tolist()

    if rows:
        write_csv(out_dir / "lan_neardup.csv", rows)
    else:
        (out_dir / "lan_neardup.csv").write_text("lan_rollout\n")
    doc = {"statistic": "share of the 520 policy steps whose 7 action_token_ids are all identical "
                        "(step t vs step t)",
           "n_lan": int(len(lan)), "n_with_partner": len(pair_share),
           "n_with_multiple_partners": int(multi),
           "lan_vs_partner": summary_stats(pair_share),
           "lan_vs_nonpartner": {**summary_stats(nonpartner),
                                 "what": "per lan rollout, the mean share against every unperturbed "
                                         "rollout of the same base task at a different init state"},
           "reference": {**summary_stats(ref),
                         "what": "all pairs of unperturbed (ori + ori2) rollouts of the same base task "
                                 "at different init states"},
           "caveat": NEARDUP_CAVEAT}
    doc["mean_share"] = doc["lan_vs_partner"]["mean"]
    doc["median_share"] = doc["lan_vs_partner"]["median"]
    (out_dir / "lan_neardup.json").write_text(json.dumps(doc, indent=1))
    return doc


# --- self-test (writes nothing) -------------------------------------------------------
def synthetic_index(seed: int = 0) -> data2.Index2:
    """The 300 real v2 entries + one synthetic v3 entry per planned job (real
    init_state_index values), random outcomes, t_success in [100, 519] for successes."""
    rng = np.random.default_rng(seed)
    cfg = common.load_config()
    p_succ = {"ori2": 0.45, "obj": 0.35, "lan": 0.45, "swap": 0.03, "task": 0.03}
    entries = data2.v2_entries()
    for j in common.jobs(cfg, common.load_assignment(cfg)):
        ok = bool(rng.random() < p_succ[j["block"]])
        entries.append({"rollout_id": j["rollout_id"], "block": j["block"], "base_task_idx": j["task"],
                        "init_state_index": j["init_state_index"], "success_ever": ok,
                        "t_success": int(rng.integers(100, 520)) if ok else None,
                        "source": "v3", "n_positions": 1})
    return data2.index_from_entries(entries)


def selftest() -> None:
    from data import Index as AbcIndex, load_index as abc_load_index, m1 as abc_m1
    fails = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # v2-only Index2 must reproduce abc's kept rows and per-task M1.
    v2 = data2.index_from_entries(data2.v2_entries())
    kept = data2.kept_mask(v2)
    n_s, n_f = int(kept[v2.y == 0].sum()), int(kept[v2.y == 1].sum())
    print("[selftest] v2-only Index2")
    check((n_s, n_f) == (38_618, 84_597), f"kept_mask success rows {n_s} (38618), failure rows {n_f} (84597)")
    old = abc_load_index()
    check(np.array_equal(old.rids, v2.rids) and np.array_equal(old.task, v2.task)
          and np.array_equal(old.y, v2.y) and np.array_equal(old.kept_len, v2.kept_len),
          "v2 Index2 rids/task/y/kept_len equal abc's Index (task = cell = base task)")
    check(all(e["instruction"] for e in data2.v2_entries()), "every v2 entry has an instruction")
    rs = np.random.default_rng(1).standard_normal((v2.n, data2.T))
    for s in data2.SEEDS:
        osplit = json.loads((data2.OLD_SPLITS_DIR / f"seed{s}.json").read_text())
        pu = old.pos(osplit["unseen"])
        a, b = abc_m1(rs, kept, old, pu), data2.m1(rs, kept, v2, pu)
        check(a == b, f"seed {s}: m1 on T0 with Index2 == abc Index ({a['per_task']:.4f})")

    for name, idx in (("real (v2 + finished v3)", data2.index_from_entries(
                           data2.v2_entries() + data2.v3_entries())),
                      ("synthetic (v2 + every planned v3 job)", synthetic_index())):
        print(f"[selftest] {name}: {idx.n} rollouts, " + ", ".join(
            f"{b} {int((idx.block == b).sum())}" for b in data2.BLOCKS))
        kept = data2.kept_mask(idx)
        for s in data2.SEEDS:
            osplit = json.loads((data2.OLD_SPLITS_DIR / f"seed{s}.json").read_text())
            sp = data2.make_split2(s, idx)
            sp["pos"] = {k: idx.pos(sp[k]) for k in ("train", "eval_seen", "unseen", "stress")}
            P = sp["pos"]
            print(f"  seed {s}: train {len(P['train'])}, eval_seen {len(P['eval_seen'])}, "
                  f"unseen {len(P['unseen'])}, stress {len(P['stress'])}; new: "
                  f"{sp['n_new_inherited_side']} inherited, {sp['n_new_drawn_side']} drawn")
            check(sp["unseen_tasks"] == osplit["unseen_tasks"], f"seed {s}: unseen tasks {sp['unseen_tasks']} = old")
            allp = np.concatenate(list(P.values()))
            check(len(allp) == idx.n and len(set(allp.tolist())) == idx.n,
                  f"seed {s}: every rollout in exactly one of train/eval_seen/unseen/stress")
            # one side per base task
            side = np.empty(idx.n, dtype=object)
            side[P["train"]] = side[P["eval_seen"]] = "seen"
            side[P["unseen"]] = "unseen"
            side[P["stress"]] = np.where(np.isin(idx.base_task[P["stress"]], sp["unseen_tasks"]),
                                         "unseen", "seen")
            ok = all(len(set(side[idx.base_task == t])) == 1 for t in range(10))
            ok &= all((side[idx.base_task == t][0] == "unseen") == (t in sp["unseen_tasks"]) for t in range(10))
            check(ok, f"seed {s}: every rollout of a base task (any block) on one side")
            check(set(idx.base_task[P["unseen"]].tolist()) <= set(sp["unseen_tasks"])
                  and not set(idx.base_task[np.concatenate([P["train"], P["eval_seen"]])].tolist())
                  & set(sp["unseen_tasks"]), f"seed {s}: train/eval_seen only seen tasks, unseen only unseen")
            for k in ("train", "eval_seen", "unseen"):
                got = sorted(r for r in sp[k] if r in set(old.rids.tolist()))
                check(got == sorted(osplit[k]), f"seed {s}: v2 {k} membership unchanged ({len(got)})")
            g_tr = {(int(idx.base_task[i]), int(idx.init_state_index[i])) for i in P["train"]}
            g_ev = {(int(idx.base_task[i]), int(idx.init_state_index[i])) for i in P["eval_seen"]}
            check(not g_tr & g_ev, f"seed {s}: no (base task, init) group in both train and eval_seen")
            # drawn groups 60/40 per task
            v2_groups = {(int(v2.base_task[i]), int(v2.init_state_index[i])) for i in range(v2.n)}
            bad = []
            for t in sp["seen_tasks"]:
                ntr = len({g for g in g_tr if g[0] == t and g not in v2_groups})
                nev = len({g for g in g_ev if g[0] == t and g not in v2_groups})
                if ntr + nev and ntr != int(round(0.6 * (ntr + nev))):
                    bad.append((t, ntr, nev))
            drawn = [(t, len({g for g in g_tr if g[0] == t and g not in v2_groups}),
                      len({g for g in g_ev if g[0] == t and g not in v2_groups})) for t in sp["seen_tasks"]]
            check(not bad, f"seed {s}: drawn groups train/eval = round(0.6 n) per task "
                           f"{[f'{t}:{a}/{b}' for t, a, b in drawn]}")
            ok_inh = True
            v2_side = {}
            for k in ("train", "eval_seen"):
                for r in osplit[k]:
                    i = v2.pos([r])[0]
                    v2_side[(int(v2.base_task[i]), int(v2.init_state_index[i]))] = k
            for k in ("train", "eval_seen"):
                for i in P[k]:
                    g = (int(idx.base_task[i]), int(idx.init_state_index[i]))
                    ok_inh &= v2_side.get(g, k) == k
            check(ok_inh, f"seed {s}: new rollouts in a v2 group inherit the v2 rollout's side")
            check(set(idx.block[P["stress"]].tolist()) <= set(data2.STRESS_BLOCKS)
                  and not np.isin(idx.block[np.concatenate([P["train"], P["eval_seen"], P["unseen"]])],
                                  data2.STRESS_BLOCKS).any(),
                  f"seed {s}: swap/task rollouts only (and all) in stress")
            # conditions and test sets, without and with a fake demotion
            for dem in ((), ("lan",), ("obj", "lan")):
                for cond, blocks in data2.CONDITIONS.items():
                    cs = data2.condition_sets(sp, idx, cond, demotions=dem)
                    allowed = set(blocks) - set(dem)
                    got = set(idx.block[np.concatenate([cs["train"], cs["eval_seen"]])].tolist())
                    exp_tr = P["train"][np.isin(idx.block[P["train"]], list(allowed))]
                    ok = got <= allowed and np.array_equal(cs["train"], exp_tr)
                    ok &= set(cs["cal"].tolist()) == {i for i in cs["eval_seen"].tolist() if idx.y[i] == 0}
                    if cond == "D0":
                        ok &= sorted(idx.rids[cs["train"]]) == sorted(osplit["train"])
                        ok &= sorted(idx.rids[cs["eval_seen"]]) == sorted(osplit["eval_seen"])
                    check(ok, f"seed {s} dem={list(dem)} {cond}: blocks {sorted(got)} (allowed {sorted(allowed)}), "
                              f"train {len(cs['train'])}, eval_seen {len(cs['eval_seen'])}, cal {len(cs['cal'])}")
                ts = data2.test_sets(sp, idx, demotions=dem)
                ok = sorted(idx.rids[ts["T0"]]) == sorted(osplit["unseen"])
                ok &= set(ts["T1"].tolist()) <= set(ts["T2"].tolist())
                ok &= set(idx.block[ts["T1"]].tolist()) <= {"ori", "ori2"}
                ok &= set(idx.block[ts["T2"]].tolist()) <= set(data2.TRAINING_BLOCKS) - set(dem)
                ok &= np.array_equal(np.sort(ts["T2"]), np.sort(P["unseen"][~np.isin(idx.block[P["unseen"]], list(dem))]))
                sb = list(data2.STRESS_BLOCKS) + list(dem)
                st = np.concatenate([ts["stress_seen"], ts["stress_unseen"]])
                ok &= len(st) == len(set(st.tolist())) and set(st.tolist()) == set(np.nonzero(np.isin(idx.block, sb))[0].tolist())
                ok &= np.isin(idx.base_task[ts["stress_unseen"]], sp["unseen_tasks"]).all()
                ok &= not np.isin(idx.base_task[ts["stress_seen"]], sp["unseen_tasks"]).any()
                check(ok, f"seed {s} dem={list(dem)}: T0 = old unseen ({len(ts['T0'])}), T1 {len(ts['T1'])} "
                          f"subset of T2 {len(ts['T2'])}, stress seen/unseen {len(ts['stress_seen'])}/"
                          f"{len(ts['stress_unseen'])} partition {sb} by base task")
            # per-cell M1 with an Index2: a timestep-only score gives exactly 0.5 per cell
            ts = data2.test_sets(sp, idx, demotions=())
            tscore = np.broadcast_to(np.arange(data2.T, dtype=float), (idx.n, data2.T))
            for tn in ("T0", "T2"):
                pos = ts[tn]
                cut = data2.m1_cut(idx, pos)
                ok_cut = all(cut[idx.task[pos] == c].min() == idx.kept_len[pos][idx.task[pos] == c].min()
                             for c in np.unique(idx.task[pos]))
                try:
                    r = data2.m1(tscore, kept, idx, pos)
                    from sklearn.metrics import roc_auc_score
                    cells = [c for c in np.unique(idx.task[pos])
                             if min(((idx.task[pos] == c) & (idx.y[pos] == 0)).sum(),
                                    ((idx.task[pos] == c) & (idx.y[pos] == 1)).sum()) >= 2]
                    mask = kept[pos] & (np.arange(data2.T)[None] < cut[:, None])
                    top = np.where(mask, tscore[pos], -np.inf).max(-1)
                    per_cell = [roc_auc_score(idx.y[pos][idx.task[pos] == c], top[idx.task[pos] == c]) for c in cells]
                    check(ok_cut and r["per_task"] == 0.5 and all(v == 0.5 for v in per_cell)
                          and r["n_tasks"] == len(cells),
                          f"seed {s} {tn}: timestep-only M1 per cell = 0.5 in all {r['n_tasks']} cells "
                          f"({sorted({data2.cell_name(c).split('/')[1] for c in cells})}), pooled floor {r['floor']:.3f}")
                except ValueError as e:
                    check(name.startswith("real") and tn == "T2", f"seed {s} {tn}: m1 raised {e}")
    print(f"[selftest] {'PASSED' if not fails else f'{len(fails)} FAILED'}")
    if fails:
        raise SystemExit(1)


# --- main -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=data2.RESULTS_DIR)
    ap.add_argument("--index-out", type=Path, default=data2.INDEX_V3)
    ap.add_argument("--rollouts-v3", type=Path, default=data2.ROLLOUTS_V3)
    ap.add_argument("--selftest", action="store_true", help="run the checks; writes nothing")
    args = ap.parse_args()
    os.chdir(REPO)
    if args.selftest:
        selftest()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc = data2.write_index(args.index_out, args.rollouts_v3)
    idx = data2.load_index2(args.index_out)
    assert idx.n == doc["n_rollouts"] and list(idx.rids[:300]) == list(
        data2.index_from_entries(data2.v2_entries()).rids), "v2 rollouts must come first, in order"
    assert len(set(idx.rids.tolist())) == idx.n, "duplicate rollout ids"
    a = audit(idx, args.out_dir, args.rollouts_v3)
    d = demotions(idx, args.out_dir)
    nd = lan_neardup(idx, args.out_dir)

    print(f"[index] {args.index_out}: {idx.n} rollouts ({doc['n_v2']} v2, {doc['n_v3']} v3)")
    for r in a["blocks"]:
        print(f"  {r['block']:7s} {r['collected']:4d}/{r['planned']:4d} collected, "
              f"success {r['successes']}/{r['collected']} ({r['success_rate']}), "
              f"failed {r['failed_rollouts']}, not run {r['not_run']}, cells skipped {r['cells_skipped']}")
    t = a["timing"]
    print(f"[collect] {t['hours']} h, {t['n_finished_in_window']} rollouts in window, "
          f"{t['rollouts_per_hour']}/h; failures.jsonl {a['n_failures_logged']} entries; "
          f"skipped cells {a['skipped_cells'] or 'none'}")
    print(f"[demotion] demoted {d['demoted'] or 'none'} (obj {d['rates']['obj']}, lan {d['rates']['lan']})")
    print(f"[lan near-dup] {nd['n_with_partner']}/{nd['n_lan']} lan rollouts with a partner: "
          f"identical-step share mean {nd['lan_vs_partner']['mean']} median {nd['lan_vs_partner']['median']}; "
          f"vs other init states {nd['lan_vs_nonpartner']['mean']}; reference (unperturbed, "
          f"different init states, {nd['reference']['n']} pairs) mean {nd['reference']['mean']} "
          f"median {nd['reference']['median']}")
    print(f"[*] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
