"""
collect.py  --  LIBERO-PRO collection worker (more_data_collection.md, handoff v1, Step 2)

Same model path and settings as collect/gen_rollouts_v2.py (imported, not copied: model
loading, predict_and_capture, the logit/token alignment check), with these differences:

  * suite and initial state are arguments (from the job list in common.py);
  * stored per rollout under /data/rollouts_v3/<rollout_id>/:
      hidden.pt    (520, 33, 4096) fp16, the mean over the 7 decision states only
                   (exactly v2's P=3 position 0: pool_positions(decision, 3)[:, 0])
      logits.pt    (520, 7, 256) fp16, as v2
      actions.npz  as v2 (executed, action_raw, action_token_ids, policy_step_env_t,
                   success_now, success_latched, env_done, is_wait_step, t_effective)
      obs.npz      as v2 (non-image observations; 0.4 MB, kept for diagnostics)
      meta.json    rollout_id, block, suite, base_task_idx, init_state_index,
                   instruction, success_ever, success_final, t_success, wall-clock
                   seconds, bytes, ... ; written last with complete: true
      frames/      every env step (both views) for trial 0 of each (block, task);
                   env step 0 only for the rest
  * the instruction is the bddl's (:language ...) line (see common.bddl_language);
  * written to <out>/.tmp_<rid>_w<k>/ and renamed on completion; one line per finished
    rollout appended to <out>/progress.jsonl;
  * a rollout that raises is retried once, then logged to <out>/failures.jsonl and
    skipped; a (block, task) cell whose last three outcomes are all failures is skipped;
  * no new rollout is started after COLLECT_HOURS_MAX from <out>/collect_started_utc,
    or when free disk < HEADROOM_GB.

Usage (env from collect_pro/env_pro.sh):
    python collect_pro/collect.py --worker 0 --n-workers 2
    python collect_pro/collect.py --only obj_task00_trial00 [--out-dir DIR]
    python collect_pro/collect.py --parity task00_trial05 --out-dir /data/tmp/abc2_parity
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "collect"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import gen_rollouts_v2 as g2  # noqa: E402  (imports libero: needs env_pro.sh)
from capture_v2 import (FRAME_VIEWS, N_ACTION_TOKENS, STORE_DTYPE, encode_jpeg,  # noqa: E402
                        extract_state_obs)
from libero.libero import benchmark, get_libero_path  # noqa: E402
from experiments.robot.libero.libero_utils import (get_libero_dummy_action,  # noqa: E402
                                                   get_libero_env, get_libero_image)
from experiments.robot.robot_utils import (invert_gripper_action,  # noqa: E402
                                           normalize_gripper_action, set_seed_everywhere)

N_POOL = 3            # predict_and_capture's P=3 layout; position 0 is the 7-state mean


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_episode(vla, processor, env, init_state, instruction, frames_dir, all_frames,
                bin_slice, vocab_size, max_steps, n_wait):
    """gen_rollouts_v2.run_episode with pooled-only storage and frame thinning."""
    env.reset()
    obs = env.set_init_state(init_state)
    frames_dir.mkdir(parents=True, exist_ok=True)

    hidden, logits, tok_ids, env_t = [], [], [], []
    executed, action_raw_all, is_wait, obs_list = [], [], [], []
    success_now, success_latched_l, env_done = [], [], []
    frame_means = []
    align = g2.AlignmentAccumulator()
    latched, t_success_env = False, None
    t0 = time.time()

    for t_env in range(n_wait + max_steps):
        wait = t_env < n_wait
        if all_frames or t_env == 0:
            for view, key in FRAME_VIEWS.items():
                if key in obs:
                    (frames_dir / f"{view}_{t_env:04d}.jpg").write_bytes(encode_jpeg(obs[key]))
        frame_means.append(float(np.mean(obs["agentview_image"])))
        obs_list.append(extract_state_obs(obs))

        if wait:
            action_env = np.asarray(get_libero_dummy_action("openvla"), dtype=np.float32)
            action_raw_all.append(np.full(N_ACTION_TOKENS, np.nan, dtype=np.float32))
        else:
            img = get_libero_image(obs, g2.RESIZE_SIZE)
            a_raw, h, lg, tk, _, _ = g2.predict_and_capture(
                vla, processor, img, instruction, N_POOL, bin_slice)
            align.add(g2.logit_alignment(lg, tk, vocab_size), t_env)
            hidden.append(h[:, 0].contiguous())                  # (33, 4096) mean
            logits.append(lg)
            tok_ids.append(np.asarray(tk, dtype=np.int32))
            env_t.append(t_env)
            action_raw_all.append(np.asarray(a_raw, dtype=np.float32))
            action_env = np.array(a_raw, dtype=np.float32, copy=True)
            action_env = normalize_gripper_action(action_env, binarize=True)
            action_env = invert_gripper_action(action_env)
        executed.append(np.asarray(action_env, dtype=np.float32))
        is_wait.append(wait)

        obs, _, done, _ = env.step(np.asarray(action_env).tolist())
        succ = bool(env.check_success())
        latched = latched or succ
        if succ and t_success_env is None:
            t_success_env = t_env
        success_now.append(succ)
        success_latched_l.append(latched)
        env_done.append(bool(done))

    H = torch.stack(hidden)                                       # (520, 33, 4096)
    L = torch.stack(logits)                                       # (520, 7, 256)
    if H.dtype != STORE_DTYPE or L.dtype != STORE_DTYPE:
        raise RuntimeError("stored tensors are not fp16")
    return {
        "hidden": H, "logits": L,
        "actions": dict(
            executed=np.stack(executed).astype(np.float32),
            action_raw=np.stack(action_raw_all).astype(np.float32),
            action_token_ids=np.stack(tok_ids).astype(np.int32),
            policy_step_env_t=np.asarray(env_t, dtype=np.int32),
            success_now=np.asarray(success_now, dtype=bool),
            success_latched=np.asarray(success_latched_l, dtype=bool),
            env_done=np.asarray(env_done, dtype=bool),
            is_wait_step=np.asarray(is_wait, dtype=bool),
            t_effective=np.arange(n_wait + max_steps, dtype=np.int32) - n_wait),
        "obs": {k: np.stack([o[k] for o in obs_list]).astype(np.float32)
                for k in sorted(obs_list[0])},
        "wall_seconds": time.time() - t0,
        "t_success": None if t_success_env is None else t_success_env - n_wait,
        "t_success_env": t_success_env,
        "success_ever": bool(latched),
        "success_final": bool(success_now[-1]),
        "align": align,
        "frame_mean_pixel": float(np.mean(frame_means)),
        "frame_min_mean": float(np.min(frame_means)),
    }


def write_rollout(tmp: Path, final: Path, ep: dict, meta: dict, progress: Path | None):
    torch.save(ep["hidden"], tmp / "hidden.pt")
    torch.save(ep["logits"], tmp / "logits.pt")
    np.savez_compressed(tmp / "actions.npz", **ep["actions"])
    np.savez_compressed(tmp / "obs.npz", **ep["obs"])
    sizes = {p.name: p.stat().st_size for p in tmp.glob("*.*") if p.is_file()}
    sizes["frames/"] = sum(p.stat().st_size for p in (tmp / "frames").glob("*.jpg"))
    meta = {**meta, "file_sizes_bytes": sizes, "bytes": sum(sizes.values()),
            "hidden_shape": list(ep["hidden"].shape), "logits_shape": list(ep["logits"].shape),
            "complete": True}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    if final.exists():                      # a stale incomplete dir; complete ones are skipped
        shutil.rmtree(final)
    os.rename(tmp, final)
    if progress is not None:
        rec = {k: meta[k] for k in ("rollout_id", "block", "suite", "base_task_idx", "trial_idx",
                                    "init_state_index", "success_ever", "success_final",
                                    "t_success", "wall_clock_seconds", "bytes")}
        rec["worker"] = meta.get("worker")
        rec["finished_utc"] = utcnow()
        append_jsonl(progress, rec)
    return meta


class EnvCache:
    """One open env at a time, rebuilt when the (suite, task) changes."""

    def __init__(self):
        self.key, self.env, self.suites = None, None, {}

    def get(self, suite_name: str, task_idx: int):
        if suite_name not in self.suites:
            self.suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        suite = self.suites[suite_name]
        base_names = self._base_names()
        # Perturbed suites list the same task names; map by name, never by position.
        pos = [t.name for t in suite.tasks].index(base_names[task_idx])
        if self.key != (suite_name, pos):
            self.close()
            task = suite.tasks[pos]
            self.env, _ = get_libero_env(task, "openvla", resolution=256)
            self.key = (suite_name, pos)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            self.instruction = common.bddl_language(bddl)
            self.task_name = task.name
            self.init_states = suite.get_task_init_states(pos)
            self.bddl = str(bddl)
        return self.env

    def _base_names(self):
        if "libero_10" not in self.suites:
            self.suites["libero_10"] = benchmark.get_benchmark_dict()["libero_10"]()
        return [t.name for t in self.suites["libero_10"].tasks]

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env, self.key = None, None


def recent_cell_failures(out: Path, block: str, task: int) -> int:
    """Consecutive most-recent failures of this (block, task) cell, across workers."""
    events = []
    for name, ok in (("progress.jsonl", True), ("failures.jsonl", False)):
        p = out / name
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("block") == block and r.get("base_task_idx") == task:
                    events.append((r.get("finished_utc", ""), ok))
    n = 0
    for _, ok in sorted(events, reverse=True):
        if ok:
            break
        n += 1
    return n


def collect_one(vla, processor, envs, job, cfg, out: Path, tmp_tag: str, bin_slice, vocab_size,
                progress: Path | None, seed: int, all_frames: bool, extra_meta: dict):
    env = envs.get(job["suite"], job["task"])
    rid = job["rollout_id"]
    tmp = out / f".tmp_{rid}_{tmp_tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    set_seed_everywhere(seed)
    started = utcnow()
    try:
        ep = run_episode(vla, processor, env, envs.init_states[job["init_state_index"]],
                         envs.instruction, tmp / "frames", all_frames, bin_slice, vocab_size,
                         cfg["max_policy_steps"], cfg["num_steps_wait"])
        if ep["align"].fatal:
            raise RuntimeError(f"logit/token alignment fatal: {ep['align'].summary()}")
        meta = {
            "rollout_id": rid, "block": job["block"], "suite": job["suite"],
            "base_task_idx": job["task"], "trial_idx": job["trial"],
            "init_state_index": job["init_state_index"], "instruction": envs.instruction,
            "task_name": envs.task_name, "bddl_file": envs.bddl,
            "success_ever": ep["success_ever"], "success_final": ep["success_final"],
            "t_success": ep["t_success"], "t_success_env": ep["t_success_env"],
            "wall_clock_seconds": round(ep["wall_seconds"], 1),
            "T_total": int(ep["hidden"].shape[0]), "T_env_total": cfg["max_policy_steps"] + cfg["num_steps_wait"],
            "num_steps_wait": cfg["num_steps_wait"], "model_checkpoint": g2.CHECKPOINT,
            "unnorm_key": g2.UNNORM_KEY, "seed": seed, "frames": "all" if all_frames else "step0",
            "hidden_semantics": "mean_over_7_decision_states (v2 P=3 position 0)",
            "logit_alignment": ep["align"].summary(), "frame_mean_pixel": ep["frame_mean_pixel"],
            "frame_min_mean": ep["frame_min_mean"], "capture_started_utc": started,
            **extra_meta,
        }
        return write_rollout(tmp, out / rid, ep, meta, progress)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def deadline_passed(cfg, out: Path) -> bool:
    p = out / "collect_started_utc"
    if not p.exists():
        return False
    start = datetime.fromisoformat(p.read_text().strip())
    return (datetime.now(timezone.utc) - start).total_seconds() > 3600 * cfg["COLLECT_HOURS_MAX"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="run just these rollout ids")
    ap.add_argument("--parity", type=str, default=None, help="re-run a v2 rollout id")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--max-jobs", type=int, default=None, help="stop after this many rollouts")
    args = ap.parse_args()

    cfg = common.load_config()
    out = Path(args.out_dir) if args.out_dir else common.out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers or int(cfg["N_WORKERS"])
    tag = f"w{args.worker}"

    print(f"[*] {utcnow()} worker {args.worker}/{n_workers} out={out} free={common.free_gb(out):.0f} GB", flush=True)
    vla, processor = g2.load_model()
    bin_slice = g2.action_bin_slice(vla)
    vocab_size = int(vla.vocab_size)
    envs = EnvCache()

    if args.parity:
        m = json.loads((Path("/data/rollouts_v2") / args.parity / "manifest.json").read_text())
        job = {"block": "parity", "task": m["task_idx"], "trial": m["trial_idx"],
               "init_state_index": m["init_state_index"], "suite": "libero_10",
               "rollout_id": args.parity}
        meta = collect_one(vla, processor, envs, job, cfg, out, tag, bin_slice, vocab_size, None,
                           int(m["seed"]), False, {"parity_of": args.parity})
        assert envs.instruction == m["task_description"], (envs.instruction, m["task_description"])
        print(f"[parity] wrote {out / args.parity} success_ever={meta['success_ever']} "
              f"(v2: {m['success_ever']}) {meta['wall_clock_seconds']}s", flush=True)
        return

    assignment = common.load_assignment(cfg)
    all_jobs = common.jobs(cfg, assignment)
    if args.only:
        mine = [j for j in all_jobs if j["rollout_id"] in set(args.only)]
    else:
        mine = [j for j in all_jobs if j["job_number"] % n_workers == args.worker]
    if not args.only:
        (out / f"worker{args.worker}.finished").unlink(missing_ok=True)
    for p in out.glob(f".tmp_*_{tag}"):        # this worker's half-written rollouts
        shutil.rmtree(p, ignore_errors=True)
    progress, failures = out / "progress.jsonl", out / "failures.jsonl"
    n_complete = sum(common.is_complete(out / j["rollout_id"]) for j in mine)
    stale = [p.name for p in out.glob(".tmp_*")]
    print(f"[*] {len(mine)} jobs for this worker: {n_complete} already complete (skipped), "
          f"{len(mine) - n_complete} pending; other workers' tmp dirs: {stale}", flush=True)
    done_n = 0
    for job in mine:
        rid = job["rollout_id"]
        if common.is_complete(out / rid):
            continue
        if not args.only and deadline_passed(cfg, out):
            print(f"[*] {utcnow()} COLLECT_HOURS_MAX reached; stopping", flush=True)
            break
        if common.free_gb(out) < cfg["HEADROOM_GB"]:
            print(f"[!] {utcnow()} free disk below HEADROOM_GB; stopping", flush=True)
            append_jsonl(failures, {"rollout_id": rid, "block": job["block"], "base_task_idx": job["task"],
                                    "error": "disk_headroom_stop", "finished_utc": utcnow(), "fatal": True})
            break
        if recent_cell_failures(out, job["block"], job["task"]) >= 3:
            print(f"[!] {rid}: cell ({job['block']}, task {job['task']}) failed 3x in a row; skipping", flush=True)
            continue
        seed = 10000 * (common.block_number(cfg, job["block"]) + 1) + 100 * job["task"] + job["trial"]
        for attempt in (1, 2):
            try:
                meta = collect_one(vla, processor, envs, job, cfg, out, tag, bin_slice, vocab_size,
                                   progress, seed, job["trial"] == 0,
                                   {"worker": args.worker, "job_number": job["job_number"], "attempt": attempt})
                done_n += 1
                print(f"[{utcnow()[:19]}] {rid} init={job['init_state_index']} "
                      f"success={meta['success_ever']} t_success={meta['t_success']} "
                      f"{meta['wall_clock_seconds']:.0f}s {meta['bytes']/1e6:.0f}MB "
                      f"free={common.free_gb(out):.0f}GB", flush=True)
                break
            except Exception as e:
                traceback.print_exc()
                envs.close()                     # rebuild the env after any error
                if attempt == 2:
                    append_jsonl(failures, {"rollout_id": rid, "block": job["block"],
                                            "base_task_idx": job["task"], "error": repr(e)[:500],
                                            "finished_utc": utcnow(), "worker": args.worker})
                    print(f"[!] {rid} failed twice; logged and skipped", flush=True)
        if args.max_jobs and done_n >= args.max_jobs:
            break
    envs.close()
    print(f"[*] {utcnow()} worker {args.worker} finished: {done_n} rollouts this session", flush=True)
    if not args.only:
        (out / f"worker{args.worker}.finished").write_text(utcnow() + "\n")


if __name__ == "__main__":
    main()
