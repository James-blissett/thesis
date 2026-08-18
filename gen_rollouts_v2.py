"""
gen_rollouts_v2.py  --  Rollout Collection v2

Runs OpenVLA in LIBERO-10 and captures hidden states, action logits, frames and raw
observations, per the v2 handoff. Adapted from `gen_rollouts.py`, which is left in place
for the existing 50-rollout corpus.

The conventions inherited from the fork's `run_libero_eval.py` are deliberate and must not
be "cleaned up":
  * NUM_STEPS_WAIT = 10 no-op settle steps at episode start (the sim drops objects in and
    they must physically settle before the policy acts),
  * the image pipeline (get_libero_image -> 224px -> 0.9 center crop),
  * gripper handling (normalize_gripper_action(binarize=True) then invert_gripper_action,
    because the RLDS dataloader aligns gripper actions as 0=close/1=open),
  * MAX_POLICY_STEPS = 520 for libero_10 (longest training demo is 505 steps).

On the 520-vs-510 arithmetic in the handoff
-------------------------------------------
The handoff says T_max = 520 with "an effective horizon of 510" after subtracting the 10
wait steps, but also requires `T_total == 520`. In this loop -- inherited unchanged from
v1 -- the 10 wait steps sit *outside* the 520, giving 530 env steps of which 520 are
policy steps. Keeping that is what makes the v2 corpus a strict superset of v1, whose
failed rollouts each hold 520 policy steps; moving to 510 would silently shorten every
trajectory relative to the old corpus. So T_total == 520 policy steps as required, and the
effective horizon is 520, not 510. `t_effective = t_env - num_steps_wait` exactly as
specified, running -10..519. Flagged for sign-off before the full run.

Usage:
    source env.sh
    python make_init_assignment.py
    python gen_rollouts_v2.py --smoke          # task 0, one P=7 trial and one P=3 trial
    python gen_rollouts_v2.py                  # all 10 tasks x 30 trials
    python gen_rollouts_v2.py --task-start 0 --task-end 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

OPENVLA_ROOT = Path("/ephemeral/code/openvla")
sys.path.insert(0, str(OPENVLA_ROOT))

from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.robot.openvla_utils import get_vla_prompt, preprocess_image  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from capture_v2 import (  # noqa: E402
    DOF_NAMES,
    N_ACTION_BINS,
    N_ACTION_TOKENS,
    RolloutWriter,
    action_bin_slice,
    assert_bin_slice_agrees,
    assert_decode_shapes,
    extract_action_logits,
    extract_decision_states,
    logit_alignment,
    manifest_is_complete,
    pool_positions,
    token_id_to_bin_index,
)

# --- Locked configuration -----------------------------------------------------------
CHECKPOINT = "openvla/openvla-7b-finetuned-libero-10"
TASK_SUITE = "libero_10"
UNNORM_KEY = "libero_10"   # MUST match the checkpoint's dataset statistics; a wrong key
                           # fails silently -- actions look plausible and score ~0%.
MAX_POLICY_STEPS = 520     # T_total; libero_10 budget
NUM_STEPS_WAIT = 10        # no-op settle steps before the policy acts
RESIZE_SIZE = 224
CENTER_CROP = True         # checkpoint was trained with image augmentation
SEED_BASE = 2000           # distinct from v1's 1000; variation still comes only from init state
DEVICE = "cuda:0"

# /ephemeral is on the 124 GB root filesystem; the corpus goes on the 492 GB /data volume
# alongside the v1 corpus. See the audit note in the handoff reply.
OUT_DIR = Path("/data/rollouts_v2")
ASSIGNMENT = Path(__file__).resolve().parent / "init_state_assignment.json"

MIN_FREE_GB = 200          # refuse to start a 16-hour run into a disk that cannot hold it
EXPECTED_SUCCESS_RATE = 0.53
SUCCESS_RATE_TOLERANCE = 0.15

# Rollout-level gate on logit/token alignment. Zero mismatches is the expectation and the
# smoke test met it (0 of 3640, twice). But this run is unattended, and a token whose
# global argmax falls outside the 256-bin action window is a rare, self-recording anomaly
# -- `predict_action` clips it, the manifest logs it, and it is diagnosable afterwards.
# Killing a nine-hour run for one such position would be the wrong trade. A genuine wiring
# bug (wrong slice, wrong flip, off-by-one in the generate-step alignment) mismatches
# essentially every position and so still dies on the first rollout. Same reasoning as v1's
# parity gate: the failure modes differ by three orders of magnitude, not by a hair.
LOGIT_MISMATCH_MAX_RATE = 0.001   # 3640 positions/rollout -> aborts at >3 bad positions


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def load_model():
    """Load OpenVLA in bf16 with flash-attention-2, mirroring the fork's `get_vla`."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = OpenVLAForActionPrediction.from_pretrained(
        CHECKPOINT,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    vla.eval()

    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

    assert UNNORM_KEY in vla.norm_stats, (
        f"unnorm_key {UNNORM_KEY!r} absent from norm_stats {list(vla.norm_stats.keys())}"
    )
    assert_bin_slice_agrees(vla, processor.tokenizer)
    return vla, processor


@torch.inference_mode()
def predict_and_capture(vla, processor, img, task_description, n_positions, bin_slice):
    """One policy query. Hidden states and logits come from the generation pass itself.

    `predict_action` forwards **kwargs straight to `generate`, so `output_hidden_states`
    and `output_logits` ride along on the call that was happening anyway -- no second
    forward pass, and the stored logits are the ones that actually decoded the actions.

    Returns (action_raw, hidden, logits, token_ids, decision, gen_out).
    `action_raw` is unnormalised but pre-gripper-fix, i.e. exactly what the action tokens
    decode to; the caller applies the gripper convention before stepping the env.
    """
    image = preprocess_image(img, CENTER_CROP)
    prompt = get_vla_prompt(task_description, CHECKPOINT)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    action_raw, gen_out = vla.predict_action(
        **inputs,
        unnorm_key=UNNORM_KEY,
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_logits=True,
    )

    decision = extract_decision_states(gen_out.hidden_states)     # (7, n_layers, d)
    hidden = pool_positions(decision, n_positions)                # (n_layers, P, d)
    logits = extract_action_logits(gen_out.logits, bin_slice)     # (7, 256)
    token_ids = gen_out.sequences[0, -N_ACTION_TOKENS:].cpu().numpy()
    return action_raw, hidden, logits, token_ids, decision, gen_out


class AlignmentAccumulator:
    """Rollout-level roll-up of the per-timestep logit/token alignment check.

    Ties between adjacent action bins are expected and harmless (see `logit_alignment`);
    they are counted and reported rather than treated as failures. A *real* mismatch --
    the generated token outside the action window, or its logit strictly below the window
    maximum -- means the logits field is not what it claims to be, and stops the run.
    """

    def __init__(self) -> None:
        self.n_positions = self.n_ok = self.n_index_equal = 0
        self.n_ties = self.n_outside = self.n_mismatch = 0
        self.max_shortfall = 0.0
        self.first_failure = None

    def add(self, rep: dict, t_env: int) -> None:
        self.n_positions += rep["n_positions"]
        self.n_ok += rep["n_ok"]
        self.n_index_equal += rep["n_index_equal"]
        self.n_ties += rep["n_tie_resolved_differently"]
        self.n_outside += rep["n_outside_window"]
        self.n_mismatch += rep["n_real_mismatch"]
        self.max_shortfall = max(self.max_shortfall, rep["max_shortfall"])
        if not rep["ok"] and self.first_failure is None:
            self.first_failure = {"t_env": t_env, **rep}

    @property
    def mismatch_rate(self) -> float:
        return self.n_mismatch / max(self.n_positions, 1)

    @property
    def ok(self) -> bool:
        """Clean. Anything else is worth printing even when the run continues."""
        return self.n_mismatch == 0

    @property
    def fatal(self) -> bool:
        """Bad enough that the logits field is not what it claims to be."""
        return self.mismatch_rate > LOGIT_MISMATCH_MAX_RATE

    def summary(self) -> dict:
        n = max(self.n_positions, 1)
        return {
            "positions_checked": self.n_positions,
            "positions_ok": self.n_ok,
            "agreement_rate": round(self.n_ok / n, 6),
            "n_index_equal": self.n_index_equal,
            "index_equal_rate": round(self.n_index_equal / n, 6),
            "n_tie_resolved_differently": self.n_ties,
            "tie_rate": round(self.n_ties / n, 6),
            "n_outside_action_window": self.n_outside,
            "n_real_mismatch": self.n_mismatch,
            "max_shortfall": round(self.max_shortfall, 6),
            "note": (
                "agreement is value-based: the generated token's stored logit equals the "
                "maximum over the 256-bin window. index_equal is the stricter argmax-index "
                "match, which bf16 ties break harmlessly in either direction."
            ),
        }


def run_episode(vla, processor, env, init_state, task_description, writer, n_positions,
                bin_slice, vocab_size, verify_first_step=False):
    """Run one episode for the full policy-step budget. No early termination.

    Returns a dict of episode-level outcomes.
    """
    env.reset()
    obs = env.set_init_state(init_state)

    success_latched = False
    t_success_env = None
    align = AlignmentAccumulator()
    verification = None
    t0 = time.time()

    for t_env in range(NUM_STEPS_WAIT + MAX_POLICY_STEPS):
        is_wait = t_env < NUM_STEPS_WAIT

        if is_wait:
            # Settle phase: the simulator drops objects in; act only once they have landed.
            action_env = np.asarray(get_libero_dummy_action("openvla"), dtype=np.float32)
            writer.add_step(t_env, True, obs, action_env)
        else:
            img = get_libero_image(obs, RESIZE_SIZE)
            action_raw, hidden, logits, token_ids, decision, gen_out = predict_and_capture(
                vla, processor, img, task_description, n_positions, bin_slice
            )

            if verify_first_step and verification is None:
                verification = build_verification(
                    vla, gen_out, decision, logits, token_ids, action_raw, bin_slice
                )

            align.add(logit_alignment(logits, token_ids, vocab_size), t_env)

            # Both helpers mutate their argument in place, so copy first: `action_raw`
            # is stored as the ground truth the action tokens decode to, and the gripper
            # convention would otherwise overwrite dimension 6 of the stored array.
            action_env = np.array(action_raw, dtype=np.float32, copy=True)
            action_env = normalize_gripper_action(action_env, binarize=True)
            action_env = invert_gripper_action(action_env)

            writer.add_step(
                t_env, False, obs, action_env,
                hidden=hidden, logits=logits,
                action_token_ids=token_ids, action_raw=action_raw,
            )

        obs, reward, done, info = env.step(np.asarray(action_env).tolist())

        succ = bool(env.check_success())
        success_latched = success_latched or succ
        if succ and t_success_env is None:
            t_success_env = t_env
        writer.add_outcome(succ, success_latched, bool(done))

    return {
        "wall_seconds": time.time() - t0,
        "t_success_env": t_success_env,
        "t_success": None if t_success_env is None else t_success_env - NUM_STEPS_WAIT,
        "success_final": bool(writer.success_now[-1]),
        "success_ever": bool(success_latched),
        "logit_alignment": align.summary(),
        "logit_alignment_ok": align.ok,
        "logit_alignment_fatal": align.fatal,
        "logit_alignment_first_failure": align.first_failure,
        "verification": verification,
        "termination_reason": "step_budget_exhausted",
    }


@torch.inference_mode()
def build_verification(vla, gen_out, decision, logits, token_ids, action_raw, bin_slice):
    """Structural proof, on one timestep, that position k really is what decodes DoF k.

    Pushing decision[k] at the final layer back through the lm_head must reproduce
    `gen_out.logits[k]`, because hidden_states[-1] is post-final-norm. If the prefill/decode
    index asymmetry were mishandled, this would not hold.
    """
    prompt_len = assert_decode_shapes(gen_out.hidden_states)
    lm_head = vla.get_output_embeddings()

    rows = []
    for k in range(N_ACTION_TOKENS):
        h = decision[k, -1].to(DEVICE, dtype=torch.bfloat16)
        recon = lm_head(h).float().cpu()
        ref = gen_out.logits[k][0].float().cpu()
        rows.append({
            "position_index": k,
            "position": "h[N]" if k == 0 else f"h[N+{k}]",
            "decodes_dof": DOF_NAMES[k],
            "generated_token_id": int(token_ids[k]),
            "lm_head_recon_argmax_token": int(recon.argmax()),
            "generate_logits_argmax_token": int(ref.argmax()),
            "max_abs_logit_diff": float((recon - ref).abs().max()),
            "stored_bin_argmax": int(logits[k].float().argmax()),
            "action_raw": float(np.asarray(action_raw)[k]),
        })

    return {
        "prompt_len_hidden_positions": prompt_len,
        "n_layers": int(len(gen_out.hidden_states[0])),
        "action_bin_slice": [bin_slice.start, bin_slice.stop],
        "positions": rows,
        "all_positions_reconstruct": all(
            r["lm_head_recon_argmax_token"] == r["generated_token_id"] for r in rows
        ),
    }


def load_assignment(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run `python make_init_assignment.py` first and commit it: "
            "the corpus cannot be reproduced or extended without it."
        )
    with open(path) as f:
        return json.load(f)


def resolve_pending(out_dir: Path, assignment: dict, task_ids, trial_ids):
    """Which rollouts still need running, discarding any partial directories.

    A directory whose manifest is missing, unparseable, or incomplete is a rollout that
    died mid-write. It is removed and redone -- never left to be half-read later.
    """
    pending, skipped, discarded = [], [], []
    for task_id in task_ids:
        spec = assignment["assignment"][str(task_id)]
        for trial in trial_ids:
            if trial >= len(spec["init_state_indices"]):
                continue
            rollout_id = f"task{task_id:02d}_trial{trial:02d}"
            d = out_dir / rollout_id
            if d.exists():
                if manifest_is_complete(d / "manifest.json"):
                    skipped.append(rollout_id)
                    continue
                if not d.name.startswith("task") or "_trial" not in d.name:
                    raise SystemExit(f"refusing to remove unexpected directory {d}")
                shutil.rmtree(d)
                discarded.append(rollout_id)
            pending.append((task_id, trial))
    return pending, skipped, discarded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="task 0, trial 0 (P=7) and trial 3 (P=3), with full verification")
    ap.add_argument("--task-start", type=int, default=0)
    ap.add_argument("--task-end", type=int, default=9, help="inclusive")
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--trial-end", type=int, default=29, help="inclusive")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--assignment", type=str, default=str(ASSIGNMENT))
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()

    assignment = load_assignment(Path(args.assignment))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        task_ids, trial_ids = [0], [0, 3]
        args.min_free_gb = min(args.min_free_gb, 5.0)
    else:
        task_ids = list(range(args.task_start, args.task_end + 1))
        trial_ids = list(range(args.trial_start, args.trial_end + 1))

    have = free_gb(out_dir)
    if have < args.min_free_gb:
        raise SystemExit(
            f"only {have:.0f} GB free on {out_dir}; need >= {args.min_free_gb:.0f} GB. "
            "Refusing to start a run that cannot finish."
        )

    pending, skipped, discarded = resolve_pending(out_dir, assignment, task_ids, trial_ids)
    print(f"[*] out_dir={out_dir} free={have:.0f} GB", flush=True)
    print(f"[*] {len(pending)} pending, {len(skipped)} already complete (skipped), "
          f"{len(discarded)} partial discarded: {discarded}", flush=True)
    if not pending:
        print("[*] nothing to do")
        return

    print(f"[*] loading {CHECKPOINT}", flush=True)
    vla, processor = load_model()
    bin_slice = action_bin_slice(vla)
    vocab_size = int(vla.vocab_size)
    print(f"[*] model loaded | unnorm_key={UNNORM_KEY} | "
          f"action_bin_slice=[{bin_slice.start},{bin_slice.stop}) | vocab_size={vocab_size}",
          flush=True)

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    commit = git_commit()
    progress_path = out_dir / "progress.jsonl"
    results = []
    verify_next = args.smoke

    by_task = {}
    for task_id, trial in pending:
        by_task.setdefault(task_id, []).append(trial)

    for task_id, trials in sorted(by_task.items()):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        spec = assignment["assignment"][str(task_id)]

        try:
            for trial in trials:
                init_idx = spec["init_state_indices"][trial]
                n_positions = spec["n_positions_by_trial"][trial]
                rollout_id = f"task{task_id:02d}_trial{trial:02d}"
                set_seed_everywhere(SEED_BASE + task_id * 100 + trial)

                writer = RolloutWriter(out_dir, rollout_id, n_positions)
                writer.prepare()
                started = datetime.now(timezone.utc).isoformat()

                try:
                    outcome = run_episode(
                        vla, processor, env, init_states[init_idx], task_description,
                        writer, n_positions, bin_slice, vocab_size,
                        verify_first_step=verify_next,
                    )
                except Exception:
                    traceback.print_exc()
                    shutil.rmtree(writer.dir, ignore_errors=True)
                    raise

                if not outcome["logit_alignment_ok"]:
                    fail = outcome["logit_alignment_first_failure"]
                    la = outcome["logit_alignment"]
                    verdict = "STOPPING" if outcome["logit_alignment_fatal"] else "continuing"
                    print(
                        f"!!! {rollout_id}: {la['n_real_mismatch']}/{la['positions_checked']} "
                        f"positions failed logit/token alignment "
                        f"(rate {la['n_real_mismatch']/la['positions_checked']:.2e}, "
                        f"threshold {LOGIT_MISMATCH_MAX_RATE:.0e}) -- {verdict}. "
                        f"first at t_env={fail['t_env']}, "
                        f"n_outside_window={fail['n_outside_window']}, "
                        f"shortfalls={fail['shortfalls']}. Recorded in the manifest.",
                        flush=True,
                    )

                if outcome["logit_alignment_fatal"]:
                    fail = outcome["logit_alignment_first_failure"]
                    shutil.rmtree(writer.dir, ignore_errors=True)
                    raise RuntimeError(
                        f"{rollout_id}: stored action logits are not consistent with the "
                        f"generated tokens; the logits field would be worthless. Stopping.\n"
                        f"  summary        : {json.dumps(outcome['logit_alignment'])}\n"
                        f"  first failure  : t_env={fail['t_env']}\n"
                        f"  generated toks : {fail['generated_token_ids']}\n"
                        f"  token bins     : {fail['token_bins']}\n"
                        f"  argmax bins    : {fail['argmax_bins']}\n"
                        f"  shortfalls     : {fail['shortfalls']}  (None = token outside "
                        f"the [{bin_slice.start},{bin_slice.stop}) action window)\n"
                        f"  n_outside_window={fail['n_outside_window']} "
                        f"n_real_mismatch={fail['n_real_mismatch']}"
                    )

                manifest = writer.flush({
                    "rollout_id": rollout_id,
                    "task_idx": task_id,
                    "task_name": task.name,
                    "task_description": task_description,
                    "trial_idx": trial,
                    "init_state_index": int(init_idx),
                    "num_steps_wait": NUM_STEPS_WAIT,
                    "t_success": outcome["t_success"],
                    "t_success_env": outcome["t_success_env"],
                    "success_final": outcome["success_final"],
                    "success_ever": outcome["success_ever"],
                    "model_checkpoint": CHECKPOINT,
                    "unnorm_key": UNNORM_KEY,
                    "task_suite": TASK_SUITE,
                    "action_bin_slice": [bin_slice.start, bin_slice.stop],
                    "vocab_size": vocab_size,
                    "git_commit": commit,
                    "capture_started_utc": started,
                    "termination_reason": outcome["termination_reason"],
                    "logit_alignment": outcome["logit_alignment"],
                    "seed": SEED_BASE + task_id * 100 + trial,
                    "assignment_seed": assignment["rng_seed"],
                })

                if verify_next and outcome["verification"] is not None:
                    with open(writer.dir / "verification.json", "w") as f:
                        json.dump(outcome["verification"], f, indent=2)
                    verify_next = False

                rec = {
                    "rollout_id": rollout_id,
                    "task_idx": task_id,
                    "trial_idx": trial,
                    "init_state_index": int(init_idx),
                    "n_positions": n_positions,
                    "success_ever": outcome["success_ever"],
                    "success_final": outcome["success_final"],
                    "t_success": outcome["t_success"],
                    "T_total": manifest["T_total"],
                    "wall_clock": round(outcome["wall_seconds"], 1),
                    "bytes": manifest["size_bytes"],
                    "mean_pixel": round(manifest["frame_stats"]["mean_pixel"], 2),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(progress_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                results.append(rec)

                n_succ = sum(r["success_ever"] for r in results)
                print(
                    f"[{len(results)}/{len(pending)}] {rollout_id} init={init_idx} P={n_positions} "
                    f"success_ever={outcome['success_ever']} t_success={outcome['t_success']} "
                    f"{outcome['wall_seconds']:.0f}s {manifest['size_bytes']/1e6:.0f}MB "
                    f"running={n_succ}/{len(results)} free={free_gb(out_dir):.0f}GB",
                    flush=True,
                )
        finally:
            env.close()

    n_succ = sum(r["success_ever"] for r in results)
    rate = n_succ / len(results)
    print(f"\n=== DONE: {n_succ}/{len(results)} success_ever ({100*rate:.1f}%) ===", flush=True)
    if abs(rate - EXPECTED_SUCCESS_RATE) > SUCCESS_RATE_TOLERANCE and len(results) >= 30:
        print(
            f"!!! success rate {100*rate:.1f}% deviates from the expected "
            f"{100*EXPECTED_SUCCESS_RATE:.0f}% by more than "
            f"{100*SUCCESS_RATE_TOLERANCE:.0f} points. Suspect the checkpoint or unnorm_key. "
            "STOP and report.",
            flush=True,
        )

    if args.smoke:
        smoke_report(out_dir, results, vla, processor)


def smoke_report(out_dir: Path, results, vla, processor):
    """The nine checks the handoff requires before the full run is signed off."""
    from prismatic.vla.action_tokenizer import ActionTokenizer

    tok = ActionTokenizer(processor.tokenizer)
    action_stats = vla.get_action_stats(UNNORM_KEY)
    print("\n" + "=" * 78)
    print("SMOKE REPORT")
    print("=" * 78)

    total_bytes_p3 = total_bytes_p7 = 0
    total_secs_p3 = total_secs_p7 = 0.0

    for rec in results:
        d = out_dir / rec["rollout_id"]
        with open(d / "manifest.json") as f:
            m = json.load(f)
        hidden = torch.load(d / "hidden.pt", map_location="cpu")
        logits = torch.load(d / "logits.pt", map_location="cpu")
        acts = np.load(d / "actions.npz")

        print(f"\n--- {rec['rollout_id']}  (P={m['n_positions']}, "
              f"init_state={m['init_state_index']}) ---")

        # (1) bytes on disk, by file
        print("[1] bytes on disk:")
        for name, nbytes in sorted(m["file_sizes_bytes"].items()):
            print(f"      {name:<16} {nbytes/1e6:10.1f} MB")
        print(f"      {'TOTAL':<16} {m['size_bytes']/1e6:10.1f} MB")

        # (2) wall clock
        print(f"[2] wall clock: {rec['wall_clock']:.0f} s "
              f"({m['T_total']/rec['wall_clock']:.2f} policy steps/s)")

        if m["n_positions"] == 7:
            total_bytes_p7, total_secs_p7 = m["size_bytes"], rec["wall_clock"]
        else:
            total_bytes_p3, total_secs_p3 = m["size_bytes"], rec["wall_clock"]

        # (4) shapes, dtype, layer count, all-zero layers
        L = hidden.shape[1]
        per_layer_absmax = hidden.float().abs().amax(dim=(0, 2, 3))
        dead = [int(i) for i, v in enumerate(per_layer_absmax.tolist()) if v == 0.0]
        print(f"[4] hidden.pt {tuple(hidden.shape)} {hidden.dtype} | L={L} "
              f"(expect 33: {'OK' if L == 33 else 'FAIL'}) | all-zero layers: {dead or 'none'}")
        print(f"    logits.pt {tuple(logits.shape)} {logits.dtype} | "
              f"finite: {bool(torch.isfinite(logits).all())}")
        print(f"    layer0 absmax={per_layer_absmax[0]:.3f}  "
              f"layer{L-1} absmax={per_layer_absmax[-1]:.3f}")
        print(f"    position_semantics: {m['position_semantics']}")

        # (5) prefill/decode asymmetry + position -> DoF mapping
        vpath = d / "verification.json"
        if vpath.exists():
            with open(vpath) as f:
                v = json.load(f)
            print(f"[5] prompt_len={v['prompt_len_hidden_positions']} hidden positions "
                  f"(prefill indexed -1, decode steps indexed 0), n_layers={v['n_layers']}")
            print(f"    lm_head(decision[k]) reproduces generate logits[k] for all k: "
                  f"{v['all_positions_reconstruct']}")
            print(f"    {'k':<3}{'position':<10}{'DoF':<9}{'gen_tok':<9}{'recon_tok':<11}"
                  f"{'max|dlogit|':<13}{'bin':<6}{'action_raw'}")
            for r in v["positions"]:
                print(f"    {r['position_index']:<3}{r['position']:<10}{r['decodes_dof']:<9}"
                      f"{r['generated_token_id']:<9}{r['lm_head_recon_argmax_token']:<11}"
                      f"{r['max_abs_logit_diff']:<13.5f}{r['stored_bin_argmax']:<6}"
                      f"{r['action_raw']:+.4f}")

        # (6) logit argmax -> ActionTokenizer -> executed action, 5 sampled timesteps
        print("[6] logit/action verification (5 sampled timesteps):")
        vocab_size = m["vocab_size"]
        idxs = np.linspace(0, m["T_total"] - 1, 5).astype(int)
        tok_ids = acts["action_token_ids"]
        raw = acts["action_raw"]
        env_t = acts["policy_step_env_t"]
        n_ok = 0
        for i in idxs:
            rep = logit_alignment(logits[i], tok_ids[i], vocab_size)
            bins = np.asarray(rep["argmax_bins"])
            # Decode via the generated token ids: on a bf16 tie the argmax index is
            # ambiguous by construction, so the meaningful question is whether the bin the
            # token sits in decodes back to the executed action.
            norm_act = tok.decode_token_ids_to_actions(np.asarray(tok_ids[i]))
            unnorm = unnormalize(norm_act, action_stats)
            executed = raw[env_t[i]]
            ok = rep["ok"] and np.allclose(unnorm, executed, atol=1e-5)
            n_ok += ok
            print(f"    t={i:3d}  argmax_bins ={bins.tolist()}")
            print(f"           token_bins  ={rep['token_bins']}  generated_tok={tok_ids[i].tolist()}")
            print(f"           decoded_action={np.array2string(unnorm, precision=4)}")
            print(f"           executed_raw  ={np.array2string(executed, precision=4)}  "
                  f"{'MATCH' if ok else 'MISMATCH'}")
            if rep["n_tie_resolved_differently"]:
                print(f"           ({rep['n_tie_resolved_differently']}/7 positions are exact "
                      f"bf16 ties between adjacent bins; equal logits, either index valid)")
        print(f"    {n_ok}/5 timesteps reproduce the executed action from the stored logits")

        la = m["logit_alignment"]
        print(f"    rollout-wide: {la['positions_ok']}/{la['positions_checked']} positions ok "
              f"({100*la['agreement_rate']:.4f}%), argmax-index equal "
              f"{100*la['index_equal_rate']:.2f}%, ties {la['n_tie_resolved_differently']}, "
              f"real mismatches {la['n_real_mismatch']}, max shortfall {la['max_shortfall']}")

        # (7) full horizon
        print(f"[7] T_total={m['T_total']} (expect 520: "
              f"{'OK' if m['T_total'] == 520 else 'FAIL'})  T_env_total={m['T_env_total']}  "
              f"success_ever={m['success_ever']} t_success={m['t_success']} "
              f"success_final={m['success_final']}")

        # (8) frames
        fs = m["frame_stats"]
        n_jpg = len(list((d / "frames").glob("*.jpg")))
        empty = [p.name for p in (d / "frames").glob("*.jpg") if p.stat().st_size == 0]
        print(f"[8] frames: {n_jpg} files (expect {2 * m['T_env_total']}), empty: {empty or 'none'}, "
              f"mean_pixel={fs['mean_pixel']:.1f} "
              f"(near-zero means EGL failed) range=[{fs['min_frame_mean']:.1f},"
              f"{fs['max_frame_mean']:.1f}]")

        # (9) wait steps
        n_wait = int(acts["is_wait_step"].sum())
        te = acts["t_effective"]
        print(f"[9] observed num_steps_wait={n_wait} (manifest says {m['num_steps_wait']}), "
              f"t_effective runs {te.min()}..{te.max()}")

    # (3) extrapolation
    print("\n" + "-" * 78)
    print("[3] extrapolation to 300 rollouts (270 at P=3, 30 at P=7)")
    if total_bytes_p3 and total_bytes_p7:
        disk = 270 * total_bytes_p3 + 30 * total_bytes_p7
        hours = (270 * total_secs_p3 + 30 * total_secs_p7) / 3600
        print(f"    disk : {disk/1e9:.0f} GB   ({total_bytes_p3/1e6:.0f} MB/rollout at P=3, "
              f"{total_bytes_p7/1e6:.0f} MB/rollout at P=7)")
        print(f"    time : {hours:.1f} h    ({total_secs_p3/60:.1f} min/rollout at P=3, "
              f"{total_secs_p7/60:.1f} min/rollout at P=7)")
        print(f"    free on {out_dir}: {free_gb(out_dir):.0f} GB  "
              f"-> {'FITS' if free_gb(out_dir) > disk/1e9 else 'DOES NOT FIT'}")
    print("-" * 78)
    print("STOP. Do not launch the full run without sign-off.")


def unnormalize(norm_action: np.ndarray, action_stats: dict) -> np.ndarray:
    """Mirror of `predict_action`'s unnormalisation, for offline verification."""
    high, low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
    mask = np.asarray(action_stats.get("mask", np.ones_like(low, dtype=bool)))
    return np.where(mask, 0.5 * (norm_action + 1) * (high - low) + low, norm_action)


if __name__ == "__main__":
    main()
