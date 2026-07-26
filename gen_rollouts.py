"""
gen_rollouts.py

Runs OpenVLA in LIBERO-10 and captures hidden states per timestep.

Adapted from the fork's `experiments/robot/libero/run_libero_eval.py`. The non-obvious
conventions are inherited from it deliberately and must not be "cleaned up":
  * `NUM_STEPS_WAIT = 10` no-op settle steps at episode start (the sim drops objects in
    and they must physically settle before the policy acts),
  * the exact image pipeline (`get_libero_image` -> 224px -> 0.9 center crop),
  * gripper action handling (`normalize_gripper_action(binarize=True)` then
    `invert_gripper_action`, because the RLDS dataloader flips the sign),
  * `MAX_STEPS = 520` for libero_10 (longest training demo is 505 steps).

W&B is deliberately not used here.

Usage:
    source env.sh
    python gen_rollouts.py --smoke                 # one seeded rollout on task 0
    python gen_rollouts.py                         # full 10 tasks x 5 rollouts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# The fork ships `experiments.robot.*` as loose modules, not an installed package.
OPENVLA_ROOT = Path("/ephemeral/code/openvla")
sys.path.insert(0, str(OPENVLA_ROOT))

from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_vla_prompt,
    preprocess_image,
)
from experiments.robot.robot_utils import (  # noqa: E402
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from capture import RolloutCapture, parity_check, teacher_forced_capture  # noqa: E402

# --- Locked configuration -----------------------------------------------------------
CHECKPOINT = "openvla/openvla-7b-finetuned-libero-10"
TASK_SUITE = "libero_10"
UNNORM_KEY = "libero_10"  # MUST match the checkpoint's dataset statistics; a wrong key
                          # fails silently -- actions look plausible and score ~0%.
MAX_STEPS = 520           # libero_10 budget (longest training demo: 505 steps)
NUM_STEPS_WAIT = 10       # no-op settle steps before the policy acts
N_ROLLOUTS_PER_TASK = 5
RESIZE_SIZE = 224
CENTER_CROP = True        # checkpoint was trained with image augmentation
SEED_BASE = 1000          # seed = SEED_BASE + global rollout index
OUT_DIR = Path("/data/corpus")
DEVICE = "cuda:0"


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
    return vla, processor


@torch.inference_mode()
def predict_and_capture(vla, processor, img, task_description):
    """One policy query + the teacher-forced re-forward that captures hidden states.

    Returns (action, hidden, parity_info). Mirrors the fork's `get_vla_action` but keeps
    `inputs` in scope so `pixel_values` can be reused for the re-forward.
    """
    image = preprocess_image(img, CENTER_CROP)
    prompt = get_vla_prompt(task_description, CHECKPOINT)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    action, gen_out = vla.predict_action(
        **inputs,
        unnorm_key=UNNORM_KEY,
        do_sample=False,
        return_dict_in_generate=True,
    )

    cap = teacher_forced_capture(
        vla, pixel_values=inputs["pixel_values"], sequences=gen_out["sequences"]
    )
    parity = parity_check(cap["action_logits"], gen_out["sequences"])
    return action, cap["hidden"], parity


def run_episode(vla, processor, env, initial_state, task_description, capture):
    """Run one episode to termination or the step budget, capturing every policy step.

    Returns (success, n_policy_steps, wall_seconds, first_parity, frame_stats).
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    done = False
    t = 0
    n_policy_steps = 0
    first_parity = None
    frame_means = []
    t0 = time.time()

    while t < MAX_STEPS + NUM_STEPS_WAIT:
        # Settle phase: the simulator drops objects in; act only once they have landed.
        if t < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        img = get_libero_image(obs, RESIZE_SIZE)
        # Track the raw agentview render: EGL failure produces well-formed all-black
        # frames that pass every structural check (Addendum A4.8).
        frame_means.append(float(np.mean(obs["agentview_image"])))

        action, hidden, parity = predict_and_capture(vla, processor, img, task_description)

        if first_parity is None:
            first_parity = parity

        # Gripper: [0,1] -> [-1,+1], binarized, then sign-flipped for OpenVLA because the
        # RLDS dataloader aligns gripper actions as 0=close/1=open.
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        # Record the observation the policy acted on alongside the action it executed.
        capture.add(hidden, obs, action, parity)
        n_policy_steps += 1

        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    wall = time.time() - t0
    frame_stats = {
        "mean_pixel": float(np.mean(frame_means)) if frame_means else 0.0,
        "min_frame_mean": float(np.min(frame_means)) if frame_means else 0.0,
        "max_frame_mean": float(np.max(frame_means)) if frame_means else 0.0,
    }
    return bool(done), n_policy_steps, wall, first_parity, frame_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one seeded rollout on task 0")
    ap.add_argument("--task-start", type=int, default=0)
    ap.add_argument("--task-end", type=int, default=9, help="inclusive")
    ap.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS_PER_TASK)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    if args.smoke:
        args.task_start, args.task_end, args.n_rollouts = 0, 0, 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading {CHECKPOINT}", flush=True)
    vla, processor = load_model()
    print(f"[*] model loaded | unnorm_key={UNNORM_KEY}", flush=True)

    task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    session_start = datetime.now(timezone.utc).isoformat()
    results = []
    parity_verified = False

    for task_id in range(args.task_start, args.task_end + 1):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)

        for ep in range(args.n_rollouts):
            global_idx = task_id * N_ROLLOUTS_PER_TASK + ep
            seed = SEED_BASE + global_idx
            set_seed_everywhere(seed)

            rollout_id = f"task{task_id:02d}_ep{ep:02d}"
            capture = RolloutCapture(out_dir, rollout_id)

            success, n_steps, wall, parity, frame_stats = run_episode(
                vla, processor, env, initial_states[ep], task_description, capture
            )

            # Parity is checked on the first rollout of every session; a mismatch means
            # the re-forward does not correspond to the rollout -> captures are garbage.
            # The gate distinguishes wiring bugs (near-0% agreement, large margins) from
            # the fp nondeterminism Addendum A2.4 anticipates (>=95%, tie-sized margins).
            parity_stats = capture.parity_summary()
            if not parity_verified:
                print(f"[*] parity (first timestep): {parity}", flush=True)
                print(f"[*] parity (rollout): {parity_stats}", flush=True)
                if not (parity_stats["rate_ok"] and parity_stats["mismatches_within_tie_band"]):
                    raise RuntimeError(
                        f"PARITY CHECK FAILED on {rollout_id}: teacher-forced pass did not "
                        f"reproduce generated action tokens beyond fp tolerance. "
                        f"first_timestep={parity} rollout={parity_stats}"
                    )
                print(
                    f"[*] parity check PASSED "
                    f"({100*parity_stats['agreement_rate']:.2f}% argmax agreement over "
                    f"{parity_stats['positions_checked']} positions; "
                    f"max mismatch gap {parity_stats['max_mismatch_top2_gap']})",
                    flush=True,
                )
                parity_verified = True

            paths = capture.flush(
                {
                    "rollout_id": rollout_id,
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_description": task_description,
                    "episode_idx": ep,
                    "seed": seed,
                    "success": success,
                    "checkpoint": CHECKPOINT,
                    "unnorm_key": UNNORM_KEY,
                    "task_suite": TASK_SUITE,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            rec = {
                "rollout_id": rollout_id,
                "task_id": task_id,
                "episode_idx": ep,
                "seed": seed,
                "success": success,
                "n_timesteps": n_steps,
                "wall_seconds": round(wall, 1),
                "steps_per_sec": round(n_steps / wall, 3) if wall > 0 else None,
                "frame_stats": frame_stats,
                "parity": parity_stats,
            }
            results.append(rec)
            n_done = len(results)
            n_succ = sum(r["success"] for r in results)
            print(
                f"[{n_done}] {rollout_id} success={success} steps={n_steps} "
                f"{wall:.0f}s ({rec['steps_per_sec']} step/s) "
                f"running={n_succ}/{n_done} ({100*n_succ/n_done:.1f}%)",
                flush=True,
            )

            # Rewrite the session summary after every rollout so a crash loses nothing.
            summary = {
                "session_start": session_start,
                "checkpoint": CHECKPOINT,
                "task_suite": TASK_SUITE,
                "unnorm_key": UNNORM_KEY,
                "seed_base": SEED_BASE,
                "max_steps": MAX_STEPS,
                "num_steps_wait": NUM_STEPS_WAIT,
                "n_rollouts": len(results),
                "n_success": n_succ,
                "success_rate": n_succ / len(results),
                "total_timesteps": sum(r["n_timesteps"] for r in results),
                "rollouts": results,
            }
            with open(out_dir / "session_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        env.close()

    n_succ = sum(r["success"] for r in results)
    print(f"\n=== DONE: {n_succ}/{len(results)} success "
          f"({100*n_succ/len(results):.1f}%), "
          f"{sum(r['n_timesteps'] for r in results)} timesteps ===", flush=True)


if __name__ == "__main__":
    main()
