"""
capture_v2.py  --  Rollout Collection v2

Hidden-state / logit / frame / observation capture for OpenVLA inside LIBERO.
Replaces `capture.py` (schema v2); the v1 module is left untouched so the existing
50-rollout corpus and its analysis scripts keep working.

What changed from v1
--------------------
* No teacher-forced re-forward pass, and therefore no parity check. v1 ran a second
  forward over prompt+7 action tokens purely to obtain h[N+7], the state of the 7th
  action token as *input*. v2 does not store that position -- it decodes nothing -- so
  the states are read straight out of `generate()`'s own `output_hidden_states`. That
  halves capture compute and removes fp-nondeterminism from the corpus entirely: the
  stored logits are literally the ones generation decoded, so argmax agreement with the
  emitted tokens is exact, not statistical.
* Seven decision states h[N..N+6] instead of eight positions [P-1 .. P+6].
* Action logits stored (Change 4).
* Frames streamed to disk as JPEGs rather than held in the .pt.
* No early termination: every episode runs the full policy-step budget, and the success
  predicate is logged per step.

Position layout (Change 3)
--------------------------
Under causal masking the state at position i is what decodes token i+1. OpenVLA emits 7
action tokens, so the seven states that produce the seven DoF are h[N .. N+6], where N is
the last prompt position:

    decision[k]  <- generate step k   ->  decodes action token k  ->  DoF k
    k=0: h[N]     the last prompt token   -> x
    k=6: h[N+6]   the 6th decode step     -> gripper

Note the shape asymmetry that makes this easy to get silently wrong:
`hidden_states[0]` is the prefill and its tensors are (1, N_seq, d), so the wanted row is
`-1`; every later step is (1, 1, d), so the wanted row is `0`. Indexing the prefill with
`0` yields a valid-looking tensor of the first prompt token. `assert_decode_shapes`
checks the asymmetry directly rather than trusting it.

Logit bin ordering
------------------
OpenVLA maps action bins onto the tail of the Llama vocabulary *in reverse*:
`bin = vocab_size - token_id`, so a higher token id is a lower (more negative) action.
Slicing the vocabulary tail in token order therefore gives bins in descending action
value. Stored logits are flipped to ascending action value, so index 0 is the most
negative action and index 255 the most positive. This matters downstream: entropy is
order-invariant but dispersion and multimodality are not.

Length accounting (Change 2)
----------------------------
The eval loop steps a dummy action for `num_steps_wait` steps before the policy acts.
Those steps have observations and frames but no forward pass, hence no hidden states and
no logits. Arrays are stored at two lengths, both recorded in the manifest:

    T_env    = num_steps_wait + T_total  (obs, frames, executed actions, success flags)
    T_total  = policy steps only         (hidden, logits, action token ids)

`policy_step_env_t` maps a hidden-state row onto its env step. See the note in
gen_rollouts_v2.py about the 510-vs-520 arithmetic in the handoff.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

CAPTURE_SCHEMA_VERSION = 3          # v2 corpus; schema 2 was the previous capture.py
N_ACTION_TOKENS = 7                 # one per DoF: x, y, z, roll, pitch, yaw, gripper
N_ACTION_BINS = 256                 # vocabulary tail width
STORE_DTYPE = torch.float16
JPEG_QUALITY = 85
DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

IMAGE_OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
FRAME_VIEWS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}

# What each stored position is, spelled out in the manifest rather than left to convention.
POSITION_SEMANTICS_P3 = [
    "mean_over_7_decision_states",
    "h_N_decodes_x__also_last_prompt_token",
    "h_N+6_decodes_gripper",
]
POSITION_SEMANTICS_P7 = [
    f"{'h_N' if k == 0 else f'h_N+{k}'}_decodes_{DOF_NAMES[k]}" for k in range(N_ACTION_TOKENS)
]

FRAME_ORIENTATION = (
    "raw env render, exactly as returned by the simulator. The policy input is "
    "img[::-1, ::-1] (180 degree rotation, see get_libero_image); apply the same "
    "rotation for an upright view."
)


class CaptureError(RuntimeError):
    """Raised when a structural invariant of the capture fails. Never caught."""


def action_bin_slice(vla) -> slice:
    """The vocabulary window holding the 256 action bins.

    Derived from the model rather than hardcoded. `predict_action` computes
    `bin = self.vocab_size - token_id` with `self.vocab_size = text_config.vocab_size -
    pad_to_multiple_of`, so the action tokens are exactly the ids in
    [vocab_size - 256, vocab_size).
    """
    v = int(vla.vocab_size)
    return slice(v - N_ACTION_BINS, v)


def assert_bin_slice_agrees(vla, tokenizer) -> None:
    """`ActionTokenizer` decodes with the *tokenizer's* vocab_size, `predict_action` with
    the *model's*. If those ever diverge, every stored logit array is keyed to a different
    bin origin than the executed action and the field is worthless."""
    if int(vla.vocab_size) != int(tokenizer.vocab_size):
        raise CaptureError(
            f"action-bin origin mismatch: model.vocab_size={vla.vocab_size} but "
            f"tokenizer.vocab_size={tokenizer.vocab_size}. ACTION_BIN_SLICE is unsafe."
        )


def assert_decode_shapes(hidden_states) -> int:
    """Verify the prefill/decode shape asymmetry and return the prompt length N_seq.

    hidden_states[0][l] is (1, N_seq, d) -- the prefill, N_seq > 1.
    hidden_states[s][l] is (1, 1, d) for s >= 1 -- one decode step each.
    """
    n_steps = len(hidden_states)
    if n_steps != N_ACTION_TOKENS:
        raise CaptureError(
            f"expected {N_ACTION_TOKENS} generate steps, got {n_steps}; max_new_tokens wrong"
        )
    prefill = hidden_states[0][0]
    if prefill.ndim != 3 or prefill.shape[1] <= 1:
        raise CaptureError(
            f"prefill hidden state has shape {tuple(prefill.shape)}; expected (1, N_seq>1, d). "
            "The prefill/decode index asymmetry is not what this code assumes."
        )
    for s in range(1, n_steps):
        step = hidden_states[s][0]
        if step.shape[1] != 1:
            raise CaptureError(
                f"decode step {s} hidden state has shape {tuple(step.shape)}; expected (1, 1, d)"
            )
    return int(prefill.shape[1])


def extract_decision_states(hidden_states) -> torch.Tensor:
    """The seven states that decode the seven action tokens -> (7, n_layers, d), fp16 CPU.

    Index -1 on the prefill, index 0 on each decode step. See the module docstring.
    """
    n_layers = len(hidden_states[0])
    prompt_last = torch.stack([hidden_states[0][l][0, -1] for l in range(n_layers)])
    decode = [
        torch.stack([hidden_states[s][l][0, 0] for l in range(n_layers)])
        for s in range(1, N_ACTION_TOKENS)
    ]
    decision = torch.stack([prompt_last] + decode)  # (7, n_layers, d)

    out = decision.to(STORE_DTYPE).cpu()
    # bf16's exponent range exceeds fp16's; an overflow here would become a silent inf.
    if not torch.isfinite(out).all():
        n_bad = int((~torch.isfinite(out)).sum())
        raise CaptureError(f"non-finite values after bf16->fp16 cast ({n_bad} elements)")
    return out


def pool_positions(decision: torch.Tensor, n_positions: int) -> torch.Tensor:
    """(7, n_layers, d) -> (n_layers, P, d) in the stored position order.

    P=7: all seven decision states, unpooled, in DoF order.
    P=3: [mean over the seven, h[N] (decodes x), h[N+6] (decodes gripper)].
    """
    if n_positions == N_ACTION_TOKENS:
        stacked = decision
    elif n_positions == 3:
        mean = decision.float().mean(dim=0).to(decision.dtype)
        stacked = torch.stack([mean, decision[0], decision[-1]])
    else:
        raise CaptureError(f"n_positions must be 3 or 7, got {n_positions}")
    return stacked.permute(1, 0, 2).contiguous()  # (n_layers, P, d)


def extract_action_logits(gen_logits, bin_slice: slice) -> torch.Tensor:
    """(7, 256) fp16 CPU, bins in ascending action value.

    `gen_logits` is `generate(...).logits`: one (1, vocab) tensor per generate step, from
    the same forward pass as the corresponding decision state, so index k of the returned
    tensor aligns positionally with decision[k].
    """
    if len(gen_logits) != N_ACTION_TOKENS:
        raise CaptureError(f"expected {N_ACTION_TOKENS} logit steps, got {len(gen_logits)}")
    raw = torch.stack([g[0] for g in gen_logits])          # (7, vocab), token-id order
    binned = raw[:, bin_slice].flip(-1)                    # -> ascending action value
    if binned.shape != (N_ACTION_TOKENS, N_ACTION_BINS):
        raise CaptureError(f"action logits have shape {tuple(binned.shape)}")
    out = binned.to(STORE_DTYPE).cpu()
    if not torch.isfinite(out).all():
        raise CaptureError("non-finite action logits after fp16 cast")
    return out


def bin_index_to_token_id(bin_index: np.ndarray, vocab_size: int) -> np.ndarray:
    """Inverse of the storage flip: stored index i <-> vocabulary id vocab_size-1-i."""
    return vocab_size - 1 - np.asarray(bin_index)


def token_id_to_bin_index(token_id: np.ndarray, vocab_size: int) -> np.ndarray:
    """Forward of the storage flip. Only meaningful for ids inside the action window."""
    return vocab_size - 1 - np.asarray(token_id)


def logit_alignment(logits: torch.Tensor, token_ids, vocab_size: int) -> Dict[str, Any]:
    """Check the stored logits against the tokens generation actually emitted.

    These are the *same* logits generate decoded with, under greedy sampling, so this is
    an identity rather than a statistical check -- but it must be phrased on values, not
    on argmax indices. `generate` argmaxes over the vocabulary in token-id order while the
    stored array is flipped to action-value order, and `torch.argmax` breaks ties by
    first index. A tie therefore resolves to the lowest token id for generate and the
    highest for the stored array, which is a disagreement about nothing.

    Ties are not rare: logits are bf16 (8 mantissa bits, ~0.125 spacing at these
    magnitudes) and adjacent action bins are frequently within that. v1's teacher-forced
    parity saw near-ties at ~3.4% of positions.

    So the invariant checked is: the generated token lies inside the action window, and
    its stored logit equals the maximum over that window. Anything else -- a wrong slice,
    a wrong flip, an off-by-one in the generate-step alignment, or an unexpected logits
    processor reranking the candidates -- shows up as a strictly positive shortfall.
    """
    tok = np.asarray(token_ids).astype(np.int64)
    lo = vocab_size - N_ACTION_BINS
    in_window = (tok >= lo) & (tok < vocab_size)

    lf = logits.float()
    slice_max = lf.max(dim=-1).values.numpy()
    argmax_bin = lf.argmax(dim=-1).numpy().astype(np.int64)

    bin_of_token = np.clip(token_id_to_bin_index(tok, vocab_size), 0, N_ACTION_BINS - 1)
    tok_logit = lf.numpy()[np.arange(len(tok)), bin_of_token]

    shortfall = np.where(in_window, slice_max - tok_logit, np.inf)
    exact = in_window & (shortfall <= 0.0)
    index_equal = exact & (argmax_bin == bin_of_token)

    return {
        "n_positions": int(len(tok)),
        "n_ok": int(exact.sum()),
        "n_index_equal": int(index_equal.sum()),
        "n_tie_resolved_differently": int((exact & ~index_equal).sum()),
        "n_outside_window": int((~in_window).sum()),
        "n_real_mismatch": int((~exact).sum()),
        "max_shortfall": float(shortfall[np.isfinite(shortfall)].max(initial=0.0)),
        "ok": bool(exact.all()),
        "generated_token_ids": tok.tolist(),
        "argmax_bins": argmax_bin.tolist(),
        "token_bins": bin_of_token.tolist(),
        "shortfalls": [None if not np.isfinite(s) else round(float(s), 5) for s in shortfall],
    }


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def extract_state_obs(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Every non-image observation as float32: eef pose, gripper state, object poses.

    Privileged sim state, acceptable because it shares the VLM observer's
    training-time-only status. Keys are task-dependent, so whatever the env exposes is
    stored verbatim.
    """
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in obs.items()
        if k not in IMAGE_OBS_KEYS and np.asarray(v).dtype != np.uint8
    }


class RolloutWriter:
    """Accumulates one rollout and flushes it to its own directory.

    `manifest.json` is written last and carries `complete: true`. A directory without a
    complete manifest is a partial rollout and is discarded on resume -- that is the whole
    resumability contract, so nothing else may be written after the manifest.

    Frames stream to disk as they arrive; hidden states and logits accumulate in RAM
    (about 420 MB at P=3, 980 MB at P=7) and are stacked once at flush.
    """

    def __init__(self, root: Path, rollout_id: str, n_positions: int) -> None:
        self.dir = Path(root) / rollout_id
        self.frames_dir = self.dir / "frames"
        self.rollout_id = rollout_id
        self.n_positions = n_positions

        self.hidden: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []
        self.action_token_ids: List[np.ndarray] = []
        self.policy_step_env_t: List[int] = []

        self.obs: List[Dict[str, np.ndarray]] = []
        self.executed: List[np.ndarray] = []
        self.action_raw: List[np.ndarray] = []
        self.is_wait: List[bool] = []
        self.success_now: List[bool] = []
        self.success_latched: List[bool] = []
        self.env_done: List[bool] = []
        self.frame_means: List[float] = []
        self.n_frames_written = 0
        self.n_layers: Optional[int] = None

    def prepare(self) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def add_step(
        self,
        t_env: int,
        is_wait_step: bool,
        raw_obs: Dict[str, Any],
        executed_action: np.ndarray,
        hidden: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        action_token_ids: Optional[np.ndarray] = None,
        action_raw: Optional[np.ndarray] = None,
    ) -> None:
        """Append one environment step. Wait steps pass hidden/logits as None."""
        for view, key in FRAME_VIEWS.items():
            if key in raw_obs:
                path = self.frames_dir / f"{view}_{t_env:04d}.jpg"
                path.write_bytes(encode_jpeg(raw_obs[key]))
                self.n_frames_written += 1

        # EGL failure produces well-formed all-black frames that pass every structural
        # check, so track the raw render's brightness explicitly.
        self.frame_means.append(float(np.mean(raw_obs["agentview_image"])))

        self.obs.append(extract_state_obs(raw_obs))
        self.executed.append(np.asarray(executed_action, dtype=np.float32))
        self.action_raw.append(
            np.full(N_ACTION_TOKENS, np.nan, dtype=np.float32)
            if action_raw is None
            else np.asarray(action_raw, dtype=np.float32)
        )
        self.is_wait.append(bool(is_wait_step))

        if is_wait_step:
            if hidden is not None or logits is not None:
                raise CaptureError("wait step carries hidden states; loop is wired wrong")
            return

        if hidden is None or logits is None or action_token_ids is None:
            raise CaptureError(f"policy step at t_env={t_env} missing capture payload")
        if self.n_layers is None:
            self.n_layers = int(hidden.shape[0])
        self.hidden.append(hidden)
        self.logits.append(logits)
        self.action_token_ids.append(np.asarray(action_token_ids, dtype=np.int32))
        self.policy_step_env_t.append(int(t_env))

    def add_outcome(self, success_now: bool, success_latched: bool, env_done: bool) -> None:
        """Success predicate for the step just taken. Called once per `add_step`."""
        self.success_now.append(bool(success_now))
        self.success_latched.append(bool(success_latched))
        self.env_done.append(bool(env_done))

    def frame_stats(self) -> Dict[str, float]:
        return {
            "mean_pixel": float(np.mean(self.frame_means)) if self.frame_means else 0.0,
            "min_frame_mean": float(np.min(self.frame_means)) if self.frame_means else 0.0,
            "max_frame_mean": float(np.max(self.frame_means)) if self.frame_means else 0.0,
            "n_frames": self.n_frames_written,
        }

    def flush(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Write hidden.pt, logits.pt, actions.npz, obs.npz, then manifest.json."""
        if not self.hidden:
            raise CaptureError(f"rollout {self.rollout_id} captured no policy steps")

        T_total = len(self.hidden)
        T_env = len(self.executed)
        if len(self.success_now) != T_env:
            raise CaptureError(
                f"outcome/step length mismatch: {len(self.success_now)} vs {T_env}"
            )

        hidden = torch.stack(self.hidden, dim=0)   # (T, n_layers, P, d)
        logits = torch.stack(self.logits, dim=0)   # (T, 7, 256)
        if hidden.shape[2] != self.n_positions:
            raise CaptureError(f"hidden has {hidden.shape[2]} positions, expected {self.n_positions}")
        if hidden.dtype != STORE_DTYPE or logits.dtype != STORE_DTYPE:
            raise CaptureError("stored tensors are not fp16")

        obs_keys = sorted(self.obs[0].keys())
        obs_stacked = {
            k: np.stack([o[k] for o in self.obs], axis=0).astype(np.float32) for k in obs_keys
        }
        if any(v.shape[0] != T_env for v in obs_stacked.values()):
            raise CaptureError("obs/step length mismatch")

        torch.save(hidden, self.dir / "hidden.pt")
        torch.save(logits, self.dir / "logits.pt")

        np.savez_compressed(
            self.dir / "actions.npz",
            executed=np.stack(self.executed).astype(np.float32),              # (T_env, 7)
            action_raw=np.stack(self.action_raw).astype(np.float32),          # (T_env, 7)
            action_token_ids=np.stack(self.action_token_ids).astype(np.int32),  # (T_total, 7)
            policy_step_env_t=np.asarray(self.policy_step_env_t, dtype=np.int32),
            success_now=np.asarray(self.success_now, dtype=bool),             # (T_env,)
            success_latched=np.asarray(self.success_latched, dtype=bool),
            env_done=np.asarray(self.env_done, dtype=bool),
            is_wait_step=np.asarray(self.is_wait, dtype=bool),
            t_effective=np.asarray(
                [t - meta["num_steps_wait"] for t in range(T_env)], dtype=np.int32
            ),
        )
        np.savez_compressed(self.dir / "obs.npz", **obs_stacked)

        sizes = {p.name: p.stat().st_size for p in sorted(self.dir.glob("*.*")) if p.is_file()}
        sizes["frames/"] = sum(p.stat().st_size for p in self.frames_dir.glob("*.jpg"))

        manifest = {
            **meta,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "T_total": T_total,
            "T_env_total": T_env,
            "n_positions": self.n_positions,
            "layers_stored": int(hidden.shape[1]),
            "token_positions": self.n_positions,
            "position_semantics": (
                POSITION_SEMANTICS_P7 if self.n_positions == N_ACTION_TOKENS else POSITION_SEMANTICS_P3
            ),
            "dtype": "float16",
            "hidden_shape": list(hidden.shape),
            "logits_shape": list(logits.shape),
            "logit_bin_order": (
                "ascending action value; stored index i corresponds to vocabulary id "
                "(vocab_size - 1 - i). Bin 0 is the most negative action."
            ),
            "length_semantics": (
                "hidden/logits/action_token_ids have T_total rows (policy steps only). "
                "obs/frames/executed/success_* have T_env_total rows (all env steps, "
                "wait steps included). policy_step_env_t maps a policy row to its env step."
            ),
            "frame_orientation": FRAME_ORIENTATION,
            "jpeg_quality": JPEG_QUALITY,
            "obs_keys": obs_keys,
            "frame_views": sorted(FRAME_VIEWS),
            "frame_stats": self.frame_stats(),
            "file_sizes_bytes": sizes,
            "size_bytes": sum(sizes.values()),
            "complete": True,
        }
        with open(self.dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Free RAM before the next rollout.
        self.hidden, self.logits = [], []
        self.obs, self.executed, self.action_raw = [], [], []
        return manifest


REQUIRED_MANIFEST_FIELDS = (
    "rollout_id", "task_idx", "task_name", "task_description", "trial_idx",
    "init_state_index", "T_total", "num_steps_wait", "t_success", "success_final",
    "success_ever", "layers_stored", "token_positions", "position_semantics", "dtype",
    "model_checkpoint", "unnorm_key", "action_bin_slice", "git_commit",
    "capture_started_utc", "termination_reason",
)


def manifest_is_complete(path: Path) -> bool:
    """True only if the manifest parses, is flagged complete, and has every field.

    A missing field is a STOP condition for the corpus, so it is also a reason to redo
    the rollout on resume rather than skip it.
    """
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        return False
    if not m.get("complete"):
        return False
    return all(k in m for k in REQUIRED_MANIFEST_FIELDS)
