"""
capture.py  --  capture schema version 2

Hidden-state + observation capture for OpenVLA action generation inside LIBERO.
Implements Step 2 as amended by Addendum A of init_instructions.md.

Why a second forward pass (Addendum A2.1-A2.2)
----------------------------------------------
A transformer emits the hidden state at position i only when the token at position i
is an *input*; the state at i is what predicts token i+1. The 7th action token is
produced as output on the final decode step and is never fed back in, so its hidden
state is never computed during `generate()`. After generation we therefore concatenate
prompt (length P) + all 7 generated action tokens and run one forward with
`use_cache=False, output_hidden_states=True`.

Position layout (Addendum A2.3)
-------------------------------
We slice positions [P-1, P, ..., P+6] -- eight in total. P-1 is the last prompt token;
P+6 is the 7th action token as input, the state that motivates the whole mechanism.

Note the model splices image patches into the sequence: `forward` builds
`[BOS, <256 projected patches>, text[1:]]`, so hidden-state indices are shifted by
n_patches relative to `input_ids`. Because the shift is a prefix insertion, the
trailing tokens keep their offsets from the *end*, so [P-1 .. P+6] is exactly the
last 8 positions of the hidden-state tensor regardless of prompt length.

Parity (Addendum A2.5)
----------------------
argmax of the re-forward logits at position i must equal the generated token at
position i+1, for i in [P-1, P+5] -- 7 comparisons. The logits at P+6 predict a token
that was never sampled, so they are not checked.

On the tolerance: Addendum A2.4 states the re-forward reproduces the generation pass
"up to floating-point nondeterminism". That nondeterminism is real and measured on this
box -- `generate()` decodes through flash-attn's cached-decode kernel while the
re-forward uses the prefill kernel, and over 32 bf16 layers the final logits differ by
~0.75 (max 1.375). Because OpenVLA maps adjacent action bins to adjacent vocabulary
ids, that noise flips the argmax at positions where the top-2 logits are nearly tied.
Measured over 280 positions: 98.9% agreement; every mismatch had a top-2 gap of
0.25-0.375, against a median gap of 8.5 where the passes agreed.

A wiring bug and a bf16 tie are therefore trivially separable, and the gate below keys
on that difference rather than on bit-exactness:

    a real bug (wrong slice / wrong image / off-by-one) -> ~0% agreement, large margins
    fp nondeterminism                                   -> >=95% agreement, tiny margins

Every mismatch margin and the agreement rate are recorded in the manifest, so the
threshold remains inspectable and revisable after the fact.

Precision (Addendum A2.6)
-------------------------
States compute in bf16 and are cast to fp16 on save; bf16's exponent range exceeds
fp16's, so the cast is checked with `isfinite` to stop an overflow becoming a silent
`inf` in the corpus.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# --- Capture configuration (locked) -------------------------------------------------
CAPTURE_SCHEMA_VERSION = 2
N_ACTION_TOKENS = 7           # OpenVLA emits 7 action tokens per timestep
N_POSITIONS = 8               # [P-1, P, ..., P+6]
STORE_DTYPE = torch.float16
POSITION_SCHEME = "positions [P-1 .. P+6]: last prompt token + 7 action-token positions"
JPEG_QUALITY = 90

# Parity gate. See the module docstring for the measurements these are drawn from.
# Measured on a full 520-step rollout: 96.6% argmax agreement, and every one of the 122
# mismatches had a top-2 gap <= 0.875 -- i.e. all sit inside the fp divergence band
# (final-logit |diff| mean 0.75, max 1.375), while agreeing positions had a median gap
# of 8.5. The band below is set above max measured divergence and far under the
# confident-decision regime, so wiring bugs and fp ties remain trivially separable.
PARITY_MIN_AGREEMENT = 0.95   # rollout-level; below this, suspect wiring not fp noise
PARITY_TIE_GAP = 2.0          # a mismatch is excusable only if its top-2 gap is under this

IMAGE_OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


@dataclass
class CaptureConfig:
    """Recorded verbatim into every manifest so a corpus is self-describing."""

    schema_version: int = CAPTURE_SCHEMA_VERSION
    n_action_tokens: int = N_ACTION_TOKENS
    n_positions: int = N_POSITIONS
    position_scheme: str = POSITION_SCHEME
    dtype: str = "float16"
    layers: str = "all hidden_states including embedding layer (index 0)"
    n_layers: Optional[int] = None  # filled on first capture (33 = embed + 32 blocks)
    jpeg_quality: int = JPEG_QUALITY
    parity_min_agreement: float = PARITY_MIN_AGREEMENT
    parity_tie_gap: float = PARITY_TIE_GAP


class ParityError(RuntimeError):
    """Raised when the teacher-forced pass fails to reproduce the generated tokens."""


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG-encode a raw uint8 RGB frame. Stored raw (not policy-resized): the policy
    input can be recomputed from the raw render, but not the reverse."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Inverse of `encode_jpeg` -- used by inspection tooling and later labeling."""
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def extract_state_obs(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Every non-image observation as float32: eef pose, gripper state, object states.

    Privileged sim state, which is acceptable because it shares the VLM observer's
    training-time-only status. Keys are task-dependent (each task exposes its own
    objects), so whatever the env provides is stored verbatim.
    """
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in obs.items()
        if k not in IMAGE_OBS_KEYS and np.asarray(v).dtype != np.uint8
    }


@torch.inference_mode()
def teacher_forced_capture(
    vla,
    pixel_values: torch.Tensor,
    sequences: torch.Tensor,
) -> Dict[str, Any]:
    """Re-forward `sequences` (prompt + 7 action tokens) and slice the 8 positions.

    Args:
        vla: the OpenVLAForActionPrediction model.
        pixel_values: image tensor exactly as passed to `predict_action`.
        sequences: `generated_outputs['sequences']`, shape (1, P+7). This is the
            post-`predict_action` prompt (which may have had token 29871 appended)
            followed by the 7 generated action tokens.

    Returns:
        'hidden' (n_layers, 8, d_model) fp16 CPU, and 'action_logits' (7, vocab) fp32
        CPU for the parity check.
    """
    if sequences.shape[0] != 1:
        raise ValueError(f"capture supports batch size 1 only, got {sequences.shape[0]}")

    attention_mask = torch.ones_like(sequences, dtype=torch.long, device=sequences.device)

    out = vla(
        input_ids=sequences,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden_states: tuple of (n_layers + 1) tensors, each (1, seq, d_model); index 0 is
    # the embedding layer, kept deliberately as the "signal already in the input" baseline.
    hidden = torch.stack([h[0, -N_POSITIONS:, :] for h in out.hidden_states], dim=0)

    hidden_fp16 = hidden.to(STORE_DTYPE).cpu()
    # bf16 range exceeds fp16 range; an overflow here would become a silent inf.
    if not torch.isfinite(hidden_fp16).all():
        n_bad = int((~torch.isfinite(hidden_fp16)).sum())
        raise ValueError(
            f"non-finite values after bf16->fp16 cast ({n_bad} elements); "
            "hidden states would be silently corrupted"
        )

    return {
        "hidden": hidden_fp16,
        "action_logits": out.logits[0, -N_POSITIONS:-1, :].float().cpu(),  # (7, vocab)
    }


def parity_check(action_logits: torch.Tensor, sequences: torch.Tensor) -> Dict[str, Any]:
    """Compare the re-forward's argmax against the tokens generation actually emitted.

    Logits at position i predict the token at i+1, so `action_logits` (positions
    [P-1 .. P+5]) is checked against the 7 generated action tokens. Position P+6 is
    excluded per Addendum A2.5 -- it predicts a token that was never sampled.

    Returns a dict including 'passed'; callers hard-fail the run when it is False.
    """
    generated = sequences[0, -N_ACTION_TOKENS:].cpu().tolist()
    reforwarded = action_logits.argmax(dim=-1).tolist()
    agreement = [int(a == b) for a, b in zip(generated, reforwarded)]

    top2 = torch.topk(action_logits, 2, dim=-1).values
    gaps = (top2[:, 0] - top2[:, 1]).tolist()  # decisiveness of the re-forward argmax

    mismatch_gaps = [round(g, 4) for g, ok in zip(gaps, agreement) if not ok]
    n_agree = int(sum(agreement))
    rate = n_agree / N_ACTION_TOKENS

    # A single timestep has only 7 positions, so a rate threshold is too coarse here (one
    # tie already drops it to 0.857). At this level we only ask that any disagreement sit
    # inside the fp-nondeterminism band Addendum A2.4 anticipates; the rate-based gate is
    # applied at rollout level, where it is statistically meaningful.
    passed = all(g < PARITY_TIE_GAP for g in mismatch_gaps)

    return {
        "passed": bool(passed),
        "agreement_rate": round(rate, 4),
        "n_agree": n_agree,
        "n_positions": N_ACTION_TOKENS,
        "generated_token_ids": generated,
        "reforwarded_token_ids": reforwarded,
        "per_position_agreement": agreement,
        "top2_gaps": [round(g, 4) for g in gaps],
        "mismatch_top2_gaps": mismatch_gaps,
        "tie_gap_threshold": PARITY_TIE_GAP,
        "min_agreement_threshold": PARITY_MIN_AGREEMENT,
    }


class RolloutCapture:
    """Accumulates the per-timestep record for one rollout, then flushes to disk.

    One `.pt` per rollout plus a sidecar `.json` manifest. The writer flushes at the end
    of each rollout, so a crash costs one rollout and never the corpus. All arrays share
    the same length T (Addendum A1).
    """

    def __init__(self, out_dir: Path, rollout_id: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_id = rollout_id
        self.config = CaptureConfig()

        self.hidden: List[torch.Tensor] = []
        self.frames_agentview: List[bytes] = []
        self.frames_wrist: List[bytes] = []
        self.obs: List[Dict[str, np.ndarray]] = []
        self.actions: List[np.ndarray] = []
        self.parities: List[Dict[str, Any]] = []

    def add(
        self,
        hidden: torch.Tensor,
        raw_obs: Dict[str, Any],
        action: np.ndarray,
        parity: Dict[str, Any],
    ) -> None:
        """Append one timestep. `raw_obs` is the LIBERO observation the policy acted on;
        `action` is the executed 7-DoF command sent to `env.step`."""
        if self.config.n_layers is None:
            self.config.n_layers = int(hidden.shape[0])

        self.hidden.append(hidden)
        self.frames_agentview.append(encode_jpeg(raw_obs["agentview_image"]))
        if "robot0_eye_in_hand_image" in raw_obs:
            self.frames_wrist.append(encode_jpeg(raw_obs["robot0_eye_in_hand_image"]))
        self.obs.append(extract_state_obs(raw_obs))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.parities.append(parity)

    def __len__(self) -> int:
        return len(self.hidden)

    def parity_summary(self) -> Dict[str, Any]:
        """Rollout-level parity statistics, for the manifest."""
        if not self.parities:
            return {}
        n_pos = sum(p["n_positions"] for p in self.parities)
        n_agree = sum(p["n_agree"] for p in self.parities)
        all_gaps = [g for p in self.parities for g in p["mismatch_top2_gaps"]]
        return {
            "positions_checked": n_pos,
            "positions_agreed": n_agree,
            "agreement_rate": round(n_agree / n_pos, 6) if n_pos else None,
            "n_mismatches": len(all_gaps),
            "max_mismatch_top2_gap": round(max(all_gaps), 4) if all_gaps else None,
            "mismatches_within_tie_band": all(g < PARITY_TIE_GAP for g in all_gaps),
            "rate_ok": (n_agree / n_pos) >= PARITY_MIN_AGREEMENT if n_pos else False,
        }

    def flush(self, meta: Dict[str, Any]) -> Dict[str, str]:
        """Write the rollout record + manifest. `meta` carries rollout-level fields."""
        if not self.hidden:
            raise ValueError(f"rollout {self.rollout_id} captured no timesteps")

        states = torch.stack(self.hidden, dim=0)  # (T, n_layers, 8, d_model)
        T = int(states.shape[0])

        # Object keys are task-dependent but fixed within a rollout; stack per key.
        obs_keys = sorted(self.obs[0].keys())
        obs_stacked = {
            k: np.stack([o[k] for o in self.obs], axis=0).astype(np.float32)
            for k in obs_keys
        }
        actions = np.stack(self.actions, axis=0).astype(np.float32)  # (T, 7)

        assert len(self.frames_agentview) == T, "frame/hidden length mismatch"
        assert actions.shape[0] == T, "action/hidden length mismatch"
        assert all(v.shape[0] == T for v in obs_stacked.values()), "obs/hidden length mismatch"

        parity_summary = self.parity_summary()
        record = {
            "hidden_states": states,
            "frames_agentview": self.frames_agentview,
            "frames_wrist": self.frames_wrist,
            "obs": obs_stacked,
            "actions": actions,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "parity_summary": parity_summary,
            **meta,
        }

        pt_path = self.out_dir / f"{self.rollout_id}.pt"
        json_path = self.out_dir / f"{self.rollout_id}.json"
        torch.save(record, pt_path)

        manifest = {
            **meta,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "n_timesteps": T,
            "tensor_shape": list(states.shape),
            "obs_keys": obs_keys,
            "has_wrist_frames": len(self.frames_wrist) == T,
            "actions_shape": list(actions.shape),
            "capture_config": asdict(self.config),
            "parity_summary": parity_summary,
            "pt_file": pt_path.name,
            "size_bytes": pt_path.stat().st_size,
        }
        with open(json_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Free RAM before the next rollout.
        self.hidden, self.frames_agentview, self.frames_wrist = [], [], []
        self.obs, self.actions, self.parities = [], [], []

        return {"pt": str(pt_path), "json": str(json_path)}
