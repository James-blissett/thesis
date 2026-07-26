"""
capture.py

Hidden-state capture for OpenVLA action generation inside LIBERO.

Why a second forward pass
-------------------------
`generate()` yields hidden states only for the positions it actually decodes at.
At the step that emits the 7th (final) action token the model has not yet consumed
that token, so no hidden state at the 7th action-token position ever exists in the
`generate()` output. To get a hidden state at every action-token position we re-run
one teacher-forced forward over `prompt + all 7 action tokens` with
`output_hidden_states=True`.

Position layout
---------------
`PrismaticForConditionalGeneration.forward` builds its multimodal sequence as

    [ BOS , <256 projected image patches> , text_tokens[1:] ]

so the hidden-state sequence is longer than `input_ids` by `n_patches`, and every
text token after BOS is shifted right by `n_patches`. The consequence that matters
here: the trailing text tokens are never displaced relative to the *end* of the
sequence, so the 8 positions we want (last prompt token + the 7 action tokens) are
exactly the **last 8** positions of the hidden-state tensor, whatever the prompt length.

Parity
------
Under greedy decoding the teacher-forced pass must reproduce the generated action
tokens. Logits at position p predict the token at p+1, so the logits at the last 8
positions excluding the final one (`[-8:-1]`) predict the 7 action tokens.
`parity_check` asserts argmax agreement at all 7; a mismatch means the re-forward
does not correspond to the rollout and the captured states are meaningless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# --- Capture configuration (locked) -------------------------------------------------
N_ACTION_TOKENS = 7           # OpenVLA emits 7 action tokens per timestep
N_POSITIONS = 8               # last prompt token + the 7 action-token positions
STORE_DTYPE = torch.float16   # fp16 on disk -> ~2.1 MB / timestep
POSITION_SCHEME = "last_prompt_token + 7 action tokens (last 8 sequence positions)"


@dataclass
class CaptureConfig:
    """Recorded verbatim into every manifest so a corpus is self-describing."""

    n_action_tokens: int = N_ACTION_TOKENS
    n_positions: int = N_POSITIONS
    position_scheme: str = POSITION_SCHEME
    dtype: str = "float16"
    layers: str = "all hidden_states including embedding layer (index 0)"
    n_layers: Optional[int] = None  # filled in on first capture (33 = embed + 32 blocks)


class ParityError(RuntimeError):
    """Raised when the teacher-forced pass fails to reproduce the generated tokens."""


@torch.inference_mode()
def teacher_forced_capture(
    vla,
    pixel_values: torch.Tensor,
    sequences: torch.Tensor,
) -> Dict[str, Any]:
    """Re-forward `sequences` (prompt + 7 action tokens) and pull the 8 positions.

    Args:
        vla: the OpenVLAForActionPrediction model.
        pixel_values: image tensor exactly as passed to `predict_action`.
        sequences: `generated_outputs['sequences']`, shape (1, L+7). This is the
            post-`predict_action` prompt (which may have had token 29871 appended)
            followed by the 7 generated action tokens.

    Returns:
        dict with 'hidden' (n_layers, 8, d_model) fp16 on CPU and 'action_logits'
        (7, vocab) float32 on CPU for the parity check.
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

    # hidden_states: tuple of length n_layers (33), each (1, seq, d_model).
    # Index 0 is the embedding layer -- kept deliberately as the "signal already in
    # the input" baseline the probe is later compared against.
    hidden = torch.stack(
        [h[0, -N_POSITIONS:, :] for h in out.hidden_states], dim=0
    )  # (n_layers, 8, d_model)

    # Logits at the last 8 positions minus the final one predict the 7 action tokens.
    action_logits = out.logits[0, -N_POSITIONS:-1, :]  # (7, vocab)

    return {
        "hidden": hidden.to(STORE_DTYPE).cpu(),
        "action_logits": action_logits.float().cpu(),
    }


def parity_check(action_logits: torch.Tensor, sequences: torch.Tensor) -> Dict[str, Any]:
    """Verify the teacher-forced pass reproduces the generated action tokens.

    Args:
        action_logits: (7, vocab) from `teacher_forced_capture`.
        sequences: (1, L+7) generated sequence.

    Returns:
        dict with 'passed', the generated and re-forwarded token ids, and a per-position
        agreement list. Callers hard-fail the run when 'passed' is False.
    """
    generated = sequences[0, -N_ACTION_TOKENS:].cpu().tolist()
    reforwarded = action_logits.argmax(dim=-1).tolist()
    agreement = [int(a == b) for a, b in zip(generated, reforwarded)]
    return {
        "passed": all(agreement),
        "generated_token_ids": generated,
        "reforwarded_token_ids": reforwarded,
        "per_position_agreement": agreement,
        "n_agree": int(sum(agreement)),
        "n_positions": N_ACTION_TOKENS,
    }


class RolloutCapture:
    """Accumulates per-timestep hidden states for one rollout, then flushes to disk.

    One `.pt` per rollout plus a sidecar `.json` manifest. The writer flushes at the
    end of each rollout so a crash costs one rollout, never the corpus.
    """

    def __init__(self, out_dir: Path, rollout_id: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_id = rollout_id
        self.frames: List[torch.Tensor] = []
        self.config = CaptureConfig()

    def add(self, hidden: torch.Tensor) -> None:
        """Append one timestep's (n_layers, 8, d_model) fp16 tensor."""
        if self.config.n_layers is None:
            self.config.n_layers = int(hidden.shape[0])
        self.frames.append(hidden)

    def __len__(self) -> int:
        return len(self.frames)

    def flush(self, meta: Dict[str, Any]) -> Dict[str, str]:
        """Write the rollout tensor + manifest. `meta` carries rollout-level fields."""
        if not self.frames:
            raise ValueError(f"rollout {self.rollout_id} captured no timesteps")

        states = torch.stack(self.frames, dim=0)  # (T, n_layers, 8, d_model)
        pt_path = self.out_dir / f"{self.rollout_id}.pt"
        json_path = self.out_dir / f"{self.rollout_id}.json"

        torch.save({"hidden_states": states, **meta}, pt_path)

        manifest = {
            **meta,
            "n_timesteps": int(states.shape[0]),
            "tensor_shape": list(states.shape),
            "capture_config": asdict(self.config),
            "pt_file": pt_path.name,
            "size_bytes": pt_path.stat().st_size,
        }
        with open(json_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.frames = []  # free RAM before the next rollout
        return {"pt": str(pt_path), "json": str(json_path)}
