"""
common3.py

Shared definitions for the deep-MLP run (deep_mlp_handoff.md, handoff version 1): the
pre-norm residual MLP, the training recipe, the fit grid, work-file paths and the
formats every abc3 script reads or writes. Everything else (index, kept rows, splits,
pools D0/D2, test sets, cells, row weights, running-sum scoring, M1, M1-per-task,
cut-variant conformal prediction, paired bootstrap) is imported unchanged from
analysis/abc2 (and through it analysis/abc).

Fit grid (§2): input in {A, C} x depth in {2, 6, 12} x pool in {D0, D2} x seed in {0, 1, 2}
x regularisation (dropout p, weight decay wd) in REG_GRID = 144 fits.
    A input = the pooled hidden state at the layer A's linear probe chose for (pool, seed)
              in abc2 (results/abc2/selections.csv, A_probe), read from /data/tmp/abc2_features
    C input = C's final-detector inputs (132 prepared series), built exactly as
              stage10_mlp.Run.c_features (Stage 3's path at the pool's betas)
A "group" is (input, depth, pool, seed); its detector key is DET_KEY = f"{input}-d{depth}"
("A-d2", ..., "C-d12").

Work file (one per fit; written by train3.py, read by eval3.py):
    WORK_DIR/<pool>/<input>_d<depth>_seed<s>_p<p>_wd<wd>.pkl, a dict:
      spec          dict (fit_spec): what the fit depends on; a file whose spec differs
                    from the current one is redone
      status        "ok" | "failed" (12-layer divergence twice, §5)
      p             float32 (N, 520): P(failure) at the best epoch for every rollout
                    (N = 1000, index order); None if status == "failed"
      best_epoch, epochs_run, best_score (best eval_seen M1-per-task), stopped_early
      history       list of per-epoch dicts: epoch, train_loss (row-weighted mean BCE),
                    lr_end, eval_seen_m1_per_task, eval_seen_m1_pooled,
                    train_m1_per_task, train_m1_pooled
      lr            learning rate of the kept attempt (1e-3, or 3e-4 after a retry)
      attempts      list of {lr, diverged, reason, best_score, epochs_run} per attempt
      diverged      bool (kept attempt diverged; only possible with status failed or
                    depth != 12, where the rule is flagged but not acted on)
      seconds, n_train_rows, d (input width), n_params, precision ("bf16" | "fp32")
      state         best-epoch state_dict, fp16 tensors on the CPU (checkpoint, <= 3.6 MB)

Selection (eval3.py): per group, the REG_GRID entry with the highest best_score among
status "ok" fits; ties go to the first in REG_GRID order (abc2's argmax rule). A group
whose four fits all failed is reported as failed.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ABC2 = REPO / "analysis" / "abc2"
sys.path.insert(0, str(ABC2))
import data2  # noqa: E402,F401  (puts analysis/abc and analysis/ on sys.path)

ABC2_RESULTS = REPO / "results" / "abc2"
RESULTS = REPO / "results" / "abc3"
WORK_DIR = Path("/data/tmp/abc3_work")
POOLS = ("D0", "D2")
SEEDS = (0, 1, 2)
INPUTS = ("A", "C")
DEPTHS = (2, 6, 12)
WIDTH = 256
REG_GRID = ((0.0, 1e-2), (0.0, 1e-1), (0.3, 1e-2), (0.3, 1e-1))   # (dropout p, weight decay)
LINEAR = {"A": "A", "C": "C"}             # key of the linear detector in results/abc2/scores/<pool>_seed<s>.npz
ABC2_MLP = {"A": "A-MLP", "C": "C-MLP"}   # key in results/abc2/scores/<pool>_seed<s>_mlp.npz

HP = {"width": WIDTH, "lr": 1e-3, "lr_retry": 3e-4, "warmup_frac": 0.05, "schedule": "linear warm-up + cosine",
      "batch": 4096, "max_epochs": 50, "patience": 5, "grad_clip": 1.0, "optimizer": "AdamW",
      "weight_decay_on": "every parameter (as abc2)", "loss": "row-weighted BCE (batch mean)",
      "reg_grid": [list(r) for r in REG_GRID]}


def det_key(inp: str, depth: int) -> str:
    return f"{inp}-d{depth}"


def reg_tag(p: float, wd: float) -> str:
    return f"p{p:g}_wd{wd:g}"


def work_path(pool: str, inp: str, depth: int, seed: int, p: float, wd: float, work_dir: Path = WORK_DIR) -> Path:
    return Path(work_dir) / pool / f"{inp}_d{depth}_seed{seed}_{reg_tag(p, wd)}.pkl"


def all_fits():
    """Every (inp, depth, pool, seed, p, wd) of the grid, in a fixed order."""
    return [(i, d, c, s, p, wd) for i in INPUTS for c in POOLS for s in SEEDS for d in DEPTHS
            for (p, wd) in REG_GRID]


# --- Model (§2, verbatim) --------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.fc = nn.Linear(w, w)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(F.relu(self.fc(self.norm(x))))


class DeepMLP(nn.Module):
    def __init__(self, d_in, depth, w=256, p=0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, w)          # first hidden layer
        self.blocks = nn.ModuleList([Block(w, p) for _ in range(depth - 1)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 1))

    def forward(self, x):
        h = F.relu(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h).squeeze(-1)


def n_params(d_in: int, depth: int, w: int = WIDTH) -> int:
    return d_in * w + w + (depth - 1) * (w * w + w + 2 * w) + 2 * w + w + 1


def lr_lambda(total_steps: int, warmup_frac: float = HP["warmup_frac"]):
    """Multiplier for LambdaLR: linear warm-up over the first warmup_frac of the steps the
    schedule allows (max_epochs x batches per epoch), then cosine decay to 0 at the end of
    that budget. Early stopping can end training before the decay finishes."""
    warm = max(1, math.ceil(warmup_frac * total_steps))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total_steps - warm))))
    return f
