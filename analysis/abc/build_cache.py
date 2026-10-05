"""
build_cache.py

The A/B/C feature cache (handoff section 3): one pass over /data/rollouts_v2, writing

    /data/tmp/abc_features/layer{L:02d}.npy   (300, 520, 4096) fp16, L = 0..32
    /data/tmp/abc_features/action_tokens.npy  (300, 520, 7) int32
    /data/tmp/abc_features/rollout_ids.json   rollout order (= corpus_v2_index.json)
    /data/tmp/abc_features/done.npy           (300,) bool, per-rollout completion

Each layer file holds the pooled hidden state (mean over the 7 decision states) exactly
as compute_constraints.pooled_and_hN produces it, cast back to fp16. That cast is
lossless: P = 3 rollouts store the pool in fp16 already, and pooled_and_hN puts the
P = 7 mean through the same fp16 round trip. Index 32 is post-norm.

I/O-bound like compute_constraints.py (~0.31 GB/s off the corpus disk), so it runs
single-threaded; expect ~10 minutes for 147 GB. It resumes: rollouts marked in done.npy
are skipped, and done.npy is only updated after that rollout's rows are flushed.

Usage (from the repo root):
    source env.sh
    python analysis/abc/build_cache.py                    # full build (run in tmux)
    python analysis/abc/build_cache.py --verify           # Stage 1 check on the built cache
    python analysis/abc/build_cache.py --limit 2 --out-dir /data/tmp/abc_features_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))   # analysis/, for frozen modules
from compute_constraints import pooled_and_hN               # noqa: E402

from data import (CACHE_DIR, CORPUS_ROOT, D_MODEL, N_STORED, T,  # noqa: E402
                  cache_layer_path, load_index)

BYTES_NEEDED = (N_STORED * T * D_MODEL * 2 + T * 7 * 4) * 300
VERIFY_LAYER = 15


def open_or_create(path: Path, shape, dtype) -> np.ndarray:
    if path.exists():
        mm = np.load(path, mmap_mode="r+")
        if mm.shape != shape or mm.dtype != np.dtype(dtype):
            raise SystemExit(f"{path} is {mm.shape} {mm.dtype}, expected {shape} {dtype}")
        return mm
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def read_rollout(rid: str, n_positions: int) -> tuple[torch.Tensor, np.ndarray]:
    """Pooled (520, 33, 4096) fp32 and action_token_ids (520, 7) int32 for one rollout."""
    d = CORPUS_ROOT / rid
    x = torch.load(d / "hidden.pt", map_location="cpu")
    if tuple(x.shape) != (T, N_STORED, n_positions, D_MODEL):
        raise SystemExit(f"{rid}: hidden.pt is {tuple(x.shape)}")
    pooled, _ = pooled_and_hN(x, n_positions)
    tok = np.load(d / "actions.npz")["action_token_ids"]
    if tok.shape != (T, 7):
        raise SystemExit(f"{rid}: action_token_ids is {tok.shape}")
    return pooled, tok.astype(np.int32)


def build(out_dir: Path, limit: int | None) -> None:
    idx = load_index()
    n = idx.n
    out_dir.mkdir(parents=True, exist_ok=True)

    order_path = out_dir / "rollout_ids.json"
    if order_path.exists():
        if json.loads(order_path.read_text()) != idx.rids.tolist():
            raise SystemExit(f"{order_path} disagrees with corpus_v2_index.json order")
    else:
        order_path.write_text(json.dumps(idx.rids.tolist()))

    done_path = out_dir / "done.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
    if not done.any():
        free = shutil.disk_usage(out_dir).free
        print(f"[*] free on {out_dir}: {free / 1e9:.0f} GB, need {BYTES_NEEDED / 1e9:.0f} GB")
        if free < BYTES_NEEDED * 1.05:
            raise SystemExit("not enough free space")

    layers = [open_or_create(cache_layer_path(L, out_dir), (n, T, D_MODEL), np.float16)
              for L in range(N_STORED)]
    tokens = open_or_create(out_dir / "action_tokens.npy", (n, T, 7), np.int32)

    todo = [i for i in range(n) if not done[i]]
    if limit is not None:
        todo = todo[:limit]
    print(f"[*] {int(done.sum())}/{n} already cached; {len(todo)} to do -> {out_dir}")

    t0 = time.time()
    for k, i in enumerate(todo, 1):
        pooled, tok = read_rollout(idx.rids[i], int(idx.n_positions[i]))
        half = pooled.half().numpy()                      # (520, 33, 4096)
        if not np.array_equal(half.astype(np.float32), pooled.numpy()):
            raise SystemExit(f"{idx.rids[i]}: pooled state is not fp16-exact")
        for L in range(N_STORED):
            layers[L][i] = half[:, L, :]
            layers[L].flush()
        tokens[i] = tok
        tokens.flush()
        done[i] = True
        np.save(done_path, done)

        if k % 25 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"[{k:3d}/{len(todo)}] {el / 60:5.1f} min  {el / k:.2f} s/rollout  "
                  f"eta {(len(todo) - k) * el / k / 60:5.1f} min", flush=True)

    print(f"[*] {int(done.sum())}/{n} cached")


def verify(out_dir: Path) -> None:
    """Cached layer 15 equals pooled_and_hN exactly, for one P = 3 and one P = 7 rollout,
    and the cached action tokens equal actions.npz."""
    idx = load_index()
    done = np.load(out_dir / "done.npy")
    print(f"[verify] {int(done.sum())}/{idx.n} rollouts cached")
    layer = np.load(cache_layer_path(VERIFY_LAYER, out_dir), mmap_mode="r")
    tokens = np.load(out_dir / "action_tokens.npy", mmap_mode="r")
    ok = True
    for P in (3, 7):
        cands = np.where(done & (idx.n_positions == P))[0]
        if cands.size == 0:
            print(f"[verify] no cached P={P} rollout to check")
            ok = False
            continue
        i = int(cands[0])
        pooled, tok = read_rollout(idx.rids[i], P)
        cached = np.asarray(layer[i]).astype(np.float32)
        same = np.array_equal(cached, pooled[:, VERIFY_LAYER, :].numpy())
        same_tok = np.array_equal(np.asarray(tokens[i]), tok)
        print(f"[verify] {idx.rids[i]} (P={P}): layer {VERIFY_LAYER} exact match {same}; "
              f"action tokens match {same_tok}")
        ok &= same and same_tok
    if not done.all():
        print("[verify] cache incomplete")
        ok = False
    print(f"[verify] {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--limit", type=int, default=None, help="cache at most N more rollouts")
    ap.add_argument("--verify", action="store_true", help="check the built cache and exit")
    args = ap.parse_args()
    if args.verify:
        verify(args.out_dir)
    else:
        build(args.out_dir, args.limit)


if __name__ == "__main__":
    main()
