"""
build_cache2.py

The Step 6 feature cache for the combined v2 + v3 corpus (more_data_collection.md,
handoff v1, Step 6), modelled on analysis/abc/build_cache.py. One pass over the rollouts
of corpus_v3_index.json, in index order, writing

    /data/tmp/abc2_features/layer{L:02d}.npy   (N, 520, 4096) fp16, L = 0..32
    /data/tmp/abc2_features/action_tokens.npy  (N, 520, 7) int32
    /data/tmp/abc2_features/rollout_ids.json   rollout order (= the index's)
    /data/tmp/abc2_features/done.npy           (N,) bool, per-rollout completion

Rows hold the pooled hidden state (mean over the 7 decision states). v2 rows are made
exactly as the old cache made them (build_cache.read_rollout -> pooled_and_hN -> fp16,
lossless); v3 rows are hidden.pt's (520, 33, 4096) fp16 copied unchanged. Index 32 is
post-norm.

Disk. About 0.141 GB per rollout (141 GB for 1000). The memmaps are sparse until
written, so before creating or extending anything the script requires free space on the
target >= bytes still to write + HEADROOM_GB (30, configs/more_data.yaml), prints both
numbers and refuses otherwise; it also stops (resumably) if free space falls below the
headroom mid-build. Resumable: rollouts marked in done.npy are skipped, and done.npy is
only updated after that rollout's rows are flushed. I/O bound, single process.

--verify: rollout_ids.json equals the index order; every rollout is done; layer 15 rows
of all v2 rollouts equal the old cache /data/tmp/abc_features/layer15.npy exactly (rows
looked up by rollout id through its rollout_ids.json), or, if the old cache is gone,
equal pooled_and_hN output for 5 v2 rollouts including a P = 7 one; all 33 layers of up
to 3 v3 rollouts (first, middle, last) equal their hidden.pt exactly; action tokens of
every rollout equal actions.npz. Exits 1 on failure.

Usage (from the repo root):
    source env.sh
    python analysis/abc2/build_cache2.py                 # full build (run in tmux)
    python analysis/abc2/build_cache2.py --verify
    python analysis/abc2/build_cache2.py --index DEV.json --out-dir /data/tmp/abc2_dev/x/cache
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data2 import CACHE_DIR, INDEX_V3, OLD_CACHE_DIR, Index2, load_index2  # noqa: E402
from build_cache import open_or_create, read_rollout  # noqa: E402  (analysis/abc, on sys.path via data2)
from data import D_MODEL, N_STORED, T, cache_layer_path  # noqa: E402

BYTES_PER_ROLLOUT = N_STORED * T * D_MODEL * 2 + T * 7 * 4
HEADROOM_GB = 30
VERIFY_LAYER = 15
N_THREADS = 2


def existing_parent(p: Path) -> Path:
    p = p.resolve()
    while not p.exists():
        p = p.parent
    return p


def read_v3(d: Path) -> tuple[np.ndarray, np.ndarray]:
    """hidden.pt (520, 33, 4096) fp16 as numpy, and action_token_ids (520, 7) int32."""
    meta = json.loads((d / "meta.json").read_text())
    if not meta.get("complete") or d.name.startswith("."):
        raise SystemExit(f"{d.name}: not a finished rollout")
    x = torch.load(d / "hidden.pt", map_location="cpu")
    if tuple(x.shape) != (T, N_STORED, D_MODEL) or x.dtype != torch.float16:
        raise SystemExit(f"{d.name}: hidden.pt is {tuple(x.shape)} {x.dtype}")
    tok = np.load(d / "actions.npz")["action_token_ids"]
    if tok.shape != (T, 7):
        raise SystemExit(f"{d.name}: action_token_ids is {tok.shape}")
    return x.numpy(), tok.astype(np.int32)


def read_row(idx: Index2, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(520, 33, 4096) fp16 and (520, 7) int32 for row i of the index."""
    if idx.source[i] == "v3":
        return read_v3(idx.rollout_dir(i))
    pooled, tok = read_rollout(str(idx.rids[i]), int(idx.n_positions[i]))
    half = pooled.half().numpy()
    if not np.array_equal(half.astype(np.float32), pooled.numpy()):
        raise SystemExit(f"{idx.rids[i]}: pooled state is not fp16-exact")
    return half, tok


def build(index: Path, out_dir: Path, limit: int | None, headroom_gb: float) -> None:
    idx = load_index2(index)
    n = idx.n
    order_path = out_dir / "rollout_ids.json"
    done_path = out_dir / "done.npy"
    if order_path.exists() and json.loads(order_path.read_text()) != idx.rids.tolist():
        raise SystemExit(f"{order_path} disagrees with {index} order")
    done = np.load(done_path) if done_path.exists() else np.zeros(n, dtype=bool)
    if done.shape != (n,):
        raise SystemExit(f"{done_path} has shape {done.shape}, index has {n} rollouts")

    todo = [i for i in range(n) if not done[i]]
    if limit is not None:
        todo = todo[:limit]
    need = BYTES_PER_ROLLOUT * len(todo)
    free = shutil.disk_usage(existing_parent(out_dir)).free
    print(f"[*] free on {existing_parent(out_dir)}: {free / 1e9:.1f} GB; to write "
          f"{len(todo)} rollouts x {BYTES_PER_ROLLOUT / 1e9:.3f} GB = {need / 1e9:.1f} GB "
          f"+ headroom {headroom_gb:.0f} GB = {need / 1e9 + headroom_gb:.1f} GB", flush=True)
    if free < need + headroom_gb * 1e9:
        raise SystemExit("not enough free space: refusing to start")

    out_dir.mkdir(parents=True, exist_ok=True)
    if not order_path.exists():
        order_path.write_text(json.dumps(idx.rids.tolist()))
    layers = [open_or_create(cache_layer_path(L, out_dir), (n, T, D_MODEL), np.float16)
              for L in range(N_STORED)]
    tokens = open_or_create(out_dir / "action_tokens.npy", (n, T, 7), np.int32)
    print(f"[*] {int(done.sum())}/{n} already cached; {len(todo)} to do -> {out_dir}", flush=True)

    t0 = time.time()
    for k, i in enumerate(todo, 1):
        half, tok = read_row(idx, i)
        for L in range(N_STORED):
            layers[L][i] = half[:, L, :]
            layers[L].flush()
        tokens[i] = tok
        tokens.flush()
        done[i] = True
        np.save(done_path, done)

        if k % 25 == 0 or k == len(todo) or k <= 2:
            el = time.time() - t0
            print(f"[{k:4d}/{len(todo)}] {el / 60:5.1f} min  {el / k:.2f} s/rollout  "
                  f"eta {(len(todo) - k) * el / k / 60:5.1f} min", flush=True)
        if k % 25 == 0 and shutil.disk_usage(out_dir).free < headroom_gb * 1e9:
            raise SystemExit(f"free space fell below {headroom_gb} GB: stopping (resumable)")

    print(f"[*] {int(done.sum())}/{n} cached")


def verify(index: Path, out_dir: Path, old_cache: Path) -> None:
    idx = load_index2(index)
    ok = True

    order = json.loads((out_dir / "rollout_ids.json").read_text())
    same_order = order == idx.rids.tolist()
    print(f"[verify] rollout_ids.json equals {index} order: {same_order}")
    ok &= same_order
    done = np.load(out_dir / "done.npy")
    print(f"[verify] {int(done.sum())}/{idx.n} rollouts cached")
    ok &= bool(done.all())

    layer = np.load(cache_layer_path(VERIFY_LAYER, out_dir), mmap_mode="r")
    v2 = np.nonzero(done & (idx.source == "v2"))[0]
    old_layer = cache_layer_path(VERIFY_LAYER, old_cache)
    old_ids = old_cache / "rollout_ids.json"
    if old_layer.exists() and old_ids.exists():
        old_pos = {r: j for j, r in enumerate(json.loads(old_ids.read_text()))}
        old_done = np.load(old_cache / "done.npy") if (old_cache / "done.npy").exists() else None
        old = np.load(old_layer, mmap_mode="r")
        bad = [str(idx.rids[i]) for i in v2
               if str(idx.rids[i]) not in old_pos
               or (old_done is not None and not old_done[old_pos[str(idx.rids[i])]])
               or not np.array_equal(np.asarray(layer[i]), np.asarray(old[old_pos[str(idx.rids[i])]]))]
        print(f"[verify] layer {VERIFY_LAYER}: {len(v2) - len(bad)}/{len(v2)} v2 rows equal the old "
              f"cache {old_layer} exactly" + (f"; mismatches {bad[:5]}" if bad else ""))
        ok &= not bad and len(v2) > 0
    else:
        p7 = [i for i in v2 if idx.n_positions[i] == 7][:1]
        p3 = [i for i in v2 if idx.n_positions[i] == 3]
        check = sorted(set(p7 + p3[:max(0, 5 - len(p7))]))
        for i in check:
            pooled, _ = read_rollout(str(idx.rids[i]), int(idx.n_positions[i]))
            same = np.array_equal(np.asarray(layer[i]).astype(np.float32),
                                  pooled[:, VERIFY_LAYER, :].numpy())
            print(f"[verify] {idx.rids[i]} (P={idx.n_positions[i]}): layer {VERIFY_LAYER} equals "
                  f"pooled_and_hN: {same}")
            ok &= same
        ok &= bool(p7) and len(check) >= min(5, len(v2))

    v3 = np.nonzero(done & (idx.source == "v3"))[0]
    for i in sorted({int(v3[0]), int(v3[len(v3) // 2]), int(v3[-1])}) if len(v3) else []:
        half, _ = read_v3(idx.rollout_dir(i))
        same = all(np.array_equal(np.load(cache_layer_path(L, out_dir), mmap_mode="r")[i],
                                  half[:, L, :]) for L in range(N_STORED))
        print(f"[verify] {idx.rids[i]} (v3): all {N_STORED} layers equal hidden.pt: {same}")
        ok &= same

    tokens = np.load(out_dir / "action_tokens.npy", mmap_mode="r")
    bad_tok = [str(idx.rids[i]) for i in np.nonzero(done)[0]
               if not np.array_equal(np.asarray(tokens[i]),
                                     np.load(idx.rollout_dir(i) / "actions.npz")["action_token_ids"])]
    print(f"[verify] action tokens equal actions.npz: {int(done.sum()) - len(bad_tok)}/{int(done.sum())}"
          + (f"; mismatches {bad_tok[:5]}" if bad_tok else ""))
    ok &= not bad_tok

    print(f"[verify] {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX_V3)
    ap.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--old-cache", type=Path, default=OLD_CACHE_DIR, help="for --verify")
    ap.add_argument("--headroom-gb", type=float, default=HEADROOM_GB)
    ap.add_argument("--limit", type=int, default=None, help="cache at most N more rollouts")
    ap.add_argument("--verify", action="store_true", help="check the built cache and exit")
    args = ap.parse_args()
    torch.set_num_threads(N_THREADS)
    if args.verify:
        verify(args.index, args.out_dir, args.old_cache)
    else:
        build(args.index, args.out_dir, args.limit, args.headroom_gb)


if __name__ == "__main__":
    main()
