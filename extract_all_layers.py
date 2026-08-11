"""
extract_all_layers.py

One pass over the corpus that writes the probe feature cache for EVERY stored layer,
so the per-layer scheme-B sweep does not re-read 42.5 GB once per layer.

Why this exists. load_or_cache() in control_diagnostic.py falls back to
load_features(corpus_dir, layer), which opens all 50 .pt files and keeps one layer.
Running that 33 times means 33 full passes over the corpus at ~0.24 GB/s -- roughly
100 minutes of pure disk, and if the layers are run in parallel they thrash the same
device. Each .pt holds all 33 layers already, so one pass can fill every cache.

BYTE-IDENTITY IS THE POINT. The per-layer features must be exactly what
load_features() would have produced, or the sweep is not comparable to the locked
layer-15 run. This script therefore reuses probe_layer's own slice expression per
layer rather than a single fused reduction over the layer axis: mean(dim=1) on a
(T,7,4096) slice and mean(dim=2) on a (T,33,7,4096) block reduce the same numbers but
not necessarily in the same order, and float32 addition is not associative. Slicing
per layer costs ~0.05 s per layer per file (~80 s over the corpus) and removes the
question entirely.

The existing layer-15 cache is treated as the reference: if it is present, this script
verifies its own layer-15 output against it bitwise and refuses to continue on
mismatch. It never overwrites that file -- results/probe_late_window/metrics.json
records its sha256 as provenance.

Usage:
    source env.sh
    python extract_all_layers.py                 # all stored layers, skip existing
    python extract_all_layers.py --layers 0-32
    python extract_all_layers.py --overwrite     # includes the layer-15 reference
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from control_diagnostic import CACHE_PATH
from probe_layer import ACTION_POS_SLICE, CORPUS_DIR


def parse_layers(spec: str | None, n_layers: int) -> list[int]:
    """"0-32" / "5,15,25" / None -> explicit layer list, bounds-checked."""
    if spec is None:
        return list(range(n_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    bad = [l for l in out if not 0 <= l < n_layers]
    if bad:
        raise SystemExit(f"layers {bad} out of range for {n_layers} stored layers")
    return sorted(set(out))


def stored_layer_count(corpus_dir: Path) -> tuple[int, Path]:
    """Read the layer axis off the first rollout without keeping the tensor."""
    manifests = sorted(corpus_dir.glob("task*_ep*.json"))
    if not manifests:
        raise FileNotFoundError(f"no rollout manifests in {corpus_dir}")
    with open(manifests[0]) as f:
        first = json.load(f)
    rec = torch.load(corpus_dir / first["pt_file"], map_location="cpu")
    n = int(rec["hidden_states"].shape[1])
    del rec
    return n, manifests[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    ap.add_argument("--layers", type=str, default=None,
                    help='e.g. "0-32" or "5,15,25"; default every stored layer')
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite caches that already exist (including layer 15)")
    args = ap.parse_args()

    t0 = time.time()
    corpus_dir = Path(args.corpus_dir)
    n_layers, probe_file = stored_layer_count(corpus_dir)
    layers = parse_layers(args.layers, n_layers)
    print(f"[*] {n_layers} stored layers (from {probe_file.name}); "
          f"extracting {len(layers)}: {layers[0]}..{layers[-1]}", flush=True)

    # A cache that already exists is left alone unless --overwrite. Layer 15 is the
    # reference for the identity check, so it is verified rather than rewritten.
    reference_layer = None
    ref_path = Path(str(CACHE_PATH).format(layer=15))
    if 15 in layers and ref_path.exists() and not args.overwrite:
        reference_layer = 15
        print("[*] layer 15 cache exists -> verifying against it, not rewriting",
              flush=True)

    todo = [l for l in layers
            if args.overwrite
            or not Path(str(CACHE_PATH).format(layer=l)).exists()
            or l == reference_layer]
    skipped = sorted(set(layers) - set(todo))
    if skipped:
        print(f"[*] skipping {len(skipped)} layers with caches already present: "
              f"{skipped}", flush=True)
    if not todo:
        print("[*] nothing to do")
        return

    manifests = sorted(corpus_dir.glob("task*_ep*.json"))
    parts: dict[int, list[np.ndarray]] = {l: [] for l in todo}
    y_parts, g_parts, t_parts, meta = [], [], [], []

    est_gb = sum(p.stat().st_size for p in corpus_dir.glob("task*_ep*.pt")) / 1e9
    print(f"[*] one pass over {len(manifests)} rollouts / {est_gb:.1f} GB", flush=True)

    for i, mp in enumerate(manifests, 1):
        with open(mp) as f:
            man = json.load(f)
        t_file = time.time()
        rec = torch.load(corpus_dir / man["pt_file"], map_location="cpu")
        hs = rec["hidden_states"]  # (T, n_layers, 8, d_model) fp16
        if hs.shape[1] != n_layers:
            raise RuntimeError(
                f"{man['rollout_id']}: {hs.shape[1]} layers, expected {n_layers}")

        for l in todo:
            # Exactly probe_layer.load_features' expression, per layer.
            feats = hs[:, l, ACTION_POS_SLICE, :].float().mean(dim=1).numpy()
            parts[l].append(feats.astype(np.float32))

        label = 0 if man["success"] else 1
        T = int(hs.shape[0])
        y_parts.append(np.full(T, label, dtype=np.int64))
        g_parts.append(np.full(T, man["rollout_id"], dtype=object))
        t_parts.append(np.full(T, man["task_id"], dtype=np.int64))
        meta.append({
            "rollout_id": man["rollout_id"],
            "task_id": man["task_id"],
            "success": bool(man["success"]),
            "label": label,
            "n_timesteps": T,
        })
        del rec, hs
        print(f"    [{i:>2}/{len(manifests)}] {man['rollout_id']} T={T} "
              f"({time.time() - t_file:.1f}s)", flush=True)

    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)
    task_ids = np.concatenate(t_parts)
    meta_arr = np.array(meta, dtype=object)
    print(f"[*] pass done in {time.time() - t0:.0f}s | {len(y)} timesteps | "
          f"failure {int(y.sum())}/{len(y)}", flush=True)

    # --- Identity check against the locked layer-15 cache ------------------------------
    if reference_layer is not None:
        z = np.load(ref_path, allow_pickle=True)
        X_new = np.concatenate(parts[reference_layer], axis=0)
        same = (X_new.shape == z["X"].shape
                and np.array_equal(X_new, z["X"])
                and np.array_equal(y, z["y"])
                and np.array_equal(groups.astype(str), z["groups"].astype(str)))
        if not same:
            raise SystemExit(
                "layer 15 features do not reproduce the existing cache bitwise -- the "
                "sweep would not be comparable to the locked run. Aborting without "
                "writing anything.")
        print(f"[*] layer 15 reproduces {ref_path} bitwise; leaving that file untouched",
              flush=True)
        del parts[reference_layer], X_new, z

    # --- Write one cache per layer, in load_or_cache's exact format --------------------
    for l in sorted(parts):
        cache = Path(str(CACHE_PATH).format(layer=l))
        cache.parent.mkdir(parents=True, exist_ok=True)
        X = np.concatenate(parts[l], axis=0)
        np.savez(cache, X=X, y=y, groups=groups, task_ids=task_ids, meta=meta_arr)
        print(f"[*] layer {l:>2} -> {cache} ({cache.stat().st_size / 1e6:.0f} MB)",
              flush=True)
        parts[l] = []
        del X

    print(f"\n=== {len(parts)} caches written, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
