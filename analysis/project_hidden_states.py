"""
project_hidden_states.py

2-d projections (UMAP + t-SNE) of OpenVLA hidden states across layers, for visual
inspection of trajectory structure in the LIBERO-10 mini-corpus.

Prior known-good behaviour on the full 98.5k corpus: structured trajectory manifolds at
mid-layers, with no clean visual failure zone. That absence is expected and is precisely
what motivates the linear probe. At ~10-26k timesteps the structure will be sparser --
that is a consequence of scale, not a discrepancy.

Design is locked (Step 5 of init_instructions.md):
  * features identical to the probe -- mean over the 7 action-token positions
  * one layer in memory at a time
  * PCA to 50 components before either method; never UMAP/t-SNE on raw 4096-d
  * three colourings per layer x method: outcome, task ID, normalised time-in-rollout
  * every embedding persisted, so replotting never recomputes

Usage:
    source env.sh
    python analysis/project_hidden_states.py                    # all layers, both methods
    python analysis/project_hidden_states.py --layers 15        # just layer 15
    python analysis/project_hidden_states.py --replot-only      # redraw PNGs from saved .npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

# --- Locked configuration -----------------------------------------------------------
LAYERS = [5, 15, 20, 25, 32]
CORPUS_DIR = Path("/data/corpus")
RESULTS_DIR = Path("results/projections")
SEED = 42

PCA_COMPONENTS = 50
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
TSNE_PERPLEXITY = 30

# Stored positions are [P-1, P, ..., P+6]; the 7 action-token positions are indices 1..7.
ACTION_POS_SLICE = slice(1, 8)


def load_layer(corpus_dir: Path, layer: int):
    """Load features for one layer only, one rollout at a time (keeps peak RAM low).

    Returns (X [N,4096] float32, index dict of per-timestep arrays).
    """
    manifests = sorted(corpus_dir.glob("task*_ep*.json"))
    if not manifests:
        raise FileNotFoundError(f"no rollout manifests in {corpus_dir}")

    X_parts, rollout_ids, task_ids, timesteps, norm_time, outcomes = [], [], [], [], [], []
    for mp in manifests:
        with open(mp) as f:
            man = json.load(f)
        rec = torch.load(corpus_dir / man["pt_file"], map_location="cpu")
        hs = rec["hidden_states"]
        feats = hs[:, layer, ACTION_POS_SLICE, :].float().mean(dim=1).numpy().astype(np.float32)
        T = feats.shape[0]

        X_parts.append(feats)
        rollout_ids.append(np.full(T, man["rollout_id"], dtype=object))
        task_ids.append(np.full(T, man["task_id"], dtype=np.int64))
        timesteps.append(np.arange(T, dtype=np.int64))
        # Normalised time within rollout -- the colouring that exposes trajectory structure.
        norm_time.append((np.arange(T, dtype=np.float32) / max(T - 1, 1)))
        outcomes.append(np.full(T, 0 if man["success"] else 1, dtype=np.int64))
        del rec, hs

    index = {
        "rollout_id": np.concatenate(rollout_ids).astype(str),
        "task_id": np.concatenate(task_ids),
        "timestep": np.concatenate(timesteps),
        "norm_time": np.concatenate(norm_time),
        "outcome": np.concatenate(outcomes),  # 1 = failure
    }
    return np.concatenate(X_parts, axis=0), index


def embed(X: np.ndarray, method: str):
    """PCA-50 then UMAP or t-SNE. Never run either on raw 4096-d."""
    n_comp = min(PCA_COMPONENTS, X.shape[0], X.shape[1])
    Xp = PCA(n_components=n_comp, random_state=SEED).fit_transform(X)

    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=SEED,
        )
        return reducer.fit_transform(Xp)

    if method == "tsne":
        # No subsampling needed at this corpus size.
        return TSNE(
            n_components=2,
            perplexity=min(TSNE_PERPLEXITY, max(5, (Xp.shape[0] - 1) // 3)),
            init="pca",
            random_state=SEED,
        ).fit_transform(Xp)

    raise ValueError(f"unknown method {method!r}")


def npz_path(results_dir: Path, layer: int, method: str) -> Path:
    return results_dir / f"proj_layer{layer:02d}_{method}_seed{SEED}.npz"


def plot_all(emb: np.ndarray, index: dict, results_dir: Path, layer: int, method: str):
    """Three PNGs: outcome, task ID, normalised time-within-rollout."""
    base = f"layer{layer:02d}_{method}_seed{SEED}"
    common = dict(s=2, linewidths=0, rasterized=True)

    # (a) rollout outcome
    fig, ax = plt.subplots(figsize=(7, 6))
    for val, lab, col in ((0, "success", "#2f7ab5"), (1, "failure", "#c8452f")):
        m = index["outcome"] == val
        ax.scatter(emb[m, 0], emb[m, 1], c=col, label=f"{lab} (n={int(m.sum())})", **common)
    ax.set_title(f"Layer {layer} | {method.upper()} | rollout outcome")
    ax.legend(markerscale=6, loc="best", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(results_dir / f"{base}_outcome.png", dpi=150); plt.close(fig)

    # (b) task ID
    fig, ax = plt.subplots(figsize=(7.6, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=index["task_id"], cmap="tab10", vmin=-0.5,
                    vmax=9.5, **common)
    ax.set_title(f"Layer {layer} | {method.upper()} | task ID")
    cb = fig.colorbar(sc, ax=ax, ticks=range(10)); cb.set_label("task ID")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(results_dir / f"{base}_task.png", dpi=150); plt.close(fig)

    # (c) normalised time within rollout -- the colouring that exposes trajectory structure
    fig, ax = plt.subplots(figsize=(7.6, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=index["norm_time"], cmap="viridis", **common)
    ax.set_title(f"Layer {layer} | {method.upper()} | normalised time in rollout")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("t / T")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(results_dir / f"{base}_time.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--methods", type=str, nargs="+", default=["umap", "tsne"])
    ap.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--replot-only", action="store_true",
                    help="redraw PNGs from saved .npz without recomputing embeddings")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for layer in args.layers:
        X = index = None
        for method in args.methods:
            out_npz = npz_path(results_dir, layer, method)
            t0 = time.time()

            if out_npz.exists() or args.replot_only:
                if not out_npz.exists():
                    print(f"[!] layer {layer} {method}: no saved embedding, skipping", flush=True)
                    continue
                d = np.load(out_npz, allow_pickle=True)
                emb = d["embedding"]
                index = {k: d[k] for k in ("rollout_id", "task_id", "timestep",
                                           "norm_time", "outcome")}
                print(f"[*] layer {layer} {method}: reusing {out_npz.name}", flush=True)
            else:
                if X is None:  # load the layer once, reuse across methods
                    print(f"[*] loading layer {layer} features", flush=True)
                    X, index = load_layer(Path(args.corpus_dir), layer)
                    print(f"[*] layer {layer} features {X.shape}", flush=True)
                print(f"[*] layer {layer} {method}: embedding...", flush=True)
                emb = embed(X, method)
                np.savez_compressed(out_npz, embedding=emb, layer=layer, method=method,
                                    seed=SEED, **index)
                print(f"[*] layer {layer} {method}: {time.time()-t0:.0f}s -> {out_npz.name}",
                      flush=True)

            plot_all(emb, index, results_dir, layer, method)
            print(f"[*] layer {layer} {method}: 3 PNGs written", flush=True)

        del X

    print(f"\n=== projections complete -> {results_dir} ===")


if __name__ == "__main__":
    main()
