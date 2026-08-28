"""
probe_layer.py

Single-layer linear probe for runtime failure detection on the LIBERO-10 mini-corpus.
Pilots one cell of the full layer x timestep AUROC heatmap (thesis Contribution 3).

The probe is deliberately linear: limited expressivity means a high AUROC is evidence
about the representation itself rather than about the classifier.

Design is locked (Step 4 of init_instructions.md):
  * features -- layer-15 hidden state, mean over the 7 action-token positions
  * labels   -- rollout success flag broadcast to every timestep (1 = failure)
  * split    -- GroupShuffleSplit by rollout_id; NEVER split within a rollout, because
                adjacent timesteps are near-duplicates and a within-rollout split
                produces a meaningless ~0.99 AUROC
  * model    -- StandardScaler (fit on train only) -> LogisticRegression
  * control  -- refit with the rollout-to-label mapping permuted; expect ~0.5

Usage:
    source env.sh
    python analysis/probe_layer.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# --- Locked configuration -----------------------------------------------------------
LAYER = 15                 # index into the hidden_states tuple (0 = embedding layer)
CORPUS_DIR = Path("/data/corpus")
RESULTS_DIR = Path("results/probe_pilot")
TEST_SIZE = 0.2
RANDOM_STATE = 42
C = 1.0
MAX_ITER = 2000

# Stored positions are [P-1, P, ..., P+6]; index 0 is the last prompt token, so the 7
# action-token positions are indices 1..7.
ACTION_POS_SLICE = slice(1, 8)
POSITION_SCHEME = "mean over the 7 action-token positions (stored indices 1..7 of 8)"

# Shuffle control tolerances, widened for small n per the spec.
CONTROL_FLAG_ABOVE = 0.55
CONTROL_ACCEPTABLE = (0.40, 0.60)


def corpus_manifest_hash(manifests: list[Path]) -> str:
    """sha256 over the concatenated per-rollout manifests, so a metrics.json can be
    traced to the exact corpus it was computed from."""
    h = hashlib.sha256()
    for p in sorted(manifests):
        h.update(p.read_bytes())
    return h.hexdigest()


def load_features(corpus_dir: Path, layer: int):
    """Extract one 4096-d feature vector per timestep, one rollout at a time.

    Returns (X float32 [N,4096], y int [N], groups [N], task_ids [N], rollout_meta list).
    """
    manifests = sorted(corpus_dir.glob("task*_ep*.json"))
    if not manifests:
        raise FileNotFoundError(f"no rollout manifests in {corpus_dir}")

    X_parts, y_parts, g_parts, t_parts, meta = [], [], [], [], []
    for mp in manifests:
        with open(mp) as f:
            man = json.load(f)
        pt_path = corpus_dir / man["pt_file"]

        rec = torch.load(pt_path, map_location="cpu")
        hs = rec["hidden_states"]  # (T, n_layers, 8, d_model) fp16
        if layer >= hs.shape[1]:
            raise IndexError(f"layer {layer} out of range for {hs.shape[1]} stored layers")

        # fp16 -> fp32 before averaging; mean over the 7 action-token positions.
        feats = hs[:, layer, ACTION_POS_SLICE, :].float().mean(dim=1).numpy()

        # 1 = failure, 0 = success. Broadcast the rollout outcome to every timestep.
        label = 0 if man["success"] else 1
        T = feats.shape[0]

        X_parts.append(feats.astype(np.float32))
        y_parts.append(np.full(T, label, dtype=np.int64))
        g_parts.append(np.full(T, man["rollout_id"], dtype=object))
        t_parts.append(np.full(T, man["task_id"], dtype=np.int64))
        meta.append(
            {
                "rollout_id": man["rollout_id"],
                "task_id": man["task_id"],
                "success": bool(man["success"]),
                "label": label,
                "n_timesteps": T,
            }
        )
        del rec, hs

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(g_parts, axis=0)
    task_ids = np.concatenate(t_parts, axis=0)
    return X, y, groups, task_ids, meta, manifests


def fit_and_score(X_tr, y_tr, X_te, y_te):
    """StandardScaler (train-only fit) -> LogisticRegression; returns test probabilities."""
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    ).fit(scaler.transform(X_tr), y_tr)
    return clf.predict_proba(scaler.transform(X_te))[:, 1]


def rollout_level_auroc(probs, y_te, groups_te):
    """Aggregate timestep probabilities per rollout (mean and max) and score.

    Only ~10 test rollouts at this corpus size, so these are descriptive, not conclusive.
    """
    out = {}
    uniq = np.unique(groups_te)
    r_true = np.array([y_te[groups_te == g][0] for g in uniq])
    if len(np.unique(r_true)) < 2:
        return {"mean": None, "max": None, "note": "only one class among test rollouts"}
    for agg, fn in (("mean", np.mean), ("max", np.max)):
        r_score = np.array([fn(probs[groups_te == g]) for g in uniq])
        out[agg] = float(roc_auc_score(r_true, r_score))
    out["n_test_rollouts"] = int(len(uniq))
    return out


def class_composition(meta, ids):
    """Rollout counts per class, plus per-task composition (the task-identity confound
    is inspectable rather than corrected -- see interpretation guardrails)."""
    sel = [m for m in meta if m["rollout_id"] in set(ids)]
    per_task: dict[str, dict[str, int]] = {}
    for m in sel:
        d = per_task.setdefault(str(m["task_id"]), {"failure": 0, "success": 0})
        d["failure" if m["label"] == 1 else "success"] += 1
    return {
        "n_rollouts": len(sel),
        "n_failure_rollouts": sum(1 for m in sel if m["label"] == 1),
        "n_success_rollouts": sum(1 for m in sel if m["label"] == 0),
        "per_task": dict(sorted(per_task.items(), key=lambda kv: int(kv[0]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args()

    t0 = time.time()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading layer {args.layer} features from {args.corpus_dir}", flush=True)
    X, y, groups, task_ids, meta, manifests = load_features(Path(args.corpus_dir), args.layer)
    print(f"[*] features {X.shape} | {len(meta)} rollouts | "
          f"failure timesteps {int(y.sum())}/{len(y)}", flush=True)

    # --- Split by rollout. Within-rollout splits leak: adjacent timesteps are near-duplicates.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    g_tr, g_te = groups[tr], groups[te]
    print(f"[*] split: train {len(tr)} ts / {len(np.unique(g_tr))} rollouts | "
          f"test {len(te)} ts / {len(np.unique(g_te))} rollouts", flush=True)

    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        raise RuntimeError(
            f"a split is single-class (train classes {np.unique(y[tr])}, "
            f"test {np.unique(y[te])}); AUROC undefined. Corpus too imbalanced."
        )

    # --- Real probe ---------------------------------------------------------------
    probs = fit_and_score(X[tr], y[tr], X[te], y[te])
    ts_auroc = float(roc_auc_score(y[te], probs))
    ro_auroc = rollout_level_auroc(probs, y[te], g_te)
    print(f"[*] timestep AUROC: {ts_auroc:.4f}", flush=True)
    print(f"[*] rollout AUROC : {ro_auroc}", flush=True)

    # --- Mandatory shuffle control -------------------------------------------------
    # Permute the rollout->label mapping, then broadcast; same split. Expect ~0.5.
    rng = np.random.RandomState(RANDOM_STATE)
    rid_to_label = {m["rollout_id"]: m["label"] for m in meta}
    rids = sorted(rid_to_label)
    permuted = dict(zip(rids, rng.permutation([rid_to_label[r] for r in rids])))
    y_perm = np.array([permuted[g] for g in groups], dtype=np.int64)

    if len(np.unique(y_perm[tr])) < 2 or len(np.unique(y_perm[te])) < 2:
        ctrl_auroc, ctrl_note = None, "permuted split single-class; control undefined"
    else:
        ctrl_probs = fit_and_score(X[tr], y_perm[tr], X[te], y_perm[te])
        ctrl_auroc = float(roc_auc_score(y_perm[te], ctrl_probs))
        ctrl_note = None
    print(f"[*] shuffle-control AUROC: {ctrl_auroc}", flush=True)

    leakage_flag = ctrl_auroc is not None and ctrl_auroc > CONTROL_FLAG_ABOVE
    ctrl_in_range = ctrl_auroc is not None and (
        CONTROL_ACCEPTABLE[0] <= ctrl_auroc <= CONTROL_ACCEPTABLE[1]
    )
    if leakage_flag:
        print(f"\n!!! WARNING: shuffle-control AUROC {ctrl_auroc:.4f} > {CONTROL_FLAG_ABOVE} "
              f"-- indicates LEAKAGE; do not trust the probe result !!!\n", flush=True)

    # --- Outputs -------------------------------------------------------------------
    metrics = {
        "layer": args.layer,
        "position_scheme": POSITION_SCHEME,
        "seed": RANDOM_STATE,
        "corpus_dir": str(args.corpus_dir),
        "corpus_manifest_hash": corpus_manifest_hash(manifests),
        "n_rollouts": len(meta),
        "timestep_auroc": ts_auroc,
        "rollout_auroc": ro_auroc,
        "shuffle_control_auroc": ctrl_auroc,
        "shuffle_control_note": ctrl_note,
        "shuffle_control_in_acceptable_range": ctrl_in_range,
        "shuffle_control_acceptable_range": list(CONTROL_ACCEPTABLE),
        "leakage_flag": bool(leakage_flag),
        "split": {
            "method": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) by rollout_id",
            "n_train_timesteps": int(len(tr)),
            "n_test_timesteps": int(len(te)),
            "train_failure_timesteps": int(y[tr].sum()),
            "test_failure_timesteps": int(y[te].sum()),
            "train": class_composition(meta, np.unique(g_tr)),
            "test": class_composition(meta, np.unique(g_te)),
        },
        "model": {
            "scaler": "StandardScaler (fit on train only)",
            "classifier": f"LogisticRegression(C={C}, class_weight='balanced', "
                          f"solver='lbfgs', max_iter={MAX_ITER}, random_state={RANDOM_STATE})",
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        results_dir / "scores_test.npz",
        probs=probs,
        labels=y[te],
        rollout_ids=g_te.astype(str),
        task_ids=task_ids[te],
    )

    # Optional figure: per-rollout mean score by class.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        uniq = np.unique(g_te)
        r_mean = np.array([probs[g_te == g].mean() for g in uniq])
        r_true = np.array([y[te][g_te == g][0] for g in uniq])
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 1, 12)
        ax.hist(r_mean[r_true == 0], bins=bins, alpha=0.65, label="success rollouts")
        ax.hist(r_mean[r_true == 1], bins=bins, alpha=0.65, label="failure rollouts")
        ax.set_xlabel("per-rollout mean P(failure)")
        ax.set_ylabel("rollouts")
        ax.set_title(f"Layer {args.layer} probe, test rollouts (timestep AUROC {ts_auroc:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "rollout_score_hist.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # a figure is optional; never fail the run over it
        print(f"[!] figure skipped: {e}", flush=True)

    print(f"\n=== layer {args.layer} | timestep AUROC {ts_auroc:.4f} | "
          f"control {ctrl_auroc} | {metrics['runtime_seconds']}s ===")
    print(f"=== wrote {results_dir}/metrics.json, scores_test.npz ===")


if __name__ == "__main__":
    main()
