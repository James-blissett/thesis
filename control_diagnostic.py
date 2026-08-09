"""
control_diagnostic.py

Variance diagnostic for the layer-15 probe's shuffle control. NOT a redesign of the
probe: it imports probe_layer's locked feature extraction, split, and model unchanged,
and only re-runs the already-mandated control across many permutation draws.

Motivation. The single-draw control in the locked design landed at 0.395, marginally
below the [0.40, 0.60] acceptance band. That band was written for a control whose
sampling noise is small, but at this corpus size the test fold holds 10 rollouts of
which 2 are failures, so the control is effectively a draw from a very coarse
distribution. One draw cannot distinguish 0.395 from 0.50, which leaves the gate unable
to either certify or reject the primary 0.670 result. This script estimates that
distribution so the gate becomes decidable.

Two independent questions, reported separately:

  (A) Control distribution -- fixed split (random_state=42, exactly the locked one),
      N permutations of the rollout->label mapping. Answers: is 0.395 an ordinary draw
      from the null, and does the null centre on 0.5?

  (B) Split sensitivity -- true labels, N different GroupShuffleSplit seeds. Answers:
      how much of the primary 0.670 is a property of the representation versus of which
      10 rollouts happened to land in the test fold. Diagnostic only; the locked result
      remains the seed-42 number.

  (C) Matched null -- split resampled AND labels permuted. The null distribution for
      (B), so the two can be compared with only the labels differing. Off by default;
      enable with --n-matched.

This script only produces distributions. The significance statistics are computed from
them by analyse_control.py -- see the warning on `p_null_ge_true_split_mean_INVALID`.

Usage:
    source env.sh
    python control_diagnostic.py                 # 20 permutations, 20 splits
    python control_diagnostic.py --n-perm 200 --n-splits 20 --n-matched 200
    python analyse_control.py                    # valid tests over the above
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from probe_layer import (
    CONTROL_ACCEPTABLE,
    LAYER,
    RANDOM_STATE,
    TEST_SIZE,
    fit_and_score,
    load_features,
)

CORPUS_DIR = Path("/data/corpus")
RESULTS_DIR = Path("results/probe_pilot")
# Feature cache lives on /data: 19289 x 4096 float32 is ~316 MB, too big for the root
# volume's headroom and pointless to recompute at ~150 s a time.
CACHE_PATH = Path("/data/tmp/probe_feats_layer{layer}.npz")


def load_or_cache(corpus_dir: Path, layer: int, use_cache: bool):
    """load_features(), memoised to /data so repeat diagnostics are cheap."""
    cache = Path(str(CACHE_PATH).format(layer=layer))
    if use_cache and cache.exists():
        print(f"[*] loading cached features from {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["groups"], z["task_ids"], list(z["meta"])

    print(f"[*] extracting layer {layer} features from {corpus_dir} (slow path)", flush=True)
    X, y, groups, task_ids, meta, _ = load_features(corpus_dir, layer)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, y=y, groups=groups, task_ids=task_ids,
                 meta=np.array(meta, dtype=object))
        print(f"[*] cached features to {cache}", flush=True)
    return X, y, groups, task_ids, meta


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else None,
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--n-matched", type=int, default=0,
                    help="matched null: resample split AND permute labels, N draws")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, task_ids, meta = load_or_cache(
        Path(args.corpus_dir), args.layer, use_cache=not args.no_cache
    )
    print(f"[*] features {X.shape} | {len(meta)} rollouts | "
          f"failure timesteps {int(y.sum())}/{len(y)}", flush=True)

    rid_to_label = {m["rollout_id"]: m["label"] for m in meta}
    rids = sorted(rid_to_label)
    base_labels = np.array([rid_to_label[r] for r in rids])

    # --- (A) Control distribution on the locked split --------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    g_te = groups[te]
    n_test_fail = len({g for g in g_te if rid_to_label[g] == 1})
    print(f"[*] locked split: {len(np.unique(g_te))} test rollouts, "
          f"{n_test_fail} of them failures", flush=True)

    ctrl_aurocs, ctrl_skipped = [], 0
    for i in range(args.n_perm):
        rng = np.random.RandomState(1000 + i)
        permuted = dict(zip(rids, rng.permutation(base_labels)))
        y_perm = np.array([permuted[g] for g in groups], dtype=np.int64)
        # A permutation that leaves either fold single-class has no defined AUROC; it is
        # a property of the draw, not a failure, so count it rather than aborting.
        if len(np.unique(y_perm[tr])) < 2 or len(np.unique(y_perm[te])) < 2:
            ctrl_skipped += 1
            continue
        p = fit_and_score(X[tr], y_perm[tr], X[te], y_perm[te])
        a = float(roc_auc_score(y_perm[te], p))
        ctrl_aurocs.append(a)
        print(f"    control perm {i:3d} (seed {1000+i}): AUROC {a:.4f}", flush=True)

    # --- (B) Split sensitivity with true labels ---------------------------------------
    split_aurocs, split_rows, split_skipped = [], [], 0
    for i in range(args.n_splits):
        g = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=2000 + i)
        s_tr, s_te = next(g.split(X, y, groups=groups))
        if len(np.unique(y[s_tr])) < 2 or len(np.unique(y[s_te])) < 2:
            split_skipped += 1
            continue
        p = fit_and_score(X[s_tr], y[s_tr], X[s_te], y[s_te])
        a = float(roc_auc_score(y[s_te], p))
        te_rids = np.unique(groups[s_te])
        nf = int(sum(rid_to_label[r] == 1 for r in te_rids))
        split_aurocs.append(a)
        split_rows.append({
            "split_seed": 2000 + i,
            "timestep_auroc": a,
            "n_test_rollouts": int(len(te_rids)),
            "n_test_failure_rollouts": nf,
            "n_test_success_rollouts": int(len(te_rids) - nf),
        })
        print(f"    split seed {2000+i}: AUROC {a:.4f} "
              f"({nf} fail / {len(te_rids)-nf} succ in test)", flush=True)

    # --- (C) Matched null -------------------------------------------------------------
    # (A) varies the permutation on one fixed split; (B) varies the split with true
    # labels. Comparing them conflates two variance sources. This resamples the split
    # *and* permutes, so it is the null for (B): the only systematic difference between
    # the two distributions is whether the labels are real.
    #
    # NB the split seeds here run 2000..2000+n_matched, so they coincide with (B)'s
    # 2000..2019 only when n_matched == n_splits. They are not paired in general. That
    # is harmless -- split seeds are exchangeable draws from one generator, so a wider
    # sweep is a better-sampled null -- but do not describe the two sets as paired.
    matched_aurocs, matched_skipped = [], 0
    for i in range(args.n_matched):
        g = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=2000 + i)
        s_tr, s_te = next(g.split(X, y, groups=groups))
        rng = np.random.RandomState(5000 + i)
        permuted = dict(zip(rids, rng.permutation(base_labels)))
        y_perm = np.array([permuted[gg] for gg in groups], dtype=np.int64)
        if len(np.unique(y_perm[s_tr])) < 2 or len(np.unique(y_perm[s_te])) < 2:
            matched_skipped += 1
            continue
        p = fit_and_score(X[s_tr], y_perm[s_tr], X[s_te], y_perm[s_te])
        a = float(roc_auc_score(y_perm[s_te], p))
        matched_aurocs.append(a)
        print(f"    matched null {i:3d} (split {2000+i}, perm {5000+i}): AUROC {a:.4f}",
              flush=True)

    ctrl_stats = summarise(ctrl_aurocs) if ctrl_aurocs else None
    split_stats = summarise(split_aurocs) if split_aurocs else None
    matched_stats = summarise(matched_aurocs) if matched_aurocs else None

    # Where the two locked numbers sit inside their own null / split distributions.
    locked_ctrl, locked_primary = 0.3954645826266934, 0.6699309387673692
    ctrl_pctile = (
        float((np.asarray(ctrl_aurocs) <= locked_ctrl).mean() * 100) if ctrl_aurocs else None
    )
    # One-sided permutation p: how often does a *permuted* labelling reach the real
    # AUROC? This is the question the control exists to answer.
    perm_p = (
        float((np.asarray(ctrl_aurocs) >= locked_primary).mean()) if ctrl_aurocs else None
    )

    out = {
        "layer": args.layer,
        "purpose": "variance diagnostic for the locked shuffle control; not a redesign",
        "locked_result": {
            "timestep_auroc": locked_primary,
            "shuffle_control_auroc": locked_ctrl,
            "control_acceptable_range": list(CONTROL_ACCEPTABLE),
        },
        "locked_split_test_composition": {
            "n_test_rollouts": int(len(np.unique(g_te))),
            "n_test_failure_rollouts": int(n_test_fail),
        },
        "control_distribution": {
            "description": "locked split (seed 42); rollout->label mapping permuted",
            "stats": ctrl_stats,
            "values": ctrl_aurocs,
            "n_skipped_single_class": ctrl_skipped,
            "locked_control_percentile": ctrl_pctile,
            "permutation_p_value_vs_primary": perm_p,
        },
        "split_sensitivity": {
            "description": "true labels; GroupShuffleSplit seeds 2000+ (diagnostic only, "
                           "the locked result remains the seed-42 number)",
            "stats": split_stats,
            "per_split": split_rows,
            "n_skipped_single_class": split_skipped,
        },
        "matched_null": {
            "description": "split resampled (seeds 2000+) AND labels permuted; the null "
                           "distribution for split_sensitivity",
            "stats": matched_stats,
            "values": matched_aurocs,
            "n_skipped_single_class": matched_skipped,
            # WARNING -- do not quote this field as a p-value. It compares individual
            # null draws (std ~0.16) against the *mean* of the true-label splits (SE
            # ~0.03), i.e. a draw against an average, so it reads small whether or not
            # an effect exists. The valid tests are computed post-hoc from `values`:
            # resample n_splits of these draws, take the mean, and locate the true mean
            # in that distribution; plus a rank-sum on the two sets. Retained only
            # because it appears in the run logs.
            "p_null_ge_true_split_mean_INVALID": (
                float((np.asarray(matched_aurocs) >= np.mean(split_aurocs)).mean())
                if matched_aurocs and split_aurocs else None
            ),
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = results_dir / "control_diagnostic.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== control distribution (locked split) ===", flush=True)
    if ctrl_stats:
        print(f"    mean {ctrl_stats['mean']:.4f}  median {ctrl_stats['median']:.4f}  "
              f"std {ctrl_stats['std']}", flush=True)
        print(f"    range [{ctrl_stats['min']:.4f}, {ctrl_stats['max']:.4f}]  "
              f"p05 {ctrl_stats['p05']:.4f}  p95 {ctrl_stats['p95']:.4f}", flush=True)
        print(f"    locked control 0.3955 sits at the {ctrl_pctile:.0f}th percentile "
              f"of this null", flush=True)
        print(f"    permutation p (null >= primary 0.6699): {perm_p:.3f}", flush=True)
    print("=== split sensitivity (true labels) ===", flush=True)
    if split_stats:
        print(f"    mean {split_stats['mean']:.4f}  median {split_stats['median']:.4f}  "
              f"std {split_stats['std']}", flush=True)
        print(f"    range [{split_stats['min']:.4f}, {split_stats['max']:.4f}]", flush=True)
    if matched_stats:
        print("=== matched null (split resampled + labels permuted) ===", flush=True)
        print(f"    mean {matched_stats['mean']:.4f}  median {matched_stats['median']:.4f}  "
              f"std {matched_stats['std']}", flush=True)
        print(f"    range [{matched_stats['min']:.4f}, {matched_stats['max']:.4f}]", flush=True)
        print(f"    true-label split mean {split_stats['mean']:.4f} vs matched null "
              f"{matched_stats['mean']:.4f}", flush=True)
        print(f"    [INVALID, do not quote] raw null>=true-mean fraction: "
              f"{out['matched_null']['p_null_ge_true_split_mean_INVALID']:.3f}", flush=True)
        print("    -> for the valid test, resample n_splits null draws from "
              "matched_null.values, take means, and locate the true mean", flush=True)
    print(f"=== wrote {path} ===", flush=True)


if __name__ == "__main__":
    main()
