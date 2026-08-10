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

Distributions produced, reported separately:

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

  (A-wt) / (C-wt) Within-task nulls -- as (A) and (C), but labels are permuted only
      among rollouts that SHARE A task_id. Off by default; --n-perm-wt / --n-matched-wt.

Why the within-task nulls exist. (A) and (C) permute the rollout->label mapping
*globally*, which destroys the task->outcome correlation along with the failure signal.
Per-task success runs 40-100% here, so a probe that decoded only *which task this is*
beats those nulls comfortably, and a significant result against them cannot be read as
failure-specific. Permuting within task leaves task identity exactly where it was and
destroys only the failure signal, so a gap between the true statistic and the
within-task null is failure-specific by construction.

Degenerate tasks. A task whose rollouts are all-success or all-failure is inert under a
within-task permutation -- shuffling identical labels changes nothing -- so those
rollouts contribute label-invariant rows to both the true and the null side, and the
only thing a probe can extract from them is task identity. --exclude-degenerate-tasks
drops them (detected from the manifests, never hardcoded) and recomputes EVERY
distribution, true ones included, on the surviving rollouts. That is the primary
analysis; the full-corpus run is kept for continuity with the 2026-08-01 numbers. Null
and true statistics must always come from identical rollouts or the comparison is void.

This script only produces distributions. The significance statistics are computed from
them by analyse_control.py -- see the warning on `p_null_ge_true_split_mean_INVALID`.

Usage:
    source env.sh
    python control_diagnostic.py                 # 20 permutations, 20 splits

    # primary analysis (degenerate tasks excluded) and its secondary counterpart:
    python control_diagnostic.py --n-perm 200 --n-splits 20 --n-matched 200 \
        --n-perm-wt 200 --n-matched-wt 200 --exclude-degenerate-tasks
    python control_diagnostic.py --n-perm 200 --n-splits 20 --n-matched 200 \
        --n-perm-wt 200 --n-matched-wt 200

    python analyse_control.py                    # valid tests over the above
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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

# Permutation seed bases, one per distribution, so no two distributions ever share a
# draw and any single value can be reproduced from its printed seed.
SEED_BASE_CTRL = 1000        # (A)   global permutation, locked split
SEED_BASE_SPLIT = 2000       # (B)   split reseeding, true labels -- also reused by C/C-wt
SEED_BASE_CTRL_WT = 3000     # (A-wt) within-task permutation, locked split
SEED_BASE_MATCHED = 5000     # (C)   global permutation, resampled split
SEED_BASE_MATCHED_WT = 7000  # (C-wt) within-task permutation, resampled split

# The published full-corpus numbers from the 2026-08-01 locked run, kept for continuity.
PUBLISHED_LOCKED_AUROC = 0.6699309387673692
PUBLISHED_LOCKED_CONTROL = 0.3954645826266934


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


# --- Rollout bookkeeping ---------------------------------------------------------------

def degenerate_task_ids(meta) -> list[int]:
    """Tasks whose rollouts are all-success or all-failure, read off the manifests.

    A within-task permutation cannot move a label inside such a task, so these rollouts
    sit identically in the true and the null statistic and carry only task identity.
    """
    labels_by_task: dict[int, set[int]] = defaultdict(set)
    for m in meta:
        labels_by_task[int(m["task_id"])].add(int(m["label"]))
    return sorted(t for t, labs in labels_by_task.items() if len(labs) < 2)


def subset_by_tasks(X, y, groups, task_ids, meta, drop_tasks):
    """Restrict every array and the rollout metadata to tasks not in drop_tasks."""
    if not drop_tasks:
        return X, y, groups, task_ids, meta
    drop = set(int(t) for t in drop_tasks)
    keep = ~np.isin(task_ids, list(drop))
    meta_keep = [m for m in meta if int(m["task_id"]) not in drop]
    return X[keep], y[keep], groups[keep], task_ids[keep], meta_keep


def composition(meta) -> dict:
    """Rollout counts overall and per task, derived from the manifests at runtime."""
    per_task: dict[str, dict[str, int]] = {}
    for m in sorted(meta, key=lambda m: (int(m["task_id"]), str(m["rollout_id"]))):
        d = per_task.setdefault(str(m["task_id"]), {"failure": 0, "success": 0})
        d["failure" if m["label"] == 1 else "success"] += 1
    return {
        "n_rollouts": len(meta),
        "n_failure_rollouts": sum(1 for m in meta if m["label"] == 1),
        "n_success_rollouts": sum(1 for m in meta if m["label"] == 0),
        "n_timesteps": int(sum(m["n_timesteps"] for m in meta)),
        "per_task": per_task,
    }


# --- Permutation schemes ----------------------------------------------------------------

def global_permutation(rids, base_labels, rid_to_task, rng) -> dict:
    """Shuffle the rollout->label mapping across the whole corpus (nulls A and C).

    Destroys the failure signal AND the task->outcome correlation, so it cannot separate
    failure signal from task identity.
    """
    return dict(zip(rids, rng.permutation(base_labels)))


def within_task_permutation(rids, base_labels, rid_to_task, rng) -> dict:
    """Shuffle labels only among rollouts sharing a task_id (nulls A-wt and C-wt).

    Task identity is left exactly as it was; only the failure signal is destroyed. Tasks
    whose labels are all identical are unchanged by construction -- that is why the
    primary analysis excludes them.
    """
    label_of = dict(zip(rids, base_labels))
    by_task: dict[int, list] = defaultdict(list)
    for r in rids:
        by_task[int(rid_to_task[r])].append(r)

    mapping = {}
    for task in sorted(by_task):
        members = sorted(by_task[task])
        mapping.update(zip(members, rng.permutation([label_of[r] for r in members])))
    return mapping


# --- Distribution runners ---------------------------------------------------------------

def fixed_split_null(X, groups, tr, te, rids, base_labels, rid_to_task,
                     permute_fn, n, seed_base, tag):
    """N permuted-label refits on ONE fixed split (the (A)/(A-wt) family)."""
    aurocs, skipped = [], 0
    for i in range(n):
        rng = np.random.RandomState(seed_base + i)
        mapping = permute_fn(rids, base_labels, rid_to_task, rng)
        y_perm = np.array([mapping[g] for g in groups], dtype=np.int64)
        # A permutation that leaves either fold single-class has no defined AUROC; it is
        # a property of the draw, not a failure, so count it rather than aborting.
        if len(np.unique(y_perm[tr])) < 2 or len(np.unique(y_perm[te])) < 2:
            skipped += 1
            continue
        p = fit_and_score(X[tr], y_perm[tr], X[te], y_perm[te])
        a = float(roc_auc_score(y_perm[te], p))
        aurocs.append(a)
        print(f"    {tag} perm {i:3d} (seed {seed_base + i}): AUROC {a:.4f}", flush=True)
    return aurocs, skipped


def resampled_split_null(X, y, groups, rids, base_labels, rid_to_task,
                         permute_fn, n, perm_seed_base, tag, test_size=TEST_SIZE):
    """N refits with the split resampled AND the labels permuted (the (C)/(C-wt) family).

    This is the null for distribution (B): both sides vary the split, so the only
    systematic difference between them is whether the labels are real.

    NB the split seeds run SEED_BASE_SPLIT..+n, so they coincide with (B)'s only when n
    equals --n-splits. They are not paired in general. That is harmless -- split seeds
    are exchangeable draws from one generator, so a wider sweep is a better-sampled null
    -- but do not describe the two sets as paired.
    """
    aurocs, skipped = [], 0
    for i in range(n):
        g = GroupShuffleSplit(n_splits=1, test_size=test_size,
                              random_state=SEED_BASE_SPLIT + i)
        s_tr, s_te = next(g.split(X, y, groups=groups))
        rng = np.random.RandomState(perm_seed_base + i)
        mapping = permute_fn(rids, base_labels, rid_to_task, rng)
        y_perm = np.array([mapping[gg] for gg in groups], dtype=np.int64)
        if len(np.unique(y_perm[s_tr])) < 2 or len(np.unique(y_perm[s_te])) < 2:
            skipped += 1
            continue
        p = fit_and_score(X[s_tr], y_perm[s_tr], X[s_te], y_perm[s_te])
        a = float(roc_auc_score(y_perm[s_te], p))
        aurocs.append(a)
        print(f"    {tag} {i:3d} (split {SEED_BASE_SPLIT + i}, perm {perm_seed_base + i}): "
              f"AUROC {a:.4f}", flush=True)
    return aurocs, skipped


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
    ap.add_argument("--n-perm-wt", type=int, default=0,
                    help="A-wt: locked split, labels permuted WITHIN task, N draws")
    ap.add_argument("--n-matched-wt", type=int, default=0,
                    help="C-wt: resample split AND permute labels WITHIN task, N draws")
    ap.add_argument("--exclude-degenerate-tasks", action="store_true",
                    help="drop tasks that are all-success or all-failure (detected from "
                         "the manifests) and recompute every distribution on the rest")
    ap.add_argument("--out-name", type=str, default=None,
                    help="output filename; defaults to control_diagnostic[_nondegenerate].json")
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

    # --- Subset selection ------------------------------------------------------------
    # Counts are read off the manifest-derived metadata every run; nothing is hardcoded.
    degenerate = degenerate_task_ids(meta)
    full_comp = composition(meta)
    per_task_str = ", ".join(
        "task {}: {}F/{}S".format(k, v["failure"], v["success"])
        for k, v in full_comp["per_task"].items()
    )
    print(f"[*] per-task composition: {per_task_str}", flush=True)
    print(f"[*] degenerate tasks (all one outcome, inert under within-task permutation): "
          f"{degenerate}", flush=True)

    dropped = degenerate if args.exclude_degenerate_tasks else []
    if dropped:
        X, y, groups, task_ids, meta = subset_by_tasks(X, y, groups, task_ids, meta, dropped)
    comp = composition(meta)
    print(f"[*] analysis subset: {'degenerate tasks excluded' if dropped else 'all tasks'} "
          f"-> {comp['n_rollouts']} rollouts "
          f"({comp['n_failure_rollouts']} failure / {comp['n_success_rollouts']} success), "
          f"{comp['n_timesteps']} timesteps", flush=True)

    if args.n_perm_wt or args.n_matched_wt:
        n_inert = sum(1 for m in meta if int(m["task_id"]) in set(degenerate))
        if n_inert:
            print(f"[!] {n_inert} rollouts belong to degenerate tasks and are INERT under "
                  f"the within-task nulls; run --exclude-degenerate-tasks for the primary "
                  f"analysis", flush=True)

    rid_to_label = {m["rollout_id"]: m["label"] for m in meta}
    rid_to_task = {m["rollout_id"]: int(m["task_id"]) for m in meta}
    rids = sorted(rid_to_label)
    base_labels = np.array([rid_to_label[r] for r in rids])

    # --- The locked split, recomputed on whatever subset is in play -------------------
    # True and null statistics must come from identical rollouts, so the "true" number
    # this run is judged against is computed here rather than quoted from the full-corpus
    # published value.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    g_te = groups[te]
    n_test_fail = len({g for g in g_te if rid_to_label[g] == 1})
    print(f"[*] locked split: {len(np.unique(g_te))} test rollouts, "
          f"{n_test_fail} of them failures", flush=True)

    locked_probs = fit_and_score(X[tr], y[tr], X[te], y[te])
    locked_auroc = float(roc_auc_score(y[te], locked_probs))

    rng = np.random.RandomState(RANDOM_STATE)
    single_map = dict(zip(rids, rng.permutation(base_labels)))
    y_single = np.array([single_map[g] for g in groups], dtype=np.int64)
    if len(np.unique(y_single[tr])) < 2 or len(np.unique(y_single[te])) < 2:
        locked_ctrl = None
    else:
        locked_ctrl = float(roc_auc_score(
            y_single[te], fit_and_score(X[tr], y_single[tr], X[te], y_single[te])))
    reproduces = (
        not dropped
        and abs(locked_auroc - PUBLISHED_LOCKED_AUROC) < 1e-9
        and locked_ctrl is not None
        and abs(locked_ctrl - PUBLISHED_LOCKED_CONTROL) < 1e-9
    )
    print(f"[*] locked-split true AUROC {locked_auroc:.4f} | single global control "
          f"{locked_ctrl if locked_ctrl is None else round(locked_ctrl, 4)}"
          f"{' | reproduces the published full-corpus pair' if reproduces else ''}",
          flush=True)

    # --- (A) Control distribution on the locked split, global permutation --------------
    ctrl_aurocs, ctrl_skipped = fixed_split_null(
        X, groups, tr, te, rids, base_labels, rid_to_task,
        global_permutation, args.n_perm, SEED_BASE_CTRL, "control")

    # --- (A-wt) Same split, labels permuted WITHIN task --------------------------------
    ctrl_wt_aurocs, ctrl_wt_skipped = fixed_split_null(
        X, groups, tr, te, rids, base_labels, rid_to_task,
        within_task_permutation, args.n_perm_wt, SEED_BASE_CTRL_WT, "control-wt")

    # --- (B) Split sensitivity with true labels ---------------------------------------
    split_aurocs, split_rows, split_skipped = [], [], 0
    for i in range(args.n_splits):
        g = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                              random_state=SEED_BASE_SPLIT + i)
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
            "split_seed": SEED_BASE_SPLIT + i,
            "timestep_auroc": a,
            "n_test_rollouts": int(len(te_rids)),
            "n_test_failure_rollouts": nf,
            "n_test_success_rollouts": int(len(te_rids) - nf),
        })
        print(f"    split seed {SEED_BASE_SPLIT + i}: AUROC {a:.4f} "
              f"({nf} fail / {len(te_rids)-nf} succ in test)", flush=True)

    # --- (C) Matched null, global permutation -----------------------------------------
    matched_aurocs, matched_skipped = resampled_split_null(
        X, y, groups, rids, base_labels, rid_to_task,
        global_permutation, args.n_matched, SEED_BASE_MATCHED, "matched null")

    # --- (C-wt) Matched null, within-task permutation ----------------------------------
    # The null for (B) that holds task identity fixed. B-vs-C-wt on the non-degenerate
    # subset is the primary comparison of the whole analysis.
    matched_wt_aurocs, matched_wt_skipped = resampled_split_null(
        X, y, groups, rids, base_labels, rid_to_task,
        within_task_permutation, args.n_matched_wt, SEED_BASE_MATCHED_WT,
        "matched null-wt")

    ctrl_stats = summarise(ctrl_aurocs) if ctrl_aurocs else None
    ctrl_wt_stats = summarise(ctrl_wt_aurocs) if ctrl_wt_aurocs else None
    split_stats = summarise(split_aurocs) if split_aurocs else None
    matched_stats = summarise(matched_aurocs) if matched_aurocs else None
    matched_wt_stats = summarise(matched_wt_aurocs) if matched_wt_aurocs else None

    def pctile_of(values, x):
        return float((np.asarray(values) <= x).mean() * 100) if values else None

    def p_ge(values, x):
        # One-sided permutation p: how often does a permuted labelling reach the real
        # AUROC? This is the question a control exists to answer.
        return float((np.asarray(values) >= x).mean()) if values else None

    out = {
        "layer": args.layer,
        "purpose": "variance diagnostic for the locked shuffle control; not a redesign",
        "subset": {
            "exclude_degenerate_tasks": bool(args.exclude_degenerate_tasks),
            "degenerate_task_ids": degenerate,
            "degenerate_task_rule": "every rollout of the task shares one outcome, so a "
                                    "within-task permutation cannot change its labels",
            "excluded_task_ids": dropped,
            "counts_source": "derived at runtime from the per-rollout manifests",
            "full_corpus_composition": full_comp,
            "analysed_composition": comp,
        },
        "locked_result": {
            # Computed on the current subset -- this is the number the nulls below are
            # judged against, and on the full corpus it reproduces the published pair.
            "timestep_auroc": locked_auroc,
            "shuffle_control_auroc": locked_ctrl,
            "control_acceptable_range": list(CONTROL_ACCEPTABLE),
            "published_full_corpus": {
                "timestep_auroc": PUBLISHED_LOCKED_AUROC,
                "shuffle_control_auroc": PUBLISHED_LOCKED_CONTROL,
            },
            "reproduces_published_full_corpus": bool(reproduces),
        },
        "locked_split_test_composition": {
            "n_test_rollouts": int(len(np.unique(g_te))),
            "n_test_failure_rollouts": int(n_test_fail),
        },
        "control_distribution": {
            "description": "locked split (seed 42); rollout->label mapping permuted "
                           "GLOBALLY -- destroys task identity along with the signal",
            "permutation_scheme": "global",
            "stats": ctrl_stats,
            "values": ctrl_aurocs,
            "n_skipped_single_class": ctrl_skipped,
            "locked_control_percentile": pctile_of(ctrl_aurocs, locked_ctrl)
                                         if locked_ctrl is not None else None,
            "permutation_p_value_vs_primary": p_ge(ctrl_aurocs, locked_auroc),
        },
        "control_distribution_within_task": {
            "description": "locked split (seed 42); labels permuted WITHIN task_id -- "
                           "task identity held fixed, only the failure signal destroyed",
            "permutation_scheme": "within_task",
            "stats": ctrl_wt_stats,
            "values": ctrl_wt_aurocs,
            "n_skipped_single_class": ctrl_wt_skipped,
            "permutation_p_value_vs_primary": p_ge(ctrl_wt_aurocs, locked_auroc),
        },
        "split_sensitivity": {
            "description": "true labels; GroupShuffleSplit seeds 2000+ (diagnostic only, "
                           "the locked result remains the seed-42 number)",
            "stats": split_stats,
            "per_split": split_rows,
            "n_skipped_single_class": split_skipped,
        },
        "matched_null": {
            "description": "split resampled (seeds 2000+) AND labels permuted GLOBALLY; "
                           "the task-confounded null distribution for split_sensitivity",
            "permutation_scheme": "global",
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
        "matched_null_within_task": {
            "description": "split resampled (seeds 2000+) AND labels permuted WITHIN "
                           "task_id; the failure-specific null for split_sensitivity. "
                           "B vs this, on the non-degenerate subset, is the primary "
                           "comparison.",
            "permutation_scheme": "within_task",
            "stats": matched_wt_stats,
            "values": matched_wt_aurocs,
            "n_skipped_single_class": matched_wt_skipped,
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    name = args.out_name or (
        "control_diagnostic_nondegenerate.json" if args.exclude_degenerate_tasks
        else "control_diagnostic.json"
    )
    path = results_dir / name
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    def report(title, stats):
        if not stats:
            return
        print(f"=== {title} ===", flush=True)
        print(f"    mean {stats['mean']:.4f}  median {stats['median']:.4f}  "
              f"std {stats['std']}", flush=True)
        print(f"    range [{stats['min']:.4f}, {stats['max']:.4f}]  "
              f"p05 {stats['p05']:.4f}  p95 {stats['p95']:.4f}", flush=True)

    print(f"\n=== locked split (subset: {'non-degenerate' if dropped else 'all tasks'}) ===")
    print(f"    true AUROC {locked_auroc:.4f}", flush=True)
    report("control distribution (A: global perm, locked split)", ctrl_stats)
    if ctrl_stats:
        print(f"    permutation p (null >= true {locked_auroc:.4f}): "
              f"{out['control_distribution']['permutation_p_value_vs_primary']:.3f}",
              flush=True)
    report("control distribution (A-wt: within-task perm, locked split)", ctrl_wt_stats)
    if ctrl_wt_stats:
        print(f"    permutation p (null >= true {locked_auroc:.4f}): "
              f"{out['control_distribution_within_task']['permutation_p_value_vs_primary']:.3f}",
              flush=True)
    report("split sensitivity (B: true labels)", split_stats)
    report("matched null (C: global perm + resampled split)", matched_stats)
    report("matched null (C-wt: within-task perm + resampled split)", matched_wt_stats)
    if split_stats and matched_wt_stats:
        line = (f"    PRIMARY: true-label split mean {split_stats['mean']:.4f} vs "
                f"within-task null {matched_wt_stats['mean']:.4f}")
        if matched_stats:
            line += f" (global null {matched_stats['mean']:.4f})"
        print(line, flush=True)
        print("    -> run analyse_control.py for the valid test (bootstrap of means)",
              flush=True)
    print(f"=== wrote {path} ===", flush=True)


if __name__ == "__main__":
    main()
