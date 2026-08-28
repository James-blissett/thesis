"""
probe_late_window.py

Label scheme B for the layer-15 probe: restrict the probe to the LATE WINDOW of each
rollout instead of using every timestep.

Why. In the locked scheme every timestep inherits its rollout's final outcome. At
timestep 3 a doomed rollout looks identical to a successful one, so a large share of the
rows labelled "failure" are wrong at the timestep level. Restricting to late timesteps,
where the failure has physically manifested, removes the worst of that label noise. The
features, split machinery, scaler and classifier are otherwise byte-identical to
probe_layer.py -- only which rows are kept changes.

THE TRAP THIS SCRIPT AVOIDS. Failures run to the ~520-step cap; successes stop early.
Selecting "the last K steps by absolute index" would put failure examples at t~450-520,
a time region successes never reach, and anything correlated with elapsed time becomes a
shortcut that reads as failure signal. Selection is therefore by NORMALISED time only:

    keep timestep t (0-indexed) of a rollout of length T iff t / T >= 0.8

i.e. the final 20% of each rollout's OWN length. T is derived per rollout from the
tensor at runtime. No absolute-index selection appears anywhere in this file.

WHAT THE WINDOW RULE DOES NOT FIX. It stops the *selection* from manufacturing a time
confound, but it cannot remove the one the corpus already has: failures run to the ~520
cap and successes stop early, so a failure's late window still sits at absolute steps
~416-520 while a success's sits at ~230-290. If the hidden state encodes elapsed time at
all, the probe can read "this rollout has been running a long time" instead of "this
rollout is failing". The within-task null does not separate these either -- within a
task, failures are still the long rollouts -- so a within-task-significant result under
scheme B licenses "failure-specific signal" but NOT "an internal representation of
failure as such". Separating them needs a duration-matched or elapsed-time-regressed
control; that is out of scope for this pilot and belongs in the full-corpus run.

MANDATORY CONTROL. A late-window AUROC on its own re-imports exactly the confound the
within-task permutation was built to isolate: per-task success runs 40-100%, so a probe
that decoded only task identity would beat a global null. Every headline number here is
therefore accompanied by the within-task null (labels permuted only among rollouts
sharing a task_id) computed on the non-degenerate subset -- tasks that are all-success or
all-failure are inert under that permutation and are dropped, with counts derived from
the manifests at runtime.

Usage:
    source env.sh
    python analysis/probe_late_window.py
    python analysis/probe_late_window.py --n-perm-wt 200 --n-matched-wt 200 --n-splits 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from analyse_control import aggregate_test, single_split_test
from control_diagnostic import (
    CACHE_PATH,
    SEED_BASE_CTRL,
    SEED_BASE_CTRL_WT,
    SEED_BASE_MATCHED,
    SEED_BASE_MATCHED_WT,
    SEED_BASE_SPLIT,
    composition,
    degenerate_task_ids,
    fixed_split_null,
    global_permutation,
    load_or_cache,
    resampled_split_null,
    subset_by_tasks,
    summarise,
    within_task_permutation,
)
from probe_layer import (
    LAYER,
    POSITION_SCHEME,
    RANDOM_STATE,
    TEST_SIZE,
    corpus_manifest_hash,
    fit_and_score,
    rollout_level_auroc,
)

CORPUS_DIR = Path("/data/corpus")
RESULTS_DIR = Path("results/probe_late_window")

# The window rule. Fraction of each rollout's own length, NOT a step count.
LATE_WINDOW_FRAC = 0.8
WINDOW_RULE = (
    "keep timestep t (0-indexed) of a rollout of length T iff t/T >= 0.8; "
    "T derived per rollout from the tensor. Normalised time only -- selecting by "
    "absolute index would place failures in a time region successes never reach."
)


def late_window_mask(groups: np.ndarray, meta, frac: float):
    """Boolean mask over timesteps keeping the final `1 - frac` of each rollout.

    Returns (mask, normalised_time). Also verifies that each rollout's rows are one
    contiguous block in original timestep order and that their count matches the
    manifest's n_timesteps -- the assumption the whole window rule rests on.
    """
    n_by_rid = {m["rollout_id"]: int(m["n_timesteps"]) for m in meta}
    mask = np.zeros(groups.shape[0], dtype=bool)
    norm_t = np.full(groups.shape[0], np.nan, dtype=np.float64)

    for rid in sorted(n_by_rid):
        idx = np.flatnonzero(groups == rid)
        T = int(idx.size)
        if T == 0:
            raise RuntimeError(f"{rid}: no rows in the feature matrix")
        if idx[-1] - idx[0] + 1 != T:
            raise RuntimeError(
                f"{rid}: rows are not contiguous, so row order cannot be assumed to be "
                f"timestep order; the window rule would be meaningless")
        if T != n_by_rid[rid]:
            raise RuntimeError(
                f"{rid}: {T} feature rows but manifest says {n_by_rid[rid]} timesteps")
        t_over_T = np.arange(T, dtype=np.float64) / T
        norm_t[idx] = t_over_T
        mask[idx] = t_over_T >= frac

    return mask, norm_t


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window_counts(groups, y, task_ids, norm_t, meta) -> dict:
    """Kept-timestep counts, overall / per class / per rollout, plus the time span kept.

    The time span is the diagnostic that shows the window is not an absolute-index cut:
    the kept region covers the same normalised interval for every rollout regardless of
    length or outcome.
    """
    per_rollout = []
    n_by_rid = {m["rollout_id"]: int(m["n_timesteps"]) for m in meta}
    lab_by_rid = {m["rollout_id"]: int(m["label"]) for m in meta}
    for rid in sorted(n_by_rid):
        sel = groups == rid
        per_rollout.append({
            "rollout_id": rid,
            "label": lab_by_rid[rid],
            "T_full": n_by_rid[rid],
            "n_kept": int(sel.sum()),
            "kept_fraction": float(sel.sum() / n_by_rid[rid]),
        })
    return {
        "n_timesteps_kept": int(groups.shape[0]),
        "n_failure_timesteps": int((y == 1).sum()),
        "n_success_timesteps": int((y == 0).sum()),
        "n_rollouts": int(len(np.unique(groups))),
        "normalised_time_range_kept": [float(np.nanmin(norm_t)), float(np.nanmax(norm_t))],
        "kept_fraction_min": min(r["kept_fraction"] for r in per_rollout),
        "kept_fraction_max": max(r["kept_fraction"] for r in per_rollout),
        "per_rollout": per_rollout,
    }


def true_split_distribution(X, y, groups, rid_to_label, n_splits):
    """The 20-seed true-label split sweep, identical machinery to control_diagnostic (B)."""
    aurocs, rows, skipped = [], [], 0
    for i in range(n_splits):
        g = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                              random_state=SEED_BASE_SPLIT + i)
        s_tr, s_te = next(g.split(X, y, groups=groups))
        if len(np.unique(y[s_tr])) < 2 or len(np.unique(y[s_te])) < 2:
            skipped += 1
            continue
        p = fit_and_score(X[s_tr], y[s_tr], X[s_te], y[s_te])
        a = float(roc_auc_score(y[s_te], p))
        te_rids = np.unique(groups[s_te])
        nf = int(sum(rid_to_label[r] == 1 for r in te_rids))
        aurocs.append(a)
        rows.append({
            "split_seed": SEED_BASE_SPLIT + i,
            "timestep_auroc": a,
            "n_test_rollouts": int(len(te_rids)),
            "n_test_failure_rollouts": nf,
            "n_test_success_rollouts": int(len(te_rids) - nf),
        })
        print(f"    split seed {SEED_BASE_SPLIT + i}: AUROC {a:.4f} "
              f"({nf} fail / {len(te_rids)-nf} succ in test)", flush=True)
    return aurocs, rows, skipped


def locked_split_fit(X, y, groups, task_ids):
    """The locked seed-42 GroupShuffleSplit, refit on whatever rows are passed in."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        raise RuntimeError("a locked-split fold is single-class; AUROC undefined")
    probs = fit_and_score(X[tr], y[tr], X[te], y[te])
    return tr, te, probs, float(roc_auc_score(y[te], probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--corpus-dir", type=str, default=str(CORPUS_DIR))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--window-frac", type=float, default=LATE_WINDOW_FRAC)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--n-perm-wt", type=int, default=200)
    ap.add_argument("--n-matched-wt", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=200,
                    help="global-null counterpart of --n-perm-wt, for contrast")
    ap.add_argument("--n-matched", type=int, default=200,
                    help="global-null counterpart of --n-matched-wt, for contrast")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, task_ids, meta = load_or_cache(
        Path(args.corpus_dir), args.layer, use_cache=not args.no_cache
    )
    print(f"[*] full features {X.shape} | {len(meta)} rollouts | "
          f"failure timesteps {int(y.sum())}/{len(y)}", flush=True)

    # --- Apply the window ------------------------------------------------------------
    mask, norm_t = late_window_mask(groups, meta, args.window_frac)
    Xw, yw, gw, tw, ntw = X[mask], y[mask], groups[mask], task_ids[mask], norm_t[mask]
    del X
    counts_all = window_counts(gw, yw, tw, ntw, meta)
    print(f"[*] late window t/T >= {args.window_frac}: {counts_all['n_timesteps_kept']} "
          f"timesteps kept of {mask.size} "
          f"({counts_all['n_failure_timesteps']} failure / "
          f"{counts_all['n_success_timesteps']} success)", flush=True)
    print(f"[*] kept normalised-time range "
          f"[{counts_all['normalised_time_range_kept'][0]:.3f}, "
          f"{counts_all['normalised_time_range_kept'][1]:.3f}]; per-rollout kept fraction "
          f"{counts_all['kept_fraction_min']:.3f}-{counts_all['kept_fraction_max']:.3f}",
          flush=True)

    rid_to_label_all = {m["rollout_id"]: m["label"] for m in meta}

    # --- Scheme B on all tasks (continuity with the locked probe) ---------------------
    print("[*] all tasks: locked split + true-label split sweep", flush=True)
    tr, te, probs, auroc_all = locked_split_fit(Xw, yw, gw, tw)
    ro_auroc_all = rollout_level_auroc(probs, yw[te], gw[te])
    print(f"    locked-split timestep AUROC {auroc_all:.4f} | rollout {ro_auroc_all}",
          flush=True)
    split_all, rows_all, skipped_all = true_split_distribution(
        Xw, yw, gw, rid_to_label_all, args.n_splits)

    # --- The primary subset: degenerate tasks dropped ---------------------------------
    degenerate = degenerate_task_ids(meta)
    Xs, ys, gs, ts, meta_s = subset_by_tasks(Xw, yw, gw, tw, meta, degenerate)
    nts = ntw[np.isin(tw, degenerate, invert=True)]
    comp_s = composition(meta_s)
    counts_sub = window_counts(gs, ys, ts, nts, meta_s)
    print(f"[*] non-degenerate subset (dropping tasks {degenerate}): "
          f"{comp_s['n_rollouts']} rollouts "
          f"({comp_s['n_failure_rollouts']} failure / {comp_s['n_success_rollouts']} "
          f"success), {counts_sub['n_timesteps_kept']} late-window timesteps", flush=True)

    rid_to_label = {m["rollout_id"]: m["label"] for m in meta_s}
    rid_to_task = {m["rollout_id"]: int(m["task_id"]) for m in meta_s}
    rids = sorted(rid_to_label)
    base_labels = np.array([rid_to_label[r] for r in rids])

    s_tr, s_te, s_probs, auroc_sub = locked_split_fit(Xs, ys, gs, ts)
    ro_auroc_sub = rollout_level_auroc(s_probs, ys[s_te], gs[s_te])
    print(f"    subset locked-split timestep AUROC {auroc_sub:.4f} | "
          f"rollout {ro_auroc_sub}", flush=True)
    split_sub, rows_sub, skipped_sub = true_split_distribution(
        Xs, ys, gs, rid_to_label, args.n_splits)

    # --- Nulls on the subset. The within-task ones are the mandatory control. ----------
    ctrl, ctrl_skipped = fixed_split_null(
        Xs, gs, s_tr, s_te, rids, base_labels, rid_to_task,
        global_permutation, args.n_perm, SEED_BASE_CTRL, "control")
    ctrl_wt, ctrl_wt_skipped = fixed_split_null(
        Xs, gs, s_tr, s_te, rids, base_labels, rid_to_task,
        within_task_permutation, args.n_perm_wt, SEED_BASE_CTRL_WT, "control-wt")
    matched, matched_skipped = resampled_split_null(
        Xs, ys, gs, rids, base_labels, rid_to_task,
        global_permutation, args.n_matched, SEED_BASE_MATCHED, "matched null")
    matched_wt, matched_wt_skipped = resampled_split_null(
        Xs, ys, gs, rids, base_labels, rid_to_task,
        within_task_permutation, args.n_matched_wt, SEED_BASE_MATCHED_WT,
        "matched null-wt")

    # --- Significance, same tests as analyse_control.py --------------------------------
    true_sub = np.asarray(split_sub, dtype=float)
    tests = {}
    if ctrl:
        tests["single_split_vs_global_null"] = single_split_test(
            auroc_sub, np.asarray(ctrl, dtype=float))
    if ctrl_wt:
        tests["single_split_vs_within_task_null"] = single_split_test(
            auroc_sub, np.asarray(ctrl_wt, dtype=float))
    if matched:
        tests["aggregate_vs_global_null"] = aggregate_test(
            true_sub, np.asarray(matched, dtype=float), args.n_boot)
    if matched_wt:
        tests["aggregate_vs_within_task_null"] = aggregate_test(
            true_sub, np.asarray(matched_wt, dtype=float), args.n_boot)

    wt = tests.get("aggregate_vs_within_task_null")
    if wt is None:
        claim = "within-task control not run -- no failure-specific claim is licensed"
    elif wt["p_bootstrap_of_means"] < 0.05:
        claim = ("late-window labels: signal survives the within-task null on the "
                 "non-degenerate subset -- directional failure-specific signal, pilot-scale")
    else:
        claim = ("late-window labels: signal does NOT survive the within-task null on the "
                 "non-degenerate subset -- failure signal not demonstrated at this corpus "
                 "size under scheme B")

    cache = Path(str(CACHE_PATH).format(layer=args.layer))
    manifests = sorted(Path(args.corpus_dir).glob("task*_ep*.json"))

    metrics = {
        "label_scheme": "B -- late window of each rollout, rollout outcome broadcast to "
                        "the kept timesteps only",
        "layer": args.layer,
        "position_scheme": POSITION_SCHEME,
        "window": {
            "rule": WINDOW_RULE,
            "frac": args.window_frac,
            "selection": "normalised time (t/T); absolute-index selection is never used",
            "residual_confound": (
                "The rule stops the selection from manufacturing a time confound but "
                "cannot remove the corpus's own: failures run to the ~520 cap, so their "
                "late window sits at absolute steps ~416-520 while successes' sits at "
                "~230-290. The within-task null does not separate elapsed-time decoding "
                "from failure decoding either, since within a task the failures are still "
                "the long rollouts. A within-task-significant result licenses "
                "'failure-specific signal', not 'a representation of failure as such'."
            ),
        },
        "seeds": {
            "locked_split": RANDOM_STATE,
            "split_sweep": [SEED_BASE_SPLIT + i for i in range(args.n_splits)],
            "global_perm_locked_split": [SEED_BASE_CTRL + i for i in range(args.n_perm)],
            "within_task_perm_locked_split":
                [SEED_BASE_CTRL_WT + i for i in range(args.n_perm_wt)],
            "global_perm_resampled_split":
                [SEED_BASE_MATCHED + i for i in range(args.n_matched)],
            "within_task_perm_resampled_split":
                [SEED_BASE_MATCHED_WT + i for i in range(args.n_matched_wt)],
            "bootstrap": RANDOM_STATE,
        },
        "provenance": {
            "corpus_dir": str(args.corpus_dir),
            "corpus_manifest_hash": corpus_manifest_hash(manifests),
            "feature_cache": str(cache),
            "feature_cache_sha256": file_sha256(cache) if cache.exists() else None,
        },
        "counts": {
            "all_tasks": counts_all,
            "non_degenerate_subset": counts_sub,
            "degenerate_task_ids": degenerate,
            "degenerate_task_rule": "every rollout of the task shares one outcome, so a "
                                    "within-task permutation cannot change its labels",
            "counts_source": "derived at runtime from the per-rollout manifests",
            "subset_composition": comp_s,
        },
        "all_tasks": {
            "locked_split_timestep_auroc": auroc_all,
            "locked_split_rollout_auroc": ro_auroc_all,
            "split_sweep": {
                "stats": summarise(split_all) if split_all else None,
                "per_split": rows_all,
                "n_skipped_single_class": skipped_all,
            },
            "note": "includes the degenerate tasks, which the within-task null cannot "
                    "touch; reported for continuity, not as the primary number",
        },
        "non_degenerate_subset": {
            "locked_split_timestep_auroc": auroc_sub,
            "locked_split_rollout_auroc": ro_auroc_sub,
            "split_sweep": {
                "stats": summarise(split_sub) if split_sub else None,
                "per_split": rows_sub,
                "n_skipped_single_class": skipped_sub,
            },
            "global_null_locked_split": {
                "stats": summarise(ctrl) if ctrl else None,
                "values": ctrl,
                "n_skipped_single_class": ctrl_skipped,
            },
            "within_task_null_locked_split": {
                "stats": summarise(ctrl_wt) if ctrl_wt else None,
                "values": ctrl_wt,
                "n_skipped_single_class": ctrl_wt_skipped,
            },
            "global_null_resampled_split": {
                "stats": summarise(matched) if matched else None,
                "values": matched,
                "n_skipped_single_class": matched_skipped,
            },
            "within_task_null_resampled_split": {
                "stats": summarise(matched_wt) if matched_wt else None,
                "values": matched_wt,
                "n_skipped_single_class": matched_wt_skipped,
            },
            "tests": tests,
        },
        "claim_reached": claim,
        "model": "identical to probe_layer.py: StandardScaler (train-only fit) -> "
                 "LogisticRegression(C=1.0, class_weight='balanced', lbfgs, max_iter=2000)",
        "split": "identical to probe_layer.py: GroupShuffleSplit by rollout_id, "
                 f"test_size={TEST_SIZE}",
        "runtime_seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        results_dir / "scores_test.npz",
        probs=probs,
        labels=yw[te],
        rollout_ids=gw[te].astype(str),
        task_ids=tw[te],
        norm_t=ntw[te],
        subset_probs=s_probs,
        subset_labels=ys[s_te],
        subset_rollout_ids=gs[s_te].astype(str),
        subset_task_ids=ts[s_te],
        subset_norm_t=nts[s_te],
    )

    print(f"\n=== label scheme B (late window t/T >= {args.window_frac}) ===")
    print(f"all tasks    : locked split {auroc_all:.4f} | "
          f"{args.n_splits}-seed mean {metrics['all_tasks']['split_sweep']['stats']['mean']:.4f} "
          f"(sd {metrics['all_tasks']['split_sweep']['stats']['std']:.4f})")
    st = metrics["non_degenerate_subset"]["split_sweep"]["stats"]
    print(f"subset       : locked split {auroc_sub:.4f} | "
          f"{args.n_splits}-seed mean {st['mean']:.4f} (sd {st['std']:.4f})")
    for label, key in (("global", "aggregate_vs_global_null"),
                       ("within-task", "aggregate_vs_within_task_null")):
        a = tests.get(key)
        if a:
            sd = "n/a" if a["sd_above_null"] is None else f"{a['sd_above_null']:.1f}"
            print(f"vs {label:11s}: null {a['null_mean']:.4f} (n={a['n_null_draws']}) | "
                  f"{sd} SD | p {a['p_bootstrap_reportable']} | "
                  f"ranksum {a['p_ranksum']:.4f} -> {a['verdict']}")
    print(f"CLAIM        : {claim}")
    print(f"=== wrote {results_dir}/metrics.json, scores_test.npz "
          f"({metrics['runtime_seconds']}s) ===")


if __name__ == "__main__":
    main()
