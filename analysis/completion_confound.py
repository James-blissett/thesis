"""
completion_confound.py

Step 2c Part A: does the mid-rollout act_rep signal measure impending failure, or
does it measure successes having already finished and gone quiet?

THE CONFOUND. Every rollout runs the full 520-step budget regardless of outcome --
gen_rollouts_v2.run_episode has no early termination. A success that completes at
t = 200 keeps emitting policy steps for another 320. If the policy freezes once the
task is done, then late in an episode "frozen" marks SUCCESS, not failure, and the
pooled AUROC is a mixture of two opposite effects. This script fixes the sign of that
mixture before Part B spends any effort on scoring.

TIME BASE, VERIFIED AGAINST THE CAPTURE CODE. gen_rollouts_v2 steps t_env over
range(NUM_STEPS_WAIT + MAX_POLICY_STEPS), records t_success_env at the first t_env
whose env.step() leaves check_success() true, and stores

    t_success = t_success_env - NUM_STEPS_WAIT

which is exactly the policy-step index used by policy_step_env_t and therefore exactly
the `t` index of every constraint series. No offset is applied here. Note that success
is checked AFTER the action at index t_success executes, so t_success is the step whose
action completed the task and the episode is still genuinely in progress at its start.

Usage:
    source env.sh
    python analysis/completion_confound.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constraint_auroc import ACT_REP_EPS

PARQUET = Path("constraints/all.parquet")
INDEX_JSON = Path("corpus_v2_index.json")
CDF_LEVELS = (0.05, 0.25, 0.50, 0.75)
REP_PEAK = (186, 221)      # act_rep AUROC-vs-t peak located in step 2b


def load_actions():
    """act_mag / act_rep as (n_rollouts, T), plus t_first_success in policy steps."""
    tbl = pq.read_table(PARQUET, filters=[("constraint_name", "in", ["act_mag"])],
                        columns=["rollout_id", "t", "value"])
    rids = sorted(set(pc.unique(tbl.column("rollout_id")).to_pylist()))
    t = tbl.column("t").to_numpy(zero_copy_only=False).astype(np.int64)
    T = int(t.max()) + 1
    ridx = pc.index_in(tbl.column("rollout_id"),
                       value_set=pa.array(rids, type=pa.string())
                       ).to_numpy(zero_copy_only=False).astype(np.int64)
    v = tbl.column("value").to_numpy(zero_copy_only=False).astype(np.float32)
    flat = ridx * T + t
    if flat.size != len(rids) * T or np.unique(flat).size != flat.size:
        raise SystemExit("act_mag does not fill the (rollout, t) grid")
    mag = np.empty(len(rids) * T, dtype=np.float32)
    mag[flat] = v
    mag = mag.reshape(len(rids), T)
    rep = np.where(np.isnan(mag), np.nan,
                   (mag <= ACT_REP_EPS).astype(np.float32)).astype(np.float32)

    index = {e["rollout_id"]: e for e in json.loads(INDEX_JSON.read_text())["rollouts"]}
    tfs = np.array([index[r]["t_success"] if index[r]["t_success"] is not None else np.nan
                    for r in rids], dtype=np.float64)
    succ = np.array([bool(index[r]["success_ever"]) for r in rids])
    if not np.array_equal(np.isfinite(tfs), succ):
        raise SystemExit("t_success is defined exactly on the success_ever rollouts -- "
                         "this assertion should be unreachable")
    return rids, mag, rep, tfs, succ, T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/completion_confound.json")
    args = ap.parse_args()

    rids, mag, rep, tfs, succ, T = load_actions()
    ts = np.sort(tfs[succ])
    n = ts.size
    print(f"[*] {len(rids)} rollouts, {int(succ.sum())} success_ever, T = {T}\n")

    # --- A1: empirical CDF of t_first_success -------------------------------------
    print("A1. empirical CDF of t_first_success over the "
          f"{n} success_ever rollouts")
    print(f"    range [{ts.min():.0f}, {ts.max():.0f}]  median {np.median(ts):.0f}  "
          f"mean {ts.mean():.1f}")
    cdf = {}
    for q in CDF_LEVELS:
        # Smallest t at which the empirical CDF has reached q, i.e. the ceil(q*n)-th
        # order statistic. Reported on the integer step grid the series live on.
        k = int(np.ceil(q * n)) - 1
        cdf[q] = float(ts[k])
        print(f"    CDF = {q:.2f}  reached at t = {ts[k]:.0f}   "
              f"({k+1}/{n} successes have completed)")

    lo, hi = REP_PEAK
    frac_lo = float((ts <= lo).mean())
    frac_hi = float((ts <= hi).mean())
    print(f"\n    act_rep AUROC-vs-t peak sits at t = {lo}-{hi}.")
    print(f"    by t = {lo}: {frac_lo:6.1%} of successes have already completed "
          f"({int((ts <= lo).sum())}/{n})")
    print(f"    by t = {hi}: {frac_hi:6.1%} of successes have already completed "
          f"({int((ts <= hi).sum())}/{n})")

    # --- A2: behaviour before vs after completion ---------------------------------
    print("\nA2. success rollouts, before vs after their own completion step")
    rows = []
    for label, arr in (("act_rep", rep), ("act_mag", mag)):
        pre_v, post_v, pre_n, post_n = [], [], 0, 0
        for i in np.nonzero(succ)[0]:
            k = int(tfs[i])
            a = arr[i]
            pre = a[:k][np.isfinite(a[:k])]
            post = a[k:][np.isfinite(a[k:])]
            pre_v.append(pre.sum()); post_v.append(post.sum())
            pre_n += pre.size; post_n += post.size
        pre_m = float(np.sum(pre_v) / pre_n)
        post_m = float(np.sum(post_v) / post_n)
        rows.append((label, pre_m, post_m, pre_n, post_n))
        print(f"    {label:<8} t <  t_first_success: {pre_m:.5f}  ({pre_n:,} rows)")
        print(f"    {label:<8} t >= t_first_success: {post_m:.5f}  ({post_n:,} rows)")
        print(f"    {label:<8} ratio after/before  : {post_m/pre_m:6.2f}x")

    # The mean is not enough for act_mag. Report the whole distribution, because a
    # flat mean can hide a total change of shape -- and here it does.
    print("\n    act_mag distribution, success rollouts, pre vs post completion")
    dist = {}
    for tag, sl in (("pre", lambda a, k: a[:k]), ("post", lambda a, k: a[k:])):
        x = np.concatenate([sl(mag[i], int(tfs[i]))[np.isfinite(sl(mag[i], int(tfs[i])))]
                            for i in np.nonzero(succ)[0]])
        q = np.percentile(x, [25, 50, 75, 90, 99])
        dist[tag] = {"n": int(x.size), "mean": float(x.mean()),
                     "p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2]),
                     "p90": float(q[3]), "p99": float(q[4]),
                     "frac_frozen": float((x <= ACT_REP_EPS).mean())}
        print(f"      {tag:<4} p25/p50/p75/p90/p99  " + "  ".join(f"{v:.4f}" for v in q)
              + f"   frac frozen {(x <= ACT_REP_EPS).mean():.3f}")

    # Does the AUROC-vs-t decline track the completion CDF? Read the curve step 2b
    # wrote rather than recomputing it, so this compares against the drawn figure.
    corr = None
    curve = Path("results/action_auroc_by_t_success_ever_w21.csv")
    if curve.exists():
        import csv as _csv
        cr = [r for r in _csv.DictReader(open(curve))
              if r["constraint"] == "act_rep_w21"]
        tt = np.array([int(r["t"]) for r in cr])
        au = np.array([float(r["auroc_raw"]) for r in cr])
        cdf_t = np.array([(ts <= t).mean() for t in range(T)])
        pk = int(tt[np.argmax(au)])
        seg = tt >= pk
        corr = float(np.corrcoef(au[seg], cdf_t[tt[seg]])[0, 1])
        print(f"\n    act_rep_w21 AUROC-vs-t peaks at t = {pk} ({au.max():.4f}); "
              f"from there to t = 519")
        print(f"    corr(AUROC, completion CDF) = {corr:+.4f} over {int(seg.sum())} "
              f"timesteps -- the decline tracks the CDF rise")
    else:
        print(f"\n    [--] {curve} absent; run plot_action_auroc_by_t.py --window 21")

    rep_pre, rep_post = rows[0][1], rows[0][2]
    freeze = rep_post > rep_pre
    print(f"\n    successes {'FREEZE' if freeze else 'KEEP MOVING'} after completing.")
    print("    => contaminated AUROC "
          + ("UNDERSTATES the true failure signal: post-completion success rows look "
             "failure-like (frozen) and are labelled success."
             if freeze else
             "OVERSTATES the true failure signal: post-completion success rows keep "
             "moving and inflate the contrast against frozen failures."))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_rollouts": len(rids), "n_success_ever": int(succ.sum()), "T": T,
        "t_first_success": {"min": float(ts.min()), "max": float(ts.max()),
                            "median": float(np.median(ts)), "mean": float(ts.mean()),
                            "cdf": {str(k): v for k, v in cdf.items()}},
        "act_rep_peak_window": list(REP_PEAK),
        "frac_completed_by_peak_start": frac_lo,
        "frac_completed_by_peak_end": frac_hi,
        "pre_post": {r[0]: {"pre_mean": r[1], "post_mean": r[2],
                            "pre_rows": r[3], "post_rows": r[4]} for r in rows},
        "successes_freeze_after_completion": bool(freeze),
        "act_mag_distribution": dist,
        "corr_auroc_vs_completion_cdf_after_peak": corr,
    }, open(out, "w"), indent=1)
    print(f"\n[*] wrote {out}")


if __name__ == "__main__":
    main()
