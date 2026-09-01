"""
constraint_auroc.py

Step 2: rank each constraint series directly against outcome and test it against
permutation nulls. Nothing is trained -- these are raw scores ranked, so there is no
split, no seed variance and no classifier. Input is constraints/all.parquet only;
step 1 is never recomputed.

POLARITY, STATED ONCE. label 1 = FAILURE = NOT success under the active corpus label,
matching probe_layer.py. AUROC is computed with failure as the positive class, so
auroc_raw > 0.5 means "higher value of this constraint => more likely a failure".
`auroc` is the folded max(raw, 1-raw) and `sign` carries the direction: '+' when
higher-is-failure, '-' when the constraint is anti-correlated with failure.

THE STATISTIC. Mann-Whitney rank-sum AUROC, ranks assigned once per cell with average
ranks for ties:

    AUROC = (R1 - n1(n1+1)/2) / (n1 * n0)

where R1 is the summed rank of the failure rows. For the per-timestep schemes the
rollout label is broadcast, so a *rollout-level* label permutation never changes the
ranks -- only which rollouts count as failures. That is what makes 1000 nulls cheap:
per-rollout rank-sums and counts are computed once, and every permutation is then one
matrix product against the bank of permuted label vectors. Done naively (re-ranking
inside the permutation loop) this step would take hours instead of seconds.

Note the identity is exact under total ties: if every score is equal, every rank is
(N+1)/2 and the formula returns exactly 0.5. Degenerate cells need no special case.

FOUR PLACES THE CSV DEVIATES FROM THE STEP-2 BRIEF, all reconciled in the CSV rather
than in the plotter, as instructed:

  1. `layer` is -1 for non-layered series, not NaN. plot_constraint_auroc.py does
     int(r["layer"]) and would die on a NaN. -1 is what its reader already expects.
  2. Both `n_rollouts_used` (the brief) and `n_rollouts` (what the plotter reads) are
     emitted, with identical values.
  3. Scheme values are A / B / rollout_max / rollout_mean -- the strings the plotter
     facets on -- not the brief's shorthand RM / Rmu.
  4. xl_final covers l = 0..30, not 0..31. It is anchored to layer 31, so l=31 would be
     cos(h31, h31) = 0 by construction. Step 1 stored 31 columns; the parquet confirms.

THE B WINDOW IS DEFINED ON INTEGER t. The stored t_norm is float32, and at t = 364
(T = 520) it rounds to just under 0.7, so a `t_norm >= 0.7` filter silently drops that
timestep and yields 155 rather than 156. The window is therefore t >= ceil(0.7 * T).

SELECTION EFFECT. 133 series x 4 schemes x 3 corpora is 1596 dependent tests with
uncorrected nulls. Any maximum over that grid is biased upward and there is no
pre-registered cell here, unlike the layer-15 probe. Report peaks as post-hoc.

Usage:
    source env.sh
    python analysis/constraint_auroc.py --limit-rollouts 20   # subset check
    python analysis/constraint_auroc.py                       # full corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import rankdata

# Rollout-level label permutation, reused rather than reimplemented. Both helpers take
# (rids, base_labels, rid_to_task, rng) and return a rid -> label dict, i.e. they
# permute at the rollout level exactly as the broadcast requires.
from control_diagnostic import global_permutation, within_task_permutation

PARQUET = Path("constraints/all.parquet")
OUT_CSV = Path("results/constraint_auroc.csv")
N_PERM = 1000
SEED = 0
B_FRACTION = 0.7
MIN_DEFINED = 20          # a rollout enters a cell only with >= this many defined t
ACT_REP_EPS = 1e-6

SCHEMES = ["A", "B", "rollout_max", "rollout_mean"]
BASELINES = ["t", "t_norm"]
CORPORA = ["success_ever", "success_final", "success_ever_strict"]


def load_dense(path: Path, limit_rollouts: int | None):
    """all.parquet -> dense [n_series, n_rollouts, T] plus rollout metadata.

    Row order in the parquet is not relied upon; every row is placed by an explicit
    (series, rollout, t) index and the mapping is asserted bijective.
    """
    tbl = pq.read_table(path, columns=["rollout_id", "task_id", "outcome",
                                       "outcome_final", "outcome_group", "t",
                                       "T", "constraint_name", "layer", "value"])

    rid_vals = tbl.column("rollout_id")
    rids = sorted(set(pc.unique(rid_vals).to_pylist()))
    if limit_rollouts:
        rids = rids[:limit_rollouts]

    # index_in against an explicit value set keeps the mapping independent of whether
    # the column comes back dictionary-encoded, and yields null for rollouts not in the
    # subset -- which is exactly the --limit-rollouts filter.
    ridx = np.asarray(pc.index_in(rid_vals, value_set=pa.array(rids, type=pa.string()))
                      .to_numpy(zero_copy_only=False).astype(np.float64))
    keep = np.isfinite(ridx)

    cn = tbl.column("constraint_name")
    names = sorted(set(pc.unique(cn).to_pylist()))
    cidx = np.asarray(pc.index_in(cn, value_set=pa.array(names, type=pa.string()))
                      .to_numpy(zero_copy_only=False))
    layer = tbl.column("layer").to_numpy(zero_copy_only=False).astype(np.int64)
    t = tbl.column("t").to_numpy(zero_copy_only=False).astype(np.int64)
    val = tbl.column("value").to_numpy(zero_copy_only=False).astype(np.float32)

    ridx, cidx, layer, t, val = (a[keep] for a in (ridx, cidx, layer, t, val))
    ridx = ridx.astype(np.int64)

    T = int(np.max(t)) + 1
    skey = cidx * 64 + (layer + 1)
    uk = np.unique(skey)
    sidx = np.searchsorted(uk, skey)
    series = []
    for k in uk:
        series.append((names[int(k // 64)], int(k % 64) - 1))

    n_s, n_r = len(uk), len(rids)
    flat = (sidx * n_r + ridx) * T + t
    expect = n_s * n_r * T
    if flat.size != expect or np.unique(flat).size != expect:
        raise SystemExit(f"parquet does not fill the (series, rollout, t) grid: "
                         f"{flat.size} rows, {np.unique(flat).size} distinct slots, "
                         f"expected {expect}")
    M = np.empty(expect, dtype=np.float32)
    M[flat] = val
    M = M.reshape(n_s, n_r, T)

    # Rollout metadata: one row per rollout, taken from the first occurrence.
    first = np.zeros(n_r, dtype=np.int64)
    seen = np.zeros(n_r, dtype=bool)
    order = np.argsort(ridx, kind="stable")
    for i in order:
        r = ridx[i]
        if not seen[r]:
            first[r], seen[r] = i, True
    meta = {
        "rollout_id": rids,
        "task_id": tbl.column("task_id").to_numpy(zero_copy_only=False)[keep][first],
        "outcome": tbl.column("outcome").to_numpy(zero_copy_only=False)[keep][first],
        "outcome_final": tbl.column("outcome_final")
                            .to_numpy(zero_copy_only=False)[keep][first],
        "outcome_group": np.array(tbl.column("outcome_group").take(
            pa.array(np.where(keep)[0][first])).to_pylist()),
    }
    return M, series, meta, T


def add_derived(M, series, T):
    """act_rep (from act_mag) and the two clock baselines, as first-class series."""
    idx = {k: i for i, k in enumerate(series)}
    act_mag = M[idx[("act_mag", -1)]]
    act_rep = np.where(np.isnan(act_mag), np.nan,
                       (act_mag <= ACT_REP_EPS).astype(np.float32))

    n_r = M.shape[1]
    tvec = np.arange(T, dtype=np.float32)
    t_series = np.broadcast_to(tvec, (n_r, T)).copy()
    tn_series = t_series / float(T)

    M = np.concatenate([M, act_rep[None], t_series[None], tn_series[None]], axis=0)
    series = list(series) + [("act_rep", -1), ("t", -1), ("t_norm", -1)]
    return M, series


def perm_bank(rids, labels, rid_to_task, fn, n_perm, seed):
    """n_perm rollout-level label permutations as a (n_perm, n_rollouts) float matrix.

    Built once per corpus and reused across every series and scheme, so the nulls are
    paired: two constraints are compared against the same shuffles, not independent
    ones. The rollout SET differs between corpora (the strict slice drops 28), so the
    bank is regenerated per corpus from the same seed rather than shared across them.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, len(rids)), dtype=np.float64)
    for i in range(n_perm):
        m = fn(rids, labels, rid_to_task, rng)
        out[i] = [m[r] for r in rids]
    return out


def rank_stats(values, rollout_idx, n_rollouts):
    """rankdata once, then per-rollout rank-sums and counts."""
    r = rankdata(values)
    ranksum = np.bincount(rollout_idx, weights=r, minlength=n_rollouts)
    count = np.bincount(rollout_idx, minlength=n_rollouts).astype(np.float64)
    return ranksum, count


def auroc(ranksum, count, labels):
    """Failure-positive AUROC. `labels` is (n_rollouts,) or a (K, n_rollouts) bank."""
    n1 = labels @ count
    R1 = labels @ ranksum
    n0 = count.sum() - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        return (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fold(a):
    return np.maximum(a, 1.0 - a)


def cell(mat, scheme, t_lo, labels, banks, n_rollouts):
    """One (series, scheme) cell: observed statistic plus both nulls.

    Returns None when nothing survives the defined-timestep floor.
    """
    if scheme in ("A", "B"):
        w = mat[:, t_lo:] if scheme == "B" else mat
        ok = np.isfinite(w)
        n_def_per_rollout = ok.sum(1)
        use = n_def_per_rollout >= MIN_DEFINED
        if use.sum() < 2:
            return None
        rr, tt = np.nonzero(ok & use[:, None])
        values = w[rr, tt]
        rollout_idx = rr
    else:
        w = mat[:, t_lo:]
        ok = np.isfinite(w)
        n_def_per_rollout = ok.sum(1)
        use = n_def_per_rollout >= MIN_DEFINED
        if use.sum() < 2:
            return None
        with warnings.catch_warnings():
            # Rollouts with nothing defined in the window produce all-NaN slices; they
            # are dropped by `use` on the next line. Expected, not worth a warning.
            warnings.simplefilter("ignore", RuntimeWarning)
            agg = (np.nanmax(w, axis=1) if scheme == "rollout_max"
                   else np.nanmean(w, axis=1))
        rollout_idx = np.nonzero(use)[0]
        values = agg[rollout_idx]
        good = np.isfinite(values)
        rollout_idx, values = rollout_idx[good], values[good]
        if rollout_idx.size < 2:
            return None
        use = np.zeros(n_rollouts, dtype=bool)
        use[rollout_idx] = True

    ranksum, count = rank_stats(values, rollout_idx, n_rollouts)
    lab = labels * use                      # dropped rollouts contribute nothing
    if lab.sum() < 1 or (use.sum() - lab.sum()) < 1:
        return None

    raw = float(auroc(ranksum, count, lab.astype(np.float64)))
    obs = fold(raw)
    out = {
        "auroc_raw": raw, "auroc": obs,
        "sign": "+" if raw >= 0.5 else "-",
        "n_rollouts_used": int(use.sum()),
        "n_rollouts_dropped": int(n_rollouts - use.sum()),
        "n_defined": int(count.sum()),
        "n_distinct": int(np.unique(values).size),
    }
    for tag, bank in banks.items():
        nul = fold(auroc(ranksum, count, bank * use))
        nul = nul[np.isfinite(nul)]
        out[f"null_{tag}_mean"] = float(nul.mean())
        out[f"null_{tag}_p95"] = float(np.percentile(nul, 95))
        out[f"p_{tag}"] = float((1 + int((nul >= obs).sum())) / (nul.size + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=str(PARQUET))
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    ap.add_argument("--limit-rollouts", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    t0 = time.time()
    M, series, meta, T = load_dense(Path(args.parquet), args.limit_rollouts)
    M, series = add_derived(M, series, T)
    n_r = M.shape[1]
    t_lo = int(ceil(B_FRACTION * T))
    print(f"[*] {len(series)} series x {n_r} rollouts x {T} timesteps  "
          f"(B window t >= {t_lo}, {T - t_lo} timesteps)  loaded in {time.time()-t0:.1f}s")

    rids = list(meta["rollout_id"])
    rid_to_task = {r: int(t) for r, t in zip(rids, meta["task_id"])}
    rows: list[dict] = []
    dropped: list[dict] = []

    for corpus in CORPORA:
        if corpus == "success_final":
            lab_full = meta["outcome_final"].astype(np.float64)
            sel = np.ones(n_r, dtype=bool)
        elif corpus == "success_ever_strict":
            lab_full = meta["outcome"].astype(np.float64)
            sel = meta["outcome_group"] != "success_lost"
        else:
            lab_full = meta["outcome"].astype(np.float64)
            sel = np.ones(n_r, dtype=bool)

        sub = [r for r, s in zip(rids, sel) if s]
        sub_lab = lab_full[sel]
        banks = {}
        for tag, fn in (("global", global_permutation), ("task", within_task_permutation)):
            b = perm_bank(sub, sub_lab, rid_to_task, fn, args.n_perm, SEED)
            full = np.zeros((args.n_perm, n_r), dtype=np.float64)
            full[:, np.nonzero(sel)[0]] = b
            banks[tag] = full
        labels = lab_full * sel

        for (name, layer), mat in zip(series, M):
            for scheme in SCHEMES:
                res = cell(mat * np.where(sel, 1.0, np.nan)[:, None],
                           scheme, t_lo, labels, banks, n_r)
                if res is None:
                    dropped.append({"constraint": name, "layer": layer,
                                    "scheme": scheme, "corpus": corpus,
                                    "reason": "fewer than 2 rollouts cleared the "
                                              "defined-timestep floor"})
                    continue
                degenerate = bool(name in BASELINES or res.pop("n_distinct") < 2)
                rows.append({
                    "constraint": name, "layer": layer, "scheme": scheme,
                    "corpus": corpus,
                    "n_rollouts_used": res["n_rollouts_used"],
                    "n_rollouts": res["n_rollouts_used"],   # plotter's column name
                    "n_rollouts_dropped": res["n_rollouts_dropped"],
                    "n_defined": res["n_defined"],
                    "auroc_raw": round(res["auroc_raw"], 6),
                    "auroc": round(res["auroc"], 6), "sign": res["sign"],
                    "null_global_mean": round(res["null_global_mean"], 6),
                    "null_global_p95": round(res["null_global_p95"], 6),
                    "p_global": round(res["p_global"], 6),
                    "null_task_mean": round(res["null_task_mean"], 6),
                    "null_task_p95": round(res["null_task_p95"], 6),
                    "p_task": round(res["p_task"], 6),
                    "degenerate_flag": degenerate,
                })
        print(f"[*] {corpus}: {len(rows)} rows cumulative  ({time.time()-t0:.1f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- Sanity assertions: fail loudly ------------------------------------------
    by = {(r["constraint"], r["layer"], r["scheme"], r["corpus"]): r for r in rows}
    for corpus in CORPORA:
        r = by.get(("t", -1, "A", corpus))
        if r is None:
            raise SystemExit(f"ASSERTION: no Scheme-A row for baseline t, {corpus}")
        if abs(r["auroc_raw"] - 0.5) > 1e-9:
            raise SystemExit(f"ASSERTION FAILED: Scheme-A AUROC for t is "
                             f"{r['auroc_raw']!r}, not 0.5 ({corpus}) -- labelling bug")
    print(f"[ok] Scheme-A baseline t == 0.5 to 1e-9 in all {len(CORPORA)} corpora")

    # act_rep under rollout_mean is the repeat rate; task03_trial28 emits 5 distinct
    # action rows in 520 steps, so it must sit at or beside the top of that ranking.
    STUCK = "task03_trial28"
    if STUCK in rids:
        sidx = {k: i for i, k in enumerate(series)}[("act_rep", -1)]
        i = rids.index(STUCK)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full_rate = np.nanmean(M[sidx], axis=1)      # whole rollout
            win_rate = np.nanmean(M[sidx][:, t_lo:], axis=1)   # B window
        # Two checks, because rank and value answer different questions here. Over the
        # whole rollout this is the single most stuck rollout in the corpus, so rank is
        # exact and rank 1 is the right test. Inside the B window it is NOT rank 1: 21
        # rollouts sit at exactly 1.0, perfectly frozen for the whole late window, while
        # this one still moves twice. Ranking against a 21-way tie at the ceiling is
        # meaningless, so the window check tests the value instead.
        rank_full = int((full_rate > full_rate[i]).sum()) + 1
        if rank_full != 1:
            raise SystemExit(f"ASSERTION FAILED: {STUCK} ranks {rank_full} on whole-"
                             f"rollout act_rep ({full_rate[i]:.4f}); expected rank 1")
        if win_rate[i] < 0.95:
            raise SystemExit(f"ASSERTION FAILED: {STUCK} B-window act_rep is "
                             f"{win_rate[i]:.4f}, expected >= 0.95")
        n_ceiling = int((win_rate >= win_rate[i]).sum())
        print(f"[ok] {STUCK}: whole-rollout act_rep {full_rate[i]:.4f} ranks 1 of "
              f"{len(rids)}; B-window {win_rate[i]:.4f} (>= 0.95), with {n_ceiling} "
              f"rollouts at or above it")
    else:
        print(f"[--] {STUCK} not in this subset; its act_rep assertion is not exercised")

    n_series, n_deg = len(series), sum(1 for r in rows if r["degenerate_flag"])
    expect = n_series * len(SCHEMES) * len(CORPORA)
    if len(rows) + len(dropped) != expect:
        raise SystemExit(f"ASSERTION FAILED: {len(rows)} rows + {len(dropped)} dropped "
                         f"!= full grid {expect}")
    print(f"[ok] rows {len(rows)} + dropped {len(dropped)} == grid {expect} "
          f"({n_deg} flagged degenerate)")
    print(f"[*] wrote {out} in {time.time()-t0:.1f}s")

    json.dump({"n_rows": len(rows), "n_series": n_series, "grid": expect,
               "dropped_cells": dropped,
               "t_lo": t_lo, "n_perm": args.n_perm, "seed": SEED,
               "min_defined": MIN_DEFINED,
               "polarity": "label 1 = failure = NOT success under the active corpus"},
              open(out.with_suffix(".meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
