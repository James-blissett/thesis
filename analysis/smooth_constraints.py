"""
smooth_constraints.py

Step 2b: centred rolling means of the three action-temporal series, appended to
constraints/all.parquet. Derived from the parquet alone -- step 1 is never
recomputed and /data/rollouts_v2 is never touched.

WHAT IS ADDED. For w in {5, 11, 21, 51}, three families:

    act_mag_w{w}[t]   from act_mag  (already in the parquet)
    act_rep_w{w}[t]   from act_rep  (act_mag <= 1e-6, the same definition
                                     constraint_auroc.add_derived uses)
    act_dir_w{w}[t]   from act_dir  (already in the parquet)

12 new series x 300 rollouts x 520 timesteps = 1,872,000 rows, layer = -1.

WHY act_rep IS NOT ALSO WRITTEN UNSMOOTHED. constraint_auroc.add_derived()
synthesises ("act_rep", -1) from act_mag at load time. Writing it here too would
give load_dense two rows per (series, rollout, t) slot and trip its bijection
assertion. The smoothed variants carry new names, so they do not collide, and the
unsmoothed baseline keeps coming from add_derived exactly as before -- the step-2
grid is unchanged.

THE FILTER, STATED ONCE. Centred, truncated, NaN-skipping:

    half = floor(w/2)                       window = [t-half, t+half] clipped to [0, T-1]
    min_periods = ceil(w/2)                 defined inputs required inside that window
    value = mean of the defined inputs, or NaN if fewer than min_periods are defined

No padding and no reflection: at the first and last `half` steps the window is simply
truncated, so it holds between half+1 and w inputs. Every rollout is filtered on its own
row, so no window ever spans a rollout boundary. All three families use the identical
NaN-skipping rule. It is only *load-bearing* for act_dir, which is undefined whenever
the arm did not move and whose defined count falls from ~240 rollouts early to ~96 at
t = 519, but applying it uniformly is what keeps act_mag_w{w}[0] and act_rep_w{w}[0]
NaN-propagating correctly off the single NaN at t = 0 rather than blanking the first
`half` steps outright.

Interaction with min_periods at the edges. act_mag[0] is NaN by construction, so for
w = 5 (half = 2, min_periods = 3) the window at t = 0 is [0, 2] -- three slots, two
defined -- and act_mag_w5[0] is NaN. t = 1 sees [0, 3], three defined, and is the first
defined output. For w = 51 the first defined output is likewise t = 1.

Rewrite, not append: parquet has no in-place append. The source file is copied row group
by row group into a temporary file, the new row groups are written after it, and the
result is moved over the original only once it has been re-opened and verified. The
original is kept as all.parquet.bak.

Usage:
    source env.sh
    python analysis/smooth_constraints.py --dry-run   # report, write nothing
    python analysis/smooth_constraints.py
"""

from __future__ import annotations

import argparse
import shutil
import time
from math import ceil, floor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PARQUET = Path("constraints/all.parquet")
WINDOWS = (5, 11, 21, 51)
ACT_REP_EPS = 1e-6          # identical to constraint_auroc.ACT_REP_EPS
BASE_SERIES = ("act_mag", "act_rep", "act_dir")
SEED = 0                    # nothing here is stochastic; recorded for the run log


def rolling_mean_centred(x: np.ndarray, w: int) -> np.ndarray:
    """(n_rollouts, T) -> centred, truncated, NaN-skipping rolling mean.

    Cumulative sums over the value array (NaN -> 0) and over the validity mask give
    every window's sum and defined-count in O(n*T) with no Python loop over t. Sums are
    accumulated in float64 across at most 520 terms, so the subtraction is exact enough
    that the count -- an integer cumsum -- is what actually decides definedness.
    """
    n, T = x.shape
    half = floor(w / 2)
    min_periods = ceil(w / 2)

    ok = np.isfinite(x)
    vals = np.where(ok, x, 0.0).astype(np.float64)

    cs = np.zeros((n, T + 1), dtype=np.float64)
    cm = np.zeros((n, T + 1), dtype=np.int64)
    np.cumsum(vals, axis=1, out=cs[:, 1:])
    np.cumsum(ok, axis=1, out=cm[:, 1:])

    t = np.arange(T)
    lo = np.maximum(t - half, 0)
    hi = np.minimum(t + half, T - 1) + 1        # exclusive

    tot = cs[:, hi] - cs[:, lo]
    cnt = cm[:, hi] - cm[:, lo]

    out = np.full((n, T), np.nan, dtype=np.float32)
    good = cnt >= min_periods
    np.divide(tot, np.maximum(cnt, 1), out=tot)
    out[good] = tot[good].astype(np.float32)
    return out


def dense_for(tbl: pa.Table, name: str, rids: list[str], T: int):
    """The (n_rollouts, T) matrix for one constraint, plus the row order it came in.

    Rows are placed by explicit (rollout, t) index rather than by trusting file order,
    and the mapping is asserted bijective -- the same contract load_dense enforces.
    """
    m = pc.equal(tbl.column("constraint_name"), name)
    sub = tbl.filter(m)
    ridx = pc.index_in(sub.column("rollout_id"),
                       value_set=pa.array(rids, type=pa.string())
                       ).to_numpy(zero_copy_only=False).astype(np.int64)
    t = sub.column("t").to_numpy(zero_copy_only=False).astype(np.int64)
    val = sub.column("value").to_numpy(zero_copy_only=False).astype(np.float32)

    flat = ridx * T + t
    expect = len(rids) * T
    if flat.size != expect or np.unique(flat).size != expect:
        raise SystemExit(f"{name}: {flat.size} rows / {np.unique(flat).size} distinct "
                         f"(rollout, t) slots, expected {expect} of each")
    M = np.empty(expect, dtype=np.float32)
    M[flat] = val
    return M.reshape(len(rids), T), sub, flat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=str(PARQUET))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.parquet)
    t0 = time.time()
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    print(f"[*] {src}: {pf.metadata.num_rows:,} rows, "
          f"{pf.metadata.num_row_groups} row groups")

    tbl = pq.read_table(src, filters=[("constraint_name", "in", ["act_mag", "act_dir"])])
    rids = sorted(set(pc.unique(tbl.column("rollout_id")).to_pylist()))
    T = int(pc.max(tbl.column("t")).as_py()) + 1
    print(f"[*] {len(rids)} rollouts x {T} timesteps read back for act_mag / act_dir "
          f"({time.time()-t0:.1f}s)")

    mag, tmpl, flat = dense_for(tbl, "act_mag", rids, T)
    dirm, _, _ = dense_for(tbl, "act_dir", rids, T)
    rep = np.where(np.isnan(mag), np.nan,
                   (mag <= ACT_REP_EPS).astype(np.float32)).astype(np.float32)
    base = {"act_mag": mag, "act_rep": rep, "act_dir": dirm}

    for k, v in base.items():
        nd = np.isfinite(v).sum(0)
        print(f"[*] base {k:<8} defined per t: min {nd.min():3d} max {nd.max():3d} "
              f"first {nd[0]:3d} last {nd[-1]:3d}")

    # Template row order: act_mag's rows carry every metadata column already, so each new
    # series is that table with constraint_name and value swapped out. `flat` maps the
    # dense grid back into exactly this row order.
    inv = np.empty(flat.size, dtype=np.int64)
    inv[flat] = np.arange(flat.size)
    n_rows = tmpl.num_rows

    new_tables = []
    for w in WINDOWS:
        for name in BASE_SERIES:
            sm = rolling_mean_centred(base[name], w)
            col = sm.reshape(-1)[inv].astype(np.float32)
            new_name = f"{name}_w{w}"
            arrs = []
            for f in schema:
                if f.name == "constraint_name":
                    arrs.append(pa.array([new_name] * n_rows, type=f.type))
                elif f.name == "value":
                    arrs.append(pa.array(col, type=f.type))
                else:
                    arrs.append(tmpl.column(f.name).combine_chunks())
            new_tables.append(pa.Table.from_arrays(arrs, schema=schema))
            nd = np.isfinite(sm).sum(0)
            fin = sm[np.isfinite(sm)]
            print(f"    {new_name:<14} defined t: {int(np.isfinite(sm).any(0).sum()):3d}/{T}"
                  f"  n_def first/mid/last {nd[0]:3d}/{nd[T//2]:3d}/{nd[-1]:3d}"
                  f"  value mean {fin.mean():.5f} sd {fin.std():.5f}")

    total_new = sum(t.num_rows for t in new_tables)
    print(f"[*] {len(new_tables)} new series, {total_new:,} rows "
          f"({time.time()-t0:.1f}s)")

    if args.dry_run:
        print("[*] --dry-run: nothing written")
        return

    tmp = src.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    for i in range(pf.metadata.num_row_groups):
        writer.write_table(pf.read_row_group(i))
    for t_new in new_tables:
        writer.write_table(t_new)
    writer.close()

    check = pq.ParquetFile(tmp)
    got = check.metadata.num_rows
    want = pf.metadata.num_rows + total_new
    if got != want:
        raise SystemExit(f"verification failed: {tmp} has {got:,} rows, expected {want:,}")
    names = pq.read_table(tmp, columns=["constraint_name"]).column("constraint_name")
    n_series = len(pc.unique(names).to_pylist())
    print(f"[ok] {tmp.name}: {got:,} rows, {n_series} distinct constraint names")

    bak = src.with_suffix(".parquet.bak")
    if not bak.exists():
        shutil.copy2(src, bak)
        print(f"[*] original preserved as {bak.name}")
    tmp.replace(src)
    print(f"[*] wrote {src} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
