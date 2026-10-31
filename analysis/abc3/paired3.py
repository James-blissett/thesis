"""
paired3.py

Paired comparisons on T0 for the deep-MLP run (deep_mlp_handoff.md section 3), with
abc2's paired bootstrap reused unchanged (analysis/abc2/bootstrap2.py: load_side,
make_draws, side_boot, compare, check_identity; RNG_SEED 20261011, N_BOOT 1000,
resampling within cell within seed). Metrics: bootstrap2.METRICS = m1_per_task, bal_acc,
tpr (alarm at alpha 0.1, cut variant, each seed's stored conformal band held fixed: the
alarm intervals understate the uncertainty). Seeds 0, 1, 2.

Sides (one per pool x detector; detector names as analysis/abc3/FORMATS.md):
    A-d2 ... C-d12   results/abc3/scores/<pool>_seed<s>_deep.npz, key = the name
    A-linear/C-linear results/abc2/scores/<pool>_seed<s>.npz, key A / C
Comparisons (x - y), 26 in all, x 3 metrics:
    vs_linear  (12)  <inp>-d<depth> - <inp>-linear, same pool       (pool D0, D2; depth 2, 6, 12)
    vs_depth2  (8)   <inp>-d<depth> - <inp>-d2, same pool           (depth 6, 12)
    D2_vs_D0   (6)   D2 <inp>-d<depth> - D0 <inp>-d<depth>          (depth 2, 6, 12)
`improvement` only if the 95% percentile interval lies entirely above zero;
`significant_worsening` if entirely below; else no significant change (bootstrap2.verdict).

All sides share ONE set of resamples: bootstrap2.make_draws on the first side's T0 test
set, after checking (bootstrap2.check_same_test_set) that every side has the same T0
rollouts, labels, kept lengths and cells. The draws depend only on the test set, seeds,
n_boot and rng seed, so they are the same resamples as abc2's paired_T0.csv and
mlp_paired_T0.csv. Each side is bootstrapped once (side_boot) and reused.

The side's SeedEval.key is set to its FORMATS name after loading (dataclasses.replace),
so bootstrap2.compare's labels read e.g. "D0:C-d6 - D0:C-linear" (detector_x C-d6,
detector_y C-linear) and, for D2_vs_D0, "D2 - D0" (detector C-d6).

Identity check: every side's identity-resample values against results/abc3/final_m1.csv,
cp_curves.csv (alpha 0.1 off-grid) and final_table.csv (eval3.py outputs, abc2 column
format) under the side's FORMATS name, |diff| <= 1e-6 (bootstrap2.check_identity). A side
with no rows to compare fails. Exit 1 if any side fails (unless --no-strict).

A deep detector group that eval3 reports as failed (no keys in the npz) is skipped with
its comparisons; skipped comparisons are listed in paired_T0_check.json and printed.

Outputs (results/abc3/, format in FORMATS.md "paired3 outputs"):
    paired_T0.csv           bootstrap2 paired csv columns + family, input, depth, pool
    paired_T0_per_seed.csv  bootstrap2 per-seed columns + family, input, depth, pool
    paired_T0_check.json    passed, tolerance, rng_seed, n_boot, max_abs_diff, sides, ...

--selftest: the same machinery on abc2's MLP comparison (results/abc2/scores/<pool>_seed<s>
_mlp.npz A-MLP / C-MLP vs <pool>_seed<s>.npz A / C, D0 and D2), written to
results/abc3/paired/selftest/, and compared with results/abc2/mlp_paired_T0.csv,
mlp_paired_T0_per_seed.csv and mlp_paired_T0_check.json (every bootstrap2 column must be
identical as written; identity-check values equal); plus abc2's linear D2 - D0 for A and C
(same-name sides, the D2_vs_D0 path) against results/abc2/paired_T0*.csv, with the identity
check against abc2's final_table.csv. Exit 1 on any mismatch.

Usage (repo root, CPU only, well under a minute):
    source env.sh
    python analysis/abc3/paired3.py              # real run
    python analysis/abc3/paired3.py --selftest   # reproduce abc2's mlp_paired_T0.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "abc2"))
import bootstrap2 as b2  # noqa: E402
from pipe2_common import read_csv, write_csv  # noqa: E402

ABC2_RESULTS = REPO / "results" / "abc2"
RESULTS = REPO / "results" / "abc3"
POOLS = ("D0", "D2")
SEEDS = (0, 1, 2)
INPUTS = ("A", "C")
DEPTHS = (2, 6, 12)
TEST_SET = "T0"
EXTRA = ("family", "input", "depth", "pool")


@dataclass
class SideSpec:
    pool: str
    name: str              # detector name in the outputs (FORMATS.md)
    files: list            # one npz per seed, in SEEDS order
    key: str               # detector key inside the npz


@dataclass
class Comp:
    family: str
    input: str
    depth: object
    pool: str              # "" for D2_vs_D0
    x: tuple               # (pool, name)
    y: tuple


@dataclass
class Run:
    specs: dict = field(default_factory=dict)       # (pool, name) -> SideSpec
    comps: list = field(default_factory=list)


def has_key(path: Path, key: str) -> bool:
    if not Path(path).exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        return f"{key}_m1" in z.files


# --- Plans --------------------------------------------------------------------------------
def real_plan(seeds) -> Run:
    run = Run()
    for pool in POOLS:
        for inp in INPUTS:
            run.specs[(pool, f"{inp}-linear")] = SideSpec(
                pool, f"{inp}-linear", [ABC2_RESULTS / "scores" / f"{pool}_seed{s}.npz" for s in seeds], inp)
            for d in DEPTHS:
                n = f"{inp}-d{d}"
                run.specs[(pool, n)] = SideSpec(
                    pool, n, [RESULTS / "scores" / f"{pool}_seed{s}_deep.npz" for s in seeds], n)
    for pool in POOLS:
        for inp in INPUTS:
            for d in DEPTHS:
                run.comps.append(Comp("vs_linear", inp, d, pool, (pool, f"{inp}-d{d}"), (pool, f"{inp}-linear")))
    for pool in POOLS:
        for inp in INPUTS:
            for d in DEPTHS[1:]:
                run.comps.append(Comp("vs_depth2", inp, d, pool, (pool, f"{inp}-d{d}"), (pool, f"{inp}-d2")))
    for inp in INPUTS:
        for d in DEPTHS:
            run.comps.append(Comp("D2_vs_D0", inp, d, "", ("D2", f"{inp}-d{d}"), ("D0", f"{inp}-d{d}")))
    return run


def selftest_plan(seeds) -> Run:
    """abc2 stage10_mlp.paired(): per pool, A-MLP - A, C-MLP - C (that order)."""
    run = Run()
    for pool in POOLS:
        for inp in INPUTS:
            mlp, lin = f"{inp}-MLP", inp
            run.specs[(pool, mlp)] = SideSpec(
                pool, mlp, [ABC2_RESULTS / "scores" / f"{pool}_seed{s}_mlp.npz" for s in seeds], mlp)
            run.specs[(pool, lin)] = SideSpec(
                pool, lin, [ABC2_RESULTS / "scores" / f"{pool}_seed{s}.npz" for s in seeds], lin)
            run.comps.append(Comp("selftest_mlp", inp, 2, pool, (pool, mlp), (pool, lin)))
    return run


# --- Execution ----------------------------------------------------------------------------
def execute(run: Run, *, table_dir: Path, check_table: Path, n_boot: int, rng_seed: int,
            log=print) -> dict:
    t0 = time.time()
    missing = {}
    for sk, sp in run.specs.items():
        bad = [str(f) for f in sp.files if not has_key(f, sp.key)]
        if bad:
            missing[sk] = bad
    for sk, bad in missing.items():
        log(f"[skip] side {sk[0]} {sk[1]}: key {run.specs[sk].key!r} missing in {bad}")
    comps = [c for c in run.comps if c.x not in missing and c.y not in missing]
    skipped = [c for c in run.comps if c not in comps]
    used = list(dict.fromkeys([c.x for c in comps] + [c.y for c in comps]))
    if not comps:
        raise SystemExit("no comparison has both sides available")

    sides = {}
    for sk in used:
        sp = run.specs[sk]
        loaded = b2.load_side(sp.files, sp.key, TEST_SET)
        for s in loaded:
            if s.cond != sp.pool:
                raise SystemExit(f"{s.path}: cond {s.cond!r} != expected pool {sp.pool!r}")
        sides[sk] = [dataclasses.replace(s, key=sp.name) for s in loaded]
    first = sides[used[0]]
    for sk, s in sides.items():
        b2.check_same_test_set(first, s)
    draws = b2.make_draws(first, n_boot, rng_seed)
    t_load = time.time() - t0
    boots = {sk: b2.side_boot(s, draws) for sk, s in sides.items()}

    rows, per_seed = [], []
    for c in comps:
        r = b2.compare(sides[c.x], sides[c.y], draws, test_set=TEST_SET, n_boot=n_boot,
                       rng_seed=rng_seed, x_boot=boots[c.x], y_boot=boots[c.y])
        extra = {"family": c.family, "input": c.input, "depth": c.depth, "pool": c.pool}
        rows += [{**q, **extra} for q in r["rows"]]
        per_seed += [{**q, **extra} for q in r["per_seed"]]

    checks = [b2.check_identity(s, TEST_SET, table_dir, check_table, detector_name=run.specs[sk].name)
              for sk, s in sides.items()]
    passed = all(c["passed"] for c in checks)
    return {"rows": rows, "per_seed": per_seed, "checks": checks, "passed": passed,
            "skipped": [f"{c.family}: {c.x[0]}:{c.x[1]} - {c.y[0]}:{c.y[1]}" for c in skipped],
            "n_comparisons": len(comps), "n_sides": len(sides), "seconds": time.time() - t0,
            "seconds_load": t_load,
            "n_test": [len(s.rids) for s in first], "n_cells": [len(d) for d in draws]}


def write_outputs(res: dict, out_dir: Path, name: str, *, n_boot: int, rng_seed: int,
                  table_dir: Path, check_table: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / name, res["rows"])
    write_csv(out_dir / name.replace(".csv", "_per_seed.csv"), res["per_seed"])
    (out_dir / name.replace(".csv", "_check.json")).write_text(json.dumps(
        {"test_set": TEST_SET, "passed": res["passed"], "tolerance": b2.CHECK_TOL, "rng_seed": rng_seed,
         "n_boot": n_boot, "table_dir": str(table_dir), "check_table": str(check_table),
         "max_abs_diff": max((c["max_abs_diff"] or 0) for c in res["checks"]),
         "n_comparisons": res["n_comparisons"], "skipped_comparisons": res["skipped"],
         "draws": "one set, bootstrap2.make_draws on the shared T0 test set",
         "n_test_per_seed": res["n_test"], "n_cells_per_seed": res["n_cells"],
         "sides": res["checks"]}, indent=1, default=float))
    return out_dir / name


def summarize(res: dict, log=print) -> None:
    for c in res["checks"]:
        log(f"  identity {c['cond']}/{c['detector']}: n={c['n_compared']} max|d|={c['max_abs_diff']} "
            f"{'ok' if c['passed'] else 'FAILED'}{' (' + c['note'] + ')' if c.get('note') else ''}")
    log(f"  {'family':10s} {'comparison':26s} {'metric':12s} {'point':>7s} {'mean':>7s}  "
        f"{'95% CI':18s} verdict")
    for r in res["rows"]:
        lab = f"{r['cond_x']}:{r['detector_x']} - {r['cond_y']}:{r['detector_y']}"
        log(f"  {r['family']:10s} {lab:26s} {r['metric']:12s} {r['diff_point']:+.3f} "
            f"{r['diff_mean']:+.3f}  [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]  {r['verdict']}")


# --- Selftest against abc2's mlp_paired_T0 -------------------------------------------------
def _cmp_csv(mine: Path, ref: Path, label: str, keep=None) -> list[str]:
    a, b = read_csv(mine), read_csv(ref)
    if keep is not None:
        b = [r for r in b if keep(r)]
    errs = []
    if len(a) != len(b):
        return [f"{label}: {len(a)} rows vs {len(b)} in {ref}"]
    cols = list(b[0].keys())
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in cols:
            if ra.get(k) != rb[k]:
                errs.append(f"{label} row {i} {k}: {ra.get(k)!r} != {rb[k]!r}")
    return errs


def selftest(args) -> int:
    seeds = tuple(args.seeds)
    out = RESULTS / "paired" / "selftest"
    check_table = ABC2_RESULTS / "mlp_final_table.csv"
    res = execute(selftest_plan(seeds), table_dir=ABC2_RESULTS, check_table=check_table,
                  n_boot=args.n_boot, rng_seed=args.rng_seed)
    p = write_outputs(res, out, "mlp_paired_T0.csv", n_boot=args.n_boot, rng_seed=args.rng_seed,
                      table_dir=ABC2_RESULTS, check_table=check_table)
    summarize(res)
    errs = _cmp_csv(p, ABC2_RESULTS / "mlp_paired_T0.csv", "paired")
    errs += _cmp_csv(out / "mlp_paired_T0_per_seed.csv", ABC2_RESULTS / "mlp_paired_T0_per_seed.csv", "per_seed")
    ref = json.loads((ABC2_RESULTS / "mlp_paired_T0_check.json").read_text())
    refs = {(s["cond"], s["detector"]): s for s in ref["sides"]}
    n_vals = 0
    for c in res["checks"]:
        r = refs.get((c["cond"], c["detector"]))
        if r is None:
            errs.append(f"check: side {c['cond']}/{c['detector']} not in abc2's check json")
            continue
        got = {q["what"]: (q["got"], q["ref"]) for q in c["comparisons"]}
        want = {q["what"]: (q["got"], q["ref"]) for q in r["comparisons"]}
        if set(got) != set(want):
            errs.append(f"check {c['cond']}/{c['detector']}: compared {sorted(got)} vs {sorted(want)}")
        for k in set(got) & set(want):
            n_vals += 1
            if not np.allclose(got[k], want[k], rtol=0, atol=1e-12, equal_nan=True):
                errs.append(f"check {c['cond']}/{c['detector']} {k}: {got[k]} != {want[k]}")
    if len(res["checks"]) != len(refs):
        errs.append(f"check: {len(res['checks'])} sides vs {len(refs)} in abc2")
    if not res["passed"]:
        errs.append("identity check failed")
    print(f"[selftest] {res['n_comparisons']} comparisons, {res['n_sides']} sides, {res['seconds']:.1f} s -> {p}")

    # Part 2: same-name pool comparison (the D2_vs_D0 path), abc2's linear D2 - D0 for A, C,
    # against results/abc2/paired_T0.csv (identity check against abc2's final_table.csv).
    run2 = Run()
    for inp in INPUTS:
        for pool in ("D2", "D0"):
            run2.specs[(pool, inp)] = SideSpec(
                pool, inp, [ABC2_RESULTS / "scores" / f"{pool}_seed{s}.npz" for s in seeds], inp)
        run2.comps.append(Comp("selftest_pool", inp, 0, "", ("D2", inp), ("D0", inp)))
    ct2 = ABC2_RESULTS / "final_table.csv"
    res2 = execute(run2, table_dir=ABC2_RESULTS, check_table=ct2, n_boot=args.n_boot, rng_seed=args.rng_seed)
    p2 = write_outputs(res2, out, "linear_D2_D0_T0.csv", n_boot=args.n_boot, rng_seed=args.rng_seed,
                       table_dir=ABC2_RESULTS, check_table=ct2)
    summarize(res2)
    keep = lambda r: r["comparison"] == "D2 - D0" and r["detector"] in INPUTS  # noqa: E731
    errs += _cmp_csv(p2, ABC2_RESULTS / "paired_T0.csv", "linear D2-D0", keep)
    errs += _cmp_csv(out / "linear_D2_D0_T0_per_seed.csv", ABC2_RESULTS / "paired_T0_per_seed.csv",
                     "linear D2-D0 per_seed", keep)
    if not res2["passed"] or any(c["n_compared"] != 15 for c in res2["checks"]):
        errs.append(f"part 2 identity check: passed={res2['passed']}, n_compared="
                    f"{[c['n_compared'] for c in res2['checks']]} (expected 15 each)")
    print(f"[selftest] part 2: {res2['n_comparisons']} comparisons -> {p2}")
    if errs:
        print(f"[selftest] FAILED ({len(errs)} mismatches):")
        for e in errs[:40]:
            print("   ", e)
        return 1
    print(f"[selftest] PASSED: {len(read_csv(p))} paired rows and the per-seed rows identical to "
          f"results/abc2/mlp_paired_T0*.csv; {n_vals} identity-check values equal abc2's "
          f"(max |diff| vs tables {max(c['max_abs_diff'] for c in res['checks']):.2g}); "
          f"linear D2 - D0 (A, C) rows identical to results/abc2/paired_T0*.csv, identity check "
          f"vs final_table.csv passed (max |diff| {max(c['max_abs_diff'] for c in res2['checks']):.2g})")
    return 0


# --- CLI ----------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true", help="reproduce abc2's mlp_paired_T0.csv (scratch output)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--table-dir", type=Path, default=RESULTS, help="final_m1.csv / cp_curves.csv for the check")
    ap.add_argument("--check-table", type=Path, default=None, help="default <table-dir>/final_table.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--n-boot", type=int, default=b2.N_BOOT)
    ap.add_argument("--rng-seed", type=int, default=b2.RNG_SEED)
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                    help="exit 1 if the identity check fails (default on)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args)

    check_table = args.check_table or args.table_dir / "final_table.csv"
    for f in (args.table_dir / "final_m1.csv", args.table_dir / "cp_curves.csv", check_table):
        if not f.exists():
            print(f"[!] missing {f} (eval3.py output); the identity check cannot run")
            return 1
    res = execute(real_plan(tuple(args.seeds)), table_dir=args.table_dir, check_table=check_table,
                  n_boot=args.n_boot, rng_seed=args.rng_seed)
    p = write_outputs(res, args.out_dir, f"paired_{TEST_SET}.csv", n_boot=args.n_boot,
                      rng_seed=args.rng_seed, table_dir=args.table_dir, check_table=check_table)
    print(f"[{TEST_SET}] {res['n_comparisons']} comparisons x {len(b2.METRICS)} metrics, {args.n_boot} reps, "
          f"{res['n_sides']} sides, one set of draws (n_test per seed {res['n_test']}): "
          f"{res['seconds']:.1f} s (loading {res['seconds_load']:.1f} s) -> {p}")
    for s in res["skipped"]:
        print(f"[skip] {s}")
    summarize(res)
    mx = max((c["max_abs_diff"] or 0) for c in res["checks"])
    print(f"[{TEST_SET}] identity check {'PASSED' if res['passed'] else 'FAILED'} "
          f"({len(res['checks'])} sides, max |diff| {mx:.2g}, tolerance {b2.CHECK_TOL:g})")
    if not res["passed"]:
        for c in res["checks"]:
            if not c["passed"]:
                bad = [q for q in c["comparisons"] if q["abs_diff"] > b2.CHECK_TOL]
                why = c.get("note") or (f"{len(bad)} values off, e.g. {bad[0]}" if bad else "nothing compared")
                print(f"[!] {c['cond']}/{c['detector']}: {why}")
        print("[!] IDENTITY CHECK FAILED: the recomputation does not reproduce eval3's tables")
        return 1 if args.strict else 0
    if res["skipped"]:
        print(f"[!] {len(res['skipped'])} comparisons skipped (failed groups; see paired_T0_check.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
