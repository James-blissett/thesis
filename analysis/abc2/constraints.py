"""
constraints.py

Step 6 constraints for the v3 rollouts (more_data_collection.md, handoff v1, Step 6).
For every v3 rollout, the constraint series the A/B/C pipeline uses, computed exactly as
analysis/compute_constraints.py computed constraints/<rid>.npz for the v2 corpus:

    constraints_v3/<rollout_id>.npz   float32, all (520, ...):
        act_mag (520,)  act_dir (520,)  grip_flip (520,)  emb_temp (520, 32)
        xl_adj (520, 31)  xl_final (520, 31)  xl_spread (520,)

Same keys, shapes, dtype and NaN conventions as constraints/<rid>.npz (NaN at t = 0 for
the temporal series; act_dir also NaN at t = 1 and wherever a delta norm is ~0).
`xl_final_hN` and every `*_nosink` key are not stored (Stages 3-6 never used them;
xl_final_hN needs the single-position h[N], which v3 does not store). `act_rep` is
derived downstream (prep.load_series), as for v2.

Computation path. The frozen functions of compute_constraints are called unchanged:
normalised_actions -> action_constraints for the action series, and hidden_constraints
(-> cos_dist_pair) for the hidden-state series. v3's hidden.pt is (520, 33, 4096) fp16,
the mean over the 7 decision states that capture_v2 stored as position 0 of a P = 3
rollout; pooled_and_hN turns that position into fp32 with `.float()`, and so does this
script. hidden_constraints also wants h[N] and the norm weight g, which only feed
xl_final_hN; it is given the pooled state and g = 1 there and that output is discarded,
so every stored series comes out of the identical code.

--parity recomputes the same series for five v2 rollouts (two P = 7, three P = 3) from
/data/rollouts_v2 through pooled_and_hN and the same per-rollout function, into a scratch
dir, and compares them with constraints/<rid>.npz (max |diff| <= 1e-4, NaN in exactly the
same places, per key). Writes results/abc2/constraints_parity.json; exits 1 on failure.

I/O-bound (one ~140 MB hidden.pt per rollout), so single process. Resumable: an output
that exists and passes the key / shape / dtype check is skipped. Each file is written to
a temp name and renamed. Only v3 rollouts whose meta.json says complete: true are read.

Usage (from the repo root):
    source env.sh
    python analysis/abc2/constraints.py --parity          # self-check against constraints/
    python analysis/abc2/constraints.py                   # v3 rows of corpus_v3_index.json
    python analysis/abc2/constraints.py --all-finished    # every finished v3 rollout (no index needed)
    python analysis/abc2/constraints.py --rids ori2_task00_trial00 obj_task00_trial00
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data2 import CONSTRAINTS_V2, CONSTRAINTS_V3, INDEX_V3, ROLLOUTS_V2, ROLLOUTS_V3, v3_entries  # noqa: E402

sys.path.append(str(HERE.parent))                     # analysis/, for frozen modules
from compute_constraints import (D_MODEL, N_RAW, N_STORED, action_constraints,  # noqa: E402
                                 hidden_constraints, normalised_actions, pooled_and_hN)

T = 520
KEYS = {"act_mag": (T,), "act_dir": (T,), "grip_flip": (T,), "emb_temp": (T, N_RAW),
        "xl_adj": (T, N_RAW - 1), "xl_final": (T, N_RAW - 1), "xl_spread": (T,)}
PARITY_RIDS = ("task00_trial00", "task01_trial02",                    # P = 7
               "task03_trial05", "task05_trial27", "task09_trial29")  # P = 3
PARITY_TOL = 1e-4
PARITY_JSON = Path("results/abc2/constraints_parity.json")
PARITY_SCRATCH = Path("/data/tmp/abc2_dev/constraints/parity")
N_THREADS = 2


# --- Per-rollout computation -------------------------------------------------------------
def series_from(pooled: torch.Tensor, acts) -> dict:
    """The seven stored series from a pooled (520, 33, 4096) fp32 state and actions.npz,
    via compute_constraints' own functions (see the module docstring)."""
    pe = acts["policy_step_env_t"]
    series = action_constraints(normalised_actions(acts["action_token_ids"]),
                                acts["executed"][pe][:, 6])
    hid = hidden_constraints(pooled, pooled, torch.ones(D_MODEL))  # hN, g feed xl_final_hN only
    series.update({k: hid[k] for k in ("emb_temp", "xl_adj", "xl_final", "xl_spread")})
    return {k: series[k] for k in KEYS}


def check_actions(rid: str, acts, n_wait: int) -> None:
    pe = acts["policy_step_env_t"]
    if not np.array_equal(pe, np.arange(n_wait, n_wait + T)):
        raise SystemExit(f"{rid}: policy_step_env_t is not contiguous from the wait offset")
    if acts["action_token_ids"].shape != (T, 7):
        raise SystemExit(f"{rid}: action_token_ids is {acts['action_token_ids'].shape}")


def compute_v3(rid: str, root: Path = ROLLOUTS_V3) -> dict:
    d = root / rid
    meta = json.loads((d / "meta.json").read_text())
    if not meta.get("complete") or d.name.startswith("."):
        raise SystemExit(f"{rid}: not a finished rollout")
    if int(meta["T_total"]) != T:
        raise SystemExit(f"{rid}: T_total {meta['T_total']}")
    acts = np.load(d / "actions.npz")
    check_actions(rid, acts, int(meta["num_steps_wait"]))
    x = torch.load(d / "hidden.pt", map_location="cpu")
    if tuple(x.shape) != (T, N_STORED, D_MODEL) or x.dtype != torch.float16:
        raise SystemExit(f"{rid}: hidden.pt is {tuple(x.shape)} {x.dtype}")
    pooled = x.float()                 # as pooled_and_hN does for the P = 3 pooled position
    del x
    return series_from(pooled, acts)


def compute_v2(rid: str, root: Path = ROLLOUTS_V2) -> dict:
    """The same series for a v2 rollout, pooled via pooled_and_hN (parity only)."""
    d = root / rid
    man = json.loads((d / "manifest.json").read_text())
    P = int(man["n_positions"])
    acts = np.load(d / "actions.npz")
    check_actions(rid, acts, int(man["num_steps_wait"]))
    x = torch.load(d / "hidden.pt", map_location="cpu")
    if tuple(x.shape) != (T, N_STORED, P, D_MODEL):
        raise SystemExit(f"{rid}: hidden.pt is {tuple(x.shape)}")
    pooled, _ = pooled_and_hN(x, P)
    del x
    return series_from(pooled, acts)


# --- Output ------------------------------------------------------------------------------
def output_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        z = np.load(path)
        if set(z.files) != set(KEYS):
            return False
        return all(z[k].shape == s and z[k].dtype == np.float32 for k, s in KEYS.items())
    except Exception:
        return False


def write_atomic(path: Path, series: dict) -> None:
    for k, s in KEYS.items():
        if series[k].shape != s or series[k].dtype != np.float32:
            raise SystemExit(f"{path.stem}:{k} is {series[k].shape} {series[k].dtype}")
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **series)
    os.replace(tmp, path)


def guard_out_dir(out_dir: Path) -> None:
    if out_dir.resolve() == CONSTRAINTS_V2.resolve():
        raise SystemExit("refusing to write into constraints/ (v2, frozen)")


# --- Modes -------------------------------------------------------------------------------
def run(rids: list[str], out_dir: Path, force: bool) -> None:
    guard_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rids if force or not output_ok(out_dir / f"{r}.npz")]
    print(f"[*] {len(rids)} v3 rollouts; {len(rids) - len(todo)} already done; "
          f"{len(todo)} to compute -> {out_dir}", flush=True)
    t0 = time.time()
    for k, rid in enumerate(todo, 1):
        write_atomic(out_dir / f"{rid}.npz", compute_v3(rid))
        if k % 25 == 0 or k == len(todo) or k <= 2:
            el = time.time() - t0
            print(f"[{k:4d}/{len(todo)}] {el / 60:5.1f} min  {el / k:.2f} s/rollout  "
                  f"eta {(len(todo) - k) * el / k / 60:5.1f} min", flush=True)
    bad = [r for r in rids if not output_ok(out_dir / f"{r}.npz")]
    if bad:
        raise SystemExit(f"{len(bad)} outputs fail the key/shape check: {bad[:5]}")
    print(f"[*] {len(rids)}/{len(rids)} constraint files present and well-formed")


def compare(new: dict, ref) -> dict:
    out = {}
    for k in KEYS:
        a, b = new[k], ref[k]
        if a.shape != b.shape:
            out[k] = {"max_abs_diff": None, "nan_mismatch": None, "pass": False,
                      "note": f"shape {a.shape} vs {b.shape}"}
            continue
        na, nb = np.isnan(a), np.isnan(b)
        mism = int((na != nb).sum())
        both = ~na & ~nb
        diff = float(np.abs(a[both].astype(np.float64) - b[both]).max()) if both.any() else 0.0
        out[k] = {"max_abs_diff": diff, "nan_mismatch": mism, "n_nan": int(nb.sum()),
                  "dtype_match": a.dtype == b.dtype,
                  "pass": bool(mism == 0 and diff <= PARITY_TOL and a.dtype == b.dtype)}
    return out


def parity(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    guard_out_dir(scratch)
    report = {"tolerance": PARITY_TOL, "reference": str(CONSTRAINTS_V2), "rollouts": {}}
    ok = True
    for rid in PARITY_RIDS:
        t0 = time.time()
        series = compute_v2(rid)
        write_atomic(scratch / f"{rid}.npz", series)
        res = compare(dict(np.load(scratch / f"{rid}.npz")), np.load(CONSTRAINTS_V2 / f"{rid}.npz"))
        P = int(json.loads((ROLLOUTS_V2 / rid / "manifest.json").read_text())["n_positions"])
        rpass = all(v["pass"] for v in res.values())
        ok &= rpass
        report["rollouts"][rid] = {"n_positions": P, "pass": rpass, "keys": res}
        worst = max(res, key=lambda k: res[k]["max_abs_diff"] or 0)
        print(f"[parity] {rid} (P={P}) {'PASS' if rpass else 'FAIL'}  "
              f"worst max|diff| {res[worst]['max_abs_diff']:.2e} ({worst}); "
              f"NaN mismatches {sum(v['nan_mismatch'] or 0 for v in res.values())}  "
              f"[{time.time() - t0:.1f} s]", flush=True)
    report["pass"] = bool(ok)
    PARITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARITY_JSON.write_text(json.dumps(report, indent=1))
    print(f"[parity] {'PASSED' if ok else 'FAILED'} -> {PARITY_JSON}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX_V3, help="corpus_v3_index.json; v3 rows only")
    ap.add_argument("--all-finished", action="store_true",
                    help="every finished v3 rollout under /data/rollouts_v3 (ignores --index)")
    ap.add_argument("--rids", nargs="+", default=None, help="these v3 rollout ids only")
    ap.add_argument("--out-dir", type=Path, default=CONSTRAINTS_V3)
    ap.add_argument("--force", action="store_true", help="recompute even if the output passes its check")
    ap.add_argument("--parity", action="store_true", help="v2 parity check against constraints/ and exit")
    ap.add_argument("--scratch", type=Path, default=PARITY_SCRATCH, help="--parity output dir")
    args = ap.parse_args()

    torch.set_num_threads(N_THREADS)
    if args.parity:
        parity(args.scratch)
        return
    if args.rids:
        rids = list(args.rids)
    elif args.all_finished:
        rids = [e["rollout_id"] for e in v3_entries()]
    else:
        entries = json.loads(args.index.read_text())["rollouts"]
        rids = [e["rollout_id"] for e in entries if e["source"] == "v3"]
    run(rids, args.out_dir, args.force)


if __name__ == "__main__":
    main()
