"""
audit.py

Stage 0 of results_collection_handoff.md (version 4): storage and inventory audit.

  --cache      confirm the full-set feature cache (/data/tmp/abc2_features) covers every
               original-set rollout the main run needs: rollout order equals the index,
               every row is marked done, every layer file has the expected shape, and
               for a few original-set rollouts (incl. one with 7 stored positions) the
               cached layer equals compute_constraints.pooled_and_hN read from
               /data/rollouts_v2 (the abc2 build check, repeated here).
  --inventory  which per-row outputs exist, for which rollouts and test sets
  --timing     time one A probe fit, one B grid fit (one layer, 12 outputs) and one
               2-layer MLP fit on the original training set (seed 0), on the GPU

Writes results/thesis_results/logs/audit_<part>.json; CHECKPOINT_0_audit.md is written
from these by hand-checked text.

Usage (repo root):
    source env.sh
    python analysis/thesis_results/audit.py --cache --inventory --timing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = common.OUT_DIR / "logs"


def check_cache() -> dict:
    import torch
    from compute_constraints import pooled_and_hN
    idx = common.ctx().idx
    cache = common.CACHE_DIR
    rids = json.loads((cache / "rollout_ids.json").read_text())
    rids = rids["rollout_ids"] if isinstance(rids, dict) else rids
    res = {"cache_dir": str(cache), "n_rows": len(rids)}
    res["order_equals_index"] = list(rids) == idx.rids.tolist()
    ori = np.nonzero(idx.block == "ori")[0]
    res["n_original_rows"] = int(len(ori))
    res["original_rows_are_first_300"] = bool(np.array_equal(ori, np.arange(300)))
    done = np.load(cache / "done.npy")
    res["done_all"] = bool(np.asarray(done).all())
    res["done_original"] = bool(np.asarray(done)[ori].all())
    shapes = {}
    for L in range(33):
        a = np.load(cache / f"layer{L:02d}.npy", mmap_mode="r")
        shapes[L] = list(a.shape)
    res["layer_shapes_ok"] = all(s == [len(rids), 520, 4096] for s in shapes.values())
    at = np.load(cache / "action_tokens.npy", mmap_mode="r")
    res["action_tokens_shape"] = list(at.shape)
    # Spot check against the stored hidden states (layer 15 and 28, 4 original rollouts,
    # one with 7 stored positions).
    p7 = [i for i in ori if idx.n_positions[i] == 7][:1]
    p3 = [i for i in ori if idx.n_positions[i] == 3][:3]
    checks = []
    for i in p7 + p3:
        h = torch.load(str(idx.rollout_dir(i) / "hidden.pt"), map_location="cpu")
        pooled, _ = pooled_and_hN(h, int(idx.n_positions[i]))
        pooled = pooled.numpy()
        for L in (15, 28):
            c = np.asarray(np.load(cache / f"layer{L:02d}.npy", mmap_mode="r")[i])
            ref = pooled[:, L].astype(np.float16)
            checks.append({"rollout": str(idx.rids[i]), "n_positions": int(idx.n_positions[i]),
                           "layer": L, "max_abs_diff": float(np.abs(c.astype(np.float32)
                                                                    - ref.astype(np.float32)).max())})
    res["spot_checks"] = checks
    res["spot_checks_exact"] = all(c["max_abs_diff"] == 0.0 for c in checks)
    # Original rows touched by the main run: every original rollout (train, eval-seen,
    # unseen) plus every LIBERO-PRO rollout in a test set or the stress set.
    res["covers_main_run"] = bool(res["order_equals_index"] and res["done_all"]
                                  and res["layer_shapes_ok"])
    return res


def inventory() -> dict:
    """Which per-row outputs exist (files and rollout coverage)."""
    idx = common.ctx().idx
    inv = {}

    def npz_info(path, keys=None):
        p = Path(path)
        if not p.exists():
            return {"file": str(p), "exists": False}
        z = np.load(p, allow_pickle=False)
        d = {"file": str(p), "exists": True, "keys": sorted(z.files)[:60]}
        if "rollout_ids" in z.files:
            d["n_rollouts"] = int(len(z["rollout_ids"]))
        if "p" in z.files:
            d["p_shape"] = list(z["p"].shape)
        if "names" in z.files:
            d["n_names"] = int(len(z["names"]))
        return d

    inv["A_34_probes_original_set"] = npz_info("/data/tmp/abc_scores/a_probes_seed0.npz")
    inv["C_33_probes_original_set"] = npz_info("/data/tmp/abc_scores/c_probes_seed0.npz")
    inv["B_grid_probes_original_set"] = {H: npz_info(f"/data/tmp/abc_scores/b_probes_h{H}_seed0.npz")
                                         for H in (0, 10, 25)}
    inv["abc2_D0_scores"] = npz_info("results/abc2/scores/D0_seed0.npz")
    inv["abc2_D0_mlp_scores"] = npz_info("results/abc2/scores/D0_seed0_mlp.npz")
    inv["abc2_D0_B10_scores"] = npz_info("results/abc2/scores/D0_seed0_b10.npz")
    inv["abc3_D0_deep_scores"] = npz_info("results/abc3/scores/D0_seed0_deep.npz")
    inv["abc_B_final_scores"] = {H: npz_info(f"results/abc/scores/b_h{H}_seed0.npz") for H in (0, 10, 25)}
    inv["abc_C_final_scores"] = npz_info("results/abc/scores/c_seed0.npz")
    inv["abc_A_final_scores"] = npz_info("results/abc/scores/a_seed0.npz")
    inv["label_free_scores"] = {H: npz_info(f"results/b_label_free/scores/lf_all_h{H}_seed0.npz")
                                for H in (0, 10, 25)}
    inv["abc2_work_A_grid_files"] = sorted(p.name for p in Path("/data/tmp/abc2_scores/work/D0/a_grid_seed0").glob("*.pkl"))[:3]
    w = Path("/data/tmp/abc2_scores/work/D0/a_grid_seed0/layer00.pkl")
    if w.exists():
        import pickle
        d = pickle.load(w.open("rb"))
        inv["abc2_work_A_grid_content"] = sorted(d.keys())
    w = Path("/data/tmp/abc2_scores/work/D0/c_grid_seed0.pkl")
    if w.exists():
        import pickle
        d = pickle.load(w.open("rb"))
        inv["abc2_work_C_grid_content"] = sorted(d.keys())
    inv["abc2_B_work_dir_files"] = len(list(Path("/data/tmp/abc2_scores/b").glob("*")))
    inv["index_blocks"] = {b: int((idx.block == b).sum()) for b in np.unique(idx.block)}
    return inv


def timing() -> dict:
    """One A probe fit, one B grid fit (one layer, 12 outputs), one 2-layer MLP fit; seed 0."""
    import torch
    import setup_a as abc_a
    import setup_b as abc_b
    import stage10_mlp
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    s = 0
    sets = ctx.sets("D0", s)
    train = common.train_rows_and_weights(idx, kept, sets["train"])
    res = {}
    t0 = time.time()
    X = common.load_layer(16)
    torch.cuda.synchronize()
    res["load_layer_s"] = time.time() - t0
    t0 = time.time()
    abc_a.fit_and_score(X, train, list(common.C_GRID))
    torch.cuda.synchronize()
    res["A_probe_fit_all_6_c_s"] = time.time() - t0
    t0 = time.time()
    abc_a.fit_and_score(X, train, [1e-6, 1e-5])
    torch.cuda.synchronize()
    res["A_probe_fit_to_c_1e-5_s"] = time.time() - t0
    # B: one layer, 12 outputs, refit on all train at the grid c.
    bt = common.b_targets(s)
    ys = [abc_b.lookahead(bt[abc_b.target_name(kd, 16)], H) for kd in abc_b.TARGET_KINDS for H in abc_b.HS]
    tr = sets["train"]
    rr = np.nonzero(kept[tr])
    t0 = time.time()
    abc_b.fit_grid_block(X, (tr[rr[0]], rr[1]), ys, [1e-4])
    torch.cuda.synchronize()
    res["B_grid_fit_one_layer_12_outputs_one_c_s"] = time.time() - t0
    t0 = time.time()
    abc_b.fit_grid_block(X, (tr[rr[0]], rr[1]), ys, list(common.C_GRID))
    torch.cuda.synchronize()
    res["B_grid_fit_one_layer_12_outputs_all_6_c_s"] = time.time() - t0
    # 2-layer MLP (stage10_mlp recipe), one weight decay.
    inp = stage10_mlp.Inputs(X, train[0], train[1])
    t0 = time.time()
    r = stage10_mlp.train_one(inp, train, 1e-2, s, ctx, sets["eval_seen"])
    torch.cuda.synchronize()
    res["MLP_fit_one_wd_s"] = time.time() - t0
    res["MLP_epochs_run"] = r["epochs_run"]
    res["n_train_rows"] = int(len(train[0]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for flag, fn, name in ((a.cache, check_cache, "cache"), (a.inventory, inventory, "inventory"),
                           (a.timing, timing, "timing")):
        if flag:
            t0 = time.time()
            r = fn()
            r["seconds"] = time.time() - t0
            (OUT / f"audit_{name}.json").write_text(json.dumps(r, indent=1, default=str))
            print(json.dumps(r, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
