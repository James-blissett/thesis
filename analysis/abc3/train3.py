"""
train3.py

Fits the deep-MLP grid of deep_mlp_handoff.md §2 (see common3.py for the grid, the model
and the work-file format). One work file per fit, written atomically; a fit whose work
file exists with the current spec is skipped, so the script is resumable and several
copies can run at once on disjoint --inputs / --pools / --seeds.

Training (same recipe for every depth; abc2's stage10_mlp where the handoff is silent):
    inputs standardised with train-row mean / sd (stage10_mlp.Inputs: unweighted, sd 0 -> 1);
    loss = batch mean of row weight x BCE (data.train_rows_and_weights: the abc2 rows and
    weights); AdamW(lr 1e-3, weight decay wd on every parameter), LambdaLR with 5% linear
    warm-up then cosine decay over max_epochs x batches per epoch; batch 4096 rows shuffled
    across rollouts each epoch (CPU generator seeded with the split seed); gradient norm
    clipped at 1.0; at most 50 epochs; after every epoch eval_seen M1-per-task of the
    running-sum score, stop when it has not improved (strictly) for 5 epochs, keep the best
    epoch's weights. torch.manual_seed(split seed) right before the model is built.
    Mixed precision: bf16 autocast for forward passes if results/abc3/precision.json says
    so (sanity3.py --bf16 writes it), else fp32. Loss and sigmoid in fp32.
Divergence (§5): a fit diverged if its train loss is non-finite at some epoch (training
stops there) or its eval_seen M1-per-task is below 0.5 (or NaN) at every epoch. A diverged
12-layer fit is retried once at lr 3e-4; if that diverges too, the fit is saved with
status "failed". Other depths are only flagged.

Usage (repo root):
    source env.sh
    python analysis/abc3/train3.py [--inputs A C] [--pools D0 D2] [--seeds 0 1 2] [--depths 2 6 12]
    python analysis/abc3/train3.py --count            # fits done / planned, then exit
Progress: one JSON line per finished fit in results/abc3/logs/fits.jsonl.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import time
from pathlib import Path

import numpy as np

import common3 as c3
from common3 import DEPTHS, HP, INPUTS, POOLS, REG_GRID, SEEDS, DeepMLP, lr_lambda, work_path

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import data2  # noqa: E402
import setup_c as abc_c  # noqa: E402
import stage3_c  # noqa: E402
import stage10_mlp  # noqa: E402
import stage4_a  # noqa: E402
from data2 import T, train_rows_and_weights  # noqa: E402
from pipe2_common import Ctx, load_pkl, log, m1_safe, save_pkl  # noqa: E402

DEV = stage10_mlp.DEV
EVAL_CHUNK = 30
FITS_LOG = c3.RESULTS / "logs" / "fits.jsonl"
PRECISION_FILE = c3.RESULTS / "precision.json"


def amp(precision: str):
    if precision == "bf16" and DEV.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, inp, pos: np.ndarray, precision: str, out: np.ndarray | None = None) -> np.ndarray:
    """P(failure) for every row of the rollouts at `pos`, written into an (N, T) float32 array."""
    model.eval()
    if out is None:
        out = np.zeros((inp.X.shape[0], T), dtype=np.float32)
    for i in range(0, len(pos), EVAL_CHUNK):
        p = pos[i:i + EVAL_CHUNK]
        with amp(precision):
            z = model(inp.rollouts(p))
        if z.dim() == 3:                     # plain stage10 MLP keeps the last axis
            z = z.squeeze(-1)
        out[p] = torch.sigmoid(z.float()).cpu().numpy()
    return out


def train_fit(inp, train, seed: int, ctx: Ctx, es_pos, tr_pos, *, depth: int, p: float, wd: float,
              lr: float, precision: str, plain: bool = False, track_train: bool = True) -> dict:
    """One fit. plain=True is abc2's stage10_mlp.train_one exactly (plain 256-256 MLP, constant
    lr, no clipping, fp32) for the sanity check; otherwise the §2 recipe."""
    r, t, y, w = train
    n = len(r)
    rr = torch.from_numpy(r).to(DEV)
    tt = torch.from_numpy(t).to(DEV)
    yy = torch.from_numpy(y.astype(np.float32)).to(DEV)
    ww = torch.from_numpy(w.astype(np.float32)).to(DEV)
    torch.manual_seed(seed)
    model = (stage10_mlp.make_mlp(inp.d) if plain else DeepMLP(inp.d, depth, c3.WIDTH, p)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = -(-n // HP["batch"])
    sched = None if plain else torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(HP["max_epochs"] * steps_per_epoch))
    gen = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    hist, since, nonfinite = [], 0, False
    for ep in range(1, HP["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEV)
        tot = 0.0
        for i in range(0, n, HP["batch"]):
            b = perm[i:i + HP["batch"]]
            with amp(precision):
                z = model(inp.rows(rr[b], tt[b]))
            if z.dim() == 2:
                z = z.squeeze(-1)
            loss = (ww[b] * F.binary_cross_entropy_with_logits(z.float(), yy[b], reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not plain:
                torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            tot += float(loss) * len(b)
        h = {"epoch": ep, "train_loss": tot / n, "lr_end": opt.param_groups[0]["lr"]}
        if not np.isfinite(h["train_loss"]):
            nonfinite = True
            h.update(eval_seen_m1_per_task=np.nan, eval_seen_m1_pooled=np.nan)
            hist.append(h)
            break
        pr = predict(model, inp, es_pos, precision)
        m = m1_safe(abc_c.running_sum(pr, ctx.kept), ctx.kept, ctx.idx, es_pos)
        sc = float(m["per_task"])
        h.update(eval_seen_m1_per_task=sc, eval_seen_m1_pooled=float(m["pooled"]))
        if track_train:
            pr = predict(model, inp, tr_pos, precision)
            mt = m1_safe(abc_c.running_sum(pr, ctx.kept), ctx.kept, ctx.idx, tr_pos)
            h.update(train_m1_per_task=float(mt["per_task"]), train_m1_pooled=float(mt["pooled"]))
        hist.append(h)
        if np.isfinite(sc) and sc > best["score"]:
            best = {"score": sc, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
            since = 0
        else:
            since += 1
            if since >= HP["patience"]:
                break
    es = np.array([x["eval_seen_m1_per_task"] for x in hist], dtype=float)
    below = bool(np.all(~(es >= 0.5)))           # NaN counts as below
    diverged = nonfinite or below
    reason = "non-finite train loss" if nonfinite else ("eval_seen M1-per-task < 0.5 at every epoch" if below else "")
    model.load_state_dict(best["state"])
    p_all = predict(model, inp, np.arange(inp.X.shape[0]), precision) if best["epoch"] > 0 else None
    return {"best_epoch": best["epoch"], "epochs_run": len(hist), "best_score": float(best["score"]),
            "stopped_early": len(hist) < HP["max_epochs"], "history": hist, "p": p_all,
            "diverged": diverged, "reason": reason,
            "n_params": int(sum(x.numel() for x in model.parameters())),
            "state": {k: v.detach().to("cpu", torch.float16) for k, v in best["state"].items()}}


def rid_hash(rids) -> str:
    return stage10_mlp.rid_hash(rids)


class Grid:
    def __init__(self, ctx: Ctx, precision: str, work_dir: Path):
        self.ctx, self.precision, self.work_dir = ctx, precision, work_dir
        self.sel = {c: stage10_mlp.pipeline_inputs(c3.ABC2_RESULTS, c, SEEDS) for c in POOLS}
        self.sets = {(c, s): ctx.sets(c, s) for c in POOLS for s in SEEDS}

    def input_desc(self, inp: str, pool: str, seed: int):
        return self.sel[pool]["a_probe"][seed] if inp == "A" else {"betas": self.sel[pool]["betas"]}

    def spec(self, inp, depth, pool, seed, p, wd) -> dict:
        sets = self.sets[(pool, seed)]
        return {"input": inp, "depth": depth, "pool": pool, "seed": seed, "p": p, "wd": wd,
                "input_desc": self.input_desc(inp, pool, seed),
                "train": rid_hash(self.ctx.idx.rids[sets["train"]]),
                "eval_seen": rid_hash(self.ctx.idx.rids[sets["eval_seen"]]),
                "n": int(self.ctx.idx.n), "hp": HP, "precision": self.precision}

    def path(self, *k) -> Path:
        return work_path(*k, work_dir=self.work_dir)

    def done(self, inp, depth, pool, seed, p, wd) -> bool:
        f = self.path(pool, inp, depth, seed, p, wd)
        if not f.exists():
            return False
        try:
            w = load_pkl(f)
        except Exception:
            return False
        if w.get("spec") != self.spec(inp, depth, pool, seed, p, wd) or w.get("status") not in ("ok", "failed"):
            return False
        return w["status"] == "failed" or (w["p"] is not None and w["p"].shape == (self.ctx.idx.n, T))

    def fit_group(self, X: torch.Tensor, inp: str, pool: str, seed: int, depths) -> None:
        todo = [(d, p, wd) for d in depths for (p, wd) in REG_GRID if not self.done(inp, d, pool, seed, p, wd)]
        if not todo:
            return
        sets = self.sets[(pool, seed)]
        train = train_rows_and_weights(self.ctx.idx, self.ctx.kept, sets["train"])
        X_inp = stage10_mlp.Inputs(X, train[0], train[1])
        for d, p, wd in todo:
            t0 = time.time()
            attempts, lr = [], HP["lr"]
            while True:
                res = train_fit(X_inp, train, seed, self.ctx, sets["eval_seen"], sets["train"],
                                depth=d, p=p, wd=wd, lr=lr, precision=self.precision)
                attempts.append({"lr": lr, "diverged": res["diverged"], "reason": res["reason"],
                                 "best_score": res["best_score"], "epochs_run": res["epochs_run"]})
                if res["diverged"] and d == 12 and lr == HP["lr"]:
                    log(f"[fit {inp} d{d} {pool} s{seed} {c3.reg_tag(p, wd)}] DIVERGED ({res['reason']}); "
                        f"retry at lr {HP['lr_retry']:g}")
                    lr = HP["lr_retry"]
                    continue
                break
            failed = res["diverged"] and d == 12
            out = {"spec": self.spec(inp, d, pool, seed, p, wd), "status": "failed" if failed else "ok",
                   **{k: res[k] for k in ("best_epoch", "epochs_run", "best_score", "stopped_early", "history",
                                          "diverged", "n_params", "state")},
                   "p": None if failed else res["p"], "lr": lr, "attempts": attempts,
                   "seconds": time.time() - t0, "n_train_rows": len(train[0]), "d": X_inp.d,
                   "precision": self.precision}
            if out["status"] == "ok" and out["p"] is None:
                out["status"] = "failed"
            f = self.path(pool, inp, d, seed, p, wd)
            f.parent.mkdir(parents=True, exist_ok=True)
            save_pkl(f, out)
            h = res["history"]
            log(f"[fit {inp} d{d} {pool} s{seed} {c3.reg_tag(p, wd)}] {out['status']}: best epoch "
                f"{res['best_epoch']}/{res['epochs_run']}, eval_seen M1/task {res['best_score']:.4f}, "
                f"loss {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f}, lr {lr:g}, "
                f"{out['seconds']:.0f} s, {len(train[0])} rows, d={X_inp.d}, {out['n_params']} params"
                + (f", DIVERGED ({res['reason']})" if res["diverged"] else ""))
            FITS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FITS_LOG.open("a") as fh:
                fh.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "fit": f.name, "pool": pool, "status": out["status"],
                                     "seconds": round(out["seconds"], 1), "best_score": res["best_score"],
                                     "best_epoch": res["best_epoch"], "lr": lr}) + "\n")
        del X_inp

    def c_features(self, pool: str, seed: int) -> torch.Tensor:
        run = stage10_mlp.Run(self.ctx, pool, SEEDS, ["C-MLP"], self.sel[pool])
        if not hasattr(self, "_cfam"):
            self._cfam = {}
        if pool not in self._cfam:
            self._cfam[pool] = stage3_c.smoothed_families(self.ctx, self.sel[pool]["betas"])
        run._c_feats = self._cfam[pool]
        return run.c_features(seed)


def make_ctx(work_dir: Path) -> Ctx:
    return Ctx(index_path=c3.REPO / data2.INDEX_V3, cache_dir=data2.CACHE_DIR,
               splits_dir=c3.ABC2_RESULTS / "splits", out_dir=c3.RESULTS,
               work_dir=work_dir / "ctx", demotions_path=c3.ABC2_RESULTS / "demotions.json",
               seeds=SEEDS, n_jobs=1)


def read_precision(arg: str | None) -> str:
    if arg:
        return arg
    if not PRECISION_FILE.exists():
        raise SystemExit(f"{PRECISION_FILE} missing: run sanity3.py --bf16 first, or pass --precision")
    return json.loads(PRECISION_FILE.read_text())["precision"]


def count(grid: Grid) -> tuple[int, int, int]:
    fits = c3.all_fits()
    done = [f for f in fits if grid.done(*f)]
    failed = sum(load_pkl(grid.path(c, i, d, s, p, wd))["status"] == "failed" for (i, d, c, s, p, wd) in done)
    return len(done), len(fits), failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=list(INPUTS))
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    ap.add_argument("--precision", choices=["bf16", "fp32"], default=None)
    ap.add_argument("--work-dir", type=Path, default=c3.WORK_DIR)
    ap.add_argument("--count", action="store_true")
    args = ap.parse_args(argv)
    import os
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 3)))
    os.chdir(c3.REPO)
    ctx = make_ctx(args.work_dir)
    grid = Grid(ctx, read_precision(args.precision), args.work_dir)
    if args.count:
        d, n, f = count(grid)
        print(f"{d} {n} {f}")
        return 0
    if DEV.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("ABC3_GPU_FRAC", 0.45)))
    t_all = time.time()
    log(f"[train3] precision {grid.precision}; inputs {args.inputs}; pools {args.pools}; seeds {args.seeds}; "
        f"depths {args.depths}; work {args.work_dir}")
    if "A" in args.inputs:
        need = {}
        for c in args.pools:
            for s in args.seeds:
                if any(not grid.done("A", d, c, s, p, wd) for d in args.depths for (p, wd) in REG_GRID):
                    need.setdefault(grid.sel[c]["a_probe"][s], []).append((c, s))
        if need:
            stage4_a.check_cache(ctx)
        for probe, todo in need.items():
            tl = time.time()
            X = stage4_a.load_probe_input(probe, ctx.cache_dir).to(DEV)
            log(f"[train3 A] {probe} loaded ({time.time() - tl:.0f} s) for {todo}")
            for c, s in todo:
                grid.fit_group(X, "A", c, s, args.depths)
            del X
            torch.cuda.empty_cache()
    if "C" in args.inputs:
        for c in args.pools:
            for s in args.seeds:
                if any(not grid.done("C", d, c, s, p, wd) for d in args.depths for (p, wd) in REG_GRID):
                    X = grid.c_features(c, s)
                    grid.fit_group(X, "C", c, s, args.depths)
                    del X
    d, n, f = count(grid)
    log(f"[train3] finished this invocation ({time.time() - t_all:.0f} s); grid {d}/{n} done, {f} failed")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
