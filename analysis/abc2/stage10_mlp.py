"""
stage10_mlp.py

Step 10 of more_data_collection.md: the MLP ablation, for D0 and D2 and seeds 0-2, on the
combined corpus. Two detectors per (condition, seed), each a 2-hidden-layer MLP in place of
the linear probe:

  A-MLP   input = the pooled hidden state at the stored index A selected for that
          condition and seed (pipeline selections.csv, A_probe; read with
          stage4_a.load_probe_input, so "action_tokens" would also work).
  C-MLP   input = C's final-detector inputs: the 132 prepared constraint series,
          built by Stage 3's own code path (stage3_c.smoothed_families at the condition's
          betas from selections.csv, then setup_c.seed_features on that seed's train set).

SAFE's training code is NOT in the OpenVLA fork on the box (/ephemeral/code/openvla: the
"add safe changes" commit only touches rollout / eval / token-uncertainty utilities; no
failure-detector or MLP training), so the hyperparameters are Step 10's fallback:
    standardise (mean / sd over the train rows, unweighted, sd 0 -> 1, as
    setup_a.fit_and_score) -> Linear(d, 256) -> ReLU -> Linear(256, 256) -> ReLU ->
    Linear(256, 1) (PyTorch default init); loss = binary cross-entropy weighted by the
    existing row weights (data.train_rows_and_weights: the same rows, >= 5-successes cut,
    1 / n_class(t) rescaled to mean 1), batch mean of w * BCE; AdamW(lr 1e-3, weight
    decay wd on every parameter, PyTorch defaults otherwise), batch 4096, rows shuffled
    each epoch; at most 50 epochs; after every epoch the eval_seen M1-per-task of the
    running-sum score is computed and training stops when it has not improved (strictly)
    for 5 epochs; the best epoch's weights are kept. torch.manual_seed(split seed) before
    the model is built and a CPU generator seeded with the split seed for the shuffles.
    wd from WD_GRID = (1e-4, 1e-2, 1e-1, 1): per condition and detector, the best mean
    over seeds of the best-epoch eval_seen M1-per-task (the c rule of Stages 3-4: plain
    argmax, ties to the first in grid order).
Scoring, M1 and conformal prediction exactly as the linear detectors (stage6_cp /
stage5_b): score = setup_c.running_sum of P(failure) over kept rows; M1 / M1-per-task by
pipe2_common.m1_safe on eval_seen, T0, T1, T2; band cp_eval.fit_band calibrated on the
condition's successful eval_seen rollouts, halves default_rng(CAL_SPLIT_SEED + seed),
h for the 15 alphas + 0.1, evaluate_cp with the cut variant; stress at alpha 0.1 with
stage6_cp.stress_cuts / alarms. P(failure) is kept as float32 (no fp16 rounding; that was
an A storage artefact).

Inputs (--pipe-dir, default results/abc2 = the pipeline's --out-dir, run it first):
    selections.csv           beta_action / beta_emb_temp / beta_cross_layer, A_probe per seed
    scores/<cond>_seed<s>.npz  the linear detectors, for the paired bootstrap only
A condition without pipeline selections is skipped (reason in mlp_status.json).

Outputs (<out> = --out-dir, default results/abc2; rows of other conditions already in a
file are kept, rows of the conditions run are replaced):
    mlp_final_table.csv   final_table.csv's columns; detectors A-MLP, C-MLP
    mlp_final_m1.csv      final_m1.csv's columns (per seed)
    mlp_cp_curves.csv     cp_curves.csv's columns (16 alphas, cut)
    mlp_stress.csv        stress.csv's columns (alpha 0.1)
    mlp_selections.csv    selections.csv's columns: per (cond, seed) A-MLP_input /
                          C-MLP_input, wd_A-MLP / wd_C-MLP (score = mean over seeds),
                          A-MLP_best_epoch / _epochs_run, same for C-MLP
    mlp_grid.csv          every wd: cond, seed, detector, wd, chosen, best_epoch,
                          epochs_run, stopped_early, test_set, M1 columns (best epoch)
    mlp_train_log.csv     per epoch: cond, seed, detector, wd, epoch, train_loss
                          (weighted mean BCE), eval_seen_m1_per_task, eval_seen_m1_pooled
    scores/<cond>_seed<s>_mlp.npz  the pipeline npz format with detector keys A-MLP and
                          C-MLP: K_m1, K_cp (both the running sum), K_p, K_mu, K_xi, K_h,
                          K_band01, K_R, K_t_end, K_n_d1, K_n_d2; plus rollout_ids, cond,
                          seed, alphas, kept_len, y, cell, base_task, block, cal_rids,
                          train_rids, eval_seen_rids, stress_*_cut, selections (JSON)
    mlp_paired_T0.csv     paired bootstrap on T0 (bootstrap2 --pair, same resamples as
                          the pipeline's paired_T0.csv), A-MLP vs A and C-MLP vs C per
                          condition, concatenated; also mlp_paired_T0_per_seed.csv and
                          mlp_paired_T0_check.json (bootstrap2's identity check against
                          mlp_final_table.csv and the pipeline's final_m1 / cp_curves);
                          the per-pair files are in <out>/mlp_paired/
    mlp_status.json       {"status": done|failed|skipped, "reason", conds_done, ...}
Work (--work-dir, default /data/tmp/abc2_scores/mlp; resumable per (cond, seed,
detector) and per wd inside; deletable after the run): <cond>/mlp_<det>_seed{s}.pkl with
per-wd history, metrics and best-epoch P(failure) (float32, ~2 MB per wd at N = 1000).
A work file is redone if its input (A's layer, C's betas), its train set or the
hyperparameters changed.

Usage (repo root, after pipeline.py has run D0 and D2):
    source env.sh
    ABC2_GPU_FRAC=1.0 OMP_NUM_THREADS=3 python analysis/abc2/stage10_mlp.py --cond D0 D2 \
        || python analysis/abc2/stage10_mlp.py --mark-failed "stage10_mlp.py exited non-zero"
    python analysis/abc2/stage10_mlp.py --paired-only      # redo the bootstrap only
    python analysis/abc2/stage10_mlp.py --mark-skipped "time"
Paths: --index, --cache-dir, --splits-dir, --pipe-dir (alias --in-dir), --out-dir,
--work-dir, --demotions (defaults: the real run's). --detectors A-MLP C-MLP, --seeds.
GPU: torch.cuda.set_per_process_memory_fraction(ABC2_GPU_FRAC), default 6/46 (via
stage4_a.gpu_setup). The pooled layer stays on the GPU in fp16 (4.3 GB at N = 1000) and
batches are standardised on the fly: ~6 GB peak at N = 1000; use ABC2_GPU_FRAC=1.0.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2  # noqa: E402
from pipe2_common import (EVAL_SETS, STRESS_SETS, Ctx, load_pkl, log, m1_row, m1_safe,  # noqa: E402
                          mean_std, read_csv, save_pkl, write_csv)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import cp_eval as abc_cp  # noqa: E402
import setup_c as abc_c  # noqa: E402
import stage3_c  # noqa: E402
import stage4_a  # noqa: E402
import stage6_cp  # noqa: E402
from data2 import T, m1_cut, train_rows_and_weights  # noqa: E402

DETS = ("A-MLP", "C-MLP")
LINEAR = {"A-MLP": "A", "C-MLP": "C"}
WD_GRID = (1e-4, 1e-2, 1e-1, 1.0)
HP = {"hidden": (256, 256), "lr": 1e-3, "batch": 4096, "max_epochs": 50, "patience": 5,
      "optimizer": "AdamW", "loss": "row-weighted BCE (batch mean)", "wd_grid": list(WD_GRID)}
ALPHAS = stage6_cp.ALPHAS                       # 15 grid levels, then 0.1
EVAL_CHUNK = 30                                 # rollouts per inference chunk
FAMS = ("action", "emb_temp", "cross_layer")
OUT_TABLES = {"final_m1": "mlp_final_m1.csv", "cp": "mlp_cp_curves.csv", "stress": "mlp_stress.csv",
              "sel": "mlp_selections.csv", "grid": "mlp_grid.csv", "log": "mlp_train_log.csv"}
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Missing(Exception):
    """A prerequisite (pipeline output) is missing: the condition is skipped."""


# --- Pipeline selections -------------------------------------------------------------------
def pipeline_inputs(pipe_dir: Path, cond: str, seeds) -> dict:
    """betas (C's inputs) and A's chosen probe per seed, from the pipeline's selections.csv."""
    sel = [r for r in read_csv(pipe_dir / "selections.csv") if r["cond"] == cond]
    if not sel:
        raise Missing(f"no {cond} rows in {pipe_dir / 'selections.csv'} (run pipeline.py first)")
    betas = {}
    for fam in FAMS:
        v = {r["value"] for r in sel if r["selection"] == f"beta_{fam}"}
        if len(v) != 1:
            raise Missing(f"{cond}: beta_{fam} missing or ambiguous in selections.csv: {v}")
        betas[fam] = v.pop()
    probe = {}
    for s in seeds:
        v = [r["value"] for r in sel if r["selection"] == "A_probe" and int(r["seed"]) == s]
        if len(v) != 1:
            raise Missing(f"{cond} seed {s}: A_probe missing in selections.csv (pipeline seeds "
                          f"{sorted({int(r['seed']) for r in sel})})")
        probe[s] = v[0]
    return {"betas": betas, "a_probe": probe}


# --- Model and training --------------------------------------------------------------------
def make_mlp(d: int) -> nn.Module:
    h1, h2 = HP["hidden"]
    return nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))


class Inputs:
    """(N, T, d) inputs on DEV (any float dtype) with train-row standardisation applied on
    the fly, so no standardised copy of the train rows is kept."""

    def __init__(self, X: torch.Tensor, r: np.ndarray, t: np.ndarray):
        self.X = X
        self.d = X.shape[-1]
        rr = torch.from_numpy(r).to(DEV)
        tt = torch.from_numpy(t).to(DEV)
        s1 = torch.zeros(self.d, dtype=torch.float64, device=DEV)
        s2 = torch.zeros(self.d, dtype=torch.float64, device=DEV)
        for i in range(0, len(r), 8192):
            xb = X[rr[i:i + 8192], tt[i:i + 8192]].double()
            s1 += xb.sum(0)
            s2 += (xb * xb).sum(0)
        n = len(r)
        mu = s1 / n
        sd = torch.sqrt(torch.clamp(s2 / n - mu * mu, min=0.0))
        self.mu = mu.float()
        self.sd = torch.where(sd > 0, sd, torch.ones_like(sd)).float()

    def rows(self, rr: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        return (self.X[rr, tt].float() - self.mu) / self.sd

    def rollouts(self, pos: np.ndarray) -> torch.Tensor:
        return (self.X[torch.from_numpy(pos).to(DEV)].float() - self.mu) / self.sd


@torch.no_grad()
def predict(model: nn.Module, inp: Inputs, pos: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """P(failure) for every row of the rollouts at `pos`, written into an (N, T) array."""
    model.eval()
    if out is None:
        out = np.zeros((inp.X.shape[0], T), dtype=np.float32)
    for i in range(0, len(pos), EVAL_CHUNK):
        p = pos[i:i + EVAL_CHUNK]
        z = model(inp.rollouts(p)).squeeze(-1)
        out[p] = torch.sigmoid(z).cpu().numpy()
    return out


def train_one(inp: Inputs, train, wd: float, seed: int, ctx: Ctx, es_pos: np.ndarray) -> dict:
    """Fit one MLP; early stopping on eval_seen M1-per-task (patience HP['patience'])."""
    r, t, y, w = train
    n = len(r)
    rr = torch.from_numpy(r).to(DEV)
    tt = torch.from_numpy(t).to(DEV)
    yy = torch.from_numpy(y.astype(np.float32)).to(DEV)
    ww = torch.from_numpy(w.astype(np.float32)).to(DEV)
    torch.manual_seed(seed)
    model = make_mlp(inp.d).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": 0, "state": copy.deepcopy(model.state_dict()), "m": None}
    hist = []
    since = 0
    for ep in range(1, HP["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEV)
        tot = 0.0
        for i in range(0, n, HP["batch"]):
            b = perm[i:i + HP["batch"]]
            z = model(inp.rows(rr[b], tt[b])).squeeze(-1)
            loss = (ww[b] * F.binary_cross_entropy_with_logits(z, yy[b], reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        p = predict(model, inp, es_pos)
        m = m1_safe(abc_c.running_sum(p, ctx.kept), ctx.kept, ctx.idx, es_pos)
        sc = float(m["per_task"])
        hist.append({"epoch": ep, "train_loss": tot / n, "eval_seen_m1_per_task": sc,
                     "eval_seen_m1_pooled": float(m["pooled"])})
        if np.isfinite(sc) and sc > best["score"]:
            best = {"score": sc, "epoch": ep, "state": copy.deepcopy(model.state_dict()), "m": m}
            since = 0
        else:
            since += 1
            if since >= HP["patience"]:
                break
    model.load_state_dict(best["state"])
    p_all = predict(model, inp, np.arange(inp.X.shape[0]))
    return {"best_epoch": best["epoch"], "epochs_run": len(hist), "best_score": best["score"],
            "stopped_early": len(hist) < HP["max_epochs"], "history": hist, "p": p_all}


# --- One condition ---------------------------------------------------------------------------
def rid_hash(rids) -> str:
    return hashlib.sha1("\n".join(map(str, rids)).encode()).hexdigest()[:16]


class Run:
    def __init__(self, ctx: Ctx, cond: str, seeds, dets, inp: dict):
        self.ctx, self.cond, self.seeds, self.dets = ctx, cond, list(seeds), list(dets)
        self.betas, self.a_probe = inp["betas"], inp["a_probe"]
        self.sets = {s: ctx.sets(cond, s) for s in self.seeds}
        self._c_feats = None

    def wfile(self, det: str, s: int) -> Path:
        return self.ctx.wpath(self.cond, f"mlp_{det}_seed{s}.pkl")

    def spec(self, det: str, s: int) -> dict:
        inp = self.a_probe[s] if det == "A-MLP" else {"betas": self.betas}
        return {"det": det, "input": inp, "train": rid_hash(self.ctx.idx.rids[self.sets[s]["train"]]),
                "eval_seen": rid_hash(self.ctx.idx.rids[self.sets[s]["eval_seen"]]),
                "n": int(self.ctx.idx.n), "hp": HP}

    def work(self, det: str, s: int) -> dict:
        p = self.wfile(det, s)
        if p.exists():
            w = load_pkl(p)
            if w["spec"] == self.spec(det, s):
                return w
        return {"spec": self.spec(det, s), "wd": {}}

    def missing(self, det: str, s: int) -> list:
        w = self.work(det, s)
        return [wd for wd in WD_GRID if wd not in w["wd"]]

    def fit(self, det: str, s: int, X: torch.Tensor) -> None:
        """Every missing wd for (det, seed) on inputs X (N, T, d) on DEV; saved per wd."""
        w = self.work(det, s)
        todo = [wd for wd in WD_GRID if wd not in w["wd"]]
        if not todo:
            return
        sets = self.sets[s]
        train = train_rows_and_weights(self.ctx.idx, self.ctx.kept, sets["train"])
        inp = Inputs(X, train[0], train[1])
        evals = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
        for wd in todo:
            t0 = time.time()
            res = train_one(inp, train, wd, s, self.ctx, sets["eval_seen"])
            sc = abc_c.running_sum(res["p"], self.ctx.kept)
            res["metrics"] = {k: m1_row(m1_safe(sc, self.ctx.kept, self.ctx.idx, pos))
                              for k, pos in evals.items()}
            res["seconds"] = time.time() - t0
            res["n_train_rows"] = len(train[0])
            res["d"] = inp.d
            w["wd"][wd] = res
            save_pkl(self.wfile(det, s), w)
            h = res["history"]
            log(f"[mlp {self.cond} seed {s} {det}] wd {wd:g}: best epoch {res['best_epoch']}/"
                f"{res['epochs_run']}, eval_seen M1-per-task {res['best_score']:.4f}, loss "
                f"{h[0]['train_loss']:.4f} -> {h[res['best_epoch'] - 1]['train_loss']:.4f} "
                f"({res['seconds']:.0f} s, {len(train[0])} rows, d={inp.d})")
        del inp

    def c_features(self, s: int) -> torch.Tensor:
        """C's final-detector inputs for seed s: (N, T, 132) float32 on DEV, Stage 3's path."""
        if self._c_feats is None:
            self._c_feats = stage3_c.smoothed_families(self.ctx, self.betas)
        feats, _ = abc_c.seed_features(self._c_feats, self.ctx.idx, self.ctx.kept, self.sets[s]["train"])
        names = list(feats)
        X = np.stack([feats[n] for n in names], -1)
        del feats
        self.c_inputs = names
        return torch.from_numpy(X).to(DEV)

    def choose(self) -> dict:
        """wd per detector: best mean over seeds of the best-epoch eval_seen M1-per-task."""
        self.W = {(d, s): self.work(d, s) for d in self.dets for s in self.seeds}
        self.wd, self.wd_score = {}, {}
        for d in self.dets:
            score = {wd: mean_std([self.W[(d, s)]["wd"][wd]["best_score"] for s in self.seeds])
                     for wd in WD_GRID}
            self.wd[d] = max(WD_GRID, key=lambda wd: score[wd][0] if np.isfinite(score[wd][0]) else -np.inf)
            self.wd_score[d] = score
            log(f"[mlp {self.cond}] {d}: wd {self.wd[d]:g} (eval_seen M1-per-task "
                f"{score[self.wd[d]][0]:.4f}; grid "
                + ", ".join(f"{wd:g}: {score[wd][0]:.4f}" for wd in WD_GRID) + ")")
        return self.wd

    def input_label(self, d: str, s: int) -> str:
        if d == "A-MLP":
            return f"{self.a_probe[s]} (pooled hidden state)" if self.a_probe[s].startswith("layer") \
                else f"{self.a_probe[s]}"
        return ("C final inputs (132 prepared series) betas " +
                " ".join(f"{f}={self.betas[f]}" for f in FAMS))

    def selections(self, s: int) -> dict:
        out = {"hp": HP, "betas": self.betas, "a_probe": self.a_probe[s]}
        for d in self.dets:
            r = self.W[(d, s)]["wd"][self.wd[d]]
            out[d] = {"input": self.input_label(d, s), "wd": self.wd[d],
                      "wd_score": self.wd_score[d][self.wd[d]][0], "best_epoch": r["best_epoch"],
                      "epochs_run": r["epochs_run"], "n_train_rows": r["n_train_rows"], "d": r["d"]}
        return out

    def outputs(self) -> dict:
        """CP, M1, stress rows, selections, grid and training log; one npz per seed."""
        idx, kept, ctx = self.ctx.idx, self.ctx.kept, self.ctx
        m1_rows, cp_rows, stress_rows, sel_rows, grid_rows, log_rows = [], [], [], [], [], []
        for s in self.seeds:
            sets = self.sets[s]
            evals = {k: sets[k] for k in EVAL_SETS if len(sets[k])}
            cuts = {k: m1_cut(idx, pos) for k, pos in evals.items()}
            cal = sets["cal"]
            npz = {"rollout_ids": idx.rids, "cond": self.cond, "seed": s,
                   "alphas": np.array(ALPHAS), "kept_len": idx.kept_len, "y": idx.y,
                   "cell": idx.task, "base_task": idx.base_task, "block": idx.block,
                   "cal_rids": idx.rids[cal], "train_rids": idx.rids[sets["train"]],
                   "eval_seen_rids": idx.rids[sets["eval_seen"]]}
            for d in self.dets:
                wd = self.wd[d]
                r = self.W[(d, s)]["wd"][wd]
                p = r["p"]
                sc = abc_c.running_sum(p, kept).astype(np.float32)
                mu, xi, R, t_end, n1, n2 = abc_cp.fit_band(
                    sc, kept, cal, np.random.default_rng(stage6_cp.CAL_SPLIT_SEED + s))
                hs = np.array([abc_cp.band_h(R, a) for a in ALPHAS])
                detail = f"{self.input_label(d, s).split(' (')[0]} wd={wd:g} epoch={r['best_epoch']}"
                head = {"cond": self.cond, "seed": s, "detector": d, "detail": detail}
                for ts, pos in evals.items():
                    m1_rows.append({**head, "test_set": ts, **m1_row(m1_safe(sc, kept, idx, pos))})
                    for a, h in zip(ALPHAS, hs):
                        e = abc_cp.evaluate_cp(sc, kept, idx, pos, mu, xi, h, cuts[ts])
                        cp_rows.append({**head, "test_set": ts, "alpha": float(a), "variant": "cut",
                                        "on_alpha_grid": bool(a in abc_cp.ALPHAS), "h": h,
                                        "bal_acc": e["bal_acc"], "tpr": e["tpr"], "tnr": e["tnr"],
                                        "far": 1 - e["tnr"], "det_time": e["det_time"],
                                        "n": int(len(pos)), "n_fail": int(idx.y[pos].sum()),
                                        "n_cal": int(len(cal)), "n_d1": n1, "n_d2": n2, "t_end": t_end})
                npz.update({f"{d}_m1": sc, f"{d}_cp": sc, f"{d}_p": p.astype(np.float32),
                            f"{d}_mu": mu, f"{d}_xi": xi, f"{d}_h": hs, f"{d}_band01": mu + hs[-1] * xi,
                            f"{d}_R": R, f"{d}_t_end": t_end, f"{d}_n_d1": n1, f"{d}_n_d2": n2})
                for ss in STRESS_SETS:
                    pos = sets[ss]
                    if not len(pos):
                        continue
                    cut = stage6_cp.stress_cuts(ctx, s, sets, pos)
                    fired = stage6_cp.alarms(sc, kept, pos, mu, xi, hs[-1], cut)
                    npz[f"{ss}_cut"] = cut
                    for blk in ["all"] + sorted(set(idx.block[pos].tolist())):
                        m = np.ones(len(pos), bool) if blk == "all" else idx.block[pos] == blk
                        stress_rows.append({**head, "stress_set": ss, "block": blk,
                                            "alpha": stage6_cp.ALPHA_TABLE, "n": int(m.sum()),
                                            "n_fail": int(idx.y[pos][m].sum()),
                                            "share_alarm_before_cut": float(fired[m].mean()),
                                            "mean_cut": float(cut[m].mean())})
                for g in WD_GRID:
                    rg = self.W[(d, s)]["wd"][g]
                    for ts, m in rg["metrics"].items():
                        grid_rows.append({"cond": self.cond, "seed": s, "detector": d, "wd": g,
                                          "chosen": g == wd, "best_epoch": rg["best_epoch"],
                                          "epochs_run": rg["epochs_run"],
                                          "stopped_early": rg["stopped_early"],
                                          "seconds": rg["seconds"], "test_set": ts, **m})
                    for hrow in rg["history"]:
                        log_rows.append({"cond": self.cond, "seed": s, "detector": d, "wd": g, **hrow})

                def add(name, value, crit, score=""):
                    sel_rows.append({"cond": self.cond, "seed": s, "selection": name, "value": value,
                                     "criterion": crit, "score_eval_seen": score})
                add(f"{d}_input", self.input_label(d, s),
                    "A_probe of this seed (pipeline)" if d == "A-MLP" else "betas of this condition (pipeline)")
                add(f"wd_{d}", wd, "mean over seeds of best-epoch eval_seen M1-per-task",
                    self.wd_score[d][wd][0])
                add(f"{d}_best_epoch", r["best_epoch"], "best eval_seen M1-per-task (this seed)",
                    r["best_score"])
                add(f"{d}_epochs_run", r["epochs_run"], f"early stopping, patience {HP['patience']}, "
                    f"max {HP['max_epochs']}")
            npz["selections"] = json.dumps(self.selections(s), default=float)
            out = ctx.scores_dir / f"{self.cond}_seed{s}_mlp.npz"
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, **npz)
            tmp.replace(out)
        return {"final_m1": m1_rows, "cp": cp_rows, "stress": stress_rows, "sel": sel_rows,
                "grid": grid_rows, "log": log_rows}


# --- Output assembly (stage5_b's, copied) ------------------------------------------------------
def merge_csv(path: Path, rows: list[dict], conds) -> list[dict]:
    """Keep rows of other conditions already in `path`, replace those of `conds`."""
    old = [r for r in read_csv(path) if r.get("cond") not in set(conds)]
    write_csv(path, old + rows)
    return read_csv(path)


def final_table(m1: list[dict], cp: list[dict]) -> list[dict]:
    """pipeline.assemble's final_table.csv, from the CSV rows (same columns)."""
    cp = [r for r in cp if r["on_alpha_grid"] == "False"]
    final = []
    for cond, ts, det in dict.fromkeys((r["cond"], r["test_set"], r["detector"]) for r in m1):
        mr = [r for r in m1 if (r["cond"], r["test_set"], r["detector"]) == (cond, ts, det)]
        cr = [r for r in cp if (r["cond"], r["test_set"], r["detector"]) == (cond, ts, det)]
        row = {"cond": cond, "test_set": ts, "detector": det, "n_seeds": len(mr)}
        for k in ("m1_pooled", "m1_per_task"):
            row[f"{k}_mean"], row[f"{k}_std"] = mean_std([float(r[k]) for r in mr])
        row["floor_mean"] = mean_std([float(r["floor"]) for r in mr])[0]
        row["n_cells"] = " ".join(r["n_cells"] for r in mr)
        for k in ("bal_acc", "tpr", "far", "det_time"):
            row[f"a01_{k}_mean"], row[f"a01_{k}_std"] = mean_std([float(r[k]) for r in cr])
        row["n_cal_mean"] = mean_std([float(r["n_cal"]) for r in cr])[0]
        row["n_mean"] = mean_std([float(r["n"]) for r in mr])[0]
        final.append(row)
    return final


def write_outputs(out_dir: Path, results: dict) -> None:
    conds = list(results)
    merged = {}
    for key, fname in OUT_TABLES.items():
        rows = [r for c in conds for r in results[c][key]]
        merged[key] = merge_csv(out_dir / fname, rows, conds)
    write_csv(out_dir / "mlp_final_table.csv", final_table(merged["final_m1"], merged["cp"]))


# --- Paired bootstrap (bootstrap2 --pair) ------------------------------------------------------
def paired(args, conds, dets) -> dict:
    """bootstrap2 --pair for MLP vs linear per condition on T0, concatenated."""
    import bootstrap2
    sub = args.out_dir / "mlp_paired"
    rows, per_seed, sides, done = [], [], [], []
    for cond in conds:
        for d in dets:
            fx = str(args.out_dir / "scores" / f"{cond}_seed{{seed}}_mlp.npz")
            fy = str(args.pipe_dir / "scores" / f"{cond}_seed{{seed}}.npz")
            need = [Path(f.format(seed=s)) for f in (fx, fy) for s in args.seeds]
            if not all(p.exists() for p in need):
                log(f"[mlp paired] {cond} {d}: missing {[str(p) for p in need if not p.exists()]}; skipped")
                continue
            name = f"T0_{cond}_{d}.csv"
            rc = bootstrap2.main(["--pair", f"{fx}:{d}", f"{fy}:{LINEAR[d]}", "--in-dir", str(args.pipe_dir),
                                  "--out-dir", str(sub), "--test-set", "T0",
                                  "--seeds", *map(str, args.seeds), "--demotions", str(args.demotions),
                                  "--check-table", str(args.out_dir / "mlp_final_table.csv"),
                                  "--out-name", name])
            if rc:
                raise RuntimeError(f"bootstrap2 returned {rc} for {cond} {d}")
            rows += read_csv(sub / name)
            per_seed += read_csv(sub / name.replace(".csv", "_per_seed.csv"))
            chk = json.loads((sub / name.replace(".csv", "_check.json")).read_text())
            sides += chk["sides"]
            done.append((cond, d))
    if not done:
        return {"passed": None, "pairs": []}
    # keep rows of conditions not re-run
    old = [r for r in read_csv(args.out_dir / "mlp_paired_T0.csv") if r["cond_x"] not in set(conds)]
    old_ps = [r for r in read_csv(args.out_dir / "mlp_paired_T0_per_seed.csv") if r["cond_x"] not in set(conds)]
    write_csv(args.out_dir / "mlp_paired_T0.csv", old + rows)
    write_csv(args.out_dir / "mlp_paired_T0_per_seed.csv", old_ps + per_seed)
    mx = max((c["max_abs_diff"] or 0) for c in sides)
    passed = all(c["passed"] for c in sides)
    (args.out_dir / "mlp_paired_T0_check.json").write_text(json.dumps(
        {"test_set": "T0", "passed": passed, "tolerance": bootstrap2.CHECK_TOL,
         "rng_seed": bootstrap2.RNG_SEED, "n_boot": bootstrap2.N_BOOT,
         "check_table": str(args.out_dir / "mlp_final_table.csv"), "max_abs_diff": mx,
         "pairs": [f"{c} {d} vs {LINEAR[d]}" for c, d in done], "sides": sides},
        indent=1, default=float))
    log(f"[mlp paired] {len(done)} pairs -> {args.out_dir / 'mlp_paired_T0.csv'}; identity check "
        f"{'PASSED' if passed else 'FAILED'} (max |diff| {mx:.2g})")
    return {"passed": passed, "max_abs_diff": mx, "pairs": [f"{c} {d}" for c, d in done]}


# --- Status ------------------------------------------------------------------------------------
def write_status(out_dir: Path, status: str, reason: str, **extra) -> dict:
    p = out_dir / "mlp_status.json"
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        d = {}
    d.update({"status": status, "reason": reason,
              "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, **extra)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1, default=str))
    return d


# --- Main --------------------------------------------------------------------------------------
def run_all(args, ctx: Ctx, runs: dict) -> dict:
    t_all = time.time()
    if any(r.missing(d, s) for r in runs.values() for d in r.dets for s in r.seeds):
        if DEV.type == "cuda":
            stage4_a.gpu_setup()
        else:
            log("[mlp] WARNING: no GPU; training on the CPU")
    # A-MLP: layer-outer, each cache file read once for every (cond, seed) using it.
    need = {}
    for cond, r in runs.items():
        if "A-MLP" in r.dets:
            for s in r.seeds:
                if r.missing("A-MLP", s):
                    need.setdefault(r.a_probe[s], []).append((cond, s))
    if need:
        stage4_a.check_cache(ctx)
    for probe, todo in need.items():
        tl = time.time()
        X = stage4_a.load_probe_input(probe, ctx.cache_dir).to(DEV)
        log(f"[mlp A] {probe} loaded ({time.time() - tl:.0f} s) for {todo}")
        for cond, s in todo:
            runs[cond].fit("A-MLP", s, X)
        del X
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    for cond, r in runs.items():
        if "C-MLP" in r.dets:
            for s in r.seeds:
                if r.missing("C-MLP", s):
                    X = r.c_features(s)
                    r.fit("C-MLP", s, X)
                    del X
            r._c_feats = None
    results = {}
    for cond, r in runs.items():
        r.choose()
        results[cond] = r.outputs()
    write_outputs(args.out_dir, results)
    for row in read_csv(args.out_dir / "mlp_final_table.csv"):
        if row["cond"] in runs:
            log(f"   {row['cond']} {row['detector']:5s} {row['test_set']:9s} M1-per-task "
                f"{float(row['m1_per_task_mean']):.4f}±{float(row['m1_per_task_std'] or 'nan'):.4f}  "
                f"pooled {float(row['m1_pooled_mean']):.4f}  BA@0.1 {float(row['a01_bal_acc_mean']):.4f}  "
                f"TPR {float(row['a01_tpr_mean']):.3f}  FAR {float(row['a01_far_mean']):.3f}  "
                f"cells {row['n_cells']}")
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEV.type == "cuda" else float("nan")
    log(f"[mlp] trained and scored ({time.time() - t_all:.0f} s; peak GPU memory {peak:.2f} GB)")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cond", nargs="+", default=["D0", "D2"], choices=list(data2.CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(data2.SEEDS))
    ap.add_argument("--detectors", nargs="+", default=list(DETS), choices=list(DETS))
    ap.add_argument("--index", type=Path, default=data2.INDEX_V3)
    ap.add_argument("--cache-dir", type=Path, default=data2.CACHE_DIR)
    ap.add_argument("--splits-dir", type=Path, default=data2.SPLITS_DIR)
    ap.add_argument("--out-dir", type=Path, default=data2.RESULTS_DIR)
    ap.add_argument("--pipe-dir", "--in-dir", dest="pipe_dir", type=Path, default=None,
                    help="the pipeline's output dir (selections.csv, scores/); default --out-dir")
    ap.add_argument("--work-dir", type=Path, default=Path("/data/tmp/abc2_scores/mlp"))
    ap.add_argument("--demotions", type=Path, default=data2.DEMOTIONS)
    ap.add_argument("--no-paired", action="store_true", help="skip the paired bootstrap")
    ap.add_argument("--paired-only", action="store_true", help="only (re)do the paired bootstrap")
    ap.add_argument("--mark-failed", metavar="REASON", default=None)
    ap.add_argument("--mark-skipped", metavar="REASON", default=None)
    args = ap.parse_args(argv)
    args.pipe_dir = args.pipe_dir or args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mark_failed is not None or args.mark_skipped is not None:
        st = args.out_dir / "mlp_status.json"
        try:
            cur = json.loads(st.read_text()) if st.exists() else {}
        except Exception:
            cur = {}
        if args.mark_skipped is not None:
            write_status(args.out_dir, "skipped", args.mark_skipped)
        elif cur.get("status") != "failed":        # keep a more specific failure reason
            write_status(args.out_dir, "failed", args.mark_failed)
        return 0

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 3)))
    t0 = time.time()
    try:
        if args.paired_only:
            pr = paired(args, args.cond, args.detectors)
            st = json.loads((args.out_dir / "mlp_status.json").read_text()) \
                if (args.out_dir / "mlp_status.json").exists() else {}
            write_status(args.out_dir, st.get("status", "done"), st.get("reason", ""), paired_T0=pr)
            return 0
        ctx = Ctx(index_path=args.index, cache_dir=args.cache_dir, splits_dir=args.splits_dir,
                  out_dir=args.out_dir, work_dir=args.work_dir, demotions_path=args.demotions,
                  seeds=tuple(args.seeds), n_jobs=1)
        runs, skipped = {}, {}
        for cond in args.cond:
            try:
                inp = pipeline_inputs(args.pipe_dir, cond, args.seeds)
            except Missing as e:
                skipped[cond] = str(e)
                log(f"[mlp {cond}] SKIPPED: {e}")
                continue
            runs[cond] = Run(ctx, cond, args.seeds, args.detectors, inp)
            log(f"[mlp {cond}] betas {inp['betas']}; A probes {inp['a_probe']}")
        if not runs:
            write_status(args.out_dir, "skipped", "; ".join(f"{c}: {r}" for c, r in skipped.items()))
            return 0
        log(f"index {args.index}: {ctx.idx.n} rollouts; conds {list(runs)}; seeds {args.seeds}; "
            f"device {DEV}; out {args.out_dir}; work {args.work_dir}")
        run_all(args, ctx, runs)
        pr = {"passed": None, "pairs": []} if args.no_paired else paired(args, list(runs), args.detectors)
        st = args.out_dir / "mlp_status.json"
        try:
            prev = json.loads(st.read_text()) if st.exists() else {}
        except Exception:
            prev = {}
        done = sorted(set(prev.get("conds_done", [])) | set(runs))
        reason = "" if not skipped else "skipped: " + "; ".join(f"{c}: {r}" for c, r in skipped.items())
        write_status(args.out_dir, "done", reason, conds_done=done, conds_skipped=skipped,
                     seeds=args.seeds, detectors=args.detectors, seconds=round(time.time() - t0),
                     safe_code_found=False, hyperparameters=HP,
                     wd={c: r.wd for c, r in runs.items()}, paired_T0=pr)
        return 0
    except (Exception, SystemExit) as e:  # the MLP must never block the report
        traceback.print_exc()
        write_status(args.out_dir, "failed", f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
