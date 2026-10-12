"""
stage5_b.py

Step 11 of more_data_collection.md: setup B at H = 10 ONLY, per condition (D0, D2) and
seed, on the combined corpus. analysis/abc/setup_b.py restricted to H = 10 and
generalised to (index, condition, seed, test sets) with the pipeline's machinery
(pipe2_common.Ctx, data2.condition_sets / test_sets). Every fitting / scoring function
is setup_b's own, imported: lookahead, two_means_threshold, balanced_weights, fit_multi
(via fit_grid_block), predict_block, target_auroc; the final detector is
setup_c.fit_prob + running_sum with data.train_rows_and_weights; the band is
cp_eval.fit_band / band_h / evaluate_cp; stress cuts and alarms are stage6_cp's.

Inputs taken from the pipeline's outputs for the SAME condition (--pipe-dir, default
results/abc2; run pipeline.py for the condition first):
    beta per family    <pipe>/selections.csv rows beta_action / beta_emb_temp /
                       beta_cross_layer (as setup_b used Stage 3's beta)
    unstable targets   <pipe>/tables/<cond>/constraint_m1_summary.csv: primary series
                       with unstable = True at the family's chosen beta (as setup_b read
                       results/abc/constraint_m1.csv); those (layer, target) are skipped
Targets: prep.smooth at that beta, prep.fit_scale_orient on the condition's train set
per seed, NaN off kept rows (setup_b.prepared_targets with per-rollout paths).

Grid, per layer l in 0..31 and target in [act_rep, act_mag, emb_temp[l],
xl_final[min(l,30)]] minus unstable ones, H = 10:
    y[t] = max(target[t : t+11]) within kept rows, binarised at the two-means split of
    the training values; input = layer l's pooled hidden state; balanced class weights;
    all of one layer's targets fitted together on the GPU (setup_b.fit_grid_block).
    Joint-H fit (--joint-h, default on; --no-joint-h fits the H = 10 outputs alone):
    each layer's block is fitted exactly as setup_b fitted it, with one output per
    (target, H) for H in (0, 10, 25), in setup_b's order (target-major, then H), the
    same warm start over sorted C_GRID, LBFGS max_iter and objective; ONLY the H = 10
    outputs are then used (AUROCs, thresholds, probe outputs). The H = 0 / 25 outputs
    are a numerical device and are never saved or reported. Why: the outputs'
    objectives are independent, but LBFGS optimises their sum and stops at max_iter /
    tolerance as a whole, so fitting 4 outputs instead of setup_b's 12 lands at a
    slightly different point (per-row probabilities up to ~0.004 apart). That was enough
    to flip the final detector's c on D0 (1e-6 instead of abc's 1e-3) and fail the
    Step 11 D0 check (T0 M1-per-task 0.6968 vs 0.7086) in the first real run; on abc's
    own probe outputs the downstream code reproduces 0.7086 exactly.
    pass 1: refit on all kept train rows at every c in C_GRID (warm-started), target
            AUROC on eval_seen, T0, T1, T2.  B's grid c = the best mean over seeds of the
            mean over probes of eval_seen target AUROC (handoff v10).
    pass 2: at that c, out-of-fold outputs on train (GroupKFold(5) by rollout,
            thresholds and standardisation refitted per fold) and a refit on all of
            train for every other rollout the condition uses.
Final detector: the grid's probabilities (out-of-fold on train) as inputs, trained on
outcome as A / C (row weights, >= 5-successes cut), running-sum score; c over C_GRID by
the best mean over seeds of eval_seen M1-per-task. Per-target ablations at that c.
Conformal prediction (cut variant, as stage6_cp): calibrated on the condition's
successful eval_seen rollouts, halves drawn with default_rng(CAL_SPLIT_SEED + seed),
15 alphas + 0.1, evaluated on eval_seen, T0, T1, T2; stress share alarmed before the
cut at alpha 0.1 with stage6_cp's cut rule. Layers are read once for all conditions
(layer-outer), as stage4_a does.

Outputs (<out> = --out-dir, default results/abc2; rows of other conditions already in a
file are kept, rows of the conditions run are replaced):
    b_final_table.csv   same columns as the pipeline's final_table.csv, detector "B H=10"
    b_final_m1.csv      as final_m1.csv;   b_cp_curves.csv  as cp_curves.csv
    b_stress.csv        as stress.csv;     b_selections.csv as selections.csv
    b_grid.csv          cond, seed, layer, target, target_series, H, c, chosen_c,
                        test_set, auroc (own-target AUROC), threshold
    b_final_grid.csv    final detector at every c (eval_seen M1-per-task, sign consistency)
    b_ablation.csv      final detector on one target kind's probes, at the final c
    scores/<cond>_seed<s>_b10.npz   the pipeline's npz format with detector key B
                        (B_m1, B_cp, B_p, B_mu, B_xi, B_h, B_band01, B_R, B_t_end,
                        B_n_d1, B_n_d2, plus the shared keys and stress cuts)
    b_status.json       {"status": done|failed|skipped, "reason", "d0_check", ...}
Work (--work-dir, default /data/tmp/abc2_scores/b; resumable, deletable after the run):
    <cond>/b_p1_seed{s}_L{l}.pkl   pass-1 AUROCs
    <cond>/b_p2_seed{s}_L{l}.npz   pass-2 probe outputs, fp16, only the rollouts the
                                   condition uses (~2.5 MB each at N = 1000)
    <cond>/b_final.pkl             final detectors, CP and table rows

D0 check (Step 11): D0's T0 M1-per-task of B H=10 must equal results/abc/b_final.csv
(H = 10, model full, per_task_unseen_mean) within 0.005. Run automatically after a full
D0 run and by --check-d0; written into b_status.json["d0_check"].

Usage (repo root, after pipeline.py has run the conditions):
    source env.sh
    ABC2_GPU_FRAC=1.0 python analysis/abc2/stage5_b.py --cond D0 D2 \
        || python analysis/abc2/stage5_b.py --mark-failed "stage5_b exited non-zero"
    python analysis/abc2/stage5_b.py --check-d0           # compare D0 with results/abc
    python analysis/abc2/stage5_b.py --check 1e-3         # GPU vs scikit-learn, one probe
Dev options: --layers, --seeds, --index, --cache-dir, --splits-dir, --out-dir,
--pipe-dir, --work-dir, --demotions; --compare-abc-grid compares <out>/b_grid.csv (D0,
H = 10) with results/abc/b_grid.csv.
GPU: torch.cuda.set_per_process_memory_fraction(ABC2_GPU_FRAC), default 6/46. At
N = 1000 D2 needs ~10 GB: run with ABC2_GPU_FRAC=1.0 on a free GPU.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import argparse  # noqa: E402
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
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import cp_eval as abc_cp  # noqa: E402
import prep  # noqa: E402
import setup_b as abc_b  # noqa: E402
import setup_c as abc_c  # noqa: E402
import stage4_a  # noqa: E402
import stage6_cp  # noqa: E402
from data2 import T, m1_cut, train_rows_and_weights  # noqa: E402

H = 10
JOINT_HS = abc_b.HS                                    # (0, 10, 25): setup_b's fitting block
JOINT_NOTE = ("each layer's probes fitted jointly with the H=0 and H=25 targets (12 outputs, "
              "setup_b's layout, warm start, max_iter and objective), only the H=10 outputs kept; "
              "the H=10-only fit (4 outputs) stops LBFGS at a different point (per-row p up to "
              "~0.004 apart), which flipped the final c on D0 and failed the Step 11 D0 check")
DET = f"B H={H}"
KEY = "B"
LAYERS = list(range(32))
C_GRID = abc_c.C_GRID
TARGET_KINDS = abc_b.TARGET_KINDS
TARGET_FAMS = {"action": ["act_rep", "act_mag"], "emb_temp": ["emb_temp"],
               "cross_layer": ["xl_final"]}            # = setup_b.prepared_targets
N_FOLDS = abc_b.N_FOLDS
ALPHAS = stage6_cp.ALPHAS                              # 15 grid levels, then 0.1
D0_REF = Path("results/abc/b_final.csv")
D0_TOL = 0.005
CHECK = {"layer": 16, "kind": "emb_temp", "seed": 0}  # setup_b.CHECK at H = 10
OUT_TABLES = {"final_m1": "b_final_m1.csv", "cp": "b_cp_curves.csv", "stress": "b_stress.csv",
              "sel": "b_selections.csv", "grid": "b_grid.csv", "final_grid": "b_final_grid.csv",
              "ablation": "b_ablation.csv"}


class Missing(Exception):
    """A prerequisite (pipeline output) is missing: the condition is skipped."""


# --- Pipeline selections ---------------------------------------------------------------
def pipeline_inputs(pipe_dir: Path, cond: str) -> dict:
    """beta per family and unstable target series, from the pipeline's outputs."""
    sel = [r for r in read_csv(pipe_dir / "selections.csv") if r["cond"] == cond]
    if not sel:
        raise Missing(f"no {cond} rows in {pipe_dir / 'selections.csv'} (run pipeline.py first)")
    betas = {}
    for fam in TARGET_FAMS:
        v = {r["value"] for r in sel if r["selection"] == f"beta_{fam}"}
        if len(v) != 1:
            raise Missing(f"{cond}: beta_{fam} missing or ambiguous in selections.csv: {v}")
        betas[fam] = v.pop()
    pipe_seeds = sorted({int(r["seed"]) for r in sel})
    s2 = read_csv(pipe_dir / "tables" / cond / "constraint_m1_summary.csv")
    if not s2:
        raise Missing(f"{pipe_dir}/tables/{cond}/constraint_m1_summary.csv missing")
    kinds = {b: f for f, bs in TARGET_FAMS.items() for b in bs}
    unstable, seen = set(), set()
    for r in s2:
        if r["variant"] != "primary" or r["constraint"] not in kinds:
            continue
        if r["beta"] != betas[kinds[r["constraint"]]]:
            continue
        name = abc_c.series_name(r["constraint"], int(r["layer"]))
        seen.add(name)
        if r["unstable"] == "True":
            unstable.add(name)
    need = {abc_b.target_name(kd, L) for L in LAYERS for kd in TARGET_KINDS}
    if need - seen:
        raise Missing(f"{cond}: Stage 2 table lacks {sorted(need - seen)[:5]}")
    return {"betas": betas, "unstable": sorted(unstable), "pipe_seeds": pipe_seeds}


def layer_targets(unstable) -> dict:
    return {L: [kd for kd in TARGET_KINDS if abc_b.target_name(kd, L) not in set(unstable)]
            for L in LAYERS}


def prepared_targets(ctx: Ctx, cond: str, seeds, betas: dict) -> dict:
    """setup_b.prepared_targets for a condition: {(seed, series): (N, 520) float32, NaN
    off kept rows}, prepared at the condition's beta, scaled / oriented on its train set."""
    S, names = ctx.series("primary")
    out = {}
    for fam, bases in TARGET_FAMS.items():
        ks = [k for k, (b, _) in enumerate(names) if b in bases]
        nm = [names[k] for k in ks]
        beta = 0.0 if betas[fam] == "none" else float(betas[fam])
        E = prep.smooth(S[ks], nm, beta)
        for s in seeds:
            mu, sd, sign, _ = prep.fit_scale_orient(E, ctx.idx, ctx.kept, ctx.sets(cond, s)["train"])
            P = prep.apply_prep(E, mu, sd, sign).astype(np.float32)
            for k, (b, L) in enumerate(nm):
                out[(s, abc_c.series_name(b, L))] = np.where(ctx.kept, P[k], np.nan)
        del E
    return out


def eval_sets(sets: dict) -> dict:
    return {k: sets[k] for k in EVAL_SETS if len(sets[k])}


def used_positions(sets: dict) -> np.ndarray:
    """Every rollout the condition fits on or evaluates (train, eval_seen, T2, stress)."""
    return np.unique(np.concatenate([sets[k] for k in ("train", "eval_seen", "T0", "T1", "T2")
                                     + STRESS_SETS]))


# --- Grid --------------------------------------------------------------------------------
class Run:
    """Per-condition state for one invocation."""

    def __init__(self, ctx: Ctx, cond: str, seeds, layers, inputs: dict, joint_h: bool = True):
        self.ctx, self.cond, self.seeds, self.layers = ctx, cond, list(seeds), list(layers)
        self.joint_h = bool(joint_h)
        self.fit_hs = list(JOINT_HS) if self.joint_h else [H]
        self.betas, self.unstable = inputs["betas"], inputs["unstable"]
        self.pipe_seeds = inputs["pipe_seeds"]
        self.kinds = layer_targets(self.unstable)
        self.sets = {s: ctx.sets(cond, s) for s in self.seeds}
        self._targets = None
        self.c_grid = None
        self.c_grid_score = None

    @property
    def targets(self):
        if self._targets is None:
            t0 = time.time()
            self._targets = prepared_targets(self.ctx, self.cond, self.seeds, self.betas)
            log(f"[b {self.cond}] targets prepared ({time.time() - t0:.0f} s)")
        return self._targets

    def ys(self, s, L):
        """The fitting block's look-ahead targets in setup_b.run_grid's order (target-major,
        then H over self.fit_hs), and the block columns of the H = 10 outputs (one per
        target in self.kinds[L])."""
        ys, cols = [], []
        for kd in self.kinds[L]:
            P = self.targets[(s, abc_b.target_name(kd, L))]
            for hh in self.fit_hs:
                if hh == H:
                    cols.append(len(ys))
                ys.append(abc_b.lookahead(P, hh))
        return ys, cols

    def p1_path(self, s, L):
        return self.ctx.wpath(self.cond, f"b_p1_seed{s}_L{L:02d}.pkl")

    def p2_path(self, s, L):
        return self.ctx.wpath(self.cond, f"b_p2_seed{s}_L{L:02d}.npz")

    def p1_ok(self, s, L):
        p = self.p1_path(s, L)
        if not p.exists():
            return False
        w = load_pkl(p)
        return (w["betas"] == self.betas and w["kinds"] == self.kinds[L]
                and w.get("fit_hs") == self.fit_hs)

    def p2_ok(self, s, L):
        p = self.p2_path(s, L)
        if not p.exists():
            return False
        z = np.load(p)
        return (float(z["c"]) == self.c_grid and list(z["kinds"]) == self.kinds[L]
                and str(z["betas"]) == json.dumps(self.betas, sort_keys=True)
                and "fit_hs" in z and [int(h) for h in z["fit_hs"]] == self.fit_hs)

    def pass1(self, s, L, Xall):
        """Refit on all kept train rows at every c; own-target AUROC on each eval set."""
        idx, kept = self.ctx.idx, self.ctx.kept
        sets = self.sets[s]
        tr = sets["train"]
        rr = np.nonzero(kept[tr])
        ys, cols = self.ys(s, L)
        fits, thr, mu, sd = abc_b.fit_grid_block(Xall, (tr[rr[0]], rr[1]), ys, list(C_GRID))
        evs = eval_sets(sets)
        ev = np.unique(np.concatenate(list(evs.values())))
        auroc = {}
        for c in C_GRID:
            full = np.zeros((len(cols), idx.n, T), dtype=np.float32)
            full[:, ev] = abc_b.predict_block(Xall, ev, *fits[c], mu, sd)[cols]   # H = 10 only
            auroc[c] = {k: [abc_b.target_auroc(full[i], ys[j], thr[j], kept, pos)
                            for i, j in enumerate(cols)] for k, pos in evs.items()}
        save_pkl(self.p1_path(s, L), {"cond": self.cond, "seed": s, "layer": L,
                                      "betas": self.betas, "kinds": self.kinds[L],
                                      "fit_hs": self.fit_hs,
                                      "thr": thr[cols].tolist(), "auroc": auroc})

    def choose_c(self):
        """B's grid c: best mean over seeds of the mean over probes of eval_seen target
        AUROC (setup_b.run_grid, restricted to H = 10)."""
        W = {(s, L): load_pkl(self.p1_path(s, L)) for s in self.seeds for L in self.layers}
        score = {c: np.array([np.nanmean([a for L in self.layers
                                          for a in W[(s, L)]["auroc"][c]["eval_seen"]])
                              for s in self.seeds]) for c in C_GRID}
        self.c_grid = max(C_GRID, key=lambda c: score[c].mean())
        self.c_grid_score = {c: (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
                             for c, v in score.items()}
        rows = []
        for (s, L), w in W.items():
            for c in C_GRID:
                for ts, au in w["auroc"][c].items():
                    for j, kd in enumerate(w["kinds"]):
                        rows.append({"cond": self.cond, "seed": s, "layer": L, "target": kd,
                                     "target_series": abc_b.target_name(kd, L), "H": H, "c": c,
                                     "chosen_c": c == self.c_grid, "test_set": ts,
                                     "auroc": au[j], "threshold": w["thr"][j]})
        self.grid_rows = rows
        log(f"[b {self.cond}] grid c = {self.c_grid:g} (eval_seen target AUROC "
            + ", ".join(f"{c:g}: {v[0]:.4f}" for c, v in self.c_grid_score.items()) + ")")

    def pass2(self, s, L, Xall):
        """At the grid c: out-of-fold on train, refit on all train for the rest."""
        kept = self.ctx.kept
        sets = self.sets[s]
        tr = sets["train"]
        need = used_positions(sets)
        ys, cols = self.ys(s, L)
        c = self.c_grid
        out = np.zeros((len(cols), self.ctx.idx.n, T), dtype=np.float32)   # H = 10 only
        for f_tr, f_te in GroupKFold(n_splits=N_FOLDS).split(tr, groups=tr):
            ftr = tr[f_tr]
            rr = np.nonzero(kept[ftr])
            fits, thr, mu, sd = abc_b.fit_grid_block(Xall, (ftr[rr[0]], rr[1]), ys, [c])
            out[:, tr[f_te]] = abc_b.predict_block(Xall, tr[f_te], *fits[c], mu, sd)[cols]
        rr = np.nonzero(kept[tr])
        fits, thr, mu, sd = abc_b.fit_grid_block(Xall, (tr[rr[0]], rr[1]), ys, [c])
        rest = np.setdiff1d(need, tr)
        out[:, rest] = abc_b.predict_block(Xall, rest, *fits[c], mu, sd)[cols]
        p = self.p2_path(s, L)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, p=out[:, need].astype(np.float16), pos=need, c=self.c_grid,
                 kinds=np.array(self.kinds[L]), betas=json.dumps(self.betas, sort_keys=True),
                 fit_hs=np.array(self.fit_hs), rids=self.ctx.idx.rids[need])
        tmp.replace(p)

    # --- Final detector, CP, tables ------------------------------------------------------
    def features(self, s) -> tuple[list, dict]:
        """{probe name: (N, 520) float32}: pass-2 outputs (fp16, as setup_b stored them);
        rollouts the condition never uses are 0 (they are never fitted or scored)."""
        names, feats = [], {}
        for L in self.layers:
            z = np.load(self.p2_path(s, L))
            pos = z["pos"]
            for j, kd in enumerate(z["kinds"]):
                nm = abc_b.probe_name(L, str(kd), H)
                f = np.zeros((self.ctx.idx.n, T), dtype=np.float32)
                f[pos] = z["p"][j].astype(np.float32)
                names.append(nm)
                feats[nm] = f
        return names, feats

    def final_key(self):
        return {"c_grid": self.c_grid, "betas": self.betas, "kinds": self.kinds,
                "seeds": self.seeds, "layers": self.layers, "fit_hs": self.fit_hs}

    def final(self, n_jobs: int) -> dict:
        path = self.ctx.wpath(self.cond, "b_final.pkl")
        if path.exists():
            w = load_pkl(path)
            if w["key"] == self.final_key():
                log(f"[b {self.cond}] final detector: reusing {path}")
                return w
        t0 = time.time()
        idx, kept = self.ctx.idx, self.ctx.kept
        feats, trains, evals = {}, {}, {}
        for s in self.seeds:
            names, feats[s] = self.features(s)
            trains[s] = train_rows_and_weights(idx, kept, self.sets[s]["train"])
            evals[s] = eval_sets(self.sets[s])
        log(f"[b {self.cond}] final detector: {len(names)} inputs; fitting "
            f"{len(self.seeds) * len(C_GRID)} (seed, c) jobs")
        jobs = Parallel(n_jobs=n_jobs)(
            delayed(final_job)(s, c, names, feats[s], idx, kept, evals[s], trains[s])
            for s in self.seeds for c in C_GRID)
        res = {(j["seed"], j["c"]): j for j in jobs}

        def seen(c):
            return np.array([res[(s, c)]["metrics"]["eval_seen"]["m1_per_task"] for s in self.seeds])

        c_final = max(C_GRID, key=lambda c: seen(c).mean())
        final_grid = []
        for c in C_GRID:
            cs = np.array([res[(s, c)]["coef"] for s in self.seeds])
            mu_, sd_ = mean_std(seen(c))
            final_grid.append({"cond": self.cond, "c": c, "chosen": c == c_final,
                               "final_seen_task_mean": mu_, "final_seen_task_std": sd_,
                               "sign_consistent": int(np.sum(np.all(cs > 0, 0) | np.all(cs < 0, 0))),
                               "n_inputs": len(names),
                               "converged": all(res[(s, c)]["converged"] for s in self.seeds)})
        abl = Parallel(n_jobs=n_jobs)(
            delayed(final_job)(s, c_final, [n for n in names if n.split(":")[1] == kd],
                               feats[s], idx, kept, evals[s], trains[s], keep_p=False)
            for s in self.seeds for kd in TARGET_KINDS)
        ablation = []
        for j in abl:
            for ts, m in j["metrics"].items():
                ablation.append({"cond": self.cond, "seed": j["seed"],
                                 "model": f"only_{j['names'][0].split(':')[1]}", "c": c_final,
                                 "n_inputs": len(j["names"]), "test_set": ts, **m})
        out = {"key": self.final_key(), "names": names, "c_final": c_final,
               "c_final_score": float(seen(c_final).mean()), "final_grid": final_grid,
               "ablation": ablation,
               "p": {s: res[(s, c_final)]["p"] for s in self.seeds},
               "coef": {s: res[(s, c_final)]["coef"] for s in self.seeds},
               "unconverged": sorted({c for (s, c), j in res.items() if not j["converged"]})}
        save_pkl(path, out)
        log(f"[b {self.cond}] final c = {c_final:g} (eval_seen M1-per-task "
            f"{out['c_final_score']:.4f}); unconverged c: {out['unconverged']} ({time.time() - t0:.0f} s)")
        return out

    def selections(self, fin) -> dict:
        skipped = sorted({abc_b.target_name(kd, L) for L in self.layers for kd in TARGET_KINDS
                          if kd not in self.kinds[L]})
        return {"H": H, "betas": self.betas, "skipped_unstable_targets": skipped,
                "c_grid": self.c_grid, "c_grid_score": self.c_grid_score[self.c_grid][0],
                "c_final": fin["c_final"], "c_final_score": fin["c_final_score"],
                "n_probes": len(fin["names"]), "layers": self.layers, "seeds": self.seeds,
                "pipe_seeds": self.pipe_seeds}

    def cp_and_tables(self, fin) -> dict:
        """CP, M1 and stress rows per seed (stage6_cp.run_seed for one detector) and the
        per-seed npz."""
        idx, kept, ctx = self.ctx.idx, self.ctx.kept, self.ctx
        sel = self.selections(fin)
        detail = f"H={H} grid c={self.c_grid:g} final c={fin['c_final']:g} ({len(fin['names'])} probes)"
        m1_rows, cp_rows, stress_rows, sel_rows = [], [], [], []
        for s in self.seeds:
            sets = self.sets[s]
            p = fin["p"][s]
            sc = abc_c.running_sum(p, kept).astype(np.float32)
            evals = eval_sets(sets)
            cuts = {k: m1_cut(idx, pos) for k, pos in evals.items()}
            cal = sets["cal"]
            mu, xi, R, t_end, n1, n2 = abc_cp.fit_band(
                sc, kept, cal, np.random.default_rng(stage6_cp.CAL_SPLIT_SEED + s))
            hs = np.array([abc_cp.band_h(R, a) for a in ALPHAS])
            head = {"cond": self.cond, "seed": s, "detector": DET, "detail": detail}
            for ts, pos in evals.items():
                m1_rows.append({**head, "test_set": ts, **m1_row(m1_safe(sc, kept, idx, pos))})
                for a, h in zip(ALPHAS, hs):
                    r = abc_cp.evaluate_cp(sc, kept, idx, pos, mu, xi, h, cuts[ts])
                    cp_rows.append({**head, "test_set": ts, "alpha": float(a), "variant": "cut",
                                    "on_alpha_grid": bool(a in abc_cp.ALPHAS), "h": h,
                                    "bal_acc": r["bal_acc"], "tpr": r["tpr"], "tnr": r["tnr"],
                                    "far": 1 - r["tnr"], "det_time": r["det_time"],
                                    "n": int(len(pos)), "n_fail": int(idx.y[pos].sum()),
                                    "n_cal": int(len(cal)), "n_d1": n1, "n_d2": n2, "t_end": t_end})
            npz = {"rollout_ids": idx.rids, "cond": self.cond, "seed": s,
                   "alphas": np.array(ALPHAS), "kept_len": idx.kept_len, "y": idx.y,
                   "cell": idx.task, "base_task": idx.base_task, "block": idx.block,
                   "cal_rids": idx.rids[cal], "train_rids": idx.rids[sets["train"]],
                   "eval_seen_rids": idx.rids[sets["eval_seen"]],
                   f"{KEY}_m1": sc, f"{KEY}_cp": sc, f"{KEY}_p": p.astype(np.float32),
                   f"{KEY}_mu": mu, f"{KEY}_xi": xi, f"{KEY}_h": hs,
                   f"{KEY}_band01": mu + hs[-1] * xi, f"{KEY}_R": R, f"{KEY}_t_end": t_end,
                   f"{KEY}_n_d1": n1, f"{KEY}_n_d2": n2}
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
            npz["selections"] = json.dumps(sel)
            out = ctx.scores_dir / f"{self.cond}_seed{s}_b{H}.npz"
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez_compressed(tmp, **npz)
            tmp.replace(out)

            def add(name, value, crit, score):
                sel_rows.append({"cond": self.cond, "seed": s, "selection": name, "value": value,
                                 "criterion": crit, "score_eval_seen": score})
            for fam, b in self.betas.items():
                add(f"B_beta_{fam}", b, "pipeline selections.csv (this condition)", "")
            add("B_skipped_unstable_targets", " ".join(sel["skipped_unstable_targets"]),
                "unstable at the chosen beta in the pipeline's constraint_m1_summary.csv", "")
            add("c_B_grid", self.c_grid, "mean over seeds of the probe-mean eval_seen own-target AUROC",
                self.c_grid_score[self.c_grid][0])
            add("c_B_final", fin["c_final"], "mean over seeds of eval_seen M1-per-task",
                fin["c_final_score"])
            add("B_n_probes", len(fin["names"]), f"H={H}, layers {self.layers[0]}-{self.layers[-1]}"
                f" ({len(self.layers)})", "")
        return {"final_m1": m1_rows, "cp": cp_rows, "stress": stress_rows, "sel": sel_rows,
                "grid": self.grid_rows, "final_grid": fin["final_grid"],
                "ablation": fin["ablation"]}


def final_job(s, c, names, feats, idx, kept, evals, train, keep_p=True):
    model, p, ok = abc_c.fit_prob(names, feats, train, c)
    sc = abc_c.running_sum(p, kept)
    return {"seed": s, "c": c, "names": names, "converged": ok,
            "metrics": {k: m1_row(m1_safe(sc, kept, idx, pos)) for k, pos in evals.items()},
            "coef": model[-1].coef_[0].tolist(), "p": p if keep_p else None}


# --- Output assembly ---------------------------------------------------------------------
def merge_csv(path: Path, rows: list[dict], conds) -> list[dict]:
    """Keep rows of other conditions already in `path`, replace those of `conds`."""
    old = [r for r in read_csv(path) if r.get("cond") not in set(conds)]
    allr = old + rows
    write_csv(path, allr)
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
    write_csv(out_dir / "b_final_table.csv", final_table(merged["final_m1"], merged["cp"]))


# --- Status and the D0 check --------------------------------------------------------------
def write_status(out_dir: Path, status: str, reason: str, **extra) -> dict:
    p = out_dir / "b_status.json"
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        d = {}
    d.update({"status": status, "reason": reason,
              "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, **extra)
    d.setdefault("d0_check", {})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1, default=str))
    return d


def check_d0(out_dir: Path) -> dict:
    """D0 / T0 M1-per-task of B H=10 vs results/abc/b_final.csv (H=10, full, unseen)."""
    ref = [r for r in read_csv(D0_REF) if r["H"] == str(H) and r["model"] == "full"]
    new = [r for r in read_csv(out_dir / "b_final_table.csv")
           if r["cond"] == "D0" and r["test_set"] == "T0" and r["detector"] == DET]
    if not ref or not new:
        d = {"passed": False, "error": f"missing rows (ref {len(ref)}, D0/T0 {len(new)})"}
    else:
        r, n = ref[0], new[0]
        sel = {x["selection"]: x["value"] for x in read_csv(out_dir / "b_selections.csv")
               if x["cond"] == "D0"}
        diff = float(n["m1_per_task_mean"]) - float(r["per_task_unseen_mean"])
        d = {"quantity": "B H=10 final detector, unseen (T0) M1-per-task, mean over seeds",
             "reference": float(r["per_task_unseen_mean"]),
             "reference_std": float(r["per_task_unseen_std"]),
             "reproduced": float(n["m1_per_task_mean"]), "reproduced_std": float(n["m1_per_task_std"]),
             "diff": diff, "tolerance": D0_TOL, "passed": abs(diff) <= D0_TOL,
             "n_seeds": int(n["n_seeds"]),
             "reference_pooled": float(r["pooled_unseen_mean"]),
             "reproduced_pooled": float(n["m1_pooled_mean"]),
             "reference_c_final": float(r["c"]), "c_final": sel.get("c_B_final"),
             "reference_c_grid": 1e-4, "c_grid": sel.get("c_B_grid")}
    st0 = json.loads((out_dir / "b_status.json").read_text()) if (out_dir / "b_status.json").exists() else {}
    d["fit"] = st0.get("fit", {"joint_h": False, "note": "H=10 outputs fitted alone"})
    print(f"[D0 check] B H=10 T0 M1-per-task: reproduced {d.get('reproduced', float('nan')):.4f} "
          f"vs results/abc {d.get('reference', float('nan')):.4f} (diff {d.get('diff', float('nan')):+.4f}, "
          f"tolerance {D0_TOL}): {'PASSED' if d['passed'] else 'FAILED'}"
          + (f" [{d['error']}]" if "error" in d else ""), flush=True)
    st = json.loads((out_dir / "b_status.json").read_text()) if (out_dir / "b_status.json").exists() else {}
    write_status(out_dir, st.get("status", "done"), st.get("reason", ""), d0_check=d)
    return d


def compare_abc_grid(out_dir: Path) -> None:
    """Dev check: <out>/b_grid.csv (D0, H=10) against results/abc/b_grid.csv, per probe
    and seed: eval_seen vs abc's seen, T0 vs abc's unseen."""
    ref = {(int(r["layer"]), r["target"], float(r["c"])): r for r in read_csv(Path("results/abc/b_grid.csv"))
           if r["H"] == str(H)}
    mine = [r for r in read_csv(out_dir / "b_grid.csv") if r["cond"] == "D0"]
    diffs = {"eval_seen": [], "T0": []}
    for r in mine:
        lab = {"eval_seen": "seen", "T0": "unseen"}.get(r["test_set"])
        if lab is None:
            continue
        k = (int(r["layer"]), r["target"], float(r["c"]))
        if k not in ref:
            continue
        diffs[r["test_set"]].append((abs(float(r["auroc"]) - float(ref[k][f"auroc_{lab}_s{r['seed']}"])),
                                     k, int(r["seed"])))
    for ts, v in diffs.items():
        if not v:
            print(f"[compare] {ts}: no matching rows")
            continue
        a = np.array([x[0] for x in v])
        at = np.array([x[0] for x in v if x[1][2] == 1e-4])
        worst = max(v)
        print(f"[compare] {ts}: {len(a)} (probe, c, seed) rows; max |diff| {a.max():.5f}, mean "
              f"{a.mean():.5f}; at c=1e-4: max {at.max() if len(at) else float('nan'):.5f} "
              f"(abc rounds to 4 d.p.); worst {worst[1]} seed {worst[2]}")


# --- --check: GPU vs scikit-learn on one probe --------------------------------------------
def check(run: Run, c: float, cache_dir: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    L, s = CHECK["layer"], CHECK["seed"]
    kept, idx = run.ctx.kept, run.ctx.idx
    sets = run.sets[s]
    tr = sets["train"]
    rr = np.nonzero(kept[tr])
    rows = (tr[rr[0]], rr[1])
    ys, cols = run.ys(s, L)
    j = cols[run.kinds[L].index(CHECK["kind"])]
    Xall = stage4_a.load_probe_input(f"layer{L:02d}", cache_dir)
    t0 = time.time()
    fits, thr, mu, sd = abc_b.fit_grid_block(Xall, rows, ys, [c])
    up = sets["T0"]
    full = np.zeros((idx.n, T), dtype=np.float32)
    full[up] = abc_b.predict_block(Xall, up, *fits[c], mu, sd)[j]
    t_gpu = time.time() - t0
    a_gpu = abc_b.target_auroc(full, ys[j], thr[j], kept, up)

    y = (ys[j][rows] > thr[j]).astype(int)
    Xtr = Xall[torch.from_numpy(rows[0]).to(abc_b.DEV), torch.from_numpy(rows[1]).to(abc_b.DEV)].float().cpu().numpy()
    t0 = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
    model.fit(Xtr, y)
    t_sk = time.time() - t0
    del Xtr
    Xu = Xall[torch.from_numpy(up).to(abc_b.DEV)].float().cpu().numpy()
    psk = np.zeros((idx.n, T), dtype=np.float32)
    psk[up] = model.predict_proba(Xu.reshape(-1, Xu.shape[-1]))[:, 1].reshape(len(up), T)
    a_sk = abc_b.target_auroc(psk, ys[j], thr[j], kept, up)
    ok = abs(a_gpu - a_sk) <= abc_b.CHECK_TOL
    print(f"[check] {run.cond} layer {L}, target {abc_b.target_name(CHECK['kind'], L)}, H = {H}, "
          f"seed {s}, c = {c:g} (threshold {thr[j]:.3f}, train positives {y.mean():.3f}, "
          f"{len(y)} rows); GPU {t_gpu:.1f} s ({len(ys)} probes jointly), scikit-learn {t_sk:.1f} s")
    print(f"   T0 target AUROC  GPU {a_gpu:.4f}  sklearn {a_sk:.4f}  diff {abs(a_gpu - a_sk):.4f}")
    print(f"[check] {'PASSED' if ok else 'FAILED'} (tolerance {abc_b.CHECK_TOL})", flush=True)


# --- Main ----------------------------------------------------------------------------------
def run_all(args, ctx: Ctx, runs: dict) -> dict:
    t_all = time.time()
    # Pass 1, layer-outer over every condition.
    todo1 = {L: [(c, s) for c, r in runs.items() for s in r.seeds if L in r.layers and not r.p1_ok(s, L)]
             for L in args.layers}
    if any(todo1.values()):
        stage4_a.gpu_setup()
        stage4_a.check_cache(ctx)
    t0 = time.time()
    for k, L in enumerate(args.layers):
        if not todo1[L]:
            continue
        X = stage4_a.load_probe_input(f"layer{L:02d}", ctx.cache_dir)
        for cond, s in todo1[L]:
            runs[cond].pass1(s, L, X)
        del X
        torch.cuda.empty_cache()
        log(f"[b pass 1 {k + 1:2d}/{len(args.layers)}] layer {L}: {len(todo1[L])} (cond, seed) "
            f"({time.time() - t0:.0f} s)")
    for r in runs.values():
        r.choose_c()
    # Pass 2 at each condition's grid c.
    todo2 = {L: [(c, s) for c, r in runs.items() for s in r.seeds if L in r.layers and not r.p2_ok(s, L)]
             for L in args.layers}
    if any(todo2.values()):
        stage4_a.gpu_setup()
        stage4_a.check_cache(ctx)
    t0 = time.time()
    for k, L in enumerate(args.layers):
        if not todo2[L]:
            continue
        X = stage4_a.load_probe_input(f"layer{L:02d}", ctx.cache_dir)
        for cond, s in todo2[L]:
            runs[cond].pass2(s, L, X)
        del X
        torch.cuda.empty_cache()
        log(f"[b pass 2 {k + 1:2d}/{len(args.layers)}] layer {L}: {len(todo2[L])} (cond, seed) "
            f"({time.time() - t0:.0f} s)")
    results = {}
    for cond, r in runs.items():
        r.fin = r.final(args.n_jobs)
        results[cond] = r.cp_and_tables(r.fin)
    write_outputs(args.out_dir, results)
    for row in final_table(read_csv(args.out_dir / "b_final_m1.csv"), read_csv(args.out_dir / "b_cp_curves.csv")):
        if row["cond"] in runs:
            log(f"   {row['cond']} {row['test_set']:9s} M1-per-task {row['m1_per_task_mean']:.4f}"
                f"±{row['m1_per_task_std']:.4f}  pooled {row['m1_pooled_mean']:.4f}  "
                f"BA@0.1 {row['a01_bal_acc_mean']:.4f}  TPR {row['a01_tpr_mean']:.3f}  "
                f"FAR {row['a01_far_mean']:.3f}  cells {row['n_cells']}")
    log(f"[b] done ({time.time() - t_all:.0f} s)")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cond", nargs="+", default=["D0", "D2"], choices=list(data2.CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(data2.SEEDS))
    ap.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    ap.add_argument("--index", type=Path, default=data2.INDEX_V3)
    ap.add_argument("--cache-dir", type=Path, default=data2.CACHE_DIR)
    ap.add_argument("--splits-dir", type=Path, default=data2.SPLITS_DIR)
    ap.add_argument("--out-dir", type=Path, default=data2.RESULTS_DIR)
    ap.add_argument("--pipe-dir", type=Path, default=None,
                    help="where the pipeline's selections.csv and tables/ are (default --out-dir)")
    ap.add_argument("--work-dir", type=Path, default=Path("/data/tmp/abc2_scores/b"))
    ap.add_argument("--demotions", type=Path, default=data2.DEMOTIONS)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--joint-h", action=argparse.BooleanOptionalAction, default=True,
                    help="fit each layer's probes jointly with the H=0/25 targets as setup_b did, "
                         "keep only H=10 (default on; see the docstring)")
    ap.add_argument("--check", type=float, default=None, metavar="C",
                    help="GPU vs scikit-learn on one probe (first --cond), writes nothing")
    ap.add_argument("--check-d0", action="store_true", help="compare D0 with results/abc/b_final.csv")
    ap.add_argument("--compare-abc-grid", action="store_true",
                    help="dev: compare <out>/b_grid.csv (D0) with results/abc/b_grid.csv")
    ap.add_argument("--mark-failed", metavar="REASON", default=None)
    ap.add_argument("--mark-skipped", metavar="REASON", default=None)
    args = ap.parse_args(argv)
    args.pipe_dir = args.pipe_dir or args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mark_failed is not None or args.mark_skipped is not None:
        st = args.out_dir / "b_status.json"
        cur = json.loads(st.read_text()) if st.exists() else {}
        if args.mark_skipped is not None:
            write_status(args.out_dir, "skipped", args.mark_skipped)
        elif cur.get("status") != "failed":        # keep a more specific failure reason
            write_status(args.out_dir, "failed", args.mark_failed)
        return 0
    if args.check_d0:
        return 0 if check_d0(args.out_dir)["passed"] else 1
    if args.compare_abc_grid:
        compare_abc_grid(args.out_dir)
        return 0

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 3)))
    t0 = time.time()
    try:
        ctx = Ctx(index_path=args.index, cache_dir=args.cache_dir, splits_dir=args.splits_dir,
                  out_dir=args.out_dir, work_dir=args.work_dir, demotions_path=args.demotions,
                  seeds=tuple(args.seeds), n_jobs=args.n_jobs)
        runs, skipped = {}, {}
        for cond in args.cond:
            try:
                inp = pipeline_inputs(args.pipe_dir, cond)
            except Missing as e:
                skipped[cond] = str(e)
                log(f"[b {cond}] SKIPPED: {e}")
                continue
            if not set(args.seeds) <= set(inp["pipe_seeds"]):
                log(f"[b {cond}] warning: seeds {args.seeds} not all in the pipeline's {inp['pipe_seeds']}")
            runs[cond] = Run(ctx, cond, args.seeds, args.layers, inp, joint_h=args.joint_h)
            log(f"[b {cond}] betas {inp['betas']}; skipped unstable targets "
                f"{sorted(set(inp['unstable']) & {abc_b.target_name(kd, L) for L in args.layers for kd in TARGET_KINDS})}; "
                f"{sum(len(runs[cond].kinds[L]) for L in args.layers)} probes")
        if args.check is not None:
            if not runs:
                raise SystemExit(f"no condition available: {skipped}")
            stage4_a.gpu_setup()
            check(next(iter(runs.values())), args.check, args.cache_dir)
            return 0
        if not runs:
            write_status(args.out_dir, "skipped", "; ".join(f"{c}: {r}" for c, r in skipped.items()))
            return 0
        log(f"index {args.index}: {ctx.idx.n} rollouts; conds {list(runs)}; seeds {args.seeds}; "
            f"layers {args.layers[0]}..{args.layers[-1]} ({len(args.layers)}); out {args.out_dir}; "
            f"work {args.work_dir}; fit block H {list(JOINT_HS) if args.joint_h else [H]} "
            f"(only H={H} kept)")
        run_all(args, ctx, runs)
        st = args.out_dir / "b_status.json"
        prev = json.loads(st.read_text()) if st.exists() else {}
        done = sorted(set(prev.get("conds_done", [])) | set(runs))
        full = len(args.layers) == len(LAYERS) and sorted(args.seeds) == list(data2.SEEDS)
        reason = "" if not skipped else "skipped: " + "; ".join(f"{c}: {r}" for c, r in skipped.items())
        write_status(args.out_dir, "done", reason, conds_done=done, conds_skipped=skipped,
                     full_run=full, seconds=round(time.time() - t0),
                     fit={"joint_h": args.joint_h,
                          "fit_block_H": list(JOINT_HS) if args.joint_h else [H],
                          "kept_H": H, "note": JOINT_NOTE if args.joint_h else
                          "H=10 outputs fitted alone (4 outputs per layer)"},
                     selections={c: r.selections(r.fin) for c, r in runs.items()})
        if "D0" in runs and full:
            check_d0(args.out_dir)
        return 0
    except (Exception, SystemExit) as e:  # B must never block the report
        traceback.print_exc()
        write_status(args.out_dir, "failed", f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
