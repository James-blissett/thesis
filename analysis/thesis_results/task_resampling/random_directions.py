"""
random_directions.py

§6 of task_resampling_handoff.md: random directions through the hidden state as a control
for B's probes (headline setup: full training set, H = 10 for B; no refits).

B's probe input, read from the code (probes_b.run_layer -> setup_b.fit_grid_block /
predict_block, common.load_layer -> stage4_a.load_probe_input):
  * the hidden state of layer L is the stored feature cache layer<L>.npy, which holds,
    per policy step, the mean over the 7 decision states (the action-token positions;
    abc2/build_cache2.py), fp16;
  * it is standardised per dimension with the mean and the (biased) standard deviation
    over the kept rows of that split's training rollouts (sd = 0 -> 1), fitted once on all
    training rows for the refit that scores eval-seen and unseen rollouts;
  * a probe's per-step output is sigmoid(w . x_t + b); the rollout score is the running sum
    over kept rows, read at its maximum before the cut (= the sum to the cut, as the
    outputs are positive). The cut is the shortest kept length in the rollout's
    task-and-perturbation group within the test set.

Random directions, per layer and split: 200 unit directions d (standard normal, scaled to
length 1, one draw for all layers and splits, seed 20261102); per-step score x_t . d on the
same standardised x_t; rollout score = sum over kept rows before the cut (the running sum
read at the cut; the projection can be negative, so its running maximum would not be the
sum to the cut, and the handoff asks for the sum). Since x . d is linear, the sum is
computed as ((sum_t x_t) - n mu) / sd . d. Per-task AUROC (groups with >= 2 of each class)
on eval-seen and on the unseen tasks incl. LIBERO-PRO. Sign set on eval-seen: if a
direction's eval-seen per-task AUROC is below 0.5 in a split, it is flipped in that split
(its unseen per-task AUROC becomes exactly 1 - AUROC: every rollout of a group has the
same number of rows before the cut).

Steps:
  directions  GPU, resumable per layer -> /data/tmp/thesis_results_work/task_resampling/random_dirs/L<ll>.npz
  check       the scorer applied to B's saved layer-0 probe outputs must reproduce their
              per-task AUROC in figures/data/per_probe_outcome_auroc.csv
  agreement   Spearman between B's layer-L probes and A's layer-L probe rollout scores,
              within each group of the unseen tasks incl. LIBERO-PRO, averaged over groups,
              then over splits (L = 0, 1, 10, 26)
  table       -> task_resampling/random_directions.csv, agreement_with_a.csv
  figure      F8 redrawn with the band (old one kept as F8_per_probe_detector_b_noband.*)

Usage: source env.sh; THESIS_POOL=D2 python analysis/thesis_results/task_resampling/random_directions.py [steps]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr  # noqa: E402
from tr import OUT, SEEDS, WORK, common, log, read_csv, write_csv  # noqa: E402

N_DIR = 200
DIR_SEED = 20261102
LAYERS = list(range(32))
RD = WORK / "random_dirs"
KINDS = ("act_rep", "act_mag", "emb_temp", "xl_final")
AGREE_LAYERS = (0, 1, 10, 26)
FIG = tr.fc.OUT / "figures"


def directions() -> np.ndarray:
    p = RD / "directions.npy"
    if p.exists():
        return np.load(p)
    D = np.random.default_rng(DIR_SEED).standard_normal((N_DIR, 4096))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    RD.mkdir(parents=True, exist_ok=True)
    np.save(p, D)
    return D


def per_task_auroc(S: np.ndarray, y: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Per-task AUROC of rollout scores S (n, k) for every column: mean over groups with
    >= 2 of each class (data.m1's per_task)."""
    out, q = np.zeros(S.shape[1]), 0
    for c in np.unique(cell):
        m = cell == c
        yy = y[m]
        n1 = int(yy.sum())
        n0 = len(yy) - n1
        if min(n0, n1) < tr.MIN_PER_CLASS:
            continue
        r = rankdata(S[m], axis=0, method="average")
        out += (r[yy == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        q += 1
    return out / q


def row_mask(pos: np.ndarray) -> np.ndarray:
    """Kept rows before the cut, (len(pos), T), as data.m1."""
    ctx = common.ctx()
    cut = common.m1_cut(ctx.idx, pos)
    return ctx.kept[pos] & (np.arange(common.T)[None, :] < cut[:, None])


# --- GPU --------------------------------------------------------------------------------
def run_directions() -> None:
    import torch
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    D = torch.from_numpy(directions().T.astype(np.float32)).cuda()             # (4096, 200)
    for L in LAYERS:
        out = RD / f"L{L:02d}.npz"
        if out.exists():
            continue
        X = common.load_layer(L)                                                 # (N, 520, 4096) fp16, GPU
        res = {}
        for s in SEEDS:
            st = common.sets(s)
            trn = st["train"]
            rr = np.nonzero(kept[trn])
            rt = (torch.from_numpy(trn[rr[0]]).cuda(), torch.from_numpy(rr[1]).cuda())
            Xt = X[rt].float()
            mu = Xt.mean(0)
            sd = Xt.std(0, unbiased=False)
            sd = torch.where(sd > 0, sd, torch.ones_like(sd))
            del Xt
            for name in ("eval_seen", "T2"):
                pos = st[name]
                M = torch.from_numpy(row_mask(pos).astype(np.float32)).cuda()   # (n, T)
                xs = torch.zeros((len(pos), X.shape[2]), device="cuda")
                for i in range(0, len(pos), 64):
                    sl = torch.from_numpy(pos[i:i + 64]).cuda()
                    xs[i:i + 64] = torch.einsum("nt,ntd->nd", M[i:i + 64], X[sl].float())
                S = (((xs - M.sum(1, keepdim=True) * mu) / sd) @ D).cpu().numpy().astype(np.float64)
                res[(s, name)] = per_task_auroc(S, idx.y[pos], idx.task[pos])
            del mu, sd
        np.savez(out, auc_seen=np.stack([res[(s, "eval_seen")] for s in SEEDS]),
                 auc_unseen=np.stack([res[(s, "T2")] for s in SEEDS]), seeds=np.array(SEEDS), layer=L)
        del X
        torch.cuda.empty_cache()
        log(f"[random] layer {L}: median unseen per-task AUROC (unflipped) "
            f"{np.median(np.stack([res[(s, 'T2')] for s in SEEDS]).mean(0)):.3f}")


# --- probe outputs ----------------------------------------------------------------------
def rollout_scores(p: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Running sum of per-step outputs (k, N, T) read at the cut -> (len(pos), k)."""
    m = row_mask(pos)
    return np.einsum("nt,knt->nk", m.astype(np.float32), np.nan_to_num(p[:, pos]))


def b_layer(L: int, s: int, H: int = 10):
    z = np.load(common.WORK_DIR / "b_probes" / f"L{L:02d}.npz")
    hs = [int(h) for h in z["hs"]]
    j = list(SEEDS).index(s)
    n = common.ctx().idx.n
    names, arrs = [], []
    for i, kd in enumerate(z["kinds"]):
        f = np.zeros((n, common.T), np.float32)
        f[z["pos"]] = z["p"][j, i * len(hs) + hs.index(H)].astype(np.float32)
        names.append(str(kd))
        arrs.append(f)
    return names, np.stack(arrs)


def a_layer(L: int, s: int) -> np.ndarray:
    z = np.load(common.WORK_DIR / "a_probes" / f"layer{L:02d}.npz")
    return z["p_refit"][list(SEEDS).index(s)].astype(np.float32)[None]


def check() -> None:
    ref = {}
    for r in read_csv(tr.fc.OUT / "figures/data/per_probe_outcome_auroc.csv"):
        if r["detector"] == "B" and r["series"] == "probe" and r["H"] == "10" and r["layer"] == "0" \
                and r["test set"] == common.TEST_SETS["T2"]:
            ref[(r["target"].split("[")[0], int(r["seed"]))] = float(r["per-task AUROC"])
    worst = 0.0
    ctx = common.ctx()
    for s in SEEDS:
        pos = common.sets(s)["T2"]
        names, P = b_layer(0, s)
        a = per_task_auroc(rollout_scores(P, pos), ctx.idx.y[pos], ctx.idx.task[pos])
        for n, v in zip(names, a):
            worst = max(worst, abs(v - ref[(n, s)]))
    res = {"max_abs_diff": worst, "passed": worst < 1e-4,
           "what": "scorer (sum to the cut, per-task AUROC) on B's saved layer-0 outputs vs per_probe_outcome_auroc.csv"}
    common.write_json(OUT / "logs" / "random_directions_scorer_check.json", res)
    log(f"[random] scorer check max |diff| {worst:.2e} ({'passed' if res['passed'] else 'FAILED'})")
    if not res["passed"]:
        raise SystemExit("scorer check failed")


def agreement() -> list:
    ctx = common.ctx()
    rows = []
    for L in AGREE_LAYERS:
        per = {}
        for s in SEEDS:
            pos = common.sets(s)["T2"]
            names, P = b_layer(L, s)
            SB = rollout_scores(P, pos)
            SA = rollout_scores(a_layer(L, s), pos)[:, 0]
            cell = ctx.idx.task[pos]
            for k, n in enumerate(names):
                rho = [spearmanr(SB[cell == c, k], SA[cell == c])[0] for c in np.unique(cell) if (cell == c).sum() >= 3]
                per.setdefault(n, []).append(float(np.nanmean(rho)))
        for n, v in per.items():
            rows.append({"layer": L, "B probe target": n, "Spearman with A's probe at the same layer (mean over splits)":
                         round(float(np.mean(v)), 4), "std over splits": round(float(np.std(v, ddof=1)), 4),
                         "per split": " ".join(f"{x:.4f}" for x in v)})
    return rows


# --- table ------------------------------------------------------------------------------
def probe_aurocs() -> tuple[dict, dict]:
    """B's probe and A's probe per-task AUROC on the headline test set, mean over splits."""
    b, a = {}, {}
    for r in read_csv(tr.fc.OUT / "figures/data/per_probe_outcome_auroc.csv"):
        if r["test set"] != common.TEST_SETS["T2"] or r["series"] != "probe":
            continue
        if r["detector"] == "B" and r["H"] == "10":
            b.setdefault((int(r["layer"]), r["target"].split("[")[0]), []).append(float(r["per-task AUROC"]))
        elif r["detector"] == "A" and r["layer"] != "":
            a.setdefault(int(r["layer"]), []).append(float(r["per-task AUROC"]))
    return {k: float(np.mean(v)) for k, v in b.items()}, {k: float(np.mean(v)) for k, v in a.items()}


def layer_values(L: int) -> dict:
    z = np.load(RD / f"L{L:02d}.npz")
    seen, unseen = z["auc_seen"], z["auc_unseen"]                            # (3, 200)
    signed = np.where(seen < 0.5, 1 - unseen, unseen)
    return {"signed": signed.mean(0), "unflipped": unseen.mean(0), "flipped_share": float((seen < 0.5).mean()),
            "seen_signed": np.where(seen < 0.5, 1 - seen, seen).mean(0)}


def table() -> None:
    b, a = probe_aurocs()
    agree = agreement()
    write_csv(OUT / "agreement_with_a.csv", agree)
    ag = {(r["layer"], r["B probe target"]): r["Spearman with A's probe at the same layer (mean over splits)"] for r in agree}
    rows = []
    for L in LAYERS:
        v = layer_values(L)
        row = {"layer": L}
        for tag, name in (("signed", "sign set on eval-seen"), ("unflipped", "unflipped")):
            q = np.percentile(v[tag], [5, 50, 95])
            row[f"random directions, {name}: median"] = round(float(q[1]), 4)
            row[f"random directions, {name}: 5th percentile"] = round(float(q[0]), 4)
            row[f"random directions, {name}: 95th percentile"] = round(float(q[2]), 4)
        row["share of (direction, split) pairs flipped"] = round(v["flipped_share"], 3)
        row["A's probe at this layer"] = round(a[L], 4) if L in a else ""
        row["share of random directions (sign set) A's probe beats"] = round(float((v["signed"] < a[L]).mean()), 3) if L in a else ""
        for kd in KINDS:
            if (L, kd) in b:
                row[f"B probe {kd}"] = round(b[(L, kd)], 4)
                row[f"share of random directions (sign set) B {kd} beats"] = round(float((v["signed"] < b[(L, kd)]).mean()), 3)
                row[f"share of random directions (unflipped) B {kd} beats"] = round(float((v["unflipped"] < b[(L, kd)]).mean()), 3)
            else:
                row[f"B probe {kd}"] = row[f"share of random directions (sign set) B {kd} beats"] = ""
                row[f"share of random directions (unflipped) B {kd} beats"] = ""
            if L in AGREE_LAYERS:
                row[f"Spearman with A's layer-{L} probe, B {kd}"] = ag.get((L, kd), "")
        rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(OUT / "random_directions.csv", [{c: r.get(c, "") for c in cols} for r in rows])
    np.savez(RD / "summary.npz", signed=np.stack([layer_values(L)["signed"] for L in LAYERS]),
             unflipped=np.stack([layer_values(L)["unflipped"] for L in LAYERS]))
    log(f"[random] table: {len(rows)} layers")


# --- figure -----------------------------------------------------------------------------
def figure() -> None:
    sys.path.insert(0, str(tr.HERE.parent / "final"))
    import headline
    for ext in ("pdf", "png"):
        old = FIG / f"F8_per_probe_detector_b_noband.{ext}"
        if not old.exists():
            shutil.copyfile(FIG / f"F8_per_probe_detector_b.{ext}", old)
    fg = headline.figures_mod()
    band = np.stack([np.percentile(layer_values(L)["signed"], [5, 95]) for L in LAYERS])
    write_csv(tr.fc.OUT / "figures/data/F8_random_direction_band.csv",
              [{"layer": L, "5th percentile": round(float(lo), 4), "95th percentile": round(float(hi), 4),
                "what": "per-task-and-perturbation AUROC of 200 random directions (sign set on eval-seen), mean over splits"}
               for L, (lo, hi) in zip(LAYERS, band)])
    rows, ylim = fg.per_probe_ylim()

    def draw_band(ax):
        ax.fill_between(LAYERS, band[:, 0], band[:, 1], color="#d9d8d3", alpha=0.7, lw=0, zorder=0,
                        label="random directions, 5th-95th percentile (sign set on eval-seen)")
    fig, ax = fg.plt.subplots(figsize=(fg.W, 7 * fg.CM))
    fg.per_probe_panel(ax, rows, "B", [draw_band], ylim, H=10)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28))
    fg.save(fig, "F8_per_probe_detector_b")
    clipped = int((band[:, 0] < ylim[0]).sum())
    log(f"[random] F8 redrawn with the band; y-range {ylim}; band below the lower limit at {clipped} layers")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["check", "directions", "table", "figure"]
    for st_ in steps:
        {"directions": run_directions, "check": check, "table": table, "figure": figure,
         "agreement": lambda: print(agreement())}[st_]()
