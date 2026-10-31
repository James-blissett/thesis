"""
smoothing.py

Check 4 of results_followup_handoff.md (§5): smoothing sensitivity for B and C, headline
setup only (full training set, tested on unseen tasks incl. LIBERO-PRO, seeds 0-2).

Settings (beta per family: emb_temp / action / cross-layer):
    current   none / 0.83 / 0.96   (the full training set's selection)
    none      none / none / none
    b083      0.83 / 0.83 / 0.83
    b096      0.96 / 0.96 / 0.96

For each setting the constraint preparation is redone from the raw series with the
existing code (prep.smooth = fill then EMA; prep.fit_scale_orient = scale on the
successful train kept rows, orient by train-split per-task AUROC; per seed), so signs,
sign-unstable flags and B's binarisation thresholds follow the setting. Held fixed:
splits, seeds, B's grid c (1e-4, the full set's), C's probe c (the full set's,
results/abc2/selections.csv), the conformal calibration (evaluate.eval_key).

Stages (each resumable; work files under /data/tmp/thesis_results_work/final/full/smoothing/):
  signs   per setting, sign per seed and the unstable flag of every primary constraint;
          cross-checked against the Stage 2 table (results/abc2/constraint_m1_summary.csv),
          which holds the same quantities for every beta
  c       C's 33 probes refitted on all train rows (setup_c.fit_prob, CPU)
  b       B's grid refitted on all train rows at H = 0, 10, 25 (setup_b.fit_grid_block,
          GPU), skipping the setting's sign-unstable targets. Each layer's features are
          loaded once and every pending setting is fitted from that load, one call per
          setting and seed with the same outputs as the original grid (the joint LBFGS
          stops on the summed loss, so the output set must match for "current" to
          reproduce). No out-of-fold fits: no trained combiner is reported.
  eval    detectors per setting: C single best probe (eval-seen per-task AUROC) and
          unweighted mean; B single best probe, unweighted mean, embedding-temporal mean
          per H; the best untrained constraint (emb_temp at the setting's emb_temp beta,
          layer re-chosen on eval-seen per-task AUROC, running maximum). Each evaluated
          as the main run (evaluate.eval_key), paired bootstrap against "current", and
          the sanity check of "current" against the headline rows (|diff| <= 0.005).

Outputs: results/thesis_results_final/tables/smoothing_sensitivity_{b,c,signs}.csv,
tables/data/smoothing_sensitivity_per_seed.csv, logs/smoothing_sanity_check.json.

Usage: source env.sh; python analysis/thesis_results/final/smoothing.py [signs c b eval]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc  # noqa: E402
from fc import SEEDS, T, common, det, log, read_csv, write_csv  # noqa: E402

SETTINGS = {
    "current": {"emb_temp": "none", "action": "0.83", "cross_layer": "0.96"},
    "none": {"emb_temp": "none", "action": "none", "cross_layer": "none"},
    "b083": {"emb_temp": "0.83", "action": "0.83", "cross_layer": "0.83"},
    "b096": {"emb_temp": "0.96", "action": "0.96", "cross_layer": "0.96"},
}
PLAIN = {"current": "current (selected): emb_temp none, action family 0.83, cross-layer family 0.96",
         "none": "no smoothing", "b083": "beta = 0.83 everywhere", "b096": "beta = 0.96 everywhere"}
WD = fc.FWD / "smoothing"
HS = common.HS


def wpath(*parts) -> Path:
    p = WD.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def check_current() -> None:
    sel = common.selections()["betas"]
    if sel != SETTINGS["current"]:
        raise SystemExit(f"'current' setting {SETTINGS['current']} differs from the full set's selection {sel}")


# --- signs ------------------------------------------------------------------------------
def signs() -> dict:
    """{setting: {"signs": {series: [sign per seed]}, "unstable": [series]}}."""
    import prep
    import setup_c as abc_c
    import stage3_c
    out_p = wpath("signs.json")
    if out_p.exists():
        return json.loads(out_p.read_text())
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    res = {}
    for st, betas in SETTINGS.items():
        fams = stage3_c.smoothed_families(ctx, betas)
        sg = {}
        for fam, (E, names) in fams.items():
            per = []
            for s in SEEDS:
                _, _, sign, _ = prep.fit_scale_orient(E, idx, kept, common.sets(s)["train"])
                per.append(sign)
            for k, (b, L) in enumerate(names):
                sg[abc_c.series_name(b, L)] = [float(p[k]) for p in per]
        del fams
        res[st] = {"betas": betas, "signs": sg,
                   "unstable": sorted(n for n, v in sg.items() if len(set(v)) > 1)}
        log(f"[smoothing] signs {st}: {len(res[st]['unstable'])} sign-unstable constraints")
    fc.common.write_json(out_p, res)
    return res


def signs_table(sg: dict) -> list:
    """smoothing_sensitivity_signs.csv rows, with the Stage 2 table's flags beside."""
    fam = {"act_mag": "action", "act_dir": "action", "grip_flip": "action", "act_rep": "action",
           "emb_temp": "emb_temp", "xl_adj": "cross_layer", "xl_final": "cross_layer", "xl_spread": "cross_layer"}
    s2 = {}
    for r in read_csv(fc.REPO / "results/abc2/constraint_m1_summary.csv"):
        if r["cond"] == common.POOL and r["variant"] == "primary" and r["test_set"] == "eval_seen":
            nm = f"{r['constraint']}[{r['layer']}]" if r["layer"] != "-1" else r["constraint"]
            s2[(nm, r["beta"])] = (r["signs_all_seeds"], r["unstable"] == "True")
    rows = []
    for st, d in sg.items():
        for nm, v in d["signs"].items():
            base = nm.split("[")[0]
            beta = d["betas"][fam[base]]
            ref = s2.get((nm, beta))
            sgn = "".join("+" if x > 0 else "-" for x in v)
            rows.append({"setting": PLAIN[st], "constraint": nm, "family": fam[base].replace("_", "-"),
                         "smoothing beta": beta, "sign per seed (+: higher means failure)": sgn,
                         "sign-unstable": len(set(v)) > 1,
                         "used as a B target": base in ("act_rep", "act_mag", "emb_temp", "xl_final"),
                         "Stage 2 table sign per seed": ref[0] if ref else "",
                         "Stage 2 table sign-unstable": ref[1] if ref else "",
                         "agrees with the Stage 2 table": (ref is not None and ref[0] == sgn and ref[1] == (len(set(v)) > 1)),
                         "setting key": st})
    return rows


def b_unstable(sg: dict, st: str) -> list:
    import setup_b as abc_b
    need = {abc_b.target_name(kd, L) for L in range(32) for kd in abc_b.TARGET_KINDS}
    return sorted(set(sg[st]["unstable"]) & need)


# --- C ----------------------------------------------------------------------------------
def fit_c(n_jobs: int = 8) -> None:
    import setup_c as abc_c
    import stage3_c
    from joblib import Parallel, delayed
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    c = common.selections()["c_C_probes"]
    probes = dict(stage3_c.PROBES)
    names = list(probes)
    for st, betas in SETTINGS.items():
        todo = [s for s in SEEDS if not wpath("c", st, f"seed{s}.npz").exists()]
        if not todo:
            continue
        t0 = time.time()
        fams = stage3_c.smoothed_families(ctx, betas)
        for s in todo:
            tr = common.sets(s)["train"]
            feats, sgn = abc_c.seed_features(fams, idx, kept, tr)
            train = common.train_rows_and_weights(idx, kept, tr)
            res = Parallel(n_jobs=n_jobs)(delayed(abc_c.fit_prob)(inp, {k: feats[k] for k in inp}, train, c)
                                          for inp in probes.values())
            P = np.stack([r[1] for r in res]).astype(np.float32)
            unconv = [n for n, r in zip(names, res) if not r[2]]
            out = wpath("c", st, f"seed{s}.npz")
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez(tmp, names=np.array(names), p_refit=P, c=c, betas=json.dumps(betas, sort_keys=True),
                     unconverged=json.dumps(unconv), signs=json.dumps(sgn))
            tmp.replace(out)
            del feats
            log(f"[smoothing] C {st} seed {s}: 33 probes, c = {c:g}, unconverged {unconv} ({time.time() - t0:.0f} s)")
        del fams


# --- B ----------------------------------------------------------------------------------
def b_positions(s: int) -> np.ndarray:
    st = common.sets(s)
    return np.unique(np.concatenate([st[k] for k in ("eval_seen", "cal", "T0", "T1", "T2", "stress")]))


def fit_b() -> None:
    import setup_b as abc_b
    import stage5_b
    import torch
    sg = signs()
    meta = json.loads((common.WORK_DIR / "b_probes" / "meta.json").read_text())
    c = float(meta["c"])
    kinds = {st: stage5_b.layer_targets(b_unstable(sg, st)) for st in SETTINGS}
    if kinds["current"] != {int(k): v for k, v in meta["kinds"].items()}:
        raise SystemExit("'current' B targets differ from the full set's grid (b_probes/meta.json)")
    pending = {st: [L for L in range(32) if not wpath("b", st, f"L{L:02d}.npz").exists()] for st in SETTINGS}
    if not any(pending.values()):
        return
    targets = {}
    t0 = time.time()
    for st in SETTINGS:
        if pending[st]:
            targets[st] = stage5_b.prepared_targets(common.ctx(), common.POOL, SEEDS, SETTINGS[st])
            log(f"[smoothing] B targets {st} prepared ({time.time() - t0:.0f} s)")
    kept = common.ctx().kept
    pos = {s: b_positions(s) for s in SEEDS}
    for L in range(32):
        todo = [st for st in SETTINGS if L in pending[st]]
        if not todo:
            continue
        X = common.load_layer(L)
        for st in todo:
            kd = kinds[st][L]
            z = {"kinds": np.array(kd), "hs": np.array(HS), "c": c, "betas": json.dumps(SETTINGS[st], sort_keys=True)}
            for s in SEEDS:
                tr = common.sets(s)["train"]
                rr = np.nonzero(kept[tr])
                if kd:
                    ys = [abc_b.lookahead(targets[st][(s, abc_b.target_name(k, L))], H) for k in kd for H in HS]
                    fits, thr, mu, sd = abc_b.fit_grid_block(X, (tr[rr[0]], rr[1]), ys, [c])
                    p = abc_b.predict_block(X, pos[s], *fits[c], mu, sd)
                else:
                    p, thr = np.zeros((0, len(pos[s]), T), np.float32), np.zeros(0)
                z[f"p{s}"] = p.astype(np.float16)
                z[f"pos{s}"] = pos[s]
                z[f"thr{s}"] = thr
            out = wpath("b", st, f"L{L:02d}.npz")
            tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez(tmp, **z)
            tmp.replace(out)
        del X
        torch.cuda.empty_cache()
        log(f"[smoothing] B layer {L:2d}: {todo} ({time.time() - t0:.0f} s)")
    fc.common.write_json(wpath("b", "meta.json"), {st: {"betas": SETTINGS[st], "c": c, "unstable": b_unstable(sg, st),
                                                       "n_probes_per_H": sum(len(v) for v in kinds[st].values())}
                                                  for st in SETTINGS})


def b_outputs(st: str, H: int, s: int):
    n = common.ctx().idx.n
    names, arrs = [], []
    for L in range(32):
        z = np.load(wpath("b", st, f"L{L:02d}.npz"))
        hs = [int(h) for h in z["hs"]]
        p, pos = z[f"p{s}"], z[f"pos{s}"]
        for i, kd in enumerate(z["kinds"]):
            f = np.zeros((n, T), np.float32)
            f[pos] = p[i * len(hs) + hs.index(H)].astype(np.float32)
            names.append(f"L{L:02d}:{kd}:H{H}")
            arrs.append(f)
    return names, np.stack(arrs)


# --- detectors per setting ---------------------------------------------------------------
def key(st: str, name: str) -> str:
    return f"SM_{st}__{name}"


def build_detectors() -> None:
    import prep
    import stage6_cp
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    sg = signs()
    for st, betas in SETTINGS.items():
        # C
        if not (det.done(key(st, "C_best")) and det.done(key(st, "C_mean"))):
            best, mean, sel = {}, {}, {}
            for s in SEEDS:
                z = np.load(wpath("c", st, f"seed{s}.npz"))
                names, ref = [str(n) for n in z["names"]], z["p_refit"]
                sums = np.stack([common.abc_c.running_sum(ref[k], kept) for k in range(len(names))])
                seen = det.stacked_seen_per_task(sums, s)
                kb = int(np.argmax(seen))
                sel[s] = {"probe": names[kb], "eval_seen_per_task": float(seen[kb])}
                best[s] = det.entry(ref[kb], f"{names[kb]} c={float(z['c']):g}")
                mean[s] = det.entry(ref.mean(0), "33 probes")
            det.save(key(st, "C_best"), best, f"new (smoothing check, {PLAIN[st]}): best of C's 33 refitted probes", sel)
            det.save(key(st, "C_mean"), mean, f"new (smoothing check, {PLAIN[st]}): mean of C's 33 refitted probes")
        # B
        for H in HS:
            ks = [key(st, f"B_best_h{H}"), key(st, f"B_mean_h{H}"), key(st, f"B_meanemb_h{H}")]
            if all(det.done(k) for k in ks):
                continue
            best, mean, memb, sel = {}, {}, {}, {}
            for s in SEEDS:
                names, P = b_outputs(st, H, s)
                sums = np.cumsum(np.where(kept[None], P, 0.0), axis=2, dtype=np.float32)
                seen = det.stacked_seen_per_task(sums, s)
                del sums
                kb = int(np.nanargmax(seen))
                sel[s] = {"probe": names[kb], "eval_seen_per_task": float(seen[kb]), "n_probes": len(names)}
                best[s] = det.entry(P[kb], names[kb])
                mean[s] = det.entry(P.mean(0), f"{len(names)} probes")
                emb = [k for k, n in enumerate(names) if n.split(":")[1] == "emb_temp"]
                memb[s] = det.entry(P[emb].mean(0), f"{len(emb)} embedding-temporal probes")
                del P
            src = f"new (smoothing check, {PLAIN[st]})"
            det.save(ks[0], best, f"{src}: best of B's refitted grid probes against outcome", sel)
            det.save(ks[1], mean, f"{src}: mean of B's refitted grid probes")
            det.save(ks[2], memb, f"{src}: mean of B's refitted embedding-temporal probes")
        # best untrained constraint: emb_temp at this setting's emb_temp beta, layer on eval-seen per-task
        k = key(st, "emb_best")
        if not det.done(k):
            beta = betas["emb_temp"]
            S, nm = ctx.series("primary")
            ks_ = [i for i, (b, _) in enumerate(nm) if b == "emb_temp"]
            E = prep.smooth(S[ks_], [nm[i] for i in ks_], 0.0 if beta == "none" else float(beta))
            score = np.zeros((len(SEEDS), len(ks_)))
            for j, s in enumerate(SEEDS):
                mu, sd, sign, _ = prep.fit_scale_orient(E, idx, kept, common.sets(s)["train"])
                X = prep.apply_prep(E, mu, sd, sign)
                score[j] = det.stacked_seen_per_task(X, s)
            del E
            Lb = int(nm[ks_[int(np.argmax(score.mean(0)))]][1])
            per = {}
            for s in SEEDS:
                x, run, sign = stage6_cp.emb_scores(ctx, common.sets(s)["train"], Lb, beta)
                run = np.where(run > -1e300, run, np.nan)
                per[s] = {"p": x.astype(np.float32), "m1": x.astype(np.float32), "cp": run.astype(np.float32),
                          "detail": f"emb_temp[{Lb}] beta={beta} sign={'+' if sign > 0 else '-'}"}
            det.save(k, per, f"new (smoothing check, {PLAIN[st]}): emb_temp, layer re-chosen on eval-seen "
                     "per-task AUROC, running maximum",
                     {"layer": Lb, "beta": beta, "eval_seen_per_task_mean": float(score.mean(0).max())})
        log(f"[smoothing] detectors {st} built")


ROWS = ([("C", "single best probe (of 33)", "C_best", ""), ("C", "unweighted mean of 33 probe outputs (primary)", "C_mean", "")]
        + [r for H in HS for r in (("B", "single best probe, chosen on eval-seen per-task AUROC against outcome", f"B_best_h{H}", H),
                                   ("B", "unweighted mean of all grid probe outputs (primary)", f"B_mean_h{H}", H),
                                   ("B", "unweighted mean of the embedding-temporal probe outputs (secondary)", f"B_meanemb_h{H}", H))]
        + [("Reference", "best single untrained constraint (emb_temp, layer re-chosen on eval-seen per-task AUROC), running maximum",
            "emb_best", "")])
HEADLINE_KEY = {"C_best": "C_best", "C_mean": "C_mean", "emb_best": "emb_best",
                **{f"B_{v}_h{H}": f"B_{v}_h{H}" for v in ("best", "mean", "meanemb") for H in HS}}


def evaluate_all() -> None:
    fc.quiet()
    build_detectors()
    sg = signs()
    bmeta = json.loads(wpath("b", "meta.json").read_text())
    keys = [key(st, r[2]) for st in SETTINGS for r in ROWS]
    per = fc.per_seed_rows(keys)
    files = fc.write_npz(keys, "smoothing")
    # per-seed metrics on the headline test set
    au = {(r["key"], int(r["seed"])): r for r in per["auroc"] if r["test_set"] == fc.HEAD_NAME}
    al = {(r["key"], int(r["seed"])): r for r in per["alarm"] if r["test_set"] == fc.HEAD_NAME and not r["on_alpha_grid"]}
    # sanity check: "current" against the headline rows (the full training set's run)
    hd = {(r["key"], int(r["seed"])): float(r["m1_per_task"])
          for r in read_csv(fc.FULL / "tables" / "data" / "detector_auroc_per_seed.csv") if r["test_set"] == fc.HEAD_NAME}
    sanity = []
    for _, _, k, _ in ROWS:
        new = [float(au[(key("current", k), s)]["m1_per_task"]) for s in SEEDS]
        old = [hd[(HEADLINE_KEY[k], s)] for s in SEEDS]
        sanity.append({"row": k, "current setting (this check), mean": float(np.mean(new)),
                       "headline row, mean": float(np.mean(old)), "difference of means": float(np.mean(new) - np.mean(old)),
                       "largest per-seed difference": float(np.max(np.abs(np.subtract(new, old)))),
                       "choice per seed (this check)": [au[(key("current", k), s)]["detail"] for s in SEEDS],
                       "passed (|difference of means| <= 0.005)": abs(np.mean(new) - np.mean(old)) <= 0.005})
    ok = all(r["passed (|difference of means| <= 0.005)"] for r in sanity)
    fc.common.write_json(fc.OUT / "logs" / "smoothing_sanity_check.json", {"passed": ok, "rows": sanity})
    log(f"[smoothing] sanity check {'passed' if ok else 'FAILED'}: "
        + "; ".join(f"{r['row']} {r['difference of means']:+.4f}" for r in sanity))
    # paired bootstrap against "current"
    pairs = [(files, key(st, k), files, key("current", k), f"{PLAIN[st]} minus current, {d} {v}" + (f", H = {H}" if H != "" else ""),
              PLAIN[st], PLAIN["current"]) for st in SETTINGS if st != "current" for d, v, k, H in ROWS]
    boot = {(r["x key"], r["metric"]): r for r in fc.bootstrap_rows(pairs)}
    out_b, out_c, per_seed = [], [], []
    for st in SETTINGS:
        for d, v, k, H in ROWS:
            kk = key(st, k)
            row = {"setting": PLAIN[st], "detector": d, "variant": v, "look-ahead H": H, "test set": fc.HEAD_NAME}
            fc.add_stat(row, "per-task-and-perturbation AUROC", {s: float(au[(kk, s)]["m1_per_task"]) for s in SEEDS})
            row["cells averaged per seed"] = " ".join(str(au[(kk, s)]["n_cells"]) for s in SEEDS)
            fc.add_stat(row, "balanced accuracy at alpha 0.1", {s: float(al[(kk, s)]["bal_acc"]) for s in SEEDS})
            for met, nm in (("per-task-and-perturbation AUROC", "AUROC"), ("balanced accuracy at alpha 0.1", "balanced accuracy")):
                b = boot.get((kk, met))
                row[f"difference from current, {nm}"] = b["difference"] if b else (0.0 if st == "current" else "")
                row[f"difference from current, {nm}, 95% interval low"] = b["95% interval low"] if b else ""
                row[f"difference from current, {nm}, 95% interval high"] = b["95% interval high"] if b else ""
                row[f"difference from current, {nm}, verdict"] = (b["verdict"].replace("x higher", "higher than current")
                                                                  .replace("x lower", "lower than current") if b else "")
            if d == "B":
                row["B probes per look-ahead H"] = bmeta[st]["n_probes_per_H"]
                row["B targets skipped (sign-unstable)"] = " ".join(bmeta[st]["unstable"])
            row["sign-unstable constraints (all families)"] = len(sg[st]["unstable"])
            row["choice per seed"] = " | ".join(au[(kk, s)]["detail"] for s in SEEDS)
            row["row key"] = kk
            (out_c if d in ("C", "Reference") else out_b).append(row)
            if d == "Reference":
                out_b.append(row)
            for s in SEEDS:
                per_seed.append({"setting": PLAIN[st], "detector": d, "variant": v, "look-ahead H": H, "seed": s,
                                 "test set": fc.HEAD_NAME, "per-task-and-perturbation AUROC": fc.r4(float(au[(kk, s)]["m1_per_task"])),
                                 "balanced accuracy at alpha 0.1": fc.r4(float(al[(kk, s)]["bal_acc"])),
                                 "choice": au[(kk, s)]["detail"], "row key": kk})
    t = fc.OUT / "tables"
    write_csv(t / "smoothing_sensitivity_b.csv", out_b)
    write_csv(t / "smoothing_sensitivity_c.csv", out_c)
    write_csv(t / "smoothing_sensitivity_signs.csv", signs_table(sg))
    write_csv(t / "data" / "smoothing_sensitivity_per_seed.csv", per_seed)
    write_csv(t / "data" / "smoothing_sensitivity_paired.csv", list(boot.values()))
    log("[smoothing] tables written")


def main() -> None:
    check_current()
    args = sys.argv[1:] or ["signs", "c", "b", "eval"]
    for a in args:
        {"signs": signs, "c": fit_c, "b": fit_b, "eval": evaluate_all}[a]()
        log(f"[smoothing] stage {a} ok")


if __name__ == "__main__":
    main()
