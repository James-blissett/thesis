"""
checks.py

Checks 1-3 and the Methods values of results_followup_handoff.md (§2, §3, §4, §6), on the
headline setup (full training set, unseen tasks incl. LIBERO-PRO). No refits: new rows
are unweighted means of saved probe outputs (the B grid work files of the two earlier
runs), evaluated with evaluate.eval_key and compared with the paired bootstrap.

  check1   B's embedding-temporal mean (pre-registered secondary) and the exploratory
           mean of every grid probe except the xl_final probes, each H, against A's mean,
           C's mean, B's mean of all grid probes and the best untrained constraint.
           -> tables/b_secondary_rows.csv, tables/b_secondary_paired.csv
  build_common (run once per training set: THESIS_POOL=D0 and D2) B's mean over the
           (layer, target) pairs present in both grids, overall and per target kind,
           each H; per-seed npz for the bootstrap.
  check2   training-data ablation on the headline test set, full minus original:
           the existing comparison (A's mean, C's mean, best untrained constraint,
           baselines; results/thesis_results_full/tables/training_set_comparison.csv),
           then B on the common probe set. -> tables/table6_training_data_headline.csv
  check3   where the LIBERO-PRO detection comes from: per-task AUROC of the primary rows
           on the original rollouts, original + new unperturbed, and each cell family of
           the headline test set; B mean vs A mean within each.
           -> tables/test_set_breakdown.csv
  methods  every number the Methods quote, full and original training set side by side.
           -> tables/methods_values.csv

Usage: source env.sh; THESIS_POOL=D2 python analysis/thesis_results/final/checks.py [check1 check2 check3 methods]
       (check2 runs the D0 side of `common` in a subprocess)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc  # noqa: E402
from fc import HEAD, HEAD_NAME, SEEDS, common, det, ev, log, read_csv, write_csv  # noqa: E402

HS = common.HS
TAB = fc.OUT / "tables"
KINDS = ("act_rep", "act_mag", "emb_temp", "xl_final")
EXPL = "exploratory (excludes the cross-layer target)"


def label(key: str) -> str:
    if key.startswith("F_B_noxl_h"):
        return f"B: unweighted mean of every grid probe except the xl_final probes, {EXPL}, H = {key.split('_h')[1]}"
    if key.startswith("F_B_common"):
        kd = key[len("F_B_common_"):].rsplit("_h", 1)[0]
        what = "all grid probes" if kd == "" or key.startswith("F_B_common_h") else f"the {kd} probes"
        return f"B: unweighted mean of {what} on the common probe set, H = {key.rsplit('_h', 1)[1]}"
    d, v, _, H = ev.LABEL[key]
    return f"{d}: {v}" + (f", H = {H}" if H != "" else "")


# --- Check 1 ----------------------------------------------------------------------------
def build_noxl() -> None:
    for H in HS:
        k = f"F_B_noxl_h{H}"
        if det.done(k):
            continue
        per = {}
        for s in SEEDS:
            names, P = det.b_outputs(H, s)
            ks = [i for i, n in enumerate(names) if n.split(":")[1] != "xl_final"]
            per[s] = det.entry(P[ks].mean(0), f"{len(ks)} probes (every grid probe except the xl_final probes)")
            del P
        det.save(k, per, f"new: mean of B's refitted grid probes except the xl_final probes ({EXPL})")


def metric_rows(keys, test_sets=(HEAD,)) -> dict:
    """{(key, test set, seed): {"auroc": row, "alarm": row at alpha 0.1}}."""
    out = {}
    for k in keys:
        r = fc.evaluate(k)
        for x in r["auroc"]:
            if x["test_set"] in test_sets:
                out.setdefault((k, x["test_set"], x["seed"]), {})["auroc"] = x
        for x in r["alarm"]:
            if x["test_set"] in test_sets and not x["on_alpha_grid"]:
                out.setdefault((k, x["test_set"], x["seed"]), {})["alarm"] = x
        for x in r["stress"]:
            if x["stress_set"] == "stress" and x["block"] == "all":
                out.setdefault((k, "stress", x["seed"]), {})["stress"] = x
    return out


def check1() -> None:
    build_noxl()
    comps = ["A_mean", "C_mean", "emb_best"]
    keys = comps + [k for H in HS for k in (f"B_mean_h{H}", f"B_meanemb_h{H}", f"F_B_noxl_h{H}")]
    M = metric_rows(keys)
    rows = []
    for k in keys:
        row = {"row": label(k), "test set": HEAD_NAME}
        fc.add_stat(row, "per-task-and-perturbation AUROC", {s: float(M[(k, HEAD, s)]["auroc"]["m1_per_task"]) for s in SEEDS})
        fc.add_stat(row, "balanced accuracy at alpha 0.1", {s: float(M[(k, HEAD, s)]["alarm"]["bal_acc"]) for s in SEEDS})
        fc.add_stat(row, "stress set: share alarmed before the cut at alpha 0.1",
                    {s: float(M[(k, "stress", s)]["stress"]["share_alarm_before_cut"]) for s in SEEDS})
        row["probes averaged per seed"] = " | ".join(M[(k, HEAD, s)]["auroc"]["detail"] for s in SEEDS)
        row["status"] = (EXPL if k.startswith("F_B_noxl") else
                         "pre-registered secondary (label-free run, 19 Oct)" if "meanemb" in k else
                         "primary" if k.endswith("_mean") or "B_mean_h" in k else "reference")
        row["row key"] = k
        rows.append(row)
    write_csv(TAB / "b_secondary_rows.csv", rows)
    files = fc.write_npz(keys, "check1")
    pairs = []
    for H in HS:
        for x in (f"B_meanemb_h{H}", f"F_B_noxl_h{H}"):
            for y in ("A_mean", "C_mean", f"B_mean_h{H}", "emb_best"):
                pairs.append((files, x, files, y, f"{label(x)} vs {label(y)}", label(x), label(y)))
    prs = fc.bootstrap_rows(pairs)
    for r in prs:
        r["status of x"] = EXPL if r["x key"].startswith("F_B_noxl") else "pre-registered secondary"
    write_csv(TAB / "b_secondary_paired.csv", prs)
    log(f"[checks] check1: {len(rows)} rows, {len(prs)} paired rows")


# --- Check 2 ----------------------------------------------------------------------------
def grid_meta(pool: str) -> dict:
    return json.loads((fc.WORK_ROOT / {"D0": "original", "D2": "full"}[pool] / "b_probes" / "meta.json").read_text())


def common_set() -> dict:
    m0, m2 = grid_meta("D0"), grid_meta("D2")
    p0 = {(int(L), k) for L, v in m0["kinds"].items() for k in v}
    p2 = {(int(L), k) for L, v in m2["kinds"].items() for k in v}
    both = sorted(p0 & p2)
    fmt = lambda s: " ".join(f"L{L:02d}:{k}" for L, k in sorted(s))  # noqa: E731
    return {"common": both, "n_common": len(both), "n_original": len(p0), "n_full": len(p2),
            "only_original": fmt(p0 - p2), "only_full": fmt(p2 - p0),
            "per_kind": {kd: sum(1 for _, k in both if k == kd) for kd in KINDS}}


def common_keys() -> list:
    return [f"F_B_common_h{H}" for H in HS] + [f"F_B_common_{kd}_h{H}" for H in HS for kd in KINDS]


def build_common() -> None:
    cs = common_set()
    want = {f"L{L:02d}:{k}" for L, k in cs["common"]}
    for H in HS:
        if all(det.done(k) for k in [f"F_B_common_h{H}"] + [f"F_B_common_{kd}_h{H}" for kd in KINDS]):
            continue
        per = {"": {}, **{kd: {} for kd in KINDS}}
        for s in SEEDS:
            names, P = det.b_outputs(H, s)
            sel = [i for i, n in enumerate(names) if n.rsplit(":", 1)[0] in want]
            assert len(sel) == cs["n_common"], (len(sel), cs["n_common"])
            per[""][s] = det.entry(P[sel].mean(0), f"{len(sel)} common probes")
            for kd in KINDS:
                ks = [i for i in sel if names[i].split(":")[1] == kd]
                per[kd][s] = det.entry(P[ks].mean(0), f"{len(ks)} common {kd} probes")
            del P
        src = f"new: mean of B's grid probes on the (layer, target) pairs present in both grids ({common.POOL_NAME})"
        det.save(f"F_B_common_h{H}", per[""], src, cs)
        for kd in KINDS:
            det.save(f"F_B_common_{kd}_h{H}", per[kd], src + f", {kd} only", cs)
    fc.write_npz(common_keys(), "check2")
    log(f"[checks] common probe set ({common.POOL_NAME}): {cs['n_common']} probes per H")


def check2() -> None:
    env = {**os.environ, "THESIS_POOL": "D0"}
    subprocess.run([sys.executable, __file__, "build_common"], env=env, check=True)
    build_common()
    cs = common_set()
    rows = []
    src = fc.FULL / "tables" / "training_set_comparison.csv"
    for r in read_csv(src):
        if r["test set"] in (HEAD_NAME, "stress set"):
            rows.append({"part": "1. existing comparison (read, not recomputed)", **r,
                         "source": str(src.relative_to(fc.REPO))})
    fo = sorted((fc.WORK_ROOT / "final" / "original" / "npz" / "check2").glob("seed*.npz"))
    ff = sorted((fc.WORK_ROOT / "final" / "full" / "npz" / "check2").glob("seed*.npz"))
    pairs = [(ff, k, fo, k, "full minus original training set", label(k), label(k)) for k in common_keys()]
    for b in fc.bootstrap_rows(pairs):
        k = b["x key"]
        kd = k[len("F_B_common_"):].rsplit("_h", 1)[0] if not k.startswith("F_B_common_h") else "all"
        rows.append({"part": "2. B on the common probe set (new)", "detector": label(k), "test set": b["test set"],
                     "metric": b["metric"], "original training set": b["y value"], "full training set": b["x value"],
                     "difference (full minus original)": b["difference"], "95% interval low": b["95% interval low"],
                     "95% interval high": b["95% interval high"],
                     "verdict": b["verdict"].replace("x higher", "full higher").replace("x lower", "full lower"),
                     "original per seed": b["y per seed"], "full per seed": b["x per seed"], "row key": k,
                     "common probes per H": cs["n_common"] if kd == "all" else cs["per_kind"][kd],
                     "probes per H, original grid / full grid": f"{cs['n_original']} / {cs['n_full']}",
                     "excluded from the original grid (not in the full grid)": cs["only_original"],
                     "excluded from the full grid (not in the original grid)": cs["only_full"] or "none",
                     "source": "new: unweighted means of the saved grid outputs of both runs"})
    write_csv(TAB / "table6_training_data_headline.csv", rows)
    log(f"[checks] check2: common probe set {cs['n_common']} (original {cs['n_original']}, full {cs['n_full']}); "
        f"{len(rows)} rows")


# --- Check 3 ----------------------------------------------------------------------------
SUBSETS = [("T0", None, "unseen tasks, original rollouts"),
           ("T1", None, "unseen tasks, original + new unperturbed rollouts"),
           ("T2", 0, "LIBERO-PRO test set, unperturbed cells (original + new unperturbed)"),
           ("T2", 1, "LIBERO-PRO test set, object perturbation cells"),
           ("T2", 2, "LIBERO-PRO test set, instruction perturbation cells"),
           ("T2", "all", "LIBERO-PRO test set, all cells (headline)")]


def subset_pos(s: int, ts: str, fam) -> np.ndarray:
    idx = common.ctx().idx
    pos = common.sets(s)[ts]
    if fam is None or fam == "all":
        return pos
    return pos[idx.task[pos] // 10 == fam]


def check3() -> None:
    ctx = common.ctx()
    idx, kept = ctx.idx, ctx.kept
    keys = ["A_mean", "C_mean"] + [f"B_mean_h{H}" for H in HS] + [f"B_meanemb_h{H}" for H in HS] + ["emb_best", "timestep"]
    rows = []
    for k in keys:
        E = ev.entries(k)
        for ts, fam, name in SUBSETS:
            row = {"row": label(k), "rollouts": name}
            per, cells, nf, ns = {}, [], [], []
            for s in SEEDS:
                pos = subset_pos(s, ts, fam)
                r = fc.m1_row(fc.m1_safe(E[s]["m1"], kept, idx, pos))
                per[s] = float(r["m1_per_task"])
                cells.append(str(r["n_cells"]))
                nf.append(int(idx.y[pos].sum()))
                ns.append(int((1 - idx.y[pos]).sum()))
            fc.add_stat(row, "per-task AUROC (per task-and-perturbation cell)", per)
            row["cells averaged per seed"] = " ".join(cells)
            row["failures per seed"] = " ".join(map(str, nf))
            row["successes per seed"] = " ".join(map(str, ns))
            row["row key"] = k
            rows.append(row)
    files = sorted((fc.WORK_ROOT / "full" / "npz").glob("seed*.npz"))
    pairs = [(files, f"B_mean_h{H}", files, "A_mean", f"B mean (H = {H}) vs A mean", label(f"B_mean_h{H}"), label("A_mean"))
             for H in HS]
    boot = []
    for ts, fam, name in SUBSETS:
        sub = None if fam in (None, "all") else (lambda f_: (lambda side, path: side.cell // 10 == f_))(fam)
        boot += fc.bootstrap_rows(pairs, test_set=ts, subset=sub, test_label=name)
    for b in boot:
        b["metric"] = b["metric"].replace("per-task-and-perturbation AUROC", "per-task AUROC (per task-and-perturbation cell)")
    write_csv(TAB / "test_set_breakdown.csv", rows)
    write_csv(TAB / "test_set_breakdown_paired.csv", boot)
    log(f"[checks] check3: {len(rows)} rows, {len(boot)} paired rows")


# --- §6 Methods values -------------------------------------------------------------------
def methods() -> None:
    rows = []
    sel = read_csv(fc.FULL / "tables" / "selections_comparison.csv")
    for r in sel:
        rows.append({"value": r["selection"], "full training set (headline)": r["full training set"],
                     "original training set": r["original training set"], "same": r["same"],
                     "source": "results/thesis_results_full/tables/selections_comparison.csv"})

    def per_run(base, key, field):
        out = {}
        for r in read_csv(base / "tables" / "data" / "detector_auroc_per_seed.csv"):
            if r["key"] == key and r["test_set"] == HEAD_NAME:
                out[int(r["seed"])] = r[field]
        return " / ".join(out[s] for s in SEEDS)

    def wd(detail):
        return " / ".join(sorted({d.split("wd=")[1].split()[0] for d in detail.split(" / ") if "wd=" in d}))

    extra = [
        ("weight decay, A's 2-layer MLP", wd(per_run(fc.FULL, "A_mlp", "detail")), wd(per_run(fc.ORIG, "A_mlp", "detail")),
         "detector_auroc_per_seed.csv (detail of A's 2-layer MLP)"),
        ("weight decay, C's 2-layer MLP on all 132 constraints (exploratory)", wd(per_run(fc.FULL, "C_mlp132", "detail")),
         wd(per_run(fc.ORIG, "C_mlp132", "detail")), "detector_auroc_per_seed.csv (detail)"),
        ("B's embedding-temporal probes per H", str(sum("emb_temp" in v for v in grid_meta("D2")["kinds"].values())),
         str(sum("emb_temp" in v for v in grid_meta("D0")["kinds"].values())), "b_probes/meta.json of each run"),
    ]
    cal = {}
    for base, tag in ((fc.FULL, "full"), (fc.ORIG, "original")):
        d = {}
        for r in read_csv(base / "tables" / "data" / "detector_alarm_per_seed.csv"):
            if r["key"] == "A_mean" and r["on_alpha_grid"] == "False" and r["test_set"] == HEAD_NAME:
                d[int(r["seed"])] = r["n_cal"]
        cal[tag] = " / ".join(d[s] for s in SEEDS)
    extra.append(("calibration rollouts per seed (successful eval-seen rollouts)", cal["full"], cal["original"],
                  "detector_alarm_per_seed.csv (n_cal)"))
    ctx = common.ctx()
    for nm in ("train", "eval_seen", "T2", "T0"):
        f_ = " / ".join(str(len(ctx.sets("D2", s)[nm])) for s in SEEDS)
        o_ = " / ".join(str(len(ctx.sets("D0", s)[nm])) for s in SEEDS)
        extra.append((f"rollouts per seed, {common.TEST_SETS.get(nm, 'training split')}", f_, o_, "results/abc2/splits"))
    for v, f_, o_, src in extra:
        rows.append({"value": v, "full training set (headline)": f_, "original training set": o_, "same": f_ == o_,
                     "source": src})
    write_csv(TAB / "methods_values.csv", rows)
    log(f"[checks] methods: {len(rows)} values")


def main() -> None:
    fc.quiet()
    args = sys.argv[1:] or ["check1", "check2", "check3", "methods"]
    for a in args:
        globals()[a]()
        log(f"[checks] {a} ok")


if __name__ == "__main__":
    main()
