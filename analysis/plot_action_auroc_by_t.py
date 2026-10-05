"""
plot_action_auroc_by_t.py

Per-timestep AUROC for the action-temporal constraints: at each timestep t, rank the
rollouts by that single timestep's constraint value and score against outcome. Answers
"WHEN does the signal appear", which neither the pooled schemes nor the ROC curves can
show. Reads constraints/all.parquet only.

WHY THE NULL BAND IS NOT OPTIONAL HERE. Every point is an AUROC over at most 300
rollouts and nothing else -- there is no pooling to average away noise. With 163
failures and 137 successes the standard error of a single AUROC at 0.5 is about 0.034
(Hanley-McNeil), so a signal-free curve still wanders +-0.07 at 2 SE, and this figure
draws 520 of them. Spikes are expected and mean nothing on their own; only sustained
departures from the band are readable.

--- THE TWO THINGS THE BAND HAS TO GET RIGHT ------------------------------------------

(1) PER-SERIES, PER-TIMESTEP n. The band is recomputed independently for every series at
    every timestep over exactly the rollouts defined there, so it carries that series'
    own per-timestep n. This matters most for act_dir, which is undefined whenever the
    arm did not move: its n falls from ~240 early to 96 at t = 519, and the band widens
    as 1/sqrt(n) to match. Measured on success_ever: half-width 0.0757 at n = 191
    (t = 5) against 0.1053 at n = 96 (t = 519), a ratio of 1.39 where sqrt(191/96) =
    1.41. The late-t act_dir excursions are expected under the null and the band says so.

    The bands are drawn in each series' own colour and each is labelled, because two
    grey bands of very different width stacked at alpha 0.20 read as one band and
    invite exactly the "this was computed at n = 300" misreading they are meant to
    prevent.

(2) A SMOOTHED OBSERVED CURVE IS ONLY EVER COMPARED AGAINST A SMOOTHED NULL. Under
    --window W the observed curve is the per-timestep AUROC of the SMOOTHED series
    act_*_w{W} written by smooth_constraints.py. Its band is then the per-timestep
    AUROC of that same smoothed matrix under each of the 1000 permuted label vectors --
    the identical pipeline, so the null draws carry the identical filter and the
    identical temporal autocorrelation as the observed curve. Nothing is compared
    across smoothing levels.

    MEASURED, NOT ASSUMED: this band does NOT narrow with W. --audit-nulls on
    success_ever gives a mean half-width for act_mag of 0.0575 unsmoothed against
    0.0581 / 0.0576 / 0.0571 / 0.0560 at W = 5 / 11 / 21 / 51. That is the right
    behaviour and it is worth stating plainly, because the intuition "smoothing
    averages away noise, so the band should shrink" is wrong here. AUROC at a fixed t
    is a rank statistic over the rollouts defined at t; smoothing the data changes
    those ranks but not the permutation distribution's spread, which is set by n and
    by the tie structure. What smoothing removes is the OBSERVED curve's independent
    across-t wobble, so a persistent displacement stops being masked by it. For
    act_mag that is the whole effect: 33.9% of timesteps outside the band unsmoothed
    against 72.4% at W = 51, with the band essentially fixed.

    THE OTHER CONSTRUCTION, AND WHY IT IS NOT INTERCHANGEABLE. Smoothing each of the
    1000 permuted AUROC-vs-t curves computed on RAW data gives a much narrower band
    (act_mag 0.0327 at W = 51, 1.7x tighter) because it averages across t, where the
    null draws do carry across-t independence. That band is correct -- but for a
    DIFFERENT statistic: smooth(AUROC-vs-t(raw)), not AUROC-vs-t(smooth). The two
    observed curves are not the same curve: measured correlation 0.95 / 0.93 / 0.92
    for act_mag at W = 5 / 21 / 51 and only 0.87 / 0.83 / 0.78 for act_dir, with
    max|difference| up to 0.19. Each statistic must be read against its own null, and
    pairing AUROC-vs-t(smooth) with the curve-filtered band would be anti-conservative
    by that 1.7x. This figure draws AUROC-vs-t(smooth), because that is what the
    act_*_w{W} series in the parquet are. --audit-nulls prints both.

HOW MANY INDEPENDENT LOOKS THERE ACTUALLY ARE. The printed "~5% expected by chance"
holds for independent timesteps and the unsmoothed curve is close enough to that. A
W-wide filter makes adjacent points share W-1 of their W inputs, so the effective
number of independent looks falls to roughly T/W -- about 10 at W = 51, not 520. The
excursion COUNT is therefore not a test statistic at large W and is printed only as a
description of the drawn curve; the effective count is printed beside it.

WHY THE SECOND PANEL. The number of rollouts behind each act_dir point varies
enormously across time. A per-timestep AUROC computed on 96 rollouts is not comparable
to one computed on 300, and the noise floor moves with it. That count is a different
quantity on a different scale, so it gets its own panel rather than a second y-axis.

Curves are RAW, not folded, so act_mag's anti-correlation stays visible as a dip below
0.5 rather than being mirrored away. Failure is the positive class.

Usage:
    source env.sh
    python analysis/plot_action_auroc_by_t.py
    python analysis/plot_action_auroc_by_t.py --window 21
    python analysis/plot_action_auroc_by_t.py --window 21 --audit-nulls
    python analysis/plot_action_auroc_by_t.py --corpus success_final --dark
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import rankdata

from constraint_auroc import ACT_REP_EPS, B_FRACTION, SEED, auroc
from control_diagnostic import within_task_permutation
from plot_auroc_by_layer import palette, style_axes
from smooth_constraints import rolling_mean_centred

PARQUET = Path("constraints/all.parquet")
INDEX_JSON = Path("corpus_v2_index.json")
OUT_DIR = Path("results")
N_PERM = 1000
MIN_ROLLOUTS = 30          # below this a per-timestep AUROC is not worth drawing

BASES = [("act_mag", "series1", "-"),
         ("act_rep", "series3", (0, (1, 1.6))),
         ("act_dir", "series2", (0, (5, 2)))]


def load_from_parquet(corpus: str, window: int):
    """The three action series at this smoothing level, straight out of all.parquet.

    act_rep is the one series with two provenances: the unsmoothed baseline is
    synthesised from act_mag exactly as constraint_auroc.add_derived does (and is
    deliberately not stored in the parquet, which would collide with it), while
    act_rep_w{W} is read from the parquet like everything else.
    """
    # act_rep has no unsmoothed row in the parquet by design, so at window 0 it is the
    # one series not requested from the file.
    want = ([f"{n}_w{window}" for n, _, _ in BASES] if window
            else [n for n, _, _ in BASES if n != "act_rep"])
    need = sorted(set(want) | {"act_mag", "act_dir"})

    tbl = pq.read_table(PARQUET, filters=[("constraint_name", "in", need)],
                        columns=["rollout_id", "outcome", "outcome_final",
                                 "outcome_group", "t", "constraint_name", "value"])
    have = set(pc.unique(tbl.column("constraint_name")).to_pylist())
    missing = [w for w in want if w not in have]
    if missing:
        raise SystemExit(f"{PARQUET} has no series {missing} -- run "
                         f"analysis/smooth_constraints.py first")

    rids_all = sorted(set(pc.unique(tbl.column("rollout_id")).to_pylist()))
    T = int(pc.max(tbl.column("t")).as_py()) + 1

    def dense(name):
        sub = tbl.filter(pc.equal(tbl.column("constraint_name"), name))
        ridx = pc.index_in(sub.column("rollout_id"),
                           value_set=pa.array(rids_all, type=pa.string())
                           ).to_numpy(zero_copy_only=False).astype(np.int64)
        t = sub.column("t").to_numpy(zero_copy_only=False).astype(np.int64)
        v = sub.column("value").to_numpy(zero_copy_only=False).astype(np.float32)
        flat = ridx * T + t
        if flat.size != len(rids_all) * T or np.unique(flat).size != flat.size:
            raise SystemExit(f"{name}: parquet does not fill the (rollout, t) grid")
        M = np.empty(len(rids_all) * T, dtype=np.float32)
        M[flat] = v
        return M.reshape(len(rids_all), T)

    mag_raw = dense("act_mag")
    rep_raw = np.where(np.isnan(mag_raw), np.nan,
                       (mag_raw <= ACT_REP_EPS).astype(np.float32)).astype(np.float32)

    raw = {"act_mag": mag_raw, "act_rep": rep_raw, "act_dir": dense("act_dir")}
    data = ({n: dense(f"{n}_w{window}") for n, _, _ in BASES} if window
            else {k: v.copy() for k, v in raw.items()})

    # Labels and the corpus slice, read from the same rows rather than re-derived.
    first = {}
    rid_col = tbl.column("rollout_id").to_pylist()
    oc = tbl.column("outcome").to_numpy(zero_copy_only=False)
    ocf = tbl.column("outcome_final").to_numpy(zero_copy_only=False)
    ocg = tbl.column("outcome_group").to_pylist()
    for i, r in enumerate(rid_col):
        if r not in first:
            first[r] = (int(oc[i]), int(ocf[i]), ocg[i])

    keep, labels = [], []
    for j, r in enumerate(rids_all):
        o, of, grp = first[r]
        if corpus == "success_final":
            lab = of
        elif corpus == "success_ever_strict":
            if grp == "success_lost":
                continue
            lab = o
        else:
            lab = o
        keep.append(j)
        labels.append(lab)
    keep = np.asarray(keep)
    rids = [rids_all[j] for j in keep]
    data = {k: v[keep] for k, v in data.items()}
    raw = {k: v[keep] for k, v in raw.items()}
    return data, raw, np.asarray(labels, dtype=int), rids, T


def per_timestep_auroc(mat: np.ndarray, labels: np.ndarray, bank: np.ndarray,
                       keep_draws: bool = False):
    """AUROC at each timestep, the count behind each point, and the null draws.

    At a fixed t, permuting the rollout labels does not change the ranks, so the null
    at that t is the exact permutation distribution of this statistic -- one matrix
    product against the bank, restricted to the rollouts defined at t. That restriction
    is what makes the band carry this series' own per-timestep n.
    """
    T = mat.shape[1]
    obs = np.full(T, np.nan)
    n_def = np.zeros(T, dtype=int)
    draws = np.full((bank.shape[0], T), np.nan) if keep_draws else None
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)

    for t in range(T):
        col = mat[:, t]
        ok = np.isfinite(col)
        n_def[t] = int(ok.sum())
        y = labels[ok]
        if n_def[t] < MIN_ROLLOUTS or y.sum() < 2 or (len(y) - y.sum()) < 2:
            continue
        r = rankdata(col[ok])
        count = np.ones(r.size)
        obs[t] = auroc(r, count, y.astype(np.float64))
        nul = auroc(r, count, bank[:, ok])
        if keep_draws:
            draws[:, t] = nul
        nul = nul[np.isfinite(nul)]
        lo[t], hi[t] = np.percentile(nul, [2.5, 97.5])
    return obs, lo, hi, n_def, draws


def band_from_draws(draws: np.ndarray):
    """2.5 / 97.5 percentiles down the permutation axis, NaN columns tolerated."""
    T = draws.shape[1]
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    for t in range(T):
        d = draws[:, t]
        d = d[np.isfinite(d)]
        if d.size:
            lo[t], hi[t] = np.percentile(d, [2.5, 97.5])
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="success_ever")
    ap.add_argument("--window", type=int, default=0,
                    help="smoothing window W; 0 draws the unsmoothed series")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--audit-nulls", action="store_true",
                    help="print matched vs raw vs curve-filtered band widths")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    W = args.window
    data, raw, labels, rids, T = load_from_parquet(args.corpus, W)
    t_lo = int(ceil(B_FRACTION * T))

    index = {e["rollout_id"]: e for e in json.loads(INDEX_JSON.read_text())["rollouts"]}
    rid_to_task = {r: int(index[r]["task_idx"]) for r in rids}

    rng = np.random.default_rng(SEED)
    bank = np.empty((args.n_perm, len(rids)), dtype=np.float64)
    for i in range(args.n_perm):
        m = within_task_permutation(rids, labels, rid_to_task, rng)
        bank[i] = [m[r] for r in rids]
    tag = f"w{W}" if W else "unsmoothed"
    print(f"[*] {len(rids)} rollouts, {int(labels.sum())} failures, "
          f"{args.n_perm} within-task shuffles, series {tag}")

    c = palette(args.dark)
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(11.0, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3.0, 1.0]})
    fig.patch.set_facecolor(c["surface"])
    for a in (ax, axn):
        style_axes(a, c)
        a.axvspan(t_lo, T - 1, color=c["muted"], alpha=0.07, linewidth=0, zorder=0)

    ax.axhline(0.5, color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.55)
    ax.annotate("Scheme B window", xy=(t_lo, 0.5), xytext=(6, 0),
                textcoords="offset points", color=c["muted"], fontsize=8,
                va="bottom", ha="left")

    t = np.arange(T)
    curves: list[tuple] = []
    for name, slot, dash in BASES:
        obs, lo, hi, n_def, draws = per_timestep_auroc(
            data[name], labels, bank, keep_draws=args.audit_nulls)
        label = f"{name}_w{W}" if W else name

        # Band first, in this series' own colour, so the pairing is unambiguous.
        ax.fill_between(t, lo, hi, color=c[slot], alpha=0.16, linewidth=0,
                        label=f"{label} null (2.5–97.5 pct)", zorder=2)
        ax.plot(t, obs, color=c[slot], linewidth=1.4, linestyle=dash, label=label,
                zorder=3)
        axn.plot(t, n_def, color=c[slot], linewidth=1.4, linestyle=dash)

        curves.append((label, obs, lo, hi, n_def))
        fin = np.isfinite(obs)
        out = fin & ((obs > hi) | (obs < lo))
        above = fin & (obs > hi)
        eff = int(fin.sum()) / max(1, W)
        print(f"  {label:<13} outside the matched band: {int(out.sum()):3d} of "
              f"{int(fin.sum()):3d} defined timesteps "
              f"({int(out.sum())/max(1,int(fin.sum())):5.1%})"
              f"  [{int(above.sum())} above, {int(out.sum()-above.sum())} below]"
              + (f"  ~{eff:.0f} independent looks at w={W}, so read the sign and the "
                 f"run length, not this count" if W else "; ~5% expected by chance"))

        if args.audit_nulls:
            half_m = np.nanmean((hi - lo) / 2)
            _, rlo, rhi, _, rdraws = per_timestep_auroc(
                raw[name], labels, bank, keep_draws=True)
            half_r = np.nanmean((rhi - rlo) / 2)
            note = ""
            if W:
                # Each permuted RAW curve is one row, so the filter runs on the whole
                # bank at once and never spans two draws.
                clo, chi = band_from_draws(rolling_mean_centred(rdraws, W))
                note = f"  curve-filtered raw null {np.nanmean((chi - clo) / 2):.4f}"
            print(f"    band half-width  matched {half_m:.4f}   raw-series "
                  f"{half_r:.4f}{note}")

    ax.set_ylabel("AUROC at this timestep (raw)", color=c["muted"], fontsize=10)
    ax.set_ylim(0.30, 0.70)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.0, ncol=3,
                    handlelength=2.2, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(c["muted"])
    axn.set_ylabel("rollouts\ndefined", color=c["muted"], fontsize=9)
    axn.set_xlabel("timestep t", color=c["muted"], fontsize=10)
    axn.set_xlim(0, T - 1)
    axn.set_ylim(0, len(rids) * 1.05)

    smooth_note = (f"centred rolling mean, w = {W}" if W else "unsmoothed")
    fig.suptitle(f"Action constraints, AUROC at each timestep  ·  {smooth_note}  ·  "
                 f"corpus {args.corpus}, n = {len(rids)}",
                 color=c["text"], fontsize=12.5, x=0.008, ha="left", y=0.985)
    fig.text(0.5, 0.012,
             "Each band is that series' own within-task permutation null (1000 "
             "shuffles, seed 0), recomputed at every timestep over exactly the "
             "rollouts defined there,\nso it widens as 1/√n where act_dir thins. "
             "Smoothed curves are scored against nulls put through the identical "
             "filter. Raw, not folded — act_mag below 0.5 means less movement "
             "predicts failure.",
             color=c["muted"], fontsize=7.5, va="bottom", ha="center", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))

    stem = f"action_auroc_by_t_{args.corpus}" + (f"_w{W}" if W else "")
    out_png = Path(args.out_dir) / (stem + ("_dark" if args.dark else "") + ".png")
    fig.savefig(out_png, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out_png}")

    # The numbers behind the figure, so a claim about a sustained excursion can be
    # checked against the band it is claimed to leave rather than eyeballed off a PNG.
    if not args.dark:
        out_csv = Path(args.out_dir) / (stem + ".csv")
        with open(out_csv, "w") as f:
            f.write("constraint,window,corpus,t,auroc_raw,null_lo,null_hi,n_defined\n")
            for label, obs, lo, hi, n_def in curves:
                for i in range(T):
                    if not np.isfinite(obs[i]):
                        continue
                    f.write(f"{label},{W},{args.corpus},{i},{obs[i]:.6f},"
                            f"{lo[i]:.6f},{hi[i]:.6f},{n_def[i]}\n")
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
