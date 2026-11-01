"""
layer0.py

§6 extra check of task_resampling_handoff.md: B's unweighted mean without its layer-0
probes (H = 10) and A's unweighted mean without its layer-0 probe, built exactly as
detectors.build_a / build_b build the full means (unweighted mean of the saved refitted
probe outputs, running-sum score), from the full training set's saved outputs. No refits;
the conformal band is fitted by evaluate.eval_key as for every other detector (it is not
used here: §6 asks for per-task AUROC only).

Sanity check: the mean over every probe must reproduce the saved A_mean / B_mean_h10
per-step outputs.

Work files: /data/tmp/thesis_results_work/task_resampling/{detectors,eval,npz/layer0}/.
Usage: source env.sh; THESIS_POOL=D2 python analysis/thesis_results/task_resampling/layer0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr  # noqa: E402
from tr import SEEDS, WORK, common, log  # noqa: E402

det, ev, fc = tr.fc.det, tr.fc.ev, tr.fc
det.WD = WORK / "detectors"
ev.EV = WORK / "eval"
fc.FWD = WORK


def build() -> None:
    if det.done("TR_B_nolayer0_h10") and det.done("TR_A_nolayer0"):
        return
    ref_b, ref_a = det.load("B_mean_h10")["seeds"], det.load("A_mean")["seeds"]
    b, a, worst = {}, {}, 0.0
    for s in SEEDS:
        names, P = det.b_outputs(10, s)
        worst = max(worst, float(np.abs(P.mean(0) - ref_b[s]["p"]).max()))
        ks = [i for i, n in enumerate(names) if not n.startswith("L00:")]
        b[s] = det.entry(P[ks].mean(0), f"{len(ks)} probes (every grid probe except the {len(names) - len(ks)} layer-0 probes)")
        del P
        names, ref, _ = det.a_outputs(s)
        ref = ref.astype(np.float32)
        worst = max(worst, float(np.abs(ref.mean(0) - ref_a[s]["p"]).max()))
        ks = [i for i, n in enumerate(names) if n != "layer00"]
        a[s] = det.entry(ref[ks].mean(0), f"{len(ks)} probes (every probe except layer00)")
    if worst > 1e-5:
        raise SystemExit(f"mean of all probes does not reproduce the saved means (max |diff| {worst:.2e})")
    log(f"[layer0] full means reproduced from the probe outputs (max |diff| {worst:.1e})")
    det.save("TR_B_nolayer0_h10", b, "task resampling: mean of B's grid probes (H = 10) except layer 0")
    det.save("TR_A_nolayer0", a, "task resampling: mean of A's probes except layer00")


if __name__ == "__main__":
    build()
    files = fc.write_npz(["TR_B_nolayer0_h10", "TR_A_nolayer0", "B_mean_h10", "A_mean"], "layer0")
    log(f"[layer0] npz: {files[0].parent}")
