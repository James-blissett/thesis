# B label-free combiner: run log

Handoff: `b_label_free_handoff.md`, version 1 (19 Oct 2026). Run 19 Oct 2026 with
`python analysis/b_label_free/lf_combiner.py` (38 s, CPU only, no model loading, no forward passes).

## Outcome information still used (verbatim from handoff section 3)

Neither variant trains anything on outcome, but two preparation steps upstream of the probes used outcome on train/eval-seen:
- **Orientation:** each constraint's sign was set by its train-split M1-per-task (one bit per constraint).
- **Smoothing β:** chosen per constraint family on eval-seen M1 in Stage 2 (`emb_temp`: none).

The conformal band is calibrated on successful rollouts, which needs to know which rollouts succeeded.

## Probe outputs: reused, not refitted

Stage 5 saved every grid probe's per-row output in `/data/tmp/abc_scores/b_probes_h{H}_seed{s}.npz`
(written 5 Oct 2026 11:45:13, by the same run as `results/abc/b_grid.csv`, 0.4 s apart).
For eval-seen and unseen rollouts these are the refit-on-all-train probes at the grid's c (1e-4),
which is what the handoff asks for, so nothing was refitted. 124 probes per H; 30 are `emb_temp[ℓ]`.

Sanity check (handoff 2.2), applied to the reused outputs: each probe's own-target AUROC was
recomputed (targets prepared with `setup_b.prepared_targets`, threshold refitted on all train kept
rows) and compared with `b_grid.csv` at the chosen c. That is 2,232 values (124 probes × 3 H × 3
seeds × {eval-seen, unseen}). The largest |diff| was 0.0012 (L0 xl_final, H = 10, seed 2, unseen),
and none exceeded 0.005. Per-value detail is in `sanity_check.csv`. The small differences are
expected: the outputs are stored as fp16, and `b_grid.csv` comes from pass 1 (warm-started across
the c grid) while the saved outputs come from pass 2 (fitted at c = 1e-4 from zero).

## Deviations and additions

1. **Reuse instead of refit**: as above. The handoff allows this.
2. **Filename**: the handoff cites `abc-implementation-handoff.md`; the repo file is
   `abc_implementation_handoff.md`. This is naming only.
3. **Handoff copy**: there was no copy in the repo, so one was saved as `b_label_free_handoff.md`
   (version 1).
4. **Extra output**: `sanity_check.csv`.
5. **Extra columns in `lf_final.csv`**: `floor_pooled_seen` / `floor_pooled_unseen` (the timestep-only
   floor; `b_final.csv` has no floor column) and the α = 0.1 cut alarm columns
   (`a01_cut_{bal_acc,tpr,tnr,det_time}`, mean / std / per seed). False-alarm rate = 1 − TNR.
6. **Trained-B alarm values**: per seed from `results/abc/cp_curves.csv` (detector `B H={H}`,
   variant `cut`, α = 0.1). Their means match `final_table.csv`. Nothing was recomputed.
7. **Warnings**: numpy prints "Mean of empty slice" / "Degrees of freedom <= 0" from
   `cp_eval.fit_band`. These come from time steps before any D1 calibration rollout is running;
   `fit_band` discards those columns (it keeps only `[first:]`). Stage 6 ran the same code, and
   `cp_eval.log` probably shows nothing only because it captured stdout. The figure prints the same
   warning for steps with fewer than 5 failures still running, which are left blank on purpose.
8. **Length-check figure**: unseen failures only, one line per variant at each H, as specified. No
   success line is drawn.
