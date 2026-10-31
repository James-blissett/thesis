# Deep-MLP run: does depth help A and C?

**Handoff version 1 — 13 Oct 2026, 14:00 AEDT.** If the copy in the repo doesn't say version 1 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box.*
*Builds on `more-data-handoff.md` (version 1) and the finished `results/abc2/` run. Every definition there (splits, pools D0/D1/D2, test sets T0/T1/T2/Stress, cells, row weights, running-sum scoring, M1, M1-per-task, cut-variant conformal prediction, paired bootstrap) is unchanged. This document only says what is new.*

---

## 0. The job

One unattended run, no human checkpoints. Train MLPs with **2, 6 and 12 hidden layers** as A's and C's detectors on pools D0 and D2, and answer one question in one report: **does a deeper MLP beat the linear detector or the 2-layer MLP on T0?**

No new rollouts, no new features, no changes to the existing pipeline. Only the detector's depth (and its regularisation, chosen by the usual eval-seen rule) varies.

---

## 1. Rules

Same as `more-data-handoff.md` §1:
- You launch and babysit long jobs yourself, in tmux, with logs on disk. No stops for sign-off.
- No git commands that write. One proposed commit message at the end of the report.
- New code goes in `analysis/abc3/`; results in `results/abc3/`. Don't edit `analysis/abc2/` or anything older; import from it.
- Never delete or overwrite `/data/rollouts_v2/`, `/data/rollouts_v3/`, `constraints*/`, `results/abc/`, `results/abc2/`, or the abc2 feature cache.
- Resumable: `bash analysis/abc3/run_all.sh` skips any fit whose output already exists and passes its check.
- Append decisions and deviations to `results/abc3/RUNLOG.md`.
- Poll at most every 10 minutes with a short status script.

---

## 2. The model

Plain stacked MLPs become hard to train well beyond a few layers. Without residual connections and normalisation, a weak 12-layer result could mean "depth doesn't help" or "the optimiser failed", and the run couldn't tell which. So every depth uses the same **pre-norm residual MLP**:

```python
class Block(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.fc = nn.Linear(w, w)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        return x + self.drop(F.relu(self.fc(self.norm(x))))

class DeepMLP(nn.Module):
    def __init__(self, d_in, depth, w=256, p=0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, w)          # first hidden layer
        self.blocks = nn.ModuleList([Block(w, p) for _ in range(depth - 1)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 1))
    def forward(self, x):
        h = F.relu(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h).squeeze(-1)
```

- `depth` ∈ {2, 6, 12} = number of hidden layers. Width 256 throughout.
- **Depth 2 is re-trained here with this architecture**, so the 2 vs 6 vs 12 comparison is like for like. The abc2 plain 2-layer MLP is kept as a reference row, not as the comparison.
- Inputs standardised with train-set statistics, as before.

**Training** (same recipe for every depth):
- Binary cross-entropy with the existing row weights (`1 / n_class(t)`, rescaled to mean 1, rows only where ≥ 5 train successes remain).
- AdamW, learning rate 1e-3 with 5% linear warm-up then cosine decay, batch 4096, at most 50 epochs, gradient clipping at norm 1.0.
- Batches shuffled across rollouts.
- Early stopping on eval-seen M1-per-task, evaluated once per epoch, patience 5; keep the best epoch's weights.
- **Regularisation grid** (the abc2 run found the weight-decay grid alone did almost nothing): dropout `p` ∈ {0, 0.3} × weight decay ∈ {1e-2, 1e-1}. Pick the best of the four by eval-seen M1-per-task, separately for each (input, depth, pool, seed).
- Torch seed = split seed. Use mixed precision (bf16) if it reproduces fp32 eval-seen M1 within 0.005 on one fit; otherwise fp32.

**Inputs:**
- **A-deep:** the pooled hidden state at the layer A's linear probe selected for that (pool, seed) in abc2.
- **C-deep:** C's final-detector inputs, exactly as in abc2.

**Grid size:** 2 inputs × 3 depths × 2 pools (D0, D2) × 3 seeds × 4 regularisation settings = 144 fits.

---

## 3. Evaluation

For every (input, depth, pool, seed), with the best regularisation setting:

1. **Train-set M1-per-task** (on train rollouts, same scoring). This is the diagnostic that tells overfitting from underfitting: if train climbs towards 1 while T0 stays flat or falls, the model is memorising.
2. **Eval-seen M1-per-task** (optimistic, since it picked the epoch and the regularisation; label it so).
3. **M1-per-task and pooled M1 on T0, T1, T2.**
4. **Cut-variant conformal prediction** at all 15 α levels, calibrated on the pool's successful eval-seen rollouts. At α = 0.1: balanced accuracy, share of failures caught, false-alarm rate, mean detection time.
5. **Stress set:** share flagged before the cut at α = 0.1.
6. **Parameter count** per depth and input.

**Sanity check before the full grid:** depth 2, C input, D0, seed 0, with a plain (non-residual, no LayerNorm) MLP and abc2's settings must reproduce abc2's C-MLP D0 seed-0 M1-per-task on T0 within ± 0.01. If not, find out why before going on.

**Paired comparisons on T0** (same paired bootstrap as abc2, 1,000 resamples within cell within seed), for ranking (M1-per-task) and alarm (balanced accuracy, failures caught):
- each depth against the **linear** detector in the same pool and input (from `results/abc2/`);
- depth 6 and 12 against **depth 2** in the same pool and input;
- D2 against D0 for each depth and input.

Improvement only if the 95% interval excludes zero. As in abc2, the alarm intervals hold each seed's conformal band fixed: report them, and say they understate the uncertainty.

---

## 4. The report

`results/abc3/REPORT.md`, with the csv behind every table. Lead with the answer.

1. **Headline:** two or three sentences. Does 6 or 12 layers beat linear or 2 layers on T0, for ranking or alarm?
2. **Depth table:** rows = input × pool × depth (plus the linear and abc2 2-layer reference rows); columns = parameter count, train / eval-seen / T0 / T1 / T2 M1-per-task (mean ± std over seeds), chosen dropout and weight decay per seed.
3. **Alarm table at α = 0.1**, same rows, on T0.
4. **Paired differences** on T0 with 95% intervals.
5. **Figures:**
   - M1-per-task against depth (linear plotted at depth 0), one panel per input, lines for train, eval-seen and T0, D0 and D2 as separate colours. This one figure shows whether depth overfits.
   - Training curves (train loss, eval-seen M1-per-task against epoch) for one seed of each depth, C input, D2.
6. **Stress table.**
7. **Deviations and failures** from `RUNLOG.md`.
8. **One proposed commit message.**

A null result is a complete answer; say it plainly.

---

## 5. Decisions you make yourself

- Dropping bf16 if it fails its check.
- If a 12-layer fit diverges (loss NaN or eval-seen M1 below 0.5 at every epoch), retry once at learning rate 3e-4; if it fails again, log it and report the cell as failed.
- Any detail left open: choose what the abc2 MLP code did, and log it.

## 6. Hard stops

Stop and write the reason at the top of `REPORT.md` only if `results/abc2/`, the abc2 splits or the abc2 feature cache are missing, or the GPU is unusable. Otherwise work around, log and continue.

## 7. Not in this run

- No new rollouts, D1, B, or wider (rather than deeper) models.
- No temporal inputs (windows of past steps). Every model still sees one timestep at a time.
- No changes to scoring, features or the conformal method.
