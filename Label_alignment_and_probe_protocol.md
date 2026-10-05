# Label alignment and probe protocol — spec v2

*Thesis: How To Know Your Robot · OpenVLA (greedy) on LIBERO-PRO / LIBERO-10 · 300-rollout corpus `[T=520, 33, 4096]`*
*Status: locked 4 Oct 2026 (v2.4). Replaces v1. Ideas considered and dropped are in Appendix A.*

---

## 0. The plan in one page

**Three setups. Each has specific probes or scores, then one final detector:**

| | Probes / scores (each one specific) | Final detector |
|---|---|---|
| **A (baseline)** | 34 probes: each raw input (33 stored hidden states + the action tokens) → episode outcome | the best single probe |
| **C** | 33 probes mirroring A: each raw layer's own constraints (32, layers 0–31) + the action constraints (1) → episode outcome | one linear model over all the constraint scores → outcome |
| **B** | 128 probes: each raw layer's hidden state (32) → each of 4 constraints, H steps ahead | one linear model over the probe outputs → outcome |

- **A** is SAFE's setup: does the approach work at all on raw inputs?
- **C** swaps each raw input for its constraints, probe for probe: do a couple of consistency numbers carry what the raw input carries?
- **B** uses the constraints as labels: can the hidden states *anticipate* the constraints, and does that give earlier warning?

**One data treatment:** successful rollouts are cut at the step they succeed; failures keep all 520 steps (§1).

**One evaluation, same for all three:** SAFE's protocol. Trajectory ROC-AUC (M1) and accuracy vs detection time with conformal-prediction thresholds (M4) (§5).

**Never done:** constraints as both input and label of the same model. It would read its own answer off its input.

---

## 1. Data

**Cut successes at completion.** Every rollout was stepped to 520 regardless of outcome. After a success completes the task, OpenVLA mostly sits still, which looks like failure but carries a success label. Measured on the 137 successes:

| | `act_rep` (fraction of steps repeating the previous action) | rows |
|---|---|---|
| before `t_first_success` | 0.168 | 38,618 |
| at or after `t_first_success` | 0.476 | 32,485 |

So:
- **Successes:** keep steps 0 to `t_first_success − 1`.
- **Failures:** keep all 520 steps.

This is right-censoring, and it's also what SAFE's data looked like: LIBERO normally ends an episode at success, so SAFE's successes stopped at completion while failures ran to the step cap. Every training and evaluation step below uses only these kept rows.

**Splits (SAFE; Lu et al.).** 7 seen tasks / 3 unseen, 3 seeds of task split. Within seen tasks, split by whole rollout 60/40 into train / eval-seen, so no rollout's timesteps appear on both sides. Eval-seen is also the calibration set for conformal prediction.

**Hidden states (Alain & Bengio; Lu et al.; SAFE).** 33 stored states, indexed 0–32 (0 = embedding). Per timestep, the mean over the 7 states that decode the 7 action tokens; the first of these is the last prompt token's state. fp16 → fp32. **Index 32 is post-RMSNorm, not raw** (confirmed in the repo), so raw layers are 0–31. Hidden-state constraints cover 0–31 only; A still probes index 32 but flags it.

---

## 2. The consistency constraints

Each constraint gives one score per timestep. Three families:

| Family | Constraints | What it measures |
|---|---|---|
| Action, over time | `act_mag`, `act_dir`, `grip_flip`, `act_rep` | Is the action stream behaving oddly? |
| Hidden state, over time | `emb_temp` at each layer | Has the model's internal state stopped changing, or jumped? |
| Hidden state, across layers | `xl_adj`, `xl_final`, `xl_spread`, `xl_final_hN` | Do the layers disagree with each other more than usual? |

Which constraints carry the signal is a result to be measured, not an assumption. `act_rep` was the strongest in the early analysis, but the method uses all of them.

**Preparing them (used by both B and C):**
1. **Smooth.** Apply a causal exponential moving average, so one odd timestep doesn't count but a sustained run does. The smoothing factor β sets how far back it remembers (roughly 1/(1−β) steps). FailureSpot uses β = 0.8, about 5 steps, which is probably too short here: the constraint analysis used windows of 11, 21 and 51 steps. Try the matching values, β ≈ 0.83, 0.91 and 0.96, plus no smoothing at all, and pick one per constraint family on eval-seen. (First results: `emb_temp` is best unsmoothed; the action and cross-layer constraints need smoothing.)
2. **Scale.** Z-score each smoothed constraint against the successful rows of the training split, so every score reads as "how many standard deviations from a normal successful rollout?" Smoothing comes first because it shrinks a signal's spread; scaling by the raw spread would leave the smoothed values on an odd scale.
3. **Orient.** Flip signs so higher means more failure-like, using the training split: flip a constraint if its train-split M1 is below 0.5. This uses the episode outcome (one bit per constraint), so B's probes aren't entirely free of outcome information; state that when describing B.

All three steps are fitted on the training split only and then applied unchanged to every other row.

**First check, before any training:** score each constraint against the episode outcome on its own (M1). A constraint that can't separate failures from successes by itself can't contribute to B or C.

---

## 3. The three setups

**The principle.** A linear probe asks one specific question: can this one thing be read off this one representation? So every probe here has a single input and a single target. Anything that needs combining is combined afterwards, on the outputs, by a separate small model.

**One model everywhere.** Every trained model in A, B and C is the same thing: a `StandardScaler` followed by L2 logistic regression with scikit-learn defaults (Alain & Bengio; SEP; Tillman & Mossing). Only the input and the label change.

**Naming.** A *probe* reads one specific thing: one layer's hidden state, the action tokens, or one layer's constraints. A *final detector* reads everything from a setup at once. All three setups have both.

**No probe on a single number.** A logistic regression with one input can only stretch and shift it, so its AUROC equals the raw score's. Every probe here has at least two inputs; single constraint scores are reported with their own AUROC directly.

### A — raw inputs → episode outcome (baseline)

- **34 probes:** one per stored hidden state (33, indices 0–32; index 32 is post-norm and flagged) plus one on the action tokens.
- **Action tokens** means the 7 numbers the model outputs for each action *before* they're decoded into the continuous robot action: one token (a bin index, 0–255) per action dimension.
- **Label:** the rollout's outcome, given to every kept timestep (SAFE's labelling).
- **Final detector:** the best single probe, chosen on eval-seen and reported on unseen tasks (SAFE's selection rule).
- **Reference row:** SAFE-MLP on the last layer, as published, if SAFE's code is in the fork.

No constraints appear anywhere in A. That's what keeps it comparable to SAFE.

### C — constraints → episode outcome

C mirrors A probe for probe, with each raw input replaced by the constraints computed from it.

- **33 probes:**
  - one per raw layer (32, layers 0–31), reading that layer's own constraints: its `emb_temp`, `xl_adj` and `xl_final`;
  - one reading the action constraints (`act_mag`, `act_dir`, `grip_flip`, `act_rep`).
- **Label:** the rollout's outcome, as in A.
- **The layer-by-layer comparison with A:** at each layer, do two or three constraint numbers detect failure as well as the full 4096-number hidden state? Plot both against layer on one figure.
- **Each individual constraint score is also reported with its own AUROC** (§2, first check).
- **Final detector:** one linear model over all the constraint scores together, trained on outcome (§3.1). Its weights show which constraints matter.
- The constraints are computed live at inference. They need only the current and previous timestep, and the hidden states come from the same forward pass.

**Edge convention.** `xl_adj` and `xl_final` exist for layers 0–30 (each compares with a later layer, up to 31). Layer 31's probe reuses layer 30's values.

**Ablation:** leave out one family at a time (action; hidden state over time; hidden state across layers).

### B — hidden states → constraints

- **A grid of probes:** each raw layer's hidden state (0–31) predicts each of four constraint targets: `act_rep`, action change (`act_mag`), that layer's own `emb_temp`, and that layer's `xl_final`. 128 probes. (`xl_adj` isn't used as a target: its direction flips between seeds at several layers.)
- **Each probe predicts its constraint H steps ahead** (H = 10 by default; also 0 and 25). That's what makes the probe worth having over computing the constraint directly.
- **Target:** the prepared constraint score, turned into 0/1 with SEP's split (the value that best divides the training scores into a low group and a high group).
- **No outcome labels in these probes.** They learn "this constraint is about to fire", nothing about failure.
- **The result in itself:** a layer × constraint map of where in the model each kind of inconsistency can be anticipated.
- **Final detector:** one linear model over the probe outputs, trained on outcome (§3.1).

**Ablation:** feed the final detector different subsets of constraints.

**The leak rule.** A probe can't be asked to predict something computable from its own input at the same timestep. With H > 0 this can't happen. At H = 0, drop any target that depends only on the probe's current input.

### 3.1 The final detector in B and C

One L2 logistic regression: inputs are the scores (all the constraint scores for C; the probe outputs for B), label is the episode outcome, output is one failure score per timestep.

**Why linear rather than an MLP:**
- **Its weights are the result.** Which constraints matter is read directly from them. An MLP would hide that.
- **Little independent data.** Rows within a rollout are near-duplicates, so the real sample size is the ~125 training rollouts, not the ~120k rows. A small model overfits less, and the test is on unseen tasks. SAFE kept its models tiny for the same reason.
- **Fair comparison.** Same model class in A, B and C, so differences come from the inputs and labels, not from model capacity.

**What linear can't capture, and the cheap fix.** A constraint that's abnormal in *both* directions (hidden state frozen or jumping) isn't a straight-line effect. Give the model the absolute z-score as an extra input for those.

**MLP as an ablation.** SAFE-MLP's architecture (2 layers × 256) on the same inputs. If it beats the linear model clearly on unseen tasks, the constraints interact in ways a linear model misses, which is worth reporting.

**One rule for B.** Train the final detector on probe outputs for rollouts those probes were *not* trained on (out-of-fold predictions). Otherwise it learns to trust whichever probe overfit most.

### What the comparisons show

| Comparison | Question it answers |
|---|---|
| A vs C, layer by layer | At each layer, do that layer's constraints carry as much as its raw hidden state? |
| C vs B | C notices constraints firing; B anticipates them. Does B alarm earlier at similar accuracy? |
| C, family ablation | Do hidden-state constraints add to action constraints? |
| B's layer × constraint map | Where in the model can each inconsistency be anticipated? |

### Guarding against length

After the cut, failures run longer than successes, so late-looking timesteps are mostly failures. A detector could learn "late = failure" without learning anything about failure.

- **In training:** weight rows so each class carries equal weight at every timestep, and drop rows after the last success in the training split.
- **Check:** plot each detector's mean score against `t` on held-out failures. It should rise around the stall, not steadily with `t`.
- **In evaluation:** M1 removes length differences *within* a task but not between tasks (§5), so every M1 is reported against a timestep-only floor and alongside M1-per-task.

---

## 4. Controls and baselines

**Controls.**
- Layer-0 probe: the input-level baseline (Molinari et al.; Lu et al.).
- Shuffled-label probe (Hewitt & Liang, 2019): same probe on labels shuffled across rollouts. Report real minus shuffled.
- Timestep-only score: the length check.

**Literature baselines.** Token max-probability and average entropy (SAFE appendix; free from the logits). FailureSpot's original action signal, run on this corpus (§6).

---

## 5. Evaluation (SAFE's protocol, unchanged)

Every detector is scored on the same held-out rollouts.

**M1 — trajectory ROC-AUC.** Per task, cut every test rollout at the length of that task's shortest rollout. Score each rollout once, by its highest score up to that cut. ROC-AUC over rollouts. (SAFE §5.4 and App. B.)

**M1's limit, and the two guards.** Within a task every rollout is cut at the same length. But there's one ROC-AUC across all tasks, and tasks differ in both cut length and failure rate, so a score that only tracks time still picks up which tasks fail more. Measured on this corpus, a timestep-only score gets 0.61 on eval-seen and 0.55 on unseen. SAFE's numbers carry the same effect. So:
- **The timestep-only M1 is reported as a floor** next to every M1.
- **M1-per-task** is reported alongside: the same computation within each task, averaged over tasks. A timestep-only score gets exactly 0.5 on it.

**M4 — accuracy vs detection time, with conformal prediction (CP).**
- **Threshold:** SAFE's one-sided functional CP band, a threshold that varies over the rollout. It's calibrated on the successful rollouts in eval-seen, so that a new success stays under it at every timestep with probability 1 − α. SAFE follows Xu et al. (FAIL-Detect, App. B, "adaptive modulation"). CP works on any score, so it applies to the final detectors of A, B and C alike.
- **Sweep α** over 15 levels from 0.02 to 0.9 (as FailureSpot does).
- **Balanced accuracy** = ½ × (fraction of failures that alarmed + fraction of successes that didn't).
- **Detection time** = step of the first alarm ÷ rollout length, set to 1 if it never alarms; averaged over failed rollouts.
- **Shown as** balanced accuracy against detection time; top-left is best. Tables report one fixed α (0.1).
- **To check:** how SAFE's code sets the band past the end of the calibration successes, since successes end at completion.

**Statistics.** Mean ± std over the 3 task-split seeds.

**Optional: human onset labels.** Marking one onset timestep per failed rollout on the unseen-task split (FailureSpot's protocol, about an hour) would let you test whether the constraints fire at the right *time*. Not required for A/B/C.

---

## 6. FailureSpot: validation, and where this goes further

FailureSpot (Ma et al., Sep 2026) trains a detector on VLA hidden states using labels derived from action consistency, on OpenVLA and LIBERO-10. That's independent evidence the approach is worth pursuing. It also means "first to use consistency signals as automatic labels for a VLA failure detector" can't be claimed.

**Where this thesis goes further:**

| | FailureSpot | This thesis |
|---|---|---|
| Detector input | hidden states, last layer | hidden states at every layer, compared |
| Constraints computed on | actions only | actions **and** the hidden states themselves (over time, across layers) |
| Failure signature assumed | jumpy or large actions | freezing, which is how OpenVLA fails |
| Human labels for training | 15% of training trajectories | none |

**Their own results show the gap.** With action-derived labels alone, their detector's trajectory AUROC on OpenVLA is below chance (33–47), and likewise on π0, which fails by repeating itself. Only π0-FAST, which fails by swinging around, works (74–79). Their labels treat jumpy actions as failure, so they point the wrong way for a policy that freezes. This is a reading of their tables, not something they state.

**Tested on this corpus (Stage 2), and only partly confirmed.** Their exact signal scores 0.49 on unseen tasks: at chance, not clearly below it. Flipped, it reaches 0.57, barely above the timestep-only floor of 0.55. So the supported statement is that their action signal is **uninformative** for OpenVLA here, not that it's inverted. Caveats: this is their raw signal, not a detector trained on it, and the corpus is LIBERO-PRO, not the LIBERO-10 they used.

---

## 7. What's claimed

| Claim | Status |
|---|---|
| Consistency signals as automatic labels for a VLA failure detector | Not novel (FailureSpot); cite as validation |
| Consistency computed on the hidden states themselves, over time and across layers | Novel |
| Layer-by-layer comparison across every layer | Novel for VLA failure detection (SAFE and FailureSpot use the last layer; SAFE lists this as future work) |
| FailureSpot's action signal carries no usable signal for OpenVLA on this corpus (at chance), while hidden-state consistency does | Supported by Stage 2; the stronger "it inverts" claim is not |
| Failure looks like *less* change: every hidden-state and action-size constraint is lower in failed rollouts, in every seed | Supported by Stage 2 |
| A vs C vs B on unseen tasks, and the layer × constraint map | New |

---

## 8. Open items

Implementation detail lives in `abc-implementation-handoff.md`.

- [ ] Score each constraint against outcome on its own (§2).
- [ ] Run FailureSpot's exact action signal on the corpus (§6).
- [ ] Check whether SAFE's CP code is in the fork, and how it handles the band past the calibration successes.
- [ ] Discuss the FailureSpot framing with Don.

---

## Appendix A — Considered, not in the plan

- **Comparing three data cuts** (no cut; cut at completion; cut everything at each task's shortest success). Dropped for simplicity; only the cut at completion is used.
- **Pooled AUROC** over every (rollout, timestep) row. After the cut it rewards any score that grows with time. Used only in the early constraint validation.
- **Time-stratified AUROC and `t_floor`.** AUROC computed separately at each timestep over rollouts still running, averaged up to the last timestep with ≥ 20 successes left. Immune to length (Heagerty & Zheng, 2005). Worth reviving if a reviewer questions length effects in M4.
- **A flat alarm threshold** in place of CP. Dropped in favour of SAFE's method.
- **Combining all constraints into one label for B** (by averaging, or with Snorkel's label model). Dropped: it blurs what each probe is asked. B now has one probe per constraint, combined afterwards.
- **Combining A's 34 probes** into one detector (the same linear model as B and C, over A's probe outputs). Optional extra row; the baseline is the best single probe, as in SAFE.
- **One probe per individual constraint in C** (66 single-input probes). Dropped: a one-input probe only rescales its input, so it adds nothing over the constraint's own AUROC. Replaced by one probe per layer's constraints.

## References

SAFE (Gu et al.) · FailureSpot (Ma et al., arXiv 2609.04277) · FAIL-Detect (Xu et al.) · Semantic Entropy Probes (Kossen et al., 2024) · Lu et al. 2025 · Molinari et al. 2025 · Alain & Bengio 2016 · Tillman & Mossing 2025 · Hewitt & Liang 2019 · Ratner et al. 2017 (Snorkel) · Wolpert 1992 · Heagerty & Zheng 2005.