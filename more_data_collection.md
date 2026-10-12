# More-data run: LIBERO-PRO collection and A/B/C re-evaluation

**Handoff version 1 — 11 Oct 2026, 20:30 AEDT.** If the copy in the repo doesn't say version 1 on this line, it's stale: replace it.

*For Claude Code, working in the `thesis-introspection` repo on the Brev box.*
*Definitions (kept rows, constraint preparation, probe, row weights, running-sum scoring, M1, M1-per-task, conformal prediction) are in `abc-implementation-handoff.md` (version 10) and are unchanged. This document only says what is new. Where the two disagree on something new, this one wins.*

---

## 0. The job

One unattended run, start to finish, no human checkpoints:

1. Collect about 700 new OpenVLA rollouts: the unused LIBERO-10 initial states, plus LIBERO-PRO perturbed versions of the same ten tasks.
2. Re-run the existing A / C / constraint / conformal-prediction pipeline, **unchanged**, on three nested training pools (original 300; plus the new unperturbed rollouts; plus the perturbed ones).
3. Answer one question in a single report: **does more data improve the detectors, measured on the same unseen-task test rollouts as before?**

The pipeline is frozen. The only thing that varies is the data. Do not tune scoring, features, smoothing rules or thresholds beyond the selection rules the existing handoff already defines.

---

## 1. Rules for this run

**Changed for this run only**
- **You launch and babysit long jobs yourself.** Run them in tmux with logs on disk. Do not print launch commands and wait.
- **No stops for sign-off.** Every gate in this document is a check you run and act on yourself. Keep going until §5's report is written, or a hard stop in §7 applies.

**Unchanged**
- **No git commands that write.** Never `git add`, `git commit` or `git push`. Put one proposed commit message at the end of the report.
- **Don't edit existing files.** `collect/`, `analysis/*.py` and `analysis/abc/` are frozen. New code goes in `collect_pro/` and `analysis/abc2/`. Copying an existing file into a new folder and changing the copy is fine; importing from existing modules is better.
- **Never delete or overwrite** anything under `/data/rollouts_v2/`, `constraints/`, `results/abc/` or the existing index and split files. The one permitted deletion is the derived feature cache `/data/tmp/abc_features/`, and only if §4 Step 1's disk rule needs the space.
- Label convention: 1 = failure, 0 = success. Outcome is `success_ever`.
- Run from the repo root after `source env.sh`. `MUJOCO_GL=egl`.
- The pins `transformers==4.40.1` and `timm==0.9.10` must survive any install. Use `pip install --no-deps` for anything new and run `pip check` afterwards.

**Everything must be resumable.** One entrypoint, `bash run_all.sh`, runs every step in order and skips any step or rollout whose output already exists and passes its check. If your session dies, re-running that one command must carry on from where it stopped.

**Keep a log.** Append to `results/abc2/RUNLOG.md` as you go: timestamp, step, what happened, any decision you made and why. Every deviation from this document goes there and is repeated in the report.

**Polling.** While jobs run, check at most every 10 minutes with a script (`status.sh`) that prints five lines or fewer: rollouts done / planned, rollouts per hour, estimated finish, free disk, last error. Don't tail full logs into your context.

---

## 2. Knobs

Put these at the top of `configs/more_data.yaml`. Use these values unless a rule in this document tells you to change one.

| Knob | Value | Meaning |
|---|---|---|
| `PER_TASK_ORI2` | 20 | new unperturbed rollouts per task (the unused initial states) |
| `PER_TASK_OBJ` | 20 | object-perturbed rollouts per task |
| `PER_TASK_LAN` | 20 | instruction-perturbed rollouts per task |
| `PER_TASK_SWAP` | 5 | position-perturbed rollouts per task (stress test only) |
| `PER_TASK_TASK` | 5 | task-perturbed rollouts per task (stress test only) |
| `N_WORKERS` | 2 | parallel collection processes on the one GPU |
| `COLLECT_HOURS_MAX` | 36 | wall-clock cap on collection; when reached, finish rollouts in flight and move on |
| `SEEDS` | 0, 1, 2 | same as before |
| `HEADROOM_GB` | 30 | disk kept free at all times |

Planned total: 700 new rollouts, 1,000 with the existing corpus.

---

## 3. Background you need

**The existing corpus is unperturbed LIBERO-10.** 300 rollouts in `/data/rollouts_v2/`, 30 per task, using 30 of each task's 50 initial states (recorded in `init_state_assignment.json`). Confirm this in Step 1 from the collection config; if the corpus turns out to contain perturbed suites, stop (§7).

**LIBERO-PRO** (`github.com/Zxy-MLlab/LIBERO-PRO`, branch `master`) is a fork of LIBERO. Its perturbed suites are ordinary LIBERO suites with their own bddl and init files:

| Block | Suite name | What changes |
|---|---|---|
| `obj` | `libero_10_object` | object appearance, colour, size |
| `lan` | `libero_10_lan` | instruction paraphrased |
| `swap` | `libero_10_swap` | object positions exchanged |
| `task` | `libero_10_task` | the task itself (different target object or goal) |

- The bddl and init files are pre-built on Hugging Face: dataset `zhouxueyang/LIBERO-Pro`. They go into `libero/libero/bddl_files/` and `libero/libero/init_files/`. Use the pre-built files; don't generate suites with `perturbation.py`.
- The suites are registered in the fork's `libero/libero/benchmark/__init__.py` and `libero_suite_task_map.py`. Object perturbations also need the fork's extra assets.
- Max steps is 520 for every `libero_10_*` suite.
- **`unnorm_key` must stay the one the existing corpus used** (read it from the existing capture config) for every suite. Do not let it be derived from the suite name.
- The environment suite (`libero_10_env`) is **not used**: the LIBERO-PRO README reports that swapping environments can move objects at random.

**What to expect.** LIBERO-PRO's published numbers on LIBERO-10 are object 0.81, position 0.00, instruction 0.96, task 0.00, against 0.93 unperturbed. Those are for a checkpoint much stronger than this one (this corpus is at 137/300), so treat them only as a guide to shape: `obj` and `lan` should give a mix of outcomes, `swap` and `task` should fail almost every time.

---

## 4. The run

### Step 1 — Preflight

1. Confirm: `/data/rollouts_v2` has 300 complete rollouts; `constraints/` has 300 files; `results/abc/` has `splits/seed{0,1,2}.json`, `final_table.csv`, `cp_curves.csv`; the checkpoint loads; GPU is free.
2. Read the existing capture code and config in `collect/` and record every setting: checkpoint, `unnorm_key`, greedy decoding, image preprocessing, wait steps (10), horizon (520, no early stop), how the 7 decision states are pooled, how `success_now` is logged.
3. Check whether the installed `libero` package already has the `libero_10_object` suite registered. If not, install LIBERO-PRO in the least invasive way that works (fork clone with `pip install -e . --no-deps`, or adding its suite files, bddl, init and assets to the current install). Before changing anything, save `pip freeze` and a copy of `~/.libero/config.yaml`. LIBERO asks an interactive question on first import if that config is missing: write the config yourself so nothing blocks.
4. Download the Hugging Face dataset and place the four `libero_10_*` suites.
5. **Disk rule.** Budget 0.29 GB per new rollout (0.145 stored + 0.145 feature cache), plus 42 GB to rebuild the cache for the existing 300, plus `HEADROOM_GB`. If free space is short, first count the 42 GB you'd get by deleting `/data/tmp/abc_features/`. If still short, scale all five `PER_TASK_*` knobs down by one common factor (round down) until it fits. If fewer than 200 new rollouts fit, hard stop. Refine the 0.29 figure after the smoke test and re-apply the rule once.

### Step 2 — Capture code, `collect_pro/`

Same model path and settings as the existing capture, with these differences only:

- **Suite and initial state are arguments.**
- **Stored per rollout**, under `/data/rollouts_v3/<rollout_id>/`:
  - `hidden.pt`: `(520, 33, 4096)` fp16, the mean over the 7 decision states. Pooled only; no per-position states.
  - `actions.npz` and `logits.pt`: same contents and shapes as v2.
  - `meta.json`: `rollout_id`, `block`, `suite`, `base_task_idx`, `init_state_index`, `instruction` (the exact string given to the model), `success_ever`, `success_final`, `t_success`, wall-clock seconds, bytes.
  - `frames/`: every step for trial 0 of each (block, task); frame 0 only for the rest.
- **Rollout ids:** `<block>_task<TT>_trial<NN>`, blocks `ori2`, `obj`, `lan`, `swap`, `task`.
- **Initial states.** `ori2`: the 20 indices per task not in `init_state_assignment.json`. Other blocks: draw without replacement from that suite's init states with `np.random.default_rng(1000 + block_number)`; if a suite has fewer states than asked for, use them all. Write every assignment to `init_state_assignment_v3.json` before collecting.
- **Job order:** sort by (trial index, block in the order `ori2, obj, lan, swap, task`, task index). A run that's cut short is then balanced across blocks and tasks. Worker `k` takes jobs where `job_number % N_WORKERS == k`.
- Write to a temporary folder and rename on completion, so a killed rollout leaves nothing half-written. Append one line per finished rollout to `/data/rollouts_v3/progress.jsonl`.
- A rollout that raises is retried once, then logged and skipped. If a (block, task) cell fails three times in a row, skip the rest of that cell and log it. One broken cell or block must not stop the run.

### Step 3 — Smoke test and parity (self-checked)

Run all of these, fix what fails, and only then start the full collection.

1. **Environment parity.** Re-run one existing rollout (same task, same initial state as a v2 rollout of your choice) through the new capture path. Its action tokens must match the stored ones exactly for the first 50 steps and in at least 95% of all steps. Its pooled hidden state at `t = 0` must match the stored one (cosine above 0.999 at every layer). This proves the new install and the new capture code reproduce the old corpus. If it fails, the install or the capture differs: find out which and fix it.
2. **One rollout per perturbed block.** For each: shapes and dtypes correct; 520 steps even if it succeeds early; no all-zero layer; frames open and aren't black.
3. **The perturbation is really applied.** `lan` and `task`: the instruction string differs from the original task's. `obj` and `swap`: frame 0 differs visibly from the original task's frame 0 at the same step (mean absolute pixel difference well above the difference between two unperturbed initial states of that task). Save a side-by-side image per (block, task) to `results/abc2/perturbation_check/`.
4. **Resume.** Kill a rollout mid-way, restart, confirm finished rollouts are skipped and the partial one is redone cleanly.
5. **Two workers.** Start both; confirm both fit in GPU memory and report rollouts per hour. If the second worker runs out of memory or the pair is slower than one alone, set `N_WORKERS = 1`.
6. Re-apply the disk rule with the measured bytes per rollout.

### Step 4 — Collect

Launch the workers in tmux. Poll with `status.sh`. Restart a worker that dies (the run is resumable). Stop launching new rollouts when the job list is done or `COLLECT_HOURS_MAX` is reached.

### Step 5 — Index and audit

- Write `corpus_v3_index.json`: one entry per rollout, old and new, with the old fields plus `block` (`ori` for the existing 300), `base_task_idx`, `init_state_index`, `instruction`, `source` (`v2` or `v3`). The old 300 come first, in their existing order.
- Audit table: successes / total per (block, task) and per block.
- **Roles.** `ori`, `ori2`, `obj`, `lan` are training-role blocks. `swap` and `task` are stress-test only: never used for training, selection or calibration. **Demotion rule:** if `obj` or `lan` has an overall success rate below 10% or above 90%, it becomes stress-test only as well; log it.
- **Near-duplicate check for `lan`.** A `lan` rollout shares its scene with the unperturbed rollout at the same task and initial state, and only the instruction differs. For each such pair, report the share of steps with identical action tokens. This says how much new information the `lan` block really holds.

### Step 6 — Constraints and feature cache for all rollouts

- `analysis/abc2/constraints.py`: compute, from the pooled hidden states and actions, exactly the series the pipeline uses: `act_mag`, `act_dir`, `grip_flip`, `emb_temp`, `xl_adj`, `xl_final`, `xl_spread` (and `act_rep` derived as before). Skip `xl_final_hN` and the `_nosink` variants; Stages 3–6 never used them. Write to `constraints_v3/<rollout_id>.npz`.
- **Parity:** recompute these for five existing rollouts and compare with the stored `constraints/*.npz`: equal within 1e-4 everywhere, NaNs in the same places. Fix until it passes.
- Feature cache at `/data/tmp/abc2_features/`: one fp16 memmap per stored index, `(N, 520, 4096)`, rows in index order, plus `action_tokens.npy`. Check that the first 300 rows of layer 15 equal the old cache (or `pooled_and_hN` output if the old cache was deleted).

### Step 7 — Splits and conditions

**Splits**, per seed, saved to `results/abc2/splits/seed{s}.json`:
- Unseen tasks: the same three base tasks as `results/abc/splits/seed{s}.json`. **Every rollout of a base task, in any block, goes to the same side.** A perturbed version of an unseen task is unseen.
- Existing rollouts keep exactly their old train / eval-seen membership.
- New rollouts in seen tasks: group by (base task, initial-state index). A group that contains an existing rollout inherits its side. The remaining groups are split 60/40 train / eval-seen per task, with `np.random.default_rng(s)`. This keeps near-duplicate scenes out of opposite sides.
- Print class counts per split, per block.

**Training pools (conditions):**

| Condition | Blocks | Planned size |
|---|---|---|
| D0 | `ori` | 300 |
| D1 | `ori`, `ori2` | 500 |
| D2 | `ori`, `ori2`, plus `obj` and `lan` unless demoted | 900 |

A condition's train, eval-seen and conformal-calibration sets are that split restricted to the condition's blocks.

**Test sets** (unseen tasks only):

| Test set | Rollouts |
|---|---|
| T0 | the original unseen rollouts. Identical for every condition; **this is the headline comparison.** |
| T1 | unseen `ori` + `ori2` |
| T2 | unseen rollouts from all training-role blocks |
| Stress | all `swap` and `task` rollouts (and any demoted block), reported separately for seen and unseen base tasks |

Every condition is evaluated on every test set.

**Cells.** "Per task" now means per cell: (base task, family), where family is `unperturbed` (`ori` and `ori2` together), `obj` or `lan`. M1's cut length is the shortest kept length in the cell within the set being evaluated. M1-per-task is the mean over cells with at least 2 rollouts of each class; state how many cells were averaged. On T0 this is the old definition exactly.

### Step 8 — Regression gate (self-checked)

Copy the Stage 1–4 and Stage 6 code into `analysis/abc2/`, generalised to take an index, a condition and a test set. Run it for D0 on T0. It must reproduce `results/abc/`:

| Quantity (unseen, mean over seeds) | Reference | Tolerance |
|---|---|---|
| `emb_temp[28]` alone, M1-per-task | 0.725 | ± 0.005 |
| C final detector, M1-per-task | 0.717 | ± 0.005 |
| A final detector, M1-per-task | 0.716 | ± 0.005 |
| `emb_temp[28]`, balanced accuracy at α = 0.1 (cut variant) | 0.556 | ± 0.01 |
| A, balanced accuracy at α = 0.1 (cut variant) | 0.537 | ± 0.01 |
| C, balanced accuracy at α = 0.1 (cut variant) | 0.501 | ± 0.01 |

Take the references from `results/abc/final_table.csv` and `cp_curves.csv` if they differ from this table, and say so. If the gate fails, the new code differs from the old: find the difference and fix it. If it still fails after three serious attempts, carry on, and put **REGRESSION GATE FAILED** with the numbers at the top of the report.

### Step 9 — Main experiment

For each condition (D0, D1, D2) and each seed, with every selection made on **that condition's eval-seen set** by the existing rules (β per family, `c`, A's best probe, the `emb_temp` layer). Unseen rollouts are never used to choose anything.

1. **Constraints alone** (old Stage 2): every constraint's M1 and M1-per-task, with signs and the unstable flags, the timestep-only floor, and the token baselines.
2. **C** (old Stage 3): 32 per-layer probes, the action probe, the final detector.
3. **A** (old Stage 4): 34 probes, final detector = best probe on eval-seen M1-per-task, shuffled-label control.
4. **Conformal prediction** (old Stage 6, **cut variant only**): the same one-sided functional band, calibrated on the condition's successful eval-seen rollouts, for A, C and the single best `emb_temp` layer. All 15 α levels. Record the number of calibration rollouts.

Evaluate each on T0, T1, T2 and Stress. For Stress, the cut for a rollout is its base task's `unperturbed` cell cut from the same seed (eval-seen for seen tasks, T1 for unseen); report the share of rollouts that raise an alarm before the cut, at α = 0.1.

**Paired comparison.** For D1 − D0 and D2 − D0 on T0, for each headline number: bootstrap 1,000 times, resampling rollouts within each cell within each seed, recompute the seed-averaged metric for both conditions on the same resample, and report the mean difference with its 95% interval. Call a change an **improvement only if the interval excludes zero.**

### Step 10 — MLP ablation (after Step 9; must not block the report)

The reason for collecting more data was to make a larger model worth trying. For D0 and D2:

- **A-MLP:** input = the pooled hidden state at the layer A selected; **C-MLP:** input = C's final-detector inputs.
- Shape: standardised input → 256 → 256 → 1, ReLU. If SAFE's training code is in the OpenVLA fork on the box, use its hyperparameters and say so. Otherwise: binary cross-entropy with the existing row weights, AdamW, learning rate 1e-3, batch 4096, weight decay chosen from {1e-4, 1e-2, 1e-1, 1}, at most 50 epochs, stopping on eval-seen M1-per-task with patience 5, torch seed = split seed.
- Scoring, M1, M1-per-task and conformal prediction exactly as for the linear models. Same test sets, same paired bootstrap against the linear model in the same condition.

### Step 11 — B (last; must not block the report)

B at **H = 10 only**, for D0 and D2: the layer × target grid on the GPU, out-of-fold outputs, final detector, conformal prediction, as in the existing handoff. D0 must reproduce `results/abc/b_final.csv` for H = 10 within ± 0.005. If B fails or time is short, say so and leave it out.

---

## 5. The report

Write `results/abc2/REPORT.md` and the csv files behind every table. Lead with the answer.

1. **Headline:** two or three sentences. Did D1 or D2 improve on D0, on T0, for ranking (M1-per-task) and for the alarm (balanced accuracy and share of failures caught at α = 0.1)? Use the "improvement" rule from Step 9.
2. **Corpus table:** planned vs collected, successes / total per block and per (block, task), demotions, skipped cells, hours taken, the `lan` near-duplicate figure.
3. **Ranking table:** rows = detector × condition (`emb_temp` alone, A, C); columns = M1-per-task on T0, T1, T2 (mean ± std over seeds), pooled M1, the timestep floor, number of cells averaged.
4. **Alarm table at α = 0.1:** balanced accuracy, share of failures caught, false-alarm rate on successes, mean detection time, number of calibration rollouts. Same rows; T0, T1, T2.
5. **Paired differences** on T0 with 95% intervals.
6. **Stress table:** share of `swap` and `task` rollouts flagged before the cut, per detector and condition.
7. **MLP and B tables**, if run.
8. **Figures:** M1-per-task against training-pool size (one line per detector, error bars over seeds); balanced accuracy against α per condition; the perturbation check sheet.
9. **Selections made:** β, `c`, chosen layers per condition and seed.
10. **Deviations and failures:** everything from `RUNLOG.md` that departs from this document, including the regression gate result.
11. **One proposed commit message.**

Report numbers as they are. A null result (no interval excludes zero) is a complete and acceptable answer; say it plainly.

---

## 6. Decisions you make yourself

- How to install LIBERO-PRO, as long as Step 3's parity check passes.
- Scaling the `PER_TASK_*` knobs under the disk rule; dropping to one worker.
- Skipping a broken cell or block; demoting a block under the Step 5 rule.
- Deleting the old feature cache if the disk rule needs it.
- Any implementation detail this document and the existing handoff leave open. Choose the option closest to what the existing code does, and log it.

---

## 7. Hard stops

Stop, write what happened at the top of `REPORT.md`, and end the run only if:

- the existing corpus, constraints or `results/abc/` are missing or incomplete;
- the existing corpus is not unperturbed LIBERO-10;
- the checkpoint won't load or the GPU is unusable;
- fewer than 200 new rollouts fit on disk;
- environment parity (Step 3.1) can't be made to pass. Without it the new rollouts can't be pooled with the old ones.

Anything else: fix it, work around it, or skip that part, log it, and continue.

---

## 8. Not in this run

- No changes to scoring, features, smoothing, the probe, or the conformal method.
- No environment perturbations, no combined perturbations, no other LIBERO suites or checkpoints.
- No B at H = 0 or 25, no leave-one-family-out ablations, no `_nosink` variants, no human onset labels.
- No edits to the spec documents.