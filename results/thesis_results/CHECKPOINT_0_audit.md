# Checkpoint 0: storage and inventory audit

*results_collection_handoff.md (version 4), §3. Written 2026-10-30 ~13:31 UTC by the unattended run. Evidence: `logs/audit_cache.json`, `logs/audit_inventory.json`, `logs/audit_timing.json` (made by `analysis/thesis_results/audit.py`). Nothing to commit at this stage.*

**Summary.** Everything the main run needs is on disk. The original-set feature cache is gone, as expected. The full-set feature cache covers every original-set rollout: its rows are in index order, every row is marked done, and spot checks against the stored hidden states match exactly. Saved per-row outputs cover the 300 original rollouts for every probe family. They do not cover the LIBERO-PRO test sets, apart from the final detectors of the more-data run. So A's 34 probes, C's 33 probes and B's 124 probes at each H have to be refitted on the original training set and applied to the LIBERO-PRO and stress rollouts. That takes minutes on the GPU. Estimates: **main run about 2.5 h, background run about 2.5 h** (compute only; details below).

## 1. Storage

| Item | Exists | Size |
|---|---|---|
| `/data/rollouts_v2/` (300 original rollouts) | yes | 138 GB, 300 rollouts (+ index file) |
| `/data/rollouts_v3/` (700 new rollouts) | yes | 94 GB, 700 rollouts (+ progress/marker files) |
| original-set feature cache `/data/tmp/abc_features/` | **no** (deleted by the more-data run under its disk rule, RUNLOG 2026-10-12T08:43Z) | — |
| full-set feature cache `/data/tmp/abc2_features/` | yes | 131 GB: 33 layer files (1000, 520, 4096) fp16, action tokens (1000, 520, 7), `done.npy`, `rollout_ids.json` |
| `constraints/` | yes | 305 MB, 300 rollout files (+3 other files) |
| `constraints_v3/` | yes | 115 MB, 700 files |
| `results/abc/` | yes | 44 MB |
| `results/abc2/` | yes | 186 MB |
| `results/abc3/` | yes | 171 MB |
| `results/b_label_free/` | yes | 17 MB |
| `/data/tmp/abc_scores/` (saved probe outputs, original set) | yes | 406 MB |
| `/data/tmp/abc2_scores/` (more-data work files) | yes | 215 MB; the B work folder is empty |

**Free disk:** `/data` 36 GB free (93 % used). The repo is on the root file system (`/dev/vda1`, 90 GB free), so `results/thesis_results*/` don't use `/data`. Work files for this run go in `/data/tmp/thesis_results_work/`. Expected size is about 1.5 GB per training set (probe outputs, fp16), well inside the 15 GB floor. No new feature cache is built.

**Full-set cache covers the main run** (`logs/audit_cache.json`). Row order equals `corpus_v3_index.json`, and the 300 original rollouts are rows 0–299. `done.npy` is true for all 1,000 rows. All 33 layer files have shape (1000, 520, 4096). Layers 15 and 28 match `compute_constraints.pooled_and_hN` read from `/data/rollouts_v2` exactly (max |diff| 0.0) for 4 original rollouts, one of them with 7 stored positions (`task00_trial00`). The more-data run also reproduced every original-set regression-gate number from this cache. Nothing needs rebuilding.

## 2. Inventory of saved per-row outputs

"Original 300" means train, eval-seen and unseen-task original rollouts. Probe outputs on train rollouts are noted where they are out-of-fold.

| Item | Eval-seen | Unseen, original rollouts | Unseen incl. LIBERO-PRO / stress set | File |
|---|---|---|---|---|
| A's 34 probes (all of them, at c = 1e-5) | yes | yes | **no** (the more-data run kept per-probe metrics only: `results/abc2/a_probes.csv`, D0 rows) | `/data/tmp/abc_scores/a_probes_seed{s}.npz` (34 × 300 × 520, fp16; train rows are refit, not out-of-fold) |
| A's single best probe | yes | yes | yes | `results/abc2/scores/D0_seed{s}.npz` key `A` (1,000 rollouts); `results/abc/scores/a_seed{s}.npz` (300) |
| C's 33 probes (at c = 1) | yes | yes | **no** (metrics only: `results/abc2/c_probes.csv`) | `/data/tmp/abc_scores/c_probes_seed{s}.npz` (33 × 300 × 520) |
| B's 124 grid probes, each H | yes | yes | **no** (the more-data run's H = 10 probe outputs were work files and have been deleted) | `/data/tmp/abc_scores/b_probes_h{H}_seed{s}.npz` (124 × 300 × 520, fp16; train rows out-of-fold, others refit on all train) |
| C's 132-input logistic regression | yes | yes | yes | `results/abc2/scores/D0_seed{s}.npz` key `C`; `results/abc/scores/c_seed{s}.npz` (300, incl. leave-one-family-out) |
| C's 2-layer MLP (132 inputs) | yes | yes | yes | `results/abc2/scores/D0_seed{s}_mlp.npz` key `C-MLP` |
| A's 2-layer MLP | yes | yes | yes | same file, key `A-MLP` |
| Deep MLPs (2-, 6-, 12-layer residual), both training sets | yes | yes | yes | `results/abc3/scores/{D0,D2}_seed{s}_deep.npz` |
| B's trained combiner, H = 0 / 10 / 25 | yes | yes | H = 10 only | `results/abc/scores/b_h{H}_seed{s}.npz` (300); `results/abc2/scores/D0_seed{s}_b10.npz` (1,000, H = 10) |
| B's unweighted means (all 124; embedding-temporal 30), each H | yes | yes | **no** | `results/b_label_free/scores/lf_{all,emb}_h{H}_seed{s}.npz` (300) |
| Best untrained constraint (`emb_temp[28]`, pooled-rule layer) | yes | yes | yes | `results/abc2/scores/D0_seed{s}.npz` key `emb` |

**What has to be computed for the LIBERO-PRO test sets** (everything else is reused):
- **A:** refit the 34 probes on the original training set (GPU, the existing `setup_a.fit_and_score`) and score all 1,000 rollouts. The same pass gives the out-of-fold outputs for A's new trained combiner.
- **C:** refit the 33 probes (CPU, the existing `setup_c.fit_prob`) and score all rollouts, plus out-of-fold outputs.
- **B:** refit the grid on all training rows at the grid's c (1e-4, joint 12-output fit as in Stage 5) and score every non-training rollout. Training-row outputs are the saved out-of-fold ones. The refit is checked against the saved outputs on the original eval-seen and unseen rollouts.

**Other findings from reading the earlier runs:**
- **Shuffled-label control:** it exists only for each seed's *selected* A probe, with 5 permutations (`results/abc/a_shuffled.csv`, `results/abc2/a_shuffled.csv`). §4.3 asks for a value per layer, so that becomes new work (cheap: one fit takes 0.25 s).
- **Best untrained constraint layer:** the earlier choice (`emb_temp[28]`) used eval-seen *pooled* AUROC. The per-task rule in §4.2 picks **layer 26** on the original training set (eval-seen per-task 0.757 vs 0.742 for layer 28; on unseen tasks, original rollouts, the two are level at 0.725; `results/abc2/constraint_m1_summary.csv`). The new baseline row uses layer 26, and both are reported.
- **Detection time:** the existing cut-variant code (`cp_eval.evaluate_cp` with a cut) divides the first alarm step by the rollout's cut length (the shortest kept length in its task, or task and perturbation cell, within the evaluated set), sets it to 1 if there is no alarm, and averages over failed rollouts. That is §2's definition, so nothing needs flagging.
- **Conformal alarm for the baselines:** it has not been computed for the token baselines, FailureSpot or the timestep-only score. Only per-task and pooled AUROC exist (`results/abc2/constraint_m1.csv`). These are new work.

## 3. Timing (original training set, seed 0, L40S; `logs/audit_timing.json`)

| Step | Time |
|---|---|
| Load one layer of the full-set cache to the GPU (4.3 GB) | 22 s |
| One A probe fit, warm-started over all 6 c | 4.1 s |
| One A probe fit up to the chosen c = 1e-5 | 0.25 s |
| One B grid fit, one layer, 12 outputs, one c (refit) | 0.53 s |
| One B grid fit, one layer, 12 outputs, all 6 c (grid selection) | 7.8 s |
| One 2-layer MLP fit (one weight decay; early stopping at epoch 6 of 50) | 0.47 s |
| Training rows (seed 0) | 47,347 |

**Main run estimate: about 2.5 h of compute.**
- A: 34 probes × (22 s load + 3 seeds × [refit + 5 out-of-fold fits + 5 shuffled-label fits + scoring 1,000 rollouts ≈ 15 s]) ≈ 40 min.
- B: 32 layers × (22 s + 3 seeds × [refit + scoring ≈ 6 s]) ≈ 20 min, plus B's MLP and the persistence baseline, under 10 min.
- C probes, combiners, the new MLPs, conformal alarms for about 25 detectors × 3 seeds × 3 test sets, and the per-probe profiles: about 30 min on CPU.
- Paired bootstraps (about 20 comparisons × 1,000 resamples): about 20 min.
- Tables, figures and the report: about 15 min.

Writing and checking the code takes longer than the compute.

**Background run estimate (full training set): about 2.5 h.**
- B's grid at all three H: pass 1 at all 6 c, then 5-fold out-of-fold and a refit, 3 seeds, about 45 min.
- A's 34 probes with out-of-fold outputs: about 45 min.
- C, the combiners, the MLPs, the alarms, the bootstraps and the comparison table: about 1 h.
