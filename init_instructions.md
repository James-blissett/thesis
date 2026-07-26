# Handoff: Environment Rebuild, Mini-Corpus, and Single-Layer Linear Probe

## Role and scope

The ephemeral volume was wiped: environment, capture code, and the full rollout corpus are gone. You are rebuilding the environment from a known-good recipe, recovering or recreating the hidden-state capture module, generating a 50-rollout mini-corpus, training a linear probe on one layer, and reinstating a projection-visualisation script. The design below is fixed. Do not redesign, do not add an MLP, do not modify the model. Priority order if time runs short: environment > harness > capture > corpus > probe > projections.

**Git discipline is the standing failure mode of this project.** Work was lost twice because nothing was pushed. The user is the sole author of commits: **never run `git add`, `git commit`, or `git push` yourself.** Wherever this doc says "commit X" or marks a [PUSH] checkpoint, it means: pause, tell the user exactly which files are ready and why now is a checkpoint, propose a one-line commit message, and wait for the user to confirm they have committed and pushed before starting any dependent long-running step. Read-only git (`status`, `diff`, `log`) is fine and encouraged for verifying state. Nothing long-running starts until the code it depends on is confirmed pushed.

## Context

- Thesis: runtime failure detection for OpenVLA (7B, Llama-2 backbone, 33 transformer layers, d_model=4096) on LIBERO-10.
- This probe pilots a full layer x timestep AUROC heatmap (thesis Contribution 3). Linear probes are deliberate: limited expressivity makes high AUROC evidence about the representation itself.
- Hardware: Brev L40S 48GB (Crusoe). Work in `/ephemeral/code/thesis-introspection`.
- **Storage layout (verified on the box)**: `/ephemeral` is a directory on the 124 GB root volume (~105 GB free) — code only. The big data volume is **`/data`** (`/dev/vdb`, 492 GB, ~467 GB free), which already has `corpus/`, `hf-cache/`, and `tmp/` directories. All large artifacts — HF model cache, pip build TMPDIR, and the rollout corpus — go under `/data`, never `/ephemeral`. The committed `env.sh` in the thesis repo already encodes this.
- **State at handoff** (verified 2026-07-26): user is SSH'd in; git SSH auth works (`git fetch` succeeds) and git identity is configured. The `thesis` repo is cloned at `/ephemeral/code/thesis-introspection` on `main` tracking `origin/main`; it is **not empty** — it contains `env.sh` (runtime env vars, already committed) and this document. The user's fork of `vla-safe/openvla` is confirmed to be **`github.com/James-blissett/openvla`** (HEAD matches upstream) and is **not yet cloned** to the box. Python environment, LIBERO, the model checkpoint, and all other code do not exist yet — Step 0 starts from the fork clone onward.
- **Verified environment facts** (do not re-derive): Python 3.10.12 system-wide, `python3 -m venv` works (no system `pip3` — use the venv's pip); tmux 3.2a; passwordless sudo; 8 CPUs, 144 GB RAM; L40S 46 GB VRAM, driver 565.57.01; **nvcc 12.6 at `/usr/local/cuda/bin` but NOT on PATH** — the flash-attn compile needs `export PATH=/usr/local/cuda/bin:$PATH` and `export CUDA_HOME=/usr/local/cuda` first; `libegl1` present, `libosmesa6-dev` NOT yet installed (Step 0.3 is still required); huggingface.co reachable.

## Step 0 — Environment rebuild (known-good recipe, follow exactly)

1. `cd /ephemeral/code` — `/data/tmp`, `/data/hf-cache`, and `/data/corpus` already exist; do not recreate them.
2. `df -h /data` — record free space (was ~467 GB at handoff). Abort and report if < 60 GB free.
3. System deps: `sudo apt-get update && sudo apt-get install -y libosmesa6-dev` (non-obvious but required for MuJoCo headless; verified not yet installed — this step is real, and sudo is passwordless).
4. **Already done by the user**: git auth is configured and the private `thesis` repo is cloned at `/ephemeral/code/thesis-introspection`, tracking `origin/main` — do all thesis work inside it. **Not yet done**: clone *the user's fork* `git@github.com:James-blissett/openvla.git` (name verified via `ls-remote`; HEAD matches upstream) into `/ephemeral/code`, not the upstream. Add `vla-safe/openvla` as an `upstream` remote on it for reference.
5. Python 3.10 venv. Install **in this order** (order matters):
   a. `torch==2.2.0` (cu121)
   b. LIBERO (from source) **before** safe-openvla
   c. safe-openvla — respects its pins `transformers==4.40.1`, `timm==0.9.10`
   d. flash-attn 2.5.8: `PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TMPDIR=/data/tmp MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation` (nvcc is 12.6 and lives at `/usr/local/cuda/bin` but is not on PATH — without these exports the compile fails immediately; compile takes 20-40 min on 8 cores; run in tmux; other work can proceed meanwhile)
6. Runtime env: the repo's committed `env.sh` already sets `MUJOCO_GL=egl`, `HF_HOME=/data/hf-cache`, `TMPDIR=/data/tmp` — `source env.sh` before every run rather than creating a new file. Extend it with the CUDA PATH/`CUDA_HOME` exports from 5d (that edit is a [PUSH]-checkpoint change).
7. Sanity: import torch/transformers/libero, `nvidia-smi`, one dummy OpenVLA forward. `pip check` clean.
8. [PUSH] Commit any setup scripts / env files created.

## Step 1 — Bring up OpenVLA inside LIBERO (the rollout harness)

This is the glue Step 0 does not provide. Base the driver on the fork's existing LIBERO eval script (`experiments/robot/libero/run_libero_eval.py` in the `vla-safe/openvla` fork) and adapt it into `gen_rollouts.py`. Do not rewrite from scratch — the eval script already encodes the non-obvious conventions (settle steps, image pipeline, action un-normalization, step budgets).

1. `source env.sh` (sets `HF_HOME=/data/hf-cache`) **before** any model download — the checkpoint is ~15 GB and must land on the 492 GB `/data` volume, not the 124 GB root volume that holds `/ephemeral`.
2. **Checkpoint**: `openvla/openvla-7b-finetuned-libero-10` (public on HF; the LIBERO-10-finetuned variant behind the published 53.7% figure — the reference the sanity gate checks against). Load with `torch_dtype=torch.bfloat16`, `attn_implementation="flash_attention_2"`, on cuda.
3. **`unnorm_key = "libero_10"`** — must match the checkpoint's dataset statistics. A wrong key fails *silently*: actions un-normalized against wrong statistics look plausible and score ~0%. If the sanity gate later trips low, check this first, before anything else.
4. **Task suite**: `libero_10` via LIBERO's benchmark API, using the suite's bundled initial states. Keep the eval script's conventions intact: the ~10 no-op settle steps at episode start (objects need to physically settle before the policy acts), its exact image extraction/resize pipeline, and the per-suite max-step budget (~520 for libero_10).
5. **Smoke test before anything builds on this**: one seeded rollout on task 0, end to end, under `MUJOCO_GL=egl`, in tmux. Confirm headless rendering produces real frames (not black), the episode terminates, a success flag returns, and log steps/sec (this calibrates the corpus wall-time estimate). [PUSH] the harness once the smoke test passes.

## Step 2 — Rebuild `capture.py` (from spec; the repo contains only `env.sh` and this document — no prior capture code survives, there is nothing to recover)

Build to this spec exactly:

- **Teacher-forced re-forward**: after `generate()` produces the 7 action tokens for a timestep, run one additional forward pass over the full prompt + all 7 action tokens with `output_hidden_states=True`. This is required because `generate()` never yields the hidden state at the 7th action token.
- **Positions captured**: 8 per timestep — last prompt token + the 7 action-token positions.
- **Layers captured**: the full `hidden_states` tuple including the embedding layer (index 0). The embedding layer is a required baseline later: it distinguishes signal constructed by the model's computation from signal already present in the input.
- **Storage**: fp16, ~2.1 MB/timestep. One `.pt` per rollout + JSON manifest carrying: task ID, task description string, rollout ID, seed, success flag, timestep count, capture config (positions, layers, dtype), timestamp.
- **`parity_check`**: verify the teacher-forced pass reproduces the generated action tokens (argmax agreement at each of the 7 positions). Run it on the first rollout of every generation session; hard-fail the run on mismatch.
- [PUSH] Commit `capture.py` and pass parity check **before** launching corpus generation.

## Step 3 — Mini-corpus generation (the long pole; launch early, run in tmux)

- **Config**: all 10 LIBERO-10 tasks x 5 rollouts each = 50 rollouts, seeded (seed = 1000 + rollout index for reproducibility). Vanilla OpenVLA, standard LIBERO-10 initial states, no perturbations.
- **Output location**: `/data/corpus` (already exists; 467 GB free). Never write corpus data under `/ephemeral` — the root volume only has ~105 GB free.
- Expected: failed rollouts run to the ~520-step cap while successes terminate early, so ~10-18k timesteps total, ~21-38 GB at 2.1 MB/timestep, roughly 1.5-3 h wall time (recalibrate from the smoke test's steps/sec, remembering capture doubles the forward passes). Log per-rollout wall time and success flag as generation proceeds.
- **Sanity gate**: the previously measured baseline was 52.8% success on LIBERO-10 (published reference: 53.7%). No artifact of that run survives, so this number is the only remaining ground truth for validating the entire environment rebuild. If the mini-corpus lands wildly off (< 25% or > 80% overall), the rebuild is wrong somewhere (checkpoint, pins, env vars, unnorm key) — stop and report rather than probing a broken corpus. Expect meaningful variance at 5 rollouts/task; the gate is deliberately wide.
- Write rollouts incrementally (writer flushes per rollout) so a crash loses one rollout, not the corpus.
- [PUSH] Commit the generation driver script before launch. Data itself is not committed (too large) — manifests only, if small.

## Step 4 — Linear probe (`probe_layer.py`) — locked design

Write this while the corpus generates; run it when generation completes.

- **Features**: layer-15 hidden state, mean over the 7 action-token positions. One 4096-d float vector per timestep.
- **Labels**: rollout success flag broadcast to all timesteps. Binary: 1 = failure, 0 = success.
- **Split**: `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` with `groups=rollout_id`. Never split within a rollout — adjacent timesteps are near-duplicates and within-rollout splits produce meaningless ~0.99 AUROC. Report rollout counts per class per split.
- **Model**: `StandardScaler()` then `LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42)`. Scaler fit on train only. CPU, sklearn, no PyTorch.
- **Metrics**:
  - Timestep-level test AUROC (primary metric).
  - Rollout-level AUROC via per-rollout mean and max of timestep probabilities (report, but note: only ~10 test rollouts, so treat as descriptive, not conclusive).
  - **Mandatory control**: refit with the rollout-to-label mapping permuted (then broadcast), same split. Expect AUROC ~ 0.5; if > 0.55 (tolerance widened for small n: [0.40, 0.60] acceptable at ~10k timesteps), flag loudly — indicates leakage.
- **Outputs** to `results/probe_pilot/`: `metrics.json` (all AUROCs, split sizes, class counts, layer, position scheme, seed, corpus manifest hash, timestamp) and `scores_test.npz` (per-timestep probabilities, rollout IDs, labels). Optional single PNG: histogram of per-rollout mean scores by class.
- Single self-contained script, config constants at top (`LAYER = 15`), modularise later. Runtime target < 10 min at this corpus size.
- [PUSH] Commit script before running; commit results after.

## Step 5 — Reinstate projections (`project_hidden_states.py`)

Recreates a script lost in the wipe. Prior known-good behaviour on the full 98.5k corpus: structured trajectory manifolds at mid-layers, no clean visual failure zone (expected; it motivates the probe). At 10k timesteps the structure will be sparser — note this in the summary rather than treating it as a discrepancy.

- Config constants: `LAYERS = [5, 15, 20, 25, 32]`, paths, seed.
- **Features**: identical to the probe (mean over 7 action tokens). One layer in memory at a time.
- **Pre-reduction**: PCA to 50 components per layer (`random_state=42`) before either method. Never run UMAP/t-SNE on raw 4096-d.
- **UMAP**: full mini-corpus, `n_neighbors=30`, `min_dist=0.1`, `metric='cosine'`, `random_state=42`. `pip install umap-learn` after everything else; if numba complains under Python 3.10, pin `numba<0.60`. Run `pip check` afterwards — the transformers/timm pins must survive.
- **t-SNE**: full mini-corpus (no subsampling needed at 10k), `perplexity=30`, `init='pca'`, `random_state=42`.
- **Colourings**: per layer x method, three PNGs — (a) rollout outcome, (b) task ID, (c) normalised time-within-rollout (continuous colormap). The time colouring is what exposes trajectory structure.
- **Persistence**: each 2-d embedding + index mapping (rollout ID, timestep) as `.npz` under `results/projections/`, filenames encoding layer/method/seed. Replotting must never recompute.
- [PUSH] Commit script; commit embeddings and PNGs (small at this scale).

## Acceptance criteria

1. Environment rebuilt; `pip check` clean; pins intact; parity check passing.
2. 50-rollout corpus on disk with manifests; overall success rate reported and within the sanity gate.
3. `probe_layer.py` pushed; `metrics.json` with timestep AUROC, both rollout AUROCs, shuffle-control AUROC in [0.40, 0.60].
4. `project_hidden_states.py` pushed; embeddings + PNGs for at least layer 15 (both methods, all three colourings); remaining layers may finish in tmux post-session.
5. Every [PUSH] checkpoint was surfaced to the user with a proposed commit message, and the user's pushes are verified read-only at the end (`git fetch && git log origin/main` — all expected files present in remote history).
6. No modifications to environment pins or the model.

## Interpretation guardrails (for the closing summary)

- With 50 rollouts, the probe result is a pipeline-validation and directional signal, not the thesis result. AUROC meaningfully above 0.5 at layer 15: consistent with linearly decodable failure-relevant signal mid-network; motivates the full corpus + heatmap. Do not claim more.
- AUROC near 0.5 is not a null result: the broadcast-label scheme is noisy by design and n is small. Note and stop.
- A known confound at this scale: per-task success rates vary widely (12-84% in the prior run), so the probe can partially exploit task identity. Report per-task composition of train/test splits so this is inspectable; do not attempt to correct for it today.