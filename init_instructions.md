# Handoff: Environment Rebuild, Mini-Corpus, and Single-Layer Linear Probe

## Role and scope things 

The ephemeral volume was wiped: environment, capture code, and the full rollout corpus are gone. You are rebuilding the environment from a known-good recipe, recovering or recreating the hidden-state capture module, generating a 50-rollout mini-corpus, training a linear probe on one layer, and reinstating a projection-visualisation script. The design below is fixed. Do not redesign, do not add an MLP, do not modify the model. Priority order if time runs short: environment > capture > corpus > probe > projections.

**Git discipline is the standing failure mode of this project.** Work was lost twice because nothing was pushed. The user is the sole author of commits: **never run `git add`, `git commit`, or `git push` yourself.** Wherever this doc says "commit X" or marks a [PUSH] checkpoint, it means: pause, tell the user exactly which files are ready and why now is a checkpoint, propose a one-line commit message, and wait for the user to confirm they have committed and pushed before starting any dependent long-running step. Read-only git (`status`, `diff`, `log`) is fine and encouraged for verifying state. Nothing long-running starts until the code it depends on is confirmed pushed.

## Context

- Thesis: runtime failure detection for OpenVLA (7B, Llama-2 backbone, 33 transformer layers, d_model=4096) on LIBERO-10.
- This probe pilots a full layer x timestep AUROC heatmap (thesis Contribution 3). Linear probes are deliberate: limited expressivity makes high AUROC evidence about the representation itself.
- Hardware: Brev L40S 48GB (Crusoe). Work in `/ephemeral/code/thesis-introspection`.

## Step 0 — Environment rebuild (known-good recipe, follow exactly)

1. `mkdir -p /ephemeral/tmp /ephemeral/code && cd /ephemeral/code`
2. `df -h /ephemeral` — record free space. Abort and report if < 60 GB free.
3. System deps: `sudo apt-get update && sudo apt-get install -y libosmesa6-dev` (non-obvious but required for MuJoCo headless).
4. Clone the private `thesis` repo (github.com/James-blissett/thesis — currently empty; cloning it anyway verifies auth works and establishes the push target before anything is written) and the `vla-safe/openvla` rollout-generation fork. If git auth is not yet configured on this instance, stop and ask the user to complete the SSH-key/PAT step first.
5. Python 3.10 venv. Install **in this order** (order matters):
   a. `torch==2.2.0` (cu121)
   b. LIBERO (from source) **before** safe-openvla
   c. safe-openvla — respects its pins `transformers==4.40.1`, `timm==0.9.10`
   d. flash-attn 2.5.8: `TMPDIR=/ephemeral/tmp MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation` (compile takes 20-40 min; run in tmux; other work can proceed meanwhile)
6. Runtime env: `export MUJOCO_GL=egl` (add to venv activate or a sourced env file in the repo).
7. Sanity: import torch/transformers/libero, `nvidia-smi`, one dummy OpenVLA forward. `pip check` clean.
8. [PUSH] Commit any setup scripts / env files created.

## Step 1 — Rebuild `capture.py` (from spec; the repo is empty, there is nothing to recover)

Build to this spec exactly:

- **Teacher-forced re-forward**: after `generate()` produces the 7 action tokens for a timestep, run one additional forward pass over the full prompt + all 7 action tokens with `output_hidden_states=True`. This is required because `generate()` never yields the hidden state at the 7th action token.
- **Positions captured**: 8 per timestep — last prompt token + the 7 action-token positions.
- **Layers captured**: the full `hidden_states` tuple including the embedding layer (index 0). The embedding layer is a required baseline later: it distinguishes signal constructed by the model's computation from signal already present in the input.
- **Storage**: fp16, ~2.1 MB/timestep. One `.pt` per rollout + JSON manifest carrying: task ID, task description string, rollout ID, seed, success flag, timestep count, capture config (positions, layers, dtype), timestamp.
- **`parity_check`**: verify the teacher-forced pass reproduces the generated action tokens (argmax agreement at each of the 7 positions). Run it on the first rollout of every generation session; hard-fail the run on mismatch.
- [PUSH] Commit `capture.py` and pass parity check **before** launching corpus generation.

## Step 2 — Mini-corpus generation (the long pole; launch early, run in tmux)

- **Config**: all 10 LIBERO-10 tasks x 5 rollouts each = 50 rollouts, seeded (seed = 1000 + rollout index for reproducibility). Vanilla OpenVLA, standard LIBERO-10 initial states, no perturbations.
- Expected: ~10k timesteps total, ~21 GB, ~1.5-2.5 h wall time. Log per-rollout wall time and success flag as generation proceeds.
- **Sanity gate**: the previously measured baseline was 52.8% success on LIBERO-10 (published reference: 53.7%). No artifact of that run survives, so this number is the only remaining ground truth for validating the entire environment rebuild. If the mini-corpus lands wildly off (< 25% or > 80% overall), the rebuild is wrong somewhere (checkpoint, pins, env vars, unnorm key) — stop and report rather than probing a broken corpus. Expect meaningful variance at 5 rollouts/task; the gate is deliberately wide.
- Write rollouts incrementally (writer flushes per rollout) so a crash loses one rollout, not the corpus.
- [PUSH] Commit the generation driver script before launch. Data itself is not committed (too large) — manifests only, if small.

## Step 3 — Linear probe (`probe_layer.py`) — locked design

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

## Step 4 — Reinstate projections (`project_hidden_states.py`)

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