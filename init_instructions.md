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

---
 
## Addendum A — Capture amendments (supersedes parts of Step 2; apply BEFORE corpus generation)
 
**Rationale**: later thesis phases derive per-timestep failure labels post-hoc from consistency constraints — programmatic checks on robot/object state and a VLM observer reading camera frames. That labeling is only post-hoc if the things constraints evaluate are stored. Hidden states alone support exactly one labeling scheme (rollout-outcome broadcast) forever. Stored observations are a capture-time commitment of the same kind as token position: irreversible.
 
### A1 — Extended per-timestep record
 
Add to each per-rollout `.pt`, all arrays sharing the same length T as the hidden-state tensor:
 
- **`frames_agentview`**: the raw env-rendered agentview RGB per timestep, JPEG-encoded bytes (quality ~90). Store the raw render, not the policy-resized version — the policy input can always be recomputed from it; the reverse is lossy. Add the wrist/eye-in-hand camera under `frames_wrist` if the env provides one.
- **`obs`**: dict of float32 arrays from the LIBERO observation — end-effector position and orientation, gripper joint state, and object states as the env exposes them. This is privileged sim state; that is acceptable because it shares the VLM's training-time-only status.
- **`actions`**: the executed un-normalized 7-DoF action, float32 [T, 7].
- Manifest gains a `capture_schema_version` field; set it to 2.
Size delta ~0.05 MB/timestep against 2.1 MB of hidden states — negligible; storage estimates in Step 3 are unchanged.
 
### A2 — Last-action-token mechanics (exact spec for the teacher-forced re-forward)
 
The subtlety, stated precisely so it is implemented rather than approximated:
 
1. **Why `generate()` cannot provide it**: a transformer emits the hidden state at position i only when the token at position i is an *input*. The state at position i is what predicts token i+1 (the off-by-one). The 7th action token is produced as output on the final decode step and never fed back in, so its hidden state is never computed during generation.
2. **The re-forward**: after generation, concatenate prompt (length P) + all 7 generated action tokens into one sequence and run a single forward with `use_cache=False`, `output_hidden_states=True`. `hidden_states` is a tuple of (n_layers + 1) tensors, each [1, P+7, 4096]; index 0 is the embedding layer.
3. **Slice positions [P-1, P, P+1, ..., P+6]** — eight positions. P-1 is the last prompt token; P+6 is the 7th action token as input, the state that motivates this entire mechanism.
4. **This is exact, not approximate**: causal attention means the state at position i depends only on tokens ≤ i, so positions P-1 through P+5 reproduce the generation pass's states (up to floating-point nondeterminism), and P+6 is computed under identical conditions.
5. **Parity check, operationalized**: argmax of the re-forward logits at position i must equal the generated token at position i+1, for i in [P-1, P+5]. The logits at P+6 predict a token that was never sampled — there is nothing to check there; do not flag it as a mismatch.
6. **Precision**: states compute in bf16; cast to fp16 on save and `assert torch.isfinite(...).all()` after the cast — bf16's range exceeds fp16's, and an overflow becomes a silent `inf` in the corpus.
### A3 — Mid-flight application rule
 
- If `capture.py` exists but corpus generation has not launched: patch, re-pass parity, [PUSH], then launch.
- If generation is already running: stop it, patch, restart from rollout 0. A mixed-schema corpus is worth less than the partial hours it cost at this scale.
### A4 — Acceptance criteria (append)
 
7. Every rollout `.pt` contains `hidden`, `frames_agentview`, `obs`, `actions` with matching T and `capture_schema_version: 2` in its manifest.
8. One stored frame decoded and visually inspected — real scene content, not black (the EGL failure mode produces well-formed all-black JPEGs that pass every structural check).

---

## STATUS — end of session 2026-07-26

**Done: Steps 0-3. Remaining: Steps 4-5 (scripts written and validated, not yet run).**

### Where the data is

- **Corpus: `/data/corpus` — 50/50 rollouts, 19,289 timesteps, 40 GB.** On `/dev/vdb`, a
  *different physical volume* from `/ephemeral` (`/dev/vda1`). The previous wipes hit
  `/ephemeral`; `/data` survived them. The `.pt` files are far too large for git and are
  **not** backed up anywhere else — treat them as reproducible-but-expensive (~1.4 h to
  regenerate via `python gen_rollouts.py`), not as safe.
- **Manifests: `manifests/` in this repo (236 KB, 51 files)** — the 50 per-rollout
  manifests plus `session_summary.json`. These are committed, so the corpus's provenance
  (seeds, success flags, parity stats, per-rollout timings) survives even if `/data` is lost.
- Model checkpoint cached at `/data/hf-cache` (15 GB), venv at `/ephemeral/code/venv`.

### Sanity gate: PASSED

**30/50 = 60.0% success**, against 52.8% previously measured and 53.7% published. Inside
the 25-80% gate. Per-task successes (t0..t9): `2 3 5 2 2 5 3 2 3 3` — the wide spread the
doc predicts, and the task-identity confound to report per Step 4.

### Environment: six corrections to Step 0

The Step 0 "verified facts" were wrong in six places; all are fixed and encoded in
`setup_env.sh`, which rebuilds the environment unattended. Do not re-derive these:

1. `python3 -m venv` fails — `ensurepip` missing; needs `apt install python3.10-venv`.
2. LIBERO's top-level `libero/` has no `__init__.py`, so PEP 660 editable installs map
   nothing; install with `--config-settings editable_mode=compat`.
3. LIBERO prompts interactively on first import; pre-seed `~/.libero/config.yaml`.
4. **robosuite 1.4.0 needs the mujoco 2.3.x C API** — pip resolves 3.x, whose `mj_fullM`
   signature differs, failing at env construction. Pin `mujoco==2.3.2`.
5. `opencv-python 5.x` requires numpy>=2, conflicting with torch 2.2 / TF 2.15. Pin
   `opencv-python==4.10.0.84`.
6. `tensorflow-metadata 1.21` needs protobuf>=5.27, which TF 2.15 forbids. Pin
   `tensorflow-metadata==1.14.0`, which forces protobuf 3.20.3, which in turn requires
   `wandb==0.16.6` to keep `pip check` clean.

`pip check` is clean and all required pins are intact (`transformers 4.40.1`,
`timm 0.9.10`, `tokenizers 0.19.1`, `numpy 1.26.4`, `flash-attn 2.5.8`).
Measured throughput: **3.94 policy steps/s** including the capture re-forward;
**~2.2 MB/timestep** on disk.

### Parity gate — amended, with the measurement behind it

Addendum A2.5's bit-exact argmax gate **cannot pass in bf16**, and the reason is
structural, not a bug. `generate()` decodes through flash-attn's cached-decode kernel
while the re-forward uses the prefill kernel; over 32 bf16 layers the final logits
diverge by mean 0.75 / max 1.375. Because OpenVLA maps adjacent action bins to adjacent
vocabulary ids, that noise flips the argmax wherever the top-2 logits are nearly tied.

Measured over a full 520-step rollout (3,640 positions): **96.6% agreement; all 122
mismatches had a top-2 gap <= 0.875**, versus a **median gap of 8.5** where the passes
agreed. A wiring bug looks nothing like this (~0% agreement, large margins).

Per A2.4's "up to floating-point nondeterminism", the gate in `capture.py` now requires
rollout-level agreement >= 95% **and** every mismatch inside a 2.0 tie band, so real bugs
still hard-fail. Every rollout manifest records its own agreement rate and mismatch
margins, so the threshold stays inspectable. **The stored hidden states are unaffected:**
the re-forward is fed the tokens that were actually generated and executed, so the tie
affects only the verification logit, never the representation.

### Pick up here

```bash
cd /ephemeral/code/thesis-introspection && source env.sh
python probe_layer.py                            # ~1 min; -> results/probe_pilot/
python project_hidden_states.py --layers 15      # then remaining layers in tmux
```

Both scripts are written and validated end-to-end against a synthetic 12-rollout corpus
(probe recovered an injected signal at AUROC 1.0 with an uninflated shuffle control;
both projection methods emitted all 3 colourings and persisted embeddings) and their
feature extraction is confirmed against the real corpus (1.5 s / 3 rollouts, so ~25 s
for all 50 — well inside the <10 min target). Neither has been run on the real corpus,
so **no AUROC number exists yet.**

On reading the result, follow the interpretation guardrails above: check the
shuffle-control AUROC lands in [0.40, 0.60] *before* reading anything into the primary
number, and report the per-task split composition, since per-task success ranges 40-100%
here and the probe can partially exploit task identity.
