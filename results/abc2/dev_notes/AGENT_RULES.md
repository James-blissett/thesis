# Rules for every subagent in the more-data run (read first, follow exactly)

Repo: /ephemeral/code/thesis-introspection. Always `cd` there and `source env.sh` before python.
Spec: `more_data_collection.md` (handoff v1, what is new) and `abc_implementation_handoff.md`
(handoff v10, definitions). Where they disagree on something new, more_data_collection.md wins.

## Hard rules
- NO git command that writes (no add/commit/push/mv/stash/checkout/reset/rm). Read-only git is fine.
- Do NOT edit frozen files: `collect/`, `analysis/*.py`, `analysis/abc/`, `collect_pro/` (owned by the
  lead), the spec documents, `configs/more_data.yaml`. Import from them instead (copy-then-modify into
  your own new file is allowed). `analysis/abc2/data2.py` is owned by the lead unless your brief says
  otherwise. Only create/modify the files your brief assigns you.
- Never delete or overwrite anything under /data/rollouts_v2/, /data/rollouts_v3/ (collection is
  writing there right now — read only), constraints/, results/abc/, /data/tmp/abc_features/,
  corpus_v2_index.json, results/abc/splits/. Never write corpus_v3_index.json or
  results/abc2/splits/ (the lead does that after collection) — use dev copies under your scratch dir.
- Keep the pins transformers==4.40.1 and timm==0.9.10. Do not pip install anything.
- Never kill or attach to tmux sessions you did not create (md-w0 and md-w1 are the collection
  workers; moredata-supervisor is the supervisor). Name any tmux session you create
  `md-agent-<yourname>-<x>`, and make sure it has exited before you finish.

## Shared machine (collection runs for the next ~12 h on the same box)
- GPU: one L40S, 46 GB; the two collection workers hold ~32.5 GB. You may use at most 6 GB of GPU
  memory: call `torch.cuda.set_per_process_memory_fraction(6/46)` in any process that uses CUDA, and
  check `nvidia-smi` first. If free GPU memory is under 10 GB, use the CPU instead.
- CPU: 8 cores, the workers need ~3. Use at most 4 cores (joblib n_jobs<=4, torch.set_num_threads(4),
  OMP_NUM_THREADS=4).
- RAM: 144 GB total; stay under ~50 GB.
- Disk is TIGHT on /data (budgeted to the GB). Never write more than 3 GB in total. Put scratch in
  `/data/tmp/abc2_dev/<yourname>/` and delete your own scratch when done (only what you created).
  Memmaps created with np.lib.format.open_memmap are full-size files: size test caches small (a dev
  index of a handful of rollouts).

## Data facts
- v2 corpus: /data/rollouts_v2/<rid>/ (300; see abc handoff §1). Read pooled states with
  analysis/compute_constraints.pooled_and_hN (P=3 and P=7).
- v3 corpus (being collected): /data/rollouts_v3/<rid>/ with hidden.pt (520,33,4096) fp16 = pooled
  mean only; actions.npz, logits.pt as v2; meta.json (block, suite, base_task_idx, init_state_index,
  instruction, success_ever, success_final, t_success, ...). Only dirs with meta.json complete:true
  and not starting with "." are finished. Rollout ids `<block>_task<TT>_trial<NN>`.
- analysis/abc2/data2.py: Index2 (task = CELL id = base_task + 10*family_code, so abc's m1/m1_cut/
  kept_mask "per task" = per cell), write_index(), load_index2(), make_split2(), load_split2(),
  condition_sets(), test_sets(), demoted(). Dev index of v2 + finished v3 rollouts:
  `data2.write_index(Path('/data/tmp/abc2_dev/<you>/idx.json'))`.
- Old (abc) feature cache /data/tmp/abc_features/layer{L:02d}.npy (300,520,4096), will be deleted
  after collection; old results in results/abc/.
- No `__init__.py` in analysis/abc2 (a package named abc shadows stdlib). Scripts put
  analysis/abc and analysis/ on sys.path as analysis/abc/*.py do; don't name a new module the same as
  an analysis/abc or analysis/ module (data, prep, setup_a, ... are taken) — use a `2` suffix.

## Deliverable
- Code with a module docstring (what, outputs, usage), in the style of analysis/abc/*.py.
- A short note `results/abc2/dev_notes/<yourname>.md`: files written, how to run, the exact command(s)
  the lead should put in run_all.sh, test/parity results with numbers, anything unresolved.
- Final message to the lead: <= 25 lines, the same essentials.
