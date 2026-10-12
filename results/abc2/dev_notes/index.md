# Dev note: `index` (Step 5 index / audit / demotion / lan near-dup; Step 7 split checks)

## Files
- NEW `analysis/abc2/build_index.py`: Step 5. Writes `corpus_v3_index.json` (via `data2.write_index`), then
  `audit_block_task.csv`, `audit_block.csv`, `audit.md`, `demotions.json`, `lan_neardup.csv`, `lan_neardup.json`
  under `--out-dir` (default `results/abc2/`). `--index-out` (default repo root `corpus_v3_index.json`),
  `--rollouts-v3` (default `/data/rollouts_v3`). `--selftest` writes nothing and checks data2 (below).
  CPU only, reads meta/actions/logs only, about 1 s.
- EDITED `analysis/abc2/data2.py` (public names and Index2 fields unchanged):
  1. `v2_entries()`: corpus_v2_index.json has no `task_description`, so every v2 entry had `instruction: null`.
     Now it is read from `/data/rollouts_v2/<rid>/manifest.json` (`task_description`); v2 entries also get `suite: libero_10`.
  2. `v3_entries()`: a v3 dir counts only if meta.json says `complete: true` AND hidden.pt, logits.pt and actions.npz exist
     (same test as `collect_pro.common.is_complete`).
  3. `make_split2()`: `set(idx.rids)` was rebuilt for every element of the final list comprehension. It is now built once.
     The output is the same.
  4. `load_split2(seed, idx, splits_dir=None)`: new optional arg so dev runs can read splits written with `--splits-dir`.
  No logic bug was found in make_split2 / condition_sets / test_sets; all checks below pass unchanged.

## Commands for the lead (after collection, from repo root, `source env.sh`)
```
python analysis/abc2/build_index.py --selftest      # optional; writes nothing; must end "[selftest] PASSED"
python analysis/abc2/build_index.py                 # corpus_v3_index.json + results/abc2/{audit*,demotions.json,lan_neardup.*}
python analysis/abc2/data2.py --splits              # results/abc2/splits/seed{0,1,2}.json (reads demotions only via helpers)
```
Order matters: `demotions.json` must exist before any stage calls `condition_sets`/`test_sets` with the default
`demotions=None`. build_index always rebuilds its outputs from disk (deterministic). It does not skip on existing outputs,
so in run_all.sh guard it with `[ -f corpus_v3_index.json ] ||` if you want skip-on-exists.

## Definitions chosen
- Audit `planned`: `collect_pro/common.jobs(cfg, load_assignment(cfg))` per cell. For `ori`, the v2 assignment (30 per task).
  `failed_rollouts`: ids in failures.jsonl that never completed (excluding `fatal` disk-stop records). `not_run` = planned −
  collected − failed. `cell_skipped` uses collect.py's own rule, recomputed from progress.jsonl and failures.jsonl (a cell whose
  3 most recent events are failures, and which is short of its plan). collect.py only prints cell skips, it does not log them.
- Hours: `collect_started_utc` to the last `finished_utc` in progress.jsonl. Rollouts/hour counts only rollouts finished in
  that window. The 4 smoke-test rollouts (obj/lan/swap/task _task00_trial00) finished before the start and are reported
  separately.
- Demotion: rates for all blocks are written. Only obj and lan are eligible. A block with 0 rollouts is not evaluated.
- lan near-dup: the partner is the unperturbed (ori/ori2) rollout with the same base task and init_state_index. ori2 is the
  complement of v2 within libero_10's 50 states, so every lan rollout gets exactly one partner once ori2 is complete.
  The statistic is the share of the 520 steps with all 7 action tokens equal at the same t. Extra columns: share of
  individual tokens equal, first differing step, and the mean share against every unperturbed rollout of the same task at
  another init state (`lan_vs_nonpartner`). The reference covers ALL same-task pairs of ori+ori2 at different init states
  (no sampling). The caveat (lan init states are fresh draws, not the same scenes) is in the json.
- make_split2 groups by (base task, init_state_index) across blocks, as the spec says, even though obj/lan/ori2 indices
  point into different suites' init files (all 5 suites' init files differ; I checked md5 per task). This is harmless: it
  only adds constraints. One consequence: obj/lan rollouts whose index equals a v2 index inherit that v2 rollout's side.

## Test results (`--selftest`, 156 checks, all PASS)
- v2-only Index2: kept rows 38,618 success / 84,597 failure; rids/task/y/kept_len identical to abc `Index`; `m1` on T0
  with Index2 == abc `m1` with abc Index (random score, seeds 0/1/2: per-task 0.5152/0.6309/0.4541, exact dict equality).
- Real index (300 v2 + 8 v3 at 12:12 UTC) and synthetic index (300 real v2 + 700 synthetic v3, one per planned job, real
  init_state_index, random outcomes, t_success for successes), seeds 0, 1, 2:
  unseen tasks equal results/abc/splits (5,6,9 / 3,4,7 / 1,2,6). Every rollout appears in exactly one of
  train/eval_seen/unseen/stress. Every base task, in any block, is on one side. v2 train/eval_seen/unseen membership is
  unchanged (126/84/90). No (task, init) group is in both train and eval_seen. New rollouts in a v2 group inherit its
  side. Drawn groups split exactly round(0.6 n) per task (synthetic: 12/8 in all 7 seen tasks; ~170 new rollouts
  inherited and ~250 drawn per seed). swap/task appear only in stress, and all of them are there.
  Synthetic sizes, seed 0: train/eval/unseen/stress 378/252/270/100.
- Conditions with demotions (), (lan), (obj, lan): D0/D1/D2 contain only their allowed blocks. D2 drops demoted blocks.
  D0 train/eval_seen equal the old split. cal = eval_seen successes. Synthetic seed 0: D0 126/84, D1 210/140,
  D2 378/252 (290/200 with lan demoted).
- Test sets: T0 = the old unseen list (90). T1 ⊂ T2. T2 = unseen minus demoted blocks (synthetic 150 ⊂ 270).
  stress_seen/unseen partition swap+task+demoted by base task (70/30; 210/90 with lan demoted).
- Per-cell M1 with Index2: m1_cut is per cell. A timestep-only score gives per_task exactly 0.5 in every cell
  (T0: 3 cells; synthetic T2: 9 cells = unperturbed/obj/lan × 3 tasks), and n_tasks = number of qualifying cells.
- build_index on real data into scratch, 12:12 UTC: 308 rollouts. ori 137/300; ori2 1/4; obj, lan, swap, task 0/1 each.
  obj and lan are "demoted" on that partial data, which is meaningless this early. lan near-dup: 0/1 lan with a partner
  (its init 25 is an uncollected ori2 index). Reference over 4,470 unperturbed different-init pairs: mean 0.0158,
  median 0.0. Partner path checked with fake lan entries: self-pair share 1.0; a different-init pair 0.006, with step 0
  already different. Audit failure/skip logic checked with a fake failures.jsonl: 3 failures → cell skipped; a fatal
  disk stop is counted separately.
- `data2.py --splits --splits-dir <scratch>` on the dev index ran, and `load_split2(..., splits_dir)` + `test_sets` read it back.

## Unresolved / for the lead
- The near-dup premise fails (RUNLOG Step 1.4). Expect the lan-vs-partner share to be about equal to the reference
  (≈0.02). Report it with the caveat.
- Read demotion results only after collection: with partial data, obj/lan are trivially outside 10–90%.
- Scratch `/data/tmp/abc2_dev/index/` deleted.
