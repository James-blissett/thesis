# Dev note: `constraints` (Step 6: v3 constraints + feature cache)

## Files written
- `analysis/abc2/constraints.py`: Step 6 constraints for v3 rollouts. Writes `constraints_v3/<rid>.npz` with `act_mag, act_dir, grip_flip, emb_temp(520,32), xl_adj(520,31), xl_final(520,31), xl_spread`, all float32. Same keys, shapes and NaN conventions as `constraints/`. No `xl_final_hN`, no `*_nosink`, no `act_rep`.
  It calls the frozen `compute_constraints` functions unchanged: `normalised_actions` → `action_constraints`, and `hidden_constraints` (→ `cos_dist_pair`). For v3 the pooled state is `hidden.pt.float()`, the same op `pooled_and_hN` applies to the P=3 pooled position. `hidden_constraints` is given `hN=pooled, g=1` because those only feed `xl_final_hN`, which is thrown away.
  Writes are atomic (`.<rid>.tmp.npz` then `os.replace`). It skips outputs that pass the key/shape/dtype check. It refuses `--out-dir constraints`. It reads only v3 dirs whose meta.json has `complete: true`.
- `analysis/abc2/build_cache2.py`: Step 6 feature cache (imports `open_or_create` and `read_rollout` from `analysis/abc/build_cache.py`). v2 rows go through `pooled_and_hN` → fp16, the same as the old cache. v3 rows are copied unchanged from hidden.pt.
  Disk check before anything is created: free ≥ (rollouts still to write × 0.141 GB) + 30 GB (`--headroom-gb`). It prints the numbers and exits 1 otherwise, creating nothing. During the build it stops, resumably, if free space drops under the headroom. Resume works through `done.npy`, which is written after the flush.
- `results/abc2/constraints_parity.json`: parity report.
- `constraints_v3/*.npz`: real outputs for the 6 v3 rollouts finished at the time (ori2_task00/01, obj, lan, swap, task task00 trial00).

## Commands for run_all.sh
```
python analysis/abc2/constraints.py --parity            # exits 1 on failure
python analysis/abc2/constraints.py                     # v3 rows of corpus_v3_index.json -> constraints_v3/
python analysis/abc2/build_cache2.py                    # -> /data/tmp/abc2_features/ (needs ~141 GB + 30 GB free)
python analysis/abc2/build_cache2.py --verify           # exits 1 on failure
```
(`constraints.py --all-finished` scans /data/rollouts_v3 directly, so it can run before the index exists. `--rids ...` handles specific rollouts.) Both scripts are single-process. I ran them pinned to 2 cores (`taskset -c 6,7`, `OMP_NUM_THREADS=2`), and they call `torch.set_num_threads(2)`.

## Test / parity results
- **Constraint parity (5 v2 rollouts, task00_trial00 and task01_trial02 at P=7; task03_trial05, task05_trial27, task09_trial29 at P=3): PASSED and bit-exact.** Max |diff| is 0.0 on every key of every rollout, with 0 NaN mismatches (e.g. task00_trial00 has act_dir 234 NaN, emb_temp 32, act_mag/grip_flip 1 each, all in the same places). Dtypes match.
- **v3 spot-check** (`ori2_task00_trial00` against the 30 v2 task-0 rollouts): shapes are correct. NaN at t=0 for act_mag/act_dir/grip_flip/emb_temp, and none for the xl_* series, the same as v2. Value ranges, v3 against v2 (min/median/max):
  - act_mag 0/0.101/1.70 vs 0/0.145/2.44
  - act_dir median 1.45 vs 1.42
  - emb_temp 0/0.036/0.85 vs 0/0.040/0.89
  - xl_adj 0.004/0.052/0.82 vs 0.004/0.051/0.85
  - xl_final 0.04/0.41/1.01 vs 0.03/0.39/1.03
  - xl_spread 0.073/0.080/0.086 vs 0.071/0.079/0.089
  
  A second `--all-finished` run skipped all 6 (resume works).
- **Cache dev test** (cut-down `data2.write_index` copy: task00_trial00 P=7, task03_trial05, task05_trial27, ori2_task00_trial00, obj_task00_trial00). I built 3 rollouts with `--limit 3`, then resumed for the other 2. `--verify` PASSED:
  - order matches
  - 5/5 done
  - layer 15 of all 3 v2 rows equals the old cache exactly (looked up by id through `/data/tmp/abc_features/rollout_ids.json`)
  - both v3 rollouts equal hidden.pt on all 33 layers
  - tokens match 5/5

  With `--old-cache` pointed at a missing dir, the fallback compared against `pooled_and_hN` (P=7 + 2×P=3) and PASSED. Negative test: after corrupting one v2 layer-15 value and one v3 token, verify FAILED, naming exactly those two rollouts (exit 1). Disk refusal test: `--headroom-gb 1000` refused (exit 1) and created nothing.
- All scratch (`/data/tmp/abc2_dev/constraints/`) has been deleted. Peak scratch use was about 1.1 GB.

## Runtime estimates (measured on this box while collection was running)
- Constraints: about 1.0 s per v3 rollout with hidden.pt warm in page cache. A cold v2 P=3 hidden.pt (0.42 GB) read in 1.66 s (≈0.25 GB/s), so a cold v3 read (0.14 GB) adds about 0.55 s, giving about 1.1–1.5 s per rollout. **700 v3 rollouts: about 13–18 min.**
- Cache: 8-rollout timing build (3 cold v2 P=3, 1 cold v2 P=7, 4 warm v3) took 15.6 s. That works out to about 2.9 s per v2 P=3, about 5 s per P=7, and about 0.5 s per warm v3 (about 1.1 s cold). **Full ~1000 rollouts (270 P=3 + 30 P=7 + 700 v3; ~241 GB read, ~141 GB written on the same /data disk): about 28 min, and up to ~45 min if the disk is contended.**

## Unresolved / notes
- With the old cache present, `--verify` compares all v2 rows of layer 15 against it, which reads about 2.5 GB. That is cheap.
- If `/data/tmp/abc_features/` has been deleted by the time the real cache is verified, verify falls back to `pooled_and_hN` on 5 v2 rollouts, including a P=7 one.
- `build_cache2.py` checks free space against what is still to be written. The memmaps are sparse, so a resumed build only needs the remainder.
