#!/usr/bin/env bash
# run_all.sh -- the one resumable entrypoint of the more-data run
# (more_data_collection.md, handoff v1, section 1). Re-running it carries on from where
# it stopped: every step has a done-marker in results/abc2/logs/done_<step> and the
# scripts themselves skip finished work.
#
# Exit codes: 0 = everything finished (report written); 10 = collection still running
# (workers (re)started in tmux md-w*); 1 = a required step failed (see
# results/abc2/logs/run_all.error and the step's log).
#
# Steps after collection: gate re-run on the old cache (Step 8) -> index/audit/demotion/
# near-dup (Step 5) -> splits (Step 7) -> constraints + parity (Step 6) -> delete the old
# cache if the disk rule needs it, build + verify the new cache (Step 6) -> pipeline
# D0 D1 D2 (Step 9) -> paired bootstrap -> report -> MLP (Step 10) -> B H=10 (Step 11) ->
# final report. MLP and B never block the report.
set -uo pipefail
cd /ephemeral/code/thesis-introspection
source env.sh >/dev/null 2>&1
L=results/abc2/logs
mkdir -p "$L"
ERR="$L/run_all.error"
rm -f "$ERR"
OUT=/data/rollouts_v3
export PYTHONUNBUFFERED=1

note() { echo "[run_all $(date -u +%FT%H:%M:%SZ)] $*" | tee -a "$L/run_all.log"; }
fail() { note "FAILED: $*"; echo "$(date -u +%FT%H:%M:%SZ) $*" > "$ERR"; exit 1; }
done_() { [ -f "$L/done_$1" ]; }
mark() { date -u +%FT%H:%M:%SZ > "$L/done_$1"; }
step() {  # step NAME CMD... : run once, log to $L/NAME.log, mark done on success
  local name=$1; shift
  if done_ "$name"; then note "skip $name (done)"; return 0; fi
  note "start $name: $*"
  if "$@" >> "$L/$name.log" 2>&1; then mark "$name"; note "done $name"; return 0; fi
  return 1
}

# --- Step 4: collection -------------------------------------------------------------
N=$(python -c "import yaml;print(yaml.safe_load(open('configs/more_data.yaml'))['N_WORKERS'])")
collection_done() {
  for k in $(seq 0 $((N-1))); do
    tmux has-session -t md-w$k 2>/dev/null && return 1
    [ -f "$OUT/worker$k.finished" ] || return 1
  done
  return 0
}
if ! done_ collect; then
  if collection_done; then
    mark collect; note "collection finished"
  else
    bash collect_pro/launch_workers.sh | tee -a "$L/run_all.log"
    note "collection running; re-run run_all.sh when md-w* have exited"
    exit 10
  fi
fi

# --- Step 8 re-run (lead's own check, needs the old cache, full GPU) ---------------------
if [ -d /data/tmp/abc_features ] && ! done_ gate_rerun; then
  [ -f results/abc2/regression_gate_agentrun.json ] || cp results/abc2/regression_gate.json results/abc2/regression_gate_agentrun.json
  [ -f results/abc2/regression_gate_agentrun.md ] || cp results/abc2/regression_gate.md results/abc2/regression_gate_agentrun.md
  rm -rf /data/tmp/abc2_dev/gate_rerun
  if step gate_rerun env ABC2_GPU_FRAC=0.9 python analysis/abc2/regression_gate.py --scratch /data/tmp/abc2_dev/gate_rerun; then
    rm -rf /data/tmp/abc2_dev/gate_rerun
  else
    note "gate re-run exited non-zero (see $L/gate_rerun.log); continuing, report shows the gate result"
    mark gate_rerun
  fi
fi

# --- Step 5: index, audit, demotion, lan near-duplicates ---------------------------------
step index python analysis/abc2/build_index.py || fail "build_index"
# --- Step 7: splits -----------------------------------------------------------------------
step splits python analysis/abc2/data2.py --splits || fail "splits"
# --- Step 6: constraints ------------------------------------------------------------------
step constraints_parity python analysis/abc2/constraints.py --parity || fail "constraints parity"
step constraints python analysis/abc2/constraints.py || fail "constraints"

# --- Step 6: feature cache (disk rule: delete the old cache only if needed) -------------
if ! done_ cache; then
  NEED=$(python - <<'PY'
import json, shutil
n = json.load(open("corpus_v3_index.json"))["n_rollouts"]
need = n * 33 * 520 * 4096 * 2 / 1e9 + 30 + 2
free = shutil.disk_usage("/data").free / 1e9
print("delete" if free < need else "keep", f"{need:.0f}", f"{free:.0f}")
PY
)
  note "cache disk check: $NEED (action need_GB free_GB)"
  if [[ $NEED == delete* ]] && [ -d /data/tmp/abc_features ] && [ ! -f /data/tmp/abc2_features/done.npy ]; then
    done_ gate_rerun || fail "refusing to delete the old cache before the gate re-run"
    note "deleting /data/tmp/abc_features (disk rule, Step 1.5)"
    rm -rf /data/tmp/abc_features
    echo "- **$(date -u +%FT%H:%MZ)** — Step 6: deleted /data/tmp/abc_features (42 GB) under the Step 1 disk rule before building /data/tmp/abc2_features ($NEED)." >> results/abc2/RUNLOG.md
  fi
  step cache_build python analysis/abc2/build_cache2.py || fail "cache build"
  step cache python analysis/abc2/build_cache2.py --verify || fail "cache verify"
fi

# --- Step 9: main experiment ----------------------------------------------------------------
step pipeline env ABC2_GPU_FRAC=0.95 OMP_NUM_THREADS=4 python analysis/abc2/pipeline.py --cond D0 D1 D2 --n-jobs 4 || fail "pipeline"
step bootstrap python analysis/abc2/bootstrap2.py || fail "bootstrap T0"
step bootstrap_t12 python analysis/abc2/bootstrap2.py --test-set T1 T2 || note "bootstrap T1/T2 failed (optional)"
python analysis/abc2/report2.py >> "$L/report.log" 2>&1 || fail "report (first pass)"
note "report written (before MLP/B)"

# --- Step 10: MLP (must not block the report) ------------------------------------------------
if ! done_ mlp; then
  if env ABC2_GPU_FRAC=0.95 OMP_NUM_THREADS=3 python analysis/abc2/stage10_mlp.py --cond D0 D2 >> "$L/stage10_mlp.log" 2>&1; then
    note "MLP done"
  else
    python analysis/abc2/stage10_mlp.py --mark-failed "stage10_mlp.py exited non-zero (see logs/stage10_mlp.log)" >> "$L/stage10_mlp.log" 2>&1
    note "MLP failed (marked)"
  fi
  mark mlp
  python analysis/abc2/report2.py >> "$L/report.log" 2>&1
fi

# --- Step 11: B at H = 10 (must not block the report) ------------------------------------------
if ! done_ b10; then
  if env ABC2_GPU_FRAC=0.95 OMP_NUM_THREADS=4 python analysis/abc2/stage5_b.py --cond D0 D2 --n-jobs 4 >> "$L/stage5_b.log" 2>&1; then
    note "B done"
  else
    python analysis/abc2/stage5_b.py --mark-failed "stage5_b.py exited non-zero (see logs/stage5_b.log)" >> "$L/stage5_b.log" 2>&1
    note "B failed (marked)"
  fi
  python analysis/abc2/stage5_b.py --check-d0 >> "$L/stage5_b.log" 2>&1 || true
  mark b10
fi

python analysis/abc2/report2.py >> "$L/report.log" 2>&1 || fail "report (final)"
mark all
note "ALL DONE: results/abc2/REPORT.md"
exit 0
