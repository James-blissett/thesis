#!/usr/bin/env bash
# Background run (results_collection_handoff.md §6): the full training set (repo key D2)
# through the same pipeline, into results/thesis_results_full/, then the original-vs-full
# comparison and the selections report. Resumable: every step skips finished work files.
# B's grid (inside run_gpu_probes.sh) is retried once; if it fails twice the failure is
# logged and everything that does not need B still runs.
# Markers: results/thesis_results_full/logs/full_run.done / .error
set -u
cd /ephemeral/code/thesis-introspection
source env.sh
export THESIS_POOL=D2 OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
OUT=results/thesis_results_full
mkdir -p $OUT/logs; rm -f $OUT/logs/full_run.done $OUT/logs/full_run.error
P=analysis/thesis_results
log() { echo "== $(date -u +%T) $*"; }
if [ ! -f $OUT/logs/gpu_probes.done ]; then
  bash $P/run_gpu_probes.sh || { log "GPU probes failed once; retrying"; bash $P/run_gpu_probes.sh; } \
    || { log "GPU probes failed twice"; echo "B grid / GPU probes failed twice" > $OUT/logs/b_grid_failed; }
fi
if [ -f $OUT/logs/b_grid_failed ]; then
  log "B grid failed twice: running the steps that do not need B"
  for s in "probes_c.py" "detectors.py --only reused a c baselines" "mlp_new.py c" "detectors.py --only mlps" \
           "evaluate.py" "checks.py" "tables.py" "figures.py"; do
    python $P/$s || log "FAILED: $s"
  done
else
  touch $OUT/logs/gpu_probes.done
  bash $P/run_main.sh || { echo "main pipeline failed (see main_pipeline log)" > $OUT/logs/full_run.error; exit 1; }
fi
python $P/full_compare.py || { echo "full_compare failed" > $OUT/logs/full_run.error; exit 1; }
THESIS_POOL=D0 python $P/tables.py t6 || true
date -u +%FT%TZ > $OUT/logs/full_run.done
