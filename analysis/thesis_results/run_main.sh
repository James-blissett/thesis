#!/usr/bin/env bash
# Everything after the GPU probe refits, for one training set (THESIS_POOL=D0 main run,
# D2 = background run via run_full.sh). Each step skips finished work files, so the
# whole script is resumable: re-run it after any failure.
# Markers: <out>/logs/main_pipeline.done / .error
set -u
cd /ephemeral/code/thesis-introspection
source env.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
OUT=results/thesis_results; [ "${THESIS_POOL:-D0}" = "D2" ] && OUT=results/thesis_results_full
mkdir -p $OUT/logs; rm -f $OUT/logs/main_pipeline.done $OUT/logs/main_pipeline.error
P=analysis/thesis_results
step() { echo "== $(date -u +%T) $*"; "$@" || { echo "FAILED: $*" > $OUT/logs/main_pipeline.error; exit 1; }; }
[ -f $OUT/logs/gpu_probes.done ] || step bash $P/run_gpu_probes.sh
step python $P/probes_c.py
step python $P/detectors.py --only reused a c b b_targets baselines
step python $P/mlp_new.py c
step python $P/mlp_new.py b
step python $P/detectors.py --only mlps
step python $P/evaluate.py
step python $P/checks.py
step python $P/profiles.py
step python $P/b_constraint.py
step python $P/paired.py
step python $P/corpus.py
[ "${THESIS_POOL:-D0}" = "D0" ] && step python $P/example.py
step python $P/tables.py
step python $P/figures.py
date -u +%FT%TZ > $OUT/logs/main_pipeline.done
