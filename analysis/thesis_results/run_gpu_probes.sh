#!/usr/bin/env bash
# GPU probe refits for one training set (THESIS_POOL=D0 main, D2 background):
# A's GPU-vs-sklearn check, A's 34 probes, B's grid. Resumable (each step skips
# finished work files). Markers: <out>/logs/gpu_probes.done / .error
set -u
cd /ephemeral/code/thesis-introspection
source env.sh
OUT=results/thesis_results; [ "${THESIS_POOL:-D0}" = "D2" ] && OUT=results/thesis_results_full
mkdir -p $OUT/logs; rm -f $OUT/logs/gpu_probes.done $OUT/logs/gpu_probes.error
export OMP_NUM_THREADS=8
step() { echo "== $(date -u +%T) $*"; "$@" || { echo "FAILED: $*" > $OUT/logs/gpu_probes.error; exit 1; }; }
[ -f $OUT/logs/check_a_gpu_vs_sklearn.json ] || step python analysis/thesis_results/probes_a.py --check
step python analysis/thesis_results/probes_a.py
step python analysis/thesis_results/probes_b.py
date -u +%FT%TZ > $OUT/logs/gpu_probes.done
