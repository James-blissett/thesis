#!/usr/bin/env bash
# Check 4 (results_followup_handoff.md §5): smoothing sensitivity for B and C, in tmux.
# Resumable: every stage skips finished work files. Markers in results/thesis_results_final/logs/:
# smoothing.done / smoothing.error.
set -u
cd /ephemeral/code/thesis-introspection
source env.sh
export THESIS_POOL=D2 OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
OUT=results/thesis_results_final
mkdir -p $OUT/logs; rm -f $OUT/logs/smoothing.done $OUT/logs/smoothing.error
for stage in signs c b eval; do
  echo "== $(date -u +%T) stage $stage"
  python analysis/thesis_results/final/smoothing.py $stage || { echo "stage $stage failed" > $OUT/logs/smoothing.error; exit 1; }
done
date -u +%FT%TZ > $OUT/logs/smoothing.done
