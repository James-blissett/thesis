#!/bin/bash
# Deep-MLP run (deep_mlp_handoff.md v1), resumable: bash analysis/abc3/run_all.sh
# Fits whose work file exists with the current spec are skipped (train3.py); the
# evaluation, bootstrap and report are cheap (< 1 min) and are always redone.
set -e
cd "$(dirname "$0")/../.." && source env.sh
mkdir -p results/abc3/logs
[ -f results/abc3/sanity_check.json ] || python analysis/abc3/sanity3.py --plain
[ -f results/abc3/precision.json ] || python analysis/abc3/sanity3.py --bf16
python analysis/abc3/train3.py --inputs C >> results/abc3/logs/train_C.log 2>&1
python analysis/abc3/train3.py --inputs A >> results/abc3/logs/train_A.log 2>&1
python analysis/abc3/eval3.py > results/abc3/logs/eval3.log 2>&1
python analysis/abc3/paired3.py > results/abc3/logs/paired3.log 2>&1
python analysis/abc3/report3.py > results/abc3/logs/report3.log 2>&1
echo "run_all done $(date -u +%FT%TZ)"
