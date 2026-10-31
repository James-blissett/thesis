#!/bin/bash
# One train3.py worker for the given --inputs; log under results/abc3/logs/.
cd "$(dirname "$0")/../.." && source env.sh
name=$1; shift
ABC3_GPU_FRAC=${ABC3_GPU_FRAC:-0.45} python analysis/abc3/train3.py "$@" >> results/abc3/logs/train_${name}.log 2>&1
echo "exit $? $(date -u +%FT%TZ)" >> results/abc3/logs/train_${name}.log
