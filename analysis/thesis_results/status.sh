#!/usr/bin/env bash
# At most five lines: stage, running tmux jobs, latest log line, disk, GPU.
cd /ephemeral/code/thesis-introspection
R=results/thesis_results
echo "state: $(sed -n 1p $R/STATE.md 2>/dev/null | cut -c1-150)"
echo "tmux: $(tmux ls 2>/dev/null | grep -o '^tr-[^:]*' | tr '\n' ' ')"
L=$(ls -t $R/logs/*.log results/thesis_results_full/logs/*.log 2>/dev/null | head -1)
[ -n "$L" ] && echo "last log $(basename $L): $(tail -1 $L | cut -c1-140)"
echo "disk /data free: $(df -BG /data | awk 'NR==2{print $4}') | markers: $(ls $R/logs/*.done $R/logs/*.error 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ' | cut -c1-120)"
echo "gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
