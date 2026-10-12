#!/usr/bin/env bash
# Start any collection worker whose tmux session (md-w<k>) is not running and whose
# finished marker is absent. Safe to re-run: the worker skips complete rollouts.
cd /ephemeral/code/thesis-introspection
source env.sh >/dev/null 2>&1
N=$(python -c "import yaml;print(yaml.safe_load(open('configs/more_data.yaml'))['N_WORKERS'])")
OUT=/data/rollouts_v3
mkdir -p "$OUT" results/abc2/logs
[ -f "$OUT/collect_started_utc" ] || date -u +%Y-%m-%dT%H:%M:%S+00:00 > "$OUT/collect_started_utc"
for k in $(seq 0 $((N-1))); do
  if tmux has-session -t md-w$k 2>/dev/null; then echo "md-w$k running"; continue; fi
  if [ -f "$OUT/worker$k.finished" ]; then echo "worker $k finished"; continue; fi
  tmux new-session -d -s md-w$k "cd /ephemeral/code/thesis-introspection && source env.sh && source collect_pro/env_pro.sh && python collect_pro/collect.py --worker $k --n-workers $N >> results/abc2/logs/collect_w$k.log 2>&1"
  echo "started md-w$k"
  sleep 20
done
