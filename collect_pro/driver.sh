#!/usr/bin/env bash
# Runs run_all.sh until it finishes (exit 0) or fails (exit 1); while collection runs
# (exit 10) it re-checks every 5 minutes, which also restarts any dead worker.
# Launch: tmux new-session -d -s md-runall 'bash collect_pro/driver.sh'
cd /ephemeral/code/thesis-introspection
while true; do
  bash run_all.sh >> results/abc2/logs/driver.log 2>&1
  rc=$?
  echo "$(date -u +%FT%H:%M:%SZ) run_all exit $rc" >> results/abc2/logs/driver.log
  [ $rc -eq 10 ] || break
  sleep 300
done
echo "$rc" > results/abc2/logs/driver.exit
