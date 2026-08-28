#!/usr/bin/env bash
# run_all_layers.sh -- scheme B (probe_late_window.py) across every stored layer.
#
# Assumes extract_all_layers.py has already filled /data/tmp/probe_feats_layer*.npz.
# Without those caches each worker falls back to load_features() and re-reads the whole
# 42.5 GB corpus, and eight of them do it at once.
#
# OMP_NUM_THREADS=1 is not a throttle, it is a speedup: lbfgs on a 3844x4096 matrix is
# measurably faster single-threaded (0.31 s/fit vs 0.62 s at 8 threads -- BLAS thread
# overhead dominates at this size). One thread per worker, eight workers.
#
# Each layer writes its own results dir, so nothing races on a shared metrics.json.
#
# Usage:
#   source env.sh
#   ./run_all_layers.sh                 # layers 0-32, 8 at a time
#   ./run_all_layers.sh 0 32 4          # first last jobs
set -euo pipefail

FIRST="${1:-0}"
LAST="${2:-32}"
JOBS="${3:-8}"

OUT_ROOT="results/probe_all_layers"
LOG_DIR="/data/tmp/layer_sweep_logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "!! venv not active -- run 'source env.sh' first" >&2
    exit 1
fi

missing=()
for L in $(seq "$FIRST" "$LAST"); do
    [[ -f "/data/tmp/probe_feats_layer${L}.npz" ]] || missing+=("$L")
done
if (( ${#missing[@]} )); then
    echo "!! no feature cache for layers: ${missing[*]}" >&2
    echo "!! run 'python analysis/extract_all_layers.py' first, or these workers will each" >&2
    echo "!! re-read the full corpus concurrently." >&2
    exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[*] layers $FIRST..$LAST, $JOBS at a time, 1 BLAS thread each"
echo "[*] logs -> $LOG_DIR/layerNN.log ; results -> $OUT_ROOT/layerNN/"
START=$(date +%s)

seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} bash -c '
    L={}
    D=$(printf "%s/layer%02d" "'"$OUT_ROOT"'" "$L")
    LOG=$(printf "%s/layer%02d.log" "'"$LOG_DIR"'" "$L")
    if python analysis/probe_late_window.py --layer "$L" --results-dir "$D" > "$LOG" 2>&1; then
        printf "    layer %2d done: %s\n" "$L" "$(grep -m1 "^subset" "$LOG" || echo ok)"
    else
        printf "!!  layer %2d FAILED -- see %s\n" "$L" "$LOG"
    fi
'

echo "[*] sweep finished in $(( ($(date +%s) - START) / 60 )) min"
echo "[*] next: python analysis/plot_auroc_by_layer.py"
