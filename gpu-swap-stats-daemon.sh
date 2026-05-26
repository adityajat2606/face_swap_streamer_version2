#!/bin/bash
# gpu-swap-stats-daemon.sh
# Runs alongside version2 (started/stopped by run-dev.sh). Waits for the next
# face-swap, captures GPU stats, appends a summary to out/gpu-swap-stats.log,
# and re-arms for the next swap. Per-swap raw CSVs go under out/gpu-csv/.

set -u
PROJ="/mnt/c/AI_Team/Nehanth/face-swap-streamer-version2"
WATCHER="$PROJ/gpu-swap-stats.sh"
LOG="$PROJ/out/gpu-swap-stats.log"
CSV_DIR="$PROJ/out/gpu-csv"
mkdir -p "$CSV_DIR"

# Block ~indefinitely between swaps (24 h) so the watcher doesn't time out
# during quiet periods. Tunable via env if needed.
export WAIT_START_MAX=${WAIT_START_MAX:-86400}

trap 'echo "[gpu-stats-daemon] stopping @ $(date -Iseconds)" >> "$LOG"; exit 0' INT TERM

echo "[gpu-stats-daemon] started pid=$$ @ $(date -Iseconds)" >> "$LOG"

while true; do
  TS=$(date +%Y%m%d-%H%M%S)
  CSV="$CSV_DIR/swap-$TS.csv"
  {
    echo
    echo "============================================================"
    echo "  swap @ $(date -Iseconds)   csv: $CSV"
    echo "============================================================"
    CSV="$CSV" bash "$WATCHER" || true
  } >> "$LOG" 2>&1
  sleep 1
done
