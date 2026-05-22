#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run-dev.sh  --  Launch the *version2* (development) face-swap-streamer
#
# version2 is the copy used to improve / enhance the software, so it runs
# SEPARATELY from the production v1 instance:
#   * production v1  -> port 8080   (folder: face-swap-streamer)
#   * dev version2   -> port 8090   (folder: face-swap-streamer-version2)
#
# It uses the same proven WSL venv + GPU (RTX 5080 via /dev/dxg) as prod.
#
# Usage:
#   bash run-dev.sh            # start (or report if already up) + print links
#   bash run-dev.sh stop       # stop the dev server
#   bash run-dev.sh restart    # stop then start
#   FACESWAP_PORT=8091 bash run-dev.sh   # override the port
# ---------------------------------------------------------------------------
set -euo pipefail

PROJ="/mnt/c/AI_Team/Nehanth/face-swap-streamer-version2"
VENV="/home/svbtr/streamer_venv"
PS="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
PORT="${FACESWAP_PORT:-8090}"
LOG="$PROJ/out/dev-webapp.log"
PIDFILE="$PROJ/out/dev-webapp.pid"

port_in_use() { ss -tln 2>/dev/null | grep -q ":${PORT} "; }

stop_server() {
  if [[ -f "$PIDFILE" ]]; then
    local pid; pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping dev server (pid $pid) ..."
      pkill -TERM -P "$pid" 2>/dev/null || true   # workers
      kill -TERM "$pid" 2>/dev/null || true
      sleep 2
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
  # fallback: anything still bound to our port that lives under version2
  pkill -f "FACESWAP_ROOT=$PROJ" 2>/dev/null || true
}

print_links() {
  local lan ts
  lan="$("$PS" -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {\$_.InterfaceAlias -eq 'Wi-Fi'}).IPAddress" 2>/dev/null | tr -d '\r' | head -1)"
  ts="$("$PS" -NoProfile -Command "try { (tailscale ip -4) } catch { '' }" 2>/dev/null | tr -d '\r' | head -1)"
  echo ""
  echo "=========================================================="
  echo "  face-swap-streamer  VERSION2 (dev)  is UP on the GPU"
  echo "=========================================================="
  echo "  local:     http://localhost:${PORT}/"
  [[ -n "$lan" ]] && echo "  LAN:       http://${lan}:${PORT}/"
  [[ -n "$ts"  ]] && echo "  Tailscale: http://${ts}:${PORT}/"
  echo "  logs:      $LOG"
  echo "=========================================================="
}

case "${1:-start}" in
  stop)    stop_server; echo "dev server stopped."; exit 0 ;;
  restart) stop_server ;;
  start|"") : ;;
  *) echo "usage: bash run-dev.sh [start|stop|restart]"; exit 1 ;;
esac

if port_in_use; then
  echo "version2 dev server already running on port ${PORT}."
  print_links
  exit 0
fi

cd "$PROJ"
mkdir -p "$PROJ/out"

# --- GPU / runtime env (mirrors the working production setup) -------------
export FACESWAP_ROOT="$PROJ"
export FACESWAP_PORT="$PORT"
# GFPGAN restoration adds a 3rd ONNX session per worker on the single 16GB
# RTX 5080. At 6 workers (buffalo_l + inswapper + GFPGAN ×6) VRAM hit ~96% and
# CUDA spilled to system RAM over PCIe -> throughput collapsed to ~3 fps. With
# 3 workers every session stays resident in VRAM and runs at full speed.
# 6 workers only fits with GFPGAN OFF. Each worker = buffalo_l + inswapper
# (+ GFPGAN if enabled). 6×(those three) overruns the 16GB card (~15.7GB ->
# spill over PCIe -> ~3fps, the original regression). With GFPGAN disabled
# below, 6×(buffalo_l + inswapper) fits comfortably and runs fast.
export FACESWAP_WORKERS="${FACESWAP_WORKERS:-6}"
export FACESWAP_DET_SIZE="${FACESWAP_DET_SIZE:-480}"
# GFPGAN (512px restore) is the VRAM hog that makes 6 workers spill. Off here so
# 6 workers fit + run fast. The colour-match + unsharp paste (FACESWAP_ENHANCE,
# default on) still improves every face. To get GFPGAN back, set
# FACESWAP_ENHANCER=1 AND drop FACESWAP_WORKERS to <=4.
export FACESWAP_ENHANCER="${FACESWAP_ENHANCER:-0}"
# Only run GFPGAN on faces whose longest side is >= this many px (when enabled).
export FACESWAP_ENHANCER_MIN_FACE="${FACESWAP_ENHANCER_MIN_FACE:-90}"
# Matching: raise the cosine-sim threshold off the very permissive 0.15 default
# so only confident matches swap (stops swapping onto the wrong person). Too
# high would drop the real target on hard frames (= flicker), so 0.25 is a
# balance with the max-over-members matcher.
export FACESWAP_REF_THRESH="${FACESWAP_REF_THRESH:-0.25}"
# Slightly weaker LAB colour transfer -> the swapped face's colour shifts less
# frame-to-frame (reduces shimmer/flicker) while still blending into the scene.
export FACESWAP_COLOR_STRENGTH="${FACESWAP_COLOR_STRENGTH:-0.5}"
export FACESWAP_FFMPEG="/usr/bin/ffmpeg"
export FACESWAP_FACE_MODEL="buffalo_l"
export FACESWAP_VIDEO_ENCODER="h264_nvenc"
NV="$VENV/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

echo "starting version2 dev server on port ${PORT} (model warmup ~30s) ..."
nohup "$VENV/bin/python" webapp_mp.py > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# wait up to 120s for the port to come up
for _ in $(seq 1 120); do
  port_in_use && break
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "ERROR: server process died during startup. Last log lines:"
    tail -n 30 "$LOG"
    exit 1
  fi
  sleep 1
done

if port_in_use; then
  print_links
else
  echo "ERROR: server did not bind port ${PORT} within 120s. Last log lines:"
  tail -n 30 "$LOG"
  exit 1
fi
