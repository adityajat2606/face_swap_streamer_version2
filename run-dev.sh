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
STATS_PIDFILE="$PROJ/out/gpu-stats.pid"
STATS_LOG="$PROJ/out/gpu-swap-stats.log"
STATS_DAEMON="$PROJ/gpu-swap-stats-daemon.sh"

port_in_use() { ss -tln 2>/dev/null | grep -q ":${PORT} "; }

ensure_stats_daemon() {
  [[ -x "$STATS_DAEMON" ]] || return 0
  if [[ -f "$STATS_PIDFILE" ]] && kill -0 "$(cat "$STATS_PIDFILE")" 2>/dev/null; then
    return 0
  fi
  nohup bash "$STATS_DAEMON" >/dev/null 2>&1 &
  echo $! > "$STATS_PIDFILE"
  echo "  gpu stats: tail -f $STATS_LOG   (daemon pid $(cat "$STATS_PIDFILE"))"
}

stop_stats_daemon() {
  if [[ -f "$STATS_PIDFILE" ]]; then
    local pid; pid="$(cat "$STATS_PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$STATS_PIDFILE"
  fi
}

stop_server() {
  stop_stats_daemon
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
  ensure_stats_daemon
  exit 0
fi

cd "$PROJ"
mkdir -p "$PROJ/out"

# --- GPU / runtime env (mirrors the working production setup) -------------
export FACESWAP_ROOT="$PROJ"
export FACESWAP_PORT="$PORT"
# Quality-optimized defaults (Tier 1): 4 workers + GFPGAN ON + 800px detection.
# Each worker holds 3 ONNX sessions in VRAM: buffalo_l detector + inswapper +
# GFPGAN restorer. At 4 workers that fits in 16 GB (~13-14 GiB peak). Pushing to
# 6 workers WITH GFPGAN spills CUDA to system RAM over PCIe -> ~3 fps collapse.
# If you need more throughput and can sacrifice the GFPGAN polish, set
# FACESWAP_ENHANCER=0 and bump FACESWAP_WORKERS up to 6.
export FACESWAP_WORKERS="${FACESWAP_WORKERS:-4}"
# 800px detection (insightface code default) catches smaller/farther faces in
# crowd scenes. 480 was the old throughput-tilt setting and missed people.
export FACESWAP_DET_SIZE="${FACESWAP_DET_SIZE:-800}"
# GFPGAN (512px restore) is the single biggest quality lever — sharpens skin,
# eyes, hair after the inswapper paste. Required for "looks real" output.
# Pair with FACESWAP_WORKERS<=4 (see comment above) or it will OOM-spill.
export FACESWAP_ENHANCER="${FACESWAP_ENHANCER:-1}"
# Tier 3 (2026-05-24): CodeFormer chained AFTER GFPGAN. GFPGAN smooths skin
# (removes inswapper's 128px paste artefacts) but over-smooths fine detail;
# CodeFormer puts back eye/lip/hair detail on top. Weight 0.7 is the
# "looks-like-the-source" sweet spot — push to 0.85 for max restoration at the
# cost of some identity drift; drop to 0.5 if it looks "too cleaned-up".
# VRAM: adds ~0.6 GB/worker; 4 workers × all four ONNX sessions ≈ 14 GiB peak
# on the 16 GB 5080. If you see NVENC rc=171 errors, drop FACESWAP_WORKERS=3.
#
# 2026-05-25: DISABLED. Post-mortem of the 6682-frame crowd job showed VRAM
# peaked at 15,839 / 16,384 MiB (96.7%) — CUDA arena spilled to host RAM over
# PCIe, GPU sat in P8 idle 77% of samples, throughput collapsed to 0.10 fps.
# Dropping CodeFormer (keeping GFPGAN) brings the peak to ~13.4 GiB, comfortably
# off the ceiling. Re-enable with FACESWAP_CODEFORMER=1 only after pairing it
# with FACESWAP_WORKERS=3.
export FACESWAP_CODEFORMER="${FACESWAP_CODEFORMER:-0}"
export FACESWAP_CODEFORMER_WEIGHT="${FACESWAP_CODEFORMER_WEIGHT:-0.7}"
export FACESWAP_CODEFORMER_BLEND="${FACESWAP_CODEFORMER_BLEND:-0.8}"
# Only run GFPGAN on faces whose longest side is >= this many px (when enabled).
# 180 = roughly "≥1/6 of a 1080p frame's height" — hero/foreground faces. Below
# that the restorer cost outweighs the perceptual gain, especially under
# gender-mode crowd swaps where DET_SIZE=800 picks up many small background
# faces (was 90 → caused ~20 enhanced faces/frame and 0.1 fps on crowd jobs).
export FACESWAP_ENHANCER_MIN_FACE="${FACESWAP_ENHANCER_MIN_FACE:-180}"
# CodeFormer is the costliest stage (second 512px restorer). Raise its floor
# above GFPGAN's so the chained stack only runs on hero faces; the swap +
# GFPGAN is plenty for mid-sized faces.
export FACESWAP_CODEFORMER_MIN_FACE="${FACESWAP_CODEFORMER_MIN_FACE:-220}"
# Matching: raise the cosine-sim threshold off the very permissive 0.15 default
# so only confident matches swap (stops swapping onto the wrong person). Too
# high would drop the real target on hard frames (= flicker), so 0.25 is a
# balance with the max-over-members matcher.
# Only used when FACESWAP_MATCH_MODE=identity. Ignored under gender-mode.
export FACESWAP_REF_THRESH="${FACESWAP_REF_THRESH:-0.25}"
# Tier 2 crowd-mode (chosen 2026-05-24): swap EVERY detected face using a
# same-gender avatar (cycling when more faces of that gender than avatars).
# No similarity threshold, so faces beyond the uploaded avatar count are still
# fully swapped. Set FACESWAP_MATCH_MODE=identity to fall back to the cluster +
# cosine-sim path that only swaps recognised identities.
export FACESWAP_MATCH_MODE="${FACESWAP_MATCH_MODE:-gender}"
# Detection NMS: suppress duplicate boxes for the same physical face (their
# overlapping soft paste masks were the source of multi-face paste-bleed).
# 0.5 keeps genuinely adjacent people, drops only true duplicates.
export FACESWAP_NMS_IOU="${FACESWAP_NMS_IOU:-0.5}"
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
  ensure_stats_daemon
else
  echo "ERROR: server did not bind port ${PORT} within 120s. Last log lines:"
  tail -n 30 "$LOG"
  exit 1
fi
