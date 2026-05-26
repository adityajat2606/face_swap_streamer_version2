#!/bin/bash
# Watch the RTX 5080 while a face-swap stream runs, then print a full stats
# summary when the swap finishes. Auto-detects start (GPU util >25% sustained
# 3s) and end (GPU util <5% for 15s). Useful for benchmarking version2.
#
# Usage:
#   bash gpu-swap-stats.sh             # run, wait for stream, print summary
#   CSV=/tmp/mine.csv bash gpu-swap-stats.sh   # custom CSV path
#
# Exits 0 on clean run, 1 on timeout waiting for a stream to start.

set -u
NVSMI=${NVSMI:-/usr/lib/wsl/lib/nvidia-smi}
CSV=${CSV:-/tmp/gpu-swap.csv}
START_THRESHOLD=${START_THRESHOLD:-25}   # %util to consider "swap started"
END_THRESHOLD=${END_THRESHOLD:-5}        # %util considered idle
START_CONSEC=${START_CONSEC:-3}          # seconds above START_THRESHOLD
END_CONSEC=${END_CONSEC:-15}             # seconds below END_THRESHOLD
WAIT_START_MAX=${WAIT_START_MAX:-300}    # give up if no stream in this many s
SWAP_MAX=${SWAP_MAX:-1800}               # hard cap on swap duration sampled

command -v "$NVSMI" >/dev/null || { echo "nvidia-smi not at $NVSMI"; exit 2; }

: > "$CSV"
"$NVSMI" \
  --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,clocks.mem,fan.speed \
  --format=csv -lms 1000 -f "$CSV" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; exit' INT TERM

util() { "$NVSMI" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' '; }

echo "[gpu-swap-stats] sampling to $CSV — waiting up to ${WAIT_START_MAX}s for stream to start..."
HIT=0; WAITED=0
until [ "$HIT" -ge "$START_CONSEC" ]; do
  u=$(util); [ "$u" -gt "$START_THRESHOLD" ] && HIT=$((HIT+1)) || HIT=0
  WAITED=$((WAITED+1))
  if [ "$WAITED" -gt "$WAIT_START_MAX" ]; then
    echo "[gpu-swap-stats] TIMEOUT — no swap detected in ${WAIT_START_MAX}s"
    kill $SAMPLER 2>/dev/null; exit 1
  fi
  sleep 1
done
START=$(date +%s)
echo "[gpu-swap-stats] SWAP_START $(date -d @$START +%T)"

echo "[gpu-swap-stats] waiting for idle (${END_CONSEC}s under ${END_THRESHOLD}% util)..."
IDLE=0; RUN=0
until [ "$IDLE" -ge "$END_CONSEC" ]; do
  u=$(util); [ "$u" -lt "$END_THRESHOLD" ] && IDLE=$((IDLE+1)) || IDLE=0
  RUN=$((RUN+1))
  if [ "$RUN" -gt "$SWAP_MAX" ]; then echo "[gpu-swap-stats] TIMEOUT — swap >${SWAP_MAX}s"; break; fi
  sleep 1
done
END=$(date +%s)
DUR=$((END-START-END_CONSEC))
[ "$DUR" -lt 0 ] && DUR=0
echo "[gpu-swap-stats] SWAP_END   $(date -d @$END +%T)   active duration ~${DUR}s"

sleep 1
kill $SAMPLER 2>/dev/null
wait 2>/dev/null
SAMPLES=$(wc -l < "$CSV")
echo "[gpu-swap-stats] $SAMPLES rows (incl header)"
echo

python3 - "$CSV" <<'PY'
import csv, statistics as s, sys
rows = list(csv.reader(open(sys.argv[1])))
if len(rows) < 3:
    print("not enough samples"); sys.exit(0)
hdr = [h.strip() for h in rows[0]]
data = [dict(zip(hdr, [c.strip() for c in r])) for r in rows[1:]]
def num(x): return float(x.split()[0])
fields = [
    ("GPU util",       'utilization.gpu [%]',           '%'),
    ("Mem-bus util",   'utilization.memory [%]',        '%'),
    ("VRAM used",      'memory.used [MiB]',             'MiB'),
    ("VRAM free",      'memory.free [MiB]',             'MiB'),
    ("Power",          'power.draw [W]',                'W'),
    ("Temperature",    'temperature.gpu',               'C'),
    ("SM clock",       'clocks.current.sm [MHz]',       'MHz'),
    ("Mem clock",      'clocks.current.memory [MHz]',   'MHz'),
    ("Fan",            'fan.speed [%]',                 '%'),
]
util = [num(r['utilization.gpu [%]']) for r in data]
vram = [num(r['memory.used [MiB]'])   for r in data]
print(f"  window: {data[0]['timestamp']}  ->  {data[-1]['timestamp']}  ({len(data)} samples)")
print(f"  pstates: " + ", ".join(f"{p}:{[r['pstate'] for r in data].count(p)}" for p in sorted({r['pstate'] for r in data})))
print(f"  {'metric':22s}  {'min':>9s}  {'avg':>9s}  {'max':>9s}")
print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}")
for name, key, unit in fields:
    vals = [num(r[key]) for r in data]
    print(f"  {name:22s}  {min(vals):>9.1f}  {s.mean(vals):>9.1f}  {max(vals):>9.1f}  {unit}")
active = [u for u in util if u > 20]
if active:
    print(f"\n  active samples (util>20%): {len(active)}/{len(util)}   avg util when active: {s.mean(active):.1f}%")
print(f"  VRAM start/end: {vram[0]:.0f} -> {vram[-1]:.0f} MiB   peak: {max(vram):.0f} MiB   delta: {vram[-1]-vram[0]:+.0f}")
PY
