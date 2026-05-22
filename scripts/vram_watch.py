"""Stream VRAM/util to stdout as CSV (CLAUDE.md §19.4).

Usage: python scripts/vram_watch.py  (Ctrl+C to stop)
Falls back gracefully if pynvml is unavailable.
"""

from __future__ import annotations

import csv
import sys
import time


def main() -> int:
    try:
        import pynvml
    except ImportError:
        print("pynvml not installed (pip install pynvml)", file=sys.stderr)
        return 2
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    w = csv.writer(sys.stdout)
    w.writerow(["timestamp", "used_mb", "util_pct", "temp_c"])
    try:
        while True:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            w.writerow([time.time(), mem.used // (1024**2), util.gpu, temp])
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
