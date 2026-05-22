"""GPU/VRAM telemetry poller (CLAUDE.md §17.3, §19.4).

Background thread polling pynvml (preferred) or ``nvidia-smi`` (fallback) at a
fixed rate, writing ``gpu_telemetry.csv`` and feeding gauges into the
Observatory. Degrades to a no-op if neither backend is available, so it is safe
to start on a CPU-only host.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .logging_setup import get_logger
from .observability import MetricsRegistry

_log = get_logger("face_swap.gpu")

_CSV_HEADER = ["timestamp", "gpu_util_pct", "vram_used_mb", "vram_total_mb",
               "temperature_c", "power_w"]


def _read_pynvml(handle) -> tuple[float, float, float, float, float] | None:
    try:
        import pynvml

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:  # noqa: BLE001 - power query optional
            power = 0.0
        return (float(util.gpu), mem.used / (1024**2), mem.total / (1024**2),
                float(temp), power)
    except Exception:  # noqa: BLE001 - pynvml missing / no device
        return None


def _read_nvidia_smi() -> tuple[float, float, float, float, float] | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
        first = out.strip().splitlines()[0]
        u, used, total, temp, power = (p.strip() for p in first.split(","))
        return (float(u), float(used), float(total), float(temp), float(power))
    except Exception:  # noqa: BLE001
        return None


class GpuTelemetry:
    def __init__(self, csv_path: Path, metrics: MetricsRegistry | None = None, hz: float = 1.0):
        self.csv_path = Path(csv_path)
        self.metrics = metrics
        self.interval = 1.0 / max(hz, 1e-3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._backend = "none"

    def _init_backend(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._backend = "pynvml"
            return
        except Exception:  # noqa: BLE001
            self._handle = None
        if shutil.which("nvidia-smi"):
            self._backend = "nvidia-smi"
        else:
            self._backend = "none"

    def _poll_once(self) -> tuple[float, float, float, float, float] | None:
        if self._backend == "pynvml":
            return _read_pynvml(self._handle)
        if self._backend == "nvidia-smi":
            return _read_nvidia_smi()
        return None

    def _run(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_CSV_HEADER)
            while not self._stop.is_set():
                row = self._poll_once()
                if row is not None:
                    util, used, total, temp, power = row
                    writer.writerow([time.time(), util, used, total, temp, power])
                    fh.flush()
                    if self.metrics is not None:
                        self.metrics.set_gauge("gpu_util_pct", util)
                        self.metrics.set_gauge("vram_used_mb", used)
                        self.metrics.observe("vram_used_mb_hist", used)
                        self.metrics.set_gauge("gpu_temp_c", temp)
                self._stop.wait(self.interval)

    def start(self) -> None:
        self._init_backend()
        if self._backend == "none":
            _log.info("gpu_telemetry_disabled", reason="no pynvml / nvidia-smi")
            return
        _log.info("gpu_telemetry_started", backend=self._backend)
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> GpuTelemetry:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
