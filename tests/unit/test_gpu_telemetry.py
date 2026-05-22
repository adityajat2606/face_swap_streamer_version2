from __future__ import annotations

from face_swap.gpu_telemetry import GpuTelemetry
from face_swap.observability import MetricsRegistry


def test_no_backend_is_noop(tmp_path, monkeypatch):
    """On a host without pynvml/nvidia-smi, start() is a safe no-op."""
    import face_swap.gpu_telemetry as gt

    monkeypatch.setattr(gt.shutil, "which", lambda _name: None)
    # force pynvml import to fail
    monkeypatch.setitem(__import__("sys").modules, "pynvml", None)
    g = GpuTelemetry(tmp_path / "gpu.csv", MetricsRegistry(), hz=10)
    g.start()
    assert g._backend == "none"
    g.stop()  # must not raise


def test_poll_once_none_when_disabled(tmp_path):
    g = GpuTelemetry(tmp_path / "gpu.csv")
    g._backend = "none"
    assert g._poll_once() is None


def test_context_manager(tmp_path, monkeypatch):
    import face_swap.gpu_telemetry as gt

    monkeypatch.setattr(gt.shutil, "which", lambda _name: None)
    monkeypatch.setitem(__import__("sys").modules, "pynvml", None)
    with GpuTelemetry(tmp_path / "g.csv") as g:
        assert g._backend == "none"
