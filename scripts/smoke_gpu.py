"""GPU smoke test (CLAUDE.md §3.5). Run: python scripts/smoke_gpu.py

Exits non-zero if CUDA is unavailable or a matmul fails. On RTX 5080 expect
capability (12, 0). See §19 if capability check fails.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed (install the 'gpu' extra on the RTX host)")
        return 2
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    x = torch.randn(1024, 1024, device="cuda")
    val = torch.matmul(x, x).sum().item()
    print("matmul OK", val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
