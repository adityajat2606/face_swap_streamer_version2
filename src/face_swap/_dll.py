"""Windows secure-DLL-search registration for CUDA/cuDNN/TensorRT.

Py 3.8+ on Windows ignores PATH for native imports; you must call
``os.add_dll_directory`` and KEEP the cookie alive (CLAUDE.md issue #1, #9).
Call :func:`register_cuda_dlls` once before importing onnxruntime/insightface.
"""

from __future__ import annotations

import os
import sys

_dll_cookies: list = []  # MUST stay referenced — GC'd cookies = lost paths.


def register_cuda_dlls() -> None:
    if sys.platform != "win32" or _dll_cookies:
        return
    sp = os.path.join(sys.prefix, "Lib", "site-packages")
    bin_dirs = [
        *(os.path.join(sp, "nvidia", sub, "bin")
          for sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                      "cuda_nvrtc", "nvjitlink")),
        os.path.join(sp, "tensorrt_libs"),  # different layout
    ]
    for d in bin_dirs:
        if os.path.isdir(d):
            try:
                _dll_cookies.append(os.add_dll_directory(d))
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
