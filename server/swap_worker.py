"""
Multiprocess swap-worker entry point (Phase 5 of the GPU-saturation plan).

Each worker process loads its own InsightFace `FaceAnalysis` (buffalo_l) +
`INSwapper` (inswapper_128_fp16.onnx) ONE TIME at startup, then runs a
single-threaded loop pulling frame slots out of shared memory, doing
detect + match + fused swap+paste, writing the result back into the
same slot, and acking via the shared result queue.

Why processes, not threads (see Phase 3 post-mortem `3edf180`):
- ORT releases the GIL during inference, but the Python glue around each
  ORT call (numpy slicing, paste-back, attribute lookups on Face objects,
  queue puts/gets) is GIL-serialised. Splitting into more in-process
  threads regressed throughput because the GIL contention got worse.
- Separate processes give each worker its own GIL and its own ORT/CUDA
  context. We pay the cost of model load × N workers at startup, then
  steady-state is fully parallel up to the GPU's compute ceiling.

Loaded in isolation (no webapp dependency) — webapp.py imports
`worker_main` and spawns processes; this module never imports webapp.

Sub-task 5.1: ships the entry point + DLL discovery + model loading.
Sub-task 5.2 will add SwapRequest / SwapResponse / FramePool.
"""
from __future__ import annotations
import os
import sys
import time
import pickle
import traceback
from dataclasses import dataclass
from typing import Optional


# ---- Windows DLL discovery for onnxruntime + CUDA ---------------------------
# Each spawned worker process gets a fresh Python interpreter; the
# os.add_dll_directory cookies from the parent (webapp.py) do NOT propagate.
# This block MUST run BEFORE `import onnxruntime` / `import insightface`.
# See CLAUDE.md issue #1.

_dll_cookies = []   # keep cookies alive — GC'd cookies = lost search paths


def _register_cuda_dll_dirs() -> None:
    """Add nvidia-*-cu12 and tensorrt_libs DLL dirs to Windows' secure DLL
    search path. Safe to call multiple times — duplicate dirs are silently
    ignored by Windows. No-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    sp = os.path.join(sys.prefix, "Lib", "site-packages")
    bin_dirs = [
        # nvidia-cudnn-cu12, nvidia-cublas-cu12, etc.
        *(os.path.join(sp, "nvidia", sub, "bin")
          for sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                      "cuda_nvrtc", "nvjitlink")),
        # tensorrt-cu12 puts its DLLs under tensorrt_libs/ — different layout
        os.path.join(sp, "tensorrt_libs"),
    ]
    for b in bin_dirs:
        if os.path.isdir(b):
            try:
                _dll_cookies.append(os.add_dll_directory(b))
            except OSError:
                pass
            os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]


_register_cuda_dll_dirs()


# ---- IPC dataclasses (Sub-task 5.2 will flesh these out) --------------------

@dataclass
class SwapRequest:
    """Master -> worker: process the frame in `slot_id` (frame index `frame_idx`).

    If `end` is True, all other fields are ignored — the worker should ack
    with a SwapResponse(frame_idx=-1) and exit cleanly.
    """
    frame_idx: int = -1
    slot_id: int = -1
    end: bool = False


@dataclass
class SwapResponse:
    """Worker -> master: frame at `frame_idx` is done in `slot_id`.

    `n_swapped` is how many faces this worker swapped into the frame
    (used by master to accumulate `swap_count`). `worker_id` is the
    sending worker so the master can debug per-worker stalls.

    If `frame_idx == -1` this is the end-of-stream ack.
    `error` is non-empty if the worker hit an exception while processing
    this frame — master should treat that as a fatal job error.
    """
    frame_idx: int = -1
    slot_id: int = -1
    n_swapped: int = 0
    worker_id: int = -1
    elapsed_ms: float = 0.0
    error: str = ""


class FramePool:
    """Pool of fixed-size shared-memory slots backing one BGR frame each.

    Owned by the master process; workers `attach` to slots by name via
    `multiprocessing.shared_memory.SharedMemory(name=...)`. Each slot is
    a contiguous uint8 buffer sized to fit `H * W * 3` bytes; the same
    buffer holds the input frame on dispatch and the swapped output on
    return (workers overwrite in place; master copies out to ffmpeg
    before recycling the slot).

    Sizing: at 1080p one slot is ~6.2 MB. With N_workers=4 and 4 slots
    per worker (16 slots) that's ~100 MB of pinned shared RAM — fine.

    Lifecycle from the master's POV:
        pool = FramePool(n_slots=N_workers * 4, shape=(H, W, 3))
        slot = pool.acquire()                 # blocks if pool empty
        arr  = pool.view(slot)                # numpy view, no copy
        arr[...] = decoded_frame              # fill in place
        in_q.put(SwapRequest(frame_idx=k, slot_id=slot))
        # ... worker processes, writes result back into the same slot ...
        resp = out_q.get()
        out_arr = pool.view(resp.slot_id)
        ffmpeg.stdin.write(out_arr.tobytes()) # or out_arr.copy() first
        pool.release(resp.slot_id)            # back into the free pool

    Workers never call acquire/release — they just `view(slot_id)` the
    slot the master picked, mutate it in place, and ack.

    Cleanup: `close()` releases the SharedMemory handles; on Windows
    Python's reference counter unlinks them when the last handle is
    dropped. We additionally call `unlink()` defensively on master shutdown
    because dangling shm names linger across process crashes on some
    Windows builds.
    """

    def __init__(self, n_slots: int, shape: tuple, dtype=None):
        import numpy as np
        from multiprocessing import shared_memory
        if dtype is None:
            dtype = np.uint8
        self.n_slots = int(n_slots)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._np = np
        self._shm_mod = shared_memory

        nbytes = int(np.prod(self.shape) * self.dtype.itemsize)
        self.nbytes = nbytes
        self._shms = []                # list[SharedMemory]
        self._views = []               # list[np.ndarray] — one per slot
        for _ in range(self.n_slots):
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._shms.append(shm)
            self._views.append(
                np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            )

        # Free-slot pool. Uses a thread-safe Queue so the master's
        # demux thread can `acquire()` (block on empty) while a
        # background result-drainer calls `release()` from another
        # thread. We keep it bounded to n_slots so accidental
        # double-release is loud (`Full`) instead of silent.
        import queue as _queue
        self._free: "_queue.Queue[int]" = _queue.Queue(maxsize=self.n_slots)
        for i in range(self.n_slots):
            self._free.put(i)

    @property
    def names(self) -> list:
        """Pass these to each worker so it can attach to the same slots."""
        return [s.name for s in self._shms]

    def acquire(self, timeout: Optional[float] = None) -> int:
        """Block until a slot is free; return its id. `timeout=None` waits
        forever; pass a small timeout to detect master stalls."""
        return self._free.get(timeout=timeout)

    def release(self, slot_id: int) -> None:
        """Return a slot to the free pool. Raises queue.Full on double-release."""
        self._free.put_nowait(int(slot_id))

    def view(self, slot_id: int):
        """Zero-copy numpy view of the slot. Master and worker both call this."""
        return self._views[int(slot_id)]

    def free_count(self) -> int:
        return self._free.qsize()

    def close(self) -> None:
        """Drop all SharedMemory handles + unlink on POSIX/Windows. Idempotent."""
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except (FileNotFoundError, OSError):
                # Already unlinked, or platform doesn't require it.
                pass
        self._shms.clear()
        self._views.clear()


# ---- Swap-quality enhancement (Phase 6) ------------------------------------
# Goal: make swapped faces look sharper and less "pasted on / AI" WITHOUT
# touching detection/matching. Two additive, env-gated stages applied per face:
#
#   1. Custom paste-back (replaces inswapper's default soft blend):
#        a. LAB colour transfer  — push the swapped 128px crop's mean/std toward
#           the original target crop so it adopts the scene's lighting + skin
#           tone. The colour mismatch is the #1 giveaway that a face was swapped.
#        b. Unsharp mask         — recover micro-detail lost when the 128px
#           inswapper output is scaled up to a larger face.
#      Both run on the cheap 128x128 crop, so cost is negligible.
#
#   2. GFPGAN ONNX face restoration (optional, needs the model file):
#        re-aligns the swapped face to 512, runs GFPGAN, pastes the restored
#        face back with a feathered mask. This is the real fix for blur on
#        close-up faces. `blend` keeps some of the swapped face so identity /
#        texture don't drift toward GFPGAN's smoothing.
#
# Everything is reversible: FACESWAP_ENHANCE=0 restores inswapper's stock paste,
# FACESWAP_ENHANCER=0 disables GFPGAN. Prod (v1) is unaffected.


def _match_faces_to_sources(sims, thresh):
    """One-to-one face<->source assignment per frame. Returns [(face_i, src_i)].

    Replaces plain per-face argmax, which left the female source unused (and the
    actress's face skipped) whenever the male match was stronger. When #faces ==
    #sources (the duet case) every source is forced onto a distinct face with no
    threshold, so the female face is always swapped; otherwise each source claims
    its best distinct face above threshold, then extra faces match their best
    source above threshold (crowds / repeated identities).
    """
    import numpy as _np

    T, S = sims.shape
    out = []
    face_taken = [False] * T
    src_used = [False] * S
    force = (T == S)
    order = _np.dstack(
        _np.unravel_index(_np.argsort(sims, axis=None)[::-1], sims.shape)
    )[0]
    for pair in order:
        ti, si = int(pair[0]), int(pair[1])
        if face_taken[ti] or src_used[si]:
            continue
        if (not force) and float(sims[ti, si]) < thresh:
            break
        out.append((ti, si))
        face_taken[ti] = True
        src_used[si] = True
        if all(src_used) or all(face_taken):
            break
    for ti in range(T):
        if face_taken[ti]:
            continue
        si = int(_np.argmax(sims[ti]))
        if float(sims[ti, si]) >= thresh:
            out.append((ti, si))
    return out


def _color_transfer(src, ref, strength, np, cv2):
    """Match `src`'s colour statistics to `ref` in LAB (Reinhard transfer).

    Both BGR uint8, same HxW. `strength` in [0,1] blends the matched result back
    toward the original swap so we correct lighting without washing out the
    swapped identity. Returns BGR uint8.
    """
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for i in range(3):
        smean, sstd = s[..., i].mean(), s[..., i].std() + 1e-6
        rmean, rstd = r[..., i].mean(), r[..., i].std() + 1e-6
        out[..., i] = (s[..., i] - smean) * (rstd / sstd) + rmean
    matched = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if strength >= 0.999:
        return matched
    return cv2.addWeighted(matched, strength, src, 1.0 - strength, 0.0)


def _unsharp(img, amount, cv2, radius=1.0):
    """Mild unsharp mask. `amount`<=0 is a no-op."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)


def _feathered_face_mask(M, src_h, src_w, dst_h, dst_w, np, cv2):
    """Build a soft paste mask: a full-white aligned crop warped back to the
    frame, eroded inward and Gaussian-feathered (same recipe inswapper uses).
    Returns a (dst_h, dst_w, 1) float32 mask in [0,1], or None if the warped
    region is empty.
    """
    IM = cv2.invertAffineTransform(M)
    white = np.full((src_h, src_w), 255.0, dtype=np.float32)
    mask = cv2.warpAffine(white, IM, (dst_w, dst_h), borderValue=0.0)
    mask[mask > 20] = 255
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return None, IM
    msize = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
    k = max(msize // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(msize // 20, 5)
    blur = tuple(2 * i + 1 for i in (k, k))
    mask = cv2.GaussianBlur(mask, blur, 0) / 255.0
    return mask[:, :, None], IM


def _paste_enhanced(frame, aimg, bgr_fake, M, cfg, np, cv2):
    """Colour-match + sharpen the swapped 128px crop, then paste it back into
    `frame` with a feathered mask. Mirrors inswapper's paste geometry exactly
    (same M / mask recipe) so only the *pixels* change, not the placement.
    Returns the merged frame (uint8); falls back to `frame` if the mask is empty.
    """
    if cfg["color"]:
        bgr_fake = _color_transfer(bgr_fake, aimg, cfg["color_strength"], np, cv2)
    bgr_fake = _unsharp(bgr_fake, cfg["sharpen"], cv2)
    h, w = frame.shape[:2]
    mask, IM = _feathered_face_mask(M, aimg.shape[0], aimg.shape[1], h, w, np, cv2)
    if mask is None:
        return frame
    warped = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0.0)
    merged = mask * warped + (1.0 - mask) * frame.astype(np.float32)
    return merged.astype(np.uint8)


class FaceEnhancer:
    """GFPGAN ONNX face-restoration wrapper (CUDA via the existing onnxruntime).

    `enhance(frame, kps, blend)` aligns the face to the model's input size,
    restores it, and pastes the result back into a copy of `frame` with a
    feathered mask. `blend` in [0,1] mixes restored vs. current pixels inside
    the mask (1.0 = full GFPGAN). Operates on whatever face the kps point at —
    we call it right after the swap so it restores the *swapped* face.
    """

    def __init__(self, model_path, providers, np, cv2):
        import onnxruntime
        # Memory-frugal CUDA options. N workers each hold a GFPGAN session on a
        # single GPU alongside inswapper + buffalo_l + the NVENC encoder. ORT's
        # default EXHAUSTIVE conv-algo search + max workspace over-allocates
        # VRAM, which starves NVENC ("Failed unlocking input buffer", ffmpeg
        # rc=171) and collapses throughput to a crawl. Skip the exhaustive
        # search, don't grab max workspace, and only grow the arena as needed.
        # Optionally hard-cap per-session VRAM via FACESWAP_ENHANCER_GPU_MEM_MB.
        cuda_opts = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        }
        gpu_mb = os.getenv("FACESWAP_ENHANCER_GPU_MEM_MB", "").strip()
        if gpu_mb:
            cuda_opts["gpu_mem_limit"] = str(int(gpu_mb) * 1024 * 1024)
        providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shp = self.session.get_inputs()[0].shape
        self.size = int(shp[2]) if isinstance(shp[2], int) else 512
        self.np = np
        self.cv2 = cv2

    def providers(self):
        try:
            return self.session.get_providers()
        except Exception:
            return None

    def enhance(self, frame, kps, blend):
        from insightface.utils import face_align
        np, cv2 = self.np, self.cv2
        aimg, M = face_align.norm_crop2(frame, kps, self.size)
        # BGR uint8 -> RGB, scale to [-1, 1], NCHW float32.
        blob = aimg[:, :, ::-1].astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        out = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        # [-1,1] -> [0,1], CHW RGB -> HWC BGR uint8.
        out = np.clip(out * 0.5 + 0.5, 0, 1).transpose(1, 2, 0)[:, :, ::-1]
        restored = (out * 255.0).astype(np.uint8)
        h, w = frame.shape[:2]
        mask, IM = _feathered_face_mask(M, self.size, self.size, h, w, np, cv2)
        if mask is None:
            return frame
        if blend < 0.999:
            mask = mask * blend
        warped = cv2.warpAffine(restored, IM, (w, h), borderValue=0.0)
        merged = mask * warped + (1.0 - mask) * frame.astype(np.float32)
        return merged.astype(np.uint8)


def _load_enhancer_config(np, cv2, worker_id, providers):
    """Read FACESWAP_* enhancement env into a config dict, and load GFPGAN if
    enabled and the model file is present. Never raises — on any problem it
    disables the enhancer and logs, so a missing/broken model can't kill swaps.
    """
    def _flag(name, default):
        return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")

    cfg = {
        "enhance": _flag("FACESWAP_ENHANCE", "1"),          # custom colour+sharpen paste
        "color": _flag("FACESWAP_COLOR_MATCH", "1"),
        "color_strength": float(os.getenv("FACESWAP_COLOR_STRENGTH", "0.6")),
        "sharpen": float(os.getenv("FACESWAP_SHARPEN", "0.4")),
        "enhancer": None,                                    # FaceEnhancer or None
        "enhancer_blend": float(os.getenv("FACESWAP_ENHANCER_BLEND", "0.8")),
        # GFPGAN is the costliest stage (512px restore + two full-frame warps per
        # face). Restoration only visibly helps faces that are large on screen;
        # tiny/distant faces gain nothing but pay the same ~40ms. Skip GFPGAN when
        # the detected face's longest bbox side is below this many pixels (they
        # still get the near-free colour+sharpen paste). 0 = restore every face.
        "enhancer_min_face": float(os.getenv("FACESWAP_ENHANCER_MIN_FACE", "0")),
    }
    if not cfg["enhance"]:
        print(f"[worker-{worker_id}] enhancement disabled (FACESWAP_ENHANCE=0)", flush=True)
        return cfg

    if _flag("FACESWAP_ENHANCER", "1"):
        root = os.getenv("FACESWAP_ROOT") or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        model_path = os.getenv("FACESWAP_ENHANCER_MODEL",
                               os.path.join(root, "models", "gfpgan_1.4.onnx"))
        if os.path.isfile(model_path):
            try:
                cfg["enhancer"] = FaceEnhancer(model_path, providers, np, cv2)
                print(f"[worker-{worker_id}] GFPGAN enhancer loaded "
                      f"({os.path.basename(model_path)} @ {cfg['enhancer'].size}px, "
                      f"blend={cfg['enhancer_blend']}, "
                      f"providers={cfg['enhancer'].providers()})", flush=True)
            except Exception as e:
                print(f"[worker-{worker_id}] GFPGAN load FAILED ({e}) — "
                      f"continuing with colour+sharpen only", flush=True)
                cfg["enhancer"] = None
        else:
            print(f"[worker-{worker_id}] GFPGAN model not found at {model_path} — "
                  f"colour+sharpen only", flush=True)
    print(f"[worker-{worker_id}] enhance: color={cfg['color']} "
          f"strength={cfg['color_strength']} sharpen={cfg['sharpen']} "
          f"gfpgan={'on' if cfg['enhancer'] else 'off'} "
          f"min_face={cfg['enhancer_min_face']:.0f}px", flush=True)
    return cfg


# ---- Worker entry point ----------------------------------------------------

def worker_main(
    worker_id: int,
    in_q,                       # mp.Queue[SwapRequest]
    out_q,                      # mp.Queue[SwapResponse]
    shm_names: list,            # list[str] of SharedMemory names (one per slot)
    shape: tuple,               # (H, W, 3) — slot frame shape
    ref_embs_bytes: bytes,      # pickled numpy.ndarray, shape (S, D) — centroids
    ref_sources_pickled: bytes, # pickled list[SourceSpec-like dict]
    ref_members_pickled: bytes, # pickled list[numpy.ndarray (M_s, D)] — per-source cluster members
    det_size: int,              # face detector input (square)
    det_thresh: float,          # detector confidence threshold
    ref_thresh: float,          # cosine-sim threshold for source-match
    models_face_dir: Optional[str],  # FACESWAP_FACE_MODEL or None
    inswapper_path: str,        # absolute path to inswapper_128_fp16.onnx
) -> None:
    """Process entry. Loads models, then loops on `in_q` until END.

    Sub-task 5.1: stub — load models, ack startup, drain in_q with end-of-stream
    response only. Real per-frame swap logic lands in sub-task 5.3 (after the
    master wiring is also in place — keeps the diff reviewable).
    """
    t0 = time.perf_counter()
    try:
        # Late imports — must follow _register_cuda_dll_dirs() above.
        # cv2 + numpy first; they're cheap and don't touch CUDA.
        import numpy as np  # noqa: F401  (used once SwapResponse handling lands)
        import cv2          # noqa: F401  (used by paste-back internals)
        import insightface
        from insightface.app import FaceAnalysis

        face_model = models_face_dir or "buffalo_l"
        # Frugal CUDA options for the MAIN models (detector + inswapper), mirroring
        # the GFPGAN enhancer below. With N workers each loading buffalo_l +
        # inswapper on one 16 GB card, ORT's default EXHAUSTIVE conv search + max
        # workspace + power-of-two arena over-allocates VRAM until CUDA spills to
        # system RAM over PCIe — which collapses throughput to ~1-2 fps. HEURISTIC
        # search, no max workspace, and grow-as-needed keep every worker resident
        # in VRAM so the GPU runs at full speed. Optional hard per-session cap via
        # FACESWAP_WORKER_GPU_MEM_MB.
        _cuda_opts = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        }
        _gpu_mb = os.getenv("FACESWAP_WORKER_GPU_MEM_MB", "").strip()
        if _gpu_mb:
            _cuda_opts["gpu_mem_limit"] = str(int(_gpu_mb) * 1024 * 1024)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [_cuda_opts, {}]

        # Stagger model-file reads across workers. When N>=6 workers all run
        # onnx.load() on the same .onnx file simultaneously on Windows, the
        # protobuf parser intermittently fails with `DecodeError: Error parsing
        # message` — Windows-level concurrent-read returns a partial buffer to
        # one of the racers. Sleep ~1s per worker_id before opening any models
        # to serialise the load path. Costs `(N-1) × 1s` to first-frame at
        # job start; not in the steady-state hot loop.
        time.sleep(worker_id * 1.0)

        print(f"[worker-{worker_id}] loading FaceAnalysis({face_model}) "
              f"det_size={det_size} det_thresh={det_thresh}...", flush=True)
        fa = FaceAnalysis(name=face_model, providers=providers,
                          provider_options=provider_options)
        fa.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)

        print(f"[worker-{worker_id}] loading inswapper from {inswapper_path}...",
              flush=True)
        sw = insightface.model_zoo.get_model(inswapper_path, providers=providers,
                                             provider_options=provider_options)

        # Verify CUDA actually loaded — see CLAUDE.md issue #8.
        try:
            active = sw.session.get_providers()
        except AttributeError:
            active = None
        if active and active == ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"[worker-{worker_id}] inswapper loaded on CPU only — "
                f"CUDA failed to initialise. Check cuDNN DLL discovery."
            )

        load_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[worker-{worker_id}] models loaded in {load_ms:.0f} ms "
              f"(providers={active})", flush=True)

        # Unpickle reference embeddings + source faces (master computed these once).
        # ref_embs: (S, D) float32 normed embeddings, one row per source.
        # ref_sources_raw: list[dict] with key 'src_face_dict' holding the source-image
        #                  Face as a plain dict (insightface.Face.__reduce__ is buggy
        #                  across the spawn boundary — we re-wrap as Face here, where
        #                  insightface is already imported).
        from insightface.app.common import Face
        ref_embs = pickle.loads(ref_embs_bytes)
        ref_sources_raw = pickle.loads(ref_sources_pickled)
        ref_sources = [
            {"src_face": Face(s["src_face_dict"]), "gender": s["gender"]}
            for s in ref_sources_raw
        ]
        ref_embs_T = ref_embs.T  # (D, S) — pre-transpose for the per-frame matmul
        # Per-source cluster member embeddings for NN-over-members matching.
        # Shape per element: (M_s, D). max(tgt @ member.T) is more robust than
        # tgt @ centroid because it captures pose / lighting variation within
        # the cluster. Stack into (total_M, D) + a (total_M,) source-index map
        # so the per-frame matmul stays one big GEMM, then per-cluster max.
        ref_members_per = pickle.loads(ref_members_pickled)  # list of (M_s, D)
        if any(m.size > 0 for m in ref_members_per):
            stacked = np.concatenate(
                [m for m in ref_members_per if m.size > 0], axis=0
            ).astype(np.float32)
            source_idx_map = np.concatenate(
                [np.full(m.shape[0], s, dtype=np.int32)
                 for s, m in enumerate(ref_members_per) if m.size > 0]
            )
            stacked_T = stacked.T   # (D, total_M) for tgt @ stacked_T
        else:
            stacked_T = None
            source_idx_map = None

        # Attach the shared-memory slots and build per-slot numpy views ONCE.
        # Per-frame we just index into `slot_views[slot_id]`; no slicing or
        # SharedMemory lookup in the hot loop.
        from multiprocessing import shared_memory
        slots = []
        slot_views = []
        for name in shm_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"[worker-{worker_id}] shared memory '{name}' not found: {e}"
                ) from None
            slots.append(shm)
            slot_views.append(np.ndarray(shape, dtype=np.uint8, buffer=shm.buf))

        # Load swap-quality enhancement config (colour+sharpen paste, GFPGAN).
        # Reads FACESWAP_* env (inherited from the parent). Never raises.
        from insightface.utils import face_align
        enh = _load_enhancer_config(np, cv2, worker_id, providers)
        swap_size = sw.input_size[0]   # 128 — for re-deriving the aligned crop

        # Tell master we're ready (frame_idx=-2 is the startup-ack convention).
        out_q.put(SwapResponse(frame_idx=-2, worker_id=worker_id))

        # ---- Main per-frame loop -----------------------------------------------
        # In-process pipeline is intentionally single-threaded: every per-frame
        # step (decode-from-shm, detect, match, fused swap+paste, encode-to-shm)
        # is GIL-serialised against itself anyway, and parallelism comes from
        # running N of these processes in parallel. Adding threads here just
        # brings back the GIL contention we measured in Phase 3.
        while True:
            req: SwapRequest = in_q.get()
            if req.end:
                out_q.put(SwapResponse(frame_idx=-1, worker_id=worker_id))
                break

            tA = time.perf_counter()
            slot_id = int(req.slot_id)
            frame_idx = int(req.frame_idx)
            try:
                # In-place numpy view of the slot; master already wrote the input
                # BGR frame here. We mutate it in place with the swap result.
                frame = slot_views[slot_id]

                # Detect + match against the pre-stacked source references.
                # NN-over-members: per target face, per source, compute the MAX
                # cosine sim across all cluster members (not just the centroid).
                # A face that closely matches any one member is "in the cluster"
                # even if its embedding has drifted from the centroid due to
                # pose / lighting. This kills threshold-boundary flicker —
                # max-over-members is much more stable per-frame than
                # centroid-only sim.
                tgt_faces = fa.get(frame)
                n_swapped = 0
                if tgt_faces:
                    tgt_embs = np.stack([f.normed_embedding for f in tgt_faces]).astype(np.float32)
                    if stacked_T is not None and source_idx_map is not None:
                        # (T, total_M) — sim of each target to every cluster member.
                        all_sims = tgt_embs @ stacked_T
                        S = len(ref_sources)
                        # Per source: take max across that source's columns.
                        sims = np.full((all_sims.shape[0], S), -1.0, dtype=np.float32)
                        for s in range(S):
                            cols = (source_idx_map == s)
                            if cols.any():
                                sims[:, s] = all_sims[:, cols].max(axis=1)
                    else:
                        sims = tgt_embs @ ref_embs_T      # fallback to centroid sim
                    for ti, si in _match_faces_to_sources(sims, ref_thresh):
                        tface = tgt_faces[ti]
                        src_face = ref_sources[si]["src_face"]
                        if enh["enhance"]:
                            # Custom paste: get the raw 128px swap + affine
                            # matrix, colour-match it to the target lighting
                            # and sharpen, then paste back ourselves.
                            bgr_fake, M = sw.get(frame, tface, src_face,
                                                 paste_back=False)
                            aimg, _ = face_align.norm_crop2(
                                frame, tface.kps, swap_size)
                            frame[...] = _paste_enhanced(
                                frame, aimg, bgr_fake, M, enh, np, cv2)
                            # GFPGAN restoration on the just-swapped face,
                            # but only when the face is large enough to
                            # benefit — skipping tiny faces is the main
                            # FPS lever in multi-face / crowd scenes.
                            if enh["enhancer"] is not None:
                                x1, y1, x2, y2 = tface.bbox
                                face_px = max(x2 - x1, y2 - y1)
                                if face_px >= enh["enhancer_min_face"]:
                                    frame[...] = enh["enhancer"].enhance(
                                        frame, tface.kps, enh["enhancer_blend"])
                        else:
                            # Stock inswapper fused swap+paste (byte-for-byte
                            # identical to the Phase 2 path). FACESWAP_ENHANCE=0.
                            frame[...] = sw.get(frame, tface, src_face,
                                                paste_back=True)
                        n_swapped += 1

                elapsed = (time.perf_counter() - tA) * 1000.0
                out_q.put(SwapResponse(
                    frame_idx=frame_idx, slot_id=slot_id,
                    n_swapped=n_swapped, worker_id=worker_id,
                    elapsed_ms=elapsed,
                ))
            except Exception as e:
                # Per-frame failure shouldn't kill the worker — report and continue.
                # Master decides whether a single-frame error is fatal.
                tb = traceback.format_exc()
                print(f"[worker-{worker_id}] frame {frame_idx} error: {e}\n{tb}",
                      flush=True)
                out_q.put(SwapResponse(
                    frame_idx=frame_idx, slot_id=slot_id,
                    worker_id=worker_id,
                    error=f"{type(e).__name__}: {e}",
                ))

        # Release the shared-memory handles cleanly (master owns + unlinks).
        for s in slots:
            try:
                s.close()
            except Exception:
                pass
        print(f"[worker-{worker_id}] exited cleanly", flush=True)

    except Exception as e:
        # Surface to master so the job fails loudly instead of stalling.
        tb = traceback.format_exc()
        print(f"[worker-{worker_id}] FATAL: {e}\n{tb}", flush=True)
        try:
            out_q.put(SwapResponse(
                frame_idx=-1, worker_id=worker_id,
                error=f"{type(e).__name__}: {e}",
            ))
        except Exception:
            pass
        sys.exit(1)
