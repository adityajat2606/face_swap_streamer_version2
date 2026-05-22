"""Reference extraction: cluster the video's own faces by identity, then bind
each source (by gender) to a cluster (PRD §FR-4, §16 Step 3-4).

The swap is routed by matching per-frame faces to the *video person's* identity
cluster (NN over members), not to the new source photo (a different person). The
greedy clustering is pure numpy and unit-tested; ``extract_references`` wraps it
with the GPU detector (validated on the GPU host).
"""

from __future__ import annotations

import numpy as np

from .matching import assign_sources_to_clusters


def cluster_embeddings(
    embs: np.ndarray,
    scores: np.ndarray,
    gender_labels: list[str],
    *,
    thresh: float = 0.30,
) -> list[dict]:
    """Greedy identity clustering (no gender filter — PRD §FR-4 rationale).

    At each step take the unused candidate with the most neighbours within
    ``thresh`` cosine, claim its neighbourhood as a cluster, repeat. Returns
    clusters sorted by size desc, each: ``{rep, members(np int), size, gender,
    m_frac, f_frac}``. Pure numpy.
    """
    n = len(scores)
    if n == 0:
        return []
    embs = embs.astype(np.float32)
    sim = embs @ embs.T
    scores = np.asarray(scores, np.float32)
    unused = np.ones(n, dtype=bool)
    clusters: list[dict] = []
    while unused.any():
        votes = (sim > thresh).astype(np.float32).sum(axis=1) * scores * unused.astype(np.float32)
        rep = int(np.argmax(votes))
        if not unused[rep] or votes[rep] <= 0:
            break
        members = np.where(unused & (sim[rep] > thresh))[0]
        if len(members) == 0:
            unused[rep] = False
            continue
        mg = [gender_labels[m] for m in members]
        n_m = sum(1 for g in mg if g == "M")
        n_f = sum(1 for g in mg if g == "F")
        tot = max(1, n_m + n_f)
        clusters.append({
            "rep": rep,
            "members": members,
            "size": int(len(members)),
            "score": float(scores[rep]),
            "gender": "M" if n_m >= n_f else "F",
            "m_frac": n_m / tot,
            "f_frac": n_f / tot,
        })
        unused[members] = False
    clusters.sort(key=lambda c: (-c["size"], -c["score"]))
    return clusters


def extract_references(
    detector,
    video_path: str,
    source_genders: list[str],
    *,
    sample_sec: float = 1.0,
    max_samples: int = 300,
    min_face_w: int = 25,
) -> list[dict]:  # pragma: no cover - needs GPU detector + a real video
    """Scan the video, cluster identities, bind each source to a cluster.

    Returns one dict per source (source-ordered): ``{ref_emb (D,), ref_members
    (M,D), gender, ref_frame, ref_votes}``. GPU-host only.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        from .errors import InputError

        raise InputError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * sample_sec))

    cand_emb: list[np.ndarray] = []
    cand_score: list[float] = []
    cand_gender: list[str] = []
    cand_frame: list[int] = []
    i = 0
    while i < total and len(cand_emb) < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        for f in detector.app.get(fr):
            w = f.bbox[2] - f.bbox[0]
            if w < min_face_w:
                continue
            cand_emb.append(np.asarray(f.normed_embedding, np.float32))
            cand_score.append(float(w * f.det_score))
            cand_gender.append(f.sex)
            cand_frame.append(i)
        i += step
    cap.release()
    if not cand_emb:
        from .errors import DetectionError

        raise DetectionError("no face found in the video")

    embs = np.stack(cand_emb)
    clusters = cluster_embeddings(embs, np.asarray(cand_score), cand_gender)
    assign = assign_sources_to_clusters(source_genders, clusters)

    out = []
    for si, ci in enumerate(assign):
        c = clusters[min(ci, len(clusters) - 1)]
        mems = embs[c["members"]].astype(np.float32)
        cen = mems.mean(axis=0)
        nrm = float(np.linalg.norm(cen))
        out.append({
            "ref_emb": (cen / nrm).astype(np.float32) if nrm > 0 else mems[0],
            "ref_members": mems,
            "gender": source_genders[si],
            "ref_frame": int(cand_frame[c["rep"]]),
            "ref_votes": c["size"],
        })
    return out
