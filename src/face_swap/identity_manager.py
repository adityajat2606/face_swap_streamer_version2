"""Identity management — reference embeddings and hero/heroine assignment.

Stateless / pure functions (CLAUDE.md §11). The reference-loading helper takes a
detector callable so this module never imports insightface directly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .errors import AmbiguousIdentityError, InputError
from .types import Identity, TrackState

# A detector that returns a list of (embedding, area) for an image path.
RefEmbedFn = Callable[[str], list[tuple[np.ndarray, float]]]


def load_identity(embed_fn: RefEmbedFn, ref_path: str) -> np.ndarray:
    """Return the unit-norm ArcFace embedding of the largest face in ``ref_path``."""
    faces = embed_fn(ref_path)
    if not faces:
        raise InputError(f"no face found in reference {ref_path}")
    faces = sorted(faces, key=lambda fa: fa[1], reverse=True)
    emb = np.asarray(faces[0][0], dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def assign_identities(
    tracks: list[TrackState],
    hero_emb: np.ndarray,
    heroine_emb: np.ndarray,
    *,
    ambiguity_margin: float = 0.05,
) -> dict[int, Identity]:
    """Map track_id -> identity via Hungarian on (track, identity) cosine sim.

    Raises :class:`AmbiguousIdentityError` when the two best tracks are within
    ``ambiguity_margin`` of each other for both identities (§7.4).
    """
    if not tracks:
        return {}
    identities: list[Identity] = ["hero", "heroine"]
    ref_embs = [hero_emb, heroine_emb]

    # similarity matrix S[t, i]
    sim = np.zeros((len(tracks), len(identities)), dtype=np.float64)
    for ti, tr in enumerate(tracks):
        for ii, ref in enumerate(ref_embs):
            sim[ti, ii] = _cosine_sim(tr.last_embedding, ref)

    # Hungarian maximizes similarity == minimizes (1 - sim).
    row_ind, col_ind = linear_sum_assignment(1.0 - sim)
    mapping: dict[int, Identity] = {tr.track_id: "unknown" for tr in tracks}
    for r, c in zip(row_ind, col_ind, strict=True):
        mapping[tracks[r].track_id] = identities[c]

    # Ambiguity guard: if the top two tracks per identity are within the margin
    # for *both* identities, we cannot tell who is who.
    if len(tracks) >= 2:
        for ii in range(len(identities)):
            col = np.sort(sim[:, ii])[::-1]
            if len(col) >= 2 and abs(col[0] - col[1]) < ambiguity_margin:
                # only ambiguous if it's ambiguous for both identities
                other = 1 - ii
                col2 = np.sort(sim[:, other])[::-1]
                if len(col2) >= 2 and abs(col2[0] - col2[1]) < ambiguity_margin:
                    raise AmbiguousIdentityError(
                        "hero/heroine assignment ambiguous "
                        f"(similarity margins < {ambiguity_margin}); manual confirmation required"
                    )
    return mapping
