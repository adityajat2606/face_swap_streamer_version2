from __future__ import annotations

import numpy as np

from face_swap.reference_extractor import cluster_embeddings


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_two_identities_form_two_clusters(rng):
    # two identity centroids in different directions; jittered members each
    a = _unit([1, 0, 0, 0])
    b = _unit([0, 1, 0, 0])
    embs, scores, genders = [], [], []
    for _ in range(6):
        embs.append(_unit(a + 0.05 * rng.normal(size=4)))
        scores.append(1.0)
        genders.append("M")
    for _ in range(4):
        embs.append(_unit(b + 0.05 * rng.normal(size=4)))
        scores.append(1.0)
        genders.append("F")
    clusters = cluster_embeddings(np.stack(embs), np.array(scores), genders, thresh=0.5)
    assert len(clusters) == 2
    # largest first; sizes 6 then 4
    assert clusters[0]["size"] == 6 and clusters[1]["size"] == 4
    assert clusters[0]["gender"] == "M" and clusters[1]["gender"] == "F"


def test_empty_input():
    assert cluster_embeddings(np.zeros((0, 4), np.float32), np.array([]), []) == []


def test_single_identity_one_cluster(rng):
    a = _unit([1, 0, 0])
    embs = [_unit(a + 0.03 * rng.normal(size=3)) for _ in range(5)]
    clusters = cluster_embeddings(np.stack(embs), np.ones(5), ["M"] * 5, thresh=0.5)
    assert len(clusters) == 1 and clusters[0]["size"] == 5
