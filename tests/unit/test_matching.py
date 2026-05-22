from __future__ import annotations

import numpy as np

from face_swap.matching import SourceTracker, assign_sources_to_clusters, bbox_iou


def _c(size, m_frac):
    return {"size": size, "m_frac": m_frac, "f_frac": 1.0 - m_frac}


def test_assign_clean_duet():
    clusters = [_c(500, 0.95), _c(400, 0.05)]
    assign = assign_sources_to_clusters(["M", "F"], clusters)
    assert assign[0] == 0 and assign[1] == 1


def test_assign_female_smaller_and_mislabeled():
    # actress cluster (idx 2) is smaller AND hard-labeled 'M' (m_frac 0.55)
    clusters = [_c(500, 0.95), _c(200, 0.90), _c(80, 0.55)]
    assign = assign_sources_to_clusters(["M", "F"], clusters)
    assert assign[0] == 0           # male -> male lead
    assert assign[1] == 2           # female -> the actress cluster, not the male extra


def test_assign_no_female_present_degrades():
    clusters = [_c(500, 0.95), _c(200, 0.90)]
    assign = assign_sources_to_clusters(["M", "F"], clusters)
    assert assign[0] == 0 and assign[1] == 1   # distinct, no crash


def test_assign_two_female_sources():
    clusters = [_c(600, 0.95), _c(300, 0.10), _c(150, 0.20)]
    assign = assign_sources_to_clusters(["F", "F"], clusters)
    assert set(assign) == {1, 2}    # both female clusters, not the male lead


def test_bbox_iou():
    assert bbox_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert bbox_iou([0, 0, 2, 2], [10, 10, 12, 12]) == 0.0


def _bb(x):
    return np.array([x, 0, x + 20, 20], np.float32)


def test_tracker_duet_forces_both():
    tr = SourceTracker(2, 0.15)
    # female sub-threshold but T==S forces both
    picks = tr.match(np.array([[0.5, 0.05], [0.05, 0.10]]), [_bb(0), _bb(100)])
    srcs = {si for _, si in picks}
    assert srcs == {0, 1}


def test_tracker_hysteresis_holds_through_angle_dip():
    tr = SourceTracker(2, 0.15)
    extra = _bb(300)
    male, fem = _bb(0), _bb(100)
    seq = [
        np.array([[0.5, 0.05], [0.05, 0.40], [0.04, 0.03]]),  # lock
        np.array([[0.5, 0.05], [0.05, 0.10], [0.04, 0.03]]),  # dip -> sticky
        np.array([[0.5, 0.05], [0.05, 0.07], [0.04, 0.03]]),  # deeper -> carry
        np.array([[0.5, 0.05], [0.05, 0.33], [0.04, 0.03]]),  # recover
    ]
    for sims in seq:
        picks = tr.match(sims, [male, fem, extra])
        assert any(si == 1 for _, si in picks), "female dropped"
        assert not any(ti == 2 for ti, _ in picks), "extra wrongly swapped"


def test_tracker_absent_lead_not_forced():
    tr = SourceTracker(2, 0.15)
    picks = tr.match(np.array([[0.5, 0.04]]), [_bb(0)])  # 1 face, 2 sources
    assert picks == [(0, 0)]   # only the male; female not forced onto him
