from __future__ import annotations

import numpy as np

from face_swap import metrics as M


def test_detection_success_rate():
    assert M.detection_success_rate(98, 100) == 0.98
    assert M.detection_success_rate(0, 0) == 0.0


def test_identity_cosine_distance():
    a = np.array([1.0, 0, 0])
    assert M.identity_cosine_distance(a, a) < 1e-6
    assert abs(M.identity_cosine_distance(a, np.array([0, 1.0, 0])) - 1.0) < 1e-6


def test_identity_drift_window():
    ref = np.array([1.0, 0])
    embs = [np.array([1.0, 0])] * 50 + [np.array([0.0, 1.0])] * 50
    drift = M.identity_drift_max_window(embs, ref, window=100)
    assert drift > 0.9


def test_percentile_and_median():
    vals = list(range(1, 101))
    assert 49 <= M.percentile(vals, 50) <= 51
    assert 94 <= M.percentile(vals, 95) <= 96
    assert M.median(vals) == 50.5
    assert M.percentile([], 95) == 0.0


def test_parse_nvidia_smi_util():
    log = "42 %\n50 %\n58 %\n"
    assert abs(M.parse_nvidia_smi_util(log) - 0.5) < 1e-6
    assert M.parse_nvidia_smi_util("") == 0.0


def test_parse_nvidia_smi_csv():
    assert abs(M.parse_nvidia_smi_util("40\n60\n") - 0.5) < 1e-6


def test_color_shift_delta_e_zero_for_identical():
    img = np.full((16, 16, 3), 120, np.uint8)
    assert M.color_shift_delta_e(img, img.copy()) < 1e-6
