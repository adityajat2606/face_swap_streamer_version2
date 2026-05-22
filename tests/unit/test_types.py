from __future__ import annotations

import numpy as np

from face_swap.types import BBox, FaceDetection, FrameResult, Landmarks


def test_bbox_geometry():
    b = BBox(0, 0, 4, 3, 0.9)
    assert b.width == 4 and b.height == 3
    assert b.area == 12
    assert b.diag == 5.0
    assert b.center == (2.0, 1.5)


def test_bbox_iou_identical_and_disjoint():
    a = BBox(0, 0, 2, 2)
    assert a.iou(a) == 1.0
    far = BBox(10, 10, 12, 12)
    assert a.iou(far) == 0.0


def test_bbox_iou_half_overlap():
    a = BBox(0, 0, 2, 2)
    b = BBox(1, 0, 3, 2)
    # intersection 2, union 6
    assert abs(a.iou(b) - 2 / 6) < 1e-9


def test_bbox_zero_area_no_div_zero():
    a = BBox(0, 0, 0, 0)
    assert a.iou(a) == 0.0


def test_frame_result_replace_and_json():
    fr = FrameResult(frame_idx=5, verdict="PASS", flicker_score=0.01,
                     components={"embedding": 0.0})
    fr2 = fr.replace(verdict="WARNING", retry_count=2)
    assert fr.verdict == "PASS" and fr2.verdict == "WARNING" and fr2.retry_count == 2
    obj = fr2.to_json_obj()
    assert obj["frame"] == 5 and obj["verdict"] == "WARNING"
    assert obj["retry_count"] == 2


def test_array_types_are_not_hashed_on_eq():
    lm = Landmarks(points=np.zeros((5, 2), np.float32))
    det = FaceDetection(frame_idx=0, bbox=BBox(0, 0, 1, 1), landmarks=lm,
                        embedding=np.ones(512, np.float32))
    # eq=False -> identity equality, no ambiguous-truth ValueError
    assert det == det
    assert det != FaceDetection(frame_idx=0, bbox=BBox(0, 0, 1, 1), landmarks=lm)
