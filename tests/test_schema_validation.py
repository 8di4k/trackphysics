"""Regression tests for schema boundary validation (SCHEMA-TIMESTAMPS, SCHEMA-UNSORTED)."""

from __future__ import annotations

import numpy as np
import pytest

import trackphysics as tp


def _det(frame: int) -> tp.Detection:
    return tp.Detection(frame=frame, bbox=np.array([0.0, 0.0, 1.0, 1.0]), track_id=1)


def test_timestamps_length_mismatch_raises() -> None:
    dets = [_det(f) for f in range(3)]
    with pytest.raises(ValueError, match="timestamps length"):
        tp.TrackSequence(detections=dets, fps=30.0, timestamps=np.array([0.0, 0.1]))


def test_timestamps_must_be_1d() -> None:
    dets = [_det(f) for f in range(2)]
    with pytest.raises(ValueError, match="1-D"):
        tp.TrackSequence(detections=dets, fps=30.0, timestamps=np.zeros((2, 1)))


def test_out_of_order_frames_raise() -> None:
    dets = [_det(f) for f in (0, 2, 1)]
    with pytest.raises(ValueError, match="strictly increasing frame"):
        tp.TrackSequence(detections=dets, fps=30.0)


def test_duplicate_frames_raise() -> None:
    # A single object cannot occupy one frame twice; duplicates would give dt=0 -> degenerate
    # (inf/nan) velocity downstream, so they must fail loudly at the boundary.
    dets = [_det(f) for f in (0, 1, 1, 2)]
    with pytest.raises(ValueError, match="strictly increasing frame"):
        tp.TrackSequence(detections=dets, fps=30.0)


def test_in_order_with_gaps_is_ok() -> None:
    dets = [_det(f) for f in (0, 2, 5)]
    track = tp.TrackSequence(detections=dets, fps=30.0)
    assert len(track) == 3


def test_detection_owns_its_bbox_and_keypoints() -> None:
    # Mutating the caller's arrays must not bleed into the Detection (SCHEMA-ALIASING).
    bbox = np.array([0.0, 1.0, 2.0, 3.0])
    kp = np.array([[10.0, 20.0], [30.0, 40.0]])
    det = tp.Detection(frame=0, bbox=bbox, track_id=1, keypoints=kp)
    bbox[0] = 999.0
    kp[0, 0] = 999.0
    assert det.bbox[0] == 0.0
    assert det.keypoints is not None and det.keypoints[0, 0] == 10.0


def test_times_returns_a_copy_not_the_internal_timestamps() -> None:
    ts = np.array([0.0, 0.1, 0.2])
    track = tp.TrackSequence(detections=[_det(i) for i in range(3)], fps=10.0, timestamps=ts)
    out = track.times()
    out[0] = 99.0
    assert track.times()[0] == 0.0  # internal array untouched
    assert ts[0] == 0.0  # and the caller's original array untouched
