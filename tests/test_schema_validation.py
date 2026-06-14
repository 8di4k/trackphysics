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
    with pytest.raises(ValueError, match="non-decreasing frame"):
        tp.TrackSequence(detections=dets, fps=30.0)


def test_in_order_with_gaps_is_ok() -> None:
    dets = [_det(f) for f in (0, 2, 5)]
    track = tp.TrackSequence(detections=dets, fps=30.0)
    assert len(track) == 3
