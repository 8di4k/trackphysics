"""Adapter tests: plain arrays, CSV, and duck-typed supervision input -> our schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import trackphysics as tp


def test_from_generic_groups_and_orders() -> None:
    # Two tracks, rows intentionally out of frame order.
    frames = [2, 0, 1, 0]
    boxes = np.array([[0, 0, 2, 2]] * 4, dtype=float)
    track_ids = [7, 7, 7, 9]
    seqs = tp.from_generic(frames=frames, boxes=boxes, track_ids=track_ids, fps=30.0)
    assert [s.detections[0].track_id for s in seqs] == [7, 9]  # sorted by id
    seq7 = seqs[0]
    assert list(seq7.frames) == [0, 1, 2]  # ordered by frame


def test_from_csv_roundtrip(tmp_path: Path) -> None:
    csv = tmp_path / "track.csv"
    csv.write_text(
        "frame,track_id,x1,y1,x2,y2,score\n"
        "0,1,0,0,10,10,0.9\n"
        "1,1,1,1,11,11,0.8\n"
    )
    seqs = tp.from_csv(csv, fps=60.0)
    assert len(seqs) == 1
    assert len(seqs[0]) == 2
    assert seqs[0].detections[0].score == pytest.approx(0.9)


@dataclass
class FakeDetections:
    xyxy: np.ndarray
    tracker_id: np.ndarray
    confidence: np.ndarray


def test_from_supervision_duck_typed() -> None:
    frames = [
        FakeDetections(
            xyxy=np.array([[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 6.0, 6.0]]),
            tracker_id=np.array([1, 2]),
            confidence=np.array([0.9, 0.7]),
        ),
        FakeDetections(
            xyxy=np.array([[0.5, 0.5, 5.5, 5.5]]),
            tracker_id=np.array([1]),
            confidence=np.array([0.95]),
        ),
    ]
    seqs = tp.from_supervision(frames, fps=120.0)
    ids = sorted(s.detections[0].track_id for s in seqs)
    assert ids == [1, 2]
    track1 = next(s for s in seqs if s.detections[0].track_id == 1)
    assert len(track1) == 2  # present in both frames


def test_from_supervision_requires_tracker_id() -> None:
    @dataclass
    class Untracked:
        xyxy: np.ndarray
        tracker_id: None = None

    with pytest.raises(ValueError):
        tp.from_supervision([Untracked(xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]))], fps=30.0)
