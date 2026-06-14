"""Adapter hardening: partially-missing optional columns preserve per-row provenance.

Covers the column-handling fixes in the generic and supervision adapters:

* a CSV with a partially-blank optional column keeps the present values and yields
  ``None`` only for the blank rows (no all-or-nothing drop);
* :func:`from_generic` maps a per-element ``NaN`` score and a ``-1`` class-id sentinel
  to ``None`` while leaving present values untouched;
* :func:`from_supervision` across frames with mixed confidence/class-id availability
  preserves the present values;
* a fully-present column still round-trips exactly as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import trackphysics as tp
from trackphysics.core.adapters.generic import MISSING_CLASS_ID


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "track.csv"
    path.write_text(text)
    return path


def test_from_csv_partial_score_keeps_present_and_nones_blank(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "frame,track_id,x1,y1,x2,y2,score\n"
        "0,1,0,0,10,10,0.9\n"
        "1,1,1,1,11,11,\n"  # blank score cell
        "2,1,2,2,12,12,0.7\n",
    )
    seqs = tp.from_csv(path, fps=60.0)
    assert len(seqs) == 1
    dets = seqs[0].detections
    assert len(dets) == 3
    # Present scores survive; only the blank row is None (not all dropped).
    assert dets[0].score == pytest.approx(0.9)
    assert dets[1].score is None
    assert dets[2].score == pytest.approx(0.7)


def test_from_csv_partial_class_id_uses_sentinel_none(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "frame,track_id,x1,y1,x2,y2,class_id\n"
        "0,1,0,0,10,10,3\n"
        "1,1,1,1,11,11,\n"  # blank class_id cell
        "2,1,2,2,12,12,5\n",
    )
    seqs = tp.from_csv(path, fps=30.0)
    dets = seqs[0].detections
    assert [d.class_id for d in dets] == [3, None, 5]


def test_from_csv_fully_present_score_roundtrips(tmp_path: Path) -> None:
    # Regression guard: a clean column behaves exactly as before the hardening.
    path = _write(
        tmp_path,
        "frame,track_id,x1,y1,x2,y2,score\n"
        "0,1,0,0,10,10,0.9\n"
        "1,1,1,1,11,11,0.8\n",
    )
    seqs = tp.from_csv(path, fps=60.0)
    assert [d.score for d in seqs[0].detections] == pytest.approx([0.9, 0.8])


def test_from_csv_no_optional_columns_yields_none(tmp_path: Path) -> None:
    # No score/class_id header at all -> both stay None for every detection.
    path = _write(
        tmp_path,
        "frame,track_id,x1,y1,x2,y2\n0,1,0,0,10,10\n1,1,1,1,11,11\n",
    )
    seqs = tp.from_csv(path, fps=30.0)
    dets = seqs[0].detections
    assert all(d.score is None for d in dets)
    assert all(d.class_id is None for d in dets)


def test_from_generic_nan_score_maps_to_none() -> None:
    frames = [0, 1, 2]
    boxes = np.array([[0, 0, 2, 2]] * 3, dtype=float)
    track_ids = [1, 1, 1]
    scores = [0.9, float("nan"), 0.7]
    seqs = tp.from_generic(
        frames=frames, boxes=boxes, track_ids=track_ids, fps=30.0, scores=scores
    )
    dets = seqs[0].detections
    assert dets[0].score == pytest.approx(0.9)
    assert dets[1].score is None
    assert dets[2].score == pytest.approx(0.7)


def test_from_generic_sentinel_class_id_maps_to_none() -> None:
    frames = [0, 1, 2]
    boxes = np.array([[0, 0, 2, 2]] * 3, dtype=float)
    track_ids = [1, 1, 1]
    class_ids = [4, MISSING_CLASS_ID, 6]
    seqs = tp.from_generic(
        frames=frames, boxes=boxes, track_ids=track_ids, fps=30.0, class_ids=class_ids
    )
    assert [d.class_id for d in seqs[0].detections] == [4, None, 6]


def test_from_generic_clean_arrays_unchanged() -> None:
    # Fully-present columns (no NaN, no sentinel) behave exactly as before.
    frames = [0, 1]
    boxes = np.array([[0, 0, 2, 2]] * 2, dtype=float)
    track_ids = [1, 1]
    seqs = tp.from_generic(
        frames=frames,
        boxes=boxes,
        track_ids=track_ids,
        fps=30.0,
        scores=[0.5, 0.6],
        class_ids=[2, 3],
    )
    dets = seqs[0].detections
    assert [d.score for d in dets] == pytest.approx([0.5, 0.6])
    assert [d.class_id for d in dets] == [2, 3]


@dataclass
class _DetsConf:
    xyxy: np.ndarray
    tracker_id: np.ndarray
    confidence: np.ndarray


@dataclass
class _DetsNoConf:
    xyxy: np.ndarray
    tracker_id: np.ndarray


def test_from_supervision_mixed_confidence_preserves_present() -> None:
    # Frame 0 carries confidence, frame 1 (a different object) does not. The present
    # confidence must survive rather than be dropped by a length mismatch.
    frame0 = _DetsConf(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]]),
        tracker_id=np.array([1]),
        confidence=np.array([0.9]),
    )
    frame1 = _DetsNoConf(
        xyxy=np.array([[10.0, 10.0, 15.0, 15.0]]),
        tracker_id=np.array([2]),
    )
    seqs = tp.from_supervision([frame0, frame1], fps=120.0)
    by_id = {s.detections[0].track_id: s for s in seqs}
    assert by_id[1].detections[0].score == pytest.approx(0.9)
    # Frame 1 lacked confidence -> NaN -> None for that detection.
    assert by_id[2].detections[0].score is None


def test_from_supervision_mixed_confidence_within_frame_object() -> None:
    # Same track id seen across two frames, only the first frame has confidence.
    frame0 = _DetsConf(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]]),
        tracker_id=np.array([7]),
        confidence=np.array([0.8]),
    )
    frame1 = _DetsNoConf(
        xyxy=np.array([[1.0, 1.0, 6.0, 6.0]]),
        tracker_id=np.array([7]),
    )
    seqs = tp.from_supervision([frame0, frame1], fps=60.0)
    assert len(seqs) == 1
    scores = [d.score for d in seqs[0].detections]
    assert scores[0] == pytest.approx(0.8)
    assert scores[1] is None


def test_from_supervision_clean_confidence_roundtrips() -> None:
    # Regression guard: every frame carries confidence -> unchanged behaviour.
    frame0 = _DetsConf(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]]),
        tracker_id=np.array([1]),
        confidence=np.array([0.9]),
    )
    frame1 = _DetsConf(
        xyxy=np.array([[0.5, 0.5, 5.5, 5.5]]),
        tracker_id=np.array([1]),
        confidence=np.array([0.95]),
    )
    seqs = tp.from_supervision([frame0, frame1], fps=120.0)
    scores = [d.score for d in seqs[0].detections]
    assert scores == pytest.approx([0.9, 0.95])
    assert not any(s is None or math.isnan(s) for s in scores)


@dataclass
class _DetsUntracked:
    xyxy: np.ndarray
    tracker_id: None = None


def test_from_supervision_skips_empty_untracked_frame() -> None:
    # A detector emits an empty frame (no detections, tracker_id None) before/without
    # tracking; it must be SKIPPED, not abort the whole conversion (the docstring promises
    # untracked frames are skipped).
    empty = _DetsUntracked(xyxy=np.zeros((0, 4)))
    tracked = _DetsConf(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]]), tracker_id=np.array([1]), confidence=np.array([0.9])
    )
    seqs = tp.from_supervision([empty, tracked], fps=30.0)
    assert len(seqs) == 1
    assert seqs[0].detections[0].track_id == 1


def test_from_supervision_nonempty_untracked_frame_still_raises() -> None:
    # But a frame WITH detections and no tracker_id is a real error (tracking not run).
    bad = _DetsUntracked(xyxy=np.array([[0.0, 0.0, 5.0, 5.0]]))
    with pytest.raises(ValueError, match="tracker_id"):
        tp.from_supervision([bad], fps=30.0)


def test_from_generic_rejects_mismatched_optional_length() -> None:
    with pytest.raises(ValueError, match="scores must have length 3"):
        tp.from_generic(
            frames=[0, 1, 2], boxes=np.zeros((3, 4)), track_ids=[1, 1, 1], fps=30.0,
            scores=[0.9, 0.8],
        )
    with pytest.raises(ValueError, match="keypoints must have length 3"):
        tp.from_generic(
            frames=[0, 1, 2], boxes=np.zeros((3, 4)), track_ids=[1, 1, 1], fps=30.0,
            keypoints=np.zeros((2, 4, 2)),
        )
