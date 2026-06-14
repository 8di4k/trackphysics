"""Generic adapter: plain arrays / CSV into our input schema.

Thin and swappable (BRIEF.md §4): the core never depends on any upstream library's
types. This adapter turns flat per-detection rows (frame, track_id, bbox, ...) into one
:class:`TrackSequence` per track id.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..schema import Detection, FloatArray, SkeletonGraph, TrackSequence

MISSING_CLASS_ID = -1
"""Sentinel for a per-detection missing ``class_id`` in a flat ``class_ids`` array.

A row carrying this value maps to :attr:`Detection.class_id` ``None`` rather than an
opaque class id of ``-1``. This lets a declared-but-partially-blank column preserve
per-row alignment instead of being dropped wholesale (BRIEF.md §10: never silently
lose provenance)."""


def _score_or_none(score_arr: FloatArray | None, i: int) -> float | None:
    """Map element ``i`` of a score column to a value or ``None``.

    A ``NaN`` entry denotes a per-detection missing score; a fully-present column has
    no ``NaN`` and so round-trips unchanged.
    """
    if score_arr is None:
        return None
    value = float(score_arr[i])
    return None if np.isnan(value) else value


def _class_or_none(class_arr: npt.NDArray[np.int64] | None, i: int) -> int | None:
    """Map element ``i`` of a class-id column to a value or ``None``.

    A :data:`MISSING_CLASS_ID` entry denotes a per-detection missing class id; a
    fully-present column (no sentinels) round-trips unchanged.
    """
    if class_arr is None:
        return None
    value = int(class_arr[i])
    return None if value == MISSING_CLASS_ID else value


def from_generic(
    *,
    frames: npt.ArrayLike,
    boxes: npt.ArrayLike,
    track_ids: npt.ArrayLike,
    fps: float,
    scores: npt.ArrayLike | None = None,
    class_ids: npt.ArrayLike | None = None,
    keypoints: npt.ArrayLike | None = None,
    timestamps: npt.ArrayLike | None = None,
    image_size: tuple[int, int] | None = None,
    skeleton: SkeletonGraph | None = None,
) -> list[TrackSequence]:
    """Build one :class:`TrackSequence` per track id from flat per-detection arrays.

    Args:
        frames: ``(N,)`` integer frame index per detection.
        boxes: ``(N, 4)`` ``xyxy`` pixel boxes.
        track_ids: ``(N,)`` integer track id per detection.
        fps: Frames per second.
        scores: Optional ``(N,)`` detector confidences. A ``NaN`` entry marks a
            per-detection missing score and maps to :attr:`Detection.score` ``None``.
        class_ids: Optional ``(N,)`` opaque class ids. A ``-1`` sentinel marks a
            per-detection missing class id and maps to :attr:`Detection.class_id`
            ``None``.
        keypoints: Optional ``(N, K, 2)`` or ``(N, K, 3)`` keypoints in pixels.
        timestamps: Optional ``(N,)`` per-detection timestamps in seconds.
        image_size: Optional ``(W, H)`` in pixels.
        skeleton: Optional skeleton graph shared by all tracks.

    Returns:
        One :class:`TrackSequence` per distinct track id, each ordered by frame, sorted
        by ascending track id.
    """
    frame_arr = np.asarray(frames, dtype=np.int64)
    box_arr = np.asarray(boxes, dtype=np.float64)
    id_arr = np.asarray(track_ids, dtype=np.int64)
    n = frame_arr.shape[0]
    if box_arr.shape != (n, 4):
        raise ValueError(f"boxes must have shape ({n}, 4), got {box_arr.shape}")
    if id_arr.shape != (n,):
        raise ValueError(f"track_ids must have shape ({n},), got {id_arr.shape}")

    score_arr = None if scores is None else np.asarray(scores, dtype=np.float64)
    class_arr = None if class_ids is None else np.asarray(class_ids, dtype=np.int64)
    kp_arr = None if keypoints is None else np.asarray(keypoints, dtype=np.float64)
    ts_arr = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)

    rows_by_id: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        rows_by_id[int(id_arr[i])].append(i)

    sequences: list[TrackSequence] = []
    for tid in sorted(rows_by_id):
        rows = sorted(rows_by_id[tid], key=lambda i: int(frame_arr[i]))
        detections = [
            Detection(
                frame=int(frame_arr[i]),
                bbox=box_arr[i],
                track_id=tid,
                score=_score_or_none(score_arr, i),
                keypoints=None if kp_arr is None else kp_arr[i],
                class_id=_class_or_none(class_arr, i),
            )
            for i in rows
        ]
        track_ts: FloatArray | None = None
        if ts_arr is not None:
            track_ts = np.asarray([ts_arr[i] for i in rows], dtype=np.float64)
        sequences.append(
            TrackSequence(
                detections=detections,
                fps=fps,
                timestamps=track_ts,
                skeleton=skeleton,
                image_size=image_size,
            )
        )
    return sequences


def from_csv(
    path: str | Path,
    *,
    fps: float,
    image_size: tuple[int, int] | None = None,
    skeleton: SkeletonGraph | None = None,
) -> list[TrackSequence]:
    """Load tracks from a CSV with header columns.

    Required columns: ``frame``, ``track_id``, ``x1``, ``y1``, ``x2``, ``y2``.
    Optional columns: ``score``, ``class_id``. One row per detection.

    When the header declares an optional column, it is passed through for **every** row,
    preserving per-row alignment: a blank ``score`` cell becomes ``NaN`` (then
    :attr:`Detection.score` ``None``) and a blank ``class_id`` cell becomes the
    :data:`MISSING_CLASS_ID` sentinel (then :attr:`Detection.class_id` ``None``). A
    single blank cell no longer drops the whole column (BRIEF.md §10).
    """
    frames: list[int] = []
    boxes: list[list[float]] = []
    track_ids: list[int] = []
    scores: list[float] = []
    class_ids: list[int] = []
    has_score = False
    has_class = False

    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must have header columns {sorted(required)}")
        has_score = "score" in reader.fieldnames
        has_class = "class_id" in reader.fieldnames
        for row in reader:
            frames.append(int(float(row["frame"])))
            track_ids.append(int(float(row["track_id"])))
            boxes.append([float(row[c]) for c in ("x1", "y1", "x2", "y2")])
            if has_score:
                cell = row.get("score")
                scores.append(float(cell) if cell else float("nan"))
            if has_class:
                cell = row.get("class_id")
                class_ids.append(int(float(cell)) if cell else MISSING_CLASS_ID)

    return from_generic(
        frames=frames,
        boxes=boxes,
        track_ids=track_ids,
        fps=fps,
        scores=scores if has_score else None,
        class_ids=class_ids if has_class else None,
        image_size=image_size,
        skeleton=skeleton,
    )


__all__ = ["MISSING_CLASS_ID", "from_csv", "from_generic"]
