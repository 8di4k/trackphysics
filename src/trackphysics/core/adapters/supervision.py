"""Adapter for `supervision <https://supervision.roboflow.com>`_ tracker output.

Deliberately **duck-typed**: the core never imports ``supervision`` (it lives behind the
``[supervision]`` extra). This adapter reads the attributes a tracked ``sv.Detections``
exposes (``xyxy``, ``tracker_id``, optionally ``confidence`` / ``class_id``), so if the
upstream type changes we change only this one file (BRIEF.md §4).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..schema import SkeletonGraph, TrackSequence
from .generic import MISSING_CLASS_ID, from_generic


def from_supervision(
    detections_iter: Iterable[Any],
    fps: float,
    *,
    image_size: tuple[int, int] | None = None,
    skeleton: SkeletonGraph | None = None,
) -> list[TrackSequence]:
    """Convert per-frame ``supervision`` detections into one track per id.

    Args:
        detections_iter: An iterable of per-frame items. Each item is either an
            ``sv.Detections``-like object (frame index taken from enumeration order) or a
            ``(frame_index, detections)`` tuple. The detections object must expose
            ``xyxy`` ``(M, 4)`` and ``tracker_id`` ``(M,)``; ``confidence`` and
            ``class_id`` are used when present.
        fps: Frames per second.
        image_size: Optional ``(W, H)`` in pixels.
        skeleton: Optional skeleton graph shared by all tracks.

    Returns:
        One :class:`TrackSequence` per distinct ``tracker_id``. Detections whose
        ``tracker_id`` is ``None`` (untracked) are skipped.

    When some frames carry ``confidence`` / ``class_id`` and others do not, the columns
    are built in lockstep with the kept detections so their length always matches: a
    frame missing ``confidence`` contributes ``NaN`` (mapped to ``score=None``) and a
    frame missing ``class_id`` contributes the sentinel (mapped to ``class_id=None``).
    The column is passed through whenever **any** frame provided it, so mixed
    availability no longer drops the present values (BRIEF.md §10).
    """
    frames: list[int] = []
    boxes: list[list[float]] = []
    track_ids: list[int] = []
    scores: list[float] = []
    class_ids: list[int] = []
    any_score = False
    any_class = False

    for enum_idx, item in enumerate(detections_iter):
        if isinstance(item, tuple) and len(item) == 2:
            frame_idx, dets = int(item[0]), item[1]
        else:
            frame_idx, dets = enum_idx, item

        xyxy = np.asarray(dets.xyxy, dtype=np.float64)
        tracker_id = getattr(dets, "tracker_id", None)
        if tracker_id is None:
            if xyxy.shape[0] == 0:
                continue  # empty/untracked-empty frame: skip it, per the docstring
            raise ValueError(
                "supervision Detections lack tracker_id; run a tracker (e.g. ByteTrack) first"
            )
        tracker_id = np.asarray(tracker_id)
        confidence = getattr(dets, "confidence", None)
        class_id = getattr(dets, "class_id", None)

        any_score = any_score or confidence is not None
        any_class = any_class or class_id is not None

        for j in range(xyxy.shape[0]):
            tid = tracker_id[j]
            if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                continue
            frames.append(frame_idx)
            boxes.append([float(v) for v in xyxy[j]])
            track_ids.append(int(tid))
            # Build columns in lockstep with the kept detections: a frame missing the
            # column contributes NaN / the sentinel so length always matches and
            # from_generic maps it back to None per element.
            scores.append(float(confidence[j]) if confidence is not None else float("nan"))
            class_ids.append(
                int(class_id[j]) if class_id is not None else MISSING_CLASS_ID
            )

    if not frames:
        return []

    return from_generic(
        frames=frames,
        boxes=boxes,
        track_ids=track_ids,
        fps=fps,
        scores=scores if any_score else None,
        class_ids=class_ids if any_class else None,
        image_size=image_size,
        skeleton=skeleton,
    )


__all__ = ["from_supervision"]
