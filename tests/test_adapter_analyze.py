"""Tests for the validation integration point (validation/adapter_analyze.py).

``adapter_analyze`` is the documented single seam between the DIY ruler harness and the real
engine: it reads a ``track.json`` (frames + per-frame centroids, ``null`` = gap) and returns
an :class:`EngineResult` (tier, speed, 95% CI, ``at_frame``). It has no other test coverage,
so a regression in the gap handling, the metric/non-metric branching, or the segment-start
``at_frame`` wiring would pass CI silently. These tests build ``track.json`` dicts from a
known synthetic arc and assert the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import generate_track, look_at_camera  # noqa: E402
from validation.adapter_analyze import adapter_analyze  # noqa: E402

from trackphysics.core.schema import TrackSequence  # noqa: E402


def _track_json(track: TrackSequence, *, drop: set[int] | None = None) -> dict[str, object]:
    """Render a TrackSequence as a track.json dict; ``drop`` nulls those frame indices (gaps)."""
    drop = drop or set()
    boxes = np.array([d.bbox for d in track.detections], dtype=np.float64)
    centroids: list[list[float] | None] = []
    frames: list[int] = []
    for det, box in zip(track.detections, boxes, strict=True):
        frames.append(int(det.frame))
        if int(det.frame) in drop:
            centroids.append(None)
        else:
            centroids.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return {
        "fps": track.fps,
        "image_size": list(track.image_size) if track.image_size else None,
        "frames": frames,
        "centroids": centroids,
    }


def _cooperative_track() -> TrackSequence:
    cam = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7), image_size=(1280, 720))
    track, _ = generate_track(
        cam,
        fps=120.0,
        launch_position=(3.0, 0.0, 0.0),
        launch_velocity=(6.0, 0.0, 7.0),
        drag_coeff=0.2,
        duration=1.2,
    )
    return track


def test_cooperative_track_yields_metric_with_ci_and_at_frame() -> None:
    res = adapter_analyze(_track_json(_cooperative_track()))
    assert res.tier == "metric"
    assert res.speed_m_s is not None and res.speed_m_s > 0
    assert res.ci95 is not None
    lo, hi = res.ci95
    assert lo <= res.speed_m_s <= hi
    # The speed pertains to the detected segment-start frame, exposed for coverage tests.
    assert res.at_frame is not None
    assert 0.0 <= res.confidence <= 1.0
    assert res.source


def test_gap_centroids_are_dropped_but_metric_survives() -> None:
    # Nulling a short run of frames must not break the metric road: the engine accounts for
    # the missing frames itself (the surviving observations still pin the arc).
    track = _cooperative_track()
    mid = int(track.frames[len(track.detections) // 2])
    res = adapter_analyze(_track_json(track, drop={mid, mid + 1, mid + 2}))
    assert res.tier == "metric"
    assert res.speed_m_s is not None


def test_too_few_observations_returns_insufficient_pixel() -> None:
    track = _cooperative_track()
    keep = {int(track.frames[0]), int(track.frames[1])}  # only 2 non-null centroids
    drop = {int(f) for f in track.frames} - keep
    res = adapter_analyze(_track_json(track, drop=drop))
    assert res.tier == "pixel"
    assert res.source == "insufficient_track"
    assert res.speed_m_s is None and res.ci95 is None


def test_missing_image_size_is_accepted() -> None:
    payload = _track_json(_cooperative_track())
    payload["image_size"] = None
    res = adapter_analyze(payload)
    assert res.tier in {"metric", "relative", "pixel"}
