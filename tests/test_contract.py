"""Stage B contract smoke tests: the public surface imports and its invariants hold."""

from __future__ import annotations

import numpy as np
import pytest

import trackphysics as tp


def test_public_surface_imports() -> None:
    for name in [
        "Detection",
        "SkeletonGraph",
        "TrackSequence",
        "Segment",
        "Tier",
        "Quantity",
        "TrajectoryEstimate",
        "Event",
        "Plane",
        "GroundingContext",
        "PhysicsPreset",
        "EventDetector",
        "AnalysisResult",
        "KinematicsResult",
        "TrackQualityReport",
        "analyze",
        "register_preset",
        "get_preset",
        "list_presets",
        "combine_confidence",
    ]:
        assert hasattr(tp, name), f"missing public symbol: {name}"


def test_detection_validates_bbox_shape() -> None:
    with pytest.raises(ValueError):
        tp.Detection(frame=0, bbox=np.array([1.0, 2.0, 3.0]), track_id=1)
    det = tp.Detection(frame=0, bbox=np.array([0.0, 0.0, 10.0, 20.0]), track_id=1)
    assert np.allclose(det.center, [5.0, 10.0])


def test_quantity_confidence_bounds() -> None:
    tp.Quantity(value=3.0, unit="m/s", tier=tp.Tier.METRIC, confidence=0.9, source="x")
    with pytest.raises(ValueError):
        tp.Quantity(value=3.0, unit="m/s", tier=tp.Tier.METRIC, confidence=1.5, source="x")


def test_tier_rank_ordering() -> None:
    assert tp.Tier.METRIC.rank > tp.Tier.RELATIVE.rank > tp.Tier.PIXEL.rank


def test_track_times_from_fps_and_frames() -> None:
    dets = [
        tp.Detection(frame=f, bbox=np.array([0.0, 0.0, 1.0, 1.0]), track_id=7)
        for f in (0, 2, 4)
    ]
    track = tp.TrackSequence(detections=dets, fps=100.0)
    assert np.allclose(track.times(), [0.0, 0.02, 0.04])
    assert track.centers().shape == (3, 2)


def test_preset_registry_roundtrip() -> None:
    from dataclasses import dataclass

    @dataclass
    class _Dummy:
        name: str = "dummy"
        diameter_m: float | None = None
        mass_kg: float | None = None
        drag_coeff: float | None = None
        magnus: bool = False

        def detect_segments(self, track: tp.TrackSequence) -> list[tp.Segment]:
            return []

        def fit(self, segment: tp.Segment, ctx: tp.GroundingContext) -> tp.TrajectoryEstimate:
            raise NotImplementedError

        def event_detectors(self) -> list[tp.EventDetector]:
            return []

    tp.register_preset(_Dummy())
    assert "dummy" in tp.list_presets()
    assert tp.get_preset("dummy").name == "dummy"
    with pytest.raises(KeyError):
        tp.get_preset("does-not-exist")


def test_combine_confidence_zero_factor_dominates() -> None:
    assert tp.combine_confidence(0.9, 0.0, 0.8) == 0.0
    assert tp.combine_confidence() == 0.0
    assert 0.0 < tp.combine_confidence(0.5, 0.5) <= 1.0


def test_grounding_metric_reference_flag() -> None:
    assert not tp.GroundingContext().has_metric_reference
    assert tp.GroundingContext(reference_scale=0.01).has_metric_reference


def test_analyze_generic_path_returns_result() -> None:
    # A short, non-ballistic track: generic analysis must still return a usable result
    # with a RELATIVE-tier floor and never fabricate METRIC.
    dets = [
        tp.Detection(frame=f, bbox=np.array([f * 1.0, 0.0, f * 1.0 + 10.0, 10.0]), track_id=1)
        for f in range(8)
    ]
    track = tp.TrackSequence(detections=dets, fps=60.0)
    res = tp.analyze(track)
    assert isinstance(res, tp.AnalysisResult)
    assert res.trajectories, "expected at least the relative-lift floor"
    assert res.trajectory.tier in (tp.Tier.RELATIVE, tp.Tier.PIXEL)
    assert res.meta["preset"] is None


def test_sphere_preset_registered() -> None:
    assert "sphere" in tp.list_presets()
    assert tp.get_preset("sphere").name == "sphere"
