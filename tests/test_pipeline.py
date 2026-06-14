"""End-to-end pipeline tests: analyze() on synthetic arcs, exercising the full stack.

These are the integration consumers of the whole engine (BRIEF.md §16 "no feature
without a consumer"): generator -> adapter-free TrackSequence -> analyze -> provenance.
They assert the two things that matter most: recovered metric is accurate on a clean arc,
and the engine never fabricates metric when scale is unrecoverable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import trackphysics as tp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import generate_track, look_at_camera  # noqa: E402


def _clean_arc(fps: float = 120.0) -> tuple[tp.TrackSequence, float]:
    cam = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7))
    track, gt = generate_track(
        cam, fps=fps, launch_velocity=(6.0, 0.0, 7.0), drag_coeff=1e-9, duration=1.2
    )
    return track, float(gt.speed[0])


def test_clean_arc_recovers_metric_speed() -> None:
    track, true_speed = _clean_arc()
    res = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext())
    est = res.trajectory
    assert est.tier == tp.Tier.METRIC
    assert est.velocity.unit == "m/s"
    speed = float(np.linalg.norm(np.asarray(est.velocity.value)[0]))
    assert abs(speed - true_speed) / true_speed < 0.15  # within 15% of truth


def test_non_ballistic_does_not_fabricate_metric() -> None:
    # Near-constant-velocity motion: no usable free-flight signature.
    cam = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7))
    track, _gt = generate_track(
        cam, fps=120.0, launch_velocity=(5.0, 4.0, 0.2), gravity=1e-6, drag_coeff=1e-9,
        duration=0.8,
    )
    res = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext())
    assert res.trajectory.tier != tp.Tier.METRIC  # honest fallback


def test_reference_scale_yields_metric_even_without_arc() -> None:
    track, _ = _clean_arc()
    grounded = tp.GroundingContext(reference_scale=0.01)
    res = tp.analyze(track, preset="sphere", grounding=grounded)
    tiers = [t.tier for t in res.trajectories]
    assert tp.Tier.METRIC in tiers


def _truth_at_segment_start(est: tp.TrajectoryEstimate, gt: object) -> float:
    """True speed at the engine's reported (segment-start) frame — the consistent instant."""
    frame = est.velocity.frame
    assert isinstance(frame, tuple)
    frames = np.asarray(gt.frames, dtype=np.int64)  # type: ignore[attr-defined]
    idx = int(np.where(frames == int(frame[0]))[0][0])
    return float(np.asarray(gt.speed)[idx])  # type: ignore[attr-defined]


def test_cooperative_camera_ci_covers_truth() -> None:
    # Level camera, realistic drag: the engine's CI (measurement + v0.1 systematic floor)
    # COVERS the true speed at the instant it reports -> the harness's calibrated-PASS case.
    cam = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7))
    track, gt = generate_track(
        cam, fps=120.0, launch_velocity=(6.0, 0.0, 7.0), drag_coeff=0.2, duration=1.2
    )
    est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
    assert est.tier == tp.Tier.METRIC
    true_speed = _truth_at_segment_start(est, gt)  # compared at the SAME frame
    lo, hi = est.meta["launch_speed_ci95"]  # type: ignore[misc]
    assert lo <= true_speed <= hi


def test_steep_camera_ci_is_overconfident() -> None:
    # A STEEPLY pitched camera grossly violates gravity-as-a-ruler yet still fits a clean
    # parabola: the engine emits METRIC with a tight CI that does NOT cover the true speed at
    # the reported instant -> the genuine overconfident-FAIL case (a mild tilt, by contrast,
    # the engine tolerates and would correctly PASS).
    cam = look_at_camera(eye=(3.0, -4.0, 7.0), target=(3.0, 0.0, 0.5))
    track, gt = generate_track(
        cam, fps=120.0, launch_velocity=(6.0, 0.0, 7.0), drag_coeff=0.2, duration=1.2
    )
    est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
    assert est.tier == tp.Tier.METRIC
    true_speed = _truth_at_segment_start(est, gt)
    lo, hi = est.meta["launch_speed_ci95"]  # type: ignore[misc]
    assert not (lo <= true_speed <= hi)  # overconfident: tight CI misses the true speed


def test_relative_fallback_has_no_metric_ci() -> None:
    cam = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7))
    track, _gt = generate_track(
        cam, fps=120.0, launch_velocity=(5.0, 4.0, 0.2), gravity=1e-6, drag_coeff=1e-9,
        duration=0.8,
    )
    est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
    assert est.tier != tp.Tier.METRIC
    assert est.meta.get("launch_speed_ci95") is None  # no metric claim -> no CI


def test_generic_path_has_relative_floor_and_quality() -> None:
    track, _ = _clean_arc()
    res = tp.analyze(track)  # no preset
    assert any(t.tier == tp.Tier.RELATIVE for t in res.trajectories)
    assert 0.0 <= res.quality.overall_score <= 1.0
    assert res.meta["preset"] is None
