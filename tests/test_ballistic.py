"""Tests for free-flight segment detection + gravity-as-a-ruler fit (BRIEF.md §12.2, §10).

All synthetic data is built INSIDE this module (no bench / cross-group imports). The core
fixture is a projected parabola: a free-flying object under constant gravity, viewed by a
static, horizontal camera with a known constant meters-per-pixel scale, so the world
vertical maps to image-``y`` and the true scale is known exactly. This lets us assert
*absolute* metric error (the thing real uninstrumented video cannot give us).

Coverage:
* (a) detection finds the arc;
* (b) the fit recovers metric scale and speed within tolerance at ``Tier.METRIC``;
* (c) a non-ballistic (linear) track does NOT emit METRIC — honest fallback;
* (d) a supplied ``reference_scale`` yields METRIC from the supplied scale;
* (e) RANSAC/IRLS reject a few injected outlier points.
"""

from __future__ import annotations

import numpy as np

from trackphysics.core.ballistic import (
    QuadraticFit,
    detect_ballistic_segments,
    fit_ballistic,
    irls_quadratic,
    ransac_quadratic,
)
from trackphysics.core.grounding import GroundingContext
from trackphysics.core.provenance import Tier
from trackphysics.core.schema import Detection, Segment, TrackSequence

# --------------------------------------------------------------------------------------
# Synthetic ground truth.
# --------------------------------------------------------------------------------------

G = 9.81  # m/s^2
FPS = 120.0
S_TRUE = 0.004  # meters per pixel (true scale we will try to recover)
BOX = 12.0  # bbox half-extent in px (constant; irrelevant to center motion)


def _parabola_track(
    *,
    n: int = 30,
    vx_world: float = 8.0,
    vy_world: float = 12.0,
    x0_px: float = 100.0,
    y0_px: float = 600.0,
    jitter_px: float = 0.0,
    seed: int = 0,
    fps: float = FPS,
) -> tuple[TrackSequence, float]:
    """Build a projected-parabola TrackSequence and return it with the true world speed.

    World motion: ``x(t)=vx*t``, ``y(t)=vy*t - 0.5 g t^2`` (meters). Projection: pixels =
    meters / S_TRUE, with image-``y`` pointing DOWN, so a rising object moves to smaller
    pixel-``y``. The vertical pixel acceleration is therefore ``+g / S_TRUE`` (downward),
    which the fit converts back to scale ``g / |a_px| = S_TRUE``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / fps
    x_world = vx_world * t
    y_world = vy_world * t - 0.5 * G * t * t
    # image-y is down: subtract world-up displacement from a baseline pixel row.
    x_px = x0_px + x_world / S_TRUE
    y_px = y0_px - y_world / S_TRUE
    if jitter_px > 0.0:
        x_px = x_px + rng.normal(0.0, jitter_px, size=n)
        y_px = y_px + rng.normal(0.0, jitter_px, size=n)

    dets: list[Detection] = []
    for f in range(n):
        cx, cy = x_px[f], y_px[f]
        bbox = np.array([cx - BOX, cy - BOX, cx + BOX, cy + BOX], dtype=np.float64)
        dets.append(Detection(frame=f, bbox=bbox, track_id=1))
    track = TrackSequence(detections=dets, fps=fps, image_size=(1920, 1080))
    true_speed = float(np.hypot(vx_world, vy_world))
    return track, true_speed


def _linear_track(n: int = 30, vx_px: float = 5.0, vy_px: float = 4.0) -> TrackSequence:
    """Constant-velocity (zero-acceleration) pixel track: must NOT be promoted to metric."""
    dets: list[Detection] = []
    for f in range(n):
        cx = 100.0 + vx_px * f
        cy = 200.0 + vy_px * f
        bbox = np.array([cx - BOX, cy - BOX, cx + BOX, cy + BOX], dtype=np.float64)
        dets.append(Detection(frame=f, bbox=bbox, track_id=2))
    return TrackSequence(detections=dets, fps=FPS, image_size=(1920, 1080))


def _full_segment(track: TrackSequence) -> Segment:
    idx = np.arange(len(track), dtype=np.int64)
    frames = track.frames
    return Segment(
        start_frame=int(frames[0]), end_frame=int(frames[-1]),
        kind="ballistic", indices=idx,
    )


# --------------------------------------------------------------------------------------
# (a) Detection.
# --------------------------------------------------------------------------------------


def test_detect_finds_ballistic_arc() -> None:
    track, _ = _parabola_track(n=30)
    segments = detect_ballistic_segments(track)
    assert segments, "expected at least one ballistic segment"
    seg = max(segments, key=lambda s: 0 if s.indices is None else s.indices.size)
    assert seg.kind == "ballistic"
    assert seg.indices is not None
    # The detected arc should cover the bulk of the 30-frame parabola.
    assert seg.indices.size >= 20


def test_detect_returns_empty_on_short_track() -> None:
    track, _ = _parabola_track(n=4)
    assert detect_ballistic_segments(track) == []


def test_detect_tolerates_mild_jitter() -> None:
    # At 60 fps the per-step curvature signal (~0.68 px) comfortably exceeds 0.15 px
    # jitter; the detector must still register the arc. (At very high fps the curvature
    # signal shrinks below the jitter and detection legitimately becomes impossible —
    # that is physics, not a detector defect, so we test in a fair SNR regime.)
    track, _ = _parabola_track(n=40, jitter_px=0.15, seed=3, fps=60.0)
    segments = detect_ballistic_segments(track)
    assert segments, "jittered parabola should still register as ballistic"


# --------------------------------------------------------------------------------------
# (b) Metric scale + speed recovery via gravity-as-a-ruler.
# --------------------------------------------------------------------------------------


def test_fit_recovers_metric_scale_and_speed() -> None:
    track, true_speed = _parabola_track(n=30)
    seg = _full_segment(track)
    ctx = GroundingContext(gravity=G)
    est = fit_ballistic(track, seg, ctx, rng=np.random.default_rng(0))

    assert est.tier is Tier.METRIC
    assert est.positions.tier is Tier.METRIC
    assert est.velocity.unit == "m/s"
    assert est.velocity.source == "ballistic_fit"

    recovered_scale = float(est.meta["scale_m_per_px"])  # type: ignore[arg-type]
    assert recovered_scale == np.float64(recovered_scale)
    assert abs(recovered_scale - S_TRUE) / S_TRUE < 0.03

    # Initial recovered speed (first few samples) should match the true launch speed.
    vel = np.asarray(est.velocity.value, dtype=np.float64)
    assert vel.shape == (len(track), 3)
    initial_speed = float(np.hypot(vel[1, 0], vel[1, 1]))
    assert abs(initial_speed - true_speed) / true_speed < 0.05

    # Depth axis is honestly zeroed at the monocular tier.
    pos = np.asarray(est.positions.value, dtype=np.float64)
    assert np.allclose(pos[:, 2], 0.0)
    assert 0.0 <= est.goodness_of_fit <= 1.0
    assert 0.0 < est.positions.confidence <= 1.0


def test_metric_meta_carries_launch_speed_and_ci() -> None:
    track, true_speed = _parabola_track(n=30)
    seg = _full_segment(track)
    est = fit_ballistic(track, seg, GroundingContext(gravity=G), rng=np.random.default_rng(0))
    assert est.tier is Tier.METRIC
    speed = float(est.meta["launch_speed_m_s"])  # type: ignore[arg-type]
    ci = est.meta["launch_speed_ci95"]
    assert ci is not None
    lo, hi = ci  # type: ignore[misc]
    assert lo <= speed <= hi  # the point estimate lies within its own CI
    assert np.isfinite(lo) and np.isfinite(hi)
    # On a clean, exact parabola the recovered launch speed matches truth closely.
    assert abs(speed - true_speed) / true_speed < 0.05


def test_fit_metric_high_confidence_on_clean_arc() -> None:
    track, _ = _parabola_track(n=30)
    seg = _full_segment(track)
    est = fit_ballistic(track, seg, GroundingContext(gravity=G), rng=np.random.default_rng(1))
    assert est.tier is Tier.METRIC
    # A clean, dense, low-residual arc should earn a healthy confidence.
    assert est.positions.confidence > 0.6
    assert est.goodness_of_fit > 0.6


# --------------------------------------------------------------------------------------
# (c) Honesty: non-ballistic track must NOT be promoted to METRIC.
# --------------------------------------------------------------------------------------


def test_linear_track_does_not_emit_metric() -> None:
    track = _linear_track(n=30)
    seg = _full_segment(track)
    est = fit_ballistic(track, seg, GroundingContext(gravity=G), rng=np.random.default_rng(0))
    assert est.tier is not Tier.METRIC
    assert est.tier is Tier.RELATIVE
    assert est.positions.tier is Tier.RELATIVE
    assert est.positions.unit is None
    assert est.meta.get("fallback_reason") == "sanity_gate_failed"


def test_linear_track_not_detected_as_ballistic() -> None:
    track = _linear_track(n=30)
    # Zero curvature -> no constant non-zero vertical acceleration -> no ballistic run.
    assert detect_ballistic_segments(track) == []


def test_too_short_segment_falls_back() -> None:
    track, _ = _parabola_track(n=30)
    seg = Segment(start_frame=0, end_frame=1, kind="ballistic",
                  indices=np.array([0, 1], dtype=np.int64))
    est = fit_ballistic(track, seg, GroundingContext(gravity=G))
    assert est.tier is Tier.RELATIVE
    assert est.meta.get("fallback_reason") == "too_few_points"


# --------------------------------------------------------------------------------------
# (d) Supplied reference scale -> METRIC from the supplied scale.
# --------------------------------------------------------------------------------------


def test_reference_scale_yields_metric_from_supplied_scale() -> None:
    track, _ = _parabola_track(n=30)
    seg = _full_segment(track)
    supplied = 0.01  # deliberately different from S_TRUE so we can detect which won
    ctx = GroundingContext(reference_scale=supplied, gravity=G)
    est = fit_ballistic(track, seg, ctx, rng=np.random.default_rng(0))

    assert est.tier is Tier.METRIC
    assert est.positions.source == "reference_scale"
    assert est.velocity.source == "reference_scale"
    assert abs(float(est.meta["scale_m_per_px"]) - supplied) < 1e-12  # type: ignore[arg-type]


def test_reference_scale_works_even_on_linear_track() -> None:
    # When the caller GIVES scale, we honor it regardless of motion signature.
    track = _linear_track(n=30)
    seg = _full_segment(track)
    ctx = GroundingContext(reference_scale=0.02, gravity=G)
    est = fit_ballistic(track, seg, ctx)
    assert est.tier is Tier.METRIC
    assert est.positions.source == "reference_scale"


# --------------------------------------------------------------------------------------
# (e) RANSAC / IRLS reject injected outliers.
# --------------------------------------------------------------------------------------


def test_ransac_rejects_outliers() -> None:
    t = np.arange(25, dtype=np.float64) / FPS
    y = 600.0 - (12.0 * t - 0.5 * G * t * t) / S_TRUE
    y_corrupt = y.copy()
    # Inject a few gross outliers far off the parabola.
    for k in (5, 12, 19):
        y_corrupt[k] += 400.0
    fit = ransac_quadratic(t, y_corrupt, rng=np.random.default_rng(7))
    assert isinstance(fit, QuadraticFit)
    # Outlier rows should be flagged as non-inliers.
    for k in (5, 12, 19):
        assert fit.inlier_mask[k] < 0.5
    # And the recovered acceleration should match the clean parabola closely.
    true_accel = G / S_TRUE
    assert abs(fit.accel - true_accel) / true_accel < 0.05


def test_irls_refines_toward_clean_acceleration() -> None:
    rng = np.random.default_rng(11)
    t = np.arange(30, dtype=np.float64) / FPS
    y = 600.0 - (12.0 * t - 0.5 * G * t * t) / S_TRUE
    y_noisy = y + rng.normal(0.0, 0.8, size=t.size)
    # A couple of hard outliers IRLS should down-weight.
    y_noisy[8] += 300.0
    y_noisy[21] -= 300.0
    fit = irls_quadratic(t, y_noisy, rng=rng)
    true_accel = G / S_TRUE
    assert abs(fit.accel - true_accel) / true_accel < 0.08
    assert fit.inlier_mask[8] < 0.5
    assert fit.inlier_mask[21] < 0.5


def _antigravity_track(n: int = 30, fps: float = FPS) -> TrackSequence:
    """A clean parabola that accelerates UPWARD in the image (a_px < 0).

    Physically anti-gravitational under the constant-depth assumption: it is NOT free fall,
    so gravity must not be used as its ruler. Must fall back to RELATIVE (PROV-01).
    """
    t = np.arange(n, dtype=np.float64) / fps
    x_px = 100.0 + (8.0 / S_TRUE) * t
    # Concave-down in image-y => negative vertical pixel acceleration (upward).
    y_px = 600.0 + (8.0 / S_TRUE) * t - 0.5 * (G / S_TRUE) * t * t
    dets: list[Detection] = []
    for f in range(n):
        cx, cy = x_px[f], y_px[f]
        bbox = np.array([cx - BOX, cy - BOX, cx + BOX, cy + BOX], dtype=np.float64)
        dets.append(Detection(frame=f, bbox=bbox, track_id=3))
    return TrackSequence(detections=dets, fps=fps, image_size=(1920, 1080))


def test_antigravity_arc_not_promoted_to_metric() -> None:
    track = _antigravity_track()
    seg = _full_segment(track)
    est = fit_ballistic(track, seg, GroundingContext(gravity=G), rng=np.random.default_rng(0))
    assert est.tier is Tier.RELATIVE  # the cardinal §10 invariant
    assert est.meta.get("fallback_reason") == "sanity_gate_failed"


def test_antigravity_arc_not_detected_as_ballistic() -> None:
    track = _antigravity_track()
    assert detect_ballistic_segments(track) == []


def test_metric_trajectory_is_robust_to_outliers() -> None:
    # BAL-RAWPOS: the emitted METRIC velocity must come from the robust fit, so gross
    # outlier frames produce NO spurious velocity spike.
    track, true_speed = _parabola_track(n=32)
    for k in (6, 15, 24):
        cx, cy = track.detections[k].center
        ncx, ncy = cx + 250.0, cy - 180.0
        track.detections[k].bbox = np.array(
            [ncx - BOX, ncy - BOX, ncx + BOX, ncy + BOX], dtype=np.float64
        )
    seg = _full_segment(track)
    est = fit_ballistic(track, seg, GroundingContext(gravity=G), rng=np.random.default_rng(2))
    assert est.tier is Tier.METRIC
    vel = np.asarray(est.velocity.value, dtype=np.float64)
    speeds = np.hypot(vel[:, 0], vel[:, 1])
    # A finite-difference of raw centers would spike to ~100+ m/s at the corrupted frames;
    # the robust-fit trajectory stays physically bounded near the true launch speed.
    assert float(speeds.max()) < 1.5 * true_speed


def test_fit_with_outliers_still_recovers_metric() -> None:
    track, true_speed = _parabola_track(n=32)
    # Corrupt a few detections' centers with gross pixel jumps.
    for k in (6, 15, 24):
        cx, cy = track.detections[k].center
        cx += 250.0
        cy -= 180.0
        track.detections[k].bbox = np.array(
            [cx - BOX, cy - BOX, cx + BOX, cy + BOX], dtype=np.float64
        )
    seg = _full_segment(track)
    est = fit_ballistic(track, seg, GroundingContext(gravity=G), rng=np.random.default_rng(2))
    assert est.tier is Tier.METRIC
    recovered_scale = float(est.meta["scale_m_per_px"])  # type: ignore[arg-type]
    assert abs(recovered_scale - S_TRUE) / S_TRUE < 0.05
