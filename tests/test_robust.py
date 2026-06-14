"""Tests for the robustness machinery (BRIEF.md §11).

All fixtures are built inline (no bench / cross-group imports). Each behaviour is tested
on both its success path and its honesty path (correct refusal / fallback / flagging).
"""

from __future__ import annotations

import numpy as np

from trackphysics.core.robust.imputation import impute_gaps
from trackphysics.core.robust.quality import assess_quality
from trackphysics.core.robust.smoothing import SmoothingResult, robust_smooth
from trackphysics.core.schema import Detection, TrackSequence


def _ar1_noise(n: int, rho: float, sigma: float, seed: int) -> np.ndarray:
    """Generate AR(1) (correlated, non-IID) noise of length ``n``."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n, dtype=np.float64)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = rho * out[i - 1] + eps[i]
    return out


def _track_from_centers(
    centers: np.ndarray, frames: np.ndarray, fps: float = 100.0
) -> TrackSequence:
    """Build a TrackSequence whose bbox centroids equal the given centers."""
    dets: list[Detection] = []
    for f, (cx, cy) in zip(frames.tolist(), centers.tolist(), strict=True):
        bbox = np.array([cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0], dtype=np.float64)
        dets.append(Detection(frame=int(f), bbox=bbox, track_id=1))
    return TrackSequence(detections=dets, fps=fps)


# --------------------------------------------------------------------------------------
# (a) robust_smooth: reduces RMSE on an AR(1)-jittered ramp; mask excludes gross outliers
# --------------------------------------------------------------------------------------


def test_robust_smooth_reduces_rmse_on_ar1_ramp() -> None:
    n = 60
    t = np.linspace(0.0, 1.0, n)
    truth = 5.0 + 30.0 * t  # a clean ramp
    noisy = truth + _ar1_noise(n, rho=0.7, sigma=0.8, seed=0)

    res = robust_smooth(noisy, t, degree=1)
    assert isinstance(res, SmoothingResult)

    rmse_raw = float(np.sqrt(np.mean((noisy - truth) ** 2)))
    rmse_smooth = float(np.sqrt(np.mean((res.smoothed - truth) ** 2)))
    assert rmse_smooth < rmse_raw
    assert res.smoothed.shape == noisy.shape


def test_robust_smooth_inlier_mask_excludes_gross_outliers() -> None:
    n = 50
    t = np.linspace(0.0, 1.0, n)
    truth = 2.0 + 10.0 * t + 4.0 * t**2
    noisy = truth + _ar1_noise(n, rho=0.5, sigma=0.3, seed=1)

    # Inject gross outliers at known indices.
    outlier_idx = [10, 25, 40]
    for i in outlier_idx:
        noisy[i] += 50.0

    res = robust_smooth(noisy, t, degree=2)
    for i in outlier_idx:
        assert not res.inlier_mask[i], f"outlier at {i} should be rejected"
    # The vast majority of clean points are kept (Tukey may trim a few borderline
    # AR(1) samples, which is acceptable; the key property is rejecting the gross ones).
    assert res.inlier_mask.sum() >= int(0.8 * n)


def test_robust_smooth_handles_degenerate_short_input() -> None:
    # Fewer points than degree+1 must not raise.
    t = np.array([0.0, 0.1])
    y = np.array([1.0, 2.0])
    res = robust_smooth(y, t, degree=2)
    assert res.smoothed.shape == (2,)
    assert res.inlier_mask.all()


# --------------------------------------------------------------------------------------
# (b) impute_gaps: fills a coherent gap (stitched=True), refuses an incoherent jump
# --------------------------------------------------------------------------------------


def test_impute_gaps_fills_coherent_gap() -> None:
    # Smooth linear motion with a clean gap in the middle.
    frames = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12], dtype=np.float64)
    # value follows v(f) = 2*f along a straight line -> extrapolation is exact.
    vals = (2.0 * frames).astype(np.float64)
    res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=15)

    assert len(res.gaps) == 1
    gap = res.gaps[0]
    assert gap.start_frame == 4 and gap.end_frame == 9
    assert gap.length == 4
    assert gap.stitched is True
    assert gap.coherence > 0.5

    # Filled values exist (no nan) across the densified range.
    assert not np.any(np.isnan(res.values))
    # Frame 6 (index 6) should be filled near the true value 12.0.
    filled = res.values[np.searchsorted(res.frames, 6.0)]
    assert abs(float(filled) - 12.0) < 1.0


def test_impute_gaps_fills_curved_arc_along_the_curve_not_a_chord() -> None:
    # A stitched gap over an ACCELERATING (parabolic) arc must be filled along the validated
    # curve, not a straight chord (ROB-FILL-CHORD). For a pure quadratic the curvature-
    # preserving fill is exact, whereas a chord across the gap deviates by the sagitta.
    frames = np.array([0, 1, 2, 3, 4, 5, 11, 12, 13, 14], dtype=np.float64)  # gap 6..10
    accel = 2.0
    vals = 0.5 * accel * frames**2
    res = impute_gaps(frames, vals, coherence_tol=5.0, max_gap=15, n_post=3)
    assert res.gaps[0].stitched is True

    gap_mask = ~res.observed_mask
    truth = 0.5 * accel * res.frames**2
    curved_err = float(np.max(np.abs(res.values[gap_mask] - truth[gap_mask])))
    chord = np.interp(res.frames, [5.0, 11.0], [0.5 * accel * 25, 0.5 * accel * 121])
    chord_err = float(np.max(np.abs(chord[gap_mask] - truth[gap_mask])))
    assert curved_err < 1e-6  # exact for a quadratic
    assert chord_err > 5.0  # the chord is badly biased here
    assert curved_err < chord_err


def test_impute_gaps_refuses_incoherent_jump() -> None:
    # Linear motion before the gap, then a large teleport after it (id-switch-like).
    pre_frames = np.arange(0, 6, dtype=np.float64)
    pre_vals = 2.0 * pre_frames
    post_frames = np.arange(10, 14, dtype=np.float64)
    # Post-gap values are wildly off the predicted line (would be ~20..26); make them 200+.
    post_vals = 200.0 + 2.0 * post_frames
    frames = np.concatenate([pre_frames, post_frames])
    vals = np.concatenate([pre_vals, post_vals])

    res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=15)
    assert len(res.gaps) == 1
    gap = res.gaps[0]
    assert gap.stitched is False
    assert gap.coherence < 0.5
    # The refused gap must be left as nan (never silently bridged).
    inside = (res.frames > 5.0) & (res.frames < 10.0)
    assert np.all(np.isnan(res.values[inside]))


def test_impute_gaps_refuses_too_long_gap() -> None:
    frames = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0], dtype=np.float64)
    vals = 2.0 * frames
    res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=10)
    assert len(res.gaps) == 1
    assert res.gaps[0].stitched is False  # 96 missing > max_gap


def test_impute_gaps_no_gaps_passthrough() -> None:
    frames = np.arange(0, 6, dtype=np.float64)
    vals = np.column_stack([frames, 3.0 * frames])  # (T, 2)
    res = impute_gaps(frames, vals)
    assert res.gaps == []
    assert res.values.shape == (6, 2)
    assert res.observed_mask.all()


# --------------------------------------------------------------------------------------
# (c) assess_quality: flags an injected gap and a jitter burst; completeness < 1
# --------------------------------------------------------------------------------------


def test_assess_quality_flags_gap_and_completeness() -> None:
    # Frames 0..4 then 12..16: a 7-frame gap; missing frames -> completeness < 1.
    frames = np.concatenate([np.arange(0, 5), np.arange(12, 17)]).astype(np.float64)
    centers = np.column_stack([2.0 * frames, 1.0 * frames])
    track = _track_from_centers(centers, frames)

    report = assess_quality(track)
    reasons = {f.reason for f in report.flags}
    assert "gap" in reasons
    assert report.completeness < 1.0
    assert report.overall_score < 1.0
    # The gap flag spans the missing region.
    gap_flag = next(f for f in report.flags if f.reason == "gap")
    assert gap_flag.start_frame == 4 and gap_flag.end_frame == 12


def test_assess_quality_flags_jitter_burst() -> None:
    n = 60
    frames = np.arange(n, dtype=np.float64)
    # Smooth quadratic base motion.
    base_x = 1.0 + 3.0 * frames + 0.05 * frames**2
    base_y = 2.0 + 1.0 * frames
    rng = np.random.default_rng(7)
    # Mild background noise everywhere.
    x = base_x + rng.normal(0.0, 0.05, size=n)
    y = base_y + rng.normal(0.0, 0.05, size=n)
    # Inject a jitter burst in a contiguous window.
    burst = slice(25, 33)
    x[burst] += rng.normal(0.0, 8.0, size=8)
    y[burst] += rng.normal(0.0, 8.0, size=8)
    centers = np.column_stack([x, y])
    track = _track_from_centers(centers, frames)

    report = assess_quality(track)
    reasons = {f.reason for f in report.flags}
    assert "jitter" in reasons
    # At least one jitter flag overlaps the injected burst window [25, 32].
    jitter_flags = [f for f in report.flags if f.reason == "jitter"]
    assert any(f.start_frame <= 32 and f.end_frame >= 25 for f in jitter_flags)


def test_assess_quality_clean_track_scores_high() -> None:
    n = 40
    frames = np.arange(n, dtype=np.float64)
    centers = np.column_stack([3.0 * frames, 2.0 * frames])
    track = _track_from_centers(centers, frames)

    report = assess_quality(track)
    assert report.completeness == 1.0
    assert report.overall_score > 0.9
    assert all(f.reason != "gap" for f in report.flags)


def test_assess_quality_empty_track() -> None:
    track = TrackSequence(detections=[], fps=60.0)
    report = assess_quality(track)
    assert report.completeness == 0.0
    assert report.overall_score == 0.0
    assert report.flags == []
