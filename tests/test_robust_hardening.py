"""Hardening tests for the robustness machinery (BRIEF.md §11).

Covers three regressions in the consistency of the robustness layer:

* **ROB-002** — the coherence gate must accept a genuinely coherent *curved* (drag /
  parabolic) continuation across a gap even when the pre-gap window fits (near-)perfectly
  so the smoother residual scale collapses; it must still refuse an incoherent teleport.
* **ROB-003** — the reported ``coherence`` score must not contradict the stitch decision:
  its ``0.5`` contour coincides with the stitch threshold, so ``coherence >= 0.5`` iff the
  gap was stitched.
* **ROB-004** — a localised jitter burst against an otherwise-clean track must still be
  flagged even though the robust smoother drives its residual scale toward zero.

All fixtures are built inline (no bench / cross-group imports).
"""

from __future__ import annotations

import numpy as np

from trackphysics.core.robust.imputation import impute_gaps
from trackphysics.core.robust.quality import assess_quality
from trackphysics.core.schema import Detection, TrackSequence


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
# (1) ROB-002: coherent curved (parabolic) gap STITCHES; incoherent teleport REFUSED.
# --------------------------------------------------------------------------------------


def _parabola(frames: np.ndarray) -> np.ndarray:
    """A curved (drag-like) 2-D arc: x rises linearly, y is a downward parabola."""
    x = 4.0 + 3.0 * frames
    y = 2.0 + 6.0 * frames - 0.4 * frames**2
    return np.column_stack([x, y])


def test_coherent_curved_gap_stitches() -> None:
    # Pre-gap: a short, near-clean window so the smoother residual scale collapses toward
    # zero (this is exactly the condition that used to floor the gate at ~1e-6 and falsely
    # refuse a coherent curved continuation). Post-gap points lie ON the true parabola.
    pre_frames = np.arange(0, 6, dtype=np.float64)
    post_frames = np.arange(11, 15, dtype=np.float64)
    frames = np.concatenate([pre_frames, post_frames])

    truth = _parabola(frames)
    rng = np.random.default_rng(0)
    # Mild jitter on the pre-window only; post-gap points sit exactly on the curve so the
    # only thing that can refuse them is a broken (too-tight) tolerance.
    vals = truth.copy()
    n_pre = pre_frames.shape[0]
    vals[:n_pre] += rng.normal(0.0, 0.02, size=(n_pre, 2))

    res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=15)
    assert len(res.gaps) == 1
    gap = res.gaps[0]
    assert gap.start_frame == 5 and gap.end_frame == 11
    assert gap.stitched is True, "a coherent curved continuation must stitch"
    assert gap.coherence >= 0.5
    # No nan remains across the (now stitched) span.
    assert not np.any(np.isnan(res.values))


def test_incoherent_teleport_refused_on_curve() -> None:
    # Same curved setup, but the post-gap points teleport far off the predicted arc.
    pre_frames = np.arange(0, 6, dtype=np.float64)
    post_frames = np.arange(11, 15, dtype=np.float64)
    frames = np.concatenate([pre_frames, post_frames])

    truth = _parabola(frames)
    vals = truth.copy()
    n_pre = pre_frames.shape[0]
    # Shove the post-gap points hundreds of units away: a clear kinematic discontinuity.
    vals[n_pre:] += np.array([300.0, -250.0])

    res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=15)
    assert len(res.gaps) == 1
    gap = res.gaps[0]
    assert gap.stitched is False, "an incoherent teleport must be refused"
    assert gap.coherence < 0.5
    inside = (res.frames > 5.0) & (res.frames < 11.0)
    assert np.all(np.isnan(res.values[inside]))


# --------------------------------------------------------------------------------------
# (2) ROB-003: documented coherence/stitch relationship holds at/near the boundary.
# --------------------------------------------------------------------------------------


def test_coherence_05_contour_coincides_with_stitch_threshold() -> None:
    # Straight-line pre-window (exact fit -> collapsed residual scale, so the motion-aware
    # displacement floor governs the tolerance). We sweep the post-gap offset across the
    # stitch boundary and assert: for EVERY offset, stitched iff coherence >= 0.5, and the
    # sweep actually straddles the boundary (contains both outcomes).
    pre_frames = np.arange(0, 6, dtype=np.float64)
    post_frames = np.arange(9, 12, dtype=np.float64)
    frames = np.concatenate([pre_frames, post_frames])
    base = np.column_stack([2.0 * frames, 5.0 * frames])  # exact line, step ~ const
    n_pre = pre_frames.shape[0]

    saw_stitched = False
    saw_refused = False
    for offset in np.linspace(0.0, 3.0, 25):
        vals = base.copy()
        vals[n_pre:] += np.array([offset, 0.0])
        res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=15)
        gap = res.gaps[0]
        # The core invariant: the score never contradicts the decision.
        assert gap.stitched == (gap.coherence >= 0.5), (
            f"offset={offset}: stitched={gap.stitched} but coherence={gap.coherence}"
        )
        saw_stitched |= gap.stitched
        saw_refused |= not gap.stitched

    assert saw_stitched and saw_refused, "sweep must cross the stitch boundary"


def test_coherence_is_one_for_exact_continuation() -> None:
    # An exact-line continuation (zero deviation) maps to coherence == 1.0 under the new
    # monotone score tol/(tol+worst) at worst==0.
    frames = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13], dtype=np.float64)
    vals = (2.0 * frames).astype(np.float64)
    res = impute_gaps(frames, vals, coherence_tol=3.0, max_gap=15)
    gap = res.gaps[0]
    assert gap.stitched is True
    assert abs(gap.coherence - 1.0) < 1e-6


# --------------------------------------------------------------------------------------
# (3) ROB-004: localised jitter burst on an otherwise-clean track is still flagged.
# --------------------------------------------------------------------------------------


def test_jitter_burst_flagged_on_clean_line_track() -> None:
    # A perfectly clean line (so robust_smooth's residual scale collapses to ~0) with a
    # short jitter burst injected. The old gate skipped the whole jitter branch when
    # scale ~ 0; the fallback threshold from the median step displacement must still flag.
    n = 50
    frames = np.arange(n, dtype=np.float64)
    base_x = 3.0 * frames
    base_y = 2.0 * frames
    x = base_x.copy()
    y = base_y.copy()
    rng = np.random.default_rng(3)
    burst = slice(20, 27)
    x[burst] += rng.normal(0.0, 6.0, size=7)
    y[burst] += rng.normal(0.0, 6.0, size=7)
    centers = np.column_stack([x, y])
    track = _track_from_centers(centers, frames)

    report = assess_quality(track)
    reasons = {f.reason for f in report.flags}
    assert "jitter" in reasons, "a burst against a clean track must be flagged"
    jitter_flags = [f for f in report.flags if f.reason == "jitter"]
    assert any(f.start_frame <= 26 and f.end_frame >= 20 for f in jitter_flags)


def test_jitter_burst_flagged_on_clean_parabola_track() -> None:
    # Same idea on a clean parabola (degree-2 fit is exact -> scale ~ 0): fallback fires.
    n = 50
    frames = np.arange(n, dtype=np.float64)
    base_x = 1.0 + 2.0 * frames + 0.03 * frames**2
    base_y = 5.0 + 1.5 * frames
    x = base_x.copy()
    y = base_y.copy()
    rng = np.random.default_rng(11)
    burst = slice(30, 36)
    x[burst] += rng.normal(0.0, 7.0, size=6)
    y[burst] += rng.normal(0.0, 7.0, size=6)
    centers = np.column_stack([x, y])
    track = _track_from_centers(centers, frames)

    report = assess_quality(track)
    assert "jitter" in {f.reason for f in report.flags}


def test_truly_clean_track_has_no_jitter_flag() -> None:
    # Sanity guard for the fallback: a clean line with NO burst must NOT be flagged as
    # jitter just because the residual scale is ~0.
    n = 40
    frames = np.arange(n, dtype=np.float64)
    centers = np.column_stack([3.0 * frames, 2.0 * frames])
    track = _track_from_centers(centers, frames)

    report = assess_quality(track)
    assert all(f.reason != "jitter" for f in report.flags)
