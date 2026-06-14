"""Non-finite / degenerate-input hardening (BRIEF.md §11 — the dirty-track moat).

These are regression guards for a cluster of defects where NaN/inf observations or
degenerate timing either CRASHED the pipeline or — worse — slipped a fabricated METRIC /
full-confidence result past the provenance gates (the cardinal §10 sin). Real tracker
output routinely carries a NaN box on an occluded frame, so the engine must degrade
honestly, never crash and never fabricate.

Covered:
* ``combine_confidence`` treats a non-finite factor as a FAILED cue (0.0), not a perfect
  one (1.0), and a zero-weighted factor does not veto the result.
* ``fit_ballistic`` drops non-finite observations before fitting; a poisoned fit can never
  bypass the metric sanity gate to emit a NaN-scale METRIC quantity.
* ``analyze`` / ``relative_lift`` survive a NaN bbox instead of crashing on a NaN
  goodness_of_fit.
* the degenerate all-zeros relative-lift velocity ships at zero confidence, not the
  position confidence.
"""

from __future__ import annotations

import numpy as np

import trackphysics as tp
from trackphysics.core.ballistic import fit_ballistic
from trackphysics.core.grounding import GroundingContext
from trackphysics.core.lift import relative_lift
from trackphysics.core.provenance import Tier, combine_confidence
from trackphysics.core.schema import Detection, Segment, TrackSequence

# --------------------------------------------------------------------------------------
# combine_confidence — the central confidence primitive
# --------------------------------------------------------------------------------------


def test_combine_confidence_treats_nonfinite_as_failed_cue() -> None:
    assert combine_confidence(float("nan")) == 0.0
    assert combine_confidence(float("inf")) == 0.0
    # A NaN cue must drag the result to zero (fail-closed), never inflate it toward 1.0.
    assert combine_confidence(0.9, float("nan")) == 0.0
    assert combine_confidence(0.9, 0.8, float("-inf")) == 0.0


def test_combine_confidence_zero_weight_does_not_veto() -> None:
    # A factor excluded from the mean (weight 0) must not force the whole result to 0 just
    # because its value is 0 — only a positively-weighted zero is a hard disqualifier.
    assert combine_confidence(0.9, 0.0, weights=(1.0, 0.0)) == 0.9
    assert combine_confidence(0.9, 0.8, weights=(1.0, 0.0)) == 0.9
    # A positively-weighted zero still vetoes.
    assert combine_confidence(0.9, 0.0, weights=(1.0, 1.0)) == 0.0


# --------------------------------------------------------------------------------------
# fit_ballistic — non-finite observations must not crash or fabricate METRIC
# --------------------------------------------------------------------------------------

_BOX = 12.0
_G = 9.81
_SCALE = 0.004  # m/px used only to synthesise a pixel arc


def _ballistic_track(ys: list[float], xs: np.ndarray, fps: float = 120.0) -> TrackSequence:
    dets = [
        Detection(
            frame=i,
            bbox=np.array([xs[i] - _BOX, ys[i] - _BOX, xs[i] + _BOX, ys[i] + _BOX]),
            track_id=1,
        )
        for i in range(len(ys))
    ]
    return TrackSequence(detections=dets, fps=fps)


def _seg(n: int) -> Segment:
    return Segment(
        start_frame=0, end_frame=n - 1, kind="ballistic", indices=np.arange(n, dtype=np.int64)
    )


def test_fit_ballistic_with_one_nan_center_recovers_without_fabricating_metric() -> None:
    t = np.arange(8) / 120.0
    ys = list(600.0 - (12.0 * t - 0.5 * _G * t * t) / _SCALE)
    ys[4] = np.nan  # one occlusion-poisoned center
    xs = 100.0 + (8.0 * t) / _SCALE
    est = fit_ballistic(_ballistic_track(ys, xs), _seg(8), GroundingContext(gravity=_G))
    # The cardinal invariant: a poisoned fit may fall back to RELATIVE, or recover METRIC by
    # dropping the bad point — but a METRIC result must carry only FINITE values/scale.
    assert est.tier in (Tier.METRIC, Tier.RELATIVE)
    assert np.isfinite(est.positions.value).all()
    assert np.isfinite(est.velocity.value).all()
    assert 0.0 <= est.goodness_of_fit <= 1.0
    if est.tier is Tier.METRIC:
        assert np.isfinite(float(est.meta["scale_m_per_px"]))  # type: ignore[arg-type]
        assert np.isfinite(float(est.meta["launch_speed_m_s"]))  # type: ignore[arg-type]


def test_fit_ballistic_all_nan_segment_falls_back_relative() -> None:
    n = 8
    ys = [float("nan")] * n
    xs = np.full(n, np.nan)
    est = fit_ballistic(_ballistic_track(ys, xs), _seg(n), GroundingContext())
    assert est.tier is Tier.RELATIVE
    assert 0.0 <= est.goodness_of_fit <= 1.0


def test_fit_ballistic_with_inf_center_does_not_emit_nan_metric() -> None:
    t = np.arange(9) / 120.0
    ys = list(600.0 - (12.0 * t - 0.5 * _G * t * t) / _SCALE)
    ys[2] = float("inf")
    xs = 100.0 + (8.0 * t) / _SCALE
    est = fit_ballistic(_ballistic_track(ys, xs), _seg(9), GroundingContext(gravity=_G))
    assert np.isfinite(est.positions.value).all()
    assert np.isfinite(est.velocity.value).all()


# --------------------------------------------------------------------------------------
# analyze() / relative_lift — a NaN bbox must not crash the always-run floor
# --------------------------------------------------------------------------------------


def _linear_track_with_nan_at(nan_index: int | None, n: int = 10) -> TrackSequence:
    dets = [
        Detection(frame=i, bbox=np.array([100.0 + i, 200.0 + i, 110.0 + i, 210.0 + i]), track_id=1)
        for i in range(n)
    ]
    if nan_index is not None:
        dets[nan_index].bbox = np.array([np.nan, np.nan, np.nan, np.nan])
    return TrackSequence(detections=dets, fps=60.0, image_size=(640, 480))


def test_analyze_survives_nan_bbox() -> None:
    res = tp.analyze(_linear_track_with_nan_at(5))
    assert res.trajectories  # did not crash; a floor estimate exists
    floor = next(t for t in res.trajectories if t.tier is Tier.RELATIVE)
    assert 0.0 <= floor.goodness_of_fit <= 1.0


def test_relative_lift_nan_bbox_yields_finite_goodness_of_fit() -> None:
    est = relative_lift(_linear_track_with_nan_at(3))
    assert np.isfinite(est.goodness_of_fit)
    assert 0.0 <= est.goodness_of_fit <= 1.0
    assert 0.0 <= est.positions.confidence <= 1.0


def test_relative_lift_degenerate_velocity_has_zero_confidence() -> None:
    # Duplicate timestamps -> velocity cannot be differentiated -> all-zeros placeholder.
    dets = [
        Detection(frame=i, bbox=np.array([10.0 + i, 20.0 + i, 12.0 + i, 22.0 + i]), track_id=1)
        for i in range(5)
    ]
    track = TrackSequence(
        detections=dets, fps=100.0, timestamps=np.array([0.0, 0.01, 0.01, 0.02, 0.03])
    )
    est = relative_lift(track)
    assert est.velocity.source == "finite_difference_degenerate"
    assert est.velocity.confidence == 0.0  # the fabricated velocity is NOT measured
    assert np.all(est.velocity.value == 0.0)
