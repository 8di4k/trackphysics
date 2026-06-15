"""Tests for object-size-as-ruler + the SNR gate + the §10 size↔gravity cross-check.

All synthetic data is built INSIDE this module (no bench / cross-group imports), mirroring
``test_ballistic.py``. The core fixture is the same projected parabola, but with a
controllable bbox *size* series so the apparent-size channel carries (or fails to carry)
scale information.

Coverage:
* ``apparent_size_px`` geometry; ``read_object_size`` None-paths, clean scale, noise/SNR,
  dynamic-range; ``scale_agreement`` arithmetic incl. the fail-closed non-finite path.
* the size ruler is *exact* when clean (it inverts the projection);
* integration: a consistent size cue leaves METRIC confidence/CI unchanged (agreement ⇒ no
  inflation), an inconsistent informative cue LOWERS confidence and WIDENS the CI, a
  sub-noise cue changes nothing, and ``diameter_m=None`` is byte-for-byte the old behaviour.
"""

from __future__ import annotations

import numpy as np

from trackphysics.core.ballistic import fit_ballistic
from trackphysics.core.grounding import GroundingContext
from trackphysics.core.objsize import (
    apparent_size_px,
    read_object_size,
    scale_agreement,
)
from trackphysics.core.provenance import Tier
from trackphysics.core.schema import Detection, Segment, TrackSequence

G = 9.81
FPS = 120.0
S_TRUE = 0.004  # meters per pixel
BOX = 12.0  # bbox half-extent in px (so apparent diameter = 24 px)
# A physical diameter consistent with the true scale: D / (2*BOX) == S_TRUE.
D_CONSISTENT = S_TRUE * 2.0 * BOX  # 0.096 m


def _size_track(
    *,
    n: int = 30,
    vx_world: float = 8.0,
    vy_world: float = 12.0,
    half_extent: np.ndarray | float = BOX,
    seed: int = 0,
) -> TrackSequence:
    """Projected parabola with a controllable per-frame bbox half-extent (size channel)."""
    t = np.arange(n, dtype=np.float64) / FPS
    x_px = 200.0 + (vx_world * t) / S_TRUE
    y_px = 700.0 - (vy_world * t - 0.5 * G * t * t) / S_TRUE
    he = np.full(n, float(half_extent)) if np.isscalar(half_extent) else np.asarray(half_extent)
    dets: list[Detection] = []
    for f in range(n):
        cx, cy, h = x_px[f], y_px[f], float(he[f])
        bbox = np.array([cx - h, cy - h, cx + h, cy + h], dtype=np.float64)
        dets.append(Detection(frame=f, bbox=bbox, track_id=1))
    return TrackSequence(detections=dets, fps=FPS, image_size=(1920, 1080))


def _full_segment(track: TrackSequence) -> Segment:
    idx = np.arange(len(track), dtype=np.int64)
    frames = track.frames
    return Segment(start_frame=int(frames[0]), end_frame=int(frames[-1]),
                   kind="ballistic", indices=idx)


# --------------------------------------------------------------------------------------
# apparent_size_px
# --------------------------------------------------------------------------------------


def test_apparent_size_square_box() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 25.0]])
    sizes = apparent_size_px(boxes)
    assert np.isclose(sizes[0], 10.0)  # 10x10 -> 10
    assert np.isclose(sizes[1], np.sqrt(10.0 * 20.0))  # geomean(10, 20)


def test_apparent_size_rejects_bad_shape() -> None:
    import pytest

    with pytest.raises(ValueError):
        apparent_size_px(np.zeros((3, 2)))


# --------------------------------------------------------------------------------------
# read_object_size
# --------------------------------------------------------------------------------------


def test_read_object_size_none_paths() -> None:
    a = np.full(10, 24.0)
    assert read_object_size(a, None) is None
    assert read_object_size(a, 0.0) is None
    assert read_object_size(a, -1.0) is None
    assert read_object_size(np.full(3, 24.0), 0.1) is None  # < min_points


def test_read_object_size_clean_scale_and_snr() -> None:
    a = np.full(20, 24.0)  # perfectly constant apparent size
    r = read_object_size(a, D_CONSISTENT)
    assert r is not None
    assert np.isclose(r.scale_m_per_px, D_CONSISTENT / 24.0)
    assert np.isclose(r.scale_m_per_px, S_TRUE)
    assert r.rel_noise < 1e-6
    assert r.informative_for_scale  # essentially no noise
    assert not r.informative_for_depth  # constant size -> no dynamic range


def test_read_object_size_noise_raises_rel_noise() -> None:
    rng = np.random.default_rng(3)
    a = 24.0 + rng.normal(0.0, 5.0, size=40)  # ~20% relative noise
    r = read_object_size(np.maximum(a, 1.0), D_CONSISTENT)
    assert r is not None
    assert r.rel_noise > 0.12
    assert not r.informative_for_scale  # sub-noise: gate refuses the absolute scale


def test_read_object_size_dynamic_range_detected() -> None:
    # A smooth size ramp (depth motion) with tiny noise: high snr_dyn, depth-usable.
    a = np.linspace(20.0, 40.0, 40)
    r = read_object_size(a, D_CONSISTENT)
    assert r is not None
    assert r.dynamic_range_px > 15.0
    assert r.informative_for_depth


# --------------------------------------------------------------------------------------
# size ruler exactness (it inverts the projection)
# --------------------------------------------------------------------------------------


def test_size_ruler_exact_per_frame() -> None:
    # Apparent diameter from the pinhole law d = f*D/Z over a depth ramp; the ruler D/d
    # must return exactly the true per-frame scale Z/f.
    focal = 900.0
    depth = np.linspace(5.0, 9.0, 30)
    d_px = focal * D_CONSISTENT / depth
    size_scale = D_CONSISTENT / d_px
    true_scale = depth / focal
    assert np.allclose(size_scale, true_scale, rtol=1e-9)


# --------------------------------------------------------------------------------------
# scale_agreement
# --------------------------------------------------------------------------------------


def test_scale_agreement_identical_and_disagree() -> None:
    rel, agree = scale_agreement(0.004, 0.004)
    assert np.isclose(rel, 0.0) and np.isclose(agree, 1.0)
    rel, agree = scale_agreement(0.004 * 1.10, 0.004, tol=0.25)
    assert np.isclose(rel, 0.10, atol=1e-6)
    assert np.isclose(agree, 1.0 - 0.10 / 0.25, atol=1e-6)
    # at/over tol -> zero agreement
    _, agree = scale_agreement(0.004 * 1.30, 0.004, tol=0.25)
    assert agree == 0.0


def test_scale_agreement_fails_closed_on_nonfinite() -> None:
    rel, agree = scale_agreement(float("nan"), 0.004)
    assert not np.isfinite(rel) and agree == 0.0
    rel, agree = scale_agreement(0.004, 0.0)  # non-positive gravity scale
    assert not np.isfinite(rel) and agree == 0.0


# --------------------------------------------------------------------------------------
# Integration into fit_ballistic
# --------------------------------------------------------------------------------------


def _fit(track: TrackSequence, diameter_m: float | None):
    return fit_ballistic(track, _full_segment(track), GroundingContext(), diameter_m=diameter_m)


def _ci_width(est) -> float:
    ci = est.meta["launch_speed_ci95"]
    return float(ci[1] - ci[0])


def test_diameter_none_is_unchanged_default() -> None:
    track = _size_track()
    est = _fit(track, None)
    assert est.tier == Tier.METRIC
    assert "object_size_ruler" not in est.meta  # no size logic ran


def test_consistent_size_cue_leaves_metric_intact() -> None:
    track = _size_track()
    base = _fit(track, None)
    cross = _fit(track, D_CONSISTENT)
    assert cross.tier == Tier.METRIC
    assert cross.meta["object_size_ruler"] == "cross_checked"
    assert cross.meta["scale_cross_check_rel_disc"] < 0.02  # scales agree
    assert np.isclose(cross.meta["scale_agreement"], 1.0, atol=1e-3)
    # Agreement must NOT inflate: confidence and CI are unchanged vs the no-size baseline.
    assert np.isclose(cross.velocity.confidence, base.velocity.confidence, rtol=1e-6)
    assert np.isclose(_ci_width(cross), _ci_width(base), rtol=1e-6)


def test_inconsistent_informative_size_cue_lowers_confidence_and_widens_ci() -> None:
    track = _size_track()
    base = _fit(track, None)
    # A size 10% too small => size scale 10% off gravity => informative disagreement.
    bad = _fit(track, D_CONSISTENT * 0.90)
    assert bad.tier == Tier.METRIC  # one-sided: we widen/discount, never hard-downgrade here
    assert bad.meta["object_size_ruler"] == "cross_checked"
    assert bad.meta["scale_cross_check_rel_disc"] > 0.05
    assert bad.velocity.confidence < base.velocity.confidence  # discounted
    assert _ci_width(bad) > _ci_width(base)  # widened


def test_gross_disagreement_zeroes_confidence_but_keeps_metric_tier() -> None:
    # rel_disc >= agreement_tol => agreement 0 => confidence driven to exactly 0.0, while the
    # tier stays METRIC (the one-sided cross-check discounts/ widens; it does NOT flip the
    # tier — that is the depth guard's job). Pin this load-bearing contract.
    track = _size_track()
    base = _fit(track, None)
    gross = _fit(track, D_CONSISTENT * 0.5)  # size scale 50% off gravity => over tol
    assert gross.meta["object_size_ruler"] == "cross_checked"
    assert gross.meta["scale_agreement"] == 0.0
    assert gross.velocity.confidence == 0.0  # exactly zero, not merely small
    assert gross.tier == Tier.METRIC
    assert _ci_width(gross) > _ci_width(base)  # still widened, not collapsed


def test_one_sided_invariant_across_the_agreement_ramp() -> None:
    # For EVERY informative reading, the cross-check may only lower confidence and widen the
    # CI vs the no-size baseline — never the reverse, anywhere on the agreement ramp.
    track = _size_track()
    base = _fit(track, None)
    bw, bc = _ci_width(base), base.velocity.confidence
    for mult in (0.90, 0.95, 1.0, 1.05, 1.15):
        est = _fit(track, D_CONSISTENT * mult)
        assert est.velocity.confidence <= bc + 1e-9, f"confidence inflated at mult={mult}"
        assert _ci_width(est) >= bw - 1e-9, f"CI narrowed at mult={mult}"


def test_read_object_size_filters_non_finite() -> None:
    # The np.isfinite filter inside read_object_size is the SOLE defense against a NaN/inf
    # apparent-size series producing a NaN-scale METRIC over-claim (apparent_size_px
    # propagates NaN). Pin it: embedded non-finite values are dropped, scale is unaffected;
    # an all-non-finite series yields None (no reading).
    a = np.full(20, 24.0)
    a[3] = np.nan
    a[7] = np.inf
    r = read_object_size(a, D_CONSISTENT)
    assert r is not None
    assert r.n == 18  # the two non-finite samples were dropped
    assert np.isclose(r.scale_m_per_px, D_CONSISTENT / 24.0)
    assert read_object_size(np.full(20, np.nan), D_CONSISTENT) is None


def test_scale_agreement_zero_exactly_at_tolerance_boundary() -> None:
    tol = 0.25
    rel, agree = scale_agreement(0.004 * (1.0 + tol), 0.004, tol=tol)
    assert np.isclose(rel, tol)
    assert agree == 0.0  # boundary maps to exactly zero, not a tiny positive value


def test_sub_noise_size_cue_changes_nothing() -> None:
    rng = np.random.default_rng(7)
    # Heavy correlated-ish size noise: rel size noise well above the gate -> uninformative.
    noisy_he = np.maximum(BOX + rng.normal(0.0, 5.0, size=30), 1.0)
    track = _size_track(half_extent=noisy_he)
    base = _fit(track, None)
    sub = _fit(track, D_CONSISTENT)
    assert sub.meta["object_size_ruler"] == "sub_noise"
    # No information -> no claim either way: confidence/CI identical to the baseline.
    assert np.isclose(sub.velocity.confidence, base.velocity.confidence, rtol=1e-6)
    assert np.isclose(_ci_width(sub), _ci_width(base), rtol=1e-6)
