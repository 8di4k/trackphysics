"""Opt-in point-only depth-domination guard (§10 tier-hole, BAL-DEPTH-GUARD).

A trajectory flying along the optical axis still fits a clean in-plane parabola, so the metric
gate trusts it even though the in-plane speed is a small, unflagged fraction of the true 3D
motion. The 2D in-plane aspect ratio is a point-only proxy (low aspect => depth-dominated).
The guard is DEFAULT-OFF: with no guard (or a disabled one) behaviour is unchanged; enabled, it
continuously discounts confidence / widens the CI as aspect falls, and conservatively
downgrades METRIC->RELATIVE past a hard floor. These tests pin all of that, and that flagging
is decoupled from the (unchanged) default path.
"""

from __future__ import annotations

import numpy as np
import pytest

import trackphysics as tp
from trackphysics.core.grounding import DepthDominationGuard, GroundingContext
from trackphysics.core.schema import Detection, TrackSequence


def _arc(aspect: float, *, n: int = 20, fps: float = 120.0, a_px: float = 3000.0) -> TrackSequence:
    """A clean downward pixel parabola (passes the metric gate) with a CHOSEN in-plane aspect.

    Vertical (image-y) is a gravity-like parabola; horizontal (image-x) is a line whose extent
    is ``aspect`` times the vertical extent, so ptp(u)/ptp(v) == aspect by construction.
    """
    t = np.arange(n) / fps
    v = 300.0 + 120.0 * t + 0.5 * a_px * t * t
    vext = float(np.ptp(v))
    u = 200.0 + (aspect * vext) * (t / t[-1])
    h = 6.0
    dets = [
        Detection(frame=i, bbox=np.array([u[i] - h, v[i] - h, u[i] + h, v[i] + h]), track_id=1)
        for i in range(n)
    ]
    return TrackSequence(detections=dets, fps=fps, image_size=(1280, 720))


def _traj(track: TrackSequence, guard: DepthDominationGuard | None) -> tp.TrajectoryEstimate:
    ctx = GroundingContext(depth_guard=guard)
    return tp.analyze(track, preset="sphere", grounding=ctx).trajectory


# --------------------------------------------------------------------------------------
# depth_score: continuous, cliff-free, correct boundaries
# --------------------------------------------------------------------------------------


def test_depth_score_boundaries_and_monotone() -> None:
    g = DepthDominationGuard(enabled=True, soft_aspect=2.5, hard_aspect=0.8)
    assert g.depth_score(3.0) == 0.0          # >= soft: no depth domination
    assert g.depth_score(2.5) == 0.0          # at soft
    assert g.depth_score(0.8) == pytest.approx(1.0)  # at hard floor
    assert g.depth_score(0.1) == 1.0          # below hard, clamped
    mid = g.depth_score(1.65)                  # midpoint of [0.8, 2.5]
    assert 0.45 < mid < 0.55
    # strictly increasing as aspect falls through the ramp (no cliff)
    xs = [2.4, 2.0, 1.6, 1.2, 0.9]
    scores = [g.depth_score(x) for x in xs]
    assert all(b > a for a, b in zip(scores[:-1], scores[1:], strict=True))


def test_guard_validation() -> None:
    with pytest.raises(ValueError, match="hard_aspect must not exceed soft_aspect"):
        DepthDominationGuard(hard_aspect=3.0, soft_aspect=2.0)
    with pytest.raises(ValueError, match="confidence_penalty"):
        DepthDominationGuard(confidence_penalty=1.5)
    with pytest.raises(ValueError, match="ci_widen"):
        DepthDominationGuard(ci_widen=-0.1)


# --------------------------------------------------------------------------------------
# default-OFF invariance, then the three enabled regimes
# --------------------------------------------------------------------------------------


def test_guard_default_off_is_unchanged() -> None:
    track = _arc(0.5)  # a depth-dominated arc the gate would otherwise trust
    none = _traj(track, None)
    disabled = _traj(track, DepthDominationGuard(enabled=False))
    assert none.tier is tp.Tier.METRIC
    assert disabled.tier is tp.Tier.METRIC
    assert disabled.velocity.confidence == pytest.approx(none.velocity.confidence)


def test_clear_in_plane_arc_is_not_penalized() -> None:
    track = _arc(4.0)  # aspect >> soft: clearly in-plane
    off = _traj(track, None)
    on = _traj(track, DepthDominationGuard(enabled=True))
    assert on.tier is tp.Tier.METRIC
    assert float(on.meta["depth_domination_score"]) == 0.0  # type: ignore[arg-type]
    assert on.velocity.confidence == pytest.approx(off.velocity.confidence)


def test_mid_aspect_arc_keeps_metric_but_discounts_confidence_and_widens_ci() -> None:
    track = _arc(1.5)  # between hard and soft
    off = _traj(track, None)
    on = _traj(track, DepthDominationGuard(enabled=True))
    assert on.tier is tp.Tier.METRIC
    score = float(on.meta["depth_domination_score"])  # type: ignore[arg-type]
    assert 0.0 < score < 1.0
    assert on.velocity.confidence < off.velocity.confidence  # continuous penalty
    off_ci = off.meta["launch_speed_ci95"]
    on_ci = on.meta["launch_speed_ci95"]
    assert off_ci is not None and on_ci is not None
    on_w = float(on_ci[1]) - float(on_ci[0])      # type: ignore[index]
    off_w = float(off_ci[1]) - float(off_ci[0])   # type: ignore[index]
    assert on_w > off_w  # CI widened (the recovered scale is depth-biased)
    assert float(on.meta["in_plane_aspect"]) == pytest.approx(1.5, abs=0.05)  # type: ignore[arg-type]


def test_depth_dominated_arc_is_downgraded_to_relative() -> None:
    track = _arc(0.5)  # aspect <= hard floor -> hopeless depth domination
    off = _traj(track, None)
    on = _traj(track, DepthDominationGuard(enabled=True))
    assert off.tier is tp.Tier.METRIC  # the gate alone trusts it (the §10 hole)
    assert on.tier is tp.Tier.RELATIVE  # guard conservatively downgrades
    assert on.meta.get("fallback_reason") == "depth_domination_downgrade"
