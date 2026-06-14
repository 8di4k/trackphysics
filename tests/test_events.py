"""Tests for generic, domain-free event detection (BRIEF.md §12.3).

Covers the success path AND the honesty path for each detector:

* bounce fires at a known vertical-velocity sign reversal, with pre/post velocity in
  the payload (success), and does NOT fire on jittery noise with no true reversal
  (honesty);
* contact fires when two tracks approach within the scale-aware threshold while one
  undergoes a sudden velocity change (success), and does NOT fire when the objects stay
  far apart or no discontinuity occurs (honesty);
* release fires at the onset of a constructed free-flight (constant-acceleration)
  segment.

All fixtures are built inline from the Stage-B contract types — no imports of ``bench``
or other groups' modules.
"""

from __future__ import annotations

import numpy as np

from trackphysics.core.events import BounceDetector, ReleaseDetector, detect_contacts
from trackphysics.core.provenance import Quantity, Tier, TrajectoryEstimate
from trackphysics.core.schema import Detection, Segment, TrackSequence

FPS = 100.0


def _bbox(cx: float, cy: float, half: float = 5.0) -> np.ndarray:
    """An xyxy bbox centered at (cx, cy) with the given half-size."""
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float64)


def _quantity(payload: dict[str, object], key: str) -> Quantity:
    """Fetch a payload entry, asserting it is a Quantity (narrows for the type checker)."""
    value = payload[key]
    assert isinstance(value, Quantity)
    return value


def _track_from_centers(
    centers: np.ndarray, *, track_id: int = 1, half: float = 5.0
) -> TrackSequence:
    """Build a TrackSequence whose detection centers follow ``centers`` (T, 2)."""
    dets = [
        Detection(frame=f, bbox=_bbox(float(c[0]), float(c[1]), half), track_id=track_id)
        for f, c in enumerate(centers)
    ]
    return TrackSequence(detections=dets, fps=FPS, image_size=(1000, 1000))


def _estimate_with_vertical_velocity(
    vy: np.ndarray,
    *,
    tier: Tier = Tier.PIXEL,
    gof: float = 0.9,
    segment: Segment | None = None,
) -> tuple[TrajectoryEstimate, TrackSequence]:
    """Build a (T, 2) velocity estimate with vertical column ``vy`` and a matching track.

    The horizontal velocity is a constant so the only sign behavior of interest is in the
    vertical column the detector inspects.
    """
    t = vy.shape[0]
    vx = np.full(t, 3.0, dtype=np.float64)
    velocity = np.stack([vx, vy], axis=1)
    # Positions are an arbitrary integral; the detectors key off velocity here.
    positions = np.cumsum(velocity, axis=0) / FPS
    centers = positions * 100.0 + 500.0
    track = _track_from_centers(centers)
    est = TrajectoryEstimate(
        positions=Quantity(positions, unit="px", tier=tier, confidence=0.9, source="t"),
        velocity=Quantity(velocity, unit="px/s", tier=tier, confidence=0.9, source="t"),
        tier=tier,
        goodness_of_fit=gof,
        segment=segment,
    )
    return est, track


# --------------------------------------------------------------------------------------
# (a) Bounce fires at a known sign reversal with pre/post velocity in payload
# --------------------------------------------------------------------------------------


def test_bounce_fires_at_known_reversal() -> None:
    # Vertical velocity: clearly positive (down) for 6 frames, flips to clearly negative
    # (up) at frame 6. The reversal sample is index 6.
    vy = np.concatenate([np.full(6, 4.0), np.full(6, -4.0)])
    est, track = _estimate_with_vertical_velocity(vy)

    events = BounceDetector().detect(est, track)

    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "impact"
    assert ev.frame == 6
    assert 0.0 < ev.confidence <= 1.0
    pre = _quantity(ev.payload, "pre_velocity")
    post = _quantity(ev.payload, "post_velocity")
    assert pre.value > 0.0 > post.value
    assert pre.tier is Tier.PIXEL and post.tier is Tier.PIXEL


def test_bounce_emit_kind_configurable() -> None:
    vy = np.concatenate([np.full(5, 3.0), np.full(5, -3.0)])
    est, track = _estimate_with_vertical_velocity(vy)
    events = BounceDetector(emit_kind="bounce").detect(est, track)
    assert [e.kind for e in events] == ["bounce"]


def test_bounce_works_for_world_axis_sign_convention() -> None:
    # World up-axis convention: a bounce is up(+) -> down... no; physically the object
    # comes down (negative on an up-axis) and rebounds up (positive). Either way it is a
    # reversal; the detector must fire regardless of which sign comes first.
    vy = np.concatenate([np.full(5, -4.0), np.full(5, 4.0)])
    est, track = _estimate_with_vertical_velocity(vy, tier=Tier.RELATIVE)
    events = BounceDetector().detect(est, track)
    assert len(events) == 1
    assert events[0].frame == 5
    # Relative tier: velocity unit on the estimate is propagated, tier preserved.
    assert _quantity(events[0].payload, "pre_velocity").tier is Tier.RELATIVE


# --------------------------------------------------------------------------------------
# (b) Honesty: jittery vertical velocity with no true reversal does NOT fire
# --------------------------------------------------------------------------------------


def test_bounce_does_not_fire_on_jitter() -> None:
    # Strongly downward motion with small zero-mean jitter that never produces a real,
    # large-magnitude reversal. A naive sign-change detector would misfire; ours must not.
    rng = np.random.default_rng(0)
    vy = 6.0 + rng.normal(0.0, 0.3, size=40)  # stays well above zero, never reverses
    est, track = _estimate_with_vertical_velocity(vy)
    events = BounceDetector().detect(est, track)
    assert events == []


def test_bounce_ignores_tiny_reversal_near_zero() -> None:
    # Motion decelerates to near zero and jitters across zero by a negligible amount.
    # This is jitter, not a bounce; the magnitude gate must reject it.
    base = np.linspace(5.0, 0.0, 10)
    rng = np.random.default_rng(1)
    tail = rng.normal(0.0, 0.05, size=10)  # tiny crossings around zero
    vy = np.concatenate([base, tail])
    est, track = _estimate_with_vertical_velocity(vy)
    events = BounceDetector(min_speed_fraction=0.3).detect(est, track)
    assert events == []


def test_bounce_empty_for_too_short_series() -> None:
    vy = np.array([2.0, -2.0])  # only two samples
    est, track = _estimate_with_vertical_velocity(vy)
    assert BounceDetector().detect(est, track) == []


def test_bounce_falls_back_to_positions_for_scalar_velocity() -> None:
    # A scalar (T,) speed series cannot express signed vertical motion, so the detector
    # must fall back to differencing the vertical position column.
    t = 12
    vertical_pos = np.concatenate([np.linspace(0.0, 1.0, 6), np.linspace(1.0, 0.0, 6)])
    positions = np.stack([np.linspace(0.0, 1.0, t), vertical_pos], axis=1)
    speed = np.abs(np.gradient(vertical_pos)) * FPS
    centers = positions * 100.0 + 400.0
    track = _track_from_centers(centers)
    est = TrajectoryEstimate(
        positions=Quantity(positions, unit=None, tier=Tier.RELATIVE, confidence=0.8, source="t"),
        velocity=Quantity(speed, unit=None, tier=Tier.RELATIVE, confidence=0.8, source="t"),
        tier=Tier.RELATIVE,
        goodness_of_fit=0.8,
    )
    events = BounceDetector().detect(est, track)
    assert len(events) >= 1
    # The reversal of the vertical position derivative is around the peak (index ~5-6).
    assert any(4 <= e.frame <= 7 for e in events)


# --------------------------------------------------------------------------------------
# (c) Contact: two approaching tracks, one with a sudden velocity change
# --------------------------------------------------------------------------------------


def test_contact_fires_on_proximity_plus_velocity_jump() -> None:
    # Track A drifts toward B, reaching closest approach around frame 10. Track B has only
    # a slow drift before then (a low speed scale), then at frame 10 is struck and shoots
    # downward (a sharp velocity discontinuity that dominates its own speed scale).
    n = 20
    a_centers = np.stack([np.linspace(160.0, 240.0, n), np.full(n, 100.0)], axis=1)

    b_centers = np.zeros((n, 2), dtype=np.float64)
    b_centers[:, 1] = 100.0
    for i in range(n):
        b_centers[i, 0] = 200.0 + 0.5 * i  # slow drift -> small speed scale
    for i in range(11, n):
        b_centers[i, 1] = 100.0 + 8.0 * (i - 10)  # impulse downward at frame 10

    track_a = _track_from_centers(a_centers, track_id=1, half=6.0)
    track_b = _track_from_centers(b_centers, track_id=2, half=6.0)

    events = detect_contacts(track_a, track_b)

    assert len(events) >= 1
    frames = [e.frame for e in events]
    assert any(8 <= f <= 12 for f in frames)
    ev = next(e for e in events if 8 <= e.frame <= 12)
    assert ev.kind == "contact"
    assert 0.0 < ev.confidence <= 1.0
    assert _quantity(ev.payload, "separation").tier is Tier.PIXEL
    assert ev.payload["track_ids"] == (1, 2)


def test_contact_does_not_fire_when_objects_stay_far_apart() -> None:
    # Both move and B has a velocity jump, but they are never within proximity.
    n = 20
    a_centers = np.stack([np.linspace(0.0, 200.0, n), np.full(n, 50.0)], axis=1)
    b_centers = np.stack([np.linspace(0.0, 200.0, n), np.full(n, 800.0)], axis=1)
    for i in range(10, n):
        b_centers[i, 1] = 800.0 + 30.0 * (i - 9)  # sudden jump, but far from A
    track_a = _track_from_centers(a_centers, track_id=1, half=5.0)
    track_b = _track_from_centers(b_centers, track_id=2, half=5.0)
    assert detect_contacts(track_a, track_b) == []


def test_contact_does_not_fire_without_velocity_discontinuity() -> None:
    # The two tracks overlap closely the entire time but move smoothly in parallel — no
    # discontinuity, so there is no kinematic evidence of a contact.
    n = 20
    a_centers = np.stack([np.linspace(0.0, 190.0, n), np.full(n, 100.0)], axis=1)
    b_centers = a_centers + np.array([3.0, 0.0])  # always within proximity, smooth
    track_a = _track_from_centers(a_centers, track_id=1, half=8.0)
    track_b = _track_from_centers(b_centers, track_id=2, half=8.0)
    assert detect_contacts(track_a, track_b) == []


def test_contact_debounces_adjacent_frames_by_default() -> None:
    # A single physical contact can satisfy both gates on two CONSECUTIVE frames. Here A
    # sweeps horizontally past a (nominally fixed) B, coming within the proximity threshold
    # only for the two frames straddling closest approach (10 and 11). Across exactly those
    # two frames B undergoes a sustained velocity discontinuity (a brief constant-
    # acceleration nudge), so both frames clear the discontinuity gate while inside
    # proximity -> two qualifying contacts on adjacent frames.
    #
    # The documented default (min_separation_frames=2) must debounce this single physical
    # contact to ONE event; passing min_separation_frames=1 disables debouncing and emits
    # both. This is the regression guard for CONTACT-DEBOUNCE-OFFBYONE.
    n = 24
    half = 6.0
    frames = np.arange(n)
    bx, by = 200.0, 100.0
    # A moves ~12 px/frame so it is within the proximity threshold for ~2 frames as it
    # passes through B's location near frame 10.5.
    a_centers = np.stack([bx - 12.0 * (10.5 - frames), np.full(n, by)], axis=1)
    b_centers = np.stack([np.full(n, bx), np.full(n, by)], axis=1)
    # Localized downward parabola on B over frames 9..12 -> a sustained velocity jump.
    bump = np.where((frames >= 9) & (frames <= 12), 3.0 * (frames - 9) ** 2, 0.0)
    b_centers[:, 1] += bump
    track_a = _track_from_centers(a_centers, track_id=1, half=half)
    track_b = _track_from_centers(b_centers, track_id=2, half=half)

    # Debouncing disabled: both adjacent qualifying frames emit.
    raw = detect_contacts(track_a, track_b, min_separation_frames=1)
    assert [e.frame for e in raw] == [10, 11]

    # Default behavior: the adjacent-frame pair collapses to a single contact event.
    default = detect_contacts(track_a, track_b)
    assert [e.frame for e in default] == [10]


def test_contact_requires_overlapping_frames() -> None:
    a = _track_from_centers(np.stack([np.arange(5.0), np.zeros(5)], axis=1), track_id=1)
    # Track B on disjoint frame indices.
    dets_b = [
        Detection(frame=f, bbox=_bbox(0.0, 0.0), track_id=2) for f in range(100, 105)
    ]
    b = TrackSequence(detections=dets_b, fps=FPS)
    assert detect_contacts(a, b) == []


# --------------------------------------------------------------------------------------
# (d) Release: onset of a constructed free-flight segment
# --------------------------------------------------------------------------------------


def test_release_fires_at_onset_of_constant_acceleration() -> None:
    # First a stretch of constant velocity (not free flight), then a constant-acceleration
    # ramp (free flight under gravity). Release should be reported at the ramp onset.
    flat = np.full(6, 0.0)  # at rest / constant velocity (zero accel)
    ramp = np.linspace(0.0, 9.0, 10)  # vertical velocity increasing linearly -> const accel
    vy = np.concatenate([flat, ramp])
    est, track = _estimate_with_vertical_velocity(vy, tier=Tier.RELATIVE, gof=0.85)

    events = ReleaseDetector().detect(est, track)

    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "release"
    # Onset should land at or just after the transition into the ramp (index ~5-7).
    assert 4 <= ev.frame <= 8
    assert 0.0 < ev.confidence <= 1.0
    assert ev.payload["onset"] == "constant_acceleration"


def test_release_uses_labeled_ballistic_segment_when_present() -> None:
    # If the estimate's segment is already labeled a free-flight ("ballistic") window,
    # release is simply that window's onset frame.
    vy = np.linspace(0.0, 10.0, 12)
    indices = np.arange(12, dtype=np.int64)
    seg = Segment(start_frame=0, end_frame=11, kind="ballistic", indices=indices)
    est, track = _estimate_with_vertical_velocity(vy, segment=seg)
    events = ReleaseDetector().detect(est, track)
    assert len(events) == 1
    assert events[0].kind == "release"
    assert events[0].frame == 0
    assert events[0].payload["onset"] == "segment_start"


def test_release_empty_for_too_short_series() -> None:
    vy = np.array([1.0, 2.0])
    est, track = _estimate_with_vertical_velocity(vy)
    assert ReleaseDetector().detect(est, track) == []


def test_release_does_not_fire_on_pure_noise() -> None:
    # Zero-mean noise has no sustained constant-acceleration run -> no release.
    rng = np.random.default_rng(2)
    vy = rng.normal(0.0, 1.0, size=30)
    est, track = _estimate_with_vertical_velocity(vy)
    # Strict consistency makes the noise fail the constant-acceleration gate.
    assert ReleaseDetector(accel_consistency=0.05).detect(est, track) == []


# --------------------------------------------------------------------------------------
# Protocol conformance: detectors satisfy the EventDetector contract
# --------------------------------------------------------------------------------------


def test_detectors_satisfy_event_detector_protocol() -> None:
    from trackphysics.core.presets.base import EventDetector

    assert isinstance(BounceDetector(), EventDetector)
    assert isinstance(ReleaseDetector(), EventDetector)
