"""Generic, domain-free motion-event detection (BRIEF.md §12.3).

Every detector here defines an event *purely kinematically* — by the geometry and
derivatives of motion — and never by what the moving object *is*. The engine emits a
small generic vocabulary (``"bounce"`` / ``"impact"``, ``"contact"``, ``"release"``);
a domain layer maps those generic kinds into its own taxonomy *outside* the core
(BRIEF.md §6, hook 4). No detector here knows or assumes any object identity.

Three detectors are provided:

* :class:`BounceDetector` — an impact/bounce is a sign reversal of the *vertical*
  velocity component (downward → upward). Implements the :class:`EventDetector`
  protocol.
* :func:`detect_contacts` — a contact is spatial *proximity* between two tracked
  objects coincident with a velocity discontinuity (a sudden change) in at least one of
  them. Proximity is scale-aware (relative to bounding-box size), so it works at any
  tier without a metric reference.
* :class:`ReleaseDetector` — best-effort: the onset of a free-flight (ballistic)
  segment after non-ballistic motion. Implements :class:`EventDetector`.

All physical quantities placed in an event payload are wrapped in :class:`Quantity`
with an explicit tier/unit/confidence (BRIEF.md §10). Confidences are derived from real
kinematic factors via :func:`combine_confidence`, never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .provenance import Event, Quantity, Tier, TrajectoryEstimate, combine_confidence
from .schema import FloatArray, TrackSequence

# --------------------------------------------------------------------------------------
# Conventions and shared helpers
# --------------------------------------------------------------------------------------

# Image coordinate convention (see schema.Detection): pixel ``y`` increases *downward*.
# Therefore an object moving downward in the scene has POSITIVE pixel ``y``-velocity, and
# an object moving upward has NEGATIVE pixel ``y``-velocity. A bounce/impact off a surface
# below the object is thus a transition from positive (down) to negative (up) vertical
# velocity. For a metric/relative estimate the world ``y`` (or ``z``) axis points *up*, so
# the same physical bounce appears as negative→positive on that axis; the detector keys
# off the reversal regardless of axis sign convention (see :func:`_bounce_indices`).


def _column(arr: FloatArray, index: int) -> FloatArray:
    """Return a 1-D view/copy of column ``index`` of a 2-D array as float64."""
    return np.asarray(arr[:, index], dtype=np.float64)


def _finite_difference(values: FloatArray, times: FloatArray) -> FloatArray:
    """Central finite-difference derivative of ``values`` w.r.t. ``times``.

    Returns an array the same length as the inputs. Endpoints use one-sided
    differences. Used to turn a pixel-position series into a velocity series when an
    estimate does not already carry one.
    """
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n = values.shape[0]
    deriv = np.zeros(n, dtype=np.float64)
    if n < 2:
        return deriv
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        dt = times[hi] - times[lo]
        if dt <= 0:
            deriv[i] = 0.0
        else:
            deriv[i] = (values[hi] - values[lo]) / dt
    return deriv


def _vertical_velocity_series(est: TrajectoryEstimate, track: TrackSequence) -> FloatArray:
    """Extract the vertical-velocity series implied by an estimate.

    Prefers the estimate's own ``velocity`` array. A ``(T, D)`` velocity uses its last
    spatial column as the vertical axis (world up-axis for 3D, pixel-``y`` for 2D). A
    ``(T,)`` scalar series cannot expose a signed vertical component, so we fall back to
    differencing the vertical position. If neither is array-valued, we difference the
    track's pixel centers as a last resort.
    """
    vel = est.velocity.value
    if isinstance(vel, np.ndarray) and vel.ndim == 2 and vel.shape[1] >= 2:
        return _column(vel, vel.shape[1] - 1)

    pos = est.positions.value
    times = _estimate_times(est, track)
    if isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[1] >= 2:
        vertical_pos = _column(pos, pos.shape[1] - 1)
        return _finite_difference(vertical_pos, times)

    # Last resort: difference the track's pixel-y centers over the segment window.
    centers = track.centers()
    if centers.shape[0] >= 2:
        return _finite_difference(_column(centers, 1), track.times())
    return np.zeros(0, dtype=np.float64)


def _estimate_times(est: TrajectoryEstimate, track: TrackSequence) -> FloatArray:
    """Per-sample times aligned with an estimate's position/velocity arrays.

    Uses the segment's materialized ``indices`` into the track when available; otherwise
    falls back to the full track timeline truncated/padded to the array length.
    """
    track_times = track.times()
    seg = est.segment
    if seg is not None and seg.indices is not None and seg.indices.size > 0:
        idx = seg.indices
        if int(idx.max()) < track_times.shape[0]:
            return track_times[idx]
    arr = est.velocity.value
    if not (isinstance(arr, np.ndarray) and arr.ndim >= 1):
        arr = est.positions.value
    n = arr.shape[0] if isinstance(arr, np.ndarray) and arr.ndim >= 1 else track_times.shape[0]
    if track_times.shape[0] >= n:
        return track_times[:n]
    # Synthesize a uniform timeline if the track is shorter than the estimate arrays.
    fps = track.fps if track.fps > 0 else 1.0
    return np.arange(n, dtype=np.float64) / fps


def _frame_for_index(est: TrajectoryEstimate, track: TrackSequence, index: int) -> int:
    """Map a position into the estimate's arrays back to an absolute frame index."""
    seg = est.segment
    if seg is not None and seg.indices is not None and 0 <= index < seg.indices.size:
        frames = track.frames
        det_index = int(seg.indices[index])
        if 0 <= det_index < frames.shape[0]:
            return int(frames[det_index])
    frames = track.frames
    if 0 <= index < frames.shape[0]:
        return int(frames[index])
    if seg is not None:
        return int(seg.start_frame + index)
    return int(index)


# --------------------------------------------------------------------------------------
# Bounce / impact detection
# --------------------------------------------------------------------------------------


def _bounce_indices(
    v_vertical: FloatArray,
    *,
    min_speed: float,
) -> list[int]:
    """Indices where vertical velocity reverses sign across a non-trivial magnitude.

    A reversal is reported at sample ``i`` (the first post-reversal sample) when the
    smoothed velocity changes sign between ``i-1`` and ``i`` AND the larger of the two
    surrounding magnitudes exceeds ``min_speed`` (rejects jitter around zero). Detection
    is symmetric in axis-sign convention: it fires on down→up *or* up→down so that it is
    valid whether the vertical axis points up (world) or down (pixel-``y``); see the
    module-level convention note.
    """
    n = v_vertical.shape[0]
    if n < 3:
        return []
    indices: list[int] = []
    for i in range(1, n):
        a = float(v_vertical[i - 1])
        b = float(v_vertical[i])
        # Require a genuine sign flip with both sides nonzero.
        if a == 0.0 or b == 0.0:
            continue
        if np.sign(a) == np.sign(b):
            continue
        if max(abs(a), abs(b)) < min_speed:
            continue
        indices.append(i)
    return indices


@dataclass
class BounceDetector:
    """Detect bounces/impacts as vertical-velocity sign reversals (BRIEF.md §12.3).

    The definition is fully generic: a bounce/impact is the instant the vertical velocity
    component reverses sign (downward → upward, or the axis-flipped equivalent). No object
    semantics are involved. To be robust to tracker jitter the series is lightly smoothed
    before reversal detection, and reversals whose surrounding speed is below
    ``min_speed_fraction`` of the segment's vertical-speed scale are discarded.

    Emitted ``kind`` is ``"impact"`` by default (the more general term); set
    ``emit_kind="bounce"`` for callers that prefer that label. Each event carries the
    pre- and post-reversal vertical velocity, wrapped as :class:`Quantity`, in its
    payload.
    """

    smoothing_window: int = 3
    """Odd length of the moving-average pre-smoothing window (``1`` disables it)."""

    min_speed_fraction: float = 0.15
    """Reversal is ignored unless the larger surrounding |vertical velocity| exceeds this
    fraction of the segment's robust vertical-speed scale — the jitter rejector."""

    emit_kind: str = "impact"
    """Generic event ``kind`` to emit (``"impact"`` or ``"bounce"``)."""

    def detect(self, est: TrajectoryEstimate, track: TrackSequence) -> list[Event]:
        """Return generic bounce/impact events for one trajectory estimate."""
        v_raw = _vertical_velocity_series(est, track)
        if v_raw.shape[0] < 3:
            return []
        v = _smooth(v_raw, self.smoothing_window)

        # Robust vertical-speed scale (median absolute value), used for the jitter gate
        # and for normalizing reversal sharpness into a confidence factor.
        scale = float(np.median(np.abs(v)))
        if scale <= 0.0:
            scale = float(np.max(np.abs(v)))
        if scale <= 0.0:
            return []
        min_speed = self.min_speed_fraction * scale

        unit = _velocity_unit(est)
        tier = est.tier
        events: list[Event] = []
        for i in _bounce_indices(v, min_speed=min_speed):
            pre = float(v[i - 1])
            post = float(v[i])
            frame = _frame_for_index(est, track, i)
            confidence = self._confidence(pre, post, scale, est.goodness_of_fit)
            events.append(
                Event(
                    kind=self.emit_kind,
                    frame=frame,
                    confidence=confidence,
                    payload={
                        "pre_velocity": Quantity(
                            value=pre,
                            unit=unit,
                            tier=tier,
                            confidence=confidence,
                            source="finite_difference",
                            frame=frame,
                        ),
                        "post_velocity": Quantity(
                            value=post,
                            unit=unit,
                            tier=tier,
                            confidence=confidence,
                            source="finite_difference",
                            frame=frame,
                        ),
                        "axis": "vertical",
                    },
                )
            )
        return events

    def _confidence(self, pre: float, post: float, scale: float, gof: float) -> float:
        """Confidence from reversal sharpness, magnitude margin, and fit quality.

        * ``sharpness`` — total velocity change ``|pre - post|`` normalized by the
          segment speed scale; a crisp, large reversal is more trustworthy than a faint
          one near zero.
        * ``margin`` — how far the smaller surrounding magnitude sits above the jitter
          floor (a reversal flanked by two solid speeds beats one with one tiny side).
        * ``gof`` — the estimate's goodness-of-fit, so a bounce read off a poorly fit
          trajectory is discounted.

        Combined with :func:`combine_confidence` (weighted geometric mean) so any single
        weak factor pulls the result down — confidence is earned, not assumed (§10).
        """
        change = abs(pre - post)
        sharpness = float(np.clip(change / (2.0 * scale), 0.0, 1.0))
        smaller = min(abs(pre), abs(post))
        margin = float(np.clip(smaller / scale, 0.0, 1.0))
        gof_factor = float(np.clip(gof, 0.0, 1.0))
        return combine_confidence(
            sharpness, margin, gof_factor, weights=(0.5, 0.2, 0.3)
        )


# --------------------------------------------------------------------------------------
# Contact detection (two tracks)
# --------------------------------------------------------------------------------------


def _smooth(values: FloatArray, window: int) -> FloatArray:
    """Centered moving-average smoother; ``window <= 1`` is a no-op.

    Edges are handled by shrinking the window so the output keeps the input length and
    does not introduce phantom reversals at the boundaries.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if window <= 1 or n == 0:
        return values.copy()
    half = window // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.mean(values[lo:hi]))
    return out


def _bbox_diagonal(bbox: FloatArray) -> float:
    """Pixel diagonal length of an ``xyxy`` bounding box (its characteristic size)."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return float(np.hypot(x1 - x0, y1 - y0))


def _index_by_frame(track: TrackSequence) -> dict[int, int]:
    """Map each absolute frame index to its position in the track's detection list."""
    return {int(f): i for i, f in enumerate(track.frames)}


def detect_contacts(
    track_a: TrackSequence,
    track_b: TrackSequence,
    *,
    proximity_fraction: float = 1.0,
    velocity_jump_fraction: float = 0.5,
    min_separation_frames: int = 2,
) -> list[Event]:
    """Detect contacts between two tracks: proximity plus a velocity discontinuity.

    A contact is reported on a *shared* frame where (1) the two objects' pixel centers
    lie within a *scale-aware* proximity threshold, and (2) at least one object undergoes
    a sudden velocity change (a discontinuity) at that frame. The engine asserts only the
    kinematic fact of contact; what the contact *means* is decided by a domain layer
    (BRIEF.md §6).

    Scale awareness: the proximity threshold is ``proximity_fraction`` times the mean of
    the two bounding-box diagonals on that frame, so the same code works for large or
    small objects, near or far, at any tier — no metric reference required.

    Args:
        track_a, track_b: the two object tracks. They need not share an ``fps`` cadence
            beyond having overlapping frame indices.
        proximity_fraction: proximity threshold as a multiple of the mean bbox diagonal.
        velocity_jump_fraction: a velocity change counts as a discontinuity when its
            magnitude exceeds this fraction of the moving object's own speed scale.
        min_separation_frames: minimum frame gap required between successive contact
            events; a new event is suppressed unless ``frame - last_emitted`` is at least
            this value. The default of ``2`` forces a gap of two or more frames, so a
            single physical contact spanning adjacent frames (N and N+1) emits exactly
            once. Pass ``1`` to disable debouncing and emit on every qualifying frame.

    Returns:
        A list of generic ``"contact"`` events, ordered by frame, each carrying the pixel
        separation and the per-track velocity-jump magnitudes (as :class:`Quantity`).
    """
    idx_a = _index_by_frame(track_a)
    idx_b = _index_by_frame(track_b)
    shared = sorted(set(idx_a) & set(idx_b))
    if len(shared) < 2:
        return []

    centers_a = track_a.centers()
    centers_b = track_b.centers()
    jump_a, scale_a = _speed_jump_series(track_a)
    jump_b, scale_b = _speed_jump_series(track_b)

    events: list[Event] = []
    last_emitted: int | None = None
    for frame in shared:
        ia = idx_a[frame]
        ib = idx_b[frame]
        ca = centers_a[ia]
        cb = centers_b[ib]
        separation = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        size = 0.5 * (
            _bbox_diagonal(track_a.detections[ia].bbox)
            + _bbox_diagonal(track_b.detections[ib].bbox)
        )
        if size <= 0.0:
            continue
        threshold = proximity_fraction * size
        if separation > threshold:
            continue

        ja = float(jump_a[ia]) if ia < jump_a.shape[0] else 0.0
        jb = float(jump_b[ib]) if ib < jump_b.shape[0] else 0.0
        disc_a = scale_a > 0.0 and ja > velocity_jump_fraction * scale_a
        disc_b = scale_b > 0.0 and jb > velocity_jump_fraction * scale_b
        if not (disc_a or disc_b):
            continue

        if last_emitted is not None and (frame - last_emitted) < min_separation_frames:
            continue
        last_emitted = frame

        confidence = _contact_confidence(
            separation, threshold, ja, scale_a, jb, scale_b
        )
        events.append(
            Event(
                kind="contact",
                frame=frame,
                confidence=confidence,
                payload={
                    "separation": Quantity(
                        value=separation,
                        unit="px",
                        tier=Tier.PIXEL,
                        confidence=confidence,
                        source="finite_difference",
                        frame=frame,
                    ),
                    "velocity_jump_a": Quantity(
                        value=ja,
                        unit="px/s",
                        tier=Tier.PIXEL,
                        confidence=confidence,
                        source="finite_difference",
                        frame=frame,
                    ),
                    "velocity_jump_b": Quantity(
                        value=jb,
                        unit="px/s",
                        tier=Tier.PIXEL,
                        confidence=confidence,
                        source="finite_difference",
                        frame=frame,
                    ),
                    "track_ids": (
                        track_a.detections[ia].track_id,
                        track_b.detections[ib].track_id,
                    ),
                },
            )
        )
    return events


def _speed_jump_series(track: TrackSequence) -> tuple[FloatArray, float]:
    """Per-frame velocity-change magnitude and a robust speed scale for one track.

    Velocity is the pixel-center finite difference; the "jump" at frame ``i`` is the
    magnitude of the change in that velocity vector between ``i-1`` and ``i`` (a discrete
    acceleration), which spikes at a sudden direction/speed change. The scale is the
    median speed, used to make the discontinuity test scale-aware.
    """
    centers = track.centers()
    times = track.times()
    n = centers.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64), 0.0
    vx = _finite_difference(_column(centers, 0), times)
    vy = _finite_difference(_column(centers, 1), times)
    speed = np.hypot(vx, vy)
    jump = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        jump[i] = float(np.hypot(vx[i] - vx[i - 1], vy[i] - vy[i - 1]))
    scale = float(np.median(speed[speed > 0])) if np.any(speed > 0) else 0.0
    return jump, scale


def _contact_confidence(
    separation: float,
    threshold: float,
    jump_a: float,
    scale_a: float,
    jump_b: float,
    scale_b: float,
) -> float:
    """Confidence from proximity tightness and the strength of the velocity jump.

    * ``proximity`` — closer than the threshold scores higher (touching ≈ 1).
    * ``discontinuity`` — the strongest normalized velocity jump across the two tracks,
      saturating at 1; a sharp jump is strong evidence of a real interaction.
    Combined with :func:`combine_confidence` so a contact that is close but shows no
    kinematic change, or shows a change but is far, is appropriately discounted (§10).
    """
    proximity = float(np.clip(1.0 - separation / threshold, 0.0, 1.0)) if threshold > 0 else 0.0
    norm_a = jump_a / scale_a if scale_a > 0 else 0.0
    norm_b = jump_b / scale_b if scale_b > 0 else 0.0
    discontinuity = float(np.clip(max(norm_a, norm_b), 0.0, 1.0))
    return combine_confidence(proximity, discontinuity, weights=(0.4, 0.6))


# --------------------------------------------------------------------------------------
# Release detection (best-effort)
# --------------------------------------------------------------------------------------


@dataclass
class ReleaseDetector:
    """Best-effort: detect the onset of a free-flight segment (BRIEF.md §12.3).

    "Release" is the kinematic transition from non-free-flight motion into free flight —
    the first sample at which the vertical motion begins to follow constant downward
    acceleration (gravity). This is deliberately simple and clearly best-effort: we scan
    the vertical-velocity series for the start of a sustained, near-linear ramp (constant
    acceleration) and report its onset frame. It never claims metric tier; it only marks
    *where* free flight appears to begin.

    The detector keys off the estimate's own segment when that segment is already labeled
    a free-flight (``"ballistic"``) window — then release is simply that window's onset.
    Otherwise it searches the velocity series for the onset of constant acceleration.
    """

    min_run: int = 3
    """Minimum number of consecutive constant-acceleration samples to call it free
    flight (rejects a one-off blip)."""

    accel_consistency: float = 0.35
    """Maximum allowed relative spread of successive acceleration estimates within the
    run for it to count as "constant" (smaller = stricter)."""

    def detect(self, est: TrajectoryEstimate, track: TrackSequence) -> list[Event]:
        """Return at most one generic ``"release"`` event for this estimate."""
        seg = est.segment
        if seg is not None and seg.kind == "ballistic":
            frame = _frame_for_index(est, track, 0)
            confidence = combine_confidence(
                float(np.clip(est.goodness_of_fit, 0.0, 1.0)),
                0.7,
                weights=(0.7, 0.3),
            )
            return [
                Event(
                    kind="release",
                    frame=frame,
                    confidence=confidence,
                    payload={"onset": "segment_start", "segment_kind": seg.kind},
                )
            ]

        v = _vertical_velocity_series(est, track)
        if v.shape[0] < self.min_run + 1:
            return []
        times = _estimate_times(est, track)
        accel = _finite_difference(_smooth(v, 3), times)
        onset = self._first_constant_accel_run(accel)
        if onset is None:
            return []
        frame = _frame_for_index(est, track, onset)
        confidence = self._confidence(accel, onset, est.goodness_of_fit)
        return [
            Event(
                kind="release",
                frame=frame,
                confidence=confidence,
                payload={
                    "onset": "constant_acceleration",
                    "acceleration": Quantity(
                        value=float(np.median(accel[onset : onset + self.min_run])),
                        unit=_acceleration_unit(est),
                        tier=est.tier,
                        confidence=confidence,
                        source="finite_difference",
                        frame=frame,
                    ),
                },
            )
        ]

    def _first_constant_accel_run(self, accel: FloatArray) -> int | None:
        """Index of the first sample beginning a near-constant-acceleration run."""
        n = accel.shape[0]
        for start in range(0, n - self.min_run + 1):
            window = accel[start : start + self.min_run]
            mean = float(np.mean(window))
            if abs(mean) <= 0.0:
                continue
            spread = float(np.max(np.abs(window - mean)))
            if spread <= self.accel_consistency * abs(mean):
                return start
        return None

    def _confidence(self, accel: FloatArray, onset: int, gof: float) -> float:
        """Confidence from acceleration constancy over the run and the estimate fit."""
        window = accel[onset : onset + self.min_run]
        mean = float(np.mean(window))
        if mean == 0.0:
            return 0.0
        spread = float(np.max(np.abs(window - mean)))
        constancy = float(np.clip(1.0 - spread / abs(mean), 0.0, 1.0))
        gof_factor = float(np.clip(gof, 0.0, 1.0))
        return combine_confidence(constancy, gof_factor, weights=(0.6, 0.4))


# --------------------------------------------------------------------------------------
# Unit helpers — keep payload Quantities honest about tier-appropriate units
# --------------------------------------------------------------------------------------


def _velocity_unit(est: TrajectoryEstimate) -> str | None:
    """Velocity unit string matching an estimate's tier (no fabricated metric units)."""
    if est.velocity.unit is not None:
        return est.velocity.unit
    return {Tier.METRIC: "m/s", Tier.RELATIVE: None, Tier.PIXEL: "px/s"}[est.tier]


def _acceleration_unit(est: TrajectoryEstimate) -> str | None:
    """Acceleration unit string matching an estimate's tier."""
    return {Tier.METRIC: "m/s^2", Tier.RELATIVE: None, Tier.PIXEL: "px/s^2"}[est.tier]


__all__ = ["BounceDetector", "ReleaseDetector", "detect_contacts"]
