"""Relative-3D lift — the always-available tier floor (BRIEF.md §12.1).

When no metric scale can be earned (no free-flight segment to apply gravity-as-a-ruler,
and no caller-supplied ``reference_scale`` / ``reference_plane``), the engine must still
produce *something* useful and honest. This module produces a normalized, internally
consistent, **scale-free** 3D estimate at :class:`Tier.RELATIVE`. It never claims
:class:`Tier.METRIC` — that would violate "validated, not plausible" (BRIEF.md §10).

Method (deliberately simple and dependency-light — numpy only):

* **In-plane channels (x, y).** Pixel bbox centroids are normalized into a dimensionless
  ``[-1, 1]``-ish frame. If ``image_size`` is known we divide by the larger image
  dimension and recentre on the image middle, which keeps the aspect ratio and yields a
  stable, resolution-independent coordinate. If ``image_size`` is absent we fall back to
  normalizing by the observed pixel spread of the track itself.
* **Depth channel (z) — inverse-size proxy.** Under a pinhole camera the apparent size of
  a rigid object scales as ``1 / depth``. So a *larger* bounding box means *nearer*. We
  take a robust apparent-size scalar per frame (the geometric mean of bbox width and
  height), and define a relative inverse depth ``z_rel = s / median(s)`` so that ``z_rel``
  is dimensionless and centred near 1: values > 1 are nearer than the track's typical
  depth, values < 1 are farther. This is a *monotone, scale-free* depth ordering, not a
  metric distance — exactly what the RELATIVE tier promises.

Velocity is the finite-difference time derivative of these relative positions, expressed
per second (the time axis is real seconds, but the spatial axes are dimensionless, so the
unit is dimensionless-per-second and we honestly report ``unit=None``).

Confidence is derived (never hardcoded) from real cues via
:func:`trackphysics.core.provenance.combine_confidence`:

* **observation density** — fraction of the spanned frames actually observed (gaps hurt),
* **smoothness** — how much of the relative-position signal is coherent motion vs.
  frame-to-frame jitter (bursty noise hurts).
"""

from __future__ import annotations

import numpy as np

from .objsize import apparent_size_px
from .provenance import Quantity, Tier, TrajectoryEstimate, combine_confidence
from .schema import FloatArray, Segment, TrackSequence

__all__ = ["relative_lift"]

_EPS = 1e-9


def _normalized_xy(centers: FloatArray, image_size: tuple[int, int] | None) -> FloatArray:
    """Map pixel centroids to a dimensionless, scale-free in-plane frame, shape ``(T, 2)``.

    With a known ``image_size`` we recentre on the image middle and divide by the larger
    image dimension (preserving aspect ratio). Without it we recentre on the track's own
    mean and divide by the larger observed pixel spread, so the output stays bounded and
    resolution-independent either way.
    """
    if image_size is not None:
        width, height = image_size
        scale = float(max(width, height))
        scale = max(scale, _EPS)
        cx = 0.5 * float(width)
        cy = 0.5 * float(height)
        out = np.empty_like(centers)
        out[:, 0] = (centers[:, 0] - cx) / scale
        out[:, 1] = (centers[:, 1] - cy) / scale
        return out
    mean = centers.mean(axis=0)
    spread = centers.max(axis=0) - centers.min(axis=0)
    scale = max(float(spread.max()), _EPS)
    normalized: FloatArray = np.asarray((centers - mean) / scale, dtype=np.float64)
    return normalized


def _relative_depth(sizes: FloatArray) -> FloatArray:
    """Dimensionless inverse-depth proxy from apparent size, shape ``(T,)``.

    Normalized by the median apparent size so it is centred near 1 and scale-free. Larger
    apparent size -> nearer -> larger value (BRIEF.md §12.1).
    """
    median = float(np.median(sizes))
    median = max(median, _EPS)
    depth: FloatArray = np.asarray(sizes / median, dtype=np.float64)
    return depth


def _observation_density(track: TrackSequence) -> float:
    """Fraction of the spanned frames actually observed, in ``[0, 1]``.

    A track that skips many frames over its span is sparser and less trustworthy. Returns
    1.0 for a single observation (no span to be sparse over).
    """
    frames = track.frames
    if frames.size <= 1:
        return 1.0
    span = int(frames[-1] - frames[0]) + 1
    if span <= 0:
        return 1.0
    return float(min(1.0, frames.size / span))


def _smoothness(positions: FloatArray) -> float:
    """Coherent-motion fraction of a relative-position series, in ``[0, 1]``.

    Compares the energy of the second difference (a jitter/acceleration-noise proxy) to the
    energy of the first difference (overall motion). A clean trajectory has most of its
    energy in smooth motion and little in second-difference noise, so the score is near 1;
    a jittery one drops toward 0. Returns 1.0 for series too short to have curvature.
    """
    # Compute over finite rows only: a single non-finite coordinate (a dirty-tracker
    # NaN/inf bbox) would otherwise make motion/jitter NaN and propagate a NaN
    # goodness_of_fit that crashes TrajectoryEstimate's [0, 1] check (LIFT-NAN). Excluding
    # the bad rows degrades gracefully instead.
    finite = np.isfinite(positions).all(axis=1)
    pos = positions[finite]
    if pos.shape[0] < 3:
        return 0.0 if finite.sum() < positions.shape[0] else 1.0
    d1 = np.diff(pos, axis=0)
    d2 = np.diff(pos, n=2, axis=0)
    motion = float(np.sum(d1 * d1))
    jitter = float(np.sum(d2 * d2))
    if not (np.isfinite(motion) and np.isfinite(jitter)):
        return 0.0
    if motion <= _EPS:
        # No motion at all: a static, perfectly coherent (if uninformative) track.
        return 1.0
    return float(1.0 / (1.0 + jitter / motion))


def relative_lift(track: TrackSequence) -> TrajectoryEstimate:
    """Lift a 2D track to a scale-free relative-3D estimate (BRIEF.md §12.1).

    Produces a normalized / 2.5D position series ``(T, 3)`` and its time derivative, both
    at :class:`Tier.RELATIVE` with dimensionless units. This is the always-available floor:
    it makes no metric claim and never emits :class:`Tier.METRIC`.

    Assumptions and honesty notes:

    * The depth channel is an *inverse-size proxy* (apparent size ~ 1/depth under a pinhole
      camera). It is a monotone, scale-free depth *ordering*, not a metric distance.
    * In-plane axes are normalized pixel coordinates (by ``image_size`` if present, else by
      the track's own pixel spread). They are dimensionless.
    * Velocity is a finite difference over real seconds, but the spatial axes are
      dimensionless, so the reported unit is ``None`` (dimensionless-per-second).
    * Static-camera precondition applies (BRIEF.md §7); camera motion would corrupt both
      the in-plane and inverse-size channels.

    Args:
        track: The input track (boxes + ids; keypoints unused here).

    Returns:
        A :class:`TrajectoryEstimate` at :class:`Tier.RELATIVE` whose ``positions`` is a
        ``(T, 3)`` array-valued :class:`Quantity` and whose ``velocity`` is a ``(T, 3)``
        array-valued :class:`Quantity`, both dimensionless. ``goodness_of_fit`` reports the
        smoothness cue. For an empty track, returns empty ``(0, 3)`` arrays at zero
        confidence.

    Raises:
        ValueError: if ``fps`` is non-positive and no explicit timestamps are present
            (propagated from :meth:`TrackSequence.times`).
    """
    n = len(track)
    frames = track.frames
    frame_span: int | tuple[int, int] | None
    frame_span = None if n == 0 else (int(frames[0]), int(frames[-1]))

    if n == 0:
        empty_pos = np.empty((0, 3), dtype=np.float64)
        empty_vel = np.empty((0, 3), dtype=np.float64)
        positions = Quantity(
            value=empty_pos,
            unit=None,
            tier=Tier.RELATIVE,
            confidence=0.0,
            source="relative_lift",
            frame=frame_span,
        )
        velocity = Quantity(
            value=empty_vel,
            unit=None,
            tier=Tier.RELATIVE,
            confidence=0.0,
            source="finite_difference",
            frame=frame_span,
        )
        return TrajectoryEstimate(
            positions=positions,
            velocity=velocity,
            tier=Tier.RELATIVE,
            goodness_of_fit=0.0,
            segment=None,
            meta={"empty": True},
        )

    bboxes = track.bboxes()
    centers = track.centers()
    times = track.times()

    xy = _normalized_xy(centers, track.image_size)
    z = _relative_depth(apparent_size_px(bboxes))
    pos = np.empty((n, 3), dtype=np.float64)
    pos[:, :2] = xy
    pos[:, 2] = z

    # Finite-difference velocity over real seconds; central differences interior, one-sided
    # at the ends. np.gradient handles non-uniform spacing (variable fps) when given the
    # time axis, but needs at least two samples and strictly increasing times.
    vel: FloatArray
    if n >= 2 and np.all(np.diff(times) > 0):
        vel = np.asarray(np.gradient(pos, times, axis=0), dtype=np.float64)
        vel_source = "finite_difference"
        vel_degenerate = False
    else:
        vel = np.zeros_like(pos)
        vel_source = "finite_difference_degenerate"
        vel_degenerate = True

    density = _observation_density(track)
    smoothness = _smoothness(pos)
    confidence = combine_confidence(density, smoothness)
    # The degenerate branch returns a FABRICATED all-zeros velocity (no derivable motion:
    # <2 samples or non-increasing/duplicate times). It must NOT inherit the position
    # confidence — the velocity was not measured, so report it at zero confidence
    # (LIFT-DEGEN-VEL).
    vel_confidence = 0.0 if vel_degenerate else confidence

    segment = Segment(
        start_frame=int(frames[0]),
        end_frame=int(frames[-1]),
        kind="relative_lift",
        indices=np.arange(n, dtype=np.int64),
    )

    positions = Quantity(
        value=pos,
        unit=None,
        tier=Tier.RELATIVE,
        confidence=confidence,
        source="relative_lift",
        frame=frame_span,
    )
    velocity = Quantity(
        value=vel,
        unit=None,
        tier=Tier.RELATIVE,
        confidence=vel_confidence,
        source=vel_source,
        frame=frame_span,
    )
    return TrajectoryEstimate(
        positions=positions,
        velocity=velocity,
        tier=Tier.RELATIVE,
        goodness_of_fit=smoothness,
        segment=segment,
        meta={
            "observation_density": density,
            "smoothness": smoothness,
            "depth_model": "inverse_apparent_size",
            "scale_free": True,
        },
    )
