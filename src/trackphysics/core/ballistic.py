"""Free-flight ("ballistic") segment detection + gravity-as-a-ruler physics fit.

This module implements the heart of the v0.1 metric story (BRIEF.md §7, §10, §12.2):

1. :func:`detect_ballistic_segments` finds windows of a pixel track whose *motion
   signature* is consistent with constant-acceleration projectile motion — roughly
   constant vertical (image-``y``) acceleration and near-zero horizontal acceleration.

2. :func:`fit_ballistic` fits ``y_px(t) = y0 + vy0*t + 0.5*a_px*t^2`` over a segment with
   a RANSAC hypothesis search followed by IRLS (iteratively reweighted least squares)
   refinement, then recovers metric scale by **gravity-as-a-ruler**: under a static,
   roughly-horizontal camera with near-constant depth over the short arc, the world
   vertical axis maps to image-``y`` and a known gravitational acceleration ``g`` (m/s^2)
   pins the meters-per-pixel scale to ``s = g / |a_px|`` (px/s^2 -> m/px).

The provenance rules of BRIEF.md §10 are enforced strictly:

* Metric tier is emitted **only** when scale is genuinely earned — either a supplied
  ``reference_scale`` (or a derived scale from a ``reference_plane``), or a
  gravity-as-a-ruler fit that passes a physical-sanity gate (downward acceleration of
  plausible magnitude, low fit residual, enough inliers).
* When scale is not earned, the fit falls back **honestly** to a scale-free
  :data:`~trackphysics.core.provenance.Tier.RELATIVE` estimate (positions normalized to
  start, velocity in normalized units). It never fabricates metric.

Depth (the out-of-plane axis) is unknown at this monocular v0.1 tier; the third position
component is set to ``0`` with a ``meta`` note. Full 3D depth is a v0.2 / stereo concern.

Honest scope of the gate (BRIEF.md §10): single-arc gravity-as-a-ruler *assumes* ``g``
(``s = g / a_px``), so it cannot independently *recover* and cross-check ``g`` from one
arc. The gate verifies the motion is gravity-consistent (downward sign, plausible
magnitude, low residual, enough inliers) — not a recovered-``g`` tolerance. An independent
recovered-``g`` / cross-arc scale-consistency check is a v0.2 (multi-arc / stereo) item.

Only numpy + the trackphysics contract types are used (BRIEF.md §15). ``scipy`` is not
required here: the quadratic fit is a 3-parameter linear least-squares problem solved with
``numpy.linalg.lstsq``, which keeps the dependency surface minimal and Jetson-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .grounding import GroundingContext
from .objsize import apparent_size_px, read_object_size, scale_agreement
from .provenance import Quantity, Tier, TrajectoryEstimate, combine_confidence
from .schema import FloatArray, Segment, TrackSequence
from .shape import inplane_shape_features

BoolArray = npt.NDArray[np.bool_]
"""Alias for a boolean ndarray (inlier masks)."""

__all__ = [
    "QuadraticFit",
    "detect_ballistic_segments",
    "fit_ballistic",
    "irls_quadratic",
    "ransac_quadratic",
]

# --------------------------------------------------------------------------------------
# Tunables. Conservative defaults; the benchmark (BRIEF.md §14) calibrates these.
# --------------------------------------------------------------------------------------

_MIN_SEGMENT_POINTS = 7
"""Minimum observations for a credible quadratic fit (3 params + margin)."""

_SMOOTH_WINDOW = 5
"""Odd window length for the moving-average pre-smoother used in detection."""

# Physical-sanity bounds on the recovered vertical pixel acceleration. Real free flight
# has a clearly non-zero downward acceleration; anything near zero is linear motion and
# must NOT be promoted to metric (that is the honest-fallback path).
_MIN_ABS_PIXEL_ACCEL = 1.0
"""px/s^2: below this, vertical accel is indistinguishable from linear motion."""

# Fit-residual tolerance, expressed relative to the segment's vertical pixel extent. A
# good parabola hugs its observations; a large normalized residual fails the gate.
_RESIDUAL_FRACTION_TOL = 0.05
"""RMS residual / vertical extent above which the metric gate fails."""

_MIN_INLIER_FRACTION = 0.6
"""Fraction of segment points that must survive RANSAC+IRLS for a trusted fit."""

_SIZE_AGREEMENT_TOL = 0.25
"""Relative discrepancy between the size-ruler and gravity-ruler scales mapped to zero
agreement (the cross-check's confidence multiplier reaches 0 here). Calibrated on synthetic
sweeps (``bench/size_ruler.py``); refittable, not a universal constant (§10)."""

_SIZE_CI_WIDEN = 2.0
"""The metric CI's systematic half-width is scaled by ``1 + _SIZE_CI_WIDEN * rel_disc``
when an informative size ruler DISAGREES with the gravity scale — an independent cue that
the recovered scale is biased (typically by depth motion) widens the honest band."""

_SIZE_MAX_WIDEN_DISC = 10.0
"""Cap on the relative-discrepancy term that feeds the CI widening, so a pathological (e.g.
overflowing) ``rel_disc`` cannot produce a non-finite CI bound. At this point ``agreement``
is already 0, so the cap keeps the band finite without weakening the maximally-cautious
intent (a ~21x-widened systematic floor)."""

_SYSTEMATIC_REL_FLOOR = 0.08
"""Relative systematic-uncertainty floor on the recovered launch speed.

The fit-covariance CI captures *measurement* noise only. v0.1 also carries a *model*
discrepancy the fit cannot see: a pure-quadratic (drag-free) model and the
gravity-as-a-ruler assumptions (static, ~horizontal camera, near-constant depth). The
benchmark's model-mismatch curve puts realistic-drag error at a few percent, so the engine
must NOT claim sub-percent precision. This floor makes the emitted CI an honest *total*
uncertainty: cooperative conditions (modest drag) stay covered, while a gross
assumption violation (e.g. a pitched camera) exceeds it and is correctly flagged as
overconfident. Tightening this is a v0.2 item (drag-augmented fit, stereo)."""


@dataclass
class QuadraticFit:
    """Result of fitting ``y(t) = c0 + c1*t + c2*t^2`` with robust inlier selection.

    All quantities are in the *input* units of the fit (pixels and seconds when fed pixel
    centers and times). ``accel`` is ``2 * c2`` (the second derivative).
    """

    c0: float
    c1: float
    c2: float
    accel: float
    """Second time-derivative ``d^2 y / d t^2`` = ``2 * c2`` (px/s^2 for pixel input)."""

    inlier_mask: FloatArray
    """Boolean-as-float mask (1.0 inlier / 0.0 outlier), shape ``(N,)``."""

    rms_residual: float
    """Root-mean-square fit residual over the inliers, in input units (e.g. px)."""

    def predict(self, t: FloatArray) -> FloatArray:
        """Evaluate the fitted quadratic at times ``t``."""
        return self.c0 + self.c1 * t + self.c2 * t * t


def _moving_average(x: FloatArray, window: int) -> FloatArray:
    """Centered moving average with edge clamping; returns an array of the same length.

    Used only as a *detection* pre-smoother to tame jitter before differentiating. The
    physics fit itself operates on the raw observations (smoothing biases derivatives).
    """
    n = x.shape[0]
    if window <= 1 or n < window:
        return x.astype(np.float64, copy=True)
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(x, (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _second_difference(values: FloatArray, times: FloatArray) -> FloatArray:
    """Local second derivative via three-point finite differences on possibly uneven t.

    Returns an array of length ``len(values) - 2`` aligned with the interior points.
    """
    n = values.shape[0]
    if n < 3:
        return np.empty(0, dtype=np.float64)
    out = np.empty(n - 2, dtype=np.float64)
    for k in range(1, n - 1):
        t_prev, t_curr, t_next = times[k - 1], times[k], times[k + 1]
        h1 = t_curr - t_prev
        h2 = t_next - t_curr
        denom = 0.5 * (h1 + h2) * h1 * h2
        if denom <= 0.0:
            out[k - 1] = np.nan
            continue
        # Non-uniform three-point second derivative.
        out[k - 1] = (
            h2 * values[k - 1] - (h1 + h2) * values[k] + h1 * values[k + 1]
        ) / denom
    return out


def detect_ballistic_segments(
    track: TrackSequence,
    *,
    min_points: int = _MIN_SEGMENT_POINTS,
    accel_consistency_tol: float = 0.45,
    horizontal_accel_fraction: float = 0.5,
) -> list[Segment]:
    """Find free-flight segments by motion signature (BRIEF.md §12.2).

    A free-flight run is a contiguous window where the vertical (image-``y``)
    acceleration is approximately *constant* (consistent with constant-acceleration
    projectile motion) and the horizontal (image-``x``) acceleration is *near zero*
    relative to the vertical one. Detection smooths the pixel centers, estimates the
    per-point vertical and horizontal accelerations by finite differences, and grows the
    longest window satisfying both conditions.

    Args:
        track: The input track. Short tracks (fewer than ``min_points`` detections after
            interior trimming) yield ``[]``.
        min_points: Minimum observations for a returned segment.
        accel_consistency_tol: Allowed spread of the vertical acceleration, as a fraction
            of its median magnitude, for the window to count as "constant accel".
        horizontal_accel_fraction: Max ratio of median ``|a_x|`` to median ``|a_y|`` for
            the window to count as "horizontal accel near zero".

    Returns:
        A list of ``Segment(kind="ballistic", indices=...)`` covering the detected runs,
        ordered by start index. Empty if no qualifying window exists.
    """
    n = len(track)
    if n < max(min_points, 3):
        return []

    centers = track.centers()
    times = track.times()
    xs = _moving_average(centers[:, 0], _SMOOTH_WINDOW)
    ys = _moving_average(centers[:, 1], _SMOOTH_WINDOW)

    ay = _second_difference(ys, times)  # length n-2, aligned to interior indices 1..n-2
    ax = _second_difference(xs, times)
    if ay.size == 0:
        return []

    valid = np.isfinite(ay) & np.isfinite(ax)
    abs_ay = np.abs(ay)
    abs_ax = np.abs(ax)

    # Per-interior-point boolean: vertical accel is appreciable, DOWNWARD (gravity-
    # consistent: image-y increases), and dominates horizontal. The signed `ay >= floor`
    # (not abs) keeps anti-gravity runs from ever being proposed as ballistic, consistent
    # with the signed metric gate.
    median_ay = float(np.median(abs_ay[valid])) if np.any(valid) else 0.0
    accel_floor = max(_MIN_ABS_PIXEL_ACCEL, 0.25 * median_ay)
    point_ok = valid & (ay >= accel_floor) & (abs_ax <= horizontal_accel_fraction * abs_ay)

    segments: list[Segment] = []
    interior_start = 1  # interior point k corresponds to detection index k

    run_start: int | None = None
    for k in range(point_ok.size):
        if point_ok[k]:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None:
                seg = _finalize_run(
                    track,
                    times,
                    ys,
                    run_start + interior_start,
                    k - 1 + interior_start,
                    ay[run_start : k],
                    min_points,
                    accel_consistency_tol,
                )
                if seg is not None:
                    segments.append(seg)
                run_start = None
    if run_start is not None:
        seg = _finalize_run(
            track,
            times,
            ys,
            run_start + interior_start,
            point_ok.size - 1 + interior_start,
            ay[run_start:],
            min_points,
            accel_consistency_tol,
        )
        if seg is not None:
            segments.append(seg)

    return segments


def _finalize_run(
    track: TrackSequence,
    times: FloatArray,
    ys: FloatArray,
    lo_idx: int,
    hi_idx: int,
    run_accels: FloatArray,
    min_points: int,
    accel_consistency_tol: float,
) -> Segment | None:
    """Validate one candidate run and, if it qualifies, build a Segment.

    Expands the interior run by one point on each side (interior accel indices exclude the
    window endpoints) so the returned segment includes the full observed arc.
    """
    start = max(0, lo_idx - 1)
    end = min(len(track) - 1, hi_idx + 1)
    if end - start + 1 < min_points:
        return None

    finite = run_accels[np.isfinite(run_accels)]
    if finite.size == 0:
        return None
    median = float(np.median(np.abs(finite)))
    if median < _MIN_ABS_PIXEL_ACCEL:
        return None
    spread = float(np.std(finite))
    if median > 0 and spread / median > accel_consistency_tol:
        return None

    frames = track.frames
    indices = np.arange(start, end + 1, dtype=np.int64)
    return Segment(
        start_frame=int(frames[start]),
        end_frame=int(frames[end]),
        kind="ballistic",
        indices=indices,
        source_track=track,  # stateless fit: the segment carries its own track
    )


def ransac_quadratic(
    t: FloatArray,
    y: FloatArray,
    *,
    n_iters: int = 200,
    residual_tol: float | None = None,
    min_inliers: int | None = None,
    rng: np.random.Generator | None = None,
) -> QuadraticFit:
    """RANSAC hypothesis search for ``y(t) = c0 + c1*t + c2*t^2``.

    Repeatedly samples 3 points (the minimal set for a quadratic), fits the exact
    quadratic, and counts inliers within ``residual_tol``. The hypothesis with the most
    inliers wins; a final least-squares re-fit over those inliers produces the returned
    coefficients. This isolates a clean free-flight arc from sparse false positives /
    outliers before IRLS refines it (BRIEF.md §11).

    Args:
        t: Sample times, shape ``(N,)``.
        y: Observations, shape ``(N,)``.
        n_iters: Number of random minimal-sample hypotheses.
        residual_tol: Inlier threshold on ``|y - y_hat|``. Defaults to a robust estimate
            (``2 * MAD`` of residuals from an initial all-points fit, floored at 1.0).
        min_inliers: Minimum inliers for a hypothesis to be considered. Defaults to
            ``max(3, ceil(0.5 * N))``.
        rng: Optional numpy ``Generator`` for deterministic sampling.

    Returns:
        A :class:`QuadraticFit` over the best inlier set.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = t.shape[0]
    if n < 3:
        raise ValueError("ransac_quadratic needs at least 3 points")
    if rng is None:
        rng = np.random.default_rng(0)

    base = _lstsq_quadratic(t, y)
    base_resid = np.abs(y - base.predict(t))
    if residual_tol is None:
        mad = float(np.median(np.abs(base_resid - np.median(base_resid))))
        residual_tol = max(1.0, 2.0 * 1.4826 * mad)
    if min_inliers is None:
        min_inliers = max(3, int(np.ceil(0.5 * n)))

    if n == 3:
        return base

    best_mask = base_resid <= residual_tol
    best_count = int(np.count_nonzero(best_mask))

    indices = np.arange(n)
    for _ in range(n_iters):
        sample = rng.choice(indices, size=3, replace=False)
        ts, yss = t[sample], y[sample]
        if np.unique(ts).size < 3:
            continue
        coeffs = np.polyfit(ts, yss, 2)
        pred = np.polyval(coeffs, t)
        resid = np.abs(y - pred)
        mask = resid <= residual_tol
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_count < min_inliers:
        # No clean consensus; fall back to the all-points fit so the caller still gets a
        # usable (if lower-confidence) estimate. The gate downstream handles trust.
        return base
    return _lstsq_quadratic(t[best_mask], y[best_mask], full_mask=best_mask)


def irls_quadratic(
    t: FloatArray,
    y: FloatArray,
    *,
    seed: QuadraticFit | None = None,
    n_iters: int = 10,
    tukey_c: float = 4.685,
    rng: np.random.Generator | None = None,
) -> QuadraticFit:
    """Iteratively reweighted least-squares refinement of a quadratic fit.

    Down-weights large-residual observations using Tukey's biweight, re-fitting until the
    weights stabilize. This is the smooth complement to RANSAC's hard inlier selection:
    RANSAC picks the arc, IRLS polishes it while suppressing residual jitter / mild
    outliers (BRIEF.md §11).

    Args:
        t: Sample times, shape ``(N,)``.
        y: Observations, shape ``(N,)``.
        seed: Optional starting fit; if ``None``, RANSAC provides the seed.
        n_iters: Maximum reweighting iterations.
        tukey_c: Tukey biweight tuning constant (in units of robust residual scale).
        rng: Forwarded to RANSAC when ``seed`` is ``None``.

    Returns:
        The refined :class:`QuadraticFit`; ``inlier_mask`` marks points retaining
        non-negligible weight.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = t.shape[0]
    if n < 3:
        raise ValueError("irls_quadratic needs at least 3 points")

    fit = seed if seed is not None else ransac_quadratic(t, y, rng=rng)
    weights = np.ones(n, dtype=np.float64)
    design = np.vstack([np.ones(n), t, t * t]).T

    for _ in range(n_iters):
        resid = y - fit.predict(t)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = max(1e-9, 1.4826 * mad)
        u = resid / (tukey_c * scale)
        new_weights = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if np.count_nonzero(new_weights) < 3:
            break
        coeffs = _weighted_lstsq(design, y, new_weights)
        c0, c1, c2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        new_fit = _make_fit(t, y, c0, c1, c2, mask=new_weights > 1e-6)
        if np.allclose(new_weights, weights, atol=1e-6):
            fit, weights = new_fit, new_weights
            break
        fit, weights = new_fit, new_weights

    return fit


def _lstsq_quadratic(
    t: FloatArray,
    y: FloatArray,
    *,
    full_mask: BoolArray | None = None,
) -> QuadraticFit:
    """Unweighted least-squares quadratic fit.

    Fits over the points in ``t``/``y``. ``full_mask`` is an optional boolean mask over a
    *larger* parent array (the original N points) recording which were used as inliers; it
    is carried through to the returned fit's ``inlier_mask`` for reporting. When omitted,
    every supplied point is treated as an inlier.
    """
    design = np.vstack([np.ones_like(t), t, t * t]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    c0, c1, c2 = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    pred = c0 + c1 * t + c2 * t * t
    resid = y - pred
    rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("inf")
    if full_mask is not None:
        out_mask = np.asarray(full_mask, dtype=np.float64)
    else:
        out_mask = np.ones(t.shape[0], dtype=np.float64)
    return QuadraticFit(c0=c0, c1=c1, c2=c2, accel=2.0 * c2, inlier_mask=out_mask,
                        rms_residual=rms)


def _weighted_lstsq(design: FloatArray, y: FloatArray, weights: FloatArray) -> FloatArray:
    """Solve a weighted linear least-squares problem ``min sum w_i (X b - y)_i^2``."""
    sqrt_w = np.sqrt(weights)[:, None]
    coeffs, *_ = np.linalg.lstsq(design * sqrt_w, y * sqrt_w[:, 0], rcond=None)
    return np.asarray(coeffs, dtype=np.float64)


def _make_fit(
    t: FloatArray,
    y: FloatArray,
    c0: float,
    c1: float,
    c2: float,
    *,
    mask: BoolArray | None = None,
) -> QuadraticFit:
    """Assemble a :class:`QuadraticFit`, computing RMS residual over the inliers.

    ``mask`` (same length as ``t``) marks which points count as inliers for the residual
    and for the reported ``inlier_mask``. When ``None``, all points are inliers.
    """
    pred = c0 + c1 * t + c2 * t * t
    resid = y - pred
    bool_mask = (
        np.ones(t.shape[0], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    inliers = resid[bool_mask]
    rms = float(np.sqrt(np.mean(inliers**2))) if inliers.size else float("inf")
    return QuadraticFit(
        c0=c0,
        c1=c1,
        c2=c2,
        accel=2.0 * c2,
        inlier_mask=bool_mask.astype(np.float64),
        rms_residual=rms,
    )


def _scale_from_reference(ctx: GroundingContext) -> float | None:
    """Return a supplied metric scale (meters-per-pixel) if the context provides one.

    A ``reference_scale`` is taken directly. A ``reference_plane`` alone does not, by
    itself, fix a pixel-to-meter scale without camera intrinsics (a v0.2 concern), so it
    is *not* converted to a scalar scale here; the gravity-as-a-ruler path handles the
    monocular case. If only a plane is given, this returns ``None`` and the engine relies
    on gravity-as-a-ruler.
    """
    if ctx.reference_scale is not None:
        return float(ctx.reference_scale)
    return None


def fit_ballistic(
    track: TrackSequence,
    segment: Segment,
    ctx: GroundingContext,
    *,
    diameter_m: float | None = None,
    rng: np.random.Generator | None = None,
) -> TrajectoryEstimate:
    """Fit a ballistic arc and recover metric scale by gravity-as-a-ruler (BRIEF.md §10).

    Pipeline:

    1. Slice the segment's pixel centers and times from ``track``.
    2. Fit ``y_px(t)`` with RANSAC (arc selection) + IRLS (robust refinement).
    3. Determine scale:

       * If ``ctx.reference_scale`` is supplied -> use it directly; tier ``METRIC``,
         ``source="reference_scale"``.
       * Else gravity-as-a-ruler: ``s = gravity / |a_px|`` (m/px). Emit ``METRIC`` **only**
         if the physical-sanity gate passes (downward accel of plausible magnitude, low
         normalized residual, enough inliers).
       * Otherwise fall back **honestly** to a scale-free ``RELATIVE`` estimate.

    4. Build position (``(T, 3)``) and velocity (``(T, 3)``) Quantities. In-plane axes use
       ``s * (pixel displacement from the segment start)``; the depth axis is ``0`` (a v0.2
       / stereo concern), noted in ``meta``. Velocity is differentiated from positions.

    The ``confidence`` is derived from real factors via
    :func:`~trackphysics.core.provenance.combine_confidence` — never hardcoded.

    Args:
        track: The track the segment indexes into.
        segment: A segment (typically ``kind="ballistic"``) with materialized ``indices``.
        ctx: Grounding context; supplies ``gravity`` and any metric reference.
        diameter_m: Optional known physical object size (meters), supplied by a preset
            (e.g. ``SpherePreset.diameter_m``). When given AND the boxes carry apparent
            size, the engine runs the **object-size-as-ruler §10 cross-cue check**: it
            derives an independent meters-per-pixel scale ``D / d_px`` and compares it to the
            gravity-recovered scale. The check is ONE-SIDED conservative — disagreement (with
            an informative size channel) lowers confidence and widens the CI; agreement never
            inflates them — so it can only make the engine more cautious, never over-claim.
            ``None`` (the default) ⇒ no size logic, behaviour unchanged.
        rng: Optional generator for deterministic RANSAC.

    Returns:
        A :class:`TrajectoryEstimate` at ``METRIC`` tier on success, else ``RELATIVE``.
    """
    idx = _segment_indices(track, segment)
    seg_centers = track.centers()[idx]
    seg_times = track.times()[idx]

    # Drop non-finite observations (NaN/inf centers — a routine dirty-tracker condition)
    # BEFORE fitting. A single non-finite value poisons the least-squares fit so that
    # ``a_px``/residual become NaN; because every comparison against NaN is False, such a
    # fit slips through the metric sanity gate and would emit a NaN-scale METRIC result —
    # the cardinal §10 sin — or crash downstream. Detection masks non-finite (lines ~208,
    # ~280); the fit path must too. We fit only the finite observations and fall back
    # honestly if too few remain (BAL-NAN-GATE).
    finite = np.isfinite(seg_centers).all(axis=1) & np.isfinite(seg_times)
    kept = idx[finite]
    centers = seg_centers[finite]
    if kept.size:
        times = seg_times[finite] - seg_times[finite][0]  # time relative to first kept obs
        frame_range = (int(track.frames[kept[0]]), int(track.frames[kept[-1]]))
    else:
        times = np.zeros(0, dtype=np.float64)
        frame_range = (int(segment.start_frame), int(segment.end_frame))

    if centers.shape[0] < 3:
        # Too short (or too few finite points) to fit anything physical: honest fallback.
        return _relative_fallback(centers, times, frame_range, segment, reason="too_few_points")

    # Robustly fit BOTH image axes (RANSAC arc selection + IRLS refinement). The emitted
    # trajectory is built from THESE fits (see _build_metric_estimate), so points the
    # robust fit down-weighted as outliers do not leak into the returned product — the
    # robustness machinery cleans the actual output, not merely the scale/confidence
    # (BAL-RAWPOS, BRIEF.md §11).
    yfit = irls_quadratic(times, centers[:, 1], rng=rng)
    xfit = irls_quadratic(times, centers[:, 0], rng=rng)

    a_px = yfit.accel
    n_seg = centers.shape[0]
    inlier_count = int(np.count_nonzero(yfit.inlier_mask > 0.5))
    inlier_fraction = inlier_count / float(n_seg)
    vertical_extent = float(np.ptp(centers[:, 1])) or 1.0
    residual_fraction = yfit.rms_residual / vertical_extent
    # Observation density / completeness over the segment SPAN (BRIEF.md §10): the
    # fraction of spanned frames actually observed. Distinct from inlier survival — a
    # clean parabola with large temporal gaps must not earn maximal confidence.
    span = frame_range[1] - frame_range[0] + 1
    completeness = float(np.clip(n_seg / span, 0.0, 1.0)) if span > 0 else 1.0

    # Sub-confidence factors (BRIEF.md §10): each in [0, 1].
    residual_factor = float(np.clip(1.0 - residual_fraction / _RESIDUAL_FRACTION_TOL, 0.0, 1.0))
    inlier_factor = float(np.clip(inlier_fraction, 0.0, 1.0))

    # Scale-invariant 2D shape descriptors of the segment (domain-agnostic geometry). Exposed
    # in the metric meta so a per-deployment calibrator / the depth guard can read the same
    # observable features the engine saw, without recomputing from the track.
    shape = inplane_shape_features(centers)

    supplied_scale = _scale_from_reference(ctx)
    if supplied_scale is not None:
        # Scale is given outright -> METRIC, with confidence/gof from FIT quality only.
        # No gravitational sanity is evaluated on this path, so none is fabricated into the
        # score (PROV-03).
        confidence = combine_confidence(residual_factor, inlier_factor, completeness)
        gof = _goodness_of_fit(residual_fraction, inlier_fraction, completeness=completeness)
        return _build_metric_estimate(
            xfit, yfit, times, supplied_scale, frame_range, segment,
            source="reference_scale", confidence=confidence, gof=gof,
            meta={"scale_m_per_px": supplied_scale, "a_px": a_px, "completeness": completeness,
                  "residual_fraction": residual_fraction, "inlier_fraction": inlier_fraction,
                  **shape},
        )

    # Gravity-as-a-ruler. The world-vertical maps to image-y; a_px is px/s^2 (downward +).
    sane, sanity_margin = _passes_sanity_gate(
        a_px=a_px,
        residual_fraction=residual_fraction,
        inlier_fraction=inlier_fraction,
    )
    if not sane:
        return _relative_fallback(
            centers, times, frame_range, segment, reason="sanity_gate_failed",
            a_px=a_px, residual_fraction=residual_fraction, inlier_fraction=inlier_fraction,
            xfit=xfit, yfit=yfit,
        )

    scale = ctx.gravity / a_px  # meters per pixel; a_px > 0 guaranteed by the signed gate
    if not (np.isfinite(scale) and scale > 0.0):
        # Defense in depth: the gate guarantees a finite a_px >= floor, so this should be
        # unreachable, but a non-finite scale must never reach _build_metric_estimate.
        return _relative_fallback(
            centers, times, frame_range, segment, reason="non_finite_scale",
            a_px=a_px, residual_fraction=residual_fraction, inlier_fraction=inlier_fraction,
            xfit=xfit, yfit=yfit,
        )
    sanity_factor = float(np.clip(sanity_margin, 0.0, 1.0))
    confidence = combine_confidence(residual_factor, inlier_factor, completeness, sanity_factor)
    gof = _goodness_of_fit(
        residual_fraction, inlier_fraction, sanity_margin=sanity_margin, completeness=completeness
    )
    guard_meta: dict[str, object] = {}
    systematic_floor = _SYSTEMATIC_REL_FLOOR

    # Opt-in point-only depth-domination guard (§10 tier-hole, BAL-DEPTH-GUARD). An object
    # flying along the optical axis still fits a clean in-plane parabola, so the gate above
    # trusts it — but the in-plane metric then captures only a fraction of the true 3D motion.
    # The in-plane aspect ratio (horizontal/vertical pixel extent) is a point-only proxy: low
    # aspect => depth-dominated. CONTINUOUS confidence discount + CI widening as it rises, with
    # a CONSERVATIVE hard downgrade past the 'hopeless' floor. Default-OFF; behaviour unchanged
    # unless a caller supplies an enabled guard. Flagging depth-domination is NOT recovering
    # depth (that needs stereo/size).
    guard = ctx.depth_guard
    if guard is not None and guard.enabled:
        aspect = shape["aspect"]
        if aspect <= guard.hard_aspect:
            return _relative_fallback(
                centers, times, frame_range, segment, reason="depth_domination_downgrade",
                a_px=a_px, residual_fraction=residual_fraction, inlier_fraction=inlier_fraction,
                xfit=xfit, yfit=yfit,
            )
        score = guard.depth_score(aspect)
        confidence = confidence * (1.0 - guard.confidence_penalty * score)
        systematic_floor = _SYSTEMATIC_REL_FLOOR * (1.0 + guard.ci_widen * score)
        guard_meta = {"in_plane_aspect": aspect, "depth_domination_score": score}

    # Object-size-as-ruler cross-cue consistency check (§10). When a known object size is
    # supplied, a second, independent meters-per-pixel scale ``D / d_px`` is available from
    # the apparent box size. It is folded in ONE-SIDED: an informative size channel that
    # DISAGREES with the gravity scale lowers confidence and widens the CI (it is evidence
    # the gravity scale is biased, typically by depth motion); agreement never inflates them.
    # A sub-noise size channel is recorded as uninformative and changes nothing — no signal,
    # no claim. Flagging a biased scale this way is distinct from RECOVERING depth (stereo).
    size_meta: dict[str, object] = {}
    if diameter_m is not None:
        kept_bboxes = track.bboxes()[kept]
        reading = read_object_size(apparent_size_px(kept_bboxes), diameter_m)
        if reading is None:
            size_meta = {"object_size_ruler": "unavailable"}
        elif not reading.informative_for_scale:
            size_meta = {
                "object_size_ruler": "sub_noise",
                "size_scale_m_per_px": reading.scale_m_per_px,
                "size_rel_noise": reading.rel_noise,
                "size_snr_abs": reading.snr_abs,
                "size_snr_dyn": reading.snr_dyn,
            }
        else:
            rel_disc, agreement = scale_agreement(
                reading.scale_m_per_px, scale, tol=_SIZE_AGREEMENT_TOL
            )
            confidence = confidence * agreement  # one-sided: max 1.0 (no boost)
            # Cap the widening at a large but FINITE multiple so a pathological (overflowing)
            # rel_disc cannot emit a non-finite CI bound. agreement is already 0 in that
            # regime, so this preserves the maximally-cautious intent while staying finite.
            widen_disc = rel_disc if np.isfinite(rel_disc) else _SIZE_MAX_WIDEN_DISC
            widen_disc = min(widen_disc, _SIZE_MAX_WIDEN_DISC)
            systematic_floor = systematic_floor * (1.0 + _SIZE_CI_WIDEN * widen_disc)
            size_meta = {
                "object_size_ruler": "cross_checked",
                "size_scale_m_per_px": reading.scale_m_per_px,
                "gravity_scale_m_per_px": scale,
                "scale_cross_check_rel_disc": rel_disc,
                "scale_agreement": agreement,
                "size_snr_abs": reading.snr_abs,
                "size_snr_dyn": reading.snr_dyn,
                "size_informative_for_depth": reading.informative_for_depth,
            }

    return _build_metric_estimate(
        xfit, yfit, times, scale, frame_range, segment,
        source="ballistic_fit", confidence=confidence, gof=gof,
        systematic_rel_floor=systematic_floor,
        meta={
            "scale_m_per_px": scale,
            "a_px": a_px,
            "gravity": ctx.gravity,
            "inlier_fraction": inlier_fraction,
            "residual_fraction": residual_fraction,
            "completeness": completeness,
            **shape,
            **guard_meta,
            **size_meta,
        },
    )


def _segment_indices(track: TrackSequence, segment: Segment) -> FloatArray:
    """Resolve the detection indices a segment covers (materialized or by frame range)."""
    if segment.indices is not None:
        return np.asarray(segment.indices, dtype=np.int64)
    frames = track.frames
    mask = (frames >= segment.start_frame) & (frames <= segment.end_frame)
    return np.nonzero(mask)[0].astype(np.int64)


def _passes_sanity_gate(
    *, a_px: float, residual_fraction: float, inlier_fraction: float
) -> tuple[bool, float]:
    """Physical-sanity gate for promoting a fit to METRIC (BRIEF.md §10).

    Requires (1) an appreciable vertical acceleration in the **gravity-consistent
    direction** — under the schema's image convention (``y`` points down) free fall
    accelerates *downward*, i.e. ``a_px`` must be positive and at least
    ``_MIN_ABS_PIXEL_ACCEL``; an upward (negative) vertical acceleration is physically
    anti-gravitational and is NOT free flight, so it must fall back to RELATIVE rather than
    have ``g`` used as its ruler — (2) a low normalized fit residual, and (3) enough
    surviving inliers.

    Returns ``(passes, sanity_margin)`` where ``sanity_margin`` in ``[0, 1]`` summarizes
    how comfortably the residual and inlier conditions are met (used as a confidence cue).
    """
    # Fail CLOSED on non-finite inputs. A NaN/inf ``a_px`` or residual (from a fit poisoned
    # by a bad observation) compares False against every threshold below, so without this
    # guard the gate would PASS and ``g / a_px`` would emit a NaN-scale METRIC result
    # (BAL-NAN-GATE). A fit that cannot be evaluated has not earned metric scale.
    if not (np.isfinite(a_px) and np.isfinite(residual_fraction) and np.isfinite(inlier_fraction)):
        return False, 0.0
    # SIGNED check: must accelerate downward (image-y increases) AND appreciably. Using
    # abs() here would promote upward-accelerating (anti-gravity) arcs to METRIC — the
    # cardinal §10 sin of earning scale from a motion that is not gravitational free fall.
    if a_px < _MIN_ABS_PIXEL_ACCEL:
        return False, 0.0
    if residual_fraction > _RESIDUAL_FRACTION_TOL:
        return False, 0.0
    if inlier_fraction < _MIN_INLIER_FRACTION:
        return False, 0.0
    residual_margin = 1.0 - residual_fraction / _RESIDUAL_FRACTION_TOL
    inlier_margin = (inlier_fraction - _MIN_INLIER_FRACTION) / (1.0 - _MIN_INLIER_FRACTION)
    margin = float(np.clip(0.5 * (residual_margin + inlier_margin), 0.0, 1.0))
    return True, margin


def _goodness_of_fit(
    residual_fraction: float,
    inlier_fraction: float,
    *,
    sanity_margin: float | None = None,
    completeness: float | None = None,
) -> float:
    """Average the plausibility terms ACTUALLY measured on the calling path into ``[0, 1]``.

    ``residual_score`` and ``inlier_fraction`` always count. ``sanity_margin`` counts only
    when a gravitational-sanity check was performed (the gravity-as-a-ruler path) — the
    supplied-reference-scale path performs no such check, so no full-marks sanity term is
    fabricated there (PROV-03). ``completeness`` counts when provided.
    """
    residual_score = float(np.clip(1.0 - residual_fraction / _RESIDUAL_FRACTION_TOL, 0.0, 1.0))
    terms = [residual_score, float(np.clip(inlier_fraction, 0.0, 1.0))]
    if sanity_margin is not None:
        terms.append(float(np.clip(sanity_margin, 0.0, 1.0)))
    if completeness is not None:
        terms.append(float(np.clip(completeness, 0.0, 1.0)))
    return float(np.clip(sum(terms) / len(terms), 0.0, 1.0))


def _finite_difference_velocity(positions: FloatArray, times: FloatArray) -> FloatArray:
    """Per-axis velocity via numpy gradient on possibly non-uniform time samples."""
    if positions.shape[0] < 2:
        return np.zeros_like(positions)
    if bool(np.any(np.diff(times) <= 0)):
        # Non-increasing times (duplicate/decreasing frames) would make np.gradient divide
        # by a zero dt and emit inf/nan velocity. Degrade to zeros, mirroring lift.py's guard
        # (the schema now rejects duplicate frames, so this is defence in depth for direct
        # callers of fit_ballistic).
        return np.zeros_like(positions)
    vel = np.empty_like(positions)
    for axis in range(positions.shape[1]):
        vel[:, axis] = np.gradient(positions[:, axis], times, edge_order=1)
    return vel


def _fitted_positions_velocity(
    xfit: QuadraticFit, yfit: QuadraticFit, times: FloatArray, scale: float
) -> tuple[FloatArray, FloatArray]:
    """Positions (scaled) and analytic velocity from the robust per-axis fits.

    Positions are read off the fitted curves (referenced to the segment start), so
    down-weighted outliers do not appear in the product. Velocity is the analytic
    derivative ``v = c1 + 2*c2*t`` of each fit — not ``np.gradient`` of noisy raw centers,
    which would re-inject the very spikes the robust fit removed (BAL-RAWPOS).
    """
    n = times.shape[0]
    x_pred = xfit.predict(times)
    y_pred = yfit.predict(times)
    positions = np.zeros((n, 3), dtype=np.float64)
    positions[:, 0] = scale * (x_pred - x_pred[0])
    positions[:, 1] = scale * (y_pred - y_pred[0])
    velocity = np.zeros((n, 3), dtype=np.float64)
    velocity[:, 0] = scale * (xfit.c1 + 2.0 * xfit.c2 * times)
    velocity[:, 1] = scale * (yfit.c1 + 2.0 * yfit.c2 * times)
    return positions, velocity


def _coeff_cov(
    times: FloatArray, inlier_mask: FloatArray, rms_residual: float
) -> FloatArray | None:
    """Covariance of the quadratic coefficients ``[c0, c1, c2]`` over the inlier set.

    Standard OLS result ``Cov = sigma^2 (A^T A)^-1`` with ``A`` the Vandermonde design over
    inlier times and ``sigma^2`` the dof-corrected residual variance. Returns ``None`` when
    there are too few inliers or the normal matrix is singular (then no CI is emitted).
    """
    mask = inlier_mask > 0.5
    t = times[mask]
    n = int(t.shape[0])
    if n <= 3:
        return None
    design = np.column_stack([np.ones(n), t, t * t])
    try:
        ata_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    sigma2 = (rms_residual**2) * n / max(n - 3, 1)  # dof correction
    return np.asarray(sigma2 * ata_inv, dtype=np.float64)


def _launch_speed_and_ci(
    xfit: QuadraticFit,
    yfit: QuadraticFit,
    times: FloatArray,
    scale: float,
    *,
    scale_from_gravity: bool,
    systematic_rel_floor: float = _SYSTEMATIC_REL_FLOOR,
) -> tuple[float, tuple[float, float] | None]:
    """Launch speed (m/s) at the segment start and its 95% confidence interval.

    The CI is parametric error propagation (delta method) of the fit-coefficient covariance
    through ``scale`` and the planar speed ``s = scale * hypot(vx0, vy0)``. It reflects
    fit / measurement uncertainty only — it does NOT know about systematic bias from a
    violated assumption (e.g. a pitched camera), which is exactly why a coverage test
    against an independent ground truth is meaningful (an overconfident, biased estimate
    yields a tight CI that fails to cover the truth).

    ``systematic_rel_floor`` is the relative systematic-uncertainty floor; callers may widen
    it above :data:`_SYSTEMATIC_REL_FLOOR` (e.g. the depth-domination guard, where the
    recovered scale is depth-biased).
    """
    vx0, vy0 = float(xfit.c1), float(yfit.c1)
    s_px = float(np.hypot(vx0, vy0))
    speed = float(scale * s_px)
    systematic_half = systematic_rel_floor * abs(speed)

    def _floor_only() -> tuple[float, tuple[float, float] | None]:
        # The systematic floor is always defensible (depends only on speed). Emit it even
        # when the measurement covariance is unavailable, so a METRIC speed never ships with
        # NO band at all — that would understate uncertainty, not overstate it (CI-02).
        return speed, (speed - systematic_half, speed + systematic_half)

    if s_px < 1e-9:
        return speed, None  # direction/speed genuinely undefined
    cov_x = _coeff_cov(times, xfit.inlier_mask, xfit.rms_residual)
    cov_y = _coeff_cov(times, yfit.inlier_mask, yfit.rms_residual)
    if cov_x is None or cov_y is None:
        return _floor_only()

    var = (scale * vx0 / s_px) ** 2 * float(cov_x[1, 1])  # contribution from vx0
    if scale_from_gravity:
        c2y = float(yfit.c2)
        if abs(c2y) < 1e-12:
            return _floor_only()
        # speed depends on the y-fit's (c1, c2): vy0 = c1, and scale = g / (2 c2). Include
        # their covariance (delta method): d speed/d c1y, d speed/d c2y.
        d_c1y = scale * vy0 / s_px
        d_c2y = -s_px * scale / c2y
        grad = np.array([d_c1y, d_c2y], dtype=np.float64)
        var += float(grad @ cov_y[1:3, 1:3] @ grad)
    else:
        # Supplied scale is treated as exact; only vy0 carries uncertainty.
        var += (scale * vy0 / s_px) ** 2 * float(cov_y[1, 1])

    if not np.isfinite(var) or var < 0.0:
        return _floor_only()
    # Total half-width: measurement (1.96 sigma) combined with the v0.1 systematic floor
    # (in quadrature), so the CI is an honest TOTAL uncertainty, not fit-noise only.
    measurement_half = 1.96 * float(np.sqrt(var))
    half = float(np.hypot(measurement_half, systematic_half))
    return speed, (speed - half, speed + half)


def _build_metric_estimate(
    xfit: QuadraticFit,
    yfit: QuadraticFit,
    times: FloatArray,
    scale: float,
    frame_range: tuple[int, int],
    segment: Segment,
    *,
    source: str,
    confidence: float,
    gof: float,
    meta: dict[str, object],
    systematic_rel_floor: float = _SYSTEMATIC_REL_FLOOR,
) -> TrajectoryEstimate:
    """Construct a METRIC-tier estimate from the robust fits: positions m, velocity m/s."""
    positions_m, velocity_m = _fitted_positions_velocity(xfit, yfit, times, scale)
    # positions_m[:, 2] stays 0: depth unknown at this monocular tier (v0.2/stereo).

    note_meta = dict(meta)
    note_meta["depth_axis"] = "unknown_at_monocular_tier_set_to_zero"
    note_meta["trajectory_source"] = "robust_fit"
    # Launch speed + 95% CI (parametric, from the fit). A consumer/validator reads these
    # directly; the CI lets a coverage test catch overconfident (biased) metric output.
    launch_speed, ci95 = _launch_speed_and_ci(
        xfit, yfit, times, scale, scale_from_gravity=(source != "reference_scale"),
        systematic_rel_floor=systematic_rel_floor,
    )
    note_meta["launch_speed_m_s"] = launch_speed
    note_meta["launch_speed_ci95"] = ci95

    positions_q = Quantity(
        value=positions_m, unit="m", tier=Tier.METRIC,
        confidence=confidence, source=source, frame=frame_range,
    )
    velocity_q = Quantity(
        value=velocity_m, unit="m/s", tier=Tier.METRIC,
        confidence=confidence, source=source, frame=frame_range,
    )
    return TrajectoryEstimate(
        positions=positions_q,
        velocity=velocity_q,
        tier=Tier.METRIC,
        goodness_of_fit=gof,
        segment=segment,
        meta=note_meta,
    )


def _relative_fallback(
    centers: FloatArray,
    times: FloatArray,
    frame_range: tuple[int, int],
    segment: Segment,
    *,
    reason: str,
    a_px: float | None = None,
    residual_fraction: float | None = None,
    inlier_fraction: float | None = None,
    xfit: QuadraticFit | None = None,
    yfit: QuadraticFit | None = None,
) -> TrajectoryEstimate:
    """Honest scale-free fallback: a RELATIVE estimate when metric scale is not earned.

    Positions are displacements from the segment start normalized by the pixel extent (so
    the result is scale-free / dimensionless); velocity is their per-second rate. When the
    robust fits are available (the sanity-gate-failed path) the product is read off those
    fits — outliers stay rejected even in the fallback — otherwise (too-short path) raw
    centers are used. The tier is RELATIVE and NEVER METRIC (BRIEF.md §10).
    """
    if xfit is not None and yfit is not None:
        x_pred = xfit.predict(times)
        y_pred = yfit.predict(times)
        dx = x_pred - x_pred[0]
        dy = y_pred - y_pred[0]
        extent = float(np.hypot(np.ptp(x_pred), np.ptp(y_pred))) or 1.0
        positions_rel = np.zeros((times.shape[0], 3), dtype=np.float64)
        positions_rel[:, 0] = dx / extent
        positions_rel[:, 1] = dy / extent
        velocity_rel = np.zeros((times.shape[0], 3), dtype=np.float64)
        velocity_rel[:, 0] = (xfit.c1 + 2.0 * xfit.c2 * times) / extent
        velocity_rel[:, 1] = (yfit.c1 + 2.0 * yfit.c2 * times) / extent
    elif centers.shape[0] == 0:
        # No finite observations at all (e.g. an all-NaN segment): an empty, zero-confidence
        # RELATIVE estimate rather than a crash.
        positions_rel = np.zeros((0, 3), dtype=np.float64)
        velocity_rel = np.zeros((0, 3), dtype=np.float64)
    else:
        disp_px = centers - centers[0]
        extent = float(np.linalg.norm(np.ptp(centers, axis=0))) or 1.0
        positions_rel = np.zeros((centers.shape[0], 3), dtype=np.float64)
        positions_rel[:, 0] = disp_px[:, 0] / extent
        positions_rel[:, 1] = disp_px[:, 1] / extent
        velocity_rel = _finite_difference_velocity(positions_rel, times)

    # Confidence at RELATIVE tier reflects only geometric coherence, not earned scale.
    if residual_fraction is not None and inlier_fraction is not None:
        residual_factor = float(
            np.clip(1.0 - residual_fraction / _RESIDUAL_FRACTION_TOL, 0.0, 1.0)
        )
        confidence = combine_confidence(residual_factor, float(np.clip(inlier_fraction, 0.0, 1.0)))
    else:
        confidence = combine_confidence(0.5)

    meta: dict[str, object] = {
        "fallback_reason": reason,
        "tier_note": "scale_not_earned_fell_back_to_relative",
    }
    if a_px is not None:
        meta["a_px"] = a_px
    if residual_fraction is not None:
        meta["residual_fraction"] = residual_fraction
    if inlier_fraction is not None:
        meta["inlier_fraction"] = inlier_fraction

    positions_q = Quantity(
        value=positions_rel, unit=None, tier=Tier.RELATIVE,
        confidence=confidence, source="relative_fallback", frame=frame_range,
    )
    velocity_q = Quantity(
        value=velocity_rel, unit=None, tier=Tier.RELATIVE,
        confidence=confidence, source="relative_fallback", frame=frame_range,
    )
    return TrajectoryEstimate(
        positions=positions_q,
        velocity=velocity_q,
        tier=Tier.RELATIVE,
        goodness_of_fit=float(np.clip(confidence, 0.0, 1.0)),
        segment=segment,
        meta=meta,
    )
