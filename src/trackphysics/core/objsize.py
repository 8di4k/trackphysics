"""Object-size-as-ruler — a second, independent monocular scale cue (BRIEF.md §10, §12).

Under a pinhole camera an object of true diameter ``D`` (meters) projects to an apparent
diameter ``d_px = f * D / Z`` pixels at depth ``Z``. The local meters-per-pixel scale at
that object is therefore::

    s = Z / f = D / d_px

i.e. the local scale can be **computed from D and d_px without knowing f or Z** (no camera
calibration). It still *equals* ``Z/f``, so it varies along the arc as depth changes — which
is exactly why the apparent-size dynamic range carries depth information. So a *known object
size* is a second "ruler" for monocular metric scale, beside
gravity-as-a-ruler (:mod:`trackphysics.core.ballistic`). Comparing the two independent
scale estimates is the **§10 cross-cue consistency check**: if they disagree and the size
channel is informative, the gravity scale is suspect — typically because depth motion is
biasing the pixel acceleration the gravity ruler relies on (the documented depth-blindness
limitation). A cross-check can only ever *lower* confidence / *widen* the band, never
inflate them, so it cannot manufacture a metric over-claim.

``D`` is an advisory numeric input a preset may carry (e.g. ``SpherePreset.diameter_m``).
The core treats it as an opaque positive length — never as object semantics (§6). The
mechanism is generic: any object whose true size is known.

THE SNR GATE (the crux this module exists to make explicit). The size channel is only
usable when its signal exceeds the detector's box-size measurement noise, and there are
TWO distinct uses with TWO different signal-to-noise ratios:

* **Absolute scale** ``s = D / median(d_px)``. Its relative error is ``~ sigma_d / d_px``,
  so it is trustworthy iff the *relative* size noise is small — i.e. ``snr_abs =
  median(d_px) / sigma_d`` is high. This needs only the apparent size at one instant.
* **Depth-motion recovery** (the missing out-of-plane axis). This needs the apparent-size
  *dynamic range* across the arc to exceed the noise — i.e. ``snr_dyn =
  dynamic_range(d_px) / sigma_d`` high.

These come apart sharply for a *small, distant* object: the apparent diameter can be large
enough in *relative* terms to give a marginal absolute scale (``snr_abs`` moderate) while
its *change* over the trajectory's depth swing is a fraction of a pixel — below box-
regression noise (``snr_dyn`` ≈ 1), so depth recovery is hopeless even though an absolute
cross-check is not. The benchmark (``bench/size_ruler.py``) maps both breakpoints and
places real regimes on them. This gate **refuses the channel honestly** below threshold
rather than fabricating scale from noise (§10).

Only numpy + the contract types are used (BRIEF.md §15).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import FloatArray

__all__ = [
    "ObjectSizeReading",
    "SizeRulerThresholds",
    "apparent_size_px",
    "read_object_size",
    "scale_agreement",
]

_EPS = 1e-9


def apparent_size_px(bboxes: FloatArray) -> FloatArray:
    """Per-frame apparent-size scalar from ``xyxy`` bboxes, shape ``(T,)``.

    Uses the geometric mean of width and height so the proxy is insensitive to anisotropic
    aspect changes and degrades gracefully for thin boxes. Non-positive extents are clamped
    to a small epsilon so the proxy stays finite for degenerate detections. This is the
    SINGLE source of the apparent-size measurement — the relative-3D lift reads it too, so
    the inverse-depth proxy and the size ruler never diverge on the definition.
    """
    arr = np.asarray(bboxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"bboxes must have shape (T, 4), got {arr.shape}")
    widths = np.maximum(np.abs(arr[:, 2] - arr[:, 0]), _EPS)
    heights = np.maximum(np.abs(arr[:, 3] - arr[:, 1]), _EPS)
    sizes: FloatArray = np.sqrt(widths * heights)
    return sizes


@dataclass(frozen=True)
class SizeRulerThresholds:
    """Gate thresholds for the object-size ruler.

    These are a REFITTABLE artifact, NOT universal constants: the defaults are calibrated on
    synthetic sweeps (``bench/size_ruler.py``) and are conservative. A deployment with a
    characterised detector may refit them. The *mechanism* (size SNR) is domain-agnostic
    (§6); only the tuned numbers are regime-flavoured.
    """

    min_points: int = 7
    """Minimum finite apparent-size observations for any reading. Matches the ballistic
    segment-detection minimum so the quadratic size de-trend always has the same support as
    the fit it cross-checks (and ≥4 residual DOF for a degree-2 trend)."""
    max_rel_noise: float = 0.12
    """Max ``sigma_d / median(d_px)`` for the ABSOLUTE size scale to be trusted
    (equivalently ``snr_abs >= 1 / max_rel_noise``)."""
    min_snr_dyn: float = 3.0
    """Min ``dynamic_range(d_px) / sigma_d`` for the size channel to carry usable
    DEPTH-MOTION signal (the harder, separate use)."""
    agreement_tol: float = 0.25
    """Relative scale discrepancy mapped to zero agreement in :func:`scale_agreement`."""


DEFAULT_THRESHOLDS = SizeRulerThresholds()


@dataclass
class ObjectSizeReading:
    """The size ruler's read of one arc: an absolute scale plus its honesty diagnostics."""

    scale_m_per_px: float
    """Absolute meters-per-pixel from ``D / median(d_px)`` (depth-free)."""
    median_size_px: float
    noise_px: float
    """Robust per-frame box-size noise (de-trended MAD), the gate's denominator."""
    dynamic_range_px: float
    """Peak-to-peak of the smooth (depth-driven) size trend across the arc."""
    rel_noise: float
    """``noise_px / median_size_px`` — the absolute-scale relative error proxy."""
    snr_abs: float
    """``median_size_px / noise_px``."""
    snr_dyn: float
    """``dynamic_range_px / noise_px``."""
    n: int
    informative_for_scale: bool
    """True iff ``rel_noise <= max_rel_noise``: the absolute scale is trustworthy."""
    informative_for_depth: bool
    """True iff ``snr_dyn >= min_snr_dyn``: the channel can carry depth-motion signal."""


def read_object_size(
    apparent_px: FloatArray,
    diameter_m: float | None,
    *,
    thresholds: SizeRulerThresholds = DEFAULT_THRESHOLDS,
) -> ObjectSizeReading | None:
    """Read absolute scale + SNR diagnostics from an apparent-size series and a known size.

    Returns ``None`` (the channel is unusable) when ``diameter_m`` is missing/non-positive,
    or fewer than ``thresholds.min_points`` finite positive observations remain, or the
    median apparent size is degenerate.

    The de-trend rationale: along a free-flight arc, depth — hence apparent size ``f*D/Z``
    — varies *smoothly* and with curvature (``Z`` is ~parabolic, its reciprocal too), so a
    **degree-2** polynomial in the sample index captures the depth-driven component. Its
    peak-to-peak is the usable DYNAMIC RANGE; the residual about it is the per-frame box-size
    NOISE (robust MAD → sigma). Using degree 2 (not 1) matters: a linear de-trend would leave
    the depth *curvature* in the residual and overstate the noise on short, strongly-curved
    arcs. ``min_points`` guarantees ≥4 residual DOF for this estimate.
    """
    if diameter_m is None or not (np.isfinite(diameter_m) and diameter_m > 0.0):
        return None
    arr = np.asarray(apparent_px, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    n = int(arr.size)
    if n < thresholds.min_points:
        return None
    median = float(np.median(arr))
    if not (np.isfinite(median) and median > 0.0):
        return None

    idx = np.arange(n, dtype=np.float64)
    # Degree-2 de-trend (n >= min_points = 7 guarantees >= 4 residual DOF). A linear trend
    # would leak the depth curvature into the residual and overstate the noise.
    coeffs = np.polyfit(idx, arr, 2)
    trend = np.polyval(coeffs, idx)
    resid = arr - trend
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    noise = max(noise, _EPS)
    dynamic_range = float(np.ptp(trend))

    rel_noise = noise / median
    snr_abs = median / noise
    snr_dyn = dynamic_range / noise
    scale = float(diameter_m) / median

    return ObjectSizeReading(
        scale_m_per_px=scale,
        median_size_px=median,
        noise_px=noise,
        dynamic_range_px=dynamic_range,
        rel_noise=rel_noise,
        snr_abs=snr_abs,
        snr_dyn=snr_dyn,
        n=n,
        informative_for_scale=bool(rel_noise <= thresholds.max_rel_noise),
        informative_for_depth=bool(snr_dyn >= thresholds.min_snr_dyn),
    )


def scale_agreement(
    size_scale: float,
    gravity_scale: float,
    *,
    tol: float = DEFAULT_THRESHOLDS.agreement_tol,
) -> tuple[float, float]:
    """Compare two independent meters-per-pixel scales.

    Returns ``(rel_disc, agreement)`` where ``rel_disc = |size - gravity| / gravity`` and
    ``agreement = clip(1 - rel_disc / tol, 0, 1)`` (1 = identical, 0 once the discrepancy
    reaches ``tol``). On a non-finite or non-positive input the scales cannot be compared:
    ``rel_disc`` is ``inf`` and ``agreement`` is ``0.0`` (treated as full disagreement,
    never as confirmation — failing the check must not look like passing it, §10).
    """
    if not (
        np.isfinite(size_scale)
        and np.isfinite(gravity_scale)
        and size_scale > 0.0
        and gravity_scale > 0.0
        and tol > 0.0
    ):
        return float("inf"), 0.0
    rel_disc = abs(size_scale - gravity_scale) / gravity_scale
    agreement = float(np.clip(1.0 - rel_disc / tol, 0.0, 1.0))
    return float(rel_disc), agreement
