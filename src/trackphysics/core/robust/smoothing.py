"""Robust smoothing via IRLS-gated polynomial least squares (BRIEF.md §11).

Real tracker output is dirty: jitter, gross outliers, and bursts of correlated noise.
A plain least-squares fit is pulled toward outliers; this module instead runs
*iteratively reweighted least squares* (IRLS) with a robust loss — fit a low-degree
polynomial in time, measure residuals, downweight (and effectively drop) points whose
residuals exceed a robust threshold, then refit. The threshold is derived from a robust
scale estimate (the median absolute deviation), so it adapts to the noise level instead
of assuming a fixed, IID variance.

We deliberately do not assume IID noise: real failures arrive in runs. IRLS with a
hard inlier gate tolerates correlated bursts because a contiguous run of grossly
inconsistent samples is rejected as a block once its residuals blow past the gate,
rather than each being trusted in isolation.

Pure numpy/scipy; no heavy dependencies (the core must run on an edge device, §15).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..schema import FloatArray

BoolArray = npt.NDArray[np.bool_]
"""Alias for a boolean ndarray (inlier / mask outputs)."""

# Consistency constant making the MAD an unbiased estimator of the standard deviation
# for Gaussian data: sigma_hat = 1.4826 * MAD.
_MAD_TO_SIGMA = 1.4826

# Tukey biweight tuning constant (in units of robust sigma). The conventional 4.685
# gives ~95% efficiency under Gaussian noise while fully rejecting far outliers.
_TUKEY_C = 4.685


@dataclass
class SmoothingResult:
    """Output of :func:`robust_smooth`.

    All arrays are aligned with the input ``values`` along the time axis ``T``.
    """

    smoothed: FloatArray
    """Fitted/smoothed series, same shape as the input ``values`` (``(T,)`` or
    ``(T, D)``)."""

    weights: FloatArray
    """Per-point robust weights in ``[0, 1]``, shape ``(T,)`` (shared across columns).
    Near 0 means the point was treated as an outlier."""

    residuals: FloatArray
    """Per-point residual magnitude (Euclidean across columns), shape ``(T,)``."""

    inlier_mask: BoolArray
    """Boolean mask, shape ``(T,)``: ``True`` where the point is an inlier (weight above
    a small floor)."""

    scale: float
    """Robust residual scale (MAD-based sigma) at convergence; a noise-level summary."""

    n_iter: int
    """Number of IRLS iterations actually run."""


def _as_2d(values: FloatArray) -> tuple[FloatArray, bool]:
    """Return ``values`` as ``(T, D)`` plus a flag indicating it was originally 1-D."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"values must be 1-D (T,) or 2-D (T, D), got shape {arr.shape}")


def _design_matrix(times: FloatArray, degree: int) -> FloatArray:
    """Vandermonde-style polynomial basis in (centred, scaled) time for conditioning."""
    t = np.asarray(times, dtype=np.float64)
    span = t.max() - t.min()
    # Degenerate timing (span <= 0): fall back to a constant model evaluated at zero.
    t_norm = np.zeros_like(t) if span <= 0 else (t - t.mean()) / span
    return np.vander(t_norm, N=degree + 1, increasing=True)


def _tukey_weights(unit_residuals: FloatArray) -> FloatArray:
    """Tukey biweight: weight 0 beyond ``_TUKEY_C`` robust sigmas, smooth within."""
    u = unit_residuals / _TUKEY_C
    inside = np.abs(u) < 1.0
    w = np.zeros_like(u)
    w[inside] = (1.0 - u[inside] ** 2) ** 2
    return w


def robust_smooth(
    values: FloatArray,
    times: FloatArray,
    *,
    degree: int = 2,
    max_iter: int = 10,
    tol: float = 1e-6,
    min_inlier_weight: float = 1e-3,
) -> SmoothingResult:
    """Robustly smooth a (possibly multi-dimensional) time series with IRLS.

    Fits a degree-``degree`` polynomial in time by iteratively reweighted least squares
    with a Tukey biweight loss. Each iteration: solve the weighted normal equations,
    compute per-point residual magnitudes, estimate a robust scale from the median
    absolute deviation of inlier residuals, and recompute weights — points beyond
    roughly ``_TUKEY_C`` robust sigmas get zero weight (rejected), the rest are
    smoothly downweighted. Repeats until the weights stabilise or ``max_iter``.

    Args:
        values: Series to smooth, shape ``(T,)`` or ``(T, D)``.
        times: Strictly increasing-ish time stamps, shape ``(T,)``.
        degree: Polynomial degree (2 = locally constant acceleration; a good default
            for short physical arcs).
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance on the max change in weights between iterations.
        min_inlier_weight: Weight floor below which a point is reported as an outlier
            in ``inlier_mask``.

    Returns:
        A :class:`SmoothingResult` with the smoothed series, per-point weights and
        residuals, an inlier mask, the robust scale, and the iteration count.
    """
    y, was_1d = _as_2d(values)
    t = np.asarray(times, dtype=np.float64)
    n = y.shape[0]
    if t.shape != (n,):
        raise ValueError(f"times must have shape ({n},), got {t.shape}")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    # Cannot fit a degree-``degree`` polynomial through fewer than degree+1 points.
    eff_degree = min(degree, max(0, n - 1))
    if n == 0:
        empty = np.empty((0,), dtype=np.float64)
        return SmoothingResult(
            smoothed=y.reshape(-1) if was_1d else y,
            weights=empty,
            residuals=empty,
            inlier_mask=np.empty((0,), dtype=bool),
            scale=0.0,
            n_iter=0,
        )

    design = _design_matrix(t, eff_degree)
    weights = np.ones(n, dtype=np.float64)
    fitted = y.copy()
    resid_mag = np.zeros(n, dtype=np.float64)
    scale = 0.0
    used_iter = 0

    for it in range(1, max_iter + 1):
        used_iter = it
        sqrt_w = np.sqrt(weights)[:, None]
        wa = design * sqrt_w
        wy = y * sqrt_w
        coeffs, *_ = np.linalg.lstsq(wa, wy, rcond=None)
        fitted = design @ coeffs
        resid = y - fitted
        resid_mag = np.sqrt(np.sum(resid**2, axis=1))

        # Robust scale from inlier-ish residuals. Guard degeneracy carefully (ROB-001):
        # only an *all-residuals-tiny* case means "the fit explains everything". A median
        # near zero with a large-residual minority is MAD breakdown (a clean majority plus
        # gross outliers) — there we must NOT reset to all-ones weights, or the outliers we
        # already rejected get re-admitted. Instead floor the scale and keep weighting.
        med = float(np.median(resid_mag))
        mad = float(np.median(np.abs(resid_mag - med)))
        scale = _MAD_TO_SIGMA * mad
        y_span = float(np.max(np.abs(y))) if y.size else 0.0
        degenerate = 1e-9 * (1.0 + y_span)
        if float(np.max(resid_mag)) <= degenerate:
            # Every residual is ~0: the fit truly explains all points -> all inliers.
            weights = np.ones(n, dtype=np.float64)
            break
        if scale <= degenerate:
            # MAD breakdown: clean majority drives MAD -> 0 while gross outliers remain.
            # Floor the scale so the minority of large residuals is still rejected.
            scale = degenerate

        new_weights = _tukey_weights(resid_mag / scale)
        if float(np.max(np.abs(new_weights - weights))) < tol:
            weights = new_weights
            break
        weights = new_weights

    inlier_mask = weights > min_inlier_weight
    smoothed: FloatArray = fitted.reshape(-1) if was_1d else fitted
    return SmoothingResult(
        smoothed=smoothed,
        weights=weights,
        residuals=resid_mag,
        inlier_mask=inlier_mask,
        scale=scale,
        n_iter=used_iter,
    )


__all__ = ["BoolArray", "SmoothingResult", "robust_smooth"]
