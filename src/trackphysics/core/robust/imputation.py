"""Coherence-gated gap / occlusion handling (BRIEF.md §11).

Trackers lose objects to occlusion and false negatives, leaving gaps in the frame
index sequence. We can reconstruct short-to-mid gaps by predicting across them with a
robust smooth, but we must do so *honestly*: a recovered span is only stitched into the
track if the observations *after* the gap are coherent with the physically predicted
trajectory. If the post-gap evidence disagrees (a kinematic discontinuity consistent
with an ID-switch or a genuinely separate motion), the gap is left as
segment-terminating and the caller is expected to lower confidence — we never silently
bridge an incoherent jump.

The coherence gate compares the smoother's extrapolation at the first post-gap samples
against the actual observations, normalised by a scale that is the larger of the in-track
robust residual scale and a *physically meaningful* displacement floor (a small multiple
of the median per-step displacement of the pre-window). The displacement floor matters:
a short, over-determined pre-window of clean motion fits (near-)perfectly, so the residual
scale collapses toward zero — without a motion-aware floor the gate would reject genuinely
coherent curved (drag / parabolic) continuations as if any sub-pixel deviation were
catastrophic. This mirrors the residual-gated imputation used in occlusion-robust
trajectory literature, generalised to a domain-free setting.

The agreement score is mapped so that its ``0.5`` contour coincides exactly with the stitch
threshold: ``coherence = coherence_tol / (coherence_tol + worst)`` is ``0.5`` when the worst
normalised deviation equals ``coherence_tol`` (the stitch boundary), ``> 0.5`` inside it
(stitched), and ``< 0.5`` outside it (refused). So "coherence ≥ 0.5" and "stitched" mean the
same thing — the score never contradicts the decision.

Pure numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..schema import FloatArray
from .smoothing import SmoothingResult, robust_smooth


class _Smoother(Protocol):
    """Callable signature compatible with :func:`robust_smooth`.

    Lets a caller inject an alternative smoother for prediction across gaps while
    keeping this module decoupled from any single implementation.
    """

    def __call__(
        self, values: FloatArray, times: FloatArray
    ) -> SmoothingResult: ...


@dataclass
class GapRecord:
    """One detected gap in the frame index sequence and how it was handled."""

    start_frame: int
    """Last observed frame *before* the gap."""

    end_frame: int
    """First observed frame *after* the gap."""

    length: int
    """Number of missing frames inside the gap (``end_frame - start_frame - 1``)."""

    stitched: bool
    """``True`` if the gap was filled (post-gap observations were coherent); ``False``
    if the gap is segment-terminating (incoherent post-gap evidence)."""

    coherence: float
    """Normalised post-gap agreement score in ``[0, 1]``: 1 = perfectly coherent,
    0 = grossly inconsistent. Its ``0.5`` contour coincides with the stitch threshold, so
    ``coherence >= 0.5`` iff the gap was stitched. Drives downstream confidence."""


@dataclass
class ImputationResult:
    """Output of :func:`impute_gaps`."""

    frames: FloatArray
    """Densified frame indices over the spanned range, shape ``(N,)`` (float for a
    uniform array type; values are whole numbers)."""

    values: FloatArray
    """Series aligned with ``frames``: observed where present, imputed across stitched
    gaps, and left as ``nan`` across gaps that were refused, shape ``(N,)`` or
    ``(N, D)``."""

    gaps: list[GapRecord] = field(default_factory=list)
    observed_mask: FloatArray = field(
        default_factory=lambda: np.empty((0,), dtype=bool)
    )
    """Boolean mask over ``frames``: ``True`` where the value is a real observation."""


def _default_smoother(values: FloatArray, times: FloatArray) -> SmoothingResult:
    return robust_smooth(values, times, degree=2)


def _as_2d(values: FloatArray) -> tuple[FloatArray, bool]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"values must be 1-D (T,) or 2-D (T, D), got shape {arr.shape}")


def impute_gaps(
    frames: FloatArray,
    values: FloatArray,
    *,
    smoother: _Smoother | None = None,
    coherence_tol: float = 3.0,
    max_gap: int = 15,
    n_post: int = 3,
) -> ImputationResult:
    """Fill short-to-mid gaps, but only when post-gap evidence stays coherent.

    Detects gaps as jumps greater than one in the (assumed ascending) ``frames`` index
    sequence. For each gap, it fits a robust smooth to the observations *before* the gap
    (plus a leading context window) and predicts across the gap. It then checks the
    first ``n_post`` observations after the gap against that prediction. If the worst
    normalised deviation is within ``coherence_tol`` robust sigmas, the gap is stitched
    (filled by interpolating the predicted endpoints into the observed ones); otherwise
    it is recorded as segment-terminating (``stitched=False``) and the missing frames
    are left as ``nan``.

    Args:
        frames: Ascending frame indices of the observations, shape ``(T,)``.
        values: Observed series, shape ``(T,)`` or ``(T, D)``.
        smoother: Optional smoother used for cross-gap prediction; defaults to a
            degree-2 :func:`robust_smooth`.
        coherence_tol: Maximum allowed normalised post-gap deviation (in robust sigmas)
            for a gap to be stitched.
        max_gap: Gaps longer than this many missing frames are never stitched (too far
            to predict honestly).
        n_post: Number of post-gap observations used for the coherence check.

    Returns:
        An :class:`ImputationResult` with densified frames, the imputed series
        (``nan`` across refused gaps), per-gap records, and an observed mask.
    """
    smooth = smoother if smoother is not None else _default_smoother
    f = np.asarray(frames, dtype=np.float64)
    y2d, was_1d = _as_2d(values)
    n_obs = y2d.shape[0]
    if f.shape != (n_obs,):
        raise ValueError(f"frames must have shape ({n_obs},), got {f.shape}")

    if n_obs == 0:
        empty = np.empty((0,), dtype=np.float64)
        return ImputationResult(
            frames=empty,
            values=empty if was_1d else np.empty((0, y2d.shape[1]), dtype=np.float64),
            gaps=[],
            observed_mask=np.empty((0,), dtype=bool),
        )

    f_int = np.rint(f).astype(np.int64)
    if n_obs >= 2 and bool(np.any(np.diff(f_int) <= 0)):
        # Non-increasing/duplicate frames (after rounding) would let numpy fancy assignment
        # silently keep only the last colliding write, discarding observations without a
        # trace. The documented contract is ascending frames; fail loudly instead.
        raise ValueError(
            "impute_gaps requires strictly increasing frame indices; got non-increasing or "
            "duplicate frames after rounding"
        )
    dim = y2d.shape[1]
    full_frames = np.arange(int(f_int[0]), int(f_int[-1]) + 1, dtype=np.int64)
    n_full = full_frames.shape[0]
    out = np.full((n_full, dim), np.nan, dtype=np.float64)
    observed = np.zeros(n_full, dtype=bool)

    # Map observations onto the dense grid.
    idx_in_full = f_int - f_int[0]
    out[idx_in_full] = y2d
    observed[idx_in_full] = True

    gaps: list[GapRecord] = []
    for k in range(1, n_obs):
        step = int(f_int[k] - f_int[k - 1])
        if step <= 1:
            continue
        gap_len = step - 1
        start_frame = int(f_int[k - 1])
        end_frame = int(f_int[k])

        # Context window before the gap for the predictive smooth. Take a generous
        # window (at least 5 samples when available) so the robust residual scale is a
        # meaningful noise estimate rather than the ~0 of an exactly-determined fit.
        pre_lo = max(0, k - max(5, n_post + 2))
        pre_f = f[pre_lo:k]
        pre_y = y2d[pre_lo:k]
        post_hi = min(n_obs, k + n_post)
        post_f = f[k:post_hi]
        post_y = y2d[k:post_hi]

        stitched, coherence = _evaluate_gap(
            smooth, pre_f, pre_y, post_f, post_y, gap_len, max_gap, coherence_tol
        )

        if stitched:
            _fill_linear(out, idx_in_full, k, dim)

        gaps.append(
            GapRecord(
                start_frame=start_frame,
                end_frame=end_frame,
                length=gap_len,
                stitched=stitched,
                coherence=coherence,
            )
        )

    out_values: FloatArray = out.reshape(-1) if was_1d else out
    return ImputationResult(
        frames=full_frames.astype(np.float64),
        values=out_values,
        gaps=gaps,
        observed_mask=observed,
    )


def _evaluate_gap(
    smooth: _Smoother,
    pre_f: FloatArray,
    pre_y: FloatArray,
    post_f: FloatArray,
    post_y: FloatArray,
    gap_len: int,
    max_gap: int,
    coherence_tol: float,
) -> tuple[bool, float]:
    """Decide whether a single gap is coherent enough to stitch.

    Returns ``(stitched, coherence)`` where ``coherence`` is a monotone score in
    ``[0, 1]`` derived from the worst normalised post-gap deviation, with its ``0.5``
    contour pinned to the stitch threshold (``coherence >= 0.5`` iff stitched).
    """
    if gap_len > max_gap:
        return False, 0.0
    if pre_f.shape[0] < 2 or post_f.shape[0] == 0:
        # Not enough context to predict, or nothing to check against: refuse honestly.
        return False, 0.0

    res = smooth(pre_y, pre_f)
    # Predict across the gap by extrapolating the robust polynomial fit.
    pred = _extrapolate(pre_f, res.smoothed, post_f)

    # Scale for normalisation. The in-track robust residual scale alone is fragile: a
    # short, over-determined pre-window of clean motion fits (near-)perfectly, so
    # ``res.scale`` collapses toward zero and every sub-pixel post-gap deviation would
    # read as catastrophic, falsely refusing coherent curved (drag / parabolic)
    # continuations (ROB-002). We floor the scale by a physically meaningful displacement
    # granularity — a small multiple of the median per-step displacement of the
    # pre-window — so the tolerance reflects how far the object actually moves per frame,
    # not numerical fit noise.
    motion_floor = _step_displacement_floor(pre_y)
    scale = max(res.scale, motion_floor)
    if scale <= 0.0:
        # Truly static pre-window with a perfect fit: fall back to a tiny absolute floor
        # so we still normalise, rather than dividing by zero.
        span = float(np.max(np.abs(pre_y)) - np.min(np.abs(pre_y))) if pre_y.size else 0.0
        scale = 1e-6 * max(1.0, span)

    dev = np.sqrt(np.sum((post_y - pred) ** 2, axis=1)) / scale
    worst = float(np.max(dev))
    stitched = worst <= coherence_tol
    # Monotone agreement score whose 0.5 contour coincides with the stitch threshold
    # (ROB-003): coherence == 0.5 exactly when worst == coherence_tol, > 0.5 when
    # stitched, < 0.5 when refused. So the score never contradicts the decision.
    coherence = float(coherence_tol / (coherence_tol + worst)) if coherence_tol > 0 else 0.0
    return stitched, coherence


def _step_displacement_floor(pre_y: FloatArray) -> float:
    """Physically meaningful scale floor: a small fraction of the median per-step move.

    Uses the median Euclidean displacement between consecutive pre-gap samples — the
    natural motion granularity of the track — scaled down so the gate still resolves
    deviations finer than a full step while not collapsing to numerical fit noise. A
    deviation of one median step over a one-frame extrapolation is well within the
    default ``coherence_tol`` (so a smoothly curving arc is accepted), whereas a teleport
    of many median steps blows past it (so an ID-switch is still refused).
    """
    if pre_y.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(pre_y, axis=0), axis=1)
    med = float(np.median(steps))
    # A modest fraction of the median step: large enough to absorb sub-step curvature
    # and smoother slack, small enough that a gross teleport still fails the gate.
    return 0.1 * med


def _extrapolate(
    src_t: FloatArray, src_vals: FloatArray, dst_t: FloatArray
) -> FloatArray:
    """Polynomial extrapolation matching the smoother's effective degree per column."""
    n = src_t.shape[0]
    deg = min(2, max(0, n - 1))
    vals2d = src_vals if src_vals.ndim == 2 else src_vals.reshape(-1, 1)
    out = np.empty((dst_t.shape[0], vals2d.shape[1]), dtype=np.float64)
    for c in range(vals2d.shape[1]):
        coeffs = np.polyfit(src_t, vals2d[:, c], deg)
        out[:, c] = np.polyval(coeffs, dst_t)
    return out


def _fill_linear(
    out: FloatArray, idx_in_full: FloatArray, k: int, dim: int
) -> None:
    """Linearly interpolate the dense grid between observed endpoints of a gap."""
    lo = int(idx_in_full[k - 1])
    hi = int(idx_in_full[k])
    n_steps = hi - lo
    for c in range(dim):
        a = out[lo, c]
        b = out[hi, c]
        for s in range(1, n_steps):
            out[lo + s, c] = a + (b - a) * (s / n_steps)


__all__ = ["GapRecord", "ImputationResult", "impute_gaps"]
