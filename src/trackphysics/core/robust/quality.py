"""Track-quality diagnostics — a product feature, not debug output (BRIEF.md §11).

The engine's job is to report *where* a track is unreliable rather than silently
emitting a best guess. :func:`assess_quality` scans a :class:`TrackSequence` and returns
a :class:`TrackQualityReport` of generic, domain-free flags:

* ``"gap"`` — missing frames in the index sequence (occlusion / false negatives).
* ``"jitter"`` — a burst where observations deviate strongly from a robust smooth.
* ``"id_switch"`` — an instantaneous position jump incompatible with smooth motion,
  the kinematic signature of an identity swap.
* ``"low_density"`` — too few observations over the spanned range to trust the track.

Each flag carries a span and a severity in ``[0, 1]``. The report also exposes
``completeness`` (observed / expected frames) and an aggregate ``overall_score``.

Pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..results import QualityFlag, TrackQualityReport
from ..schema import FloatArray, TrackSequence
from .smoothing import robust_smooth

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# A jump is "id-switch-like" when a single-step displacement exceeds this many robust
# sigmas of the step-to-step displacement distribution.
_ID_SWITCH_SIGMAS = 6.0

# A point is part of a jitter burst when its smoothing residual exceeds this many robust
# sigmas. Lower than the id-switch gate: jitter is many moderate deviations, not one
# huge jump.
_JITTER_SIGMAS = 3.5

# Below this observed/expected density over the span, the whole track is flagged.
_LOW_DENSITY_THRESHOLD = 0.6


def _mad_sigma(x: FloatArray) -> float:
    """MAD-based robust standard deviation of a 1-D array (0.0 if degenerate)."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _spans_from_mask(frames: IntArray, mask: BoolArray) -> list[tuple[int, int]]:
    """Group consecutive ``True`` entries of ``mask`` into ``(start_frame, end_frame)``."""
    spans: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, flagged in enumerate(mask):
        if flagged and run_start is None:
            run_start = i
        elif not flagged and run_start is not None:
            spans.append((int(frames[run_start]), int(frames[i - 1])))
            run_start = None
    if run_start is not None:
        spans.append((int(frames[run_start]), int(frames[-1])))
    return spans


def assess_quality(track: TrackSequence) -> TrackQualityReport:
    """Diagnose where and how badly a track is unreliable.

    Detects gaps, jitter bursts, ID-switch-like jumps, and low observation density,
    emitting one :class:`QualityFlag` per affected region. Computes ``completeness``
    (observed frames / expected frames over the span) and an ``overall_score`` that
    starts at the completeness and is further reduced by the severity of detected
    flags — so a track that is both sparse and jittery scores lower than one that is
    merely sparse.

    Args:
        track: The track to assess.

    Returns:
        A :class:`TrackQualityReport`.
    """
    frames = track.frames.astype(np.int64)
    n = frames.shape[0]
    flags: list[QualityFlag] = []

    if n == 0:
        return TrackQualityReport(
            flags=[], completeness=0.0, overall_score=0.0, notes={"n_detections": 0}
        )

    span = int(frames[-1] - frames[0]) + 1
    expected = span
    completeness = float(n / expected) if expected > 0 else 1.0

    # --- Gaps: any step > 1 in the frame index sequence. ---
    gap_count = 0
    total_missing = 0
    if n >= 2:
        steps = np.diff(frames)
        for k in range(steps.shape[0]):
            step = int(steps[k])
            if step > 1:
                missing = step - 1
                gap_count += 1
                total_missing += missing
                # Severity grows with gap length, saturating; sub-1 always.
                severity = float(1.0 - np.exp(-missing / 5.0))
                flags.append(
                    QualityFlag(
                        start_frame=int(frames[k]),
                        end_frame=int(frames[k + 1]),
                        reason="gap",
                        severity=severity,
                    )
                )

    centers = track.centers()
    times = track.times()

    # Median per-step displacement: the track's own spatial granularity. Computed once
    # and reused for both the id-switch gate and (as a fallback scale) the jitter gate.
    disp = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    med_step = float(np.median(disp)) if disp.size else 0.0

    # --- ID-switch-like jumps: single-step displacement far beyond the typical step. ---
    if n >= 3:
        med = med_step
        sigma = _mad_sigma(disp)
        if sigma > 1e-9:
            jump_idx = np.where(disp > med + _ID_SWITCH_SIGMAS * sigma)[0]
            for j in jump_idx:
                excess = (float(disp[j]) - med) / (sigma + 1e-12)
                severity = float(1.0 - np.exp(-excess / _ID_SWITCH_SIGMAS))
                flags.append(
                    QualityFlag(
                        start_frame=int(frames[j]),
                        end_frame=int(frames[j + 1]),
                        reason="id_switch",
                        severity=max(0.0, min(1.0, severity)),
                    )
                )

    # --- Jitter bursts: contiguous runs of high smoothing residual. ---
    if n >= 4:
        res = robust_smooth(centers, times, degree=2)
        # The robust smoother rejects a localised burst against an otherwise-clean track,
        # so its residual scale collapses toward 0 (the clean majority drives the MAD).
        # If we gated jitter on ``res.scale > 1e-9`` alone we would skip the whole branch
        # and never flag that burst (ROB-004). When the scale is degenerate, fall back to
        # an absolute threshold derived from the track's own spatial granularity (the
        # median inter-frame step), so a burst still stands out against clean motion.
        if res.scale > 1e-9:
            jitter_thresh = _JITTER_SIGMAS * res.scale
            denom = res.scale + 1e-12
        elif med_step > 1e-9:
            jitter_thresh = _JITTER_SIGMAS * med_step
            denom = med_step
        else:
            jitter_thresh = float("inf")
            denom = 1.0
        jitter_mask = res.residuals > jitter_thresh
        for s_start, s_end in _spans_from_mask(frames, jitter_mask):
            region = (frames >= s_start) & (frames <= s_end)
            worst = float(np.max(res.residuals[region]) / denom)
            severity = float(1.0 - np.exp(-worst / (2.0 * _JITTER_SIGMAS)))
            flags.append(
                QualityFlag(
                    start_frame=s_start,
                    end_frame=s_end,
                    reason="jitter",
                    severity=max(0.0, min(1.0, severity)),
                )
            )

    # --- Low observation density over the spanned range. ---
    if completeness < _LOW_DENSITY_THRESHOLD:
        severity = float(min(1.0, 1.0 - completeness))
        flags.append(
            QualityFlag(
                start_frame=int(frames[0]),
                end_frame=int(frames[-1]),
                reason="low_density",
                severity=severity,
            )
        )

    overall = float(np.clip(completeness, 0.0, 1.0))
    for flag in flags:
        # Each flag erodes the score multiplicatively by its severity, so independent
        # problems compound rather than one masking another.
        overall *= 1.0 - 0.5 * flag.severity
    overall = float(np.clip(overall, 0.0, 1.0))

    notes: dict[str, object] = {
        "n_detections": n,
        "expected_frames": expected,
        "gap_count": gap_count,
        "missing_frames": total_missing,
    }
    return TrackQualityReport(
        flags=flags,
        completeness=float(np.clip(completeness, 0.0, 1.0)),
        overall_score=overall,
        notes=notes,
    )


__all__ = ["assess_quality"]
