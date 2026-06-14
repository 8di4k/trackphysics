"""Benchmark metrics: error, degradation curves, calibration, gate behaviour (§14.4).

These quantify the four things the benchmark exists to prove (BRIEF.md §14.4):

* **Metric error** vs exact synthetic ground truth — absolute, in meters / m·s⁻¹
  (:func:`position_error`, :func:`velocity_error`, :func:`speed_error`).
* **Degradation curves** — error as a function of a swept corruption parameter
  (:func:`degradation_curve`).
* **Provenance calibration** — reliability of stated confidence vs empirical correctness,
  summarised by Expected Calibration Error (:func:`reliability`).
* **Goodness-of-fit gate behaviour** — does the engine correctly choose METRIC vs a
  lower-tier fallback? (:func:`gate_precision_recall`).

All errors align prediction and ground truth by *frame* via :func:`align_by_frame`, so a
prediction defined on a subset of frames (the common case after gap bursts) is scored only
where both are defined.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from trackphysics.core.schema import FloatArray


def align_by_frame(
    pred_frames: FloatArray,
    pred_values: FloatArray,
    gt_frames: FloatArray,
    gt_values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Align predicted and ground-truth value series on their common frames.

    ``pred_values`` / ``gt_values`` are ``(N, ...)`` arrays indexed parallel to their frame
    arrays. Returns ``(pred_common, gt_common)`` restricted to frames present in both, in
    ascending frame order. Raises if there is no overlap.
    """
    pf = np.asarray(pred_frames).astype(np.int64)
    gf = np.asarray(gt_frames).astype(np.int64)
    common = np.intersect1d(pf, gf)
    if common.size == 0:
        raise ValueError("no common frames between prediction and ground truth")
    pred_idx = {int(f): i for i, f in enumerate(pf)}
    gt_idx = {int(f): i for i, f in enumerate(gf)}
    pred_sel = np.array([pred_idx[int(f)] for f in common], dtype=np.int64)
    gt_sel = np.array([gt_idx[int(f)] for f in common], dtype=np.int64)
    return np.asarray(pred_values)[pred_sel], np.asarray(gt_values)[gt_sel]


def _rms_rowwise(diff: FloatArray) -> float:
    """RMS of per-row Euclidean norms (or of scalars for a 1-D input)."""
    arr = np.asarray(diff, dtype=np.float64)
    if arr.ndim == 1:
        per_sample = np.abs(arr)
    else:
        per_sample = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
    if per_sample.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(per_sample**2)))


def position_error(pred_positions: FloatArray, gt_positions: FloatArray) -> float:
    """RMS 3D position error in meters between frame-aligned ``(T, 3)`` arrays."""
    pred = np.asarray(pred_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    return _rms_rowwise(pred - gt)


def velocity_error(pred_velocities: FloatArray, gt_velocities: FloatArray) -> float:
    """RMS 3D velocity error in m/s between frame-aligned ``(T, 3)`` arrays."""
    pred = np.asarray(pred_velocities, dtype=np.float64)
    gt = np.asarray(gt_velocities, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    return _rms_rowwise(pred - gt)


def speed_error(pred_speed: FloatArray, gt_speed: FloatArray) -> float:
    """RMS scalar-speed error in m/s between frame-aligned ``(T,)`` arrays."""
    pred = np.asarray(pred_speed, dtype=np.float64)
    gt = np.asarray(gt_speed, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    return _rms_rowwise(pred - gt)


def degradation_curve(
    run_fn: Callable[[float], float],
    sweep_values: Sequence[float] | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate ``run_fn`` across a corruption-parameter sweep (BRIEF.md §14.2, §14.4).

    ``run_fn(level) -> error`` is called once per swept value (e.g. error as gap length,
    jitter variance, or id-switch rate increases). Returns ``(params, errors)`` as parallel
    float arrays suitable for plotting a degradation curve.
    """
    params = np.asarray(sweep_values, dtype=np.float64)
    errors = np.array([float(run_fn(float(v))) for v in params], dtype=np.float64)
    return params, errors


def reliability(
    pred_conf: FloatArray,
    correct: FloatArray,
    n_bins: int = 10,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Reliability-diagram data + Expected Calibration Error (BRIEF.md §10, §14.4).

    Bins stated confidence into ``n_bins`` equal-width bins over ``[0, 1]`` and compares the
    mean confidence in each bin to the empirical accuracy (fraction of ``correct`` True).
    A perfectly calibrated predictor lies on the diagonal; ECE is the count-weighted mean
    absolute gap between confidence and accuracy.

    ``pred_conf`` is a float array in ``[0, 1]``; ``correct`` is a parallel boolean (or
    0/1) array. Returns ``(bin_centers, empirical_acc, counts, ece)``. Empty bins report
    ``NaN`` accuracy and contribute nothing to ECE.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf = np.asarray(pred_conf, dtype=np.float64)
    corr = np.asarray(correct).astype(np.float64)
    if conf.shape != corr.shape:
        raise ValueError(f"pred_conf and correct must match shape: {conf.shape} vs {corr.shape}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acc = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    total = conf.size
    ece = 0.0
    if total == 0:
        return centers, acc, counts, 0.0

    # Right-closed bins so confidence == 1.0 lands in the final bin.
    bin_idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(np.count_nonzero(mask))
        counts[b] = cnt
        if cnt == 0:
            continue
        bin_acc = float(np.mean(corr[mask]))
        bin_conf = float(np.mean(conf[mask]))
        acc[b] = bin_acc
        ece += (cnt / total) * abs(bin_conf - bin_acc)
    return centers, acc, counts, float(ece)


def gate_precision_recall(
    predicted_metric: FloatArray,
    scale_recoverable: FloatArray,
) -> dict[str, float]:
    """Precision/recall of the METRIC-vs-fallback decision (BRIEF.md §10, §14.4).

    Treats "emit METRIC tier" as the positive prediction and "scale is genuinely
    recoverable" as the ground-truth positive. A well-behaved engine should emit metric
    exactly when scale is earned, and honestly fall back otherwise.

    Both inputs are parallel boolean (or 0/1) arrays. Returns a dict with ``precision``,
    ``recall``, ``f1``, ``accuracy``, and the raw confusion counts ``tp``/``fp``/``fn``/``tn``.
    Precision/recall/f1 are ``0.0`` when their denominator is zero (no positives).
    """
    pred = np.asarray(predicted_metric).astype(bool)
    truth = np.asarray(scale_recoverable).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"inputs must match shape: {pred.shape} vs {truth.shape}")

    tp = float(np.count_nonzero(pred & truth))
    fp = float(np.count_nonzero(pred & ~truth))
    fn = float(np.count_nonzero(~pred & truth))
    tn = float(np.count_nonzero(~pred & ~truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


__all__ = [
    "align_by_frame",
    "degradation_curve",
    "gate_precision_recall",
    "position_error",
    "reliability",
    "speed_error",
    "velocity_error",
]
