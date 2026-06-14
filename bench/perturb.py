"""Correlated track-corruption models for the benchmark (BRIEF.md §14.2).

Real tracker failures are bursty and correlated, *not* IID per-frame noise. Per-point IID
noise loses intra-trajectory correlation (Hug et al., 2024) and produces an over-clean test
that teaches "the smoother's rules"; we instead model full-trajectory / correlated
corruption so the engine's robustness moat is measured honestly (BRIEF.md §11, §14.2).

Each perturbation takes a clean :class:`TrackSequence`, a parametric corruption level, and a
``numpy.random.Generator`` (passed in for deterministic, seedable runs — never the global
RNG). They return a *new* corrupted :class:`TrackSequence`; inputs are never mutated.

Provided corruptions (each independently sweepable):

* :func:`apply_gap_bursts` — drop runs of consecutive frames (occlusion / false negatives).
* :func:`apply_id_switches` — relabel ``track_id`` at switch points (tracker identity errors).
* :func:`apply_correlated_jitter` — AR(1) correlated positional noise on bbox corners.
* :func:`apply_false_positives_dropouts` — spurious detections and isolated missed frames.

:func:`corrupt` composes them from a :class:`CorruptionConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackphysics.core.schema import Detection, TrackSequence


def _clone_detection(det: Detection) -> Detection:
    """Deep-ish copy of a detection (bbox/keypoints copied so callers can't alias)."""
    return Detection(
        frame=det.frame,
        bbox=np.asarray(det.bbox, dtype=np.float64).copy(),
        track_id=det.track_id,
        score=det.score,
        keypoints=None if det.keypoints is None else np.asarray(det.keypoints).copy(),
        class_id=det.class_id,
    )


def _rebuild(track: TrackSequence, detections: list[Detection]) -> TrackSequence:
    """Build a new TrackSequence preserving timing/skeleton/image metadata."""
    return TrackSequence(
        detections=detections,
        fps=track.fps,
        timestamps=None if track.timestamps is None else np.asarray(track.timestamps).copy(),
        skeleton=track.skeleton,
        image_size=track.image_size,
    )


def apply_gap_bursts(
    track: TrackSequence,
    *,
    burst_len: int,
    n_bursts: int,
    rng: np.random.Generator,
) -> TrackSequence:
    """Drop ``n_bursts`` runs of ``burst_len`` consecutive detections (occlusion bursts).

    Burst start indices are sampled without replacement so bursts do not perfectly overlap;
    the actual number of removed detections may be slightly less than
    ``n_bursts * burst_len`` when bursts abut or reach the track end.
    """
    if burst_len < 0 or n_bursts < 0:
        raise ValueError("burst_len and n_bursts must be non-negative")
    n = len(track)
    if n == 0 or burst_len == 0 or n_bursts == 0:
        return _rebuild(track, [_clone_detection(d) for d in track.detections])

    max_start = max(1, n - burst_len)
    n_pick = min(n_bursts, max_start)
    starts = rng.choice(max_start, size=n_pick, replace=False)
    drop = np.zeros(n, dtype=bool)
    for s in starts:
        drop[int(s) : int(s) + burst_len] = True

    kept = [_clone_detection(d) for d, dr in zip(track.detections, drop, strict=True) if not dr]
    return _rebuild(track, kept)


def apply_id_switches(
    track: TrackSequence,
    *,
    rate: float,
    rng: np.random.Generator,
    new_id_offset: int = 1000,
) -> TrackSequence:
    """Relabel ``track_id`` at random switch points (tracker identity errors).

    ``rate`` in ``[0, 1]`` is the per-frame probability of a switch *onset*. At each onset
    the id flips to a fresh value (offset by ``new_id_offset``) and stays there until the
    next onset, so switches form contiguous runs rather than IID per-frame flips.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    dets = [_clone_detection(d) for d in track.detections]
    if not dets:
        return _rebuild(track, dets)

    base_id = dets[0].track_id
    current = base_id
    switch_count = 0
    for det in dets:
        if rng.random() < rate:
            switch_count += 1
            current = base_id + new_id_offset + switch_count
        det.track_id = current
    return _rebuild(track, dets)


def apply_correlated_jitter(
    track: TrackSequence,
    *,
    sigma_px: float,
    rho: float,
    rng: np.random.Generator,
) -> TrackSequence:
    """Add AR(1) temporally-correlated positional noise to bbox corners.

    The per-frame noise follows ``e_t = rho * e_{t-1} + sqrt(1 - rho^2) * sigma * z_t`` with
    ``z_t`` standard normal, applied identically to both bbox corners so the box translates
    (centre jitters) without spuriously changing size. ``rho`` in ``[0, 1)`` sets the
    temporal correlation; ``rho = 0`` reduces to IID noise of std ``sigma_px``.
    """
    if sigma_px < 0:
        raise ValueError("sigma_px must be non-negative")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    dets = [_clone_detection(d) for d in track.detections]
    n = len(dets)
    if n == 0 or sigma_px == 0.0:
        return _rebuild(track, dets)

    innov_scale = np.sqrt(1.0 - rho * rho) * sigma_px
    noise = np.empty((n, 2), dtype=np.float64)
    prev = rng.standard_normal(2) * sigma_px  # stationary-variance initial state
    noise[0] = prev
    for i in range(1, n):
        prev = rho * prev + innov_scale * rng.standard_normal(2)
        noise[i] = prev

    for det, (dx, dy) in zip(dets, noise, strict=True):
        det.bbox = det.bbox + np.array([dx, dy, dx, dy], dtype=np.float64)
    return _rebuild(track, dets)


def apply_size_noise(
    track: TrackSequence,
    *,
    sigma_px: float,
    rho: float,
    rng: np.random.Generator,
) -> TrackSequence:
    """Add AR(1)-correlated noise to box SIZE (extent), holding the centroid fixed.

    Distinct from :func:`apply_correlated_jitter`, which *translates* the box and therefore
    preserves its size: this perturbs the apparent box *extent* — the size channel the
    object-size-as-ruler reads — leaving the centre untouched. It models a detector's
    box-regression noise on the size dimension, the limiting factor for size-derived scale.

    The same AR(1) increment ``e_t`` is added to the half-width and half-height each frame
    (a pure isotropic size wobble), clamped so extents stay positive. The resulting apparent
    diameter (geometric mean of width and height) wobbles by ``~2 * e_t``. ``rho`` in
    ``[0, 1)`` sets the temporal correlation; ``rho = 0`` reduces to IID size noise.

    Note: the positivity clamp rectifies large negative excursions, so it induces a small
    UPWARD apparent-size bias that grows only at very low snr_abs (sub-noise regime, where
    the size channel is already refused) — negligible in the usable range.
    """
    if sigma_px < 0:
        raise ValueError("sigma_px must be non-negative")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    dets = [_clone_detection(d) for d in track.detections]
    n = len(dets)
    if n == 0 or sigma_px == 0.0:
        return _rebuild(track, dets)

    innov_scale = np.sqrt(1.0 - rho * rho) * sigma_px
    noise = np.empty(n, dtype=np.float64)
    prev = float(rng.standard_normal() * sigma_px)  # stationary-variance initial state
    noise[0] = prev
    for i in range(1, n):
        prev = rho * prev + innov_scale * float(rng.standard_normal())
        noise[i] = prev

    eps = 1e-3
    for det, e in zip(dets, noise, strict=True):
        x0, y0, x1, y1 = det.bbox
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        hw = max(0.5 * (x1 - x0) + e, eps)
        hh = max(0.5 * (y1 - y0) + e, eps)
        det.bbox = np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float64)
    return _rebuild(track, dets)


def apply_false_positives_dropouts(
    track: TrackSequence,
    *,
    fp_rate: float,
    drop_rate: float,
    rng: np.random.Generator,
    fp_jump_px: float = 60.0,
) -> TrackSequence:
    """False-negative dropouts plus false-positive-style outlier *displacement*.

    With probability ``drop_rate`` each detection is removed (an isolated false negative).
    With probability ``fp_rate`` a surviving detection is *displaced* by a large random jump
    of scale ``fp_jump_px`` — the tracker latching onto a spurious nearby location. Both
    rates are per-frame in ``[0, 1]``.

    Scope note (honest labelling): because a :class:`TrackSequence` is one object's track
    with strictly-increasing frames, this models a false positive as a *displaced* existing
    detection, not an *extra* injected box (which would need a duplicate/extra frame the
    schema forbids for a single track). It therefore exercises outlier-displacement
    robustness — what RANSAC/IRLS must reject — rather than ghost-track injection. Genuine
    spurious-track injection is a multi-track concern (a separate ghost id), out of scope for
    this single-track corruption.
    """
    if not 0.0 <= fp_rate <= 1.0 or not 0.0 <= drop_rate <= 1.0:
        raise ValueError("fp_rate and drop_rate must be in [0, 1]")
    out: list[Detection] = []
    for det in track.detections:
        if rng.random() < drop_rate:
            continue
        clone = _clone_detection(det)
        if rng.random() < fp_rate:
            jump = rng.uniform(-fp_jump_px, fp_jump_px, size=2)
            clone.bbox = clone.bbox + np.array(
                [jump[0], jump[1], jump[0], jump[1]], dtype=np.float64
            )
            clone.score = None if clone.score is None else 0.5 * clone.score
        out.append(clone)
    return _rebuild(track, out)


@dataclass
class CorruptionConfig:
    """Composed corruption levels. Each block is skipped when its level is the no-op value.

    Application order is fixed and deliberate: jitter and false positives first (they
    perturb measurements that *exist*), then id switches (relabel survivors), then gap
    bursts last (remove runs from whatever remains).
    """

    jitter_sigma_px: float = 0.0
    jitter_rho: float = 0.0
    size_noise_sigma_px: float = 0.0
    size_noise_rho: float = 0.0
    fp_rate: float = 0.0
    drop_rate: float = 0.0
    id_switch_rate: float = 0.0
    gap_burst_len: int = 0
    n_gap_bursts: int = 0


def corrupt(
    track: TrackSequence,
    config: CorruptionConfig,
    rng: np.random.Generator,
) -> TrackSequence:
    """Apply all enabled corruptions from ``config`` in a fixed, deterministic order."""
    out = track
    if config.jitter_sigma_px > 0.0:
        out = apply_correlated_jitter(
            out, sigma_px=config.jitter_sigma_px, rho=config.jitter_rho, rng=rng
        )
    if config.size_noise_sigma_px > 0.0:
        out = apply_size_noise(
            out, sigma_px=config.size_noise_sigma_px, rho=config.size_noise_rho, rng=rng
        )
    if config.fp_rate > 0.0 or config.drop_rate > 0.0:
        out = apply_false_positives_dropouts(
            out, fp_rate=config.fp_rate, drop_rate=config.drop_rate, rng=rng
        )
    if config.id_switch_rate > 0.0:
        out = apply_id_switches(out, rate=config.id_switch_rate, rng=rng)
    if config.gap_burst_len > 0 and config.n_gap_bursts > 0:
        out = apply_gap_bursts(
            out, burst_len=config.gap_burst_len, n_bursts=config.n_gap_bursts, rng=rng
        )
    return out


__all__ = [
    "CorruptionConfig",
    "apply_correlated_jitter",
    "apply_false_positives_dropouts",
    "apply_gap_bursts",
    "apply_id_switches",
    "apply_size_noise",
    "corrupt",
]
