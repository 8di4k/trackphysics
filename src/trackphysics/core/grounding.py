"""Grounding context — the generic, optional hook by which scale enters the core.

The core knows nothing about courts, fields, or roads. It only accepts a *generic*
known plane and/or a *generic* known metric scale if a domain layer chooses to supply
one (BRIEF.md §6, hook 3). If neither is given, the engine attempts gravity-as-a-ruler;
if that fails (no ballistic segment), it falls back to the RELATIVE tier — it never
fabricates a METRIC result (BRIEF.md §7, §10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import FloatArray


@dataclass
class Plane:
    """An infinite plane in the caller's metric coordinate frame."""

    point: FloatArray
    """A point on the plane, shape ``(3,)``."""

    normal: FloatArray
    """Plane normal, shape ``(3,)``; normalized on construction."""

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        if self.point.shape != (3,) or self.normal.shape != (3,):
            raise ValueError("point and normal must both have shape (3,)")
        norm = float(np.linalg.norm(self.normal))
        if norm == 0.0:
            raise ValueError("normal must be non-zero")
        self.normal = self.normal / norm


@dataclass
class DepthDominationGuard:
    """Opt-in (default-OFF) point-only guard for the §10 depth-blindness tier-hole.

    A trajectory flying along the optical axis still projects a *clean in-plane parabola*,
    which the metric gate trusts — so the engine emits METRIC most readily exactly where it
    is most 3D-blind, and the in-plane speed it returns is a small, unflagged fraction of the
    true 3D motion. The 2D trajectory SHAPE is a point-only proxy: a depth-dominated arc has a
    low **in-plane aspect ratio** (horizontal/vertical pixel extent). This guard, when
    enabled, uses that ratio to *continuously* discount metric confidence (and widen the CI)
    as the arc looks more depth-dominated, with a *conservative hard downgrade*
    (METRIC→RELATIVE) only past a 'hopeless' floor.

    IMPORTANT: flagging depth-domination is NOT recovering depth (that needs stereo / a real
    detector's size channel). This only stops the engine over-trusting an in-plane metric.

    The thresholds are a REFITTABLE artifact, NOT a universal constant: the defaults are
    *calibrated on a single evaluation set (TT3D)* and are **default-OFF**. Do not promote to
    default-ON, and do not hardcode domain-specific values into the core, until validated on a
    second domain (§6, §10). ``aspect`` is pure 2D geometry (no object semantics), so the
    mechanism is domain-agnostic; only the tuned thresholds are domain-flavoured.
    """

    enabled: bool = False
    soft_aspect: float = 2.5
    """Aspect at/above which no penalty applies (clearly in-plane motion)."""
    hard_aspect: float = 0.8
    """Aspect at/below which METRIC is conservatively downgraded to RELATIVE (hopeless depth
    domination). Must not exceed ``soft_aspect``."""
    confidence_penalty: float = 0.6
    """Max fractional confidence reduction, reached as aspect → ``hard_aspect`` (in ``[0,1]``)."""
    ci_widen: float = 1.0
    """The CI's systematic half-width is scaled by ``1 + ci_widen * depth_score`` (the
    recovered scale is also depth-biased), so a depth-suspect metric ships a wider band."""

    def __post_init__(self) -> None:
        if self.hard_aspect > self.soft_aspect:
            raise ValueError("hard_aspect must not exceed soft_aspect")
        if not 0.0 <= self.confidence_penalty <= 1.0:
            raise ValueError("confidence_penalty must be in [0, 1]")
        if self.ci_widen < 0.0:
            raise ValueError("ci_widen must be non-negative")

    def depth_score(self, aspect: float) -> float:
        """Continuous depth-domination score in ``[0, 1]`` from the in-plane aspect ratio
        (0 = clearly in-plane, 1 = at the hard floor). No cliff: scales linearly between the
        soft and hard aspect thresholds."""
        span = self.soft_aspect - self.hard_aspect
        if span <= 0.0:
            return 1.0 if aspect <= self.hard_aspect else 0.0
        return float(min(1.0, max(0.0, (self.soft_aspect - aspect) / span)))


@dataclass
class GroundingContext:
    """Optional scale/orientation references plus the gravity constant used for checks.

    Supply ``reference_plane`` and/or ``reference_scale`` to ground estimates metrically;
    leave both ``None`` to let the engine try gravity-as-a-ruler and fall back to RELATIVE
    on failure.
    """

    reference_plane: Plane | None = None
    reference_scale: float | None = None
    """A known metric scale, e.g. meters-per-pixel on a reference plane, or a known
    object size used to fix scale."""

    gravity: float = 9.81
    """Gravitational acceleration in ``m/s^2``, used both as the ruler in ballistic
    fitting and as a physical-sanity reference."""

    depth_guard: DepthDominationGuard | None = None
    """Opt-in point-only depth-domination guard (§10 tier-hole). ``None`` or disabled →
    behaviour is unchanged. See :class:`DepthDominationGuard`."""

    def __post_init__(self) -> None:
        if self.reference_scale is not None and self.reference_scale <= 0:
            raise ValueError("reference_scale must be positive when provided")
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")

    @property
    def has_metric_reference(self) -> bool:
        """True if the caller supplied any explicit metric grounding (plane and/or scale).

        Note: a ``reference_plane`` alone does NOT fix a pixel-to-meter scale in monocular
        v0.1 (that needs camera intrinsics, a v0.2 concern) — use :attr:`has_metric_scale`
        to ask whether scale is actually grounded.
        """
        return self.reference_plane is not None or self.reference_scale is not None

    @property
    def has_metric_scale(self) -> bool:
        """True if the context can actually fix a metric scale in v0.1.

        Only a ``reference_scale`` does so monocularly; a plane alone cannot (see
        :attr:`has_metric_reference`). This is what determines whether the engine could earn
        METRIC tier from the grounding rather than from gravity-as-a-ruler."""
        return self.reference_scale is not None


__all__ = ["DepthDominationGuard", "GroundingContext", "Plane"]
