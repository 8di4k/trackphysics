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


__all__ = ["GroundingContext", "Plane"]
