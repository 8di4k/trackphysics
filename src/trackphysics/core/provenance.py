"""Provenance & confidence model — "validated, not plausible" (BRIEF.md §10).

Every physical quantity the engine emits is a :class:`Quantity` carrying an explicit
:class:`Tier` (is it metric, relative, or pixel-space?) and a calibrated confidence in
``[0, 1]``. No public function returns a bare physical float. Metric tier is emitted
*only* when scale is genuinely recovered; otherwise the engine falls back honestly to a
lower tier — it never fakes metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .schema import FloatArray, Segment


class Tier(Enum):
    """Provenance tier of a physical quantity, in descending strength."""

    METRIC = "metric"
    """Real-world units, scale genuinely recovered or supplied."""

    RELATIVE = "relative"
    """2.5D / normalized; internally consistent but scale-free."""

    PIXEL = "pixel"
    """Image-space 2D plus pixel-rate kinematics."""

    @property
    def rank(self) -> int:
        """Higher is stronger: METRIC=2, RELATIVE=1, PIXEL=0. For ranking estimates."""
        return {Tier.PIXEL: 0, Tier.RELATIVE: 1, Tier.METRIC: 2}[self]


@dataclass
class Quantity:
    """A physical value with provenance. The atomic output unit of the engine.

    ``value`` may be a scalar or an array (e.g. a ``(T, 3)`` position series or a
    ``(T,)`` speed series). ``confidence`` must be calibrated against empirical
    correctness (see the benchmark's reliability plot), not invented.
    """

    value: float | FloatArray
    unit: str | None
    """e.g. ``"m"``, ``"m/s"``, ``"rad"``, ``"rad/s"``, ``"px"``, ``"px/s"``, or
    ``None`` for dimensionless / normalized quantities."""

    tier: Tier
    confidence: float
    source: str
    """How this value was produced, e.g. ``"ballistic_fit"``, ``"finite_difference"``,
    ``"triangulation"``, ``"reference_scale"``."""

    frame: int | tuple[int, int] | None = None
    """The frame or ``(start, end)`` frame range this quantity pertains to."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not isinstance(self.value, (int, float)):
            self.value = np.asarray(self.value, dtype=np.float64)


@dataclass
class TrajectoryEstimate:
    """A motion estimate over a segment, with its overall tier and fit quality."""

    positions: Quantity
    """Array-valued: ``(T, 3)`` (metric/relative) or ``(T, 2)`` (pixel)."""

    velocity: Quantity
    """Array-valued ``(T, 3)`` velocity, or a ``(T,)`` speed series."""

    tier: Tier
    goodness_of_fit: float
    """Physical-plausibility score in ``[0, 1]``: how well observations match the
    fitted physical model (residuals + sanity margins). Gates metric tier (§10)."""

    segment: Segment | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.goodness_of_fit <= 1.0:
            raise ValueError(
                f"goodness_of_fit must be in [0, 1], got {self.goodness_of_fit}"
            )


@dataclass
class Event:
    """A generic, domain-free motion event.

    ``kind`` is one of a small generic vocabulary (``"impact"``, ``"bounce"``,
    ``"contact"``, ``"release"``). A domain layer maps ``kind`` to its own taxonomy
    *outside* the core (BRIEF.md §6, hook 4).
    """

    kind: str
    frame: int
    confidence: float
    payload: dict[str, object] = field(default_factory=dict)
    """e.g. pre/post velocity (as :class:`Quantity`), location (tiered)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


def combine_confidence(*factors: float, weights: tuple[float, ...] | None = None) -> float:
    """Combine sub-confidence factors into a single calibrated-ready confidence.

    Uses a weighted geometric mean so that any single near-zero factor (e.g. a failed
    physical-sanity check) drags the result toward zero — confidence should not survive
    one strong disqualifier just because the other cues look good (§10). The output is
    still subject to empirical *calibration* against the benchmark; this function only
    fixes the combination rule, not the mapping to ground-truth correctness.

    Each factor must lie in ``[0, 1]``. With no factors, returns ``0.0``.
    """
    vals = [max(0.0, min(1.0, f)) for f in factors]
    if not vals:
        return 0.0
    if weights is None:
        w = np.ones(len(vals), dtype=np.float64)
    else:
        if len(weights) != len(vals):
            raise ValueError("weights length must match number of factors")
        w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and sum to a positive value")
    # Weighted geometric mean: exp(sum(w_i * ln v_i) / sum(w_i)); a zero factor -> 0.
    if any(v == 0.0 for v in vals):
        return 0.0
    log_mean = float(np.sum(w * np.log(vals)) / np.sum(w))
    return float(np.exp(log_mean))


__all__ = [
    "Event",
    "Quantity",
    "Tier",
    "TrajectoryEstimate",
    "combine_confidence",
]
