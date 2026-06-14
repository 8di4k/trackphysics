"""Aggregate result containers returned by :func:`trackphysics.analyze`.

Track-quality diagnostics are a first-class product feature, not debug output: the
engine reports *where* a track is unreliable rather than silently emitting a best guess
(BRIEF.md §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import Event, Quantity, TrajectoryEstimate


@dataclass
class KinematicsResult:
    """Scale-invariant kinematics from a keypoint graph (BRIEF.md §12.4).

    Keyed by the angle triple ``(a, b, c)`` of keypoint indices, where the angle is
    measured at the middle vertex ``b``. Values are array-valued :class:`Quantity`
    objects (a ``(T,)`` series), valid even at PIXEL/RELATIVE tier because angles are
    scale-invariant.
    """

    angles: dict[tuple[int, int, int], Quantity] = field(default_factory=dict)
    angular_velocities: dict[tuple[int, int, int], Quantity] = field(default_factory=dict)


@dataclass
class QualityFlag:
    """A flagged span of a track where reliability is degraded."""

    start_frame: int
    end_frame: int
    reason: str
    """Generic, e.g. ``"gap"``, ``"jitter"``, ``"id_switch"``, ``"low_density"``."""

    severity: float
    """In ``[0, 1]``; higher means less trustworthy."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")


@dataclass
class TrackQualityReport:
    """Where, and how badly, a track is unreliable."""

    flags: list[QualityFlag] = field(default_factory=list)
    completeness: float = 1.0
    """Fraction of expected frames actually observed over the track span, in ``[0, 1]``."""

    overall_score: float = 1.0
    """Aggregate trustworthiness in ``[0, 1]``."""

    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Everything :func:`trackphysics.analyze` produces for one track."""

    trajectories: list[TrajectoryEstimate] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    kinematics: KinematicsResult | None = None
    quality: TrackQualityReport = field(default_factory=TrackQualityReport)
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def trajectory(self) -> TrajectoryEstimate:
        """The primary trajectory estimate (strongest tier, then best fit).

        Convenience accessor for the common single-arc case (the hero demo uses
        ``res.trajectory.velocity``). Raises if no trajectory was produced.
        """
        if not self.trajectories:
            raise ValueError("no trajectory estimate available for this track")
        return max(
            self.trajectories,
            key=lambda e: (e.tier.rank, e.goodness_of_fit),
        )


__all__ = [
    "AnalysisResult",
    "KinematicsResult",
    "QualityFlag",
    "TrackQualityReport",
]
