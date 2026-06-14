"""Physics preset protocol + registry — the runtime hook for domain physics.

A preset bundles the *mechanism* for a class of motion (how to detect its segments, how
to fit them, what events to look for). Generic presets live in the core (e.g. a
``sphere`` mechanism); concrete domain constants (a specific object's diameter/mass) are
supplied by a domain layer that registers or parameterizes a preset (BRIEF.md §6, §8).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..grounding import GroundingContext
from ..provenance import Event, TrajectoryEstimate
from ..schema import Segment, TrackSequence


@runtime_checkable
class EventDetector(Protocol):
    """Detects generic events given a trajectory estimate and the originating track."""

    def detect(self, est: TrajectoryEstimate, track: TrackSequence) -> list[Event]: ...


@runtime_checkable
class PhysicsPreset(Protocol):
    """A recognizer for a class of motion. Implementations are usually dataclasses.

    The numeric attributes are optional: a generic preset may carry ``None`` for all of
    them and have a domain layer fill them in. They are advisory inputs to ``fit``, never
    domain semantics by themselves.
    """

    name: str
    diameter_m: float | None
    mass_kg: float | None
    drag_coeff: float | None
    magnus: bool

    def detect_segments(self, track: TrackSequence) -> list[Segment]:
        """Find candidate segments matching this preset's motion signature."""
        ...

    def fit(self, segment: Segment, ctx: GroundingContext) -> TrajectoryEstimate:
        """Fit the physical model over a segment, producing a tiered estimate."""
        ...

    def event_detectors(self) -> list[EventDetector]:
        """Event detectors appropriate to this preset's motion."""
        ...


_REGISTRY: dict[str, PhysicsPreset] = {}


def register_preset(preset: PhysicsPreset) -> None:
    """Register a preset under its ``name``. Re-registering the same name overwrites."""
    _REGISTRY[preset.name] = preset


def get_preset(name: str) -> PhysicsPreset:
    """Look up a registered preset by name. Raises :class:`KeyError` if absent."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"no preset named {name!r}; registered presets: {available}"
        ) from exc


def list_presets() -> list[str]:
    """Names of all registered presets, sorted."""
    return sorted(_REGISTRY)


__all__ = [
    "EventDetector",
    "PhysicsPreset",
    "get_preset",
    "list_presets",
    "register_preset",
]
