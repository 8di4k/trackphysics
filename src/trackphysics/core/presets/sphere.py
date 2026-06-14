"""The ``sphere`` physics preset — the single v0.1 generic recognizer (BRIEF.md §16).

A "sphere" here is a *mechanism*, not a domain object: any roughly-spherical free-flying
object whose 2D track admits ballistic-segment detection and a gravity-as-a-ruler fit.
NO concrete numbers are baked in — ``diameter_m`` / ``mass_kg`` / ``drag_coeff`` default to
``None``. A domain layer that knows a specific object's size/mass supplies them by
constructing :class:`SpherePreset` with those values (BRIEF.md §6, §8); they are advisory
inputs to fitting, never domain semantics in the core.

The :class:`~trackphysics.core.presets.base.PhysicsPreset` protocol's ``fit(segment, ctx)``
does not receive the track, but the ballistic fit needs the originating track's pixel
centers. We resolve this **statelessly**: :func:`detect_ballistic_segments` attaches the
originating track to each :class:`~trackphysics.core.schema.Segment` (``source_track``), and
:meth:`SpherePreset.fit` reads it from the segment. The registered preset is therefore safe
to share — interleaved/concurrent ``detect``/``fit`` across tracks cannot contaminate one
another, because the track travels with the segment, not on the preset instance. A legacy
:meth:`bind` cache remains only as a fallback for segments constructed without a
``source_track``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ballistic import detect_ballistic_segments, fit_ballistic
from ..grounding import GroundingContext
from ..provenance import TrajectoryEstimate
from ..schema import Segment, TrackSequence
from .base import EventDetector, register_preset


@dataclass
class SpherePreset:
    """Generic free-flight recognizer implementing the ``PhysicsPreset`` protocol.

    Attributes mirror the protocol. The numeric ones are optional advisory constants a
    domain layer may fill in; the core never requires them and never interprets them as
    semantics.
    """

    name: str = "sphere"
    diameter_m: float | None = None
    mass_kg: float | None = None
    drag_coeff: float | None = None
    magnus: bool = False

    _track: TrackSequence | None = field(default=None, repr=False, compare=False)
    """Legacy fallback track, used only if a segment carries no ``source_track``."""

    def bind(self, track: TrackSequence) -> SpherePreset:
        """Legacy: associate a fallback track for segments lacking ``source_track``.

        Returns ``self`` for chaining. Kept for backward compatibility; the stateless
        ``source_track`` path is preferred and used by :meth:`detect_segments`.
        """
        self._track = track
        return self

    def detect_segments(self, track: TrackSequence) -> list[Segment]:
        """Detect free-flight segments by motion signature (delegates to ballistic core).

        The returned segments carry ``source_track`` so :meth:`fit` is stateless.
        """
        self._track = track  # legacy fallback only; segments carry their own track
        return detect_ballistic_segments(track)

    def fit(self, segment: Segment, ctx: GroundingContext) -> TrajectoryEstimate:
        """Fit a segment via gravity-as-a-ruler (delegates to the ballistic core).

        Uses ``segment.source_track`` (stateless, safe under any interleaving). Falls back
        to a previously bound track only for segments constructed without a ``source_track``.

        Raises:
            RuntimeError: if the segment carries no track and none was bound.
        """
        track = segment.source_track if segment.source_track is not None else self._track
        if track is None:
            raise RuntimeError(
                "SpherePreset.fit needs a track: pass a segment from detect_segments "
                "(which attaches source_track) or call bind(track) first"
            )
        return fit_ballistic(track, segment, ctx)

    def event_detectors(self) -> list[EventDetector]:
        """Return this preset's event detectors.

        The bounce detector lives in ``trackphysics.core.events`` (written by another
        group). It is imported **lazily** here so importing this module never hard-depends
        on ``events`` being present yet.
        """
        from ..events import BounceDetector  # noqa: PLC0415  (intentional lazy import)

        return [BounceDetector()]


# Register the one v0.1 generic preset at import time (BRIEF.md §16: one solid preset).
register_preset(SpherePreset())

__all__ = ["SpherePreset"]
