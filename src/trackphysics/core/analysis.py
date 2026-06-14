"""Top-level entry point.

``analyze`` is the small, stable public surface (BRIEF.md §16): the schema in, an
:class:`AnalysisResult` out. It orchestrates the pieces without owning their physics:

* a physics preset (if given) detects free-flight segments and fits each one;
* the preset's event detectors run over its estimates;
* a relative-3D lift is *always* appended as the tier floor (§12.1) — so even with no
  metric scale, callers get a usable, honestly-labelled RELATIVE estimate;
* keypoint-graph kinematics and a track-quality report are computed unconditionally;
* with no preset (tier-limited generic analysis), default generic event detectors run
  over the relative lift so bounce/release events are still surfaced.

Provenance is preserved end to end: nothing here promotes a tier. The primary estimate
(``AnalysisResult.trajectory``) is chosen by tier strength then goodness-of-fit, so a
genuine METRIC fit outranks the RELATIVE floor, but the floor is never discarded.
"""

from __future__ import annotations

from .events import BounceDetector, ReleaseDetector
from .grounding import GroundingContext
from .kinematics import compute_kinematics
from .lift import relative_lift
from .presets.base import PhysicsPreset, get_preset
from .provenance import Event, TrajectoryEstimate
from .results import AnalysisResult
from .robust.quality import assess_quality
from .schema import TrackSequence


def _resolve_preset(preset: str | PhysicsPreset | None) -> PhysicsPreset | None:
    if preset is None:
        return None
    if isinstance(preset, str):
        return get_preset(preset)
    return preset


def analyze(
    track: TrackSequence,
    *,
    preset: str | PhysicsPreset | None = None,
    grounding: GroundingContext | None = None,
) -> AnalysisResult:
    """Analyze one track into motion physics with provenance.

    Args:
        track: The input track (boxes + ids, optionally keypoints).
        preset: A registered preset name, a :class:`PhysicsPreset` instance, or ``None``
            for tier-limited generic analysis (relative lift + kinematics + generic
            events only).
        grounding: Optional metric grounding. When absent, the engine attempts
            gravity-as-a-ruler and falls back to the RELATIVE tier on failure.

    Returns:
        An :class:`AnalysisResult` bundling trajectory estimate(s), events, kinematics,
        per-quantity provenance, and a track-quality report.
    """
    ctx = grounding if grounding is not None else GroundingContext()
    resolved = _resolve_preset(preset)

    trajectories: list[TrajectoryEstimate] = []
    events: list[Event] = []

    # Preset-driven physics: detect free-flight segments, fit each, run the preset's
    # event detectors over the resulting estimates.
    if resolved is not None:
        segments = resolved.detect_segments(track)
        detectors = resolved.event_detectors()
        for segment in segments:
            est = resolved.fit(segment, ctx)
            trajectories.append(est)
            for detector in detectors:
                events.extend(detector.detect(est, track))

    # Always-available RELATIVE-tier floor (§12.1). Selection by (tier, fit) means this
    # never masks a stronger METRIC estimate, but guarantees a usable estimate exists.
    floor = relative_lift(track)
    trajectories.append(floor)

    # Generic mode (no preset): still surface generic events, computed off the floor.
    if resolved is None:
        for detector in (BounceDetector(), ReleaseDetector()):
            events.extend(detector.detect(floor, track))

    kinematics = compute_kinematics(track)
    quality = assess_quality(track)

    meta: dict[str, object] = {
        "preset": resolved.name if resolved is not None else None,
        "n_trajectories": len(trajectories),
        "tiers": [t.tier.value for t in trajectories],
        # Report whether scale is actually grounded (a plane alone does not fix scale in
        # v0.1), so the flag matches the tier the engine could earn from the grounding.
        "grounded": ctx.has_metric_scale,
    }
    return AnalysisResult(
        trajectories=trajectories,
        events=events,
        kinematics=kinematics,
        quality=quality,
        meta=meta,
    )


__all__ = ["analyze"]
