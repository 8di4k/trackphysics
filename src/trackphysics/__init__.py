"""trackphysics — object-agnostic physics layer for computer-vision tracks.

The public surface is intentionally small and stable (BRIEF.md §16): the input schema,
the three-tier provenance types, the preset protocol + registry, the grounding hook, and
:func:`analyze`.
"""

from __future__ import annotations

from .core.adapters.generic import from_csv, from_generic
from .core.adapters.supervision import from_supervision
from .core.analysis import analyze
from .core.events import BounceDetector, ReleaseDetector, detect_contacts
from .core.grounding import DepthDominationGuard, GroundingContext, Plane
from .core.kinematics import compute_kinematics
from .core.lift import relative_lift
from .core.presets import SpherePreset
from .core.presets.base import (
    EventDetector,
    PhysicsPreset,
    get_preset,
    list_presets,
    register_preset,
)
from .core.provenance import Event, Quantity, Tier, TrajectoryEstimate, combine_confidence
from .core.results import (
    AnalysisResult,
    KinematicsResult,
    QualityFlag,
    TrackQualityReport,
)
from .core.robust.quality import assess_quality
from .core.schema import Detection, FloatArray, Segment, SkeletonGraph, TrackSequence

__version__ = "0.1.0.dev0"

__all__ = [
    "AnalysisResult",
    "BounceDetector",
    "DepthDominationGuard",
    "Detection",
    "Event",
    "EventDetector",
    "FloatArray",
    "GroundingContext",
    "KinematicsResult",
    "PhysicsPreset",
    "Plane",
    "QualityFlag",
    "Quantity",
    "ReleaseDetector",
    "Segment",
    "SkeletonGraph",
    "SpherePreset",
    "Tier",
    "TrackQualityReport",
    "TrackSequence",
    "TrajectoryEstimate",
    "__version__",
    "analyze",
    "assess_quality",
    "combine_confidence",
    "compute_kinematics",
    "detect_contacts",
    "from_csv",
    "from_generic",
    "from_supervision",
    "get_preset",
    "list_presets",
    "register_preset",
    "relative_lift",
]
