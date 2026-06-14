"""Synthetic ground-truth generator for the benchmark substrate (BRIEF.md §14.1).

Generates exact-metric 3D trajectories from a physics model (gravity + quadratic
aerodynamic drag, optional Magnus lift), projects them through known camera
intrinsics/extrinsics, and packages the result as a :class:`TrackSequence` plus a
:class:`GroundTruth` record holding the true 3D state and metric scale. Because we
*generated* the scene, the absolute metric error is knowable — impossible on
uninstrumented real video.
"""

from __future__ import annotations

from .generator import (
    CameraSpec,
    GroundTruth,
    arc_scenario,
    generate_track,
    look_at_camera,
    project_points,
    simulate_trajectory,
    steep_arc_scenario,
)

__all__ = [
    "CameraSpec",
    "GroundTruth",
    "arc_scenario",
    "generate_track",
    "look_at_camera",
    "project_points",
    "simulate_trajectory",
    "steep_arc_scenario",
]
