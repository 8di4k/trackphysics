"""Tests for the direct 3D-truth validation path (validation/run_3d_validation.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import generate_track, look_at_camera  # noqa: E402
from validation.run_3d_validation import (  # noqa: E402
    Trajectory3D,
    in_plane_speed,
    validate_trajectory,
)


def _traj(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Trajectory3D:
    cam = look_at_camera(eye=eye, target=target)
    track, gt = generate_track(
        cam, fps=120.0, launch_velocity=(6.0, 0.0, 7.0), drag_coeff=0.2, duration=1.2
    )
    return Trajectory3D(track=track, gt_positions_m=gt.positions, optical_axis=cam.R[2])


def test_in_plane_speed_drops_axis_component() -> None:
    # Velocity purely along the optical axis has zero in-plane speed; perpendicular is full.
    axis = np.array([0.0, 1.0, 0.0])
    assert in_plane_speed(np.array([0.0, 5.0, 0.0]), axis) == 0.0
    assert in_plane_speed(np.array([3.0, 0.0, 4.0]), axis) == 5.0


def test_level_trajectory_ci_covers_inplane_truth() -> None:
    res = validate_trajectory(_traj((3.0, -7.0, 1.5), (3.0, 0.0, 1.7)))
    assert res.tier == "metric"
    assert res.ci_covers_inplane is True
    # In-plane and full-3D truth coincide for a level camera (no depth motion to miss).
    assert abs(res.inplane_truth - res.full3d_truth) < 0.2  # type: ignore[arg-type]


def test_steep_trajectory_is_overconfident() -> None:
    res = validate_trajectory(_traj((3.0, -4.0, 7.0), (3.0, 0.0, 0.5)))
    assert res.tier == "metric"
    assert res.ci_covers_inplane is False  # gravity-as-a-ruler grossly violated -> CI misses
