"""Direct 3D-truth validation: single view -> 2D track -> analyze() -> recovered metric vs
held-out independent 3D ground truth (e.g. TT3D's multi-camera 3D).

This supersedes the ruler branch when a dataset carries INDEPENDENT 3D truth (not
monocular-reconstructed — see validation/README.md). For each trajectory we run the engine
on the single-view 2D track and compare the recovered launch speed against the true speed at
the engine's reported instant (the segment-start frame).

Honest scope: v0.1 is monocular and recovers only IN-PLANE metric (the depth/optical-axis
component is zeroed). So the primary, fair comparison is against the IN-PLANE projection of
the true 3D velocity. We ALSO report the full-3D error to expose the depth-blindness (large
when the object moves substantially toward/away from the camera — the regime stereo/v0.2
addresses). CI coverage is checked against the in-plane truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import trackphysics as tp
from trackphysics.core.schema import FloatArray, TrackSequence


@dataclass
class Trajectory3D:
    """One trajectory: a single-view 2D track plus its independent 3D ground truth.

    ``gt_positions_m`` is aligned 1:1 with ``track.detections`` (row i ↔ detection i).
    ``optical_axis`` is the camera viewing direction in world coords (unit); the in-plane
    projection drops the component along it. For a calibrated dataset it is the camera +Z
    axis in world (e.g. ``CameraSpec.R[2]``).
    """

    track: TrackSequence
    gt_positions_m: FloatArray   # (T, 3) world meters, aligned to track.detections
    optical_axis: FloatArray     # (3,) unit
    name: str = ""


@dataclass
class TrajectoryResult:
    name: str
    tier: str
    recovered_speed: float | None = None
    inplane_truth: float | None = None
    full3d_truth: float | None = None
    inplane_error: float | None = None
    full3d_error: float | None = None
    ci_covers_inplane: bool | None = None


@dataclass
class Validation3DReport:
    results: list[TrajectoryResult] = field(default_factory=list)

    @property
    def metric_results(self) -> list[TrajectoryResult]:
        return [r for r in self.results if r.tier == "metric" and r.inplane_error is not None]

    def summary(self) -> dict[str, float]:
        m = self.metric_results
        n = len(m)
        if n == 0:
            return {"n_total": float(len(self.results)), "n_metric": 0.0}
        inplane = np.array([r.inplane_error for r in m], dtype=np.float64)
        full3d = np.array([r.full3d_error for r in m], dtype=np.float64)
        covered = sum(1 for r in m if r.ci_covers_inplane)
        return {
            "n_total": float(len(self.results)),
            "n_metric": float(n),
            "mean_inplane_error_m_s": float(np.mean(inplane)),
            "median_inplane_error_m_s": float(np.median(inplane)),
            "mean_full3d_error_m_s": float(np.mean(full3d)),
            "ci_coverage_rate": covered / n,
        }


def in_plane_speed(velocity: FloatArray, optical_axis: FloatArray) -> float:
    """Speed of the component of ``velocity`` perpendicular to the optical axis (m/s).

    This is the quantity a monocular engine can recover; the along-axis (depth) component is
    invisible to it.
    """
    axis = optical_axis / float(np.linalg.norm(optical_axis))
    perp = velocity - float(velocity @ axis) * axis
    return float(np.linalg.norm(perp))


def _true_velocity(positions: FloatArray, times: FloatArray) -> FloatArray:
    if positions.shape[0] < 2:
        return np.zeros_like(positions)
    vel = np.empty_like(positions)
    for axis in range(positions.shape[1]):
        vel[:, axis] = np.gradient(positions[:, axis], times, edge_order=1)
    return vel


def validate_trajectory(traj: Trajectory3D) -> TrajectoryResult:
    """Run the engine on the 2D track and compare the recovered speed to the 3D truth."""
    est = tp.analyze(traj.track, preset="sphere", grounding=tp.GroundingContext()).trajectory
    if est.tier is not tp.Tier.METRIC:
        return TrajectoryResult(name=traj.name, tier=est.tier.value)

    frame = est.velocity.frame
    if not isinstance(frame, tuple):
        return TrajectoryResult(name=traj.name, tier="metric")
    matches = np.where(traj.track.frames == int(frame[0]))[0]
    if matches.size == 0:
        return TrajectoryResult(name=traj.name, tier="metric")
    i = int(matches[0])

    times = traj.track.times()
    v_true = _true_velocity(traj.gt_positions_m, times)[i]
    inplane = in_plane_speed(v_true, traj.optical_axis)
    full3d = float(np.linalg.norm(v_true))
    recovered = float(est.meta["launch_speed_m_s"])  # type: ignore[arg-type]
    ci = est.meta.get("launch_speed_ci95")
    covers = None
    if ci is not None:
        lo, hi = float(ci[0]), float(ci[1])  # type: ignore[index]
        covers = lo <= inplane <= hi
    return TrajectoryResult(
        name=traj.name, tier="metric", recovered_speed=recovered,
        inplane_truth=inplane, full3d_truth=full3d,
        inplane_error=abs(recovered - inplane), full3d_error=abs(recovered - full3d),
        ci_covers_inplane=covers,
    )


def run(trajectories: list[Trajectory3D]) -> Validation3DReport:
    return Validation3DReport(results=[validate_trajectory(t) for t in trajectories])


__all__ = [
    "Trajectory3D",
    "TrajectoryResult",
    "Validation3DReport",
    "in_plane_speed",
    "run",
    "validate_trajectory",
]
