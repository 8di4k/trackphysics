"""Synthetic exact-metric ground-truth generator (BRIEF.md §14.1).

The pipeline is:

1. :func:`simulate_trajectory` integrates the true 3D motion of a point object under
   gravity + quadratic aerodynamic drag (and an optional Magnus lift term) with a
   dependency-light RK4 integrator.
2. :func:`project_points` maps the true 3D path through a known pinhole
   :class:`CameraSpec` to clean 2D pixel coordinates.
3. :func:`generate_track` assembles a :class:`TrackSequence` of bounding boxes (built
   from the projected centre and a depth-scaled projected radius) together with a
   :class:`GroundTruth` record carrying the true 3D positions/velocities, the true
   speed series, the camera, and the local metric scale (meters-per-pixel near the arc).

Everything here is domain-agnostic: a "free-flight" arc of a nameless object, never any
specific kind of object (BRIEF.md §6). This module is benchmark substrate only and is
*not* part of the installable core's dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trackphysics.core.schema import Detection, FloatArray, TrackSequence


@dataclass
class CameraSpec:
    """A pinhole camera: intrinsics ``K``, world->camera rotation ``R``/translation ``t``.

    A world point ``X`` (shape ``(3,)``) projects as ``x_cam = R @ X + t`` (camera frame,
    +Z forward) then ``p = K @ x_cam`` with pixel coords ``p[:2] / p[2]``.
    """

    K: FloatArray
    """Intrinsic matrix, shape ``(3, 3)``."""

    R: FloatArray
    """World-to-camera rotation, shape ``(3, 3)``."""

    t: FloatArray
    """World-to-camera translation, shape ``(3,)``."""

    image_size: tuple[int, int]
    """``(W, H)`` in pixels."""

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {self.K.shape}")
        if self.R.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must have shape (3,), got {self.t.shape}")

    @property
    def focal_px(self) -> float:
        """Mean focal length in pixels, ``(fx + fy) / 2``."""
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))


@dataclass
class GroundTruth:
    """Exact metric ground truth accompanying a synthetic :class:`TrackSequence`."""

    positions: FloatArray
    """True 3D positions, shape ``(T, 3)`` in meters (world frame)."""

    velocities: FloatArray
    """True 3D velocities, shape ``(T, 3)`` in m/s (world frame)."""

    speed: FloatArray
    """True scalar speed series ``|v|``, shape ``(T,)`` in m/s."""

    times: FloatArray
    """Per-frame time, shape ``(T,)`` in seconds."""

    camera: CameraSpec
    metric_scale: float
    """Meters-per-pixel near the arc (median depth / focal length). A monocular scale
    reference for sanity-checking recovered metric tier."""

    frames: FloatArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    """Frame indices actually retained in the emitted track, shape ``(T,)``."""


def _accel(
    pos: FloatArray,
    vel: FloatArray,
    g_vec: FloatArray,
    drag_k: float,
    magnus_vec: FloatArray | None,
) -> FloatArray:
    """Acceleration of a point object: gravity + quadratic drag (+ optional Magnus).

    ``a = g_vec - drag_k * |v| * v + (magnus_vec x v)``. ``drag_k`` bundles the
    aerodynamic factor ``0.5 * rho * Cd * A / m`` (units 1/m). ``pos`` is accepted for
    signature symmetry with general force models but is unused for free flight.
    """
    del pos  # free-flight forces depend only on velocity
    speed = float(np.linalg.norm(vel))
    a = g_vec - drag_k * speed * vel
    if magnus_vec is not None:
        a = a + np.cross(magnus_vec, vel)
    return a


def simulate_trajectory(
    *,
    launch_position: FloatArray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    launch_velocity: FloatArray | tuple[float, float, float] = (8.0, 0.0, 6.0),
    mass_kg: float = 0.05,
    diameter_m: float = 0.065,
    drag_coeff: float = 0.47,
    air_density: float = 1.225,
    gravity: float = 9.81,
    fps: float = 120.0,
    duration: float = 1.2,
    magnus_coeff: float = 0.0,
    spin_axis: FloatArray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate true 3D free-flight motion with RK4 (BRIEF.md §14.1).

    The world frame is +Z up, so gravity acts as ``(0, 0, -gravity)``. Drag is quadratic:
    ``a_drag = -(0.5 * rho * Cd * A / m) * |v| * v``. An optional Magnus lift term
    ``magnus_coeff * (spin_axis_unit x v)`` is added when ``magnus_coeff != 0``.

    Returns ``(positions, velocities, times)`` with shapes ``(T, 3)``, ``(T, 3)``,
    ``(T,)``. ``T = floor(duration * fps) + 1`` evenly spaced samples at ``1/fps``.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if mass_kg <= 0 or diameter_m <= 0:
        raise ValueError("mass_kg and diameter_m must be positive")

    p0 = np.asarray(launch_position, dtype=np.float64)
    v0 = np.asarray(launch_velocity, dtype=np.float64)
    if p0.shape != (3,) or v0.shape != (3,):
        raise ValueError("launch_position and launch_velocity must have shape (3,)")

    area = np.pi * (0.5 * diameter_m) ** 2
    drag_k = 0.5 * air_density * drag_coeff * area / mass_kg
    g_vec = np.array([0.0, 0.0, -gravity], dtype=np.float64)

    magnus_vec: FloatArray | None = None
    if magnus_coeff != 0.0:
        axis = np.asarray(spin_axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm == 0.0:
            raise ValueError("spin_axis must be non-zero when magnus_coeff != 0")
        magnus_vec = magnus_coeff * (axis / axis_norm)

    n_steps = int(np.floor(duration * fps))
    dt = 1.0 / fps
    times = np.arange(n_steps + 1, dtype=np.float64) * dt

    positions = np.empty((n_steps + 1, 3), dtype=np.float64)
    velocities = np.empty((n_steps + 1, 3), dtype=np.float64)
    pos = p0.copy()
    vel = v0.copy()
    positions[0] = pos
    velocities[0] = vel

    for i in range(1, n_steps + 1):
        k1v = _accel(pos, vel, g_vec, drag_k, magnus_vec)
        k1x = vel
        k2v = _accel(pos + 0.5 * dt * k1x, vel + 0.5 * dt * k1v, g_vec, drag_k, magnus_vec)
        k2x = vel + 0.5 * dt * k1v
        k3v = _accel(pos + 0.5 * dt * k2x, vel + 0.5 * dt * k2v, g_vec, drag_k, magnus_vec)
        k3x = vel + 0.5 * dt * k2v
        k4v = _accel(pos + dt * k3x, vel + dt * k3v, g_vec, drag_k, magnus_vec)
        k4x = vel + dt * k3v
        pos = pos + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        vel = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        positions[i] = pos
        velocities[i] = vel

    return positions, velocities, times


def project_points(points3d: FloatArray, cam: CameraSpec) -> FloatArray:
    """Project world points through a pinhole camera to pixel coordinates.

    ``points3d`` has shape ``(T, 3)``; returns ``(T, 2)`` pixel ``[x, y]``. Points with a
    non-positive camera-frame depth (behind the camera) yield ``NaN`` pixels.
    """
    pts = np.asarray(points3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points3d must have shape (T, 3), got {pts.shape}")
    cam_pts = pts @ cam.R.T + cam.t  # (T, 3) in camera frame
    depth = cam_pts[:, 2]
    img_h = cam_pts @ cam.K.T  # (T, 3) homogeneous image coords
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    valid = depth > 0
    out[valid] = img_h[valid, :2] / depth[valid, None]
    return out


def camera_depths(points3d: FloatArray, cam: CameraSpec) -> FloatArray:
    """Camera-frame Z (depth) of each world point, shape ``(T,)``."""
    pts = np.asarray(points3d, dtype=np.float64)
    return (pts @ cam.R.T + cam.t)[:, 2]


def look_at_camera(
    *,
    eye: FloatArray | tuple[float, float, float],
    target: FloatArray | tuple[float, float, float],
    focal_px: float = 900.0,
    image_size: tuple[int, int] = (1280, 720),
    up: FloatArray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> CameraSpec:
    """Build a :class:`CameraSpec` for a camera at ``eye`` looking toward ``target``.

    Uses a right-handed look-at with the camera +Z pointing from ``eye`` to ``target``
    (so objects in front have positive depth). The principal point is the image centre.
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    forward = target_v - eye_v
    fnorm = float(np.linalg.norm(forward))
    if fnorm == 0.0:
        raise ValueError("eye and target must differ")
    forward = forward / fnorm

    right = np.cross(forward, up_v)
    rnorm = float(np.linalg.norm(right))
    if rnorm == 0.0:
        raise ValueError("up vector must not be parallel to the view direction")
    right = right / rnorm
    true_up = np.cross(right, forward)

    # Camera axes as world-frame rows: x=right, y=down (=-up), z=forward.
    rot = np.stack([right, -true_up, forward], axis=0)
    trans = -rot @ eye_v

    w, h = image_size
    k = np.array(
        [[focal_px, 0.0, 0.5 * w], [0.0, focal_px, 0.5 * h], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraSpec(K=k, R=rot, t=trans, image_size=image_size)


def generate_track(
    cam: CameraSpec,
    *,
    track_id: int = 1,
    fps: float = 120.0,
    keep_in_frame: bool = True,
    launch_position: FloatArray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    launch_velocity: FloatArray | tuple[float, float, float] = (8.0, 0.0, 6.0),
    mass_kg: float = 0.05,
    diameter_m: float = 0.065,
    drag_coeff: float = 0.47,
    air_density: float = 1.225,
    gravity: float = 9.81,
    duration: float = 1.2,
    magnus_coeff: float = 0.0,
    spin_axis: FloatArray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[TrackSequence, GroundTruth]:
    """Simulate, project, and package a synthetic track with exact ground truth.

    The motion parameters mirror :func:`simulate_trajectory` and are forwarded to it. Each
    frame's bounding box is built from the projected centre plus a depth-scaled projected
    radius ``r_px = focal * (diameter / 2) / depth``. When ``keep_in_frame`` is True, frames
    whose projected centre falls outside the image (or behind the camera) are dropped,
    mimicking a real arc that enters and exits view.

    Returns ``(track, ground_truth)``. The two are frame-aligned: ``ground_truth.positions``
    etc. are filtered to exactly the frames retained in ``track``.
    """
    positions, velocities, times = simulate_trajectory(
        launch_position=launch_position,
        launch_velocity=launch_velocity,
        mass_kg=mass_kg,
        diameter_m=diameter_m,
        drag_coeff=drag_coeff,
        air_density=air_density,
        gravity=gravity,
        fps=fps,
        duration=duration,
        magnus_coeff=magnus_coeff,
        spin_axis=spin_axis,
    )

    pixels = project_points(positions, cam)
    depths = camera_depths(positions, cam)
    speed = np.linalg.norm(velocities, axis=1)

    w, h = cam.image_size
    in_view = (
        np.isfinite(pixels).all(axis=1)
        & (depths > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    keep = in_view if keep_in_frame else np.isfinite(pixels).all(axis=1)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("no in-frame samples; check camera placement / launch parameters")

    radius_px = cam.focal_px * (0.5 * diameter_m) / depths[idx]
    detections: list[Detection] = []
    for local, frame_idx in enumerate(idx):
        cx, cy = pixels[frame_idx]
        r = float(radius_px[local])
        bbox = np.array([cx - r, cy - r, cx + r, cy + r], dtype=np.float64)
        detections.append(
            Detection(frame=int(frame_idx), bbox=bbox, track_id=track_id, score=1.0)
        )

    track = TrackSequence(detections=detections, fps=fps, image_size=(w, h))

    # Metric scale near the arc: median depth / focal -> meters per pixel at that depth.
    metric_scale = float(np.median(depths[idx]) / cam.focal_px)

    gt = GroundTruth(
        positions=positions[idx],
        velocities=velocities[idx],
        speed=speed[idx],
        times=times[idx],
        camera=cam,
        metric_scale=metric_scale,
        frames=idx.astype(np.float64),
    )
    return track, gt


def arc_scenario(*, fps: float = 120.0) -> tuple[TrackSequence, GroundTruth]:
    """A gentle free-flight arc viewed side-on (a ready-made generic scenario)."""
    cam = look_at_camera(eye=(0.0, -8.0, 1.5), target=(4.0, 0.0, 1.5), focal_px=900.0)
    return generate_track(
        cam,
        fps=fps,
        launch_position=(0.0, 0.0, 1.0),
        launch_velocity=(7.0, 0.0, 6.0),
        drag_coeff=0.2,
        duration=1.4,
    )


def steep_arc_scenario(*, fps: float = 120.0) -> tuple[TrackSequence, GroundTruth]:
    """A steep, high-launch free-flight arc (a second ready-made generic scenario)."""
    cam = look_at_camera(eye=(2.0, -10.0, 2.0), target=(2.0, 0.0, 4.0), focal_px=1000.0)
    return generate_track(
        cam,
        fps=fps,
        launch_position=(0.0, 0.0, 1.0),
        launch_velocity=(3.0, 0.5, 11.0),
        drag_coeff=0.25,
        duration=2.0,
    )
