"""Tests for Group G5: relative-3D lift + keypoint-graph kinematics (BRIEF.md §12.1, §12.4).

All synthetic fixtures are built inline (no dependency on bench or other groups). The suite
covers both the success path (correct geometry / kinematics) and the honesty path (RELATIVE
tier floor never claims metric; PIXEL tier on projected angles; graceful empty / no-skeleton
fallback; low-confidence keypoint gating).
"""

from __future__ import annotations

import math

import numpy as np

from trackphysics.core.kinematics import compute_kinematics
from trackphysics.core.lift import relative_lift
from trackphysics.core.provenance import Tier
from trackphysics.core.results import KinematicsResult
from trackphysics.core.schema import Detection, SkeletonGraph, TrackSequence

# --------------------------------------------------------------------------- helpers


def _box_at(cx: float, cy: float, size: float) -> np.ndarray:
    """A square xyxy bbox of side ``size`` centred at ``(cx, cy)``."""
    h = 0.5 * size
    return np.array([cx - h, cy - h, cx + h, cy + h], dtype=np.float64)


def _moving_track(n: int = 12, fps: float = 30.0) -> TrackSequence:
    """A track that translates across the image while its apparent size shrinks."""
    dets: list[Detection] = []
    for f in range(n):
        cx = 100.0 + 20.0 * f
        cy = 200.0 - 5.0 * f
        size = 60.0 - 2.0 * f  # shrinking -> receding in the inverse-size depth proxy
        dets.append(Detection(frame=f, bbox=_box_at(cx, cy, size), track_id=1))
    return TrackSequence(detections=dets, fps=fps, image_size=(1280, 720))


def _keypoint_detection(
    frame: int, pts: list[tuple[float, float]], confs: list[float] | None = None
) -> Detection:
    """A detection whose bbox encloses the keypoints, carrying (K,2) or (K,3) keypoints."""
    arr2 = np.array(pts, dtype=np.float64)
    if confs is None:
        kp: np.ndarray = arr2
    else:
        kp = np.column_stack([arr2, np.array(confs, dtype=np.float64)])
    x0, y0 = arr2.min(axis=0) - 5.0
    x1, y1 = arr2.max(axis=0) + 5.0
    bbox = np.array([x0, y0, x1, y1], dtype=np.float64)
    return Detection(frame=frame, bbox=bbox, track_id=1, keypoints=kp)


# --------------------------------------------------------------------------- (a) lift


def test_relative_lift_tier_shape_and_never_metric() -> None:
    track = _moving_track()
    est = relative_lift(track)

    assert est.tier is Tier.RELATIVE
    assert est.positions.tier is Tier.RELATIVE
    assert est.velocity.tier is Tier.RELATIVE
    # Honesty: the relative floor must NEVER fabricate metric. Compare via .rank so the
    # check is a genuine runtime assertion (and not statically folded away).
    metric_rank = Tier.METRIC.rank
    assert est.tier.rank < metric_rank
    assert est.positions.tier.rank < metric_rank
    assert est.velocity.tier.rank < metric_rank

    pos = np.asarray(est.positions.value)
    vel = np.asarray(est.velocity.value)
    assert pos.shape == (len(track), 3)
    assert vel.shape == (len(track), 3)

    # Scale-free => dimensionless units reported honestly.
    assert est.positions.unit is None
    assert est.velocity.unit is None
    assert 0.0 <= est.positions.confidence <= 1.0


def test_relative_lift_depth_is_inverse_size_ordering() -> None:
    """A shrinking apparent size must yield a monotonically decreasing relative depth."""
    track = _moving_track()
    est = relative_lift(track)
    z = np.asarray(est.positions.value)[:, 2]
    # Apparent size shrinks every frame -> object recedes -> z_rel strictly decreases.
    assert np.all(np.diff(z) < 0.0)
    # Median-normalized proxy is centred near 1.
    assert math.isclose(float(np.median(z)), 1.0, rel_tol=1e-6)


def test_relative_lift_velocity_tracks_constant_motion() -> None:
    """Constant pixel velocity -> roughly constant relative in-plane velocity."""
    track = _moving_track()
    est = relative_lift(track)
    vx = np.asarray(est.velocity.value)[:, 0]
    # Interior samples should be near-equal (constant translation), and non-zero.
    interior = vx[1:-1]
    assert np.ptp(interior) < 1e-6
    assert abs(float(interior[0])) > 0.0


def test_relative_lift_empty_track_is_honest() -> None:
    est = relative_lift(TrackSequence(detections=[], fps=30.0))
    assert est.tier is Tier.RELATIVE
    assert np.asarray(est.positions.value).shape == (0, 3)
    assert np.asarray(est.velocity.value).shape == (0, 3)
    assert est.positions.confidence == 0.0
    assert est.goodness_of_fit == 0.0


def test_relative_lift_gap_lowers_density_confidence() -> None:
    """A track with a frame gap should be no more confident than a dense one."""
    dense = _moving_track(n=8)
    sparse_dets = [
        Detection(frame=f, bbox=_box_at(100.0 + 20.0 * f, 200.0, 60.0), track_id=1)
        for f in (0, 1, 2, 9, 10, 11)  # big gap in the middle
    ]
    sparse = TrackSequence(detections=sparse_dets, fps=30.0, image_size=(1280, 720))
    c_dense = relative_lift(dense).positions.confidence
    c_sparse = relative_lift(sparse).positions.confidence
    assert c_sparse <= c_dense


# ----------------------------------------------------------------- (b) angle geometry


def test_right_angle_recovers_half_pi() -> None:
    """An L-shaped triple (b->a horizontal, b->c vertical) has a pi/2 interior angle."""
    skeleton = SkeletonGraph(num_keypoints=3, edges=[(0, 1), (1, 2)])
    # a=(1,0), b=(0,0), c=(0,1): vectors b->a = +x, b->c = +y => 90 degrees.
    det = _keypoint_detection(0, [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)])
    track = TrackSequence(detections=[det], fps=30.0, skeleton=skeleton)
    res = compute_kinematics(track)

    q = res.angles[(0, 1, 2)]
    assert q.tier is Tier.PIXEL  # honesty: projected 2D angle, not true 3D
    assert q.unit == "rad"
    angle = float(np.asarray(q.value)[0])
    assert math.isclose(angle, math.pi / 2.0, abs_tol=1e-9)


def test_collinear_triple_recovers_pi() -> None:
    """A straight (collinear) triple has an interior angle of pi (limbs point opposite)."""
    skeleton = SkeletonGraph(num_keypoints=3, edges=[(0, 1), (1, 2)])
    # a=(-1,0), b=(0,0), c=(1,0): b->a = -x, b->c = +x => 180 degrees.
    det = _keypoint_detection(0, [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)])
    track = TrackSequence(detections=[det], fps=30.0, skeleton=skeleton)
    res = compute_kinematics(track)
    angle = float(np.asarray(res.angles[(0, 1, 2)].value)[0])
    assert math.isclose(angle, math.pi, abs_tol=1e-9)


# -------------------------------------------------------- (c) angular velocity rate


def test_constant_rate_rotation_recovers_angular_velocity() -> None:
    """A limb sweeping at a constant rate yields that rate as the angular velocity."""
    skeleton = SkeletonGraph(num_keypoints=3, edges=[(0, 1), (1, 2)])
    fps = 50.0
    omega = 1.5  # rad/s, the swept rate of the b->c limb about vertex b
    n = 20
    dets: list[Detection] = []
    for f in range(n):
        t = f / fps
        theta = omega * t  # angle of limb b->c grows linearly
        # a fixed along +x; c rotating; angle a-b-c = theta.
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        c = (math.cos(theta), math.sin(theta))
        dets.append(_keypoint_detection(f, [a, b, c]))
    track = TrackSequence(detections=dets, fps=fps, skeleton=skeleton)
    res = compute_kinematics(track)

    angvel = np.asarray(res.angular_velocities[(0, 1, 2)].value)
    assert res.angular_velocities[(0, 1, 2)].unit == "rad/s"
    # Interior samples (central differences) should equal omega within tolerance.
    interior = angvel[2:-2]
    assert np.all(np.isfinite(interior))
    assert np.allclose(interior, omega, atol=1e-3)


# ------------------------------------------------------- (d) no-skeleton / robustness


def test_no_skeleton_returns_empty_result() -> None:
    track = _moving_track()  # no skeleton attached, none passed
    res = compute_kinematics(track)
    assert isinstance(res, KinematicsResult)
    assert res.angles == {}
    assert res.angular_velocities == {}


def test_low_confidence_keypoints_are_gated_to_nan() -> None:
    """Frames with a sub-floor keypoint confidence are gated out (NaN) and lower trust."""
    skeleton = SkeletonGraph(num_keypoints=3, edges=[(0, 1), (1, 2)])
    pts = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]
    dets = [
        _keypoint_detection(0, pts, confs=[0.9, 0.9, 0.9]),  # good
        _keypoint_detection(1, pts, confs=[0.1, 0.9, 0.9]),  # one bad keypoint
        _keypoint_detection(2, pts, confs=[0.9, 0.9, 0.9]),  # good
    ]
    track = TrackSequence(detections=dets, fps=30.0, skeleton=skeleton)
    res = compute_kinematics(track, min_keypoint_confidence=0.5)

    angles = np.asarray(res.angles[(0, 1, 2)].value)
    assert np.isfinite(angles[0]) and np.isfinite(angles[2])
    assert math.isnan(angles[1])  # gated frame
    # Two of three frames survived -> confidence strictly below a fully-valid track.
    assert 0.0 < res.angles[(0, 1, 2)].confidence < 1.0


def test_skeleton_argument_overrides_track_skeleton() -> None:
    """An explicitly passed skeleton is used even when the track carries one."""
    attached = SkeletonGraph(num_keypoints=3, edges=[])  # no edges -> no triples
    det = _keypoint_detection(0, [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)])
    track = TrackSequence(detections=[det], fps=30.0, skeleton=attached)

    # With the attached (edgeless) skeleton, no angles.
    assert compute_kinematics(track).angles == {}

    # Passing a connected skeleton explicitly overrides and yields the angle.
    override = SkeletonGraph(num_keypoints=3, edges=[(0, 1), (1, 2)])
    res = compute_kinematics(track, skeleton=override)
    assert (0, 1, 2) in res.angles


def test_missing_keypoints_yield_empty_when_absent() -> None:
    """A track whose detections carry no keypoints produces all-NaN angle series."""
    skeleton = SkeletonGraph(num_keypoints=3, edges=[(0, 1), (1, 2)])
    dets = [
        Detection(frame=f, bbox=_box_at(50.0, 50.0, 20.0), track_id=1) for f in range(4)
    ]
    track = TrackSequence(detections=dets, fps=30.0, skeleton=skeleton)
    res = compute_kinematics(track)
    angles = np.asarray(res.angles[(0, 1, 2)].value)
    assert np.all(np.isnan(angles))
    assert res.angles[(0, 1, 2)].confidence == 0.0
