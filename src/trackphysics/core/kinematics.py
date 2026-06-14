"""Scale-invariant kinematics from a keypoint graph (BRIEF.md §12.4).

Given per-frame keypoints and a :class:`SkeletonGraph`, compute, for every connected
triple ``(a, b, c)`` (where ``(a, b)`` and ``(b, c)`` are edges), the angle at the middle
vertex ``b`` over time, plus its angular velocity. Joint angles are *scale-invariant*, so
they remain meaningful even when no metric scale exists — which is precisely why they are a
genuinely useful output at the PIXEL/RELATIVE tier (BRIEF.md §12.4).

HONESTY (a hard rule, BRIEF.md §10): these are the **2D image-plane (projected) angles**,
NOT the true 3D joint angles. Recovering true 3D angles needs per-keypoint depth, which is
a v0.2 capability. Accordingly every emitted :class:`Quantity` is tagged
:class:`Tier.PIXEL` and its ``source`` says ``"image_plane_angle"`` /
``"image_plane_angular_velocity"`` so a downstream consumer is never misled into treating a
projected angle as a metric 3D angle.

Robustness: keypoints may carry a confidence channel (``K == 3``). A frame's angle is only
emitted when all three of its keypoints clear ``min_keypoint_confidence`` and form a
non-degenerate (non-coincident) vertex; otherwise that frame's angle is ``NaN`` and is
skipped by the angular-velocity finite difference. The per-angle :class:`Quantity`
confidence is derived (never hardcoded) from the fraction of frames that survived this
gate, via :func:`trackphysics.core.provenance.combine_confidence`.
"""

from __future__ import annotations

import numpy as np

from .provenance import Quantity, Tier, combine_confidence
from .results import KinematicsResult
from .schema import FloatArray, SkeletonGraph, TrackSequence

__all__ = ["compute_kinematics"]

_EPS = 1e-9


def _angle_triples(skeleton: SkeletonGraph) -> list[tuple[int, int, int]]:
    """Enumerate connected triples ``(a, b, c)`` with edges ``(a, b)`` and ``(b, c)``.

    Edges are treated as undirected. For each middle vertex ``b`` we form one triple per
    unordered pair of its neighbours. Triples are returned with ``a < c`` and de-duplicated,
    so the angle at ``b`` is reported once per neighbour pair regardless of edge order.
    """
    neighbours: dict[int, set[int]] = {}
    for i, j in skeleton.edges:
        if i == j:
            continue
        neighbours.setdefault(i, set()).add(j)
        neighbours.setdefault(j, set()).add(i)

    triples: set[tuple[int, int, int]] = set()
    for b, adj in neighbours.items():
        ordered = sorted(adj)
        for idx, a in enumerate(ordered):
            for c in ordered[idx + 1 :]:
                triples.add((a, b, c))
    return sorted(triples)


def _keypoint_xy_conf(track: TrackSequence, num_keypoints: int) -> tuple[FloatArray, FloatArray]:
    """Stack keypoints into ``(T, K, 2)`` positions and ``(T, K)`` confidences.

    Missing keypoints (a detection without a keypoint array, or with too few rows) are
    encoded as ``NaN`` position and ``0.0`` confidence. A ``(K, 2)`` keypoint array (no
    confidence channel) is treated as fully confident (``1.0``).
    """
    n = len(track)
    xy = np.full((n, num_keypoints, 2), np.nan, dtype=np.float64)
    conf = np.zeros((n, num_keypoints), dtype=np.float64)
    for t, det in enumerate(track.detections):
        kp = det.keypoints
        if kp is None:
            continue
        rows = min(kp.shape[0], num_keypoints)
        xy[t, :rows, :] = kp[:rows, :2]
        if kp.shape[1] == 3:
            conf[t, :rows] = kp[:rows, 2]
        else:
            conf[t, :rows] = 1.0
    return xy, conf


def _vertex_angles(
    xy: FloatArray,
    conf: FloatArray,
    triple: tuple[int, int, int],
    min_keypoint_confidence: float,
) -> FloatArray:
    """Per-frame image-plane angle at the middle vertex, shape ``(T,)``, ``NaN`` if gated.

    The angle is the unsigned interior angle in ``[0, pi]`` between the vectors ``b->a`` and
    ``b->c``, computed with :func:`numpy.arctan2` of the cross/dot for numerical stability.
    A frame is gated to ``NaN`` if any of the three keypoints is below the confidence floor,
    is non-finite, or if either limb vector is degenerate (near-zero length).
    """
    a, b, c = triple
    pa = xy[:, a, :]
    pb = xy[:, b, :]
    pc = xy[:, c, :]

    v1 = pa - pb
    v2 = pc - pb
    len1 = np.hypot(v1[:, 0], v1[:, 1])
    len2 = np.hypot(v2[:, 0], v2[:, 1])

    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    angles = np.arctan2(np.abs(cross), dot)

    confident = (
        (conf[:, a] >= min_keypoint_confidence)
        & (conf[:, b] >= min_keypoint_confidence)
        & (conf[:, c] >= min_keypoint_confidence)
    )
    finite = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1) & np.isfinite(pc).all(axis=1)
    non_degenerate = (len1 > _EPS) & (len2 > _EPS)
    valid = confident & finite & non_degenerate

    out = np.where(valid, angles, np.nan)
    return out.astype(np.float64)


def _angular_velocity(angles: FloatArray, times: FloatArray) -> FloatArray:
    """Finite-difference time derivative of an angle series, shape ``(T,)``, in rad/s.

    Central differences over real seconds (one-sided at the ends). Frames whose angle is
    ``NaN`` (gated out) propagate ``NaN`` into the adjacent derivative samples that depend on
    them, so an unreliable frame never silently produces a confident velocity. Returns
    all-``NaN`` if fewer than two finite angle samples exist or times are not strictly
    increasing.

    Note: the angles are the UNSIGNED interior angle in ``[0, pi]`` (``arctan2(|cross|,
    dot)``), so no ``2*pi`` wrap is possible and ``np.unwrap`` would be a no-op — we do not
    apply it. The genuine artifact for an unsigned angle is *reflection* at the 0/pi folds
    (the rate sign flips), which unwrap cannot fix anyway; a signed angle would be needed for
    a continuous rate and is a v0.2 concern.
    """
    n = angles.shape[0]
    if n < 2 or not np.all(np.diff(times) > 0):
        return np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(angles)
    if finite.sum() < 2:
        return np.full(n, np.nan, dtype=np.float64)
    # NaN-gated frames stay NaN in the array; np.gradient's central difference does not use
    # the sample itself for interior points, so a finite derivative survives where both
    # neighbours are finite. The validity mask below re-asserts NaN where a neighbour is gated.
    grad = np.gradient(angles, times, axis=0)
    # A derivative sample is trustworthy only if the angle samples it consumed are finite.
    valid = finite.copy()
    valid[1:-1] = finite[:-2] & finite[2:]  # central difference neighbours
    if n >= 2:
        valid[0] = finite[0] & finite[1]
        valid[-1] = finite[-1] & finite[-2]
    return np.where(valid, grad, np.nan).astype(np.float64)


def compute_kinematics(
    track: TrackSequence,
    skeleton: SkeletonGraph | None = None,
    *,
    min_keypoint_confidence: float = 0.0,
) -> KinematicsResult:
    """Compute scale-invariant joint angles and angular velocities (BRIEF.md §12.4).

    For every connected triple ``(a, b, c)`` in the skeleton graph, computes the image-plane
    angle at the middle vertex ``b`` over time and its angular velocity. Both are wrapped as
    array-valued :class:`Quantity` objects at :class:`Tier.PIXEL` (these are *projected* 2D
    angles, not true 3D joint angles; see module docstring and BRIEF.md §10).

    Args:
        track: The input track; its per-frame keypoints supply the geometry.
        skeleton: Connectivity graph. If ``None``, falls back to ``track.skeleton``; if that
            is also ``None``, an empty :class:`KinematicsResult` is returned (no skeleton =>
            no angles to compute).
        min_keypoint_confidence: Per-keypoint confidence floor in ``[0, 1]``. A frame's angle
            is emitted only when all three of its keypoints meet this floor; otherwise that
            frame is ``NaN`` and is skipped downstream. Default ``0.0`` admits all keypoints
            (including a ``(K, 2)`` array that carries no confidence channel).

    Returns:
        A :class:`KinematicsResult` keyed by the ``(a, b, c)`` triple. Each ``angles`` value
        is a ``(T,)`` :class:`Quantity` with ``unit="rad"`` and each ``angular_velocities``
        value a ``(T,)`` :class:`Quantity` with ``unit="rad/s"``, both at
        :class:`Tier.PIXEL`. Gated frames are ``NaN`` in the series; per-angle confidence is
        derived from the fraction of frames that survived gating.
    """
    graph = skeleton if skeleton is not None else track.skeleton
    if graph is None:
        return KinematicsResult()

    triples = _angle_triples(graph)
    if not triples or len(track) == 0:
        return KinematicsResult()

    xy, conf = _keypoint_xy_conf(track, graph.num_keypoints)
    times = track.times()
    n = len(track)

    frames = track.frames
    frame_span: tuple[int, int] = (int(frames[0]), int(frames[-1]))

    angles_out: dict[tuple[int, int, int], Quantity] = {}
    angvel_out: dict[tuple[int, int, int], Quantity] = {}

    for triple in triples:
        angles = _vertex_angles(xy, conf, triple, min_keypoint_confidence)
        angvel = _angular_velocity(angles, times)

        valid_frac = float(np.isfinite(angles).sum()) / float(n) if n else 0.0
        angle_conf = combine_confidence(valid_frac)
        # Angular velocity additionally loses the unreliable boundary/neighbour samples.
        vel_valid_frac = float(np.isfinite(angvel).sum()) / float(n) if n else 0.0
        angvel_conf = combine_confidence(valid_frac, vel_valid_frac)

        angles_out[triple] = Quantity(
            value=angles,
            unit="rad",
            tier=Tier.PIXEL,
            confidence=angle_conf,
            source="image_plane_angle",
            frame=frame_span,
        )
        angvel_out[triple] = Quantity(
            value=angvel,
            unit="rad/s",
            tier=Tier.PIXEL,
            confidence=angvel_conf,
            source="image_plane_angular_velocity",
            frame=frame_span,
        )

    return KinematicsResult(angles=angles_out, angular_velocities=angvel_out)
