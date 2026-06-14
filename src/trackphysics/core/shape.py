"""Scale-invariant 2D trajectory-shape descriptors (domain-agnostic geometry).

A point sequence in the image plane has a *shape* independent of its pixel scale: how
elongated it is, which way it leans, how straight it is, and the ratio of its horizontal to
vertical span. These are pure geometry — no object semantics whatsoever (§6) — yet they
encode the viewpoint a monocular track was seen from, which is why they carry the point-only
signal the depth-domination guard (BRIEF.md §10) and a per-deployment calibrator both use.

All four descriptors are scale-invariant (ratios / orientations / normalized eigenvalues),
so they are valid at any tier and for any object.
"""

from __future__ import annotations

import numpy as np

from .schema import FloatArray

__all__ = ["SHAPE_FEATURE_NAMES", "inplane_shape_features"]

SHAPE_FEATURE_NAMES = ("aspect", "straightness", "pca_angle", "pca_ecc")


def inplane_shape_features(uv: FloatArray) -> dict[str, float]:
    """Four scale-invariant shape descriptors of a ``(T, 2)`` pixel point sequence.

    * ``aspect``       — horizontal pixel extent / vertical pixel extent (``ptp(u)/ptp(v)``).
      A low value means the sequence spans little horizontally relative to vertically.
    * ``straightness`` — chord / arc-length, in ``(0, 1]``: 1 is a straight line, lower is
      more curved.
    * ``pca_angle``    — orientation of the principal axis of the point cloud, folded to
      ``[0, pi)`` (undirected).
    * ``pca_ecc``      — elongation ``sqrt(1 - lambda_min/lambda_max)``: 0 round, →1 a line.

    Sequences shorter than 3 points return all-``NaN`` (no shape to speak of).
    """
    pts = np.asarray(uv, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        return dict.fromkeys(SHAPE_FEATURE_NAMES, float("nan"))
    u, v = pts[:, 0], pts[:, 1]
    aspect = float(np.ptp(u)) / (float(np.ptp(v)) + 1e-9)
    chord = float(np.hypot(u[-1] - u[0], v[-1] - v[0]))
    arc = float(np.sum(np.hypot(np.diff(u), np.diff(v))))
    straightness = chord / (arc + 1e-9)
    centered = pts - pts.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(centered.T @ centered / len(centered))  # ascending
    principal = eigvec[:, 1]
    pca_angle = float(np.arctan2(principal[1], principal[0]) % np.pi)  # undirected -> [0, pi)
    l_major, l_minor = float(eigval[1]), float(eigval[0])
    pca_ecc = float(np.sqrt(max(0.0, 1.0 - l_minor / (l_major + 1e-12))))
    return {
        "aspect": aspect,
        "straightness": straightness,
        "pca_angle": pca_angle,
        "pca_ecc": pca_ecc,
    }
