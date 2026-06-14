"""Scale-invariant 2D trajectory-shape descriptors (trackphysics.core.shape).

These are the point-only, domain-agnostic geometry features the depth guard and the
per-deployment calibrator both read. The property that matters is **scale-invariance** (a
descriptor must not change if the whole track is zoomed), which the guard's fixed-scale
fixtures do not exercise directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from trackphysics.core.shape import SHAPE_FEATURE_NAMES, inplane_shape_features


def _curve() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 25)
    return np.column_stack([300.0 * t, 50.0 * t + 80.0 * t * t])  # a leaning, slightly curved arc


def test_shape_features_are_scale_invariant() -> None:
    base = inplane_shape_features(_curve())
    for factor in (0.1, 10.0, 1000.0):
        scaled = inplane_shape_features(_curve() * factor)  # zoom about the origin
        for k in SHAPE_FEATURE_NAMES:
            assert scaled[k] == pytest.approx(base[k], abs=1e-9)


def test_horizontal_line_is_elongated_straight_and_wide() -> None:
    uv = np.column_stack([np.linspace(0.0, 100.0, 10), np.full(10, 5.0)])  # flat horizontal
    f = inplane_shape_features(uv)
    assert f["aspect"] > 1e6                              # ~no vertical extent
    assert f["straightness"] == pytest.approx(1.0, abs=1e-6)
    assert f["pca_ecc"] > 0.999                           # a line is maximally eccentric
    assert f["pca_angle"] < 1e-6                          # principal axis ~ horizontal


def test_too_short_sequence_is_nan() -> None:
    f = inplane_shape_features(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert all(np.isnan(f[k]) for k in SHAPE_FEATURE_NAMES)
