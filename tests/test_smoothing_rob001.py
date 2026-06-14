"""Regression test for ROB-001: the MAD-degeneracy guard must not re-admit outliers.

When the inlier residuals collapse to ~0 (a clean majority perfectly explained by the
fit) but a sparse minority of gross outliers remains, the robust smoother must still flag
those outliers — not blanket-reset all weights to 1 and report them as inliers.
"""

from __future__ import annotations

import numpy as np

from trackphysics.core.robust.smoothing import robust_smooth


def test_rejects_outliers_with_zero_inlier_noise() -> None:
    t = np.arange(40, dtype=np.float64)
    y = 2.0 + 0.5 * t  # an exact line: inlier residuals collapse to ~0
    y[10] += 50.0
    y[25] -= 40.0
    res = robust_smooth(y, t, degree=2)
    assert not bool(res.inlier_mask[10]), "gross outlier 10 wrongly kept as inlier"
    assert not bool(res.inlier_mask[25]), "gross outlier 25 wrongly kept as inlier"
    # The clean majority is retained.
    assert int(res.inlier_mask.sum()) >= 36


def test_all_clean_points_remain_inliers() -> None:
    t = np.arange(30, dtype=np.float64)
    y = 1.0 - 0.3 * t + 0.05 * t * t  # exact parabola, no outliers
    res = robust_smooth(y, t, degree=2)
    assert bool(res.inlier_mask.all()), "a perfectly clean fit must keep all points"
