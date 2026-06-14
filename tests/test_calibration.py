"""Per-deployment refittable calibrator (trackphysics.calibration.DeploymentCalibrator).

A calibration LAYER on top of the core: fit on one deployment's labelled metric emissions, it
de-biases the launch speed and emits an input-conditioned CI from runtime-observable features.
These tests pin: de-bias removes a systematic bias, the conditioned CI hits its target coverage
in-distribution, JSON round-trips exactly, the feature vector is readable from a real METRIC
estimate, and the artifact refuses to fit on too little data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import trackphysics as tp
from trackphysics.core.schema import Detection, TrackSequence


def _synthetic_rows(
    n: int = 160, seed: int = 0
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Features + recovered + truth with an a_px-LED systematic bias (mirrors the real finding)."""
    rng = np.random.default_rng(seed)
    a_px = rng.uniform(1000.0, 4000.0, n)
    aspect = rng.uniform(0.3, 8.0, n)
    truth = rng.uniform(3.0, 9.0, n)
    bias = 0.0008 * (a_px - 2500.0)            # systematic, feature-dependent
    recovered = truth + bias + rng.normal(0.0, 0.15, n)
    rows = [
        dict(residual_fraction=0.01, inlier_fraction=1.0, completeness=1.0, a_px=float(a_px[i]),
             gof=0.95, aspect=float(aspect[i]), straightness=0.99, pca_angle=0.3, pca_ecc=0.99)
        for i in range(n)
    ]
    return rows, recovered, truth


def test_debias_removes_systematic_bias_and_ci_is_calibrated() -> None:
    rows, recovered, truth = _synthetic_rows()
    cal = tp.DeploymentCalibrator.fit(rows, recovered, truth, provenance="synthetic")
    raw = float(np.mean(np.abs(recovered - truth)))
    deb, covered = [], 0
    for i, row in enumerate(rows):
        speed, (lo, hi) = cal.apply(row, float(recovered[i]))
        deb.append(abs(speed - truth[i]))
        covered += int(lo <= truth[i] <= hi)
    assert float(np.mean(deb)) < 0.4 * raw            # de-bias clearly helps
    assert abs(covered / len(rows) - 0.95) < 0.05     # conditioned CI hits ~target in-distribution


def test_json_round_trip_is_exact() -> None:
    rows, recovered, truth = _synthetic_rows()
    cal = tp.DeploymentCalibrator.fit(rows, recovered, truth, provenance="rig-A")
    restored = tp.DeploymentCalibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert restored.provenance == "rig-A"
    a = cal.apply(rows[0], float(recovered[0]))
    b = restored.apply(rows[0], float(recovered[0]))
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])


def test_fit_refuses_too_few_rows() -> None:
    rows, recovered, truth = _synthetic_rows(n=5)
    with pytest.raises(ValueError, match="at least"):
        tp.DeploymentCalibrator.fit(rows, recovered, truth, provenance="tiny")


# --------------------------------------------------------------------------------------
# integration with a real METRIC estimate
# --------------------------------------------------------------------------------------


def _metric_arc(aspect: float = 4.0) -> TrackSequence:
    t = np.arange(20) / 120.0
    v = 300.0 + 120.0 * t + 0.5 * 3000.0 * t * t
    u = 200.0 + (aspect * float(np.ptp(v))) * (t / t[-1])
    h = 6.0
    dets = [
        Detection(frame=i, bbox=np.array([u[i] - h, v[i] - h, u[i] + h, v[i] + h]), track_id=1)
        for i in range(20)
    ]
    return TrackSequence(detections=dets, fps=120.0, image_size=(1280, 720))


def test_features_from_estimate_reads_metric_meta() -> None:
    est = tp.analyze(_metric_arc(), preset="sphere", grounding=tp.GroundingContext()).trajectory
    assert est.tier is tp.Tier.METRIC
    feats = tp.features_from_estimate(est)
    assert feats is not None
    assert set(feats) == set(tp.CALIBRATOR_FEATURES)
    assert feats["aspect"] == pytest.approx(4.0, abs=0.05)
    # a RELATIVE estimate yields None (no metric features to recalibrate)
    rel = tp.relative_lift(_metric_arc())
    assert tp.features_from_estimate(rel) is None


def test_apply_to_recalibrates_a_real_estimate() -> None:
    rows, recovered, truth = _synthetic_rows()
    cal = tp.DeploymentCalibrator.fit(rows, recovered, truth, provenance="synthetic")
    est = tp.analyze(_metric_arc(), preset="sphere", grounding=tp.GroundingContext()).trajectory
    speed, (lo, hi) = cal.apply_to(est)
    assert np.isfinite(speed) and lo < speed < hi
