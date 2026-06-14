"""Tests for the benchmark substrate: generator, perturbation, metrics (BRIEF.md §14).

Covers both the success path and the honesty/edge path required by the brief:

* (a) with drag ~ 0 the generated vertical world acceleration ~ gravity;
* (b) ``project_points`` reproduces a hand-computed pinhole projection;
* (c) a gap burst removes the expected count; jitter raises positional variance and is
      seed-deterministic;
* (d) ``position_error`` is 0 for identical arrays and grows with offset;
* (e) reliability of a perfectly-calibrated synthetic set has small ECE.

Plus extra honesty checks: degradation curves are monotone-ish, the gate
precision/recall handles a perfect and a degenerate classifier, and frame alignment
restricts scoring to the common frames.
"""

from __future__ import annotations

import numpy as np
from bench.metrics import (
    align_by_frame,
    degradation_curve,
    gate_precision_recall,
    position_error,
    reliability,
    speed_error,
    velocity_error,
)
from bench.perturb import (
    CorruptionConfig,
    apply_correlated_jitter,
    apply_gap_bursts,
    apply_id_switches,
    corrupt,
)
from bench.synth.generator import (
    CameraSpec,
    arc_scenario,
    generate_track,
    look_at_camera,
    project_points,
    simulate_trajectory,
)

# --------------------------------------------------------------------------- #
# (a) physics: near-zero drag -> vertical world acceleration ~ gravity
# --------------------------------------------------------------------------- #


def test_vertical_acceleration_matches_gravity_without_drag() -> None:
    fps = 240.0
    gravity = 9.81
    # Near-zero drag: tiny coefficient so the quadratic-drag term is negligible.
    pos, vel, times = simulate_trajectory(
        launch_position=(0.0, 0.0, 0.0),
        launch_velocity=(5.0, 0.0, 8.0),
        drag_coeff=1e-9,
        gravity=gravity,
        fps=fps,
        duration=1.0,
    )
    dt = 1.0 / fps
    # Central-difference vertical acceleration from the velocity series.
    az = np.gradient(vel[:, 2], dt)
    assert np.allclose(az, -gravity, atol=1e-3)
    # Horizontal velocity is essentially constant (no horizontal force).
    assert np.allclose(vel[:, 0], vel[0, 0], atol=1e-3)
    assert pos.shape == vel.shape == (times.size, 3)


def test_drag_decelerates_horizontal_motion() -> None:
    # Honesty path: with real drag, horizontal speed must DECAY, not stay constant.
    _, vel, _ = simulate_trajectory(
        launch_velocity=(20.0, 0.0, 5.0),
        drag_coeff=0.8,
        diameter_m=0.1,
        mass_kg=0.02,
        fps=120.0,
        duration=1.0,
    )
    assert vel[-1, 0] < vel[0, 0]


def test_magnus_term_curves_trajectory() -> None:
    # Spin about +Z should deflect an x-moving object in y (Magnus curve).
    _, vel_no, _ = simulate_trajectory(
        launch_velocity=(15.0, 0.0, 0.0), drag_coeff=1e-9, magnus_coeff=0.0, duration=0.5
    )
    _, vel_yes, _ = simulate_trajectory(
        launch_velocity=(15.0, 0.0, 0.0),
        drag_coeff=1e-9,
        magnus_coeff=5.0,
        spin_axis=(0.0, 0.0, 1.0),
        duration=0.5,
    )
    assert abs(vel_no[-1, 1]) < 1e-6
    assert abs(vel_yes[-1, 1]) > 1.0


# --------------------------------------------------------------------------- #
# (b) projection: hand-computed pinhole projection of one point
# --------------------------------------------------------------------------- #


def test_project_points_matches_hand_computed_pinhole() -> None:
    # Identity rotation, camera at origin, point straight ahead at depth 5 (+Z forward).
    fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    cam = CameraSpec(K=k, R=np.eye(3), t=np.zeros(3), image_size=(1280, 720))

    point = np.array([[2.0, -1.0, 5.0]])  # x=2, y=-1, depth=5
    px = project_points(point, cam)
    # Expected: u = fx * x/z + cx, v = fy * y/z + cy
    expected_u = fx * 2.0 / 5.0 + cx
    expected_v = fy * (-1.0) / 5.0 + cy
    assert np.allclose(px[0], [expected_u, expected_v])


def test_project_points_behind_camera_is_nan() -> None:
    cam = CameraSpec(
        K=np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]),
        R=np.eye(3),
        t=np.zeros(3),
        image_size=(1280, 720),
    )
    behind = np.array([[1.0, 1.0, -3.0]])  # negative depth
    px = project_points(behind, cam)
    assert np.all(np.isnan(px[0]))


def test_look_at_camera_centers_target() -> None:
    cam = look_at_camera(
        eye=(0.0, -5.0, 0.0), target=(0.0, 0.0, 0.0), focal_px=900.0, image_size=(640, 480)
    )
    px = project_points(np.array([[0.0, 0.0, 0.0]]), cam)
    # Target should project to the image centre.
    assert np.allclose(px[0], [320.0, 240.0], atol=1e-6)


def test_generate_track_produces_aligned_ground_truth() -> None:
    track, gt = arc_scenario(fps=120.0)
    assert len(track) == gt.positions.shape[0]
    assert gt.velocities.shape == gt.positions.shape
    assert gt.speed.shape[0] == len(track)
    assert gt.metric_scale > 0.0
    # bbox sizes are positive (depth-scaled radius).
    for det in track.detections:
        x0, y0, x1, y1 = det.bbox
        assert x1 > x0 and y1 > y0
    # frames retained match the detection frames.
    assert np.array_equal(gt.frames.astype(int), track.frames)


# --------------------------------------------------------------------------- #
# (c) perturbation: gap-burst count, jitter variance, seed determinism
# --------------------------------------------------------------------------- #


def _clean_track() -> object:
    cam = look_at_camera(eye=(0.0, -8.0, 1.5), target=(4.0, 0.0, 1.5), focal_px=900.0)
    track, _ = generate_track(
        cam,
        fps=120.0,
        launch_position=(0.0, 0.0, 1.0),
        launch_velocity=(7.0, 0.0, 6.0),
        drag_coeff=0.2,
        duration=1.4,
    )
    return track


def test_gap_burst_removes_expected_count() -> None:
    track = _clean_track()
    n0 = len(track)  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    out = apply_gap_bursts(track, burst_len=5, n_bursts=2, rng=rng)  # type: ignore[arg-type]
    # Two non-overlapping bursts of length 5 -> 10 frames removed (track is long enough).
    assert n0 - len(out) == 10


def test_gap_burst_noop_when_zero() -> None:
    track = _clean_track()
    rng = np.random.default_rng(1)
    out = apply_gap_bursts(track, burst_len=0, n_bursts=3, rng=rng)  # type: ignore[arg-type]
    assert len(out) == len(track)  # type: ignore[arg-type]


def test_jitter_increases_variance_and_is_seed_deterministic() -> None:
    track = _clean_track()
    centers0 = track.centers()  # type: ignore[attr-defined]

    out_a = apply_correlated_jitter(
        track, sigma_px=3.0, rho=0.6, rng=np.random.default_rng(42)  # type: ignore[arg-type]
    )
    out_b = apply_correlated_jitter(
        track, sigma_px=3.0, rho=0.6, rng=np.random.default_rng(42)  # type: ignore[arg-type]
    )
    # Same seed -> identical corruption.
    assert np.allclose(out_a.centers(), out_b.centers())

    # Jitter increases positional spread relative to the smooth ground-truth path.
    resid_clean = centers0 - centers0  # zero by construction
    resid_jit = out_a.centers() - centers0
    assert np.var(resid_jit) > np.var(resid_clean)

    # Different seed -> different realization.
    out_c = apply_correlated_jitter(
        track, sigma_px=3.0, rho=0.6, rng=np.random.default_rng(7)  # type: ignore[arg-type]
    )
    assert not np.allclose(out_a.centers(), out_c.centers())


def test_id_switch_changes_ids_and_is_bursty() -> None:
    track = _clean_track()
    rng = np.random.default_rng(3)
    out = apply_id_switches(track, rate=0.2, rng=rng)  # type: ignore[arg-type]
    ids = {d.track_id for d in out.detections}
    # A high switch rate over a long track yields more than one distinct id.
    assert len(ids) >= 2


def test_id_switch_zero_rate_preserves_ids() -> None:
    track = _clean_track()
    rng = np.random.default_rng(4)
    out = apply_id_switches(track, rate=0.0, rng=rng)  # type: ignore[arg-type]
    orig = {d.track_id for d in track.detections}  # type: ignore[attr-defined]
    assert {d.track_id for d in out.detections} == orig


def test_corrupt_composes_and_does_not_mutate_input() -> None:
    track = _clean_track()
    n0 = len(track)  # type: ignore[arg-type]
    centers0 = track.centers().copy()  # type: ignore[attr-defined]
    cfg = CorruptionConfig(
        jitter_sigma_px=2.0,
        jitter_rho=0.5,
        drop_rate=0.05,
        id_switch_rate=0.1,
        gap_burst_len=3,
        n_gap_bursts=1,
    )
    out = corrupt(track, cfg, np.random.default_rng(11))  # type: ignore[arg-type]
    # Input is untouched.
    assert len(track) == n0  # type: ignore[arg-type]
    assert np.allclose(track.centers(), centers0)  # type: ignore[attr-defined]
    # Output lost at least the gap-burst frames.
    assert len(out) <= n0


# --------------------------------------------------------------------------- #
# (d) metrics: position error is 0 for identical arrays and grows with offset
# --------------------------------------------------------------------------- #


def test_position_error_zero_and_grows() -> None:
    rng = np.random.default_rng(0)
    gt = rng.standard_normal((20, 3))
    assert position_error(gt, gt) == 0.0
    shifted_small = gt + np.array([0.1, 0.0, 0.0])
    shifted_large = gt + np.array([1.0, 0.0, 0.0])
    e_small = position_error(shifted_small, gt)
    e_large = position_error(shifted_large, gt)
    assert 0.0 < e_small < e_large
    assert np.isclose(e_small, 0.1)
    assert np.isclose(e_large, 1.0)


def test_velocity_and_speed_error() -> None:
    gt_v = np.tile(np.array([3.0, 0.0, 4.0]), (10, 1))  # speed 5
    assert velocity_error(gt_v, gt_v) == 0.0
    gt_speed = np.full(10, 5.0)
    pred_speed = np.full(10, 6.0)
    assert np.isclose(speed_error(pred_speed, gt_speed), 1.0)


def test_align_by_frame_uses_common_frames_only() -> None:
    pred_frames = np.array([0, 2, 4, 6])
    pred_vals = np.array([[0.0, 0.0, 0.0], [2.0, 0, 0], [9.0, 0, 0], [6.0, 0, 0]])
    gt_frames = np.array([0, 2, 6])
    gt_vals = np.array([[0.0, 0, 0], [2.0, 0, 0], [6.0, 0, 0]])
    p, g = align_by_frame(pred_frames, pred_vals, gt_frames, gt_vals)
    assert p.shape == g.shape == (3, 3)
    # Frame 4 (only in pred) excluded -> error is 0 on the common frames.
    assert position_error(p, g) == 0.0


# --------------------------------------------------------------------------- #
# degradation curve
# --------------------------------------------------------------------------- #


def test_degradation_curve_is_monotone_for_linear_run_fn() -> None:
    params, errors = degradation_curve(lambda lvl: 0.5 * lvl + 1.0, [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(params, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.all(np.diff(errors) > 0)


# --------------------------------------------------------------------------- #
# (e) reliability: perfectly-calibrated set has small ECE
# --------------------------------------------------------------------------- #


def test_reliability_perfectly_calibrated_has_small_ece() -> None:
    # Construct a perfectly-calibrated set: within each confidence bucket the empirical
    # accuracy equals the bucket's confidence.
    rng = np.random.default_rng(123)
    confidences = []
    correct = []
    for conf in np.linspace(0.05, 0.95, 10):
        n = 2000
        confidences.append(np.full(n, conf))
        correct.append((rng.random(n) < conf).astype(float))
    pred_conf = np.concatenate(confidences)
    corr = np.concatenate(correct)

    centers, acc, counts, ece = reliability(pred_conf, corr, n_bins=10)
    assert centers.shape == acc.shape == counts.shape == (10,)
    assert counts.sum() == pred_conf.size
    # Large samples + true calibration -> ECE should be tiny.
    assert ece < 0.02


def test_reliability_miscalibrated_has_large_ece() -> None:
    # Honesty path: overconfident predictor (states 0.99 but is right only ~50%).
    n = 5000
    pred_conf = np.full(n, 0.99)
    rng = np.random.default_rng(0)
    corr = (rng.random(n) < 0.5).astype(float)
    _, _, _, ece = reliability(pred_conf, corr, n_bins=10)
    assert ece > 0.4


def test_reliability_empty_input() -> None:
    centers, acc, counts, ece = reliability(np.array([]), np.array([]), n_bins=5)
    assert ece == 0.0
    assert counts.sum() == 0
    assert np.all(np.isnan(acc))


# --------------------------------------------------------------------------- #
# gate precision/recall
# --------------------------------------------------------------------------- #


def test_gate_precision_recall_perfect_classifier() -> None:
    truth = np.array([True, True, False, False, True])
    pred = truth.copy()
    res = gate_precision_recall(pred, truth)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["accuracy"] == 1.0


def test_gate_precision_recall_overeager_metric() -> None:
    # Honesty path: an engine that ALWAYS claims metric (never falls back) -> low precision,
    # full recall. This is exactly the failure mode the gate exists to catch (§10).
    truth = np.array([True, False, False, False])
    pred = np.array([True, True, True, True])
    res = gate_precision_recall(pred, truth)
    assert np.isclose(res["precision"], 0.25)
    assert res["recall"] == 1.0
    assert res["fp"] == 3.0


def test_gate_precision_recall_never_metric() -> None:
    # An engine that never emits metric: zero precision/recall, but no false positives.
    truth = np.array([True, True, False])
    pred = np.array([False, False, False])
    res = gate_precision_recall(pred, truth)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0
    assert res["fp"] == 0.0
