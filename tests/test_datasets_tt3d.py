"""Tests for the TT3D loader (validation/datasets_tt3d.py).

The real TT3D dataset is licensed and not bundled, so the loader normally only runs when a
human drops the data in place — meaning a format/parse regression would pass CI silently.
These tests synthesise a fixture in the *exact* TT3D on-disk format (a ``<view>.yaml`` plus
``<view>/NNN.csv`` rows with columns ``Timestamp,X,Y,Z,u,v,l,theta``) from a known synthetic
arc, so the parser, calibration handling, gap dropping, and 3D/2D alignment are all exercised
against a known-good source. The camera is built *from* an authoritative Rodrigues vector,
so the yaml's ``rvec`` is the single source of truth for the optical axis the loader derives.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import (  # noqa: E402
    CameraSpec,
    GroundTruth,
    generate_track,
    look_at_camera,
)
from validation.datasets_tt3d import (  # noqa: E402
    VIEWS,
    DatasetUnavailable,
    load_tt3d,
)
from validation.run_3d_validation import validate_trajectory  # noqa: E402

from trackphysics.core.schema import TrackSequence  # noqa: E402

_CSV_HEADER = ["Timestamp", "X", "Y", "Z", "u", "v", "l", "theta"]


def _side_camera() -> CameraSpec:
    """A cooperative side-on camera, re-expressed through its Rodrigues vector.

    Round-tripping the look-at rotation through ``rvec`` guarantees the fixture yaml's
    ``rvec`` reproduces this exact rotation (so the loader's derived optical axis must match
    ``cam.R[2]``).
    """
    base = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7), image_size=(1280, 720))
    rvec = Rotation.from_matrix(base.R).as_rotvec()
    r = Rotation.from_rotvec(rvec).as_matrix()
    return CameraSpec(K=base.K, R=r, t=base.t, image_size=base.image_size)


def _centroids(track: TrackSequence) -> np.ndarray:
    boxes = np.array([d.bbox for d in track.detections], dtype=np.float64)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])


def _write_yaml(path: Path, cam: CameraSpec) -> None:
    rvec = Rotation.from_matrix(cam.R).as_rotvec()
    w, h = cam.image_size
    path.write_text(
        f"rvec: [{rvec[0]}, {rvec[1]}, {rvec[2]}]\n"
        f"tvec: [{cam.t[0]}, {cam.t[1]}, {cam.t[2]}]\n"
        f"f: {cam.focal_px}\n"
        f"w: {float(w)}\n"
        f"h: {float(h)}\n"
    )


def _write_csv(
    path: Path,
    track: TrackSequence,
    gt: GroundTruth,
    *,
    gap_at: int | None = None,
) -> None:
    """Write one TT3D-format trajectory CSV.

    If ``gap_at`` is given, an extra row with non-finite ``u,v`` (ball out of frame) is
    inserted before retained row ``gap_at``, carrying a sentinel 3D position the loader must
    drop along with the 2D gap (so 3D/2D alignment is preserved).
    """
    centroids = _centroids(track)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for i in range(len(track.detections)):
            if gap_at is not None and i == gap_at:
                writer.writerow([gt.times[i], 999.0, 999.0, 999.0, "nan", "nan", 0.0, 0.0])
            x, y, z = gt.positions[i]
            u, v = centroids[i]
            writer.writerow([gt.times[i], x, y, z, u, v, 0.0, 0.0])


def _build_fixture(
    root: Path,
    *,
    view: str = "side",
    n_traj: int = 2,
    noisy: bool = True,
    gap_at: int | None = None,
) -> tuple[CameraSpec, list[GroundTruth]]:
    cam = _side_camera()
    _write_yaml(root / f"{view}.yaml", cam)
    view_dir = root / (view if noisy else f"{view}_no_noise")
    view_dir.mkdir(parents=True, exist_ok=True)
    gts: list[GroundTruth] = []
    for k in range(n_traj):
        track, gt = generate_track(
            cam,
            fps=120.0,
            launch_position=(3.0, 0.0, 0.0),
            launch_velocity=(6.0, 0.0, 7.0 + 0.5 * k),
            drag_coeff=0.2,
            duration=1.2,
        )
        _write_csv(view_dir / f"{k:03d}.csv", track, gt, gap_at=gap_at if k == 0 else None)
        gts.append(gt)
    return cam, gts


def test_load_tt3d_reconstructs_tracks_and_truth(tmp_path: Path) -> None:
    cam, gts = _build_fixture(tmp_path, n_traj=2)
    trajs = load_tt3d(tmp_path, view="side", noisy=True)

    assert len(trajs) == 2
    assert [t.name for t in trajs] == ["side/000", "side/001"]
    for traj, gt in zip(trajs, gts, strict=True):
        # 3D truth is aligned 1:1 with the reconstructed 2D track.
        assert traj.gt_positions_m.shape == (len(traj.track.detections), 3)
        np.testing.assert_allclose(traj.gt_positions_m, gt.positions, rtol=0, atol=1e-9)
        # The optical axis derived from the yaml rvec matches the camera's +Z-in-world.
        np.testing.assert_allclose(traj.optical_axis, cam.R[2], rtol=0, atol=1e-9)
        assert traj.track.image_size == (1280, 720)
        # fps is recovered from the timestamp spacing.
        assert traj.track.fps == pytest.approx(120.0, rel=1e-6)


def test_loaded_trajectory_recovers_metric_with_covering_ci(tmp_path: Path) -> None:
    # A cooperative side-on geometry: the engine should earn METRIC and its CI should cover
    # the in-plane truth — the full loader -> engine -> comparison chain on TT3D-shaped data.
    _build_fixture(tmp_path, n_traj=1)
    (traj,) = load_tt3d(tmp_path, view="side", noisy=True)
    res = validate_trajectory(traj)
    assert res.tier == "metric"
    assert res.inplane_error is not None and res.inplane_error < 1.5
    assert res.ci_covers_inplane is True


def test_gap_rows_dropped_and_alignment_preserved(tmp_path: Path) -> None:
    cam, gts = _build_fixture(tmp_path, n_traj=1, gap_at=10)
    (traj,) = load_tt3d(tmp_path, view="side", noisy=True)
    # The non-finite-uv row is dropped, so the track has exactly the retained detections...
    assert len(traj.track.detections) == gts[0].positions.shape[0]
    # ...and the sentinel 3D position from the gap row never enters the aligned truth.
    assert not np.any(np.all(traj.gt_positions_m == 999.0, axis=1))
    np.testing.assert_allclose(traj.gt_positions_m, gts[0].positions, rtol=0, atol=1e-9)


def test_noisy_false_reads_no_noise_directory(tmp_path: Path) -> None:
    _build_fixture(tmp_path, view="side", n_traj=1, noisy=False)
    trajs = load_tt3d(tmp_path, view="side", noisy=False)
    assert len(trajs) == 1
    assert trajs[0].name == "side/000"


def test_missing_root_raises_unavailable(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailable):
        load_tt3d(tmp_path / "does_not_exist", view="side")


def test_missing_view_directory_raises_unavailable(tmp_path: Path) -> None:
    # Calibration yaml present but the view's CSV directory absent.
    cam = _side_camera()
    _write_yaml(tmp_path / "side.yaml", cam)
    with pytest.raises(DatasetUnavailable):
        load_tt3d(tmp_path, view="side", noisy=True)


def test_bad_view_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="view must be one of"):
        load_tt3d(tmp_path, view="front")


def test_views_constant_is_the_documented_set() -> None:
    assert VIEWS == ("side", "oblique", "back")
