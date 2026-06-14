"""TT3D loader — INDEPENDENT 3D ground truth (table tennis, multi-camera).

TT3D (cogsys-tuebingen/tt3d, "3D Reconstruction Benchmark"): the 3D ball trajectory was
recorded with a MULTI-CAMERA setup (genuinely independent — not monocular-reconstructed,
unlike LATTE-MV; see validation/README.md). The 2D observations are synthetic projections
of that real 3D through three known single-camera viewpoints — ``side``, ``oblique``,
``back`` (and ``*_no_noise`` clean variants) — which is exactly the monocular input we test.

Repo:   https://github.com/cogsys-tuebingen/tt3d  (data under data/evaluation/)
Videos: https://cloud.cs.uni-tuebingen.de/index.php/s/SCKq85JZEmKoC6J
LICENSE: none declared in the repo (=> all rights reserved by default). Treat as
internal-use only; clear permission/terms before publishing any numbers built on it.

This loader does NOT bundle or download data. Point ``load_tt3d`` at a local
``data/evaluation`` directory. Each ``<view>/NNN.csv`` has columns
``Timestamp,X,Y,Z,u,v,l,theta`` (time s; 3D metres, table-centre origin, Z up; 2D pixels;
blur l,theta ignored). Each ``<view>.yaml`` gives ``rvec`` (Rodrigues), ``tvec``, ``f``,
``w``, ``h``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from validation.run_3d_validation import Trajectory3D

from trackphysics import from_generic

TT3D_REPO = "https://github.com/cogsys-tuebingen/tt3d"
VIEWS = ("side", "oblique", "back")
_BBOX_HALF_PX = 6.0


class DatasetUnavailable(RuntimeError):
    """Raised when TT3D data is not present locally."""


def _parse_calib(path: Path) -> dict[str, object]:
    """Parse the simple ``<view>.yaml`` (no PyYAML dependency)."""
    out: dict[str, object] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = (s.strip() for s in line.split(":", 1))
        if raw.startswith("["):
            out[key] = [float(x) for x in raw.strip("[]").split(",")]
        else:
            out[key] = float(raw)
    return out


def _optical_axis(rvec: list[float]) -> np.ndarray:
    """Camera +Z axis (viewing direction) in world coords, from the Rodrigues vector.

    OpenCV convention ``x_cam = R @ X_world + t`` (R maps world->camera), so the camera +Z
    axis expressed in world is ``R^T @ [0,0,1]`` = the third row of ``R``.
    """
    rot = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64)).as_matrix()
    return np.asarray(rot[2, :], dtype=np.float64)


def load_tt3d(
    eval_root: str | Path, *, view: str = "side", noisy: bool = True
) -> list[Trajectory3D]:
    """Load one TT3D viewpoint's trajectories as :class:`Trajectory3D` objects.

    Args:
        eval_root: path to the dataset's ``data/evaluation`` directory.
        view: ``"side"`` | ``"oblique"`` | ``"back"`` (``side`` is the best monocular
            geometry — motion mostly perpendicular to the optical axis; ``back`` looks down
            the table so most motion is along the optical axis = depth-blind).
        noisy: ``True`` uses the noisy 2D observations, ``False`` the ``*_no_noise`` variant.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    root = Path(eval_root)
    if not root.exists():
        raise DatasetUnavailable(
            f"TT3D evaluation data not found at {root} ({TT3D_REPO}; data/evaluation/). "
            "This loader does not download data; clone the repo and pass data/evaluation."
        )
    calib = _parse_calib(root / f"{view}.yaml")
    rvec, width, height = calib["rvec"], calib["w"], calib["h"]
    assert isinstance(rvec, list) and isinstance(width, float) and isinstance(height, float)
    axis = _optical_axis(rvec)
    image_w, image_h = int(width), int(height)
    view_dir = root / (view if noisy else f"{view}_no_noise")
    if not view_dir.is_dir():
        raise DatasetUnavailable(f"view directory {view_dir} not found")

    trajectories: list[Trajectory3D] = []
    for csv_path in sorted(view_dir.glob("*.csv")):
        times: list[float] = []
        uv: list[tuple[float, float]] = []
        xyz: list[list[float]] = []
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                u, v = float(row["u"]), float(row["v"])
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue  # ball out of frame -> gap
                times.append(float(row["Timestamp"]))
                uv.append((u, v))
                xyz.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
        if len(times) < 3:
            continue
        t = np.asarray(times, dtype=np.float64)
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.04
        fps = 1.0 / dt if dt > 0 else 25.0
        frames = [int(round((ti - t[0]) / dt)) for ti in t]
        h = _BBOX_HALF_PX
        boxes = [[cu - h, cv - h, cu + h, cv + h] for cu, cv in uv]
        track = from_generic(
            frames=frames,
            boxes=np.asarray(boxes, dtype=np.float64),
            track_ids=[1] * len(frames),
            fps=fps,
            image_size=(image_w, image_h),
        )[0]
        trajectories.append(
            Trajectory3D(
                track=track,
                gt_positions_m=np.asarray(xyz, dtype=np.float64),
                optical_axis=axis,
                name=f"{view}/{csv_path.stem}",
            )
        )
    return trajectories


__all__ = ["VIEWS", "DatasetUnavailable", "TT3D_REPO", "load_tt3d"]
