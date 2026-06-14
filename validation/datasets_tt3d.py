"""TT3D loader — INDEPENDENT 3D ground truth (table tennis, multi-camera, CVPRW'25).

TT3D is the primary first real-data validation source: a single-camera view is the input,
while the multi-camera-triangulated 3D trajectory is held out as ground truth (genuinely
independent — NOT monocular-reconstructed, unlike LATTE-MV; see validation/README.md).

This loader does NOT bundle or download data (license unconfirmed — likely non-commercial;
fine for internal validation, verify before publishing numbers). Point it at a locally
prepared directory. Expected per-trajectory layout (convert the released files to this):

    <root>/<clip>/track2d.csv      # one view: columns frame,track_id,x1,y1,x2,y2  (px)
    <root>/<clip>/gt3d.csv         # columns frame,X,Y,Z  (world metres), aligned by frame
    <root>/<clip>/optical_axis.txt # 3 floats: the chosen view's camera +Z axis in world

Once present, each clip yields a Trajectory3D for run_3d_validation.run(...).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from validation.run_3d_validation import Trajectory3D

from trackphysics import from_generic
from trackphysics.core.adapters.generic import from_csv

TT3D_URL = "https://github.com/(TT3D-repo)"  # fill with the exact repo when released/located
_BBOX_HALF_PX = 6.0


class DatasetUnavailable(RuntimeError):
    """Raised when TT3D data is not present locally."""


def trajectory_from_arrays(
    *,
    frames: list[int],
    centroids_px: list[tuple[float, float]],
    gt_positions_m: np.ndarray,
    fps: float,
    optical_axis: tuple[float, float, float] | np.ndarray,
    name: str = "",
) -> Trajectory3D:
    """Build a Trajectory3D from raw arrays (the reusable core for any real loader/test).

    ``gt_positions_m`` must be aligned 1:1 with ``frames`` (row i ↔ frames[i]).
    """
    h = _BBOX_HALF_PX
    boxes = [[cx - h, cy - h, cx + h, cy + h] for cx, cy in centroids_px]
    seq = from_generic(
        frames=frames,
        boxes=np.asarray(boxes, dtype=np.float64),
        track_ids=[1] * len(frames),
        fps=fps,
    )[0]
    return Trajectory3D(
        track=seq,
        gt_positions_m=np.asarray(gt_positions_m, dtype=np.float64),
        optical_axis=np.asarray(optical_axis, dtype=np.float64),
        name=name,
    )


def _load_clip(clip_dir: Path, fps: float) -> Trajectory3D:
    track2d = clip_dir / "track2d.csv"
    gt3d = clip_dir / "gt3d.csv"
    axis_file = clip_dir / "optical_axis.txt"
    for f in (track2d, gt3d, axis_file):
        if not f.exists():
            raise DatasetUnavailable(f"missing {f.name} in {clip_dir}")

    seq = from_csv(track2d, fps=fps)[0]
    frame_to_row: dict[int, list[float]] = {}
    with gt3d.open(newline="") as fh:
        for row in csv.DictReader(fh):
            frame_to_row[int(float(row["frame"]))] = [float(row[c]) for c in ("X", "Y", "Z")]
    positions = np.array([frame_to_row[int(f)] for f in seq.frames], dtype=np.float64)
    optical_axis = np.array(axis_file.read_text().split(), dtype=np.float64)
    return Trajectory3D(track=seq, gt_positions_m=positions, optical_axis=optical_axis,
                        name=clip_dir.name)


def load_tt3d(root: str | Path, *, fps: float = 200.0) -> list[Trajectory3D]:
    """Load TT3D clips as Trajectory3D objects. Raises DatasetUnavailable if absent."""
    root_dir = Path(root)
    if not root_dir.exists():
        raise DatasetUnavailable(
            f"TT3D not found at {root_dir}. This loader does not download data ({TT3D_URL}); "
            "prepare clips as <clip>/{track2d.csv,gt3d.csv,optical_axis.txt} and pass the root."
        )
    clips = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    if not clips:
        raise DatasetUnavailable(f"no clip subdirectories under {root_dir}")
    return [_load_clip(d, fps) for d in clips]


__all__ = ["DatasetUnavailable", "TT3D_URL", "load_tt3d", "trajectory_from_arrays"]
