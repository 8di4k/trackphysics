"""Real-data validation loaders (BRIEF.md §14.3) — the sim2real check.

These are the bridge from the synthetic envelope to real footage. They deliberately do
NOT bundle or auto-download data (the sets are large and licensed separately); each loader
expects a locally-prepared path and raises an actionable error otherwise. The loaders
target our own input schema (one :class:`TrackSequence` per object), so once data is
present they plug straight into ``analyze()`` and the metrics in :mod:`bench.metrics`.

Priority per the brief: **LATTE-MV** first (a near-real monocular validation source with
public ball CSVs + 3D reconstructions), then one automotive set (TuSimple velocity or
BrnoCompSpeed) for cross-domain generality.
"""

from __future__ import annotations

from pathlib import Path

from trackphysics.core.adapters.generic import from_csv
from trackphysics.core.schema import TrackSequence

# --- dataset references (annotated in BRIEF.md §22) --------------------------------------
LATTE_MV_URL = "https://github.com/sastry-group/LATTE-MV"
TUSIMPLE_VELOCITY_NOTE = "TuSimple velocity estimation benchmark (automotive)."
BRNO_COMPSPEED_NOTE = "BrnoCompSpeed (high-quality optics; pair with corruption sweeps)."


class DatasetUnavailable(RuntimeError):
    """Raised when a real dataset has not been downloaded/prepared locally."""


def _require_dir(path: str | Path, name: str, url_or_note: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise DatasetUnavailable(
            f"{name} not found at {p}. This loader does not download data. "
            f"Obtain it ({url_or_note}), prepare it locally, and pass its path. "
            f"See BRIEF.md §14.3 / §22."
        )
    return p


def load_latte_mv(root: str | Path, *, fps: float = 120.0) -> list[TrackSequence]:
    """Load LATTE-MV object tracks as :class:`TrackSequence` objects.

    Expects a locally-prepared directory of per-clip CSVs with columns
    ``frame,track_id,x1,y1,x2,y2`` (convert the released tracking CSVs to this layout).
    The 3D reconstructions shipped with LATTE-MV are the validation ground truth for
    recovered metric estimates and a baseline to compare against.
    """
    root_dir = _require_dir(root, "LATTE-MV", LATTE_MV_URL)
    tracks: list[TrackSequence] = []
    for csv_path in sorted(root_dir.rglob("*.csv")):
        tracks.extend(from_csv(csv_path, fps=fps))
    if not tracks:
        raise DatasetUnavailable(
            f"no per-clip CSVs found under {root_dir}; expected columns "
            "frame,track_id,x1,y1,x2,y2"
        )
    return tracks


def load_automotive_csv(path: str | Path, *, fps: float) -> list[TrackSequence]:
    """Load an automotive velocity set (TuSimple / BrnoCompSpeed) prepared as CSV.

    Cross-domain generality check (§14.3): the same generic ``sphere``/ballistic machinery
    and provenance contract must behave sensibly on a completely different domain. Provide
    a CSV with columns ``frame,track_id,x1,y1,x2,y2`` (optionally ``score,class_id``).
    """
    _require_dir(Path(path).parent, "automotive dataset", TUSIMPLE_VELOCITY_NOTE)
    return from_csv(path, fps=fps)


__all__ = [
    "BRNO_COMPSPEED_NOTE",
    "DatasetUnavailable",
    "LATTE_MV_URL",
    "TUSIMPLE_VELOCITY_NOTE",
    "load_automotive_csv",
    "load_latte_mv",
]
