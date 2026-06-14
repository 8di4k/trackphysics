"""Hero demo: real speed of a flying object from ONE static camera (BRIEF.md §9.6, §21).

The bottom block is the ~5 lines a user writes. Everything above it just manufactures a
stand-in for a real tracker's output (so the example runs with no video or model): we
simulate a thrown object, project it through a single known camera, and wrap each frame's
box in a tiny object that quacks like ``supervision``'s ``Detections``.

Run:  python examples/hero_speed.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import trackphysics as tp

# Make the in-repo `bench` package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import generate_track, look_at_camera  # noqa: E402


@dataclass
class FakeDetections:
    """Minimal ``sv.Detections`` look-alike: enough for the from_supervision adapter."""

    xyxy: np.ndarray
    tracker_id: np.ndarray
    confidence: np.ndarray


def _fake_tracker_output() -> tuple[list[FakeDetections], float]:
    """Stand in for a real detector+tracker: a thrown object seen by one camera."""
    fps = 120.0
    cam = look_at_camera(eye=(3.0, -7.0, 1.6), target=(3.0, 0.0, 1.6))
    track, _gt = generate_track(
        cam,
        fps=fps,
        launch_velocity=(6.0, 0.0, 7.0),  # world frame, +Z up
        drag_coeff=1e-9,                   # near-vacuum: a clean ballistic arc
        duration=1.2,
    )
    per_frame = [
        FakeDetections(
            xyxy=det.bbox[None, :],
            tracker_id=np.array([det.track_id]),
            confidence=np.array([1.0]),
        )
        for det in track.detections
    ]
    return per_frame, fps


def main() -> None:
    detections_per_frame, fps = _fake_tracker_output()

    # ---- the hero block: tracker output -> real-world speed, with provenance ----------
    track = tp.from_supervision(detections_per_frame, fps=fps)[0]
    result = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext())
    v = result.trajectory.velocity                 # a Quantity, not a bare float
    speed = float(np.linalg.norm(np.asarray(v.value)[0]))  # launch speed magnitude
    print(f"launch speed = {speed:.2f} {v.unit}  (tier={v.tier.value}, conf={v.confidence:.2f})")
    # -----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
