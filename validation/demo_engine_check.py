"""Synthetic demo of Check B (the gravity/engine road) through the REAL engine.

Proves the calibration behavior the validation harness relies on, with no camera. The
coverage test compares the engine's launch speed against the true speed AT THE SAME FRAME
the engine reports it for (the detected segment start) — comparing against a different
instant on a decelerating arc would invalidate the verdict.

Outcomes (honest): a level camera and a *moderately* tilted one both PASS — the engine's
95% CI (measurement + v0.1 systematic floor) covers the truth. A *steeply* pitched camera
grossly violates the gravity-as-a-ruler assumption and FAILs as overconfident: a tight
METRIC CI that does not cover the truth.

Check A (the ruler road, ruler-scaled vertical acceleration vs g) is validated with the
physical ruler in the real-data harness and is out of scope here.

Run:  python -m validation.demo_engine_check     (from repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import GroundTruth, generate_track, look_at_camera  # noqa: E402
from validation.adapter_analyze import EngineResult, adapter_analyze  # noqa: E402

_CONDITIONS = {
    # label: (camera eye, camera target). level/tilt are cooperative-enough; steep grossly
    # violates the world-vertical <-> image-y mapping gravity-as-a-ruler assumes.
    "level": ((3.0, -7.0, 1.5), (3.0, 0.0, 1.7)),
    "tilt-mild": ((3.0, -6.0, 4.5), (3.0, 0.0, 1.0)),
    "steep-down": ((3.0, -4.0, 7.0), (3.0, 0.0, 0.5)),
}


def _scenario(
    eye: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[dict[str, Any], GroundTruth]:
    cam = look_at_camera(eye=eye, target=target)
    track, gt = generate_track(
        cam, fps=120.0, launch_velocity=(6.0, 0.0, 7.0), drag_coeff=0.2, duration=1.2
    )
    centers = track.centers()
    payload = {
        "fps": 120.0,
        "image_size": list(cam.image_size),
        "frames": [int(f) for f in track.frames],
        "centroids": [[float(c[0]), float(c[1])] for c in centers],
    }
    return payload, gt


def _truth_at(gt: GroundTruth, frame: int | None) -> float | None:
    """True speed at the frame the engine reports its speed for (segment start)."""
    if frame is None:
        return None
    matches = np.where(gt.frames.astype(int) == frame)[0]
    return float(gt.speed[int(matches[0])]) if matches.size else None


def _verdict(res: EngineResult, truth: float | None) -> str:
    if res.tier != "metric" or res.ci95 is None or truth is None:
        return "FALLBACK (no metric claim)"
    lo, hi = res.ci95
    return "PASS (CI covers truth)" if lo <= truth <= hi else "FAIL (overconfident)"


def main() -> None:
    print("Check B — gravity/engine road through the real trackphysics engine (synthetic).")
    print("Truth is compared at the engine's segment-start frame (consistent instant).\n")
    header = f"{'condition':11s} {'speed':>6s} {'CI95':>15s} {'truth@frame':>12s} {'bias':>6s}"
    print(header + "  verdict")
    for label, (eye, target) in _CONDITIONS.items():
        payload, gt = _scenario(eye, target)
        res = adapter_analyze(payload)
        truth = _truth_at(gt, res.at_frame)
        if res.tier == "metric" and res.ci95 is not None and res.speed_m_s is not None and truth:
            lo, hi = res.ci95
            bias = f"{100 * (res.speed_m_s - truth) / truth:+.0f}%"
            row = f"{res.speed_m_s:6.2f} {f'[{lo:.2f},{hi:.2f}]':>15s} {truth:>12.2f} {bias:>6s}"
        else:
            row = f"{'-':>6s} {'-':>15s} {'-':>12s} {'-':>6s}"
        print(f"{label:11s} {row}  {_verdict(res, truth)}")
    print(
        "\nExpected: level & tilt-mild -> PASS (covered); steep-down -> FAIL (overconfident). "
        "The CI is measurement noise + the v0.1 systematic floor, so cooperative geometries "
        "are covered while a gross assumption violation is correctly flagged."
    )


if __name__ == "__main__":
    main()
