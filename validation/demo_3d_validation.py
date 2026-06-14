"""Synthetic proof of the direct 3D-truth validation path (stand-in for TT3D).

Generates trajectories with KNOWN 3D (the bench simulator's true positions + camera) and
runs run_3d_validation over them: single view -> 2D track -> analyze() -> recovered metric
speed vs the in-plane projection of the true 3D velocity, per-trajectory + aggregate (mean
in-plane error, full-3D error, CI coverage rate). With real TT3D data this same runner
consumes datasets_tt3d.load_tt3d(...) instead of the synthetic generator below.

Run:  python -m validation.demo_3d_validation     (from repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.synth.generator import generate_track, look_at_camera  # noqa: E402
from validation.run_3d_validation import Trajectory3D, run  # noqa: E402


def _make_trajectories() -> list[Trajectory3D]:
    trajs: list[Trajectory3D] = []
    rng = np.random.default_rng(0)
    # Mostly cooperative (level) geometry; one steep camera to show the depth-blindness.
    specs = [("level", (3.0, -7.0, 1.5), (3.0, 0.0, 1.7))] * 8 + [
        ("steep", (3.0, -4.0, 7.0), (3.0, 0.0, 0.5))
    ] * 2
    for i, (label, eye, target) in enumerate(specs):
        cam = look_at_camera(eye=eye, target=target)
        vx = float(rng.uniform(4.0, 8.0))
        vz = float(rng.uniform(5.0, 9.0))
        track, gt = generate_track(
            cam, track_id=1, fps=120.0, launch_velocity=(vx, 0.0, vz),
            drag_coeff=0.2, duration=1.2,
        )
        trajs.append(
            Trajectory3D(
                track=track,
                gt_positions_m=gt.positions,          # aligned 1:1 with track.detections
                optical_axis=cam.R[2],                # camera +Z (forward) in world
                name=f"{label}-{i}",
            )
        )
    return trajs


def main() -> None:
    report = run(_make_trajectories())
    print("Direct 3D-truth validation (synthetic stand-in for TT3D):\n")
    print(f"{'trajectory':12s} {'recovered':>9s} {'inplane':>8s} {'full3D':>7s} {'covers?':>8s}")
    for r in report.results:
        if r.tier == "metric" and r.recovered_speed is not None:
            print(
                f"{r.name:12s} {r.recovered_speed:9.2f} {r.inplane_truth:8.2f} "
                f"{r.full3d_truth:7.2f} {str(r.ci_covers_inplane):>8s}"
            )
        else:
            print(f"{r.name:12s} {'(' + r.tier + ')':>9s}")
    s = report.summary()
    print("\nsummary:")
    print(f"  metric / total      : {s['n_metric']:.0f} / {s['n_total']:.0f}")
    print(f"  mean in-plane error : {s.get('mean_inplane_error_m_s', float('nan')):.3f} m/s")
    print(f"  mean full-3D error  : {s.get('mean_full3d_error_m_s', float('nan')):.3f} m/s")
    print(f"  CI coverage rate    : {s.get('ci_coverage_rate', float('nan')):.2f}")
    print(
        "\nLevel geometry: in-plane ≈ full-3D (low error, CI covers). Steep geometry: the "
        "in-plane truth is what the monocular engine can see; full-3D error grows because the "
        "depth component is invisible (a v0.2/stereo concern)."
    )


if __name__ == "__main__":
    main()
