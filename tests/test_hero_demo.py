"""Keep the README hero-demo transcript in sync with the actual deterministic output.

The example is fully deterministic (fixed synthetic arc + seeded RANSAC), so its printed
speed must match the value quoted in README.md. This guards against the transcript silently
drifting from reality (README-HERO-STALE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import trackphysics as tp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.hero_speed import _fake_tracker_output  # noqa: E402

_README_SPEED = "9.10"  # the value quoted in README.md's hero transcript


def test_hero_demo_output_matches_readme() -> None:
    per_frame, fps = _fake_tracker_output()
    track = tp.from_supervision(per_frame, fps=fps)[0]
    v = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory.velocity
    speed = float(np.linalg.norm(np.asarray(v.value)[0]))
    assert v.tier is tp.Tier.METRIC
    assert f"{speed:.2f}" == _README_SPEED
