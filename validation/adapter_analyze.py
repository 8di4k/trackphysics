"""ADAPT-ME wiring: a track.json (from track_extract.py) -> the real trackphysics engine.

This is the single integration point between the validation harness and the engine. It is
now wired to the real ``trackphysics.analyze`` (no stub) and reads the launch speed + 95%
CI the engine emits in ``meta``.

track.json schema (what track_extract.py produces, what this consumes):

    {
      "fps": 120.0,
      "image_size": [W, H] | null,            # pixels; helps normalization, optional
      "frames":    [int, ...],                # sampled frame indices (gaps = missing/None)
      "centroids": [[cx, cy] | null, ...]     # per-frame object centroid in px; null = gap
    }

Check B contract: this is the GRAVITY road (what analyze() does). Do NOT feed the ruler
scale into the engine here — the ruler is Check A's independent road; feeding it would
collapse the two roads into one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import trackphysics as tp

# Our Detection requires a bbox; the ballistic metric uses only the centroid, so a nominal
# half-size around the centroid is sufficient (its value does not affect the metric path).
_BBOX_HALF_PX = 6.0


@dataclass
class EngineResult:
    """What Check B reads off the engine."""

    tier: str                       # "metric" | "relative" | "pixel"
    speed_m_s: float | None         # speed at `at_frame`; None unless tier == "metric"
    ci95: tuple[float, float] | None  # 95% CI on the speed; None only if direction degenerate
    confidence: float               # calibrated scalar in [0, 1]
    source: str                     # e.g. "ballistic_fit" | "reference_scale" | "relative_fallback"
    at_frame: int | None = None     # frame the speed pertains to (detected segment start).
    """CRITICAL for coverage tests: ``speed_m_s`` is the speed at the *segment-start* frame,
    not necessarily the track's first frame. Compare any independent truth (ruler-derived
    speed, drop-test) at THIS frame — mixing instants on a decelerating arc invalidates the
    coverage verdict."""


def load_track(path: str | Path) -> dict[str, Any]:
    """Read a track.json file."""
    return json.loads(Path(path).read_text())  # type: ignore[no-any-return]


def adapter_analyze(track_json: dict[str, Any]) -> EngineResult:
    """Run the gravity road (engine) on a track.json and return speed/tier/CI.

    Gaps (null centroids) are dropped — the engine accounts for missing frames itself.
    """
    fps = float(track_json["fps"])
    raw_size = track_json.get("image_size")
    image_size = (int(raw_size[0]), int(raw_size[1])) if raw_size else None

    frames: list[int] = []
    boxes: list[list[float]] = []
    for frame, centroid in zip(track_json["frames"], track_json["centroids"], strict=True):
        if centroid is None:  # gap / empty frame
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        h = _BBOX_HALF_PX
        frames.append(int(frame))
        boxes.append([cx - h, cy - h, cx + h, cy + h])

    if len(frames) < 3:
        return EngineResult("pixel", None, None, 0.0, "insufficient_track")

    track = tp.from_generic(
        frames=frames,
        boxes=np.asarray(boxes, dtype=np.float64),
        track_ids=[1] * len(frames),
        fps=fps,
        image_size=image_size,
    )[0]

    # GRAVITY road: no reference_scale (that is Check A's ruler road).
    est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
    velocity = est.velocity
    if velocity.tier is not tp.Tier.METRIC:
        return EngineResult(
            velocity.tier.value, None, None, float(velocity.confidence), velocity.source
        )

    speed = float(est.meta["launch_speed_m_s"])  # type: ignore[arg-type]
    raw_ci = est.meta.get("launch_speed_ci95")
    ci95 = (float(raw_ci[0]), float(raw_ci[1])) if raw_ci is not None else None  # type: ignore[index]
    # The speed is at the segment-start frame; expose it so coverage is tested there.
    frame = velocity.frame
    at_frame = int(frame[0]) if isinstance(frame, tuple) else None
    return EngineResult(
        "metric", speed, ci95, float(velocity.confidence), velocity.source, at_frame
    )


__all__ = ["EngineResult", "adapter_analyze", "load_track"]
