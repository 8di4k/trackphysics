# trackphysics

[![CI](https://github.com/8di4k/trackphysics/actions/workflows/ci.yml/badge.svg)](https://github.com/8di4k/trackphysics/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![types: mypy strict](https://img.shields.io/badge/types-mypy--strict-blue.svg)

**The physics layer of the computer-vision stack.** An object-agnostic, Apache-2.0-licensed Python
library that turns *object tracks from video* (boxes + ids, optionally keypoints) into *the
physics of motion* — 3D position, velocity, trajectories, and generic motion events — each
annotated with whether it is **metric**, **relative**, or **pixel**, and how confident we are.

It sits **above** trackers (which give pixel trajectories) and **below** domain apps (which
give meaning):

```
[ Detection ]   YOLO / RF-DETR / Roboflow Inference …
      ↓
[ Tracking ]    supervision / norfair / ByteTrack …   (boxes + ids)
      ↓
[ PHYSICS ]     ←—— this library  (3D lift, velocity, events, provenance)
      ↓
[ Domain apps ] sports / robotics / drones / traffic …  (semantics, metrics)
```

Nobody had packaged this integration layer as a clean developer library; everyone
re-implements it per domain. We package it once, make it robust to messy real-world tracks,
and **prove** accuracy with a benchmark.

---

## Three principles

1. **Domain-agnostic core.** Zero domain semantics in the engine — no `class_id`
   interpretation, no sport/object-specific constants. A CI test greps the package for a
   denylist of domain terms and fails the build on any hit. Domain knowledge enters only
   through explicit hooks (presets, a grounding slot, a skeleton graph).
2. **Provenance-first — "validated, not plausible."** Every physical output is a `Quantity`
   carrying a `Tier` (`metric` / `relative` / `pixel`) and a calibrated `confidence`. Metric
   tier is emitted **only** when scale is genuinely recovered; otherwise the engine falls
   back honestly. It never fabricates metric.
3. **Robustness is the moat.** The algorithms are public; the value is behaving well on
   *dirty* tracks (gaps, ID-switches, jitter, false positives) and **measuring** that from
   the first commit.

## Install

```bash
pip install -e .                 # core: numpy + scipy only (Jetson-friendly)
pip install -e ".[supervision]"  # + the supervision adapter
pip install -e ".[bench]"        # + matplotlib for the benchmark harness
pip install -e ".[dev]"          # + mypy, ruff, pytest
```

## Hero demo — real speed of a flying object from ONE static camera

```python
import trackphysics as tp

track = tp.from_supervision(my_detections, fps=120)[0]
res   = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext())
v     = res.trajectory.velocity          # a Quantity: value + unit + tier + confidence
print(v.value, v.unit, v.tier, v.confidence)
```

A runnable version (it manufactures its own tracker output, so no video/model is needed)
lives in [`examples/hero_speed.py`](examples/hero_speed.py):

```text
$ python examples/hero_speed.py
launch speed = 9.07 m/s  (tier=metric, conf=1.00)      # true speed 9.22 m/s
```

The scale here is recovered purely from gravity acting as a ruler — no court/field
calibration, no known object size.

## One-line `supervision` integration

```python
import trackphysics as tp
tracks = tp.from_supervision(byte_track_detections, fps=120)   # -> one TrackSequence per id
```

The adapter is duck-typed and lives behind an optional extra — the core never imports
`supervision`, so if its types change we change one file, not the engine.

## The three tiers

| Tier | Meaning | Emitted when |
|------|---------|--------------|
| `METRIC` | real-world units (m, m/s) | scale genuinely recovered (gravity-as-a-ruler that passes a physical-sanity gate, or a supplied `reference_scale`/`reference_plane`) |
| `RELATIVE` | 2.5D / normalized, scale-free | no metric scale available — the always-present floor |
| `PIXEL` | image-space 2D + pixel-rate kinematics | e.g. 2D-projected joint angles |

`analyze()` always returns a usable result: a metric estimate when earned, a relative-lift
floor otherwise, plus generic events, scale-invariant keypoint kinematics, and a
track-quality report saying *where* the track is unreliable.

## Benchmark — the accuracy/robustness envelope

Run `python -m bench.run` to regenerate the full report (numbers + plots) under
[`bench/report/`](bench/report/). On synthetic data with exact metric ground truth (a
physics-simulated, camera-projected ballistic arc) under **correlated** (not IID) corruption:

- **Clean metric accuracy.** Mean launch-speed error ≈ **0.28 m/s** (~2%) — measured on arcs
  with **realistic aerodynamic drag** (coefficient 0.2), a genuine model mismatch for the
  v0.1 pure-quadratic fit. (On drag-free arcs it is ~0.04 m/s, but that would be grading the
  estimator on its own simplifying assumption, so the headline uses the harder, honest case.)
- **Graceful degradation (the moat).** Mean |launch-speed error| under a rising
  false-positive/outlier rate, on **drag-free** arcs so the fitting-robustness effect is
  isolated (drag is a separate axis). Both methods use identical gravity-as-a-ruler; only the
  fit differs (robust RANSAC+IRLS vs naive `polyfit`):

  | outlier rate | 0% | 10% | 20% | 30% | 40% | 50% |
  |---|---|---|---|---|---|---|
  | naive polyfit | 0.01 | 0.06 | 0.13 | 0.12 | 0.15 | 0.17 |
  | **robust (ours)** | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.03 |

  At 50% outliers the robust fit is ~6× more accurate than the naive baseline. (Against
  pure zero-mean *jitter* the two are comparable, as expected — RANSAC/IRLS defends against
  outliers, not Gaussian noise; the benchmark reports this honestly.)
- **Model mismatch (drag).** A dedicated curve reports speed error vs the true drag
  coefficient (≈0.01 m/s at zero drag rising to ≈1.1 m/s at drag 1.0) — the honest cost of
  the v0.1 pure-quadratic fit. A drag-augmented fit is a clean future extension.
- **Provenance calibration.** Expected Calibration Error of stated confidence vs empirical
  correctness ≈ **0.06** — stated confidence ≈ actual correctness.
- **Metric-vs-fallback gate.** Positive = emits METRIC; positives are clean recoverable arcs,
  negatives include trivially-unrecoverable tracks **and a hard case** (a clean parabola from
  a *pitched* camera, where gravity-as-a-ruler is violated): **precision ≈ 0.67, recall 1.00,
  F1 ≈ 0.80**. It never fabricates metric on trivially-unrecoverable input, but it *cannot*
  detect the pitched-camera assumption violation monocularly in v0.1 — there it still emits a
  biased metric (~21% speed bias). That limitation is measured and published, not hidden.

Real-data validation loaders (LATTE-MV, then an automotive set) are scaffolded in
[`bench/datasets.py`](bench/datasets.py); they expect locally-prepared data and do not
auto-download. Set `TRACKPHYSICS_LATTE_MV` to a prepared directory to add a real-data pass;
otherwise the report prints an explicit **synthetic-only / sim2real-unmeasured** banner.

> **Known characteristics (v0.1).** Gravity-as-a-ruler assumes a static, roughly-horizontal
> camera with near-constant depth over a short arc, and fits a pure ballistic parabola
> (aerodynamic drag is a clean future extension). Two honest consequences, both measured in
> the report: (1) a **pitched/oblique camera** breaks the world-vertical↔image-y mapping and
> yields a biased metric estimate the engine can't yet detect monocularly (a v0.2 cross-check
> / stereo item); (2) scale recovery is **SNR-limited** — at high fps the per-frame curvature
> signal can fall below positional jitter, and the engine then honestly falls back rather
> than guess. Single-arc gravity-as-a-ruler *assumes* g; it does not independently recover
> and cross-check it (also v0.2).

## What this is *not*

Not a detector or tracker (we consume their output), not a foundation model, not a CV
runtime, not human biomechanics, and **not** any domain product. Domain layers are separate
projects built on top. See `BRIEF.md` for the full brief.

## Development

```bash
python -m ruff check src bench tests examples
python -m mypy            # mypy --strict on the package
python -m pytest          # unit + integration + the domain-guardrail test
python -m bench.run       # regenerate the benchmark report
```

## License & contributing

[Apache-2.0](LICENSE) — permissive, with an explicit patent grant (chosen over MIT for
that grant, and over AGPL because this is an adoption play where copyleft friction works
against the goal; see `BRIEF.md` §19).

Contributions require a lightweight [Contributor License Agreement](CLA.md) (see
[CONTRIBUTING.md](CONTRIBUTING.md)). The CLA keeps the project's options open — including a
possible future dual-license of the core — without imposing AGPL-style friction on users
today. (Domain products built *on top* can be proprietary regardless of the core license.)
