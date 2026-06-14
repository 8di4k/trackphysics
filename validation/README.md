# validation/ — sim-to-real harness

This folder validates the engine's **metric** claims against the physical world. The design
principle: **success = calibration, not accuracy.** A metric speed that is 8% off can PASS
if the engine's stated CI covers the truth; a 3%-off estimate with a tight CI that excludes
the truth FAILs (overconfident).

## Ground truth must be INDEPENDENT (not monocular-reconstructed)

You cannot validate a gravity-derived metric against a dataset whose own 3D was built from
gravity + geometry — that is circular (BRIEF.md §10 at the data level). In particular
**LATTE-MV is NOT valid ground truth**: its 3D ball trajectory is itself monocular-
reconstructed (plane projection + projectile/Stokes-drag + self-calibration, ~8.9 cm
self-error). Its loaders are still useful as a source of *input* tracks and robustness
material — never as truth. Independent truth means multi-camera/stereo 3D, motion capture,
or a sensor (radar/Hawk-Eye). See the `validation-ground-truth-strategy` project memory.

## Two paths, in priority order

1. **Direct 3D-truth (primary).** When a dataset carries independent 3D truth, hold the 3D
   out and compare the engine's recovered metric against it. Single view → 2D track →
   `analyze()` → recovered speed vs true speed, per-trajectory error + CI coverage. This is
   stronger than the ruler road and is what [`run_3d_validation.py`](run_3d_validation.py)
   does. **TT3D** (table tennis, multi-cam 200 Hz 3D, CVPRW'25) is the first target — load
   it via [`datasets_tt3d.py`](datasets_tt3d.py).
   - *In-plane caveat:* v0.1 is monocular and recovers only the **in-plane** metric (depth
     axis zeroed). The fair comparison is against the in-plane projection of the true 3D
     velocity; the runner also reports full-3D error to expose the depth-blindness. Table
     tennis is also spin/Magnus-heavy — a **stress regime** for gravity-as-a-ruler, so a
     failure there must be distinguished from an engine bug. Keep a controlled DIY
     **heavy-ball, no-spin throw** as a clean anchor, and TrackMan/radar speeds as an
     orthogonal speed check.
2. **Ruler road (DIY fallback, no 3D truth).** With only a phone + a physical ruler: the
   ruler fixes meters-per-pixel, and ruler-scaled *vertical acceleration* should be ≈ 9.81
   ("is the world cooperative?"). `analyze()` then emits a speed + CI; PASS iff the CI covers
   the ruler-derived speed. Use this only when no independent 3D is available.

Either way a failure points at a specific layer instead of a vague "it's off."

## The single integration point

[`adapter_analyze.py`](adapter_analyze.py) — wired to the **real** engine. It reads a
`track.json`, runs the gravity road, and returns `EngineResult(tier, speed_m_s, ci95,
confidence, source)`. Important: it does **not** pass the ruler scale into `analyze()` —
that is Check A's road; mixing them would collapse the two roads.

`track.json` schema (produced by your `track_extract.py`):

```json
{
  "fps": 120.0,
  "image_size": [1280, 720],
  "frames":    [0, 1, 2, ...],
  "centroids": [[cx, cy], null, [cx, cy], ...]
}
```
`null` centroids are gaps (occlusion / empty frames) and are preserved; the engine accounts
for missing frames itself.

## Where the engine emits the CI

The launch speed and its CI come from `analyze(...).trajectory.meta`:

```python
est = analyze(track, preset="sphere", grounding=GroundingContext()).trajectory
est.tier                       # Tier.METRIC iff scale was earned
est.meta["launch_speed_m_s"]   # float
est.meta["launch_speed_ci95"]  # (lo, hi) or None
```

The CI is **measurement uncertainty (fit covariance) ⊕ a systematic floor** (default ~8%,
`trackphysics.core.ballistic._SYSTEMATIC_REL_FLOOR`) reflecting v0.1's known model
discrepancy: a pure-quadratic (drag-free) fit under the gravity-as-a-ruler assumptions. So
the CI is an honest *total* uncertainty, not fit-noise only — cooperative conditions stay
covered (PASS) while a gross assumption violation (steep camera) exceeds it (FAIL).
Tightening the floor is a v0.2 item (drag-augmented fit, stereo).

**Reference instant:** the speed is at the engine's *detected segment-start* frame
(`adapter_analyze` returns it as `at_frame`; `est.velocity.frame[0]`). Compare any
independent truth at THAT frame — on a decelerating arc, frame 0 is a different instant and
mixing them invalidates the verdict.

## Run the demos (no camera)

```bash
python -m validation.demo_3d_validation   # PRIMARY: direct 3D-truth path (per-traj + CI coverage)
python -m validation.demo_engine_check    # ruler/engine-road illustration (level/tilt PASS, steep FAIL)
```

## Bringing data / your kit in

- **TT3D (or any independent-3D set):** prepare clips per `datasets_tt3d.py`'s schema
  (`track2d.csv`, `gt3d.csv`, `optical_axis.txt`), then
  `run(load_tt3d(root))` → per-trajectory error + CI coverage.
- **DIY phone + ruler kit:** drop your tested files alongside the adapter —
  `track_extract.py` (video → `track.json`, gaps preserved), `run_validation.py` (calls
  `adapter_analyze(track_json)` → `tier`, `speed_m_s`, `ci95`, `at_frame`), `PROTOCOL.md` /
  `EXAMPLE_report.md`. Use this when no independent 3D is available; keep a heavy-ball,
  no-spin throw as the clean anchor.

`adapter_analyze` and `run_3d_validation.run` already return everything the checks need, so
wiring is import + call.
