# validation/ — sim-to-real harness for the gravity road

This folder validates the engine's **metric** claims against the physical world. The design
principle: **success = calibration, not accuracy.** A metric speed that is 8% off can PASS
if the engine's stated CI covers the truth; a 3%-off estimate with a tight CI that excludes
the truth FAILs (overconfident).

## Two independent roads to meters

- **Check A — the ruler road (no gravity).** A ruler in frame fixes meters-per-pixel.
  Ruler-scaled *vertical acceleration* should be ≈ 9.81 m/s². This asks "is the world
  cooperative?" (static, ~horizontal camera, ruler in the trajectory plane) — entirely
  without the engine. Under tilt it drops below g, localizing the failure to the *setup*.
- **Check B — the gravity road (the engine).** `analyze()` recovers metric scale from
  gravity and emits a launch speed **+ 95% CI**. PASS iff the CI covers the independent
  (ruler-derived) truth. This asks "is the engine right?"

Splitting the two roads means a failure points at a specific layer instead of a vague "it's
off."

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
covered (PASS) while a gross assumption violation (pitched camera) exceeds it (FAIL).
Tightening the floor is a v0.2 item (drag-augmented fit, stereo).

## Run the engine-road demo (no camera)

```bash
python -m validation.demo_engine_check
# level      -> PASS (CI covers truth);  tilt-30deg -> FAIL (overconfident)
```

## Bringing your kit in

Drop your tested files alongside the adapter:

- `track_extract.py` — video → `track.json` (OpenCV; gaps preserved).
- `run_validation.py` — `track.json` + ruler → report; call `adapter_analyze(track_json)`
  for Check B (it returns `tier`, `speed_m_s`, `ci95`).
- `PROTOCOL.md` / `EXAMPLE_report.md` — how to shoot (ruler + drop test), conditions
  (level / tilt), ≥10 throws per condition, pass/fail criteria.

`adapter_analyze` already returns everything Check B needs, so wiring is import + call.
