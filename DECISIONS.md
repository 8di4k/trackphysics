# Decision log — measured limitations & resolutions

Engineering decisions and **measured** limitations surfaced by building and validation,
recorded in the style of BRIEF.md §10 (provenance is a hard contract, not a vibe).

> Quantitative numbers from the all-rights-reserved **TT3D** set are kept in the **internal**
> validation artifact (`TRACKPHYSICS_VALIDATION_RESULTS.md` + `per_trajectory_dump.csv`
> alongside the local dataset clone), **not** committed here — clear terms before publishing.
> This log states the qualitative engineering conclusions only.

---

## 2026-06-14 — Real-data validation: architecture confirmed, quantitative provenance falsified

First non-circular real-data validation: the engine run against **independent multi-camera
3D** (TT3D) via the standing harness (`validation/run_tt3d.py`). The result is a **diagnosis,
not a green light** — and exactly the diagnosis a provenance-first project needs.

- **Tier discipline holds in aggregate.** METRIC is emitted on only a minority of trajectories
  with honest RELATIVE fallback otherwise; the engine does not fake metric to look good. The
  format-risk did not materialize — the loader's assumptions matched the real data unchanged.

- **MEASURED LIMITATION A — the metric gate is blind to depth (a §10 hole).** A trajectory
  flying along the optical axis still projects a *clean in-plane parabola*, which the gate
  (low residual, enough inliers, appreciable downward acceleration) accepts. Consequently the
  engine emits METRIC **more** often on the **most depth-dominated** arcs (metric-rate rises
  monotonically with true depth-domination; the correlation is positive). The emitted value is
  a correct *in-plane* metric (depth is zeroed and noted in `meta`), but it is a small,
  **unflagged** fraction of the true 3D motion — the gate rewards exactly where it is most
  3D-blind.
  - *Caveat on the fix:* the obvious deployable downgrade signal — apparent bbox-size change ⇒
    depth motion — is **structurally absent on TT3D** (point-only `u,v` observations; the
    loader uses constant synthetic boxes), so it is untestable on this dataset. The testable
    software paths are (a) a real detector's size channel, or (b) the `sphere` preset's known
    diameter used as an *object-size-as-ruler* and cross-checked against gravity-as-a-ruler
    (a §10 cross-cue consistency check). Stereo (v0.2) removes the limitation outright.

- **MEASURED LIMITATION B — the fixed systematic CI floor falsifies provenance off-regime.**
  The v0.1 constant ~8% systematic floor (`ballistic._SYSTEMATIC_REL_FLOOR`) is calibrated for
  cooperative moderate-drag conditions. On the spin/drag/depth stress regime the stated 95% CI
  **under-covers** the truth badly and **regime-dependently** (coverage varies several-fold
  across the three camera viewpoints). A larger fixed floor cannot cover three regimes at once
  — the resolution is **input-conditioned uncertainty**: CI width as a learned function of
  observable features (fit residual, inlier fraction, completeness, recovered acceleration /
  scale, a depth-motion proxy). This is **v0.2 provenance priority #1**; the measured
  per-trajectory error distribution (internal artifact) is the training / reliability target.

- **Root-cause synthesis.** The signed speed bias **flips sign with camera geometry** (one
  view over-estimates, another under-estimates) and is **strongly anti-correlated with the
  recovered pixel acceleration**. So it is a **scale-recovery error driven by depth motion
  violating the constant-depth assumption**, not a clean drag offset — a drag-augmented fit
  alone will not remove it. **Limitation A (tier) and Limitation B (CI/bias) share one root:
  monocular depth-blindness**, which corrupts both the gate decision and the recovered scale.

**Status vs DoD §17:** "real-data validation ran" = **YES**; "provenance validated" = **NO**
(this entry is the falsification + the labelled error distribution to fix it). The Apache-2.0
license headers and the §10 provenance redline remain open work. The TT3D harness
(`validation/run_tt3d.py`) is the standing real-data eval; its quantitatives are internal
(license).

---

## 2026-06-14 (follow-up) — Discriminative-uncertainty scaffold: the point-only ceiling is REFUTED

Built a two-stage calibration scaffold (`validation/calibrate.py`, fed by
`validation/make_tt3d_dump.py`): Stage 1 de-biases the point estimate from runtime-observable
features, Stage 2 fits an input-conditioned CI on the residual, under leave-one-view-out CV.
A first pass on **fit-quality** features alone (residual, inliers, completeness, a_px, gof)
failed to generalize and tempted a "point-only is exhausted" conclusion. **That conclusion was
wrong** — it had not tested trajectory **shape**.

Adding four scale-invariant, point-only trajectory-shape features (in-plane aspect ratio,
chord/arc straightness, PCA orientation and eccentricity of the (u,v) cloud) changed the
picture:

- **DECIDED — a point-only depth signal EXISTS.** The in-plane **aspect ratio** is a *strong*
  proxy for depth-domination (the others weaker). The earlier "no point-only depth signal /
  size-or-stereo strictly required" framing applied only to the fit-quality features and is
  withdrawn for shape features.
- **DECIDED — the §10 tier-hole (Limitation A) is point-only fixable, validated.** Thresholding
  the metric emissions on the aspect-based depth score **monotonically reduces the retained
  set's full-3D error** (dropping the depth-dominated arcs the gate wrongly trusts). So the
  honest downgrade of depth-dominated arcs does NOT require a size channel or stereo — those
  are for *recovering* the missing depth, not for *flagging* it. (Implementing this in the
  `sphere` gate is the queued software fix — to be landed only once validated on size-bearing
  data too, never hardcoded; §6.)
- **DECIDED (locked) — per-deployment calibration is a deployable v0.2 path.** With geometry
  in-distribution (a fixed rig: stationary cameras, a padel court), the Stage-1 de-bias plus
  Stage-2 input-conditioned CI reaches ~nominal 95% coverage. Ship it as a **refittable
  per-deployment artifact**, never as hardcoded constants (geometry-specific; §6).
- **Why cross-geometry still fails under leave-one-view-out — and the corrected cause.** It is
  NOT lack of signal: it is that three discrete, well-separated viewpoints make leave-one-out a
  pure **extrapolation** into an unseen region of shape-space, where a strong viewpoint feature
  hurts. The blocker for a *general* (geometry-agnostic) calibrated CI is therefore **viewpoint
  DIVERSITY in the calibration set (a continuum of angles)**, not the absence of an observable
  signal. Size/stereo is one way to get richer geometry, but the limiting resource is
  calibration-data diversity, not the feature.

### §10 AMENDMENT (provenance redline — a strengthening, not a retreat)

> A **calibrated** metric CI is earned only under one of: (a) a depth-bearing cue (a real
> detector's apparent-size channel, or stereo), or (b) a **per-deployment** calibration fit on
> the deployment's own (fixed) geometry. Absent both — i.e. unknown geometry with no
> per-deployment fit — the engine must **widen the CI or downgrade the tier** rather than emit a
> tight CI it cannot honour. The downgrade signal is now known to be **available point-only**
> (a trajectory-shape / depth-domination feature), so this clause is implementable without new
> sensors. "Tight metric CI on unknown geometry" is a §10 violation.

**Status:** ceiling claim resolved (refuted) by the geo-test, as required before locking.
`validation/calibrate.py` + `validation/make_tt3d_dump.py` are committed as **refittable
artifacts** (code only — no TT3D numbers; the fitted coefficients are deployment-specific and
must never enter the core, §6). Quantitatives remain in the internal artifact (license).
