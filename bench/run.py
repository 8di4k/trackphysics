"""Benchmark runner — the accuracy / robustness envelope (BRIEF.md §14).

Produces, on synthetic data with exact metric ground truth:

1. **Clean metric accuracy** of recovered launch speed (gravity-as-a-ruler).
2. **Degradation curves**: launch-speed error vs corruption level, for our robust
   RANSAC+IRLS fit against a naive ``polyfit`` baseline. Both use the *identical*
   gravity-as-a-ruler scale step, so the only variable is robust vs non-robust fitting —
   this isolates the moat (§14.4 "we degrade more gracefully than naive baselines").
3. **Provenance calibration**: a reliability plot of stated confidence vs empirical
   correctness, with ECE, over the full ``analyze(preset="sphere")`` pipeline.
4. **Gate behavior**: precision/recall of the METRIC-vs-fallback decision — does the
   engine correctly refuse metric when scale is unrecoverable?

Run:  python -m bench.run        (writes plots + report under bench/report/)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import trackphysics as tp  # noqa: E402
from bench.datasets import DatasetUnavailable, load_latte_mv  # noqa: E402
from bench.metrics import gate_precision_recall, reliability  # noqa: E402
from bench.perturb import CorruptionConfig, corrupt  # noqa: E402
from bench.synth.generator import GroundTruth, generate_track, look_at_camera  # noqa: E402
from trackphysics.core.ballistic import irls_quadratic, ransac_quadratic  # noqa: E402

GRAVITY = 9.81
REPORT_DIR = Path(__file__).resolve().parent / "report"
CORRECT_REL_TOL = 0.25  # a metric speed within 25% of truth counts as "correct"


def make_arc(
    rng: np.random.Generator, *, fps: float = 120.0, drag_coeff: float = 0.2
) -> tuple[tp.TrackSequence, GroundTruth]:
    """A ballistic arc viewed by a roughly-horizontal camera (gravity-as-a-ruler holds).

    Uses REALISTIC aerodynamic drag by default (``drag_coeff=0.2``), NOT a drag-free arc.
    The v0.1 fit is a pure quadratic (constant-acceleration) model, so a dragged arc is a
    genuine model mismatch — the headline accuracy is therefore measured on data the
    estimator does *not* perfectly model, avoiding the circularity of grading the engine on
    its own simplifying assumption (BENCH-01 / BRIEF.md §14's over-clean-synthetic warning).
    """
    vx = float(rng.uniform(4.0, 8.0))
    vz = float(rng.uniform(5.0, 9.0))
    depth = float(rng.uniform(6.0, 9.0))
    cam = look_at_camera(eye=(3.0, -depth, 1.5), target=(3.0, 0.0, 1.7))
    return generate_track(
        cam, fps=fps, launch_velocity=(vx, 0.0, vz), drag_coeff=drag_coeff, duration=1.2
    )


def _truth_at_segment_start(est: tp.TrajectoryEstimate, gt: GroundTruth) -> float | None:
    """True speed at the frame the engine reports its speed for (the segment start).

    The engine's launch speed is the speed at the START of the detected free-flight segment,
    which on a decelerating arc is a later frame than launch. Comparing it against the true
    speed at frame 0 is a reference-instant error; compare at the segment-start frame instead.
    """
    frame = est.velocity.frame
    if not isinstance(frame, tuple):
        return None
    matches = np.where(gt.frames.astype(np.int64) == int(frame[0]))[0]
    return float(gt.speed[int(matches[0])]) if matches.size else None


def make_steep_arc(rng: np.random.Generator) -> tuple[tp.TrackSequence, GroundTruth]:
    """A clean ballistic arc viewed by a STEEPLY pitched (downward-looking) camera.

    This grossly violates gravity-as-a-ruler (world-vertical no longer maps to image-y, depth
    changes strongly along the arc), so the recovered METRIC scale is badly biased (~tens of
    percent at the segment-start instant) — a genuine "metric-untrustworthy" HARD NEGATIVE for
    the gate. A *mild* tilt, by contrast, the engine tolerates (~few percent), so it would NOT
    be a fair hard negative; the steep geometry is the real failure surface (BENCH-03).
    """
    vx = float(rng.uniform(4.0, 8.0))
    vz = float(rng.uniform(5.0, 9.0))
    cam = look_at_camera(eye=(3.0, -4.0, 7.0), target=(3.0, 0.0, 0.5))
    return generate_track(
        cam, fps=120.0, launch_velocity=(vx, 0.0, vz), drag_coeff=0.2, duration=1.2
    )


def make_non_ballistic(rng: np.random.Generator) -> tp.TrackSequence:
    """Near-constant-velocity motion: no usable free-flight signature (scale NOT recoverable)."""
    cam = look_at_camera(eye=(3.0, -7.0, 1.5), target=(3.0, 0.0, 1.7))
    track, _gt = generate_track(
        cam, fps=120.0, launch_velocity=(5.0, 4.0, 0.2), gravity=1e-6, drag_coeff=1e-9,
        duration=0.8,
    )
    return track


def _launch_speed_fit(track: tp.TrackSequence, *, robust: bool, rng: np.random.Generator) -> float:
    """Estimate launch speed (m/s) via gravity-as-a-ruler from a 2D track.

    Both branches share the scale step ``s = g / |a_px|``; they differ only in whether the
    quadratic fit to image-y is robust (RANSAC + IRLS) or naive (``np.polyfit``).
    """
    t = track.times()
    centers = track.centers()
    if t.shape[0] < 4:
        return float("nan")
    x, y = centers[:, 0], centers[:, 1]
    if robust:
        seed = ransac_quadratic(t, y, rng=rng)
        fy = irls_quadratic(t, y, seed=seed, rng=rng)
        a_px, vy0 = fy.accel, fy.c1
        mask = fy.inlier_mask > 0.5
        tx, xx = (t[mask], x[mask]) if int(mask.sum()) >= 2 else (t, x)
        vx0 = float(np.polyfit(tx, xx, 1)[0])
    else:
        cy = np.polyfit(t, y, 2)
        a_px, vy0 = float(2.0 * cy[0]), float(cy[1])
        vx0 = float(np.polyfit(t, x, 1)[0])
    if abs(a_px) < 1e-9:
        return float("nan")
    scale = GRAVITY / abs(a_px)  # meters per pixel
    return float(scale * np.hypot(vx0, vy0))


@dataclass
class Curve:
    levels: list[float]
    robust_err: list[float]
    baseline_err: list[float]


def degradation_sweep(
    field: str, levels: list[float], *, n_seeds: int = 16, base_seed: int = 0
) -> Curve:
    """Mean absolute launch-speed error vs a single corruption parameter."""
    robust_errs: list[float] = []
    baseline_errs: list[float] = []
    for li, level in enumerate(levels):
        r_acc: list[float] = []
        b_acc: list[float] = []
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + 1000 * li + s)
            if field == "drag_coeff":
                # Model-mismatch axis: vary true drag, NO track corruption. Both fits share
                # the (pure-quadratic) model bias, so this isolates the model-mismatch error.
                track, gt = make_arc(rng, drag_coeff=level)
                cfg = CorruptionConfig()
            else:
                # ISOLATION: the corruption sweeps use DRAG-FREE arcs so the robust-vs-naive
                # fitting difference is not masked by the (common-mode) drag model bias. Drag
                # is its own axis (the model-mismatch curve); mixing it in here would hide the
                # very robustness effect these curves exist to measure.
                track, gt = make_arc(rng, drag_coeff=0.0)
                if field == "jitter_sigma_px":
                    cfg = CorruptionConfig(jitter_sigma_px=level, jitter_rho=0.6)
                elif field == "fp_rate":
                    cfg = CorruptionConfig(fp_rate=level)
                else:
                    raise ValueError(f"unsupported sweep field: {field}")
            true_speed = float(gt.speed[0])
            dirty = corrupt(track, cfg, rng)
            r = _launch_speed_fit(dirty, robust=True, rng=np.random.default_rng(base_seed + s))
            b = _launch_speed_fit(dirty, robust=False, rng=rng)
            if np.isfinite(r):
                r_acc.append(abs(r - true_speed))
            if np.isfinite(b):
                b_acc.append(abs(b - true_speed))
        robust_errs.append(float(np.mean(r_acc)) if r_acc else float("nan"))
        baseline_errs.append(float(np.mean(b_acc)) if b_acc else float("nan"))
    return Curve(levels=levels, robust_err=robust_errs, baseline_err=baseline_errs)


def _short_track(rng: np.random.Generator) -> tp.TrackSequence:
    """A genuinely too-short arc: below the segment-detection minimum -> not recoverable."""
    track, _gt = make_arc(rng)
    return tp.TrackSequence(
        detections=track.detections[:5], fps=track.fps, image_size=track.image_size
    )


def run_gate(*, n_runs: int = 120, base_seed: int = 50) -> dict[str, float]:
    """Precision/recall of the METRIC-vs-fallback decision, including a HARD negative.

    Positive (metric trustworthy) = clean ballistic arc, horizontal camera, scale earnable.
    Negative (should refuse / untrustworthy) =
      * trivial: near-constant-velocity motion, or a too-short track; AND
      * HARD (BENCH-03): a clean parabola viewed by a STEEPLY pitched camera, where
        gravity-as-a-ruler is grossly violated so the recovered scale is badly biased. v0.1
        cannot detect this monocularly, so these surface as false positives — the gate is
        deliberately NOT trivially separable. (A *mild* tilt the engine tolerates, so it
        would be an unfair hard negative; the steep geometry is the real failure surface.)

    Bias is measured at the engine's segment-start instant. Also reports the steep-camera
    false-positive rate and mean relative speed bias so the limitation is quantified.
    """
    predicted_metric: list[float] = []
    scale_recoverable: list[float] = []
    steep_total = 0
    steep_metric = 0
    steep_bias: list[float] = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        kind = i % 4
        if kind in (0, 1):  # genuinely recoverable: clean arcs at varied frame rates
            track, _gt = make_arc(rng, fps=120.0 if kind == 0 else 60.0)
            scale_recoverable.append(1.0)
            est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
            predicted_metric.append(1.0 if est.tier == tp.Tier.METRIC else 0.0)
        elif kind == 2:  # trivially unrecoverable
            track = _short_track(rng) if (i % 2 == 0) else make_non_ballistic(rng)
            scale_recoverable.append(0.0)
            est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
            predicted_metric.append(1.0 if est.tier == tp.Tier.METRIC else 0.0)
        else:  # HARD negative: STEEP camera -> metric is badly biased / untrustworthy
            track, gt = make_steep_arc(rng)
            scale_recoverable.append(0.0)
            est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
            is_metric = est.tier == tp.Tier.METRIC
            predicted_metric.append(1.0 if is_metric else 0.0)
            steep_total += 1
            if is_metric:
                steep_metric += 1
                # Bias at the segment-start instant the engine actually reports (not frame 0).
                truth = _truth_at_segment_start(est, gt)
                speed = float(est.meta["launch_speed_m_s"])  # type: ignore[arg-type]
                if truth:
                    steep_bias.append(abs(speed - truth) / truth)
    gate = gate_precision_recall(np.array(predicted_metric), np.array(scale_recoverable))
    gate["steep_false_positive_rate"] = steep_metric / steep_total if steep_total else 0.0
    gate["steep_mean_rel_bias"] = float(np.mean(steep_bias)) if steep_bias else float("nan")
    return gate


def run_calibration(
    *, n_runs: int = 160, base_seed: int = 300
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], float]:
    """Reliability data + clean error, over METRIC emissions across a jitter spread.

    Light-to-moderate jitter that still passes the gate yields a *spread* of stated
    confidences (residual erodes confidence), which is what makes the reliability plot
    informative.
    """
    jitter_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    confidences: list[float] = []
    correct: list[float] = []
    clean_errors: list[float] = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        track, gt = make_arc(rng)
        sigma = jitter_levels[i % len(jitter_levels)]
        dirty = corrupt(track, CorruptionConfig(jitter_sigma_px=sigma, jitter_rho=0.6), rng)
        est = tp.analyze(dirty, preset="sphere", grounding=tp.GroundingContext()).trajectory
        if est.tier != tp.Tier.METRIC:
            continue
        # Compare at the segment-start instant the engine reports (not frame 0).
        true_speed = _truth_at_segment_start(est, gt)
        if true_speed is None:
            continue
        speed = float(est.meta["launch_speed_m_s"])  # type: ignore[arg-type]
        confidences.append(float(est.velocity.confidence))
        correct.append(1.0 if abs(speed - true_speed) / true_speed < CORRECT_REL_TOL else 0.0)
        if sigma == 0.0:
            clean_errors.append(abs(speed - true_speed))
    rel = reliability(np.array(confidences), np.array(correct), n_bins=8)
    clean_err = float(np.mean(clean_errors)) if clean_errors else float("nan")
    return rel, clean_err


def _plot_curve(curve: Curve, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.levels, curve.baseline_err, "o--", label="naive polyfit baseline", color="#c44")
    ax.plot(curve.levels, curve.robust_err, "o-", label="robust (RANSAC+IRLS)", color="#1a7")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean |launch-speed error|  (m/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_reliability(rel: tuple[np.ndarray, np.ndarray, np.ndarray, float], path: Path) -> None:
    centers, acc, counts, ece = rel
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    valid = np.isfinite(acc)
    ax.plot(centers[valid], acc[valid], "o-", color="#36c", label="empirical")
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("empirical correctness")
    ax.set_title(f"Provenance reliability (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_real_data() -> dict[str, object] | None:
    """Optional LATTE-MV sim2real pass when ``TRACKPHYSICS_LATTE_MV`` points to prepared data.

    Runs ``analyze`` over real tracks and reports the tier distribution. Quantitative metric
    error needs the dataset's 3D reconstruction as ground truth (a deeper v0.2 integration);
    this pass exercises the real-data code path end to end so it cannot silently rot.
    """
    root = os.environ.get("TRACKPHYSICS_LATTE_MV")
    if not root:
        return None
    try:
        tracks = load_latte_mv(root)
    except DatasetUnavailable as exc:
        return {"status": "unavailable", "detail": str(exc)}
    n_metric = sum(
        tp.analyze(t, preset="sphere", grounding=tp.GroundingContext()).trajectory.tier
        == tp.Tier.METRIC
        for t in tracks
    )
    return {"status": "ran", "n_tracks": len(tracks), "n_metric_emitted": int(n_metric)}


def main(*, smoke: bool = False) -> None:
    if smoke:
        # Fast end-to-end sanity for CI (BRIEF.md §15 "tiny benchmark smoke run"): a few
        # seeds / levels, no plots, and does NOT overwrite the published report.
        fp_curve = degradation_sweep("fp_rate", [0.0, 0.3], n_seeds=3)
        gate = run_gate(n_runs=12)
        _rel, clean_err = run_calibration(n_runs=18)
        assert all(np.isfinite(v) for v in fp_curve.robust_err), "smoke: non-finite error"
        print(f"bench smoke OK — clean_err={clean_err:.3f} m/s, gate F1={gate['f1']:.2f}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running synthetic benchmark (this is the published accuracy/robustness envelope)...")

    fp_curve = degradation_sweep("fp_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    jitter_curve = degradation_sweep("jitter_sigma_px", [0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    drag_curve = degradation_sweep("drag_coeff", [0.0, 0.1, 0.2, 0.47, 1.0])
    gate = run_gate()
    rel, clean_err = run_calibration()
    real = run_real_data()
    centers, acc, counts, ece = rel

    _plot_curve(fp_curve, "Degradation under false positives / outliers", "false-positive rate",
                REPORT_DIR / "degradation_false_positives.png")
    _plot_curve(jitter_curve, "Degradation under correlated jitter", "jitter sigma (px)",
                REPORT_DIR / "degradation_jitter.png")
    _plot_curve(drag_curve, "Model mismatch: pure-quadratic fit vs true drag", "drag coefficient",
                REPORT_DIR / "model_mismatch_drag.png")
    _plot_reliability(rel, REPORT_DIR / "reliability.png")

    report = {
        "clean_launch_speed_error_m_s": clean_err,
        "headline_arc_drag_coeff": 0.2,
        "degradation_false_positives": fp_curve.__dict__,
        "degradation_jitter": jitter_curve.__dict__,
        "model_mismatch_drag": drag_curve.__dict__,
        "calibration": {"ece": float(ece), "bin_centers": centers.tolist(),
                        "empirical_acc": [None if not np.isfinite(a) else float(a) for a in acc],
                        "counts": counts.tolist()},
        "metric_gate": gate,
        "real_data": real,
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    md = _render_markdown(clean_err, fp_curve, jitter_curve, drag_curve, gate, float(ece), real)
    (REPORT_DIR / "report.md").write_text(md)
    print(md)
    print(f"\nWrote plots + report.json + report.md to {REPORT_DIR}")


def _fmt(xs: list[float]) -> str:
    return "  ".join(f"{x:5.2f}" if np.isfinite(x) else "  nan" for x in xs)


def _real_data_line(real: dict[str, object] | None) -> str:
    if real is None:
        return (
            "> **SYNTHETIC-ONLY RUN.** No real footage was evaluated; the sim2real gap is "
            "**UNMEASURED**. Set `TRACKPHYSICS_LATTE_MV` to a prepared LATTE-MV directory to "
            "add a real-data pass (BRIEF.md §14.3 / §19)."
        )
    if real.get("status") == "ran":
        return (
            f"> Real-data (LATTE-MV) pass: ran `analyze` on {real['n_tracks']} tracks; "
            f"{real['n_metric_emitted']} emitted METRIC. (Quantitative sim2real error vs the "
            "dataset's 3D reconstruction is a v0.2 integration.)"
        )
    return f"> Real-data pass requested but unavailable: {real.get('detail', '')}"


def _render_markdown(
    clean_err: float,
    fp: Curve,
    jitter: Curve,
    drag: Curve,
    gate: dict[str, float],
    ece: float,
    real: dict[str, object] | None,
) -> str:
    steep_fp = gate.get("steep_false_positive_rate", float("nan"))
    steep_bias = gate.get("steep_mean_rel_bias", float("nan"))
    lines = [
        "# trackphysics v0.1 — benchmark report (synthetic)",
        "",
        "Generated by `python -m bench.run`. Exact metric ground truth from a physics-",
        "simulated, camera-projected ballistic arc; corruption is correlated (not IID).",
        "",
        _real_data_line(real),
        "",
        f"## Clean metric accuracy\n\nMean launch-speed error on uncorrupted arcs "
        f"(**realistic drag = 0.2**, a genuine model mismatch for the v0.1 pure-quadratic "
        f"fit): **{clean_err:.3f} m/s**.",
        "",
        "## Graceful degradation (the moat)",
        "",
        "Mean |launch-speed error| (m/s) on **drag-free** arcs (drag is a separate axis",
        "below): both methods use identical gravity-as-a-ruler, so the only variable is the",
        "fit — robust RANSAC+IRLS vs naive polyfit. Drag-free isolates the fitting-robustness",
        "effect; mixing drag in would mask it under the common-mode model bias.",
        "",
        "**False-positive / outlier rate**",
        "```",
        f"level     {_fmt(fp.levels)}",
        f"baseline  {_fmt(fp.baseline_err)}",
        f"robust    {_fmt(fp.robust_err)}",
        "```",
        "",
        "**Correlated jitter (sigma px)** — robust ≈ baseline here, as expected: RANSAC/IRLS",
        "defends against outliers, not zero-mean Gaussian jitter.",
        "```",
        f"level     {_fmt(jitter.levels)}",
        f"baseline  {_fmt(jitter.baseline_err)}",
        f"robust    {_fmt(jitter.robust_err)}",
        "```",
        "",
        "**Model mismatch — true drag coefficient** (no corruption; both fits share the",
        "pure-quadratic model bias, so the curve is the honest model-mismatch error):",
        "```",
        f"drag      {_fmt(drag.levels)}",
        f"robust    {_fmt(drag.robust_err)}",
        "```",
        "",
        "## Provenance calibration",
        "",
        f"Expected Calibration Error (ECE) of stated confidence vs empirical correctness: "
        f"**{ece:.3f}**.",
        "",
        "## Metric-vs-fallback gate",
        "",
        "Positive = emits METRIC; ground truth positive = scale genuinely trustworthy. The",
        "negatives include a HARD case — a clean parabola from a STEEPLY pitched camera, where",
        "gravity-as-a-ruler is grossly violated (not trivially separable). Speed bias is measured",
        "at the engine's segment-start instant (the frame it reports), not frame 0.",
        "",
        f"- precision **{gate['precision']:.2f}**, recall **{gate['recall']:.2f}**, "
        f"F1 **{gate['f1']:.2f}**, accuracy **{gate['accuracy']:.2f}**",
        "- confusion: "
        f"tp={gate['tp']:.0f} fp={gate['fp']:.0f} fn={gate['fn']:.0f} tn={gate['tn']:.0f}",
        f"- **known limitation:** on steeply-pitched arcs the engine still emits METRIC "
        f"{steep_fp * 100:.0f}% of the time with a mean speed bias of {steep_bias * 100:.0f}% "
        "— it cannot detect the violated assumption monocularly (a v0.2 cross-check item).",
        "",
        "![degradation: false positives](degradation_false_positives.png)",
        "![degradation: jitter](degradation_jitter.png)",
        "![model mismatch: drag](model_mismatch_drag.png)",
        "![reliability](reliability.png)",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="trackphysics benchmark runner")
    parser.add_argument(
        "--smoke", action="store_true",
        help="fast end-to-end sanity run (CI); no plots, does not write the report",
    )
    main(smoke=parser.parse_args().smoke)
