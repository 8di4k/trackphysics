"""Object-size-as-ruler study: when does a known object size give usable metric scale?

This is the controlled, generalizable answer to "do we need stereo, or does a detector's
apparent-size channel carry scale?" — the question a single real-detector run cannot answer
cleanly (its result confounds "is the idea viable at SNR X" with "what SNR does this
dataset/detector produce"). Here the synthetic generator emits the *exact* apparent size
(``d_px = f*D/Z``), so we can sweep the controlling quantity — apparent-size signal vs
box-size noise — and read off the breakpoints, then place a real regime on the curve
analytically.

Three studies (BRIEF.md §10, §14):

1. **Clean ruler accuracy & the cross-check's value.** Three geometries: side-on
   (gravity-as-a-ruler holds), genuine depth-domination (a *level* camera with motion along
   the optical axis — image-y still maps to world-vertical, so the bias is purely the
   constant-depth violation, the documented §10 limitation), and a steeply-tilted camera (a
   *different* violation: world-vertical no longer maps to image-y). The size ruler is
   essentially *exact* when clean (it inverts the projection); gravity-as-a-ruler is biased
   on both non-benign geometries; and the size↔gravity cross-check discrepancy is large
   exactly there — an independent size cue *flags a biased gravity scale*, whatever the cause
   (depth motion OR tilt). Flagging the bias is not recovering the depth.
2. **The SNR gate.** Sweep correlated box-size noise on the depth-domination arc; report the
   size
   ruler's per-frame scale error against the realized ``snr_abs = median(d_px)/sigma_d``.
   The breakpoint (error crossing a tolerance) is the gate threshold — valid for ANY
   object/distance, not one dataset.
PERMANENTLY SEPARATE GEOMETRY AXES. Two same-class confounds have been caught — tilt-vs-depth
(BENCH-03) and skew-vs-depth (review of this study). A pitched camera is a *tilt* axis, not a
*depth* axis. Study 1 keeps three distinct geometries — ``side_on`` / ``depth_motion`` (LEVEL
camera) / ``steep_tilt`` (pitched camera) — and the smoke asserts the depth case stays level and
the tilt case stays pitched, so the axes can never silently merge.

3. **Real-regime placement (analytic).** A small, distant object (the regime that motivated
   this — e.g. a 40 mm object at a few meters: ``d≈8–14 px``, box noise ``≈1 px`` ⇒
   ``snr_abs≈10``; apparent-size change over the depth swing ``≈1–4 px`` ⇒ ``snr_dyn≈1–4``)
   is interpolated onto the study-2 curve. Verdict: a *marginal absolute cross-check* but
   *infeasible depth recovery* — refuting "size is useless" while reinforcing that
   *recovering* the depth needs stereo.

Run:  python -m bench.size_ruler            (writes report under bench/report/)
      python -m bench.size_ruler --smoke    (fast CI sanity, no plots, no report write)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bench.perturb import apply_size_noise  # noqa: E402
from bench.synth.generator import (  # noqa: E402
    GroundTruth,
    camera_depths,
    generate_track,
    look_at_camera,
)
from trackphysics.core.ballistic import irls_quadratic  # noqa: E402
from trackphysics.core.objsize import (  # noqa: E402
    apparent_size_px,
    read_object_size,
    scale_agreement,
)
from trackphysics.core.schema import TrackSequence  # noqa: E402

GRAVITY = 9.81
REPORT_DIR = Path(__file__).resolve().parent / "report"
SCALE_ERR_TOL = 0.10  # the size ruler is "usable" while per-frame scale error stays <10%
DIAMETER_M = 0.22  # a generic medium object; only the apparent-size SNR matters, not the value

# Real small-distant regime (e.g. a 40 mm object at a few meters), from published intrinsics
# + depths (see DECISIONS.md). Used ONLY for analytic placement on the synthetic curve.
REAL_SNR_ABS = 10.0  # ~10 px object / ~1 px box noise
REAL_SNR_DYN = 2.5  # ~1–4 px apparent-size change over the depth swing / ~1 px noise


def _side_arc(rng: np.random.Generator) -> tuple[TrackSequence, GroundTruth]:
    """Side-on arc: motion across the view, depth ≈ constant — gravity-as-a-ruler holds."""
    vx = float(rng.uniform(5.0, 7.0))
    vz = float(rng.uniform(6.0, 8.0))
    cam = look_at_camera(eye=(4.0, -9.0, 1.6), target=(4.0, 0.0, 1.7))
    return generate_track(
        cam, fps=120.0, launch_position=(0.0, 0.0, 1.0), launch_velocity=(vx, 0.0, vz),
        diameter_m=DIAMETER_M, drag_coeff=0.1, duration=1.3,
    )


def _depth_arc(rng: np.random.Generator) -> tuple[TrackSequence, GroundTruth]:
    """Genuine depth-domination: a LEVEL camera with motion along the optical axis.

    The camera is unpitched (eye and target at equal height) so image-y maps to world-
    vertical — gravity-as-a-ruler's vertical assumption holds. The object moves toward the
    camera along the view axis, so depth changes ~3x across the arc: the constant-depth
    assumption is violated and gravity-as-a-ruler is biased PURELY by depth motion (the
    documented §10 limitation), with no vertical-axis mis-mapping confound. Apparent size
    tracks the depth, so an independent size cross-check flags the bias.
    """
    vy = float(rng.uniform(6.0, 8.0))  # toward the camera (along -y, the optical axis)
    vz = float(rng.uniform(7.0, 9.0))  # vertical (free-flight)
    cam = look_at_camera(eye=(0.0, -14.0, 1.6), target=(0.0, 0.0, 1.6))
    return generate_track(
        cam, fps=120.0, launch_position=(0.0, -2.0, 1.0), launch_velocity=(0.0, -vy, vz),
        diameter_m=DIAMETER_M, drag_coeff=0.1, duration=1.3,
    )


def _tilt_arc(rng: np.random.Generator) -> tuple[TrackSequence, GroundTruth]:
    """Steeply-pitched camera: a DIFFERENT gravity-as-a-ruler violation (vertical mis-map).

    Pitched ~58° below horizontal, image-y is ~0% world-vertical, so gravity-as-a-ruler is
    badly biased mostly by the broken world-vertical→image-y mapping (NOT depth motion).
    Included to show the size cross-check flags a biased gravity scale whatever the cause —
    not only the depth-domination case. (This is the BENCH-03 hard-negative surface.)
    """
    vx = float(rng.uniform(5.0, 7.0))
    vz = float(rng.uniform(6.0, 8.0))
    cam = look_at_camera(eye=(3.0, -4.0, 7.0), target=(3.0, 0.0, 0.5))
    return generate_track(
        cam, fps=120.0, launch_position=(0.0, 0.0, 1.0), launch_velocity=(vx, 0.0, vz),
        diameter_m=DIAMETER_M, drag_coeff=0.1, duration=1.3,
    )


def _image_y_vertical_fraction(gt: GroundTruth) -> float:
    """How much the camera's image-y axis aligns with world-vertical, in ``[0, 1]``.

    ``cam.R[1]`` is the image-y (down) axis in world coordinates; its world-Z component's
    magnitude is ~1 for a LEVEL camera (image-y ↔ world-vertical, gravity-as-a-ruler valid)
    and ~0 for a fully-pitched camera. Used by the smoke to keep the depth-domination axis
    (level) and the tilt axis (pitched) from silently merging.
    """
    return float(abs(gt.camera.R[1][2]))


def _gravity_scale(track: TrackSequence) -> float | None:
    """Gravity-as-a-ruler meters-per-pixel from a direct robust fit to image-y."""
    t = track.times()
    centers = track.centers()
    if t.shape[0] < 4:
        return None
    fy = irls_quadratic(t, centers[:, 1])
    a_px = float(fy.accel)
    if not np.isfinite(a_px) or a_px <= 1e-6:
        return None
    return GRAVITY / a_px


def _true_scale_per_frame(gt: GroundTruth) -> np.ndarray:
    """True meters-per-pixel at each frame: depth / focal (the projection's own scale)."""
    depths = camera_depths(gt.positions, gt.camera)
    return depths / gt.camera.focal_px


def _size_scale_per_frame_error(track: TrackSequence, gt: GroundTruth) -> float:
    """Median per-frame relative error of the size ruler ``D/d_px`` vs the true scale."""
    apparent = apparent_size_px(track.bboxes())
    true_scale = _true_scale_per_frame(gt)
    n = min(apparent.shape[0], true_scale.shape[0])
    size_scale = DIAMETER_M / apparent[:n]
    rel = np.abs(size_scale - true_scale[:n]) / true_scale[:n]
    return float(np.median(rel))


@dataclass
class CleanRow:
    geometry: str
    size_ruler_per_frame_err: float
    gravity_scale_err_vs_truth: float
    cross_check_rel_disc: float


def study_clean(*, n_seeds: int = 24, base_seed: int = 0) -> list[CleanRow]:
    """Clean (noise-free) ruler accuracy + the size↔gravity cross-check, per geometry."""
    rows: list[CleanRow] = []
    geometries = (("side_on", _side_arc), ("depth_motion", _depth_arc), ("steep_tilt", _tilt_arc))
    for name, maker in geometries:
        size_errs: list[float] = []
        grav_errs: list[float] = []
        discs: list[float] = []
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            track, gt = maker(rng)
            true_med = float(np.median(_true_scale_per_frame(gt)))
            size_errs.append(_size_scale_per_frame_error(track, gt))
            g = _gravity_scale(track)
            if g is not None and true_med > 0:
                grav_errs.append(abs(g - true_med) / true_med)
                apparent = apparent_size_px(track.bboxes())
                size_arc = DIAMETER_M / float(np.median(apparent))
                rel_disc, _ = scale_agreement(size_arc, g)
                if np.isfinite(rel_disc):
                    discs.append(rel_disc)
        rows.append(
            CleanRow(
                geometry=name,
                size_ruler_per_frame_err=float(np.median(size_errs)),
                gravity_scale_err_vs_truth=(
                    float(np.median(grav_errs)) if grav_errs else float("nan")
                ),
                cross_check_rel_disc=float(np.median(discs)) if discs else float("nan"),
            )
        )
    return rows


@dataclass
class SnrCurve:
    snr_abs: list[float]
    scale_err: list[float]
    breakpoint_snr_abs: float


def study_snr_sweep(
    *, sigmas: list[float], n_seeds: int = 24, base_seed: int = 100
) -> SnrCurve:
    """Size-ruler per-frame scale error vs realized ``snr_abs`` under correlated size noise.

    Sweeps box half-extent noise ``sigma`` on the steep arc; for each level it records the
    REALIZED ``snr_abs`` (from the reading, not the nominal sigma) and the median per-frame
    scale error. The breakpoint is where error first crosses ``SCALE_ERR_TOL`` — the gate.
    """
    snr_pts: list[float] = []
    err_pts: list[float] = []
    for li, sigma in enumerate(sigmas):
        snrs: list[float] = []
        errs: list[float] = []
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + 1000 * li + s)
            track, gt = _depth_arc(rng)
            dirty = (
                apply_size_noise(track, sigma_px=sigma, rho=0.5, rng=rng) if sigma > 0 else track
            )
            errs.append(_size_scale_per_frame_error(dirty, gt))
            reading = read_object_size(apparent_size_px(dirty.bboxes()), DIAMETER_M)
            if reading is not None and np.isfinite(reading.snr_abs):
                snrs.append(reading.snr_abs)
        snr_pts.append(float(np.median(snrs)) if snrs else float("nan"))
        err_pts.append(float(np.median(errs)))

    # Breakpoint = the snr ABOVE which error stays under tolerance (the conservative gate).
    # Scan from the HIGH-snr end and take the LAST (lowest-snr) crossing, so a non-monotone
    # curve near the tolerance reports the snr a deployment actually needs, not an optimistic
    # earlier dip below it. Interpolate in log(snr); NaN if never crossed.
    order = np.argsort(snr_pts)  # ascending snr
    sa = np.array(snr_pts)[order]
    ea = np.array(err_pts)[order]
    bp = float("nan")
    for i in range(len(sa) - 1, 0, -1):  # high snr -> low
        e_hi, e_lo = ea[i], ea[i - 1]  # e_lo at lower snr, e_hi at higher snr
        if (e_lo - SCALE_ERR_TOL) * (e_hi - SCALE_ERR_TOL) <= 0 and e_lo != e_hi:
            frac = (SCALE_ERR_TOL - e_lo) / (e_hi - e_lo)
            bp = float(np.exp(np.log(sa[i - 1]) + frac * (np.log(sa[i]) - np.log(sa[i - 1]))))
            break
    return SnrCurve(snr_abs=snr_pts, scale_err=err_pts, breakpoint_snr_abs=bp)


def _interp_err_at_snr(curve: SnrCurve, snr: float) -> float:
    """Interpolate the study-2 scale-error curve at a given snr_abs (log-x)."""
    order = np.argsort(curve.snr_abs)
    sa = np.array(curve.snr_abs)[order]
    ea = np.array(curve.scale_err)[order]
    valid = np.isfinite(sa) & np.isfinite(ea)
    sa, ea = sa[valid], ea[valid]
    if sa.size == 0:
        return float("nan")
    return float(np.interp(np.log(snr), np.log(sa), ea))


def _plot_snr(curve: SnrCurve, real_err: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(curve.snr_abs)
    sa = np.array(curve.snr_abs)[order]
    ea = np.array(curve.scale_err)[order]
    ax.plot(sa, ea, "o-", color="#1a7", label="size ruler (synthetic sweep)")
    ax.axhline(SCALE_ERR_TOL, ls="--", color="#888",
               label=f"usable tolerance ({SCALE_ERR_TOL:.0%})")
    if np.isfinite(curve.breakpoint_snr_abs):
        ax.axvline(curve.breakpoint_snr_abs, ls=":", color="#36c",
                   label=f"gate breakpoint (snr≈{curve.breakpoint_snr_abs:.1f})")
    if np.isfinite(real_err):
        ax.plot([REAL_SNR_ABS], [real_err], "*", ms=15, color="#c44",
                label=f"real small-distant regime (snr≈{REAL_SNR_ABS:.0f}, err≈{real_err:.0%})")
    ax.set_xscale("log")
    ax.set_xlabel("realized snr_abs = median(d_px) / sigma_d")
    ax.set_ylabel("median per-frame scale error")
    ax.set_title("Object-size-as-ruler: the SNR gate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _render_markdown(
    clean: list[CleanRow], curve: SnrCurve, real_err: float
) -> str:
    depth_gate = "INFEASIBLE" if REAL_SNR_DYN < 3.0 else "feasible"
    depth = next((r for r in clean if r.geometry == "depth_motion"), None)
    tilt = next((r for r in clean if r.geometry == "steep_tilt"), None)
    side = next((r for r in clean if r.geometry == "side_on"), None)
    lines = [
        "# Object-size-as-ruler — synthetic SNR study",
        "",
        "Generated by `python -m bench.size_ruler`. The synthetic generator emits the exact",
        "apparent size `d_px = f*D/Z`, so this maps the size channel's usable envelope under",
        "controlled box-size noise — a generalizable answer to *stereo vs size*, with a real",
        "small-distant regime placed on the curve analytically (BRIEF.md §10, §14).",
        "",
        "## 1. Clean ruler accuracy + the size↔gravity cross-check",
        "",
        "| geometry | size ruler per-frame err | gravity scale err vs truth | "
        "cross-check rel-disc |",
        "|---|---|---|---|",
    ]
    for r in clean:
        lines.append(
            f"| {r.geometry} | {r.size_ruler_per_frame_err:.1%} | "
            f"{r.gravity_scale_err_vs_truth:.1%} | {r.cross_check_rel_disc:.1%} |"
        )
    if side is not None and depth is not None and tilt is not None:
        max_exact = max(r.size_ruler_per_frame_err for r in (side, depth, tilt))
        d_g, d_x = depth.gravity_scale_err_vs_truth, depth.cross_check_rel_disc
        t_g, t_x = tilt.gravity_scale_err_vs_truth, tilt.cross_check_rel_disc
        lines += [
            "",
            f"The size ruler is near-exact when clean (≤{max_exact:.1%} per-frame across all "
            "three) — it inverts the projection. Gravity-as-a-ruler is accurate side-on "
            f"({side.gravity_scale_err_vs_truth:.1%} err, {side.cross_check_rel_disc:.1%} disc) "
            "but biased when its assumptions break, and the cross-check catches both:",
            "",
            f"- **depth motion** (level camera — the documented §10 limitation, image-y still "
            f"maps to vertical): {d_g:.1%} gravity error, flagged by a {d_x:.1%} cross-check disc;",
            f"- **camera tilt** (a *different* violation — vertical mis-map): {t_g:.1%} error, "
            f"flagged by {t_x:.1%}.",
            "",
            "So an independent size cue **flags a biased gravity scale whatever the cause** — "
            "depth OR tilt. (Flagging the bias is not recovering the depth.)",
        ]
    lines += [
        "",
        "## 2. The SNR gate (absolute scale)",
        "",
        "Size-ruler median per-frame scale error vs realized `snr_abs = median(d_px)/sigma_d`",
        "(depth-motion arc, correlated box-size noise). The gate is ~invariant to object",
        "size/distance — it depends on snr_abs, since rel scale error ≈ 1/snr_abs by",
        "construction (verified across object sizes; see DECISIONS.md):",
        "```",
        "snr_abs   " + "  ".join(f"{s:6.1f}" for s in curve.snr_abs),
        "err       " + "  ".join(f"{e:6.1%}" for e in curve.scale_err),
        "```",
        f"Breakpoint (error crosses {SCALE_ERR_TOL:.0%}): **snr_abs ≈ "
        f"{curve.breakpoint_snr_abs:.1f}**." if np.isfinite(curve.breakpoint_snr_abs)
        else "Breakpoint: not crossed within the swept range.",
        "",
        "## 3. Real small-distant regime (analytic placement)",
        "",
        f"At the real regime's `snr_abs ≈ {REAL_SNR_ABS:.0f}` the curve gives a size-scale error",
        f"of **≈ {real_err:.0%}** — a *marginal absolute cross-check*, not a tight ruler.",
        "",
        "The separate **depth-recovery** use is judged here **by threshold, not measured**: it",
        "needs `snr_dyn ≥ 3` (`SizeRulerThresholds.min_snr_dyn`, a refittable default — there",
        "is no depth-recovery sweep here; that recovery path is unimplemented v0.2 work).",
        f"The real regime's `snr_dyn ≈ {REAL_SNR_DYN:.1f}` is below it ⇒ depth recovery from the",
        f"size channel is **{depth_gate}**.",
        "",
        "> **Verdict (region-scoped — usefulness is a function of snr_abs, not universal).**",
        "> *Below* the gate (small/distant: e.g. ~40 mm at a few meters, snr_abs≈10) the size",
        "> channel is only a **marginal absolute-scale cross-check** and depth recovery is",
        "> infeasible — but this is the regime where size *must* fail, not a general indictment.",
        "> *Above* the gate (large/close: basketball, thrown box, robot payload — diameter",
        "> tens–hundreds of px, snr_abs ≫ gate) size becomes a **STRONG absolute-scale cue** that",
        "> competes with gravity-as-a-ruler and partially substitutes for stereo *for scale*",
        "> (depth-*velocity* recovery is hard even there). That high-SNR regime is NOT reachable",
        "> on a small/distant dataset and is **unvalidated on real data** — the one real-detector",
        "> run worth doing is confirming these (synthetic-calibrated) thresholds on a large/close",
        "> object, not another small-object pass. Recovering the missing depth still needs stereo.",
        "> Numbers for a specific dataset belong in the internal validation artifact (license).",
        "",
        "![size ruler SNR gate](size_ruler.png)",
        "",
    ]
    return "\n".join(lines)


def main(*, smoke: bool = False) -> None:
    if smoke:
        clean = study_clean(n_seeds=3)
        curve = study_snr_sweep(sigmas=[0.5, 8.0], n_seeds=3)  # spans the ~10% breakpoint
        by = {r.geometry: r for r in clean}
        assert all(np.isfinite(r.size_ruler_per_frame_err) for r in clean), "smoke: nan size err"
        # The size ruler is near-exact when clean across every geometry.
        assert max(r.size_ruler_per_frame_err for r in clean) < 0.05, "smoke: ruler not ~exact"
        # The genuine depth-motion arc (level camera) must bias gravity purely by depth, and
        # the cross-check must flag it — that is the §10 demonstration.
        assert by["depth_motion"].gravity_scale_err_vs_truth > 0.10, "smoke: depth arc unbiased"
        assert by["depth_motion"].cross_check_rel_disc > 0.10, "smoke: cross-check missed depth"
        # Lock the in-plane/depth/tilt axis separation: the depth demonstrator MUST stay a
        # level camera (image-y ~ world-vertical) and the tilt case MUST stay pitched, so the
        # skew-vs-depth confound the review caught can never silently return.
        rng0 = np.random.default_rng(0)
        depth_vert = _image_y_vertical_fraction(_depth_arc(rng0)[1])
        tilt_vert = _image_y_vertical_fraction(_tilt_arc(rng0)[1])
        assert depth_vert > 0.95, f"smoke: depth arc not level (image-y vertical {depth_vert:.2f})"
        assert tilt_vert < 0.7, f"smoke: tilt arc not pitched (image-y vertical {tilt_vert:.2f})"
        print(
            "size_ruler smoke OK — "
            f"side_size_err={by['side_on'].size_ruler_per_frame_err:.3f}, "
            f"depth_gravity_err={by['depth_motion'].gravity_scale_err_vs_truth:.3f}, "
            f"depth_crosscheck_disc={by['depth_motion'].cross_check_rel_disc:.3f}, "
            f"breakpoint_snr={curve.breakpoint_snr_abs:.1f}"
        )
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running object-size-as-ruler SNR study...")
    clean = study_clean()
    curve = study_snr_sweep(sigmas=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    real_err = _interp_err_at_snr(curve, REAL_SNR_ABS)

    _plot_snr(curve, real_err, REPORT_DIR / "size_ruler.png")
    report = {
        "clean": [asdict(r) for r in clean],
        "snr_sweep": asdict(curve),
        "real_regime": {
            "snr_abs": REAL_SNR_ABS,
            "snr_dyn": REAL_SNR_DYN,
            "interpolated_scale_err": real_err,
            "depth_recovery_feasible": bool(REAL_SNR_DYN >= 3.0),
        },
        "scale_err_tol": SCALE_ERR_TOL,
    }
    (REPORT_DIR / "size_ruler.json").write_text(json.dumps(report, indent=2))
    md = _render_markdown(clean, curve, real_err)
    (REPORT_DIR / "size_ruler.md").write_text(md)
    print(md)
    print(f"\nWrote size_ruler.{{png,json,md}} to {REPORT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="object-size-as-ruler SNR study")
    parser.add_argument("--smoke", action="store_true", help="fast CI sanity; no plots/report")
    main(smoke=parser.parse_args().smoke)
