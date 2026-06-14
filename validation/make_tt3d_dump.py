"""Emit a per-trajectory feature dump from a TT3D evaluation root, for calibrate.py.

For every trajectory in every viewpoint (clean + noisy) this runs the engine and records the
RUNTIME-OBSERVABLE features a discriminative-uncertainty model can use, plus the GT label
(``depth_gt``) and the validation outcome (in-plane / full-3D error, current CI, coverage).

Observable features fall in two groups:
  * fit-quality: n, resid_frac, inlier_frac, completeness, a_px, gof
  * trajectory SHAPE / geo (scale-invariant, point-only, encode viewpoint):
      aspect       = horizontal / vertical pixel extent of the segment   (ptp(u)/ptp(v))
      straightness = chord / arc-length in (0,1]   (1 = straight, <1 = curved)
      pca_angle    = orientation of the (u,v) cloud's principal axis, folded to [0, pi)
      pca_ecc      = elongation, sqrt(1 - lambda_min/lambda_max)  (0 round .. 1 line)
  * size_prox = relative apparent bbox-size range (DEAD on TT3D: point-only obs, fixed boxes)

``depth_gt`` (median |v·optical_axis|/|v| of the TRUE 3-D velocity) is a GT oracle, NOT a
runtime feature — it is the label Stage 0 correlates the observables against.

Run (from the repo root, with the package installed):

    python -m validation.make_tt3d_dump --root <tt3d>/data/evaluation --out dump.csv

NOTE: TT3D is all-rights-reserved; the emitted CSV is an INTERNAL artifact — do not commit
the numbers. This builder (code) is fine to keep; the dump it writes is not.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from validation.datasets_tt3d import VIEWS, load_tt3d
from validation.run_3d_validation import Trajectory3D, _true_velocity, validate_trajectory

import trackphysics as tp
from trackphysics.core.schema import TrackSequence
from trackphysics.core.shape import inplane_shape_features as geo_features  # single source of truth


def apparent_size(track: TrackSequence) -> np.ndarray:
    b = np.array([d.bbox for d in track.detections], float)
    w = np.abs(b[:, 2] - b[:, 0])
    h = np.abs(b[:, 3] - b[:, 1])
    return np.asarray(np.sqrt(np.maximum(w, 1e-9) * np.maximum(h, 1e-9)), dtype=np.float64)


def gt_depth_frac(traj: Trajectory3D) -> float:
    """GT oracle: median fraction of the TRUE 3-D speed that lies along the optical axis."""
    t = traj.track.times()
    v = _true_velocity(traj.gt_positions_m, t)
    sp = np.linalg.norm(v, axis=1)
    ok = sp > 1e-6
    if not ok.any():
        return float("nan")
    ax = traj.optical_axis / np.linalg.norm(traj.optical_axis)
    return float(np.median(np.abs(v @ ax)[ok] / sp[ok]))


_COLS = [
    "cell", "name", "tier", "n", "resid_frac", "inlier_frac", "completeness", "a_px", "scale",
    "conf", "gof", "size_prox", "aspect", "straightness", "pca_angle", "pca_ecc", "depth_gt",
    "fallback", "rec", "ci_lo", "ci_hi", "inplane_truth", "full3d_truth", "inplane_err",
    "full3d_err", "ci_covers",
]


def build(root: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for view in VIEWS:
        for noisy in (False, True):
            cell = f"{view}{'' if noisy else '_clean'}"
            for traj in load_tt3d(root, view=view, noisy=noisy):
                track = traj.track
                est = tp.analyze(track, preset="sphere", grounding=tp.GroundingContext()).trajectory
                m = est.meta
                seg = est.segment
                has_idx = seg is not None and seg.indices is not None
                idx = seg.indices if has_idx else np.arange(len(track))  # type: ignore[union-attr]
                uv = track.centers()[idx]
                size = apparent_size(track)
                row: dict[str, object] = dict(
                    cell=cell, name=traj.name, tier=est.tier.value, n=len(track.detections),
                    resid_frac=m.get("residual_fraction"), inlier_frac=m.get("inlier_fraction"),
                    completeness=m.get("completeness"), a_px=m.get("a_px"),
                    scale=m.get("scale_m_per_px"), conf=float(est.velocity.confidence),
                    gof=float(est.goodness_of_fit),
                    size_prox=float(np.ptp(size) / max(np.median(size), 1e-9)),
                    depth_gt=gt_depth_frac(traj), fallback=m.get("fallback_reason"),
                    **geo_features(uv),
                )
                if est.tier is tp.Tier.METRIC and isinstance(est.velocity.frame, tuple):
                    r = validate_trajectory(traj)
                    ci = m.get("launch_speed_ci95")
                    lo, hi = (float(ci[0]), float(ci[1])) if isinstance(ci, tuple) else (None, None)
                    row.update(
                        rec=m.get("launch_speed_m_s"),
                        ci_lo=lo, ci_hi=hi,
                        inplane_truth=r.inplane_truth, full3d_truth=r.full3d_truth,
                        inplane_err=r.inplane_error, full3d_err=r.full3d_error,
                        ci_covers=r.ci_covers_inplane,
                    )
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TT3D_EVAL_ROOT"),
                    help="TT3D data/evaluation directory (or set TT3D_EVAL_ROOT)")
    ap.add_argument("--out", required=True, help="output CSV path (INTERNAL — do not commit)")
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("pass --root <tt3d>/data/evaluation (or set TT3D_EVAL_ROOT)")
    rows = build(args.root)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _COLS})
    print(f"wrote {args.out}  rows={len(rows)}")


if __name__ == "__main__":
    main()
