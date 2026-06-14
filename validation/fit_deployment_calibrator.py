"""Fit a per-deployment calibrator from a TT3D dump and measure its IN-DISTRIBUTION value.

Per-deployment means: fit on the deployment's own (fixed) geometry and apply there. To show
the value without a real second capture of the same rig, we use pooled random k-fold over the
dump (geometry SEEN in train) — the in-distribution regime the artifact targets. (Leave-one-
VIEW-out is the cross-geometry regime and is expected to fail; see DECISIONS.md / calibrate.py.)

Reports, per fold and averaged: |bias| before/after de-bias, the engine's current fixed-floor
CI coverage, and the calibrator's input-conditioned CI coverage (target 0.95). Optionally saves
a calibrator (fit on all rows) to ``--out`` for reuse.

Run:  python -m validation.fit_deployment_calibrator --dump <dump.csv> [--out cal.json]

NOTE: the dump is a TT3D-derived INTERNAL artifact (license); the saved calibrator's
coefficients are geometry-specific — do not commit either.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from trackphysics.calibration import CALIBRATOR_FEATURES, DeploymentCalibrator

# the dump abbreviates two columns; everything else matches the feature name 1:1
_DUMP_COL = {"residual_fraction": "resid_frac", "inlier_fraction": "inlier_frac"}


def _load(
    path: str,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, float]] = []
    rec: list[float] = []
    truth: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["tier"] != "metric" or r["rec"] in (None, "", "None"):
                continue
            if r["inplane_truth"] in (None, "", "None"):
                continue
            feats = {f: float(r[_DUMP_COL.get(f, f)]) for f in CALIBRATOR_FEATURES}
            if any(not np.isfinite(v) for v in feats.values()):
                continue
            rows.append(feats)
            rec.append(float(r["rec"]))
            truth.append(float(r["inplane_truth"]))
            ci_lo.append(float(r["ci_lo"]))
            ci_hi.append(float(r["ci_hi"]))
    return rows, np.array(rec), np.array(truth), np.array(ci_lo), np.array(ci_hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", help="optional path to save a calibrator (fit on ALL rows)")
    args = ap.parse_args()

    rows, rec, truth, ci_lo, ci_hi = _load(args.dump)
    n = len(rows)
    print(f"loaded {n} METRIC rows with finite features from {args.dump}")
    idx = np.arange(n)
    raw_b, deb_b, cov_engine, cov_cal = [], [], [], []
    for f in range(args.folds):
        te = idx % args.folds == f
        tr = ~te
        cal = DeploymentCalibrator.fit(
            [rows[i] for i in idx[tr]], rec[tr], truth[tr], provenance=f"pooled-fold{f}"
        )
        rb, db, ce, cc = [], [], 0, 0
        for i in idx[te]:
            speed, (lo, hi) = cal.apply(rows[i], float(rec[i]))
            rb.append(abs(rec[i] - truth[i]))
            db.append(abs(speed - truth[i]))
            ce += int(ci_lo[i] <= truth[i] <= ci_hi[i])
            cc += int(lo <= truth[i] <= hi)
        raw_b.append(np.mean(rb))
        deb_b.append(np.mean(db))
        cov_engine.append(ce / int(te.sum()))
        cov_cal.append(cc / int(te.sum()))
    print(f"pooled {args.folds}-fold (geometry in-distribution):")
    print(f"  |bias|        raw={np.mean(raw_b):.3f} -> de-biased={np.mean(deb_b):.3f}")
    print(f"  CI coverage   engine(fixed floor)={np.mean(cov_engine):.2f}  "
          f"calibrator(input-conditioned)={np.mean(cov_cal):.2f}   (target 0.95)")

    if args.out:
        cal_all = DeploymentCalibrator.fit(rows, rec, truth, provenance=f"fit_on_all:{args.dump}")
        with open(args.out, "w") as fh:
            json.dump(cal_all.to_dict(), fh, indent=2)
        print(f"saved calibrator (n_fit={cal_all.n_fit}) -> {args.out}  [INTERNAL — do not commit]")


if __name__ == "__main__":
    main()
