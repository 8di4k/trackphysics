"""Two-stage discriminative-uncertainty scaffold from the TT3D per-trajectory dump.

Built to the spec: Stage 0 (does any RUNTIME-OBSERVABLE feature proxy depth?), Stage 1
(de-bias the point estimate from observable features — a_px-led, because the signed bias is
systematic and predictable), Stage 2 (input-conditioned CI on the de-biased residual), all
under LEAVE-ONE-VIEW-OUT CV so we measure generalization across geometry, not memorization.

Observable features only: fit-quality (n, resid_frac, inlier_frac, completeness, a_px, gof)
PLUS point-only trajectory-SHAPE/geo features (aspect, straightness, pca_angle, pca_ecc) that
encode viewpoint. depth_gt is a LABEL (GT oracle) used ONLY in Stage 0. |a-g|/g is omitted
(the engine assumes g, so it is identically 0).

Direction-finder, NOT a production calibrator: ~hundreds of metric rows over 3 folds. Any
shipped calibration must be a REFITTABLE artifact, never hardcoded into the domain-agnostic
core (that is what keeps §6 intact). Coefficients are deployment/geometry-specific.

Build the dump with ``make_tt3d_dump.py`` (or any harness emitting the same columns):

    python -m validation.calibrate --dump <per_trajectory_dump.csv>
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

OBS_FEATURES = ["n", "resid_frac", "inlier_frac", "completeness", "a_px", "gof",
                "aspect", "straightness", "pca_angle", "pca_ecc"]  # + point-only shape/geo
Z95 = 1.959964


def _f(x: object) -> float:
    if x in (None, "", "None"):
        return np.nan
    return float(x)  # type: ignore[arg-type]


def load_metric_rows(path: str) -> list[dict[str, object]]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r["tier"] != "metric":
            continue
        if r["rec"] in (None, "", "None") or r["inplane_truth"] in (None, "", "None"):
            continue
        out.append(r)
    return out


def view_of(cell: str) -> str:
    return cell.replace("_clean", "")


def _design(rows: list[dict[str, object]], feats: list[str]) -> np.ndarray:
    return np.array([[_f(r[f]) for f in feats] for r in rows], dtype=np.float64)


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (mean, std, kept_idx) dropping near-constant columns (zero train variance)."""
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    kept = [i for i in range(train.shape[1]) if sd[i] > 1e-9]
    return mu, sd, kept


def _fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS with intercept; X already standardized. Returns coeffs incl. intercept (last)."""
    A = np.column_stack([X, np.ones(X.shape[0])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.asarray(beta, dtype=np.float64)


def _apply_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(np.column_stack([X, np.ones(X.shape[0])]) @ beta, dtype=np.float64)


def stage0(rows: list[dict[str, object]]) -> None:
    print("\n" + "=" * 72)
    print("STAGE 0 — does any RUNTIME-OBSERVABLE feature proxy depth_gt (the label)?")
    print("=" * 72)
    depth = np.array([_f(r["depth_gt"]) for r in rows])
    for f in OBS_FEATURES + ["size_prox"]:
        v = np.array([_f(r[f]) for r in rows])
        ok = np.isfinite(v) & np.isfinite(depth)
        if ok.sum() < 3 or np.std(v[ok]) < 1e-12:
            print(f"  corr(depth_gt, {f:12s}) =   n/a (constant/degenerate)")
            continue
        c = np.corrcoef(v[ok], depth[ok])[0, 1]
        print(f"  corr(depth_gt, {f:12s}) = {c:+.3f}")
    print("  -> if every observable corr is ~0, there is NO point-only depth signal and the")
    print("     tier-hole fix (B) strictly requires a size-bearing channel (real detector).")


def run_loocv(rows: list[dict[str, object]]) -> None:
    views = sorted({view_of(str(r["cell"])) for r in rows})
    truth = np.array([_f(r["inplane_truth"]) for r in rows])
    rec = np.array([_f(r["rec"]) for r in rows])
    bias = rec - truth
    ci_lo = np.array([_f(r["ci_lo"]) for r in rows])
    ci_hi = np.array([_f(r["ci_hi"]) for r in rows])
    rv = np.array([view_of(str(r["cell"])) for r in rows])

    print("\n" + "=" * 72)
    print(f"LEAVE-ONE-VIEW-OUT  (views: {views};  {len(rows)} metric rows total)")
    print("=" * 72)
    print(f"{'held-out':9s} {'n':>4s} | {'|bias| raw':>10s} {'|bias| deb':>10s} | "
          f"{'cov cur':>7s} {'cov fix':>7s} {'cov cond':>8s}")

    agg: dict[str, list[float]] = {"raw": [], "deb": [], "cur": [], "fix": [], "cond": []}
    for held in views:
        tr = rv != held
        te = rv == held
        Xtr_full = _design([r for r, m in zip(rows, tr, strict=True) if m], OBS_FEATURES)
        Xte_full = _design([r for r, m in zip(rows, te, strict=True) if m], OBS_FEATURES)
        ytr_bias, yte_bias = bias[tr], bias[te]

        mu, sd, kept = _standardize(Xtr_full)
        Ztr = (Xtr_full[:, kept] - mu[kept]) / sd[kept]
        Zte = (Xte_full[:, kept] - mu[kept]) / sd[kept]
        Ztr = np.nan_to_num(Ztr)
        Zte = np.nan_to_num(Zte)

        # --- Stage 1: de-bias the point estimate ---
        beta_b = _fit_linear(Ztr, ytr_bias)
        bias_hat_te = _apply_linear(Zte, beta_b)
        deb_resid_te = yte_bias - bias_hat_te  # residual of the de-biased estimate

        # --- Stage 2: input-conditioned CI on the de-biased residual ---
        # model conditional log-scale of the TRAIN de-biased residual, then calibrate k so
        # train coverage hits 95%; apply to held-out.
        bias_hat_tr = _apply_linear(Ztr, beta_b)
        deb_resid_tr = ytr_bias - bias_hat_tr
        log_abs_tr = np.log(np.abs(deb_resid_tr) + 1e-6)
        beta_s = _fit_linear(Ztr, log_abs_tr)
        sigma_tr = np.exp(_apply_linear(Ztr, beta_s))
        sigma_te = np.exp(_apply_linear(Zte, beta_s))
        k = float(np.quantile(np.abs(deb_resid_tr) / np.maximum(sigma_tr, 1e-9), 0.95))
        cond_half_te = k * sigma_te
        cov_cond = float(np.mean(np.abs(deb_resid_te) <= cond_half_te))

        # --- baselines for comparison ---
        # current engine CI (as-emitted) coverage on held-out:
        cov_cur = float(np.mean((ci_lo[te] <= truth[te]) & (truth[te] <= ci_hi[te])))
        # fixed-floor recalibrated: a single constant half-width tuned to 95% on TRAIN
        # de-biased residuals (i.e. "just widen the CI to 0.95" — the naive option).
        fixed_half = float(np.quantile(np.abs(deb_resid_tr), 0.95))
        cov_fix = float(np.mean(np.abs(deb_resid_te) <= fixed_half))

        n = int(te.sum())
        print(f"{held:9s} {n:>4d} | {np.mean(np.abs(yte_bias)):>10.3f} "
              f"{np.mean(np.abs(deb_resid_te)):>10.3f} | "
              f"{cov_cur:>7.2f} {cov_fix:>7.2f} {cov_cond:>8.2f}")
        agg["raw"].append(np.mean(np.abs(yte_bias)))
        agg["deb"].append(np.mean(np.abs(deb_resid_te)))
        agg["cur"].append(cov_cur)
        agg["fix"].append(cov_fix)
        agg["cond"].append(cov_cond)

    print("-" * 72)
    print(f"{'MEAN':9s} {'':>4s} | {np.mean(agg['raw']):>10.3f} {np.mean(agg['deb']):>10.3f} | "
          f"{np.mean(agg['cur']):>7.2f} {np.mean(agg['fix']):>7.2f} {np.mean(agg['cond']):>8.2f}")
    print("\nLegend: |bias| raw = current systematic over/under-estimate; |bias| deb = after")
    print("Stage-1 de-bias (held-out). cov cur = engine's CI today; cov fix = naive widen-to-95%")
    print("(constant width); cov cond = Stage-2 input-conditioned CI. Target coverage = 0.95.")
    print("Watch `back` held-out: bias~0 but depth-variance high -> hardest for Stage 2, and")
    print("exactly where the missing depth feature bites.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()
    rows = load_metric_rows(args.dump)
    print(f"loaded {len(rows)} METRIC rows from {args.dump}")
    # per-view metric counts
    from collections import Counter
    print("metric rows by view:", dict(Counter(view_of(str(r["cell"])) for r in rows)))
    stage0(rows)
    run_loocv(rows)


if __name__ == "__main__":
    main()
