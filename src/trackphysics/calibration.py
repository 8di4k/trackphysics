"""Per-deployment, refittable recalibration of the engine's metric launch speed + CI.

This is a calibration LAYER on top of the physics core (not part of it): a small linear model
that, fit on a *single deployment's own* labelled metric emissions (a fixed camera geometry),
(1) de-biases the recovered launch speed and (2) replaces the engine's fixed systematic CI
floor with an **input-conditioned** confidence interval — both from RUNTIME-OBSERVABLE features
that a :class:`~trackphysics.core.provenance.TrajectoryEstimate` already carries.

Why per-deployment: real-data validation showed a calibrated CI does NOT generalize across
camera geometry from these features (the signed bias flips by viewpoint), but IS recoverable
in-distribution — i.e. fit on the geometry you will apply it to. See ``DECISIONS.md``. A
:class:`DeploymentCalibrator` therefore carries its ``provenance`` and is valid only for the
rig it was fit on; applying it cross-geometry is a misuse.

It is a REFITTABLE ARTIFACT, never hardcoded into the core (§6): the coefficients are
deployment-specific. Serialize with :meth:`DeploymentCalibrator.to_dict` /
:meth:`from_dict` (plain JSON-able dicts).

Two honesty properties make it safe to ship:

* **A one-time LABELLED capture is required to fit.** ``fit`` needs the rig's metric emissions
  paired with independent ground-truth speeds; without labels there is no calibrator. That is
  the price of "reward for calibration".
* **It refuses out-of-distribution inputs.** :meth:`apply` checks the input against the
  fit-time support and, when outside it, returns the engine's speed unchanged with no CI
  (``in_support=False``) rather than a confident-but-wrong recalibration — see
  :class:`CalibrationResult`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .core.provenance import Tier, TrajectoryEstimate
from .core.schema import FloatArray

__all__ = [
    "CALIBRATOR_FEATURES",
    "CalibrationResult",
    "DeploymentCalibrator",
    "features_from_estimate",
]

CALIBRATOR_FEATURES: tuple[str, ...] = (
    "residual_fraction", "inlier_fraction", "completeness", "a_px", "gof",
    "aspect", "straightness", "pca_angle", "pca_ecc",
)
"""Runtime-observable features used for recalibration, all readable from a METRIC estimate
(``meta`` + ``goodness_of_fit``). The full-track length ``n`` is deliberately excluded — it is
not a property of the estimate and added no signal worth the coupling."""


def features_from_estimate(est: TrajectoryEstimate) -> dict[str, float] | None:
    """Extract the calibrator feature vector from a METRIC estimate, or ``None`` if it is not
    METRIC / is missing a feature (e.g. a supplied-scale estimate with no shape descriptors)."""
    if est.tier is not Tier.METRIC:
        return None
    feats: dict[str, float] = {}
    for name in CALIBRATOR_FEATURES:
        if name == "gof":
            feats[name] = float(est.goodness_of_fit)
            continue
        value = est.meta.get(name)
        if value is None:
            return None
        feats[name] = float(value)  # type: ignore[arg-type]
    return feats


def _design(z: FloatArray) -> FloatArray:
    """Standardized features with an intercept column appended."""
    return np.column_stack([z, np.ones(z.shape[0])])


@dataclass
class CalibrationResult:
    """Outcome of applying a :class:`DeploymentCalibrator` to one metric emission.

    ``in_support`` is the load-bearing field: a calibrator is honest ONLY on inputs that
    resemble its fit set (a fixed geometry). When the input falls outside that support the
    de-bias and the input-conditioned CI would themselves be over-confident — the §10 sin one
    floor up — so the calibrator REFUSES: it returns the engine's original speed unchanged and
    ``ci95=None`` (the caller keeps the engine's own CI), with ``in_support=False``.
    """

    speed_m_s: float
    """De-biased speed when ``in_support``; the engine's original recovered speed otherwise."""
    ci95: tuple[float, float] | None
    """Input-conditioned CI when ``in_support``; ``None`` otherwise (fall back to the engine CI)."""
    in_support: bool
    support_distance: float
    """Standardized (diagonal-Mahalanobis) distance of the input from the fit-set centre, for
    diagnostics / softer policies than the hard refusal."""


@dataclass
class DeploymentCalibrator:
    """A fitted per-deployment recalibration (de-bias + input-conditioned CI).

    Construct with :meth:`fit`; apply with :meth:`apply` / :meth:`apply_to`. Treat instances as
    immutable, serializable artifacts (:meth:`to_dict`).
    """

    feature_names: tuple[str, ...]
    mean: list[float]
    std: list[float]
    bias_coef: list[float]
    """Linear de-bias model over standardized features (+ intercept last)."""
    scale_coef: list[float]
    """Linear model of ``log|residual|`` over standardized features (+ intercept last)."""
    ci_k: float
    """Multiplier calibrating the conditioned half-width to ``coverage`` on the fit set."""
    coverage: float
    provenance: str
    n_fit: int
    feature_min: list[float]
    """Per-feature fit-time minimum (the support box, for the OOD check)."""
    feature_max: list[float]
    """Per-feature fit-time maximum."""
    support_radius: float
    """Fit-time standardized-distance radius (max diagonal-Mahalanobis distance over the fit
    set): an input beyond this — or outside the per-feature box — is out-of-distribution."""

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[Mapping[str, float]],
        recovered: Sequence[float] | FloatArray,
        truth: Sequence[float] | FloatArray,
        *,
        provenance: str,
        feature_names: tuple[str, ...] = CALIBRATOR_FEATURES,
        coverage: float = 0.95,
    ) -> DeploymentCalibrator:
        """Fit on one deployment's labelled metric emissions.

        ``feature_rows[i]`` are the observable features (see :data:`CALIBRATOR_FEATURES`),
        ``recovered[i]`` the engine's launch speed, ``truth[i]`` the independent ground-truth
        speed (e.g. the in-plane truth from a 3D set). ``provenance`` should name the
        deployment/geometry so misuse cross-geometry is auditable.
        """
        x = np.array(
            [[float(row[n]) for n in feature_names] for row in feature_rows], dtype=np.float64
        )
        rec = np.asarray(recovered, dtype=np.float64)
        tru = np.asarray(truth, dtype=np.float64)
        if x.shape[0] < len(feature_names) + 2:
            raise ValueError(
                f"need at least {len(feature_names) + 2} rows to fit; got {x.shape[0]}"
            )
        bias = rec - tru
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std_safe = np.where(std > 1e-9, std, 1.0)
        z = np.nan_to_num((x - mean) / std_safe)
        design = _design(z)
        bias_coef, *_ = np.linalg.lstsq(design, bias, rcond=None)
        residual = bias - design @ bias_coef
        scale_coef, *_ = np.linalg.lstsq(design, np.log(np.abs(residual) + 1e-6), rcond=None)
        sigma = np.exp(design @ scale_coef)
        ci_k = float(np.quantile(np.abs(residual) / np.maximum(sigma, 1e-9), coverage))
        # Fit-time support, for the OOD guard: the per-feature box and the radius of the
        # standardized point cloud. An input outside either is out-of-distribution.
        support_radius = float(np.max(np.sqrt(np.sum(z * z, axis=1)))) if z.shape[0] else 0.0
        return cls(
            feature_names=tuple(feature_names),
            mean=[float(v) for v in mean],
            std=[float(v) for v in std_safe],
            bias_coef=[float(v) for v in bias_coef],
            scale_coef=[float(v) for v in scale_coef],
            ci_k=ci_k,
            coverage=coverage,
            provenance=provenance,
            n_fit=int(x.shape[0]),
            feature_min=[float(v) for v in np.min(x, axis=0)],
            feature_max=[float(v) for v in np.max(x, axis=0)],
            support_radius=support_radius,
        )

    def _standardized(self, features: Mapping[str, float]) -> FloatArray:
        x = np.array([float(features[n]) for n in self.feature_names], dtype=np.float64)
        z = np.nan_to_num((x - np.asarray(self.mean)) / np.asarray(self.std))
        return np.asarray(z, dtype=np.float64)

    def support_check(
        self,
        features: Mapping[str, float],
        *,
        box_margin: float = 0.05,
        radius_margin: float = 0.10,
    ) -> tuple[bool, float]:
        """Is ``features`` within the calibrator's fit-time support? Returns ``(in, distance)``.

        Two cheap, conservative checks (bias toward declaring OOD, since falling back to the
        engine default is the safe under-claim): (1) every feature within its fit-time
        ``[min, max]`` extended by ``box_margin`` of the range, and (2) the standardized
        distance within ``support_radius`` extended by ``radius_margin``. A NaN feature is OOD.
        """
        x = np.array([float(features[n]) for n in self.feature_names], dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return False, float("inf")
        lo = np.asarray(self.feature_min)
        hi = np.asarray(self.feature_max)
        span = np.maximum(hi - lo, 1e-9)
        in_box = bool(np.all(x >= lo - box_margin * span) and np.all(x <= hi + box_margin * span))
        distance = float(np.sqrt(np.sum(self._standardized(features) ** 2)))
        in_radius = distance <= self.support_radius * (1.0 + radius_margin)
        return (in_box and in_radius), distance

    def apply(self, features: Mapping[str, float], recovered_speed: float) -> CalibrationResult:
        """Recalibrate a single emission, REFUSING out-of-distribution inputs.

        When ``features`` are within the fit-time support, returns the de-biased speed and an
        input-conditioned CI (half-width ``ci_k * exp(scale_model(features))``, calibrated to
        ``coverage`` on the deployment's fit set). When they are OUT of support, the calibrator
        would itself be over-confident, so it refuses: the original ``recovered_speed`` is
        returned unchanged with ``ci95=None`` (use the engine's own CI) and
        ``in_support=False``. This is the OOD guard — the provenance of the provenance.
        """
        in_support, distance = self.support_check(features)
        if not in_support:
            return CalibrationResult(
                speed_m_s=float(recovered_speed), ci95=None,
                in_support=False, support_distance=distance,
            )
        aug = np.append(self._standardized(features), 1.0)
        bias_hat = float(aug @ np.asarray(self.bias_coef))
        debiased = float(recovered_speed) - bias_hat
        sigma = float(np.exp(aug @ np.asarray(self.scale_coef)))
        half = self.ci_k * sigma
        return CalibrationResult(
            speed_m_s=debiased, ci95=(debiased - half, debiased + half),
            in_support=True, support_distance=distance,
        )

    def apply_to(self, est: TrajectoryEstimate) -> CalibrationResult:
        """Convenience: read features + recovered speed straight off a METRIC estimate."""
        feats = features_from_estimate(est)
        if feats is None:
            raise ValueError("estimate is not METRIC or lacks the calibrator features")
        recovered = est.meta.get("launch_speed_m_s")
        if recovered is None:
            raise ValueError("estimate has no launch_speed_m_s to recalibrate")
        return self.apply(feats, float(recovered))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable dict (round-trips through :meth:`from_dict`)."""
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DeploymentCalibrator:
        kwargs = dict(data)
        kwargs["feature_names"] = tuple(kwargs["feature_names"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]
