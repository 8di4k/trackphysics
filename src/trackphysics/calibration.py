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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .core.provenance import Tier, TrajectoryEstimate
from .core.schema import FloatArray

__all__ = ["CALIBRATOR_FEATURES", "DeploymentCalibrator", "features_from_estimate"]

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
        )

    def _augmented(self, features: Mapping[str, float]) -> FloatArray:
        x = np.array([float(features[n]) for n in self.feature_names], dtype=np.float64)
        z = np.nan_to_num((x - np.asarray(self.mean)) / np.asarray(self.std))
        return np.append(z, 1.0)

    def apply(
        self, features: Mapping[str, float], recovered_speed: float
    ) -> tuple[float, tuple[float, float]]:
        """Recalibrate a single emission: returns ``(debiased_speed, (lo, hi))``.

        The CI is fully input-conditioned (it replaces the engine's fixed systematic floor):
        a half-width ``ci_k * exp(scale_model(features))`` calibrated to ``coverage`` on the
        deployment's fit set.
        """
        aug = self._augmented(features)
        bias_hat = float(aug @ np.asarray(self.bias_coef))
        debiased = float(recovered_speed) - bias_hat
        sigma = float(np.exp(aug @ np.asarray(self.scale_coef)))
        half = self.ci_k * sigma
        return debiased, (debiased - half, debiased + half)

    def apply_to(self, est: TrajectoryEstimate) -> tuple[float, tuple[float, float]]:
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
