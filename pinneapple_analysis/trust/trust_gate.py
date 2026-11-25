"""pinneapple_analysis.trust.trust_gate — ``TrustGate``: pre-consumption
trust scoring of a trained model's own prediction, before it is allowed
to feed a live digital twin.

Distinct from two existing, adjacent mechanisms — this is neither of them:

- ``pinneapple_llm.guardrail.PhysicsGuardrail`` verifies a *problem spec*
  an LLM proposed (residual re-check, dimensional analysis, conservation)
  before training even starts.
- ``pinneapple_systems.digital_twin.monitoring.anomaly.MahalanobisDetector``
  scores a live sensor *innovation vector* against a running Kalman
  filter's state covariance, continuously, once a twin is already
  running.

``TrustGate`` scores one specific prediction from an already-trained
model, combining up to three independent [0, 1] sub-scores:

- an out-of-distribution (OOD) score: the fraction of query points whose
  Mahalanobis distance to the model's training-coordinate distribution has
  a chi-squared p-value (``scipy.stats.chi2.sf``) of at least ``ood_level``
  (default 0.01: the point lies inside the 99 % region of a Gaussian with
  the training mean and covariance). Each point is tested on its own -- the
  same Mahalanobis-distance +
  chi-squared concept ``MahalanobisDetector`` already uses, reimplemented
  here (rather than called directly) because it fits once from a static
  training sample instead of updating per Kalman step, a different call
  shape;
- a residual score: re-evaluating a PDE residual on the model's own
  prediction (any ``pinneapple_systems.component_modeling
  .physics_residuals``-shaped ``residual(model, coords, **kwargs) ->
  Tensor`` function, or a ``pinneapple_systems.component_library.Physics``
  instance);
- an optional ensemble-variance score, if a ``DeepEnsemble`` (or anything
  exposing ``.predict(x) -> (mean, std)``) is supplied.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2

from pinneapple_core.errors import BadValue, NotConfigured, ShapeMismatch, UnknownName


def _unwrap(out: Any) -> torch.Tensor:
    return out.y if hasattr(out, "y") else out


@dataclass
class TrustScore:
    combined: float
    ood_score: float | None = None
    residual_score: float | None = None
    ensemble_score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class TrustGate:
    def __init__(self, *, ood_weight: float = 0.4, residual_weight: float = 0.4, ensemble_weight: float = 0.2,
                 ood_level: float = 0.01):
        if not 0.0 < ood_level < 1.0:
            raise BadValue(f"ood_level must be in (0, 1), got {ood_level}",
                           fix="pass ood_level=0.01 for the 99 % region (the default), or another probability "
                               "between 0 and 1 exclusive")
        self.ood_level = float(ood_level)
        self.ood_weight = ood_weight
        self.residual_weight = residual_weight
        self.ensemble_weight = ensemble_weight
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._dim: int | None = None

    # -- fitting the OOD reference distribution --------------------------

    def fit(self, training_coords: torch.Tensor) -> TrustGate:
        """Fit the OOD reference from a real sample of training
        coordinates (mean + covariance)."""
        x = training_coords.detach().cpu().numpy().astype(np.float64)
        self._mean = x.mean(axis=0)
        cov = np.atleast_2d(np.cov(x, rowvar=False))
        self._cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-8)
        self._dim = x.shape[1]
        return self

    def fit_from_bounds(self, domain_bounds: dict[str, tuple[float, float]]) -> TrustGate:
        """Cheap approximation when no real training-coordinate sample is
        kept around: treats the domain as a uniform box and derives a
        diagonal covariance from each axis's range (``var = range**2 / 12``,
        the variance of a ``Uniform(a, b)``). Coarser than :meth:`fit` —
        use it only when the real sample isn't available."""
        bounds = list(domain_bounds.values())
        mean = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=np.float64)
        var = np.array([((hi - lo) ** 2) / 12.0 for lo, hi in bounds], dtype=np.float64)
        self._mean = mean
        self._cov_inv = np.diag(1.0 / np.clip(var, 1e-8, None))
        self._dim = len(bounds)
        return self

    @staticmethod
    def coord_bounds_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, tuple[float, float]]:
        """Reads ``checkpoint["metadata"]["domain_bounds"]`` if present —
        the common place a training pipeline in this repo records domain
        bounds. Raises ``KeyError`` (loudly, not a silent empty dict) if
        absent, since a made-up domain would make :meth:`fit_from_bounds`
        silently meaningless."""
        meta = checkpoint.get("metadata", {}) or {}
        bounds = meta.get("domain_bounds")
        if not bounds:
            raise UnknownName("the checkpoint records no domain bounds under metadata['domain_bounds']",
                              fix="re-save the checkpoint with metadata={'domain_bounds': {'x': (0, 1), ...}}, or "
                                  "build the gate from the training sample with fit(training_coords)")
        return {k: tuple(v) for k, v in bounds.items()}

    # -- sub-scores --------------------------------------------------------

    def ood_p_values(self, x: Any) -> np.ndarray:
        """Per query point: probability of a Mahalanobis distance this large or larger under the fitted
        distribution (chi-squared with ``dim`` degrees of freedom). Low => far out of distribution. NaN
        coordinates give a p-value of 0."""
        if self._mean is None or self._cov_inv is None:
            raise NotConfigured("the gate has no reference distribution yet, so it cannot score an input",
                                fix="call fit(training_coords) with a sample of the training points, or "
                                    "fit_from_bounds(domain_bounds) when only the box is known")
        xq = (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)).astype(np.float64)
        if xq.ndim == 1:                          # one point, or a batch of 1-D points
            xq = xq.reshape(-1, 1) if self._dim == 1 else xq.reshape(1, -1)
        elif xq.ndim != 2:
            raise ShapeMismatch(f"query points must have shape (n, {self._dim}); got {xq.shape}",
                                fix=f"pass a 2-D array with {self._dim} column(s), or one 1-D point of "
                                    f"{self._dim} coordinate(s)")
        if xq.shape[1] != self._dim:
            raise ShapeMismatch(f"query points have {xq.shape[1]} coordinates, the gate was fitted on {self._dim}",
                                fix=f"select the {self._dim} training coordinate(s) in the same order used for "
                                    f"fit(), or refit on {xq.shape[1]}")
        delta = xq - self._mean
        d2 = np.einsum("ni,ij,nj->n", delta, self._cov_inv, delta)
        return np.nan_to_num(chi2.sf(d2, df=self._dim), nan=0.0)

    def _ood_score(self, x: torch.Tensor) -> float:
        # Each point is tested on its own: the p-value of the batch MEAN let a batch of far-out points that
        # average to the training mean (x = -50 and x = +50 sigma) score as perfectly in-distribution.
        p = self.ood_p_values(x)
        return float(np.mean(p >= self.ood_level)) if p.size else 1.0

    def _residual_score(self, model: Any, x: torch.Tensor, residual_fn: Callable[..., torch.Tensor], residual_kwargs: dict[str, Any]) -> float:
        r = residual_fn(model, x, **residual_kwargs)
        rmse = float(torch.sqrt((r ** 2).mean()).item())
        return 1.0 / (1.0 + rmse)

    def _ensemble_score(self, ensemble: Any, x: torch.Tensor) -> float:
        mean, std = ensemble.predict(x)
        mean_t = _unwrap(mean)
        rel_std = float((std.abs() / mean_t.abs().clamp_min(1e-6)).mean().item())
        return 1.0 / (1.0 + rel_std)

    # -- combined score ------------------------------------------------

    def score(
        self,
        model: Any,
        x: torch.Tensor,
        *,
        residual_fn: Callable[..., torch.Tensor] | None = None,
        residual_kwargs: dict[str, Any] | None = None,
        ensemble: Any | None = None,
    ) -> TrustScore:
        """Combines whichever sub-scores are available into one [0, 1]
        weighted average (renormalized over only the parts actually
        computed). Raises if none of {fitted OOD, residual_fn, ensemble}
        were provided — a trust score with no evidence is not a score."""
        details: dict[str, Any] = {}
        parts, weights = [], []

        ood = None
        if self._mean is not None:
            ood = self._ood_score(x)
            pv = self.ood_p_values(x)
            details["ood_fraction_inside"] = ood
            details["ood_p_value_min"] = float(pv.min()) if pv.size else None
            details["ood_level"] = self.ood_level
            parts.append(ood)
            weights.append(self.ood_weight)

        residual = None
        if residual_fn is not None:
            residual = self._residual_score(model, x, residual_fn, residual_kwargs or {})
            details["residual_score"] = residual
            parts.append(residual)
            weights.append(self.residual_weight)

        ens = None
        if ensemble is not None:
            ens = self._ensemble_score(ensemble, x)
            details["ensemble_score"] = ens
            parts.append(ens)
            weights.append(self.ensemble_weight)

        if not parts:
            raise ValueError(
                "TrustGate.score() has no evidence to combine — call fit()/fit_from_bounds() first, "
                "and/or pass residual_fn and/or ensemble."
            )

        wsum = sum(weights)
        combined = sum(p * w for p, w in zip(parts, weights, strict=True)) / wsum
        return TrustScore(combined=combined, ood_score=ood, residual_score=residual, ensemble_score=ens, details=details)


    def authorize(
        self,
        model: Any,
        x: torch.Tensor,
        *,
        residual_fn: Callable[..., torch.Tensor],
        max_residual_rmse: float,
        residual_kwargs: dict[str, Any] | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Strict batch authorization; no weighted score can override a veto.

        Requires fitted OOD evidence and a finite physical residual. The
        residual tolerance is in the units of the supplied residual.
        Exceptions from a verifier propagate: callers must not consume
        predictions when authorization cannot complete.
        """
        if not np.isfinite(max_residual_rmse) or max_residual_rmse < 0:
            raise ValueError("max_residual_rmse must be finite and nonnegative")
        reasons = []
        if x.numel() == 0 or not bool(torch.isfinite(x).all()):
            return False, ("empty or nonfinite input",)
        if self._mean is None:
            reasons.append("OOD reference missing")
        else:
            p = self.ood_p_values(x)
            if not p.size or not np.all(p >= self.ood_level):
                reasons.append("out-of-distribution input")
        r = residual_fn(model, x, **(residual_kwargs or {}))
        if r.numel() == 0 or not bool(torch.isfinite(r).all()):
            reasons.append("empty or nonfinite physical residual")
        elif float(torch.sqrt(r.double().square().mean()).item()) > max_residual_rmse:
            reasons.append("physical residual exceeds tolerance")
        return not reasons, tuple(reasons)
