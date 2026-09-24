"""Hybrid model: a data-driven surrogate for intermediate quantities + an explicit physics post-model.

When the quantity of interest is *derived* from the flow by an established physics model (erosion from
particle impacts, fatigue from loads, heat-transfer correlations from wall quantities, ...), learning the
derived quantity end to end hides the physics model inside the regressor. The alternative implemented
here keeps it explicit:

    params --(surrogate, learned)--> intermediate quantities --(post-model, physics)--> derived quantity

* the surrogate learns only the intermediate quantities (e.g. impact-velocity/angle statistics at a
  wall), from simulation data;
* the physics post-model is a plain function of those intermediates and the parameters, with a name and a
  scientific source. It is never trained, so it stays auditable and can be swapped (another erosion
  correlation, other material constants) **without retraining** (:meth:`HybridSurrogatePhysics.with_post_model`);
* when the surrogate gives a predictive distribution (mean/std, or samples), the uncertainty is pushed
  through the post-model by Monte Carlo. The post-model is usually nonlinear, so the derived quantity
  evaluated at the surrogate mean is *not* the mean of the derived quantity; both are reported.

This module is numpy-only. :class:`GaussianProcessSurrogate` uses scikit-learn's
``GaussianProcessRegressor`` when it is installed (the same choice as
``pinneapple_design.design_optimizer.BayesianDesignOptimizer``) and a small numpy GP otherwise. No torch
is needed by this module (importing it through ``pinneapple_neural`` still runs that package's
``__init__``, which imports torch, as the rest of the library does).

Honesty scope: the Monte Carlo band only carries the surrogate's own predictive uncertainty (as the
surrogate reports it: independent Gaussian per output unless the surrogate provides a sampler) through
the post-model. Post-model (physics) uncertainty and the surrogate's calibration are not included; check
the coverage on held-out data before trusting the band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
Sampler = Callable[[int, np.random.Generator], np.ndarray]

__all__ = [
    "PhysicsPostModel",
    "CallablePostModel",
    "Surrogate",
    "SurrogatePrediction",
    "GaussianProcessSurrogate",
    "PCASurrogate",
    "PerQuantitySurrogate",
    "TransformedSurrogate",
    "HybridPrediction",
    "HybridSurrogatePhysics",
]


# --------------------------------------------------------------------------- physics post-model

@runtime_checkable
class PhysicsPostModel(Protocol):
    """An explicit physics model applied to the surrogate's intermediate quantities.

    ``__call__(intermediates, params)``:

    * ``intermediates``: ``{name: array of shape (batch, *shape_of_that_quantity)}``;
    * ``params``: ``{param_name: array of shape (batch,)}``;
    * returns the derived quantity, shape ``(batch,)`` or ``(batch, k)``.

    It must be vectorized over the batch axis (the Monte Carlo propagation calls it once on
    ``n_samples * n_points`` rows). ``name`` identifies the model; ``source`` is the scientific
    reference of the model and of its constants (a standard, a paper), which is carried into every
    prediction.
    """

    name: str
    source: str

    def __call__(self, intermediates: Mapping[str, np.ndarray], params: Mapping[str, np.ndarray]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class CallablePostModel:
    """Wrap a plain function as a :class:`PhysicsPostModel`."""

    fn: Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], np.ndarray]
    name: str
    source: str

    def __post_init__(self):
        if not self.name or not self.source:
            raise ValueError("a physics post-model needs a name and a source (the reference of the model)")

    def __call__(self, intermediates, params):
        return self.fn(intermediates, params)


# --------------------------------------------------------------------------- surrogates

@dataclass
class SurrogatePrediction:
    """Predictive distribution of a surrogate over ``m`` flat outputs at ``n`` points.

    ``mean``: (n, m). ``std``: (n, m) or None (point prediction only). ``sampler``: optional
    ``(n_samples, rng) -> (n_samples, n, m)``; when absent and ``std`` is given, samples are
    independent Gaussians per output.
    """

    mean: np.ndarray
    std: Optional[np.ndarray] = None
    sampler: Optional[Sampler] = None

    @property
    def has_distribution(self) -> bool:
        return self.sampler is not None or self.std is not None

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.sampler is not None:
            return np.asarray(self.sampler(n_samples, rng), dtype=float)
        if self.std is None:
            raise ValueError("this prediction has no predictive distribution (std/sampler) to sample from")
        eps = rng.standard_normal((n_samples,) + self.mean.shape)
        return self.mean[None] + eps * self.std[None]


class Surrogate(Protocol):
    """Minimal surrogate interface: ``fit(X, Y)`` with X (n, d), Y (n, m); ``predict(X)`` returns a
    :class:`SurrogatePrediction`, a ``(mean, std)`` tuple (std may be None) or a mean array."""

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Surrogate":
        ...

    def predict(self, X: np.ndarray):
        ...


def _as_prediction(out, n: int) -> SurrogatePrediction:
    if isinstance(out, SurrogatePrediction):
        p = out
    elif isinstance(out, tuple) and len(out) == 2:
        mean, std = out
        p = SurrogatePrediction(np.asarray(mean, dtype=float), None if std is None else np.asarray(std, dtype=float))
    else:
        p = SurrogatePrediction(np.asarray(out, dtype=float))
    if p.mean.ndim == 1:
        p = SurrogatePrediction(p.mean[:, None], None if p.std is None else p.std.reshape(-1, 1), p.sampler)
    if p.mean.shape[0] != n:
        raise ValueError(f"surrogate returned {p.mean.shape[0]} rows for {n} points")
    if p.std is not None and p.std.shape != p.mean.shape:
        raise ValueError(f"surrogate std shape {p.std.shape} != mean shape {p.mean.shape}")
    return p


class _NumpyGP:
    """Squared-exponential GP with one (isotropic) length scale and a nugget chosen by marginal likelihood
    on a small grid. Inputs are expected to be standardized.

    Fallback when scikit-learn is not installed. Targets are standardized per output.
    """

    def __init__(self, noise: float = 1e-6):
        self.noise = noise

    @staticmethod
    def _k(A, B, ls):
        d = (A[:, None, :] - B[None, :, :]) / ls
        return np.exp(-0.5 * np.sum(d ** 2, axis=-1))

    def fit(self, X, y):
        n, d = X.shape
        self.X = X
        self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0
        z = (y - self.mu) / self.sd
        best = (-math.inf, None)
        for ls in (0.3, 0.6, 1.0, 2.0, 4.0):
            for nz in (1e-6, 1e-3, 1e-2, 1e-1):
                K = self._k(X, X, np.full(d, ls)) + nz * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, z))
                ll = -0.5 * z @ a - np.log(np.diag(L)).sum()
                if ll > best[0]:
                    best = (ll, (ls, nz, L, a))
        if best[1] is None:
            raise np.linalg.LinAlgError("numpy GP: no stable kernel on the grid")
        ls, self.noise, self.L, self.alpha = best[1]
        self.ls = np.full(d, ls)
        return self

    def predict(self, Xs):
        Ks = self._k(Xs, self.X, self.ls)
        mean = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(1.0 + self.noise - np.sum(v ** 2, axis=0), 0.0)
        return mean * self.sd + self.mu, np.sqrt(var) * self.sd


class GaussianProcessSurrogate:
    """One independent Gaussian process per output column.

    With scikit-learn: ``GaussianProcessRegressor`` with ``C * RBF(ARD) + White`` and ``normalize_y``;
    without it, :class:`_NumpyGP`. ``backend`` records which one ran.
    """

    def __init__(self, *, n_restarts: int = 2, random_state: int = 0, use_sklearn: Optional[bool] = None):
        self.n_restarts, self.random_state = n_restarts, random_state
        if use_sklearn is None:
            try:
                import sklearn  # noqa: F401
                use_sklearn = True
            except ImportError:
                use_sklearn = False
        self.backend = "sklearn" if use_sklearn else "numpy"
        self.models: List[object] = []

    def fit(self, X, Y):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        Y = Y[:, None] if Y.ndim == 1 else Y
        self.models = []
        for j in range(Y.shape[1]):
            if self.backend == "sklearn":
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

                kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3))
                          + WhiteKernel(1e-2, (1e-8, 1e0)))
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                              random_state=self.random_state)
                gp.fit(X, Y[:, j])
            else:
                gp = _NumpyGP().fit(X, Y[:, j])
            self.models.append(gp)
        return self

    def predict(self, X) -> SurrogatePrediction:
        if not self.models:
            raise RuntimeError("GaussianProcessSurrogate.predict called before fit")
        X = np.asarray(X, dtype=float)
        mus, sds = [], []
        for gp in self.models:
            mu, sd = gp.predict(X, return_std=True) if self.backend == "sklearn" else gp.predict(X)
            mus.append(mu)
            sds.append(sd)
        return SurrogatePrediction(np.stack(mus, axis=1), np.stack(sds, axis=1))


class PCASurrogate:
    """Reduce the outputs with PCA, fit ``base`` on the scores, map predictions back.

    ``n_components`` fixes the number of modes; otherwise the smallest number reaching
    ``variance`` of the training variance is kept. The predictive distribution of the scores (from
    ``base``) is mapped back by sampling; the discarded modes add no uncertainty (the truncation error
    is reported in ``explained_variance``, not hidden).
    """

    def __init__(self, base: Surrogate, *, n_components: Optional[int] = None, variance: Optional[float] = None):
        if (n_components is None) == (variance is None):
            raise ValueError("give exactly one of n_components or variance")
        if variance is not None and not 0 < variance <= 1:
            raise ValueError("variance must be in (0, 1]")
        self.base, self.n_components_req, self.variance_req = base, n_components, variance

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        self.mean_ = Y.mean(axis=0)
        U, S, Vt = np.linalg.svd(Y - self.mean_, full_matrices=False)
        var = S ** 2
        frac = np.cumsum(var) / var.sum() if var.sum() > 0 else np.ones_like(var)
        if self.n_components_req is not None:
            k = min(self.n_components_req, len(S))
        else:
            k = int(np.searchsorted(frac, self.variance_req - 1e-12) + 1)
        self.n_components = max(k, 1)
        self.components_ = Vt[: self.n_components]
        self.explained_variance = float(frac[self.n_components - 1]) if len(frac) else 1.0
        self.base.fit(np.asarray(X, dtype=float), (Y - self.mean_) @ self.components_.T)
        return self

    def predict(self, X) -> SurrogatePrediction:
        X = np.asarray(X, dtype=float)
        z = _as_prediction(self.base.predict(X), len(X))
        mean = z.mean @ self.components_ + self.mean_
        sampler = None
        if z.has_distribution:
            W, m0 = self.components_, self.mean_
            sampler = lambda s, rng: z.sample(s, rng) @ W + m0  # noqa: E731
        return SurrogatePrediction(mean, None, sampler)


class TransformedSurrogate:
    """Fit ``base`` on ``forward(Y)``; predictions are mapped back with ``inverse``.

    The returned mean is ``inverse(mean in the transformed space)`` (a point estimate, e.g. the
    median for a log transform); the distribution is mapped back sample by sample.
    """

    def __init__(self, base: Surrogate, forward: Callable[[np.ndarray], np.ndarray],
                 inverse: Callable[[np.ndarray], np.ndarray], name: str = "transform"):
        self.base, self.forward, self.inverse, self.name = base, forward, inverse, name

    def fit(self, X, Y):
        self.base.fit(np.asarray(X, dtype=float), self.forward(np.asarray(Y, dtype=float)))
        return self

    def predict(self, X) -> SurrogatePrediction:
        X = np.asarray(X, dtype=float)
        z = _as_prediction(self.base.predict(X), len(X))
        mean = self.inverse(z.mean)
        sampler = None
        if z.has_distribution:
            inv = self.inverse
            sampler = lambda s, rng: inv(z.sample(s, rng))  # noqa: E731
        return SurrogatePrediction(mean, None, sampler)


class PerQuantitySurrogate:
    """One surrogate per block of output columns (e.g. one per intermediate quantity).

    ``blocks``: ``[(n_columns, surrogate), ...]`` in column order. Predictions are concatenated; the
    blocks are sampled independently (no cross-block correlation is modelled).
    """

    def __init__(self, blocks: Sequence[Tuple[int, Surrogate]]):
        self.blocks = list(blocks)

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        if sum(w for w, _ in self.blocks) != Y.shape[1]:
            raise ValueError(f"blocks cover {sum(w for w, _ in self.blocks)} columns, Y has {Y.shape[1]}")
        j = 0
        for w, s in self.blocks:
            s.fit(X, Y[:, j:j + w])
            j += w
        return self

    def predict(self, X) -> SurrogatePrediction:
        X = np.asarray(X, dtype=float)
        preds = [_as_prediction(s.predict(X), len(X)) for _, s in self.blocks]
        mean = np.concatenate([p.mean for p in preds], axis=1)
        if not all(p.has_distribution for p in preds):
            return SurrogatePrediction(mean)
        sampler = lambda n, rng: np.concatenate([p.sample(n, rng) for p in preds], axis=2)  # noqa: E731
        return SurrogatePrediction(mean, None, sampler)


# --------------------------------------------------------------------------- hybrid

@dataclass
class HybridPrediction:
    """Output of :meth:`HybridSurrogatePhysics.predict`.

    * ``intermediates``: surrogate point prediction per quantity, shape (n, *shape).
    * ``derived_at_mean``: post-model applied to ``intermediates`` (plug-in).
    * ``derived_mean`` / ``derived_std`` / ``derived_lower`` / ``derived_upper``: Monte Carlo statistics
      of the derived quantity (``interval`` central band), or None when ``uncertainty == "none"``
      (the surrogate gave no predictive distribution, or ``n_samples == 0``).
    """

    intermediates: Dict[str, np.ndarray]
    derived_at_mean: np.ndarray
    derived_mean: Optional[np.ndarray]
    derived_std: Optional[np.ndarray]
    derived_lower: Optional[np.ndarray]
    derived_upper: Optional[np.ndarray]
    interval: float
    n_samples: int
    uncertainty: str
    post_model: Dict[str, str]
    derived_samples: Optional[np.ndarray] = None


class HybridSurrogatePhysics:
    """Surrogate of intermediate quantities composed with an explicit :class:`PhysicsPostModel`.

    >>> hyb = HybridSurrogatePhysics(GaussianProcessSurrogate(), post_model)
    >>> hyb.fit(params, {"velocity": V})                 # trains the surrogate only
    >>> out = hyb.predict(params_new, n_samples=2000)     # intermediates + derived, MC band
    >>> hyb2 = hyb.with_post_model(other_model)           # same fitted surrogate, other physics

    ``surrogate`` is either one surrogate for all intermediates (flattened and concatenated in the order
    of the dict given to :meth:`fit`) or a mapping ``{quantity: surrogate}`` with one surrogate per
    intermediate quantity (e.g. PCA for a histogram, a log-GP for a positive scalar).
    """

    def __init__(self, surrogate: Union[Surrogate, Mapping[str, Surrogate]], post_model: PhysicsPostModel, *,
                 param_names: Optional[Sequence[str]] = None):
        _check_post_model(post_model)
        self.surrogates = dict(surrogate) if isinstance(surrogate, Mapping) else None
        self.surrogate, self.post_model = (None if self.surrogates is not None else surrogate), post_model
        self.param_names = list(param_names) if param_names is not None else None
        self.layout: Optional[List[Tuple[str, Tuple[int, ...]]]] = None
        self.fitted = False
        self.n_train = 0

    # --- data layout -------------------------------------------------------------------------

    def _params(self, params) -> Tuple[np.ndarray, List[str]]:
        if isinstance(params, Mapping):
            names = self.param_names or sorted(params)
            missing = [k for k in names if k not in params]
            if missing:
                raise KeyError(f"missing parameters {missing}")
            X = np.stack([np.asarray(params[k], dtype=float).reshape(-1) for k in names], axis=1)
        else:
            X = np.asarray(params, dtype=float)
            X = X[:, None] if X.ndim == 1 else X
            names = self.param_names or [f"x{j}" for j in range(X.shape[1])]
            if len(names) != X.shape[1]:
                raise ValueError(f"{X.shape[1]} parameter columns but param_names has {len(names)}")
        return X, names

    def _flatten(self, intermediates: Mapping[str, ArrayLike], n: int) -> np.ndarray:
        cols = []
        for name, shape in self.layout:
            a = np.asarray(intermediates[name], dtype=float)
            if a.shape != (n,) + shape:
                raise ValueError(f"intermediate {name!r}: shape {a.shape}, expected {(n,) + shape}")
            cols.append(a.reshape(n, -1))
        return np.concatenate(cols, axis=1)

    def _unflatten(self, M: np.ndarray) -> Dict[str, np.ndarray]:
        lead = M.shape[:-1]
        out, j = {}, 0
        for name, shape in self.layout:
            size = int(np.prod(shape)) if shape else 1
            out[name] = M[..., j:j + size].reshape(lead + shape)
            j += size
        return out

    # --- API ---------------------------------------------------------------------------------

    def fit(self, params, intermediates: Mapping[str, ArrayLike]) -> "HybridSurrogatePhysics":
        """Train the surrogate ``params -> intermediates``. The post-model is not involved."""
        if not intermediates:
            raise ValueError("no intermediate quantities to learn")
        X, names = self._params(params)
        self.param_names = names
        n = X.shape[0]
        self.layout = [(k, tuple(np.asarray(v).shape[1:])) for k, v in intermediates.items()]
        if self.surrogates is not None:
            missing = sorted(set(intermediates) ^ set(self.surrogates))
            if missing:
                raise KeyError(f"one surrogate per intermediate quantity: mismatch on {missing}")
            self.surrogate = PerQuantitySurrogate(
                [(int(np.prod(shape)) if shape else 1, self.surrogates[k]) for k, shape in self.layout])
        Y = self._flatten(intermediates, n)
        if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(X)):
            raise ValueError("non-finite values in the training data")
        self.surrogate.fit(X, Y)
        self.fitted, self.n_train = True, n
        return self

    def with_post_model(self, post_model: PhysicsPostModel) -> "HybridSurrogatePhysics":
        """A hybrid with the same (already fitted) surrogate and another post-model. Nothing is retrained."""
        _check_post_model(post_model)
        other = HybridSurrogatePhysics(self.surrogate, post_model, param_names=self.param_names)
        other.surrogates = self.surrogates
        other.layout, other.fitted, other.n_train = self.layout, self.fitted, self.n_train
        return other

    def state(self) -> Dict[str, object]:
        """The fitted surrogate side only (picklable if the surrogates are). The post-model is left out on
        purpose: it is supplied again at load time (:meth:`from_state`), possibly a different one."""
        if not self.fitted:
            raise RuntimeError("nothing to save before fit")
        return {"surrogate": self.surrogate, "surrogates": self.surrogates, "layout": self.layout,
                "param_names": self.param_names, "n_train": self.n_train}

    @classmethod
    def from_state(cls, state: Mapping[str, object], post_model: PhysicsPostModel) -> "HybridSurrogatePhysics":
        """Rebuild a fitted hybrid from :meth:`state` with the given post-model (no retraining)."""
        hyb = cls(state["surrogate"], post_model, param_names=state["param_names"])
        hyb.surrogates = state["surrogates"]
        hyb.layout, hyb.n_train, hyb.fitted = [(k, tuple(sh)) for k, sh in state["layout"]], state["n_train"], True
        return hyb

    def predict_intermediates(self, params) -> Tuple[Dict[str, np.ndarray], SurrogatePrediction, Dict[str, np.ndarray]]:
        if not self.fitted:
            raise RuntimeError("HybridSurrogatePhysics.predict called before fit")
        X, names = self._params(params)
        pred = _as_prediction(self.surrogate.predict(X), len(X))
        width = sum(int(np.prod(s)) if s else 1 for _, s in self.layout)
        if pred.mean.shape[1] != width:
            raise ValueError(f"surrogate returned {pred.mean.shape[1]} outputs, layout needs {width}")
        return self._unflatten(pred.mean), pred, {k: X[:, j] for j, k in enumerate(names)}

    def predict(self, params, *, n_samples: int = 1000, interval: float = 0.95, seed: Optional[int] = 0,
                keep_samples: bool = False) -> HybridPrediction:
        """Intermediates + derived quantity; Monte Carlo band when the surrogate gives a distribution."""
        if not 0 < interval < 1:
            raise ValueError("interval must be in (0, 1)")
        inter, pred, P = self.predict_intermediates(params)
        n = len(next(iter(P.values())))
        at_mean = np.asarray(self.post_model(inter, P), dtype=float)
        info = {"name": self.post_model.name, "source": self.post_model.source}
        if n_samples <= 0 or not pred.has_distribution:
            return HybridPrediction(inter, at_mean, None, None, None, None, interval, 0, "none", info)
        rng = np.random.default_rng(seed)
        S = pred.sample(n_samples, rng)                               # (s, n, m)
        flat = self._unflatten(S.reshape(n_samples * n, -1))
        Prep = {k: np.tile(v, n_samples) for k, v in P.items()}
        D = np.asarray(self.post_model(flat, Prep), dtype=float)
        D = D.reshape((n_samples, n) + D.shape[1:])
        q = (1 - interval) / 2
        return HybridPrediction(
            inter, at_mean, D.mean(axis=0), D.std(axis=0), np.quantile(D, q, axis=0), np.quantile(D, 1 - q, axis=0),
            interval, n_samples, "monte_carlo", info, D if keep_samples else None)


def _check_post_model(pm) -> None:
    for attr in ("name", "source"):
        if not getattr(pm, attr, None):
            raise ValueError(f"post-model without {attr!r}: the physics model must say what it is and where it "
                             "comes from")
    if not callable(pm):
        raise TypeError("post-model must be callable(intermediates, params)")
