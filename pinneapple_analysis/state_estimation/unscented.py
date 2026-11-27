"""Unscented Kalman filter and particle filter: the two the module was missing (#215).

The EKF linearises ``f`` and ``h`` with a Jacobian and is wrong wherever the nonlinearity bends inside the step; the
EnKF samples the covariance with an ensemble and needs a large one to be stable. The UKF carries the covariance
through a small, deterministic set of sigma points — no Jacobian, no sampling noise — and the particle filter keeps a
full non-Gaussian belief, which is the only one of the four that can represent two modes.

All of them drive the same ``f``/``h`` callables and the same ``predict`` → ``update`` → ``step`` recursion as
:mod:`pinneapple_analysis.state_estimation.kalman`, so a case can change filter without being rewritten.

    >>> import numpy as np
    >>> from pinneapple_analysis.state_estimation.unscented import UnscentedKalmanFilter
    >>> ukf = UnscentedKalmanFilter(2, 1, f=lambda x: x, h=lambda x: x[:1], Q=1e-4 * np.eye(2), R=0.01 * np.eye(1))
    >>> ukf.initialize(np.array([0.0, 1.0]), np.eye(2))
    >>> out = ukf.step(np.array([0.6]))          # doctest: +SKIP
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["ParticleFilter", "UnscentedKalmanFilter"]


def _cholesky_psd(matrix: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a symmetric matrix, with the jitter a covariance accumulated over a long run needs."""
    m = np.asarray(matrix, dtype=np.float64)
    m = 0.5 * (m + m.T)
    try:
        return np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        jitter = max(1e-12, 1e-10 * float(np.trace(m)) / max(len(m), 1))
        for _ in range(8):
            try:
                return np.linalg.cholesky(m + jitter * np.eye(len(m)))
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise ValueError("the covariance is not positive definite and adding jitter did not fix it; "
                         "check Q and P0") from None


class UnscentedKalmanFilter:
    """Unscented Kalman filter: sigma points carry mean and covariance through ``f`` and ``h``.

    Parameters
    ----------
    n_state, n_obs:
        Dimensions of the state and of the observation.
    f, h:
        State transition ``x_{k-1} -> x_k`` and observation operator ``x_k -> y_k``. Neither needs a Jacobian.
    Q, R:
        Process and observation noise covariances.
    alpha, beta, kappa:
        Spread of the sigma points (``alpha``), how much the prior is trusted for the covariance of the propagated
        mean (``beta``), and the secondary scaling (``kappa``, ``3 - n`` by default). The defaults are the ones in
        Julier and Uhlmann (1997) and Wan and van der Merwe (2000); ``alpha = 1`` is the plain unscented transform.
    """

    def __init__(self, n_state: int, n_obs: int, f, h, Q: np.ndarray | None = None, R: np.ndarray | None = None,
                 *, alpha: float = 1e-3, beta: float = 2.0, kappa: float | None = None) -> None:
        self.n_state = int(n_state)
        self.n_obs = int(n_obs)
        self.f = f
        self.h = h
        self.Q = np.eye(self.n_state) * 1e-4 if Q is None else np.asarray(Q, dtype=np.float64)
        self.R = np.eye(self.n_obs) * 1e-2 if R is None else np.asarray(R, dtype=np.float64)
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = (3.0 - self.n_state) if kappa is None else float(kappa)
        n = self.n_state
        self.lam = self.alpha ** 2 * (n + self.kappa) - n
        scale = n + self.lam
        if scale <= 0:
            raise ValueError("alpha and kappa make n + lambda non-positive; increase alpha")
        self._wm = np.full(2 * n + 1, 1.0 / (2.0 * scale))
        self._wc = np.full(2 * n + 1, 1.0 / (2.0 * scale))
        self._wm[0] = self.lam / scale
        self._wc[0] = self.lam / scale + (1.0 - self.alpha ** 2 + self.beta)
        self._scale = np.sqrt(scale)
        self.x = np.zeros(self.n_state)
        self.P = np.eye(self.n_state)

    # -- sigma points -------------------------------------------------------------------------------------
    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """``(2n + 1, n)``: the mean and the symmetric columns of ``sqrt((n + lambda) P)`` around it."""
        root = self._scale * _cholesky_psd(P)
        points = np.empty((2 * self.n_state + 1, self.n_state), dtype=np.float64)
        points[0] = x
        points[1:self.n_state + 1] = x + root.T
        points[self.n_state + 1:] = x - root.T
        return points

    def _mean_cov(self, points: np.ndarray, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        mean = self._wm @ points
        spread = points - (mean if center is None else center)
        cov = (self._wc[:, None] * spread).T @ spread
        return mean, cov

    # -- the recursion ------------------------------------------------------------------------------------
    def initialize(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        self.x = np.asarray(x0, dtype=np.float64).ravel()
        if P0 is not None:
            self.P = np.asarray(P0, dtype=np.float64)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Propagate every sigma point through ``f``; the covariance it comes back with is the unscented one."""
        points = self.sigma_points(self.x, self.P)
        propagated = np.array([np.asarray(self.f(p), dtype=np.float64).ravel() for p in points])
        x_pred, P_pred = self._mean_cov(propagated)
        self.x = x_pred
        self.P = P_pred + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Fuse observation ``y``: the gain comes from the cross-covariance of state and observation."""
        y = np.asarray(y, dtype=np.float64).ravel()
        points = self.sigma_points(self.x, self.P)
        observed = np.array([np.asarray(self.h(p), dtype=np.float64).ravel() for p in points])
        z_mean, S = self._mean_cov(observed)
        S = S + self.R
        dx = points - self.x
        dz = observed - z_mean
        cross = (self._wc[:, None] * dx).T @ dz
        K = cross @ np.linalg.solve(S, np.eye(len(y)))
        innovation = y - z_mean
        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return {"x": self.x.copy(), "P": self.P.copy(), "K": K, "innovation": innovation, "innovation_cov": S}

    def step(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Predict + update in one call, like the EKF and the EnKF."""
        self.predict()
        return self.update(y)


class ParticleFilter:
    """Bootstrap (sequential importance resampling) particle filter.

    The belief is the cloud itself, so it can be two modes wide — which none of the Kalman family can — at the cost
    of needing enough particles for the tail and a model whose noise can be sampled.

    Parameters
    ----------
    n_particles:
        Size of the cloud. Doubling it halves the Monte Carlo error of a mean in the usual square-root way.
    f, h:
        State transition and observation operator, as in the other filters.
    Q, R:
        Process and observation noise covariances; both are used as Gaussian proposals.
    ess_fraction:
        Resample when the effective sample size falls below this fraction of the cloud. Resampling every step throws
        away information; never resampling lets a handful of particles carry the whole belief.
    """

    def __init__(self, n_state: int, n_obs: int, f, h, Q: np.ndarray | None = None, R: np.ndarray | None = None,
                 *, n_particles: int = 2000, ess_fraction: float = 0.5, seed: int = 0) -> None:
        if int(n_particles) < 10:
            raise ValueError(f"n_particles must be at least 10; got {n_particles}")
        self.n_state = int(n_state)
        self.n_obs = int(n_obs)
        self.f = f
        self.h = h
        self.Q = np.eye(self.n_state) * 1e-4 if Q is None else np.asarray(Q, dtype=np.float64)
        self.R = np.eye(self.n_obs) * 1e-2 if R is None else np.asarray(R, dtype=np.float64)
        self.n_particles = int(n_particles)
        self.ess_fraction = float(ess_fraction)
        self.rng = np.random.default_rng(seed)
        self.particles = np.zeros((self.n_state, self.n_particles))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self._chol_Q = _cholesky_psd(self.Q)
        self._chol_R = _cholesky_psd(self.R)
        self._R_inv = np.linalg.inv(self.R)

    # -- belief -------------------------------------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        return self.particles @ self.weights

    @property
    def covariance(self) -> np.ndarray:
        spread = self.particles - self.mean[:, None]
        return (spread * self.weights) @ spread.T

    @property
    def ess(self) -> float:
        """Effective sample size: how many particles the weights are really worth (1 / sum w^2)."""
        return float(1.0 / np.sum(self.weights ** 2))

    # -- the recursion ------------------------------------------------------------------------------------
    def initialize(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        cov = self.Q if P0 is None else np.asarray(P0, dtype=np.float64)
        root = _cholesky_psd(cov)
        self.particles = x0[:, None] + root @ self.rng.standard_normal((self.n_state, self.n_particles))
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def initialize_from_samples(self, samples: np.ndarray) -> None:
        """Start from a cloud somebody else drew — the case a Gaussian cannot start from: two modes, a bound, a
        previous run. ``samples`` is ``(n_state, n_particles)``."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.shape != (self.n_state, self.n_particles):
            raise ValueError(f"samples must be ({self.n_state}, {self.n_particles}); got {samples.shape}")
        self.particles = samples.copy()
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def predict(self) -> None:
        noise = self._chol_Q @ self.rng.standard_normal((self.n_state, self.n_particles))
        for i in range(self.n_particles):
            self.particles[:, i] = np.asarray(self.f(self.particles[:, i]), dtype=np.float64).ravel() + noise[:, i]

    def update(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Reweight by the Gaussian likelihood of ``y`` under each particle, then resample if the cloud collapsed."""
        y = np.asarray(y, dtype=np.float64).ravel()
        predicted = np.column_stack([np.asarray(self.h(self.particles[:, i]), dtype=np.float64).ravel()
                                     for i in range(self.n_particles)])
        residual = y[:, None] - predicted
        d2 = np.einsum("in,ij,jn->n", residual, self._R_inv, residual)
        if not np.all(np.isfinite(d2)):
            raise FloatingPointError("the observation produces a non-finite likelihood (check y and R for NaN)")
        log_w = np.log(np.clip(self.weights, 1e-300, None)) - 0.5 * d2
        log_w -= np.max(log_w)                       # a likelihood can underflow the whole cloud to zero
        self.weights = np.exp(log_w) / float(np.sum(np.exp(log_w)))
        mean, ess = self.mean.copy(), self.ess          # the estimate this update produced, before resampling
        if ess <= 1.0 + 1e-9 and self.n_particles > 1:
            # Not an error: the weights are still well defined and the filter will recover. It is the signal that
            # the model or R is wrong for this observation, and it must not pass in silence.
            warnings.warn(f"the observation collapsed the cloud to one particle (closest is {np.sqrt(float(np.min(d2))):.3g} "
                          "standard deviations away); check R and the units of y", RuntimeWarning, stacklevel=2)
        innovation = y - predicted @ self.weights
        resampled = ess < self.ess_fraction * self.n_particles
        if resampled:
            self._systematic_resample()
        return {"x": mean, "ess": ess, "innovation": innovation, "resampled": np.array(resampled)}

    def _systematic_resample(self) -> None:
        """One uniform draw for the whole cloud: lower variance than multinomial, and it never loses the tails."""
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        idx = np.searchsorted(cumulative, positions)
        self.particles = self.particles[:, idx]
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def step(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Predict + update in one call."""
        self.predict()
        return self.update(y)
