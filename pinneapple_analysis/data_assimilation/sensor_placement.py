"""Field reconstruction from sparse sensors with a Gaussian process, and greedy placement of the next sensors.

:class:`GaussianProcessField` is a small NumPy Gaussian process with an anisotropic Matern kernel (Rasmussen &
Williams 2006, eq. 4.14-4.17; nu in {1/2, 3/2, 5/2, inf}), a fixed or data-mean prior, Cholesky-based posterior
mean and variance (algorithm 2.1), closed-form leave-one-out residuals (eq. 5.12) and an optional fit of the
hyper-parameters by maximising the log marginal likelihood (eq. 5.8) with ``scipy.optimize.minimize``.

:func:`greedy_sensor_placement` chooses sensor locations one at a time among candidate points, each time taking the
candidate that most reduces the integrated posterior variance ``sum_x w(x) var(f(x) | sensors)`` over a set of
domain points. The posterior variance of a Gaussian process does not depend on the measured values, so the design
needs only the kernel. Greedy selection of GP sensor locations and its near-optimality for submodular criteria are
studied by Krause, Singh & Guestrin (2008); for the variance-reduction criterion see also Das & Kempe (2008). Adding
a sensor at ``c`` lowers the posterior covariance by the rank-one term
``Sigma(:, c) Sigma(c, :) / (Sigma(c, c) + noise)``, so each greedy step costs one pass over the candidates.

References
----------
Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

Krause, A., Singh, A. & Guestrin, C. (2008). Near-optimal sensor placements in Gaussian processes: theory, efficient
algorithms and empirical studies. *J. Mach. Learn. Res.* 9, 235-284.

Das, A. & Kempe, D. (2008). Algorithms for subset selection in linear regression. *Proc. 40th ACM STOC*, 45-54.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

__all__ = ["matern_kernel", "GaussianProcessField", "SensorPlacement", "greedy_sensor_placement"]

_JITTER = 1e-12


def matern_kernel(A, B, lengthscales=1.0, variance: float = 1.0, nu: float = 2.5) -> np.ndarray:
    """Anisotropic Matern covariance ``k(a, b)`` between the rows of ``A`` (n, d) and ``B`` (m, d).

    With the scaled distance ``r = sqrt(sum_i ((a_i - b_i) / l_i)^2)`` (Rasmussen & Williams 2006, eq. 4.17):
    ``nu = 0.5``: ``s2 exp(-r)``; ``nu = 1.5``: ``s2 (1 + sqrt3 r) exp(-sqrt3 r)``;
    ``nu = 2.5``: ``s2 (1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r)``; ``nu = inf``: ``s2 exp(-r^2 / 2)``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    ls = np.broadcast_to(np.asarray(lengthscales, dtype=float), (A.shape[1],))
    a, b = A / ls, B / ls
    r2 = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)
    r = np.sqrt(r2)
    if nu == 0.5:
        k = np.exp(-r)
    elif nu == 1.5:
        s = np.sqrt(3.0) * r
        k = (1.0 + s) * np.exp(-s)
    elif nu == 2.5:
        s = np.sqrt(5.0) * r
        k = (1.0 + s + s * s / 3.0) * np.exp(-s)
    elif np.isinf(nu):
        k = np.exp(-0.5 * r2)
    else:
        raise ValueError(f"nu must be 0.5, 1.5, 2.5 or inf, got {nu}")
    return variance * k


class GaussianProcessField:
    """Gaussian-process regression of a scalar field from point sensors (Rasmussen & Williams 2006, ch. 2).

    Parameters
    ----------
    lengthscales : Matern length scale per input dimension (scalar for isotropic), in the units of ``X``.
    variance : prior variance ``s2`` of the field, in (field units)^2.
    nu : Matern smoothness, 0.5, 1.5, 2.5 or ``np.inf`` (squared exponential).
    noise : variance of the sensor noise added to the diagonal of the training covariance.
    mean : prior mean, a float, or ``"data"`` for the sample mean of ``y`` at :meth:`fit` (held fixed afterwards,
        including in :meth:`loo`).
    """

    def __init__(self, lengthscales=1.0, variance: float = 1.0, nu: float = 2.5, noise: float = 1e-10,
                 mean: float | str = 0.0) -> None:
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance)
        self.nu = nu
        self.noise = float(noise)
        self.mean = mean
        self.X: np.ndarray | None = None

    def kernel(self, A, B) -> np.ndarray:
        """Prior covariance between the rows of ``A`` and ``B``."""
        return matern_kernel(A, B, self.lengthscales, self.variance, self.nu)

    # -- fitting -------------------------------------------------------------------------------------------------
    def fit(self, X, y) -> GaussianProcessField:
        """Condition on sensor positions ``X`` (n, d) and readings ``y`` (n,)."""
        X = np.asarray(X, dtype=float)
        X = X[:, None] if X.ndim == 1 else X
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows, y has {len(y)}")
        self.X, self.y = X, y
        self.m_ = float(np.mean(y)) if self.mean == "data" else float(self.mean)
        K = self.kernel(X, X) + (self.noise + _JITTER * self.variance) * np.eye(len(X))
        self.L_ = linalg.cholesky(K, lower=True)
        self.alpha_ = linalg.cho_solve((self.L_, True), y - self.m_)
        return self

    def _check(self) -> None:
        if self.X is None:
            raise RuntimeError("call fit(X, y) first")

    def predict(self, Xs, return_std: bool = False, return_cov: bool = False):
        """Posterior mean at ``Xs`` (Rasmussen & Williams 2006, alg. 2.1); with ``return_std`` / ``return_cov`` also
        the standard deviation / covariance of the latent field (sensor noise not included)."""
        self._check()
        Xs = np.asarray(Xs, dtype=float)
        Xs = Xs[:, None] if Xs.ndim == 1 else Xs
        Ks = self.kernel(Xs, self.X)
        mu = self.m_ + Ks @ self.alpha_
        if not (return_std or return_cov):
            return mu
        V = linalg.solve_triangular(self.L_, Ks.T, lower=True)
        if return_cov:
            return mu, self.kernel(Xs, Xs) - V.T @ V
        var = np.maximum(self.variance - (V * V).sum(0), 0.0)
        return mu, np.sqrt(var)

    def loo(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out residuals ``y_i - mu_{-i}(x_i)`` and predictive variances, in closed form
        (Rasmussen & Williams 2006, eq. 5.12): ``[K^-1 y]_i / [K^-1]_ii`` and ``1 / [K^-1]_ii``, where ``K`` includes
        the noise. Hyper-parameters and the prior mean are held fixed."""
        self._check()
        Kinv = linalg.cho_solve((self.L_, True), np.eye(len(self.X)))
        d = np.diag(Kinv)
        return self.alpha_ / d, 1.0 / d

    def log_marginal_likelihood(self) -> float:
        """``log p(y | X) = -1/2 y^T K^-1 y - sum log L_ii - n/2 log 2 pi`` (Rasmussen & Williams 2006, eq. 5.8)."""
        self._check()
        r = self.y - self.m_
        return float(-0.5 * r @ self.alpha_ - np.log(np.diag(self.L_)).sum() - 0.5 * len(r) * np.log(2 * np.pi))

    def fit_hyperparameters(self, X, y, fit_noise: bool = False, log_bounds=(-7.0, 7.0)) -> GaussianProcessField:
        """Maximise the log marginal likelihood over log length scales, log variance (and log noise when
        ``fit_noise``) with L-BFGS-B, then :meth:`fit` with the optimum. ``log_bounds`` bounds every natural-log
        parameter.

        The likelihood of a Gaussian process has more than one optimum, and a single run from the caller's guess can
        land in a degenerate one (a length scale that collapses, so the fit interpolates its samples and predicts the
        mean everywhere else). The optimiser therefore also starts from the data: a third of each axis' span and the
        variance of ``y``. The best of the runs wins, so the answer no longer depends on the guess it started from."""
        X = np.asarray(X, dtype=float)
        X = X[:, None] if X.ndim == 1 else X
        d = X.shape[1]
        ls0 = np.broadcast_to(self.lengthscales, (d,)).copy()
        theta0 = np.log(np.r_[ls0, self.variance] if not fit_noise else np.r_[ls0, self.variance,
                                                                               max(self.noise, 1e-10)])
        span = np.ptp(X, axis=0)
        span[span <= 0] = 1.0
        data_start = np.log(np.r_[0.3 * span, max(float(np.var(y)), 1e-12)] if not fit_noise
                            else np.r_[0.3 * span, max(float(np.var(y)), 1e-12), max(self.noise, 1e-10)])
        starts = [theta0] if np.allclose(data_start, theta0) else [theta0, data_start]

        def nll(theta):
            self.lengthscales, self.variance = np.exp(theta[:d]), float(np.exp(theta[d]))
            if fit_noise:
                self.noise = float(np.exp(theta[d + 1]))
            try:
                return -self.fit(X, y).log_marginal_likelihood()
            except linalg.LinAlgError:
                return 1e25

        best, best_nll = None, np.inf
        for start in starts:
            res = optimize.minimize(nll, np.clip(start, *log_bounds), method="L-BFGS-B",
                                    bounds=[log_bounds] * len(start))
            if res.fun < best_nll:
                best, best_nll = res.x, float(res.fun)
        nll(best)
        return self


@dataclass
class SensorPlacement:
    """Result of :func:`greedy_sensor_placement`.

    indices : chosen rows of ``candidates``, in order of selection.
    points : the chosen candidate coordinates (k, d).
    integrated_variance : weighted posterior variance over the domain before the first pick and after each pick
        (k + 1,), non-increasing.
    """
    indices: np.ndarray
    points: np.ndarray
    integrated_variance: np.ndarray


def greedy_sensor_placement(gp: GaussianProcessField, candidates, n_sensors: int, domain=None, weights=None,
                            existing=None) -> SensorPlacement:
    """Choose ``n_sensors`` locations among ``candidates`` that greedily minimise the integrated posterior variance.

    At each step the candidate ``c`` maximising ``sum_x w(x) Sigma(x, c)^2 / (Sigma(c, c) + noise)`` is added and
    the posterior covariance is downdated by that rank-one term (Krause, Singh & Guestrin 2008; Das & Kempe 2008).
    Only the kernel and noise of ``gp`` are used; it need not be fitted.

    Parameters
    ----------
    gp : :class:`GaussianProcessField` giving the kernel and the sensor noise.
    candidates : admissible sensor locations (n_c, d).
    n_sensors : number of sensors to add (at most ``n_c``).
    domain : points (n_x, d) over which the variance is integrated (default: the candidates).
    weights : quadrature weights (n_x,) (default: ``1 / n_x``, i.e. the mean variance).
    existing : sensors already installed (n_e, d); the design is conditioned on them.
    """
    C = np.asarray(candidates, dtype=float)
    C = C[:, None] if C.ndim == 1 else C
    n_c = len(C)
    if not 0 <= n_sensors <= n_c:
        raise ValueError(f"n_sensors must be in [0, {n_c}]")
    if domain is None:
        P, ix_dom, ix_cand = C, np.arange(n_c), np.arange(n_c)
    else:
        D = np.asarray(domain, dtype=float)
        D = D[:, None] if D.ndim == 1 else D
        P = np.vstack([D, C])
        ix_dom, ix_cand = np.arange(len(D)), len(D) + np.arange(n_c)
    w = np.full(len(ix_dom), 1.0 / len(ix_dom)) if weights is None else np.asarray(weights, dtype=float)
    noise = gp.noise + _JITTER * gp.variance
    S = gp.kernel(P, P)
    if existing is not None:
        E = np.asarray(existing, dtype=float)
        E = E[:, None] if E.ndim == 1 else E
        Kpe = gp.kernel(P, E)
        L = linalg.cholesky(gp.kernel(E, E) + noise * np.eye(len(E)), lower=True)
        V = linalg.solve_triangular(L, Kpe.T, lower=True)
        S = S - V.T @ V

    chosen: list[int] = []
    ivar = [float(w @ np.diag(S)[ix_dom])]
    available = np.ones(n_c, dtype=bool)
    for _ in range(n_sensors):
        Sdc = S[np.ix_(ix_dom, ix_cand)]
        gain = (w[:, None] * Sdc * Sdc).sum(0) / (np.maximum(np.diag(S)[ix_cand], 0.0) + noise)
        gain[~available] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        available[j] = False
        c = ix_cand[j]
        col = S[:, c].copy()
        S = S - np.outer(col, col) / (max(col[c], 0.0) + noise)
        ivar.append(float(w @ np.maximum(np.diag(S)[ix_dom], 0.0)))
    idx = np.array(chosen, dtype=int)
    return SensorPlacement(indices=idx, points=C[idx], integrated_variance=np.array(ivar))
