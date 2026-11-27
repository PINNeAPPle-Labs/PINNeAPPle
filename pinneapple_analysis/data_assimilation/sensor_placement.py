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

The greedy walk is one answer to a combinatorial question, not the only one (#352). :class:`Layout` is a design with
its criterion and its price; :func:`random_sensor_placement` and :func:`maximin_sensor_placement` are the two
baselines (the sensor lottery, and spreading them evenly); :func:`evolutionary_sensor_placement` searches the same
problem with the library's genetic algorithm and, unlike the walk, can hold a hard constraint -- a budget per sensor
or a minimum distance between them, which is what mounting points and cable routes are; and
:func:`layout_curves` puts the four on one axis, the criterion against the number of sensors. Two criteria are
available throughout: the weighted integrated variance ``sum_x w(x) var(f(x))`` and the variance of a quantity of
interest ``a^T f``, ``a^T Sigma a`` -- the second is the one to use when only one number matters.

Measured in ``tests/test_sensor_placement.py`` (``2026-11-26``), criterion against the number of sensors, 1/2/4/8:

a 32 x 20 grid on a 1 x 0.6 plate with an isotropic Matern (``l = 0.18``, ``nu = 5/2``), scored on the variance of
the mean temperature over the hot corner (prior ``0.647``)::

    k                        1        2        4        8
    greedy              0.0539   0.0341   0.0089   0.0006
    evolutionary        0.2130   0.0471   0.0527   0.0324
    uniform (maximin)   0.6470   0.3799   0.3778   0.2681
    random              0.5427   0.4147   0.3171   0.2134

and 80 stations down a 3 m slope with a long Matern (``l = 0.35``, ``nu = 3/2``), weighted towards the toe (prior
``1.000``)::

    k                        1        2        4        8
    greedy              0.4629   0.2913   0.0965   0.0193
    evolutionary        0.4629   0.2528   0.0907   0.0342
    uniform (maximin)   1.0000   0.9443   0.3542   0.1020
    random              0.7102   0.5439   0.4091   0.1822

The greedy walk wins where the criterion is the one it optimises step by step (the plate, three orders of magnitude
by eight sensors) and is beaten on the slope at two and four sensors (0.2913 against 0.2528, 0.0965 against
0.0907): the guarantee is a bound, not optimality. What the search buys beyond that is the constraint the walk
cannot see -- at eight sensors on the plate the walk's closest pair is 0.063 apart and its criterion 0.0006; the
spaced search, held to 0.15, reaches 0.173 and 0.104, against 0.374 and 0.268 for the even spread.

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

__all__ = ["matern_kernel", "GaussianProcessField", "SensorPlacement", "Layout", "greedy_sensor_placement",
           "evolutionary_sensor_placement", "maximin_sensor_placement", "random_sensor_placement",
           "layout_curves", "closest_pair"]

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
                            existing=None, qoi=None, cost=None, budget=None) -> SensorPlacement:
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
    qoi : a quantity of interest ``a^T f`` as its coefficient vector (n_x,) over ``domain``. The criterion becomes
        the variance of that one number, ``a^T Sigma a``, instead of the weighted pointwise variance, and the greedy
        step maximises ``(a^T Sigma(:, c))^2 / (Sigma(c, c) + noise)``. ``weights`` is then ignored.
    cost : the price of each candidate (n_c,), default one per sensor.
    budget : the most the chosen sensors may cost. The step maximises the gain per unit of cost and the walk stops
        when the next best candidate no longer fits -- the budgeted greedy of Krause & Golovin (2014, sec. 1.2),
        whose ratio guarantee survives the cost.
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
    a = None if qoi is None else np.asarray(qoi, dtype=float)
    if a is not None and a.shape != (len(ix_dom),):
        raise ValueError(f"qoi must have one coefficient per domain point ({len(ix_dom)}), got {a.shape}")
    price = np.ones(n_c) if cost is None else np.asarray(cost, dtype=float)
    if price.shape != (n_c,):
        raise ValueError(f"cost must have one price per candidate ({n_c}), got {price.shape}")
    if budget is not None and budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    noise = gp.noise + _JITTER * gp.variance
    S = gp.kernel(P, P)
    if existing is not None:
        E = np.asarray(existing, dtype=float)
        E = E[:, None] if E.ndim == 1 else E
        Kpe = gp.kernel(P, E)
        L = linalg.cholesky(gp.kernel(E, E) + noise * np.eye(len(E)), lower=True)
        V = linalg.solve_triangular(L, Kpe.T, lower=True)
        S = S - V.T @ V

    def criterion(S_now: np.ndarray) -> float:
        sub = S_now[np.ix_(ix_dom, ix_dom)]
        return float(w @ np.maximum(np.diag(sub), 0.0)) if a is None else float(a @ sub @ a)

    chosen: list[int] = []
    ivar = [criterion(S)]
    spent = 0.0
    available = np.ones(n_c, dtype=bool)
    for _ in range(n_sensors):
        Sdc = S[np.ix_(ix_dom, ix_cand)]
        if a is None:
            gain = (w[:, None] * Sdc * Sdc).sum(0)
        else:
            gain = (a @ Sdc) ** 2
        gain = gain / (np.maximum(np.diag(S)[ix_cand], 0.0) + noise) / price
        gain[~available] = -np.inf
        if budget is not None:
            gain[spent + price > budget] = -np.inf
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]):
            break                                    # nothing left that fits the budget
        chosen.append(j)
        available[j] = False
        spent += float(price[j])
        c = ix_cand[j]
        col = S[:, c].copy()
        S = S - np.outer(col, col) / (max(col[c], 0.0) + noise)
        ivar.append(criterion(S))
    idx = np.array(chosen, dtype=int)
    return SensorPlacement(indices=idx, points=C[idx], integrated_variance=np.array(ivar))


# ------------------------------------------------------------------ layouts and their comparison

def _as_points(a) -> np.ndarray:
    P = np.asarray(a, dtype=float)
    return P[:, None] if P.ndim == 1 else P


@dataclass
class Layout:
    """A set of sensors chosen by one strategy, and the criterion it reaches.

    strategy : ``"greedy"``, ``"evolutionary"``, ``"random"`` or ``"uniform"``.
    indices : the chosen rows of ``candidates``.
    points : the chosen coordinates (k, d).
    criterion : the value of the criterion for this set -- the weighted integrated posterior variance, or the
        variance of the quantity of interest when one was given. Lower is better, and it is comparable across the
        strategies only when they were scored on the same domain, weights and ``qoi``.
    cost : the total price of the chosen sensors.
    history : for the evolutionary search, the best criterion per generation; empty for the one-shot layouts.
    """
    strategy: str
    indices: np.ndarray
    points: np.ndarray
    criterion: float
    cost: float
    history: np.ndarray


class _Criterion:
    """The posterior criterion of a layout, with everything that does not depend on the layout precomputed.

    Conditioning on ``m`` sensors costs one ``m x m`` factorisation and an ``n_x m`` product, so scoring a whole
    layout is cheap even where the greedy walk, which pays ``O(n^2)`` per step, would not be -- which is what makes
    a search over thousands of layouts affordable. ``Kdd`` is computed once per instance.
    """

    def __init__(self, gp: GaussianProcessField, domain, weights=None, qoi=None, existing=None):
        self.kernel = gp.kernel
        self.Kdd = gp.kernel(domain, domain)
        self.domain = domain
        self.noise = gp.noise + _JITTER * gp.variance
        self.weights = (np.full(len(domain), 1.0 / len(domain)) if weights is None
                        else np.asarray(weights, dtype=float))
        self.qoi = None if qoi is None else np.asarray(qoi, dtype=float)
        self.existing = np.empty((0, np.shape(domain)[1])) if existing is None else _as_points(existing)

    def __call__(self, sensors) -> float:
        """The criterion with ``sensors`` installed on top of the ``existing`` ones."""
        S = np.vstack([self.existing, _as_points(sensors)])
        sub = self.Kdd
        if len(S):
            Kss = self.kernel(S, S) + self.noise * np.eye(len(S))
            Kds = self.kernel(self.domain, S)
            sub = self.Kdd - Kds @ linalg.cho_solve(linalg.cho_factor(Kss, lower=True), Kds.T)
        if self.qoi is not None:
            return float(self.qoi @ sub @ self.qoi)
        return float(self.weights @ np.maximum(np.diag(sub), 0.0))


def random_sensor_placement(gp: GaussianProcessField, candidates, n_sensors: int, domain=None, weights=None,
                            existing=None, qoi=None, seed: int = 0) -> Layout:
    """The baseline that has no criterion at all: ``n_sensors`` candidates drawn uniformly, without replacement."""
    C = _as_points(candidates)
    if not 0 <= n_sensors <= len(C):
        raise ValueError(f"n_sensors must be in [0, {len(C)}]")
    D = C if domain is None else _as_points(domain)
    idx = np.sort(np.random.default_rng(int(seed)).choice(len(C), size=n_sensors, replace=False))
    crit = _Criterion(gp, D, weights, qoi, existing)
    return Layout("random", idx, C[idx], crit(C[idx]), float(n_sensors), np.array([]))


def maximin_sensor_placement(gp: GaussianProcessField, candidates, n_sensors: int, domain=None, weights=None,
                             existing=None, qoi=None) -> Layout:
    """The space-filling baseline: the candidates furthest from each other and from what is already installed.

    Farthest-point sampling ignores the criterion completely -- it is what a "put them evenly spread" layout does,
    and the comparison against the greedy walk is what says whether spreading evenly is enough.
    """
    C = _as_points(candidates)
    if not 0 <= n_sensors <= len(C):
        raise ValueError(f"n_sensors must be in [0, {len(C)}]")
    D = C if domain is None else _as_points(domain)
    taken = np.zeros(len(C), dtype=bool)
    d = np.full(len(C), np.inf)
    if existing is not None and np.asarray(existing).size:
        d = np.min(np.linalg.norm(C[:, None, :] - _as_points(existing)[None, :, :], axis=-1), axis=1)
    idx = []
    for _ in range(n_sensors):
        d = np.where(taken, -np.inf, d)
        j = int(np.argmax(d))
        idx.append(j)
        taken[j] = True
        d = np.minimum(d, np.linalg.norm(C - C[j], axis=1))
    idx = np.array(idx, dtype=int)
    crit = _Criterion(gp, D, weights, qoi, existing)
    return Layout("uniform", idx, C[idx], crit(C[idx]), float(n_sensors), np.array([]))


def closest_pair(points) -> float:
    """The smallest distance between two of ``points``; ``inf`` for fewer than two."""
    P = _as_points(points)
    if len(P) < 2:
        return float("inf")
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return float(np.min(d[np.triu_indices(len(P), 1)]))



def _decode(x: np.ndarray, n_sensors: int) -> np.ndarray:
    """A real vector over the candidates to a set of sensors: the ``n_sensors`` highest scores.

    The encoding is one real value per candidate and the layout is its top ``n_sensors``, so a mutation moves one
    site in or out and no design is ever an invalid set with fewer or more sensors than asked for.
    """
    return np.sort(np.argsort(-np.asarray(x, dtype=float), kind="stable")[:n_sensors])


def evolutionary_sensor_placement(gp: GaussianProcessField, candidates, n_sensors: int, domain=None, weights=None,
                                  existing=None, qoi=None, cost=None, budget=None, min_spacing: float | None = None,
                                  population_size: int = 30, n_generations: int = 25, seed: int = 0) -> Layout:
    """The same combinatorial problem, searched with the library's genetic algorithm instead of greedily.

    Greedy has a ratio guarantee but no way to honour a hard constraint, and a layout usually has one: mounting
    points, cable routes, a budget. The search here is
    :class:`pinneapple_design.design_optimizer.EvolutionaryDesignOptimizer`, driven with one real value per candidate and decoded to the ``n_sensors`` highest
    (see :func:`_decode`); a layout over the budget is penalised in proportion to the overshoot, so the search
    trades a slightly worse criterion for a layout it can actually build.

    ``min_spacing`` is the hard constraint the greedy walk cannot express: no two sensors closer than that. A layout
    that breaks it is penalised in proportion to how close its worst pair is, so ``maximin_sensor_placement`` is
    always feasible and the search has to trade criterion for distances it can actually build.

    ``multi_objective=True`` is passed to the optimizer to keep it on its (mu+lambda) genetic path: the same search
    runs whether or not the optional ``cma`` package is installed, which is what makes the layout reproducible.
    """
    C = _as_points(candidates)
    n_c = len(C)
    if not 0 <= n_sensors <= n_c:
        raise ValueError(f"n_sensors must be in [0, {n_c}]")
    D = C if domain is None else _as_points(domain)
    price = np.ones(n_c) if cost is None else np.asarray(cost, dtype=float)
    if price.shape != (n_c,):
        raise ValueError(f"cost must have one price per candidate ({n_c}), got {price.shape}")
    if budget is not None and budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    crit = _Criterion(gp, D, weights, qoi, existing)
    scale = max(abs(crit(np.empty((0, C.shape[1])))), 1.0)

    from pinneapple_design import design_optimizer as _do

    def objective(x) -> float:
        idx = _decode(x, n_sensors)
        value = crit(C[idx])
        if budget is not None:
            over = float(price[idx].sum()) - float(budget)
            if over > 0.0:
                value += 1e3 * scale * over / float(budget)
        if min_spacing is not None and n_sensors > 1:
            gap = closest_pair(C[idx])
            if gap < min_spacing:
                value += 1e3 * scale * (min_spacing - gap) / float(min_spacing)
        return value

    opt = _do.EvolutionaryDesignOptimizer(
        _do.DesignOptimizerConfig(method="evolutionary", n_iters=n_generations, population_size=population_size,
                              mutation_std=0.15, crossover_rate=0.7),
        seed=seed, multi_objective=True)
    bounds = np.tile(np.array([0.0, 1.0]), (n_c, 1))
    best_x, best_y, history = None, float("inf"), []
    for _ in range(n_generations):
        xs = opt.ask(bounds, population_size)
        ys = [objective(x) for x in xs]
        opt.tell(xs, ys)
        j = int(np.argmin(ys))
        if ys[j] < best_y:
            best_y, best_x = float(ys[j]), np.asarray(xs[j], dtype=float)
        history.append(best_y)                       # the best so far, which is what the search can only improve
    idx = _decode(best_x, n_sensors)
    return Layout("evolutionary", idx, C[idx], crit(C[idx]), float(price[idx].sum()), np.array(history))


def layout_curves(gp: GaussianProcessField, candidates, n_sensors_grid, domain=None, weights=None, existing=None,
                  qoi=None, cost=None, budget=None, min_spacing: float | None = None,
                  strategies=("greedy", "evolutionary", "random", "uniform"),
                  n_random: int = 20, population_size: int = 30, n_generations: int = 25,
                  seed: int = 0) -> dict[str, np.ndarray]:
    """The criterion against the number of sensors, one curve per strategy -- the comparison the issue asks for.

    Returns ``{strategy: array}`` with one value per entry of ``n_sensors_grid``, all scored by the same
    :class:`_Criterion`, plus ``"prior"``: the criterion before any sensor. Greedy is one nested walk, so its curve
    costs what its largest ``k`` costs; the evolutionary curve runs once per ``k``; random is averaged over
    ``n_random`` draws (the same seed for every ``k``, so the curves are comparable).
    """
    ks = [int(k) for k in n_sensors_grid]
    if not ks:
        raise ValueError("n_sensors_grid is empty")
    C = _as_points(candidates)
    D = C if domain is None else _as_points(domain)
    crit = _Criterion(gp, D, weights, qoi, existing)
    out: dict[str, np.ndarray] = {"prior": np.array([crit(np.empty((0, C.shape[1])))])}
    if "greedy" in strategies:
        walk = greedy_sensor_placement(gp, C, max(ks), domain=D, weights=weights, existing=existing, qoi=qoi,
                                       cost=cost, budget=budget)
        hist = walk.integrated_variance
        out["greedy"] = np.array([hist[min(k, len(hist) - 1)] for k in ks])
    if "evolutionary" in strategies:
        out["evolutionary"] = np.array([
            evolutionary_sensor_placement(gp, C, k, domain=D, weights=weights, existing=existing, qoi=qoi,
                                          cost=cost, budget=budget, min_spacing=min_spacing,
                                          population_size=population_size, n_generations=n_generations,
                                          seed=seed + i).criterion
            for i, k in enumerate(ks)])
    if "random" in strategies:
        out["random"] = np.array([
            float(np.mean([random_sensor_placement(gp, C, k, domain=D, weights=weights, existing=existing, qoi=qoi,
                                                   seed=seed * 1000 + d).criterion for d in range(n_random)]))
            for k in ks])
    if "uniform" in strategies:
        out["uniform"] = np.array([
            maximin_sensor_placement(gp, C, k, domain=D, weights=weights, existing=existing, qoi=qoi).criterion
            for k in ks])
    unknown = set(strategies) - {"greedy", "evolutionary", "random", "uniform"}
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}")
    return out
