"""Compound events through a copula of two variables: Gaussian and Gumbel (upper-tail dependent) copulas fitted by
maximum pseudo-likelihood on ranks, chosen by AIC against independence, with joint ("both exceed") probabilities,
joint return periods and bootstrap intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["CopulaFit", "pseudo_observations", "fit_copula", "gumbel_copula_sample", "gaussian_copula_sample",
           "joint_exceedance", "copula_cdf"]


def pseudo_observations(x) -> np.ndarray:
    """Ranks / (n + 1): the margins' empirical distribution, so the copula fit does not depend on them."""
    x = np.asarray(x, float)
    return stats.rankdata(x) / (len(x) + 1.0)


# -- families -----------------------------------------------------------------

def _gumbel_logpdf(u, v, th):
    a, b = -np.log(u), -np.log(v)
    A = a ** th + b ** th
    Ai = A ** (1.0 / th)
    return (-Ai - np.log(u) - np.log(v) + (th - 1.0) * (np.log(a) + np.log(b)) + (1.0 / th - 2.0) * np.log(A)
            + np.log(Ai + th - 1.0))


def _gumbel_cdf(u, v, th):
    return np.exp(-(((-np.log(u)) ** th + (-np.log(v)) ** th) ** (1.0 / th)))


def _gauss_logpdf(u, v, r):
    a, b = stats.norm.ppf(u), stats.norm.ppf(v)
    return -0.5 * np.log1p(-r * r) - (r * r * (a * a + b * b) - 2 * r * a * b) / (2 * (1 - r * r))


def _gauss_cdf(u, v, r):
    a, b = np.atleast_1d(stats.norm.ppf(u)), np.atleast_1d(stats.norm.ppf(v))
    mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]])
    return np.array([mvn.cdf([p, q]) for p, q in zip(a, b, strict=True)])


def copula_cdf(family: str, u, v, param: float):
    if family == "gumbel":
        return _gumbel_cdf(u, v, param)
    if family == "gaussian":
        return _gauss_cdf(u, v, param)
    if family == "independence":
        return np.asarray(u) * np.asarray(v)
    raise ValueError(f"unknown copula family {family!r}")


def gumbel_copula_sample(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Gumbel copula draws (Marshall-Olkin with a positive alpha-stable frailty, alpha = 1/theta, Kanter's
    representation)."""
    if theta < 1:
        raise ValueError("the Gumbel copula needs theta >= 1")
    if theta == 1:
        return rng.uniform(size=(n, 2))
    al = 1.0 / theta
    V = rng.uniform(0, np.pi, size=n)
    W = rng.exponential(size=n)
    A = (np.sin(al * V) ** al * np.sin((1 - al) * V) ** (1 - al) / np.sin(V)) ** (1.0 / (1 - al))
    S = (A / W) ** ((1 - al) / al)
    E = rng.exponential(size=(n, 2))
    return np.exp(-(E / S[:, None]) ** al)


def gaussian_copula_sample(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return stats.norm.cdf(z)


@dataclass
class CopulaFit:
    family: str
    param: float
    aic: float
    aics: dict[str, float]
    params: dict[str, float]
    kendall_tau: float
    n: int

    @property
    def upper_tail_dependence(self) -> float:
        """``lambda_U = lim P(V > t | U > t)`` as ``t -> 1``: ``2 - 2^(1/theta)`` for Gumbel, 0 otherwise."""
        return float(2 - 2 ** (1.0 / self.param)) if self.family == "gumbel" else 0.0

    def cdf(self, u, v):
        return copula_cdf(self.family, u, v, self.param)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "param": self.param, "aic": self.aic, "aics": self.aics,
                "params": self.params, "kendall_tau": self.kendall_tau, "n": self.n,
                "upper_tail_dependence": self.upper_tail_dependence}


def fit_copula(x, y, families=("gaussian", "gumbel")) -> CopulaFit:
    """Maximum pseudo-likelihood fit of each family on the ranks of ``(x, y)``; the family with the lowest AIC
    (independence has AIC 0) is chosen."""
    u, v = pseudo_observations(x), pseudo_observations(y)
    tau = float(stats.kendalltau(x, y).statistic)
    aics, params = {"independence": 0.0}, {"independence": float("nan")}
    for fam in families:
        if fam == "gumbel":
            f = lambda t: -np.sum(_gumbel_logpdf(u, v, t))  # noqa: E731
            r = optimize.minimize_scalar(f, bounds=(1.0 + 1e-6, 30.0), method="bounded",
                                         options={"xatol": 1e-8})
        elif fam == "gaussian":
            f = lambda t: -np.sum(_gauss_logpdf(u, v, t))  # noqa: E731
            r = optimize.minimize_scalar(f, bounds=(-0.999, 0.999), method="bounded", options={"xatol": 1e-8})
        else:
            raise ValueError(f"unknown copula family {fam!r}")
        aics[fam] = float(2 * r.fun + 2)
        params[fam] = float(r.x)
    fam = min(aics, key=aics.get)
    return CopulaFit(fam, params[fam], aics[fam], aics, params, tau, len(u))


def joint_exceedance(fit: CopulaFit, u, v):
    """``P(U > u, V > v) = 1 - u - v + C(u, v)`` for marginal non-exceedance probabilities ``u`` and ``v``."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return 1.0 - u - v + fit.cdf(u, v)
