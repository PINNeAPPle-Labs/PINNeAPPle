"""Uncertain inputs of a study: one distribution per parameter, sampled by Monte Carlo, Latin hypercube or Sobol.

A parameter is a dict (or anything with ``to_dict()``) such as::

    {"name": "k", "dist": "uniform", "low": 150, "high": 400, "unit": "W/(m K)"}
    {"name": "T", "dist": "normal", "mean": 25, "std": 2}
    {"name": "V", "dist": "triangular", "low": 0.5, "mode": 2, "high": 6}
    {"name": "h", "dist": "lognormal", "median": 30, "gsd": 1.2}       # geometric standard deviation
    {"name": "t", "dist": "truncnormal", "mean": 2, "std": 0.3, "low": 1.5, "high": 2.5}

or a ``{name: (low, high)}`` mapping (uniform on each range). Every distribution is sampled through its inverse CDF
from a point of the unit hypercube, so the same design (QMC, LHS, Saltelli, Morris, polynomial chaos) serves every
distribution. Used by :mod:`pinneapple_analysis.sensitivity` and the propagation in
:func:`pinneapple_analysis.uncertainty.tolerance.propagate_space`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import qmc

KINDS = ("uniform", "normal", "truncnormal", "lognormal", "triangular", "constant")


@dataclass(frozen=True)
class UncertainParameter:
    """One uncertain input. ``ppf`` maps probabilities in [0, 1] to values; ``bounds`` is the support (or the
    0.1 %-99.9 % range for an unbounded distribution)."""
    name: str
    dist: str = "uniform"
    params: dict[str, float] = field(default_factory=dict)
    unit: str = ""
    nominal: float | None = None

    def __post_init__(self):
        if self.dist not in KINDS:
            raise ValueError(f"{self.name}: unknown distribution {self.dist!r}; use one of {', '.join(KINDS)}")
        need = {"uniform": ("low", "high"), "normal": ("mean", "std"), "truncnormal": ("mean", "std", "low", "high"),
                "lognormal": ("median", "gsd"), "triangular": ("low", "mode", "high"), "constant": ("value",)}
        missing = [k for k in need[self.dist] if k not in self.params]
        if missing:
            raise ValueError(f"{self.name}: a {self.dist} distribution needs {', '.join(missing)}")
        p = self.params
        if self.dist in ("uniform", "truncnormal", "triangular") and not p["high"] > p["low"]:
            raise ValueError(f"{self.name}: high must be greater than low")
        if self.dist in ("normal", "truncnormal") and not p["std"] > 0:
            raise ValueError(f"{self.name}: std must be positive")
        if self.dist == "lognormal" and not (p["median"] > 0 and p["gsd"] > 1):
            raise ValueError(f"{self.name}: lognormal needs median > 0 and gsd > 1")
        if self.dist == "triangular" and not p["low"] <= p["mode"] <= p["high"]:
            raise ValueError(f"{self.name}: mode must lie in [low, high]")

    @property
    def frozen(self):
        p = self.params
        if self.dist == "uniform":
            return stats.uniform(p["low"], p["high"] - p["low"])
        if self.dist == "normal":
            return stats.norm(p["mean"], p["std"])
        if self.dist == "truncnormal":
            a, b = (p["low"] - p["mean"]) / p["std"], (p["high"] - p["mean"]) / p["std"]
            return stats.truncnorm(a, b, loc=p["mean"], scale=p["std"])
        if self.dist == "lognormal":
            return stats.lognorm(np.log(p["gsd"]), scale=p["median"])
        if self.dist == "triangular":
            w = p["high"] - p["low"]
            return stats.triang((p["mode"] - p["low"]) / w, loc=p["low"], scale=w)
        return None

    @property
    def bounded(self) -> bool:
        return self.dist in ("uniform", "truncnormal", "triangular", "constant")

    def ppf(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if self.dist == "constant":
            return np.full_like(u, self.params["value"])
        return self.frozen.ppf(np.clip(u, 1e-12, 1 - 1e-12))

    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.dist == "constant":
            return (x >= self.params["value"]).astype(float)
        return self.frozen.cdf(x)

    @property
    def bounds(self) -> tuple[float, float]:
        if self.dist == "constant":
            return (self.params["value"], self.params["value"])
        if self.bounded:
            return (self.params["low"], self.params["high"])
        lo, hi = self.frozen.ppf([0.001, 0.999])
        return (float(lo), float(hi))

    @property
    def center(self) -> float:
        if self.nominal is not None:
            return float(self.nominal)
        if self.dist == "constant":
            return float(self.params["value"])
        return float(self.frozen.median())

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "dist": self.dist, **self.params}
        if self.unit:
            d["unit"] = self.unit
        if self.nominal is not None:
            d["nominal"] = self.nominal
        return d


_META = ("name", "dist", "distribution", "unit", "nominal", "type", "description")


def as_parameter(spec: Any, name: str | None = None) -> UncertainParameter:
    """One parameter from a dict, a ``(low, high)`` pair (uniform) or an :class:`UncertainParameter`."""
    if isinstance(spec, UncertainParameter):
        return spec
    if hasattr(spec, "to_dict") and not isinstance(spec, Mapping):
        spec = spec.to_dict()
    if isinstance(spec, Mapping):
        d = dict(spec)
        dist = str(d.get("dist") or d.get("distribution") or "uniform").lower()
        nm = str(d.get("name") or name or "")
        if not nm:
            raise ValueError(f"parameter without a name: {spec!r}")
        params = {k: float(v) for k, v in d.items() if k not in _META and v is not None and v != ""}
        if dist == "uniform" and "low" not in params and "bounds" in d:
            params = {"low": float(d["bounds"][0]), "high": float(d["bounds"][1])}
        params.pop("bounds", None)
        nominal = d.get("nominal")
        return UncertainParameter(nm, dist, params, str(d.get("unit") or ""),
                                  None if nominal in (None, "") else float(nominal))
    if isinstance(spec, Sequence) and len(spec) == 2 and name:
        return UncertainParameter(name, "uniform", {"low": float(spec[0]), "high": float(spec[1])})
    raise ValueError(f"cannot read a parameter from {spec!r}")


def as_space(space: Any) -> list[UncertainParameter]:
    """A list of parameters from a list of specs, a ``{name: spec or (low, high)}`` mapping, or an object with a
    ``parameters`` attribute (e.g. a ParameterSpace contract)."""
    if hasattr(space, "parameters") and not isinstance(space, Mapping):
        space = space.parameters
    if isinstance(space, Mapping):
        if "parameters" in space:
            return as_space(space["parameters"])
        out = [as_parameter(v, k) for k, v in space.items()]
    else:
        out = [as_parameter(v) for v in space]
    names = [p.name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate parameter names in {names}")
    if not out:
        raise ValueError("the parameter space is empty")
    return out


def to_values(space: Sequence[UncertainParameter], U: np.ndarray) -> np.ndarray:
    """Map points of the unit hypercube (n, d) to parameter values through each inverse CDF."""
    U = np.atleast_2d(np.asarray(U, dtype=float))
    return np.column_stack([p.ppf(U[:, i]) for i, p in enumerate(space)])


def unit_sample(d: int, n: int, method: str = "lhs", seed: int = 0) -> np.ndarray:
    """``n`` points of the unit hypercube: ``"mc"`` (plain random), ``"lhs"`` (Latin hypercube) or ``"sobol"``
    (scrambled Sobol sequence; ``n`` is used as is, a power of two keeps its balance properties)."""
    if method in ("mc", "random"):
        return np.random.default_rng(seed).random((n, d))
    if method == "lhs":
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    if method == "sobol":
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)
    raise ValueError(f"unknown sampling method {method!r}; use 'mc', 'lhs' or 'sobol'")


def sample_space(space: Any, n: int, method: str = "lhs", seed: int = 0) -> np.ndarray:
    """``n`` samples (n, d) of the parameters of ``space`` (see :func:`as_space`)."""
    sp = as_space(space)
    return to_values(sp, unit_sample(len(sp), int(n), method, seed))


__all__ = ["UncertainParameter", "as_parameter", "as_space", "to_values", "unit_sample", "sample_space", "KINDS"]
