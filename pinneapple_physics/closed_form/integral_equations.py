"""Closed-form solutions of the integral-equation presets (``pinneapple_physics.pde_environment.presets.integral``).

Each entry of :data:`ANALYTIC` is ``(name, matcher(problem) -> bool, factory(problem) -> fn(X) -> (N, 1))``, the same
shape :mod:`pinneapple_physics.closed_form.electromagnetics` uses, so ``pp.solve(problem, "analytic")`` resolves these
problems like any other preset.

The pairs here are *source and solution together*. For an integral equation the closed form is only half the case: the
free term ``f`` has to be the one that makes that ``u`` a solution, and if the two drift apart the preset silently
stops testing anything. So every case exposes both, and each pair is checked against a numerical integral in
``tests/test_integral_presets.py``.

The four cases, with ``lambda`` kept off the eigenvalue at ``lambda = 1`` (the Fredholm alternative: there the unit
kernel makes the system singular, so no finite comparison means anything):

``fredholm_unit_exp``       K = 1, f = exp(x) - lambda (e - 1), u = exp(x)
``fredholm_separable_exp``  K = x s (degenerate, rank one), f = exp(x) - lambda x, u = exp(x), since
                            int_0^1 s e^s ds = 1
``volterra_unit_exp``       K = 1, f = 1, u = exp(x)
``log_kernel_linear``       K = ln|x - s|, u = a + b x, f = u - lambda int_0^1 ln|x - s| u(s) ds, with the integral
                            written out below so the source stays elementary
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ANALYTIC", "fredholm_unit_exp", "fredholm_unit_exp_source", "fredholm_separable_exp",
    "fredholm_separable_exp_source", "volterra_unit_exp", "volterra_unit_exp_source",
    "log_kernel_linear", "log_kernel_linear_source", "log_kernel_moment0", "log_kernel_moment1",
]

_E = math.e


# ------------------------------------------------------------------------------------------------ K = 1, Fredholm
def fredholm_unit_exp(x, lam: float = 0.5):
    """``u = exp(x)``, for ``u(x) = f(x) + lambda int_0^1 u ds`` with :func:`fredholm_unit_exp_source`."""
    return np.exp(np.asarray(x, float))


def fredholm_unit_exp_source(x, lam: float = 0.5):
    """``f = exp(x) - lambda (e - 1)``: the integral of ``exp`` over ``[0, 1]`` is ``e - 1``."""
    return np.exp(np.asarray(x, float)) - lam * (_E - 1.0)


# ------------------------------------------------------------------------------------------------ K = x s, Fredholm
def fredholm_separable_exp(x, lam: float = 0.5):
    """``u = exp(x)``, for the degenerate kernel ``K(x, s) = x s``."""
    return np.exp(np.asarray(x, float))


def fredholm_separable_exp_source(x, lam: float = 0.5):
    """``f = exp(x) - lambda x``: with ``K = x s`` the integral is ``x int_0^1 s e^s ds = x``, since
    ``int s e^s ds = e^s (s - 1)`` gives ``0 - (-1) = 1``."""
    return np.exp(np.asarray(x, float)) - lam * np.asarray(x, float)


# ------------------------------------------------------------------------------------------------ K = 1, Volterra
def volterra_unit_exp(x):
    """``u = exp(x)``, for ``u(x) = 1 + int_0^x u ds``."""
    return np.exp(np.asarray(x, float))


def volterra_unit_exp_source(x):
    """``f = 1``."""
    return np.ones_like(np.asarray(x, float))


# ------------------------------------------------------------------------------------------------ K = ln|x - s|
def log_kernel_moment0(x):
    """``int_0^1 ln|x - s| ds = x ln x + (1 - x) ln(1 - x) - 1``, with ``0 ln 0 = 0`` at the ends."""
    x = np.asarray(x, float)
    left = np.where(x > 0.0, x * np.log(np.clip(x, 1e-300, None)), 0.0)
    right = np.where(x < 1.0, (1.0 - x) * np.log(np.clip(1.0 - x, 1e-300, None)), 0.0)
    return left + right - 1.0


def log_kernel_moment1(x):
    """``int_0^1 s ln|x - s| ds``, from ``u = s - x``::

        int (u + x) ln|u| du = [u^2/2 (ln|u| - 1/2)] + x [u (ln|u| - 1)],  u from -x to 1 - x
    """
    x = np.asarray(x, float)

    def g(u):
        return np.where(u != 0.0, 0.5 * u * u * (np.log(np.abs(np.where(u != 0.0, u, 1.0))) - 0.5), 0.0)

    def h(u):
        return np.where(u != 0.0, u * (np.log(np.abs(np.where(u != 0.0, u, 1.0))) - 1.0), 0.0)

    return (g(1.0 - x) - g(-x)) + x * (h(1.0 - x) - h(-x))


def log_kernel_linear(x, a: float = 1.0, b: float = 0.5):
    """``u = a + b x``: non-constant, so a model cannot pass by learning a constant."""
    return a + b * np.asarray(x, float)


def log_kernel_linear_source(x, lam: float = 0.5, a: float = 1.0, b: float = 0.5):
    """``f = u - lambda (a M0 + b M1)`` with the two exact moments of ``ln|x - s|``."""
    u = log_kernel_linear(x, a, b)
    return u - lam * (a * log_kernel_moment0(x) + b * log_kernel_moment1(x))


# ------------------------------------------------------------------------------------------------ registry
def _meta(p) -> dict:
    return dict(getattr(p.pde, "meta", None) or {}) if p.pde is not None else {}


def _params(p) -> dict:
    out = dict(p.pde.params)
    out.update({k: v.value for k, v in p.parameters.items() if v.value is not None})
    return out


def _is(name):
    return lambda p: _meta(p).get("analytic") == name


def _unit_exp(p):
    q = _params(p)
    return lambda X: np.stack([fredholm_unit_exp(X[:, 0], q.get("lambda", 0.5))], axis=1)


def _separable(p):
    q = _params(p)
    return lambda X: np.stack([fredholm_separable_exp(X[:, 0], q.get("lambda", 0.5))], axis=1)


def _volterra(p):
    return lambda X: np.stack([volterra_unit_exp(X[:, 0])], axis=1)


def _log_linear(p):
    q = _params(p)
    return lambda X: np.stack([log_kernel_linear(X[:, 0], q.get("a", 1.0), q.get("b", 0.5))], axis=1)


ANALYTIC = [
    ("integral_fredholm_unit_exp", _is("integral_fredholm_unit_exp"), _unit_exp),
    ("integral_fredholm_separable_exp", _is("integral_fredholm_separable_exp"), _separable),
    ("integral_volterra_unit_exp", _is("integral_volterra_unit_exp"), _volterra),
    ("integral_log_kernel_linear", _is("integral_log_kernel_linear"), _log_linear),
]
