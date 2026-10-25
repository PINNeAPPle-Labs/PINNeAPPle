"""The test problems of the X-TFC paper (Schiassi et al., arXiv:2005.10632v1), as the paper states them:
equation, domain, constraints, exact solution and the run settings it reports (neurons, activation, weight
range, training grid). No result of the paper is in this module; the targets live in
``examples/reproductions/xtfc_schiassi2021/TARGETS.md``.

Each case is an :class:`~pinneapple_simulation.numerical_solvers.xtfc_general.XTFCProblem` plus its settings.
The constraint function of every case is the paper's exact solution (only its traces on the constrained faces
are read, which are the paper's constraint data). Problem 7's right-hand side is built from its exact solution
because the printed one does not survive text extraction; every other right-hand side is the printed one.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from pinneapple_simulation.numerical_solvers.xtfc_general import Constraint, XTFCProblem

__all__ = ["XTFCCase", "XTFC_CASES", "xtfc_case"]

PI = math.pi


@dataclass(frozen=True)
class XTFCCase:
    name: str
    problem: XTFCProblem
    exact: dict[str, Callable[[torch.Tensor], torch.Tensor]]
    n_train: int                      # points per dimension of the training grid
    n_neurons: int
    activation: str
    weight_range: tuple[float, float]
    paper_iterations: int | None      # iterations the paper reports for a non-linear solve
    paper_tol: float | None
    source: str
    test_points: dict[str, np.ndarray] | None = None   # test points the paper lists (ODE tables)

    def exact_np(self, X, field: str | None = None) -> np.ndarray:
        f = field or next(iter(self.exact))
        with torch.no_grad():
            return self.exact[f](torch.as_tensor(np.atleast_2d(X), dtype=torch.float64)).numpy()


def _c(X, i):
    return X[:, i]


# ---- ODE Problem 1 (Sec. 3.1.1): y'' = (y^3 - 2 y^2) / (2 t^2), y(1) = 1, y(2) = 4/3, y = 2t/(t+1)
def _ode1_res(X, D):
    t = _c(X, 0)
    y, ytt = D[("y", (0,))], D[("y", (2,))]
    return ytt - (y ** 3 - 2 * y ** 2) / (2 * t ** 2)


def _ode1_exact(X):
    t = _c(X, 0)
    return 2 * t / (t + 1)


# ---- ODE Problem 2 (Sec. 3.1.2): Lagaris et al. system on [0, 3]
def _ode2_res(X, D):
    t = _c(X, 0)
    y1, y2 = D[("y1", (0,))], D[("y2", (0,))]
    r1 = D[("y1", (1,))] - (torch.cos(t) + y1 ** 2 + y2 - (1 + t ** 2 + torch.sin(t) ** 2))
    r2 = D[("y2", (1,))] - (2 * t - (1 + t ** 2) * torch.sin(t) + y1 * y2)
    return [r1, r2]


# ---- PDE Problem 1 (Sec. 3.2.1)
def _p1_res(X, D):
    x, y = _c(X, 0), _c(X, 1)
    return D[("f", (2, 0))] + D[("f", (0, 2))] - torch.exp(-x) * (x - 2 + y ** 3 + 6 * y)


def _p1_exact(X):
    x, y = _c(X, 0), _c(X, 1)
    return torch.exp(-x) * (x + y ** 3)


# ---- PDE Problems 2 and 3 (Secs. 3.2.2-3.2.3): exact f = y^2 sin(pi x)
def _p2_res(X, D):
    x, y = _c(X, 0), _c(X, 1)
    return D[("f", (2, 0))] + D[("f", (0, 2))] - (2 - PI ** 2 * y ** 2) * torch.sin(PI * x)


def _p3_res(X, D):
    x, y = _c(X, 0), _c(X, 1)
    s = torch.sin(PI * x)
    return D[("f", (2, 0))] + D[("f", (0, 2))] + D[("f", (0, 0))] * D[("f", (0, 1))] \
        - s * (2 - PI ** 2 * y ** 2 + 2 * y ** 3 * s)


def _p2_exact(X):
    return _c(X, 1) ** 2 * torch.sin(PI * _c(X, 0))


# ---- PDE Problem 4 (Sec. 3.2.4): f_xx = f_t on (x, t) in [0, 1]^2
def _p4_res(X, D):
    return D[("f", (2, 0))] - D[("f", (0, 1))]


def _p4_exact(X):
    return torch.sin(PI * _c(X, 0)) * torch.exp(-PI ** 2 * _c(X, 1))


# ---- PDE Problem 5 (Sec. 3.2.5): f_xx + f_yy = f_t on [0, 2] x [0, 1] x [0, 1]
_L5, _H5 = 2.0, 1.0


def _p5_res(X, D):
    return D[("f", (2, 0, 0))] + D[("f", (0, 2, 0))] - D[("f", (0, 0, 1))]


def _p5_exact(X):
    x, y, t = _c(X, 0), _c(X, 1), _c(X, 2)
    return torch.sin(PI * x / _L5) * torch.sin(PI * y / _H5) * torch.exp(-(PI ** 2 / _L5 ** 2 + PI ** 2 / _H5 ** 2) * t)


# ---- PDE Problem 6 (Sec. 3.2.6): z_t z_x + z_y = printed RHS on [0, 1]^3, variables (x, y, t)
def p6_rhs(X):
    x, y, t = _c(X, 0), _c(X, 1), _c(X, 2)
    cs = torch.cos(2 * PI * x * y)
    return t ** 2 + (t - 1) * x + 2 * PI * x * cs + (2 * t * y + x * y) * ((t - 1) * y + 2 * PI * y * cs)


def _p6_res(X, D):
    return D[("z", (0, 0, 1))] * D[("z", (1, 0, 0))] + D[("z", (0, 1, 0))] - p6_rhs(X)


def _p6_exact(X):
    x, y, t = _c(X, 0), _c(X, 1), _c(X, 2)
    return torch.sin(2 * PI * x * y) + t ** 2 * y + (t - 1) * x * y


# ---- PDE Problem 7 (Sec. 3.2.7): f_x f_y f_z + f_tt = RHS on [0, 1]^4, RHS from the exact solution
def _p7_exact(X):
    x, y, z, t = _c(X, 0), _c(X, 1), _c(X, 2), _c(X, 3)
    return t ** 2 * torch.sin(2 * PI * z) + torch.sin(x ** 2 * y) + x * y ** 1.5 * z + x * y * t * (z - 1) * (t - 1)


def p7_rhs(X):
    with torch.enable_grad():
        Xr = X.detach().clone().requires_grad_(True)
        f = _p7_exact(Xr)
        g = torch.autograd.grad(f.sum(), Xr, create_graph=True)[0]
        ftt = torch.autograd.grad(g[:, 3].sum(), Xr)[0][:, 3]
    return (g[:, 0] * g[:, 1] * g[:, 2] + ftt).detach()


def _p7_res(X, D):
    return D[("f", (1, 0, 0, 0))] * D[("f", (0, 1, 0, 0))] * D[("f", (0, 0, 1, 0))] + D[("f", (0, 0, 0, 2))] \
        - p7_rhs(X)


_SRC = "Schiassi et al., arXiv:2005.10632v1"
_DIR01 = lambda d: [Constraint(d, 0.0), Constraint(d, 1.0)]  # noqa: E731

XTFC_CASES: dict[str, XTFCCase] = {
    "ode1": XTFCCase(
        "ode1", XTFCProblem([(1.0, 2.0)], {"y": [Constraint(0, 1.0), Constraint(0, 2.0)]}, {"y": _ode1_exact},
                            _ode1_res, [("y", (0,)), ("y", (2,))]),
        {"y": _ode1_exact}, 51, 51, "logistic", (-10.0, 10.0), 5, 4.440892098500626e-16,
        f"{_SRC} Sec. 3.1.1 p. 8-11, Fig. 3 (n = L = 51)",
        {"y": np.linspace(1.0, 2.0, 11)[:, None]}),
    "ode2": XTFCCase(
        "ode2", XTFCProblem([(0.0, 3.0)], {"y1": [Constraint(0, 0.0)], "y2": [Constraint(0, 0.0)]},
                            {"y1": lambda X: torch.sin(_c(X, 0)), "y2": lambda X: 1 + _c(X, 0) ** 2},
                            _ode2_res, [("y1", (0,)), ("y1", (1,)), ("y2", (0,)), ("y2", (1,))]),
        {"y1": lambda X: torch.sin(_c(X, 0)), "y2": lambda X: 1 + _c(X, 0) ** 2},
        100, 100, "logistic", (-10.0, 10.0), 9, 10e-6, f"{_SRC} Sec. 3.1.2 p. 10-14, Figs. 5-6 (n = L = 100)",
        {"y1": np.linspace(0.3, 3.0, 10)[:, None], "y2": np.linspace(0.3, 3.0, 10)[:, None]}),
    "pde1": XTFCCase(
        "pde1", XTFCProblem([(0.0, 1.0)] * 2, {"f": _DIR01(0) + _DIR01(1)}, {"f": _p1_exact}, _p1_res,
                            [("f", (2, 0)), ("f", (0, 2))], linear=True),
        {"f": _p1_exact}, 30, 170, "tanh", (-1.0, 1.0), None, None, f"{_SRC} Sec. 3.2.1 p. 15-17"),
    "pde2": XTFCCase(
        "pde2", XTFCProblem([(0.0, 1.0)] * 2, {"f": _DIR01(0) + [Constraint(1, 0.0), Constraint(1, 1.0, 1)]},
                            {"f": _p2_exact}, _p2_res, [("f", (2, 0)), ("f", (0, 2))], linear=True),
        {"f": _p2_exact}, 30, 170, "tanh", (-1.0, 1.0), None, None, f"{_SRC} Sec. 3.2.2 p. 17-18"),
    "pde3": XTFCCase(
        "pde3", XTFCProblem([(0.0, 1.0)] * 2, {"f": _DIR01(0) + [Constraint(1, 0.0), Constraint(1, 1.0, 1)]},
                            {"f": _p2_exact}, _p3_res,
                            [("f", (0, 0)), ("f", (0, 1)), ("f", (2, 0)), ("f", (0, 2))]),
        {"f": _p2_exact}, 20, 150, "tanh", (-1.0, 1.0), 10, None, f"{_SRC} Sec. 3.2.3 p. 19-20"),
    "pde4": XTFCCase(
        "pde4", XTFCProblem([(0.0, 1.0)] * 2, {"f": _DIR01(0) + [Constraint(1, 0.0)]}, {"f": _p4_exact},
                            _p4_res, [("f", (2, 0)), ("f", (0, 1))], linear=True),
        {"f": _p4_exact}, 30, 196, "tanh", (-1.0, 1.0), None, None, f"{_SRC} Sec. 3.2.4 p. 21"),
    "pde5": XTFCCase(
        "pde5", XTFCProblem([(0.0, _L5), (0.0, _H5), (0.0, 1.0)],
                            {"f": [Constraint(0, 0.0), Constraint(0, _L5), Constraint(1, 0.0), Constraint(1, _H5),
                                   Constraint(2, 0.0)]},
                            {"f": _p5_exact}, _p5_res, [("f", (2, 0, 0)), ("f", (0, 2, 0)), ("f", (0, 0, 1))],
                            linear=True),
        {"f": _p5_exact}, 13, 400, "tanh", (-1.0, 1.0), None, None, f"{_SRC} Sec. 3.2.5 p. 21-23"),
    "pde6": XTFCCase(
        "pde6", XTFCProblem([(0.0, 1.0)] * 3, {"z": [Constraint(0, 0.0), Constraint(1, 0.0), Constraint(2, 1.0)]},
                            {"z": _p6_exact}, _p6_res, [("z", (1, 0, 0)), ("z", (0, 1, 0)), ("z", (0, 0, 1))]),
        {"z": _p6_exact}, 8, 255, "tanh", (-1.0, 1.0), 10, None, f"{_SRC} Sec. 3.2.6 p. 23-24"),
    "pde7": XTFCCase(
        "pde7", XTFCProblem([(0.0, 1.0)] * 4,
                            {"f": [Constraint(0, 0.0), Constraint(1, 0.0), Constraint(2, 1.0)] + _DIR01(3)},
                            {"f": _p7_exact}, _p7_res,
                            [("f", (1, 0, 0, 0)), ("f", (0, 1, 0, 0)), ("f", (0, 0, 1, 0)), ("f", (0, 0, 0, 2))]),
        {"f": _p7_exact}, 5, 340, "tanh", (-1.0, 1.0), 10, None, f"{_SRC} Sec. 3.2.7 p. 24-25"),
}


def xtfc_case(name: str) -> XTFCCase:
    return XTFC_CASES[name]
