"""The cases of the campaign (docs/research/cfd_direct_from_stl.md) with their published values and the geometry
files every method reads (a polygon in 2-D, an STL in 3-D, generated here).

- C1, cylinder in a channel, Schafer & Turek (1996) case 2D-1 (Re 20): Cd, Cl, Delta p. Values: the refined
  computations of John & Matthies (2001) / Nabh (1998), inside the benchmark intervals.
- C2, sphere in uniform flow, Re 20 / 50 / 100: Cd from the correlation of Clift, Grace & Weber (1978) for
  20 <= Re <= 260; Schiller & Naumann (1933) next to it.
- C3, backward-facing step of Armaly et al. (1983), expansion ratio 1.942: the experimental primary reattachment
  length x1/S as tabulated in Table VIII of arXiv:2602.04670 (the first open tabulation reachable from the campaign's
  container; the paper itself was not reachable).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

__all__ = ["SchaferTurek", "SCHAFER_TUREK", "Sphere", "clift_grace_weber", "schiller_naumann", "cylinder_polygon",
           "sphere_stl", "BackwardStep", "ARMALY_X1", "reattachment"]


@dataclass(frozen=True)
class SchaferTurek:
    """Channel [0, L] x [0, H], cylinder of diameter D at c, parabolic inflow of peak U_max, viscosity nu."""

    L: float = 2.2
    H: float = 0.41
    c: tuple = (0.2, 0.2)
    D: float = 0.1
    U_max: float = 0.3
    nu: float = 1e-3
    Cd: float = 5.57953523384
    Cl: float = 0.010618948146
    dp: float = 0.11752016697
    p_front: tuple = (0.15, 0.2)
    p_back: tuple = (0.25, 0.2)

    @property
    def U_mean(self) -> float:
        return 2.0 * self.U_max / 3.0

    @property
    def Re(self) -> float:
        return self.U_mean * self.D / self.nu

    @property
    def box(self):
        return (0.0, 0.0), (self.L, self.H)

    def inflow(self, y: np.ndarray) -> np.ndarray:
        return 4.0 * self.U_max * y * (self.H - y) / self.H ** 2

    def coefficients(self, F: np.ndarray) -> tuple[float, float]:
        """(Cd, Cl) from the force per unit depth (density 1)."""
        q = 0.5 * self.U_mean ** 2 * self.D
        return float(F[0] / q), float(F[1] / q)

    def profile_points(self) -> dict[str, np.ndarray]:
        """Velocity profiles of the metric: u(y) at x = 0.15, 0.25, 0.4, 1.0 (points inside the fluid)."""
        y = np.linspace(0.0, self.H, 83)[1:-1]
        out = {}
        for x in (0.15, 0.25, 0.4, 1.0):
            P = np.column_stack([np.full_like(y, x), y])
            keep = np.hypot(P[:, 0] - self.c[0], P[:, 1] - self.c[1]) > 0.5 * self.D + 1e-3
            out[f"x={x}"] = P[keep]
        return out


SCHAFER_TUREK = SchaferTurek()


def cylinder_polygon(n: int = 512, case: SchaferTurek = SCHAFER_TUREK) -> np.ndarray:
    """The cylinder as a regular n-gon inscribed in the circle (counter-clockwise)."""
    t = 2 * np.pi * np.arange(n) / n
    r = 0.5 * case.D
    return np.column_stack([case.c[0] + r * np.cos(t), case.c[1] + r * np.sin(t)])


def clift_grace_weber(re: float) -> float:
    """Sphere drag, Clift, Grace & Weber, *Bubbles, Drops and Particles* (1978), Table 5.2, 20 <= Re <= 260."""
    if not 20.0 <= re <= 260.0:
        raise ValueError("this branch of the correlation holds for 20 <= Re <= 260")
    return 24.0 / re * (1.0 + 0.1935 * re ** 0.6305)


def schiller_naumann(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


@dataclass(frozen=True)
class Sphere:
    """Sphere of diameter D at the origin in the box [-5D, 15D] x [-5D, 5D]^2, uniform inflow U along +x."""

    Re: float
    D: float = 1.0
    U: float = 1.0
    upstream: float = 5.0
    downstream: float = 15.0
    half_width: float = 5.0

    @property
    def nu(self) -> float:
        return self.U * self.D / self.Re

    @property
    def box(self):
        w = self.half_width * self.D
        return (-self.upstream * self.D, -w, -w), (self.downstream * self.D, w, w)

    @property
    def Cd_ref(self) -> float:
        return clift_grace_weber(self.Re)

    @property
    def Cd_schiller_naumann(self) -> float:
        return schiller_naumann(self.Re)

    def drag_coefficient(self, Fx: float) -> float:
        return float(Fx / (0.5 * self.U ** 2 * math.pi * self.D ** 2 / 4))

    def axis_points(self) -> np.ndarray:
        """The wake axis from the rear of the sphere to 5 D behind it."""
        x = np.linspace(0.5 * self.D + 0.02 * self.D, 5.5 * self.D, 60)
        return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])


def sphere_stl(path: str, D: float = 1.0, subdivisions: int = 5) -> str:
    """Write the sphere as an icosphere STL (20 * 4^subdivisions triangles; 20,480 at 5)."""
    from pinneapple_simulation.immersed.geometry import icosphere, write_stl
    V, F = icosphere(0.5 * D, subdivisions)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return write_stl(path, V, F)


# Armaly et al. (1983), experimental x1/S, as tabulated in arXiv:2602.04670, Table VIII
ARMALY_X1 = {100: 2.96, 200: 4.90, 400: 8.45}


@dataclass(frozen=True)
class BackwardStep:
    """Armaly's step with S = 1: inlet channel y in [S, H] for x in [-upstream, 0], then y in [0, H] to x = downstream;
    H / h = 1.942 (H = 10.1 mm, h = 5.2 mm, S = 4.9 mm); parabolic inflow of mean U; Re = U 2 h / nu."""

    Re: float
    S: float = 1.0
    U: float = 1.0
    upstream: float = 5.05            # 5 S, plus a fraction that keeps the step face off the grid planes
    downstream: float = 30.0

    @property
    def h(self) -> float:
        return self.S * 5.2 / 4.9

    @property
    def H(self) -> float:
        return self.S * 10.1 / 4.9

    @property
    def nu(self) -> float:
        return self.U * 2 * self.h / self.Re

    @property
    def x1_ref(self) -> float:
        return ARMALY_X1[int(self.Re)]

    @property
    def box(self):
        return (-self.upstream, 0.0), (self.downstream, self.H)

    def polygon(self) -> np.ndarray:
        S, H, a, b = self.S, self.H, -self.upstream, self.downstream
        return np.array([[a, S], [0.0, S], [0.0, 0.0], [b, 0.0], [b, H], [a, H]], float)

    def inflow(self, y: np.ndarray) -> np.ndarray:
        s = (np.asarray(y, float) - self.S) / self.h
        return np.where((s > 0) & (s < 1), 6.0 * self.U * s * (1 - s), 0.0)

    def profile_points(self) -> dict[str, np.ndarray]:
        y = np.linspace(0.0, self.H, 63)[1:-1]
        return {f"x/S={x}": np.column_stack([np.full_like(y, x * self.S), y]) for x in (2, 5, 10)}


def reattachment(x: np.ndarray, u_near_wall: np.ndarray, x0: float = 0.0) -> float:
    """End of the primary recirculation: of the stretches downstream of ``x0`` where the near-wall velocity is
    negative, the longest one, and the point (linear interpolation) where it turns positive again. The corner eddy
    at the foot of the step (tiny, of either sign) is skipped that way. NaN if there is no reversed flow."""
    x, u = np.asarray(x, float), np.asarray(u_near_wall, float)
    m = x > x0
    x, u = x[m], u[m]
    neg = u < 0
    if not neg.any():
        return float("nan")
    edges = np.diff(np.concatenate([[0], neg.astype(int), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]          # run [s, e)
    k = int(np.argmax(x[ends - 1] - x[starts]))
    j = ends[k]
    if j >= len(x):
        return float("nan")
    return float(x[j - 1] - u[j - 1] * (x[j] - x[j - 1]) / (u[j] - u[j - 1]))
