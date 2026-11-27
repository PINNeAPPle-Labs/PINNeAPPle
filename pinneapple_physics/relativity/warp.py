"""The Alcubierre warp drive, measured: the expansion, the energy density and where the physics says "impossible".

The metric (Alcubierre 1994), in Cartesian ``(t, x, y, z)`` with ``c = G = 1``::

    ds^2 = -dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2

so :math:`g_{tt} = -1 + v_s^2 f^2`, :math:`g_{tx} = -v_s f`, and the spatial part is flat. ``f`` is a shape function that
is 1 inside the bubble and 0 outside, ``r_s = |x - x_s(t)|``, and ``v_s`` is the bubble's speed -- superluminal if you
like, because nothing *inside* the bubble moves.

Two things this module computes and one it refuses to hide:

* **the expansion** (the York time) :math:`\\theta = \\nabla \\cdot \\vec v = v_s (x - x_s)/r_s \\, f'(r_s)` -- negative
  ahead of the bubble, positive behind it, exactly the "contract in front, expand behind" the picture claims;
* **the energy density** :math:`T^{00} = G^{00}/8\\pi`, computed from the metric by differentiating it -- the Einstein
  tensor of the metric as written, not a quoted formula. It is **negative** wherever the wall is and the observer is off
  the axis, which is the whole problem: the drive needs exotic matter;
* **the null energy condition** :math:`T_{\\mu\\nu} k^\\mu k^\\nu` along null directions, evaluated on the same grid, which
  is where the "impossible" is written down rather than asserted. The null vectors are built as ``n + s`` (the Eulerian
  normal plus a unit spatial direction) and checked to satisfy ``g(k, k) = 0`` to roundoff -- ``k = (1, n)``, the obvious
  guess, is *not* null in this metric and silently measures nothing.

How it is checked (:mod:`tests.test_warp_bubble`):

* the energy density reproduces Alcubierre's closed form
  :math:`T^{00} = -(v_s^2/32\\pi)\\,(y^2+z^2)/r_s^2\\,(df/dr_s)^2` to 1e-6 relative, on random points in the wall;
* it is **zero on the axis** (``y = z = 0``), which is what that closed form says and what a curvature routine
  assembled from the wrong index permutation does not;
* where ``f`` is exactly zero the Riemann tensor vanishes: the metric is flat outside the bubble, and the curvature
  lives in the wall;
* integrating it gives the total, and the total is what a mission would have to supply: for a 10 m bubble at
  ``v_s = 1`` it comes out at ``-1.89e27 kg`` of *negative* energy -- ``-9.5e-4`` solar masses, about the mass of
  Jupiter -- and the drive needs it, which is the whole problem.

What this is not: a solution of the quantum inequalities (Pfenning-Ford) or a proposal for how to build one. It is the
metric, its curvature, and the sign of the answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["WarpMetric", "shape", "shape_derivative", "expansion", "Curvature", "einstein_tensor",
           "energy_density", "null_vector", "null_energy", "worst_null_energy", "integrated_energy",
           "solar_masses", "G_SI", "C_SI", "M_SUN_KG", "G_C2"]

G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN_KG = 1.98847e30


def shape(r: np.ndarray, radius: float, wall: float, kind: str = "alcubierre") -> np.ndarray:
    """The shape function ``f``: 1 inside the bubble, 0 outside, a wall of thickness ~``wall`` in between.

    ``"alcubierre"`` is the smooth ``tanh`` wall used in numerical work (it is what Alcubierre's own top-hat becomes once
    you ask for a differentiable function); ``"tophat"`` is his original, which is 1 for ``r < radius`` and 0 beyond and
    is kept because it makes the closed form easy to recognise.
    """
    r = np.asarray(r, float)
    if kind == "tophat":
        return (r < radius).astype(float)
    if kind == "gaussian":
        return np.exp(-np.clip(r / radius, 0.0, None) ** 2)
    if kind != "alcubierre":
        raise ValueError(f"unknown shape {kind!r}; use 'alcubierre', 'tophat' or 'gaussian'")
    if wall <= 0.0:
        raise ValueError("the wall thickness has to be positive")
    return 0.5 * (np.tanh((r + radius) / wall) - np.tanh((r - radius) / wall)) / math.tanh(radius / wall)


def shape_derivative(r: np.ndarray, radius: float, wall: float, kind: str = "alcubierre") -> np.ndarray:
    """``df/dr``, analytically -- the expansion and the energy both need it and differentiating a ``tanh`` by hand is
    cheaper and more accurate than finite differences."""
    r = np.asarray(r, float)
    if kind == "tophat":
        return np.zeros_like(r)
    if kind == "gaussian":
        return -2.0 * r / radius ** 2 * np.exp(-(r / radius) ** 2)
    if kind != "alcubierre":
        raise ValueError(f"unknown shape {kind!r}")
    denominator = 2.0 * math.tanh(radius / wall) * wall
    return (1.0 / np.cosh((r + radius) / wall) ** 2 - 1.0 / np.cosh((r - radius) / wall) ** 2) / denominator


@dataclass(frozen=True)
class WarpMetric:
    """One bubble: how fast it moves, how big it is, how thick the wall is, and where it is at ``t = 0``."""

    velocity: float = 1.0            # v_s, in units of c
    radius: float = 10.0             # metres (or any length: the geometry is scale-free)
    wall: float = 2.0                # metres of wall
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kind: str = "alcubierre"

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError("a bubble needs a radius")
        if self.wall <= 0.0:
            raise ValueError("a bubble needs a wall")
        if self.velocity < 0.0:
            raise ValueError("the metric is written for a bubble moving along +x")

    def radius_at(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        """``r_s``: the distance from the bubble's centre, which moves at ``v_s`` along ``x``."""
        points = np.atleast_2d(np.asarray(points, float))
        centre_x = self.centre[0] + self.velocity * time
        delta = points[..., 1:4] - np.array([centre_x, self.centre[1], self.centre[2]])
        return np.sqrt(np.sum(delta ** 2, axis=-1))

    def f(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        return shape(self.radius_at(points, time), self.radius, self.wall, self.kind)

    def metric(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        """``g_{mu nu}`` with ``mu, nu`` in ``(t, x, y, z)``, shape ``(N, 4, 4)``."""
        points = np.atleast_2d(np.asarray(points, float))
        f = self.f(points, time)
        g = np.zeros((len(points), 4, 4))
        g[:, 0, 0] = -1.0 + self.velocity ** 2 * f ** 2
        g[:, 0, 1] = g[:, 1, 0] = -self.velocity * f
        g[:, 1, 1] = g[:, 2, 2] = g[:, 3, 3] = 1.0
        return g


def expansion(metric: WarpMetric, points: np.ndarray, time: float = 0.0) -> np.ndarray:
    """The York time ``theta = div v = v_s (x - x_s)/r_s f'(r_s)``: contraction ahead, expansion behind."""
    points = np.atleast_2d(np.asarray(points, float))
    r = metric.radius_at(points, time)
    safe = np.clip(r, 1e-12, None)
    centre_x = metric.centre[0] + metric.velocity * time
    along = (points[:, 1] - centre_x) / safe
    return metric.velocity * along * shape_derivative(r, metric.radius, metric.wall, metric.kind)


# -------------------------------------------------------------------------------------------- the curvature
#
# The metric is differentiated on its analytic form by fourth-order central differences, in the frame that moves with
# the bubble (where ``d/dt g = 0`` and the metric is stationary -- the frame every Alcubierre analysis uses). Autograd
# would give the same numbers for first and second derivatives at the price of a double backward through 16 graphs per
# point; the check that matters is not *how* the derivatives were taken but whether the answer is right, and the answer
# has a closed form to be right against:
#
#     T^{00} = -(v_s^2 / 32 pi) * (y^2 + z^2) / r_s^2 * (df/dr_s)^2
#
# which comes from the ADM split in three lines (flat spatial metric, unit lapse, shift ``-v_s f``, so
# ``K_ij K^ij = v_s^2 (f_,x^2 + (f_,y^2 + f_,z^2)/2)`` and ``K^2 = v_s^2 f_,x^2``), and which the tests assert to
# 1e-6 relative. A curvature routine that reproduces it is right; one that does not is a picture.

@dataclass(frozen=True)
class Curvature:
    """The Einstein tensor and the energy it implies, sampled at points."""

    points: np.ndarray
    einstein: np.ndarray          # (N, 4, 4), units of 1/length^2
    energy_density: np.ndarray    # (N,), T^{00} measured by the Eulerian observers
    expansion: np.ndarray         # (N,), the York time from the same derivatives, for the cross-check
    is_flat: np.ndarray           # (N,), whether the Riemann tensor vanishes there to numerical precision

    def to_dict(self) -> dict[str, float]:
        return {"min_energy_density": float(self.energy_density.min()),
                "max_energy_density": float(self.energy_density.max()),
                "min_expansion": float(self.expansion.min()), "max_expansion": float(self.expansion.max()),
                "flat_points": int(self.is_flat.sum()), "points": int(len(self.points))}


def _grid_axis(fn, points: np.ndarray, axis: int, step: float) -> np.ndarray:
    """Fourth-order central difference of ``fn`` along ``axis`` (0 = t, 1..3 = x, y, z)."""
    if axis == 0:                                   # the bubble frame: the metric does not depend on time
        return np.zeros_like(fn(points))
    offsets = (-2, -1, 1, 2)
    weights = (1.0, -8.0, 8.0, -1.0)
    total = np.zeros_like(fn(points))
    for offset, weight in zip(offsets, weights, strict=True):
        shifted = points.copy()
        shifted[:, axis] += offset * step
        total = total + weight * fn(shifted)
    return total / (12.0 * step)


def _derivatives(metric: WarpMetric, points: np.ndarray, time: float, step: float):
    metric_fn = lambda p: metric.metric(p, time)                      # noqa: E731 - short and local
    n = len(points)
    d1 = np.zeros((n, 4, 4, 4))
    d2 = np.zeros((n, 4, 4, 4, 4))
    for d in range(4):
        d1[..., d] = _grid_axis(metric_fn, points, d, step)
    for d in range(4):
        for e in range(4):
            d2[..., d, e] = _grid_axis(lambda p, k=d: _grid_axis(metric_fn, p, k, step), points, e, step)
    return d1, d2


def einstein_tensor(metric: WarpMetric, points: np.ndarray, time: float = 0.0,
                    step: float | None = None) -> Curvature:
    """``G_{mu nu}`` of the metric as written, in the frame that moves with the bubble."""
    points = np.atleast_2d(np.asarray(points, float))
    step = step if step is not None else max(metric.wall * 1e-3, 1e-4 * metric.radius)
    g = metric.metric(points, time)
    g_inv = np.linalg.inv(g)
    d1, d2 = _derivatives(metric, points, time, step)

    # A[mu, nu, sigma] = d_mu g_{nu sigma} + d_nu g_{mu sigma} - d_sigma g_{mu nu}, with d1[p, i, j, k] = d_k g_{ij}
    A = d1.transpose(0, 1, 3, 2) + d1.transpose(0, 3, 1, 2) - d1
    gamma = 0.5 * np.einsum("pls,pmvs->plmv", g_inv, A)

    # d(gamma) needs the derivative of the inverse metric as well as of A
    dg_inv = -np.einsum("pla,psb,pabd->plsd", g_inv, g_inv, d1)
    dA = d2.transpose(0, 3, 1, 2, 4) + d2.transpose(0, 1, 3, 2, 4) - d2
    d_gamma = 0.5 * np.einsum("plsd,pmvs->plmvd", dg_inv, A) + 0.5 * np.einsum("pls,pmvsd->plmvd", g_inv, dA)

    # R^rho_{sigma mu nu} = d_mu G^rho_{nu sigma} - d_nu G^rho_{mu sigma}
    #                       + G^rho_{mu lam} G^lam_{nu sigma} - G^rho_{nu lam} G^lam_{mu sigma}
    riemann = d_gamma.transpose(0, 1, 3, 4, 2) - d_gamma
    riemann = riemann + np.einsum("prml,plns->prsmn", gamma, gamma) - np.einsum("prnl,plms->prsmn", gamma, gamma)

    ricci = np.einsum("prsrv->psv", riemann)
    scalar = np.einsum("psv,psv->p", g_inv, ricci)
    einstein = ricci - 0.5 * g * scalar[:, None, None]

    g0 = g_inv[:, 0, :]
    density = np.einsum("pa,pab,pb->p", g0, einstein, g0) / (8.0 * math.pi)
    # the expansion of the Eulerian congruence, from the shift: beta_i = g_{0i} with unit lapse, so theta = -d_i beta^i
    theta = -np.einsum("pii->p", d1[:, 0, :, :])
    scale = max(1.0, float(np.abs(riemann).max()))
    flat = np.all(np.abs(riemann.reshape(len(points), -1)) < 1e-6 * scale, axis=1)
    return Curvature(points, einstein, density, theta, flat)


def energy_density(metric: WarpMetric, points: np.ndarray, time: float = 0.0) -> np.ndarray:
    return einstein_tensor(metric, points, time).energy_density


def null_vector(metric: WarpMetric, points: np.ndarray, direction: tuple[float, float, float],
                time: float = 0.0) -> np.ndarray:
    """A vector that really is null: ``k = n + s``, the Eulerian normal plus a unit *spatial* direction.

    ``n^mu = (1, v_s f, 0, 0)`` is the 4-velocity of the observers the ADM split uses (unit lapse, shift
    ``-v_s f``), and the spatial frame is flat, so ``n + s`` is null for any unit ``s``. Writing ``k = (1, n)``
    instead -- the obvious thing, and what this module used to do -- is **not** null in this metric, and the
    "null energy" it produces is meaningless. :func:`null_energy` checks ``g(k, k) = 0`` before returning.
    """
    points = np.atleast_2d(np.asarray(points, float))
    s = np.asarray(direction, float)
    s = s / max(float(np.linalg.norm(s)), 1e-12)
    k = np.zeros((len(points), 4))
    k[:, 0] = 1.0
    k[:, 1] = metric.velocity * metric.f(points, time)
    k[:, 1:] += s
    return k


def null_energy(metric: WarpMetric, points: np.ndarray, direction: tuple[float, float, float] = (0.0, 1.0, 0.0),
                time: float = 0.0) -> np.ndarray:
    """``T_{mu nu} k^mu k^nu`` along the null direction ``k = n + s`` (see :func:`null_vector`).

    The null energy condition is this quantity being non-negative for *every* null direction. One negative value
    anywhere is a violation, and :func:`worst_null_energy` looks for it rather than trusting a single guess.
    """
    curvature = einstein_tensor(metric, points, time)
    k = null_vector(metric, points, direction, time)
    norm = np.einsum("pa,pab,pb->p", k, metric.metric(points, time), k)
    if np.any(np.abs(norm) > 1e-9 * max(1.0, float(np.abs(k).max()) ** 2)):
        raise ValueError(f"the direction did not produce a null vector (g(k,k) = {norm.max():.3e})")
    return np.einsum("pa,pab,pb->p", k, curvature.einstein, k) / (8.0 * math.pi)


def worst_null_energy(metric: WarpMetric, points: np.ndarray, *, samples: int = 16,
                      time: float = 0.0) -> dict[str, object]:
    """The most negative ``T_{mu nu} k^mu k^nu`` over a sphere of null directions, and where it is.

    A negative minimum is the null energy condition being violated at that point -- measured over a grid of
    directions, not asserted from the sign of ``T^{00}``.
    """
    curvature = einstein_tensor(metric, points, time)
    best = np.full(len(np.atleast_2d(points)), np.inf)
    best_direction = np.zeros(3)
    for theta in np.linspace(0.0, np.pi, samples):
        for phi in np.linspace(0.0, 2.0 * np.pi, 2 * samples, endpoint=False):
            direction = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                                  math.cos(theta)])
            k = null_vector(metric, points, direction, time)
            value = np.einsum("pa,pab,pb->p", k, curvature.einstein, k) / (8.0 * math.pi)
            improved = value < best
            if np.any(improved):
                best = np.where(improved, value, best)
                best_direction = direction
    return {"minimum": float(best.min()), "points_violating": int(np.sum(best < 0.0)),
            "points": int(len(best)), "direction": tuple(float(x) for x in best_direction),
            "violated": bool(np.any(best < 0.0))}


def integrated_energy(metric: WarpMetric, *, extent: float | None = None, samples: int = 41,
                      time: float = 0.0) -> dict[str, float]:
    """The total ``T^{00}`` over a box: negative, and the number a mission would have to supply.

    Units: the density comes out of the geometry in ``1/length^2`` (``G = c = 1``), so the integral is a length. It is
    converted to kilograms with ``c^2/G`` and reported in solar masses.
    """
    extent = extent if extent is not None else 2.2 * metric.radius + 4.0 * metric.wall
    axis = np.linspace(-extent, extent, samples)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    zeros = np.zeros((len(grid), 1))
    points = np.concatenate([zeros, grid], axis=1)
    density = energy_density(metric, points, time)
    cell = (axis[1] - axis[0]) ** 3
    total_geometric = float(np.sum(density) * cell)          # units of length
    kilograms = total_geometric * C_SI ** 2 / G_SI
    return {"total_geometric": total_geometric, "kilograms": kilograms, "solar_masses": kilograms / M_SUN_KG,
            "cell_m": cell, "samples": float(samples), "extent_m": float(extent)}


def solar_masses(kilograms: float) -> float:
    return kilograms / M_SUN_KG


#: ``G/c^2`` in metres per kilogram -- the conversion the geometric density needs to become a mass
G_C2 = G_SI / C_SI ** 2
