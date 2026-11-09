"""Fields of a VVD run: velocity, vorticity, pressure, stream function, streamlines, and what the wall sees.

Everything is evaluated *from the state* (domains plus the source sheet), so a field is what the simulation would
answer right now -- no interpolation of snapshots, no grid in the solver itself. The vorticity of a domain is the
Gaussian blob ``gamma / (pi eps^2) exp(-r^2/eps^2)`` it carries, the pressure is the Bernoulli form documented in
:meth:`~.solver.VVD2D.pressure`.

The stream function is that of the free domains and the free stream; the source sheet is a distribution of sources
and contributes a multi-valued potential, which is why it is left out (and said so).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .bodies import PanelSet

__all__ = [
    "FieldGrid",
    "grid_field",
    "pressure_coefficient",
    "streamlines",
    "surface_pressure",
    "surface_friction",
    "vorticity_field",
]


@dataclass
class FieldGrid:
    """A Cartesian block of fields, ready to plot or to save."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    vorticity: np.ndarray
    pressure: np.ndarray
    streamfunction: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.u, self.v)

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    def to_dict(self) -> dict[str, np.ndarray]:
        return {"x": self.x, "y": self.y, "u": self.u, "v": self.v, "vorticity": self.vorticity,
                "pressure": self.pressure, "streamfunction": self.streamfunction}


def _grid(extent: tuple[float, float, float, float] | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    if extent is None:
        extent = (-1.0, 4.0, -2.5, 2.5)
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, int(n))
    ys = np.linspace(y0, y1, int(n))
    return np.meshgrid(xs, ys)


def grid_field(sim, extent: tuple[float, float, float, float] | None = None,
               n: int = 160, fields: tuple[str, ...] = ("u", "v", "vorticity", "pressure", "streamfunction")
               ) -> FieldGrid:
    """Sample the simulation on an ``n x n`` grid over ``extent = (x0, x1, y0, y1)``."""
    X, Y = _grid(extent, n)
    points = np.column_stack([X.ravel(), Y.ravel()])
    zeros = np.zeros(X.shape)
    data = {"u": zeros, "v": zeros, "vorticity": zeros, "pressure": zeros, "streamfunction": zeros}
    if "u" in fields or "v" in fields or "pressure" in fields:
        velocity = sim.fluid_velocity(points)
    if "u" in fields:
        data["u"] = velocity[:, 0].reshape(X.shape)
    if "v" in fields:
        data["v"] = velocity[:, 1].reshape(X.shape)
    if "vorticity" in fields:
        dom = sim.domains
        if len(dom):
            data["vorticity"] = kernels.blob_vorticity(points, dom.x, dom.y, dom.gamma, dom.eps).reshape(X.shape)
    if "pressure" in fields:
        data["pressure"] = sim.pressure(points).reshape(X.shape)
    if "streamfunction" in fields:
        dom = sim.domains
        psi = (sim.config.u_inf[1] * points[:, 0] - sim.config.u_inf[0] * points[:, 1])
        if len(dom):
            psi = psi + kernels.blob_streamfunction(points, dom.x, dom.y, dom.gamma, dom.eps)
        data["streamfunction"] = psi.reshape(X.shape)
    return FieldGrid(X, Y, **data)


def vorticity_field(sim, points: np.ndarray) -> np.ndarray:
    """Vorticity at arbitrary ``points``."""
    dom = sim.domains
    if len(dom) == 0:
        return np.zeros(len(np.atleast_2d(points)))
    return kernels.blob_vorticity(points, dom.x, dom.y, dom.gamma, dom.eps)


def surface_pressure(sim, panels: PanelSet | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(midpoints, Cp)`` on the wall, ``Cp = (p - p_inf) / (rho U^2 / 2)``."""
    panels = sim.panels if panels is None else panels
    if len(panels) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    dynamic = 0.5 * sim.config.rho * sim.config.speed ** 2
    return panels.mid, sim.wall_pressure(panels) / dynamic


def pressure_coefficient(sim, points: np.ndarray) -> np.ndarray:
    """``Cp`` at arbitrary points."""
    dynamic = 0.5 * sim.config.rho * sim.config.speed ** 2
    return sim.pressure(points) / dynamic


def surface_friction(sim, panels: PanelSet | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(midpoints, Cf)`` on the wall, with the wall shear from the vorticity of the near-wall domains."""
    panels = sim.panels if panels is None else panels
    if len(panels) == 0 or len(sim.domains) == 0:
        return (np.zeros((0, 2)), np.zeros(0)) if panels is None else (panels.mid, np.zeros(len(panels)))
    omega = kernels.blob_vorticity(panels.mid, sim.domains.x, sim.domains.y, sim.domains.gamma, sim.domains.eps)
    tau = sim.config.rho * sim.config.nu * 2.0 * omega
    dynamic = 0.5 * sim.config.rho * sim.config.speed ** 2
    return panels.mid, tau / dynamic


def streamlines(sim, seeds: np.ndarray, step: float = 0.02, n_steps: int = 400,
                stop_inside: bool = True) -> list[np.ndarray]:
    """Integrate ``dx/ds = V/|V|`` from each seed, stopping when a point reaches a body or stagnates."""
    seeds = np.atleast_2d(np.asarray(seeds, dtype=float))
    lines: list[np.ndarray] = []
    for seed in seeds:
        points = [seed.copy()]
        p = seed.astype(float).copy()
        for _ in range(int(n_steps)):
            v = sim.fluid_velocity(p[None, :])[0]
            speed = float(np.hypot(v[0], v[1]))
            if not np.isfinite(speed) or speed < 1e-6:
                break
            p = p + step * v / speed
            points.append(p.copy())
            if stop_inside and len(sim.assembly) and sim.assembly.inside(p[None, :], sim.time)[0]:
                points.pop()
                break
            if np.hypot(*(p - points[0])) > 40.0:
                break
        lines.append(np.asarray(points))
    return lines
