"""A configurable 3-D + time FDTD study on top of :mod:`pinneapple_simulation.electromagnetics.fdtd`: a domain in
wavelengths, sources (Hertzian dipoles, soft E sources) and materials (spheres, boxes, cylinders with any
:class:`~pinneapple_simulation.electromagnetics.media.Medium`) from a plain dict, a refinement ladder, the radiated
power by the near-to-far-field transform, snapshots of the field in space and time, and the Maxwell checks of
those snapshots.

    from pinneapple_science.em.fdtd_study import FDTDConfig, run_fdtd
    cfg = FDTDConfig.from_dict({"f0": 1e9, "size": 2.4, "cells_per_wavelength": 15,
                                "sources": [{"kind": "dipole", "component": "Ez", "at": [0, 0, 0]}],
                                "materials": [{"shape": "sphere", "center": [0.45, 0, 0], "radius": 0.2,
                                               "eps_r": 4.0}]})
    out = run_fdtd(cfg, snapshots=24)
    out.power, out.pattern(theta), out.frames, out.checks

Positions are in wavelengths, measured from the centre of the domain. The grid is cubic; CPML on every face.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_simulation.electromagnetics.constants import C0
from pinneapple_simulation.electromagnetics.fdtd import (
    FDTD,
    NearToFarField,
    YeeGrid,
    fill_fraction,
    modulated_gaussian,
)
from pinneapple_simulation.electromagnetics.media import Medium

__all__ = ["FDTDConfig", "FDTDRun", "run_fdtd", "SHAPES"]

SHAPES = ("sphere", "box", "cylinder")


@dataclass
class FDTDConfig:
    f0: float = 1e9
    size: float = 2.4                       # domain edge, wavelengths (inside the CPML)
    cells_per_wavelength: int = 15
    npml: int = 10
    bandwidth: float = 0.5                  # of the modulated Gaussian, relative to f0
    periods: float = 14.0                   # run time in periods of f0
    sources: list[dict[str, Any]] = field(default_factory=lambda: [{"kind": "dipole", "component": "Ez",
                                                                   "at": [0.0, 0.0, 0.0]}])
    materials: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FDTDConfig:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown FDTD configuration keys {sorted(unknown)}")
        cfg = cls(**dict(d))
        for m in cfg.materials:
            if m.get("shape") not in SHAPES:
                raise ValueError(f"material shape {m.get('shape')!r} not in {SHAPES}")
        for s in cfg.sources:
            if s.get("kind", "dipole") not in ("dipole", "soft"):
                raise ValueError(f"source kind {s.get('kind')!r}: 'dipole' (impressed current) or 'soft' (E)")
        return cfg

    @property
    def wavelength(self) -> float:
        return C0 / self.f0


@dataclass
class FDTDRun:
    config: FDTDConfig
    dx: float
    n: int
    power: float                            # W at f0, near-to-far-field
    current_moment: complex                 # I l of the first dipole at f0 (A m)
    ntff: Any
    frames: np.ndarray                      # (n_snap, n, n) E_z on the plane y = 0 (V/m)
    frame_times: np.ndarray
    volume: dict[str, np.ndarray]           # E and H on a sub-block in space and time (x, y, z, t)
    volume_grid: dict[str, np.ndarray]
    slices: dict[str, np.ndarray]           # (n_snap, n, n) E_z on the three central planes
    energy: np.ndarray
    steps: int

    def pattern(self, theta, phi=0.0):
        """Far-field power pattern |E|^2 at f0 (not normalised)."""
        th = np.atleast_1d(np.asarray(theta, float))
        Et, Ep = self.ntff.far_field(th, np.full_like(th, phi))
        return np.abs(Et[0]) ** 2 + np.abs(Ep[0]) ** 2


def _shape_mask(g: YeeGrid, m: Mapping[str, Any], centre: float, lam: float):
    c = np.asarray(m.get("center", [0, 0, 0]), float) * lam + centre
    if m["shape"] == "sphere":
        r = m["radius"] * lam
        inside = lambda x, y, z: (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= r * r  # noqa: E731
    elif m["shape"] == "box":
        h = np.asarray(m["size"], float) * lam / 2
        inside = lambda x, y, z: (abs(x - c[0]) <= h[0]) & (abs(y - c[1]) <= h[1]) & (abs(z - c[2]) <= h[2])  # noqa: E731
    else:
        r = m["radius"] * lam
        hz = m["length"] * lam / 2
        inside = lambda x, y, z: ((x - c[0]) ** 2 + (y - c[1]) ** 2 <= r * r) & (abs(z - c[2]) <= hz)  # noqa: E731
    return fill_fraction(g, inside, oversample=4)


def run_fdtd(cfg: FDTDConfig, *, snapshots: int = 24, volume_half: int = 3, volume_steps: int = 24,
             maxwell_block: bool = True, snap_window: tuple[float, float] = (2.0, 7.0)) -> FDTDRun:
    """Run the configured study: returns power at f0 (near-to-far-field box two cells inside the CPML), the
    snapshots of E_z on the three central planes during ``snap_window`` (periods of f0), a small block of all six
    components around a point 0.6 wavelength from the first source on ``volume_steps`` consecutive steps (for the
    Maxwell checks in x, y, z, t; ``maxwell_block=False`` skips it when the domain has no room) and the discrete
    energy in time."""
    lam = cfg.wavelength
    dx = lam / cfg.cells_per_wavelength
    n = int(round(cfg.size * cfg.cells_per_wavelength)) + 2 * cfg.npml
    n += 1 - n % 2                               # odd: a centre node
    g = YeeGrid((n, n, n), (dx, dx, dx))
    sim = FDTD(g, {"x": "cpml", "y": "cpml", "z": "cpml"}, npml=cfg.npml)
    c = n // 2
    centre = c * dx
    for m in cfg.materials:
        sim.set_material(_shape_mask(g, m, centre, lam), Medium(m.get("eps_r", 1.0), m.get("mu_r", 1.0),
                                                                m.get("sigma", 0.0)))
    w = modulated_gaussian(cfg.f0, cfg.bandwidth * cfg.f0)
    first = None
    for s in cfg.sources:
        idx = tuple(int(round(c + v * cfg.cells_per_wavelength)) for v in s.get("at", [0, 0, 0]))
        if s.get("kind", "dipole") == "dipole":
            sim.add_current(s.get("component", "Ez"), idx, w, s.get("amplitude", 1.0))
            first = first or (idx, s.get("amplitude", 1.0))
        else:
            sim.add_source(s.get("component", "Ez"), idx, w, "soft")
    b = cfg.npml + 2
    ntff = NearToFarField(sim, [cfg.f0], ((b, n - b), (b, n - b), (b, n - b)))
    T = cfg.periods / cfg.f0
    total = int(math.ceil(T / sim.dt))
    # snapshots while the pulse crosses the domain: ``snap_window`` in periods of f0
    snap_every = max(1, int(round(((snap_window[1] - snap_window[0]) / cfg.f0) / sim.dt / snapshots)))
    t_snap0 = int(round(snap_window[0] / cfg.f0 / sim.dt))
    frames, times, slices, energy = [], [], {"xy": [], "xz": [], "yz": []}, []
    # sub-block for the 4-D Maxwell check: 0.6 lambda (or as far as the domain allows) from the first source along
    # the first of +-x, +-y, +-z
    # where the whole block is vacuum (the check uses eps0 and mu0)
    p0 = (first[0] if first else (c, c, c))
    off = min(int(round(0.6 * cfg.cells_per_wavelength)), n // 2 - cfg.npml - volume_half - 2)
    blk = None
    for ax in range(3 if off > volume_half else 0):
        for sgn in (1, -1):
            q = list(p0)
            q[ax] += sgn * off
            cand = tuple(slice(v - volume_half, v + volume_half + 1) for v in q)
            if all(s.start >= cfg.npml and s.stop <= n - cfg.npml for s in cand) and np.all(sim.eps_inf[cand] == 1.0):
                blk = cand
                break
        if blk:
            break
    if blk is None and maxwell_block:
        raise ValueError("no vacuum block beside the source for the Maxwell checks (enlarge the domain)")
    if blk is None:
        volume_steps = 0
    vol = {k: [] for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    t_vol0 = int(round(total * 0.35))
    for k in range(total):
        sim.step()
        if k % max(1, total // 200) == 0:
            energy.append((sim.n * sim.dt, sim.energy()))
        if t_vol0 <= k < t_vol0 + volume_steps:
            _, f = sim.snapshot()
            for name in vol:
                vol[name].append(f[name][blk])
        if k >= t_snap0 and (k - t_snap0) % snap_every == 0 and len(times) < snapshots:
            Ez = sim.field_at_nodes("Ez")
            frames.append(Ez[:, c, :].copy())
            slices["xz"].append(Ez[:, c, :].copy())
            slices["xy"].append(Ez[:, :, c].copy())
            slices["yz"].append(Ez[c, :, :].copy())
            times.append(sim.n * sim.dt)
    tj = (np.arange(sim.n) + 0.5) * sim.dt
    Il = np.sum(np.array([w(x) for x in tj]) * np.exp(-2j * np.pi * cfg.f0 * tj)) * sim.dt * dx ** 3
    if first:
        Il *= first[1]
    power = float(ntff.radiated_power(n_theta=91, n_phi=36)[0])
    if blk is None:
        blk = (slice(0, 0),) * 3
    grid_vol = {"x": (np.arange(blk[0].start, blk[0].stop)) * dx, "y": (np.arange(blk[1].start, blk[1].stop)) * dx,
                "z": (np.arange(blk[2].start, blk[2].stop)) * dx,
                "t": (np.arange(volume_steps) + t_vol0 + 0.5) * sim.dt}
    volume = {k: np.moveaxis(np.asarray(v), 0, -1) for k, v in vol.items()} if volume_steps else {}
    return FDTDRun(cfg, dx, n, power, complex(Il), ntff, np.asarray(frames), np.asarray(times), volume, grid_vol,
                   {k: np.asarray(v) for k, v in slices.items()}, np.asarray(energy), total)


def ladder_values(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]
