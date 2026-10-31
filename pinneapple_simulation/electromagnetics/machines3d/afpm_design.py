"""Axial-flux PM machine design in two steps: a quick quasi-3D estimate, then a check in full 3-D on the same grid.

:func:`estimate` cuts the active annulus into radial slices with flux-parallel faces (the usual quasi-3D model,
:meth:`.afpm.AFPM3D.quasi3d_models`) and returns the cogging over one period, its peak-to-peak and the no-load flux
linkage of a coil around a stator tooth. :func:`verify_3d` solves the same machine in full 3-D on the same grid,
with air inside, outside, above and below, and returns the same quantities with their ratio to the estimate:
what quasi-3D leaves out (the fringing at the inner and outer radius) for this design. On the default machine the
estimate overstated the cogging by 11 to 12 % and the flux linkage by 2 to 3 %
(``examples/electrodynamics/24_afpm_quasi3d.py``); :func:`verify_3d` measures it for the design at hand.

The back-EMF constant follows from the tooth linkage: with the magnet field of one tooth varying as the
fundamental of the pole pattern, the peak EMF of one tooth coil per turn and per mechanical rad/s is
``p * lambda_tooth`` (``p`` pole pairs); :func:`estimate` and :func:`verify_3d` give it as ``ke_tooth_V_s``.

:func:`field_view_of` gives a 3-D view of ``|B|`` on the magnets, the iron and the gap layers of a solved model
(:func:`pinneapple_tools.visualization.studio.fieldview.field_view`: a glTF and its description).
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .afpm import AFPM3D, AIR, AFPMMachine, quasi3d_slices

__all__ = ["estimate", "verify_3d", "field_view_of", "machine_from", "grid_for"]

MAX_STEPS = 480
SAMPLES_PER_PERIOD = 16       # rotor angles per cogging period (16 and 32 give the same peak-to-peak, 0.01 %)


def grid_for(machine: AFPMMachine, dr: float = 0.008, dz: float = 0.004, min_steps: int = 180,
             steps_per_period: int = 8) -> dict[str, Any]:
    """The coarse grid for a design: steps round the axis a multiple of twice the poles and twice the slots (pole
    and slot centres on the mesh's symmetry planes), at least ``min_steps`` and at least ``steps_per_period`` steps
    per cogging period. Raises when that needs more than 480 steps (a slot-pole combination with a very
    short cogging period). Magnet and tooth edges are shares of the cells they cut (``AFPM3D(edges="fraction")``),
    so a magnet arc or slot opening between two grid steps is the one asked for, not the nearest whole cells, and
    the rotor turns by any angle (:data:`SAMPLES_PER_PERIOD` angles per cogging period)."""
    period_steps = math.lcm(machine.slots, machine.poles)                  # cogging periods round the axis
    base = math.lcm(2 * machine.poles, 2 * machine.slots, 2 * period_steps)
    n = base * max(1, math.ceil(max(min_steps, steps_per_period * period_steps) / base))
    if n > MAX_STEPS:
        raise ValueError(f"{machine.slots} slots and {machine.poles} poles need {n} steps round the axis "
                         f"(cogging period {360 / period_steps:g} deg); the designer stops at {MAX_STEPS}")
    return {"n_theta": n, "dr": dr, "dz": dz, "edges": "fraction"}


def machine_from(params: dict[str, Any] | None = None) -> AFPMMachine:
    """An :class:`AFPMMachine` from a dict of its fields (missing ones keep their defaults)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(AFPMMachine)}
    unknown = set(params or {}) - names
    if unknown:
        raise ValueError(f"unknown machine parameters: {sorted(unknown)}")
    return AFPMMachine(**(params or {}))


def _angles(model: AFPM3D) -> np.ndarray:
    """Rotor angles over one cogging period: :data:`SAMPLES_PER_PERIOD` of them with fractional edges (the magnet
    edges fall inside cells), else the even grid steps (the grid maps onto itself under two-step rotations)."""
    if model.edges == "fraction":
        return np.arange(SAMPLES_PER_PERIOD) * model.machine.cogging_period / SAMPLES_PER_PERIOD
    return np.arange(0, round(model.machine.cogging_period / model.step), 2) * model.step


def _summary(m: AFPMMachine, thetas, cog, lam, t0: float, tets: int) -> dict[str, Any]:
    cog = np.asarray(cog, float)
    return {"thetas_deg": np.degrees(thetas).round(6).tolist(), "cogging_Nm": cog.tolist(),
            "cogging_pkpk_Nm": float(cog.max() - cog.min()), "tooth_linkage_Wb": float(abs(lam)),
            "ke_tooth_V_s": float(m.poles / 2 * abs(lam)), "tets": int(tets),
            "seconds": time.perf_counter() - t0}


def estimate(params: dict[str, Any] | None = None, n_slices: int = 1, grid: dict[str, float] | None = None,
             keep: bool = False) -> dict[str, Any]:
    """Quasi-3D: ``n_slices`` radial slices of the active annulus, each with flux-parallel faces, on a grid of
    ``grid`` (default :func:`grid_for`: 8 mm radially, 4 mm axially, 192 steps for 12 slots and 8 poles). ``keep``: also return the slice models and the solution at the
    linkage angle (for :func:`field_view_of`)."""
    t0 = time.perf_counter()
    m = machine_from(params)
    grid = grid or grid_for(m)
    slices = quasi3d_slices(m, n_slices, air_z=0.015, **grid)
    thetas = _angles(slices[0])
    link = math.pi / m.slots                          # magnet 0 centred on tooth 0
    cog = np.zeros(len(thetas))
    lam, tets, kept = 0.0, 0, []
    for q in slices:
        cog += q.cogging(thetas)
        s = q.solve(link)
        lam += q.tooth_linkage(s)
        tets += len(q.mesh.tets)
        q.release()
        if keep:
            kept.append((q, s))
    out = {"machine": _fields(m), "model": "quasi-3D", "n_slices": n_slices, "grid": grid,
           **_summary(m, thetas, cog, lam, t0, tets)}
    if keep:
        out["_models"] = kept
    return out


def verify_3d(params: dict[str, Any] | None = None, grid: dict[str, float] | None = None,
              against: dict[str, Any] | None = None, keep: bool = False) -> dict[str, Any]:
    """Full 3-D on the same grid as :func:`estimate` (air all round); with ``against`` (an :func:`estimate`
    result), the ratios estimate / 3-D of the cogging peak-to-peak and the tooth flux linkage."""
    t0 = time.perf_counter()
    m = machine_from(params)
    grid = grid or grid_for(m)
    model = AFPM3D(m, air_r=0.015, air_z=0.015, **grid)
    thetas = _angles(model)
    cog = model.cogging(thetas)
    s = model.solve(math.pi / m.slots)
    lam = model.tooth_linkage(s)
    out = {"machine": _fields(m), "model": "3-D", "grid": grid,
           **_summary(m, thetas, cog, lam, t0, len(model.mesh.tets))}
    model.release()
    if against is not None:
        out["estimate_over_3d"] = {"cogging_pkpk": against["cogging_pkpk_Nm"] / out["cogging_pkpk_Nm"],
                                   "tooth_linkage": against["tooth_linkage_Wb"] / out["tooth_linkage_Wb"]}
    if keep:
        out["_models"] = [(model, s)]
    return out


def field_view_of(model: AFPM3D, sol, title: str = "", max_faces: int | None = 200_000):
    """3-D view of ``|B|`` (T) on the magnets, the iron and the gap layers of ``model`` at the solution ``sol``:
    a :class:`~pinneapple_tools.visualization.studio.fieldview.FieldView` (``.glb()`` and ``.meta``)."""
    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_tools.visualization.studio.fieldview import field_view

    m = model.mesh
    gap = model.zkind[model.cell[2]] == "gap"
    sel = (m.region != AIR) | (gap & (np.hypot(m.centroids[:, 0], m.centroids[:, 1]) > model.machine.r_in)
                               & (np.hypot(m.centroids[:, 0], m.centroids[:, 1]) < model.machine.r_out))
    tets = m.tets[sel]
    used, inv = np.unique(tets, return_inverse=True)
    mesh = Mesh(points=m.nodes[used] * 1e3, blocks=[CellBlock("tetra", inv.reshape(-1, 4))])
    B = np.linalg.norm(sol.B[sel], axis=1)
    Bz = sol.B[sel, 2]
    return field_view(mesh, [{"name": "B", "values": B, "label": "|B|", "unit": "T", "where": "cell"},
                             {"name": "Bz", "values": Bz, "label": "B axial", "unit": "T", "where": "cell"}],
                      title=title or f"AFPM ({model.machine.topology}), |B| at no load", length_unit="mm",
                      max_faces=max_faces)


def _fields(m: AFPMMachine) -> dict[str, Any]:
    import dataclasses
    return dataclasses.asdict(m)
