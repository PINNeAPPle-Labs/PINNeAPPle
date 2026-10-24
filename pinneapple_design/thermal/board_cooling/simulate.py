"""Reference simulation of a cooled board: the 3D board model and the heat sink model, coupled.

* Board: :func:`pinneapple_physics.pcb.solve` (layered 3D finite volumes, JEDEC two-resistor packages, convection +
  radiation from both faces at the fan air speed).
* Heat sink on the hot part: :func:`pinneapple_design.thermal.heatsink.evaluate_design` (3D finite-volume base,
  fin correlations, pressure drop) gives the case-to-air resistance ``R = (T_source - T_air) / Q`` (interface
  material + spreading + fins) for the heat ``Q`` that actually goes up through the case.
* Coupling: ``Q`` depends on ``R`` (the rest goes into the board) and, in still air, ``R`` depends on ``Q``; the two
  are iterated to a fixed point (relative change of ``Q`` below ``tol``).

``simulate(..., pessimistic=True)`` repeats the coupled solve with the convection band of both models (board surface
coefficients x 0.8, heat-sink coefficient x 0.85: the H_BAND of each engine, typical correlation accuracy) so a design
can be required to meet its limits even with pessimistic cooling.

Board-top temperatures are returned on a fixed grid (``MAP_SHAPE``) whatever the solver grid, so solutions of
different fidelity can be compared point by point and used as snapshots of a field surrogate.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from pinneapple_design.thermal.heatsink import DesignInput, OperatingInput, evaluate_design
from pinneapple_physics.closed_form.plate_fin_heatsink import evaluate as hs_network
from pinneapple_physics.pcb import Environment, solve
from pinneapple_physics.pcb.engine import H_BAND as BOARD_BAND

from .mission import CoolingDesign, CoolingMission

MAP_SHAPE = (40, 64)                 # (ny, nx) board-top map of every result
FIDELITY = {"train": {"n_cells": 40, "hs_grid": 28},       # snapshots of the surrogate
            "verify": {"n_cells": 80, "hs_grid": 56}}      # independent, finer verification


def _to_map(t_top: np.ndarray, board) -> np.ndarray:
    """Cell-centred board-top temperatures -> the fixed MAP_SHAPE grid (bilinear, edges held)."""
    from scipy.interpolate import RegularGridInterpolator
    ny, nx = t_top.shape
    yc = (np.arange(ny) + 0.5) * board.depth_mm / ny
    xc = (np.arange(nx) + 0.5) * board.width_mm / nx
    f = RegularGridInterpolator((yc, xc), t_top, bounds_error=False, fill_value=None)
    Y, X = np.meshgrid(*map_axes(board), indexing="ij")
    pts = np.stack([np.clip(Y, yc[0], yc[-1]), np.clip(X, xc[0], xc[-1])], -1)
    return f(pts)


def map_axes(board) -> tuple[np.ndarray, np.ndarray]:
    """(y, x) cell centres of MAP_SHAPE in mm."""
    ny, nx = MAP_SHAPE
    return (np.arange(ny) + 0.5) * board.depth_mm / ny, (np.arange(nx) + 0.5) * board.width_mm / nx


def sink_mass_and_air(mission: CoolingMission, d: CoolingDesign) -> dict[str, float]:
    """Closed-form heat-sink quantities that do not need the coupled solve: mass, pressure drop and air power
    (``pinneapple_physics.closed_form.plate_fin_heatsink``)."""
    src = mission.hot_component()
    di = _design_input(d)
    op = OperatingInput(src.power_w, mission.t_ambient_c, d.air_velocity_m_s, src.w_mm, src.d_mm,
                        tim_k_mm2_w=mission.tim_k_mm2_w)
    net = hs_network(di.geometry(), op.operating())
    return {"mass_g": net["mass_kg"] * 1e3, "pressure_drop_pa": net["pressure_drop_pa"],
            "fan_power_w": net["fan_power_w"], "airflow_m3_h": net["airflow_m3_h"]}


def _design_input(d: CoolingDesign) -> DesignInput:
    return DesignInput(d.base_width_mm, d.base_depth_mm, d.base_thickness_mm, int(d.n_fins), d.fin_thickness_mm,
                       d.fin_height_mm, d.material)


def _coupled(mission: CoolingMission, d: CoolingDesign | None, *, n_cells: int, hs_grid: int, h_board: float,
             hs_pessimistic: bool, tol: float, max_iter: int) -> dict[str, Any]:
    src = mission.hot_component()
    V = 0.0 if d is None else d.air_velocity_m_s
    env = Environment(mission.t_ambient_c, V)
    comps = [copy.copy(c) for c in mission.components]
    hot = next(c for c in comps if c.name == mission.hot)
    q = src.power_w * 0.8
    hist: list[dict[str, float]] = []
    sink = None
    converged = d is None
    for _ in range(1 if d is None else max_iter):
        if d is not None:
            op = OperatingInput(max(q, 1e-3), mission.t_ambient_c, V, src.w_mm, src.d_mm,
                                tim_k_mm2_w=mission.tim_k_mm2_w, t_limit_c=mission.tj_limit_c)
            sink = evaluate_design(_design_input(d), op, grid=hs_grid, map_size=24)
            k = sink["kpis"]
            t_src = k["t_source_band_c"][1] if hs_pessimistic else k["t_source_max_c"]
            hot.r_top_k_w = (t_src - mission.t_ambient_c) / max(q, 1e-3)
        r = solve(mission.board, comps, env, n_cells=n_cells, h_scale=h_board)
        if d is None:
            break
        c = next(c for c in r["components"] if c["name"] == mission.hot)
        q_new = c["power_w"] * (1 - c["heat_to_board_pct"] / 100)
        hist.append({"q_sink_w": q_new, "r_sink_k_w": hot.r_top_k_w})
        if abs(q_new - q) <= tol * src.power_w:
            q, converged = q_new, True
            break
        q = q_new
    return {"board": r, "sink": sink, "q_sink_w": q if d is not None else 0.0, "history": hist,
            "converged": converged}


def simulate(mission: CoolingMission, design: CoolingDesign | None, *, fidelity: str = "train",
             pessimistic: bool = True, tol: float = 1e-4, max_iter: int = 12) -> dict[str, Any]:
    """Junction temperatures, board-top map, heat paths and checks of ``design`` (None: the bare board in still
    air, the starting point). ``fidelity``: a key of FIDELITY or ``{"n_cells", "hs_grid"}``."""
    fid = FIDELITY[fidelity] if isinstance(fidelity, str) else dict(fidelity)
    t0 = time.perf_counter()
    nom = _coupled(mission, design, n_cells=fid["n_cells"], hs_grid=fid["hs_grid"], h_board=1.0,
                   hs_pessimistic=False, tol=tol, max_iter=max_iter)
    pes = (_coupled(mission, design, n_cells=fid["n_cells"], hs_grid=fid["hs_grid"], h_board=1 - BOARD_BAND,
                    hs_pessimistic=True, tol=tol, max_iter=max_iter) if pessimistic else None)
    r = nom["board"]
    names = [c["name"] for c in r["components"]]
    tj = {c["name"]: c["t_junction_c"] for c in r["components"]}
    out: dict[str, Any] = {
        "design": None if design is None else design.to_dict(),
        "tj_c": tj,
        "tj_pessimistic_c": None if pes is None else {c["name"]: c["t_junction_c"]
                                                       for c in pes["board"]["components"]},
        "map_c": _to_map(r["t_top_c"], mission.board),
        "board_max_c": float(max(r["t_top_c"].max(), r["t_bottom_c"].max())),
        "energy_balance_board": r["energy_balance_rel_error"],
        "heat_split_w": {**r["heat_split_w"], "through_heat_sink": nom["q_sink_w"]},
        "q_sink_w": nom["q_sink_w"],
        "coupling_iterations": len(nom["history"]),
        "coupling_converged": nom["converged"] and (pes is None or pes["converged"]),
        "h_board": r["h"], "grid": r["grid"], "fidelity": fid,
        "names": names,
    }
    if design is not None:
        s = nom["sink"]
        out.update(sink_mass_and_air(mission, design))
        out["sink"] = {"r_case_to_air_k_w": (s["kpis"]["t_source_max_c"] - mission.t_ambient_c) / nom["q_sink_w"],
                       "resistances_k_w": s["resistances_k_w"], "checks": s["checks"],
                       "warnings": s["warnings"], "mode": s["kpis"]["mode"], "h_w_m2k": s["kpis"]["h_w_m2k"],
                       "fin_efficiency": s["kpis"]["fin_efficiency"], "field3d": s["field3d"],
                       "t_base_max_c": s["kpis"]["t_source_max_c"]}
    out["seconds"] = time.perf_counter() - t0
    return out


def response(mission: CoolingMission, result: dict[str, Any]) -> np.ndarray:
    """The scalar responses a surrogate learns, in a fixed order: junction temperatures (nominal, then
    pessimistic) of ``mission.components``."""
    names = [c.name for c in mission.components]
    pes = result["tj_pessimistic_c"] or result["tj_c"]
    return np.array([result["tj_c"][n] for n in names] + [pes[n] for n in names], float)


def _sim_row(args):
    mission, x, fidelity = args
    r = simulate(mission, CoolingDesign.from_vector(x, mission.material), fidelity=fidelity)
    return x, response(mission, r), r["map_c"], r["seconds"], r["energy_balance_board"], r["coupling_converged"]


def _one_thread() -> None:
    """Pool initializer: one BLAS / OpenMP thread per worker (the workers already fill the cores)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def simulate_many(mission: CoolingMission, X: np.ndarray, *, fidelity: str = "train", workers: int = 4
                  ) -> dict[str, np.ndarray]:
    """Reference solves of many designs in parallel: responses (n, 2 n_parts), board-top maps (n, ny, nx), wall
    seconds per solve, board energy-balance errors and coupling convergence flags."""
    jobs = [(mission, np.asarray(x, float), fidelity) for x in X]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_one_thread) as pool:
            rows = pool.map(_sim_row, jobs)
    else:
        rows = [_sim_row(j) for j in jobs]
    return {"X": np.array([r[0] for r in rows]), "Y": np.array([r[1] for r in rows]),
            "maps": np.array([r[2] for r in rows]), "seconds": np.array([r[3] for r in rows]),
            "energy_balance": np.array([r[4] for r in rows]), "converged": np.array([r[5] for r in rows])}
