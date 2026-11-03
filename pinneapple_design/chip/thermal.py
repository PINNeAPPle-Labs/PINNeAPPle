"""Die and package thermal model of a flip-chip: a 3-D compact conduction network (cell-centred finite volumes, the
assembler of :func:`pinneapple_condense.fv.assemble`) from the board, through the C4 layer, the die, the TIM and the
lid, to the heat sink, with the leakage-temperature feedback solved to its fixed point.

Assumptions (stated in every result): the lid and the heat sink have the die's footprint (no lateral spreading
beyond the die, a conservative choice); the heat sink is a uniform conductance ``1 / (R_sink A_die)`` on the lid
top; the board path is a uniform conductance on the substrate bottom; the sides are adiabatic; conductivities are
constant (silicon's falls by about 20 % from 25 to 100 C, not modelled); heat is generated on the die's active
(bottom, flip-chip) cell row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .floorplan import Floorplan

__all__ = ["StackLayer", "DieStack", "default_stack", "solve_thermal", "slot_responses", "stack_closed_form"]


@dataclass(frozen=True)
class StackLayer:
    name: str
    thickness_m: float
    k_w_mk: float
    nz: int = 1


@dataclass(frozen=True)
class DieStack:
    """Layers bottom to top; ``active``: the layer whose bottom cell row generates the heat; ``contacts``: (lower,
    upper, m^2 K/W) interface resistances; ``r_sink_k_w``: heat-sink resistance (lid top to ambient, K/W) and
    ``r_board_k_w``: substrate bottom to ambient through the board (K/W)."""
    layers: tuple[StackLayer, ...]
    active: str = "die"
    contacts: tuple[tuple[str, str, float], ...] = ()
    r_sink_k_w: float = 0.25
    r_board_k_w: float = 10.0
    t_ambient_c: float = 45.0
    assumptions: tuple[str, ...] = field(default=(
        "lid and heat sink with the die footprint (no spreading beyond the die)",
        "uniform heat-sink and board conductances", "adiabatic sides", "constant conductivities",
        "heat generated on the die's active (bottom, flip-chip) cell row"))

    def refined(self, factor: int = 2) -> DieStack:
        return replace(self, layers=tuple(replace(L, nz=L.nz * factor) for L in self.layers))


def default_stack() -> DieStack:
    """Organic substrate, C4 bumps in underfill (effective k), 500 um silicon, 50 um TIM, 2 mm copper lid.
    Representative room-temperature values (Incropera, Table A.1, for Si and Cu; datasheet ranges for the rest)."""
    return DieStack((StackLayer("substrate", 0.8e-3, 2.0, 2), StackLayer("c4", 80e-6, 1.5, 1),
                     StackLayer("die", 0.5e-3, 148.0, 4), StackLayer("tim", 50e-6, 4.0, 1),
                     StackLayer("lid", 2.0e-3, 400.0, 2)),
                    contacts=(("die", "tim", 5e-6), ("tim", "lid", 5e-6)))


def _grid(stack: DieStack):
    dz = np.concatenate([np.full(L.nz, L.thickness_m / L.nz) for L in stack.layers])
    kz = np.concatenate([np.full(L.nz, L.k_w_mk) for L in stack.layers])
    R = np.zeros(dz.size - 1)
    bounds, k0 = {}, 0
    for L in stack.layers:
        bounds[L.name] = (k0, k0 + L.nz)
        k0 += L.nz
    for a, b, r in stack.contacts:
        if bounds[a][1] != bounds[b][0]:
            raise ValueError(f"contact {a}/{b}: layers are not adjacent")
        R[bounds[a][1] - 1] = r
    return dz, kz, R, bounds[stack.active][0]


class _Operator:
    """The assembled and factored conduction operator of a die grid (nx, ny) and a stack."""

    def __init__(self, stack: DieStack, die_w: float, die_h: float, nx: int, ny: int):
        from pinneapple_condense.fv import assemble, factor
        dz, kz, R, ka = _grid(stack)
        self.nx, self.ny, self.nz, self.ka = nx, ny, dz.size, ka
        A = die_w * die_h
        k = np.broadcast_to(kz[None, None, :], (nx, ny, dz.size)).copy()
        t0 = time.perf_counter()
        self.sys = assemble(k, dz, R, die_w / nx, 1.0 / (stack.r_sink_k_w * A), 1.0 / (stack.r_board_k_w * A),
                            dy=die_h / ny)
        self.lu = factor(self.sys.K)
        self.setup_s = time.perf_counter() - t0
        self.stack = stack

    def rise(self, q_active: np.ndarray) -> np.ndarray:
        """Temperature rise of every cell (nx, ny, nz) for heat ``q_active`` (W per cell) on the active row."""
        f = np.zeros((self.nx, self.ny, self.nz))
        f[:, :, self.ka] = q_active
        return self.lu.solve(f.ravel()).reshape(self.nx, self.ny, self.nz)


def _block_maps(fp: Floorplan, nx: int, ny: int, placement=None):
    pl = fp.placement if placement is None else placement
    return [fp.slot_mask(s, nx, ny) for s in pl]


def solve_thermal(fp: Floorplan, stack: DieStack | None = None, *, nx: int = 24, ny: int | None = None,
                  placement=None, leakage: bool = True, tol: float = 1e-6, max_iter: int = 100,
                  operator: _Operator | None = None) -> dict[str, Any]:
    """Steady junction temperatures with the leakage of every cell at its own temperature (fixed point).

    The fixed-point map ``T -> T_amb + K^-1 q(T)`` contracts when the loop gain ``max eig K^-1 diag(dq/dT)`` is
    below one; the measured ratio of successive corrections estimates it, and a gain at or above one is thermal
    runaway (no steady state)."""
    stack = stack or default_stack()
    ny = nx if ny is None else ny
    op = operator or _Operator(stack, fp.die_w_mm * 1e-3, fp.die_h_mm * 1e-3, nx, ny)
    masks = _block_maps(fp, nx, ny, placement)
    t_amb = stack.t_ambient_c
    dyn = np.zeros((nx, ny))
    leak0 = np.zeros((nx, ny))
    beta = np.zeros((nx, ny))
    tref = np.full((nx, ny), 85.0)
    for b, m in zip(fp.blocks, masks, strict=True):
        n = int(m.sum())
        dyn[m] = b.dynamic_w / n
        leak0[m] = b.leakage_w / n
        beta[m] = b.leak_beta
        tref[m] = b.t_ref_c
    Tj = np.full((nx, ny), t_amb)
    hist, ratios, prev = [], [], None
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        with np.errstate(over="ignore"):      # a runaway grows without bound: stopped below and reported
            q = dyn + (leak0 * np.exp(beta * (Tj - tref)) if leakage else leak0)
        T = t_amb + op.rise(q)
        new = T[:, :, op.ka]
        d = float(np.max(np.abs(new - Tj)))
        hist.append(d)
        if prev is not None and prev > 0:
            ratios.append(d / prev)
        prev, Tj = d, new
        if not leakage or d < tol:
            break
        if not np.isfinite(d) or d > 1e4:
            break
    gain = float(np.median(ratios[-5:])) if ratios else 0.0
    with np.errstate(over="ignore"):
        q = dyn + (leak0 * np.exp(beta * (Tj - tref)) if leakage else leak0)
    i, j = np.unravel_index(int(np.argmax(Tj)), Tj.shape)
    hot = next((b.name for b, m in zip(fp.blocks, masks, strict=True) if m[i, j]), None)
    P = float(q.sum())
    tmax = float(Tj.max())
    return {"T_junction": Tj, "T": T, "T_max_c": tmax, "T_mean_c": float(Tj.mean()), "hotspot_block": hot,
            "hotspot_xy_mm": [(i + 0.5) * fp.die_w_mm / nx, (j + 0.5) * fp.die_h_mm / ny],
            "power_w": P, "leakage_w": float((q - dyn).sum()), "theta_ja_k_w": (tmax - t_amb) / P,
            "block_t_max_c": {b.name: float(Tj[m].max()) for b, m in zip(fp.blocks, masks, strict=True)},
            "iterations": it, "loop_gain": gain, "runaway": bool(gain >= 1.0 or not np.isfinite(hist[-1])
                                                                 or hist[-1] >= tol and leakage),
            "converged": bool(hist[-1] < tol or not leakage), "seconds": time.perf_counter() - t0 + op.setup_s,
            "cells": int(nx * ny * op.nz), "grid": [nx, ny, op.nz], "assumptions": list(stack.assumptions),
            "_operator": op}


def slot_responses(fp: Floorplan, op: _Operator) -> np.ndarray:
    """Junction-temperature rise (n_slots, nx, ny) per watt spread uniformly over each slot: by superposition any
    placement's map (at fixed power) is a weighted sum of these, which makes the placement search cheap."""
    out = []
    for s in range(fp.n_slots):
        m = fp.slot_mask(s, op.nx, op.ny)
        out.append(op.rise(m / m.sum())[:, :, op.ka])
    return np.asarray(out)


def stack_closed_form(stack: DieStack, area_m2: float, power_w: float) -> dict[str, float]:
    """1-D die stack under uniform power: the series resistances from the source plane (the centre of the active
    cell row) up through the die, TIM, lid and sink, and down through the C4 layer, substrate and board, in parallel.
    ``T_j = T_amb + P R_up R_down / (R_up + R_down)``."""
    dz, kz, R, ka = _grid(stack)
    # node centres: resistance between successive centres, then the half cells and the boundary conductances
    between = [(dz[i] / 2 / kz[i] + dz[i + 1] / 2 / kz[i + 1] + R[i]) / area_m2 for i in range(dz.size - 1)]
    r_up = sum(between[ka:]) + dz[-1] / 2 / kz[-1] / area_m2 + stack.r_sink_k_w
    r_down = sum(between[:ka]) + dz[0] / 2 / kz[0] / area_m2 + stack.r_board_k_w
    r_par = r_up * r_down / (r_up + r_down)
    # the same stack with the source on the die's bottom face (continuous, no half cell): the discretisation offset
    r_up_face = sum(L.thickness_m / L.k_w_mk for L in stack.layers[[x.name for x in stack.layers].index(stack.active):]
                    ) / area_m2 + sum(r for *_, r in stack.contacts) / area_m2 + stack.r_sink_k_w
    r_down_face = sum(L.thickness_m / L.k_w_mk for L in stack.layers[:[x.name for x in stack.layers]
                                                                     .index(stack.active)]) / area_m2 \
        + stack.r_board_k_w
    return {"R_up_k_w": r_up, "R_down_k_w": r_down, "R_ja_k_w": r_par, "T_j_c": stack.t_ambient_c + power_w * r_par,
            "R_ja_face_k_w": r_up_face * r_down_face / (r_up_face + r_down_face),
            "fraction_up": r_down / (r_up + r_down)}
