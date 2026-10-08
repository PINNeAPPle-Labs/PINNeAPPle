"""3D finite-volume thermal model of a populated PCB.

* The board is resolved layer by layer (one cell per copper or dielectric
  layer through the thickness, a structured grid in-plane), each layer with
  its own in-plane / through-plane conductivity (``pcb_thermal.Layer``).
* Components use the JEDEC two-resistor compact model: a junction node
  tied to the board cells under the footprint through theta_JB (split by
  overlap area) and to the air through theta_JC in series with the case-top
  convection. The footprint shields that part of the board face from the air.
* Thermal vias add plated copper through the dielectric under a component.
* Board faces lose heat by convection + radiation with coefficients from
  ``pcb_thermal.convection_coefficients``; those depend on the surface
  temperature, so the solve is iterated to a fixed point.
* Optional chassis contact along the board edges (card-edge / wedge-lock
  cooling) at a fixed temperature.

Unknowns are temperatures above ambient; the sparse system is solved with
AMG-preconditioned conjugate gradients (direct solve as fallback). Energy balance (heat to air + chassis = power) is returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from pinneapple_physics.closed_form.pcb_thermal import (
    K_CU,
    PACKAGES,
    Environment,
    Layer,
    convection_coefficients,
    stackup,
    via_copper_fraction,
)

MM = 1e-3


@dataclass
class Component:
    name: str
    x_mm: float                      # centre on the board
    y_mm: float
    power_w: float
    package: str = "QFN-32 5x5"
    side: str = "top"
    tj_max_c: float = 125.0
    theta_jb: Optional[float] = None  # K/W, default from package
    theta_jc: Optional[float] = None  # K/W (junction to case top)
    w_mm: Optional[float] = None
    d_mm: Optional[float] = None
    h_mm: Optional[float] = None
    vias: int = 0                    # thermal vias under the part
    via_drill_mm: float = 0.3
    via_plating_um: float = 25.0

    def resolved(self) -> "Component":
        p = PACKAGES.get(self.package, PACKAGES["QFN-32 5x5"])
        return Component(**{**self.__dict__,
                            "theta_jb": self.theta_jb if self.theta_jb is not None else p["theta_jb"],
                            "theta_jc": self.theta_jc if self.theta_jc is not None else p["theta_jc"],
                            "w_mm": self.w_mm or p["w_mm"], "d_mm": self.d_mm or p["d_mm"],
                            "h_mm": self.h_mm or p["h_mm"]})


@dataclass
class Board:
    width_mm: float = 160.0
    depth_mm: float = 100.0          # vertical extent when mounted vertically / flow length
    n_copper: int = 4
    thickness_mm: float = 1.6
    outer_oz: float = 1.0
    inner_oz: float = 1.0
    outer_coverage: float = 0.3
    inner_coverage: float = 0.9
    chassis_edges: bool = False      # edges clamped to a chassis
    chassis_t_c: Optional[float] = None  # default: ambient
    edge_h_w_m2k: float = 2000.0     # contact conductance of the clamped edges

    def layers(self) -> List[Layer]:
        return stackup(self.n_copper, self.thickness_mm, self.outer_oz, self.inner_oz,
                       self.outer_coverage, self.inner_coverage)


def _overlap(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)


def validate(board: Board, comps: List[Component]) -> List[str]:
    errs = []
    if board.width_mm <= 0 or board.depth_mm <= 0 or board.thickness_mm <= 0:
        errs.append("board dimensions must be positive")
    for c in comps:
        r = c.resolved()
        if r.power_w < 0:
            errs.append(f"{c.name}: negative power")
        if (r.x_mm - r.w_mm / 2 < -1e-9 or r.x_mm + r.w_mm / 2 > board.width_mm + 1e-9
                or r.y_mm - r.d_mm / 2 < -1e-9 or r.y_mm + r.d_mm / 2 > board.depth_mm + 1e-9):
            errs.append(f"{c.name}: footprint extends beyond the board")
        if r.side not in ("top", "bottom"):
            errs.append(f"{c.name}: side must be 'top' or 'bottom'")
        if r.theta_jb <= 0 or r.theta_jc <= 0:
            errs.append(f"{c.name}: theta_JB and theta_JC must be positive")
    return errs


def overlaps(comps: List[Component]) -> List[str]:
    """Same-side component pairs whose footprints overlap (a layout error the
    heat model would otherwise silently superpose)."""
    rs = [c.resolved() for c in comps]
    out = []
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            if (a.side == b.side and abs(a.x_mm - b.x_mm) < (a.w_mm + b.w_mm) / 2 - 1e-9
                    and abs(a.y_mm - b.y_mm) < (a.d_mm + b.d_mm) / 2 - 1e-9):
                out.append(f"{a.name} and {b.name} overlap on the {a.side} side -- move one of them.")
    return out


def _linear_solve(M: sps.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Algebraic multigrid-preconditioned CG (thin layered boards are strongly
    anisotropic, which AMG handles well); direct solve if pyamg is missing."""
    try:
        import pyamg
    except ImportError:
        return spla.spsolve(M.tocsc(), b)
    ml = pyamg.smoothed_aggregation_solver(M, symmetry="symmetric", max_coarse=500)
    x = ml.solve(b, tol=1e-10, accel="cg", maxiter=200)
    if not np.all(np.isfinite(x)) or np.linalg.norm(M @ x - b) > 1e-6 * np.linalg.norm(b):
        x = spla.spsolve(M.tocsc(), b)
    return x


class _Model:
    """Discretisation of one board + component set on one grid."""

    def __init__(self, board: Board, comps: List[Component], env: Environment, n_cells: int,
                 k_xy_scale: float):
        self.board, self.comps, self.env = board, comps, env
        layers = self.layers = board.layers()
        nz = self.nz = len(layers)
        W, D = self.W, self.D = board.width_mm * MM, board.depth_mm * MM
        nx = self.nx = max(8, int(round(n_cells * np.sqrt(W / D))))
        ny = self.ny = max(8, int(round(n_cells * n_cells / nx)))
        dx, dy = self.dx, self.dy = W / nx, D / ny
        xe, ye = np.linspace(0, W, nx + 1), np.linspace(0, D, ny + 1)
        A = self.A = dx * dy
        t = self.t = np.array([la.thickness_m for la in layers])
        kxy = np.array([la.k_xy() for la in layers]) * k_xy_scale
        kz0 = np.array([la.k_z() for la in layers])
        nb = self.nb = nx * ny * nz
        idx = self.idx = np.arange(nb).reshape(nz, ny, nx)
        nc = self.nc = len(comps)

        self.fp = []
        self.cover = {"top": np.zeros((ny, nx)), "bottom": np.zeros((ny, nx))}
        kz = self.kz = np.broadcast_to(kz0[:, None, None], (nz, ny, nx)).copy()
        for c in comps:
            ov = np.outer(_overlap(ye, (c.y_mm - c.d_mm / 2) * MM, (c.y_mm + c.d_mm / 2) * MM),
                          _overlap(xe, (c.x_mm - c.w_mm / 2) * MM, (c.x_mm + c.w_mm / 2) * MM))
            self.fp.append(ov)
            self.cover[c.side] += ov / A
            if c.vias > 0:
                f = via_copper_fraction(c.vias, c.via_drill_mm, c.via_plating_um,
                                        c.w_mm * c.d_mm * MM * MM)
                frac = ov / A * f
                for k, la in enumerate(layers):
                    if la.kind == "dielectric":
                        kz[k] += frac * (K_CU - kz[k])
        for s in self.cover:
            np.clip(self.cover[s], 0.0, 1.0, out=self.cover[s])

        rows, cols, vals = [], [], []
        diag = self.diag = np.zeros(nb + nc)

        def link(a, b, g):
            a, b, g = a.ravel(), b.ravel(), np.broadcast_to(g, a.shape).ravel()
            rows.extend([a, b])
            cols.extend([b, a])
            vals.extend([-g, -g])
            np.add.at(diag, a, g)
            np.add.at(diag, b, g)

        gx = (kxy * t * dy / dx)[:, None, None]
        gy = (kxy * t * dx / dy)[:, None, None]
        link(idx[:, :, :-1], idx[:, :, 1:], np.broadcast_to(gx, (nz, ny, nx - 1)))
        link(idx[:, :-1, :], idx[:, 1:, :], np.broadcast_to(gy, (nz, ny - 1, nx)))
        rz = t[:, None, None] / (2 * kz * A)
        link(idx[:-1], idx[1:], 1.0 / (rz[:-1] + rz[1:]))

        self.rhs = np.zeros(nb + nc)
        self.face_layer = {"top": 0, "bottom": nz - 1}
        for ci, c in enumerate(comps):
            k = self.face_layer[c.side]
            m = self.fp[ci] > 0
            ai = self.fp[ci][m]
            area_fp = c.w_mm * c.d_mm * MM * MM
            g = 1.0 / (c.theta_jb * area_fp / ai + t[k] / (2 * kz[k][m] * ai))
            link(np.full(int(m.sum()), nb + ci), idx[k][m], g)
            self.rhs[nb + ci] = c.power_w

        self.chassis_theta = ((board.chassis_t_c if board.chassis_t_c is not None
                               else env.t_ambient_c) - env.t_ambient_c)
        self.g_edge = np.zeros((nz, ny, nx))
        if board.chassis_edges:
            for k in range(nz):
                self.g_edge[k, :, 0] += board.edge_h_w_m2k * t[k] * dy
                self.g_edge[k, :, nx - 1] += board.edge_h_w_m2k * t[k] * dy
                self.g_edge[k, 0, :] += board.edge_h_w_m2k * t[k] * dx
                self.g_edge[k, ny - 1, :] += board.edge_h_w_m2k * t[k] * dx
        self.base = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                    np.concatenate(cols))), shape=(nb + nc, nb + nc))

    def solve_h(self, h: Dict[str, float]) -> Dict[str, Any]:
        d, b = self.diag.copy(), self.rhs.copy()
        g_face = {}
        for s, k in self.face_layer.items():
            g_face[s] = (1 - self.cover[s]) * self.A / (self.t[k] / (2 * self.kz[k]) + 1.0 / h[s])
            d[self.idx[k].ravel()] += g_face[s].ravel()
        d[:self.nb] += self.g_edge.ravel()
        b[:self.nb] += (self.g_edge * self.chassis_theta).ravel()
        g_air = np.array([1.0 / (c.theta_jc + 1.0 / (h[c.side] * c.w_mm * c.d_mm * MM * MM))
                          for c in self.comps])
        d[self.nb:] += g_air
        theta = _linear_solve((self.base + sps.diags(d)).tocsr(), b)
        th = theta[:self.nb].reshape(self.nz, self.ny, self.nx)
        ts = {}
        for s, k in self.face_layer.items():
            wgt = 1 - self.cover[s]
            ts[s] = self.env.t_ambient_c + float(np.sum(th[k] * wgt) / max(wgt.sum(), 1e-12))
        return {"theta": theta, "th": th, "g_face": g_face, "g_air": g_air, "ts": ts, "h": h}


def _h_for(model: _Model, ts: Dict[str, float], h_scale: float):
    info = convection_coefficients(model.W, model.D, (ts["top"], ts["bottom"]), model.env)
    return {"top": info["h_top"] * h_scale, "bottom": info["h_bottom"] * h_scale}, info


def solve(board: Board, comps: List[Component], env: Environment, *, n_cells: int = 60,
          h_override: Optional[Dict[str, float]] = None, k_xy_scale: float = 1.0,
          h_scale: float = 1.0, max_iter: int = 40, coarse_cells: int = 24) -> Dict[str, Any]:
    """Steady temperatures. ``h_override`` fixes {'top','bottom'} coefficients
    (tests / calibration); ``k_xy_scale`` / ``h_scale`` scale the in-plane
    conductivity and the surface coefficients (calibration / uncertainty).

    The surface coefficients depend on the mean face temperatures only, so
    their fixed point is found on a coarse grid, then the fine grid is solved
    with them and the fixed point re-checked (one more fine solve if needed).
    """
    errs = validate(board, comps)
    if errs:
        raise ValueError("; ".join(errs))
    comps = [c.resolved() for c in comps]
    Ta = env.t_ambient_c
    info: Dict[str, Any] = {"warnings": []}
    iters, converged = 0, True
    if h_override:
        h = {"top": h_override["top"], "bottom": h_override["bottom"]}
    else:
        coarse = _Model(board, comps, env, min(coarse_cells, n_cells), k_xy_scale)
        ts = {"top": Ta + 15.0, "bottom": Ta + 15.0}
        converged = False
        for iters in range(1, max_iter + 1):  # noqa: B007 (reported after the loop)
            h, info = _h_for(coarse, ts, h_scale)
            new = coarse.solve_h(h)["ts"]
            if max(abs(new[s] - ts[s]) for s in ts) < 0.005:
                converged = True
                ts = new
                break
            ts = {s: 0.5 * (new[s] + ts[s]) for s in ts}
        h, info = _h_for(coarse, ts, h_scale)

    model = _Model(board, comps, env, n_cells, k_xy_scale)
    sol = model.solve_h(h)
    if not h_override:
        h2, info2 = _h_for(model, sol["ts"], h_scale)
        if max(abs(h2[s] - h[s]) / h[s] for s in h) > 0.01:       # re-check on the fine grid
            h, info = h2, info2
            sol = model.solve_h(h)
    return _post(model, sol, info, iters, converged)


def _post(model: _Model, sol: Dict[str, Any], info: Dict[str, Any], iters: int,
          converged: bool) -> Dict[str, Any]:
    comps, Ta = model.comps, model.env.t_ambient_c
    th, theta, g_face, g_air, h = sol["th"], sol["theta"], sol["g_face"], sol["g_air"], sol["h"]
    tj = theta[model.nb:]
    fl = model.face_layer
    q_face = {s: float(np.sum(g_face[s] * th[k])) for s, k in fl.items()}
    q_comp_air = g_air * tj
    q_chassis = float(np.sum(model.g_edge * (th - model.chassis_theta)))
    p_total = float(sum(c.power_w for c in comps))
    q_out = q_face["top"] + q_face["bottom"] + float(q_comp_air.sum()) + q_chassis

    comp_out = []
    for ci, c in enumerate(comps):
        w = model.fp[ci]
        t_board = Ta + float(np.sum(th[fl[c.side]] * w) / np.sum(w))
        t_j = Ta + float(tj[ci])
        comp_out.append({
            "name": c.name, "package": c.package, "side": c.side, "power_w": c.power_w,
            "x_mm": c.x_mm, "y_mm": c.y_mm, "w_mm": c.w_mm, "d_mm": c.d_mm, "h_mm": c.h_mm,
            "theta_jb": c.theta_jb, "theta_jc": c.theta_jc, "vias": c.vias,
            "t_junction_c": t_j,
            "t_case_top_c": t_j - float(q_comp_air[ci]) * c.theta_jc,
            "t_board_under_c": t_board,
            "tj_max_c": c.tj_max_c,
            "margin_c": c.tj_max_c - t_j,
            "heat_to_board_pct": 100 * (1 - float(q_comp_air[ci]) / c.power_w) if c.power_w > 0 else None,
        })
    return {
        "components": comp_out,
        "t_top_c": th[0] + Ta,
        "t_bottom_c": th[-1] + Ta,
        "grid": {"nx": model.nx, "ny": model.ny, "nz": model.nz,
                 "dx_mm": model.dx / MM, "dy_mm": model.dy / MM},
        "h": {"top": h["top"], "bottom": h["bottom"],
              "top_conv": info.get("h_conv_top"), "top_rad": info.get("h_rad_top"),
              "bottom_conv": info.get("h_conv_bottom"), "bottom_rad": info.get("h_rad_bottom"),
              "regime_top": info.get("regime_top"), "regime_bottom": info.get("regime_bottom")},
        "heat_split_w": {"board_top_face": q_face["top"], "board_bottom_face": q_face["bottom"],
                         "component_tops": float(q_comp_air.sum()), "chassis": q_chassis},
        "power_w": p_total,
        "energy_balance_rel_error": abs(q_out - p_total) / p_total if p_total > 0 else 0.0,
        "iterations": iters,
        "converged": converged,
        "correlation_warnings": info.get("warnings", []),
        "mean_surface_c": dict(sol["ts"]),
        "layers": [{"name": la.name, "kind": la.kind, "thickness_um": la.thickness_m * 1e6,
                    "coverage": la.coverage if la.kind == "copper" else None,
                    "k_xy": la.k_xy(), "k_z": la.k_z()} for la in model.layers],
    }
