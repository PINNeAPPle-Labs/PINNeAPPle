"""JSON-argument entry points for library capabilities whose own API takes objects (a soil, a wheel, a snapshot
set, a preset problem), so they can be library-call nodes. Each function builds the objects with the library's own
constructors and calls the library; nothing here computes physics itself.
"""
from __future__ import annotations

import importlib.util
import shutil
import time
from typing import Literal

import numpy as np

from pinneapple_core.workflows import StepFailed

PINN_ARCHITECTURES = Literal["modified_mlp", "vanilla_pinn", "siren", "hash_grid_mlp", "vpinn", "xtfc", "pielm",
                             "elm", "rbf", "neural_gp", "bench_res_mlp", "bench_fourier_mlp"]
SOILS = Literal["GRC-1", "JSC-1A", "Ding2011-TYII", "FJS-1/Hurrell2025", "KLS-1", "Apollo-lunar"]
MESHLESS_DOMAINS = Literal["square", "lshape", "disk", "annulus", "plate_hole"]


def _solver(name: str, **kw):
    """A solver of the numerical-solver registry (all solver modules registered first)."""
    from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all
    register_all()
    return SolverRegistry.build(name, **kw)


def _sol_summary(sol) -> dict:
    info = {k: v for k, v in (sol.info or {}).items() if isinstance(v, (int, float, str, bool))}
    return {"method": sol.method, "wall_time_s": float(sol.wall_time_s), **info}


def solve_preset(preset: str = "burgers_1d", method: Literal["analytic", "reference", "pinn", "fem"] = "analytic",
                 architecture: PINN_ARCHITECTURES = "modified_mlp", epochs: int = 50, hidden_dim: int = 32,
                 n_layers: int = 3, n_collocation: int = 512, seed: int = 0) -> dict:
    """``pp.solve`` on a library preset; for ``pinn`` the architecture is any PINN variant of the model registry."""
    import pinneapple as pp
    if preset not in pp.list_presets():
        raise StepFailed(f"unknown preset {preset!r}")
    opts = {}
    if method == "pinn":
        opts = dict(architecture=architecture, epochs=epochs, hidden_dim=hidden_dim, n_layers=n_layers,
                    n_collocation=n_collocation, seed=seed)
    return _sol_summary(pp.solve(preset, method, **opts))


def compare_preset(preset: str = "burgers_1d", methods: list = ("analytic", "pinn"), reference: str = "analytic",
                   architecture: PINN_ARCHITECTURES = "modified_mlp", epochs: int = 50, n_points: int = 2048) -> dict:
    """``pp.compare`` of methods on a preset against a reference, on the same points."""
    import pinneapple as pp
    opts = {"pinn": {"architecture": architecture, "epochs": epochs, "hidden_dim": 32, "n_layers": 3,
                     "n_collocation": 512}}
    return pp.compare(preset, list(methods), reference=reference, options=opts, n_points=n_points).to_dict()


def lbm_channel(nx: int = 64, ny: int = 32, Re: float = 100.0, u_in: float = 0.05, steps: int = 200,
                save_every: int = 50) -> dict:
    """Lattice Boltzmann channel flow (``SolverRegistry.build('lbm')``)."""
    s = _solver("lbm", nx=nx, ny=ny, Re=Re, u_in=u_in)
    out = s(steps=steps, save_every=save_every)
    r = out.result.detach().cpu().numpy()
    return {"shape": list(r.shape), "max": float(np.nanmax(r)), "min": float(np.nanmin(r)),
            "mean": float(np.nanmean(r))}


AHMED = {"length": 1.044, "width": 0.389, "height": 0.288, "nose_radius": 0.100, "slant_length": 0.222,
         "reference_cd_25deg": 0.285, "reference_re": 4.3e6}
TAU_MIN = 0.52                        # below this the BGK collision is at its stability limit and the forces are noise
LB = {"u": 0.10, "balance_tol": 0.10, "settle_tol": 0.05}


def _body_mask(shape, dx: float, x0: float, slant_deg: float, clearance: float,
               dims: tuple[float, float, float] = (AHMED["length"], AHMED["width"], AHMED["height"])):
    """The body as LBM solid cells, built from the request: a box ``dims`` = (length, width, height) in metres with its
    front edges rounded and its rear face slanted by ``slant_deg``, the nose at x = 0 and the lower face ``clearance``
    above the road (z = 0). With the reference proportions it is the Ahmed body (Ahmed, Ramm & Faltin 1984); with the
    user's dimensions it is the same family at that size. Written from the definition, so a job needs no mesh file."""
    import math

    import torch

    L, W, H = (float(d) for d in dims)
    R = AHMED["nose_radius"] / AHMED["length"] * L          # the reference proportions, scaled to the request
    Ls = AHMED["slant_length"] / AHMED["length"] * L
    nx, ny, nz = shape
    if min(L, W, H) <= 0:
        raise ValueError(f"the body dimensions must be positive, got {dims}")
    x = x0 + (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5 - ny / 2) * dx
    z = (np.arange(nz) + 0.5) * dx
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
    nose = (grid_x >= 0.0) & (grid_x <= R)
    inset = np.where(nose, R * (1 - np.sin(np.arccos(np.clip(1 - grid_x / R, -1.0, 1.0)))), 0.0)
    top = np.where(grid_x > L - Ls, clearance + H - (grid_x - (L - Ls)) * math.tan(math.radians(slant_deg)),
                   clearance + H)
    inside = ((grid_x >= 0.0) & (grid_x <= L) & (np.abs(grid_y) <= W / 2 - inset)
              & (grid_z >= clearance + inset) & (grid_z <= top - inset))
    return torch.as_tensor(inside)


def car_wind_tunnel_3d(slant_deg: float = 25.0, speed_kmh: float = 120.0, clearance_m: float = 0.15,
                       length_m: float = AHMED["length"], width_m: float = AHMED["width"],
                       height_m: float = AHMED["height"], cells_per_length: int = 32, Re: float = 250.0,
                       steps: int = 6000, smagorinsky: float = 0.12, samples: int = 12,
                       u_lattice: float = LB["u"], geometry: str = "",
                       body: Literal["ahmed", "formula"] = "ahmed", wheelbase: float = 3.60,
                       front_wing_aoa: float = 4.0, rear_wing_aoa: float = 6.0, drs_open: bool = False) -> dict:
    """Simulate a car in a 3-D lattice-Boltzmann wind tunnel (``WindTunnelLBM``, D3Q19, Smagorinsky LES, moving road):
    a road vehicle, a racing car or the reference Ahmed body, from its geometry to drag and lift, the force history and
    the mean field on three slices. Simulação 3-D de aerodinâmica externa de carro (arrasto, sustentação, esteira).

    **The geometry comes from the request**, three ways, and none of them needs a file to exist first:

    * ``body="ahmed"`` (default): ``length_m``, ``width_m``, ``height_m`` and ``slant_deg`` build a road-car body
      (rounded nose, rear slant, a gap under it above the moving road); the reference proportions are the Ahmed body;
    * ``body="formula"``: the library's parametric open-wheel car (``vehicles3d.FormulaCar``: monocoque, floor with
      diffuser, front and rear wings with DRS, exposed wheels) built from ``wheelbase``, ``front_wing_aoa``,
      ``rear_wing_aoa`` and ``drs_open`` — the knobs a request in words names;
    * ``geometry`` (an STL path): the user's own closed surface, scaled so its length is ``length_m`` and centred where
      the body sits; that path needs ``trimesh``.

    ``Re`` is clamped to what the lattice can hold at this resolution (``tau >= TAU_MIN``); the value says so in
    ``re_max``/``re_clamped``. The drag is reported with its regime: the reference coefficient of the Ahmed body (0.285
    at 25 deg) belongs to Re 4.3e6, which this grid cannot reach, and the same tunnel over-estimates a sphere by about
    a third at 16 cells — so the number is a coarse-grid result, not the car's.
    """
    import torch

    from pinneapple_simulation.numerical_solvers.lbm_tunnel import WindTunnelLBM

    surfaces = None
    L, W, H = float(length_m), float(width_m), float(height_m)
    if body == "formula":                              # the library's own parametric car, from the request's knobs
        from pinneapple_design.geometry.vehicles3d import FormulaCar
        surfaces = list(FormulaCar(wheelbase=float(wheelbase), front_wing_aoa=float(front_wing_aoa),
                                   rear_wing_aoa=float(rear_wing_aoa), drs_open=bool(drs_open)).cfd_bodies().values())
        lo = np.min([np.min(np.asarray(v, float), axis=0) for v, _ in surfaces], axis=0)
        hi = np.max([np.max(np.asarray(v, float), axis=0) for v, _ in surfaces], axis=0)
        L, W, H = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])
        origin = (float(lo[0]), float(lo[1]))
    n = max(16, int(cells_per_length))
    dx = L / n
    clearance = max(float(clearance_m), 4.0 * dx)      # the gap under the body has to be resolved, or drag is fiction
    if body == "formula" and n < 48:
        raise ValueError(f"the formula car needs at least 48 cells along its length (wings and wheels are thinner "
                         f"than a coarser cell), got {n}")
    u_lat = float(u_lattice)
    re_max = u_lat * n * 3.0 / (TAU_MIN - 0.5)         # nu = (tau - 0.5) / 3
    re = min(float(Re), re_max)
    nu = u_lat * n / re
    # the tunnel follows the body: 3.7 lengths long, 4.6 widths wide, three heights above it (the reference
    # proportions put the Ahmed body where the original study did)
    x0, x1, y_half = -1.2 * L, 2.5 * L, 2.3 * W
    shape = (int(round((x1 - x0) / dx)), int(round(2 * y_half / dx)),
             int(round((clearance + 3.0 * H) / dx)))
    tunnel = WindTunnelLBM(shape, nu=nu, u_in=(u_lat, 0.0, 0.0), smagorinsky=smagorinsky,
                           faces={"x-": "inlet", "x+": "outlet", "y-": "wall", "y+": "wall",
                                  "z-": "moving", "z+": "wall"})
    if surfaces is None and geometry:                  # the user's own surface, in place of the parametric body
        import trimesh
        surface = trimesh.load(str(geometry), force="mesh")
        lo, hi = surface.bounds
        surface.apply_scale(L / float(hi[0] - lo[0]))          # its length becomes length_m
        lo, hi = surface.bounds
        surface.apply_translation((0.0 - lo[0], -(lo[1] + hi[1]) / 2, clearance - lo[2]))
        surfaces = [(np.asarray(surface.vertices), np.asarray(surface.faces))]
    if surfaces is not None:                           # one voxelization per closed component, united
        import trimesh

        from pinneapple_design.geometry.ops.lbm_bridge import mesh_to_obstacle_mask
        bounds = {"x": (x0, x0 + shape[0] * dx), "y": (-y_half, y_half), "z": (0.0, shape[2] * dx)}
        if body == "formula":                          # its own origin, in metres, to the tunnel's frame (nose at
            z0 = min(float(np.min(np.asarray(v, float)[:, 2])) for v, _ in surfaces)   # x = 0, road at z = 0 and a
            surfaces = [(np.c_[np.asarray(v, float)[:, 0] - origin[0], np.asarray(v, float)[:, 1] - origin[1],  # gap the
                              np.asarray(v, float)[:, 2] - z0 + clearance], f)         # grid can hold)
                        for v, f in surfaces]
        mask = np.zeros(shape, dtype=bool)
        for verts, faces in surfaces:
            mask |= np.asarray(mesh_to_obstacle_mask(trimesh.Trimesh(vertices=np.asarray(verts, float),
                                                                     faces=np.asarray(faces), process=False),
                                                     bounds, shape), dtype=bool)
        mask = torch.as_tensor(mask)
    else:
        mask = _body_mask(shape, dx, x0, slant_deg, clearance, (L, W, H))
    tunnel.add_solid(mask)
    hist: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    tunnel.run(int(steps), every=max(1, int(steps) // max(2, int(samples))),
               callback=lambda m: hist.append((m.t, float(m.force_on_solid()[0]))))
    wall_s = time.perf_counter() - t0
    force = tunnel.force_on_solid()
    frontal = float(mask.permute(1, 2, 0).reshape(-1, shape[0]).sum(0).max())   # the most blocked y-z station, cells**2
    q = 0.5 * 1.0 * u_lat ** 2 * frontal                # lattice units: rho = 1
    h_cd = [round(f / q, 4) for _, f in hist]
    tail = h_cd[-max(2, len(h_cd) // 4):]
    mean_cd = sum(tail) / len(tail)
    spread = (max(tail) - min(tail)) / abs(mean_cd) if mean_cd else float("inf")
    fld = tunnel.fields()
    u = np.moveaxis(fld["u"], 0, -1)                   # (nx, ny, nz, 3)
    j, k = u.shape[1] // 2, u.shape[2] // 2
    flux_in, flux_out = float(u[0, :, :, 0].sum()), float(u[-1, :, :, 0].sum())
    balance = abs(flux_out - flux_in) / abs(flux_in or 1.0)
    dec = lambda a: np.round(np.asarray(a, float), 4).tolist()     # noqa: E731
    checks = [
        {"code": "simulate.converged", "status": "pass" if spread <= LB["settle_tol"] else "fail",
         "value": round(spread, 4), "limit": LB["settle_tol"],
         "detail": f"drag coefficient over the last {len(tail)} of {len(h_cd)} samples: mean {mean_cd:.3f}, "
                   f"spread {spread:.1%}"},
        {"code": "simulate.balance", "status": "pass" if balance <= LB["balance_tol"] else "fail",
         "value": round(balance, 4), "limit": LB["balance_tol"],
         "detail": f"inflow {flux_in:.4g} vs outflow {flux_out:.4g} (relative difference {balance:.2%}; the "
                   f"lattice is weakly compressible, Ma {u_lat * 3 ** 0.5:.2f})"},
    ]
    return {
        "kind": "cfd3d", "body": ("formula" if body == "formula" else
                                  "ahmed" if not geometry else str(geometry).rsplit("/", 1)[-1]),
        "geometry": (f"parametric formula car, {L:g} x {W:g} x {H:g} m (wheelbase {wheelbase:g} m, front wing "
                     f"{front_wing_aoa:g} deg, rear wing {rear_wing_aoa:g} deg, DRS {'open' if drs_open else 'closed'})"
                     if body == "formula" else
                     f"parametric {L:g} x {W:g} x {H:g} m, slant {slant_deg:g} deg" if not geometry else
                     f"surface {str(geometry).rsplit('/', 1)[-1]} scaled to {L:g} m"),
        "dims_m": {"length": L, "width": W, "height": H}, "slant_deg": float(slant_deg),
        "speed_kmh": float(speed_kmh),
        "Re": round(re, 3), "re_requested": float(Re), "re_max": round(re_max, 3), "re_clamped": re < float(Re),
        "tau": round(3.0 * nu + 0.5, 4), "cells_per_length": n, "cell_m": round(dx, 5),
        "clearance_m": round(clearance, 4), "shape": list(shape), "steps": int(steps),
        "cd": round(float(force[0]) / q, 4), "cl": round(float(force[2]) / q, 4), "cd_mean": round(mean_cd, 4),
        "cd_history": h_cd, "frontal_area_cells": round(frontal, 2), "wall_s": round(wall_s, 1),
        "mean_streamwise_speed": round(float(np.abs(u[..., 0]).mean()), 5),
        # the slices as a page draws them: one cell per row, so a row is a height (side view) or a width (plan view)
        "fields": {"u, side view (x→ z↑)": dec(u[:, j, :, 0].T[::-1]), "w, side view (x→ z↑)": dec(u[:, j, :, 2].T[::-1]),
                   "u, plan view (x→ y↑)": dec(u[:, :, k, 0].T[::-1])},
        "checks": checks,
        "reference": {"cd": AHMED["reference_cd_25deg"], "re": AHMED["reference_re"],
                      "source": "Ahmed, Ramm & Faltin (1984), SAE 840300",
                      "note": "the reference is at Re 4.3e6; this run is at the Re the grid can hold"},
        "limitations": [
            f"coarse lattice-Boltzmann ({n} cells along the body, Re {re:g}, tau {3.0 * nu + 0.5:.3f}): the drag "
            f"coefficient of the real body ({AHMED['reference_cd_25deg']} at Re {AHMED['reference_re']:g}) is not "
            "reproduced, and the same tunnel gives +33 % on a sphere of 16 cells at Re 100 against Schiller-Naumann",
            "no wheels, no yaw, no rotating flow; the road moves with the free stream and the body is the Ahmed "
            "reference geometry, not the user's car",
            f"clearance raised to {clearance:.3f} m so the underbody gap is resolved",
        ],
    }


def beam_deflection(L_m: float = 1.0, E_Pa: float = 2.0e11, I_m4: float = 8.33e-6, q_N_per_m: float = 1000.0,
                    nx: int = 200, bc_type: int = 0) -> dict:
    """Euler-Bernoulli beam boundary-value problem by finite differences (``beam_bvp_fdm``)."""
    s = _solver("beam_bvp_fdm", nx=nx, bc_type=bc_type)
    out = s(L_m, E_Pa, I_m4, q_N_per_m)
    w = out.result.detach().cpu().numpy().ravel()
    return {"max_abs_deflection_m": float(np.abs(w).max()), "nodes": int(w.size),
            **{k: float(v) for k, v in (out.extras or {}).items() if isinstance(v, (int, float))}}


def xtfc_pde(pde: str = "burgers_1d", n_basis: int = 60, n_query: int = 40) -> dict:
    """Extreme theory of functional connections solver (``xtfc_pde``)."""
    s = _solver("xtfc_pde", pde=pde, n_basis=n_basis)
    out = s(n_query=n_query)
    return {k: float(v) for k, v in (out.extras or {}).items() if isinstance(v, (int, float))} | {
        "result_shape": list(out.result.shape)}


def wheel_on_soil(soil: SOILS = "GRC-1", radius_m: float = 0.25, width_m: float = 0.2, load_N: float = 100.0,
                  slip: float = 0.2, gravity: float = 1.62) -> dict:
    """Wong-Reece wheel-soil model at steady state (``pinneapple_simulation.terramechanics``)."""
    from pinneapple_simulation.terramechanics import Wheel, WongReeceModel, get_soil
    m = WongReeceModel(get_soil(soil), Wheel(radius_m, width_m), g=gravity)
    return {k: v for k, v in m.steady_state(load_N, slip).items() if isinstance(v, (int, float, str))}


def meshless_compare(kind: Literal["poisson", "heat", "helmholtz"] = "poisson",
                     exact: str = "sin(pi*x)*sin(pi*y)", domain: MESHLESS_DOMAINS = "square",
                     methods: list = ("kansa", "rbf_fd", "fd", "fem"), k: float = 4.0, alpha: float = 1.0) -> dict:
    """Meshless vs mesh-based methods on one problem (``meshless_study``); expressions go through the library's
    own expression checks."""
    from pinneapple_simulation.meshless import Problem, meshless_study
    from pinneapple_simulation.meshless import domain as make_domain
    st = meshless_study(Problem(kind=kind, exact=exact, k=k, alpha=alpha), make_domain(domain), methods=tuple(methods))
    return st.to_dict() if hasattr(st, "to_dict") else {"result": str(st)}


def rom_compare(Y=None, A=None, methods: list = (), ranks: list = (), budget_s: float = 5.0, seed: int = 0) -> dict:
    """Reduced-order models compared on snapshots (``pinneapple_neural.architectures.rom.study.rom_study``):
    parametric snapshots ``Y`` (samples, dofs) for parameters ``A``."""
    from pinneapple_data.snapshots import snapshots_from_arrays
    from pinneapple_neural.architectures.rom.study import rom_study
    if Y is None:
        raise StepFailed("connect a dataset")
    snaps = snapshots_from_arrays({"u": np.asarray(Y, float)}, params=np.asarray(A, float),
                                  param_names=[f"p{i}" for i in range(np.asarray(A).shape[1])])
    res = rom_study(snaps, methods=list(methods) or None, ranks=list(ranks) or None, budget_s=budget_s, seed=seed)
    return res.to_dict()


def parametric_pod(A=None, Y=None, r: int = 16, regressor: Literal["rbf", "gpr", "nearest", "linear"] = "rbf",
                   test_fraction: float = 0.2, seed: int = 0) -> dict:
    """Parametric POD (``ParametricPOD``) fitted on a dataset, relative L2 on held-out samples."""
    from pinneapple_neural.architectures.rom import ParametricPOD
    A, Y = np.asarray(A, float), np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(A))
    nt = max(1, int(round(test_fraction * len(A))))
    te, tr = idx[:nt], idx[nt:]
    m = ParametricPOD(r=r, regressor=regressor, seed=seed).fit(A[tr], Y[tr])
    pred = np.asarray(m.predict(A[te]))
    rel = np.linalg.norm(pred - Y[te], axis=1) / np.maximum(np.linalg.norm(Y[te], axis=1), 1e-12)
    return {"test_rel_l2_mean": float(rel.mean()), "test_rel_l2_max": float(rel.max()), "train": int(len(tr)),
            "test": int(len(te)), "rank": r, "regressor": regressor}


def surrogate_study(X=None, Y=None, example: Literal["none", "burgers_fields", "airfoil_rans"] = "none",
                    methods: list = ("ridge", "gp", "gbt"), budget: float = 10.0, seeds: int = 1) -> dict:
    """Classical ML / GP / tree / neural surrogates compared (``pinneapple_analysis.surrogate.run_study``)."""
    from pinneapple_analysis import surrogate as su
    if example != "none":
        data = getattr(su.examples, example)(**({"n": 32} if example == "burgers_fields" else {}))
    else:
        if X is None:
            raise StepFailed("connect a dataset or pick a built-in example")
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        arrays = {**{f"x{i}": X[:, i] for i in range(X.shape[1])}, **{f"y{j}": Y[:, j] for j in range(Y.shape[1])}}
        data = su.from_arrays(arrays, input_names=[f"x{i}" for i in range(X.shape[1])],
                              output_names=[f"y{j}" for j in range(Y.shape[1])])
    res = su.run_study(data, methods=list(methods), budget=budget, seeds=seeds)
    out = {"best": str(getattr(res, "best", ""))}
    for k in ("trust", "limitations", "methods"):
        v = getattr(res, k, None)
        if v is not None:
            out[k] = v if isinstance(v, (str, int, float, list, dict)) else str(v)[:2000]
    return out


def calculix_cantilever(length_m: float = 1.0, b_m: float = 0.1, h_m: float = 0.1, nx: int = 20,
                        element_type: Literal["C3D20R", "C3D8", "C3D8R"] = "C3D20R", run: bool = True) -> dict:
    """A cantilever deck written with the library (``cantilever_hex_mesh`` + ``write_inp``); solved with CalculiX
    when ``ccx`` (or its Docker image) is available, otherwise the deck is written and the node says so."""
    import tempfile
    from pathlib import Path

    from pinneapple_simulation.external_solvers import calculix as cx
    model = cx.cantilever_hex_mesh(L=length_m, b=b_m, h=h_m, nx=nx, element_type=element_type)
    d = Path(tempfile.mkdtemp(prefix="ccx_"))
    cx.write_inp(model, d / "beam.inp")
    out = {"deck": str(d / "beam.inp"), "nodes": int(len(getattr(model, "nodes", []) or [])),
           "ccx_available": bool(cx.ccx_backend())}
    if run and cx.ccx_backend():
        r = cx.run_ccx(d, "beam")
        out["run"] = str(r)[:500]
    elif run:
        out["note"] = "ccx not available: deck written, not solved"
    return out


def fmu_simulate(file: str = "", stop_time: float = 1.0, step_size: float = 0.01) -> dict:
    """Simulate an FMU from the data folder (``pinneapple_simulation.external_solvers.modelica.simulate_fmu``;
    needs fmpy)."""
    from pinneapple_simulation.external_solvers.modelica import FMUSimConfig, simulate_fmu

    from .config import safe_data_path
    if importlib.util.find_spec("fmpy") is None:
        raise StepFailed("needs fmpy (pip install fmpy)")
    res = simulate_fmu(str(safe_data_path(file)), FMUSimConfig(stop_time=stop_time, step_size=step_size))
    return {k: {"final": float(np.asarray(v)[-1]), "n": int(np.asarray(v).size)} for k, v in res.items()}


def newton_validate(solver: Literal["xpbd", "semi_implicit", "featherstone", "mujoco"] = "xpbd",
                    cases: list = ("free_fall", "pendulum_period", "bounce")) -> dict:
    """Rigid-body solver validation against closed forms (``newton_physics.validate``; needs NVIDIA Newton)."""
    if importlib.util.find_spec("newton") is None:
        raise StepFailed("needs NVIDIA Newton (pip install newton)")
    from pinneapple_simulation.external_solvers import newton_physics as nwt
    return nwt.validate(solver=solver, cases=list(cases))


def pict_cavity(re: float = 100.0, nx: int = 32, engine: Literal["auto", "pict", "local"] = "local",
                iterations: int = 50) -> dict:
    """Lid-driven cavity and its checks against Ghia et al. (1982)
    (``pinneapple_simulation.external_solvers.pict.run_cavity``).

    ``engine="local"`` (the default) runs this repository's CPU vorticity-streamfunction solver, so the node works on
    any machine; ``"pict"`` runs PICT on CUDA when it is installed (and says what is missing when it is not), and
    ``"auto"`` picks PICT if it can run. The engine that produced the numbers is in the output.
    """
    from pinneapple_simulation.external_solvers.pict import CavityCase, PICTRunConfig, run_cavity

    case = CavityCase(nx=int(nx), ny=int(nx), Re=float(re), refine=False)
    result = run_cavity(case, PICTRunConfig(iterations=int(iterations)), engine=engine)
    checks = result.checks()
    return {
        "engine": result.engine,
        "layout": result.layout,
        "reynolds": float(re),
        "cells": int(nx),
        "ghia_u_rms": float(checks["ghia_u"]["rms"]) if "ghia_u" in checks else None,
        "ghia_v_rms": float(checks["ghia_v"]["rms"]) if "ghia_v" in checks else None,
        "continuity": float(checks["continuity"]["value"]),
        "walls": float(checks["walls"]["value"]),
        "lid": float(checks["lid"]["value"]),
        "passed": bool(checks["passed"]),
        "wall_time_s": float(result.wall_time_s),
    }


def openfoam_available() -> dict:
    """Is OpenFOAM installed (blockMesh on PATH)?"""
    return {"openfoam": bool(shutil.which("blockMesh")), "calculix": bool(shutil.which("ccx"))}


def model_architecture(name: str = "fno", in_dim: int = 1, out_dim: int = 1, hidden: int = 64) -> dict:
    """Build any model of the registry (operators, GNNs, PINN variants, ROMs, recurrent, transformers) with its
    default arguments plus sizes, and count its parameters: a preview before wiring data to it."""
    import inspect

    from pinneapple_neural.architectures.registry import ModelRegistry

    from . import learn
    cls_kw = {}
    spec = ModelRegistry.spec(name) if hasattr(ModelRegistry, "spec") else None
    fn = getattr(spec, "builder", None) or getattr(spec, "cls", None)
    sig = inspect.signature(fn) if fn is not None else None
    for k, v in (("in_dim", in_dim), ("out_dim", out_dim), ("in_channels", in_dim), ("out_channels", out_dim),
                 ("branch_dim", in_dim), ("trunk_dim", 1), ("node_in_dim", in_dim), ("node_dim", in_dim),
                 ("hidden", hidden), ("hidden_dim", hidden), ("width", hidden), ("r", min(hidden, 16))):
        if sig is None or k in sig.parameters:
            cls_kw[k] = v
    try:
        m = ModelRegistry.build(name, **cls_kw)
    except TypeError as exc:
        raise StepFailed(f"{name}: {exc}") from None
    g = learn.layer_graph(m)
    return {"model": name, "parameters": g["total_params"], "layers": len(g["nodes"]), "arguments": cls_kw}


DOE_METHODS = Literal["lhs", "sobol", "halton", "random", "maximin", "full_factorial", "fractional_factorial"]


def doe_design(parameters: dict = None, n: int = 16, method: DOE_METHODS = "lhs", seed: int = 0,
               out_dir: str = "") -> dict:
    """``pinneapple_data.doe.design`` over a parameter space given as ``{"name": [low, high], ...}`` (or a
    ``Parameter`` dict per name). Writes the cases as CSV in the node's folder."""
    from pathlib import Path

    from pinneapple_data.contracts import ParameterSpace
    from pinneapple_data.doe.designs import METHODS, design
    if method not in METHODS:
        raise StepFailed(f"unknown method {method!r}; one of {METHODS}")
    parameters = parameters or {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}
    specs = []
    for name, v in parameters.items():
        spec = dict(v) if isinstance(v, dict) else {"low": float(v[0]), "high": float(v[1])}
        specs.append({"name": name, **spec})
    try:
        space = ParameterSpace(name="flow_doe", parameters=specs)
    except Exception as exc:  # noqa: BLE001 - pydantic validation message is the useful part
        raise StepFailed(f"parameter space: {exc}") from exc
    d = design(space, n, method, seed=seed)
    path = ""
    if out_dir:
        path = str(Path(out_dir) / "cases.csv")
        d.cases.to_csv(path, index=False)
    from pinneapple_data.doe.coverage import coverage
    cov = coverage(d.unit, space.names)
    return {"method": d.method, "n_cases": len(d), "min_distance": float(_min_distance(d.unit)),
            "cases_csv": path, "coverage": {k: v for k, v in cov.items() if isinstance(v, (int, float, str))}}


def _min_distance(unit):
    from pinneapple_data.doe.designs import min_distance
    return min_distance(unit)


GEOMETRY_FORMATS = Literal["stl", "obj", "glb", "usda", "step"]


def geometry_export(name: str = "caarc_building", fmt: GEOMETRY_FORMATS = "stl", lod: str = "light",
                    out_dir: str = "") -> dict:
    """Build a parametric geometry of ``pinneapple_design.geometry_library`` and write it (STL / OBJ / glTF binary /
    USD ASCII / STEP) in the node's folder."""
    from pathlib import Path

    from pinneapple_design import geometry_library as gl
    if name not in gl.list():
        raise StepFailed(f"unknown geometry {name!r}; one of {gl.list()}")
    geom = gl.build(name, lod=lod)
    path = geom.export(str(Path(out_dir or ".") / f"{name}.{fmt}"), fmt)
    info = gl.info(name)
    return {"geometry": name, "format": fmt, "file": path, "bytes": Path(path).stat().st_size,
            **{k: v for k, v in info.items() if isinstance(v, (int, float, str))}}


def design_intent(text: str = "a heat sink for a 20 W chip in still air", seed: int = 0, budget: int = 200,
                  n_screen: int = 500, out_dir: str = "") -> dict:
    """``pinneapple_design.intent.design``: a plain-language goal to a verified design package (files in the node's
    folder), or the questions it still needs, or why it is out of scope."""
    from pinneapple_design.intent.api import design
    res = design(text, seed=seed, budget=budget, n_screen=n_screen, out_dir=out_dir or None)
    kind = type(res).__name__
    out = res.to_dict() if hasattr(res, "to_dict") else {}
    return {"outcome": kind, **{k: v for k, v in out.items() if k not in ("candidates",)}}
