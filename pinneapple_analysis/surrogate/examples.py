"""Example surrogate datasets that need no download.

``airfoil_rans``  the zoo dataset ``airfoil_rans_cst_re4e6``: OpenFOAM RANS (k-omega SST, Re = 4e6) of CST airfoils;
                  inputs the six CST coefficients and the angle of attack, outputs Cl and Cd (Cm is not recorded);
                  group = the shape (whole airfoils can be held out). Its train and test splits are pooled (the
                  study re-splits); unsteady runs of the train split, recorded at the final iteration rather than
                  time-averaged, are left out.
``burgers_fields`` viscous Burgers ``u_t + u u_x = nu u_xx`` on x in [-1, 1], u(x, 0) = -sin(pi x), u(+-1, t) = 0,
                  solved exactly (Cole-Hopf, Gauss-Hermite quadrature: ``pinneapple_physics.closed_form.burgers``)
                  for ``n`` viscosities log-uniform in ``nu_range``; output the field u(x, t) on an ``nx`` x ``nt``
                  grid, t in [0, t_max]. Small viscosities form a steep front at x = 0 (a near-discontinuity); the
                  recommended shift split holds out the smallest viscosities.
``cantilever_3d``  the 3-D solid FEM cantilever of the ROM Lab (``apps/rom_lab/examples/cantilever_designs.npz``: 100
                  Latin-hypercube designs over height H, width W and load angle) as a field dataset on its
                  31 x 4 x 7 node lattice: displacement (ux, uy, uz, mm) and von Mises stress (MPa). The FEM solve
                  time is measured here, on this machine, by solving ``n_timed`` stored designs again (and checking
                  that they reproduce the stored fields). The shift split holds out the smallest load angles.
                  The von Mises stress is declared non-negative (``meta["nonnegative"]``): the study trains on
                  its square root and predicts its square.
"""
from __future__ import annotations

import numpy as np

from .data import SurrogateData

__all__ = ["airfoil_rans", "burgers_fields", "burgers_fdm_baseline", "cantilever_3d", "EXAMPLES"]


def airfoil_rans() -> SurrogateData:
    import pinneapple as pp
    inputs = ["T0", "T1", "T2", "M0", "M1", "M2", "alpha_deg"]
    outputs = ["Cl", "Cd"]
    X, Y, G, status, dropped = [], [], [], [], 0
    for split in ("train", "test"):
        d = pp.datasets.load("airfoil_rans_cst_re4e6", split=split)
        st = np.asarray(d["status"]).astype(str)
        keep = st != "unsteady" if split == "train" else np.ones(len(st), bool)
        dropped += int((~keep).sum())
        X.append(np.stack([np.asarray(d[c], float)[keep] for c in inputs], 1))
        Y.append(np.stack([np.asarray(d[c], float)[keep] for c in outputs], 1))
        G.append(np.asarray(d["shape_id"])[keep])
        status.append(st[keep])
    st = np.concatenate(status)
    meta = {"source": "PINNeAPPle zoo dataset airfoil_rans_cst_re4e6 (OpenFOAM simpleFoam, k-omega SST, Re 4e6)",
            "solver": "OpenFOAM", "group": "shape_id", "constraints": ["Cd > 0"], "unsteady_runs_left_out": dropped,
            "status_counts": {str(k): int((st == k).sum()) for k in np.unique(st)}}
    return SurrogateData(np.concatenate(X), np.concatenate(Y), inputs, outputs, groups=np.concatenate(G), meta=meta)


def burgers_fields(n: int = 96, nu_range: tuple[float, float] = (0.004, 0.1), nx: int = 48, nt: int = 12,
                   t_max: float = 0.9, seed: int = 0) -> SurrogateData:
    import time

    from pinneapple_physics.closed_form.burgers import burgers_sine_exact
    rng = np.random.default_rng(seed)
    nus = np.exp(rng.uniform(np.log(nu_range[0]), np.log(nu_range[1]), n))
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, t_max, nt)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")
    t0 = time.perf_counter()
    U = np.stack([burgers_sine_exact(Xg, Tg, float(nu)) for nu in nus])
    cost = (time.perf_counter() - t0) / n
    coords = np.stack([Xg.ravel(), Tg.ravel()], 1)
    meta = {"source": "viscous Burgers, exact solution (Cole-Hopf, 120-node Gauss-Hermite quadrature)",
            "solver": "pinneapple_physics.closed_form.burgers.burgers_sine_exact", "solver_s_per_sample": cost,
            "shift": {"column": "nu", "side": "low"},
            "constraints": ["abs(u) <= 1.005"],     # maximum principle |u| <= max |u(x, 0)| = 1, 0.5 % slack
            "pde": {"type": "burgers", "coords": {"x": "x", "t": "t"}, "coefficients": {"nu": "nu"}}}
    return SurrogateData(nus[:, None], U, ["nu"], ["u"], coords=coords, coord_names=["x", "t"], meta=meta)


def cantilever_3d(path, n_timed: int = 2) -> SurrogateData:
    """See the module docstring. ``path``: the snapshot file (``load_snapshots`` of the same file gives the mesh of
    each design and the clamp and load sets for a 3-D view; ``snapshot_values`` maps predictions back)."""
    import time

    from pinneapple_data.snapshots import load_snapshots
    from pinneapple_lab.experiments.rom import cantilever

    from .data import from_snapshots
    with open(path, "rb") as f:
        snaps = load_snapshots([("cantilever_designs.npz", f.read())])
    times = []
    for i in range(n_timed):                                     # time the FEM here, and check it is the data's FEM
        t0 = time.perf_counter()
        r = cantilever(*snaps.params[i])
        times.append(time.perf_counter() - t0)
        stored = snaps.field_values("von_mises")[i]
        if not np.allclose(r.von_mises / 1e6, stored, rtol=1e-4, atol=1e-4):
            raise RuntimeError(f"design {i}: the FEM no longer reproduces the stored von Mises stress")
    meta = {"source": "3-D solid FEM cantilever (pinneapple_lab.experiments.rom.cantilever, 30 x 3 x 6 hexahedra), "
                      "100 Latin-hypercube designs", "solver": "pinneapple_simulation.numerical_solvers.solid_fem",
            "solver_s_per_sample": float(np.median(times)) if times else None, "solver_s_timed_on": len(times),
            "shift": {"column": "angle_deg", "side": "low"}, "nonnegative": ["von_mises"]}
    return from_snapshots(snaps, meta=meta)


def burgers_fdm_baseline(nx: int = 201, cfl: float = 0.4):
    """A numerical-solver baseline for ``burgers_fields``: the library's explicit upwind/FTCS finite-difference
    scheme (``pinneapple_simulation.numerical_solvers.fdm``) on ``nx`` points with a stable time step, interpolated
    to the dataset grid. Returns ``solve(X, coords) -> (n, n_points)``; its error against the exact solution and its
    cost per query are what a surrogate competes with."""
    from pinneapple_simulation.numerical_solvers.fdm import _ftcs_burgers_1d

    def solve(X: np.ndarray, coords: np.ndarray) -> np.ndarray:
        xs, ts = np.unique(coords[:, 0]), np.unique(coords[:, 1])
        x = np.linspace(-1.0, 1.0, nx)
        dx = x[1] - x[0]
        out = []
        for nu in np.asarray(X, float)[:, 0]:
            dt = cfl * min(dx / 1.0, dx * dx / (2.0 * nu))
            nt = int(np.ceil(ts.max() / dt))
            dt = ts.max() / nt
            traj = _ftcs_burgers_1d(-np.sin(np.pi * x), float(nu), dx, dt, nt)        # (nx, nt + 1)
            tt = np.linspace(0.0, ts.max(), nt + 1)
            k = np.clip(np.searchsorted(tt, ts), 0, nt)
            field = np.stack([np.interp(xs, x, traj[:, j]) for j in k], 1)          # (len(xs), len(ts))
            out.append(field.ravel())
        return np.stack(out)
    solve.description = f"explicit upwind / FTCS finite differences, {nx} points, CFL {cfl}"
    return solve


EXAMPLES = {"airfoil_rans": airfoil_rans, "burgers_fields": burgers_fields}
