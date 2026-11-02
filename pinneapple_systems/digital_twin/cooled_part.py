"""A 3-D thermal digital twin of a part cooled by internal passages (a battery cold plate, a cooled mould, a cold
plate under power electronics), built from the part's geometry file.

    grid = voxelize_part("plate.stl", pitch=(0.01, 0.004, 0.002), unit="m")
    model = CooledPart(grid, h=2500.0, heat_flux=2000.0 / 0.06, T_in=25.0)          # W/m2 on the loaded face
    plant = CooledPart(voxelize_part("plate.stl", pitch=(0.005, 0.002, 0.001)), h=2500.0, heat_flux=..., T_in=25.0)
    rec = simulate_readings(plant, sensors, t_end=900, dt=1.0, noise=0.05, fault={"passage": 2, "t": 400, ...})
    out = run_twin(model, rec, sensors)               # field, coefficients with bands, alarms, checks

**Geometry.** :func:`voxelize_part` puts the file on a box grid (the pitch may differ per axis; the file's length
unit is stated by the caller: STL stores none). A cell is solid when its centre is inside the closed surface (ray
parity along the last axis). The coolant passages are the void regions of the bounding box that cross it along the
flow axis without touching its side faces; every solid face on a passage is a wall of that passage.

**Physics** (constant properties, SI units, temperatures in degC)::

    rho c dT/dt = div(k grad T)              in the solid
    -k dT/dn = h_p (T - T_in)                on the walls of passage p
    -k dT/dn = -q''                          on the loaded face (the face of the bounding box at the top of
                                              ``load_axis``: where the cells sit), adiabatic elsewhere

The coolant is taken at its inlet temperature along the whole passage (no heating along the flow): a stated
simplification, good when the coolant's rise is small next to the wall-to-coolant difference. Time stepping is
implicit (backward Euler), with the walls at their nominal coefficients in the matrix, factorised once, and the
departure of each coefficient from nominal (a clogging passage) taken explicitly. Conduction conserves energy, so
at steady state with nominal coefficients the heat removed by the walls equals the heat put in (the test checks it).

**Twin.** :func:`run_twin` is a stochastic ensemble Kalman filter with perturbed observations (Evensen 2009,
*Data Assimilation*, ch. 9), written in ensemble space so no state-by-state matrix is formed (``TwinRuntime``'s
filter holds dense covariances, which a 3-D field does not fit). The members step a POD-Galerkin reduction of the
full model (:class:`ReducedPart`), so the filter keeps up with the sensors. The state is the field plus
``log h_p`` of every passage, the coefficients as random walks: a passage that clogs loses its coefficient, and
the filter follows it.
An alarm is raised when a passage's coefficient stays below ``alarm_fraction`` of its nominal value, with its upper
2-sigma band below too, for ``alarm_samples`` samples in a row.

**Replay.** Without live data, :func:`simulate_readings` makes a recording from a *plant*: the same physics on a
finer grid (so the twin is not checked on its own discretisation), a blockage of one passage at a stated time and
Gaussian sensor noise. The recording says ``simulated: True``; the twin's checks on it are a replay of a known
fault, not a measurement of the real part.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["PartGrid", "voxelize_part", "CooledPart", "ReducedPart", "simulate_readings", "run_twin", "field_view_of",
           "ALUMINIUM"]

ALUMINIUM = {"k": 167.0, "rho": 2700.0, "cp": 896.0}     # 6061-T6: W/(m K), kg/m3, J/(kg K)
_LENGTH = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}


@dataclass
class PartGrid:
    """The part on a box grid: ``solid`` (nx, ny, nz) bool, ``passage`` (nx, ny, nz) int (0: not a passage,
    1..P: passage number), ``origin`` [m] (lower corner of the box), ``d`` [m] (pitch per axis), ``flow_axis``."""
    origin: np.ndarray
    d: np.ndarray
    solid: np.ndarray
    passage: np.ndarray
    flow_axis: int
    source: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.solid.shape)

    @property
    def n_passages(self) -> int:
        return int(self.passage.max())

    def centres(self) -> np.ndarray:
        """Centres [m] of every grid cell, (nx, ny, nz, 3)."""
        ax = [self.origin[i] + self.d[i] * (np.arange(n) + 0.5) for i, n in enumerate(self.shape)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1)

    def solid_volume(self) -> float:
        return float(self.solid.sum() * np.prod(self.d))


def _load(geometry) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, (str, bytes)) or hasattr(geometry, "__fspath__"):
        import trimesh
        m = trimesh.load(str(geometry), force="mesh")
        return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
        return np.asarray(geometry.vertices, float), np.asarray(geometry.faces, np.int64)
    V, F = geometry
    return np.asarray(V, float), np.asarray(F, np.int64)


def _inside(V: np.ndarray, F: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """(len(xs), len(ys), len(zs)) bool: cell centres inside the closed surface, by the parity of the crossings of
    a ray along z through each (x, y) column."""
    T = V[F]                                             # (nt, 3, 3)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    keep = np.abs(det) > 1e-30                           # triangles seen edge-on from the ray never cross it
    a, b, c, det = a[keep], b[keep], c[keep], det[keep]
    span = max(float(np.ptp(V[:, 0])), float(np.ptp(V[:, 1])), 1e-12)
    X, Y = np.meshgrid(xs + 1.234567e-7 * span, ys + 2.345678e-7 * span, indexing="ij")   # off every edge
    px, py = X.ravel(), Y.ravel()
    out = np.zeros((len(px), len(zs)), bool)
    chunk = max(1, 4_000_000 // max(1, len(a)))
    for s in range(0, len(px), chunk):
        qx, qy = px[s:s + chunk, None], py[s:s + chunk, None]
        l1 = ((b[:, 1] - c[:, 1]) * (qx - c[:, 0]) + (c[:, 0] - b[:, 0]) * (qy - c[:, 1])) / det
        l2 = ((c[:, 1] - a[:, 1]) * (qx - c[:, 0]) + (a[:, 0] - c[:, 0]) * (qy - c[:, 1])) / det
        l3 = 1.0 - l1 - l2
        hit = (l1 > 0) & (l2 > 0) & (l3 > 0)
        z = l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2]
        z = np.where(hit, z, np.inf)
        below = (z[:, :, None] < zs[None, None, :]).sum(1)            # crossings below each cell centre
        out[s:s + chunk] = below % 2 == 1
    return out.reshape(len(xs), len(ys), len(zs))


def voxelize_part(geometry, pitch: float | Sequence[float], *, unit: str = "m", flow_axis: int | None = None
                  ) -> PartGrid:
    """The part on a box grid, with its coolant passages.

    Parameters
    ----------
    geometry : str or path or trimesh.Trimesh or (V, F)
        A closed surface: a file trimesh reads (STL, OBJ, PLY...), a mesh, or vertices and triangles.
    pitch : float or (3,) [m]
        Cell size per axis (adjusted so a whole number of cells spans the bounding box).
    unit : str
        Length unit of the file's coordinates: "m", "mm", "cm" or "in".
    flow_axis : int, optional
        Axis the passages run along; by default the longest axis of the bounding box with passages along it.

    Returns
    -------
    grid : PartGrid
    """
    if unit not in _LENGTH:
        raise ValueError(f"unit is one of {sorted(_LENGTH)}")
    V, F = _load(geometry)
    V = V * _LENGTH[unit]
    lo, hi = V.min(0), V.max(0)
    p = np.broadcast_to(np.asarray(pitch, float), (3,))
    n = np.maximum(1, np.round((hi - lo) / p).astype(int))
    d = (hi - lo) / n
    ax = [lo[i] + d[i] * (np.arange(n[i]) + 0.5) for i in range(3)]
    solid = _inside(V, F, *ax)
    if not solid.any():
        raise ValueError("no cell centre is inside the surface: is it closed, and is the pitch smaller than the part?")
    from scipy import ndimage
    lab, k = ndimage.label(~solid)
    order = [flow_axis] if flow_axis is not None else list(np.argsort(-(hi - lo)))
    passage = np.zeros(solid.shape, int)
    for axis in order:
        others = [i for i in range(3) if i != axis]
        cnt = 0
        for j in range(1, k + 1):
            m = lab == j
            touches = lambda i, side: m.take(0 if side == 0 else -1, axis=i).any()   # noqa: E731
            if touches(axis, 0) and touches(axis, 1) and not any(touches(i, s) for i in others for s in (0, 1)):
                cnt += 1
                passage[m] = cnt
        if cnt:
            flow_axis = int(axis)
            break
    if flow_axis is None:
        raise ValueError("no coolant passage: no void region crosses the part along an axis without touching its "
                         "sides")
    return PartGrid(lo, d, solid, passage, flow_axis,
                    {"unit": unit, "triangles": int(len(F)), "bbox_min": lo.tolist(), "bbox_max": hi.tolist()})


class CooledPart:
    """The discretised physics of a part on a :class:`PartGrid` (see the module text).

    Parameters
    ----------
    grid : PartGrid
    h : float or sequence [W/(m2 K)]
        Nominal wall heat-transfer coefficient, one value or one per passage.
    heat_flux : float [W/m2]
        Heat flux into the loaded face (or give ``heat_load``).
    T_in : float [degC]
        Coolant inlet temperature.
    material : dict
        ``k`` [W/(m K)], ``rho`` [kg/m3], ``cp`` [J/(kg K)]; aluminium by default.
    load_axis : int
        The loaded face is the top of this axis (where the cells sit on the part).
    heat_load : float, optional [W]
        Total heat into the loaded face, instead of ``heat_flux``.
    """

    def __init__(self, grid: PartGrid, h: float | Sequence[float], heat_flux: float | None, T_in: float, *,
                 material: Mapping[str, float] = ALUMINIUM, load_axis: int = 2, heat_load: float | None = None):
        import scipy.sparse as sp
        if (heat_flux is None) == (heat_load is None):
            raise ValueError("give heat_flux or heat_load")
        self.grid, self.T_in, self.load_axis = grid, float(T_in), int(load_axis)
        self.material = dict(material)
        P = grid.n_passages
        self.h_nominal = np.broadcast_to(np.asarray(h, float), (P,)).copy()
        S, d = grid.solid, grid.d
        self.index = -np.ones(S.shape, int)
        self.index[S] = np.arange(int(S.sum()))
        self.n = int(S.sum())
        self.cells = np.argwhere(S)
        self.C = self.material["rho"] * self.material["cp"] * float(np.prod(d))       # J/K per cell
        area = [d[1] * d[2], d[0] * d[2], d[0] * d[1]]
        rows, cols, vals = [], [], []
        wall = np.zeros((self.n, P))                                                     # wall area per cell, passage
        for a in range(3):
            g = self.material["k"] * area[a] / d[a]
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[a], sl1[a] = slice(0, -1), slice(1, None)
            i0, i1 = self.index[tuple(sl0)], self.index[tuple(sl1)]
            both = (i0 >= 0) & (i1 >= 0)
            u, v = i0[both], i1[both]
            rows += [u, v, u, v]
            cols += [v, u, u, v]
            vals += [np.full(len(u), -g), np.full(len(u), -g), np.full(len(u), g), np.full(len(u), g)]
            p0, p1 = grid.passage[tuple(sl0)], grid.passage[tuple(sl1)]
            for (ii, pp) in ((i0, p1), (i1, p0)):                                         # solid next to a passage
                m = (ii >= 0) & (pp > 0)
                np.add.at(wall, (ii[m], pp[m] - 1), area[a])
        self.K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(self.n, self.n))
        self.wall = wall
        self.wall_cells = np.flatnonzero(wall.sum(1) > 0)
        top = self.index.take(-1, axis=self.load_axis)
        self.loaded = top[top >= 0]
        self.loaded_area = float(len(self.loaded) * area[self.load_axis])
        self.heat_flux = float(heat_flux) if heat_flux is not None else float(heat_load) / self.loaded_area
        self.source = np.zeros(self.n)
        self.source[self.loaded] = self.heat_flux * area[self.load_axis]                 # W per cell
        self._lu: dict[float, Any] = {}

    @property
    def heat_in(self) -> float:
        """Heat put into the part [W]."""
        return float(self.source.sum())

    def _factor(self, dt: float):
        if dt not in self._lu:
            import scipy.sparse as sp
            from scipy.sparse.linalg import splu
            A = sp.identity(self.n, format="csc") + (dt / self.C) * (self.K + sp.diags(self.wall @ self.h_nominal))
            self._lu[dt] = splu(A.tocsc())
        return self._lu[dt]

    def step(self, T: np.ndarray, h: np.ndarray, dt: float, load: float = 1.0) -> np.ndarray:
        """Advance ``T`` (n,) or (n, m) [degC] by ``dt`` [s] with coefficients ``h`` (P,) or (P, m) [W/(m2 K)];
        ``load`` scales the heat flux. Backward Euler with the walls at their nominal coefficients in the matrix
        (factorised once per ``dt``) and the difference ``h - h_nominal`` taken explicitly: exact when ``h`` is
        nominal, first order otherwise, and stable for ``h`` up to about twice nominal at any ``dt``."""
        T = np.asarray(T, float)
        one = T.ndim == 1
        T2, H = (T[:, None], np.asarray(h, float).reshape(-1, 1)) if one else (T, np.asarray(h, float))
        D0 = (self.wall @ self.h_nominal)[:, None]                                       # W/K per cell
        Dh = self.wall @ H                                                               # (n, m)
        rhs = T2 + (dt / self.C) * (load * self.source[:, None] + D0 * self.T_in + (D0 - Dh) * (T2 - self.T_in))
        out = self._factor(float(dt)).solve(rhs)
        return out[:, 0] if one else out

    def heat_removed(self, T: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Heat [W] taken by each passage at the field ``T`` (n,): sum of h_p A (T - T_in) over its walls."""
        h = self.h_nominal if h is None else np.asarray(h, float)
        return (self.wall * (np.asarray(T, float)[:, None] - self.T_in)).sum(0) * h

    def sensor_cells(self, sensors: Sequence[Sequence[float]]) -> np.ndarray:
        """Index of the solid cell nearest each sensor position [m]."""
        C = self.grid.centres()[self.grid.solid]
        P = np.asarray(sensors, float).reshape(-1, 3)
        return np.array([int(np.argmin(((C - p) ** 2).sum(1))) for p in P])

    def steady(self, h: np.ndarray | None = None, load: float = 1.0, dt: float = 30.0, tol: float = 1e-6,
               max_steps: int = 20_000) -> np.ndarray:
        """The steady field (marched in time until it stops changing by ``tol`` degC per step)."""
        h = self.h_nominal if h is None else np.asarray(h, float)
        T = np.full(self.n, self.T_in)
        for _ in range(max_steps):
            T1 = self.step(T, h, dt, load)
            if np.max(np.abs(T1 - T)) < tol:
                return T1
            T = T1
        return T


class ReducedPart:
    """A POD-Galerkin reduced model of a :class:`CooledPart` (Holmes, Lumley & Berkooz, *Turbulence, Coherent
    Structures, Dynamical Systems and Symmetry*, 1996, ch. 3): the field ``T = T_in + Phi a`` with ``r`` modes taken
    from snapshots of the full model (warm-up at nominal coefficients, and with each passage's coefficient at
    ``fractions`` of nominal), and the full model's backward-Euler step projected on them. The departure of each
    coefficient from nominal enters through one ``r x r`` matrix per passage, so an ensemble of a few dozen
    members steps in microseconds instead of a sparse solve per member.

    ``energy``: the fraction of the snapshots' energy the modes keep; ``max_modes`` caps ``r``.
    """

    def __init__(self, full: CooledPart, *, dt_snap: float = 5.0, t_snap: float | None = None,
                 fractions: Sequence[float] = (0.2, 0.6), loads: Sequence[float] = (1.0,),
                 energy: float = 1 - 1e-9, max_modes: int = 80):
        self.full = full
        P = full.grid.n_passages
        tau = full.C * full.n / max(float((full.wall @ full.h_nominal).sum()), 1e-30)   # lumped time constant
        t_snap = float(t_snap) if t_snap is not None else 6.0 * tau
        steps = max(4, int(round(t_snap / dt_snap)))
        cases = [full.h_nominal.copy()]
        for p in range(P):
            for f in fractions:
                h = full.h_nominal.copy()
                h[p] *= f
                cases.append(h)
        snaps = []
        for ld in loads:
            for h in cases:
                T = np.full(full.n, full.T_in)
                for _ in range(steps):
                    T = full.step(T, h, dt_snap, ld)
                    snaps.append(T - full.T_in)
        S = np.array(snaps).T
        U, s, _ = np.linalg.svd(S, full_matrices=False)
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        r = int(min(max_modes, np.searchsorted(e, energy) + 1))
        self.Phi = U[:, :r]
        self.r, self.energy_kept, self.tau = r, float(e[r - 1]), tau
        Phi = self.Phi
        self.Kr = Phi.T @ (full.K @ Phi)
        self.D0r = Phi.T @ ((full.wall @ full.h_nominal)[:, None] * Phi)
        self.Wr = np.stack([Phi.T @ (full.wall[:, p][:, None] * Phi) for p in range(P)])     # (P, r, r)
        self.sr = Phi.T @ full.source
        self._inv: dict[float, np.ndarray] = {}

    def _A_inv(self, dt: float) -> np.ndarray:
        if dt not in self._inv:
            self._inv[dt] = np.linalg.inv(np.eye(self.r) + (dt / self.full.C) * (self.Kr + self.D0r))
        return self._inv[dt]

    def step(self, a: np.ndarray, h: np.ndarray, dt: float, load: float = 1.0) -> np.ndarray:
        """Advance the coordinates ``a`` (r, m) by ``dt`` [s] with coefficients ``h`` (P, m)."""
        dh = self.full.h_nominal[:, None] - np.asarray(h, float)                           # (P, m)
        corr = np.einsum("prs,sm,pm->rm", self.Wr, a, dh)
        return self._A_inv(float(dt)) @ (a + (dt / self.full.C) * (load * self.sr[:, None] + corr))

    def field(self, a: np.ndarray) -> np.ndarray:
        """Temperature [degC] of the cells from coordinates ``a`` (r,) or (r, m)."""
        return self.full.T_in + self.Phi @ a


def _schedule(load: Any, t: float) -> float:
    if load is None:
        return 1.0
    if callable(load):
        return float(load(t))
    v = 1.0
    for t0, x in sorted(load):
        if t >= t0:
            v = float(x)
    return v


def _blocked(h: np.ndarray, fault: Mapping[str, Any] | None, t: float) -> np.ndarray:
    h = h.copy()
    if fault:
        p, t0 = int(fault["passage"]) - 1, float(fault["t"])
        ramp, frac = float(fault.get("ramp", 60.0)), float(fault.get("fraction", 0.3))
        if t > t0:
            s = min(1.0, (t - t0) / ramp) if ramp > 0 else 1.0
            h[p] *= 1.0 - s * (1.0 - frac)
    return h


def simulate_readings(plant: CooledPart, sensors: Sequence[Sequence[float]], *, t_end: float, dt: float = 1.0,
                      noise: float = 0.05, fault: Mapping[str, Any] | None = None, load: Any = None,
                      substeps: int = 1, seed: int = 0) -> dict[str, Any]:
    """A recording from the plant: thermocouples every ``dt`` [s] until ``t_end`` [s], Gaussian noise of
    ``noise`` [K]. ``fault``: ``{"passage": p (1-based), "t": s, "ramp": s, "fraction": h_final / h_nominal}``.
    ``load``: a factor on the heat flux, constant, ``f(t)`` or ``[(t, factor), ...]``. The part starts at the
    coolant temperature.

    Returns
    -------
    rec : dict
        ``t`` (n,) [s], ``y`` (n, s) [degC], ``load`` (n,), ``h_true`` (n, P) [W/(m2 K)], ``sensors`` (s, 3) [m],
        ``noise`` [K], ``fault``, ``simulated`` (True).
    """
    rng = np.random.default_rng(seed)
    idx = plant.sensor_cells(sensors)
    T = np.full(plant.n, plant.T_in)
    ts = np.arange(1, int(round(t_end / dt)) + 1) * dt
    Y, L, Hs = [], [], []
    for t in ts:
        for j in range(substeps):
            tj = t - dt + (j + 1) * dt / substeps
            T = plant.step(T, _blocked(plant.h_nominal, fault, tj), dt / substeps, _schedule(load, tj))
        Y.append(T[idx] + rng.normal(0.0, noise, len(idx)))
        L.append(_schedule(load, t))
        Hs.append(_blocked(plant.h_nominal, fault, t))
    return {"t": ts, "y": np.array(Y), "load": np.array(L), "h_true": np.array(Hs),
            "sensors": np.asarray(sensors, float).reshape(-1, 3), "noise": float(noise),
            "fault": dict(fault) if fault else None, "simulated": True, "T_final": T}


def run_twin(model: CooledPart, rec: Mapping[str, Any], *, members: int = 32, h_spread: float = 0.3,
             q_log_h: float = 0.01, q_T: float = 0.002, inflation: float = 1.01, alarm_fraction: float = 0.7,
             alarm_samples: int = 5, burn_in: float = 0.2, reduced: ReducedPart | None = None,
             seed: int = 0) -> dict[str, Any]:
    """Keep the model in step with the recording ``rec`` (``t``, ``y``, ``sensors``, ``noise``; ``load``
    optional) and watch the passages.

    Parameters
    ----------
    model : CooledPart
    rec : dict
        As :func:`simulate_readings` returns, or the same keys from a live or recorded source.
    members : int
        Ensemble size.
    h_spread : float
        Prior spread of ``log h`` (0.3: about 30 %).
    q_log_h : float [1/sqrt(s)]
        Random-walk rate of ``log h``.
    q_T : float [K/sqrt(s)]
        Model-error noise on each of the reduced model's coordinates.
    alarm_fraction, alarm_samples : float, int
        A passage alarms when ``exp(mean + 2 std)`` of its ``h`` stays below ``alarm_fraction`` of nominal for
        ``alarm_samples`` samples in a row.
    burn_in : float
        Fraction of the recording left out of the tracking check (the filter's start-up).
    reduced : ReducedPart, optional
        The reduced model the filter steps (built from ``model`` when not given).

    Returns
    -------
    out : dict
        ``t``; ``y`` measured and ``y_pred`` (forecast mean, before each update) per sensor [degC]; ``h_mean``,
        ``h_lo``, ``h_hi`` (n, P) [W/(m2 K)] (2-sigma band); ``alarms`` (passage, t raised); ``T_mean`` and
        ``T_std`` (n_cells,) at the end [degC]; ``innovation_rms`` [K]; ``checks``.
    """
    rng = np.random.default_rng(seed)
    t, Y = np.asarray(rec["t"], float), np.asarray(rec["y"], float)
    load = np.asarray(rec.get("load", np.ones(len(t))), float)
    sigma = float(rec["noise"])
    rom = reduced if reduced is not None else ReducedPart(model)
    idx = model.sensor_cells(rec["sensors"])
    Hs = rom.Phi[idx]                                                    # sensors observe their cells
    P, r, m = model.grid.n_passages, rom.r, int(members)
    a = rng.normal(0.0, 0.01, (r, m))                                    # the part starts at the coolant temperature
    g0 = np.log(model.h_nominal)[:, None]
    G = g0 + rng.normal(0.0, h_spread, (P, m))
    R = sigma ** 2
    out = {k: [] for k in ("y_pred", "h_mean", "h_lo", "h_hi", "nis")}
    alarms, run = [], np.zeros(P, int)
    raised: dict[int, float] = {}
    prev = 0.0
    for i, ti in enumerate(t):
        dt = ti - prev
        prev = ti
        a = rom.step(a, np.exp(G), dt, load[i]) + rng.normal(0.0, q_T * math.sqrt(dt), (r, m))
        G = G + rng.normal(0.0, q_log_h * math.sqrt(dt), (P, m))
        X = np.vstack([a, G])
        mu = X.mean(1, keepdims=True)
        X = mu + inflation * (X - mu)
        Hx = model.T_in + Hs @ X[:r]
        A = X - X.mean(1, keepdims=True)
        B = Hx - Hx.mean(1, keepdims=True)
        Syy = B @ B.T / (m - 1) + R * np.eye(len(idx))
        Sxy = A @ B.T / (m - 1)
        innov = Y[i] - Hx.mean(1)
        out["y_pred"].append(Hx.mean(1))
        out["nis"].append(float(innov @ np.linalg.solve(Syy, innov)))
        D = Y[i][:, None] + rng.normal(0.0, sigma, (len(idx), m)) - Hx
        X = X + Sxy @ np.linalg.solve(Syy, D)
        a, G = X[:r], np.clip(X[r:], g0 - 6, g0 + 0.7)
        gm, gs = G.mean(1), G.std(1, ddof=1)
        out["h_mean"].append(np.exp(gm))
        out["h_lo"].append(np.exp(gm - 2 * gs))
        out["h_hi"].append(np.exp(gm + 2 * gs))
        low = np.exp(gm + 2 * gs) < alarm_fraction * model.h_nominal
        run = np.where(low, run + 1, 0)
        for p in np.flatnonzero(run == alarm_samples):
            if p + 1 not in raised:
                raised[p + 1] = float(ti)
                alarms.append({"passage": int(p + 1), "t": float(ti),
                               "h_estimate": float(np.exp(gm[p])), "h_nominal": float(model.h_nominal[p])})
    Tf = rom.field(a)
    res = {k: np.array(v) for k, v in out.items()}
    k0 = int(burn_in * len(t))
    innov = Y[k0:] - res["y_pred"][k0:]
    rms = float(np.sqrt(np.mean(innov ** 2)))
    res.update(t=t, y=Y, alarms=alarms, T_mean=Tf.mean(1), T_std=Tf.std(1, ddof=1), innovation_rms=rms,
               sensor_cells=idx, simulated=bool(rec.get("simulated", False)), modes=rom.r)
    checks = [{"code": "twin.tracks", "status": "pass" if rms <= 2.0 * sigma else "fail", "value": rms,
               "detail": f"rms of the sensor innovations after the first {burn_in:.0%} of the record: "
                         f"{rms:.3g} K against 2 x noise = {2 * sigma:.3g} K"}]
    fault = rec.get("fault")
    if fault:
        p0, tf = int(fault["passage"]), float(fault["t"])
        right = [a for a in alarms if a["passage"] == p0 and a["t"] > tf]
        wrong = [a for a in alarms if a["passage"] != p0 or a["t"] <= tf]
        ok = bool(right) and not wrong
        detail = (f"blocked passage {p0} at t = {tf:g} s: " +
                  (f"alarm at t = {right[0]['t']:g} s ({right[0]['t'] - tf:g} s after); " if right else "no alarm; ") +
                  (f"{len(wrong)} false alarm(s): {wrong}" if wrong else "no false alarm"))
        for code in ("twin.alarms", "verify.replay"):
            checks.append({"code": code, "status": "pass" if ok else "fail",
                           "value": right[0]["t"] - tf if right else None, "detail": detail})
    else:
        checks.append({"code": "twin.alarms", "status": "inconclusive", "value": len(alarms),
                       "detail": "no known fault in the record: alarms cannot be checked against one"})
    res["checks"] = checks
    return res


def field_view_of(model: CooledPart, T: np.ndarray, T_std: np.ndarray | None = None, *,
                  sensors: Sequence[Sequence[float]] = (), alarms: Sequence[Mapping[str, Any]] = (),
                  title: str = "Digital twin: temperature"):
    """A field view (``pinneapple-field-view/v1``) of the field on the part's cells, with the sensors and the
    walls of alarmed passages marked."""
    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_tools.visualization.studio.fieldview import field_view
    g = model.grid
    nx, ny, nz = g.shape
    pid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k                    # noqa: E731
    ci, cj, ck = model.cells[:, 0], model.cells[:, 1], model.cells[:, 2]
    conn = np.stack([pid(ci, cj, ck), pid(ci + 1, cj, ck), pid(ci + 1, cj + 1, ck), pid(ci, cj + 1, ck),
                     pid(ci, cj, ck + 1), pid(ci + 1, cj, ck + 1), pid(ci + 1, cj + 1, ck + 1),
                     pid(ci, cj + 1, ck + 1)], 1)
    used, inv = np.unique(conn, return_inverse=True)
    k_ = used % (nz + 1)
    j_ = (used // (nz + 1)) % (ny + 1)
    i_ = used // ((nz + 1) * (ny + 1))
    pts = g.origin + np.c_[i_, j_, k_] * g.d
    mesh = Mesh(points=pts, blocks=[CellBlock("hexahedron", inv.reshape(-1, 8))])
    fields = [{"name": "T", "values": T, "label": "Temperature (twin estimate)", "unit": "degC", "where": "cell"}]
    if T_std is not None:
        fields.append({"name": "T_std", "values": T_std, "label": "Temperature spread (1 sigma)", "unit": "K",
                       "kind": "error", "where": "cell"})
    C = g.centres()[g.solid]
    markers = [{"name": f"sensor {i + 1}", "kind": "load", "points": [p]} for i, p in enumerate(sensors)]
    for a in alarms:
        w = model.wall[:, a["passage"] - 1] > 0
        markers.append({"name": f"passage {a['passage']}: clogging alarm", "kind": "hotspot", "points": C[w],
                        "detail": f"h {a['h_estimate']:.0f} of {a['h_nominal']:.0f} W/(m2 K) at t = {a['t']:g} s"})
    return field_view(mesh, fields, markers=markers, title=title, length_unit="m")
