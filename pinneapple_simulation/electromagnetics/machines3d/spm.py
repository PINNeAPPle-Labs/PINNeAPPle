"""Surface-PM machine in 3-D on edge elements: the 2-D polar mesh of
:class:`~pinneapple_simulation.electromagnetics.machine.SPMMachine` extruded along the stack, with magnet skew,
magnet end regions and the Maxwell-stress torque.

Every node ring of the cross-section carries ``n_theta`` equally spaced nodes (the rotor core starts at
``r_inner``, a flux-tangential boundary standing in for the shaft), so turning the rotor by a whole number of
angular steps ``2 pi / n_theta`` maps the mesh onto itself. Rotor position and skew are therefore relabellings of
the rotor elements of each z layer (which tetrahedra are magnet, which are air, and the direction of their
remanence), on one mesh for every angle and every skew: cogging and its skew dependence carry no remeshing
noise, and the multi-slice 2-D model (:meth:`SPM3D.multislice`) is the same discretisation with one layer per slice.

Skew (of the magnets, relative to the stator): ``skew(z)`` gives the rotor offset of the layer at axial position
``z``, rounded to the angular step; :func:`step_skew` and :func:`continuous_skew` give the usual ones. End regions:
``end_length`` of air beyond each end of the stack (``0``: flux-parallel end planes, the 2-D condition).
Boundaries: ``n x A = 0`` (``B . n = 0``) on the inner and outer cylinders and the end planes.

Iron is linear (``mu_r_iron``) in this model; magnets are linear with ``mu_r_magnet`` and remanence ``Br``
(parallel or radial magnetisation of the 2-D machine). No winding currents: cogging and the no-load field.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..fem3d.nedelec import NU0, EdgeSpace, solve_magnetostatic, tree_gauge
from ..machine import SPMMachine
from ..mesh import Ring, polar_mesh
from .extrude import extrude

__all__ = ["SPM3D", "step_skew", "continuous_skew", "no_skew"]

TWO_PI = 2 * math.pi


def no_skew(z: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(z, float))


def step_skew(total: float, steps: int, length: float) -> Callable[[np.ndarray], np.ndarray]:
    """``steps`` axial segments of equal length, segment ``k`` turned by ``total (k - (steps-1)/2) / (steps-1)``
    (the convention of :func:`~pinneapple_simulation.electromagnetics.machines.robustness.step_offsets` with step
    angle ``total / (steps - 1)``)."""
    def f(z):
        z = np.asarray(z, float)
        k = np.clip(np.floor((z + length / 2) / length * steps), 0, steps - 1)
        return total * (k - (steps - 1) / 2) / max(steps - 1, 1)
    return f


def continuous_skew(total: float, length: float) -> Callable[[np.ndarray], np.ndarray]:
    """Linear skew from ``-total/2`` at ``z = -L/2`` to ``+total/2`` at ``z = +L/2``."""
    def f(z):
        return total * np.clip(np.asarray(z, float) / length, -0.5, 0.5)
    return f


@dataclass
class SPM3D:
    machine: SPMMachine = field(default_factory=SPMMachine)
    n_theta: int = 360
    r_inner: float | None = None          # default 0.4 r_rotor
    core_rings: int = 3
    stack_layers: int = 8
    end_length: float = 0.0
    end_layers: int = 0
    mu_r_iron: float = 1000.0

    def __post_init__(self):
        m = self.machine
        if self.n_theta % (2 * m.pole_pairs) or (m.slots and self.n_theta % m.slots):
            raise ValueError("n_theta must be a multiple of 2*pole_pairs and of slots")
        self.step = TWO_PI / self.n_theta
        r_in = 0.4 * m.r_rotor if self.r_inner is None else self.r_inner
        ref = SPMMachine(**{**_fields(m), "n_theta": self.n_theta, "rotor_core_inner": 0.0})
        rings = [Ring(r_in + (m.r_rotor - r_in) * k / self.core_rings, self.n_theta, rotating=True)
                 for k in range(self.core_rings + 1)]
        rings += [Ring(r.r, self.n_theta, rotating=r.rotating) for r in ref._rings if r.r > m.r_rotor + 1e-12]
        self.rings = rings
        self.mesh2d = polar_mesh(rings, ref._classify, theta=0.0, region_names=ref.region_names())
        self._ref = ref
        c = self.mesh2d.centroids()
        self._r = np.hypot(c[:, 0], c[:, 1])
        self._phi = np.mod(np.arctan2(c[:, 1], c[:, 0]), TWO_PI)
        rot = np.array([rings[k].rotating and rings[k + 1].rotating for k in range(len(rings) - 1)])
        self._rot = rot[self.mesh2d.layer]
        L, e = m.length, self.end_length
        zs = np.linspace(-L / 2, L / 2, self.stack_layers + 1)
        if e > 0 and self.end_layers > 0:
            ze = np.linspace(0.0, e, self.end_layers + 1)[1:]
            zs = np.r_[-L / 2 - ze[::-1], zs, L / 2 + ze]
        self.z = zs
        # cut order: even positions along every ring before odd ones (no handedness, see extrude)
        rank = np.arange(len(self.mesh2d.points))
        for ring in self.mesh2d.rings:
            rank[ring] = ring + (np.arange(len(ring)) % 2) * len(self.mesh2d.points)
        self.mesh, self._tri, self._lay = extrude(self.mesh2d.points, self.mesh2d.tris, zs, rank=np.argsort(np.argsort(rank)))
        zc = 0.5 * (zs[:-1] + zs[1:])
        self._zc = zc
        self._in_stack = np.abs(zc) < L / 2
        self.space = EdgeSpace(self.mesh)
        self._dir = (self.mesh.boundary_edges, np.zeros(len(self.mesh.boundary_edges)))
        gk, g, r0 = m.gap_layers, m.airgap, m.r_magnet
        self.annuli = [(r0, r0 + g * gk / (2 * gk + 1)), (r0 + g * (gk + 1) / (2 * gk + 1), m.r_bore)]

    # ------------------------------------------------------------------ labels and materials
    def labels2d(self, theta: float) -> list[str]:
        """Region of every 2-D element with the rotor turned by ``theta`` (rounded to the angular step)."""
        th = round(theta / self.step) * self.step
        phi = np.where(self._rot, np.mod(self._phi - th, TWO_PI), self._phi)
        return [self._ref._classify(int(ly), float(r), float(p))
                for ly, r, p in zip(self.mesh2d.layer, self._r, phi, strict=True)]

    def materials(self, theta: float, skew: Callable = no_skew) -> tuple[np.ndarray, np.ndarray]:
        """Reluctivity and remanence per tetrahedron for rotor angle ``theta`` and the magnet ``skew``."""
        m = self.machine
        nu = np.full(len(self.mesh.tets), NU0)
        Br = np.zeros((len(self.mesh.tets), 3))
        tau = math.pi / m.pole_pairs
        offsets = np.round((theta + np.asarray(skew(self._zc), float)) / self.step) * self.step
        cache: dict[float, list[str]] = {}
        c = self.mesh.centroids
        for k in range(len(self._zc)):
            sel = self._lay == k
            if not self._in_stack[k]:
                continue                                    # end region: air
            th = float(offsets[k])
            if th not in cache:
                cache[th] = self.labels2d(th)
            lab = np.asarray(cache[th], dtype=object)[self._tri[sel]]
            idx = np.flatnonzero(sel)
            iron = np.isin(lab, ("stator_iron", "rotor_core"))
            nu[idx[iron]] = NU0 / self.mu_r_iron
            for p in range(2 * m.pole_pairs):
                mk = lab == f"magnet{p}"
                if not mk.any():
                    continue
                t = idx[mk]
                nu[t] = NU0 / m.mu_r_magnet
                sgn = 1.0 if p % 2 == 0 else -1.0
                ang = (np.arctan2(c[t, 1], c[t, 0]) if m.magnetization == "radial"
                       else np.full(len(t), p * tau + th))
                Br[t, 0] = sgn * m.Br * np.cos(ang)
                Br[t, 1] = sgn * m.Br * np.sin(ang)
        return nu, Br

    # ------------------------------------------------------------------ solve and post-process
    def _system(self):
        """Tree-gauged unknowns and a factorisation of the matrix with the whole magnet ring at the magnet
        reluctivity (magnets and inter-magnet air differ only by ``mu_r_magnet``): the preconditioner of every
        solve, since the mesh and the iron never change."""
        if getattr(self, "_sys", None) is None:
            mesh, sp = self.mesh, self.space
            de = self._dir[0]
            tree = tree_gauge(mesh, de, np.unique(mesh.edges[de]))
            u = np.setdiff1d(np.setdiff1d(np.arange(sp.n), de), tree)
            nu, _ = self.materials(0.0)
            c = mesh.centroids
            r = np.hypot(c[:, 0], c[:, 1])
            m = self.machine
            ring = (r > m.r_rotor) & (r < m.r_magnet) & (np.abs(c[:, 2]) < m.length / 2)
            nu = nu.copy()
            nu[ring] = NU0 / m.mu_r_magnet
            K = sp.stiffness(nu)[u][:, u].tocsr()
            self._sys = (u, _factor(K))
        return self._sys

    def solve(self, theta: float = 0.0, skew: Callable = no_skew, a0=None, rtol: float = 1e-10, direct=None):
        """No-load field at rotor angle ``theta``: preconditioned CG on the tree-gauged system (a few iterations:
        the preconditioner is the factorised matrix of :meth:`_system`). ``direct=True`` falls back to
        :func:`~pinneapple_simulation.electromagnetics.fem3d.nedelec.solve_magnetostatic` (ungauged CG)."""
        nu, Br = self.materials(theta, skew)
        if direct is False:
            return solve_magnetostatic(self.mesh, nu, Br=Br, dirichlet=self._dir, space=self.space, a0=a0)
        import scipy.sparse.linalg as spla

        u, solve0 = self._system()
        K = self.space.stiffness(nu)[u][:, u].tocsr()
        b = self.space.magnet_load(nu, Br)[u]
        P = spla.LinearOperator(K.shape, matvec=solve0, dtype=float)
        it = [0]
        x, info = spla.cg(K, b, x0=None if a0 is None else a0[u], rtol=rtol, M=P, maxiter=200,
                          callback=lambda _: it.__setitem__(0, it[0] + 1))
        if info:
            raise RuntimeError(f"preconditioned CG did not converge (info={info})")
        a = np.zeros(self.space.n)
        a[u] = x
        return _Solution(a, self.space.curl(a), {"iterations": it[0], "unknowns": len(u)})

    def release(self) -> None:
        """Free the factorised preconditioner (PARDISO holds it in MKL memory until freed)."""
        sys_ = getattr(self, "_sys", None)
        if sys_ is not None and hasattr(sys_[1], "free"):
            sys_[1].free()
        self._sys = None

    def torque(self, sol) -> float:
        """Arkkio (Maxwell-stress) torque, N m, over the two gap annuli on either side of the moving band (as in
        the 2-D model), over the whole axial length of the mesh; the length-weighted mean of the two."""
        c = self.mesh.centroids
        r = np.hypot(c[:, 0], c[:, 1])
        er = c[:, :2] / r[:, None]
        Bxy = sol.B[:, :2]
        Br_ = (Bxy * er).sum(1)
        Bt = Bxy[:, 1] * er[:, 0] - Bxy[:, 0] * er[:, 1]
        num = den = 0.0
        for r1, r2 in self.annuli:
            s = (r > r1) & (r < r2)
            T = float(np.sum(NU0 * r[s] * Br_[s] * Bt[s] * self.space.vol[s]) / (r2 - r1))
            num += T * (r2 - r1)
            den += r2 - r1
        return num / den

    def gap_flux(self, sol, pole_angle: float = 0.0) -> float:
        """Radial flux density averaged over the gap annuli within half a pole pitch of ``pole_angle`` and over
        the stack (T)."""
        m = self.machine
        c = self.mesh.centroids
        r = np.hypot(c[:, 0], c[:, 1])
        phi = np.arctan2(c[:, 1], c[:, 0])
        d = np.abs((phi - pole_angle + math.pi) % TWO_PI - math.pi)
        s = (r > self.annuli[0][0]) & (r < self.annuli[1][1]) & (d < math.pi / (2 * m.pole_pairs))
        s &= np.abs(c[:, 2]) < m.length / 2
        Brad = (sol.B[s, :2] * (c[s, :2] / r[s, None])).sum(1)
        return float(np.sum(Brad * self.space.vol[s]) / self.space.vol[s].sum())

    def cogging(self, thetas, skew: Callable = no_skew) -> np.ndarray:
        """No-load torque at each rotor angle (warm-started CG)."""
        out, a = [], None
        for th in thetas:
            s = self.solve(th, skew, a0=a)
            a = s.a
            out.append(self.torque(s))
        return np.array(out)

    # ------------------------------------------------------------------ multi-slice 2-D on the same mesh
    def slice_model(self) -> SPM3D:
        """One straight layer of the stack with flux-parallel end planes: the 2-D model on the same mesh."""
        return SPM3D(self.machine, n_theta=self.n_theta, r_inner=self.rings[0].r, core_rings=self.core_rings,
                     stack_layers=1, end_length=0.0, end_layers=0, mu_r_iron=self.mu_r_iron)

    def multislice(self, thetas, offsets, slice_model: SPM3D | None = None) -> np.ndarray:
        """Multi-slice torque: the mean over the slices at rotor offsets ``offsets`` of the straight one-layer
        model, times the stack length / slice length ratio (the torque of the full stack)."""
        sm = slice_model or self.slice_model()
        scale = self.machine.length / (sm.z[-1] - sm.z[0])
        off, count = np.unique(np.round(np.asarray(offsets, float) / sm.step).astype(int), return_counts=True)
        res = []
        for th in thetas:
            vals = np.array([sm.torque(sm.solve(th + k * sm.step)) for k in off])
            res.append(scale * float(np.sum(vals * count) / count.sum()))
        return np.array(res)


@dataclass
class _Solution:
    a: np.ndarray
    B: np.ndarray
    info: dict


def _factor(K):
    """``x -> K^-1 x`` by MKL PARDISO (pypardiso, when importable) or SuperLU."""
    try:
        _mkl_path()
        import pypardiso

        solver = pypardiso.PyPardisoSolver()
        solver.factorize(K)

        def solve(v):
            return solver.solve(K, np.asarray(v, float))
        solve.free = lambda: solver.free_memory(everything=True)   # MKL keeps the factor until told
        return solve
    except Exception:  # noqa: BLE001 - optional accelerator; SuperLU is always there
        import scipy.sparse.linalg as spla

        lu = spla.splu(K.tocsc(), permc_spec="COLAMD")
        return lu.solve


def _mkl_path() -> None:
    """pypardiso looks for ``libmkl_rt.so.2``; point it at the MKL runtime pip installed if that is another one."""
    import glob
    import os
    import sys

    if os.environ.get("PYPARDISO_MKL_RT"):
        return
    for d in (os.path.join(sys.prefix, "lib"), "/usr/local/lib", "/usr/lib"):
        hits = sorted(glob.glob(os.path.join(d, "libmkl_rt.so*")))
        if hits:
            os.environ["PYPARDISO_MKL_RT"] = hits[-1]
            return


def _fields(m: SPMMachine) -> dict:
    import dataclasses
    return {f.name: getattr(m, f.name) for f in dataclasses.fields(m) if f.init}


