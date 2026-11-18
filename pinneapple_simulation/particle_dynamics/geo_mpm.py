"""Material point method for geomechanics: granular collapse, landslides, dam breaks, soft bodies.

MLS-MPM with APIC transfer and quadratic B-splines (Hu et al. 2018) on a rectangular grid in physical units,
in 2-D or 3-D, vectorised in PyTorch. Materials (one per body, several bodies per run):

* ``Sand(friction_deg, E, nu, rho)``: Drucker-Prager plasticity with the return mapping in Hencky strain of
  Klár et al. (2016), cohesionless, non-associative (no dilation); in 2-D, plane strain by default (the
  out-of-plane elastic stretch is tracked, so the total out-of-plane strain stays zero, as in a laboratory flume)
  with the cone matched to Mohr-Coulomb in plane strain; in 3-D, matched in triaxial compression;
* ``Fluid(K, rho, gamma)``: weakly compressible, pressure from J (Tait-like);
* ``Elastic(E, nu, rho)``: fixed-corotated hyperelastic solid.

Walls are the domain faces (``"slip"``, ``"sticky"`` or a Coulomb friction coefficient); bodies are seeded with
``ppc`` particles per cell per axis. Output frames follow GeoTaichi's layout
(``<out>/particles/MPMParticle<NNNNNN>.npz`` with ``position``, ``velocity``, ``volume``, ``stress``, ``body``), so
``read_particle_frames`` reads runs from either code.

    from pinneapple_simulation.particle_dynamics.geo_mpm import GeoMPM, Sand, box
    sim = GeoMPM(domain=(1.2, 0.5), dx=0.005, floor="friction", wall_friction=0.4)
    sim.add_body(box((0.0, 0.0), (0.2, 0.2)), Sand(friction_deg=30.0))
    frames = sim.run(t_end=1.0, save_every=0.05, out="runs/collapse")

References: Hu, Fang, Ge, Qu, Zhu, Pradhana, Jiang (2018), ACM TOG 37(4); Klár, Gast, Pradhana, Fu, Schroeder,
Jiang, Teran (2016), ACM TOG 35(4); Stomakhin et al. (2014) for the fixed-corotated model.
"""
from __future__ import annotations

import glob
import itertools
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Sand", "Fluid", "Elastic", "GeoMPM", "box", "read_particle_frames", "runout"]


@dataclass
class Sand:
    """Sand: cohesionless Drucker-Prager plasticity with friction angle, Young's modulus ``E``, Poisson ratio ``nu``
    and density.
    """
    friction_deg: float = 30.0
    E: float = 3.5e5
    nu: float = 0.3
    rho: float = 2000.0


@dataclass
class Fluid:
    K: float = 2.0e5                 # bulk modulus of the weakly compressible fluid, Pa
    rho: float = 1000.0
    gamma: float = 7.0


@dataclass
class Elastic:
    E: float = 1.0e5
    nu: float = 0.3
    rho: float = 1000.0


def box(lo: Sequence[float], hi: Sequence[float]):
    """A rectangular (or box) body: returns an inside-test for ``add_body``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lambda p: np.all((p >= lo) & (p <= hi), axis=1), (lo, hi)


def _lame(E: float, nu: float) -> tuple[float, float]:
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def _svd_rot(F: torch.Tensor):
    """SVD with U, V proper rotations (signed singular values, as in Klár et al.)."""
    U, S, Vh = torch.linalg.svd(F)
    neg_u = torch.linalg.det(U) < 0
    neg_v = torch.linalg.det(Vh) < 0
    U = U.clone()
    Vh = Vh.clone()
    S = S.clone()
    U[neg_u, :, -1] *= -1
    S[neg_u, -1] *= -1
    Vh[neg_v, -1, :] *= -1
    S[neg_v, -1] *= -1
    return U, S, Vh


class GeoMPM:
    """2D/3D MLS-MPM simulator in physical units: bodies, walls (slip, sticky or friction), a stable step from the CFL
    number and recorded frames.
    """
    def __init__(self, domain: Sequence[float], dx: float, gravity: Sequence[float] | None = None,
                 floor: str = "friction", walls: str = "slip", wall_friction: float = 0.5, cfl: float = 0.3,
                 flip: float = 0.0, plane_strain: bool = True, dtype=torch.float64):
        self.domain = np.asarray(domain, float)
        self.d = len(self.domain)
        self.dx = float(dx)
        self.n = np.ceil(self.domain / dx).astype(int) + 1               # grid nodes per axis
        g = np.zeros(self.d)
        g[-1] = -9.81
        self.g = torch.tensor(gravity if gravity is not None else g, dtype=dtype)
        self.floor, self.walls, self.mu_w = floor, walls, float(wall_friction)
        self.cfl, self.dtype = cfl, dtype
        self.bodies: list[tuple[str, object]] = []
        self.x = torch.zeros((0, self.d), dtype=dtype)
        self.mat: list[int] = []
        self.flip = flip                 # unused (APIC); kept for API symmetry with PIC/FLIP blends
        self.plane_strain = plane_strain  # 2-D sand: zero total out-of-plane strain (laboratory columns, slopes)

    # ---------------------------------------------------------------- setup
    def add_body(self, shape, material, ppc: int = 2, velocity: Sequence[float] | None = None,
                 name: str | None = None) -> int:
        """Seed ``ppc``**d particles per cell inside ``shape`` (from ``box`` or any (inside_fn, (lo, hi)))."""
        inside, (lo, hi) = shape
        h = self.dx / ppc
        axes = [np.arange(lo[i] + h / 2, hi[i], h) for i in range(self.d)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, self.d)
        P = P[inside(P)]
        P = P[np.all((P > 2 * self.dx) & (P < self.domain - 2 * self.dx), axis=1)]
        k = len(self.bodies)
        self.bodies.append((name or f"body{k}", material))
        n = len(P)
        v = np.zeros((n, self.d)) if velocity is None else np.tile(np.asarray(velocity, float), (n, 1))
        self._append(P, v, k, h ** self.d, material.rho)
        return n

    def _append(self, P, v, k, vol, rho):
        t = lambda a: torch.tensor(a, dtype=self.dtype)                  # noqa: E731
        n = len(P)
        eye = torch.eye(self.d, dtype=self.dtype).expand(n, self.d, self.d)
        if len(self.x) == 0:
            self.x, self.v = t(P), t(v)
            self.F, self.C = eye.clone(), torch.zeros(n, self.d, self.d, dtype=self.dtype)
            self.vol0, self.m = torch.full((n,), vol, dtype=self.dtype), torch.full((n,), vol * rho, dtype=self.dtype)
            self.body = torch.full((n,), k, dtype=torch.long)
            self.Fz = torch.ones(n, dtype=self.dtype)
        else:
            self.x, self.v = torch.cat([self.x, t(P)]), torch.cat([self.v, t(v)])
            self.F = torch.cat([self.F, eye.clone()])
            self.C = torch.cat([self.C, torch.zeros(n, self.d, self.d, dtype=self.dtype)])
            self.vol0 = torch.cat([self.vol0, torch.full((n,), vol, dtype=self.dtype)])
            self.m = torch.cat([self.m, torch.full((n,), vol * rho, dtype=self.dtype)])
            self.body = torch.cat([self.body, torch.full((n,), k, dtype=torch.long)])
            self.Fz = torch.cat([self.Fz, torch.ones(n, dtype=self.dtype)])

    def stable_dt(self) -> float:
        c = 0.0
        for _, m in self.bodies:
            if isinstance(m, Fluid):
                c = max(c, math.sqrt(m.K / m.rho))
            else:
                mu, lam = _lame(m.E, m.nu)
                c = max(c, math.sqrt((lam + 2 * mu) / m.rho))
        vmax = float(self.v.norm(dim=1).max()) if len(self.v) else 0.0
        return self.cfl * self.dx / (c + vmax + 1e-12)

    # ---------------------------------------------------------------- constitutive models
    def _kirchhoff(self) -> torch.Tensor:
        """Kirchhoff stress per particle; also applies the plastic return mapping to F (sand)."""
        n, d = len(self.x), self.d
        tau = torch.zeros(n, d, d, dtype=self.dtype)
        eye = torch.eye(d, dtype=self.dtype)
        for k, (_, m) in enumerate(self.bodies):
            sel = self.body == k
            if not bool(sel.any()):
                continue
            F = self.F[sel]
            if isinstance(m, Fluid):
                J = torch.linalg.det(F).clamp(min=0.1)
                p = m.K / m.gamma * (J ** (-m.gamma) - 1.0)
                tau[sel] = -(J * p)[:, None, None] * eye
                self.F[sel] = J.pow(1.0 / d)[:, None, None] * eye                # no shear memory
            elif isinstance(m, Elastic):
                mu, lam = _lame(m.E, m.nu)
                U, S, Vh = _svd_rot(F)
                R = U @ Vh
                J = torch.linalg.det(F)
                tau[sel] = 2 * mu * (F - R) @ F.mT + (lam * (J - 1) * J)[:, None, None] * eye
            else:                                                                # Sand: Drucker-Prager
                mu, lam = _lame(m.E, m.nu)
                U, S, Vh = _svd_rot(F)
                eps = torch.log(S.clamp(min=1e-6))
                if d == 2 and self.plane_strain:          # out-of-plane elastic stretch: total strain stays zero
                    eps = torch.cat([eps, torch.log(self.Fz[sel].clamp(min=1e-6))[:, None]], 1)
                dd = eps.shape[1]
                tr = eps.sum(1)
                dev = eps - tr[:, None] / dd
                dn = dev.norm(dim=1)
                phi = math.radians(m.friction_deg)
                if dd > d:          # Drucker-Prager matched to Mohr-Coulomb in plane strain
                    alpha = math.sqrt(2.0) * math.tan(phi) / math.sqrt(9 + 12 * math.tan(phi) ** 2)
                else:               # matched in triaxial compression (Klár et al. 2016)
                    alpha = math.sqrt(2.0 / 3.0) * 2 * math.sin(phi) / (3 - math.sin(phi))
                dgamma = dn + (dd * lam + 2 * mu) / (2 * mu) * tr * alpha
                H = eps.clone()
                tension = tr >= 0
                H[tension] = 0.0                                                 # case II: cone tip
                yield_ = (~tension) & (dgamma > 0)
                H[yield_] = eps[yield_] - (dgamma[yield_] / dn[yield_].clamp(min=1e-12))[:, None] * dev[yield_]
                if dd > d:
                    self.Fz[sel] = torch.exp(H[:, d])
                S_new = torch.exp(H[:, :d])
                self.F[sel] = U @ torch.diag_embed(S_new) @ Vh
                tau_diag = (2 * mu * H + lam * H.sum(1, keepdim=True))[:, :d]
                tau[sel] = U @ torch.diag_embed(tau_diag) @ U.mT
        return tau

    # ---------------------------------------------------------------- one step
    def _stencil(self):
        fx = self.x / self.dx
        base = torch.floor(fx - 0.5).long()
        f = fx - base.to(self.dtype)                                             # in [0.5, 1.5)
        w = torch.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1.0) ** 2, 0.5 * (f - 0.5) ** 2], -1)
        return base, f, w

    def _flat(self, node: torch.Tensor) -> torch.Tensor:
        idx = node[:, 0]
        for i in range(1, self.d):
            idx = idx * int(self.n[i]) + node[:, i]
        return idx

    def step(self, dt: float) -> None:
        d, n_nodes = self.d, int(np.prod(self.n))
        vol = self.vol0 * torch.linalg.det(self.F)
        tau = self._kirchhoff()
        stress = -dt * 4.0 / self.dx ** 2 * self.vol0[:, None, None] * tau
        affine = stress + self.m[:, None, None] * self.C
        base, f, w = self._stencil()
        gm = torch.zeros(n_nodes, dtype=self.dtype)
        gp = torch.zeros(n_nodes, d, dtype=self.dtype)
        offs = list(itertools.product(range(3), repeat=d))
        cache = []
        for off in offs:
            o = torch.tensor(off)
            wt = torch.ones(len(self.x), dtype=self.dtype)
            for i in range(d):
                wt = wt * w[:, i, off[i]]
            dpos = (o.to(self.dtype) - f) * self.dx
            idx = self._flat(base + o)
            cache.append((idx, wt, dpos))
            gm.index_add_(0, idx, wt * self.m)
            gp.index_add_(0, idx, wt[:, None] * (self.m[:, None] * self.v + (affine @ dpos[..., None])[..., 0]))
        has = gm > 0
        gv = torch.zeros_like(gp)
        gv[has] = gp[has] / gm[has, None] + dt * self.g
        self._boundary(gv)
        nv = torch.zeros_like(self.v)
        nC = torch.zeros_like(self.C)
        for idx, wt, dpos in cache:
            g = gv[idx]
            nv += wt[:, None] * g
            nC += 4.0 / self.dx ** 2 * wt[:, None, None] * g[:, :, None] * dpos[:, None, :]
        self.v, self.C = nv, nC
        self.F = (torch.eye(d, dtype=self.dtype) + dt * nC) @ self.F
        self.x = self.x + dt * nv
        lo, hi = 1.5 * self.dx, torch.tensor(self.domain, dtype=self.dtype) - 1.5 * self.dx
        self.x = torch.max(torch.min(self.x, hi), torch.full_like(self.x, lo))
        self._vol = vol

    def _boundary(self, gv: torch.Tensor) -> None:
        """Domain faces: the floor (last axis, low side) and the other walls; 2 nodes thick."""
        if not hasattr(self, "_walls"):
            grids = torch.meshgrid(*[torch.arange(int(k)) for k in self.n], indexing="ij")
            self._walls = []
            for ax in range(self.d):
                idx = grids[ax].reshape(-1)
                for side, sel in ((-1, idx < 2), (1, idx > int(self.n[ax]) - 3)):
                    kind = self.floor if (ax == self.d - 1 and side == -1) else self.walls
                    self._walls.append((ax, side, torch.nonzero(sel)[:, 0], kind))
        for ax, side, sel, kind in self._walls:
            v = gv[sel]
            vn = v[:, ax] * (-side)                       # positive = moving away from the wall (inward)
            if kind == "sticky":
                v[:] = 0.0
            else:
                into = vn < 0
                vt = v.clone()
                vt[:, ax] = 0.0
                if kind == "friction":
                    vt_n = vt.norm(dim=1, keepdim=True)
                    red = (1 - self.mu_w * (-vn).clamp(min=0)[:, None] / vt_n.clamp(min=1e-12)).clamp(min=0)
                    vt = torch.where(into[:, None], vt * red, vt)
                v = torch.where(into[:, None], vt, v)
            gv[sel] = v

    # ---------------------------------------------------------------- run and output
    def run(self, t_end: float, save_every: float, out: str | None = None, dt: float | None = None,
            log=None) -> list[dict]:
        """Integrate to ``t_end``; returns frames (and writes them under ``out`` when given)."""
        frames, t, k, next_save = [], 0.0, 0, 0.0
        while t <= t_end + 1e-12:
            if t >= next_save - 1e-12:
                fr = self.frame(t)
                frames.append(fr)
                if out:
                    write_particle_frame(out, len(frames) - 1, fr)
                next_save += save_every
                if log:
                    log(f"t = {t:.3f} s, {k} steps")
            h = dt or self.stable_dt()
            h = min(h, next_save - t) if next_save > t else h
            self.step(max(h, 1e-9))
            t += h
            k += 1
        return frames

    def frame(self, t: float) -> dict:
        tau = self._kirchhoff_readonly()
        J = torch.linalg.det(self.F)
        sigma = tau / J[:, None, None]
        return {"time": t, "position": self.x.numpy().copy(), "velocity": self.v.numpy().copy(),
                "volume": (self.vol0 * J).numpy(), "stress": sigma.numpy(), "body": self.body.numpy().copy()}

    def _kirchhoff_readonly(self) -> torch.Tensor:
        F, Fz = self.F.clone(), self.Fz.clone()
        tau = self._kirchhoff()
        self.F, self.Fz = F, Fz
        return tau


# -------------------------------------------------------------------------------------- frames on disk
def write_particle_frame(out: str, i: int, fr: dict) -> str:
    d = os.path.join(out, "particles")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"MPMParticle{i:06d}.npz")
    np.savez(p, **{k: np.asarray(v) for k, v in fr.items()})
    return p


def read_particle_frames(out: str, kind: str = "MPM", allow_pickle: bool = False) -> list[dict]:
    """Frames of a run, from this module or from GeoTaichi (``<out>/particles/<kind>Particle*.npz``; ``kind`` "MPM"
    or "DEM"). GeoTaichi stores some entries (``state_vars``) as pickled objects: pass ``allow_pickle=True`` only
    for files you trust. Velocities stored as (vx, vy, vz) tuples are stacked into (n, d)."""
    files = sorted(glob.glob(os.path.join(out, "particles", f"{kind}Particle*.npz")))
    frames = []
    for f in files:
        z = np.load(f, allow_pickle=allow_pickle)
        fr = {}
        for k in z.files:
            try:
                v = z[k]
            except ValueError:                     # pickled entry with allow_pickle=False
                continue
            if v.dtype == object and v.ndim == 1 and len(v) in (2, 3):
                try:
                    v = np.stack([np.asarray(c, float) for c in v], -1)
                except (TypeError, ValueError):
                    pass
            fr[k] = v
        frames.append(fr)
    return frames


def runout(position: np.ndarray, axis: int = 0, floor_band: float | None = None, quantile: float = 0.995) -> float:
    """Front position of a deposit along ``axis`` (a high quantile, robust to a few stray particles)."""
    p = np.asarray(position)[:, axis]
    return float(np.quantile(p, quantile))
