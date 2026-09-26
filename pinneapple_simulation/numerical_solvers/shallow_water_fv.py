"""2D shallow-water finite-volume solver for free-surface water: walls, gates, virtual sensors,
live health checks, CSV export and comparison with experiments.

Numerics (Toro, "Shock-Capturing Methods for Free-Surface Shallow Flows", Wiley 2001):
- conservative variables (h, hu, hv) on a Cartesian grid, dimensional splitting-free 2D update;
- MUSCL reconstruction of (h, u, v) with the minmod limiter + SSP-RK2 (Heun) in time;
- HLL flux with the wet/dry wave-speed estimates of Toro (dry side: u -+ 2c);
- walls: any cell can be solid; faces between fluid and solid use a mirrored ghost state
  (normal velocity reversed), so walls are drawn with a boolean mask;
- gates: a group of solid cells removed at a given time ("release the gate").

Pressure is hydrostatic by construction of the shallow-water equations, p(z) = rho g (h - z):
the probes report that. For non-hydrostatic impact pressure use the SPH solvers.
Wall forces are measured from the numerical momentum flux through the wall faces, i.e. what the
wall actually does to the water in the scheme, not an after-the-fact formula.

Validated against exact dam-break solutions (Ritter 1892 dry bed; Stoker 1957 wet bed) and
still water (wall force = rho g h^2 / 2 per unit width) in tests/test_shallow_water_fv.py.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

G = 9.81
RHO = 1000.0
DRY = 1e-8


@dataclass
class Sensor:
    name: str
    kind: str  # "depth" | "pressure" | "velocity" | "wall_force"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # pressure probe elevation above the bed
    faces: Optional[List[Tuple[str, int, int, int]]] = None  # wall_force: (axis, i, j, sign)
    values: List[float] = field(default_factory=list)


class ShallowWater2D:
    """Water on an ``nx x ny`` grid of cell size ``dx x dy`` (domain boundary: solid walls)."""

    def __init__(self, nx: int, ny: int, dx: float, dy: Optional[float] = None, *, g: float = G, rho: float = RHO,
                 cfl: float = 0.45, second_order: bool = True):
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy or dx
        self.g, self.rho, self.cfl, self.second_order = g, rho, cfl, second_order
        self.h = np.zeros((nx, ny))
        self.hu = np.zeros((nx, ny))
        self.hv = np.zeros((nx, ny))
        self.solid = np.zeros((nx, ny), bool)
        self.gates: Dict[str, Tuple[np.ndarray, float]] = {}
        self.sensors: List[Sensor] = []
        self.t = 0.0
        self.times: List[float] = []
        self.health: List[Dict[str, float]] = []
        self._v0: Optional[float] = None
        self._wall_flux = {"x": np.zeros((nx + 1, ny)), "y": np.zeros((nx, ny + 1))}

    # ── set-up ──────────────────────────────────────────────────────────
    def centers(self):
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(x, y, indexing="ij")

    def add_wall(self, mask: np.ndarray) -> None:
        self.solid |= mask
        self._clear_solid()

    def add_gate(self, name: str, mask: np.ndarray, release_time: float) -> None:
        self.solid |= mask
        self.gates[name] = (mask.copy(), release_time)
        self._clear_solid()

    def _clear_solid(self):
        for a in (self.h, self.hu, self.hv):
            a[self.solid] = 0.0

    def add_depth_gauge(self, name, x, y):
        self.sensors.append(Sensor(name, "depth", x, y))

    def add_pressure_probe(self, name, x, y, z=0.0):
        self.sensors.append(Sensor(name, "pressure", x, y, z))

    def add_velocity_probe(self, name, x, y):
        self.sensors.append(Sensor(name, "velocity", x, y))

    def add_wall_force_sensor(self, name: str, wall_mask: np.ndarray) -> None:
        """Force (N) on the fluid-facing faces of the cells in ``wall_mask`` (sum over faces)."""
        faces = []
        s = wall_mask
        for i in range(self.nx):
            for j in range(self.ny):
                if not s[i, j]:
                    continue
                if i > 0 and not self.solid[i - 1, j]:
                    faces.append(("x", i, j, +1))  # face x_{i-1/2}: water on the left pushes +x
                if i < self.nx - 1 and not self.solid[i + 1, j]:
                    faces.append(("x", i + 1, j, -1))
                if j > 0 and not self.solid[i, j - 1]:
                    faces.append(("y", i, j, +1))
                if j < self.ny - 1 and not self.solid[i, j + 1]:
                    faces.append(("y", i, j + 1, -1))
        self.sensors.append(Sensor(name, "wall_force", faces=faces))

    def stir(self, x, y, radius, u, v):
        """Impose a velocity (u, v) on the water within ``radius`` of (x, y)."""
        X, Y = self.centers()
        m = ((X - x) ** 2 + (Y - y) ** 2 <= radius ** 2) & ~self.solid & (self.h > DRY)
        self.hu[m], self.hv[m] = self.h[m] * u, self.h[m] * v

    def volume(self) -> float:
        return float(self.h.sum() * self.dx * self.dy)

    # ── numerics ────────────────────────────────────────────────────────
    @staticmethod
    def _minmod(a, b):
        return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def _flux_x(self, hL, uL, vL, hR, uR, vR):
        g = self.g
        cL, cR = np.sqrt(g * np.maximum(hL, 0)), np.sqrt(g * np.maximum(hR, 0))
        sL = np.minimum(uL - cL, uR - cR)
        sR = np.maximum(uL + cL, uR + cR)
        dryL, dryR = hL <= DRY, hR <= DRY
        sL = np.where(dryL, uR - 2 * cR, sL)
        sR = np.where(dryR, uL + 2 * cL, sR)
        FL = [hL * uL, hL * uL * uL + 0.5 * g * hL * hL, hL * uL * vL]
        FR = [hR * uR, hR * uR * uR + 0.5 * g * hR * hR, hR * uR * vR]
        UL, UR = [hL, hL * uL, hL * vL], [hR, hR * uR, hR * vR]
        den = np.where(np.abs(sR - sL) > 1e-14, sR - sL, 1.0)
        out = []
        for fl, fr, ul, ur in zip(FL, FR, UL, UR):
            fhll = (sR * fl - sL * fr + sL * sR * (ur - ul)) / den
            out.append(np.where(sL >= 0, fl, np.where(sR <= 0, fr, fhll)))
        both_dry = dryL & dryR
        return [np.where(both_dry, 0.0, f) for f in out]

    def _faces(self, h, u, v, axis):
        """Left/right states at every face along ``axis`` (0 = x, 1 = y), walls mirrored."""
        un, ut = (u, v) if axis == 0 else (v, u)
        solid = self.solid
        if self.second_order:
            def slope(q):
                d = np.diff(q, axis=axis)
                pad = [(0, 0), (0, 0)]
                pad[axis] = (1, 1)
                dp = np.pad(d, pad)
                s = self._minmod(np.take(dp, range(0, dp.shape[axis] - 1), axis=axis),
                                 np.take(dp, range(1, dp.shape[axis]), axis=axis))
                # no slope next to walls / dry cells (first order there)
                return np.where(solid | (h <= DRY), 0.0, s)
            sh, su, sv = slope(h), slope(un), slope(ut)
        else:
            sh = su = sv = np.zeros_like(h)
        # values at the minus/plus side of each cell
        hm, hp = h - 0.5 * sh, h + 0.5 * sh
        um, up = un - 0.5 * su, un + 0.5 * su
        vm, vp = ut - 0.5 * sv, ut + 0.5 * sv
        n = h.shape[axis]
        take = lambda a, idx: np.take(a, idx, axis=axis)
        # interior faces 1..n-1 between cell k-1 (left) and k (right)
        L = [take(hp, range(0, n - 1)), take(up, range(0, n - 1)), take(vp, range(0, n - 1))]
        R = [take(hm, range(1, n)), take(um, range(1, n)), take(vm, range(1, n))]
        sL, sR = take(solid, range(0, n - 1)), take(solid, range(1, n))
        # mirror across walls
        Lm = [np.where(sL, R[0], L[0]), np.where(sL, -R[1], L[1]), np.where(sL, R[2], L[2])]
        Rm = [np.where(sR, L[0], R[0]), np.where(sR, -L[1], R[1]), np.where(sR, L[2], R[2])]
        both = sL & sR
        # domain boundary faces: reflective walls
        first = [take(hm, [0]), take(um, [0]), take(vm, [0])]
        last = [take(hp, [n - 1]), take(up, [n - 1]), take(vp, [n - 1])]
        lb = ([first[0], -first[1], first[2]], first)
        rb = (last, [last[0], -last[1], last[2]])
        cat = lambda a, b, c: np.concatenate([a, b, c], axis=axis)
        HL = cat(lb[0][0], Lm[0], rb[0][0]); UL = cat(lb[0][1], Lm[1], rb[0][1]); VL = cat(lb[0][2], Lm[2], rb[0][2])
        HR = cat(lb[1][0], Rm[0], rb[1][0]); UR = cat(lb[1][1], Rm[1], rb[1][1]); VR = cat(lb[1][2], Rm[2], rb[1][2])
        both = cat(take(solid, [0]), both, take(solid, [n - 1]))
        return (np.maximum(HL, 0), UL, VL, np.maximum(HR, 0), UR, VR), both

    def _rhs(self, h, hu, hv):
        u = np.where(h > DRY, hu / np.maximum(h, DRY), 0.0)
        v = np.where(h > DRY, hv / np.maximum(h, DRY), 0.0)
        (hL, uL, vL, hR, uR, vR), zx = self._faces(h, u, v, 0)
        Fh, Fn, Ft = self._flux_x(hL, uL, vL, hR, uR, vR)
        Fh, Fn, Ft = (np.where(zx, 0.0, f) for f in (Fh, Fn, Ft))
        (hL, vL_, uL_, hR, vR_, uR_), zy = self._faces(h, u, v, 1)
        Gh, Gn, Gt = self._flux_x(hL, vL_, uL_, hR, vR_, uR_)
        Gh, Gn, Gt = (np.where(zy, 0.0, f) for f in (Gh, Gn, Gt))
        dh = -(np.diff(Fh, axis=0) / self.dx + np.diff(Gh, axis=1) / self.dy)
        dhu = -(np.diff(Fn, axis=0) / self.dx + np.diff(Gt, axis=1) / self.dy)
        dhv = -(np.diff(Ft, axis=0) / self.dx + np.diff(Gn, axis=1) / self.dy)
        for d in (dh, dhu, dhv):
            d[self.solid] = 0.0
        return (dh, dhu, dhv), (Fn, Gn)

    def _dt(self):
        h = self.h
        c = np.sqrt(self.g * np.maximum(h, 0))
        u = np.where(h > DRY, np.abs(self.hu) / np.maximum(h, DRY), 0.0)
        v = np.where(h > DRY, np.abs(self.hv) / np.maximum(h, DRY), 0.0)
        smax = max(float(np.max(u + c)) / self.dx, float(np.max(v + c)) / self.dy, 1e-12)
        return self.cfl / smax

    def step(self, dt: Optional[float] = None) -> float:
        for name, (mask, t_rel) in list(self.gates.items()):
            if self.t >= t_rel:
                self.solid &= ~mask
                del self.gates[name]
        if self._v0 is None:
            self._v0 = self.volume()
        dt = dt or self._dt()
        U0 = (self.h.copy(), self.hu.copy(), self.hv.copy())
        (k1, (Fx1, Fy1)) = self._rhs(*U0)
        U1 = tuple(a + dt * k for a, k in zip(U0, k1))
        U1 = (np.maximum(U1[0], 0.0),) + U1[1:]
        (k2, (Fx2, Fy2)) = self._rhs(*U1)
        U2 = [0.5 * a + 0.5 * (b + dt * k) for a, b, k in zip(U0, U1, k2)]
        neg = U2[0] < 0
        self.min_depth_before_clip = float(U2[0].min())
        U2[0] = np.maximum(U2[0], 0.0)
        dry = U2[0] <= DRY
        U2[1][dry] = 0.0
        U2[2][dry] = 0.0
        self.h, self.hu, self.hv = U2
        self._wall_flux = {"x": 0.5 * (Fx1 + Fx2), "y": 0.5 * (Fy1 + Fy2)}
        self.t += dt
        self._record(dt, int(neg.sum()))
        return dt

    def run(self, t_end: float, max_steps: int = 10 ** 7) -> None:
        n = 0
        while self.t < t_end - 1e-12 and n < max_steps:
            self.step(min(self._dt(), t_end - self.t))
            n += 1

    # ── sensors, health, data ───────────────────────────────────────────
    def _sample(self, a, x, y):
        fi = np.clip(x / self.dx - 0.5, 0, self.nx - 1)
        fj = np.clip(y / self.dy - 0.5, 0, self.ny - 1)
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))
        i1, j1 = min(i0 + 1, self.nx - 1), min(j0 + 1, self.ny - 1)
        wx, wy = fi - i0, fj - j0
        return float((1 - wx) * (1 - wy) * a[i0, j0] + wx * (1 - wy) * a[i1, j0]
                     + (1 - wx) * wy * a[i0, j1] + wx * wy * a[i1, j1])

    def _record(self, dt, n_negative):
        self.times.append(self.t)
        for s in self.sensors:
            if s.kind == "depth":
                s.values.append(self._sample(self.h, s.x, s.y))
            elif s.kind == "pressure":
                d = self._sample(self.h, s.x, s.y)
                s.values.append(self.rho * self.g * max(d - s.z, 0.0))
            elif s.kind == "velocity":
                h = self._sample(self.h, s.x, s.y)
                s.values.append(math.hypot(self._sample(self.hu, s.x, s.y), self._sample(self.hv, s.x, s.y)) / h
                                if h > 1e-6 else 0.0)
            elif s.kind == "wall_force":
                f = 0.0
                for axis, i, j, sign in s.faces:
                    if axis == "x":
                        f += sign * self._wall_flux["x"][i, j] * self.dy
                    else:
                        f += sign * self._wall_flux["y"][i, j] * self.dx
                s.values.append(self.rho * f)
        v = self.volume()
        c = np.sqrt(self.g * np.maximum(self.h, 0))
        speed = np.where(self.h > DRY, np.hypot(self.hu, self.hv) / np.maximum(self.h, DRY), 0.0)
        self.health.append({
            "t": self.t, "dt": dt,
            "volume_rel_drift": (v - self._v0) / self._v0 if self._v0 else 0.0,
            "cfl": float(dt * np.max(speed + c) / min(self.dx, self.dy)),
            "min_depth_before_clip": getattr(self, "min_depth_before_clip", 0.0),
            "negative_depth_cells": n_negative,
            "finite": bool(np.isfinite(self.h).all() and np.isfinite(self.hu).all() and np.isfinite(self.hv).all()),
        })

    def health_report(self, volume_tol: float = 1e-9) -> Dict[str, object]:
        """Summary of the live checks: stability (finite, CFL) and conservation (volume drift)."""
        if not self.health:
            return {"ok": True, "steps": 0}
        drift = max(abs(hh["volume_rel_drift"]) for hh in self.health)
        cfl = max(hh["cfl"] for hh in self.health)
        finite = all(hh["finite"] for hh in self.health)
        problems = []
        if not finite:
            problems.append("non-finite state (instability)")
        if drift > volume_tol:
            problems.append(f"volume drift {drift:.2e} > {volume_tol:.0e}")
        if cfl > 1.0:
            problems.append(f"CFL {cfl:.2f} > 1")
        return {"ok": not problems, "steps": len(self.health), "max_volume_rel_drift": drift, "max_cfl": cfl,
                "problems": problems}

    def export_csv(self, path: str) -> str:
        names = [s.name for s in self.sensors]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + names + ["volume_rel_drift", "cfl"])
            for k, t in enumerate(self.times):
                w.writerow([f"{t:.6g}"] + [f"{s.values[k]:.8g}" for s in self.sensors]
                           + [f"{self.health[k]['volume_rel_drift']:.3e}", f"{self.health[k]['cfl']:.4f}"])
        return path

    def compare(self, sensor: str, t_exp: Sequence[float], y_exp: Sequence[float]) -> Dict[str, float]:
        """RMSE / normalised RMSE / bias of a sensor against experimental samples (linear interpolation)."""
        s = next(x for x in self.sensors if x.name == sensor)
        y = np.interp(np.asarray(t_exp, float), np.asarray(self.times), np.asarray(s.values))
        e = y - np.asarray(y_exp, float)
        rmse = float(np.sqrt(np.mean(e ** 2)))
        span = float(np.ptp(y_exp)) or 1.0
        return {"rmse": rmse, "nrmse": rmse / span, "bias": float(e.mean()), "n": len(e)}


# ── exact dam-break solutions (1D, frictionless, horizontal bed) ─────────
def ritter(x, t, h0, x0=0.0, g=G):
    """Dam break on a dry bed (Ritter 1892): depth and velocity."""
    x = np.asarray(x, float)
    c0 = math.sqrt(g * h0)
    xi = (x - x0) / t
    h = np.where(xi <= -c0, h0, np.where(xi >= 2 * c0, 0.0, (2 * c0 - xi) ** 2 / (9 * g)))
    u = np.where((xi > -c0) & (xi < 2 * c0), 2.0 / 3.0 * (xi + c0), 0.0)
    return h, u


def stoker(x, t, h0, h1, x0=0.0, g=G):
    """Dam break on a wet bed h1 < h0 (Stoker 1957): rarefaction + constant state + bore."""
    from scipy.optimize import brentq

    x = np.asarray(x, float)
    c0 = math.sqrt(g * h0)

    def mismatch(hm):
        S = math.sqrt(g * hm * (hm + h1) / (2 * h1))  # bore speed into still water h1
        return 2 * (c0 - math.sqrt(g * hm)) - S * (1 - h1 / hm)

    hm = brentq(mismatch, h1 * (1 + 1e-12), h0)
    cm = math.sqrt(g * hm)
    um = 2 * (c0 - cm)
    S = math.sqrt(g * hm * (hm + h1) / (2 * h1))
    xi = (x - x0) / t
    h = np.where(xi <= -c0, h0,
                 np.where(xi <= um - cm, (2 * c0 - xi) ** 2 / (9 * g),
                          np.where(xi <= S, hm, h1)))
    u = np.where(xi <= -c0, 0.0, np.where(xi <= um - cm, 2.0 / 3.0 * (xi + c0), np.where(xi <= S, um, 0.0)))
    return h, u
