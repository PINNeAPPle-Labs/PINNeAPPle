"""Lagrangian particles in a liquid: soft-sphere DEM contacts, drag, buoyancy and turbulent dispersion, in a
prescribed carrier flow (one-way coupling). Fast enough for tens of thousands of particles on a laptop, and meant
for process pictures and trends (suspension, mixing, settling), not for calibrated design numbers.

    from pinneapple_simulation.numerical_solvers.particles import StirredTank, suspend
    tank = StirredTank(R=0.075, H=0.15)
    res = suspend(tank, n=20000, d=3e-3, rho_p=1200.0, rpm=lambda t: min(150.0, 6.0 * t), t_end=30.0)
    res["frames"][k]["x"], res["frames"][k]["speed"], res["top_fraction"], res["rpm"]

The stirred-tank flow is an analytic, exactly divergence-free model: a solid-body / free-vortex swirl from the
impeller and one meridional circulation loop (down along the shaft for a down-pumping pitched-blade turbine, up
along the wall), both scaled by the impeller tip speed, with velocity fluctuations as an Ornstein-Uhlenbeck process.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Liquid:
    rho: float = 998.0
    mu: float = 1.0e-3


WATER = Liquid()


def drag_factor(Re: np.ndarray) -> np.ndarray:
    """Schiller-Naumann correction to Stokes drag (Re < 1000)."""
    return 1.0 + 0.15 * np.power(np.maximum(Re, 0.0), 0.687)


def terminal_velocity(d: float, rho_p: float, liq: Liquid = WATER, g: float = 9.81) -> float:
    """Settling velocity of an isolated sphere with Schiller-Naumann drag (fixed point)."""
    v = (rho_p - liq.rho) * g * d * d / (18 * liq.mu)
    for _ in range(200):
        Re = liq.rho * v * d / liq.mu
        v = (rho_p - liq.rho) * g * d * d / (18 * liq.mu * drag_factor(np.array(Re)))
    return float(v)


@dataclass
class StirredTank:
    """Flat-bottom cylindrical tank (axis z, bottom at z = 0, liquid up to H) with a centred impeller of diameter
    ``D_imp`` at clearance ``C``. ``swirl`` and ``loop``: swirl and circulation speeds as fractions of the tip speed;
    ``turbulence``: velocity fluctuation as a fraction of the tip speed; ``T_L``: their Lagrangian time scale in
    impeller revolutions."""
    R: float = 0.075
    H: float = 0.15
    D_imp: float | None = None
    C: float | None = None
    swirl: float = 0.35
    loop: float = 0.25
    turbulence: float = 0.12
    T_L: float = 0.3
    blades: int = 4
    pitch_deg: float = 45.0
    down_pumping: bool = True

    def __post_init__(self):
        self.D_imp = self.D_imp or 2 * self.R * 2 / 3
        self.C = self.C or self.H / 3

    def tip_speed(self, rpm: float) -> float:
        return math.pi * self.D_imp * rpm / 60.0

    def velocity(self, x: np.ndarray, rpm: float) -> np.ndarray:
        """Mean liquid velocity at points (n, 3)."""
        V = self.tip_speed(rpm)
        if V == 0:
            return np.zeros_like(x)
        r = np.hypot(x[:, 0], x[:, 1]) + 1e-12
        z = np.clip(x[:, 2], 0, self.H)
        ri = self.D_imp / 2
        # swirl: solid body inside the impeller radius, free vortex outside, decaying to zero at the wall and bottom
        ut = self.swirl * V * np.where(r < ri, r / ri, ri / r) * (1 - (r / self.R) ** 4) * np.sin(np.pi * np.minimum(
            z / (2 * self.C), 0.5) + 0.0) ** 0.5
        # meridional loop from the Stokes stream function psi = A r^2 (1 - r^2/R^2) sin(pi z / H): exactly
        # divergence-free, no flow through the wall, the bottom or the surface
        A = (-1 if self.down_pumping else 1) * self.loop * V / 2
        s, c = np.sin(np.pi * z / self.H), np.cos(np.pi * z / self.H)
        uz = A * (2 - 4 * r * r / self.R ** 2) * s
        ur = -A * r * (1 - r * r / self.R ** 2) * (np.pi / self.H) * c
        er = x[:, :2] / r[:, None]
        u = np.empty_like(x)
        u[:, 0] = ur * er[:, 0] - ut * er[:, 1]
        u[:, 1] = ur * er[:, 1] + ut * er[:, 0]
        u[:, 2] = uz
        return u

    def surfaces(self, angle: float = 0.0, n: int = 96):
        """Tank wall (open top), bottom, shaft and impeller blades turned by ``angle`` (rad), as (name, V, F,
        material) for the studio renderer."""
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        R, H = self.R, self.H
        Hw = 1.08 * H
        wall_V = np.concatenate([np.c_[R * np.cos(th), R * np.sin(th), np.zeros(n)],
                                 np.c_[R * np.cos(th), R * np.sin(th), np.full(n, Hw)]])
        wall_F = np.array([(i, (i + 1) % n, n + (i + 1) % n) for i in range(n)] +
                          [(i, n + (i + 1) % n, n + i) for i in range(n)])
        bot_V = np.r_[[[0, 0, 0]], np.c_[R * np.cos(th), R * np.sin(th), np.zeros(n)]]
        bot_F = np.array([(0, 1 + (i + 1) % n, 1 + i) for i in range(n)])
        surf_V = np.r_[[[0, 0, H]], np.c_[0.995 * R * np.cos(th), 0.995 * R * np.sin(th), np.full(n, H)]]
        surf_F = np.array([(0, 1 + i, 1 + (i + 1) % n) for i in range(n)])
        rs, m = 0.035 * self.D_imp, 16
        ts = np.linspace(0, 2 * np.pi, m, endpoint=False)
        sh_V = np.concatenate([np.c_[rs * np.cos(ts), rs * np.sin(ts), np.full(m, self.C)],
                               np.c_[rs * np.cos(ts), rs * np.sin(ts), np.full(m, 1.25 * H)]])
        sh_F = np.array([(i, (i + 1) % m, m + (i + 1) % m) for i in range(m)] + [(i, m + (i + 1) % m, m + i) for i in range(m)])
        bl_V, bl_F = [], []
        w, hb = 0.5 * self.D_imp, 0.2 * self.D_imp                   # blade span and height
        pa = math.radians(self.pitch_deg)
        for b in range(self.blades):
            a = angle + 2 * np.pi * b / self.blades
            e1 = np.array([math.cos(a), math.sin(a), 0.0])           # along the blade
            e2 = np.array([-math.sin(a) * math.cos(pa), math.cos(a) * math.cos(pa), math.sin(pa)])   # pitched chord
            o = np.array([0, 0, self.C])
            q = [o + rs * e1 - 0.5 * hb * e2, o + (rs + w) * e1 - 0.5 * hb * e2, o + (rs + w) * e1 + 0.5 * hb * e2,
                 o + rs * e1 + 0.5 * hb * e2]
            k = len(bl_V)
            bl_V += q
            bl_F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
        return [("tank_wall", wall_V, wall_F, "glass"), ("tank_bottom", bot_V, bot_F, "glass"),
                ("liquid_surface", surf_V, surf_F, "liquid"), ("shaft", sh_V, sh_F, "steel"),
                ("impeller", np.array(bl_V), np.array(bl_F), "steel")]


class DEM:
    """Equal spheres (diameter ``d``) with linear spring-dashpot normal contacts (restitution ``e``, collision time
    ``t_c``), drag towards the liquid velocity, buoyancy-reduced gravity, and contacts with the tank (wall, bottom,
    surface). Verlet neighbour lists with a skin."""

    def __init__(self, x: np.ndarray, d: float, rho_p: float, liq: Liquid = WATER, e: float = 0.4,
                 t_c: float | None = None, dt: float = 5e-4, g: float = 9.81, seed: int = 0):
        self.x = np.asarray(x, float).copy()
        self.v = np.zeros_like(self.x)
        self.d, self.rho_p, self.liq, self.g, self.dt = d, rho_p, liq, g, dt
        self.m = rho_p * math.pi * d ** 3 / 6
        self.t_c = t_c or 10 * dt
        m_eff = self.m / 2
        ln_e = math.log(e)
        self.k = m_eff * (math.pi ** 2 + ln_e ** 2) / self.t_c ** 2
        self.c = -2 * m_eff * ln_e / self.t_c
        self.kw, self.cw = 2 * self.k, 2 * self.c                   # sphere-wall: the wall has infinite mass
        self.tau_p = rho_p * d * d / (18 * liq.mu)
        self.g_eff = g * (1 - liq.rho / rho_p)
        self.u_turb = np.zeros_like(self.x)
        self.rng = np.random.default_rng(seed)
        self._pairs = None
        self._x_ref = None
        self.skin = 0.3 * d

    def _neighbours(self):
        from scipy.spatial import cKDTree
        if self._x_ref is None or np.max(np.abs(self.x - self._x_ref)) > 0.5 * self.skin:
            self._pairs = cKDTree(self.x).query_pairs(self.d + self.skin, output_type="ndarray")
            self._x_ref = self.x.copy()
        return self._pairs

    def contact_forces(self) -> tuple[np.ndarray, float]:
        P = self._neighbours()
        F = np.zeros_like(self.x)
        if len(P) == 0:
            return F, 0.0
        i, j = P[:, 0], P[:, 1]
        dx = self.x[i] - self.x[j]
        dist = np.linalg.norm(dx, axis=1)
        ov = self.d - dist
        on = ov > 0
        if not on.any():
            return F, 0.0
        i, j, dx, dist, ov = i[on], j[on], dx[on], dist[on], ov[on]
        n = dx / np.maximum(dist, 1e-12)[:, None]
        vn = ((self.v[i] - self.v[j]) * n).sum(1)
        f = (self.k * ov - self.c * vn)[:, None] * n
        np.add.at(F, i, f)
        np.add.at(F, j, -f)
        return F, float(ov.max() / self.d)

    def wall_forces(self, tank: StirredTank) -> np.ndarray:
        F = np.zeros_like(self.x)
        a = self.d / 2
        r = np.hypot(self.x[:, 0], self.x[:, 1]) + 1e-12
        er = np.c_[self.x[:, 0] / r, self.x[:, 1] / r, np.zeros(len(r))]
        ov = r + a - tank.R
        m = ov > 0
        vn = (self.v[m] * er[m]).sum(1)
        F[m] -= (self.kw * ov[m] + self.cw * vn)[:, None] * er[m]
        ov = a - self.x[:, 2]                                        # bottom
        m = ov > 0
        F[m, 2] += self.kw * ov[m] - self.cw * self.v[m, 2]
        ov = self.x[:, 2] + a - tank.H                               # free surface (as a slip lid)
        m = ov > 0
        F[m, 2] -= self.kw * ov[m] + self.cw * self.v[m, 2]
        return F

    def step(self, u_mean: np.ndarray, sigma: float, T_L: float, tank: StirredTank | None = None) -> float:
        dt = self.dt
        if sigma > 0:                                               # Ornstein-Uhlenbeck velocity fluctuation
            a = math.exp(-dt / T_L)
            self.u_turb = a * self.u_turb + sigma * math.sqrt(1 - a * a) * self.rng.standard_normal(self.x.shape)
        else:
            self.u_turb[:] = 0
        u = u_mean + self.u_turb
        rel = u - self.v
        Re = self.liq.rho * np.linalg.norm(rel, axis=1) * self.d / self.liq.mu
        Fc, ovmax = self.contact_forces()
        acc = rel * (drag_factor(Re) / self.tau_p)[:, None] + Fc / self.m
        acc[:, 2] -= self.g_eff
        if tank is not None:
            acc += self.wall_forces(tank) / self.m
        self.v += dt * acc
        self.x += dt * self.v
        return ovmax


def packed_bed(tank: StirredTank, n: int, d: float, seed: int = 0) -> np.ndarray:
    """Initial positions: a loose cubic lattice filling the tank from the bottom (it settles in the first second)."""
    rng = np.random.default_rng(seed)
    h = 1.05 * d
    g = np.arange(-tank.R + d, tank.R - d + 1e-12, h)
    X, Y = np.meshgrid(g, g, indexing="ij")
    ring = np.hypot(X, Y) < tank.R - 0.75 * d
    layer = np.c_[X[ring], Y[ring]]
    out = []
    z = d / 2 + 0.05 * d
    while sum(len(o) for o in out) < n:
        out.append(np.c_[layer, np.full(len(layer), z)])
        z += h
    x = np.concatenate(out)[:n]
    return x + rng.uniform(-0.02, 0.02, x.shape) * d


def suspend(tank: StirredTank, n: int = 20000, d: float = 3e-3, rho_p: float = 1200.0,
            rpm: Callable[[float], float] = lambda t: min(150.0, 6.0 * t), t_end: float = 30.0, dt: float = 5e-4,
            frame_every: float = 0.25, liq: Liquid = WATER, seed: int = 0, log=None) -> dict:
    """Suspension of an initially settled bed as the impeller speeds up. Returns frames (positions, speed), times,
    rpm, the fraction of particles in the top third, the fraction off the bottom and the largest contact overlap."""
    x0 = packed_bed(tank, n, d, seed)
    dem = DEM(x0, d, rho_p, liq, dt=dt, seed=seed)
    frames, times, rpms, top, off, ovs = [], [], [], [], [], []
    bed0 = None
    steps = int(round(t_end / dt))
    every = max(1, int(round(frame_every / dt)))
    ov_max = 0.0
    for s in range(steps + 1):
        t = s * dt
        N = float(rpm(t))
        if s % every == 0:
            if bed0 is None and t >= 0.5 * frame_every:
                bed0 = float(np.percentile(dem.x[:, 2], 95))
            frames.append({"x": dem.x.astype(np.float32), "speed": np.linalg.norm(dem.v, axis=1).astype(np.float32)})
            times.append(t)
            rpms.append(N)
            top.append(float(np.mean(dem.x[:, 2] > 2 * tank.H / 3)))
            off.append(float(np.mean(dem.x[:, 2] > (bed0 or tank.H) + d)))
            ovs.append(ov_max)
            ov_max = 0.0
            if log and len(frames) % 20 == 0:
                log(f"t = {t:.2f} s, {N:.0f} rpm, top third {100 * top[-1]:.1f} %")
        if s == steps:
            break
        V = tank.tip_speed(N)
        ov_max = max(ov_max, dem.step(tank.velocity(dem.x, N), tank.turbulence * V,
                                      tank.T_L * 60.0 / max(N, 1e-6) if N > 0 else 1.0, tank))
    return {"frames": frames, "times": np.array(times), "rpm": np.array(rpms), "top_fraction": np.array(top),
            "suspended_fraction": np.array(off), "max_overlap": np.array(ovs), "bed_height": bed0, "d": d,
            "n": n, "rho_p": rho_p, "tank": tank}


def zwietering_njs(tank: StirredTank, d: float, rho_p: float, mass_ratio: float, liq: Liquid = WATER,
                   S: float = 6.0, g: float = 9.81) -> float:
    """Just-suspended impeller speed (rpm) of Zwietering (1958): N_js = S nu^0.1 (g drho/rho_L)^0.45 X^0.13
    d^0.2 / D^0.85, X in mass percent of solids to liquid. ``S`` depends on the impeller and the geometry
    (about 4-8 for pitched-blade turbines)."""
    nu = liq.mu / liq.rho
    X = 100 * mass_ratio
    n = S * nu ** 0.1 * (g * (rho_p - liq.rho) / liq.rho) ** 0.45 * X ** 0.13 * d ** 0.2 / tank.D_imp ** 0.85
    return 60.0 * n
