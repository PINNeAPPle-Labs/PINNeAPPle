"""Wong-Reece rigid-wheel / deformable-soil model with slip-sinkage, grousers, multi-pass, bulldozing and lateral force.

Every quantity is vectorised over wheels / operating points (numpy, piecewise Gauss-Legendre quadrature on the contact
arc). Angles are measured from the vertical through the axle, positive towards the front; the soil surface ahead of
the wheel is the reference for the sinkage.

Stresses on the contact arc (Wong & Reece 1967; Wong 2008)
----------------------------------------------------------
Entry angle ``theta_f = acos(1 - z / R)``, exit angle ``theta_r = -c3 theta_f`` and angle of the maximum normal stress
``theta_m = (a0 + a1 s) theta_f``. The depth of the rim below the soil surface is ``h(theta) = R (cos theta* -
cos theta_f)`` with ``theta* = theta`` ahead of ``theta_m`` and, behind it, the Wong-Reece mapping of the rear arc onto
the front arc ``theta* = theta_f - (theta - theta_r) / (theta_m - theta_r) (theta_f - theta_m)``.

* Normal stress, Bekker pressure-sinkage: ``sigma = (k_c / b + k_phi) h^n``.
* Slip-sinkage: the exponent grows with slip, ``n = n0 + n1 |s|`` (Ding et al. 2010, 2011; calibrated with held-out points in
  :func:`.testbed.calibrate_holdout`).
* Shear, Janosi-Hanamoto with Mohr-Coulomb strength: ``tau_x = (c + sigma tan phi) (1 - exp(-j_x / K))`` with the
  shear displacement ``j_x = R_s [(theta_f - theta) - (1 - s)(sin theta_f - sin theta)]``.
* Grousers: shear displacement on the equivalent radius ``R_s = R + lambda h_g`` (Ding et al. 2011).
* Lateral shear on a wheel with slip angle beta (Ishigami et al. 2007): ``j_y = R (1 - s)(theta_f - theta) tan beta``,
  ``tau_y = (c + sigma tan phi)(1 - exp(-j_y / K_y))``.

Forces (wheel frame: x forward along the soil surface, z along its normal, y along the axle)::

    F_x = R b int (tau_x cos - sigma sin)            drawbar pull
    F_z = R b int (sigma cos + tau_x sin)            load carried
    M_y = R R_s b int tau_x                          soil torque the drive must supply
    F_y = -sign(beta) [R b int |tau_y| + R_b]        lateral force, opposing the side slip

Bulldozing (Bekker 1969; Wong 2008): passive earth pressure on the soil wedge pushed by the wheel,
``R_b = b (0.67 c z K_pc + 0.5 z^2 gamma K_pgamma)`` with ``K_pc = (N_c - tan phi) cos^2 phi``,
``K_pgamma = (2 N_gamma / tan phi + 1) cos^2 phi`` and Terzaghi's bearing capacity factors. On the side face it uses
the side area (contact length instead of width); ahead of the wheel it is optional (``bulldozing="both"``) because the
``sigma sin`` term of Wong-Reece already holds the compaction resistance.

Multi-pass (repetitive loading, Wong 2008)
------------------------------------------
``precompaction`` is the deepest sinkage the soil ahead of the wheel has already carried (``z_u``, e.g. the rut of a
wheel in front). The soil rebounds elastically by ``p_u / k_u`` with ``p_u = k z_u^n`` and the unloading-reloading
modulus ``k_u = k_u0 + A_u z_u``; reloading follows ``p = k_u h`` until ``p_u`` and the virgin curve beyond it. The
sinkage passed in is then measured from the original (undisturbed) surface. As in Wong's multi-pass treatment of
consecutive wheels in one track, the shear displacement the soil already carries (``prior_shear``, left by the wheel
in front at its exit angle, :attr:`ContactResult.j_exit`) is added to the shear displacement of the following wheel,
so a wheel in a rut mobilises the soil strength over a shorter arc.

References are listed in :mod:`.soils` and ``docs/core_concepts/lunar_rover_terramechanics.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .soils import Soil
from .wheel import Wheel

_GL: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl(n: int):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


def bearing_capacity_factors(phi_deg) -> tuple[np.ndarray, np.ndarray]:
    """Terzaghi's general-shear bearing capacity factors (N_c, N_gamma) in the closed forms used by Wong (2008):
    ``N_q = exp((3 pi / 2 - phi) tan phi) / (2 cos^2(pi / 4 + phi / 2))``, ``N_c = (N_q - 1) / tan phi`` and the
    common approximation ``N_gamma = 2 (N_q + 1) tan phi`` (Vesic)."""
    phi = np.radians(np.asarray(phi_deg, float))
    t = np.tan(phi)
    nq = np.exp((1.5 * np.pi - phi) * t) / (2.0 * np.cos(np.pi / 4 + phi / 2) ** 2)
    return (nq - 1.0) / t, 2.0 * (nq + 1.0) * t


def bulldozing_resistance(soil: Soil, depth, width, g: float):
    """Bulldozing resistance [N] of a soil wedge of ``depth`` pushed by a face of ``width`` (Bekker 1969, Wong 2008)."""
    depth = np.clip(np.asarray(depth, float), 0.0, None)
    nc, ng = bearing_capacity_factors(soil.phi_deg)
    cos2 = math.cos(math.radians(soil.phi_deg)) ** 2
    kpc = (nc - soil.tan_phi) * cos2
    kpg = (2.0 * ng / soil.tan_phi + 1.0) * cos2
    gamma = soil.rho * g
    return np.asarray(width, float) * (0.67 * soil.c * depth * kpc + 0.5 * depth ** 2 * gamma * kpg)


@dataclass
class ContactResult:
    """Forces per operating point (all arrays of shape (N,)), plus the stresses on the arc when requested."""
    Fx: np.ndarray            # drawbar pull [N]
    Fy: np.ndarray            # lateral force [N]
    Fz: np.ndarray            # normal load carried [N]
    My: np.ndarray            # soil (resisting) torque about the axle [N m]
    theta_f: np.ndarray       # entry angle [rad]
    theta_r: np.ndarray       # exit angle [rad]
    z_eff: np.ndarray         # sinkage below the current soil surface [m]
    bulldozing: np.ndarray    # longitudinal + lateral bulldozing resistance actually applied [N]
    j_exit: np.ndarray | None = None     # shear displacement the soil keeps behind the wheel [m]
    theta: np.ndarray | None = None      # (N, M) quadrature nodes [rad]
    sigma: np.ndarray | None = None      # (N, M) normal stress [Pa]
    tau_x: np.ndarray | None = None      # (N, M) longitudinal shear stress [Pa]
    tau_y: np.ndarray | None = None      # (N, M) lateral shear stress [Pa]

    def as_array(self) -> np.ndarray:
        return np.stack([np.atleast_1d(self.Fx), np.atleast_1d(self.Fz), np.atleast_1d(self.My)], -1)


class WongReeceModel:
    """Wong-Reece wheel on one soil.

    >>> from pinneapple_simulation.terramechanics import WongReeceModel, Wheel, get_soil
    >>> m = WongReeceModel(get_soil("GRC-1"), Wheel(radius=0.25, width=0.2))
    >>> r = m.forces(slip=[0.1, 0.3], sinkage=[0.01, 0.02])
    >>> r.Fx, r.Fz, r.My

    ``bulldozing``: "lateral" (default, side face only), "both", or "none". ``n_gl``: Gauss-Legendre nodes on each of
    the three smooth pieces of the arc. ``slip_sinkage``: which slips raise the sinkage exponent ``n = n0 + n1 s``:
    "drive" (default, driving slip s > 0 only -- the range Ding et al. identified it on), "both" (``n1 |s|``, skid
    too) or "none".
    """

    def __init__(self, soil: Soil, wheel: Wheel, g: float = 1.62, bulldozing: str = "lateral", n_gl: int = 24,
                 slip_sinkage: str = "drive"):
        if bulldozing not in ("lateral", "both", "none"):
            raise ValueError("bulldozing must be 'lateral', 'both' or 'none'")
        if slip_sinkage not in ("drive", "both", "none"):
            raise ValueError("slip_sinkage must be 'drive', 'both' or 'none'")
        self.soil, self.wheel, self.g = soil, wheel, float(g)
        self.bulldozing = bulldozing
        self.slip_sinkage = slip_sinkage
        self.n_gl = int(n_gl)

    def physics_config(self) -> dict:
        """Everything that sets this model's forces, as plain data: the soil (name and every parameter), the wheel,
        gravity and the model options. A surrogate trained on the model records it (e.g. in a zoo model card), so a
        later change of the library's physics can be detected with :func:`physics_mismatch`."""
        S, W = self.soil, self.wheel
        g = W.grousers
        soil = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in
                (("name", S.name), ("c", S.c), ("phi_deg", S.phi_deg), ("K", S.K), ("Ky", S.Ky), ("k_c", S.k_c),
                 ("k_phi", S.k_phi), ("n0", S.n0), ("n1", S.n1), ("rho", S.rho), ("a0", S.a0), ("a1", S.a1),
                 ("c3", S.c3), ("k_u0", S.k_u0), ("A_u", S.A_u))}
        wheel = {"radius": float(W.radius), "width": float(W.width), "grouser_count": int(g.count),
                 "grouser_height": float(g.height), "shear_radius_factor": float(g.shear_radius_factor)}
        return {"model": "Wong-Reece", "soil": soil, "wheel": wheel, "g": self.g, "bulldozing": self.bulldozing,
                "slip_sinkage": self.slip_sinkage}

    @classmethod
    def from_physics_config(cls, cfg: dict, n_gl: int = 24) -> WongReeceModel:
        """The model a :meth:`physics_config` describes (e.g. the one recorded in a zoo model card)."""
        from .soils import Soil
        from .wheel import Grousers
        sp = dict(cfg["soil"])
        soil = Soil(sp.pop("name"), **sp)
        w = cfg["wheel"]
        g = Grousers(count=int(w["grouser_count"]), height=float(w["grouser_height"]),
                     shear_radius_factor=float(w["shear_radius_factor"]))
        return cls(soil, Wheel(radius=w["radius"], width=w["width"], grousers=g), g=cfg["g"],
                   bulldozing=cfg["bulldozing"], slip_sinkage=cfg["slip_sinkage"], n_gl=n_gl)

    def sinkage_exponent(self, slip) -> np.ndarray:
        """Exponent of the pressure-sinkage law at ``slip``: ``n0 + n1 s`` (Ding et al. 2010, 2011), see
        ``slip_sinkage``."""
        s = np.asarray(slip, float)
        if self.slip_sinkage == "none":
            return np.full(s.shape, self.soil.n0)
        s_eff = np.abs(s) if self.slip_sinkage == "both" else np.clip(s, 0.0, None)
        return self.soil.n0 + self.soil.n1 * s_eff

    # ------------------------------------------------------------------ stresses
    def _k(self) -> float:
        return self.soil.k_c / self.wheel.width + self.soil.k_phi

    def normal_stress(self, h, n, zu):
        """Normal stress [Pa] at depth ``h`` below the current surface, exponent ``n``, prior sinkage ``zu``."""
        k = self._k()
        h = np.clip(h, 0.0, None)
        if not np.any(zu > 0):
            return k * np.where(h > 0, np.maximum(h, 1e-30) ** n, 0.0)
        pu = k * np.maximum(zu, 0.0) ** n
        ku = self.soil.k_u0 + self.soil.A_u * zu
        de = np.where(zu > 0, pu / np.maximum(ku, 1e-9), 0.0)          # elastic rebound of the rut
        virgin = k * np.maximum(zu - de + h, 1e-30) ** n
        return np.where(h <= de, ku * h, virgin)

    def rebound(self, zu) -> np.ndarray:
        """Elastic rebound [m] of soil unloaded from sinkage ``zu`` (depth of a rut = ``zu - rebound``)."""
        zu = np.asarray(zu, float)
        n = self.soil.n0
        pu = self._k() * np.maximum(zu, 0.0) ** n
        return np.where(zu > 0, pu / np.maximum(self.soil.k_u0 + self.soil.A_u * zu, 1e-9), 0.0)

    # ------------------------------------------------------------------ forces
    def forces(self, slip, sinkage, slip_angle=0.0, precompaction=0.0, prior_shear=0.0, *,
               stresses: bool = False, sinkage_slip=None) -> ContactResult:
        """Forces for arrays of slip [-], sinkage below the original surface [m], slip angle [rad], prior sinkage [m]
        and prior shear displacement [m] of the soil ahead (all broadcast to one shape). ``sinkage_slip`` (default:
        ``slip``) is the slip that sets the sinkage exponent, e.g. a slip gated by the wheel speed in a dynamic
        simulation, where the slip ratio of a nearly stopped wheel is not meaningful."""
        ss = slip if sinkage_slip is None else sinkage_slip
        s, z, beta, zu, j0, ss = (np.asarray(a, float) for a in np.broadcast_arrays(slip, sinkage, slip_angle,
                                                                                     precompaction, prior_shear, ss))
        shape = s.shape
        s, z, beta, zu, j0 = s.ravel(), z.ravel(), beta.ravel(), np.clip(zu.ravel(), 0.0, None), j0.ravel()
        S, W = self.soil, self.wheel
        R, b, Rs = W.radius, W.width, W.shear_radius
        s = np.clip(s, -0.99, 0.99)
        n = self.sinkage_exponent(np.clip(ss.ravel(), -0.99, 0.99))
        offset = np.clip(zu - self.rebound(zu), 0.0, None)            # current surface below the original one
        z_eff = np.clip(z - offset, 0.0, 0.95 * R)
        on = z_eff > 1e-7
        theta_f = np.arccos(np.clip(1.0 - z_eff / R, -1.0, 1.0))
        theta_m = (S.a0 + S.a1 * s) * theta_f
        theta_r = -S.c3 * theta_f
        x, w = _gl(self.n_gl)

        def piece(lo, hi):
            half = 0.5 * (hi - lo)[:, None]
            th = 0.5 * (hi + lo)[:, None] + half * x[None]
            return th, half * w[None]

        mid0 = np.clip(np.minimum(0.0, theta_m), theta_r, None)
        pieces = [piece(theta_r, mid0), piece(mid0, np.maximum(theta_m, mid0)), piece(np.maximum(theta_m, mid0),
                                                                                       theta_f)]
        th = np.concatenate([p[0] for p in pieces], 1)
        wt = np.concatenate([p[1] for p in pieces], 1)
        tf, tm, tr = theta_f[:, None], theta_m[:, None], theta_r[:, None]
        rear = th < tm
        ratio = (tf - tm) * (th - tr) / np.maximum(tm - tr, 1e-12)
        th_eq = np.where(rear, tf - ratio, th)
        h = R * (np.cos(th_eq) - np.cos(tf))
        sig = self.normal_stress(h, n[:, None], zu[:, None])
        strength = S.c + sig * S.tan_phi
        jx = j0[:, None] + Rs * ((tf - th) - (1.0 - s)[:, None] * (np.sin(tf) - np.sin(th)))
        tau_x = np.sign(jx) * strength * (1.0 - np.exp(-np.abs(jx) / S.K))
        jy = R * (1.0 - s)[:, None] * (tf - th) * np.abs(np.tan(beta))[:, None]
        tau_y = strength * (1.0 - np.exp(-jy / S.Ky))
        integ = lambda f: (f * wt).sum(1)  # noqa: E731
        Fx = R * b * integ(tau_x * np.cos(th) - sig * np.sin(th))
        Fz = R * b * integ(sig * np.cos(th) + tau_x * np.sin(th))
        My = R * Rs * b * integ(tau_x)
        Fy_shear = R * b * integ(tau_y)
        bull = np.zeros_like(Fx)
        if self.bulldozing != "none":
            # side face: wedge as deep as the sinkage over the contact length (only when sliding sideways)
            length = R * (np.sin(theta_f) - np.sin(theta_r))
            side = bulldozing_resistance(S, z_eff, length, self.g) * np.minimum(np.abs(np.tan(beta)) / 0.05, 1.0)
            Fy_shear = Fy_shear + side
            bull = bull + side
            if self.bulldozing == "both":
                front = bulldozing_resistance(S, z_eff, b, self.g)
                Fx = Fx - front
                bull = bull + front
        Fy = -np.sign(beta) * Fy_shear
        j_exit = j0 + Rs * ((theta_f - theta_r) - (1.0 - s) * (np.sin(theta_f) - np.sin(theta_r)))
        out = [np.where(on, a, 0.0) for a in (Fx, Fy, Fz, My, theta_f, theta_r, z_eff, bull)]
        res = ContactResult(*[a.reshape(shape) for a in out], j_exit=np.where(on, j_exit, j0).reshape(shape))
        if stresses:
            res.theta, res.sigma, res.tau_x, res.tau_y = th, sig * on[:, None], tau_x * on[:, None], \
                tau_y * on[:, None] * -np.sign(beta)[:, None]
        return res

    # ------------------------------------------------------------------ equilibrium
    def sinkage_for_load(self, load, slip, slip_angle=0.0, precompaction=0.0, prior_shear=0.0,
                         z_max: float | None = None, iters: int = 60, sinkage_slip=None) -> np.ndarray:
        """Sinkage [m] with ``F_z(slip, z) = load`` (vectorised bisection; F_z grows with z on the bracket).
        ``sinkage_slip`` as in :meth:`forces` (default: ``slip``)."""
        ss = slip if sinkage_slip is None else sinkage_slip
        W, s, beta, zu, j0, ss = (np.asarray(a, float) for a in np.broadcast_arrays(load, slip, slip_angle,
                                                                                     precompaction, prior_shear, ss))
        shape = W.shape
        W, s, beta, zu, j0, ss = W.ravel(), s.ravel(), beta.ravel(), zu.ravel(), j0.ravel(), ss.ravel()
        lo = np.clip(zu - self.rebound(zu), 0.0, None)
        hi = np.full_like(W, z_max if z_max is not None else 0.9 * self.wheel.radius) + lo
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = self.forces(s, mid, beta, zu, j0, sinkage_slip=ss).Fz
            up = f < W
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        return (0.5 * (lo + hi)).reshape(shape)

    def steady_state(self, load, slip, slip_angle=0.0, precompaction=0.0, prior_shear=0.0, sinkage_slip=None) -> dict:
        """Single-wheel testbed operating point: sinkage for the load, then drawbar pull, torque and lateral force.
        ``sinkage_slip``: the slip that sets the sinkage exponent if it differs from ``slip`` (e.g. a lagged slip)."""
        z = self.sinkage_for_load(load, slip, slip_angle, precompaction, prior_shear, sinkage_slip=sinkage_slip)
        r = self.forces(slip, z, slip_angle, precompaction, prior_shear, sinkage_slip=sinkage_slip)
        W = np.asarray(load, float)
        return {"slip": np.asarray(slip, float), "sinkage": z, "drawbar_pull": r.Fx, "torque": r.My,
                "lateral": r.Fy, "load": r.Fz, "dp_over_w": r.Fx / W, "j_exit": r.j_exit}


def slip_ratio(v_long, omega, radius, v_eps: float = 1e-3):
    """Longitudinal slip from the wheel-centre speed along the surface and the wheel speed (driving
    ``1 - v / (R omega)``, braking ``R omega / v - 1``), regularised near standstill by ``v_eps`` [m/s]."""
    v_long, omega = np.asarray(v_long, float), np.asarray(omega, float)
    vw = omega * radius
    den = np.maximum(np.maximum(np.abs(vw), np.abs(v_long)), v_eps)
    return np.clip((vw - v_long) / den, -0.99, 0.99)


def slip_angle(v_long, v_lat, v_eps: float = 1e-3):
    """Slip angle [rad] of a wheel from its centre velocity components along and across its heading."""
    return np.arctan2(np.asarray(v_lat, float), np.abs(np.asarray(v_long, float)) + v_eps)


def physics_mismatch(recorded: dict | None, current: dict, rtol: float = 1e-9) -> list[str]:
    """Keys (dotted paths) where a recorded :meth:`WongReeceModel.physics_config` differs from ``current``; a key the
    recorded configuration lacks counts as a difference ("not recorded"). Empty list: same physics."""
    if not recorded:
        return ["physics configuration not recorded"]
    out = []

    def walk(a, b, path):
        if isinstance(b, dict):
            for k, v in b.items():
                if not isinstance(a, dict) or k not in a:
                    out.append(f"{path}{k} (not recorded)")
                else:
                    walk(a[k], v, f"{path}{k}.")
            return
        key = path[:-1]
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12):
                out.append(f"{key}: {a:g} -> {b:g}")
        elif a != b:
            out.append(f"{key}: {a} -> {b}")
    walk(recorded, current, "")
    return out


__all__ = ["WongReeceModel", "ContactResult", "bulldozing_resistance", "bearing_capacity_factors", "slip_ratio",
           "slip_angle", "physics_mismatch"]
