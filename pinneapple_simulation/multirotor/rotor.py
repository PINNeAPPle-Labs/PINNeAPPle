"""Rotor aerodynamics of a multirotor: momentum theory and blade-element momentum theory (BEMT) in hover and axial
flight, with the Prandtl tip loss.

Conventions (Leishman, *Principles of Helicopter Aerodynamics*, 2nd ed., 2006, ch. 2-3): rotor coefficients
``C_T = T / (rho A (Omega R)^2)``, ``C_P = C_Q = P / (rho A (Omega R)^3)``, inflow ratio ``lambda = (V_c + v_i) /
(Omega R)``, solidity ``sigma = N c / (pi R)``, figure of merit ``FM = C_T^{3/2} / (sqrt(2) C_P)``. Propeller
coefficients (the UIUC / APC convention) ``c_T = T / (rho n^2 D^4) = C_T pi^3 / 4`` and
``c_P = P / (rho n^3 D^5) = C_P pi^4 / 4`` with ``n`` in rev/s.

The BEMT balances, on each annulus, the momentum thrust ``dC_T = 4 F lambda (lambda - lambda_c) r dr`` against the
blade-element thrust ``dC_T = (sigma / 2) (U / Omega R)^2 (c_l cos phi - c_d sin phi) dr`` (exact inflow angle
``phi = atan(lambda / r)``) and solves for ``lambda`` by bracketing; ``small_angle=True`` uses ``cos phi = 1``,
``sin phi = phi`` and ``U = Omega r``, the form in which an ideally twisted rotor without tip loss and profile drag
has uniform inflow and exactly the momentum-theory induced power ``C_P = C_T^{3/2} / sqrt(2)`` (the closed-form check
of :func:`ideal_twist_check`).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Airfoil", "Rotor", "calibrate_airfoil", "apc_10x4_7", "APC_10X4_7_GEOM", "APC_10X4_7_STATIC", "APC_SOURCE", "momentum_hover_power", "induced_velocity", "bemt", "ideal_twist_check",
           "prop_coefficients", "RHO_SEA_LEVEL"]

RHO_SEA_LEVEL = 1.225          # kg/m^3, ISA sea level


def induced_velocity(thrust: float, radius: float, rho: float = RHO_SEA_LEVEL) -> float:
    """Momentum-theory hover induced velocity ``v_h = sqrt(T / (2 rho A))`` [m/s]."""
    return math.sqrt(max(thrust, 0.0) / (2.0 * rho * math.pi * radius ** 2))


def momentum_hover_power(thrust: float, radius: float, rho: float = RHO_SEA_LEVEL) -> float:
    """Ideal hover power ``T^{3/2} / sqrt(2 rho A)`` [W] (momentum theory, uniform inflow)."""
    return max(thrust, 0.0) ** 1.5 / math.sqrt(2.0 * rho * math.pi * radius ** 2)


@dataclass
class Airfoil:
    """A linear lift curve with a stall clip and a parabolic drag polar: ``c_l = a (alpha - alpha_0)`` within
    ``[cl_min, cl_max]``, ``c_d = cd0 + cd2 (c_l - cl_cd_min)^2`` (stated defaults: a thin cambered section at low
    Reynolds number, ``a = 5.7 /rad``, ``alpha_0 = -4 deg``, ``c_d0 = 0.012``; change them for a measured polar)."""
    lift_slope: float = 5.7
    alpha0_deg: float = -4.0
    cl_max: float = 1.2
    cl_min: float = -0.6
    cd0: float = 0.012
    cd2: float = 0.02
    cl_cd_min: float = 0.3
    re_ref: float = 0.0
    re_exponent: float = -0.5
    re_lift_exponent: float = 0.0

    def cd0_at(self, re):
        """Profile drag at zero lift; with ``re_ref > 0`` scaled as ``cd0 (Re / Re_ref)^re_exponent`` (the laminar
        skin-friction trend that dominates low-Reynolds propellers)."""
        if self.re_ref <= 0:
            return self.cd0
        return self.cd0 * (max(float(re), 1.0) / self.re_ref) ** self.re_exponent

    def cl(self, alpha, re=None):
        a = self.lift_slope
        if re is not None and self.re_ref > 0 and self.re_lift_exponent:
            a = a * min((max(float(re), 1.0) / self.re_ref) ** self.re_lift_exponent, 1.5)
        return np.clip(a * (np.asarray(alpha, float) - math.radians(self.alpha0_deg)), self.cl_min, self.cl_max)

    def cd(self, cl, re=None):
        cl = np.asarray(cl, float)
        cd0 = self.cd0 if re is None else self.cd0_at(re)
        return cd0 + self.cd2 * (cl - self.cl_cd_min) ** 2


@dataclass
class Rotor:
    """A rotor: tip ``radius`` [m], ``blades``, chord ``c/R`` and pitch ``beta`` [deg] at stations ``r/R`` (the
    UIUC geometry-file convention), ``root`` cut-out ``r/R``."""
    radius: float
    blades: int
    r: Sequence[float]
    chord_over_r: Sequence[float]
    beta_deg: Sequence[float]
    root: float = 0.15
    airfoil: Airfoil = field(default_factory=Airfoil)
    name: str = ""

    def stations(self, n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mid-points ``r``, widths ``dr``, chord ``c/R`` and pitch [rad] of ``n`` annuli from the root to the tip."""
        edges = np.linspace(max(self.root, float(np.min(self.r))), 1.0, n + 1)
        rm = 0.5 * (edges[1:] + edges[:-1])
        c = np.interp(rm, self.r, self.chord_over_r)
        b = np.radians(np.interp(rm, self.r, self.beta_deg))
        return rm, np.diff(edges), c, b

    @classmethod
    def ideal(cls, radius: float, blades: int, chord_over_r: float, theta_tip_deg: float, root: float = 0.1,
              airfoil: Airfoil | None = None) -> Rotor:
        """Constant chord with the ideal twist ``theta = theta_tip / r`` (uniform inflow in hover, Leishman 3.3.3)."""
        r = np.linspace(root, 1.0, 200)
        return cls(radius, blades, r, np.full_like(r, chord_over_r), np.degrees(math.radians(theta_tip_deg) / r),
                   root=root, airfoil=airfoil or Airfoil(), name="ideal twist")


def _tip_loss(blades, r, lam):
    phi = np.maximum(lam / np.maximum(r, 1e-6), 1e-6)
    f = 0.5 * blades * (1.0 - r) / (r * phi)
    return np.clip((2.0 / math.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0)), 1e-4, 1.0)


NU_AIR = 1.46e-5               # m^2/s, kinematic viscosity of air at 15 C


def bemt(rotor: Rotor, omega: float, v_climb: float = 0.0, rho: float = RHO_SEA_LEVEL, *, n: int = 40,
         tip_loss: bool = True, profile_drag: bool = True, small_angle: bool = False) -> dict:
    """Thrust [N], torque [N m], power [W], ``C_T``, ``C_P``, ``FM`` and the propeller ``c_T``, ``c_P`` of
    ``rotor`` at ``omega`` [rad/s] and axial speed ``v_climb`` [m/s], with the spanwise inflow, angle of attack and
    lift coefficient."""
    from scipy.optimize import brentq
    R = rotor.radius
    vt = omega * R
    lam_c = v_climb / vt
    r, dr, c, beta = rotor.stations(n)
    sigma = rotor.blades * c / math.pi
    lam = np.zeros_like(r)
    af = rotor.airfoil

    def loads(lmb, ri, si, bi, ci):
        phi = math.atan2(lmb, ri)
        re = vt * math.hypot(ri, lmb) * ci * R / NU_AIR
        cl = float(af.cl(bi - (lmb / ri if small_angle else phi), re))
        cd = float(af.cd(cl, re)) if profile_drag else 0.0
        if small_angle:
            u2, cphi, sphi = ri * ri, 1.0, lmb / ri
        else:
            u2, cphi, sphi = ri * ri + lmb * lmb, math.cos(phi), math.sin(phi)
        dct = 0.5 * si * u2 * (cl * cphi - cd * sphi)
        dcq = 0.5 * si * u2 * (cl * sphi + cd * cphi) * ri
        return dct, dcq, cl, bi - (lmb / ri if small_angle else phi)

    def residual(lmb, i):
        F = float(_tip_loss(rotor.blades, r[i], lmb)) if tip_loss else 1.0
        return 4.0 * F * lmb * (lmb - lam_c) * r[i] - loads(lmb, r[i], sigma[i], beta[i], c[i])[0]

    for i in range(len(r)):
        lo, hi = max(lam_c, 0.0) + 1e-9, max(lam_c, 0.0) + 1.0
        if residual(lo, i) * residual(hi, i) > 0:          # no thrust-producing inflow (blade at or below zero lift)
            lam[i] = max(lam_c, 0.0)
            continue
        lam[i] = brentq(residual, lo, hi, args=(i,), xtol=1e-12)
    out = np.array([loads(lam[i], r[i], sigma[i], beta[i], c[i]) for i in range(len(r))])
    CT = float(np.sum(out[:, 0] * dr))
    CQ = float(np.sum(out[:, 1] * dr))
    A = math.pi * R ** 2
    T = CT * rho * A * vt ** 2
    Q = CQ * rho * A * vt ** 2 * R
    P = Q * omega
    FM = CT ** 1.5 / (math.sqrt(2.0) * CQ) if CQ > 0 and CT > 0 and v_climb == 0 else float("nan")
    return {"thrust_N": T, "torque_Nm": Q, "power_W": P, "CT": CT, "CP": CQ, "FM": FM,
            "cT_prop": CT * math.pi ** 3 / 4, "cP_prop": CQ * math.pi ** 4 / 4, "r": r.tolist(),
            "inflow": lam.tolist(), "alpha_deg": np.degrees(out[:, 3]).tolist(), "cl": out[:, 2].tolist(),
            "omega": omega, "rpm": omega * 60 / (2 * math.pi), "v_climb": v_climb}


def prop_coefficients(rotor: Rotor, rpms: Sequence[float], rho: float = RHO_SEA_LEVEL, **kw) -> dict:
    """Static propeller ``c_T`` and ``c_P`` (UIUC convention) at each RPM."""
    rows = [bemt(rotor, rpm * 2 * math.pi / 60, 0.0, rho, **kw) for rpm in rpms]
    return {"rpm": list(map(float, rpms)), "cT": [x["cT_prop"] for x in rows], "cP": [x["cP_prop"] for x in rows],
            "thrust_N": [x["thrust_N"] for x in rows], "power_W": [x["power_W"] for x in rows]}


def ideal_twist_check(radius: float = 0.127, blades: int = 2, chord_over_r: float = 0.1, theta_tip_deg: float = 8.0,
                      rpm: float = 6000.0, rho: float = RHO_SEA_LEVEL) -> dict:
    """The closed-form check of the BEMT: an ideally twisted rotor without tip loss and profile drag (small angles)
    has uniform inflow, so its power equals the momentum-theory ideal ``T^{3/2} / sqrt(2 rho A)``. Returns the
    relative power difference and the spread of the inflow."""
    rot = Rotor.ideal(radius, blades, chord_over_r, theta_tip_deg, airfoil=Airfoil(alpha0_deg=0.0, cl_max=10.0,
                                                                                  cl_min=-10.0))
    b = bemt(rot, rpm * 2 * math.pi / 60, rho=rho, n=80, tip_loss=False, profile_drag=False, small_angle=True)
    # the blades work on the annulus outside the root cut-out: the momentum disk is that annulus
    ideal = momentum_hover_power(b["thrust_N"], radius * math.sqrt(1.0 - rot.root ** 2), rho)
    lam = np.asarray(b["inflow"])
    return {"power_W": b["power_W"], "momentum_power_W": ideal, "rel_error": abs(b["power_W"] / ideal - 1.0),
            "inflow_spread": float((lam.max() - lam.min()) / lam.mean()), "thrust_N": b["thrust_N"]}


#: APC Slow Flyer 10 x 4.7 (UIUC Propeller Data Site, volume 1, Brandt & Selig 2011): geometry (r/R, c/R, beta deg)
#: from apcsf_10x4.7_geom.txt and the static test apcsf_10x4.7_static_kt0835.txt (RPM, c_T, c_P), read from a mirror of
#: the database (github.com/amr120/UIUC_Propellers, commit 6ce4e0a) because m-selig.ae.illinois.edu was not reachable
APC_10X4_7_GEOM = ((0.15, 0.109, 21.11), (0.20, 0.132, 23.90), (0.25, 0.156, 24.65), (0.30, 0.176, 24.11),
                   (0.35, 0.193, 22.78), (0.40, 0.206, 21.01), (0.45, 0.216, 19.00), (0.50, 0.223, 17.06),
                   (0.55, 0.226, 15.33), (0.60, 0.225, 13.82), (0.65, 0.219, 12.51), (0.70, 0.210, 11.36),
                   (0.75, 0.197, 10.27), (0.80, 0.179, 9.32), (0.85, 0.157, 8.36), (0.90, 0.130, 7.27),
                   (0.95, 0.087, 6.15), (1.00, 0.042, 5.04))
APC_10X4_7_STATIC = ((2377, 0.1059, 0.0431), (2676, 0.1079, 0.0437), (2947, 0.1079, 0.0437), (3234, 0.1104, 0.0444),
                     (3494, 0.1117, 0.0450), (3762, 0.1143, 0.0460), (4029, 0.1158, 0.0466), (4319, 0.1177, 0.0474),
                     (4590, 0.1200, 0.0484), (4880, 0.1223, 0.0494), (5147, 0.1237, 0.0500), (5417, 0.1252, 0.0508),
                     (5715, 0.1263, 0.0513), (5960, 0.1278, 0.0520), (6226, 0.1286, 0.0524), (6528, 0.1299, 0.0531))
APC_SOURCE = ("UIUC Propeller Data Site, vol. 1 (Brandt & Selig 2011): apcsf_10x4.7_geom.txt and "
              "apcsf_10x4.7_static_kt0835.txt")


def apc_10x4_7(airfoil: Airfoil | None = None, scale: float = 1.0) -> Rotor:
    """The APC Slow Flyer 10 x 4.7 (D = 10 in) as a :class:`Rotor`; ``scale`` scales the diameter (same c/R and
    pitch angles: a geometrically similar propeller)."""
    g = np.asarray(APC_10X4_7_GEOM)
    return Rotor(0.127 * scale, 2, g[:, 0], g[:, 1], g[:, 2], root=0.15, airfoil=airfoil or Airfoil(),
                 name=f"APC SF 10x4.7 x{scale:g}")


def calibrate_airfoil(rotor: Rotor, rpm, cT, cP, *, train=None, re_ref: float = 1e5) -> dict:
    """Fit the polar (lift slope, zero-lift angle, ``c_d0`` and the Reynolds exponents of lift and drag at
    ``re_ref``) to measured static ``c_T``, ``c_P`` on the ``train`` rows (relative least squares) and report the
    errors on the held-out rows before (the rotor's own polar) and after."""
    from dataclasses import replace

    from scipy.optimize import least_squares
    rpm, cT, cP = (np.asarray(a, float) for a in (rpm, cT, cP))
    train = np.ones(len(rpm), bool) if train is None else np.asarray(train, bool)

    def pred(af, rows):
        p = prop_coefficients(replace(rotor, airfoil=af), rpm[rows], n=24)
        return np.asarray(p["cT"]), np.asarray(p["cP"])

    def make(x):
        return replace(rotor.airfoil, lift_slope=x[0], alpha0_deg=x[1], cd0=x[2], re_ref=re_ref,
                       re_exponent=x[3], re_lift_exponent=x[4])

    def res(x):
        t, p = pred(make(x), train)
        return np.r_[t / cT[train] - 1, p / cP[train] - 1]

    af0 = rotor.airfoil
    x0 = [af0.lift_slope, af0.alpha0_deg, max(af0.cd0, 0.015), -0.5, 0.1]
    fit = least_squares(res, x0, bounds=([3.0, -8.0, 0.004, -1.0, 0.0], [6.5, 2.0, 0.06, 0.0, 0.5]), x_scale="jac")
    af = make(fit.x)
    test = ~train if (~train).any() else train

    def errs(a):
        t, p = pred(a, test)
        return {"cT_max_rel": float(np.max(np.abs(t / cT[test] - 1))), "cP_max_rel": float(np.max(np.abs(p / cP[test] - 1))),
                "cT_rms_rel": float(np.sqrt(np.mean((t / cT[test] - 1) ** 2))),
                "cP_rms_rel": float(np.sqrt(np.mean((p / cP[test] - 1) ** 2)))}
    return {"airfoil": af, "params": dict(zip(("lift_slope", "alpha0_deg", "cd0", "re_exponent", "re_lift_exponent"),
                                               map(float, fit.x), strict=True)),
            "before": errs(af0), "after": errs(af), "n_train": int(train.sum()), "n_test": int(test.sum())}
