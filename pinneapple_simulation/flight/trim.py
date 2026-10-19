"""Trim, ground equilibrium and stability derivatives of an :class:`~.aircraft.Aircraft`.

* :func:`trim`: steady straight flight at true airspeed V, altitude h and flight-path angle gamma: angle of attack,
  elevator and throttle that zero the body accelerations (u_dot, w_dot, q_dot), wings level, no sideslip;
* :func:`ground_equilibrium`: height and pitch attitude at rest on the gear (engines idle);
* :func:`stability_derivatives`: the usual non-dimensional derivatives by central differences of the full model about
  a trimmed state (rates non-dimensionalised with b / 2V and c / 2V), and the static margin -Cm_alpha / CL_alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .aircraft import Aircraft, Controls
from .rigid_body import euler_to_quat, state_vector


@dataclass
class TrimResult:
    V: float
    altitude: float
    gamma: float
    alpha: float
    theta: float
    elevator: float
    throttle: float
    x: np.ndarray
    controls: Controls
    residual: float
    CL: float
    feasible: bool

    def as_dict(self):
        return {"V": self.V, "altitude": self.altitude, "gamma_deg": math.degrees(self.gamma),
                "alpha_deg": math.degrees(self.alpha), "theta_deg": math.degrees(self.theta),
                "elevator_deg": math.degrees(self.elevator), "throttle": self.throttle, "CL": self.CL,
                "residual": self.residual, "feasible": self.feasible}


def _flight_state(V, h, alpha, gamma, psi=0.0):
    return state_vector((0.0, 0.0, -h), (V * math.cos(alpha), 0.0, V * math.sin(alpha)), (0.0, alpha + gamma, psi))


def trim(ac: Aircraft, V: float, altitude: float, gamma: float = 0.0, flap: float = 0.0, gear_down: bool = False,
         psi: float = 0.0) -> TrimResult:
    """Steady straight flight, wings level; the residual is the norm of (u_dot, w_dot, q_dot * c) [m/s^2]."""
    def controls(p):
        return Controls(elevator=p[1], throttle=p[2], flap=flap, gear_down=gear_down)

    def res(p):
        x = _flight_state(V, altitude, p[0], gamma, psi)
        ev = ac.evaluate(0.0, x, controls(p))
        d = ev["dx"]
        return np.array([d[3], d[5], d[11] * ac.af.mac])

    ws = 2 * ac.wing_loading / (ac.atmosphere(np.array([0, 0, -altitude]))[0] * V * V)
    a0 = ws / max(ac.aero.vlm.CL[1], 1e-3)
    sol = least_squares(res, [a0, 0.0, 0.5], bounds=([-0.3, -0.6, 0.0], [0.5, 0.6, 1.5]), xtol=1e-12, ftol=1e-12,
                        gtol=1e-12)
    a, de, dt = sol.x
    x = _flight_state(V, altitude, a, gamma, psi)
    ev = ac.evaluate(0.0, x, controls(sol.x))
    r = float(np.linalg.norm(res(sol.x)))
    feasible = r < 1e-6 and dt <= 1.0 and abs(de) < math.radians(25)
    return TrimResult(V, altitude, gamma, float(a), float(a + gamma), float(de), float(dt), x, controls(sol.x), r,
                      float(ev["aero"]["CL"]), bool(feasible))


def ground_equilibrium(ac: Aircraft, flap: float = 0.0, throttle: float = 0.0, psi: float = 0.0,
                       x_pos: float = 0.0, y_pos: float = 0.0) -> np.ndarray:
    """State at rest on the gear: height and pitch attitude with zero vertical force and pitching moment."""
    u = Controls(flap=flap, throttle=throttle, brake=1.0)

    def x_of(p):
        return np.r_[x_pos, y_pos, p[0], 0.0, 0.0, 0.0, euler_to_quat(0.0, p[1], psi), 0.0, 0.0, 0.0]

    def res(p):
        ev = ac.evaluate(0.0, x_of(p), u)
        return np.array([ev["dx"][5], ev["dx"][11] * ac.af.mac])

    z0 = -max(-float(ac.to_body(g.contact)[2]) for g in ac.gear) if ac.gear else 0.0
    sol = least_squares(res, [z0 * 1.0 - 0.0, 0.0], xtol=1e-12, ftol=1e-12)
    return x_of(sol.x)


def _coeffs(ac: Aircraft, x, u: Controls) -> dict[str, float]:
    ev = ac.evaluate(0.0, x, u)
    ae = ev["aero"]
    q, S, b, c = ae["q"], ac.af.wing_area, ac.af.span, ac.af.mac
    F, M = ae["F"], ae["M"]
    a, be = ae["alpha"], ae["beta"]
    # wind axes (no sideslip in the reference state): lift up, drag back
    CL = (-F[2] * math.cos(a) + F[0] * math.sin(a)) / (q * S)
    CD = (-F[0] * math.cos(a) - F[2] * math.sin(a)) / (q * S)
    return {"CL": CL, "CD": CD, "CY": F[1] / (q * S), "Cl": M[0] / (q * S * b), "Cm": M[1] / (q * S * c),
            "Cn": M[2] / (q * S * b), "alpha": a, "beta": be}


def stability_derivatives(ac: Aircraft, tr: TrimResult, d_angle: float = math.radians(0.5),
                          d_rate: float = 0.01, d_ctl: float = math.radians(1.0)) -> dict[str, float]:
    """Aerodynamic derivatives [/rad] about the trim state (engines included at the trim throttle: the thrust line
    contributes to Cm through its moment arm only through the trim, not the derivatives, which use the aero loads)."""
    x0, u0 = tr.x, tr.controls
    V = tr.V
    b, c = ac.af.span, ac.af.mac
    out: dict[str, float] = {}

    def at(dalpha=0.0, dbeta=0.0, p=0.0, q=0.0, r=0.0, **dctl):
        a = tr.alpha + dalpha
        u_b = V * math.cos(a) * math.cos(dbeta)
        v_b = V * math.sin(dbeta)
        w_b = V * math.sin(a) * math.cos(dbeta)
        x = x0.copy()
        x[3:6] = [u_b, v_b, w_b]
        x[10:13] = [p, q, r]
        u = replace(u0, **{k: getattr(u0, k) + v for k, v in dctl.items()})
        return _coeffs(ac, x, u)

    def d(key_fn, step, name_map):
        cp, cm = key_fn(step), key_fn(-step)
        for coef, name in name_map.items():
            out[name] = (cp[coef] - cm[coef]) / (2 * step)

    d(lambda s: at(dalpha=s), d_angle, {"CL": "CL_alpha", "Cm": "Cm_alpha", "CD": "CD_alpha"})
    d(lambda s: at(dbeta=s), d_angle, {"CY": "CY_beta", "Cl": "Cl_beta", "Cn": "Cn_beta"})
    sp = d_rate * 2 * V / b
    sq = d_rate * 2 * V / c
    for coef, nm in (("Cl", "Cl_p"), ("Cn", "Cn_p"), ("CY", "CY_p")):
        out[nm] = (at(p=sp)[coef] - at(p=-sp)[coef]) / (2 * d_rate)
    for coef, nm in (("CL", "CL_q"), ("Cm", "Cm_q")):
        out[nm] = (at(q=sq)[coef] - at(q=-sq)[coef]) / (2 * d_rate)
    for coef, nm in (("Cl", "Cl_r"), ("Cn", "Cn_r"), ("CY", "CY_r")):
        out[nm] = (at(r=sp)[coef] - at(r=-sp)[coef]) / (2 * d_rate)
    for ctl, names in (("elevator", {"CL": "CL_de", "Cm": "Cm_de"}), ("aileron", {"Cl": "Cl_da", "Cn": "Cn_da"}),
                       ("rudder", {"CY": "CY_dr", "Cl": "Cl_dr", "Cn": "Cn_dr"}), ("flap", {"CL": "CL_df", "Cm": "Cm_df"})):
        d(lambda s, ctl=ctl: at(**{ctl: s}), d_ctl, names)
    out["static_margin"] = -out["Cm_alpha"] / out["CL_alpha"]
    out["x_np"] = float(ac.cg[0] + out["static_margin"] * c)
    return out


def modes(ac: Aircraft, tr: TrimResult, eps: float = 1e-5) -> dict[str, object]:
    """Eigenvalues of the 12-state linearisation (position, velocity, attitude, rates; the quaternion's vector part
    as the attitude) about a trimmed state, with the controls fixed, by central differences of the full model."""
    from .rigid_body import quat_multiply
    x0, u0 = tr.x, tr.controls
    q0 = x0[6:10]

    def expand(y):
        x = x0.copy()
        x[0:6] = x0[0:6] + y[0:6]
        dq = np.r_[1.0, 0.5 * y[6:9]]
        q = quat_multiply(q0, dq)
        x[6:10] = q / np.linalg.norm(q)
        x[10:13] = x0[10:13] + y[9:12]
        return x

    def f(y):
        x = expand(y)
        d = ac.evaluate(0.0, x, u0)["dx"]
        qd = d[6:10]
        # body-frame attitude error rate: 2 (q0^-1 q)_dot vector part = omega (small)
        q = x[6:10]
        conj = q * np.array([1, -1, -1, -1])
        rot = 2 * quat_multiply(conj, qd)[1:]
        return np.r_[d[0:6], rot, d[10:13]]

    A = np.zeros((12, 12))
    for j in range(12):
        e = np.zeros(12)
        e[j] = eps
        A[:, j] = (f(e) - f(-e)) / (2 * eps)
    ev = np.linalg.eigvals(A)
    return {"A": A, "eigenvalues": ev}


def classify_modes(eigs: np.ndarray, tol: float = 1e-6) -> dict[str, dict[str, float]]:
    """Name the complex pairs by frequency (short period: the fastest longitudinal-looking pair is left to the caller
    with the eigenvectors; here only frequency, damping and the real roots are returned sorted)."""
    out = []
    for lam in eigs:
        if lam.imag > tol:
            wn = abs(lam)
            out.append({"wn": float(wn), "zeta": float(-lam.real / wn), "period": float(2 * math.pi / lam.imag)})
    real = sorted(float(lam.real) for lam in eigs if abs(lam.imag) <= tol)
    return {"oscillatory": sorted(out, key=lambda d: -d["wn"]), "real": real}
