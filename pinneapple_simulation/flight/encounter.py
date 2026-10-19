"""Turbulence and gust encounters in flight, and their load-factor statistics.

:func:`turbulence_encounter` trims the aircraft at a speed and altitude, flies it through a wind field (by default a
frozen 3-D von Karman field, so the two wings meet different gusts) stick-fixed or under the autopilot, and returns
the history with :func:`encounter_metrics`: RMS and peak of the vertical load-factor increment at the CG and at each
cabin station, lateral load factor, bank and pitch excursions, airspeed and angle-of-attack excursions.

Checks against closed forms:

* :func:`gust_sensitivity`: the quasi-steady load-factor response to a vertical gust, A = rho V CL_alpha / (2 W/S)
  per m/s of gust; A sigma_w is the RMS increment of a rigid wing that does not move and has no lift lag (an upper
  bound: plunge and pitch relieve it, unsteady lift build-up too);
* :func:`discrete_gust` + :func:`pratt_check`: the full 1-cosine gust of half-length H = 12.5 MAC, compared with the
  Pratt-Walker formula (alleviation factor K_g from the mass ratio mu, which also accounts for unsteady lift build-up
  the quasi-steady model leaves out).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .aircraft import Aircraft
from .control import Autopilot, LoopDesign, StickFixed
from .rigid_body import G0
from .simulate import FlightHistory, rms, simulate
from .trim import stability_derivatives, trim
from .turbulence import (
    DiscreteGust,
    FrozenField3D,
    FrozenLine,
    TurbulenceSpec,
    pratt_gust_load,
    turbulence_spec,
)


def gust_sensitivity(ac: Aircraft, V: float, rho: float, cl_alpha: float) -> float:
    """dn / dw_gust [1/(m/s)] of a fixed wing in quasi-steady flow."""
    return rho * V * cl_alpha / (2 * ac.wing_loading)


def stall_angle(ac: Aircraft, trim_alpha: float, trim_CL: float, cl_alpha: float) -> float:
    """Angle of attack [rad] at which the linear lift reaches the aircraft's clean CLmax (``ac.clmax_clean``, an
    assumption of the catalog): alpha_trim + (CLmax - CL_trim) / CL_alpha. Beyond it the linear lattice is outside
    its validity (no stall in the model)."""
    return trim_alpha + (ac.clmax_clean - trim_CL) / cl_alpha


def encounter_metrics(ac: Aircraft, hist: FlightHistory, t_skip: float = 5.0,
                      alpha_stall: float | None = None) -> dict[str, Any]:
    """Statistics after ``t_skip``. With ``alpha_stall`` [rad] the run is also checked against the model's validity:
    the fraction of time above the stall angle and ``departed`` (bank beyond 90 deg or alpha 10 deg past stall), in
    which case the numbers describe a loss of control the quasi-steady linear model cannot represent."""
    sel = hist.t >= t_skip
    dn = hist.nz[sel] - 1.0
    phi, theta = hist.euler[sel, 0], hist.euler[sel, 1]
    out: dict[str, Any] = {
        "aircraft": ac.name, "duration_s": float(hist.t[sel][-1] - hist.t[sel][0]),
        "dnz_cg_rms": rms(dn), "dnz_cg_max": float(dn.max()), "dnz_cg_min": float(dn.min()),
        "ny_cg_rms": rms(hist.ny[sel]), "ny_cg_peak": float(np.abs(hist.ny[sel]).max()),
        "phi_rms_deg": math.degrees(rms(phi)), "phi_peak_deg": math.degrees(float(np.abs(phi - phi.mean()).max())),
        "theta_rms_deg": math.degrees(rms(theta)),
        "theta_peak_deg": math.degrees(float(np.abs(theta - theta.mean()).max())),
        "p_rms_deg_s": math.degrees(rms(hist.x[sel, 10])), "q_rms_deg_s": math.degrees(rms(hist.x[sel, 11])),
        "V_rms": rms(hist.airspeed[sel]), "alpha_rms_deg": math.degrees(rms(hist.alpha[sel])),
        "alpha_max_deg": math.degrees(float(hist.alpha[sel].max())),
        "altitude_rms_m": rms(hist.altitude[sel]),
        "wind_w_rms": rms(hist.wind_cg[sel, 2]), "wind_v_rms": rms(hist.wind_cg[sel, 1]),
        "wind_u_rms": rms(hist.wind_cg[sel, 0]),
        "cabin": {}}
    if alpha_stall is not None:
        a = hist.alpha[sel]
        out["alpha_stall_deg"] = math.degrees(alpha_stall)
        out["time_above_stall_frac"] = float(np.mean(a > alpha_stall))
        out["departed"] = bool(out["phi_peak_deg"] > 90.0 or a.max() > alpha_stall + math.radians(10.0))
    for k in hist.cabin_nz:
        d = hist.cabin_nz[k][sel] - 1.0
        out["cabin"][k] = {"dnz_rms": rms(d), "dnz_peak": float(np.abs(d).max()),
                           "ny_rms": rms(hist.cabin_ny[k][sel]), "ny_peak": float(np.abs(hist.cabin_ny[k][sel]).max())}
    return out


def turbulence_encounter(ac: Aircraft, V: float, altitude: float, spec: TurbulenceSpec | None = None,
                         wind=None, duration: float = 60.0, dt: float = 0.01, controller: str = "autopilot",
                         field: str = "3d", seed: int = 0, record_gamma: bool = False, record_every: int = 1,
                         design: LoopDesign | None = None, t_skip: float = 5.0):
    """Fly ``duration`` seconds through turbulence. ``spec`` defaults to severe at ``altitude`` (von Karman);
    ``field`` "3d" (:class:`FrozenField3D`, resolved to wavelengths of ~4 MAC) or "line" (:class:`FrozenLine`, the same gust
    across the span); a ready ``wind`` field overrides both. Returns (history, metrics, trim, derivatives, wind)."""
    tr = trim(ac, V, altitude)
    d = stability_derivatives(ac, tr)
    if wind is None:
        spec = spec or turbulence_spec(altitude, "severe")
        if field == "3d":
            # resolved band up to k_max = pi / (2 dx_min); its line spectra follow the target up to ~k_max / 4, i.e.
            # wavelengths >= 4 MAC with dx_min = MAC / 4 (verification.turbulence_checks); log shells: the mode count
            # does not grow with k_max
            wind = FrozenField3D(spec, dx_min=max(ac.af.mac / 4, 0.2), n_shells=128, seed=seed)
        else:
            wind = FrozenLine(spec, length=max(4 * V * duration, 20 * spec.L_w), dx=max(ac.af.mac / 2, 0.25),
                              seed=seed)
    ctl = StickFixed(tr.controls) if controller == "stick_fixed" else Autopilot(ac, tr, d, design)
    hist = simulate(ac, tr.x, ctl, duration, dt, wind=wind, record_gamma=record_gamma, record_every=record_every)
    m = encounter_metrics(ac, hist, t_skip, stall_angle(ac, tr.alpha, tr.CL, d["CL_alpha"]))
    rho = ac.atmosphere(tr.x)[0]
    A = gust_sensitivity(ac, V, rho, d["CL_alpha"])
    m.update(controller=controller, field=field, V=V, altitude=altitude, wing_loading=ac.wing_loading,
             cl_alpha=d["CL_alpha"], gust_sensitivity=A,
             quasi_steady_dnz_rms=A * (m["wind_w_rms"] if m["wind_w_rms"] > 0 else float("nan")),
             spec=spec.as_dict() if spec is not None else None, trim=tr.as_dict())
    return hist, m, tr, d, wind


def discrete_gust(ac: Aircraft, V: float, altitude: float, U: float, H: float | None = None,
                  duration: float | None = None, dt: float = 0.005):
    """Stick-fixed response to an up-gust of amplitude U [m/s] (full 1-cosine, half-length H, default 12.5 MAC)
    starting 2 s ahead of the aircraft. Returns (history, dnz_max)."""
    tr = trim(ac, V, altitude)
    H = H or 12.5 * ac.af.mac
    g = DiscreteGust(amplitude=U, length=H, start=2.0 * V, shape="full")
    T = duration or (2.0 + 2 * H / V + 4.0)
    hist = simulate(ac, tr.x, StickFixed(tr.controls), T, dt, wind=g)
    return hist, float((hist.nz - 1.0).max()), tr


def pratt_check(ac: Aircraft, V: float, altitude: float, U_de: float) -> dict[str, float]:
    """Simulated peak dn for a 12.5-MAC 1-cosine gust against the Pratt-Walker formula at the same equivalent gust
    and speed. The model is quasi-steady, so its peak sits between Pratt (with K_g) and the unalleviated value."""
    rho = ac.atmosphere(np.array([0, 0, -altitude]))[0]
    sigma = rho / 1.225
    U = U_de / math.sqrt(sigma)
    hist, dn, tr = discrete_gust(ac, V, altitude, U)
    d = stability_derivatives(ac, tr)
    p = pratt_gust_load(ac.wing_loading, d["CL_alpha"], ac.af.mac, U_de, V * math.sqrt(sigma), rho=rho, g=G0)
    return {"aircraft": ac.name, "V": V, "altitude": altitude, "U_de": U_de, "dn_sim": dn, "dn_pratt": p["dn"],
            "dn_unalleviated": p["dn_unalleviated"], "K_g": p["K_g"], "mu": p["mu"], "cl_alpha": d["CL_alpha"],
            "ratio_sim_to_unalleviated": dn / p["dn_unalleviated"], "ratio_sim_to_pratt": dn / p["dn"]}
