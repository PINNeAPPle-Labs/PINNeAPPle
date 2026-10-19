"""Verification of the flight-dynamics building blocks, as numbers a guide or a CI job can quote.

    from pinneapple_simulation.flight import verification
    rep = verification.run()                 # dict: turbulence, trim, conservation, lattice

* turbulence: variance and band-averaged PSD ratio (generated / target) of each generator and component;
* trim: trim residual and the drift of airspeed, altitude and load factor over 10 s stick fixed, every aircraft;
* conservation: energy and angular-momentum error of the integrator for a spinning body under gravity;
* lattice: lift of the lattice-in-the-loop model against the steady vortex lattice at two angles.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import catalog
from .control import StickFixed
from .rigid_body import state_vector
from .simulate import energy_check, simulate
from .trim import trim
from .turbulence import (
    DrydenFilter,
    FrozenField3D,
    FrozenLine,
    psd,
    psd_temporal,
    turbulence_spec,
    welch_psd,
)

CRUISE = {"light_ga": (55.0, 1500.0), "regional_turboprop": (120.0, 3000.0), "a320": (128.0, 3000.0),
          "motor_glider": (30.0, 1500.0)}
# speeds for the turbulence and gust studies at 3000 m: the cruise speeds, except the glider, flown at 40 m/s
# (CL ~0.6) as glider pilots speed up in rough air; nearer its best-glide speed little alpha margin to the clean
# CLmax is left for gusts (an assumption of the studies, not a limit)
ENCOUNTER_SPEED = {"light_ga": 55.0, "regional_turboprop": 120.0, "a320": 128.0, "motor_glider": 40.0}


def band_ratios(W, P, target, lo, hi, nbins=8):
    edges = np.geomspace(lo, hi, nbins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        k = (W >= a) & (W < b)
        if k.sum() >= 2:
            out.append(float(P[k].mean() / target(W[k]).mean()))
    return out


def turbulence_checks(seed: int = 3) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": [], "curves": []}
    h = 3000.0
    for model in ("dryden", "von_karman"):
        s = turbulence_spec(h, "severe", model)
        line = FrozenLine(s, length=400_000.0, dx=2.0, seed=seed)
        for i, c in enumerate("uvw"):
            W, P = welch_psd(line.series[i], 2.0, 32)
            r = band_ratios(W, P, lambda w, c=c, s=s: psd(s, w, c), 4 / s.lengths[i], 0.5)
            out["cases"].append({"generator": "FrozenLine", "model": model, "component": c,
                                 "sigma_target": float(s.sigma[i]), "sigma": float(line.series[i].std()),
                                 "psd_ratio_min": min(r), "psd_ratio_max": max(r)})
            if c in "uw":
                sel = (W > 0) & (W < 0.6)
                out["curves"].append({"title": f"{model.replace('_', ' ')}, {c}: frozen line", "W": W[sel],
                                      "measured": P[sel], "target": psd(s, W[sel], c)})
    s = turbulence_spec(h, "moderate", "dryden")
    V, dt = 100.0, 0.02
    y = DrydenFilter(s, V, dt, seed=seed).series(200_000)
    for i, c in enumerate("uvw"):
        W, P = welch_psd(y[:, i], dt, 32)
        r = band_ratios(W, P, lambda w, c=c: psd_temporal(s, w, V, c), 4 * V / s.lengths[i], 10.0)
        out["cases"].append({"generator": "DrydenFilter", "model": "dryden", "component": c,
                             "sigma_target": float(s.sigma[i]), "sigma": float(y[:, i].std()),
                             "psd_ratio_min": min(r), "psd_ratio_max": max(r)})
        if c == "w":
            sel = (W > 0) & (W < 20)
            out["curves"].append({"title": "Dryden forming filter, w (V = 100 m/s)", "W": W[sel], "measured": P[sel],
                                  "target": psd_temporal(s, W[sel], V, c), "xlabel": "frequency [rad/s]"})
    s = turbulence_spec(h, "severe")
    f = FrozenField3D(s, dx_min=2.0, seed=seed, n_shells=96, n_dirs=32)
    x = np.arange(0, 200_000, 4.0)
    U = f.components(np.c_[x, np.zeros_like(x), np.zeros_like(x)])
    for i, c in enumerate("uvw"):
        W, P = welch_psd(U[:, i], 4.0, 16)
        r = band_ratios(W, P, lambda w, c=c, s=s: psd(s, w, c), 5 / s.L_w, 0.2)
        rh = band_ratios(W, P, lambda w, c=c, s=s: psd(s, w, c), 0.2, 0.95 * f.k_max_used, nbins=4)
        out["cases"].append({"generator": "FrozenField3D", "model": "von_karman", "component": c,
                             "sigma_target": float(s.sigma[i] * math.sqrt(f.variance_fraction)),
                             "sigma": float(U[:, i].std()), "psd_ratio_min": min(r), "psd_ratio_max": max(r),
                             "psd_ratio_high_band": rh})
        if c == "w":
            sel = W > 0
            out["curves"].append({"title": "3-D random-mode field, w along a line", "W": W[sel], "measured": P[sel],
                                  "target": psd(s, W[sel], c),
                                  "markers": [(f.k_max_used, "field cutoff k_max = π/(2Δx)"),
                                              (2 * math.pi / 34.1, "2π / A320-class span")]})
    out["field3d_variance_fraction"] = f.variance_fraction
    out["field3d_band_rad_m"] = [f.k_min_used, f.k_max_used]
    out["field3d_dx_min_m"] = f.dx_min
    return out


def trim_checks(names: list[str] | None = None, hold: float = 10.0) -> list[dict[str, Any]]:
    rows = []
    for n in names or catalog.names():
        ac = catalog.build(n)
        V, h = CRUISE[n]
        tr = trim(ac, V, h)
        hist = simulate(ac, tr.x, StickFixed(tr.controls), hold, 0.01)
        rows.append({"aircraft": n, "V": V, "altitude": h, **tr.as_dict(), "hold_s": hold,
                     "dV_max": float(np.abs(hist.airspeed - V).max()), "dh_max": float(np.abs(hist.altitude - h).max()),
                     "dnz_max": float(np.abs(hist.nz - 1).max()),
                     "dtheta_max_deg": float(np.degrees(np.abs(hist.euler[:, 1] - tr.theta).max()))})
    return rows


def conservation_check() -> dict[str, Any]:
    ac = catalog.build("a320")
    x0 = state_vector((0, 0, -3000), (120, 3, 4), (0.3, 0.2, 0.5), (0.4, 0.9, -0.6))
    return {dt: energy_check(ac, x0, 20.0, dt) for dt in (0.02, 0.01, 0.005)}


def lattice_check() -> list[dict[str, Any]]:
    rows = []
    for n in catalog.names():
        ac = catalog.build(n)
        for a_deg in (0.0, 4.0):
            a = math.radians(a_deg)
            out = ac.aero.loads(np.array([50 * math.cos(a), 0, 50 * math.sin(a)]), np.zeros(3), 1.225)
            ref = ac.aero.vlm.coefficients(a, 0.0)["CL"]
            rows.append({"aircraft": n, "alpha_deg": a_deg, "CL_loop": out["CL"], "CL_vlm": ref,
                         "rel_diff": abs(out["CL"] - ref) / abs(ref)})
    return rows


def run() -> dict[str, Any]:
    t = turbulence_checks()
    curves = t.pop("curves")
    return {"turbulence": t, "trim": trim_checks(), "conservation": conservation_check(), "lattice": lattice_check(),
            "_curves": curves}
