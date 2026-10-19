"""Orbit decay of low-Earth-orbit satellites by atmospheric drag, with a physics-informed model and its checks.

Physics (near-circular orbit, orbit-averaged): da/dt = -B rho(h) sqrt(mu a), B = C_D A / m (m^2/kg), h = a - R_E.
The thermospheric density comes from NRLMSIS 2.0 (``pymsis``, MIT; Emmert et al. 2021) driven by the solar flux
F10.7 and the geomagnetic index Ap, averaged around the orbit (latitude band of the inclination, all local times).
Without ``pymsis`` a piecewise-exponential standard atmosphere is used (Vallado, *Fundamentals of Astrodynamics*,
table 8-4, mean solar activity).

* :func:`reference_decay`  classical integration (adaptive RK) of the ODE, with a Monte-Carlo envelope over the
  uncertain inputs (solar activity, ballistic coefficient)
* :func:`train_decay_pinn` a PINN a(t) = a0 + tau s g_theta(tau) (initial condition exact) trained on the residual
  of the ODE; judged against the reference integration and its envelope
* :func:`sgp4_decay`       the altitude trend that SGP4 itself predicts from a TLE (its own power-law density and
  the fitted B*), an independent drag model
"""
from __future__ import annotations

import math
import time

import numpy as np

MU = 3.986004418e14            # m^3/s^2
R_E = 6378137.0                # m
RHO0_BSTAR = 0.15696615        # kg/m^2 per Earth radius: B* = B rho0 / 2 (SGP4 convention)

# Vallado table 8-4: base altitude (km), nominal density (kg/m^3), scale height (km)
_EXP = np.array([
    (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473), (130, 8.484e-9, 12.636),
    (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523), (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835), (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00)])


def exponential_density(h_km):
    h = np.asarray(h_km, dtype=float)
    i = np.clip(np.searchsorted(_EXP[:, 0], h, side="right") - 1, 0, len(_EXP) - 1)
    return _EXP[i, 1] * np.exp(-(h - _EXP[i, 0]) / _EXP[i, 2])


def msis_available() -> bool:
    try:
        import pymsis  # noqa: F401
        return True
    except ImportError:
        return False


def orbit_mean_density(h_km, date="2026-10-13T00:00", f107=150.0, f107a=150.0, ap=15.0, inclination_deg=51.6,
                       n_lat=7, n_lt=8):
    """Orbit-averaged NRLMSIS 2.0 total mass density (kg/m^3) at altitudes ``h_km``: mean over latitudes sampled
    along the orbit (|lat| <= inclination, weighted by the time spent there) and over all local times."""
    h = np.atleast_1d(np.asarray(h_km, dtype=float))
    if not msis_available():
        return exponential_density(h)
    import pymsis
    # time spent at argument of latitude u is uniform: lat = asin(sin i sin u)
    u = (np.arange(n_lat) + 0.5) / n_lat * (np.pi / 2)
    lats = np.degrees(np.arcsin(np.sin(np.radians(inclination_deg)) * np.sin(u)))
    lons = np.linspace(0, 360, n_lt, endpoint=False)
    t = np.datetime64(str(date).replace("Z", "")[:19])
    out = pymsis.calculate(t, lons, lats, h, f107s=[f107], f107as=[f107a], aps=[[ap] * 7])
    rho = np.asarray(out)[..., 0]                       # (1, nlon, nlat, nalt) total mass density
    return rho.reshape(-1, len(h)).mean(axis=0)


class DensityTable:
    """log-density on an altitude grid, interpolated (NumPy and PyTorch) so models can be evaluated fast."""

    def __init__(self, h_km=None, **kw):
        self.h = np.arange(150.0, 1001.0, 5.0) if h_km is None else np.asarray(h_km, float)
        self.logrho = np.log(orbit_mean_density(self.h, **kw))
        self.kw = kw
        self.model = "NRLMSIS 2.0 (pymsis)" if msis_available() else "exponential (Vallado table 8-4)"

    def __call__(self, h_km):
        return np.exp(np.interp(h_km, self.h, self.logrho))

    def torch(self, h_km):
        import torch
        hh = torch.as_tensor(self.h, dtype=h_km.dtype)
        lr = torch.as_tensor(self.logrho, dtype=h_km.dtype)
        x = torch.clamp(h_km, float(self.h[0]), float(self.h[-1]) - 1e-6)
        i = torch.clamp(torch.searchsorted(hh, x.detach(), right=True) - 1, 0, len(self.h) - 2)
        w = (x - hh[i]) / (hh[i + 1] - hh[i])
        return torch.exp(lr[i] * (1 - w) + lr[i + 1] * w)


def ballistic_from_bstar(bstar_per_er: float) -> float:
    """B = C_D A / m (m^2/kg) implied by an SGP4 B* (1/Earth radii): B* = B rho0 / 2."""
    return 2.0 * bstar_per_er / RHO0_BSTAR


def _rhs(table, B):
    def f(_t, a):
        h = (a[0] - R_E) / 1000.0
        return [-B * table(h) * math.sqrt(MU * a[0])]
    return f


def reference_decay(h0_km: float, B: float, days: float, table: DensityTable, n: int = 200,
                    reentry_km: float = 150.0):
    """Altitude (km) vs time (days) from an adaptive Runge-Kutta integration of the decay ODE."""
    from scipy.integrate import solve_ivp
    t_eval = np.linspace(0, days * 86400.0, n)

    def hit(_t, a):
        return (a[0] - R_E) / 1000.0 - reentry_km
    hit.terminal = True
    sol = solve_ivp(_rhs(table, B), (0, days * 86400.0), [R_E + h0_km * 1000.0], t_eval=t_eval, rtol=1e-10,
                    atol=1e-3, events=hit, method="DOP853")
    h = np.full(n, np.nan)
    h[: sol.y.shape[1]] = (sol.y[0] - R_E) / 1000.0
    return t_eval / 86400.0, h


def envelope(h0_km, B, days, *, f107=150.0, f107a=None, ap=15.0, date="2026-10-13T00:00", inclination_deg=51.6,
             f107_rel_sigma=0.2, B_rel_sigma=0.15, n_samples=24, seed=0, n=200):
    """Monte-Carlo spread of the reference decay over the uncertain inputs: solar flux (log-normal, 20 %),
    geomagnetic activity (Ap x 0.5..2) and the ballistic coefficient (log-normal, 15 %). Returns t, median, 5 %,
    95 % altitude curves."""
    rng = np.random.default_rng(seed)
    curves = []
    f107a = f107 if f107a is None else f107a
    for _ in range(n_samples):
        fs = f107 * math.exp(rng.normal(0, f107_rel_sigma))
        tab = DensityTable(date=date, f107=fs, f107a=f107a * fs / f107, ap=ap * math.exp(rng.uniform(-0.7, 0.7)),
                           inclination_deg=inclination_deg)
        t, h = reference_decay(h0_km, B * math.exp(rng.normal(0, B_rel_sigma)), days, tab, n=n)
        curves.append(h)
    C = np.array(curves)
    return t, np.nanmedian(C, 0), np.nanpercentile(C, 5, 0), np.nanpercentile(C, 95, 0)


def train_decay_pinn(h0_km: float = 415.0, B: float = 0.005, days: float = 60.0, *, f107: float = 150.0,
                     ap: float = 15.0, date: str = "2026-10-13T00:00", inclination_deg: float = 51.6,
                     iters: int = 1500, n_colloc: int = 256, seed: int = 0, with_envelope: bool = True) -> dict:
    """Train the decay PINN and return its report: prediction, reference, envelope, checks and verdict."""
    import torch
    torch.manual_seed(seed)
    t0 = time.time()
    table = DensityTable(date=date, f107=f107, f107a=f107, ap=ap, inclination_deg=inclination_deg)
    T = days * 86400.0
    a0 = R_E + h0_km * 1000.0
    rate0 = B * table(h0_km) * math.sqrt(MU * a0)          # m/s at t=0
    s = max(rate0 * T * 3.0, 1.0)                           # metres: scale of the total decay

    net = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 1)).double()
    kf = torch.tensor([1.0, 2.0], dtype=torch.float64) * math.pi

    def a_of(tau):
        x = torch.cat([tau, torch.sin(tau * kf), torch.cos(tau * kf)], -1)
        return a0 - s * tau * torch.nn.functional.softplus(net(x))   # decay is monotone by construction

    def residual(tau):
        tau = tau.requires_grad_(True)
        a = a_of(tau)
        da = torch.autograd.grad(a.sum(), tau, create_graph=True)[0] / T
        h = (a - R_E) / 1000.0
        rhs = -B * table.torch(h) * torch.sqrt(MU * a)
        return (da - rhs) / rate0

    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-4)
    hist = []
    for it in range(iters):
        tau = torch.rand(n_colloc, 1, dtype=torch.float64)
        loss = residual(tau).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == iters - 1:
            hist.append({"iter": it, "physics": loss.item()})
    t_ref, h_ref = reference_decay(h0_km, B, days, table)
    tau = torch.tensor(t_ref / days, dtype=torch.float64)[:, None]
    with torch.no_grad():
        h_pinn = ((a_of(tau) - R_E) / 1000.0).numpy()[:, 0]
    res = residual(tau.clone()).detach().numpy()[:, 0]
    ok = np.isfinite(h_ref)
    err_m = np.abs(h_pinn - h_ref)[ok] * 1000.0
    out = {"h0_km": h0_km, "B_m2_kg": B, "days": days, "f107": f107, "ap": ap, "date": date,
           "inclination_deg": inclination_deg, "density_model": table.model,
           "t_days": t_ref.tolist(), "h_pinn_km": h_pinn.tolist(), "h_ref_km": np.where(ok, h_ref, None).tolist(),
           "decay_rate_m_per_day_t0": rate0 * 86400.0, "history": hist}
    checks = [
        {"id": "decay_vs_rk", "name": "Max |PINN - RK integration| over the window", "value": float(err_m.max()), "unit": "m",
         "threshold": max(50.0, 0.02 * float(np.nanmax(h_ref[0] - h_ref)) * 1000.0),
         "status": None},
        {"id": "decay_ode_residual", "name": "ODE residual (RMS / initial decay rate)", "value": float(np.sqrt((res ** 2).mean())), "unit": "-",
         "threshold": 0.05, "status": None},
        {"id": "decay_monotone", "name": "Monotone decay (da/dt < 0 everywhere)", "value": float(np.max(np.diff(h_pinn))), "unit": "km",
         "threshold": 0.0, "status": None},
    ]
    for c in checks:
        c["status"] = "pass" if c["value"] <= c["threshold"] else "fail"
    if with_envelope:
        te, med, lo, hi = envelope(h0_km, B, days, f107=f107, ap=ap, date=date, inclination_deg=inclination_deg,
                                   n_samples=16)
        out.update(env_median_km=np.nan_to_num(med, nan=-1).tolist(), env_p05_km=np.nan_to_num(lo, nan=-1).tolist(),
                   env_p95_km=np.nan_to_num(hi, nan=-1).tolist())
        inside = float(np.mean((h_pinn >= lo - 1e-3) & (h_pinn <= hi + 1e-3)))
        checks.append({"id": "decay_in_envelope", "name": "PINN inside the 5-95 % input-uncertainty envelope", "value": inside,
                       "unit": "fraction", "threshold": 0.95, "status": "pass" if inside >= 0.95 else "warn",
                       "higher_is_better": True})
    nf = sum(c["status"] == "fail" for c in checks)
    nw = sum(c["status"] == "warn" for c in checks)
    out["checks"] = checks
    out["verdict"] = {"status": "trustworthy" if nf == nw == 0 else "use_with_caution" if nf == 0 else
                      "not_trustworthy"}
    out["train_seconds"] = time.time() - t0
    return out


def sgp4_decay(line1: str, line2: str, days: float = 30.0, step_min: float = 10.0):
    """Mean altitude (km) vs time (days) predicted by SGP4 from a TLE: the geocentric radius averaged over each
    orbit (removes the eccentricity and J2 oscillations), minus the equatorial radius."""
    from sgp4.api import Satrec
    sat = Satrec.twoline2rv(line1, line2)
    jd0 = sat.jdsatepoch + sat.jdsatepochF
    t = np.arange(0, days * 1440.0, step_min / 2) / 1440.0
    jd = jd0 + t
    whole = np.floor(jd - 0.5) + 0.5
    e, r, _v = sat.sgp4_array(whole, jd - whole)
    rr = np.linalg.norm(r, axis=1) * 1000.0
    period_min = 2 * math.pi / sat.no_kozai
    w = max(1, int(round(period_min / (step_min / 2))))
    kernel = np.ones(w) / w
    mean_r = np.convolve(rr, kernel, mode="valid")
    tm = t[w // 2: w // 2 + len(mean_r)]
    ok = e[w // 2: w // 2 + len(mean_r)] == 0
    return tm[ok], (mean_r[ok] - R_E) / 1000.0, {"bstar": sat.bstar, "B_from_bstar": ballistic_from_bstar(sat.bstar),
                                                  "epoch_jd": jd0, "inclination_deg": math.degrees(sat.inclo)}


ISS_B = 0.005   # m^2/kg: C_D 2.2, ~950 m^2 mean cross-section, ~420 t (an assumption, not a measured value)


def compare_with_sgp4(line1: str, line2: str, days: float = 30.0, f107: float = 70.0, ap: float = 8.0,
                      B: float = ISS_B) -> dict:
    """Decay predicted by this model (NRLMSIS + a physical ballistic coefficient B) next to SGP4's own prediction
    from the same TLE. SGP4's B* is a fitted drag term for its own power-law atmosphere, not a physical ballistic
    coefficient (converted with B* = B rho0 / 2 it is typically an order of magnitude smaller), so the two drag
    models are independent; agreement of the decay rates within a factor 2 is the expectation."""
    from .timescales import datetime_from_jd
    t, h, info = sgp4_decay(line1, line2, days)
    date = datetime_from_jd(info["epoch_jd"]).strftime("%Y-%m-%dT%H:%M")
    tab = DensityTable(date=date, f107=f107, f107a=f107, ap=ap, inclination_deg=info["inclination_deg"])
    tr, hr = reference_decay(float(h[0]), B, days, tab)
    rate_sgp4 = -np.polyfit(t, h, 1)[0] * 1000.0          # m/day
    rate_model = -np.polyfit(tr[np.isfinite(hr)], hr[np.isfinite(hr)], 1)[0] * 1000.0
    return {"t_sgp4": t.tolist(), "h_sgp4": h.tolist(), "t_model": tr.tolist(), "h_model": hr.tolist(),
            "rate_sgp4_m_day": float(rate_sgp4), "rate_model_m_day": float(rate_model),
            "ratio": float(rate_model / rate_sgp4) if rate_sgp4 else float("nan"), "f107": f107, "ap": ap, "B": B,
            "density_model": tab.model, **info}
