"""Orbit determination and space traffic: a numerical propagator with drag, the covariance of an orbit by lead
time, conjunctions, and the trust-facing checks that say when a propagation stops being usable.

The pieces, and the independent reference each is measured against in
``tests/test_astro_orbit_determination.py``:

* :func:`propagate` -- two-body + J2 + drag of a Cartesian state, with the density of :func:`default_density`
  (NRLMSISE 2.0 through ``pymsis`` when it is installed, else the Vallado table already in :mod:`.decay`) and an
  optional ``storm`` multiplier on that density. Measured against :func:`propagate_kepler` (analytic, universal
  variables), against the conserved energy and angular momentum of the two-body problem, and its secular decay of
  the semi-major axis against the orbit-averaged ``da/dt = -B rho sqrt(mu a)`` of :mod:`.decay`.
* :func:`sgp4_states` / :func:`sgp4_self_consistency` -- SGP4 from a TLE (the ``sgp4`` package, Vallado et al. 2006),
  and the check that a propagation reproduces the element set it was built from: the revolutions it actually makes
  against the TLE's own mean motion, and its osculating elements at the epoch against the mean elements of the TLE
  (which differ by the Brouwer short-period terms).
* :func:`propagate_covariance` / :func:`monte_carlo_covariance` / :func:`position_covariance` -- the covariance by
  lead time, by the State Transition Matrix and by Monte Carlo; for a two-body orbit the along-track growth is also
  known in closed form (an in-plane velocity error of ``dv`` puts the satellite ``3 dv t`` down-track), and
  :func:`trust_horizon` turns that growth into the lead time at which the orbit stops being usable.
* :func:`conjunction_probability` -- TCA by a dense scan of the relative distance and the 2-D probability of
  collision in the encounter plane (Foster & Estes 1992; Alfano 2005) with the propagated covariance, on top of
  :func:`pinneapple_physics.astro.conjunction.collision_probability_2d`.
* :func:`regime_changes` -- the residual that flags a **manoeuvre** (a step in the semi-major axis) and a
  **storm-driven drag regime** (a step in its rate) without either being in the model.
* :func:`tle_check` / :func:`tle_measurement_check` -- a bad TLE fit: the checksum, the element ranges, whether the
  propagation reproduces the element set, and whether the fit is old enough to have lost the satellite.
* :func:`LearnedTleCorrection` -- the learned correction to the TLE propagation error, fitted on later TLEs as truth
  (:func:`tle_pair_residuals` builds that training set from a TLE archive); :func:`traffic_ensemble` combines it
  with SGP4 and the numerical propagator in :class:`pinneapple_physics.ensemble.PhysicsEnsemble`, whose residual is
  :func:`orbit_residual` (how badly a predicted trajectory satisfies the equations of motion).

What needs the network, and is therefore **not** reproduced here: CelesTrak GP data (current TLEs, and the archived
TLEs of a past event), the space-weather indices that drive the density (F10.7 and Ap, from GFZ Potsdam / NOAA
SWPC / CelesTrak), and the SGP4-VER verification suite. :data:`STARLINK_FEB_2022` records the declared real-world
case -- the 49 satellites of the 2022-02-03 launch, 38 of which were lost to the storm of 2022-02-04 -- with the
inputs a reproduction needs; what is measured here is the mechanism (a density jump and the decay it produces), not
the event.

References: Vallado, Crawford, Hujsak & Kelso, AIAA 2006-6753 (SGP4); Vallado, *Fundamentals of Astrodynamics and
Applications*, 4th ed., 2013 (universal variables, table 8-4 atmosphere); Foster & Estes, NASA JSC-25898, 1992 and
Alfano, AAS 05-354, 2005 (2-D collision probability); Marcum, 1948 (the closed form of the isotropic case);
Emmert et al., J. Space Weather Space Clim. 11, 2021 (NRLMSISE 2.0); SpaceX's 2022-02-08 update on the lost
Starlink satellites.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import decay
from .conjunction import collision_probability_2d
from .satellites import Satellite

MU_EARTH = 398600.4418          # km^3/s^2
R_E = 6378.137                  # km, equatorial radius (WGS84)
J2 = 1.08262668e-3
OMEGA_EARTH = 7.292115e-5       # rad/s, the atmosphere's rotation for the relative velocity of the drag

__all__ = ["DragModel", "StormDensity", "default_density", "density_model_name", "propagate", "propagate_kepler",
           "semi_major_axis", "mean_semi_major_axis",
           "state_transition", "rtn_frame", "sgp4_states", "tle_period_s", "sgp4_self_consistency", "tle_checksum",
           "tle_check", "tle_measurement_check", "propagate_covariance", "monte_carlo_covariance",
           "position_covariance", "trust_horizon", "encounter_plane", "conjunction_probability", "regime_changes",
           "reentry_warning", "correction_features", "LearnedTleCorrection", "tle_pair_residuals",
           "residual_by_lead_time", "orbit_residual", "traffic_ensemble", "STARLINK_FEB_2022"]


# ----------------------------------------------------------------------------------------------- atmospheres
_DENSITY_TABLE: dict = {}


def default_density(h_km):
    """Orbit-averaged thermospheric density (kg/m^3) at altitude ``h_km``: NRLMSISE 2.0 when ``pymsis`` is installed,
    the Vallado piecewise-exponential table of :mod:`.decay` otherwise (:func:`density_model_name` says which).

    The model is evaluated once on the altitude grid of :class:`pinneapple_physics.astro.decay.DensityTable` and
    interpolated from then on: a propagator calls the density at every integrator stage, where the model itself
    (which probes for ``pymsis`` on each call) is three hundred times slower than the interpolation.
    """
    key = "default"
    if key not in _DENSITY_TABLE:
        _DENSITY_TABLE[key] = decay.DensityTable()
    return _DENSITY_TABLE[key](h_km)


def density_model_name() -> str:
    """Which atmosphere :func:`default_density` is using on this machine."""
    return "NRLMSISE 2.0 (pymsis)" if decay.msis_available() else "exponential (Vallado table 8-4)"


class StormDensity:
    """A density model multiplied by a factor: the geomagnetic-storm injection.

    In operation the factor is what a JB2008 / NRLMSISE-00 run driven by the event's Ap and F10.7 would give; offline
    it is the synthetic jump that the drag response is measured against (``decay.reference_decay`` takes any
    ``table(h) -> rho`` callable, so this is also the adapter for the decay ODE of :mod:`.decay`).
    """

    def __init__(self, factor: float = 1.0, model: Callable | None = None):
        if factor < 0:
            raise ValueError("the storm factor is a multiplier on the density: it cannot be negative")
        self.factor, self.model = float(factor), model or default_density

    def __call__(self, h_km):
        return self.factor * self.model(np.asarray(h_km, dtype=float))


@dataclass
class DragModel:
    """Ballistic coefficient ``B = C_D A / m`` (m^2/kg), the atmosphere it flies in, and an optional time-dependent
    multiplier on the density. ``storm(t_s) -> factor`` is 1.0 when None (quiet atmosphere)."""
    B_m2_kg: float = 0.005
    density: Callable | None = None            # default_density: the cached table of this module
    storm: Callable | None = None

    def __post_init__(self):
        if self.density is None:
            self.density = default_density

    def rho(self, t_s: float, h_km: float) -> float:
        f = 1.0 if self.storm is None else float(self.storm(t_s))
        return float(np.asarray(self.density(h_km)).reshape(-1)[0]) * f

    def accel(self, t_s: float, r, v) -> np.ndarray:
        """Drag acceleration (km/s^2) on an inertial state (km, km/s): ``-B rho |v_rel| v_rel / 2`` with the
        velocity relative to the rotating atmosphere. ``B rho`` is in 1/m and the velocity in km/s, so the result
        carries a factor 1e3 to come out in km/s^2."""
        r = np.asarray(r, dtype=float)
        rn = float(np.linalg.norm(r))
        v_rel = np.asarray(v, dtype=float) - np.cross([0.0, 0.0, OMEGA_EARTH], r)
        rho = self.rho(t_s, max(rn - R_E, 0.0))
        return -0.5 * self.B_m2_kg * rho * 1e3 * float(np.linalg.norm(v_rel)) * v_rel


# ----------------------------------------------------------------------------------------------- propagation
def _accel(t_s, r, v, mu, j2_value, Re, drag):
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = float(np.linalg.norm(r))
    g = -mu / rn**3
    a = g * r
    if j2_value:
        z2 = (r[2] / rn) ** 2
        k = 1.5 * j2_value * mu * Re**2 / rn**5
        a = a + np.array([k * r[0] * (5 * z2 - 1), k * r[1] * (5 * z2 - 1), k * r[2] * (5 * z2 - 3)])
    if drag is not None:
        a = a + drag.accel(t_s, r, v)
    return a


def propagate(state0, t_s, *, drag: DragModel | None = None, j2: bool = True, mu: float = MU_EARTH,
              Re: float = R_E, rtol: float = 1e-11, atol: float = 1e-9) -> np.ndarray:
    """States ``(len(t_s), 6)`` (km, km/s) of the numerical propagation from ``state0``: two-body, J2 and drag,
    integrated with DOP853. ``t_s``: seconds after the epoch state."""
    from scipy.integrate import solve_ivp

    t_eval = np.atleast_1d(np.asarray(t_s, dtype=float))
    if t_eval.size == 0:
        return np.zeros((0, 6))
    j2_value = J2 if j2 else 0.0
    rhs = lambda t, s: np.concatenate([s[3:], _accel(t, s[:3], s[3:], mu, j2_value, Re, drag)])  # noqa: E731
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), list(map(float, state0)), t_eval=t_eval, method="DOP853",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"propagation failed: {sol.message}")
    return sol.y.T


def _stumpff(z):
    """Stumpff functions C(z) and S(z) of the universal-variable Kepler problem (Vallado, table 2-3)."""
    z = np.asarray(z, dtype=float)
    c2 = np.empty_like(z)
    c3 = np.empty_like(z)
    pos, neg = z > 1e-8, z < -1e-8
    small = ~(pos | neg)
    sz = np.sqrt(z[pos])
    c2[pos] = (1.0 - np.cos(sz)) / z[pos]
    c3[pos] = (sz - np.sin(sz)) / sz**3
    sz = np.sqrt(-z[neg])
    c2[neg] = (np.cosh(sz) - 1.0) / (-z[neg])
    c3[neg] = (np.sinh(sz) - sz) / sz**3
    c2[small] = 0.5
    c3[small] = 1.0 / 6.0
    return c2, c3


def propagate_kepler(state0, t_s, *, mu: float = MU_EARTH, tol: float = 1e-12, itmax: int = 64) -> np.ndarray:
    """Analytic two-body propagation (universal variables, Stumpff functions) of ``state0`` to ``t_s``: the closed
    form the numerical propagator is checked against, and the fast propagator used for every covariance here."""
    t = np.atleast_1d(np.asarray(t_s, dtype=float))
    r0v = np.asarray(state0[:3], dtype=float)
    v0v = np.asarray(state0[3:], dtype=float)
    r0, v0sq = float(np.linalg.norm(r0v)), float(v0v @ v0v)
    vr0 = float(r0v @ v0v) / r0
    alpha = 2.0 / r0 - v0sq / mu                       # 1/a
    sqrt_mu = math.sqrt(mu)
    chi = np.where(np.abs(alpha) > 1e-12, sqrt_mu * alpha * t, r0 * vr0 / sqrt_mu * t)
    chi = np.maximum(chi, 1e-9)
    for _ in range(itmax):
        c2, c3 = _stumpff(alpha * chi**2)
        d = ((r0 * vr0 / sqrt_mu * chi**2 * c2 + (1.0 - alpha * r0) * chi**3 * c3 + r0 * chi - sqrt_mu * t)
             / (r0 * vr0 / sqrt_mu * chi * (1.0 - alpha * chi**2 * c3) + (1.0 - alpha * r0) * chi**2 * c2 + r0))
        chi = chi - d
        if np.all(np.abs(d) < tol):
            break
    c2, c3 = _stumpff(alpha * chi**2)
    f, g = 1.0 - chi**2 / r0 * c2, t - chi**3 / sqrt_mu * c3
    r = np.outer(f, r0v) + np.outer(g, v0v)
    rn = np.linalg.norm(r, axis=1)
    fd = sqrt_mu / (rn * r0) * (alpha * chi**3 * c3 - chi)
    gd = 1.0 - chi**2 / rn * c2
    v = np.outer(fd, r0v) + np.outer(gd, v0v)
    return np.concatenate([r, v], axis=1)


def semi_major_axis(states, mu: float = MU_EARTH):
    """Semi-major axis (km) of each state from its energy: ``1/a = 2/r - v^2/mu``."""
    S = np.atleast_2d(np.asarray(states, dtype=float))
    r = np.linalg.norm(S[:, :3], axis=1)
    v2 = np.sum(S[:, 3:] ** 2, axis=1)
    return 1.0 / (2.0 / r - v2 / mu)


def mean_semi_major_axis(states, t_s, period_s: float | None = None):
    """The semi-major axis averaged over one orbital period: the osculating value carries the J2 short-period
    oscillation (kilometres, on a decay of metres per day), so this -- the quantity a TLE's mean motion reports --
    is what a drag rate has to be measured on. Returns ``(t_days, a_mean_km)`` on the times where a full period is
    inside the window; the period is the one of the orbit in ``states`` unless it is given."""
    t = np.atleast_1d(np.asarray(t_s, dtype=float))
    a = semi_major_axis(states)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    per = float(period_s) if period_s is not None else 2.0 * math.pi * math.sqrt(float(np.median(a)) ** 3 / MU_EARTH)
    w = max(1, int(round(per / step)))
    if w >= len(a):
        raise ValueError("the window is shorter than one period: nothing to average")
    kernel = np.ones(w) / w
    return (t[w // 2: w // 2 + len(a) - w + 1] / 86400.0, np.convolve(a, kernel, mode="valid"))


def rtn_frame(r, v) -> np.ndarray:
    """Rows R (radial), T (along-track), N (cross-track) of a state, ``(3, 3)``."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    R = r / np.linalg.norm(r)
    N = np.cross(r, v)
    N = N / np.linalg.norm(N)
    return np.stack([R, np.cross(N, R), N])


def state_transition(state0, t_s, *, propagator: Callable | None = None, rel_step: float = 1e-7) -> np.ndarray:
    """State transition matrix ``(len(t_s), 6, 6)`` by central differences of ``propagator(state0, t_s)`` (two-body
    by default). Exact for a linear system; for Kepler it carries the linearisation error, which is what
    :func:`monte_carlo_covariance` measures."""
    prop = propagator or propagate_kepler
    x0 = np.asarray(state0, dtype=float)
    t = np.atleast_1d(np.asarray(t_s, dtype=float))
    Phi = np.zeros((len(t), 6, 6))
    for j in range(6):
        h = rel_step * max(abs(x0[j]), 1e-3)
        sp, sm = x0.copy(), x0.copy()
        sp[j] += h
        sm[j] -= h
        Phi[:, :, j] = (prop(sp, t) - prop(sm, t)) / (2.0 * h)
    return Phi


def propagate_covariance(state0, P0, t_s, *, propagator: Callable | None = None, rel_step: float = 1e-7,
                         along_track_floor_km_day: float = 0.0) -> np.ndarray:
    """Covariance ``(len(t_s), 6, 6)`` of the state, ``Phi P0 Phi^T``, plus an optional along-track floor that grows
    linearly and stands for what the two-body linearisation does not know about: the density (a relative error in
    ``B*rho`` of ``e`` turns into ``1.5 e a`` of semi-major axis and ``~3 e v t`` down-track, which is why the
    drag uncertainty and not the initial one dominates a LEO covariance after a day)."""
    Phi = state_transition(state0, t_s, propagator=propagator, rel_step=rel_step)
    P0 = np.asarray(P0, dtype=float)
    P = np.einsum("kij,jl,kml->kim", Phi, P0, Phi)
    if along_track_floor_km_day:
        t = np.atleast_1d(np.asarray(t_s, dtype=float)) / 86400.0
        F = rtn_frame(np.asarray(state0[:3]), np.asarray(state0[3:]))
        s = (along_track_floor_km_day * t) ** 2
        for k in range(len(t)):
            P[k, :3, :3] += s[k] * np.outer(F[1], F[1])
    return P


def monte_carlo_covariance(state0, P0, t_s, *, n: int = 300, seed: int = 0, propagator: Callable | None = None):
    """Covariance ``(len(t_s), 6, 6)`` of ``n`` samples drawn from ``N(state0, P0)`` and propagated: the check on the
    linear covariance, which it agrees with while the linearisation holds."""
    prop = propagator or propagate_kepler
    x0, P0 = np.asarray(state0, dtype=float), np.asarray(P0, dtype=float)
    t = np.atleast_1d(np.asarray(t_s, dtype=float))
    rng = np.random.default_rng(seed)
    S = x0 + rng.normal(size=(int(n), 6)) @ np.linalg.cholesky(P0).T
    ref = prop(x0, t)
    out = np.zeros((len(t), 6, 6))
    for i in range(len(S)):
        d = prop(S[i], t) - ref                          # deviations from the reference trajectory
        for k in range(len(t)):
            out[k] += np.outer(d[k], d[k])
    return out / max(len(S) - 1, 1)


def position_covariance(states, cov) -> dict:
    """Position sigmas (km) of a state covariance, resolved in the RTN frame of the reference state: the radial,
    along-track and cross-track uncertainty of a lead time, as an orbit's trust horizon needs them."""
    S = np.atleast_2d(np.asarray(states, dtype=float))
    C = np.asarray(cov, dtype=float)
    sig = np.zeros((len(S), 3))
    for k in range(len(S)):
        F = rtn_frame(S[k, :3], S[k, 3:])
        sig[k] = np.sqrt(np.maximum(np.diag(F @ C[k][:3, :3] @ F.T), 0.0))
    r1 = np.sqrt(np.sum(sig**2, axis=1))
    return {"sigma_radial_km": sig[:, 0], "sigma_along_km": sig[:, 1], "sigma_cross_km": sig[:, 2],
            "sigma_3d_km": r1, "radius_95_km": 2.7955 * r1}


def trust_horizon(state0, P0, *, tolerance_km: float = 1.0, days: float = 7.0, n: int = 48,
                  propagator: Callable | None = None, along_track_floor_km_day: float = 0.0,
                  errors_km=None) -> dict:
    """The lead time at which an orbit stops being usable: the first time the position uncertainty (3-D 1 sigma of
    the propagated covariance, or the measured ``errors_km`` on the same grid when they are given) reaches
    ``tolerance_km``. Linear interpolation between the grid points, so the answer is a number with the tolerance it
    was asked for: the trust horizon is a definition, and this is the one used here."""
    t_days = np.linspace(0.0, float(days), int(n) + 1)
    t_s = t_days * 86400.0
    C = propagate_covariance(state0, P0, t_s, propagator=propagator,
                             along_track_floor_km_day=along_track_floor_km_day)
    S = (propagator or propagate_kepler)(state0, t_s)
    pos = position_covariance(S, C)
    y = pos["sigma_3d_km"] if errors_km is None else np.asarray(errors_km, dtype=float)
    criterion = "covariance (3-D 1 sigma)" if errors_km is None else "measured position error"
    above = np.nonzero(y >= tolerance_km)[0]
    if above.size == 0:
        return {"t_hours": float(days * 24.0), "reached": False, "criterion": criterion,
                "tolerance_km": float(tolerance_km), "sigma_km_at_horizon": float(y[-1]),
                "growth_km_per_day": float((y[-1] - y[0]) / max(days, 1e-12)), "t_days": t_days, "sigma_km": y}
    i = int(above[0])
    if i == 0:
        t_hit = 0.0
    else:
        y0, y1 = y[i - 1], y[i]
        w = 0.0 if y1 == y0 else (tolerance_km - y0) / (y1 - y0)
        t_hit = (t_days[i - 1] + w * (t_days[i] - t_days[i - 1])) * 24.0
    return {"t_hours": float(t_hit), "reached": True, "criterion": criterion,
            "tolerance_km": float(tolerance_km), "sigma_km_at_horizon": float(np.interp(t_hit, t_days * 24.0, y)),
            "growth_km_per_day": float((y[i] - y[max(i - 1, 0)]) / max(t_days[i] - t_days[max(i - 1, 0)], 1e-12)),
            "t_days": t_days, "sigma_km": y}


# ----------------------------------------------------------------------------------------------- TLE and SGP4
def sgp4_states(line1: str, line2: str, t_s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position (km) and velocity (km/s) from SGP4, in TEME, ``t_s`` seconds after the TLE epoch (``sgp4`` package)."""
    sat = Satellite("tle", line1, line2)
    jd = sat.epoch_jd + np.atleast_1d(np.asarray(t_s, dtype=float)) / 86400.0
    r, v, err = sat.teme(jd)
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float), np.asarray(err)


def tle_period_s(line1: str, line2: str) -> float:
    """Orbital period (s) of the TLE's own mean motion: ``2 pi / no_kozai``."""
    sat = Satellite("tle", line1, line2)
    return 2.0 * math.pi / float(sat.satrec.no_kozai) * 60.0


def _revolutions(line1, line2, days=1.0, step_s=20.0) -> float:
    """Revolutions SGP4 actually makes in ``days``: the in-plane angle of the position about the orbit normal,
    unwrapped."""
    t = np.arange(0.0, days * 86400.0 + step_s, step_s)
    r, v, _ = sgp4_states(line1, line2, t)
    n = np.cross(r, v)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    e1 = r[0] / np.linalg.norm(r[0])
    x, y = r @ e1, r @ np.cross(n[0], e1)
    return float((np.unwrap(np.arctan2(y, x))[-1] - np.unwrap(np.arctan2(y, x))[0]) / (2.0 * math.pi))


def sgp4_self_consistency(line1: str, line2: str, *, days: float = 1.0) -> dict:
    """Does the propagation reproduce the element set it was built from?

    Two measurements, both offline: (1) the revolutions SGP4 makes in ``days`` against the revolutions implied by the
    TLE's own mean motion; (2) the osculating elements of the SGP4 state at the epoch against the mean elements of the
    TLE. They are not equal by construction anywhere except at the epoch and differ by the Brouwer short-period
    terms -- ``delta a`` of a few km, ``delta i`` of a few hundredths of a degree for a LEO object -- so the checks
    in :func:`tle_check` use a tolerance calibrated on a real element set, not zero.
    """
    sat = Satellite("tle", line1, line2)
    r, v, err = sat.teme(np.array([sat.epoch_jd]))
    x = np.concatenate([r[0], v[0]])
    a_osc = float(semi_major_axis(x)[0])
    rn = float(np.linalg.norm(r[0]))
    hvec = np.cross(r[0], v[0])
    evec = ((float(v[0] @ v[0]) - MU_EARTH / rn) * r[0] - float(r[0] @ v[0]) * v[0]) / MU_EARTH
    e_osc = float(np.linalg.norm(evec))
    i_osc = math.degrees(math.acos(float(hvec[2]) / float(np.linalg.norm(hvec))))
    n_rad_s = float(sat.satrec.no_kozai) / 60.0
    a_mean = (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0)
    rev_meas = _revolutions(line1, line2, days)
    rev_tle = float(sat.satrec.no_kozai) * 1440.0 * days / (2.0 * math.pi)
    return {"a_osculating_km": a_osc, "a_from_mean_motion_km": float(a_mean), "da_km": a_osc - float(a_mean),
            "da_rel": (a_osc - float(a_mean)) / float(a_mean),
            "e_osculating": e_osc, "e_tle": float(sat.satrec.ecco), "de": e_osc - float(sat.satrec.ecco),
            "i_osculating_deg": i_osc, "i_tle_deg": math.degrees(float(sat.satrec.inclo)),
            "di_deg": i_osc - math.degrees(float(sat.satrec.inclo)),
            "revolutions_measured": rev_meas, "revolutions_from_tle": rev_tle,
            "revolutions_rel": (rev_meas - rev_tle) / rev_tle,
            "sgp4_error_codes": sorted({int(c) for c in np.atleast_1d(err)}),
            "period_s": tle_period_s(line1, line2), "bstar": float(sat.satrec.bstar),
            "epoch_jd": float(sat.epoch_jd)}


def tle_checksum(line: str) -> tuple[int, int]:
    """(computed, transmitted) modulo-10 checksum of a TLE line: the digits plus 1 for every minus sign, in
    columns 1-68."""
    if len(line) < 69:
        raise ValueError("a TLE line has 69 characters")
    body = line[:68]
    total = sum(int(c) for c in body if c.isdigit()) + body.count("-")
    try:
        sent = int(line[68])
    except ValueError:
        sent = -1
    return total % 10, sent


def tle_check(line1: str, line2: str, *, span_days: float = 1.0, max_age_days: float = 3.0,
              epoch_jd: float | None = None) -> dict:
    """Is this element set a fit worth propagating? Every check is offline.

    * the two checksums, and the satellite numbers of the two lines agree (a truncated or retyped TLE);
    * the elements are in range and the orbit is above the surface at perigee (an element set whose perigee is below
      the ground has already decayed: propagating it produces a trajectory, not a satellite);
    * the propagation reproduces the element set (:func:`sgp4_self_consistency`): the revolutions against the TLE's
      mean motion and the osculating--mean differences within the short-period amplitudes (``da_rel`` 2e-3,
      ``di`` 0.2 deg). A TLE that fails this is not a fit of this trajectory;
    * SGP4 raises no error over the span, and, when a reference epoch is given, the fit is not older than
      ``max_age_days`` (an old fit is a bad fit for today, not a broken one: ``status`` is ``warn``).
    """
    checks = []

    def add(cid, name, ok, value=None, threshold=None, unit="-", warn_only=False):
        checks.append({"id": cid, "name": name, "value": value, "threshold": threshold, "unit": unit,
                       "status": "pass" if ok else ("warn" if warn_only else "fail")})

    try:
        c1, s1 = tle_checksum(line1)
        c2, s2 = tle_checksum(line2)
        add("checksum", "TLE checksums", c1 == s1 and c2 == s2, f"{c1}/{s1}, {c2}/{s2}", "equal")
        add("satnum", "Satellite number of the two lines agree", line1[2:7] == line2[2:7], line1[2:7] + line2[2:7],
            "equal")
        sat = Satellite("tle", line1, line2)
    except Exception as exc:                                    # noqa: BLE001  a TLE that does not even parse
        add("parse", "The two lines parse as a TLE", False, str(exc)[:60], "parses")
        return {"suspect": True, "checks": checks, "flags": ["parse"]}
    a_mean = (MU_EARTH / (float(sat.satrec.no_kozai) / 60.0) ** 2) ** (1.0 / 3.0)
    e, inc = float(sat.satrec.ecco), math.degrees(float(sat.satrec.inclo))
    perigee = a_mean * (1.0 - e) - R_E
    add("elements", "0 <= e < 1, 0 <= i <= 180 deg, n > 0", 0.0 <= e < 1.0 and 0.0 <= inc <= 180.0
        and sat.satrec.no_kozai > 0, f"e={e:.6f} i={inc:.4f}", "in range")
    add("perigee", "Perigee above the surface (km)", perigee > 0.0, float(perigee), 0.0, "km")
    sc = sgp4_self_consistency(line1, line2, days=span_days)
    add("reproduces_mean_motion", "Revolutions over the span vs the TLE's own mean motion", abs(sc["revolutions_rel"]) < 1e-2,
        float(sc["revolutions_rel"]), 1e-2, "relative")
    add("osculating_semi_major_axis", "Osculating a vs a from the mean motion (short-period term)", abs(sc["da_rel"]) < 2e-3,
        float(sc["da_rel"]), 2e-3, "relative")
    add("osculating_inclination", "Osculating i vs the TLE's mean i (short-period term)", abs(sc["di_deg"]) < 0.2,
        float(sc["di_deg"]), 0.2, "deg")
    add("sgp4_errors", "SGP4 error codes over the span (0 = fine)", all(c == 0 for c in sc["sgp4_error_codes"]),
        sc["sgp4_error_codes"], [0], "code")
    if epoch_jd is not None:
        age = float(epoch_jd) - sc["epoch_jd"]
        add("age", "Age of the fit at the requested epoch (days)", abs(age) <= max_age_days, abs(age), max_age_days,
            "day", warn_only=True)
    flags = [c["id"] for c in checks if c["status"] == "fail"]
    return {"suspect": bool(flags), "checks": checks, "flags": flags,
            "self_consistency": sc, "period_s": sc["period_s"]}


def tle_measurement_check(line1: str, line2: str, **kw) -> bool:
    """The :class:`pinneapple_physics.ensemble.PhysicsEnsemble` hook: ``True`` when the element set is suspect and the
    ensemble should predict with its learned weights and learn nothing from it."""
    return bool(tle_check(line1, line2, **kw)["suspect"])


# ----------------------------------------------------------------------------------------------- conjunctions
def encounter_plane(rel_pos, rel_vel, cov_rel):
    """The encounter plane of a conjunction: the miss vector and the two principal sigmas of the relative covariance
    projected onto the plane perpendicular to the relative velocity (Foster & Estes 1992).

    ``(xm, zm, sigma_x, sigma_z)`` in km, with the plane axes at the principal directions of the projected
    covariance, so that :func:`pinneapple_physics.astro.conjunction.collision_probability_2d` applies directly.
    Returns the basis too, so a caller can say which way the miss lies.
    """
    rho = np.asarray(rel_pos, dtype=float)
    vel = np.asarray(rel_vel, dtype=float)
    C = np.asarray(cov_rel, dtype=float)[:3, :3]
    n_hat = vel / np.linalg.norm(vel)
    e1 = rho - float(rho @ n_hat) * n_hat
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-12 else np.cross(n_hat, [1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    E = np.stack([e1, e2])
    P2 = E @ C @ E.T
    w, V = np.linalg.eigh(P2)
    miss = E @ rho
    miss_p = V.T @ miss
    return float(miss_p[0]), float(miss_p[1]), float(math.sqrt(max(w[0], 0.0))), float(math.sqrt(max(w[1], 0.0))), E


def conjunction_probability(chief, deputy, cov_chief, cov_deputy, *, hard_body_radius_km: float = 0.010,
                            duration_s: float | None = None, n_scan: int = 2000, velocity=True,
                            propagator: Callable | None = None) -> dict:
    """TCA and the 2-D probability of collision of two orbits with propagated covariances.

    Both states are propagated with the same two-body model over one period of the chief (a dense scan brackets the
    minimum of the relative distance, a golden-section search refines it); at the TCA the relative covariance is
    ``cov_chief + cov_deputy``, projected into the encounter plane, and the Pc is the Gaussian mass inside the
    hard-body disk of radius ``hard_body_radius_km``. ``velocity=False`` evaluates the covariances at the epoch
    instead of propagating them (the shortcut some screeners take, kept here so its cost can be measured).
    """
    from scipy.optimize import minimize_scalar

    prop = propagator or propagate_kepler
    c0, d0 = np.asarray(chief, dtype=float), np.asarray(deputy, dtype=float)
    mu = MU_EARTH
    if duration_s is None:
        duration_s = 2.0 * math.pi * math.sqrt(float(semi_major_axis(c0)[0]) ** 3 / mu)
    t = np.linspace(0.0, float(duration_s), int(n_scan))
    dist = np.linalg.norm(prop(c0, t)[:, :3] - prop(d0, t)[:, :3], axis=1)
    i = int(np.argmin(dist))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]
    if hi <= lo:                                               # the scan is one point: keep it
        t_tca = float(t[i])
    else:
        f = lambda tt: float(np.linalg.norm(prop(c0, [tt])[0, :3] - prop(d0, [tt])[0, :3]))  # noqa: E731
        t_tca = float(minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4}).x)
    sc, sd = prop(c0, [t_tca])[0], prop(d0, [t_tca])[0]
    rho, rel_v = sc[:3] - sd[:3], sc[3:] - sd[3:]
    if velocity:
        Cc = propagate_covariance(c0, cov_chief, [t_tca], propagator=prop)[0]
        Cd = propagate_covariance(d0, cov_deputy, [t_tca], propagator=prop)[0]
    else:
        Cc, Cd = np.asarray(cov_chief, dtype=float), np.asarray(cov_deputy, dtype=float)
    xm, zm, sx, sz, _ = encounter_plane(rho, rel_v, Cc + Cd)
    pc = collision_probability_2d(xm, zm, sx, sz, hard_body_radius_km)
    return {"t_tca_s": t_tca, "miss_distance_km": float(np.linalg.norm(rho)),
            "relative_speed_km_s": float(np.linalg.norm(rel_v)), "hard_body_radius_km": hard_body_radius_km,
            "sigma_x_km": sx, "sigma_z_km": sz, "miss_x_km": xm, "miss_z_km": zm, "collision_probability": pc,
            "relative_covariance_propagated": bool(velocity),
            "method": "two-body TCA scan; 2-D Pc in the encounter plane with the summed propagated covariance "
                      "(Foster & Estes 1992, Alfano 2005)"}


# ----------------------------------------------------------------------------------------------- residuals
def regime_changes(t_days, a_km, *, model_km=None, k: float = 4.0, min_side: int = 8, max_events: int = 3,
                   max_splits: int = 200, min_separation_days: float | None = None,
                   min_magnitude: float = 0.0, min_edge_frac: float = 0.1) -> list[dict]:
    """The unexplained change in the drag/along-track residual: a **manoeuvre**, a step in the semi-major axis at one
    time, or a **drag regime**, a step in its rate that leaves ``a`` continuous.

    ``t_days`` and ``a_km``: the measured semi-major axis (from the propagator's states, or from the mean motion of a
    later TLE). ``model_km`` is what the quiet model says the axis should be, at the same times (interpolated): the
    residual is then ``a - model`` and a quiet arc is flat, which is the sensitive case. Without a model the series is
    taken as it is and only a linear trend is removed -- the curvature of a real decay over days then eats into the
    significance of a rate break, which is what the caller passes a model for.

    At every candidate split both a jump and a hinge are put in the same least-squares model and each is tested with
    its partial F statistic given the other (1 dof each, ``z = sqrt(F)``, signed by the coefficient); nothing below
    ``k`` sigma is flagged. Candidates closer than ``min_separation_days`` (a tenth of the span by default) to an
    event already reported are skipped: one physical event is one entry -- a burn changes the drag rate as well as
    the axis, and reporting the same break twice under two names would say nothing new. ``min_magnitude`` is the
    materiality of the caller in the units of the event (km for a jump, km/day for a rate): significance alone flags
    whatever is left once the main event is removed, however small. Candidates within ``min_edge_frac`` of either
    end are not considered: a break against a handful of points at the edge of the window fits the curvature of the
    decay and says nothing.
    """
    t = np.asarray(t_days, dtype=float)
    a = np.asarray(a_km, dtype=float)
    if t.size != a.size or t.size < 2 * int(min_side) + 2:
        raise ValueError("need at least 2*min_side + 2 samples")
    if model_km is None:
        residual = a.copy()
    else:
        m = np.asarray(model_km, dtype=float)
        if m.ndim == 2 and m.shape[0] == 2 and m.shape[1] != 2:
            m = m.T                                    # (t, a) passed as two arrays rather than as pairs
        if m.ndim == 2:
            residual = a - np.interp(t, m[:, 0], m[:, 1])
        elif m.size == a.size:
            residual = a - m
        else:
            raise ValueError("model_km must be the axis at t_days, or an (n, 2) array of (t_days, a_km)")
    tc = t - t.mean()
    base = [np.ones_like(tc)] if model_km is not None else [np.ones_like(tc), tc, tc**2]
    events = []
    span = float(t.max() - t.min())
    sep = 0.1 * span if min_separation_days is None else float(min_separation_days)
    lo_t, hi_t = t.min() + float(min_edge_frac) * span, t.max() - float(min_edge_frac) * span
    inside = np.nonzero((t >= lo_t) & (t <= hi_t))[0]
    inside = inside[(inside >= int(min_side)) & (inside <= len(t) - int(min_side) - 1)]
    if inside.size == 0:
        return []
    n_cand = min(int(max_splits), inside.size)
    candidates = inside[np.unique(np.linspace(0, inside.size - 1, max(n_cand, 1)).astype(int))]
    floor_rss = (1e-9 * float(np.max(np.abs(a)))) ** 2 * len(a)
    for _ in range(int(max_events)):
        best = None
        for i in candidates:
            d = t[i]
            if any(abs(d - e["t_days"]) <= sep for e in events):
                continue
            jump = (t >= d).astype(float)
            hinge = np.maximum(t - d, 0.0)
            full = np.column_stack([*base, jump, hinge])
            coef, *_ = np.linalg.lstsq(full, residual, rcond=None)
            rss_full = float(np.sum((residual - full @ coef) ** 2))
            dof = max(len(t) - full.shape[1], 1)
            if rss_full <= floor_rss:                    # the residual is at the floor: nothing to explain
                continue
            n_jump, n_hinge = full.shape[1] - 2, full.shape[1] - 1
            for kind, col in (("manoeuvre", n_jump), ("drag", n_hinge)):
                sub = np.delete(full, col, axis=1)
                c2, *_ = np.linalg.lstsq(sub, residual, rcond=None)
                rss_sub = float(np.sum((residual - sub @ c2) ** 2))
                z = math.copysign(math.sqrt(max((rss_sub - rss_full) / (rss_full / dof), 0.0)), coef[col])
                if abs(coef[col]) < abs(float(min_magnitude)):
                    continue
                if best is None or abs(z) > abs(best[3]):
                    best = (kind, float(d), float(coef[col]), z, int(i), coef)
        if best is None or abs(best[3]) <= float(k):
            break
        kind, when, mag, z, i, coef = best
        events.append({"kind": kind, "t_days": when, "magnitude": mag, "z": z,
                       "magnitude_unit": "km" if kind == "manoeuvre" else "km/day", "index": i})
        hinge = np.maximum(t - when, 0.0)
        jump = (t >= when).astype(float)
        residual = residual - (coef[-2] * jump if kind == "manoeuvre" else coef[-1] * hinge)
    return sorted(events, key=lambda e: e["t_days"])


def reentry_warning(h0_km: float, B_m2_kg: float, *, storm_factor: float = 1.0, days: float = 60.0,
                    n: int = 600, reentry_km: float = 150.0, base_density: Callable | None = None) -> dict:
    """A storm warning for a satellite near re-entry: the remaining lifetime from the decay ODE of :mod:`.decay`
    with the density as it is, and with it multiplied by ``storm_factor``.

    The lifetime is censored at ``days`` (``censored_*`` says so: a satellite that survives the horizon has no
    measured lifetime in it), so ``shortening_fraction`` is only quoted when the quiet case actually re-enters; the
    altitude loss over the horizon is reported either way. The warning fires when the storm brings the re-entry
    inside a horizon in which the quiet atmosphere would not have (or removes more than a quarter of the life left):
    the regime in which a satellite in a low parking orbit cannot raise itself in time.
    """
    base = base_density or default_density

    def life(factor):
        t, h = decay.reference_decay(h0_km, B_m2_kg, days, StormDensity(factor, base), n=n, reentry_km=reentry_km)
        ok = np.isfinite(h)
        if ok.all():
            return float(days), True, t, h
        k = int(np.argmin(ok))
        k = len(h) - 1 if not ok.any() else max(k - 1, 0)
        return float(t[k]), False, t, h

    l_quiet, cens_q, t_q, h_q = life(1.0)
    l_storm, cens_s, t_s, h_s = life(float(storm_factor))
    short = None if cens_q else (0.0 if l_quiet <= 0 else 1.0 - l_storm / l_quiet)
    fires = (not cens_s) and (cens_q or (short is not None and short > 0.25))
    if fires:
        reason = ("the storm brings the re-entry inside the horizon"
                  if cens_q else "the storm removes more than 25 % of the life left")
    else:
        reason = "no storm-driven re-entry inside the horizon"
    loss_q = float(h_q[0] - np.nanmin(h_q)) if np.isfinite(h_q).any() else float("nan")
    loss_s = float(h_s[0] - np.nanmin(h_s)) if np.isfinite(h_s).any() else float("nan")
    return {"h0_km": float(h0_km), "B_m2_kg": float(B_m2_kg), "storm_factor": float(storm_factor),
            "lifetime_days_quiet": l_quiet, "lifetime_days_storm": l_storm,
            "censored_quiet": cens_q, "censored_storm": cens_s, "horizon_days": float(days),
            "shortening_fraction": short,
            "altitude_loss_km_quiet": loss_q, "altitude_loss_km_storm": loss_s,
            "warning": bool(fires), "reason": reason, "density_model": density_model_name(),
            "t_days": t_q, "h_km_quiet": h_q, "h_km_storm": h_s}


# ----------------------------------------------------------------------------------------------- learned correction
def correction_features(t_ahead_s, bstar: float = 3.0e-5) -> np.ndarray:
    """Rows ``[tau, tau^2, (B*/1e-4) tau, ln(1+tau)]`` with ``tau`` in days: a drag-mismatch along-track error grows
    like the square of the lead time (a wrong ``da/dt`` is a wrong ``a`` after ``t``, hence a wrong mean motion and a
    drift that integrates once more), the fitted ``B*`` scales that growth, and the log term absorbs the short-time
    behaviour. Every feature vanishes at zero lead time, so a correction fitted on these columns leaves the epoch --
    where a TLE is exact by construction -- where it is.
    """
    tau = np.atleast_1d(np.asarray(t_ahead_s, dtype=float)) / 86400.0
    b = float(bstar) / 1e-4
    return np.column_stack([tau, tau**2, b * tau, np.log1p(tau)])


class LearnedTleCorrection:
    """The learned correction to a TLE propagation, in the radial / along-track / cross-track frame of the propagated
    orbit. Fitted by ridge regression on residuals whose truth is a **later TLE** (:func:`tle_pair_residuals`), so
    what it learns is the systematic part of the SGP4 error -- the drag term that the fitted ``B*`` gets wrong.

    ``fit(t_s, rtn_km, bstar=...)`` where ``rtn_km`` is ``(n, 3)``; ``predict(t_s, bstar=...)`` the same shape;
    ``apply(states, t_s, bstar=...)`` moves a propagated trajectory by it. There is no free intercept and every
    feature is zero at ``t = 0``, so the correction vanishes at the epoch instead of shifting a whole arc; the ridge
    penalty acts on the standardised columns.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge, self.coef = float(ridge), None

    def fit(self, t_s, rtn_km, *, bstar: float = 3.0e-5) -> LearnedTleCorrection:
        X = correction_features(t_s, bstar)
        Y = np.atleast_2d(np.asarray(rtn_km, dtype=float))
        if Y.shape[0] != X.shape[0]:
            Y = Y.T
        # ridge scaled by each column's own energy, and no centring: the columns are kept in their physical units, so
        # the fitted correction is exactly zero at the epoch
        A = X.T @ X + self.ridge * np.diag(np.maximum(np.sum(X**2, axis=0), 1e-30))
        self.coef = np.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, t_s, *, bstar: float = 3.0e-5) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit() first: an unfitted correction has no coefficients")
        return correction_features(t_s, bstar) @ self.coef

    def apply(self, states, t_s, *, bstar: float = 3.0e-5) -> np.ndarray:
        """Correct a propagated trajectory ``(n, 6)`` (km, km/s) by the learned RTN offset."""
        S = np.atleast_2d(np.asarray(states, dtype=float)).copy()
        d = self.predict(t_s, bstar=bstar)
        for k in range(len(S)):
            F = rtn_frame(S[k, :3], S[k, 3:])
            S[k, :3] = S[k, :3] + d[k] @ F
        return S


def tle_pair_residuals(tles, *, rtn=True) -> dict:
    """The training set of :class:`LearnedTleCorrection` from a chronological list of TLEs of one object.

    For each consecutive pair, the earlier TLE is propagated to the epoch of the later one with SGP4 and the
    difference is recorded in the RTN frame of the propagated orbit: the error the propagation made, with the later
    TLE as truth (a TLE is itself a fit accurate to ~1 km, which is the floor of this target). ``tles``: ``(name,
    line1, line2)`` triples or objects with ``line1``/``line2``. Returns ``{"t_s", "rtn_km", "bstar", "lead_days"}``.

    This needs a TLE archive: CelesTrak's GP data keeps only the current elements, so a training set has to be
    collected object by object over time (or taken from the archived GP history), which is the network part of this
    module and is not run in the test suite.
    """
    rows = []
    for a, b in zip(tles[:-1], tles[1:], strict=True):
        l1a, l2a = (a.line1, a.line2) if hasattr(a, "line1") else (a[1], a[2])
        l1b, l2b = (b.line1, b.line2) if hasattr(b, "line1") else (b[1], b[2])
        sa = Satellite("tle", l1a, l2a)
        sb = Satellite("tle", l1b, l2b)
        lead_s = (sb.epoch_jd - sa.epoch_jd) * 86400.0
        if lead_s <= 0:
            continue
        r, v, err = sa.teme(np.array([sa.epoch_jd + lead_s / 86400.0]))
        truth, tv, terr = sb.teme(np.array([sb.epoch_jd]))
        if int(err[0]) or int(terr[0]):
            continue
        F = rtn_frame(r[0], v[0])
        rows.append({"t_s": lead_s, "rtn_km": F @ (truth[0] - r[0]), "bstar": float(sa.satrec.bstar),
                     "lead_days": lead_s / 86400.0})
    if not rows:
        return {"t_s": np.zeros(0), "rtn_km": np.zeros((0, 3)), "bstar": 0.0, "lead_days": np.zeros(0)}
    return {"t_s": np.array([r["t_s"] for r in rows]), "rtn_km": np.array([r["rtn_km"] for r in rows]),
            "bstar": float(np.mean([r["bstar"] for r in rows])),
            "lead_days": np.array([r["lead_days"] for r in rows])}


def residual_by_lead_time(errors_km, lead_hours, model_km) -> dict:
    """The residual the trust side needs: the measured error against the error the model expected, by lead time.
    ``model_km`` is the sigma the covariance predicted at the same lead times. A ratio above 1 is a propagation that
    is worse than it says it is, and the ratio by lead time is what the manoeuvre and storm flags read."""
    e = np.asarray(errors_km, dtype=float)
    m = np.maximum(np.asarray(model_km, dtype=float), 1e-12)
    return {"lead_hours": np.asarray(lead_hours, dtype=float), "error_km": e, "model_km": m, "ratio": e / m,
            "max_ratio": float(np.max(e / m)), "median_ratio": float(np.median(e / m))}


def orbit_residual(prediction, query) -> float:
    """Physics residual of a predicted trajectory, for :class:`PhysicsEnsemble`: how badly the predicted positions
    ``(n, 3)`` at ``query["t_s"]`` satisfy the two-body equation, by a fourth-order finite difference of the second
    derivative, relative to the two-body acceleration. Label-free: it needs no reference, only the prediction."""
    r = np.atleast_2d(np.asarray(prediction, dtype=float))[:, :3]
    t = np.atleast_1d(np.asarray(query["t_s"], dtype=float))
    if len(r) < 5 or len(r) != len(t):
        return float("nan")
    h = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), h, rtol=1e-6):
        return float("nan")
    acc = (-r[4:] + 16.0 * r[3:-1] - 30.0 * r[2:-2] + 16.0 * r[1:-3] - r[:-4]) / (12.0 * h * h)
    ref = -MU_EARTH * r[2:-2] / np.linalg.norm(r[2:-2], axis=1, keepdims=True) ** 3
    return float(np.sqrt(np.mean(np.sum((acc - ref) ** 2, axis=1)) / np.mean(np.sum(ref**2, axis=1))))


def traffic_ensemble(correction: LearnedTleCorrection | None = None, *, B: float | None = None,
                     density: Callable | None = None, **ensemble_kw):
    """The three witnesses of one orbit in a :class:`pinneapple_physics.ensemble.PhysicsEnsemble`: SGP4 from the TLE,
    the numerical propagator seeded from the same osculating state with drag, and SGP4 plus the learned correction.

    Query: ``{"tle": (line1, line2), "t_s": array of seconds after the TLE epoch}``, plus optional ``"B"`` and
    ``"storm"`` (a multiplier on the density, as a callable of time). Every expert returns positions ``(n, 3)``;
    ``residual_fn=orbit_residual`` scores them with no reference at all, and giving ``references`` to ``run()`` turns
    the run into a supervised one. ``measurement_check`` is :func:`tle_measurement_check`: a TLE that does not hold
    together makes the ensemble predict with its learned weights and learn nothing from the case. The learned expert
    is only offered when a fitted correction is given, so nothing is claimed about an unfitted model.
    """
    from ..ensemble import Expert, PhysicsEnsemble

    dens = density or default_density

    def seed(query):
        r, v, err = sgp4_states(*query["tle"], [0.0])
        if int(err[0]):
            raise ValueError(f"SGP4 refused the element set (code {int(err[0])})")
        return np.concatenate([r[0], v[0]])

    def expert_sgp4(q):
        return sgp4_states(*q["tle"], q["t_s"])[0]

    def expert_numerical(q):
        x0 = seed(q)
        drag = DragModel(B if B is not None else decay.ballistic_from_bstar(Satellite("tle", *q["tle"]).satrec.bstar),
                         dens, q.get("storm"))
        return propagate(x0, q["t_s"], drag=drag)[:, :3]

    experts = [Expert("sgp4", expert_sgp4, cost=1.0, kind="analytic"),
               Expert("numerical", expert_numerical, cost=10.0, kind="numerical")]
    if correction is not None:
        def expert_corrected(q):
            r, v, _ = sgp4_states(*q["tle"], q["t_s"])
            bstar = float(Satellite("tle", *q["tle"]).satrec.bstar)
            S = correction.apply(np.concatenate([r, v], axis=1), q["t_s"], bstar=bstar)
            return S[:, :3]
        experts.append(Expert("sgp4+correction", expert_corrected, cost=2.0, kind="hybrid"))
    kw = {"residual_fn": orbit_residual, "residual_lookahead": 1.0, "mode": "combine",
          "measurement_check": lambda q: tle_measurement_check(*q["tle"])}
    kw.update(ensemble_kw)
    return PhysicsEnsemble(experts, **kw)


STARLINK_FEB_2022 = {
    "event": "geomagnetic storm of 2022-02-04 and the Starlink satellites of the 2022-02-03 launch",
    "reported": {
        "launched": 49, "lost": 38, "date_launch": "2022-02-03", "date_storm": "2022-02-04",
        "deployment_altitude_km": 210, "target_altitude_km": 340,
        "source": "SpaceX update of 2022-02-08: the storm warmed the atmosphere, density at the deployment "
                  "altitude rose by up to 50 %, and the drag rose with it; the satellites could not raise their "
                  "orbit out of the low regime and re-entered",
    },
    "needed_to_reproduce": {
        "tles": "CelesTrak GP history for the 49 objects, 2022-02-03 .. 2022-02-15 (two or more element sets per "
                "object per day): the along-track decay rate before and after the storm, object by object",
        "space_weather": "3-hourly Ap and daily F10.7 for 2022-02-01 .. 2022-02-15 (GFZ Potsdam / NOAA SWPC / "
                         "CelesTrak), the drivers of NRLMSISE-00 or JB2008",
        "ballistic_coefficient": "B = C_D A / m for a Starlink v1.5 in its low-drag attitude: a mass near 260 kg "
                                 "and the deployed cross-section, to be taken from the operator's data, not assumed "
                                 "here",
        "what_this_module_gives": "propagate() with a storm() factor on the density, regime_changes() on the "
                                  "measured semi-major axis to time the drag change, and reentry_warning() for the "
                                  "lifetime against the storm",
    },
    "declared_not_reproduced": "the event itself: no TLE archive and no space-weather indices are in this "
                               "repository, so what the tests measure is the mechanism (a density jump and the "
                               "decay and lifetime it produces at 210 km), not the loss of these satellites",
}
