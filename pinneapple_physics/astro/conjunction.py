"""Conjunction screening: closest approach in Clohessy-Wiltshire relative motion, collision probability, J2 orbits.

* :func:`cw_closest_approach` -- time and distance of closest approach of a deputy to its chief, from the exact
  Clohessy-Wiltshire solution (``presets.astrophysics._cw_exact_state`` / :func:`cw_exact_velocity`): a dense
  scan brackets the minimum of |rho(t)|^2, a bounded scalar minimiser refines it, and the residual
  ``|rho . v| / (|rho| |v|)`` (zero at a true closest approach) is reported.
* :func:`collision_probability_2d` -- the 2-D Gaussian collision probability in the encounter plane (Foster &
  Estes 1992; Alfano 2005): the mass of N((xm, zm), diag(sx^2, sz^2)) inside the hard-body disk of radius r_c
  centred at the origin, by adaptive quadrature. :func:`collision_probability_centered` is the closed form for a
  centred isotropic Gaussian, 1 - exp(-r_c^2 / (2 sigma^2)) (Chan 2008), used to check the quadrature.
* :func:`screen_cw_conjunction` -- the two together, with the assumptions written into the result.
* :func:`two_body_j2_rhs` / :func:`propagate_two_body_j2` -- two-body + J2 acceleration
  a = (3/2) J2 mu Re^2 / r^5 [x(5 z^2/r^2 - 1), y(5 z^2/r^2 - 1), z(5 z^2/r^2 - 3)] (Vallado 2013, ch. 9), the
  same model as the ``satellite_j2_perturbation`` preset, integrated with DOP853.

Scope, as for the source app (PINNeAPPle-apps ``satellite_conjunction_screening``): CW is linear relative motion
about a circular reference orbit, valid for short windows (no drag, J2, third bodies or manoeuvres); the deputy is
given as a relative state in the chief's Hill frame (x radial, y along-track, z cross-track); the collision
probability uses a caller-supplied isotropic combined sigma, not a propagated covariance from orbit
determination, so it illustrates the method and is not an operational Pc. Large sigma relative to the miss
distance is the regime where this 2-D Pc is known to be hard to interpret (Elkantassi & Davison 2022).

References: Clohessy & Wiltshire, J. Aerospace Sci. 27(9), 1960; Foster & Estes, NASA JSC-25898, 1992;
Alfano, AAS 05-354, 2005; Chan, Spacecraft Collision Probability, AIAA, 2008; Vallado, Fundamentals of
Astrodynamics and Applications, 4th ed., 2013.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["ClosestApproach", "cw_closest_approach", "collision_probability_2d", "collision_probability_centered",
           "screen_cw_conjunction", "two_body_j2_rhs", "propagate_two_body_j2", "cw_mean_motion"]


def cw_mean_motion(a_km: float, mu_km3_s2: float = 398600.4418) -> float:
    """Mean motion n = sqrt(mu / a^3) of the (circular) reference orbit, rad/s."""
    return math.sqrt(mu_km3_s2 / a_km ** 3)


@dataclass
class ClosestApproach:
    t_tca_s: float
    position_km: list           # relative position at TCA (Hill frame)
    velocity_km_s: list         # relative velocity at TCA
    miss_distance_km: float
    relative_speed_km_s: float
    perpendicularity_residual: float   # |rho . v| / (|rho| |v|); 0 at a true closest approach

    def to_dict(self) -> dict:
        return asdict(self)


def cw_closest_approach(n: float, state0, duration_s: float, n_samples: int = 2000) -> ClosestApproach:
    """Closest approach over [0, duration_s] for the CW deputy state ``state0 = (x0, y0, z0, vx0, vy0, vz0)``
    (km, km/s) about a reference orbit of mean motion ``n`` (rad/s)."""
    from scipy.optimize import minimize_scalar

    from pinneapple_physics.pde_environment.presets.astrophysics import (
        _cw_exact_state,
        cw_exact_velocity,
    )

    if n <= 0 or duration_s <= 0 or n_samples < 3:
        raise ValueError("need n > 0, duration_s > 0 and n_samples >= 3")
    s0 = tuple(float(v) for v in state0)
    t = np.linspace(0.0, float(duration_s), int(n_samples))
    p = _cw_exact_state(t, n, *s0)
    d2 = p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2
    i = int(np.argmin(d2))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]

    def dist2(tt):
        q = _cw_exact_state(np.array([tt]), n, *s0)
        return float(q["x"][0] ** 2 + q["y"][0] ** 2 + q["z"][0] ** 2)

    t_tca = float(minimize_scalar(dist2, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}).x)
    q = _cw_exact_state(np.array([t_tca]), n, *s0)
    v = cw_exact_velocity(np.array([t_tca]), n, *s0)
    rho = np.array([q["x"][0], q["y"][0], q["z"][0]])
    vel = np.array([v["vx"][0], v["vy"][0], v["vz"][0]])
    miss, speed = float(np.linalg.norm(rho)), float(np.linalg.norm(vel))
    perp = float(abs(rho @ vel) / max(miss * speed, 1e-30))
    return ClosestApproach(t_tca, rho.tolist(), vel.tolist(), miss, speed, perp)


def collision_probability_2d(xm: float, zm: float, sigma_x: float, sigma_z: float, r_c: float) -> float:
    """Probability mass of N((xm, zm), diag(sigma_x^2, sigma_z^2)) inside the disk x^2 + z^2 <= r_c^2."""
    from scipy import integrate

    if r_c <= 0:
        return 0.0
    if sigma_x <= 0 or sigma_z <= 0:
        raise ValueError("sigma_x and sigma_z must be positive")
    norm = 1.0 / (2.0 * math.pi * sigma_x * sigma_z)

    def f(z, x):
        return norm * math.exp(-0.5 * (((x - xm) / sigma_x) ** 2 + ((z - zm) / sigma_z) ** 2))

    pc, _ = integrate.dblquad(f, -r_c, r_c, lambda x: -math.sqrt(max(r_c * r_c - x * x, 0.0)),
                              lambda x: math.sqrt(max(r_c * r_c - x * x, 0.0)), epsabs=1e-13, epsrel=1e-10)
    return float(min(max(pc, 0.0), 1.0))


def collision_probability_centered(r_c: float, sigma: float) -> float:
    """Closed form for a centred isotropic Gaussian: 1 - exp(-r_c^2 / (2 sigma^2)) (Rayleigh CDF)."""
    return float(-math.expm1(-(r_c ** 2) / (2.0 * sigma ** 2)))


def screen_cw_conjunction(n: float, state0, *, duration_s: float | None = None, n_samples: int = 2000,
                          hard_body_radius_km: float = 0.010, sigma_km: float = 1.0) -> dict:
    """Closest approach + collision probability for one CW encounter (one reference period by default).

    At the closest approach the relative position lies in the encounter plane (perpendicular to the relative
    velocity), so with an isotropic sigma the whole miss distance can be put on one in-plane axis.
    """
    period = 2.0 * math.pi / n
    duration = float(duration_s if duration_s is not None else period)
    ca = cw_closest_approach(n, state0, duration, n_samples)
    pc = collision_probability_2d(ca.miss_distance_km, 0.0, sigma_km, sigma_km, hard_body_radius_km)
    warnings = []
    if ca.perpendicularity_residual > 1e-3:
        warnings.append("closest approach sits at the edge of the window (rho . v != 0): the minimum may lie outside it")
    if sigma_km > ca.miss_distance_km:
        warnings.append("sigma is larger than the miss distance: the 2-D Pc is hard to interpret in this regime")
    return {"reference_period_s": period, "window_s": duration, "closest_approach": ca.to_dict(),
            "hard_body_radius_km": hard_body_radius_km, "hard_body_violated": ca.miss_distance_km < hard_body_radius_km,
            "sigma_km": sigma_km, "collision_probability": pc,
            "method": "Clohessy-Wiltshire closed form; 2-D Gaussian Pc in the encounter plane "
                      "(Foster & Estes 1992, Alfano 2005), isotropic caller-supplied sigma",
            "warnings": warnings}


def two_body_j2_rhs(mu: float, J2: float, Re: float):
    """Right-hand side f(t, [x, y, z, vx, vy, vz]) of two-body motion plus the J2 acceleration."""
    def rhs(t, s):
        x, y, z, vx, vy, vz = s
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        k = 1.5 * J2 * mu * Re * Re / (r2 * r2 * r)
        w = 5.0 * z * z / r2
        g = -mu / (r2 * r)
        return [vx, vy, vz, g * x + k * x * (w - 1.0), g * y + k * y * (w - 1.0), g * z + k * z * (w - 3.0)]
    return rhs


def propagate_two_body_j2(state0, t_eval, *, mu: float = 398600.4418, J2: float = 1.08262668e-3,
                          Re: float = 6378.137, rtol: float = 1e-11, atol: float = 1e-8) -> np.ndarray:
    """States (len(t_eval), 6) of two-body + J2 motion from ``state0`` (km, km/s), DOP853."""
    from scipy.integrate import solve_ivp

    t_eval = np.asarray(t_eval, dtype=np.float64)
    sol = solve_ivp(two_body_j2_rhs(mu, J2, Re), [0.0, float(t_eval[-1])], list(map(float, state0)),
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"J2 propagation failed: {sol.message}")
    return sol.y.T
