"""Point-mass ballistics over a flat Earth: gravity and drag in an exponential atmosphere.

State ``s = [x, y, z, vx, vy, vz]`` in m and m/s, ``z`` up. Drag deceleration ``rho(z) |v| v / (2 beta)`` with the
ballistic coefficient ``beta = m / (C_D A)`` in kg/m^2 (a larger ``beta`` flies further). Density
``rho(z) = rho0 exp(-z / H)`` with ``rho0 = 1.225 kg/m^3`` and ``H = 7200 m`` (the exponential atmosphere of
Zarchan, *Tactical and Strategic Missile Guidance*; the library's ISA function stops at the tropopause, and these
trajectories go higher).

Textbook model with illustrative numbers; simulation only.
"""
from __future__ import annotations

import math

import numpy as np

G = 9.80665
RHO0 = 1.225
H_SCALE = 7200.0


def density(z):
    """Air density of the exponential atmosphere.

    Parameters
    ----------
    z : float or array [m]
        Altitude (negative values are clipped to 0).

    Returns
    -------
    rho : float or array [kg/m3]
        Density.
    """
    return RHO0 * np.exp(-np.maximum(z, 0.0) / H_SCALE)


def ballistic_rhs(s, beta: float):
    """Time derivative of the state ``[x, y, z, vx, vy, vz]`` under gravity and drag.

    Parameters
    ----------
    s : array (6,)
        State in m and m/s.
    beta : float [kg/m2]
        Ballistic coefficient (``inf`` for no drag).

    Returns
    -------
    ds : array (6,)
        ``[vx, vy, vz, ax, ay, az]`` in m/s and m/s^2.
    """
    s = np.asarray(s, dtype=float)
    v = s[3:6]
    a = np.array([0.0, 0.0, -G])
    if np.isfinite(beta):
        a = a - density(s[2]) * np.linalg.norm(v) * v / (2.0 * beta)
    return np.concatenate([v, a])


def rk4_step(f, s, dt: float):
    """One classical Runge-Kutta step of ``ds/dt = f(s)``."""
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def launch_velocity(speed: float, elevation_deg: float, azimuth_deg: float):
    """Velocity vector from speed, elevation above the horizon and azimuth (from +x towards +y).

    Parameters
    ----------
    speed : float [m/s]
    elevation_deg : float [deg]
    azimuth_deg : float [deg]

    Returns
    -------
    v : array (3,) [m/s]
    """
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return speed * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def fly_ballistic(p0, v0, beta: float, *, dt: float = 0.05, t_max: float = 600.0, t0: float = 0.0):
    """Integrate a ballistic trajectory until it reaches the ground (``z = 0``).

    Parameters
    ----------
    p0 : array (3,) [m]
        Initial position.
    v0 : array (3,) [m/s]
        Initial velocity.
    beta : float [kg/m2]
        Ballistic coefficient (``inf`` for vacuum).
    dt : float [s]
        Time step.
    t_max : float [s]
        Longest flight integrated.
    t0 : float [s]
        Time of the initial state.

    Returns
    -------
    trajectory : dict
        ``t`` (N,) [s], ``pos`` (N, 3) [m], ``vel`` (N, 3) [m/s]; ``impact`` (3,) [m] and ``t_impact`` [s], the
        ground crossing interpolated inside the last step (``None`` when ``t_max`` is reached first); ``apex`` [m].
    """
    s = np.concatenate([np.asarray(p0, float), np.asarray(v0, float)])
    ts, ss = [t0], [s]
    t = t0
    impact = t_imp = None
    while t < t0 + t_max:
        s_new = rk4_step(lambda x: ballistic_rhs(x, beta), s, dt)
        t += dt
        if s_new[2] < 0.0 <= s[2] and t > t0 + dt * 0.5:
            w = s[2] / (s[2] - s_new[2])
            impact = s[:3] + w * (s_new[:3] - s[:3])
            t_imp = t - dt + w * dt
            s_new = np.concatenate([impact, s[3:] + w * (s_new[3:] - s[3:])])
            ts.append(t_imp)
            ss.append(s_new)
            break
        s = s_new
        ts.append(t)
        ss.append(s)
    S = np.array(ss)
    return {"t": np.array(ts), "pos": S[:, :3], "vel": S[:, 3:], "impact": impact, "t_impact": t_imp,
            "apex": float(S[:, 2].max())}


def vacuum_range(speed: float, elevation_deg: float) -> float:
    """Range of a vacuum trajectory from and to the ground, ``v^2 sin(2 theta) / g``.

    Returns
    -------
    R : float [m]
    """
    return speed ** 2 * math.sin(2 * math.radians(elevation_deg)) / G


def vacuum_apex(speed: float, elevation_deg: float) -> float:
    """Apex height of a vacuum trajectory launched from the ground, ``v^2 sin^2(theta) / (2 g)``.

    Returns
    -------
    h : float [m]
    """
    return (speed * math.sin(math.radians(elevation_deg))) ** 2 / (2 * G)
