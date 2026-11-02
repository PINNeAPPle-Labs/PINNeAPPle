"""Charged particles in electric and magnetic fields: the relativistic Boris pusher, field models, closed-form
references and ready-made studies (issue #1106).

The integrator is the Boris scheme (Boris 1970; relativistic form as in Birdsall & Langdon, *Plasma Physics via
Computer Simulation*). It advances ``u = gamma v``:

    u- = u + (q E / m) dt/2                      half electric kick
    t  = (q B / (m gamma-)) dt/2,  s = 2 t / (1 + |t|^2)
    u' = u- + u- x t,  u+ = u- + u' x s          exact-norm magnetic rotation
    u  = u+ + (q E / m) dt/2                     half electric kick
    x += u / gamma dt

It is second-order accurate and time-reversible, and in a pure magnetic field it conserves |u| (the kinetic energy)
to round-off for any step: the rotation preserves the norm. That is why it is the standard particle pusher in
plasma codes.

    from pinneapple_simulation.electromagnetics import particles as P
    out = P.push(x0=[[0, 0, 0]], v0=[[1e5, 0, 0]], q=P.Q_E, m=P.M_P,
                 B=P.uniform((0, 0, 1e-3)), dt=1e-8, n_steps=4000)
    out["x"]          # (n_saved, n_particles, 3)

Field arguments are callables ``f(x, t) -> (N, 3)`` in SI (V/m, T). ``uniform``, ``magnetic_dipole``,
``magnetic_bottle`` and ``penning_trap`` build the common ones, and the Biot-Savart routines of
:mod:`.statics` can be wrapped the same way.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .constants import C0, M_E, M_P, MU0, Q_E

__all__ = [
    "Q_E", "M_E", "M_P", "C0",
    "push", "uniform", "magnetic_dipole", "magnetic_bottle", "penning_trap", "add_fields",
    "lorentz_factor", "larmor_radius", "cyclotron_frequency", "exb_drift", "gradb_drift", "loss_cone_angle",
    "penning_frequencies", "guiding_center", "magnetic_moment", "pitch_velocity",
    "cyclotron_study", "drift_study", "magnetic_bottle_study", "dipole_trap_study", "penning_study",
    "mass_spectrometer_study", "dipole_field_line",
]

Field = Callable[[np.ndarray, float], np.ndarray]
EARTH_RADIUS = 6.371e6           # m
EARTH_DIPOLE_MOMENT = 8.0e22     # A m^2 (approximate present-day value; illustrative)


# ------------------------------------------------------------------------------------------------- fields
def uniform(vector: Sequence[float]) -> Field:
    """A uniform, static field (V/m for E, T for B)."""
    v = np.asarray(vector, float).reshape(1, 3)
    return lambda x, t: np.broadcast_to(v, np.shape(x)).copy()


def add_fields(*fields: Field) -> Field:
    """The sum of several field callables."""
    return lambda x, t: sum(f(x, t) for f in fields)


def magnetic_dipole(moment: Sequence[float] = (0.0, 0.0, -EARTH_DIPOLE_MOMENT),
                    center: Sequence[float] = (0.0, 0.0, 0.0)) -> Field:
    """Field of a point magnetic dipole, B = mu0/(4 pi) (3 r_hat (m . r_hat) - m) / r^3 [T].

    The default is an Earth-like dipole: the moment points to geographic south, so B points north at the equator."""
    m = np.asarray(moment, float)
    c = np.asarray(center, float)

    def B(x, t):
        r = np.asarray(x, float) - c
        rn = np.linalg.norm(r, axis=-1, keepdims=True)
        rh = r / rn
        return MU0 / (4 * math.pi) * (3 * rh * (rh @ m)[..., None] - m) / rn**3

    return B


def magnetic_bottle(B0: float, L: float) -> Field:
    """Axisymmetric magnetic bottle (paraxial, divergence-free): Bz = B0 (1 + (z/L)^2), B_rho = -rho B0 z / L^2.

    The field is weakest at z = 0 and grows to the mirror ratio R = Bz(z)/B0 = 1 + (z/L)^2."""
    def B(x, t):
        x = np.asarray(x, float)
        z = x[..., 2]
        bz = B0 * (1 + (z / L) ** 2)
        k = -B0 * z / L**2                 # B_rho / rho  (from div B = 0: B_rho = -(rho/2) dBz/dz)
        return np.stack([k * x[..., 0], k * x[..., 1], bz], -1)

    return B


def penning_trap(V0: float, d: float) -> Field:
    """Electric field of an ideal Penning trap, Phi = V0 (2 z^2 - x^2 - y^2) / (4 d^2): E = V0/(2 d^2) (x, y, -2 z).

    It confines a positive charge axially when V0 > 0. Use it with a uniform axial B."""
    def E(x, t):
        x = np.asarray(x, float)
        k = V0 / (2 * d**2)
        return np.stack([k * x[..., 0], k * x[..., 1], -2 * k * x[..., 2]], -1)

    return E


def dipole_field_line(L_shell: float, n: int = 200, longitude_deg: float = 0.0, radius: float = EARTH_RADIUS
                      ) -> np.ndarray:
    """Points (n, 3) [m] of the dipole field line r = L R cos^2(latitude) between its two footpoints at r = R."""
    lam_max = math.acos(math.sqrt(1.0 / L_shell))
    lam = np.linspace(-lam_max, lam_max, n)
    r = L_shell * radius * np.cos(lam) ** 2
    ph = math.radians(longitude_deg)
    return np.stack([r * np.cos(lam) * math.cos(ph), r * np.cos(lam) * math.sin(ph), r * np.sin(lam)], 1)


# ------------------------------------------------------------------------------------------------- integrator
def push(x0: Sequence[Sequence[float]], v0: Sequence[Sequence[float]], q: float, m: float, *,
         E: Field | None = None, B: Field | None = None, dt: float, n_steps: int, save_every: int = 1,
         relativistic: bool = True, t0: float = 0.0, stop: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
         ) -> dict[str, Any]:
    """Advance N particles with the (relativistic) Boris pusher.

    Parameters
    ----------
    x0, v0 : (N, 3) initial positions [m] and velocities [m/s].
    q, m : charge [C] and rest mass [kg], the same for all particles (call once per species).
    E, B : field callables ``f(x, t) -> (N, 3)`` [V/m], [T]; ``None`` is zero.
    dt : time step [s]; n_steps : number of steps; save_every : keep every n-th state.
    relativistic : advance u = gamma v (True) or v (False, the classic Boris scheme).
    stop : optional ``stop(x, v) -> (N,) bool``; a particle that meets it is frozen from then on (e.g. lost through a
        mirror, hit the ground). Its index and time are reported in ``stopped``/``t_stop``.

    Returns
    -------
    dict with ``t`` (n_saved,), ``x`` and ``v`` (n_saved, N, 3), ``gamma`` and ``kinetic_energy`` [J] (n_saved, N),
    ``stopped`` (N,) bool and ``t_stop`` (N,) [s] (nan when never stopped).
    """
    x = np.array(x0, float).reshape(-1, 3)
    v = np.array(v0, float).reshape(-1, 3)
    n = len(x)
    if np.any(np.linalg.norm(v, axis=1) >= C0):
        raise ValueError("initial speed must be below c")
    zero = lambda xx, tt: np.zeros_like(xx)  # noqa: E731
    E = E or zero
    B = B or zero
    c2 = C0 * C0
    if relativistic:
        g = 1.0 / np.sqrt(1.0 - (v * v).sum(1) / c2)
        u = v * g[:, None]
    else:
        u = v.copy()
    qm = q / m
    alive = np.ones(n, bool)
    t_stop = np.full(n, np.nan)
    ts, xs, vs, gs = [], [], [], []

    def gamma_of(uu):
        return np.sqrt(1.0 + (uu * uu).sum(1) / c2) if relativistic else np.ones(len(uu))

    def save(tt):
        gg = gamma_of(u)
        ts.append(tt)
        xs.append(x.copy())
        vs.append(u / gg[:, None])
        gs.append(gg)

    t = t0
    save(t)
    for k in range(1, n_steps + 1):
        Ef = E(x, t)
        Bf = B(x, t)
        um = u + qm * Ef * (0.5 * dt)
        gm = gamma_of(um)
        tv = qm * Bf * (0.5 * dt) / gm[:, None]
        sv = 2 * tv / (1 + (tv * tv).sum(1))[:, None]
        up = um + np.cross(um + np.cross(um, tv), sv)
        un = up + qm * Ef * (0.5 * dt)
        gn = gamma_of(un)
        xn = x + un / gn[:, None] * dt
        u = np.where(alive[:, None], un, u)
        x = np.where(alive[:, None], xn, x)
        t = t0 + k * dt
        if stop is not None:
            hit = alive & np.asarray(stop(x, u / gamma_of(u)[:, None]), bool)
            t_stop[hit] = t
            alive &= ~hit
        if k % save_every == 0:
            save(t)
        if not alive.any():
            if k % save_every:
                save(t)
            break
    G = np.array(gs)
    return {"t": np.array(ts), "x": np.array(xs), "v": np.array(vs), "gamma": G,
            "kinetic_energy": (G - 1.0) * m * c2 if relativistic else 0.5 * m * (np.array(vs) ** 2).sum(-1),
            "stopped": ~alive, "t_stop": t_stop}


# ------------------------------------------------------------------------------------------------- closed forms
def lorentz_factor(speed: float) -> float:
    """gamma = 1 / sqrt(1 - v^2/c^2)."""
    return 1.0 / math.sqrt(1.0 - (speed / C0) ** 2)


def cyclotron_frequency(q: float, m: float, B: float, speed: float = 0.0) -> float:
    """Angular gyrofrequency |q| B / (gamma m) [rad/s] (gamma from the speed; 0 gives the non-relativistic value)."""
    return abs(q) * B / (lorentz_factor(speed) * m)


def larmor_radius(q: float, m: float, B: float, v_perp: float, speed: float | None = None) -> float:
    """Gyroradius gamma m v_perp / (|q| B) [m]; ``speed`` (default v_perp) sets gamma."""
    return lorentz_factor(v_perp if speed is None else speed) * m * v_perp / (abs(q) * B)


def exb_drift(E: Sequence[float], B: Sequence[float]) -> np.ndarray:
    """E x B drift velocity E x B / |B|^2 [m/s] (independent of charge and mass)."""
    E, B = np.asarray(E, float), np.asarray(B, float)
    return np.cross(E, B) / (B @ B)


def gradb_drift(q: float, m: float, v_perp: float, B: Sequence[float], grad_B: Sequence[float]) -> np.ndarray:
    """Grad-B drift (m v_perp^2 / (2 q B^3)) B x grad|B| [m/s] (non-relativistic, guiding-centre limit)."""
    B, gB = np.asarray(B, float), np.asarray(grad_B, float)
    b = math.sqrt(B @ B)
    return m * v_perp**2 / (2 * q * b**3) * np.cross(B, gB)


def loss_cone_angle(B_min: float, B_max: float) -> float:
    """Loss-cone half-angle arcsin(sqrt(B_min / B_max)) [rad]: particles with a smaller pitch angle at B_min escape."""
    return math.asin(math.sqrt(B_min / B_max))


def penning_frequencies(q: float, m: float, B: float, V0: float, d: float) -> dict[str, float]:
    """Angular frequencies [rad/s] of an ideal Penning trap: cyclotron qB/m, axial sqrt(q V0/(m d^2)),
    modified cyclotron and magnetron w_pm = w_c/2 +- sqrt(w_c^2/4 - w_z^2/2) (trapped when w_c^2 > 2 w_z^2)."""
    wc = q * B / m
    wz = math.sqrt(q * V0 / (m * d * d))
    disc = wc * wc / 4 - wz * wz / 2
    if disc <= 0:
        raise ValueError("not confined radially: need (qB/m)^2 > 2 q V0/(m d^2)")
    r = math.sqrt(disc)
    return {"cyclotron": wc, "axial": wz, "modified_cyclotron": wc / 2 + r, "magnetron": wc / 2 - r}


def guiding_center(x: np.ndarray, v: np.ndarray, B: np.ndarray, q: float, m: float, gamma: Any = 1.0) -> np.ndarray:
    """Guiding centre x + gamma m (v x B) / (q |B|^2) of particles at x with velocity v in the local field B."""
    x, v, B = (np.asarray(a, float) for a in (x, v, B))
    g = np.asarray(gamma, float)[..., None] if np.ndim(gamma) else gamma
    return x + g * m * np.cross(v, B) / (q * (B * B).sum(-1, keepdims=True))


def magnetic_moment(v: np.ndarray, B: np.ndarray, m: float, gamma: Any = 1.0) -> np.ndarray:
    """First adiabatic invariant mu = p_perp^2 / (2 m |B|) [J/T] (gamma m v_perp^2 / 2B non-relativistically scaled)."""
    v, B = np.asarray(v, float), np.asarray(B, float)
    bn = np.linalg.norm(B, axis=-1)
    bh = B / bn[..., None]
    vpar = (v * bh).sum(-1)
    vperp2 = (v * v).sum(-1) - vpar**2
    g = np.asarray(gamma, float)
    return (g * g) * m * vperp2 / (2 * bn)


def pitch_velocity(speed: float, pitch_deg: float, b_hat: Sequence[float] = (0, 0, 1),
                   phase_deg: float = 0.0) -> np.ndarray:
    """Velocity of the given speed at a pitch angle (degrees from b_hat) and gyrophase."""
    b = np.asarray(b_hat, float)
    b = b / np.linalg.norm(b)
    e1 = np.cross(b, [1.0, 0, 0] if abs(b[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    a, p = math.radians(pitch_deg), math.radians(phase_deg)
    return speed * (math.cos(a) * b + math.sin(a) * (math.cos(p) * e1 + math.sin(p) * e2))


# ------------------------------------------------------------------------------------------------- studies
def _period_from_crossings(t: np.ndarray, s: np.ndarray) -> float:
    """Mean period of an oscillating signal from its upward zero crossings (linear interpolation)."""
    s = s - s.mean()
    i = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(i) < 2:
        return float("nan")
    tc = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
    return float(np.mean(np.diff(tc)))


def cyclotron_study(kinetic_energy_eV: float = 1.0e6, B: float = 1.0, particle: str = "electron",
                    steps_per_turn: int = 200, turns: int = 5) -> dict[str, Any]:
    """Gyration in a uniform B (along z) at a given kinetic energy: radius, frequency and energy drift from the
    Boris pusher against gamma m v / (qB), qB/(gamma m) and exact conservation.

    Parameters
    ----------
    kinetic_energy_eV : float [eV]
    B : float [T]
    particle : {"electron", "proton"}
    steps_per_turn : int [-]
    turns : int [-]
    """
    q, m = (-Q_E, M_E) if particle == "electron" else (Q_E, M_P)
    gamma = 1.0 + kinetic_energy_eV * Q_E / (m * C0 * C0)
    speed = C0 * math.sqrt(1.0 - 1.0 / gamma**2)
    w = abs(q) * B / (gamma * m)
    dt = 2 * math.pi / w / steps_per_turn
    out = push([[0, 0, 0]], [[speed, 0, 0]], q, m, B=uniform((0, 0, B)), dt=dt, n_steps=steps_per_turn * turns)
    xy = out["x"][:, 0, :2]
    gc = guiding_center(out["x"][:, 0], out["v"][:, 0], np.array([0, 0, B]), q, m, out["gamma"][:, 0])
    radius = float(np.linalg.norm(xy - gc[:, :2], axis=1).mean())
    period = _period_from_crossings(out["t"], out["v"][:, 0, 0])
    ke = out["kinetic_energy"][:, 0]
    return {"particle": particle, "gamma": gamma, "speed_over_c": speed / C0,
            "radius_m": radius, "radius_exact_m": larmor_radius(q, m, B, speed),
            "radius_nonrelativistic_m": m * speed / (abs(q) * B),
            "frequency_rad_s": 2 * math.pi / period, "frequency_exact_rad_s": w,
            "frequency_nonrelativistic_rad_s": abs(q) * B / m,
            "energy_rel_drift": float(np.abs(ke - ke[0]).max() / ke[0]),
            "t": out["t"], "x": out["x"][:, 0]}


def drift_study(B: float = 1.0e-3, E: float = 100.0, grad_scale: float = 20.0, energy_eV: float = 10.0,
                periods: int = 20, steps_per_turn: int = 120) -> dict[str, Any]:
    """E x B drift (uniform E along y, B along z) and grad-B drift (B_z = B (1 + x/grad_scale)) of a proton,
    measured from the guiding-centre motion against E/B and the grad-B formula. The grad-B formula is the
    guiding-centre limit: it needs a Larmor radius much smaller than ``grad_scale`` (the result reports the ratio).

    Parameters
    ----------
    B : float [T]
    E : float [V/m]
    grad_scale : float [m]
    energy_eV : float [eV]
    periods : int [-]
    steps_per_turn : int [-]
    """
    q, m = Q_E, M_P
    v = math.sqrt(2 * energy_eV * Q_E / m)
    w = q * B / m
    dt = 2 * math.pi / w / steps_per_turn
    n = periods * steps_per_turn
    a = push([[0, 0, 0]], [[v, 0, 0]], q, m, E=uniform((0, E, 0)), B=uniform((0, 0, B)), dt=dt, n_steps=n,
             relativistic=False)
    gca = guiding_center(a["x"][:, 0], a["v"][:, 0], np.array([0, 0, B]), q, m)
    vd_exb = float(np.polyfit(a["t"], gca[:, 0], 1)[0])

    def Bgrad(x, t):
        x = np.asarray(x, float)
        bz = B * (1 + x[..., 0] / grad_scale)
        return np.stack([np.zeros_like(bz), np.zeros_like(bz), bz], -1)

    b = push([[0, 0, 0]], [[v, 0, 0]], q, m, B=Bgrad, dt=dt, n_steps=n, relativistic=False)
    Bl = Bgrad(b["x"][:, 0], 0.0)
    gcb = guiding_center(b["x"][:, 0], b["v"][:, 0], Bl, q, m)
    vd_grad = float(np.polyfit(b["t"], gcb[:, 1], 1)[0])
    exact_grad = gradb_drift(q, m, v, [0, 0, B], [B / grad_scale, 0, 0])
    return {"exb_drift_m_s": vd_exb, "exb_drift_exact_m_s": float(exb_drift([0, E, 0], [0, 0, B])[0]),
            "gradb_drift_m_s": vd_grad, "gradb_drift_exact_m_s": float(exact_grad[1]),
            "larmor_radius_m": m * v / (q * B), "grad_scale_m": grad_scale,
            "larmor_over_grad_scale": m * v / (q * B) / grad_scale,
            "t": a["t"], "x_exb": a["x"][:, 0], "x_gradb": b["x"][:, 0]}


def magnetic_bottle_study(B0: float = 0.01, L: float = 1.0, z_mirror: float = 2.0, energy_eV: float = 100.0,
                          n_particles: int = 60, bounces: float = 3.0, seed: int = 0) -> dict[str, Any]:
    """Protons released at the centre of a magnetic bottle at pitch angles spread over 5-90 degrees: which escape
    past the mirror points at |z| = z_mirror, against the loss cone arcsin(sqrt(B0/B(z_mirror))); and how well the
    magnetic moment of a trapped particle is conserved.

    Parameters
    ----------
    B0 : float [T]
    L : float [m]
    z_mirror : float [m]
    energy_eV : float [eV]
    n_particles : int [-]
    bounces : float [-]
    seed : int [-]
    """
    q, m = Q_E, M_P
    v = math.sqrt(2 * energy_eV * Q_E / m)
    Bf = magnetic_bottle(B0, L)
    B_max = B0 * (1 + (z_mirror / L) ** 2)
    rng = np.random.default_rng(seed)
    pitch = np.sort(rng.uniform(5.0, 90.0, n_particles))
    v0 = np.array([pitch_velocity(v, p, phase_deg=float(rng.uniform(0, 360))) for p in pitch])
    w = q * B0 / m
    dt = 2 * math.pi / w / 60
    t_bounce = 2 * math.pi * L / v * 4           # generous: a few axial oscillations
    n = int(bounces * t_bounce / dt)
    out = push(np.zeros((n_particles, 3)), v0, q, m, B=Bf, dt=dt, n_steps=n, save_every=10, relativistic=False,
               stop=lambda x, vv: np.abs(x[:, 2]) > z_mirror)
    escaped = out["stopped"]
    cone = math.degrees(loss_cone_angle(B0, B_max))
    predicted = pitch < cone
    trapped = np.nonzero(~escaped)[0]
    mu_var = float("nan")
    if len(trapped):
        k = trapped[len(trapped) // 2]
        xs, vs = out["x"][:, k], out["v"][:, k]
        mu = magnetic_moment(vs, Bf(xs, 0.0), m)
        mu_var = float((mu.max() - mu.min()) / mu.mean())
    return {"loss_cone_deg": cone, "mirror_ratio": B_max / B0, "pitch_deg": pitch, "escaped": escaped,
            "agreement": float((escaped == predicted).mean()),
            "mismatch_pitch_deg": pitch[escaped != predicted],
            "mu_relative_variation": mu_var, "t": out["t"], "x": out["x"]}


def dipole_trap_study(energy_MeV: float = 10.0, L_shell: float = 2.0, pitch_deg: float = 30.0,
                      species: str = "proton", duration_s: float = 150.0, steps_per_gyration: int = 25,
                      save_every: int = 50) -> dict[str, Any]:
    """A particle trapped in an Earth-like dipole field (radiation-belt picture): gyration, bounce between mirror
    points and longitudinal drift around the planet. Reports the measured bounce and drift periods next to the
    standard dipole approximations (Hamlin et al. 1961 / Walt 1994, for equatorial pitch angle alpha):

        T_bounce ~ 4 L R_E / v * (1.30 - 0.56 sin alpha)
        T_drift  ~ 2 pi q B_E R_E^2 / (3 L gamma m v^2) / (0.35 + 0.15 sin alpha)    [B_E: equatorial surface field]

    Parameters
    ----------
    energy_MeV : float [MeV]
    L_shell : float [-]
    pitch_deg : float [deg]
    species : {"proton", "electron"}
    duration_s : float [s]
    steps_per_gyration : int [-]
    save_every : int [-]
    """
    q, m = (Q_E, M_P) if species == "proton" else (-Q_E, M_E)
    Bf = magnetic_dipole()
    R = L_shell * EARTH_RADIUS
    x0 = np.array([[R, 0.0, 0.0]])
    B_eq = float(np.linalg.norm(Bf(x0, 0.0)))
    gamma = 1.0 + energy_MeV * 1e6 * Q_E / (m * C0 * C0)
    v = C0 * math.sqrt(1.0 - 1.0 / gamma**2)
    v0 = pitch_velocity(v, pitch_deg, b_hat=Bf(x0, 0.0)[0], phase_deg=0.0)[None]
    w = abs(q) * B_eq / (gamma * m)
    dt = 2 * math.pi / w / steps_per_gyration
    n = int(duration_s / dt)
    out = push(x0, v0, q, m, B=Bf, dt=dt, n_steps=n, save_every=save_every,
               stop=lambda x, vv: np.linalg.norm(x, axis=1) < EARTH_RADIUS)
    x = out["x"][:, 0]
    sa = math.sin(math.radians(pitch_deg))
    B_E = float(np.linalg.norm(Bf(np.array([[EARTH_RADIUS, 0, 0]]), 0.0)))
    tb_approx = 4 * L_shell * EARTH_RADIUS / v * (1.30 - 0.56 * sa)
    td_approx = 2 * math.pi * abs(q) * B_E * EARTH_RADIUS**2 / (3 * L_shell * gamma * m * v * v) / (0.35 + 0.15 * sa)
    tb = _period_from_crossings(out["t"], x[:, 2])
    lon = np.unwrap(np.arctan2(x[:, 1], x[:, 0]))
    td = float(2 * math.pi / abs(np.polyfit(out["t"], lon, 1)[0])) if len(lon) > 3 else float("nan")
    return {"species": species, "gamma": gamma, "B_equator_T": B_eq, "gyro_period_s": 2 * math.pi / w,
            "bounce_period_s": tb, "bounce_period_approx_s": tb_approx,
            "drift_period_s": td, "drift_period_approx_s": td_approx,
            "drift_direction": "westward" if np.sign(lon[-1] - lon[0]) * np.sign(q) < 0 else "eastward",
            "lost": bool(out["stopped"][0]), "t": out["t"], "x": x}


def penning_study(B: float = 1.0, V0: float = 10.0, d: float = 5e-3, particle: str = "proton",
                  x0_m: Sequence[float] = (1e-4, 0.0, 1e-4), periods: int = 40, steps_per_turn: int = 80,
                  v0_m_s: Sequence[float] = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    """A charge in an ideal Penning trap: the three eigenfrequencies measured from the motion (spectra of x and z)
    against the closed forms.

    Parameters
    ----------
    B : float [T]
    V0 : float [V]
    d : float [m]
    particle : {"proton", "electron"}
    x0_m : list of float [m]
    periods : int [-]
    steps_per_turn : int [-]
    v0_m_s : list of float [m/s]
    """
    q, m = (Q_E, M_P) if particle == "proton" else (-Q_E, M_E)
    V = V0 if q > 0 else -V0
    f = penning_frequencies(abs(q), m, B, abs(V), d)
    dt = 2 * math.pi / f["modified_cyclotron"] / steps_per_turn
    T_mag = 2 * math.pi / f["magnetron"]
    n = int(max(periods * 2 * math.pi / f["axial"], 3 * T_mag) / dt)
    out = push([list(x0_m)], [list(v0_m_s)], q, m, E=penning_trap(V, d), B=uniform((0, 0, B)), dt=dt,
               n_steps=n, relativistic=False)
    t, x = out["t"], out["x"][:, 0]
    nfft = 1 << int(np.ceil(np.log2(len(t)) + 2))
    freqs = 2 * math.pi * np.fft.rfftfreq(nfft, dt)

    def peak(sig, lo, hi):
        S = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(len(sig)), nfft))
        sel = (freqs > lo) & (freqs < hi)
        i = np.argmax(np.where(sel, S, 0))
        # parabolic refinement of the peak
        if 0 < i < len(S) - 1:
            a, b, c = np.log(S[i - 1] + 1e-300), np.log(S[i] + 1e-300), np.log(S[i + 1] + 1e-300)
            i = i + 0.5 * (a - c) / (a - 2 * b + c)
        return float(i * (freqs[1] - freqs[0]))

    wz = peak(x[:, 2], 0.3 * f["axial"], 3 * f["axial"])
    w_plus = peak(x[:, 0], 0.5 * f["modified_cyclotron"], 2 * f["modified_cyclotron"])
    # magnetron: the slow rotation of the centre of the fast cyclotron circle (moving average over one cyclotron turn)
    k = np.ones(steps_per_turn) / steps_per_turn
    cx, cy = np.convolve(x[:, 0], k, "valid"), np.convolve(x[:, 1], k, "valid")
    tc = t[steps_per_turn // 2: steps_per_turn // 2 + len(cx)]
    w_minus = float(abs(np.polyfit(tc, np.unwrap(np.arctan2(cy, cx)), 1)[0]))
    return {"exact": f, "measured": {"axial": wz, "modified_cyclotron": w_plus, "magnetron": w_minus},
            "t": t, "x": x}


def mass_spectrometer_study(B: float = 0.5, energy_eV: float = 2000.0,
                            masses_u: Sequence[float] = (12.0, 13.0, 14.0), charge_state: int = 1) -> dict[str, Any]:
    """A 180-degree magnetic sector: ions of equal kinetic energy enter a uniform B and land at 2 r, with
    r = sqrt(2 m E_k) / (q B). The landing points from the pusher against the closed form, and the separation of
    neighbouring masses.

    Parameters
    ----------
    B : float [T]
    energy_eV : float [eV]
    masses_u : list of float [u]
    charge_state : int [-]
    """
    U = 1.66053906892e-27
    q = charge_state * Q_E
    res = []
    for mu in masses_u:
        m = mu * U
        v = math.sqrt(2 * energy_eV * Q_E / m)
        r = m * v / (q * B)
        dt = math.pi * r / v / 400
        out = push([[0, 0, 0]], [[0, v, 0]], q, m, B=uniform((0, 0, B)), dt=dt, n_steps=800, relativistic=False,
                   stop=lambda x, vv: (vv[:, 1] < 0) & (x[:, 1] <= 0))
        land = float(out["x"][-1, 0, 0]) if out["stopped"][0] else float("nan")
        res.append({"mass_u": mu, "radius_exact_m": r, "landing_m": land, "landing_exact_m": 2 * r,
                    "x": out["x"][:, 0]})
    return {"B": B, "energy_eV": energy_eV, "ions": res,
            "separation_m": [res[i + 1]["landing_m"] - res[i]["landing_m"] for i in range(len(res) - 1)]}
