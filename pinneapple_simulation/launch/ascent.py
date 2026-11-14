"""The three-degree-of-freedom ascent of a launch vehicle: position, velocity and mass in an Earth-centred inertial
frame, driven by thrust along a commanded direction, gravity (with J2 optionally), drag against the 1976 standard
atmosphere, and the Earth's rotation under the pad.

The flight has three parts, each with its own guidance:

* **The pitch program** (stages one and two). A vertical rise, then the thrust pitched above the local horizon along
  a schedule in *altitude*: the pitch falls from 90 deg at 1.5 km to ``pitch_end_deg`` at ``pitch_end_altitude_km``
  and stays there, following the local vertical and the azimuth of the orbit plane. With the exponent < 1 the vehicle
  pulls over gently where the dynamic pressure is highest and faster once the air is thin, which keeps the peak
  dynamic pressure and the gravity losses where a launcher wants them.
* **The coast to apogee.** The second stage is cut off when its tanks run dry, leaving a transfer orbit whose apogee
  is the target altitude; the vehicle coasts ballistically to that apogee (no thrust, gravity and drag only).
* **The circularisation** (the kick stage). At apogee the last stage burns along the velocity-to-be-gained,
  ``v_circular(r_target) - v``, until the velocity of the circular orbit at the target altitude is reached: a short
  prograde burn, the way a small launcher inserts a payload

Every loss is integrated over the whole flight, so the velocity bookkeeping closes:

    dv_thrust - dv_gravity - dv_drag - dv_steering = |v_cutoff| - |v_liftoff|

    from pinneapple_simulation.launch.ascent import Site, Target, simulate_ascent
    from pinneapple_simulation.launch.vehicle import small_launcher
    hist = simulate_ascent(small_launcher(), Site("Tanegashima", 30.4, 130.97), Target(500.0))
    hist.final_orbit()          # a, e, i, perigee, apogee, period of the inserted orbit
    hist.losses                 # the measured gravity, drag and steering losses [m/s]

The orbit is osculating and two-body: the model has no third-body, no solar pressure and no oblateness of the
inserted orbit (J2, when enabled, acts on the trajectory, and the check in :mod:`.verification` accounts for it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import atmosphere

__all__ = ["MU_EARTH", "R_EARTH", "OMEGA_EARTH", "J2_EARTH", "Earth", "Site", "Target", "GuidanceConfig",
           "AscentHistory", "simulate_ascent", "launch_azimuth_deg", "orbit_elements", "circular_speed"]

_OMEGA_Z = np.array([0.0, 0.0, 1.0])
MU_EARTH = 3.986004418e14        # WGS84 gravitational parameter [m^3/s^2]
R_EARTH = 6.371e6                # mean radius of the spherical Earth of this model [m]
OMEGA_EARTH = 7.2921150e-5       # sidereal rotation rate [rad/s]
J2_EARTH = 1.08262668e-3         # second zonal harmonic


@dataclass(frozen=True)
class Earth:
    """The planet the ascent is flown around: a sphere, rotating, optionally with J2."""
    mu: float = MU_EARTH
    radius: float = R_EARTH
    omega: float = OMEGA_EARTH
    j2: float = 0.0              # 0 keeps the ascent two-body (the default); J2_EARTH turns oblateness on


#: the default planet, built once (a singleton, so no call sits in an argument default)
EARTH = Earth()


@dataclass(frozen=True)
class Site:
    """A launch site: latitude and longitude [deg] and its elevation [m] above the mean radius."""
    name: str
    lat_deg: float
    lon_deg: float
    elevation_m: float = 0.0

    def position(self, earth: Earth | None = None):
        """Inertial position [m] at t = 0, with the ECI frame frozen to the Earth-fixed frame at that instant."""
        earth = earth or EARTH
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        r = earth.radius + self.elevation_m
        return np.array([r * math.cos(lat) * math.cos(lon), r * math.cos(lat) * math.sin(lon), r * math.sin(lat)])

    def velocity(self, earth: Earth | None = None):
        """Inertial velocity [m/s] of the pad: the Earth's rotation carries it eastwards, omega x r."""
        earth = earth or EARTH
        return earth.omega * _cross([0.0, 0.0, 1.0], self.position(earth))

    def east(self):
        """Local east unit vector."""
        lon = math.radians(self.lon_deg)
        return np.array([-math.sin(lon), math.cos(lon), 0.0])


def launch_azimuth_deg(lat_deg: float, inclination_deg: float, eastward: bool = True):
    """Azimuth [deg from north] of the launch that inserts into ``inclination_deg`` from ``lat_deg``.

    ``sin(azimuth) = cos(inclination) / cos(latitude)``; a due-east launch (90 deg) gives an inclination equal to the
    latitude, which is what Tanegashima does for its 30 deg missions, and an inclination above 90 deg (retrograde)
    gives an azimuth past 180 deg, towards the south. ``eastward=False`` mirrors it to the western azimuth.
    """
    c = math.cos(math.radians(inclination_deg)) / math.cos(math.radians(lat_deg))
    if abs(c) > 1.0 + 1e-12:
        raise ValueError(f"an inclination of {inclination_deg} deg cannot be reached from latitude {lat_deg} deg")
    az = math.degrees(math.asin(max(-1.0, min(1.0, c))))
    if az < 0.0:
        az = 180.0 - az                       # a retrograde inclination: an azimuth past 180 deg
    return az if eastward else 360.0 - az


def circular_speed(radius_m: float, mu: float = MU_EARTH) -> float:
    """Circular orbital speed [m/s] at ``radius_m``."""
    return math.sqrt(mu / radius_m)


@dataclass(frozen=True)
class Target:
    """The orbit to insert into: a circular (or near-circular) orbit at ``altitude_km`` and its inclination.

    ``inclination_deg=None`` takes the launch latitude: the due-east launch that the ascent flies.
    """
    altitude_km: float = 500.0
    inclination_deg: float | None = None


@dataclass
class GuidanceConfig:
    """The knobs of the ascent guidance (defaults are the ones the example flies)."""
    vertical_rise_s: float = 10.0       # thrust along the local vertical for this long
    pitch_start_altitude_km: float = 1.5   # the pitch program starts here ...
    pitch_end_altitude_km: float = 300.0   # ... and has reached pitch_end_deg here (the concept's ascent)
    pitch_end_deg: float = 0.0             # pitch above the local horizon at pitch_end_altitude_km
    pitch_exponent: float = 0.6            # xi = 90 - (90 - pitch_end) f**exponent, f the altitude fraction
    coast_to_apogee: bool = True           # the last stage coasts to apogee and circularises there (a kick stage)
    max_turn_rate_deg_s: float = 6.0    # the commanded direction may not turn faster than this
    cutoff_tol_m_s: float = 0.5         # engine cut-off when the velocity to be gained is this small
    altitude_tolerance_km: float = 2.0  # ... and the altitude is this close to the target


@dataclass
class AscentHistory:
    """The recorded ascent: dense-enough arrays for the plots and the 3-D scene, the events, the losses, the orbit."""
    t: np.ndarray
    r: np.ndarray                   # (T, 3) inertial position [m]
    v: np.ndarray                   # (T, 3) inertial velocity [m/s]
    mass: np.ndarray                # (T,) [kg]
    altitude: np.ndarray            # (T,) geometric altitude [m]
    speed: np.ndarray               # (T,) inertial speed [m/s]
    speed_rel: np.ndarray           # (T,) speed relative to the atmosphere [m/s]
    mach: np.ndarray
    q: np.ndarray                   # dynamic pressure [Pa]
    thrust: np.ndarray              # (T,) [N]
    pitch_deg: np.ndarray           # (T,) thrust direction above the local horizon
    gamma_deg: np.ndarray           # (T,) flight-path angle above the local horizon
    stage: np.ndarray               # (T,) stage index flying (-1 after cut-off)
    events: dict
    losses: dict
    vehicle: object
    site: object
    target: object
    earth: Earth
    plane_normal: np.ndarray        # unit normal of the orbital plane (inertial)
    azimuth_deg: float

    # ------------------------------------------------------------------ derived quantities
    def final_orbit(self) -> dict:
        """The osculating orbit at the last recorded state (after cut-off): elements, perigee, apogee, period."""
        return orbit_elements(self.r[-1], self.v[-1], self.earth.mu, self.earth.radius)

    def telemetry(self, step: int = 1) -> dict:
        """The arrays a dashboard, a HUD or a JSON report reads, thinned by ``step``."""
        s = slice(None, None, max(1, int(step)))
        lat, lon, _ = eci_to_geodetic(self.r[s], self.t[s], self.earth)
        return {
            "t_s": self.t[s], "altitude_km": self.altitude[s] / 1e3, "speed_m_s": self.speed[s],
            "speed_rel_m_s": self.speed_rel[s], "mach": self.mach[s], "q_pa": self.q[s],
            "thrust_kn": self.thrust[s] / 1e3, "mass_kg": self.mass[s], "pitch_deg": self.pitch_deg[s],
            "gamma_deg": self.gamma_deg[s], "stage": self.stage[s], "lat_deg": lat, "lon_deg": lon,
        }

    def downrange_km(self) -> np.ndarray:
        """Great-circle distance flown from the pad [km]."""
        lat, lon, _ = eci_to_geodetic(self.r, self.t, self.earth)
        lat0, lon0 = math.radians(self.site.lat_deg), math.radians(self.site.lon_deg)
        la, lo = np.radians(lat), np.radians(lon)
        c = np.sin(lat0) * np.sin(la) + math.cos(lat0) * np.cos(la) * np.cos(lo - lon0)
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) * math.pi / 180.0 * self.earth.radius / 1e3

    def summary(self) -> dict:
        orb = self.final_orbit()
        return {
            "site": self.site.name, "azimuth_deg": self.azimuth_deg,
            "target_altitude_km": self.target.altitude_km,
            "cutoff_time_s": float(self.events.get("cutoff_s", np.nan)),
            "max_q_pa": float(np.max(self.q)), "max_q_time_s": float(self.t[int(np.argmax(self.q))]),
            "max_q_altitude_km": float(self.altitude[int(np.argmax(self.q))] / 1e3),
            "max_mach": float(np.max(self.mach)),
            "max_acceleration_g": float(self.events.get("max_accel_g", np.nan)),
            "perigee_km": orb["perigee_km"], "apogee_km": orb["apogee_km"],
            "eccentricity": orb["e"], "inclination_deg": orb["inclination_deg"],
            "period_s": orb["period_s"], "losses_m_s": self.losses, "events": self.events,
            "downrange_km": float(self.downrange_km()[-1]),
        }


# ---------------------------------------------------------------------- orbit geometry
def orbit_elements(r, v, mu=MU_EARTH, radius=R_EARTH) -> dict:
    """Osculating elements of the orbit of state ``(r, v)`` (metres, m/s): a, e, i, perigee and apogee altitude."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    energy = vn * vn / 2.0 - mu / rn
    a = -mu / (2.0 * energy) if energy < 0 else float("inf")
    h = _cross(r, v)
    hn = _norm(h)
    evec = ((vn * vn - mu / rn) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(evec))
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / hn)))) if hn > 0 else float("nan")
    raan = math.degrees(math.atan2(h[0], -h[1])) % 360.0 if hn > 0 else float("nan")
    rp = a * (1.0 - e)
    ra = a * (1.0 + e)
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu) if np.isfinite(a) and a > 0 else float("inf")
    return {"a_m": a, "e": e, "inclination_deg": inc, "raan_deg": raan,
            "perigee_km": (rp - radius) / 1e3 if np.isfinite(a) else float("nan"),
            "apogee_km": (ra - radius) / 1e3 if np.isfinite(a) else float("nan"),
            "perigee_radius_m": rp, "apogee_radius_m": ra, "period_s": period,
            "specific_energy_j_kg": energy, "specific_angular_momentum_m2_s": hn,
            "radius_m": rn, "speed_m_s": vn}


def eci_to_geodetic(r, t, earth: Earth | None = None):
    """Geodetic (spherical) latitude, longitude [deg] and altitude [m] of inertial positions ``r`` at times ``t``."""
    earth = earth or EARTH
    R = np.atleast_2d(np.asarray(r, float))
    tt = np.atleast_1d(np.asarray(t, float))
    th = earth.omega * tt
    c, s = np.cos(th), np.sin(th)
    x = c * R[:, 0] + s * R[:, 1]
    y = -s * R[:, 0] + c * R[:, 1]
    z = R[:, 2]
    rad = np.sqrt(x * x + y * y + z * z)
    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, rad - earth.radius


# ---------------------------------------------------------------------- the guidance
class _Guidance:
    """The steering law of one ascent run (holds the pitch state between calls)."""

    def __init__(self, site: Site, target: Target, earth: Earth, config: GuidanceConfig, n_stages: int = 1):
        self.n_stages = n_stages
        self.site = site
        self.target = target
        self.earth = earth
        self.cfg = config
        p0 = site.position(earth)
        up = p0 / np.linalg.norm(p0)
        if target.inclination_deg is None:
            self.azimuth_deg = 90.0
            east = site.east()
        else:
            self.azimuth_deg = launch_azimuth_deg(site.lat_deg, target.inclination_deg)
            east = _azimuth_direction(p0, self.azimuth_deg)
        self.u0 = up
        self.h0 = east
        self.plane_normal = _unit(_cross(p0, east))       # the orbit plane the ascent stays in
        self.pitch_from_vertical = 0.0                      # radians, the state of the turn
        self.last_dir = up.copy()

    # -- the two laws
    def _pitch_program(self, t, r):
        """Thrust direction: up for the vertical rise, then a pitch schedule in *altitude*.

        The pitch above the local horizon falls from 90 deg at ``pitch_start_altitude_km`` to ``pitch_end_deg`` at
        ``pitch_end_altitude_km`` with the exponent of the schedule, and stays there above it. Pitching by the altitude
        flown (not by the clock) keeps the vehicle out of the dense air whatever its thrust-to-weight: it is the
        schedule a launcher follows; :mod:`.sizing` closes the vehicle around it on the simulation.
        """
        cfg = self.cfg
        up = _unit(r)
        hdir = self._hdir(up)
        if t <= cfg.vertical_rise_s:
            return up
        alt_km = (np.linalg.norm(r) - self.earth.radius) / 1e3
        span = max(cfg.pitch_end_altitude_km - cfg.pitch_start_altitude_km, 1e-6)
        frac = min(1.0, max(0.0, (alt_km - cfg.pitch_start_altitude_km) / span))
        xi = math.radians(90.0 - (90.0 - cfg.pitch_end_deg) * frac ** cfg.pitch_exponent)
        return _unit(math.sin(xi) * up + math.cos(xi) * hdir)

    def _hdir(self, up):
        """Unit horizontal direction of the launch azimuth at radius direction ``up`` (in the orbit plane)."""
        h = _cross(self.plane_normal, up)
        return _unit(h) if _norm(h) > 1e-9 else self.h0

    def _circularize(self, r, v):
        """Kick-stage steering: gain the velocity that the circular orbit at the target altitude still lacks.

        Thrusting along the velocity-to-be-gained with the gravity of the short burn left out is the right law for a
        circularisation at apogee: the burn lasts tens of seconds and is almost entirely prograde.
        """
        up = _unit(r)
        hdir = self._hdir(up)
        r_target = self.earth.radius + self.target.altitude_km * 1e3
        return _unit(circular_speed(r_target, self.earth.mu) * hdir - v)

    def _rate_limit(self, u, dt_hint):
        """``u`` turned from the last committed direction by at most ``max_turn_rate_deg_s * dt_hint`` (pure: the
        committed direction is only advanced by :meth:`commit`, once per integrator step)."""
        max_turn = math.radians(self.cfg.max_turn_rate_deg_s) * max(dt_hint, 1e-3)
        c = float(np.clip(_dot(u, self.last_dir), -1.0, 1.0))
        if math.acos(c) <= max_turn:
            return u
        axis = _cross(self.last_dir, u)
        if _norm(axis) < 1e-12:
            return u
        return _unit(_rotate(self.last_dir, _unit(axis), max_turn))

    def direction(self, t, r, v, mass, stage_index, stage, dv_remaining, dt_hint=0.05):
        """The unit thrust direction of this step and the law that produced it (no side effect)."""
        if self.cfg.coast_to_apogee and stage_index == self.n_stages - 1 and t > self.cfg.vertical_rise_s + 1.0:
            u, law = self._circularize(r, v), "circularisation"
        else:
            u, law = self._pitch_program(t, r), "pitch program"
        return self._rate_limit(u, dt_hint), law

    def commit(self, u):
        """Advance the steering state to the direction the step was flown with."""
        self.last_dir = _unit(u)

    def pitch_deg(self, r, u):
        up = _unit(r)
        return math.degrees(math.asin(float(np.clip(_dot(u, up), -1.0, 1.0))))


def _cross(a, b):
    """3-D cross product without numpy's argument normalisation (that path costs a millisecond a call)."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    return np.array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])


def _norm(a):
    return math.sqrt(float(a[0]) * float(a[0]) + float(a[1]) * float(a[1]) + float(a[2]) * float(a[2]))


def _dot(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _unit(a):
    a = np.asarray(a, float)
    n = _norm(a)
    return a / n if n > 0 else a


def _rotate(vec, axis, angle):
    """Rodrigues rotation of ``vec`` about the unit ``axis`` by ``angle`` (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    return vec * c + _cross(axis, vec) * s + axis * _dot(axis, vec) * (1.0 - c)


def _azimuth_direction(r, azimuth_deg):
    """Unit horizontal vector at ``r`` pointing along an azimuth measured from north (clockwise)."""
    up = _unit(r)
    north = _unit(np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], up) * up)
    east = _unit(_cross([0.0, 0.0, 1.0], up))
    az = math.radians(azimuth_deg)
    return _unit(math.cos(az) * north + math.sin(az) * east)


# ---------------------------------------------------------------------- the simulation
def _gravity(r, earth: Earth):
    rn2 = _dot(r, r)
    rn = math.sqrt(rn2)
    a = -earth.mu * np.asarray(r, float) / rn ** 3
    if earth.j2:
        z2 = r[2] * r[2]
        k = 1.5 * earth.j2 * earth.mu * earth.radius ** 2 / rn ** 5
        a = a + k * np.array([r[0] * (5 * z2 / rn2 - 1), r[1] * (5 * z2 / rn2 - 1), r[2] * (5 * z2 / rn2 - 3)])
    return a


def simulate_ascent(vehicle, site: Site, target: Target, *, earth: Earth | None = None, dt: float = 0.05,
                    t_max: float = 1500.0, record_dt: float = 0.5, guidance: GuidanceConfig | None = None,
                    coast_s: float = 0.0, terminal_tol_m_s: float = 0.5) -> AscentHistory:
    """Fly ``vehicle`` from ``site`` to ``target`` and return the recorded :class:`AscentHistory`.

    ``dt`` is the integration step (RK4), ``record_dt`` the sampling of the returned arrays, ``coast_s`` the time
    flown ballistically after cut-off (0 stops the run at the engine cut-off). The feedback guidance is evaluated at
    every integrator substep, so the trajectory is the closed loop, not a pre-computed pitch program.
    """
    earth = earth or EARTH
    cfg = guidance or GuidanceConfig()
    cfg = replace(cfg, cutoff_tol_m_s=terminal_tol_m_s)
    earth = Earth(mu=earth.mu, radius=earth.radius, omega=earth.omega, j2=earth.j2)
    veh = vehicle
    stages = veh.stages
    guide = _Guidance(site, target, earth, cfg, n_stages=len(stages))

    r = site.position(earth).astype(float)
    v = site.velocity(earth).astype(float)
    m = float(veh.liftoff_mass_kg)
    v0 = v.copy()

    stage, prop_left, flying, fairing_on = 0, stages[0].propellant_kg, True, veh.fairing_kg > 0.0
    t, next_record = 0.0, 0.0
    rec = {k: [] for k in ("t", "r", "v", "mass", "altitude", "speed", "speed_rel", "mach", "q", "thrust",
                           "pitch_deg", "gamma_deg", "stage")}
    events = {"liftoff_s": 0.0, "liftoff_mass_kg": m, "meco_s": [], "separation_s": [], "ignition_s": [],
              "guides": []}
    losses = {"dv_thrust_m_s": 0.0, "dv_gravity_m_s": 0.0, "dv_drag_m_s": 0.0, "dv_steering_m_s": 0.0}
    max_q, max_accel = -1.0, 0.0
    cutoff_done = False
    coasting = False

    def ambient(rr):
        """Altitude [m] and the atmospheric state there (one evaluation: it is on the hot path)."""
        alt = _norm(rr) - earth.radius
        return alt, atmosphere.state(max(alt, 0.0))

    def drag_force(rr, vv, air):
        v_rel = vv - _cross(_OMEGA_Z * earth.omega, rr)
        vrn = _norm(v_rel)
        mach = vrn / air["a"]
        idx = stage if stage >= 0 else len(stages) - 1
        D = 0.5 * air["rho"] * vrn * vrn * veh.cd(mach) * veh.drag_area_m2(idx)
        return D, v_rel, vrn, air["rho"], mach

    def step(prop):
        """One integrator step: the guidance is a zero-order hold over it (a 20 Hz digital controller)."""
        nonlocal r, v, m, t, stage, prop_left, flying, max_q, max_accel, cutoff_done, next_record, fairing_on
        nonlocal coasting
        h = dt
        T, u_cmd, mdot, law = 0.0, guide.last_dir, 0.0, "coast"
        if coasting:
            up = _unit(r)
            if _dot(v, up) <= 0.0:                       # apogee: light the kick stage
                coasting, flying = False, True
                events["apogee_s"] = t
                events["apogee_altitude_km"] = (_norm(r) - earth.radius) / 1e3
                events["ignition_s"].append(t)
                events["coast_s"] = events["apogee_s"] - events.get("coast_start_s", t)
        if flying and stage >= 0:
            st = stages[stage]
            _, air = ambient(r)
            T, mdot = st.thrust_n(air["p"]), st.mass_flow_kg_s(air["p"])
            # velocity still in this stage's tank: c ln(m / (m - propellant left)), exact whatever has been jettisoned
            dv_rem = st.isp_vac_s * 9.80665 * math.log(m / max(m - prop_left, 1e-9))
            u_cmd, law = guide.direction(t, r, v, m, stage, st, dv_rem, dt)
            if not events["guides"] or events["guides"][-1][1] != law:
                events["guides"].append([round(t, 3), law])
            if mdot > 0:
                remaining = prop_left / mdot
                # a tank that is numerically empty must not shorten the step to nothing (t would stop advancing)
                h = remaining if 1e-6 < remaining < dt else dt
                if remaining <= 1e-6:
                    prop_left = 0.0
        guide.commit(u_cmd)

        def deriv(y):
            rr, vv, mm = y
            _, air = ambient(rr)
            D, v_rel, vrn, _, _ = drag_force(rr, vv, air)
            a_drag = -(D / mm) * (v_rel / vrn) if vrn > 1e-9 else np.zeros(3)
            a_thrust = (T / mm) * u_cmd if T > 0 else np.zeros(3)
            return (vv, _gravity(rr, earth) + a_thrust + a_drag, -mdot if T > 0 else 0.0)

        y = (r, v, m)
        k1 = deriv(y)
        k2 = deriv(tuple(y[i] + 0.5 * h * k1[i] for i in range(3)))
        k3 = deriv(tuple(y[i] + 0.5 * h * k2[i] for i in range(3)))
        k4 = deriv(tuple(y[i] + h * k3[i] for i in range(3)))
        r, v, m = tuple(y[i] + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3))
        t += h
        alt = _norm(r) - earth.radius

        if flying and stage >= 0:
            _, air = ambient(r)
            D, v_rel, vrn, rho, mach = drag_force(r, v, air)
            g = _gravity(r, earth)
            vhat = v / _norm(v)
            # the bookkeeping is against the inertial velocity: |v| only changes with the component of each
            # acceleration along v, which is what the identity below integrates
            cos_alpha = float(np.clip(_dot(u_cmd, vhat), -1.0, 1.0))
            cos_beta = float(np.clip(_dot(v_rel / max(vrn, 1e-9), vhat), -1.0, 1.0))
            losses["dv_thrust_m_s"] += (T / m) * h
            losses["dv_gravity_m_s"] += -_dot(g, vhat) * h
            losses["dv_drag_m_s"] += (D / m) * cos_beta * h
            losses["dv_steering_m_s"] += (T / m) * (1.0 - cos_alpha) * h
            max_accel = max(max_accel, (T + D) / m)
            prop_left = max(prop_left - mdot * h, 0.0)
            q = 0.5 * rho * vrn * vrn
            if q > max_q:
                max_q = q
                events.update({"max_q_s": t, "max_q_pa": q, "max_q_altitude_km": alt / 1e3, "max_q_mach": mach})
            r_target = earth.radius + target.altitude_km * 1e3
            hdir = _unit(_cross(guide.plane_normal, _unit(r)))
            last = stage == len(stages) - 1
            if prop_left <= 1e-9:                        # a tank that is empty ends the burn, whatever the target
                if last:
                    _shut_down("propellant exhausted")
                else:
                    events["meco_s"].append(t)
                    events.setdefault("meco_altitude_km", alt / 1e3)
                    events.setdefault("meco_speed_m_s", _norm(v))
                    m -= stages[stage].dry_mass_kg
                    stage += 1
                    prop_left = stages[stage].propellant_kg
                    events["separation_s"].append(t)
                    if cfg.coast_to_apogee and stage == len(stages) - 1:
                        coasting, flying = True, False          # a kick stage: coast to apogee, then circularise
                        events["coast_start_s"] = t
                    else:
                        events["ignition_s"].append(t)
            if flying and last:
                if cfg.coast_to_apogee:
                    # the kick stage circularises: cut off when the orbit *is* circular, on the osculating perigee,
                    # which rises monotonically through the burn (a velocity error is not monotone near apogee, and the
                    # cut-off would then depend on the step)
                    if orbit_elements(r, v, earth.mu, earth.radius)["perigee_km"] >= (
                            target.altitude_km - cfg.altitude_tolerance_km):
                        _shut_down("cut-off with the orbit circular")
                elif (_norm(v - circular_speed(r_target, earth.mu) * hdir) < cfg.cutoff_tol_m_s
                      and abs(alt - target.altitude_km * 1e3) < cfg.altitude_tolerance_km * 1e3):
                    _shut_down("cut-off on the target velocity")

        elif not flying and stage >= 0:
            # the coast pays gravity and drag too, and the velocity bookkeeping covers the whole flight
            _, air = ambient(r)
            D, v_rel, vrn, _, _ = drag_force(r, v, air)
            g = _gravity(r, earth)
            vhat = v / _norm(v)
            cos_beta = float(np.clip(_dot(v_rel / max(vrn, 1e-9), vhat), -1.0, 1.0))
            losses["dv_gravity_m_s"] += -_dot(g, vhat) * h
            losses["dv_drag_m_s"] += (D / m) * cos_beta * h

        if fairing_on and alt >= veh.fairing_jettison_altitude_km * 1e3:
            m -= veh.fairing_kg
            fairing_on = False
            events["fairing_jettison_s"] = t
            events["fairing_jettison_altitude_km"] = alt / 1e3

        # ---- record
        if t >= next_record - 1e-9:
            _record(u_cmd)
            while next_record <= t:
                next_record += record_dt

    def _shut_down(why):
        nonlocal flying, stage, cutoff_done
        alt = _norm(r) - earth.radius
        events["meco_s"].append(t)
        events["cutoff_s"] = t
        events["cutoff_altitude_km"] = alt / 1e3
        events["cutoff_speed_m_s"] = _norm(v)
        events["cutoff_reason"] = why
        events["propellant_left_kg"] = float(prop_left)
        events["propellant_fraction_left"] = float(prop_left / stages[stage].propellant_kg) if stage >= 0 else 0.0
        if stage == len(stages) - 1:
            events["meco_altitude_km"] = alt / 1e3
            events["meco_speed_m_s"] = _norm(v)
        flying, stage, cutoff_done = False, -1, True

    def _record(u_cmd):
        alt, air = ambient(r)
        v_rel = v - _cross(_OMEGA_Z * earth.omega, r)
        vrn = _norm(v_rel)
        rho = air["rho"]
        T = stages[stage].thrust_n(air["p"]) if (flying and stage >= 0) else 0.0
        rec["t"].append(t)
        rec["r"].append(r.copy())
        rec["v"].append(v.copy())
        rec["mass"].append(m)
        rec["altitude"].append(alt)
        rec["speed"].append(_norm(v))
        rec["speed_rel"].append(vrn)
        rec["mach"].append(vrn / air["a"])
        rec["q"].append(0.5 * rho * vrn * vrn)
        rec["thrust"].append(T)
        rec["pitch_deg"].append(guide.pitch_deg(r, u_cmd))
        rec["gamma_deg"].append(math.degrees(math.asin(float(np.clip(
            _dot(v_rel / max(vrn, 1e-9), _unit(r)), -1.0, 1.0)))))
        rec["stage"].append(stage if flying else (-2 if coasting else -1))

    next_record = 0.0
    while t < t_max:
        step(prop_left)
        if cutoff_done and (coast_s <= 0.0 or t >= events["cutoff_s"] + coast_s):
            break

    events["max_accel_g"] = max_accel / 9.80665
    events["stopped"] = "cut-off" if cutoff_done else "t_max"
    hist = AscentHistory(
        t=np.array(rec["t"]), r=np.array(rec["r"]), v=np.array(rec["v"]), mass=np.array(rec["mass"]),
        altitude=np.array(rec["altitude"]), speed=np.array(rec["speed"]), speed_rel=np.array(rec["speed_rel"]),
        mach=np.array(rec["mach"]), q=np.array(rec["q"]), thrust=np.array(rec["thrust"]),
        pitch_deg=np.array(rec["pitch_deg"]), gamma_deg=np.array(rec["gamma_deg"]),
        stage=np.array(rec["stage"]), events=events, losses=losses, vehicle=veh, site=site, target=target,
        earth=earth, plane_normal=guide.plane_normal, azimuth_deg=guide.azimuth_deg,
    )
    hist.losses["dv_achieved_m_s"] = _norm(v) - _norm(v0)
    hist.losses["closure_m_s"] = float(
        losses["dv_thrust_m_s"] - losses["dv_gravity_m_s"] - losses["dv_drag_m_s"] - losses["dv_steering_m_s"]
        - hist.losses["dv_achieved_m_s"])
    return hist
