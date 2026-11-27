"""Heliocentric states of the planets for trajectory design: a local JPL DE kernel when one is installed, and the
published JPL Keplerian elements as the offline fallback.

Trajectory design needs a planet state at an arbitrary epoch, thousands of times per optimisation run, and it needs
it in the **J2000 mean ecliptic** frame that the patched-conic geometry is drawn in (the main-belt inclination of a
transfer is a number a designer reads off directly). Both sources below return that frame.

* :class:`KernelEphemeris` reads the JPL DE kernel that :mod:`..ephemeris` already finds (``skyfield-data`` ships
  DE421, 1899-2053) and rotates the ICRF state to the J2000 ecliptic with :func:`..frames.ecliptic_matrix`. This is
  the source to use for work at the arcsecond level; it is what the checks in ``docs/core_concepts/trajectory_design.md``
  are measured against.
* :class:`AnalyticEphemeris` propagates the **JPL "Approximate Positions of the Planets"** elements (Standish,
  Table 1: semi-major axis, eccentricity, inclination, mean longitude, longitude of perihelion and longitude of the
  ascending node at J2000, plus their rates per Julian century, ecliptic and equinox of J2000, valid 1800-2050). It
  is offline, dependency-free, and accurate to the residuals in the guide -- enough for design, not for navigation.

Both expose ``state(body, jd_tdb) -> (r_km, v_km_s)`` with shape ``(m, 3)`` for ``m`` epochs, and
``.source``, a one-line description of where the numbers come from. :func:`default_ephemeris` returns the kernel one
when a kernel is present and the analytic one otherwise, so a design script is reproducible on a machine with no
ephemeris downloaded -- quote ``.source`` in a report.

:class:`GTOPEphemeris` is the third source, and it is not a better one: it is the ESA ACT table the GTOP benchmark
numbers are quoted in, kept so that a number from this package can be put next to a published one. See its
docstring for what the difference costs.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .. import frames
from ..ephemeris import GM, GM_SUN, RADIUS_KM
from ..ephemeris import default_ephemeris as _kernel_ephemeris
from ..orbits import solve_kepler

__all__ = ["AnalyticEphemeris", "GTOPEphemeris", "KernelEphemeris", "default_ephemeris", "J2000_ELEMENTS",
           "ACT_ELEMENTS", "DESIGN_BODIES"]

AU_KM = 149597870.7
DAY_S = 86400.0
J2000 = 2451545.0

#: Bodies a patched-conic design can fly by, with the radii and GM the geometry needs.
DESIGN_BODIES = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")

#: JPL "Approximate Positions of the Planets" (E. M. Standish), Table 1. Elements at J2000 and their rates per
#: Julian century; angles in degrees, ``a`` in au, referred to the mean ecliptic and equinox of J2000. The Earth
#: row is the Earth-Moon barycentre, 4 700 km from the Earth's centre at most (0.003 % of 1 au), which the design
#: does not resolve. Table 2 of the same page adds the small long-period terms of Jupiter to Neptune; they are
#: **not** included here -- the residual they leave is measured against DE421 in the guide, so nothing about it is
#: hidden. https://ssd.jpl.nasa.gov/planets/approx_pos.html
J2000_ELEMENTS = {
    "mercury": {"a": 0.38709927, "e": 0.20563593, "I": 7.00497902, "L": 252.25032350,
                "peri": 77.45779628, "node": 48.33076593,
                "a_rate": 0.00000037, "e_rate": 0.00001906, "I_rate": -0.00594749,
                "L_rate": 149472.67411175, "peri_rate": 0.16047689, "node_rate": -0.12534081},
    "venus": {"a": 0.72333566, "e": 0.00677672, "I": 3.39467605, "L": 181.97909950,
              "peri": 131.60246718, "node": 76.67984255,
              "a_rate": 0.00000390, "e_rate": -0.00004107, "I_rate": -0.00078890,
              "L_rate": 58517.81538729, "peri_rate": 0.00268329, "node_rate": -0.27769418},
    "earth": {"a": 1.00000261, "e": 0.01671123, "I": -0.00001531, "L": 100.46457166,
              "peri": 102.93768193, "node": 0.0,
              "a_rate": 0.00000562, "e_rate": -0.00004392, "I_rate": -0.01294668,
              "L_rate": 35999.37244981, "peri_rate": 0.32327364, "node_rate": 0.0},
    "mars": {"a": 1.52371034, "e": 0.09339410, "I": 1.84969142, "L": -4.55343205,
             "peri": -23.94362959, "node": 49.55953891,
             "a_rate": 0.00001847, "e_rate": 0.00007882, "I_rate": -0.00813131,
             "L_rate": 19140.30268499, "peri_rate": 0.44441088, "node_rate": -0.29257343},
    "jupiter": {"a": 5.20288700, "e": 0.04838624, "I": 1.30439695, "L": 34.39644051,
                "peri": 14.72847983, "node": 100.47390909,
                "a_rate": -0.00011607, "e_rate": -0.00013253, "I_rate": -0.00183714,
                "L_rate": 3034.74612775, "peri_rate": 0.21252668, "node_rate": 0.20469106},
    "saturn": {"a": 9.53667594, "e": 0.05386179, "I": 2.48599187, "L": 49.95424423,
               "peri": 92.59887831, "node": 113.66242448,
               "a_rate": -0.00125060, "e_rate": -0.00050991, "I_rate": 0.00193609,
               "L_rate": 1222.49362201, "peri_rate": -0.41897216, "node_rate": -0.28867794},
    "uranus": {"a": 19.18916464, "e": 0.04725744, "I": 0.77263783, "L": 313.23810451,
               "peri": 170.95427630, "node": 74.01692503,
               "a_rate": -0.00196176, "e_rate": -0.00004397, "I_rate": -0.00242939,
               "L_rate": 428.48202785, "peri_rate": 0.40805281, "node_rate": 0.04240589},
    "neptune": {"a": 30.06992276, "e": 0.00859048, "I": 1.77004347, "L": -55.12002969,
                "peri": 44.96476227, "node": 131.78422574,
                "a_rate": 0.00026291, "e_rate": 0.00005105, "I_rate": 0.00035372,
                "L_rate": 218.45945325, "peri_rate": -0.32241464, "node_rate": -0.00508664},
}


def _rotation_ecliptic(node_deg, inc_deg, omega_deg):
    """Perifocal -> J2000 ecliptic rotation ``Rz(node) Rx(inc) Rz(omega)`` of the JPL page, ``(3, 3)``.

    ``omega`` is the *argument* of perihelion, ``omega = perihelion longitude - node``; the table lists the
    longitude. :func:`~..frames.rot_z` and friends are passive (coordinate-frame) rotations -- ``rot_z(a)`` turns a
    vector by ``-a`` -- so each factor is negated here, and the matrix returned is the active one: a perifocal
    column vector is rotated as ``R @ x``.
    """
    r = np.radians
    return frames.rot_z(-r(node_deg)) @ frames.rot_x(-r(inc_deg)) @ frames.rot_z(-r(omega_deg))


def _rotation_rate(node_deg, inc_deg, omega_deg, node_rate, inc_rate, omega_rate, h=1e-3):
    """Time derivative of :func:`_rotation_ecliptic` per Julian century, by central difference over ``2 h``
    centuries: the angles are linear in ``T``, so the difference is exact to the second derivative of a rotation,
    which is ``O(h^2) = 1e-6`` of the rate itself."""
    T = np.array([-h, 0.0, h])
    R = np.array([_rotation_ecliptic(node_deg + node_rate * t, inc_deg + inc_rate * t, omega_deg + omega_rate * t)
                  for t in T])
    return (R[2] - R[0]) / (2 * h)


class AnalyticEphemeris:
    """Keplerian planet states from the JPL approximate elements (Standish, Table 1).

    The secular rates of the elements are carried, so the mean longitude, perihelion, node and eccentricity all
    move between 1800 and 2050; the small periodic terms of Jupiter to Neptune are not (see :data:`J2000_ELEMENTS`).

    A single epoch takes a scalar ``math`` path -- a trajectory optimiser asks for one state at a time, millions of
    times per run, and the array path below costs 120 us of NumPy dispatch for one planet. The two paths agree bit
    for bit, which ``tests/test_trajectory_design.py`` asserts.
    """

    source = "JPL approximate positions of the planets (Standish, Table 1, J2000 ecliptic)"

    def __init__(self, bodies=DESIGN_BODIES):
        missing = set(bodies) - set(J2000_ELEMENTS)
        if missing:
            raise ValueError(f"no approximate elements for {sorted(missing)}")
        self.bodies = tuple(bodies)
        self._cache = {}

    def _precompute(self, body: str):
        if body not in self._cache:
            el = J2000_ELEMENTS[body]
            omega = el["peri"] - el["node"]
            omega_rate = el["peri_rate"] - el["node_rate"]
            rot = _rotation_ecliptic(el["node"], el["I"], omega)
            rot_rate = _rotation_rate(el["node"], el["I"], omega, el["node_rate"], el["I_rate"], omega_rate)
            self._cache[body] = (rot, rot_rate, tuple(map(tuple, rot.tolist())), tuple(map(tuple, rot_rate.tolist())))
        return self._cache[body]

    def elements(self, body: str, jd_tdb) -> dict:
        """Varying elements at the epoch: ``a_au``, ``e``, ``i_deg``, ``L_deg``, ``peri_deg``, ``node_deg``."""
        el = J2000_ELEMENTS[body]
        T = (np.atleast_1d(np.asarray(jd_tdb, dtype=float)) - J2000) / 36525.0
        out = {k: el[k] + el[f"{k}_rate"] * T for k in ("a", "e", "I", "L")}
        out["peri"] = el["peri"] + el["peri_rate"] * T
        out["node"] = el["node"] + el["node_rate"] * T
        return out

    def state(self, body: str, jd_tdb):
        """Heliocentric position (km) and velocity (km/s), J2000 ecliptic, ``(m, 3)`` each."""
        jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        if body == "sun":
            return np.zeros((len(jd), 3)), np.zeros((len(jd), 3))
        if np.ndim(jd_tdb) == 0:
            r, v = self._state_scalar(body, float(jd[0]))
            return np.asarray([r]), np.asarray([v])
        el = self.elements(body, jd_tdb)
        rot, rot_rate, _, _ = self._precompute(body)
        a, e = el["a"] * AU_KM, el["e"]
        M = np.radians(el["L"] - el["peri"])
        E = solve_kepler(M, e)
        cosE, sinE = np.cos(E), np.sin(E)
        b = a * np.sqrt(np.maximum(1 - e**2, 0.0))
        # perifocal position and its time derivative; n from the osculating ellipse of that epoch
        n = np.sqrt(GM_SUN / a**3)
        one_minus_e_cos = 1 - e * cosE
        x_p = np.stack([a * (cosE - e), b * sinE, np.zeros_like(a)], -1)
        v_p = np.stack([-a * sinE * n / one_minus_e_cos, b * cosE * n / one_minus_e_cos,
                        np.zeros_like(a)], -1)
        r = x_p @ rot.T
        v = v_p @ rot.T + x_p @ rot_rate.T * (1.0 / 36525.0) / DAY_S
        return r, v

    def _state_scalar(self, body: str, jd: float):
        """One epoch, plain floats: :meth:`state`'s formula term for term, without the NumPy dispatch."""
        el = J2000_ELEMENTS[body]
        T = (jd - J2000) / 36525.0
        a = (el["a"] + el["a_rate"] * T) * AU_KM
        e = el["e"] + el["e_rate"] * T
        M = math.radians((el["L"] + el["L_rate"] * T) - (el["peri"] + el["peri_rate"] * T)) % (2.0 * math.pi)
        _, _, rot, rot_rate = self._precompute(body)
        E = M + 0.85 * e * (1.0 if math.sin(M) >= 0.0 else -1.0)
        for _ in range(50):
            dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < 1e-14:
                break
        cosE, sinE = math.cos(E), math.sin(E)
        b = a * math.sqrt(max(1.0 - e * e, 0.0))
        n = math.sqrt(GM_SUN / (a * a * a))
        om = 1.0 - e * cosE
        xp = (a * (cosE - e), b * sinE, 0.0)
        vp = (-a * sinE * n / om, b * cosE * n / om, 0.0)
        scale = (1.0 / 36525.0) / DAY_S
        r = tuple(sum(xp[k] * rot[j][k] for k in range(3)) for j in range(3))
        v = tuple(sum(vp[k] * rot[j][k] for k in range(3)) + scale * sum(xp[k] * rot_rate[j][k] for k in range(3))
                  for j in range(3))
        return r, v



class KernelEphemeris:
    """Planet states from the local JPL DE kernel (:mod:`..ephemeris`), rotated to the J2000 ecliptic.

    Raises :class:`FileNotFoundError` when no kernel is installed -- :func:`default_ephemeris` then falls back to
    :class:`AnalyticEphemeris`.
    """

    source = "JPL DE kernel (skyfield-data DE421 or $PINNEAPPLE_DE_KERNEL), mean ecliptic J2000"

    def __init__(self, path: str | None = None):
        self.kernel = _kernel_ephemeris(path)
        self.rotation = frames.ecliptic_matrix(None)
        self.source = f"JPL DE kernel {self.kernel.path}, mean ecliptic J2000"

    def state(self, body: str, jd_tdb):
        """Heliocentric position (km) and velocity (km/s), J2000 ecliptic, ``(m, 3)`` each."""
        jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        p, v = self.kernel.barycentric(body, jd)
        sp, sv = self.kernel.barycentric("sun", jd)
        return (p - sp) @ self.rotation.T, (v - sv) @ self.rotation.T


def _poly(coefficients, T):
    """``c0 + c1 T + c2 T**2 + ...`` as a Python float."""
    out = 0.0
    for c in reversed(coefficients):
        out = out * T + c
    return out


def _act_elements(body: str, T: float) -> tuple[float, float, float, float, float, float]:
    """``(a_au, e, i_deg, node_deg, arg_of_perihelion_deg, mean_longitude_deg)`` of the ACT table at ``T``."""
    row = ACT_ELEMENTS[body]
    a = row["a"]
    e = _poly(row["e"], T)
    inc = _poly(row["i"], T)
    node = _poly(row["node"], T)
    argp = _poly(row["argp"], T)
    mean_longitude = row["L0"] + _poly(row["motion"], T) * T
    return a, e, inc, node, argp, mean_longitude


#: The analytic planet elements of the ESA ACT toolbox (``Pl_Eph_An.cpp``), in the form the file states them: ``a``
#: in au, the other five as polynomials in Julian centuries from **J1900** (``T = (mjd2000 + 36525)/36525``), degrees.
#: ``argp`` is the *argument* of perihelion (the classic Almanac ``omega``, not the longitude) and
#: ``L0 + motion(T) T`` the mean longitude. These are the classic Astronomical Almanac / Meeus elements, and they are
#: what the GTOP benchmark numbers are quoted in -- they differ from the modern JPL table in :data:`J2000_ELEMENTS`
#: by up to 4e-6 au (6e5 km) in ``a``, which at Cassini1's epochs is ~1.2e6 km of position and 0.4 km/s of velocity.
#: Not a better ephemeris, not a worse one: the one the published 4.9307 km/s belongs to.
ACT_ELEMENTS = {
    "mercury": {"a": 0.38709860,
                "e": (0.205614210, 0.000020460, -0.000000030),
                "i": (7.002880555555555560, 1.86083333333333333e-3, -1.83333333333333333e-5),
                "node": (47.1459444444444444, 1.185208333333333330, 1.73888888888888889e-4),
                "argp": (28.7537527777777778, 3.70280555555555556e-1, 1.20833333333333333e-4),
                "motion": (1.49472515288888889e5, 6.38888888888888889e-6), "L0": 1.02279380555555556e2},
    "venus": {"a": 0.72333160,
              "e": (0.006820690, -0.000047740, 0.0000000910),
              "i": (3.393630555555555560, 1.00583333333333333e-3, -9.72222222222222222e-7),
              "node": (75.7796472222222222, 8.9985e-1, 4.1e-4),
              "argp": (54.3841861111111111, 5.08186111111111111e-1, -1.38638888888888889e-3),
              "motion": (5.8517803875e4, 1.28605555555555556e-3), "L0": 2.12603219444444444e2},
    "earth": {"a": 1.000000230,
              "e": (0.016751040, -0.000041800, -0.0000001260),
              "i": (0.0,), "node": (0.0,),
              "argp": (1.01220833333333333e2, 1.7191750, 4.52777777777777778e-4, 3.33333333333333333e-6),
              "motion": (3.599904975e4, -1.50277777777777778e-4, -3.33333333333333333e-6), "L0": 3.58475844444444444e2},
    "mars": {"a": 1.5236883990,
             "e": (0.093312900, 0.0000920640, -0.0000000770),
             "i": (1.850333333333333330, -6.75e-4, 1.26111111111111111e-5),
             "node": (48.7864416666666667, 7.70991666666666667e-1, -1.38888888888888889e-6, -5.33333333333333333e-6),
             "argp": (285.431761111111111, 1.069766666666666670, 1.3125e-4, 4.13888888888888889e-6),
             "motion": (1.91398585e4, 1.80805555555555556e-4, 1.19444444444444444e-6), "L0": 3.19529425e2},
    "jupiter": {"a": 5.2025610,
                "e": (0.048334750, 0.000164180, -0.00000046760, -0.00000000170),
                "i": (1.308736111111111110, -5.69611111111111111e-3, 3.88888888888888889e-6),
                "node": (99.4433861111111111, 1.010530, 3.52222222222222222e-4, -8.51111111111111111e-6),
                "argp": (273.277541666666667, 5.99431666666666667e-1, 7.0405e-4, 5.07777777777777778e-6),
                "motion": (3.03469202388888889e3, -7.21588888888888889e-4, 1.78444444444444444e-6),
                "L0": 2.25328327777777778e2},
    "saturn": {"a": 9.5547470,
               "e": (0.055892320, -0.00034550, -0.0000007280, 0.000000000740),
               "i": (2.492519444444444440, -3.91888888888888889e-3, -1.54888888888888889e-5, 4.44444444444444444e-8),
               "node": (112.790388888888889, 8.73195138888888889e-1, -1.52180555555555556e-4,
                        -5.30555555555555556e-6),
               "argp": (338.307772222222222, 1.085220694444444440, 9.78541666666666667e-4, 9.91666666666666667e-6),
               "motion": (1.22155146777777778e3, -5.01819444444444444e-4, -5.19444444444444444e-6),
               "L0": 1.75466216666666667e2},
    "uranus": {"a": 19.218140,
               "e": (0.04634440, -0.000026580, 0.0000000770),
               "i": (7.72463888888888889e-1, 6.25277777777777778e-4, 3.95e-5),
               "node": (73.4770972222222222, 4.98667777777777778e-1, 1.31166666666666667e-3),
               "argp": (98.0715527777777778, 9.85765e-1, -1.07447222222222222e-3, -6.05555555555555556e-7),
               "motion": (4.28379113055555556e2, 7.88444444444444444e-5, 1.11111111111111111e-9),
               "L0": 7.26488194444444444e1},
    "neptune": {"a": 30.109570,
                "e": (0.008997040, 0.0000063300, -0.0000000020),
                "i": (1.779241666666666670, -9.54361111111111111e-3, -9.11111111111111111e-6),
                "node": (130.681358333333333, 1.0989350, 2.49866666666666667e-4, -4.71777777777777778e-6),
                "argp": (276.045966666666667, 3.25639444444444444e-1, 1.4095e-4, 4.11333333333333333e-6),
                "motion": (2.18461339722222222e2, -7.03333333333333333e-5), "L0": 3.77306694444444444e1},
}

#: Julian date of and centuries per unit of the ACT table's time argument: ``T`` runs from J1900.0. The toolbox
#: evaluates ``T = (mjd2000 + 36525)/36525``, and ``jd = mjd2000 + 2451544.5``, so its epoch is JD 2415019.5.
_ACT_EPOCH_JD = 2451544.5 - 36525.0
_ACT_CENTURY_D = 36525.0
_ACT_AU_KM = 149597870.66
#: The Sun's ``mu`` the ACT table's velocities are built with (``Pl_Eph_An.cpp``), km^3/s^2.
_ACT_MU_SUN = 1.32712428e11


class GTOPEphemeris:
    """The ESA ACT analytic planet elements -- the ephemeris the GTOP benchmark problems are stated in.

    :class:`AnalyticEphemeris` is the one to design with: it is the modern JPL table and it matches the DE kernel to
    the residuals in the guide. This class exists for one job, and it is a real one: the published optima of the GTOP
    problems were computed against *these* elements, so a number from this package can only be put next to 4.9307
    km/s if the planets are where the benchmark put them. Cassini1's published decision vector scores 4.9307 km/s
    here and 5.81 km/s with the JPL table (the positions differ by ~1.2e6 km at 1997-1999 epochs), so the two
    ephemerides are not interchangeable and a report must say which one it used -- ``.source`` does.

    The positions and velocities are heliocentric and in the ecliptic frame of the table, km and km/s.
    """

    source = "ESA ACT analytic elements (GTOP toolbox), the ephemeris the GTOP benchmarks are stated in"

    def __init__(self, bodies=DESIGN_BODIES):
        missing = set(bodies) - set(ACT_ELEMENTS)
        if missing:
            raise ValueError(f"no ACT elements for {sorted(missing)}")
        self.bodies = tuple(bodies)

    def state(self, body: str, jd_tdb):
        """Heliocentric position (km) and velocity (km/s) of ``body`` at ``jd_tdb``, ``(m, 3)`` each."""
        jd = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
        if body == "sun":
            return np.zeros((len(jd), 3)), np.zeros((len(jd), 3))
        out_r = np.empty((len(jd), 3))
        out_v = np.empty((len(jd), 3))
        for i, t in enumerate(jd):
            r, v = self._state_scalar(body, float(t))
            out_r[i] = r
            out_v[i] = v
        return out_r, out_v

    def _state_scalar(self, body: str, jd: float):
        """One epoch, plain floats: the perifocal ellipse rotated into the table's ecliptic."""
        T = (jd - _ACT_EPOCH_JD) / _ACT_CENTURY_D
        a_au, e, inc_deg, node_deg, argp_deg, mean_deg = _act_elements(body, T)
        a = a_au * _ACT_AU_KM
        inc, node, omega = math.radians(inc_deg), math.radians(node_deg), math.radians(argp_deg)
        M = math.fmod(math.radians(mean_deg), 2.0 * math.pi)
        # Newton on Kepler's equation from the ACT's own starter, to its own tolerance
        E = M + e * math.cos(M)
        for _ in range(100):
            step = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
            E -= step
            if abs(step) <= 1e-13:
                break
        b = a * math.sqrt(max(1.0 - e * e, 0.0))
        n = math.sqrt(_ACT_MU_SUN / (a * a * a))
        cE, sE = math.cos(E), math.sin(E)
        xp, yp = a * (cE - e), b * sE
        one = 1.0 - e * cE
        vxp, vyp = -a * n * sE / one, b * n * cE / one
        cO, sO = math.cos(node), math.sin(node)
        co, so = math.cos(omega), math.sin(omega)
        ci, si = math.cos(inc), math.sin(inc)
        rot = ((cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si),
               (sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si),
               (so * si, co * si, ci))
        r = tuple(rot[j][0] * xp + rot[j][1] * yp for j in range(3))
        v = tuple(rot[j][0] * vxp + rot[j][1] * vyp for j in range(3))
        return r, v


@lru_cache(maxsize=2)
def default_ephemeris() -> AnalyticEphemeris | KernelEphemeris:
    """The kernel when one is installed, the analytic elements otherwise; ``.source`` says which."""
    try:
        return KernelEphemeris()
    except (FileNotFoundError, ImportError, OSError):
        return AnalyticEphemeris()


def body_radius_km(body: str) -> float:
    return RADIUS_KM[body]


def body_gm(body: str) -> float:
    return GM[body]
