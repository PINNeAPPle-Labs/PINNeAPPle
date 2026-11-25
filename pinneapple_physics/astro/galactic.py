"""Galactic dynamics: an orbit in a mass model, through galpy (#522).

Where :mod:`pinneapple_physics.astro.orbits` propagates a Keplerian orbit around one body, this module integrates a
trajectory through a whole galaxy's mass distribution -- a disc, a bulge and a dark halo -- which is what a rotation
curve, a stream or a dark-matter fit is computed from.

    from pinneapple_physics.astro import galactic

    path = galactic.integrate_orbit(xyz=(8.0, 0.0, 0.02), vxyz=(0.0, 220.0, 0.0), t_end=2.0)
    path.checks          # energy and angular-momentum drift over the 2 Gyr: the integrator's own error, stated
    sample = path.to_sample()                       # a PhysicalSample: t, x, y, z, vx, vy, vz with units
    galactic.rotation_curve("mw2014", radii_kpc=(1, 2, 4, 8, 16))

The mass model is galpy's (Bovy 2015, ApJS 216, 29, BSD-3-Clause), used in its natural units -- lengths in ``ro``
(8 kpc) and velocities in ``vo`` (220 km/s) -- so a mass enters a component as ``amp = G M / (ro vo^2)``; the
conversion is written out in :data:`G_KPC_KMS2_PER_MSUN` rather than hidden in a fitted number.

Two models are offered, and nothing else is pretended:

``mw2014``
    galpy's ``MWPotential2014``: a power-law bulge with a cutoff, a Miyamoto-Nagai disc and an NFW halo, fitted to
    rotation-curve and other Milky Way data by Bovy (2015). This one carries the paper's own parameters.
``disc_bulge_halo``
    the same three components spelled out with masses and scale lengths in physical units, so a fit can move them:
    ``M_disc = 6.8e10 Msun``, ``a = 3.0 kpc``, ``b = 0.28 kpc``; ``M_bulge = 0.5e10 Msun``, ``a = 0.6 kpc``;
    ``M_halo`` with a scale radius of 16 kpc. These are the numbers galpy's own model uses, expressed in Msun and kpc.

An orbit is reported with its checks, not on its own: the relative drift of the energy and of the angular momentum
``Lz`` over the integration say whether the time step was fine enough to trust the trajectory at all.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GALPY_INSTALL_HINT",
    "G_KPC_KMS2_PER_MSUN",
    "POTENTIALS",
    "MassModel",
    "OrbitPath",
    "available",
    "integrate_orbit",
    "potential",
    "rotation_curve",
]

GALPY_INSTALL_HINT = 'galactic dynamics needs galpy: pip install galpy (or pip install "pinneapple[astro]")'

#: ``G`` in kpc km^2 s^-2 Msun^-1. galpy's potentials take ``G = 1`` with lengths in ``ro`` and velocities in ``vo``,
#: so a mass in Msun becomes ``amp = G * M / (ro * vo**2)``. Cross-check: galpy's own Miyamoto-Nagai disc in
#: ``MWPotential2014`` has ``amp = 0.7574802`` at ``a = 3 kpc``, which this conversion turns back into 6.82e10 Msun.
G_KPC_KMS2_PER_MSUN = 4.300917270e-6


def _galpy():
    """``(galpy.potential, galpy.orbit.Orbit)``, or an ImportError that says how to get them."""
    try:
        from galpy import potential as gpot
        from galpy.orbit import Orbit
    except Exception as exc:                                     # ImportError, or a broken install of its own deps
        raise ImportError(GALPY_INSTALL_HINT) from exc
    return gpot, Orbit


def available() -> tuple[bool, str]:
    """Whether galpy can be imported, and the reason when it cannot (the sentence a caller can print)."""
    try:
        import galpy  # noqa: F401
    except Exception as exc:
        return False, f"{GALPY_INSTALL_HINT} ({type(exc).__name__}: {exc})"
    return True, ""


def _amp(mass_msun: float, ro: float, vo: float) -> float:
    """A mass in Msun as a galpy amplitude (``G = 1``, lengths in ``ro``, velocities in ``vo``)."""
    return G_KPC_KMS2_PER_MSUN * float(mass_msun) / (float(vo) ** 2 * float(ro))


def _mw2014(ro: float = 8.0, vo: float = 220.0):
    gpot, _ = _galpy()
    return gpot.MWPotential2014


def _disc_bulge_halo(ro: float = 8.0, vo: float = 220.0):
    """A Miyamoto-Nagai disc, a Hernquist bulge and an NFW halo, with the parameters galpy's own model uses."""
    gpot, _ = _galpy()
    return gpot.MiyamotoNagaiPotential(amp=_amp(6.82e10, ro, vo), a=3.0 / ro, b=0.28 / ro) + \
        gpot.HernquistPotential(amp=_amp(0.5e10, ro, vo), a=0.6 / ro) + \
        gpot.NFWPotential(amp=_amp(1.0e12, ro, vo), a=16.0 / ro)


@dataclass(frozen=True)
class MassModel:
    """One galactic mass model: what it is, where the numbers come from, and how to build the galpy object."""

    name: str
    citation: str
    build: Callable[..., Any]

    def __call__(self, ro: float = 8.0, vo: float = 220.0):
        return self.build(ro, vo)


POTENTIALS: dict[str, MassModel] = {
    "mw2014": MassModel("mw2014", "galpy's MWPotential2014 (Bovy 2015, ApJS 216, 29): bulge + disc + NFW halo",
                        _mw2014),
    "disc_bulge_halo": MassModel(
        "disc_bulge_halo",
        "Miyamoto-Nagai disc (6.82e10 Msun, a=3.0 kpc, b=0.28 kpc) + Hernquist bulge (0.5e10 Msun, a=0.6 kpc) + "
        "NFW halo (1e12 Msun, a=16 kpc): the components of Bovy (2015) written out in physical units",
        _disc_bulge_halo),
}


def potential(name: str = "mw2014", *, ro: float = 8.0, vo: float = 220.0):
    """The galpy potential of a named model. ``KeyError`` for a name that is not in :data:`POTENTIALS`."""
    if name not in POTENTIALS:
        raise KeyError(f"unknown mass model {name!r}; known: {sorted(POTENTIALS)}")
    return POTENTIALS[name](ro, vo)


def _cylindrical(xyz: Sequence[float], vxyz: Sequence[float]) -> list[float]:
    """Cartesian ``(x, y, z, vx, vy, vz)`` in kpc and km/s as galpy's ``[R, vR, vT, z, vz, phi]``, in natural units.

    galpy's ``vxvv`` is cylindrical, not cartesian: the first three numbers are ``R, v_R, v_phi``. Handing it a
    cartesian triple silently integrates a different orbit -- an ``x = 8, vx = 0, vy = 220`` setup becomes ``R = 8,
    v_R = 0`` in both conventions, so the mistake survives the obvious test and only shows up in ``z`` and ``phi``.
    The conversion is therefore done here, once, and named.
    """
    x, y, z = (float(v) for v in xyz)
    vx, vy, vz = (float(v) for v in vxyz)
    r = float(np.hypot(x, y))
    if r == 0.0:
        raise ValueError("x = y = 0 leaves the azimuth undefined; give the orbit a non-zero cylindrical radius")
    phi = float(np.arctan2(y, x))
    v_r = (x * vx + y * vy) / r
    v_t = (x * vy - y * vx) / r
    return [r, v_r, v_t, z, vz, phi]


@dataclass
class OrbitPath:
    """An integrated trajectory, with the checks that say whether the integration can be trusted.

    Everything is in physical units: ``t`` in Gyr, positions in kpc, velocities in km/s. ``checks`` holds the relative
    drift of the energy and of ``Lz`` between the first sample and the worst one -- two quantities a symplectic
    integrator must keep, so a drift is the integrator's step being too coarse, not the galaxy being interesting.
    """

    t_gyr: np.ndarray
    xyz_kpc: np.ndarray                    # (n, 3)
    vxyz_kms: np.ndarray                   # (n, 3)
    model: str
    ro: float
    vo: float
    galpy_version: str
    checks: dict[str, float]

    @property
    def x(self) -> np.ndarray:
        return self.xyz_kpc[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz_kpc[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz_kpc[:, 2]

    @property
    def radius_kpc(self) -> np.ndarray:
        """Cylindrical radius in the plane, ``hypot(x, y)``: the quantity a rotation curve is plotted against."""
        return np.hypot(self.x, self.y)

    def to_sample(self):
        """The orbit as a ``PhysicalSample``: one variable per column, units in ``schema["units"]``.

        The trail is a trajectory, so ``domain["type"]`` is ``"points"`` (the rows are samples of a path, not cells of
        a grid), and ``provenance`` carries the model, its citation and the conservation checks -- a reader who gets
        this sample alone can say which galaxy model produced it and how well it was integrated.
        """
        import xarray as xr

        from pinneapple_data.physical_sample import PhysicalSample

        t = np.asarray(self.t_gyr, float)
        data = {
            "x": (("t",), self.xyz_kpc[:, 0], {"units": "kpc"}),
            "y": (("t",), self.xyz_kpc[:, 1], {"units": "kpc"}),
            "z": (("t",), self.xyz_kpc[:, 2], {"units": "kpc"}),
            "vx": (("t",), self.vxyz_kms[:, 0], {"units": "km/s"}),
            "vy": (("t",), self.vxyz_kms[:, 1], {"units": "km/s"}),
            "vz": (("t",), self.vxyz_kms[:, 2], {"units": "km/s"}),
        }
        ds = xr.Dataset(data, coords={"t": ("t", t, {"units": "Gyr"})})
        units = {k: "kpc" if k in ("x", "y", "z") else "km/s" for k in ("x", "y", "z", "vx", "vy", "vz")}
        units["t"] = "Gyr"
        return PhysicalSample(
            state=ds,
            schema={"units": units,
                    "variables": {k: {"unit": u, "kind": "coordinate" if k == "t" else "field"}
                                  for k, u in units.items()}},
            domain={"type": "points", "n": int(t.size), "kind": "trajectory"},
            provenance={"reader": "galpy", "galpy_version": self.galpy_version, "model": self.model,
                        "model_citation": POTENTIALS[self.model].citation if self.model in POTENTIALS else "",
                        "ro_kpc": self.ro, "vo_kms": self.vo, "checks": dict(self.checks),
                        "integration": "galpy Orbit.integrate over the times given", "units": "physical (kpc, km/s, Gyr)"},
            extras={"checks": dict(self.checks)},
        )


def integrate_orbit(
    *,
    xyz: Sequence[float] = (8.0, 0.0, 0.02),
    vxyz: Sequence[float] = (0.0, 220.0, 0.0),
    t_end: float = 2.0,
    n: int = 201,
    model: str = "mw2014",
    ro: float = 8.0,
    vo: float = 220.0,
    method: str = "symplec4_c",
) -> OrbitPath:
    """Integrate one orbit in a galactic mass model and return the path with its conservation checks.

    xyz      : start position in kpc (cartesian, Sun-centred or Galactic -- the model is the same either way).
    vxyz     : start velocity in km/s.
    t_end, n : the orbit is sampled at ``n`` times from 0 to ``t_end`` Gyr.
    model    : a name in :data:`POTENTIALS` (``"mw2014"`` or ``"disc_bulge_halo"``).
    ro, vo   : the model's length and velocity scales (kpc, km/s); they set how a mass becomes a galpy amplitude.
    method   : galpy's integrator; the symplectic one is the default because it conserves energy by construction.

    >>> path = integrate_orbit(t_end=1.0)             # doctest: +SKIP  (needs galpy)
    >>> path.checks["energy_drift"] < 1e-6            # doctest: +SKIP
    True
    """
    if t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    if n < 2:
        raise ValueError(f"n must be at least 2 samples, got {n}")
    _gpot, Orbit = _galpy()
    pot = potential(model, ro=ro, vo=vo)
    start = _cylindrical(xyz, vxyz)
    natural = [start[0] / ro, start[1] / vo, start[2] / vo, start[3] / ro, start[4] / vo, start[5]]
    orbit = Orbit(vxvv=natural, ro=ro, vo=vo)
    # Everything below is galpy's natural units, converted here and not by galpy: its "physical output" converts the
    # *values* an orbit returns but not the times it is integrated at (measured on galpy 1.12.1 -- 0.2285 is taken as
    # 0.2285 natural time units, 8.3 Myr, not 0.2285 Gyr), so asking galpy for Gyr silently integrates a 27x shorter
    # orbit. One natural length is ro, one natural velocity vo, one natural time ro/vo.
    orbit.turn_physical_off()
    time_unit_gyr = float(ro) / float(vo)
    times_nat = np.linspace(0.0, float(t_end) / time_unit_gyr, int(n))
    orbit.integrate(times_nat, pot, method=method)
    xyz_kpc = np.column_stack([np.asarray(orbit.x(times_nat)) * ro,
                               np.asarray(orbit.y(times_nat)) * ro,
                               np.asarray(orbit.z(times_nat)) * ro]).astype(float)
    vxyz_kms = np.column_stack([np.asarray(orbit.vx(times_nat)) * vo,
                                np.asarray(orbit.vy(times_nat)) * vo,
                                np.asarray(orbit.vz(times_nat)) * vo]).astype(float)
    energy = np.asarray(orbit.E(times_nat), dtype=float)
    lz = np.asarray(orbit.Lz(times_nat), dtype=float)

    def _drift(values: np.ndarray) -> float:
        if values.size == 0 or values[0] == 0.0:
            return float("nan")
        return float(np.nanmax(np.abs(values - values[0])) / abs(values[0]))

    checks = {"energy_drift": _drift(energy), "lz_drift": _drift(lz),
              "energy": float(energy[0]) if energy.size else float("nan"),
              "lz_kpc_kms": float(lz[0]) * float(ro) * float(vo),   # natural Lz -> kpc km/s
              "max_radius_kpc": float(np.nanmax(np.hypot(xyz_kpc[:, 0], xyz_kpc[:, 1]))),
              "min_z_kpc": float(np.nanmin(xyz_kpc[:, 2])), "max_z_kpc": float(np.nanmax(xyz_kpc[:, 2])),
              "samples": int(n), "t_end_gyr": float(t_end), "integrator": method}
    try:
        import galpy
        version = str(galpy.__version__)
    except Exception:                                            # pragma: no cover - galpy was just imported
        version = ""
    return OrbitPath(t_gyr=np.linspace(0.0, float(t_end), int(n)), xyz_kpc=xyz_kpc, vxyz_kms=vxyz_kms,
                     model=str(model), ro=float(ro), vo=float(vo), galpy_version=version, checks=checks)


def rotation_curve(model: str = "mw2014", radii_kpc: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                   *, ro: float = 8.0, vo: float = 220.0) -> dict[str, np.ndarray]:
    """The circular-speed curve of a mass model at the radii given, in kpc and km/s.

    ``v_circ(r)`` is the model's own prediction, the quantity a rotation-curve fit compares with a measurement, so it
    is what an inference over ``POTENTIALS`` moves. Returned as two arrays rather than one, so the radii and the speeds
    cannot be paired up wrongly downstream.
    """
    gpot, _ = _galpy()
    pot = potential(model, ro=ro, vo=vo)
    r = np.asarray(list(radii_kpc), dtype=float)
    if np.any(r <= 0):
        raise ValueError("radii must be positive")
    v = np.asarray(gpot.vcirc(pot, r / ro), dtype=float) * vo
    return {"r_kpc": r, "v_circ_kms": v}
