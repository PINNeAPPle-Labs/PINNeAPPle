"""Galactic dynamics through galpy (#522): an orbit in a mass model, its checks, and the curve it predicts.

galpy is optional and it is probed in a **fresh process** before anything is imported here: on a machine where the
OpenMP runtime is already loaded by another library (torch and galpy on macOS is the case that was measured), the
import aborts the interpreter instead of raising, and an abort cannot be caught -- a probe that returns a non-zero
status is the only way to skip instead of killing the test session. The reason is printed, never swallowed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinneapple_physics.astro import galactic

REPO = Path(__file__).resolve().parents[1]


def _galpy_or_skip() -> None:
    ok, why = galactic.available()
    if not ok:
        pytest.skip(why)
    probe = subprocess.run([sys.executable, "-c", "import galpy"], capture_output=True, text=True, timeout=600)
    if probe.returncode != 0:
        pytest.skip(f"galpy cannot be loaded in a fresh process here (exit {probe.returncode}): "
                    f"{(probe.stderr or probe.stdout).strip()[-400:]}")


def test_without_galpy_the_install_line_is_what_comes_out(monkeypatch):
    for name in [m for m in sys.modules if m == "galpy" or m.startswith("galpy.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "galpy", None)
    with pytest.raises(ImportError, match="pip install galpy"):
        galactic.integrate_orbit(t_end=1.0)
    ok, why = galactic.available()
    assert not ok and "pip install galpy" in why


def test_an_unknown_mass_model_is_refused_by_name():
    with pytest.raises(KeyError, match="unknown mass model"):
        galactic.potential("andromeda")
    with pytest.raises(KeyError, match="unknown mass model"):
        galactic.rotation_curve("andromeda")


def test_a_start_on_the_axis_has_no_azimuth_and_is_refused():
    _galpy_or_skip()
    with pytest.raises(ValueError, match="azimuth"):
        galactic.integrate_orbit(xyz=(0.0, 0.0, 0.0), vxyz=(0.0, 220.0, 0.0))


def test_the_orbit_conserves_energy_and_angular_momentum():
    """The check a trajectory has to pass before it is reported: a symplectic integrator on a static potential."""
    _galpy_or_skip()
    path = galactic.integrate_orbit(t_end=2.0, n=401)
    assert path.checks["integrator"] == "symplec4_c"
    assert path.checks["energy_drift"] < 1e-6
    assert path.checks["lz_drift"] < 1e-6
    assert path.checks["samples"] == 401 and path.checks["t_end_gyr"] == 2.0
    assert path.t_gyr.size == 401 and path.xyz_kpc.shape == (401, 3)


def test_the_orbit_is_a_physical_trajectory_not_a_scaled_one():
    """A circular orbit at 8 kpc with 220 km/s must come back round: the radius stays, the star goes round once."""
    _galpy_or_skip()
    path = galactic.integrate_orbit(xyz=(8.0, 0.0, 0.0), vxyz=(0.0, 220.0, 0.0), t_end=1.0, n=201)
    assert np.allclose(path.radius_kpc, 8.0, atol=5e-3)
    assert np.allclose(path.z, 0.0, atol=1e-6)
    assert path.radius_kpc[-1] == pytest.approx(8.0, abs=5e-3)


def test_the_time_axis_is_gyr_checked_against_the_azimuthal_period():
    """A circular orbit has a period of ``2 pi R / v``: 0.2285 Gyr at 8 kpc and 220 km/s.

    This is the check that catches a unit slip in the integration times -- galpy's "physical output" converts the
    values an orbit returns but not the times it is integrated at (a 27x shorter orbit, silently), so the conversion is
    done in :func:`integrate_orbit` and this test is what keeps it honest.
    """
    _galpy_or_skip()
    period = 2.0 * np.pi * 8.0 / 220.0                            # kpc / (km/s) = Gyr
    path = galactic.integrate_orbit(xyz=(8.0, 0.0, 0.0), vxyz=(0.0, 220.0, 0.0), t_end=period, n=3)
    assert path.t_gyr[-1] == pytest.approx(period)
    assert path.x[-1] == pytest.approx(8.0, abs=0.05)             # back where it started, one revolution later
    assert path.y[-1] == pytest.approx(0.0, abs=0.05)
    half = galactic.integrate_orbit(xyz=(8.0, 0.0, 0.0), vxyz=(0.0, 220.0, 0.0), t_end=period / 2.0, n=3)
    assert half.x[-1] == pytest.approx(-8.0, abs=0.05)            # and on the other side half a period later


def test_a_vertical_launch_oscillates_through_the_disc():
    """Started 1 kpc above the plane with no vertical speed, a star must fall through it and come back: the vertical
    epicyclic oscillation, which conserves both energy and ``Lz`` (nothing in this model breaks the axisymmetry)."""
    _galpy_or_skip()
    path = galactic.integrate_orbit(xyz=(8.0, 0.0, 1.0), vxyz=(0.0, 220.0, 0.0), t_end=1.0, n=401)
    assert path.checks["min_z_kpc"] < 0.0 < path.checks["max_z_kpc"]   # it crossed the plane, so it oscillates
    assert abs(path.checks["max_z_kpc"] - 1.0) < 0.1              # and came back to about where it started
    assert path.radius_kpc.min() >= 8.0 - 1e-6 and path.radius_kpc.max() < 8.5   # the vertical motion couples to R
    assert path.checks["energy_drift"] < 1e-6
    assert path.checks["lz_drift"] < 1e-6


def test_the_rotation_curve_of_the_published_model_is_vo_at_ro():
    """galpy normalises ``MWPotential2014`` so that ``v_circ(ro) = vo``: a one-number check of the whole chain."""
    _galpy_or_skip()
    curve = galactic.rotation_curve("mw2014", radii_kpc=(8.0,))
    assert curve["v_circ_kms"][0] == pytest.approx(220.0, rel=1e-6)
    wide = galactic.rotation_curve("mw2014")
    assert wide["r_kpc"].size == wide["v_circ_kms"].size == 6
    assert np.all(wide["v_circ_kms"] > 0)


def test_the_second_model_curve_differs_from_the_published_one_and_says_so():
    _galpy_or_skip()
    a = galactic.rotation_curve("mw2014", radii_kpc=(8.0, 16.0))["v_circ_kms"]
    b = galactic.rotation_curve("disc_bulge_halo", radii_kpc=(8.0, 16.0))["v_circ_kms"]
    assert b[0] > a[0]                                            # a heavier, more extended halo
    assert "Bovy" in galactic.POTENTIALS["mw2014"].citation


def test_the_result_is_exposed_as_a_sample_with_units():
    """The orbit reaches the rest of the library the same way every other result does: as a ``PhysicalSample``."""
    _galpy_or_skip()
    path = galactic.integrate_orbit(t_end=1.0, n=51)
    sample = path.to_sample()
    assert sample.domain["type"] == "points"
    assert sample.provenance["model"] == "mw2014"
    assert sample.provenance["galpy_version"] == path.galpy_version
    assert sample.provenance["checks"]["energy_drift"] == path.checks["energy_drift"]
    assert sample.schema["units"]["t"] == "Gyr" and sample.schema["units"]["vx"] == "km/s"
    assert sample.state["x"].attrs["units"] == "kpc"
    np.testing.assert_allclose(sample.state["x"].values, path.x)
    np.testing.assert_allclose(sample.state["vy"].values, path.vxyz_kms[:, 1])

    astropy = pytest.importorskip("astropy")
    from astropy import units as u

    from pinneapple_data.astro import units as pp_units

    vx = pp_units.quantity_of(sample, "vx")
    assert vx.unit == u.km / u.s
    assert astropy.units.Quantity(vx.value, vx.unit).to(u.m / u.s).value[0] == pytest.approx(vx.value[0] * 1000.0)
