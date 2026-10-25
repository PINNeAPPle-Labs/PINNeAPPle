"""pinneapple_physics.closed_form.thin_airfoil: thin-airfoil lift and Hoerner profile drag."""
from __future__ import annotations

import math

import pytest

from pinneapple_physics.closed_form.thin_airfoil import (
    hoerner_profile_cd,
    naca4_camber_slope,
    thin_airfoil_cl,
    turbulent_flat_plate_cf,
)


def test_symmetric_section_is_two_pi_alpha():
    assert thin_airfoil_cl(0.0, 0.4, 4.0) == pytest.approx(2 * math.pi * math.radians(4.0), rel=1e-12)


def test_cambered_lift_matches_zero_lift_angle_of_the_presets():
    """Cl = 2 pi (alpha - alpha_L0) with alpha_L0 from the independent quadrature in pinneapple_presets (NACA 2412:
    alpha_L0 = -2.077 deg, the worked example of Anderson, Fundamentals of Aerodynamics, ch. 4)."""
    from pinneapple_presets.cases.airfoil import thin_airfoil_zero_lift

    a_l0 = thin_airfoil_zero_lift("2412")
    assert a_l0 == pytest.approx(-2.077, abs=0.01)
    for alpha in (-2.0, 0.0, 4.0, 8.0):
        assert thin_airfoil_cl(0.02, 0.4, alpha) == pytest.approx(2 * math.pi * math.radians(alpha - a_l0), abs=2e-4)


def test_lift_grows_with_camber_and_slope_changes_sign_at_max_camber():
    assert thin_airfoil_cl(0.04, 0.4, 4.0) > thin_airfoil_cl(0.0, 0.4, 4.0)
    s = naca4_camber_slope([0.1, 0.4, 0.9], 0.02, 0.4)
    assert s[0] > 0 and s[1] == 0 and s[2] < 0


def test_profile_drag_is_two_skin_frictions_at_zero_thickness_and_grows_with_thickness():
    re = 3e6
    assert hoerner_profile_cd(0.0, re) == pytest.approx(2 * 0.074 / re**0.2)
    assert hoerner_profile_cd(0.18, re) > hoerner_profile_cd(0.08, re)
    assert hoerner_profile_cd(0.12, 1e7) < hoerner_profile_cd(0.12, 1e6)
    with pytest.raises(ValueError):
        turbulent_flat_plate_cf(0.0)
