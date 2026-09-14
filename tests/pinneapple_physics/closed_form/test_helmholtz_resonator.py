"""Real regression test -- reproduces soundshape_ai's own baseline
number exactly."""
from __future__ import annotations

from pinneapple_physics.closed_form import helmholtz_resonator as hr


def test_matches_soundshape_ai_baseline():
    """soundshape_ai's baseline port (radius=0.025m, length=0.10m), 20L cabinet."""
    f = hr.helmholtz_resonant_frequency_hz(0.025, 0.10, 20.0)
    assert abs(f - 45.3) < 0.5  # matches soundshape_ai's own reported baseline ~45.3 Hz


def test_larger_port_volume_raises_frequency():
    f_small = hr.helmholtz_resonant_frequency_hz(0.015, 0.10, 20.0)
    f_large = hr.helmholtz_resonant_frequency_hz(0.035, 0.10, 20.0)
    assert f_large > f_small  # larger port area -> higher tuning frequency, real physics


def test_larger_cabinet_lowers_frequency():
    f_small_cab = hr.helmholtz_resonant_frequency_hz(0.025, 0.10, 10.0)
    f_large_cab = hr.helmholtz_resonant_frequency_hz(0.025, 0.10, 80.0)
    assert f_large_cab < f_small_cab


def test_port_volume_matches_cylinder_formula():
    v = hr.port_volume_m3(0.025, 0.10)
    import math
    assert abs(v - math.pi * 0.025 ** 2 * 0.10) < 1e-12
