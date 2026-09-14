"""Real regression tests -- reproduces ridetune_ai's own baseline
numbers exactly, and its confirmed real finding that fatigue life is
exactly 2x more sensitive to thickness than to width."""
from __future__ import annotations

from pinneapple_physics.closed_form import cantilever_fatigue as cf

ARM_LENGTH_M = 0.35
FATIGUE_STRENGTH_COEFF_PA = 1200e6
FATIGUE_STRENGTH_EXPONENT = -0.09


def test_root_stress_matches_ridetune_ai_baseline():
    """ridetune_ai's baseline arm (width=0.05m, thickness=0.022m),
    length=0.35m, load=1200N: sigma = 6*F*L/(w*t^2)."""
    sigma = cf.cantilever_root_bending_stress_pa(0.05, 0.022, 1200.0, ARM_LENGTH_M)
    expected = 6.0 * 1200.0 * ARM_LENGTH_M / (0.05 * 0.022 ** 2)
    assert abs(sigma - expected) < 1.0  # Pa, effectively exact


def test_position_profile_matches_root_at_x0_and_zero_at_tip():
    sigma_root = cf.cantilever_bending_stress_at_position_pa(0.05, 0.022, 1200.0, ARM_LENGTH_M, 0.0)
    sigma_scalar = cf.cantilever_root_bending_stress_pa(0.05, 0.022, 1200.0, ARM_LENGTH_M)
    sigma_tip = cf.cantilever_bending_stress_at_position_pa(0.05, 0.022, 1200.0, ARM_LENGTH_M, ARM_LENGTH_M)
    assert abs(sigma_root - sigma_scalar) < 1.0
    assert sigma_tip == 0.0


def test_thickness_is_exactly_twice_as_sensitive_as_width_to_fatigue_life():
    """Real confirmed finding from ridetune_ai's sensitivity-analysis
    feature: because life ~ (w*t^2)^(1/b) under this power-law model,
    the elasticity of life w.r.t. thickness is exactly 2x that of
    width. Checked here via finite differences on the ported formula."""
    def life(w, t):
        sigma = cf.cantilever_root_bending_stress_pa(w, t, 1200.0, ARM_LENGTH_M)
        return cf.basquin_fatigue_life_cycles(sigma, FATIGUE_STRENGTH_COEFF_PA, FATIGUE_STRENGTH_EXPONENT)

    w0, t0, h = 0.05, 0.022, 1e-6
    d_life_d_w = (life(w0 + h, t0) - life(w0 - h, t0)) / (2 * h)
    d_life_d_t = (life(w0, t0 + h) - life(w0, t0 - h)) / (2 * h)
    life0 = life(w0, t0)
    elasticity_w = d_life_d_w * w0 / life0
    elasticity_t = d_life_d_t * t0 / life0
    assert abs(elasticity_t / elasticity_w - 2.0) < 1e-3


def test_allowable_stress_round_trips_with_fatigue_life():
    target_life = 1e6
    sigma_allow = cf.basquin_allowable_stress_pa(target_life, FATIGUE_STRENGTH_COEFF_PA, FATIGUE_STRENGTH_EXPONENT)
    life_back = cf.basquin_fatigue_life_cycles(sigma_allow, FATIGUE_STRENGTH_COEFF_PA, FATIGUE_STRENGTH_EXPONENT)
    assert abs(life_back - target_life) / target_life < 1e-6
