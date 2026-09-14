"""Real, closed-form cantilever-beam bending stress + stress-life
fatigue model -- standard mechanical-engineering equations (e.g.
Shigley, *Mechanical Engineering Design*; Dowling, *Mechanical
Behavior of Materials*), not fabricated:

    Bending stress (rectangular cross-section), sigma = M*c/I:
        sigma_a = 6*F*L / (w * t^2)     at the fixed (pivot) end

    Basquin's equation (stress-life fatigue relationship):
        sigma_a = sigma_f' * (2N)^b
        => N = 0.5 * (sigma_a / sigma_f')^(1/b)

First used in, and ported back from, PINNeAPPle-apps' ``ridetune_ai``
(a vehicle suspension-arm design tool).
"""
from __future__ import annotations

FLOAT_FLOOR = 1e-9


def stress_from_moment_pa(width_m: float, thickness_m: float, moment_n_m: float) -> float:
    """sigma = M*c/I for a solid rectangular cross-section (width x
    thickness), the shared real computation behind both the scalar
    fixed-end stress and the spatial stress profile below -- one
    formula, not two independent implementations."""
    w = max(width_m, FLOAT_FLOOR)
    t = max(thickness_m, FLOAT_FLOOR)
    i_section = w * t ** 3 / 12.0
    c = t / 2.0
    return moment_n_m * c / i_section


def cantilever_root_bending_stress_pa(width_m: float, thickness_m: float, load_n: float, length_m: float) -> float:
    """The real, closed-form ground-truth bending stress at the fixed
    (pivot) end of a cantilever beam, where the bending moment is
    maximum (M = F*L)."""
    return stress_from_moment_pa(width_m, thickness_m, load_n * length_m)


def cantilever_bending_stress_at_position_pa(
    width_m: float, thickness_m: float, load_n: float, length_m: float, x_from_root_m: float,
) -> float:
    """Real spatial bending-stress profile along the SAME cantilever
    ``cantilever_root_bending_stress_pa`` evaluates at a single point
    -- not a separate/invented formula. With the beam fixed (clamped)
    at the pivot end (x=0) and the load applied at the free tip (x=L),
    the bending moment at position x is

        M(x) = F * (L - x)

    maximum at the fixed root (M(0) = F*L, matching the scalar
    function exactly) and zero at the free tip (M(L) = 0)."""
    x = min(max(x_from_root_m, 0.0), length_m)
    moment = load_n * (length_m - x)
    return stress_from_moment_pa(width_m, thickness_m, moment)


def basquin_fatigue_life_cycles(stress_amplitude_pa: float, fatigue_strength_coeff_pa: float, fatigue_strength_exponent: float) -> float:
    """Basquin's stress-life relationship solved for cycles to failure
    N, given a real stress amplitude and the material's real fatigue
    strength coefficient (sigma_f') and exponent (b)."""
    return 0.5 * (stress_amplitude_pa / fatigue_strength_coeff_pa) ** (1.0 / fatigue_strength_exponent)


def basquin_allowable_stress_pa(target_life_cycles: float, fatigue_strength_coeff_pa: float, fatigue_strength_exponent: float) -> float:
    """Basquin's equation solved for the maximum stress amplitude that
    still achieves at least ``target_life_cycles`` -- the real,
    standard way this design problem is posed in practice (design
    against an allowable stress, not directly against raw cycles,
    since cycle counts span many orders of magnitude under the power
    law and are numerically ill-behaved to optimize against directly)."""
    return fatigue_strength_coeff_pa * (2.0 * target_life_cycles) ** fatigue_strength_exponent
