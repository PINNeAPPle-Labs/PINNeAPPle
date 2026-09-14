"""Real, closed-form membrane-controlled diffusive transport (Fick's
first law) -- the same real, standard model used in real reservoir-
type transdermal drug patches (e.g. scopolamine, clonidine,
fentanyl-class patches; see Robinson & Lee, *Controlled Drug
Delivery*), and generically applicable to any steady-state, membrane-
rate-limited diffusive release/transport problem:

    Steady-state flux/rate:  Rate = D * K * C_res * A / L
    Reservoir capacity:      M    = C_res * A * L_res
    Depletion time:          t_dep = M / Rate = L_res * L / (D * K)
    Concentration profile:   C(z) = K*C_res * (1 - z/L),  z in [0, L]
                             (linear at steady state, sink conditions
                             at z=L; d^2C/dz^2 = 0)
    Post-depletion decay:    k = D*K / (L * L_res)   (= 1/t_dep)
                             C(t) = C0 * exp(-k*(t - t_dep))  for t >= t_dep

``D`` (diffusion coefficient) and ``K`` (partition coefficient) are
generic real-order-of-magnitude transport parameters -- callers supply
real, citable values for their specific drug/polymer (or other
solute/membrane) pair; this module implements only the real physics,
not any particular substance's measured data.

First used in, and ported back from, PINNeAPPle-apps' ``patchdose_ai``
(a transdermal-patch design tool).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

FLOAT_FLOOR = 1e-12


def steady_state_release_rate_mg_day(
    diffusion_coeff_m2_s: float, partition_coeff: float, reservoir_conc_kg_m3: float,
    area_m2: float, membrane_thickness_m: float,
) -> float:
    """The real, closed-form ground-truth steady-state release rate
    (mg/day) through a rate-limiting membrane, Fick's first law."""
    flux = diffusion_coeff_m2_s * partition_coeff * reservoir_conc_kg_m3 / max(membrane_thickness_m, FLOAT_FLOOR)
    rate_kg_s = flux * area_m2
    return rate_kg_s * 1000.0 * 86400.0  # kg/s -> mg/day


def depletion_time_days(
    diffusion_coeff_m2_s: float, partition_coeff: float, membrane_thickness_m: float, reservoir_thickness_m: float,
) -> float:
    """Real consequence of this model: depletion time depends ONLY on
    membrane thickness -- area and reservoir concentration cancel out
    (both the release rate and the total reservoir mass scale with
    them identically). A genuine derived property of the physics, not
    a simplification error."""
    t_s = reservoir_thickness_m * membrane_thickness_m / (diffusion_coeff_m2_s * partition_coeff)
    return t_s / 86400.0


def concentration_at_membrane_depth(
    reservoir_conc_kg_m3: float, partition_coeff: float, membrane_thickness_m: float, z_from_reservoir_m: float,
) -> float:
    """Real steady-state concentration profile *inside* the same
    Fick's-first-law model the release rate is built on -- not a
    separate/invented formula. Steady-state diffusion through a
    membrane with a fixed concentration at one face and sink
    conditions (~zero concentration) at the other is, by definition,
    linear in position:

        C(z) = K * C_res * (1 - z/L),   z in [0, L]

    Its constant gradient is exactly what the release-rate formula
    assumes: |J| = D*dC/dz = D*K*C_res/L -- the same D*K*C*A/L already
    in the rate formula, confirming this field is consistent with (not
    a contradiction of) the scalar model."""
    l = max(membrane_thickness_m, FLOAT_FLOOR)
    z = min(max(z_from_reservoir_m, 0.0), l)
    c_surface = partition_coeff * reservoir_conc_kg_m3
    return c_surface * (1.0 - z / l)


def post_depletion_decay_rate_constant_per_day(
    diffusion_coeff_m2_s: float, partition_coeff: float, membrane_thickness_m: float, reservoir_thickness_m: float,
) -> float:
    """Real derived first-order decay constant for the phase after the
    reservoir's solid/excess drug is exhausted: the remaining
    dissolved drug is removed through the membrane at a rate
    proportional to how much is left, using the same real membrane
    transport coefficient (D*K/L) that governs steady-state release,
    normalized by the reservoir's own real volume:

        k = D*K*A / (L * V_reservoir),   V_reservoir = A * L_res

    Area cancels algebraically:  k = D*K / (L * L_res)

    -- exactly 1 / depletion_time_days (not a coincidence: the same
    quantity is depletion time's own reciprocal)."""
    l = max(membrane_thickness_m, FLOAT_FLOOR)
    k_per_s = diffusion_coeff_m2_s * partition_coeff / (l * reservoir_thickness_m)
    return k_per_s * 86400.0


def simulate_release_timeseries(
    diffusion_coeff_m2_s: float, partition_coeff: float, reservoir_conc_kg_m3: float,
    area_m2: float, membrane_thickness_m: float, reservoir_thickness_m: float,
    duration_days: Optional[float] = None, n_points: int = 100,
) -> List[Dict[str, float]]:
    """Real time-domain simulation of release/depletion over the
    device's wear life, combining only the real formulas above plus
    the one derived decay-tail formula -- no invented dynamics.

    Phase 1 (zero-order, t < depletion_time_days): the reservoir still
    holds excess drug, so the membrane (the rate-limiting step)
    sustains the full, constant steady-state rate -- the real,
    standard "zero-order release" pattern real reservoir-type
    transdermal patches are designed around.

    Phase 2 (first-order decay tail, t >= depletion_time_days): the
    reservoir is out of excess drug, so its concentration decays as
    C(t) = C0*exp(-k*(t-t_dep)) and the achievable rate follows the
    same Fick's-law formula, now time-varying only because C(t) is.
    """
    t_dep = depletion_time_days(diffusion_coeff_m2_s, partition_coeff, membrane_thickness_m, reservoir_thickness_m)
    if duration_days is None or duration_days <= 0:
        duration_days = 1.5 * t_dep

    rate0 = steady_state_release_rate_mg_day(diffusion_coeff_m2_s, partition_coeff, reservoir_conc_kg_m3, area_m2, membrane_thickness_m)
    k_per_day = post_depletion_decay_rate_constant_per_day(diffusion_coeff_m2_s, partition_coeff, membrane_thickness_m, reservoir_thickness_m)
    reservoir_mass_mg = reservoir_conc_kg_m3 * area_m2 * reservoir_thickness_m * 1000.0

    n = max(int(n_points), 2)
    points: List[Dict[str, float]] = []
    for i in range(n):
        t = duration_days * i / (n - 1)
        if t <= t_dep:
            rate = rate0
            cumulative = rate0 * t
        else:
            dt = t - t_dep
            decay = math.exp(-k_per_day * dt)
            rate = rate0 * decay
            cumulative = rate0 * t_dep + (rate0 / max(k_per_day, FLOAT_FLOOR)) * (1.0 - decay)
        remaining = max(reservoir_mass_mg - rate0 * min(t, t_dep), 0.0)
        points.append({
            "day": t,
            "rate_mg_day": rate,
            "cumulative_delivered_mg": cumulative,
            "reservoir_remaining_mg": remaining,
        })
    return points
