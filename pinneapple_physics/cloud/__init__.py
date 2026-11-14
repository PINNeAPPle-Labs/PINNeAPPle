"""Cloud microphysics: the capacitance (vapour-diffusion) growth law of a single ice crystal and its integration.

    from pinneapple_physics.cloud import DepositionalIceGrowth, solve_mass_ratio
    model = DepositionalIceGrowth(T=233.0, p=97190.0, Si=1.2, kinetics="nelson_baker")
    ratio = solve_mass_ratio(model, np.arange(0.0, 501.0), m0=1.5e-10)   # m/m0 over 500 s

The kinetics are pluggable (``"none"``, ``"nelson_baker"``, or any ``G(r, T, Si)`` callable), which is what makes
the law usable as the forward model of a neural-ODE equation-discovery run: the network replaces ``G`` and
everything else stays the published physics. Equations, constants and their sources are in
:mod:`pinneapple_physics.cloud.ice_deposition`.
"""
from .ice_deposition import (
    PLANETARY_CONSTANTS,
    DepositionalIceGrowth,
    alpha_nelson_baker,
    conductivity_air,
    diffusivity_vapour,
    esat_ice,
    mass_from_radius,
    modified_diffusivity,
    radius_from_mass,
    scrit_harrington2019,
    transfer_coefficient,
    transfer_coefficient_continuum,
)
from .ice_growth import solve_mass, solve_mass_ratio

__all__ = ["PLANETARY_CONSTANTS", "DepositionalIceGrowth", "esat_ice", "diffusivity_vapour", "conductivity_air",
           "scrit_harrington2019", "alpha_nelson_baker", "modified_diffusivity", "transfer_coefficient",
           "transfer_coefficient_continuum", "radius_from_mass", "mass_from_radius", "solve_mass",
           "solve_mass_ratio"]
