"""Cloud microphysics: depositional growth of a single ice crystal from vapour.

The model is the capacitance (diffusion) growth law with the surface-kinetic correction, as written for the
levitation-diffusion-chamber experiments of Pokrifka et al. (2020, 2023) and used by Lamb et al. (2026) to learn
the unknown part of the growth law with a neural ODE:

    dm/dt = 4 pi r (S_i - 1) G(r, T, S_i, alpha)                                        (Lamb et al. 2026, Eq. 7)

``G`` combines vapour and thermal diffusion to the crystal surface and is reduced from the continuum value by
surface attachment kinetics, which enter through a deposition coefficient ``alpha``. Two limits are available:

- :func:`transfer_coefficient` — ``G`` with the modified diffusivity of :func:`modified_diffusivity`
  (continuum and molecular regimes bridged by the jump length and ``alpha``);
- :func:`transfer_coefficient_continuum` — ``G_c``, the same expression with ``alpha -> 1`` (no surface kinetics).

Everything here is closed form and dependency-light (numpy only); :mod:`pinneapple_physics.cloud.ice_growth`
integrates the ODE.

Constants, parameterisations and their sources
---------------------------------------------
=====================================  ====================================================================
quantity                                source
=====================================  ====================================================================
``esat_i(T)``                           Murphy & Koop (2005), QJRMS 131, Eq. 13 of Lamb et al. (2026)
``Dv(T, p)``                            Pruppacher & Klett (1998), Eq. 14 of Lamb et al. (2026)
``ka(T)``                               Pruppacher & Klett (1998), Eq. 15 of Lamb et al. (2026)
``scrit(T)``                            Harrington et al. (2019), JAS 76, Eq. 11 of Lamb et al. (2026)
``alpha(S_i, T)``                       Nelson & Baker (1996), Eq. 10 of Lamb et al. (2026)
remaining constants                     Table S1 of Lamb et al. (2026)
=====================================  ====================================================================

References
----------
Lamb, K. D., Harrington, J. Y., Moyle, A. M., Pokrifka, G. F., Clouser, B. W., Ebert, V., Moehler, O., &
Saathoff, H. (2026). Discovering how ice crystals grow using neural ODEs and symbolic regression.
*Science Advances* 12(39), eaee6174. doi:10.1126/sciadv.aee6174 (preprint arXiv:2510.17935).
Harrington, J. Y., Moyle, A., Hanson, L. E., & Morrison, H. (2019). On calculating deposition coefficients and
aspect-ratio evolution in approximate models of ice crystal vapor growth. *JAS* 76, 1609-1625.
Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice and supercooled water for atmospheric
applications. *QJRMS* 131, 1539-1565.
Nelson, J., & Baker, M. B. (1996). New theoretical framework for studies of vapor growth and sublimation of small
ice crystals. *JAS* 53, 2853-2862.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PLANETARY_CONSTANTS",
    "esat_ice",
    "diffusivity_vapour",
    "conductivity_air",
    "scrit_harrington2019",
    "alpha_nelson_baker",
    "modified_diffusivity",
    "transfer_coefficient",
    "transfer_coefficient_continuum",
    "radius_from_mass",
    "mass_from_radius",
    "DepositionalIceGrowth",
]

#: Table S1 of Lamb et al. (2026); SI units except where noted.
PLANETARY_CONSTANTS = {
    "rho_ice": 910.0,        # kg/m3, density of ice
    "rho_water": 1000.0,     # kg/m3, density of water
    "Mw": 18.0e-3,           # kg/mol, molecular mass of water
    "R": 8.3144521,          # J/mol/K, universal gas constant
    "Ra": 287.05,            # J/kg/K, gas constant of air
    "Rv": 461.51,            # J/kg/K, gas constant of water vapour
    "cp": 1005.0,            # J/kg/K, specific heat capacity of air
    "Ls": 2.837e6,           # J/kg, latent heat of sublimation
    "Lv": 2.5e6,             # J/kg, latent heat of vaporisation
    "g": 9.81,               # m/s2
    "alpha_thermal": 1.0,    # thermal deposition coefficient (no kinetic correction to conduction)
    "lambda_air": 8.0e-8,    # m, mean free path of water molecules in air
    "delta_v": 1.3 * 8.0e-8,  # m, vapour jump length (1.3 lambda)
    "delta_T": 2.16e-7,      # m, thermal jump length
    "T0": 273.15,            # K, reference temperature of the diffusivity fit
    "p0": 101325.0,          # Pa, reference pressure of the diffusivity fit
}

#: Murphy & Koop (2005) coefficients of the saturation vapour pressure over ice.
_MK = (9.550426, 5723.265, 3.53068, 0.00728332)


def esat_ice(T):
    """Saturation vapour pressure over ice in Pa (Murphy & Koop 2005, Lamb et al. 2026 Eq. 13).

    ``T`` in kelvin; scalar or array.
    """
    a0, a1, a2, a3 = _MK
    T = np.asarray(T, dtype=np.float64)
    return np.exp(a0 - a1 / T + a2 * np.log(T) - a3 * T)


def diffusivity_vapour(T, p, c=PLANETARY_CONSTANTS):
    """Diffusivity of water vapour in air in m2/s (Pruppacher & Klett 1998, Lamb et al. 2026 Eq. 14).

    ``Dv = 2.11e-5 (T/T0)^1.94 (p0/p)``; the fit is quoted for -40 to 40 C, and the levitation-chamber and AIDA
    experiments use it outside that range, as the paper does.
    """
    T = np.asarray(T, dtype=np.float64)
    return 2.11e-5 * (T / c["T0"]) ** 1.94 * (c["p0"] / np.asarray(p, dtype=np.float64))


def conductivity_air(T, c=PLANETARY_CONSTANTS):
    """Thermal conductivity of air in W/m/K (Pruppacher & Klett 1998, Lamb et al. 2026 Eq. 15).

    The paper writes ``ka = 4.187e-3 (5.69 + 0.017 (T - 273.15))`` in cal/cm/s/K; the factor 4.187 converts it
    to W/m/K, which is what :func:`transfer_coefficient` needs.
    """
    T = np.asarray(T, dtype=np.float64)
    return 4.187e-3 * (5.69 + 0.017 * (T - c["T0"]))


def scrit_harrington2019(T, c=PLANETARY_CONSTANTS):
    """Critical (threshold) supersaturation of the basal face, fractional, Harrington et al. (2019).

    Lamb et al. (2026) Eq. 11 quotes the compact fit ``scrit = 9.6066e-5 |Tc|^1.9171`` with ``Tc`` the temperature
    in Celsius. The published piecewise fit it approximates (Harrington et al. 2019, Table 1) is what this
    function evaluates, below -30 C through the quadratic used for the basal (c-axis) face; the two agree to
    within a few percent over the 205-240 K range of the chamber experiments.
    """
    T = np.asarray(T, dtype=np.float64)
    Tc = np.clip(T - c["T0"], -100.0, -1.0)          # deviation from 0 C, capped to the fit's validity range
    sc = 3.7955 + 0.10614 * Tc + 0.00753 * Tc**2     # delT < -30 C branch, the one these experiments sit in
    return sc / 100.0


def alpha_nelson_baker(Si, T, m_exp=1.0):
    """Deposition coefficient of Nelson & Baker (1996), Lamb et al. (2026) Eq. 10.

    ``alpha = (s/scrit)^m tanh[(scrit/s)^m]`` with ``s = S_i - 1`` the supersaturation above the surface and ``m``
    the surface-growth exponent (``m = 1`` spiral dislocations, ``m > 10`` ledge nucleation). ``m = 1`` is the value
    the paper's synthetic data set is generated with.
    """
    s = np.maximum(np.asarray(Si, dtype=np.float64) - 1.0, 1e-12)
    scrit = scrit_harrington2019(T)
    return np.clip((s / scrit) ** m_exp * np.tanh((scrit / s) ** m_exp), 0.0, 1.0)


def modified_diffusivity(r, T, p, alpha, c=PLANETARY_CONSTANTS):
    """``Dv*`` in m2/s: vapour diffusivity reduced by surface attachment kinetics (Lamb et al. 2026 Eq. 9).

    ``Dv* = Dv / ( r/(r+Delta_v) + 4 Dv/(r alpha w) )`` with ``w = sqrt(8 R T / (pi Mw))`` the molecular speed of
    water vapour. The first term is the gas-phase jump-length correction, the second the surface-kinetic one:
    kinetics bite when ``4 Dv/(r alpha w)`` reaches order one, i.e. for small crystals and small ``alpha``
    (``alpha = 1`` and ``r >> Delta_v`` recovers ``Dv``).
    """
    r = np.asarray(r, dtype=np.float64)
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 1e-12, None)
    T = np.asarray(T, dtype=np.float64)
    w = np.sqrt(8.0 * c["R"] * T / (np.pi * c["Mw"]))
    dv = diffusivity_vapour(T, p, c)
    return dv / (r / (r + c["delta_v"]) + 4.0 * dv / (r * alpha * w))


def transfer_coefficient(r, T, p, alpha, c=PLANETARY_CONSTANTS):
    """``G`` in s/m: the combined vapour + thermal transfer coefficient of Lamb et al. (2026) Eq. 8.

    ``G = [ R T / (esat_i(T) Dv* Mw) + (Ls / (ka T)) (Ls Mw / (R T) - 1) ]^-1``, the bracketed sum being the
    series resistance of vapour diffusion and heat conduction away from a growing crystal. Kinetic effects enter
    only through ``Dv*``; the thermal term is left in the continuum limit, as the paper does.
    """
    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    Ls, Mw, R = c["Ls"], c["Mw"], c["R"]
    dv_star = modified_diffusivity(r, T, p, alpha, c)
    ka = conductivity_air(T, c)
    vapour = R * T / (esat_ice(T) * dv_star * Mw)
    thermal = (Ls / (ka * T)) * (Ls * Mw / (R * T) - 1.0)
    return 1.0 / (vapour + thermal)


def transfer_coefficient_continuum(r, T, p, c=PLANETARY_CONSTANTS):
    """``G_c`` in s/m: :func:`transfer_coefficient` in the continuum limit ``alpha -> 1`` (Eq. 12)."""
    return transfer_coefficient(r, T, p, 1.0, c)


def radius_from_mass(m, c=PLANETARY_CONSTANTS):
    """Equivalent spherical radius in m of an ice crystal of mass ``m`` in kg."""
    return (3.0 * np.asarray(m, dtype=np.float64) / (4.0 * np.pi * c["rho_ice"])) ** (1.0 / 3.0)


def mass_from_radius(r, c=PLANETARY_CONSTANTS):
    """Mass in kg of a spherical ice crystal of radius ``r`` in m."""
    return 4.0 / 3.0 * np.pi * c["rho_ice"] * np.asarray(r, dtype=np.float64) ** 3


@dataclass(frozen=True)
class DepositionalIceGrowth:
    """The growth law of Lamb et al. (2026) Eq. 7 for one crystal, with a pluggable ``G``.

    ``kinetics`` selects where the unknown physics lives:

    - ``"none"``         ``G = G_c`` (no surface kinetics), the paper's baseline;
    - ``"nelson_baker"`` ``alpha`` from :func:`alpha_nelson_baker`, its parameterised baseline;
    - a callable         ``G(r, T, Si) -> G`` in s/m, which is where a learned model is supplied.

    ``capacitance`` is ``C = r`` (a sphere), the paper's assumption for these micron-scale crystals.
    """

    T: float = 230.0
    p: float = 97190.0
    Si: float = 1.2
    kinetics: object = "none"
    capacitance: float = 1.0

    def transfer(self, r):
        """``G`` at radius ``r`` in m, by the selected kinetics (s/m)."""
        if callable(self.kinetics):
            return np.asarray(self.kinetics(r, self.T, self.Si), dtype=np.float64)
        if self.kinetics == "none":
            return transfer_coefficient_continuum(r, self.T, self.p)
        if self.kinetics == "nelson_baker":
            alpha = alpha_nelson_baker(self.Si, self.T)
            return transfer_coefficient(r, self.T, self.p, alpha)
        raise ValueError(f"unknown kinetics {self.kinetics!r}")

    def dmdt(self, m):
        """``dm/dt`` in kg/s at mass ``m`` in kg (Eq. 7 with ``C = r``)."""
        r = radius_from_mass(m)
        return 4.0 * np.pi * self.capacitance * r * (self.Si - 1.0) * self.transfer(r)
