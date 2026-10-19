"""SI dimensions of electromagnetic quantities and a dimensional check of Maxwell's equations.

Every field, source and material parameter of this package is in SI. :data:`QUANTITIES` gives the SI unit of
each named quantity (parsed by :mod:`pinneapple_data.physical_units`, dimension vector over
``(m, kg, s, K, A, mol)``); :data:`EQUATIONS` writes each law as a list of terms, each a product of quantities
and derivative operators, and :func:`check_equations` verifies that all terms of every law have the same
dimension. :func:`check_units` checks user-declared units of fields against the table, so a field stored in
``V/cm`` or a permeability given in ``H/m`` mislabelled as ``T`` is caught before any residual is computed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pinneapple_data import physical_units as _pu

__all__ = ["QUANTITIES", "EQUATIONS", "dimension", "term_dimension", "check_equations", "check_units",
           "UnitProblem"]

#: quantity -> SI unit (as text understood by ``pinneapple_data.physical_units.parse_unit``)
QUANTITIES: dict[str, str] = {
    "E": "V/m",                    # electric field
    "D": "coulomb/m^2",            # electric displacement
    "B": "T",                      # magnetic flux density
    "H": "A/m",                    # magnetic field
    "J": "A/m^2",                  # current density
    "K": "A/m",                    # surface current density
    "rho": "coulomb/m^3",          # charge density
    "sigma_s": "coulomb/m^2",      # surface charge density
    "phi": "V",                    # electric scalar potential
    "A": "Wb/m",                   # magnetic vector potential
    "eps": "farad/m",              # permittivity
    "mu": "henry/m",               # permeability
    "sigma": "siemens/m",          # conductivity
    "S": "W/m^2",                  # Poynting vector
    "u": "J/m^3",                  # electromagnetic energy density
    "p": "W/m^3",                  # power density (J.E)
    "M": "A/m",                    # magnetisation
    "P": "coulomb/m^2",            # polarisation
    "q": "coulomb",                # charge
    "v": "m/s",                    # velocity
    "F": "N",                      # force
    "m": "kg",                     # mass
    "omega": "1/s",                # angular frequency
    "t": "s",
    "x": "m",
}

#: law -> terms; a term is a tuple of factors, a factor is a quantity name, ``"d/dt"`` or ``"d/dx"``
#: (curl and divergence are both one spatial derivative), or ``"1/<name>"``.
EQUATIONS: dict[str, list[tuple[str, ...]]] = {
    "gauss_electric": [("d/dx", "D"), ("rho",)],                       # div D = rho
    "gauss_electric_vacuum": [("d/dx", "E"), ("rho", "1/eps")],         # div E = rho / eps0
    "gauss_magnetic": [("d/dx", "B"), ("d/dx", "B")],                  # div B = 0 (one term, compared with itself)
    "faraday": [("d/dx", "E"), ("d/dt", "B")],                         # curl E = -dB/dt
    "ampere_maxwell": [("d/dx", "H"), ("J",), ("d/dt", "D")],          # curl H = J + dD/dt
    "continuity": [("d/dt", "rho"), ("d/dx", "J")],                    # d rho/dt + div J = 0
    "poynting": [("d/dt", "u"), ("d/dx", "S"), ("J", "E")],            # du/dt + div S + J.E = 0
    "poynting_vector": [("S",), ("E", "H")],                           # S = E x H
    "energy_density": [("u",), ("E", "D"), ("B", "H")],                # u = (E.D + B.H) / 2
    "constitutive_D": [("D",), ("eps", "E"), ("P",)],                  # D = eps0 E + P
    "constitutive_B": [("B",), ("mu", "H"), ("mu", "M")],              # B = mu0 (H + M)
    "ohm": [("J",), ("sigma", "E")],                                   # J = sigma E
    "potentials_E": [("E",), ("d/dx", "phi"), ("d/dt", "A")],          # E = -grad phi - dA/dt
    "potentials_B": [("B",), ("d/dx", "A")],                           # B = curl A
    "wave_speed": [("1/eps", "1/mu"), ("v", "v")],                     # c^2 = 1 / (eps mu)
    "lorentz_force": [("F",), ("q", "E"), ("q", "v", "B")],            # F = q (E + v x B)
    "newton": [("F",), ("m", "v", "omega")],                           # F = m dv/dt
    "skin_depth": [("x", "x"), ("1/omega", "1/mu", "1/sigma")],        # delta^2 = 2 / (omega mu sigma)
}

Dim = tuple


def dimension(unit: str) -> Dim:
    """Dimension vector of a unit expression."""
    return _pu.parse_unit(unit).dim


def _add(a: Dim, b: Dim, s: int = 1) -> Dim:
    return tuple(x + s * y for x, y in zip(a, b, strict=True))


_X = dimension("m")
_T = dimension("s")


def term_dimension(term: tuple[str, ...], quantities: Mapping[str, str] = QUANTITIES) -> Dim:
    """Dimension of a product of factors (see :data:`EQUATIONS`)."""
    d = _pu.DIMLESS
    for f in term:
        if f == "d/dx":
            d = _add(d, _X, -1)
        elif f == "d/dt":
            d = _add(d, _T, -1)
        elif f.startswith("1/"):
            d = _add(d, dimension(quantities[f[2:]]), -1)
        else:
            d = _add(d, dimension(quantities[f]))
    return d


@dataclass(frozen=True)
class UnitProblem:
    where: str
    message: str


def check_equations(equations: Mapping[str, list[tuple[str, ...]]] = EQUATIONS,
                    quantities: Mapping[str, str] = QUANTITIES) -> list[UnitProblem]:
    """Every term of every law must have the dimension of the first term; returns the violations."""
    out = []
    for name, terms in equations.items():
        ref = term_dimension(terms[0], quantities)
        for t in terms[1:]:
            d = term_dimension(t, quantities)
            if d != ref:
                out.append(UnitProblem(name, f"term {'*'.join(t)} has {_pu.dim_str(d)}, "
                                             f"first term {'*'.join(terms[0])} has {_pu.dim_str(ref)}"))
    return out


def check_units(declared: Mapping[str, str]) -> list[UnitProblem]:
    """Compare declared units of named quantities (``{"E": "V/m", "B": "mT"}``) with :data:`QUANTITIES`.

    A different scale (``mT`` for ``B``) is reported as a scale problem only if the factor is not 1, so the
    caller can convert; a different dimension is an error. Unknown names are skipped."""
    out = []
    for name, unit in declared.items():
        if name not in QUANTITIES:
            continue
        try:
            u = _pu.parse_unit(unit)
        except _pu.UnitError as e:
            out.append(UnitProblem(name, str(e)))
            continue
        want = dimension(QUANTITIES[name])
        if u.dim != want:
            out.append(UnitProblem(name, f"{unit!r} has dimension {_pu.dim_str(u.dim)}, {name} needs "
                                         f"{_pu.dim_str(want)} ({QUANTITIES[name]})"))
        elif abs(u.factor - 1.0) > 1e-12:
            out.append(UnitProblem(name, f"{unit!r} is {u.factor:g} x SI ({QUANTITIES[name]}); convert before "
                                         "computing residuals"))
    return out
