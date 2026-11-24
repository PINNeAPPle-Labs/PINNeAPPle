"""Public validation suites, with the dimensionless numbers their case pages publish (#171).

The regime audit answers "which bands of Reynolds and Mach does this data never reach?" on a project's own dataset.
The second half of the acceptance is the harder one: the same question asked of a **public benchmark**, where the
numbers are not ours to choose. A suite written here is only the published freestream conditions of cases whose
landing page states them, so nothing in it can be tuned to make the audit look good — and what it shows is that a
well-known reference set is missing a band that most external aerodynamics lives in.

NASA Turbulence Modeling Resource (TMR), ``https://tmbwg.github.io/turbmodels/``, read 2026-11-24. Every number below
is stated on the cited case page, and the ones the page does **not** publish are absent here rather than guessed:

========================  ========================================  ==============================  ==============
case                      Reynolds                                  Mach                            page
========================  ========================================  ==============================  ==============
2DZP flat plate          5e6 per unit length of x                  0.2 ("M = 0.2 or less")         ``flatplate_val``
2DN00 NACA 0012          6e6 per chord                             0.15 ("run M = 0.15")           ``naca0012_val``
2DBFS backward step      3.6e4 per step height, 5e3 per theta     0.128                           ``backstep_val``
2DZPH high-Mach plate    15e6 per unit length of x                 2.0 and 5.0 (three walls at 5)   ``ZPGflatplateSS_val``
ATB transonic bump       not stated                                0.875                           ``axibump_val``
ACC-SBLI Mach-7 corner   not stated                                7.11 (inflow, from 7.05)        ``axiswblim7_val``
========================  ========================================  ==============================  ==============

The Reynolds number is a function of the length it is based on, so cases based on a unit length, a chord and a step
height are kept in **separate columns**: one column with all three mixed in would answer a question about a number
that does not exist. ``reynolds`` holds only the per-unit-length-of-x values, which is the column a
``reynolds_flat_plate`` regime can read; the others are reported under their own basis.

>>> from pinneapple_data.benchmark_suites import audit_nasa_tmr          # doctest: +SKIP
>>> card = audit_nasa_tmr()                                              # doctest: +SKIP
>>> card["regime.mach"].evidence["missing"]                              # doctest: +SKIP
['subsonic']
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PublicCase", "NASA_TMR", "nasa_tmr_cases", "nasa_tmr_splits", "audit_nasa_tmr"]

#: the length a published Reynolds number is based on; only ``"x"`` values can be read by a flat-plate regime
_PER_UNIT_X = "per unit length of x"

NASA_TMR = "NASA TMR"


@dataclass(frozen=True)
class PublicCase:
    """One published case: what its page states, and where it states it."""

    case_id: str
    name: str
    reference: str
    mach: float | None = None
    reynolds: float | None = None
    reynolds_basis: str = ""
    note: str = ""
    suite: str = NASA_TMR

    def __post_init__(self) -> None:
        if self.reynolds is not None and not self.reynolds_basis:
            raise ValueError(f"{self.case_id}: a Reynolds number without the length it is based on says nothing")
        if self.mach is None and self.reynolds is None:
            raise ValueError(f"{self.case_id}: a case with no dimensionless number is not a regime sample")


#: The cases of the NASA TMR suite whose page states a Mach number or a Reynolds number. Two cases of the suite
#: (the axisymmetric bump and the compression corner) publish no Reynolds number, which is why they carry none here.
NASA_TMR_CASES: tuple[PublicCase, ...] = (
    PublicCase("2DZP", "2D zero-pressure-gradient flat plate",
               "https://tmbwg.github.io/turbmodels/flatplate_val.html",
               mach=0.2, reynolds=5e6, reynolds_basis=_PER_UNIT_X,
               note="the page recommends 'M = 0.2 or less'; 0.2 is the top of that range"),
    PublicCase("2DN00", "NACA 0012 airfoil", "https://tmbwg.github.io/turbmodels/naca0012_val.html",
               mach=0.15, reynolds=6e6, reynolds_basis="chord",
               note="Re = 6e6 per chord; the page recommends running at M = 0.15"),
    PublicCase("2DBFS", "2D backward-facing step", "https://tmbwg.github.io/turbmodels/backstep_val.html",
               mach=0.128, reynolds=3.6e4, reynolds_basis="step height H",
               note="Re_H = 36000 (earlier listings of 50000 were in error); Re_theta = 5000 ahead of the step"),
    PublicCase("2DZPH", "2D ZPG high-Mach-number flat plate",
               "https://tmbwg.github.io/turbmodels/ZPGflatplateSS_val.html",
               mach=2.0, reynolds=15e6, reynolds_basis=_PER_UNIT_X,
               note="Re = 15e6 per unit length; the suite runs M = 2 and M = 5 (three wall temperatures at 5)"),
    PublicCase("2DZPH", "2D ZPG high-Mach-number flat plate (M = 5)",
               "https://tmbwg.github.io/turbmodels/ZPGflatplateSS_val.html",
               mach=5.0, reynolds=15e6, reynolds_basis=_PER_UNIT_X,
               note="the M = 5 counterpart of the case above: two entries, because a suite covers two regimes"),
    PublicCase("ATB", "Axisymmetric transonic bump", "https://tmbwg.github.io/turbmodels/axibump_val.html",
               mach=0.875, note="the page states the supercritical M = 0.875 and no Reynolds number"),
    PublicCase("ACC-SBLI", "Axisymmetric compression corner near Mach 7",
               "https://tmbwg.github.io/turbmodels/axiswblim7_val.html",
               mach=7.11, note="inflow Mach, raised from the experiment's 7.05 because the leading edge is outside "
                               "the domain; the page states no Reynolds number"),
)


def nasa_tmr_cases() -> list[PublicCase]:
    """The published cases, as records (copy of the frozen tuple, for a caller that wants to filter them)."""
    return list(NASA_TMR_CASES)


def _column(cases: list[PublicCase], name: str, get) -> list[float]:
    """One column, with NaN where the page states no value: a hole in the publication is not a zero."""
    out = []
    for c in cases:
        v = get(c)
        out.append(float("nan") if v is None else float(v))
    return out


def nasa_tmr_splits() -> dict[str, dict[str, list[float]]]:
    """The suite as an audit split: one column per basis of Reynolds, so the audit cannot compare a chord Reynolds
    number with a per-unit-length one. ``reynolds`` is the flat-plate basis; ``reynolds_chord`` and
    ``reynolds_step_h`` carry the others."""
    cases = nasa_tmr_cases()
    x = [c if c.reynolds_basis == _PER_UNIT_X else None for c in cases]

    def by_basis(basis: str):
        return _column(cases, "reynolds", lambda c: c.reynolds if c.reynolds_basis == basis else None)

    return {"suite": {
        "mach": _column(cases, "mach", lambda c: c.mach),
        "reynolds": _column(x, "reynolds", lambda c: None if c is None else c.reynolds),
        "reynolds_chord": by_basis("chord"),
        "reynolds_step_h": by_basis("step height H"),
    }}


def audit_nasa_tmr(*, regimes: tuple[str, ...] | list[str] = ("mach", "reynolds_flat_plate"),
                   name: str = "NASA TMR validation suite") -> Any:
    """Run the dataset audit — with its regime check — on the published suite, and return the card.

    The card's ``regime.*`` checks name the bands the suite never reaches; ``missing_regions()`` in their evidence is
    the interval a case would have to be added in. Nothing else in the card is about regimes (the suite is a table of
    published conditions, not training data), so a caller reads those checks and the audit's own findings separately.
    """
    from .audit import audit_dataset

    return audit_dataset(nasa_tmr_splits(), name=name, regimes=list(regimes),
                         missing_documented=("reynolds", "reynolds_chord", "reynolds_step_h"))
