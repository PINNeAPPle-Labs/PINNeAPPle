"""The published numbers the classical cases are measured against, transcribed as data with their citation (#272).

A validation case is only as good as the table it is compared with, so the table lives in the repository as data —
not as a comment, not as a picture, and not as a number someone remembered. Every entry below names the table it
came from, and each one is checked on import of the test suite against what the table must be true of on its own:
the endpoint values (the lid moves at ``u = 1``, a wall does not move), strictly increasing coordinates, and one
value per point. A transcription error that survives those checks would have to be a plausible number in the wrong
row, which is why the runs in ``tests/test_cfd_validation.py`` compare a real solver against these tables: a
mis-transcribed row cannot agree with a Navier-Stokes solution at 1-2 % on three meshes.

Lid-driven cavity — Ghia, Ghia and Shin (1982)
---------------------------------------------
U. Ghia, K.N. Ghia and C.T. Shin, "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a
Multigrid Method", *Journal of Computational Physics* 48(3), 387-411, 1982, Table I (Re = 100, 400, 1000) and
Table II (Re = 10000); doi:10.1016/0021-9991(82)90058-4.

``u`` is the horizontal velocity along the **vertical** centreline ``x = 0.5``, as a function of ``y``; ``v`` is the
vertical velocity along the **horizontal** centreline ``y = 0.5``, as a function of ``x``. The cavity is the unit
square with the lid at ``y = 1`` moving at ``U = 1``, and ``Re = U L / nu``. Ghia's own grid was 129x129; their stated
accuracy is the last digit of the tables (three significant figures, the fourth digit interpolated).

Natural convection in a square cavity — de Vahl Davis (1983)
-----------------------------------------------------------
G. de Vahl Davis, "Natural Convection of Air in a Square Cavity: A Bench Mark Numerical Solution", *International
Journal for Numerical Methods in Fluids* 3(3), 249-264, 1983, Table I; doi:10.1002/fld.1650030305.

The benchmark Nusselt number is the mean over the hot wall, air (Pr = 0.71), insulated top and bottom, side walls at
``T_h`` and ``T_c``. The published values below are the benchmark grid-independent ones; de Vahl Davis states them to
three significant figures, and the uncertainty he quotes at Ra = 1e6 is 0.6 %.

What is deliberately not here
-----------------------------
Armaly et al. (1983, *J. Fluid Mech.* 127, 473-496) publish the reattachment length of the backward-facing step as a
figure, and the digitised table that circulates in the literature is a secondary source with no single provenance.
:data:`ARMALY_REATTACHMENT` is therefore **empty**: the case is built and run by
:mod:`~pinneapple_simulation.cfd_validation.step`, and the reference is filled in from the paper or from a documented
digitisation rather than from memory. Pitz and Daily's profiles are in the same position.

CFDBench — Luo, Chen and Zhang (2023), the tube flow's operating parameters
--------------------------------------------------------------------------
Y. Luo, Y. Chen and Z. Zhang, "CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics",
arXiv:2310.05963, Table 3 (the operating parameters of the tube flow subset). Dataset: HuggingFace
``chen-yingfa/CFDBench``, licence Apache-2.0, read from the dataset card.

This one is not a table of *answers*, and :data:`CFDBENCH_TUBE` does not pretend otherwise. The tube is a water-air
two-phase flow driven into a tube full of air and solved in Ansys Fluent, so the paper publishes **no** physical
solution to be scored against — its tube tables (Tables 12 and 13) are the NMSE of neural operators against the
archive's own fields. What Table 3 does publish is the *definition of the cases*: the baseline tube and the three
parameter sets (``bc``, ``prop``, ``geo``) with the values each one sweeps. Transcribing it is what turns "the
archive, read however the reader likes" into "the cases the paper says are there", and
:func:`~pinneapple_simulation.cfd_validation.cfdbench.tube_against_table` holds the archive against it case by case.

The transcription earns its keep on the first reading. The ``prop`` family stores its density as the ten-value
*index* ``0..9``, not the ``10 .. 1000 kg/m3`` the table publishes, while its viscosity is stored as the value — so
``Re = rho U L / mu`` computed from the archive's own ``case.json`` is a hundredfold short for that family and is not
the Reynolds number of the flow. The other two families match the table exactly, value for value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Profile", "GHIA_U", "GHIA_V", "GHIA_RE", "DEV_VAHL_DAVIS_NU", "ARMALY_REATTACHMENT", "PROFILES",
           "CaseTable", "CFDBENCH_TUBE", "profile", "nu_for_grashof", "get", "as_dicts", "disputes"]


@dataclass(frozen=True)
class Profile:
    """One published centreline profile: the points, the values, and the table they were transcribed from."""

    case: str                      # "ghia-1982"
    quantity: str                  # "u" or "v"
    coordinate: str                # the axis the points are along: "y" or "x"
    reynolds: float
    points: tuple[float, ...]
    values: tuple[float, ...]
    citation: str
    note: str = ""
    disputed: tuple[int, ...] = ()          # stations whose transcription our own converged run contradicts
    dispute: str = ""                       # what was measured there, and what the difference means

    def __post_init__(self) -> None:
        if len(self.points) != len(self.values):
            raise ValueError(f"{self.case} Re={self.reynolds} {self.quantity}: {len(self.points)} points, "
                             f"{len(self.values)} values")
        if len(self.points) < 3:
            raise ValueError(f"{self.case} Re={self.reynolds} {self.quantity}: a profile of {len(self.points)} points")
        for a, b in zip(self.points, self.points[1:], strict=False):
            if not b > a:
                raise ValueError(f"{self.case} Re={self.reynolds} {self.quantity}: the coordinates are not "
                                 f"increasing ({a} then {b})")

    @property
    def span(self) -> tuple[float, float]:
        return self.points[0], self.points[-1]

    def to_dict(self) -> dict[str, Any]:
        return {"case": self.case, "quantity": self.quantity, "coordinate": self.coordinate,
                "reynolds": self.reynolds, "points": list(self.points), "values": list(self.values),
                "citation": self.citation, "note": self.note,
                "disputed": [{"index": i, "point": self.points[i]} for i in self.disputed],
                "dispute": self.dispute}


_GHIA_1982 = ("Ghia, Ghia and Shin (1982), J. Comput. Phys. 48(3), 387-411, "
              "Table I (Re = 100-1000) / Table II (Re = 10000), doi:10.1016/0021-9991(82)90058-4")

#: The ``y`` positions of Ghia's ``u`` profiles: the same 17 stations at every Reynolds number.
_GHIA_Y = (0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,
           0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000)

#: The ``x`` positions of Ghia's ``v`` profiles.
_GHIA_X = (0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000,
           0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000)

_GHIA_U_VALUES = {
    100: (0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581,
          -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000),
    400: (0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,
          0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000),
    1000: (0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080,
           0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000),
    10000: (0.00000, -0.12166, -0.13811, -0.15482, -0.21743, -0.31997, -0.36508, -0.31685, -0.29366,
            -0.12722, 0.04130, 0.20618, 0.37976, 0.42452, 0.47400, 0.54719, 1.00000),
}

_GHIA_V_VALUES = {
    100: (0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454,
          -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000),
    400: (0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186,
          -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000),
    1000: (0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526,
           -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000),
    10000: (0.00000, 0.25707, 0.26786, 0.27685, 0.29305, 0.32606, 0.30600, 0.29628, 0.01653,
            -0.21132, -0.35051, -0.66563, -0.59931, -0.55069, -0.48590, -0.39017, 0.00000),
}

#: The Reynolds numbers Ghia, Ghia and Shin publish.
GHIA_RE = (100, 400, 1000, 10000)

GHIA_U: dict[int, Profile] = {re: Profile("ghia-1982", "u", "y", float(re), _GHIA_Y, _GHIA_U_VALUES[re], _GHIA_1982)
                              for re in GHIA_RE}
GHIA_V: dict[int, Profile] = {re: Profile("ghia-1982", "v", "x", float(re), _GHIA_X, _GHIA_V_VALUES[re], _GHIA_1982)
                              for re in GHIA_RE}

# One station of the Re = 400 ``v`` column is disputed, and it is recorded rather than corrected: every other station of
# that column agrees with our converged 128x128 solution to 0.005 or better, and x = 0.9063 agrees to nothing -- the
# transcribed value is -0.23827, the solution gives -0.387, and the neighbours of both put the true profile near -0.38.
# The value is left exactly as transcribed (a table is not edited to match a run) and the station is excluded from the
# error, with the measurement that made it doubtful in the note. Confirming it needs the paper, not another run.
GHIA_V[400] = Profile("ghia-1982", "v", "x", 400.0, _GHIA_X, _GHIA_V_VALUES[400], _GHIA_1982,
                      disputed=(11,),
                      dispute="x = 0.9063: the transcribed -0.23827 against -0.3870 measured by the converged 128x128 "
                              "run (every other station of this column agrees to 0.005); one of the two rows is wrong "
                              "and the paper has to settle which")

#: Every transcribed profile, for the checks and for the example that writes them out.
PROFILES: tuple[Profile, ...] = tuple(GHIA_U.values()) + tuple(GHIA_V.values())


def profile(case: str, quantity: str, reynolds: float) -> Profile:
    """One profile by case, quantity and Reynolds number, with a message that lists what exists when it does not."""
    known = {"ghia-1982": {"u": GHIA_U, "v": GHIA_V}}
    if case not in known:
        raise KeyError(f"{case}: no transcribed profiles for that case (have: {', '.join(sorted(known))})")
    if quantity not in known[case]:
        raise KeyError(f"{case}: no {quantity!r} profile (have: {', '.join(sorted(known[case]))})")
    re = int(round(float(reynolds)))
    if re not in known[case][quantity]:
        raise KeyError(f"{case} {quantity}: Re = {reynolds:g} was not published "
                       f"(have: {', '.join(str(k) for k in known[case][quantity])})")
    return known[case][quantity][re]


#: Mean Nusselt number over the hot wall of a square cavity of air (Pr = 0.71), ``Ra -> Nu``.
DEV_VAHL_DAVIS_NU: dict[float, float] = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}
DEV_VAHL_DAVIS = ("de Vahl Davis (1983), Int. J. Numer. Methods Fluids 3(3), 249-264, Table I, "
                  "doi:10.1002/fld.1650030305")

#: Armaly et al. (1983)'s reattachment length against Reynolds number, ``Re -> x1/S``. **Empty on purpose**: the
#: published curve is a figure, and a secondary digitisation has no single provenance. See the module docstring.
ARMALY_REATTACHMENT: dict[float, float] = {}


@dataclass(frozen=True)
class CaseTable:
    """A published table of *operating parameters*: the case it defines and the values each subset sweeps.

    The point of the type is the provenance. A table of answers names the run that confirms it (a mistyped row cannot
    agree with a Navier-Stokes solution on three meshes); a table of parameters is confirmed by the data instead —
    every value it publishes has to be found in the archive, and every value in the archive has to be one it
    publishes. :func:`~pinneapple_simulation.cfd_validation.cfdbench.tube_against_table` is that check.
    """

    case: str
    citation: str
    definition: dict[str, float | str]         # the baseline case every subset is a departure from
    sets: dict[str, tuple[float, ...]]         # subset, or subset/parameter, -> the values the paper publishes
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"case": self.case, "citation": self.citation, "definition": dict(self.definition),
                "sets": {k: list(v) for k, v in self.sets.items()}, "note": self.note}


#: The tube flow of CFDBench, Table 3, transcribed. ``bc`` sweeps the inlet velocity over 50 values; ``prop`` is the
#: 10 x 10 grid of density against viscosity; ``geo`` is five diameters by five of the ten length-to-diameter ratios,
#: which is the 25 geometries the paper counts. The archive names the diameter ``height`` and the length ``width``.
CFDBENCH_TUBE = CaseTable(
    case="cfdbench-tube",
    citation="Luo, Chen and Zhang (2023), arXiv:2310.05963, Table 3 (tube flow operating parameters)",
    definition={"length_l": 1.0, "diameter_d": 0.1, "density_rho": 100.0, "viscosity_mu": 0.1,
                "inlet_velocity": 1.0, "walls": "top and bottom, no-slip", "inlet": "velocity", "outlet": "pressure",
                "time_step": 0.01, "grid": "interpolated to 64 x 64"},
    sets={
        # uB = 0.1, 0.2, ... 5.0 m/s: fifty values, which is the fifty cases the paper counts
        "bc": tuple(round(0.1 * i, 10) for i in range(1, 51)),
        # rho = 10, 120, ... 1000 kg/m3, in steps of 110
        "prop/density": tuple(10.0 + 110.0 * i for i in range(10)),
        # mu = 0.01, 0.12, ... 1.0 Pa s, in steps of 0.11
        "prop/viscosity": tuple(round(0.01 + 0.11 * i, 10) for i in range(10)),
        "geo/diameter": (0.01, 0.05, 0.1, 0.3, 0.5),
        # the paper's d/l set; the archive stores these as l/d, which is the reciprocal
        "geo/ratio": (1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0, 75.0, 100.0),
    },
    note="the paper's baseline is rho = 100, mu = 0.1, u_in = 1, d = 0.1, l = 1; the archive's prop family stores "
         "density as the ten-value index 0..9 rather than the 10..1000 kg/m3 of this table, so the Reynolds number of "
         "that family cannot be read off case.json"
)


def nu_for_grashof(ra: float) -> float:
    """The published mean Nusselt number at Grashof (Rayleigh) number ``ra``, with a message listing the four
    benchmarks when the number is outside the published range."""
    for key, value in DEV_VAHL_DAVIS_NU.items():
        if abs(key - float(ra)) <= 1e-12 * max(1.0, abs(key)):
            return value
    raise KeyError(f"Ra = {ra:g}: the benchmark publishes "
                   f"{', '.join(f'{k:g}' for k in sorted(DEV_VAHL_DAVIS_NU))} only")


def get(name: str) -> Any:
    """``get("ghia-1982")`` -> the four ``u`` profiles; ``get("de-vahl-davis-1983")`` -> the Nusselt numbers."""
    return {"ghia-1982": GHIA_U, "de-vahl-davis-1983": DEV_VAHL_DAVIS_NU}.get(name)


def disputes() -> list[dict[str, Any]]:
    """Every station whose transcribed value our own run contradicts, as records: what, where, and why."""
    return [{"case": p.case, "quantity": p.quantity, "reynolds": p.reynolds, "point": p.points[i],
             "transcribed": p.values[i], "reason": p.dispute}
            for p in PROFILES for i in p.disputed]


def as_dicts() -> list[dict[str, Any]]:
    """Every transcribed table as plain data, for a JSON record next to a run."""
    return [p.to_dict() for p in PROFILES]
