"""Heat conduction through a wall, with convection on the cold side.

The study answers one question — how hot does the far face run — and answers it three times: by the series
resistance of one-dimensional steady conduction (exact when the wall is thin against its width and the edges are
insulated, which is the assumption the family states), by the immersed finite-cell solver on the real two-dimensional
wall, and by the same solver on a second grid so the discretisation error is visible rather than assumed.

Reference: Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, the one-dimensional plane wall with a
convection boundary (the series-resistance result ``q'' = (T_h - T_inf) / (L/k + 1/h)``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pinneapple_design.intent.spec import Slot

from ..contract import StudyContract
from ..family import (
    ANALYTIC,
    CONVERGENCE,
    NUMERICAL,
    SURROGATE,
    Measurement,
    StudyFamily,
    check,
    numeric,
    relative_error,
)

KELVIN = 273.15


def closed_form(thickness: float, conductivity: float, hot_side: float, ambient: float,
                h_conv: float) -> dict[str, float]:
    """One-dimensional steady conduction through the wall with convection on the far face.

    ``q'' = (T_hot - T_ambient) / (L/k + 1/h)`` and the far face sits at ``T_ambient + q''/h``. The wall's own
    resistance and the film resistance add in series, which is the whole physics of this study.
    """
    if not (thickness > 0 and conductivity > 0 and h_conv > 0):
        raise ValueError("thickness, conductivity and the convection coefficient must all be positive")
    resistance = thickness / conductivity + 1.0 / h_conv
    flux = (hot_side - ambient) / resistance
    return {"cold_face_temperature": ambient + flux / h_conv, "heat_flux": flux,
            "thermal_resistance": resistance}


class HeatWall(StudyFamily):
    name = "heat_wall"
    title = "Steady heat conduction through a wall, cooled by convection"
    domain = "heat transfer"
    objective = "the steady temperature of the wall and its far face"
    qoi = "cold_face_temperature"
    qoi_unit = "K"
    qoi_label = "cold-face temperature"
    keywords = {"heat": 1.0, "thermal": 1.0, "temperature": 0.8, "conduction": 1.2, "conductivity": 1.2,
                "insulat": 0.8, "wall": 0.8, "cooling": 0.6, "convection": 1.0, "steel": 0.6, "furnace": 0.8,
                "oven": 0.8, "hot face": 1.0, "far face": 1.0, "cold face": 0.8, "slab": 0.8, "panel": 0.6}
    equations = (
        "d/dx (k dT/dx) = 0   (one-dimensional steady conduction, no source)",
        "q'' = h (T_cold - T_inf)   (convection on the far face)",
        "T_cold = T_inf + q'' / h,  q'' = (T_hot - T_inf) / (L/k + 1/h)   (series resistance)",
    )
    assumptions = (
        "Steady state and no volumetric source.",
        "Conduction across the wall is one-dimensional: the wall is long and wide against its thickness and its "
        "edges are insulated, so the two-dimensional solve should agree with the closed form.",
        "The conductivity is constant (temperature-independent), and the surface is grey and diffuse with a single "
        "convection coefficient over the whole face.",
    )
    limitations = (
        "No radiation: a hot face above roughly 400 K in still air loses a share of its heat by radiation that this "
        "study does not model.",
        "No contact resistance, no layered wall, no transient: the answer is the steady state of one homogeneous "
        "slab.",
        "The convection coefficient is an input, not a prediction: this study does not compute h from a flow.",
    )
    references = (
        "Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, ch. 3 (one-dimensional steady conduction, "
        "thermal resistance network).",
    )
    slots = (
        Slot("thickness", "wall thickness", unit="m", display_unit="mm", keywords=("thickness", "thick", "wall"),
             critical=True, bounds=(1e-6, 10.0), role="scale",
             question="How thick is the wall?"),
        Slot("conductivity", "thermal conductivity", unit="W/(m*K)", keywords=("conductivity", "k", "w/mk"),
             critical=True, bounds=(1e-3, 1e4), role="material",
             question="What is the thermal conductivity of the wall?"),
        Slot("hot_side", "hot-face temperature", unit="K", display_unit="degC",
             keywords=("hot", "inside", "inner", "temperature", "°c", "celsius", "source"), critical=True, bounds=(1.0, 5000.0),
             role="load", question="What is the temperature of the hot face?"),
        Slot("ambient", "ambient temperature", unit="K", display_unit="degC",
             keywords=("ambient", "air", "environment", "surroundings"), default=293.15,
             default_source="20 °C, the usual laboratory ambient; change it if the air is different",
             bounds=(1.0, 2000.0), role="environment", question="What is the ambient temperature?"),
        Slot("h_conv", "convection coefficient", unit="W/(m2*K)", keywords=("convection", "film", "h", "w/m2k"),
             critical=True, bounds=(0.1, 1e5), role="environment",
             question="What is the convection coefficient on the cold face?"),
        Slot("width", "wall width", unit="m", display_unit="mm", keywords=("width", "wide"),
             default=0.04, default_source="40 mm, the width of the two-dimensional model; the closed form does not "
                                           "depend on it", bounds=(1e-4, 10.0), role="scale",
             question="How wide is the wall?"),
    )
    hints = (
        ("thickness", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm|m|in)\s+(?:thick|thickness|wall)\b"),
        ("conductivity", r"(?i)\bk\s*=\s*(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>w/m\*k|w/mk|w/\(m\*k\))\b"),
        ("h_conv", r"(?i)\bh\s*=\s*(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>w/m\^2\*k|w/m2k|w/\(m\^2\*k\))\b"),
    )
    validation = ("the solver against the closed form on the same inputs",
                  "the quantity of interest under grid refinement")

    # ------------------------------------------------------------------ labels
    def length_scale(self, contract: StudyContract) -> float:
        """The wall's own thickness: a grid coarser than that cannot show the gradient through it."""
        return numeric(contract.get("thickness"), 0.04)

    def analytic_label(self) -> str:
        return "series-resistance closed form"

    def analytic_method(self) -> str:
        return "one-dimensional steady conduction with a convective film resistance in series"

    def numerical_label(self) -> str:
        return "immersed finite cell, two-dimensional"

    def numerical_method(self) -> str:
        return ("pinneapple_simulation.immersed.solve_heat on a box of the wall: the hot face fixed at its "
                "temperature, a convection boundary on the cold face, the edges insulated")

    # ------------------------------------------------------------------ the rungs
    def evaluate(self, rung: str, contract: StudyContract, *, workdir: str | None = None,
                 options: dict[str, Any] | None = None) -> Measurement:
        options = options or {}
        thickness = numeric(contract.get("thickness"))
        conductivity = numeric(contract.get("conductivity"))
        hot_side = numeric(contract.get("hot_side"))
        ambient = numeric(contract.get("ambient"), 293.15)
        h_conv = numeric(contract.get("h_conv"))
        width = numeric(contract.get("width"), 0.04)

        if rung == ANALYTIC:
            closed = closed_form(thickness, conductivity, hot_side, ambient, h_conv)
            return Measurement(
                rung=ANALYTIC, outputs=closed,
                units={"cold_face_temperature": "K", "heat_flux": "W/m2", "thermal_resistance": "m2*K/W"},
                checks=[check("the closed form returns a finite far-face temperature",
                              np.isfinite(closed["cold_face_temperature"]),
                              value=closed["cold_face_temperature"], limit="finite")],
                notes=[f"q'' = {closed['heat_flux']:.6g} W/m2 through a resistance of "
                       f"{closed['thermal_resistance']:.6g} m2 K/W"])

        if rung == NUMERICAL:
            spacing = float(options.get("h") or self.spacing(contract, NUMERICAL))
            return self._immersed(rung, thickness, conductivity, hot_side, ambient, h_conv, width, spacing)

        if rung == CONVERGENCE:
            return self._convergence(thickness, conductivity, hot_side, ambient, h_conv, width,
                                     float(options.get("h") or self.spacing(contract, CONVERGENCE)))

        if rung == SURROGATE:
            return self._surrogate(contract, thickness, conductivity, hot_side, ambient, h_conv, width)

        raise ValueError(f"unknown rung {rung!r}")

    def _convergence(self, thickness: float, conductivity: float, hot_side: float, ambient: float, h_conv: float,
                     width: float, fine_spacing: float) -> Measurement:
        """The same solve twice: the coarse spacing and the fine one, so the change is measured, not assumed."""
        coarse_spacing = 2.0 * fine_spacing
        coarse = self._immersed(CONVERGENCE, thickness, conductivity, hot_side, ambient, h_conv, width,
                                coarse_spacing)
        fine = self._immersed(CONVERGENCE, thickness, conductivity, hot_side, ambient, h_conv, width, fine_spacing)
        coarse_value = coarse.outputs["cold_face_temperature"]
        fine_value = fine.outputs["cold_face_temperature"]
        return Measurement(
            rung=CONVERGENCE,
            outputs={"cold_face_temperature": fine_value, "coarse_cold_face_temperature": coarse_value,
                     "grid_spacing": fine_spacing, "coarse_grid_spacing": coarse_spacing},
            units={"cold_face_temperature": "K", "coarse_cold_face_temperature": "K", "grid_spacing": "m",
                   "coarse_grid_spacing": "m"},
            checks=coarse.checks + fine.checks,
            wall_s=coarse.wall_s + fine.wall_s,
            notes=[f"coarse {coarse_spacing * 1e3:.2f} mm and fine {fine_spacing * 1e3:.2f} mm; "
                   f"{coarse_value:.6g} K then {fine_value:.6g} K"])

    def _immersed(self, rung: str, thickness: float, conductivity: float, hot_side: float, ambient: float,
                  h_conv: float, width: float, spacing: float) -> Measurement:
        from pinneapple_simulation.immersed import (
            Convection,
            Dirichlet,
            Discretization,
            box,
            solve_heat,
        )

        geometry = box((0.0, 0.0), (thickness, width))
        discretization = Discretization(geometry, spacing)
        hot = lambda P: np.abs(P[:, 0]) < 1e-12                                  # noqa: E731 - face selector
        cold = lambda P: np.abs(P[:, 0] - thickness) < 1e-12                     # noqa: E731 - face selector
        solution, seconds = self.timed(lambda: solve_heat(
            discretization, conductivity=conductivity, source=0.0,
            bcs=[Dirichlet(hot_side, where=hot), Convection(h_conv, ambient, where=cold)]))

        probe = np.array([[thickness, width / 2.0]])
        cold_face = float(solution(probe)[0])
        reference = closed_form(thickness, conductivity, hot_side, ambient, h_conv)
        balance = solution.balance or {}
        energy = balance.get("imbalance")
        checks = [check("the solution is finite", bool(np.isfinite(cold_face)), value=cold_face, limit="finite")]
        if energy is not None:
            checks.append(check("energy balance of the discrete system", float(energy) <= 1e-8,
                                value=float(energy), limit=1e-8,
                                detail="the net boundary flux equals the source integral"))
        checks.append(check("the two-dimensional solve is within the family's tolerance of the closed form",
                            relative_error(cold_face, reference["cold_face_temperature"]) <= self.tolerance,
                            value=round(relative_error(cold_face, reference["cold_face_temperature"]), 6),
                            limit=self.tolerance,
                            detail="the edges are insulated, so the two answers should coincide; the gap is the "
                                   "discretisation error plus the film treatment at the cold face"))
        return Measurement(
            rung=rung, outputs={"cold_face_temperature": cold_face, "grid_spacing": spacing},
            units={"cold_face_temperature": "K", "grid_spacing": "m"}, checks=checks, wall_s=seconds,
            notes=[f"background grid {spacing * 1e3:.2f} mm, {discretization.n_dofs} degrees of freedom"])

    def _surrogate(self, contract: StudyContract, thickness: float, conductivity: float, hot_side: float,
                   ambient: float, h_conv: float, width: float) -> Measurement:
        """A sweep of the wall thickness through the solver, and a fitted surrogate scored on held-out thicknesses.

        The sweep is the only expensive part of this rung; it runs the *same* numerical rung, so the surrogate is
        scored against the solver it replaces, not against the closed form.
        """
        import torch

        from pinneapple_design.surrogate_inverse.core import fit_surrogate

        n = 24
        rng = np.random.default_rng(contract.provenance.seed)
        low, high = 0.4 * thickness, 1.6 * thickness
        design = np.sort(rng.uniform(low, high, size=n))
        values = []
        for one in design:
            measurement = self._immersed(NUMERICAL, float(one), conductivity, hot_side, ambient, h_conv, width,
                                         self.spacing(contract, NUMERICAL) * 2.0)
            values.append(measurement.outputs["cold_face_temperature"])
        y = np.asarray(values)
        x = design.reshape(-1, 1)
        fit = fit_surrogate(torch.tensor(x, dtype=torch.float64), torch.tensor(y.reshape(-1, 1), dtype=torch.float64),
                            bounds=[(float(low), float(high))], epochs=400, val_fraction=0.25,
                            seed=contract.provenance.seed)
        return Measurement(
            rung=SURROGATE, outputs={"held_out_relative_error": float(fit.val_mean_rel_err),
                                     "n_designs": float(n)},
            units={"held_out_relative_error": "1"},
            checks=[check("the surrogate was fitted and scored on held-out designs", fit.n_val > 0,
                          value=fit.n_val, limit=">= 1")],
            notes=[f"the wall thickness swept over {low * 1e3:.1f} to {high * 1e3:.1f} mm, {n} solver runs at the "
                   f"coarse spacing; mean relative error on the {fit.n_val} held-out designs "
                   f"{fit.val_mean_rel_err:.3%} (p95 {fit.val_p95_rel_err:.3%})"])


WALL = HeatWall()

__all__ = ["HeatWall", "WALL", "closed_form"]
