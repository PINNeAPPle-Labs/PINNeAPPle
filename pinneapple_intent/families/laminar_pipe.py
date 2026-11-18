"""Pressure drop of a laminar flow along a circular pipe.

Three answers to the same question. The closed form is Hagen-Poiseuille, which is exact when the flow is steady,
fully developed and laminar. The numerical rung solves the radial momentum balance of that flow by finite
differences on the real radius (wall no-slip, symmetry on the axis) and integrates the profile to the flow rate, so
it never uses the formula it is checked against. The convergence rung refines the radial grid. ``physics_ai`` mode
adds a surrogate of the pressure drop over the pipe length.

References: Hagen (1839) and Poiseuille (1840) for the law; White, *Viscous Fluid Flow*, ch. 3, for the derivation
and the friction factor ``f = 64/Re``.
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


def hagen_poiseuille(diameter: float, length: float, flow_rate: float, viscosity: float,
                     density: float) -> dict[str, float]:
    """``dp = 8 mu L Q / (pi R^4)`` with the Reynolds number and the mean velocity that say whether it applies."""
    if diameter <= 0 or length <= 0 or viscosity <= 0:
        raise ValueError("diameter, length and viscosity must be positive")
    radius = diameter / 2.0
    area = np.pi * radius**2
    velocity = flow_rate / area
    pressure_drop = 8.0 * viscosity * length * flow_rate / (np.pi * radius**4)
    reynolds = density * velocity * diameter / viscosity
    return {"pressure_drop": float(pressure_drop), "mean_velocity": float(velocity), "reynolds": float(reynolds)}


def radial_profile(diameter: float, length: float, flow_rate: float, viscosity: float, n: int) -> dict[str, float]:
    """Solve ``(1/r) d/dr (r du/dr) = (1/mu) dp/dx`` by finite differences and return the pressure drop that the
    integrated profile implies for the requested flow rate.

    The equation is linear in ``dp/dx``, so it is solved once for a unit gradient and the answer is scaled: the
    integrated flow rate per unit gradient gives the gradient that carries ``flow_rate``. The discretisation is the
    conservative one — fluxes at the cell faces, ``r`` weighted — so the symmetry condition on the axis is the flux
    across ``r = 0`` being zero rather than an assumption about the first node.
    """
    if n < 4:
        raise ValueError("n must be at least 4")
    radius = diameter / 2.0
    nodes = np.linspace(0.0, radius, n)          # u[0] on the axis, u[n-1] at the wall
    dr = nodes[1] - nodes[0]
    faces = (np.arange(n - 1) + 0.5) * dr        # r at the face between node i and i + 1
    m = n - 2                                    # unknowns u[1..n-2]: u[0] mirrors u[1], u[n-1] = 0 (no-slip)
    matrix = np.zeros((m, m))
    rhs = np.zeros(m)
    for i in range(1, n - 1):
        row = i - 1
        below = faces[i - 1]                     # r_{i-1/2}
        above = faces[i] if i < n - 1 else 0.0   # r_{i+1/2}; the wall term is zero because u[n-1] = 0
        if i == 1:
            # the axis node carries the symmetry condition: u[0] = u[1], so the -below * u[0] term folds into
            # u[1] and leaves only the face above it in the diagonal
            matrix[row, row] = above
        else:
            matrix[row, row] = below + above
            matrix[row, row - 1] = -below
        if i < n - 2:
            matrix[row, row + 1] = -above
        rhs[row] = nodes[i] * dr**2 / viscosity
    interior = np.linalg.solve(matrix, rhs)
    profile = np.zeros(n)
    profile[1:-1] = interior
    profile[0] = interior[0] if m else 0.0       # symmetry: the axis value mirrors the first interior node
    flow_per_gradient = float(np.sum(2.0 * np.pi * nodes * profile) * dr)
    if flow_per_gradient <= 0:
        raise ValueError("the finite-difference solve returned a non-positive flow rate")
    gradient = flow_rate / flow_per_gradient
    return {"pressure_drop": float(gradient * length), "flow_per_gradient": flow_per_gradient,
            "radial_nodes": float(n)}


class LaminarPipe(StudyFamily):
    name = "laminar_pipe"
    title = "Pressure drop of a laminar flow along a circular pipe"
    domain = "fluid mechanics"
    objective = "the pressure needed to push a given flow rate through a pipe"
    qoi = "pressure_drop"
    qoi_unit = "Pa"
    qoi_label = "pressure drop"
    keywords = {"pipe": 1.2, "pressure drop": 1.4, "pressure": 0.6, "laminar": 1.2, "poiseuille": 1.4,
                "tube": 0.8, "flow": 0.6, "viscosity": 1.0, "water": 0.8, "oil": 0.8, "duct": 0.6, "hose": 0.8,
                "line": 0.4, "pump": 0.6, "hydraulic": 0.6}
    equations = (
        "mu (1/r) d/dr (r du/dr) = dp/dx   (steady, fully developed, axisymmetric momentum balance)",
        "u(R) = 0 and du/dr(0) = 0   (no-slip at the wall, symmetry on the axis)",
        "dp = 8 mu L Q / (pi R^4)   (Hagen-Poiseuille, the closed form the solver is checked against)",
        "Re = rho u D / mu, laminar when Re < 2300",
    )
    assumptions = (
        "Steady, fully developed, laminar and incompressible flow.",
        "A Newtonian fluid of constant viscosity, in a straight circular pipe of constant section.",
        "The entrance length and the local losses (bends, fittings, a valve) are not part of this study: it prices "
        "the straight run only, so a real circuit needs those added.",
        "No roughness: at the Reynolds numbers this study admits, a laminar profile does not feel it.",
    )
    limitations = (
        "Turbulent flow is outside the family: above the Reynolds number this study reports, the answer it gives is "
        "the answer to a different problem.",
        "No heat transfer, no compressibility, no non-Newtonian rheology, no transient start-up.",
        "The pipe is a straight line: elbows, tees, valves, and the developing region near the inlet are not priced.",
    )
    references = (
        "Hagen, G. (1839) and Poiseuille, J. L. (1840): the fourth-power law for laminar pipe flow.",
        "White, F. M., Viscous Fluid Flow, ch. 3 (the Hagen-Poiseuille solution and f = 64/Re).",
    )
    slots = (
        Slot("diameter", "pipe diameter", unit="m", display_unit="mm", keywords=("diameter", "bore", "pipe"),
             critical=True, bounds=(1e-4, 5.0), role="scale", question="What is the inside diameter of the pipe?"),
        Slot("length", "pipe length", unit="m", keywords=("length", "long", "run"), critical=True,
             bounds=(1e-3, 1e6), role="scale", question="How long is the straight run of pipe?"),
        Slot("flow_rate", "volumetric flow rate", unit="m^3/s", display_unit="L/min",
             keywords=("flow rate", "flow", "l/min", "l/s", "m3/h", "m3/s", "q"), critical=True,
             bounds=(1e-12, 10.0), role="load", question="What volumetric flow rate must the pipe carry?"),
        Slot("viscosity", "dynamic viscosity", unit="Pa*s", display_unit="mPa*s",
             keywords=("viscosity", "mu", "pa·s", "pa s", "cp"), critical=True, bounds=(1e-6, 1e3), role="material",
             question="What is the dynamic viscosity of the fluid?"),
        Slot("density", "density", unit="kg/m^3", keywords=("density", "rho", "kg/m3"), default=998.2,
             default_source="water at 20 °C (998.2 kg/m3); change it for another fluid",
             bounds=(0.01, 2e4), role="material", question="What is the density of the fluid?"),
    )
    hints = (
        ("diameter", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm|m|in|inch)\s*(?:inside\s+|internal\s+|inner\s+)?(?:dia(?:meter)?\s+|bore\s+)?pipe\b"),
        ("length", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm|m|km|in|ft)\s+(?:long|of\s+pipe|run)\b"),
        ("flow_rate", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>l/min|l/s|m3/h|m3/s|gpm|ml/min)\b"),
        ("viscosity", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mpa\.?s|pa\.?s|cp|poise)\b"),
    )
    validation = ("the finite-difference solve against Hagen-Poiseuille",
                  "the pressure drop under radial grid refinement",
                  "the Reynolds number against the laminar limit the family claims")

    # ------------------------------------------------------------------ labels
    def analytic_label(self) -> str:
        return "Hagen-Poiseuille closed form"

    def analytic_method(self) -> str:
        return "the exact solution of the steady, fully developed, axisymmetric momentum balance"

    def numerical_label(self) -> str:
        return "finite differences on the radial momentum balance"

    def numerical_method(self) -> str:
        return ("(1/r) d/dr (r du/dr) = dp/dx discretised on the real radius, wall no-slip and axis symmetry, the "
                "profile integrated to the flow rate and the gradient that carries it solved for")

    # ------------------------------------------------------------------ the rungs
    def evaluate(self, rung: str, contract: StudyContract, *, workdir: str | None = None,
                 options: dict[str, Any] | None = None) -> Measurement:
        options = options or {}
        diameter = numeric(contract.get("diameter"))
        length = numeric(contract.get("length"))
        flow_rate = numeric(contract.get("flow_rate"))
        viscosity = numeric(contract.get("viscosity"))
        density = numeric(contract.get("density"), 998.2)

        if rung == ANALYTIC:
            closed = hagen_poiseuille(diameter, length, flow_rate, viscosity, density)
            laminar = closed["reynolds"] < 2300.0
            return Measurement(
                rung=ANALYTIC,
                outputs=closed, units={"pressure_drop": "Pa", "mean_velocity": "m/s", "reynolds": "1"},
                checks=[check("the flow is laminar, where the closed form applies", laminar,
                              value=round(closed["reynolds"], 1), limit=2300.0,
                              detail="above Re = 2300 the laminar law stops describing the flow and this study's "
                                     "answer is the answer to a different problem")],
                notes=[f"mean velocity {closed['mean_velocity']:.4g} m/s, Re = {closed['reynolds']:.1f}"])

        if rung == NUMERICAL:
            return self._finite_difference(rung, diameter, length, flow_rate, viscosity, density,
                                           int(options.get("n") or 256))
        if rung == CONVERGENCE:
            return self._convergence(diameter, length, flow_rate, viscosity, density)

        if rung == SURROGATE:
            return self._surrogate(contract, diameter, length, flow_rate, viscosity, density)

        raise ValueError(f"unknown rung {rung!r}")

    def _finite_difference(self, rung: str, diameter: float, length: float, flow_rate: float, viscosity: float,
                           density: float, n: int) -> Measurement:
        closed = hagen_poiseuille(diameter, length, flow_rate, viscosity, density)
        solved, seconds = self.timed(lambda: radial_profile(diameter, length, flow_rate, viscosity, n))
        got = solved["pressure_drop"]
        err = relative_error(got, closed["pressure_drop"])
        return Measurement(
            rung=rung,
            outputs={"pressure_drop": got, "reynolds": closed["reynolds"], "radial_nodes": float(n)},
            units={"pressure_drop": "Pa", "reynolds": "1", "radial_nodes": "1"},
            checks=[
                check("the discrete profile is positive everywhere", solved["flow_per_gradient"] > 0,
                      value=solved["flow_per_gradient"], limit="> 0"),
                check("the finite-difference solve is within the family's tolerance of the closed form",
                      err <= self.tolerance, value=round(err, 6), limit=self.tolerance,
                      detail=f"{got:.6g} Pa from the discrete profile against {closed['pressure_drop']:.6g} Pa "
                             f"from Hagen-Poiseuille; relative error {err:.3%}"),
            ],
            notes=[f"{n} radial nodes over a radius of {diameter / 2 * 1e3:.2f} mm"], wall_s=seconds)

    def _convergence(self, diameter: float, length: float, flow_rate: float, viscosity: float,
                     density: float) -> Measurement:
        """The same solve on a coarse and a fine radial grid, so the discretisation error is measured."""
        coarse = self._finite_difference(CONVERGENCE, diameter, length, flow_rate, viscosity, density, 16)
        fine = self._finite_difference(CONVERGENCE, diameter, length, flow_rate, viscosity, density, 256)
        coarse_value = coarse.outputs["pressure_drop"]
        fine_value = fine.outputs["pressure_drop"]
        return Measurement(
            rung=CONVERGENCE,
            outputs={"pressure_drop": fine_value, "coarse_pressure_drop": coarse_value,
                     "radial_nodes": 256.0, "coarse_radial_nodes": 16.0},
            units={"pressure_drop": "Pa", "coarse_pressure_drop": "Pa", "radial_nodes": "1",
                   "coarse_radial_nodes": "1"},
            checks=coarse.checks + fine.checks, wall_s=coarse.wall_s + fine.wall_s,
            notes=[f"16 then 256 radial nodes; {coarse_value:.6g} Pa then {fine_value:.6g} Pa"])

    def _surrogate(self, contract: StudyContract, diameter: float, length: float, flow_rate: float,
                   viscosity: float, density: float) -> Measurement:
        import torch

        from pinneapple_design.surrogate_inverse.core import fit_surrogate

        n = 24
        rng = np.random.default_rng(contract.provenance.seed)
        low, high = 0.5 * length, 2.0 * length
        design = np.sort(rng.uniform(low, high, size=n))
        values = [radial_profile(diameter, float(one), flow_rate, viscosity, 128)["pressure_drop"] for one in design]
        fit = fit_surrogate(torch.tensor(design.reshape(-1, 1), dtype=torch.float64),
                            torch.tensor(np.asarray(values).reshape(-1, 1), dtype=torch.float64),
                            bounds=[(float(low), float(high))], epochs=400, val_fraction=0.25,
                            seed=contract.provenance.seed)
        return Measurement(
            rung=SURROGATE,
            outputs={"held_out_relative_error": float(fit.val_mean_rel_err), "n_designs": float(n)},
            units={"held_out_relative_error": "1"},
            checks=[check("the surrogate was fitted and scored on held-out designs", fit.n_val > 0,
                          value=fit.n_val, limit=">= 1")],
            notes=[f"the pipe length swept over {low:.3g} to {high:.3g} m, {n} solver runs; mean relative error on "
                   f"the {fit.n_val} held-out runs {fit.val_mean_rel_err:.3%} (p95 {fit.val_p95_rel_err:.3%})"])


PIPE = LaminarPipe()

__all__ = ["LaminarPipe", "PIPE", "hagen_poiseuille", "radial_profile"]
