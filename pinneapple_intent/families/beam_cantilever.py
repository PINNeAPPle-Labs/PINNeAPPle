"""Bending of a cantilever beam under a tip load.

The closed form is Euler-Bernoulli beam theory, which is exact for a slender beam and neglects shear. The solver is
the library's immersed finite-cell elasticity on the two-dimensional section, which is **plane strain** — so the
modulus the closed form must use is ``E / (1 - nu^2)``, not ``E``. Getting that convention wrong is the classic way
this comparison is made to look better or worse than it is; the family states it, uses it, and puts it in the report.

References: Euler (1744) and Bernoulli for the elastic curve; Timoshenko, *Strength of Materials*, part I, ch. 5, for
the cantilever with an end load and for the shear correction the closed form omits; Boresi & Chong, *Elasticity*, for
plain strain replacing ``E`` with ``E / (1 - nu^2)``.
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


def closed_form(length: float, height: float, depth: float, youngs_modulus: float, poisson: float,
                load: float) -> dict[str, float]:
    """Tip deflection of a cantilever on the plane-strain convention the two-dimensional solver uses.

    ``delta = P L^3 / (3 E' I)`` with ``E' = E / (1 - nu^2)`` and ``I = h^3 / 12`` for one metre of depth; ``P`` is
    the tip load per metre of depth, which is how the two-dimensional model feels it.
    """
    if length <= 0 or height <= 0 or depth <= 0 or youngs_modulus <= 0 or load <= 0:
        raise ValueError("length, height, depth, modulus and load must all be positive")
    plane_strain_modulus = youngs_modulus / (1.0 - poisson**2)
    inertia = height**3 / 12.0
    load_per_depth = load / depth
    deflection = load_per_depth * length**3 / (3.0 * plane_strain_modulus * inertia)
    max_stress = load_per_depth * length * (height / 2.0) / inertia
    return {"tip_deflection": float(deflection), "max_bending_stress": float(max_stress),
            "plane_strain_modulus": float(plane_strain_modulus), "slenderness": float(length / height)}


class CantileverBeam(StudyFamily):
    name = "cantilever_beam"
    title = "Tip deflection of a cantilever beam under an end load"
    domain = "solid mechanics"
    objective = "how far the end of a cantilever moves under a load"
    qoi = "tip_deflection"
    qoi_unit = "m"
    qoi_label = "tip deflection"
    keywords = {"beam": 1.2, "cantilever": 1.4, "deflection": 1.2, "bending": 1.0, "bend": 0.8, "elasticity": 0.8,
                "structural": 0.6, "young": 1.2, "modulus": 1.0, "stress": 0.5, "bracket": 0.8, "load": 0.4,
                "stiffness": 1.0, "arm": 0.6, "strut": 0.8, "section": 0.4}
    equations = (
        "E' I d^2 w/dx^2 = M(x),  E' = E / (1 - nu^2)   (Euler-Bernoulli, plane strain)",
        "w(0) = 0, w'(0) = 0   (clamped root)",
        "delta = P L^3 / (3 E' I),  I = h^3 / 12   (tip load, one metre of depth)",
        "sigma_max = P L (h/2) / I   (outer fibre, bending only)",
    )
    assumptions = (
        "Small deflection, linear elasticity, and a beam slender enough for Euler-Bernoulli: the closed form has no "
        "shear deformation, which is why the family reports the slenderness L/h next to the answer.",
        "The two-dimensional solver is plane strain, so the closed form compares against E / (1 - nu^2).",
        "The root is fully clamped (both displacement components fixed) and the section is rectangular, "
        "homogeneous and isotropic.",
    )
    limitations = (
        "No plasticity, no buckling, no contact and no fatigue: the answer is the elastic response to one load.",
        "Deflections above roughly a tenth of the span are outside the linear theory this compares with.",
        "The stress reported is the bending stress of the closed form, not the peak the solver finds at the clamp, "
        "where a real fillet and a real contact pressure change the number.",
    )
    references = (
        "Timoshenko, Strength of Materials, part I, ch. 5 (cantilever with an end load, and the shear correction "
        "this closed form omits).",
        "Boresi & Chong, Elasticity in Engineering Mechanics (plane strain replaces E with E / (1 - nu^2)).",
    )
    slots = (
        Slot("length", "beam length", unit="m", display_unit="mm", keywords=("length", "long", "span"),
             critical=True, bounds=(1e-3, 100.0), role="scale", question="How long is the beam?"),
        Slot("height", "section depth in the plane of bending", unit="m", display_unit="mm",
             keywords=("height", "thickness", "depth in plane", "section"), critical=True, bounds=(1e-4, 10.0),
             role="scale", question="What is the beam's section depth in the plane of bending?"),
        Slot("depth", "out-of-plane depth", unit="m", display_unit="mm", keywords=("width", "out of plane", "depth"),
             default=1.0, default_source="1 m, the unit depth of the two-dimensional model; the load is entered as "
                                          "a total over it and divided by it",
             bounds=(1e-3, 100.0), role="scale", question="What is the out-of-plane depth of the beam?"),
        Slot("youngs_modulus", "Young's modulus", unit="Pa", display_unit="GPa",
             keywords=("young", "modulus", "gpa", "stiffness"), critical=True, bounds=(1e5, 1e13), role="material",
             question="What is the Young's modulus of the material?"),
        Slot("poisson", "Poisson's ratio", unit="", keywords=("poisson", "ratio"), default=0.3,
             default_source="0.3, the usual value for steel and aluminium alloys",
             bounds=(-0.99, 0.49), role="material", question="What is the Poisson's ratio?"),
        Slot("load", "tip load", unit="N", display_unit="kN", keywords=("load", "force", "n", "kn"),
             critical=True, bounds=(1e-6, 1e9), role="load", question="What load acts at the tip?"),
    )
    hints = (
        ("length", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm|m)\s+(?:long|span|beam)\b"),
        ("height", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm|m)\s+(?:deep|depth|high|tall|thick)\b"),
        ("load", r"(?i)\b(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>n|kn|mn)\s*(?:of\s+)?(?:load|force)\b"),
        ("youngs_modulus", r"(?i)\be\s*=\s*(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>gpa|mpa|kpa|pa)\b"),
    )
    # the bending gradient is across the section and the immersed stencil is wider than a beam element's, so the
    # section needs more cells than a heat wall does to show the same convergence
    cells_through_scale = 24
    validation = ("the solver against the closed form on the same inputs",
                  "the quantity of interest under grid refinement",
                  "the slenderness, against the range Euler-Bernoulli applies to")

    # ------------------------------------------------------------------ labels
    def length_scale(self, contract: StudyContract) -> float:
        """The section depth: the bending gradient is across it, so the grid has to resolve it, not the span."""
        return numeric(contract.get("height"), 0.02)

    def analytic_label(self) -> str:
        return "Euler-Bernoulli closed form (plane strain)"

    def analytic_method(self) -> str:
        return "the elastic curve of a cantilever with an end load, with E replaced by E/(1 - nu^2)"

    def numerical_label(self) -> str:
        return "immersed finite cell, plane-strain linear elasticity"

    def numerical_method(self) -> str:
        return ("pinneapple_simulation.immersed.solve_elasticity on a box of the section: the root face fully "
                "clamped, the tip face loaded in shear")

    # ------------------------------------------------------------------ the rungs
    def evaluate(self, rung: str, contract: StudyContract, *, workdir: str | None = None,
                 options: dict[str, Any] | None = None) -> Measurement:
        options = options or {}
        length = numeric(contract.get("length"))
        height = numeric(contract.get("height"))
        depth = numeric(contract.get("depth"), 1.0)
        youngs = numeric(contract.get("youngs_modulus"))
        poisson = numeric(contract.get("poisson"), 0.3)
        load = numeric(contract.get("load"))

        if rung == ANALYTIC:
            closed = closed_form(length, height, depth, youngs, poisson, load)
            slender = closed["slenderness"] >= 10.0
            return Measurement(
                rung=ANALYTIC, outputs=closed,
                units={"tip_deflection": "m", "max_bending_stress": "Pa", "plane_strain_modulus": "Pa",
                       "slenderness": "1"},
                checks=[check("the beam is slender enough for Euler-Bernoulli", slender,
                              value=round(closed["slenderness"], 2), limit=10.0,
                              detail="below L/h = 10 shear deformation is no longer negligible and the closed form "
                                     "underestimates the deflection")],
                notes=[f"slenderness L/h = {closed['slenderness']:.1f}, outer-fibre bending stress "
                       f"{closed['max_bending_stress'] / 1e6:.4g} MPa"])

        if rung == NUMERICAL:
            spacing = float(options.get("h") or self.spacing(contract, NUMERICAL))
            return self._immersed(rung, length, height, depth, youngs, poisson, load, spacing)

        if rung == CONVERGENCE:
            return self._convergence(length, height, depth, youngs, poisson, load,
                                     float(options.get("h") or self.spacing(contract, CONVERGENCE)))

        if rung == SURROGATE:
            return self._surrogate(contract, length, height, depth, youngs, poisson, load)

        raise ValueError(f"unknown rung {rung!r}")

    def _immersed(self, rung: str, length: float, height: float, depth: float, youngs: float, poisson: float,
                  load: float, spacing: float) -> Measurement:
        from pinneapple_simulation.immersed import (
            Discretization,
            Displacement,
            Force,
            box,
            solve_elasticity,
        )

        geometry = box((0.0, 0.0), (length, height))
        discretization = Discretization(geometry, spacing)
        root = lambda P: np.abs(P[:, 0]) < 1e-12                                 # noqa: E731 - face selector
        tip = lambda P: np.abs(P[:, 0] - length) < 1e-12                         # noqa: E731 - face selector
        force = (0.0, -load / depth)                                            # per metre of depth
        solution, seconds = self.timed(lambda: solve_elasticity(
            discretization, E=youngs, nu=poisson,
            bcs=[Displacement(0.0, where=root), Force(force, where=tip)]))

        probe = np.array([[length, height / 2.0]])
        moved = float(solution(probe)[0, 1])
        reference = closed_form(length, height, depth, youngs, poisson, load)
        deflection = abs(moved)
        balance = solution.balance or {}
        checks = [check("the solution is finite", bool(np.isfinite(deflection)), value=deflection, limit="finite")]
        reaction = balance.get("reaction_N")
        if reaction is not None:
            applied = abs(float(force[1]))
            checks.append(check("the root reaction balances the applied load", abs(abs(float(reaction[1])) - applied)
                                / applied <= 1e-6, value=float(reaction[1]), limit=-applied,
                                detail="equilibrium of the discrete system"))
        err = relative_error(deflection, reference["tip_deflection"])
        checks.append(check("the solver is within the family's tolerance of the closed form",
                            err <= self.tolerance, value=round(err, 6), limit=self.tolerance,
                            detail=f"{deflection:.6g} m from the solver against "
                                   f"{reference['tip_deflection']:.6g} m from the closed form; relative error "
                                   f"{err:.2%}. The gap is the discretisation of the clamp and the shear the closed "
                                   f"form omits"))
        return Measurement(
            rung=rung, outputs={"tip_deflection": deflection, "grid_spacing": spacing},
            units={"tip_deflection": "m", "grid_spacing": "m"}, checks=checks, wall_s=seconds,
            notes=[f"background grid {spacing * 1e3:.2f} mm, {discretization.n_dofs} degrees of freedom"])

    def _convergence(self, length: float, height: float, depth: float, youngs: float, poisson: float,
                     load: float, fine_spacing: float) -> Measurement:
        coarse_spacing = 2.0 * fine_spacing
        coarse = self._immersed(CONVERGENCE, length, height, depth, youngs, poisson, load, coarse_spacing)
        fine = self._immersed(CONVERGENCE, length, height, depth, youngs, poisson, load, fine_spacing)
        coarse_value = coarse.outputs["tip_deflection"]
        fine_value = fine.outputs["tip_deflection"]
        return Measurement(
            rung=CONVERGENCE,
            outputs={"tip_deflection": fine_value, "coarse_tip_deflection": coarse_value,
                     "grid_spacing": fine_spacing, "coarse_grid_spacing": coarse_spacing},
            units={"tip_deflection": "m", "coarse_tip_deflection": "m", "grid_spacing": "m",
                   "coarse_grid_spacing": "m"},
            checks=coarse.checks + fine.checks, wall_s=coarse.wall_s + fine.wall_s,
            notes=[f"coarse {coarse_spacing * 1e3:.2f} mm and fine {fine_spacing * 1e3:.2f} mm; "
                   f"{coarse_value:.6g} m then {fine_value:.6g} m"])

    def _surrogate(self, contract: StudyContract, length: float, height: float, depth: float, youngs: float,
                   poisson: float, load: float) -> Measurement:
        import torch

        from pinneapple_design.surrogate_inverse.core import fit_surrogate

        n = 24
        rng = np.random.default_rng(contract.provenance.seed)
        low, high = 0.5 * length, 1.5 * length
        design = np.sort(rng.uniform(low, high, size=n))
        values = []
        for one in design:
            measurement = self._immersed(NUMERICAL, float(one), height, depth, youngs, poisson, load,
                                         self.spacing(contract, NUMERICAL) * 2.0)
            values.append(measurement.outputs["tip_deflection"])
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
            notes=[f"the beam length swept over {low:.3g} to {high:.3g} m, {n} solver runs; mean relative error on "
                   f"the {fit.n_val} held-out runs {fit.val_mean_rel_err:.3%} (p95 {fit.val_p95_rel_err:.3%})"])


BEAM = CantileverBeam()

__all__ = ["CantileverBeam", "BEAM", "closed_form"]
