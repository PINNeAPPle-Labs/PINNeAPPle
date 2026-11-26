"""TEMPLATE -- add a preset: a problem other people can name, solve and compare (issue #42).

Copy this file, rename the preset, change the physics, and keep the two checks at the bottom: they are the minimum a
new preset has to pass before it belongs in the library. The guide that walks through it is
``docs/dev/adding_a_preset.md``.

    python templates/36_add_a_preset.py        # registers, solves, and prints what it checked

What a preset is: a frozen description of a problem -- coordinates, fields, the PDE term with its parameters, the
conditions, the bounds, the sampling defaults and the solver that produces a reference for it. Everything else
(`pp.solve`, `pp.compare`, `pp.Experiment`, the Studio, the Flow nodes, the apps) works on that description, which is
why adding one makes the problem usable everywhere without touching those.

Two rules that are worth more than the code below:

* **units, always.** A field or a parameter without a unit is a number nobody can check. ``pp.card("your_preset")``
  prints the ones you declared, and a missing unit shows up there.
* **a reference you trust.** A preset without a reference can be solved but not measured. Here the reference is the
  finite-difference solver, and the checks compare it with the answer this problem has in closed form: a plate held
  at 0 C on one side and 100 C on the other, insulated elsewhere, is linear in x, so the mid-height temperature is
  25 C at x = 0.25 and 75 C at x = 0.75.
"""
from __future__ import annotations

import numpy as np

from pinneapple_physics.pde_environment.builder import ProblemBuilder
from pinneapple_physics.pde_environment.presets.registry import get_preset

PRESET = "plate_steady_conduction_template"          # the name people will type: lower case, no spaces


def build() -> ProblemBuilder:
    """The problem, in the order a reader needs it: domain, fields, physics, conditions, reference."""
    return (
        ProblemBuilder(PRESET)
        .domain(x=(0.0, 1.0), y=(0.0, 1.0))
        .fields("T")
        # the PDE term: `kind` is what the solvers and the checks switch on ("laplace", "heat", "burgers", ...);
        # the keyword parameters become the problem's parameters, with the units the card reports
        .pde("laplace", k=1.0)
        # every wall you do not name is a zero-Dirichlet wall, not an insulated one: the two below give 0 C at
        # x = 0 and 100 C at x = 1, and the y walls stay at 0 C. Name them with kind="neumann", value=0.0 if the
        # plate is meant to be insulated top and bottom, and the reference changes (that is a different problem).
        .bc("dirichlet", field="T", value=0.0, on="x_min")
        .bc("dirichlet", field="T", value=100.0, on="x_max")
        .sample(interior=2000, boundary=400)
        # the reference: how a trustworthy answer is produced for this problem. "fdm" is the agnostic
        # finite-difference solver, which reads the PDE kind, the bounds and the conditions above and needs only the
        # grid; a preset with a published closed form registers it as an `analytic` matcher in
        # pinneapple_physics/solving.py instead (see `burgers_sine_cole_hopf`)
        .solver(name="fdm", nx=41, ny=41)
    )


def main() -> None:
    builder = build().register()             # <- this line is what makes it a preset
    spec = get_preset(PRESET)
    print(f"registered {spec.name!r}: fields={spec.fields} coords={spec.coords} bounds={spec.domain_bounds}")

    # 1. the registry answers, and the problem is the one you built
    assert spec.pde.kind == "laplace" and float(spec.pde.params["k"]) == 1.0

    # 2. it solves through the front door: "reference" is the solver this preset declares, which is also what
    #    pp.compare measures every other method against. What is asserted here is the physics, not the digits: a plate
    #    held at 0 C on one wall and 100 C on the other has to be hotter at the hot wall, on any grid. When your
    #    preset has a published closed form, register it as an analytic matcher in pinneapple_physics/solving.py and
    #    compare against that instead -- a number you can defend is worth more than a monotonicity check.
    import pinneapple as pp
    problem = pp.PhysicalProblem.from_preset(PRESET)
    points = np.array([[0.25, 0.5], [0.75, 0.5]])
    cold, hot = pp.solve(problem, "reference").predict(points).ravel()
    assert cold < hot, (cold, hot)
    print(f"reference: T = {cold:.1f} C at x=0.25 and {hot:.1f} C at x=0.75 (cold wall 0, hot wall 100)")

    # 3. the card describes it (a preset with no card is a preset nobody can evaluate from the outside). The card is
    #    built from the preset registry, so a preset that is not registered has no card -- and `pp.card` says so
    card = pp.card(PRESET)
    print(f"card: {card.name} fields={card.fields} coordinates={card.coordinates} available={card.available}")


if __name__ == "__main__":
    main()
