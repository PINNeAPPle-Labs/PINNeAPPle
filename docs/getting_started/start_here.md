# Start here (docs)

The page you are reading is the site version of [`START_HERE.md`](../../START_HERE.md) at the
repository root. Same content, same four commands; that file is the one that gets executed by
`tests/test_front_door.py`, so it is the one to trust if the two ever disagree.

## Four commands, before any physics

```bash
pip install pinneapple
pinneapple doctor start          # is this machine ready? one fix command per missing item
pinneapple recommend "problem"    # which preset, which method, which model, and why
pinneapple card <preset>          # what it takes, what it gives back, the program to run
pinneapple new my_study           # a project of your own: case.yaml + runner + README
```

In Python: `pp.doctor(profile)`, `pp.recommend(text)`, `pp.card(name)`, `pp.new(name, ...)`.

## Which path is yours

| you are | start with | then |
|---|---|---|
| an engineer with a problem, and no Python | `pinneapple new my_case --template plain`; edit `case.yaml` only | [installation](installation.md), [project layout](project_layout.md) |
| learning physics-AI | `pp.card("burgers_1d")`, then `pp.solve` with the closed form, then train and compare | [first PINN](first_pinn.md), [choosing a method](../guides/choosing_a_method.md) |
| a researcher with pipelines and papers | `pinneapple new my_study --template research` | [adding an experiment](../guides/adding_an_experiment.md), [IPython](quickstart.md) |
| a team putting this behind a product | `pinneapple doctor enterprise` | [physics checks in CI](../guides/physics_checks_in_ci.md), [pp_validate](../guides/pp_validate.md) |

## Describing your own PDE: one route, documented

Two code paths can describe a PDE. Use `ProblemBuilder`; it is the one that trains anywhere:

```python
import numpy as np
import pinneapple as pp
from pinneapple_physics import ProblemBuilder

spec = (ProblemBuilder("my_heat")
        .domain(x=(0.0, 1.0), t=(0.0, 1.0))
        .fields("u")
        .pde("heat_equation", alpha=0.01)
        .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0:1]))
        .bc("dirichlet", field="u", value=0.0, on="x_boundary")
        .build())
sol = pp.solve(spec, method="pinn", epochs=2000)
print(sol.metadata())
```

`SymbolicPDE` (`pinneapple_physics.symbolic_pde`) compiles a SymPy residual with autograd through the
expression tree. It is the advanced route: reach for it when the equation you need is not a kind the
builder knows, and read `compiler.py`'s module docstring first, which explains which of the two
paths to use and why.

## What "it runs" means here

`pp.card(name)` tries the cheapest method once and says whether the preset produced a field on this
machine. Thirty-two of the forty presets whose boundaries are named geometry tags declare the box or
cylinder those tags map onto, and `pp.solve` sends it for you, so they train from their name alone.
When a preset still cannot run, the card names what is missing (FEniCS, OpenFOAM) and hands over
`standalone_snippet(name)`: the same equation and fields, on a box, with boundaries you set.
`pp.recommend(text)` prefers a preset that runs and says which closer one it passed over.
