# ProblemDefinition

A `ProblemDefinition` (in code: `ProblemSpec`) encodes everything needed to
describe a physics problem as data, before any network or solver exists:
which PDE, over which coordinates and fields, subject to which boundary/
initial conditions, with which physical scales and domain bounds.

## Where it lives

`ProblemSpec` and its building blocks live in
`pinneapple_physics.pde_environment` (re-exported at `pinneapple_physics`):

- `ProblemSpec` — the full specification: `dim`, `coords`, `fields`, a
  `PDETermSpec` describing the equation, a tuple of `conditions`, a
  `ScaleSpec`, and `domain_bounds`.
- `PDETermSpec` — the equation descriptor: a `kind` string (e.g. `"burgers"`,
  `"navier_stokes_incompressible"`, `"heat"`, `"wave"`, `"elasticity"`,
  `"darcy"`, `"helmholtz"`, `"advection"`, `"reaction_diffusion"`), the
  fields it involves, and numeric `params`.
- `ScaleSpec` — normalization by length `L`, velocity `U`, and diffusivity
  `alpha`.
- `ConditionSpec` and its concrete forms `DirichletBC`, `NeumannBC`,
  `RobinBC`, `InitialCondition`, `DataConstraint` — each pairs a spatial/
  temporal selector with a prescribed value or supervised target.

## Constructing conditions

```python
from pinneapple_physics import DirichletBC, NeumannBC, InitialCondition

# Simple form: dict of field -> constant value
bc = DirichletBC({"u": 0.0, "v": 0.0})

# Full form: named region, explicit selector and value function
bc = DirichletBC(
    "inlet",
    fields=("u", "v"),
    selector_type="callable",
    selector=lambda X, ctx: X[:, 0] < 1e-6,
    value_fn=lambda X, ctx: ...,
    weight=10.0,
)
```

`NeumannBC` prescribes a normal flux, `RobinBC` a linear combination of value
and normal derivative, `InitialCondition` a `t=0` condition, and
`DataConstraint` a supervised loss against measured/reference points.

## Presets

Most problems don't need to be built by hand. `pinneapple_physics` ships
40+ registered presets covering academic PDEs (`burgers_1d`, `laplace_2d`,
`poisson_2d`), CFD (`ns_incompressible_2d/3d`, `lid_driven_cavity_3d`,
`channel_flow_3d`, `pipe_flow_3d`), aerospace, turbomachinery, automotive,
structural, and other engineering domains:

```python
from pinneapple_physics import get_preset, list_presets

spec = get_preset("ns_incompressible_2d", Re=200.0)
list_presets()  # -> all registered preset names
```

Custom presets register the same way, via
`pinneapple_physics.pde_environment.presets.registry.register_preset`.

## Fluent construction and identification

`ProblemBuilder` offers a chained, imperative alternative to constructing a
`ProblemSpec` directly (used throughout the Arena API):

```python
from pinneapple_physics import ProblemBuilder

spec = (
    ProblemBuilder("heat_1d")
    .domain(x=(0, 1), t=(0, 1))
    .fields("u")
    .pde("heat_1d", alpha=0.01)
    .ic(field="u", fn=lambda X: ...)
    .bc("dirichlet", field="u", value=0.0, on="x_boundary")
    .build()
)
```

`identify(description)` (module-level in `pinneapple_physics`, wrapping
`pde_environment.capabilities.identify_pde`) takes a free-text description
and returns ranked guesses at which PDE family it matches — useful as a
first step before picking or writing a preset.

A `ProblemSpec` is pure data: it has no model, no sampled points, and no
loss function yet. Sampling comes from [Geometry & Domain](geometry_domain.md);
turning it into a trainable loss is the job of [PINN / Physics](pinn.md).


## Transforms: `pp.transforms`

`pinneapple_core.transforms` (`pp.transforms`) holds composable transforms that work on tables (`{name: array or tensor}`),
on `PhysicsDataset` objects and, for the scaling ones, on a `PhysicalProblem`.

```python
from pinneapple_core.transforms import Nondimensionalize, Scale, Coordinate, Periodic, FourierFeatures

nd = Nondimensionalize(problem)              # coordinates to [0, 1], fields to O(1), PDE parameters rescaled
small = nd.apply(problem)                    # a new problem; small.metadata["transforms"] records the step
predict = nd.pull_back(solution.predict, problem.coords, problem.fields)    # answers in the original units

t = Scale({"x": 2.0}) >> Coordinate("log", ("t",), ("logt",)) >> FourierFeatures(("x",), 8)
out = t.forward(table); back = t.inverse(out)      # back == table (inverse runs in reverse order)
```

| Transform | What it does | Invertible | Problem-level |
|---|---|---|---|
| `Scale(scale, shift)` | `y = (x - shift) / scale` on named columns | yes | yes |
| `Nondimensionalize(problem)` | `Scale` with scales taken from the problem (extents, field ranges, `U = L/T` for Burgers) | yes | yes |
| `Coordinate(kind, inputs, outputs)` | polar, cylindrical, spherical, log | yes | no |
| `Symmetry.reflect / rotate` | mirror or rotate positions and the vector fields | yes | no |
| `Periodic(coord, period, harmonics)` | `cos`/`sin` embedding, periodic by construction | yes | no |
| `FourierFeatures(coords, n, scale, seed)` | random Fourier features added as columns (`.module()` for a torch layer) | drops the features | no |

On a problem the scaling transforms rewrite the domain, the conditions (selectors and values, with Neumann values scaled by `L/U`) and the PDE
parameters. The rules are in `list_pde_rules()` (`laplace`, `poisson`, `heat_equation`, `wave_equation`, `burgers`); a PDE without a rule, anisotropic
space scaling, or Burgers with `U T != L` raises instead of leaving inconsistent coefficients. `Scale.source(fn, problem)` rescales a source term. Each
`apply` appends `{name, params, invertible, input_fingerprint, output_fingerprint}` to `metadata["transforms"]` (problems) or `provenance` (datasets);
the transformed problem drops its `reference_solver`, which solves the original problem. Example, Burgers with `nu = 0.01` on `x in [-1, 1]`, `t in [0, 1]`:
`L = 2`, `T = 1`, `U = 2`, so `nu' = nu T / L^2 = 0.0025`; the tests check that the rescaled exact solution has a vanishing residual with `nu'` and not with `nu`.
