# Public namespace

This is the decision behind roadmap item X17 (#201): what `import pinneapple as pp`
exposes, and how the project describes itself. It is a target, not a description
of the current release — the "today" column says what exists, because a target
map that pretends to be current is worse than none.

## Positioning

PINNeAPPle is a physics computing framework. A physics-informed neural network
is one technology inside it, not the thing it is: the same problem can be solved
by a solver, a surrogate, an operator, a reduced-order model or a hybrid, and the
library's job is to let a reader compare them and know when to trust each one.

That gives one sentence for the README and one for the docs:

- **One line:** a physics computing framework where a PINN is one method among
  several, and every result carries the evidence that it still holds.
- **What it refuses to be:** a PINN training library with solvers attached. When
  a classical solver is cheaper and sufficient, saying so is a correct answer.

The consequence for the API: the front doors are `pp.solve`, `pp.compare`,
`pp.validate` and `pp.design`, which are method-agnostic. `pp.build_model` and
`pp.Trainer` are still public, but they are the door for someone who already
decided to train a neural network.

## Target map

| Namespace | Holds | Today |
|---|---|---|
| `pp.physics` | `PhysicalProblem`, `Parameter`, `Quantity`, presets, PDE families, dimensional checking | Missing. The symbols are top level; the pieces are `pp.env`, `pp.pinn`, `pp.symbolic`, `pp.presets` |
| `pp.data` | `PhysicalSample`, collation, collocation, dataset audit, lineage | Lazy: `pp.data` → `pinneapple_data` |
| `pp.nn` | Architectures, training policy, losses, weight balancers | Missing. `pp.models` (`pinneapple_models`) and `pp.train` (`pinneapple_train`) exist |
| `pp.optim` | Optimizers, schedulers, design optimization, Bayesian optimization | Missing. Spread over `pp.design_opt` and `pinneapple_neural` |
| `pp.linalg` | Linear algebra the physics code needs and NumPy does not name | Does not exist |
| `pp.sparse` | Sparse operators, assembly, matrix-free products | Does not exist |
| `pp.geometry` | SDF/CSG primitives, parametric geometries, CAD import | Missing. `pp.geom` and `pp.geometries` exist; `pp.circle`, `pp.rectangle` are top level |
| `pp.mesh` | Mesh containers, quality, repair, meshing adapters | Missing. `pp.cae` → `pinneapple_data.cae` |
| `pp.solver` | Classical solvers (FEM/FVM/FDM/LBM/SPH) and external adapters | Missing under that name: `pp.solvers` → `pinneapple_solvers` |
| `pp.surrogate` | Operator networks, GNNs, ROMs, hybrid surrogates | Does not exist; spread over `pinneapple_neural` and `pp.benchmark` |
| `pp.viz` | Field rendering, viewers, exports for figures and 3D | Exists, but points at `pinneapple_tools.visualization.studio` (the Studio scene builder), not the figure style |
| `pp.analysis` | Validation, verification, uncertainty, sensitivity, cost | Missing. `pp.uncertainty`, `pp.sensitivity`, `pp.inverse`, `pp.assimilation` exist |
| `pp.decision` | What to run next, never the physical result | Exists: `pp.decision` and `pp.decide` |
| `pp.experiments` | `Experiment`, reproducible runs, artifacts, evidence | Partial: `pp.experiments.reproduce(id)` answers `reproduced` / `divergent` / `refused` for a recorded run (#1480); `Experiment` itself stays top level and `pp.artifact` holds the `.ppx` container |
| `pp.workflows` | DAG execution, executors, run events | Missing. `pinneapple_core.workflows`, reachable as `pp.core` |

## Rules the map enforces

1. **A namespace names a plane, not a package.** `pp.data` is the data plane;
   it is not a promise that every symbol lives in `pinneapple_data`. The seven
   planes are described in [the plane model](seven_planes.md).
2. **Two names for one thing, never three.** When a namespace is added, the
   older top-level symbol stays as a documented alias for at least one minor
   release, and `docs/api_stability.md` records the removal.
3. **Nothing is added to `pp` that is not stable.** A new capability lands in
   its package first; promotion to `pp.<namespace>` happens when a test pins the
   contract.
4. **A method-specific door is named as such.** `pp.build_model`, `pp.Trainer`
   and `pp.list_models` say "neural network" in their docstrings, so nobody
   arrives at the library believing it can only train one.
5. **The map is checked, not trusted.** `tests/test_public_namespace.py` asserts
   which names resolve and which raise, so a rename cannot pass silently.

## Migration

The aliases are additive: `pp.physics`, `pp.mesh`, `pp.viz` and `pp.workflows`
are the first four to create, because they are the ones a user types and cannot
find today. Each is a thin module that re-exports the package it names, with a
test, and none of them changes the underlying package layout.
