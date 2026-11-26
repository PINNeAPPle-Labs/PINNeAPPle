# D1 — reconciling the three problem specifications

**Decision record** for issue #150 (strategy item S1, decision D1 in
[STRATEGY_PHYSICS_AI.md](STRATEGY_PHYSICS_AI.md)). The short version: the three specifications **stay as they are**,
and the conversions between them live in one module, named, with their losses written down and tested.

## The three specifications

| | `pinneapple_problemdesign.schema.ProblemSpec` | `pinneapple_physics.pde_environment.spec.ProblemSpec` | `pinneapple_data.upd_types.ConditionSpec` |
|---|---|---|---|
| what it is | the **elicited** problem: what the user said | the **PDE** problem: what the solver runs | one **sampling** condition of a UPD shard |
| shape | `title`, `goal`, `task_type`, `inputs`, `outputs`, `horizon`, `data`, `physics`, `geometry`, `validation`, `deployment`, `constraints`, `assumptions`, `risks` | `name`, `dim`, `coords`, `fields`, `pde` (`PDETermSpec`), `conditions`, `sample_defaults`, `scales`, `field_ranges`, `references`, `domain_bounds`, `solver_spec`, `meta` | `name`, `type` (initial / boundary / interior / slice), `equation`, `n`, `options` |
| what it is for | the conversation, the plan text, the code generator | the compiler, the solvers, the checks, the reference | how many points, and of which kind, to sample |
| free text? | yes, deliberately | no: selectors and value functions are code | the equation is a string kept "for coordination only" |
| can it run? | no | yes | no |

None is a superset of the others, which is why D1 does not ask for one to replace the others.

## The shared fields, and how they map

This table is also a data structure — `SHARED_FIELDS` in
`pinneapple_physics/pde_environment/reconcile.py` — so the tests read the same list this page prints.

| field | elicited | PDE | sampling |
|---|---|---|---|
| name | `title` | `name` | `name` |
| fields | `outputs` | `fields` | — |
| coordinates | `inputs` | `coords` | — |
| bounds | `geometry.domain` (prose) | `domain_bounds` | — |
| equations | `physics.governing_equations` | `pde.kind` | `equation` |
| boundary conditions | `physics.boundary_conditions` | `conditions` of kind dirichlet/neumann/robin | `type=boundary` |
| initial conditions | `physics.initial_conditions` | `conditions` of kind initial | `type=initial` |
| known parameters | `physics.parameters_known` | `parameters` with role `fixed` | — |
| unknown parameters | `physics.parameters_unknown` | `parameters` with role `unknown` | — |
| units | `physics.units` | `units` | — |
| points | — | `sample_defaults` | `n` |

## The converters

`pinneapple_physics/pde_environment/reconcile.py` — functions, no new class (D1 is explicit about that):

| function | direction | what it is |
|---|---|---|
| `pde_to_design(spec)` | PDE → elicited | for reports, the assistant and the plan text; goes through `PhysicalProblem.to_problem_design()` |
| `design_to_pde(design, *, pde_kind, coords=, fields=, domain_bounds=)` | elicited → PDE | `pde_kind` is **required**: which equation is meant is the decision the conversation does not make |
| `upd_condition_to_pde(cond, *, fields, value=)` | sampling → PDE | returns a `ConditionBridge`: the condition, what the PDE side cannot hold (`carried`), and what could not be done (`lost`) |
| `pde_condition_to_upd(cond, *, carried=, n=)` | PDE → sampling | the shared fields, plus whatever `carried` hands back |

`LOSSES` in the same module states, in one sentence each, what every direction drops and why dropping it is right.
A converter that loses information without saying so is how a problem that was never described reaches a solver.

## Why not one class

D1 says "do not add a new class in this issue", and the reason holds beyond this issue: the three specifications are
read by three different layers at three different times (before the decision, at solve time, after the run). Folding
them into one type would either make the elicited spec carry code it cannot have, or make the PDE spec tolerate
missing or prose fields it must not accept. Conversion functions keep each type honest about what it is for.

## Consequences

- A project can start elicited, be solved as a PDE, and be stored as samples — with the conversion named at each
  boundary, and `spec.meta["elicited"]` keeping the prose that the PDE side cannot hold.
- The equation kind is a decision, not a guess: `design_to_pde` refuses to invent it.
- The sampling ↔ PDE condition bridge is lossy in one specific place (a value function cannot be recovered from an
  equation string). It says so and refuses to invent a value.
- Anything else that wants a fourth representation should extend `SHARED_FIELDS` and `LOSSES` rather than add a
  converter somewhere else.
