# The front door: six functions

Six functions are the whole vocabulary a new user has to learn. Everything else in PINNeAPPle is real and documented,
but it lives in its module — the [module index](../org/reference/index.html) has every public symbol with the module
it belongs to, and each guide says which module a step uses. The point of the short list is not that the library is
small; it is that deciding whether PINNeAPPle fits your problem never requires reading it.

| Function | What it answers | Module |
|---|---|---|
| `pp.doctor()` | what this machine still needs, one line per missing piece with the fix | `pinneapple_core.onboarding` |
| `pp.recommend(problem)` | which preset, method and model for a problem in words, and why | `pinneapple_core.onboarding` |
| `pp.card(preset)` | what a preset models: fields, units, parameters, regime, references | `pinneapple_core.onboarding.catalog` |
| `pp.new(name)` | a project of your own: `case.yaml` and `run.py` you edit | `pinneapple` (the scaffolder) |
| `pp.solve(problem, method)` | one solution of one problem with the method you choose | `pinneapple_physics.solving` |
| `pp.compare(problem, methods)` | the same problem solved several ways, scored against one reference | `pinneapple_physics.solving` |

The same six exist as commands: `pinneapple doctor start`, `pinneapple recommend "…"`, `pinneapple card <preset>`,
`pinneapple new my_study`.

## Everything else, and where it is

The next step is not another function — it is a module that owns the concept:

| You want | Go to |
|---|---|
| to describe your own PDE, with conditions and units | `pp.PhysicalProblem`, `pp.presets` ([guide](../getting_started/first_pinn.md)) |
| to record runs reproducibly and reload them | `pp.Experiment` ([solving a problem](solve_a_problem.md)) |
| to find a method by name instead of by memory | `pp.list_methods()`, `pp.describe_method()` |
| to check whether a prediction can be trusted | `pp.validate`, `pinneapple_llm.PhysicsGuardrail` ([guide](pp_validate.md)) |
| to add a method, a solver or a preset | [choosing a method](choosing_a_method.md), `pinneapple_physics.solving.register_method` |
| to run it as a campaign, a twin or a service | `pinneapple_simulation`, `pinneapple_systems`, the apps in `apps/` |

`tests/test_front_door.py` holds the rule the README obeys: the front-door pages may present at most these six
functions, and any other `pp.<name>` a page names has to be a class or a module, because a seventh function is how a
front door stops being one (issue #45).
