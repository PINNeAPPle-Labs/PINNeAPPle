# API stability: the core and everything else

This page sets out what PINNeAPPle promises on the way to 1.0 (issue #1135). The lists live in code, in
`pinneapple_stability`, and this page explains them.

## Two classes

| Class | What is promised | Where |
|---|---|---|
| **core** | Semantic versioning. A breaking change needs a deprecation period: the old name keeps working, with a warning that names the replacement, for at least one minor release. It also needs a changelog entry. | `pinneapple_stability.CORE` (module prefixes) and `CORE_PP` (`pp.*` names) |
| **experimental** | No compatibility promise between versions. The code, tests and development are the same as before. | Every other public module |

The **labs** packages are experimental packages that are far from the core: they are little used outside their own
tests, and many of their tools are untested. They also say so in a warning, the first time they are imported in a
process:

```text
ExperimentalWarning: pinneapple_condense is experimental: it works and is tested where its catalog says so, but its
API may change between versions without a deprecation period ...
```

To silence it, use either of these:
- `PINNEAPPLE_QUIET_EXPERIMENTAL=1`
- `warnings.filterwarnings("ignore", category=pinneapple_stability.ExperimentalWarning)`

```python
from pinneapple_stability import stability
stability("pinneapple_physics.pinn_solver.compiler")   # 'core'
stability("pinneapple", "science")                      # 'experimental'  (pp.science)
stability("pinneapple_arena.pointwise")                 # 'experimental (labs)'
```

The Studio catalog shows the class of every tool (`stability`) and counts them.

## Nothing is removed

Being experimental is not deprecation:
- No module, feature or extra's dependencies is deleted because it is experimental.
- Removal only ever applies to the *old name* of a core function that was renamed, after its deprecation period.

## Promotion: experimental to core

A module joins the core in any minor release once all three of these hold:
1. **Covered:** none of its public tools is "discovered" in the Studio catalog; each one is tested or verified.
2. **Used:** something outside its own tests uses it, such as an example, an app, the engineering app or a
   documented task.
3. **Frozen:** its names are added to the API snapshot (#1128).

The promotions of each release are listed in its changelog.

## How the core was chosen

The core was measured on `main` after 0.8.0 from three sources:
- imports in `examples/`, `apps/`, `templates/` and `notebooks/`;
- `pp.*` uses in the docs and examples (`solve` 141, `zoo` 126, `validate` 104, `science` 73, `trust_card` 50, ...);
- the tools the engineering app calls (`generate_mesh`, `step_study`).

| Area | Core modules |
|---|---|
| Facade | `pinneapple`: `pp.solve`, `zoo`, `validate`, `trust_card`, `datasets`, `benchmark`, `viz`, `compare`, `presets`, `geometries`, `design`, `doctor`/`recommend`/`card`/`new`, `closed_form`, `uncertainty`, `build_model`, `get_preset`, `list_*`, `to_jsonable` |
| Data | `pinneapple_core.jsonable`, `pinneapple_core.onboarding`, `pinneapple_data.contracts`, `.physical_sample`, `.cae` |
| Physics | `pinneapple_physics.pinn_solver`, `.pde_environment`, `.closed_form`, `.solving` |
| Models | `pinneapple_neural.architectures.pinns`, `.neural_operators`, `.rom`, `pinneapple_neural.trainer` |
| Geometry | `pinneapple_design.geometry.core` (`MeshData`), `.io`, `.brep`, `.mesh` |
| Trust | `pinneapple_analysis.validation`, `pinneapple_analysis.trust` |
| Cases | `pinneapple_presets` |
| Views | `pinneapple_tools.visualization.studio.fieldview` |

The labs packages are:
- `arena`, `worldmodel`, `quantum`, `app`, `decision`, `check`;
- `condense`, `reason`, `orchestration`, `query`, `pdb`, `autolab`, `systems.sports`.

`pp.science` is used a lot, but most of its contents are untested. For now it is experimental, and it is the first
candidate for promotion.

**Status:** this is a proposal under review in #1127. Change the lists in `pinneapple_stability` and this page
together.
