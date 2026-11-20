# Add something to PINNeAPPle

Six extension points, one page each, and the same shape for all of them: a commented template you copy, the line that
makes the change real, and the minimum test that has to come with it. The rule behind the whole page is simple: if a
new contributor has to ask a question, the guide is missing an answer, not the contributor a step.

| You want to add | Start from | The test that comes with it |
|---|---|---|
| **a preset** (a problem others can name and solve) | [`templates/36_add_a_preset.py`](../../templates/36_add_a_preset.py) — guide: [adding a preset](adding_a_preset.md) | `tests/pinneapple_physics/test_preset_core_contracts.py` audits every registered preset; add your own physics check |
| **a solver or method** | `pinneapple_physics/solving.py` (`register_method`), and the custom method in `tests/test_solve_compare_experiment.py` | one test that measures the method against a reference on a problem with a known answer |
| **a model architecture** | `pinneapple_neural/architectures/registry.py` (`ModelRegistry.register`), listed by `pp.list_architectures()` | `tests/pinneapple_neural/test_pinn_architectures_core_contracts.py` — every registered architecture is built, trained for a few steps and checked |
| **a benchmark case** | `templates/28_arena_benchmark.py`, then the arena's case format | `tests/test_validation_arena.py` replays the stored cases; a new case needs its expected verdict |
| **a data adapter (connector)** | `pinneapple_data/adapters/` — a reader that turns a source into a `PhysicalSample` or a table | `tests/pinneapple_data/test_contracts.py` — the contract a reader has to produce, and its round trip |
| **an app** | `apps/` — an app is a folder with a package of the same name inside it, holding `api.py` (the FastAPI routes) and `static/` (the page); `apps/heatsink_sizer` is a complete small one | `tests/test_app_helpers.py` plus one HTTP test of your routes |

## What every one of them has to do

1. **Say what it is** in one place the library reads: a preset registers itself, a method registers itself, an
   architecture registers itself, an adapter returns a contract. If the only way to find your addition is the source
   file, nothing else in the library can use it — not the Studio, not the Flow nodes, not `pp.recommend`.
2. **Carry its units and its provenance.** A number without a unit cannot be checked, and a result without
   `solver`/`version`/`seed`/inputs cannot be reproduced (`pinneapple_data.provenance`, the rule the provenance gate
   applies to every artifact in the tree).
3. **Bring a test that fails without it.** Not "it imports": a number against a reference, a case that would have
   caught the bug you just fixed.
4. **Regenerate what is generated**: `python scripts/reference/build.py` for the module index, a `changelog.d`
   fragment for the user-visible change, and `python -m pinneapple_studio coverage` if a new public symbol appears
   (the Studio asks every public symbol for a tool or a reason).

## The checklist, in the order it bites

- [ ] the new name is in the registry the library reads, not only in your module
- [ ] units on every field, parameter and output
- [ ] provenance recorded where the addition produces numbers
- [ ] one test that fails without the addition
- [ ] `make test`, `make e2e` and `make gates` pass
- [ ] the module reference and a changelog fragment are committed with it

`tests/test_adding_guides.py` holds this page to its own promise: every template, test and module named here has to
exist, and the preset template has to run, so a contributor following it cannot land somewhere that no longer exists.
