# Add a preset

A worth-while goal: someone outside this repository adds a problem to PINNeAPPle — one that other people can name,
solve, compare and evaluate — without asking anybody a question. This page is that path, and it is the one the other
five "how to add" guides follow: a commented template you copy, the change that makes it real, and the test that has
to come with it.

The template is [`templates/36_add_a_preset.py`](../../templates/36_add_a_preset.py). Run it first:

```bash
python templates/36_add_a_preset.py
```

It registers a plate held at 0 °C on one wall and 100 °C on the other, solves it, and prints what it checked. If that
runs, the rest of this page is only about doing the same for your problem.

## What a preset is

A frozen description of a problem: coordinates, fields, the PDE term with its parameters, the conditions, the
bounding box, the sampling defaults and the solver that produces a reference for it. Everything else in the library
is written against that description, which is why adding one makes the problem usable in `pp.solve`, `pp.compare`,
`pp.Experiment`, the Studio, the Flow nodes and the apps without touching any of them.

## The four steps

1. **Copy the template** to where your problem belongs. Academic and textbook problems go in
   `pinneapple_physics/pde_environment/presets/academics.py`, industrial ones in `industry.py`, engineering ones in
   `engineering.py`; a problem that is yours alone can stay in a module of your own next to the template's import.
2. **Change the physics**: `.domain(...)` for the bounding box (name every coordinate — the dimension is the number of
   coordinates, and the solvers and checks switch on it), `.fields(...)` for the unknowns, `.pde(kind, **params)` for
   the equation. The `kind` string is what the solvers, the residual checks and the verification rules switch on; a
   `kind` nobody knows solves as a generic residual and gets no physics check, so reuse one of the existing kinds
   unless you are adding a family (`pinneapple_physics/pde_environment/capabilities.py`).
3. **Declare every condition and every wall.** This is where new presets usually go wrong: a wall you do not name is
   a **zero-Dirichlet** wall, not an insulated one. The template's plate has 0 °C and 100 °C on the x walls and
   therefore 0 °C on the y walls; making it insulated top and bottom means naming those walls with
   `kind="neumann", value=0.0`, and that is a different reference.
4. **Give it a reference.** `.solver(name="fdm", nx=..., ny=...)` is the agnostic finite-difference solver: it reads
   the PDE, the bounds and the conditions and only needs a grid. A preset with a published closed form is better
   served by an analytic matcher (`pinneapple_physics/solving.py`, the `burgers_sine_cole_hopf` entry) — then
   `pp.solve(problem, "analytic")` works for anyone and the preset can be validated against an exact answer.

## The test that has to come with it

A preset without a test is a file. Put yours in `tests/pinneapple_physics/test_preset_core_contracts.py` (the
contract suite every preset has to satisfy) or in a new `tests/test_preset_<yours>.py`, and cover three things:

```python
def test_my_preset_registers_and_solves():
    import pinneapple as pp
    problem = pp.PhysicalProblem.from_preset("my_preset")      # 1. the registry answers
    ref = pp.solve(problem, "reference").predict(points)        # 2. the reference runs
    assert cold < hot                                           # 3. the physics holds (a number you can defend)
    assert {"x", "y"} == set(problem.coords)                     # and the description is the one you meant
```

`tests/pinneapple_physics/test_preset_core_contracts.py` already walks every registered preset and checks the
description (units, bounds, conditions, the reference), so a preset added to the registry is audited there without
writing anything; your own test is for the physics that only you know how to check.

Then:

```bash
python -m pytest tests/pinneapple_physics/test_preset_core_contracts.py tests/test_preset_mine.py -q
python scripts/reference/build.py        # the module reference lists the new symbol
python scripts/changelog.py check        # a changelog.d fragment for the user-visible change
```

## Checklist

- [ ] the name is lower case with underscores, and it does not collide with `pp.list_presets()`
- [ ] every field and every parameter has a unit (a number without a unit cannot be checked)
- [ ] every wall is named (the ones you leave out become zero-Dirichlet)
- [ ] the reference solver is declared, and a test compares the answer against something independent
- [ ] `pp.card("my_preset")` prints what you expect — the card is what the front door shows
- [ ] a test asserts the registration, the solution and the physics
- [ ] the module reference and a `changelog.d` fragment are regenerated

## If something does not work

| Symptom | Usual cause |
|---|---|
| `no closed-form solution is registered for problem ...` | you asked for `"analytic"` and the preset has no closed-form matcher: use `"reference"` for the declared solver |
| `reference solver returned a grid over ('x', 'y'), but the problem has coordinates ('x',)` | the problem is 1-D and the solver built a 2-D grid: declare both coordinates, or use a 1-D-capable solver |
| the answer is off by a plausible factor | a wall you did not name became zero-Dirichlet |
| `pp.card` says `available=False` | the preset is not registered: `ProblemBuilder(...).register()` is the line that does it |
