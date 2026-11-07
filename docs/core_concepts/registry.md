# One registry for every extensible object type

`pp.register` puts a solver, a model, a preset, an operator, a dataset, a benchmark, a geometry, a material or an
optimizer into the library under one protocol, instead of a registry per type with its own shape:

```python
import pinneapple as pp

pp.register(MySolver, kind="solver", name="my_solver")     # or: @pp.register(kind="solver")
pp.registered("solver")                                    # every name, the library's own included
pp.resolve("solver", "my_solver")                          # the object itself
pp.unregister("solver", "my_solver")
```

The same four calls work for all nine kinds. Each kind declares once what it accepts, in `pp.registry.KINDS`, and
registering something of the wrong type fails with `WrongType`, naming the kind and what it takes:

| Kind | Accepts | Where it also lands |
|---|---|---|
| `physics` | a preset factory: `name -> ProblemSpec` | `pinneapple_physics` presets, so `pp.get_preset(name)` finds it |
| `model` | a `torch.nn.Module` subclass | `ModelRegistry`, so `pp.list_models()` lists it |
| `solver` | `fn(problem, **options) -> Solution` | `pp.solve`'s methods, so `pp.solve(problem, "my_solver")` runs it |
| `operator` | `fn(values, ...) -> array` | — (this module is its home) |
| `dataset` | a `DatasetInfo`, with `loader=fn` | `DatasetRegistry`, keyed by its `id` |
| `benchmark` | a callable that runs the benchmark | — |
| `geometry` | a `Geometry` instance or subclass | — |
| `material` | a `Material` | `pinneapple_design`'s `MATERIALS` |
| `optimizer` | a `PhysicsOptimizer` subclass | — |

Where the fourth column says "this module", the kind had no registry of its own before: `pp.resolve` is how the rest of
the library — and a plugin — finds what was registered there. What the kind reaches is reported back:
`pp.registry.KINDS["dataset"].describe("my_data")["bridge"]` says `DatasetRegistry`, or
`unavailable: ValueError: a dataset needs loader=fn…` when the bridge could not be made, which is a report and not a
failure: the object is registered either way.

## Inferred kinds, and when the registry refuses to guess

`pp.register(obj)` without `kind` works only when the object's own type settles it — a `torch.nn.Module` subclass is a
model, a `Material` a material, a `Geometry` a geometry, a `DatasetInfo` a dataset, a `PhysicsOptimizer` subclass an
optimizer. A plain callable could be a solver, an operator or a benchmark, so it raises `AmbiguousKind` and asks for
`kind=` instead of picking one and being wrong later, inside a solve.

## The protocol

`Registry` is the same class for every kind: `register(name, obj, replace=False, **meta)`, `get(name)`, `list()`,
`unregister(name)`, `describe(name)`, `in`, `len`. The errors are one vocabulary: `UnknownKind`, `UnknownName`,
`AlreadyRegistered`, `WrongType`, `AmbiguousKind` — all subclasses of `RegistryError`, so a caller that only wants to
know "the registry said no" catches one exception.

```python
from pinneapple_core.registry import AlreadyRegistered, RegistryError, WrongType

try:
    pp.register(42, kind="operator", name="not_an_operator")
except WrongType as exc:
    print(exc)        # operator expects a callable (values, ...) -> array, got int
```

`register` returns the object, so it also works as a decorator. `unregister` removes the name from the unified registry
**and** from the home it was bridged to, so the library does not keep a coupon it was told to forget.

## What it is not

- not a plugin loader: discovering code from an installed package is the entry-point work of the extensibility series
  (#1472–#1481); this is the registry those entry points register into;
- not a rewrite of the existing homes: `pp.list_methods`, `pp.list_presets`, `pp.list_models` and `MATERIALS` keep
  working exactly as before, and registering through `pp.register` simply reaches them.
