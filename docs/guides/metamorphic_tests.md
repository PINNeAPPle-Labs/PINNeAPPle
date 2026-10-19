# Metamorphic tests: checking a surrogate without ground truth

Most physics gives relations between inputs and outputs that hold whatever the true answer is: mirror the geometry
and the flow mirrors; keep the Reynolds number and the dimensionless answer stays; add heat and the peak temperature
does not fall; double the source of a linear problem and the answer doubles. These are metamorphic relations, and
they test a surrogate where no reference solution exists. `pinneapple_analysis.validation.metamorphic` runs them on
any callable model (NumPy arrays or dicts of columns in, arrays or dicts out).

```python
from pinneapple_analysis.validation.metamorphic import box_sampler, check_relations, relations_from_spec

def peak(x):                                       # any model: here the exact peak temperature of a heated slab
    return {"T": x["T0"] + x["q"] * x["L"] ** 2 / (8 * x["k"])}

relations = relations_from_spec([
    {"name": "more-heat-hotter", "kind": "monotonic", "input": "q", "output": "T", "direction": "increasing"},
    {"name": "better-conductor-cooler", "kind": "monotonic", "input": "k", "output": "T", "direction": "decreasing"},
    {"name": "same-q-L2", "kind": "invariant", "inputs": {"q": "4 * q", "L": "L / 2"}, "output": "T", "tol": 1e-9},
])
box = box_sampler({"q": (1e3, 1e5), "L": (0.01, 0.1), "k": (1, 400), "T0": (250, 350)})
report = check_relations(peak, relations, box, n=64, search="hypothesis")
print(report.summary())
assert report.passed
```

## Relations

| Relation | Class / spec `kind` | Holds when |
|---|---|---|
| Geometric symmetry | `Symmetry` / `symmetric` | `f(T x) = S(x, f(x))`; `reflect_field`, `rotate_field` for gridded inputs |
| Similarity scaling | `Similarity` | the dimensionless output is unchanged when the dimensionless groups are (same Re at another size, Darcy with `a` and `f` scaled together) |
| Invariance | `Invariance` / `invariant` | an input change that must not matter does not move the output |
| Monotonicity | `Monotonicity` / `monotonic` | more of an input gives no less (no more) of an output; the violation is the wrong-way move over the spread of the output, default tolerance 1e-3 |
| Superposition | `Superposition` / `superposition` | `f(a + b) = f(a) + f(b)` and `f(2a) = 2 f(a)` for linear problems (`affine=True` subtracts `f(0)`) |
| Resolution consistency | `ResolutionConsistency` | an integral of an operator's output is the same with the input at another resolution |

Violations are relative: to the RMS of the output over the samples (default), to each sample's own output
(`per_sample`, for positive quantities) or to a fixed `scale` such as the model's accepted error. The worst sample is
returned as the counterexample. With `search="hypothesis"` (and the `hypothesis` package installed) a property-based
search for a violating sample runs as well; without the package the report says it used random search only.

## Zoo models

`pinneapple_hub.zoo.relations.check_zoo_relations(model)` runs the relations each zoo model's physics implies:

| Model | Relation | Worst violation (128 samples) | Tolerance |
|---|---|---|---|
| `airfoil_cst_rans_mlp` | mirror camber and angle of attack: lift flips | 0.83 x accepted Cl error (0.01) | 2 |
| | moment flips | 0.91 x accepted Cm error (0.002) | 2 |
| | drag unchanged | 1.4 % | 4 % |
| | lift grows with angle of attack (-2 to 8 deg) | 0 | 1e-3 |
| `heatsink_forced_mlp` | resistance falls with air speed; pressure drop grows with it; resistance falls with conductivity | 0 | 1e-3 |
| | resistance independent of power (air speed 2-6 m/s, per sample) | 1.1 % | 2 % |

The mirror tolerance is twice the card's accepted mean error because both sides of the relation are predictions.
The power invariance is **not** tested below 2 m/s: there the physics engine that generated the training data is not
power-linear either (its resistance rises 6.5 % when the power doubles at 0.6 m/s), and the model follows it.

## In `pinneapple check`

Declare relations on a `[[model]]` with a `domain` (rule [PC303](physics_checks_in_ci.md#pc303)):

```toml
[[model]]
name = "peak-temperature"
path = "surrogates/peak_temperature.py:predict"
inputs = ["q", "L", "k", "T0"]
outputs = ["T"]
domain = { q = [1e3, 1e5], L = [0.01, 0.1], k = [1, 400], T0 = [250, 350] }
relations = [
  { name = "better-conductor-cooler", kind = "monotonic", input = "k", output = "T", direction = "decreasing" },
]
```

## How it was checked

`tests/test_metamorphic.py`, with the finite-volume Darcy solver (`operator_benchmarks.darcy_2d`) as the reference:

- the solver obeys superposition in the source to 4.7e-16, reflection and 90-degree rotation of `(a, f)` to round-off,
  and `u(c a, c f) = u(a, f)`; the mean pressure on a 32² grid and its 16² cell average agree to 1.6 %;
- an FNO trained on the same map (`FNO2d`, test error 10.8 %) violates superposition by 0.68 of the output scale;
- a model that ignores the source fails similarity scaling, a tilted Darcy solution fails reflection, an airfoil
  model with a 0.03 lift offset fails the camber mirror (about 6 x the accepted error), and a slab model with a unit
  slip in `k` fails monotonicity; each reports its counterexample;
- the zoo models pass the relations above.

## Limits

- A relation only tests what it states. A model can satisfy every relation and still be wrong by a constant factor
  that the relations do not constrain; use it alongside residual and reference checks.
- Random and property-based search find violations, they do not prove their absence.
- Tolerances must come from the physics and the model's stated accuracy. A relation that does not hold for the real
  physics in part of the box (the heat sink's power invariance at low air speed) is restricted to where it holds,
  not loosened.
