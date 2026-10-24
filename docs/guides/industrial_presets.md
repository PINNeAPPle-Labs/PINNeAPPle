# Industrial presets

A preset is an engineering problem already set up, solved and validated with PINNeAPPle. You open it, replace the
defaults with your own values and run it. Each preset carries:

- **a data sheet**: the sector, the question it answers, every input in SI units with a realistic default and the
  source of that default, the outputs and the validity domain;
- **the pipeline**: geometry, then the model, then the solver (and optionally the surrogate), then post-processing,
  then a decision;
- **checks on every run**: a trust report with conservation, convergence, correlation ranges and the
  applicability domain;
- **references**: comparisons against citable values with the error measured (standards, published benchmarks,
  analytic solutions, independent solvers). A preset with no independent reference says so on its data sheet;
- **optional capabilities**, where the physics allows them: a 3-D scene of the result, a surrogate, an inversion,
  an optimisation checked with the solver, and design variants run side by side.

```python
import pinneapple as pp

pp.presets.names()                                     # the catalogue
p = pp.presets.load("water_distribution_network")
p.card()                                               # data sheet (JSON-ready; the Hub shows it)
r = p.run(demand_factor=1.3, hw_c=100)                 # your values, SI units
print(r)                                               # outputs, decision, trust report
p.validate()                                           # reference cases with the measured error
p.optimize()                                           # e.g. least-cost diameters, re-solved and checked
p.invert()                                             # e.g. Hazen-Williams C from logged pressures
pp.presets.load("water_hammer_valve_closure").run(fidelity="surrogate", t_close=6.0)
```

Command line:

```bash
python -m pinneapple_presets list
python -m pinneapple_presets card power_pcb_junction
python -m pinneapple_presets run power_pcb_junction vias=16 t_ambient=45
python -m pinneapple_presets validate steel_footbridge_pedestrian
python -m pinneapple_presets report all            # rewrites examples/industrial_presets/ and the Hub index
```

In the Flow Studio, the `preset.run` node takes the preset name, a dictionary of overrides and the fidelity
(`solver` or `surrogate`). The `preset.validate` node runs the references.

## Catalogue

The table is in [examples/industrial_presets/README.md](../../examples/industrial_presets/README.md). It is
regenerated from the stored results, so every number in it was measured. Each preset folder holds its README:
the data sheet, the measured outputs, the validation table, the surrogate hold-out error, the inversion, the
optimisation, the variants and the figures.

## Surrogates

`SurrogateSpec(inputs, outputs, n_train, n_test, bounds, log, log_outputs)` declares the surrogate.
`python -m pinneapple_presets surrogate <name>` then builds it in five steps:

1. It draws a Latin hypercube over the inputs.
2. It runs the real solver at every point.
3. It fits one anisotropic Gaussian process per output.
4. It measures the error on an independent hold-out set.
5. It stores everything as a few-kB JSON file in `pinneapple_presets/data/surrogates/`.

A `run(fidelity="surrogate")` call returns each output with its standard deviation. When the inputs fall outside
the training box, or when a parameter differs from the value the surrogate was trained at, the call raises a
warning, the trust report marks the check "contradicts", and nothing is extrapolated silently.
`pinneapple_presets.surrogate.serve_spec(name)` describes the surrogate for a model server: its inputs with bounds
and units, its outputs, its error and the call that runs it.

## Adding a preset

1. Subclass `pinneapple_presets.Preset`.
2. Set the data-sheet attributes: `params` as `Param(name, unit, default, source, lo, hi)`, and `outputs`.
3. Implement `solve(p) -> dict`, `checks(p, out) -> [Check]`, `decide(p, out)` and `references()`. Add the optional
   `scene`, `figures`, `invert`, `optimize` and `variants` methods where the physics allows them.
4. Register the class in `pinneapple_presets/registry.py`.
5. Run `python -m pinneapple_presets report <name>`.
6. Add a test.

Every default needs its source. Every reference needs its citation, plus the tolerance it must meet.
