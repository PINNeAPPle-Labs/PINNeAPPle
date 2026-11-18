# Large Physics Model pipelines: one model per design space

An LPM is *one trained model for a whole design space*. `pinneapple_neural.lpm` holds the model; this pipeline turns a
parametric family of bodies into the dataset it is trained on, fits it, and reports what the fit measured.
`pinneapple_neural.lpm_pipeline` is the pipeline, `pinneapple_neural.lpm_families` the five families that ship with it.

```python
from pinneapple_neural.lpm_pipeline import run, families

summary = run(families()["pp_fin"], "out/pp_fin", n_geometries=80, conditions_per_geometry=3, epochs=400)
print(summary["held_out_relative_l2"], summary["ood_probes_flagged"])
```

or from the command line:

```bash
python -m pinneapple_neural.lpm_pipeline pp_nozzle --geometries 80 --conditions 3 --epochs 400
python -m pinneapple_neural.lpm_pipeline pp_nozzle --dataset-only --geometries 2000   # data, no fit
```

## The five stages

| Stage | What it does | Where it is written |
|---|---|---|
| draw | samples a geometry, then `conditions` operating points for that same body | in memory (`DesignSpec`) |
| build | surface point cloud + normals (the geometry the model sees), normalised conditioning vector, target field | in memory (`lpm.Design`) |
| write | one `.npz` per design, plus a manifest with the reference, the bounds and every SHA-256 | `dataset/` |
| fit | split **by geometry**, batched design-space training, held-out measurement | `model/model.pt` |
| card | what this run measured and what it did not | `model/card.json`, `summary.json` |

## The five families

| Name | Vertical | Field | Reference |
|---|---|---|---|
| `pp_fin` | electronics cooling: extruded-fin heat sink | fin and base temperature (°C) | 1-D fin theory on the `plate_fin_heatsink` resistance network (Incropera; Elenbaas; Teertstra) |
| `pp_airfoil` | UAV, wind and compressor sections | surface Cp | Joukowski conformal map with the Kutta condition — exact potential flow (Milne-Thomson; Anderson) |
| `pp_nozzle` | propulsion: converging-diverging nozzle | wall Mach and p/p_chamber | quasi-1-D isentropic flow (Sutton & Biblarz; NACA 1135) |
| `pp_exchanger` | process and HVAC: shell-and-tube exchanger | dimensionless shell and tube temperatures | Kern shell side, Gnielinski tube side, ε-NTU from the 1-2 energy equations |
| `pp_bracket` | structural: L-bracket under a tip load | displacement (m) and von Mises (Pa) | immersed finite-element elasticity (`pinneapple_simulation.immersed`), solver-backed |

Each family declares its `reference` — the model behind the target, its citation and the assumptions it does **not**
include — and every card repeats it. A held-out number from this pipeline is therefore a statement about the fit to
that reference. It is not a validation of the model for a real part, and the card says so in `trust.not_measured`.

Two of the five predict dimensional fields (the heat sink in °C, the bracket in m and Pa); the other three predict
dimensionless ones (Cp, Mach and p/p0, θ) because that is how their physics scales. `pp_exchanger` reconstructs the
physical temperatures as `T_cold_in + θ (T_hot_in − T_cold_in)`, and `pp_nozzle` as `(p/p0) × p_chamber`.

## Three rules the pipeline enforces

**The split is by geometry, never by design.** A design space runs several operating points on the same body. Splitting
by design would let the model see a body during training and be scored on that same body at another operating point,
which reads as generalisation without being it. `_split_by_geometry` holds out whole bodies.

**The point cloud is the geometry, and everything else is conditioned.** A taller fin, a thicker section or a wider
nozzle changes the points, so the geometry code sees it. Parameters the outline cannot carry — a tube count, a
shell-to-length ratio — go into `geometry_features` and join the conditioning vector next to the operating point. A
family that leaves out a parameter its field depends on gives the model two bodies it cannot tell apart, and the fit
pays for it: `pp_exchanger` declares three such features for exactly that reason.

**A design the reference cannot represent is refused, not written.** `FieldFamily.validate` returns the reason;
`draw` re-draws the operating point and `build` raises rather than writing a number the reference never justified
(a forced-convection draw at 1e-6 m/s is numerically finite and physically meaningless).

Field scaling (`field_scale`) brings every field to order one before the loss sees it, so a family predicting a
displacement in metres and a stress in pascal does not train on the stress alone. Errors are always reported in the
family's physical units, with the scale undone.

## What a run measures

`summary.json` and `model/card.json` carry, for that run only:

- `held_out.relative_l2` and `relative_l2_p90` on bodies held out by geometry;
- `held_out.constant_field_baseline_relative_l2` — predicting the training-average field, the bar any surrogate must
  clear;
- `ensemble_spread` — the spread between bagged heads, **not** a confidence level;
- `ood` — the library's geometry-code rule on the family's probes outside the family and on the held-out designs,
  reported as measured: a probe the model does not flag is not thereby inside its validity domain;
- `trust.decision` — always `not_verified_until_independent_review`, with the list of what was not measured.

A family also declares `tolerance`, the held-out relative L2 its own gate asks for. The gate is reported with the
measured value, never used to hide it.

## Adding a family

Subclass `FieldFamily` and implement two methods:

```python
from pinneapple_neural.lpm_pipeline import DesignSpec, FieldFamily, Reference

class MyFamily(FieldFamily):
    name, title, domain = "pp_mine", "...", "..."
    field_names, field_units = ("field_a",), ("Pa",)
    geometry_params = {"length_m": (0.1, 1.0)}          # how the body is drawn
    operating_params = {"load_n": (1e3, 1e4)}           # how it is run
    characteristic_length_m = 1.0                        # the geometry is divided by this
    reference = Reference("...", "...citation...", ("assumption it does not include",))

    def points_and_normals(self, spec: DesignSpec):      # (P, dim) points and outward normals, metres
        ...
    def field(self, spec: DesignSpec):                   # (P, len(field_names)), from the reference
        ...
```

Then add it to `families()`. If the reference has a validity range, override `validate`; if the body has geometry the
outline cannot show, declare `geometry_features` and `geometry_feature_values`. The tests in
`tests/test_lpm_pipeline.py` are the template: every family is checked against an identity that does not come from the
family itself (lift against ρUΓ, the area-Mach round trip, ε-NTU against its closed form, the fin equation, linearity
and mesh convergence of the FEM).

## Limits

- The five references are screening models: potential flow has no boundary layer, quasi-1-D flow has no shocks or
  separation, fin theory is 1-D, the exchanger holds UA constant with fixed water properties, and the bracket is
  linear 2-D elasticity. Each family lists its own assumptions.
- The models are only as good as their data. A design space of a few dozen bodies is a smoke test; the held-out error
  falls with the number of bodies, which is what generating the full datasets is for.
- Nothing here validates a model against a solver run or a test. That is an independent review, and the card's
  `trust.not_measured` says so.
