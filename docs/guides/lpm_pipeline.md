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

## Fitting data this repository did not produce

The five families above compute their own targets, so a good fit proves the *model* can learn a physics this
repository implements — not that it can learn physics as it is actually solved. `pinneapple_neural.lpm_real_data`
closes that gap with a **published campaign**: `pp_airfoil_rans` fits the `airfoil_rans_cst_re4e6_fields` dataset, 436
steady 2-D RANS runs (OpenFOAM `simpleFoam`, k-omega SST, y+ < 1) over 100 Latin-hypercube CST shapes at four
incidences plus four NACA sections at nine, on a shared 228 x 88 O-grid, cc-by-4.0.

Nothing is drawn here: `specs()` returns the campaign's runs exactly as recorded, and `published_split()` hands back
the 15 % of shapes its authors held out before publishing, so a held-out number means what it means in the campaign.
The geometry is the **wall ring** of the O-grid, the field is surface Cp from the solved kinematic pressure, and the
condition is the incidence.

```python
from pinneapple_neural.lpm_real_data import data_gates
print(data_gates())                              # what passed the gates and what was refused
python -m pinneapple_neural.lpm_pipeline pp_airfoil_rans --dataset-only --out out/rans
```

Three gates stand between the data and the fit, and a run that fails one is not written:

| Gate | What it catches |
|---|---|
| the run's own status | the campaign flags 32 runs unsteady; a final-iteration snapshot of an oscillation is not a steady answer |
| the wall pressure must integrate to the campaign's published Cl | the ring, the normals or the sign was read wrong — this is the only check of the extraction itself |
| surface Cp inside what a RANS section can produce | a mis-indexed grid, before it is trained on |

The campaign's own verification — grid study on NACA 2412 (Cl 0.6614/0.6625/0.6624 at 6840/20064/45144 cells, Cd GCI
1.9 %), XFOIL 6.99 (Cl −2.3 %, Cd +4.5 %), a 0.110/deg lift slope against thin-airfoil theory, and NASA TMR NACA 0012
SST (Cd 0.00812 vs 0.0081) — is quoted in the reference and repeated in every card. It is the campaign's, not this
pipeline's. What this family still does not do: it does not validate the RANS itself, and it does not turn a 2-D
section into a 3-D wing.

### What it measured

372 runs pass the gates (100 shapes, 314 train / 58 test by the campaign's own split, no refusal), and the wall
pressure integrates to the campaign's published Cl to a median of 0.0033 in coefficient, worst 0.0159 — the residual
is the viscous part of the reported coefficient, which a pressure integral does not contain.

| Run | Held-out relative L2 | p90 | Cl from the predicted Cp |
|---|---|---|---|
| 96 wide, 150 epochs, 32 designs/batch | 0.216 | 0.321 | 9.3 % median, r = 0.979 |
| 128 wide, 700 epochs, 48 designs/batch, 48-code, 8 frequencies, 3 attention layers | **0.057** | 0.125 | **2.8 % median, p90 6.8 %, r = 0.998** |

The constant-field baseline on the same 58 designs is 0.959, so the model explains what a mean field cannot. The
second row is `train(dataset, out, epochs=700, batch_designs=48, width=128, code=48, n_freq=8, n_heads=6,
attn_layers=3, lr=1.5e-3)`; it takes 19 minutes on one MPS device. The first row is the same data at the defaults,
and it is a useful comparison: on this space the limit was training budget, not the data — the error fell by 3.8x
without a single extra solver run. Pointwise Cp error is now flat along the chord (0.05 of the local rms from the
leading edge to 70 % chord); what remains sits near the trailing edge, where Cp is small and the relative error is
large while the absolute error is not.

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
