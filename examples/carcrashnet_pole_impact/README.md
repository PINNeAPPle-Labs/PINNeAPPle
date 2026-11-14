# Pole-impact crash benchmark (CarCrashNet protocol)

A DP1000 bumper beam held by two DP600 crash boxes, driven into a rigid pole: seven design variables to the five
crashworthiness quantities, and the undeformed mesh to the displacement sequence of every node. The full description,
the mapping to the published benchmark and the limits of the recreation are in
[`docs/guides/crash_and_surface_benchmarks.md`](../../docs/guides/crash_and_surface_benchmarks.md).

```bash
# a couple of minutes: 6 designs, a short event, 2 epochs
python examples/carcrashnet_pole_impact/run.py --smoke

# the campaign this folder ships results for: 120 designs, 1 ms of the event, 11 frames
python examples/carcrashnet_pole_impact/run.py --designs 120 --parallel 7

# the parametric protocol: the seven design variables reach the field surrogates as per-node channels, so a design
# held out of training can be asked about, and the peak displacement of every held-out design is scored at the end
python examples/carcrashnet_pole_impact/run.py --designs 120 --parallel 7 --conditioning design --models transolver
```

Outputs in `_out/` (not committed — regenerate them):

| file | what it is |
|---|---|
| `pole_impact_fields.npz` | the field dataset: points, part labels, the design vector, times and the displacement of every node |
| `pole_impact_quantities.csv` | the tabular dataset: design vector + quantities, one row per design |
| `tabular_leaderboard.md` / `.json` | Ridge … CatBoost on the same split, plus the permutation importance of the variables |
| `field_leaderboard.md` / `.json` | the field models ranked with per-design relative L2 and bootstrap intervals |
| `kpi_peak_displacement.json` | the quantity-level KPI: reference and predicted peak displacement of every held-out design |
| `figures/peak_displacement_parity.png` | reference against predicted peak, one point per held-out design |
| `figures/displacement_surface.png` | the deformed beam as a surface, reference and surrogate on the same colour scale |
| `campaign.json` | the manifest: generator, solver note, design box, mesh, end time — and the designs used |
| `failures.json` | every design whose run raised (nothing is dropped silently) |

The labels come from `pinneapple_simulation.numerical_solvers.crash_fem` (explicit hexahedral dynamics, J2 plasticity,
penalty contact), not from OpenRadioss, and the case is at coupon scale. Thickness acts on the flow stress and the
density of a fixed mesh instead of moving the mesh, which holds for axial crush and under-represents bending stiffness.

Every design shares one mesh, so the geometry alone does not say *which* design a sample is: the seven variables travel
inside the sample as per-node channels (`FieldSample.extra`, mapped to `[0, 1]` by
`pinneapple_simulation.pipelines.pole_impact.design_vector`). Without them a field model can only reproduce the average
of the designs it was trained on; with them it is parametric, and a design held out of training can be asked about.
A campaign written before those channels existed is brought forward without re-running the solver by
`with_design_channels`, which rebuilds them from the design kept in each sample's `meta`.
