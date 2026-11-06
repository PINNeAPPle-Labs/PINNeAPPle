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
```

Outputs in `_out/` (not committed — regenerate them):

| file | what it is |
|---|---|
| `pole_impact_fields.npz` | the field dataset: points, part labels, times and the displacement of every node |
| `pole_impact_quantities.csv` | the tabular dataset: design vector + quantities, one row per design |
| `tabular_leaderboard.md` / `.json` | Ridge … CatBoost on the same split, plus the permutation importance of the variables |
| `field_leaderboard.md` / `.json` | the field models ranked with per-design relative L2 and bootstrap intervals |
| `campaign.json` | the manifest: generator, solver note, design box, mesh, end time — and the designs used |
| `failures.json` | every design whose run raised (nothing is dropped silently) |

The labels come from `pinneapple_simulation.numerical_solvers.crash_fem` (explicit hexahedral dynamics, J2 plasticity,
penalty contact), not from OpenRadioss, and the case is at coupon scale. Thickness acts on the flow stress and the
density of a fixed mesh instead of moving the mesh, which holds for axial crush and under-represents bending stiffness.
