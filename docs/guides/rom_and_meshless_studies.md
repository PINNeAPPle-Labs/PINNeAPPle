# Reduced-order models and meshless methods: measured, not assumed

Two studies that answer "which method, at which cost, for which error" on your data, with every number measured on
held-out snapshots or against a known solution. They are the library behind the ROM Lab (day 19) and the Meshless
Method Lab (day 26).

## ROM study

```python
from pinneapple_data.snapshots import load_snapshots
from pinneapple_neural.architectures.rom import rom_study, ReducedModel

snaps = load_snapshots([("wake.npz", open("wake.npz", "rb").read())])   # OpenFOAM times, VTU series, npz, CSV
res = rom_study(snaps, ranks=[2, 4, 8, 16], budget_s=5)                # forecast split: the last 20 % of times
print(res.to_dict(tolerance=0.05)["best"])                              # fastest model under 5 % held-out error
m = res.models[("dmd", 16)]
open("wake_dmd.rom.npz", "wb").write(m.to_bytes())                      # arrays + JSON manifest, no pickle
open("wake_dmd.onnx", "wb").write(m.to_onnx())                          # for ModelGateway.publish(kind="onnx")
```

| Method | Data | What it is |
|---|---|---|
| `pod_rbf` | both | POD + thin-plate RBF regression of the coefficients (in time: interpolation, poor extrapolation) |
| `pod_gpr` | parametric | POD + one Gaussian process per coefficient |
| `nearest` | parametric | the nearest training snapshot: the baseline every ROM must beat |
| `dmd` | time | exact (uncentred) DMD; the mean is one of its modes |
| `opinf` | time | quadratic Operator Inference on centred POD coefficients, regularisation chosen inside the training window |
| `neural` | both | an MLP trained within `budget_s` seconds (`train_within_budget`): parameters to field, or a residual stepper of the POD coefficients |

Every row reports the held-out error (time series: relative to the fluctuation about the training mean; several
fields: each scaled by its training RMS, with per-field errors), the POD projection error at that rank (the floor),
fit time, inference time per snapshot (`time_per_query`) and the speed-up when the full-order time per snapshot is
known (`hf_seconds`). Time steppers predict from the latest training snapshot at or before the requested time.

## Meshless study

```python
from pinneapple_simulation.meshless import Problem, domain, meshless_study

res = meshless_study(Problem("poisson", exact="sin(pi*x)*cos(2*y) + x*y"), domain("lshape"), h=[0.1, 0.05, 0.025])
for w in res.wins:
    print(w["criterion"], w["method"], w["value"])
```

Kansa, RBF-FD, embedded finite differences (Shortley-Weller), P1 finite elements on the Delaunay mesh of the nodes
and a PINN on the same nodes, for Poisson, Helmholtz and heat (Crank-Nicolson) with Dirichlet conditions. Measured
on the L-shape above (`tests/test_meshless.py`): RBF-FD (degree 4) converges at order ≈ 4.4, FD and FEM at ≈ 2,
Kansa reaches 1e-6 with a condition number of 1e20 (and is unstable when time-stepped), FEM has the best conditioned
system. Linear elasticity and Neumann conditions are not implemented yet; R-OAMS is not in the library.
