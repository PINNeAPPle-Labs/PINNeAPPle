# 3-D multiphysics optimisation of an electric motor on a pretrained surrogate

A traction motor is the textbook multiphysics problem: the electromagnetic torque heats the windings and the
magnets, the magnet temperature changes the remanence and the demagnetisation margin, the coolant flow sets the
temperature, the skew sets the ripple, and the stack length sets all of them at once. Optimising it with the FEA in
the loop costs one nonlinear field solve per candidate, which is why motor design still takes weeks.

This page covers the module that closes that loop **on a learned model of the whole chain**:
`pinneapple_simulation.electromagnetics.machines.motor3d`. The reference model is the physics; the surrogate is
trained on it once; the search then runs thousands of designs per second with no FEA and no CFD per candidate, and
the front it produces is re-checked against the physics.

## What exists in the library, and what this adds

| Piece | Where | Role here |
|---|---|---|
| 2-D nonlinear magnetostatic FEA of the cross-section, Prius 2004 geometry | `machines.IPMMachine`, `machines.prius_2004` | the field solve of the reference model |
| Air-gap mesher with a moving band, one sector copied around the circle | `machines.mesher` | the mesh |
| Iron loss from the flux density over an electrical period, magnet eddy loss | `machines.IPMMachine.losses` | the loss channels |
| End-winding resistance and leakage, outside the 2-D cross-section | `machines.endwinding`, `machines.drive` | the axial part of the winding |
| Lumped thermal network, one node per part | `machines.thermal.machine_network` | the radial thermal model this module extends along `z` |
| Demagnetisation margin against `Hcj(T)` | `machines.thermal.demagnetisation_margin` | the magnet check |
| **Axial-resolved thermal network** | **`motor3d.thermal3d`** | hot spot *place*, not only level |
| **The coupled multiphysics chain as one call** | **`motor3d.model.Motor3DModel`** | one design in, every KPI out |
| **Design of experiments + stored dataset** | **`motor3d.dataset`** | the training set with its provenance |
| **Ensemble surrogate with a model card** | **`motor3d.surrogate`** | fast, and honest about its error |
| **Search that runs on the surrogate only** | **`motor3d.optimize`** | zero FEA per candidate |
| **The whole study as one command** | **`motor3d.workflow`** | data, train, optimise, verify, report |

## The reference model

`Motor3DModel.solve(design)` runs the chain in one call. Each step is a piece of the library, and the coupling
between them is what makes the problem multiphysics:

1. **electromagnetic** -- `IPMMachine.operating_point` over a set of rotor angles at the design's current and current
   angle, with the magnet remanence at a first-guess magnet temperature;
2. **skew** -- the stack is cut into `n_segments` axial slices twisted by `skew_deg`; each slice contributes the
   torque of the cross-section at its own electrical angle, so slot harmonics cancel as they do in a skewed machine,
   and the fundamental is reduced by the skew factor `sin(x)/x`;
3. **losses** -- iron loss (stator and rotor) from the flux density over an electrical period, resistance-limited
   magnet eddy loss, and copper loss split between the slot and the end windings, with a winding resistance that
   depends on the stack length;
4. **cooling and temperature** -- the axial network of `thermal3d`, whose jacket conductance follows the coolant flow
   as `flow^0.8`;
5. **back to the electromagnetics** -- `Br(T)` of the grade corrects torque, iron loss (through `B^2`) and the
   demagnetisation margin, the copper loss is re-evaluated at the winding temperature and the network is solved again
   until the temperatures stop moving.

```python
from pinneapple_simulation.electromagnetics.machines.motor3d import Motor3DModel, Motor3DDesign, PRIUS

model = Motor3DModel()                       # 192-node gap mesh, 12 rotor angles, 4 axial segments
r = model.solve(PRIUS)                       # the 2004 Prius design at 250 A, 50 deg, 1500 rpm, 6 L/min
print(r["torque_Nm"], r["T_winding_C"], r["demag_ratio"])
print(r["T_winding_profile_C"])              # the winding along the stack, end to end
```

Designs whose cross-section the mesher refuses come back `valid=False` with the reason, without a solve.

### The axial thermal network

`machine_network` gives one temperature per part: enough for a level, not for a place. `AxialThermalModel` keeps the
same radial conductances and gives every part `n_segments` nodes along the stack, linked by `k A / dz`, with the end
windings as one node outside it and the jacket spread over the outer surface of every yoke segment.

The check that it is the same physics: with `n_segments=1` it reproduces `machine_network` to round-off
(`tests/test_motor3d.py::test_axial_network_with_one_segment_is_the_library_seven_node_model`), and with more
segments the winding becomes hottest in the middle of the stack instead of at a single averaged node.

```python
from pinneapple_simulation.electromagnetics.machines.motor3d import axial_network

net = axial_network(PRIUS.geometry(), n_segments=6)
P = net.powers(200.0, 400.0, 8000.0, 3000.0, 20.0, 40.0)   # yoke, teeth, slot Cu, end-winding Cu, rotor, magnets
T = net.network.steady(P)
print([round(T[net.index("winding", k)], 1) for k in range(6)])
```

## The dataset

```python
from pinneapple_simulation.electromagnetics.machines.motor3d import build_dataset

ds = build_dataset(1200, seed=20260101, workers=8, extra=[PRIUS])
ds.save("motor3d_data")
print(ds.meta["rejection_rate"], ds.meta["seconds_per_design"])
```

The eleven design variables (`motor3d.design.VARIABLES`) cover the rotor cross-section (`W4`, `H3`, `H0`, `W0`,
`H1`), the axial direction (`L`, `skew_deg`), the operating point (`I_peak`, `gamma_deg`, `rpm`) and the cooling
(`flow_L_min`). They are declared once, with units and bounds, and the design space, the columns of the dataset and
the inputs of the surrogate are all built from that declaration.

The DOE is the library's (`pinneapple_data.doe.design`) with the geometry test as a feasibility filter, so no case is
spent on a cross-section the mesher would reject -- and the rejection rate of the box is itself a result: at the
corners of the space a third of the designs cannot be built.

## The surrogate

`MotorSurrogate3D` is an **ensemble** of fully connected networks over the eleven inputs and 26 outputs: the fourteen
scalars of `OUTPUTS` and the three axial profiles of four segments each. Strictly positive outputs are trained in log
space, so a 300 W loss and a 30 N m torque train together.

```python
from pinneapple_simulation.electromagnetics.machines.motor3d import MotorSurrogate3D, SurrogateConfig

model = MotorSurrogate3D(SurrogateConfig(ensemble=5)).fit(ds)
print(model.card()["heldout"]["torque_Nm"])   # mae, r2, mean_rel, p95_rel, on designs never trained on
model.save("motor3d_model", version="v1")
```

Three properties matter more than the size of the network:

* **the ensemble spread** is the model's own uncertainty -- `predict` returns a mean *and* a spread per output;
* **validity**: `out_of_box` reports variables outside the training box, `extrapolation` the distance to the data in
  units of its own spread;
* **the card**: every saved model carries its held-out errors, its training-set description and the SHA-256 of its
  weights, and `load` refuses weights that no longer match the card.

The model shipped with the library is `load_pretrained()`.

## The optimisation, without FEA

```python
from pinneapple_simulation.electromagnetics.machines.motor3d import design_problem, optimize, verify_front

out = optimize(design_problem(surrogate=model), budget=4000, seed=0)
print(out["designs_per_second"], out["n_reference_solves"], len(out["front"]))    # ... 0 ...

ver = verify_front(out, n=8)          # the only place an FEA call is spent
print(ver["still_feasible"], ver["rank_agreement"], ver["errors"]["torque_Nm"])
```

The problem is the one the application asks: maximise torque density and efficiency, minimise magnet mass, subject to
the winding temperature, the magnet temperature, the demagnetisation ratio and the ripple. The search is NSGA-II from
`pinneapple_design.design_optimizer`, and every candidate it evaluates is a network forward pass.

`verify_front` is the honesty step: designs spread across the front are re-evaluated by the reference model, and the
result reports the per-output error, whether the designs are still feasible under the same limits, and the Spearman
agreement between the surrogate's and the reference's ranking of the front.

## The whole study

```bash
python -m pinneapple_simulation.electromagnetics.machines.motor3d.workflow all --out motor3d_run --n 1200 --workers 8
```

writes `dataset/`, `model/` (weights and card), `optimization.json`, `verification.json`, `benchmark.json`,
`figures/` and `report.md`, and every step has its own subcommand so the expensive one can be re-run alone.

## Limits

* the field is 2-D per axial segment: axial flux paths enter through the end-winding resistance and the thermal
  network, not through a 3-D field solve;
* the coolant is a boundary conductance that follows the flow as `flow^0.8`, not a conjugate heat transfer solution;
* the structural side (bridge stress at overspeed) is not evaluated;
* the surrogate is valid inside the box of its card, and its spread is a proxy for uncertainty, not a calibrated
  interval.

The alternative for the same goal is `machines.design.hybrid_vs_fea_only`, which keeps the FEA in the loop and uses a
Gaussian process to decide *which* designs deserve a solve. That is the right tool when every design must be FEA; the
surrogate here is the right tool when the chain is expensive enough to be worth learning once and the front is worth
verifying at the end.
