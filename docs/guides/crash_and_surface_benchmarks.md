# Crash and surface-pressure benchmarks, recreated

Two recent vehicle-engineering benchmarks ask the same question in two regimes: *can a learned surrogate replace the
solver on a geometry family, at a cost that makes design iteration possible?* This guide recreates both **inside
PINNeAPPle** — same protocols, same metrics, same way of splitting — with the library's own label generators, so the
whole chain (geometry → simulation → dataset → surrogate → leaderboard) stays in one place and every number carries its
provenance.

| the published benchmark | this recreation |
|---|---|
| [CarBench](https://arxiv.org/abs/2512.07847) — *A large-scale benchmark for vehicle aerodynamic surrogates* (Commun. Eng., [s44172-026-00805-2](https://www.nature.com/articles/s44172-026-00805-2)), on DrivAerNet++ | [Surface pressure over a parametric car family](#1-surface-pressure-over-a-parametric-car-family) |
| [CarCrashNet](https://arxiv.org/abs/2605.07098) — *a large-scale dataset and hierarchical neural solver for structural crash simulation* | [Pole impact of a bumper-beam coupon](#2-pole-impact-of-a-bumper-beam-coupon) |

Both papers are read for their protocol, not copied: no code from them is used, and where a row of their tables has no
implementation here, the row is named in the "not implemented" list below rather than filled with a stand-in.

## What the recreation adds to the library

| module | what it is |
|---|---|
| `pinneapple_simulation.numerical_solvers.crash_fem` | explicit structural crash dynamics: central-difference integration at the exact critical step, 2×2×2 Gauss points, corotational kinematics, J2 plasticity with erosion, penalty contact against a rigid wall or pole, energies and crashworthiness quantities |
| `pinneapple_simulation.numerical_solvers.crash_cases` | the pole-impact coupon (bumper beam + two crash boxes, seven design variables, a fixed mesh with 3 to 11 semantic parts) |
| `pinneapple_simulation.pipelines.pole_impact` | a sweep of that coupon → displacement-sequence dataset + quantities table + manifest |
| `pinneapple_simulation.pipelines.car_pressure` | a parametric car family → lattice-Boltzmann wall pressure → surface-pressure dataset + coefficients table |
| `pinneapple_simulation.numerical_solvers.lbm_tunnel` (extended) | `wall_pressure`, `surface_pressure`, `run_averaged`, `pressure`, `sample_field`: the wall-pressure extraction a surface-presslabel needs |
| `pinneapple_neural.architectures.crash_solver` | the hierarchical, part-aware surrogate (local component attention → part transformer → interface message passing → temporal decoder) |
| `pinneapple_neural.architectures.point_surrogates` | PointNet, its wide residual variant, EdgeConv, a part-aware PointNet |
| `pinneapple_arena.fieldbench` | the benchmark itself: containers, metrics with bootstrap intervals, paired significance tests, one trainer for every row, the three-resolution protocol, the leaderboard, and the tabular counterpart |

## 1. Surface pressure over a parametric car family

Geometry point cloud in, **surface kinematic pressure** out, over a family of cars built by
`pinneapple_design.geometry.vehicles3d.RoadCar`: three roof lines (fastback, notchback, estateback) × open or closed
wheel arches × thirteen numeric design variables (length, width, roof height, ride height, deck height, windshield and
rear-window angles, boat tail, diffuser angle, wheel radius/width, wheelbase, front overhang). Every design shares the
same protocol: the body is voxelized into the library's 3-D lattice-Boltzmann tunnel (D3Q19, Smagorinsky, moving road),
run to a steady state (and time-averaged where it sheds), and the pressure of every wall face becomes the label of the
point at its centre.

```bash
python examples/carbench_surface_pressure/run.py --designs 40 --parallel 7 \
    --cells-per-length 44 --steps 3000 --epochs 40
```

Outputs, in `examples/carbench_surface_pressure/_out/`: `surface_pressure.npz` (a
`pinneapple_arena.fieldbench.FieldDataset`), `coefficients.csv` (Cd, Cl and areas per design), `leaderboard.md` and
`leaderboard.json`, and the Lab manifest of the run.

**The label is coarser than the published one, and the campaign says how coarse.** CarBench labels its designs with
steady RANS (k-ω SST) on surface meshes of ~450,000 points; here the label is a lattice-Boltzmann field on ~2,000 wall
faces, at a resolution where the lattice Reynolds number is O(10²), not the 10⁷ of a car. Three consequences are
recorded with every dataset instead of being glossed over:

- the **Reynolds number** the grid can hold (`re`, `re_max` per design) and the resolution (`cells_per_length`);
- the **largest density excursion** of the lattice (`max_density_deviation`): a weakly-compressible solver that reaches
  20 % in density is telling you its Mach number is not small;
- the **label's own coefficients** (`cd`, `cl` in `coefficients.csv`): a coarse cascade over a car body over-predicts
  drag, and a campaign whose mean Cd is far from the physical 0.3 is a campaign whose labels are qualitative.

Because the ground clearance of a road car is one cell at this resolution, the default labels the body **in free air**
(`road=False`): a one-cell gap between the underbody and the moving road is a staircase nozzle and produces pressure
spikes that have nothing to do with the car. `road=True` puts the road back — and is only meaningful once the ride
height is resolved by several cells, which costs roughly the square of the cells per length. Absolute error levels are
therefore **not** comparable with the published table; the protocol, the ranking of families and the cost are.

## 2. Pole impact of a bumper-beam coupon

A DP1000 bumper beam held by two DP600 crash boxes, driven into a rigid pole — the component-scale case of the crash
corpus. Seven design variables (impact velocity, the two wall thicknesses, the two flow stresses, pole radius, lateral
offset) produce the five crashworthiness quantities of the published benchmark (peak wall force, peak deceleration,
peak internal energy, peak plastic work, kinetic-energy absorption) **and** the displacement of every node at every
frame, so the same campaign feeds a tabular benchmark and a field benchmark.

```bash
python examples/carcrashnet_pole_impact/run.py --designs 120 --parallel 7 --end-time 1e-3
```

Two things are different from the paper and are stated in the campaign manifest as well: the solver is this library's
explicit hexahedral crash solver (J2 plasticity, erosion, penalty contact) rather than OpenRadioss with shells, and the
case is at coupon scale. The *coupon* keeps one property the full-vehicle benchmark relies on: a **fixed mesh**, so
every design shares its nodes and part labels, which is what a field surrogate (and a spatial comparison between
designs) needs. Thickness acts through the flow stress and the density instead of moving the mesh, a scaling that holds
for axial crush and under-represents bending stiffness — the note is in `crash_cases`.

### The part hierarchy

`bumper_beam_assembly(n_parts=...)` gives the same mesh at three hierarchy depths (3, 6 and 11 semantic parts), which is
how the published ablations sweep the part information: `crash_box_left`, `bumper_beam_y2`, `crash_box_right_front`, …
The hierarchical surrogate (`CrashSolverModel`) uses those labels as its decomposition, and `PartPointNetSurrogate`
uses them as one more input channel — the difference between the two is exactly what the ablation measures.

### The design vector, and the quantity-level KPI

The coupon has a **fixed mesh**, which is what lets designs be compared node by node and is also why the geometry
alone cannot identify a design: every sample has the same points, the same part labels and the same wall thickness.
The design reaches the model as data — `FieldSample.extra`, the seven variables mapped to `[0, 1]` by
`pinneapple_simulation.pipelines.pole_impact.design_vector` and broadcast to every node — and `run.py --conditioning
design` feeds it to the rows (`build_model_specs(extra_dim=7)`). Without it a field surrogate can only reproduce the
average of the training designs; with it, a design held out of the split is a legitimate question.

The benchmark scores the field, so it is easy to forget what the design was run for. `extreme_errors` scores the
**other** quantity from the same predictions: the extreme of one channel over the points and frames — the peak
displacement along the impact direction, the number a crash report quotes. Both are logged: `field_best_rel_l2` for the
field and `kpi_peak_displacement_*` for the quantity, and the figures of the run are the parity plot of the peak and the
deformed beam drawn as a surface (reference and surrogate on the same colour scale).

## The metrics, and why they are reported the way they are

`pinneapple_arena.fieldbench.metrics` keeps two aggregations apart, as both benchmarks do:

- **relative L2 per geometry, then averaged** — the headline number, insensitive to how many points a geometry happens
  to have;
- **R² and absolute errors pooled over every point** of every geometry, with the P50/P90/P95/P99 profile.

Every mean carries a bootstrap interval (stratified by category when the categories are unequal), and a comparison
between two models is a **paired** bootstrap over the per-geometry scores: mean difference, its interval, the share of
geometries where one wins, plus the sign-flip permutation and Wilcoxon signed-rank p-values. A ranking whose interval
crosses zero is reported as a tie, not as a win.

The three-resolution protocol of the aerodynamic benchmark is `PatchwiseProtocol`: score only the points the model saw
(`subsampled`), interpolate its prediction to the full field (`interpolated`, RBF) or predict patches and keep their
centres (`patchwise`).

## What is *not* implemented here

Named explicitly, so no table row is a stand-in:

- **GeoTransolver**, **FIGConvUNet**, **PointMAE**, **PointTransformer**, **TripNet**, **AB-UPT** and the voxel FNO of the
  aerodynamic benchmark have no implementation in this repository. The operator and point rows that do exist
  (`transolver`, `transolver_lite`, `transolver_large`, `pointnet`, `pointnet_wide`, `edgeconv`, `part_pointnet`,
  `crash_solver`) carry their own names.
- **TabPFN v2** and **AutoGluon** are not part of the tabular comparison; every other family of the published table
  (Ridge, Lasso, kNN, SVR, MLP, random forest, extra trees, XGBoost, LightGBM, CatBoost when installed) is.
- **OpenRadioss, LS-DYNA and OpenFOAM** are not run: the labels come from this library's own solvers.

## Choosing between the rows

For a geometry family with a few thousand surface points and ~10² designs, the physics-attention family converges
fastest per epoch and the point models are the cheapest per prediction; the hierarchical model only earns its
extra machinery when the structure really has parts (it does on the crash coupons, it does not on a car body treated as
one closed surface). The tabular campaign is the opposite lesson: gradient-boosted trees beat every neural model on
seven scalar inputs, and the linear models already recover most of the energy-like targets.

## References

- Elrefaie, Shu, Klenk & Ahmed, *CarCrashNet: a large-scale dataset and hierarchical neural solver for data-driven
  structural crash simulation*, arXiv:[2605.07098](https://arxiv.org/abs/2605.07098).
- *A large-scale benchmark for vehicle aerodynamic surrogates* (CarBench, on DrivAerNet++), Communications
  Engineering, [s44172-026-00805-2](https://www.nature.com/articles/s44172-026-00805-2);
  arXiv:[2512.07847](https://arxiv.org/abs/2512.07847).
- Wu et al., *Transolver: a fast transformer solver for PDEs on general geometries*, ICML 2024,
  arXiv:[2402.02366](https://arxiv.org/abs/2402.02366) — the physics attention both the operator rows and the local
  component encoder use (already in the library, `pinneapple_neural.architectures.neural_operators.transolver`).
- Qi et al., *PointNet*, CVPR 2017, arXiv:[1612.00593](https://arxiv.org/abs/1612.00593); Wang et al., *Dynamic Graph
  CNN for learning on point sets*, ACM TOG 2019, arXiv:[1801.07829](https://arxiv.org/abs/1801.07829).
- Hou et al., *A lattice Boltzmann model for the simulation of high Reynolds number flows*, 1994, and Lehmann et al.,
  2022, on 16-bit storage of shifted distributions — the methods the tunnel in `lbm_tunnel` implements.
- Elrefaie et al., *DrivAerNet++*, the aerodynamic corpus the CarBench splits follow.
