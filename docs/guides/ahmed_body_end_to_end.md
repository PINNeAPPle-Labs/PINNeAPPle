# An end-to-end CFD surrogate study: the Ahmed body

From geometry to a digital twin and back to wind-tunnel data, every step a library call and every number measured:
`examples/end_to_end/ahmed_body` (scripts `01`-`07`), the app `apps/ahmed_twin` and the flow
`pinneapple_flow/flows/ahmed_body_end_to_end.json` run the same calls.

## Why the Ahmed body

It is the standard bluff-body test case of road-vehicle aerodynamics: a public, fully specified geometry
(1.044 x 0.389 x 0.288 m, 0.1 m front radii, a 0.222 m rear slant), a published drag curve against the slant angle
(Ahmed, Ramm & Faltin, SAE 840300, 1984, 60 m/s) that has a known difficulty (the drag peak just below 30 degrees,
which steady RANS does not reproduce), and an external flow that the generic OpenFOAM pipeline
(`pinneapple_simulation.numerical_solvers.external_flow.ExternalFlow`) already runs for any STL. A finned heat sink
would have given correlations rather than measurements for the inverse step; the Ahmed body gives measurements.

## Stages and the calls behind them

| Stage | Library | Example |
|---|---|---|
| Geometry: parametric (`make_body`, a fixed 1751-point field template for every variant) and real CAD (`ahmed_cad`: OpenCASCADE B-rep through gmsh, STEP + STL) imported back with checks and named patches (`import_body`) | `pinneapple_autolab.domains.ahmed` | `01_cfd_campaign.py cad` |
| Simulator: snappyHexMesh with 5 prism layers, simpleFoam k-omega SST with wall functions, moving road, half model; per run checkMesh, mesh quality, residuals, force stationarity, mass balance from `phi` (`pinneapple_check`), y+ range (`qa_checks`, `run_trust_report`) | `AhmedAero` | |
| Mesh family with a constant refinement ratio sqrt(2); Richardson extrapolation and GCI (Celik et al. 2008) | `pinneapple_autolab.study.mesh_study` | `01 ... mesh`, `02_analysis.py` |
| Validation against the wind tunnel | `ahmed_study.grid_and_validation` | `01 ... validation`, `02` |
| DOE: maximin Latin hypercube with discrepancy and minimum distance against random designs; resumable runs with a cost ledger | `pinneapple_autolab.study.campaign` | `01 ... doe` |
| Dataset: scalars, surface Cp, provenance as a `DatasetSpec`; slant > 34 deg held out as the extrapolation split | `pinneapple_autolab.study.dataset`, `ahmed_study.study_dataset` | `02` |
| Surrogates at equal data and budget (GP, MLP ensemble, response surface; POD-GP, POD-MLP, POD-RBF, DeepONet), 5-fold CV, cross-conformal 90 % intervals, physics checks on predictions, a written selection rule | `pinneapple_autolab.study.benchmark` | `02` |
| Sobol indices, NSGA-II with an uncertainty constraint and CFD checks of the Pareto picks, acquisition by uncertainty against random runs | `pinneapple_autolab.study.extras` | `02`, `03_more_cfd.py` |
| Inverse: amortised posterior (mixture density network) of the slant from a drag measurement; synthetic checks first; least squares and Metropolis baselines | `pinneapple_analysis.inverse_problems.amortized`, `ahmed_study.inverse` | `04_inverse.py` |
| Digital twin: sliders, Cp in 3-D, 90 % band, trust-region warning, verifying CFD | `pinneapple_autolab.study.twin`, `apps/ahmed_twin` | `05_twin.py` |
| Visuals: Blender Cycles stills and video on a dark stage, three.js viewer, OpenUSD stage with the trust decision | `ahmed_study.visuals`, `pp.viz.render(..., background="dark", cmap="rdbu", floor=True)` | `06_visuals.py` |
| Evidence report and package, cost accounting, experiment manifest | `pinneapple_autolab.study.report` | `07_report.py` |

## The physics checks on surrogate predictions

A surrogate's prediction is checked like a solver result (`ahmed.prediction_checks`), on a Sobol sample of the
design space: drag and pressure drag positive; friction drag positive and falling as the speed (the Reynolds number)
rises; the pressure coefficient never above the incompressible stagnation value 1 (with a margin) and close to 1 on
the nose; and the pressure drag integrated from the predicted Cp field equal to the scalar model's pressure drag,
within the tolerance the CFD itself shows on the same template. A family that fails one is not eligible.

## Results

The measured tables (grid convergence, validation, surrogate errors and coverage, inverse results, costs) are in
`examples/end_to_end/ahmed_body/README.md` and `results/report.html`, written from the JSON files in `results/`.

## Limits

Steady RANS with k-omega SST and wall functions; the stilts of the wind-tunnel model are left out; the reference is
the total drag of one wind-tunnel campaign. The surrogates are valid in the DOE's box (slant 0-40 deg, clearance
30-80 mm, speed 20-60 m/s) and the twin says when a slider point is far from every CFD run.
