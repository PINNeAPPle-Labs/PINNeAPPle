Day 13 of 30: Simulation Campaign Planner

Simulation campaigns are often planned by hand: a grid that grows with every parameter, a few "interesting" combinations, and reruns of cases that already exist under another name. The holes only show up later, when a surrogate extrapolates into them.

The Simulation Campaign Planner takes a parameter space (bounds, units, constraints), a budget and the runs you already have. It returns a design of experiments (Latin hypercube, maximin, Sobol, factorial, or sequential infill around existing runs), a run order, coverage numbers before and after, and the case files.

In the video: an airfoil space with 7 free parameters (6 CST shape weights and the angle of attack) and 436 existing OpenFOAM runs. Infill places 24 new cases where those runs are sparsest. The biggest hole in the space (fill distance) goes from 0.730 to 0.688, 6 % smaller, and the discrepancy from 0.011 to 0.009. The download is a 5.0 MB zip with OpenFOAM simpleFoam case folders, tables of all 24 cases, a resumable Python runner and a DatasetSpec skeleton for the next step.

It also flags the old data: 2 near-duplicate pairs, and 18 runs (the NACA 4412 and 4415 polars) outside the declared bounds, left out of the coverage. All 4 plan checks pass: constraints, no repeated case, no repeat of an existing run, 1-D coverage. The page also states its limit: coverage is measured in parameter space, not in the response.

Live: https://campaign.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/simulation_campaign)

#DesignOfExperiments #CFD #OpenFOAM #SimulationEngineering #ScientificML #Aerodynamics #PINNeAPPle
