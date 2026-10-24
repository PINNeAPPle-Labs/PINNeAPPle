Day 17 of 30: Engineering Design Optimizer

Heat sinks and similar parts are often sized by hand: a few variants in a spreadsheet or a solver, with temperature, pressure drop, stress and extrusion limits checked at the end. Optimisers exist, but the usual failure is an "optimum" from a fast model that the real solver rejects.

The Engineering Design Optimizer takes design variables, objectives and physical and manufacturing constraints. It searches with a fast model (a zoo surrogate, a model fitted to your table of designs, or a closed-form model), then re-runs the chosen candidates with a high-fidelity reference and reports what holds.

In the video, NSGA-II makes 600 calls to the heat-sink neural surrogate: 371 feasible designs, 6 on the Pareto front. The best mass is 82.4 g against 486 g for the baseline (-83 %), and the best unit cost is 2.355 against 6.145 (-62 %, with a placeholder cost model). The 3-D finite-volume engine re-ran 5 candidates. All 5 were verified, with the largest fast-vs-reference gap at 1.45 % (pressure drop 10.69 vs 10.54 Pa).

For each candidate, the feasibility report gives every constraint's value, margin and status. One candidate runs at 84.42 of the 85 °C limit. That constraint is active, and the reference gives 84.39 °C. The StudyResult passes 9 of 9 checks. candidates_to_run.csv is ready for your own solver.

Live: https://optimize.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/engineering_optimizer)

#DesignOptimization #MultiObjective #SurrogateModels #ThermalEngineering #Simulation #EngineeringDesign #NSGA2
