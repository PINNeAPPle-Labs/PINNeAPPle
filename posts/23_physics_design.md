Day 23 of 30: Physics Design Optimizer

A differentiable surrogate, PINN or neural operator lets an optimiser follow gradients instead of sampling blindly. Whether that beats NSGA-II or Bayesian optimisation on a given design is rarely measured, and when it is, the methods often got different budgets and the "optimum" was never re-run in the solver.

The Physics Design Optimizer takes a parametric design with explicit objectives, constraints and a physics model, runs gradient, evolutionary and Bayesian methods on the same model, budget and seeds, and re-runs the chosen candidates in the high-fidelity model.

In the video: a six-parameter plate-fin heat sink (60 W, 35 °C air at 2 m/s), minimising mass and pressure drop under limits on temperature, pressure drop, fin stress and fin gap. Five methods, 80 evaluations each, one seed. Gradient through the surrogate gave the best front: hypervolume 1.18, against NSGA-II 1.14, Bayesian 1.10, Latin hypercube 1.06 and SLSQP 0.90. It reached 95 % of the best front after 40 evaluations and was still improving at 80.

The 3-D finite-volume engine verified 4 of 4 candidates, with surrogate errors of 0.24-0.91 % for temperature and pressure drop. Each candidate shows its constraint margins, with the active temperature limit flagged. The gradients are the surrogate's, not the solver's, so a gradient optimum is verified like any other, and one seed does not rank methods in general. Downloads: STL/STEP per candidate, CSVs and a StudyResult with 8 of 8 checks passing.

Live: https://design.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/physics_design)

#DesignOptimization #PhysicsInformedML #Surrogates #BayesianOptimization #ThermalEngineering #Simulation #EngineeringDesign
