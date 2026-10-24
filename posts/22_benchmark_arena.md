Day 22 of 30: Physics AI Benchmark Arena

Which method should model your physics: a Gaussian process, an MLP, FNO, DeepONet or a PINN? Published comparisons use different data, budgets, stopping rules and seeds, so their rankings don't carry over.

The Benchmark Arena runs every method under one protocol. All methods get the same split, the same wall-clock budget per training and the same stopping rule, over several seeds. They are ranked per regime: interpolation, few data, extrapolation, geometry change and steep fronts. Next to the error it reports cost, constraint violations, the PDE residual and an independent audit of the splits.

In the video I upload 96 exact viscous Burgers fields and declare the PDE and the bound |u| ≤ 1.005. The run uses 1 s per training and 2 seeds, and takes 21 s. The Gaussian process ranks first everywhere: 0.510 % relative L2 error in interpolation and 1.71 % when extrapolating to low viscosity. The MLP has 0.956 % and the PINN 51.1 %.

The checks add what the leaderboard misses. 50 % of the MLP's held-out fields and 46.7 % of the GP's break the bound. The PINN breaks it 0 % of the time and has the lowest PDE residual ratio (0.214), yet it is the least accurate. Physics checks alone don't certify accuracy. The audit of the splits returns FAIL on effective sample count (1 of 54), so the ranking carries that flag.

Downloads: report.md and a StudyResult JSON.

Try it: https://arena.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/benchmark_arena)

#ScientificML #PhysicsInformedML #NeuralOperators #Benchmarking #SurrogateModeling #CFD #MachineLearning
