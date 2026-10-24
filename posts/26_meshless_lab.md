Day 26 of 30: Meshless Method Lab

Meshing complex or moving domains is hard, and meshless methods promise to skip it. They bring their own costs: dense or ill-conditioned systems, shape parameters, stencil choices, training budgets. Which method is right for a given domain and tolerance is an empirical question.

The Meshless Method Lab runs the comparison instead of assuming an answer. Kansa (global RBF), RBF-FD and a PINN run on the same nodes as finite differences and linear finite elements, over a refinement study, with errors measured at 2,000 independent points.

On the example, a Poisson problem on an L-shape over three spacings, the study took 11.3 s. RBF-FD was the most accurate at h = 0.025 (relative L2 7.4e-7, observed order 4.36). Kansa was the cheapest way to reach 0.001: 1.2 ms on 96 nodes. But its condition number reached 9e20 and its order stalled at 0.40 on the last refinement. FD and FEM stayed near 9e-4 with the expected order of 2. FEM had the best-conditioned system (475) and was the least sensitive to perturbed data. The PINN, given 3 s of training on a busy shared CPU (49 steps), reached 0.44.

Each criterion names its winner from the measured numbers. The checks confirm the FD and FEM orders match theory (2.01 and 1.96) and that every method gave a finite solution. A scope card warns that smooth exact solutions favour high-order methods; a corner-singularity example is included to test that. StudyResult JSON, every solution as CSV and the full study are downloadable.

Live: https://meshless.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/meshless_lab)

#MeshlessMethods #RBF #FiniteElements #PINN #NumericalMethods #ScientificComputing #Engineering
