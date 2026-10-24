A PINN loss is a sum of terms: PDE residual, boundary values, data. Their scales can differ by orders of magnitude, and the usual fix is to hand-tune the weights between them.

PINNeAPPle now ships MultiAdam (Yao et al., ICML 2023), contributed by @F123Dash in PR #848. Each loss term keeps its own Adam moments, and the step averages the normalised per-term updates. You call opt.step([loss_pde, loss_bc]) with no backward(). Multiply one term by a constant and its update stays the same up to eps. The repo tests check this, plus an exact match with torch Adam for a single term.

No benchmark was committed, so we ran one ourselves on 4 CPU cores. We compared Adam and MultiAdam with the same net, points, steps and lr, on 3 seeds, with rel-L2 measured against the exact solution:
- 1-D Poisson with the boundary term weighted by 1e-3: median rel-L2 of 2.12 with Adam and 0.00375 with MultiAdam. Adam fits the PDE and gets the boundary wrong.
- The same problem with weight 1: Adam wins, 0.000212 against 0.0037.
- 2-D Helmholtz, sin(pi x) sin(4 pi y), 3000 steps: MultiAdam did not help (1.04, against 0.997 for Adam). Adam with betas (0.99, 0.99) was best at 0.119. MultiAdam's PDE residual was still falling when the run stopped.

So MultiAdam fixes one failure mode, a badly scaled term. It is not a default optimiser. These are small nets with a single lr and no tuning. The run script and the JSON with every curve are in the media folder.

Code: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/pinneapple_neural/trainer/multiadam.py

#PINN #PhysicsInformedML #SciML #Optimization #PyTorch #OpenSource #Python

---
## Media
1. `media/multiadam-pinn/01_method.png`: what MultiAdam changes compared with Adam, the two-line API, and credit to @F123Dash (PR #848).
2. `media/multiadam-pinn/02_poisson_weighted.png`: 1-D Poisson with the boundary term ×1e-3. rel-L2 vs step for the 3 optimisers (median and 3-seed band).
3. `media/multiadam-pinn/03_term_balance.png`: the loss terms vs step. Adam leaves the boundary error high, and MultiAdam drives both terms down.
4. `media/multiadam-pinn/04_weight_invariance.png`: final rel-L2 per seed for boundary weights 1 and 1e-3. MultiAdam is insensitive to the weight, and Adam wins at weight 1.
5. `media/multiadam-pinn/05_helmholtz_honest.png`: 2-D Helmholtz, where MultiAdam did not help within 3000 steps. rel-L2 and the PDE residual vs step.
6. `media/multiadam-pinn/06_summary.png`: a table of median rel-L2 for 3 cases × 3 optimisers, with the limits.
(Not for posting: `run_multiadam_bench.py`, `make_carousel.py` and `multiadam_bench.json` hold the reproducible run.)

## Sources
All run numbers come from media/multiadam-pinn/multiadam_bench.json, produced by media/multiadam-pinn/run_multiadam_bench.py (run from the repo root at main e716a065, torch 2.14.1, 4 threads, 351 s total).
- 2.12 (Adam, Poisson w_bc=1e-3): problems["poisson1d_wbc1e-3"].summary.adam.rel_l2_median = 2.1179
- 0.00375 (MultiAdam, Poisson w_bc=1e-3): problems["poisson1d_wbc1e-3"].summary.multiadam.rel_l2_median = 0.003748
- 0.000212 (Adam, Poisson w_bc=1): problems["poisson1d_wbc1"].summary.adam.rel_l2_median = 0.0002121
- 0.0037 (MultiAdam, Poisson w_bc=1): problems["poisson1d_wbc1"].summary.multiadam.rel_l2_median = 0.003696
- 1.04 / 0.997 / 0.119 (Helmholtz MultiAdam / Adam / Adam betas 0.99): problems["helmholtz2d"].summary.{multiadam,adam,adam_b99}.rel_l2_median = 1.0407 / 0.99695 / 0.11878
- "PDE residual still falling" (Helmholtz MultiAdam): problems["helmholtz2d"].runs["multiadam/s*"].terms[*][0], going from about 7180 to 250 / 1948 / 53.7 at step 3000
- Adam boundary error 3-7 (slide 3): problems["poisson1d_wbc1e-3"].runs["adam/s*"].final_terms[1] / 1e-3 = 3.08, 7.00, 6.69
- MultiAdam range 1.7e-4 to 9.6e-3 (slide 4): problems["poisson1d_wbc*"].summary.multiadam.rel_l2_per_seed
- Scale invariance and the single-term match with torch Adam: tests/test_multiadam.py::test_rescaling_one_term_does_not_change_the_step, ::test_single_term_matches_adam
- Algorithm and the PR #848 / @F123Dash credit: pinneapple_neural/trainer/multiadam.py docstring; git log 223d95cd "Merge pull request #848 from F123Dash/feat/multiadam-optimizer"
