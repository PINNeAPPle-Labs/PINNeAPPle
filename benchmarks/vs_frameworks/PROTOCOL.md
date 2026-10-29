# Cross-framework benchmark protocol: PINNeAPPle vs PhysicsNeMo / PhysicsNeMo-Sym, DeepXDE, neuraloperator

Goal set by the owner: **understand the current state, not make PINNeAPPle win.** A result where PINNeAPPle loses
is reported the same way as one where it wins. This file is committed before any benchmark result; every later
change is listed under "Deviations" in `REPORT.md` with its reason.

The neutral parts (problems, references, point sets, initial weights, evaluator, statistics, run records, the Darcy
data and the operator training loop) are in `pinneapple_tools.benchmark_suite.crossframework`; nothing there imports
a competitor. One runner per framework and track is in this folder; each starts with a comment citing the official
example or documentation it follows.

## Frameworks (pinned)

| Framework | Version | Notes |
|---|---|---|
| PINNeAPPle | this repository at the commit under test | |
| PhysicsNeMo | `nvidia-physicsnemo==2.2.2` | operators (FNO) |
| PhysicsNeMo-Sym | `nvidia-physicsnemo.sym==2.4.0` | PINNs |
| DeepXDE | `deepxde==1.15.0`, `DDE_BACKEND=pytorch` | every framework on the same torch |
| neuraloperator | `neuraloperator==2.0.0` | reference FNO |

All on torch 2.14.1 (CPU), Python 3.13, float32. A framework that cannot be installed or run is a result; it is never
replaced by a re-implementation under its name.

## Hardware and timing

* CPU only (no GPU). PhysicsNeMo is GPU-first (AMP, CUDA graphs, multi-GPU); those paths are not exercised. Every
  runner takes `--device cuda` for a later GPU run.
* One thread per run (`torch.set_num_threads(1)`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`), at most two runs at a time on
  the 4 cores. Each run records the CPU model, wall time, process CPU time, iterations and the load average before
  and after; the driver records how many runs were going at the same time.
* Time caps (A2) are **process CPU time** (with one thread and a quiet machine this equals wall time; on a shared
  machine it does not depend on the other processes). Per-iteration costs are also measured on a quiet machine
  (load < 1, one run at a time) in a separate short timing run.

## Tracks

### A. PINN forward problems (PINNeAPPle, DeepXDE, PhysicsNeMo-Sym)

Problems (`crossframework/problems.py`):

| Problem | Equation and data | Reference |
|---|---|---|
| Burgers 1-D | `u_t + u u_x = nu u_xx`, nu = 0.01/pi, x in [-1, 1], t in [0, 1], u0 = -sin(pi x), u(+-1) = 0 (Raissi et al. 2019) | Cole-Hopf by Gauss-Hermite quadrature, `pinneapple_physics.closed_form.burgers.burgers_sine_exact` (checked: 120 nodes agree with 250 nodes to 3e-15 and with adaptive quadrature to 1e-14) |
| Poisson 2-D | `u_xx + u_yy = f` on the unit square, u = 0 on the boundary, u = sin(pi x) sin(pi y) | manufactured |
| Helmholtz 2-D | `u_xx + u_yy + k^2 u = q` on [-1, 1]^2, u = 0 on the boundary, u = sin(a1 pi x) sin(a2 pi y), a1 = 1, a2 = 4, k = 1 (Wang, Teng, Perdikaris 2021) | manufactured |
| Allen-Cahn 1-D | `u_t = d u_xx + 5 u - 5 u^3`, d = 1e-4, periodic on [-1, 1], u0 = x^2 cos(pi x) (Raissi et al. 2019) | Fourier pseudo-spectral (2048 modes) + ETDRK4 (dt = 1e-4), `crossframework.problems.allen_cahn_reference`; resolution study in the tests (2048 vs 4096 modes: max difference 1.7e-4) |

Test points: Burgers 256 x 100 grid on [-1, 1] x [0, 0.99] (Raissi's grid); Poisson and Helmholtz 101 x 101;
Allen-Cahn 512 x 201 (periodic x grid). Residual points: 2000 fresh uniform interior points.

**A1, identical configuration.** tanh MLP 2 -> 50 x 4 -> 1, Xavier-uniform weights and zero biases, float32. The
weights of seed s are drawn once (`crossframework/weights.py`) and loaded into every framework's network, so seed s
starts from the same function everywhere. The same training points, generated once and saved
(`crossframework/protocol.py::ensure_point_set`): N_f = 10000 uniform interior points, 400 boundary/initial points
(time-dependent: 200 at t = 0 and 100 on each wall; steady: 100 per edge). Loss = mean squared PDE residual + mean
squared boundary error (+ mean squared initial error), weights 1. Full batch, the same points every iteration. Adam,
lr 1e-3 constant, no gradient clipping. A fixed number of iterations, the same for all (2000; chosen from the
measured cost so that a run takes 3-7 min on this CPU). Seeds 0-4. Framework settings that differ from this by
default are switched to it and named in the runner (PhysicsNeMo-Sym: weight norm off, gradient clipping 0.5 off,
per-point weight 1/N because its loss sums over points; PINNeAPPle: condition weight 1 and `LossWeights(1, 1, 1, 1)`
instead of 10 x 10).

Allen-Cahn is not run in A1 (first item of the cut order below, decided in advance from the compute budget).

**A2, each framework's own recipe** under the same time cap per run (240 s of process CPU time, the same cap for all), seeds
0-2. Each framework uses the configuration of its official example for that PDE, or its documented recommended
options; where it has no example for a PDE, its closest one, said so in the runner:

* PINNeAPPle: `pp.solve(problem, "pinn", ...)` with its defaults (modified MLP 64 x 4, Adam lr 1e-3, 2048 fresh
  collocation points per step, its default condition weights). `pp.solve` has no time cap, so the number of epochs is
  set from a 100-epoch probe to fill the cap.
* DeepXDE: `examples/pinn_forward/{Burgers, Poisson_Lshape, Helmholtz_Dirichlet_2d, Allen_Cahn}.py` (Adam then
  L-BFGS where the example does, hard constraints where the example uses them); a callback stops at the cap.
* PhysicsNeMo-Sym: `examples/helmholtz` (Helmholtz; also the structure for Poisson) and `examples/wave_equation/
  wave_1d.py` (structure for Burgers and Allen-Cahn): the default `fully_connected` arch, its batch sizes, Adam with
  `tf_exponential_lr`, loss `sum`; `Solver.solve(sigterm_handler=...)` stops at the cap.
* If one framework uses a trick in A2 (hard constraints, L-BFGS, resampling), the others may use their documented
  equivalent only if their own example does.

### B. Operator learning (PINNeAPPle FNO, PhysicsNeMo FNO, neuraloperator FNO)

* Data: zoo dataset `darcy_2d_li2021` (1000 train / 200 test, 128 x 128). The Hugging Face download is blocked here,
  so the dataset is regenerated with the library recipe (`operator_datasets.generate(only="darcy")`) and accepted
  only if both files match the SHA-256 of the zoo manifest. Downsampled to 64 x 64 by cell averaging
  (`operator_datasets.block_average`), the same for all.
* Split: train rows 0-899, validation 900-999 (reported, never used to choose anything), test 200.
* Normalisation (the same for all): input a scaled by the training mean and standard deviation; output u per-pixel
  unit Gaussian from the training split, decoded before the loss.
* FNO: 4 Fourier layers, 12 modes per dimension, width (hidden channels) 32, a grid positional channel; everything
  else each library's default (lifting / projection sizes, padding, skip type). Parameter counts are reported and
  structural differences named.
* Training, the same loop for all three models (`crossframework/darcy.py`): batch 20, Adam lr 1e-3, weight decay
  1e-4, cosine schedule over all steps, loss = per-sample relative L2 averaged over the batch, the same epochs and
  the same data order per seed. Seeds 0-2.
* Metrics: test relative L2 (mean of per-sample values; median and max too), parameter count, time per epoch (CPU,
  one thread), inference latency (batch 1 and batch 20).

### C. Inverse problem (PINNeAPPle, DeepXDE; PhysicsNeMo-Sym through its documented inverse approach if it runs)

* Identify nu of Burgers from 2000 samples of the exact solution at uniform random points with 1 % Gaussian noise
  (standard deviation 0.01 x std(u) of the samples), the same samples for all (`crossframework/inverse.py`).
* The same network as A1 (shared weights), Adam lr 1e-3, the same number of iterations; nu starts at 0.05 (true
  value 0.003183). Each framework's documented way to make nu trainable: PINNeAPPle
  `pinneapple_neural.trainer.inverse_pinn.InversePINN` (with its built-in loss balancing, clipping and early
  stopping), DeepXDE `dde.Variable` + `external_trainable_variables` (`examples/pinn_inverse/diffusion_1d_inverse.py`).
* Metrics: relative error of nu, field relative L2 on the Burgers test grid, time. Seeds 0-4.

### D. Measured ergonomics (optional, factual only)

Non-comment lines of each runner, install size of each framework in the venv, import time. No opinions.

## Neutral evaluation

Every runner exports predictions on the same test points (and the 5-point stencil around the residual points) as
`.npz`; one evaluator (`crossframework/evaluate.py`) computes every metric: relative L2, max absolute error and the
RMS PDE residual from second-order central differences of the exported values (h = 1e-3 Burgers, 2e-3 Allen-Cahn,
1e-2 Poisson / Helmholtz; the same formula and step for every framework). A framework's own reported metric is never
used.

## Statistics

Median and min-max over seeds, and every raw value in `results/*.json`. Paired per-seed ratios where seeds are paired
(A1: the same initial weights; C: the same samples). With 3-5 seeds the exact sign test cannot reach p < 0.05
(the smallest p with 5 seeds is 0.0625), so differences are described with their spread, not called significant.

## Compute budget and cut order

At most about 8 CPU-hours of benchmark runs. If the protocol does not fit, cut in this order and record each cut:
1. Allen-Cahn in A1 (decided in advance, see above).
2. Track D.
3. A1 seeds, down to 3.

A competitor is never cut while PINNeAPPle is kept.

## Fairness rules

1. This protocol is committed before any result; deviations are listed in the report.
2. Same machine, same torch, float32, the same thread count per run; load recorded; timing comparisons from runs on
   a quiet machine.
3. One neutral evaluator from exported predictions.
4. No hyper-parameter search for anyone (A1 fixed, A2 the frameworks' own documented settings), no tuning on test
   data.
5. Median and min-max over seeds, all raw values kept.
6. Failures, divergence, NaN, crashes and "could not install" are results.
7. Competitor code follows its official docs and examples; no deliberately weak settings.
8. CPU-only caveat stated plainly.
9. The report states where each framework is better or worse with numbers and spread, and lists existing repository
   claims the measurements contradict (without rewriting them).
