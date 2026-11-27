# Official benchmark cases

Three problems with a closed-form solution, scored by one protocol: the same seeds, the same training budget,
the same evaluation points and the same metrics. The table below is generated from
`benchmarks/cases/results.json` by `python scripts/benchmark_cases.py --docs` -- do not edit it by hand.

## The protocol

* **seeds**: 0, 1, 2 -- the spread across seeds is reported as `[min, max]`, never mixed with the point interval;
* **budget**: 300 epochs, 512 collocation points, network 32 x 3 -- identical for every seed and every case;
* **metrics**: relative_l2, rmse, max_abs, computed on the fixed evaluation grid of each case;
* **interval**: bootstrap over the evaluation points, 95%, 2000 resamples (seed 12345) -- it says how much the metric depends on the points scored, not on the seed;
* **in-domain** is the window the network was trained on; **extrapolation** is the rest of the domain. A method that only publishes the first number is hiding the second;
* **cost** is the wall time of one run on one CPU and is informational: it is a property of the machine, so a
  reproduction compares the metrics and never the time.

## The published table

| case | in-domain relative L2 | extrapolation relative L2 | rmse (in-domain) | max abs (in-domain) | wall time per seed |
|---|---|---|---|---|---|
| 01 heat_1d | 5.12e-03 [1.99e-03, 0.011] | 0.582 [0.415, 0.796] | 3.18e-03 | 0.010 | 6.5 s |
| 02 heat_2d | 0.022 [0.021, 0.023] | 0.814 [0.518, 1.150] | 8.87e-03 | 0.035 | 5.5 s |
| 03 burgers_1d | 0.157 [0.154, 0.159] | 0.790 [0.504, 1.071] | 0.106 | 0.601 | 3.8 s |

## The cases

### Case 01 -- heat_1d

* equation: `alpha T_xx = T_t, alpha = 0.05, T(x, 0) = sin(pi x), T = 0 at x = 0 and x = 1`
* reference: closed form: exp(-alpha pi^2 t) sin(pi x)
* train: t in [0, 0.5] (t <= 0.5)
* evaluation: 41 x 41 grid on x, t in [0, 1]
* budget: 300 epochs, 512 collocation points

### Case 02 -- heat_2d

* equation: `alpha (T_xx + T_yy) = T_t, alpha = 0.05, T = sin(pi x) sin(pi y) at t = 0, T = 0 on the walls`
* reference: closed form: exp(-2 alpha pi^2 t) sin(pi x) sin(pi y)
* train: t in [0, 0.4] (t <= 0.4)
* evaluation: 21^3 grid on x, y, t in [0, 1]
* budget: 300 epochs, 512 collocation points

### Case 03 -- burgers_1d

* equation: `u_t + u u_x = nu u_xx, nu = 0.01, u(x, 0) = -sin(pi x), u = 0 at x = -1 and x = 1`
* reference: Cole-Hopf by Gauss-Hermite quadrature (pinneapple_physics.closed_form.burgers)
* train: t in [0, 0.5] (t <= 0.5)
* evaluation: 129 x 41 grid on x, t in [0, 1]
* budget: 300 epochs, 512 collocation points

## Reproducing it

```bash
python scripts/benchmark_cases.py --run      # trains every case and rewrites the results and this page
python scripts/benchmark_cases.py --reproduce  # trains again and compares against the table above
python scripts/benchmark_cases.py --check      # verifies the declaration and this page, no training
```

A reproduction matches when every metric is within 25% of the published value; the runner says which cell did not.
