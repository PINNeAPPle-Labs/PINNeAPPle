# Bayesian optimization: fewer reference simulations for the same answer

Roadmap item #102. A reference simulation is anything expensive — a CFD run, a finite-element solve, an experiment on
a rig — so the currency of a design loop is the number of calls, and the loop is judged on that.

```python
import pinneapple as pp

result = pp.BayesianOptimizer(cost_of, [("thickness", (0.5, 3.0)), ("angle", (-5, 15))],
                              n_initial=5, n_iter=20, batch_size=2, seed=0).run()
result.best_x, result.best_y      # the design to build and the cost it measured
result.evals_to_reach(target)     # how many calls it took, or None if it never got there
```

`cost_of` takes an `(n, d)` array of designs and returns one value per row — the same shape a solver sweep produces.
`constraints=[(callable, limit)]` takes the loop into feasibility-weighted territory: a point is feasible when
`callable(x) <= limit`, the constraint is modelled by a second Gaussian process on the worst violation, and a candidate
the surrogate expects to break it is not proposed at all.

## The measurement

| problema | orçamento | método | atingiu o alvo | chamadas até o alvo (mediana) | melhor valor (mediana) |
|---|---|---|---|---|---|
| branin | 24 | Bayes | 100% | 16 | 0.4205 |
| branin | 24 | aleatório | 20% | 6 | 1.915 |
| hartmann3 | 30 | Bayes | 100% | 12 | -3.851 |
| hartmann3 | 30 | aleatório | 60% | 14 | -3.614 |

Five seeds each, same budget, same seed per pair. Three columns because one is not enough: how often each method
reaches the target **within the budget**, how many calls it took when it did, and the best value it ended with. A
median of calls computed only over the runs that succeeded, while the others never arrived, would flatter the method
that failed most — on Branin the random sampler's "6 calls" is a single lucky seed out of five.

Reproduce with `python scripts/benchmark_bayesopt.py` (`--seeds`, `--budget`, `--markdown`).

## What it is built on

A Gaussian process with an anisotropic squared-exponential kernel, its length scale fitted by maximizing the log
marginal likelihood, on a **standardized output** — a surrogate on numbers of order 1e5 and one of order 1 need the
same prior. Three acquisition functions: expected improvement (default), probability of improvement and upper
confidence bound.

Three things it does deliberately, each of which was a wrong answer first:

- the length scales are bounded relative to the spread of each input: letting all hyperparameters float lets the
  optimiser collapse the kernel onto the initial points, which fits perfectly and predicts the mean everywhere;
- the incumbent is the best **feasible** value: an infeasible incumbent makes every improvement negative and the
  acquisition underflows to zero, after which the loop picks at random while looking like it is optimizing;
- a constrained loop returns a feasible best even if an infeasible point measured better.

## What it refuses

An empty or reversed bound, an acquisition it does not know, a constraint that is not `(callable, limit)`, an
objective that returns the wrong number of values, and an empty budget. A silent fallback to random sampling would
make the comparison that justifies the module meaningless.
