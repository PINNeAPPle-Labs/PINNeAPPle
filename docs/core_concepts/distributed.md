# Distributed physics: sweeps, partitioned meshes and decomposed training

`pp.distributed` (`pinneapple_core.distributed`) runs physics workloads in parallel, and not only training: a
parameter sweep of simulations, an ensemble of runs, a mesh operator applied part by part, and a PINN trained by
domain decomposition. It is numpy plus, for the last one, PyTorch.

| Function | What it runs |
|---|---|
| `map(fn, items)` | `[fn(x) for x in items]` on a worker pool, in input order, with per-task errors collected or raised |
| `sweep(fn, grid)` | `fn(**params)` over the Cartesian product of a grid (or an explicit list of dicts); a failure is recorded in its slot, never raised |
| `ensemble(fn, n, seed=0)` | `n` independent runs `fn(seed=s_i)` with seeds spawned from `numpy.random.SeedSequence` |
| `partition_mesh(mesh, n_parts)` | the mesh cut into parts with one halo layer, and which nodes and cells each part owns |
| `map_mesh(fn, mesh, values, n_parts)` | a nodal operator on every part, gathered back into one global array |
| `decompose_box(lo, hi, parts, overlap)` | overlapping subdomain boxes for a Schwarz iteration |
| `train_decomposed(...)` | a PINN trained by additive Schwarz: the subdomains of one iteration train concurrently |

## Workers

`backend="process"` uses joblib's loky pool when joblib is installed (`pip install "pinneapple[distributed]"`), and
`concurrent.futures.ProcessPoolExecutor` otherwise. The difference that matters in use:

- with loky, **closures and lambdas** can be sent to workers;
- without it, the worker needs an **importable function**, so a lambda fails at pickle time.

`ensemble` and `map_mesh` work either way: they hand the workers module-level functions rather than closures. The
tests cover both paths, and skip the closure ones when joblib is absent. `backend="thread"` suits numpy and torch
code that releases the GIL; `backend="serial"` runs in the caller, which is what to use to debug a failing task.

The worker count is capped at the number of tasks, so `sweep(fn, grid_of_64, workers=64)` on a 4-core laptop starts 64
processes only if there are 64 configurations — and each process pays its own import cost, which for torch is a couple
of seconds. For a sweep of short simulations, fewer workers is faster.

## Sweeps and ensembles

`sweep` returns a `SweepResult`: `records` in grid order, each with `params`, `result`, `error` and `time_s`;
`ok`, `failed`, `results()`, `table()`, and `best(key, minimize=True)` over the configurations that succeeded. A
configuration that raises does not abort the sweep — that is the point of a sweep, since a parameter region that
diverges is a result.

An empty grid raises `ValueError` rather than running one configuration with no parameters.

`ensemble` spawns statistically independent seeds from one `seed`, so the runs are reproducible whatever the number of
workers: the seed list depends on the seed, not on the schedule.

## Partitioned meshes

`partition_mesh` cuts by recursive coordinate bisection of the cell centres, splitting along the widest axis and giving
each part a share of the cells proportional to the number of parts it must produce. Every cell that shares a node with
an owned cell is added as one halo layer, so a nodal operator that needs the cells around a node — gradient recovery,
an integral assembled per node — is exact on the owned nodes of its part.

The shares are proportional rather than halves because halving balances only powers of two: five parts of a 252-cell
mesh come out as 63, 63, 63, 32, 31 cells, with half the work on some parts. Proportional bisection gives 50, 51, 50,
50, 51.

`map_mesh` returns an array of shape `(N, ...)`: each row written by the part that owns that node. The result is
compared against the global operator in the tests.

## Domain decomposition (additive Schwarz)

`train_decomposed(residual, boundary, make_model, lo, hi, ...)` cuts a box into strips along one axis, each extended by
`overlap` into its neighbours, and trains one network per subdomain:

- inside, the PDE residual `residual(model, x)`, differentiated with respect to `x`;
- on the faces that lie on the true boundary, `boundary(X)` as a target;
- on the interface faces, the neighbour's prediction from the **previous** iteration.

Within an iteration the subdomains are independent, so they train concurrently; the interface values are exchanged
between iterations, which is what makes it additive Schwarz. With `reference` the history records the relative L2 error
after every iteration.

Measured on `-u'' = pi^2 sin(pi x)`, `u(0) = u(1) = 0`: two subdomains, overlap 0.4, 8 iterations, 400 Adam steps each,
reach a relative L2 error under 2% in about 3 s on a laptop. Four subdomains converge more slowly at the same settings
— information moves one subdomain per iteration, the known weakness of Schwarz without a coarse space.

Data-parallel gradient averaging across devices (`torch.distributed`) is not part of this module; it is what
`pp.distributed` deliberately leaves to the training backend.
