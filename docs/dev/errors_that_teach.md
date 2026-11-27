# Errors that teach (#46, roadmap A5)

Every failure names what broke and what to do next. The messages live in the code (
`pinneapple_core/errors.py`), and each one is raised through a class that also inherits the standard
exception a caller already catches, so ``except ValueError`` / ``except ImportError`` keep working.

This file is generated from `pinneapple_core.errors.AUDITED`; run
`python scripts/build_error_audit.py` after changing the table. `tests/test_errors_that_teach.py`
triggers every entry below and fails if the error comes back without its fix.

30 audited errors.

| # | id | site | the failure | was | now says |
|---|---|---|---|---|---|
| 1 | `cost-repeats` | `pinneapple_analysis.cost.profiler:measure` | measure() called with repeats < 1 | repeats must be >= 1 | pass repeats=5 (the default) or any integer >= 1 |
| 2 | `cost-sizes` | `pinneapple_analysis.cost.profiler:scaling_study` | scaling_study() with fewer than three distinct sizes | need at least 3 distinct sizes | pass at least 3 distinct sizes spanning a factor of 10, e.g. [10**3, 10**4, 10**5] |
| 3 | `cost-complexity` | `pinneapple_analysis.cost.profiler:expected_exponent` | an unknown complexity string | unknown complexity {declared!r}; use a number or one of [...] | use a number or one of '1', 'log n', 'n', 'n log n', 'n^2', 'n^3' |
| 4 | `cost-budget` | `pinneapple_analysis.cost.profiler:Budget.validate` | an operation over its budget | exceeds ... (no next step) | raise the limit in Budget(...) or reduce the problem size; the violations above name the quantity |
| 5 | `budget-empty` | `pinneapple_analysis.cost.equal_budget:ComputeBudget.__post_init__` | a ComputeBudget with no limit set | a ComputeBudget needs at least one of wall_s, flops, steps | set at least one of wall_s=..., flops=..., steps=... |
| 6 | `budget-nonpositive` | `pinneapple_analysis.cost.equal_budget:ComputeBudget.__post_init__` | a negative or zero budget limit | {name} must be positive, got {v} | use a positive limit; omit the field (None) when that quantity should not be limited |
| 7 | `l2-shape` | `pinneapple_analysis.cost.equal_budget:relative_l2` | pred and true of different shapes | shape mismatch {p.shape} vs {t.shape} | reshape or resample one to the other; relative_l2 compares point by point |
| 8 | `per-query` | `pinneapple_analysis.cost.equal_budget:AmortizedCost.per_query` | per_query(0) (or any n < 1) | n_queries must be >= 1 | pass n_queries >= 1: the cost per query is undefined for zero queries |
| 9 | `method-unknown` | `pinneapple_catalog.methods:get_method` | an id that is not in the catalog | unknown method 'X' | call pinneapple_catalog.methods.list_methods() and use one of those ids |
| 10 | `problem-unknown` | `pinneapple_analysis.benchmark.registry:get_problem` | a benchmark problem that is not registered | unknown problem 'X'; registered: [...] | call pinneapple_analysis.benchmark.problems() and use one of those ids |
| 11 | `bench-method-unknown` | `pinneapple_analysis.benchmark.registry:get_method` | a benchmark method that is not registered | unknown method 'X'; registered: [...] | call pinneapple_analysis.benchmark.methods() and use one of those ids |
| 12 | `bench-space` | `pinneapple_analysis.benchmark.registry:Method.space` | asking a method for the hyper-parameter grid it does not declare | X has no hyper-parameter space for Y | train this method without a search, or add space_<method id>() to the problem |
| 13 | `bench-train` | `pinneapple_analysis.benchmark.registry:Method.train` | training a method on a problem that has no implementation of it | X cannot train on Y | pick a method the problem implements (see problem.pairs()) or add train_<method id>() |
| 14 | `bench-config-budget` | `pinneapple_analysis.benchmark.config:BenchmarkConfig` | a benchmark config without a budget | budget is required (wall_s, flops or steps) | pass budget={'wall_s': 60} (or flops=... / steps=...) |
| 15 | `bench-config-fields` | `pinneapple_analysis.benchmark.config:BenchmarkConfig.from_dict` | an unknown key in the benchmark config | unknown benchmark config fields: [...] | drop or rename the key: the error lists the known ones |
| 16 | `solver-unknown` | `pinneapple_simulation.numerical_solvers.registry:SolverRegistry.spec` | an unregistered solver name | Unknown solver 'X'. Available: [...] | call SolverRegistry.list() and use one of those names |
| 17 | `solver-ctor` | `pinneapple_simulation.numerical_solvers.registry:SolverRegistry.build` | keyword arguments the solver's constructor does not take | SolverRegistry.build('X'): ... The constructor is ... | pass the constructor's own arguments: the error prints the signature |
| 18 | `trust-not-fitted` | `pinneapple_analysis.trust.trust_gate:TrustGate.ood_p_values` | scoring OOD before fitting the reference distribution | TrustGate.fit()/.fit_from_bounds() must be called before OOD scoring. | call fit(training_coords) or fit_from_bounds(domain_bounds) first |
| 19 | `trust-no-bounds` | `pinneapple_analysis.trust.trust_gate:TrustGate.coord_bounds_from_checkpoint` | a checkpoint that records no domain bounds | checkpoint metadata has no 'domain_bounds' entry. | re-save the checkpoint with metadata['domain_bounds'], or build the gate with fit(training_coords) |
| 20 | `trust-ood-level` | `pinneapple_analysis.trust.trust_gate:TrustGate.__init__` | an out-of-range ood_level | ood_level must be in (0, 1); got {v} | pass ood_level between 0 and 1 exclusive, e.g. 0.01 for the 99 % region |
| 21 | `unit-convert-from` | `pinneapple_core.physics_tensor:PhysicsTensor.convert` | converting a tensor that carries no unit | 'name' has no unit to convert from | build it with pp.tensor(..., units='m/s'), or drop convert() |
| 22 | `unit-convert-across` | `pinneapple_core.physics_tensor:PhysicsTensor.convert` | converting between two different dimensions | cannot convert 'm/s' to 'Pa': the dimensions differ | convert only within a dimension (m -> km); a different dimension is a different quantity |
| 23 | `unit-product` | `pinneapple_core.physics_tensor:_no_unit_product` | multiplying two dimensioned tensors | cannot apply 'mul' to ...: a unit algebra would be needed | drop the labels with .without_labels() first, then label the product with pp.tensor(..., units=...) |
| 24 | `backend-unknown` | `pinneapple_analysis.dynamics._backends:require` | asking for a backend name the toolkit does not have | unknown backend {backend!r}; one of [...] | use one of the four it knows ('torch', 'jax', 'sympy', 'scipy'); capabilities() lists them |
| 25 | `missing-pyarrow` | `pinneapple_data.contracts.tables:_pyarrow` | writing or reading a Parquet table without pyarrow | Parquet needs pyarrow: pip install pyarrow | pip install pyarrow |
| 26 | `missing-h5py` | `pinneapple_data.contracts.tables:_h5py` | writing or reading an HDF5 table without h5py | HDF5 needs h5py: pip install h5py | pip install h5py |
| 27 | `missing-hub` | `pinneapple_catalog.resources:fetch` | fetching a hosted resource without the hub extra | fetch requires the "hub" extra: pip install "pinneapple[hub]" | pip install "pinneapple[hub]"  (installs huggingface_hub) |
| 28 | `missing-onnx` | `pinneapple_analysis.surrogate.model_source:onnx_function` | opening an ONNX model without onnxruntime | an ONNX model needs onnxruntime ... | pip install onnxruntime (or the [onnx] extra); TorchScript and a plain callable need nothing |
| 29 | `missing-retrieval` | `pinneapple_analysis.retrieval.index:_require_sentence_transformers` | building a retrieval index without the retrieval extra | requires the 'retrieval' extra ... | pip install "pinneapple[retrieval]"  (installs sentence-transformers) |
| 30 | `backend-missing` | `pinneapple_analysis.dynamics._backends:require` | asking a dynamics method for a backend that is not installed | BackendUnavailable: {backend} is not installed. {cmd} | install it with the command the error prints, or pick an installed one from capabilities() |

## Not audited, on purpose

- internal invariant failures ("this should never happen"): the user has no action, so there is nothing to
  teach; they stay plain asserts inside private functions.
- numerical failures already handled by a check that reports instead of raising (the benchmark ``Check``
  objects, ``pp.doctor()``): those carry their own ``fix`` field and are not errors.
