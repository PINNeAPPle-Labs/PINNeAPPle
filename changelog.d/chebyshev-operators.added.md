- Added three neural operators for parametric PDE surrogates: `ChebyshevPolynomialNeuralOperator`
  (mesh-free Chebyshev expansion with FiLM-style conditioning on the scenario parameters,
  arXiv:2602.01737), and `ChebyshevBranchOperator` / `HadamardDeepONet`, the spatially resolved
  branch/trunk decomposition of arXiv:2511.02962 where the branch emits the whole field for every
  instant and a trunk reweights it in time instead of contracting it over a reduced basis. All three
  are in the neural-operator registry under `cpno` / `chebyshev_polynomial_neural_operator`,
  `chebyshev_branch_trunk` and `hadamard_deeponet`, with `OperatorOutput` and the same
  `forward(..., y_true=, return_loss=)` convention as the existing operators. The published
  parameter counts of the reference configurations are asserted in
  `tests/test_chebyshev_operators.py`, which is what ties the implementation to the papers it cites.
