- **One assimilation object over the two analysis kernels, and a model trained here inside both (#346)**:
  `pinneapple_analysis.data_assimilation.assimilation.Assimilation(model, background, B, observations, n_steps)`
  holds the problem -- the model step (any differentiable callable, `(..., n) -> (..., n)`, so the same function
  serves one state in 4D-Var and `n_ens` in the ensemble), the background with its covariance, the observation
  records (`step`, `y`, `H`, `R` with `H` any differentiable callable, a trained network included) -- and
  `analyse("4dvar" | "enkf", **options)` runs one named kernel on it: 4D-Var through autograd, or the delivered
  ensemble Kalman analysis with localisation and inflation (`cyclic_localisation` builds the Gaspari-Cohn weights
  for a state on a ring). The method has to be named because the two are different estimators; the report is the
  same shape, with the RMSE against a truth when one is given (`{"start", "end", "window"}`), the ensemble spread
  and the final cost. The kernels stay separate algorithms: the ensemble filter re-anchors at every observation,
  4D-Var fits the whole window.
  `data_assimilation.neural` trains the surrogate with which the neural half is now proved: a periodic convolution
  (`PeriodicTendency`) fitted to the tendency of Lorenz-96 and integrated by the RK4 step of the truth, and
  `neural_twin_experiment` runs the two kernels with the true step and with the network over the same observations
  and background. Measured over three seeds (`docs/dev/neural_assimilation.json`,
  `python -m pinneapple_analysis.data_assimilation.neural --seeds 0,1,2 --train-steps 300,1500`): the surrogate's
  held-out one-step error is 0.07 (state std 3.64) and its 20-step free-run error 1.27; the ensemble filter built
  on it matches the true-model filter (analysis RMSE 0.64 against 0.64, 0.84-1.14x per seed) and both are 5.3x
  better than no assimilation (3.39), while 4D-Var inherits the model's error (0.83 against 0.42, 0.88-4.10x per
  seed) though it still beats no assimilation by 2.6-6.1x; a network with a fifth of the training (one-step 0.39)
  gives 2.30 in both kernels, 3.6x and 5.5x worse than the true-model path -- the negative result, asserted in
  `tests/pinneapple_analysis/test_neural_assimilation.py` together with the two positive ones and a trained
  observation operator through which the adjoint is checked.
