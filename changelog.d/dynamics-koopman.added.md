- `pinneapple_analysis.dynamics.koopman`: Koopman operator and spectral analysis, guide
  `docs/guides/dynamics_koopman.md`.
  - `DMD` (exact DMD, TLS debiasing, rank by Gavish-Donoho, an energy fraction or fixed) and `EDMD` on an
    `ObservableDictionary` (identity, monomials, RBF, Fourier, custom, neural observables from a PyTorch module).
  - Both return a `KoopmanOperator` with `KoopmanEigenvalues` (discrete and continuous), `KoopmanEigenfunctions`,
    `DynamicMode`s, ResDMD residuals, stability, prediction and the generator.
  - Generator EDMD from derivative data.
  - `KoopmanReconstruction` and `SpectralAnalyzer`: spectrum, noise sensitivity, order selection, and comparison
    with a nonlinear state-space regression.
  - Inputs are trajectories, ensembles and PDE snapshots.
  - This `KoopmanOperator` is a fitted NumPy operator. The trainable
    `pinneapple_neural.architectures.rom.koopman.KoopmanOperator` is unchanged.
  - Benchmarks: `python -m pinneapple_analysis.dynamics.benchmarks koopman`.
