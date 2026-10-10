# Reduced-order models

A reduced-order model (ROM) replaces a solver by a few numbers per state: the coefficients of a small basis
(POD) and a cheap rule for how they change, either in time (DMD, Operator Inference, Koopman) or with the design
parameters (POD with regression). PINNeAPPle has both families in `pinneapple_neural.architectures.rom`. The lab
experiment `rom_study` checks them on the library's own solvers against a baseline.

| class | what it learns | use it for |
|---|---|---|
| `POD` | an orthonormal basis of the snapshots (SVD), energy per mode, incremental updates | compression; the basis of every other ROM |
| `DynamicModeDecomposition` | a linear map between consecutive snapshots: eigenvalues (frequency, growth), modes, rollout | periodic and weakly nonlinear dynamics (wakes, oscillations); spectral analysis |
| `OperatorInference` | quadratic latent dynamics `a' = A a + H (a ⊗ a) + b` by ridge regression, discrete or continuous | dynamics with quadratic nonlinearities (incompressible flow), longer forecasts than DMD |
| `HAVOK`, `SINDy` | delay-embedded linear model with forcing; sparse equations | chaotic signals; interpretable models |
| `KoopmanAutoencoder`, `NeuralROM`, `ROMHybrid`, `DeepUQROM` | nonlinear latent spaces and dynamics (neural) | strongly nonlinear dynamics where linear subspaces need too many modes |
| `ParametricPOD` | POD of one snapshot per design + regression of the coefficients on the parameters (RBF, Gaussian process with uncertainty, linear, nearest) | parameter → full field surrogates of any solver whose fields share one mesh |

## Time-dependent: forecasting a wake

```python
import torch
from pinneapple_neural.architectures.rom import POD, DynamicModeDecomposition, OperatorInference

X = torch.tensor(snapshots, dtype=torch.float64)          # (T, D): one flattened field per time
pod = POD(r=20).fit(X[:n_train])
pod.explained_variance_ratio_                              # energy per mode
dmd = DynamicModeDecomposition(r=12).fit(X[:n_train][None])
dmd.frequencies(dt)                                        # frequencies of the eigenvalues
forecast = dmd.rollout(X[n_train - 1][None], steps=100)   # (1, 101, D)
a = pod.encode(X[:n_train])[:, :8]
oi = OperatorInference(r=8, use_quadratic=True).fit(a[None])
a_forecast = oi.rollout(a[-1][None], 100)
```

The operators keep the dtype of the data (float64 in, float64 out).

## Parametric: a full field for a new design without running the solver

```python
from pinneapple_neural.architectures.rom import ParametricPOD, latin_hypercube

P, names = latin_hypercube(80, {"H": (0.12, 0.3), "W": (0.06, 0.2), "angle": (0, 90)})
X = [solve(*p) for p in P]                                 # one solver run per design, same mesh
rom = ParametricPOD(r=16, energy=None, regressor="gpr", normalize=True).fit(features(P), X)
X_new, X_std = rom.predict(features(P_new), return_std=True)
rom.error(features(P_test), X_test)                        # relative L2 per design
rom.projection_error(X_test)                               # the floor any regressor can reach
```

Three things decide the accuracy, in this order:

1. Inputs where the response is smooth. A cantilever's stiffness goes as W H³: regress on `log H`, `log W`.
2. `normalize=True` when the response spans orders of magnitude. It separates the amplitude (log of the norm,
   regressed by RBF) from the shape (unit norm, POD).
3. Enough designs. Doubling from 40 to 80 in three parameters cut the median error on unseen designs by three to
   five times.

Always compare with the nearest training design (`regressor="nearest"`). A surrogate that does not clearly beat it
has not learned anything.

## Results in the lab (`python -m pinneapple_lab sweep rom_study -g case=cylinder_wake,beam_parametric`)

**`beam_parametric`**: 80 FEM solutions of a 3-D cantilever (`solid_fem`, C3D8I), 20 unseen designs inside the box.

| field | best ROM | median error | worst design | nearest design (baseline) |
|---|---|---|---|---|
| displacement | POD-GPR | 0.4 % | 4.3 % | 5.2 % |
| von Mises | POD-GPR + amplitude split | 1.7 % | 11 % | 6.4 % |

A prediction takes about 50 µs against about 1 s for the FEM solve. The worst designs sit near the corners of the
box, where the training designs are sparse. That is where to add designs (active learning on the GPR standard
deviation).

**`cylinder_wake`**: POD modes, the DMD shedding frequency against a probe spectrum, and DMD and Operator Inference
forecasts beyond the training window against persistence. The numbers are in the run's metrics.

## Limits

- Linear subspaces (POD) need many modes for moving fronts and shocks. Use the neural ROMs there, or
  shift the snapshots first.
- Parametric ROMs interpolate. Outside the training box they extrapolate without warning, apart from the GPR
  standard deviation.
- The snapshots must share a mesh (same numbering). Different meshes need interpolation onto a common grid first.
