# Large-eddy simulation: the three subgrid closures

An LES resolves the large scales of a turbulent flow and models the small ones. In a PINN that split is written
into the residual: the network represents the *filtered* velocity and pressure, and a subgrid closure supplies the
eddy viscosity `nu_sgs` that stands in for the stresses of the scales that are not resolved.

This guide covers the three algebraic closures that ship with `pinneapple_physics`: Smagorinsky, dynamic
Smagorinsky and WALE. They share one residual assembly, so the only thing that differs between them — on the same
flow, at the same points — is the closure.

```python
import torch
from pinneapple_physics.pde_environment import (
    SmagorinskyResiduals, DynamicSmagorinskyResiduals, WALEResiduals,
)
```

## 1. What the residual is

All three solve the filtered incompressible Navier-Stokes equations with the subgrid stress folded into an
effective viscosity (`p` is pressure over density):

```
du_i/dt + u_j du_i/dx_j = -dp/dx_i + d/dx_j[(nu + nu_sgs)(du_i/dx_j + du_j/dx_i)] + f_i
du_i/dx_i = 0
```

The model maps `(x, y, z, t) -> (u, v, w, p)`, so the network has four outputs on four inputs, and the residual
comes back as a dict:

```python
residuals = WALEResiduals(nu=1e-5, cw=0.325, delta=0.01)
r = residuals(model, x_col)                    # x_col: (N, 4) = (x, y, z, t)
loss = sum(v.pow(2).mean() for k, v in r.items() if k != "nu_sgs")
```

`nu_sgs` is a diagnostic, not a residual: it is what the closure produced on those points, and it is the first
thing to look at when an LES PINN will not converge.

`delta` is the filter width: a constant for a uniform mesh, or a callable `delta_fn(x) -> (N, 1)` for a real
one (e.g. `cubeRootVol` per-cell sizes read by
`pinneapple_simulation.external_solvers.openfoam.mesh_reader`).

## 2. Choosing the closure

| closure | `nu_sgs` | constants | when |
|---|---|---|---|
| `SmagorinskyResiduals` | `(Cs Δ)^2 \|S\|` | `Cs = 0.17` | first LES of an attached or mildly separated flow; same `Cs` the LBM solvers here use |
| `DynamicSmagorinskyResiduals` | `C Δ^2 \|S\|`, `C` computed from the field | `test_ratio = 2` | the default in production codes: it switches itself off where the flow is resolved or laminar |
| `WALEResiduals` | `(Cw Δ)^2 (Sd:Sd)^{3/2} / ((S:S)^{5/2} + (Sd:Sd)^{5/4})` | `Cw = 0.325` | wall-bounded flow: correct `y^3` near-wall behaviour without a damping function |

`|S| = sqrt(2 S_ij S_ij)` is the strain-rate magnitude and `Sd` the traceless symmetric part of the squared
velocity-gradient tensor.

The three are also reachable by name, next to the RANS closures:

```python
from pinneapple_physics import get_rans_preset

get_rans_preset("smagorinsky", nu=1e-5, cs=0.17, delta=0.01)
get_rans_preset("dynamic-smagorinsky", nu=1e-5, delta=0.01)   # alias: "dsmag"
get_rans_preset("wale", nu=1e-5, cw=0.325, delta=0.01)
```

## 3. The constant model, and the two things it gets wrong

The constant-coefficient model does not vanish at a wall, where it must. Pass a damping factor if `y+` is
available from the geometry:

```python
import torch
from pinneapple_physics.pde_environment import SmagorinskyResiduals, van_driest

NU, U_TAU, WALL_Y = 1e-5, 0.05, 0.0

def y_plus(x):        # distance to the wall, in wall units -- problem data, not closure data
    wall_distance = (x[:, 1:2] - WALL_Y).clamp_min(0.0)
    return wall_distance * U_TAU / NU

residuals = SmagorinskyResiduals(
    nu=NU, cs=0.17, delta=0.01,
    damping=lambda x: van_driest(y_plus(x)),           # D = (1 - exp(-y+/25))^2
)
```

The damping is off by default because `y+` is problem data (it needs the wall distance *and* the friction
velocity), not closure data — the library will not invent it.

## 4. The dynamic model on scattered points

The dynamic procedure needs two filters: the grid filter `Δ` and a test filter `Δ̂ = test_ratio · Δ`. A PINN has
no grid, so the test filter is built from the collocation point cloud itself: the `k` nearest neighbours of every
point, Gaussian-weighted and normalised (`filter_stencil`, `apply_stencil`). It is a genuine filter — weights sum
to one, and it reproduces a linear field exactly on a symmetric stencil — and it stays differentiable, because it
averages values the network has already evaluated at every point of the batch.

```python
from pinneapple_physics.pde_environment import DynamicSmagorinskyResiduals

residuals = DynamicSmagorinskyResiduals(
    nu=1e-5, delta=0.01,
    test_ratio=2.0,        # Δ̂ = 2Δ, the test filter width
    neighbors=8,           # stencil size on the point cloud
    averaging="batch",     # <·> over the batch (homogeneous directions)
    clip=True,             # Lilly: C >= 0, no anti-diffusive viscosity
)
r = residuals(model, x_col)
residuals.last_germano["C"]     # the coefficient actually used
```

Two approximations come with it, and they are the model's, not a shortcut:

- the test-level strain is taken from the filtered velocity gradient (`S_hat ~= strain(filter(g))`), the usual
  commutation assumption — exact for a uniform filter in a homogeneous direction;
- `<·>` is a spatial average. `averaging="batch"` assumes the batch spans homogeneous directions, which is what
  the dynamic procedure requires and what a plane-averaged channel LES does. `averaging="none"` gives a pointwise
  coefficient instead: admissible, noisier, and the main reason to keep `clip=True`.

## 5. What is checked

The closures are in `pinneapple_physics/pde_environment/turbulence_presets.py`; the claims above are checked in
`tests/pinneapple_physics/test_les_closures.py`:

- the full residual of each closure against an independently hand-derived SymPy closed form (the same
  cross-implementation check the RANS closures get);
- `nu_sgs` for Smagorinsky on a linear field, where it is exactly `(Cs Δ)^2 sqrt(2S:S)`;
- solid-body rotation, which has zero strain *and* is an exact solution of the equations with a balancing
  centripetal pressure: the closure must return `nu_sgs = 0` and the whole residual must vanish;
- the Germano–Lilly normal equation, the homogeneity of `C` under a velocity rescaling, the growth of `nu_sgs`
  with the filter width, filter partition of unity and linear reproduction, and gradients flowing through the
  filter to the network.

The a-priori benchmark is checked in `tests/pinneapple_physics/test_sgs3d_benchmark.py`, where the measurement
matters as much as the closures: spectral derivatives against analytic ones, a field whose product stays inside
the cut-off (true stress exactly zero), the filter against an analytic truncation, the metrics against a perfect,
an inverted and a doubled model, and the ideal viscosity against a known coefficient.

## 6. Which one wins on a real flow (a priori)

A-priori scoring answers the question before any simulation: given a resolved field from a flow whose exact
subfilter stress is known, how close is each closure's stress to it? `pinneapple_physics.multiscale.sgs3d` runs
that on a Taylor-Green DNS -- sharp spectral filter, aliasing-free products, the real closures, and a learned
closure trained on several snapshots and scored on a later one it never saw:

```python
from pinneapple_physics.multiscale.sgs3d import run_benchmark

report = run_benchmark(n=64, re=1600, t_train=(4, 5, 6, 7), t_test=9.0)
for name, m in report["table"].items():
    print(name, m["corr"], m["mag_ratio"], m["transfer_corr"])
```

or from the shell, `python -m pinneapple_physics.multiscale.sgs3d --n 64 --re 1600`. Read `corr` and `mag_ratio`
together (either one alone is easy to fake), and `transfer_corr` last: it is the stress component projected on
the strain, i.e. the energy the filter actually moved.

Two things the numbers will show, and neither is a bug:

- an **eddy-viscosity shape is a limitation, not a coefficient error**. Smagorinsky and WALE correlate only
  weakly with the true stress per component, and even at the least-squares `Cs` their magnitude stays well under
  the true one: what the filter removes is not proportional to `S_ij`. `nu_ideal_mean` and
  `nu_ideal_negative_fraction` in the report say where the difference is -- a large negative fraction means the
  truth backscatters and no positive-coefficient model can follow it. Note also that `corr` is a property of the
  closure's *shape* alone: two eddy-viscosity models with different coefficients (Smagorinsky and the dynamic
  one, when its coefficient is a single averaged number) score exactly the same `corr` and differ only in
  `mag_ratio`.
- the **learned closure is only as good as its training set**. Fitted to a single snapshot it scores 1.0 on that
  snapshot and at chance on the next, which is memorisation, not a closure; `run_benchmark` therefore trains it on
  the `t_train` snapshots and reports the held-out one.

The reference run (DNS 64³, coarse 32³, `kc = 32/3`, `Δ = 0.295`, Re = 1600, trained on t = 4–7, held out at
t = 9, `steps = 3000`) gives:

```
closure                   corr     mag  transfer    nu_ideal
GradientModel            0.510   0.399     0.340   1.485e-03
Smagorinsky              0.089   0.252     0.375   1.485e-03
DynamicSmagorinsky       0.089   0.123     0.375   1.485e-03
WALE                     0.001   0.177     0.009   1.485e-03
Learned                  0.086   0.695    -0.046   1.485e-03
```

Read it as: `GradientModel` is the control -- it uses the same resolved gradient as the others but is not forced
into the shape `tau ~ S`, and it correlates far better. The gap between it and the eddy-viscosity closures is the
price of the eddy-viscosity assumption itself, not of any coefficient. WALE is near zero here: it is built for
walls, and a smooth decaying vortex is not where it earns its keep. The learned closure recovers the magnitude
the algebraic ones miss, but not the transfer. Numbers this low are what the literature reports for a-priori
correlations of eddy-viscosity closures; a benchmark that produced 0.9 for them would be measuring itself.

This is an **a-priori** test. A closure that scores well here can still be unstable in an LES, because a-priori
correlation says nothing about how the error feeds back into the resolved scales. At `n = 48` the same run gives
markedly poorer correlations throughout: a 48³ DNS of Re = 1600 is under-resolved by t = 9, so the band the filter
keeps is partly numerical.

## 7. What is not here

- **Wall modelling (WMLES) and detached-eddy simulation (DES)** — not implemented. For a wall-bounded LES at a
  Reynolds number a resolved near-wall layer cannot afford, neither of these closures is enough on its own.
- **A-posteriori testing in 3-D.** The 2-D spectral case has it (`aposteriori_test` in
  `pinneapple_physics.multiscale`); running an LES with each closure in 3-D and comparing the resolved statistics
  against a filtered DNS is the missing half of the benchmark above.
- **Learning the subgrid stress in 3-D beyond the a-priori fit.** `learn_sgs_stress` in `sgs3d` learns from
  filtered DNS pairs, but it is trained a priori and never run through a solver. For 2-D spectral turbulence a
  closure trained *through* the solver exists (`train_closure_aposteriori`).
- **OpenFOAM LES cases.** The bridge in `pinneapple_simulation.external_solvers.openfoam` runs cases and reads
  fields; it does not choose or configure an LES model for you.
