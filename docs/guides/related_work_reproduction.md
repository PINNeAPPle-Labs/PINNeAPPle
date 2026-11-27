# Related work, and what has actually been reproduced

PINNeAPPle builds on published work by other groups. This page is the index of those lines and — the part that
matters — **which of their numbers this repository has reproduced, and where the number lives**. It is the companion of
issue [#403](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/403).

## What "reproduced" means on this page

A line is *reproduced* when a test in this repository computes the published quantity and compares it with the
reference at a declared tolerance, and the command to run it is written down. A line is *declared* when the path
exists (a loader, a module, a catalog entry) but the comparison has not been made here. Nothing is listed as
reproduced because a module exists.

## The table

| line of work | reference | what this repository has | status | where |
|---|---|---|---|---|
| **X-TFC** (Schiassi, De Florio, Furfaro, Mortari) | Leake & Mortari, *Deep Theory of Functional Connections*, [arXiv:1812.08625](https://arxiv.org/abs/1812.08625); Johnston & Mortari, [arXiv:2011.04700](https://arxiv.org/abs/2011.04700) | `pinneapple_neural/architectures/pinns/xtfc.py` (+ `gated_xtfc`, `xtfc_general`, `xtfc_inverse`, `xtfc_subdomain`) and `pinneapple_simulation/numerical_solvers/xtfc_ivp.py`, `xtfc_pde.py` | **reproduced** for the heat equation (below); the parametric-DE, epidemiological-discovery and BGK cases of the papers are **declared** | `tests/test_xtfc_pde.py`, `tests/test_xtfc_ivp.py`… |
| **Ashton's automotive datasets** (AWS) | AhmedML [arXiv:2407.20801](https://arxiv.org/abs/2407.20801); DrivAerML [arXiv:2408.11969](https://arxiv.org/abs/2408.11969); WindsorML [arXiv:2407.19320](https://arxiv.org/abs/2407.19320) | catalog entries with licence, size and paper in `pinneapple_catalog/resources.py`; the Ahmed body geometry for the library's own CFD | **declared** — one loader and one benchmark across the three is #407, still open | — |
| **NVIDIA PhysicsNeMo parity** | [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) | parity work tracked in #228, #230, #232; the backend seam is #75; CANTO #229; OpenUSD export #85 | **declared** — the cross-framework baselines are #409, still open | — |
| **NVIDIA Warp kernels** | [NVIDIA Warp](https://github.com/NVIDIA/warp) | the black-hole hydro has a PyTorch reference and a Numba mirror (`pinneapple_physics/blackhole/hydro_numba.py`) | **declared** — the fused-Warp backend is #410, still open | — |
| **NVIDIA Omniverse / Cosmos / Isaac** | [Omniverse](https://www.nvidia.com/en-us/omniverse/), [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | Twin3D exports OpenUSD (#85) and has a viewer that needs no Omniverse (#183) | **declared** — the live link and the Cosmos plausibility benchmark are #411, still open | — |
| **Earth-2 / Earth2Studio interop** | [Earth2Studio](https://github.com/NVIDIA/earth2studio) | the weather trust horizon work (#322, #323) | **declared** — the interop issue is open | — |

## The one reproduced line, with the numbers

X-TFC against the heat equation, whose solution on `x ∈ [0, 1]`, `t ∈ [0, 1]` is `u(x, t) = exp(-alpha pi^2 t) sin(pi x)`
for the module's default initial and boundary conditions:

```bash
PYTHONPATH=. python -m pytest tests/test_xtfc_pde.py -q
```

Measured with `n_basis = 40`, `n_collocation_x = n_collocation_t = 20`, `seed = 1`, `alpha = 1`, maximum absolute error
against the closed form over 60 points in `x`:

| `t` | max abs error |
|---|---|
| 0,00 | 0 (exact by construction) |
| 0,25 | 5,75e-4 |
| 0,50 | 5,62e-4 |
| 1,00 | 3,05e-4 |

The initial and boundary conditions are satisfied **identically** — the constrained expression is built that way, not
fitted — so the initial-condition error is zero to machine precision for any seed, which the test asserts at `1e-9`
over three seeds. That is the property that distinguishes X-TFC from a penalty-based PINN, and it is worth stating
with the number.

Two limits are recorded rather than smoothed over: the test file's own note explains that Burgers at the standard
`nu = 0.01/pi` has an interior layer this fixed-basis Gauss-Newton fit does not resolve, so the nonlinear path is
validated at `nu = 0,1` instead, where the flow is still genuinely nonlinear but has no layer. The papers' sharper
cases are the declared work, not a passed test.

## Credit

Every line above is other people's work. The modules here re-implement published methods and are checked against
published answers; where the comparison has not been made, this page says so instead of saying "parity". Authors and
preprints are cited at the top of each module and in the table above, and the datasets keep their own licences
(CC-BY-SA for the Ashton sets) — see `pinneapple_catalog/resources.py` for the entry each one carries.
