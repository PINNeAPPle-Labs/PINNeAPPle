# Numerical limits of the OpenFOAM connector: the decisions, and the case behind each

`buoyantSimpleFoam`, `buoyantPimpleFoam` and free-surface flow with k-omega SST each hit a real limit in earlier case
campaigns (#132). A limit recorded as "it failed once" is not something anyone can act on, so each one here has three
things: **what was measured**, **the minimal case that demonstrates it**, and **the decision** the connector makes
about it — enforce it, work around it, or declare it unsupported.

Everything below was measured on OpenFOAM v2606 (macOS arm64, native build) with
`tests/test_openfoam_limits.py`, which writes the cases and runs the solvers. The registry the pre-flight rules read
is `pinneapple_data.preflight.limits`:

```python
from pinneapple_data.preflight import decision, limits_for, unsupported

decision("interIsoFoam")        # 'accepted'
[lim.name for lim in limits_for("buoyantSimpleFoam")]
[lim.name for lim in unsupported()]          # what this library does not claim to model
```

## The three decisions

| solver | decision | what the connector does |
|---|---|---|
| `buoyantSimpleFoam` | accepted | refuses to pass a case without `constant/g` or `pRefCell`/`pRefPoint` (both are fatal stops), and warns when relaxation is switched off |
| `buoyantPimpleFoam` | accepted | refuses a case without a `rho` linear solver, warns when `correctPhi`/`pcorr` are missing (the run completes with a 12x larger first-step continuity error) |
| `interIsoFoam` + k-omega SST | **supported, and run for real** | refuses a free-surface case with a k-omega model until `wallDist`, an omega convection term and an omega linear solver exist |

## What each limit is, and what it costs

### `buoyantSimpleFoam`

| limit | measured | decision |
|---|---|---|
| no `constant/g` | `FOAM FATAL ERROR: cannot find file ".../constant/g"` | accepted, enforced |
| no pressure reference | `FOAM FATAL IO ERROR: Unable to set reference cell for field p / Please supply either pRefCell or pRefPoint` | accepted, enforced |
| no relaxation | p_rgh initial residual 0.388 after 300 iterations with factors 1.0, against 0.110 with 0.7 | accepted, reported |
| thermophysical package | v2606 selects `heRhoThermo` (a `hePsiThermo` case stops at start-up) and wants `compressible::alphatWallFunction` | worked around by the case writer |

The relaxation row is the interesting one: nothing crashes. The run reaches its end time and reports success, with a
pressure residual four times higher than the same case relaxed — which is why the report reads the residual history
and not the exit code.

### `buoyantPimpleFoam`

| limit | measured | decision |
|---|---|---|
| no `rho` linear solver | `FOAM FATAL IO ERROR: Entry 'rho' not found in dictionary "system/fvSolution/solvers"` | accepted, enforced |
| no start-up flux correction | first-step local continuity error `0.000129275` with `correctPhi yes` against `0.00161142` without — the same case, the same ten steps | accepted, warned |
| cold-start pressure field | a cavity started from `p = p_rgh = 1e5` with `U = 0` at Co ≈ 1 dies in the first PIMPLE iteration with `FOAM_SIGFPE`; the tutorial case, which sets `hRef` and a hydrostatic `p_rgh`, runs | accepted, documented |

The second row is the one that costs a campaign real time: a run that completes, reports nothing, and is wrong from
its first step onwards. The pre-flight now warns, and the campaign's accounting (see the campaign runner) is what
catches it downstream.

### `interIsoFoam` with k-omega SST

The open question in the issue was whether free-surface flow with k-omega SST runs at all. It does: the shipped
`interFoam/RAS/damBreak` case switched from kEpsilon to kOmegaSST ran 71 steps to `endTime` 0.2 s in 1.05 s of
execution time, with the water fraction between `-7.7e-05` and `1.0005` and a maximum Courant number of 1.02.

Getting there costs three dictionary entries that kEpsilon did not need, each of which is a fatal stop if missing:

| missing entry | error |
|---|---|
| `wallDist { method meshWave; }` in `system/fvSchemes` | `Entry 'method' not found in dictionary "system/fvSchemes/wallDist"` |
| `div(rhoPhi,omega)` in `divSchemes` | `Entry 'div(rhoPhi,omega)' not found in dictionary "system/fvSchemes/divSchemes"` |
| an `omega`/`omegaFinal` linear solver entry | `Entry 'omegaFinal' not found in dictionary "system/fvSolution/solvers"` |

The pre-flight checks all three whenever a multiphase case declares a k-omega model, so switching the model no longer
costs a solver run to discover them.

Two further findings from the same case are recorded as limits with no rule, because no rule can catch them:

- **`setFields` after `restore0Dir`.** The volume fraction is not part of `0.orig`. A case whose `0/` was restored
  after `setFields` runs to its end time with `alpha = 0` everywhere and prints no error. The phase fraction in the log
  (`Phase-1 volume fraction = 0`) is the only evidence, and the campaign records it.
- **Turbulence in the gas phase.** The k and omega fields in air are stable and numerical, and no measurement in this
  library's campaigns calibrates them. That is declared **unsupported** rather than presented as validated physics.

## Repeating the measurement

```bash
# the registry, the rules and the runs (the runs skip when no OpenFOAM is installed)
python -m pytest tests/test_openfoam_limits.py -q

# with an OpenFOAM that is not found automatically
FOAM_BASHRC=/path/to/etc/bashrc python -m pytest tests/test_openfoam_limits.py -q
```

The file holds the minimal cases: `buoyant_cavity` (a 20x20x1 closed cavity, hot left wall, k-omega SST, 200 SIMPLE
iterations) and the tutorial dam-break switched to kOmegaSST. Each limit in `pinneapple_data.preflight.limits` names
the case that demonstrates it, so a reviewer can go from a table row to the run that produced it.
