# Rollout validation: when an autoregressive surrogate stops being trustworthy

An autoregressive model (one step at a time, its output fed back as input) can look accurate for a few steps and
then lose energy, blur, drift or blow up. A one-step error says nothing about that.
`pinneapple_analysis.validation.rollout` runs the model over a horizon and reports where it stops being
trustworthy. Every test is checked against a classical solver.

```python
from pinneapple_analysis.validation.rollout import rollout, validate_rollout

pred = rollout(model_step, u0, 300)                 # (301, n): u0 and 300 model steps
rep = validate_rollout(pred, ndim=1, ref=reference, dt=1.0, conserved=["mass"], chaotic=True,
                       long_ref=long_reference)     # a long run of the solver, from any initial state
rep.evidence["divergence"]["trust_horizon"]         # earliest crossing, in time units
[(c["id"], c["status"]) for c in rep.checks]

import pinneapple as pp                             # the same checks in the validation report
pp.validate(model_step, {"rollout": {"ndim": 1, "dt": 1.0, "conserved": ["mass"], "chaotic": True, "steps": 300}},
            data={"u0": u0, "reference": reference, "long_u0": u1, "long_reference": long_reference})
```

## What is measured

| Function | Measures |
|---|---|
| `invariant_drift` | mass, energy (or any callable): drift rate relative to the initial value and, with a reference, the largest relative deviation from the reference's own series (decaying flows lose energy; the model must lose it at the same rate) |
| `spectral_drift` | energy in the top third of the resolved shells, over a trailing window, relative to the reference at the same time (or to a training spectrum) |
| `divergence_time` | first time each of pointwise error (20 %), spectral bias (25 %) and invariant deviation (5 %) crosses its tolerance, each over a trailing window of 10 frames; the trust horizon is the earliest |
| `long_run_statistics` | for chaotic systems: Wasserstein-1 distance between value histograms, decorrelation time, and the time-averaged energy in the low, mid and high thirds of the resolved shells; each also between the two halves of the reference, giving the sampling noise floor |

## Checks

| ID | Fails when |
|---|---|
| `rollout.finite` | a NaN or Inf appears (with the first bad step) |
| `rollout.invariant.<name>` | a declared conserved quantity drifts more than `drift` (1e-3) relative |
| `rollout.horizon` | never fails: a finite trust horizon is a *warn* asking for review, with the horizon as its value |
| `rollout.spectrum` | the high-band energy leaves the 25 % band before the pointwise error crosses (with `chaotic=True`, a crossing after the error crossing is expected and does not fail) |
| `rollout.invariants` | the invariants deviate from the reference's before the pointwise error crosses |
| `rollout.statistics` | any long-run distance is more than 2× its noise floor (with minimum floors of 1e-3 of the field scale, 0.05 in log-band energy and 10 % of the decorrelation time) |

`pp.validate` runs the long-run statistics only above `level="quick"`. Tolerances can be overridden with
`tolerances={...}` (`error`, `spectrum`, `invariants`, `drift`, `floor`).

## Results (from `tests/test_rollout_validation.py`)

Kuramoto–Sivashinsky (`pinneapple_simulation.numerical_solvers.kuramoto_sivashinsky`, L = 32π, 128 modes, ETDRK4,
the discretisation cross-validated against Exponax), frames one time unit apart:

| "Model" | error crosses | spectrum crosses | checks |
|---|---|---|---|
| the same solver with dt/2 | t = 57 | t = 82 | all pass; horizon *warn* at 57 (chaos: the Lyapunov time sets it) |
| solver + 30 % energy removed above mode 20 every step | t = 7 | t = 1 | `rollout.spectrum` fails; long-run statistics fail (band energy 0.52 vs floor 0.05) |
| FNO trained on one-step pairs | finite, < 57 | | horizon set by the pointwise error |

The halved-step solver from a different initial state passes `rollout.statistics` (0.048 vs floor 0.053). On 2-D
decaying turbulence (`multiscale.ns2d`, 64², ν = 1e-3), the halved-step solver passes everything with no finite
horizon over 3 time units.

## Limits (honest results)

- The pointwise horizon of a chaotic system is set by the Lyapunov time: an exact integrator has one too. Read
  `rollout.horizon` against the horizon of a reference integrator, not as a failure.
- The long-run statistics need a long reference run (here 1500 frames); with short runs the noise floor is large and
  the check is weak. A per-shell maximum of the spectrum ratio was too noisy to detect 30 % high-band damping; band
  energies are used instead.
- Mass conservation is checked as drift of the spatial mean; for a field with zero mean the scale is the mean of
  |u|, so a drift is relative to that.
- The invariants are those of a periodic box; with walls or sources supply your own callables.
