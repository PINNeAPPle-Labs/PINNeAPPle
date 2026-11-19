---
title: "The local cavity engine stops on a wall-clock guard, so the same case answers differently on a busy machine"
labels: [enhancement, solvers]
---

**Who has it:** `examples/pict_cavity` and anything that runs the local vorticity–streamfunction engine
(`pinneapple_simulation.external_solvers.pict.local_solver`) on a machine that is not idle.

**What happens:** `LocalSolverConfig.max_time_s` defaults to 300 s, and the march stops there when the vorticity
residual has not reached `steady_tol` in time. That is a wall-clock stopping criterion, so the number the solver
returns depends on what else is running on the machine.

The same 128 × 128 cavity at Re = 1000, run twice on this repository's example:

| machine | stops on | steps | residual | rms of `u(x = 0.5)` vs Ghia |
|---|---|---|---|---|
| free core | `steady_tol` | — | < 1e-6 | 8.13e-3 (what `examples/pict_cavity/cavity.json` records, 22.5 s) |
| 10 cores loaded by 27 unrelated processes | `max_time_s` = 300 s | 9 000 | 4.2e-2 | 1.21e-2 |

Both pass the example's 0.05 tolerance, so nothing fails and nobody notices — the example just quietly answers a
less converged question. The same case at Re = 100 is unaffected (4 052 steps, 1.382e-3, identical on a loaded and an
idle machine), which is why the difference looks like a Re or mesh effect when it is only scheduling.

**What PINNeAPPle already has:** `meta["steps"]` and `meta["final_residual"]` are reported by the run, and
`LocalSolverConfig.max_steps` (400 000) already exists as a work budget.

**Proposal:** make the stopping criterion a function of work rather than of the clock — either drop `max_time_s` to
`None` as the default and rely on `max_steps`, or keep both but record which one ended the run in the result
(`meta["stopped_by"] = "residual" | "steps" | "time"`) and surface it in `cavity.json`. The example should then say
in its own `README` what it costs to converge, not how long it happened to take.

**Acceptance (rule D1 — independent reference):** the same case, run once on an idle machine and once with every
core saturated, returns bit-identical velocity fields when neither run hits a guard; and a run that does hit a guard
is reported as not converged, so a check that depends on it can no longer pass silently.

**Origin:** reproducing the "blank machine + agent + open-source toolchain" experiment (2026-11-13): the Re = 1000
cavity came out at 1.2e-2 on a loaded machine against the 8.1e-3 the repository records for the same case.
