# Offshore Digital Twin — progressive implementation ledger

This folder contains research-only MVPs, **not a production digital twin**.

| Track | Current seed | Next issue |
|---|---|---|
| 1 Hydraulics | offshore_twin.py, analytic identities | #1759 |
| 2 Drilling/riser | solvers.py kinematic power + axial bar | #1760 |
| 3 Transient | transient.py 0D exact relaxation, NOT 1D PDE | #1761 |
| 4 Data/surrogate | scenarios.py, surrogate.py nearest-neighbor baseline | #1762 |
| 5 Physics checks | verification.py local checks, NOT VeriPhysics | #1763 |
| 6 Engineering API | separate API PR #351 typed payload, NOT endpoint | API #349 |
| 7 Studio | separate API PR #351 scene descriptor, NOT rendered UI | API #350 |
| 8 Scenario library | scenario_library.py synthetic variants | #1764 |
| 9 Benchmark | benchmark.py analytic identities, NOT independent benchmarks | #1765 |

Engineering API: https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/pull/351

## Required evidence before merging

- Run all tests with `python -m unittest discover -s examples/offshore_digital_twin -p 'test_*.py'`
- Run corresponding API contract tests.
- Validate hydrostatic/friction against independent published references.
- Define real transient mass/momentum/energy equations and numerical convergence.
- Wire workers, authentication, observability, persistent results and UI.
- Integrate actual VeriPhysics gates and uncertainty calibration.
- Distinguish experimental synthetic outputs from validated operational data.

Do not present this repository as a certified well-control system or use for operational decisions.
