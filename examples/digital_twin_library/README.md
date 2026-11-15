# Digital Twin Library — five scientific seeds

Five **standalone deterministic 0D analytical/lumped MVPs**:
- FPSO: separator liquid throughput and simple mass balance (issue #1770)
- Blast Furnace: idealized fuel-to-iron sensible heat (issue #1777)
- Chemical Plant: ideal CSTR first-order conversion (issue #1779)
- Reusable Rocket: instantaneous vertical force balance (issue #1784)
- Lunar Base: solar generation and electrical demand (issue #1801)

Run from this directory: `python -m unittest discover -s . -p 'test_*.py'`.
Example: `python -c "from twins import run; print(run('fpso'))"`.

The public API is `run(name, overrides)`, returning SI-unit parameters, synthetic outputs,
`verified: false`, and explicit scope. No numerical PDE solver, neural surrogate,
telemetry, persistent state, coupling, geometry or VeriPhysics integration is claimed.
These examples are **not** suitable for engineering design, flight operations, clinical,
nuclear, or safety-critical decisions. They are not validated against independent data.

Next: reuse the PINNeAPPle experiment/model registries, actual solver adapters, workers,
VeriPhysics, independent benchmark datasets and uncertainty/validity gating.
See the five GitHub issues above and the corresponding Engineering API draft PR.
