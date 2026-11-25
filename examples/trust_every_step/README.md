# Trust Every Step — thermal demonstration

Integration scaffold for PRs #2385, #2391, #2392 and #2393.
Merge those dependencies before running this example; it is intentionally not
a standalone example on the current main branch.

Run from the repository root:

    python -m examples.trust_every_step.run

A steady one-dimensional conduction problem has the analytic solution
T(x)=qx. A linear surrogate learns the coefficient of xq from synthetic data.
Independent random cases check prediction error and interval admission.
The demonstration then checks a scoped policy and an authenticated attestation.

Expected controls: clean acceptance; rejection of nonfinite training data,
biased predictions, excessively wide intervals and revoked attestations.
These are expected outcomes, not measured results. The example has not been run.

This is a small integration scaffold, not a production pipeline or solver
benchmark. Dataset admission is a local finite-value check, not the full audit.
Solver convergence, workflow executor enforcement, strict evidence ledger and
runtime OOD authorization are not integrated here. The fixed key and clock
are demo-only. No external validation or independent certification is claimed.
