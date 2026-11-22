# The seven planes

This is roadmap item X40 (#227): the admission test for the core. A feature
belongs in the core only if it strengthens one of these seven planes or a
connection between two of them. A feature that does neither is a plugin, an app,
or a product — not core.

The point is not taxonomy for its own sake. Every time the core grew into
something that was really a product concern (accounts, billing, dashboards,
serving), the library got heavier for people who only wanted to solve a PDE.
The planes are the line that is easy to state and hard to argue with.

| Plane | Owns | Does not own |
|---|---|---|
| **Physics IR** | Problem, equations, units, dimensional consistency, boundary and initial conditions, `PhysicalProblem`, PDE families, symbolic form | Training, data storage, plots |
| **Data** | `PhysicalSample`, datasets, collation, sampling, lineage, audit, dataset and model catalogue | Model architecture, solver execution |
| **Compute** | Executors, backends (PyTorch/JAX/NumPy), device selection, parallelism, workflow DAGs, run events | Science: no plane decides what is correct |
| **AI** | Architectures, training policy, losses, operators, GNNs, ROMs, surrogates, active learning | Verification, product UX |
| **Engineering** | Design optimization, geometry, studies, campaigns, decision layer ("what to run next") | The physical result itself |
| **Validation** | Conservation, boundary, symmetry, dimensional checks, V&V, convergence, uncertainty, sensitivity, trust reports | Deciding what the user should do next |
| **Evidence** | Artifacts, hashes, provenance, run records, trust cards, model cards, reproduction | Producing the numbers |

Two rules follow, and they are the ones that get used in review:

1. **A plane never imports the plane above it in a way that makes it unusable
   alone.** Physics IR must run with no AI extra installed. Evidence must be
   writable by a solver run that never touched a neural network.
2. **A feature that needs two planes names the connection.** "Reproducible
   execution from an experiment id" is Data → Evidence, and belongs in the core
   because both planes are core. "Entitlements enforced server-side" is a
   product concern with no plane, and belongs in the API repository.

## Admission test

Ask, in order:

1. Which plane does this strengthen? If none, stop: it is not core.
2. Does it work when that plane is installed alone? If not, it is a connection —
   name both planes and put the code where the lower one can be used without the
   higher.
3. Does it need accounts, billing, storage of tenant data, or a UI? Then it is
   product, not core, whatever its scientific content.
4. Could a user of the core get this by writing ten lines against what already
   exists? If so, a guide is the deliverable, not an API.

## Matrix: 0.6.3 → 1.0 → 2.0

Impact is how much a plane gains; difficulty is engineering effort including
verification; dependency says what must land first. The architectural priority
is the product of the first two divided by the third — a ranking to argue with,
not a schedule.

| Feature | Plane(s) | Impact | Difficulty | Depends on | Priority | Target |
|---|---|---|---|---|---|---|
| Physics IR with units, dimensional check | Physics IR | high | high | unit layer (#110) | 1 | 0.6.3 |
| One `PhysicsValidator` returning every check with a status | Validation | high | medium | — | 1 | 0.6.3 |
| Seven-plane admission test in review | all | medium | low | this document | 1 | 0.6.3 |
| Run records with hash, environment, parameters | Evidence | high | medium | — | 1 | 0.7.0 |
| Reproducible execution from an experiment id | Data + Evidence | high | high | run records, artifact store | 1 | 0.8.0 |
| Structured run events and `pp runs why` | Compute | medium | medium | workflow engine | 2 | 0.8.0 |
| One catalogue facade over every `list_models()` | Data | medium | low | one registry protocol | 1 | 0.8.0 |
| Decision-level surrogate validation (flip rate, ranking) | Validation + Engineering | high | medium | validation package | 1 | 0.8.0 |
| Plugin contract with diagnostics | Compute | medium | high | plugin discovery | 3 | 0.9.0 |
| Cross-version schema migrations | Evidence | medium | medium | versioned schemas | 2 | 0.9.0 |
| Multi-fidelity workflow | AI + Data | high | high | cost ladder | 3 | 1.0 |
| Kubernetes and cloud batch executors | Compute | low for core, high for users | medium | executor contract | 4 | 1.0 |
| FMU export of a validated surrogate | AI + Evidence | high for industry | high | export pipeline | 4 | 1.0 |
| A second AI backend that trains the same problem | AI + Compute | medium | high | backend abstraction | 4 | 2.0 |
| GPU-native solver kernels | Compute | high | very high | sparse plane | 5 | 2.0 |

Priority 1 features are what a reviewer should be able to point at when refusing
a Priority 3 pull request that adds surface area.
