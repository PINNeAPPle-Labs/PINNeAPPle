# Physics AI Strategy — structure and first implementation

> **Status:** structured from the owner's research (October 2026), which covers the public PINNeAPPle repository, recent Physics AI / Scientific ML / CAE job postings, and public moves by NVIDIA, Siemens, Dassault, SimScale, PhysicsX, PTC, Blue Origin, Periodic Labs and Engineering AI startups.
>
> **Verification status:** this document does **not** verify the market facts quoted in the research (company launches, customer counts, dataset sizes, pricing, marketing claims). Treat them as working hypotheses until checked against primary sources.
>
> **Scope of this file:** what the research recommends, how that maps to the code that exists today, what is implemented, and what is decided versus still open.

---

## 1. Thesis

Physics AI is moving from a question of neural architecture ("which network solves this PDE best?") to a question of **engineering infrastructure**. Commercial value shifts from the model to the accelerated workflow: explore more designs, turn hours of simulation into seconds of prediction, operate a digital twin, and let an agent run the engineering process.

Evolution proposed by the research:

```
Solver → Surrogate → Physics AI → Design AI → Digital Twin → Engineering Agent → Autonomous Engineering
```

Recommended positioning: **an open, vendor-neutral infrastructure layer between physical models and engineering applications.** PINNeAPPle does not replace solvers (OpenFOAM, FEniCS, Ansys, Siemens). It sits above them and composes them.

Recommended one-line description:

> An open, vendor-neutral infrastructure for building, validating and deploying AI systems for physical engineering.

## 2. Current state of the repository

Verified by inspecting module directories and `ROADMAP.md` on `main` (October 2026). "Exists" means a module is present, not that it is validated in production.

| Layer | Modules present | Main gap |
|---|---|---|
| Problem specification | `pinneapple_problemdesign` (`ProblemSpec` for elicited problems), `pinneapple_physics/pde_environment` (`ProblemSpec` for PDEs), `pinneapple_data/upd_types` (`ConditionSpec`) | Three overlapping specs; no single universal physical problem (see §5, Decision D1). |
| Physics AI models | `pinneapple_neural`, `pinneapple_models`, `pinneapple_physics` (PINN, neural operators, GNN, ROM, SINDy, symplectic models) | No common interface; the user chooses the architecture. |
| Physics engines | `pinneapple_solvers`, `pinneapple_simulation`; bridges to OpenFOAM and FEniCS in the CFD product | Bridges are per product, not a common backend contract. |
| Geometry | `SDF`, `CSG`, mesh and GNN code | No geometry as a first-class layer. |
| Data | `pinneapple_data`, `pinneapple_registry` (problem, dataset, model, experiment stores), `pinneapple_pdb` | Stores exist separately; no lineage graph linking them. |
| Trust and validation | `pinneapple_analysis/trust/trust_gate.py` (OOD, residual, ensemble per prediction); `pinneapple_analysis/verification/evidence_graph.py`; `pinneapple_hub/model_card.py` | No deployment decision aggregating the checks (addressed in §4). |
| Optimization | Bayesian optimization code and design modules (`pinneapple_design`) | Not connected to surrogate validation. |
| Agents | `pinneapple_llm` (guardrails), `pinneapple_tools`, `pinneapple_problemdesign` | No agent that runs the full workflow through the platform APIs. |
| Benchmarking | `pinneapple_arena`, `benchmarks/` | Not yet a reference benchmark with domain-of-validity reporting. |
| Foundation / world models | `pinneapple_worldmodel` (described as generalist) | No multi-domain demonstration against baselines. |

## 3. The research's diagnosis: nine problems, one architecture

The research lists twenty problems. The ones that drive the plan:

1. **CAD / engineering intent → executable simulation.** Missing: an engineering problem compiler (intent → physics → geometry → mesh → solver → surrogate → validation).
2. **Generating data efficiently.** Missing: a simulation factory (CAD → design of experiments → solver → dataset → quality control), with active learning to pick the next simulations.
3. **Choosing the right model.** Missing: a model selector that compares candidates (PINN, operator, GNN, ROM) on a benchmark, instead of a fixed choice.
4. **Out-of-distribution generalization.** Missing: a generalization suite that tests geometry, physics, resolution and regime OOD separately.
5. **Knowing when to trust a surrogate.** Missing: a trust layer that combines residual checks, OOD, uncertainty and boundary-condition violations into a decision.
6. **Deciding AI versus solver per request.** Missing: an adaptive fidelity engine.
7. **Surrogate, solver and optimizer are separate systems.** Missing: a design loop (requirements → candidates → physics AI → high-fidelity validation → Pareto front → engineer review).
8. **Generated geometry ignores physics.** Missing: a physics-aware generator.
9. **Engineering language is not executable.** Missing: intent → physics selection → geometry → material → boundary conditions.
10. **Agents cannot call engineering tools safely.** Missing: a typed, auditable tool surface.
11. **Simulations are not reproducible.** Missing: experiment lineage (commit, environment, data, seeds, hardware, outputs).
12. **Multi-fidelity learning.** Missing: training on cheap, medium and high-fidelity data together.
13. **Transfer across geometries and applications.**
14. **Neural operator scaling.**
15. **Physics AI on bad problems** (stiff, multi-scale, sparse data).
16. **Connecting the physical world to the model** (sensors, state estimation).
17. **Multi-objective optimization.**
18. **Explaining model decisions to engineers.**
19. **Connecting multiple solvers.**
20. **No universal representation of a physical problem.**

**Priority list from the research (ten problems, in order):** engineering intent → simulation (1); simulation data factory (2); model selector (3); OOD generalization (4); trust layer (5); adaptive fidelity (6); design loop (7); orchestrator (8); runtime and digital twin (9); universal physical problem (10). Foundation models and world models come after these.

## 4. Products the research proposes (seven internal products)

| Internal product | Pipeline | Existing base | Status |
|---|---|---|---|
| Problem Compiler | natural language or CAD → universal physical problem | `problemdesign`, `physics/pde_environment` | Specification gap (D1). |
| Simulation Factory | CAD → DOE → solver → dataset → quality control | `simulation`, `data`, CFD product pipeline | Works per product; not generalized. |
| Physics Model Factory | dataset → architecture search → training → UQ → benchmark → registry | `neural`, `train`, `registry`, `arena` | Pieces exist; no single pipeline. |
| **Physics Trust** | prediction → physics checks → OOD → UQ → validation → trust score | `analysis/trust`, `hub/model_card` | **First slice implemented (§5).** |
| Design Explorer | requirements → generate → simulate → surrogate → optimize → Pareto | `design`, `analysis` | Not connected to validation. |
| Physics Agent | engineer ↔ agent ↔ CAD / solver / model / optimizer | `llm`, `tools` | Not built. |
| Physics Runtime | trained model → API → edge → digital twin → sensor stream → continuous validation | `systems/digital_twin`, `hub` | Partial. |

The research also proposes **Geometric AI** (CAD, B-Rep, SDF, point cloud, mesh and graph representations) and **Multi-fidelity learning** as cross-cutting capabilities.

## 5. Implementation status

### 5.1 Done: deployment trust report (Physics Trust, first slice)

`pinneapple_analysis/trust/trust_report.py` combines independent checks into one decision: `APPROVED`, `REVIEW` or `REJECT`.

- `Check`: one check with status `supports`, `contradicts` or `not_run`, criticality, weight and reason. The vocabulary matches the proofs used by the CFD product's verification module.
- `TrustReport`: score (weighted over executed checks), coverage, reasons, Markdown and dict output.
- `from_trust_score`: converts the sub-scores of the existing `TrustGate` (OOD, PDE residual, ensemble) into checks.

Decision rules (design choices, not standards):
- any failed **critical** check → `REJECT`;
- any other failed check → `REVIEW`;
- coverage below 60% → `REVIEW`;
- score below 0.80 → `REVIEW`;
- otherwise → `APPROVED`.

Tests: `tests/test_trust_report.py` (9 tests) and `tests/test_trust_gate.py` pass.

### 5.2 Decided, not yet implemented

- **D1 — Universal physical problem.** Do **not** add a third specification class. First reconcile `problemdesign.ProblemSpec`, `physics.pde_environment.ProblemSpec` and `upd_types.ConditionSpec` into one representation with explicit conversions, then add the universal problem as a façade over it. Reason: a third class increases fragmentation, which is the problem the research describes.
- **D2 — Trust report is linked to model records.** Next step: store the `TrustReport` in `ModelCard` and in the registry, so every published model carries its decision and reasons.
- **D3 — Calibrate trust limits.** The 0.80 score, 60% coverage and 0.7 sub-score limits are provisional. Calibrate them on the existing surrogates of the CFD product, whose measured errors range from 0.23% to about 10%.

## 6. Demonstrations the research recommends

Five demonstrations, each chosen to show the full loop rather than a single model:

1. **Aerospace:** wing geometry (STEP) → CFD → neural operator → drag/lift prediction → geometry optimization → high-fidelity CFD validation.
2. **Automotive:** vehicle geometry → simulation factory (about 10,000 designs) → neural operator → OOD validation → aerodynamic optimization.
3. **Structural:** bracket → FEA → surrogate → stress prediction → mass optimization → uncertainty → validation.
4. **Industrial digital twin:** CFD/FEA → surrogate → sensor stream → ensemble Kalman filter → anomaly detection → prediction.
5. **Agentic engineering (flagship):** "Design a cooling system that keeps the electronics below 80 °C while minimizing mass and pressure drop" → the system interprets the problem, generates geometry, selects physics, runs simulations, trains a surrogate, optimizes, validates, and returns five candidate designs.

The CFD product (pipe erosion) is a vertical slice of demonstration 1's pattern and should be the reference for the surrogate and trust pieces.

## 7. Phases

| Phase | Focus | Entry condition |
|---|---|---|
| 1 | Universal physical problem (D1) | — |
| 2 | Simulation factory | Phase 1 |
| 3 | Physics model factory | Phase 2 |
| 4 | Trust, OOD and validation (§5.1, D2, D3) | Phase 3 |
| 5 | Adaptive fidelity | Phase 4 |
| 6 | Design optimization | Phase 5 |
| 7 | Engineering agent | Phases 1–6 provide APIs for the agent |
| 8 | Digital twin runtime | Phase 4 (trust) and phase 6 |
| 9 | Foundation and world models | Phases 1–8 |

The research's rule for the sequence: each stage feeds the next, forming a closed loop — problem → data → model → trust → decision → optimization → deployment → real world → more data.

**Recommendation from the research:** temporarily freeze horizontal expansion of models, and invest in the sequence above. The repository already contains most primitives; the work is composition, not new architectures.

## 8. What the research recommends not doing

- Do not add more architectures without an industrial problem that needs them.
- Do not try to build a new general-purpose solver or a new CAE platform.
- Do not compete head-on with SimScale on CFD/FEA.
- Do not build a generic "ChatGPT for CFD": it is easy to copy.

## 9. Decisions the owner must take

These are not decided here:

- Whether the product is sold as a platform, as engineering applications, or as both.
- The first industrial vertical (the research points at aerospace and automotive; the current product is pipe erosion).
- Whether to publish a reference architecture or keep the design internal until the first demonstration exists.
- The trust limits (D3) and the erosion accuracy target for the CFD product (today only ΔP has a target).

## 10. Caveats

- Market and company statements in the research are not verified here (see the status note at the top).
- The research text is long and was partly duplicated in the source file; this document keeps only the structure, not every example.
- Career-related remarks in the source research are intentionally not part of this repository document.
