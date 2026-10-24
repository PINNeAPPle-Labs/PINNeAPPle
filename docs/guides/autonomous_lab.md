# Autonomous Physics Lab: an AI that manages evidence for engineering decisions

`pinneapple_autolab` runs one scientific cycle on any reference simulator:

```
mission ─► plan ─► build ─► simulate ─► learn ─► optimise ─► assess ─► choose the next simulation ─┐
 (objectives,   (checks of    (reference    (surrogates     (NSGA-II on   (validated domain,          │
  limits,        geometry and  solver,       compared on     the           P(meets the limits),        │
  variables,     boundary      cost logged)  held-out data,  surrogate)    error where the decision    │
  budget)        conditions)                 with an                       is made) ─► favourable /    │
                                             uncertainty)                  uncertain / out of domain   │
        ▲                                                                                              │
        └──────────────────────────── the result joins the evidence ◄──── simulate ◄──────────────────┘
                                                             … until the budget leaves only the final verifications
deliver: a simulated design (never a prediction), its trust report, the ledger of every run
```

What makes it different from an optimiser with a surrogate is what it refuses to do:

- it never simulates a design whose geometry or boundary conditions fail the simulator's checks;
- it cannot change the mission's operating conditions (the simulator owns them);
- it never spends more simulations than the budget, and keeps `reserve_verify` runs for the end;
- a limit relative to the baseline ("keep the downforce") is applied only after the baseline was simulated;
- the error of every surrogate is measured on data it did not see (k-fold cross-validation, and the prediction of
  every chosen design recorded *before* its simulation);
- the recommendation is the best *simulated* design that meets every limit; a design that was only predicted is
  labelled as such everywhere.

## Quick start

```python
from pinneapple_autolab import AutonomousLab, Budget, FunctionSimulator, Limit, Mission
from pinneapple_design.design_optimizer.study import Objective, Variable

sim = FunctionSimulator(lambda p: {"drag": 1 + (p["a"] - .3) ** 2 + .5 * p["b"] ** 2,
                                   "down": 2 + p["a"] + .8 * p["b"]}, ["drag", "down"])
mission = Mission("toy", "lower drag, keep the downforce", [Variable("a", 0, 1), Variable("b", 0, 1)],
                  [Objective("drag", "min")], [Limit("down", lower_frac_of_baseline=1.0)],
                  baseline={"a": .5, "b": .5}, budget=Budget(12, 1), acceptance={"drag": .03, "down": .03})
lab = AutonomousLab(mission, sim, "runs/toy", families=("gp", "quadratic"), nsga_budget=300, n_pool=200)
summary = lab.run(n_doe=6)
print(summary["conclusion"])            # a simulated design, its change against the baseline
```

Every step is also a method (`plan`, `baseline`, `doe`, `simulate`, `learn`, `optimize`, `assess`, `choose_next`,
`verify`, `probe`, `snapshot`, `summary`, `showcase`), so an agent can drive the same lab one call at a time; each
call writes an event to `events.jsonl` and only the ledger's runs are evidence.

## The pieces

| Module | What it does |
|---|---|
| `mission` | `Mission` (variables, objectives, `Limit`s absolute or relative to the baseline, `Budget`, acceptance tolerances, fixed conditions, hypotheses) |
| `simulator` | `Simulator` interface: `check(params) -> issues`, `run(params, workdir) -> SimOutcome(outputs, ok, cost_s, fields, meta)`; `FunctionSimulator` |
| `ledger` | `Ledger` of `RunRecord`s (purpose, reason, prediction before the run, outputs, cost), budget, environment (CPU, versions, commit), `events.jsonl` |
| `surrogates` | `GPModel`, `MLPEnsemble`, `QuadraticRSM`; `compare_models` (k-fold RMSE, relative error, R², 95 % coverage, fit time); `FieldSurrogate` (POD + per-mode model) and `field_errors` per region |
| `domain` | `ValidatedDomain`: distance to the nearest simulated design in units of their typical spacing, and the simulated range of every variable, with the reasons in words |
| `trust` | `assess_candidates` (P(feasible), constrained expected improvement, a `TrustReport` per candidate, labels), `choose_next` (the written rule below), `verification_checks` |
| `qualification` | `solver_qualification`: the same design on several meshes / iteration counts, outside the budget |
| `report` | figures from the snapshot (Pareto, models, learning, field triptych, trust cards) and the Markdown report |
| `domains.racecar` | `RaceCarAero` (OpenFOAM, `FormulaCar`) and `flagship_mission()` |

### The triage

| Label | Meaning | What the lab does with it |
|---|---|---|
| **favourable** | inside the validated domain, P(meets every limit) ≥ 0.9, 95 % intervals within the acceptance tolerances | confirms the most promising one by simulation |
| **uncertain** | inside the domain, but the prediction cannot settle the decision: P(feasible) between 0.05 and 0.9, an interval wider than the tolerance, or a large expected field error in the critical region (e.g. the diffuser) | simulates it when its constrained expected improvement is at least the best favourable one's: that is where the decision changes |
| **out of domain** | the model is extrapolating (too far from every simulated design, or outside the simulated range of a variable) | if the model's best design is here, simulates it before believing it |
| **unlikely** | P(feasible) < 0.05 | not pursued |

## Adding a physical domain

Write a `Simulator` with honest `check` and `run` (a failed run returns `ok=False` and its reason, never made-up
outputs), and a mission factory; the lab, the triage, the report and the dashboard work unchanged. A field on a
fixed set of points (`field_names`, `regions`, `field_points`) enables the field surrogate and the critical-region
check. `pinneapple_autolab/domains/racecar.py` is the reference.

## The AI Aerodynamicist

`examples/autolab_aero/autolab_aero.py`: *"Find a lower-drag aerodynamic design at 70 m/s. Preserve the required
downforce, stay within the geometry constraints, and validate the final design using CFD."* on a parametric
open-wheel car (front and rear wing angles, diffuser angle, ride height), OpenFOAM simpleFoam k-ω SST on
snappyHexMesh, half model over a moving road. Full report, ledger and figures:
[`examples/autolab_aero/_out/report.md`](../../examples/autolab_aero/_out/report.md). Measured on this repository's
4-core container (OpenFOAM v1912), 2026-10-24:

**Solver qualification** (baseline, forces averaged over iterations 400-800 because the steady RANS forces keep
oscillating by about 2 % after 500 iterations):

| mesh | cells | drag vs fine | downforce vs fine | balance vs fine |
|---|---|---|---|---|
| medium | 0.21 M | -8.8 % | **-43.7 %** (wing elements one or two cells thick) | -3.4 % |
| campaign (wings and floor at the fine levels) | 0.42 M | -0.2 % | -8.4 % | +9.6 % |
| fine | 0.82 M | 0 | 0 | 0 |

**Campaign**: 20 CFD runs (1 baseline, 10 space-filling, 6 chosen one at a time, 3 final verifications), 3.3 h
of solver time, no failed run.

| | judged before its CFD | CFD said |
|---|---|---|
| run012, run013 | out of domain (the model's optimum, extrapolating) | verified; drag predicted within 0.8 % and 1.2 % |
| run014 - run017 | uncertain (P(feasible) 0.73 - 0.84) | 1 verified, **3 rejected** (downforce or balance limit violated; run016's drag 3036 N against 2849 ± 46 N predicted) |
| run018 - run020 | favourable | 3 verified |

**Recommendation**: run019 (simulated), drag 2795 N against 3315 N (**-15.7 %**), downforce 7319 N (limit
7038 N), balance 1.073 (limit 0.913 - 1.116). **On the fine mesh** (outside the budget): drag **-13.95 %**
(5.5 x the iteration noise), downforce +0.8 % (within the noise), balance 1.029 against a fine-mesh limit of 1.018:
**outside by 1 %, within 1.4 x the noise**. The drag reduction is resolved by the solver on both meshes; the balance
limit is not resolved on the campaign mesh (+9.6 % bias), which the report flags.

Surrogates (5-fold cross-validation, 20 runs): MLP ensemble chosen for every output, held-out mean relative error
drag 2.4 %, downforce 10.2 %, balance 4.3 %; the quadratic response surface needs more runs than its 28
coefficients and is skipped. Surface Cp (POD + GP): held-out RMSE 0.07 on the body, 0.12 on the diffuser, 0.21-0.24
on the wings (the wings have vertices only at their span ends).
