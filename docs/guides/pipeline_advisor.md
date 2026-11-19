# The pipeline advisor: a problem in words, the study pipeline out

Ten questions decide how a simulation study with PINNeAPPle is carried out, and most of them are answered before
anything runs: is the physical model right, is the problem 3-D, is a mesh needed, does the user already have the
geometry, the mesh and a dataset, which solver fits this physics and what one run costs, sweep the solver or train a
surrogate, which architecture, how to generate the dataset, and what to check and look at at every stage.

`pinneapple_advisor` answers them in that order and returns the answer as data (JSON) and as a drawing (Mermaid, SVG,
HTML or Markdown). It never runs the pipeline it suggests.

```python
from pinneapple_advisor import advise

plan = advise("transient heat in a 3-D bracket, I have the STEP but no mesh",
              answers={"evaluations_needed": 300})
print([(s.id, s.status) for s in plan.stages])
# [('physics', 'confirm'), ('geometry', 'have'), ('mesh', 'generate'), ('solver', 'generate'), ('dataset', 'generate'),
#  ('training', 'generate'), ('verification', 'confirm'), ('visualization', 'have')]
print(plan.total_estimate().to_dict()["total_hours"], "h")     # 17.3 h: the campaign is 25 x a 500k-cell transient
print(plan.diagram("mermaid"))
plan.write("advice")          # plan.json, pipeline.mermaid, pipeline.svg, pipeline.html, pipeline.md
```

## What a stage carries

Every stage answers one or more of the ten questions (`answers`), and says what it needs, what it produces, which
PINNeAPPle calls carry it out, what it costs and what it must pass:

```python
stage = plan.stage("dataset")
stage.status            # "generate": the user has no dataset
stage.answers           # [5, 6, 9]: dataset, solver-and-cost, dataset recipe
stage.estimate.basis    # "25 simulações x 2400 s cada (estimated: 1 x 500,000 cells x 24,000 iterations ...)"
stage.estimate.confidence   # "order of magnitude", or "measured" when the user gave a real solve time
stage.checks            # coverage, data gates, failures recorded -- each with the call that does it
[t.call.rsplit(":", 1)[-1] for t in stage.tools]
# ['plan_doe', 'design', 'run_campaign', 'build_dataset', 'region_split', 'run_gates']
```

`status` is `have` (the user already has it), `generate` (to build), `confirm` (to be confirmed), `optional` or
`skip`. The diagram colours exactly that, so the picture and the JSON cannot disagree.

## The decisions, and where each comes from

| Question | Decided by |
|---|---|
| 1. Is the physical model right | `pinneapple_physics.pde_environment.capabilities.identify_pde`, then `pinneapple_problemdesign.formulate` to compose and check a model no family covers (dimensions, closure, conditions, manufactured solution) |
| 2. 3-D or not | the brief (the sentence, or the answer to the question) |
| 3. A mesh | a real shape needs one; a 3-D canonical domain needs a volumetric one; a 1-D/2-D analytic domain needs none |
| 4. Have or generate | the user's own geometry and mesh are imported (and repaired, and quality-checked), never rebuilt |
| 5. A dataset | `pinneapple_analysis.verification.architecture_recommendation.recommend_architecture` decides whether a surrogate pays off at all; a dataset on hand is used |
| 6. Which solver, what cost | the documented scope of every classical family (`_CLASSICAL_SOLVER_SCOPE`, one citation each) through `pinneapple_advisor.compose.documented_families`, with `select_solver_family` deciding and installation reported separately + the cost model of `pinneapple_advisor.estimate` |
| 7. Sweep or surrogate | the same recommendation API: solves × evaluations against the direct budget, and against the samples a surrogate needs |
| 8. Which architecture | `recommend_architecture` ranks PINN, XPINN, DeepONet, FNO, MeshGraphNet, PINO, inverse PINN |
| 9. The dataset recipe | `pinneapple_autolab.study.campaign.plan_doe` (maximin, coverage metrics) + `dataset.region_split` (a test region, not random rows) |
| 10. Checks and views | `pp.trust_card`, `pinneapple_analysis.validation.physics_learning.verify`, `grid_convergence`, `error_map`, `pp.viz.field_view`, the Studio |

The family the plan recommends is the one the physics asks for, not the one this machine happens to have
installed: the documented scope (heat in 3-D is `fdm3d`, linear elasticity in 3-D is `solid_fem`, a 3-D wave is
`wave3d_fdm`, 2-D Darcy flow is `darcy2d_fvm`, 1-D reaction-diffusion is `reaction_diffusion1d`, a turbulent 3-D
flow is `openfoam`) decides, and installation only breaks a tie between families the scope already allows. A family
the scope names but the machine cannot run stays the recommendation, with the fact said out loud and the estimate
assuming it; a case no classical family documents -- reaction-diffusion in 2-D, say, which the library solves in 1-D only --
says exactly that, instead of implying the machine is missing something.

The language model is not in that table on purpose. It reads a free-text description into the brief (and answers the
predefined questions when the words do not match a choice); every value it proposes is validated against the closed
set of fields, and everything that decides a stage comes from the APIs above. Without a model, the advisor still
works: the keyword rules read the sentence and the questions fill the rest.

## Reading a description

```python
from pinneapple_advisor import brief, next_questions

b = brief("escoamento laminar 2D em um canal, tenho a malha, erro de 2 %", llm=False)
b.values["physics_family"], b.values["dimension"], b.values["mesh_present"], b.values["accuracy_target"]
# ('navier_stokes', 2, True, 0.02)
[q["id"] for q in next_questions(b)][:4]        # what the sentence did not answer
```

`llm=None` (the default) tries a local Ollama server (`ollama serve`, nothing leaves the machine) for the fields the
rules left empty; a server that is down costs a note, not the advice. `llm=False` stays offline, and `llm=<callable>`
plugs in any provider (this is how the tests stub it).

## The cost model

```python
from pinneapple_advisor.estimate import CostModel

model = CostModel()
est = model.solve(family="navier_stokes", dim=3, cells=2_000_000, turbulent=True, cores=8, n_evaluations=50)
est.seconds_per_evaluation, est.total_seconds / 3600, est.confidence
est.assumptions            # every assumption behind the number, written down
```

The formula is cells × iterations × seconds-per-cell-iteration, divided by a sub-linear speedup on the cores, with
the iterations multiplied for a nonlinear continuation, a turbulence closure and a time loop. It is an order of
magnitude, and it says so. Two ways to stop guessing: pass `solve_seconds_measured` (the plan then reports
`confidence="measured"` everywhere downstream), or call `model.calibrate(measured_seconds, cells=..., family=...)`
once and keep the calibrated model for the rest of the study.

## The outputs

* `plan.json` — `pinneapple.advisor/1`: the brief, every stage, the edges, the decisions with their numbers, the open
  questions, the warnings and the campaign estimate. This is what an API returns.
* `plan.diagram("mermaid")` — a `flowchart TD` for GitHub and any documentation tool;
* `plan.diagram("svg")` — a self-contained SVG (rows are the dependency depth, arrows are `depends_on`), for a page
  with no network;
* `plan.diagram("html")` — the drawing plus every stage in detail, the decisions and the open questions;
* `plan.diagram("markdown")` — the Mermaid source and a stage table for a repository;
* `plan.video("pipeline.mp4")` — the same drawing revealed stage by stage (mp4 through ffmpeg, an animated GIF
  without it), for a presentation or a quick review.

From a terminal:

```bash
python -m pinneapple_advisor --text "transient heat in a 3D bracket" --offline --out advice
python -m pinneapple_advisor --questions          # the predefined questions, as JSON
python -m pinneapple_advisor --text "..." --format mermaid
python -m pinneapple_advisor --text "..." --format video --out advice   # writes advice/pipeline.mp4
```

In a flow, the node `advisor.pipeline` writes the same files into the run folder and shows the page in the Flow
Studio.

## What it does not claim

The estimates are order-of-magnitude unless measured. The architecture recommendation matches documented design
intent, not accuracy: `pinneapple_autolab.study.benchmark` is what compares families at equal budget on the user's own
dataset, and the plan lists it as the training stage's check. Nothing in the plan is executed, and the plan says so
in the page footer.
