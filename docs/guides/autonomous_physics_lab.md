# Autonomous Physics Lab

A mission in plain language becomes a plan of **controlled library calls**; every call leaves evidence (arguments, a
summary of the result, wall time, error) and the mission ends in a decision with its trust report. The agent never
runs code of its own: it can only order the tools registered for its kind of mission, with the arguments each tool
declares.

```python
from pinneapple_autolab import Tool, Toolbox, Step, TemplatePlanner, LLMPlanner, execute

box = Toolbox("demo", [Tool("add", lambda state, a=1, b=2: {"sum": a + b}, "add two numbers", {"a": "x", "b": "y"})])
plan = TemplatePlanner([Step("add", {"a": 3}, "why this step")]).plan("add three", box)
run = execute(plan, box)                 # run.state["add"] == {"sum": 5}; run.evidence[0].seconds, .summary
```

| Piece | What it does |
|---|---|
| `Tool(name, fn, description, inputs, summary)` | a library call `fn(state, **args) -> dict`; only the arguments in `inputs` are accepted |
| `Toolbox(kind, tools)` | the controlled set of tools of one mission kind; `check(step)` says why a step cannot run |
| `TemplatePlanner(steps)` | deterministic: the steps a mission kind always needs (no network, no API key) |
| `LLMPlanner(fallback, provider=..., llm=..., required=...)` | a language model orders the same tools as JSON; an unknown tool or argument, a missing required step or any failure falls back to the template plan and the reason goes into `plan.notes` |
| `execute(plan, toolbox)` | runs the steps in order through the toolbox only, `Run(plan, state, evidence)` |

**Language model.** `LLMPlanner` uses `pinneapple_llm.call_llm` (Anthropic, OpenAI or a local Ollama model). With no
API key and no local model, as in the runs below, it falls back to the template plan and says so. To plan with a
model: set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`, or run Ollama) and pass `planner="llm"`. The model only orders and
parameterises the tools; the numbers in the report still come from the solvers.

## Demo 2: the self-optimizing engineer (board cooling)

`pinneapple_design.thermal.board_cooling`: a board with an overheating part, a junction limit, the room for a heat sink
and a fan budget.

```python
from pinneapple_design.thermal.board_cooling import DEFAULT_TEXT, run_mission

run = run_mission(DEFAULT_TEXT, out_dir="out")        # planner="llm" to let a model order the tools
print(run.state["verify"]["decision"], run.state["deliver"]["report"])
```

The plan (template planner):

| Step | Tool | Library |
|---|---|---|
| 1 | `read_mission` | `parse_mission`: numbers from the text with the words they came from, defaults listed as such |
| 2 | `baseline` | `simulate(mission, None)`: bare board in still air |
| 3 | `screen_natural` | the largest heat sink the envelope allows, still air, several fin gaps |
| 4 | `sample` | `simulate_many`: Latin-hypercube designs (base, fins, height, air speed), coupled reference solves |
| 5 | `learn` | `holdout`: POD + Gaussian-process surrogate of the junction temperatures and the board-top map, measured on held-out designs; more solves until max error <= 3 K and the +-2 std band covers >= 90 % |
| 6 | `optimize` | `design_loop`: lightest heat sink that meets every limit with pessimistic cooling at mean + 2 std, fan air power and fin gap constraints (differential evolution); the optimum is solved on the reference, added to the snapshots and the search repeated until the reference confirms it |
| 7 | `verify` | `verify`: the design on finer grids (board cells x2 per direction, heat-sink base grid x2) and the trust report |
| 8 | `deliver` | figures, glTF / USD, the self-contained three.js viewer, Blender Cycles stills and the animation (GIF / MP4), `report.md`, `summary.json`, `trust_report.json` |

**Reference** (`simulate`): `pinneapple_physics.pcb.solve` (layered 3-D finite volumes, JEDEC two-resistor packages,
convection + radiation) coupled to `pinneapple_design.thermal.heatsink.evaluate_design` (3-D finite-volume base, fin
correlations, pressure drop) through the case-top resistance (`Component.r_top_k_w`), iterated to a fixed point of the
heat that goes up through the sink. *Pessimistic* repeats the solve with each engine's convection band (board x 0.8,
heat sink x 0.85).

**Trust report** (`verify`): every junction under its limit (pessimistic, fine grids), fan budget and envelope
(critical); board and heat-sink energy balance, coupling convergence, the heat-sink engine's own checks, grid
convergence; the surrogate within its band on the verified design, hold-out error, coverage and the nearest-snapshot
baseline, and applicability (inside the design space, no farther from a snapshot than the held-out designs).

Needs scikit-learn (the Gaussian processes) and, for the renders, `pip install bpy`. Run it:
`python examples/autolab/self_optimizing_engineer.py` (`--smoke` for 24 solves without renders,
`--no-render` without Blender). The measured results of the run in the repository are in
[`examples/autolab/_out/self_optimizing_engineer/report.md`](../../examples/autolab/_out/self_optimizing_engineer/report.md).
Flow Studio node: `autolab.board_cooling`.

**Scope.** Engineering models with published correlations, not CFD: uniform air temperature in the enclosure, ducted
flow through the fins (no bypass), typical package data. The air paths in the 3-D scene are an energy balance per
strip of air, drawn to show where the heat goes. No conjugate CFD (OpenFOAM) run is part of this demo.
