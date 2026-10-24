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

## Demo 3: the digital twin that knows when it is wrong (pipe loop)

`pinneapple_systems.digital_twin.pipe_loop`: a pumped, heated water loop (pump, heater, pipe A with a 90° elbow, a T
junction, branches B and C with valves to two tanks) with ten sensors (four pressures, three flow meters, three
temperatures) and five inputs (pump speed, two valve openings, inlet temperature, heater duty).

```python
from pinneapple_systems.digital_twin.pipe_loop import run_mission

run = run_mission(out_dir="out")                     # planner="llm" to let a model order the tools
b = run.state["benchmark"]["benchmark"]
print(b["confusion"], b["accuracy"], run.state["trust"]["decision"])
```

| Step | Tool | What it does |
|---|---|---|
| 1 | `read_mission` | what to watch and which answers are asked for |
| 2 | `build_twin` | the fast model: POD + Gaussian processes of the ten readings over the operating envelope (`NORMAL_BOX`), measured on held-out operating points |
| 3 | `calibrate` | alarm thresholds from healthy runs (not guessed) |
| 4 | `monitor` | three events and a healthy hour: an obstruction in branch B, a drifting flow meter on B, the pump driven above the envelope |
| 5 | `request_simulation` | where the model is out of its domain: the reference at the new operating point vs the readings (does the plant still obey the physics?), then reference solves in the widened envelope, refit, monitor again |
| 6 | `benchmark` | controlled cases with known answers (random kind, size, location, onset, operating history): confusion matrix, location accuracy, Kalman size error, false alarms, detection delay |
| 7 | `trust` | the twin's trust report |
| 8 | `deliver` | control-room figures, confusion matrix, GIF, 3-D scene (pressure on the walls, transparent elbow with its velocity profile, sensors coloured by the verdict), viewer, Blender animation, report |

**How it tells the causes apart.** Ten physics residuals are computed from the readings alone: mass balance at the
tee, Darcy-Weisbach / Colebrook flow of each pipe from its pressure drop, the pump curve, both valve laws, the heater
energy balance and both branch heat losses. Each fault moves the residual vector in its own direction (a biased
sensor moves every residual it enters; an obstruction in pipe B moves only the Darcy residual of B, mass and energy
still balance), so the window mean of the residuals is matched to the fault directions by least squares; a noisy
sensor shows as variance excess along its squared direction. An extended Kalman filter tracks an extra loss
coefficient per pipe, so an obstruction comes with its size and uncertainty. The model is out of its domain when the
inputs leave the training envelope, or when the fast model disagrees with readings that satisfy every physics
residual; then the twin asks for a reference simulation instead of trusting its fast model.

Run it: `python examples/autolab_digital_twin/twin_knows_when_wrong.py` (`--smoke`: 3 cases per kind, no renders).
Measured results of the run in the repository:
[`examples/autolab_digital_twin/_out/report.md`](../../examples/autolab_digital_twin/_out/report.md). Flow Studio
node: `autolab.pipe_twin`.

**Scope.** The plant is simulated and its faults injected, so every answer is known; a real plant adds unmodelled
effects and sensor behaviour. The quasi-steady model holds for valve moves slower than ten acoustic round trips
(`quasi_steady_valid`); faster ones need the water-hammer solver. The elbow streamlines are the 1/7 power-law
profile, not CFD.
