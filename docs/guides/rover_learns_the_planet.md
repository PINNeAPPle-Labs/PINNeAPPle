# The rover that learns the planet

*PINNeAPPle Autonomous Physics Lab, demo 4.* A mission in plain language drives a full-scale VIPER-class rover across
a synthetic site whose true soils are known. The first route is planned on the orbital map, which calls a patch of
soft sand "compact regolith"; the rover stalls in it, learns the soil from what its wheels measured, replans with
the learned risk and reaches the goal. Every step is checked: the physics of the runs, the agreement between the
models, and how close the learned soil is to the truth.

```python
from pinneapple_simulation.rover.autolab import RoverMission, demo_site

m = RoverMission(demo_site(), "Leve o rover até (37, 15) sem atolar, escorregamento máximo de 40%, "
                              "minimizando a energia.")
res = m.run(out_dir="out")               # report.md, trust_report.md, summary.json, figures
[r.meta["ended"] for r in res.runs]      # e.g. ["stalled: mean slip ... > limit 0.4", "goal"]
res.report.decisao                       # APPROVED / REVIEW / REJECT
```

The full demo, with the renders and the video: `python examples/autolab_rover/rover_learns_planet.py`
(about 10 min for the mission and 15 min for the renders on 4 CPU cores). The numbers of the committed run are in
[`examples/autolab_rover/results/report.md`](../../examples/autolab_rover/results/report.md) and
`summary.json` next to it; the Flow Studio node is `rover.learn_planet`.

## The loop

| Stage | Tool | What happens |
|---|---|---|
| mission | `parse_mission` | text -> `MissionSpec` (goal, start, slip limit, risk limit, objective, speed). `RuleParser` by default; an LLM adapter with the `extract(text, schema)` interface of `CallLLM` can be passed, and its values are kept only when their quote is in the text |
| plan | agent | the fixed tool sequence of `RoverMission.plan()`; only the tools of `TOOLS` are called |
| build | `survey_site` | soil units of the orbital map -> one Gaussian belief per unit over `(log10 k_phi, n0, n1, phi)` |
| simulate | `drive` | the multibody VIPER (`VehicleSimulator`, Wong-Reece wheels on several soils, pure-pursuit route following) on the *true* terrain; a traction monitor stops it when the 2-s mean front-wheel slip exceeds the limit |
| learn | `estimate_soil` | the run's log -> noisy measurements -> iterated EKF per unit, then the Birge-ratio check of the residuals; a quarter of each unit's rows is held out |
| learn | `compare_models` | analytic Wong-Reece (orbital soil and learned soil) and the zoo surrogate `rover_contact_grc1_viper` against the measured operating points |
| optimize | `plan_route` | risk-aware Dijkstra on the learned belief, from the point the rover backs out to |
| verify | `verify` | `TrustReport` (below) |

## Models and their limits

* **Site** (`site.py`): height = tilt + Gaussian mounds (a ridge with a pass); soil units are ellipses. The soft sand
  is a synthetic truth chosen for the demo (`SOFT_SAND_NOTE`); compact regolith and firm ground are the library's
  GRC-1 and JSC-1A with their recorded provenance.
* **Rover dynamics**: `pinneapple_simulation.vehicle` (multibody tree, Wong-Reece wheels with slip-sinkage, multipass
  and ruts, PI wheel-speed drives, steering, active suspension). `VehicleSimulator` gained ground of several soils
  (`soil_map`), route following and a stop rule for this demo.
* **Planner model** (`wheel_models.grade_table`): quasi-static, equal wheel loads `m g cos(a) / n`, drawbar pull
  `m g sin(a) / n`; no load transfer, no cross-slope. The trust report measures how far it is from the multibody run.
* **Sensors** (`Sensors`): load, slip, sinkage and torque with Gaussian noise; the levels are stated assumptions,
  not datasheet values. Slip is averaged over a centred 0.3-s window; the slip that sets the sinkage exponent is
  lagged as the simulator lags it.
* **Estimator** (`SoilBelief`): iterated EKF, observation model = the steady single wheel (sinkage and torque from the
  measured load and slip), measured-input noise carried through the model, front wheels only (the rear ones run in
  ruts). Cohesion, K and k_c stay at the orbital soil's values.
* **Risk** (`RiskPlanner`): share of belief samples in which the rover cannot climb a grade at a slip at or below the
  limit; the soil is checked at both ends, the middle and half a track to either side of every edge.

## Trust report

| Check | Meaning |
|---|---|
| `goal_reached`, `slip_within_limit` (critical) | the final run reached the goal with the monitored slip at or below the limit |
| `energy_covers_climb` | drive energy >= potential energy gained |
| `time_step_converged` | the start of the final route at dt and dt/2 agree within 5 % |
| `planner_vs_multibody_energy`, `_slip` | quasi-static prediction (learned belief) vs the multibody run of the final route |
| `replanned_route_risk`, `failed_route_risk_learned` | the final route is within the risk limit and the first one is not, both under the learned belief |
| `learned_soil_predictive` | held-out measurements predicted within 20 % (sinkage) |
| `learned_soil_calibrated`, `learned_soil_closer_to_truth` | synthetic site only: error within 3 posterior sd, and smaller than the prior's |
| `models_used_in_validity` | the planner uses the analytic law; the zoo surrogate's share of points outside its box is reported |

A failed check is reported, not hidden: the decision is REVIEW (or REJECT for a critical one) and the report says
which check failed and by how much.

## Renders

`render.py` (Blender Cycles through `pip install bpy`): `render_overview` (terrain coloured by the learned risk,
both routes, the rover at the stall and at the goal), `render_hero` (close-up of the stall on a light studio
background with a hard Sun shadow and sand grains instanced on the berms the terrain model computed; the grains are
visual only) and `render_traverse` (chase camera, ruts growing, MP4 and GIF). The rover is the repository's
`data/geometries/viper_rover/viper_rover.light.glb`; VIPER has no blade, so the regolith piles are the berms of the
wheel ruts.
