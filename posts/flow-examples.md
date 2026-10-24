An example script shows that something worked once, on someone's machine. It doesn't say what each step cost, or whether the result still holds after the next refactor.

In PINNeAPPle we rebuilt the library's examples as flows: JSON node graphs (geometry, mesh, equations, FD/FEM solver, PINN, compare, Lab). They open in Flow Studio, but they also run without any GUI: `python -m pinneapple_flow run heat_1d`. Every run that finishes is written as a Lab experiment manifest and gets an A–F seal.

The committed batch has 14 flows. Burgers PINN, heat, Poisson, DeepONet/FNO, inverse heat, El Niño on the ENSO models, an aircraft take-off with a Cycles render, a lunar rover dashboard, a plate from 2-D to 3-D, and three twins/surrogates built from plain text. Every node was recomputed on CPU:
- All 14 flows finished, and all 95 nodes reached "done". Wall times go from 4.6 s (sensor twin) to 3768.5 s (Burgers PINN).
- Take-off: 403 m to 35 ft. Raymer gives 410 m with the simulated lift-off speed (-2 %).
- Inverse heat calibrates the diffusivity to 0.099983 from 400 noisy sensors. The true value is 0.1. The heat 1-D PINN is at 3.35e-04 relative L2 against the exact solution.

The grading is the part I care about. Seven flows carry a Lab node: 5 grade B and 2 grade F. The Burgers PINN is within 7.29e-03 of Cole-Hopf, but its residual check fails locally (max 1.10 against a limit of 0.3), so it gets an F. The plate edited to 3-D also gets an F. Both stay in the table, and each seal says what would raise it.

Flows, results and the runner: https://github.com/PINNeAPPle-Labs/PINNeAPPle/tree/main/examples/flow_studio

#ScientificMachineLearning #PINN #Reproducibility #Simulation #PhysicsInformedML #OpenSource #Python

---
## Media

1. `media/flow-examples/flow_examples.mp4` (58 s, 1080x1350, 1.9 MB). The video opens on the title, the headless command and the heat 1-D node graph. Next comes the aircraft take-off flow, with the committed Cycles clip slowed to 0.67x, then the Cycles still at 35 ft and its distances against Raymer. After that, the lunar rover dashboard flow: the committed 150-frame dashboard clip, 12.5 s. Then heat 1-D (seal B) next to Burgers 1-D (seal F), with the reason for the F. The results table follows: 14 flows, wall times and seals. It closes on an end card with the repo URL.

## Sources

All files are under examples/flow_studio/results/. The flow definitions are in pinneapple_flow/flows/*.json.
- 14 flows, all "done": RESULTS.md table (14 rows, status column), each `*/summary.json` -> `status`
- 95 nodes, all "done": sum of `len(nodes)` over the 14 `*/summary.json` (2+8+2+8+10+8+8+4+2+7+5+11+11+9), each `nodes.*.status == "done"`, `cached_nodes == []`
- 4.6 s: planned_digital_twin/summary.json -> wall_seconds (RESULTS.md row "Digital twin on a sensor stream")
- 3768.5 s: burgers_1d/summary.json -> wall_seconds
- 403 m to 35 ft: aircraft_takeoff/summary.json -> nodes.takeoff.metrics.distance_35ft_m = 402.94
- 410 m, -2 %: aircraft_takeoff/summary.json -> nodes.takeoff.metrics.raymer_sim_VLOF_total_35ft_m = 409.86, sim_over_raymer_sim_VLOF = 0.9831
- 0.099983: inverse_heat/summary.json -> nodes.calibrate.summary "alpha = 0.099983" (RESULTS.md row "Inverse heat"); true value 0.1 and 400 sensors (noise 0.01): pinneapple_flow/flows/inverse_heat.json -> true_eq.params.coefficients.alpha, sensors.params.n_points
- 3.35e-04: heat_1d/summary.json -> nodes.validate.metrics.PINN_rel_l2 = 3.347e-04
- 5 B / 2 F: the `validate.lab` nodes in heat_1d, poisson_2d, planned_plate_2d_to_3d, lunar_rover_dashboard, lunar_rover_dashboard_quick (seal B) and burgers_1d, planned_plate_2d_to_3d_edited (seal F) -> nodes.lab.metrics.seal.value; RESULTS.md "lab: tier ..." lines
- 7.29e-03, max 1.10 vs 0.3: burgers_1d/summary.json -> nodes.validate.metrics.PINN_rel_l2 = 7.29e-03; nodes.validate.warnings[0] ("max 1.10e+00 (<= 0.3 ...)")
- "what would raise it": nodes.lab.warnings ("to raise it: E: make the physics check pass")
- Manifest on every finished run: docs/guides/flow_schema.md lines 148-150 (`on_run_finished`, `pinneapple_lab.manifest.from_flow_run`)
- Video only: ground roll 244 m and Raymer 1.1 V_s 311 m (+30 %) -> aircraft_takeoff/summary.json nodes.takeoff.metrics.ground_roll_m = 244.2, raymer_1p1Vs_total_35ft_m = 311.0, sim_over_raymer_1p1Vs = 1.296; lunar 8.63 m in 45 s, slip 0.018, sinkage 36.6 mm, 1825.8 s -> lunar_rover_dashboard/summary.json nodes.simulate.summary, wall_seconds; heat FD 6.58e-06 -> heat_1d/summary.json nodes.validate.metrics.FD_rel_l2; every wall time in the table -> RESULTS.md "wall time" column
