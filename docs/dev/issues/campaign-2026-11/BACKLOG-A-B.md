# Backlog — series A (prediction), E (infrastructure), index, and series B (generative 3D geometry + real simulation)

Format: `## ID | Title | labels: ...` then the concept notes the issue body must expand.
The title is used verbatim as the GitHub issue title.

## A1 | ENSO v2: a sealed out-of-sample El Niño forecast, calibrated probabilities and regime-conditional skill | labels: enhancement, experiment, priority: P2, effort: L, data, benchmark, uq

Follow-on to the committed study in `examples/enso_forecasting/` (guide `docs/guides/enso_forecasting.md`,
code `pinneapple_systems.time_series.enso`, results in `examples/enso_forecasting/results/`). That study is done and
its numbers stand; this issue adds the one thing a hindcast cannot buy — a forecast issued before the answer exists.
Core: pre-register forecast origins forward in time (the first monthly origin after the data end), write the plume to a
file, hash it, commit it, and score it only when the verifying observations are published. Second core: the ensemble
was a fixed equal-weight mixture and the guide says so; replace it with calibrated weights (EMOS / BMA / CRPS-trained
weights on the validation window) and report reliability, PIT histogram, RPS and log score with bootstrap intervals.
Third: regime-conditional skill (El Niño / La Niña / neutral by ONI threshold), multi-year leads, and the documented
limits — PMEL warm-water-volume product, fixed 1982–2011 anomaly base and the +0.05 K/decade trend, thermocline from
WOD D20 instead of a reanalysis. Add families the zoo lacked: FNO/Transolver on SST maps, a GNN on the equatorial
waveguide, LETKF vs EnKF with honest ensemble dispersion (the EnKF was under-dispersed: 17 % coverage at lead 6).
Acceptance must include a pre-registration date stamped before the verification date, and a stated consequence if the
sealed forecast misses.

## E1 | Scientific cinematic pipeline: every frame of a demonstration video traceable to a solver output | labels: enhancement, experiment, priority: P2, effort: L, visualization, twin3d, interop

Shared infrastructure for the whole campaign. Today: `pinneapple_blender.cycles` renders a scene from arrays
(`CyclesScene`, `render_frames`, `frames_to_video`), `pinneapple_blender.export` writes a numbered `.ply` sequence and
`pinneapple_blender.render.build_scene` builds a Blender scene headless, `pinneapple_twin3d.scene.export` writes
`scene.json` + `geometry.glb` + `fields.bin` with a bundled web viewer and optional OpenUSD, `pp.dashboard.to_video`
writes a GIF/MP4, and `pinneapple_worldmodel.physical_world.physics_checks(video)` checks a rendered video against
physics. Missing: one pipeline that takes a solver trajectory (mesh or point cloud + fields + time) and produces a
frame manifest in which every frame carries the case id, solver, git commit, input hash and field statistics, so a
viewer can prove the video is not hand-authored; camera presets that keep scale bars, legends and units; a
colour-map contract (perceptual, consistent, with the range stored); and a verification step that re-projects a few
frames back to solver arrays and asserts agreement. Also: budget (frames, samples/frame, wall-clock per minute of
video), a fallback when no Blender is installed (twin3d web viewer + dashboard), and a check that the pipeline refuses
to render a frame whose provenance fields are empty.

## D0 | Experiment campaign: reproducible Physics-AI demonstrations, one evidence protocol, and what each one may claim | labels: documentation, experiment, priority: P2, effort: M, roadmap

The series index and the protocol every issue in the campaign follows. Content: purpose (a demonstration is a
reproducible experiment, not an animation); the five stages of the Fiction-to-Reality protocol (define the promise →
separate the known physics → build and run the model → push it to its limit → map what is missing); the campaign's
use of the machinery that already exists (`pp.futures` with its six evidence axes and two verdicts,
`PLAUSIBILITY_CLASSES = established|extrapolated|hypothetical|fictional`, `PROVENANCE_KINDS`, `STAGES`,
`pp.science.atlas()`, `pinneapple_lab.experiment` manifest + grade + seal); the rule that a simulation fills
`mathematical_consistency`, `numerical_stability` and `physical_plausibility` and can never fill
`experimental_validation` or `technological_feasibility`; the five artefacts every experiment must ship (reproducible
case, video with provenance, limit map, gap analysis, evidence level); the requirement that optimiser results are
re-verified on a refined mesh and compared with a baseline; and the honesty rules (no fabricated numbers, no claim
that a render proves a technology, speculative subjects always carry their class). Ends with the campaign table
(filled in after the issues exist) and a short "how to claim this on stage" note.

## B1 | MORPH: a machine that learns its own physics and redesigns itself, in simulation | labels: enhancement, experiment, priority: P2, effort: L, design-optimization, neural-operators, agents

A flexible wing or beam whose mass/stiffness/damping are treated as unknown at the start. The system perturbs it,
identifies the parameters, chooses the best model family from evidence (PINN, FNO, DeepONet, GNN, classical solver),
predicts, then optimises a morphing configuration (camber, span, sweep of a 2-D/3-D section) for lift or stiffness,
and the prediction is registered before the "measurement" (here: the reference solver run) is allowed. Physics:
linear/nonlinear beam + aeroelastic coupling; identification via `pinneapple_analysis.inverse_problems`
(identifiability) on synthetic truth with noise; the honest part is that in simulation the truth is known, so the
experiment must use a held-out configuration and report the identification error against it. Reuse
`design.aero.aircraft3d`/`case3d` for the aero side, `nonlinear_beam_fem`/`solid_fem` for the structure,
`neural_operators`, `pp.compare`, `pp.futures` for the assessment. Missing: the orchestration loop (perturb →
identify → select model → optimise → verify) as one recorded experiment, and a differentiable or surrogate-coupled
aeroelastic path fast enough to close the loop.

## B2 | AERIS: an aircraft whose geometry changes in flight, optimised for the conditions it is in | labels: enhancement, experiment, priority: P3, effort: L, design-optimization, geometry, solvers

A morphing airframe (variable sweep, span, camber, or an articulated winglet) evaluated at several flight conditions,
with the geometry optimised as a function of the condition rather than once. Physics: potential-flow/VLM from
`pinneapple_design.aero` (`panel`, `vlm`, `aircraft3d`, `case3d`, `optimize3d`) plus the 6-DOF and turbulence side of
`pinneapple_simulation.flight` (`trim`, `turbulence_encounter`, `simulate_takeoff`); an honest quasi-steady assumption
for the morphing rate, plus a mass and actuation penalty for the mechanism so the morphing wing cannot win for free.
Missing: a parametric morphing geometry whose aerodynamic model is re-generated as the geometry changes, and a
multi-point optimisation that reports the Pareto front of (L/D at cruise, trimmed C_L max, actuator mass).

## B3 | XENOFORM: a structure the optimiser grows to carry load and to route flow at the same time | labels: enhancement, experiment, priority: P3, effort: L, design-optimization, geometry, fem, solvers

3-D density-based topology optimisation with two physics in one objective: structural compliance and Stokes
dissipation (Borrvall–Petersson) of a fluid in the same design domain, coupled by a Brinkman penalisation and a
blocked/fluid density field. Exists: `pinneapple_design.topology` gives SIMP, filters, Heaviside projection, OC and
MMA, `Compliance2D`, `ThermalCompliance2D`, `StokesDissipation2D` — all 2-D today. The work is the 3-D extension and
a multi-physics objective with a volume constraint shared between the two fields. Verification: re-solve the optimum
with `external_solvers.openfoam` on a body-fitted mesh and with `external_solvers.calculix`, report GCI, and compare
against a uniform lattice and a hand-designed truss at equal solid volume.

## B4 | LEVITATE: a controller that learns to hold a body in a magnetic field, and what that costs | labels: enhancement, experiment, priority: P3, effort: L, solvers, uq, training

Levitation is unstable by construction (Earnshaw), so the experiment is about the control, not the magic. Physics:
magnetostatic field from `pinneapple_simulation.electromagnetics.statics` (and `eddy`, `hysteresis` for losses),
rigid-body dynamics with the field force, a linearised model around the operating point, then closed-loop control
(LQR/MPC or a learned policy), then a robustness sweep on sensor noise and coil saturation. Report: levitation
stiffness, control current, electrical power per unit mass held, settling time, and the load range in which the loop
still holds. Missing: dynamic two-way coupling between the field solution and the body position (currently the modules
are static/analytic), and a stability-margin metric in the report.

## B5 | LIVING MATTER: a cellular structure that becomes stiffer, softer or more conductive region by region | labels: enhancement, experiment, priority: P3, effort: L, design-optimization, fem, geometry

A lattice of actuated cells (a stiffness or conductivity switched per cell) reconfigured for a task: maximum global
stiffness, a target compliance pattern, or a heat path. Exists: `pinneapple_design.bricks` (`Assembly`,
`analyse_assembly`, `evolve`, `surrogate_search`, `compare`) for building-block statics, `topology` for the continuous
limit, `solid_fem`/`plane_stress_fem` for verification. Physics: linear elasticity per configuration, an explicit
switching cost and actuation time, and a power budget for holding a configuration. Missing: a cell-level reconfiguration
model (state per cell, connectivity constraints) and the metric that separates "the best configuration" from "the best
configuration reachable from the current one".

## B6 | FLOW GOD: control a channel flow with what the model has learned, not with a hand-tuned schedule | labels: enhancement, experiment, priority: P3, effort: L, solvers, neural-operators, uq

Closed-loop control of a confined flow (wall-normal blowing/suction or a set of internal jets) to hit a target:
mixing, heat transfer, or drag reduction. Exists: `numerical_solvers.internal_flow`, `thermal_channel_2d`,
`tvd_advection`, `lbm`, `stable_fluids`, `accelerated_foam.learned`/`warm_start` for learned initialisation.
Missing: a fast reduced-order model good enough for a control loop, a policy class, and an honest comparison against
the best steady forcing found by optimisation and against a hand-tuned schedule. Report control cost (actuation
energy), achieved objective, and stability of the closed loop under a disturbance not present in training.

## B7 | VORTEX TREE: a 3-D manifold a topology optimiser grows to spread flow evenly at minimum loss | labels: enhancement, experiment, priority: P2, effort: L, design-optimization, solvers, geometry

One inlet, many outlets in a 3-D volume; objective: minimise total pressure loss at fixed flow split, with a
uniformity constraint on the outlet fluxes. Exists: Brinkman-penalised Stokes topology optimisation in
`pinneapple_design.topology` (`StokesDissipation2D`, `optimize`, `DesignChain`, MMA/OC) — 2-D; the OpenFOAM side
(`external_solvers.openfoam`, `pipelines.freecad_openfoam`, `PipelineSpec`, `Sweep`) and mesh handling
(`design.geometry.mesh`, `stl`). The work: 3-D Stokes/Brinkman optimisation at a resolution where the branches are
resolved (order 10^6 elements is enough to look convincing and is honest about cost), an extract-and-remesh step to a
body-fitted mesh (STL → `snappyHexMesh`), the re-solve with `simpleFoam`, and GCI. Baselines: a T-junction or a
conventional bifurcating manifold of the same volume, and the unoptimised volume. Deliverables must include the STL of
the optimum, the pressure/velocity fields, the objective history and the per-iteration geometry sequence for the video.

## B8 | THERMAL ORGANISM: a heat sink the optimiser grows, branch by branch | labels: enhancement, experiment, priority: P2, effort: L, design-optimization, solvers, geometry

Topology-optimised heat sink for a fixed footprint and volume: minimise thermal resistance (or peak temperature) under
a pressure-drop budget, with conjugate conduction–convection. Exists: `pinneapple_design.thermal.heatsink` (`engine`,
`field`, `sizer`, `surrogate`, `trust`), `fin_array_speed`, `topology.ThermalCompliance2D`, the board-cooling mission
(`design.thermal.board_cooling.run_mission`), `thermal_channel_2d`. Missing: a conjugate 3-D solve inside the design
loop (today conduction-only topology optimisation and separate fin correlations), and the mapping from the optimised
density field to a manufacturable fin geometry. Verification: the optimised sink re-solved with OpenFOAM
(`chtMultiRegionFoam` or a conjugate case) at two mesh levels, and compared with a straight-fin array of equal volume
sized by `fin_array_speed`.

## B9 | AERODYNAMIC EVOLUTION: a body whose shape evolves to cut drag, generation by generation | labels: enhancement, experiment, priority: P2, effort: L, design-optimization, geometry, solvers

External flow around a parametric 3-D body (a simplified vehicle or a bluff body) with shape variables from
`pinneapple_design.geometry.gen` (`sdf_shapes`, `car2d`, `domains3d`, `cadquery_gen`) and aerodynamic evaluation from
`design.aero` (panel/VLM where valid, OpenFOAM where not), optimised for drag at fixed frontal area and volume, then
re-verified on a refined mesh. Physics: incompressible RANS at moderate Re; the honest caveat is that panel/VLM methods
cannot predict separation, so the surrogate trained on coarse CFD must be checked against the fine case and the
dispersion reported. Baselines: the initial body, a spheroid of the same volume, and the shape from a
gradient-free search of the same budget. Deliverables: generation sequence (geometry + pressure field per generation)
for the video, the drag history, and the final STL.

## B10 | STRESS GARDEN: material migrating to where the load is, in 3-D | labels: enhancement, experiment, priority: P2, effort: L, design-optimization, fem, geometry

3-D compliance topology optimisation under a load case, with stress and displacement limits: minimises mass (or
compliance) subject to a von Mises bound, then re-analysed with a different solver than the optimiser used. Exists:
`pinneapple_design.topology` (`optimize`, `Compliance2D`, `elasticity_ke`, `DensityFilter`, `heaviside`, `MMA`),
`solid_fem`, `elasticity3d_fdm`, `external_solvers.calculix`, `design.geometry.mesh`. Work: 3-D problems (the existing
problems are 2-D), a stress-constraint handling good enough not to produce grey material, and the export/re-import
round trip to CalculiX. Acceptance: the optimum re-solved in CalculiX at 2x refinement changes peak stress by less than
a stated margin and no element exceeds the stress limit; the Pareto front of mass vs peak stress is reported.

## B11 | VORTEX ENGINE: a geometry that learns to mix two streams | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

Two inlet streams, one outlet; objective: minimise the coefficient of variation of the outlet concentration at fixed
pressure drop. Physics: steady incompressible flow with a passive scalar (advection–diffusion), Peclet number sets the
difficulty. Exists: `numerical_solvers.internal_flow`, `tvd_advection`, `fdm`/`fvm`, `stable_fluids`, `lbm`,
`topology` for the geometry variables. Missing: a scalar transport equation wired into the internal-flow solver and a
mixing metric computed on the outlet plane. Baseline: an empty chamber and a static mixer with helical elements of the
same length.

## B12 | FLUID MACHINE: blades the optimiser finds for a rotor | labels: enhancement, experiment, priority: P3, effort: L, design-optimization, solvers, geometry

Parametrised blade rows (stagger, camber, thickness, lean) for a given duty (pressure ratio, flow coefficient),
optimised on efficiency at the design point and checked off-design. Exists: `external_solvers.turbodesigner` (meanline
design), `design.aero` for the cascade side, `external_solvers.openfoam` for the RANS, `auto` DOE/NSGA-II machinery in
`pinneapple_autolab` and `design.aero.optimize`. Missing: a rotating-zone (MRF) case template and a surrogate trained
on the DOE that is honest about extrapolation. Deliverables: the blade geometry, the efficiency vs flow-coefficient
curve, and the comparison with the meanline design at equal solidity.

## B13 | SHAPE-SHIFTING METAMATERIAL: a lattice whose unit cells are chosen for the behaviour it must have | labels: enhancement, experiment, priority: P3, effort: L, fem, geometry, design-optimization

A lattice of unit cells with different topology (auxetic, chiral, compression–twist) assembled so that the global
response (Poisson's ratio, stiffness anisotropy, buckling load) hits a target. Exists: `pinneapple_design.geometry.gen`
(`sdf_shapes`, `primitives`, `domains3d`), `solid_fem`, `plane_stress_fem`, `nonlinear_beam_fem`, `elasticity3d_fdm`,
`bricks` for assembly-level statics. Missing: a unit-cell homogenisation step (compute the effective stiffness tensor
from a cell) and a selection/optimisation over a library of cells under a connectivity constraint. Verification:
homogenised prediction vs a full-lattice solve on a 3x3x3 patch, and the buckling load vs Euler/plate theory.

## B14 | INVISIBLE WEATHER: a 3-D convection cell, and the scaling law it must reproduce | labels: enhancement, experiment, priority: P3, effort: L, solvers, visualization

Rayleigh–Bénard convection in a 3-D box: heat a plate, cool the top, let the pattern emerge; then change the cavity
geometry or the heating distribution and re-measure. Physics: Boussinesq Navier–Stokes + energy, controlled by Ra and Pr;
the quantitative claims are the critical Rayleigh number for onset and the Nusselt–Rayleigh exponent, both known from
theory/experiment. Exists: `geophysics.energy_balance` (planetary balance, not convection), `thermal_channel_2d`,
`lbm`, `ns3d_spectral`, `external_solvers.openfoam` (buoyant solvers). Missing: a buoyant 3-D solver path with a
Nusselt-number measurement and an onset study. Deliverable: the plume render from the computed field.

## B15 | TENTACLE ENGINE: a continuum arm that squeezes through a gap and still lifts | labels: enhancement, experiment, priority: P3, effort: L, fem, design-optimization

A soft/continuum manipulator with a stiffness distribution and a cable or pressure actuation scheme; it must pass
through an opening smaller than its resting diameter and then exert a grasp force. Exists: `multibody.arm`,
`robot_sim` (`arms`, `sim`, `spec`, `learn`, `checks`), `nonlinear_beam_fem`, `mpm`/`sph` for the soft contact side.
Missing: a rod model large enough in rotation for a continuum arm (Cosserat-type), or an honest justification that the
nonlinear beam is the right approximation, and a contact model that does not allow penetration. Metrics: reachable
workspace, peak strain, grasp force, actuator work, and the failure mode when the gap is too small.

## B16 | CLIMATE SKIN: a facade whose panels move, and what that does to the wind around a building | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

A building envelope of actuated panels; the panel angles are the control variable; the objective is to reduce the mean
and peak load on the structure or to increase ventilation in a target region. Physics: external incompressible flow
around a bluff body at building-scale Reynolds numbers, RANS with a stated turbulence model. Exists:
`pipelines.freecad_openfoam` (parametric geometry → mesh → case → sweep → surrogate), `external_solvers.openfoam`,
`design.geometry.*`, `autolab` for the DOE/optimisation loop. Missing: a geometry-morphing loop that remeshes per panel
configuration at affordable cost (or a porous/roughness representation of the panels as the cheap model, with a
validation against the resolved case). Baselines: a flat facade, and the same building with the best single static
panel angle.

## B17 | SKYBRAKE: a capsule that steers its own entry with movable surfaces | labels: enhancement, experiment, priority: P3, effort: L, solvers, uq

Entry trajectory with deployable aerodynamic surfaces; optimise the control schedule for landing dispersion under
limits on deceleration, heat flux and angle of attack. Physics: compressible flow (`compressible_fv`,
`external_solvers.openfoam`), rigid-body 6-DOF (`vehicle.sim`, `flight.rigid_body`), a heating estimate from a
correlation rather than a full chemistry model. The issue must state plainly that continuum CFD degrades as the
Knudsen number rises and that no ablation/chemistry is modeled, so the numbers are conceptual. Deliverable: the
trajectory family, the load/heat envelopes, and the surface-deflection schedule; the "zig-zag" is a consequence of the
optimiser, not an animation choice.

## B18 | BLOOM TURBINE: a rotor whose blades open as the wind rises | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

A small wind rotor with articulated blades; the fold angle and pitch schedule are optimised for annual energy at a
site while respecting a thrust/load limit and a cut-out condition. Exists: `external_solvers.openfoam` (moving-mesh /
AMI needed), `design.aero` for the blade sections, `design.geometry.gen.airfoil`, `autolab` for the campaign.
Missing: a blade-element-momentum model (cheap, defensible) and a moving-mesh CFD template for at least one
configuration to check the BEM. Metrics: power coefficient vs tip-speed ratio for each fold state, thrust, and the
load reduction the fold buys at the same energy.

## B19 | VORTEX RIDER: thrust from vortex rings, measured rather than implied | labels: enhancement, experiment, priority: P3, effort: L, solvers, visualization

A pulsed jet that forms vortex rings and a body that reacts to them; objective: impulse per unit energy and turning
capability. Physics: transient incompressible flow with a moving body (`sph`, `lbm`, `pimpleFoam`), thrust from the
momentum flux through a control surface (not from a formula applied afterwards). Exists: `sph`/`isph`/`dfsph`,
`lbm_tunnel`, `external_solvers.openfoam`, `vehicle.sim` for the body. Missing: the control-surface momentum accounting
and a ring-circulation estimator. The issue must include the sanity bound: the mean thrust cannot exceed the momentum
flux the nozzle delivers, and the report must show where on that bound the design sits.

## B20 | MECHAMORPH: a machine that walks into a gap and turns itself into a bridge | labels: enhancement, experiment, priority: P3, effort: L, fem, design-optimization, agents

A legged/articulated structure that locomotes, then locks its joints and becomes a load-bearing span. Physics:
multibody kinematics plus contact for the walking phase, then a locked configuration analysed as a structure with the
same members. Exists: `multibody.tree`, `robot_sim`, `terramechanics`, `bricks.analyse_assembly`, `solid_fem`.
Missing: a joint-locking model (locked joints transmit moment; free joints do not) and a single optimisation that
scores both phases. Metrics: traverse time, energy, locked-span deflection under a defined load, and the mass penalty
of the locking mechanism. Baselines: a fixed truss of the same mass carried to the gap.

## B21 | VORTEX GATE: can a flow field really deflect a passing object? | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

An annular nozzle array creates a rotating flow region; a body passes through it; measure the lateral force and the
trajectory change. Physics: transient incompressible flow plus 6-DOF body coupling (`sph`, `lbm`,
`external_solvers.openfoam`, `flight.rigid_body`), and the momentum-flux bound on achievable side force. The point of
the issue is the feasibility calculation, not the visual: the report must show the ratio of available aerodynamic force
to the body's weight for a stated body and speed, and the verdict follows from that ratio. Existing pieces:
`immersed_boundary_fdm`, `cylinder_wake`, `external_flow`. Missing: two-way coupling and the force-control-surface
accounting.

## B22 | ORBITAL INFLATION: a compact module that expands into a rotating ring | labels: enhancement, experiment, priority: P3, effort: L, fem, uq, visualization

A stowed structure that inflates/deploys into a ring, stiffens, then spins up to provide apparent gravity; the
optimisation is over the deployment sequence and the spin-up profile so that no member is overstressed and the ring
stays controllable. Physics: membrane/shell deployment (no shell element today — the gap must be stated), rigid-body
rotordynamics, centripetal load, and the comfort constraints (rotation rate, gravity gradient across the body). Exists:
`nbody_rebound` and `pinneapple_physics.astro` for the orbital side, `solid_fem` for a coarse structural check,
`twin3d` for the geometry. Deliverables: the deployment timeline, the stress/deflection at the worst instant, the
apparent-gravity map in the ring, and the honest note that this is centrifugal acceleration in the rotating frame.

## B23 | AEROMORPH: a race car that changes its own aerodynamics for the corner it is in | labels: enhancement, experiment, priority: P2, effort: M, design-optimization, solvers, visualization

A lap is discretised into segments with different needs (drag on the straight, downforce in the corner, cooling
everywhere); the aero configuration (wing angles, ride height, diffuser) is chosen per segment; lap time is the
objective, computed with a point-mass or simple 3-DOF vehicle model. Exists: `pinneapple_autolab`
(`AutonomousLab`, `domains.racecar.RaceCarAero`, `flagship_mission()`, `dashboard.build_dashboard`), `design.aero`,
`external_solvers.openfoam`, `vehicle.sim`. Missing: the per-segment configuration selection on top of the existing
mission loop, and the actuator bandwidth/energy model that makes a "free" morphing car impossible. Baseline: the best
single fixed configuration for the same lap — the deliverable is the delta, not the shape.

## B24 | HYDROFLIGHT: a hull that leaves the water and flies on foils | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

Transition from displacement to foil-borne motion: resistance vs speed, take-off speed, foil incidence schedule,
lateral stability in a turn. Physics: hydrofoil lift/drag from a lifting-line or panel method plus a wetted-area
resistance model, with a free-surface-coupled CFD check on one configuration; 6-DOF with the wetted-surface state.
Exists: `design.aero.panel`/`vlm` reusable for the hydrofoil sections, `vehicle.sim`, `free_surface`,
`shallow_water_fv` (insufficient for this — say so). Missing: a foil–free-surface solver path; the honest plan is
lifting-line + empirical free-surface correction as v1 and one OpenFOAM multiphase case as the check. Metrics:
resistance curve, take-off speed, stability margin, and the disagreement between v1 and the CFD check.

## B25 | INVERTED WORLD: a vehicle that drives on a ceiling, and what holds it there | labels: enhancement, experiment, priority: P3, effort: M, solvers, geometry

Surface-to-surface transitions (floor → wall → ceiling → curved corner) with an explicit adhesion mechanism
(magnetic, suction or dry adhesive) whose force budget is modelled, not assumed. Exists: `terramechanics`
(Bekker–Wong), `rigid_body`, `vehicle.sim`, `robot_sim`, `mujoco`/`newton_physics` bridges. Missing: an adhesion model
with a measured-style limit surface (normal + shear), and a gait planner that respects it. Metrics: maximum traversable
curvature, adhesion force margin, slip at the corner, and the failure mode when the margin goes negative.
