# Backlog — series D (speculative technologies, fiction-to-reality)

Format: `## ID | Title | labels: ...` then the concept notes the issue body must expand.

Every D issue must carry, in its `## Realizability and evidence class` section: the `PLAUSIBILITY_CLASS` it holds
today (`established|extrapolated|hypothetical|fictional`), which of the six evidence axes a simulation can actually
fill, and a ranked list of what would have to be true — material, actuator, energy, control, measurement — for the
promise to move up one class. The verdict is the deliverable as much as the render is: several of these should end in
`CONTRADICTED` or `NOT FEASIBLE`, and that is a successful outcome, not a failure of the experiment.

## D1 | SHAPE-SHIFTING MATTER: one population of modules, several machines | labels: enhancement, experiment, priority: P3, effort: L, design-optimization, agents, fem

Units reorganise into a two-wheeler, then a quadrotor frame, then a manipulator arm. Physics: module geometry and
connector strength set the achievable shapes; the assembled object is checked structurally and, for the flight case,
aerodynamically. Exists: `bricks` (assembly statics and search), `topology`, `solid_fem`, `design.aero` for the rotor
frame, `particle_dynamics.rigid_body` for the units. Missing: a connector model with strength and stiffness, a
transformation planner (which module goes where, in what order, without collisions) and a mass/power budget that makes
the three objects a trade-off rather than a wish. Acceptance: for each of the three targets, the assembled object
carries its stated load and the transformation completes; the report states the module count at which each target
fails. Cross-reference C1 (MORPHOGENESIS) for the assembly engine and D24 (OMNIMATTER) for the intent interface.

## D2 | T-1000 LAB: a liquid-metal machine, and the physics that refuses it | labels: enhancement, experiment, priority: P3, effort: L, solvers, uq

Liquid metal that splits into limbs, holds a shape and reforms, as in the film. Physics, in order of how much it hurts:
surface tension dominates at any scale where a droplet is small (capillary length `sqrt(γ/ρg)` for a metal is a few
millimetres); a free liquid cannot sustain a bending moment; shaping requires a body force (magnetic, electric,
acoustic) whose magnitude must be compared with the capillary pressure; an oxide skin changes the rheology. Exists:
`electromagnetics.statics`/`eddy`/`hysteresis` for the field, `sph`/`isph` and OpenFOAM `interFoam` for the free surface,
`free_surface`. Missing: a magnetohydrodynamic free-surface solver with surface tension, and a metric that separates
"the shape is achieved" from "the shape does work". Expected outcome: `CONTRADICTED` as portrayed, `CONDITIONAL` for
small-scale shaping without load — the issue should predict this and then show it, or show why not.

## D3 | AUTONOMOUS WORLD BUILDER: robots that assemble a station without a foreman | labels: enhancement, experiment, priority: P3, effort: L, agents, orchestration, solvers

Many units ferry and attach modules in orbit under a coordination policy; the objective is assembly time and total
propellant, subject to no collision and a stable intermediate structure at every step. Physics: rigid-body orbital
dynamics (`nbody_rebound`, `pinneapple_physics.astro`), contact/berthing as a discrete event, and a structural check of
each intermediate configuration. Exists: `robot_sim`, `multibody`, `bricks.analyse_assembly`, `pinneapple_autolab`
(`Mission`, `Simulator`, `choose_next`) for the planning loop. Missing: a multi-agent planner, a berthed-joint model,
and the "stable at every intermediate step" constraint, which is where most real assembly plans fail. Deliverable: the
assembly plan, the instability that first appears and the plan that avoids it.

## D4 | INSTANT EXOSUIT: armour that folds out of a pack and takes the load | labels: enhancement, experiment, priority: P3, effort: L, fem, design-optimization

A deployable exoskeleton that opens around a body model, locks joints and routes actuator torque so that a lifted load
produces acceptable joint moments. Physics: multibody kinematics with joint limits, a torque budget per actuator, and a
structural check of the deployed frame under the lift load. Exists: `multibody`, `robot_sim`, `solid_fem`, `topology`
for the frame, `vehicle`-style mass budgeting. Missing: an ergonomic load criterion (what joint moments and contact
pressures are acceptable) and a deployment kinematics with a stowed geometry that fits a stated pack. The report must
say that no human safety claim is made: the numbers are mechanical, not physiological.

## D5 | OMNIVEHICLE: one machine for air, surface and water, with the trade-off shown | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

A vehicle that flies, floats and dives by reconfiguring. Physics: a mass and buoyancy budget that closes in all three
media, aerodynamic and hydrodynamic models (panel/VLM and lifting-line as the cheap pair, CFD as the check), and 6-DOF
in each medium with the transition states included. Exists: `design.aero`, `flight` (6-DOF, trim, turbulence),
`vehicle.sim`, `sph`/`lbm`, `external_solvers.openfoam`. Missing: the transition modelling (a body crossing the surface)
and a multi-objective optimisation whose output is the trade-off surface, not a single winning design. The issue must
state that flying and diving requirements conflict and that the experiment's value is quantifying the conflict.

## D6 | FUELLESS DEEP SPACE CRAFT: a sail, its trajectory, and the patience it needs | labels: enhancement, experiment, priority: P3, effort: M, solvers, design-optimization

A solar sail changes orbit by radiation pressure; the experiment optimises sail orientation history for a stated
transfer and reports the achieved change versus a propulsive reference. Physics: radiation pressure `2P/c` (reflectance
dependent), on a large thin membrane, integrated with two-body + J2 dynamics; structural wrinkling limits the optical
model. Exists: `nbody_rebound`, `pinneapple_physics.astro.conjunction.propagate_two_body_j2`, `closed_form.rocket_nozzle`
for the propulsive reference, `solid_fem` for a coarse membrane stress check. Missing: a sail force model as a function
of attitude and wrinkle state, and a trajectory optimiser over the attitude history. The issue must include the honest
scale: the acceleration is in the 0.1–1 mm/s² class, so the transfer takes months and the result is a trajectory, not a
manoeuvre.

## D7 | GRAVITY-FREE PLATFORM: a stage that holds a load without touching it | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

A magnetically levitated stage carries a payload and holds position against a disturbance; the design variables are the
magnet/coil layout and the controller. Physics: magnetostatic field and force, eddy-current damping, the Earnshaw
instability (the loop is mandatory), and a power/thermal budget for the coils. Exists: `electromagnetics.statics`,
`eddy`, `hysteresis`, `materials`, `machines`, `eddy_current_fdm`, `machine`; plus linear control design on a state-space
model. Missing: a force–current–gap characteristic calibrated against the field solution and a closed-loop stability
margin analysis under saturation. Deliverable: the levitation stiffness, the disturbance rejection bandwidth, and the
gap/load range; cross-reference D15 (INERTIA CONTROL LAB) for the isolation question.

## D8 | WALKING INDUSTRIAL MEGASTRUCTURE: a factory that steps over the terrain | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

A very large legged platform traverses uneven ground and stays level enough to keep working. Physics: multibody
dynamics with contact, soil bearing capacity and sinkage (`terramechanics`, Bekker–Wong), ground pressure limits, and
stability against tipping on a slope. Exists: `terramechanics`, `multibody.tree`, `robot_sim`, `rigid_body`,
`mujoco`/`newton_physics` bridges, `vehicle.viz`. Missing: a gait planner for a statically stable six/eight-legged
machine at this scale and a soil-interaction model with the pressure limits that actually govern the design. Metrics:
ground pressure (kPa), energy per metre, maximum slope, tipping margin, and the terrain class where it stops working.

## D9 | ACTIVE CAMOUFLAGE SKIN: a surface that matches what is behind it, from a stated viewpoint | labels: enhancement, experiment, priority: P3, effort: M, visualization, design-optimization

A panel array whose local reflectance/emission is optimised so that the rendered appearance of the object matches the
background from a defined set of viewpoints and wavelengths. Physics: radiometry and BRDF, not a texture animation; the
metric is the image difference between the object and the background under stated illumination. Exists:
`pinneapple_blender.cycles` (`CyclesScene`, materials, `render`, `render_frames`) as the renderer, `twin3d` for the
geometry, `pinneapple_analysis.validation.error_map` for the comparison, and the optimisers in `design`. Missing: an
optical model with an explicit BRDF parameterisation, a viewpoint/illumination sampling scheme, and the statement that
this is appearance matching for a defined observer — not invisibility. The issue must quantify the failure outside the
optimised viewpoints (that number is the result).

## D10 | SONIC CLOAK: an acoustic metamaterial that hides a region at one frequency band | labels: enhancement, experiment, priority: P3, effort: M, solvers, design-optimization, fem

A shell of structured material reduces the scattered acoustic pressure from a target object over a stated band.
Physics: Helmholtz equation with density/stiffness distribution, transmission loss and scattering cross-section as
metrics; the band limitation is the whole story (a passive linear cloak is narrow-band and often requires
anisotropic or extreme parameters). Exists: `pinneapple_analysis.inspection.acoustic` (1-D wave FDTD for NDT) and
`phased_array_ultrasonic`, which are the wave-propagation precedents, plus `pinneapple_design.topology` for the material
distribution and `electromagnetics.waves` for the analogous formulation. Missing: a 2-D/3-D Helmholtz solver with
radiation boundary conditions, and an optimisation over material parameters with physical bounds. Acceptance: the
scattering reduction is reported as a function of frequency, with the band edges and the parameter values required; if
the required density is negative, the issue says so.

## D11 | REGENERATIVE STRUCTURE: a crack, a repair, and how much of the strength comes back | labels: enhancement, experiment, priority: P2, effort: M, fem, uq

A member with a crack is repaired by an activated material; the experiment measures the recovered stiffness and
strength and the new stress concentrations the repair introduces. Physics: linear elastic fracture mechanics (`K_I`,
`K_IC`), a repair patch with its own modulus and bond strength, and a before/after stress analysis. Exists:
`solid_fem`, `plane_stress_fem`, `elasticity3d_fdm`, `bricks` for the member network, `pinneapple_reason` for the
failure-mode narrative, `pinneapple_analysis.uncertainty.margin` for the tolerance on the repaired properties.
Missing: a fracture module (no crack-tip field today) and a bond/interface element. Acceptance: repaired strength as a
percentage of the intact member, with the load path that limits it; and one case where the repair shifts failure
elsewhere, since that is what actually happens.

## D12 | EARTHQUAKE SHIELD: a foundation whose stiffness changes during the shaking | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

A building on adaptive isolation: the isolator stiffness/damping schedule is optimised over a set of ground motions to
cut peak floor acceleration. Physics: wave propagation in the soil column plus an MDOF building model, excitation from
recorded accelerograms. Exists: `pinneapple_data.datasets.external_sources` includes `designsafe` (earthquake and
structural data), `numerical_solvers` (`fdm`, `beam_fe`, `plane_stress_fem`), `pinneapple_analysis.uncertainty.margin`
for the design margins. Missing: a soil–structure interaction model with a layered soil column and a control law that
can actually be implemented by a real isolator. Metrics: peak floor acceleration, inter-storey drift, isolator
displacement, and the robustness across motions not used in the optimisation. The report must state that passive
isolation is established engineering and that only the adaptive part is the novelty.

## D13 | WORMHOLE LAB: light through a throat that may not exist | labels: enhancement, experiment, priority: P3, effort: M, solvers, visualization

Ray-trace null geodesics through a Morris–Thorne-type metric and show what a distant observer would see: multiple
images, Einstein rings, the shadow of the throat. Physics: geodesic equations from the metric, integrated numerically;
the metric's parameters are design inputs, and the energy-condition violation at the throat is computed and reported.
Exists: `numerical_solvers` ODE integrators (`irk_gauss_legendre`, `xtfc_ivp`, `fdm`), `twin3d` and
`pinneapple_blender.cycles` for the image, `pp.futures` for the assessment. Missing: a geodesic integrator with an
adaptive step near the throat, a ray-marching renderer, and a metric module that keeps units and constants explicit.
The issue must state the boundary of the claim: solving a metric is mathematics; it says nothing about whether such a
metric can be produced.

## D14 | ARTIFICIAL GRAVITY GENERATOR: a ring that spins, and the comfort it can and cannot buy | labels: enhancement, experiment, priority: P3, effort: M, fem, uq, visualization

A rotating habitat: choose radius and rotation rate for a target apparent gravity, then measure what it costs —
gravity gradient across the body, Coriolis effects on moving objects, structural mass, and the rotation rate below
which most people adapt (about 2 rpm is the usual comfort limit). Physics: rigid-body rotation, centripetal
acceleration `ω²r`, ring hoop stress, and the gradient `Δg/g ≈ 2h/r`. Exists: `solid_fem` for a coarse ring stress,
`twin3d` for the geometry, `pinneapple_physics.astro` for the orbital context, `vehicle`/`multibody` for a moving
object in the rotating frame. Missing: a rotating-frame dynamics model with the Coriolis trajectory of a thrown object
(the demonstration that lands with an audience) and a mass estimate for the ring. Verdict: the principle is
established; the issue is engineering and physiology, and the report must not claim that spin gravity is "solved".

## D15 | INERTIA CONTROL LAB: making a payload feel less of a manoeuvre | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

A platform with internal moving masses and dampers reduces the acceleration transmitted to a payload during a
prescribed manoeuvre. Physics: multibody dynamics, base excitation, active/passive isolation and its stroke limits; the
conservation laws bound what is possible and the issue must compute that bound. Exists: `multibody`, `rigid_body`,
`pinneapple_analysis.uncertainty.margin`, control design on a state-space model, `vehicle.sim` for the manoeuvre
profile. Missing: an optimisation over the internal-mass schedule with stroke and power constraints, and a metric that
is honest (peak payload acceleration and settling time versus the unmitigated case). The issue must state explicitly
that inertia is not cancelled: only the transmitted acceleration is reduced, and momentum still goes somewhere.

## D16 | PORTAL ENGINE: two rooms joined at a boundary, and what the fields do there | labels: enhancement, experiment, priority: P3, effort: M, solvers, visualization

Two separated 3-D domains are identified along matching boundary patches, so an object, a flow or a heat flux entering
one leaves the other. Physics: the boundary condition problem — velocity, pressure, temperature and traction continuity
across an identified pair of patches; the field solution is standard once the identification is imposed, which is what
makes this an honest geometry/BC experiment rather than a physics claim. Exists: `external_solvers.openfoam` (cyclic
and mapped patches exist in OpenFOAM), `numerical_solvers` for an in-house FDM/FV version, `twin3d` for the two rooms.
Missing: a general identified-boundary implementation in our own discretisation, and the energy accounting for what
crosses. Baselines: the same two rooms with a conventional duct between them — the comparison is the deliverable.
`PLAUSIBILITY_CLASS`: `hypothetical` for a traversable wormhole, `established` for the boundary-value problem; the issue
must keep those two claims apart in the same document.

## D17 | MATTER TRANSPORTER: what it costs to reconstruct an object from its description | labels: enhancement, experiment, priority: P3, effort: M, data, benchmark

Take an object, encode its description (geometry + material state per voxel), transmit the description, reconstruct it,
and measure the error and the information bill. Physics/maths: information content per atom, the bit budget for a
stated resolution, and the reconstruction error as a function of resolution; no claim about faster-than-light or about
destroying the original. Exists: `design.geometry` (voxel/mesh/sample), `twin3d` for the reconstruction, `zarr` stores
for the payload, `pinneapple_analysis.validation.error_map` for the geometric error. Missing: the encoding/reconstruction
pipeline and the information accounting, both of which are straightforward and are the actual experiment. The result
must include the order-of-magnitude gap to a human body (about 10^28 atoms), so the reader sees the distance rather than
a claim.

## D18 | SPACE FOLDER: a metric that shortens a path, and the price of the deformation | labels: enhancement, experiment, priority: P3, effort: M, solvers, visualization

Compute geodesics in a deformed spatial metric and show that a path which is long in flat space is short in the
deformed one; then compute what the deformation requires (curvature, stress-energy, energy density) with the chosen
field equations. Physics: differential geometry, geodesic equations, and the energy-condition check. Exists: the same
ODE and rendering stack as D13, `twin3d` for the visual, `pp.futures` for the assessment. Missing: a metric family with
tunable deformation, a length-comparison experiment, and the energy accounting. `PLAUSIBILITY_CLASS`: `fictional` as a
technology, `established` as a geometry exercise; the issue must compute the number that makes the distinction concrete
rather than rhetorical.

## D19 | QUANTUM TELEPORTER: a state that moves without a particle moving, and its fidelity | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

Simulate the standard teleportation protocol on a few qubits: prepare, entangle, Bell-measure, apply the classical
correction, and measure the fidelity of the transferred state under noise. Physics: quantum state vectors and density
matrices, depolarising/amplitude-damping channels, and the no-signalling property that makes the protocol useless for
communication. Exists: `pinneapple_quantum` is a package in this repository — check its API and cite only what it
actually provides; if it lacks a circuit simulator, that is the work item. Missing: a state-vector/density-matrix
simulator with channels and a fidelity/entanglement-fidelity estimator, plus a no-signalling check (the reduced state of
the receiver must not change before the classical bit arrives). The issue must state the scope: states teleport, matter
does not, and there is no faster-than-light signalling.

## D20 | TIME DILATION ENGINE: two clocks, two histories, one number | labels: enhancement, experiment, priority: P2, effort: M, solvers, benchmark

Integrate proper time along two trajectories — one staying, one travelling fast and/or deep in a gravity well — and
report the accumulated difference, with a numerical-error analysis. Physics: `Δτ = ∫ sqrt(1 - v²/c²) dt` in flat space,
the Schwarzschild factor `sqrt(1 - 2GM/rc²)` for the gravitational case, and the composition for a real mission profile.
Exists: `numerical_solvers` (RK integrators), `nbody_rebound`/`pinneapple_physics.astro` for the trajectory,
`pp.compare` for method comparison, `closed_form` for the analytic limits. Missing: a relativistic proper-time
integrator that shares the trajectory machinery, and the analytic checks (constant-velocity, circular-orbit, weak-field
limits) that make the number trustworthy. This is the most rigorous of the D series and the acceptance criteria should
include the order of convergence against the closed form in each limit.

## D21 | CAUSALITY LAB: closed timelike curves, and whether the equations admit a consistent history | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

Integrate trajectories in a spacetime that contains closed timelike curves and look for self-consistent solutions,
following the standard treatment in which the local physics is well defined and global consistency constrains the
initial data. Physics: general relativity's causal structure, CTCs in known solutions, and the self-consistency
condition; the issue must say that this is a mathematical exploration of a metric, not a proposal for a machine.
Exists: ODE integration and the rendering stack (D13), `pp.futures` for the epistemic classification. Missing: the
consistency solver and a measure of how constrained the allowed initial data are. Deliverable: the boundary between
"consistent histories exist" and "the data set is empty" as a function of the metric parameters.

## D22 | TEMPORAL LOOP: a system that returns to its own initial state, and the dynamics that follow | labels: enhancement, experiment, priority: P3, effort: S, solvers

A dynamical system with a reset map that sends the state back to a neighbourhood of its start after a fixed time; the
interesting content is the dependence on the reset (fixed point, period doubling, chaos) and the analogy to iterative
learning. Physics/maths: maps with reset, bifurcation analysis, Lyapunov exponents. Exists: `numerical_solvers`
(`reaction_diffusion`, `kuramoto_sivashinsky`, `xtfc_ivp`), `pinneapple_analysis` sensitivity, `pp.dashboard` for the
animation. Missing: the reset-map framework and the bifurcation sweep. The issue must be explicit that this is a
mathematical analogy for loops in narrative and in iterative design, and is not a statement about time travel.

## D23 | CHRONOSCOPE: reconstructing a past state from what is left of it | labels: enhancement, experiment, priority: P3, effort: M, uq, solvers

Given sparse, noisy, incomplete measurements taken later, reconstruct an earlier field (for example the initial
condition of a diffusion or advection problem) and report the uncertainty of the reconstruction. Physics: backward
integration of a parabolic problem is ill-posed, so the experiment is about regularisation, the information that is
genuinely lost, and the honest confidence interval. Exists: `pinneapple_analysis.inverse_problems` (identifiability,
contracts), `uncertainty`, `adaptive_learning`-style next-observation selection, `fdm`/`fv` solvers,
`pinneapple_analysis.validation.error_map`. Missing: an adjoint/backward solver with regularisation and a
posterior-sampling step. Acceptance: the reconstruction error is reported as a function of the number of sensors and
the noise level, with the point at which the answer stops depending on the data.

## D24 | OMNIMATTER: speak a part, and matter arranges itself into it | labels: enhancement, experiment, priority: P3, effort: L, agents, llm, design-optimization, fem

A sentence describes a part ("a bracket that holds 20 kg with a 200 mm span"); the system turns the intent into a
geometry, plans how modular units assemble it, checks the assembled part mechanically, and reports whether it is
feasible, how long it takes and what is missing. Physics: statics of the assembled part, connector strength, and a
mass budget. Exists: `pp.design(text)` and `pinneapple_design.intent` (goal → verified design or a refusal),
`bricks` for assembly statics, `pinneapple_autolab` for the mission loop, `pinneapple_reason` for the failure-mode
explanation, `llm` nodes for the language side. Missing: the assembly planner (unit placement and sequence) and the
connector/strength model. The interface is the language; the physics is the referee — the issue must require that an
impossible request produces a refusal with the reason, not a plausible-looking object.

## D25 | BIG HERO SWARM: a swarm that loses a fifth of its units and keeps holding the object | labels: enhancement, experiment, priority: P3, effort: L, agents, solvers, uq

Units cling together to form a structure that carries a load; part of the structure is destroyed mid-task and the
remaining units reorganise. Physics: per-unit motion and adhesion budget, global stability of the assembled form, and
the reconfiguration path (no collapse at any intermediate step). Exists: `particle_dynamics.rigid_body`,
`bricks.analyse_assembly`, `multibody`, `pinneapple_autolab` (`assess_candidates`, `choose_next`) for the coordination
loop. Missing: a distributed coordination algorithm, a damage model that removes units physically, and the stability
metric at every intermediate state. Acceptance: the load is still carried after a stated fraction of units is removed,
or the report says at which fraction it fails — either is a result.

## D26 | THOUGHTFORM: from a described object to a structure that can actually be built | labels: enhancement, experiment, priority: P2, effort: M, agents, llm, design-optimization

The honest, measurable version of "imagine it and it materialises": a set of natural-language object descriptions is
converted to geometry, checked structurally and against manufacturing limits, and the success rate is reported as a
benchmark. Exists: `pp.design` / `pinneapple_design.intent`, `design.geometry` and its generators, `topology` for
material distribution, `solid_fem` for the check, `pinneapple_lab` for the manifest, and the benchmark harness in
`pinneapple_analysis.benchmark`. Missing: a fixed prompt set with acceptance criteria, a scoring rubric (did it parse
the intent, did the object pass the physics check, was it buildable), and the comparison against a human CAD answer for
a small subset. The deliverable is the score, not a demo; if the success rate is low, the issue reports it.

## D27 | REACTIVE ARMOR: a layer that stiffens only where it is hit | labels: enhancement, experiment, priority: P3, effort: L, fem, solvers, uq

A protective layer whose local stiffness or damping switches on impact; compare the transmitted impulse and the peak
pressure to a passive armour of the same areal mass. Physics: elastic wave propagation in a layered medium with a
switching modulus, an impact load, and energy absorption; the comparison at equal mass is the whole experiment. Exists:
`elasticity3d_fdm`, `plane_stress_fem`, `beam_fe`, `decomposition`-style spectral analysis, `pinneapple_analysis.uncertainty`.
Missing: an impact/transient dynamic solver with a rate- or pressure-dependent constitutive law, and a pass/fail criterion
(a protected-object acceleration or pressure threshold). The issue must state that this is a simulation of a mechanism,
not a materials qualification; the sensing and switching hardware is out of scope.

## D28 | UNIVERSAL TOOL: one module set, three tools, one force budget | labels: enhancement, experiment, priority: P3, effort: M, design-optimization, fem

Modules reorganise into a wrench, a gripper and a gear; for each, the experiment checks the function that matters —
transmitted torque, grip force, tooth load — and reports the mass and the time to reconfigure. Physics: kinetostatic
force transmission, contact, and member strength. Exists: `multibody`, `robot_sim`, `bricks`, `solid_fem`, `topology`
for the member layout. Missing: a contact/force-transmission model per tool and a single optimisation that scores all
three functions on the same module inventory. The result must include the tool where the modules fail, because that
bounds how universal the set can be.

## D29 | BAYMAX ENGINE: a soft body that changes shape to reach and to carry | labels: enhancement, experiment, priority: P3, effort: L, solvers, fem

A soft, inflatable structure that deforms to reach into a confined space, grasp a delicate object and carry a load;
the optimisation chooses the body shape and the actuation pressure profile. Physics: large-deformation soft body with
pressure actuation, contact with a compliant object, and a pressure/force limit. Exists: `mpm`, `sph`/`isph`,
`nonlinear_beam_fem`, `multibody`, `robot_sim`. Missing: an inflatable/pressure-actuated soft body model (pneumatic
artificial muscle behaviour) and a contact model that respects the object's fragility limit. Metrics: reachable
volume, grasp force, peak contact pressure on the object, and the pressure/energy needed; the report must include the
failure mode when the object is too fragile for the available pressure.

## D30 | SELF-BUILDING CITY: local rules that end in a structure, or in a collapse | labels: enhancement, experiment, priority: P3, effort: L, agents, orchestration, fem

Small robots ferry and place parts to build a bridge, a tower or a shelter without a central plan; the experiment
reports whether the result stands, how long it took, and how much material was wasted. Physics: agent motion and
carrying capacity, discrete placement events, and a structural check after every placement (an intermediate structure
that cannot stand is invalid). Exists: `robot_sim`, `multibody`, `bricks.analyse_assembly`/`evolve`, `terramechanics`
for the ground, `pinneapple_autolab` for the coordination. Missing: the local-rule policy, the placement sequencing
with support constraints, and the incremental stability analysis. Acceptance: the structure stands, and the report
identifies the first placement after which the "plan" would have collapsed without a correction.

## D31 | MEDIBOTS: particles navigating a vessel to a target | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

Micron-scale carriers in a vessel branch toward a target under a steering field, against flow and diffusion; the
metrics are targeting efficiency, time to target and the fraction lost. Physics: Stokes flow in a branching channel,
particle advection plus Brownian diffusion, magnetic steering force, and the margination effects that decide where
particles actually go. Exists: `numerical_solvers.internal_flow`, `lbm`, `particles`, `electromagnetics.statics`,
external `openfoam`. Missing: a particle–flow coupled transport with diffusion and a targeting metric. The issue must
state the biomedical scope limit: no drug release, no biocompatibility, no in-vivo claim — only transport.

## D32 | FORCE FIELD LAB: moving an object without touching it, with the mechanism named | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

Pick one real trapping mechanism (magnetic, acoustic, optical, electrostatic) and compute the trapping force and
stiffness for an object of a stated size; then report the range of sizes where the trap works and the energy deposited
in the object. Physics: the force scaling of each mechanism (`F ∝ a³ ∇B²/μ0` for a magnetic bead, radiation pressure and
gradient force for optical tweezers, Gor'kov for acoustic), which is what makes the answer quantitative. Exists:
`electromagnetics.statics`/`waves`, `pinneapple_analysis.inspection.phased_array_ultrasonic` (acoustic field synthesis),
`sph`/`isph` for the fluid side. Missing: a trap-stiffness computation per mechanism and a comparison table. The issue
must produce the table: which mechanism, which object size, how many newtons.

## D33 | ENERGY SHIELD: a barrier that attenuates a blast or a wave instead of stopping it | labels: enhancement, experiment, priority: P3, effort: L, fem, solvers, uq

A structured barrier (metamaterial or layered) reduces the impulse transmitted to a protected region; the comparison is
against a solid wall of the same mass and against no barrier. Physics: transient elastodynamics/acoustics in a
heterogeneous medium, impedance mismatch, and the transmitted impulse/peak pressure as metrics. Exists: `beam_fe`,
`plane_stress_fem`, `elasticity3d_fdm`, `analysis.inspection.acoustic` (1-D wave), `topology` for the material layout,
`external_solvers.calculix`/`openradioss` for an explicit dynamics check. Missing: a 2-D/3-D transient wave solver with
an absorbing boundary and the impulse metric. The issue must state the band/impulse range where the barrier works and
the mass penalty; a universal shield is not the claim.

## D34 | MIRROR DIMENSION: an environment whose geometry is consistent and impossible | labels: enhancement, experiment, priority: P3, effort: M, visualization, solvers

A 3-D environment built from identified boundaries so that walking in one direction returns you to a place that is
elsewhere, and light and gravity behave coherently in the mapped space. Physics: identified-boundary geometry (as D16),
geodesics for light, and rigid-body dynamics inside the transformed space; the requirement is that there is no
"because it is magic" step — every path is computable. Exists: `twin3d` (scene, viewer, glb), `pinneapple_blender`,
`numerical_solvers` for the trajectories. Missing: the boundary identification in the physics/visualisation layer and a
navigable consistency check (a loop must return the same state). Deliverable: the loop closure test, plus the render.

## D35 | HARD-LIGHT CONSTRUCTOR: how much force light can actually apply | labels: enhancement, experiment, priority: P3, effort: S, visualization, uq

Compute the radiation-pressure force on a structure formed by light (optical binding, optical lattice) and compare it
with the weight of any macroscopic object; then report the scale at which it becomes interesting (trapped
micro-particles). Physics: momentum flux `P/c`, gradient forces, and the optical binding potential. Exists:
`electromagnetics.waves`, `mie`, `media`, `statics`, plus `pp.closed_form` for the order-of-magnitude check.
Missing: a force computation for a structured light field and the scaling table. Expected outcome: `CONTRADICTED`
for macroscopic rigid structures, `established` for optical tweezers; the value of the issue is the number (newtons
per watt) and the honest statement of what would have to change.

## D36 | VORTEX SHIELD: deflecting dust and droplets with a flow instead of a wall | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

A device creates a flow that turns away particles approaching a protected region; effectiveness depends on the particle
Stokes number, which is the scientific content. Physics: continuous flow field plus particle tracking with drag and
inertia; the capture/deflection efficiency as a function of Stokes number, particle size and wind speed. Exists:
`numerical_solvers.immersed_boundary_fdm`/`cylinder_wake`/`external_flow`, `particle_dynamics`, OpenFOAM through
`external_solvers.openfoam`, `sph` for the droplets. Missing: a Lagrangian particle tracker with a deflection metric on
top of the flow solution. The report must include the particle size below which the flow cannot deflect anything
(they just follow it) and the power the device needs.

## D37 | CAUSALITY MANIPULATOR: simulate the consequences before choosing the action | labels: enhancement, experiment, priority: P2, effort: M, surrogate, uq, agents

A decision aid: for a system with a physics model, enumerate candidate actions, propagate them through a surrogate with
uncertainty, rank them, and then verify the chosen action with the expensive solver. The scientific content is the
value of the surrogate (did the ranking survive verification?) and the calibration of the uncertainty. Exists:
`pinneapple_analysis.uncertainty` (propagation, tolerances), `surrogate_inverse` (`fit_surrogate`, `descend`,
`VerificationCard`), `pinneapple_autolab` (`compare_models`, `assess_candidates`, `choose_next`), `pp.dashboard`.
Missing: the action-enumeration layer and a regret metric (how much worse the chosen action was than the best action
found by brute force). Acceptance: the reported regret over a held-out decision set, and the cost saved.

## D38 | IMPOSSIBLE ROOM: a room that cannot exist in ordinary geometry, and stays consistent | labels: enhancement, experiment, priority: P3, effort: M, visualization, solvers

A navigable environment whose local geometry is ordinary but whose global connectivity is not (identified boundaries
again, with gravity and light agreement), used to demonstrate that a locally physical space can be globally strange.
Physics: the same identified-boundary machinery as D16/D34, plus a consistency audit (energy, momentum, path closure).
Exists: `twin3d.viewer`, `pinneapple_blender`, D16/D34 work. Missing: the audit tooling and the navigation test.
Deliverable: the environment, the audit report, and the list of the points where a viewer would notice the trick.

## B26 | WAVE EATER: a floating structure that changes shape to take energy from the waves | labels: enhancement, experiment, priority: P3, effort: L, solvers, design-optimization

A wave-energy converter whose geometry or stiffness is tuned per sea state; the optimisation maximises captured power
over a wave spectrum while respecting structural and mooring loads. Physics: linear wave theory with a two-body
oscillator model, added mass and radiation damping from a boundary-element-style coefficient set, and the relative
motion that drives the power take-off; a free-surface CFD check on one design. Exists: `shallow_water_fv` and
`free_surface` (not sufficient for radiation/diffraction — the issue must say so), `solid_fem` for the structure,
`kuramoto_sivashinsky`-style spectral tools for the wave spectra, `Sweep`/`fit_surrogate` for the design campaign.
Missing: a radiation/diffraction coefficient source (external BEM tool or published coefficients) and the PTO model.
Metrics: capture width ratio, peak mooring load, structural utilisation, and the sea state where the device must stop.
