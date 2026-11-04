# Backlog — series C (programmable matter and nanomaterials)

Format: `## ID | Title | labels: ...` then the concept notes the issue body must expand.

These issues share one infrastructure question and must say so: a particle/matter engine with binding rules, an
assembly planner, and a mechanical check of the assembled part. `pinneapple_futures` already ships a
`programmable_matter` demonstrator, but it is a 1-D conduction problem used to make the epistemics concrete — it is
not matter assembly. Every C issue must state the scale it models (mesoscale units, not atoms) and must not claim that
a simulation demonstrates a nanomaterial that can be manufactured.

## C1 | MORPHOGENESIS: units that assemble themselves into a part, take it apart and build another | labels: enhancement, experiment, priority: P2, effort: L, solvers, design-optimization, agents

Thousands of mesoscale units with local binding rules start dispersed and form a 3-D part (a gear, a gripper), which
is then disassembled and rebuilt as a different part under an external command. Physics: particle dynamics with
short-range binding, excluded volume and a driving field (`pinneapple_simulation.particle_dynamics`: `particles`,
`rigid_body`, `mpm`), and a mechanical check of the assembled part (`pinneapple_design.bricks.analyse_assembly`,
`solid_fem`). The inverse problem is the real content: given a target shape (an STL), find the unit states and the
binding rules that produce it. Missing: a binding-rule engine with breakable links and a state machine per unit, an
assembly sequencing planner, and defect metrics (missing units, wrong links, porosity). This issue must also fix the
honest scope: unit count and capabilities are a design choice, not a claim about what chemistry can do.

## C2 | NANOSWARM: a gripper that emerges from magnetic particles under an applied field | labels: enhancement, experiment, priority: P3, effort: L, solvers, uq

A cloud of superparamagnetic particles under a controlled field forms fingers and a palm, closes on an object and
reconfigures. Physics: dipole–dipole interaction, field gradient force from `pinneapple_simulation.electromagnetics`
(`statics`, `analytic`), drag at low Reynolds number, and the aggregation instability that limits the usable field.
Missing: a particle–field coupled solver with hydrodynamic drag, and a grip-force estimator. The report must include the
scale argument: what force a given particle size and field can transmit, and why a hand-sized, human-strength gripper
made of free nanoparticles is not what the model shows.

## C3 | LIVING LATTICE: one architecture, several mechanical functions, chosen by topology | labels: enhancement, experiment, priority: P3, effort: L, fem, geometry, design-optimization

A lattice of units whose connectivity (which neighbours are joined) is the design variable; the same population must
provide high stiffness, then high damping/absorption, then a compliant path, by reorganising. Physics: linear
elasticity per topology, with stability (buckling) checked; a switch cost and time to reorganise. Exists:
`bricks.Assembly`/`analyse_assembly`/`evolve`, `topology`, `plane_stress_fem`/`solid_fem`, `nonlinear_beam_fem`.
Missing: a connectivity-constrained topology search (units keep their physical positions) and a metric for
"reachability" — can the structure get from configuration A to B without passing through an unstable state. That
reachability requirement is what separates this from plain topology optimisation and must be the stated novelty.

## C4 | NANOTURBINE: a rotor that assembles itself in a fluid and then pumps | labels: enhancement, experiment, priority: P3, effort: L, solvers, uq

Helical blades and a shaft assemble from units in a viscous fluid, then rotate under an external torque and drive a
flow. Physics: low-Reynolds-number hydrodynamics (Stokes), rigid-body rotation, and particle–fluid coupling; the
performance metric is volumetric flow per unit torque. Exists: `particles` (SPH), `isph`/`dfsph`, `rigid_body`,
`lbm`; `external_solvers.openfoam` for a body-fitted check on the assembled geometry. Missing: two-way particle–fluid
coupling at low Re and a stable assembly state (what keeps the blades attached while they turn). The issue must state
the Re range and why a macroscale analogue would not transfer.

## C5 | NANOORIGAMI: a flat sheet that folds itself into a 3-D shape | labels: enhancement, experiment, priority: P2, effort: M, fem, geometry, design-optimization

A plate with programmed hinge lines and eigenstrain (or a stiffness/expansion pattern) folds into a target 3-D shape
when stimulated; the design variable is the layout of hinges and the actuation field. Physics: plate/shell kinematics
with elastic hinges, geometric nonlinearity for large rotations, and a snap-through/bistability check. Exists:
`plane_stress_fem`, `nonlinear_beam_fem`, `beam_fe`, `elasticity3d_fdm`, `design.geometry.gen` (`sdf_shapes`,
`primitives`), `solid_fem`. Missing: a shell/plate element good for large rotation (the honest gap), and a
target-shape objective (distance between the folded surface and the target mesh) with a foldability constraint (no
self-intersection). This is the best entry point of the whole C series because it can be validated at bench scale
before any nanoscale claim, and the issue must say that.

## C6 | FERROFLUID ARCHITECT: shapes a magnetic field can command, and the ones it cannot | labels: enhancement, experiment, priority: P2, effort: M, solvers, uq

A ferrofluid surface under a variable field: the Rosensweig instability, spikes, rings, and a target 3-D shape to be
approached by optimising the field. Physics: incompressible flow with a magnetic body force, surface tension and gravity;
the linear stability analysis gives the critical magnetisation and the most unstable wavelength, and those are the
reference results the simulation must reproduce before any shaping claim. Exists: `electromagnetics.statics`/`analytic`
for the field, `electromagnetics.materials` for the magnetisation law, `sph`/`isph` and `free_surface` for the interface,
OpenFOAM `interFoam` through `external_solvers.openfoam`. Missing: a coupled magnetic–capillary free-surface solver
in-process, and a shape-error metric. The issue should state that the phenomenon is real and used in seals and dampers,
so the experiment is about control of the shape, not about whether ferrofluids exist.

## C7 | GRANULAR CONSTRUCTOR: a pile of grains that becomes a bridge and then an arch | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq

Grains with controlled shape/friction assemble under vibration or a magnetic/electrostatic field into a load-bearing
structure, then reorganise when the load path changes. Physics: discrete element method with contact and friction
(Coulomb), granular jamming, and the force chains that carry the load. Exists: `terramechanics` (Bekker–Wong, soil
wheels, not DEM), `rigid_body`, `mpm`, and the external DEM bridges `external_solvers.geotaichi` and
`external_solvers.genesis`. Missing: a DEM path with the specific inter-particle law this experiment needs and a
stability metric for the assembled structure; the work may legitimately be "wrap the external DEM and verify it",
which the issue should say instead of pretending an in-process DEM exists. Baseline: the same grains poured without the
field/actuation.

## C8 | SELF-HEALING MACHINE: a truss that breaks, and then repairs itself | labels: enhancement, experiment, priority: P2, effort: M, fem, solvers, uq

A truss network whose members can fail and be re-formed by mobile units; a member is removed, load redistributes, the
units migrate and rebuild it, and the structure is re-analysed. Physics: linear elasticity for the intact/damaged/repaired
states, a failure criterion on the members, a mobility and energy budget for the repair, and the load-redistribution
path. Exists: `bricks.Assembly`/`analyse_assembly`, `solid_fem`, `topology`, and `pinneapple_futures.atlas()` marks
"self-healing and adaptive structures" as a **planned** subarea — this issue is what makes it available, which the body
should state. Missing: a damage/re-formation model for the members and a recovery metric (stiffness and strength
recovered per unit of added mass and energy). Acceptance must compare intact / damaged / repaired on the same load case,
and must include a case where repair makes the structure worse, because that is the real result.

## C9 | MICROFACTORY: an assembly line inside a channel | labels: enhancement, experiment, priority: P3, effort: M, solvers, design-optimization

Micron-scale components carried by a low-Re flow meet at designed stations and bind; a tool assembles, transports and
releases. Physics: Stokes flow in a channel with particle transport, short-range binding and a designed obstacle
landscape. Exists: `numerical_solvers.internal_flow`, `lbm`, `particles`, `external_solvers.openfoam`, and the
`pipelines.freecad_openfoam` sweep machinery for the channel geometry. Missing: transport of particles with binding
events, and yield metrics (assemblies completed per unit time, per cent of components wasted, blockage rate). The
issue should report the yield, not the animation of a successful grab.

## C10 | THERMAL MORPH: a part that changes function with temperature | labels: enhancement, experiment, priority: P3, effort: M, fem, solvers

A shape-memory or bimetallic member that assumes one configuration when hot and another when cold, executing a task
(a clamp that grips, a lock that releases) with no motor. Physics: thermoelasticity with a transformation (shape memory)
or a two-material thermal expansion mismatch; the design variables are the layer/geometry and the thermal profile.
Exists: `thermal.heatsink`/`fin_cell` for the thermal side, `thermal_channel_2d`, `plane_stress_fem`,
`nonlinear_beam_fem`, `energy_balance`. Missing: a constitutive model for the transformation and a coupled
thermal–mechanical solve. Verification: the classical bimetal curvature formula `kappa = 6 Δα ΔT (1+m)^2 / (…)` for the
bimetallic case, before any shape-memory claim.

## C11 | ACTIVE MATTER: a crowd of units that organises itself, and the parameters where it stops | labels: enhancement, experiment, priority: P3, effort: M, solvers, uq, benchmark

Active particles that convert energy into motion, with alignment and noise; the experiment maps the order–disorder
transition, the clustering regime and the collapse regime as a function of density, noise and alignment strength, and
compares with the known Vicsek-style results. Exists: `particle_dynamics.particles`, `reaction_diffusion` (as a
continuum analogue), `pinneapple_simulation` neighbour search (`numerical_solvers.neighbors`), and the UQ machinery.
Missing: an active-particle model with the order parameter and susceptibility measurement, and a phase diagram with
error bars. Deliverable: the phase diagram, not a pretty swirl; the swirl is the figure that comes out of it.

## C12 | MOLECULAR LOCK: a cage that closes on a particle and opens on command | labels: enhancement, experiment, priority: P3, effort: L, solvers, uq

A coarse-grained molecular assembly (beads with bonds and non-bonded interactions) forms a cage around a payload and
releases it when the interaction parameters change. Physics: `pinneapple_simulation.particle_dynamics.molecular_dynamics`
(Lennard-Jones + velocity Verlet, differentiable), plus a free-energy estimate for the open/closed states. Exists: the
LJ integrator, differentiable in PyTorch, and the differentiable-physics style of the package. Missing: thermostatting
(Langevin/Nose–Hoover), implicit solvent, and a free-energy method (umbrella sampling or thermodynamic integration) —
without which the open/closed preference is not a number. The issue must state clearly that this is coarse-grained, is
not atomistic, and cannot support a claim about a specific DNA-origami or protein structure.
