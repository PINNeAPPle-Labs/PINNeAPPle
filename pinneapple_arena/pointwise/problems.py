"""
Physics problems catalog for the PINNeAPPle Arena.

Exposes PROBLEMS_CATALOG: dict[str, PhysicsProblem]: 34 problems in 7 physics domains, with their equations,
parameters, boundary/initial conditions and the PINNeAPPle preset key where one exists. The reference data of
each problem is ``pinneapple_arena.pointwise.datasets``.

3D benchmarks: taylor_green_vortex, darcy_flow_3d, stokes_sphere.
"""
from __future__ import annotations

from pinneapple_arena.pointwise.schema import (
    ColumnHints,
    DifficultyLevel,
    Dimension,
    PhysicsArea,
    PhysicsProblem,
    ProblemParameter,
)

# ---------------------------------------------------------------------------
# Column-hint factory
# ---------------------------------------------------------------------------

def _hints(
    coord_cols: list[str],
    output_cols: list[str],
    time_col: str | None = None,
    notes: str = "",
) -> ColumnHints:
    """Build a ColumnHints object; csv_header = all cols joined with comma."""
    all_cols = coord_cols + output_cols
    return ColumnHints(
        coord_cols=coord_cols,
        output_cols=output_cols,
        time_col=time_col,
        csv_header=",".join(all_cols),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _p(
    name: str,
    default: float,
    min_val: float,
    max_val: float,
    description: str,
    unit: str = "",
) -> ProblemParameter:
    return ProblemParameter(
        name=name,
        default=default,
        min_val=min_val,
        max_val=max_val,
        description=description,
        unit=unit,
    )


# ===========================================================================
# FLUID MECHANICS  (8 problems)
# ===========================================================================

burgers_1d = PhysicsProblem(
    id="burgers_1d",
    name="1D Viscous Burgers Equation",
    description=(
        "The viscous Burgers equation is a fundamental nonlinear PDE that combines "
        "convective transport with diffusion. It serves as a canonical benchmark for "
        "shock formation and wave-steepening phenomena, and is widely used to validate "
        "PINN implementations."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.ONE_D,
    equations=[
        "∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²",
        "x ∈ [-1, 1], t ∈ [0, 1]",
    ],
    applications=[
        "Shock wave modeling",
        "Traffic flow theory",
        "Acoustics nonlinear propagation",
        "PINN algorithm benchmarking",
    ],
    initial_conditions="u(x, 0) = -sin(πx)",
    boundary_conditions="u(-1, t) = 0, u(1, t) = 0 (homogeneous Dirichlet)",
    parameters=[
        _p("nu", 0.01, 1e-4, 0.1, "Kinematic viscosity / diffusion coefficient", "m²/s"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset="burgers_1d",
    tags=["viscous", "nonlinear", "time-dependent", "shock", "benchmark"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["u"],
        time_col="t",
        notes="",
    ),
)

navier_stokes_2d = PhysicsProblem(
    id="navier_stokes_2d",
    name="2D Incompressible Navier-Stokes",
    description=(
        "Two-dimensional incompressible Navier-Stokes equations governing viscous "
        "fluid flow. The momentum equations are coupled to the continuity constraint "
        "∇·u = 0, resulting in a nonlinear saddle-point system solved for velocity "
        "(u, v) and pressure p."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u",
        "∇·u = 0",
        "∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + (1/Re)(∂²u/∂x² + ∂²u/∂y²)",
        "∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + (1/Re)(∂²v/∂x² + ∂²v/∂y²)",
        "∂u/∂x + ∂v/∂y = 0",
    ],
    applications=[
        "Channel flow simulation",
        "Aerodynamic lift/drag prediction",
        "Microfluidics",
        "Cardiovascular hemodynamics",
    ],
    initial_conditions="u(x, y, 0) = 0, v(x, y, 0) = 0, p(x, y, 0) = 0",
    boundary_conditions=(
        "No-slip on solid walls: u = v = 0; "
        "Inlet: parabolic velocity profile; "
        "Outlet: Neumann ∂p/∂n = 0"
    ),
    parameters=[
        _p("Re", 100.0, 1.0, 1000.0, "Reynolds number (Re = ρUL/μ)", "—"),
        _p("rho", 1.0, 0.1, 100.0, "Fluid density", "kg/m³"),
        _p("mu", 0.01, 1e-4, 1.0, "Dynamic viscosity", "Pa·s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="ns_incompressible_2d_default",
    tags=["viscous", "incompressible", "nonlinear", "time-dependent", "CFD"],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["u", "v", "p"],
        time_col="t",
        notes="",
    ),
)

navier_stokes_3d = PhysicsProblem(
    id="navier_stokes_3d",
    name="3D Incompressible Navier-Stokes",
    description=(
        "Full three-dimensional incompressible Navier-Stokes equations. Extends the "
        "2D formulation to three spatial dimensions (x, y, z), solving for velocity "
        "components (u, v, w) and pressure p. Requires a 3D geometry definition."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.THREE_D,
    equations=[
        "ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u",
        "∇·u = 0",
        "∂u/∂x + ∂v/∂y + ∂w/∂z = 0",
    ],
    applications=[
        "3D pipe flow",
        "Turbomachinery internal flows",
        "Aircraft aerodynamics",
        "HVAC system design",
    ],
    initial_conditions="u = v = w = 0, p = 0 at t = 0",
    boundary_conditions=(
        "No-slip on all solid walls; inlet velocity profile; "
        "pressure outlet condition ∂p/∂n = 0"
    ),
    parameters=[
        _p("Re", 100.0, 1.0, 500.0, "Reynolds number", "—"),
        _p("rho", 1.0, 0.1, 100.0, "Fluid density", "kg/m³"),
        _p("mu", 0.01, 1e-4, 1.0, "Dynamic viscosity", "Pa·s"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=True,
    pinneapple_preset="ns_incompressible_3d_default",
    tags=["viscous", "incompressible", "nonlinear", "3D", "CFD", "time-dependent"],
    column_hints=_hints(
        coord_cols=["x", "y", "z", "t"],
        output_cols=["u", "v", "w", "p"],
        time_col="t",
        notes="",
    ),
)

taylor_green_vortex = PhysicsProblem(
    id="taylor_green_vortex",
    name="Taylor-Green Vortex (3D Turbulence DNS)",
    description=(
        "The Taylor-Green vortex is the canonical benchmark for 3D turbulence DNS and "
        "PINN/neural-operator validation. An analytically defined velocity field on the "
        "triply-periodic box [0,2π]³ transitions through vorticity cascade, kinetic-energy "
        "dissipation, and small-scale mixing. The exact initial condition is used to train "
        "physics-ML models that predict temporal evolution of velocity, pressure, kinetic "
        "energy, and enstrophy spectra."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.THREE_D,
    equations=[
        "∂u/∂t + u·∇u = −∇p + (1/Re)∇²u  (NS momentum)",
        "∇·u = 0  (incompressibility)",
        "IC: u =  sin x cos y cos z",
        "IC: v = −cos x sin y cos z",
        "IC: w = 0",
        "IC: p = (cos 2x + cos 2y)(cos 2z + 2) / 16",
        "Periodic BC on all three directions: [0, 2π]³",
    ],
    applications=[
        "Turbulence DNS benchmark (Re=1600)",
        "Kinetic energy & enstrophy dissipation prediction",
        "Neural operator surrogate for LES/DNS",
        "Neural operator / PINN 3D turbulence benchmark",
        "Vortex breakdown & cascade validation",
    ],
    initial_conditions=(
        "u₀ = sin(x)cos(y)cos(z), v₀ = −cos(x)sin(y)cos(z), w₀ = 0, "
        "p₀ = (cos 2x + cos 2y)(cos 2z + 2)/16"
    ),
    boundary_conditions="Triply periodic on [0, 2π]³ — no walls",
    parameters=[
        _p("Re", 1600.0, 100.0, 5000.0, "Reynolds number (Re=1600 → turbulent transition)", "—"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset="taylor_green_vortex",
    tags=["turbulence", "DNS", "3D", "periodic", "benchmark", "vortex"],
    column_hints=_hints(
        coord_cols=["x", "y", "z", "t"],
        output_cols=["u", "v", "w", "p"],
        time_col="t",
        notes="",
    ),
)

darcy_flow_3d = PhysicsProblem(
    id="darcy_flow_3d",
    name="3D Darcy Flow — Porous Media (FNO Benchmark)",
    description=(
        "Three-dimensional Darcy flow in a porous medium: −∇·(K∇p) = f on [0,1]³. "
        "This is a canonical benchmark for Fourier Neural Operators (FNO), widely used "
        "for CO₂ geological storage, reservoir simulation, and semiconductor substrate "
        "thermal management. The analytical manufactured solution p = sin(πx)sin(πy)sin(πz) "
        "provides an exact reference with known boundary conditions and source term f = 3π²K·p."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.THREE_D,
    equations=[
        "−∇·(K∇p) = f  (Darcy pressure equation)",
        "u = −K∇p  (Darcy velocity vector field)",
        "f = 3π²K sin(πx)sin(πy)sin(πz)  (manufactured source)",
        "Exact solution: p = sin(πx)sin(πy)sin(πz)",
    ],
    applications=[
        "CO₂ geological storage simulation (100,000× speedup vs. traditional solvers)",
        "Reservoir simulation (Stone Ridge Technology on AWS)",
        "FNO training & benchmarking",
        "Groundwater flow & contaminant transport",
        "Semiconductor substrate pressure distribution",
    ],
    initial_conditions="N/A — steady-state problem",
    boundary_conditions="Homogeneous Dirichlet p = 0 on all six faces of [0,1]³",
    parameters=[
        _p("K", 1.0, 0.01, 10.0, "Permeability coefficient (isotropic)", "m²/(Pa·s)"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="darcy_pressure_only_3d",
    tags=["porous-media", "Darcy", "FNO", "3D", "steady", "reservoir", "CO2"],
    column_hints=_hints(
        coord_cols=["x", "y", "z"],
        output_cols=["p", "u", "v", "w"],
        time_col=None,
        notes="",
    ),
)

stokes_sphere = PhysicsProblem(
    id="stokes_sphere",
    name="Stokes Flow Around a Sphere (External Aero)",
    description=(
        "Exact closed-form (Lamb's) solution for creeping viscous flow (Re → 0) around "
        "a fixed sphere of radius R in a uniform freestream U∞ in the z-direction. "
        "The full 3D velocity field (u_x, u_y, u_z) and pressure p are expressed "
        "analytically. On the sphere surface, the pressure coefficient Cp follows the "
        "inviscid potential-flow distribution Cp = 1 − (9/4)sin²θ, which exhibits the "
        "characteristic front stagnation (Cp=1) and equatorial suction (Cp=−1.25). "
        "Used as a standard external aerodynamics reference and validation benchmark."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.THREE_D,
    equations=[
        "μ∇²u = ∇p,  ∇·u = 0  (Stokes momentum, Re→0)",
        "u_r = U∞(1 − 3R/(2r) + R³/(2r³)) cos θ  (radial component)",
        "u_θ = −U∞(1 − 3R/(4r) − R³/(4r³)) sin θ  (polar component)",
        "p = p∞ − 3μU∞R cos θ/(2r²)",
        "Cp|surface = 1 − (9/4)sin²θ  (potential flow limit)",
    ],
    applications=[
        "External aerodynamics validation benchmark",
        "Automotive drag coefficient prediction",
        "Particle sedimentation in viscous fluids",
        "Aerosol & atmospheric particulate transport",
        "Drug-carrier particle flow in bloodstream",
    ],
    initial_conditions="N/A — steady Stokes flow",
    boundary_conditions="No-slip on sphere u=0 at r=R; far-field u → U∞ê_z as r→∞",
    parameters=[
        _p("U_inf", 1.0, 0.1, 10.0, "Free-stream velocity U∞", "m/s"),
        _p("R", 0.5, 0.05, 2.0, "Sphere radius", "m"),
        _p("Re", 0.1, 1e-4, 1.0, "Reynolds number (must satisfy Re ≪ 1 for Stokes)", "—"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="stokes_sphere",
    tags=["Stokes", "sphere", "external-aero", "3D", "steady", "analytical"],
    column_hints=_hints(
        coord_cols=["x", "y", "z"],
        output_cols=["u", "v", "w", "p"],
        time_col=None,
        notes="",
    ),
)

lid_driven_cavity = PhysicsProblem(
    id="lid_driven_cavity",
    name="Lid-Driven Cavity Flow",
    description=(
        "Classic 2D benchmark where a square cavity [0,1]² is filled with viscous "
        "fluid. The top wall (lid) moves at constant velocity U = 1 m/s, driving a "
        "primary recirculation vortex. Well-characterized reference solutions exist "
        "for Re = 100, 400, 1000 (Ghia et al. 1982)."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u",
        "∇·u = 0",
    ],
    applications=[
        "CFD algorithm validation",
        "Vortex dynamics study",
        "Turbulence onset research",
        "PINN benchmark suite",
    ],
    initial_conditions="u = v = p = 0 everywhere at t = 0",
    boundary_conditions=(
        "Top lid: u = 1, v = 0; "
        "Bottom, left, right walls: u = v = 0 (no-slip)"
    ),
    parameters=[
        _p("Re", 100.0, 10.0, 1000.0, "Reynolds number", "—"),
        _p("U_lid", 1.0, 0.1, 10.0, "Lid velocity", "m/s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["incompressible", "benchmark", "recirculation", "cavity", "viscous"],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["u", "v", "p"],
        time_col="t",
        notes="",
    ),
)

bernoulli_flow = PhysicsProblem(
    id="bernoulli_flow",
    name="Bernoulli Inviscid Flow",
    description=(
        "Bernoulli's principle for steady, inviscid, incompressible flow along a "
        "streamline. The energy equation p + ½ρv² + ρgh = const relates pressure, "
        "kinetic energy, and potential energy. Useful for rapid nozzle/diffuser "
        "design and airspeed estimation."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.ONE_D,
    equations=[
        "p + ½ρv² + ρgh = C  (Bernoulli's equation)",
        "ρv A = const  (continuity, 1D duct)",
    ],
    applications=[
        "Nozzle and diffuser design",
        "Pitot tube calibration",
        "Venturi meter analysis",
        "Hydraulic jump estimation",
    ],
    initial_conditions="v(0) = v_in, p(0) = p_in (inlet conditions)",
    boundary_conditions="Inlet: specified velocity v_in and pressure p_in; outlet: atmospheric pressure p_out",
    parameters=[
        _p("rho", 1.0, 0.1, 1000.0, "Fluid density", "kg/m³"),
        _p("v_in", 1.0, 0.01, 100.0, "Inlet velocity", "m/s"),
        _p("p_in", 101325.0, 1e4, 1e6, "Inlet pressure", "Pa"),
        _p("g", 9.81, 0.0, 20.0, "Gravitational acceleration", "m/s²"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["inviscid", "steady", "1D", "analytical", "compressible"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["u", "p"],
        time_col="t",
        notes="",
    ),
)

stokes_flow = PhysicsProblem(
    id="stokes_flow",
    name="2D Stokes Flow (Creeping Flow)",
    description=(
        "Stokes (creeping) flow regime where inertial effects are negligible compared "
        "to viscous forces (Re << 1). The nonlinear convective term is dropped from "
        "Navier-Stokes, yielding a linear system: -∇p + μ∇²u = 0 with ∇·u = 0."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "-∇p + μ∇²u = 0",
        "∇·u = 0",
        "-∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²) = 0",
        "-∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²) = 0",
        "∂u/∂x + ∂v/∂y = 0",
    ],
    applications=[
        "Microfluidics (Lab-on-a-chip)",
        "Biological cell motility",
        "Lubrication theory",
        "Polymer processing (Re << 1)",
    ],
    initial_conditions="N/A (steady-state problem)",
    boundary_conditions=(
        "No-slip on solid boundaries; specified velocity or pressure at inlet/outlet"
    ),
    parameters=[
        _p("mu", 0.001, 1e-6, 1.0, "Dynamic viscosity", "Pa·s"),
        _p("p_inlet", 1.0, 0.0, 100.0, "Inlet pressure", "Pa"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["viscous", "linear", "steady", "low-Re", "creeping"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["u", "v", "p"],
        time_col=None,
        notes="",
    ),
)

darcy_flow = PhysicsProblem(
    id="darcy_flow",
    name="Darcy Porous Media Flow",
    description=(
        "Darcy's law describes flow through a porous medium: the Darcy velocity is "
        "proportional to the pressure gradient and inversely proportional to viscosity. "
        "Combined with mass conservation, this yields an elliptic PDE for pressure."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "u = -(k/μ) ∇p  (Darcy velocity)",
        "∇·u = f  (mass conservation / source term)",
        "-∇·((k/μ) ∇p) = f",
    ],
    applications=[
        "Groundwater aquifer modeling",
        "Oil and gas reservoir simulation",
        "Filtration system design",
        "Soil consolidation (Terzaghi)",
    ],
    initial_conditions="p(x, y, 0) = p₀ (initial pressure field, for transient variant)",
    boundary_conditions=(
        "Dirichlet (specified pressure) on inlet/outlet boundaries; "
        "Neumann (no-flow) on impermeable walls"
    ),
    parameters=[
        _p("k", 1e-12, 1e-15, 1e-9, "Permeability", "m²"),
        _p("mu", 1e-3, 1e-6, 1.0, "Dynamic viscosity", "Pa·s"),
        _p("phi", 0.2, 0.01, 0.5, "Porosity", "—"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="darcy_flow",
    tags=["porous-media", "elliptic", "subsurface", "linear", "steady"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["p", "u", "v"],
        time_col=None,
        notes="",
    ),
)

channel_poiseuille = PhysicsProblem(
    id="channel_poiseuille",
    name="Poiseuille Channel Flow",
    description=(
        "Pressure-driven laminar flow between two infinite parallel plates (plane "
        "Poiseuille flow). The exact analytical solution is a parabolic velocity "
        "profile u = U_max(1 - (y/H)²). Used as a validation benchmark for 2D NS "
        "solvers and PINNs."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "μ ∂²u/∂y² = ∂p/∂x = -G  (constant pressure gradient)",
        "v = 0",
        "Analytical solution: u(y) = G/(2μ) · (H² - y²)",
    ],
    applications=[
        "Microfluidic chip design",
        "Blood flow in capillaries",
        "PINN validation against analytical solutions",
        "Lubrication bearing analysis",
    ],
    initial_conditions="u = 0 (flow starts from rest)",
    boundary_conditions="No-slip at y = ±H: u = 0; periodic in x",
    parameters=[
        _p("H", 1.0, 0.01, 10.0, "Half channel height", "m"),
        _p("G", 1.0, 0.01, 100.0, "Pressure gradient magnitude", "Pa/m"),
        _p("mu", 1e-3, 1e-5, 1.0, "Dynamic viscosity", "Pa·s"),
        _p("Re", 100.0, 1.0, 2000.0, "Reynolds number", "—"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["analytical", "laminar", "parabolic", "benchmark", "viscous"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["u", "v", "p"],
        time_col=None,
        notes="",
    ),
)

cylinder_flow_2d = PhysicsProblem(
    id="cylinder_flow_2d",
    name="2D Flow Past Cylinder (CFD Benchmark)",
    description=(
        "Classic 2-D incompressible Navier-Stokes flow past a circular cylinder — "
        "the canonical benchmark for validating CFD and PINN solvers against published "
        "reference data (Williamson 1989, Zdravkovich 1997). "
        "The benchmark quantifies drag coefficient Cd, recirculation length Lw/D, "
        "and Strouhal number St across Reynolds numbers Re = 20–180. "
        "Vortex shedding (Kármán vortex street) onset occurs at Re ≈ 47; "
        "PINN solvers are expected to capture both the steady (Re ≤ 40) and "
        "unsteady periodic shedding regimes. "
        "Reference: Raissi, Perdikaris & Karniadakis (2019) and Williamson (1989)."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = −∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²)",
        "ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = −∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²)",
        "∂u/∂x + ∂v/∂y = 0   (incompressibility)",
        "Re = ρ U∞ D / μ",
        "Cd = 2Fd / (ρ U∞² D L)   (drag validation target)",
        "St = f D / U∞             (Strouhal number validation target)",
    ],
    applications=[
        "CFD solver validation and benchmarking",
        "Bridge pier scour and hydrodynamic load estimation",
        "Offshore riser and pipeline vortex-induced vibrations",
        "Heat exchanger tube bundle design",
        "Wind engineering — bluff-body aerodynamics",
        "PINN accuracy benchmarking (Raissi et al. 2019)",
    ],
    initial_conditions=(
        "Steady state for Re ≤ 40: u(x,y) = U∞, v = 0, p = 0 far-field. "
        "Unsteady (Re > 47): perturbed inlet triggers vortex shedding."
    ),
    boundary_conditions=(
        "Cylinder surface: no-slip u = v = 0 (D = 1, centred at origin). "
        "Inlet (x = −8D): u = U∞, v = 0. "
        "Outlet (x = 25D): ∂u/∂n = 0, p = 0 (Neumann outflow). "
        "Top/bottom (y = ±8D): slip ∂u/∂y = 0, v = 0."
    ),
    parameters=[
        _p("Re",    100.0,  20.0, 180.0, "Reynolds number Re = ρ U∞ D / μ", "—"),
        _p("U_inf",   1.0,   0.1,  10.0, "Free-stream velocity", "m/s"),
        _p("D",       1.0,   0.1,   5.0, "Cylinder diameter", "m"),
        _p("rho",     1.0,   0.1, 100.0, "Fluid density", "kg/m³"),
        _p("t_end",  10.0,   1.0, 100.0, "Simulation end time (unsteady)", "s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=True,
    pinneapple_preset=None,
    latex_equations=[
        r"\rho\!\left(\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu\!\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)",
        r"\rho\!\left(\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}\right) = -\frac{\partial p}{\partial y} + \mu\!\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)",
        r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad (\text{incompressibility})",
        r"\mathrm{Re} = \frac{\rho\, U_\infty D}{\mu}, \qquad C_d = \frac{2F_d}{\rho U_\infty^2 D L}, \qquad \mathrm{St} = \frac{f D}{U_\infty}",
    ],
    tags=[
        "cylinder", "CFD", "benchmark", "drag", "Strouhal", "wake",
        "vortex shedding", "Kármán", "incompressible", "Raissi", "Williamson",
        "scour", "validation",
    ],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["u", "v", "p"],
        time_col="t",
        notes="u,v = velocity components; p = pressure; domain centred on cylinder",
    ),
)


# ===========================================================================
# SOLID MECHANICS  (5 problems)
# ===========================================================================

linear_elasticity_2d = PhysicsProblem(
    id="linear_elasticity_2d",
    name="2D Linear Elasticity (Plane Stress)",
    description=(
        "Plane-stress linear elasticity for thin plates. The equilibrium equations "
        "∇·σ + f = 0 are coupled to the constitutive law σ = C:ε and the "
        "strain-displacement relation ε = ½(∇u + ∇uᵀ). Solved for displacement "
        "field (u_x, u_y)."
    ),
    domain=PhysicsArea.SOLID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "∇·σ + f = 0  (equilibrium)",
        "σ = C:ε  (linear elastic constitutive law)",
        "ε = ½(∇u + ∇uᵀ)  (small strain)",
        "∂σ_xx/∂x + ∂σ_xy/∂y + f_x = 0",
        "∂σ_xy/∂x + ∂σ_yy/∂y + f_y = 0",
    ],
    applications=[
        "Structural panel analysis",
        "Sheet metal forming",
        "Plate and shell components",
        "Pressure vessel design",
    ],
    initial_conditions="N/A (static analysis)",
    boundary_conditions=(
        "Dirichlet: prescribed displacement on clamped edges; "
        "Neumann: prescribed traction σ·n = t on loaded surfaces"
    ),
    parameters=[
        _p("E", 210e9, 1e6, 1e12, "Young's modulus", "Pa"),
        _p("nu", 0.3, 0.0, 0.499, "Poisson's ratio", "—"),
        _p("rho", 7800.0, 100.0, 20000.0, "Density", "kg/m³"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="plane_stress_2d",
    tags=["elasticity", "static", "plane-stress", "structural", "linear"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["u_x", "u_y", "sigma_xx", "sigma_yy", "sigma_xy"],
        time_col=None,
        notes="",
    ),
)

linear_elasticity_3d = PhysicsProblem(
    id="linear_elasticity_3d",
    name="3D Linear Elasticity",
    description=(
        "Full three-dimensional linear elasticity. The governing equations are the "
        "equilibrium conditions ∇·σ + f = 0, the isotropic Hooke's law constitutive "
        "relation, and the small-strain kinematic equation. Solved for the 3D "
        "displacement vector (u_x, u_y, u_z). Requires a 3D geometry."
    ),
    domain=PhysicsArea.SOLID_MECHANICS,
    dimension=Dimension.THREE_D,
    equations=[
        "∇·σ + f = 0  (3D equilibrium)",
        "σ_ij = λ ε_kk δ_ij + 2μ ε_ij  (isotropic Hooke's law)",
        "ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)  (small strain)",
    ],
    applications=[
        "3D structural component analysis",
        "Engine block stress analysis",
        "Implant biomechanics",
        "Additive manufacturing part qualification",
    ],
    initial_conditions="N/A (static analysis)",
    boundary_conditions=(
        "Dirichlet: zero displacement on fixed faces; "
        "Neumann: applied traction on loaded surfaces"
    ),
    parameters=[
        _p("E", 210e9, 1e6, 1e12, "Young's modulus", "Pa"),
        _p("nu", 0.3, 0.0, 0.499, "Poisson's ratio", "—"),
        _p("rho", 7800.0, 100.0, 20000.0, "Density", "kg/m³"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=True,
    pinneapple_preset="linear_elasticity_3d",
    tags=["elasticity", "3D", "static", "structural", "linear"],
    column_hints=_hints(
        coord_cols=["x", "y", "z"],
        output_cols=["u_x", "u_y", "u_z", "sigma_xx", "sigma_yy", "sigma_zz"],
        time_col=None,
        notes="",
    ),
)

crash_analysis = PhysicsProblem(
    id="crash_analysis",
    name="Nonlinear Structural Dynamics / Crash Analysis",
    description=(
        "Nonlinear structural dynamics for large-deformation impact scenarios. "
        "Combines material nonlinearity (plasticity), geometric nonlinearity (large "
        "strains), and contact mechanics. Governed by the dynamic momentum balance "
        "∇·σ + ρb = ρa, with finite-strain kinematics."
    ),
    domain=PhysicsArea.SOLID_MECHANICS,
    dimension=Dimension.THREE_D,
    equations=[
        "∇·σ + ρb = ρa  (dynamic momentum balance)",
        "σ = f(F, history)  (nonlinear constitutive, e.g. J2 plasticity)",
        "F = ∂x/∂X  (deformation gradient)",
        "J = det(F) > 0  (physical constraint)",
    ],
    applications=[
        "Automotive crash simulation",
        "Ballistic impact protection",
        "Drop-test qualification",
        "Energy absorber design",
    ],
    initial_conditions="u = 0, v = v_impact at t = 0",
    boundary_conditions=(
        "Impact velocity prescribed on striker surface; "
        "fixed or symmetry BCs on structure base"
    ),
    parameters=[
        _p("E", 210e9, 1e6, 1e12, "Young's modulus", "Pa"),
        _p("rho", 7800.0, 100.0, 20000.0, "Density", "kg/m³"),
        _p("v_impact", 10.0, 0.1, 200.0, "Impact velocity", "m/s"),
        _p("L", 1.0, 0.1, 10.0, "Rod length", "m"),
        _p("R", 0.05, 0.005, 0.5, "Rod radius", "m"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=True,
    pinneapple_preset=None,
    tags=["nonlinear", "dynamics", "plasticity", "large-deformation", "3D", "impact"],
    column_hints=_hints(
        coord_cols=["x", "y", "z", "t"],
        output_cols=["u_x", "u_y", "u_z"],
        time_col="t",
        notes="",
    ),
)

beam_bending = PhysicsProblem(
    id="beam_bending",
    name="Euler-Bernoulli Beam Bending",
    description=(
        "Classical Euler-Bernoulli beam theory for slender beams under transverse "
        "loading q(x). The fourth-order ODE EI ∂⁴w/∂x⁴ = q(x) relates the lateral "
        "deflection w to the distributed load. Valid for small deflections and "
        "slender beams (L/h >> 1)."
    ),
    domain=PhysicsArea.SOLID_MECHANICS,
    dimension=Dimension.ONE_D,
    equations=[
        "EI ∂⁴w/∂x⁴ = q(x)  (Euler-Bernoulli equation)",
        "M = EI ∂²w/∂x²  (bending moment)",
        "V = EI ∂³w/∂x³  (shear force)",
    ],
    applications=[
        "Bridge girder design",
        "Aircraft wing spar analysis",
        "Micro-cantilever MEMS sensors",
        "Offshore platform structural analysis",
    ],
    initial_conditions="N/A (static analysis)",
    boundary_conditions=(
        "Simply supported: w = 0, M = 0 at x = 0 and x = L; "
        "or Clamped: w = 0, ∂w/∂x = 0 at fixed end"
    ),
    parameters=[
        _p("E", 210e9, 1e6, 1e12, "Young's modulus", "Pa"),
        _p("I", 1e-6, 1e-10, 1e-2, "Second moment of area", "m⁴"),
        _p("L", 1.0, 0.01, 100.0, "Beam length", "m"),
        _p("q0", 1000.0, 0.0, 1e8, "Uniform distributed load intensity", "N/m"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["beam", "structural", "analytical", "1D", "bending"],
    column_hints=_hints(
        coord_cols=["x"],
        output_cols=["u_y", "sigma"],
        time_col=None,
        notes="",
    ),
)

thermoelasticity_2d = PhysicsProblem(
    id="thermoelasticity_2d",
    name="2D Coupled Thermoelasticity",
    description=(
        "Coupled thermoelasticity problem where thermal expansion induces mechanical "
        "stresses and vice versa. The system couples the heat equation to the elastic "
        "equilibrium equations through the thermal strain εᵀ = α(T - T₀)δ."
    ),
    domain=PhysicsArea.SOLID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "ρcₚ ∂T/∂t = ∇·(κ∇T) + Q  (heat equation)",
        "∇·σ + f = 0  (mechanical equilibrium)",
        "σ = C:(ε - εᵀ)  (thermoelastic constitutive)",
        "εᵀ = α(T - T₀)I  (thermal strain)",
    ],
    applications=[
        "Turbine blade thermal stress",
        "Electronic packaging reliability",
        "Nuclear fuel rod analysis",
        "Welding residual stress prediction",
    ],
    initial_conditions="T(x, y, 0) = T₀, u = v = 0 at t = 0",
    boundary_conditions=(
        "Temperature: Dirichlet on heated/cooled surfaces, Neumann elsewhere; "
        "Displacement: Dirichlet on fixed supports, traction on loaded faces"
    ),
    parameters=[
        _p("E", 210e9, 1e6, 1e12, "Young's modulus", "Pa"),
        _p("nu", 0.3, 0.0, 0.499, "Poisson's ratio", "—"),
        _p("alpha", 12e-6, 1e-7, 1e-4, "Coefficient of thermal expansion", "1/K"),
        _p("kappa", 50.0, 0.1, 1000.0, "Thermal conductivity", "W/(m·K)"),
        _p("rho_cp", 3.5e6, 1e4, 1e8, "Volumetric heat capacity (ρcₚ)", "J/(m³·K)"),
        _p("T0", 293.15, 100.0, 2000.0, "Reference temperature", "K"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset="thermoelasticity_2d",
    tags=["thermoelasticity", "coupled", "thermal", "structural", "2D"],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["u_x", "u_y", "T"],
        time_col="t",
        notes="",
    ),
)


# ===========================================================================
# ELECTROMAGNETISM  (4 problems)
# ===========================================================================

poisson_electrostatics = PhysicsProblem(
    id="poisson_electrostatics",
    name="Electrostatic Poisson Equation",
    description=(
        "Governing equation for electrostatic potential in a medium with permittivity "
        "ε and free charge density ρ_f. The Poisson equation -∇·(ε∇φ) = ρ_f is the "
        "fundamental equation of electrostatics, reducing to Laplace's equation in "
        "charge-free regions."
    ),
    domain=PhysicsArea.ELECTROMAGNETISM,
    dimension=Dimension.TWO_D,
    equations=[
        "-∇·(ε∇φ) = ρ_f  (Poisson equation)",
        "E = -∇φ  (electric field)",
        "D = εE  (electric displacement)",
        "-(∂²φ/∂x² + ∂²φ/∂y²) = ρ_f/ε₀  (in free space)",
    ],
    applications=[
        "Capacitor electric field analysis",
        "Semiconductor device simulation",
        "Ion trap design",
        "Electrochemical cell modeling",
    ],
    initial_conditions="N/A (elliptic steady-state problem)",
    boundary_conditions=(
        "Dirichlet: specified potential φ on conductors; "
        "Neumann: E_n = 0 on symmetry/far-field boundaries"
    ),
    parameters=[
        _p("epsilon_r", 1.0, 1.0, 10000.0, "Relative permittivity", "—"),
        _p("rho_charge", 0.0, -1e6, 1e6, "Free charge density", "C/m³"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset="poisson_2d",
    tags=["electrostatics", "elliptic", "Poisson", "potential", "linear"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["phi"],
        time_col=None,
        notes="",
    ),
)

helmholtz_wave = PhysicsProblem(
    id="helmholtz_wave",
    name="Helmholtz Wave Equation",
    description=(
        "The Helmholtz equation ∇²u + k²u = f is the frequency-domain form of the "
        "wave equation after time-harmonic factoring e^(-iωt). The wave number "
        "k = ω/c relates angular frequency to wave speed. Solutions can exhibit "
        "high-frequency oscillations that challenge standard neural network approaches."
    ),
    domain=PhysicsArea.ELECTROMAGNETISM,
    dimension=Dimension.TWO_D,
    equations=[
        "∇²u + k²u = f(x, y)  (Helmholtz equation)",
        "∂²u/∂x² + ∂²u/∂y² + k²u = f",
        "k = ω/c  (wave number)",
    ],
    applications=[
        "Acoustic resonance in cavities",
        "Radar cross-section computation",
        "Room acoustics simulation",
        "Optical waveguide analysis",
    ],
    initial_conditions="N/A (frequency domain, no time dependence)",
    boundary_conditions=(
        "Dirichlet: u = 0 (sound-hard wall) or u = g (driven boundary); "
        "Sommerfeld/absorbing BC at far field"
    ),
    parameters=[
        _p("k", 2.0, 0.1, 50.0, "Wave number", "1/m"),
        _p("omega", 6.28, 0.1, 314.0, "Angular frequency", "rad/s"),
        _p("c", 340.0, 1.0, 3e8, "Wave speed", "m/s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="helmholtz",
    tags=["wave", "frequency-domain", "oscillatory", "acoustic", "electromagnetic"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["u_re", "u_im"],
        time_col=None,
        notes="u_re/u_im = real/imaginary parts of complex solution",
    ),
)

laplace_potential = PhysicsProblem(
    id="laplace_potential",
    name="Laplace Potential Field",
    description=(
        "Laplace's equation ∇²φ = 0 governs steady-state diffusion, electrostatic "
        "potential in charge-free regions, irrotational incompressible flow potential, "
        "and gravitational potential. Solutions are harmonic functions that satisfy "
        "the mean-value property."
    ),
    domain=PhysicsArea.ELECTROMAGNETISM,
    dimension=Dimension.TWO_D,
    equations=[
        "∇²φ = 0  (Laplace equation)",
        "∂²φ/∂x² + ∂²φ/∂y² = 0",
        "E = -∇φ  (electric field from potential)",
    ],
    applications=[
        "Electrostatic shielding design",
        "Potential flow aerodynamics",
        "Steady heat conduction",
        "Groundwater flow potential",
    ],
    initial_conditions="N/A (elliptic steady-state problem)",
    boundary_conditions=(
        "Dirichlet: specified potential φ = g on boundary segments; "
        "Neumann: ∂φ/∂n = 0 on insulating/symmetry boundaries"
    ),
    parameters=[
        _p("phi_top", 1.0, -100.0, 100.0, "Potential on top boundary", "V"),
        _p("phi_bottom", 0.0, -100.0, 100.0, "Potential on bottom boundary", "V"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset="laplace_2d",
    tags=["Laplace", "harmonic", "elliptic", "potential", "linear", "steady"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["phi"],
        time_col=None,
        notes="",
    ),
)

magnetostatics_2d = PhysicsProblem(
    id="magnetostatics_2d",
    name="2D Magnetostatics (Vector Potential)",
    description=(
        "Static magnetic field problem solved via the magnetic vector potential A. "
        "In 2D (x-y plane with z-directed current), the z-component of A satisfies "
        "a Poisson-type equation ∇²A_z = -μJ_z. The magnetic flux density is then "
        "B = ∇×A."
    ),
    domain=PhysicsArea.ELECTROMAGNETISM,
    dimension=Dimension.TWO_D,
    equations=[
        "∇×H = J  (Ampere's law)",
        "∇·B = 0  (Gauss's law for magnetism)",
        "B = μH  (constitutive relation)",
        "∇²A_z = -μJ_z  (vector potential formulation)",
        "B_x = ∂A_z/∂y, B_y = -∂A_z/∂x",
    ],
    applications=[
        "Transformer core design",
        "Electric motor magnetic field",
        "Electromagnet design",
        "MRI gradient coil analysis",
    ],
    initial_conditions="N/A (static problem)",
    boundary_conditions=(
        "Dirichlet: A_z = 0 on outer boundary (zero tangential B); "
        "Continuity of A and ∂A/∂n across material interfaces"
    ),
    parameters=[
        _p("mu_r", 1000.0, 1.0, 1e6, "Relative permeability", "—"),
        _p("J_z", 1e6, -1e8, 1e8, "Current density (z-component)", "A/m²"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["magnetostatics", "vector-potential", "elliptic", "EM", "static"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["A_z", "B_x", "B_y"],
        time_col=None,
        notes="",
    ),
)

hartmann_mhd = PhysicsProblem(
    id="hartmann_mhd",
    name="Hartmann MHD Duct Flow (3D)",
    description=(
        "Magnetohydrodynamic (MHD) pressure-driven flow in a rectangular duct with a "
        "transverse magnetic field B — the canonical Hartmann flow benchmark in "
        "computational MHD research. The magnetic field brakes the fluid "
        "through the Lorentz force J×B, flattening the parabolic Poiseuille profile into "
        "a characteristic plug flow with thin Hartmann boundary layers at the walls. "
        "Exact closed-form solution: u(y)=(GH²/μHa²)(1−cosh(Ha·y/H)/cosh(Ha))."
    ),
    domain=PhysicsArea.ELECTROMAGNETISM,
    dimension=Dimension.THREE_D,
    equations=[
        "μ∇²u − σB²u = ∂p/∂x  (Hartmann MHD momentum)",
        "∇·u = 0  (incompressibility)",
        "Exact: u(y) = (GH²/μHa²)(1 − cosh(Ha·y/H)/cosh(Ha))",
        "Ha = B·H·√(σ/μ)  (Hartmann number)",
        "Ha=0 → Poiseuille parabola; Ha→∞ → plug flow with thin wall layers",
    ],
    applications=[
        "Hartmann MHD physics benchmark",
        "Liquid-metal blanket cooling in fusion reactors",
        "Electromagnetic pump design",
        "MHD generator flow optimization",
        "Semiconductor crystal growth in magnetic fields",
    ],
    initial_conditions="N/A — steady fully-developed duct flow",
    boundary_conditions=(
        "No-slip: u=0 at Hartmann walls (y=±H) and side walls (z=0,Lz); "
        "Inlet: developed profile; outlet: Neumann ∂p/∂n=0"
    ),
    parameters=[
        _p("Ha",  5.0,  0.0, 50.0, "Hartmann number  Ha=BH√(σ/μ)  (0=Poiseuille, ≫1=plug)", "—"),
        _p("Re", 100.0, 1.0, 500.0, "Reynolds number  Re=ρUH/μ", "—"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset="lid_driven_cavity_3d",
    tags=["MHD", "Hartmann", "magnetic", "duct", "3D", "analytical", "fusion"],
    column_hints=_hints(
        coord_cols=["x", "y", "z", "t"],
        output_cols=["u", "v", "w", "B_x", "B_y", "p"],
        time_col="t",
        notes="",
    ),
)


# ===========================================================================
# QUANTUM MECHANICS  (3 problems)
# ===========================================================================

schrodinger_stationary = PhysicsProblem(
    id="schrodinger_stationary",
    name="Time-Independent Schrödinger Equation",
    description=(
        "The stationary (time-independent) Schrödinger equation yields the energy "
        "eigenstates and eigenvalues of a quantum system. For a particle of mass m "
        "in a potential V(x), the equation is an eigenvalue problem for the wave "
        "function ψ and energy E."
    ),
    domain=PhysicsArea.QUANTUM_MECHANICS,
    dimension=Dimension.ONE_D,
    equations=[
        "-ℏ²/(2m) ∂²ψ/∂x² + V(x)ψ = Eψ  (time-independent Schrödinger)",
        "-ℏ²/(2m) ψ'' + V(x)ψ - Eψ = 0",
        "∫|ψ|² dx = 1  (normalization constraint)",
    ],
    applications=[
        "Quantum well energy levels",
        "Hydrogen atom orbital structure",
        "Particle-in-a-box benchmarking",
        "Tunneling probability calculation",
    ],
    initial_conditions="N/A (eigenvalue problem, no time evolution)",
    boundary_conditions="ψ(x→±∞) = 0 (bound states); or ψ(0) = ψ(L) = 0 (infinite square well)",
    parameters=[
        _p("hbar", 1.0546e-34, 1e-40, 1e-30, "Reduced Planck constant", "J·s"),
        _p("m", 9.109e-31, 1e-35, 1e-25, "Particle mass", "kg"),
        _p("V0", 0.0, -1e-18, 1e-18, "Potential barrier height", "J"),
        _p("L", 1e-9, 1e-12, 1e-6, "Domain length", "m"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["quantum", "eigenvalue", "wavefunction", "stationary", "1D"],
    column_hints=_hints(
        coord_cols=["x"],
        output_cols=["psi_re", "psi_im"],
        time_col=None,
        notes="psi_re/psi_im = real/imaginary parts of wavefunction psi(x)",
    ),
)

schrodinger_timedep = PhysicsProblem(
    id="schrodinger_timedep",
    name="Time-Dependent Schrödinger Equation",
    description=(
        "The time-dependent Schrödinger equation governs the full quantum dynamics "
        "of a particle. The complex wave function ψ(x, t) evolves under the "
        "Hamiltonian operator. This problem requires handling complex-valued outputs "
        "or splitting into real/imaginary components."
    ),
    domain=PhysicsArea.QUANTUM_MECHANICS,
    dimension=Dimension.ONE_D,
    equations=[
        "iℏ ∂ψ/∂t = [-ℏ²/(2m) ∂²/∂x² + V(x)] ψ  (TDSE)",
        "Probability density: ρ = |ψ|² = ψ_R² + ψ_I²",
        "Conservation: ∂ρ/∂t + ∂J/∂x = 0  (probability current J)",
    ],
    applications=[
        "Quantum wave packet dynamics",
        "Quantum tunneling time scales",
        "Laser-matter interaction",
        "Quantum computing gate simulation",
    ],
    initial_conditions="ψ(x, 0) = ψ₀(x)  (Gaussian wave packet or eigenstate)",
    boundary_conditions="ψ(0, t) = ψ(L, t) = 0 (Dirichlet) or absorbing boundary conditions",
    parameters=[
        _p("hbar", 1.0, 0.01, 10.0, "Reduced Planck constant (normalized units)", "—"),
        _p("m", 1.0, 0.01, 100.0, "Particle mass (normalized units)", "—"),
        _p("V0", 0.0, -10.0, 100.0, "Potential amplitude", "—"),
        _p("sigma0", 0.1, 0.01, 1.0, "Initial wave packet width", "—"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["quantum", "time-dependent", "complex-valued", "wave-packet", "1D"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["psi_re", "psi_im"],
        time_col="t",
        notes="psi_re/psi_im = real/imaginary parts of psi(x,t)",
    ),
)

gross_pitaevskii = PhysicsProblem(
    id="gross_pitaevskii",
    name="Gross-Pitaevskii Equation (BEC)",
    description=(
        "The Gross-Pitaevskii equation (GPE) is a mean-field model for Bose-Einstein "
        "condensates (BEC). It extends the Schrödinger equation with a nonlinear "
        "self-interaction term g|ψ|² that accounts for two-body contact interactions "
        "between bosons."
    ),
    domain=PhysicsArea.QUANTUM_MECHANICS,
    dimension=Dimension.ONE_D,
    equations=[
        "iℏ ∂ψ/∂t = [-ℏ²/(2m) ∂²/∂x² + V(x) + g|ψ|²] ψ  (GPE)",
        "N = ∫|ψ|² dx  (normalization = atom number)",
        "Nonlinear term: g = 4πℏ²a_s/m  (g > 0 repulsive, g < 0 attractive)",
    ],
    applications=[
        "Bose-Einstein condensate dynamics",
        "Superfluidity modeling",
        "Vortex nucleation in rotating BECs",
        "Matter-wave solitons",
    ],
    initial_conditions="ψ(x, 0) = ψ_TF(x)  (Thomas-Fermi ground state approximation)",
    boundary_conditions="ψ(±L/2, t) = 0  (hard wall trap boundaries)",
    parameters=[
        _p("g", 1.0, -10.0, 100.0, "Nonlinear interaction strength", "—"),
        _p("hbar", 1.0, 0.1, 10.0, "Reduced Planck constant (normalized)", "—"),
        _p("m", 1.0, 0.1, 10.0, "Boson mass (normalized)", "—"),
        _p("omega", 1.0, 0.01, 100.0, "Harmonic trap frequency", "rad/s"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["quantum", "BEC", "nonlinear", "condensate", "complex-valued", "1D"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["psi_re", "psi_im"],
        time_col="t",
        notes="psi_re/psi_im = real/imaginary parts of condensate wavefunction",
    ),
)


# ===========================================================================
# THERMODYNAMICS  (3 problems)
# ===========================================================================

chip_heat_3d = PhysicsProblem(
    id="chip_heat_3d",
    name="Chip Heat-Sink Cooling (Fin Array 3D)",
    description=(
        "Forced-convection cooling of a semiconductor chip through an aluminum rectangular-fin "
        "heat sink — a well-established thermal management benchmark for fin-array cooling. "
        "N fins of height H, thickness t, and length L stand on a "
        "hot chip base at T_base, cooled by air at T_amb with convection coefficient h. "
        "Exact 1-D fin theory gives the temperature profile along each fin: "
        "T(z) = T_amb + (T_base−T_amb)·cosh(m(H−z))/cosh(mH),  m=√(2h/kt)."
    ),
    domain=PhysicsArea.THERMODYNAMICS,
    dimension=Dimension.THREE_D,
    equations=[
        "d²T/dz² − m²(T−T_amb) = 0  (1-D fin equation)",
        "m = √(2h / (k·t))  (fin efficiency parameter)",
        "T(z) = T_amb + (T_base−T_amb)·cosh(m(H−z)) / cosh(mH)",
        "Q_fin = k·A·m·(T_base−T_amb)·tanh(mH)  (heat transfer per fin)",
        "η_fin = tanh(mH) / (mH)  (fin efficiency)",
    ],
    applications=[
        "Chip-cooling fin-array physics benchmark",
        "GPU / ASIC thermal management design",
        "Data-center hot-aisle cooling",
        "Electric vehicle power-electronics cooling",
        "LED luminaire thermal design",
    ],
    initial_conditions="N/A — steady-state conjugate heat transfer",
    boundary_conditions=(
        "Base: T = T_base (isothermal chip surface); "
        "Fin tip: ∂T/∂z = 0 (adiabatic); "
        "Fin surfaces: −k ∂T/∂n = h(T−T_amb) (Newton convection)"
    ),
    parameters=[
        _p("T_base", 80.0, 40.0, 150.0, "Chip junction temperature", "°C"),
        _p("T_amb",  25.0, 10.0,  50.0, "Ambient air temperature",   "°C"),
        _p("h_conv", 200.0, 20.0, 2000.0, "Convective heat-transfer coefficient", "W/m²K"),
        _p("k_fin", 200.0, 10.0, 400.0, "Fin thermal conductivity (Al≈200 W/mK)", "W/mK"),
        _p("N_fins",   6.0, 2.0,  12.0, "Number of fins", "—"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="steady_heat_conduction_3d",
    tags=["thermal", "fins", "chip", "3D", "steady", "datacenter"],
    column_hints=_hints(
        coord_cols=["x", "y", "z", "t"],
        output_cols=["T"],
        time_col="t",
        notes="",
    ),
)

heat_equation_1d = PhysicsProblem(
    id="heat_equation_1d",
    name="1D Heat Diffusion Equation",
    description=(
        "The parabolic heat (diffusion) equation governs temperature evolution in "
        "one spatial dimension. The thermal diffusivity α = κ/(ρcₚ) controls how "
        "quickly temperature disturbances propagate through the medium."
    ),
    domain=PhysicsArea.THERMODYNAMICS,
    dimension=Dimension.ONE_D,
    equations=[
        "∂T/∂t = α ∂²T/∂x²  (1D heat equation)",
        "α = κ/(ρcₚ)  (thermal diffusivity)",
        "Steady state: d²T/dx² = 0  →  T = ax + b",
    ],
    applications=[
        "Thermal annealing of metals",
        "Fin heat transfer analysis",
        "Geological heat flow",
        "Battery thermal management",
    ],
    initial_conditions="T(x, 0) = T_init(x)  (e.g., Gaussian hot spot or uniform T₀)",
    boundary_conditions=(
        "Dirichlet: T(0, t) = T_L, T(L, t) = T_R; "
        "or Neumann: ∂T/∂x = 0 (insulated ends)"
    ),
    parameters=[
        _p("alpha", 1e-5, 1e-8, 1e-3, "Thermal diffusivity", "m²/s"),
        _p("kappa", 50.0, 0.01, 500.0, "Thermal conductivity", "W/(m·K)"),
        _p("T_L", 100.0, -273.0, 3000.0, "Left boundary temperature", "°C"),
        _p("T_R", 0.0, -273.0, 3000.0, "Right boundary temperature", "°C"),
        _p("L", 1.0, 0.001, 100.0, "Domain length", "m"),
    ],
    difficulty=DifficultyLevel.BEGINNER,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["heat", "diffusion", "parabolic", "linear", "time-dependent", "1D"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["T"],
        time_col="t",
        notes="",
    ),
)

heat_equation_2d = PhysicsProblem(
    id="heat_equation_2d",
    name="2D Heat Equation with Source",
    description=(
        "Two-dimensional transient heat conduction with an optional volumetric heat "
        "source Q. The equation ∂T/∂t = α∇²T + Q/(ρcₚ) governs temperature in "
        "planar domains. Covers both steady-state (∇²T = -Q/κ) and transient cases."
    ),
    domain=PhysicsArea.THERMODYNAMICS,
    dimension=Dimension.TWO_D,
    equations=[
        "∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²) + Q/(ρcₚ)  (transient 2D heat equation)",
        "Steady state: ∇²T = -Q/κ  (Poisson equation for temperature)",
        "α = κ/(ρcₚ)  (thermal diffusivity)",
    ],
    applications=[
        "Electronic component cooling",
        "Solar panel thermal management",
        "Geothermal energy extraction",
        "Nuclear reactor heat removal",
    ],
    initial_conditions="T(x, y, 0) = T₀  (uniform initial temperature)",
    boundary_conditions=(
        "Mixed: Dirichlet on heated/cooled surfaces; "
        "Neumann ∂T/∂n = 0 on insulated edges; "
        "Robin/convection: -κ∂T/∂n = h(T - T_∞)"
    ),
    parameters=[
        _p("alpha", 1e-5, 1e-8, 1e-3, "Thermal diffusivity", "m²/s"),
        _p("kappa", 50.0, 0.01, 500.0, "Thermal conductivity", "W/(m·K)"),
        _p("rho_cp", 3.5e6, 1e4, 1e8, "Volumetric heat capacity", "J/(m³·K)"),
        _p("Q", 0.0, 0.0, 1e8, "Volumetric heat source", "W/m³"),
        _p("h_conv", 10.0, 0.0, 10000.0, "Convective heat transfer coefficient", "W/(m²·K)"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="heat_equation_2d",
    tags=["heat", "diffusion", "parabolic", "2D", "source", "time-dependent"],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["T"],
        time_col="t",
        notes="",
    ),
)


# ===========================================================================
# WAVE PHYSICS  (2 problems)
# ===========================================================================

wave_equation = PhysicsProblem(
    id="wave_equation",
    name="2D Wave Equation",
    description=(
        "The scalar wave equation ∂²u/∂t² = c²∇²u governs the propagation of "
        "acoustic, seismic, and electromagnetic waves. The wave speed c determines "
        "the dispersion relation. This hyperbolic PDE requires both initial position "
        "and velocity conditions."
    ),
    domain=PhysicsArea.WAVE_PHYSICS,
    dimension=Dimension.TWO_D,
    equations=[
        "∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)  (2D wave equation)",
        "c = wave speed  (m/s)",
        "Dispersion relation: ω² = c²(k_x² + k_y²)",
    ],
    applications=[
        "Seismic wave modeling",
        "Ultrasound medical imaging",
        "Non-destructive testing",
        "Room acoustics simulation",
    ],
    initial_conditions=(
        "u(x, y, 0) = u₀(x, y)  (initial displacement); "
        "∂u/∂t(x, y, 0) = v₀(x, y)  (initial velocity)"
    ),
    boundary_conditions=(
        "Dirichlet: u = 0 on rigid walls; "
        "Sommerfeld: ∂u/∂t + c ∂u/∂n = 0 at absorbing boundaries"
    ),
    parameters=[
        _p("c", 1.0, 0.01, 1e4, "Wave propagation speed", "m/s"),
        _p("omega", 3.14, 0.1, 100.0, "Angular frequency of source", "rad/s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="wave_equation",
    tags=["wave", "hyperbolic", "time-dependent", "acoustic", "2D"],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["u"],
        time_col="t",
        notes="",
    ),
)

klein_gordon = PhysicsProblem(
    id="klein_gordon",
    name="Klein-Gordon Equation",
    description=(
        "The Klein-Gordon equation is the relativistic wave equation for a scalar "
        "field, describing spin-0 particles (e.g., pions). It reduces to the ordinary "
        "wave equation when the rest-mass term vanishes. The equation has both "
        "dispersive and oscillatory character."
    ),
    domain=PhysicsArea.WAVE_PHYSICS,
    dimension=Dimension.ONE_D,
    equations=[
        "(∂²/∂t² - c²∂²/∂x² + m²c⁴/ℏ²)ψ = 0  (Klein-Gordon equation)",
        "In natural units (c=ℏ=1): (□ + m²)ψ = 0",
        "Dispersion relation: ω² = c²k² + m²c⁴/ℏ²",
    ],
    applications=[
        "Relativistic quantum field theory",
        "Pion dynamics in nuclear physics",
        "Nonlinear optics (coupled envelope equations)",
        "Cosmological scalar field models",
    ],
    initial_conditions=(
        "ψ(x, 0) = ψ₀(x)  (initial field profile); "
        "∂ψ/∂t(x, 0) = π₀(x)  (conjugate momentum)"
    ),
    boundary_conditions="ψ(0, t) = ψ(L, t) = 0 (Dirichlet) or periodic",
    parameters=[
        _p("c", 1.0, 0.1, 10.0, "Wave speed", "m/s"),
        _p("m", 1.0, 0.0, 100.0, "Particle rest mass (natural units)", "—"),
        _p("hbar", 1.0, 0.01, 10.0, "Reduced Planck constant (natural units)", "—"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["relativistic", "wave", "hyperbolic", "dispersive", "1D", "quantum"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["u"],
        time_col="t",
        notes="",
    ),
)


helmholtz_3d_cavity = PhysicsProblem(
    id="helmholtz_3d_cavity",
    name="3D Acoustic Cavity Resonance (Helmholtz)",
    description=(
        "Frequency-domain acoustic standing-wave modes in a rigid rectangular cavity — "
        "the 3-D generalisation of the Helmholtz equation for electromagnetic and acoustic "
        "resonance problems. With sound-hard (Neumann) walls, the "
        "exact eigenfunction for mode (n,m,l) is the cosine product "
        "p(x,y,z)=cos(nπx/a)cos(mπy/b)cos(lπz/c) at resonance frequency "
        "f_res = c·√(n²/a²+m²/b²+l²/c²)/2. The model predicts the pressure "
        "amplitude pattern that PINN / operator models must learn."
    ),
    domain=PhysicsArea.WAVE_PHYSICS,
    dimension=Dimension.THREE_D,
    equations=[
        "∇²p + k²p = 0  (Helmholtz equation)",
        "∂p/∂n = 0  on all six walls (rigid/sound-hard BC)",
        "p(x,y,z) = cos(nπx/a)cos(mπy/b)cos(lπz/c)  (exact mode)",
        "k_res = π·√(n²/a²+m²/b²+l²/c²)",
        "f_res = c_sound·k_res / (2π)  Hz",
    ],
    applications=[
        "EM waveguide / cavity resonance benchmark",
        "Room acoustics — standing-wave mode prediction",
        "Microwave resonator / cavity filter design",
        "Laser cavity mode structure",
        "Seismic tomography 3-D wave propagation",
    ],
    initial_conditions="N/A — frequency-domain steady-state",
    boundary_conditions="∂p/∂n = 0 on all walls (rigid-wall sound-hard Neumann BC)",
    parameters=[
        _p("a", 1.0, 0.2, 3.0, "Cavity length (x)", "m"),
        _p("b", 0.8, 0.2, 3.0, "Cavity width  (y)", "m"),
        _p("c", 0.6, 0.2, 3.0, "Cavity height (z)", "m"),
        _p("n_mode", 2.0, 0.0, 5.0, "Modal index n  (x-direction)", "—"),
        _p("m_mode", 1.0, 0.0, 5.0, "Modal index m  (y-direction)", "—"),
        _p("l_mode", 1.0, 0.0, 5.0, "Modal index l  (z-direction)", "—"),
        _p("c_sound", 343.0, 100.0, 1500.0, "Speed of sound", "m/s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset="helmholtz",
    tags=["acoustics", "Helmholtz", "3D", "resonance", "analytical", "waveguide"],
    column_hints=_hints(
        coord_cols=["x", "y", "z"],
        output_cols=["p_re", "p_im"],
        time_col=None,
        notes="p_re/p_im = real/imaginary parts of pressure",
    ),
)

# ===========================================================================
# NONLINEAR DYNAMICS  (1 problem)
# ===========================================================================

kdv_equation = PhysicsProblem(
    id="kdv_equation",
    name="Korteweg-de Vries (KdV) Equation",
    description=(
        "The Korteweg-de Vries (KdV) equation models nonlinear shallow-water waves "
        "and supports soliton solutions — localized, stable, particle-like wave "
        "packets that preserve their shape after collisions. The balance between "
        "nonlinear steepening and dispersive spreading produces solitons."
    ),
    domain=PhysicsArea.NONLINEAR_DYNAMICS,
    dimension=Dimension.ONE_D,
    equations=[
        "∂u/∂t + 6u ∂u/∂x + ∂³u/∂x³ = 0  (KdV equation)",
        "One-soliton solution: u(x,t) = (c/2) sech²(√(c/4)(x - ct))",
    ],
    applications=[
        "Shallow water solitary wave propagation",
        "Internal ocean waves",
        "Optical solitons in fibers",
        "Plasma ion-acoustic waves",
    ],
    initial_conditions="u(x, 0) = sech²(x)  (single soliton initial condition)",
    boundary_conditions="u → 0 as x → ±∞ (rapidly decaying); periodic BC for finite domains",
    parameters=[
        _p("c_soliton", 1.0, 0.01, 10.0, "Soliton speed", "m/s"),
        _p("L", 20.0, 1.0, 200.0, "Domain half-length", "m"),
    ],
    difficulty=DifficultyLevel.ADVANCED,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["soliton", "nonlinear", "dispersive", "integrable", "1D", "KdV"],
    column_hints=_hints(
        coord_cols=["x", "t"],
        output_cols=["u"],
        time_col="t",
        notes="",
    ),
)


# ===========================================================================
# 2D VISUALIZATION SHOWCASE  (2 problems)
# ===========================================================================

airfoil_naca0012_2d = PhysicsProblem(
    id="airfoil_naca0012_2d",
    name="NACA 0012 Airfoil — 2D Cp Heatmap",
    description=(
        "Two-dimensional potential flow around the NACA 0012 symmetric airfoil solved "
        "with the Hess-Smith panel method (source + thin-airfoil vortex). The pressure "
        "coefficient Cp = 1 − V²/V∞² is evaluated on a dense 60×45 Cartesian grid "
        "spanning the full near-field domain. The airfoil body appears as a white "
        "region in the heatmap. Increasing α reveals the suction peak on the upper "
        "surface and the stagnation zone near the leading edge."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "Δφ = 0  (potential flow, incompressible irrotational)",
        "Cp = 1 − V²/V∞²  (pressure coefficient)",
        "Hess-Smith panel method: σⱼ sources + γ vortex (Kutta condition)",
        "Boundary condition: ∂φ/∂n = 0  (no-penetration on airfoil surface)",
        "Kutta condition: γ = π·V∞·c·sin(α)  (thin-airfoil theory)",
    ],
    applications=[
        "Aerodynamic pressure distribution visualization (CFD validation)",
        "Lift and drag coefficient prediction via PINN",
        "Inverse design: Cp → geometry shape optimization",
        "Wind turbine blade profile analysis",
        "Racing car wing aerodynamics",
    ],
    initial_conditions="N/A — steady-state potential flow",
    boundary_conditions="No-penetration on airfoil surface; Cp → 0 far field; Kutta condition at trailing edge",
    parameters=[
        _p("alpha_deg", 5.0, -15.0, 25.0, "Angle of attack", "°"),
        _p("V_inf",     1.0,   0.1, 100.0, "Freestream velocity", "m/s"),
        _p("chord",     1.0,   0.1,  10.0, "Chord length", "m"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["airfoil", "NACA 0012", "2D", "Cp", "heatmap", "panel method", "aerodynamics", "potential flow"],
    column_hints=_hints(
        coord_cols=["x", "y"],
        output_cols=["u", "v", "p", "Cp"],
        time_col=None,
        notes="Cp = pressure coefficient",
    ),
)

shallow_water_2d = PhysicsProblem(
    id="shallow_water_2d",
    name="Shallow Water Equations — 2D Gravity Wave",
    description=(
        "Linearised Shallow Water Equations (SWE) on a periodic square domain — the "
        "mathematical foundation for atmospheric/oceanic gravity wave propagation used "
        "in physics-based climate and flood prediction models. "
        "An exact FFT solution is computed for a Gaussian height perturbation at the "
        "domain centre: η̂(k,t) = η̂₀(k)·cos(c|k|t) where c = √(g·h₀). The heatmap "
        "shows the dimensionless height anomaly η/h₀ at time t, revealing the "
        "characteristic circular gravity-wave rings spreading outward."
    ),
    domain=PhysicsArea.FLUID_MECHANICS,
    dimension=Dimension.TWO_D,
    equations=[
        "∂η/∂t + h₀ ∇·u = 0  (mass conservation)",
        "∂u/∂t = −g ∇η       (momentum equation, linearised)",
        "→  ∂²η/∂t² = c² ∇²η  with c = √(g·h₀)  (wave equation)",
        "η̂(k,t) = η̂₀(k)·cos(c|k|t)  (exact FFT solution)",
        "η₀(x,y) = A·exp(−((x−0.5)²+(y−0.5)²)/(2σ²))  (Gaussian IC)",
    ],
    applications=[
        "Climate/weather shallow-water benchmark",
        "Tsunami propagation in shallow coastal waters",
        "Tidal bore simulation in estuaries",
        "Flood inundation modelling (BRLi / AXA risk)",
        "Atmospheric gravity wave parameterisation (ECMWF IFS)",
    ],
    initial_conditions="η(x,y,0) = A·exp(−r²/2σ²)  (Gaussian bump at domain centre); ∂η/∂t|₀ = 0",
    boundary_conditions="Periodic on all four sides (toroidal domain [0,1]²)",
    parameters=[
        _p("g",      9.81, 0.1,   100.0, "Gravitational acceleration", "m/s²"),
        _p("h0",    10.0,  0.5,  5000.0, "Undisturbed water depth", "m"),
        _p("A",      1.0,  0.01,   50.0, "Initial wave amplitude", "m"),
        _p("sigma",  0.06, 0.01,    0.3, "Gaussian bump width σ", ""),
        _p("t_eval", 0.40, 0.0,    10.0, "Evaluation time t", "s"),
    ],
    difficulty=DifficultyLevel.INTERMEDIATE,
    requires_geometry=False,
    pinneapple_preset=None,
    tags=["SWE", "climate", "gravity wave", "2D", "FFT", "ocean", "atmosphere", "tsunami"],
    column_hints=_hints(
        coord_cols=["x", "y", "t"],
        output_cols=["h", "u", "v"],
        time_col="t",
        notes="h = water depth",
    ),
)


# ===========================================================================
# Assembled catalog
# ===========================================================================

PROBLEMS_CATALOG: dict[str, PhysicsProblem] = {
    # Fluid Mechanics
    "burgers_1d": burgers_1d,
    "navier_stokes_2d": navier_stokes_2d,
    "navier_stokes_3d": navier_stokes_3d,
    # 3D multi-physics benchmarks
    "taylor_green_vortex": taylor_green_vortex,
    "darcy_flow_3d": darcy_flow_3d,
    "stokes_sphere": stokes_sphere,
    "lid_driven_cavity": lid_driven_cavity,
    "bernoulli_flow": bernoulli_flow,
    "stokes_flow": stokes_flow,
    "darcy_flow": darcy_flow,
    "channel_poiseuille": channel_poiseuille,
    "cylinder_flow_2d": cylinder_flow_2d,
    # Solid Mechanics
    "linear_elasticity_2d": linear_elasticity_2d,
    "linear_elasticity_3d": linear_elasticity_3d,
    "crash_analysis": crash_analysis,
    "beam_bending": beam_bending,
    "thermoelasticity_2d": thermoelasticity_2d,
    # Electromagnetism
    "poisson_electrostatics": poisson_electrostatics,
    "helmholtz_wave": helmholtz_wave,
    "laplace_potential": laplace_potential,
    "magnetostatics_2d": magnetostatics_2d,
    "hartmann_mhd": hartmann_mhd,
    # Quantum Mechanics
    "schrodinger_stationary": schrodinger_stationary,
    "schrodinger_timedep": schrodinger_timedep,
    "gross_pitaevskii": gross_pitaevskii,
    # Thermodynamics
    "chip_heat_3d": chip_heat_3d,
    "heat_equation_1d": heat_equation_1d,
    "heat_equation_2d": heat_equation_2d,
    # Wave Physics
    "wave_equation": wave_equation,
    "helmholtz_3d_cavity": helmholtz_3d_cavity,
    "klein_gordon": klein_gordon,
    # Nonlinear Dynamics
    "kdv_equation": kdv_equation,
    # 2D Visualization showcase
    "airfoil_naca0012_2d": airfoil_naca0012_2d,
    "shallow_water_2d":    shallow_water_2d,
}

assert len(PROBLEMS_CATALOG) == 34, (
    f"Expected 34 problems, got {len(PROBLEMS_CATALOG)}"
)

# ── KaTeX equation overrides ─────────────────────────────────────────────────
# Applied only to problems that don't already have latex_equations.
_LATEX_EQ_OVERRIDES: dict[str, list[str]] = {
    "burgers_1d": [
        r"\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}",
        r"x \in [-1,1],\; t \in [0,1]",
    ],
    "navier_stokes_2d": [
        r"\rho\!\left(\frac{\partial\mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p + \mu\nabla^2\mathbf{u}",
        r"\nabla\cdot\mathbf{u} = 0",
        r"\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{\mathrm{Re}}\!\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)",
        r"\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{\mathrm{Re}}\!\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)",
        r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0",
    ],
    "navier_stokes_3d": [
        r"\rho\!\left(\frac{\partial\mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p + \mu\nabla^2\mathbf{u}",
        r"\nabla\cdot\mathbf{u} = 0",
        r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0",
    ],
    "taylor_green_vortex": [
        r"\frac{\partial\mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + \frac{1}{\mathrm{Re}}\nabla^2\mathbf{u},\quad\nabla\cdot\mathbf{u}=0",
        r"u_0 = \sin x\cos y\cos z,\quad v_0 = -\cos x\sin y\cos z,\quad w_0 = 0",
        r"p_0 = \tfrac{1}{16}\!\left(\cos 2x + \cos 2y\right)\!\left(\cos 2z + 2\right)",
    ],
    "darcy_flow_3d": [
        r"-\nabla\cdot(K\nabla p) = f",
        r"\mathbf{u} = -K\nabla p",
        r"p_{\text{exact}} = \sin(\pi x)\sin(\pi y)\sin(\pi z)",
    ],
    "stokes_sphere": [
        r"\mu\nabla^2\mathbf{u} = \nabla p,\quad\nabla\cdot\mathbf{u} = 0",
        r"u_r = U_\infty\!\left(1 - \frac{3R}{2r} + \frac{R^3}{2r^3}\right)\!\cos\theta",
        r"u_\theta = -U_\infty\!\left(1 - \frac{3R}{4r} - \frac{R^3}{4r^3}\right)\!\sin\theta",
        r"p = p_\infty - \frac{3\mu U_\infty R\cos\theta}{2r^2}",
    ],
    "lid_driven_cavity": [
        r"\rho\!\left(\frac{\partial\mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p + \mu\nabla^2\mathbf{u}",
        r"\nabla\cdot\mathbf{u} = 0",
    ],
    "bernoulli_flow": [
        r"p + \tfrac{1}{2}\rho v^2 + \rho g h = C",
        r"\rho v A = \mathrm{const}\quad\text{(continuity)}",
    ],
    "stokes_flow": [
        r"-\nabla p + \mu\nabla^2\mathbf{u} = \mathbf{0},\quad\nabla\cdot\mathbf{u} = 0",
        r"-\frac{\partial p}{\partial x} + \mu\!\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0",
        r"-\frac{\partial p}{\partial y} + \mu\!\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0",
        r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0",
    ],
    "darcy_flow": [
        r"\mathbf{u} = -\frac{k}{\mu}\nabla p",
        r"\nabla\cdot\mathbf{u} = f",
        r"-\nabla\cdot\!\left(\frac{k}{\mu}\nabla p\right) = f",
    ],
    "channel_poiseuille": [
        r"\mu\frac{\partial^2 u}{\partial y^2} = \frac{\partial p}{\partial x} = -G",
        r"v = 0",
        r"u(y) = \frac{G}{2\mu}\!\left(H^2 - y^2\right)\quad\text{(exact)}",
    ],
    "linear_elasticity_2d": [
        r"\nabla\cdot\boldsymbol{\sigma} + \mathbf{f} = \mathbf{0}",
        r"\boldsymbol{\sigma} = \mathbf{C}:\boldsymbol{\varepsilon},\quad\boldsymbol{\varepsilon} = \tfrac{1}{2}\!\left(\nabla\mathbf{u} + \nabla\mathbf{u}^\top\right)",
        r"\frac{\partial\sigma_{xx}}{\partial x} + \frac{\partial\sigma_{xy}}{\partial y} + f_x = 0",
        r"\frac{\partial\sigma_{xy}}{\partial x} + \frac{\partial\sigma_{yy}}{\partial y} + f_y = 0",
    ],
    "linear_elasticity_3d": [
        r"\nabla\cdot\boldsymbol{\sigma} + \mathbf{f} = \mathbf{0}",
        r"\sigma_{ij} = \lambda\,\varepsilon_{kk}\delta_{ij} + 2\mu\,\varepsilon_{ij}",
        r"\varepsilon_{ij} = \tfrac{1}{2}\!\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)",
    ],
    "crash_analysis": [
        r"\nabla\cdot\boldsymbol{\sigma} + \rho\mathbf{b} = \rho\mathbf{a}",
        r"\boldsymbol{\sigma} = f(\mathbf{F},\,\text{history})\quad(\text{e.g. J}_{2}\text{ plasticity})",
        r"\mathbf{F} = \frac{\partial\mathbf{x}}{\partial\mathbf{X}},\quad J = \det(\mathbf{F}) > 0",
    ],
    "beam_bending": [
        r"EI\frac{\partial^4 w}{\partial x^4} = q(x)",
        r"M = EI\frac{\partial^2 w}{\partial x^2},\quad V = EI\frac{\partial^3 w}{\partial x^3}",
    ],
    "thermoelasticity_2d": [
        r"\rho c_p\frac{\partial T}{\partial t} = \nabla\cdot(\kappa\nabla T) + Q",
        r"\nabla\cdot\boldsymbol{\sigma} + \mathbf{f} = \mathbf{0}",
        r"\boldsymbol{\sigma} = \mathbf{C}:(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^T),\quad\boldsymbol{\varepsilon}^T = \alpha(T-T_0)\mathbf{I}",
    ],
    "poisson_electrostatics": [
        r"-\nabla\cdot(\varepsilon\nabla\varphi) = \rho_f",
        r"\mathbf{E} = -\nabla\varphi,\quad\mathbf{D} = \varepsilon\mathbf{E}",
        r"-\!\left(\frac{\partial^2\varphi}{\partial x^2} + \frac{\partial^2\varphi}{\partial y^2}\right) = \frac{\rho_f}{\varepsilon_0}",
    ],
    "helmholtz_wave": [
        r"\nabla^2 u + k^2 u = f(x,y)",
        r"\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + k^2 u = f",
        r"k = \frac{\omega}{c}",
    ],
    "laplace_potential": [
        r"\nabla^2\varphi = 0",
        r"\frac{\partial^2\varphi}{\partial x^2} + \frac{\partial^2\varphi}{\partial y^2} = 0",
        r"\mathbf{E} = -\nabla\varphi",
    ],
    "magnetostatics_2d": [
        r"\nabla\times\mathbf{H} = \mathbf{J},\quad\nabla\cdot\mathbf{B} = 0",
        r"\mathbf{B} = \mu\mathbf{H}",
        r"\nabla^2 A_z = -\mu J_z",
        r"B_x = \frac{\partial A_z}{\partial y},\quad B_y = -\frac{\partial A_z}{\partial x}",
    ],
    "hartmann_mhd": [
        r"\mu\nabla^2 u - \sigma B^2 u = \frac{\partial p}{\partial x}",
        r"\nabla\cdot\mathbf{u} = 0",
        r"u(y) = \frac{GH^2}{\mu\,\mathrm{Ha}^2}\!\left(1 - \frac{\cosh(\mathrm{Ha}\cdot y/H)}{\cosh(\mathrm{Ha})}\right)",
        r"\mathrm{Ha} = BH\sqrt{\sigma/\mu}",
    ],
    "schrodinger_stationary": [
        r"-\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V(x)\psi = E\psi",
        r"-\frac{\hbar^2}{2m}\psi'' + V(x)\psi - E\psi = 0",
        r"\int|\psi|^2\,dx = 1\quad\text{(normalization)}",
    ],
    "schrodinger_timedep": [
        r"i\hbar\frac{\partial\psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi",
        r"\rho = |\psi|^2 = \psi_R^2 + \psi_I^2",
        r"\frac{\partial\rho}{\partial t} + \frac{\partial J}{\partial x} = 0\quad\text{(probability conservation)}",
    ],
    "gross_pitaevskii": [
        r"i\hbar\frac{\partial\psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x) + g|\psi|^2\right]\psi",
        r"N = \int|\psi|^2\,dx,\quad g = \frac{4\pi\hbar^2 a_s}{m}",
    ],
    "chip_heat_3d": [
        r"\frac{d^2T}{dz^2} - m^2(T - T_{\mathrm{amb}}) = 0",
        r"m = \sqrt{\frac{2h}{k\,t_f}},\quad T(z) = T_{\mathrm{amb}} + (T_{\mathrm{base}} - T_{\mathrm{amb}})\frac{\cosh(m(H-z))}{\cosh(mH)}",
        r"\eta_{\mathrm{fin}} = \frac{\tanh(mH)}{mH}",
    ],
    "heat_equation_1d": [
        r"\frac{\partial T}{\partial t} = \alpha\frac{\partial^2 T}{\partial x^2}",
        r"\alpha = \frac{\kappa}{\rho c_p}\quad\text{(thermal diffusivity)}",
    ],
    "heat_equation_2d": [
        r"\frac{\partial T}{\partial t} = \alpha\!\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right) + \frac{Q}{\rho c_p}",
        r"\nabla^2 T = -\frac{Q}{\kappa}\quad\text{(steady state)}",
        r"\alpha = \frac{\kappa}{\rho c_p}",
    ],
    "wave_equation": [
        r"\frac{\partial^2 u}{\partial t^2} = c^2\!\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)",
        r"\omega^2 = c^2(k_x^2 + k_y^2)\quad\text{(dispersion relation)}",
    ],
    "helmholtz_3d_cavity": [
        r"\nabla^2 p + k^2 p = 0,\quad\left.\frac{\partial p}{\partial n}\right|_{\partial\Omega} = 0",
        r"p_{nml}(x,y,z) = \cos\frac{n\pi x}{a}\cos\frac{m\pi y}{b}\cos\frac{l\pi z}{c}",
        r"k_{\mathrm{res}} = \pi\sqrt{\frac{n^2}{a^2}+\frac{m^2}{b^2}+\frac{l^2}{c^2}}",
    ],
    "klein_gordon": [
        r"\frac{\partial^2\psi}{\partial t^2} - c^2\frac{\partial^2\psi}{\partial x^2} + \frac{m^2c^4}{\hbar^2}\psi = 0",
        r"(\Box + m^2)\psi = 0\quad(c=\hbar=1)",
        r"\omega^2 = c^2 k^2 + \frac{m^2 c^4}{\hbar^2}",
    ],
    "kdv_equation": [
        r"\frac{\partial u}{\partial t} + 6u\frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0",
        r"u(x,t) = \frac{c}{2}\operatorname{sech}^2\!\!\left(\sqrt{\tfrac{c}{4}}\,(x - ct)\right)\quad\text{(1-soliton)}",
    ],
    "airfoil_naca0012_2d": [
        r"\nabla^2\varphi = 0\quad\text{(potential flow)}",
        r"C_p = 1 - \frac{V^2}{V_\infty^2}",
        r"\left.\frac{\partial\varphi}{\partial n}\right|_{\text{surface}} = 0,\quad\gamma = \pi V_\infty c\sin\alpha",
    ],
    "shallow_water_2d": [
        r"\frac{\partial\eta}{\partial t} + h_0\nabla\cdot\mathbf{u} = 0",
        r"\frac{\partial\mathbf{u}}{\partial t} = -g\nabla\eta",
        r"\frac{\partial^2\eta}{\partial t^2} = c^2\nabla^2\eta,\quad c = \sqrt{g h_0}",
    ],
}

for _pid, _latex in _LATEX_EQ_OVERRIDES.items():
    if _pid in PROBLEMS_CATALOG and not PROBLEMS_CATALOG[_pid].latex_equations:
        PROBLEMS_CATALOG[_pid] = PROBLEMS_CATALOG[_pid].model_copy(update={"latex_equations": _latex})
