"""
Pre-built 3D geometry catalog for the PINNeAPPle Arena.

Exposes GEOMETRIES_CATALOG: dict[str, GeometryPreset] with 8 ready-to-use
3D domains, and a helper function get_geometry_mesh_config() that returns
the mesh configuration dict compatible with PINNeAPPle's geometry builders.
"""
from __future__ import annotations

from pinneapple_arena.pointwise.schema import GeometryPreset, GeometryType

# ---------------------------------------------------------------------------
# 1. Cube domain
# ---------------------------------------------------------------------------

cube_domain = GeometryPreset(
    id="cube_domain",
    name="Cubic Domain [0,1]³",
    description=(
        "A simple axis-aligned unit cube spanning [0,1]³. Suitable as a general-purpose "
        "domain for 3D PINN problems including heat conduction, elasticity, and "
        "electromagnetic field problems where the domain shape is not critical."
    ),
    geometry_type=GeometryType.BOX,
    dimensions={
        "Lx": 1.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "x_min": 0.0,
        "y_min": 0.0,
        "z_min": 0.0,
    },
    supported_problems=[
        "linear_elasticity_3d",
        "heat_equation_2d",
        "poisson_electrostatics",
        "laplace_potential",
        "magnetostatics_2d",
        "thermoelasticity_2d",
    ],
    mesh_resolution_default=20,
    image_placeholder="cube_domain.png",
)

# ---------------------------------------------------------------------------
# 2. Cylinder domain
# ---------------------------------------------------------------------------

cylinder_domain = GeometryPreset(
    id="cylinder_domain",
    name="Cylinder (R=1, H=2)",
    description=(
        "Solid circular cylinder with radius R = 1 m and height H = 2 m, axis aligned "
        "with z. Suitable for shaft torsion, rotating machinery, axisymmetric heat "
        "conduction, and pipe cross-section problems."
    ),
    geometry_type=GeometryType.CYLINDER,
    dimensions={
        "radius": 1.0,
        "height": 2.0,
        "axis": "z",
        "x_center": 0.0,
        "y_center": 0.0,
        "z_min": 0.0,
    },
    supported_problems=[
        "linear_elasticity_3d",
        "heat_equation_2d",
        "thermoelasticity_2d",
        "crash_analysis",
    ],
    mesh_resolution_default=15,
    image_placeholder="cylinder_domain.png",
)

# ---------------------------------------------------------------------------
# 3. Sphere domain
# ---------------------------------------------------------------------------

sphere_domain = GeometryPreset(
    id="sphere_domain",
    name="Sphere (R=1)",
    description=(
        "Unit sphere centered at the origin. Ideal for isotropic field problems with "
        "spherical symmetry: electrostatic potential, gravitational field, and "
        "elastic deformation of spherical components. Can exploit symmetry for "
        "reduced domains."
    ),
    geometry_type=GeometryType.SPHERE,
    dimensions={
        "radius": 1.0,
        "x_center": 0.0,
        "y_center": 0.0,
        "z_center": 0.0,
    },
    supported_problems=[
        "poisson_electrostatics",
        "laplace_potential",
        "linear_elasticity_3d",
        "heat_equation_2d",
    ],
    mesh_resolution_default=15,
    image_placeholder="sphere_domain.png",
)

# ---------------------------------------------------------------------------
# 4. Pipe flow domain
# ---------------------------------------------------------------------------

pipe_flow = GeometryPreset(
    id="pipe_flow",
    name="Circular Pipe (hollow cylinder)",
    description=(
        "Hollow cylindrical pipe domain suitable for 3D internal Navier-Stokes flow. "
        "Inner radius R_in = 0.1 m, outer radius R_out = 1 m (for thick-walled "
        "structural studies) or purely the fluid interior (R_in = 0, R_out = 0.05 m). "
        "Length L = 2 m, flow along z-axis."
    ),
    geometry_type=GeometryType.PIPE,
    dimensions={
        "radius_inner": 0.0,
        "radius_outer": 0.05,
        "length": 2.0,
        "axis": "z",
        "x_center": 0.0,
        "y_center": 0.0,
        "z_min": 0.0,
    },
    supported_problems=[
        "navier_stokes_3d",
        "channel_poiseuille",
        "stokes_flow",
        "darcy_flow",
        "linear_elasticity_3d",
    ],
    mesh_resolution_default=18,
    image_placeholder="pipe_flow.png",
)

# ---------------------------------------------------------------------------
# 5. Rectangular channel 3D
# ---------------------------------------------------------------------------

channel_3d = GeometryPreset(
    id="channel_3d",
    name="Rectangular Channel 3D (4×1×1)",
    description=(
        "Rectangular duct with dimensions 4 m (x) × 1 m (y) × 1 m (z). Flow enters "
        "at x = 0 and exits at x = 4. The aspect ratio 4:1:1 accommodates "
        "fully-developed channel flow with sufficient upstream length for inlet "
        "effects to decay."
    ),
    geometry_type=GeometryType.CHANNEL,
    dimensions={
        "Lx": 4.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "x_min": 0.0,
        "y_min": 0.0,
        "z_min": 0.0,
    },
    supported_problems=[
        "navier_stokes_3d",
        "channel_poiseuille",
        "stokes_flow",
        "heat_equation_2d",
        "darcy_flow",
    ],
    mesh_resolution_default=20,
    image_placeholder="channel_3d.png",
)

# ---------------------------------------------------------------------------
# 6. Lid-driven cavity 3D
# ---------------------------------------------------------------------------

lid_driven_cavity_3d = GeometryPreset(
    id="lid_driven_cavity_3d",
    name="Lid-Driven Cubic Cavity (1×1×1)",
    description=(
        "Cubic cavity [0,1]³ where the top face (z = 1) moves at constant velocity "
        "U = 1 m/s in the x-direction, driving 3D recirculation vortices. The 3D "
        "extension of the classic 2D benchmark reveals spanwise Taylor-Görtler "
        "instabilities at moderate Reynolds numbers."
    ),
    geometry_type=GeometryType.CAVITY,
    dimensions={
        "Lx": 1.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "x_min": 0.0,
        "y_min": 0.0,
        "z_min": 0.0,
        "lid_face": "z_max",
        "lid_velocity_x": 1.0,
    },
    supported_problems=[
        "navier_stokes_3d",
        "lid_driven_cavity",
        "stokes_flow",
    ],
    mesh_resolution_default=16,
    image_placeholder="lid_driven_cavity_3d.png",
)

# ---------------------------------------------------------------------------
# 7. L-bracket 3D
# ---------------------------------------------------------------------------

l_bracket_3d = GeometryPreset(
    id="l_bracket_3d",
    name="L-Shaped Bracket (Structural)",
    description=(
        "L-shaped structural bracket formed by two rectangular beams joined at a "
        "right angle. Arm 1: 2 m × 0.5 m × 0.5 m (along x); Arm 2: 0.5 m × 2 m × 0.5 m "
        "(along y). Classic benchmark for stress concentration analysis around "
        "re-entrant corners in linear elasticity."
    ),
    geometry_type=GeometryType.L_BRACKET,
    dimensions={
        "arm1_Lx": 2.0,
        "arm1_Ly": 0.5,
        "arm1_Lz": 0.5,
        "arm2_Lx": 0.5,
        "arm2_Ly": 2.0,
        "arm2_Lz": 0.5,
        "thickness": 0.5,
    },
    supported_problems=[
        "linear_elasticity_3d",
        "crash_analysis",
        "thermoelasticity_2d",
    ],
    mesh_resolution_default=12,
    image_placeholder="l_bracket_3d.png",
)

# ---------------------------------------------------------------------------
# 8. NACA 0012 airfoil wing (extruded)
# ---------------------------------------------------------------------------

airfoil_wing = GeometryPreset(
    id="airfoil_wing",
    name="NACA 0012 Extruded Wing",
    description=(
        "NACA 0012 symmetric airfoil section extruded in the z-direction (span) to "
        "create a finite-span wing. Chord length c = 1 m, span b = 2 m, angle of "
        "attack α = 0°. The surrounding fluid domain is a C-type box "
        "10c upstream, 15c downstream, and 10c in height."
    ),
    geometry_type=GeometryType.AIRFOIL,
    dimensions={
        "chord": 1.0,
        "span": 2.0,
        "angle_of_attack_deg": 0.0,
        "domain_upstream": 10.0,
        "domain_downstream": 15.0,
        "domain_height": 10.0,
        "naca_series": "0012",
        "n_profile_points": 200,
    },
    supported_problems=[
        "navier_stokes_3d",
        "stokes_flow",
        "linear_elasticity_3d",
        "crash_analysis",
    ],
    mesh_resolution_default=25,
    image_placeholder="airfoil_wing.png",
)


# ---------------------------------------------------------------------------
# Mesh config builder
# ---------------------------------------------------------------------------

def get_geometry_mesh_config(geometry_id: str, resolution: int) -> dict:
    """
    Return a mesh configuration dictionary for a given geometry preset and
    resolution, suitable for use with PINNeAPPle geometry builders.

    Parameters
    ----------
    geometry_id : str
        Key into GEOMETRIES_CATALOG.
    resolution : int
        Number of collocation points per side (used to compute total point counts).

    Returns
    -------
    dict
        Mesh configuration dict consumable by PINNeAPPle's domain builders
        (e.g., STLDomainBatchBuilder, PhysicsDomain3D, ChannelDomain3D, etc.).

    Raises
    ------
    KeyError
        If geometry_id is not found in the catalog.
    """
    if geometry_id not in GEOMETRIES_CATALOG:
        raise KeyError(
            f"Geometry '{geometry_id}' not found. "
            f"Available: {list(GEOMETRIES_CATALOG.keys())}"
        )

    geo = GEOMETRIES_CATALOG[geometry_id]
    dims = geo.dimensions
    n_interior = resolution ** 3
    n_boundary = 6 * (resolution ** 2)

    _base = {
        "geometry_id": geometry_id,
        "geometry_type": geo.geometry_type.value,
        "resolution": resolution,
        "n_collocation": n_interior,
        "n_boundary": n_boundary,
        "dimensions": dims,
    }

    if geo.geometry_type == GeometryType.BOX:
        return {
            **_base,
            "builder": "BoxDomain3D",
            "bounds": {
                "x": (dims.get("x_min", 0.0), dims.get("x_min", 0.0) + dims["Lx"]),
                "y": (dims.get("y_min", 0.0), dims.get("y_min", 0.0) + dims["Ly"]),
                "z": (dims.get("z_min", 0.0), dims.get("z_min", 0.0) + dims["Lz"]),
            },
            "sample_mode": "latin_hypercube",
        }

    if geo.geometry_type == GeometryType.CYLINDER:
        return {
            **_base,
            "builder": "CylinderDomain3D",
            "radius": dims["radius"],
            "height": dims["height"],
            "axis": dims.get("axis", "z"),
            "center": (dims.get("x_center", 0.0), dims.get("y_center", 0.0)),
            "z_min": dims.get("z_min", 0.0),
            "sample_mode": "rejection",
        }

    if geo.geometry_type == GeometryType.SPHERE:
        return {
            **_base,
            "builder": "SphereDomain3D",
            "radius": dims["radius"],
            "center": (
                dims.get("x_center", 0.0),
                dims.get("y_center", 0.0),
                dims.get("z_center", 0.0),
            ),
            "sample_mode": "rejection",
        }

    if geo.geometry_type == GeometryType.PIPE:
        return {
            **_base,
            "builder": "PipeDomain3D",
            "radius_inner": dims["radius_inner"],
            "radius_outer": dims["radius_outer"],
            "length": dims["length"],
            "axis": dims.get("axis", "z"),
            "center": (dims.get("x_center", 0.0), dims.get("y_center", 0.0)),
            "z_min": dims.get("z_min", 0.0),
            "inlet_tag": "inlet",
            "outlet_tag": "outlet",
            "wall_tag": "wall",
            "sample_mode": "rejection",
        }

    if geo.geometry_type == GeometryType.CHANNEL:
        return {
            **_base,
            "builder": "ChannelDomain3D",
            "bounds": {
                "x": (dims.get("x_min", 0.0), dims.get("x_min", 0.0) + dims["Lx"]),
                "y": (dims.get("y_min", 0.0), dims.get("y_min", 0.0) + dims["Ly"]),
                "z": (dims.get("z_min", 0.0), dims.get("z_min", 0.0) + dims["Lz"]),
            },
            "flow_axis": "x",
            "inlet_tag": "inlet",
            "outlet_tag": "outlet",
            "wall_tags": ["y_min", "y_max", "z_min", "z_max"],
            "sample_mode": "uniform_grid",
        }

    if geo.geometry_type == GeometryType.CAVITY:
        return {
            **_base,
            "builder": "CavityDomain3D",
            "bounds": {
                "x": (dims.get("x_min", 0.0), dims.get("x_min", 0.0) + dims["Lx"]),
                "y": (dims.get("y_min", 0.0), dims.get("y_min", 0.0) + dims["Ly"]),
                "z": (dims.get("z_min", 0.0), dims.get("z_min", 0.0) + dims["Lz"]),
            },
            "moving_face": dims.get("lid_face", "z_max"),
            "lid_velocity": {
                "x": dims.get("lid_velocity_x", 1.0),
                "y": 0.0,
                "z": 0.0,
            },
            "sample_mode": "latin_hypercube",
        }

    if geo.geometry_type == GeometryType.L_BRACKET:
        return {
            **_base,
            "builder": "LBracketDomain3D",
            "arm1": {
                "Lx": dims["arm1_Lx"],
                "Ly": dims["arm1_Ly"],
                "Lz": dims["arm1_Lz"],
            },
            "arm2": {
                "Lx": dims["arm2_Lx"],
                "Ly": dims["arm2_Ly"],
                "Lz": dims["arm2_Lz"],
            },
            "clamped_face": "x_min",
            "load_face": "arm2_y_max",
            "sample_mode": "latin_hypercube",
        }

    if geo.geometry_type == GeometryType.AIRFOIL:
        return {
            **_base,
            "builder": "AirfoilWingDomain3D",
            "naca_series": dims.get("naca_series", "0012"),
            "chord": dims["chord"],
            "span": dims["span"],
            "angle_of_attack_deg": dims.get("angle_of_attack_deg", 0.0),
            "domain_upstream": dims.get("domain_upstream", 10.0),
            "domain_downstream": dims.get("domain_downstream", 15.0),
            "domain_height": dims.get("domain_height", 10.0),
            "n_profile_points": dims.get("n_profile_points", 200),
            "farfield_tag": "farfield",
            "airfoil_tag": "airfoil",
            "inlet_tag": "inlet",
            "outlet_tag": "outlet",
            "sample_mode": "curvature_adapted",
        }

    # Fallback for any unlisted geometry type
    return {**_base, "builder": "GenericDomain3D", "sample_mode": "latin_hypercube"}


# ---------------------------------------------------------------------------
# 9. Flow Past Cylinder (2D external flow benchmark)
# ---------------------------------------------------------------------------

flow_past_cylinder_2d = GeometryPreset(
    id="flow_past_cylinder_2d",
    name="Flow Past Cylinder (2D Benchmark)",
    description=(
        "Rectangular channel domain with a circular cylinder obstacle — the canonical "
        "CFD validation geometry (Williamson 1989, Raissi et al. 2019). "
        "Domain: x ∈ [-8D, 25D], y ∈ [-8D, 8D] with cylinder centred at origin "
        "(diameter D = 1). Inlet at x = -8D, outlet at x = 25D, slip walls at y = ±8D. "
        "Used to validate drag coefficient Cd, Strouhal number St, and recirculation "
        "length Lw/D at Re = 20–180."
    ),
    geometry_type=GeometryType.CHANNEL,
    dimensions={
        "Lx": 33.0,       # total x-length: 8D upstream + 25D downstream
        "Ly": 16.0,       # total y-height: ±8D
        "Lz": 1.0,        # unit depth (2-D slice)
        "x_min": -8.0,    # inlet at -8D
        "y_min": -8.0,
        "z_min": 0.0,
        # Cylinder obstacle parameters (used by PINN domain samplers)
        "obstacle_type": "cylinder",
        "cylinder_x": 0.0,
        "cylinder_y": 0.0,
        "cylinder_radius": 0.5,   # D = 1 → R = 0.5
    },
    supported_problems=["cylinder_flow_2d", "navier_stokes_2d"],
    mesh_resolution_default=40,
    image_placeholder="flow_past_cylinder_2d.png",
)


# ---------------------------------------------------------------------------
# Assembled catalog
# ---------------------------------------------------------------------------

GEOMETRIES_CATALOG: dict[str, GeometryPreset] = {
    "cube_domain": cube_domain,
    "cylinder_domain": cylinder_domain,
    "sphere_domain": sphere_domain,
    "pipe_flow": pipe_flow,
    "channel_3d": channel_3d,
    "lid_driven_cavity_3d": lid_driven_cavity_3d,
    "l_bracket_3d": l_bracket_3d,
    "airfoil_wing": airfoil_wing,
    "flow_past_cylinder_2d": flow_past_cylinder_2d,
}

assert len(GEOMETRIES_CATALOG) == 9, (
    f"Expected 9 geometries, got {len(GEOMETRIES_CATALOG)}"
)
