"""2-D viscous vortex domains (VVD): incompressible Navier-Stokes with mesh-free Lagrangian vorticity.

The method, after Dynnikova and Dynnikov: the vorticity of the flow is carried by *domains* -- blobs of circulation
``gamma`` and core radius ``eps`` -- which move with the velocity the whole field induces on them (free stream plus
the regularised Biot-Savart sum), while viscosity enters as a deterministic *drift* down the vorticity gradient,
both between the domains and from the walls. The body is a closed polygon cut into panels: a source sheet on them
keeps the flow out, and the wall hands vorticity to the fluid at the rate Lighthill's relation asks for, which is
what builds the boundary layer, its separation and the wake.

That is a solver with no mesh, no turbulence model and no user parameter that decides where vorticity goes.

* :mod:`.bodies`   -- rigid bodies (polygons, rotating and hinged) and their generators: cylinder, ellipse, plate,
  NACA 4-digit sections, a deploying spoiler.
* :mod:`.domains`  -- the set of vortex domains, their core radii, the merge that bounds their number, and removal
  that reports the circulation it takes out.
* :mod:`.kernels`  -- the integrals: the regularised blob kernel, the closed-form panel sheets, the diffusion drifts.
* :mod:`.boundary` -- the wall: source sheet for no through flow, vortex sheet for no slip, and Lighthill's flux.
* :mod:`.solver`   -- :class:`VVD2D`, the time loop, the forces and the history of a run.
* :mod:`.fields`   -- velocity, vorticity, pressure, stream function, streamlines, wall ``Cp`` and ``Cf``.
* :mod:`.cases`    -- the cylinder (verification) and the NACA 0012 with a deploying spoiler (the reference case).
* :mod:`.study`    -- run a case, reduce it to coefficients and a Strouhal number, decide on the physics checks.
* :mod:`.verification` -- what the solver is checked against, and the numbers it reproduces.

    from pinneapple_simulation.vortex_domains import cylinder_case, run_study

    result = run_study(cylinder_case(re=40, panels=96, t_end=10.0))
    print(result.summary())        # Cl, Cd, Cm, and which checks failed
    for check in result.checks:    # impermeability and displacement pass; no_slip and drag do not (see the guide)
        print(check["name"], check["ok"], check["detail"])

The two references the module is held to: the drag of a cylinder at ``Re = 40``, ``Cd = 1.3375`` (Dennis & Chang,
*J. Fluid Mech.* 42, 1970) and its shedding at ``Re = 200``, ``St = 0.1968`` (Williamson & Brown, *J. Fluids
Struct.* 12, 1998). ``python -m pinneapple_simulation.vortex_domains`` prints what this checkout reproduces
(``docs/guides/vortex_domains.md``).
"""
from __future__ import annotations

from .bodies import (
    Assembly,
    Body,
    Motion,
    PanelSet,
    cylinder,
    deploy,
    deploying_spoiler,
    ellipse,
    fixed,
    hinge,
    inside_polygon,
    naca0012,
    naca4,
    plate,
    polygon,
    rotating,
    translating,
)
from .boundary import (
    SourceSheet,
    VortexSheet,
    create_vorticity,
    edge_gradient,
    edge_velocity,
    surface_slip,
    vorticity_flux,
)
from .cases import Case, cylinder_case, naca0012_spoiler_case, spoiler_hinge
from .domains import Domains
from .fields import (
    FieldGrid,
    grid_field,
    streamlines,
    surface_friction,
    surface_pressure,
    vorticity_field,
)
from .solver import VVD2D, VVDConfig, step_size
from .study import StudyResult, run_study, strouhal_number

__all__ = [
    "Assembly",
    "Body",
    "Case",
    "Domains",
    "FieldGrid",
    "Motion",
    "PanelSet",
    "SourceSheet",
    "StudyResult",
    "VVD2D",
    "VVDConfig",
    "VortexSheet",
    "create_vorticity",
    "cylinder",
    "cylinder_case",
    "deploy",
    "deploying_spoiler",
    "edge_gradient",
    "edge_velocity",
    "ellipse",
    "fixed",
    "grid_field",
    "hinge",
    "inside_polygon",
    "naca0012",
    "naca0012_spoiler_case",
    "naca4",
    "plate",
    "polygon",
    "rotating",
    "run_study",
    "spoiler_hinge",
    "step_size",
    "streamlines",
    "strouhal_number",
    "surface_friction",
    "surface_pressure",
    "surface_slip",
    "translating",
    "vorticity_field",
    "vorticity_flux",
]
