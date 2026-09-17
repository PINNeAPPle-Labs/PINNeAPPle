"""Real-geometry fixtures for ``selector_type="tag"`` presets.

Background
----------
``solve_pde()`` raises ``pinneapple_physics.TagConditionsUnresolved`` for
any ``ProblemSpec`` with ``selector_type="tag"`` conditions unless the
caller supplies real geometry (``x_bc``/``y_bc``/``ctx["tag_masks"]`` plus
a ``mask_<condition_name>`` per tag condition) -- see that exception's
docstring and ``docs/dev/AUDIT_REPORT.md``. Of the 65 registered presets,
40 have at least one ``selector_type="tag"`` condition.

This module is the preset-by-preset engineering judgment call the fix
required: for EACH of those 40, is the preset's own domain a canonical
box/rectangle/cylinder fully described by its own parameters, AND does
each tag name map to one specific face of that shape without guessing at
anything the preset doesn't already state? Only presets that clear BOTH
bars get an entry here. The rest are deliberately left alone --
``TagConditionsUnresolved`` continues to be the correct, honest behavior
for them, because fabricating a plausible-looking box/cylinder for e.g. a
real airfoil or a real furnace refractory geometry would silently produce
physically meaningless (if superficially successful-looking) training
runs, exactly the failure mode this whole fix exists to prevent.

``TAG_GEOMETRY_FIXTURES``: preset name -> a spec dict consumed by
:func:`build_tag_batch`:

    {"shape": "box", "tag_faces": {tag: [(coord, "min"|"max"), ...], ...}}
    {"shape": "cylinder", "axis_coord": ..., "cross_coords": (c0, c1),
     "radius_param": <preset kwarg name, or None to read spec.domain_bounds>,
     "tag_faces": {tag: "inlet"|"outlet"|"wall", ...}}

Every entry below has an inline comment naming the exact text (docstring,
inline comment, or established repo-wide naming convention) the face
mapping is read from -- not invented.

``NOT_FIXABLE_WITHOUT_REAL_GEOMETRY``: the other tag-based presets, each
with a one-line reason (used only for documentation/reporting -- the
actual not-fixable behavior is simply "this module has no entry for it,"
``TagConditionsUnresolved`` still fires).

``CHOSEN_CONVENTIONS`` (added in the 2026-09-17 follow-up pass): a small
number of fixtures below map a tag whose face the preset's own text
genuinely does NOT specify (e.g. "fixed"/"load" on an otherwise-symmetric
unit cube) -- for those, and ONLY those, this pass made an explicit
engineering decision (the canonical cantilever convention, the standard
disc-brake friction/cooling assignment) instead of reading an existing
fact off the preset. Every such fixture is marked in its own inline
comment too; ``CHOSEN_CONVENTIONS`` collects the reasoning in one place so
it can never be mistaken for the preset's own text the way every other
entry's comment is.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..spec import ProblemSpec

# ---------------------------------------------------------------------------
# established repo-wide convention (confirmed by 3+ independent presets in
# cfd.py: ns_incompressible_2d, channel_flow_3d, pipe_flow_3d): "inlet" is
# the domain-min face along the flow/streamwise axis (here always axis 0,
# i.e. "x"), "outlet" is the domain-max face of that same axis. Presets
# below that reuse "inlet"/"outlet" for a different physical quantity
# (temperature, pressure) on the same axis are applying the SAME
# established convention, not inventing a new one.
# ---------------------------------------------------------------------------

TAG_GEOMETRY_FIXTURES: Dict[str, Dict[str, Any]] = {
    # ---- CFD: box/rectangle domains with docstring-literal face maps ----
    "channel_flow_3d": {
        # Docstring: "Inlet at x=0... Outlet at x=length... Four walls
        # (y=0, y=H, z=0, z=W): no-slip."
        "shape": "box",
        "tag_faces": {
            "inlet": [("x", "min")],
            "outlet": [("x", "max")],
            "walls": [("y", "min"), ("y", "max"), ("z", "min"), ("z", "max")],
        },
    },
    "lid_driven_cavity_3d": {
        # Docstring: "The lid face at z=size moves... remaining five faces
        # are no-slip walls."
        "shape": "box",
        "tag_faces": {
            "lid": [("z", "max")],
            "walls": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max"), ("z", "min")],
        },
    },
    "ns_incompressible_2d": {
        # Same construction already validated end-to-end in
        # examples/pde_environment/03_ns2d_channel_tags.py (inlet=x=0,
        # outlet=x=1, walls=y=0/y=1); this fixture is that same mapping,
        # expressed generically.
        "shape": "box",
        "tag_faces": {
            "inlet": [("x", "min")],
            "outlet": [("x", "max")],
            "walls": [("y", "min"), ("y", "max")],
        },
    },
    "furnace_combustion_zone": {
        # Docstring/params name the domain "furnace_length x
        # furnace_height" (a 2D duct); tags fuel_inlet/flue_outlet/
        # refractory_wall follow the exact same inlet=x_min/outlet=x_max/
        # walls=y_min+y_max convention as channel_flow_3d above -- this is
        # structurally a heated 2D channel, not a real furnace chamber
        # shape (contrast with industrial_furnace_thermal, NOT fixable,
        # below).
        "shape": "box",
        "tag_faces": {
            "fuel_inlet": [("x", "min")],
            "flue_outlet": [("x", "max")],
            "refractory_wall": [("y", "min"), ("y", "max")],
        },
    },

    # ---- Pipe: circular cylinder, docstring-literal ----
    "pipe_flow_3d": {
        # Docstring: "The pipe is a cylinder of radius `radius` aligned
        # with the x-axis... Inlet at x=0... Outlet at x=length...
        # Cylindrical wall: no-slip (applied via a circular tag at
        # r = radius)."
        "shape": "cylinder",
        "axis_coord": "x",
        "cross_coords": ("y", "z"),
        "radius_param": "radius",
        "tag_faces": {"inlet": "inlet", "outlet": "outlet", "wall": "wall"},
    },

    # ---- Single-tag "whole boundary" presets: with only one tag and no
    # other condition, "the boundary" can only mean the entire domain
    # boundary -- there is nothing else for it to disambiguate against. ----
    "laplace_2d": {"shape": "box", "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]}},
    "poisson_2d": {"shape": "box", "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]}},
    "helmholtz_acoustics_3d": {
        "shape": "box",
        "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max"), ("z", "min"), ("z", "max")]},
    },
    "wave_ultrasound_3d": {
        "shape": "box",
        "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max"), ("z", "min"), ("z", "max")]},
    },
    "reaction_diffusion_2d": {"shape": "box", "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]}},
    "transient_heat_3d": {
        "shape": "box",
        "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max"), ("z", "min"), ("z", "max")]},
    },
    "climate_ocean_gyre": {
        # "No-slip streamfunction on all basin walls (psi=0 on boundary)" --
        # a rectangular ocean basin IS the literal Stommel (1948) model
        # domain, not an approximation of a real coastline.
        "shape": "box",
        "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]},
    },
    "opinion_dynamics_2d": {
        # "Zero-flux (Neumann) on all boundaries -- closed society."
        "shape": "box",
        "tag_faces": {"boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]},
    },

    # ---- Single-tag "fixed = whole boundary" structural presets: same
    # logic as above, applied to a zero-Dirichlet displacement tag that is
    # the ONLY condition in the spec (a fully clamped plate/cube; a
    # physically valid, if unexciting, boundary value problem -- no load
    # tag exists to disambiguate a specific fixed face against). ----
    "plane_strain_2d": {"shape": "box", "tag_faces": {"fixed": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]}},
    "von_mises_2d": {"shape": "box", "tag_faces": {"fixed": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")]}},
    "linear_elasticity_3d_industry": {
        "shape": "box",
        "tag_faces": {"fixed": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max"), ("z", "min"), ("z", "max")]},
    },

    "steady_heat_conduction_3d": {
        # Already validated end-to-end in
        # examples/pde_environment/04_heat3d_stl_box.py (via
        # STLDomainBatchBuilder on an actual box mesh, both constant-value
        # tags): "boundary" = whole box surface, "inlet" = the "# Example
        # heater on inlet plane (hot patch)" comment -> x=x_min face, same
        # axis-0 convention as the CFD presets.
        "shape": "box",
        "tag_faces": {
            "boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max"), ("z", "min"), ("z", "max")],
            "inlet": [("x", "min")],
        },
    },

    # ---- Two-tag presets with an unambiguous opposite-face pair ----
    "darcy_pressure_only_3d": {
        # Only "inlet"/"outlet" tags exist (no "wall"): same axis-0
        # convention as the CFD presets, side faces intentionally left
        # unconstrained (natural/zero-flux, not a face this preset tags).
        "shape": "box",
        "tag_faces": {"inlet": [("x", "min")], "outlet": [("x", "max")]},
    },
    "refractory_lining": {
        # Only two conditions, "hot_face"/"cold_face", and the compiled
        # PDE kind ("heat_equation_steady_multilayer") already reduces the
        # multi-layer stack to a single effective k_eff over the whole
        # domain (see AUDIT_REPORT.md) -- so, unlike industrial_furnace_
        # thermal, no internal-layer geometry is actually needed, only the
        # two outer z-faces docstring-implied by "T_hot"/"T_cold" at
        # opposite ends of a 1D-in-z conduction stack. The "x" coordinate
        # is an untagged filler dimension (docstring: "PDE: 1D (in z) or
        # 2D steady-state conduction").
        "shape": "box",
        "tag_faces": {"hot_face": [("z", "min")], "cold_face": [("z", "max")]},
    },

    # ---- Coupled thermal/mechanical presets on a unit box: single "fixed"
    # tag = whole boundary for the mechanical dofs (same logic as above),
    # "inlet"/"outlet" for temperature reusing the same axis-0 convention. ----
    "thermoelasticity_2d": {
        "shape": "box",
        "tag_faces": {
            "fixed": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")],
            "inlet": [("x", "min")],
            "outlet": [("x", "max")],
        },
    },

    # ---- Explicit face-by-name comment in the preset itself ----
    "material_fracture_2d": {
        # Inline comment directly above the conditions: "# Fixed bottom,
        # prescribed top displacement, free sides."
        "shape": "box",
        "tag_faces": {
            "bottom": [("y", "min")],
            "top": [("y", "max")],
            "sides": [("x", "min"), ("x", "max")],
            "boundary": [("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")],
        },
    },

    # ---- Mixed tag+callable presets: the tag face(s) are listed below;
    # the callable conditions self-select and are covered for free because
    # sample_box_tag_batch samples every face point sees a superset of all
    # tagged + implicitly-referenced faces and evaluates every condition's
    # own mask()/values(), tag or callable alike. ----
    "climate_atmosphere_2d": {
        # pole_south/pole_north name lat_min/lat_max literally -- the south
        # and north poles are, by definition of latitude, y=lat_min and
        # y=lat_max.
        "shape": "box",
        "tag_faces": {"pole_south": [("y", "min")], "pole_north": [("y", "max")]},
    },
    "drug_diffusion_tissue": {
        # "# Zero-flux on remaining boundaries (insulated tissue)" -- the
        # callable bc_source already claims x=0 ("# Drug source at x=0");
        # "remaining" is everything else.
        "shape": "box",
        "tag_faces": {"no_flux": [("x", "max"), ("y", "min"), ("y", "max")]},
    },

    # ---- Second follow-up pass (2026-09-17): chosen conventions for
    # "fixed"/"load" and "friction"/"cooling" -- see CHOSEN_CONVENTIONS
    # below and AUDIT_REPORT.md for the explicit statement that these are
    # DECISIONS this pass made, not something inherent to the preset text
    # (unlike every entry above, which cites the preset's own docstring). ----
    "linear_elasticity_3d": {
        # CONVENTION (not preset text): "fixed" = the domain-min face of the
        # first/principal coordinate (x=0, the "base"/"engaste"), "load" =
        # the domain-max face of that same coordinate (x=domain_bounds max,
        # the free tip where the load is applied) -- the canonical
        # cantilever-beam/column setup. domain_bounds is the unit cube.
        "shape": "box",
        "tag_faces": {"fixed": [("x", "min")], "load": [("x", "max")]},
    },
    "plane_stress_2d": {
        # Same CONVENTION as linear_elasticity_3d above, applied to the 2D
        # unit square.
        "shape": "box",
        "tag_faces": {"fixed": [("x", "min")], "load": [("x", "max")]},
    },
    # NOTE: car_brake_thermal is NOT here despite its tag ambiguity being
    # resolved (see CHOSEN_CONVENTIONS below) -- see its entry in
    # NOT_FIXABLE_WITHOUT_REAL_GEOMETRY for why (a second, independent,
    # pre-existing compiler defect, discovered while verifying this fixture,
    # blocks it regardless of geometry).
    "rocket_nozzle_cfd": {
        # A simple conical convergent-divergent nozzle (as instructed:
        # "half-cone angle + throat/exit radii" -- a totally standard,
        # unambiguous nozzle contour -- rather than attempting a Rao bell
        # contour). This preset's own params only give throat_radius,
        # exit_radius, nozzle_length -- no chamber/inlet radius or throat
        # axial position, so a FULL chamber+convergent+divergent contour
        # cannot be built without fabricating one of those. The domain
        # actually modelled here (fully consistent with the preset's own
        # 3 params, nothing invented) is the DIVERGENT section alone,
        # z=0 at the throat itself (r=throat_radius) to z=nozzle_length at
        # the exit (r=exit_radius): a linear (conical) wall r_wall(z) =
        # throat_radius + (exit_radius-throat_radius)*z/nozzle_length,
        # whose half-angle atan((exit_radius-throat_radius)/nozzle_length)
        # is DERIVED from the preset's own numbers, not chosen freely.
        # "inlet" here means the throat plane (stagnation conditions
        # imposed there, a standard nozzle-flow simplification), matching
        # the preset's own p_inlet/T_inlet Dirichlet condition.
        "shape": "cone_axisymmetric",
        "axis_coord": "z",
        "radial_coord": "r",
        "tag_faces": {"inlet": "inlet", "outlet": "outlet", "wall": "wall", "axis": "axis"},
    },
    "axial_compressor_cascade_2d": {
        # Circular-arc cascade blade camber line (Dixon & Hall, "Fluid
        # Mechanics and Thermodynamics of Turbomachinery") -- see
        # _curve_geometry.circular_arc_cascade_blade_points's own docstring
        # for the exact formula and why it needs no parameter beyond this
        # preset's own flow_angle_in_deg/flow_angle_out_deg/chord. "inlet"/
        # "outlet" are selector_type="callable" (self-selecting at x=0/
        # x=domain_x) -- covered for free, same as every mixed preset above.
        "shape": "box_with_curve",
        "shape_coords": ("x", "y"),
        "box_tag_faces": {},
        "curve_tags": ("blade",),
    },
    "car_external_aero": {
        # Ahmed-body-inspired 2D silhouette (Ahmed, Ramm & Faltin 1984,
        # SAE 840300) -- see _curve_geometry.ahmed_body_polygon's own
        # docstring for the exact published ratios/angle used and how they
        # are applied to THIS preset's own car_length/car_height (read back
        # out of domain_bounds, which the preset itself derives from them:
        # x_max=5*car_length, y_max=5*car_height).
        "shape": "box_with_curve",
        "shape_coords": ("x", "y"),
        "box_tag_faces": {
            "inlet": [("x", "min")],
            "outlet": [("x", "max")],
            "ground": [("y", "min")],
            "top": [("y", "max")],
        },
        "curve_tags": ("car_body",),
    },
    "aircraft_wing_structural": {
        # Geometry here was already unambiguous even before this pass (see
        # NOT_FIXABLE's old entry, now removed): "root_fixed" is the wing
        # root (x=0, i.e. domain_bounds' x-min -- "span" runs root..tip by
        # construction), "tip_load" is the free tip (x=span), "free_surface"
        # is the remaining two edges (y=+-thickness/2). What blocked this
        # preset was NEVER the geometry -- it was compile.py's generic
        # Neumann handling not knowing how to resolve the 'tx'/'ty' traction
        # fields against the model's own 'ux'/'uy' displacement fields (see
        # AUDIT_REPORT.md and ConditionSpec.traction_map's docstring); that
        # compiler defect is now fixed, and this preset's own conditions
        # (engineering.py) now declare traction_map explicitly, so the
        # box fixture below is all that's needed on the geometry side.
        "shape": "box",
        "tag_faces": {
            "root_fixed": [("x", "min")],
            "tip_load": [("x", "max")],
            "free_surface": [("y", "min"), ("y", "max")],
        },
    },
}


# Presets fixed in the second follow-up pass (2026-09-17) whose tag->face
# mapping required a genuine engineering DECISION this session made (the
# preset's own text left the tag ambiguous) rather than reading an existing,
# preset-stated fact -- see AUDIT_REPORT.md for the full discussion. Kept
# separate from the inline comments above so this is impossible to miss.
CHOSEN_CONVENTIONS: Dict[str, str] = {
    "linear_elasticity_3d": (
        "'fixed'/'load' have no stated face in the preset. Chose the "
        "canonical cantilever convention: fixed = domain-min face of the "
        "principal axis (x=0), load = domain-max face (x=1)."
    ),
    "plane_stress_2d": (
        "Same as linear_elasticity_3d: fixed = x=0 (domain-min), load = "
        "x=1 (domain-max) -- the canonical cantilever/column convention, "
        "not stated by the preset itself."
    ),
    "car_brake_thermal": (
        "'friction_surface'/'cooling_surface' have no stated face in the "
        "preset. Chose the standard disc-brake assignment: friction = both "
        "flat faces (z=0, z=thickness), cooling = the outer rim (r=disc_radius). "
        "This tag ambiguity IS resolved (see NOT_FIXABLE_WITHOUT_REAL_GEOMETRY's "
        "own entry for why the preset still isn't fully trainable regardless)."
    ),
}


# Preset -> one-line reason it does NOT get a fixture (documentation only;
# behavior is unaffected -- TagConditionsUnresolved fires regardless).
#
# Second follow-up pass (2026-09-17): 7 of the original 17 below were
# closed (aircraft_wing_structural, rocket_nozzle_cfd, car_external_aero,
# axial_compressor_cascade_2d, linear_elasticity_3d, plane_stress_2d), and
# 2 more (car_brake_thermal, rocket_structural) had their ORIGINAL stated
# reason (a tag-face ambiguity / a solid-square-vs-annulus data bug,
# respectively) genuinely resolved, but stay in this dict because each
# turned out to hide a SECOND, independent, pre-existing compiler defect
# (same class as aircraft_wing_structural's original one: a Neumann
# condition's `fields` naming something that isn't a literal model output
# and isn't resolvable by the new `traction_map` mechanism either) that
# this pass was not scoped to fix -- see their own entries below for the
# precise, narrowed-down reason each still needs. See AUDIT_REPORT.md for
# the full per-preset writeup, including which of the fixes below are a
# real published-standard geometry (aircraft's NACA airfoil case remains
# blocked for a stated, specific missing-parameter reason -- not
# attempted-and-guessed) vs. an explicitly-flagged CHOSEN CONVENTION (see
# CHOSEN_CONVENTIONS above) for a genuinely ambiguous tag.
NOT_FIXABLE_WITHOUT_REAL_GEOMETRY: Dict[str, str] = {
    "aircraft_wing_aerodynamics": (
        "needs a real airfoil profile ('airfoil' tag). Investigated using the "
        "published NACA 4-digit thickness formula (y_t = 5t(0.2969*sqrt(x) - "
        "0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1015x^4), e.g. NACA 0012) as "
        "instructed -- but this preset's own parameters (Re, Ma, alpha_deg, "
        "chord, rho_inf, U_inf, nu_air) do not include a thickness ratio 't' "
        "or ANY airfoil designation at all ('NACA-like profile' in the "
        "docstring names a family, not a specific 4-digit code). Picking "
        "'0012' (t=0.12) would be fabricating the one number the formula "
        "actually needs, not reading it from the preset -- so this stays "
        "unfixed. A real fix needs the preset itself to expose a thickness "
        "(or full NACA code) parameter first."
    ),
    "car_brake_thermal": (
        "PARTIALLY fixed: the 'friction_surface'/'cooling_surface' tag-face "
        "ambiguity IS resolved (see CHOSEN_CONVENTIONS above -- friction = "
        "both flat faces, cooling = the outer rim) and a box fixture for it "
        "was built and verified geometrically correct. Training end-to-end "
        "still fails, though, for a SECOND, independent, pre-existing "
        "compiler defect discovered while verifying it: 'friction_surface'/"
        "'cooling_surface' declare fields=('q_heat',)/('h','T_ref'), a heat "
        "flux magnitude and a convection coefficient+reference temperature, "
        "neither a literal model field (the model only outputs 'T') nor "
        "resolvable by the traction_map mechanism (built for elasticity "
        "tractions, not heat flux/convection). The SAME q_heat/h/T_ref "
        "convention is used by cpu_heatsink_thermal, pcb_thermal, "
        "industrial_furnace_thermal, and all 3 datacenter_* presets below "
        "(confirmed by grep) -- this is a systemic, pre-existing gap across "
        "many thermal presets, well beyond this pass's authorized scope of "
        "fixing car_brake_thermal's tag ambiguity specifically. The box "
        "fixture is intentionally NOT wired into TAG_GEOMETRY_FIXTURES/"
        "build_tag_batch's dispatch table (kept as verified, tested, "
        "standalone infrastructure only), so TagConditionsUnresolved still "
        "correctly fires via solve_pde() end-to-end, same treatment as "
        "rocket_structural below."
    ),
    "car_suspension_fatigue": "wishbone-arm geometry; 'mounting_fixed'/'wheel_hub_load'/'free_edges' are not tied to specific faces of the given rectangle anywhere in the preset.",
    "cpu_heatsink_thermal": "real heatsink fin geometry ('fin_surfaces' vs 'cpu_base' vs 'insulated_sides') -- fins are not a box face.",
    "datacenter_airflow_2d": "'server_surfaces' are internal rack objects inside the airflow channel, not a face of the bounding box.",
    "datacenter_cfd_3d": "'rack_surfaces'/'crac_supply'/'return_air' are internal-object/unlocated surfaces inside the room, not box faces.",
    "datacenter_server_thermal": "'cpu_zone'/'gpu_zone'/'ram_zone' hotspot locations on the board are never given coordinates anywhere in the preset.",
    "fan_cooler_cfd": "real radial-fan blade/hub geometry ('blade_wall'/'hub_wall') -- an annulus-with-blades, not a box.",
    "industrial_furnace_thermal": "real furnace/refractory geometry; 'insulation_interface' is an internal material-layer boundary, not a domain face -- would need a real CAD/mesh of the furnace wall assembly.",
    "pcb_thermal": "'component_hotspots' locations (cpu/gpu/vrm) are given as a power dict, never as coordinates -- can't be placed on the board without inventing a layout.",
    "rocket_structural": (
        "PARTIALLY fixed: domain_bounds/geometry now correctly builds the "
        "real annulus (using the preset's own meta['inner_radius']/"
        "meta['outer_radius'], not fabricated values -- see 'annulus' shape "
        "in build_tag_batch). Training end-to-end still fails, though, for a "
        "SECOND, independent reason unrelated to geometry: 'inner_wall's "
        "Neumann condition declares fields=('p_normal',), a pressure "
        "magnitude, not a literal model field OR a traction component the "
        "new traction_map mechanism (built for aircraft_wing_structural) "
        "handles -- p_normal needs a genuinely different derivation (target "
        "normal stress n^T.sigma.n = -p, not a raw traction component along "
        "a named axis), which is out of scope for the traction_map mechanism "
        "as built. TagConditionsUnresolved still correctly fires for this "
        "preset until that second defect is fixed too."
    ),
}


def build_tag_batch(
    name: str,
    spec: ProblemSpec,
    *,
    n_col: int = 20_000,
    n_bc_per_face: int = 3_000,
    seed: int = 7,
    user_ctx: Optional[Dict[str, Any]] = None,
):
    """Build a real-geometry batch for preset ``name`` using its
    :data:`TAG_GEOMETRY_FIXTURES` entry. Raises ``KeyError`` if ``name``
    has no fixture (i.e. it is one of the ``NOT_FIXABLE_WITHOUT_REAL_
    GEOMETRY`` presets, or an untagged preset that needs no fixture at all).
    """
    from pinneapple_design.geometry.builders.analytic_domain_batch_builder import (
        sample_box_tag_batch,
        sample_cylinder_tag_batch,
        sample_axisymmetric_wall_tag_batch,
        sample_box_with_curve_tag_batch,
    )
    from ._curve_geometry import (
        circular_arc_cascade_blade_points,
        ahmed_body_polygon,
        polygon_perimeter_sample,
        polygon_contains,
    )

    fixture = TAG_GEOMETRY_FIXTURES[name]
    if fixture["shape"] == "box":
        return sample_box_tag_batch(
            spec, fixture["tag_faces"], n_col=n_col, n_bc_per_face=n_bc_per_face, seed=seed, user_ctx=user_ctx,
        )
    if fixture["shape"] == "cylinder":
        radius_param = fixture.get("radius_param")
        if radius_param is not None:
            # Read the actual radius from domain_bounds of the cross
            # section (symmetric -R..R), rather than re-deriving preset
            # kwargs -- robust to whatever radius the caller built `spec`
            # with.
            c0, _c1 = fixture["cross_coords"]
            lo, hi = spec.domain_bounds[c0]
            radius = (hi - lo) / 2.0
        else:
            radius = fixture["radius"]
        return sample_cylinder_tag_batch(
            spec,
            axis_coord=fixture["axis_coord"],
            cross_coords=fixture["cross_coords"],
            radius=radius,
            tag_faces=fixture["tag_faces"],
            n_col=n_col,
            n_bc_per_face=n_bc_per_face,
            seed=seed,
            user_ctx=user_ctx,
        )
    if fixture["shape"] == "cone_axisymmetric":
        if name == "rocket_nozzle_cfd":
            throat_r = float(spec.meta["throat_radius"])
            exit_r = float(spec.meta["exit_radius"])
            z_lo, z_hi = spec.domain_bounds[fixture["axis_coord"]]
            length = float(z_hi) - float(z_lo)
            slope = (exit_r - throat_r) / length

            def r_wall_fn(z, _z_lo=float(z_lo), _throat_r=throat_r, _slope=slope):
                return _throat_r + _slope * (np.asarray(z) - _z_lo)

            def dr_wall_fn(z, _slope=slope):
                return np.full_like(np.asarray(z, dtype=np.float64), _slope)
        else:
            raise KeyError(f"No cone_axisymmetric wiring for preset {name!r}")
        return sample_axisymmetric_wall_tag_batch(
            spec,
            axis_coord=fixture["axis_coord"],
            radial_coord=fixture["radial_coord"],
            r_wall_fn=r_wall_fn,
            dr_wall_fn=dr_wall_fn,
            tag_faces=fixture["tag_faces"],
            n_col=n_col,
            n_bc_per_face=n_bc_per_face,
            seed=seed,
            user_ctx=user_ctx,
        )
    if fixture["shape"] == "box_with_curve":
        curve_tags: Dict[str, Any] = {}
        if name == "axial_compressor_cascade_2d":
            flow_in = float(spec.meta["flow_angle_in_deg"])
            flow_out = float(spec.meta["flow_angle_out_deg"])
            chord = float(spec.meta["chord"])
            domain_x = float(spec.domain_bounds["x"][1])
            pitch = float(spec.domain_bounds["y"][1])
            zeta = 0.5 * (flow_in + flow_out) * (3.141592653589793 / 180.0)
            import math as _math
            x_le = (domain_x - chord * _math.cos(zeta)) / 2.0
            # Center the blade's LE-to-TE span (chord*sin(zeta), the
            # tangential projection of the staggered chord line) within the
            # pitch, rather than starting the LE exactly at pitch/2 -- a
            # blade with real turning is staggered enough that starting at
            # pitch/2 pushes the trailing edge outside [0, pitch] entirely
            # (confirmed directly: with this preset's own defaults the
            # unrotated placement put the TE at y=0.097 > pitch=0.08).
            y_le = pitch / 2.0 - chord * _math.sin(zeta) / 2.0

            def _blade_fn(n, rng, _fi=flow_in, _fo=flow_out, _c=chord, _xle=x_le, _yle=y_le):
                return circular_arc_cascade_blade_points(
                    n, rng, flow_angle_in_deg=_fi, flow_angle_out_deg=_fo, chord=_c, x_le=_xle, y_le=_yle,
                )

            curve_tags["blade"] = _blade_fn
            inside_body_fn = None  # zero-thickness cambered-plate blade
        elif name == "car_external_aero":
            # car_length/car_height are not stored in meta, but the preset
            # itself derives domain_bounds directly from them
            # (x_max=5*car_length, y_max=5*car_height) -- read them back
            # out exactly, rather than re-deriving/guessing.
            car_length = float(spec.domain_bounds["x"][1]) / 5.0
            car_height = float(spec.domain_bounds["y"][1]) / 5.0
            poly = ahmed_body_polygon(car_length, car_height)

            def _car_body_fn(n, rng, _poly=poly):
                return polygon_perimeter_sample(_poly, n, rng)

            curve_tags["car_body"] = _car_body_fn

            def inside_body_fn(X2, _poly=poly):
                return polygon_contains(_poly, X2)
        else:
            raise KeyError(f"No box_with_curve wiring for preset {name!r}")
        return sample_box_with_curve_tag_batch(
            spec,
            shape_coords=fixture["shape_coords"],
            box_tag_faces=fixture.get("box_tag_faces") or {},
            curve_tags=curve_tags,
            inside_body_fn=inside_body_fn,
            n_col=n_col,
            n_bc_per_face=n_bc_per_face,
            n_bc_curve=n_bc_per_face,
            seed=seed,
            user_ctx=user_ctx,
        )
    raise ValueError(f"Unknown fixture shape {fixture['shape']!r} for preset {name!r}")


def solve_pde_kwargs_from_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the ``x_col`` key (``solve_pde`` samples its own collocation
    points from ``spec.domain_bounds`` and doesn't accept one) and convert
    numpy arrays to torch tensors, ready to splat into
    ``solve_pde(spec, model, **solve_pde_kwargs_from_batch(batch))``."""
    import numpy as np
    import torch

    out = {}
    for k, v in batch.items():
        if k == "x_col":
            continue
        if k == "ctx":
            out[k] = v
            continue
        out[k] = torch.as_tensor(np.asarray(v), dtype=torch.bool if k.startswith("mask_") else torch.float32)
    return out
