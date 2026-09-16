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
"""
from __future__ import annotations

from typing import Any, Dict, Optional

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

}


# Preset -> one-line reason it does NOT get a fixture (documentation only;
# behavior is unaffected -- TagConditionsUnresolved fires regardless).
NOT_FIXABLE_WITHOUT_REAL_GEOMETRY: Dict[str, str] = {
    "aircraft_wing_aerodynamics": "needs a real NACA-like airfoil profile ('airfoil' tag) -- not deducible from farfield-box params alone.",
    "aircraft_wing_structural": (
        "the geometry/face mapping IS unambiguous (x=0..span is root..tip by "
        "construction, 'free_surface' is the remaining y faces) -- but a "
        "SEPARATE, pre-existing defect blocks it regardless of geometry: "
        "its 'tip_load'/'free_surface' Neumann conditions use fields=('ty',)/"
        "('tx','ty') (traction labels), while compile.py's generic Neumann "
        "handling does `fvals[f] for f in cond.fields` against the model's "
        "own PDE fields (ux, uy) -- 'ty'/'tx' KeyError immediately, with "
        "correct geometry supplied and confirmed by hand this session. "
        "Fixing this needs a real traction-from-stress Neumann evaluation "
        "in the elasticity compiler branch, out of scope for a geometry-only fix."
    ),
    "car_external_aero": "needs a real car-body silhouette ('car_body' tag) -- a bluff body shape, not a box face.",
    "car_suspension_fatigue": "wishbone-arm geometry; 'mounting_fixed'/'wheel_hub_load'/'free_edges' are not tied to specific faces of the given rectangle anywhere in the preset.",
    "cpu_heatsink_thermal": "real heatsink fin geometry ('fin_surfaces' vs 'cpu_base' vs 'insulated_sides') -- fins are not a box face.",
    "datacenter_airflow_2d": "'server_surfaces' are internal rack objects inside the airflow channel, not a face of the bounding box.",
    "datacenter_cfd_3d": "'rack_surfaces'/'crac_supply'/'return_air' are internal-object/unlocated surfaces inside the room, not box faces.",
    "datacenter_server_thermal": "'cpu_zone'/'gpu_zone'/'ram_zone' hotspot locations on the board are never given coordinates anywhere in the preset.",
    "fan_cooler_cfd": "real radial-fan blade/hub geometry ('blade_wall'/'hub_wall') -- an annulus-with-blades, not a box.",
    "industrial_furnace_thermal": "real furnace/refractory geometry; 'insulation_interface' is an internal material-layer boundary, not a domain face -- would need a real CAD/mesh of the furnace wall assembly.",
    "linear_elasticity_3d": "'fixed' and 'load' are two different, unlocated tags on a unit cube -- no docstring/comment states which face is which (unlike material_fracture_2d's explicit 'fixed bottom, prescribed top' comment).",
    "pcb_thermal": "'component_hotspots' locations (cpu/gpu/vrm) are given as a power dict, never as coordinates -- can't be placed on the board without inventing a layout.",
    "plane_stress_2d": "'fixed' and 'load' are two different, unlocated tags on a unit square -- same ambiguity as linear_elasticity_3d.",
    "rocket_nozzle_cfd": "real convergent-divergent nozzle contour (the 'wall' tag is a curved profile, not a straight cylindrical or planar face) -- needs an actual nozzle contour function (e.g. Rao/conical), not given.",
    "rocket_structural": "domain_bounds is a solid square [-outer,outer]^2, but the physical part is an ANNULUS (inner_radius to outer_radius) -- the preset's own stated domain_bounds don't even encode the hole, so there is nothing consistent to sample without redefining the domain.",
    "axial_compressor_cascade_2d": "real compressor blade cascade profile ('blade' tag) -- a curved airfoil shape, not deducible from domain_bounds.",
    "car_brake_thermal": "'friction_surface' vs 'cooling_surface' assignment to the disc's flat faces (z=0/z=thickness) vs outer rim (r=disc_radius) is not stated anywhere in the preset -- multiple physically-plausible assignments exist (both flat faces are friction surfaces in a real disc; outer rim OR a flat face could be 'cooling') and nothing disambiguates which is meant.",
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
