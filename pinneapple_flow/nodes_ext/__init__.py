"""Node types that reproduce the ``examples/*`` studies as flows, one module per domain, registered through the public
API (:func:`pinneapple_flow.registry.node`, :func:`pinneapple_flow.types.register_port_type`). Each node wraps library
calls; the flows that use them are under ``pinneapple_flow/flows/examples/<domain>/`` and the inventory of examples
is :mod:`pinneapple_flow.examples` (``docs/flow_studio/EXAMPLES_AS_FLOWS.md``)."""
from ..examples import register_hook  # noqa: E402
from . import (  # noqa: F401
    aero,
    ahmed_design,
    aircraft_surrogate,
    architectures,
    arena_pipelines,
    astro,
    autolab_caps,
    benchmarks,
    bio_workflow,
    bricks,
    cfd,
    chip_design,
    control_eng,
    core_ts,
    design,
    design_optimization,
    digital_twin,
    drone_design,
    electromagnetics,
    em_antenna,
    em_fem3d,
    em_prius,
    equation_discovery,
    examples,
    flight,
    football,
    forecasting,
    geomechanics,
    inverse_workflow,
    lbm,
    learned_sim,
    lithography,
    model_benchmark,
    model_use,
    multiphysics,
    non_hermitian,
    pde_env,
    pinn_examples,
    pinn_solver,
    potential,
    problemdesign,
    reactors,
    reduced_order,
    render,
    rover_design,
    satellite_design,
    security,
    solver_registry,
    solvers,
    structures,
    surrogate_inverse,
    surrogate_modeling,
    terramechanics,
    use_cases,
    validation,
    validation_gallery,
    valve_jt,
    veriphysics_grade,
    viz,
    vortex,
    wing_design,
    worldmodel,
)

register_hook()                         # example-flow runs become Lab experiments (pinneapple_flow.examples)


def _place_in_palette() -> None:
    """Every example-domain node in the palette section of its category (pinneapple_flow.palette.place)."""
    from .. import palette
    from ..registry import REGISTRY
    for spec in REGISTRY.specs():
        if spec.fn.__module__.startswith(__name__ + "."):
            palette.place(spec.type, palette.section_of(spec))


_place_in_palette()
