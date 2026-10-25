"""Node types that reproduce the ``examples/*`` studies as flows, one module per domain, registered through the public
API (:func:`pinneapple_flow.registry.node`, :func:`pinneapple_flow.types.register_port_type`). Each node wraps library
calls; the flows that use them are under ``pinneapple_flow/flows/examples/<domain>/`` and the inventory of examples
is :mod:`pinneapple_flow.examples` (``docs/flow_studio/EXAMPLES_AS_FLOWS.md``)."""
from ..examples import register_hook  # noqa: E402
from . import (  # noqa: F401
    aero,
    architectures,
    arena_pipelines,
    astro,
    autolab_caps,
    benchmarks,
    bricks,
    cfd,
    core_ts,
    design,
    electromagnetics,
    em_antenna,
    em_prius,
    examples,
    flight,
    football,
    forecasting,
    geomechanics,
    lbm,
    learned_sim,
    non_hermitian,
    pde_env,
    pinn_examples,
    pinn_solver,
    problemdesign,
    render,
    security,
    solver_registry,
    solvers,
    structures,
    surrogate_inverse,
    use_cases,
    validation,
    validation_gallery,
    viz,
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
