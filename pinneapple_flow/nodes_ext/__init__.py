"""Node types that reproduce the ``examples/*`` studies as flows, one module per domain, registered through the public
API (:func:`pinneapple_flow.registry.node`, :func:`pinneapple_flow.types.register_port_type`). Each node wraps library
calls; the flows that use them are under ``pinneapple_flow/flows/examples/<domain>/`` and the inventory of examples
is :mod:`pinneapple_flow.examples` (``docs/flow_studio/EXAMPLES_AS_FLOWS.md``)."""
from ..examples import register_hook  # noqa: E402
from . import (  # noqa: F401
    aero,
    astro,
    cfd,
    core_ts,
    design,
    electromagnetics,
    em_prius,
    examples,
    flight,
    forecasting,
    geomechanics,
    pinn_examples,
    problemdesign,
    render,
    security,
    solvers,
    structures,
    use_cases,
    validation,
    viz,
)

register_hook()                         # example-flow runs become Lab experiments (pinneapple_flow.examples)
