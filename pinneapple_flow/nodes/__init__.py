"""Built-in node types (imported by :data:`pinneapple_flow.registry.REGISTRY` on first use)."""
from ..libcalls import auto_specs, library_specs
from ..registry import REGISTRY
from . import (  # noqa: F401
    analysis,
    data,
    methods,
    problem,
    solvers,
    surrogates,
    systems,
    twin_nodes,
    vehicle,
    viz,
)

for _spec in library_specs() + auto_specs():
    REGISTRY.register(_spec)
