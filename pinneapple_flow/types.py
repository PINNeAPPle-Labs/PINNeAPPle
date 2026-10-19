"""Port types of the flow graph and the rule that says which output may feed which input.

Every node port has one of the types below. An edge ``source.port -> target.port`` is valid when the types are equal,
when the target accepts ``any``, or when the pair is in :data:`CONVERSIONS` (a value of the source type carries what
the target needs, e.g. a ``problem`` holds its ``geometry``).

The values that travel along edges are JSON: small values inline, arrays and weights as artifact files written by the
node (``{"$artifact": "solution.npz"}``, resolved against the node's run folder).
"""
from __future__ import annotations

PORT_TYPES: dict[str, str] = {
    "geometry": "domain and its dimension (interval, rectangle, box, disk, sphere), bounds, time interval",
    "mesh": "discretisation of a geometry (grid spacing or node count per axis)",
    "equations": "validated equation system: PDEs, boundary and initial conditions, coefficient values",
    "problem": "geometry + equations, ready to solve (maps onto pp.PhysicalProblem when a preset matches)",
    "solution": "a field sampled on points or a grid (coordinates + values, .npz artifact)",
    "model": "a trained model (weights + configuration artifact) that can predict",
    "dataset": "arrays or a table for training or calibration (.npz / .csv artifact)",
    "series": "a time series (time + one or more channels)",
    "metrics": "measured numbers (errors, residuals, timings), with how they were measured",
    "params": "parameter values (estimates, best hyperparameters, coefficients)",
    "architecture": "layer graph of a network with parameter counts",
    "figure": "an image artifact (PNG)",
    "scene": "a 3-D scene payload for the three.js viewer",
    "twin": "state of a digital twin (estimates, residuals, alarms)",
    "report": "a report artifact (HTML / Markdown / JSON)",
    "vehicle": "a ground vehicle: preset and its parameters (mass, wheels) from pinneapple_simulation.vehicle",
    "soil": "a terramechanics soil: preset and its Bekker / Wong-Reece parameters",
    "site": "a terrain scenario (craters, rocks, slope) for a vehicle run",
    "vehicle_run": "a simulated vehicle traverse (time histories per wheel and the rutted terrain artifact)",
    "any": "accepts every type",
}

#: (source type, target type) pairs that connect although the names differ
CONVERSIONS: frozenset[tuple[str, str]] = frozenset({
    ("problem", "geometry"),        # a problem carries its geometry
    ("problem", "equations"),       # ... and its equations
    ("mesh", "geometry"),           # a mesh knows its geometry
    ("solution", "dataset"),        # a solved field is training / calibration data
    ("series", "dataset"),
    ("dataset", "series"),
    ("twin", "series"),             # the twin's logged stream
})


def compatible(source: str, target: str) -> bool:
    """True when an output of type ``source`` may feed an input of type ``target``."""
    if source not in PORT_TYPES or target not in PORT_TYPES:
        return False
    return source == target or target == "any" or source == "any" or (source, target) in CONVERSIONS


def register_port_type(name: str, description: str, *, converts_to=(), converts_from=()) -> None:
    """Add a port type (for another package's node types, e.g. a CFD case). ``converts_to`` / ``converts_from``:
    existing types it may feed / be fed from although the names differ. Re-registering the same description is a
    no-op; a different one for an existing name is refused, so documents stay portable."""
    global CONVERSIONS
    if not name or not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise ValueError(f"port type {name!r}: letters, digits and _ only, starting with a letter")
    if name in PORT_TYPES and PORT_TYPES[name] != description:
        raise ValueError(f"port type {name!r} is already registered with another meaning")
    unknown = [t for t in (*converts_to, *converts_from) if t not in PORT_TYPES and t != name]
    if unknown:
        raise ValueError(f"unknown port types {unknown}")
    PORT_TYPES[name] = description
    CONVERSIONS = frozenset(CONVERSIONS | {(name, t) for t in converts_to} | {(t, name) for t in converts_from})
