"""Scientific validators and their registry (stage E).

``available()`` lists what is registered; ``run(name, ctx)`` executes one.
The built-ins are registered as a side effect of importing
:mod:`pinneapple_llm.qa.validators`, so a caller that has imported the
package can enumerate them without knowing the module layout.
"""
from __future__ import annotations

from .base import (  # noqa: F401
    Mesh, SensorReading, ValidationContext, ValidatorResult,
    available, blocked, get, not_applicable, register, run,
)

__all__ = [
    "Mesh", "SensorReading", "ValidationContext", "ValidatorResult",
    "available", "blocked", "get", "not_applicable", "register", "run",
]
