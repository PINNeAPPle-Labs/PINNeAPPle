"""Adapters: one per application. Importing this package registers all of them."""
from . import checks, freecad, openfoam, paraview  # noqa: F401  (registration)
from .base import (
                   ActionFailed,
                   ActionResult,
                   Adapter,
                   StepContext,
                   get_adapter,
                   known_actions,
                   list_adapters,
                   register,
)

__all__ = ["Adapter", "ActionResult", "ActionFailed", "StepContext", "register", "get_adapter", "list_adapters",
           "known_actions"]
