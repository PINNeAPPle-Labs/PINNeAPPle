"""Base SolverOutput and SolverBase for numerical solvers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SolverOutput:
    """
    Standard output for solvers / numerical methods.
    result:
      - main output tensor (spectra, modes, field solution, etc.)
    extras:
      - metadata (freqs, imfs, diagnostics, residuals)
    """
    result: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class SolverBase(nn.Module):
    """Base class for numerical solvers."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mean((a - b) ** 2)

    def solve(self, *args, **kwargs):
        """Uniform entry point shared by every registered solver (#957).

        A ``ProblemSpec`` as first argument goes to ``solve_from_spec`` when the solver has one; anything else
        goes to ``forward`` (the solver's own signature: ``SolverRegistry.describe(name)`` shows it). Solvers
        with their own ``solve`` override this.
        """
        if args and _is_problem_spec(args[0]) and hasattr(self, "solve_from_spec"):
            return self.solve_from_spec(*args, **kwargs)
        return self(*args, **kwargs)


def _is_problem_spec(obj) -> bool:
    return all(hasattr(obj, a) for a in ("pde", "conditions", "fields", "coords"))


def entry_point(cls) -> str:
    """Name of the method that does a solver's work: its own ``solve``, else ``solve_from_spec``, else ``forward``."""
    if any("solve" in k.__dict__ for k in cls.__mro__ if k is not SolverBase):
        return "solve"
    return "solve_from_spec" if hasattr(cls, "solve_from_spec") else "forward"
