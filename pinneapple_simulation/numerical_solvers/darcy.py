"""2-D Darcy flow from a ``ProblemSpec``: ``-div(a grad u) = f`` on the unit square, ``u = 0`` on the boundary.

The solve itself is :func:`pinneapple_simulation.numerical_solvers.operator_benchmarks.darcy_2d` -- cell-centred
finite volumes, harmonic-mean face coefficients (exact flux continuity across a jump aligned with a face), sparse
direct solve, second order for smooth coefficients, checked against a manufactured solution in
``tests/test_operator_benchmarks.py``. This module is only its registry face, so a ``ProblemSpec`` with
``pde.kind = "darcy"`` reaches it through ``solve_from_spec``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .operator_benchmarks import darcy_2d
from .registry import SolverRegistry

__all__ = ["Darcy2DFVMSolver"]


@SolverRegistry.register(
    name="darcy2d_fvm",
    family="pde",
    description="2-D Darcy flow -div(a grad u) = f on the unit square with u = 0 on the boundary, cell-centred "
                "finite volumes with harmonic face coefficients and a sparse direct solve",
    tags=["fvm", "darcy", "porous", "2d", "elliptic"],
)
class Darcy2DFVMSolver(SolverBase):
    """``solve_from_spec`` for :func:`darcy_2d`.

    The spec's parameters (``spec.pde.params``): ``n`` cells per axis (default 64), ``permeability`` a constant
    or a callable ``a(x, y)`` of the cell centres (default 1.0), and ``source`` a constant or a callable
    ``f(x, y)`` (default 1.0). The boundary is held at zero, and the domain is the unit square, because that is
    what the solver documents; a spec that asks for another one is refused by name.
    """

    def solve_from_spec(self, spec) -> SolverOutput:
        kind = str(getattr(spec.pde, "kind", "")).lower().replace("-", "_").replace(" ", "_")
        if "darcy" not in kind:
            raise ValueError(f"darcy2d_fvm solves Darcy flow, and the spec's kind is '{kind}'.")
        if int(getattr(spec, "dim", 2)) != 2:
            raise ValueError(f"darcy2d_fvm is a 2-D solver; the spec is {getattr(spec, 'dim', '?')}-D.")
        bounds = dict(getattr(spec, "domain_bounds", {}) or {})
        if bounds and (tuple(round(v, 9) for v in bounds.get("x", (0.0, 1.0))) != (0.0, 1.0)
                       or tuple(round(v, 9) for v in bounds.get("y", (0.0, 1.0))) != (0.0, 1.0)):
            raise ValueError("darcy2d_fvm solves on the unit square (the domain darcy_2d documents); this spec's "
                             f"domain_bounds are {bounds}.")
        params = dict(getattr(spec.pde, "params", {}) or {})
        n = int(params.get("n", 64))
        cells = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(cells, cells, indexing="ij")

        permeability = params.get("permeability", params.get("a", 1.0))
        a = (np.asarray(permeability(X, Y), dtype=np.float64) if callable(permeability)
             else np.full((n, n), float(permeability)))
        source = params.get("source", params.get("f", 1.0))
        f: Any = (np.asarray(source(X, Y), dtype=np.float64) if callable(source) else float(source))

        conditions = tuple(getattr(spec, "conditions", ()) or ())
        if conditions:
            from . import _bc
            named = _bc.dirichlet_conditions(spec)
            points = torch.from_numpy(np.column_stack([X.ravel()[:: n], Y.ravel()[: n]]).astype(np.float32))
            for cond in named:
                wall = _bc.values(cond, points)
                if float(np.max(np.abs(wall.detach().cpu().numpy()))) > 1e-12:
                    raise ValueError(
                        f"darcy2d_fvm holds the whole boundary at zero, and the dirichlet condition "
                        f"'{getattr(cond, 'name', '')}' asks for a non-zero value there.")
            others = [c for c in conditions if getattr(c, "kind", None) != "dirichlet"]
            if others:
                raise ValueError("darcy2d_fvm has u = 0 on the boundary only; the spec asks for "
                                 f"{sorted({c.kind for c in others})}.")

        u = darcy_2d(a, f)
        return SolverOutput(
            result=torch.from_numpy(np.asarray(u, dtype=np.float32)),
            losses={"max_u": torch.tensor(float(np.max(u)))},
            extras={"method": "darcy2d_fvm", "n": n, "u_min": float(np.min(u)), "u_max": float(np.max(u)),
                    "a_range": (float(np.min(a)), float(np.max(a))),
                    "cell_centres": {"x": cells, "y": cells}},
        )
