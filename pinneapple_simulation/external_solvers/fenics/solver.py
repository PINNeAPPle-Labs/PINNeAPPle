"""FEniCS integration — workflow layer on top of ``numerical_solvers.fenics_bridge``.

The low-level bridge (``numerical_solvers.fenics_bridge.FEnicsBridge``) does the FEM assembly and solve. This module
adds:

  FEniCSConfig     — structured configuration dataclass
  solve_and_package — one-call solve + UPD packaging
  dof_to_upd       — convert raw DOF vector + mesh coords to PhysicalSample
  FEniCSWorkflow   — convenience wrapper for repeated solves (e.g. parametric sweeps)

The bridge takes a *problem spec* (``kind``, ``domain_bounds``, ``conditions``, ``parameters``); :class:`FEniCSConfig`
says the same thing in this module's vocabulary, so :func:`solve_and_package` translates one into the other. That
translation is the whole of the wrapper's work: a wrapper that names a class or a keyword that does not exist fails on
the first call, and nobody notices while the import stays lazy.

For the one-interface-every-solver route use the solver contract instead
(``pinneapple_simulation.external_solvers.get_solver("fenicsx")``), which wraps the same bridge with
``check_environment()``, a status with the log evidence and results in the same schema as the other backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FEniCSConfig:
    """Configuration for a FEniCS solve.

    Parameters
    ----------
    pde : PDE identifier understood by FEniCSBridge
        e.g. "heat_equation_steady", "linear_elasticity_plane_stress"
    domain : domain specification dict accepted by FEniCSBridge
        e.g. {"type": "rectangle", "x": [0, 1], "y": [0, 1], "nx": 32, "ny": 32}
    bcs : list of boundary condition specs accepted by FEniCSBridge
    params : PDE parameter values (conductivity k, Young's modulus E, etc.)
    solver_opts : additional solver options forwarded to FEniCSBridge
    """
    pde: str
    domain: dict[str, Any]
    bcs: list[dict[str, Any]]
    params: dict[str, float] = field(default_factory=dict)
    solver_opts: dict[str, Any] = field(default_factory=dict)


class FEniCSUnavailable(ImportError):
    """FEniCS (DOLFINx or legacy FEniCS) is not importable, so there is nothing to solve with."""


def problem_spec(config: FEniCSConfig) -> Any:
    """The bridge's problem spec (``kind``, ``domain_bounds``, ``conditions``, ``parameters``) for a config.

    ``domain`` is the rectangle ``{"x": [x0, x1], "y": [y0, y1], "nx": n, "ny": n}`` the bridge's mesher builds;
    ``bcs`` become boundary objects with ``type``, ``side`` and ``value``; ``params`` is passed through as the PDE
    parameters (``k`` and ``f`` for the heat equation, ``E``/``nu``/``f`` for elasticity).
    """
    from types import SimpleNamespace

    domain = dict(config.domain or {})
    x0, x1 = (float(v) for v in domain.get("x", (0.0, 1.0)))
    y0, y1 = (float(v) for v in domain.get("y", (0.0, 1.0)))
    conditions = {}
    for i, bc in enumerate(config.bcs or ()):
        bc = dict(bc)
        conditions[bc.pop("name", f"bc{i}")] = SimpleNamespace(type=bc.get("type", "dirichlet"),
                                                              side=bc.get("side"),
                                                              value=bc.get("value", 0.0),
                                                              **{k: v for k, v in bc.items()
                                                                 if k not in ("type", "side", "value")})
    return SimpleNamespace(kind=config.pde, domain_bounds=(x0, y0, x1, y1), conditions=conditions,
                           parameters=dict(config.params or {}),
                           mesh_nx=int(domain.get("nx", 32)), mesh_ny=int(domain.get("ny", 32)))


def solve_and_package(config: FEniCSConfig):
    """Run a FEniCS solve and return a UPD PhysicalSample.

    Returns
    -------
    PhysicalSample with ``state`` = the field vectors, ``domain`` = the coordinates of their degrees of freedom (so a
    value and its point stay together), ``provenance`` = the backend, its version and the PDE.

    Raises ``FEniCSUnavailable`` when neither DOLFINx nor legacy FEniCS is installed -- the fix is in the message.
    """
    try:
        from pinneapple_simulation.numerical_solvers import fenics_bridge
    except ImportError as exc:                        # pragma: no cover - the module itself imports
        raise FEniCSUnavailable(f"the FEniCS bridge could not be imported: {exc}") from None
    if not fenics_bridge._FENICS_AVAILABLE:           # noqa: SLF001 - the bridge's own probe, not a second one
        raise FEniCSUnavailable(
            "pinneapple_simulation.numerical_solvers.fenics_bridge needs FEniCS: install DOLFINx "
            "(conda install -c conda-forge fenics-dolfinx=0.9 mpich) or legacy FEniCS, then run again."
        )
    spec = problem_spec(config)
    bridge = fenics_bridge.FEnicsBridge(mesh_nx=spec.mesh_nx, mesh_ny=spec.mesh_ny, **config.solver_opts)
    output = bridge.forward(spec)
    if "error" in (output.extras or {}):
        raise RuntimeError(f"the FEniCS solve failed: {output.extras['error']}")
    return _output_to_upd(output, config)


def _output_to_upd(solver_output: Any, config: FEniCSConfig):
    """Convert a FEniCSBridge SolverOutput to a UPD PhysicalSample (fields from ``extras``, points from the DOFs)."""
    import torch

    from pinneapple_data.physical_sample import PhysicalSample

    extras = dict(getattr(solver_output, "extras", None) or {})
    raw = dict(extras.get("fields") or {})
    if not raw:                                       # the bridge that only had ``result``: one flattened vector
        result = np.asarray(getattr(solver_output, "result", []), dtype=float).ravel()
        raw = {"u": result} if result.size else {}

    fields: dict[str, torch.Tensor] = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in raw.items()}
    coords: dict[str, np.ndarray] = {}
    coordinates = extras.get("dof_coordinates")
    xy = np.asarray(coordinates) if coordinates is not None else np.empty((0, 0))
    if xy.ndim == 2 and xy.size and len(fields.get("u", ())) == xy.shape[0]:
        for i, name in enumerate(["x", "y", "z"][: xy.shape[1]]):
            coords[name] = xy[:, i]

    return PhysicalSample(
        state=fields,
        domain={"type": "points" if coords else "grid", "coords": coords},
        provenance={"version": "0.1", "source": "fenics", "pde": config.pde,
                    "backend": extras.get("backend"), "dofs": extras.get("dofs")},
        schema={"units": {}},
    )


def dof_to_upd(
    dof_vector: np.ndarray,
    mesh_coords: np.ndarray,
    field_name: str = "u",
):
    """Package a raw FEniCS DOF vector and mesh coordinates as a UPD PhysicalSample.

    Parameters
    ----------
    dof_vector : (N,) or (N, F) array of DOF values
    mesh_coords : (N, d) array of node coordinates
    field_name : name for the field in the PhysicalSample

    Returns
    -------
    PhysicalSample
    """
    import torch

    from pinneapple_data.physical_sample import PhysicalSample

    coord_names = ["x", "y", "z"]
    coords = {
        coord_names[i]: mesh_coords[:, i]
        for i in range(min(mesh_coords.shape[1], 3))
    }

    return PhysicalSample(
        state={field_name: torch.as_tensor(np.asarray(dof_vector), dtype=torch.float32)},
        domain={"type": "grid", "coords": coords},
        provenance={"version": "0.1", "source": "fenics"},
        schema={"units": {}},
    )


class FEniCSWorkflow:
    """Convenience wrapper for repeated FEniCS solves over a parameter space.

    Useful for generating training datasets for surrogate models or PINNs.

    Examples
    --------
    >>> base_cfg = FEniCSConfig(pde="heat_equation_steady", domain=..., bcs=...)
    >>> wf = FEniCSWorkflow(base_cfg)
    >>> samples = wf.sweep({"k": [0.5, 1.0, 2.0, 5.0]})
    """

    def __init__(self, base_config: FEniCSConfig) -> None:
        self.base_config = base_config

    def solve(self, param_override: dict[str, float] | None = None):
        """Run a single solve with optional parameter override."""
        cfg = self.base_config
        if param_override:
            cfg = FEniCSConfig(
                pde=cfg.pde,
                domain=cfg.domain,
                bcs=cfg.bcs,
                params={**cfg.params, **param_override},
                solver_opts=cfg.solver_opts,
            )
        return solve_and_package(cfg)

    def sweep(self, param_grid: dict[str, list[float]]) -> list[Any]:
        """Solve for each combination in a parameter grid.

        Parameters
        ----------
        param_grid : dict mapping parameter name → list of values.
            All combinations are evaluated (Cartesian product).

        Returns
        -------
        list of PhysicalSample, one per parameter combination
        """
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []
        for combo in product(*values):
            overrides = dict(zip(keys, combo))
            results.append(self.solve(overrides))
        return results
