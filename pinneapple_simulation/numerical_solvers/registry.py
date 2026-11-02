"""Solver registry.

Pinneapple uses "solvers" in two distinct ways:
  1) Numerical PDE solvers (FDM/FEM/FVM/LBM/Spectral)
  2) Signal/feature solvers for time-series preprocessing (FFT/HHT/Wavelet/SSA/...)
  3) Particle methods (SPH variants)

This registry gives the Arena and training pipelines a unified catalog.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from .base import SolverBase, entry_point


@dataclass
class SolverSpec:
    name: str
    family: str
    cls: Type[SolverBase]
    description: str = ""
    tags: List[str] | None = None


class SolverRegistry:
    _REG: Dict[str, SolverSpec] = {}

    @classmethod
    def register(
        cls,
        *,
        name: str,
        family: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        key = str(name).lower().strip()
        if not key:
            raise ValueError("Solver name must be non-empty")

        def deco(solver_cls: Type[SolverBase]):
            if key in cls._REG and cls._REG[key].cls is not solver_cls:
                raise KeyError(f"Solver '{key}' already registered")
            cls._REG[key] = SolverSpec(name=key, family=str(family), cls=solver_cls, description=description, tags=tags or [])
            return solver_cls

        return deco

    @classmethod
    def spec(cls, name: str) -> SolverSpec:
        key = str(name).lower().strip()
        if key not in cls._REG:
            raise KeyError(f"Unknown solver '{name}'. Available: {cls.list()}")
        return cls._REG[key]

    @classmethod
    def list(cls, family: Optional[str] = None) -> List[str]:
        if family is None:
            return sorted(cls._REG.keys())
        fam = str(family)
        return sorted([k for k, v in cls._REG.items() if v.family == fam])

    @classmethod
    def build(cls, name: str, **kwargs) -> SolverBase:
        """Construct a solver. Every solver has ``solve(...)``: see ``describe(name)`` for what it takes."""
        spec = cls.spec(name)
        try:
            inspect.signature(spec.cls).bind(**kwargs)
        except TypeError as e:
            raise TypeError(f"SolverRegistry.build({name!r}): {e}. The constructor is "
                            f"{spec.cls.__name__}{inspect.signature(spec.cls)}") from None
        return spec.cls(**kwargs)

    @classmethod
    def describe(cls, name: str) -> Dict[str, Any]:
        """How to run a registered solver: its class, constructor, entry point and the call it expects (#957)."""
        spec = cls.spec(name)
        ep = entry_point(spec.cls)
        method = getattr(spec.cls, ep)
        first_arg = next((p for p in inspect.signature(method).parameters if p != "self"), "")
        call_sig = str(inspect.signature(method)).replace("(self, ", "(").replace("(self)", "()")
        if first_arg in ("spec", "problem_spec") and ep != "solve":
            example = f"SolverCatalog().build({spec.name!r}).solve(problem_spec)"
        else:
            example = f"SolverCatalog().build({spec.name!r}).solve{call_sig}"
        doc = inspect.getdoc(method) or inspect.getdoc(spec.cls) or ""
        return {
            "name": spec.name, "family": spec.family,
            "class": f"{spec.cls.__module__}.{spec.cls.__qualname__}",
            "constructor": f"{spec.cls.__name__}{inspect.signature(spec.cls)}",
            "entry_point": ep, "call": f"{ep}{call_sig}",
            "takes_problem_spec": first_arg in ("spec", "problem_spec"),
            "description": spec.description or (doc.splitlines()[0] if doc else ""),
            "example": example,
        }


# --------- register built-ins
# Keep imports at bottom to avoid import cycles.

_ALL_SOLVER_MODULES = [
    "fft", "hilbert_huang", "eemd", "ceemdan", "vmd", "wavelet", "sst",
    "ssa", "stl",
    "fdm", "fem", "fvm", "lbm", "spectral", "jax_cfd_solver",
    "sph", "isph", "dfsph",
    "meshfree",
    "xtfc_ivp", "eddy_current_fdm", "immersed_boundary_fdm", "beam_bvp_fdm",
    "elasticity3d_fdm", "nonlinear_beam_fem",
    "xtfc_pde", "xtfc_subdomain", "discrete_time_pinn",
    "nbody_rebound",
]


def register_all() -> Dict[str, str]:
    """Import solver modules so their @register decorators execute.

    Each module is imported independently: one solver needing an
    uninstalled optional dependency (e.g. `wavelet` needs `pywt`) used to
    raise straight out of this function and silently skip registering
    every solver listed *after* it too (a plain top-to-bottom list of
    `from . import x` statements aborts on the first failure) -- found via
    `SolverRegistry.list()` returning only 8 of the ~26 solvers that
    exist in this package. Returns a dict of {module_name: error_message}
    for any that failed to import, so callers can inspect what's missing
    instead of that failure being invisible.
    """
    import importlib

    failures: Dict[str, str] = {}
    for mod_name in _ALL_SOLVER_MODULES:
        try:
            importlib.import_module(f".{mod_name}", __package__)
        except Exception as e:
            failures[mod_name] = str(e)
    return failures


@dataclass
class SolverCatalog:
    """
    Convenience wrapper (dict-like catalog) around SolverRegistry.

    Useful for code that prefers:
        cat = SolverCatalog(); solver = cat.build("eemd", ...)

    Auto-calls register_all() once so all solvers are available.
    """

    registry: SolverRegistry = None

    def __post_init__(self):
        register_all()
        self.registry = SolverRegistry()

    def list(self):
        return self.registry.list()

    def get(self, name: str):
        return self.registry.spec(name)

    def build(self, name: str, **kwargs):
        return self.registry.build(name, **kwargs)

    def describe(self, name: str) -> Dict[str, Any]:
        return self.registry.describe(name)
