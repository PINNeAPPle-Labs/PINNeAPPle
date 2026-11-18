"""The case description PICT needs: a wall-refined grid, a lid-driven cavity, and its PISOtorch domain.

Everything here except :func:`build_cavity_domain` runs without PICT installed: the grid is built with numpy and
checked (spacing monotone, geometric ratio, corners exact), and the case validates itself before any CUDA call.
"""
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PICTAvailability", "pict_availability", "pict_available",
    "wall_refined_grid", "CavityCase", "build_cavity_domain",
]

# PICT installs two importable names: the compiled core and the PISO loop written around it.
_PICT_MODULES = ("PISOtorch", "PISOtorch_simulation")


@dataclass
class PICTAvailability:
    """What is present and what is missing, one line each. ``available`` is the AND of all of it."""

    has_core: bool
    has_simulation: bool
    has_cuda: bool
    missing: tuple[str, ...]
    device: str = ""
    hint: str = ""

    @property
    def available(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "has_core": self.has_core, "has_simulation": self.has_simulation,
                "has_cuda": self.has_cuda, "missing": list(self.missing), "device": self.device,
                "hint": self.hint}


def pict_availability() -> PICTAvailability:
    """Probe PICT without importing it far enough to fail: two modules and a CUDA device are needed."""
    missing: list[str] = []
    spec = {name: importlib.util.find_spec(name) is not None for name in _PICT_MODULES}
    for name, present in spec.items():
        if not present:
            missing.append(name)
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        device = "cuda" if has_cuda else "cpu"
        torch_version = torch.__version__
    except Exception:                                  # pragma: no cover - torch is a core dependency
        has_cuda, device, torch_version = False, "", "absent"
    if not has_cuda:
        missing.append("a CUDA device (PICT's kernels are CUDA only; torch reports cpu)")
    hint = ""
    if missing:
        hint = ("install PICT from source: git clone https://github.com/tum-pbs/PICT.git, then "
                "cd extensions && python setup.py install; it needs Linux, an NVIDIA GPU and CUDA")
    return PICTAvailability(has_core=spec[_PICT_MODULES[0]], has_simulation=spec[_PICT_MODULES[1]],
                            has_cuda=has_cuda, missing=tuple(missing), device=device,
                            hint=f"{hint}; torch {torch_version}" if hint else f"torch {torch_version}")


def pict_available() -> bool:
    """``True`` when PICT could run here (both modules import and a CUDA device is visible)."""
    return pict_availability().available


def _weights(n_cells: int, base: float, refine_lo: bool, refine_hi: bool) -> np.ndarray:
    """Cell widths on a unit interval, geometrically refined towards the low and/or high end."""
    if n_cells < 1:
        raise ValueError("a grid needs at least one cell")
    if base < 1.0:
        raise ValueError("base is the width ratio between neighbouring cells and must be >= 1")
    idx = np.arange(n_cells)
    if refine_lo and refine_hi:
        exponent = np.minimum(idx, n_cells - 1 - idx)      # symmetric about the middle
    elif refine_lo:
        exponent = idx
    elif refine_hi:
        exponent = n_cells - 1 - idx
    else:
        exponent = np.zeros(n_cells, dtype=int)
    w = base ** exponent.astype(float)
    return w / w.sum()


def wall_refined_grid(nx: int, ny: int, *, bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
                      refine: tuple[str, ...] = ("-x", "+x", "-y", "+y"), base: float = 1.10) -> np.ndarray:
    """Vertex coordinates of a structured, wall-refined quad grid, in PICT's layout ``(1, 2, ny+1, nx+1)``.

    ``bounds`` is ``(x0, y0, x1, y1)``. ``refine`` names the walls the cells are packed towards (``"-x"`` is the low-x
    wall). ``base`` is the ratio between the widths of neighbouring cells.

    PICT ships its own generator inside its example scripts (``lib/data/shapes.py``), which is not part of the
    package it installs; this one is ours, and `tests/test_pict_bridge.py` checks its spacing, its symmetry and its
    corners instead of trusting it.
    """
    x0, y0, x1, y1 = (float(v) for v in bounds)
    if not (x1 > x0 and y1 > y0):
        raise ValueError("bounds must satisfy x1 > x0 and y1 > y0")
    wx = _weights(int(nx), base, "-x" in refine, "+x" in refine) * (x1 - x0)
    wy = _weights(int(ny), base, "-y" in refine, "+y" in refine) * (y1 - y0)
    xs = np.concatenate([[x0], x0 + np.cumsum(wx)])
    ys = np.concatenate([[y0], y0 + np.cumsum(wy)])
    xs[-1], ys[-1] = x1, y1                            # exact corners, whatever the float sum did
    grid = np.empty((1, 2, ny + 1, nx + 1), dtype=np.float64)   # cast to the solver dtype when handed over
    grid[0, 0] = xs[None, :]
    grid[0, 1] = ys[:, None]
    return grid


@dataclass
class CavityCase:
    """A lid-driven cavity in ``[0, 1]^2``: no-slip walls, the lid moving at ``lid_velocity`` in +x.

    ``Re = lid_velocity * L / viscosity`` with ``L = 1``, so ``viscosity = lid_velocity / Re``. The Ghia et al.
    (1982) reference profile used by :mod:`~pinneapple_simulation.external_solvers.pict.validation` assumes the lid
    at the **y+** wall moving in **+x**; moving the lid elsewhere mirrors the profile and the check will say so.
    """

    nx: int = 32
    ny: int = 32
    Re: float = 100.0
    lid_velocity: float = 1.0
    refine: bool = True
    base: float = 1.10
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    lid_side: str = "+y"

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in m^2/s for this Reynolds number (``L = 1 m``)."""
        return self.lid_velocity * (self.bounds[2] - self.bounds[0]) / float(self.Re)

    @property
    def walls(self) -> tuple[str, ...]:
        """The walls to refine towards: all four when ``refine`` is set."""
        return ("-x", "+x", "-y", "+y") if self.refine else ()

    def check(self) -> list[str]:
        """Problems with the description, before any solver is touched. Empty when it is usable."""
        problems = []
        if self.nx < 4 or self.ny < 4:
            problems.append("nx and ny are the number of cells and must be at least 4")
        if self.Re <= 0:
            problems.append("Re must be positive")
        if self.lid_velocity == 0:
            problems.append("a cavity with a still lid has no flow: lid_velocity is 0")
        if self.base < 1.0:
            problems.append("base is the cell-width ratio and must be >= 1")
        if self.lid_side not in ("-x", "+x", "-y", "+y"):
            problems.append(f"lid_side must be one of '-x', '+x', '-y', '+y', not {self.lid_side!r}")
        if self.bounds[2] <= self.bounds[0] or self.bounds[3] <= self.bounds[1]:
            problems.append("bounds must satisfy x1 > x0 and y1 > y0")
        return problems

    def grid(self) -> np.ndarray:
        """Vertex coordinates in PICT's layout, ``(1, 2, ny+1, nx+1)``."""
        problems = self.check()
        if problems:
            raise ValueError("; ".join(problems))
        return wall_refined_grid(self.nx, self.ny, bounds=self.bounds, refine=self.walls, base=self.base)

    def summary(self) -> dict[str, Any]:
        """The case in SI units, for a manifest or a log line."""
        return {"nx": self.nx, "ny": self.ny, "Re": float(self.Re), "lid_velocity": float(self.lid_velocity),
                "viscosity": self.viscosity, "lid_side": self.lid_side, "refine": self.refine,
                "base": self.base, "bounds": list(self.bounds)}


def build_cavity_domain(case: CavityCase, *, device: str = "cuda", dtype: Any = None):
    """The ``PISOtorch.Domain`` and block for ``case``, with all boundaries closed and the lid set.

    Returns ``(domain, block)`` after ``domain.PrepareSolve()``, ready for
    :class:`pinneapple_simulation.external_solvers.pict.runner.PICTRunConfig`. Raises :class:`ImportError` with the
    install line when PICT is not available here.
    """
    availability = pict_availability()
    if not availability.available:
        raise ImportError(f"PICT cannot run here: {'; '.join(availability.missing)}. {availability.hint}")

    import PISOtorch
    import torch

    dtype = dtype or torch.float32
    problems = case.check()
    if problems:
        raise ValueError("; ".join(problems))
    grid = torch.tensor(case.grid(), dtype=dtype, device=device).contiguous()

    domain = PISOtorch.Domain(2, torch.tensor([case.viscosity], dtype=dtype),
                              name="CavityDomain", device=torch.device(device), dtype=dtype,
                              passiveScalarChannels=0)
    block = domain.CreateBlock(vertexCoordinates=grid, name="CavityBlock")
    block.CloseAllBoundaries()                                    # every wall no-slip
    lid = torch.tensor([[case.lid_velocity, 0.0]], dtype=dtype, device=torch.device(device))
    block.CloseBoundary(case.lid_side, lid)                        # the moving lid
    domain.PrepareSolve()
    return domain, block
