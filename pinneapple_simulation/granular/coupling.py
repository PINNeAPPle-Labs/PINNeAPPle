"""Two-way-capable fluid drag on a granular assembly: the particle side of a DEM-CFD coupling.

A mill runs wet as often as dry, and the drag a fluid exerts on the charge is what decides whether the slurry
carries the fines or the charge settles.  This module adds the particle side of that coupling to a
:class:`~pinneapple_simulation.granular.world.World`: the fluid velocity is a callable (an analytic field, a SPH
solution resampled onto the particle positions, or a field read back from an OpenFOAM run), and the drag force
follows the Schiller-Naumann correlation already used by
:func:`pinneapple_simulation.numerical_solvers.particles.terminal_velocity`, so a single settling body has to
reproduce that function's answer -- that is the check in ``tests/test_granular_coupling.py``.

    >>> import numpy as np
    >>> from pinneapple_simulation.granular import Sphere, World
    >>> from pinneapple_simulation.granular.coupling import FluidCoupling
    >>> world = World(gravity=(0.0, -9.81, 0.0))
    >>> _ = world.add(Sphere(1.0e-3, density=2500.0))
    >>> world.finalize()
    >>> coupling = FluidCoupling(velocity=lambda x: np.zeros_like(x), rho=998.0, mu=1.0e-3)
    >>> coupling.attach(world)                                  # doctest: +ELLIPSIS
    0
    >>> coupling.drag_force(world).shape
    (1, 3)

The porosity factor ``eps^-chi`` (Di Felice 1994, *The voidage function for fluid-particle interaction systems*,
Int. J. Multiphase Flow 20:153-159) is applied when ``porosity=True`` and uses the local solids fraction measured on
a grid of cells about four particle diameters wide.  It is off by default: the voidage function describes a *bed*,
and applying it to a lone particle is a modelling choice rather than a correction -- with a dilute assembly it
returns the single-particle drag to within the grid's resolution.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["FluidCoupling", "schiller_naumann_drag"]

_WATER_RHO = 998.2
_WATER_MU = 1.002e-3


def _is_drag_of(source, coupling: FluidCoupling) -> bool:
    """True when ``source`` is the lambda :meth:`FluidCoupling.attach` registered."""
    closure = getattr(source, "__closure__", None) or ()
    return any(cell.cell_contents is coupling for cell in closure)


def schiller_naumann_drag(reynolds: np.ndarray) -> np.ndarray:
    """Drag coefficient ``24/Re (1 + 0.15 Re^0.687)`` (valid to Re ~ 1000, extended to 800 by the source)."""
    re = np.maximum(np.asarray(reynolds, dtype=float), 1.0e-12)
    cd = 24.0 / re * (1.0 + 0.15 * np.power(re, 0.687))
    return np.where(re < 1.0e-6, 24.0 / 1.0e-6, cd)


@dataclass
class FluidCoupling:
    """Drag of a carrier fluid on the bodies of a world.

    ``velocity`` is either a constant ``(3,)`` array or a callable ``positions (n, 3) -> velocities (n, 3)``.
    ``porosity`` adds the Di Felice voidage correction from the local solids fraction; ``chi`` is its exponent
    (3.7 is the usual value for the whole Reynolds range).
    """

    velocity: Callable[[np.ndarray], np.ndarray] | np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    rho: float = _WATER_RHO
    mu: float = _WATER_MU
    porosity: bool = False             # opt-in: the voidage function is meant for a bed, not a lone particle
    buoyancy: bool = True              # add -rho_f V g, so gravity and drag give the submerged weight
    chi: float = 3.7
    cells: int = 8                     # grid resolution per axis for the local solids fraction
    _index: int | None = None

    def field(self, positions: np.ndarray) -> np.ndarray:
        v = self.velocity
        if callable(v):
            out = np.asarray(v(positions), dtype=float)
        else:
            out = np.broadcast_to(np.asarray(v, dtype=float), positions.shape).copy()
        if out.shape != positions.shape:
            raise ValueError(f"the fluid velocity field returned {out.shape}, expected {positions.shape}")
        return out

    def solids_fraction(self, world) -> np.ndarray:
        """Local solids fraction at each body, on a grid of cells about two particle diameters wide.

        The grid box is the assembly's bounding box grown by one diameter and the cell size follows the particles,
        so a single isolated body lands in a cell much bigger than itself and reads a voidage near one -- which is
        what makes the correction reduce to the single-particle drag at infinite dilution instead of blowing up.
        """
        diameter = float(2.0 * np.max(np.where(world.is_sphere, world.sphere_radius, world.radius)))
        lo = world.position.min(0) - 2.0 * diameter
        hi = world.position.max(0) + 2.0 * diameter
        span = np.maximum(hi - lo, 6.0 * diameter)       # a lone body must not fill its own cell
        n = int(np.clip(math.ceil(float(span.max()) / max(4.0 * diameter, 1.0e-30)), 1, 16))
        idx = np.clip(((world.position - lo) / span * n).astype(int), 0, n - 1)
        flat = (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]
        occupied = np.zeros(n ** 3)
        np.add.at(occupied, flat, world.shape_volume)
        cell_volume = float(np.prod(span / n))
        fraction = occupied / max(cell_volume, 1.0e-30)
        return np.clip(fraction[flat], 0.0, 0.64)        # random close packing is the physical ceiling

    def drag_force(self, world) -> np.ndarray:
        """Fluid force on every body [N]: the drag on the slip, plus buoyancy when enabled."""
        u = self.field(world.position)
        slip = u - world.velocity
        speed = np.sqrt((slip * slip).sum(1))
        radius = np.where(world.is_sphere, world.sphere_radius, world.radius)
        diameter = 2.0 * radius
        reynolds = self.rho * speed * diameter / max(self.mu, 1.0e-30)
        cd = schiller_naumann_drag(reynolds)
        area = np.pi * radius ** 2
        magnitude = 0.5 * self.rho * cd * area * speed * speed            # 1/2 rho Cd A |u - v|^2
        drag = magnitude[:, None] * slip / np.maximum(speed, 1.0e-30)[:, None]
        if self.porosity:
            dilution = np.clip(1.0 - self.solids_fraction(world), 1.0e-3, 1.0)
            drag = drag * np.power(dilution, -self.chi)[:, None]
        if self.buoyancy:
            drag = drag - self.rho * world.shape_volume[:, None] * world.gravity[None, :]
        return drag

    def attach(self, world) -> int:
        """Register the drag as a force source of ``world`` (returns its index in ``world.force_sources``)."""
        self._index = world.add_force_source(lambda w: self.drag_force(w))
        return self._index

    def detach(self, world) -> None:
        """Remove the drag source again (e.g. to run the same world dry)."""
        if self._index is None:
            return
        world.force_sources = [s for s in world.force_sources if not _is_drag_of(s, self)]
        self._index = None
