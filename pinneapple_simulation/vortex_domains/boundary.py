"""The wall boundary of the VVD solver: the source sheet for no through flow, the vortex sheet for no slip.

Two conditions hold on a solid wall, and they are solved by two panel systems.

No through flow (:class:`SourceSheet`)
    A source sheet cancels the normal velocity of everything else (free stream, free domains, the bound vortex
    sheet and the body's own motion): ``A sigma = -n . (V_inf + V_others - V_body)``, with the classical ``+1/2``
    self term. On a cylinder it reproduces ``Cp = 1 - 4 sin^2(theta)`` to machine precision (see the tests).

No slip (:class:`VortexSheet`)
    A vortex sheet on the panels stands for the vorticity attached to the wall. It is solved each step so that the
    tangential velocity at every panel midpoint equals the wall's, ``B gamma = (V_body - V_others) . t``, and its
    *change* since the previous step is shed into the flow as new vortex domains (:func:`shed_change`). Nothing is
    created and nothing destroyed at the wall -- the vorticity that leaves the surface is exactly the part the
    attached sheet gave up -- which is what keeps the total circulation of the system under control.

The Kutta condition is not imposed: at a sharp trailing edge the sheet solve produces the large strength that
cannot satisfy no slip there, and shedding it is what puts the right circulation into the wake.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .bodies import PanelSet

__all__ = ["SourceSheet", "VortexSheet", "create_vorticity", "edge_gradient", "edge_velocity",
           "shear_slip", "surface_slip", "vorticity_flux"]


def surface_slip(panels: PanelSet, velocity_at_mid: np.ndarray) -> np.ndarray:
    """Tangential component of ``velocity_at_mid - V_body``, positive along the panel tangent."""
    relative = np.asarray(velocity_at_mid, dtype=float) - panels.velocity
    return (relative * panels.tangent).sum(axis=1)


def shear_slip(panels: PanelSet, velocity_at_mid: np.ndarray) -> np.ndarray:
    """Normal component of ``velocity_at_mid - V_body`` -- the residual of the no-through-flow condition."""
    relative = np.asarray(velocity_at_mid, dtype=float) - panels.velocity
    return (relative * panels.normal).sum(axis=1)


@dataclass
class SourceSheet:
    """A constant-strength source sheet on the panels, enforcing ``V . n = V_body . n`` at every midpoint."""

    panels: PanelSet
    sigma: np.ndarray
    residual: float = 0.0

    @classmethod
    def solve(cls, panels: PanelSet, u_inf: np.ndarray, others: np.ndarray,
              matrix: np.ndarray | None = None) -> SourceSheet:
        """Solve for the sheet that cancels the normal velocity at the midpoints.

        ``others`` is the velocity at the midpoints of everything that is not this sheet (free domains, the bound
        vortex sheet), i.e. ``V_inf + others`` is the flow the sheet has to bend.
        """
        if len(panels) == 0:
            return cls(panels, np.zeros(0))
        if matrix is None:
            matrix = kernels.source_sheet_system(panels.mid, panels.a, panels.b, panels.normal)
        target = -((u_inf[None, :] + others - panels.velocity) * panels.normal).sum(axis=1)
        sigma = np.linalg.solve(matrix, target)
        return cls(panels, sigma)

    def velocity(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocity induced by the sheet at ``points``."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if len(self.panels) == 0 or points.size == 0:
            return np.zeros(len(points)), np.zeros(len(points))
        return kernels.panel_flux(points, self.panels.a, self.panels.b, self.sigma, "source")

    def at_midpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Sheet velocity at its own collocation points, from the fluid side.

        The other panels give the velocity at the midpoint; the panel's own share is the classical half jump,
        ``+sigma/2`` along the outward normal (the fluid-side limit).
        """
        u, v = kernels.panel_flux(self.panels.mid, self.panels.a, self.panels.b, self.sigma, "source",
                                  exclude_diagonal=True)
        u = u + 0.5 * self.sigma * self.panels.normal[:, 0]
        v = v + 0.5 * self.sigma * self.panels.normal[:, 1]
        return u, v

    def impermeability(self, u_inf: np.ndarray, others: np.ndarray) -> np.ndarray:
        """Normal velocity at the midpoints, relative to the wall: zero where the sheet is doing its job."""
        u, v = self.at_midpoints()
        total = np.column_stack([u + u_inf[0] + others[:, 0], v + u_inf[1] + others[:, 1]])
        return shear_slip(self.panels, total)


@dataclass
class VortexSheet:
    """A constant-strength vortex sheet on the panels: the vorticity attached to the wall."""

    panels: PanelSet
    gamma: np.ndarray
    residual: float = 0.0

    @classmethod
    def solve(cls, panels: PanelSet, u_inf: np.ndarray, others: np.ndarray,
              matrix: np.ndarray | None = None) -> VortexSheet:
        """Solve for the sheet that makes the tangential velocity at the midpoints equal to the wall's."""
        if len(panels) == 0:
            return cls(panels, np.zeros(0))
        if matrix is None:
            matrix = kernels.vortex_sheet_system(panels.mid, panels.a, panels.b, panels.tangent)
        target = ((panels.velocity - (u_inf[None, :] + others)) * panels.tangent).sum(axis=1)
        gamma = np.linalg.solve(matrix, target)
        return cls(panels, gamma)

    def velocity(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if len(self.panels) == 0 or points.size == 0:
            return np.zeros(len(points)), np.zeros(len(points))
        return kernels.panel_flux(points, self.panels.a, self.panels.b, self.gamma, "vortex")

    def at_midpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Sheet velocity at its collocation points, from the fluid side (the ``+gamma/2`` half jump)."""
        u, v = kernels.panel_flux(self.panels.mid, self.panels.a, self.panels.b, self.gamma, "vortex",
                                  exclude_diagonal=True)
        u = u + 0.5 * self.gamma * self.panels.tangent[:, 0]
        v = v + 0.5 * self.gamma * self.panels.tangent[:, 1]
        return u, v

    def slip(self, u_inf: np.ndarray, others: np.ndarray) -> np.ndarray:
        """Tangential velocity at the midpoints, relative to the wall: zero where no slip holds."""
        u, v = self.at_midpoints()
        total = np.column_stack([u + u_inf[0] + others[:, 0], v + u_inf[1] + others[:, 1]])
        return surface_slip(self.panels, total)


def edge_velocity(panels: PanelSet, velocity_at_mid: np.ndarray) -> np.ndarray:
    """Tangential velocity of the outer flow at the wall -- the boundary layer's edge velocity ``U_e``.

    It is the slip left by the potential flow (free stream, free domains and the source sheet), i.e. exactly the
    velocity the boundary layer has to absorb.
    """
    return surface_slip(panels, velocity_at_mid)


def edge_gradient(panels: PanelSet, u_e: np.ndarray, min_spacing: float = 0.0) -> np.ndarray:
    """``dU_e/ds`` along the wall, by a central difference along each body's panel chain.

    ``min_spacing`` keeps the difference from exploding on the panels of a sharp trailing edge, which are far
    shorter than anything the wall can resolve: the wall cannot create vorticity on a scale finer than the domains
    that will carry it.
    """
    u_e = np.asarray(u_e, dtype=float)
    out = np.zeros_like(u_e)
    for index in np.unique(panels.body) if len(panels) else []:
        mask = np.flatnonzero(panels.body == index)
        if mask.size < 3:
            continue
        u = u_e[mask]
        half = 0.5 * (panels.length[mask] + np.roll(panels.length[mask], -1))
        # s[k] is the arclength at the midpoint of panel k; the spacing between midpoints k-1 and k+1
        ds = np.maximum(np.roll(half, 1) + half, 2.0 * min_spacing)
        out[mask] = (np.roll(u, -1) - np.roll(u, 1)) / ds
    return out


def vorticity_flux(panels: PanelSet, u_e: np.ndarray, min_spacing: float = 0.0,
                   limit: float | None = None) -> np.ndarray:
    """Lighthill's wall vorticity flux ``nu dw/dn = U_e dU_e/ds``, the rate at which the wall creates vorticity.

    Negative where the outer flow accelerates (the wall injects negative vorticity), positive where it decelerates,
    zero at the stagnation points and at the pressure minimum -- so the boundary layer, its separation and the wake
    all come out of the outer flow instead of from a free parameter. ``U_e dU_e/ds`` does not depend on which way
    ``s`` runs (both factors change sign together).
    """
    flux = np.asarray(u_e, dtype=float) * edge_gradient(panels, u_e, min_spacing)
    if limit is not None:
        # a sharp trailing edge leaves a velocity singularity in the potential flow (the Kutta condition would
        # smooth it); the wall cannot create vorticity at a rate the domains cannot carry, so the flux is capped
        flux = np.clip(flux, -limit, limit)
    return flux


def create_vorticity(panels: PanelSet, u_e: np.ndarray, dt: float, eps: float,
                     offset: float = 0.6, threshold: float = 0.0,
                     limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The domains the wall creates in one step: circulation ``-U_e dU_e/ds * length * dt`` at every panel.

    Returns ``(points, circulation)``. This is the only way vorticity enters the flow: the body hands it to the
    fluid at the rate Lighthill's relation asks for.
    """
    if len(panels) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    circulation = vorticity_flux(panels, u_e, min_spacing=eps, limit=limit) * panels.length * dt
    keep = np.abs(circulation) > threshold
    points = panels.mid + panels.normal * (offset * eps)
    return points[keep], circulation[keep]
