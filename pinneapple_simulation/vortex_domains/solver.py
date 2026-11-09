"""The 2-D viscous vortex domains (VVD) solver: incompressible Navier-Stokes, Lagrangian, mesh-free.

One step of :class:`VVD2D`, in the order the method requires (Dynnikova's VVD scheme, as implemented by the
reference ``vvflow`` release, reimplemented here from its documentation and the papers it cites):

1. the panels of every body at this instant; the core radius of each domain (its second neighbour, floored by a
   third of the panel length) and the merge of the domains that have become too close;
2. the source sheet that makes the walls impermeable, given the free stream, the free domains and the wall motion;
3. the vorticity the wall creates this step, at Lighthill's rate ``U_e dU_e/ds`` per unit length -- that flux is
   the boundary layer, its separation and the wake, and it is the only way vorticity enters the flow;
4. the velocity of every domain: free stream + the Biot-Savart sum over the domains (regularised Lamb kernel)
   + the source sheet + the two diffusion drifts (domain-domain and wall);
5. one midpoint (predictor-corrector) step of the positions;
6. domains that ended up inside a body are mirrored back out (their circulation is kept) and domains that left the
   box are dropped with their circulation reported as *lost*;
7. the hydrodynamic force as the wall pressure integrated over the panels, plus the friction of the wall shear.

Everything a run produces is in :attr:`VVD2D.history`: time, force, torque, number of domains, circulation,
impermeability and no-slip residuals, and the largest domain displacement of the step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import kernels
from .bodies import Assembly, PanelSet
from .boundary import SourceSheet, create_vorticity, edge_velocity, shear_slip, surface_slip
from .domains import Domains

__all__ = ["VVD2D", "VVDConfig", "step_size"]


@dataclass
class VVDConfig:
    """What a VVD run is asked for.

    ``re`` is the Reynolds number on ``reference_length`` and the reference velocity; ``nu = U_ref * L_ref / re``.
    ``dt`` follows the stability rule of the method (``dt <= 2 Re eps^2`` in ``U = L = 1`` units) when it is None.
    """

    re: float = 40_000.0
    u_inf: tuple[float, float] = (1.0, 0.0)
    rho: float = 1.0
    t_end: float = 1.0
    dt: float | None = None
    dt_safety: float = 1.0
    eps_factor: float = 1.0 / 3.0
    merge_factor: float = 0.4
    creation_offset: float = 0.6
    flux_limit: float = 4.0
    drift_limit: float = 0.5
    prune: float = 1e-9
    reference_length: float = 1.0
    reference_velocity: float | None = None
    margin: float = 15.0
    max_domains: int = 20000
    eps_growth: float = 1.25
    check_no_slip: bool = True
    store_snapshots: bool = False

    def __post_init__(self) -> None:
        if self.re <= 0:
            raise ValueError("re must be positive")
        if self.t_end <= 0:
            raise ValueError("t_end must be positive")
        if not 0.0 < self.eps_factor <= 1.0:
            raise ValueError("eps_factor must be in (0, 1]")
        self.u_inf = tuple(float(c) for c in self.u_inf)                 # type: ignore[assignment]

    @property
    def speed(self) -> float:
        """Magnitude of the free stream."""
        return float(np.hypot(*self.u_inf))

    @property
    def u_inf_array(self) -> np.ndarray:
        return np.asarray(self.u_inf, dtype=float)

    @property
    def velocity_scale(self) -> float:
        return float(self.reference_velocity if self.reference_velocity is not None else self.speed)

    @property
    def nu(self) -> float:
        """Kinematic viscosity implied by ``re`` on the reference velocity and length."""
        return self.velocity_scale * self.reference_length / self.re

    def with_(self, **changes: Any) -> VVDConfig:
        """A copy of this configuration with some fields replaced."""
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data.update(changes)
        return VVDConfig(**data)


def step_size(panels: PanelSet, re: float, eps_factor: float = 1.0 / 3.0, safety: float = 1.0,
              u_scale: float = 1.0) -> float:
    """Time step: the tighter of the method's viscous rule and the advective one.

    The viscous rule ``dt <= 2 Re eps^2`` is what keeps the diffusion drift stable; it is *not* the binding one at
    high Reynolds number, where advecting a domain across its own core (``dt <= eps / U``) is. Taking the smaller
    of the two is what lets one configuration serve both ``Re = 40`` and ``Re = 4e4``.
    """
    if len(panels) == 0:
        raise ValueError("cannot size a step for a body without panels")
    eps = eps_factor * float(np.min(panels.length))
    viscous = re * eps ** 2
    advective = safety * eps / max(float(u_scale), 1e-30)
    return float(min(viscous, advective))


class VVD2D:
    """A 2-D VVD simulation over an :class:`~.bodies.Assembly` of rigid bodies."""

    def __init__(self, assembly: Assembly, config: VVDConfig | None = None, name: str = "vvd"):
        self.assembly = assembly
        self.config = config or VVDConfig()
        self.name = name
        self.time = 0.0
        self.step_count = 0
        self.domains = Domains()
        self.panels = assembly.panels(0.0)
        self.sheet = SourceSheet(self.panels, np.zeros(len(self.panels)))
        self._eps_growth = 1.0
        self.dt = (self.config.dt if self.config.dt is not None
                   else step_size(self.panels, self.config.re, self.config.eps_factor, self.config.dt_safety,
                                  u_scale=max(self.config.speed, 1e-12)))
        self.shed = 0.0                     # circulation handed to the fluid at the walls so far
        self.lost = 0.0                     # circulation that left the box
        self.force = np.zeros(2)
        self.torque = 0.0
        self.friction = np.zeros(2)
        self.history: dict[str, list[float]] = {k: [] for k in
                                                ("t", "fx", "fy", "torque", "fx_friction", "fy_friction",
                                                 "n_domains", "circulation", "shed", "lost",
                                                 "impermeability", "slip", "max_displacement", "eps_floor")}
        self.snapshots: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self._matrix_key: tuple | None = None

    # -- infrastructure ---------------------------------------------------------------------------------
    def _panels_at(self, t: float) -> PanelSet:
        return self.assembly.panels(t)

    def _sheet_matrix(self, panels: PanelSet) -> np.ndarray:
        """The source-sheet matrix, cached while the geometry does not change."""
        key = (len(panels), float(np.round(panels.a.sum(), 12)), float(np.round(panels.length.sum(), 12)))
        if self._matrix is None or self._matrix_key != key:
            self._matrix = kernels.source_sheet_system(panels.mid, panels.a, panels.b, panels.normal)
            self._matrix_key = key
        return self._matrix

    def eps_floor(self, panels: PanelSet | None = None) -> float:
        """Core-radius floor: a third of the mean panel, grown when the run would exceed ``max_domains``.

        Growing it is how the solver trades resolution for a bounded cost instead of dying: the domains merge,
        the boundary layer gets coarser, and the run keeps going (and says so, in ``history["eps_floor"]``).
        """
        panels = self.panels if panels is None else panels
        if len(panels) == 0:
            return 1e-3
        floor = self.config.eps_factor * float(np.mean(panels.length)) * self._eps_growth
        return floor

    def induced_by_domains(self, points: np.ndarray) -> np.ndarray:
        """Velocity of the free domains at ``points`` (no free stream, no sheets)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if len(self.domains) == 0:
            return np.zeros_like(points)
        u, v = kernels.blob_velocity(points, self.domains.x, self.domains.y, self.domains.gamma, self.domains.eps)
        return np.column_stack([u, v])

    def fluid_velocity(self, points: np.ndarray, domains: Domains | None = None) -> np.ndarray:
        """Full fluid velocity at ``points``: free stream + domains + the source sheet."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        dom = self.domains if domains is None else domains
        u = np.zeros(len(points))
        v = np.zeros(len(points))
        if len(dom):
            u, v = kernels.blob_velocity(points, dom.x, dom.y, dom.gamma, dom.eps)
        u = u + self.config.u_inf[0]
        v = v + self.config.u_inf[1]
        if len(self.sheet.panels):
            su, sv = self.sheet.velocity(points)
            u = u + su
            v = v + sv
        return np.column_stack([u, v])

    def drift(self, points: np.ndarray, domains: Domains | None = None) -> np.ndarray:
        """The two diffusive drifts (domain-domain and wall) at ``points``, clamped to the stability limit."""
        dom = self.domains if domains is None else domains
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if len(dom) == 0:
            return np.zeros_like(points)
        nu = self.config.nu
        vx, vy = kernels.vortex_diffusion_drift(dom.x, dom.y, dom.gamma, dom.eps, nu)
        if len(self.panels):
            wx, wy = kernels.wall_diffusion_drift(points[:, 0], points[:, 1], dom.eps,
                                                  self.panels.mid, self.panels.vector, nu)
            vx = vx + wx
            vy = vy + wy
        limit = self.config.drift_limit * dom.eps / self.dt
        mag = np.hypot(vx, vy)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(mag > limit, limit / np.where(mag > 0, mag, 1.0), 1.0)
        return np.column_stack([vx * scale, vy * scale])

    # -- one step ---------------------------------------------------------------------------------------
    def step(self) -> None:
        """Advance the flow by one time step."""
        cfg = self.config
        t = self.time
        self.panels = panels = self._panels_at(t)
        while len(self.domains) > cfg.max_domains:
            self._eps_growth *= cfg.eps_growth
        eps_floor = self.eps_floor(panels)
        self.domains.update_epsilon(eps_floor)
        if cfg.merge_factor > 0 and len(self.domains) > 1:
            self.domains.merge(eps_floor * cfg.merge_factor, factor=cfg.merge_factor)

        domain_at_mid = self.induced_by_domains(panels.mid)

        # 2. no through flow: the source sheet
        if len(panels):
            self.sheet = SourceSheet.solve(panels, cfg.u_inf_array, domain_at_mid, self._sheet_matrix(panels))
            sheet_mid = self.sheet.at_midpoints()
            at_wall = domain_at_mid + np.column_stack(sheet_mid) + cfg.u_inf_array[None, :]
            points, circulation = create_vorticity(panels, edge_velocity(panels, at_wall), self.dt, eps_floor,
                                                   cfg.creation_offset, limit=cfg.flux_limit * cfg.speed ** 2)
        else:
            self.sheet = SourceSheet(panels, np.zeros(0))
            points, circulation = np.zeros((0, 2)), np.zeros(0)

        # 3. the vorticity the wall creates this step (Lighthill's flux)
        keep = np.abs(circulation) > cfg.prune * max(1.0, float(np.abs(circulation).max(initial=0.0)))
        points, circulation = points[keep], circulation[keep]
        if points.size:
            self.domains.extend(points, circulation)
            self.shed += float(circulation.sum())
            self.domains.update_epsilon(eps_floor)

        # 4./5. advect with a midpoint (predictor-corrector) step
        max_displacement = 0.0
        if len(self.domains):
            positions = self.domains.positions
            v1 = self.fluid_velocity(positions) + self.drift(positions)
            half = positions + 0.5 * self.dt * v1
            v2 = self.fluid_velocity(half) + self.drift(half)
            max_displacement = float((self.dt * np.hypot(v2[:, 0], v2[:, 1])).max())
            self.domains.x = positions[:, 0] + self.dt * v2[:, 0]
            self.domains.y = positions[:, 1] + self.dt * v2[:, 1]
            self.domains.velocity = v2

        # 6. keep the vorticity in the fluid: mirror back what went inside, report what left the box
        if len(self.domains):
            positions = self.domains.positions
            inside = self.assembly.inside(positions, t + self.dt)
            if np.any(inside):
                self._mirror_out(inside)
                positions = self.domains.positions
            lo, hi = self._box()
            outside = ((positions[:, 0] < lo[0]) | (positions[:, 0] > hi[0])
                       | (positions[:, 1] < lo[1]) | (positions[:, 1] > hi[1]))
            if np.any(outside):
                dropped = self.domains.subsets(outside)
                self.lost += float(dropped.gamma.sum())
                self.domains.remove(outside)

        # 7. the force: the wall pressure integrated over the panels, plus the friction of the wall shear
        self.force = self._pressure_force(panels) if len(panels) else np.zeros(2)
        self.friction = self._friction_force(panels)
        self.torque = self._torque(circulation if points.size else None, points if points.size else None)

        # 8. book keeping
        slip_residual = impermeability = 0.0
        if len(panels):
            field_at_mid = domain_at_mid + np.column_stack(self.sheet.at_midpoints()) + cfg.u_inf_array[None, :]
            slip_residual = float(np.abs(surface_slip(panels, field_at_mid)).max())
            impermeability = float(np.abs(shear_slip(panels, field_at_mid)).max())
        self.time += self.dt
        self.step_count += 1
        self._record(impermeability, slip_residual, max_displacement)
        if cfg.store_snapshots:
            self.snapshots.append(self.snapshot())

    def _mirror_out(self, inside: np.ndarray) -> None:
        """Put the domains that entered a body back outside the nearest panel (circulation untouched)."""
        panels = self.panels
        mask = np.asarray(inside, dtype=bool)
        pts = self.domains.positions[mask]
        dist = ((pts[:, None, :] - panels.mid[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(dist, axis=1)
        offset = 0.5 * self.eps_floor(panels)
        reflected = panels.mid[nearest] + panels.normal[nearest] * offset
        self.domains.x[np.flatnonzero(mask)] = reflected[:, 0]
        self.domains.y[np.flatnonzero(mask)] = reflected[:, 1]

    def _box(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.assembly) == 0:
            return np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        vertices = np.vstack([b.world_vertices(0.0) for b in self.assembly.bodies])
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        center = 0.5 * (lo + hi)
        half = 0.5 * span * self.config.margin
        return center - half, center + half

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Pressure relative to the free stream, ``rho (V_inf^2 - |V|^2) / 2 + rho sum_i V_i . u_i``.

        The first term is the steady Bernoulli part (which carries the potential flow, sources included) and the
        second is the unsteady part written over the domains and the velocity they move with.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        velocity = self.fluid_velocity(points)
        dynamic = 0.5 * self.config.rho * (self.config.speed ** 2 - (velocity ** 2).sum(axis=1))
        if len(self.domains):
            dynamic = dynamic + self.config.rho * kernels.blob_pressure_sum(
                points, self.domains.x, self.domains.y, self.domains.gamma, self.domains.velocity, self.domains.eps)
        return dynamic

    def wall_pressure(self, panels: PanelSet | None = None) -> np.ndarray:
        """Pressure at the panel midpoints, from the fluid side of the sheets.

        Pressure on the wall needs the one-sided limit of the sheets (the ``+sigma/2`` and ``+gamma/2`` half jumps):
        evaluating the generic :meth:`pressure` exactly on a panel would read the singular branch instead.
        """
        panels = self.panels if panels is None else panels
        if len(panels) == 0:
            return np.zeros(0)
        induced = self.induced_by_domains(panels.mid)
        u, v = self.sheet.at_midpoints()
        velocity = induced + self.config.u_inf_array[None, :] + np.column_stack([u, v])
        dynamic = 0.5 * self.config.rho * (self.config.speed ** 2 - (velocity ** 2).sum(axis=1))
        if len(self.domains):
            dynamic = dynamic + self.config.rho * kernels.blob_pressure_sum(
                panels.mid, self.domains.x, self.domains.y, self.domains.gamma, self.domains.velocity,
                self.domains.eps)
        return dynamic

    def _pressure_force(self, panels: PanelSet) -> np.ndarray:
        """``-integral p n dl`` over the panels: the wall pressure pushing on the body."""
        p = self.wall_pressure(panels)
        return -(p[:, None] * panels.normal * panels.length[:, None]).sum(axis=0)

    def _friction_force(self, panels: PanelSet) -> np.ndarray:
        """Wall shear integrated over the panels: ``-rho nu omega_wall t`` with ``omega`` from the near-wall domains."""
        if len(self.domains) == 0 or len(panels) == 0:
            return np.zeros(2)
        omega = kernels.blob_vorticity(panels.mid, self.domains.x, self.domains.y,
                                       self.domains.gamma, self.domains.eps)
        # a wall is half a free space: the image doubles the vorticity the boundary layer carries there
        tau = self.config.rho * self.config.nu * 2.0 * omega
        return (tau[:, None] * panels.tangent * panels.length[:, None]).sum(axis=0)

    def _torque(self, circulation: np.ndarray | None, points: np.ndarray | None) -> float:
        """Torque about the first body's reference point, from the vorticity the wall handed to the fluid."""
        if circulation is None or points is None:
            return 0.0
        p = points - self._pivot()
        return float(0.5 * (circulation * (p[:, 0] ** 2 + p[:, 1] ** 2)).sum() / self.dt)

    def _pivot(self) -> np.ndarray:
        if len(self.assembly) == 0:
            return np.zeros(2)
        return self.assembly.bodies[0].motion.ring(self.time)[0]

    def _record(self, impermeability: float, slip: float, max_displacement: float) -> None:
        h = self.history
        h["t"].append(self.time)
        h["fx"].append(float(self.force[0]))
        h["fy"].append(float(self.force[1]))
        h["torque"].append(self.torque)
        h["fx_friction"].append(float(self.friction[0]))
        h["fy_friction"].append(float(self.friction[1]))
        h["n_domains"].append(len(self.domains))
        h["circulation"].append(self.domains.total_circulation)
        h["shed"].append(self.shed)
        h["lost"].append(self.lost)
        h["impermeability"].append(impermeability)
        h["slip"].append(slip)
        h["max_displacement"].append(max_displacement)
        h["eps_floor"].append(self.eps_floor())

    # -- running ----------------------------------------------------------------------------------------
    def run(self, t_end: float | None = None, progress: Any = None) -> dict[str, list[float]]:
        """Advance until ``t_end`` (default: the configuration's), calling ``progress(self)`` every step."""
        target = self.config.t_end if t_end is None else float(t_end)
        if target <= self.time:
            raise ValueError("t_end must be larger than the current time")
        n_steps = int(np.ceil((target - self.time) / self.dt - 1e-9))
        for _ in range(n_steps):
            self.step()
            if progress is not None:
                progress(self)
        return self.history

    # -- results ----------------------------------------------------------------------------------------
    @property
    def coefficients(self) -> dict[str, float]:
        """``cl``, ``cd`` and ``cm`` at the last step, on ``reference_length`` and the reference velocity."""
        dynamic = 0.5 * self.config.rho * self.config.velocity_scale ** 2 * self.config.reference_length
        total = self.force + self.friction
        return {"cl": float(total[1] / dynamic), "cd": float(total[0] / dynamic),
                "cm": float(self.torque / (dynamic * self.config.reference_length)),
                "cd_pressure": float(self.force[0] / dynamic), "cd_friction": float(self.friction[0] / dynamic),
                "t": self.time}

    def series(self) -> dict[str, np.ndarray]:
        """The history as arrays."""
        return {k: np.asarray(v, dtype=float) for k, v in self.history.items()}

    def snapshot(self) -> dict[str, Any]:
        """A copy of everything a picture or a restart needs."""
        return {
            "t": self.time,
            "x": self.domains.x.copy(), "y": self.domains.y.copy(),
            "gamma": self.domains.gamma.copy(), "eps": self.domains.eps.copy(),
            "panels": {"a": self.panels.a.copy(), "b": self.panels.b.copy(), "sigma": self.sheet.sigma.copy()},
            "force": self.force.copy(), "friction": self.friction.copy(),
        }

    def load_snapshot(self, snap: dict[str, Any]) -> None:
        """Restore the state of :meth:`snapshot` (the geometry is re-derived from the assembly)."""
        self.time = float(snap["t"])
        self.domains = Domains(snap["x"], snap["y"], snap["gamma"])
        self.domains.eps = np.asarray(snap["eps"], dtype=float)
        self.panels = self._panels_at(self.time)
        self.sheet = SourceSheet(self.panels, np.asarray(snap["panels"]["sigma"], dtype=float))
        self.force = np.asarray(snap["force"], dtype=float)
        self.friction = np.asarray(snap["friction"], dtype=float)

    def __repr__(self) -> str:                                          # pragma: no cover - debugging aid
        return (f"VVD2D({self.name!r}, Re={self.config.re:g}, t={self.time:.3f}, "
                f"n={len(self.domains)}, dt={self.dt:.3e})")
