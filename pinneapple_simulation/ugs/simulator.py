"""The reference simulator for underground gas storage: single-phase real gas in a porous medium, with wells.

The model
---------
One equation for the pressure, written as a material balance in surface volumes::

    d/dt [ phi(p) / Bg(p) ] = div[ (1/Bg) (k/mu(p)) grad p ] + q_wells

``Bg`` is the gas formation volume factor (reservoir m3 per m3 at standard conditions), so the accumulator
``phi/Bg`` is the surface volume that one cubic metre of bulk rock holds, the flux is a surface volume per
second, and every rate the simulator reports is in m3 at standard conditions -- the unit an operator books.

Space is discretised with a two-point flux approximation: the flux across a face is
``T (1/Bg)/mu (p_b - p_a)`` with ``T`` the harmonic transmissibility from the grid and the density-mobility
group taken from the upstream cell, which keeps the scheme monotone for a strongly pressure-dependent
``1/Bg``. Time is fully implicit (backward Euler) and every step is solved by Newton with an analytic
Jacobian and a sparse direct solve, with a backtracking line search that keeps the pressure positive.

Wells
-----
A well is a source/sink in one cell, ``q = WI (1/Bg)/mu (p_bhp - p)``, resolved by the active set of
:func:`pinneapple_simulation.ugs.wells.operating_point`: a rate target that would need a bottom-hole pressure
outside the well's limits switches to that pressure, and a rate above the well's limit is capped.

Diagnostics
-----------
Every run reports, at every time step, the pressure field, the pore-volume-weighted average pressure, the gas
in place, the rate and the bottom-hole pressure of every well, the daily deliverability and a **measured
mass-balance error**: the gas in place at the end minus the gas in place at the start minus what the wells
moved, divided by the gas in place the run started with. The well terms enter the residual at the new time
level, so that number is the solver's own tolerance, not an estimate. Non-convergence is reported, never
dressed: :attr:`SimulationResult.converged` is ``False`` and :attr:`SimulationResult.reason` says at which
time and after how many Newton iterations the step stopped.

Units: pressure Pa, length m, permeability m2, viscosity Pa s, rate m3 at standard conditions per second
(m3/day in the reports).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .wells import Control, Schedule, operating_point

__all__ = ["SimConfig", "SimulationResult", "UGSSimulator"]

SECONDS_PER_DAY = 86400.0
P_MIN = 1.0e3          # the pressure a backtracking step is not allowed to go below [Pa]


@dataclass(frozen=True)
class SimConfig:
    """Time stepping and Newton tolerances."""

    dt_days: float = 1.0
    max_newton: int = 30
    tol_rel: float = 1e-9          # convergence: max |dp| <= tol_rel * max(p)
    max_backtrack: int = 8


@dataclass
class SimulationResult:
    """What a run produced, with the diagnostics an engineer reads first."""

    times_days: np.ndarray               # (n_steps + 1,)
    pressure: np.ndarray                 # (n_steps + 1, n_cells) [Pa]
    mean_pressure: np.ndarray            # (n_steps + 1,) pore-volume weighted [Pa]
    gas_in_place: np.ndarray             # (n_steps + 1,) [m3 standard]
    well_rate: dict[str, np.ndarray]     # (n_steps + 1,) [m3 standard / day], positive for injection
    well_bhp: dict[str, np.ndarray]      # (n_steps + 1,) [Pa]
    well_cumulative: dict[str, np.ndarray]   # (n_steps + 1,) [m3 standard], positive for injection
    mass_balance_error: float            # (GIP_end - GIP_0 - net injected) / GIP_0
    converged: bool
    reason: str = ""
    newton_iterations: list[int] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.times_days) - 1

    def deliverability(self) -> np.ndarray:
        """Total withdrawal rate at every step [m3 standard per day], positive when gas leaves the reservoir."""
        total = np.zeros_like(self.times_days)
        for r in self.well_rate.values():
            total -= r
        return total

    def summary(self) -> str:
        """One paragraph with the numbers a report quotes."""
        gip0 = float(self.gas_in_place[0])
        produced = float(-np.sum([c[-1] for c in self.well_cumulative.values()]))
        return (
            f"{self.n_steps} steps to {self.times_days[-1]:.2f} d, "
            f"mean pressure {self.mean_pressure[0] / 1e6:.3f} -> {self.mean_pressure[-1] / 1e6:.3f} MPa, "
            f"gas in place {gip0 / 1e6:.4f} -> {self.gas_in_place[-1] / 1e6:.4f} Mm3, "
            f"net withdrawal {produced / 1e6:.4f} Mm3, "
            f"mass balance error {self.mass_balance_error:.2e}, "
            f"peak deliverability {self.deliverability().max():,.1f} m3/d, "
            f"{'converged' if self.converged else 'NOT converged: ' + self.reason}"
        )


class UGSSimulator:
    """Fully implicit single-phase gas simulator on a :class:`~pinneapple_simulation.ugs.grid.Grid`.

    ``p0`` is the initial pressure, a scalar or a value per cell. The wells' indices come from the grid unless
    the well carries its own.
    """

    def __init__(self, grid, fluid, rock, wells, *, p0: float):
        self.grid = grid
        self.fluid = fluid
        self.rock = rock
        self.wells = list(wells)
        self.p0 = np.broadcast_to(np.asarray(p0, dtype=float), (grid.n_cells,)).astype(float).copy()
        self.wi = {}
        for w in self.wells:
            self.wi[w.name] = grid.well_index(w.cell, w.r_w, w.skin) if getattr(w, "wi", None) is None else w.wi
        names = [w.name for w in self.wells]
        if len(set(names)) != len(names):
            raise ValueError(f"well names must be unique, got {names}")
        self.well_by_name = {w.name: w for w in self.wells}

    # ------------------------------------------------------------------ state
    def _state(self, p: np.ndarray):
        phi = self.rock.porosity(self.grid.phi0, p)
        dphi = self.rock.dporosity_dp(self.grid.phi0, p)
        index = self.fluid.index(p)
        dindex = self.fluid.dindex_dp(p)
        mu = self.fluid.mu(p)
        dmu = self.fluid.dmu_dp(p)
        n = phi * index
        dn = dphi * index + phi * dindex
        lam = index / mu
        dlam = (dindex * mu - index * dmu) / (mu * mu)
        return n, dn, lam, dlam

    def gas_in_place(self, p: np.ndarray) -> float:
        """Gas in place [m3 standard] of the pressure field ``p``."""
        n, _, _, _ = self._state(p)
        return float(np.sum(self.grid.volumes * n))

    def gas_in_place_uniform(self, p: float) -> float:
        """Gas in place [m3 standard] if the whole reservoir were at the pressure ``p``."""
        return self.gas_in_place(np.full(self.grid.n_cells, float(p)))

    def pressure_for_gas_in_place(self, gip: float, *, guess: float | None = None,
                                  tol: float = 1e-10, max_iters: int = 60) -> float:
        """The uniform pressure that holds ``gip`` [m3 standard]: the tank relation inverted by Newton.

        This is what a material-balance calculation asks for, and what the tank baseline of the benchmark uses.
        Raises ``ValueError`` rather than returning a pressure when the target is outside what the rock holds at
        any positive pressure.
        """
        p = float(guess if guess is not None else np.mean(self.p0))
        for _ in range(max_iters):
            n, dn, _, _ = self._state(np.full(self.grid.n_cells, p))
            f = float(np.sum(self.grid.volumes * n)) - gip
            df = float(np.sum(self.grid.volumes * dn))
            if df == 0.0:
                raise ValueError("the gas in place does not depend on the pressure here (incompressible fluid "
                                 "and rock): the material balance cannot be inverted")
            p_new = p - f / df
            if p_new <= P_MIN:
                raise ValueError(f"the target gas in place {gip:g} m3 needs a pressure below {P_MIN:g} Pa")
            if abs(p_new - p) <= tol * max(p, 1.0):
                return float(p_new)
            p = p_new
        raise ValueError(f"the material balance did not converge for {gip:g} m3 after {max_iters} iterations")

    def mean_pressure(self, p: np.ndarray) -> float:
        """Pore-volume-weighted average pressure [Pa]."""
        phi = self.rock.porosity(self.grid.phi0, p)
        pv = self.grid.volumes * phi
        return float(np.sum(pv * p) / np.sum(pv))

    # ------------------------------------------------------------------ assembly
    def _assemble(self, p: np.ndarray, n_old: np.ndarray, dt: float, controls: dict[str, Control],
                  jacobian: bool = True):
        g = self.grid
        V = g.volumes
        n, dn, lam, dlam = self._state(p)
        R = V * (n - n_old) / dt
        diag = V * dn / dt

        a, b, T = g.conn_a, g.conn_b, g.conn_T
        dp = p[b] - p[a]
        up_b = dp > 0.0
        lam_f = np.where(up_b, lam[b], lam[a])
        flux = T * lam_f * dp                       # into cell a, out of cell b
        # bincount, not R[a] -= flux: a cell meets several connections and fancy-index assignment would
        # silently keep only one of the contributions
        nc = g.n_cells
        R += np.bincount(a, weights=-flux, minlength=nc)
        R += np.bincount(b, weights=flux, minlength=nc)
        rows = [a, a, b, b]
        cols = [a, b, a, b]
        if jacobian:
            dq_dpb = np.where(up_b, T * (dlam[b] * dp + lam[b]), T * lam[a])
            dq_dpa = np.where(up_b, -T * lam[b], T * (dlam[a] * dp - lam[a]))
            vals = [-dq_dpa, -dq_dpb, dq_dpa, dq_dpb]
        else:
            vals = None

        # boundaries held at a fixed pressure
        if g.boundary_cells.size:
            c = g.boundary_cells
            pb = g.boundary_p
            Tb = g.boundary_T
            out = pb < p[c]
            lam_cell = lam[c]
            lam_bnd = (self.fluid.index(pb) / self.fluid.mu(pb))
            lam_fb = np.where(out, lam_cell, lam_bnd)
            q_in = Tb * lam_fb * (pb - p[c])
            R[c] -= q_in
            dq_dp = np.where(out, Tb * (dlam[c] * (pb - p[c]) - lam_cell), -Tb * lam_bnd)
            diag[c] -= dq_dp

        # wells
        well_terms = []
        for w in self.wells:
            ctrl = controls.get(w.name)
            c = int(w.cell)
            q, bhp, dq = operating_point(w, ctrl, float(p[c]), float(lam[c]), float(dlam[c]),
                                         float(self.wi[w.name]))
            well_terms.append((w, ctrl, q, bhp, dq))
            if q != 0.0:
                R[c] -= q
                diag[c] -= dq

        if not jacobian:
            return R, None, well_terms

        rows = np.concatenate([rows[0], rows[1], rows[2], rows[3], np.arange(g.n_cells)])
        cols = np.concatenate([cols[0], cols[1], cols[2], cols[3], np.arange(g.n_cells)])
        vals = np.concatenate([vals[0], vals[1], vals[2], vals[3], diag])
        J = coo_matrix((vals, (rows, cols)), shape=(g.n_cells, g.n_cells)).tocsr()
        return R, J, well_terms

    # ------------------------------------------------------------------ newton
    def _solve_step(self, p: np.ndarray, n_old: np.ndarray, dt: float, controls: dict[str, Control],
                    config: SimConfig) -> tuple[np.ndarray, bool, int, float, str]:
        p = p.copy()
        scale = max(float(np.max(p)), 1.0)
        tol = config.tol_rel * scale
        # a residual at the level of the round-off of the accumulation term is converged, whatever the update
        # says: below it the linear solve is solving for noise
        res_scale = float(np.max(self.grid.volumes * np.abs(n_old) / dt))
        res_tol = 1e-12 * max(res_scale, 1e-30)
        last_update = np.inf
        r0 = 0.0
        for it in range(1, config.max_newton + 1):
            R, J, _ = self._assemble(p, n_old, dt, controls, jacobian=True)
            r0 = float(np.max(np.abs(R))) if R.size else 0.0
            if r0 <= res_tol:
                return p, True, it - 1, r0, ""
            dp = spsolve(J.tocsc(), -R)
            if not np.all(np.isfinite(dp)):
                return p, False, it, r0, f"the linear solve returned a non-finite update at iteration {it}"
            damp = 1.0
            accepted = False
            hit_floor = False
            for _ in range(config.max_backtrack):
                trial = p + damp * dp
                if np.all(trial > P_MIN):
                    R_trial, _, _ = self._assemble(trial, n_old, dt, controls, jacobian=False)
                    if float(np.max(np.abs(R_trial))) <= (1.0 - 1e-4 * damp) * r0:
                        p = trial
                        last_update = float(np.max(np.abs(damp * dp)))
                        accepted = True
                        break
                else:
                    hit_floor = True
                damp *= 0.5
            if not accepted:
                if hit_floor:
                    return p, False, it, r0, (
                        f"the Newton step needs a pressure below {P_MIN:g} Pa at iteration {it} "
                        f"(max |R| = {r0:.3e} m3/s): the rate asked for is beyond what the grid holds")
                return p, False, it, r0, (
                    f"Newton could not decrease the residual at iteration {it} "
                    f"(max |R| = {r0:.3e} m3/s, floor {res_tol:.3e})")
            if last_update <= tol:
                return p, True, it, r0, ""
        return p, False, config.max_newton, r0, (
            f"no convergence in {config.max_newton} Newton iterations "
            f"(last update {last_update:.3e} Pa, tolerance {tol:.3e} Pa)")

    # ------------------------------------------------------------------ run
    def run(self, schedule: Schedule, config: SimConfig | None = None) -> SimulationResult:
        """Run ``schedule`` and return the fields, the well histories and the checks."""
        config = config or SimConfig()
        steps = schedule.steps(config.dt_days)
        p = self.p0.copy()
        n_old = self._state(p)[0].copy()
        gip0 = float(np.sum(self.grid.volumes * n_old))
        if gip0 <= 0.0:
            raise ValueError("the initial state holds no gas: check porosity and initial pressure")

        times = [0.0]
        fields = [p.copy()]
        means = [self.mean_pressure(p)]
        gips = [gip0]
        rates = {w.name: [0.0] for w in self.wells}
        bhps = {w.name: [float(p[w.cell])] for w in self.wells}
        cums = {w.name: [0.0] for w in self.wells}
        newton_iters: list[int] = []
        residuals: list[float] = []
        cumulative = {w.name: 0.0 for w in self.wells}
        converged = True
        reason = ""
        t = 0.0
        for dt_days, period_index in steps:
            controls = schedule.periods[period_index].controls
            dt = dt_days * SECONDS_PER_DAY
            p_new, ok, iters, resid, why = self._solve_step(p, n_old, dt, controls, config)
            newton_iters.append(iters)
            residuals.append(resid)
            if not ok:
                converged = False
                reason = f"at t = {t + dt_days:.4f} d: {why}"
                break
            _, _, well_terms = self._assemble(p_new, n_old, dt, controls, jacobian=False)
            p = p_new
            n_old = self._state(p)[0].copy()
            t += dt_days
            times.append(t)
            fields.append(p.copy())
            means.append(self.mean_pressure(p))
            gips.append(float(np.sum(self.grid.volumes * n_old)))
            for w, _ctrl, q, bhp, _dq in well_terms:
                rates[w.name].append(q * SECONDS_PER_DAY)
                bhps[w.name].append(bhp)
                cumulative[w.name] += q * dt
                cums[w.name].append(cumulative[w.name])

        gas_in_place = np.array(gips)
        net_injected = float(sum(c[-1] for c in cums.values()))
        balance = (gas_in_place[-1] - gas_in_place[0] - net_injected) / gas_in_place[0]
        return SimulationResult(
            times_days=np.array(times),
            pressure=np.array(fields),
            mean_pressure=np.array(means),
            gas_in_place=gas_in_place,
            well_rate={k: np.array(v) for k, v in rates.items()},
            well_bhp={k: np.array(v) for k, v in bhps.items()},
            well_cumulative={k: np.array(v) for k, v in cums.items()},
            mass_balance_error=float(balance),
            converged=converged,
            reason=reason,
            newton_iterations=newton_iters,
            residuals=residuals,
        )
