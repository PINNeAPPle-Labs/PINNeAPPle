"""Interior permanent-magnet synchronous machine: 2-D nonlinear A_z FEA and its post-processing.

    from pinneapple_simulation.electromagnetics.machines import IPMMachine, prius_2004
    m = IPMMachine(prius_2004(), n_gap=960)
    op = m.operating_point(I_peak=250.0, gamma=math.radians(50))  # torque, psi_d, psi_q over rotor positions
    T, psi_d, psi_q = op["torque_mean"], op["psi_d"], op["psi_q"]

Conventions. Rotor angle ``theta`` counter-clockwise (mechanical radians), torque on the rotor counter-clockwise
positive. The d-axis is the axis of the magnet flux linkage (found from a no-load solution,
:meth:`IPMMachine.pm_flux_angle`); the amplitude-invariant Park transform ``x_d + j x_q = (2/3)(x_a + a x_b +
a^2 x_c) e^{-j theta_e}`` with ``theta_e = p theta + phi0`` gives ``psi_d``, ``psi_q``, and phase currents
``i_k = I cos(theta_e + pi/2 + gamma - 2 pi k / 3)`` put ``i_d = -I sin(gamma)``, ``i_q = I cos(gamma)``
(``gamma = 0`` on the q-axis, ``gamma > 0`` towards -d; pyleecan's current angle ``Phi0`` is ``gamma + 90 deg``).
Electromagnetic torque from the flux linkages: ``T = 1.5 p (psi_d i_q - psi_q i_d)``; from the field:
Arkkio's Maxwell-stress torque averaged over the rotor-side and stator-side air layers.

Materials: rotor and stator iron follow a nonlinear B-H curve (stacking factor applied), magnets are linear
(``mu_r``, ``Br`` at a temperature), pockets, slot openings and slots are air for the field. The shaft is outside
the model (A = 0 on the rotor's inner circle and on the stator's outer circle: no flux leaves the
lamination there).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..fem import Geometry, solve_magnetostatic
from ..machine import MachineSolution, belt_winding
from ..materials import NU0
from ..postprocess import (
    CoilSide,
    element_flux_density_history,
    flux_linkage,
    iron_loss,
    iron_loss_components,
    magnet_eddy_loss_resistance_limited,
    torque_maxwell_stress,
)
from .geometry import MachineGeometry
from .mesher import MachineMesh
from .steel import M400_50A, MAGNET_PRIUS, MagnetGrade, SteelGrade, TableIronLoss

__all__ = ["IPMMachine", "Winding"]

TWO_PI = 2.0 * math.pi
_A = np.exp(2j * math.pi / 3)


@dataclass
class Winding:
    """Phase, orientation and turns of the coil side in each slot (single layer), series turns per phase,
    parallel paths. ``layout``: (phase, sign) per slot, default a 60-degree phase belt."""

    turns_per_slot: float = 9.0
    parallel_paths: int = 1
    layout: list | None = None


@dataclass
class IPMMachine:
    geometry: MachineGeometry
    steel: SteelGrade = M400_50A
    magnet: MagnetGrade = MAGNET_PRIUS
    magnet_temperature: float = 20.0
    winding: Winding = field(default_factory=Winding)
    n_gap: int = 960
    h_max: float | None = None
    grading: float = 0.2
    tol: float = 1e-8

    def __post_init__(self):
        g = self.geometry
        self.pole_pairs = g.rotor.poles // 2
        self.length = g.length
        self.mm = MachineMesh(g, n_gap=self.n_gap, h_max=self.h_max, grading=self.grading)
        if self.winding.layout is None:
            self.winding.layout = belt_winding(g.stator.slots, self.pole_pairs)
        self.curve = self.steel.curve(g.stacking_factor)
        self._phi0 = None
        names = self.mm.region_names
        reg = self.mm.region
        self._iron = np.isin(reg, [names.index("rotor_iron"), names.index("stator_iron")])
        self._magnet = ~np.isnan(self.mm.magnet_angle)
        self._rotor_iron = reg == names.index("rotor_iron")
        self._stator_iron = reg == names.index("stator_iron")

    # ------------------------------------------------------------------ field solution
    @property
    def Br(self) -> float:
        return self.magnet.Br(self.magnet_temperature)

    def coil_sides(self) -> list[CoilSide]:
        return [CoilSide(f"slot{k}", ph, self.winding.turns_per_slot, sg)
                for k, (ph, sg) in enumerate(self.winding.layout)]

    def _sources(self, mesh, g, theta, currents, magnets: bool = True):
        J = np.zeros(mesh.n_elements)
        for cs in self.coil_sides():
            m = mesh.mask(cs.region)
            J[m] = cs.sign * cs.turns * currents[cs.phase] / (self.winding.parallel_paths * g.area[m].sum())
        Br = np.zeros((mesh.n_elements, 2))
        if magnets:
            a = self.mm.magnet_angle[self._magnet] + theta
            Br[self._magnet, 0] = self.Br * np.cos(a)
            Br[self._magnet, 1] = self.Br * np.sin(a)
        return J, Br

    def solve(self, theta: float = 0.0, currents: Sequence[float] = (0.0, 0.0, 0.0), A0=None,
              nu_frozen: np.ndarray | None = None, magnets: bool = True) -> MachineSolution:
        """Nonlinear magnetostatic solution at rotor angle ``theta`` with instantaneous phase ``currents`` (A).
        ``nu_frozen``: element reluctivities of an earlier solution, which make the problem linear (frozen
        permeability); ``magnets=False`` removes the magnet sources."""
        mesh = self.mm.mesh(theta)
        g = Geometry.of(mesh)
        J, Br = self._sources(mesh, g, theta, currents, magnets)
        if nu_frozen is not None:
            sol = solve_magnetostatic(mesh, nu_frozen, J=J, Br=Br, geometry=g)
        else:
            nu = np.full(mesh.n_elements, NU0)
            nu[self._magnet] = NU0 / self.magnet.mu_r
            sol = solve_magnetostatic(mesh, nu, J=J, Br=Br, nonlinear=[(self._iron, self.curve)], geometry=g,
                                      A0=A0, tol=self.tol)
        return MachineSolution(theta, tuple(currents), mesh, sol, J)

    def torque(self, ms: MachineSolution) -> float:
        """Arkkio torque (N m), length-weighted mean over the rotor-side and stator-side air layers."""
        num = den = 0.0
        for r1, r2, el in self.mm.annuli:
            num += torque_maxwell_stress(ms.mesh, ms.B, r1, r2, self.length, elements=el) * (r2 - r1)
            den += r2 - r1
        return num / den

    def coenergy(self, theta: float, currents) -> float:
        return self.length * self.solve(theta, currents).sol.coenergy

    def torque_coenergy(self, theta: float, currents, steps: int = 1) -> float:
        """Virtual-work torque dW'/dtheta at constant currents, fourth-order differences at multiples of the
        band step (identical band triangles at those angles)."""
        h = steps * self.mm.band_step
        W = [self.coenergy(theta + k * h, currents) for k in (-2, -1, 1, 2)]
        return (W[0] - 8 * W[1] + 8 * W[2] - W[3]) / (12 * h)

    def flux_linkage(self, ms: MachineSolution) -> np.ndarray:
        return flux_linkage(ms.mesh, ms.sol.geometry, ms.A, self.coil_sides(), 3, self.length,
                            self.winding.parallel_paths)

    # ------------------------------------------------------------------ dq frame
    def pm_flux_angle(self) -> float:
        """Electrical angle ``phi0`` of the d-axis (pole 0, on +x at ``theta = 0``) in the frame of the winding.

        The fundamental of the magnet flux-linkage vector of a rotor symmetric about its pole axis lies at
        ``arg(conj(W_A)) + 90 deg`` with ``W_A = sum_s c_s exp(j p phi_s)`` the winding phasor of phase A (conductors
        ``c_s`` per slot at ``phi_s``); its sign (``+ 180 deg``) is taken from one no-load solution. A single solution's
        flux vector would carry the slot harmonics: on the Prius it is 0.79 electrical degrees off the fundamental,
        which moves the torque at pure negative d-axis current by about 12 N m."""
        if self._phi0 is None:
            psi = self.flux_linkage(self.solve(0.0))
            snap = float(np.angle((2 / 3) * (psi[0] + _A * psi[1] + _A * _A * psi[2])))
            S = self.geometry.stator.slots
            W = sum(cs.sign * cs.turns * np.exp(1j * self.pole_pairs * TWO_PI * k / S)
                    for k, cs in enumerate(self.coil_sides()) if cs.phase == 0)
            geo = float(np.angle(np.conj(W))) + 0.5 * math.pi
            if math.cos(geo - snap) < 0:
                geo += math.pi
            self._phi0 = float(np.angle(np.exp(1j * geo)))
        return self._phi0

    def theta_e(self, theta: float) -> float:
        return self.pole_pairs * theta + self.pm_flux_angle()

    def park(self, x_abc, theta: float) -> tuple[float, float]:
        v = (2 / 3) * (x_abc[0] + _A * x_abc[1] + _A * _A * x_abc[2]) * np.exp(-1j * self.theta_e(theta))
        return float(v.real), float(v.imag)

    def currents_dq(self, theta: float, i_d: float, i_q: float) -> tuple:
        v = complex(i_d, i_q) * np.exp(1j * self.theta_e(theta))
        return tuple(float((v * np.conj(_A ** k)).real) for k in range(3))

    def phase_currents(self, theta: float, I_peak: float, gamma: float) -> tuple:
        return self.currents_dq(theta, -I_peak * math.sin(gamma), I_peak * math.cos(gamma))

    @property
    def load_period(self) -> float:
        """Mechanical period of the torque and of the dq flux linkages under sinusoidal currents: 60 electrical
        degrees (a shift of one phase belt) for a three-phase winding with an integer number of slots per pole
        and phase. Use a full electrical period (``2 pi / p``) for other windings."""
        return TWO_PI / (6 * self.pole_pairs)

    @property
    def cogging_period(self) -> float:
        return TWO_PI / math.lcm(self.geometry.stator.slots, self.geometry.rotor.poles)

    def positions(self, n: int, period: float | None = None, theta0: float = 0.0) -> np.ndarray:
        """``n`` rotor angles over ``period`` (default :attr:`load_period`), each a multiple of the band step."""
        period = period or self.load_period
        k = np.round((theta0 + period * np.arange(n) / n) / self.mm.band_step)
        return k * self.mm.band_step

    def operating_point(self, i_d: float | None = None, i_q: float | None = None, I_peak: float | None = None,
                        gamma: float | None = None, n_positions: int = 6, period: float | None = None,
                        keep: bool = False) -> dict:
        """Solve over ``n_positions`` rotor angles with sinusoidal currents fixed in the dq frame.

        Returns torque (Arkkio) per position and its mean, ripple (peak to peak / mean), psi_d, psi_q
        (means), the dq torque 1.5 p (psi_d i_q - psi_q i_d), and with ``keep`` the solutions."""
        if I_peak is not None:
            i_d, i_q = -I_peak * math.sin(gamma), I_peak * math.cos(gamma)
        thetas = self.positions(n_positions, period)
        T, pd, pq, sols = [], [], [], []
        A0 = None
        for th in thetas:
            ms = self.solve(th, self.currents_dq(th, i_d, i_q), A0=A0)
            A0 = ms.A
            T.append(self.torque(ms))
            d, q = self.park(self.flux_linkage(ms), th)
            pd.append(d)
            pq.append(q)
            if keep:
                sols.append(ms)
        T, pd, pq = np.array(T), np.array(pd), np.array(pq)
        out = {"i_d": i_d, "i_q": i_q, "thetas": thetas, "torque": T, "torque_mean": float(T.mean()),
               "ripple": float((T.max() - T.min()) / abs(T.mean())) if T.mean() else float("nan"),
               "psi_d": float(pd.mean()), "psi_q": float(pq.mean()), "psi_d_all": pd, "psi_q_all": pq,
               "torque_dq": float(1.5 * self.pole_pairs * (pd.mean() * i_q - pq.mean() * i_d))}
        if keep:
            out["solutions"] = sols
        return out

    def inductances(self, i_d: float, i_q: float, theta: float = 0.0) -> dict:
        """Frozen-permeability flux split at one operating point: the permeability of the nonlinear solution is
        frozen, so the field is the sum of a magnet part and a current part (exactly, the problem being linear
        with the frozen reluctivities). Returns ``psi_pm`` (magnet flux on the d-axis with the frozen
        permeability), apparent ``L_d = psi_d,i / i_d``, ``L_q = psi_q,i / i_q`` from the current part, and the
        cross terms of unit d and q currents."""
        cur = self.currents_dq(theta, i_d, i_q)
        ms = self.solve(theta, cur)
        nu = ms.sol.nu
        pm = self.park(self.flux_linkage(self.solve(theta, (0.0, 0.0, 0.0), nu_frozen=nu)), theta)
        cu = self.park(self.flux_linkage(self.solve(theta, cur, nu_frozen=nu, magnets=False)), theta)
        ud = self.park(self.flux_linkage(self.solve(theta, self.currents_dq(theta, 1.0, 0.0), nu_frozen=nu,
                                                    magnets=False)), theta)
        uq = self.park(self.flux_linkage(self.solve(theta, self.currents_dq(theta, 0.0, 1.0), nu_frozen=nu,
                                                    magnets=False)), theta)
        total = self.park(self.flux_linkage(ms), theta)
        return {"psi_d": total[0], "psi_q": total[1], "psi_pm_d": pm[0], "psi_pm_q": pm[1],
                "psi_i_d": cu[0], "psi_i_q": cu[1], "L_dd": ud[0], "L_qd": ud[1], "L_dq": uq[0], "L_qq": uq[1],
                "superposition_error": float(np.hypot(total[0] - pm[0] - cu[0], total[1] - pm[1] - cu[1])
                                             / np.hypot(*total))}

    def mtpa(self, I_peak: float, n_positions: int = 4, bracket: tuple = (0.0, 1.4), tol: float = 0.5e-2):
        """Current angle gamma (rad) of maximum mean torque at current amplitude ``I_peak`` (golden section
        on the FEA torque averaged over ``n_positions`` rotor angles, within ``bracket``)."""
        f = lambda gm: -self.operating_point(I_peak=I_peak, gamma=gm, n_positions=n_positions)["torque_mean"]  # noqa: E731
        a, b = bracket
        r = (math.sqrt(5) - 1) / 2
        c, d = b - r * (b - a), a + r * (b - a)
        fc, fd = f(c), f(d)
        while b - a > tol:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - r * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + r * (b - a)
                fd = f(d)
        g = 0.5 * (a + b)
        return g, -f(g)

    # ------------------------------------------------------------------ no load
    def no_load(self, n: int = 24, n_flux: int | None = None) -> dict:
        """Cogging torque at ``n`` angles over one cogging period and phase flux linkages at ``n_flux`` (default ``n``)
        angles over one electrical period (for a back-EMF waveform, >= 120: the EMF carries the slot harmonics)."""
        th_c = self.positions(n, period=self.cogging_period)
        th_e = self.positions(n_flux or n, period=TWO_PI / self.pole_pairs)
        cog = []
        A0 = None
        for th in th_c:
            ms = self.solve(th, A0=A0)
            A0 = ms.A
            cog.append(self.torque(ms))
        psi = []
        A0 = None
        for th in th_e:
            ms = self.solve(th, A0=A0)
            A0 = ms.A
            psi.append(self.flux_linkage(ms))
        return {"theta_cogging": th_c, "cogging": np.array(cog), "theta": th_e, "psi": np.array(psi)}

    def back_emf(self, psi: np.ndarray, theta: np.ndarray, rpm: float) -> dict:
        """Phase back-EMF (V) from flux linkages over one electrical period: ``e = omega dpsi/dtheta``
        (spectral derivative); also the fundamental and line-to-line rms values."""
        from ..postprocess import back_emf

        e = back_emf(psi, theta, rpm * TWO_PI / 60.0)
        F = np.fft.rfft(e[:, 0]) / len(e)
        amp = 2 * np.abs(F)
        e1 = amp[1]
        ell = e[:, 0] - e[:, 1]
        Fl = np.fft.rfft(ell) / len(ell)
        return {"e": e, "fundamental_peak": float(e1), "phase_rms": float(np.sqrt(np.mean(e[:, 0] ** 2))),
                "line_rms": float(np.sqrt(np.mean(ell ** 2))), "line_fundamental_rms": float(2 * abs(Fl[1]) / math.sqrt(2)),
                "harmonics_phase": amp[1:].tolist(),
                "thd": float(np.sqrt(np.sum(amp[2:] ** 2)) / e1),
                "thd_line": float(np.sqrt(np.sum(np.abs(2 * Fl[2:]) ** 2)) / (2 * abs(Fl[1])))}

    # ------------------------------------------------------------------ losses
    def _period(self, i_d: float, i_q: float, steps: int):
        """Solutions over one electrical period with fixed dq currents, and B in each element's own frame."""
        thetas = self.positions(steps, period=TWO_PI / self.pole_pairs)
        sols, A0 = [], None
        for th in thetas:
            ms = self.solve(th, self.currents_dq(th, i_d, i_q), A0=A0)
            A0 = ms.A
            sols.append(ms)
        rot = np.zeros(self.mm.n_elements, bool)
        rot[self.mm.rotor_elements] = True
        return thetas, sols, element_flux_density_history([s.B for s in sols], thetas, rot)

    def _magnet_loss(self, sols, f: float) -> float:
        # A in the rotor frame is A at the rotor nodes (they turn with the rotor)
        A_hist = np.array([s.A for s in sols])
        pieces = [self.mm.region == self.mm.region_names.index(f"magnet{k}") for k in range(self.geometry.rotor.poles)]
        return magnet_eddy_loss_resistance_limited(sols[0].mesh, sols[0].sol.geometry, A_hist, 1.0 / f,
                                                   1.0 / self.magnet.resistivity, pieces, self.length)

    def losses(self, rpm: float, i_d: float, i_q: float, steps: int = 48, loss_model=None,
               n_harmonics: int | None = None) -> dict:
        """Iron loss (stator, rotor) from the flux density over one electrical period (each element in its own
        frame; ``loss_model`` default: the catalogue table of the steel) and resistance-limited magnet eddy
        loss (each magnet an insulated piece; segment them with ``VRotor.magnet_segments``)."""
        model = loss_model or TableIronLoss(self.steel)
        thetas, sols, Bh = self._period(i_d, i_q, steps)
        f = rpm / 60.0 * self.pole_pairs
        area = sols[0].sol.geometry.area * self.geometry.stacking_factor
        Pst, dst = iron_loss(Bh, f, model, area, self._stator_iron, self.length, n_harmonics)
        Prt, drt = iron_loss(Bh, f, model, area, self._rotor_iron, self.length, n_harmonics)
        T = np.array([self.torque(s) for s in sols])
        return {"frequency_Hz": f, "stator_iron_W": Pst, "rotor_iron_W": Prt, "magnet_W": self._magnet_loss(sols, f),
                "torque_mean": float(T.mean()), "torque": T, "thetas": thetas,
                "stator_loss_density": dst, "rotor_loss_density": drt}

    def loss_components(self, i_d: float, i_q: float, alpha: float, beta: float, steps: int = 24,
                        n_harmonics: int | None = None) -> dict:
        """Speed-free loss data of one operating point: iron-loss sums ``S = (S_h, S_c, S_e)`` of stator and rotor
        (:func:`~..postprocess.iron_loss_components`; loss ``k_h f^alpha S_h + k_c f^2 S_c + k_e f^1.5 S_e``) and
        the magnet loss coefficient ``P_m / f^2`` (resistance limited), with the mean torque and dq flux."""
        thetas, sols, Bh = self._period(i_d, i_q, steps)
        area = sols[0].sol.geometry.area * self.geometry.stacking_factor
        rho = self.steel.density
        out = {"stator": iron_loss_components(Bh, area, self._stator_iron, alpha, beta, rho, self.length, n_harmonics),
               "rotor": iron_loss_components(Bh, area, self._rotor_iron, alpha, beta, rho, self.length, n_harmonics),
               "magnet_per_f2": self._magnet_loss(sols, 1.0)}
        T = np.array([self.torque(s) for s in sols])
        dq = np.array([self.park(self.flux_linkage(s), th) for s, th in zip(sols, thetas, strict=True)])
        out.update(torque_mean=float(T.mean()), psi_d=float(dq[:, 0].mean()), psi_q=float(dq[:, 1].mean()))
        return out
