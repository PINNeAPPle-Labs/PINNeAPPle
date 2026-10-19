"""Drive model of a synchronous machine: dq steady state, MTPA and flux weakening under inverter limits,
efficiency map and drive-cycle energy.

The machine enters as a *flux map*, any object with ``psi(i_d, i_q) -> (psi_d, psi_q)``, ``torque(i_d, i_q)`` and
``core_loss(i_d, i_q, f_e)`` (vectorised over arrays):

* :class:`LinearDQ` -- constant ``L_d``, ``L_q``, ``psi_m``: ``T = 1.5 p (psi_m i_q + (L_d - L_q) i_d i_q)``; its MTPA,
  base speed and flux-weakening torque have closed forms (:meth:`LinearDQ.mtpa_closed_form`,
  :meth:`LinearDQ.max_torque_closed_form`) that verify the numerical optimiser.
* :class:`FluxMap` -- tables of the FEA (:meth:`FluxMap.from_fea` on an :class:`~.pmsm.IPMMachine`): mean torque and dq
  flux linkages on an ``(i_d, i_q)`` grid (saturation and cross-saturation included), and per grid point the
  frequency-free iron-loss sums and the magnet-loss coefficient (:meth:`~.pmsm.IPMMachine.loss_components`), so the
  core loss at any speed is ``k_h f^alpha S_h + k_c f^2 S_c + k_e f^1.5 S_e + c_m f^2``.

Steady state (amplitude-invariant dq, peak values): ``v_d = R i_d - w_e psi_q``, ``v_q = R i_q + w_e psi_d``, phase
current amplitude ``|i| <= I_max``, phase voltage amplitude ``|v| <= V_max`` (``V_dc / sqrt 3`` for space-vector PWM
in its linear range, :func:`v_max_svpwm`). :class:`Drive` picks for a requested torque and speed the feasible current
of least total loss (copper ``1.5 R |i|^2`` plus core loss); without the voltage limit this is the
maximum-torque-per-ampere point when core loss is ignored, and at high speed it is the flux-weakening point.

:class:`Vehicle` turns a speed trace into wheel torque and speed (road load ``m g C_rr + 0.5 rho C_d A v^2``,
inertia, single gear with efficiency); :func:`cycle_energy` integrates the battery-side energy of a drive over it,
with regeneration up to the drive's torque envelope. :func:`wltc_class3b` loads the WLTC class 3b speed trace
(UN GTR No. 15, Annex 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["LinearDQ", "FluxMap", "Drive", "Vehicle", "cycle_energy", "wltc_class3b", "v_max_svpwm",
           "copper_resistance", "prius_phase_resistance"]

TWO_PI = 2.0 * math.pi
RHO_CU_20 = 1.68e-8  # Ohm m, annealed copper at 20 degC (IEC 60028: 1/58 Ohm mm^2/m = 1.724e-8 for standard)
ALPHA_CU = 3.93e-3   # 1/K


def v_max_svpwm(v_dc: float) -> float:
    """Largest phase-voltage amplitude of space-vector PWM in its linear range: ``V_dc / sqrt(3)``."""
    return v_dc / math.sqrt(3.0)


def copper_resistance(series_turns: float, conductor_area: float, stack_length: float, end_length: float,
                      temperature: float = 20.0, rho20: float = 1.724e-8, alpha: float = ALPHA_CU,
                      parallel_paths: int = 1) -> float:
    """DC phase resistance: ``rho(T) N 2 (L + l_end) / (A a^2)`` with ``N`` series turns per phase, ``A`` the copper
    area of one conductor, ``l_end`` the end-turn length per side, ``a`` parallel paths (IEC 60028 standard copper
    ``1.724e-8 Ohm m`` at 20 degC by default)."""
    rho = rho20 * (1.0 + alpha * (temperature - 20.0))
    return rho * series_turns * 2.0 * (stack_length + end_length) / (conductor_area * parallel_paths ** 2)


def prius_phase_resistance(temperature: float = 100.0) -> tuple[float, dict]:
    """Phase resistance of the 2004 Prius motor from the winding data of pyleecan's ``Toyota_Prius.json``: 72 series
    turns, 13 strands of 0.912 mm per conductor, 83.82 mm stack. The end-turn length per side is an estimate: a
    half circle over the coil pitch (6 slots at the mid-slot radius) plus twice the axial overhang (19.4 mm)."""
    from .geometry import prius_2004

    g = prius_2004()
    r_mid = g.stator.r_in + 0.5 * (g.stator.H0 + g.stator.H2)
    pitch = 6 * TWO_PI * r_mid / g.stator.slots
    end = 0.5 * math.pi * pitch + 2 * 0.019366
    area = 13 * math.pi * (0.912e-3 / 2) ** 2
    R = copper_resistance(72, area, g.length, end, temperature)
    return R, {"series_turns": 72, "conductor_area_mm2": area * 1e6, "end_turn_per_side_m": end,
               "temperature_C": temperature, "R_phase_ohm": R}


# --------------------------------------------------------------------------------------------- flux maps
@dataclass
class LinearDQ:
    """Machine with constant inductances: ``psi_d = psi_m + L_d i_d``, ``psi_q = L_q i_q``."""

    pole_pairs: int
    psi_m: float
    L_d: float
    L_q: float
    core_loss_coeffs: tuple = (0.0, 1.0, 0.0)  # (c1, alpha, c2): P = c1 f^alpha + c2 f^2, independent of current

    def psi(self, i_d, i_q):
        return self.psi_m + self.L_d * np.asarray(i_d, float), self.L_q * np.asarray(i_q, float)

    def torque(self, i_d, i_q):
        d, q = self.psi(i_d, i_q)
        return 1.5 * self.pole_pairs * (d * i_q - q * i_d)

    def core_loss(self, i_d, i_q, f_e):
        c1, a, c2 = self.core_loss_coeffs
        f = np.abs(np.asarray(f_e, float))
        return np.broadcast_to(c1 * f ** a + c2 * f ** 2, np.broadcast(np.asarray(i_d), np.asarray(i_q), f).shape)

    def mtpa_closed_form(self, current: float) -> tuple[float, float]:
        """MTPA currents at amplitude ``current``: ``i_d = (psi_m - sqrt(psi_m^2 + 8 (L_q - L_d)^2 I^2)) / (4 (L_q - L_d))``."""
        dL = self.L_q - self.L_d
        if abs(dL) < 1e-15:
            return 0.0, current
        i_d = (self.psi_m - math.sqrt(self.psi_m ** 2 + 8.0 * dL ** 2 * current ** 2)) / (4.0 * dL)
        return i_d, math.sqrt(max(current ** 2 - i_d ** 2, 0.0))

    def max_torque_closed_form(self, w_e: float, I_max: float, V_max: float) -> tuple[float, float, float]:
        """Largest torque at electrical speed ``w_e`` with ``R = 0`` (current circle and voltage ellipse):
        the MTPA point below base speed, the circle-ellipse intersection above it (valid up to the
        maximum-torque-per-volt region, i.e. while ``psi_m > L_d I_max``)."""
        i_d, i_q = self.mtpa_closed_form(I_max)
        if w_e * math.hypot(self.L_q * i_q, self.psi_m + self.L_d * i_d) <= V_max:
            return float(self.torque(i_d, i_q)), i_d, i_q
        a = self.L_d ** 2 - self.L_q ** 2
        b = 2.0 * self.psi_m * self.L_d
        c = self.psi_m ** 2 + self.L_q ** 2 * I_max ** 2 - (V_max / w_e) ** 2
        if abs(a) < 1e-30:
            i_d = -c / b
        else:
            disc = b * b - 4 * a * c
            roots = [(-b + sg * math.sqrt(max(disc, 0.0))) / (2 * a) for sg in (1, -1)]
            roots = [r for r in roots if -I_max - 1e-12 <= r <= 0.0]
            i_d = max(roots)
        i_q = math.sqrt(max(I_max ** 2 - i_d ** 2, 0.0))
        return float(self.torque(i_d, i_q)), i_d, i_q


@dataclass
class FluxMap:
    """Tabulated machine from the FEA. Grids ``i_d`` (ascending, <= 0) and ``i_q`` (ascending, >= 0); the
    machine's symmetry extends the maps to negative ``i_q`` (``psi_d`` even, ``psi_q`` and torque odd)."""

    pole_pairs: int
    i_d: np.ndarray
    i_q: np.ndarray
    psi_d: np.ndarray       # (n_d, n_q)
    psi_q: np.ndarray
    T: np.ndarray
    loss_i_d: np.ndarray | None = None
    loss_i_q: np.ndarray | None = None
    S_iron: np.ndarray | None = None      # (n_d', n_q', 3): stator + rotor sums (S_h, S_c, S_e)
    magnet_per_f2: np.ndarray | None = None  # (n_d', n_q')
    loss_coeffs: tuple = (0.0, 1.0, 2.0, 0.0, 0.0)  # (k_h, alpha, beta, k_c, k_e)
    _interp: dict = field(default_factory=dict, repr=False)

    def _f(self, name, table, gd, gq):
        if name not in self._interp:
            from scipy.interpolate import RegularGridInterpolator

            method = "cubic" if min(len(gd), len(gq)) >= 4 else "linear"
            self._interp[name] = RegularGridInterpolator((gd, gq), table, method=method, bounds_error=False,
                                                         fill_value=None)
        return self._interp[name]

    def _eval(self, name, table, i_d, i_q, odd: bool, gd=None, gq=None):
        gd = self.i_d if gd is None else gd
        gq = self.i_q if gq is None else gq
        i_d, i_q = np.broadcast_arrays(np.asarray(i_d, float), np.asarray(i_q, float))
        v = self._f(name, table, gd, gq)(np.stack([i_d.ravel(), np.abs(i_q).ravel()], -1)).reshape(i_d.shape)
        return np.where(i_q < 0, -v, v) if odd else v

    def psi(self, i_d, i_q):
        return (self._eval("psi_d", self.psi_d, i_d, i_q, False), self._eval("psi_q", self.psi_q, i_d, i_q, True))

    def torque(self, i_d, i_q):
        return self._eval("T", self.T, i_d, i_q, True)

    def core_loss_parts(self, i_d, i_q, f_e) -> tuple[np.ndarray, np.ndarray]:
        """Iron loss (stator + rotor) and magnet eddy loss (W) at electrical frequency ``f_e``."""
        if self.S_iron is None:
            z = np.zeros(np.broadcast(np.asarray(i_d), np.asarray(i_q), np.asarray(f_e)).shape)
            return z, z
        k_h, alpha, _, k_c, k_e = self.loss_coeffs
        f = np.abs(np.asarray(f_e, float))
        S = [self._eval(f"S{k}", self.S_iron[..., k], i_d, i_q, False, self.loss_i_d, self.loss_i_q) for k in range(3)]
        Pm = self._eval("Pm", self.magnet_per_f2, i_d, i_q, False, self.loss_i_d, self.loss_i_q)
        S = [np.maximum(s, 0.0) for s in S]
        return (k_h * f ** alpha * S[0] + k_c * f ** 2 * S[1] + k_e * f ** 1.5 * S[2],
                np.maximum(Pm, 0.0) * f ** 2)

    def core_loss(self, i_d, i_q, f_e):
        iron, magnet = self.core_loss_parts(i_d, i_q, f_e)
        return iron + magnet

    @classmethod
    def from_fea(cls, machine, I_max: float, n_d: int = 9, n_q: int = 9, n_positions: int = 6,
                 loss_n: int = 5, loss_steps: int = 24, loss_fit=None, progress=None) -> FluxMap:
        """Sample ``machine`` (an :class:`~.pmsm.IPMMachine`) on ``i_d in [-I_max, 0]``, ``i_q in [0, I_max]``:
        torque and flux on an ``n_d x n_q`` grid (``n_positions`` rotor angles each), loss data on a coarser
        ``loss_n x loss_n`` grid (``loss_steps`` angles over an electrical period each). ``loss_fit``: a
        :class:`~.steel.LossFit` (default: :func:`~.steel.fit_iron_loss` of the machine's steel up to 1 kHz)."""
        from .steel import fit_iron_loss

        gd, gq = np.linspace(-I_max, 0.0, n_d), np.linspace(0.0, I_max, n_q)
        T = np.zeros((n_d, n_q))
        pd, pq = np.zeros_like(T), np.zeros_like(T)
        for a, i_d in enumerate(gd):
            for b, i_q in enumerate(gq):
                op = machine.operating_point(i_d=float(i_d), i_q=float(i_q), n_positions=n_positions)
                T[a, b], pd[a, b], pq[a, b] = op["torque_mean"], op["psi_d"], op["psi_q"]
                if progress:
                    progress(f"flux map {a * n_q + b + 1}/{n_d * n_q}")
        fit = loss_fit or fit_iron_loss(machine.steel, f_max=1000.0)
        k_h, alpha, beta, k_c, k_e = fit.fitted
        out = cls(machine.pole_pairs, gd, gq, pd, pq, T, loss_coeffs=(k_h, alpha, beta, k_c, k_e))
        if loss_n:
            ld, lq = np.linspace(-I_max, 0.0, loss_n), np.linspace(0.0, I_max, loss_n)
            S = np.zeros((loss_n, loss_n, 3))
            Pm = np.zeros((loss_n, loss_n))
            for a, i_d in enumerate(ld):
                for b, i_q in enumerate(lq):
                    c = machine.loss_components(float(i_d), float(i_q), alpha, beta, steps=loss_steps)
                    S[a, b] = c["stator"] + c["rotor"]
                    Pm[a, b] = c["magnet_per_f2"]
                    if progress:
                        progress(f"loss map {a * loss_n + b + 1}/{loss_n ** 2}")
            out.loss_i_d, out.loss_i_q, out.S_iron, out.magnet_per_f2 = ld, lq, S, Pm
        return out

    def to_npz(self, path) -> None:
        np.savez(path, pole_pairs=self.pole_pairs, i_d=self.i_d, i_q=self.i_q, psi_d=self.psi_d, psi_q=self.psi_q,
                 T=self.T, loss_coeffs=np.array(self.loss_coeffs),
                 **({} if self.S_iron is None else {"loss_i_d": self.loss_i_d, "loss_i_q": self.loss_i_q,
                                                     "S_iron": self.S_iron, "magnet_per_f2": self.magnet_per_f2}))

    @classmethod
    def from_npz(cls, path) -> FluxMap:
        z = np.load(path)
        kw = {k: z[k] for k in ("loss_i_d", "loss_i_q", "S_iron", "magnet_per_f2") if k in z}
        return cls(int(z["pole_pairs"]), z["i_d"], z["i_q"], z["psi_d"], z["psi_q"], z["T"],
                   loss_coeffs=tuple(float(v) for v in z["loss_coeffs"]), **kw)


# --------------------------------------------------------------------------------------------- drive
@dataclass
class Drive:
    """A machine (flux map) fed by a voltage-source inverter: current and voltage amplitude limits, phase
    resistance, end-winding leakage inductance ``L_end`` (:mod:`.endwinding`). ``n_d`` sets the resolution of the search
    over ``i_d`` (refined by golden section)."""

    machine: object
    R: float
    I_max: float
    V_max: float
    n_d: int = 241
    L_end: float = 0.0  # end-winding leakage (H), added to psi_d and psi_q (no torque: L i_d i_q - L i_q i_d = 0)

    def _currents_for_torque(self, T_req: float, i_d: np.ndarray, iters: int = 60):
        """``i_q`` with ``torque(i_d, i_q) = T_req`` for each ``i_d`` (bisection on [0, I_max] for T_req >= 0,
        mirrored for braking), NaN where the current limit cannot reach it."""
        sg = 1.0 if T_req >= 0 else -1.0
        lo = np.zeros_like(i_d)
        hi = np.sqrt(np.maximum(self.I_max ** 2 - i_d ** 2, 0.0))
        ok = sg * self.machine.torque(i_d, sg * hi) >= abs(T_req) * (1 - 1e-12)  # MTPA at I_max itself
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            above = sg * self.machine.torque(i_d, sg * mid) >= abs(T_req)
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        return np.where(ok, sg * hi, np.nan)

    def voltage(self, i_d, i_q, w_e):
        d, q = self.machine.psi(i_d, i_q)
        d = d + self.L_end * np.asarray(i_d, float)
        q = q + self.L_end * np.asarray(i_q, float)
        v_d = self.R * i_d - w_e * q
        v_q = self.R * i_q + w_e * d
        return np.hypot(v_d, v_q)

    def losses(self, i_d, i_q, w_e):
        f = np.abs(w_e) / TWO_PI
        return 1.5 * self.R * (np.asarray(i_d) ** 2 + np.asarray(i_q) ** 2) + self.machine.core_loss(i_d, i_q, f)

    def operating_point(self, T_req: float, speed: float, include_core_loss: bool = True) -> dict | None:
        """Least-loss feasible currents for torque ``T_req`` (N m) at mechanical speed ``speed`` (rad/s), or None.
        Returns currents, voltage amplitude, losses and efficiency (motoring ``P/(P+L)``, generating ``(|P|-L)/|P|``)."""
        w_e = self.machine.pole_pairs * speed

        def evaluate(i_d):
            i_q = self._currents_for_torque(T_req, i_d)
            v = self.voltage(i_d, i_q, w_e)
            loss = (self.losses(i_d, i_q, w_e) if include_core_loss
                    else 1.5 * self.R * (i_d ** 2 + i_q ** 2))
            bad = ~np.isfinite(i_q) | (v > self.V_max * (1 + 1e-12))
            return i_q, v, np.where(bad, np.inf, loss)

        i_d = np.linspace(-self.I_max, 0.0, self.n_d)
        if T_req == 0.0 and speed == 0.0:
            return {"i_d": 0.0, "i_q": 0.0, "v": 0.0, "loss": 0.0, "efficiency": float("nan"), "T": 0.0}
        i_q, v, loss = evaluate(i_d)
        k = int(np.argmin(loss))
        if not np.isfinite(loss[k]):  # at most a sliver is feasible: the torque envelope itself
            top = self.max_torque(speed, generating=T_req < 0)
            if abs(top["T"]) < abs(T_req) * (1 - 1e-9):
                return None
            x, iq_x = np.array([top["i_d"]]), np.array([top["i_q"]])
            return self._result(T_req, speed, x, iq_x, self.voltage(x, iq_x, w_e))
        a, b = i_d[max(k - 1, 0)], i_d[min(k + 1, self.n_d - 1)]
        r = (math.sqrt(5) - 1) / 2
        for _ in range(40):  # golden section on the loss along i_d
            c, d = b - r * (b - a), a + r * (b - a)
            lc, ld = evaluate(np.array([c, d]))[2]
            if lc <= ld:
                b = d
            else:
                a = c
        x = np.array([0.5 * (a + b)])
        iq_x, v_x, l_x = evaluate(x)
        if not np.isfinite(l_x[0]) or l_x[0] > loss[k]:
            x, iq_x, v_x, l_x = i_d[k:k + 1], i_q[k:k + 1], v[k:k + 1], loss[k:k + 1]
        return self._result(T_req, speed, x, iq_x, v_x)

    def _result(self, T_req, speed, x, iq_x, v_x) -> dict:
        w_e = self.machine.pole_pairs * speed
        P = T_req * speed
        L = float(self.losses(x, iq_x, w_e)[0])
        eff = P / (P + L) if P > 0 else ((-P - L) / -P if P < 0 else 0.0)
        out = {"i_d": float(x[0]), "i_q": float(iq_x[0]), "v": float(np.asarray(v_x)[0]), "loss": L,
               "efficiency": float(eff), "T": T_req, "copper": float(1.5 * self.R * (x[0] ** 2 + iq_x[0] ** 2)),
               "core": float(self.machine.core_loss(x, iq_x, abs(w_e) / TWO_PI)[0])}
        if hasattr(self.machine, "core_loss_parts"):
            iron, magnet = self.machine.core_loss_parts(x, iq_x, abs(w_e) / TWO_PI)
            out["iron"], out["magnet"] = float(iron[0]), float(magnet[0])
        return out

    def _outer(self, gamma, w_e, sg, ns: int = 200, iters: int = 50):
        """Largest feasible current amplitude along the rays ``i = s (-sin gamma, sg cos gamma)``, ``s <= I_max``
        (0 where no point of the ray is feasible)."""
        gamma = np.atleast_1d(np.asarray(gamma, float))
        cd, cq = -np.sin(gamma), sg * np.cos(gamma)
        S = np.linspace(self.I_max, 0.0, ns)[:, None]
        ok = self.voltage(S * cd, S * cq, w_e) <= self.V_max
        first = np.argmax(ok, axis=0)  # first feasible from the outside
        has = ok.any(axis=0)
        hi = np.where(first > 0, S[np.maximum(first - 1, 0), 0], self.I_max)  # infeasible (or I_max)
        lo = S[first, 0]  # feasible
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = self.voltage(mid * cd, mid * cq, w_e) <= self.V_max
            lo = np.where(f, mid, lo)
            hi = np.where(f, hi, mid)
        lo = np.where(first == 0, self.I_max, lo)
        return np.where(has, lo, 0.0)

    def max_torque(self, speed: float, n: int = 361, generating: bool = False) -> dict:
        """Largest feasible torque (or braking torque) at mechanical speed ``speed``. Along any ray of current
        angle ``gamma`` in [0, 90 deg] towards -d the torque grows with the amplitude, so the maximum lies on the
        outer boundary of the feasible set (current circle or voltage ellipse, which covers the
        maximum-torque-per-volt region): a scan over ``n`` angles, refined by repeated zooming."""
        w_e = self.machine.pole_pairs * speed
        sg = -1.0 if generating else 1.0

        def T_of(gm):
            s_ = self._outer(gm, w_e, sg)
            return sg * self.machine.torque(-s_ * np.sin(gm), sg * s_ * np.cos(gm)), s_

        g = np.linspace(0.0, 0.5 * math.pi, n)
        T, s_ = T_of(g)
        k = int(np.argmax(T))
        a, b = g[max(k - 1, 0)], g[min(k + 1, n - 1)]
        gm, Tg, sgm = g[k:k + 1], T[k:k + 1], s_[k:k + 1]
        for _ in range(14):  # zoom: the feasible set can start abruptly (circle meets ellipse), so no bracketing
            gg = np.linspace(a, b, 21)
            tt, ss = T_of(gg)
            j = int(np.argmax(tt))
            if tt[j] >= Tg[0]:
                gm, Tg, sgm = gg[j:j + 1], tt[j:j + 1], ss[j:j + 1]
            a, b = gg[max(j - 1, 0)], gg[min(j + 1, 20)]
        return {"T": sg * float(Tg[0]), "i_d": float(-sgm[0] * np.sin(gm[0])), "i_q": float(sg * sgm[0] * np.cos(gm[0])),
                "gamma": float(gm[0])}

    def efficiency_map(self, speeds, torques, relative: bool = False) -> dict:
        """Efficiency and loss (NaN where infeasible) on ``speeds`` (rad/s) x ``torques``, and the motoring and braking
        envelopes. ``torques`` in N m (negative = generating), or with ``relative=True`` as fractions in [-1, 1] of the
        motoring (> 0) and braking (< 0) envelope at each speed, so the map reaches the envelope everywhere; the
        torque values used are returned as a (speeds, torques) array ``T``."""
        speeds, torques = np.asarray(speeds, float), np.asarray(torques, float)
        env_m = np.array([self.max_torque(w)["T"] for w in speeds])
        env_g = np.array([-self.max_torque(w, generating=True)["T"] for w in speeds])
        if relative:
            T = np.where(torques[None, :] >= 0, torques[None, :] * env_m[:, None], torques[None, :] * env_g[:, None])
        else:
            T = np.broadcast_to(torques[None, :], (len(speeds), len(torques))).copy()
        eff = np.full(T.shape, np.nan)
        loss = np.full_like(eff, np.nan)
        for a, w in enumerate(speeds):
            for b in range(T.shape[1]):
                t = float(T[a, b])
                limit = env_m[a] if t >= 0 else env_g[a]
                if w <= 0 or abs(t) > limit * (1 + 1e-9) or t == 0:
                    continue
                op = self.operating_point(t, float(w))
                if op is not None:
                    eff[a, b], loss[a, b] = op["efficiency"], op["loss"]
        return {"speeds": speeds, "torques": torques, "T": T, "efficiency": eff, "loss": loss, "envelope": env_m,
                "envelope_generating": env_g}


# --------------------------------------------------------------------------------------------- vehicle and cycle
def wltc_class3b() -> tuple[np.ndarray, np.ndarray]:
    """WLTC class 3b: time (s) and vehicle speed (m/s), 1 Hz, 0..1800 s (UN GTR No. 15, Annex 1)."""
    v = np.loadtxt(Path(__file__).parent / "data" / "wltc_class3b.csv")
    return np.arange(len(v), dtype=float), v / 3.6


@dataclass
class Vehicle:
    """Longitudinal vehicle: test mass (kg), rolling resistance, drag area C_d A (m^2), wheel radius (m), single gear
    ratio and its efficiency, rotating-mass factor (equivalent inertia as a fraction of the mass)."""

    mass: float
    c_rr: float
    cd_a: float
    wheel_radius: float
    gear_ratio: float
    gear_efficiency: float = 0.97
    rotating_mass_factor: float = 0.03
    air_density: float = 1.2
    g: float = 9.80665

    def wheel_force(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        roll = np.where(v > 0, self.mass * self.g * self.c_rr, 0.0)
        return self.mass * (1 + self.rotating_mass_factor) * a + roll + 0.5 * self.air_density * self.cd_a * v ** 2

    def motor_demand(self, t: np.ndarray, v: np.ndarray):
        """Motor speed (rad/s) and torque (N m) at the mid-points of the trace (central acceleration)."""
        tm = 0.5 * (t[1:] + t[:-1])
        vm = 0.5 * (v[1:] + v[:-1])
        a = np.diff(v) / np.diff(t)
        F = self.wheel_force(vm, a)
        Tw = F * self.wheel_radius
        eta = np.where(Tw >= 0, 1.0 / self.gear_efficiency, self.gear_efficiency)
        return tm, vm, vm / self.wheel_radius * self.gear_ratio, Tw / self.gear_ratio * eta, F


def cycle_energy(drive: Drive, vehicle: Vehicle, t: np.ndarray, v: np.ndarray, envelope_speeds: int = 25) -> dict:
    """Battery-side energy of ``drive`` over the speed trace (``t`` s, ``v`` m/s): motoring ``P_mech + losses``,
    regeneration ``P_mech - losses`` (negative) up to the braking-torque envelope, the rest by friction brakes.
    Operating points come from a table of :meth:`Drive.operating_point` on a speed x torque grid, interpolated
    bilinearly in loss (tabulating keeps the cycle to a few hundred optimisations)."""
    from scipy.interpolate import RegularGridInterpolator

    tm, vm, w, T, F = vehicle.motor_demand(t, v)
    dt = np.diff(t)
    w_max = max(float(w.max()) * 1.02, 1.0)
    sp = np.linspace(0.0, w_max, envelope_speeds)
    env_m = np.array([drive.max_torque(s)["T"] for s in sp])
    env_g = np.array([-drive.max_torque(s, generating=True)["T"] for s in sp])
    T_cap = np.interp(w, sp, env_m)
    T_brk = np.interp(w, sp, env_g)
    if np.any(T > T_cap * (1 + 1e-9)):
        raise ValueError(f"the drive cannot follow the trace: {int(np.sum(T > T_cap))} points above the envelope")
    T_m = np.maximum(T, -T_brk)  # braking beyond the envelope goes to the friction brakes
    nT = 41
    tq = np.linspace(-float(env_g.max()), float(env_m.max()), nT)
    parts = ("loss", "copper", "iron", "magnet")
    grids = {k: np.full((len(sp), nT), np.nan) for k in parts}
    for a, s in enumerate(sp):
        for b, tt in enumerate(tq):
            if (tt >= 0 and tt <= env_m[a] * (1 + 1e-9)) or (tt < 0 and -tt <= env_g[a] * (1 + 1e-9)):
                op = drive.operating_point(float(tt), float(s)) if s > 0 else None
                if op is not None:
                    for k in parts:
                        grids[k][a, b] = op.get(k, op["core"] if k == "iron" else 0.0)
    # fill unreachable cells by the nearest reachable value along torque (outside the envelope, never used)
    for g in grids.values():
        for a in range(len(sp)):
            good = np.isfinite(g[a])
            g[a] = np.interp(tq, tq[good], g[a][good]) if good.any() else 0.0
    q = np.column_stack([w, T_m])
    split = {k: RegularGridInterpolator((sp, tq), grids[k])(q) for k in parts}
    loss = split["loss"]
    P_mech = T_m * w
    moving = (w > 0) | (T_m != 0)
    P_batt = np.where(moving, P_mech + loss, 0.0)
    E = {"battery_J": float(np.sum(P_batt * dt)), "motoring_J": float(np.sum(np.where(P_batt > 0, P_batt, 0) * dt)),
         "regenerated_J": float(-np.sum(np.where(P_batt < 0, P_batt, 0) * dt)),
         "drive_losses_J": float(np.sum(np.where(moving, loss, 0) * dt)),
         "friction_brake_J": float(np.sum((T_m - T) * w * dt)),
         "wheel_positive_J": float(np.sum(np.maximum(F * vm, 0) * dt)),
         "distance_m": float(np.sum(vm * dt))}
    E["battery_Wh_per_km"] = E["battery_J"] / 3600.0 / (E["distance_m"] / 1000.0)
    E.update(t=tm, speed=w, torque=T_m, torque_demand=T, loss=loss,
             loss_copper=np.where(moving, split["copper"], 0.0), loss_iron=np.where(moving, split["iron"], 0.0),
             loss_magnet=np.where(moving, split["magnet"], 0.0))
    return E
