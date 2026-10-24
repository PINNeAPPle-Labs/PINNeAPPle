"""How the twin knows it is wrong: physics residuals, a fast model, a Kalman filter and its domain of validity.

Three independent sources of evidence at every step:

1. **Physics residuals** (:class:`Residuals`), from the readings alone: mass balance at the tee; Darcy-Weisbach /
   Colebrook flow of each pipe from its measured pressure drop vs its flow meter; the pump curve; each valve's
   orifice law; the heater energy balance; each branch's heat loss. Each residual involves a few sensors and
   components, so a fault moves the residual vector in its own *direction* (the derivative of the residuals with
   respect to that fault). A biased sensor moves every residual it appears in; an obstruction in pipe B moves only
   the Darcy residual of pipe B (mass and energy still balance).
2. **The fast model** (:class:`Surrogate`): a POD + Gaussian-process surrogate of the ten readings over the operating
   envelope it was trained on, with its predictive standard deviation; its innovations say "the plant does not do
   what I predict".
3. **The domain of validity**: the inputs against the training envelope (normalised excess outside the box) and
   the Gaussian-process spread; plus the quasi-steady validity of the model for valve moves.

A Kalman filter (:class:`pinneapple_analysis.state_estimation.kalman.ExtendedKalmanFilter`) tracks an extra loss
coefficient per pipe (random walk) from the measured pressure drops and flows, so a process change comes with its
size and uncertainty.

:class:`Diagnoser` turns the evidence into one of four answers -- normal, process changed (which pipe), sensor wrong
(which sensor, bias or noise), model outside its domain (which input) -- with thresholds calibrated on healthy runs
(:meth:`Diagnoser.calibrate`), and says what it needs next: inspect the pipe, recalibrate the sensor, or run the
reference simulation at the new operating point (and add it to the model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plant import (
    CP_WATER,
    INPUTS,
    NORMAL_BOX,
    SENSORS,
    SIGMA,
    G,
    Loop,
    readings,
    steady_state,
    water,
)

RESIDUALS = ("mass_tee", "darcy_A", "darcy_B", "darcy_C", "pump_curve", "valve_B", "valve_C", "heater_energy",
             "heat_loss_B", "heat_loss_C")
COMPONENTS = ("A", "B", "C")          # pipes that can be obstructed


class Residuals:
    """Physics residuals of one reading vector, in sensor units, and their noise standard deviation (propagated
    from SIGMA by finite differences)."""

    def __init__(self, loop: Loop):
        self.loop = loop

    def raw(self, y: np.ndarray, u: dict[str, float], k_extra: dict[str, float] | None = None) -> np.ndarray:
        L = self.loop
        ke = {"A": 0.0, "B": 0.0, "C": 0.0, **(k_extra or {})}
        Pp, Pt, PB, PC, QA, QB, QC, TA, TB, TC = y
        rho, mu = water(min(max(0.5 * (u["T_in"] + TA), 0.0), 99.0))
        bar = rho * G / 1e5
        qA, qB, qC = QA / 3600, QB / 3600, QC / 3600
        A, B, C = L.pipe_A, L.pipe_B, L.pipe_C

        def q_of(pipe, dp_bar, k):
            return 3600 * pipe.flow_from_head_loss(dp_bar / bar, rho, mu, k)

        kvB = L.kv0_B / max(u["u_B"], 0.02) ** 2
        kvC = L.kv0_C / max(u["u_C"], 0.02) ** 2
        vB, vC = qB / B.area, qC / C.area

        def t_loss(pipe, q):
            return L.t_room_c + (TA - L.t_room_c) * math.exp(-pipe.ua_w_k / (rho * max(q, 1e-6) * CP_WATER))

        return np.array([
            QA - QB - QC,
            QA - q_of(A, Pp - Pt, ke["A"]),
            QB - q_of(B, Pt - PB, ke["B"]),
            QC - q_of(C, Pt - PC, ke["C"]),
            Pp - bar * L.pump_head(qA, u["n"]),
            PB - bar * (L.z_B + kvB * vB * abs(vB) / (2 * G)),
            PC - bar * (L.z_C + kvC * vC * abs(vC) / (2 * G)),
            TA - u["T_in"] - u["Q_heat_kW"] * 1e3 / (rho * max(qA, 1e-6) * CP_WATER),
            TB - t_loss(B, qB),
            TC - t_loss(C, qC)])

    def jacobian(self, y: np.ndarray, u: dict[str, float]) -> np.ndarray:
        """d residual / d reading (10 x 10), one-sided differences of a tenth of each sensor's sigma."""
        r0 = self.raw(y, u)
        J = np.empty((len(RESIDUALS), len(SENSORS)))
        for j in range(len(SENSORS)):
            dy = np.zeros(len(SENSORS))
            dy[j] = 0.1 * SIGMA[j]
            J[:, j] = (self.raw(y + dy, u) - r0) / dy[j]
        return J

    def evaluate(self, y: np.ndarray, u: dict[str, float]) -> dict[str, np.ndarray]:
        """Residuals, their noise sigma, the normalised residuals and the directions of every fault hypothesis
        (columns: a 1-sigma bias of each sensor, then a real unit extra loss coefficient in each pipe), normalised."""
        r = self.raw(y, u)
        J = self.jacobian(y, u)
        sig = np.sqrt((J ** 2) @ SIGMA ** 2)
        D = J * SIGMA[None, :] / sig[:, None]                          # sensor bias of 1 sigma -> z
        comp = []
        for c in COMPONENTS:     # a real extra K of 1 in pipe c, seen by the clean model: r(K=0) - r(K=1) -> z
            comp.append((r - self.raw(y, u, {c: 1.0})) / sig)
        return {"r": r, "sigma": sig, "z": r / sig, "D_sensor": D, "D_comp": np.stack(comp, 1)}


# ----------------------------------------------------------------------------- fast model
@dataclass
class Surrogate:
    """POD + Gaussian-process model of the ten readings over NORMAL_BOX (the twin's fast model)."""
    n_train: int = 120
    seed: int = 0
    model: Any = field(default=None, repr=False)
    lo: np.ndarray = field(default=None, repr=False)
    hi: np.ndarray = field(default=None, repr=False)

    def fit(self, loop: Loop) -> Surrogate:
        from scipy.stats import qmc

        from pinneapple_neural.architectures.rom.parametric import ParametricPOD
        self.lo = np.array([NORMAL_BOX[k][0] for k in INPUTS])
        self.hi = np.array([NORMAL_BOX[k][1] for k in INPUTS])
        U = self.lo + qmc.LatinHypercube(d=len(INPUTS), seed=self.seed).random(self.n_train) * (self.hi - self.lo)
        Y = np.array([readings(steady_state(loop, dict(zip(INPUTS, x, strict=True)))) for x in U])
        self._scale = Y.std(0) + 1e-12
        self._mean = Y.mean(0)
        self.model = ParametricPOD(r=len(SENSORS), energy=None, regressor="gpr").fit(U, (Y - self._mean) / self._scale)
        self.U_, self.Y_ = U, Y
        return self

    def predict(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, s = self.model.predict(np.atleast_2d(U), return_std=True)
        return m * self._scale + self._mean, s * self._scale

    def excess(self, u: np.ndarray) -> float:
        """How far the inputs are outside the training box, in box widths (0 inside)."""
        w = self.hi - self.lo
        return float(np.max(np.maximum(np.maximum(self.lo - u, u - self.hi), 0.0) / w))


# ----------------------------------------------------------------------------- Kalman filter
class LossTracker:
    """EKF on the extra loss coefficient of each pipe (random walk), observed through the measured pressure drop
    of each pipe given its measured flow."""

    def __init__(self, loop: Loop, q_walk: float = 0.02, k0_sigma: float = 0.5):
        from pinneapple_analysis.state_estimation.kalman import ExtendedKalmanFilter
        self.loop = loop
        self._ctx: dict[str, Any] = {}
        self.ekf = ExtendedKalmanFilter(3, 3, f=lambda x: x, h=self._h, Q=np.eye(3) * q_walk ** 2,
                                        R=np.eye(3) * (2 * SIGMA[0] ** 2), F_jac=lambda x: np.eye(3))
        self.ekf.initialize(np.zeros(3), np.eye(3) * k0_sigma ** 2)

    def _h(self, x):
        c = self._ctx
        rho, mu, bar = c["rho"], c["mu"], c["bar"]
        L = self.loop
        return np.array([bar * p.head_loss(q, rho, mu, k) for p, q, k in
                         ((L.pipe_A, c["qA"], x[0]), (L.pipe_B, c["qB"], x[1]), (L.pipe_C, c["qC"], x[2]))])

    def step(self, y: np.ndarray, u: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        Pp, Pt, PB, PC, QA, QB, QC, TA = y[:8]
        rho, mu = water(min(max(0.5 * (u["T_in"] + TA), 0.0), 99.0))
        self._ctx = {"rho": rho, "mu": mu, "bar": rho * G / 1e5, "qA": QA / 3600, "qB": QB / 3600,
                     "qC": QC / 3600}
        out = self.ekf.step(np.array([Pp - Pt, Pt - PB, Pt - PC]))
        return out["x"].copy(), np.sqrt(np.diag(out["P"]))


# ----------------------------------------------------------------------------- decision
HYPOTHESES = [f"sensor:{s}" for s in SENSORS] + [f"process:{c}" for c in COMPONENTS]


@dataclass
class Diagnoser:
    """Windowed evidence -> hypothesis. ``window`` steps; thresholds from :meth:`calibrate` (healthy runs)."""
    loop: Loop
    surrogate: Surrogate
    window: int = 18
    thresholds: dict[str, float] = field(default_factory=dict)
    ood_threshold: float = 0.02             # 2 % of a box width outside the training envelope

    def evidence(self, run: dict[str, Any]) -> dict[str, np.ndarray]:
        """Per-step evidence over a plant run (output of :func:`.plant.run_plant`)."""
        t, Y = run["t"], run["y"]
        U = np.stack([run["u"][k] for k in INPUTS], 1)
        res = Residuals(self.loop)
        Z, Dsen, Dcomp = [], [], []
        for i in range(len(t)):
            e = res.evaluate(Y[i], dict(zip(INPUTS, U[i], strict=True)))
            Z.append(e["z"])
            Dsen.append(e["D_sensor"])
            Dcomp.append(e["D_comp"])
        Yp, Sp = self.surrogate.predict(U)
        innov = (Y - Yp) / np.sqrt(SIGMA ** 2 + Sp ** 2)
        kf = LossTracker(self.loop)
        K, Ks = zip(*(kf.step(Y[i], dict(zip(INPUTS, U[i], strict=True))) for i in range(len(t))), strict=True)
        ood = np.array([self.surrogate.excess(u) for u in U])
        return {"t": t, "z": np.array(Z), "D_sensor": np.array(Dsen), "D_comp": np.array(Dcomp),
                "innovation": innov, "surrogate_std": Sp, "surrogate_pred": Yp, "ood": ood,
                "k_extra": np.array(K), "k_extra_std": np.array(Ks)}

    def statistics(self, ev: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Windowed statistics per step: mean shift (chi2 of sqrt(W) x window mean of z), variance excess (window
        mean of z^2 - 1, summed), surrogate innovation chi2, input excess."""
        W = self.window
        z, inn = ev["z"], ev["innovation"]
        n = len(z)
        mean = np.array([z[max(0, i - W + 1):i + 1].mean(0) for i in range(n)])
        msq = np.array([(z[max(0, i - W + 1):i + 1] ** 2).mean(0) for i in range(n)])
        imean = np.array([inn[max(0, i - W + 1):i + 1].mean(0) for i in range(n)])
        cnt = np.minimum(np.arange(1, n + 1), W)[:, None]
        return {"shift": ((np.sqrt(cnt) * mean) ** 2).sum(1), "spread": np.clip(msq - mean ** 2 - 1, 0, None).sum(1),
                "model": ((np.sqrt(cnt) * imean) ** 2).sum(1), "ood": ev["ood"], "mean_z": mean, "msq_z": msq}

    def calibrate(self, runs: list[dict[str, Any]], quantile: float = 1.0, margin: float = 1.25) -> Diagnoser:
        """Thresholds = ``margin`` x the ``quantile`` of each statistic over healthy runs (after the first window)."""
        acc: dict[str, list] = {"shift": [], "spread": [], "model": []}
        for r in runs:
            s = self.statistics(self.evidence(r))
            for k in acc:
                acc[k].append(s[k][self.window:])
        self.thresholds = {k: margin * float(np.quantile(np.concatenate(v), quantile)) for k, v in acc.items()}
        return self

    def isolate(self, mean_z: np.ndarray, msq_z: np.ndarray, D_sensor: np.ndarray, D_comp: np.ndarray,
                kind: str) -> tuple[str, float, dict[str, float]]:
        """Best-matching hypothesis: for a mean shift, the fault direction that explains the window mean of z by
        least squares (``alpha d``); for a spread increase, the sensor whose squared direction explains the
        variance excess. Returns (hypothesis, fitted size, fit quality per hypothesis)."""
        fits = {}
        if kind == "shift":
            dirs = [D_sensor[:, j] for j in range(D_sensor.shape[1])] + [D_comp[:, j] for j in range(D_comp.shape[1])]
            for h, d in zip(HYPOTHESES, dirs, strict=True):
                a = float(d @ mean_z / max(d @ d, 1e-12))
                if h.startswith("process") and a < 0:
                    fits[h] = (float(np.sum(mean_z ** 2)), a)              # an obstruction only adds loss
                    continue
                fits[h] = (float(np.sum((mean_z - a * d) ** 2)), a)
        else:
            ex = np.clip(msq_z - mean_z ** 2 - 1, 0, None)
            for j, s in enumerate(SENSORS):
                d = D_sensor[:, j] ** 2
                a = float(d @ ex / max(d @ d, 1e-12))
                fits[f"sensor:{s}"] = (float(np.sum((ex - max(a, 0) * d) ** 2)), a)
        best = min(fits, key=lambda h: fits[h][0])
        return best, fits[best][1], {h: v[0] for h, v in fits.items()}

    def diagnose(self, run: dict[str, Any], ev: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        """The diagnosis at every step and the final verdict: kind, target, evidence, first alarm, next action."""
        ev = ev or self.evidence(run)
        st = self.statistics(ev)
        th = self.thresholds
        n = len(ev["t"])
        kinds = []
        for i in range(n):
            if st["ood"][i] > self.ood_threshold:
                kinds.append("domain")
            elif i >= self.window and st["shift"][i] > th["shift"]:
                kinds.append("shift")
            elif i >= self.window and st["spread"][i] > th["spread"]:
                kinds.append("spread")
            elif i >= self.window and st["model"][i] > th["model"]:
                kinds.append("model")
            else:
                kinds.append("normal")
        alarm = [i for i, k in enumerate(kinds) if k != "normal"]
        first = alarm[0] if alarm else None
        i = n - 1
        k = kinds[i]
        out: dict[str, Any] = {"first_alarm_s": None if first is None else float(ev["t"][first]),
                               "statistics_final": {s: float(st[s][i]) for s in ("shift", "spread", "model", "ood")},
                               "thresholds": dict(th), "kinds": kinds, "stats": st, "evidence": ev}
        if k == "normal":
            out.update(kind="normal", target="", size=None, action="none: physics, model and domain agree")
        elif k == "domain":
            u = {kk: float(run["u"][kk][i]) for kk in INPUTS}
            x = np.array([u[kk] for kk in INPUTS])
            w = self.surrogate.hi - self.surrogate.lo
            j = int(np.argmax(np.maximum(np.maximum(self.surrogate.lo - x, x - self.surrogate.hi), 0) / w))
            out.update(kind="domain", target=INPUTS[j], size=u[INPUTS[j]],
                       action=f"run the reference simulation at {INPUTS[j]} = {u[INPUTS[j]]:.3g} and add it to the "
                              "model; until then trust the physics residuals, not the fast model")
        elif k == "model":
            out.update(kind="domain", target="inside the envelope", size=None,
                       action="the fast model disagrees with physics-consistent readings: run the reference here and "
                              "refit")
        else:
            h, a, fits = self.isolate(st["mean_z"][i], st["msq_z"][i], ev["D_sensor"][i], ev["D_comp"][i],
                                      "shift" if k == "shift" else "spread")
            kind, target = h.split(":")
            out["fits"] = fits
            if kind == "process":
                j = COMPONENTS.index(target)
                kx, ks = float(ev["k_extra"][i, j]), float(ev["k_extra_std"][i, j])
                out.update(kind="process", target=target, size=kx, size_std=ks,
                           action=f"pipe {target}: extra loss coefficient {kx:.2f} +- {2 * ks:.2f} (Kalman); "
                                  "inspect it for a partial obstruction")
            else:
                j = SENSORS.index(target)
                mode = "drift" if k == "shift" else "noise"
                out.update(kind="sensor", target=target, mode=mode,
                           size=float(a * SIGMA[j]) if mode == "drift" else float(1 + max(a, 0)) ** 0.5,
                           action=f"{target}: {'bias' if mode == 'drift' else 'noise'} -- check / recalibrate it; "
                                  "meanwhile estimate it from the other sensors")
        return out
