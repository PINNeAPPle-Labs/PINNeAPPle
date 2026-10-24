"""What the rover measures, and what it infers about the soil from it.

Measurements (:func:`measure`): from a :class:`~pinneapple_simulation.vehicle.VehicleRun`, per wheel and sample, the
normal load (suspension), slip (visual odometry against wheel odometry), sinkage (wheel camera) and motor torque
(motor current), each with Gaussian sensor noise (:class:`Sensors`; the noise levels are stated assumptions, not
datasheet values). Only the front wheels are used: they run on undisturbed soil (the rear wheels run in the ruts of
the front ones, which the steady single-pass model below does not describe).

Inference (:class:`SoilBelief`): a Gaussian belief over ``theta = (log10 k_phi, n0, n1, phi_deg)`` per mapped soil
unit, started from the soil the orbital map assigns to the unit, updated with an iterated extended Kalman filter
(static parameters, no process noise). The observation model is the steady Wong-Reece wheel: for the measured load
and slip it predicts the sinkage and the soil torque. Cohesion, the shear modulus K and k_c stay at the prior soil's
values (not identifiable from these signals at these loads). ``samples()`` draws soils from the belief for the
planner's risk estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...terramechanics.soils import Soil
from ...terramechanics.wheel import Wheel
from ...terramechanics.wong_reece import WongReeceModel

PARAMS = ("log10_k_phi", "n0", "n1", "phi_deg")
PRIOR_SD = np.array([0.30, 0.20, 0.40, 4.0])       # spread assumed around the orbital class (a stated choice)


def soil_theta(soil: Soil) -> np.ndarray:
    return np.array([math.log10(soil.k_phi), soil.n0, soil.n1, soil.phi_deg])


def theta_soil(base: Soil, theta) -> Soil:
    t = np.asarray(theta, float)
    from dataclasses import replace
    return replace(base, k_phi=float(10 ** t[0]), n0=float(max(t[1], 0.3)), n1=float(max(t[2], 0.0)),
                   phi_deg=float(np.clip(t[3], 10.0, 60.0)), name=f"{base.name} (estimated)")


@dataclass(frozen=True)
class Sensors:
    """Noise levels (one standard deviation) of the rover's measurements; stated assumptions."""
    sinkage_sd: float = 0.004          # [m] wheel camera
    slip_sd: float = 0.02              # [-] visual odometry vs wheel odometry
    torque_rel: float = 0.05           # motor current -> torque, relative
    torque_abs: float = 0.5            # [N m]
    load_rel: float = 0.05             # suspension load estimate, relative
    seed: int = 0


def measure(run, sensors: Sensors | None = None, wheels=None, every: int = 4, min_speed: float = 0.02,
            max_slip: float = 0.7, slip_window: float = 0.3, sink_length: float = 0.15) -> dict:
    """Noisy measurements from a vehicle run (front wheels by default), one row per wheel and kept sample.

    Slip is measured at every output sample with noise ``sensors.slip_sd`` and averaged over a centred window of
    ``slip_window`` seconds (the rover processes its log after the fact; a trailing window would delay the slip and
    bias the soil estimate); the slip that sets the sinkage exponent is that measured
    slip lagged over ``sink_length`` metres of rim travel, the same first-order lag the multibody simulator uses (a
    model setting, see :mod:`pinneapple_simulation.vehicle.sim`). Keeps rows every ``every`` outputs where the
    wheel touches the soil, turns, and slips at most ``max_slip`` (beyond it the wheel is digging in, a transient the
    steady model does not describe). Returns arrays ``t, x, y, wheel, unit, load, slip, slip_sink, sinkage,
    torque`` and the noise-free values (``*_true``)."""
    sensors = sensors or Sensors()
    rng = np.random.default_rng(sensors.seed)
    W = run.wheel
    spec = run.model.spec
    wheels = [i for i, c in enumerate(spec.corners) if c.front > 0] if wheels is None else list(wheels)
    t = run.t
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    nwin = max(1, int(round(slip_window / dt)))
    kern = np.ones(nwin) / nwin
    slip_meas, slip_sink = {}, {}
    for i in wheels:
        noisy = W["slip"][:, i] + sensors.slip_sd * rng.standard_normal(len(t))
        h = nwin // 2
        pad = np.r_[np.full(h, noisy[0]), noisy, np.full(nwin - 1 - h, noisy[-1])]
        sm = np.convolve(pad, kern, mode="valid")                  # centred average (the log is processed afterwards)
        lag = np.zeros(len(t))
        for k in range(1, len(t)):
            a = min(abs(W["omega"][k, i]) * spec.wheel.radius * dt / max(sink_length, 1e-9), 1.0)
            lag[k] = lag[k - 1] + a * (np.clip(sm[k], -0.99, 0.99) - lag[k - 1])
        slip_meas[i], slip_sink[i] = sm, lag
    keep = np.arange(0, len(t), every)
    keep = keep[t[keep] > max(1.0, slip_window)]
    rows = {k: [] for k in ("t", "x", "y", "wheel", "unit", "load_true", "slip_true", "sinkage_true", "torque_true",
                            "slip", "slip_sink")}
    for k in keep:
        for i in wheels:
            if (W["sinkage"][k, i] <= 1e-4 or abs(W["omega"][k, i]) * spec.wheel.radius < min_speed
                    or abs(W["slip"][k, i]) > max_slip):
                continue
            rows["t"].append(t[k])
            rows["x"].append(W["contact_point"][k, i, 0])
            rows["y"].append(W["contact_point"][k, i, 1])
            rows["wheel"].append(i)
            rows["unit"].append(int(W["soil_idx"][k, i]) if "soil_idx" in W else 0)
            rows["load_true"].append(W["load"][k, i])
            rows["slip_true"].append(W["slip"][k, i])
            rows["sinkage_true"].append(W["sinkage"][k, i])
            rows["torque_true"].append(W["drive_torque"][k, i])
            rows["slip"].append(slip_meas[i][k])
            rows["slip_sink"].append(slip_sink[i][k])
    out = {k: np.asarray(v, float if k not in ("wheel", "unit") else int) for k, v in rows.items()}
    n = len(out["t"])
    out["slip"] = np.clip(out["slip"], -0.95, 0.95)
    out["load"] = out["load_true"] * (1 + sensors.load_rel * rng.standard_normal(n))
    out["sinkage"] = np.clip(out["sinkage_true"] + sensors.sinkage_sd * rng.standard_normal(n), 0.0, None)
    out["torque"] = out["torque_true"] + np.hypot(sensors.torque_abs, sensors.torque_rel * np.abs(out["torque_true"])
                                                  ) * rng.standard_normal(n)
    out["slip_sd"] = sensors.slip_sd / math.sqrt(nwin)
    return out


@dataclass
class UnitBelief:
    base: Soil
    mean: np.ndarray
    cov: np.ndarray
    n_obs: int = 0
    history: list = field(default_factory=list)       # (t, mean, sd) after each update
    chi2_dof: float = float("nan")                    # reduced chi-square of the residuals (after calibrate)
    scale: float = 1.0                                # covariance scale applied by calibrate

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def soil(self) -> Soil:
        return theta_soil(self.base, self.mean)


class SoilBelief:
    """Gaussian beliefs over the soil parameters of each mapped unit (see the module docstring).

    ``obs_sd_sinkage`` and ``obs_sd_torque`` (abs, rel) are the observation noise of the filter: sensor noise plus an
    allowance for what the steady single-wheel model leaves out of the multibody rover (dynamics, slip lag, load
    transfer); ``input_sd`` the noise of the measured slip and (relative) load, carried through the model; stated
    choices (match them to :class:`Sensors`)."""

    def __init__(self, prior_soils: list[Soil], wheel: Wheel, g: float = 1.62, prior_sd=PRIOR_SD,
                 obs_sd_sinkage: float = 0.008, obs_sd_torque=(2.0, 0.10), input_sd=(0.008, 0.05), n_gl: int = 12):
        self.wheel, self.g, self.n_gl = wheel, float(g), n_gl
        self.units = [UnitBelief(s, soil_theta(s), np.diag(np.asarray(prior_sd, float) ** 2)) for s in prior_soils]
        self.prior = [(u.mean.copy(), u.cov.copy()) for u in self.units]
        self.obs_sd_sinkage = float(obs_sd_sinkage)
        self.obs_sd_torque = tuple(obs_sd_torque)
        self.input_sd = tuple(input_sd)           # (slip sd, relative load sd) of the measured inputs

    # ------------------------------------------------------------------ observation model
    def predict(self, unit: int, theta, load, slip, slip_sink=None) -> np.ndarray:
        """Predicted (sinkage [m], torque [N m]) stacked as one vector [z..., tau...]; ``slip_sink`` is the slip
        that sets the sinkage exponent (default: ``slip``)."""
        soil = theta_soil(self.units[unit].base, theta)
        ss = WongReeceModel(soil, self.wheel, g=self.g, n_gl=self.n_gl).steady_state(load, slip,
                                                                                    sinkage_slip=slip_sink)
        return np.r_[ss["sinkage"], ss["torque"]]

    def _R(self, unit, theta, load, slip, slip_sink, torque):
        """Observation covariance: model allowance plus the measured inputs' noise (slip, load) carried through the
        model (errors in variables, linearised: each row's sinkage and torque share its slip and load)."""
        n = len(torque)
        sz = np.full(n, self.obs_sd_sinkage)
        st = np.hypot(self.obs_sd_torque[0], self.obs_sd_torque[1] * np.abs(torque))
        R = np.diag(np.r_[sz, st] ** 2)
        ds, dW = 0.01, 0.02 * np.maximum(np.abs(load), 1.0)
        h0 = self.predict(unit, theta, load, slip, slip_sink)
        Js = (self.predict(unit, theta, load, slip + ds, slip_sink + ds) - h0) / ds
        JW = (self.predict(unit, theta, load + dW, slip, slip_sink) - h0) / np.r_[dW, dW]
        for J, var in ((Js, np.full(n, self.input_sd[0] ** 2)), (JW, (self.input_sd[1] * load) ** 2)):
            a, b = J[:n], J[n:]
            R[np.arange(n), np.arange(n)] += a * a * var
            R[np.arange(n) + n, np.arange(n) + n] += b * b * var
            R[np.arange(n), np.arange(n) + n] += a * b * var
            R[np.arange(n) + n, np.arange(n)] += a * b * var
        return R

    def update(self, unit: int, load, slip, sinkage, torque, t: float = 0.0, iters: int = 10,
               slip_sink=None) -> UnitBelief:
        """Iterated EKF update of one unit with a batch of measurements."""
        b = self.units[unit]
        load, slip = np.asarray(load, float), np.asarray(slip, float)
        slip_sink = slip if slip_sink is None else np.asarray(slip_sink, float)
        y = np.r_[np.asarray(sinkage, float), np.asarray(torque, float)]
        m0, P0 = b.mean.copy(), b.cov.copy()
        Rv = self._R(unit, m0, load, slip, slip_sink, np.asarray(torque, float))
        m = m0.copy()
        eps = np.array([0.01, 0.01, 0.02, 0.1])
        for _ in range(iters):
            h = self.predict(unit, m, load, slip, slip_sink)
            H = np.stack([(self.predict(unit, m + eps[j] * np.eye(4)[j], load, slip, slip_sink) - h) / eps[j]
                          for j in range(4)], 1)
            S = H @ P0 @ H.T + Rv
            K = np.linalg.solve(S, H @ P0).T
            m_new = m0 + K @ (y - h - H @ (m0 - m))
            if np.max(np.abs(m_new - m) / np.sqrt(np.diag(P0))) < 1e-3:
                m = m_new
                break
            m = m_new
        P = (np.eye(4) - K @ H) @ P0
        b.mean, b.cov = m, 0.5 * (P + P.T)
        b.n_obs += len(load)
        b.history.append((float(t), b.mean.copy(), b.sd.copy()))
        return b

    def assimilate(self, meas: dict, batch: int = 8, units=None, calibrate: bool = True) -> None:
        """Feed measurements in time order, ``batch`` rows per update, to the unit each row was taken on.

        ``calibrate``: afterwards scale each unit's covariance by the reduced chi-square of its residuals when it
        exceeds 1 (the Birge ratio): a filter whose observation noise is too small for what the model leaves out
        reports that through its own residuals, and its uncertainty is widened accordingly (``UnitBelief.scale``)."""
        order = np.argsort(meas["t"], kind="stable")
        for u in sorted(set(meas["unit"][order].tolist()) if units is None else units):
            rows = order[meas["unit"][order] == u]
            for k in range(0, len(rows), batch):
                r = rows[k:k + batch]
                self.update(u, meas["load"][r], meas["slip"][r], meas["sinkage"][r], meas["torque"][r],
                            t=float(meas["t"][r[-1]]), slip_sink=meas["slip_sink"][r])
            if calibrate and len(rows) > 4:
                self.calibrate(u, meas, rows)

    def chi2(self, unit: int, meas: dict, rows) -> tuple[float, int]:
        """Chi-square of the residuals of ``rows`` under the current mean (with the filter's covariance), and the
        degrees of freedom (observations minus parameters)."""
        rows = np.asarray(rows, int)
        b = self.units[unit]
        chi, n = 0.0, 0
        for k in range(0, len(rows), 16):
            r = rows[k:k + 16]
            load, slip, ss = meas["load"][r], meas["slip"][r], meas["slip_sink"][r]
            y = np.r_[meas["sinkage"][r], meas["torque"][r]]
            e = y - self.predict(unit, b.mean, load, slip, ss)
            R = self._R(unit, b.mean, load, slip, ss, meas["torque"][r])
            chi += float(e @ np.linalg.solve(R, e))
            n += len(e)
        return chi, max(n - len(PARAMS), 1)

    def calibrate(self, unit: int, meas: dict, rows) -> float:
        chi, dof = self.chi2(unit, meas, rows)
        b = self.units[unit]
        b.chi2_dof = chi / dof
        b.scale = max(1.0, b.chi2_dof)
        b.cov = b.cov * b.scale
        if b.history:
            t, m, _ = b.history[-1]
            b.history.append((t, m.copy(), b.sd.copy()))
        return b.scale

    # ------------------------------------------------------------------ use
    def samples(self, unit: int, n: int = 24, seed: int = 0) -> list[Soil]:
        """Soils drawn from the belief of ``unit`` (the mean first)."""
        b = self.units[unit]
        rng = np.random.default_rng(seed + 101 * unit)
        L = np.linalg.cholesky(b.cov + 1e-12 * np.eye(4))
        th = [b.mean] + [b.mean + L @ rng.standard_normal(4) for _ in range(n - 1)]
        return [theta_soil(b.base, t) for t in th]

    def recovery(self, unit: int, truth: Soil) -> dict:
        """Error of the belief against a known true soil: per parameter the error, the posterior sd and the error in
        posterior sds (a calibrated belief keeps it within about +-2 to 3)."""
        b = self.units[unit]
        tt = soil_theta(truth)
        err = b.mean - tt
        prior_err = self.prior[unit][0] - tt
        return {p: {"true": float(tt[i]), "estimate": float(b.mean[i]), "sd": float(b.sd[i]),
                    "error": float(err[i]), "z": float(err[i] / max(b.sd[i], 1e-12)),
                    "prior_error": float(prior_err[i])} for i, p in enumerate(PARAMS)}

    def predictive_check(self, unit: int, meas: dict, rows) -> dict:
        """Prediction error of the current belief on measurement rows it was not updated with."""
        rows = np.asarray(rows, int)
        h = self.predict(unit, self.units[unit].mean, meas["load"][rows], meas["slip"][rows], meas["slip_sink"][rows])
        n = len(rows)
        ez, et = h[:n] - meas["sinkage"][rows], h[n:] - meas["torque"][rows]
        return {"n": n, "sinkage_rmse_mm": float(1e3 * np.sqrt(np.mean(ez ** 2))),
                "torque_rmse_Nm": float(np.sqrt(np.mean(et ** 2)))}


__all__ = ["PARAMS", "PRIOR_SD", "Sensors", "measure", "SoilBelief", "UnitBelief", "soil_theta", "theta_soil"]
