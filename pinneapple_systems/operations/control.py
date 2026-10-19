"""Physics-based control of the heated tank: model predictive control against PID and rule-based baselines, in
closed loop on the same simulated plant, in simulation / recommendation mode only.

Controllers (all expose ``act(t, reading, estimate) -> (u, info)``):

``PIDBaseline``        two decoupled loops (level -> pump flow, temperature -> heater power) with feed-forward
                       bias, saturation and anti-windup (:class:`~pinneapple_systems.component_modeling.control.PIDController`);
``RuleBasedBaseline``  what many plants run today: hysteresis bands that switch the pump between two rates and the
                       heater between off and a high power;
``LinearMPC``          successive linearisation of the tank's balances at every step, zero-order-hold
                       discretisation (matrix exponential), a quadratic program over the horizon solved with SLSQP
                       (hard actuator and rate limits, soft output constraints);
``NonlinearMPC``       the full nonlinear balances over the horizon: CasADi + IPOPT with multiple shooting when
                       CasADi is installed, otherwise single shooting with SciPy's SLSQP and batched finite-difference
                       gradients. ``model=`` may be any model with ``step(x, v)`` -- e.g. a reduced-order surrogate --
                       which then runs through the SciPy path.

The MPCs act on the state estimated by a :class:`~pinneapple_systems.operations.twin.TwinRuntime` (ensemble Kalman
filter with an unaccounted outflow and a heater efficiency in the state), which is what makes them offset-free
under unmeasured disturbances. :func:`run_closed_loop` simulates one controller on a :class:`ControlScenario` and
:func:`compare_controllers` runs several on identical scenarios (same noise, same disturbances) and tabulates
tracking error, constraint violations, energy, actuator travel and solve time.

There is deliberately no path from these controllers to real equipment: :func:`recommend` returns proposed actions
and their predicted consequences for a person to review.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from pinneapple_systems.component_modeling.control import PIDController

from .plant import (
    Fault,
    SensorNoise,
    TankParams,
    TankSimulator,
    inputs_for_steady_state,
    tank_rhs,
    tank_step,
)
from .twin import AlertRules, LumpedTankModel, TwinRuntime

try:  # optional: nonlinear MPC with IPOPT
    import casadi as ca
except ImportError:  # pragma: no cover - exercised when casadi is missing
    ca = None

MODE = "simulation / recommendation only: no command path to real equipment"


# --------------------------------------------------------------------------------------------------------------
# Problem description
# --------------------------------------------------------------------------------------------------------------


@dataclass
class ControlSpec:
    """Objectives, constraints and actuator limits.

    ``setpoints`` is a list of ``(time, h_ref, T_ref)`` steps. Tracking errors are normalised by ``h_scale`` and
    ``T_scale``; ``w_energy`` prices heater power (per unit of ``Q / Q_max`` and step), ``w_move`` actuator moves.
    Output constraints are soft (penalty ``w_constraint`` per normalised violation squared) so that a disturbance
    beyond the actuators' reach degrades gracefully instead of making the problem infeasible."""

    dt: float = 20.0
    horizon: int = 15
    setpoints: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 1.2, 25.0), (600.0, 1.2, 29.0), (1800.0, 1.0, 29.0)])
    h_bounds: tuple[float, float] = (0.6, 1.6)
    T_max: float = 31.0
    q_in_bounds: tuple[float, float] = (0.0, 0.015)
    Q_bounds: tuple[float, float] = (0.0, 600e3)
    dq_max: float = 0.002            # pump flow change per step [m3/s]
    dQ_max: float = 80e3             # heater power change per step [W]
    h_scale: float = 0.05
    T_scale: float = 0.5
    w_h: float = 1.0
    w_T: float = 1.0
    w_move: float = 10.0
    w_energy: float = 0.05
    w_constraint: float = 1e4

    def reference(self, t: float) -> tuple[float, float]:
        h, T = self.setpoints[0][1], self.setpoints[0][2]
        for ts, hr, Tr in self.setpoints:
            if t >= ts:
                h, T = hr, Tr
        return float(h), float(T)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ControlSpec:
        """Build from a JSON object; unknown keys are an error, missing keys take the defaults."""
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown control-spec fields {sorted(bad)}")
        kw = dict(d)
        for k in ("h_bounds", "q_in_bounds", "Q_bounds"):
            if k in kw:
                kw[k] = tuple(float(v) for v in kw[k])
        if "setpoints" in kw:
            kw["setpoints"] = sorted((float(a), float(b), float(c)) for a, b, c in kw["setpoints"])
        spec = cls(**kw)
        if spec.horizon < 2 or spec.dt <= 0:
            raise ValueError("horizon must be >= 2 and dt > 0")
        return spec


@dataclass
class ControlScenario:
    """The plant side of a closed-loop study: duration, disturbances and faults (shared by every controller)."""

    duration: float = 3600.0
    x0: tuple[float, float] = (1.2, 25.0)
    T_in_steps: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 15.0), (2400.0, 11.0)])
    T_amb: float = 20.0
    faults: list[Fault] = field(default_factory=lambda: [Fault("valve", 1200.0, 3600.0, -0.2)])
    noise: SensorNoise = field(default_factory=SensorNoise)
    seed: int = 0

    def T_in(self, t: float) -> float:
        v = self.T_in_steps[0][1]
        for ts, val in self.T_in_steps:
            if t >= ts:
                v = val
        return float(v)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["faults"] = [f.to_dict() for f in self.faults]
        d["noise"] = dict(self.noise.sigma)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ControlScenario:
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown scenario fields {sorted(bad)}")
        kw = dict(d)
        if "faults" in kw:
            kw["faults"] = [Fault(**{**f, "end": float("inf") if f.get("end") is None else f["end"]})
                            for f in kw["faults"]]
        if "noise" in kw:
            kw["noise"] = SensorNoise({**SensorNoise().sigma, **{k: float(v) for k, v in kw["noise"].items()}})
        if "T_in_steps" in kw:
            kw["T_in_steps"] = sorted((float(a), float(b)) for a, b in kw["T_in_steps"])
        if "x0" in kw:
            kw["x0"] = tuple(float(v) for v in kw["x0"])
        return cls(**kw)


# --------------------------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------------------------


class PIDBaseline:
    """Level loop on the pump, temperature loop on the heater, both with feed-forward bias and anti-windup."""

    name = "PID"
    uses_estimate = False

    def __init__(self, spec: ControlSpec, params: TankParams | None = None, kp_h: float = 0.02,
                 ki_h: float = 1e-4, kp_T: float = 60e3, ki_T: float = 400.0):
        p = params or TankParams()
        h0, T0 = spec.reference(0.0)
        u0 = inputs_for_steady_state(p, h0, T0, 15.0, 20.0)
        self.spec = spec
        self.level = PIDController(kp_h, ki_h, 0.0, h0, output_limits=spec.q_in_bounds, bias=float(u0[0]))
        self.temp = PIDController(kp_T, ki_T, 0.0, T0, output_limits=spec.Q_bounds, bias=float(u0[1]))
        self.gains = {"kp_h": kp_h, "ki_h": ki_h, "kp_T": kp_T, "ki_T": ki_T}

    def act(self, t: float, reading: dict[str, float], estimate: dict | None = None):
        self.level.setpoint, self.temp.setpoint = self.spec.reference(t)
        u = np.array([self.level.update(reading["h"], self.spec.dt), self.temp.update(reading["T"], self.spec.dt)])
        return u, {}


class RuleBasedBaseline:
    """Hysteresis rules: pump at a high rate below ``h_ref - band_h`` and a low rate above ``h_ref + band_h``; heater
    at ``Q_on`` below ``T_ref - band_T`` and off above ``T_ref + band_T``."""

    name = "Rule-based"
    uses_estimate = False

    def __init__(self, spec: ControlSpec, band_h: float = 0.05, band_T: float = 0.5, q_low: float = 0.004,
                 q_high: float = 0.010, Q_on: float = 600e3):
        self.spec, self.band_h, self.band_T = spec, band_h, band_T
        self.q_low, self.q_high, self.Q_on = q_low, q_high, Q_on
        self.pump_high, self.heater_on = True, True
        self.rules = {"band_h": band_h, "band_T": band_T, "q_low": q_low, "q_high": q_high, "Q_on": Q_on}

    def act(self, t: float, reading: dict[str, float], estimate: dict | None = None):
        h_ref, T_ref = self.spec.reference(t)
        if reading["h"] < h_ref - self.band_h:
            self.pump_high = True
        elif reading["h"] > h_ref + self.band_h:
            self.pump_high = False
        if reading["T"] < T_ref - self.band_T:
            self.heater_on = True
        elif reading["T"] > T_ref + self.band_T:
            self.heater_on = False
        return np.array([self.q_high if self.pump_high else self.q_low, self.Q_on if self.heater_on else 0.0]), {}


# --------------------------------------------------------------------------------------------------------------
# MPC
# --------------------------------------------------------------------------------------------------------------


def _estimate_vector(estimate: dict) -> tuple[np.ndarray, float, float]:
    st = estimate["state"]
    return np.array([st["h"], st["T"]]), float(st.get("q_leak", 0.0)), float(st.get("eta", 1.0))


class _MPCBase:
    uses_estimate = True

    def __init__(self, spec: ControlSpec, params: TankParams | None = None):
        self.spec = spec
        self.p = params or TankParams()
        self.u_prev: np.ndarray | None = None
        self.U_warm: np.ndarray | None = None
        self.solve_ms: list[float] = []
        self.n_fail = 0
        lo = np.array([spec.q_in_bounds[0], spec.Q_bounds[0]])
        hi = np.array([spec.q_in_bounds[1], spec.Q_bounds[1]])
        self.u_lo, self.u_hi = lo, hi
        self.u_scale = hi - lo
        self.du_max = np.array([spec.dq_max, spec.dQ_max])

    def _refs(self, t: float) -> np.ndarray:
        return np.array([self.spec.reference(t + (k + 1) * self.spec.dt) for k in range(self.spec.horizon)])

    def _cost_terms(self, X: np.ndarray, U: np.ndarray, R: np.ndarray, u_prev: np.ndarray) -> float:
        """Objective for predicted states X (N, 2) (after each move), moves U (N, 2), references R (N, 2)."""
        s = self.spec
        eh = (X[:, 0] - R[:, 0]) / s.h_scale
        eT = (X[:, 1] - R[:, 1]) / s.T_scale
        dU = np.diff(np.vstack([u_prev, U]), axis=0) / self.u_scale
        vh = np.maximum(0, s.h_bounds[0] - X[:, 0]) + np.maximum(0, X[:, 0] - s.h_bounds[1])
        vT = np.maximum(0, X[:, 1] - s.T_max)
        return float(s.w_h * eh @ eh + s.w_T * eT @ eT + s.w_move * np.sum(dU ** 2)
                     + s.w_energy * np.sum(U[:, 1] / s.Q_bounds[1])
                     + s.w_constraint * (np.sum((vh / s.h_scale) ** 2) + np.sum((vT / s.T_scale) ** 2)))

    def _start(self, reading: dict[str, float]) -> np.ndarray:
        if self.u_prev is None:
            self.u_prev = np.array([reading["q_in"], reading["Q"]], dtype=float)
        return self.u_prev

    def _finish(self, U: np.ndarray, X: np.ndarray, t: float, t0: float, ok: bool, extra: dict):
        self.solve_ms.append(1000 * (time.perf_counter() - t0))
        if not ok:
            self.n_fail += 1
        self.U_warm = np.vstack([U[1:], U[-1:]])
        u = np.clip(U[0], np.maximum(self.u_lo, self.u_prev - self.du_max),
                    np.minimum(self.u_hi, self.u_prev + self.du_max))
        self.u_prev = u
        times = (t + self.spec.dt * np.arange(1, self.spec.horizon + 1)).tolist()
        info = {"plan_t": times, "plan_q_in": U[:, 0].tolist(), "plan_Q": U[:, 1].tolist(),
                "pred_h": X[:, 0].tolist(), "pred_T": X[:, 1].tolist(), "solve_ms": self.solve_ms[-1],
                "solver_ok": bool(ok), **extra}
        return u, info


class LinearMPC(_MPCBase):
    """Successively linearised MPC: a QP over the horizon with SLSQP."""

    name = "Linear MPC"

    def _linearize(self, x0, u0, d, q_leak, eta):
        p, eps = self.p, 1e-6
        f0 = tank_rhs(x0, u0, d, p, eta=eta, q_leak=q_leak)
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        for j in range(2):
            dx = np.zeros(2)
            dx[j] = eps * max(1.0, abs(x0[j]))
            A[:, j] = (tank_rhs(x0 + dx, u0, d, p, eta=eta, q_leak=q_leak) - f0) / dx[j]
            du = np.zeros(2)
            du[j] = eps * max(1e-3, abs(u0[j]))
            B[:, j] = (tank_rhs(x0, u0 + du, d, p, eta=eta, q_leak=q_leak) - f0) / du[j]
        c = f0 - A @ x0 - B @ u0
        M = np.zeros((5, 5))
        M[:2, :2], M[:2, 2:4], M[:2, 4] = A, B, c
        E = expm(M * self.spec.dt)
        return E[:2, :2], E[:2, 2:4], E[:2, 4]

    def act(self, t: float, reading: dict[str, float], estimate: dict):
        t0 = time.perf_counter()
        s, N = self.spec, self.spec.horizon
        x0, q_leak, eta = _estimate_vector(estimate)
        u_prev = self._start(reading)
        d = np.array([reading["T_in"], reading["T_amb"]])
        Ad, Bd, cd = self._linearize(x0, u_prev, d, q_leak, eta)
        # condensed prediction X = Phi x0 + G vec(U) + C
        Phi = np.zeros((N, 2, 2))
        G = np.zeros((N, 2, N, 2))
        C = np.zeros((N, 2))
        Ak = np.eye(2)
        for k in range(N):
            Ak = Ad @ Ak
            Phi[k] = Ak
            C[k] = (Ad @ C[k - 1] if k else 0) + cd
            for j in range(k + 1):
                G[k, :, j, :] = np.linalg.matrix_power(Ad, k - j) @ Bd
        Gm = G.reshape(2 * N, 2 * N)
        base = (np.einsum("kij,j->ki", Phi, x0) + C).reshape(-1)
        R = self._refs(t)
        U0 = self.U_warm if self.U_warm is not None else np.repeat(u_prev[None], N, 0)
        us = self.u_scale

        def unpack(z):
            return z.reshape(N, 2) * us

        def fun(z):
            U = unpack(z)
            X = (base + Gm @ U.reshape(-1)).reshape(N, 2)
            return self._cost_terms(X, U, R, u_prev)

        def jac(z):
            U = unpack(z)
            X = (base + Gm @ U.reshape(-1)).reshape(N, 2)
            gX = np.zeros((N, 2))
            gX[:, 0] = 2 * s.w_h * (X[:, 0] - R[:, 0]) / s.h_scale ** 2
            gX[:, 1] = 2 * s.w_T * (X[:, 1] - R[:, 1]) / s.T_scale ** 2
            lo_v = np.maximum(0, s.h_bounds[0] - X[:, 0])
            hi_v = np.maximum(0, X[:, 0] - s.h_bounds[1])
            gX[:, 0] += 2 * s.w_constraint * (hi_v - lo_v) / s.h_scale ** 2
            gX[:, 1] += 2 * s.w_constraint * np.maximum(0, X[:, 1] - s.T_max) / s.T_scale ** 2
            gU = (Gm.T @ gX.reshape(-1)).reshape(N, 2)
            dU = np.diff(np.vstack([u_prev, U]), axis=0) / us ** 2
            gmove = 2 * s.w_move * dU
            gU += gmove
            gU[:-1] -= gmove[1:]
            gU[:, 1] += s.w_energy / s.Q_bounds[1]
            return (gU * us).reshape(-1)

        # rate limits |u_k - u_{k-1}| <= du_max as linear inequalities on z
        D = np.zeros((N, N))
        D[np.arange(N), np.arange(N)] = 1
        D[np.arange(1, N), np.arange(N - 1)] = -1
        rows, rhs = [], []
        for i in range(2):
            Di = np.zeros((N, 2 * N))
            Di[:, i::2] = D * us[i]
            off = np.zeros(N)
            off[0] = u_prev[i]
            rows += [Di, -Di]
            rhs += [self.du_max[i] + off, self.du_max[i] - off]
        Aineq, bineq = np.vstack(rows), np.concatenate(rhs)
        cons = [{"type": "ineq", "fun": lambda z: bineq - Aineq @ z, "jac": lambda z: -Aineq}]
        bounds = [(self.u_lo[i] / us[i], self.u_hi[i] / us[i]) for _ in range(N) for i in range(2)]
        z0 = np.clip((U0 / us).reshape(-1), [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(fun, z0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 100, "ftol": 1e-6})
        U = unpack(res.x)
        X = (base + Gm @ U.reshape(-1)).reshape(N, 2)
        return self._finish(U, X, t, t0, res.success, {"cost": float(res.fun), "iterations": int(res.nit)})


class NonlinearMPC(_MPCBase):
    """Nonlinear MPC on the tank balances (CasADi/IPOPT if available) or on any ``model`` (SciPy)."""

    def __init__(self, spec: ControlSpec, params: TankParams | None = None, model=None,
                 backend: str = "auto"):
        super().__init__(spec, params)
        self.model = model
        if model is not None and abs(float(model.dt) - spec.dt) > 1e-9:
            raise ValueError(f"the model steps {model.dt} s but the controller samples every {spec.dt} s")
        rom = getattr(model, "rom", model)
        self._rom_ca = getattr(rom, "casadi_step", None) if model is not None else None
        if backend == "auto":
            backend = "casadi" if (ca is not None and (model is None or self._rom_ca is not None)) else "scipy"
        if backend == "casadi" and (ca is None or (model is not None and self._rom_ca is None)):
            raise ValueError("the CasADi backend needs casadi and either the built-in tank model or a model with "
                             "casadi_step (a ControlledOpInf surrogate)")
        self.backend = backend
        self.name = "Nonlinear MPC" + (" (surrogate)" if model is not None else "")
        if backend == "casadi":
            self._build_casadi()

    # ---------------------------------------------------------------------------------------------- CasADi
    def _build_casadi(self):
        s, p, N = self.spec, self.p, self.spec.horizon
        opti = ca.Opti()
        X = opti.variable(2, N + 1)
        U = opti.variable(2, N)
        S = opti.variable(3, N)                      # slacks: h low, h high, T high (normalised)
        x0 = opti.parameter(2)
        up = opti.parameter(2)
        d = opti.parameter(2)
        R = opti.parameter(2, N)
        th = opti.parameter(2)                       # q_leak, eta

        def rhs(x, u):
            h = ca.fmax(x[0], 1e-4)
            q_out = p.k_out * ca.sqrt(h)
            dh = (u[0] - q_out - th[0]) / p.area
            dT = (u[0] * (d[0] - x[1]) + (th[1] * u[1] - p.ua * (x[1] - d[1])) / (p.rho * p.cp)) / \
                (p.area * ca.fmax(h, p.h_min))
            return ca.vertcat(dh, dT)

        def rk4(x, u):
            if self._rom_ca is not None:                 # surrogate: one discrete step of its own dt
                return self._rom_ca(x, ca.vertcat(u, d))
            hs = s.dt / 2
            for _ in range(2):
                k1 = rhs(x, u)
                k2 = rhs(x + hs / 2 * k1, u)
                k3 = rhs(x + hs / 2 * k2, u)
                k4 = rhs(x + hs * k3, u)
                x = x + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return x

        us = self.u_scale
        J = 0
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == rk4(X[:, k], U[:, k]))
            eh = (X[0, k + 1] - R[0, k]) / s.h_scale
            eT = (X[1, k + 1] - R[1, k]) / s.T_scale
            du = (U[:, k] - (up if k == 0 else U[:, k - 1])) / us
            J += s.w_h * eh ** 2 + s.w_T * eT ** 2 + s.w_move * ca.sumsqr(du) + s.w_energy * U[1, k] / s.Q_bounds[1]
            J += s.w_constraint * ca.sumsqr(S[:, k])
            opti.subject_to(X[0, k + 1] >= s.h_bounds[0] - S[0, k] * s.h_scale)
            opti.subject_to(X[0, k + 1] <= s.h_bounds[1] + S[1, k] * s.h_scale)
            opti.subject_to(X[1, k + 1] <= s.T_max + S[2, k] * s.T_scale)
            opti.subject_to(opti.bounded(self.u_lo, U[:, k], self.u_hi))
            opti.subject_to(opti.bounded(-self.du_max, U[:, k] - (up if k == 0 else U[:, k - 1]), self.du_max))
        opti.subject_to(ca.vec(S) >= 0)
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 200, "tol": 1e-6,
                                                    "acceptable_tol": 1e-4, "sb": "yes"})
        self._ca = {"opti": opti, "X": X, "U": U, "S": S, "x0": x0, "up": up, "d": d, "R": R, "th": th}

    def _act_casadi(self, t, x0, u_prev, d, q_leak, eta):
        c, N = self._ca, self.spec.horizon
        opti = c["opti"]
        R = self._refs(t)
        opti.set_value(c["x0"], x0)
        opti.set_value(c["up"], u_prev)
        opti.set_value(c["d"], d)
        opti.set_value(c["R"], R.T)
        opti.set_value(c["th"], [q_leak, eta])
        U0 = self.U_warm if self.U_warm is not None else np.repeat(u_prev[None], N, 0)
        opti.set_initial(c["U"], U0.T)
        Xg = [x0]
        for k in range(N):
            if self.model is None:
                Xg.append(tank_step(Xg[-1], U0[k], d, self.p, self.spec.dt, eta=eta, q_leak=q_leak))
            else:
                Xg.append(np.asarray(self.model.step(Xg[-1], np.concatenate([U0[k], d])))[:2])
        opti.set_initial(c["X"], np.array(Xg).T)
        opti.set_initial(c["S"], 0)
        try:
            sol = opti.solve()
            U, X, ok = np.array(sol.value(c["U"])).T, np.array(sol.value(c["X"])).T[1:], True
            cost = float(sol.value(opti.f))
        except RuntimeError:
            U, X, ok = np.array(opti.debug.value(c["U"])).T, np.array(opti.debug.value(c["X"])).T[1:], False
            cost = float("nan")
        return U, X, ok, {"cost": cost, "backend": "casadi/ipopt"}

    # ---------------------------------------------------------------------------------------------- SciPy
    def _rollout_batch(self, x0, Ub, d, q_leak, eta):
        """Predicted states (B, N, 2) for a batch of plans Ub (B, N, 2)."""
        B, N = Ub.shape[0], Ub.shape[1]
        X = np.repeat(np.asarray(x0, float)[None], B, 0)
        out = np.zeros((B, N, 2))
        for k in range(N):
            if self.model is None:
                X = tank_step(X.T, Ub[:, k].T, d, self.p, self.spec.dt, substeps=2, eta=eta, q_leak=q_leak).T
            else:
                V = np.column_stack([Ub[:, k], np.repeat(d[None], B, 0)])
                X = np.asarray(self.model.step(X, V))[:, :2]
            out[:, k] = X
        return out

    def _act_scipy(self, t, x0, u_prev, d, q_leak, eta):
        N = self.spec.horizon
        R = self._refs(t)
        us = self.u_scale
        U0 = self.U_warm if self.U_warm is not None else np.repeat(u_prev[None], N, 0)
        nz = 2 * N
        eps = 1e-5

        def costs(Z):
            Ub = Z.reshape(-1, N, 2) * us
            Xb = self._rollout_batch(x0, Ub, d, q_leak, eta)
            return np.array([self._cost_terms(Xb[b], Ub[b], R, u_prev) for b in range(len(Z))])

        cache = {}

        def fun_and_grad(z):
            key = z.tobytes()
            if key not in cache:
                Z = np.vstack([z, z + eps * np.eye(nz)])
                c = costs(Z)
                cache.clear()
                cache[key] = (c[0], (c[1:] - c[0]) / eps)
            return cache[key]

        D = np.zeros((N, N))
        D[np.arange(N), np.arange(N)] = 1
        D[np.arange(1, N), np.arange(N - 1)] = -1
        rows, rhs = [], []
        for i in range(2):
            Di = np.zeros((N, nz))
            Di[:, i::2] = D * us[i]
            off = np.zeros(N)
            off[0] = u_prev[i]
            rows += [Di, -Di]
            rhs += [self.du_max[i] + off, self.du_max[i] - off]
        Aineq, bineq = np.vstack(rows), np.concatenate(rhs)
        bounds = [(self.u_lo[i] / us[i], self.u_hi[i] / us[i]) for _ in range(N) for i in range(2)]
        z0 = np.clip((U0 / us).reshape(-1), [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(lambda z: fun_and_grad(z)[0], z0, jac=lambda z: fun_and_grad(z)[1], bounds=bounds,
                       constraints=[{"type": "ineq", "fun": lambda z: bineq - Aineq @ z, "jac": lambda z: -Aineq}],
                       method="SLSQP", options={"maxiter": 60, "ftol": 1e-6})
        U = res.x.reshape(N, 2) * us
        X = self._rollout_batch(x0, U[None], d, q_leak, eta)[0]
        return U, X, bool(res.success), {"cost": float(res.fun), "backend": "scipy/slsqp", "iterations": int(res.nit)}

    def act(self, t: float, reading: dict[str, float], estimate: dict):
        t0 = time.perf_counter()
        x0, q_leak, eta = _estimate_vector(estimate)
        if self.model is not None:
            q_leak, eta = 0.0, 1.0                    # the surrogate has no parameter inputs
        u_prev = self._start(reading)
        d = np.array([reading["T_in"], reading["T_amb"]])
        if self.backend == "casadi":
            U, X, ok, extra = self._act_casadi(t, x0, u_prev, d, q_leak, eta)
        else:
            U, X, ok, extra = self._act_scipy(t, x0, u_prev, d, q_leak, eta)
        return self._finish(U, X, t, t0, ok, extra)


# --------------------------------------------------------------------------------------------------------------
# Closed loop
# --------------------------------------------------------------------------------------------------------------


def make_estimator(spec: ControlSpec, params: TankParams | None = None, seed: int = 0) -> TwinRuntime:
    """The state estimator the MPCs read: an EnKF twin of the tank with unaccounted outflow and heater efficiency."""
    model = LumpedTankModel(params, dt=spec.dt, estimate=("q_leak", "eta"))
    rules = AlertRules(limits={}, param_limits={}, forecast_steps=0, drift_every=10 ** 9)
    return TwinRuntime(model, ["h", "T"], {"h": 0.003, "T": 0.05},
                       {"h": 3e-4, "T": 0.015, "q_leak": 5e-5, "eta": 3e-3}, n_ens=40, rules=rules, seed=seed)


def run_closed_loop(controller, spec: ControlSpec, scenario: ControlScenario,
                    params: TankParams | None = None) -> dict:
    """Simulate ``controller`` on ``scenario``. Returns the trajectories, the controller's plans (every 10th
    step, MPC only) and :func:`performance` metrics."""
    p = params or TankParams()
    h0, T0 = scenario.x0
    u = inputs_for_steady_state(p, h0, T0, scenario.T_in(0.0), scenario.T_amb)
    sim = TankSimulator(p, scenario.x0, scenario.faults, scenario.noise, scenario.seed)
    est = make_estimator(spec, p, scenario.seed) if getattr(controller, "uses_estimate", False) else None
    n = int(round(scenario.duration / spec.dt))
    cols = {k: [] for k in ("t", "h", "T", "h_meas", "T_meas", "q_in", "Q", "h_ref", "T_ref", "T_in", "q_leak_est",
                            "eta_est", "solve_ms")}
    plans = []
    for k in range(n):
        t = sim.t
        d = (scenario.T_in(t), scenario.T_amb)
        reading = sim.read(u, d)
        reading["t"] = t
        estimate = est.ingest(reading) if est is not None else None
        u, info = controller.act(t, reading, estimate)
        u = np.clip(u, [spec.q_in_bounds[0], spec.Q_bounds[0]], [spec.q_in_bounds[1], spec.Q_bounds[1]])
        h_ref, T_ref = spec.reference(t)
        for key, val in (("t", t), ("h", sim.x[0]), ("T", sim.x[1]), ("h_meas", reading["h"]),
                         ("T_meas", reading["T"]), ("q_in", u[0]), ("Q", u[1]), ("h_ref", h_ref), ("T_ref", T_ref),
                         ("T_in", d[0]), ("solve_ms", info.get("solve_ms", 0.0)),
                         ("q_leak_est", estimate["state"]["q_leak"] if estimate else np.nan),
                         ("eta_est", estimate["state"]["eta"] if estimate else np.nan)):
            cols[key].append(float(val))
        if info and k % 10 == 0:
            plans.append({"t": t, **{k2: info[k2] for k2 in ("plan_t", "plan_q_in", "plan_Q", "pred_h", "pred_T")}})
        sim.step(u, d, spec.dt)
    out = {k: np.array(v) for k, v in cols.items()}
    return {"controller": controller.name, "trajectory": out, "plans": plans,
            "metrics": performance(out, spec), "failed_solves": getattr(controller, "n_fail", 0)}


def performance(tr: dict[str, np.ndarray], spec: ControlSpec) -> dict:
    """Integrated absolute tracking errors (true states), constraint violations, energy, actuator travel and
    solve time of a closed-loop trajectory."""
    dt = spec.dt
    eh, eT = tr["h"] - tr["h_ref"], tr["T"] - tr["T_ref"]
    vh = np.maximum(0, spec.h_bounds[0] - tr["h"]) + np.maximum(0, tr["h"] - spec.h_bounds[1])
    vT = np.maximum(0, tr["T"] - spec.T_max)
    return {"iae_h_m_s": float(np.sum(np.abs(eh)) * dt), "iae_T_K_s": float(np.sum(np.abs(eT)) * dt),
            "rmse_h_m": float(np.sqrt(np.mean(eh ** 2))), "rmse_T_K": float(np.sqrt(np.mean(eT ** 2))),
            "max_T_K": float(tr["T"].max()), "max_violation_T_K": float(vT.max()),
            "max_violation_h_m": float(vh.max()), "time_in_violation_s": float(np.sum((vT > 0) | (vh > 0)) * dt),
            "heater_energy_kWh": float(np.sum(tr["Q"]) * dt / 3.6e6), "pumped_m3": float(np.sum(tr["q_in"]) * dt),
            "heater_travel_kW": float(np.sum(np.abs(np.diff(tr["Q"]))) / 1e3),
            "pump_travel_L_s": float(np.sum(np.abs(np.diff(tr["q_in"]))) * 1e3),
            "mean_solve_ms": float(np.mean(tr["solve_ms"])), "max_solve_ms": float(np.max(tr["solve_ms"]))}


def compare_controllers(spec: ControlSpec | None = None, scenario: ControlScenario | None = None,
                        params: TankParams | None = None, controllers: Sequence[str] = ("pid", "rules", "lmpc",
                                                                                          "nmpc"),
                        surrogate=None) -> dict:
    """Run the named controllers on the same scenario. Names: ``pid``, ``rules``, ``lmpc``, ``nmpc`` and
    ``smpc`` (nonlinear MPC on ``surrogate``, a model with ``step(x, v)``)."""
    spec = spec or ControlSpec()
    scenario = scenario or ControlScenario()
    makers = {"pid": lambda: PIDBaseline(spec, params), "rules": lambda: RuleBasedBaseline(spec),
              "lmpc": lambda: LinearMPC(spec, params), "nmpc": lambda: NonlinearMPC(spec, params),
              "smpc": lambda: NonlinearMPC(spec, params, model=surrogate)}
    unknown = set(controllers) - set(makers)
    if unknown:
        raise ValueError(f"unknown controllers {sorted(unknown)}; choose from {sorted(makers)}")
    if "smpc" in controllers and surrogate is None:
        raise ValueError("'smpc' needs a surrogate model")
    runs = {}
    for name in controllers:
        ctrl = makers[name]()
        runs[name] = run_closed_loop(ctrl, spec, scenario, params)
        runs[name]["backend"] = getattr(ctrl, "backend", None) or ("scipy/slsqp" if name == "lmpc" else "-")
    return {"mode": MODE, "spec": spec.to_dict(), "scenario": scenario.to_dict(), "runs": runs,
            "casadi_available": ca is not None}


def recommend(estimate: dict[str, float], reading: dict[str, float], spec: ControlSpec | None = None,
              params: TankParams | None = None, t: float = 0.0, method: str = "nmpc") -> dict:
    """Proposed next actions for a person to review. ``estimate``: state (``h``, ``T`` and optionally ``q_leak``,
    ``eta``); ``reading``: current ``q_in``, ``Q``, ``T_in``, ``T_amb``. Nothing is sent anywhere."""
    spec = spec or ControlSpec()
    ctrl = LinearMPC(spec, params) if method == "lmpc" else NonlinearMPC(spec, params)
    u, info = ctrl.act(t, reading, {"state": estimate})
    R = ctrl._refs(t)
    X = np.column_stack([info["pred_h"], info["pred_T"]])
    return {"mode": MODE, "controller": ctrl.name, "proposed_next": {"q_in": float(u[0]), "Q": float(u[1])},
            "plan": {k: info[k] for k in ("plan_t", "plan_q_in", "plan_Q", "pred_h", "pred_T")},
            "predicted": {"iae_h_m_s": float(np.sum(np.abs(X[:, 0] - R[:, 0])) * spec.dt),
                          "iae_T_K_s": float(np.sum(np.abs(X[:, 1] - R[:, 1])) * spec.dt),
                          "max_T": float(X[:, 1].max()), "min_h": float(X[:, 0].min()), "max_h": float(X[:, 0].max()),
                          "heater_energy_kWh": float(np.sum(info["plan_Q"]) * spec.dt / 3.6e6)},
            "solver_ok": info["solver_ok"], "solve_ms": info["solve_ms"]}
