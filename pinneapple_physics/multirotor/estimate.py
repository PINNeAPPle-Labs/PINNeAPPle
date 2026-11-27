"""Who the vehicle actually is, learned in flight -- and how long that answer can be trusted (#335).

:mod:`.residuals` measures the disagreement between a *declared* vehicle and the one flying. This module asks the
question underneath it: what does the vehicle weigh, how hard is it to turn, and how much does the air push back? The
answer is estimated **in flight**, from what a flight controller already has -- the commands it sent, the IMU, the GPS
velocity, the pack telemetry -- by recursive least squares on the same 6-DoF model, and then measured against the
truth of the simulator.

The three pieces, in the order the experiment uses them:

* :class:`OnlineExpert` -- mass, inertia tensor and drag area, from a forgetting-factor RLS whose regressor is the
  Newton-Euler equation itself. *What is identifiable, and when*, is not assumed: it is the structure of the equation.
  In a hover only the thrust row carries information, so only the mass is identified; the drag area needs airspeed and
  the inertia needs angular acceleration, so a hover that identifies the mass to 1 % still has no idea what the
  inertia is. That is a property of the physics, not of the estimator, and the recursive covariance says it in
  numbers.
* :class:`ResidualCorrection` -- a small learned model of what the nominal model does *not* have: **wind** and
  **ground effect**. Fitted on flights flying in conditions the nominal model does not describe, and measured on
  flights it never saw.
* :func:`confidence_horizon` -- the number the issue asks for, in seconds: *this model answers for N seconds before
  its residual crosses the threshold*. It is built from the two things that actually move: the parameter covariance
  (poor excitation means a wider answer, which means less time) and the residual's own drift (a fault growing means
  less time still).

Everything here runs on the recorded sensor stream of a :class:`~pinneapple_physics.multirotor.sim.Flight` and never
looks ahead: ``run`` is the replay of a recursion that was causal sample by sample, which is what makes its answer
something a flight controller could have produced at that instant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import QUAD_X, Airframe, RotorTable
from .residuals import CHANNELS, FORCE

__all__ = ["RHO", "REFERENCE_SPEED_M_S", "RecursiveLeastSquares", "ParameterEstimate", "EstimateHistory", "OnlineExpert",
           "ResidualCorrection", "GE_DECAYS", "Horizon", "confidence_horizon", "quaternion_matrix"]

RHO = 1.225
#: the decay lengths of the ground-effect basis, in metres. Two terms instead of the declared ``h0 = 0.35 m`` on
#: purpose: the fit is not handed the answer it is supposed to find, only a basis wide enough to contain it.
GE_DECAYS = (0.25, 0.50)
#: the airspeed a model is quoted at when its uncertainty is turned into a horizon. Five metres per second is what the
#: 1.5 kg quad of the example reaches in its circle; the number matters only in that every flight is asked the same one.
REFERENCE_SPEED_M_S = 5.0


def quaternion_matrix(q) -> np.ndarray:
    """Body-to-world rotation of ``(w, x, y, z)``, the same convention as :class:`~pinneapple_physics.multirotor.model.State`."""
    w, x, y, z = np.asarray(q, float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class RecursiveLeastSquares:
    """Forgetting-factor least squares, with the parameter vector normalised so one prior covers all of it.

    ``theta`` is dimensionless by construction -- a mass *ratio*, an inertia *ratio*, a drag area per unit mass over
    the nominal -- so a single prior uncertainty is meaningful and ``theta = 1`` means "the declared vehicle". The
    forgetting factor is what keeps the filter alive: it multiplies the covariance by ``1 / forget`` before every
    update, so a direction the flight stops exciting gets its uncertainty back at a known rate, which is the rate
    :func:`confidence_horizon` turns into seconds.

    The growth is bounded at the **prior**, and the bound is the point. Forgetting exists so that a parameter which
    *was* identified can follow a change; it must not manufacture doubt about one the flight has never seen, because
    the prior already says how unknown that is, and there is nothing that can say more. Left unbounded, the vertical
    drag area -- which no flight of this vehicle ever excites -- reaches several times its own prior in twenty seconds
    and makes the model's quoted uncertainty meaningless. The bound is applied as one scale factor on the whole matrix,
    which keeps it positive definite (a directional forgetting would be the production answer, and is deliberately not
    what this is).
    """

    def __init__(self, initial, uncertainty: float = 0.5, *, forget: float = 0.999):
        self.theta = np.array(initial, float)
        self.prior_covariance = np.eye(len(self.theta)) * float(uncertainty) ** 2
        self.covariance = self.prior_covariance.copy()
        self.forget = float(forget)

    def update(self, design, target, noise) -> np.ndarray:
        """One block update: ``design`` is ``(k, n)``, ``target`` ``(k,)``, ``noise`` the standard deviation of each row."""
        phi = np.asarray(design, float)
        if phi.size == 0:
            self.covariance = self._bounded(self.covariance / self.forget)
            return self.theta
        prior = self.covariance / self.forget
        innovation_covariance = phi @ prior @ phi.T + np.diag(np.asarray(noise, float) ** 2)
        gain = prior @ phi.T @ np.linalg.inv(innovation_covariance)
        self.theta = self.theta + gain @ (np.asarray(target, float) - phi @ self.theta)
        # Joseph-free covariance update is fine here: the design is well conditioned in the directions that carry data
        # and the prior keeps the rest positive definite
        covariance = prior - gain @ phi @ prior
        self.covariance = self._bounded(0.5 * (covariance + covariance.T))
        return self.theta

    def _bounded(self, covariance: np.ndarray) -> np.ndarray:
        """Refuse to be more uncertain than the prior allowed, in any direction."""
        allowed = np.diag(self.prior_covariance)
        ratio = float(np.max(np.diag(covariance) / np.maximum(allowed, 1e-12)))
        return covariance / ratio if ratio > 1.0 else covariance

    def sigma(self, design) -> np.ndarray:
        """The standard deviation of the prediction ``design @ theta``, in the units of the target."""
        phi = np.asarray(design, float)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", phi, self.covariance, phi), 0.0))


class _LowPass:
    """One first-order filter per channel, at the *motor's* time constant, applied to both sides of the equation.

    The rotor speed the ESC reports is the *command*; the speed the propeller is turning is that command put through a
    first-order lag of ``motor_tau_s`` (:class:`~pinneapple_physics.multirotor.model.Multirotor` integrates exactly
    that). Filtering the accelerometer, the gyro and the velocity at the same time constant filters both sides of the
    Newton-Euler equation by the same ``1 / (1 + tau s)``, so the lag cancels to first order instead of biasing the
    thrust the estimator divides by. Without it, the commanding signal and the flown thrust differ by
    ``2 tau omega_dot`` during a maneuver, which is several per cent on the parameter that multiplies the thrust.
    """

    def __init__(self, tau: float, channels: int):
        self.tau = float(tau)
        self.state: list[np.ndarray | None] = [None] * int(channels)

    def reset(self) -> None:
        self.state = [None] * len(self.state)

    def __call__(self, i: int, value, dt: float) -> np.ndarray:
        value = np.asarray(value, float)
        if self.state[i] is None:
            self.state[i] = value.copy()
            return self.state[i]
        if self.tau <= 0.0:
            self.state[i] = value.copy()
            return self.state[i]
        alpha = min(dt / (self.tau + dt), 1.0)
        self.state[i] = self.state[i] + alpha * (value - self.state[i])
        return self.state[i]


@dataclass(frozen=True)
class ParameterEstimate:
    """What the vehicle is, as of one instant, with the uncertainty the filter carries."""

    time: float
    mass_kg: float
    inertia: np.ndarray                  # kg m^2, body axes
    drag_area: np.ndarray                # Cd * A, m^2, body axes
    mass_std_kg: float
    inertia_std: np.ndarray
    drag_std: np.ndarray
    model_sigma_m_s2: float              # the 1-sigma error of the predicted specific force, m/s^2

    def relative_mass_error(self, truth_kg: float) -> float:
        return self.mass_kg / truth_kg - 1.0

    def to_dict(self) -> dict:
        return {"time": self.time, "mass_kg": self.mass_kg, "inertia": [float(v) for v in self.inertia],
                "drag_area": [float(v) for v in self.drag_area], "mass_std_kg": self.mass_std_kg,
                "inertia_std": [float(v) for v in self.inertia_std],
                "drag_std": [float(v) for v in self.drag_std],
                "model_sigma_m_s2": self.model_sigma_m_s2}


@dataclass
class EstimateHistory:
    """Every estimate the expert produced over one flight, one per sample."""

    time: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray                  # (n, 3)
    drag: np.ndarray                     # (n, 3)
    mass_std: np.ndarray
    inertia_std: np.ndarray
    drag_std: np.ndarray
    model_sigma: np.ndarray
    truth_mass_kg: float | None = None
    truth_inertia: tuple[float, float, float] | None = None
    truth_drag_area: tuple[float, float, float] | None = None

    def __len__(self) -> int:
        return len(self.time)

    def index(self, t: float) -> int:
        return int(min(max(np.searchsorted(self.time, t), 0), len(self.time) - 1))

    def at(self, t: float) -> ParameterEstimate:
        k = self.index(t)
        return ParameterEstimate(float(self.time[k]), float(self.mass[k]), self.inertia[k].copy(),
                                 self.drag[k].copy(), float(self.mass_std[k]), self.inertia_std[k].copy(),
                                 self.drag_std[k].copy(), float(self.model_sigma[k]))

    def settle_time(self, *, tolerance: float = 0.05, after_s: float = 0.0) -> float | None:
        """The first time the mass is within ``tolerance`` of the truth and stays there, or ``None``."""
        if self.truth_mass_kg is None:
            return None
        inside = np.abs(self.mass / self.truth_mass_kg - 1.0) <= tolerance
        inside &= self.time >= after_s
        # the *last* excursion out of the band is what matters: "settled" cannot be a lucky sample
        outside = np.nonzero(~inside)[0]
        if outside.size == 0:
            return float(self.time[0]) if len(self.time) else None
        if outside[-1] + 1 >= len(self.time):
            return None
        return float(self.time[outside[-1] + 1])

    def to_dict(self, *, truth: bool = True) -> dict:
        out = {"time": [float(v) for v in self.time], "mass_kg": [float(v) for v in self.mass],
               "inertia": [[float(v) for v in row] for row in self.inertia],
               "drag_area": [[float(v) for v in row] for row in self.drag],
               "mass_std_kg": [float(v) for v in self.mass_std],
               "model_sigma_m_s2": [float(v) for v in self.model_sigma]}
        if truth and self.truth_mass_kg is not None:
            out["truth"] = {"mass_kg": self.truth_mass_kg, "inertia": list(self.truth_inertia or ()),
                            "drag_area": list(self.truth_drag_area or ())}
        return out


class OnlineExpert:
    """Mass, inertia and drag area of the vehicle, recursively, from the sensors a flight controller has.

    The regression is the force and the torque equation of :mod:`.model`, written so that it is **linear in the
    parameters**, which is what makes recursive least squares the right tool instead of a search:

    *``force``* -- the accelerometer measures the specific force ``f = (-T z + drag) / m``. With the thrust ``T`` known
    from the reported rotor speed (BEMT) and the drag quadratic in the body velocity, that is linear in
    ``mu = m_nom / m`` and in ``gamma_i = CdA_i / m``; both are estimated, and the mass and the drag area are read back
    out of them (:math:`CdA_i = gamma_i m / m_{nom} \\cdot CdA_{i,nom}`).

    *``torque``* -- ``I d(omega)/dt = tau - omega x I omega`` is linear in the three diagonal inertias; the gyroscopic
    term is evaluated at the current estimate, which is the only approximation in the fit and is a 3 % term on this
    vehicle's rates.

    Which of them the flight can actually identify is decided by the regressor, not by this class: hover leaves only
    the thrust row nonzero, so only the mass converges; the drag needs airspeed, and the inertia needs angular
    acceleration, which is why :class:`~pinneapple_physics.multirotor.control.Trajectory` grew a ``weave``.
    """

    #: the order of the parameter vector: mass ratio, three drag-area ratios, three inertia ratios
    PARAMETERS = ("mass", "drag_x", "drag_y", "drag_z", "inertia_x", "inertia_y", "inertia_z")

    def __init__(self, airframe: Airframe | None = None, *, table: RotorTable | None = None, forget: float = 0.999,
                 mass_factor: float = 2.0, inertia_factor: float = 2.0, drag_factor: float = 4.0,
                 prior_uncertainty: float = 0.5, accel_noise_m_s2: float = 0.02, angular_noise_rad_s2: float = 0.30,
                 angular_window_s: float = 0.04):
        self.airframe = airframe or Airframe()
        self.table = table or RotorTable(self.airframe.rotor, omega_max=self.airframe.omega_max * 1.2)
        self.forget = float(forget)
        self.mass_factor = float(mass_factor)
        self.inertia_factor = float(inertia_factor)
        self.drag_factor = float(drag_factor)
        self.prior_uncertainty = float(prior_uncertainty)
        self.accel_noise = float(accel_noise_m_s2)
        self.angular_noise = float(angular_noise_rad_s2)
        self.angular_window_s = float(angular_window_s)
        self.nominal_inertia = np.array(self.airframe.inertia, float)
        self.nominal_drag = np.array(self.airframe.drag_area_m2, float)
        self.nominal_mass = float(self.airframe.mass_kg)
        # the arm and spin signs of the mixer, read from the geometry the airframe declares instead of written twice
        self.arms = np.array([[v[0], v[1]] for v in QUAD_X.values()]) * self.airframe.arm_m / math.sqrt(2.0)
        self.spins = np.array([v[2] for v in QUAD_X.values()])
        self.reset()

    # --------------------------------------------------------------------------------------------------- the filter
    def reset(self) -> None:
        """Back to the prior: a vehicle the user believes is ``mass_factor`` times heavier than it is."""
        mu = 1.0 / self.mass_factor
        initial = [mu, self.drag_factor * mu, self.drag_factor * mu, self.drag_factor * mu,
                   self.inertia_factor, self.inertia_factor, self.inertia_factor]
        self.rls = RecursiveLeastSquares(initial, self.prior_uncertainty, forget=self.forget)
        self.lowpass = _LowPass(self.airframe.motor_tau_s, 4)
        self._gyro: list[np.ndarray] = []
        self._time = 0.0
        self._window = 1

    # ---------------------------------------------------------------------------------------------- one sample
    def update(self, accel, gyro, velocity, attitude, rotors, dt: float) -> ParameterEstimate:
        """One sample of the recursion. Returns the estimate *after* this sample, which is what a live estimator
        would have printed at that instant -- it uses this sample and nothing later."""
        dt = float(dt)
        self._time += dt
        rotation = quaternion_matrix(attitude)
        accel = self.lowpass(0, accel, dt)
        gyro = self.lowpass(1, gyro, dt)
        world_velocity = self.lowpass(2, velocity, dt)
        command = self.lowpass(3, rotors, dt)
        velocity_body = rotation.T @ world_velocity
        climb = -float(velocity_body[2])

        thrust = self.table.thrust_many(command, climb)
        spin_torque = self.table.torque_many(command, climb)
        total_thrust = float(np.sum(thrust))

        # --- the force equation: 3 rows, the columns being (mu, gamma_x, gamma_y, gamma_z)
        kappa = 0.5 * RHO * self.nominal_drag / self.nominal_mass
        force_design = np.zeros((3, 7))
        force_design[0, 1] = -kappa[0] * velocity_body[0] * abs(velocity_body[0])
        force_design[1, 2] = -kappa[1] * velocity_body[1] * abs(velocity_body[1])
        force_design[2, 0] = -total_thrust / self.nominal_mass
        force_design[2, 3] = -kappa[2] * velocity_body[2] * abs(velocity_body[2])

        # --- the torque equation: 3 rows, the columns being the three inertia ratios
        angular = self._angular_acceleration(gyro, dt)
        target = np.asarray(accel, float)
        design = force_design
        noise = np.full(3, self.accel_noise)
        if angular is not None:
            # the couple the commanded rotors make, and the gyroscopic term at the current estimate: ``I omega_dot =
            # tau - omega x I omega`` is linear in the three diagonal inertias once that term is taken as known
            torque_design = np.zeros((3, 7))
            for i in range(3):
                torque_design[i, 4 + i] = self.nominal_inertia[i] * angular[i]
            inertia = self.rls.theta[4:7] * self.nominal_inertia
            gyroscopic = np.array([(inertia[2] - inertia[1]) * gyro[1] * gyro[2],
                                   (inertia[0] - inertia[2]) * gyro[2] * gyro[0],
                                   (inertia[1] - inertia[0]) * gyro[0] * gyro[1]])
            couple = np.array([-float(self.arms[:, 1] @ thrust), float(self.arms[:, 0] @ thrust),
                               -float(self.spins @ spin_torque)])
            design = np.vstack([force_design, torque_design])
            target = np.concatenate([target, couple - gyroscopic])
            noise = np.concatenate([noise, self.nominal_inertia * self.angular_noise])
        self.rls.update(design, target, noise)
        return self._estimate()

    def observe(self, sensors, dt: float) -> ParameterEstimate:
        """The same update, from a :class:`~pinneapple_physics.multirotor.residuals.Sensors` sample."""
        return self.update(sensors.accel, sensors.gyro, sensors.velocity, sensors.attitude, sensors.rotors, dt)

    def _angular_acceleration(self, gyro: np.ndarray, dt: float) -> np.ndarray | None:
        """``d(omega)/dt``, from a least-squares slope over the last ``angular_window_s``: a finite difference of the
        raw gyro at 500 Hz has three times the noise of this fit, and the parameter it feeds divides by it."""
        self._window = max(3, int(round(self.angular_window_s / max(dt, 1e-9))))
        self._gyro.append(np.asarray(gyro, float).copy())
        if len(self._gyro) > self._window:
            self._gyro.pop(0)
        if len(self._gyro) < self._window:
            return None
        history = np.array(self._gyro)
        centre = (len(history) - 1) / 2.0
        offsets = np.arange(len(history)) - centre
        return (offsets @ (history - history.mean(axis=0))) / float(offsets @ offsets) / dt

    def _estimate(self) -> ParameterEstimate:
        theta = self.rls.theta
        covariance = self.rls.covariance
        mu = max(float(theta[0]), 1e-6)
        mass = self.nominal_mass / mu
        # d(m)/d(mu) = -m/mu, so the mass error is the ratio's error times the mass over the ratio
        mass_std = mass / mu * math.sqrt(max(covariance[0, 0], 0.0))
        inertia = theta[4:7] * self.nominal_inertia
        inertia_std = np.sqrt(np.maximum(np.diag(covariance)[4:7], 0.0)) * self.nominal_inertia
        drag = theta[1:4] / mu * self.nominal_drag
        # CdA = gamma/mu * CdA_nom: the two errors add in quadrature, each with its own derivative
        drag_std = self.nominal_drag * np.sqrt(
            np.maximum(np.diag(covariance)[1:4], 0.0) / mu ** 2
            + (theta[1:4] / mu ** 2) ** 2 * max(covariance[0, 0], 0.0))
        return ParameterEstimate(self._time, mass, inertia, drag, mass_std, inertia_std, drag_std,
                                 float(np.sqrt(np.mean(self.rls.sigma(self.reference_force) ** 2))))

    @property
    def reference_force(self) -> np.ndarray:
        """The force regressor at the envelope the model is quoted at -- and the reason the horizon is a number.

        The uncertainty of a *prediction* depends on the condition it is asked to predict in, and at hover the drag
        columns of the regressor are zero, so a hover would look confident no matter how little it knows about the
        drag. Asking every flight the same question -- "what specific force would you predict at
        :data:`REFERENCE_SPEED_M_S`?" -- is what makes the answer comparable between a straight, slow flight and a
        maneuver, instead of favourable to whichever one moved less.
        """
        kappa = 0.5 * RHO * self.nominal_drag / self.nominal_mass
        design = np.zeros((3, 7))
        design[0, 1] = -kappa[0] * REFERENCE_SPEED_M_S ** 2
        design[1, 2] = -kappa[1] * REFERENCE_SPEED_M_S ** 2
        design[2, 0] = -self.airframe.weight_n / self.nominal_mass
        design[2, 3] = -kappa[2] * REFERENCE_SPEED_M_S ** 2
        return design

    # ---------------------------------------------------------------------------------------- a whole flight
    def run(self, flight) -> EstimateHistory:
        """Replay the recursion over a recorded flight: every sample once, in order, and nothing ahead of itself."""
        self.reset()
        if flight.accel is None:
            raise ValueError("this flight carries no sensor stream; fly it with pinneapple_physics.multirotor.sim.fly")
        dt = float(flight.time[1] - flight.time[0]) if len(flight.time) > 1 else 0.002
        n = len(flight.time)
        mass = np.zeros(n)
        inertia = np.zeros((n, 3))
        drag = np.zeros((n, 3))
        mass_std = np.zeros(n)
        inertia_std = np.zeros((n, 3))
        drag_std = np.zeros((n, 3))
        sigma = np.zeros(n)
        for k in range(n):
            estimate = self.update(flight.accel[k], flight.gyro[k], flight.velocity[k], flight.attitude[k],
                                   flight.rotors[k], dt)
            mass[k] = estimate.mass_kg
            inertia[k] = estimate.inertia
            drag[k] = estimate.drag_area
            mass_std[k] = estimate.mass_std_kg
            inertia_std[k] = estimate.inertia_std
            drag_std[k] = estimate.drag_std
            sigma[k] = estimate.model_sigma_m_s2
        return EstimateHistory(flight.time.copy(), mass, inertia, drag, mass_std, inertia_std, drag_std, sigma,
                               truth_mass_kg=flight.truth_mass_kg,
                               truth_inertia=flight.truth_inertia,
                               truth_drag_area=flight.truth_drag_area)


@dataclass(frozen=True)
class ResidualCorrection:
    """What the nominal model does not have -- a wind, and the ground pushing the thrust up -- learned from flights.

    The residual of :mod:`.residuals` is a disagreement, and two of its causes are not faults at all: a steady wind and
    the ground effect near the floor are *physics the declared vehicle does not contain*. Both are small, both are
    persistent, and a detector cannot tell either from a fault, so they are modelled here and subtracted.

    The two are not the same kind of unknown, and the difference is the whole result of this class:

    * the **ground effect** is a property of the *vehicle*: a gain on the thrust that decays with height. It is fitted
      once, over flights that fly near the floor, and it transfers to altitudes and seeds it never saw -- which is what
      makes it a fitted model rather than a fitted number. The basis is two decay lengths (:data:`GE_DECAYS`) instead
      of the ``h0 = 0.35 m`` the airframe declares, so the fit is handed a basis and not the answer, and the ``k`` it
      recovers is compared with the ``0.15`` the simulator actually used.
    * the **wind** is a property of the *day*, not of the vehicle, and on a vehicle with no airspeed sensor it is not
      transferable at all: two different wind speeds are two different constant forces, and one coefficient fitted on
      both describes neither. That is measured, not assumed (:meth:`fit` with ``pool_wind=True`` is that model, and it
      makes a calm flight *worse*). So the wind is a **state**, estimated causally per flight by the same recursive
      least squares the expert uses (:meth:`wind_estimate`): at every sample the estimate uses only samples before it,
      and it converges in the first seconds of the flight it is asked about -- including on wind speeds it has never
      seen, because it is estimating the wind and not a coefficient of it.
    """

    #: the ground-effect gains, one per decay length in :data:`GE_DECAYS`, on the thrust-proportional specific force
    ground_effect: np.ndarray
    #: the constant world-frame force per unit mass fitted across *all* the training flights at once. Only used by the
    #: pooled model, which is kept because measuring why it fails is part of the answer.
    pooled_wind: np.ndarray | None = None
    #: the standard deviation of what is left on the training flights, m/s^2
    residual_m_s2: float = 0.0
    samples: int = 0

    # ------------------------------------------------------------------------------------------------- the basis
    @staticmethod
    def basis(flight, airframe: Airframe) -> tuple[np.ndarray, np.ndarray]:
        """``(world, ground)``: the ``(n, 3, 3)`` map from a world force to the body, and the ``(n, len(GE_DECAYS))``
        ground-effect basis on the body ``z`` axis."""
        n = len(flight.time)
        rotation = np.array([quaternion_matrix(flight.attitude[k]) for k in range(n)]) if flight.attitude is not None \
            else np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        world = np.transpose(rotation, (0, 2, 1))                       # R^T: a world force, in the body frame
        table = RotorTable(airframe.rotor, omega_max=airframe.omega_max * 1.2)
        command = flight.rotors if flight.rotors is not None else np.zeros((n, 4))
        specific_thrust = np.array([float(np.sum(table.thrust_many(command[k]))) / airframe.mass_kg
                                    for k in range(n)])
        height = np.maximum(flight.altitude, 0.0)
        ground = np.zeros((n, len(GE_DECAYS)))
        for j, decay in enumerate(GE_DECAYS):
            # the sign that makes the recovered coefficient read as the ``k`` of ``1 + k exp(-h / h0)``: the model has
            # to predict *more* downward specific force than the flat-ground BEMT does, so the basis is negative
            ground[:, j] = -np.exp(-height / decay) * specific_thrust
        return world, ground

    # ----------------------------------------------------------------------------------------------- the fit
    @classmethod
    def fit(cls, flights, *, airframe: Airframe | None = None, ridge: float = 1e-8, pool_wind: bool = False,
            after_s: float = 2.0) -> ResidualCorrection:
        """Fit the ground-effect gain on ``flights``, and nothing else.

        With ``pool_wind=True`` a single constant world force is fitted across every flight as well -- the model that
        treats the wind as a coefficient of the vehicle. It is the obvious thing to do and it does not work; the example
        measures it on flights it never saw.
        """
        airframe = airframe or Airframe()
        columns, targets = [], []
        for flight in flights:
            if flight.attitude is None:
                raise ValueError("this flight carries no sensor stream; fly it with sim.fly to fit a correction")
            world, ground = cls.basis(flight, airframe)
            mask = flight.time >= after_s
            block = np.zeros((int(mask.sum()), 3, 3 + len(GE_DECAYS)))
            block[:, :, :3] = world[mask]
            block[:, 2, 3:] = ground[mask]
            columns.append(block)
            targets.append(-flight.residual[mask][:, :3])
        design = np.concatenate(columns, axis=0)
        target = np.concatenate(targets, axis=0)
        if not pool_wind:
            # the world-force columns are dropped rather than fitted: an unconstrained constant force would absorb the
            # mean of whatever conditions the training set happened to contain, and the ground-effect rows are the only
            # ones this model is allowed to learn
            design = np.zeros_like(design)
            design[:, 2, 3:] = np.concatenate([cls.basis(f, airframe)[1][f.time >= after_s] for f in flights], axis=0)
        rows = design.reshape(-1, design.shape[-1])
        values = target.reshape(-1)
        coefficients = np.linalg.solve(rows.T @ rows + ridge * np.eye(rows.shape[1]), rows.T @ values)
        residual = target - np.einsum("nij,j->ni", design, coefficients)
        return cls(ground_effect=coefficients[3:], pooled_wind=(coefficients[:3] if pool_wind else None),
                   residual_m_s2=float(np.sqrt(np.mean(residual ** 2))), samples=int(len(values)))

    # --------------------------------------------------------------------------------------------- the wind
    def wind_estimate(self, flight, *, airframe: Airframe | None = None, forget: float = 0.995,
                      uncertainty: float = 3.0, after_s: float = 2.0) -> np.ndarray:
        """The world-frame force per unit mass the flight is feeling, estimated **causally**, ``(n, 3)``, m/s^2.

        One row per sample, three parameters, and the regressor is the rotation itself: a constant force in the world is
        ``R^T F`` in the body, so the residual's three force channels are a linear measurement of ``F`` at every instant.
        The forgetting factor is 0.995 at the campaign's 200 Hz -- a two-second memory, because the thing being
        estimated is the day's weather and not the vehicle's name.
        """
        airframe = airframe or Airframe()
        world, ground = self.basis(flight, airframe)
        target = -flight.residual[:, :3] - np.einsum("nj,j->n", ground, self.ground_effect)[:, None] \
            * np.array([0.0, 0.0, 1.0])
        rls = RecursiveLeastSquares(np.zeros(3), uncertainty, forget=forget)
        estimate = np.zeros((len(flight.time), 3))
        settled = int(after_s / max(float(flight.time[1] - flight.time[0]), 1e-9))
        for k in range(len(flight.time)):
            if k < settled:
                # before the flight is quoted as beginning, the wind estimate is the prior and is not applied: a
                # correction that has not converged would make the first seconds of every flight worse
                estimate[k] = 0.0
                continue
            rls.update(world[k], target[k], np.full(3, 0.02))
            estimate[k] = rls.theta
        return estimate

    # -------------------------------------------------------------------------------------------- the output
    def correction(self, flight, *, airframe: Airframe | None = None, wind: np.ndarray | str | None = "causal"
                   ) -> np.ndarray:
        """The body specific force to *add* to the nominal prediction, ``(n, 3)``, m/s^2.

        ``wind`` selects what is used for the wind: ``"causal"`` (the per-flight estimate of :meth:`wind_estimate`),
        ``None`` (nothing), or a fixed world-frame vector.
        """
        airframe = airframe or Airframe()
        world, ground = self.basis(flight, airframe)
        out = np.zeros((len(flight.time), 3))
        out[:, 2] = ground @ np.asarray(self.ground_effect, float)
        if isinstance(wind, str) and wind == "causal":
            wind = self.wind_estimate(flight, airframe=airframe)
        if wind is not None:
            fixed = np.asarray(wind, float)
            out += fixed @ world if fixed.ndim == 1 else fixed
        return out

    def apply(self, flight, *, airframe: Airframe | None = None, wind: np.ndarray | str | None = "causal"
              ) -> np.ndarray:
        """The residual with the correction added to it: the same array, ``(n, len(CHANNELS))``, corrected."""
        corrected = flight.residual.copy()
        corrected[:, :3] = corrected[:, :3] + self.correction(flight, airframe=airframe, wind=wind)
        return corrected

    def to_dict(self) -> dict:
        out = {"ground_effect_gain": {f"{decay:g} m": float(c)
                                      for decay, c in zip(GE_DECAYS, self.ground_effect, strict=False)},
               "ground_effect_total": float(np.sum(self.ground_effect)),
               "residual_m_s2": self.residual_m_s2, "samples": self.samples}
        if self.pooled_wind is not None:
            out["pooled_wind_m_s2"] = [float(c) for c in self.pooled_wind]
        return out


@dataclass(frozen=True)
class Horizon:
    """How long the active model answers for, in the unit the question was asked in: seconds."""

    seconds: float
    level_m_s2: float            # the force residual the model is carrying now, m/s^2
    rate_m_s2_s: float           # how fast that level is growing, m/s^2 per second
    threshold_m_s2: float        # the level at which a fault is declared
    sigma_m_s2: float            # the model's own uncertainty, from the parameter covariance
    capped: bool = False         # True when the drift was below the floor and the answer is the cap

    def to_dict(self) -> dict:
        return {"seconds": self.seconds, "level_m_s2": self.level_m_s2, "rate_m_s2_s": self.rate_m_s2_s,
                "threshold_m_s2": self.threshold_m_s2, "sigma_m_s2": self.sigma_m_s2, "capped": self.capped}


def confidence_horizon(flight, history: EstimateHistory | None, calibration, *, index: int | None = None,
                       sigmas: float = 8.0, window_s: float = 1.0, margin_sigmas: float = 1.0,
                       min_rate_m_s2_s: float = 0.01, cap_s: float = 60.0,
                       threshold_m_s2: float | None = None, sigma_m_s2=None) -> Horizon:
    """*This model answers for N seconds before its residual crosses the threshold.*

    Three measured numbers, one division:

    * the **level** -- the norm of the force residual now, less its calibrated mean. It is what the model is already
      carrying, so a fault that has just landed eats the horizon in one bite;
    * the **rate** -- the least-squares slope of that level over the last ``window_s``. A fault that grows is a positive
      slope, and the horizon is the time for the level to reach the threshold at that slope;
    * the **sigma** -- the model's own uncertainty, propagated from the parameter covariance through the regressor
      (:class:`OnlineExpert`). It is subtracted as a margin, which is what makes the number depend on the *flight* and
      not only on the residual: a hover identifies the mass and nothing else, so the same vehicle answers for less time
      after a hover than after a maneuver.

    When the level is already over the threshold the horizon is zero -- the model has expired, which is the honest
    answer for a step. When nothing is drifting, the answer is the cap and ``capped`` says so.
    """
    force = np.array([CHANNELS.index(c) for c in FORCE])
    values = flight.residual[:, force] - np.asarray(calibration.mean, float)[force]
    dt = float(flight.time[1] - flight.time[0]) if len(flight.time) > 1 else 0.002
    k = (history.index(float(flight.time[-1] if index is None else flight.time[index]))
         if history is not None else len(flight.time) - 1)
    if index is not None:
        k = min(max(int(index), 0), len(flight.time) - 1)
    # the level, smoothed at a quarter of the window it is then differentiated over: the norm of a residual that rings
    # at a few hertz swings by a factor of two inside one window, and an unsmoothed slope is a statement about the
    # ringing rather than about the model drifting
    level = np.zeros(len(flight.time))
    smooth = 0.0
    alpha = min(dt / (0.25 * window_s + dt), 1.0)
    for i, raw in enumerate(np.linalg.norm(values, axis=1)):
        smooth = raw if i == 0 else smooth + alpha * (raw - smooth)
        level[i] = smooth
    width = max(3, int(round(window_s / max(dt, 1e-9))))
    start = max(k - width + 1, 0)
    window = level[start:k + 1]
    offsets = np.arange(len(window)) - (len(window) - 1) / 2.0
    # the slope is per second, not per sample: the offsets are counted in samples and the answer is asked in seconds
    slope = float(offsets @ (window - window.mean()) / max(float(offsets @ offsets), 1e-9)) / max(dt, 1e-9)
    now = float(window[-1])
    threshold = float(threshold_m_s2) if threshold_m_s2 is not None else \
        float(sigmas * np.max(np.asarray(calibration.std, float)[force]))
    if sigma_m_s2 is None:
        if history is None:
            raise ValueError("pass a history (from OnlineExpert.run) or the model uncertainty explicitly")
        sigma = float(history.model_sigma[k])
    else:
        values = np.asarray(sigma_m_s2, float)
        sigma = float(values[k]) if values.ndim else float(values)
    headroom = threshold - now - margin_sigmas * sigma
    if headroom <= 0.0:
        return Horizon(0.0, now, slope, threshold, sigma)
    if slope <= min_rate_m_s2_s:
        return Horizon(float(cap_s), now, slope, threshold, sigma, capped=True)
    return Horizon(float(min(headroom / slope, cap_s)), now, slope, threshold, sigma)
