"""The detectors, compared on equal terms, and the diagnosis that names the fault.

Three detectors on the same residual stream, because the issue's question is not "can a residual find a fault" but
"what does the residual buy over the two things people already do":

* :class:`ThresholdDetector` -- a fixed threshold on each residual channel. The baseline everybody starts with.
* :class:`KalmanDetector` -- a per-channel Kalman filter on a random-walk bias, alarming when the *innovation* leaves
  its own covariance. This is the classic innovation-based fault detector, and it is a fair one: it knows the noise.
* :class:`CusumDetector` -- a one-sided CUSUM on the same residuals, the standard change detector when the fault is a
  small persistent shift rather than a spike.

All three are tuned the same way and measured the same way: **the most sensitive setting that produces no false alarm
on the calibration flights**, then the detection delay is measured on flights it never saw. Anything else is a
comparison between three different false-alarm rates.

The diagnosis is not a classifier fitted to labels: it reads *which* channel moved. A payload pushes along body ``-z``;
a motor pushes along that motor's arm and spends power; a gust is horizontal in the world frame; a GPS glitch moves the
fix and nothing else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .residuals import CHANNELS, FORCE, TORQUE
from .sim import Flight

__all__ = ["Detection", "ThresholdDetector", "KalmanDetector", "CusumDetector", "Calibration", "calibrate",
           "diagnose", "FAULT_LABELS"]

#: what each injected fault is called in the report
FAULT_LABELS = {"payload": "payload", "motor": "degraded motor", "blade": "chipped blade", "gust": "gust",
                "gps": "GPS glitch", "magnetometer": "magnetometer interference", "vibration": "IMU vibration",
                "battery": "battery sag"}


@dataclass(frozen=True)
class Calibration:
    """What the fault-free flights say each channel does.

    ``mean`` and ``std`` standardise a sample; ``allowance`` is the per-channel *reference value* a CUSUM needs, and it
    is measured rather than assumed: a fixed allowance of one sigma lets the vibration channel -- which is rectified, so
    half its samples sit at zero and the rest arrive in bursts -- fill an accumulator to fifty sigmas on a healthy
    flight. Set at each channel's own 95th percentile, a healthy flight cannot fill it and a fault still crosses it in
    a few samples.
    """

    mean: np.ndarray
    std: np.ndarray
    allowance: np.ndarray | None = None

    def sigmas(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, float) - self.mean) / np.maximum(self.std, 1e-9)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {c: {"mean": float(self.mean[i]), "std": float(self.std[i])} for i, c in enumerate(CHANNELS)}


def calibrate(flights, *, after_s: float = 2.0) -> Calibration:
    """Mean and spread of every channel over the settled part of fault-free flights."""
    blocks = [f.residual[f.time > after_s] for f in flights]
    stacked = np.concatenate(blocks, axis=0)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    standardised = (stacked - mean) / np.maximum(std, 1e-9)
    allowance = np.maximum(np.percentile(standardised, 95, axis=0), 0.5)
    return Calibration(mean, std, allowance)


@dataclass
class Detection:
    """When a detector fired, on which channels, and how strong the evidence was at that moment."""

    detected: bool
    time: float | None
    channels: tuple[str, ...] = ()
    evidence: np.ndarray | None = None

    @property
    def delay_s(self) -> float | None:
        return self.time

    def to_dict(self) -> dict:
        return {"detected": self.detected, "time": self.time, "channels": list(self.channels)}


class Detector:
    """Common ground: a calibration, a sensitivity, and a run over a flight that returns the first alarm."""

    def __init__(self, *, sensitivity: float = 5.0, consecutive: int = 5):
        self.sensitivity = float(sensitivity)
        self.consecutive = int(consecutive)
        self.calibration: Calibration | None = None

    def fit(self, calibration: Calibration) -> Detector:
        self.calibration = calibration
        return self

    def _first(self, flight: Flight, alarm: np.ndarray) -> Detection:
        """The first index where ``alarm`` holds for ``consecutive`` samples -- ``[-1]`` when it never does."""
        if not alarm.any():
            return Detection(False, None)
        run = 0
        for k, fired in enumerate(alarm):
            run = run + 1 if fired else 0
            if run >= self.consecutive:
                start = k - self.consecutive + 1
                # name the channels that carried it: the largest standardised excursions over the *next* few samples,
                # not the single one where the alarm fired. A step in the accelerometer moves the vibration channel's
                # rolling spread on the very first sample, so naming at that sample answered "vibration" for a payload.
                window = flight.residual[start:start + max(self.consecutive, 5)]
                sigmas = np.abs(self.calibration.sigmas(window.mean(axis=0))) if self.calibration else \
                    np.abs(window.mean(axis=0))
                order = np.argsort(sigmas)[::-1]
                keep = [i for i in order if sigmas[i] >= max(0.5 * sigmas[order[0]], 2.0)][:3]
                return Detection(True, float(flight.time[start]), tuple(CHANNELS[i] for i in keep))
        return Detection(False, None)

    def run(self, flight: Flight, *, after_s: float = 2.0) -> Detection:      # pragma: no cover - interface
        raise NotImplementedError


class ThresholdDetector(Detector):
    """A fixed threshold on each channel: ``|r - mean| > sensitivity * std`` for ``consecutive`` samples."""

    def run(self, flight: Flight, *, after_s: float = 2.0) -> Detection:
        assert self.calibration is not None, "fit the detector on fault-free flights first"
        sigmas = self.calibration.sigmas(flight.residual)
        alarm = np.any(np.abs(sigmas) > self.sensitivity, axis=1)
        alarm[: int(after_s / max(flight.time[1] - flight.time[0], 1e-9))] = False
        return self._first(flight, alarm)


class KalmanDetector(Detector):
    """A per-channel Kalman filter whose innovation is the statistic.

    The state is the residual's slowly varying bias (a random walk with variance ``q`` per sample, in units of the
    calibrated sigma); the measurement is the residual sample, with variance 1 in those units. A fault is a change in
    the bias, so the *normalised innovation* ``nu / sqrt(S)`` is a standard-score stream: a persistent run above
    ``sensitivity`` is an alarm. Noise that the filter already predicts does not fire it -- which is what it buys over
    the fixed threshold.
    """

    def __init__(self, *, sensitivity: float = 5.0, consecutive: int = 5, bias_drift: float = 0.02):
        super().__init__(sensitivity=sensitivity, consecutive=consecutive)
        self.q = float(bias_drift)

    def run(self, flight: Flight, *, after_s: float = 2.0) -> Detection:
        assert self.calibration is not None, "fit the detector on fault-free flights first"
        z = self.calibration.sigmas(flight.residual)
        n, m = z.shape
        settle = int(after_s / max(flight.time[1] - flight.time[0], 1e-9))
        bias = np.zeros(m)
        covariance = np.full(m, 1e-6)
        innovation = np.zeros((n, m))
        for k in range(n):
            covariance = covariance + self.q ** 2
            gain = covariance / (covariance + 1.0)
            innovation[k] = (z[k] - bias) / np.sqrt(covariance + 1.0)
            bias = bias + gain * (z[k] - bias)
            covariance = (1.0 - gain) * covariance
        alarm = np.any(np.abs(innovation) > self.sensitivity, axis=1)
        alarm[:settle] = False
        return self._first(flight, alarm)


class CusumDetector(Detector):
    """A two-sided CUSUM on the standardised residual, for the small persistent shift a chip or a payload makes.

    Each channel carries its own *allowance* -- the reference value subtracted before accumulation -- measured as the
    95th percentile of the standardised calibration data. That is not a refinement: with the textbook one-sigma
    allowance the vibration channel alone fills its accumulator to 50 sigma on a healthy flight, because it is
    rectified and arrives in bursts.
    """

    def __init__(self, *, sensitivity: float = 5.0, consecutive: int = 5, drift: float | None = None):
        super().__init__(sensitivity=sensitivity, consecutive=consecutive)
        #: the fallback allowance, used only when the detector is run without a calibration. The real one is measured
        #: per channel (:attr:`Calibration.allowance`), because a fixed value lets a rectified channel's CUSUM run
        #: away: the vibration channel reaches 50 sigma on a healthy flight at an allowance of one.
        self.drift = 1.0 if drift is None else float(drift)

    def run(self, flight: Flight, *, after_s: float = 2.0) -> Detection:
        assert self.calibration is not None, "fit the detector on fault-free flights first"
        z = self.calibration.sigmas(flight.residual)
        n, m = z.shape
        settle = int(after_s / max(flight.time[1] - flight.time[0], 1e-9))
        positive = np.zeros(m)
        negative = np.zeros(m)
        alarm = np.zeros(n, bool)
        allowance = (self.calibration.allowance if self.calibration.allowance is not None
                     else np.full(m, self.drift))
        for k in range(settle, n):
            # the accumulators start after the vehicle has settled: a CUSUM that has already been fed the take-off
            # transient alarms on the first sample it is allowed to
            positive = np.maximum(0.0, positive + z[k] - allowance)
            negative = np.maximum(0.0, negative - z[k] - allowance)
            alarm[k] = bool(np.any((positive > self.sensitivity) | (negative > self.sensitivity)))
        alarm[: int(after_s / max(flight.time[1] - flight.time[0], 1e-9))] = False
        return self._first(flight, alarm)


def tune(detector_class, calibration: Calibration, fault_free, *, sensitivities=(2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
                                                                            8.0, 10.0, 15.0, 20.0, 30.0, 40.0),
         **kwargs):
    """The most sensitive setting of ``detector_class`` that raises no alarm on the calibration flights.

    This is what makes the comparison fair: each detector is pushed to its own edge of usability and then measured on
    flights it has never seen, instead of being given three different false-alarm budgets and one headline delay.

    The grid runs to 40 on purpose. A CUSUM accumulates, so on a residual that is autocorrelated -- this one rings at a
    few hertz -- its running sum crosses a 10-sigma bar during a *healthy* flight; the setting it actually needs to stay
    quiet is far above the threshold detector's, and letting the grid stop at 10 would have reported a detector that
    only looks worse because it was not allowed to be tuned.
    """
    for sensitivity in sensitivities:
        detector = detector_class(sensitivity=sensitivity, **kwargs).fit(calibration)
        if not any(detector.run(f).detected for f in fault_free):
            return detector
    return detector_class(sensitivity=sensitivities[-1], **kwargs).fit(calibration)


# ----------------------------------------------------------------------------------------------------- the diagnosis
def _rotor_from_torque(torque_body: np.ndarray) -> int:
    """Which motor the torque residual points at.

    Thrust on an arm ``(x, y)`` makes ``tau = (-y T, x T)``, so a motor delivering *less* than commanded leaves a
    residual along ``(tau_y, -tau_x) ~ (x, y)``: the arm itself. Measured on the flight of the example, motor 2 (right
    rear) leaves ``(-22.8, -22.8)``, which is that arm and only that arm.
    """
    planar = np.array([torque_body[1], -torque_body[0]])
    if float(np.linalg.norm(planar)) < 1e-9:
        return -1
    arms = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]) / math.sqrt(2.0)
    return int(np.argmax(arms @ planar / max(float(np.linalg.norm(planar)), 1e-9)))


def diagnose(flight: Flight, detection: Detection, calibration: Calibration, *, window_s: float = 1.5) -> dict:
    """Name the fault from *which* channel moved, with the evidence in sigmas.

    Two questions, in this order. **Is it the sensor or the vehicle?** A measurement check that is the dominant evidence
    is a sensor fault -- and it has to be dominant, not merely large: the vibration channel's spread is 0.004 m/s^2, so
    a payload's step in the accelerometer also moves it a little, and a rule that only asked "did vibration exceed
    three sigma" answered "IMU vibration" for a payload. **If it is the vehicle, which part?** A vertical force with a
    quiet torque is mass; a torque is a rotor, and its direction names the motor; a planar force is the air moving.
    """
    if not detection.detected:
        return {"label": "no fault detected", "confidence": 0.0, "evidence": {}}
    t0 = detection.time
    mask = (flight.time >= t0) & (flight.time <= t0 + window_s)
    if not mask.any():
        mask = np.zeros_like(flight.time, bool)
        mask[-1] = True
    sigma = calibration.sigmas(flight.residual[mask].mean(axis=0))
    evidence = {c: float(sigma[i]) for i, c in enumerate(CHANNELS)}
    force = np.array([sigma[CHANNELS.index(c)] for c in FORCE])
    torque = np.array([sigma[CHANNELS.index(c)] for c in TORQUE])
    power = evidence["power"]
    vertical = abs(float(force[2]))
    planar = float(np.linalg.norm(force[:2]))
    spin = float(np.linalg.norm(torque))
    strength = float(np.max(np.abs(sigma)))

    sensors = {"gps": abs(evidence["gps"]), "magnetometer": abs(evidence["heading"]),
               "vibration": abs(evidence["vibration"])}
    loudest_sensor, its_evidence = max(sensors.items(), key=lambda item: item[1])
    dynamics = max(vertical, planar, spin, abs(power))
    if its_evidence > 4.0 and its_evidence > 0.5 * dynamics:
        return {"label": FAULT_LABELS[loudest_sensor], "confidence": min(its_evidence / 20.0, 1.0),
                "evidence": evidence}

    if abs(power) > 4.0 and vertical < 3.0 and planar < 3.0 and spin < 4.0:
        return {"label": FAULT_LABELS["battery"], "confidence": min(abs(power) / 12.0, 1.0), "evidence": evidence}
    if spin > 4.0:
        motor = _rotor_from_torque(torque)
        # A degraded motor and a chipped blade are *not* told apart here, and the measurement says why: the controller
        # re-trims both into "more collective thrust than the nominal model explains, plus a torque asymmetry", and the
        # power-per-newton ratio that separates them in theory (1.96 against 2.02 in the flight of the example) is
        # inside the noise. Both are reported together, with the motor named.
        label = f"motor {motor} low efficiency or blade chipped" if motor >= 0 else "a rotor is failing"
        return {"label": label, "confidence": min(spin / 25.0, 1.0), "evidence": evidence}
    # a gust is a *planar* force, but not a purely planar one: the vehicle tilts into the wind to hold position, and in
    # the body frame that tilt puts the wind's force partly along -z. Measured at 10 m/s: (-18.4, -11.3). Asking for
    # "planar overwhelmingly dominant" got exactly that case wrong and called it a payload, so the rule is dominance
    # with a real margin, not a ratio of five.
    if planar > 1.2 * vertical and planar > 3.0 and abs(power) < 4.0 and spin < 4.0:
        return {"label": FAULT_LABELS["gust"], "confidence": min(planar / 20.0, 1.0), "evidence": evidence}
    if vertical > 4.0:
        return {"label": FAULT_LABELS["payload"], "confidence": min(vertical / 20.0, 1.0), "evidence": evidence}
    return {"label": "unclassified", "confidence": min(strength / 10.0, 1.0), "evidence": evidence}
