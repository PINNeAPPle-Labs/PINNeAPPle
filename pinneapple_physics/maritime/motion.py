"""The vessel's motion in time, out of a spectrum and the transfer functions, and what a point on the deck feels.

The convention is the one linear seakeeping uses: each spectral component ``a_i`` at frequency ``f_i`` drives every mode
with the transfer function evaluated at its period, and the modes add with independent random phases. The phases of the
response are not carried over from the two-dimensional solution -- a three-dimensional hull does not have them -- so the
record reproduces the variance that the spectrum and the transfer function imply, which is what an operability criterion
and a mooring load need.

The vertical displacement of a point on the deck at ``(x forward, y starboard)`` of the centre of gravity is
``z = eta_3 + x eta_5 - y eta_4`` for small angles, and its acceleration is the second derivative of that. The crane point
of the example is off the centreline, which is what makes roll matter to a lift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spectrum as sp
from .rao import TransferFunction

__all__ = ["Motion", "Point", "response", "slamming_threshold", "vertical_response", "significant_acceleration",
           "significant_relative_velocity"]


@dataclass(frozen=True)
class Point:
    """A point on the vessel, in metres from the centre of gravity: ``x`` forward, ``y`` to starboard."""

    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class Motion:
    """One realization of the vessel's motion: the components, what each mode makes of them, and the water surface."""

    freq: np.ndarray                # Hz
    wave: np.ndarray                # m, surface elevation amplitude per component
    heave: np.ndarray               # m
    roll: np.ndarray                # rad
    pitch: np.ndarray               # rad
    phase: np.ndarray               # rad, one per component, shared by every mode
    wavenumber: np.ndarray          # 1/m

    def _wave(self, time) -> np.ndarray:
        time = np.atleast_1d(np.asarray(time, dtype=float))
        return 2.0 * math.pi * np.outer(time, self.freq) + self.phase[None, :]

    def mode(self, time, name: str) -> np.ndarray:
        """The displacement of one mode: ``heave`` in m, ``roll`` and ``pitch`` in rad."""
        amplitude = {"heave": self.heave, "roll": self.roll, "pitch": self.pitch}[name]
        return np.cos(self._wave(time)) @ amplitude

    def vertical_displacement(self, time, point: Point = Point()) -> np.ndarray:
        """``eta_3 + x eta_5 - y eta_4``: how far that point of the deck is above where it started."""
        return self.mode(time, "heave") + point.x * self.mode(time, "pitch") - point.y * self.mode(time, "roll")

    def vertical_acceleration(self, time, point: Point = Point()) -> np.ndarray:
        """The second derivative of :meth:`vertical_displacement`, in m/s^2.

        This is the number a crane limit is written in: ``0.2 g`` on a lifted load is a quarter of a tonne swinging on
        a two tonne sling.
        """
        omega2 = (2.0 * math.pi * self.freq) ** 2
        amplitude = self.heave + point.x * self.pitch - point.y * self.roll
        return -(np.cos(self._wave(time)) * omega2[None, :]) @ amplitude

    def surface_elevation(self, time, point: Point = Point()) -> np.ndarray:
        """The water surface at the point's longitudinal position, ``sum a_i cos(omega t + phi - k x)``."""
        phase = self._wave(time) - point.x * self.wavenumber[None, :]
        return np.cos(phase) @ self.wave

    def relative_velocity(self, time, point: Point = Point()) -> np.ndarray:
        """How fast the hull at ``point`` is closing on the water surface, m/s: the slamming number.

        The hull and the surface are taken **in phase**: the response ratio at each frequency is a magnitude, the phase
        a three-dimensional solve would give it is not carried here, and using the phase of the wave for both is the
        long-wave limit. What that buys is the right *trend*: where the hull follows the water the difference vanishes,
        and where it cannot -- the bow of a hull whose heave and pitch responses are out of step with the surface -- the
        relative velocity is the surface's own. What it does not give is a slam count; the events in the example are
        injected, and the criterion is Ochi's on this quantity.
        """
        omega = 2.0 * math.pi * self.freq
        phase = self._wave(time)
        config = self.heave + point.x * self.pitch - point.y * self.roll - self.wave
        return -((np.sin(phase) * omega[None, :]) @ config)

    @property
    def significant_heave(self) -> float:
        """``4 sqrt(sum (a_i |H_i|)^2 / 2)``: what a heave record built from this would measure."""
        return 4.0 * math.sqrt(float(np.sum(self.heave ** 2)) / 2.0)

    @property
    def significant_roll(self) -> float:
        return 4.0 * math.sqrt(float(np.sum(self.roll ** 2)) / 2.0)

    @property
    def significant_pitch(self) -> float:
        return 4.0 * math.sqrt(float(np.sum(self.pitch ** 2)) / 2.0)


def slamming_threshold(length: float) -> float:
    """Ochi's slamming threshold at the forward perpendicular: ``0.093 sqrt(g L)`` m/s of relative velocity."""
    return 0.093 * math.sqrt(sp.G * length)


def response(freq, density, transfer: TransferFunction, *, seed: int = 0, count: int = 96,
             depth: float | None = None) -> Motion:
    """Build the motion of a vessel from a spectrum and its transfer functions.

    Modes the transfer function does not carry are zero: a :class:`~.rao.TransferFunction` without ``pitch`` gives a
    motion with no pitch, and no error, because a heave-only question is a legitimate one.
    """
    components = sp.components(freq, density, seed=seed, count=count)
    period = 1.0 / components.freq
    wavenumber = np.asarray(sp.wavenumber(2.0 * math.pi * components.freq, depth), dtype=float)
    slope = wavenumber * components.amplitude
    heave = components.amplitude * transfer.amplitude(period, "heave")
    roll = slope * transfer.amplitude(period, "roll")
    pitch = slope * transfer.amplitude(period, "pitch") if transfer.pitch is not None else np.zeros_like(roll)
    return Motion(freq=components.freq, wave=components.amplitude, heave=heave, roll=roll, pitch=pitch,
                  phase=components.phase, wavenumber=wavenumber)


def vertical_response(freq, transfer: TransferFunction, point: Point = Point()) -> np.ndarray:
    """``omega^2 (H_3 + x H_5 - y H_4)``: the acceleration a component of unit wave amplitude produces at the point.

    ``H_4`` and ``H_5`` are per radian of wave slope, so they are multiplied by ``k = omega^2/g`` in deep water to get a
    displacement per metre of amplitude. This is the spectral route to an acceleration: no sampling, and the variance it
    gives is what a record would have to average over many realizations to reproduce.
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    period = 1.0 / np.maximum(freq, 1e-9)
    omega = 2.0 * math.pi * freq
    slope = omega ** 2 / sp.G
    heave = transfer.amplitude(period, "heave")
    roll = transfer.amplitude(period, "roll") * slope
    pitch = transfer.amplitude(period, "pitch") * slope if transfer.pitch is not None else np.zeros_like(roll)
    return omega ** 2 * (heave + point.x * pitch - point.y * roll)


def significant_relative_velocity(freq, density, transfer: TransferFunction, point: Point = Point()) -> float:
    """``4 sqrt(integral omega^2 (A - 1)^2 S df)`` -- the significant relative velocity at a point, m/s.

    ``A`` is the hull's own vertical response per metre of wave amplitude at that point, ``omega^2 (A - 1)/g`` being
    the surface's, so the bracket is zero exactly where the hull rides the water.
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    density = np.nan_to_num(np.atleast_1d(np.asarray(density, dtype=float)), nan=0.0)
    period = 1.0 / np.maximum(freq, 1e-9)
    slope = (2.0 * math.pi * freq) ** 2 / sp.G
    heave = transfer.amplitude(period, "heave")
    roll = transfer.amplitude(period, "roll") * slope
    pitch = transfer.amplitude(period, "pitch") * slope if transfer.pitch is not None else np.zeros_like(roll)
    hull = heave + point.x * pitch - point.y * roll
    return 4.0 * math.sqrt(max(float(np.trapezoid((2.0 * math.pi * freq) ** 2 * (hull - 1.0) ** 2 * density, freq)),
                               0.0))


def significant_acceleration(freq, density, transfer: TransferFunction, point: Point = Point()) -> float:
    """``4 sqrt(integral |H_acc|^2 S df)`` -- the significant vertical acceleration at a point, m/s^2.

    This is the number an operability criterion is written against: the mean of the one-third highest accelerations,
    which for a narrow-band response is four standard deviations.
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    density = np.nan_to_num(np.atleast_1d(np.asarray(density, dtype=float)), nan=0.0)
    amplitude = vertical_response(freq, transfer, point)
    return 4.0 * math.sqrt(max(float(np.trapezoid(amplitude ** 2 * density, freq)), 0.0))
