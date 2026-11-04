"""A ground radar: range, azimuth and elevation of a target, with Gaussian noise and a coverage limit.

Textbook measurement model with illustrative numbers; simulation only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Radar:
    """A radar site and its measurement model.

    ``site`` [m]; ``sigma_range`` [m]; ``sigma_angle`` [rad] (azimuth and elevation); ``rate_hz`` [1/s];
    ``max_range`` [m]; ``min_elevation_deg`` [deg].
    """
    site: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sigma_range: float = 15.0
    sigma_angle: float = 1.0e-3
    rate_hz: float = 10.0
    max_range: float = 120_000.0
    min_elevation_deg: float = 2.0
    bias_range: float = 0.0
    bias_angle: float = 0.0

    @property
    def R(self):
        """Measurement noise covariance (3, 3) of ``[range, azimuth, elevation]``."""
        return np.diag([self.sigma_range ** 2, self.sigma_angle ** 2, self.sigma_angle ** 2])


def measure(radar: Radar, pos):
    """Noise-free ``[range, azimuth, elevation]`` of ``pos`` seen from the radar.

    Parameters
    ----------
    radar : Radar
    pos : array (3,) [m]

    Returns
    -------
    z : array (3,)
        Range [m], azimuth [rad] (from +x towards +y), elevation [rad].
    """
    d = np.asarray(pos, float) - radar.site
    rh = math.hypot(d[0], d[1])
    return np.array([float(np.linalg.norm(d)), math.atan2(d[1], d[0]), math.atan2(d[2], rh)])


def visible(radar: Radar, pos) -> bool:
    """Whether ``pos`` is inside the radar's coverage (range and minimum elevation)."""
    r, _, el = measure(radar, pos)
    return r <= radar.max_range and math.degrees(el) >= radar.min_elevation_deg


def measure_noisy(radar: Radar, pos, rng: np.random.Generator):
    """A measurement with the radar's biases and Gaussian noise (see :func:`measure`)."""
    z = measure(radar, pos)
    z = z + np.array([radar.bias_range, radar.bias_angle, radar.bias_angle])
    return z + rng.normal(0.0, [radar.sigma_range, radar.sigma_angle, radar.sigma_angle])


def to_cartesian(radar: Radar, z):
    """Position [m] of a ``[range, azimuth, elevation]`` measurement."""
    r, az, el = z
    return radar.site + r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
