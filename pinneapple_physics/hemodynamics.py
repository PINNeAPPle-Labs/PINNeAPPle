"""Haemodynamics from 4D-flow MRI: Womersley's closed form as the reference, sparse samples of it as the
measurement, and a Navier-Stokes PINN asked for the two things the modality does not measure (#362).

4D-flow MRI gives velocity: a few planes across a vessel, twenty-odd cardiac phases, voxels of one to three
millimetres and noise. It does not give pressure, and it does not give wall shear stress (WSS) -- both are derivatives
of what it measures, taken at a wall where there is no sample. Whether a physics-informed network can supply them is
a measurable question, and it is measured here on a **synthetic phantom**: a rigid straight artery whose exact
solution is known in closed form. No patient data is involved at any point.

The reference is Womersley's solution of pulsatile laminar flow in a rigid straight tube. For a pressure gradient
``-dp/dz = Re[G e^{i w t}]`` the axial velocity is

    W(r) = (G / (i rho w)) [1 - J0(beta r/R) / J0(beta)],      beta = alpha sqrt(-i),   alpha = R sqrt(w / nu)

with ``alpha`` the Womersley number. Superposed over harmonics it is an exact solution of the axisymmetric
Navier-Stokes equations, which is what makes it a reference and not an approximation. Its wall shear stress is the
gradient at ``r = R`` and its pressure is linear along the axis, so both target quantities exist in closed form.

    >>> from pinneapple_physics.hemodynamics import BLOOD, Womersley, sample_4d_flow
    >>> wave = Womersley.from_flow_rate(radius=0.015, nu=BLOOD.nu, omega=2 * 3.141592653589793,
    ...                                 flow=[(1, 70e-6, -1.2)])
    >>> wave.alpha, wave.peak_velocity, wave.wall_shear(0.0)
    >>> four_d = sample_4d_flow(wave, n_radial=8, n_phases=24, noise=0.05, seed=0)

What the measurement leaves out is the point. The nearest voxel centre to the wall sits half a voxel away from it
(``voxel / 2``), and the boundary layer the wall shear stress lives in is ``R / alpha`` thick. When the first is
larger than the second -- a 15 mm aorta at 60 bpm, where ``R/alpha`` is 0.7 mm and a 3 mm voxel is 1.5 mm away -- the
data contain no information about the gradient at the wall, and only the physics can supply it. :func:`resolution_sweep`
measures what that costs, and the guide ``docs/guides/hemodynamics_4dflow.md`` reports the number.

What this is not: patient data, a real MRI scanner's sampling protocol (no slice profile, no partial volume, no
velocity aliasing), a curved or compliant vessel, or a validation against measured flow. The phantom is a straight
rigid tube because that is where a closed form exists.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["BLOOD", "Blood", "Harmonic", "Womersley", "FourDFlow", "sample_4d_flow", "wss_from_profile",
           "wss_from_4d_flow", "stokes_layer", "DEFAULT_VESSELS", "PHYSIOLOGICAL_FLOW", "Vessel",
           "case_from_vessel", "PINNConfig", "WomersleyPINN", "train", "predict", "model_wall_shear",
           "model_pressure_gradient", "ensemble", "measure", "relative_l2", "two_pi_grid"]

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------------------------------
# blood and the closed form
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Blood:
    """Density and dynamic viscosity. The defaults are round textbook values for whole blood at 37 C, Newtonian:
    1060 kg/m^3 and 3.5 mPa s, which is the assumption Womersley's solution is derived under."""
    rho: float = 1060.0
    mu: float = 3.5e-3

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m^2/s]."""
        return self.mu / self.rho


BLOOD = Blood()


@dataclass(frozen=True)
class Harmonic:
    """One frequency of the pressure gradient: ``-dp/dz = Re[gradient * exp(i (order*omega*t + phase))]`` [Pa/m].

    The gradient is the amplitude of the *forcing*, not of the velocity: it is what Womersley's solution is written
    in terms of. :meth:`Womersley.from_flow_rate` builds the harmonics from a flow-rate waveform instead, which is
    the quantity a physiologist has.
    """
    order: int
    gradient: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("harmonics orders are non-negative; a negative frequency is a conjugated phase")

    @property
    def complex_amplitude(self) -> complex:
        return self.gradient * complex(math.cos(self.phase), math.sin(self.phase))


def stokes_layer(radius: float, alpha: float) -> float:
    """Thickness of the oscillatory boundary layer [m], ``R / alpha`` -- the length the wall shear stress lives in.

    It is the distance over which the velocity falls from its core value to zero at the wall, and the reason a coarse
    voxel cannot see the gradient: with ``alpha = 20`` on a 15 mm radius the layer is 0.72 mm thick.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return radius / alpha


@dataclass(frozen=True)
class _Mode:
    """The solved radial profile of one harmonic, cached on construction."""
    order: int
    gradient: complex        # complex amplitude of -dp/dz [Pa/m]
    beta: complex            # Womersley parameter, alpha*sqrt(-i*order)
    centre: complex          # amplitude of the velocity on the axis [m/s]
    gain: complex            # centre*(1 - J0(beta y)/J0(beta)) prefactor, i.e. centre itself
    wall_gradient: complex   # dW/dr at r = R [1/s]


class Womersley:
    """Pulsatile laminar flow in a rigid straight tube, as Womersley (1955) wrote it.

    ``radius`` [m], ``nu`` [m^2/s], ``omega`` [rad/s] and the harmonics of the pressure gradient. Everything else --
    velocity, flow rate, wall shear stress, pressure drop -- follows from the closed form; nothing is integrated
    numerically.

    The solution is linear in the forcing, so the harmonics superpose and any waveform that is a finite sum of
    frequencies is exact. That is what makes this a reference: at every point of the tube and every instant the
    velocity returned here satisfies the axisymmetric Navier-Stokes equations to roundoff.
    """

    def __init__(self, radius: float, nu: float, omega: float, harmonics: Sequence[Harmonic],
                 rho: float = 1060.0) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if nu <= 0:
            raise ValueError("nu must be positive")
        if omega <= 0:
            raise ValueError("omega must be positive")
        if not harmonics:
            raise ValueError("at least one harmonic is required")
        orders = [h.order for h in harmonics]
        if len(set(orders)) != len(orders):
            raise ValueError("two harmonics share an order; sum them into one")
        self.radius = float(radius)
        self.nu = float(nu)
        self.omega = float(omega)
        self.rho = float(rho)
        self.harmonics = tuple(harmonics)
        self._modes = tuple(self._mode(h) for h in sorted(self.harmonics, key=lambda h: h.order))

    # -- construction ------------------------------------------------------------------------------------------
    def _mode(self, h: Harmonic) -> _Mode:
        from scipy.special import jv
        g = h.complex_amplitude
        if h.order == 0:
            # steady Poiseuille: W = G (R^2 - r^2)/(4 mu), dW/dr = -G r/(2 mu)
            centre = g * self.radius ** 2 / (4.0 * self.rho * self.nu)
            return _Mode(0, g, 0.0 + 0.0j, centre, centre, -g * self.radius / (2.0 * self.rho * self.nu))
        alpha = self.alpha * math.sqrt(h.order)
        beta = alpha * complex(math.sqrt(0.5), -math.sqrt(0.5))       # alpha*sqrt(-i)
        centre = g / (1j * self.rho * self.omega * h.order)
        wall = centre * beta * jv(1, beta) / (self.radius * jv(0, beta))
        return _Mode(h.order, g, beta, centre, centre, wall)

    @classmethod
    def from_flow_rate(cls, radius: float, nu: float, omega: float,
                       flow: Sequence[tuple[int, float, float]], rho: float = 1060.0,
                       mu: float | None = None) -> Womersley:
        """Build from a **flow-rate** waveform instead of a pressure gradient.

        ``flow`` is a sequence of ``(order, amplitude [m^3/s], phase [rad])`` with
        ``Q(t) = Re[sum amplitude*exp(i(order*omega*t + phase))]``. The pressure gradient that drives it follows from
        Womersley's flow-rate relation ``Q = pi R^2 G Phi(beta) / (i rho omega)`` with
        ``Phi = 1 - 2 J1(beta)/(beta J0(beta))``; for the steady component ``Q0 = pi R^4 G0 / (8 mu)``.

        A flow rate is what a measurement gives ("70 mL in 0.3 s through the aortic valve"); a pressure gradient of a
        few hundred pascal per metre is not a number anyone knows for a given vessel.
        """
        from scipy.special import jv
        mu = rho * nu if mu is None else mu
        harmonics = []
        alpha = radius * math.sqrt(omega / nu)
        for order, amplitude, phase in flow:
            q = amplitude * complex(math.cos(phase), math.sin(phase))
            if order == 0:
                gradient = q * 8.0 * mu / (math.pi * radius ** 4)
            else:
                beta = alpha * math.sqrt(order) * complex(math.sqrt(0.5), -math.sqrt(0.5))
                phi = 1.0 - 2.0 * jv(1, beta) / (beta * jv(0, beta))
                gradient = q * 1j * rho * omega * order / (math.pi * radius ** 2 * phi)
            harmonics.append(Harmonic(order, abs(gradient), math.atan2(gradient.imag, gradient.real)))
        return cls(radius, nu, omega, harmonics, rho=rho)

    # -- properties --------------------------------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        """Womersley number of the fundamental, ``R sqrt(omega/nu)``."""
        return self.radius * math.sqrt(self.omega / self.nu)

    @property
    def mu(self) -> float:
        return self.rho * self.nu

    @property
    def stokes_layer(self) -> float:
        return stokes_layer(self.radius, self.alpha)

    @property
    def period(self) -> float:
        return TWO_PI / self.omega

    # -- fields ------------------------------------------------------------------------------------------------
    def velocity(self, y, tau) -> np.ndarray:
        """Axial velocity [m/s] at ``y = r/R`` in ``[0, 1]`` and phase ``tau = omega*t`` [rad]."""
        y = np.asarray(y, dtype=float)
        tau = np.asarray(tau, dtype=float)
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("y = r/R must lie in [0, 1]")
        out = np.zeros(np.broadcast(y, tau).shape, dtype=float)
        for m in self._modes:
            out = out + np.real(m.centre * self._radial(m, y) * np.exp(1j * (m.order * tau)))
        return out

    def radial_profile(self, y, tau: float = 0.0) -> np.ndarray:
        """Alias for :meth:`velocity` at one phase, for readability when a profile is meant."""
        return self.velocity(y, tau)

    def _radial(self, m: _Mode, y: np.ndarray) -> np.ndarray:
        """``1 - J0(beta y)/J0(beta)``, the shape factor of one harmonic; 1 - y^2 for the steady one."""
        if m.order == 0:
            return 1.0 - np.asarray(y, dtype=float) ** 2
        from scipy.special import jv
        return 1.0 - jv(0, m.beta * np.asarray(y, dtype=float)) / jv(0, m.beta)

    def shear_rate(self, y, tau) -> np.ndarray:
        """``dw/dr`` [1/s] at ``y = r/R`` and phase ``tau``. Signed: negative near the wall for forward flow."""
        y = np.asarray(y, dtype=float)
        tau = np.asarray(tau, dtype=float)
        out = np.zeros(np.broadcast(y, tau).shape, dtype=float)
        for m in self._modes:
            if m.order == 0:
                d = -2.0 * m.centre * y / self.radius
            else:
                from scipy.special import jv
                d = m.centre * m.beta * jv(1, m.beta * y) / (self.radius * jv(0, m.beta))
            out = out + np.real(d * np.exp(1j * (m.order * tau)))
        return out

    def wall_shear(self, tau) -> np.ndarray:
        """Wall shear stress [Pa], ``-mu dw/dr`` at ``r = R``: positive when the wall feels a forward drag.

        The minus sign is the convention, not a choice of this module: for steady flow ``WSS = G R / 2 = 4 mu Q/(pi
        R^3)``, which is the textbook Poiseuille value and positive for forward flow.
        """
        tau = np.asarray(tau, dtype=float)
        out = np.zeros(tau.shape, dtype=float)
        for m in self._modes:
            out = out + np.real(-self.mu * m.wall_gradient * np.exp(1j * (m.order * tau)))
        return out

    def flow_rate(self, tau) -> np.ndarray:
        """Volume flow [m^3/s] through the cross-section at phase ``tau``."""
        from scipy.special import jv
        tau = np.asarray(tau, dtype=float)
        out = np.zeros(tau.shape, dtype=float)
        for m in self._modes:
            if m.order == 0:
                q = m.gradient * math.pi * self.radius ** 4 / (8.0 * self.mu)
            else:
                phi = 1.0 - 2.0 * jv(1, m.beta) / (m.beta * jv(0, m.beta))
                q = m.centre * math.pi * self.radius ** 2 * phi
            out = out + np.real(q * np.exp(1j * (m.order * tau)))
        return out

    def gradient(self, tau) -> np.ndarray:
        """``-dp/dz`` [Pa/m] at phase ``tau``."""
        tau = np.asarray(tau, dtype=float)
        out = np.zeros(tau.shape, dtype=float)
        for m in self._modes:
            out = out + np.real(m.gradient * np.exp(1j * (m.order * tau)))
        return out

    def pressure_drop(self, tau, length: float) -> np.ndarray:
        """Pressure difference ``p(z) - p(z + length)`` [Pa]: the gradient times the distance, gauge-free.

        An incompressible field has no absolute pressure -- only differences between two places mean anything, and
        that is the quantity every comparison below is made on.
        """
        return self.gradient(tau) * float(length)

    # -- scales ------------------------------------------------------------------------------------------------
    def peak_velocity(self, n: int = 720) -> float:
        """Peak of ``|w|`` over the cycle and the radius, from the closed form on an ``n``-point grid."""
        y = np.linspace(0.0, 1.0, 129)[:, None]
        tau = np.linspace(0.0, TWO_PI, n, endpoint=False)[None, :]
        return float(np.max(np.abs(self.velocity(y, tau))))

    def peak_wall_shear(self, n: int = 720) -> float:
        tau = np.linspace(0.0, TWO_PI, n, endpoint=False)
        return float(np.max(np.abs(self.wall_shear(tau))))

    def peak_flow_rate(self, n: int = 720) -> float:
        tau = np.linspace(0.0, TWO_PI, n, endpoint=False)
        return float(np.max(np.abs(self.flow_rate(tau))))

    def state(self) -> dict[str, float]:
        """The numbers that describe the case, for a record."""
        return {"radius_m": self.radius, "nu_m2_s": self.nu, "rho_kg_m3": self.rho, "omega_rad_s": self.omega,
                "alpha": self.alpha, "stokes_layer_m": self.stokes_layer, "period_s": self.period,
                "peak_velocity_m_s": self.peak_velocity(), "peak_wall_shear_pa": self.peak_wall_shear(),
                "peak_flow_rate_m3_s": self.peak_flow_rate(),
                "harmonics": [{"order": h.order, "gradient_pa_m": h.gradient, "phase_rad": h.phase}
                              for h in self.harmonics]}


# --------------------------------------------------------------------------------------------------------------
# the measurement: what a 4D-flow acquisition returns, sampled from the closed form
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FourDFlow:
    """A synthetic 4D-flow acquisition of one :class:`Womersley` case.

    The samples are voxel centres, not a continuous profile: ``n_radial`` voxels across the radius put the ``j``-th
    centre at ``y = (j + 1/2)/n_radial``, so the closest sample to the wall is **half a voxel away from it** and the
    wall itself is never measured. ``n_planes`` planes along the axis repeat the same measurement with independent
    noise (the flow is developed, so the planes differ only in their noise), and ``n_phases`` cardiac phases sample
    the cycle.

    ``velocity`` is what the scanner would report: ``clean`` plus Gaussian noise of standard deviation ``sigma``,
    which is ``noise`` times the peak velocity of the case (a 5 % velocity error, the usual order for 4D-flow).
    """
    y: np.ndarray
    tau: np.ndarray
    velocity: np.ndarray
    clean: np.ndarray
    sigma: float
    radius: float
    n_planes: int
    n_phases: int
    n_radial: int
    noise: float
    seed: int

    @property
    def voxel(self) -> float:
        """In-plane voxel size [m], ``R/n_radial``."""
        return self.radius / self.n_radial

    @property
    def wall_gap(self) -> float:
        """Distance from the nearest sample to the wall [m]: half a voxel, and the reason WSS is a derivative with no
        support."""
        return 0.5 * self.voxel

    @property
    def n_points(self) -> int:
        return int(self.y.size)

    def phases(self) -> np.ndarray:
        """The cardiac phases present, in radians."""
        return np.unique(np.round(self.tau, 12))

    def profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """``(y, velocity)`` for one cycle, averaged over the planes: ``(n_radial, n_phases)``. Averaging the planes
        is what a careful analyst does with repeated measurements, and it gives the finite-difference baseline its
        best case."""
        ys = np.unique(np.round(self.y, 12))
        each = np.unique(np.round(self.tau, 12))
        out = np.zeros((ys.size, each.size))
        for i, yy in enumerate(ys):
            for j, tt in enumerate(each):
                mask = np.isclose(self.y, yy) & np.isclose(self.tau, tt)
                out[i, j] = float(np.mean(self.velocity[mask]))
        return ys, out

    def record(self) -> dict[str, float | int]:
        return {"n_planes": self.n_planes, "n_phases": self.n_phases, "n_radial": self.n_radial,
                "n_points": self.n_points, "voxel_m": self.voxel, "wall_gap_m": self.wall_gap,
                "noise_fraction": self.noise, "sigma_m_s": self.sigma, "seed": self.seed}


def sample_4d_flow(wave: Womersley, *, n_planes: int = 3, n_phases: int = 24, n_radial: int = 8,
                   noise: float = 0.05, seed: int = 0) -> FourDFlow:
    """Sample a :class:`Womersley` case the way a 4D-flow acquisition samples a vessel.

    ``n_radial`` voxels across the radius (so the wall is half a voxel from the first sample), ``n_phases`` cardiac
    phases over the cycle, ``n_planes`` planes along the axis, and Gaussian noise of ``noise`` times the peak
    velocity. Nothing else of the acquisition is modelled -- no slice profile, no partial volume, no aliasing.
    """
    if n_radial < 3:
        raise ValueError("n_radial must be at least 3 to resolve anything at all")
    if n_phases < 4:
        raise ValueError("n_phases must be at least 4")
    if n_planes < 1:
        raise ValueError("n_planes must be at least 1")
    if not 0.0 <= noise < 1.0:
        raise ValueError("noise is a fraction of the peak velocity, in [0, 1)")
    rng = np.random.default_rng(seed)
    y1 = (np.arange(n_radial) + 0.5) / n_radial
    tau1 = TWO_PI * np.arange(n_phases) / n_phases
    y = np.repeat(y1, n_planes * n_phases)
    tau = np.tile(np.tile(tau1, n_planes), n_radial)
    clean = wave.velocity(y, tau)
    sigma = noise * wave.peak_velocity()
    measured = clean + (rng.normal(size=clean.shape) * sigma if sigma > 0 else 0.0)
    return FourDFlow(y=y, tau=tau, velocity=measured, clean=clean, sigma=float(sigma), radius=wave.radius,
                     n_planes=n_planes, n_phases=n_phases, n_radial=n_radial, noise=float(noise), seed=seed)


def wss_from_profile(y: np.ndarray, velocity: np.ndarray, radius: float, mu: float, order: int = 1) -> float:
    """Wall shear stress [Pa] read straight off a measured profile: fit a polynomial in ``y`` through the
    outermost ``order + 1`` samples and extrapolate its slope to ``y = 1``.

    ``order = 1`` is the two-point difference every 4D-flow WSS pipeline starts from; ``order = 2`` fits a parabola
    through three. Both are *data only*: the extrapolation runs over the gap between the last sample and the wall,
    where there is no measurement, so the error they make is the honest floor of a velocity-only estimate and the
    number a PINN has to beat.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if y.size != v.size:
        raise ValueError("y and velocity must have the same length")
    if y.size < order + 1:
        raise ValueError(f"order {order} needs at least {order + 1} samples, got {y.size}")
    idx = np.argsort(y)[-order - 1:]
    coef = np.polyfit(y[idx], v[idx], order)
    slope = float(np.polyval(np.polyder(coef), 1.0))          # dw/dy at the wall
    return float(-mu * slope / radius)


def wss_from_4d_flow(four_d: FourDFlow, mu: float, order: int = 1) -> np.ndarray:
    """The data-only estimate of the WSS waveform [Pa]: :func:`wss_from_profile` at every measured phase."""
    ys, v = four_d.profiles()
    return np.array([wss_from_profile(ys, v[:, j], four_d.radius, mu, order) for j in range(v.shape[1])])


# --------------------------------------------------------------------------------------------------------------
# the vessels of the phantom
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Vessel:
    """One synthetic phantom: a straight rigid tube with a radius and a flow-rate waveform.

    The radii bracket the range of human arteries at 60 bpm; the waveforms are plausible round values, not
    measurements. None of this is patient data -- the whole case is generated from the closed form.
    """
    name: str
    radius: float                      # m
    flow: tuple[tuple[int, float, float], ...]   # (order, amplitude [m^3/s], phase [rad])
    note: str = ""


DEFAULT_VESSELS: tuple[Vessel, ...] = (
    Vessel("small_artery", 2.0e-3, ((0, 0.9e-6, 0.0), (1, 1.4e-6, -0.9)),
           "R = 2 mm, 60 bpm: alpha just above 2, a nearly quasi-steady profile"),
    Vessel("femoral", 4.0e-3, ((0, 3.2e-6, 0.0), (1, 5.4e-6, -0.9)),
           "R = 4 mm: alpha about 5, the profile starts to flatten"),
    Vessel("carotid", 8.0e-3, ((0, 1.2e-5, 0.0), (1, 2.2e-5, -0.8)),
           "R = 8 mm: alpha about 11, a visible boundary layer"),
    Vessel("aorta", 1.5e-2, ((0, 4.0e-5, 0.0), (1, 9.0e-5, -1.2)),
           "R = 15 mm: alpha about 20, a 0.7 mm Stokes layer that no 4D-flow voxel resolves"),
)

#: A physiological waveform: a mean, two harmonics and a systolic peak. Used for the one case that is not a single
#: frequency, because a single frequency hides everything a real waveform does to a WSS estimate.
PHYSIOLOGICAL_FLOW: tuple[tuple[int, float, float], ...] = (
    (0, 3.0e-5, 0.0), (1, 7.0e-5, -1.1), (2, 3.4e-5, -2.4), (3, 1.6e-5, -3.6), (4, 0.7e-5, -4.6),
)


# --------------------------------------------------------------------------------------------------------------
# the PINN: axisymmetric Navier-Stokes in Womersley variables
# --------------------------------------------------------------------------------------------------------------
# The network is asked for three dimensionless fields over ``y = r/R`` and ``tau = omega*t``:
#
# * ``v(y, tau) = w / W0`` -- the axial velocity, the quantity the scanner measures;
# * ``u(y, tau) = u_r / W0`` -- the radial velocity, which the exact solution has as zero and the network has to
#   discover, because the equations allow nothing else;
# * ``g(tau) = -(dp/dz) / (rho omega W0)`` -- the axial pressure gradient, which in a straight developed tube is a
#   function of time alone. Learning it as a scalar waveform *is* recovering the relative pressure: the pressure
#   difference between two planes a distance ``L`` apart is ``g rho omega W0 L``, and the absolute level is a gauge an
#   incompressible field does not have.
#
# Three assumptions make the system well posed on a domain one dimension wide, and each is an assumption about the
# phantom rather than a shortcut: the flow is **axisymmetric** (straight tube), **developed** (the axial derivative of
# every field vanishes, which is why ``z`` is not an input) and **swirl-free**. Under them the axisymmetric
# Navier-Stokes equations, scaled by ``R``, ``1/omega`` and ``W0``, are
#
#     u_y + u/y = 0
#     u_tau + A (u u_y) = C (u_yy + u_y/y - u/y^2)
#     v_tau + A (u v_y) = g + C (v_yy + v_y/y)
#
# with ``A = Re/alpha^2``, ``C = 1/alpha^2`` and ``Re = W0 R / nu``. ``C`` is 2.3e-3 at ``alpha = 21``: the viscous term
# is invisible in the core and decisive in a boundary layer ``1/alpha`` thick next to the wall. That imbalance is the
# whole difficulty of this problem and the reason the sweep below is interesting.

@dataclass(frozen=True)
class PINNConfig:
    """How the network is built and trained. The defaults are the ones the guide's sweep was run with."""
    width: int = 48
    depth: int = 4
    n_collocation: int = 3000
    adam_steps: int = 2500
    lr: float = 3e-3
    lr_final: float = 1e-4
    lbfgs_steps: int = 300
    data_weight: float = 1.0
    pde_weight: float = 1.0
    y_min: float = 0.02
    seed: int = 0


def _mlp(n_in: int, n_out: int, width: int, depth: int) -> "torch.nn.Module":  # noqa: F821
    import torch
    layers: list = []
    last = n_in
    for _ in range(depth):
        layers += [torch.nn.Linear(last, width), torch.nn.Tanh()]
        last = width
    layers += [torch.nn.Linear(last, n_out)]
    return torch.nn.Sequential(*layers)


class WomersleyPINN:
    """The model, built lazily so that importing this module does not import torch."""

    @staticmethod
    def build(config: PINNConfig | None = None):
        import torch

        config = config or PINNConfig()

        class _Net(torch.nn.Module):
            """``(y, tau) -> (u, v, g)``, with the exact symmetries built into the ansatz.

            ``v = (1 - y^2) N_v(y^2, sin tau, cos tau)`` is even in ``y`` and vanishes at the wall, so no-slip and
            axis symmetry hold to machine precision and cannot be traded away by the optimiser. ``u = y(1-y^2) N_u``
            vanishes on the axis and at the wall, as it must. ``g`` is a function of the phase alone. Periodicity is
            exact: ``tau`` enters only as ``sin`` and ``cos``, so the field at ``tau = 0`` and ``tau = 2 pi`` is the
            same value by construction, not by a penalty.
            """

            def __init__(self) -> None:
                super().__init__()
                torch.manual_seed(config.seed)
                self.net_field = _mlp(3, 2, config.width, config.depth)
                self.net_grad = _mlp(2, 1, config.width, config.depth)

            def forward(self, y, tau):
                s = y * y
                st, ct = torch.sin(tau), torch.cos(tau)
                feats = torch.cat([s, st, ct], dim=1)
                field = self.net_field(feats)
                u = y * (1.0 - s) * field[:, 0:1]
                v = (1.0 - s) * field[:, 1:2]
                g = self.net_grad(torch.cat([st, ct], dim=1))
                return torch.cat([u, v, g], dim=1)

        return _Net()


def _loss_function(alpha: float, reynolds: float, config: PINNConfig):
    """The compiled loss: three residuals, evaluated with autograd on the network's own output."""
    from pinneapple_physics.pinn_solver.factory.pinn_factory import PINNFactory, PINNProblemSpec

    a_inertia = reynolds / alpha ** 2
    c_viscous = 1.0 / alpha ** 2
    spec = PINNProblemSpec(
        pde_residuals=[
            "Derivative(u(y,tau), y) + u(y,tau)/y",
            "Derivative(u(y,tau), tau) + A*(u(y,tau)*Derivative(u(y,tau), y))"
            " - C*(Derivative(u(y,tau), (y,2)) + Derivative(u(y,tau), y)/y - u(y,tau)/y**2)",
            "Derivative(v(y,tau), tau) + A*(u(y,tau)*Derivative(v(y,tau), y)) - g(y,tau)"
            " - C*(Derivative(v(y,tau), (y,2)) + Derivative(v(y,tau), y)/y)",
        ],
        independent_vars=["y", "tau"],
        dependent_vars=["u", "v", "g"],
        constants={"A": a_inertia, "C": c_viscous},
        loss_weights={"pde": config.pde_weight},
    )
    return PINNFactory(spec).generate_loss_function()


def train(wave: Womersley, four_d: FourDFlow, config: PINNConfig | None = None,
          progress: bool = False, w0: float | None = None) -> tuple[object, dict]:
    """Fit the network to one acquisition. Returns the model and a record of the training.

    The data term is applied here rather than inside the compiled loss because it constrains one of the three fields
    (``v``); the residuals come from :func:`_loss_function`. Everything is in the dimensionless variables, and ``W0``
    is the largest measured velocity -- a scale that comes from the acquisition itself, not from the answer: what the
    network has to find is the shape and the phase, and the gradient at the wall follows from those.

    The scale is attached to the model as ``peak_velocity`` so that :func:`model_wall_shear` and
    :func:`model_pressure_gradient` undo it without being told twice.
    """
    import torch

    config = config or PINNConfig()
    w0 = float(np.max(np.abs(four_d.velocity))) if w0 is None else float(w0)
    alpha, reynolds = wave.alpha, w0 * wave.radius / wave.nu
    model = WomersleyPINN.build(config)
    model.peak_velocity = w0
    loss_fn = _loss_function(alpha, reynolds, config)

    rng = np.random.default_rng(config.seed + 7919)
    n = config.n_collocation
    # uniform in y^2 = s, which is the ansatz's own variable: it puts more points near the wall, where the
    # residual is hard, without knowing where the boundary layer is
    s = rng.uniform(config.y_min ** 2, 1.0, size=(n, 1))
    y_col = torch.tensor(np.sqrt(s), dtype=torch.float32)
    tau_col = torch.tensor(rng.uniform(0.0, TWO_PI, size=(n, 1)), dtype=torch.float32)

    y_dat = torch.tensor(four_d.y.reshape(-1, 1), dtype=torch.float32)
    tau_dat = torch.tensor(four_d.tau.reshape(-1, 1), dtype=torch.float32)
    v_dat = torch.tensor(four_d.velocity.reshape(-1, 1) / w0, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(config.adam_steps, 1),
                                                      eta_min=config.lr_final)
    history: list[dict[str, float]] = []

    def total_loss():
        pde, comps = loss_fn(model, {"collocation": (y_col, tau_col)})
        v_pred = model(y_dat, tau_dat)[:, 1:2]
        data = torch.mean((v_pred - v_dat) ** 2)
        return config.pde_weight * pde + config.data_weight * data, float(pde.detach()), float(data.detach())

    for step in range(config.adam_steps):
        opt.zero_grad(set_to_none=True)
        loss, pde, data = total_loss()
        loss.backward()
        opt.step()
        sched.step()
        if progress and (step % 250 == 0 or step == config.adam_steps - 1):
            history.append({"step": step, "pde": pde, "data": data})
            print(f"  step {step:5d}  pde={pde:.3e}  data={data:.3e}  loss={loss.item():.3e}", flush=True)

    if config.lbfgs_steps > 0:
        opt2 = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=config.lbfgs_steps,
                                 history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad(set_to_none=True)
            loss, _, _ = total_loss()
            loss.backward()
            return loss

        opt2.step(closure)
        if progress:
            _, pde, data = total_loss()
            print(f"  after LBFGS  pde={float(pde):.3e}  data={float(data):.3e}", flush=True)

    final, pde, data = total_loss()
    model.eval()
    record = {"alpha": alpha, "reynolds": reynolds, "A": reynolds / alpha ** 2, "C": 1.0 / alpha ** 2,
              "peak_velocity_m_s": w0, "collocation": config.n_collocation, "adam_steps": config.adam_steps,
              "lbfgs_steps": config.lbfgs_steps, "data_weight": config.data_weight, "seed": config.seed,
              "final_pde": pde, "final_data": data, "final_loss": final.item(), "history": history}
    return model, record


def predict(model, y, tau) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, g)`` at the given points, broadcast against each other and flattened."""
    import torch
    yy, tt = np.broadcast_arrays(np.asarray(y, dtype=np.float64), np.asarray(tau, dtype=np.float64))
    with torch.no_grad():
        yt = torch.tensor(yy.ravel().reshape(-1, 1), dtype=torch.float32)
        xt = torch.tensor(tt.ravel().reshape(-1, 1), dtype=torch.float32)
        out = model(yt, xt).numpy()
    return out[:, 0], out[:, 1], out[:, 2]


def model_wall_shear(model, tau, mu: float, radius: float, w0: float | None = None) -> np.ndarray:
    """WSS [Pa] from the network: ``-mu dw/dr`` at the wall, the radial derivative taken by autograd.

    It is the same operator the exact solution is closed-form in and the same one the data-only estimate
    extrapolates towards; the difference is where the slope comes from.
    """
    import torch
    w0 = float(model.peak_velocity) if w0 is None else float(w0)
    tt = torch.tensor(np.asarray(tau, dtype=np.float64).ravel().reshape(-1, 1), dtype=torch.float32)
    y = torch.ones_like(tt, requires_grad=True)
    v = model(y, tt)[:, 1:2]
    dv = torch.autograd.grad(v.sum(), y)[0]
    dw_dr = (dv * (w0 / radius)).detach().numpy().ravel()
    return -mu * dw_dr


def model_pressure_gradient(model, tau, wave: Womersley, w0: float | None = None) -> np.ndarray:
    """``-dp/dz`` [Pa/m] from the network, in physical units."""
    w0 = float(model.peak_velocity) if w0 is None else float(w0)
    _, _, g = predict(model, 0.5, tau)
    return g * wave.rho * wave.omega * w0


def relative_l2(pred, ref) -> float:
    """Relative L2 error, both arrays flattened: the metric every table in the guide is in terms of."""
    pred = np.asarray(pred, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    if pred.shape != ref.shape:
        raise ValueError(f"shapes differ: {pred.shape} against {ref.shape}")
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-300))


def ensemble(wave: Womersley, four_d: FourDFlow, members: int = 5, config: PINNConfig | None = None,
             progress: bool = False) -> dict:
    """An ensemble of independently seeded fits: the mean and the spread of what they disagree about.

    The spread is the uncertainty this module reports. It is a deep ensemble, not a variational posterior: it prices
    the optimisation and the initialisation, not the model form, and a boundary layer none of the members resolves is
    a bias all of them share and none of them reports. That limitation is measured in the guide rather than hidden.
    """
    config = config or PINNConfig()
    taus = two_pi_grid(four_d.n_phases)
    wss, grad, models = [], [], []
    for k in range(members):
        cfg = replace(config, seed=config.seed + 1000 * k)
        model, rec = train(wave, four_d, cfg, progress=progress)
        wss.append(model_wall_shear(model, taus, wave.mu, wave.radius))
        grad.append(model_pressure_gradient(model, taus, wave))
        models.append((model, rec))
    wss = np.array(wss)
    grad = np.array(grad)
    # a single member has no spread to report, and inventing one would be worse than saying so
    ddof = 1 if members > 1 else 0
    return {"tau": taus, "wss": wss, "gradient": grad, "members": models,
            "wss_mean": wss.mean(axis=0), "wss_std": wss.std(axis=0, ddof=ddof),
            "gradient_mean": grad.mean(axis=0), "gradient_std": grad.std(axis=0, ddof=ddof),
            "records": [r for _, r in models]}


def two_pi_grid(n_phases: int) -> np.ndarray:
    """``n_phases`` phases over one cycle, in radians, excluding the repeat of ``2 pi``."""
    return TWO_PI * np.arange(n_phases) / n_phases


def case_from_vessel(vessel: Vessel, omega: float = TWO_PI, blood: Blood = BLOOD) -> Womersley:
    """The :class:`Womersley` solution of one :class:`Vessel` at a heart rate of ``omega`` [rad/s]."""
    return Womersley.from_flow_rate(vessel.radius, blood.nu, omega, vessel.flow, rho=blood.rho)


def measure(wave: Womersley, four_d: FourDFlow, ens: dict, *, segment_length: float | None = None) -> dict:
    """Every error this module reports for one acquisition, against the closed form.

    The two targets are the wall shear stress and the pressure. The pressure is compared as the difference between
    two planes ``segment_length`` apart -- the only form in which an incompressible field has one -- and the wall
    shear stress as the waveform over the cycle. Both are relative L2 over the cycle, so the number does not depend
    on how long the studied segment is; the absolute peaks are reported beside them so that a 3 % error of 0.03 Pa
    and a 3 % error of 3 Pa are not mistaken for each other.
    """
    if segment_length is None:
        segment_length = 10.0 * wave.radius
    taus = ens["tau"]
    wss_true = wave.wall_shear(taus)
    grad_true = wave.gradient(taus)
    wss_pred = ens["wss_mean"]
    grad_pred = ens["gradient_mean"]
    wss_std = ens["wss_std"]

    yg = np.linspace(0.0, 1.0, 81)[:, None]
    vel_true = wave.velocity(yg, taus[None, :]) / float(np.max(np.abs(four_d.velocity)))
    _, vel_pred, _ = predict(ens["members"][0][0], np.repeat(yg, taus.size, axis=1),
                             np.tile(taus, (yg.size, 1)))
    # every member, so that the reported velocity error is the ensemble's and not the first member's
    vel_err = float(np.mean([relative_l2(
        predict(m, np.repeat(yg, taus.size, axis=1), np.tile(taus, (yg.size, 1)))[1], vel_true)
        for m, _ in ens["members"]]))

    fwd1 = wss_from_4d_flow(four_d, wave.mu, order=1)
    fwd2 = wss_from_4d_flow(four_d, wave.mu, order=2)
    err = np.abs(wss_pred - wss_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(wss_std > 0, err / np.maximum(wss_std, 1e-30), np.inf)
    return {
        "acquisition": four_d.record(),
        "scales": {"alpha": wave.alpha, "radius_m": wave.radius, "stokes_layer_m": wave.stokes_layer,
                   "voxel_m": four_d.voxel, "wall_gap_m": four_d.wall_gap,
                   "gap_over_stokes_layer": four_d.wall_gap / wave.stokes_layer,
                   "segment_length_m": segment_length,
                   "reynolds": float(np.max(np.abs(four_d.velocity))) * wave.radius / wave.nu},
        "truth": {"wss_peak_pa": float(np.max(np.abs(wss_true))),
                  "pressure_drop_peak_pa": float(np.max(np.abs(grad_true)) * segment_length),
                  "gradient_peak_pa_m": float(np.max(np.abs(grad_true))),
                  "velocity_peak_m_s": float(np.max(np.abs(four_d.clean)))},
        "pinn": {"wss_relative_error": relative_l2(wss_pred, wss_true),
                 "wss_peak_pa": float(np.max(np.abs(wss_pred))),
                 "pressure_gradient_relative_error": relative_l2(grad_pred, grad_true),
                 "pressure_drop_peak_pa": float(np.max(np.abs(grad_pred)) * segment_length),
                 "velocity_relative_error": vel_err,
                 "pde_residual": float(np.mean([r["final_pde"] for _, r in ens["members"]]))},
        "data_only": {"wss_relative_error_two_point": relative_l2(fwd1, wss_true),
                      "wss_relative_error_parabola": relative_l2(fwd2, wss_true),
                      "wss_peak_pa_two_point": float(np.max(np.abs(fwd1)))},
        "uncertainty": {"wss_std_mean_pa": float(np.mean(wss_std)),
                        "wss_std_max_pa": float(np.max(wss_std)),
                        "wss_std_over_peak": float(np.mean(wss_std) / max(np.max(np.abs(wss_true)), 1e-30)),
                        "pressure_gradient_std_mean_pa_m": float(np.mean(ens["gradient_std"])),
                        # one member cannot say how much the answer depends on the seed
                        "coverage_2sigma": None if len(ens["members"]) < 2 else float(
                            np.mean(err <= 2.0 * np.maximum(wss_std, 1e-30))),
                        "error_over_sigma_mean": None if len(ens["members"]) < 2 else float(
                            np.mean(np.minimum(z, 1e6))),
                        "members": len(ens["members"])},
        # the curves the figures and the table are drawn from, so that a record can be re-read without refitting
        "waveforms": {k: np.asarray(v, dtype=float).ravel().tolist() for k, v in (
            ("tau", taus), ("wss_truth", wss_true), ("wss_pinn", wss_pred), ("wss_std", wss_std),
            ("wss_two_point", fwd1), ("gradient_truth", grad_true), ("gradient_pinn", grad_pred),
            ("gradient_std", ens["gradient_std"]))},
    }
