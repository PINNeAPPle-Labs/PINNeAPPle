"""The answer of a section to a regular wave: added mass, damping, excitation and the transfer functions.

A ship is a three-dimensional body in a three-dimensional flow, and a two-dimensional section is not. What is solved
here, honestly, is the **two-dimensional** problem: a section of a hull of finite beam and draught, in water of a given
depth, moving harmonically under a linearised free surface. That is the classical strip problem, and it is where the
coefficients of strip theory come from; what is not done here is the three-dimensional radiation and diffraction of a
whole hull.

It is solved rather than quoted because the added mass and damping of a rectangular section are not closed form, and a
table recalled from memory is a number with nothing to check it against. The field is computed, and then it has to pass
the checks any correct solution passes:

* in the long-wave limit every response ratio tends to one: a body small against the wavelength moves with the water, so
  ``heave/amplitude -> 1``, ``sway/amplitude -> 1`` and ``roll/slope -> 1``;
* added mass and damping are positive, and both fall away at the short-wave end, where the body no longer makes waves;
* the roll resonance has to land where the hydrostatics put it, ``omega^2 = C44 / (I44 + A44)``.

The method is a finite-volume solve of ``div grad phi = 0`` on a graded tensor grid -- fine at the body, at least twenty
nodes per wavelength, coarser toward the boundary. The free surface carries ``dphi/dz = K phi`` with ``K = omega^2/g``,
the body is a Neumann surface with the prescribed normal velocity, and the outgoing wave is absorbed by a sponge in
which ``K`` acquires a large imaginary part (the complex-frequency trick), so the far end can be ``phi = 0``. The force
is the pressure at the wetted faces, ``F = -i omega rho oint phi n dS``.

Everything is per unit length of section, which the docstrings say where it matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spectrum as sp

__all__ = ["Section", "Grid", "TransferFunction", "solve", "sweep", "section_mass_properties", "build_grid"]

G = sp.G
RHO = sp.RHO


@dataclass(frozen=True)
class Section:
    """A rectangular section of a hull: beam ``B``, draught ``T``, and the water it sits in.

    ``kg`` is the height of the centre of gravity above the keel -- the axis the section rolls about -- and ``gyration``
    its radius of gyration for that roll, ``k_xx``. Both default to a homogeneous rectangle (``T/2`` and ``0.35 B``,
    the second being the usual first estimate for a real hull, the same one
    :mod:`pinneapple_simulation.naval.hydrostatics` uses). :meth:`from_gm` goes the other way: give the metacentric
    height the vessel is designed to, and it solves for the ``kg`` that produces it.

    ``roll_damping`` is a fraction of critical damping added to the roll, and it is not a detail: linear potential
    theory has almost no roll damping at all (the solution below gives ``zeta`` of the order of ``1e-5``), because the
    mechanisms that damp a real roll -- viscosity in the bilge, eddy shedding at the bilge keels, lift on the skeg --
    are outside it. Without a number here the roll response at resonance is a spike of forty times the wave slope; with
    the 8 % a vessel of this kind is usually quoted, it is what a seakeeping table reports.
    """

    beam: float
    draught: float
    depth: float | None = None            # water depth in m; None is deep water
    kg: float | None = None               # centre of gravity above the keel
    gyration: float | None = None         # roll radius of gyration, k_xx
    roll_damping: float | None = 0.08     # fraction of critical roll damping, on top of the potential solution

    def __post_init__(self) -> None:
        if not (self.beam > 0 and self.draught > 0):
            raise ValueError("beam and draught have to be positive")
        if self.depth is not None and self.depth <= self.draught:
            raise ValueError("the water has to be deeper than the draught")
        if self.kg is not None and not 0.0 <= self.kg <= 3.0 * self.draught:
            raise ValueError("kg has to be a height above the keel")
        if self.gyration is not None and self.gyration <= 0:
            raise ValueError("the radius of gyration has to be positive")
        if self.roll_damping is not None and not 0.0 <= self.roll_damping < 1.0:
            raise ValueError("the roll damping ratio has to be a fraction of critical")

    @classmethod
    def from_gm(cls, beam: float, draught: float, gm: float, **kwargs) -> Section:
        """The section whose metacentric height is ``gm``, from ``GM = B^2/(12 T) + T/2 - kg``."""
        return cls(beam=beam, draught=draught, kg=beam ** 2 / (12.0 * draught) + draught / 2.0 - gm, **kwargs)

    @property
    def zc(self) -> float:
        """Roll axis in the solver's frame, ``z = 0`` at the waterline and negative downwards."""
        centre = self.draught / 2.0 if self.kg is None else self.kg
        return -self.draught + centre

    @property
    def k_xx(self) -> float:
        return 0.35 * self.beam if self.gyration is None else self.gyration

    @property
    def area(self) -> float:
        """Displaced area per unit length, m^2."""
        return self.beam * self.draught

    @property
    def gm(self) -> float:
        """``GM = I_wl / volume + KB - KG = B^2/(12 T) + T/2 - kg``."""
        kg = self.draught / 2.0 if self.kg is None else self.kg
        return self.beam ** 2 / (12.0 * self.draught) + self.draught / 2.0 - kg

    def restoring(self) -> dict[str, float]:
        """Hydrostatic restoring per unit length: heave ``rho g B`` and roll ``rho g (I_wl + volume (z_b - z_g))``.

        For the rectangle ``KB = T/2``, so ``C44 = rho g (B^3/12 + B T (T/2 - kg)) = rho g volume GM``.
        """
        return {"heave": RHO * G * self.beam, "roll": RHO * G * self.area * self.gm}


def section_mass_properties(section: Section) -> dict[str, float]:
    """Mass, roll inertia and the two restoring coefficients of the section, per unit length."""
    mass = RHO * section.area
    return {"mass": mass, "inertia": mass * section.k_xx ** 2, **section.restoring()}


def wavenumber_for(omega: float, depth: float | None) -> float:
    """``k`` for one frequency, deep water when ``depth`` is ``None``."""
    return float(sp.wavenumber(np.array([omega]), depth)[0])


def incident_wave(omega: float, depth: float | None, x, z):
    """The incident potential of a unit-amplitude wave travelling in ``+x``: ``(phi, dphi/dx, dphi/dz)``.

    ``phi = (-i g A / omega) cosh(k (z + H)) / cosh(k H) exp(i k x)``, which is ``(-i g A / omega) exp(k z)`` in deep
    water.
    """
    k = wavenumber_for(omega, depth)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    amplitude = -1j * G / omega
    if depth is None:
        shape, dshape = np.exp(k * z), k * np.exp(k * z)
    else:
        shape = np.cosh(k * (z + depth)) / math.cosh(k * depth)
        dshape = k * np.sinh(k * (z + depth)) / math.cosh(k * depth)
    phase = np.exp(1j * k * x)
    return amplitude * shape * phase, amplitude * shape * 1j * k * phase, amplitude * dshape * phase


@dataclass(frozen=True)
class Grid:
    """The graded tensor grid the section problem is solved on. ``z`` runs from the bottom up to the waterline."""

    x: np.ndarray            # node centres, ascending
    z: np.ndarray            # node centres, ascending, the waterline at z = 0 (a face, not a node)
    body: np.ndarray         # (n_x, n_z) bool, True inside the hull
    sponge: np.ndarray       # (n_x,) the sponge strength at each column
    k: float                 # wavenumber of the wave being solved for

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.x.size), int(self.z.size)


def _centres(start: float, h_first: float, h_max: float, length: float, growth: float = 1.12) -> np.ndarray:
    """Cell centres from ``start``, spacing growing geometrically from ``h_first`` and capped at ``h_max``."""
    centres, position, h = [], float(start), float(h_first)
    while position < start + length - 1e-12:
        step = min(h, start + length - position)
        centres.append(position + step / 2.0)
        position += step
        h = min(h * growth, h_max)
    return np.array(centres)


def build_grid(section: Section, omega: float, *, nodes_per_wave: float = 24.0, sponge_waves: float = 3.0,
               open_waves: float = 1.2, growth: float = 1.12) -> Grid:
    """The grid for one frequency: fine at the body, one wave of open water, and a sponge at each end.

    The sponge is three wavelengths long by default, and that is not a free choice: with one and a half, the wave the
    section radiates is reflected back off the end of the domain hard enough to move the added mass of this section by
    a third, while the response ratios barely move. It is the coefficient most sensitive to the boundary, so the check
    in the tests is on it.
    """
    k = wavenumber_for(omega, section.depth)
    length_wave = 2.0 * math.pi / k
    h_max = length_wave / nodes_per_wave
    h_fine = min(section.beam / 12.0, section.draught / 6.0, 0.5 * h_max)
    half = 0.5 * section.beam
    open_length = open_waves * length_wave
    sponge_length = sponge_waves * length_wave

    inside = _centres(0.0, h_fine, h_fine, half)
    outside = _centres(half, h_fine, h_max, open_length + sponge_length, growth)
    right = np.concatenate([inside, outside])
    x = np.concatenate([-right[::-1], right])

    bottom = section.depth if section.depth is not None else 0.8 * length_wave
    upper = _centres(0.0, h_fine, h_fine, section.draught)
    lower = _centres(section.draught, h_fine, h_max, bottom - section.draught, growth)
    z = (-np.concatenate([upper, lower]))[::-1]

    body = (np.abs(x)[:, None] < half - 1e-12) & (z[None, :] > -section.draught + 1e-12)
    ramp = np.zeros_like(x)
    far = np.abs(x) > half + open_length
    ramp[far] = (np.abs(x[far]) - (half + open_length)) / max(sponge_length, 1e-12)
    return Grid(x=x, z=z, body=body, sponge=2.5 * ramp ** 2, k=k)


@dataclass(frozen=True)
class _Layout:
    """Everything that does not depend on the mode: node bookkeeping, cell measures and the free-surface coefficient."""

    grid: Grid
    index: np.ndarray
    dx: np.ndarray
    dz: np.ndarray
    cvx: np.ndarray
    cvz: np.ndarray
    k_free: np.ndarray


def _layout(grid: Grid, omega: float) -> _Layout:
    nx, nz = grid.shape
    index = np.full((nx, nz), -1, dtype=int)
    index[~grid.body] = np.arange(int((~grid.body).sum()))
    dx, dz = np.diff(grid.x), np.diff(grid.z)
    cvx = np.empty(nx)
    cvx[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    cvx[0], cvx[-1] = 0.5 * dx[0], 0.5 * dx[-1]
    cvz = np.empty(nz)
    cvz[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    cvz[0], cvz[-1] = 0.5 * dz[0], 0.5 * dz[-1]
    return _Layout(grid=grid, index=index, dx=dx, dz=dz, cvx=cvx, cvz=cvz,
                   k_free=-omega ** 2 / G * (1.0 + 1j * grid.sponge) ** 2)


def _operator(layout: _Layout):
    """The finite-volume operator: the sum of the normal derivatives out of each cell is zero, free surface included."""
    from scipy.sparse import coo_matrix

    nx, nz = layout.grid.shape
    index, body = layout.index, layout.grid.body
    rows, cols, vals = [], [], []

    def put(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        vals.append(value)

    for i in range(nx):
        for j in range(nz):
            me = index[i, j]
            if me < 0:
                continue
            diagonal = 0.0
            for step in (-1, 1):
                ii = i + step
                if not 0 <= ii < nx:
                    diagonal -= 1.0 / (0.25 * layout.dx[0] * layout.cvx[i])          # phi = 0 on the far wall
                elif not body[ii, j]:
                    distance = layout.dx[i - 1] if step < 0 else layout.dx[i]
                    coefficient = 1.0 / (distance * layout.cvx[i])
                    put(me, index[ii, j], coefficient)
                    diagonal -= coefficient
                jj = j + step
                if 0 <= jj < nz and not body[i, jj]:
                    distance = layout.dz[j - 1] if step < 0 else layout.dz[j]
                    coefficient = 1.0 / (distance * layout.cvz[j])
                    put(me, index[i, jj], coefficient)
                    diagonal -= coefficient
            if j == nz - 1:                                                          # the free surface above
                diagonal -= layout.k_free[i]
            put(me, me, diagonal)
    n = int(index.max()) + 1
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def _body_faces(layout: _Layout) -> list[tuple[int, float, float, float, float, float, float]]:
    """Every face between a fluid cell and the hull: ``(cell, x, z, m_x, m_z, length, distance)``.

    ``m`` points from the hull into the fluid; the wetted surface takes the pressure along ``-m``.
    """
    nx, nz = layout.grid.shape
    index, body = layout.index, layout.grid.body
    faces = []
    for i in range(nx):
        for j in range(nz):
            me = index[i, j]
            if me < 0:
                continue
            for step in (-1, 1):
                ii = i + step
                if 0 <= ii < nx and body[ii, j]:
                    distance = layout.dx[i - 1] if step < 0 else layout.dx[i]
                    faces.append((me, layout.grid.x[i] + step * distance / 2.0, layout.grid.z[j],
                                  float(-step), 0.0, float(layout.cvz[j]), float(distance)))
                jj = j + step
                if 0 <= jj < nz and body[i, jj]:
                    distance = layout.dz[j - 1] if step < 0 else layout.dz[j]
                    faces.append((me, layout.grid.x[i], layout.grid.z[j] + step * distance / 2.0,
                                  0.0, float(-step), float(layout.cvx[i]), float(distance)))
    return faces


def _solve(section: Section, omega: float):
    """The radiation and diffraction solves at one frequency, and what they produce."""
    from scipy.sparse.linalg import splu

    grid = build_grid(section, omega)
    layout = _layout(grid, omega)
    lu = splu(_operator(layout).tocsc())
    faces = _body_faces(layout)
    zc = section.zc
    rate = -1j * omega                        # the velocity of a unit modal displacement, for every mode

    def body_velocity(mode: str, mx: float, mz: float, x: float, z: float) -> complex:
        if mode == "sway":
            return rate * mx
        if mode == "heave":
            return rate * mz
        return rate * (x * mz - (z - zc) * mx)          # roll: U = (-Omega (z - zc), Omega x)

    def generalize(phi_at_faces) -> np.ndarray:
        """The generalized force of a pressure field on the wetted surface: ``[F_x, F_z, M_roll]``."""
        total = np.zeros(3, dtype=complex)
        for (_me, x, z, mx, mz, length, _), phi in zip(faces, phi_at_faces, strict=True):
            pressure = 1j * omega * RHO * phi
            total += -pressure * length * np.array([mx, mz, x * mz - (z - zc) * mx])
        return total

    reactance = np.zeros((3, 3), dtype=complex)
    for column, mode in enumerate(("sway", "heave", "roll")):
        rhs = np.zeros(lu.shape[0], dtype=complex)
        for me, x, z, mx, mz, _, _ in faces:
            rhs[me] += body_velocity(mode, mx, mz, x, z)
        phi = lu.solve(rhs)
        reactance[:, column] = generalize(
            [phi[me] - 0.5 * distance * body_velocity(mode, mx, mz, x, z)
             for me, x, z, mx, mz, _, distance in faces])

    # diffraction: the hull is held still, so the scattered potential cancels the incident normal velocity
    rhs = np.zeros(lu.shape[0], dtype=complex)
    incident = []
    for me, x, z, mx, mz, _, distance in faces:
        phi_i, dphi_dx, dphi_dz = incident_wave(omega, section.depth, x, z)
        normal = dphi_dx * mx + dphi_dz * mz
        rhs[me] += -normal
        incident.append((complex(phi_i), complex(normal), distance))
    phi_s = lu.solve(rhs)
    excitation = generalize(
        [phi_i + 0.5 * distance * normal + phi_s[me]
         for (me, *_), (phi_i, normal, distance) in zip(faces, incident, strict=True)])

    a = np.real(reactance) / omega ** 2
    b = np.imag(reactance) / omega
    mass = section_mass_properties(section)
    if section.roll_damping:
        critical = math.sqrt(mass["roll"] * (mass["inertia"] + a[2, 2]))
        b[2, 2] = max(b[2, 2], 0.0) + 2.0 * section.roll_damping * critical
    big_m = np.diag([mass["mass"], mass["mass"], mass["inertia"]])
    restoring = np.diag([0.0, mass["heave"], mass["roll"]])
    xi = np.linalg.solve(-omega ** 2 * (big_m + a) - 1j * omega * b + restoring, excitation)
    slope = wavenumber_for(omega, section.depth)
    return {"omega": omega, "period": 2.0 * math.pi / omega, "a": a, "b": b, "excitation": excitation, "xi": xi,
            "rao_sway": float(abs(xi[0])), "rao_heave": float(abs(xi[1])), "rao_roll": float(abs(xi[2] / slope)),
            "cells": int((~grid.body).sum())}


def solve(section: Section, omega: float) -> dict:
    """Radiation and diffraction at one frequency.

    Returns the added mass ``a`` and damping ``b`` matrices per unit length (3x3, sway/heave/roll), the exciting force
    ``excitation`` per unit wave amplitude ``[F_x, F_z, M]``, the motion ``xi`` and the response ratios
    ``rao_sway``, ``rao_heave`` (m per m) and ``rao_roll`` (rad per rad of wave slope).
    """
    if not omega > 0:
        raise ValueError(f"the frequency has to be positive, got {omega}")
    return _solve(section, omega)


@dataclass(frozen=True)
class TransferFunction:
    """The response to a unit wave on a set of periods.

    ``heave`` and ``sway`` are metres per metre of wave amplitude; ``roll`` and ``pitch`` are radians per radian of
    wave slope, which is the convention a seakeeping table uses. ``pitch`` comes from the *longitudinal* section
    (:func:`pitch_transfer_function`), which is the two-dimensional stand-in for a mode that is not two-dimensional.
    """

    periods: np.ndarray
    heave: np.ndarray
    sway: np.ndarray
    roll: np.ndarray
    period_roll: float
    pitch: np.ndarray | None = None

    def amplitude(self, period, mode: str = "heave") -> np.ndarray:
        """``|H(T)|`` interpolated to the periods asked for, zero outside the band it was computed on."""
        table = np.atleast_1d(np.asarray(self.periods, dtype=float))
        values = np.atleast_1d(np.asarray(getattr(self, mode), dtype=float))
        period = np.atleast_1d(np.asarray(period, dtype=float))
        return np.interp(period, table, values, left=0.0, right=0.0)

    def __call__(self, period, mode: str = "heave") -> np.ndarray:
        return self.amplitude(period, mode)

    def with_pitch(self, pitch: np.ndarray) -> TransferFunction:
        return TransferFunction(periods=self.periods, heave=self.heave, sway=self.sway, roll=self.roll,
                                period_roll=self.period_roll, pitch=np.asarray(pitch, dtype=float))


def sweep(section: Section, periods=None, *, periods_per_decade: int = 7) -> TransferFunction:
    """Solve the section over a band of periods (4 s to 25 s by default) and return its transfer functions."""
    if periods is None:
        periods = np.geomspace(4.0, 25.0, 4 * periods_per_decade)
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    rows = [solve(section, 2.0 * math.pi / float(period)) for period in periods]
    roll = np.array([row["rao_roll"] for row in rows])
    return TransferFunction(periods=periods, heave=np.array([row["rao_heave"] for row in rows]),
                            sway=np.array([row["rao_sway"] for row in rows]), roll=roll,
                            period_roll=float(periods[int(np.argmax(roll))]))


def pitch_transfer_function(length: float, draught: float, depth: float | None = None, *,
                            gyration: float | None = None, periods=None, kg: float | None = None,
                            roll_damping: float | None = 0.08) -> np.ndarray:
    """The pitch response per radian of wave slope, from the *longitudinal* section.

    Pitch is not a two-dimensional mode, and this is not a pitch RAO of a ship: it is the roll problem of the section
    cut along the length, which is what a strip calculation uses for the longitudinal modes. The restoring is the
    longitudinal metacentric height ``GM_L = L^2/(12 T) + T/2 - KG``, the inertia is ``m k_yy^2`` with ``k_yy`` a
    quarter of the length when it is not given, and the result has the right limits -- one at long waves, falling away
    as the waves shorten -- which is all a two-dimensional stand-in can claim.
    """
    kg_value = draught / 2.0 if kg is None else kg
    gm_long = length ** 2 / (12.0 * draught) + draught / 2.0 - kg_value
    section = Section(beam=length, draught=draught, depth=depth, kg=kg_value,
                      gyration=0.25 * length if gyration is None else gyration, roll_damping=roll_damping)
    table = np.geomspace(4.0, 25.0, 28) if periods is None else np.atleast_1d(np.asarray(periods, dtype=float))
    assert gm_long > 0, "the longitudinal metacentric height has to be positive"
    return np.array([solve(section, 2.0 * math.pi / float(period))["rao_roll"] for period in table])
