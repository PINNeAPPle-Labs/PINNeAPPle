"""Finite-difference time-domain (FDTD) solution of Maxwell's curl equations on the Yee grid, in 1-D, 2-D and 3-D.

One code path for all dimensions: the grid has ``(nx, ny, nz)`` nodes and an axis with one node is a direction of
invariance (``d/dz = 0`` in 2-D, where the six components hold the TMz set ``Ez, Hx, Hy`` and the TEz set
``Hz, Ex, Ey`` at once). Component positions follow Yee (1966): ``Ex`` at ``(i+1/2, j, k)``, ``Ey`` at
``(i, j+1/2, k)``, ``Ez`` at ``(i, j, k+1/2)``, ``Hx`` at ``(i, j+1/2, k+1/2)`` and so on; E at integer time steps,
H at half steps (leapfrog). References: A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
Finite-Difference Time-Domain Method*, 3rd ed., Artech House, 2005 (TH below), ch. 3-4 (scheme, Courant limit,
numerical dispersion), ch. 5 (total-field / scattered-field), ch. 7 (CPML), ch. 9 (dispersive media).

* **Stability.** ``dt <= 1 / (c sqrt(sum 1/dx_i^2))`` (TH eq. 4.65); :class:`YeeGrid` refuses a larger step
  unless ``allow_unstable=True`` (used by the test that shows the blow-up).
* **Boundaries** per side: ``"pec"`` (tangential E = 0 on the outer node plane), ``"pmc"`` (tangential H = 0 half a
  cell outside it, by an odd image), ``"periodic"``, ``"bloch"`` (Floquet phase ``exp(-j k_b L)`` across the
  period; complex fields) and ``"cpml"``: the convolutional PML of Roden and Gedney (Microw. Opt. Technol. Lett. 27,
  2000; TH sec. 7.9) on the outer ``npml`` cells with polynomial grading of ``sigma`` and ``kappa`` and a linear
  ``alpha`` (TH eqs. 7.60, 7.79, 7.102), terminated by PEC.
* **Media.** Per-node ``eps_inf``, ``mu_r``, ``sigma``, magnetic ``sigma_m`` (an E component takes the harmonic mean
  of the permittivity of the two nodes it joins, the series rule for E normal to an interface between them, and the
  arithmetic mean of the conductivity; ``harmonic_eps=False`` uses the arithmetic mean), and dispersive poles (:mod:`.media`): Drude (polarisation
  current, trapezoidal ADE), Debye (polarisation, trapezoidal) and Lorentz (polarisation, central second order),
  each solved together with the E update (TH sec. 9.4-9.6).
* **Sources.** Soft or hard E sources, impressed currents ``J`` (a Hertzian dipole is one cell of ``J``), and a
  total-field / scattered-field box for an axis-aligned plane wave, driven by an auxiliary 1-D FDTD with the same
  ``dx`` and ``dt``, so the incident wave has the main grid's own numerical dispersion and leaks only at round-off.
* **Monitors.** Probes (time series), running DFTs of any component over any region (phasors with the
  ``exp(+j omega t)`` convention of this package, E and H stamped at their own times), Poynting flux through a plane
  from the DFTs, and the discrete electromagnetic energy.

:func:`numerical_dispersion` gives the exact dispersion relation of the scheme, :func:`courant_limit` the step
limit.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .constants import C0, EPS0, ETA0, MU0
from .media import Debye, Drude, Lorentz, Medium

__all__ = [
    "YeeGrid", "FDTD", "courant_limit", "numerical_dispersion", "gaussian_pulse", "modulated_gaussian", "ricker",
    "sinusoid", "Probe", "DFTMonitor", "PlaneWaveTFSF", "fill_fraction",
]

AXES = "xyz"
E_NAMES = ("Ex", "Ey", "Ez")
H_NAMES = ("Hx", "Hy", "Hz")


def fill_fraction(grid: YeeGrid, inside: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                  oversample: int = 8) -> np.ndarray:
    """Fraction of each node's cell (the box of one spacing centred on the node, along the active axes) for which
    ``inside(x, y, z)`` is true, by ``oversample`` points per active axis. Node ``(i, j, k)`` is at
    ``(i dx, j dy, k dz)``."""
    o = int(oversample)
    offs = [(np.arange(o) + 0.5) / o - 0.5 if a else np.zeros(1) for a in grid.active]
    nodes = [np.arange(n) * d for n, d in zip(grid.shape, grid.spacing, strict=True)]
    frac = np.zeros(grid.shape)
    count = 0
    for ox in offs[0]:
        for oy in offs[1]:
            for oz in offs[2]:
                X, Y, Z = np.meshgrid(nodes[0] + ox * grid.spacing[0], nodes[1] + oy * grid.spacing[1],
                                      nodes[2] + oz * grid.spacing[2], indexing="ij")
                frac += np.asarray(inside(X, Y, Z), float)
                count += 1
    return frac / count


def courant_limit(spacing: Sequence[float], dims: Sequence[bool] | None = None) -> float:
    """Largest stable time step ``1 / (c sqrt(sum_i 1/dx_i^2))`` over the active axes (TH eq. 4.65)."""
    dims = dims if dims is not None else [True] * len(spacing)
    s = sum(1.0 / d ** 2 for d, a in zip(spacing, dims, strict=True) if a)
    return 1.0 / (C0 * math.sqrt(s))


def numerical_dispersion(omega: float, dt: float, spacing: Sequence[float], direction: Sequence[float],
                         n: float = 1.0) -> float:
    """Wavenumber ``k`` of the Yee scheme along unit ``direction`` at angular frequency ``omega`` in a medium of
    index ``n``: the root of ``(n / (c dt) sin(omega dt / 2))^2 = sum_i (sin(k u_i dx_i / 2) / dx_i)^2``
    (TH eq. 4.12), solved by bisection/Newton from the continuum value."""
    from scipy.optimize import brentq

    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    lhs = (n / (C0 * dt) * math.sin(omega * dt / 2)) ** 2

    def f(k):
        return sum((math.sin(k * ui * d / 2) / d) ** 2 for ui, d in zip(u, spacing, strict=True)) - lhs

    k0 = n * omega / C0
    kmax = min(math.pi / (abs(ui) * d) for ui, d in zip(u, spacing, strict=True) if abs(ui) > 1e-15)
    return float(brentq(f, 0.5 * k0, kmax * (1 - 1e-12)))


# ------------------------------------------------------------------------------------------- waveforms
def gaussian_pulse(t0: float, tau: float) -> Callable[[float], float]:
    """``exp(-((t - t0)/tau)^2)``."""
    return lambda t: math.exp(-((t - t0) / tau) ** 2)


def modulated_gaussian(f0: float, bandwidth: float, t0: float | None = None) -> Callable[[float], float]:
    """``sin(2 pi f0 (t - t0)) exp(-((t - t0)/tau)^2)`` with ``tau = 1 / (pi bandwidth)`` (zero mean, no DC)."""
    tau = 1.0 / (math.pi * bandwidth)
    t0 = 4.0 * tau if t0 is None else t0
    return lambda t: math.sin(2 * math.pi * f0 * (t - t0)) * math.exp(-((t - t0) / tau) ** 2)


def ricker(fp: float, t0: float | None = None) -> Callable[[float], float]:
    """Ricker wavelet with peak frequency ``fp``."""
    t0 = 1.5 / fp if t0 is None else t0
    return lambda t: (1 - 2 * (math.pi * fp * (t - t0)) ** 2) * math.exp(-(math.pi * fp * (t - t0)) ** 2)


def sinusoid(f0: float, ramp_periods: float = 3.0) -> Callable[[float], float]:
    """``sin(2 pi f0 t)`` switched on with a smooth ramp over ``ramp_periods`` periods."""
    tr = ramp_periods / f0

    def w(t):
        r = 1.0 if t >= tr else 0.5 * (1 - math.cos(math.pi * max(t, 0.0) / tr))
        return r * math.sin(2 * math.pi * f0 * t)
    return w


# ------------------------------------------------------------------------------------------- grid
@dataclass
class YeeGrid:
    """Node counts ``shape = (nx, ny, nz)`` (use 1 for invariant axes), spacings and the time step (``courant`` is
    the fraction of the stability limit)."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    courant: float = 0.99
    dt: float | None = None
    allow_unstable: bool = False

    def __post_init__(self):
        self.shape = tuple(int(s) for s in self.shape)
        self.spacing = tuple(float(d) for d in self.spacing)
        self.active = tuple(s > 1 for s in self.shape)
        self.dt_max = courant_limit(self.spacing, self.active)
        if self.dt is None:
            self.dt = self.courant * self.dt_max
        if self.dt > self.dt_max * (1 + 1e-12) and not self.allow_unstable:
            raise ValueError(f"dt = {self.dt:.4g} s exceeds the Courant limit {self.dt_max:.4g} s "
                             f"(c dt sqrt(sum 1/dx^2) = {self.dt / self.dt_max:.4f} > 1): the scheme is unstable")

    @property
    def ndim(self) -> int:
        return sum(self.active)

    def coords(self, component: str) -> list[np.ndarray]:
        """Coordinates of the nodes of a component along x, y, z (half-cell offsets of the Yee cell)."""
        off = _offsets(component)
        return [(np.arange(n) + o) * d for n, o, d in zip(self.shape, off, self.spacing, strict=True)]


def _offsets(component: str) -> tuple[float, float, float]:
    a = AXES.index(component[1])
    if component[0] == "E":
        return tuple(0.5 if i == a else 0.0 for i in range(3))
    return tuple(0.0 if i == a else 0.5 for i in range(3))


# ------------------------------------------------------------------------------------------- monitors
@dataclass
class Probe:
    component: str
    index: tuple[int, int, int]
    values: list = field(default_factory=list)
    times: list = field(default_factory=list)

    @property
    def series(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.times), np.asarray(self.values)


@dataclass
class DFTMonitor:
    """Running DFT ``F(omega) = sum_n f(t_n) exp(-j omega t_n) dt`` of components over a region (slices)."""

    components: tuple[str, ...]
    freqs: np.ndarray
    region: tuple[slice, slice, slice]
    data: dict = field(default_factory=dict)

    def phasor(self, component: str) -> np.ndarray:
        """``(n_freq, *region_shape)`` complex phasors."""
        return self.data[component]


@dataclass
class PlaneWaveTFSF:
    """Total-field / scattered-field box ``box = ((i0, i1), (j0, j1), (k0, k1))`` (node index ranges, inclusive of
    i0, exclusive of i1) for a plane wave travelling along +``axis`` with electric field ``polarization`` and
    waveform ``waveform(t)`` of the incident E at the box's first plane."""

    axis: str
    polarization: str
    waveform: Callable[[float], float]
    box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


# ------------------------------------------------------------------------------------------- solver
class FDTD:
    """Yee leapfrog solver. ``boundaries`` maps ``"x"``, ``"y"``, ``"z"`` to a kind or a ``(low, high)`` pair of
    ``"pec"``, ``"pmc"``, ``"periodic"``, ``"bloch"`` or ``"cpml"``; ``bloch`` takes ``bloch_k`` (rad/m) per axis."""

    def __init__(self, grid: YeeGrid, boundaries: Mapping[str, str | tuple[str, str]] | None = None, *,
                 npml: int = 10, pml_order: float = 3.0, pml_kappa_max: float = 1.0, pml_alpha_max: float | None = None,
                 pml_sigma_factor: float = 1.0, bloch_k: Mapping[str, float] | None = None,
                 harmonic_eps: bool = True):
        self.g = grid
        self.dt = grid.dt
        self.harmonic_eps = harmonic_eps
        self.npml = npml
        b = dict(boundaries or {})
        self.bc: dict[str, tuple[str, str]] = {}
        for a in AXES:
            v = b.get(a, "pec")
            self.bc[a] = (v, v) if isinstance(v, str) else tuple(v)
            if ("periodic" in self.bc[a] or "bloch" in self.bc[a]) and self.bc[a][0] != self.bc[a][1]:
                raise ValueError(f"axis {a}: periodic and Bloch boundaries apply to both sides")
        self.bloch = dict(bloch_k or {})
        self.complex = any("bloch" in v for v in self.bc.values())
        dtype = complex if self.complex else float
        shp = grid.shape
        self.E = {n: np.zeros(shp, dtype) for n in E_NAMES}
        self.H = {n: np.zeros(shp, dtype) for n in H_NAMES}
        # node materials
        self.eps_inf = np.ones(shp)
        self.mu_r = np.ones(shp)
        self.sigma = np.zeros(shp)
        self.sigma_m = np.zeros(shp)
        self._poles: list[tuple[np.ndarray, object]] = []
        self.sources: list[tuple[str, tuple, Callable, str]] = []
        self.currents: list[tuple[str, tuple, Callable, float]] = []
        self.probes: list[Probe] = []
        self.dfts: list[DFTMonitor] = []
        self.tfsf: list[dict] = []
        self.n = 0
        self._built = False
        # alpha_max (S/m) shifts the CPML pole off zero frequency so evanescent and low-frequency waves are absorbed
        # too; 0 gives the stretched-coordinate PML (TH sec. 7.9)
        self._pml_params = dict(order=pml_order, kappa_max=pml_kappa_max,
                                alpha_max=0.0 if pml_alpha_max is None else pml_alpha_max,
                                sigma_factor=pml_sigma_factor)

    # -------------------------------------------------------------------------------------- set-up
    def set_material(self, mask, medium: Medium) -> None:
        """Assign a medium to the nodes selected by ``mask``: a boolean array of the grid shape or slices, or a float
        array of fill fractions in ``[0, 1]`` (see :func:`fill_fraction`). A fraction ``f`` mixes arithmetically,
        ``eps = f eps_medium + (1 - f) eps_before`` (and ``mu``, ``sigma``, ``sigma_m``), and scales the medium's
        poles by ``f`` (their susceptibility is linear in ``delta_eps`` and ``omega_p^2``). Arithmetic averaging is
        the exact sub-cell rule for E tangential to an interface (TMz around a z-directed cylinder); for other
        polarisations it reduces, but does not remove, the staircase error (Farjadpour et al., Opt. Lett. 31,
        2006)."""
        mask = np.asarray(mask) if not isinstance(mask, tuple) and not isinstance(mask, slice) else mask
        if isinstance(mask, np.ndarray) and mask.dtype.kind == "f":
            f = np.broadcast_to(np.clip(mask, 0.0, 1.0), self.g.shape)
        else:
            m = np.zeros(self.g.shape, bool)
            m[mask] = True
            f = m.astype(float)
        sel = f > 0
        for arr, v in ((self.eps_inf, medium.eps_inf), (self.mu_r, medium.mu_r), (self.sigma, medium.sigma),
                       (self.sigma_m, medium.sigma_m)):
            arr[sel] = f[sel] * v + (1 - f[sel]) * arr[sel]
        full = f >= 1.0
        # poles: a full assignment replaces earlier poles on those nodes; partial fills add scaled poles
        self._poles = [(pm * ~full, p) for pm, p in self._poles if np.any(pm * ~full)]
        for p in medium.poles:
            if isinstance(p, Drude):
                q = lambda w, p=p: Drude(p.omega_p * math.sqrt(w), p.gamma)  # noqa: E731
            elif isinstance(p, Lorentz):
                q = lambda w, p=p: Lorentz(p.delta_eps * w, p.omega0, p.delta)  # noqa: E731
            else:
                q = lambda w, p=p: Debye(p.delta_eps * w, p.tau)  # noqa: E731
            for w in np.unique(f[sel]):
                self._poles.append((f == w, q(float(w))))
        self._built = False

    def add_source(self, component: str, index, waveform: Callable[[float], float], kind: str = "soft") -> None:
        """E source at a node: ``soft`` adds ``waveform(t)`` after the update, ``hard`` imposes it."""
        self.sources.append((component, tuple(index), waveform, kind))

    def add_current(self, component: str, index, waveform: Callable[[float], float], amplitude: float = 1.0) -> None:
        """Impressed current density ``J = amplitude * waveform(t)`` (A/m^2) at an E node; ``J dx dy dz`` is the
        current moment ``I l`` of a Hertzian dipole."""
        self.currents.append((component, tuple(index), waveform, amplitude))

    def add_probe(self, component: str, index) -> Probe:
        p = Probe(component, tuple(index))
        self.probes.append(p)
        return p

    def add_dft(self, components: Sequence[str], freqs, region=(slice(None),) * 3) -> DFTMonitor:
        freqs = np.atleast_1d(np.asarray(freqs, float))
        region = tuple(region)
        mon = DFTMonitor(tuple(components), freqs, region)
        for c in components:
            shape = np.empty(self.g.shape)[region].shape
            mon.data[c] = np.zeros((len(freqs),) + shape, complex)
        self.dfts.append(mon)
        return mon

    def add_plane_wave(self, tfsf: PlaneWaveTFSF) -> None:
        """A total-field / scattered-field plane wave (see :class:`PlaneWaveTFSF`); vacuum background assumed."""
        a = AXES.index(tfsf.axis)
        pa = AXES.index(tfsf.polarization[1])
        if pa == a:
            raise ValueError("the polarisation must be transverse to the propagation axis")
        hc = 3 - a - pa                                # H component along the remaining axis
        sign = 1.0 if (a, pa, hc) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0  # H = sign * E / eta
        n_ax = self.g.shape[a]
        dx = self.g.spacing[a]
        # auxiliary 1-D line along the axis: margin, the main grid's span, margin, CPML at both ends
        margin = 20
        npml1 = 20
        n1 = n_ax + 2 * margin + 2 * npml1
        shp = [1, 1, 1]
        shp[0] = n1
        sp = [dx, dx, dx]
        aux = FDTD(YeeGrid(tuple(shp), tuple(sp), dt=self.dt), {"x": "cpml", "y": "periodic", "z": "periodic"},
                   npml=npml1)
        i_src = npml1 + 2
        offset = npml1 + margin                         # aux index of main index 0 along the axis
        i0 = tfsf.box[a][0]
        # the waveform is the incident E at the first TF plane: the hard source on the line runs ahead by the
        # (continuum) travel time to it; the line's numerical dispersion is the main grid's own
        delay = (offset + i0 - i_src) * dx / C0
        aux.add_source("Ey", (i_src, 0, 0), lambda t, w=tfsf.waveform, d=delay: w(t + d), "hard")
        self.tfsf.append(dict(spec=tfsf, aux=aux, a=a, pa=pa, hc=hc, sign=sign, offset=offset, i_src=i_src))
        self._built = False

    def incident_monitor(self, freqs, k: int = 0) -> DFTMonitor:
        """DFT of the ``k``-th plane wave's incident E along the propagation axis over the main grid's span (on the
        auxiliary line): ``mon.phasor("Ey")[:, i]`` is the incident phasor at main index ``i`` along the axis."""
        t = self.tfsf[k]
        n = self.g.shape[t["a"]]
        return t["aux"].add_dft(["Ey"], freqs, (slice(t["offset"], t["offset"] + n), slice(None), slice(None)))

    # -------------------------------------------------------------------------------------- build
    def _component_avg(self, arr: np.ndarray, axis: int, harmonic: bool = False) -> np.ndarray:
        """Mean of the two nodes an E component (along ``axis``) joins; the last node keeps its own value. The
        harmonic mean is the right one for the permittivity seen by a component normal to an interface between
        the two nodes (the series rule for D continuous)."""
        if arr.shape[axis] == 1:
            return arr.copy()
        nxt = np.concatenate([np.take(arr, np.arange(1, arr.shape[axis]), axis=axis),
                              np.take(arr, [-1], axis=axis)], axis=axis)
        if harmonic:
            return 2.0 * arr * nxt / (arr + nxt)
        return 0.5 * (arr + nxt)

    def _build(self) -> None:
        dt = self.dt
        self.CE_new, self.CE_old, self.poleE = {}, {}, {}
        for k, n in enumerate(E_NAMES):
            eps = EPS0 * self._component_avg(self.eps_inf, k, harmonic=self.harmonic_eps)
            sig = self._component_avg(self.sigma, k)
            cnew = eps / dt + sig / 2
            cold = eps / dt - sig / 2
            states = []
            for mask, p in self._poles:
                mc = self._component_avg(mask.astype(float), k) > 0.25
                idx = np.flatnonzero(mc)
                if idx.size == 0:
                    continue
                if isinstance(p, Drude):
                    ka, kb = p.ade(dt)
                    np.put(cnew, idx, cnew.flat[idx] + kb / 2)
                    np.put(cold, idx, cold.flat[idx] - kb / 2)
                    states.append(dict(kind="drude", idx=idx, ka=ka, kb=kb, J=np.zeros(idx.size, self.E[n].dtype)))
                elif isinstance(p, Debye):
                    ka, kb = p.ade(dt)
                    np.put(cnew, idx, cnew.flat[idx] + kb / dt)
                    np.put(cold, idx, cold.flat[idx] - kb / dt)
                    states.append(dict(kind="debye", idx=idx, ka=ka, kb=kb, P=np.zeros(idx.size, self.E[n].dtype)))
                elif isinstance(p, Lorentz):
                    a1, a2, a3 = p.ade(dt)
                    z = np.zeros(idx.size, self.E[n].dtype)
                    states.append(dict(kind="lorentz", idx=idx, a1=a1, a2=a2, a3=a3, P=z, Pm=z.copy()))
                else:
                    raise TypeError(f"unknown pole {p!r}")
            self.CE_new[n], self.CE_old[n], self.poleE[n] = cnew, cold, states
        self.DH_a, self.DH_b = {}, {}
        for n in H_NAMES:
            # H nodes are offset along the two other axes; take the node value (staircase)
            mu = MU0 * self.mu_r
            sm = self.sigma_m
            r = sm * dt / (2 * mu)
            self.DH_a[n] = (1 - r) / (1 + r)
            self.DH_b[n] = (dt / mu) / (1 + r)
        self._build_pml()
        self._build_tfsf()
        self._built = True

    def _profile(self, n: int, d: float, half: bool, low: bool, high: bool):
        """CPML coefficients ``(b, c, 1/kappa)`` along one axis at integer (E-derivative) or half-integer
        (H-derivative) positions."""
        npml = self.npml
        P = self._pml_params
        m = P["order"]
        sigma_max = P["sigma_factor"] * 0.8 * (m + 1) / (ETA0 * d)
        pos = np.arange(n) + (0.5 if half else 0.0)
        depth = np.zeros(n)
        if low:
            depth = np.maximum(depth, (npml - pos) / npml)
        if high:
            depth = np.maximum(depth, (pos - (n - 1 - npml)) / npml)
        depth = np.clip(depth, 0.0, 1.0)
        sig = sigma_max * depth ** m
        kap = 1.0 + (P["kappa_max"] - 1.0) * depth ** m
        alp = P["alpha_max"] * (1.0 - depth) * (depth > 0)
        b = np.exp(-(sig / kap + alp) * self.dt / EPS0)
        den = sig * kap + kap * kap * alp
        c = np.where(den > 0, sig / np.where(den > 0, den, 1.0) * (b - 1.0), 0.0)
        return b, c, 1.0 / kap, depth > 0

    def _build_pml(self) -> None:
        self.pml = {}
        for k, a in enumerate(AXES):
            lo, hi = (s == "cpml" for s in self.bc[a])
            if not (lo or hi) or self.g.shape[k] == 1:
                continue
            n = self.g.shape[k]
            prof_e = self._profile(n, self.g.spacing[k], False, lo, hi)
            prof_h = self._profile(n, self.g.spacing[k], True, lo, hi)
            shape = [1, 1, 1]
            shape[k] = n
            self.pml[k] = {
                "E": tuple(p.reshape(shape) for p in prof_e[:3]),
                "H": tuple(p.reshape(shape) for p in prof_h[:3]),
                "psiE": {}, "psiH": {},
            }

    def _build_tfsf(self) -> None:
        for t in self.tfsf:
            spec = t["spec"]
            M = {}
            box = spec.box
            for comp in E_NAMES + H_NAMES:
                off = _offsets(comp)
                m = np.ones(self.g.shape, bool)
                for ax in range(3):
                    lo, hi = box[ax]
                    pos = np.arange(self.g.shape[ax]) + off[ax]
                    # a node is in the total-field region if lo <= pos <= hi - 1 (with half-cell positions
                    # inside the box's node span)
                    inside = (pos >= lo - 1e-9) & (pos <= hi - 1 + 1e-9) if self.g.shape[ax] > 1 else np.ones(1, bool)
                    sh = [1, 1, 1]
                    sh[ax] = self.g.shape[ax]
                    m = m & inside.reshape(sh)
                M[comp] = m
            t["M"] = M

    # -------------------------------------------------------------------------------------- operators
    def _shift(self, F: np.ndarray, ax: int, forward: bool, kind_side: tuple[str, str]) -> np.ndarray:
        """``F`` at the next (forward) or previous node along ``ax`` with the boundary's ghost value."""
        n = F.shape[ax]
        kind_lo, kind_hi = kind_side
        if kind_lo in ("periodic", "bloch"):
            out = np.roll(F, -1 if forward else 1, axis=ax)
            if kind_lo == "bloch":
                L = n * self.g.spacing[ax]
                ph = np.exp(-1j * self.bloch.get(AXES[ax], 0.0) * L)
                sl = [slice(None)] * 3
                if forward:          # node n-1 looks at node n = node 0 of the next period
                    sl[ax] = slice(n - 1, n)
                    out[tuple(sl)] = out[tuple(sl)] * ph
                else:                # node 0 looks at node -1 = node n-1 of the previous period
                    sl[ax] = slice(0, 1)
                    out[tuple(sl)] = out[tuple(sl)] / ph
            return out
        out = np.zeros_like(F)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if forward:
            src[ax], dst[ax] = slice(1, n), slice(0, n - 1)
        else:
            src[ax], dst[ax] = slice(0, n - 1), slice(1, n)
        out[tuple(dst)] = F[tuple(src)]
        return out

    def _dfwd(self, F, ax):
        if self.g.shape[ax] == 1:
            return None
        return (self._shift(F, ax, True, self.bc[AXES[ax]]) - F) / self.g.spacing[ax]

    def _dbwd(self, F, ax, h_tangential: bool):
        if self.g.shape[ax] == 1:
            return None
        prev = self._shift(F, ax, False, self.bc[AXES[ax]])
        if h_tangential and self.bc[AXES[ax]][0] == "pmc":
            sl = [slice(None)] * 3
            sl[ax] = slice(0, 1)
            prev[tuple(sl)] = -F[tuple(sl)]
        return (F - prev) / self.g.spacing[ax]

    def _pml_term(self, deriv: np.ndarray, ax: int, kind: str, name: str) -> np.ndarray:
        """Replace ``d/dx_ax`` by ``(1/kappa) d/dx_ax + psi`` inside the CPML."""
        if ax not in self.pml:
            return deriv
        P = self.pml[ax]
        b, c, ik = P[kind]
        store = P["psi" + kind]
        key = name
        if key not in store:
            store[key] = np.zeros_like(deriv)
        psi = store[key]
        psi *= b
        psi += c * deriv
        return ik * deriv + psi

    def _curl_H(self) -> dict[str, np.ndarray]:
        H = self.H
        out = {}
        for k, n in enumerate(E_NAMES):
            a1, a2 = (k + 1) % 3, (k + 2) % 3
            tot = np.zeros_like(self.E[n])
            d1 = self._dbwd(H[H_NAMES[a2]], a1, True)        # d H_a2 / d x_a1
            if d1 is not None:
                tot = tot + self._pml_term(d1, a1, "E", f"{n}{a1}")
            d2 = self._dbwd(H[H_NAMES[a1]], a2, True)        # d H_a1 / d x_a2
            if d2 is not None:
                tot = tot - self._pml_term(d2, a2, "E", f"{n}{a2}")
            out[n] = tot
        return out

    def _curl_E(self) -> dict[str, np.ndarray]:
        E = self.E
        out = {}
        for k, n in enumerate(H_NAMES):
            a1, a2 = (k + 1) % 3, (k + 2) % 3
            tot = np.zeros_like(self.H[n])
            d1 = self._dfwd(E[E_NAMES[a2]], a1)
            if d1 is not None:
                tot = tot + self._pml_term(d1, a1, "H", f"{n}{a1}")
            d2 = self._dfwd(E[E_NAMES[a1]], a2)
            if d2 is not None:
                tot = tot - self._pml_term(d2, a2, "H", f"{n}{a2}")
            out[n] = tot
        return out

    def _raw_curl(self, F: dict, kind: str) -> dict:
        """Curl without PML terms (for the TF/SF corrections, which sit away from the PML)."""
        out = {}
        names = E_NAMES if kind == "E" else H_NAMES
        src = H_NAMES if kind == "E" else E_NAMES
        for k, n in enumerate(names):
            a1, a2 = (k + 1) % 3, (k + 2) % 3
            tot = np.zeros(self.g.shape, complex if self.complex else float)
            for ax, comp, s in ((a1, src[a2], 1.0), (a2, src[a1], -1.0)):
                if self.g.shape[ax] == 1:
                    continue
                f = F.get(comp)
                if f is None:
                    continue
                d = self._dbwd(f, ax, kind == "E") if kind == "E" else self._dfwd(f, ax)
                tot = tot + s * d
            out[n] = tot
        return out

    # -------------------------------------------------------------------------------------- incident field
    def _incident(self, t: dict, which: str) -> dict[str, np.ndarray]:
        """Incident E (at step n) or H (at n + 1/2) on the main grid from the auxiliary line."""
        aux = t["aux"]
        a, pa, hc = t["a"], t["pa"], t["hc"]
        n = self.g.shape[a]
        sh = [1, 1, 1]
        sh[a] = n
        sl = slice(t["offset"], t["offset"] + n)
        if which == "E":
            line = aux.E["Ey"][sl, 0, 0]                     # E at integer positions along the line
            comp = E_NAMES[pa]
            val = line.reshape(sh)
            # the polarisation component sits at integer positions along the propagation axis
            return {comp: np.broadcast_to(val, self.g.shape)}
        line = aux.H["Hz"][sl, 0, 0]                         # H at half positions along the line
        comp = H_NAMES[hc]
        # aux: Ey along y, Hz along z for propagation +x gives H_z = +E_y / eta; map the sign to the main triad
        val = t["sign"] * line.reshape(sh)
        return {comp: np.broadcast_to(val, self.g.shape)}

    # -------------------------------------------------------------------------------------- stepping
    def step(self) -> None:
        """One leapfrog step: H to n + 1/2, then E to n + 1 (auxiliary plane-wave lines in lockstep)."""
        self.half_step_h()
        for t in self.tfsf:
            t["aux"].half_step_h()
        self.half_step_e()
        for t in self.tfsf:
            t["aux"].half_step_e()

    def _tfsf_correct(self, curl: dict, kind: str) -> None:
        """Total-field / scattered-field consistency: a node on one side whose stencil reaches the other side gets
        the incident field of the far side added (TF node) or removed (SF node), TH sec. 5.6, written as masked
        curls so it holds for any box in 1-D, 2-D and 3-D."""
        for t in self.tfsf:
            inc = self._incident(t, "E" if kind == "H" else "H")
            M = t["M"]
            c_in = self._raw_curl({c: v * M[c] for c, v in inc.items()}, kind)
            c_out = self._raw_curl({c: v * (~M[c]) for c, v in inc.items()}, kind)
            for n in (H_NAMES if kind == "H" else E_NAMES):
                curl[n] = curl[n] - c_in[n] * (~M[n]) + c_out[n] * M[n]

    def half_step_h(self) -> None:
        if not self._built:
            self._build()
        cE = self._curl_E()
        self._tfsf_correct(cE, "H")
        for n in H_NAMES:
            self.H[n] = self.DH_a[n] * self.H[n] - self.DH_b[n] * cE[n]
        self._bc_h()

    def half_step_e(self) -> None:
        dt = self.dt
        cH = self._curl_H()
        self._tfsf_correct(cH, "E")
        t_j = (self.n + 0.5) * dt
        for comp, idx, w, amp in self.currents:
            cH[comp][idx] -= amp * w(t_j)
        self._E_prev = {n: v.copy() for n, v in self.E.items()}
        for n in E_NAMES:
            old = self.E[n]
            rhs = cH[n] + self.CE_old[n] * old
            states = self.poleE[n]
            for st in states:
                if st["kind"] == "drude":
                    rhs.flat[st["idx"]] -= 0.5 * (1 + st["ka"]) * st["J"]
                elif st["kind"] == "debye":
                    rhs.flat[st["idx"]] -= (st["ka"] - 1) / dt * st["P"]
                else:
                    Pn1 = st["a1"] * st["P"] + st["a2"] * st["Pm"] + st["a3"] * old.flat[st["idx"]]
                    rhs.flat[st["idx"]] -= (Pn1 - st["P"]) / dt
                    st["Pm"], st["P"] = st["P"], Pn1
            new = rhs / self.CE_new[n]
            for st in states:
                En, En1 = old.flat[st["idx"]], new.flat[st["idx"]]
                if st["kind"] == "drude":
                    st["J"] = st["ka"] * st["J"] + st["kb"] * (En1 + En)
                elif st["kind"] == "debye":
                    st["P"] = st["ka"] * st["P"] + st["kb"] * (En1 + En)
            self.E[n] = new
        self.n += 1
        t_new = self.n * dt
        for comp, idx, w, kind in self.sources:
            if kind == "hard":
                self.E[comp][idx] = w(t_new)
            else:
                self.E[comp][idx] += w(t_new)
        self._bc_e()
        self._record(t_new)

    def _bc_e(self) -> None:
        """PEC (and the PEC behind a CPML): tangential E = 0 on the outer node planes."""
        for ax, a in enumerate(AXES):
            if self.g.shape[ax] == 1:
                continue
            lo, hi = self.bc[a]
            for side, kind in ((0, lo), (-1, hi)):
                if kind not in ("pec", "cpml"):
                    continue
                for k, n in enumerate(E_NAMES):
                    if k == ax:
                        if side == -1:       # the normal component beyond the last node lies outside
                            sl = [slice(None)] * 3
                            sl[ax] = slice(-1, None)
                            self.E[n][tuple(sl)] = 0
                        continue
                    sl = [slice(None)] * 3
                    sl[ax] = slice(0, 1) if side == 0 else slice(-1, None)
                    self.E[n][tuple(sl)] = 0

    def _bc_h(self) -> None:
        """PMC on the high side: tangential H at the last half plane is the odd image of the one before."""
        for ax, a in enumerate(AXES):
            if self.g.shape[ax] == 1 or self.bc[a][1] != "pmc":
                continue
            for k, n in enumerate(H_NAMES):
                if k == ax:
                    continue
                sl = [slice(None)] * 3
                sl[ax] = slice(-1, None)
                prev = [slice(None)] * 3
                prev[ax] = slice(-2, -1)
                self.H[n][tuple(sl)] = -self.H[n][tuple(prev)]

    def _record(self, t_e: float) -> None:
        dt = self.dt
        for p in self.probes:
            src = self.E if p.component.startswith("E") else self.H
            tt = t_e if p.component.startswith("E") else t_e - 0.5 * dt
            p.values.append(src[p.component][p.index])
            p.times.append(tt)
        for m in self.dfts:
            for c in m.components:
                src = self.E if c.startswith("E") else self.H
                tt = t_e if c.startswith("E") else t_e - 0.5 * dt
                ph = np.exp(-1j * 2 * np.pi * m.freqs * tt) * dt
                v = src[c][m.region]
                m.data[c] += ph.reshape((-1,) + (1,) * v.ndim) * v[None]

    def run(self, steps: int, every: int = 0, callback: Callable[[FDTD], None] | None = None) -> None:
        for _ in range(int(steps)):
            self.step()
            if every and callback is not None and self.n % every == 0:
                callback(self)

    def run_until(self, time: float, **kw) -> None:
        self.run(int(math.ceil((time - self.n * self.dt) / self.dt)), **kw)

    # -------------------------------------------------------------------------------------- diagnostics
    def energy(self) -> float:
        """Discrete electromagnetic energy at ``t = (n - 1/2) dt``:
        ``W = (1/2) sum eps E^{n-1} . E^n dV + (1/2) sum mu |H^{n-1/2}|^2 dV``. In a lossless cavity with PEC or PMC
        walls the Yee scheme conserves this quantity exactly (to round-off) for any stable time step; before the
        first step it is the plain energy of the initial fields."""
        dV = float(np.prod([d for d, a in zip(self.g.spacing, self.g.active, strict=True) if a]))
        prev = getattr(self, "_E_prev", None)
        we = 0.0
        for k, n in enumerate(E_NAMES):
            eps = EPS0 * self._component_avg(self.eps_inf, k)
            other = self.E[n] if prev is None else prev[n]
            we += float(np.sum(eps * np.real(self.E[n] * np.conj(other))))
        wh = sum(float(np.sum(MU0 * self.mu_r * np.abs(self.H[n]) ** 2)) for n in H_NAMES)
        return 0.5 * (we + wh) * dV

    def snapshot(self) -> tuple[float, dict[str, np.ndarray]]:
        """``(t, fields)``: all six components at the integer nodes at ``t = (n - 1/2) dt``, the time of the stored
        H, with E averaged over its two neighbouring steps and every component averaged over its half-cell
        offsets (second order in space and time). It is the input layout of
        :func:`pinneapple_physics.maxwell_checks.maxwell_checks` (H components)."""
        prev = getattr(self, "_E_prev", None)
        out = {}
        for n in E_NAMES + H_NAMES:
            if n.startswith("E"):
                F = self.E[n] if prev is None else 0.5 * (self.E[n] + prev[n])
            else:
                F = self.H[n]
            out[n] = self._to_nodes(F, n)
        return (self.n - 0.5) * self.dt, out

    def _to_nodes(self, F: np.ndarray, component: str) -> np.ndarray:
        """Average a staggered component onto the integer nodes; on a non-periodic wall the node value is the
        linear extrapolation of the two nearest interior samples (the sample half a cell outside the last node is
        not part of the domain)."""
        for ax, o in enumerate(_offsets(component)):
            n = self.g.shape[ax]
            if not o or n == 1:
                continue
            if self.bc[AXES[ax]][0] in ("periodic", "bloch"):
                F = 0.5 * (F + self._shift(F, ax, False, self.bc[AXES[ax]]))
                continue
            take = lambda i, F=F, ax=ax: np.take(F, [i], axis=ax)  # noqa: E731
            mid = 0.5 * (np.take(F, np.arange(1, n - 1), axis=ax) + np.take(F, np.arange(0, n - 2), axis=ax))
            lo = 1.5 * take(0) - 0.5 * take(1)
            hi = 1.5 * take(n - 2) - 0.5 * take(n - 3)
            F = np.concatenate([lo, mid, hi], axis=ax)
        return F

    def field_at_nodes(self, component: str) -> np.ndarray:
        """A component at the integer nodes (space only: E at ``n dt``, H at ``(n - 1/2) dt``), e.g. for plots."""
        src = self.E if component.startswith("E") else self.H
        return self._to_nodes(src[component], component)


def poynting_flux_dft(mon: DFTMonitor, axis: str, spacing: Sequence[float]) -> np.ndarray:
    """Time-averaged power ``(1/2) Re int (E x H*) . n dA`` through a plane monitor normal to ``axis`` from its DFT
    phasors (all four transverse components must be monitored); per unit length along invariant axes."""
    a = AXES.index(axis)
    t1, t2 = (a + 1) % 3, (a + 2) % 3
    E1, E2 = mon.phasor(E_NAMES[t1]), mon.phasor(E_NAMES[t2])
    H1, H2 = mon.phasor(H_NAMES[t1]), mon.phasor(H_NAMES[t2])
    S = 0.5 * np.real(E1 * np.conj(H2) - E2 * np.conj(H1))
    dA = 1.0
    for ax in (t1, t2):
        if S.shape[1 + ax] > 1:
            dA *= spacing[ax]
    return S.reshape(S.shape[0], -1).sum(1) * dA


__all__.append("poynting_flux_dft")


class NearToFarField:
    """Near-to-far-field transform on a closed box of node planes (3-D grids).

    The DFT of the tangential fields on the six faces gives the equivalent surface currents ``J = n x H`` and
    ``M = -n x E``; their radiation integrals ``N = oint J exp(j k r_hat . r') dS'`` and ``L = oint M ...`` give the
    far field (``exp(+j omega t)``, C. A. Balanis, *Advanced Engineering Electromagnetics*, 2nd ed., eqs. 6-122;
    TH ch. 8)::

        r E_theta exp(j k r) = -j k / (4 pi) (L_phi + eta N_theta)
        r E_phi   exp(j k r) =  j k / (4 pi) (L_theta - eta N_phi)

    ``box = ((i0, i1), (j0, j1), (k0, k1))`` are the node indices of the faces; the box must enclose every source
    and scatterer and lie in vacuum outside the PML. Create it before running; call :meth:`far_field` after.
    """

    def __init__(self, sim: FDTD, freqs, box):
        if not all(sim.g.active):
            raise ValueError("the near-to-far-field box needs a 3-D grid")
        self.sim = sim
        self.freqs = np.atleast_1d(np.asarray(freqs, float))
        self.box = tuple(tuple(int(v) for v in b) for b in box)
        self.faces = []
        for a in range(3):
            for side, p in ((-1.0, self.box[a][0]), (1.0, self.box[a][1])):
                region = []
                for ax in range(3):
                    if ax == a:
                        region.append(slice(p - 1, p + 1))
                    else:
                        lo, hi = self.box[ax]
                        region.append(slice(lo - 1, hi + 1))
                t1, t2 = (a + 1) % 3, (a + 2) % 3
                comps = [E_NAMES[t1], E_NAMES[t2], H_NAMES[t1], H_NAMES[t2]]
                mon = sim.add_dft(comps, self.freqs, tuple(region))
                self.faces.append((a, side, p, mon))

    def _face_fields(self, a, mon, comp):
        """Phasors of one component at the face's nodes (index 1 of the slab along ``a``; transverse nodes
        lo..hi), averaged over the component's half-cell offsets."""
        F = mon.phasor(comp)
        off = _offsets(comp)
        for ax in range(3):
            axf = ax + 1
            n = F.shape[axf]
            if off[ax]:
                F = 0.5 * (np.take(F, np.arange(1, n), axis=axf) + np.take(F, np.arange(0, n - 1), axis=axf))
            else:
                F = np.take(F, np.arange(1, n), axis=axf) if ax != a else np.take(F, [1], axis=axf)
        # along a: after averaging (offset) or taking index 1 (no offset) the slab has one plane
        F = np.take(F, [0], axis=a + 1) if F.shape[a + 1] > 1 else F
        return F

    def far_field(self, theta, phi):
        """``(E_theta, E_phi)`` far-field patterns ``r E exp(j k r)`` (V), shape ``(n_freq, *theta.shape)``."""
        th = np.asarray(theta, float)
        ph = np.asarray(phi, float)
        rhat = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
        that = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
        phat = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
        g = self.sim.g
        k = 2 * np.pi * self.freqs / C0
        Nv = np.zeros((len(k),) + th.shape + (3,), complex)
        Lv = np.zeros_like(Nv)
        for a, side, p, mon in self.faces:
            t1, t2 = (a + 1) % 3, (a + 2) % 3
            E1 = self._face_fields(a, mon, E_NAMES[t1])
            E2 = self._face_fields(a, mon, E_NAMES[t2])
            H1 = self._face_fields(a, mon, H_NAMES[t1])
            H2 = self._face_fields(a, mon, H_NAMES[t2])
            # n = side * e_a: n x H = side (H1 e_t2 - H2 e_t1), -n x E = -side (E1 e_t2 - E2 e_t1)
            J = np.zeros(E1.shape + (3,), complex)
            M = np.zeros_like(J)
            J[..., t2], J[..., t1] = side * H1, -side * H2
            M[..., t2], M[..., t1] = -side * E1, side * E2
            coords = []
            for ax in range(3):
                if ax == a:
                    coords.append(np.array([p * g.spacing[ax]]))
                else:
                    lo, hi = self.box[ax]
                    coords.append(np.arange(lo, hi + 1) * g.spacing[ax])
            X, Y, Z = np.meshgrid(*coords, indexing="ij")
            w = np.ones(X.shape)
            for ax in range(3):
                if ax == a:
                    continue
                d = g.spacing[ax]
                ww = np.full(len(coords[ax]), d)
                ww[0] = ww[-1] = d / 2
                sh = [1, 1, 1]
                sh[ax] = len(ww)
                w = w * ww.reshape(sh)
            P = np.stack([X, Y, Z], -1).reshape(-1, 3)
            wf = w.reshape(-1)
            phase = np.exp(1j * np.einsum("f,...c,pc->f...p", k, rhat, P))   # (n_freq, *theta, n_points)
            Jf = J.reshape(J.shape[0], -1, 3)
            Mf = M.reshape(M.shape[0], -1, 3)
            Nv += np.einsum("f...p,fpc,p->f...c", phase, Jf, wf)
            Lv += np.einsum("f...p,fpc,p->f...c", phase, Mf, wf)
        Nth = np.einsum("f...c,...c->f...", Nv, that)
        Nph = np.einsum("f...c,...c->f...", Nv, phat)
        Lth = np.einsum("f...c,...c->f...", Lv, that)
        Lph = np.einsum("f...c,...c->f...", Lv, phat)
        kk = k.reshape((-1,) + (1,) * th.ndim)
        Eth = -1j * kk / (4 * np.pi) * (Lph + ETA0 * Nth)
        Eph = 1j * kk / (4 * np.pi) * (Lth - ETA0 * Nph)
        return Eth, Eph

    def radiated_power(self, n_theta: int = 91, n_phi: int = 72) -> np.ndarray:
        """``(1 / (2 eta)) int (|E_theta|^2 + |E_phi|^2) dOmega`` of the far field, per frequency (W)."""
        th = np.linspace(0, np.pi, n_theta)
        ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        Et, Ep = self.far_field(TH, PH)
        U = (np.abs(Et) ** 2 + np.abs(Ep) ** 2) / (2 * ETA0)
        integrand = U * np.sin(TH)[None]
        _tr = getattr(np, "trapezoid", None) or np.trapz
        return _tr(integrand.mean(-1) * 2 * np.pi, th, axis=-1)


__all__.append("NearToFarField")
