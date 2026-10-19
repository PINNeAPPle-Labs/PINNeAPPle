"""Physics checks of electromagnetic fields from any source (a PINN, an FNO, a graph network, a solver, measured
data), without a reference solution: every check asks whether the fields obey Maxwell's equations and the
balance laws that follow from them.

Fields are components on a tensor grid. ``grid`` maps the axes present to 1-D coordinate arrays, in the order
``x, y, z, t`` (``indexing="ij"``, the layout ``pinneapple check`` reads); a field array has one dimension per axis
present. A direction that is absent is a direction of invariance (``d/dz = 0`` for a 2-D field). ``fields`` maps
component names to arrays: ``Ex, Ey, Ez``, ``Bx, ...`` (or ``Hx, ...``), ``Dx, ...``, ``Jx, ...`` and ``rho``.
Missing components are zero; ``B = mu H`` and ``D = eps E`` fill in whichever of the pair is missing. With
``omega`` the fields are complex phasors (``exp(+j omega t)``, no ``t`` axis) and ``d/dt`` is ``j omega``.

Residual checks (each relative: ``||sum of terms|| / sum of ||term||``, over interior points; 0 for an exact
field, 1 when the terms do not cancel at all; derivatives are second-order central differences):

``gauss_electric``      div D - rho
``gauss_magnetic``      div B
``faraday``             curl E + dB/dt
``ampere_maxwell``      curl H - J - dD/dt
``continuity``          d rho/dt + div J

A smooth field that obeys the law leaves a residual of order ``(k h)^2`` from the differences alone, so each
residual is also computed on the grid with every other point (``2h``): the ratio ``r(2h) / r(h)`` is about 4 when
the residual is discretisation error (the field is consistent with the law) and about 1 when the field itself
violates it. Both are reported; the extrapolated residual ``max(0, (4 r(h) - r(2h)) / 3)`` estimates the part
that does not vanish with ``h``.

Balance and network checks:

``poynting_balance``    d/dt int u dV + oint S.n dA + int J.E dV = 0 over the grid box (time domain), or
                        Re oint S.n dA + P_loss = P_source (phasors), relative to the largest term
``tangential_continuity`` n x (E1 - E2) = 0 (and n x (H1 - H2) = K) at an interface
``pec_boundary``        n x E = 0 on faces declared perfect electric conductors
``reciprocity``         ||S - S^T|| / ||S|| of scattering matrices
``passivity``           largest singular value of S <= 1 at every frequency (lossless: S^H S = I)

``maxwell_checks`` runs the checks the given fields allow and returns :class:`MaxwellCheck` objects;
``maxwell_trust_report`` turns them into a deploy decision; ``sample_checks`` is what ``pinneapple check`` runs
for ``physics = "maxwell"``. numpy only (the checker loads this module without torch).

References: Jackson, *Classical Electrodynamics*, 3rd ed., sec. 6.7 (Poynting's theorem) and 6.1 (the
equations); Pozar, *Microwave Engineering*, 4th ed., sec. 4.3 (reciprocal and lossless networks).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "EPS0", "MU0", "DEFAULT_TOLERANCES", "AXES", "MaxwellCheck", "FieldSet", "relative_residual",
    "gauss_electric", "gauss_magnetic", "faraday", "ampere_maxwell", "continuity", "poynting_balance",
    "tangential_continuity", "pec_boundary", "reciprocity", "passivity", "maxwell_checks", "maxwell_trust_report",
    "sample_checks", "sample_verdict", "save_maxwell_npz", "check_config_entry",
]

#: CODATA 2022 (same values as ``pinneapple_simulation.electromagnetics.constants``; a test keeps them equal)
MU0 = 1.25663706127e-6
EPS0 = 1.0 / (MU0 * 299_792_458.0 ** 2)

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz

#: SI unit of each component letter (as ``pinneapple_data.physical_units`` spells it)
SI_UNITS = {"E": "V/m", "B": "T", "H": "A/m", "D": "coulomb/m^2", "J": "A/m^2", "rho": "coulomb/m^3"}

AXES = ("x", "y", "z", "t")
SPACE = ("x", "y", "z")

#: relative residuals: a field resolved with ~20 points per wavelength leaves about (2 pi / 20)^2 / 6 = 1.6 % from
#: second-order differences alone; 5 % flags fields that violate the law by more than the differencing does.
DEFAULT_TOLERANCES: dict[str, float] = {
    "gauss_electric": 5e-2,
    "gauss_magnetic": 5e-2,
    "faraday": 5e-2,
    "ampere_maxwell": 5e-2,
    "continuity": 5e-2,
    "poynting_balance": 5e-2,
    "tangential_continuity": 1e-2,
    "pec_boundary": 1e-2,
    "reciprocity": 1e-6,
    "passivity": 1e-6,
    "reference": 5e-2,
}


@dataclass
class MaxwellCheck:
    name: str
    key: str
    value: float
    limit: float | None
    passed: bool | None
    critical: bool = True
    detail: str = ""
    kind: str = "physics"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "key": self.key, "value": self.value, "limit": self.limit,
                "passed": self.passed, "critical": self.critical, "detail": self.detail, "kind": self.kind}


# --------------------------------------------------------------------------------------------- field access
class FieldSet:
    """Fields on a tensor grid with derivatives; see the module docstring for the layout."""

    def __init__(self, grid: Mapping[str, Sequence[float]], fields: Mapping[str, Any], *, eps=EPS0, mu=MU0,
                 sigma=0.0, omega: float | None = None, interior: int = 1):
        self.axes = [a for a in AXES if a in grid]
        if omega is not None and "t" in self.axes:
            raise ValueError("phasor fields (omega given) have no t axis")
        self.coords = {a: np.asarray(grid[a], float) for a in self.axes}
        self.shape = tuple(len(self.coords[a]) for a in self.axes)
        self.omega = omega
        self.interior = interior
        self.raw = {k: np.asarray(v) for k, v in fields.items()}
        for k, v in self.raw.items():
            if v.shape != self.shape:
                raise ValueError(f"field {k} has shape {v.shape}, the grid {self.shape}")
        self.eps = self._material(eps)
        self.mu = self._material(mu)
        self.sigma = self._material(sigma)
        self.scale = 0.0   # sum of |f| / h over the differentiated fields: the size a derivative term could have

    def _material(self, m):
        m = np.asarray(m)
        if m.ndim == 0:
            return m
        n_space = sum(a in SPACE for a in self.axes)
        if m.ndim == n_space and "t" in self.axes:
            return m[..., None]
        return m

    def has(self, letter: str) -> bool:
        return any(f"{letter}{c}" in self.raw for c in "xyz")

    def vec(self, letter: str) -> list[np.ndarray]:
        """Components of E, B, H, D or J (with the constitutive fill-ins), zeros where absent."""
        if letter == "B" and not self.has("B") and self.has("H"):
            return [self.mu * h for h in self.vec("H")]
        if letter == "H" and not self.has("H") and self.has("B"):
            return [b / self.mu for b in self.vec("B")]
        if letter == "D" and not self.has("D") and self.has("E"):
            return [self.eps * e for e in self.vec("E")]
        if letter == "J" and not self.has("J") and np.any(self.sigma) and self.has("E"):
            return [self.sigma * e for e in self.vec("E")]
        dtype = complex if any(np.iscomplexobj(v) for v in self.raw.values()) else float
        return [self.raw.get(f"{letter}{c}", np.zeros(self.shape, dtype)) for c in "xyz"]

    def scalar(self, name: str) -> np.ndarray:
        dtype = complex if any(np.iscomplexobj(v) for v in self.raw.values()) else float
        return self.raw.get(name, np.zeros(self.shape, dtype))

    def d(self, f: np.ndarray, axis: str) -> np.ndarray:
        """Derivative along ``axis``: central differences (second order, one-sided second order at the ends);
        zero along an absent spatial axis; ``j omega f`` for ``t`` with phasors."""
        if axis == "t" and self.omega is not None:
            self.scale += abs(self.omega) * float(np.linalg.norm(f))
            return 1j * self.omega * f
        if axis not in self.axes:
            return np.zeros_like(f)
        f = np.broadcast_to(f, self.shape)
        k = self.axes.index(axis)
        if self.shape[k] < 3:
            raise ValueError(f"axis {axis} needs at least 3 points")
        self.scale += float(np.linalg.norm(f)) / float(np.min(np.diff(self.coords[axis])))
        return np.gradient(f, self.coords[axis], axis=k, edge_order=2)

    def mask(self) -> tuple[slice, ...]:
        n = self.interior
        return tuple(slice(n, s - n) if (s > 2 * n + 2 and a in SPACE) else slice(None)
                     for a, s in zip(self.axes, self.shape, strict=False))

    def coarsened(self) -> FieldSet | None:
        """The same fields on every other point of every axis with at least 7 points (or None)."""
        take = tuple(slice(None, None, 2) if s >= 7 else slice(None) for s in self.shape)
        if all(t == slice(None) for t in take):
            return None
        sub = lambda m: m[take] if np.ndim(m) == len(self.shape) else m  # noqa: E731
        grid = {a: self.coords[a][t] for a, t in zip(self.axes, take, strict=False)}
        fs = FieldSet.__new__(FieldSet)
        fs.axes, fs.coords = self.axes, grid
        fs.shape = tuple(len(grid[a]) for a in self.axes)
        fs.omega, fs.interior = self.omega, self.interior
        fs.raw = {k: v[take] for k, v in self.raw.items()}
        fs.scale = 0.0
        fs.eps, fs.mu, fs.sigma = (sub(np.broadcast_to(m, self.shape)) if np.ndim(m) else m
                                   for m in (self.eps, self.mu, self.sigma))
        return fs


def relative_residual(terms: Sequence[np.ndarray], mask=None, floor: float = 0.0) -> float:
    """``||sum terms|| / sum ||term||`` (L2 over the masked points); NaN when every term is zero or below ``floor``
    (the law does not constrain the field: e.g. div D of a field that varies only along D's normal direction)."""
    terms = [np.asarray(t) if mask is None else np.asarray(t)[mask] for t in terms]
    num = np.linalg.norm(np.sum(terms, axis=0))
    den = sum(np.linalg.norm(t) for t in terms)
    return float(num / den) if den > max(floor, 0.0) else float("nan")


def _curl(fs: FieldSet, v):
    x, y, z = v
    return [fs.d(z, "y") - fs.d(y, "z"), fs.d(x, "z") - fs.d(z, "x"), fs.d(y, "x") - fs.d(x, "y")]


def _div_terms(fs: FieldSet, v):
    return [fs.d(c, a) for c, a in zip(v, "xyz", strict=False)]


def _vector_residual(term_lists, mask, floor=0.0):
    """Relative residual of a vector law: components stacked."""
    stacked = [np.stack([t[c] for c in range(3)]) for t in term_lists]
    m = (slice(None),) + mask
    return relative_residual(stacked, m, floor)


# --- the five laws as term lists (a law holds when the terms sum to zero)
def _gauss_e(fs):
    return _div_terms(fs, fs.vec("D")) + [-fs.scalar("rho")]


def _gauss_b(fs):
    return _div_terms(fs, fs.vec("B"))


def _faraday(fs):
    return [_curl(fs, fs.vec("E")), [fs.d(b, "t") for b in fs.vec("B")]]


def _ampere(fs):
    return [_curl(fs, fs.vec("H")), [-j for j in fs.vec("J")], [-fs.d(d, "t") for d in fs.vec("D")]]


def _continuity(fs):
    return [fs.d(fs.scalar("rho"), "t")] + _div_terms(fs, fs.vec("J"))


_LAWS = {
    "gauss_electric": (_gauss_e, False, "div D - rho"),
    "gauss_magnetic": (_gauss_b, False, "div B"),
    "faraday": (_faraday, True, "curl E + dB/dt"),
    "ampere_maxwell": (_ampere, True, "curl H - J - dD/dt"),
    "continuity": (_continuity, False, "d rho/dt + div J"),
}


def _law_value(fs: FieldSet, key: str) -> float:
    fn, vector, _ = _LAWS[key]
    fs.scale = 0.0
    terms = fn(fs)
    floor = 1e-9 * fs.scale
    return _vector_residual(terms, fs.mask(), floor) if vector else relative_residual(terms, fs.mask(), floor)


def _law(key: str, fs: FieldSet, tol: float, name: str) -> MaxwellCheck:
    r = _law_value(fs, key)
    if np.isnan(r):
        return MaxwellCheck(name, key, r, tol, None, True, f"not informative: every term of {_LAWS[key][2]} is zero "
                            "for these components")
    detail = f"{_LAWS[key][2]}: ||sum of terms|| / sum of ||term|| = {r:.3g}"
    co = fs.coarsened()
    if co is not None and r > 0:
        try:
            r2 = _law_value(co, key)
            if np.isnan(r2):
                raise ValueError("coarse grid")
            ratio = r2 / r
            intrinsic = max(0.0, (4.0 * r - r2) / 3.0)
            verdict = ("discretisation error (the field obeys the law)" if ratio > 2.5 else
                       "a property of the field (it does not shrink with h)" if ratio < 1.5 else "mixed")
            detail += f"; r(2h)/r(h) = {ratio:.2f}: {verdict}; extrapolated residual {intrinsic:.3g}"
        except ValueError:
            pass
    return MaxwellCheck(name, key, r, tol, bool(np.isfinite(r) and r <= tol), True, detail)


def gauss_electric(grid, fields, *, eps=EPS0, tol: float | None = None, **kw) -> MaxwellCheck:
    fs = FieldSet(grid, fields, eps=eps, **kw)
    return _law("gauss_electric", fs, tol or DEFAULT_TOLERANCES["gauss_electric"], "Gauss's law div D = rho")


def gauss_magnetic(grid, fields, *, mu=MU0, tol: float | None = None, **kw) -> MaxwellCheck:
    fs = FieldSet(grid, fields, mu=mu, **kw)
    return _law("gauss_magnetic", fs, tol or DEFAULT_TOLERANCES["gauss_magnetic"], "Gauss's law for B, div B = 0")


def faraday(grid, fields, *, mu=MU0, omega=None, tol: float | None = None, **kw) -> MaxwellCheck:
    fs = FieldSet(grid, fields, mu=mu, omega=omega, **kw)
    return _law("faraday", fs, tol or DEFAULT_TOLERANCES["faraday"], "Faraday's law curl E = -dB/dt")


def ampere_maxwell(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega=None, tol: float | None = None,
                   **kw) -> MaxwellCheck:
    fs = FieldSet(grid, fields, eps=eps, mu=mu, sigma=sigma, omega=omega, **kw)
    return _law("ampere_maxwell", fs, tol or DEFAULT_TOLERANCES["ampere_maxwell"],
                "Ampere-Maxwell law curl H = J + dD/dt")


def continuity(grid, fields, *, omega=None, tol: float | None = None, **kw) -> MaxwellCheck:
    fs = FieldSet(grid, fields, omega=omega, **kw)
    return _law("continuity", fs, tol or DEFAULT_TOLERANCES["continuity"], "charge continuity d rho/dt + div J = 0")


# --------------------------------------------------------------------------------------------- energy balance
def _integrate(f: np.ndarray, fs: FieldSet, axes: Sequence[str]) -> np.ndarray:
    """Trapezoidal integral of ``f`` over the given spatial axes (they keep their position order)."""
    out = f
    for a in sorted(axes, key=lambda a: -fs.axes.index(a)):
        out = _trapz(out, fs.coords[a], axis=fs.axes.index(a))
    return out


def _face_flux(fs: FieldSet, S: list[np.ndarray]) -> np.ndarray:
    """Net outward flux of a vector field through the faces of the grid box (spatial axes present), per unit
    length along the absent ones; keeps the t axis if present."""
    space = [a for a in fs.axes if a in SPACE]
    total = 0.0
    for a in space:
        k = fs.axes.index(a)
        comp = S["xyz".index(a)]
        others = [b for b in space if b != a]
        for idx, sign in ((-1, 1.0), (0, -1.0)):
            face = np.take(comp, idx, axis=k)
            sub = FieldSet.__new__(FieldSet)
            sub.axes = [b for b in fs.axes if b != a]
            sub.coords = {b: fs.coords[b] for b in sub.axes}
            total = total + sign * _integrate(face, sub, others)
    return np.asarray(total)


def poynting_balance(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega: float | None = None,
                     tol: float | None = None) -> MaxwellCheck:
    """Poynting's theorem over the grid box.

    Time domain: ``U(t1) - U(t0) + int_t0^t1 [oint S.n dA + int J.E dV] dt = 0`` with ``U = int (E.D + B.H)/2 dV``
    (linear, non-dispersive media; ``J`` includes ``sigma E``), relative to ``max(|U|) + int |flux| dt``.
    Phasors: ``Re oint (E x H*)/2 . n dA + (1/2) int Re(J* . E) dV = 0`` with ``J`` the total current (impressed
    and conduction; dielectric loss through a complex ``eps`` is added as ``(omega/2) int -Im(eps) |E|^2``),
    relative to the sum of the absolute values of the terms."""
    tol = tol or DEFAULT_TOLERANCES["poynting_balance"]
    fs = FieldSet(grid, fields, eps=eps, mu=mu, sigma=sigma, omega=omega, interior=0)
    space = [a for a in fs.axes if a in SPACE]
    E, H = fs.vec("E"), fs.vec("H")
    J = fs.vec("J")
    if np.any(fs.sigma) and fs.has("J"):
        J = [j + fs.sigma * e for j, e in zip(J, E, strict=False)]
    if omega is None:
        if "t" not in fs.axes:
            return MaxwellCheck("Poynting energy balance", "poynting_balance", float("nan"), tol, None, True,
                                "not run: needs a t axis or omega")
        D, B = fs.vec("D"), fs.vec("B")
        u = 0.5 * sum((e * d + b * h) for e, d, b, h in zip(E, D, B, H, strict=False))
        S = [E[1] * H[2] - E[2] * H[1], E[2] * H[0] - E[0] * H[2], E[0] * H[1] - E[1] * H[0]]
        U = _integrate(u, fs, space)
        flux = _face_flux(fs, S)
        loss = _integrate(sum(j * e for j, e in zip(J, E, strict=False)), fs, space)
        t = fs.coords["t"]
        out = _trapz(flux + loss, t)
        res = (U[-1] - U[0]) + out
        scale = np.max(np.abs(U)) + _trapz(np.abs(flux), t) + _trapz(np.abs(loss), t)
        detail = (f"U(t1) - U(t0) = {U[-1] - U[0]:.4g}, outflow {_trapz(flux, t):.4g}, dissipated "
                  f"{_trapz(loss, t):.4g} (J per unit length along absent axes)")
    else:
        S = [0.5 * (E[1] * H[2].conj() - E[2] * H[1].conj()), 0.5 * (E[2] * H[0].conj() - E[0] * H[2].conj()),
             0.5 * (E[0] * H[1].conj() - E[1] * H[0].conj())]
        flux = float(np.real(_face_flux(fs, S)))
        p = 0.5 * np.real(sum(j.conj() * e for j, e in zip(J, E, strict=False)))
        if np.iscomplexobj(fs.eps):
            p = p + 0.5 * omega * (-np.imag(fs.eps)) * sum(np.abs(e) ** 2 for e in E)
        loss = float(_integrate(p, fs, space))
        res = flux + loss
        scale = abs(flux) + float(_integrate(np.abs(p), fs, space))
        detail = f"net outflow {flux:.4g} W, J.E (source and loss) {loss:.4g} W (per unit length along absent axes)"
    r = float(abs(res) / scale) if scale > 0 else 0.0
    return MaxwellCheck("Poynting energy balance", "poynting_balance", r, tol, bool(np.isfinite(r) and r <= tol),
                        True, detail)


# --------------------------------------------------------------------------------------------- interfaces
def tangential_continuity(F1, F2, normal, *, surface_current=None, tol: float | None = None,
                          name: str = "E") -> MaxwellCheck:
    """``n x (F1 - F2) = K`` at interface points: ``F1``, ``F2`` are (n, 3) fields on the two sides at the same
    points, ``normal`` a unit normal (3,) or (n, 3) pointing from side 2 to side 1; ``K`` the surface current
    (for H; zero for E). Value: RMS mismatch over RMS of the tangential fields."""
    tol = tol or DEFAULT_TOLERANCES["tangential_continuity"]
    F1, F2 = np.asarray(F1), np.asarray(F2)
    n = np.broadcast_to(np.asarray(normal, float), F1.shape)
    jump = np.cross(n, F1 - F2)
    if surface_current is not None:
        jump = jump - np.asarray(surface_current)
    scale = np.sqrt(np.mean(np.abs(np.cross(n, F1)) ** 2 + np.abs(np.cross(n, F2)) ** 2))
    r = float(np.sqrt(np.mean(np.abs(jump) ** 2)) / scale) if scale > 0 else 0.0
    return MaxwellCheck(f"tangential continuity of {name}", "tangential_continuity", r, tol, bool(r <= tol), True,
                        f"RMS |n x ({name}1 - {name}2){' - K' if surface_current is not None else ''}| over RMS "
                        f"tangential {name}")


def pec_boundary(grid, fields, faces: Sequence[str], *, tol: float | None = None) -> MaxwellCheck:
    """``n x E = 0`` on grid faces declared PEC (``"x_min"``, ``"y_max"``, ...): RMS tangential E on those
    faces over RMS |E| in the whole field."""
    tol = tol or DEFAULT_TOLERANCES["pec_boundary"]
    fs = FieldSet(grid, fields)
    E = fs.vec("E")
    scale = np.sqrt(np.mean(sum(np.abs(e) ** 2 for e in E)))
    vals = []
    for face in faces:
        a, side = face.split("_")
        k = fs.axes.index(a)
        idx = 0 if side == "min" else -1
        tang = [np.take(E["xyz".index(c)], idx, axis=k) for c in "xyz" if c != a]
        vals.append(np.mean(sum(np.abs(t) ** 2 for t in tang)))
    r = float(np.sqrt(np.mean(vals)) / scale) if scale > 0 and vals else 0.0
    return MaxwellCheck("PEC boundary n x E = 0", "pec_boundary", r, tol, bool(r <= tol), True,
                        f"RMS tangential E on {', '.join(faces)} over RMS |E|")


# --------------------------------------------------------------------------------------------- networks
def reciprocity(S, *, tol: float | None = None) -> MaxwellCheck:
    """``S = S^T`` for a reciprocal network (``S`` of shape (n, n) or (f, n, n)): max over frequencies of
    ``||S - S^T||_F / ||S||_F``."""
    tol = tol or DEFAULT_TOLERANCES["reciprocity"]
    S = np.asarray(S, complex)
    S = S[None] if S.ndim == 2 else S
    num = np.linalg.norm(S - np.swapaxes(S, -1, -2), axis=(-2, -1))
    den = np.maximum(np.linalg.norm(S, axis=(-2, -1)), 1e-300)
    r = float(np.max(num / den))
    return MaxwellCheck("reciprocity S = S^T", "reciprocity", r, tol, bool(r <= tol), True,
                        "max over frequencies of ||S - S^T|| / ||S|| (reciprocal media: symmetric eps, mu, sigma)")


def passivity(S, *, lossless: bool = False, tol: float | None = None) -> MaxwellCheck:
    """Passivity: ``I - S^H S`` positive semi-definite, i.e. the largest singular value of ``S`` is at most 1, at
    every frequency; value = ``max(sigma_max - 1, 0)``. ``lossless=True`` checks unitarity ``||S^H S - I||``."""
    tol = tol or DEFAULT_TOLERANCES["passivity"]
    S = np.asarray(S, complex)
    S = S[None] if S.ndim == 2 else S
    if lossless:
        eye = np.eye(S.shape[-1])
        r = float(np.max(np.linalg.norm(np.conj(np.swapaxes(S, -1, -2)) @ S - eye, axis=(-2, -1))))
        return MaxwellCheck("lossless S^H S = I", "passivity", r, tol, bool(r <= tol), True,
                            "max over frequencies of ||S^H S - I||_F")
    smax = np.linalg.svd(S, compute_uv=False)[..., 0]
    r = float(max(np.max(smax) - 1.0, 0.0))
    worst = int(np.argmax(smax))
    return MaxwellCheck("passivity (largest singular value of S <= 1)", "passivity", r, tol, bool(r <= tol), True,
                        f"max singular value {np.max(smax):.6g} at frequency index {worst}")


# --------------------------------------------------------------------------------------------- all together
def maxwell_checks(grid, fields, *, eps=EPS0, mu=MU0, sigma=0.0, omega: float | None = None,
                   laws: Sequence[str] | None = None, pec_faces: Sequence[str] = (), reference=None,
                   tolerances: Mapping[str, float] | None = None, energy: bool = True) -> list[MaxwellCheck]:
    """Every check the given fields allow: the laws whose terms are present (``laws`` restricts them), the
    Poynting balance (needs E and H or B, and a t axis or ``omega``), PEC faces, and a relative L2 comparison
    with ``reference`` (a mapping of the same component names; evaluation only)."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    fs = FieldSet(grid, fields, eps=eps, mu=mu, sigma=sigma, omega=omega)
    for k, v in fs.raw.items():
        if not np.all(np.isfinite(v)):
            return [MaxwellCheck("finite values", "finite", float("inf"), 0.0, False, True, f"{k} has NaN or inf")]
    has_t = "t" in fs.axes or omega is not None
    E, B = fs.has("E") or fs.has("D"), fs.has("B") or fs.has("H")
    applicable = {
        "gauss_electric": E,
        "gauss_magnetic": B,
        "faraday": E and B and has_t,
        "ampere_maxwell": E and B and has_t,
        "continuity": fs.has("J") and "rho" in fs.raw and has_t,
    }
    names = {"gauss_electric": "Gauss's law div D = rho", "gauss_magnetic": "Gauss's law for B, div B = 0",
             "faraday": "Faraday's law curl E = -dB/dt", "ampere_maxwell": "Ampere-Maxwell law curl H = J + dD/dt",
             "continuity": "charge continuity d rho/dt + div J = 0"}
    out = []
    for key, ok in applicable.items():
        if laws is not None and key not in laws:
            continue
        if ok:
            out.append(_law(key, fs, tol[key], names[key]))
    if energy and E and B and has_t and (laws is None or "poynting_balance" in laws):
        out.append(poynting_balance(grid, fields, eps=eps, mu=mu, sigma=sigma, omega=omega,
                                    tol=tol["poynting_balance"]))
    if pec_faces:
        out.append(pec_boundary(grid, fields, pec_faces, tol=tol["pec_boundary"]))
    if reference is not None:
        keys = [k for k in fields if k in reference]
        num = np.sqrt(sum(np.sum(np.abs(np.asarray(fields[k]) - np.asarray(reference[k])) ** 2) for k in keys))
        den = np.sqrt(sum(np.sum(np.abs(np.asarray(reference[k])) ** 2) for k in keys))
        r = float(num / den) if den > 0 else float("inf")
        out.append(MaxwellCheck("reference (relative L2)", "reference", r, tol["reference"], bool(r <= tol["reference"]),
                                True, f"components {', '.join(keys)}", kind="reference"))
    return out


def maxwell_trust_report(grid, fields, *, model: str = "", **kwargs):
    """``(TrustReport, checks)``: the deploy decision (APPROVED / REVIEW / REJECT) of :func:`maxwell_checks`."""
    from pinneapple_analysis.trust.trust_report import Check, TrustReport

    checks = maxwell_checks(grid, fields, **kwargs)
    out = [Check(c.name, "not_run" if c.passed is None else ("supports" if c.passed else "contradicts"),
                 critical=c.critical, value=c.value if np.isfinite(c.value) else None, limit=c.limit, source=c.kind,
                 reason=c.detail) for c in checks]
    return TrustReport(out, model=model), checks


def sample_checks(grid, fields, **kwargs) -> list[dict[str, Any]]:
    """The checks as ``pinneapple check`` runs them (``physics = "maxwell"``): ``{"key", "name", "value", "limit",
    "passed", "detail"}`` per check."""
    return [c.to_dict() for c in maxwell_checks(grid, fields, **kwargs)]


def sample_verdict(checks: Sequence[Mapping[str, Any]]) -> str:
    return "PASS" if all(c["passed"] is not False for c in checks) else "FAIL"


def save_maxwell_npz(path, grid: Mapping[str, Sequence[float]], fields: Mapping[str, np.ndarray], **extra):
    """Write a grid result ``pinneapple check`` reads with ``physics = "maxwell"``: 1-D axes and component arrays.
    Complex phasors are stored as ``<name>_re`` / ``<name>_im`` pairs."""
    from pathlib import Path

    arrays: dict[str, np.ndarray] = {a: np.asarray(v, float) for a, v in grid.items()}
    units = [f"{a}={'s' if a == 't' else 'm'}" for a in grid]
    for k, v in fields.items():
        v = np.asarray(v)
        names = [f"{k}_re", f"{k}_im"] if np.iscomplexobj(v) else [k]
        if np.iscomplexobj(v):
            arrays[names[0]], arrays[names[1]] = v.real, v.imag
        else:
            arrays[k] = v
        if k in SI_UNITS or k[:1] in SI_UNITS:
            units += [f"{n}={SI_UNITS.get(k, SI_UNITS.get(k[:1]))}" for n in names]
    arrays["units"] = np.array(units)
    arrays.update({k: np.asarray(v) for k, v in extra.items()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **arrays)
    return p


def check_config_entry(name: str, path: str, *, parameters: Mapping[str, Any] | None = None,
                       tolerances: Mapping[str, float] | None = None) -> str:
    """A ``[[result]]`` block of ``pinneapple-check.toml`` for a field saved by :func:`save_maxwell_npz`."""
    lines = ["[[result]]", f'name = "{name}"', f'path = "{path}"', 'physics = "maxwell"']
    if parameters:
        lines.append("parameters = { " + ", ".join(f"{k} = {v!r}" if not isinstance(v, str) else f'{k} = "{v}"'
                                                   for k, v in parameters.items()) + " }")
    if tolerances:
        lines.append("tolerances = { " + ", ".join(f'"maxwell-{k}" = {v!r}' for k, v in tolerances.items()) + " }")
    return "\n".join(lines) + "\n"
