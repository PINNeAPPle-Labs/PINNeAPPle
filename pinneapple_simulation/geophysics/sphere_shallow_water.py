"""Shallow-water equations on the rotating sphere: spectral-transform solver and the Williamson et al. test suite.

Model: vorticity-divergence form on a Gaussian grid with triangular truncation T_N (the spectral-transform method of
the NCAR shallow-water model, Hack & Jakob 1992, NCAR/TN-343+STR):

    dzeta/dt = -div(eta V)                         eta = zeta + f
    ddelta/dt = k.curl(eta V) - lap(Phi + Phi_s + |V|^2/2)
    dPhi/dt  = -div(Phi V)                          Phi = g h (fluid layer), Phi_s = g h_s (topography)

Transforms use normalised associated Legendre functions (orthonormal on [-1, 1]) and Gauss-Legendre quadrature, so
the analysis of grid products is exact up to the 2N alias-free limit (nlat >= (3N+1)/2, nlon >= 3N+1). Divergence and
curl of a flux vector are analysed directly (integration by parts against the Legendre derivative), which keeps the
global mean of Phi exactly constant: mass is conserved to round-off. Time stepping is classical RK4 (explicit; the time
step follows the gravity-wave CFL, see :func:`stable_dt`), with optional implicit del^4 hyper-diffusion.

Test cases (Williamson, Drake, Hack, Jakob & Swarztrauber 1992, J. Comput. Phys. 102:211-224):
* case 2: steady zonal geostrophic flow (exact solution = initial state), optionally rotated by alpha;
* case 5: zonal flow over an isolated mountain (no analytic solution);
* case 6: Rossby-Haurwitz wave of wavenumber 4.

Diagnostics: total mass, total energy and potential enstrophy, which the continuous equations conserve, and the
normalised l1, l2, l_inf height errors of Williamson et al. (their eqs. 82-84).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["EARTH", "SphericalHarmonics", "legendre_tables", "SphereShallowWater", "williamson_case2", "williamson_case5",
           "williamson_case6", "height_errors", "rossby_haurwitz_speed", "stable_dt"]

EARTH = {"a": 6.37122e6, "omega": 7.292e-5, "g": 9.80616}      # values prescribed by Williamson et al. (1992)


def _epsilon(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = n.astype(float)
    m = m.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.sqrt(np.clip((n * n - m * m) / (4 * n * n - 1), 0.0, None))
    return np.where(n >= np.abs(m), e, 0.0)


def legendre_tables(mu: np.ndarray, M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised associated Legendre functions P[j, m, n] (orthonormal on [-1, 1], n <= N + 1) and
    H[j, m, n] = (1 - mu^2) dP_n^m/dmu (n <= N) at the points ``mu``; zero where n < m."""
    mu = np.asarray(mu, dtype=float)
    coslat = np.sqrt(1 - mu ** 2)
    P = np.zeros((len(mu), M + 1, N + 2))
    # P_0^0 = 1/sqrt(2); diagonal and first off-diagonal, then the three-term recurrence in n
    P[:, 0, 0] = np.sqrt(0.5)
    for m in range(1, M + 1):
        P[:, m, m] = np.sqrt((2 * m + 1) / (2 * m)) * coslat * P[:, m - 1, m - 1]
    for m in range(M + 1):
        if m + 1 <= N + 1:
            P[:, m, m + 1] = np.sqrt(2 * m + 3) * mu * P[:, m, m]
        for n in range(m + 2, N + 2):
            e_n = _epsilon(np.array(n), np.array(m))
            e_n1 = _epsilon(np.array(n - 1), np.array(m))
            P[:, m, n] = (mu * P[:, m, n - 1] - e_n1 * P[:, m, n - 2]) / e_n
    nn, mm = np.meshgrid(np.arange(N + 2), np.arange(M + 1))
    eps = _epsilon(nn, mm)                                            # [m, n]
    H = np.zeros((len(mu), M + 1, N + 1))
    for n in range(N + 1):
        H[:, :, n] = -n * eps[:, n + 1] * P[:, :, n + 1] + ((n + 1) * eps[:, n] * P[:, :, n - 1] if n >= 1 else 0.0)
    valid = nn[:, : N + 1] >= mm[:, : N + 1]
    return P * (nn >= mm), H * valid


class SphericalHarmonics:
    """Scalar and vector spectral transforms for triangular truncation ``trunc`` on a Gaussian grid.

    Spectral arrays have shape (M+1, N+1) with M = N = trunc, entry [m, n] (zero where n < m). Grid arrays have shape
    (nlat, nlon), latitudes south -> north."""

    def __init__(self, trunc: int, nlat: int | None = None, nlon: int | None = None, radius: float = EARTH["a"]):
        self.N = self.M = int(trunc)
        self.nlat = int(nlat or 2 * ((3 * self.N + 1 + 3) // 4))          # >= (3N+1)/2, even
        self.nlon = int(nlon or 2 * self.nlat)
        if self.nlat < (3 * self.N + 1) / 2 or self.nlon < 3 * self.N + 1:
            raise ValueError("grid too coarse for alias-free quadratic products: nlat >= (3N+1)/2, nlon >= 3N+1")
        self.a = float(radius)
        mu, w = np.polynomial.legendre.leggauss(self.nlat)
        self.mu, self.w = mu, w
        self.lat = np.arcsin(mu)
        self.lon = 2 * np.pi * np.arange(self.nlon) / self.nlon
        self.coslat = np.sqrt(1 - mu ** 2)
        M, N = self.M, self.N
        P, H = legendre_tables(mu, M, N)
        nn, mm = np.meshgrid(np.arange(N + 2), np.arange(M + 1))
        self.P = P[:, :, : N + 1]
        self.H = H
        self.n = nn[:, : N + 1]
        self.m = mm[:, : N + 1]
        self.mask = self.n >= self.m
        self.lap_eig = -self.n * (self.n + 1) / self.a ** 2          # eigenvalues of the Laplacian
        with np.errstate(divide="ignore"):
            self.inv_lap = np.where(self.n > 0, 1.0 / np.where(self.n > 0, self.lap_eig, 1.0), 0.0)
        self._wq = self.w / (1 - mu ** 2)                             # quadrature weight for flux analysis

    # --------------------------------------------------------------------------------------------- Fourier helpers
    def _fft(self, g: np.ndarray) -> np.ndarray:
        return np.fft.rfft(g, axis=1)[:, : self.M + 1] / self.nlon

    def _ifft(self, F: np.ndarray) -> np.ndarray:
        full = np.zeros((self.nlat, self.nlon // 2 + 1), dtype=complex)
        full[:, : self.M + 1] = F * self.nlon
        return np.fft.irfft(full, n=self.nlon, axis=1)

    # --------------------------------------------------------------------------------------------- transforms
    def analysis(self, g: np.ndarray) -> np.ndarray:
        return np.einsum("j,jm,jmn->mn", self.w, self._fft(g), self.P) * self.mask

    def synthesis(self, s: np.ndarray) -> np.ndarray:
        return self._ifft(np.einsum("mn,jmn->jm", s, self.P))

    def grad_cos(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(cos(lat) * d/dx, cos(lat) * d/dy) of the scalar with spectral coefficients ``s``."""
        dl = self._ifft(np.einsum("mn,jmn->jm", 1j * self.m * s, self.P)) / self.a
        dm = self._ifft(np.einsum("mn,jmn->jm", s, self.H)) / self.a
        return dl, dm

    def div_curl(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Spectral divergence and curl of the vector whose cos(lat)-weighted components are (A, B)."""
        Fa, Fb = self._fft(A), self._fft(B)
        im = 1j * np.arange(self.M + 1)[None, :]
        div = (np.einsum("j,jm,jmn->mn", self._wq, im * Fa, self.P) - np.einsum("j,jm,jmn->mn", self._wq, Fb, self.H)) / self.a
        curl = (np.einsum("j,jm,jmn->mn", self._wq, im * Fb, self.P) + np.einsum("j,jm,jmn->mn", self._wq, Fa, self.H)) / self.a
        return div * self.mask, curl * self.mask

    def uv_cos(self, zeta: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """U = u cos(lat), V = v cos(lat) on the grid from spectral vorticity and divergence."""
        psi, chi = zeta * self.inv_lap, delta * self.inv_lap
        psi_l, psi_m = self.grad_cos(psi)
        chi_l, chi_m = self.grad_cos(chi)
        return chi_l - psi_m, psi_l + chi_m

    def integrate(self, g: np.ndarray) -> float:
        """Area integral over the sphere."""
        return float(self.a ** 2 * (2 * np.pi / self.nlon) * np.sum(self.w[:, None] * g))

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.lon, self.lat)


def stable_dt(trunc: int, gh_max: float, u_max: float, radius: float = EARTH["a"], safety: float = 0.7) -> float:
    """Explicit RK4 time step: the fastest resolved wave is (sqrt(gh) + |u|) at wavenumber ~ N/a; RK4 is stable on
    the imaginary axis up to 2.8."""
    return safety * 2.8 * radius / (trunc * (np.sqrt(gh_max) + u_max))


@dataclass
class SphereShallowWater:
    """State: spectral vorticity ``zeta``, divergence ``delta``, fluid geopotential ``phi`` (= g h)."""

    sh: SphericalHarmonics
    zeta: np.ndarray
    delta: np.ndarray
    phi: np.ndarray
    f: np.ndarray                               # Coriolis parameter on the grid
    phi_s: np.ndarray                           # surface geopotential on the grid (g h_s)
    omega: float = EARTH["omega"]
    g: float = EARTH["g"]
    hyperdiffusion: float = 0.0                 # del^4 coefficient [m^4/s]
    t: float = 0.0
    history: list = field(default_factory=list)

    @classmethod
    def from_grid(cls, sh: SphericalHarmonics, u: np.ndarray, v: np.ndarray, h: np.ndarray,
                  hs: np.ndarray | None = None, f: np.ndarray | None = None, g: float = EARTH["g"],
                  omega: float = EARTH["omega"], hyperdiffusion: float = 0.0) -> SphereShallowWater:
        cos = sh.coslat[:, None]
        div, curl = sh.div_curl(u * cos, v * cos)
        if f is None:
            f = 2 * omega * np.broadcast_to(sh.mu[:, None], u.shape)
        hs = np.zeros_like(h) if hs is None else hs
        return cls(sh, curl, div, sh.analysis(g * h), np.array(f, dtype=float), g * hs, omega, g, hyperdiffusion)

    # ------------------------------------------------------------------------------------------ dynamics
    def tendencies(self, zeta, delta, phi):
        sh = self.sh
        U, V = sh.uv_cos(zeta, delta)
        eta = sh.synthesis(zeta) + self.f
        phig = sh.synthesis(phi)
        cos2 = (sh.coslat ** 2)[:, None]
        ke = 0.5 * (U * U + V * V) / cos2
        d_eta, c_eta = sh.div_curl(U * eta, V * eta)
        d_phi, _ = sh.div_curl(U * phig, V * phig)
        bern = sh.analysis(phig + self.phi_s + ke)
        return -d_eta, c_eta - sh.lap_eig * bern, -d_phi

    def _diffuse(self, x, dt):
        if self.hyperdiffusion <= 0:
            return x
        return x / (1 + dt * self.hyperdiffusion * self.sh.lap_eig ** 2)

    def step(self, dt: float) -> None:
        y = (self.zeta, self.delta, self.phi)
        k1 = self.tendencies(*y)
        k2 = self.tendencies(*(a + 0.5 * dt * b for a, b in zip(y, k1, strict=True)))
        k3 = self.tendencies(*(a + 0.5 * dt * b for a, b in zip(y, k2, strict=True)))
        k4 = self.tendencies(*(a + dt * b for a, b in zip(y, k3, strict=True)))
        new = [a + dt / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(y, k1, k2, k3, k4, strict=True)]
        self.zeta = self._diffuse(new[0], dt)
        self.delta = self._diffuse(new[1], dt)
        self.phi = new[2]                       # no diffusion on the mass field: mass stays exact
        self.t += dt

    def run(self, t_end: float, dt: float, diagnostics_every: float | None = None,
            callback: Callable[[SphereShallowWater], None] | None = None) -> SphereShallowWater:
        n = int(round((t_end - self.t) / dt))
        every = max(1, int(round(diagnostics_every / dt))) if diagnostics_every else 0
        if every and not self.history:
            self.history.append(self.diagnostics())
        for i in range(1, n + 1):
            self.step(dt)
            if every and i % every == 0:
                self.history.append(self.diagnostics())
            if callback is not None:
                callback(self)
        return self

    # ------------------------------------------------------------------------------------------ output
    def fields(self) -> dict[str, np.ndarray]:
        sh = self.sh
        U, V = sh.uv_cos(self.zeta, self.delta)
        cos = sh.coslat[:, None]
        return {"u": U / cos, "v": V / cos, "h": sh.synthesis(self.phi) / self.g, "zeta": sh.synthesis(self.zeta),
                "delta": sh.synthesis(self.delta), "hs": self.phi_s / self.g}

    def diagnostics(self) -> dict[str, float]:
        """Total mass [kg/rho], total energy [per unit density] and potential enstrophy (Williamson et al. 1992)."""
        fl = self.fields()
        h, hs = fl["h"], fl["hs"]
        ke = 0.5 * h * (fl["u"] ** 2 + fl["v"] ** 2)
        pe = 0.5 * self.g * ((h + hs) ** 2 - hs ** 2)
        eta = fl["zeta"] + self.f
        return {"t": self.t, "mass": self.sh.integrate(h), "energy": self.sh.integrate(ke + pe),
                "potential_enstrophy": self.sh.integrate(0.5 * eta ** 2 / h)}


# ------------------------------------------------------------------------------------------------ test cases
def williamson_case2(sh: SphericalHarmonics, alpha: float = 0.0, gh0: float = 2.94e4, **kw) -> SphereShallowWater:
    """Steady zonal geostrophic flow, u0 = 2 pi a / 12 days, rotated by ``alpha`` (rad). The Coriolis parameter is
    rotated with it, so the initial state is an exact steady solution."""
    a, om, g = sh.a, kw.get("omega", EARTH["omega"]), kw.get("g", EARTH["g"])
    u0 = 2 * np.pi * a / (12 * 86400)
    lon, lat = sh.grid()
    s = -np.cos(lon) * np.cos(lat) * np.sin(alpha) + np.sin(lat) * np.cos(alpha)
    u = u0 * (np.cos(lat) * np.cos(alpha) + np.cos(lon) * np.sin(lat) * np.sin(alpha))
    v = -u0 * np.sin(lon) * np.sin(alpha)
    h = (gh0 - (a * om * u0 + 0.5 * u0 ** 2) * s ** 2) / g
    return SphereShallowWater.from_grid(sh, u, v, h, f=2 * om * s, **kw)


def williamson_case5(sh: SphericalHarmonics, **kw) -> SphereShallowWater:
    """Zonal flow (u0 = 20 m/s, h0 = 5960 m) over a conical mountain of 2000 m centred at (lon, lat) = (3 pi/2, pi/6),
    radius pi/9. ``h`` is the fluid depth (free surface minus mountain)."""
    a, om, g = sh.a, kw.get("omega", EARTH["omega"]), kw.get("g", EARTH["g"])
    u0, h0, hs0, R = 20.0, 5960.0, 2000.0, np.pi / 9
    lon, lat = sh.grid()
    r = np.sqrt(np.minimum(R ** 2, (lon - 1.5 * np.pi) ** 2 + (lat - np.pi / 6) ** 2))
    hs = hs0 * (1 - r / R)
    surface = h0 - (a * om * u0 + 0.5 * u0 ** 2) * np.sin(lat) ** 2 / g
    return SphereShallowWater.from_grid(sh, u0 * np.cos(lat), np.zeros_like(lat), surface - hs, hs=hs, **kw)


def williamson_case6(sh: SphericalHarmonics, R: int = 4, omega_w: float = 7.848e-6, K: float = 7.848e-6,
                     h0: float = 8000.0, **kw) -> SphereShallowWater:
    """Rossby-Haurwitz wave of wavenumber R (Williamson et al. 1992 eqs. 143-148)."""
    a, Om, g = sh.a, kw.get("omega", EARTH["omega"]), kw.get("g", EARTH["g"])
    w = omega_w
    lon, lat = sh.grid()
    c = np.cos(lat)
    u = a * w * c + a * K * c ** (R - 1) * (R * np.sin(lat) ** 2 - c ** 2) * np.cos(R * lon)
    v = -a * K * R * c ** (R - 1) * np.sin(lat) * np.sin(R * lon)
    A = 0.5 * w * (2 * Om + w) * c ** 2 + 0.25 * K ** 2 * c ** (2 * R) * ((R + 1) * c ** 2 + (2 * R ** 2 - R - 2)
                                                                          - 2 * R ** 2 / c ** 2)
    B = 2 * (Om + w) * K / ((R + 1) * (R + 2)) * c ** R * ((R ** 2 + 2 * R + 2) - (R + 1) ** 2 * c ** 2)
    C = 0.25 * K ** 2 * c ** (2 * R) * ((R + 1) * c ** 2 - (R + 2))
    gh = g * h0 + a ** 2 * (A + B * np.cos(R * lon) + C * np.cos(2 * R * lon))
    return SphereShallowWater.from_grid(sh, u, v, gh / g, **kw)


def rossby_haurwitz_speed(R: int = 4, omega_w: float = 7.848e-6, omega: float = EARTH["omega"]) -> float:
    """Eastward angular phase speed of the Rossby-Haurwitz wave in the non-divergent barotropic model [rad/s]
    (Haurwitz 1940); the shallow-water wave moves slightly slower."""
    return (R * (3 + R) * omega_w - 2 * omega) / ((1 + R) * (2 + R))


def height_errors(sh: SphericalHarmonics, h: np.ndarray, h_ref: np.ndarray) -> dict[str, float]:
    """Normalised l1, l2 and l_inf errors of Williamson et al. (1992), eqs. 82-84."""
    d = h - h_ref
    return {"l1": sh.integrate(np.abs(d)) / sh.integrate(np.abs(h_ref)),
            "l2": np.sqrt(sh.integrate(d * d) / sh.integrate(h_ref * h_ref)),
            "linf": float(np.max(np.abs(d)) / np.max(np.abs(h_ref)))}
