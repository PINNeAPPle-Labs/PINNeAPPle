"""An intermediate coupled model of the tropical Pacific in the spirit of Zebiak & Cane (1987, Mon. Wea. Rev.
115:2262), simplified, and the checks that every component is solved correctly.

* :class:`ShallowWaterOcean` -- linear reduced-gravity (first baroclinic mode) ocean on the equatorial beta-plane,
  Arakawa C-grid, closed basin 120E-80W (walls east and west, sponge layers north and south), Rayleigh friction
  and Newtonian damping r, wind-stress forcing tau / (rho H), RK4 in time. Free waves: the equatorial Kelvin wave
  at c = sqrt(g' H) and long Rossby waves at c / (2n + 1) (Matsuno 1966, J. Met. Soc. Japan 44:25), which reflect at
  the boundaries -- the delayed-oscillator memory of the ocean.
* :class:`GillAtmosphere` -- the steady damped long-wave response to a heating Q (Gill 1980, QJRMS 106:447) solved
  exactly per meridional mode: Q is projected on parabolic-cylinder (Hermite) functions; the Kelvin mode
  (eps + d/dx) q0 = -Q0 and the Rossby modes ((2n + 1) eps - d/dx) q_{n+1} = -(Q_{n-1} + 2n Q_{n+1}) / 2 are solved
  by FFT on a periodic longitude circle; u = (q - r) / 2, p = (q + r) / 2 with r_n = 2 (n + 2) q_{n+2}.
  :func:`gill_analytic` is the closed-form solution for Q = F(x) exp(-y^2 / 2) by periodic Green's functions.
* :func:`sst_tendency` -- the SST anomaly equation dT/dt = -eps_T T + gamma(x) S(y) h + kappa(x) S(y) tau_x (local
  damping, thermocline feedback, Ekman-upwelling / wind feedback), coefficients per longitude estimated from data.
* :class:`CoupledModel` -- ocean + SST + atmosphere (tau = mu * Gill(T), mu fitted to ERA5 stress), optional
  stochastic wind forcing and SST nudging; vectorised over a leading ensemble axis.

Validation (each returns measured vs theory): :func:`kelvin_wave_test`, :func:`rossby_wave_test`,
:func:`gill_test`, :func:`convergence_study`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["BETA", "DEG", "SECONDS_PER_MONTH", "OceanGrid", "ShallowWaterOcean", "GillAtmosphere", "gill_analytic",
           "sst_tendency", "CoupledModel", "forced_ocean_run", "kelvin_wave_test", "rossby_wave_test", "gill_test", "convergence_study"]

BETA = 2.28e-11            # 1 / (m s) at the equator
DEG = 111.195e3            # m per degree of latitude
SECONDS_PER_MONTH = 30.4375 * 86400.0
RHO_W = 1025.0


@dataclass
class OceanGrid:
    """C-grid: h and T at cell centres (ny, nx); u on the nx + 1 west/east faces; v on the ny + 1 south/north faces."""
    lon0: float = 120.0
    lon1: float = 280.0
    dlon: float = 2.5
    lat_max: float = 20.0
    dlat: float = 1.0

    def __post_init__(self):
        self.nx = int(round((self.lon1 - self.lon0) / self.dlon))
        self.ny = int(round(2 * self.lat_max / self.dlat))
        self.lon = self.lon0 + self.dlon * (np.arange(self.nx) + 0.5)
        self.lat = -self.lat_max + self.dlat * (np.arange(self.ny) + 0.5)
        self.lon_u = self.lon0 + self.dlon * np.arange(self.nx + 1)
        self.lat_v = -self.lat_max + self.dlat * np.arange(self.ny + 1)
        self.dx, self.dy = self.dlon * DEG, self.dlat * DEG       # beta-plane: no cos(lat) metric
        self.y = self.lat * DEG
        self.y_v = self.lat_v * DEG


class ShallowWaterOcean:
    """Linear reduced-gravity shallow water on the equatorial beta-plane (dimensional, SI units).

    du/dt =  f v - g' dh/dx + tau_x / (rho H) - r u
    dv/dt = -f u - g' dh/dy + tau_y / (rho H) - r v
    dh/dt = -H (du/dx + dv/dy) - r h,          f = beta y,   g' = c^2 / H

    ``r_days`` is the damping time; a sponge raises the damping to 1 / ``sponge_days`` over the ``sponge_deg``
    nearest the north and south walls. States are dicts of arrays (NumPy or torch) with any leading batch shape;
    the arithmetic runs in torch (``dtype``), which is several times faster than NumPy for ensembles.
    """

    def __init__(self, grid: OceanGrid | None = None, c: float = 2.7, H: float = 150.0, r_days: float = 900.0,
                 sponge_deg: float = 5.0, sponge_days: float = 5.0, dtype: str = "float64"):
        import torch
        self.g = grid or OceanGrid()
        self.c, self.H = float(c), float(H)
        self.gp = self.c ** 2 / self.H
        self.dtype = getattr(torch, dtype)
        g = self.g
        r = 1.0 / (r_days * 86400.0)
        rs = 1.0 / (sponge_days * 86400.0)

        def damp(lat):
            d = np.clip((np.abs(lat) - (g.lat_max - sponge_deg)) / sponge_deg, 0, 1) if sponge_deg > 0 else 0 * lat
            return r + (rs - r) * d ** 2

        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=self.dtype)  # noqa: E731
        self.r_h = t(damp(g.lat)[:, None])
        self.r_u = t(damp(g.lat)[:, None])
        self.r_v = t(damp(g.lat_v)[:, None])
        self.f_u = t(BETA * g.y[:, None])
        self.f_v = t(BETA * g.y_v[:, None])

    def zeros(self, batch=()) -> dict[str, np.ndarray]:
        g = self.g
        return {"u": np.zeros(batch + (g.ny, g.nx + 1)), "v": np.zeros(batch + (g.ny + 1, g.nx)),
                "h": np.zeros(batch + (g.ny, g.nx))}

    def max_dt(self) -> float:
        """A safe RK4 step: 2.6 / sqrt(largest gravity-wave frequency^2 + largest f^2) (the RK4 stability limit
        on the imaginary axis is 2.83)."""
        g = self.g
        om = self.c * np.sqrt((2 / g.dx) ** 2 + (2 / g.dy) ** 2)
        return 2.6 / np.hypot(om, BETA * g.lat_max * DEG)

    def rhs(self, s, taux=None, tauy=None):
        import torch.nn.functional as F
        g = self.g
        u, v, h = s["u"], s["v"], s["h"]
        vc = 0.5 * (v[..., 1:, :] + v[..., :-1, :])                      # v at centres
        vu = 0.5 * (vc[..., :, 1:] + vc[..., :, :-1])                    # at interior u points
        uc = 0.5 * (u[..., :, 1:] + u[..., :, :-1])
        uv = 0.5 * (uc[..., 1:, :] + uc[..., :-1, :])                    # at interior v points
        dui = self.f_u * vu - self.gp * (h[..., :, 1:] - h[..., :, :-1]) / g.dx - self.r_u * u[..., :, 1:-1]
        dvi = -self.f_v[1:-1] * uv - self.gp * (h[..., 1:, :] - h[..., :-1, :]) / g.dy \
            - self.r_v[1:-1] * v[..., 1:-1, :]
        if taux is not None:
            dui = dui + taux[..., :, 1:-1] / (RHO_W * self.H)
        if tauy is not None:
            dvi = dvi + tauy[..., 1:-1, :] / (RHO_W * self.H)
        du = F.pad(dui, (1, 1))                                          # walls: u = 0 east and west
        dv = F.pad(dvi, (0, 0, 1, 1))                                    # v = 0 north and south
        dh = -self.H * ((u[..., :, 1:] - u[..., :, :-1]) / g.dx + (v[..., 1:, :] - v[..., :-1, :]) / g.dy) \
            - self.r_h * h
        return {"u": du, "v": dv, "h": dh}

    def _t(self, a):
        import torch
        return a if isinstance(a, torch.Tensor) else (None if a is None else torch.as_tensor(np.asarray(a),
                                                                                           dtype=self.dtype))

    def step(self, s, dt, taux=None, tauy=None):
        """One RK4 step (stress held fixed). NumPy in, NumPy out; torch in, torch out."""
        import torch
        as_np = not isinstance(s["h"], torch.Tensor)
        st = {k: self._t(v) for k, v in s.items()}
        tx, ty = self._t(taux), self._t(tauy)

        def add(a, b, k):
            return {n: a[n] + k * b[n] for n in a}
        k1 = self.rhs(st, tx, ty)
        k2 = self.rhs(add(st, k1, 0.5 * dt), tx, ty)
        k3 = self.rhs(add(st, k2, 0.5 * dt), tx, ty)
        k4 = self.rhs(add(st, k3, dt), tx, ty)
        out = {n: st[n] + dt / 6 * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) for n in st}
        return {k: v.numpy() for k, v in out.items()} if as_np else out


# ---------------------------------------------------------------------------
# Gill atmosphere
# ---------------------------------------------------------------------------

def _phi(n_max: int, y: np.ndarray) -> np.ndarray:
    """Unnormalised parabolic-cylinder functions phi_n = H_n(y) exp(-y^2 / 2), n = 0..n_max, shape (n_max+1, len y)."""
    y = np.asarray(y, float)
    H = np.zeros((n_max + 1, len(y)))
    H[0] = 1.0
    if n_max >= 1:
        H[1] = 2 * y
    for n in range(1, n_max):
        H[n + 1] = 2 * y * H[n] - 2 * n * H[n - 1]
    return H * np.exp(-y ** 2 / 2)


class GillAtmosphere:
    """Gill (1980) long-wave response in Matsuno units (y in L_a = sqrt(c_a / beta), x in L_a, Q per unit heating).

    ``solve(Q, lat, lon)`` takes the heating on a regular (lat, lon) grid -- any extent; it is zero outside -- and
    returns nondimensional u, v, p on the same grid. The longitude circle is periodic (the whole equator). The
    meridional projection uses ``n_modes`` modes, computed by quadrature on the input latitudes (which should
    extend a few L_a, i.e. tens of degrees, from the equator for accurate projections; Q outside them is zero).
    """

    def __init__(self, eps: float = 0.1, c_a: float = 60.0, n_modes: int = 24, dlon: float = 2.5,
                 lon_offset: float = 0.0):
        self.eps, self.c_a, self.n_modes, self.dlon = float(eps), float(c_a), int(n_modes), float(dlon)
        self.L = np.sqrt(self.c_a / BETA)
        self.nlon = int(round(360.0 / self.dlon))
        self.offset = float(lon_offset) % self.dlon
        self.lon = self.offset + self.dlon * np.arange(self.nlon)
        k = 2 * np.pi * np.fft.rfftfreq(self.nlon, d=self.dlon * DEG / self.L)
        self.ik = 1j * k

    def project(self, Q: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Mode coefficients Q_n(x) (..., n_modes + 2, nlon) of Q (..., nlat, nlon) on the circle."""
        y = np.asarray(lat) * DEG / self.L
        P = _phi(self.n_modes + 1, y)
        from math import factorial
        norm = np.array([2.0 ** n * factorial(n) * np.sqrt(np.pi) for n in range(self.n_modes + 2)])
        w = np.gradient(y)                                               # quadrature weights
        return np.einsum("...yx,ny,y->...nx", Q, P, w) / norm[:, None]

    def solve_modes(self, Qn: np.ndarray):
        """Fourier coefficients of q_n and v_n from heating modes Qn (..., N+2, nlon)."""
        Qh = np.fft.rfft(Qn, axis=-1)
        N = self.n_modes
        qh = np.zeros_like(Qh)
        qh[..., 0, :] = -Qh[..., 0, :] / (self.eps + self.ik)
        for n in range(1, N):                                           # Rossby mode n -> q_{n+1}
            qh[..., n + 1, :] = -(Qh[..., n - 1, :] + 2 * n * Qh[..., n + 1, :]) / 2 / ((2 * n + 1) * self.eps
                                                                                     - self.ik)
        vh = np.zeros_like(Qh)
        vh[..., 0, :] = Qh[..., 1, :]                                   # long waves: no mixed Rossby-gravity mode
        for n in range(1, N):
            vh[..., n, :] = (self.eps + self.ik) * qh[..., n + 1, :] + Qh[..., n + 1, :]
        return qh, vh

    def fields(self, qh, vh, lat):
        N = self.n_modes
        y = np.asarray(lat) * DEG / self.L
        P = _phi(N + 2, y)
        q = np.fft.irfft(qh, n=self.nlon, axis=-1)
        v_n = np.fft.irfft(vh, n=self.nlon, axis=-1)
        r = np.zeros_like(q)
        for n in range(N - 1):
            r[..., n, :] = 2 * (n + 2) * q[..., n + 2, :]
        Pn = P[: N + 2]
        qf = np.einsum("...nx,ny->...yx", q, Pn)
        rf = np.einsum("...nx,ny->...yx", r, Pn)
        vf = np.einsum("...nx,ny->...yx", v_n, Pn)
        return {"u": 0.5 * (qf - rf), "p": 0.5 * (qf + rf), "v": vf}

    def solve(self, Q: np.ndarray, lat: np.ndarray, lon: np.ndarray, out_lat: np.ndarray | None = None
              ) -> dict[str, np.ndarray]:
        """Response to Q (..., nlat, nlon_in) given at longitudes ``lon`` (a subset of the circle at spacing
        ``dlon``); fields returned at ``out_lat`` (default ``lat``) on the input longitudes."""
        lon = (np.asarray(lon) - self.offset) % 360
        idx = np.rint(lon / self.dlon).astype(int) % self.nlon
        if not np.allclose((lon - idx * self.dlon + 180) % 360 - 180, 0, atol=1e-6):
            raise ValueError("input longitudes must lie on the atmosphere's grid (multiples of dlon)")
        full = np.zeros(Q.shape[:-1] + (self.nlon,))
        full[..., idx] = Q
        qh, vh = self.solve_modes(self.project(full, lat))
        f = self.fields(qh, vh, lat if out_lat is None else out_lat)
        return {k: v[..., idx] for k, v in f.items()}


def gill_analytic(x: np.ndarray, y: np.ndarray, L: float, eps: float, period: float) -> dict[str, np.ndarray]:
    """Closed-form long-wave Gill response on a periodic circle of length ``period`` (Matsuno units) to
    Q = F(x) exp(-y^2 / 2) with F = cos(pi x / (2 L)) for |x| < L, else 0. Kelvin part q0 = -int G_K(x - s) F(s) ds,
    Rossby part q2 = -1/2 int G_R(s - x) F(s) ds with the periodic Green's functions
    G(d) = exp(-a d) / (1 - exp(-a P)), a = eps (Kelvin), 3 eps (Rossby), d in [0, P)."""
    from scipy.integrate import quad
    k = np.pi / (2 * L)

    def F(s):
        return np.cos(k * s) if abs(s) < L else 0.0

    def green(a, d):
        return np.exp(-a * (d % period)) / (1 - np.exp(-a * period))

    q0 = np.array([-quad(lambda s, xi=xi: green(eps, xi - s) * F(s), -L, L, limit=200)[0] for xi in x])
    q2 = np.array([-0.5 * quad(lambda s, xi=xi: green(3 * eps, s - xi) * F(s), -L, L, limit=200)[0] for xi in x])
    dq2 = np.gradient(q2, x)
    P = _phi(2, y)
    u = 0.5 * (np.outer(P[0], q0) + np.outer(P[2] - 4 * P[0], q2))
    p = 0.5 * (np.outer(P[0], q0) + np.outer(P[2] + 4 * P[0], q2))
    v = np.outer(P[1], eps * q2 + dq2)
    return {"u": u, "p": p, "v": v, "q0": q0, "q2": q2}


# ---------------------------------------------------------------------------
# SST equation and the coupled model
# ---------------------------------------------------------------------------

def sst_tendency(T, h, taux_c, eps_T, gamma, kappa, S):
    """dT/dt [K/month] = -eps_T T + gamma S h + kappa S tau_x (T, h, tau_x at cell centres; gamma, kappa per
    longitude; S the meridional structure (ny, 1))."""
    return -eps_T * T + (gamma[None, :] * S) * h + (kappa[None, :] * S) * taux_c


@dataclass
class CoupledParams:
    """Parameters of :class:`CoupledModel` (SST coefficients per ocean-grid longitude, in months)."""
    eps_T: float = 1.0 / 6.0                    # 1 / month
    gamma: np.ndarray | None = None             # K / m / month
    kappa: np.ndarray | None = None             # K / (N m-2) / month
    mu: float = 0.01                            # N m-2 per unit nondimensional Gill wind per K of heating
    mu_y: float | None = None                   # meridional stress coefficient (default mu)
    L_T: float = 6.0                            # deg, meridional e-folding of the SST coupling
    noise_tau: float = 0.0                      # N m-2, std of the stochastic zonal stress
    noise_months: float = 1.0                   # decorrelation time of the stochastic stress
    noise_lon: tuple[float, float] = (130.0, 230.0)
    extra: dict = field(default_factory=dict)


class CoupledModel:
    """Linear coupled ocean-atmosphere anomaly model on :class:`OceanGrid`; time in months for the SST equation
    and seconds inside the ocean step. ``closure(T, h)`` (optional) adds a learned SST tendency [K/month] -- the
    hybrid models train it.
    """

    def __init__(self, params: CoupledParams, ocean: ShallowWaterOcean | None = None,
                 atmosphere: GillAtmosphere | None = None, dt_hours: float = 8.0, atm_every: int = 3,
                 closure: Callable | None = None, atm_lat: float = 40.0):
        self.p = params
        self.ocean = ocean or ShallowWaterOcean(dtype="float32")
        self.atm = atmosphere or GillAtmosphere(dlon=self.ocean.g.dlon, lon_offset=self.ocean.g.lon[0])
        g = self.ocean.g
        self.dt = dt_hours * 3600.0
        if self.dt > self.ocean.max_dt():
            raise ValueError(f"dt {dt_hours} h exceeds the stable step {self.ocean.max_dt() / 3600:.1f} h")
        self.steps_per_month = int(round(SECONDS_PER_MONTH / self.dt))
        self.dt = SECONDS_PER_MONTH / self.steps_per_month
        self.atm_every = int(atm_every)
        self.S = np.exp(-(g.lat / params.L_T) ** 2)[:, None]
        self.closure = closure
        # heating grid for the atmosphere: ocean longitudes, latitudes to +-atm_lat (Q = 0 off the ocean grid)
        self.atm_lat = np.arange(-atm_lat + 0.5 * g.dlat, atm_lat, g.dlat)
        self._in = np.searchsorted(self.atm_lat, g.lat[0] - 1e-6)
        self.gamma = np.zeros(g.nx) if params.gamma is None else np.asarray(params.gamma, float)
        self.kappa = np.zeros(g.nx) if params.kappa is None else np.asarray(params.kappa, float)

    # winds ------------------------------------------------------------------
    def gill_winds(self, T):
        """Nondimensional Gill u at u points and v at v points (ocean grid) for heating Q = T (K)."""
        g = self.ocean.g
        Q = np.zeros(T.shape[:-2] + (len(self.atm_lat), g.nx))
        Q[..., self._in:self._in + g.ny, :] = T
        f = self.atm.solve(Q, self.atm_lat, g.lon, out_lat=np.concatenate([g.lat, g.lat_v]))
        uc, vv = f["u"][..., : g.ny, :], f["v"][..., g.ny:, :].copy()
        vv[..., [0, -1], :] = 0.0
        u = np.zeros(T.shape[:-2] + (g.ny, g.nx + 1))
        u[..., :, 1:-1] = 0.5 * (uc[..., :, 1:] + uc[..., :, :-1])
        return u, vv

    def operator(self, rank: int | None = None, tol: float = 1e-4):
        """The Gill map T -> (u at u points, v at v points) as a truncated SVD (it is linear): applied to every
        unit SST anomaly once, then kept to the singular values above ``tol`` times the largest (or ``rank``).
        Cached in memory per model and on disk per (grid, eps, c_a, modes, tol)."""
        if getattr(self, "_op", None) is None:
            import hashlib
            import os
            from pathlib import Path
            g, at = self.ocean.g, self.atm
            key = hashlib.sha256(repr((g.lon0, g.lon1, g.dlon, g.lat_max, g.dlat, at.eps, at.c_a, at.n_modes,
                                       at.dlon, at.offset, self.atm_lat[0], len(self.atm_lat), rank, tol)
                                      ).encode()).hexdigest()[:16]
            cache = Path(os.environ.get("PINNEAPPLE_DATA_CACHE", Path.home() / ".cache" / "pinneapple" / "datasets"))
            path = cache / "enso" / f"gill_operator_{key}.npz"
            if path.exists():
                z = np.load(path)
                self._op = (z["A"], z["Bt"])
                self.op_rank = z["A"].shape[1]
                return self._op
            g = self.ocean.g
            n = g.ny * g.nx
            E = np.eye(n).reshape(n, g.ny, g.nx)
            cols = []
            for i in range(0, n, 400):
                u, v = self.gill_winds(E[i:i + 400])
                cols.append(np.concatenate([u.reshape(len(u), -1), v.reshape(len(v), -1)], axis=1))
            M = np.concatenate(cols).T                                   # (n_out, n)
            U, sv, Vt = np.linalg.svd(M, full_matrices=False)
            k = rank or int(np.sum(sv > tol * sv[0]))
            self._op = (U[:, :k] * sv[:k], Vt[:k])
            self.op_rank = k
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, A=self._op[0], Bt=self._op[1])
        return self._op

    def stress(self, T):
        g = self.ocean.g
        A, Bt = self.operator()
        flat = T.reshape(T.shape[:-2] + (-1,)) @ Bt.T @ A.T
        nu = g.ny * (g.nx + 1)
        u = flat[..., :nu].reshape(T.shape[:-2] + (g.ny, g.nx + 1))
        v = flat[..., nu:].reshape(T.shape[:-2] + (g.ny + 1, g.nx))
        muy = self.p.mu if self.p.mu_y is None else self.p.mu_y
        return self.p.mu * u, muy * v

    def zeros(self, batch=()):
        s = self.ocean.zeros(batch)
        s["T"] = np.zeros(batch + (self.ocean.g.ny, self.ocean.g.nx))
        return s

    def _noise_pattern(self, rng, batch):
        g = self.ocean.g
        lo0, lo1 = self.p.noise_lon
        x0 = rng.uniform(lo0, lo1, size=batch)
        shape_x = np.exp(-((g.lon_u - np.asarray(x0)[..., None]) / 20.0) ** 2)      # ~20 deg wide bursts
        shape_y = np.exp(-(g.lat / 8.0) ** 2)
        return shape_y[:, None] * shape_x[..., None, :]

    def run(self, state, n_months: int, *, T_obs: np.ndarray | None = None, nudge_days: float = 5.0,
            tau_ext: Callable | None = None, seed: int = 0, record: bool = True, stochastic: bool | None = None,
            noise_scale: np.ndarray | float = 1.0):
        """Integrate ``n_months`` from ``state`` (dict of u, v, h, T arrays with a common batch shape).

        ``T_obs`` (n_months, ny, nx) nudges T towards observed anomalies with time scale ``nudge_days``.
        ``tau_ext(m)`` returning (taux, tauy) at u and v points replaces the coupled stress in month m. With
        ``stochastic`` (default: when ``noise_tau`` > 0) a red-noise zonal stress of std ``noise_tau`` (times
        ``noise_scale``, per batch member) with ~20 deg wide patches at random longitudes in ``noise_lon`` is added.
        Returns the final state (NumPy) and, with ``record``, monthly means of T and h (n_months, *batch, ny, nx).
        """
        import torch
        g, oc = self.ocean.g, self.ocean
        dt_ = oc.dtype
        tt = lambda a: torch.as_tensor(np.array(a, float), dtype=dt_)  # noqa: E731
        s = {k: tt(v) for k, v in state.items()}
        batch = tuple(s["h"].shape[:-2])
        rng = np.random.default_rng(seed)
        stoch = (self.p.noise_tau > 0) if stochastic is None else stochastic
        n_up = self.steps_per_month / self.atm_every                    # atmosphere updates per month
        a = float(np.exp(-1.0 / max(self.p.noise_months, 1e-6) / n_up))
        amp = rng.standard_normal(batch) if stoch else None
        pattern = self._noise_pattern(rng, batch) if stoch else None
        scale = np.broadcast_to(np.asarray(noise_scale, float), batch) if stoch else None
        A, Bt = self.operator()
        A, Bt = tt(A), tt(Bt)
        nu = g.ny * (g.nx + 1)
        muy = self.p.mu if self.p.mu_y is None else self.p.mu_y
        eps_T = tt(np.broadcast_to(np.asarray(self.p.eps_T, float), (g.nx,)))
        gS, kS = tt(self.gamma[None, :] * self.S), tt(self.kappa[None, :] * self.S)
        Tobs = None if T_obs is None else tt(T_obs)
        relax = 30.4375 / nudge_days
        recT, recH = [], []
        dtm = self.dt / SECONDS_PER_MONTH
        taux = tauy = None
        noise = None
        for m in range(n_months):
            accT = torch.zeros_like(s["T"])
            accH = torch.zeros_like(s["h"])
            ext = tau_ext(m) if tau_ext is not None else None
            for k in range(self.steps_per_month):
                if k % self.atm_every == 0:
                    if ext is not None:
                        taux, tauy = tt(ext[0]), tt(ext[1])
                    else:
                        flat = (s["T"].reshape(batch + (-1,)) @ Bt.T) @ A.T
                        taux = self.p.mu * flat[..., :nu].reshape(batch + (g.ny, g.nx + 1))
                        tauy = muy * flat[..., nu:].reshape(batch + (g.ny + 1, g.nx))
                    if stoch:
                        amp = a * amp + np.sqrt(1 - a * a) * rng.standard_normal(batch)
                        renew = rng.random(batch) < 1.0 / max(self.p.noise_months * n_up, 1.0)
                        if np.any(renew):
                            pattern = np.where(renew[..., None, None], self._noise_pattern(rng, batch), pattern)
                        noise = tt((self.p.noise_tau * scale * amp)[..., None, None] * pattern)
                tx = taux + noise if stoch else taux
                taux_c = 0.5 * (tx[..., :, 1:] + tx[..., :, :-1])
                dT = -eps_T * s["T"] + gS * s["h"] + kS * taux_c
                if self.closure is not None:
                    dT = dT + self.closure(s["T"], s["h"])
                if Tobs is not None:
                    dT = dT + (Tobs[m] - s["T"]) * relax
                oc_s = oc.step({kk: s[kk] for kk in ("u", "v", "h")}, self.dt, tx, tauy)
                s.update(oc_s)
                s["T"] = s["T"] + dtm * dT
                if record:
                    accT += s["T"]
                    accH += s["h"]
            if record:
                recT.append((accT / self.steps_per_month).numpy())
                recH.append((accH / self.steps_per_month).numpy())
        out = {k: v.numpy() for k, v in s.items()}
        if record:
            return out, np.stack(recT), np.stack(recH)
        return out, None, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _track_peak(x, prof):
    i = int(np.argmax(prof))
    if 0 < i < len(prof) - 1:
        a, b, c = prof[i - 1], prof[i], prof[i + 1]
        return x[i] + 0.5 * (a - c) / (a - 2 * b + c) * (x[1] - x[0])
    return x[i]


def kelvin_wave_test(grid: OceanGrid | None = None, c: float = 2.7, dt_hours: float = 6.0, days: float = 40.0,
                     sigma_km: float = 1500.0) -> dict:
    """Release an equatorial Kelvin wave (h = exp(-y^2 / 2 L^2 - (x - x0)^2 / 2 s^2), u = g' h / c, v = 0) in the
    western basin without damping; its speed is the slope of the tracked equatorial h peak. Theory: c; basin
    crossing time = basin width / c."""
    g = grid or OceanGrid(lat_max=20.0)
    oc = ShallowWaterOcean(g, c=c, r_days=1e9, sponge_deg=4.0, sponge_days=5.0)
    Leq = np.sqrt(c / BETA)
    X = (g.lon - g.lon0) * DEG
    x0 = 0.2 * (g.lon1 - g.lon0) * DEG
    s = oc.zeros()
    s["h"] = np.exp(-(g.y[:, None] / Leq) ** 2 / 2 - ((X[None] - x0) / (sigma_km * 1e3)) ** 2 / 2)
    Xu = (g.lon_u - g.lon0) * DEG
    s["u"] = oc.gp / c * np.exp(-(g.y[:, None] / Leq) ** 2 / 2 - ((Xu[None] - x0) / (sigma_km * 1e3)) ** 2 / 2)
    s["u"][:, [0, -1]] = 0
    dt = dt_hours * 3600
    j0 = np.argmin(np.abs(g.lat))
    jj = [j0 - 1, j0] if g.ny % 2 == 0 else [j0]
    ts, xs = [], []
    n = int(days * 86400 / dt)
    for k in range(n + 1):
        if k % max(1, int(86400 / dt)) == 0:
            ts.append(k * dt)
            xs.append(_track_peak(X, s["h"][jj].mean(0)))
        s = oc.step(s, dt)
    ts, xs = np.asarray(ts), np.asarray(xs)
    keep = xs < X[-1] - 3 * sigma_km * 1e3                              # before it reaches the east wall
    speed = float(np.polyfit(ts[keep], xs[keep], 1)[0])
    width = (g.lon1 - g.lon0) * DEG
    return {"speed_measured": speed, "speed_theory": c, "rel_error": speed / c - 1,
            "crossing_days_measured": width / speed / 86400, "crossing_days_theory": width / c / 86400,
            "dx_km": g.dx / 1e3, "dy_km": g.dy / 1e3, "dt_hours": dt_hours}


def rossby_wave_test(grid: OceanGrid | None = None, c: float = 2.7, dt_hours: float = 6.0, days: float = 150.0,
                     sigma_km: float = 2500.0) -> dict:
    """Release a long n = 1 Rossby wave (Matsuno structure u ~ (2 y^2 - 3), h ~ (2 y^2 + 1) times exp(-y^2 / 2)
    in units of L = sqrt(c / beta)) in the eastern basin; its h centroid moves west at c / 3 in the long-wave
    limit (finite width makes it slightly slower)."""
    g = grid or OceanGrid(lat_max=25.0)
    oc = ShallowWaterOcean(g, c=c, r_days=1e9, sponge_deg=4.0, sponge_days=5.0)
    Leq = np.sqrt(c / BETA)
    yn = g.y / Leq
    X = (g.lon - g.lon0) * DEG
    Xu = (g.lon_u - g.lon0) * DEG
    x0 = 0.7 * (g.lon1 - g.lon0) * DEG
    G = np.exp(-((X - x0) / (sigma_km * 1e3)) ** 2 / 2)
    Gu = np.exp(-((Xu - x0) / (sigma_km * 1e3)) ** 2 / 2)
    s = oc.zeros()
    s["h"] = oc.H * 0.01 * ((2 * yn ** 2 + 1) * np.exp(-yn ** 2 / 2))[:, None] * G[None]
    s["u"] = c * 0.01 * ((2 * yn ** 2 - 3) * np.exp(-yn ** 2 / 2))[:, None] * Gu[None]
    s["u"][:, [0, -1]] = 0
    dt = dt_hours * 3600
    ts, xs = [], []
    n = int(days * 86400 / dt)
    band = np.abs(yn) < 3
    for k in range(n + 1):
        if k % max(1, int(86400 / dt)) == 0:
            hp = np.clip(s["h"][band], 0, None).sum(0)
            ts.append(k * dt)
            xs.append(float((hp * X).sum() / hp.sum()))
        s = oc.step(s, dt)
    ts, xs = np.asarray(ts), np.asarray(xs)
    keep = (ts > 10 * 86400) & (xs > X[0] + 2 * sigma_km * 1e3)
    speed = float(np.polyfit(ts[keep], xs[keep], 1)[0])
    return {"speed_measured": speed, "speed_theory": -c / 3, "rel_error": speed / (-c / 3) - 1,
            "crossing_days_measured": (g.lon1 - g.lon0) * DEG / abs(speed) / 86400,
            "crossing_days_theory": (g.lon1 - g.lon0) * DEG / (c / 3) / 86400}


def gill_test(nlons=(72, 144, 288, 576), eps: float = 0.1, L: float = 2.0, n_modes: int = 12) -> dict:
    """Numerical Gill response to Q = cos(pi x / 2L) exp(-y^2 / 2) on |x| < L against :func:`gill_analytic`, for
    several longitude resolutions (relative L2 error of u, v, p over the band |y| < 3)."""
    rows = []
    for nl in nlons:
        dlon = 360.0 / nl
        atm = GillAtmosphere(eps=eps, n_modes=n_modes, dlon=dlon)
        P = atm.nlon * atm.dlon * DEG / atm.L                         # circle length in L_a
        x = (atm.lon * DEG / atm.L)
        x = np.where(x > P / 2, x - P, x)                              # centre the heating at lon 0
        lat = np.linspace(-60, 60, 481)
        y = lat * DEG / atm.L
        Q = np.exp(-y ** 2 / 2)[:, None] * np.where(np.abs(x) < L, np.cos(np.pi * x / (2 * L)), 0.0)[None]
        num = atm.solve(Q, lat, atm.lon)
        o = np.argsort(x)
        ana = gill_analytic(x[o], y, L, eps, P)
        band = np.abs(y) < 3
        err = {}
        for k in ("u", "v", "p"):
            a = ana[k][band]
            n = num[k][:, o][band]
            err[k] = float(np.linalg.norm(n - a) / np.linalg.norm(a))
        rows.append({"nlon": nl, "dx_La": float(P / nl), **{f"rel_l2_{k}": v for k, v in err.items()}})
    # qualitative pattern at the finest grid
    eq = np.argmin(np.abs(y))
    u_eq = num["u"][eq, o]
    xs = x[o]
    pattern = {"u_east_of_heating": float(u_eq[(xs > L) & (xs < L + 3)].mean()),
               "u_west_of_heating": float(u_eq[(xs < -L) & (xs > -L - 3)].mean()),
               "p_min_lat_deg": float(np.abs(lat[np.argmin(num["p"][:, o][:, np.argmin(np.abs(xs + L))])]))}
    return {"eps": eps, "L": L, "rows": rows, "pattern": pattern}


def convergence_study(factors=(1, 2, 4), c: float = 2.7) -> dict:
    """Kelvin-wave speed error under joint refinement of dx, dy and dt (factor f: dlon = 5/f deg, dlat = 2/f deg,
    dt = 12/f h)."""
    rows = []
    for f in factors:
        g = OceanGrid(dlon=5.0 / f, dlat=2.0 / f, lat_max=20.0)
        r = kelvin_wave_test(g, c=c, dt_hours=12.0 / f)
        rows.append({"factor": f, "dlon": g.dlon, "dlat": g.dlat, "dt_hours": 12.0 / f,
                     "speed": r["speed_measured"], "rel_error": r["rel_error"]})
    errs = np.abs([r["rel_error"] for r in rows])
    order = [float(np.log2(errs[i] / errs[i + 1])) if errs[i + 1] > 0 else None for i in range(len(errs) - 1)]
    return {"rows": rows, "observed_order": order}


def forced_ocean_run(ocean: ShallowWaterOcean, taux: np.ndarray, *, steps_per_month: int = 91,
                     tauy: np.ndarray | None = None, state: dict | None = None) -> tuple[np.ndarray, dict]:
    """Integrate the ocean alone under prescribed monthly stress ``taux`` (months, ny, nx + 1) [N m-2] (and ``tauy``
    at v points); returns the monthly mean h (months, ny, nx) and the final state. NaN stress months are zero."""
    import torch
    dt = SECONDS_PER_MONTH / steps_per_month
    st = state or ocean.zeros(())
    s = {k: torch.as_tensor(np.array(v, float), dtype=ocean.dtype) for k, v in st.items() if k in ("u", "v", "h")}
    out = []
    for m in range(len(taux)):
        tx = torch.as_tensor(np.nan_to_num(np.asarray(taux[m], float)), dtype=ocean.dtype)
        ty = None if tauy is None else torch.as_tensor(np.nan_to_num(np.asarray(tauy[m], float)), dtype=ocean.dtype)
        acc = torch.zeros_like(s["h"])
        for _ in range(steps_per_month):
            s = ocean.step(s, dt, tx, ty)
            acc = acc + s["h"]
        out.append((acc / steps_per_month).numpy())
    return np.stack(out), {k: v.numpy() for k, v in s.items()}
