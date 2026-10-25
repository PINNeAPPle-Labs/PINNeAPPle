"""Ball flight physics: gravity, quadratic drag and Magnus lift, the inverse problem from tracked positions, and a
rolling-ball model for ground passes.

Equation of motion of the ball centre (``s = (x, y, z, vx, vy, vz)``, z up)::

    dv/dt = -g e_z - k_d |v| v + |v| (B x v),        k_d = rho A C_D / (2 m),   B = rho A C_L / (2 m) * s_hat

``C_D`` is the drag coefficient (constant over a flight, or a drag-crisis curve of the Reynolds number), ``B`` the
Magnus vector (lift coefficient times the unit spin axis). Ball constants follow the Laws of the Game (IFAB, Law 2:
circumference 68-70 cm, mass 410-450 g at the start of the match); the defaults are the mid values.

Reference data for comparison (verify the numbers at the source before quoting them):

* Asai, T., Seo, K., Kobayashi, O. & Sakashita, R. (2007) Fundamental aerodynamics of the soccer ball, Sports Eng.
  10, 101-109 (doi:10.1007/BF02844207): critical Reynolds number of soccer balls about 2.2e5-3.0e5, below that of a
  smooth sphere (about 3.5e5-4.0e5).
* Asai, T. & Seo, K. (2013) Aerodynamic drag of modern soccer balls, SpringerPlus 2, 171
  (doi:10.1186/2193-1801-2-171): wind-tunnel drag of 2006-2012 match balls through the drag crisis.
* Hong, S. & Asai, T. (2014) Effect of panel shape of soccer ball on its flight characteristics, Sci. Rep. 4, 5068
  (doi:10.1038/srep05068): Brazuca critical Re about 2.2e5-2.5e5 with C_D about 0.15-0.16 at the critical point.
* Goff, J. E. & Carre, M. J. (2009) Trajectory analysis of a soccer ball, Am. J. Phys. 77, 1020
  (doi:10.1119/1.3197187): drag and lift coefficients from launched-ball trajectories (the method used here).

:data:`DRAG_CRISIS_BAND` holds the band the fitted coefficients are compared with.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Ball", "BALL", "G", "RHO_AIR", "NU_AIR", "DRAG_CRISIS_BAND", "cd_crisis", "simulate", "FlightFit",
           "fit_flight", "predict", "extract_flights", "Flight", "kinematic_extrapolation", "fisher_identifiability",
           "rolling_fit", "rolling_travel_time", "travel_time", "CL_MAX"]

G = 9.81
RHO_AIR = 1.2          # kg/m^3, ~20 C at sea level
NU_AIR = 1.5e-5        # m^2/s


@dataclass(frozen=True)
class Ball:
    mass: float = 0.43                  # kg (Law 2: 0.41-0.45)
    diameter: float = 0.69 / np.pi      # m (circumference 0.68-0.70 m)

    @property
    def area(self) -> float:
        return np.pi * self.diameter ** 2 / 4

    def k(self, coeff: float, rho: float = RHO_AIR) -> float:
        """``rho A C / (2 m)`` in 1/m for a drag or lift coefficient."""
        return rho * self.area * coeff / (2 * self.mass)

    def reynolds(self, speed, nu: float = NU_AIR):
        return np.asarray(speed) * self.diameter / nu


BALL = Ball()

#: published band of soccer-ball drag (wind tunnel, non-spinning): sub-critical ~0.4-0.5, critical Re 2.2e5-3.0e5,
#: super-critical ~0.13-0.25 (Asai et al. 2007; Asai & Seo 2013; Hong & Asai 2014; see the module docstring)
DRAG_CRISIS_BAND = {"re_crit": (2.2e5, 3.0e5), "cd_sub": (0.40, 0.50), "cd_super": (0.13, 0.25)}


def cd_crisis(re, cd_sub: float = 0.45, cd_super: float = 0.18, re_crit: float = 2.6e5, width: float = 0.25e5):
    """A smooth drag-crisis curve: ``cd_sub`` below ``re_crit``, ``cd_super`` above (logistic of width ``width``)."""
    s = 1.0 / (1.0 + np.exp(-(np.asarray(re, float) - re_crit) / width))
    return cd_sub + (cd_super - cd_sub) * s


def _accel(v, kd, B, g, ball: Ball, cd_fn=None):
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    if cd_fn is not None:
        kd = ball.k(cd_fn(ball.reynolds(sp)))
    cr = np.stack([B[..., 1] * v[..., 2] - B[..., 2] * v[..., 1], B[..., 2] * v[..., 0] - B[..., 0] * v[..., 2],
                   B[..., 0] * v[..., 1] - B[..., 1] * v[..., 0]], axis=-1)       # B x v
    a = -kd * sp * v + sp * cr
    a[..., 2] -= g
    return a


def simulate(x0, v0, t, *, cd=0.0, magnus=(0.0, 0.0, 0.0), g=G, ball: Ball = BALL, cd_fn=None,
             dt: float = 0.005) -> np.ndarray:
    """Positions ``(len(t), 3)`` of a ball launched at ``x0`` with velocity ``v0`` at ``t[0]`` (RK4, step ``dt``).

    ``magnus`` is the vector ``B`` of the module docstring in 1/m (``ball.k(C_L) * spin axis``); ``cd_fn(Re)``
    overrides the constant ``cd`` (e.g. :func:`cd_crisis`). Batched: ``x0, v0, magnus`` of shape ``(k, 3)`` and
    ``cd, g`` of shape ``(k,)`` give ``(k, len(t), 3)``.
    """
    t = np.asarray(t, float)
    x = np.array(x0, float)
    batched = x.ndim == 2
    v = np.array(v0, float).reshape(x.shape)
    kd = ball.k(np.asarray(cd, float))
    gg = np.asarray(g, float)
    if batched:
        kd = np.broadcast_to(kd, (x.shape[0],))[:, None]
        gg = np.broadcast_to(gg, (x.shape[0],))
    B = np.broadcast_to(np.asarray(magnus, float), x.shape)
    out = np.empty(((x.shape[0],) if batched else ()) + (len(t), 3))
    tc = t[0]
    out[..., 0, :] = x
    for i in range(1, len(t)):
        T = t[i]
        while tc < T - 1e-12:
            h = min(dt, T - tc)
            a1 = _accel(v, kd, B, gg, ball, cd_fn)
            v2 = v + 0.5 * h * a1
            a2 = _accel(v2, kd, B, gg, ball, cd_fn)
            v3 = v + 0.5 * h * a2
            a3 = _accel(v3, kd, B, gg, ball, cd_fn)
            v4 = v + h * a3
            a4 = _accel(v4, kd, B, gg, ball, cd_fn)
            x = x + h / 6 * (v + 2 * v2 + 2 * v3 + v4)
            v = v + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            tc += h
        out[..., i, :] = x
    return out


# ------------------------------------------------------------------------------------------------- inverse problem
_MODELS = {
    # name: (fit Cd, fit Magnus, fit g)
    "vacuum": (False, False, False),
    "drag": (True, False, False),
    "drag_magnus": (True, True, False),
    "drag_free_g": (True, False, True),
    "vacuum_free_g": (False, False, True),
    "drag_magnus_free_g": (True, True, True),
}

#: upper bound on |C_L| when fitting the Magnus vector (generous: wind-tunnel and trajectory studies of spinning
#: footballs report lift coefficients well below it)
CL_MAX = 0.5


@dataclass
class FlightFit:
    """Result of :func:`fit_flight`: parameters, their Laplace (Gauss-Newton) standard errors and fit quality."""

    model: str
    x0: np.ndarray
    v0: np.ndarray
    cd: float
    magnus: np.ndarray
    g: float
    t0: float
    params: np.ndarray
    names: list[str]
    cov: np.ndarray
    rmse: float
    n: int
    success: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def std(self, name: str) -> float:
        i = self.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0.0)))

    @property
    def speed0(self) -> float:
        return float(np.linalg.norm(self.v0))

    @property
    def bic(self) -> float:
        k = len(self.params)
        return float(self.n * 3 * np.log(max(self.rmse, 1e-9) ** 2) + k * np.log(self.n * 3))

    def predict(self, t, ball: Ball = BALL) -> np.ndarray:
        return simulate(self.x0, self.v0, np.concatenate([[self.t0], np.asarray(t, float)]), cd=self.cd,
                        magnus=self.magnus, g=self.g, ball=ball)[1:]


def _unpack(p, model, g_fixed):
    fit_cd, fit_m, fit_g = _MODELS[model]
    x0, v0 = p[:3], p[3:6]
    i = 6
    cd = 0.0
    if fit_cd:
        cd = p[i]
        i += 1
    B = np.zeros(3)
    if fit_m:
        B = p[i:i + 3]
        i += 3
    g = p[i] if fit_g else g_fixed
    return x0, v0, cd, B, g


def fit_flight(t, xyz, model: str = "drag", *, sigma=None, ball: Ball = BALL, g: float = G,
               observe=(0, 1, 2), cd_prior: tuple[float, float] | None = None, loss: str = "soft_l1",
               f_scale: float = 0.5, cl_max: float = CL_MAX) -> FlightFit:
    """Fit a flight model to tracked positions ``xyz`` (``(n, 3)``; NaN rows are skipped) at times ``t``.

    ``model`` is one of ``vacuum``, ``drag``, ``drag_magnus``, ``drag_free_g``, ``vacuum_free_g``,
    ``drag_magnus_free_g``; each Magnus component is bounded by ``ball.k(cl_max)``. ``observe``
    selects the observed coordinates (``(0, 1)`` for 2-D tracking without ball height: the problem then becomes
    ill-posed, see :func:`fisher_identifiability`). ``cd_prior=(mean, std)`` adds a Gaussian prior on ``C_D``.
    Standard errors come from the Gauss-Newton covariance ``s^2 (J^T J)^-1`` at the optimum (Laplace).
    """
    from scipy.optimize import least_squares
    t = np.asarray(t, float)
    xyz = np.asarray(xyz, float)
    obs = list(observe)
    ok = np.all(np.isfinite(xyz[:, obs]), axis=1)
    t, xyz = t[ok], xyz[ok]
    if len(t) < 4:
        raise ValueError("need at least 4 observed positions")
    t0 = t[0]
    fit_cd, fit_m, fit_g = _MODELS[model]
    # initial guess from a quadratic fit per axis
    tt = t - t0
    c = [np.polyfit(tt, xyz[:, k], 2) if k in obs else np.array([0.0, 0.0, 0.0]) for k in range(3)]
    x0 = np.array([c[k][2] for k in range(3)])
    v0 = np.array([c[k][1] for k in range(3)])
    if 2 not in obs:                        # no height: start on the ground with a guessed elevation
        x0[2] = 0.0
        v0[2] = G * (tt[-1] - tt[0]) / 2
    names = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
    p0 = list(x0) + list(v0)
    lo = [-np.inf] * 3 + [-60.0] * 3
    hi = [np.inf] * 3 + [60.0] * 3
    if fit_cd:
        names.append("cd")
        p0.append(cd_prior[0] if cd_prior else 0.25)
        lo.append(0.0)
        hi.append(1.5)
    if fit_m:
        names += ["Bx", "By", "Bz"]
        p0 += [0.0, 0.0, 0.0]
        bm = ball.k(cl_max)
        lo += [-bm] * 3
        hi += [bm] * 3
    if fit_g:
        names.append("g")
        p0.append(G)
        lo.append(0.0)
        hi.append(30.0)
    sig = np.ones(len(obs)) if sigma is None else np.broadcast_to(np.asarray(sigma, float), (3,))[obs]
    tq = np.concatenate([[t0], t])

    def batch(P):
        P = np.atleast_2d(P)
        cols = [_unpack(p, model, g) for p in P]
        sim = simulate(np.array([c[0] for c in cols]), np.array([c[1] for c in cols]), tq,
                       cd=np.array([c[2] for c in cols]), magnus=np.array([c[3] for c in cols]),
                       g=np.array([c[4] for c in cols]), ball=ball, dt=0.02)[:, 1:]
        R = ((sim[:, :, obs] - xyz[None][:, :, obs]) / sig).reshape(len(P), -1)
        if cd_prior and fit_cd:
            R = np.concatenate([R, ((P[:, 6] - cd_prior[0]) / cd_prior[1])[:, None]], axis=1)
        return R

    def resid(p):
        return batch(p)[0]

    def jac(p):                                  # all central differences in one batched integration
        h = 1e-6 * np.maximum(1.0, np.abs(p))
        E = np.diag(h)
        R = batch(np.concatenate([p + E, p - E]))
        return ((R[:len(p)] - R[len(p):]) / (2 * h[:, None])).T

    p0 = np.clip(np.asarray(p0, float), np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
    res = least_squares(resid, p0, jac=jac, bounds=(lo, hi), loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=400)
    r = resid(res.x)
    n_obs = len(t) * len(obs)
    dof = max(n_obs - len(res.x), 1)
    s2 = float(np.sum(r[:n_obs] ** 2) / dof)
    J = jac(res.x)[:n_obs]
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((len(res.x), len(res.x)), np.nan)
    x0_, v0_, cd, B, g_ = _unpack(res.x, model, g)
    rmse = float(np.sqrt(np.mean((r[:n_obs] * np.repeat(sig[None], len(t), 0).ravel()) ** 2)))
    return FlightFit(model, x0_, v0_, float(cd), B, float(g_), float(t0), res.x, names, cov, rmse, len(t),
                     bool(res.success), {"cond_jtj": float(np.linalg.cond(J.T @ J))})


def predict(fit: FlightFit, t, ball: Ball = BALL) -> np.ndarray:
    """Positions of a fitted flight at times ``t``."""
    return fit.predict(t, ball)


def kinematic_extrapolation(t, xyz, t_new, degree: int = 2) -> np.ndarray:
    """Data-only baseline: a polynomial per axis fitted to the observed positions, evaluated at ``t_new``."""
    t = np.asarray(t, float)
    ok = np.all(np.isfinite(xyz), axis=1)
    out = np.empty((len(t_new), 3))
    for k in range(3):
        c = np.polyfit(t[ok] - t[0], np.asarray(xyz)[ok, k], degree)
        out[:, k] = np.polyval(c, np.asarray(t_new) - t[0])
    return out


# --------------------------------------------------------------------------------------------- flight extraction
@dataclass
class Flight:
    """An aerial ball flight cut from a tracking record (times in s, positions in m)."""

    match_id: Any
    t: np.ndarray
    xyz: np.ndarray
    detected: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def apex(self) -> float:
        return float(np.nanmax(self.xyz[:, 2]))


def extract_flights(track: dict[str, np.ndarray], *, match_id=None, z_on: float = 0.6, min_apex: float = 2.5,
                    min_duration: float = 0.8, max_duration: float = 4.5, max_speed: float = 40.0,
                    min_detected: float = 1.0, fps: float = 10.0, noise: float = 0.15) -> list[Flight]:
    """Aerial flights in a ball track (``frame, x, y, z, detected``, e.g. :func:`data.skillcorner_ball`).

    Every local apex of ``z`` at least ``min_apex`` high is grown down both sides while ``z`` keeps falling away
    from it (up to ``noise`` metres of jitter) and stays above ``z_on``; one frame either side is kept (launch and
    landing). A flight is kept when it lasts ``min_duration``-``max_duration`` s, every frame is detected
    (``min_detected`` = fraction required), the frame numbers have no gap and no frame-to-frame speed exceeds
    ``max_speed`` (tracking jumps). Bounces, headers and a keeper holding the ball end a flight: ``z`` stops falling.
    """
    fr, x, y, z, det = (np.asarray(track[k], float) for k in ("frame", "x", "y", "z", "detected"))
    n = len(z)
    zz = np.where(np.isfinite(z), z, -1.0)
    flights, last_end = [], -1
    apexes = [i for i in range(1, n - 1) if zz[i] >= min_apex and zz[i] >= zz[i - 1] and zz[i] > zz[i + 1]]
    for k in apexes:
        if k <= last_end:
            continue
        i = k
        lo = zz[k]
        while i - 1 >= 0 and zz[i - 1] > z_on and zz[i - 1] <= lo + noise and fr[i - 1] == fr[i] - 1:
            i -= 1
            lo = min(lo, zz[i])
        j = k
        lo = zz[k]
        while j + 1 < n and zz[j + 1] > z_on and zz[j + 1] <= lo + noise and fr[j + 1] == fr[j] + 1:
            j += 1
            lo = min(lo, zz[j])
        last_end = j
        s = slice(max(i - 1, 0), min(j + 2, n))
        P = np.stack([x[s], y[s], z[s]], 1)
        if not np.all(np.isfinite(P)) or not np.all(np.diff(fr[s]) == 1):
            continue
        T = (fr[s] - fr[s][0]) / fps
        if not (min_duration <= T[-1] <= max_duration) or np.mean(det[s]) < min_detected:
            continue
        sp = np.linalg.norm(np.diff(P, axis=0), axis=1) * fps
        if sp.max() <= max_speed:
            flights.append(Flight(match_id, T, P, det[s]))
    return flights


# ----------------------------------------------------------------------------------------------- identifiability
def fisher_identifiability(x0, v0, t, *, cd: float = 0.25, magnus=(0.0, 0.0, 0.0), sigma: float = 0.3,
                           observe=(0, 1, 2), ball: Ball = BALL, fit_magnus: bool = True) -> dict[str, Any]:
    """Cramer-Rao standard deviations of the flight parameters for positions observed with noise ``sigma`` (m).

    Compares what 3-D tracking and 2-D tracking (``observe=(0, 1)``: no ball height, as in most optical tracking
    feeds and broadcast data without a height estimate) can recover: the returned ``std`` per parameter and the
    condition number of the Fisher matrix. Derivatives by central finite differences of :func:`simulate`.
    """
    obs = list(observe)
    p = np.concatenate([x0, v0, [cd], magnus if fit_magnus else []]).astype(float)
    names = ["x0", "y0", "z0", "vx0", "vy0", "vz0", "cd"] + (["Bx", "By", "Bz"] if fit_magnus else [])
    t = np.asarray(t, float)

    def f(q):
        B = q[7:10] if fit_magnus else magnus
        return simulate(q[:3], q[3:6], t, cd=q[6], magnus=B, ball=ball, dt=0.01)[:, obs].ravel()

    J = np.empty((len(t) * len(obs), len(p)))
    for k in range(len(p)):
        h = 1e-4 * max(1.0, abs(p[k])) if k < 7 else 1e-5
        dp = np.zeros_like(p)
        dp[k] = h
        J[:, k] = (f(p + dp) - f(p - dp)) / (2 * h)
    F = J.T @ J / sigma ** 2
    w = np.linalg.eigvalsh(F)
    cov = np.linalg.inv(F + np.eye(len(p)) * w.max() * 1e-14)     # a null direction shows as a huge std
    return {"names": names, "std": dict(zip(names, np.sqrt(np.clip(np.diag(cov), 0, None)).tolist(), strict=True)),
            "cond": float(w.max() / max(w.min(), 1e-300)), "min_eig": float(w.min()), "rank": int(np.sum(w > w.max() * 1e-10))}


# ------------------------------------------------------------------------------------------------ ground passes
def rolling_fit(t, xy, *, ball: Ball = BALL) -> dict[str, float]:
    """Fit ``dv/dt = -mu g - k_d v^2`` (rolling resistance plus drag) to a ground pass observed in 2-D.

    Returns ``v0`` (m/s), ``decel`` (``mu g`` in m/s^2), ``cd`` (air drag, usually poorly constrained over a short
    roll) and the position RMSE. The path is taken along its chord (straight ground pass).
    """
    from scipy.optimize import least_squares
    t = np.asarray(t, float) - t[0]
    xy = np.asarray(xy, float)
    ok = np.all(np.isfinite(xy), axis=1)
    t, xy = t[ok], xy[ok]
    d = np.linalg.norm(xy - xy[0], axis=1)
    if len(t) < 4 or d[-1] < 3:
        raise ValueError("pass too short")

    def s_of(p, tq):
        v0, a, cd = p
        k = ball.k(cd)
        out = np.empty_like(tq)
        s = v = 0.0
        v = v0
        tc = 0.0
        h = 0.01
        for i, T in enumerate(tq):
            while tc < T - 1e-12:
                hh = min(h, T - tc)
                acc = -a - k * v * v if v > 0 else 0.0
                s += v * hh + 0.5 * acc * hh * hh
                v = max(v + acc * hh, 0.0)
                tc += hh
            out[i] = s
        return out

    v_guess = max((d[min(3, len(d) - 1)] - d[0]) / max(t[min(3, len(t) - 1)], 1e-3), 1.0)
    res = least_squares(lambda p: s_of(p, t) - d, [v_guess, 1.0, 0.2], bounds=([0.5, 0.0, 0.0], [45, 15, 1.5]),
                        loss="soft_l1", f_scale=0.3)
    r = s_of(res.x, t) - d
    return {"v0": float(res.x[0]), "decel": float(res.x[1]), "cd": float(res.x[2]),
            "rmse": float(np.sqrt(np.mean(r ** 2))), "distance": float(d[-1]), "duration": float(t[-1])}


def rolling_travel_time(distance, v0, decel: float, cd: float = 0.0, ball: Ball = BALL):
    """Time for a rolling ball launched at ``v0`` to cover ``distance`` (inf if it stops first).

    Closed form for ``cd = 0`` (constant deceleration); with drag, ``v(s)`` solves ``v dv/ds = -a - k v^2``:
    ``v^2(s) = (v0^2 + a/k) exp(-2 k s) - a/k``, and the time is integrated numerically.
    """
    distance = np.asarray(distance, float)
    v0 = np.broadcast_to(np.asarray(v0, float), distance.shape)
    if cd <= 0:
        disc = v0 ** 2 - 2 * decel * distance
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = np.where(disc >= 0, (v0 - np.sqrt(np.clip(disc, 0, None))) / max(decel, 1e-12), np.inf)
            if decel <= 1e-12:
                tt = distance / v0
        return tt
    k = ball.k(cd)
    out = np.empty(distance.shape)
    for idx in np.ndindex(distance.shape):
        D, V = distance[idx], v0[idx]
        s = np.linspace(0, D, 200)
        v2 = (V ** 2 + decel / k) * np.exp(-2 * k * s) - decel / k
        if np.any(v2 <= 1e-6):
            out[idx] = np.inf
            continue
        out[idx] = np.trapezoid(1 / np.sqrt(v2), s) if hasattr(np, "trapezoid") else np.trapz(1 / np.sqrt(v2), s)
    return out


def travel_time(distance, *, speed: float | None = None, v0: float | None = None, decel: float | None = None):
    """Ball travel time to a target: constant ``speed`` (the usual pitch-control assumption) or a rolling ball."""
    if speed is not None:
        return np.asarray(distance, float) / speed
    return rolling_travel_time(distance, v0, decel)
