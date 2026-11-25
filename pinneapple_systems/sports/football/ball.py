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

Broadcast height. The ``z`` of a broadcast tracking feed is an estimate from a monocular detector, not a
measurement, and in the SkillCorner open data it is not physically consistent: fitting gravity freely to each aerial
flight gives about 6 m/s^2 instead of 9.81 at every launch speed (#902). :class:`HeightMap` and
:func:`fit_height_map` repair the feed instead of the physics: a monotone map ``z_true = f(z_feed)`` is fitted
**jointly** over every flight of a training split with ``g`` held at 9.81 and a drag (and optionally Magnus) model per
flight, by alternating the per-flight fits with a monotone regression of the feed height on the fitted one. Gravity
is then a check rather than a parameter: on held-out matches the corrected height must give 9.81 back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Ball", "BALL", "G", "RHO_AIR", "NU_AIR", "DRAG_CRISIS_BAND", "cd_crisis", "simulate", "FlightFit",
           "fit_flight", "predict", "extract_flights", "Flight", "kinematic_extrapolation", "fisher_identifiability",
           "rolling_fit", "rolling_travel_time", "travel_time", "CL_MAX", "HeightMap", "HeightFit", "fit_height_map",
           "correct_flight", "height_recovery_test", "identifiability_of", "CR_CD_MAX", "CR_CL_MAX", "CR_SIGMA_FLOOR",
           "CAMERA_Y", "known_height_distortion", "apparent_gravity"]

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
               f_scale: float = 0.5, cl_max: float = CL_MAX, dt: float = 0.02,
               p0=None, max_nfev: int = 400) -> FlightFit:
    """Fit a flight model to tracked positions ``xyz`` (``(n, 3)``; NaN rows are skipped) at times ``t``.

    ``model`` is one of ``vacuum``, ``drag``, ``drag_magnus``, ``drag_free_g``, ``vacuum_free_g``,
    ``drag_magnus_free_g``; each Magnus component is bounded by ``ball.k(cl_max)``. ``observe``
    selects the observed coordinates (``(0, 1)`` for 2-D tracking without ball height: the problem then becomes
    ill-posed, see :func:`fisher_identifiability`). ``cd_prior=(mean, std)`` adds a Gaussian prior on ``C_D``.
    ``max_nfev`` caps the iterations (a badly scaled problem burns the budget without converging).
    Standard errors come from the Gauss-Newton covariance ``s^2 (J^T J)^-1`` at the optimum (Laplace). ``dt`` is the
    integration step of the residual (0.02 s is ample at the 10 Hz of a tracking feed); ``p0`` starts the fit from
    given parameters in the order of ``model``'s names instead of the quadratic guess (much faster when iterating).
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
    p_start = None if p0 is None else np.asarray(p0, float)
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
                       g=np.array([c[4] for c in cols]), ball=ball, dt=dt)[:, 1:]
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

    p0 = np.clip(np.asarray(p_start if p_start is not None else p0, float), np.asarray(lo) + 1e-9,
                 np.asarray(hi) - 1e-9)
    res = least_squares(resid, p0, jac=jac, bounds=(lo, hi), loss=loss, f_scale=f_scale, x_scale="jac",
                        max_nfev=max_nfev)
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


# ----------------------------------------------------------------------------------- broadcast height correction
#: a broadcast main camera around the halfway line, on the far side: its ground position ``(0, CAMERA_Y)`` in metres.
#: The height a monocular detector reports scales with the distance to the camera, so this is the reference point of
#: the pitch-dependent ``camera`` height model.
CAMERA_Y = 40.0

#: Cramer-Rao thresholds under which an aerodynamic coefficient is reported (see :func:`identifiability_of`): the
#: standard deviation of ``C_D`` and of each Magnus component, in units of the coefficient. A tenth of ``C_D`` is a
#: fifth of the 0.13-0.50 drag-crisis band of :data:`DRAG_CRISIS_BAND`; 0.05 of ``C_L`` is a fifth of the usual range
#: of a spinning football.
CR_CD_MAX = 0.10
CR_CL_MAX = 0.05

#: position noise floor of a broadcast tracking feed, metres: the bound of :func:`identifiability_of` is inflated to
#: it when a fit's own RMSE comes out below, so a coefficient is never called identified on a residual that is
#: impossibly small (a synthetic flight, or a real one too short for the residuals to reveal the tracker's noise).
CR_SIGMA_FLOOR = 0.20

#: the broadcast-height distortion used by :func:`height_recovery_test`: a known monotone map feed height -> true
#: height, close to the 0.62 scale implied by the apparent gravity of 6.09 m/s^2 in the SkillCorner data (#902), with
#: a compression of the highest balls so the test is not only a scale.
def known_height_distortion(z) -> np.ndarray:
    """The known map of :func:`height_recovery_test`: ``f(z) = 1.62 z (1 - 0.10 tanh(z / 8))``."""
    z = np.asarray(z, float)
    return 1.62 * z * (1.0 - 0.10 * np.tanh(z / 8.0))


def _interp_ext(x, xp, fp) -> np.ndarray:
    """``np.interp``, with linear extrapolation on the first and last segment outside ``[xp[0], xp[-1]]``."""
    x = np.asarray(x, float)
    xp, fp = np.asarray(xp, float), np.asarray(fp, float)
    out = np.interp(x, xp, fp)
    if len(xp) > 1:
        lo, hi = x < xp[0], x > xp[-1]
        if np.any(lo):
            out = np.where(lo, fp[0] + (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-12) * (x - xp[0]), out)
        if np.any(hi):
            out = np.where(hi, fp[-1] + (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-12) * (x - xp[-1]), out)
    return out


def _strictly_increasing(a, eps: float = 1e-6) -> np.ndarray:
    """The cumulative maximum of ``a``, with every step forced to at least ``eps`` (so the map can be inverted)."""
    a = np.maximum.accumulate(np.asarray(a, float))
    return a[0] + np.concatenate([[0.0], np.cumsum(np.maximum(np.diff(a), eps))])


def _camera_u(x, y) -> np.ndarray:
    """Normalised distance of the ball from the broadcast camera (0 at the centre spot, 0.5 at a goal line)."""
    return np.hypot(np.asarray(x, float), np.asarray(y, float) - CAMERA_Y) / CAMERA_Y - 1.0


@dataclass
class HeightMap:
    """A monotone map from the broadcast height to the physical height, ``z_true = f(z_feed)``.

    ``kind`` says what ``f`` is:

    * ``identity`` -- ``f(z) = z``: the feed taken at face value;
    * ``scale`` -- ``f(z) = a z``, ``params = (a,)``: one scale error of the provider's height estimator;
    * ``camera`` -- ``f(z, x, y) = z / (a + b u)``, ``params = (a, b)``: a perspective scale that depends on where
      the ball is, ``u`` the normalised distance to the camera of :data:`CAMERA_Y`;
    * ``spline`` -- a monotone piecewise-linear ``f`` through ``nodes`` (feed heights) with ``values`` (true
      heights), obtained by a monotone regression over every flight.

    Every kind keeps the ground fixed: ``f(0) = 0``. ``x`` and ``y`` are only used by ``camera``.
    """

    kind: str = "identity"
    params: np.ndarray | None = None
    nodes: np.ndarray | None = None
    values: np.ndarray | None = None

    @classmethod
    def identity(cls) -> HeightMap:
        return cls("identity")

    @classmethod
    def from_function(cls, fn, *, z_max: float = 40.0, n: int = 200) -> HeightMap:
        """Build a ``spline`` map sampling ``fn`` (feed height -> true height) on a uniform grid."""
        nodes = np.linspace(0.0, z_max, n)
        return cls("spline", nodes=nodes, values=_strictly_increasing(np.maximum(fn(nodes), 0.0)))

    def scale_of(self, x=None, y=None) -> np.ndarray:
        """The local multiplier ``s`` with ``z_true = s z_feed`` (``1``, ``a`` or ``a + b u``)."""
        shape = np.shape(np.asarray(x if x is not None else 1.0, float))
        if self.kind == "scale":
            return np.full(shape, float(self.params[0]))
        if self.kind == "camera":
            a, b = (float(v) for v in self.params[:2])
            return np.maximum(a + b * _camera_u(x if x is not None else 0.0, y if y is not None else 0.0), 0.05)
        if self.kind == "spline":
            slope = (self.values[1] - self.values[0]) / max(self.nodes[1] - self.nodes[0], 1e-12)
            return np.full(shape, max(float(slope), 1e-6))
        return np.ones(shape)

    def __call__(self, z, x=None, y=None) -> np.ndarray:
        """``f(z)``: the physical height of a ball the feed reports at height ``z``."""
        z = np.asarray(z, float)
        if self.kind == "scale":
            return float(self.params[0]) * z
        if self.kind == "camera":
            return z / self.scale_of(np.zeros_like(z) if x is None else x, np.zeros_like(z) if y is None else y)
        if self.kind == "spline":
            return _interp_ext(z, self.nodes, self.values)
        return z.copy()

    def to_feed(self, z, x=None, y=None) -> np.ndarray:
        """``f^-1(z)``: the height the feed would report for a physical height ``z`` (the identity map's output is
        its input; a monotone map is inverted by interpolation)."""
        z = np.asarray(z, float)
        if self.kind == "scale":
            return z / float(self.params[0])
        if self.kind == "camera":
            return z * self.scale_of(np.zeros_like(z) if x is None else x, np.zeros_like(z) if y is None else y)
        if self.kind == "spline":
            return _interp_ext(z, self.values, self.nodes)
        return z.copy()

    def local_scale(self, z) -> np.ndarray:
        """``f(z) / z`` point by point: how far the feed's height is off where the ball is.

        The identity map returns exactly ``1``: its ratio is undefined at the ground, and the guarded
        ``z / max(z, 0.3)`` the other kinds need would report a spurious 0.37 for a ball at the ball's own radius.
        """
        z = np.asarray(z, float)
        if self.kind == "identity":
            return np.ones(z.shape)
        den = np.where(np.abs(z) > 0.3, z, 0.3)
        return np.asarray(self(z), float) / den

    def is_sane(self, *, lo: float = 0.2, hi: float = 5.0) -> bool:
        """Whether the map is a usable monotone map: finite, increasing from the ground, and within ``lo``-``hi`` of
        the identity over the heights it was fitted on (the guard that stops a degenerate half-step from being kept)."""
        if self.kind == "spline":
            nodes, values = np.asarray(self.nodes, float), np.asarray(self.values, float)
            if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(values))):
                return False
            if len(nodes) < 2 or np.any(np.diff(nodes) <= 0) or np.any(np.diff(values) <= 0):
                return False
            if values[0] < 0 or nodes[0] < 0:
                return False
            up = nodes >= 1.0                     # the ratio is meaningless in the last centimetres above the ground
            if not np.any(up):
                return True
            r = values[up] / np.maximum(nodes[up], 1e-6)
            return bool(np.all((r >= lo) & (r <= hi)))
        if self.kind == "scale":
            return bool(lo <= float(self.params[0]) <= hi)
        if self.kind == "camera":
            a, b = (float(v) for v in self.params[:2])
            us = np.linspace(-1.0, 0.6, 32)
            s = a + b * us
            return bool(np.all(np.isfinite(s)) and np.all((s >= 1.0 / hi) & (s <= 1.0 / lo)))
        return True

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.params is not None:
            out["params"] = [float(v) for v in np.asarray(self.params, float)]
        if self.kind == "spline":
            out["nodes"] = np.asarray(self.nodes, float).ravel().tolist()
            out["values"] = np.asarray(self.values, float).ravel().tolist()
        return out


@dataclass
class HeightFit:
    """A joint fit of :class:`HeightMap` over every flight of a training split (see :func:`fit_height_map`)."""

    height: HeightMap
    fits: list[FlightFit | None]
    history: list[dict[str, Any]]
    kind: str
    model: str
    g: float
    n_used: int

    def residual(self) -> float:
        """RMS of the profile residual (the distance between the fitted trajectory and the feed) in feed units."""
        left = [f for f in self.history if "rms_feed" in f]
        return float(left[-1]["rms_feed"]) if left else float("nan")


def _fit_jobs(job):
    """One per-flight fit for the joint height fit (module level so it can go to a process pool)."""
    t, xyz, model, g, kw = job
    try:
        return fit_flight(t, xyz, model, g=g, **kw)
    except Exception:
        return None


_POOLS: dict[int, Any] = {}


def _pmap(fn, tasks, workers: int):
    """``fn`` over ``tasks``, in a process pool that is **reused** across calls.

    A joint height fit makes a dozen passes over the same flights, and on a platform whose default start method is
    ``spawn`` every fresh pool pays for importing numpy and scipy once per worker -- more than the fits themselves
    when the flights are short. The pool therefore lives as long as the interpreter and is closed by it.
    """
    if workers <= 1 or len(tasks) < 2:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    ex = _POOLS.get(workers)
    if ex is None:
        ex = ProcessPoolExecutor(workers)
        _POOLS[workers] = ex
    return list(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 4))))


def correct_flight(flight: Flight, height: HeightMap) -> Flight:
    """The same flight with the broadcast height replaced by the physical one, ``z -> f(z)``."""
    xyz = np.array(flight.xyz, float, copy=True)
    xyz[:, 2] = height(xyz[:, 2], xyz[:, 0], xyz[:, 1])
    return Flight(flight.match_id, flight.t, xyz, flight.detected)


def _profile_pairs(flights, fits, height: HeightMap, max_rmse: float):
    """``(z_model, z_feed, weight, u, flight_index)`` of every sample of every flight the joint fit keeps.

    A flight's weight is split over its samples (each flight counts once whatever its length) and set to zero when
    its fit failed or left a position RMSE above ``max_rmse`` metres in true-height units. ``flight_index`` says which
    entry of ``flights`` each sample came from, so a callers can group the pairs (e.g. by match).
    """
    zm, zf, w, u, grp = [], [], [], [], []
    for i, (f, fit) in enumerate(zip(flights, fits, strict=True)):
        if fit is None or not fit.success:
            continue
        if fit.rmse > max_rmse * float(np.mean(height.local_scale(f.xyz[:, 2]))):
            continue
        model = fit.predict(f.t)
        ok = np.all(np.isfinite(model), axis=1) & np.isfinite(f.xyz[:, 2])
        if ok.sum() < 3:
            continue
        n = int(ok.sum())
        zm.append(model[ok, 2])
        zf.append(f.xyz[ok, 2])
        w.append(np.full(n, 1.0 / n))
        u.append(_camera_u(f.xyz[ok, 0], f.xyz[ok, 1]))
        grp.append(np.full(n, i))
    if not zm:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int))
    return (np.concatenate(zm), np.concatenate(zf), np.concatenate(w), np.concatenate(u), np.concatenate(grp))


def _update_height(kind: str, pairs, n_nodes: int, anchor_weight: float) -> HeightMap:
    """One half-step of the joint fit: the monotone map that best sends the fitted heights onto the feed heights.

    The regression is of the **feed** height on the fitted physical height (``h = f^-1``), which is the exact
    least-squares half-step for the feed's own measurement error; the map returned is its inverse. A ``spline`` map
    is a monotone (isotonic) regression read on a grid, a ``scale`` a regression through the origin and a ``camera`` a
    weighted linear regression on ``(z, z u)``. Every kind is anchored at the ground, ``h(0) = 0``.
    """
    if kind == "identity":
        return HeightMap.identity()
    zm, zf, w, u = (np.asarray(a, float) for a in list(pairs)[:4])
    keep = np.isfinite(zm) & np.isfinite(zf) & (w > 0)
    zm, zf, w, u = zm[keep], zf[keep], w[keep], u[keep]
    if len(zm) < 10 or w.sum() <= 0:
        return HeightMap.identity()
    w = w / w.sum()
    aw = anchor_weight
    if kind == "scale":
        num = float(np.sum(w * zm * zf))
        den = float(np.sum(w * zm * zm))
        b = num / den if den > 0 else 0.0
        return HeightMap("scale", params=np.array([1.0 / b if b > 1e-6 else 1.0]))
    if kind == "camera":
        X = np.column_stack([zm, zm * u])
        sw = np.sqrt(np.concatenate([w, [aw]]))
        Xa = np.vstack([X, [[0.0, 0.0]]])
        ya = np.concatenate([zf, [0.0]])
        sol, *_ = np.linalg.lstsq(Xa * sw[:, None], ya * sw, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        a = max(a, 0.05)
        us = np.linspace(np.min(u), np.max(u), 64)
        if np.any(a + b * us <= 0.05):                    # keep the local scale positive over the observed pitch
            b = float(np.clip(b, -0.9 * a / max(np.max(np.abs(us)), 1e-6), 0.9 * a / max(np.max(np.abs(us)), 1e-6)))
        return HeightMap("camera", params=np.array([a, b]))
    from sklearn.isotonic import IsotonicRegression
    x = np.concatenate([zm, [0.0]])
    y = np.concatenate([zf, [0.0]])
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        x, y, sample_weight=np.concatenate([w, [aw]]))
    q = np.linspace(0.01, 1.0, n_nodes)                  # to the highest observed height: no blind extrapolation
    nodes_m = np.concatenate([[0.0], np.quantile(zm, q), [np.max(zm)]])
    nodes_m = np.unique(np.round(np.maximum(nodes_m, 0.0), 4))
    feed = np.maximum(iso.predict(nodes_m) - float(iso.predict(np.array([0.0]))[0]), 0.0)   # ground to ground
    feed = _strictly_increasing(feed)
    return HeightMap("spline", nodes=feed, values=nodes_m)


def _corrected_jobs(flights, height: HeightMap, model: str, g: float, kw) -> list[tuple]:
    """The per-flight fit jobs of one round of :func:`fit_height_map`, with the height already corrected."""
    jobs = []
    for f in flights:
        xyz = np.column_stack([f.xyz[:, 0], f.xyz[:, 1], height(f.xyz[:, 2], f.xyz[:, 0], f.xyz[:, 1])])
        jobs.append((f.t, xyz, model, g, kw))
    return jobs


def apparent_gravity(flights, *, max_rmse: float = 0.6, workers: int = 1,
                     fit_kw: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(mean feed height, gravity fitted freely)`` of every flight: the evidence that the feed's height is wrong.

    One :func:`fit_flight` with free gravity per flight, in feed units. A flight whose apparent gravity is ``g``
    reports its heights ``G / g`` times too small, which is the whole idea of the correction below.
    """
    fits = _pmap(_fit_jobs, [(f.t, f.xyz, "drag_free_g", G, dict(fit_kw or {})) for f in flights], workers)
    z, g = [], []
    for f, fit in zip(flights, fits, strict=True):
        if fit is None or not fit.success or not np.isfinite(fit.g):
            continue
        ok = np.isfinite(f.xyz[:, 2])
        if fit.rmse > max_rmse or not np.any(ok):
            continue
        z.append(float(np.mean(f.xyz[ok, 2])))
        g.append(float(fit.g))
    return np.asarray(z, float), np.asarray(g, float)


def _warm_start(flights, kind: str, *, max_rmse: float, workers: int, n_nodes: int, anchor_weight: float,
                fit_kw: dict[str, Any] | None = None) -> HeightMap:
    """A first map read off the flights' own apparent gravity (:func:`apparent_gravity`).

    Each flight gives one point -- the heights it lives at, expanded by ``G / g``, the factor its own free-gravity fit
    asks for -- and a monotone regression through those points is the warm start of the alternation.
    """
    z, g = apparent_gravity(flights, max_rmse=max_rmse, workers=workers, fit_kw=fit_kw)
    ok = (g > 1.0) & (g < 30.0) & (z > 0.2)
    z, g = z[ok], g[ok]
    if len(z) < 10:
        return HeightMap.identity()
    a = np.clip(G / g, 0.3, 4.0)
    if kind == "scale":
        return HeightMap("scale", params=np.array([float(np.median(a))]))
    if kind == "camera":
        return HeightMap("camera", params=np.array([float(1.0 / np.median(a)), 0.0]))
    pairs = (a * z, z, np.full(len(z), 1.0 / len(z)), np.zeros(len(z)))   # (physical, feed) as _update_height wants
    return _update_height("spline", pairs, n_nodes, anchor_weight)


def _gravity_step(flights, height: HeightMap, *, model: str = "drag_free_g", g: float = G, workers: int = 1,
                  fit_kw: dict[str, Any] | None = None, clip: tuple[float, float] = (0.7, 1.4),
                  max_rmse: float = 1.2) -> tuple[HeightMap, dict[str, Any]]:
    """One step of the map driven by gravity itself: a flight whose corrected heights still give ``g_i`` needs its
    heights multiplied by ``G / g_i``, and the map is rescaled locally to do it.

    This is the sensitive half-step the profile regression only sees weakly: with ``g`` fixed at 9.81 the per-flight
    fit can follow a wrongly scaled trajectory fairly well by moving its launch velocity, so the residual barely
    says so, while the free-gravity fit says it directly. Every kind takes the median factor: a scale or camera map
    scales its parameters, and a spline keeps the shape the regression gave it and takes the level from gravity.
    Returns the map and the factors.
    """
    corrected = [correct_flight(f, height) for f in flights]
    jobs = [(f.t, f.xyz, model, g, dict(fit_kw or {})) for f in corrected]
    fits = _pmap(_fit_jobs, jobs, workers)
    zc, s = [], []
    for f, fit in zip(corrected, fits, strict=True):
        if fit is None or not fit.success or not np.isfinite(fit.g) or fit.g <= 0 or fit.rmse > max_rmse:
            continue
        ok = np.isfinite(f.xyz[:, 2])
        if not np.any(ok):
            continue
        zc.append(float(np.mean(f.xyz[ok, 2])))
        s.append(float(np.clip(g / fit.g, clip[0], clip[1])))
    if len(zc) < 5 or height.kind == "identity":
        return height, {"n": len(zc), "factor_median": float("nan")}
    zc, s = np.asarray(zc), np.asarray(s)
    med = float(np.median(s))
    info = {"n": int(len(zc)), "factor_median": med,
            "factor_ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}
    if height.kind == "scale":
        return HeightMap("scale", params=np.asarray(height.params, float) * med), info
    if height.kind == "camera":
        return HeightMap("camera", params=np.asarray(height.params, float) * med), info
    # a spline keeps the shape the profile regression gave it and takes the overall level from gravity: scaling every
    # value keeps the map monotone and the ground on the ground
    return HeightMap("spline", nodes=np.asarray(height.nodes, float),
                     values=np.asarray(height.values, float) * med), info


def fit_height_map(flights, *, kind: str = "spline", rounds: int = 3, model: str = "drag", g: float = G,
                   n_nodes: int = 12, max_rmse: float = 0.6, workers: int = 1,
                   anchor_weight: float = 0.1, fit_kw: dict[str, Any] | None = None,
                   warm_start: bool = True, gravity_step: bool = True) -> HeightFit:
    """Fit ``z_true = f(z_feed)`` jointly over every flight of a training split, with ``g`` held at 9.81.

    The joint criterion is the feed's own least squares: for a flight ``i`` with parameters ``theta_i`` and the map
    ``h = f^-1``, the residual is ``h(z_model(t; theta_i)) - z_feed(t)``. Holding ``f`` fixed, ``theta_i`` is fitted
    by :func:`fit_flight` on the corrected height ``f(z_feed)`` (the two differ by the map's local derivative only);
    holding the trajectories fixed, the monotone half-step is :func:`_update_height`. Alternating the two is
    coordinate descent on that criterion, so the residual falls at every round. ``warm_start`` begins from the
    flights' own apparent gravity (:func:`_warm_start`), where the iteration ends up.

    ``rounds`` alternations are taken and one final fit is made with the last map, so the returned ``fits`` are
    consistent with the returned map. ``model`` is a :func:`fit_flight` model with free gravity **removed** (``g`` is
    a constant here, not a parameter); flights whose fit leaves a position RMSE above ``max_rmse`` metres are dropped
    from the map update. ``workers`` processes run the per-flight fits.
    """
    if "free_g" in model:
        raise ValueError("the joint height fit fixes gravity: use a model without free g")
    kw = dict(fit_kw or {})
    flights = list(flights)
    top = np.concatenate([np.asarray(f.xyz[:, 2], float) for f in flights]) if flights else np.array([1.0])
    top = float(np.nanmax(top)) if np.any(np.isfinite(top)) else 1.0
    grid = np.linspace(0.0, max(top, 1.0), 64)
    height, history = HeightMap.identity(), []
    if warm_start and kind != "identity":
        height = _warm_start(flights, kind, max_rmse=max_rmse, workers=workers, n_nodes=n_nodes,
                             anchor_weight=anchor_weight, fit_kw=kw)
        history.append({"round": 0, "rms_feed": float("nan"), "n_samples": len(flights), "n_used": len(flights),
                        "change_m": float("nan"), "warm": kind})
    for k in range(max(int(rounds), 1)):
        fits = _pmap(_fit_jobs, _corrected_jobs(flights, height, model, g, kw), workers)
        pairs = _profile_pairs(flights, fits, height, max_rmse)
        new = _update_height(kind, pairs, n_nodes, anchor_weight)
        zm, zf, w, _u, _g = pairs
        rms = float(np.sqrt(np.sum(w * (new(zf) - zm) ** 2) / max(w.sum(), 1e-12))) if len(zm) else float("nan")
        history.append({"round": k + 1, "rms_feed": rms, "n_samples": int(len(zm)),
                        "n_used": int(sum(1 for f in fits if f is not None and f.success)),
                        "change_m": float(np.max(np.abs(new(grid) - height(grid))))})
        if kind == "identity":
            break
        if not new.is_sane():                    # a degenerate half-step is not kept and ends the iteration
            history[-1]["rejected"] = True
            break
        height = new
        if k >= 1 and history[-1]["change_m"] < 1e-3:
            break
    if gravity_step and kind != "identity":
        candidate, ginfo = _gravity_step(flights, height, workers=workers, fit_kw=kw)
        if candidate.is_sane():
            height = candidate
        history.append({"round": "gravity", "rms_feed": float("nan"), "n_samples": ginfo.get("n", 0),
                        "n_used": ginfo.get("n", 0), "change_m": float("nan"),
                        "factor_median": ginfo.get("factor_median", float("nan"))})
    fits = _pmap(_fit_jobs, _corrected_jobs(flights, height, model, g, kw), workers)
    n_used = int(sum(1 for f in fits if f is not None and f.success))
    return HeightFit(height, fits, history, kind, model, float(g), n_used)


def identifiability_of(fit: FlightFit, *, cr_cd_max: float = CR_CD_MAX, cr_cl_max: float = CR_CL_MAX,
                       noise_floor: float = CR_SIGMA_FLOOR, ball: Ball = BALL) -> dict[str, Any]:
    """Cramer-Rao read of one flight fit: is ``C_D`` (and the spin) identified well enough to be reported?

    A fit's Laplace standard error ``s^2 (J^T J)^-1`` *is* the Cramer-Rao bound at the optimum, so the rule is: report
    ``C_D`` only where its standard deviation is at most ``cr_cd_max`` and the Magnus coefficient only where every
    component's is at most ``cr_cl_max`` (in ``C_L`` units). The spin is reported as the Magnus coefficient
    ``|B| / ball.k(1)`` rather than in rev/s: the conversion needs a lift slope the data do not identify either.

    The covariance carries the fit's own residual scale, so a flight the model happens to reproduce exactly would
    claim a zero bound -- on simulated data, and on a real flight that is simply too short for the residuals to
    reveal the noise. The bound is therefore inflated to the tracker's own position noise, ``noise_floor`` metres,
    whenever the fit's RMSE falls below it: a parameter is never called identified on the strength of a suspiciously
    clean residual.
    """
    out: dict[str, Any] = {"cd": float("nan"), "cd_std": float("nan"), "cd_identified": False,
                           "cl": float("nan"), "cl_std": float("nan"), "cl_identified": False,
                           "sigma_effective": float("nan")}
    if fit is None or not fit.success:
        return out
    rmse = float(fit.rmse)
    inflate = 1.0 if not np.isfinite(rmse) or rmse >= noise_floor else noise_floor / max(rmse, 1e-9)
    out["sigma_effective"] = max(rmse, noise_floor) if np.isfinite(rmse) else noise_floor
    if "cd" in fit.names:
        out["cd"], out["cd_std"] = float(fit.cd), fit.std("cd") * inflate
        out["cd_identified"] = bool(np.isfinite(out["cd_std"]) and out["cd_std"] <= cr_cd_max)
    b = [n for n in ("Bx", "By", "Bz") if n in fit.names]
    if b:
        stds = np.array([fit.std(n) for n in b], float) * inflate
        out["cl"] = float(np.linalg.norm(fit.magnus) / ball.k(1.0))
        out["cl_std"] = float(np.max(stds) / ball.k(1.0))
        out["cl_identified"] = bool(np.all(np.isfinite(stds)) and out["cl_std"] <= cr_cl_max)
    return out


def _synthetic_kicks(n: int, *, noise: float, fps: float, seed: int, g: float) -> list[Flight]:
    """Kicks like a broadcast feed: speed 15-30 m/s, elevation 15-40 deg, drag crisis, random spin, Gaussian noise."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sp = rng.uniform(15, 30)
        el = np.radians(rng.uniform(15, 40))
        az = rng.uniform(0, 2 * np.pi)
        v0 = sp * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        cd = float(cd_crisis(BALL.reynolds(sp)))
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        P = simulate([0, 0, 0.11], v0, np.arange(0, 2 * v0[2] / g * 0.9, 1 / fps), cd=cd,
                     magnus=BALL.k(rng.uniform(0.0, 0.25)) * axis, g=g)
        keep = P[:, 2] > 0.11
        t, P = np.arange(0, 2 * v0[2] / g * 0.9, 1 / fps)[keep], P[keep]
        out.append(Flight(f"syn{i}", t, P + rng.normal(0, noise, P.shape), np.ones(len(t))))
    return out


def height_recovery_test(n: int = 40, *, kind: str = "spline", noise: float = 0.2, seed: int = 0, rounds: int = 3,
                         f_true=known_height_distortion, model: str = "drag", workers: int = 1,
                         fit_kw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Known-answer test of the correction: flights distorted by a known ``f`` must give ``f`` back.

    Simulated kicks (true ``g = 9.81``) are pushed through the inverse of ``f_true`` (what the broadcast would report)
    and given Gaussian noise in feed units; the joint fit runs on half of them and the rest are held out. Returns the
    error of the recovered map against ``f_true`` on the observed height range, the free-gravity fit on the held-out
    corrected heights (the physics check: it must hold 9.81) and the error of the corrected heights against the truth.
    """
    kicks = _synthetic_kicks(n, noise=noise, fps=10.0, seed=seed, g=G)
    z_top = float(np.max([np.max(k.xyz[:, 2]) for k in kicks]))
    grid = np.linspace(0.0, z_top, 60)              # only where the simulated flights actually go: above that the
    truth = np.asarray(f_true(grid), float)         # map is an extrapolation and the error says nothing
    known = HeightMap.from_function(f_true, z_max=z_top + 10.0)
    fed = [Flight(k.match_id, k.t, np.column_stack([k.xyz[:, 0], k.xyz[:, 1], known.to_feed(k.xyz[:, 2])]),
                  k.detected) for k in kicks]
    k = max(2, len(fed) // 2)
    train, test = fed[:k], fed[k:]
    hf = fit_height_map(train, kind=kind, rounds=rounds, model=model, workers=workers, fit_kw=fit_kw)
    rec = hf.height(grid)
    fit_g = [fit_flight(f.t, correct_flight(f, hf.height).xyz, "drag_free_g", **(fit_kw or {})) for f in test]
    gs = np.array([f.g for f in fit_g if f is not None and f.success and f.rmse < 1.2], float)
    true_z = np.concatenate([c.xyz[:, 2] for c in kicks[k:]])
    feed_z = np.concatenate([f.xyz[:, 2] for f in test])
    corr = [correct_flight(f, hf.height) for f in test]
    corr_z = np.concatenate([f.xyz[:, 2] for f in corr])
    return {"kind": kind, "n_train": len(train), "n_test": len(test), "noise_m": noise,
            "height_range_m": [0.0, z_top],
            "f_max_abs_err_m": float(np.max(np.abs(rec - truth))),
            "f_rmse_m": float(np.sqrt(np.mean((rec - truth) ** 2))),
            "f_max_abs_err_rel": float(np.max(np.abs(rec - truth)) / max(np.max(truth), 1e-9)),
            "g_recovered_median": float(np.median(gs)) if len(gs) else float("nan"),
            "g_recovered_ci95": ([float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))] if len(gs)
                                 else [float("nan")] * 2),
            "height_rmse_before": float(np.sqrt(np.mean((feed_z - true_z) ** 2))),
            "height_rmse_after": float(np.sqrt(np.mean((corr_z - true_z) ** 2))),
            "curve": {"grid": grid.tolist(), "recovered": np.asarray(rec, float).tolist(),
                      "truth": np.asarray(truth, float).tolist()},
            "map": hf.height.as_dict(), "history": hf.history}
