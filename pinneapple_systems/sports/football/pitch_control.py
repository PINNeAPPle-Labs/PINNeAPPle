"""Pitch control: who wins the ball at a target point, from player kinematics and ball travel time.

The model of Spearman, W. (2018) Beyond Expected Goals, MIT Sloan Sports Analytics Conference: every player ``j``
reaches the target at an uncertain time with logistic CDF ``P_j(T) = 1 / (1 + exp(-pi / (sqrt(3) sigma)
(T - tau_j)))`` around his time to intercept ``tau_j``; once there he controls the ball at rate ``lambda_j``; the
control probabilities of the two teams follow ::

    dPPCF_j/dT = (1 - PPCF_att(T) - PPCF_def(T)) P_j(T) lambda_j,     T >= ball arrival time

integrated here with explicit Euler steps of ``dt`` (reimplemented from the paper's equations).

Time-to-intercept models (:func:`time_to_intercept`):

* ``"reaction"`` -- the player keeps his velocity for a reaction time, then runs straight at ``v_max`` (the
  simple form most open implementations use);
* ``"accel"`` -- acceleration-limited motion from the current velocity: the reachable set after ``t`` is the
  disc around the coasting point with the radius covered from rest at ``a_max`` up to ``v_max``; ``tau`` is the
  first ``t`` whose disc holds the target (bisection).

Ball travel time: constant speed (``ball_speed``) or the rolling-ball model of :mod:`.ball` with a kick speed that
grows with the pass length (fitted to the passes of the training match by :func:`fit_ball_travel`).

:func:`fit_params` fits the free parameters by maximum likelihood of observed pass outcomes, so the comparison
with a data-only classifier is at equal data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import ball as B

__all__ = ["PCParams", "time_to_intercept", "control_probability", "PassSet", "passes_from_metrica",
           "fit_ball_travel", "fit_params", "geometric_features", "pitch_surface"]


@dataclass(frozen=True)
class PCParams:
    tti: str = "reaction"
    v_max: float = 5.0            # m/s
    a_max: float = 7.0            # m/s^2
    reaction: float = 0.7         # s
    sigma: float = 0.45           # s, arrival-time spread
    lam: float = 4.3              # 1/s, control rate
    kappa_def: float = 1.0        # defender advantage
    gk_factor: float = 3.0        # keepers control faster (they may use their hands)
    ball: str = "constant"        # "constant" | "rolling"
    ball_speed: float = 15.0      # m/s for "constant"
    roll: tuple = (8.0, 0.4, 1.5)  # (v0 intercept, v0 per metre, deceleration) for "rolling"
    dt: float = 0.04
    horizon: float = 10.0


#: parameters of the open implementations of Spearman (2018) (v_max 5 m/s, reaction 0.7 s, sigma 0.45 s,
#: lambda 4.3 /s, ball 15 m/s), used as the unfitted baseline
DEFAULT = PCParams()

def time_to_intercept(pos, vel, target, p: PCParams) -> np.ndarray:
    """Time for players at ``pos`` with velocity ``vel`` (``(..., 2)``) to reach ``target`` (broadcastable)."""
    pos, vel, target = (np.asarray(a, float) for a in (pos, vel, target))
    if p.tti == "reaction":
        r = pos + vel * p.reaction
        return p.reaction + np.linalg.norm(target - r, axis=-1) / p.v_max
    lo = np.zeros(np.broadcast_shapes(pos.shape, target.shape)[:-1])
    hi = np.full_like(lo, 30.0)
    t_up = p.v_max / p.a_max
    for _ in range(40):
        t = 0.5 * (lo + hi)
        c = pos + vel * t[..., None]
        rad = np.where(t <= t_up, 0.5 * p.a_max * t ** 2, 0.5 * p.a_max * t_up ** 2 + p.v_max * (t - t_up))
        inside = np.linalg.norm(target - c, axis=-1) <= rad
        hi = np.where(inside, t, hi)
        lo = np.where(inside, lo, t)
    return hi


def _ball_time(dist, p: PCParams):
    if p.ball == "constant":
        return np.asarray(dist, float) / p.ball_speed
    v0 = p.roll[0] + p.roll[1] * np.asarray(dist, float)
    t = B.rolling_travel_time(dist, v0, p.roll[2])
    return np.where(np.isfinite(t), t, np.asarray(dist) / np.maximum(v0, 1e-3) * 2)


def control_probability(att_pos, att_vel, def_pos, def_vel, ball_pos, target, p: PCParams = DEFAULT,
                        att_gk=None, def_gk=None) -> np.ndarray:
    """Probability that the attacking team controls the ball at ``target`` (vectorised over events).

    Positions/velocities ``(n, players, 2)`` (NaN rows = absent players), ``ball_pos`` and ``target`` ``(n, 2)``;
    ``att_gk``/``def_gk`` are boolean masks ``(n, players)`` of goalkeepers. Returns ``(n,)``.
    """
    att_pos, att_vel, def_pos, def_vel = (np.asarray(a, float) for a in (att_pos, att_vel, def_pos, def_vel))
    tgt = np.asarray(target, float)[:, None, :]
    tau_a = time_to_intercept(att_pos, np.nan_to_num(att_vel), tgt, p)
    tau_d = time_to_intercept(def_pos, np.nan_to_num(def_vel), tgt, p)
    tau_a = np.where(np.isfinite(att_pos[..., 0]), tau_a, np.inf)
    tau_d = np.where(np.isfinite(def_pos[..., 0]), tau_d, np.inf)
    lam_a = np.full(tau_a.shape, p.lam)
    lam_d = np.full(tau_d.shape, p.lam * p.kappa_def)
    if att_gk is not None:
        lam_a = np.where(att_gk, lam_a * p.gk_factor, lam_a)
    if def_gk is not None:
        lam_d = np.where(def_gk, lam_d * p.gk_factor, lam_d)
    tb = _ball_time(np.linalg.norm(np.asarray(target) - np.asarray(ball_pos), axis=-1), p)
    k = np.pi / (np.sqrt(3) * p.sigma)
    pa = np.zeros(len(tb))
    pd = np.zeros(len(tb))
    for T in np.arange(0, p.horizon, p.dt):
        TT = (tb + T)[:, None]
        with np.errstate(over="ignore"):
            fa = 1 / (1 + np.exp(-k * (TT - tau_a)))
            fd = 1 / (1 + np.exp(-k * (TT - tau_d)))
        rem = np.clip(1 - pa - pd, 0, 1)
        pa = pa + rem * np.sum(fa * lam_a, axis=1) * p.dt
        pd = pd + rem * np.sum(fd * lam_d, axis=1) * p.dt
        s = pa + pd
        scale = np.where(s > 1, 1 / np.maximum(s, 1.0), 1.0)
        pa = pa * scale
        pd = pd * scale
        if np.all(1 - pa - pd < 1e-3):
            break
    tot = np.maximum(pa + pd, 1e-9)
    return pa / tot


# --------------------------------------------------------------------------------------------- passes from Metrica
@dataclass
class PassSet:
    """Passes with the state at the kick: ``(n, 11, 2)`` attacker/defender positions and velocities (causal Kalman
    estimates), ball position, target (pass end), label (1 = completed to a team-mate, 0 = intercepted)."""

    att_pos: np.ndarray
    att_vel: np.ndarray
    def_pos: np.ndarray
    def_vel: np.ndarray
    att_gk: np.ndarray
    def_gk: np.ndarray
    ball: np.ndarray
    target: np.ndarray
    y: np.ndarray
    duration: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.y)

    def subset(self, idx) -> PassSet:
        idx = np.asarray(idx)
        return PassSet(*(getattr(self, k)[idx] for k in ("att_pos", "att_vel", "def_pos", "def_vel", "att_gk",
                                                          "def_gk", "ball", "target", "y", "duration")), self.meta)


def passes_from_metrica(match, *, sigma_pos: float = 0.1, jerk: float = 30.0) -> PassSet:
    """Completed (``PASS``) and intercepted (``BALL LOST`` with an interception subtype) passes of a
    :class:`data.MetricaMatch`, with player states from the causal Kalman filter of :mod:`.kinematics`."""
    from .kinematics import kalman_smooth
    ev = match.events
    dt = 1 / match.fps
    st = {team: kalman_smooth(P, dt, sigma_pos=sigma_pos, jerk=jerk, causal=True) for team, P in match.players.items()}
    sel = (ev["Type"] == "PASS") | ((ev["Type"] == "BALL LOST") & ev["Subtype"].fillna("").str.contains("INTERCEPTION"))
    rows = ev[sel & np.isfinite(ev["End X"]) & np.isfinite(ev["Start X"]) & (ev["End Frame"] > ev["Start Frame"])]
    out = {k: [] for k in ("att_pos", "att_vel", "def_pos", "def_vel", "att_gk", "def_gk", "ball", "target", "y",
                           "duration")}
    for _, r in rows.iterrows():
        team = r["Team"]
        opp = "Away" if team == "Home" else "Home"
        f = int(r["Start Frame"]) - 1
        if f < 0 or f >= len(match.t):
            continue
        out["att_pos"].append(st[team]["pos"][f])
        out["att_vel"].append(st[team]["vel"][f])
        out["def_pos"].append(st[opp]["pos"][f])
        out["def_vel"].append(st[opp]["vel"][f])
        ga = np.zeros(match.players[team].shape[1], bool)
        ga[match.gk[team]] = True
        gd = np.zeros(match.players[opp].shape[1], bool)
        gd[match.gk[opp]] = True
        out["att_gk"].append(ga)
        out["def_gk"].append(gd)
        out["ball"].append([r["Start X"], r["Start Y"]])
        out["target"].append([r["End X"], r["End Y"]])
        out["y"].append(1 if r["Type"] == "PASS" else 0)
        out["duration"].append((r["End Frame"] - r["Start Frame"]) / match.fps)
    # absent players are NaN in the tracking; pad both teams to the same width
    def pad(lst):
        n = max(a.shape[0] for a in lst)
        return np.stack([np.pad(a, ((0, n - a.shape[0]),) + ((0, 0),) * (a.ndim - 1), constant_values=np.nan)
                         if a.dtype.kind == "f" else np.pad(a, (0, n - a.shape[0])) for a in lst])
    return PassSet(*(pad(out[k]) if k in ("att_pos", "att_vel", "def_pos", "def_vel", "att_gk", "def_gk")
                     else np.asarray(out[k], float) for k in out), {"game": match.game})


def fit_ball_travel(ps: PassSet, *, min_len: float = 5.0) -> dict[str, Any]:
    """Fit the rolling-ball travel time ``t(d)`` (kick speed ``v0 = a + b d``, constant deceleration) and a
    constant speed to the observed durations of the completed passes; returns both fits and their RMSE (s)."""
    from scipy.optimize import least_squares
    d = np.linalg.norm(ps.target - ps.ball, axis=1)
    m = (ps.y == 1) & (d >= min_len)
    d, t = d[m], ps.duration[m]

    def tr(q):
        v0 = q[0] + q[1] * d
        tt = B.rolling_travel_time(d, v0, q[2])
        return np.where(np.isfinite(tt), tt, 10.0) - t

    r = least_squares(tr, [8.0, 0.4, 1.5], bounds=([1, 0, 0], [30, 3, 10]), loss="soft_l1", f_scale=0.5)
    speed = float(np.sum(d * t) / np.sum(t * t))          # least squares of t = d / speed in 1/speed
    rmse_roll = float(np.sqrt(np.mean(tr(r.x) ** 2)))
    rmse_const = float(np.sqrt(np.mean((d / speed - t) ** 2)))
    return {"roll": tuple(float(v) for v in r.x), "speed": speed, "rmse_rolling": rmse_roll,
            "rmse_constant": rmse_const, "n": int(m.sum())}


def fit_params(ps: PassSet, base: PCParams, names=("sigma", "lam", "kappa_def", "v_max", "reaction"),
               *, max_iter: int = 200) -> tuple[PCParams, float]:
    """Maximum-likelihood fit of the named parameters on the pass outcomes (Nelder-Mead on log-parameters)."""
    from scipy.optimize import minimize

    from .metrics import log_loss

    x0 = np.log([getattr(base, n) for n in names])

    def nll(x):
        p = replace(base, **{n: float(np.exp(v)) for n, v in zip(names, x, strict=True)})
        q = control_probability(ps.att_pos, ps.att_vel, ps.def_pos, ps.def_vel, ps.ball, ps.target, p,
                                ps.att_gk, ps.def_gk)
        return log_loss(q, ps.y)

    r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-5})
    return replace(base, **{n: float(np.exp(v)) for n, v in zip(names, r.x, strict=True)}), float(r.fun)


def geometric_features(ps: PassSet) -> tuple[np.ndarray, list[str]]:
    """Data-only features of a pass (no kinematics, no physics): lengths, positions, distances of the nearest
    attackers and defenders to the target and to the pass line, defender counts near the target."""
    tgt = ps.target[:, None, :]
    da = np.linalg.norm(ps.att_pos - tgt, axis=2)
    dd = np.linalg.norm(ps.def_pos - tgt, axis=2)
    da = np.where(np.isfinite(da), da, 999)
    dd = np.where(np.isfinite(dd), dd, 999)
    # exclude the passer (nearest attacker to the ball) from the attacker distances
    dball = np.linalg.norm(ps.att_pos - ps.ball[:, None, :], axis=2)
    passer = np.nanargmin(np.where(np.isfinite(dball), dball, 999), axis=1)
    da[np.arange(len(da)), passer] = 999
    sa, sd = np.sort(da, 1), np.sort(dd, 1)
    seg = ps.target - ps.ball
    L = np.linalg.norm(seg, axis=1)
    u = seg / np.maximum(L, 1e-6)[:, None]
    rel = ps.def_pos - ps.ball[:, None, :]
    proj = np.clip(np.sum(rel * u[:, None, :], axis=2), 0, L[:, None])
    perp = np.linalg.norm(rel - proj[..., None] * u[:, None, :], axis=2)
    perp = np.where(np.isfinite(perp), perp, 999)
    X = np.column_stack([L, ps.ball, ps.target, sa[:, 0], sa[:, 1], sd[:, 0], sd[:, 1], sd[:, 2],
                         np.sum(dd < 5, 1), np.sum(dd < 10, 1), perp.min(1), np.sum(perp < 3, 1),
                         sd[:, 0] - sa[:, 0]])
    names = ["length", "ball_x", "ball_y", "target_x", "target_y", "att1", "att2", "def1", "def2", "def3",
             "def_5m", "def_10m", "def_line_min", "def_line_3m", "def1_minus_att1"]
    return X, names


def pitch_surface(att_pos, att_vel, def_pos, def_vel, ball_pos, p: PCParams = DEFAULT, *, nx: int = 53,
                  ny: int = 34, pitch=(105.0, 68.0), att_gk=None, def_gk=None):
    """Attacking-team control probability on an ``nx x ny`` grid for one frame (for figures)."""
    xs = np.linspace(-pitch[0] / 2, pitch[0] / 2, nx)
    ys = np.linspace(-pitch[1] / 2, pitch[1] / 2, ny)
    X, Y = np.meshgrid(xs, ys)
    tg = np.column_stack([X.ravel(), Y.ravel()])
    n = len(tg)
    rep = lambda a: np.repeat(np.asarray(a)[None], n, 0)  # noqa: E731
    q = control_probability(rep(att_pos), rep(att_vel), rep(def_pos), rep(def_vel), rep(ball_pos), tg, p,
                            None if att_gk is None else rep(att_gk), None if def_gk is None else rep(def_gk))
    return xs, ys, q.reshape(ny, nx)
