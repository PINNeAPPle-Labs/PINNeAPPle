"""Discovering physical laws from data with PINNeAPPle.

* ``kepler_law``     -- real orbital data (planets, and the moons of Jupiter and Saturn): the period-radius law and
                         its dependence on the central mass, by sparse regression over power laws.
* ``pendulum_video`` -- law -> video -> law -> video: a pendulum is filmed (rendered), its angle is measured from
                         the frames, the equation of motion is discovered (SINDy) and re-simulated, and the new video
                         is compared with the observed one, also beyond the observed window.
* ``lorenz_discovery`` -- AI-Lorenz style (De Florio, Kevrekidis & Karniadakis 2024): noisy, sparse samples of a
                         chaotic system, smoothed by an extreme-learning-machine fit (the free function of X-TFC) with
                         analytic derivatives, then SINDy recovers the equations.
* ``oscillator_discovery`` -- reuses the ``oscillator`` runs already in the lab database: recovers zeta and omega
                         of every trajectory from the data alone.
"""
from __future__ import annotations

import math

import numpy as np

from ..spec import Experiment, register

# ---------------------------------------------------------------------- shared helpers


def elm_fit(t, y, n_features=300, ridge=1e-8, seed=0, scale=None):
    """Least-squares fit of y(t) with fixed random tanh features (an extreme learning machine, the free function
    of X-TFC). Returns callables for the fit and its first and second derivatives."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    t0, t1 = float(t.min()), float(t.max())
    s = 2.0 / (t1 - t0)
    tau = (t - t0) * s - 1.0
    scale = scale or 1.0
    w = rng.uniform(-scale * 8, scale * 8, n_features)
    b = rng.uniform(-scale * 8, scale * 8, n_features)
    H = np.tanh(np.outer(tau, w) + b)
    beta = np.linalg.solve(H.T @ H + ridge * np.eye(n_features), H.T @ np.asarray(y, float))

    def f(tq, d=0):
        tq = (np.asarray(tq, float) - t0) * s - 1.0
        z = np.tanh(np.outer(tq, w) + b)
        if d == 0:
            return z @ beta
        if d == 1:
            return ((1 - z ** 2) * w) @ beta * s
        return ((-2 * z * (1 - z ** 2)) * w ** 2) @ beta * s ** 2
    return f


def windowed_elm(t, y, window=200, overlap=40, **kw):
    """ELM fits on overlapping windows (long chaotic records); returns y, y' at t (window centres stitched)."""
    t = np.asarray(t)
    n = len(t)
    ys, dys = np.zeros(n), np.zeros(n)
    wsum = np.zeros(n)
    start = 0
    while start < n - 1:
        end = min(n, start + window)
        sl = slice(start, end)
        f = elm_fit(t[sl], y[sl], **kw)
        taper = np.hanning(end - start + 2)[1:-1] + 1e-6
        ys[sl] += f(t[sl]) * taper
        dys[sl] += f(t[sl], 1) * taper
        wsum[sl] += taper
        if end == n:
            break
        start = end - overlap
    return ys / wsum, dys / wsum


class _Law:
    def __init__(self, terms, coefs, rss, bic):
        self.active_terms, self.active_coefficients, self.rss, self.bic = terms, np.asarray(coefs), rss, bic

    def equation(self, lhs):
        return f"{lhs} = " + "  ".join(f"{c:+.4g} {t}" for c, t in zip(self.active_coefficients, self.active_terms,
                                                                        strict=True))


def weak_form_2nd_order(t, u, du_hat, library, width, n_test, p=4):
    """Weak (integral) form of u'' = f(u, u') (Messenger & Bortz's weak SINDy): with compactly supported test
    functions phi_k, integrate by parts so no derivative of the data is needed for the left side or the u' term:
    int phi u'' = int phi'' u, int phi u' = -int phi' u. ``library(u, du)`` returns (columns, names); a column named
    exactly "u'" is computed in the weak form, the others use ``du_hat`` (only inside nonlinear terms)."""
    cols, names = library(u, du_hat)
    rows, tgt = [], []
    for c in np.linspace(t[0] + width / 2, t[-1] - width / 2, n_test):
        a, b = c - width / 2, c + width / 2
        m = (t >= a) & (t <= b)
        tt = t[m]
        s = (tt - a) * (b - tt)
        phi = s ** p
        dphi = p * s ** (p - 1) * (a + b - 2 * tt)
        d2phi = p * (p - 1) * s ** (p - 2) * (a + b - 2 * tt) ** 2 - 2 * p * s ** (p - 1)
        nrm = np.trapezoid(phi, tt)
        tgt.append(np.trapezoid(d2phi * u[m], tt) / nrm)
        row = [np.trapezoid(phi * cols[m, j], tt) / nrm for j in range(cols.shape[1])]
        for j, nm in enumerate(names):
            if nm in ("u'", "θ'"):
                row[j] = -np.trapezoid(dphi * u[m], tt) / nrm
        rows.append(row)
    return np.array(rows), np.array(tgt), names


def sindy(Theta, target, names, threshold=None, max_terms=3, min_contribution=0.03):
    """Sparse law selection: least squares on every subset of up to ``max_terms`` library terms, keep the one
    with the lowest BIC. Exhaustive search is affordable for the small libraries used here and, unlike
    thresholded regression, does not keep collinear look-alikes (θ and θ³ next to sin θ)."""
    import itertools
    n = len(target)
    best = None
    for k in range(1, max_terms + 1):
        for idx in itertools.combinations(range(Theta.shape[1]), k):
            A = Theta[:, idx]
            c, *_ = np.linalg.lstsq(A, target, rcond=None)
            rss = float(np.sum((target - A @ c) ** 2))
            bic = n * math.log(rss / n + 1e-300) + k * math.log(n) * 2
            if best is None or bic < best.bic:
                best = _Law([names[i] for i in idx], c, rss, bic)
    # Occam: drop terms that carry less than ``min_contribution`` of the signal, then refit
    sd = float(np.std(target)) or 1.0
    keep = [i for i, t_ in enumerate(best.active_terms)
            if abs(best.active_coefficients[i]) * float(np.std(Theta[:, names.index(t_)])) >= min_contribution * sd]
    if 0 < len(keep) < len(best.active_terms):
        idx = [names.index(best.active_terms[i]) for i in keep]
        c, *_ = np.linalg.lstsq(Theta[:, idx], target, rcond=None)
        rss = float(np.sum((target - Theta[:, idx] @ c) ** 2))
        best = _Law([names[i] for i in idx], c, rss, best.bic)
    return best


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------- Kepler: real data
# Semi-major axes and sidereal periods from the NASA planetary and satellite fact sheets
# (nssdc.gsfc.nasa.gov/planetary/factsheet, ssd.jpl.nasa.gov/sats/elem). Units: m, s.
AU, DAY, YEAR = 1.495978707e11, 86400.0, 365.25 * 86400.0
ORBITS = {
    "Sun": [("Mercury", 0.3871 * AU, 0.2408 * YEAR), ("Venus", 0.7233 * AU, 0.6152 * YEAR),
            ("Earth", 1.0000 * AU, 1.0000 * YEAR), ("Mars", 1.5237 * AU, 1.8808 * YEAR),
            ("Jupiter", 5.2029 * AU, 11.862 * YEAR), ("Saturn", 9.537 * AU, 29.457 * YEAR),
            ("Uranus", 19.189 * AU, 84.011 * YEAR), ("Neptune", 30.070 * AU, 164.79 * YEAR),
            ("Pluto", 39.482 * AU, 247.94 * YEAR)],
    "Jupiter": [("Io", 421.8e6, 1.769 * DAY), ("Europa", 671.1e6, 3.551 * DAY),
                ("Ganymede", 1070.4e6, 7.155 * DAY), ("Callisto", 1882.7e6, 16.689 * DAY)],
    "Saturn": [("Mimas", 185.54e6, 0.942 * DAY), ("Enceladus", 238.04e6, 1.370 * DAY),
               ("Tethys", 294.67e6, 1.888 * DAY), ("Dione", 377.42e6, 2.737 * DAY),
               ("Rhea", 527.07e6, 4.518 * DAY), ("Titan", 1221.87e6, 15.945 * DAY)],
}
GM_REF = {"Sun": 1.32712440018e20, "Jupiter": 1.26686534e17, "Saturn": 3.7931187e16}   # m^3 s^-2 (JPL)


@register
class KeplerLaw(Experiment):
    name = "kepler_law"
    version = "1"
    description = ("Discover Kepler's third law from real orbital data (planets, moons of Jupiter and Saturn): "
                   "sparse regression over power laws finds P ∝ a^n, n is compared with 3/2, the central masses "
                   "come out as GM, and all three systems collapse onto P = 2π sqrt(a^3 / GM).")
    tags = ["discovery", "real-data", "astronomy", "symbolic-regression"]
    params = {"exponent_grid": 0.25, "seed": 0}

    def run(self, ctx):
        rows = [(sysname, body, a, P) for sysname, bodies in ORBITS.items() for body, a, P in bodies]
        ctx.input("orbits", [{"system": s, "body": b, "a_m": a, "P_s": P} for s, b, a, P in rows])
        results = {}
        plt = _plt()
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        colors = {"Sun": "#d95f02", "Jupiter": "#7570b3", "Saturn": "#1b9e77"}
        for sysname in ORBITS:
            a = np.array([r[2] for r in rows if r[0] == sysname])
            P = np.array([r[3] for r in rows if r[0] == sysname])
            # symbolic regression over power laws: log P = c + n log a, n searched on a rational grid (simplest
            # exponent within the noise), then refit freely to quote the uncertainty
            la, lp = np.log(a), np.log(P)
            A = np.vstack([np.ones_like(la), la]).T
            (c, n), res, *_ = np.linalg.lstsq(A, lp, rcond=None)
            sigma = math.sqrt(float(np.sum((lp - A @ [c, n]) ** 2)) / max(len(la) - 2, 1))
            n_err = sigma / math.sqrt(float(np.sum((la - la.mean()) ** 2)))
            grid = np.arange(0.25, 3.01, ctx.params["exponent_grid"])
            bic = []
            for ng in grid:
                cg = float(np.mean(lp - ng * la))
                rss = float(np.sum((lp - cg - ng * la) ** 2))
                bic.append(len(la) * math.log(rss / len(la) + 1e-30) + math.log(len(la)))
            n_simple = float(grid[int(np.argmin(bic))])
            GM = float(np.mean(4 * math.pi ** 2 * a ** 3 / P ** 2))
            results[sysname] = {"exponent_fit": float(n), "exponent_err": float(n_err), "exponent_law": n_simple,
                                "GM_fit": GM, "GM_ref": GM_REF[sysname], "GM_rel_err": abs(GM / GM_REF[sysname] - 1)}
            ctx.metric(f"{sysname}_exponent", float(n))
            ctx.metric(f"{sysname}_GM_rel_error", results[sysname]["GM_rel_err"])
            ctx.check(f"{sysname}_exponent_is_3/2", value=float(n), reference=1.5, atol=max(5 * n_err, 0.005))
            ctx.check(f"{sysname}_GM_within_1pct", value=results[sysname]["GM_rel_err"], max=0.01)
            ax[0].loglog(a / AU, P / DAY, "o", color=colors[sysname], label=f"around {sysname}: P ∝ a^{n:.4f}")
            aa = np.geomspace(a.min(), a.max(), 50)
            ax[0].loglog(aa / AU, np.exp(c) * aa ** n / DAY, "-", color=colors[sysname], lw=1)
            ax[1].loglog(np.sqrt(a ** 3 / GM_REF[sysname]) / DAY, P / DAY, "o", color=colors[sysname], label=sysname)
        x = np.geomspace(1e-2, 1e6, 10)
        ax[1].loglog(x, 2 * math.pi * x, "k-", lw=1, label="P = 2π √(a³/GM)")
        ax[0].set_xlabel("semi-major axis a [AU]")
        ax[0].set_ylabel("period P [days]")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].set_xlabel("√(a³/GM) [days]")
        ax[1].set_ylabel("P [days]")
        ax[1].legend(frameon=False, fontsize=8)
        ax[0].set_title("Real orbits of 19 bodies around 3 centres", fontsize=10)
        ax[1].set_title("One law for all: collapse on P = 2π √(a³/GM)", fontsize=10)
        fig.tight_layout()
        ctx.figure("kepler", fig)
        ctx.output("discovered", results)
        mass_ratio = results["Sun"]["GM_fit"] / results["Jupiter"]["GM_fit"]
        ctx.metric("sun_to_jupiter_mass_ratio", mass_ratio)
        ctx.check("sun_jupiter_mass_ratio", value=mass_ratio, reference=1047.35, rtol=0.01)
        ds = ctx.dataset("orbits", description="Measured orbits (NASA fact sheets)", units={"a": "m", "P": "s"})
        for s, b, a, P in rows:
            ds.add(system=s, body=b, a=float(a), P=float(P))


# ---------------------------------------------------------------------- pendulum: law -> video -> law -> video
def _pendulum(theta0, omega0, g_over_L, c, dt, n):
    def acc(th, om):
        return -g_over_L * math.sin(th) - c * om
    th, om = theta0, omega0
    out = np.empty(n)
    for i in range(n):
        out[i] = th
        k1t, k1o = om, acc(th, om)
        k2t, k2o = om + 0.5 * dt * k1o, acc(th + 0.5 * dt * k1t, om + 0.5 * dt * k1o)
        k3t, k3o = om + 0.5 * dt * k2o, acc(th + 0.5 * dt * k2t, om + 0.5 * dt * k2o)
        k4t, k4o = om + dt * k3o, acc(th + dt * k3t, om + dt * k3o)
        th += dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
    return out


def _render(theta, L_px, size, noise, rng, blur=True):
    """Video frames (n, H, W) in [0, 1]: a lit rod and bob on a dark background, camera noise, motion blur."""
    H = W = size
    px, py = W / 2, H * 0.18
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = []
    prev = None
    for th in theta:
        bx, by = px + L_px * math.sin(th), py + L_px * math.cos(th)
        img = 0.06 + 0.04 * (yy / H)
        # rod: distance to segment
        vx, vy = bx - px, by - py
        tt = np.clip(((xx - px) * vx + (yy - py) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.hypot(xx - (px + tt * vx), yy - (py + tt * vy))
        img = img + 0.35 * np.exp(-(d / 1.2) ** 2)
        img = img + 0.95 * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * (0.045 * W) ** 2))
        img = img + rng.normal(0, noise, img.shape)
        if blur and prev is not None:
            img = 0.7 * img + 0.3 * prev
        prev = img
        frames.append(np.clip(img, 0, 1).astype(np.float32))
    return np.stack(frames)


def _track(frames, size, L_px):
    """Measure the angle from each frame: intensity-weighted centroid of the brightest blob."""
    H = W = size
    px, py = W / 2, H * 0.18
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for f in frames:
        thr = np.percentile(f, 99.6)
        m = np.clip(f - thr, 0, None)
        cx, cy = (m * xx).sum() / m.sum(), (m * yy).sum() / m.sum()
        out.append(math.atan2(cx - px, cy - py))
    return np.unwrap(np.array(out))


@register
class PendulumVideo(Experiment):
    name = "pendulum_video"
    version = "1"
    description = ("Law -> video -> law -> video. A large-amplitude damped pendulum is filmed (rendered with "
                   "camera noise and motion blur); its angle is measured from the frames; weak-form SINDy discovers "
                   "θ'' = -(g/L) sin θ - c θ' from a library that also offers θ, θ³, cos θ, θ'|θ'|, ...; the "
                   "discovered law is re-simulated and re-rendered, and the predicted video is compared with the "
                   "observed one, including after the observation window.")
    tags = ["discovery", "video", "inverse", "symbolic-regression"]
    params = {"g": 9.81, "L": 0.8, "damping": 0.15, "theta0": 2.4, "fps": 60, "seconds": 10.0, "noise": 0.04,
              "size": 160, "test_width": 2.0, "n_test": 120, "predict_seconds": 6.0, "seed": 0}
    space = {"theta0": (0.3, 2.9), "damping": (0.0, 0.5), "noise": (0.0, 0.1)}

    def run(self, ctx):
        p = ctx.params
        rng = np.random.default_rng(p["seed"])
        dt = 1.0 / p["fps"]
        n_obs = int(p["seconds"] * p["fps"])
        n_all = n_obs + int(p["predict_seconds"] * p["fps"])
        size = int(p["size"])
        L_px = 0.62 * size
        g_over_L = p["g"] / p["L"]
        with ctx.stage("film"):
            truth = _pendulum(p["theta0"], 0.0, g_over_L, p["damping"], dt, n_all)
            frames = _render(truth[:n_obs], L_px, size, p["noise"], rng)
        with ctx.stage("measure"):
            theta_m = _track(frames, size, L_px)
        t = np.arange(n_obs) * dt
        meas_err = float(np.sqrt(np.mean((theta_m - truth[:n_obs]) ** 2)))
        ctx.metric("angle_measurement_rmse_rad", meas_err)
        with ctx.stage("discover"):
            # u' only inside nonlinear terms comes from an ELM fit (the free function of X-TFC); everything else is
            # in weak form, so the noisy second derivative of the measured angle is never computed
            th, om = windowed_elm(t, theta_m, window=600, overlap=150, n_features=300, ridge=1e-6)

            def library(u, du):
                return (np.stack([u, du, np.sin(u), np.cos(u) - 1, u ** 3, du * np.abs(du), u * du], 1),
                        ["θ", "θ'", "sin θ", "cos θ", "θ³", "θ' |θ'|", "θ θ'"])
            Theta, target, names = weak_form_2nd_order(t, theta_m, om, library, width=p["test_width"],
                                                       n_test=p["n_test"])
            res = sindy(Theta, target, names)
        coef = dict(zip(res.active_terms, map(float, res.active_coefficients), strict=True))
        g_found = -coef.get("sin θ", 0.0) * p["L"]
        c_found = -coef.get("θ'", 0.0)
        ctx.output("discovered", {"equation": res.equation("θ''"), "coefficients": coef, "g": g_found,
                                  "damping": c_found})
        ctx.log(res.equation("θ''"))
        ctx.metric("g_discovered", g_found)
        ctx.metric("g_rel_error", abs(g_found / p["g"] - 1))
        ctx.metric("damping_discovered", c_found)
        ctx.metric("n_terms", len(res.active_terms))
        ctx.check("g_within_2pct", value=abs(g_found / p["g"] - 1), max=0.02)
        ctx.check("only_the_true_terms", set(res.active_terms) <= {"sin θ", "θ'"} and "sin θ" in res.active_terms,
                  detail=res.equation("θ''"))
        with ctx.stage("predict"):
            # re-simulate the discovered law from the measured initial state and re-render
            def acc_found(thv, omv):
                lib = {"θ": thv, "θ'": omv, "sin θ": math.sin(thv), "cos θ": math.cos(thv) - 1, "θ³": thv ** 3,
                       "θ' |θ'|": omv * abs(omv), "θ θ'": thv * omv}
                return sum(cf * lib[k] for k, cf in coef.items())
            def simulate(th0, om0, n):
                out = np.empty(n)
                thv, omv = th0, om0
                for i in range(n):
                    out[i] = thv
                    k1t, k1o = omv, acc_found(thv, omv)
                    k2t, k2o = omv + 0.5 * dt * k1o, acc_found(thv + 0.5 * dt * k1t, omv + 0.5 * dt * k1o)
                    k3t, k3o = omv + 0.5 * dt * k2o, acc_found(thv + 0.5 * dt * k2t, omv + 0.5 * dt * k2o)
                    k4t, k4o = omv + dt * k3o, acc_found(thv + dt * k3t, omv + dt * k3o)
                    thv += dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
                    omv += dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
                return out
            # initial state: least-squares fit of the discovered law to the whole observed window (assimilation)
            from scipy.optimize import least_squares
            fit = least_squares(lambda z: simulate(z[0], z[1], n_obs) - theta_m, x0=[float(th[5]), float(om[5])])
            pred = simulate(float(fit.x[0]), float(fit.x[1]), n_all)
        err_obs = float(np.sqrt(np.mean((pred[:n_obs] - truth[:n_obs]) ** 2)))
        err_future = float(np.sqrt(np.mean((pred[n_obs:] - truth[n_obs:]) ** 2)))
        ctx.metric("reconstruction_rmse_rad", err_obs)
        ctx.metric("forecast_rmse_rad", err_future)
        ctx.check("forecast_beyond_observation", value=err_future, max=0.1)
        ds = ctx.dataset("pendulum", description="Video frames, measured angle and the true angle")
        ds.add(frames=(frames * 255).astype(np.uint8), theta_measured=theta_m.astype(np.float32),
               theta_true=truth[:n_obs].astype(np.float32), g=float(p["g"]), L=float(p["L"]),
               damping=float(p["damping"]), fps=int(p["fps"]))
        plt = _plt()
        tt = np.arange(n_all) * dt
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].plot(t, theta_m, ".", ms=2, color="#999", label="measured from video")
        ax[0].plot(tt, truth, "k-", lw=1, label="true motion")
        ax[0].plot(tt, pred, "--", color="#d95f02", lw=1.4, label="discovered law")
        ax[0].axvspan(t[-1], tt[-1], color="#7570b3", alpha=0.08)
        ax[0].text(t[-1], ax[0].get_ylim()[1] * 0.85, "  beyond the video", color="#7570b3", fontsize=8)
        ax[0].set_ylabel("θ [rad]")
        ax[0].legend(frameon=False, fontsize=8, loc="lower left")
        ax[1].plot(tt, np.abs(pred - truth), color="#d95f02")
        ax[1].set_yscale("log")
        ax[1].set_ylabel("|error| [rad]")
        ax[1].set_xlabel("t [s]")
        ax[0].set_title(res.equation("θ''") + f"   →  g = {g_found:.3f} m/s²", fontsize=9)
        fig.tight_layout()
        ctx.figure("pendulum_discovery", fig)
        # side-by-side video: observed (then black, the camera stops) | rendered from the discovered law
        rec = _render(pred, L_px, size, 0.0, rng, blur=False)
        obs = np.concatenate([frames, np.zeros((n_all - n_obs, size, size), np.float32)])
        step = max(1, int(p["fps"] / 15))
        cmap = plt.get_cmap("inferno")
        gif = []
        for i in range(0, n_all, step):
            left = cmap(obs[i])[..., :3]
            right = cmap(rec[i])[..., :3]
            sep = np.ones((size, 4, 3))
            gif.append(np.concatenate([left, sep, right], 1))
        ctx.gif("observed_vs_discovered_law", gif, duration_ms=66)


# ---------------------------------------------------------------------- Lorenz: AI-Lorenz style
def _lorenz(state, sigma, rho, beta, dt, n):
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    out = np.empty((n, 3))
    s = np.array(state, float)
    for i in range(n):
        out[i] = s
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


@register
class LorenzDiscovery(Experiment):
    name = "lorenz_discovery"
    version = "1"
    description = ("AI-Lorenz style equation discovery (De Florio, Kevrekidis & Karniadakis 2024): noisy, "
                   "subsampled observations of the Lorenz system are fitted window by window with an extreme "
                   "learning machine (the free function of X-TFC), whose analytic derivatives feed SINDy over "
                   "quadratic monomials; the recovered system is integrated and compared with the true attractor.")
    tags = ["discovery", "chaos", "xtfc", "symbolic-regression"]
    params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "dt": 0.005, "subsample": 4, "t_end": 20.0,
              "noise": 0.01, "threshold": 0.08, "seed": 0}
    space = {"noise": (0.0, 0.05), "subsample": [1, 2, 4, 8]}

    def run(self, ctx):
        p = ctx.params
        rng = np.random.default_rng(p["seed"])
        n = int(p["t_end"] / p["dt"])
        truth = _lorenz([-8.0, 7.0, 27.0], p["sigma"], p["rho"], p["beta"], p["dt"], n)
        k = int(p["subsample"])
        t = np.arange(n)[::k] * p["dt"]
        obs = truth[::k] + rng.normal(0, p["noise"], truth[::k].shape) * truth.std(0)
        ctx.input("observations", obs)
        with ctx.stage("smooth"):
            S, D = [], []
            for j in range(3):
                s_, d_ = windowed_elm(t, obs[:, j], window=80, overlap=20, n_features=80, ridge=1e-6, seed=j)
                S.append(s_)
                D.append(d_)
            S, D = np.stack(S, 1), np.stack(D, 1)
        keep = slice(20, -20)
        from pinneapple_analysis.inverse_problems.missing_term import CandidateLibrary
        Theta, names = CandidateLibrary(poly_order=2).build(S[keep])
        names = [nm.replace("x0", "x").replace("x1", "y").replace("x2", "z") for nm in names]
        found = {}
        with ctx.stage("discover"):
            for j, var in enumerate("xyz"):
                res = sindy(Theta, D[keep, j], names, p["threshold"])
                found[var] = dict(zip(res.active_terms, map(float, res.active_coefficients), strict=True))
                ctx.log(res.equation(f"d{var}/dt"))
        ref = {"x": {"x": -p["sigma"], "y": p["sigma"]}, "y": {"x": p["rho"], "y": -1.0, "x z": -1.0},
               "z": {"z": -p["beta"], "x y": 1.0}}
        ok_structure = all(set(found[v]) == set(ref[v]) for v in "xyz")
        errs = [abs(found[v].get(term, 0.0) / c - 1) for v in "xyz" for term, c in ref[v].items()]
        ctx.output("discovered", found)
        ctx.metric("max_coefficient_rel_error", float(max(errs)))
        ctx.metric("sigma", -found["x"].get("x", float("nan")))
        ctx.metric("rho", found["y"].get("x", float("nan")))
        ctx.metric("beta", -found["z"].get("z", float("nan")))
        ctx.check("exact_structure", ok_structure, detail=str(found))
        ctx.check("coefficients_within_5pct", value=float(max(errs)), max=0.05)
        sig = -found["x"].get("x", p["sigma"])
        rho = found["y"].get("x", p["rho"])
        bet = -found["z"].get("z", p["beta"])
        rec = _lorenz(truth[0], sig, rho, bet, p["dt"], n)
        lyap_time = 1 / 0.906
        div = np.linalg.norm(rec - truth, axis=1) / np.linalg.norm(truth.std(0))
        cross = np.nonzero(div > 0.5)[0]
        horizon = float(cross[0] * p["dt"] / lyap_time) if len(cross) else float(n * p["dt"] / lyap_time)
        ctx.metric("prediction_horizon_lyapunov_times", horizon)
        plt = _plt()
        fig = plt.figure(figsize=(10, 4))
        a1 = fig.add_subplot(1, 2, 1, projection="3d")
        a1.plot(*truth.T, lw=0.4, color="k", alpha=0.6, label="true")
        a1.plot(*rec.T, lw=0.4, color="#d95f02", alpha=0.8, label="discovered law")
        a1.scatter(*obs[::10].T, s=1, color="#7570b3", label="noisy samples")
        a1.set_axis_off()
        a1.legend(frameon=False, fontsize=7)
        a2 = fig.add_subplot(1, 2, 2)
        tt = np.arange(n) * p["dt"] / lyap_time
        a2.plot(tt, truth[:, 0], "k-", lw=0.8, label="x true")
        a2.plot(tt, rec[:, 0], "--", color="#d95f02", lw=0.8, label="x from discovered law")
        a2.axvline(horizon, color="#7570b3", lw=1)
        a2.set_xlabel("time [Lyapunov times]")
        a2.legend(frameon=False, fontsize=7)
        eq = "; ".join(f"d{v}/dt = " + " ".join(f"{c:+.2f} {t_}" for t_, c in found[v].items()) for v in "xyz")
        fig.suptitle(eq, fontsize=8)
        fig.tight_layout()
        ctx.figure("lorenz_discovery", fig)


# ---------------------------------------------------------------------- re-discovery from lab results
@register
class OscillatorDiscovery(Experiment):
    name = "oscillator_discovery"
    version = "1"
    description = ("Uses the oscillator trajectories already in the lab database: for each one, SINDy over "
                   "{x, x', x², x x', x'²} finds x'' = -ω² x - 2ζω x' and recovers ζ and ω from the data alone.")
    tags = ["discovery", "reuse", "symbolic-regression"]
    params = {"threshold": 0.02, "max_runs": 200, "seed": 0}

    def run(self, ctx):
        from ..store import LabStore
        store = LabStore(ctx.dir.split("/runs/")[0])
        samples = [s for s in store.samples("oscillator", "trajectories") if s["method"] == "rk4"]
        samples = samples[: int(ctx.params["max_runs"])]
        if not samples:
            raise RuntimeError("no rk4 oscillator runs in the database: run `sweep oscillator` first")
        errs_z, errs_w, recs = [], [], []
        ds = ctx.dataset("discovered_laws", description="Recovered oscillator parameters per trajectory")
        for s in samples:
            t, x = np.asarray(s["t"], float), np.asarray(s["x"], float)
            f = elm_fit(t[: min(len(t), 800)], x[: min(len(t), 800)], n_features=200, ridge=1e-10)
            tt = t[5: min(len(t), 800) - 5]
            X0, X1, X2 = f(tt), f(tt, 1), f(tt, 2)
            Theta = np.stack([X0, X1, X0 ** 2, X0 * X1, X1 ** 2], 1)
            res = sindy(Theta, X2, ["x", "x'", "x²", "x x'", "x'²"], ctx.params["threshold"])
            cf = dict(zip(res.active_terms, map(float, res.active_coefficients), strict=True))
            w = math.sqrt(max(-cf.get("x", 0.0), 0.0))
            z = -cf.get("x'", 0.0) / (2 * w) if w > 0 else float("nan")
            errs_w.append(abs(w / s["omega"] - 1))
            errs_z.append(abs(z - s["zeta"]))
            recs.append((s["zeta"], z, s["omega"], w))
            ds.add(zeta_true=float(s["zeta"]), zeta_found=float(z), omega_true=float(s["omega"]),
                   omega_found=float(w), terms=",".join(res.active_terms))
        ctx.metric("n_trajectories", len(samples))
        ctx.metric("omega_rel_error_median", float(np.median(errs_w)))
        ctx.metric("zeta_abs_error_median", float(np.nanmedian(errs_z)))
        ctx.check("omega_recovered", value=float(np.median(errs_w)), max=0.01)
        ctx.check("zeta_recovered", value=float(np.nanmedian(errs_z)), max=0.01)
        plt = _plt()
        r = np.array(recs, float)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
        ax[0].plot(r[:, 0], r[:, 1], "o", ms=3, color="#d95f02")
        ax[0].plot([0, 1.5], [0, 1.5], "k-", lw=0.8)
        ax[0].set_xlabel("ζ used in the simulation")
        ax[0].set_ylabel("ζ discovered from data")
        ax[1].loglog(r[:, 2], r[:, 3], "o", ms=3, color="#7570b3")
        ax[1].plot([0.5, 10], [0.5, 10], "k-", lw=0.8)
        ax[1].set_xlabel("ω used")
        ax[1].set_ylabel("ω discovered")
        fig.suptitle(f"{len(samples)} trajectories from the lab database, equation re-discovered for each", fontsize=9)
        fig.tight_layout()
        ctx.figure("oscillator_rediscovery", fig)
