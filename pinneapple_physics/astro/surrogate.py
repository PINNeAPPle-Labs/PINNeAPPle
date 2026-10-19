"""A physics-informed neural surrogate for orbit propagation, checked against the classical integrator.

The network maps time in a window [t0, t0 + T] to a body's heliocentric position. It is trained on

* a handful of anchor states from the classical N-body integration (the "data" term),
* the two-body equation of motion r'' = -mu r / |r|^3 at collocation points (the physics residual, via autograd),
* the initial position and velocity.

It is then judged like every surrogate in PINNeAPPle: error against the reference integrator on dense held-out
times, split-conformal uncertainty bands (calibrated on held-out integrator states, guaranteed marginal coverage),
and physics checks: conservation of specific energy and angular momentum along the learned trajectory and the
equation-of-motion residual. A verdict (trustworthy / use_with_caution / not_trustworthy) follows from those
numbers with fixed, published thresholds.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from .ephemeris import AU_KM, GM, GM_SUN, default_ephemeris
from .nbody import SOLAR_SYSTEM, initial_state, integrate

# acceptance thresholds for the verdict (heliocentric angle; 1" is the accuracy budget of the observatory)
THRESHOLDS = {"max_angle_arcsec": 1.0, "energy_rel": 0.25, "angmom_rel": 0.25, "coverage_min": 0.85,
              "residual_rms": 0.1}


def reference_trajectory(body: str, jd_tdb: float, days: float, n: int = 400, eph=None):
    """Heliocentric positions (AU) and velocities (AU/day) of every body from the Yoshida-4 N-body integration."""
    eph = eph or default_ephemeris()
    r0, v0, mu = initial_state(jd_tdb, SOLAR_SYSTEM, eph)
    out = integrate(r0, v0, mu, days * 86400.0, 3600.0 * 6, "yoshida4", n_out=n)
    s = SOLAR_SYSTEM.index("sun")
    r = (out["r"] - out["r"][:, s:s + 1]) / AU_KM
    v = (out["v"] - out["v"][:, s:s + 1]) / AU_KM * 86400.0
    return out["t_s"] / 86400.0, r, v, out


def _smallbody_reference(body, jd_tdb: float, days: float, eph=None, n: int = 400):
    """Planets (heliocentric, from DE421) plus the small body (from :class:`.smallbodies.Trajectory`), in the same
    layout as :func:`reference_trajectory` with the small body appended as the last column."""
    from .ephemeris import default_ephemeris
    from .smallbodies import Trajectory
    eph = eph or default_ephemeris()
    t = np.linspace(0.0, days, n)
    jd = jd_tdb + t
    R = np.zeros((n, len(SOLAR_SYSTEM) + 1, 3))
    V = np.zeros_like(R)
    sp, sv = eph.barycentric("sun", jd)
    for j, b in enumerate(SOLAR_SYSTEM):
        p, v = eph.barycentric(b, jd)
        R[:, j], V[:, j] = (p - sp) / AU_KM, (v - sv) / AU_KM * 86400.0
    tr = Trajectory(body, min(jd_tdb, body.epoch_jd), max(jd_tdb + days, body.epoch_jd), eph)
    r, v = tr.heliocentric(jd)
    R[:, -1], V[:, -1] = r / AU_KM, v / AU_KM * 86400.0
    info = {"rel_energy_error": np.zeros(1), "rel_angmom_error": np.zeros(1), "n_force_evals": 0,
            "method": "DOP853 test particle in the DE421 field (Sun, planets, Moon) + solar 1PN"}
    return t, R, V, info


def _kepler_track(r0, v0, mu, t):
    """Two-body positions and accelerations (AU, AU/day^2) from the initial state, analytic."""
    from .orbits import elements_from_state, state_from_elements
    el = elements_from_state(r0 * AU_KM, v0 * AU_KM / 86400.0, mu * AU_KM**3 / 86400.0**2)
    p, _ = state_from_elements(el, mu * AU_KM**3 / 86400.0**2, np.asarray(t) * 86400.0)
    p = p / AU_KM
    a = -mu * p / np.linalg.norm(p, axis=1, keepdims=True) ** 3
    return p, a


# torch.func's forward-mode AD keeps its dual levels in global state: two trainings in different threads (a web
# server's worker pool) corrupt each other's levels, so trainings run one at a time.
_TRAIN_LOCK = threading.Lock()


def train_orbit_pinn(body: str = "mars", jd_tdb: float = 2461327.0, days: float = 365.0, **kwargs) -> dict:
    """Train the PINN orbit surrogate (thread-safe; see ``_train_orbit_pinn``)."""
    with _TRAIN_LOCK:
        return _train_orbit_pinn(body, jd_tdb, days, **kwargs)


def _train_orbit_pinn(body: str = "mars", jd_tdb: float = 2461327.0, days: float = 365.0, *, n_anchor: int = 8,
                      n_colloc: int = 400, iters: int = 1500, width: int = 48, depth: int = 3, seed: int = 0,
                      alpha: float = 0.1, eph=None) -> dict:
    """Train the PINN orbit surrogate and return its evaluation report (JSON-serialisable).

    Ansatz: r(t) = r_Kepler(t) + s tau^2 g_theta(tau), tau = t / T. The Kepler term is the exact two-body solution
    from the initial state, the tau^2 factor makes the initial position and velocity hard constraints, and the
    network g only learns the perturbation by the other planets. Its loss is the residual of the full equation of
    motion  r'' = -mu r/|r|^3 + sum_j mu_j [(r_j - r)/|r_j - r|^3 - r_j/|r_j|^3]  (heliocentric, with the indirect
    term; the perturbers' positions come from the reference integration) plus a few anchor states."""
    import torch
    from torch.func import jacfwd, vmap

    from pinneapple_analysis.uncertainty.conformal import ConformalPredictor

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    started = time.time()
    if isinstance(body, str):
        t_ref, R_all, V_all, nb = reference_trajectory(body, jd_tdb, days, eph=eph)
        k = SOLAR_SYSTEM.index(body)
        name = body
    else:                       # a small body (asteroid, comet): reference = the high-accuracy propagator
        t_ref, R_all, V_all, nb = _smallbody_reference(body, jd_tdb, days, eph)
        k = len(SOLAR_SYSTEM)
        name = body.name
    r_ref, v_ref = R_all[:, k], V_all[:, k]
    mu = (GM_SUN + (GM.get(body, 0.0) if isinstance(body, str) else 0.0)) / AU_KM**3 * 86400.0**2   # AU^3/day^2
    others = [j for j, b in enumerate(SOLAR_SYSTEM) if b not in ("sun", body)]
    mu_o = np.array([GM[SOLAR_SYSTEM[j]] for j in others]) / AU_KM**3 * 86400.0**2
    T = float(days)

    idx = np.arange(len(t_ref))
    anchors = np.unique(np.linspace(0, len(t_ref) - 1, n_anchor + 1).round().astype(int))[1:]
    rest = np.setdiff1d(idx, np.r_[anchors, 0])
    rng.shuffle(rest)
    cal, test = np.sort(rest[: len(rest) // 2]), np.sort(rest[len(rest) // 2:])

    rk_ref, ak_ref = _kepler_track(r_ref[0], v_ref[0], mu, t_ref)
    s = float(np.max(np.linalg.norm(r_ref - rk_ref, axis=1))) or 1e-6        # perturbation scale (AU)

    # collocation pool (fixed), with Kepler terms and perturber positions precomputed
    t_c = np.sort(rng.uniform(0, T, n_colloc))
    rk_c, ak_c = _kepler_track(r_ref[0], v_ref[0], mu, t_c)
    Ro_c = np.stack([np.stack([np.interp(t_c, t_ref, R_all[:, j, d]) for d in range(3)], -1) for j in others], 1)

    n_freq = 6

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers, d_in = [], 1 + 2 * n_freq
            for _ in range(depth):
                layers += [torch.nn.Linear(d_in, width), torch.nn.Tanh()]
                d_in = width
            layers.append(torch.nn.Linear(d_in, 3))
            self.mlp = torch.nn.Sequential(*layers)
            self.register_buffer("kf", torch.arange(1, n_freq + 1, dtype=torch.float64) * math.pi)

        def forward(self, tau):                              # tau: scalar tensor -> (3,) perturbation in AU
            f = tau * self.kf
            x = torch.cat([tau[None], torch.sin(f), torch.cos(f)])
            return s * tau**2 * self.mlp(x)

    net = Net().double()
    tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    g = lambda tau: net(tau)  # noqa: E731
    d2 = vmap(jacfwd(jacfwd(g)))
    gv = vmap(g)

    tau_c, rk_c_t, ak_c_t, Ro_t, mu_o_t = tt(t_c / T), tt(rk_c), tt(ak_c), tt(Ro_c), tt(mu_o)
    a_scale = mu / float(np.mean(np.linalg.norm(r_ref, axis=1))) ** 2 * 1e-4   # ~ size of the perturbing accel.

    def accel(r, Ro):
        rn = torch.linalg.norm(r, dim=-1, keepdim=True)
        d = Ro - r[:, None, :]
        dn = torch.linalg.norm(d, dim=-1, keepdim=True)
        Ron = torch.linalg.norm(Ro, dim=-1, keepdim=True)
        pert = (mu_o_t[None, :, None] * (d / dn**3 - Ro / Ron**3)).sum(1)
        return -mu * r / rn**3, pert

    def residual(tau, rk, ak, Ro):
        dr = gv(tau)
        r = rk + dr
        a_net = d2(tau) / T**2
        a_kep, pert = accel(r, Ro)
        return ((ak + a_net) - (a_kep + pert)) / a_scale

    tau_a, r_a = tt(t_ref[anchors] / T), tt(r_ref[anchors] - rk_ref[anchors])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=2e-5)
    history = []
    for it in range(iters):
        l_phys = residual(tau_c, rk_c_t, ak_c_t, Ro_t).pow(2).mean()
        l_data = ((gv(tau_a) - r_a) / s).pow(2).mean()
        loss = l_phys + 10 * l_data
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == iters - 1:
            history.append({"iter": it, "loss": loss.item(), "data": l_data.item(), "physics": l_phys.item()})

    # ---------------------------------------------------------------------------------------- evaluation
    tau_all = tt(t_ref / T)
    with torch.no_grad():
        dr_all = gv(tau_all).numpy()
    v_net = vmap(jacfwd(g))(tau_all).detach().numpy() / T
    from .orbits import elements_from_state, state_from_elements
    el0 = elements_from_state(r_ref[0] * AU_KM, v_ref[0] * AU_KM / 86400.0, mu * AU_KM**3 / 86400.0**2)
    _, vk = state_from_elements(el0, mu * AU_KM**3 / 86400.0**2, t_ref * 86400.0)
    r_p = rk_ref + dr_all
    v_p = vk / AU_KM * 86400.0 + v_net
    Ro_ref = R_all[:, others]
    res = residual(tau_all, tt(rk_ref), tt(ak_ref), tt(Ro_ref)).detach().numpy()
    err_km = np.linalg.norm(r_p - r_ref, axis=1) * AU_KM
    kep_err_km = np.linalg.norm(rk_ref - r_ref, axis=1) * AU_KM
    ang_arcsec = np.degrees(np.arccos(np.clip(np.sum(r_p * r_ref, 1) / (np.linalg.norm(r_p, axis=1)
                                                                       * np.linalg.norm(r_ref, axis=1)), -1, 1))) * 3600
    eps = 0.5 * np.sum(v_p**2, 1) - mu / np.linalg.norm(r_p, axis=1)
    eps_ref = 0.5 * np.sum(v_ref**2, 1) - mu / np.linalg.norm(r_ref, axis=1)
    h = np.linalg.norm(np.cross(r_p, v_p), axis=1)
    h_ref = np.linalg.norm(np.cross(r_ref, v_ref), axis=1)
    # conservation: in the perturbed problem the two-body energy and angular momentum are not constant; they vary
    # by the planetary perturbations (the signal the network has to learn). Score the PINN's error on them against
    # the size of that variation in the reference integration.
    d_eps = float(np.max(np.abs(eps - eps_ref)) / max(np.ptp(eps_ref), 1e-30))
    d_h = float(np.max(np.abs(h - h_ref)) / max(np.ptp(h_ref), 1e-30))

    # split conformal on held-out integrator states (component-wise, finite-sample quantile)
    cp = ConformalPredictor(lambda x: gv(x[:, 0]).detach(), alpha=alpha)
    cp.calibrate(tau_all[cal][:, None], tt((r_ref - rk_ref)[cal]))
    _, lo, hi = cp.predict(tau_all[:, None])
    half = ((hi - lo) / 2).numpy()
    band_km = np.linalg.norm(half, axis=1) * AU_KM
    target = r_ref - rk_ref
    covered = np.all((target[test] >= lo.numpy()[test]) & (target[test] <= hi.numpy()[test]), axis=1)

    th = THRESHOLDS
    checks = [
        {"id": "angle_vs_integrator", "name": "Heliocentric angular error vs integrator (test)", "value": float(ang_arcsec[test].max()),
         "unit": "arcsec", "threshold": th["max_angle_arcsec"],
         "status": "pass" if ang_arcsec[test].max() <= th["max_angle_arcsec"] else "fail"},
        {"id": "energy_vs_signal", "name": "Orbital energy error / perturbation signal", "value": d_eps, "unit": "ratio",
         "threshold": th["energy_rel"], "status": "pass" if d_eps <= th["energy_rel"] else "fail"},
        {"id": "angmom_vs_signal", "name": "Angular momentum error / perturbation signal", "value": d_h, "unit": "ratio",
         "threshold": th["angmom_rel"], "status": "pass" if d_h <= th["angmom_rel"] else "fail"},
        {"id": "conformal_coverage", "name": "Conformal coverage on test points", "value": float(covered.mean()), "unit": "fraction",
         "threshold": th["coverage_min"], "status": "pass" if covered.mean() >= th["coverage_min"] else "warn"},
        {"id": "eom_residual", "name": "Equation-of-motion residual (RMS / perturbing accel.)",
         "value": float(np.sqrt((res**2).mean())), "unit": "-", "threshold": th["residual_rms"],
         "status": "pass" if np.sqrt((res**2).mean()) <= th["residual_rms"] else "warn"},
        {"id": "gain_over_kepler", "name": "Improvement over the Kepler two-body solution", "value": float(kep_err_km[test].max()
                                                                                   / max(err_km[test].max(), 1e-9)),
         "unit": "x", "threshold": 10.0, "status": "pass" if kep_err_km[test].max() >= 10 * err_km[test].max()
         else "warn"},
    ]
    n_fail = sum(c["status"] == "fail" for c in checks)
    n_warn = sum(c["status"] == "warn" for c in checks)
    status = "trustworthy" if n_fail == n_warn == 0 else "use_with_caution" if n_fail == 0 else "not_trustworthy"
    return {
        "body": name, "jd_tdb": jd_tdb, "days": days, "train_seconds": time.time() - started,
        "model": {"kind": "PINN: Kepler + tau^2 x Fourier-feature MLP", "width": width, "depth": depth,
                  "fourier_modes": n_freq, "params": int(sum(p.numel() for p in net.parameters())),
                  "anchors": int(len(anchors)), "collocation": n_colloc, "iterations": iters},
        "reference": {"method": nb.get("method", "Yoshida 4th-order symplectic N-body (Sun, 8 planets, Moon), dt = 6 h"),
                      "energy_drift": float(np.abs(nb["rel_energy_error"]).max()),
                      "angmom_drift": float(nb["rel_angmom_error"].max()),
                      "force_evaluations": nb["n_force_evals"]},
        "t_days": t_ref.tolist(), "r_ref_au": r_ref.tolist(), "r_pinn_au": r_p.tolist(), "r_kepler_au": rk_ref.tolist(),
        "error_km": err_km.tolist(), "kepler_error_km": kep_err_km.tolist(), "angle_err_arcsec": ang_arcsec.tolist(),
        "band_km": band_km.tolist(), "energy_pinn": eps.tolist(), "energy_ref": eps_ref.tolist(),
        "angmom_pinn": h.tolist(), "angmom_ref": h_ref.tolist(), "residual_rms_t": np.sqrt((res**2).mean(1)).tolist(),
        "anchors": anchors.tolist(), "calibration": cal.tolist(), "test": test.tolist(), "history": history,
        "checks": checks,
        "conservation": {"energy_variation_ref": float(np.ptp(eps_ref) / abs(eps_ref.mean())),
                         "angmom_variation_ref": float(np.ptp(h_ref) / h_ref.mean())},
        "summary": {"max_error_km_test": float(err_km[test].max()), "median_error_km_test": float(np.median(err_km[test])),
                    "kepler_max_error_km": float(kep_err_km.max()), "max_angle_arcsec_test": float(ang_arcsec[test].max()),
                    "coverage": float(covered.mean()), "alpha": alpha, "band_km_median": float(np.median(band_km))},
        "verdict": {"status": status, "reasons": [f"{c['name']}: {c['status']}" for c in checks
                                                   if c["status"] != "pass"]},
    }
