"""A learned propagator for near-Earth asteroids: one network for many bodies.

The network maps (osculating elements at the epoch, time) -> the deviation of the true heliocentric position from
the Kepler orbit of those elements, over one year. It is trained once on a synthetic NEO-like population integrated
with the full propagator of :mod:`.smallbodies` (Sun + planets + Moon from DE421, solar relativity), and evaluated on
held-out bodies and on real objects (Apophis, Phaethon, Ceres). Its error is reported next to the Kepler-only
error, with the speed-up and a physics check: the Tisserand parameter with respect to Jupiter, nearly conserved by the
true motion, must stay as conserved along the learned trajectories.

    model = NeuralPropagator.load()              # shipped weights (scripts/astro/train_neural_propagator.py)
    r = model.predict(body, jd)                  # heliocentric ICRF km
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .ephemeris import AU_KM, GM_SUN
from .orbits import state_from_elements
from .smallbodies import ECL_TO_ICRF, SmallBody, Trajectory

MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "neural_propagator")
EPOCH = 2461327.5               # 2026-Oct-14.0 TDB: training epoch of the synthetic population
SPAN_DAYS = 365.0


def sample_population(n: int, seed: int = 0):
    """NEO-like osculating elements: a 0.7-2.6 AU, e 0.02-0.65 (perihelion > 0.2 AU), i 0-35 deg, random angles."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a = rng.uniform(0.7, 2.6)
        e = rng.uniform(0.02, 0.65)
        if a * (1 - e) < 0.2:
            continue
        out.append(dict(a=a, e=e, i=rng.uniform(0, 35), node=rng.uniform(0, 360), peri=rng.uniform(0, 360),
                        M=rng.uniform(0, 360)))
    return out


def _body(el, epoch=EPOCH):
    return SmallBody("synthetic", epoch, el["a"], el["e"], el["i"], el["node"], el["peri"], el["M"])


def kepler_positions(body: SmallBody, jd):
    el = {"a_km": body.a_au * AU_KM, "e": body.e, "i_deg": body.i_deg, "Omega_deg": body.node_deg,
          "omega_deg": body.peri_deg, "M_deg": body.M_deg}
    r, v = state_from_elements(el, GM_SUN, (np.asarray(jd) - body.epoch_jd) * 86400.0)
    return r @ ECL_TO_ICRF.T, v @ ECL_TO_ICRF.T


def features(body: SmallBody, tau):
    """Inputs: a, e, sin/cos i, angles as sin/cos, tau = t / span, and the Kepler mean anomaly at t (sin/cos)."""
    tau = np.atleast_1d(tau)
    n = math.sqrt(GM_SUN / (body.a_au * AU_KM) ** 3) * 86400.0          # rad/day
    Mt = np.radians(body.M_deg) + n * tau * SPAN_DAYS
    base = [body.a_au, body.e, math.sin(math.radians(body.i_deg)), math.cos(math.radians(body.i_deg))]
    for ang in (body.node_deg, body.peri_deg, body.M_deg):
        base += [math.sin(math.radians(ang)), math.cos(math.radians(ang))]
    B = np.tile(base, (len(tau), 1))
    return np.column_stack([B, tau, np.sin(Mt), np.cos(Mt), tau * np.sin(Mt), tau * np.cos(Mt)])


def rtn_basis(rk, vk):
    """Radial / transverse (along-track) / normal unit vectors of the Kepler orbit, (n, 3, 3) rows R, T, N."""
    R = rk / np.linalg.norm(rk, axis=1, keepdims=True)
    N = np.cross(rk, vk)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = np.cross(N, R)
    return np.stack([R, T, N], 1)


def make_dataset(n_bodies=160, n_t=60, seed=0, cache=None):
    """Features and targets: the deviation from the Kepler orbit in its R/T/N frame (milli-AU)."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["Y"], list(z["meta"]), z["tau"]
    els = sample_population(n_bodies, seed)
    tau = np.linspace(0, 1, n_t)
    jd = EPOCH + tau * SPAN_DAYS
    X, Y, meta = [], [], []
    for el in els:
        b = _body(el)
        tr = Trajectory(b, EPOCH, EPOCH + SPAN_DAYS)
        r, _ = tr.heliocentric(jd)
        rk, vk = kepler_positions(b, jd)
        X.append(features(b, tau))
        Y.append(np.einsum("nij,nj->ni", rtn_basis(rk, vk), r - rk) / AU_KM * 1e3)
        meta.append(el)
    X, Y = np.array(X), np.array(Y)
    if cache:
        np.savez(cache, X=X, Y=Y, meta=np.array(meta, dtype=object), tau=tau)
    return X, Y, meta, tau


class NeuralPropagator:
    def __init__(self, net=None, xm=None, xs=None, ys=None, info=None):
        self.net, self.xm, self.xs, self.ys, self.info = net, xm, xs, ys, info or {}

    @staticmethod
    def _make(d_in, width=128, depth=4):
        import torch
        layers, d = [], d_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        layers.append(torch.nn.Linear(d, 3))
        return torch.nn.Sequential(*layers).double()

    @classmethod
    def train(cls, X, Y, epochs=1500, lr=2e-3, seed=0, val_frac=0.15):
        """Adam with cosine decay; the weights with the lowest loss on held-out *bodies* (not points) are kept."""
        import copy

        import torch
        torch.manual_seed(seed)
        nv = max(1, int(len(X) * val_frac))
        Xv, Yv = X[-nv:].reshape(-1, X.shape[-1]), Y[-nv:].reshape(-1, 3)
        X, Y = X[:-nv], Y[:-nv]
        Xf, Yf = X.reshape(-1, X.shape[-1]), Y.reshape(-1, 3)
        xm, xs = Xf.mean(0), Xf.std(0) + 1e-9
        ys = Yf.std(0) + 1e-9
        net = cls._make(Xf.shape[1])
        xt = torch.tensor((Xf - xm) / xs)
        yt = torch.tensor(Yf / ys)
        xv, yv = torch.tensor((Xv - xm) / xs), torch.tensor(Yv / ys)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
        bs = 4096
        best, best_state, best_ep = float("inf"), None, 0
        for ep in range(epochs):
            perm = torch.randperm(len(xt))
            for k in range(0, len(xt), bs):
                idx = perm[k:k + bs]
                loss = (net(xt[idx]) - yt[idx]).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            if ep % 10 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    vl = float((net(xv) - yv).pow(2).mean())
                if vl < best:
                    best, best_state, best_ep = vl, copy.deepcopy(net.state_dict()), ep
        net.load_state_dict(best_state)
        return cls(net, xm, xs, ys, {"epochs": epochs, "best_epoch": best_ep, "train_points": int(len(xt)),
                                     "val_loss": best, "train_bodies": int(len(X)), "val_bodies": nv})

    def correction_km(self, body: SmallBody, jd):
        import torch
        tau = (np.atleast_1d(jd) - body.epoch_jd) / SPAN_DAYS
        X = (features(body, tau) - self.xm) / self.xs
        with torch.no_grad():
            y = self.net(torch.tensor(X)).numpy() * self.ys
        return y / 1e3 * AU_KM

    def predict(self, body: SmallBody, jd):
        rk, vk = kepler_positions(body, jd)
        rtn = self.correction_km(body, jd)
        return rk + np.einsum("nji,nj->ni", rtn_basis(rk, vk), rtn)

    def save(self, folder=MODEL_DIR):
        import torch
        os.makedirs(folder, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(folder, "net.pt"))
        json.dump({"xm": self.xm.tolist(), "xs": self.xs.tolist(), "ys": self.ys.tolist(), "info": self.info},
                  open(os.path.join(folder, "scaling.json"), "w"), indent=1)

    @classmethod
    def load(cls, folder=MODEL_DIR):
        import torch
        sc = json.load(open(os.path.join(folder, "scaling.json")))
        xm = np.array(sc["xm"])
        net = cls._make(len(xm))
        net.load_state_dict(torch.load(os.path.join(folder, "net.pt"), weights_only=True))
        net.eval()
        return cls(net, xm, np.array(sc["xs"]), np.array(sc["ys"]), sc["info"])


def tisserand(r_km, v_km_s, a_jup_au=5.2026):
    """Tisserand parameter w.r.t. Jupiter from heliocentric states (ICRF axes; inclination to the ecliptic)."""
    from .orbits import elements_from_state
    out = []
    R = (ECL_TO_ICRF.T @ np.asarray(r_km).T).T
    V = (ECL_TO_ICRF.T @ np.asarray(v_km_s).T).T
    for r, v in zip(R, V, strict=True):
        el = elements_from_state(r, v, GM_SUN)
        a = el["a_au"]
        out.append(a_jup_au / a + 2 * math.cos(math.radians(el["i_deg"])) * math.sqrt(max(a / a_jup_au * (1 - el["e"] ** 2), 0)))
    return np.array(out)


def evaluate(model: NeuralPropagator, bodies, n_t=60):
    """Errors of the learned propagator and of Kepler alone against the full integrator, per body, plus timing."""
    rows = []
    for b in bodies:
        jd = b.epoch_jd + np.linspace(0, SPAN_DAYS, n_t)
        t0 = time.time()
        tr = Trajectory(b, b.epoch_jd, b.epoch_jd + SPAN_DAYS)
        r, v = tr.heliocentric(jd)
        t_int = time.time() - t0
        t0 = time.time()
        rn = model.predict(b, jd)
        t_nn = time.time() - t0
        rk, _ = kepler_positions(b, jd)
        en = np.linalg.norm(rn - r, axis=1)
        ek = np.linalg.norm(rk - r, axis=1)
        rows.append({"name": b.name, "max_err_km": float(en.max()), "kepler_max_err_km": float(ek.max()),
                     "max_angle_arcsec": float(np.degrees(np.max(en / np.linalg.norm(r, axis=1))) * 3600),
                     "speedup": float(t_int / max(t_nn, 1e-6)), "t_days": (jd - b.epoch_jd).tolist(),
                     "err_km": en.tolist(), "kepler_err_km": ek.tolist()})
    return rows


def trust_report(folder=MODEL_DIR, real: dict | None = None) -> dict:
    """Verdict of the shipped network from its held-out evaluation: to be useful it must beat Kepler by 5x on the
    median of the per-body maximum error over unseen bodies (and not be worse than Kepler at the 90th percentile)."""
    info = json.load(open(os.path.join(folder, "scaling.json")))["info"]
    kep = info["test_kepler_max_err_km_median"]
    checks = [
        {"id": "neo_median", "name": "Unseen synthetic NEOs: median of the max error over 1 yr", "value": info["test_max_err_km_median"],
         "unit": "km", "threshold": kep / 5, "status": "pass" if info["test_max_err_km_median"] <= kep / 5 else "fail"},
        {"id": "neo_p90", "name": "Unseen synthetic NEOs: 90th percentile of the max error", "value": info["test_max_err_km_p90"],
         "unit": "km", "threshold": kep, "status": "pass" if info["test_max_err_km_p90"] <= kep else "warn"},
    ]
    if real:
        checks.append({"id": "neo_real", "args": {"name": real["name"]}, "name": f"{real['name']} (real object)", "value": real["max_err_km"], "unit": "km",
                       "threshold": real["kepler_max_err_km"],
                       "status": "pass" if real["max_err_km"] <= real["kepler_max_err_km"] else "warn"})
    status = "trustworthy" if all(c["status"] == "pass" for c in checks) else \
        "not_trustworthy" if any(c["status"] == "fail" for c in checks) else "use_with_caution"
    return {"info": info, "checks": checks, "verdict": {"status": status},
            "note": "Over a year, deviations from Kepler for near-Earth asteroids are dominated by planetary encounters, "
                    "which one network trained on ~1,000 orbits does not generalise; the gate rejects it and the app "
                    "uses the integrator (and per-body PINNs) instead."}
