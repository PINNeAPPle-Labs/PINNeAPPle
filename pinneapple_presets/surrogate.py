"""Surrogate of a preset: Latin-hypercube DOE over its parameters run with the real solver, an anisotropic (ARD)
Gaussian process per output, error measured on held-out solver runs, the training box as validity domain.

The model is stored as JSON in ``pinneapple_presets/data/surrogates/<preset>.json`` (training inputs, weights and
kernel hyper-parameters: a few kB), so it loads without retraining and without scikit-learn. ``serve_spec`` gives
the description a model server needs (inputs with bounds and units, outputs, error, call).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from scipy.optimize import minimize

from .core import Preset, call_quiet, environment, lhs

__all__ = ["ARDGP", "PresetSurrogate", "build_surrogate", "load_surrogate", "surrogate_path", "serve_spec"]

DATA = os.path.join(os.path.dirname(__file__), "data", "surrogates")


class ARDGP:
    """Gaussian process, squared-exponential kernel with one length scale per input, constant variance and a
    nugget, fitted by maximum marginal likelihood (L-BFGS on log hyper-parameters, a few restarts). Inputs are
    scaled to [0, 1] and the target standardised before fitting."""

    def fit(self, X: np.ndarray, y: np.ndarray, restarts: int = 4, seed: int = 0) -> ARDGP:
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.X = X
        self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0
        z = self._z = (y - self.mu) / self.sd
        n, d = X.shape

        def nll(th):
            ls, sf2, sn2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
            K = sf2 * self._k(X, X, ls) + (sn2 + 1e-10) * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L.T, np.linalg.solve(L, z))
            return 0.5 * z @ a + np.log(np.diag(L)).sum()

        rng = np.random.default_rng(seed)
        best = None
        for r in range(restarts):
            th0 = np.r_[np.log(rng.uniform(0.15, 1.0, d) if r else np.full(d, 0.4)), 0.0, math.log(1e-4)]
            bnds = [(math.log(0.01), math.log(20.0))] * d + [(math.log(0.05), math.log(20.0)),
                                                             (math.log(1e-8), math.log(0.1))]
            res = minimize(nll, th0, method="L-BFGS-B", bounds=bnds)
            if best is None or res.fun < best.fun:
                best = res
        th = best.x
        self.ls, self.sf2, self.sn2 = np.exp(th[:d]), float(np.exp(th[d])), float(np.exp(th[d + 1]))
        self._factor()
        return self

    def _factor(self):
        K = self.sf2 * self._k(self.X, self.X, self.ls) + (self.sn2 + 1e-10) * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self._z))

    @staticmethod
    def _k(A, B, ls):
        D = (A[:, None, :] - B[None, :, :]) / ls
        return np.exp(-0.5 * (D ** 2).sum(-1))

    def predict(self, Xs) -> tuple[np.ndarray, np.ndarray]:
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self.sf2 * self._k(Xs, self.X, self.ls)
        m = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.sf2 + self.sn2 - (v ** 2).sum(0), 0.0)
        return m * self.sd + self.mu, np.sqrt(var) * self.sd

    def to_dict(self) -> dict:
        return {"X": self.X.tolist(), "z": self._z.tolist(), "ls": self.ls.tolist(), "sf2": self.sf2,
                "sn2": self.sn2, "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_dict(cls, d: dict) -> ARDGP:
        g = cls()
        g.X, g._z = np.asarray(d["X"], float), np.asarray(d["z"], float)
        g.ls, g.sf2, g.sn2, g.mu, g.sd = np.asarray(d["ls"], float), d["sf2"], d["sn2"], d["mu"], d["sd"]
        g._factor()
        return g


def _fit(X, y, **kw) -> ARDGP:
    return ARDGP().fit(X, y, **kw)


class PresetSurrogate:
    """Trained surrogate of one preset: one ARD GP per output on unit-scaled inputs."""

    def __init__(self, meta: dict, models: dict[str, ARDGP], test: dict | None = None):
        self.meta, self.models, self.test = meta, models, test or {}
        self.inputs = meta["inputs"]
        self.bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
        self.log = set(meta.get("log", ()))
        self.log_out = set(meta.get("log_outputs", ()))

    def _u(self, P: np.ndarray) -> np.ndarray:
        U = np.empty_like(P, dtype=float)
        for i, k in enumerate(self.inputs):
            a, b = self.bounds[k]
            if k in self.log:
                U[:, i] = (np.log(P[:, i]) - math.log(a)) / (math.log(b) - math.log(a))
            else:
                U[:, i] = (P[:, i] - a) / (b - a)
        return U

    def predict(self, P) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Mean and standard deviation per output (log outputs: the median exp(m) and the delta-method sd)."""
        U = self._u(np.atleast_2d(np.asarray(P, float)))
        out = {}
        for k, g in self.models.items():
            m, s = g.predict(U)
            out[k] = (np.exp(m), np.exp(m) * s) if k in self.log_out else (m, s)
        return out

    def out_of_box(self, p: dict) -> list[str]:
        return [f"{k}={p[k]} outside the surrogate training box [{a:g}, {b:g}]"
                for k, (a, b) in self.bounds.items() if not (a - 1e-12 * abs(a) <= p[k] <= b + 1e-12 * abs(b))]

    def predict_one(self, p: dict) -> tuple[dict, list[str]]:
        fixed = self.meta.get("fixed", {})
        diff = [f"{k}={p[k]} differs from the value the surrogate was trained at ({v})"
                for k, v in fixed.items() if k in p and p[k] != v]
        pr = self.predict([[p[k] for k in self.inputs]])
        out = {}
        for k, (m, s) in pr.items():
            out[k] = float(m[0])
            out[k + "_std"] = float(s[0])
        return out, self.out_of_box(p) + diff


def surrogate_path(name: str) -> str:
    return os.path.join(DATA, f"{name}.json")


def load_surrogate(name: str) -> PresetSurrogate | None:
    path = surrogate_path(name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return PresetSurrogate(d["meta"], {k: ARDGP.from_dict(v) for k, v in d["models"].items()}, d.get("test"))


def build_surrogate(preset: Preset, *, seed: int = 0, save: bool = True, tol: float = 0.05,
                    n_train: int | None = None, n_test: int | None = None) -> PresetSurrogate:
    """Run the DOE with the real solver, fit, measure the hold-out error and (``save``) store the model."""
    spec = preset.surrogate_spec
    if spec is None:
        raise ValueError(f"{preset.name} has no surrogate spec")
    n_train = n_train or spec.n_train
    n_test = n_test or spec.n_test
    bounds = {k: tuple(spec.bounds.get(k, (preset.param(k).lo, preset.param(k).hi))) for k in spec.inputs}
    log = [k in spec.log for k in spec.inputs]
    t0 = time.perf_counter()
    Xtr = lhs(n_train, [bounds[k] for k in spec.inputs], seed=seed, log=log)
    Xte = lhs(n_test, [bounds[k] for k in spec.inputs], seed=seed + 1000, log=log)

    def evaluate(X):
        Y = []
        for row in X:
            r = call_quiet(preset.run, **dict(zip(spec.inputs, map(float, row), strict=False)))
            Y.append([float(r.outputs[o]) for o in spec.outputs])
        return np.asarray(Y)

    Ytr, Yte = evaluate(Xtr), evaluate(Xte)
    t_doe = time.perf_counter() - t0
    meta = {"preset": preset.name, "inputs": list(spec.inputs), "outputs": list(spec.outputs),
            "bounds": {k: list(v) for k, v in bounds.items()}, "log": list(spec.log),
            "log_outputs": list(spec.log_outputs),
            "fixed": {q.name: q.default for q in preset.params if q.name not in spec.inputs},
            "units": {**{k: preset.param(k).unit for k in spec.inputs},
                      **{o.name: o.unit for o in preset.outputs if o.name in spec.outputs}},
            "n_train": n_train, "n_test": n_test, "seed": seed, "model": "ARD Gaussian process (squared exponential)",
            "doe": "Latin hypercube (scipy.stats.qmc), train and test drawn independently", "tol": tol}
    sur = PresetSurrogate(meta, {})
    Utr = sur._u(Xtr)
    t1 = time.perf_counter()
    if np.any(Ytr[:, [j for j, o in enumerate(spec.outputs) if o in spec.log_outputs]] <= 0):
        raise ValueError("log_outputs must be positive over the DOE")
    sur.models = {o: _fit(Utr, np.log(Ytr[:, j]) if o in spec.log_outputs else Ytr[:, j], seed=seed)
                  for j, o in enumerate(spec.outputs)}
    t_fit = time.perf_counter() - t1
    pr = sur.predict(Xte)
    per = {}
    for j, o in enumerate(spec.outputs):
        m, s = pr[o]
        ref = Yte[:, j]
        scale = np.maximum(np.abs(ref), 1e-12)
        rel = np.abs(m - ref) / scale
        ss = ((ref - ref.mean()) ** 2).sum()
        per[o] = {"max_rel_err": float(rel.max()), "mean_rel_err": float(rel.mean()),
                  "rmse": float(np.sqrt(((m - ref) ** 2).mean())),
                  "r2": float(1 - ((m - ref) ** 2).sum() / ss) if ss > 0 else float("nan"),
                  "coverage_2sigma": float((np.abs(m - ref) <= 2 * s + 1e-12).mean())}
    meta["holdout"] = {"per_output": per, "max_rel_err": max(v["max_rel_err"] for v in per.values())}
    meta["timing_s"] = {"doe_solver_runs": t_doe, "fit": t_fit}
    meta["environment"] = environment()
    meta["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sur.meta = meta
    sur.test = {"X": Xte.tolist(), "Y": Yte.tolist()}
    if save:
        os.makedirs(DATA, exist_ok=True)
        with open(surrogate_path(preset.name), "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "models": {k: g.to_dict() for k, g in sur.models.items()},
                       "test": sur.test}, fh, separators=(",", ":"))
    return sur


def serve_spec(name: str) -> dict | None:
    """What a model server needs to expose the surrogate: inputs with bounds and units, outputs, hold-out error and
    the Python call."""
    s = load_surrogate(name)
    if s is None:
        return None
    m = s.meta
    return {"name": f"preset:{name}", "kind": "surrogate", "inputs": {k: {"bounds": m["bounds"][k],
                                                                          "unit": m["units"].get(k, "")}
                                                                      for k in m["inputs"]},
            "outputs": {o: m["units"].get(o, "") for o in m["outputs"]}, "fixed": m["fixed"],
            "holdout": m["holdout"], "call": f'pinneapple_presets.load("{name}").run(fidelity="surrogate", **inputs)'}
