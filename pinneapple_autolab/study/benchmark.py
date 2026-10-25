"""Surrogate families compared on one protocol, with calibrated uncertainty and physics checks, and a written rule
that picks one.

    board = benchmark(ds, targets=("CD", "CL"), field="cp", budget_s=30.0)
    board.table()                        # CV, test and extrapolation errors, coverage, fit time per family
    board.select()                       # the chosen family per target and the rule that chose it

Protocol (the same for every family, so the comparison is at equal data and equal budget):

1. Inputs scaled to the unit cube of the training region (:attr:`StudyDataset.bounds`).
2. K-fold cross-validation on train + val (the test and extrapolation splits are not touched): the CV error is the
   selection score, and the CV residuals over each fold's predicted std are the nonconformity scores of a
   cross-conformal interval (Vovk 2015): ``mean +- q * std``, ``q`` the finite-sample (1 - alpha) quantile.
3. The final model of each family is fitted on train + val; its test and extrapolation errors, and the coverage of
   its conformal and its raw Gaussian (1.645 std) 90 % intervals, are measured once, after selection.
4. Training budget: one-shot fits (GP, MLP ensemble, response surface, POD-RBF) report the time they used;
   iterative ones (DeepONet) get exactly ``budget_s`` of wall clock per fit (``ComputeBudget``), split over their
   ensemble members.

Scalar families: ``gp`` (Matern 5/2 ARD), ``mlp_ensemble`` (5 bootstrap MLPs), ``quadratic`` (response surface,
the baseline to beat). Field families: ``pod_gp``, ``pod_mlp``, ``pod_rbf`` (POD + per-mode GP / MLP ensemble /
thin-plate RBF) and ``deeponet`` (an ensemble of DeepONets, branch = design, trunk = surface point, trained by
:mod:`pinneapple_analysis.surrogate.methods`).

Physics checks are callables ``check(predict) -> Check`` evaluated on each family's final model; a family that
fails one is not eligible (:meth:`Board.select`).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..surrogates import FieldSurrogate, GPModel, MLPEnsemble, QuadraticRSM

__all__ = ["SCALAR_FAMILIES", "FIELD_FAMILIES", "PODRBF", "DeepONetEnsemble", "make_scalar", "make_field",
           "benchmark", "Board", "SELECTION_RULE", "conformal_q"]

SCALAR_FAMILIES = ("gp", "mlp_ensemble", "quadratic")
FIELD_FAMILIES = ("pod_gp", "pod_mlp", "pod_rbf", "deeponet")
SELECTION_RULE = ("Eligible: every physics check on the family's predictions passes. Among the eligible families, "
                  "the lowest cross-validated relative RMSE of the target wins; families within 5 % of the best CV "
                  "error are tied and the one with the narrower cross-conformal 90 % interval wins. The test and "
                  "extrapolation splits never enter the choice.")


class PODRBF:
    """POD + thin-plate-spline RBF interpolation of each retained mode's coefficient (smoothing ``smoothing``).
    The std is homoscedastic: each mode's leave-one-out RMSE, propagated through the modes."""
    name = "POD + RBF (thin plate)"

    def __init__(self, energy: float = 0.995, max_modes: int = 16, smoothing: float = 1e-6):
        self.energy, self.max_modes, self.smoothing = energy, max_modes, smoothing

    def fit(self, U: np.ndarray, S: np.ndarray) -> PODRBF:
        from scipy.interpolate import RBFInterpolator
        t0 = time.perf_counter()
        self.mean = S.mean(0)
        A = S - self.mean
        _, sv, Vt = np.linalg.svd(A, full_matrices=False)
        e = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-30)
        r = int(min(np.searchsorted(e, self.energy) + 1, self.max_modes, max(len(S) - 1, 1)))
        self.n_modes, self.modes = r, Vt[:r]
        C = A @ self.modes.T
        self.rbf = RBFInterpolator(U, C, kernel="thin_plate_spline", smoothing=self.smoothing)
        loo = np.zeros_like(C)
        for i in range(len(U)):                                     # leave-one-out residual of every coefficient
            m = np.arange(len(U)) != i
            loo[i] = RBFInterpolator(U[m], C[m], kernel="thin_plate_spline", smoothing=self.smoothing)(U[i:i + 1])[0]
        self.mode_sd = np.sqrt(np.mean((loo - C) ** 2, 0))
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        U = np.atleast_2d(U)
        mean = self.mean + self.rbf(U) @ self.modes
        std = np.sqrt((self.mode_sd ** 2) @ (self.modes ** 2))
        return mean, np.broadcast_to(std, mean.shape).copy()


class DeepONetEnsemble:
    """``n`` DeepONets (different seeds) trained by :func:`pinneapple_analysis.surrogate.methods.fit_candidate`
    within ``budget_s`` in total; mean and spread of the members. ``coords``: (n_points, 3) surface points."""
    name = "DeepONet ensemble"

    def __init__(self, coords: np.ndarray, budget_s: float = 30.0, n: int = 3, seed: int = 0,
                 options: Mapping[str, Any] | None = None):
        self.coords, self.budget_s, self.n, self.seed = np.asarray(coords, float), budget_s, n, seed
        self.options = dict(options or {"hidden": 64, "modes": 32, "depth": 3})

    def fit(self, U: np.ndarray, S: np.ndarray, U_val: np.ndarray | None = None, S_val: np.ndarray | None = None):
        import torch

        from pinneapple_analysis.cost.equal_budget import ComputeBudget
        from pinneapple_analysis.surrogate.methods import TrainContext, fit_candidate
        t0 = time.perf_counter()
        Uv = U if U_val is None else U_val
        Sv = S if S_val is None else S_val
        self.members = []
        old = torch.get_num_threads()
        torch.set_num_threads(1)                       # small nets: one thread is fastest (and fair to the others)
        try:
            for k in range(self.n):
                ctx = TrainContext(U, S, Uv, Sv, ComputeBudget(wall_s=self.budget_s / self.n), self.seed + k,
                                   (S.shape[1],), self.coords, None, dict(self.options))
                self.members.append(fit_candidate("deeponet", ctx)[0])
        finally:
            torch.set_num_threads(old)
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = np.stack([m.predict(np.atleast_2d(U)) for m in self.members])
        return P.mean(0), P.std(0) + 1e-9


def make_scalar(family: str, seed: int = 0):
    return {"gp": lambda: GPModel(restarts=2, seed=seed), "mlp_ensemble": lambda: MLPEnsemble(seed=seed),
            "quadratic": lambda: QuadraticRSM()}[family]()


def make_field(family: str, coords: np.ndarray | None = None, budget_s: float = 30.0, seed: int = 0):
    if family == "pod_gp":
        return FieldSurrogate("gp", energy=0.995, max_modes=16)
    if family == "pod_mlp":
        return FieldSurrogate("mlp_ensemble", energy=0.995, max_modes=16)
    if family == "pod_rbf":
        return PODRBF()
    if family == "deeponet":
        return DeepONetEnsemble(coords, budget_s=budget_s, seed=seed)
    raise ValueError(f"unknown field family {family!r}")


def conformal_q(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample split/cross-conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score (inf if n is
    too small for that level)."""
    s = np.sort(np.asarray(scores, float).ravel())
    k = int(math.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def _scalar_metrics(y, m, s, q, scale) -> dict[str, float]:
    e = m - y
    out = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(e ** 2))) if len(y) else float("nan"),
           "max_abs": float(np.max(np.abs(e))) if len(y) else float("nan")}
    out["rel_rmse"] = out["rmse"] / scale
    if len(y):
        out["coverage90_conformal"] = float(np.mean(np.abs(e) <= q * s)) if np.isfinite(q) else float("nan")
        out["coverage90_raw"] = float(np.mean(np.abs(e) <= 1.645 * s))
        out["half_width90"] = float(np.mean(q * s)) if np.isfinite(q) else float("inf")
        ss = float(np.sum((y - y.mean()) ** 2))
        out["r2"] = float(1 - np.sum(e ** 2) / ss) if ss > 0 else float("nan")
    return out


def _field_metrics(Y, M, S, q, regions) -> dict[str, Any]:
    if not len(Y):
        return {"n": 0}
    e = M - Y
    rel = np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(Y, axis=1), 1e-12)
    out = {"n": int(len(Y)), "rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max()),
           "rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs": float(np.max(np.abs(e))),
           "coverage90_conformal": float(np.mean(np.abs(e) <= q * S)) if np.isfinite(q) else float("nan"),
           "coverage90_raw": float(np.mean(np.abs(e) <= 1.645 * S))}
    out["regions"] = {k: float(np.sqrt(np.mean(e[:, np.asarray(v, bool)] ** 2))) for k, v in (regions or {}).items()
                      if np.asarray(v, bool).any()}
    return out


@dataclass
class Board:
    targets: list[str]
    field: str | None
    rows: dict[str, dict[str, Any]]                    # "<target>/<family>" -> metrics
    models: dict[str, Any] = field(default_factory=dict, repr=False)
    q: dict[str, float] = field(default_factory=dict)
    physics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)

    def families(self, target: str) -> list[str]:
        return [k.split("/", 1)[1] for k in self.rows if k.split("/", 1)[0] == target]

    def select(self, rule: str = SELECTION_RULE) -> dict[str, Any]:
        """The chosen family per target (``None`` when no family is eligible), by :data:`SELECTION_RULE`."""
        out: dict[str, Any] = {"rule": rule, "choice": {}, "reasons": {}}
        for t in self.targets + ([self.field] if self.field else []):
            fams = self.families(t)
            elig = [f for f in fams if all(c["passed"] for c in self.physics.get(f"{t}/{f}", []))]
            key = "cv_rel_l2" if t == self.field else "cv_rel_rmse"
            ranked = sorted(elig, key=lambda f: self.rows[f"{t}/{f}"][key])
            if not ranked:
                out["choice"][t] = None
                out["reasons"][t] = "no family passes every physics check"
                continue
            best = self.rows[f"{t}/{ranked[0]}"][key]
            tied = [f for f in ranked if self.rows[f"{t}/{f}"][key] <= 1.05 * best]
            win = min(tied, key=lambda f: self.rows[f"{t}/{f}"].get("cv_half_width90", np.inf))
            out["choice"][t] = win
            cv = ", ".join("{} {:.4f}".format(f, self.rows[t + "/" + f][key]) for f in ranked)
            out["reasons"][t] = (f"eligible {elig}; CV {key} {cv}"
                                 + (f"; tied within 5 %: {tied}, narrower interval wins" if len(tied) > 1 else ""))
        return out

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for k, r in self.rows.items():
            t, f = k.split("/", 1)
            row = {"target": t, "family": f, **{kk: v for kk, v in r.items() if not isinstance(v, dict)}}
            for split in ("test", "extrapolation"):
                for kk, v in (r.get(split) or {}).items():
                    if not isinstance(v, dict):
                        row[f"{split}_{kk}"] = v
            row["physics_passed"] = all(c["passed"] for c in self.physics.get(k, []))
            rows.append(row)
        return rows

    def predictor(self, target: str, family: str | None = None) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """``predict(U) -> (mean, conformal 90 % half width)`` of a final model (inputs in the unit cube)."""
        family = family or self.select()["choice"][target]
        m, q = self.models[f"{target}/{family}"], self.q[f"{target}/{family}"]

        def f(U):
            mu, s = m.predict(np.atleast_2d(U))
            return mu, q * s
        return f

    def to_dict(self) -> dict[str, Any]:
        return {"targets": self.targets, "field": self.field, "rows": self.rows, "q": self.q,
                "physics": self.physics, "protocol": self.protocol, "selection": self.select()}


def benchmark(ds, targets: Sequence[str], field: str | None = None, *, scalar_families: Sequence[str] = SCALAR_FAMILIES,
              field_families: Sequence[str] = FIELD_FAMILIES, k: int = 5, alpha: float = 0.1, budget_s: float = 30.0,
              seed: int = 0, physics: Mapping[str, Sequence[Callable]] | None = None,
              log: Callable[[str], None] = print) -> Board:
    """Run the protocol of the module docstring on a :class:`~pinneapple_autolab.study.dataset.StudyDataset`.
    ``physics``: ``{target: [check(predict) -> Check-like], ...}``; each check gets ``predict(U) -> (mean, std)``
    of the family's final model."""
    fit_idx = np.r_[ds.idx("train"), ds.idx("val")]
    te, ex = ds.idx("test"), ds.idx("extrapolation")
    U = ds.unit_x()
    folds = _folds(len(fit_idx), k, seed)
    board = Board(list(targets), field, {}, protocol={"k_folds": k, "alpha": alpha, "budget_s": budget_s,
                                                       "seed": seed, "n_fit": int(len(fit_idx)), "n_test": int(len(te)),
                                                       "n_extrapolation": int(len(ex)), "rule": SELECTION_RULE})
    jobs: list[tuple[str, str, Callable[[], Any], np.ndarray]] = []
    for t in targets:
        for f in scalar_families:
            jobs.append((t, f, lambda f=f: make_scalar(f, seed), ds.y(t)))
    if field:
        for f in field_families:
            jobs.append((field, f, lambda f=f: make_field(f, ds.points, budget_s, seed), ds.fields[field]))
    for t, f, make, Y in jobs:
        is_field = Y.ndim == 2
        t0 = time.perf_counter()
        res, sds = [], []
        for fold in folds:
            tr = np.setdiff1d(np.arange(len(fit_idx)), fold)
            m = _fit(make(), U[fit_idx[tr]], Y[fit_idx[tr]])
            mu, s = m.predict(U[fit_idx[fold]])
            res.append(mu - Y[fit_idx[fold]])
            sds.append(np.maximum(s, 1e-12))
        R, Sd = np.concatenate(res), np.concatenate(sds)
        cv_s = time.perf_counter() - t0
        if is_field:
            scores = np.abs(R) / Sd                       # one score per point and case (pointwise coverage)
            q = conformal_q(scores, alpha)
            yy = Y[fit_idx][np.concatenate(folds)]
            rel = np.linalg.norm(R, axis=1) / np.maximum(np.linalg.norm(yy, axis=1), 1e-12)
            row = {"cv_rel_l2": float(rel.mean()), "cv_rmse": float(np.sqrt(np.mean(R ** 2))),
                   "cv_half_width90": float(np.mean(q * Sd)) if np.isfinite(q) else float("inf")}
        else:
            q = conformal_q(np.abs(R) / Sd, alpha)
            scale = float(np.std(Y[fit_idx])) or 1.0
            row = {"cv_rmse": float(np.sqrt(np.mean(R ** 2))), "cv_rel_rmse": float(np.sqrt(np.mean(R ** 2)) / scale),
                   "cv_coverage90_raw": float(np.mean(np.abs(R) <= 1.645 * Sd)),
                   "cv_half_width90": float(np.mean(q * Sd)) if np.isfinite(q) else float("inf"), "scale": scale}
        t1 = time.perf_counter()
        final = _fit(make(), U[fit_idx], Y[fit_idx])
        row["fit_s"] = time.perf_counter() - t1
        row["cv_s"] = cv_s
        row["q"] = q
        for split, idx in (("test", te), ("extrapolation", ex)):
            if len(idx):
                mu, s = final.predict(U[idx])
                row[split] = _field_metrics(Y[idx], mu, s, q, ds.regions) if is_field else \
                    _scalar_metrics(Y[idx], mu, s, q, row["scale"])
                row[split]["pred"] = mu.tolist() if not is_field else None
        t2 = time.perf_counter()
        _ = final.predict(U[:1])
        row["infer_s"] = time.perf_counter() - t2
        key = f"{t}/{f}"
        board.rows[key], board.models[key], board.q[key] = row, final, q
        checks = []
        for chk in (physics or {}).get(t, []):
            c = chk(final.predict)
            checks.append({"name": c.name, "passed": c.status == "supports", "value": c.value, "limit": c.limit,
                           "reason": c.reason})
        board.physics[key] = checks
        log(f"{key}: CV {row.get('cv_rel_rmse', row.get('cv_rel_l2')):.4f}, fit {row['fit_s']:.1f} s, "
            f"physics {'pass' if all(c['passed'] for c in checks) else 'FAIL'}")
    return board


def _fit(model, U, Y):
    return model.fit(U, Y)
