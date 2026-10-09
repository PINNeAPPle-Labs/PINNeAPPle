"""Adaptive ensembles of physics models: PINNs, neural operators (FNO, DeepONet...), GNNs (MeshGraphNet...), numerical
solvers and closed forms, selected and combined online from the errors they make on new cases.

A query is one case: parameters, an initial or boundary condition, a geometry, a time step of a rollout. Every expert
turns the query into a prediction of the same field at the same points (adapters below handle the different inputs of
PINNs, operators and graph networks). When something reveals how good the predictions were, the weights are updated:

* a **reference** (a sensor reading, an experiment, a high-fidelity solver run on some of the cases): relative L2 error;
* the **physics residual** of each prediction (``residual_fn``), which needs no reference at all: models whose output
  violates the governing equations lose weight even in operation, where no truth is available;
* optionally the **cost** of each expert, so a cheap surrogate is preferred while it is accurate enough and the
  expensive solver takes over when it is not (adaptive fidelity).

The weighting is the prequential scheme of :mod:`pinneapple_physics.online_learning` (Fixed-Share per expert grid,
AdaHedge over the learning and switching rates): every update uses only cases already seen, so the choice of model
cannot overfit. Point-wise intervals are ensemble spread times a factor calibrated online by adaptive conformal
inference, so they are wide where the models disagree. :func:`fit_static_weights` gives the offline alternative
(convex weights fitted on a validation set, with a cross-validated estimate of their error).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .online_learning import FixedShareGrid

__all__ = ["Expert", "from_callable", "from_solution", "from_torch", "PhysicsEnsemble", "EnsembleRun",
           "fit_static_weights", "relative_l2"]


def _to_numpy(y: Any) -> np.ndarray:
    if hasattr(y, "y") and not isinstance(y, np.ndarray):          # OperatorOutput / ModelOutput
        y = y.y
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return np.asarray(y, dtype=float)


def relative_l2(pred: np.ndarray, ref: np.ndarray, weights: np.ndarray | None = None) -> float:
    d = (pred - ref) ** 2
    r = ref ** 2
    if weights is not None:
        d, r = d * weights, r * weights
    return float(np.sqrt(d.sum() / max(r.sum(), 1e-300)))


@dataclass
class Expert:
    """``predict(query) -> array`` of the common output shape; ``cost``: fixed cost per call (any unit), or None to
    use the measured wall time."""
    name: str
    predict: Callable[[Any], Any]
    cost: float | None = None
    kind: str = ""


def from_callable(name: str, fn: Callable[[Any], Any], cost: float | None = None, kind: str = "numerical") -> Expert:
    return Expert(name, lambda q: _to_numpy(fn(q)), cost, kind)


def from_solution(name: str, solution: Any, points: Callable[[Any], np.ndarray], cost: float | None = None,
                  shape: Callable[[Any], tuple] | None = None) -> Expert:
    """A ``pp.solve`` result (PINN, FEM, analytic, external solver...): evaluated at ``points(query)``."""
    def predict(q):
        y = _to_numpy(solution.predict(points(q)))
        return y.reshape(shape(q)) if shape is not None else y
    return Expert(name, predict, cost, getattr(solution, "kind", "solution"))


def from_torch(name: str, model: Any, to_input: Callable[[Any], Any], from_output: Callable[[Any, Any], Any] | None = None,
               cost: float | None = None, kind: str = "neural") -> Expert:
    """Any ``torch.nn.Module`` (FNO, DeepONet, MeshGraphNet, a PINN network...): ``to_input(query)`` builds its input
    (tensor, GraphBatch...), the output's ``.y`` is used if present, and ``from_output(y, query)`` maps it to the
    common points (e.g. drop the batch and channel axes)."""
    import torch

    def predict(q):
        model.eval()
        with torch.no_grad():
            out = model(to_input(q))
        y = out.y if hasattr(out, "y") else out
        y = from_output(y, q) if from_output is not None else y
        return _to_numpy(y)
    return Expert(name, predict, cost, kind)


@dataclass
class EnsembleRun:
    names: list[str]
    errors: np.ndarray                       # (n_cases, n_experts) relative L2 per expert (NaN without reference)
    ensemble_errors: np.ndarray              # (n_cases,)
    weights: np.ndarray                      # (n_cases, n_experts) weights used for each case (before its update)
    active: list[str]
    coverage: np.ndarray                     # fraction of points inside the interval, per case
    costs: np.ndarray                        # (n_cases, n_experts)
    predictions: list[dict[str, Any]] | None = None   # run(keep_predictions=True): field, interval, experts per case

    def summary(self) -> dict[str, Any]:
        ok = np.isfinite(self.ensemble_errors)
        mean_err = {n: float(np.nanmean(self.errors[ok, i])) for i, n in enumerate(self.names)}
        best = min(mean_err, key=mean_err.get)
        return {"ensemble": float(np.mean(self.ensemble_errors[ok])), "experts": mean_err,
                "best_single_in_hindsight": {"name": best, "error": mean_err[best]},
                "coverage": float(np.nanmean(self.coverage)), "switches": self.switches(),
                "mean_cost_per_case": float(np.sum(self.weights * self.costs, axis=1).mean())}

    def switches(self) -> list[tuple]:
        out, prev = [], None
        for i, a in enumerate(self.active):
            if a != prev:
                out.append((i, a))
                prev = a
        return out


class PhysicsEnsemble:
    """Online selection / combination of physics models.

    ``mode``: "combine" (weighted average of the fields) or "select" (the field of the leading expert; with
    ``select_only_leader=True`` only that expert is evaluated, which is how the ensemble saves cost).
    ``point_weights``: quadrature weights of the output points (cell volumes) for the L2 norms. ``min_weight``: in
    "combine" mode experts below this weight are left out of the average (still scored, so they can come back): a
    model that has failed badly does not contaminate the combination through the fixed-share floor.
    ``residual_fn(prediction, query) -> float`` (>= 0): physics residual used as a label-free loss with
    ``residual_weight``; ``cost_weight`` penalises expensive experts. ``residual_lookahead`` (> 0, needs
    ``residual_fn``): before predicting, every expert's residual on the *current* query tilts the learned weights.
    The residual needs no reference, so this reacts to a regime change on its first case instead of after the errors
    of earlier cases pile up. ``lookahead_reference="self"`` (default) compares each expert's residual with its own
    median over the last ``lookahead_window`` cases, ``w_i * (r_i / median_i) ** -residual_lookahead``: a jump means
    the case left that expert's domain, while experts with different discretisations are never ranked by their
    residual levels (a numerical scheme that satisfies its discrete equation can have a small residual and a larger
    error than a neural operator). ``"experts"`` compares the residual levels across experts,
    ``w_i * exp(-residual_lookahead * r_i / scale)``; with large values it picks the smallest residual.
    ``residual_baseline`` ({expert: residual level in its own domain}, see :func:`residual_baseline`) replaces the
    recent median as the reference of those experts: a model's recent history is out of domain right after a regime
    change, so two models whose residuals both drop cannot be told apart by it, while their in-domain levels can.
    ``conserve``: optional
    ``(weights, target_fn(query))`` to project the combined field onto the conserved total."""

    def __init__(self, experts: Sequence[Expert] | Mapping[str, Callable], mode: str = "combine",
                 etas: Sequence[float] = (0.5, 2.0, 8.0), alphas: Sequence[float] = (0.0, 0.01, 0.05),
                 point_weights: np.ndarray | None = None, residual_fn: Callable[[np.ndarray, Any], float] | None = None,
                 residual_weight: float = 0.0, cost_weight: float = 0.0, clip: float = 5.0,
                 interval_level: float | None = 0.9, conformal_gamma: float = 0.02,
                 conserve: tuple | None = None, on_error: str = "skip", explore_every: int = 10,
                 min_weight: float = 0.05, residual_lookahead: float = 0.0, lookahead_reference: str = "self",
                 lookahead_window: int = 10, residual_baseline: Mapping[str, float] | None = None):
        if isinstance(experts, Mapping):
            experts = [from_callable(n, f) for n, f in experts.items()]
        if len(experts) < 2:
            raise ValueError("give at least two experts")
        if mode not in ("combine", "select"):
            raise ValueError("mode must be 'combine' or 'select'")
        self.experts = list(experts)
        self.names = [e.name for e in self.experts]
        self.mode, self.point_weights = mode, point_weights
        self.residual_fn, self.residual_weight, self.cost_weight = residual_fn, float(residual_weight), float(cost_weight)
        self.clip, self.conserve, self.on_error = float(clip), conserve, on_error
        self.grid = FixedShareGrid(len(self.experts), etas, alphas)
        self.level = interval_level
        self.alpha_t = (1 - interval_level) if interval_level else None
        self.gamma = conformal_gamma
        self.scores: list[np.ndarray] = []
        self._res_scale: list[float] = []
        self.failures = {n: 0 for n in self.names}
        self._last: dict[str, Any] | None = None
        self.explore_every = max(1, int(explore_every))
        self._n_predict = 0
        self._last_loss = np.full(len(self.experts), np.nan)      # most recent observed loss of each expert
        self._cost_est = np.array([e.cost if e.cost is not None else np.nan for e in self.experts], dtype=float)
        self.min_weight = float(min_weight)
        self.residual_lookahead = float(residual_lookahead)
        if self.residual_lookahead > 0 and residual_fn is None:
            raise ValueError("residual_lookahead needs residual_fn")
        if lookahead_reference not in ("self", "experts"):
            raise ValueError("lookahead_reference must be 'self' or 'experts'")
        self.lookahead_reference, self.lookahead_window = lookahead_reference, max(1, int(lookahead_window))
        self._res_hist: list[list[float]] = [[] for _ in self.experts]   # residuals seen, per expert
        unknown = sorted(set(residual_baseline or {}) - set(self.names))
        if unknown:
            raise ValueError(f"residual_baseline for unknown experts: {unknown}")
        self.residual_baseline = dict(residual_baseline or {})

    # ------------------------------------------------------------------------------------------ prediction
    def weights(self) -> dict[str, float]:
        return dict(zip(self.names, self.grid.expert_weights().tolist(), strict=True))

    def _evaluate(self, query, which: Sequence[int]):
        preds, costs = [None] * len(self.experts), np.zeros(len(self.experts))
        for i in which:
            e = self.experts[i]
            t0 = time.perf_counter()
            try:
                p = np.asarray(e.predict(query), dtype=float)
                if not np.all(np.isfinite(p)):
                    raise ValueError("non-finite prediction")
                preds[i] = p
            except Exception:
                if self.on_error == "raise":
                    raise
                self.failures[e.name] += 1
            costs[i] = e.cost if e.cost is not None else time.perf_counter() - t0
            if e.cost is None:                                       # running estimate of measured wall time
                self._cost_est[i] = costs[i] if np.isnan(self._cost_est[i]) else 0.8 * self._cost_est[i] + 0.2 * costs[i]
        return preds, costs

    def _interval_factor(self) -> float:
        if self.alpha_t is None or len(self.scores) < 3:
            return float("nan")
        s = np.concatenate(self.scores[-200:])
        a = min(max(self.alpha_t, 1e-6), 1 - 1e-6)
        return float(np.quantile(s, 1 - a, method="higher"))

    def predict(self, query: Any, select_only_leader: bool = False) -> dict[str, Any]:
        """Predict one case. With ``select_only_leader`` only the leading expert is run (to save cost), except every
        ``explore_every``-th call, when all experts run so the others' errors stay up to date."""
        w = self.grid.expert_weights()
        explore = self._n_predict % self.explore_every == 0
        self._n_predict += 1
        lazy = self.mode == "select" and select_only_leader and not explore and self.residual_lookahead == 0
        which = [int(np.argmax(w))] if lazy else range(len(self.experts))
        preds, costs = self._evaluate(query, which)
        ok = [i for i in range(len(preds)) if preds[i] is not None]
        if not ok:
            raise RuntimeError("every expert failed on this query")
        stack = np.stack([preds[i] for i in ok])
        res_now = None
        if self.residual_lookahead > 0:                            # label-free score of the current case
            res_now = np.array([self.residual_fn(p, query) if p is not None else np.inf for p in preds], dtype=float)
            if self.lookahead_reference == "experts":                  # residual levels compared across experts
                fin = res_now[np.isfinite(res_now)]
                sc = float(np.median(self._res_scale[-50:])) if self._res_scale else (float(np.median(fin)) if fin.size else 1.0)
                tilt = np.minimum(res_now / (sc or 1.0), self.clip)
            else:                                                      # each expert against its own recent residuals
                tilt = np.zeros(len(preds))
                for i in ok:
                    h = self._res_hist[i][-self.lookahead_window:]
                    base = self.residual_baseline.get(self.names[i])
                    if (h or base) and np.isfinite(res_now[i]):
                        ref_i = max(float(base) if base else float(np.median(h)), 1e-300)
                        tilt[i] = np.clip(np.log(max(res_now[i], 1e-300) / ref_i), -self.clip, self.clip)
            logw = np.log(np.maximum(w, 1e-300)) - self.residual_lookahead * tilt
            w = np.exp(logw - logw[ok].max())
            for i in ok:
                if np.isfinite(res_now[i]):
                    self._res_hist[i].append(float(res_now[i]))
        wk = w[ok] / w[ok].sum()
        if self.mode == "combine" and self.min_weight > 0:          # drop near-zero weights from the average only
            keep = wk >= min(self.min_weight, wk.max())
            wk = np.where(keep, wk, 0.0) / wk[keep].sum()
        if self.mode == "select":
            lead = ok[int(np.argmax(wk))]
            y = preds[lead].copy()
        else:
            lead = ok[int(np.argmax(wk))]
            y = np.tensordot(wk, stack, axes=1)
        if self.conserve is not None:
            pw, target_fn = self.conserve
            y = y + (target_fn(query) - float(np.sum(pw * y))) / float(np.sum(pw))
        spread = stack.std(axis=0) if len(ok) > 1 else np.zeros_like(y)
        scale = float(np.sqrt(np.mean(y ** 2))) or 1.0
        width = (spread + 1e-3 * scale) * self._interval_factor()
        out = {"prediction": y, "lower": y - width, "upper": y + width, "spread": spread,
               "expert_predictions": {self.names[i]: preds[i] for i in ok}, "weights": dict(zip(self.names, w.tolist(), strict=True)),
               "active": self.names[lead], "costs": dict(zip(self.names, costs.tolist(), strict=True))}
        if res_now is not None:
            out["residuals"] = dict(zip(self.names, res_now.tolist(), strict=True))
        self._last = {"query": query, "preds": preds, "y": y, "spread": spread, "scale": scale, "costs": costs, "w": w,
                      "evaluated": set(which), "res_now": res_now}
        return out

    # ------------------------------------------------------------------------------------------ learning
    def update(self, query: Any = None, reference: np.ndarray | None = None) -> dict[str, Any]:
        """Score the last prediction (call :meth:`predict` first for the same query). Uses ``reference`` if given,
        the physics residual if ``residual_fn`` is set, and the cost if ``cost_weight`` > 0."""
        if self._last is None:
            raise RuntimeError("call predict(query) before update()")
        last = self._last
        query = last["query"] if query is None else query
        preds, n = last["preds"], len(self.experts)
        loss = np.zeros(n)
        info: dict[str, Any] = {}
        if reference is not None:
            ref = np.asarray(reference, dtype=float)
            err = np.array([relative_l2(p, ref, self.point_weights) if p is not None else np.inf for p in preds])
            loss += np.minimum(err, self.clip)
            info["errors"] = err
            info["ensemble_error"] = relative_l2(last["y"], ref, self.point_weights)
            if self.alpha_t is not None:
                denom = last["spread"] + 1e-3 * last["scale"]
                self.scores.append((np.abs(ref - last["y"]) / denom).ravel())
                f = self._interval_factor()
                if np.isfinite(f) and len(self.scores) > 3:
                    miss = float(np.mean(np.abs(ref - last["y"]) > denom * f))
                    self.alpha_t += self.gamma * ((1 - self.level) - miss)
                    info["miss_fraction"] = miss
        if self.residual_fn is not None and self.residual_weight > 0:
            res = last["res_now"] if last.get("res_now") is not None else \
                np.array([self.residual_fn(p, query) if p is not None else np.inf for p in preds], dtype=float)
            fin = res[np.isfinite(res)]
            if fin.size:
                self._res_scale.append(float(np.median(fin)) or 1.0)
            sc = float(np.median(self._res_scale[-50:])) if self._res_scale else 1.0
            loss += self.residual_weight * np.minimum(res / sc, self.clip)
            info["residuals"] = res
        if self.cost_weight > 0:                                     # known cost of every expert, not only those run
            c = np.where(np.isnan(self._cost_est), np.nanmax(self._cost_est) if np.any(np.isfinite(self._cost_est)) else 1.0,
                         self._cost_est)
            loss += self.cost_weight * c / max(c.max(), 1e-300)
        worst = self.clip * (1 + self.residual_weight + self.cost_weight)
        evaluated = last["evaluated"]
        for i in range(n):
            if i not in evaluated:                                     # not run: last known loss (no news), not a failure
                loss[i] = self._last_loss[i] if np.isfinite(self._last_loss[i]) else np.nan
        known = loss[[i for i in evaluated if np.isfinite(loss[i])]]
        loss = np.where(np.isnan(loss), np.median(known) if known.size else 0.0, loss)
        loss = np.where(np.isfinite(loss), loss, worst)
        for i in evaluated:
            self._last_loss[i] = loss[i]
        # loss of each aggregator's own combination (data part only when a reference exists)
        agg_losses = []
        for a in self.grid.aggs:
            ok = [i for i in range(n) if preds[i] is not None]
            if len(ok) < n and reference is None:
                agg_losses.append(float(a.w @ loss))
                continue
            wa = a.w[ok] / a.w[ok].sum()
            ya = np.tensordot(wa, np.stack([preds[i] for i in ok]), axes=1)
            la = float(wa @ loss[ok])                                  # upper bound by convexity
            if reference is not None:
                la = la - float(wa @ np.minimum(info["errors"][ok], self.clip)) + min(relative_l2(ya, ref, self.point_weights), self.clip)
            agg_losses.append(la)
        self.grid.update(loss, np.array(agg_losses))
        info["loss"] = loss
        return info

    def run(self, queries: Sequence[Any], references: Sequence[np.ndarray | None] | None = None,
            keep_predictions: bool = False) -> EnsembleRun:
        """Sequential evaluation: predict each case with the weights learned so far, then update with its reference
        (if any) and residual. ``keep_predictions`` stores each case's fields (ensemble, interval, every expert) in
        ``EnsembleRun.predictions``, e.g. for ``pinneapple_physics.ensemble_viz.animate_ensemble``."""
        n = len(self.experts)
        errs, ens, ws, act, cov, costs, kept = [], [], [], [], [], [], []
        for k, q in enumerate(queries):
            p = self.predict(q)
            if keep_predictions:
                kept.append({key: p[key] for key in ("prediction", "lower", "upper", "expert_predictions", "active")})
            ref = None if references is None else references[k]
            ws.append([p["weights"][m] for m in self.names])
            act.append(p["active"])
            costs.append([p["costs"][m] for m in self.names])
            info = self.update(q, ref)
            e = np.asarray(info.get("errors", np.full(n, np.nan)), dtype=float)
            errs.append(np.where(np.isfinite(e), e, np.nan))           # experts not run on this case: NaN
            ens.append(info.get("ensemble_error", np.nan))
            if ref is not None and np.all(np.isfinite(p["lower"])):
                r = np.asarray(ref)
                cov.append(float(np.mean((r >= p["lower"]) & (r <= p["upper"]))))
            else:
                cov.append(np.nan)
        return EnsembleRun(self.names, np.array(errs, dtype=float), np.array(ens, dtype=float), np.array(ws),
                           act, np.array(cov), np.array(costs), kept if keep_predictions else None)


def residual_baseline(experts: Sequence[Expert], domain_queries: Mapping[str, Sequence[Any]],
                      residual_fn: Callable[[np.ndarray, Any], float]) -> dict[str, float]:
    """Median physics residual of each expert on queries from its own domain (its training or validation cases),
    known before operation: the reference for ``PhysicsEnsemble(residual_baseline=...)``. Experts missing from
    ``domain_queries`` are left out (they fall back to their recent history)."""
    out = {}
    for e in experts:
        qs = domain_queries.get(e.name)
        if qs:
            out[e.name] = float(np.median([residual_fn(_to_numpy(e.predict(q)), q) for q in qs]))
    return out


def fit_static_weights(predictions: np.ndarray, references: np.ndarray, n_folds: int = 5,
                       point_weights: np.ndarray | None = None, seed: int = 0) -> dict[str, Any]:
    """Offline ensemble: convex weights (>= 0, sum 1) minimising the summed squared error on a validation set.

    ``predictions``: (n_cases, n_experts, ...) fields; ``references``: (n_cases, ...). Returns the weights fitted on all
    cases and a K-fold cross-validated relative L2 error of the weighted ensemble next to each single expert's, so the
    gain is measured on cases the weights did not see."""
    from scipy.optimize import minimize

    P = np.asarray(predictions, dtype=float)
    Y = np.asarray(references, dtype=float)
    n, k = P.shape[:2]
    pw = 1.0 if point_weights is None else point_weights

    def solve(idx):
        A = (P[idx] * np.sqrt(pw)).reshape(len(idx), k, -1).transpose(1, 0, 2).reshape(k, -1).T
        b = (Y[idx] * np.sqrt(pw)).reshape(-1)
        G, h = A.T @ A, A.T @ b
        res = minimize(lambda w: w @ G @ w - 2 * h @ w, np.full(k, 1 / k), jac=lambda w: 2 * (G @ w - h),
                       bounds=[(0, 1)] * k, constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},), method="SLSQP")
        return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()

    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, min(n_folds, n))
    cv_ens, cv_single = [], [[] for _ in range(k)]
    for f in folds:
        train = np.setdiff1d(order, f)
        w = solve(train)
        for i in f:
            cv_ens.append(relative_l2(np.tensordot(w, P[i], axes=1), Y[i], point_weights))
            for j in range(k):
                cv_single[j].append(relative_l2(P[i, j], Y[i], point_weights))
    w_all = solve(np.arange(n))
    single = [float(np.mean(c)) for c in cv_single]
    return {"weights": w_all, "cv_error": float(np.mean(cv_ens)), "single_errors": single,
            "best_single": int(np.argmin(single))}
