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

import heapq
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
    use the measured wall time, or a callable ``cost(query)`` when it depends on the case (a forecast whose cost
    grows with its lead time). ``provenance``: what the expert is and where it comes from — for a pre-trained model,
    the record :func:`pinneapple_physics.pretrained.from_pretrained_expert` writes (source, licence, cost method)."""
    name: str
    predict: Callable[[Any], Any]
    cost: float | Callable[[Any], float] | None = None
    kind: str = ""
    provenance: dict[str, Any] | None = None
    #: the ``torch.nn.Module`` behind the predictor when there is one -- what
    #: :class:`pinneapple_physics.embedded.ExpertModule` exports. ``None`` means the expert is Python and numpy, and
    #: an ONNX export of it would freeze the numbers the tracer saw instead of computing them.
    module: Any = None

    def cost_of(self, query: Any) -> float | None:
        """The cost of one call on this query, or None when it has to be measured (a callable cost that needs the
        query gives None for ``query=None``: the cost is then only known per case)."""
        if callable(self.cost):
            try:
                c = self.cost(query)
            except (TypeError, KeyError, AttributeError):
                return None
        else:
            c = self.cost
        return None if c is None else float(c)


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
    return Expert(name, predict, cost, kind, module=model)


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
    suspect: np.ndarray | None = None        # (n_cases,) readings flagged by ``measurement_check`` (not learned from)
    licences: dict[str, str] | None = None   # {expert: licence} of the experts that declare a pretrained provenance

    def summary(self) -> dict[str, Any]:
        ok = np.isfinite(self.ensemble_errors)
        mean_err = {n: float(np.nanmean(self.errors[ok, i])) for i, n in enumerate(self.names)}
        best = min(mean_err, key=mean_err.get)
        out = {"ensemble": float(np.mean(self.ensemble_errors[ok])), "experts": mean_err,
               "best_single_in_hindsight": {"name": best, "error": mean_err[best]},
               "coverage": float(np.nanmean(self.coverage)), "switches": self.switches(),
               "mean_cost_per_case": float(np.sum(self.weights * self.costs, axis=1).mean())}
        if self.licences:
            out["licences"] = dict(self.licences)
        return out

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
    ``lookahead_persistence`` (N > 1): the tilt counts only when the last N residuals of an expert agree (all above
    or all below its reference), and then by the smallest departure: a single noisy reading does not switch the
    model, while a real regime change (which persists) still does, N - 1 readings later.
    ``measurement_check(query) -> bool``: True when the reading itself is physically suspect (a failed balance between
    redundant sensors, an impossible jump, energy where a physical state has none). A suspect reading is predicted
    with the learned weights only (no residual tilt) and teaches nothing: it does not update the weights, the
    interval calibration or the residual history.
    ``conserve``: optional
    ``(weights, target_fn(query))`` to project the combined field onto the conserved total.
    ``router``: physics-aware routing (:class:`pinneapple_physics.physics_audit.PhysicsAwareRouter`, or any object
    with ``prior(query, names) -> {name: log-weight}`` and ``awake(query, names) -> {name: bool}``). Before
    predicting, the compatibility of each expert's assumptions and training envelope with the current case tilts the
    learned weights; experts outside their physics are asleep for that case: not used, and not penalised by it (their
    loss is set to the mixture's), so they keep their weight for when the case returns to their regime.
    ``prediction_checks``: label-free checks of each expert's prediction for the current case, run after the experts
    and before choosing (``{name: fn(prediction, query)}``, or one ``fn(name, prediction, query)`` for all), each
    returning a score in [0, 1] or ``(score, reason)``: a closure check of an LES, a cross-scale consistency check, a
    conservation check. A score below ``check_threshold`` takes the expert out of this case (it is still scored, so
    its weight learns from the failure); the others are tilted by ``score ** check_strength``. When every awake
    expert fails, the router's ``escalate_from(query, names, failed)`` (a fidelity ladder, see
    :class:`pinneapple_physics.multiscale.ScaleAwareRouter`) gives the experts to use instead, which are checked in
    turn. ``error_tolerance``: an expert whose last reference error in the same context (``error_context(query)``,
    e.g. the requested scale; one context if None) is above it fails the checks for this case too, so a fidelity
    ladder escalates when the cheap rung is not accurate enough; it is still scored and comes back once its error
    is within the tolerance again."""

    def __init__(self, experts: Sequence[Expert] | Mapping[str, Callable], mode: str = "combine",
                 etas: Sequence[float] = (0.5, 2.0, 8.0), alphas: Sequence[float] = (0.0, 0.01, 0.05),
                 point_weights: np.ndarray | None = None, residual_fn: Callable[[np.ndarray, Any], float] | None = None,
                 residual_weight: float = 0.0, cost_weight: float = 0.0, clip: float = 5.0,
                 interval_level: float | None = 0.9, conformal_gamma: float = 0.02,
                 conserve: tuple | None = None, on_error: str = "skip", explore_every: int = 10,
                 min_weight: float = 0.05, residual_lookahead: float = 0.0, lookahead_reference: str = "self",
                 lookahead_window: int = 10, residual_baseline: Mapping[str, float] | None = None,
                 lookahead_persistence: int = 1, measurement_check: Callable[[Any], bool] | None = None,
                 router: Any = None, prediction_checks: Mapping[str, Callable] | Callable | None = None,
                 check_threshold: float = 0.5, check_strength: float = 1.0, error_tolerance: float | None = None,
                 error_context: Callable[[Any], Any] | None = None):
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
        self._cost_est = np.array([c if (c := e.cost_of(None)) is not None else np.nan for e in self.experts],
                                  dtype=float)
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
        self.lookahead_persistence = max(1, int(lookahead_persistence))
        self.measurement_check = measurement_check
        self.router = router
        self.prediction_checks = prediction_checks
        self.check_threshold, self.check_strength = float(check_threshold), float(check_strength)
        self.error_tolerance, self.error_context = error_tolerance, error_context
        self._ctx_err: dict[tuple, float] = {}                        # (expert, context) -> last reference error

    # ------------------------------------------------------------------------------------------ prediction
    def weights(self) -> dict[str, float]:
        return dict(zip(self.names, self.grid.expert_weights().tolist(), strict=True))

    def provenance(self) -> dict[str, dict[str, Any]]:
        """What each expert is and where it comes from, for those that declare it (see
        :func:`pinneapple_physics.pretrained.from_pretrained_expert`): the source of the weights and their licence."""
        return {e.name: e.provenance for e in self.experts if e.provenance}

    def licences(self) -> dict[str, str]:
        """``{expert: licence identifier}`` of the experts whose provenance records one."""
        return {n: str((p.get("licence") or {}).get("spdx", "")) for n, p in self.provenance().items()}

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
            fixed = e.cost_of(query)
            costs[i] = fixed if fixed is not None else time.perf_counter() - t0
            if fixed is None:                                        # running estimate of measured wall time
                self._cost_est[i] = costs[i] if np.isnan(self._cost_est[i]) else 0.8 * self._cost_est[i] + 0.2 * costs[i]
            else:
                self._cost_est[i] = fixed
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
        w = w_learned = self.grid.expert_weights()
        prior, awake = None, np.ones(len(self.experts), dtype=bool)
        if self.router is not None:                                  # physics compatibility of the current case
            lp = self.router.prior(query, self.names)
            prior = np.array([float(lp.get(n, 0.0)) for n in self.names])
            if hasattr(self.router, "awake"):
                aw = self.router.awake(query, self.names)
                awake = np.array([bool(aw.get(n, True)) for n in self.names])
                if not awake.any():
                    awake[:] = True
            logw = np.log(np.maximum(w, 1e-300)) + prior
            w = np.where(awake, np.exp(logw - logw[awake].max()), 0.0)
        explore = self._n_predict % self.explore_every == 0
        self._n_predict += 1
        lazy = self.mode == "select" and select_only_leader and not explore and self.residual_lookahead == 0
        which = [int(np.argmax(w))] if lazy else range(len(self.experts))
        preds, costs = self._evaluate(query, which)
        ok_all = [i for i in range(len(preds)) if preds[i] is not None]
        if not ok_all:
            raise RuntimeError("every expert failed on this query")
        ok = [i for i in ok_all if awake[i]] or ok_all                # asleep experts are run (scored) but not used
        check_out, failed, escalated = {}, [], False
        if self.prediction_checks is not None or self.error_tolerance is not None:
            ok, w, awake, check_out, failed, escalated = self._run_checks(query, preds, ok, ok_all, w, w_learned,
                                                                           prior, awake)
        stack = np.stack([preds[i] for i in ok])
        res_now = None
        suspect = bool(self.measurement_check(query)) if self.measurement_check is not None else False
        if self.residual_lookahead > 0 and not suspect:            # label-free score of the current case
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
                        recent = self._res_hist[i][-(self.lookahead_persistence - 1):] if self.lookahead_persistence > 1 else []
                        logs = np.log(np.maximum([res_now[i], *recent], 1e-300) / ref_i)
                        # the last N readings must agree (all above or all below the reference): then the smallest
                        # departure counts; a single reading out of line counts for nothing
                        agree = len(logs) == self.lookahead_persistence and (np.all(logs > 0) or np.all(logs < 0))
                        t_i = float(np.sign(logs[0]) * np.min(np.abs(logs))) if (agree or len(logs) == 1) else 0.0
                        tilt[i] = np.clip(t_i, -self.clip, self.clip)
            logw = np.log(np.maximum(w, 1e-300)) - self.residual_lookahead * tilt
            w = np.exp(logw - logw[ok].max())
            for i in ok_all:
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
               "active": self.names[lead], "costs": dict(zip(self.names, costs.tolist(), strict=True)),
               "suspect": suspect}
        if res_now is not None:
            out["residuals"] = dict(zip(self.names, res_now.tolist(), strict=True))
        if prior is not None:
            out["prior"] = dict(zip(self.names, prior.tolist(), strict=True))
            out["awake"] = dict(zip(self.names, awake.tolist(), strict=True))
            out["escalate"] = bool(self.router.escalate(query, self.names)) if hasattr(self.router, "escalate") else False
        if self.prediction_checks is not None or self.error_tolerance is not None:
            out["checks"] = check_out
            out["failed_checks"] = [self.names[i] for i in failed]
            out["escalated_by_checks"] = escalated
            out["awake"] = dict(zip(self.names, awake.tolist(), strict=True))
        self._last = {"query": query, "preds": preds, "y": y, "spread": spread, "scale": scale, "costs": costs, "w": w,
                      "evaluated": set(which), "res_now": res_now,
                      "suspect": suspect, "awake": awake, "failed": set(failed)}
        return out

    def _context(self, query: Any) -> Any:
        return self.error_context(query) if self.error_context is not None else None

    def _check(self, i: int, pred: np.ndarray, query: Any) -> tuple[float, str] | None:
        if self.error_tolerance is not None:
            e = self._ctx_err.get((i, self._context(query)))
            if e is not None and e > self.error_tolerance:
                return 0.0, f"last error in this context {e:.3g} above the tolerance {self.error_tolerance:.3g}"
        pc = self.prediction_checks
        if pc is None:
            return None
        if isinstance(pc, Mapping):
            fn = pc.get(self.names[i])
            r = fn(pred, query) if fn is not None else None
        else:
            r = pc(self.names[i], pred, query)
        if r is None:
            return None
        score, why = (r[0], r[1]) if isinstance(r, tuple) else (r, "")
        return float(score), str(why)

    def _run_checks(self, query, preds, ok, ok_all, w, w_learned, prior, awake):
        """Check the awake experts; drop those that fail; escalate through the router when none is left."""
        checked: dict[int, tuple[float, str]] = {}
        failed: list[int] = []
        escalated = False
        awake = awake.copy()
        for _ in range(len(self.experts)):                           # at most one pass per rung
            for i in ok:
                if i not in checked and preds[i] is not None:
                    r = self._check(i, preds[i], query)
                    if r is not None:
                        checked[i] = r
                        if r[0] < self.check_threshold:
                            failed.append(i)
            remaining = [i for i in ok if i not in failed]
            if remaining:
                ok = remaining
                break
            if self.router is not None and hasattr(self.router, "escalate_from"):
                aw = self.router.escalate_from(query, self.names, [self.names[i] for i in failed])
                new_awake = np.array([bool(aw.get(n, False)) for n in self.names]) & ~np.isin(np.arange(len(preds)), failed)
                new_ok = [i for i in ok_all if new_awake[i]]
                if new_ok and set(new_ok) != set(ok):
                    awake, ok, escalated = new_awake, new_ok, True
                    logw = np.log(np.maximum(w_learned, 1e-300)) + (prior if prior is not None else 0.0)
                    w = np.where(awake, np.exp(logw - logw[awake].max()), 0.0)
                    continue
            break                                                      # nobody passes and nowhere to go: keep them
        if any(i not in failed for i in ok):
            for i in failed:
                awake[i] = False
        for i, (sc, _) in checked.items():
            if i in ok:
                w[i] = w[i] * max(sc, 1e-12) ** self.check_strength
        out = {self.names[i]: {"score": sc, "reason": why} for i, (sc, why) in checked.items()}
        return ok, w, awake, out, [i for i in failed if i not in ok], escalated

    # ------------------------------------------------------------------------------------------ learning
    def last_state(self) -> dict[str, Any]:
        """Opaque record of the last :meth:`predict` call (expert fields, weights used, residuals, interval state).
        Keep it and pass it to ``update(..., state=...)`` to score that prediction later, after other cases have been
        predicted: the way to learn from a label that arrives with a delay (see :meth:`run_delayed`)."""
        if self._last is None:
            raise RuntimeError("call predict(query) before last_state()")
        return self._last

    def update(self, query: Any = None, reference: np.ndarray | None = None,
               state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Score the last prediction (call :meth:`predict` first for the same query), or the prediction recorded in
        ``state`` (from :meth:`last_state`) when its label arrives later. Uses ``reference`` if given, the physics
        residual if ``residual_fn`` is set, and the cost if ``cost_weight`` > 0."""
        if state is None and self._last is None:
            raise RuntimeError("call predict(query) before update()")
        last = self._last if state is None else state
        query = last["query"] if query is None else query
        preds, n = last["preds"], len(self.experts)
        loss = np.zeros(n)
        info: dict[str, Any] = {}
        if last.get("suspect"):                                      # a suspect reading is scored, never learned from
            info["suspect"] = True
            if reference is not None:
                ref = np.asarray(reference, dtype=float)
                info["errors"] = np.array([relative_l2(p, ref, self.point_weights) if p is not None else np.inf
                                           for p in preds])
                info["ensemble_error"] = relative_l2(last["y"], ref, self.point_weights)
            return info
        if reference is not None:
            ref = np.asarray(reference, dtype=float)
            err = np.array([relative_l2(p, ref, self.point_weights) if p is not None else np.inf for p in preds])
            loss += np.minimum(err, self.clip)
            info["errors"] = err
            if self.error_tolerance is not None:
                ctx = self._context(query)
                for i in last["evaluated"]:
                    self._ctx_err[(i, ctx)] = float(err[i])
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
        asleep = ~last.get("awake", np.ones(n, dtype=bool))
        for i in last.get("failed", ()):                              # failed a check: scored by its real loss
            asleep[i] = False
        if asleep.any() and not asleep.all():                       # sleeping experts: the mixture's loss, no change
            wg = self.grid.expert_weights()
            aw = ~asleep
            loss[asleep] = float(wg[aw] @ loss[aw] / max(wg[aw].sum(), 1e-300))
        for i in evaluated:
            if not asleep[i]:
                self._last_loss[i] = loss[i]
        # loss of each aggregator's own combination (data part only when a reference exists)
        agg_losses = []
        for a in self.grid.aggs:
            ok = [i for i in range(n) if preds[i] is not None and not asleep[i]]
            if len(ok) < n - int(asleep.sum()) and reference is None:
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
        errs, ens, ws, act, cov, costs, kept, sus = [], [], [], [], [], [], [], []
        for k, q in enumerate(queries):
            p = self.predict(q)
            if keep_predictions:
                kept.append({key: p[key] for key in ("prediction", "lower", "upper", "expert_predictions", "active")})
            ref = None if references is None else references[k]
            ws.append([p["weights"][m] for m in self.names])
            act.append(p["active"])
            sus.append(bool(p.get("suspect", False)))
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
                           act, np.array(cov), np.array(costs), kept if keep_predictions else None, np.array(sus),
                           self.licences() or None)

    def run_delayed(self, queries: Sequence[Any], references: Sequence[np.ndarray | None],
                    available_at: Sequence[float], issue_times: Sequence[float] | None = None,
                    keep_predictions: bool = False, flush: bool = True) -> EnsembleRun:
        """Prequential evaluation with delayed labels: the reference of case k exists only from time
        ``available_at[k]`` on (a forecast verified at its lead time, a lab result that takes days).

        Cases are predicted in the given order, case k at ``issue_times[k]`` (default: k; the order must be
        chronological). Before predicting case k, every earlier case whose label has matured
        (``available_at[j] <= issue_times[k]``) updates the weights against the prediction stored for it
        (:meth:`last_state`), in order of availability (a heap; ties by case index). A label is never used before it
        exists, even with several overlapping events in flight. With ``available_at[k] = issue_times[k]`` (label known
        right after its own prediction) this is :meth:`run`. ``flush``: score the labels still pending after the last
        case (their errors appear in the result and the weights learn from them); with ``flush=False`` those cases
        keep NaN errors and the ensemble state is the one of the last prediction.

        Returns an :class:`EnsembleRun` like :meth:`run`; ``weights[k]`` are the weights used for case k, which reflect
        only the labels available at its issue time."""
        n_q = len(queries)
        if len(references) != n_q or len(available_at) != n_q:
            raise ValueError("queries, references and available_at must have the same length")
        times = list(range(n_q)) if issue_times is None else list(issue_times)
        if len(times) != n_q:
            raise ValueError("issue_times must have the same length as queries")
        n = len(self.experts)
        errs = np.full((n_q, n), np.nan)
        ens, cov = np.full(n_q, np.nan), np.full(n_q, np.nan)
        ws, act, costs, kept, sus, bounds = [], [], [], [], [], []
        pending: list[tuple] = []

        def learn(j: int, state: dict[str, Any]) -> None:
            ref = references[j]
            info = self.update(queries[j], ref, state=state)
            e = np.asarray(info.get("errors", np.full(n, np.nan)), dtype=float)
            errs[j] = np.where(np.isfinite(e), e, np.nan)
            ens[j] = info.get("ensemble_error", np.nan)
            lo, hi = bounds[j]
            if ref is not None and np.all(np.isfinite(lo)):
                r = np.asarray(ref)
                cov[j] = float(np.mean((r >= lo) & (r <= hi)))

        for k, q in enumerate(queries):
            while pending and pending[0][0] <= times[k]:
                _, j, state = heapq.heappop(pending)
                learn(j, state)
            p = self.predict(q)
            heapq.heappush(pending, (available_at[k], k, self.last_state()))
            bounds.append((p["lower"], p["upper"]))
            if keep_predictions:
                kept.append({key: p[key] for key in ("prediction", "lower", "upper", "expert_predictions", "active")})
            ws.append([p["weights"][m] for m in self.names])
            act.append(p["active"])
            sus.append(bool(p.get("suspect", False)))
            costs.append([p["costs"][m] for m in self.names])
        while flush and pending:
            _, j, state = heapq.heappop(pending)
            learn(j, state)
        return EnsembleRun(self.names, errs, ens, np.array(ws), act, cov, np.array(costs),
                           kept if keep_predictions else None, np.array(sus), self.licences() or None)


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
