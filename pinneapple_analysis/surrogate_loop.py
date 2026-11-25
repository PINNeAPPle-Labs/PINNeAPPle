"""Surrogate-in-the-loop optimisation with a verification budget (#297).

Optimising on a surrogate finds the surrogate's errors as readily as good designs. The industrial answer is a loop:

1. search on the surrogate;
2. send the most promising and most uncertain designs to the solver, inside a fixed budget;
3. add the results to the training data and refit;
4. repeat until the *verified* optimum stops moving.

:func:`loop` is that loop. It is built on the pieces that already exist rather than beside them:
:class:`~pinneapple_data.adaptive_sampling.GPSurrogate` (the surrogate of the adaptive sampler, #209),
:func:`~pinneapple_data.doe.designs.greedy_maximin` (the batch thinning of #209, so a batch of four is not four
neighbours in the same corner), and :func:`~pinneapple_analysis.uncertainty.applicability.route_cases`, the decision
primitive of the adaptive-fidelity router (#161) — so a design whose predicted interval straddles a constraint limit is
recognised as undetermined instead of being silently accepted.

What each round reports
-----------------------
``designs`` holds one record per high-fidelity run or certification: the predicted objective, the verified one, the
error between them, whether the design met the constraints and, when it did not, which constraint and by how much.
``rounds`` holds the error trend (mean ``|predicted - verified|`` of the round's re-runs), the best verified design so
far, how much it moved, and how many designs were certified instead of run. The loop stops when the verified optimum
has moved less than ``min_improve`` (relative) for ``patience`` rounds, when the solver budget is spent, or when
``max_rounds`` is reached.

Acquisition
-----------
``acquisition="ei"`` (default) is *constrained expected improvement*: the expected improvement of the objective over
the best feasible verified value, multiplied by the probability that every constraint is met, both read from the
surrogate's predictive mean and spread. ``"lcb"`` is the lower confidence bound times that same probability — the
cheaper variant, and the one to use with an ensemble whose mean and spread come from the members. The spread is
whatever the surrogate reports, so a Gaussian process and a bootstrap ensemble go through the same code; only the
quantity that fills ``std`` differs.

The benchmark that backs this is :func:`benchmark_loop`, on a 2-D airfoil section with the panel method of
:mod:`pinneapple_design.aero.panel` as the solver: it is asserted in ``tests/test_surrogate_loop.py``, and the numbers
there are what the promise rests on.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from pinneapple_analysis.uncertainty.applicability import route_cases
from pinneapple_data.adaptive_sampling import GPSurrogate, _as_bounds
from pinneapple_data.doe.designs import greedy_maximin

__all__ = ["Constraint", "Design", "LoopResult", "Round", "loop", "benchmark_loop", "airfoil_solver",
           "GaussianProcessSurrogate", "EnsembleSurrogate"]

_EPS = 1e-12
Z_INTERVAL = 1.96          # the router's interval: mean +- 1.96 std, the 95 % band


# ---------------------------------------------------------------------------------------------- constraints
@dataclass(frozen=True)
class Constraint:
    """One constraint on one output of the solver, with the side it is allowed to be on.

    ``sense=">="`` means the design passes when the value is at least ``limit`` (a lift floor); ``"<="`` means at most
    (a stress ceiling). ``scale`` is the tolerance the rejection reason is measured against and only affects the
    wording, not the decision.
    """
    name: str
    limit: float
    sense: str = ">="
    scale: float | None = None

    def __post_init__(self) -> None:
        if self.sense not in (">=", "<="):
            raise ValueError(f"Constraint({self.name!r}): sense must be '>=' or '<=', got {self.sense!r}")

    def ok(self, value: float) -> bool:
        return bool(value >= self.limit) if self.sense == ">=" else bool(value <= self.limit)

    def violation(self, value: float) -> float:
        """How far outside the allowed side, in output units (0 when it passes)."""
        return float(max(0.0, self.limit - value) if self.sense == ">=" else max(0.0, value - self.limit))

    def probability(self, mean, sd) -> np.ndarray:
        """``P(constraint met)`` under the surrogate's ``N(mean, sd^2)`` predictive, one value per candidate."""
        from scipy.stats import norm
        mean, sd = np.asarray(mean, float), np.maximum(np.asarray(sd, float), _EPS)
        margin = (mean - self.limit) if self.sense == ">=" else (self.limit - mean)
        return norm.cdf(margin / sd)

    def reason(self, value: float) -> str:
        if self.ok(value):
            return ""
        return f"{self.name} {value:.4g} {'<' if self.sense == '>=' else '>'} limit {self.limit:.4g}"


# ---------------------------------------------------------------------------------------------- surrogates
class GaussianProcessSurrogate:
    """One GP per output — the :class:`~pinneapple_data.adaptive_sampling.GPSurrogate` of #209, fitted column by column.

    ``predict`` returns ``(mean, std)``, each ``(n, k)`` with ``k`` the number of solver outputs (objective first).
    """

    def __init__(self, d: int, k: int = 1, seed: int = 0) -> None:
        self.d, self.k, self.seed = int(d), int(k), int(seed)
        self.model = GPSurrogate(self.d, seed=self.seed)

    def fit(self, X, y) -> GaussianProcessSurrogate:
        Y = np.asarray(y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.k = Y.shape[1]
        self._members = []
        for i in range(self.k):
            m = GPSurrogate(self.d, seed=self.seed + i)
            m.fit(np.asarray(X, float), Y[:, i])
            self._members.append(m)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        mean = np.zeros((X.shape[0], self.k))
        std = np.zeros((X.shape[0], self.k))
        for i, m in enumerate(getattr(self, "_members", [])):
            mu, sd = m.predict(X)
            mean[:, i], std[:, i] = np.asarray(mu).ravel(), np.asarray(sd).ravel()
        if not getattr(self, "_members", None):
            return mean, np.zeros_like(mean)
        return mean, std


class EnsembleSurrogate:
    """A bootstrap ensemble of small MLPs; the spread across members is the ``std`` the acquisition reads.

    ``n_members`` networks on bootstrap resamples of the rows, inputs and outputs standardised, trained with Adam for
    ``epochs``. It fits in a second on the few dozen points a verification budget buys, which is the regime this loop
    runs in; for larger data use :class:`GaussianProcessSurrogate` or any object with the same ``fit``/``predict``
    contract.
    """

    def __init__(self, d: int, k: int = 1, *, n_members: int = 4, hidden: int = 32, epochs: int = 400,
                 lr: float = 5e-3, seed: int = 0) -> None:
        self.d, self.k = int(d), int(k)
        self.n_members, self.hidden, self.epochs, self.lr, self.seed = n_members, hidden, epochs, lr, seed

    def fit(self, X, y) -> EnsembleSurrogate:
        import torch
        import torch.nn as nn

        Y = np.asarray(y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.k = Y.shape[1]
        X = np.asarray(X, float)
        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-8
        self._y_mu, self._y_sd = Y.mean(0), Y.std(0) + 1e-8
        Xn = torch.tensor((X - self._x_mu) / self._x_sd, dtype=torch.float32)
        Yn = torch.tensor((Y - self._y_mu) / self._y_sd, dtype=torch.float32)
        rng = np.random.default_rng(self.seed)
        self._members = []
        for m in range(self.n_members):
            torch.manual_seed(self.seed + 100 * m)
            net = nn.Sequential(nn.Linear(self.d, self.hidden), nn.Tanh(),
                               nn.Linear(self.hidden, self.hidden), nn.Tanh(),
                               nn.Linear(self.hidden, self.k))
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            idx = torch.tensor(rng.integers(0, len(X), len(X)))          # bootstrap: spread without a prior
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = ((net(Xn[idx]) - Yn[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            self._members.append(net)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        import torch

        X = np.atleast_2d(np.asarray(X, float))
        if not getattr(self, "_members", None):
            return np.zeros((X.shape[0], self.k)), np.zeros((X.shape[0], self.k))
        xt = torch.tensor((X - self._x_mu) / self._x_sd, dtype=torch.float32)
        with torch.no_grad():
            stack = np.stack([net(xt).numpy() for net in self._members], 0) * self._y_sd + self._y_mu
        return stack.mean(0), stack.std(0)


# ---------------------------------------------------------------------------------------------- records
@dataclass
class Design:
    """One design the loop has an answer for: run in high fidelity, or certified from the surrogate."""
    round: int
    x: list[float]
    objective: float
    constraints: dict[str, float]
    predicted_objective: float
    predicted_constraints: dict[str, float]
    status: str                       # "verified" | "certified"
    feasible: bool
    reason: str = ""                  # why it was rejected (infeasible), or why it was certified
    acquisition: float = 0.0
    codes: list[str] = field(default_factory=list)      # the router's codes (#161)

    @property
    def error(self) -> float | None:
        """``|predicted - verified|`` of the objective, once there is a verified value and a prediction to compare."""
        if self.status != "verified" or not np.isfinite(self.predicted_objective):
            return None
        return abs(self.predicted_objective - self.objective)

    def to_dict(self) -> dict:
        return {"round": self.round, "x": [float(v) for v in self.x], "objective": float(self.objective),
                "constraints": {k: float(v) for k, v in self.constraints.items()},
                "predicted_objective": float(self.predicted_objective),
                "predicted_constraints": {k: float(v) for k, v in self.predicted_constraints.items()},
                "status": self.status, "feasible": bool(self.feasible), "reason": self.reason,
                "acquisition": float(self.acquisition), "error": self.error, "codes": list(self.codes)}


@dataclass
class Round:
    """One round of the loop: what was chosen, what the solver said, and how far the verified optimum moved."""
    index: int
    n_solver: int
    design_indices: list[int]
    best_verified: float | None
    best_x: list[float] | None
    improvement: float
    surrogate_error: float | None      # mean |predicted - verified| over this round's re-runs
    n_infeasible: int
    n_certified: int
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {**{k: getattr(self, k) for k in ("index", "n_solver", "best_verified", "improvement",
                                                 "surrogate_error", "n_infeasible", "n_certified", "seconds")},
                "design_indices": list(self.design_indices),
                "best_x": None if self.best_x is None else [float(v) for v in self.best_x]}


@dataclass
class LoopResult:
    """The whole run: every design, every round, the verified optimum and the number of solver calls it cost."""
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    constraints: list[Constraint]
    designs: list[Design]
    rounds: list[Round]
    best: Design | None
    solver_calls: int
    budget: int
    stop_reason: str
    stabilised_round: int | None
    stabilised_calls: int | None = None      # solver calls spent when the stall streak that ended the loop began
    seconds: float = 0.0
    acquisition: str = "ei"

    @property
    def verified(self) -> list[Design]:
        return [d for d in self.designs if d.status == "verified"]

    def report(self) -> dict:
        """What the budget bought — the table a reviewer reads instead of the code."""
        errors = [d.error for d in self.verified if d.error is not None]
        half = len(errors) // 2 or 1
        first = float(np.mean(errors[:half])) if errors else None
        last = float(np.mean(errors[half:])) if errors else None
        return {
            "method": "surrogate-in-the-loop with a verification budget",
            "acquisition": self.acquisition,
            "budget": int(self.budget),
            "solver_calls": int(self.solver_calls),
            "n_designs": len(self.designs),
            "n_verified": len(self.verified),
            "n_certified": sum(d.status == "certified" for d in self.designs),
            "rounds": len(self.rounds),
            "stop_reason": self.stop_reason,
            "stabilised_round": self.stabilised_round,
            "solver_calls_to_stabilise": self.stabilised_calls,
            "best": None if self.best is None else self.best.to_dict(),
            "best_x": None if self.best is None else {n: float(v) for n, v in zip(self.names, self.best.x,
                                                                                strict=True)},
            "surrogate_error_first_half": first,
            "surrogate_error_second_half": last,
            "constraints": [{"name": c.name, "limit": c.limit, "sense": c.sense} for c in self.constraints],
            "rejected": [{"round": d.round, "x": [float(v) for v in d.x], "reason": d.reason}
                         for d in self.designs if not d.feasible],
            "seconds": self.seconds,
        }

    def to_dict(self) -> dict:
        return {**self.report(), "names": self.names, "designs": [d.to_dict() for d in self.designs],
                "round_table": [r.to_dict() for r in self.rounds]}


# ---------------------------------------------------------------------------------------------- acquisition
def _expected_improvement(mean, sd, best: float, xi: float) -> np.ndarray:
    """EI of a minimisation problem: ``E[max(best - Y, 0)]`` for ``Y ~ N(mean, sd^2)``."""
    from scipy.stats import norm
    sd = np.maximum(np.asarray(sd, float), _EPS)
    z = (best - np.asarray(mean, float) - xi) / sd
    return (best - mean - xi) * norm.cdf(z) + sd * norm.pdf(z)


def _score(acquisition: str, mean: np.ndarray, sd: np.ndarray, constraints: Sequence[Constraint],
           best: float, *, xi: float, beta: float = 1.5) -> np.ndarray:
    """The acquisition score, larger is better; ``mean``/``sd`` are ``(n, k)`` (objective in column 0).

    Column 0 must already be a *minimisation* objective (the loop negates it for ``maximise=True``) and ``best`` the
    best feasible value on the same scale; the constraint columns are on their own scale and are never flipped.
    """
    mu, s = np.asarray(mean[:, 0], float), np.asarray(sd[:, 0], float)
    if acquisition == "ei":
        base = _expected_improvement(mu, s, best, xi)
    elif acquisition == "lcb":
        lcb = mu - beta * np.maximum(s, _EPS)
        base = float(np.max(lcb)) - lcb
    else:
        raise ValueError(f"acquisition must be 'ei' or 'lcb'; got {acquisition!r}")
    p = np.ones_like(base)
    for i, c in enumerate(constraints):
        p = p * c.probability(mean[:, 1 + i], sd[:, 1 + i])
    return base * p


def _pick_batch(pool: np.ndarray, score: np.ndarray, n: int, known: np.ndarray, d_min: float) -> np.ndarray:
    """The ``n`` best-scoring candidates that are at least ``d_min`` apart — from each other and from ``known``.

    The score says *where*, the distance says *not twice in the same place* (#209's rule), but as a filter on the
    score order rather than as a maximin override of it. Measured on the airfoil benchmark, replacing this with a
    maximin pick *among* the best 20 % — what :func:`~pinneapple_data.adaptive_sampling.adapt` does, and the right
    thing there because its signals plateau — cost the loop 10 % of drag: the farthest candidate in the top of the
    ranking is on the edge of the promising region, and an expected-improvement peak is exactly the thing a distance
    rule throws away. The filter keeps the peak and only refuses a duplicate.
    """
    from scipy.spatial import cKDTree

    order = np.argsort(-np.asarray(score, float), kind="stable")
    points = np.asarray(pool, float)
    tree = cKDTree(np.asarray(known, float)) if len(known) else None
    chosen: list[int] = []
    for i in order:
        if tree is not None and float(tree.query(points[i])[0]) < d_min:
            continue
        if chosen and min(float(np.linalg.norm(points[i] - points[j])) for j in chosen) < d_min:
            continue
        chosen.append(int(i))
        if len(chosen) == n:
            return np.asarray(chosen)
    for i in order:                      # nothing satisfied d_min: fill with the best of what is left
        if len(chosen) == n:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return np.asarray(chosen[:n])


def _route(mean: np.ndarray, sd: np.ndarray, constraints: Sequence[Constraint], tolerance: float | None,
           z: float = Z_INTERVAL) -> list[dict]:
    """The router's own decision (#161) for every candidate: does the interval settle the objective and the limits?

    Uses :func:`~pinneapple_analysis.uncertainty.applicability.route_cases` — the function
    :func:`~pinneapple_analysis.fidelity.serve` is built on — once for the objective (its tolerance) and once per
    constraint (its limit as a pass/fail threshold). A candidate is *certified* when no code fires: its interval is
    narrower than the tolerance asked of it and contains no limit.
    """
    n = mean.shape[0]
    lo, hi = mean - z * sd, mean + z * sd
    routes = route_cases(np.full(n, "inside"), mean[:, 0], lo[:, 0], hi[:, 0], tolerance)
    for i, c in enumerate(constraints):
        limits = (c.limit, None) if c.sense == ">=" else (None, c.limit)
        sub = route_cases(np.full(n, "inside"), mean[:, 1 + i], lo[:, 1 + i], hi[:, 1 + i], None, limits=limits)
        for a, b in zip(routes, sub, strict=True):
            a["codes"] += [f"{c.name}:{code}" for code in b["codes"]]
            a["reasons"] += [f"{c.name}: {reason}" for reason in b["reasons"]]
    return routes


# ---------------------------------------------------------------------------------------------- the loop
def _solver_outputs(solver: Callable, X: np.ndarray, k: int) -> np.ndarray:
    """Call the solver on ``X`` and normalise its answer to ``(n, k)``."""
    out = np.asarray(solver(np.atleast_2d(X)), dtype=float)
    if out.ndim == 1:
        out = out[:, None]
    if out.shape[0] != X.shape[0] and out.shape[0] == 1:
        out = out.reshape(1, -1)
    if out.shape[0] != X.shape[0]:
        raise ValueError(f"the solver returned {out.shape[0]} rows for {X.shape[0]} designs")
    if out.shape[1] != k:
        raise ValueError(f"the solver returned {out.shape[1]} outputs per design but the loop expects {k} "
                         f"(objective + {k - 1} constraints); check the constraints' order")
    return out


def loop(space, surrogate, solver: Callable, budget: int = 32, *, n_initial: int = 8, batch: int = 2,
         constraints: Sequence[Constraint] = (), acquisition: str = "ei", maximise: bool = False,
         tolerance: float | None = None, certify: bool = False, min_improve: float = 1e-3,
         patience: int = 2, max_rounds: int | None = None, n_pool: int = 1024, top_fraction: float = 0.2,
         batch_rule: str = "score", d_min: float | None = None, xi: float = 0.01, seed: int = 0,
         verbose: bool = False) -> LoopResult:
    """Optimise ``surrogate`` over ``space`` while spending at most ``budget`` calls of ``solver``. (#297)

    Parameters
    ----------
    space : the box, in any of the forms :func:`~pinneapple_data.adaptive_sampling.adapt` reads (``(lo, hi)`` pairs,
        a list of dicts with ``low``/``high``, a ``(2, d)`` array, an object with ``parameters``/``bounds``).
    surrogate : an object with ``fit(X, y)`` and ``predict(X) -> (mean, std)``, both ``(n, k)``, where ``k = 1 +
        len(constraints)`` and column 0 is the objective. :class:`GaussianProcessSurrogate` and
        :class:`EnsembleSurrogate` are provided; anything with that contract works.
    solver : ``X (n, d) -> (n, k)``, the expensive path — one call answers the objective *and* every constraint, the
        way one RANS run gives both drag and lift. With no constraints a ``(n,)`` return is accepted.
    budget : high-fidelity evaluations allowed, counting the initial design. The loop never exceeds it.
    n_initial : space-filling (scrambled Sobol') designs that open the loop — the same initial design as #209, so the
        comparison between acquisition strategies is not a comparison between starting points.
    batch : designs verified per round (the size of the parallel run a campaign would submit).
    constraints : :class:`Constraint` objects, in the order the solver returns their values after the objective.
    acquisition : ``"ei"`` (constrained expected improvement) or ``"lcb"`` (lower confidence bound x feasibility).
    maximise : the objective is to be maximised (it is negated internally; the reports keep its own sign).
    tolerance : the accuracy the decision needs, in objective units; only used with ``certify=True`` to decide which
        designs can skip the solver, through the router's own rule.
    certify : let the router (#161) accept a design without a solver run when the surrogate's 95 % interval is
        narrower than ``tolerance`` and contains no constraint limit. Certified designs are recorded as such and do
        not consume the budget; the reported optimum is always verified (one call at the end if needed).
    min_improve, patience : the stopping rule — stop after ``patience`` consecutive rounds whose verified optimum
        improved by less than ``min_improve * max(|best|, 1)``.
    max_rounds : an extra cap, for a loop that keeps finding small improvements.
    n_pool, xi : the pool of candidates scored each round, and the EI exploration jitter as a fraction of the
        objective's observed spread (``xi=0.01`` is 1 % of it; 0 is greedy).
    top_fraction : with ``batch_rule="maximin"``, the share of the pool the batch is drawn from.
    batch_rule : how the batch is chosen from the scored pool. ``"score"`` (default) takes the best-scoring
        candidates, skipping any within ``d_min`` of a point already known — #209's rule ("the score says where, the
        distance says not twice in the same place") applied as a filter. ``"maximin"`` picks by distance among the
        best ``top_fraction`` of the pool instead, which is what :func:`~pinneapple_data.adaptive_sampling.adapt`
        does and what its own signals need; on this benchmark it costs the loop about 10 % of drag.
    d_min : the spacing of the batch, in box units; the default is half the pool's own spacing,
        ``0.5 * n_pool ** (-1/d)``.
    seed : Sobol' scramble, and the surrogates' own seed is the caller's business.
    verbose : print the round table as it goes.

    Returns
    -------
    LoopResult
        ``designs`` (every design with its predicted and verified numbers), ``rounds`` (the error trend and the best so
        far), ``best`` (the verified optimum), ``solver_calls``, ``stop_reason`` and ``stabilised_round``.
    """
    t0 = time.perf_counter()
    lo, hi, names = _as_bounds(space)
    if np.any(hi <= lo):
        raise ValueError(f"every bound must be increasing; got {list(zip(lo, hi, strict=True))}")
    d = lo.size
    cons = list(constraints)
    k = 1 + len(cons)
    budget, n_initial, batch = int(budget), int(n_initial), int(batch)
    if budget < 2 or n_initial < 1 or batch < 1:
        raise ValueError("budget must be >= 2 and n_initial/batch >= 1")
    if n_initial >= budget:
        raise ValueError(f"n_initial={n_initial} leaves no budget to adapt in a budget of {budget}")
    width = hi - lo
    sign = -1.0 if maximise else 1.0

    from scipy.stats import qmc
    u0 = qmc.Sobol(d=d, scramble=True, seed=seed).random(2 ** int(np.ceil(np.log2(max(n_initial, 2)))))
    X = lo + u0[:n_initial] * width
    Y = _solver_outputs(solver, X, k)
    solver_calls, round_index, stall, stabilised = n_initial, 0, 0, None
    stall_start_calls: int | None = None

    designs: list[Design] = []
    rounds: list[Round] = []

    def _record(points: np.ndarray, rows: np.ndarray, pred: np.ndarray | None, rnd: int,
                codes: list[list[str]] | None = None, scores: np.ndarray | None = None) -> list[int]:
        """Append one verified design per row; ``pred`` is the surrogate's answer before the run (``None`` for the
        opening design, which had none)."""
        idx = []
        for i in range(rows.shape[0]):
            values = {c.name: float(rows[i, 1 + j]) for j, c in enumerate(cons)}
            failed = [c for c in cons if not c.ok(values[c.name])]
            designs.append(Design(
                round=rnd, x=[float(v) for v in points[i]],
                objective=float(rows[i, 0]), constraints=values,
                predicted_objective=float("nan") if pred is None else float(pred[i, 0]),
                predicted_constraints={c.name: (float("nan") if pred is None else float(pred[i, 1 + j]))
                                       for j, c in enumerate(cons)},
                status="verified", feasible=not failed,
                reason=failed[0].reason(values[failed[0].name]) if failed else "",
                acquisition=float(0.0 if scores is None else scores[i]),
                codes=[] if codes is None else list(codes[i]),
            ))
            idx.append(len(designs) - 1)
        return idx

    def _best_verified() -> tuple[float | None, list[float] | None]:
        feasible = [d for d in designs if d.status == "verified" and d.feasible]
        if not feasible:
            return None, None
        pick = max(feasible, key=lambda d: d.objective) if maximise else min(
            feasible, key=lambda d: d.objective)
        return pick.objective, list(pick.x)

    # the opening design, verified by construction
    idx = _record(X, Y, None, 0)
    best_value, best_x = _best_verified()
    rounds.append(Round(index=0, n_solver=int(Y.shape[0]), design_indices=idx, best_verified=best_value,
                        best_x=best_x, improvement=0.0, surrogate_error=None,
                        n_infeasible=sum(not designs[i].feasible for i in idx), n_certified=0,
                        seconds=round(time.perf_counter() - t0, 3)))

    stop_reason = "budget"
    while True:
        if solver_calls >= budget:
            stop_reason = "budget"
            break
        if max_rounds is not None and round_index + 1 > max_rounds:
            stop_reason = "max_rounds"
            break
        round_index += 1
        t_round = time.perf_counter()
        Y_fit = Y.copy()
        Y_fit[:, 0] *= sign                      # the surrogate always fits a minimisation objective
        model = surrogate.fit(X, Y_fit)
        feasible_seen = [sign * d.objective for d in designs if d.feasible]
        best_for_ei = min(feasible_seen) if feasible_seen else float(np.min(Y_fit[:, 0]))

        n = min(batch, budget - solver_calls)
        U = qmc.Sobol(d=d, scramble=True, seed=seed + 1000 + round_index).random(
            2 ** int(np.ceil(np.log2(max(n_pool, 2)))))
        pool = lo + U[:n_pool] * width
        mean, sd = model.predict(pool)
        mean, sd = np.atleast_2d(mean), np.atleast_2d(sd)
        if mean.shape[1] != k:
            raise ValueError(f"the surrogate predicted {mean.shape[1]} outputs but the loop expects {k}")
        real = mean.copy()
        real[:, 0] *= sign                       # back to the objective's own units, for the records and routing
        # EI's exploration jitter is in objective units, so it is scaled by the objective's own spread: a fixed 0.01
        # is larger than the whole range of a drag coefficient (9e-3) and turns EI into pure exploration (measured:
        # the loop then spent every round probing and finished 0.5 % from the optimum instead of on it)
        y_scale = float(np.std(Y_fit[:, 0])) or 1.0
        score = _score(acquisition, mean, sd, cons, best_for_ei, xi=xi * y_scale)
        if batch_rule == "maximin":
            n_top = max(int(np.ceil(top_fraction * pool.shape[0])), 2 * n)
            top = np.argsort(-score, kind="stable")[:n_top]
            chosen = top[greedy_maximin(pool[top], n, fixed=X)]
        elif batch_rule == "score":
            chosen = _pick_batch(pool, score, n, X, d_min if d_min is not None else 0.5 * (1.0 / n_pool) ** (1.0 / d))
        else:
            raise ValueError(f"batch_rule must be 'score' or 'maximin'; got {batch_rule!r}")
        candidates = pool[chosen]
        routes = _route(real, sd, cons, tolerance)
        certified = [bool(certify and tolerance is not None and not routes[j]["reasons"]) for j in chosen]

        run = [j for j, cert in enumerate(certified) if not cert]
        new_idx: list[int] = []
        if run:
            rows = _solver_outputs(solver, candidates[run], k)
            solver_calls += len(run)
            X = np.vstack([X, candidates[run]])
            Y = np.vstack([Y, rows])
            new_idx += _record(candidates[run], rows, real[chosen][run], round_index,
                               codes=[routes[j]["codes"] for j in run], scores=score[chosen][run])
        for j, cert in enumerate(certified):
            if not cert:
                continue
            values = {c.name: float(real[chosen[j], 1 + q]) for q, c in enumerate(cons)}
            failed = [c for c in cons if not c.ok(values[c.name])]
            designs.append(Design(
                round=round_index, x=[float(v) for v in candidates[j]],
                objective=float(real[chosen[j], 0]), constraints=values,
                predicted_objective=float(real[chosen[j], 0]), predicted_constraints=dict(values),
                status="certified", feasible=not failed,
                reason=("certified by the router: " + ", ".join(routes[j]["codes"])) if not failed
                else failed[0].reason(values[failed[0].name]),
                acquisition=float(score[chosen[j]]), codes=list(routes[j]["codes"])))
            new_idx.append(len(designs) - 1)

        previous_best = best_value
        best_value, best_x = _best_verified()
        improvement = 0.0 if (previous_best is None or best_value is None) else abs(best_value - previous_best)
        # a round that finds the first feasible design has not "stopped moving": the stall counter only runs once
        # there is a verified feasible optimum to fail to improve on
        found_first = previous_best is None and best_value is not None
        errors = [designs[i].error for i in new_idx]
        errors = [e for e in errors if e is not None]
        rounds.append(Round(index=round_index, n_solver=len(run), design_indices=new_idx,
                            best_verified=best_value, best_x=best_x, improvement=float(improvement),
                            surrogate_error=(float(np.mean(errors)) if errors else None),
                            n_infeasible=sum(not designs[i].feasible for i in new_idx),
                            n_certified=sum(designs[i].status == "certified" for i in new_idx),
                            seconds=round(time.perf_counter() - t_round, 3)))
        # the stopping rule is relative to the objective's own magnitude: a drag of 9e-3 that moves by 1e-5 has
        # moved, and an absolute threshold would call that "stopped" (measured: with an absolute 1e-3 the loop
        # stopped at round 2 on the airfoil case while it was still improving by 2 % per round)
        scale = max(abs(best_value) if best_value is not None else 0.0,
                    abs(previous_best) if previous_best is not None else 0.0, 1e-12)
        stalled_now = not found_first and improvement <= min_improve * scale
        if stalled_now and stall == 0:
            stall_start_calls = solver_calls       # the optimum has been standing still since this round
        stall = 0 if found_first else (stall + 1 if stalled_now else 0)
        if verbose:
            print(f"  round {round_index:2d}  solver+{len(run)}  best {best_value}  (+{improvement:.3g})  "
                  f"err {rounds[-1].surrogate_error}  infeasible {rounds[-1].n_infeasible}  "
                  f"certified {rounds[-1].n_certified}", flush=True)
        if stall >= patience:
            stabilised = round_index - patience + 1
            stop_reason = "stabilised"
            break

    # the reported optimum is always a verified number: certify=False guarantees it, certify=True may not
    best_design = None
    feasible_verified = [d for d in designs if d.feasible and d.status == "verified"]
    if feasible_verified:
        best_design = max(feasible_verified, key=lambda x: x.objective) if maximise else min(
            feasible_verified, key=lambda x: x.objective)
    else:
        certified_feasible = [d for d in designs if d.feasible]
        if certified_feasible and solver_calls < budget:
            pick = max(certified_feasible, key=lambda x: x.objective) if maximise else min(
                certified_feasible, key=lambda x: x.objective)
            rows = _solver_outputs(solver, np.asarray([pick.x], float), k)
            solver_calls += 1
            pick.status, pick.objective = "verified", float(rows[0, 0])
            pick.constraints = {c.name: float(rows[0, 1 + j]) for j, c in enumerate(cons)}
            best_design = pick

    return LoopResult(names=names, lo=lo, hi=hi, constraints=cons, designs=designs, rounds=rounds,
                      best=best_design, solver_calls=solver_calls, budget=budget, stop_reason=stop_reason,
                      stabilised_round=stabilised,
                      stabilised_calls=stall_start_calls if stabilised is not None else None,
                      seconds=round(time.perf_counter() - t0, 3),
                      acquisition=acquisition)


# ---------------------------------------------------------------------------------------------- the benchmark
AIRFOIL_RE = 3.0e6
AIRFOIL_ALPHA = 4.0
AIRFOIL_M_BOUNDS = (0.0, 0.06)
AIRFOIL_P_BOUNDS = (0.2, 0.6)
AIRFOIL_T_BOUNDS = (0.08, 0.18)
AIRFOIL_SPACE = [{"name": "camber", "low": AIRFOIL_M_BOUNDS[0], "high": AIRFOIL_M_BOUNDS[1]},
                 {"name": "camber_position", "low": AIRFOIL_P_BOUNDS[0], "high": AIRFOIL_P_BOUNDS[1]},
                 {"name": "thickness", "low": AIRFOIL_T_BOUNDS[0], "high": AIRFOIL_T_BOUNDS[1]}]
AIRFOIL_CL_TARGET = 0.75


def _section_shape(m: float, p: float, t_c: float, n_pts: int = 121) -> np.ndarray:
    """CST weights of a NACA 4-digit section from ``(m, p, t_c)`` — the parameterisation of the design study in
    :mod:`pinneapple_design.design_optimizer.airfoil_drag` — fitted back from the surface points, so the panel method
    and that study see the same sections."""
    from pinneapple_design.aero.geometry import fit_cst
    from pinneapple_design.geometry.gen.airfoil import naca_parametric

    pts = naca_parametric(m=float(m), p=float(p), t_c=float(t_c), n_pts=n_pts).detach().cpu().numpy()
    i_te = int(np.argmax(pts[:, 0]))                      # upper surface LE -> TE, then lower surface TE -> LE
    upper, lower = pts[: i_te + 1], pts[i_te:]
    return fit_cst(upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1])


def airfoil_solver(X, *, alpha_deg: float = AIRFOIL_ALPHA, reynolds: float = AIRFOIL_RE,
                   cl_target: float = AIRFOIL_CL_TARGET, n_per_side: int = 80) -> np.ndarray:
    """The high-fidelity 2-D section evaluator: ``(camber, camber_position, thickness) -> [Cd, Cl]``.

    ``Cl`` comes from the Hess-Smith panel method (:func:`~pinneapple_design.aero.panel.panel_solve`) with the Kutta
    condition, so camber *and* thickness move the lift, and the route through 121 surface points and a CST fit is the
    same one the design study of :mod:`pinneapple_design.design_optimizer.airfoil_drag` uses. The drag is the Hoerner
    profile drag of that thickness at ``reynolds`` — the potential solver itself has no drag to give (d'Alembert). It
    is the honest shape of a cheap high-fidelity model: exact on the lift side, a boundary-layer correlation on the
    drag side, and about 4e3 times dearer than one design of the GP surrogate the loop reads it through (measured on
    this machine: 6.3 ms against 1 us per design when the pool is predicted in a single call).

    Returns an ``(n, 2)`` array of ``[Cd, Cl]`` — objective first, so the signature is the one :func:`loop` wants with
    ``constraints=[Constraint("cl", cl_target)]``.
    """
    from pinneapple_design.aero.panel import panel_solve
    from pinneapple_physics.closed_form.thin_airfoil import hoerner_profile_cd

    X = np.atleast_2d(np.asarray(X, float))
    out = np.zeros((X.shape[0], 2))
    for i, (m, p, t_c) in enumerate(X):
        res = panel_solve(_section_shape(m, p, t_c), alpha_deg=alpha_deg, n_per_side=n_per_side, close_te=True)
        out[i] = hoerner_profile_cd(float(t_c), reynolds), float(res.cl_kutta_joukowski)
    return out


def airfoil_reference(*, reynolds: float = AIRFOIL_RE, t_min: float = AIRFOIL_T_BOUNDS[0]) -> float:
    """The reference optimum of the airfoil problem: the profile drag of the thinnest section the box allows.

    The panel method's lift is the only thing camber buys, and the lift floor (:data:`AIRFOIL_CL_TARGET`) is met by a
    modest camber at that thickness, so the constrained optimum is the corner ``t = t_min`` at whichever camber
    satisfies the constraint — ``Cd`` depends on the thickness alone. Known in closed form, so the benchmark compares
    the loop against the truth and not against the best of a finer sweep.
    """
    from pinneapple_physics.closed_form.thin_airfoil import hoerner_profile_cd

    return hoerner_profile_cd(float(t_min), reynolds)


def benchmark_loop(budget: int = 16, *, n_initial: int = 6, batch: int = 2, repeats: int = 2,
                   acquisition: str = "ei", surrogate: str = "gp", seed: int = 0,
                   cl_target: float = AIRFOIL_CL_TARGET, n_per_side: int = 80) -> dict:
    """The loop against a fixed design of the same solver budget, on the 2-D airfoil of :func:`airfoil_solver`. (#297)

    Both arms spend exactly ``budget`` solver calls. The loop spends ``n_initial`` on a Sobol' design and the rest
    through the acquisition; the baseline spends all of them on the same scrambled Sobol' sequence — the fixed design
    #209's benchmark uses, so neither arm wins by a better starting set. ``repeats`` runs with different seeds;
    ``surrogate`` is ``"gp"`` (:class:`GaussianProcessSurrogate`) or ``"ensemble"`` (:class:`EnsembleSurrogate`).

    Returns the verified best ``Cd`` meeting ``Cl >= cl_target`` for each arm, the gap of each to
    :func:`airfoil_reference`, ``gain`` as ``fixed / loop`` (below 1 the loop wins), and
    ``solver_calls_to_stabilise`` — the number the issue asks for: how many expensive evaluations the loop spent
    before its verified optimum stopped moving.
    """
    lo = np.array([b["low"] for b in AIRFOIL_SPACE])
    hi = np.array([b["high"] for b in AIRFOIL_SPACE])
    cons = [Constraint("cl", cl_target)]
    picks: dict[str, list] = {"loop": [], "fixed": [], "loop_stab": [], "loop_calls": [], "loop_err": [],
                              "loop_rounds": [], "loop_calls_stab": []}
    for r in range(int(repeats)):
        if surrogate == "gp":
            model = GaussianProcessSurrogate(len(lo), 2, seed=seed + r)
        elif surrogate == "ensemble":
            model = EnsembleSurrogate(len(lo), 2, n_members=4, hidden=24, epochs=300, seed=seed + r)
        else:
            raise ValueError("surrogate must be 'gp' or 'ensemble'")
        res = loop(AIRFOIL_SPACE, model, airfoil_solver, budget=budget, n_initial=n_initial, batch=batch,
                   constraints=cons, acquisition=acquisition, seed=seed + r, n_pool=512, patience=3)
        picks["loop"].append(res.best.objective if res.best else float("inf"))
        picks["loop_stab"].append(res.stabilised_round)
        picks["loop_calls"].append(res.solver_calls)
        picks["loop_calls_stab"].append(res.stabilised_calls)
        picks["loop_rounds"].append(len(res.rounds))
        errs = [d.error for d in res.verified if d.error is not None]
        picks["loop_err"].append(float(np.mean(errs)) if errs else None)

        from scipy.stats import qmc
        U = qmc.Sobol(d=len(lo), scramble=True, seed=seed + r).random(2 ** int(np.ceil(np.log2(max(budget, 2)))))
        Yf = airfoil_solver(lo + U[:budget] * (hi - lo))
        feasible = [float(Yf[i, 0]) for i in range(len(Yf)) if Yf[i, 1] >= cl_target]
        picks["fixed"].append(min(feasible) if feasible else float("inf"))

    reference = airfoil_reference()
    loop_best = float(np.mean(picks["loop"]))
    fixed_best = float(np.mean(picks["fixed"]))
    return {
        "method": "surrogate-in-the-loop against a fixed design at the same solver budget",
        "budget": int(budget), "repeats": int(repeats), "acquisition": acquisition, "surrogate": surrogate,
        "cl_target": float(cl_target), "reference_cd": reference,
        "loop": {"best_cd": loop_best, "per_repeat": picks["loop"],
                 "gap_to_reference_pct": 100.0 * (loop_best - reference) / reference,
                 "solver_calls_to_stabilise": picks["loop_calls_stab"], "solver_calls": picks["loop_calls"],
                 "rounds": picks["loop_rounds"], "surrogate_error": picks["loop_err"]},
        "fixed": {"best_cd": fixed_best, "per_repeat": picks["fixed"],
                  "gap_to_reference_pct": 100.0 * (fixed_best - reference) / reference},
        "gain": fixed_best / loop_best if loop_best > 0 else float("inf"),
    }
