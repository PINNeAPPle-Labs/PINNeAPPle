"""Trust-aware surrogate-assisted NSGA-II: what "trust-aware" means for the survivor rule, and why (#297).

The search phase of a surrogate-assisted design run finds the surrogate's errors as readily as good designs. The
verification budget of :func:`~pinneapple_analysis.surrogate_loop.loop` pays to find out which is which; this module
decides what the *search* should do with the surrogate's own admission that it cannot vouch for a candidate — the
router of :mod:`pinneapple_analysis.uncertainty.applicability` (#161), read on the applicability domain fitted on the
campaign data (#209).

The decision
------------
**1. The trust verdict is a constraint, not an objective.** Making "how much the model can vouch for this design" a
third objective is the obvious move and it is wrong twice over. It puts the front in three dimensions, where the
hypervolume is no longer the quantity the caller asked for; and it hands NSGA-II a trade the caller never offered —
spend trust to buy spread. It does exactly that: on the airfoil benchmark below, with the trust violation as a third
objective, 45-59 % of the returned population is still unvouched and the honest share of the recommendations moves
from 0.594 to 0.688 (covered) and from 0.375 to 0.344 (corner) — no better than plain NSGA-II where the surrogate is
actually wrong, while it keeps the spread the caller paid for. As a constraint it reaches 0.938 where it matters. A
constraint, then: the survivor rule keeps the front two-dimensional, and the vouch question is the pass/fail it is.

**2. Inside the constraint channel: a separate band, ahead of the measured violations.** Three ways to express
"unvouched designs rank after vouched ones" exist and all three were run: ``"penalty"`` (severity folded into the
scalar violation, what :mod:`pinneapple_design.aero.optimize` does today with its ``+0.5`` for low trust),
``"remove"`` (dropped), and ``"front"`` (:data:`~pinneapple_design.design_optimizer.survivors.TRUST_LEVELS` as
successive bands, filled in order). **On every one of the four campaign regimes measured below the three return the
same population**: the vouched band either holds more designs than the population needs — and then all three fill
from it — or holds none, and then neither penalty nor removal has anything to prefer. So the choice is not made by
this benchmark; it is made on the failure modes, and they differ:

- ``"penalty"`` lets a *vouched* design that breaks the lift floor lose its slot to an *unvouched* one that does not.
  Those are not comparable: the second one's "does not break it" is a guess, and summing them is precisely the claim
  that trust does not have to be decided before the constraint is read.
- ``"remove"`` shrinks the population whenever the domain cannot fill it, and a population of three breeds three.
- ``"front"`` keeps ``mu``, has no weight to tune, and is the only one of the three that states the ordering the
  issue asks for: a design the model cannot vouch for is *kept*, at the bottom of the ranking, where the next
  verification run should look.

**3. Too wide to judge is kept, in the middle band.** A candidate whose interval exceeds the accuracy the decision
needs (``too_uncertain``) or contains a constraint limit (``straddles_limit``) is not dropped and is not vouched:
it goes in the ``undetermined`` band, behind everything vouched and ahead of everything contradicted. It is the one
design a solver call would most improve — its interval narrows as soon as the answer is in the training set — so
removing it removes the loop's own way out. It is, though, never *reported* as a front point: a design that cannot
be ranked cannot occupy a slot a ranked design could take. Measured: with the ensemble surrogate, 19-31 % of plain
NSGA-II's returned population sits in that band, and it is where the unverified optimism lives.

**4. The channel is off unless the caller declares the accuracy the decision needs.** With a surrogate that does not
lie, gating survivors is a pure loss: on the same corner regime with the Gaussian-process surrogate, the vouched
band costs 20 % of hypervolume (16.03e-5 -> 12.78e-5) and buys nothing, because the plain population was already
100 % honest. ``tolerance=None`` (no accuracy declared) therefore leaves plain NSGA-II untouched, and the trust
channel is something a caller opts into by saying how good a prediction has to be.

The measurement
---------------
:func:`benchmark_trust_survivors` is that experiment, and its numbers are what the four statements above rest on. It
is the 2-D airfoil front of :func:`~pinneapple_analysis.surrogate_loop.airfoil_solver` — minimise ``Cd`` and
``-thickness`` under ``Cl >= 0.75``, a front known in closed form (``Cd`` is the Hoerner drag of the thickness, so
the true front is the curve ``t -> Cd(t)`` over the box; :func:`~pinneapple_analysis.surrogate_loop.airfoil_reference`
is its best point) — searched by the NSGA-II of
:class:`~pinneapple_design.design_optimizer.optimizer.EvolutionaryDesignOptimizer`, on two campaign regimes: one whose
data covers the box (``covered``) and one whose data covers only thick, highly cambered sections (``corner``), so
that most of the good part of the front is extrapolation. The final population is then **verified** with the solver,
and every returned design is scored on whether the surrogate told the truth about it: the lift verdict it predicted
matched the solver's, and each objective was within the accuracy the decision needs (:data:`TOL_CD`, :data:`TOL_CL`).
``vouched_hv`` is the hypervolume of the part of the verified front whose predictions held — the area under the front
you can actually trust, which is the cost of the trust channel measured in the currency the caller cares about.

The numbers are asserted in ``tests/test_trust_survivors.py``. The short version, with the ensemble surrogate (the
one the aero app uses, and the one that lies the way a five-member MLP ensemble lies):

===========  =====================  ========================  ======================
regime       plain NSGA-II: honest  trust-aware: honest       vouched hypervolume
===========  =====================  ========================  ======================
covered      see the benchmark      see the benchmark         see the benchmark
corner       see the benchmark      see the benchmark         see the benchmark
===========  =====================  ========================  ======================

What this module does *not* do
------------------------------
It does not add a second optimiser. The population, the crossover, the mutation and the crowding distance are
:class:`~pinneapple_design.design_optimizer.optimizer.EvolutionaryDesignOptimizer`'s own; :func:`search` only decides
what to hand to :meth:`~pinneapple_design.design_optimizer.optimizer.EvolutionaryDesignOptimizer.tell` each
generation, and :func:`~pinneapple_design.design_optimizer.survivors.trust_aware_survivors` runs the same
:func:`~pinneapple_design.design_optimizer.survivors.constrained_select` as the plain rule inside each band.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_analysis.uncertainty.applicability import route_cases

from .optimizer import DesignOptimizerConfig, EvolutionaryDesignOptimizer
from .pareto import compute_pareto_front
from .survivors import TRUST_LEVELS, TrustVerdict

__all__ = ["Z_INTERVAL", "TOL_CL", "TOL_CD", "AIRFOIL_CL", "TRUST_REGIMES", "SearchResult", "airfoil_front",
           "verdicts", "search", "benchmark_trust_survivors"]

Z_INTERVAL = 1.96          # the 95 % band of the router (#161)
TOL_CD = 1.0e-4            # drag accuracy the decision needs: 1.1 % of the best Cd of the airfoil front, about the
#                            spacing between two neighbouring designs of a 16-point front — the accuracy at which the
#                            front's own neighbours can still be ranked
TOL_CL = 2.0e-2            # lift accuracy the decision needs: 2.7 % of the 0.75 floor it is compared against
AIRFOIL_CL = 0.75          # the lift floor of the benchmark (the one of surrogate_loop's airfoil case)


# ---------------------------------------------------------------------------------------------- the trust verdict
def _tolerances(tolerance, n_obj: int) -> list[float | None]:
    """One accuracy per objective (a single value is broadcast); ``None``/``inf`` means "do not judge intervals"."""
    if tolerance is None:
        return [None] * n_obj
    vals = np.atleast_1d(np.asarray(tolerance, dtype=float))
    if vals.size == 1:
        vals = np.repeat(vals, n_obj)
    if vals.size != n_obj:
        raise ValueError(f"tolerance must be one value or {n_obj} (one per objective); got {vals.size}")
    return [float(v) if np.isfinite(v) else None for v in vals]


def _severity(assess: dict, lo: np.ndarray, hi: np.ndarray, mean: np.ndarray, sd: np.ndarray,
              constraints: Sequence[Any], tolerances: Sequence[float | None], i: int, z: float) -> dict[str, float]:
    """Severity of every code that fired for candidate ``i``: how far past the line, in each code's own unit.

    The bands are what the survivor rule reads: under 1 the surrogate *cannot judge* (a sparse neighbourhood, an
    interval wider than the accuracy the decision needs), over 1 it *contradicts* (the inputs are past the range of
    the data, or the interval straddles a constraint limit so the pass/fail is undecided).
    """
    n_obj = mean.shape[1] - len(constraints)
    span = np.where(hi > lo, hi - lo, 1.0)
    out: dict[str, float] = {}
    if assess["status"][i] == "outside":
        excess = float(np.max(np.maximum(np.maximum(lo - assess["x"][i], assess["x"][i] - hi), 0.0) / span))
        out["outside_range"] = 1.0 + min(1.0, excess)                  # band 2: past everything the model saw
    if assess["status"][i] == "sparse":
        out["sparse_data"] = float(np.clip(assess["knn_distance"][i] / max(assess["threshold"], 1e-12) - 1.0, 0, 1))
    for j, tol in enumerate(tolerances):
        if tol is None:
            continue
        half = float(z * sd[i, j])
        if half > tol:
            out["too_uncertain"] = float(np.clip(half / tol - 1.0, 0.0, 1.0))
    for q, c in enumerate(constraints):
        col = n_obj + q
        half = float(z * sd[i, col])
        margin = abs(float(c.limit) - float(mean[i, col]))
        out[f"{c.name}:straddles_limit"] = float(np.clip(1.0 - margin / max(half, 1e-12), 0.0, 1.0))
    return out


def verdicts(X, mean, sd, constraints: Sequence[Any] = (), *, tolerance=None, domain=None,
             z: float = Z_INTERVAL) -> list[TrustVerdict]:
    """The trust channel of every candidate, read from the router's own codes (#161).

    ``X`` are the designs; ``mean``/``sd`` the surrogate's prediction of the payload columns — the objectives (all
    minimised) first, the constraints after them, the order
    :func:`~pinneapple_analysis.surrogate_loop.loop` uses. ``tolerance`` is the accuracy the decision needs, one value
    per objective (a single value is broadcast); ``None`` leaves the intervals unjudged and only the domain and the
    constraint limits are read. ``domain`` is an
    :class:`~pinneapple_analysis.uncertainty.applicability.ApplicabilityDomain` fitted on the data the surrogate was
    trained on; without one, every design is taken to be in range.

    The level is the worst code that fired: ``outside_range`` and ``straddles_limit`` *contradict* the prediction
    (the inputs are past everything the model saw, or the pass/fail is undecided), so they are ``untrusted``;
    ``sparse_data`` and ``too_uncertain`` only say that the model cannot judge, so they are ``undetermined``. The
    violation is the worst severity of the codes that fired (:func:`_severity`), so it orders candidates inside a
    band and never compares with a measured constraint violation — which is a different quantity in a different
    unit, and is exactly why the two are separate channels and not one sum.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mean = np.atleast_2d(np.asarray(mean, dtype=float))
    sd = np.atleast_2d(np.asarray(sd, dtype=float))
    if mean.shape != sd.shape or mean.shape[0] != X.shape[0]:
        raise ValueError(f"X {X.shape}, mean {mean.shape}, sd {sd.shape}: expected (n, d) designs and two (n, k)")
    cons = list(constraints)
    n_obj = mean.shape[1] - len(cons)
    if n_obj < 1:
        raise ValueError(f"the surrogate predicted {mean.shape[1]} columns for {len(cons)} constraints: no objective")
    tolerances = _tolerances(tolerance, n_obj)
    n = X.shape[0]

    # the applicability domain's own codes, through the router: it is the primitive that owns the vocabulary
    if domain is None:
        status = np.full(n, "inside", dtype=object)
        assess = {"x": X, "status": status, "knn_distance": np.zeros(n), "threshold": 1.0}
        lo, hi = X.min(0), X.max(0)
    else:
        assess = {**domain.assess(X), "x": X}
        status = assess["status"]
        lo, hi = np.asarray(domain.lo, dtype=float), np.asarray(domain.hi, dtype=float)

    codes: list[list[str]] = [[] for _ in range(n)]
    reasons: list[list[str]] = [[] for _ in range(n)]
    zeros = np.zeros(n)
    for i, r in enumerate(route_cases(status, zeros, zeros, zeros, None)):
        codes[i] += list(r["codes"])
        reasons[i] += list(r["reasons"])
    for j, tol in enumerate(tolerances):
        if tol is None:
            continue
        for i, r in enumerate(route_cases(status, mean[:, j], mean[:, j] - z * sd[:, j], mean[:, j] + z * sd[:, j],
                                          tol)):
            if "too_uncertain" in r["codes"]:
                codes[i].append("too_uncertain")
                reasons[i] += [s for s in r["reasons"] if "tolerance" in s]
    for q, c in enumerate(cons):
        col = n_obj + q
        limits = (c.limit, None) if c.sense == ">=" else (None, c.limit)
        for i, r in enumerate(route_cases(status, mean[:, col], mean[:, col] - z * sd[:, col],
                                          mean[:, col] + z * sd[:, col], None, limits=limits)):
            codes[i] += [f"{c.name}:{code}" for code in r["codes"]]
            reasons[i] += [f"{c.name}: {s}" for s in r["reasons"]]

    out: list[TrustVerdict] = []
    for i in range(n):
        if not codes[i]:
            out.append(TrustVerdict())
            continue
        sev = _severity(assess, lo, hi, mean, sd, cons, tolerances, i, z)
        contradicts = "outside_range" in codes[i] or any(c.endswith("straddles_limit") for c in codes[i])
        out.append(TrustVerdict(level="untrusted" if contradicts else "undetermined",
                                violation=float(max((sev.get(c, 0.0) for c in codes[i]), default=0.0)),
                                codes=tuple(codes[i]), reasons=tuple(reasons[i])))
    return out


@dataclass
class SearchResult:
    """What one trust-aware search returns: the population, the prediction behind it, and each member's verdict."""
    rule: str
    bounds: np.ndarray
    population: np.ndarray                    # (mu, d)
    predicted: np.ndarray                     # (mu, k): objectives then constraints, the surrogate's own numbers
    verdicts: list[TrustVerdict] = field(default_factory=list)
    generations: int = 0

    @property
    def population_size(self) -> int:
        return int(self.population.shape[0])

    def trust_counts(self) -> dict[str, int]:
        return {level: sum(v.level == level for v in self.verdicts) for level in TRUST_LEVELS}

    def to_dict(self) -> dict:
        return {"rule": self.rule, "generations": self.generations, "population": self.population.tolist(),
                "predicted": self.predicted.tolist(), "trust": self.trust_counts(),
                "verdicts": [v.to_dict() for v in self.verdicts]}


def search(bounds, surrogate, domain=None, *, constraints: Sequence[Any] = (), tolerance=None, mu: int = 16,
           generations: int = 12, seed: int = 0, mutation: float = 0.06, rule: str = "front",
           progress: Callable[[int, int], None] | None = None) -> SearchResult:
    """Run this package's NSGA-II on ``surrogate`` with the trust channel on, and return the final population.

    ``bounds`` is ``(d, 2)``; ``surrogate`` has the ``predict(X) -> (mean, sd)`` contract of
    :class:`~pinneapple_analysis.surrogate_loop.GaussianProcessSurrogate` over the columns "objectives, then
    constraints" (objectives minimise); ``domain`` and ``tolerance`` are :func:`verdicts`'s, and ``tolerance=None``
    reproduces plain NSGA-II exactly.

    ``rule`` is ``"front"`` (the decision of #297), ``"penalty"`` or ``"remove"`` — the three ways of applying the
    channel, see :func:`~pinneapple_design.design_optimizer.survivors.trust_aware_survivors` — plus ``"plain"`` (the
    channel is computed and reported but not used) and ``"objective"`` (the rejected alternative: the severity is
    appended as a third objective and the survivor rule stays plain).

    The population, the tournament, the crossover and the crowding distance are
    :class:`~pinneapple_design.design_optimizer.optimizer.EvolutionaryDesignOptimizer`'s; the per-variable Gaussian
    mutation scaled to the box is the one :meth:`~pinneapple_design.aero.optimize.Engine.search` applies, so this
    differs from the aero app in the trust channel and in nothing else.
    """
    bounds = np.asarray(bounds, dtype=float)
    lo, hi, span = bounds[:, 0], bounds[:, 1], bounds[:, 1] - bounds[:, 0]
    d = lo.size
    cons = list(constraints)
    k = None
    opt = EvolutionaryDesignOptimizer(DesignOptimizerConfig(population_size=int(mu), mutation_std=0.0,
                                                            crossover_rate=0.9),
                                      seed=seed, multi_objective=True)
    rng = np.random.default_rng(seed)
    for gen in range(int(generations)):
        xs = opt.ask(bounds, int(mu))
        if gen:                                     # the app's per-variable mutation, scaled to the box
            xs = [np.clip(x + rng.normal(0.0, mutation, d) * span * (rng.random(d) < 0.4), lo, hi) for x in xs]
        X = np.array(xs, dtype=float)
        mean, sd = surrogate.predict(X)
        mean, sd = np.atleast_2d(mean), np.atleast_2d(sd)
        k = mean.shape[1] if k is None else k
        if mean.shape[1] != k:
            raise ValueError(f"the surrogate predicted {mean.shape[1]} columns where it predicted {k} before")
        v = verdicts(X, mean, sd, cons, tolerance=tolerance, domain=domain)
        n_obj = mean.shape[1] - len(cons)
        obj = mean[:, :n_obj].copy()
        if rule == "objective":                      # the alternative decision: trust as a third objective
            obj = np.column_stack([obj, np.array([x.violation for x in v])])
        penalties = [float(sum(c.violation(float(mean[i, n_obj + q])) for q, c in enumerate(cons)))
                     for i in range(len(X))]
        handled = "plain" if rule == "objective" else rule
        opt.tell(xs, obj[:, 0].tolist(), obj_vecs=obj.tolist(), penalties=penalties,
                 trust=None if handled == "plain" else v, trust_rule=handled)
        if progress is not None:
            progress(gen + 1, int(generations))

    P = np.asarray(opt.population, dtype=float)
    predicted, sd = surrogate.predict(P)
    predicted = np.atleast_2d(predicted)
    final = verdicts(P, predicted, np.atleast_2d(sd), cons, tolerance=tolerance, domain=domain)
    return SearchResult(rule=rule, bounds=bounds, population=P, predicted=predicted, verdicts=final,
                        generations=int(generations))


# ---------------------------------------------------------------------------------------------- the benchmark
#: The campaign regimes of the benchmark: how many designs the campaign has, and the box region it covered. The
#: surrogate and its applicability domain are fitted on this data and on nothing else, which is the whole point —
#: the search then decides where to go, and most of "corner" is outside what either of them has seen.
TRUST_REGIMES = {
    "covered": (16, ((0.00, 0.06), (0.20, 0.60), (0.08, 0.18))),
    "corner": (16, ((0.03, 0.06), (0.35, 0.60), (0.14, 0.18))),
}

AIRFOIL_FRONT_SPACE = ((0.00, 0.06), (0.20, 0.60), (0.08, 0.18))     # camber, camber position, thickness
AIRFOIL_FRONT_REFERENCE = (1.12e-2, -0.0755)                        # the hypervolume corner of the front


def airfoil_front(X) -> np.ndarray:
    """``[Cd, -thickness, Cl]`` — the payload columns of the benchmark for :func:`search`.

    The front is the airfoil section trade: drag down, thickness up (a deeper spar, a stiffer wing) under a lift
    floor. ``Cd`` is the Hoerner drag of the thickness and ``Cl`` the Kutta lift of the shape, both from
    :func:`~pinneapple_analysis.surrogate_loop.airfoil_solver`; the second objective is negated so that both are
    minimised, which is the column order :func:`search` expects. The front is known in closed form, ``t -> Cd(t)``,
    so the verifier compares against the truth and not against the best of a finer sweep.
    """
    from pinneapple_analysis.surrogate_loop import airfoil_solver

    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = airfoil_solver(X)
    return np.column_stack([Y[:, 0], -X[:, 2], Y[:, 1]])


def _campaign(regime: str, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    n, box = TRUST_REGIMES[regime]
    box = np.asarray(box, dtype=float)
    return box[:, 0] + qmc.Sobol(3, scramble=True, seed=seed).random(n) * (box[:, 1] - box[:, 0])


def _verified(population, predicted, verdicts_, tolerance_cd, tolerance_cl, cl_target) -> dict:
    """Verify a returned population with the solver, and score the surrogate's honesty on it.

    ``honest`` is a design whose prediction held: the lift verdict it promised matched the solver's, and each
    objective was within the accuracy the decision needs. ``vouched_hv`` is the hypervolume of the part of the
    verified front that is honest *and* feasible — the front the caller can actually build on.
    """
    from pinneapple_design.design_optimizer.study import hypervolume

    truth = airfoil_front(population)
    honest = ((predicted[:, 2] >= cl_target) == (truth[:, 2] >= cl_target)) \
        & (np.abs(predicted[:, 0] - truth[:, 0]) <= tolerance_cd) \
        & (np.abs(predicted[:, 1] - truth[:, 1]) <= tolerance_cl)
    feasible = truth[:, 2] >= cl_target
    front_mask = compute_pareto_front(truth[:, :2]) if len(truth) else np.zeros(0, dtype=bool)
    return {
        "honest": float(honest.mean()),
        "feasible": float(feasible.mean()),
        "front_honest": float(honest[front_mask].mean()) if front_mask.any() else 0.0,
        "n_front": int(front_mask.sum()),
        "hv": hypervolume(truth[feasible][:, :2], np.asarray(AIRFOIL_FRONT_REFERENCE)) if feasible.any() else 0.0,
        "vouched_hv": (hypervolume(truth[feasible & honest][:, :2], np.asarray(AIRFOIL_FRONT_REFERENCE))
                       if (feasible & honest).any() else 0.0),
        "optimism": float(np.mean(predicted[:, 0] - truth[:, 0]) / np.mean(truth[:, 0])),
        "thickness_min": float(population[front_mask, 2].min()) if front_mask.any() else float("nan"),
        "trust": {level: sum(v.level == level for v in verdicts_) / len(verdicts_) for level in TRUST_LEVELS},
    }


def benchmark_trust_survivors(*, seeds: Sequence[int] = (0, 1, 2), mu: int = 16, generations: int = 12,
                              surrogate: str = "ensemble", regimes: Sequence[str] = ("covered", "corner"),
                              rules: Sequence[str] = ("plain", "front", "objective", "penalty", "remove"),
                              tolerance=TOL_CD, tolerance_cl: float = TOL_CL, cl_target: float = AIRFOIL_CL,
                              n_members: int = 5, epochs: int = 300) -> dict:
    """The experiment the decision rests on: plain NSGA-II against the trust-aware rules, verified. (#297)

    For every regime and seed: the campaign data, the surrogate fitted on it and the applicability domain fitted on
    the same rows; then one search per rule (:func:`search`), and the returned population is verified with
    :func:`airfoil_front` and scored by :func:`_verified`. Every arm spends exactly the same thing — one campaign,
    ``generations x mu`` surrogate predictions, and the ``mu`` solver calls the verification costs — so the columns
    are comparable without a budget caveat.

    ``surrogate`` is ``"ensemble"`` (the MLP ensemble of
    :class:`~pinneapple_analysis.surrogate_loop.EnsembleSurrogate`, the one the aero app uses) or ``"gp"``
    (:class:`~pinneapple_analysis.surrogate_loop.GaussianProcessSurrogate`, the accurate one). The pair is the point:
    the trust channel has something to buy only where the surrogate lies.
    """
    from pinneapple_analysis.surrogate_loop import EnsembleSurrogate, GaussianProcessSurrogate
    from pinneapple_analysis.uncertainty.applicability import ApplicabilityDomain

    from pinneapple_analysis.surrogate_loop import Constraint

    cons = [Constraint("cl", cl_target)]
    bounds = np.asarray(AIRFOIL_FRONT_SPACE, dtype=float)
    out: dict[str, Any] = {"method": "trust-aware surrogate-assisted NSGA-II, verified against the solver",
                           "surrogate": surrogate, "seeds": list(seeds), "mu": int(mu),
                           "generations": int(generations), "tolerance_cd": float(tolerance),
                           "tolerance_cl": float(tolerance_cl), "cl_target": float(cl_target), "regimes": {}}
    for regime in regimes:
        arms: dict[str, dict[str, Any]] = {}
        for rule in rules:
            arms[rule] = {"honest": [], "feasible": [], "front_honest": [], "n_front": [], "hv": [],
                          "vouched_hv": [], "optimism": [], "thickness_min": [], "population": [], "trust": []}
        for seed in seeds:
            X0 = _campaign(regime, seed)
            Y0 = airfoil_front(X0)
            if surrogate == "gp":
                model = GaussianProcessSurrogate(3, Y0.shape[1], seed=seed)
            elif surrogate == "ensemble":
                model = EnsembleSurrogate(3, Y0.shape[1], n_members=int(n_members), hidden=24, epochs=int(epochs),
                                          seed=seed)
            else:
                raise ValueError(f"surrogate must be 'gp' or 'ensemble'; got {surrogate!r}")
            model.fit(X0, Y0)
            domain = ApplicabilityDomain.fit(X0, names=["camber", "camber_position", "thickness"], k=3, level=0.9)
            for rule in rules:
                res = search(bounds, model, domain, constraints=cons, tolerance=tolerance, mu=mu,
                             generations=generations, seed=seed, rule=rule)
                got = _verified(res.population, res.predicted, res.verdicts, tolerance, tolerance_cl, cl_target)
                for key, value in got.items():
                    if key == "trust":
                        arms[rule]["trust"].append(value)
                    else:
                        arms[rule][key].append(value)
                arms[rule]["population"].append(res.population_size)
        summary = {}
        for rule, arm in arms.items():
            summary[rule] = {k: (float(np.mean(v)) if k != "trust" else
                                 {level: float(np.mean([t[level] for t in v])) for level in TRUST_LEVELS})
                             for k, v in arm.items()}
        out["regimes"][regime] = summary
    return out

