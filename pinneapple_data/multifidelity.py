"""Multi-fidelity regression over a table: what the cheap model buys, and how much of the expensive one is enough.

A multi-fidelity study has two fidelities of the same designs -- a cheap model that runs everywhere and an expensive
one that ran on a few designs -- and one question: *how much does the cheap one buy?* This module answers it as a
learning curve: split the expensive designs into a training part of size N and the rest, fit the target four ways,
and score each way on the designs it did not see.

===================  ==============================================================================
``low_fidelity``     no fit at all: the cheap model's own value at that design. The baseline a learned
                     correction has to beat, and the reason a study like this is worth running.
``geometry``         a surrogate of the target from the design alone -- the answer to "what if there is no
                     cheap model".
``difference``       a surrogate of ``y_high - y_low`` from the design: the cheap model everywhere plus a
                     learned correction. This is the multi-fidelity model, and it is what a co-kriging
                     study computes in the two-fidelity case.
``residual_feature`` a surrogate of the same difference, with the cheap model's value at the design among
                     its inputs. When this beats ``difference`` the correction depends on *where* the cheap
                     model is, not only on *where* the design is.

What this module deliberately does not do: **pretrain a surrogate on the cheap fidelity**. The cheap model is a
function that is already available at every design, so a surrogate of it adds nothing but its own error -- what the
1024 cheap runs buy is the shape of the correction, and the ``difference`` row measures exactly that. (With a cheap
fidelity that is itself expensive to evaluate, pretraining would be the thing to do; this is not that case.)

    >>> from pinneapple_data.multifidelity import FidelityPair, learning_curve
    >>> pair = FidelityPair.pair(X_low, y_low, X_high, y_high)
    >>> curve = learning_curve(pair, budgets=(8, 16, 32))
    >>> print(curve.table())

The regressor is kernel ridge with an RBF kernel, in closed form: no epochs, no seed noise beyond the split, and it
behaves at N = 8, where a neural network has nothing to learn from. A row of the curve is a median over repeated
random splits, with the interval the medians of the repeats span.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FidelityPair", "FieldPair", "KernelRidge", "LearningCurve", "learning_curve",
           "field_learning_curve", "residual_rank", "relative_l2", "r2", "STRATEGIES", "FIELD_STRATEGIES"]

STRATEGIES = ("low_fidelity", "geometry", "difference", "residual_feature")


# ---------------------------------------------------------------------------
# the two fidelities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FidelityPair:
    """Two fidelities of the same design space.

    ``low_at_high`` is the cheap model evaluated at the expensive designs. It is not optional and it is not
    interpolated here: a study whose cheap model cannot be evaluated where the expensive one ran is not a
    multi-fidelity study, it is two unrelated datasets, and guessing the value would make every number below a guess
    too."""

    design_low: np.ndarray          # (n_low, d)
    values_low: np.ndarray          # (n_low,)
    design_high: np.ndarray         # (n_high, d)
    values_high: np.ndarray         # (n_high,)
    low_at_high: np.ndarray         # (n_high,) the cheap model at the expensive designs
    names: tuple[str, ...] = ()

    def __post_init__(self):
        low = np.asarray(self.design_low, dtype=np.float64)
        high = np.asarray(self.design_high, dtype=np.float64)
        for name, array, rows in (("design_low", low, low.shape[0]), ("design_high", high, high.shape[0]),
                                  ("values_low", np.asarray(self.values_low), low.shape[0]),
                                  ("values_high", np.asarray(self.values_high), high.shape[0]),
                                  ("low_at_high", np.asarray(self.low_at_high), high.shape[0])):
            if np.asarray(array).shape[0] != rows:
                raise ValueError(f"{name} has {np.asarray(array).shape[0]} rows; expected {rows}")
        if low.ndim != 2 or high.ndim != 2 or low.shape[1] != high.shape[1]:
            raise ValueError(f"the two fidelities must be (n, d) with the same d: {low.shape} and {high.shape}")
        object.__setattr__(self, "design_low", low)
        object.__setattr__(self, "design_high", high)
        object.__setattr__(self, "values_low", np.asarray(self.values_low, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "values_high", np.asarray(self.values_high, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "low_at_high", np.asarray(self.low_at_high, dtype=np.float64).reshape(-1))

    @property
    def n_low(self) -> int:
        return int(self.design_low.shape[0])

    @property
    def n_high(self) -> int:
        return int(self.design_high.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.design_low.shape[1])

    def summary(self) -> str:
        return (f"{self.n_low} cheap design(s) and {self.n_high} expensive one(s) over {self.dimension} input(s); "
                f"the cheap model is {relative_l2(self.low_at_high, self.values_high):.1%} off the expensive one in "
                "relative L2")

    @classmethod
    def pair(cls, design_low, values_low, design_high, values_high, *, atol: float = 1e-9) -> FidelityPair:
        """The pair, with the cheap model evaluated at each expensive design by finding that design among the cheap
        ones.

        The match is exact to ``atol`` on the scaled inputs: a design that is not in the cheap table is an error,
        because the alternative -- the nearest cheap run -- silently answers a different question. (When the cheap
        model is a *function* rather than a table, call it at the expensive designs and build the pair directly.)"""
        low = np.asarray(design_low, dtype=np.float64)
        high = np.asarray(design_high, dtype=np.float64)
        if low.ndim != 2 or high.ndim != 2 or low.shape[1] != high.shape[1]:
            raise ValueError(f"the two fidelities must be (n, d) with the same d: {low.shape} and {high.shape}")
        scale = np.maximum(np.abs(low).max(axis=0), np.abs(high).max(axis=0))
        scale[scale == 0.0] = 1.0
        index: dict[tuple, int] = {}
        for row, design in enumerate(low):
            index.setdefault(tuple(np.round(design / scale, 12)), row)
        found, missing = [], []
        for row, design in enumerate(high):
            hit = index.get(tuple(np.round(design / scale, 12)))
            (missing if hit is None else found).append(row if hit is None else hit)
        if missing:
            raise ValueError(f"{len(missing)} expensive design(s) are not among the cheap ones, e.g. row {missing[0]}"
                             ": the cheap model has to be evaluated there to be compared")
        cheap = np.asarray(values_low, dtype=np.float64).reshape(-1)[np.asarray(found, dtype=np.int64)]
        return cls(design_low=low, values_low=values_low, design_high=high, values_high=values_high,
                   low_at_high=cheap)


# ---------------------------------------------------------------------------
# the regressor
# ---------------------------------------------------------------------------

@dataclass
class KernelRidge:
    """Kernel ridge regression with an RBF kernel, in closed form.

    The length scale is the median pairwise distance of the training designs -- a standard heuristic that needs no
    tuning, which matters at eight training points. ``scale`` divides the inputs first, so a design in metres and a
    Mach number are compared on the same footing: the caller passes the span of each input, or leaves it as the
    training range, which is the same thing when the training points cover the space."""

    ridge: float = 1e-8
    length_scale: float | None = None
    scale: np.ndarray | None = None
    _x: np.ndarray | None = field(default=None, repr=False)
    _alpha: np.ndarray | None = field(default=None, repr=False)
    _y_mean: float = 0.0

    def _scaled(self, x: np.ndarray) -> np.ndarray:
        scale = self.scale if self.scale is not None else np.ones(x.shape[1])
        return np.asarray(x, dtype=np.float64) / np.where(scale == 0.0, 1.0, scale)

    def fit(self, x, y) -> KernelRidge:
        x = self._scaled(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.size:
            raise ValueError(f"{x.shape[0]} designs against {y.size} values")
        if x.shape[0] == 1:
            raise ValueError("one design is not enough to fit: the kernel has no distance to measure")
        self._x = x
        self._y_mean = float(y.mean())
        if self.length_scale is None:
            self.length_scale = _median_distance(x)
        kernel = _rbf(x, x, float(self.length_scale))
        self._alpha = np.linalg.solve(kernel + self.ridge * np.eye(x.shape[0]), y - self._y_mean)
        return self

    def predict(self, x) -> np.ndarray:
        if self._x is None or self._alpha is None:
            raise RuntimeError("predict before fit")
        x = self._scaled(np.atleast_2d(np.asarray(x, dtype=np.float64)))
        return _rbf(x, self._x, float(self.length_scale)) @ self._alpha + self._y_mean


def _median_distance(x: np.ndarray) -> float:
    """The median pairwise distance, and never zero: a duplicated design would make the kernel a delta."""
    differences = x[:, None, :] - x[None, :, :]
    distances = np.sqrt((differences ** 2).sum(axis=-1))
    off_diagonal = distances[~np.eye(distances.shape[0], dtype=bool)]
    scale = float(np.median(off_diagonal)) if off_diagonal.size else 1.0
    return scale if scale > 0.0 else 1.0


def _rbf(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    distances = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1), 0.0))
    return np.exp(-0.5 * (distances / length_scale) ** 2)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def relative_l2(prediction, measured) -> float:
    """``||prediction - measured|| / ||measured||`` over the rows -- the form a physical target is scored in."""
    measured = np.asarray(measured, dtype=np.float64).reshape(-1)
    difference = np.asarray(prediction, dtype=np.float64).reshape(-1) - measured
    denominator = np.linalg.norm(measured)
    return float(np.linalg.norm(difference) / denominator) if denominator else float("nan")


def r2(prediction, measured) -> float:
    """The coefficient of determination; zero for the mean, negative for worse than it."""
    measured = np.asarray(measured, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    total = float(((measured - measured.mean()) ** 2).sum())
    return float(1.0 - ((measured - prediction) ** 2).sum() / total) if total else float("nan")


# ---------------------------------------------------------------------------
# the learning curve
# ---------------------------------------------------------------------------

@dataclass
class LearningCurve:
    """Every row of the curve, and the splits behind them."""

    budgets: tuple[int, ...]
    repeats: int
    scores: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # strategy -> metric -> (budget, repeat)

    def median(self, strategy: str, metric: str = "relative_l2") -> np.ndarray:
        return np.median(self.scores[strategy][metric], axis=1)

    def spread(self, strategy: str, metric: str = "relative_l2") -> np.ndarray:
        """The worst and the best of the repeats -- the spread of the seeds, not a confidence interval of the mean."""
        return np.ptp(self.scores[strategy][metric], axis=1)

    def table(self) -> str:
        strategies = list(self.scores)                  # in the order they were fitted
        header = f"{'N':>4}  " + "  ".join(f"{s:>18}" for s in strategies)
        lines = [header, "-" * len(header)]
        for i, budget in enumerate(self.budgets):
            cells = []
            for strategy in strategies:
                value = float(np.median(self.scores[strategy]["relative_l2"][i]))
                spread = float(np.ptp(self.scores[strategy]["relative_l2"][i]))
                cells.append(f"{value:10.3%} ±{spread:6.3%}")
            lines.append(f"{budget:>4}  " + "  ".join(f"{c:>18}" for c in cells))
        return "\n".join(lines)


def learning_curve(pair: FidelityPair, *, budgets=(4, 8, 16, 32), repeats: int = 10, seed: int = 0,
                   ridge: float = 1e-8, strategies=STRATEGIES) -> LearningCurve:
    """Fit each strategy on N expensive designs and score it on the ones it did not see.

    For ``N < n_high`` the split is a random ``N`` of the expensive designs (``repeats`` times, one seed), and the
    score is on the remaining ones. For ``N == n_high`` there is nothing left to hold out, so each design is predicted
    from the other ``N - 1`` (leave-one-out) -- a different question, said so in the row, and the only honest way to
    put the full budget on the same axis.

    Every strategy is scored on the *same* held-out designs of a repeat, so the comparison between them is paired.
    The scales of the inputs and of the cheap model are the pair's own, so a design in metres and one in Mach number
    are on the same footing."""
    if repeats < 1:
        raise ValueError(f"{repeats} repeats is not a curve")
    pair = pair if isinstance(pair, FidelityPair) else FidelityPair(**pair)
    n = pair.n_high
    if n < 2:
        raise ValueError(f"{n} expensive design(s): a curve needs at least two to hold one out")
    budgets = tuple(int(b) for b in budgets)
    for budget in budgets:
        if not 1 < budget <= n:
            raise ValueError(f"budget {budget} is outside (1, {n}]: it cannot be trained on and scored on the same "
                             "set")
    scale = np.maximum(np.abs(pair.design_high).max(axis=0), 1e-30)
    design = pair.design_high / scale
    cheap = pair.low_at_high
    target = pair.values_high
    cheap_scale = float(np.abs(cheap).max()) or 1.0
    scores = {s: {"relative_l2": np.full((len(budgets), repeats), np.nan),
                  "r2": np.full((len(budgets), repeats), np.nan)} for s in strategies}
    rng = np.random.default_rng(seed)
    for i, budget in enumerate(budgets):
        for repeat in range(repeats):
            if budget < n:
                order = rng.permutation(n)
                train, test = order[:budget], order[budget:]
                prediction = {s: _predict(s, design[train], cheap[train], target[train],
                                          design[test], cheap[test], ridge, scale, cheap_scale) for s in strategies}
                measured = target[test]
            else:                                   # nothing left to hold out: leave one out at a time
                prediction = {s: np.array([_predict(s, np.delete(design, j, axis=0), np.delete(cheap, j),
                                                   np.delete(target, j), design[j:j + 1], cheap[j:j + 1],
                                                   ridge, scale, cheap_scale)[0] for j in range(n)])
                              for s in strategies}
                measured = target
            for strategy in strategies:
                scores[strategy]["relative_l2"][i, repeat] = relative_l2(prediction[strategy], measured)
                scores[strategy]["r2"][i, repeat] = r2(prediction[strategy], measured)
    return LearningCurve(budgets=budgets, repeats=repeats, scores=scores)


def _predict(strategy: str, design_train: np.ndarray, cheap_train: np.ndarray, target_train: np.ndarray,
             design_eval: np.ndarray, cheap_eval: np.ndarray, ridge: float, scale: np.ndarray,
             cheap_scale: float) -> np.ndarray:
    """One strategy, fitted on the training designs and evaluated where the cheap model's value is known.

    Every strategy takes the cheap model's value at the evaluation designs, because the cheap model runs everywhere --
    that is what makes it cheap. ``geometry`` and ``difference`` do not read it; ``low_fidelity`` is nothing else;
    ``residual_feature`` reads it as an input as well as adding it back."""
    if strategy == "low_fidelity":
        return cheap_eval
    if strategy == "geometry":
        return KernelRidge(ridge=ridge, scale=scale).fit(design_train, target_train).predict(design_eval)
    if strategy == "difference":
        correction = KernelRidge(ridge=ridge, scale=scale).fit(design_train, target_train - cheap_train)
        return correction.predict(design_eval) + cheap_eval
    if strategy == "residual_feature":
        features = np.column_stack([design_train, cheap_train / cheap_scale])
        correction = KernelRidge(ridge=ridge, scale=np.append(scale, 1.0)).fit(features, target_train - cheap_train)
        return correction.predict(np.column_stack([design_eval, cheap_eval / cheap_scale])) + cheap_eval
    raise KeyError(f"'{strategy}' is not one of {STRATEGIES}")


# ---------------------------------------------------------------------------
# the same question asked of a field
# ---------------------------------------------------------------------------

FIELD_STRATEGIES = ("low_fidelity", "mean_correction", "pod_correction")


@dataclass(frozen=True)
class FieldPair:
    """Two fidelities of a *field* over one shared discretisation.

    Every expensive run's field lives on the same nodes as its cheap run's -- that is what makes the difference a
    subtraction and not an interpolation, and it is stated here as a requirement rather than hoped for. ``position``
    is whatever orders the nodes along the surface (an arc length, a coordinate); it is not used by the curve, only
    reported with it, because a field model that reads the position is a different model."""

    design_high: np.ndarray          # (n_high, d)
    values_high: np.ndarray          # (n_high, n_nodes)
    values_low: np.ndarray           # (n_high, n_nodes) the cheap model on the same nodes
    position: np.ndarray | None = None

    def __post_init__(self):
        high = np.asarray(self.values_high, dtype=np.float64)
        low = np.asarray(self.values_low, dtype=np.float64)
        design = np.asarray(self.design_high, dtype=np.float64)
        if high.ndim != 2 or low.shape != high.shape:
            raise ValueError(f"the two fidelities must be (n_runs, n_nodes) of the same shape: {high.shape} and "
                             f"{low.shape}")
        if design.shape[0] != high.shape[0]:
            raise ValueError(f"{design.shape[0]} designs against {high.shape[0]} runs")
        object.__setattr__(self, "values_high", high)
        object.__setattr__(self, "values_low", low)
        object.__setattr__(self, "design_high", design)

    @property
    def n_high(self) -> int:
        return int(self.values_high.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.values_high.shape[1])

    def residual(self) -> np.ndarray:
        return self.values_high - self.values_low

    def run_scale(self) -> np.ndarray:
        """The magnitude of each run's cheap field, ``(n_high,)``.

        The divisor that makes a correction comparable across designs: heating spans orders of magnitude, so a residual
        field measured in W m-2 says which run is hottest, not how wrong the cheap model is there. Dividing by this
        and multiplying back by the *cheap* run's magnitude is legitimate at any design, because the cheap model runs
        everywhere."""
        return np.linalg.norm(self.values_low, axis=1)

    def summary(self) -> str:
        per_run = np.linalg.norm(self.residual(), axis=1) / np.linalg.norm(self.values_high, axis=1)
        return (f"{self.n_high} expensive run(s) over {self.n_nodes} nodes; the cheap model is "
                f"{np.median(per_run):.1%} off in relative L2 per run (median), {per_run.min():.1%} to "
                f"{per_run.max():.1%}")


def residual_rank(residual: np.ndarray, *, components: int = 1, scale: np.ndarray | None = None) -> dict:
    """How much of the residual field a rank-``k`` model of the runs captures.

    The residual of a run is one vector over the nodes. Stacked over runs, its singular value decomposition says how
    many independent *shapes* of correction there are: if the first mode carries most of the energy, the correction is
    one profile with an amplitude per design, and a study of it is a study of that amplitude.

    ``scale`` divides each run's residual by a per-run magnitude first. On the capsule corpus the energy share comes
    out about the same either way (80 % raw against 81 % normalised), but the *coefficients* do not: raw, they span
    the 175x the heating spans, and no regression on six inputs learns that. Pass the norm of each run's own field
    (what :meth:`FieldPair.run_scale` returns) to measure the structure of the relative correction."""
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2:
        raise ValueError(f"the residual must be (n_runs, n_nodes): {residual.shape}")
    if scale is not None:
        divisor = np.asarray(scale, dtype=np.float64).reshape(-1, 1)
        residual = residual / np.where(divisor == 0.0, 1.0, divisor)
    left, singular, right = np.linalg.svd(residual, full_matrices=False)
    energy = singular ** 2
    share = float(energy[:components].sum() / energy.sum()) if energy.sum() > 0 else float("nan")
    return {"share": share, "modes": right[:components], "singular": singular,
            "amplitude": left[:, :components] * singular[:components]}


def field_learning_curve(pair: FieldPair, *, budgets=(8, 16, 32), repeats: int = 10, seed: int = 0,
                         components: int = 2, ridge: float = 1e-8,
                         strategies=FIELD_STRATEGIES) -> LearningCurve:
    """The learning curve of a field, with a low-rank correction instead of a network per node.

    ==================  ==============================================================================
    ``low_fidelity``    the cheap field itself, on the same nodes.
    ``mean_correction`` the cheap field plus the average *relative* residual of the training runs: one shape,
                        scaled by the run's own magnitude, and no dependence on the design. What a correction
                        buys when nothing is learned from the inputs.
    ``pod_correction``  the cheap field plus ``components`` relative residual modes, each amplitude learned from
                        the design by kernel ridge. The proper orthogonal decomposition of the residual with a
                        regression on its coefficients, which is the field analogue of ``difference``.

    The correction is modelled **relative to each run's own magnitude** (:meth:`FieldPair.run_scale`) and then scaled
    back by the cheap run's magnitude, which is known at every design because the cheap model runs everywhere. The
    normalisation is not cosmetic: the heat flux spans 175x across designs in the capsule corpus, so the coefficients
    of the raw decomposition span 175x too, and a regression on six inputs cannot learn a quantity whose scale is set
    by the design. Normalised, the first mode carries 81 % of the residual energy and the third 97 %, and the
    coefficients are O(0.1) for every run -- which is what makes them predictable at all.

    Why low rank rather than a pointwise network: with a handful of expensive runs there are thousands of nodes and
    only N fields, so a model free to choose every node's correction is fitting noise -- and the residual of a family
    of shapes turns out to be a couple of modes deep, which :func:`residual_rank` measures instead of assuming. The
    score is the relative L2 over each held-out run's whole field, then the median over runs and splits."""
    if repeats < 1:
        raise ValueError(f"{repeats} repeats is not a curve")
    n = pair.n_high
    if n < 3:
        raise ValueError(f"{n} expensive run(s): a field curve needs at least three to hold one out and keep a mean")
    budgets = tuple(int(b) for b in budgets)
    for budget in budgets:
        if not 2 < budget <= n:
            raise ValueError(f"budget {budget} is outside (2, {n}]")
    scale = np.maximum(np.abs(pair.design_high).max(axis=0), 1e-30)
    design = pair.design_high / scale
    run_scale = pair.run_scale()
    residual = pair.residual() / run_scale.reshape(-1, 1)
    scores = {s: {"relative_l2": np.full((len(budgets), repeats), np.nan),
                  "r2": np.full((len(budgets), repeats), np.nan)} for s in strategies}
    rng = np.random.default_rng(seed)
    for i, budget in enumerate(budgets):
        for repeat in range(repeats):
            if budget < n:
                order = rng.permutation(n)
                train, test = order[:budget], order[budget:]
            else:                                    # nothing to hold out: leave one run out at a time
                train, test = np.arange(n), np.arange(n)
            prediction = {s: _predict_field(s, design, residual, pair.values_low, run_scale, train, test,
                                            components, ridge, scale, leave_one_out=budget == n)
                          for s in strategies}
            measured = pair.values_high[test]
            for strategy in strategies:
                per_run = [relative_l2(prediction[strategy][j], measured[j]) for j in range(len(test))]
                scores[strategy]["relative_l2"][i, repeat] = float(np.median(per_run))
                scores[strategy]["r2"][i, repeat] = float(np.median(
                    [r2(prediction[strategy][j], measured[j]) for j in range(len(test))]))
    return LearningCurve(budgets=budgets, repeats=repeats, scores=scores)


def _predict_field(strategy: str, design: np.ndarray, relative_residual: np.ndarray, cheap: np.ndarray,
                   run_scale: np.ndarray, train: np.ndarray, test: np.ndarray, components: int, ridge: float,
                   scale: np.ndarray, *, leave_one_out: bool) -> np.ndarray:
    """The predicted fields at the ``test`` runs, one row per run."""
    if strategy == "low_fidelity":
        return cheap[test]
    if leave_one_out:
        predicted = np.empty((len(test), cheap.shape[1]), dtype=np.float64)
        for slot, run in enumerate(test):
            others = np.setdiff1d(train, [run])
            predicted[slot] = _predict_field(strategy, design, relative_residual, cheap, run_scale, others,
                                             np.array([run]), components, ridge, scale, leave_one_out=False)[0]
        return predicted
    if strategy == "mean_correction":
        return cheap[test] + relative_residual[train].mean(axis=0) * run_scale[test].reshape(-1, 1)
    if strategy == "pod_correction":
        _, _, right = np.linalg.svd(relative_residual[train], full_matrices=False)
        modes = right[:components]                                   # (k, n_nodes)
        coefficients = relative_residual[train] @ modes.T            # (n_train, k)
        correction = np.zeros((len(test), cheap.shape[1]), dtype=np.float64)
        usable = min(components, modes.shape[0])
        for mode in range(usable):
            model = KernelRidge(ridge=ridge, scale=scale).fit(design[train], coefficients[:, mode])
            correction += np.outer(model.predict(design[test]), modes[mode])
        return cheap[test] + correction * run_scale[test].reshape(-1, 1)
    raise KeyError(f"'{strategy}' is not one of {FIELD_STRATEGIES}")
