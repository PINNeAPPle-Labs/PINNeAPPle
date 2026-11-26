"""The battle scoreboard: two models, one stream of conditions, and the step at which each one leaves the domain it
was validated on (#373).

A leaderboard answers "which model is more accurate", and answers it with a mean over everything, which is the one
number that hides the failure that matters: *where* a surrogate stops being usable. This module keeps the whole
curve, step by step, and puts four things on one board:

* **what each model costs** -- the optimiser time and the prediction time of one trained model, measured per seed;
* **what each model gets wrong** -- the relative L2 against the reference at every step of the stream, not its mean;
* **who leaves its domain first**, with the criterion written down twice, because the two questions are different:

  - ``residual_alarm`` -- what the model can say about itself with **no reference at all**: the PDE residual of its
    own field, computed with the same finite-difference operator for every model, crossing ``residual_factor``
    times the value it had at the last step of its window. This is the signal a deployment can actually monitor;
  - ``error_exit`` -- the first step where the error against the reference crossed the published tolerance. This is
    the truth, available only after the fact -- and the column the scoreboard ranks by, because it is absolute:
    ranking by the self-calibrated column would put the model that was *better* in its window first, for the only
    reason that it had less room to be four times worse.

  Reporting only the second hides that the monitor does not work; reporting only the first hides that a model can
  be confidently wrong. The scoreboard ranks by ``error_exit`` and prints the alarm next to it, so a model whose
  alarm fires long after its error did is visible as such.
* **where the models disagree and the reference decides** -- the step of largest spread between the two fields, and
  which one the reference favours there. That is the moment worth watching: two surrogates, one answer each, and a
  third opinion that settles it.

The residual is deliberately *not* the residual the model was trained on. A PINN's autograd residual measures the
loss it minimised on its own collocation points; asking it again on the same points would grade the model with its
own homework. Both models are scored here with second-order central differences on the field they return, the same
operator for both, so the number means the same thing whichever family produced the field.

The exit is measured per seed, and the row carries the **first** seed that left, not the mean of the seeds: a seed
that survived has no exit, and substituting the end of the stream for it and averaging is the same mean that hides
the failure this module exists to show. A model that left in one seed of two has left, and it says so -- with the
count -- where the mean would have printed "never". A model that never left any seed is placed at the end of the
stream: it survived the whole battle, which is a result, not a missing value.

This is the measurement, not the animation: the page and the JSON are generated from one run so the published
numbers cannot drift from it. The migration to ``pp.trust`` (issue #345) is the next step for the threshold
semantics -- this module says *when* a model left; it does not yet say whether the remainder is trustworthy, which is
that layer's job. Until it lands, the record below is the honest thing this library can say about itself.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bundle_bench import BenchConfig, run_and_predict

__all__ = ["SCHEMA", "RESULTS", "PAGE", "PROTOCOL", "CASES", "BattleCase", "Field", "domain_exit",
           "relative_l2_curve", "pde_residual_curve", "scoreboard", "disagreement", "run_case", "run_all", "table",
           "page", "write", "check", "get_case"]

SCHEMA = "pinneapple.battle/1"
RESULTS = Path("benchmarks") / "battle" / "results.json"
PAGE = Path("docs") / "battle.md"

# Everything a reproduction has to keep constant, as data rather than as literals inside the loop. The two models
# share the split, the optimiser, the learning rate, the batch size and the seed; each keeps its own size knobs,
# because a Fourier operator of width 8 is not the same model as an MLP of width 64 and pretending otherwise would
# make the cost column meaningless.
PROTOCOL: dict[str, Any] = {
    "seeds": (0, 1),
    "budget": {
        "epochs": 300,
        "lr": 1e-3,
        "batch_size": 8,
        "fno": {"width": 32, "modes": 8, "layers": 4},
        "pinn": {"hidden": (64, 64, 64), "pde_weight": 1e-3},
    },
    "window": {"split_at": 0.5, "train_fraction": 0.5},
    "grid": {"nx": 129, "nt": 41},
    "metrics": ("relative_l2", "pde_residual"),
    # the exit criterion, both halves of it, as declared numbers. The error against the reference has a scale, so
    # it is compared with the tolerance this library already uses to call a field reproduced; the residual has
    # none, so it is compared with what the model itself showed at the boundary of its window, times a factor
    "exit": {"residual_factor": 4.0, "tolerance": 5e-2, "residual_calibrated_on": "last step inside the window"},
    # the rule the board is sorted by, declared as data because `check` fails a table published out of this order
    "ranking": {
        "key": "the first step past the window where the error crossed the tolerance, over the seeds (the minimum, "
               "not the mean: a seed that survived the stream has no exit, and averaging the end of the stream in as "
               "a stand-in would let a model that left in one seed outrank one that never left)",
        "tie_break": ("the model that left in more seeds", "the smaller final relative L2", "the cheaper training",
                      "the model name"),
    },
    "residual_operator": ("second-order central differences in x and t on the field the model returns; the same "
                          "operator for every model, never the training loss"),
    "cost": "wall time with torch.set_num_threads(1), after a warm-up pass, per seed; information, not a ranking key",
}
REQUIRED_PROTOCOL_FIELDS = ("seeds", "budget", "window", "grid", "metrics", "exit", "ranking")


# ------------------------------------------------------------------ the stream of cases
@dataclass(frozen=True)
class Field:
    """One battle field: the bundle both models train on, the physical axes, the reference and the residual.

    ``residual(u)`` takes an ``(nt, nx)`` field in physical units and returns the same shape with the governing
    equation's residual; it is the operator both models are scored with, and it is given here rather than derived,
    so a case cannot quietly score one family with a friendlier operator than the other.
    """
    bundle: Any
    t: np.ndarray
    x: np.ndarray
    exact: np.ndarray
    residual: Callable[[np.ndarray], np.ndarray]
    field: str = "u"
    residual_training: Callable[..., Any] | None = None


@dataclass(frozen=True)
class BattleCase:
    """A case is the stream, the reference and what the two models are allowed to see of it."""
    id: str
    name: str
    equation: str
    reference: str
    split_at: float                     # the model's window: t <= split_at, and the rest is the battle
    build: Callable[[], Field]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "equation": self.equation, "reference": self.reference,
                "split_at": float(self.split_at)}


def _central_difference(field: np.ndarray, axis: int, spacing: np.ndarray, order: int = 1) -> np.ndarray:
    """Second-order finite difference along one axis, on the interior, NaN on the two edges it cannot reach."""
    n = field.shape[axis]
    out = np.full(field.shape, np.nan, dtype=np.float64)
    if n < 2 * order + 1:
        return out
    cell = [slice(None)] * field.ndim
    ahead, behind = list(cell), list(cell)
    if order == 1:
        ahead[axis] = slice(2, None)
        behind[axis] = slice(None, -2)
        cell[axis] = slice(1, -1)
        d = spacing[2:] - spacing[:-2]
        shape = [1] * field.ndim
        shape[axis] = d.size
        out[tuple(cell)] = (field[tuple(ahead)] - field[tuple(behind)]) / d.reshape(shape)
    elif order == 2:
        centre, ahead, behind = list(cell), list(cell), list(cell)
        centre[axis] = slice(1, -1)
        ahead[axis] = slice(2, None)
        behind[axis] = slice(None, -2)
        h = float(np.mean(np.abs(np.diff(spacing))))
        out[tuple(centre)] = (field[tuple(ahead)] - 2.0 * field[tuple(centre)] + field[tuple(behind)]) / h ** 2
    else:                                         # pragma: no cover - a caller asking for another order
        raise ValueError(f"order must be 1 or 2, got {order}")
    return out


def burgers_residual(nu: float) -> Callable[[np.ndarray], np.ndarray]:
    """``u_t + u u_x - nu u_xx`` on a ``(nt, nx)`` field, for the axes declared by the case."""

    def residual(u: np.ndarray) -> np.ndarray:
        nt, nx = u.shape
        t = np.linspace(0.0, 1.0, nt)
        x = np.linspace(-1.0, 1.0, nx)
        u_t = _central_difference(u, 0, t, 1)
        u_x = _central_difference(u, 1, x, 1)
        u_xx = _central_difference(u, 1, x, 2)
        return u_t + u * u_x - nu * u_xx

    return residual


def _burgers_field(nx: int, nt: int, nu: float, split_at: float) -> Field:
    """The Burgers sine field on a grid, its closed form as the reference, and the two residual operators.

    The training residual is the same equation written in the normalised coordinates the network sees (``x`` in
    [0, 1], ``u`` standardised on the window): ``x = 2 x~ - 1`` and ``u = mean + scale u~`` give

        R = scale (u~_t + (mean + scale u~) u~_x / 2) - nu scale u~_xx / 4

    and leaving that chain rule out is the classic way to train a PINN on the wrong equation while the loss goes
    down.
    """
    from pinneapple_data.schema import AxisSpec, Physics, PhysicsDataBundle
    from pinneapple_data.schema import Field as BundleField
    from pinneapple_physics.closed_form.burgers import burgers_sine_exact

    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x, t)                      # (nt, nx)
    exact = np.asarray(burgers_sine_exact(X, T, nu=nu), dtype=np.float64)
    bundle = PhysicsDataBundle.from_arrays(
        {"field:u": exact.astype(np.float32), "axis:t": t, "axis:x": x},
        fields=[BundleField.from_array("u", exact.astype(np.float32), unit="1", axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        physics=Physics(governing="u_t + u u_x = nu u_xx, u(x, 0) = -sin(pi x), u = 0 at x = -1 and x = 1"),
        source="closed form: Cole-Hopf by Gauss-Hermite quadrature (pinneapple_physics.closed_form.burgers)",
        license="CC0-1.0", citation="Raissi, Perdikaris & Karniadakis 2019, J. Comput. Phys. 378", kind="synthetic")

    train = np.flatnonzero(t <= split_at)
    mean, scale = float(exact[train].mean()), float(exact[train].std()) or 1.0

    def residual_training(_net, _t, _x, u, du_dt, du_dx, du_dxx):
        return scale * (du_dt + (mean + scale * u) * du_dx / 2.0) - nu * scale * du_dxx / 4.0

    return Field(bundle=bundle, t=t, x=x, exact=exact, residual=burgers_residual(nu),
                 residual_training=residual_training)


def _heat_field(nx: int, nt: int, alpha: float, split_at: float) -> Field:
    """The 1-D heat field, its closed form as the reference, and the two residual operators.

    The clean case for the residual monitor: ``T = exp(-alpha pi^2 t) sin(pi x)`` satisfies the equation exactly, so
    the operator's own floor on this grid is a rounding error (measured: 2.6e-5 on 129 x 41) and the alarm column
    means what it says. The Burgers case is the one where it does not, and the record shows the floor there so the
    difference is visible instead of explained away.
    """
    from pinneapple_data.schema import AxisSpec, Physics, PhysicsDataBundle
    from pinneapple_data.schema import Field as BundleField

    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    X, T = np.meshgrid(x, t)
    exact = np.exp(-alpha * np.pi ** 2 * T) * np.sin(np.pi * X)
    bundle = PhysicsDataBundle.from_arrays(
        {"field:T": exact.astype(np.float32), "axis:t": t, "axis:x": x},
        fields=[BundleField.from_array("T", exact.astype(np.float32), unit="1", axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        physics=Physics(governing="alpha T_xx = T_t, T(x, 0) = sin(pi x), T = 0 at x = 0 and x = 1"),
        source="closed form: exp(-alpha pi^2 t) sin(pi x)", license="CC0-1.0", citation="analytic", kind="synthetic")

    train = np.flatnonzero(t <= split_at)
    scale = float(exact[train].std()) or 1.0

    def residual_training(_net, _t, _x, u, du_dt, du_dx, du_dxx):
        # x and t are already normalised to [0, 1] on this case, and T does not appear in the equation, so only the
        # field scale survives the chain rule -- the centre of the window would not
        return scale * (du_dt - alpha * du_dxx)

    def residual(u: np.ndarray) -> np.ndarray:
        u_t = _central_difference(u, 0, t, 1)
        u_xx = _central_difference(u, 1, x, 2)
        return u_t - alpha * u_xx

    return Field(bundle=bundle, t=t, x=x, exact=exact, residual=residual, field="T",
                 residual_training=residual_training)


CASES: tuple[BattleCase, ...] = (
    BattleCase(
        id="heat",
        name="heat_1d",
        equation="alpha T_xx = T_t, alpha = 0.05, T(x, 0) = sin(pi x), T = 0 at x = 0 and x = 1",
        reference="closed form: exp(-alpha pi^2 t) sin(pi x)",
        split_at=0.5,
        build=lambda: _heat_field(PROTOCOL["grid"]["nx"], PROTOCOL["grid"]["nt"], alpha=0.05, split_at=0.5)),
    BattleCase(
        id="burgers",
        name="burgers_1d",
        equation="u_t + u u_x = nu u_xx, nu = 0.01, u(x, 0) = -sin(pi x), u = 0 at x = -1 and x = 1",
        reference="closed form (Cole-Hopf by Gauss-Hermite quadrature), scored on the whole stream, the window "
                  "included",
        split_at=0.5,
        build=lambda: _burgers_field(PROTOCOL["grid"]["nx"], PROTOCOL["grid"]["nt"], nu=0.01, split_at=0.5)),
)
BY_ID = {c.id: c for c in CASES}


def get_case(case_id: str) -> BattleCase:
    if case_id not in BY_ID:
        raise KeyError(f"unknown battle case {case_id!r}; declared: {sorted(BY_ID)}")
    return BY_ID[case_id]


# ------------------------------------------------------------------ the measurements
def relative_l2_curve(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Relative L2 per step of the stream: the error curve, in the same metric the benchmark protocol uses."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    num = np.linalg.norm(pred - ref, axis=1)
    den = np.maximum(np.linalg.norm(ref, axis=1), 1e-300)
    return num / den


def pde_residual_curve(pred: np.ndarray, residual: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """RMS of the residual per step, over the points where the operator exists (the x edges are NaN)."""
    r = np.asarray(residual(np.asarray(pred, dtype=np.float64)), dtype=np.float64)
    finite = np.isfinite(r)
    counts = finite.sum(axis=1)
    sums = np.where(finite, r ** 2, 0.0).sum(axis=1)
    out = np.full(r.shape[0], np.nan)
    ok = counts > 0
    out[ok] = np.sqrt(sums[ok] / counts[ok])
    return out


def domain_exit(curve: Sequence[float], in_window: Sequence[bool], *, factor: float | None = None,
                absolute: float | None = None, calibrate: str = "last") -> dict[str, Any]:
    """The first step outside the window where ``curve`` crosses its threshold: the exit, in one function.

    Exactly one of the two thresholds is given, because the two curves need different ones and mixing them is how a
    board stops meaning anything:

    * ``absolute`` -- for a curve that already has a scale: the error against the reference is compared with the
      tolerance that this library calls a reproduced field, so "left its domain" reads the same for a model that was
      excellent in its window and for one that was mediocre. It is the column the scoreboard ranks by.
    * ``factor`` -- for a curve with no natural scale: the residual is compared with what the same model showed at
      the *boundary* of its window, times the factor. The boundary and not the worst step: the stream is
      continuous, so the model has to stay near the state it had when it left the window, and calibrating on the
      worst transient inside it (a large residual in the first epochs of training, say) would leave a monitor that
      never fires. ``calibrate="max"`` is kept for a caller that wants the stricter bar.

    Returns ``step`` (``None`` when the curve never crosses), the threshold, the value at that step and the number
    the threshold was calibrated on, so the exit can be recomputed from the published curve without re-deriving it.
    """
    if (factor is None) == (absolute is None):
        raise ValueError("give exactly one of factor (relative) and absolute (a published tolerance)")
    if calibrate not in ("last", "max"):
        raise ValueError(f"calibrate must be 'last' or 'max', got {calibrate!r}")
    values = np.asarray(curve, dtype=np.float64)
    inside = np.asarray(in_window, dtype=bool)
    if values.shape != inside.shape:
        raise ValueError(f"curve and window must be the same length, got {values.shape[0]} and {inside.shape[0]}")
    if absolute is not None:
        threshold = float(absolute)
        calibrated, how = None, "the published tolerance"
    else:
        window_values = values[inside]
        window_values = window_values[np.isfinite(window_values)]
        if window_values.size == 0:
            return {"step": None, "threshold": None, "value": None, "calibrated_on": None,
                    "reason": "no finite value inside the window to calibrate on"}
        calibrated = float(window_values[-1] if calibrate == "last" else window_values.max())
        threshold = float(factor) * calibrated
        how = f"{factor} x the model's own {'last' if calibrate == 'last' else 'worst'} value inside the window"
    for i in np.flatnonzero(~inside):
        if np.isfinite(values[i]) and values[i] > threshold:
            return {"step": int(i), "threshold": threshold, "value": float(values[i]),
                    "calibrated_on": calibrated, "criterion": how, "reason": ""}
    return {"step": None, "threshold": threshold, "value": None, "calibrated_on": calibrated,
            "criterion": how, "reason": "never crossed the threshold"}


def _summarise(values: Sequence[float | None], *, never: float) -> dict[str, Any]:
    """Per-seed scalars to one row: mean, min, max, and how many seeds produced a number at all.

    ``min`` is the **first** step the model was seen to leave, and ``left_in_seeds`` is how many seeds left at all; the
    mean substitutes ``never`` for a seed that survived, which is a stand-in and not an observation, so the board ranks
    by the minimum (see :func:`scoreboard`). ``within_stream`` says "every seed left", ``left_in_any_seed`` says "at
    least one did" -- with two seeds those are different statements, and a model that left in one and survived the
    other must not be printed as if it had never left.
    """
    clean = [never if v is None else float(v) for v in values]
    left = sum(1 for v in values if v is not None)
    return {"mean": float(np.mean(clean)) if clean else None,
            "min": float(np.min(clean)) if clean else None,
            "max": float(np.max(clean)) if clean else None,
            "within_stream": bool(clean and left == len(values)),
            "left_in_any_seed": left > 0,
            "left_in_seeds": left,
            "n_seeds": len(values)}


def _warm_up() -> None:
    """Pay torch's initialisation before the clock starts.

    The first model trained in a cold process pays for importing torch, building the thread pool and the first
    kernel launch; charging that to whichever family happens to run first would put a constant of several seconds in
    a cost column that is supposed to compare models. One forward and one backward pass of a throwaway operator is
    enough, and doing it here rather than by training a second time keeps it outside every measurement.
    """
    import torch

    from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator

    net = FourierNeuralOperator(in_channels=2, out_channels=1, width=8, modes=4, layers=1, use_grid=True)
    out = net(torch.zeros(2, 2, 16, requires_grad=True))
    out.y.sum().backward()


def _config(model: str, seed: int, field: str) -> BenchConfig:
    budget = PROTOCOL["budget"]
    size = budget[model]
    return BenchConfig(field=field, train_fraction=PROTOCOL["window"]["train_fraction"], epochs=budget["epochs"],
                       lr=budget["lr"], batch_size=budget["batch_size"], seed=int(seed),
                       width=size.get("width", 32), modes=size.get("modes", 8), layers=size.get("layers", 4),
                       hidden=tuple(size.get("hidden", (64, 64, 64))), tolerance=PROTOCOL["exit"]["tolerance"])


MODELS: tuple[str, ...] = ("fno", "pinn")
SUPERVISION = {"fno": "the window's field (data-supervised neural operator)",
               "pinn": "the window's field + the governing equation (PDE residual in the loss)"}


@contextmanager
def _one_thread():
    """One CPU thread for a whole case, restored afterwards.

    Two reasons, and both are about the number this board publishes. torch splits a reduction over as many threads as
    it has, and the split changes the rounding, so two processes with the same seed and the same code can disagree on
    the step at which an error curve crosses the tolerance -- the exit is not reproducible until the split is fixed.
    And it makes the cost column mean something: a wide FNO kernel scales with threads while a thin MLP does not, so
    letting each take whatever the machine offers measures torch's scheduler as much as the model. Measured here on a
    loaded machine: 2.5 s per 40 FNO epochs on one thread against 4.0 s on four.
    """
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_case(case_id: str, *, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Run every model over the same stream, every seed, and return the measurements the scoreboard is built from."""
    case = get_case(case_id)
    seeds = tuple(PROTOCOL["seeds"] if seeds is None else seeds)
    with _one_thread():
        return _run_case(case, seeds)


def _run_case(case: BattleCase, seeds: Sequence[int]) -> dict[str, Any]:
    """The body of :func:`run_case`, with the thread count already fixed by the caller."""
    field = case.build()
    in_window = field.t <= case.split_at
    _warm_up()
    models: dict[str, Any] = {}
    predictions: dict[str, list[np.ndarray]] = {}
    for model in MODELS:
        runs: list[dict[str, Any]] = []
        predictions[model] = []
        for seed in seeds:
            cfg = _config(model, seed, field.field)
            started = time.time()
            result, pred, _ = run_and_predict(field.bundle, cfg, model, residual=field.residual_training)
            wall = time.time() - started
            predictions[model].append(pred)
            error = relative_l2_curve(pred, field.exact)
            residual = pde_residual_curve(pred, field.residual)
            alarm = domain_exit(residual, in_window, factor=PROTOCOL["exit"]["residual_factor"])
            exit_ = domain_exit(error, in_window, absolute=PROTOCOL["exit"]["tolerance"])
            runs.append({"seed": int(seed), "train_s": result.seconds, "predict_s": result.predict_seconds,
                         "total_s": wall, "parameters": int(result.parameters),
                         "relative_l2": [float(v) for v in error],
                         "pde_residual": [None if not np.isfinite(v) else float(v) for v in residual],
                         "residual_alarm": alarm, "error_exit": exit_})
        models[model] = {
            "kind": "neural operator" if model == "fno" else "physics-informed network",
            "supervision": SUPERVISION[model],
            "parameters": runs[0]["parameters"],
            "runs": runs,
            "summary": _model_summary(runs, in_window, never=float(len(field.t))),
        }
    block: dict[str, Any] = {
        "case": case.as_dict(),
        "stream": {"n_steps": int(len(field.t)), "t": [float(v) for v in field.t],
                   "in_window": [bool(v) for v in in_window],
                   "window": "t <= split_at; the model saw only these steps",
                   "cost_s": {"window": float(field.t[in_window].max())}},
        "models": models,
        "model_order": list(MODELS),
        "reference": {"kind": case.reference,
                      "pde_residual": [None if not np.isfinite(v) else float(v)
                                       for v in pde_residual_curve(field.exact, field.residual)]},
    }
    block["scoreboard"] = scoreboard(block)
    block["disagreement"] = disagreement(block, _spread(predictions, field.exact))
    return block


def _spread(predictions: Mapping[str, Sequence[np.ndarray]], exact: np.ndarray) -> np.ndarray:
    """``||u_a - u_b|| / ||u_ref||`` per step, from the seed both models ran first -- the pair the record scores.

    Comparing two models trained with *different* seeds would fold the seed spread into the disagreement, which is
    not a disagreement between families but between two draws. The same seed is the closest thing to "the same
    condition" two families can be given.
    """
    a, b = MODELS
    left, right = predictions[a][0], predictions[b][0]
    return np.linalg.norm(left - right, axis=1) / np.maximum(np.linalg.norm(exact, axis=1), 1e-300)


def _model_summary(runs: Sequence[Mapping[str, Any]], in_window: np.ndarray, *,
                   never: float) -> dict[str, Any]:
    """Per-model scalars: what it cost, what it got wrong, and where it left.

    The error inside the window is reported next to the error outside it, and never averaged together: a model that
    is excellent on the steps it trained on and useless after them has one number close to zero and the other close
    to one, and the mean of the two describes neither.
    """
    inside_error: list[float] = []
    final_error: list[float] = []
    outside_error: list[float] = []
    exit_steps: list[float | None] = []
    alarm_steps: list[float | None] = []
    train_s: list[float] = []
    predict_s: list[float] = []
    for run in runs:
        curve = np.asarray(run["relative_l2"], dtype=np.float64)
        train_s.append(float(run["train_s"]))
        predict_s.append(float(run["predict_s"]))
        exit_steps.append(None if run["error_exit"]["step"] is None else float(run["error_exit"]["step"]))
        alarm_steps.append(None if run["residual_alarm"]["step"] is None else
                           float(run["residual_alarm"]["step"]))
        inside_error.append(float(np.max(curve[in_window])))
        outside_error.append(float(np.mean(curve[~in_window])))
        final_error.append(float(curve[-1]))
    return {"rel_l2_in_window_max": {"mean": float(np.mean(inside_error))},
            "rel_l2_out_window_mean": {"mean": float(np.mean(outside_error))},
            "rel_l2_final": {"mean": float(np.mean(final_error)), "min": float(np.min(final_error)),
                             "max": float(np.max(final_error))},
            "error_exit_step": _summarise(exit_steps, never=never),
            "residual_alarm_step": _summarise(alarm_steps, never=never),
            "train_s": {"mean": float(np.mean(train_s)), "min": float(np.min(train_s)),
                        "max": float(np.max(train_s))},
            "predict_s": {"mean": float(np.mean(predict_s))},
            "exit_threshold": {"mean": float(np.mean([r["error_exit"]["threshold"] for r in runs])),
                               "alarm_threshold": float(np.mean([r["residual_alarm"]["threshold"] for r in runs]))}}


def scoreboard(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The rows, ranked by where the measurement says each model left its domain.

    The ranking key is the one the issue asks for and the one that is measured against the reference: the **first**
    step past the window where the error crossed the threshold, over the seeds -- the minimum, not the mean. The mean
    is wrong here in the way this whole module is about: a seed that survived the stream has no exit, and
    substituting the end of the stream for it and averaging blends an observation with a stand-in. A model that left
    at step 30 in one seed and never in the other then carries 35.5, which puts it *ahead* of a model that survived
    every seed -- and "who left its domain first" is exactly the question the mean cannot answer. A model that never
    left any seed is placed at the end of the stream, which is a result and not a missing value. Ties go to the model
    that left in **more** seeds -- between two models first seen to leave at the same step, the one that did it in
    every seed is the worse of the two -- then to the smaller final error, then to the cheaper training, so the order
    is total and reproducible from the record alone, which is what lets ``check`` fail an inverted table.
    """
    rows: list[dict[str, Any]] = []
    for model in block["model_order"]:
        entry = block["models"][model]
        summary = entry["summary"]
        ex = summary["error_exit_step"]
        alarm = summary["residual_alarm_step"]
        rows.append({
            "model": model,
            "kind": entry["kind"],
            "supervision": entry["supervision"],
            "left_at_step": round(float(ex["min"]), 3),
            "left_at_step_mean": round(float(ex["mean"]), 3),
            "left_within_stream": bool(ex.get("left_in_any_seed", ex["within_stream"])),
            "left_in_seeds": int(ex.get("left_in_seeds", int(ex["within_stream"]))),
            "n_seeds": int(ex.get("n_seeds", 1)),
            "left_at_t": None,
            "alarm_at_step": round(float(alarm["min"]), 3),
            "alarm_at_step_mean": round(float(alarm["mean"]), 3),
            "alarm_within_stream": bool(alarm.get("left_in_any_seed", alarm["within_stream"])),
            "alarm_in_seeds": int(alarm.get("left_in_seeds", int(alarm["within_stream"]))),
            "final_rel_l2": round(float(summary["rel_l2_final"]["mean"]), 6),
            "train_s": round(float(summary["train_s"]["mean"]), 3),
            "predict_s": round(float(summary["predict_s"]["mean"]), 6),
            "parameters": int(entry["parameters"]),
        })
    rows.sort(key=lambda r: (r["left_at_step"], -r["left_in_seeds"], r["final_rel_l2"], r["train_s"], r["model"]))
    times = block["stream"]["t"]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        step = int(round(row["left_at_step"]))
        row["left_at_t"] = float(times[min(step, len(times) - 1)])
    return rows


def disagreement(block: Mapping[str, Any], spread: Sequence[float]) -> dict[str, Any]:
    """The step where the two models are furthest apart, and which one the reference favours there.

    ``spread[i]`` is ``||u_a - u_b|| / ||u_ref||`` at step ``i``: how far apart the two *fields* are, measured
    against the reference so it is a relative number and not a scale of whichever model happens to be large. It is
    computed from the two predictions while they are in memory and only the curve is kept -- the fields themselves
    would dwarf the record, and the curve is what the claim is about.

    The step is taken outside the window: inside it both models were fitted to the same data, so a disagreement
    there is a training artefact, not the question. The verdict is the error of each model against the reference at
    that same step -- the reference is the third opinion that settles the argument.
    """
    models = block["model_order"]
    if len(models) != 2:
        return {"available": False, "reason": f"a disagreement needs exactly two models, got {models}"}
    a, b = models
    curve_a = np.asarray(block["models"][a]["runs"][0]["relative_l2"], dtype=np.float64)
    curve_b = np.asarray(block["models"][b]["runs"][0]["relative_l2"], dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    in_window = np.asarray(block["stream"]["in_window"], dtype=bool)
    outside = np.flatnonzero(~in_window & np.isfinite(spread))
    if outside.size == 0:
        return {"available": False, "reason": "the stream has no step outside the window"}
    worst = int(outside[int(np.argmax(spread[outside]))])
    winner = a if curve_a[worst] <= curve_b[worst] else b
    return {"available": True, "step": worst, "t": float(block["stream"]["t"][worst]),
            "relative_l2": {a: float(curve_a[worst]), b: float(curve_b[worst])},
            "spread": [float(v) for v in spread], "spread_at_step": float(spread[worst]),
            "reference_favours": winner,
            "reason": "the fields are furthest apart at this step (see spread); the model with the smaller error "
                      "against the reference at the same step is the one the reference agrees with"}


def run_all(*, seeds: Sequence[int] | None = None, only: Sequence[str] | None = None) -> dict[str, Any]:
    ids = [c.id for c in CASES] if only is None else [get_case(i).id for i in only]
    protocol = dict(PROTOCOL)
    protocol["seeds"] = list(PROTOCOL["seeds"])
    protocol["metrics"] = list(PROTOCOL["metrics"])
    budget = dict(PROTOCOL["budget"])
    budget["pinn"] = {**budget["pinn"], "hidden": list(budget["pinn"]["hidden"])}
    protocol["budget"] = budget
    protocol["window"] = dict(PROTOCOL["window"])
    protocol["grid"] = dict(PROTOCOL["grid"])
    protocol["exit"] = dict(PROTOCOL["exit"])
    protocol["ranking"] = {**PROTOCOL["ranking"], "tie_break": list(PROTOCOL["ranking"]["tie_break"])}
    return {"schema": SCHEMA, "protocol": protocol,
            "cases": {cid: run_case(cid, seeds=seeds) for cid in ids}}


# ------------------------------------------------------------------ the table and the page
def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _where(step: float, at_t: float, within_stream: bool, in_seeds: int | None = None,
           n_seeds: int | None = None, *, never: str = "never (survived the whole stream)") -> str:
    """Where a model left, in the coordinates a reader needs: the step of the stream, its time, and how many seeds.

    The seed count is not decoration. A model that left in one seed of two has left, and printing it as "never" --
    which is what the mean of a censored exit used to do -- states the opposite of the measurement."""
    if not within_stream:
        return never
    where = f"step {int(round(step))} (t = {_fmt(at_t)})"
    if in_seeds is not None and n_seeds is not None and in_seeds < n_seeds:
        where += f", in {in_seeds} of {n_seeds} seeds"
    return where


def table(results: Mapping[str, Any]) -> str:
    """The scoreboard, one row per model per case: cost, error, and where it left -- plus what it claimed."""
    head = ("| rank | model | cost (train / predict) | final relative L2 | left its domain (first seed to leave) | "
            "its own residual alarm (first seed to fire) |")
    lines = [head, "|---|---|---:|---:|---|---|"]
    for _cid, block in results["cases"].items():
        times = block["stream"]["t"]
        for row in block["scoreboard"]:
            alarm_at = min(int(round(row["alarm_at_step"])), len(times) - 1)
            alarm = _where(row["alarm_at_step"], times[alarm_at], row["alarm_within_stream"],
                           row.get("alarm_in_seeds"), row.get("n_seeds"), never="never (the monitor did not fire)")
            left = _where(row["left_at_step"], row["left_at_t"], row["left_within_stream"],
                          row.get("left_in_seeds"), row.get("n_seeds"))
            lines.append(f"| {row['rank']} | {row['model']} | {row['train_s']:.1f} s / "
                         f"{row['predict_s'] * 1e3:.1f} ms | {_fmt(row['final_rel_l2'], 3)} | "
                         f"{left} | {alarm} |")
        lines += ["", f"Case `{block['case']['id']}`: {block['case']['name']}. "
                      f"Window: t <= {block['case']['split_at']} (the steps the models were trained on); the rest "
                      f"of the stream is the battle. {len(times)} steps, "
                      f"{int(sum(block['stream']['in_window']))} of them inside the window.", ""]
    return "\n".join(lines).rstrip() + "\n"


def _floor(block: Mapping[str, Any]) -> str:
    """The operator's own residual on the reference field, as one sentence per case.

    A monitor needs its floor stated next to it. On Burgers the shock layer is thinner than a cell past ``t = 1/pi``,
    so the exact solution's residual there is not a rounding error -- it is the grid. Publishing that number is how a
    reader tells "the model left its domain" apart from "this grid cannot resolve the field", instead of reading the
    alarm column as if it were absolute.
    """
    finite = [float(v) for v in block["reference"]["pde_residual"] if v is not None]
    if not finite:
        return "operator floor: the reference's residual is not finite anywhere on this grid"
    return (f"operator floor: the reference's own residual peaks at {_fmt(float(np.max(finite)), 3)} (the `t` edges "
            f"excluded), so a model's residual near that number is the grid, not the model")


def page(results: Mapping[str, Any]) -> str:
    """The documentation page, generated from the results file so the published numbers cannot drift from the run."""
    b = results["protocol"]
    ranking = b.get("ranking") or {}
    ranking_line = (f"* **ranking**: {ranking['key']}; ties go to {', then to '.join(ranking['tie_break'])}"
                    if ranking else "* **ranking**: this record does not declare one; the order is its own claim")
    parts = [
        "# Battle of the models: the scoreboard (#373)",
        "",
        "Two models, one stream of conditions, and one question that a leaderboard's mean error cannot answer:",
        "**where did each of them stop being usable?** The table below is generated from",
        f"`{RESULTS.as_posix()}` by `python scripts/battle.py --docs` -- do not edit it by hand.",
        "",
        "It is the measured half of the arena the issue asks for: the numbers, the cost and the exit, not the",
        "animation. What each model is allowed to see, what it costs, what it gets wrong at the end of the stream",
        "and the step at which it left the domain it was validated on are all here, with the criterion written",
        "down below so the exit can be recomputed from the curve.",
        "",
        "## The protocol",
        "",
        f"* **seeds**: {', '.join(str(s) for s in b['seeds'])} -- every model is trained once per seed; the error and "
        "the cost are reported as the mean over them, and the exit as the first seed that left (see *ranking*)",
        f"* **budget**: {b['budget']['epochs']} epochs, lr {b['budget']['lr']}, batch {b['budget']['batch_size']}, "
        f"identical for both models; the size knobs are each family's own (FNO {b['budget']['fno']}, "
        f"PINN {b['budget']['pinn']})",
        f"* **grid**: {b['grid']['nx']} points in x, {b['grid']['nt']} steps in t",
        f"* **window**: {b['window']['train_fraction']:.0%} of the stream, `t <= {b['window']['split_at']}` -- both "
        "models are trained on those steps only and scored on all of them",
        f"* **exit, the truth column**: the model left its domain at the first step outside the window where the "
        f"relative L2 against the reference is above `{b['exit']['tolerance']}` -- the same tolerance this library "
        "uses to call a field reproduced, so the number reads the same for a model that was excellent in its window "
        "and for one that was mediocre",
        f"* **exit, the model's own column**: the residual alarm fires at the first step outside the window where the "
        f"model's residual exceeds `{b['exit']['residual_factor']}` times the value it had at "
        f"`{b['exit']['residual_calibrated_on']}` -- a reference-free monitor, calibrated on the state the model was "
        "in when it left the window, which is the only calibration available without an answer to compare against",
        f"* **residual**: {b['residual_operator']}",
        ranking_line,
        f"* **cost**: {b['cost']}",
        "",
        "## The scoreboard",
        "",
        table(results),
        "A model that never left any seed is placed at the end of the stream -- surviving is a result, not a missing",
        "value. The order is total and recomputable from the record, which is what lets",
        "`python scripts/battle.py --check` refuse to publish it out of order.",
        "",
    ]
    for cid, block in results["cases"].items():
        c = block["case"]
        times = block["stream"]["t"]
        parts += [f"## Case `{cid}` -- {c['name']}", "",
                  f"* equation: `{c['equation']}`",
                  f"* reference: {c['reference']}",
                  f"* window: t <= {c['split_at']}",
                  f"* {_floor(block)}",
                  ""]
        for model in block["model_order"]:
            entry = block["models"][model]
            summary = entry["summary"]
            exit_step = summary["error_exit_step"]
            alarm_step = summary["residual_alarm_step"]
            left_at_t = times[min(int(round(exit_step["min"])), len(times) - 1)]
            left = _where(exit_step["min"], left_at_t, exit_step.get("left_in_any_seed", exit_step["within_stream"]),
                          exit_step.get("left_in_seeds"), exit_step.get("n_seeds"))
            alarm_at = min(int(round(alarm_step["min"])), len(times) - 1)
            alarm = _where(alarm_step["min"], times[alarm_at],
                           alarm_step.get("left_in_any_seed", alarm_step["within_stream"]),
                           alarm_step.get("left_in_seeds"), alarm_step.get("n_seeds"),
                           never="never (the monitor did not fire)")
            per_seed = ", ".join(
                f"seed {r['seed']}: "
                f"{'never' if r['error_exit']['step'] is None else 'step ' + str(r['error_exit']['step'])}"
                for r in entry["runs"])
            exit_fired = bool(exit_step.get("left_in_any_seed", exit_step["within_stream"]))
            alarm_fired = bool(alarm_step.get("left_in_any_seed", alarm_step["within_stream"]))
            alarm_threshold = _fmt(summary["exit_threshold"]["alarm_threshold"], 3)
            if alarm_fired:
                alarm_line = f"* its own residual alarm first fired at {alarm} (threshold {alarm_threshold})"
            else:
                alarm_line = f"* its own residual alarm never fired (threshold {alarm_threshold})"
                if exit_fired:
                    alarm_line += (" while the reference column says the model left its domain: a monitor that stays "
                                   "silent on a model that left is a finding, not a blank")
            parts += [f"### {model} -- {entry['kind']}", "",
                      f"* sees: {entry['supervision']}",
                      f"* {entry['parameters']:,} parameters",
                      f"* cost: {summary['train_s']['mean']:.1f} s to train "
                      f"[{summary['train_s']['min']:.1f}, {summary['train_s']['max']:.1f}] over the seeds, "
                      f"{summary['predict_s']['mean'] * 1e3:.1f} ms to predict the whole stream",
                      f"* relative L2, worst inside the window: {_fmt(summary['rel_l2_in_window_max']['mean'], 3)}; "
                      f"mean outside it: {_fmt(summary['rel_l2_out_window_mean']['mean'], 3)}; "
                      f"last step: {_fmt(summary['rel_l2_final']['mean'], 3)} "
                      f"[{_fmt(summary['rel_l2_final']['min'], 3)}, {_fmt(summary['rel_l2_final']['max'], 3)}]",
                      f"* first left its domain at {left} "
                      f"(threshold {_fmt(summary['exit_threshold']['mean'], 3)}); the seed spread of that step is "
                      f"[{_fmt(exit_step['min'], 3)}, {_fmt(exit_step['max'], 3)}] and the seeds read {per_seed}",
                      alarm_line,
                      ""]
        d = block["disagreement"]
        if d.get("available"):
            parts += ["### Where they disagree and the reference decides", "",
                      f"At step {d['step']} (t = {_fmt(d['t'])}) the two fields are furthest apart: the distance "
                      f"between them is {_fmt(d['spread_at_step'], 3)} of the reference's norm, and at that step the "
                      f"errors are "
                      f"{', '.join(f'{k} {_fmt(v, 3)}' for k, v in d['relative_l2'].items())}. "
                      f"The reference favours **{d['reference_favours']}**.", ""]
        else:
            parts += ["### Where they disagree", "", f"Not available: {d.get('reason', '')}", ""]
    parts += ["## Reproducing it", "",
              "```bash",
              "python scripts/battle.py --run      # train both models over the stream, rewrite the results and the page",
              "python scripts/battle.py --docs     # regenerate this page from the results file (no training)",
              "python scripts/battle.py --check    # the gate: declaration, record and page agree, order included",
              "```",
              "",
              "The cost is a property of the machine and is published as information; a reproduction compares the",
              "errors and the exits, never the seconds.", ""]
    return "\n".join(parts)


def write(results: Mapping[str, Any], *, root: Path | str = ".") -> tuple[Path, Path]:
    root = Path(root)
    path = root / RESULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    page_path = root / PAGE
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(page(results), encoding="utf-8")
    return path, page_path


def check(root: Path | str = ".") -> list[str]:
    """The gate: the declaration is complete, the published order is the one the measurements give, page in sync."""
    root = Path(root)
    problems: list[str] = []
    for field in REQUIRED_PROTOCOL_FIELDS:
        if field not in PROTOCOL:
            problems.append(f"the protocol does not declare {field!r}")
    if len(PROTOCOL["seeds"]) < 2:
        problems.append("one seed has no spread across seeds")
    if PROTOCOL["exit"]["residual_factor"] <= 1:
        problems.append("an exit factor of 1 declares a model out of its domain for the smallest wobble")
    results_path = root / RESULTS
    if not results_path.is_file():
        return problems + [f"{RESULTS.as_posix()} is missing: run python scripts/battle.py --run"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("schema") != SCHEMA:
        problems.append(f"schema is {results.get('schema')!r}, expected {SCHEMA!r}")
    for cid, block in results.get("cases", {}).items():
        published = block.get("scoreboard", [])
        derived = scoreboard(block)
        if published != derived:
            problems.append(f"case {cid}: the published scoreboard is not the one the measurements give "
                            f"(published {[r['model'] for r in published]}, derived "
                            f"{[r['model'] for r in derived]})")
        rows = {r["model"]: r for r in published}
        if set(rows) != set(block.get("model_order", [])):
            problems.append(f"case {cid}: the scoreboard does not have one row per model")
        for rank, row in enumerate(published, start=1):
            if row.get("rank") != rank:
                problems.append(f"case {cid}: row {row.get('model')!r} carries rank {row.get('rank')}, "
                                f"published in position {rank}")
    page_path = root / PAGE
    if not page_path.is_file():
        problems.append(f"{PAGE.as_posix()} is missing: run python scripts/battle.py --docs")
    elif page_path.read_text(encoding="utf-8") != page(results):
        problems.append(f"{PAGE.as_posix()} is not what the results file generates")
    return problems
