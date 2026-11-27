"""From an inverse model to an equation that can be checked, published and solved again (#400).

A PINN with an unknown term, a neural ODE or a neural operator learns a law and hides it in weights. This module
reads it back out as an equation, and then does the part that matters: **it puts the equation back into a solver and
compares its extrapolation against the network that produced it**. The equation is a hypothesis to verify, not a
curve fit.

The pipeline, one function per stage (all of them public, so a stage can be run alone):

1. **Sampling** -- :func:`field_dataset` / :func:`module_dataset` query a model where it is trustworthy: inside the
   box of the training points, outside of which a query is refused (:class:`Support`), optionally behind a distance
   test and a conformal band. Derivatives come from ``torch.autograd`` through the network, never from finite
   differences on noisy data -- that is the whole advantage of distilling from the model instead of from the raw
   field. :class:`PolynomialSurface` is a cheap differentiable stand-in for a solution known only on a grid.
2. **Symbolic regression** -- several back ends over the same samples, each returning a Pareto front of complexity
   against error: SINDy (:class:`~pinneapple_analysis.inverse_problems.missing_term.SINDyIdentifier`, Brunton et al.
   2016) and genetic programming
   (:func:`~pinneapple_analysis.inverse_problems.gp_symbolic.discover_expression`, #353); PySR is an optional extra
   that is used when it is installed and never required. Units, where they are declared, prune the candidates
   (:class:`~pinneapple_analysis.inverse_problems.gp_symbolic.PhysicsConstraints` for GP, a column filter for SINDy).
3. **Identifiability** -- every resample of the rows (bootstrap) and every member of an ensemble is fitted again;
   a term is kept when it survives most of them, and a coefficient carries the interval of the fits that had it. A
   term seen in one fit only is noise, and is reported as such.
4. **Verification** -- :func:`verify` scores the equation and the network that produced it against a reference on
   held-out points and outside the support; :func:`simulate_evolution` puts the equation into
   :mod:`pinneapple_physics.symbolic_pde` and integrates it, so the comparison is between dynamics, not between two
   evaluations.
5. **The record** -- :class:`EquationReport`: the SymPy expression, its LaTeX and its Python, the coefficients with
   their intervals and units, the support it was learned on, the verification scores, and the handoff
   (:meth:`EquationReport.to_system`, :meth:`EquationReport.to_pinn`) that gives the equation back to the PINN
   solver as a known term. Every quantity that is *not* available says so in :attr:`EquationReport.unavailable`
   with the reason -- no interval invented from an ensemble that did not run.

.. code-block:: python

    from pinneapple_analysis.inverse_problems.symbolic_distillation import (
        DistillationConfig, distill, field_dataset, simulate_evolution, verify)

    data = field_dataset(surface, points, coords=["x", "t"], columns={"u": ("u", ()), "u_x": ("u", ("x",)),
                         "u_xx": ("u", ("x", "x")), "u_t": ("u", ("t",))}, target="u_t")
    report = distill(data, config=DistillationConfig(backends=("sindy", "gp"), units={"u": "m/s", "u_x": "1/s"}))
    print(report.equation(), report.latex)
    report = verify(report, cases)                       # against held-out and out-of-distribution references
    U = simulate_evolution(report.to_text("u"), x=x, t=t, u0=U[:, 0])

Honest limits, measured in ``docs/guides/symbolic_distillation.md``: the SINDy arm inherits the strong form's bias
(noisy derivatives let one term stand in for part of another -- :func:`~pinneapple_analysis.inverse_problems.
discovery.discover_pde` has the weak form and names the disagreement); the bootstrap measures sampling stability, not
model error, so a bias shared by every resample stays; PySR needs Julia; and a term whose effect is below the fit
error of the network cannot be recovered at all.

References
----------
Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Discovering governing equations from data by sparse
identification of nonlinear dynamical systems. *PNAS* 113(15), 3932-3937.
Rudy, S. H., Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2017). Data-driven discovery of partial differential
equations. *Science Advances* 3(4), e1602614.
Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D. & Ho, S. (2020). Discovering
symbolic models from deep learning with inductive biases. *NeurIPS* 33 (arXiv:2006.11287).
Fasel, U., Kutz, J. N., Brunton, B. W. & Brunton, S. L. (2022). Ensemble-SINDy: Robust sparse model discovery in the
low-data, high-noise limit. *Proc. R. Soc. A* 478, 20210904.
Angelopoulos, A. N. & Bates, S. (2022). A gentle introduction to conformal prediction and distribution-free
uncertainty quantification. arXiv:2107.07511.
"""
from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Support",
    "Dataset",
    "DistillationConfig",
    "EquationReport",
    "OptionalBackendMissing",
    "PolynomialSurface",
    "field_dataset",
    "module_dataset",
    "outside_support",
    "distill",
    "verify",
    "simulate_evolution",
]


class OptionalBackendMissing(RuntimeError):
    """A back end that is an optional extra (PySR, and Julia with it) is not installed, and says how to get it."""


# ---------------------------------------------------------------------------
# where the model may be queried
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Support:
    """The region a model was asked about: a box per feature, and how far outside it a query is still allowed.

    ``bounds``
        ``{feature: (lo, hi)}`` -- the box of the points the model was fitted on, widened by ``margin``.
    ``reference``
        the rows themselves, kept for the distance test (:func:`outside_support`): a query far from every training
        row is outside the support even when it sits inside the box (a diagonal of a box is not data).
    ``distance_factor``
        how many times the 99th percentile of the reference rows' own nearest-neighbour distance a query may sit
        from its nearest reference row. ``None`` turns the distance test off.
    ``band_limit``
        a conformal half-width above which a query is refused (the 99th percentile of the band on the reference
        rows when :meth:`from_reference` measured it). ``None`` when no band was given.
    """

    bounds: dict[str, tuple[float, float]]
    names: tuple[str, ...] = ()
    reference: np.ndarray | None = None
    distance_factor: float | None = 2.0
    distance_limit: float | None = None
    band_limit: float | None = None
    n_reference: int = 0
    margin: float = 0.0
    notes: tuple[str, ...] = ()

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_reference(cls, rows: Mapping[str, Any] | np.ndarray, *, names: Sequence[str] | None = None,
                       margin: float = 0.0, distance_factor: float | None = 2.0, band: np.ndarray | None = None,
                       quantile: float = 0.99) -> Support:
        """The support of a set of fitted points: the box, the distance limit, and the band limit when ``band`` (a
        per-row conformal half-width, on the same rows) is given."""
        columns = _as_features(rows, names)
        labels = tuple(columns)
        bounds = {k: (float(np.min(v)) - margin * (float(np.max(v)) - float(np.min(v))),
                      float(np.max(v)) + margin * (float(np.max(v)) - float(np.min(v))))
                  for k, v in columns.items()}
        matrix = np.column_stack([columns[k] for k in labels])
        limit = None
        if distance_factor is not None and len(matrix) > 1:
            own = _nearest(matrix, matrix, leave_self_out=True)
            limit = float(np.quantile(own, quantile)) * float(distance_factor)
            if not np.isfinite(limit) or limit <= 0:
                limit = None
        notes = [f"the box of {len(matrix)} fitted rows, widened by {margin:.0%} of each coordinate's range"]
        band_limit = None
        if band is not None:
            band = np.asarray(band, dtype=float).ravel()
            if band.shape != (len(matrix),):
                raise ValueError(f"the band has {band.size} values for {len(matrix)} rows")
            band_limit = float(np.quantile(band, quantile))
            notes.append(f"a query whose conformal half-width exceeds {band_limit:.3g} (the {quantile:.0%} "
                         f"quantile of the band on the fitted rows) is refused")
        return cls(bounds=bounds, names=labels, reference=matrix, distance_factor=distance_factor,
                   distance_limit=limit, band_limit=band_limit, n_reference=len(matrix), margin=float(margin),
                   notes=tuple(notes))

    # -- use ------------------------------------------------------------------
    def _queries(self, queries: Mapping[str, Any] | np.ndarray) -> np.ndarray:
        """The query rows as a matrix in ``self.names`` order (a mapping is read by those names)."""
        if isinstance(queries, Mapping):
            missing = [name for name in self.names if name not in queries]
            if missing:
                raise ValueError(f"the support is over {list(self.names)}; the queries have no {missing}. Pass the "
                                 f"query coordinates, not the feature columns derived from them")
            return np.column_stack([np.asarray(queries[name], dtype=float).ravel() for name in self.names])
        return np.atleast_2d(np.asarray(queries, dtype=float))

    def inside_box(self, queries: Mapping[str, Any] | np.ndarray) -> np.ndarray:
        """The rows inside every coordinate's box."""
        matrix = self._queries(queries)
        mask = np.ones(len(matrix), dtype=bool)
        for i, name in enumerate(self.names):
            lo, hi = self.bounds[name]
            mask &= (matrix[:, i] >= lo) & (matrix[:, i] <= hi)
        return mask

    def contains(self, queries: Mapping[str, Any] | np.ndarray, band: np.ndarray | None = None) -> np.ndarray:
        """The full test: inside the box, close enough to a fitted row, and with a conformal band no wider than the
        one the fitted rows had. ``band`` is the per-row half-width of the query when the caller has one."""
        matrix = self._queries(queries)
        mask = self.inside_box(matrix)
        if self.reference is not None and self.distance_limit is not None:
            mask &= _nearest(matrix, self.reference) <= self.distance_limit
        if band is not None and self.band_limit is not None:
            mask &= np.asarray(band, dtype=float).ravel() <= self.band_limit
        return mask

    def to_dict(self) -> dict[str, Any]:
        return {"bounds": {k: [float(lo), float(hi)] for k, (lo, hi) in self.bounds.items()},
                "names": list(self.names),
                "distance_factor": self.distance_factor, "distance_limit": self.distance_limit,
                "band_limit": self.band_limit, "n_reference": self.n_reference, "margin": self.margin,
                "notes": list(self.notes),
                "reference_rows": ("not serialised: the distance test keeps its rows in memory, and the box and "
                                   "the limits that came from them are what a reader needs")}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Support:
        bounds = {k: tuple(v) for k, v in dict(document.get("bounds") or {}).items()}
        names = tuple(document.get("names") or bounds)
        return cls(bounds=bounds, names=names,   # type: ignore[arg-type]
                   distance_factor=document.get("distance_factor"), distance_limit=document.get("distance_limit"),
                   band_limit=document.get("band_limit"), n_reference=int(document.get("n_reference") or 0),
                   margin=float(document.get("margin") or 0.0), notes=tuple(document.get("notes") or ()))

    def __str__(self) -> str:
        box = ", ".join(f"{k} in [{lo:.4g}, {hi:.4g}]" for k, (lo, hi) in self.bounds.items())
        return f"{box} ({self.n_reference} rows)"


def _nearest(queries: np.ndarray, reference: np.ndarray, *, leave_self_out: bool = False,
             block: int = 512) -> np.ndarray:
    """Distance from each query row to its nearest reference row, in chunks (no scipy, no correlation matrix)."""
    queries = np.atleast_2d(np.asarray(queries, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    out = np.empty(len(queries), dtype=float)
    for start in range(0, len(queries), block):
        chunk = queries[start:start + block]
        d = np.sqrt(np.maximum(((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(-1), 0.0))
        if leave_self_out and len(chunk) == len(reference):
            np.fill_diagonal(d, np.inf)
        out[start:start + block] = d.min(axis=1)
    return out


def outside_support(queries: Mapping[str, Any] | np.ndarray, reference: Mapping[str, Any] | np.ndarray, *,
                    names: Sequence[str] | None = None, factor: float = 2.0, quantile: float = 0.99) -> np.ndarray:
    """``True`` where a query is farther from every reference row than ``factor`` times what the reference rows
    themselves are from their neighbours (at ``quantile``).

    This is the cheap half of the trust layer: a model interpolates between the rows it was fitted on, so a query
    with no row nearby is an extrapolation whatever the box says. A conformal band is the other half --
    pass it to :meth:`Support.contains`."""
    reference_matrix = np.column_stack([v for v in _as_features(reference, names).values()])
    columns = _as_features(queries, names)
    queries_matrix = np.column_stack([v for v in columns.values()])
    own = _nearest(reference_matrix, reference_matrix, leave_self_out=True)
    limit = float(np.quantile(own, quantile)) * float(factor)
    return _nearest(queries_matrix, reference_matrix) > limit


# ---------------------------------------------------------------------------
# what was read from the model
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """The samples a symbolic regression is fitted to: named feature columns, one target column, and where they
    came from.

    ``features`` is ``{name: column}`` with the names the equation will be written in -- ``u``, ``u_x``, ``u_xx``
    for a field, ``x``, ``y``, ``z`` for a neural ODE's states. Those same names are what
    :meth:`EquationReport.to_text` and the PDE compiler read, so a column called ``u_xx`` becomes the second
    derivative of ``u`` when the equation is handed to :mod:`pinneapple_physics.symbolic_pde`.
    """

    features: dict[str, np.ndarray]
    target: np.ndarray
    points: np.ndarray | None = None
    support: Support | None = None
    refused: int = 0
    name: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.features = {str(k): np.asarray(v, dtype=float).ravel() for k, v in self.features.items()}
        self.target = np.asarray(self.target, dtype=float).ravel()
        lengths = {len(v) for v in self.features.values()} | {len(self.target)}
        if len(lengths) != 1:
            sizes = {k: len(v) for k, v in self.features.items()}
            raise ValueError(f"the columns have different lengths: {sizes}, target {len(self.target)}")
        if not np.all(np.isfinite(self.target)):
            raise ValueError("the target has non-finite values: a symbolic regression cannot be fitted to them")

    @property
    def n(self) -> int:
        return len(self.target)

    @property
    def names(self) -> list[str]:
        return list(self.features)

    def matrix(self) -> np.ndarray:
        return np.column_stack([self.features[k] for k in self.features])

    def take(self, index: np.ndarray) -> Dataset:
        """The same samples at ``index`` (a bootstrap resample, or a held-out split)."""
        return Dataset(features={k: v[index] for k, v in self.features.items()}, target=self.target[index],
                       points=None if self.points is None else self.points[index], support=self.support,
                       name=self.name, notes=list(self.notes))

    def __len__(self) -> int:
        return self.n


def _as_features(rows: Mapping[str, Any] | np.ndarray, names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    if isinstance(rows, Mapping):
        return {str(k): np.asarray(v, dtype=float).ravel() for k, v in rows.items()}
    array = np.asarray(rows, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    labels = list(names) if names is not None else [f"x{i}" for i in range(array.shape[1])]
    if len(labels) != array.shape[1]:
        raise ValueError(f"{array.shape[1]} columns but {len(labels)} names")
    return {label: array[:, i] for i, label in enumerate(labels)}


def field_dataset(model: Callable[[Any], Any], points: np.ndarray, *, coords: Sequence[str],
                  columns: Mapping[str, tuple[str, tuple[str, ...]]], target: str, outputs: Sequence[str] | None
                  = None, features: Sequence[str] | None = None, support: Support | None = None,
                  band: np.ndarray | None = None, name: str = "") -> Dataset:
    """Query a field model at ``points`` and read ``columns`` off it **through autograd**.

    ``model`` is anything callable on a tensor ``(N, len(coords)) -> (N, len(outputs))``; a network's output column
    ``outputs[i]`` is the field. ``columns`` names each column that will be built:
    ``{"u": ("u", ()), "u_x": ("u", ("x",)), "u_t": ("u", ("t",))}`` is "the field, its x-derivative and its
    time-derivative"; the tuple is ``(output, differentiation coordinates in order)``. ``target`` picks the column
    the equation will be an expression for; ``features`` defaults to every column but the target.

    ``support`` and ``band`` are applied here, once: rows the support refuses are dropped and counted in
    :attr:`Dataset.refused`, so no later stage can accidentally sample outside it.
    """
    import torch

    outputs = list(outputs) if outputs is not None else _outputs_of(columns)
    for label, (out, _wrt) in columns.items():
        if out not in outputs:
            raise ValueError(f"column {label!r} reads output {out!r}, which is not in outputs={outputs}")
    if target not in columns:
        raise ValueError(f"the target {target!r} is not one of the columns {list(columns)}")
    features = list(features) if features is not None else [k for k in columns if k != target]

    X = torch.tensor(np.atleast_2d(np.asarray(points, dtype=float)), dtype=torch.float64, requires_grad=True)
    raw = model(X)
    if not torch.is_tensor(raw):
        raise TypeError("field_dataset needs a callable on torch tensors (a network, or PolynomialSurface)")
    raw = raw.reshape(len(X), -1) if raw.ndim > 1 else raw.reshape(len(X), 1)
    if raw.shape[1] != len(outputs):
        raise ValueError(f"the model returns {raw.shape[1]} outputs, {len(outputs)} were declared: {outputs}")

    cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def column(out: str, wrt: tuple[str, ...]):
        j = outputs.index(out)
        base = raw[:, j:j + 1]
        key = (out, tuple(wrt))
        if not wrt:
            return base
        if key not in cache:
            current = column(out, tuple(wrt[:-1]))
            axis = list(coords).index(wrt[-1])
            g = torch.autograd.grad(current.sum(), X, create_graph=True)[0][:, axis:axis + 1]
            cache[key] = g
        return cache[key]

    built = {label: column(out, tuple(wrt)).detach().cpu().numpy().ravel() for label, (out, wrt) in columns.items()}
    matrix = np.asarray(points, dtype=float).reshape(len(X), len(coords))
    keep = np.ones(len(X), dtype=bool) if support is None else support.contains(matrix, band=band)
    return Dataset(features={k: built[k][keep] for k in features}, target=built[target][keep],
                   points=matrix[keep], support=support, refused=int((~keep).sum()), name=name,
                   notes=[f"{int(keep.sum())} of {len(X)} queried points were inside the support"])




def _outputs_of(columns: Mapping[str, tuple[str, tuple[str, ...]]]) -> list[str]:
    seen: list[str] = []
    for out, _wrt in columns.values():
        if out not in seen:
            seen.append(out)
    return seen


def module_dataset(model: Callable[[Any], Any], points: np.ndarray, *, names: Sequence[str] | None = None,
                   column: int = 0, target_name: str = "f", support: Support | None = None,
                   band: np.ndarray | None = None, name: str = "") -> Dataset:
    """Query a model that takes the raw coordinates and returns the learned function (a neural ODE's right-hand
    side, a closure, a coefficient field): features are the coordinates, the target is output ``column``.

    Three kinds of model are read the same way, because they all answer "what is the function here": a
    ``torch.nn.Module`` (called on a tensor), a :class:`PolynomialSurface`, and anything that answers on plain
    coordinates -- an object with :meth:`~pinneapple_analysis.inverse_problems.missing_term.NeuralTermDiscovery.
    predict_term` (the term network of :class:`~pinneapple_analysis.inverse_problems.missing_term.
    NeuralTermDiscovery`, so the learned term is distilled by the same pipeline as a network's own output) or a plain
    numpy callable.
    """
    import torch

    matrix = np.atleast_2d(np.asarray(points, dtype=float))
    labels = list(names) if names is not None else [f"x{i}" for i in range(matrix.shape[1])]
    if len(labels) != matrix.shape[1]:
        raise ValueError(f"{matrix.shape[1]} coordinates but {len(labels)} names")
    if isinstance(model, (torch.nn.Module, PolynomialSurface)):
        with torch.no_grad():
            raw = model(torch.tensor(matrix, dtype=torch.float64))
        values = np.asarray(raw.detach().cpu(), dtype=float)
    else:
        answer = model.predict_term(matrix) if hasattr(model, "predict_term") else model(matrix)
        values = np.asarray(np.asarray(answer, dtype=float), dtype=float)
    values = values.reshape(len(matrix), -1)[:, column]
    built = {**{lab: matrix[:, i] for i, lab in enumerate(labels)}, target_name: values}
    keep = np.ones(len(matrix), dtype=bool) if support is None else support.contains(matrix, band=band)
    return Dataset(features={k: built[k][keep] for k in labels}, target=built[target_name][keep], points=matrix[keep],
                   support=support, refused=int((~keep).sum()), name=name)


class PolynomialSurface:
    """A differentiable surface fitted to values known on a grid: a tensor-product Chebyshev regression.

    A solution that exists as numbers on a grid (a PDE-FIND dataset, a classical solver's output) is not a model,
    but it can be made into one in a least-squares solve: ``u(x, t) = sum c_jk T_j(x) T_k(t)``. The result is
    callable on tensors, differentiable to any order by ``torch.autograd``, and fitted in milliseconds -- so a
    distillation benchmark does not have to wait for a PINN to train before it can be read.

    ``fit`` takes the points as an ``(N, d)`` array and the values at them; ``degree`` is per coordinate.
    """

    def __init__(self, coefficients: np.ndarray, degree: Sequence[int], lo: Sequence[float],
                 span: Sequence[float]) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.degree = tuple(int(d) for d in degree)
        self.lo = np.asarray(lo, dtype=float)
        self.span = np.asarray(span, dtype=float)

    # -- fitting --------------------------------------------------------------
    @classmethod
    def fit(cls, points: np.ndarray, values: np.ndarray, *, degree: int | Sequence[int],
            dtype: type = np.float64) -> PolynomialSurface:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        values = np.asarray(values, dtype=dtype).ravel()
        if len(points) != len(values):
            raise ValueError(f"{len(points)} points and {len(values)} values")
        degrees = [int(degree)] * points.shape[1] if isinstance(degree, (int, np.integer)) else [int(d) for d in degree]
        lo = points.min(axis=0)
        span = np.where(points.max(axis=0) - lo > 0, points.max(axis=0) - lo, 1.0)
        design = _chebyshev_design(_scaled(points, lo, span, np), degrees)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        return cls(coefficients, degrees, lo, span)

    # -- use ------------------------------------------------------------------
    def __call__(self, X: Any) -> Any:
        """Evaluate on a torch tensor ``(N, d)``; autograd differentiates through the recurrence."""
        import torch

        if not torch.is_tensor(X):
            X = torch.as_tensor(np.asarray(X, dtype=float))
        design = _chebyshev_design(_scaled(X, self.lo, self.span, torch), self.degree)
        return design @ torch.as_tensor(self.coefficients, dtype=X.dtype, device=X.device)

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Evaluate on plain numpy points (no derivative)."""
        design = _chebyshev_design(_scaled(np.atleast_2d(points), self.lo, self.span, np), self.degree)
        return design @ self.coefficients

    def residual(self, points: np.ndarray, values: np.ndarray) -> float:
        """RMS of the fit on the points it was fitted to (or on a fresh set)."""
        return float(np.sqrt(np.mean((self.predict(points) - np.asarray(values, dtype=float).ravel()) ** 2)))

    def to_dict(self) -> dict[str, Any]:
        return {"coefficients": self.coefficients.tolist(), "degree": list(self.degree),
                "lo": self.lo.tolist(), "span": self.span.tolist()}


def _scaled(points: Any, lo: Any, span: Any, module: Any) -> Any:
    """The points mapped to [-1, 1] per coordinate, in numpy or in torch (same code path, two modules)."""
    if module is np:
        return (np.asarray(points, dtype=float) - np.asarray(lo, dtype=float)) / np.asarray(span, dtype=float) * 2 - 1
    lo_t = module.as_tensor(np.asarray(lo, dtype=float), dtype=points.dtype)
    span_t = module.as_tensor(np.asarray(span, dtype=float), dtype=points.dtype)
    return (points - lo_t) / span_t * 2 - 1


def _chebyshev_design(z: Any, degrees: Sequence[int]) -> Any:
    """The tensor-product Chebyshev design matrix ``prod_i T_{k_i}(z_i)``, in numpy or in torch."""
    module = np if isinstance(z, np.ndarray) else None
    blocks = []
    for i, degree in enumerate(degrees):
        # one column per coordinate: 1-D in numpy (``stack`` adds the axis) and (N, 1) in torch (``cat`` needs it),
        # so a single coordinate gives a (N, k) block instead of a (N, k, 1) one the least squares refuses
        column = z[:, i] if module is not None else z[:, i:i + 1]
        cols = [column * 0 + 1]
        if degree >= 1:
            cols.append(column)
        for _ in range(2, degree + 1):
            cols.append(2 * column * cols[-1] - cols[-2])
        blocks.append(module.stack(cols, axis=1) if module is not None else _cat(cols))
    design = blocks[0]
    for block in blocks[1:]:
        outer = design[:, :, None] * block[:, None, :] if module is not None else design[:, :, None] * block[:, None, :]
        design = outer.reshape(len(design), -1)
    return design


def _cat(columns: list) -> Any:
    import torch

    return torch.cat(columns, dim=1)


# ---------------------------------------------------------------------------
# the candidate library, in the names of the data
# ---------------------------------------------------------------------------

_INDEX = re.compile(r"x(\d+)")


def _rename(label: str, names: Sequence[str]) -> str:
    """``"x0 x1"`` (the candidate library's spelling) becomes ``"u*u_x"``: a product, in the columns' own names."""
    def swap(match: re.Match) -> str:
        i = int(match.group(1))
        return names[i] if i < len(names) else match.group(0)
    return "*".join(_INDEX.sub(swap, part) for part in label.split())


def _term_text(label: str) -> str:
    """``"u u_x"`` becomes ``"u*u_x"``: a product, in the syntax the rest of the library parses."""
    if label.strip() == "1":
        return "1"
    return "*".join(part for part in label.split() if part) if " " in label else label



_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _term_sympy(label: str) -> Any:
    """The SymPy expression of one term label, parsed (never ``eval``-ed) by the GP module's reader.

    A label the expression language cannot read -- a function it does not implement, an operator it does not have --
    is refused here with the reader's own reason. Falling back to a symbol named ``"tanh(u)"`` would make the walker
    report a missing unit for a name that is not a variable at all, and a ``KeyError`` escaping the reader would crash
    the whole units filter for one bad candidate."""
    import sympy as sp

    from .gp_symbolic import parse_expression

    text = _term_text(label)
    if text == "1":
        return sp.Integer(1)
    local = {name: sp.Symbol(name, real=True) for name in _IDENTIFIER.findall(text)}
    try:
        tree = parse_expression(text)
    except (ValueError, SyntaxError, KeyError) as exc:
        if _IDENTIFIER.fullmatch(text):                  # a bare name is a column, not an expression
            return local.get(text, sp.Symbol(text, real=True))
        raise ValueError(f"the term {label!r} is not one the expression language reads: {exc}") from None
    return tree.to_sympy(local)



_BASE = (("m", 0), ("kg", 1), ("s", 2), ("K", 3), ("A", 4), ("mol", 5))


def _unit_text(dim: Sequence[int]) -> str:
    """A dimension vector ``(m, kg, s, K, A, mol)`` as a unit a reader recognises: ``m^2/s``, ``1/s``, ``1``."""
    from fractions import Fraction

    numerator, denominator = [], []
    for symbol, i in _BASE:
        power = Fraction(dim[i]).limit_denominator(4)
        if power == 0:
            continue
        target = numerator if power > 0 else denominator
        target.append(symbol if abs(power) == 1 else f"{symbol}^{abs(power)}")
    if not numerator:
        numerator = ["1"]
    up = "*".join(numerator)
    return up if not denominator else f"{up}/{'*'.join(denominator)}"


def _dimension(label: str, units: Mapping[str, str], cache: dict[str, Any]) -> tuple[Any, str | None]:
    """The dimension of a term label in ``units``, or ``(None, why)`` when the term is not a quantity at all
    (a transcendental of a dimensioned argument, the sum of two unlike dimensions, a symbol whose unit was never
    declared, a function this library's expression reader does not have). The reason is the reader's own message:
    it names what is wrong with the term, never a unit of a name that is not a variable."""
    if label in cache:
        return cache[label]
    from .gp_symbolic import DimensionError, PolicyViolation

    try:
        out = _expr_dimension(_term_sympy(label), units)
        cache[label] = (out, None)
    except (DimensionError, PolicyViolation, ValueError, TypeError, KeyError) as exc:
        cache[label] = (None, str(exc))
    return cache[label]



def _expr_dimension(expr: Any, units: Mapping[str, str]) -> tuple[int, ...]:
    """The dimension exponents of a SymPy expression, with :class:`DimensionError` for an impossible one."""
    import sympy as sp

    from pinneapple_data.physical_units import parse_unit

    dims = {}
    for name, unit in units.items():
        base = parse_unit(str(unit)).dim
        dims[sp.Symbol(name, real=True)] = tuple(base)
    zero = (0,) * 6

    def walk(e: Any) -> tuple[int, ...]:
        if isinstance(e, sp.Number):
            return zero
        if isinstance(e, sp.Symbol):
            if e not in dims:
                raise ValueError(f"the unit of {e} was not declared: the declared units cover "
                                 f"{sorted(str(s) for s in dims)}")
            return dims[e]
        if isinstance(e, sp.Add):
            first = walk(e.args[0])
            for arg in e.args[1:]:
                if walk(arg) != first:
                    raise ValueError(f"{_unit_text(first)} and {_unit_text(walk(arg))} are added")
            return first
        if isinstance(e, sp.Mul):
            out = list(zero)
            for arg in e.args:
                d = walk(arg)
                out = [a + b for a, b in zip(out, d, strict=True)]
            return tuple(out)
        if isinstance(e, sp.Pow):
            base, power = e.args
            if not power.is_number:
                raise ValueError(f"the power {power} is not a number")
            exponent = float(power)
            if exponent != int(exponent):
                raise ValueError(f"the fractional power {power} has no whole-unit dimension")
            d = walk(base)
            return tuple(int(x * exponent) for x in d)
        if isinstance(e, sp.Function):        # sin, cos, exp, log, tanh: a function of a dimensionless argument,
            for arg in e.args:                # whatever its arity, with a dimensionless result
                d = walk(arg)
                if any(d):
                    raise ValueError(f"{e.func.__name__}() takes a dimensionless argument; "
                                     f"its argument is {_unit_text(d)}")
            return zero
        raise ValueError(f"the {type(e).__name__} node {e} has no dimension rule here: only numbers, symbols, "
                         f"sums, products, whole-number powers and functions of a dimensionless argument do")

    return walk(expr)


# ---------------------------------------------------------------------------
# the back ends: one front each
# ---------------------------------------------------------------------------

@dataclass
class _Fit:
    """One fitted model: the terms with their coefficients, the expression, and the front that led to it."""

    backend: str
    coefficients: dict[str, float]          # label -> value, for the terms that survived
    expression: str                         # the right-hand side, as text in the feature names
    front: list[dict[str, Any]]
    complexity: int
    fit_error: float = float("nan")         # relative RMS on the rows it was fitted to
    held_out_error: float = float("nan")    # relative RMS on the held-out rows
    threshold: float = 0.0                  # the STRidge threshold of the SINDy arm
    rejected: list[dict[str, Any]] = field(default_factory=list)     # dropped by the units filter
    notes: list[str] = field(default_factory=list)

    @property
    def terms(self) -> list[str]:
        return list(self.coefficients)


def _relative_rms(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(target ** 2)))
    return float(np.sqrt(np.mean((prediction - target) ** 2)) / (scale if scale > 0 else 1.0))


def _stridge(theta: np.ndarray, target: np.ndarray, labels: Sequence[str],
             relative_threshold: float) -> tuple[np.ndarray, float]:
    """STRidge at a threshold given **relative to the largest coefficient of the unregularised fit**.

    :class:`~pinneapple_analysis.inverse_problems.missing_term.SINDyIdentifier` compares an absolute threshold with
    the coefficients of unit-norm columns, whose size is set by the number of rows and by the size of the target: for
    1400 rows the physical coefficients ``-1.3`` and ``0.08`` become ``56`` and ``5`` there, so a fixed threshold of
    ``0.3`` never removes anything and the sweep it was meant to make is flat. Rescaling the threshold by the largest
    coefficient the plain least squares finds makes the same numbers mean the same thing on any dataset: ``1e-3``
    keeps everything, ``0.3`` keeps only a term carrying a third of the largest one.
    """
    from .missing_term import SINDyIdentifier

    norms = np.linalg.norm(theta, axis=0)
    norms[norms == 0] = 1.0
    reference = float(np.max(np.abs(np.linalg.lstsq(theta / norms, target, rcond=None)[0]))) or 1.0
    threshold = float(relative_threshold) * reference
    coefficients = SINDyIdentifier(threshold=threshold).fit(theta, target, term_names=list(labels)).coefficients
    return coefficients, threshold


def _sindy_fit(dataset: Dataset, fit_rows: np.ndarray, held_out: np.ndarray, config: DistillationConfig) -> _Fit:
    """SINDy (STRidge over the candidate library): the threshold path is the front, the held-out rows choose."""
    from .discovery import _prune

    theta, labels = config._build_library(dataset.matrix(), dataset.names)
    keep, rejected = _units_filter(labels, config)
    theta = theta[:, keep]
    labels = [labels[i] for i in np.flatnonzero(keep)]
    if not labels:
        raise ValueError("every candidate term was refused by the units constraint: check units and target_unit")

    front: list[dict[str, Any]] = []
    best: tuple[float, float, int, np.ndarray] | None = None
    fitted: dict[float, np.ndarray] = {}
    for threshold in config.thresholds:
        xi, absolute = _stridge(theta[fit_rows], dataset.target[fit_rows], labels, float(threshold))
        fitted[float(threshold)] = xi
        error = _relative_rms(theta[held_out] @ xi, dataset.target[held_out])
        score = _aicc(error, len(held_out), int(np.count_nonzero(xi)))
        front.append({"backend": "sindy", "expression": _sindy_text(xi, labels),
                      "complexity": int(np.count_nonzero(xi)), "relative_error": error,
                      "threshold": float(threshold), "threshold_absolute": absolute, "aicc": score})
        if best is None or score < best[0] - 1e-9 or (abs(score - best[0]) < 2 and np.count_nonzero(xi) < best[2]):
            best = (score, float(threshold), int(np.count_nonzero(xi)), xi)
    assert best is not None
    threshold = best[1]
    xi = _prune(theta[fit_rows], dataset.target[fit_rows], fitted[threshold], frac=config.prune)
    coefficients = {labels[i]: float(xi[i]) for i in np.flatnonzero(xi)}
    return _Fit(backend="sindy", coefficients=coefficients, expression=_sindy_text(xi, labels),
                front=_pareto(front), complexity=len(coefficients),
                held_out_error=_relative_rms(theta[held_out] @ xi, dataset.target[held_out]),
                fit_error=_relative_rms(theta[fit_rows] @ xi, dataset.target[fit_rows]), threshold=threshold,
                rejected=rejected, notes=[f"STRidge threshold {threshold:g} chosen on the held-out rows"])


def _pareto(front: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The Pareto front of a set of fits: least complexity for its error, least error for its complexity.

    A threshold that selects one more term but fits worse is *dominated*, and a second threshold that selects the
    very same model is the same point of the front -- neither belongs on a front of complexity against error. Keeping
    them there would let a caller read a front that is not monotone (a longer model, a worse error) and would report
    one model several times as if it were several. The entries keep the arm and threshold that produced them.
    """
    by_complexity: dict[int, dict[str, Any]] = {}
    for entry in front:
        complexity = int(entry["complexity"])
        error = float(entry["relative_error"])
        if not np.isfinite(error):
            continue
        current = by_complexity.get(complexity)
        if current is None or error < float(current["relative_error"]):
            by_complexity[complexity] = dict(entry)
    kept: list[dict[str, Any]] = []
    best_error = float("inf")
    for complexity in sorted(by_complexity):
        entry = by_complexity[complexity]
        if float(entry["relative_error"]) < best_error:
            kept.append(entry)
            best_error = float(entry["relative_error"])
    return kept



def _sindy_text(xi: np.ndarray, labels: Sequence[str]) -> str:
    terms = [f"{float(c):.6g}*({_term_text(lab)})" for c, lab in zip(xi, labels, strict=True) if c != 0]
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def _aicc(relative_error: float, n: int, k: int) -> float:
    """AICc from a relative RMS: the residual sum of squares of a prediction scaled to the target."""
    k = max(k, 1)
    rss = max(relative_error ** 2 * n, 1e-300)
    return n * math.log(rss / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)


def _gp_constraints(config: DistillationConfig, labels: Sequence[str]) -> Any:
    """The physics filter the genetic-programming arm searches under: the caller's, with the declared units of the
    columns and the target added, so the search itself only breeds expressions of the right dimension. The chosen
    model and every replicate are searched under the same filter, otherwise the stability measures a different
    search from the one that produced the equation."""
    if config.units is None or config.target_unit is None:
        return config.constraints
    from .gp_symbolic import PhysicsConstraints

    base = config.constraints or PhysicsConstraints()
    return PhysicsConstraints(
        units={k: v for k, v in config.units.items() if k in labels}, target=config.target_unit,
        prefactor=base.prefactor, sign=base.sign, limits=base.limits, symmetry=base.symmetry,
        balance=base.balance, weights=base.weights, limit_tol=base.limit_tol)


def _gp_fit(dataset: Dataset, fit_rows: np.ndarray, held_out: np.ndarray, config: DistillationConfig) -> _Fit:
    """Genetic programming (#353): the Pareto front of the run is the front, its best feasible member is the model.

    An entry's ``relative_error`` is the error of the engine's own search (its front is non-dominated on that
    measure, and the model it reports is on it); the held-out error that chooses between the arms is
    :attr:`_Fit.held_out_error`, scored on the rows the search never saw."""
    from .gp_symbolic import GeneticProgramming

    labels = list(dataset.features)
    constraints = _gp_constraints(config, labels)
    engine = GeneticProgramming(population=config.population, generations=config.generations, seed=config.seed,
                                constraints=constraints, operators=config.operators, max_nodes=config.max_nodes)
    columns = {k: dataset.features[k][fit_rows] for k in labels}
    result = engine.fit(columns, dataset.target[fit_rows])
    front = _pareto([{"backend": "gp", "expression": _scaled_text(entry), "complexity": int(entry.get("nodes", 0)),
                      "relative_error": float(entry.get("relative_error", float("nan"))),
                      "feasible": bool(entry.get("feasible", True))} for entry in result.expressions])
    best = result.best
    if best is None:
        return _Fit(backend="gp", coefficients={}, expression="0", front=front, complexity=0,
                    held_out_error=float("inf"), fit_error=float("inf"),
                    notes=["genetic programming found no expression the physics filter accepts"])
    expression = _scaled_text(best)
    prediction = _evaluate_expression(best, columns, labels)
    error = _relative_rms(prediction, dataset.target[fit_rows])
    held = _relative_rms(_evaluate_expression(best, {k: dataset.features[k][held_out] for k in labels}, labels),
                         dataset.target[held_out])
    return _Fit(backend="gp", coefficients={expression: float(best.get("scale", 1.0))}, expression=expression,
                front=front, complexity=int(best.get("nodes", 0)), fit_error=error, held_out_error=held,
                notes=[f"the best of {len(result.expressions)} expressions on the Pareto front"])


def _scaled_text(entry: Mapping[str, Any]) -> str:
    scale = float(entry.get("scale", 1.0))
    return entry["expression"] if abs(scale - 1.0) <= 1e-12 else f"{scale:.6g}*({entry['expression']})"


def _evaluate_expression(entry: Mapping[str, Any], columns: Mapping[str, np.ndarray],
                         names: Sequence[str] | None = None) -> np.ndarray:
    """Evaluate a GP front entry (its expression and fitted scale) on named columns."""
    from .gp_symbolic import parse_expression

    tree = parse_expression(str(entry["expression"]))
    return float(entry.get("scale", 1.0)) * tree.evaluate({str(k): np.asarray(v, dtype=float).ravel()
                                                           for k, v in columns.items()})


def _pysr_fit(dataset: Dataset, fit_rows: np.ndarray, held_out: np.ndarray, config: DistillationConfig) -> _Fit:
    """PySR, when it is installed (it needs Julia: ``pip install pysr`` and ``python -c "import pysr"`` once)."""
    try:
        from pysr import PySRRegressor
    except ImportError as exc:                                   # pragma: no cover - optional extra
        raise OptionalBackendMissing(
            "the 'pysr' back end needs PySR and Julia: pip install pysr, then `python -c 'import pysr'` once to "
            "install the Julia side. The 'sindy' and 'gp' back ends need nothing.") from exc
    model = PySRRegressor(niterations=int(config.generations) * 2, populations=int(config.population),
                          binary_operators=["+", "-", "*", "/"], unary_operators=list(config.operators),
                          maxsize=config.max_nodes, random_state=config.seed, progress=False)   # pragma: no cover
    matrix = dataset.matrix()
    model.fit(matrix[fit_rows], dataset.target[fit_rows])                                     # pragma: no cover
    frame = model.equations_                                                                    # pragma: no cover
    front = [{"backend": "pysr", "expression": str(row["equation"]), "complexity": int(row["complexity"]),
              "relative_error": _relative_rms(np.asarray(model.predict(matrix[held_out], index=i)),
                                              dataset.target[held_out])}
             for i, (_, row) in enumerate(frame.iterrows())]                                    # pragma: no cover
    best = min(front, key=lambda entry: (entry["complexity"], entry["relative_error"]))          # pragma: no cover
    return _Fit(backend="pysr", coefficients={best["expression"]: 1.0}, expression=best["expression"],
                front=front, complexity=best["complexity"],
                held_out_error=_relative_rms(np.asarray(model.predict(matrix[held_out])), dataset.target[held_out]),
                notes=["PySR: the front is its own multi-objective search"])                    # pragma: no cover


_BACKENDS: dict[str, Callable[..., _Fit]] = {"sindy": _sindy_fit, "gp": _gp_fit, "pysr": _pysr_fit}


def _units_filter(labels: Sequence[str], config: DistillationConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep the candidate terms that are dimensionally consistent with the left-hand side.

    Only a term whose dimension is right can be multiplied by a coefficient of the declared kind and added to the
    rest, so the check is done before the regression: it removes candidates the search would otherwise spend its
    sparsity on. A term whose dimension is wrong is reported in :attr:`_Fit.rejected` with the unit it would need and
    the one it has, never dropped silently.
    """
    keep = np.ones(len(labels), dtype=bool)
    rejected: list[dict[str, Any]] = []
    if config.units is None or config.target_unit is None:
        return keep, rejected
    from pinneapple_data.physical_units import parse_unit

    target = tuple(parse_unit(str(config.target_unit)).dim)
    cache: dict[str, Any] = {}
    for i, label in enumerate(labels):
        dim, why = _dimension(label, config.units, cache)
        if dim is None:
            keep[i] = False
            rejected.append({"term": label, "reason": why})
            continue
        if config.dimensionless_coefficients and tuple(dim) != target:
            keep[i] = False
            rejected.append({"term": label, "reason": f"the term is {_unit_text(dim)}, the left-hand side is "
                                                      f"{_unit_text(target)}: the coefficient would not be "
                                                      f"dimensionless"})
    return keep, rejected


# ---------------------------------------------------------------------------
# identifiability across resamples and ensemble members
# ---------------------------------------------------------------------------

def _replicates(dataset: Dataset, names: Sequence[str], config: DistillationConfig,
                ensemble: Sequence[Dataset] | None) -> dict[str, Any]:
    """What every replicate found: ``n_bootstrap`` row resamples and every ensemble member.

    The refit keeps the threshold (SINDy) or the search budget (GP) of the chosen model: what varies is the data, or
    the member of the ensemble, which is what the stability is meant to measure. For SINDy a replicate gives one
    coefficient vector; for genetic programming it gives one expression, because that is the shape of a GP fit."""
    from .discovery import _prune

    rng = np.random.default_rng(config.seed + 7919)
    index_sets = [rng.integers(0, dataset.n, dataset.n) for _ in range(int(config.n_bootstrap))]
    if dataset.n and config.backend == "sindy":
        theta, labels = config._build_library(dataset.matrix(), names)
        keep, _rejected = _units_filter(labels, config)
        theta = theta[:, keep]
        labels = [labels[i] for i in np.flatnonzero(keep)]
        vectors = []
        for index in index_sets:
            xi, _absolute = _stridge(theta[index], dataset.target[index], labels, config.chosen_threshold)
            vectors.append(_prune(theta[index], dataset.target[index], xi, frac=config.prune))
        for member in ensemble or ():
            _check_member(member, names)
            theta_m, labels_m = config._build_library(member.matrix(), list(names))
            keep_m, _ = _units_filter(labels_m, config)
            labels_m = [label for label, kept in zip(labels_m, keep_m, strict=True) if kept]
            xi, _absolute = _stridge(theta_m[:, keep_m], member.target, labels_m, config.chosen_threshold)
            vectors.append(_prune(theta_m[:, keep_m], member.target, xi, frac=config.prune))
        return {"kind": "coefficient", "labels": labels, "vectors": np.vstack(vectors) if vectors else None}

    if config.backend == "gp":
        from .gp_symbolic import GeneticProgramming

        def search(columns: Mapping[str, np.ndarray], target: np.ndarray, seed: int) -> str:
            engine = GeneticProgramming(population=config.population, generations=config.generations, seed=seed,
                                        constraints=_gp_constraints(config, names), operators=config.operators,
                                        max_nodes=config.max_nodes)
            best = engine.fit(dict(columns), target).best
            return "0" if best is None else _scaled_text(best)

        forms: list[str] = []
        for k, index in enumerate(index_sets):
            forms.append(search({key: dataset.features[key][index] for key in names}, dataset.target[index],
                                config.seed + k + 1))
        for k, member in enumerate(ensemble or ()):
            _check_member(member, names)                # the member is fitted like the data: its own expression
            forms.append(search({key: member.features[key] for key in names}, member.target,
                                config.seed + len(index_sets) + k + 1))
        return {"kind": "form", "labels": [str(config.lhs)], "forms": forms,
                "columns": {key: dataset.features[key] for key in names}}
    return {"kind": "none", "labels": [], "vectors": None}


def _check_member(member: Dataset, names: Sequence[str]) -> None:
    if member.names != list(names):
        raise ValueError(f"an ensemble member has features {member.names}; the data has {list(names)}")


def _identifiability(fit: _Fit, replicates: Mapping[str, Any], config: DistillationConfig) -> dict[str, Any]:
    """Per-term inclusion, interval and stability, from what the replicates found."""
    kind = replicates.get("kind")
    entries: dict[str, Any] = {}
    if kind == "coefficient" and replicates.get("vectors") is not None:
        matrix = np.asarray(replicates["vectors"], dtype=float)
        labels = list(replicates["labels"])
        for i, label in enumerate(labels):
            if label not in fit.coefficients:
                continue
            column = matrix[:, i]
            present = column != 0
            # the interval is over the fits that *had* the term: a zero is a fit that dropped it, not a value of
            # its coefficient, and mixing the zeros in shrinks the interval towards a zero the term never took
            entries[label] = {
                "coefficient": float(fit.coefficients[label]),
                "inclusion": float(present.mean()),
                "interval95": ([float(np.quantile(column[present], 0.025)),
                                float(np.quantile(column[present], 0.975))] if present.any() else None),
                "same_sign": bool((np.sign(column[present]) == np.sign(fit.coefficients[label])).all())
                if present.any() else False}
    elif kind == "form" and replicates.get("forms"):
        forms = list(replicates.get("forms") or [])
        columns = replicates.get("columns") or {}
        same = sum(1 for form in forms if _same_form(form, fit.expression, columns))
        entries[str(fit.expression)] = {
            "coefficient": float(next(iter(fit.coefficients.values()), 1.0)),
            "inclusion": float(same / len(forms)),
            "interval95": None, "same_sign": True, "form": str(fit.expression),
            "interval_reason": ("genetic programming reports one fitted prefactor per expression and the replicates "
                                "found other expressions; there is no per-coefficient interval to quote")}
    else:
        return {"n_replicates": 0, "kind": kind, "terms": {}, "stable": [], "unstable": [],
                "threshold": config.stability,
                "reason": ("no replicate was run: n_bootstrap=0 and no ensemble member was given, so no term has "
                           "an inclusion probability or a coefficient interval")}
    stable = [label for label, entry in entries.items()
              if entry["inclusion"] >= config.stability and entry.get("same_sign", True)]
    unstable = [label for label in entries if label not in stable]
    return {"n_replicates": (len(replicates["vectors"]) if kind == "coefficient"
                             else len(replicates.get("forms") or [])),
            "kind": kind, "terms": entries, "stable": stable, "unstable": unstable,
            "threshold": config.stability,
            "reason": (f"{', '.join(unstable)} did not survive {config.stability:.0%} of the replicates: it is not a "
                       f"property of the model") if unstable else None}


def _same_form(a: str, b: str, columns: Mapping[str, np.ndarray], tolerance: float = 1e-3) -> bool:
    """Whether two expressions are the same law: equal up to the prefactor, measured on the sampled points.

    A run that found *nothing* (``"0"``) is not the same law as one that found something: the zero expression is a
    multiple of every expression with the prefactor 0, so a scale fitted through the origin always matches it, and
    counting those replicates as agreement would report a law as stable because half the replicates discovered
    nothing at all. Two runs are the same law only when both are zero, or both are not.
    """
    with np.errstate(all="ignore"):
        try:
            first = _evaluate_text(a, columns, list(columns))
            second = _evaluate_text(b, columns, list(columns))
        except (ValueError, KeyError, TypeError):
            return False
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        return False
    magnitude = max(float(np.sqrt(np.mean(first ** 2))), float(np.sqrt(np.mean(second ** 2))))
    zero = 1e-12 * max(magnitude, 1.0)                 # relative to the size of the quantities themselves
    first_is_zero, second_is_zero = float(np.abs(first).max()) <= zero, float(np.abs(second).max()) <= zero
    if first_is_zero or second_is_zero:
        return first_is_zero and second_is_zero
    scale = float(np.sum(first * second)) / float(np.sum(second ** 2))
    return float(np.sqrt(np.mean((first - scale * second) ** 2))) <= tolerance * float(np.sqrt(np.mean(first ** 2)))


def _evaluate_text(expression: str, columns: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    """Evaluate an expression given as text in the feature names, on named columns."""
    import sympy as sp

    symbols = [sp.Symbol(name, real=True) for name in names]
    function = sp.lambdify(symbols, _parse_math(expression), modules=["numpy"])
    values = function(*[np.asarray(columns[name], dtype=float) for name in names])
    return np.broadcast_to(np.asarray(values, dtype=float), np.shape(np.asarray(columns[names[0]], dtype=float)))



def _parse_math(text: str) -> Any:
    """Parse a right-hand side given as text in the feature names (``2.5*u_xx - u*u_x``) into SymPy."""
    import sympy as sp

    from .gp_symbolic import parse_expression

    names = sorted(set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(text))))
    local = {n: sp.Symbol(n, real=True) for n in names}
    tree = parse_expression(str(text).replace(" ", ""))
    return tree.to_sympy(local)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

@dataclass
class EquationReport:
    """An equation read out of a model, and the evidence it is the right one.

    Nothing here is defaulted to a plausible-looking number: a quantity that could not be measured is ``None`` in its
    field **and** carries the reason in :attr:`unavailable`.
    """

    lhs: str
    expression: str                                   # the right-hand side, in the feature names
    terms: list[dict[str, Any]]                       # term, coefficient, interval95, inclusion, unit
    latex: str
    code: str                                         # a Python function that evaluates the right-hand side
    sympy_expression: str                             # the expression as SymPy prints it (readable, exact)
    features: list[str]
    support: dict[str, Any]
    backend: str
    front: list[dict[str, Any]]
    fit_error: float | None                           # relative RMS against the model, on the samples
    held_out_error: float | None                      # ... on rows held out of the fit
    n_samples: int
    identifiability: dict[str, Any]
    verification: dict[str, Any] | None
    units: dict[str, str] | None
    target_unit: str | None
    coefficient_unit: str | None                      # unit of the whole right-hand side, from the target
    unavailable: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    # -- reading it -----------------------------------------------------------
    def equation(self) -> str:
        """``u_t = 0.0985*u_xx - 1.003*u*u_x``."""
        return f"{self.lhs} = {self.expression}"

    def to_sympy(self) -> Any:
        return _parse_math(self.expression)

    def predict(self, features: Mapping[str, Any]) -> np.ndarray:
        """Evaluate the right-hand side on named columns (the same names the equation is written in)."""
        missing = [k for k in self.features if k not in features]
        if missing:
            raise ValueError(f"the equation needs {self.features}; {missing} were not given")
        return _evaluate_text(self.expression, features, self.features)

    def stable_terms(self) -> list[str]:
        return list(self.identifiability.get("stable") or [])

    def summary(self) -> str:
        parts = [self.equation()]
        if self.held_out_error is not None:
            parts.append(f"held-out error {self.held_out_error:.2%} of the model's own values")
        if self.identifiability.get("n_replicates"):
            parts.append(f"{len(self.stable_terms())} of {len(self.terms)} terms survive "
                         f"{self.identifiability['n_replicates']} replicates")
        if self.verification and self.verification.get("verdict"):
            parts.append(f"verification: {self.verification['verdict']}")
        if self.unavailable:
            parts.append("not available: " + "; ".join(f"{k} ({v})" for k, v in self.unavailable.items()))
        return ". ".join(parts)

    # -- handing it back ------------------------------------------------------
    def to_text(self, field: str = "u", *, time: str = "t") -> str:
        """The right-hand side as text the PDE compiler reads: ``u`` and ``u_xx`` become the field and its second
        derivative. Raises :class:`ValueError` when a feature name is not a field, a coordinate or a derivative."""
        names = self.features
        if field not in names and not any(name.startswith(field + "_") for name in names):
            raise ValueError(f"{field!r} is not one of the equation's features {names}: there is nothing for the "
                             f"compiler to differentiate")
        return f"{field}_{time} = {self.expression}"

    def to_system(self, field: str = "u", *, time: str = "t", conditions: Sequence[Mapping[str, Any]] | None
                  = None) -> Any:
        """A :class:`pinneapple_physics.symbolic_pde.text.EquationSystem` for the equation: the handoff to the
        solver, as an evolution equation ``u_t = <right-hand side>``."""
        from pinneapple_physics.symbolic_pde.text import parse_system

        coords = [c for c in _coords_of(self.features, field) if c != time]
        return parse_system([self.to_text(field, time=time)], list(conditions or ()),
                            coords=coords + [time], fields=[field])

    def to_pinn(self, bounds: Mapping[str, tuple[float, float]], *, field: str = "u", time: str = "t",
                conditions: Sequence[Mapping[str, Any]] | None = None, **options: Any) -> tuple[Any, Any]:
        """``(PINNProblem, model)`` from :func:`pinneapple_physics.symbolic_pde.pinn.pinn_problem`: the discovered
        equation is now a known term the PINN solver can be trained against. ``options`` go to ``pinn_problem``."""
        from pinneapple_physics.symbolic_pde.pinn import pinn_problem

        return pinn_problem(self.to_system(field, time=time, conditions=conditions), dict(bounds), **options)

    # -- the document ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        from pinneapple_core.jsonable import to_jsonable

        document = {k: getattr(self, k) for k in self.__dataclass_fields__}
        document["summary"] = self.summary()
        document["equation"] = self.equation()
        return to_jsonable(document, nonfinite="str")

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> EquationReport:
        names = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(document).items() if k in names})

    def to_json(self, **kwargs: Any) -> str:
        import json

        return json.dumps(self.to_dict(), **kwargs)

    def __str__(self) -> str:
        return self.summary()


def _coords_of(features: Sequence[str], field: str) -> list[str]:
    """The coordinates a field's derivative names use: ``u_xx`` and ``u_t`` give ``x`` and ``t``."""
    found: list[str] = []
    for name in features:
        if name.startswith(field + "_"):
            for letter in name[len(field) + 1:]:
                if letter not in found:
                    found.append(letter)
    return found


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

@dataclass
class DistillationConfig:
    """How the distillation is run.

    ``backends``
        the symbolic-regression arms, tried in this order (``"sindy"``, ``"gp"``, and ``"pysr"`` when installed). A
        back end that is not installed is recorded in the report's ``unavailable`` and the others carry on.
    ``poly_order``, ``include_trig``, ``custom_terms``, ``library``
        the SINDy candidate library (:class:`~pinneapple_analysis.inverse_problems.missing_term.CandidateLibrary`);
        ``library`` replaces the built one when the caller wants terms of their own.
    ``thresholds``
        the STRidge thresholds the SINDy arm sweeps. Each is one candidate model, and the front of the arm keeps
        the ones no other threshold dominates (one point per complexity, the error falling as the model grows). They
        are fractions of the largest coefficient the unregularised fit finds, not absolute values, so the same sweep
        means the same thing whatever the number of rows and the size of the target: ``1e-3`` keeps every candidate,
        ``0.3`` keeps only the terms carrying a third of the dominant one. Each front entry reports both numbers.
    ``prune``
        drop a selected term whose contribution is below this fraction of the target's RMS (a coefficient that
        changes nothing is not a discovery).
    ``stability``
        the fraction of replicates a term has to survive to be reported as stable.
    ``n_bootstrap``
        row resamples (each a refit). ``0``: no interval, no inclusion, and the report says so.
    ``population``, ``generations``, ``operators``, ``max_nodes``, ``constraints``
        the genetic-programming arm (#353).
    ``units``, ``target_unit``, ``dimensionless_coefficients``, ``coefficient_sign``
        dimensional constraints: only a term whose dimension matches the left-hand side can be multiplied by a
        dimensionless coefficient, so ``dimensionless_coefficients=True`` keeps only those. With
        ``dimensionless_coefficients=False`` the check is inside each term (no sum of unlike units, no function of a
        dimensioned argument) and each reported coefficient carries the unit it would have to have.
    ``fit_tolerance``
        the held-out relative error above which the fit check fails.
    """

    backends: tuple[str, ...] = ("sindy", "gp")
    poly_order: int = 2
    include_trig: bool = False
    custom_terms: list[tuple[str, Callable]] | None = None
    library: Any = None
    thresholds: tuple[float, ...] = tuple(np.geomspace(1e-3, 0.3, 12))
    prune: float = 0.01
    stability: float = 0.8
    n_bootstrap: int = 40
    population: int = 120
    generations: int = 10
    operators: tuple[str, ...] = ("sin", "cos", "exp", "log", "sqrt", "tanh")
    max_nodes: int = 24
    constraints: Any = None
    units: dict[str, str] | None = None
    target_unit: str | None = None
    dimensionless_coefficients: bool = False
    coefficient_sign: str | None = None
    fit_tolerance: float = 0.05
    seed: int = 0
    lhs: str = "f"
    # set by the pipeline, not by the caller
    backend: str = ""
    chosen_threshold: float = 0.01

    def _build_library(self, matrix: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        from .missing_term import CandidateLibrary

        library = self.library if self.library is not None else CandidateLibrary(
            poly_order=self.poly_order, include_trig=self.include_trig, custom_terms=self.custom_terms)
        theta, labels = library.build(matrix)
        return theta, [_rename(label, names) for label in labels]


def distill(dataset: Dataset, *, config: DistillationConfig | None = None,
            ensemble: Sequence[Dataset] | None = None) -> EquationReport:
    """Read an equation out of ``dataset`` (the model queried inside its support) and measure how much of it the
    model really says.

    The rows are split once (80/20, seeded); every back end chooses its model on the fitting rows and is scored on the
    held-out ones; the arm with the lower held-out error wins, with the simpler model preferred when they are within
    2 %. The winner is refitted on all the rows, its terms are pruned, and its configuration -- not the data --
    decides what the replicates redo: ``n_bootstrap`` resamples of the rows and every member of ``ensemble``.

    ``ensemble`` is one :class:`Dataset` per member of a model ensemble, queried at the same kind of points (a second
    PINN, another seed of the operator). It is the only evidence about how much the *model* contributes, as opposed to
    how much the sampling does; without it the report says so.
    """
    config = config or DistillationConfig()
    started = time.perf_counter()
    if dataset.n < 10:
        raise ValueError(f"{dataset.n} samples are too few for a held-out split: sample at least 10")

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(dataset.n)
    cut = max(3, int(round(0.8 * dataset.n)))
    fit_rows, held_out = np.sort(order[:cut]), np.sort(order[cut:])

    notes: list[str] = []
    unavailable: dict[str, str] = {}
    fits: dict[str, _Fit] = {}
    for backend in config.backends:
        if backend not in _BACKENDS:
            raise ValueError(f"unknown back end {backend!r}: {sorted(_BACKENDS)}")
        try:
            fits[backend] = _BACKENDS[backend](dataset, fit_rows, held_out, config)
        except OptionalBackendMissing as exc:
            unavailable[f"backend:{backend}"] = str(exc)
            notes.append(f"the {backend!r} back end was not run: {exc}")
            continue
    if not fits:
        raise OptionalBackendMissing(f"no back end could be run: {unavailable}")

    order_of = {name: i for i, name in enumerate(config.backends)}
    ranking = sorted(fits, key=lambda name: (round(fits[name].held_out_error, 4), fits[name].complexity,
                                             order_of[name]))
    winner = ranking[0]
    if len(ranking) > 1:                                        # the simpler arm wins a near-tie
        second = ranking[1]
        simpler, costlier = fits[second], fits[winner]
        if simpler.held_out_error <= costlier.held_out_error * 1.02 and simpler.complexity < costlier.complexity:
            winner = second
            notes.append(f"{second!r} won the near-tie: its held-out error ({simpler.held_out_error:.3%}) is within "
                         f"2 % of {ranking[0]!r} ({costlier.held_out_error:.3%}) with a simpler model "
                         f"({simpler.complexity} against {costlier.complexity})")
    config.backend = winner
    chosen = fits[winner]
    config.chosen_threshold = chosen.threshold if winner == "sindy" else config.thresholds[0]

    # refit the winner on every row, so the reported coefficients use the whole sample
    full = _BACKENDS[winner](dataset, np.arange(dataset.n), np.arange(dataset.n), config)
    if winner == "gp" and full.expression != chosen.expression:
        notes.append(f"the refit on all rows found {full.expression} instead of {chosen.expression}; the held-out "
                     f"error of {chosen.expression} is what chose the arm")
    report_fit = full if winner == "sindy" else chosen
    report_fit.held_out_error = chosen.held_out_error

    replicates = _replicates(dataset, dataset.names, config, ensemble)
    identifiability = _identifiability(report_fit, replicates, config)

    if winner == "sindy" and identifiability.get("unstable"):
        # requirement of the pipeline: only the terms the replicates keep are reported as the equation
        dropped = list(identifiability["unstable"])
        stable = [label for label in report_fit.coefficients if label in identifiability["stable"]]
        xi, theta = _sindy_refit(dataset, stable, config, np.arange(dataset.n))
        report_fit.held_out_error = _relative_rms(theta[held_out] @ xi, dataset.target[held_out])
        report_fit.fit_error = _relative_rms(theta[np.arange(dataset.n)] @ xi, dataset.target)
        report_fit.coefficients = {label: float(value) for label, value in zip(stable, xi, strict=True)}
        report_fit.expression = _sindy_text(xi, stable)
        report_fit.complexity = len(stable)
        notes.append(f"{', '.join(dropped)} did not survive the replicates and are not in the equation; they are "
                     f"reported in identifiability['unstable']")

    terms = _term_table(report_fit, identifiability, config, unavailable)
    expression = report_fit.expression
    coefficient_unit = str(config.target_unit) if config.target_unit is not None else None

    support_document = dataset.support.to_dict() if dataset.support is not None else {
        "bounds": {k: [float(np.min(v)), float(np.max(v))] for k, v in dataset.features.items()},
        "notes": ["no Support was declared: the box of the sampled points is reported, which is what the model was "
                  "actually asked about"]}
    if dataset.refused:
        notes.append(f"{dataset.refused} queried points were refused by the support and are not in the samples")
    if report_fit.rejected:
        shown = "; ".join(f"{row['term']}: {row['reason']}" for row in report_fit.rejected[:6])
        more = f"; and {len(report_fit.rejected) - 6} more" if len(report_fit.rejected) > 6 else ""
        notes.append(f"{len(report_fit.rejected)} candidate terms never reached the regression: the units "
                     f"constraint refused them ({shown}{more})")
    if dataset.support is None:
        unavailable["support"] = ("no Support was given, so the box of the samples is reported rather than the "
                                  "region the model was trained on")

    report = EquationReport(
        lhs=config.lhs, expression=expression, terms=terms,
        latex=_latex_of(config.lhs, expression), code=_python_of(config.lhs, expression, dataset.names),
        sympy_expression=str(_parse_math(expression)), features=list(dataset.names),
        support=support_document, backend=winner, front=_merged_front(fits), fit_error=report_fit.fit_error,
        held_out_error=report_fit.held_out_error, n_samples=dataset.n, identifiability=identifiability,
        verification=None, units=dict(config.units) if config.units else None, target_unit=config.target_unit,
        coefficient_unit=coefficient_unit, unavailable=unavailable, notes=notes + list(report_fit.notes),
        checks=[], seconds=time.perf_counter() - started)
    if not identifiability.get("n_replicates"):
        unavailable["identifiability"] = identifiability.get("reason", "no replicate was run")
    if config.units is None or config.target_unit is None:
        unavailable["coefficient_unit"] = ("no units were declared: the coefficient's unit is what the left-hand "
                                           "side minus each term would need, and that was not computed")
    unavailable["verification"] = ("verification was not run: pass cases to verify() with a reference on held-out "
                                   "and out-of-distribution points")
    report.checks = _distillation_checks(report, config)
    report.seconds = time.perf_counter() - started
    return report


def _sindy_refit(dataset: Dataset, labels: Sequence[str], config: DistillationConfig,
                 rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares for a fixed set of terms on ``rows``: the coefficients of the reported equation.

    Returns the coefficient vector for ``rows`` and the columns of the same terms over **every** row of the dataset,
    so the caller can score the equation on the rows it was not fitted to (``theta[held_out] @ xi``)."""
    theta, all_labels = config._build_library(dataset.matrix(), dataset.names)
    keep, _rejected = _units_filter(all_labels, config)
    labels_all = [all_labels[i] for i in np.flatnonzero(keep)]
    missing = [label for label in labels if label not in labels_all]
    if missing:
        raise ValueError(f"{missing} are not terms of the library")
    columns = theta[:, [labels_all.index(label) for label in labels]]
    if not labels:
        return np.zeros(0), columns
    xi = np.linalg.lstsq(columns[rows], dataset.target[rows], rcond=None)[0]
    return xi, columns


def _term_table(fit: _Fit, identifiability: Mapping[str, Any], config: DistillationConfig,
                unavailable: dict[str, str]) -> list[dict[str, Any]]:
    """One row per term: its value, the interval the replicates give it, and the unit its coefficient needs."""
    entries = identifiability.get("terms") or {}
    rows = []
    for label, value in fit.coefficients.items():
        entry = entries.get(label) or {}
        row = {"term": label, "coefficient": float(value), "interval95": entry.get("interval95"),
               "inclusion": entry.get("inclusion"), "unit": _coefficient_unit(label, config)}
        rows.append(row)
    if fit.backend != "sindy":
        unavailable["per_term_coefficients"] = ("genetic programming fits one prefactor for a whole expression, not "
                                               "a coefficient per term: the table has that prefactor only")
    return rows


def _coefficient_unit(label: str, config: DistillationConfig) -> str | None:
    """The unit a term's coefficient must have for the term to be the left-hand side's unit."""
    if config.units is None or config.target_unit is None:
        return None
    from pinneapple_data.physical_units import parse_unit

    cache: dict[str, Any] = {}
    dim, _why = _dimension(label, config.units, cache)
    if dim is None:
        return None
    target = tuple(parse_unit(str(config.target_unit)).dim)
    return _unit_text(tuple(t - d for t, d in zip(target, dim, strict=True)))


def _merged_front(fits: Mapping[str, _Fit]) -> list[dict[str, Any]]:
    """Every arm's front, one list, each entry saying which arm, the arms in the order they were tried.

    The fronts are kept apart on purpose: ``complexity`` is an arm-specific measure of the model (the number of
    surviving terms for SINDy, the number of nodes for genetic programming), so a front is a statement about one
    search and the points of two arms are not comparable one by one -- what compares them is the held-out error, and
    that is what chooses the winner. Within an arm the list is a front, monotone by construction: one point per
    complexity with the error falling as the model grows.
    """
    out: list[dict[str, Any]] = []
    for name, fit in fits.items():
        out.extend(dict(entry, backend=name) for entry in fit.front)
    return out


def _latex_of(lhs: str, expression: str) -> str:
    import sympy as sp

    try:
        return sp.latex(sp.Eq(sp.Symbol(lhs), _parse_math(expression)))
    except (sp.SympifyError, ValueError, TypeError):
        return f"{lhs} = {expression}"


def _python_of(lhs: str, expression: str, names: Sequence[str]) -> str:
    """A function that evaluates the right-hand side, with the feature names as its parameters."""
    import sympy as sp

    arguments = ", ".join(names)
    function_name = re.sub(r"[^0-9A-Za-z_]", "_", lhs) or "rhs"
    if function_name[0].isdigit():
        function_name = "_" + function_name
    try:
        body = sp.printing.pycode(_parse_math(expression))
    except (sp.SympifyError, ValueError, TypeError):
        body = expression
    body = body.replace("math.", "np.").replace("numpy.", "np.")
    imports = "import numpy as np\n\n" if "np." in body else ""
    return f"{imports}def {function_name}({arguments}):\n    return {body}\n"


def _distillation_checks(report: EquationReport, config: DistillationConfig) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {"code": "distill.held_out_fit", "status": "pass" if (report.held_out_error or 0) <= config.fit_tolerance
         else "fail",
         "detail": f"the equation predicts the model's own values to {report.held_out_error:.2%} on rows held out of "
                   f"the fit (allowed {config.fit_tolerance:.0%})", "value": report.held_out_error}]
    if report.identifiability.get("n_replicates"):
        stable = set(report.identifiability.get("stable") or [])
        unstable = [row["term"] for row in report.terms if row["term"] not in stable]
        dropped = list(report.identifiability.get("unstable") or [])
        checks.append({"code": "distill.stable_terms", "status": "pass" if not unstable else "fail",
                       "detail": (f"every term of the equation survives {report.identifiability['n_replicates']} "
                                  f"replicates" if not unstable else
                                  f"{', '.join(unstable)} did not survive {config.stability:.0%} of the "
                                  f"{report.identifiability['n_replicates']} replicates")
                                 + (f"; dropped before the equation was refitted: {', '.join(dropped)}"
                                    if dropped else ""),
                       "value": unstable or None})
    else:
        checks.append({"code": "distill.stable_terms", "status": "fail",
                       "detail": "no replicate was run (n_bootstrap=0 and no ensemble), so no term has a stability "
                                 "score; the equation is one fit, not an identified law", "value": None})
    return checks


# ---------------------------------------------------------------------------
# verification: the equation against the model that produced it
# ---------------------------------------------------------------------------

def verify(report: EquationReport, cases: Sequence[Mapping[str, Any]], *,
           tolerance: float = 0.05) -> EquationReport:
    """Score the equation against the network that produced it, case by case, and write the verdict into the report.

    A case is a mapping with ``name``, ``features`` (named columns, as the equation is written in), ``target`` (what
    the network says there), ``reference`` (the truth, where it is known) and ``kind`` --
    ``"inside"``, ``"held_out"`` or ``"out_of_distribution"``. With a reference for both arms the equation wins a case
    when its relative RMS is lower; the verdict of the whole run is decided on the out-of-distribution cases, because
    reproducing the model where the model is trustworthy is the fit, not the discovery.

    ``simulator`` on a case (``callable(report, case) -> values``) replaces the pointwise evaluation, which is how a
    rollout through :func:`simulate_evolution` is scored: the same comparison, between dynamics.
    """
    if not cases:
        raise ValueError("verify needs at least one case")
    rows: list[dict[str, Any]] = []
    for case in cases:
        name = str(case.get("name") or f"case{len(rows)}")
        kind = str(case.get("kind") or "held_out")
        reference = case.get("reference")
        reference = None if reference is None else np.asarray(reference, dtype=float).ravel()
        row: dict[str, Any] = {"name": name, "kind": kind, "equation_error": None, "network_error": None,
                               "winner": None}
        target = case.get("target")
        if reference is not None:
            scale = float(np.sqrt(np.mean(reference ** 2))) or 1.0
            if case.get("simulator") is not None:
                simulated = np.asarray(case["simulator"](report, case), dtype=float).ravel()
                row["equation_error"] = float(np.sqrt(np.mean((simulated - reference) ** 2)) / scale)
            else:
                row["equation_error"] = _relative_rms(report.predict(case["features"]), reference)
            if target is not None:
                row["network_error"] = _relative_rms(np.asarray(target, dtype=float).ravel(), reference)
                row["winner"] = "equation" if row["equation_error"] <= row["network_error"] else "network"
            else:
                row["winner"] = "equation" if row["equation_error"] <= tolerance else "unverified"
        rows.append(row)

    # ``is not None``: a network that reproduces the reference *exactly* has an error of 0.0, which is a value and
    # not a missing one -- testing its truthiness would demote the graded case it belongs to and let the verdict be
    # decided on the held-out rows instead of on the out-of-distribution ones
    out_of_distribution = [row for row in rows
                           if row["kind"] == "out_of_distribution" and row["network_error"] is not None]
    graded = out_of_distribution or [row for row in rows if row["network_error"] is not None]
    if not graded:
        verdict = "not_run"
        detail = "no case carried both a reference and the network's own values"
    else:
        equation = float(np.mean([row["equation_error"] for row in graded]))
        network = float(np.mean([row["network_error"] for row in graded]))
        verdict = "equation_wins" if equation < network else "network_wins"
        where = "out of distribution" if out_of_distribution else "on the graded cases"
        detail = (f"{where}: the equation is {equation:.2%} from the reference, the network that produced it "
                  f"{network:.2%}")
    report.verification = {
        "verdict": verdict, "detail": detail, "cases": rows, "tolerance": tolerance,
        "reference": "out_of_distribution" if out_of_distribution else "held_out",
        "checks": [{"code": "distill.verification", "status": "pass" if verdict == "equation_wins" else "fail",
                    "detail": detail, "value": None}]}
    report.checks = [check for check in report.checks if check["code"] != "distill.verification"]
    report.checks.extend(report.verification["checks"])
    report.unavailable.pop("verification", None)
    if verdict == "not_run":
        report.unavailable["verification"] = detail
    return report


def simulate_evolution(equation: Sequence[str] | str, *, x: np.ndarray, t: np.ndarray, u0: np.ndarray,
                       field: str = "u", method: str = "spectral", coefficients: Mapping[str, float] | None = None,
                       rtol: float = 1e-6, atol: float = 1e-9) -> np.ndarray:
    """Integrate ``u_t = <right-hand side>`` by the method of lines, the right-hand side compiled twice: the text is
    parsed by :mod:`pinneapple_physics.symbolic_pde.text`, and the derivatives in it are taken by
    :class:`pinneapple_physics.symbolic_pde.SymbolicPDE` on the grid (``method="spectral"`` for a periodic grid of any
    derivative order, ``"finite_difference"`` for orders 1 and 2).

    ``equation`` is one right-hand side in the compiler's syntax (``"0.1*u_xx - u*u_x"``) or a whole equation
    (``"u_t = 0.1*u_xx - u*u_x"``); ``coefficients`` gives the value of every name in it that is not a coordinate,
    a field or a number (``{"nu": 0.1}`` for ``"u_t = nu*u_xx"``). The grid must be uniform and, for the spectral
    method, periodic.

    Returns the field on ``(len(x), len(t))``. The boundary values are the ones the solver produces (periodic), never
    taken from the data: a verification that borrows the data's boundary values is not testing the equation there.
    """
    import sympy as sp
    import torch
    from scipy.integrate import solve_ivp

    from pinneapple_physics.symbolic_pde import GridAxisSpec, SymbolicPDE
    from pinneapple_physics.symbolic_pde.text import parse_system

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    text = (list(equation) if isinstance(equation, (list, tuple)) else [equation])[0]
    if "=" not in text:
        text = f"{field}_t = {text}"
    system = parse_system([text], [], coords=["x", "t"], fields=[field], coefficients=dict(coefficients or {}))
    missing = sorted(system.used_coefficients() - set(system.coefficients))
    if missing:
        raise ValueError(f"{missing} appear in {text!r} and have no value: pass coefficients={{...}}")
    rhs = system.pdes[0].rhs.xreplace({sp.Symbol(k, real=True): sp.Float(v)
                                       for k, v in system.coefficients.items()})
    spatial = next(c for c in system.coords if c != "t")

    coords = [sp.Symbol(spatial, real=True)]
    fields = [sp.Function(field)]
    length = float(x[-1] - x[0] + (x[1] - x[0]))
    pde = SymbolicPDE(rhs, coords, fields, grad_method=method,
                      grid=GridAxisSpec((spatial,), (-1,), (length,)))
    compiled = pde.to_grid_residual_fn()

    def rate(_time: float, values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = compiled({field: torch.tensor(values.reshape(1, -1), dtype=torch.float64)})
        return out.detach().cpu().numpy().ravel()

    solution = solve_ivp(rate, (float(t[0]), float(t[-1])), np.asarray(u0, dtype=float).ravel(), t_eval=t,
                         method="BDF", rtol=rtol, atol=atol)
    if not solution.success or solution.y.shape[1] != len(t):
        raise RuntimeError(f"the integration of {text} failed: {solution.message}")
    return solution.y
