"""Um objeto de restrição, aplicado nos três lugares onde uma resposta pode violar um limite (#221, roadmap X34).

A biblioteca já sabia recusar um projeto por uma exigência dura (#172, ``pinneapple_design.design_gate``), medir
conservação e condição de contorno (#159, ``pinneapple_physics.conservation``) e carregar incerteza numa frente de
Pareto (#80, ``pinneapple_design.design_optimizer.robust``). Cada uma dessas coisas checava o seu próprio limite, do
seu próprio jeito: um limite escrito para o portão de projeto não era o mesmo número que o otimizador recebia, e
nenhum dos dois era o que a guarda de deploy perguntava. O resultado prático é o pior possível -- um modelo que passa
no treino, converge no otimizador e é servido violando o limite que o cliente escreveu.

Este módulo é a ponte que faltava: **um** :class:`pinneapple_physics.physical_problem.Constraint`, o mesmo objeto,
com **uma** aritmética (:func:`residual`), entregue em três pontos de aplicação:

``penalty`` / ``Trainer(constraints=...)``
    o limite entra no laço de treino como termo diferenciável da perda, então a rede aprende a respeitá-lo;
``optimizer_constraints`` / ``PhysicsOptimizer(constraints=...)``
    o mesmo limite vira ``ineq``/``eq`` do ``pinneapple_core.optim``, então o otimizador o trata como restrição de
    verdade (SQP, Lagrangeano aumentado, evolução) e não como penalidade disfarçada;
``ConstraintGuard``
    o mesmo limite, checado sobre a predição no momento em que ela é servida, recusando em vez de corrigir.

E, fora dos três, a mesma função :func:`check_all` responde "quantas vezes este limite foi violado" sobre um ensemble,
que é o que o relatório de decisão (#219) precisa: a probabilidade de violação de uma restrição é a média de
:func:`check` sobre as realizações, não uma estatística inventada em outro módulo.

A aritmética é uma só, e é a do ``PhysicsOptimizer``: **residual** ``g`` com ``g <= 0`` significando "satisfeita".
``g`` é ``quantidade - limite`` para ``<=``, ``limite - quantidade`` para ``>=`` e ``|quantidade - limite|`` para
``==``; para uma restrição escrita como expressão (``"mass_kg - payload_kg >= 2"``) é a mesma subtração com o sinal da
comparação dobrado dentro, lida pelo avaliador seguro de ``pinneapple_data.expressions`` (nada de ``eval``). Um ``g``
negativo é folga: é ele que o relatório ordena para dizer *qual* caso dirige o risco.

>>> import pinneapple as pp
>>> from pinneapple_physics.constraint_engine import check
>>> T = pp.Constraint("temperatura", quantity="T_max", op="<=", value=85.0, unit="degC", family="safety")
>>> check(T, {"T_max": 92.5}).violated
True
>>> round(check(T, {"T_max": 92.5}).margin, 3)          # 7.5 degC above the limit
7.5

A família (``physical``, ``conservation``, ``operational``, ``material``, ``actuator``, ``safety``) **não** muda a
aritmética: é o rótulo que o relatório agrupa, que o otimizador usa para nomear a restrição e que a guarda põe na
recusa. Fingir que seis famílias se checam de seis maneiras diferentes seria criar seis motores de novo.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TOL", "ConstraintCheck", "ConstraintReport", "ConstraintViolation", "ConstraintGuard",
           "residual", "check", "check_all", "penalty", "optimizer_constraints"]

#: a margin this close to zero is a tie, not a violation
TOL = 1e-9


# --------------------------------------------------------------------------- the record of one check
@dataclass(frozen=True)
class ConstraintCheck:
    """One constraint asked once: satisfied or not, by how much, and against which limit.

    ``margin`` keeps the sign that :func:`residual` uses (``<= 0`` satisfied), so ``slack`` -- ``-margin`` -- is how
    far inside the limit the answer sits. An answer exactly at the limit has zero slack, which is a tie and not a
    margin: a report that rounded it to "satisfied" would hide the case that is next to fail.
    """
    name: str
    family: str
    satisfied: bool
    margin: float
    value: float | None = None
    limit: float | None = None
    op: str = ""
    hard: bool = True
    detail: str = ""

    @property
    def slack(self) -> float:
        return -self.margin

    @property
    def violated(self) -> bool:
        return not self.satisfied

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "satisfied": self.satisfied, "margin": self.margin,
                "slack": self.slack, "value": self.value, "limit": self.limit, "op": self.op, "hard": self.hard,
                "detail": self.detail}


@dataclass
class ConstraintReport:
    """Every constraint asked over one answer (or over one ensemble member)."""
    checks: list[ConstraintCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no **hard** constraint is violated. A soft constraint never blocks."""
        return not self.hard_violated

    @property
    def violated(self) -> list[ConstraintCheck]:
        return [c for c in self.checks if c.violated]

    @property
    def hard_violated(self) -> list[ConstraintCheck]:
        return [c for c in self.checks if c.violated and c.hard]

    def by_name(self) -> dict[str, ConstraintCheck]:
        return {c.name: c for c in self.checks}

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        if not self.checks:
            return "nenhuma restrição checada"
        bad = self.violated
        if not bad:
            worst = min(self.checks, key=lambda c: c.margin)
            return (f"{len(self.checks)} restrição(ões) satisfeita(s); "
                    f"folga mínima {worst.slack:+.4g} em {worst.name!r}")
        parts = [f"{c.name!r} violada por {c.margin:+.4g}{'' if c.hard else ' (branda)'}" for c in bad]
        return f"{len(bad)}/{len(self.checks)} violada(s): " + "; ".join(parts)


class ConstraintViolation(RuntimeError):
    """A hard constraint was violated. Carries the :class:`ConstraintReport` that says which one and by how much."""

    def __init__(self, report: ConstraintReport, where: str = ""):
        self.report = report
        lugar = f" em {where}" if where else ""
        super().__init__(f"restrição dura violada{lugar}: {report.summary()}")


# --------------------------------------------------------------------------- the arithmetic
def _is_tensor(v: Any) -> bool:
    return type(v).__module__.split(".")[0] == "torch" and hasattr(v, "detach")


def _flatten(values: Iterable[Any]) -> list[Any]:
    """Every value, with mappings opened so a dict of tensors already makes the call tensor-valued."""
    out: list[Any] = []
    for v in values:
        if isinstance(v, Mapping):
            out.extend(v.values())
        else:
            out.append(v)
    return out


def _xp(*values: Any):
    """``(module, is_tensor)``: torch when any value is a tensor (so the margin is differentiable), else numpy."""
    if any(_is_tensor(v) for v in _flatten(values)):
        import torch
        return torch, True
    import numpy as np
    return np, False


def _scalar(v: Any) -> float:
    """A tensor, a 0-d array or a Python number as a float."""
    if _is_tensor(v):
        return float(v.detach().item())
    import numpy as np
    a = np.asarray(v)
    if a.shape == ():
        return float(a)
    raise ValueError(f"esperava um escalar, veio a forma {a.shape}")


def _worst(xp: Any, g: Any) -> Any:
    """The worst element of a margin: a field violates on its hottest cell, not on its mean."""
    return xp.max(g) if getattr(g, "shape", ()) != () else g


def _quantity(constraint: Any, values: Mapping[str, Any]) -> Any:
    """The number (or field) a bound constraint is about, from a mapping of quantity name -> value."""
    if constraint.quantity in values:
        return values[constraint.quantity]
    raise KeyError(f"restrição {constraint.name!r} é sobre {constraint.quantity!r}, que a resposta não carrega "
                   f"(ela tem {sorted(values)})")


def residual(constraint: Any, values: Mapping[str, Any]) -> Any:
    """The signed margin of ``constraint`` over ``values``: ``<= 0`` satisfied, ``> 0`` violated by that much.

    ``values`` maps quantity name -> number or field. A field (a numpy array or a torch tensor) gives an elementwise
    margin -- one per cell -- and :func:`check` takes the worst of them; with a tensor the result is a tensor and
    carries its graph, which is what lets the same object be a training penalty.
    """
    if constraint.kind == "expression":
        from pinneapple_data.expressions import residual as _expression_residual
        _module, tensor = _xp(*values.values())
        return _expression_residual(constraint.expression, values, ops="torch" if tensor else "numpy")
    if not isinstance(values, Mapping):
        raise TypeError("uma restrição é perguntada sobre um mapeamento quantidade -> valor; recebi "
                        f"{type(values).__name__} -- para um otimizador com vetor de parâmetros, passe "
                        "names=<nomes na ordem> a optimizer_constraints()")
    q = _quantity(constraint, values)
    xp, _tensor = _xp(q)
    limit = constraint.value
    if constraint.op == "<=":
        return xp.subtract(q, limit)
    if constraint.op == ">=":
        return xp.subtract(limit, q)
    return xp.abs(xp.subtract(q, limit))


def check(constraint: Any, values: Mapping[str, Any], *, tol: float = TOL) -> ConstraintCheck:
    """Ask one constraint about one answer, and record the answer (satisfied, margin, value, limit).

    A field-valued quantity is reduced to its worst element before the comparison: the report then says "the hottest
    cell is 7.5 degC above the limit", which is the statement a designer can act on.
    """
    xp, _tensor = _xp(*_flatten([values, constraint.value]))
    margin = _scalar(_worst(xp, residual(constraint, values)))
    value = None
    if constraint.kind == "bound":
        value = _scalar(_worst(xp, _quantity(constraint, values)))
    return ConstraintCheck(
        name=constraint.name, family=constraint.family or "unlabelled", satisfied=bool(margin <= tol),
        margin=margin, value=value,
        limit=None if constraint.value is None else float(constraint.value),
        op="" if constraint.kind == "expression" else constraint.op,
        hard=bool(getattr(constraint, "hard", True)),
        detail="" if constraint.family else "sem família declarada",
    )


def check_all(constraints: Iterable[Any], values: Mapping[str, Any], *, tol: float = TOL) -> ConstraintReport:
    """Every constraint asked about one answer. This is also how a probability is measured: :func:`check_all` per
    ensemble member, then the fraction of reports that violate each constraint."""
    return ConstraintReport([check(c, values, tol=tol) for c in constraints])


# --------------------------------------------------------------------------- point 1: training
def _values_for(constraint: Any, out: Any, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """What one constraint is asked about, read off the model's output and the batch."""
    vals: dict[str, Any] = dict(extra or {})
    if isinstance(out, Mapping):
        vals.update({k: v for k, v in out.items() if k != "y"})
    elif constraint.kind == "bound" and constraint.quantity not in vals:
        vals[constraint.quantity] = out          # a bare tensor is the field the constraint names
    return vals


def penalty(constraints: Sequence[Any], out: Any, values: Mapping[str, Any] | None = None,
            *, reduction: str = "mean"):
    """The constraints as one differentiable loss term: ``sum(weight * mean(relu(g) ** 2))``.

    ``out`` is what the model produced -- a tensor (the predicted field is then the quantity every ``kind="bound"``
    constraint is about) or a mapping of quantity name -> value. ``values`` carries anything else the constraints
    name, typically the batch (the design parameters of an expression constraint).

    A satisfied constraint contributes exactly zero, and the ``relu`` keeps a non-zero gradient at the boundary, so a
    network that already respects the limit is not pushed away from it. The square makes the term quiet for a small
    violation and dominant for a large one, which is what a hard limit deserves.

    A field-valued quantity is penalised **cell by cell**, not reduced to its worst cell: :func:`check` reduces because
    a report has to answer "is this satisfied", decided by the hottest cell, while a penalty has to push back
    everywhere the field is over the line -- reducing to one cell would spend a whole step's gradient budget on a
    single point.

    Returns a torch scalar. With no constraints it returns ``torch.zeros(())``, so a training loop can add it
    unconditionally.
    """
    import torch

    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    total = torch.zeros(())
    for c in constraints:
        g = residual(c, _values_for(c, out, values))
        g = g if _is_tensor(g) else torch.as_tensor(g)
        term = torch.clamp(g, min=0.0) ** 2
        total = total + float(getattr(c, "weight", 1.0)) * (term.mean() if reduction == "mean" else term.sum())
    return total


# --------------------------------------------------------------------------- point 2: the optimizer
def optimizer_constraints(constraints: Sequence[Any], names: Sequence[str] | None = None):
    """The same constraints as ``pinneapple_core.optim`` objects, for ``PhysicsOptimizer(constraints=...)``.

    ``names`` names the optimizer's parameter vector, in order, so a constraint written as ``"mass_kg <= 50"`` or
    ``"chord - 2 * thickness >= 0.01"`` finds its variables. Without ``names`` the optimizer's own
    ``argument="dict"`` convention is assumed. A bound constraint with ``op="=="`` becomes an equality constraint;
    everything else is an inequality ``g <= 0``, which is the convention both sides already use.
    """
    from pinneapple_core.optim import eq, ineq

    def as_fn(constraint: Any, order: list[str] | None):
        """The margin of ``constraint`` as a function of the optimizer's own argument."""
        if order is None:
            def fn(named: Mapping[str, Any]) -> Any:
                return residual(constraint, named)
        else:
            def fn(vector: Any) -> Any:
                return residual(constraint, dict(zip(order, vector, strict=False)))
        return fn

    order = None if names is None else list(names)
    out = []
    for c in constraints:
        fn = as_fn(c, order)
        out.append(eq(fn, c.name) if (c.kind == "bound" and c.op == "==") else ineq(fn, c.name))
    return out


# --------------------------------------------------------------------------- point 3: the deployment guard
class ConstraintGuard:
    """The same constraints as a guard around a prediction, so a deployment refuses instead of serving.

    A constraint is cheap enough to evaluate per prediction, and that is the point: the guard asks the constraint
    object itself -- not a copy, not a re-implementation in the serving code -- and raises
    :class:`ConstraintViolation` with the report when a **hard** constraint is violated. A soft constraint is reported
    and let through: the answer is still off, and stopping a production call over a number that was declared
    non-fatal would be worse than the number.

    ``on_violation="raise"`` (default) refuses; ``"report"`` evaluates, records ``last_report`` and returns the
    prediction anyway (a shadow mode, for measuring a limit in production before enforcing it); a callable is called
    with the report and then the violation is raised -- a metric, a log line or an alert.

    Usage
    -----
    >>> guard = ConstraintGuard([T], where="serve /predict")
    >>> predict = guard.wrap(model_predict)          # or: @guard
    >>> predict(x)                                   # raises ConstraintViolation on a hot prediction
    """

    def __init__(self, constraints: Sequence[Any], *, on_violation: Any = "raise", where: str = "",
                 values: Mapping[str, Any] | None = None, tol: float = TOL):
        if on_violation not in ("raise", "report") and not callable(on_violation):
            raise ValueError("on_violation must be 'raise', 'report' or a callable")
        self.constraints = list(constraints)
        if not self.constraints:
            raise ValueError("uma guarda sem restrição não guarda nada; passe ao menos uma")
        self.on_violation = on_violation
        self.where = where
        self.values = dict(values or {})
        self.tol = tol
        self.last_report: ConstraintReport | None = None

    def check(self, out: Any, values: Mapping[str, Any] | None = None) -> ConstraintReport:
        """Ask every constraint about ``out``, record the report on ``last_report``, and act on a violation."""
        report = ConstraintReport()
        for c in self.constraints:
            report.checks.append(check(c, _values_for(c, out, {**self.values, **(values or {})}), tol=self.tol))
        self.last_report = report
        if report.hard_violated:
            if callable(self.on_violation):
                self.on_violation(report)
            if self.on_violation != "report":
                raise ConstraintViolation(report, where=self.where)
        return report

    def wrap(self, predict: Any) -> Any:
        """``predict`` with the guard in front: same call, same return value, or a refusal."""
        def guarded(*args: Any, **kwargs: Any) -> Any:
            values = kwargs.pop("values", None)
            out = predict(*args, **kwargs)
            self.check(out, values)
            return out
        guarded.__name__ = getattr(predict, "__name__", "guarded")
        guarded.__doc__ = getattr(predict, "__doc__", None)
        guarded.__wrapped__ = predict
        return guarded

    def __call__(self, predict: Any) -> Any:
        return self.wrap(predict)
