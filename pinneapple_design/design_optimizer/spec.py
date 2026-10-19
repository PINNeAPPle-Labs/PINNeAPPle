"""Design problems from plain JSON specs (what an app or an API call sends).

A spec names where the numbers come from and what to optimise::

    {"source": "preset", "preset": "heatsink", "fidelity": "surrogate",
     "operating": {"power_w": 60}, "limits": {"t_source_max_c": 85}, "cost": {"material_per_kg": 6},
     "variables": [{"name": "base_width_mm", "lower": 40, "upper": 120}],     # optional bound changes
     "objectives": [{"name": "mass_g"}, {"name": "cost"}]}

    {"source": "zoo", "model": "airfoil_cst_rans_mlp", "fixed": {"alpha_deg": 4},
     "objectives": [{"name": "Cd"}], "constraints": [{"name": "Cl", "lower": 0.6}]}

    {"source": "table", "inputs": [...], "outputs": [...], "objectives": [...], "constraints": [...]}  # + a table

:func:`problem_from_spec` returns the :class:`~.study.DesignProblem` and, for a table, the fitted surrogate with its
held-out errors; :func:`reference_for_spec` the high-fidelity evaluator to verify candidates with (None when the
library has none for that source).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from typing import Any

from .study import Constraint, DesignProblem, DesignSpace, Evaluator, Objective, Variable

__all__ = ["objectives_from", "constraints_from", "problem_from_spec", "reference_for_spec", "zoo_problem",
           "SpecError"]


class SpecError(ValueError):
    """A spec that cannot become a problem (readable message for the caller)."""


def objectives_from(items: Sequence[Mapping[str, Any]]) -> list[Objective]:
    out = []
    for o in items or []:
        out.append(Objective(str(o["name"]), str(o.get("sense", "min")), str(o.get("unit", "")),
                             str(o.get("label", ""))))
    if not out:
        raise SpecError("choose at least one objective")
    return out


def constraints_from(items: Sequence[Mapping[str, Any]]) -> list[Constraint]:
    out = []
    for c in items or []:
        lo, hi = c.get("lower"), c.get("upper")
        lo = None if lo in (None, "") else float(lo)
        hi = None if hi in (None, "") else float(hi)
        if lo is None and hi is None:
            continue
        out.append(Constraint(str(c["name"]), upper=hi, lower=lo, unit=str(c.get("unit", "")),
                              kind=str(c.get("kind", "physical")), label=str(c.get("label", ""))))
    return out


def _with_bounds(space: DesignSpace, changes: Sequence[Mapping[str, Any]] | None) -> DesignSpace:
    if not changes:
        return space
    by = {c["name"]: c for c in changes}
    unknown = sorted(set(by) - set(space.names))
    if unknown:
        raise SpecError(f"unknown variables {unknown}")
    vs = []
    for v in space.variables:
        c = by.get(v.name)
        if c is None:
            vs.append(v)
            continue
        lo = float(c.get("lower", v.lower))
        hi = float(c.get("upper", v.upper))
        if not hi > lo:
            raise SpecError(f"{v.name}: upper bound must exceed lower bound")
        vs.append(replace(v, lower=lo, upper=hi))
    return DesignSpace(vs)


def _dataclass_from(cls, d: Mapping[str, Any] | None):
    names = {f.name for f in fields(cls)}
    d = dict(d or {})
    unknown = sorted(set(d) - names)
    if unknown:
        raise SpecError(f"{cls.__name__}: unknown fields {unknown}")
    return cls(**d)


def _heatsink(spec: Mapping[str, Any]) -> DesignProblem:
    from . import presets
    op = _dataclass_from(presets.HeatsinkOperating, spec.get("operating"))
    lim = presets._process_limits()
    if spec.get("limits"):
        _dataclass_from(presets.HeatsinkLimits, spec["limits"])           # names checked
        lim = replace(lim, **{k: float(v) for k, v in spec["limits"].items()})
    cost = _dataclass_from(presets.CostModel, spec.get("cost"))
    objs = [o["name"] for o in spec.get("objectives") or [{"name": "mass_g"}, {"name": "cost"}]]
    bad = [o for o in objs if o not in presets._HEATSINK_OBJECTIVES]
    if bad:
        raise SpecError(f"heat-sink objectives are {sorted(presets._HEATSINK_OBJECTIVES)}; got {bad}")
    fid = spec.get("fidelity", "surrogate")
    if fid not in presets.HEATSINK_FIDELITIES:
        raise SpecError(f"fidelity must be one of {presets.HEATSINK_FIDELITIES}")
    space = _with_bounds(DesignSpace(presets.HEATSINK_VARIABLES), spec.get("variables"))
    return presets.heatsink_problem(fid, op, lim, cost, variables=space.variables, objectives=tuple(objs))


def _airfoil(spec: Mapping[str, Any]) -> DesignProblem:
    from . import presets
    lim = dict(spec.get("limits") or {})
    unknown = sorted(set(lim) - {"cl_min", "cm_min", "aft_min"})
    if unknown:
        raise SpecError(f"airfoil limits are cl_min, cm_min, aft_min; got {unknown}")
    p = presets.airfoil_problem(**{k: float(v) for k, v in lim.items()})
    if spec.get("variables"):
        p = DesignProblem(p.name, _with_bounds(p.space, spec["variables"]), p.objectives, p.constraints, p.evaluator,
                          p.description, p.notes)
    return p


def zoo_problem(model: str, objectives: Sequence[Mapping[str, Any]], constraints: Sequence[Mapping[str, Any]] = (),
                *, variables: Sequence[Mapping[str, Any]] | None = None,
                fixed: Mapping[str, float] | None = None) -> DesignProblem:
    """A problem on a ``pp.zoo`` model: every model input that is not ``fixed`` is a design variable, bounded by the
    model's validity range (or by ``variables``)."""
    import pinneapple as pp

    from .evaluators import ZooEvaluator
    try:
        m = pp.zoo.load(model)
    except Exception as e:  # noqa: BLE001 - unknown name, missing weights: a readable error for the caller
        raise SpecError(f"cannot load zoo model {model!r}: {e}") from e
    fixed = {k: float(v) for k, v in (fixed or {}).items()}
    vr = m.card.validity_range
    given = {v["name"]: v for v in variables or []}
    vs = []
    for k in m.input_names:
        if k in fixed:
            continue
        g = given.get(k, {})
        rng = vr.get(k) if isinstance(vr.get(k), (list, tuple)) else None
        lo, hi = g.get("lower", rng[0] if rng else None), g.get("upper", rng[1] if rng else None)
        if lo is None or hi is None:
            raise SpecError(f"input {k!r} of {model!r} has no validity range: give its bounds or fix it")
        vs.append(Variable(k, float(lo), float(hi), str(g.get("kind", "continuous"))))
    if not vs:
        raise SpecError("every input is fixed: nothing to optimise")
    space = DesignSpace(vs)
    ev = ZooEvaluator(model, space, fixed=fixed)
    objs, cons = objectives_from(objectives), constraints_from(constraints)
    return DesignProblem(f"zoo:{model}", space, objs, cons, ev,
                         description=f"Optimise the outputs of the zoo model {model}",
                         notes=[f"{model}: {m.card.task}", "designs outside the model's validity range are infeasible"])


def problem_from_spec(spec: Mapping[str, Any], table=None) -> tuple[DesignProblem, Any]:
    """``(problem, fit)``: ``fit`` is the :class:`~.report.SurrogateFit` of a table source, else None."""
    src = spec.get("source", "preset")
    try:
        if src == "preset":
            name = spec.get("preset", "heatsink")
            if name == "heatsink":
                return _heatsink(spec), None
            if name == "airfoil":
                return _airfoil(spec), None
            raise SpecError(f"unknown preset {name!r} (heatsink, airfoil)")
        if src == "zoo":
            return zoo_problem(str(spec["model"]), spec.get("objectives") or [], spec.get("constraints") or [],
                               variables=spec.get("variables"), fixed=spec.get("fixed")), None
        if src == "table":
            from .report import surrogate_from_table
            if table is None:
                raise SpecError("a table source needs the table of evaluated designs")
            fit = surrogate_from_table(table, spec["inputs"], spec["outputs"], integer=spec.get("integer") or (),
                                       test_fraction=float(spec.get("test_fraction", 0.2)),
                                       seed=int(spec.get("seed", 0)))
            space = _with_bounds(fit.space, spec.get("variables"))
            outs = set(spec["outputs"])
            objs, cons = objectives_from(spec.get("objectives")), constraints_from(spec.get("constraints"))
            missing = sorted({o.name for o in objs} - outs | ({c.name for c in cons} - outs))
            if missing:
                raise SpecError(f"objectives/constraints {missing} are not among the outputs {sorted(outs)}")
            p = DesignProblem("table", space, objs, cons, fit.gp,
                              description="Optimise on a Gaussian-process surrogate of your evaluated designs",
                              notes=["surrogate trained on your table; variables are bounded by the data range",
                                     "candidates are only surrogate predictions until you verify them with your "
                                     "solver"])
            return p, fit
    except KeyError as e:
        raise SpecError(f"missing field {e}") from e
    except (TypeError, ValueError) as e:
        if isinstance(e, SpecError):
            raise
        raise SpecError(str(e)) from e
    raise SpecError(f"unknown source {src!r} (preset, zoo, table)")


def reference_for_spec(spec: Mapping[str, Any]) -> Evaluator | None:
    """The library's high-fidelity reference for a spec: the FVM engine for the heat sink, None otherwise."""
    if spec.get("source", "preset") == "preset" and spec.get("preset", "heatsink") == "heatsink":
        from . import presets
        op = _dataclass_from(presets.HeatsinkOperating, spec.get("operating"))
        cost = _dataclass_from(presets.CostModel, spec.get("cost"))
        return presets.HeatsinkEvaluator("fvm", op, cost)
    return None
