"""Reconciling the three problem specifications: decision D1 of the strategy (issue #150).

PINNeAPPle grew three descriptions of "a problem", each correct where it is used and none of them a superset:

============================  ===========================================================================
``pinneapple_problemdesign``  the **elicited** description: what the user *said*, still free text -- a goal,
``.schema.ProblemSpec``       equations as strings, geometry as a sentence, constraints and risks. Nothing in it
                              is compilable, on purpose: it is what a conversation produces before anybody decides
                              which equations are meant.
``pinneapple_physics``        the **PDE** description: coordinates, fields, the ``PDETermSpec`` the compiler and the
``.pde_environment.spec``     solvers switch on, conditions with selectors and value functions, bounds and a
``.ProblemSpec``              reference solver. This is the one that runs.
``pinneapple_data``           the **sampling** description: one condition of a UPD shard -- a ``type`` (initial /
``.upd_types.ConditionSpec``  boundary / interior / slice), the equation string kept for coordination, ``n`` points
                              and free-form options. It says how many points, not what the physics is.
============================  ===========================================================================

Conversion has to exist because a project starts elicited, is solved as a PDE and is stored as samples; and the
conversion has to be **explicit** because two of the three are lossy in one direction each, and a converter that
hides that is how a wrong problem reaches a solver.

The pivot is :class:`pinneapple_physics.physical_problem.PhysicalProblem`, which already carries tested adapters to
both ``ProblemSpec`` flavours (``from_problem_design`` / ``to_problem_design``, ``from_pde_spec`` / ``to_pde_spec``).
The functions here are the direct, named conversions on top of those, with the loss written down in
:data:`LOSSES` rather than discovered later, plus the condition-level bridge between the sampling and the PDE
description, which did not exist at all.

Nothing here adds a class: D1 says the three specifications stay as they are, and this module is functions over
them. The record of the decision, with the field-by-field table, is ``docs/strategy/D1_problem_specs.md``.

>>> import pinneapple as pp
>>> from pinneapple_physics.pde_environment.reconcile import pde_to_design, design_to_pde
>>> spec = pp.PhysicalProblem.from_preset("burgers_1d").to_pde_spec()
>>> design = pde_to_design(spec)                       # what a report or the assistant shows
>>> back = design_to_pde(design)                       # and back to something a solver can take
>>> back.fields, back.coords, back.domain_bounds
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .conditions import ConditionSpec
from .spec import ProblemSpec

__all__ = ["LOSSES", "SHARED_FIELDS", "pde_to_design", "design_to_pde", "upd_condition_to_pde",
           "pde_condition_to_upd", "ConditionBridge"]

#: The fields the three specifications share, as data, so the decision record and the tests read the same table.
#: ``(elicited attr, PDE attr, sampling attr)`` -- ``None`` where that specification has no equivalent.
SHARED_FIELDS: Dict[str, tuple[Optional[str], Optional[str], Optional[str]]] = {
    "name": ("title", "name", "name"),
    "fields": ("outputs", "fields", None),
    "coordinates": ("inputs", "coords", None),
    "bounds": ("geometry.domain (prose)", "domain_bounds", None),
    "equations": ("physics.governing_equations", "pde.kind", "equation"),
    "boundary": ("physics.boundary_conditions", "conditions[dirichlet|neumann|robin]", "type=boundary"),
    "initial": ("physics.initial_conditions", "conditions[initial]", "type=initial"),
    "parameters_known": ("physics.parameters_known", "parameters[role=fixed]", None),
    "parameters_unknown": ("physics.parameters_unknown", "parameters[role=unknown]", None),
    "units": ("physics.units", "units", None),
    "points": (None, "sample_defaults", "n"),
}

#: What each conversion drops, and why that is the right thing to drop. A converter is allowed to be lossy; a
#: converter that is lossy without saying so is how a problem that was never described reaches a solver.
LOSSES: Dict[str, str] = {
    "design_to_pde":
        "the elicited equations are prose: ``pde.kind`` has to be chosen (or the problem keeps ``pde=None`` and "
        "cannot run), and geometry described in a sentence does not become a mesh. The free-text remainder is kept "
        "in ``meta['elicited']`` so nothing is silently thrown away.",
    "pde_to_design":
        "the PDE term, the selectors and the value functions are not expressible as elicited prose: the equation "
        "becomes ``pde.kind`` as a string, conditions become their names, and the numeric detail stays on the PDE "
        "side. The round trip back recovers the name, the fields, the coordinates and the units; the bounds are "
        "prose on the elicited side (``geometry.domain``), so they come back only when :func:`design_to_pde` is "
        "given them, and the closures never.",
    "upd_condition_to_pde":
        "a sampling condition knows a ``type`` and an equation string, not a selector: the PDE condition gets "
        "``selector_type`` derived from the type and keeps the string in ``meta``; the value function is only "
        "attached when ``options['value']`` gives a constant.",
    "pde_condition_to_upd":
        "a condition with a Python value function has no equation string to store: the sampling condition keeps the "
        "name, the type and ``n``, and records ``value_fn`` as its own name (the sampling side never evaluates it).",
}

_SAMPLING_TYPE = {"initial": "initial", "dirichlet": "boundary", "neumann": "boundary", "robin": "boundary",
                  "interface": "boundary", "data": "interior"}
_PDE_KIND = {"initial": "initial", "boundary": "dirichlet", "interior": "data", "slice": "dirichlet"}


# ------------------------------------------------------------------ problem level
def pde_to_design(spec: ProblemSpec):
    """The elicited description of a PDE ``ProblemSpec`` (for reports, the assistant and the plan text).

    Goes through ``PhysicalProblem``, which owns the tested mapping and the task vocabulary; see
    :data:`LOSSES` for what an elicited description cannot hold.
    """
    from ..physical_problem import PhysicalProblem

    return PhysicalProblem.from_pde_spec(spec).to_problem_design()


def design_to_pde(design: Any, *, pde_kind: str, coords: Optional[Sequence[str]] = None,
                  fields: Optional[Sequence[str]] = None,
                  domain_bounds: Optional[Mapping[str, Sequence[float]]] = None) -> ProblemSpec:
    """A PDE ``ProblemSpec`` from an elicited one.

    ``pde_kind`` is required and has no default on purpose: the elicited description says "heat conduction in a
    fin" in prose, and which ``pde.kind`` the solvers switch on is a decision somebody makes, not something the
    converter may guess. ``coords``, ``fields`` and ``domain_bounds`` fill what the text left implicit; whatever
    the elicited side carries and the PDE side cannot hold (goal, risks, data plan, assumptions) goes to
    ``spec.meta['elicited']`` rather than being dropped.
    """
    from dataclasses import replace

    from ..physical_problem import PhysicalProblem

    problem = PhysicalProblem.from_problem_design(design)
    pde = replace(problem.pde or _empty_pde(problem), kind=pde_kind)
    # the elicited side calls the coordinates "inputs"; when the caller names them, those win
    elicited_coords = tuple(coords or problem.coords or getattr(design, "inputs", ()) or ())
    problem = replace(problem, pde=pde, coords=elicited_coords,
                      fields=tuple(fields or problem.fields))
    spec = problem.to_pde_spec()
    if domain_bounds:
        spec = replace(spec, domain_bounds={k: tuple(v) for k, v in domain_bounds.items()})
    meta = dict(spec.meta)
    meta["elicited"] = problem.intent
    return replace(spec, meta=meta)


def _empty_pde(problem: Any) -> Any:
    """A ``PDETermSpec`` with no kind, to be given one by :func:`design_to_pde`."""
    from .spec import PDETermSpec

    return PDETermSpec(kind="", fields=tuple(problem.fields), coords=tuple(problem.coords))


@dataclass
class ConditionBridge:
    """A converted condition, the data the target cannot hold, and what the conversion could not do."""
    condition: ConditionSpec
    carried: Dict[str, Any]
    lost: tuple[str, ...] = ()


def upd_condition_to_pde(cond: Any, *, fields: Sequence[str], value: Optional[float] = None) -> ConditionBridge:
    """A PDE ``ConditionSpec`` from a sampling ``pinneapple_data.upd_types.ConditionSpec``.

    ``fields`` are the fields the condition applies to (the sampling side has no idea, it stores climate variables).
    The equation string and the point count travel in ``carried``: a PDE condition is written as a selector and a
    value function, and inventing one from a string would be a guess. ``value`` attaches a constant value function,
    which is the only value the sampling side can express.
    """
    upd_type = str(getattr(cond, "type", "") or "interior")
    kind = _PDE_KIND.get(upd_type.lower(), "data")
    condition = ConditionSpec(name=str(getattr(cond, "name", "") or kind), kind=kind, fields=tuple(fields),
                              selector_type=f"upd:{upd_type}",
                              value_fn=(None if value is None else (lambda X, _v=value: _v)))
    lost: list[str] = []
    if value is None:
        lost.append("value_fn: the sampling condition carries an equation string, not a value -- pass value=")
    carried = {"equation": str(getattr(cond, "equation", "")), "n": int(getattr(cond, "n", 0) or 0),
               "type": upd_type, "options": dict(getattr(cond, "options", {}) or {})}
    return ConditionBridge(condition=condition, carried=carried, lost=tuple(lost))


def pde_condition_to_upd(cond: ConditionSpec, *, carried: Optional[Mapping[str, Any]] = None,
                         n: Optional[int] = None) -> Any:
    """A sampling ``ConditionSpec`` from a PDE one: the shared fields, plus what ``carried`` hands back.

    A PDE condition with a Python value function has no equation string of its own; passing the ``carried`` dict
    from :func:`upd_condition_to_pde` makes the round trip lossless for everything the sampling side can hold.
    """
    from pinneapple_data.upd_types import ConditionSpec as SamplingCondition

    carried = dict(carried or {})
    points = n if n is not None else int(carried.get("n") or 1024)
    kind = _SAMPLING_TYPE.get(cond.kind, "interior")
    return SamplingCondition(name=cond.name, type=carried.get("type") or kind,
                             equation=str(carried.get("equation") or cond.kind), n=points,
                             options=dict(carried.get("options") or {}))
