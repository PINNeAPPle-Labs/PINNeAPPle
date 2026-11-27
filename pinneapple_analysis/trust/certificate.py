"""pinneapple_analysis.trust.certificate — physics verification as a service: check any model's output against the
equations and return a certificate (#341).

The question this module answers is the one a consumer of somebody else's surrogate has: *does this field satisfy
the physics it claims, and where does it stop doing so?* The caller brings a prediction (a field on a grid, a point
cloud, a file) and the description of the problem; the answer is a :class:`Certificate`, a JSON document that says,
per check, the number, the limit it was judged against, where in the domain the worst point is and where that number
came from.

What it is built on — nothing here is a second engine:

===========================  ==========================================================================
the residual of each law     ``pinneapple_check.physics.equations`` -- the equation set of the linter, on the
                             derivative providers of ``pinneapple_check.numerics`` (``GridD`` / ``PointD``):
                             incompressible Navier-Stokes (continuity + momentum), heat, advection-diffusion,
                             shallow water and elasticity, plus ``conservation``, ``darcy`` and ``maxwell``
reading the prediction       ``pinneapple_check.fields.load_result`` -- OpenFOAM case, ``.npz``/``.h5`` grid,
                             VTU/VTK/CSV/XDMF point cloud, QoI table
units                        ``pinneapple_data.physical_units`` -- the same dimension and factor comparison the
                             linter makes (PC501/PC502)
the in-domain level          ``pinneapple_physics.ensemble.residual_baseline`` semantics: the median residual of
                             the *same model* on cases from its own domain
the domain gate              ``pinneapple_analysis.trust.trust_gate.TrustGate.fit_from_bounds`` and
                             :func:`pinneapple_analysis.trust.adapters.domain_from_gate`
the record                   :class:`pinneapple_analysis.trust.record.Trust` (schema ``pinneapple.trust/1``):
                             residuals per law with their in-domain reference, in-domain/domain_score, horizon and
                             provenance. The certificate embeds that record; it does not replace it
===========================  ==========================================================================

The certificate's own decision is :meth:`Certificate.badge`, which is the rule of :meth:`Trust.badge` written over
the checks the certificate ran: ``no_skill`` when a measured check proves the prediction carries no information (its
case is outside the declared domain), ``fading`` when a measured check fails or a check could not be measured, and
``good`` when every check was measured and passed. There is no weighted score, and a residual with no in-domain level
and no declared tolerance is ``unmeasured`` -- never a pass.

>>> import numpy as np
>>> from pinneapple_analysis.trust.certificate import Problem, verify
>>> x, t = np.linspace(0, 2 * np.pi, 65)[:-1], np.linspace(0, 1, 41)
>>> X, T = np.meshgrid(x, t, indexing="ij")
>>> C = np.exp(-0.05 * T) * np.sin(X - T)                    # exact solution of dC/dt + dC/dx = 0.05 d2C/dx2
>>> cert = verify({"C": C}, Problem(physics="advection-diffusion", axes={"x": x, "t": t},
...                                 parameters={"c": 1.0, "alpha": 0.05}, tolerance={"advection-diffusion": 0.05},
...                                 units={"C": "1", "x": "m", "t": "s"}))
>>> cert.badge()
'good'
>>> cert.trust.residual("advection-diffusion").value.value < 0.01
True

What it does not do, in this round: an OpenFOAM case folder (finite-volume residuals, ``pinneapple_check.fv``) is
read but the residuals are evaluated on the cell values as a point cloud, not with the conservative face sums; and
the HTTP endpoint with the upload page is not here yet (the library call is the whole surface this round).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .field_card import DEFAULT_TOLERANCES, _evaluate, _face, _points
from .record import (
    GATE_FAIL,
    GATE_NO_SKILL,
    GATE_PASS,
    GATE_UNMEASURED,
    Gate,
    Provenance,
    Residual,
    Trust,
    TrustValue,
    fingerprint_array,
    fingerprint_text,
    library_version,
    utc_now,
)

__all__ = ["SCHEMA", "Problem", "Check", "Certificate", "Calibration", "CaseResult", "verify", "read_fields",
           "calibrate", "LAWS"]

#: What the JSON says it is. A reader compares the major number and refuses what it does not know.
SCHEMA = "pinneapple.certificate/1"

#: The laws a certificate can carry, in the vocabulary of ``pinneapple_check.physics`` plus the ``conservation``
#: law of ``pinneapple_check/checks/results.py``. The first five are the ones issue #341 names.
LAWS = ("continuity", "momentum", "heat", "advection-diffusion", "equilibrium", "conservation of", "maxwell",
        "pde")

#: How a check's status maps to the verdict printed on the certificate. It is the badge, written out: no threshold
#: is added here, and the two vocabularies are the same three states.
_VERDICT = {"good": "certified", "fading": "qualified", "no_skill": "rejected"}


# -------------------------------------------------------------------------------------------------- the problem
@dataclass(frozen=True)
class Problem:
    """What the caller declares about the case, so the checks have something to judge the field against.

    ``physics`` is one of ``pinneapple_check.config.PHYSICS`` (``incompressible-flow``, ``heat``,
    ``advection-diffusion``, ``shallow-water``, ``elasticity``, ``conservation``, ``darcy``, ``maxwell``).
    ``parameters`` are the numbers of the equation, as SI values or as ``"1e-5 m2/s"`` strings (parsed by
    ``pinneapple_check.physics.parse_parameters``, which also refuses a string with the wrong dimension).

    ``axes`` is only needed when the fields arrive as arrays (a file carries its own coordinates). ``units`` is the
    unit of each field as the caller declares it; a field used by the equation with no unit on either side is a
    check that could not run. ``tolerance`` is an absolute limit per equation (``{"continuity": 0.01}``) used only
    when ``reference`` does not give the in-domain level of that law.

    ``reference`` is the honest scale of the residual: the level the same model has inside its own domain
    (``pinneapple_physics.ensemble.residual_baseline``), per law. ``domain`` is the training envelope, as
    ``{name: (lo, hi)}`` where ``name`` is a grid axis or a case parameter: it is what makes the certificate say
    ``no_skill`` when the case left the domain instead of merely "the residual is high".
    """

    physics: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    axes: Mapping[str, Sequence[float]] = field(default_factory=dict)
    boundary: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    positivity: tuple[str, ...] = ()
    flux: Mapping[str, str] = field(default_factory=dict)
    domain: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    reference: Mapping[str, float] = field(default_factory=dict)
    reference_source: str = ""
    tolerance: Mapping[str, float] = field(default_factory=dict)
    model: str = ""
    notes: tuple[str, ...] = ()

    def limit(self, law: str) -> tuple[float | None, str]:
        """The limit this law is judged against and where it came from, in the order of #345: the model's own
        in-domain level first, the caller's declared tolerance second, nothing third."""
        if law in self.reference:
            return float(self.reference[law]), self.reference_source or "the in-domain residual level declared by " \
                                                                         "the caller"
        key = next((k for k in (law, self.physics, "residual") if k in self.tolerance), None)
        if key is not None:
            return float(self.tolerance[key]), f"the tolerance declared by the caller for {key!r}"
        return None, ""

    def to_dict(self) -> dict[str, Any]:
        return {"physics": self.physics, "parameters": _plain(self.parameters), "units": dict(self.units),
                "axes": {k: [float(v) for v in np.asarray(a).ravel()[:3]] + (["..."] if len(a) > 3 else [])
                         for k, a in self.axes.items()},
                "boundary": {f: dict(c) for f, c in self.boundary.items()}, "positivity": list(self.positivity),
                "flux": dict(self.flux), "domain": {k: [float(lo), float(hi)] for k, (lo, hi) in self.domain.items()},
                "reference": {k: float(v) for k, v in self.reference.items()},
                "reference_source": self.reference_source, "tolerance": dict(self.tolerance), "model": self.model,
                "notes": list(self.notes)}


def _plain(value: Any) -> Any:
    """JSON-safe copy of a declared value (numpy scalars and arrays included)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# -------------------------------------------------------------------------------------------------- the checks
@dataclass(frozen=True)
class Check:
    """One check the certificate ran: what it decided, the number behind the decision, the limit it was judged
    against, where in the domain the worst point is, and which rule of ``pinneapple_check.rules`` it corresponds to.

    ``status`` is one of the four gate states of :mod:`pinneapple_analysis.trust.record`, so a certificate's checks
    and a :class:`Trust`'s gates are read with the same words.
    """

    name: str
    status: str
    value: float | None = None
    threshold: float | None = None
    where: str = ""
    source: str = ""
    rule: str = ""
    law: str = ""
    detail: str = ""
    direction: str = "le"          # "le": below the limit passes; "ge": above it passes (a fraction that must be 1)

    @property
    def passed(self) -> bool:
        return self.status == GATE_PASS

    def gate(self) -> Gate:
        """The check as a gate of the trust record: the same name and status, the number and the limit in the
        sentence that decided it."""
        number = "" if self.value is None else f"{self.value:.6g}"
        good, bad = ("<=", ">") if self.direction == "le" else (">=", "<")
        limit = "" if self.threshold is None else f" {good if self.status == GATE_PASS else bad} " \
                                                  f"{self.threshold:.6g}"
        at = f"; worst at {self.where}" if self.where and self.status != GATE_PASS else ""
        why = f" -- {self.detail}" if self.detail else ""
        src = f" ({self.source})" if self.source else ""
        return Gate(self.name, self.status, f"{number}{limit}{at}{src}{why}".lstrip(" ") or self.status)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "value": self.value, "threshold": self.threshold,
                "where": self.where, "source": self.source, "rule": self.rule, "law": self.law,
                "detail": self.detail, "direction": self.direction}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Check:
        return cls(document["name"], document["status"], document.get("value"), document.get("threshold"),
                   document.get("where", ""), document.get("source", ""), document.get("rule", ""),
                   document.get("law", ""), document.get("detail", ""))


@dataclass(frozen=True)
class Certificate:
    """The answer: a :class:`Trust` record with the residuals and the domain, plus the checks with their provenance.

    ``verdict()`` is the badge written out (``certified`` / ``qualified`` / ``rejected``), ``gates()`` is every
    check as a :class:`Gate` -- the certificate's own and the ones the trust record carries that the certificate did
    not measure (horizon, readings, calibration) -- and ``failures()`` is what a caller who only wants to know where
    the physics breaks reads. ``to_dict``/``to_json`` is the document, ``from_json`` reads it back.
    """

    trust: Trust
    checks: tuple[Check, ...]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    problem: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    notes: tuple[str, ...] = ()
    trust_gates: bool = False

    # -- decisions -------------------------------------------------------------------------------------------
    def gates(self) -> tuple[Gate, ...]:
        """Every check, as gates: the certificate's, plus -- when the certificate was built over a
        :class:`Trust` the caller supplied -- the trust record's sections it did not measure (horizon, the readings,
        the calibration).

        A certificate of a bare field judges the field, so the empty sections of a :meth:`Trust.empty` record are
        not gates of it: "no adapter was given for the horizon" is not a finding about the prediction. When the
        caller passes ``trust=pp.trust(ensemble, query)``, those gates do exist and are included.

        A section measured by both is listed once, under the certificate's name: the residual of a law is
        ``residual:<law>`` in both vocabularies, and a domain check is ``domain`` in both.
        """
        out = [c.gate() for c in self.checks]
        mine = {g.name for g in out}
        for g in (self.trust.gates() if self.trust_gates else ()):
            if g.name not in mine:
                out.append(g)
        return tuple(out)

    def badge(self) -> str:
        """``good`` / ``fading`` / ``no_skill``: the rule of :meth:`Trust.badge`, over :meth:`gates`."""
        from .record import FADING, GOOD, NO_SKILL

        gates = self.gates()
        if any(g.status == GATE_NO_SKILL for g in gates):
            return NO_SKILL
        if any(g.status in (GATE_FAIL, GATE_UNMEASURED) for g in gates):
            return FADING
        return GOOD

    def verdict(self) -> str:
        """The badge written out on the certificate: certified / qualified / rejected."""
        return _VERDICT[self.badge()]

    def certified(self) -> bool:
        return self.badge() == "good"

    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status in (GATE_FAIL, GATE_NO_SKILL))

    def unmeasured(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == GATE_UNMEASURED)

    def separating(self) -> str:
        """Which check decided the verdict, when one did: the sentence a consumer reads first."""
        for g in self.gates():
            if g.status == GATE_NO_SKILL:
                return f"no_skill: {g.name} -- {g.evidence}"
        bad = [g for g in self.gates() if g.status == GATE_FAIL]
        if bad:
            return "failing: " + "; ".join(f"{g.name} -- {g.evidence}" for g in bad)
        missing = [g for g in self.gates() if g.status == GATE_UNMEASURED]
        if missing:
            return "not measured: " + "; ".join(f"{g.name} -- {g.evidence}" for g in missing)
        return f"good: all {len(self.gates())} checks measured and passing"

    # -- serialisation ---------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict(),
            "badge": self.badge(),
            "why": self.separating(),
            "trust": self.trust.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "gates": [g.to_dict() for g in self.gates()],
            "inputs": _plain(self.inputs),
            "problem": _plain(self.problem),
            "trust_gates": self.trust_gates,
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Certificate:
        schema = str(document.get("schema", ""))
        if schema.split("/")[0] != SCHEMA.split("/")[0]:
            raise ValueError(f"this is a {SCHEMA.split('/')[0]} document, not a {schema or 'versioned'} one")
        return cls(Trust.from_dict(document["trust"]), tuple(Check.from_dict(c) for c in document.get("checks") or ()),
                   dict(document.get("inputs") or {}), dict(document.get("problem") or {}), schema or SCHEMA,
                   tuple(document.get("notes") or ()), bool(document.get("trust_gates")))

    @classmethod
    def from_json(cls, text: str) -> Certificate:
        return cls.from_dict(json.loads(text))

    def write(self, path: Any, *, indent: int | None = 2) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")
        return p

    def summary(self) -> str:
        """One line per check, for a terminal or a page."""
        st = {GATE_PASS: "PASS", GATE_FAIL: "FAIL", GATE_NO_SKILL: "NO-SKILL", GATE_UNMEASURED: "N/A "}
        lines = [f"{self.verdict()} ({self.trust.kind}) -- {self.separating()}"]
        for c in self.checks:
            v = "" if c.value is None else f" {c.value:.4g}"
            t = "" if c.threshold is None else f" (limit {c.threshold:.4g})"
            w = f" at {c.where}" if c.where and c.status != GATE_PASS else ""
            lines.append(f"[{st.get(c.status, c.status):8s}] {c.name}:{v}{t}{w}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# -------------------------------------------------------------------------------------------------- reading the input
def read_fields(source: Any, *, problem: Problem | None = None, time: Any = None) -> Any:
    """The prediction as a :class:`pinneapple_check.fields.FieldData`, whatever the caller brought.

    ``source`` is a path (``load_result`` reads an OpenFOAM case, ``.npz``/``.h5``/``.hdf5`` grid, ``.vtu``/
    ``.vtk``/``.xdmf``/``.msh`` point cloud or a ``.csv``), a mapping ``{name: array}`` (with ``problem.axes``
    carrying the coordinates), a ``(axes, fields)`` pair, or an already-read ``FieldData``.
    """
    if isinstance(source, (str, Path)):
        from pinneapple_check.fields import load_result

        return load_result(Path(source), time=time)
    if hasattr(source, "fields") and hasattr(source, "kind"):
        return source
    if isinstance(source, tuple) and len(source) == 2:
        axes, fields = source
        return _from_arrays(fields, axes)
    if isinstance(source, Mapping):
        axes = dict(problem.axes) if problem is not None else {}
        if not axes:
            raise ValueError("a mapping of arrays needs the coordinates: give them in Problem(axes=...)")
        return _from_arrays(source, axes)
    raise TypeError(f"cannot read a prediction from {type(source).__name__}: give a path, a {{name: array}} "
                    f"mapping with Problem.axes, a (axes, fields) pair, or a FieldData")


def _from_arrays(fields: Mapping[str, Any], axes: Mapping[str, Sequence[float]]) -> Any:
    from pinneapple_check.fields import FieldData

    coords = {k: np.asarray(v, float).ravel() for k, v in axes.items()}
    shape = tuple(len(v) for v in coords.values())
    out: dict[str, np.ndarray] = {}
    for name, value in fields.items():
        a = np.asarray(value, float)
        if a.shape == shape:
            out[name] = a
        elif len(shape) == 1 and a.shape == shape + (1,):
            out[name] = a[..., 0]
        else:
            raise ValueError(f"field {name!r} has shape {a.shape}, the axes {dict(zip(coords, shape, strict=True))} "
                             f"need {shape} (or {shape} + a component axis)")
    return FieldData(kind="grid", fields=out, axes=coords)


# -------------------------------------------------------------------------------------------------- the service
def verify(fields: Any, problem: Problem, *, trust: Trust | None = None, time: Any = None) -> Certificate:
    """Check a prediction against the equations of ``problem`` and return the :class:`Certificate`.

    ``fields`` is anything :func:`read_fields` accepts. ``trust`` is an already-built
    :class:`~pinneapple_analysis.trust.record.Trust` for the same case (``pp.trust(ensemble, query)``): its
    ``horizon``, ``measurement`` and ``uncertainty`` sections are kept as they are, and the certificate fills the
    residuals and the domain from its own measurement. Nothing is invented for the sections it does not measure.
    """
    from pinneapple_check.physics import equations, parse_parameters

    fd = read_fields(fields, problem=problem, time=time)
    checks: list[Check] = []

    params, problems = parse_parameters(problem.physics, dict(problem.parameters))
    for name, msg in problems:
        checks.append(Check(f"parameter:{name}", GATE_FAIL, detail=msg, source="parse_parameters", rule="PC504"))
    if problems:
        # a parameter with the wrong dimension is not a case the equations can be evaluated on: the certificate
        # says which one and stops, instead of reporting a residual computed from a differently-scaled number
        base = trust or Trust.empty(kind="field", model=problem.model)
        return _finish(base, checks, fd, problem, supplied=trust is not None)

    # -- the input itself ------------------------------------------------------------------------------------
    if fd.kind not in ("grid", "points"):
        checks.append(Check("input", GATE_UNMEASURED, source="pinneapple_check.fields.load_result",
                            detail=f"a {fd.kind} result is read but its residuals are not evaluated in this round "
                                   f"(finite-volume face sums are pinneapple_check.fv); export the field on a "
                                   f"grid or as a point cloud"))
        base = trust or Trust.empty(kind="field", model=problem.model)
        return _finish(base, checks, fd, problem, supplied=trust is not None)
    if fd.kind == "grid":
        axes = {k: np.asarray(v, float) for k, v in fd.axes.items()}
        shape = tuple(len(v) for v in axes.values())
    else:
        axes = {}
        shape = None
    finite = {n: bool(np.all(np.isfinite(np.asarray(v, float)))) for n, v in fd.fields.items()}
    bad = sorted(n for n, ok in finite.items() if not ok)
    checks.append(Check("finite", GATE_FAIL if bad else GATE_PASS, value=float(len(bad)),
                        threshold=0.0, detail=f"non-finite values in {', '.join(bad)}" if bad else "",
                        source="numpy.isfinite", rule="PC401"))
    if bad:
        base = trust or Trust.empty(kind="field", model=problem.model)
        return _finish(base, checks, fd, problem, supplied=trust is not None)

    checks += _unit_checks(fd, problem)
    provider, comps = _provider(fd, axes)
    if provider is None:
        checks.append(Check("input", GATE_UNMEASURED, source="pinneapple_check.numerics",
                            detail="the values do not form a tensor-product grid or a point cloud a derivative "
                                   "provider can be built on"))
        base = trust or Trust.empty(kind="field", model=problem.model)
        return _finish(base, checks, fd, problem, supplied=trust is not None)
    eqs, why = equations(problem.physics, provider, params, problem.flux or None)
    for reason in why:
        checks.append(Check(f"equation:{reason.split(':')[0]}", GATE_UNMEASURED, source="equations",
                            detail=reason, rule="PC301"))
    residual_checks = [_equation_check(eq, provider) for eq in eqs]
    checks += residual_checks
    if problem.boundary:
        checks += _boundary_checks(fd, problem, provider)
    for name in problem.positivity:
        checks.append(_positivity_check(fd, name, axes))
    domain_values: dict[str, TrustValue] = {}
    if problem.domain:
        check, domain_values = _domain_check(fd, problem, axes)
        checks.append(check)
    return _finish(trust or Trust.empty(kind="field", model=problem.model), checks, fd, problem,
                   supplied=trust is not None, domain_values=domain_values)


def _provider(fd: Any, axes: Mapping[str, np.ndarray] | None):
    """The derivative provider and the named scalar components, reusing ``pinneapple_check.numerics``."""
    from pinneapple_check.numerics import GridD, PointD, components

    if fd.kind == "grid":
        nd = len([a for a in axes if a != "t"])
        comps = components(dict(fd.fields), vector_ndim=len(axes), ncomp=max(nd, 1))
        return GridD(dict(axes), comps), comps
    if fd.points is None:
        return None, {}
    names = [a for a in ("x", "y", "z")][: max(fd.dim, 1) or 1]
    comps = components({k: fd.flat(k) for k in fd.fields}, vector_ndim=1, ncomp=len(names))
    try:
        return PointD(np.asarray(fd.points, float), comps, names), comps
    except Exception:                                  # too few points, or a degenerate cloud
        return None, {}


def _unit_checks(fd: Any, problem: Problem) -> list[Check]:
    """The unit of every field the equation uses, compared with the one the caller declared.

    The comparison is the linter's (PC501/PC502): same dimension and same factor, from
    :mod:`pinneapple_data.physical_units`. A field with no unit on either side is a check that could not run --
    a residual in units nobody stated is not comparable with anything.
    """
    from pinneapple_data import physical_units as U

    used = _used_fields(problem, fd)
    out: list[Check] = []
    for name in sorted(fd.fields):
        file_unit, declared = fd.units.get(name), problem.units.get(name)
        if not file_unit and not declared:
            out.append(Check(f"units:{name}", GATE_UNMEASURED if name in used else GATE_PASS,
                             source="pinneapple_data.physical_units",
                             detail=f"{name} has no unit in the file or in the problem"
                                    + (" and the equation uses it" if name in used else ""), rule="PC502"))
            continue
        if not (file_unit and declared):
            out.append(Check(f"units:{name}", GATE_PASS, source="pinneapple_data.physical_units",
                             detail=f"{name} is in {(declared or file_unit)!r}, declared on one side only",
                             rule="PC502"))
            continue
        a, b = U.try_parse_unit(file_unit), U.try_parse_unit(declared)
        if a is None or b is None:
            out.append(Check(f"units:{name}", GATE_UNMEASURED, source="pinneapple_data.physical_units",
                             detail=f"one of {file_unit!r} / {declared!r} is not a unit this library reads",
                             rule="PC502"))
            continue
        same = a.dim == b.dim and abs(a.factor / b.factor - 1) < 1e-9 and abs(a.offset - b.offset) < 1e-9
        detail = (f"{name}: the file says {file_unit}, the problem declares {declared}" if not same else
                  f"{name}: {file_unit}")
        out.append(Check(f"units:{name}", GATE_PASS if same else GATE_FAIL,
                         source="pinneapple_data.physical_units", detail=detail, rule="PC501"))
    return out


def _used_fields(problem: Problem, fd: Any) -> set[str]:
    try:
        from pinneapple_check.checks.results import _used_fields as used

        return used(problem.physics, fd)
    except Exception:
        return set(fd.fields)


def _equation_check(eq: Any, provider: Any) -> Check:
    """The relative residual of one equation, with the point where it is largest."""
    from pinneapple_check.numerics import rel_rms

    if eq.mask is not None and not np.any(eq.mask):
        return Check(f"residual:{eq.name}", GATE_UNMEASURED, law=eq.name, source="pinneapple_check.physics.equations",
                     detail=f"{eq.name}: every point is on the boundary, so the stencil never fits", rule="PC301")
    rel = rel_rms(eq.residual, eq.terms, eq.mask, floor=1e-6 * eq.scale)
    where = _worst_point(eq, provider)
    law = eq.name
    return Check(f"residual:{law}", GATE_UNMEASURED, value=float(rel), threshold=None, where=where, law=law,
                 source="pinneapple_check.physics.equations.rel_rms",
                 detail=f"{eq.description}", rule="PC301")


def _worst_point(eq: Any, provider: Any) -> str:
    """Where the equation is furthest from balanced, as coordinates."""
    r = np.asarray(eq.residual, float)
    shape = np.asarray(eq.mask).shape if eq.mask is not None else r.shape
    mag = np.linalg.norm(r, axis=-1) if r.ndim == len(shape) + 1 else np.abs(r)
    mag = np.where(eq.mask, mag, -np.inf) if eq.mask is not None else mag.ravel()
    if mag.size == 0 or not np.any(np.isfinite(mag)):
        return ""
    i = int(np.argmax(np.where(np.isfinite(mag), mag, -np.inf)))
    return _coords_of(i, provider)


def _coords_of(i: int, provider: Any) -> str:
    if hasattr(provider, "coords") and hasattr(provider, "axis_names"):
        shape = tuple(len(c) for c in provider.coords)
        idx = np.unravel_index(i, shape) if len(shape) > 1 else (i,)
        return ", ".join(f"{a}={provider.coords[k][j]:.4g}" for k, (a, j) in enumerate(zip(provider.axis_names, idx,
                                                                                           strict=True)))
    if hasattr(provider, "X"):
        return ", ".join(f"{a}={provider.X[i, k]:.4g}" for k, a in enumerate(provider.axes))
    return f"index {i}"


def _boundary_checks(fd: Any, problem: Problem, provider: Any) -> list[Check]:
    """The declared boundary values against the field on those faces, relative to the field's own range."""
    from .field_card import _axes

    if fd.kind != "grid":
        return [Check(f"boundary:{face}", GATE_UNMEASURED, source="pinneapple_analysis.trust.field_card._face",
                      detail="a boundary condition is checked on a tensor-product grid in this round", rule="PC201")
                for face in sorted(problem.boundary)]
    names, axes = _axes(fd.axes)
    shape = tuple(len(a) for a in axes)
    out: list[Check] = []
    tol = float(problem.tolerance.get("bc", DEFAULT_TOLERANCES["bc"]))
    for face, prescribed in problem.boundary.items():
        try:
            a, idx = _face(names, axes, face)
        except ValueError as exc:
            out.append(Check(f"boundary:{face}", GATE_UNMEASURED, detail=str(exc), rule="PC201"))
            continue
        face_axes = [axes[i] if i != a else axes[a][idx:idx + 1] for i in range(len(axes))]
        fp = _points(face_axes)
        for comp, spec in prescribed.items():
            if comp not in fd.fields:
                out.append(Check(f"boundary:{face}.{comp}", GATE_UNMEASURED, source="load_result",
                                 detail=f"field {comp} is not in the result", rule="PC201"))
                continue
            u = np.asarray(fd.fields[comp], float).reshape(shape)
            fu = np.take(u, idx, axis=a)
            gv = _evaluate(spec, fp, fu.shape)
            amplitude = float(np.ptp(u)) or float(np.max(np.abs(u))) or 1.0
            dev = np.abs(fu - gv) / amplitude
            worst = float(np.max(dev)) if dev.size else 0.0
            pos = np.unravel_index(int(np.argmax(dev)), fu.shape) if fu.ndim > 1 else (int(np.argmax(dev)),)
            full = list(pos)
            full.insert(a, idx)
            where = ", ".join(f"{names[i]}={axes[i][full[i]]:.4g}" for i in range(len(names)))
            out.append(Check(f"boundary:{face}.{comp}", GATE_PASS if worst <= tol else GATE_FAIL,
                             value=worst, threshold=tol, where=where,
                             source="pinneapple_analysis.trust.field_card", rule="PC201",
                             detail=f"{comp} on {face} against the prescribed value (absolute, relative to the "
                                    f"range of {comp})"))
    return out


def _positivity_check(fd: Any, name: str, axes: Mapping[str, np.ndarray] | None) -> Check:
    """A field that cannot be negative (a depth, a concentration, a density)."""
    if name not in fd.fields:
        return Check(f"positivity:{name}", GATE_UNMEASURED, source="load_result",
                     detail=f"field {name} is not in the result", rule="PC402")
    v = np.asarray(fd.fields[name], float)
    lo = float(np.nanmin(v))
    amplitude = float(np.ptp(v)) or float(np.abs(v).max()) or 1.0
    scale = 1e-9 * amplitude
    where = ""
    if lo < -scale:
        idx = np.unravel_index(int(np.nanargmin(v)), v.shape) if v.ndim > 1 else (int(np.nanargmin(v)),)
        if fd.kind == "grid" and axes:
            where = ", ".join(f"{a}={np.asarray(axes[a])[j]:.4g}" for a, j in zip(axes, idx, strict=False))
    return Check(f"positivity:{name}", GATE_PASS if lo >= -scale else GATE_FAIL, value=lo, threshold=0.0,
                 where=where, source="pinneapple_analysis.trust.certificate", rule="PC402",
                 detail=f"smallest value of {name} (a negative value where the field cannot be)")


def _domain_check(fd: Any, problem: Problem, axes: Mapping[str, np.ndarray] | None) -> tuple[Check, dict]:
    """The declared training envelope against the points of this case, through ``TrustGate``."""
    try:
        from .adapters import domain_from_gate
        from .trust_gate import TrustGate

        gate = TrustGate().fit_from_bounds(dict(problem.domain))
        import torch

        q = _domain_query(fd, problem, axes)
        in_domain, domain_score = domain_from_gate(gate, torch.tensor(q, dtype=torch.float64))
    except Exception as exc:
        why = f"the domain gate could not score this case ({type(exc).__name__}: {exc})"
        return (Check("domain", GATE_UNMEASURED, source="pinneapple_analysis.trust.trust_gate",
                      detail=why, rule="PC601"), {})
    values = {"in_domain": in_domain, "domain_score": domain_score}
    if not in_domain.is_measured:
        return Check("domain", GATE_UNMEASURED, source="pinneapple_analysis.trust.adapters.domain_from_gate",
                     detail=in_domain.reason, rule="PC601"), values
    inside = bool(in_domain.value == 1.0)
    return (Check("domain", GATE_PASS if inside else GATE_NO_SKILL, value=float(domain_score.value or 0.0),
                  threshold=1.0, direction="ge",
                  source="pinneapple_analysis.trust.trust_gate.TrustGate.ood_p_values", rule="PC601",
                  detail=f"fraction of points inside the declared domain {dict(problem.domain)} "
                         f"({domain_score.method})"), values)


def _domain_query(fd: Any, problem: Problem, axes: Mapping[str, np.ndarray] | None, cap: int = 4096) -> np.ndarray:
    """The points the domain gate scores: one column per name of ``problem.domain``, each either a grid axis or a
    case parameter held constant over the points of the case."""
    names = list(problem.domain)
    if fd.kind == "grid" and axes:
        grid = {k: np.asarray(v, float) for k, v in axes.items()}
        n = int(np.prod([len(v) for v in grid.values()]))
    else:
        P = np.asarray(fd.points, float) if fd.points is not None else np.zeros((1, 1))
        n = P.shape[0]
    step = max(1, n // cap)
    cols = []
    for name in names:
        if fd.kind == "grid" and name in (axes or {}):
            mesh = np.meshgrid(*[np.asarray(axes[k], float) for k in axes], indexing="ij")
            cols.append(mesh[list(axes).index(name)].ravel()[::step])
        else:
            value = problem.parameters.get(name)
            if value is None:
                raise ValueError(f"the domain names {name!r}, which is neither a grid axis nor a parameter")
            cols.append(np.full(len(range(0, n, step)), float(value)))
    return np.stack(cols, axis=1)


def _finish(base: Trust, checks: list[Check], fd: Any, problem: Problem, *, supplied: bool,
            domain_values: Mapping[str, TrustValue] | None = None) -> Certificate:
    """Decide the residual checks against their limits and assemble the certificate over ``base``."""
    decided = [_decide(c, problem) for c in checks]
    residuals = tuple(_residual(c) for c in decided if c.law and c.value is not None and c.rule == "PC301")
    trust = base
    if residuals:
        trust = trust.with_(residuals=residuals)
    if domain_values:
        trust = trust.with_(in_domain=domain_values["in_domain"], domain_score=domain_values["domain_score"])
    kind = base.kind or "field"
    trust = trust.with_(kind=kind, provenance=_provenance(base, fd, problem))
    return Certificate(trust=trust, checks=tuple(decided), inputs=_inputs(fd, problem), problem=problem.to_dict(),
                       notes=tuple(problem.notes), trust_gates=supplied)


def _decide(c: Check, problem: Problem) -> Check:
    """The residual check against the limit of its law: the only place a residual becomes pass or fail."""
    if c.rule != "PC301" or c.value is None:
        return c
    limit, source = problem.limit(c.law)
    if limit is None:
        return replace(c, status=GATE_UNMEASURED, threshold=None,
                       detail=(c.detail + "; " if c.detail else "")
                              + "no in-domain level and no declared tolerance for this law, so nothing says "
                                "whether the value is good")
    ok = bool(np.isfinite(c.value) and c.value <= limit)
    return replace(c, status=GATE_PASS if ok else GATE_FAIL, threshold=limit,
                   source=f"{c.source}; limit: {source}")


def _residual(c: Check) -> Residual:
    """One decided residual check as the ``Residual`` of the trust record (the same number, with its reference)."""
    value = TrustValue.measured(c.value, unit="[-]", source=c.source,
                                method="RMS of the residual over the RMS of the sum of the absolute terms of the "
                                       "equation, interior of the case", reference=c.threshold,
                                reference_source=(c.source.split("limit: ")[-1] if c.threshold is not None else ""),
                                where=c.where)
    return Residual(c.law, value, scope="interior", expert="")


def _provenance(base: Trust, fd: Any, problem: Problem) -> Provenance:
    hashes = []
    for name in sorted(fd.fields):
        h, _ = fingerprint_array(np.asarray(fd.fields[name], float))
        hashes.append(f"{name}:{h}")
    data_fp, data_method = fingerprint_text("|".join(hashes)) if all(x.split(":")[1] for x in hashes) else ("", "")
    model_fp, model_method = fingerprint_text(problem.model) if problem.model else ("", "")
    previous = base.provenance
    return Provenance(kind=previous.kind or "field", model=previous.model or problem.model,
                      model_fingerprint=previous.model_fingerprint or model_fp,
                      model_fingerprint_method=previous.model_fingerprint_method or model_method,
                      data_fingerprint=previous.data_fingerprint or data_fp,
                      data_fingerprint_method=previous.data_fingerprint_method or data_method,
                      library_version=previous.library_version or library_version(),
                      created=previous.created or utc_now(), inputs=dict(previous.inputs),
                      notes=tuple(previous.notes))


def _inputs(fd: Any, problem: Problem) -> dict[str, Any]:
    return {"kind": fd.kind, "source": str(fd.source) if fd.source else "", "fields": sorted(fd.fields),
            "shapes": {n: list(np.asarray(v).shape) for n, v in fd.fields.items()},
            "axes": {k: len(v) for k, v in (fd.axes or {}).items()},
            "units": {n: (problem.units.get(n) or fd.units.get(n) or "") for n in sorted(fd.fields)},
            "parameters": _plain(problem.parameters), "physics": problem.physics}


# -------------------------------------------------------------------------------------------------- calibration
@dataclass(frozen=True)
class CaseResult:
    """One case of a calibration: what the certificate said and what was true of the case."""

    label: str
    in_domain: bool
    verdict: str
    flags: Mapping[str, bool]           # check name -> flagged (fail or no_skill)
    values: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "in_domain": self.in_domain, "verdict": self.verdict,
                "flags": dict(self.flags), "values": dict(self.values)}


@dataclass(frozen=True)
class Calibration:
    """Detection rate and false alarms of a certificate, measured case by case.

    ``detection_rate`` is the fraction of out-of-domain cases the named gate flagged; ``false_alarm_rate`` is the
    fraction of in-domain cases it flagged. Both are counted on the same cases, so the two numbers are comparable;
    a gate that flags everything scores 1.0 and 1.0.
    """

    cases: tuple[CaseResult, ...]

    def _rate(self, name: str, *, in_domain: bool) -> tuple[int, int]:
        sel = [c for c in self.cases if c.in_domain is in_domain]
        flagged = [c for c in sel if c.flags.get(name, False)]
        return len(flagged), len(sel)

    def rate(self, name: str) -> tuple[float, float]:
        """(detection rate, false alarm rate) of the gate ``name``. A rate over no case is ``nan``."""
        det, n_out = self._rate(name, in_domain=False)
        fa, n_in = self._rate(name, in_domain=True)
        return (det / n_out if n_out else float("nan"), fa / n_in if n_in else float("nan"))

    def table(self, names: Iterable[str] | None = None) -> str:
        """The measured table: one line per gate, with the counts it was computed from."""
        out = [f"{'gate':28s} {'detection':>9s} {'false alarms':>13s}   (cases)"]
        for name in names or self.gates():
            det, n_out = self._rate(name, in_domain=False)
            fa, n_in = self._rate(name, in_domain=True)
            d = f"{det / n_out:.3f}" if n_out else "n/a"
            f = f"{fa / n_in:.3f}" if n_in else "n/a"
            out.append(f"{name:28s} {d:>9s} {f:>13s}   ({det}/{n_out} out, {fa}/{n_in} in)")
        return "\n".join(out)

    def gates(self) -> tuple[str, ...]:
        names: list[str] = []
        for c in self.cases:
            for k in c.flags:
                if k not in names:
                    names.append(k)
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "pinneapple.calibration/1",
                "rates": {n: {"detection": self.rate(n)[0], "false_alarm": self.rate(n)[1]} for n in self.gates()},
                "cases": [c.to_dict() for c in self.cases]}


def calibrate(cases: Iterable[tuple[str, Any, Problem, bool]]) -> Calibration:
    """Run :func:`verify` on a stream of ``(label, fields, problem, in_domain)`` cases and count the two rates.

    ``in_domain`` is the truth about the case (was it drawn from the domain the prediction's model was made for?),
    declared by whoever built the set: it is what the certificate is measured against, so it cannot come from the
    certificate.
    """
    out: list[CaseResult] = []
    for label, fields, problem, in_domain in cases:
        cert = verify(fields, problem)
        flags = {g.name: g.status in (GATE_FAIL, GATE_NO_SKILL) for g in cert.gates()}
        values = {c.name: c.value for c in cert.checks if c.value is not None}
        out.append(CaseResult(label, bool(in_domain), cert.verdict(), flags, values))
    return Calibration(tuple(out))
