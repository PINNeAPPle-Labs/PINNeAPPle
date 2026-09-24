"""Problem adapter: real ``ProblemSpec`` objects and free-form dicts -> decision state.

The rule backend and the constraints read a small set of canonical keys
(``representation``, ``reynolds``, ``n_simulations``, ...). Users do not write
problems with those names: ``{"name": "airfoil_flow", "reynolds": 1e5,
"steady": True, "geometry": "naca0012", "target": "pressure_field"}`` is a
normal way to describe a case. Without an adapter the constraint
``must_support_geometry`` never fires on that dict (there is no
``representation`` key), so FNO/PINO are offered for a body-fitted airfoil.

:func:`adapt_problem` maps such inputs to the canonical keys and keeps, for
every canonical fact, where its value came from:

``given``
    the input carried it under the canonical name or a synonym (``Re``,
    ``reynolds_number``, ``steady``/``transient``, ...);
``inferred``
    derived by an explicit, written rule from another field the input carried
    (for example "``geometry`` names a NACA airfoil -> body-fitted geometry ->
    ``unstructured_mesh``"). The rule text is recorded;
``unknown``
    nothing in the input determines it. The key is **left out** of the problem
    dict (no default is invented) and listed in ``unknown`` with the reason.
    Rules that need it say in their rationale what they assumed.

Input keys that are neither canonical nor a known synonym are kept in the
problem dict (the LLM backend can read them) and listed in ``unrecognized``:
nothing is derived from them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

UNKNOWN = "unknown"

#: Canonical facts the decision layer reads, and what each one is used for.
FACTS: Dict[str, str] = {
    "representation": "geometry representation (constraint must_support_geometry)",
    "geometry_complexity": "simple (box/channel/interval) or complex (body-fitted: airfoil, vehicle, CAD)",
    "dimension": "spatial dimension (1, 2 or 3)",
    "reynolds": "Reynolds number (model and training rules)",
    "time_dependent": "steady (False) or transient (True)",
    "target_kind": "what is predicted: 'field', 'kpi' or 'mixed'",
    "n_simulations": "number of reference simulations available (model rules)",
    "has_reference_data": "reference simulations or measurements exist (decision tree)",
    "has_analytical_solution": "an analytical/exact solution exists (decision tree)",
    "has_solver": "a solver can generate more data on demand (model rules)",
    "needs_parameter_generalization": "the model must generalize over a parameter family (model rules)",
    "geometry_varies": "the geometry itself varies across cases (constraint + model rules)",
    "is_inverse": "unknown physical parameters must be identified (model rules)",
}

#: Canonical keys copied as-is when present (not facts with provenance, but used by rules).
PASSTHROUGH = (
    "description", "objective", "max_relative_cost", "sharp_features", "multiscale", "stiff",
    "noisy_data", "conserved_quantities", "has_reference_solution", "has_lots_of_data",
    "governing_equations", "domain_context", "task_type", "geometry",
)

_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "reynolds": ("reynolds", "re", "reynolds_number", "reynoldsnumber"),
    "representation": ("representation", "mesh_type", "mesh", "grid", "grid_type", "discretization"),
    "dimension": ("dimension", "dim", "ndim", "spatial_dim", "spatial_dimension", "dims"),
    "n_simulations": ("n_simulations", "n_sims", "num_simulations", "n_cases", "simulations"),
    "has_reference_data": ("has_reference_data", "has_reference_simulation", "reference_simulation",
                           "reference_data", "has_cfd_data", "has_data", "reference_simulations"),
    "has_analytical_solution": ("has_analytical_solution", "analytical_solution", "analytic_solution",
                                "has_exact_solution", "exact_solution", "analytical"),
    "has_solver": ("has_solver", "solver_available", "solver"),
    "needs_parameter_generalization": ("needs_parameter_generalization", "parametric", "parameter_sweep",
                                       "sweep", "varying_parameters", "aoa_sweep", "aoa_sweep_deg"),
    "geometry_varies": ("geometry_varies", "varying_geometry", "geometries"),
    "is_inverse": ("is_inverse", "inverse", "inverse_problem"),
    "target_kind": ("target_kind", "target", "targets", "output", "outputs", "quantity_of_interest", "qoi"),
    "description": ("description", "name", "title", "goal"),
    "geometry": ("geometry", "geom", "shape", "body"),
}
_TIME_KEYS = ("time_dependent", "steady", "transient", "unsteady", "regime")

_COMPLEX_GEOMETRY = (
    (r"\bnaca\s*-?\s*\d{4,5}\b|naca\d{4,5}", "a NACA airfoil"),
    (r"air\s*foil|aerofoil|aerof[oó]lio", "an airfoil"),
    (r"\bwing\b|\bblade\b|impeller|propeller|turbine|rotor", "a lifting/rotating body"),
    (r"\bcar\b|vehicle|ahmed|drivaer|fuselage|aircraft", "a vehicle body"),
    (r"cylinder|sphere\b|\bbluff", "a bluff body in a flow"),
    (r"heat\s*sink|valve|manifold", "an engineering component"),
    (r"\bcad\b|\.stl\b|\.step\b|\.stp\b|\.msh\b|\.obj\b|\bstl\b", "a CAD/mesh file"),
)
_SIMPLE_GEOMETRY = (
    (r"unit[_ ]?(square|cube|interval)|\bsquare\b|rectangle|\bbox\b|\bcube\b", "a box domain"),
    (r"cavity", "a cavity (box) domain"),
    (r"^\s*channel\s*$|plane[_ ]channel|periodic", "a channel/periodic box"),
    (r"\binterval\b|\bline\b|\brod\b", "a 1D interval"),
)
_REPRESENTATIONS = {
    "unstructured_mesh": ("unstructured", "unstructured_mesh", "tet", "tetrahedral", "polyhedral",
                          "body_fitted", "body-fitted"),
    "structured_grid": ("structured", "structured_grid", "cartesian", "regular_grid", "uniform_grid",
                        "regular", "grid"),
    "point_cloud": ("point_cloud", "point cloud", "points", "scattered", "meshless"),
    "analytic": ("analytic", "csg", "sdf", "implicit"),
}
_FIELD_WORDS = re.compile(r"field|velocity|pressure|temperature|\bu\b|\bv\b|\bw\b|\bp\b|\bt\b|concentration|"
                          r"vorticity|displacement|stress")
_KPI_WORDS = re.compile(r"drag|lift|\bcd\b|\bcl\b|coefficient|\bkpi\b|scalar|pressure_drop|pressure drop|"
                        r"efficiency|force|moment|nusselt|erosion_rate|erosion|heat_flux_total|power")


@dataclass
class AdaptedProblem:
    """Result of :func:`adapt_problem`."""

    problem: Dict[str, Any]
    given: Dict[str, Any] = field(default_factory=dict)
    inferred: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unknown: Dict[str, str] = field(default_factory=dict)
    unrecognized: List[str] = field(default_factory=list)
    source: str = "dict"

    def report(self) -> Dict[str, Any]:
        """JSON-able provenance report, attached to ``Decision.diagnostics["problem_facts"]``."""
        return {
            "source": self.source,
            "given": dict(self.given),
            "inferred": {k: dict(v) for k, v in self.inferred.items()},
            "unknown": dict(self.unknown),
            "unrecognized": list(self.unrecognized),
        }

    def status(self, fact: str) -> str:
        if fact in self.given:
            return "given"
        if fact in self.inferred:
            return "inferred"
        return UNKNOWN


# --------------------------------------------------------------------------- helpers

def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "sim", "1"):
            return True
        if s in ("false", "no", "n", "nao", "não", "0", "none", ""):
            return False
        return None
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return None


def _presence_bool(v: Any) -> Optional[bool]:
    """For fields like ``analytical_solution`` / ``solver`` / ``reference_simulation``:
    a bool is itself; a non-empty value (a callable, a path, a name) means "exists"."""
    b = _as_bool(v)
    if b is not None:
        return b
    return True if v is not None else None


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _match(patterns, text: str) -> Optional[str]:
    for pat, label in patterns:
        if re.search(pat, text):
            return label
    return None


def _normalize_representation(v: Any) -> Optional[str]:
    s = str(v).strip().lower().replace("-", "_")
    for canon, words in _REPRESENTATIONS.items():
        if s == canon or s in words or s.replace(" ", "_") in words:
            return canon
    return None


def _dimension_from_text(text: str) -> Optional[int]:
    m = re.search(r"\b([123])\s*-?\s*d\b", text.lower())
    return int(m.group(1)) if m else None


def _classify_target(v: Any) -> Optional[str]:
    items = v if isinstance(v, (list, tuple, set)) else [v]
    kinds = set()
    for it in items:
        s = str(it).lower()
        if _KPI_WORDS.search(s):
            kinds.add("kpi")
        elif _FIELD_WORDS.search(s):
            kinds.add("field")
        else:
            return None  # one unclassifiable item -> the whole target is unknown
    if not kinds:
        return None
    return kinds.pop() if len(kinds) == 1 else "mixed"


# --------------------------------------------------------------------------- core

class _Builder:
    def __init__(self, source: str):
        self.out = AdaptedProblem(problem={}, source=source)

    def give(self, key: str, value: Any, via: str) -> None:
        if key in self.out.problem:
            return  # first (canonical) wins
        self.out.problem[key] = value
        self.out.unknown.pop(key, None)
        if key in FACTS:
            self.out.given[key] = value if via == key else {"value": value, "from": via}

    def infer(self, key: str, value: Any, rule: str) -> None:
        if key in self.out.problem:
            return  # a given value is never overridden by an inference
        self.out.problem[key] = value
        self.out.unknown.pop(key, None)
        self.out.inferred[key] = {"value": value, "rule": rule}

    def unknown(self, key: str, reason: str) -> None:
        if key not in self.out.problem:
            self.out.unknown[key] = f"{UNKNOWN}: {reason}"

    def finish(self) -> AdaptedProblem:
        for key in FACTS:
            if key not in self.out.problem and key not in self.out.unknown:
                self.out.unknown[key] = f"{UNKNOWN}: not provided and not inferable from the input"
        return self.out


def _adapt_mapping(raw: Mapping[str, Any], b: _Builder) -> None:
    by_norm: Dict[str, Tuple[str, Any]] = {}
    for k, v in raw.items():
        by_norm.setdefault(_norm_key(k), (k, v))
    used: set = set()

    def lookup(canon: str) -> Optional[Tuple[str, Any]]:
        for syn in _SYNONYMS.get(canon, (canon,)):
            if syn in by_norm and by_norm[syn][1] is not None:
                used.add(syn)
                return by_norm[syn]
        return None

    # passthrough canonical keys (exact names)
    for key in PASSTHROUGH:
        if key in raw:
            used.add(_norm_key(key))
            b.give(key, raw[key], key)

    # description / geometry synonyms
    for canon in ("description", "geometry"):
        hit = lookup(canon)
        if hit and canon not in b.out.problem:
            b.give(canon, hit[1], hit[0])

    # reynolds
    hit = lookup("reynolds")
    if hit:
        re_ = _as_float(hit[1])
        if re_ is None:
            b.unknown("reynolds", f"could not parse {hit[0]}={hit[1]!r} as a number")
        else:
            b.give("reynolds", re_, hit[0])

    # time dependence
    for tk in _TIME_KEYS:
        if tk not in by_norm:
            continue
        orig, v = by_norm[tk]
        used.add(tk)
        if tk == "regime":
            s = str(v).lower()
            if "steady" in s and "unsteady" not in s:
                b.give("time_dependent", False, orig)
            elif any(w in s for w in ("transient", "unsteady", "time")):
                b.give("time_dependent", True, orig)
            else:
                b.unknown("time_dependent", f"{orig}={v!r} is neither steady nor transient")
        else:
            bv = _as_bool(v)
            if bv is None:
                b.unknown("time_dependent", f"could not read {orig}={v!r} as a boolean")
            else:
                b.give("time_dependent", (not bv) if tk == "steady" else bv, orig)
        break

    # representation (explicit)
    hit = lookup("representation")
    if hit:
        rep = _normalize_representation(hit[1])
        if rep is None:
            b.unknown("representation", f"{hit[0]}={hit[1]!r} is not a known representation "
                                        f"({sorted(_REPRESENTATIONS)})")
        else:
            b.give("representation", rep, hit[0])

    # dimension
    hit = lookup("dimension")
    if hit:
        v = hit[1]
        d = int(v) if isinstance(v, int) and not isinstance(v, bool) else _dimension_from_text(str(v))
        if d in (1, 2, 3):
            b.give("dimension", d, hit[0])
        else:
            b.unknown("dimension", f"could not read {hit[0]}={v!r} as 1, 2 or 3")

    # counts / booleans
    hit = lookup("n_simulations")
    if hit:
        v = hit[1]
        n = len(v) if isinstance(v, (list, tuple)) else _as_float(v)
        if n is None or n < 0:
            b.unknown("n_simulations", f"could not read {hit[0]}={v!r} as a count")
        else:
            b.give("n_simulations", int(n), hit[0])

    for canon, reader in (("has_reference_data", _presence_bool), ("has_analytical_solution", _presence_bool),
                          ("has_solver", _presence_bool), ("is_inverse", _as_bool),
                          ("needs_parameter_generalization", _as_bool)):
        hit = lookup(canon)
        if hit:
            bv = reader(hit[1])
            if bv is None:
                b.unknown(canon, f"could not read {hit[0]}={hit[1]!r} as yes/no")
            else:
                b.give(canon, bv, hit[0])

    hit = lookup("geometry_varies")
    if hit:
        v = hit[1]
        bv = (len(v) > 1) if isinstance(v, (list, tuple)) else _as_bool(v)
        if bv is None:
            b.unknown("geometry_varies", f"could not read {hit[0]}={v!r} as yes/no")
        else:
            b.give("geometry_varies", bv, hit[0])

    hit = lookup("target_kind")
    if hit:
        kind = hit[1] if hit[0] == "target_kind" else _classify_target(hit[1])
        if kind in ("field", "kpi", "mixed"):
            b.give("target_kind", kind, hit[0])
            if hit[0] != "target_kind":
                b.out.problem.setdefault("target", hit[1])
        else:
            b.unknown("target_kind", f"{hit[0]}={hit[1]!r} is neither a field nor a KPI by name")

    if "geometry_complexity" in by_norm:
        v = str(by_norm["geometry_complexity"][1]).lower()
        if v in ("simple", "complex"):
            b.give("geometry_complexity", v, "geometry_complexity")
        else:
            b.unknown("geometry_complexity", f"geometry_complexity={v!r} is neither 'simple' nor 'complex'")

    if "target" in by_norm and "target" not in b.out.problem:
        b.out.problem["target"] = by_norm["target"][1]  # keep what the user asked to predict

    # task keyword ("task": "inverse")
    if "task" in by_norm:
        used.add("task")
        s = str(by_norm["task"][1]).lower()
        if "inverse" in s:
            b.give("is_inverse", True, by_norm["task"][0])

    known_norm = {_norm_key(k) for k in PASSTHROUGH} | {_norm_key(k) for k in FACTS} | set(_TIME_KEYS)
    for syns in _SYNONYMS.values():
        known_norm |= {_norm_key(s) for s in syns}
    known_norm.add("task")
    for k, v in raw.items():
        nk = _norm_key(k)
        if nk in known_norm:
            continue
        b.out.unrecognized.append(str(k))
        b.out.problem.setdefault(str(k), v)


def _adapt_spec(spec: Any, b: _Builder) -> None:
    """``pinneapple_problemdesign.ProblemSpec`` (duck-typed: physics, geometry, task_type)."""
    desc = getattr(spec, "goal", "") or getattr(spec, "title", "")
    if desc:
        b.give("description", desc, "ProblemSpec.goal" if spec.goal else "ProblemSpec.title")
    tt = getattr(spec, "task_type", "other")
    if tt and tt != "other":
        b.give("task_type", tt, "task_type")
        b.infer("is_inverse", tt == "inverse_problem", f"ProblemSpec.task_type={tt!r}")
        if tt == "neural_operator":
            b.infer("needs_parameter_generalization", True, "ProblemSpec.task_type='neural_operator'")
        if tt == "forecasting":
            b.infer("time_dependent", True, "ProblemSpec.task_type='forecasting'")
    if getattr(spec, "domain_context", ""):
        b.give("domain_context", spec.domain_context, "domain_context")

    g, ph = spec.geometry, spec.physics
    if g.representation:
        rep = _normalize_representation(g.representation)
        if rep:
            b.give("representation", rep, "ProblemSpec.geometry.representation")
        else:
            b.unknown("representation", f"ProblemSpec.geometry.representation={g.representation!r} "
                                        "is not a known representation")
    if g.domain:
        b.give("geometry", g.domain, "ProblemSpec.geometry.domain")
    aoa = getattr(g, "aoa_sweep_deg", None)
    if aoa and len(aoa) > 1:
        b.infer("needs_parameter_generalization", True,
                f"ProblemSpec.geometry.aoa_sweep_deg has {len(aoa)} angles -> surrogate over a parameter family")
    if getattr(g, "cad_format", None):
        b.infer("geometry_complexity", "complex", f"ProblemSpec.geometry.cad_format={g.cad_format!r}")

    if ph.governing_equations:
        b.give("governing_equations", list(ph.governing_equations), "ProblemSpec.physics.governing_equations")
    if ph.initial_conditions:
        b.infer("time_dependent", True, "ProblemSpec.physics.initial_conditions is non-empty")

    data = getattr(spec, "data", None)
    if data is not None and getattr(data, "sources", None):
        b.infer("has_reference_data", True, f"ProblemSpec.data.sources={list(data.sources)!r}")
    outputs = getattr(spec, "outputs", None) or (list(data.target_variables) if data is not None else [])
    if outputs:
        kind = _classify_target(outputs)
        if kind:
            b.infer("target_kind", kind, f"ProblemSpec outputs {list(outputs)!r}")
            b.out.problem.setdefault("target", list(outputs))
        else:
            b.unknown("target_kind", f"outputs {list(outputs)!r} are neither fields nor KPIs by name")


def _derive(b: _Builder) -> None:
    """Inferences that combine already-read facts. Each one records its rule."""
    p = b.out.problem
    geom = p.get("geometry")
    if geom is not None and not isinstance(geom, (dict, list)):
        text = str(geom).lower()
        complex_label = _match(_COMPLEX_GEOMETRY, text)
        simple_label = None if complex_label else _match(_SIMPLE_GEOMETRY, text)
        if complex_label:
            b.infer("geometry_complexity", "complex", f"geometry {geom!r} names {complex_label}")
            b.infer("representation", "unstructured_mesh",
                    f"geometry {geom!r} names {complex_label}: a body-fitted geometry that is not a "
                    "structured grid -> 'unstructured_mesh'")
        elif simple_label:
            b.infer("geometry_complexity", "simple", f"geometry {geom!r} names {simple_label}")
            b.infer("representation", "structured_grid",
                    f"geometry {geom!r} names {simple_label}, which a structured grid covers -> 'structured_grid'")
        else:
            b.unknown("representation", f"geometry {geom!r} is not a geometry name this adapter knows; "
                                        "set 'representation' explicitly")
            b.unknown("geometry_complexity", f"geometry {geom!r} is not a geometry name this adapter knows")
    if "dimension" not in p:
        for key in ("description", "geometry"):
            if p.get(key):
                d = _dimension_from_text(str(p[key]))
                if d:
                    b.infer("dimension", d, f"{key} {p[key]!r} contains '{d}D'")
                    break
    if "n_simulations" in p and "has_reference_data" not in p:
        b.infer("has_reference_data", p["n_simulations"] > 0, f"n_simulations={p['n_simulations']}")
    if p.get("has_analytical_solution") is True and "has_reference_solution" not in p:
        b.infer("has_reference_solution", True, "has_analytical_solution=True")


def adapt_problem(obj: Any) -> AdaptedProblem:
    """Adapt a ``ProblemSpec``, a free-form dict, or ``None`` to the decision layer.

    Idempotent on canonical dicts: adapting an already-adapted problem returns
    the same values (their provenance then reads as ``given``).
    """
    if obj is None:
        return _Builder("empty").finish()
    if hasattr(obj, "physics") and hasattr(obj, "geometry") and hasattr(obj, "task_type"):
        b = _Builder("ProblemSpec")
        _adapt_spec(obj, b)
    elif isinstance(obj, Mapping):
        b = _Builder("dict")
        _adapt_mapping(obj, b)
    else:
        raise TypeError(f"cannot adapt a problem of type {type(obj).__name__}")
    _derive(b)
    return b.finish()


__all__ = ["AdaptedProblem", "FACTS", "UNKNOWN", "adapt_problem"]
