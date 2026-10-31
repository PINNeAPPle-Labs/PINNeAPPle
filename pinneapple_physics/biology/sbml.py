"""SBML to :class:`~pinneapple_physics.biology.ode_system.ODESystem`, and back.

:func:`read_sbml` reads SBML Level 2 and Level 3 Core (the form of the curated BioModels entries) with the standard
library's XML parser and a MathML-to-SymPy translator; python-libsbml is not required (when it is installed,
``validate=True`` also runs its consistency checks first). What it reads, following the SBML L3V2 specification
(Hucka et al., J Integr Bioinform 16(2), 2019), section 4:

* compartments, species and parameters with their values; a species is a state when a reaction or a rate rule changes
  it and it is neither ``constant`` nor a ``boundaryCondition`` species without a rule;
* reactions: ``d[S]/dt = sum(stoichiometry * rate) / V`` for a species in concentration units
  (``hasOnlySubstanceUnits="false"``, the symbol in every formula is the concentration), ``d n_S / dt = sum(...)``
  for one in substance units; local (kinetic-law) parameters are renamed ``<reaction>_<parameter>``; Level 3
  ``conversionFactor`` (model or species);
* assignment rules (algebraic variables, sorted so each uses only earlier ones), rate rules (any symbol becomes a
  state), initial assignments, function definitions (inlined), ``stoichiometryMath`` (Level 2);
* MathML: arithmetic, ``power``/``root``/``exp``/``ln``/``log``, trigonometric and hyperbolic functions, ``abs``,
  ``floor``/``ceiling``, ``piecewise``, relations and logic, the ``time`` csymbol, ``pi``/``exponentiale``/
  ``infinity``.

Not supported, and refused with :class:`SBMLUnsupported` rather than silently ignored: events, algebraic rules,
delays, variable-size compartments, ``fast`` reactions, and packages (comp, fbc, ...). A model that needs one of these
cannot be simulated faithfully as a plain ODE system.

:func:`write_sbml` writes an :class:`ODESystem` as SBML L3V2 with rate rules (one parameter per state), assignment
rules and initial assignments, so libRoadRunner or COPASI can simulate the same equations.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .ode_system import TIME, ODESystem

__all__ = ["read_sbml", "write_sbml", "mathml_to_sympy", "SBMLUnsupported", "SBMLError", "biomodel", "list_biomodels",
           "BIOMODELS_DIR"]

MATHML = "http://www.w3.org/1998/Math/MathML"
BIOMODELS_DIR = Path(__file__).parent / "data" / "biomodels"
TIME_URL = "http://www.sbml.org/sbml/symbols/time"


class SBMLError(ValueError):
    """The document is not SBML this reader understands (malformed, missing values)."""


class SBMLUnsupported(SBMLError):
    """The model uses an SBML construct an ODE system cannot represent faithfully (events, delays, ...)."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name: str):
    return [c for c in el if _local(c.tag) == name]


def _child(el, name: str):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _list(model, list_name: str, item: str):
    lst = _child(model, list_name)
    return _children(lst, item) if lst is not None else []


def _s(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


# --------------------------------------------------------------------------- MathML
_NARY = {"plus": sp.Add, "times": sp.Mul, "and": sp.And, "or": sp.Or, "xor": sp.Xor,
         "max": sp.Max, "min": sp.Min}
_UNARY = {"exp": sp.exp, "ln": sp.log, "abs": sp.Abs, "floor": sp.floor, "ceiling": sp.ceiling, "not": sp.Not,
          "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
          "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "sech": sp.sech, "csch": sp.csch, "coth": sp.coth,
          "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "arcsinh": sp.asinh, "arccosh": sp.acosh,
          "arctanh": sp.atanh, "factorial": sp.factorial}
_REL = {"eq": sp.Eq, "neq": sp.Ne, "gt": sp.StrictGreaterThan, "lt": sp.StrictLessThan, "geq": sp.GreaterThan,
        "leq": sp.LessThan}


def _cn(el) -> sp.Expr:
    kind = el.get("type", "real")
    parts = [p.strip() for p in (el.text or "").split()]
    sep = [_local(c.tag) for c in el]
    if kind in ("e-notation", "rational") or "sep" in sep:
        a = float((el.text or "").strip())
        b = float((list(el)[0].tail or "").strip())
        return sp.Float(a * 10 ** b) if kind == "e-notation" else sp.Rational(int(a), int(b))
    txt = "".join(parts)
    if kind == "integer":
        return sp.Integer(int(txt))
    v = float(txt)
    if v.is_integer() and abs(v) < 1e15:
        return sp.Integer(int(v))
    return sp.Float(v)


def mathml_to_sympy(el, functions: Mapping[str, Any] | None = None,
                    rename: Mapping[str, str] | None = None) -> sp.Expr:
    """One MathML content element (``<math>`` or anything inside it) to a SymPy expression. ``functions`` are SBML
    function definitions (SymPy ``Lambda``); ``rename`` maps identifiers (local kinetic-law parameters)."""
    functions = functions or {}
    rename = rename or {}
    tag = _local(el.tag)
    if tag == "math":
        kids = [c for c in el if _local(c.tag) not in ("annotation", "semantics")]
        if len(kids) != 1:
            raise SBMLError("<math> must contain exactly one expression")
        return mathml_to_sympy(kids[0], functions, rename)
    if tag == "semantics":
        return mathml_to_sympy(list(el)[0], functions, rename)
    if tag == "cn":
        return _cn(el)
    if tag == "ci":
        name = (el.text or "").strip()
        return _s(rename.get(name, name))
    if tag == "csymbol":
        url = el.get("definitionURL", "")
        if url == TIME_URL:
            return TIME
        raise SBMLUnsupported(f"csymbol {url} (delays and rate-of are not supported)")
    if tag in ("true", "false"):
        return sp.true if tag == "true" else sp.false
    if tag == "pi":
        return sp.pi
    if tag == "exponentiale":
        return sp.E
    if tag == "infinity":
        return sp.oo
    if tag == "notanumber":
        return sp.nan
    if tag == "piecewise":
        args = []
        for c in el:
            t = _local(c.tag)
            kids = list(c)
            if t == "piece":
                args.append((mathml_to_sympy(kids[0], functions, rename),
                             mathml_to_sympy(kids[1], functions, rename)))
            elif t == "otherwise":
                args.append((mathml_to_sympy(kids[0], functions, rename), True))
        if not args or args[-1][1] is not True:
            args.append((sp.nan, True))
        return sp.Piecewise(*args)
    if tag == "lambda":
        raise SBMLError("lambda outside a function definition")
    if tag != "apply":
        raise SBMLUnsupported(f"MathML element <{tag}>")
    kids = list(el)
    op_el, args_el = kids[0], kids[1:]
    op = _local(op_el.tag)
    qual = {_local(a.tag): a for a in args_el if _local(a.tag) in ("degree", "logbase")}
    args = [mathml_to_sympy(a, functions, rename) for a in args_el if _local(a.tag) not in ("degree", "logbase")]
    if op == "ci":
        fname = (op_el.text or "").strip()
        if fname not in functions:
            raise SBMLError(f"call of undefined function {fname}")
        return functions[fname](*args)
    if op == "csymbol":
        raise SBMLUnsupported(f"csymbol function {op_el.get('definitionURL', '')} (delay is not supported)")
    if op in _NARY:
        if not args:
            return {"plus": sp.Integer(0), "times": sp.Integer(1), "and": sp.true, "or": sp.false}[op]
        return _NARY[op](*args)
    if op == "minus":
        return -args[0] if len(args) == 1 else args[0] - args[1]
    if op == "divide":
        return args[0] / args[1]
    if op == "power":
        return args[0] ** args[1]
    if op == "root":
        deg = mathml_to_sympy(list(qual["degree"])[0], functions, rename) if "degree" in qual else 2
        return args[0] ** (sp.Integer(1) / deg)
    if op == "log":
        base = mathml_to_sympy(list(qual["logbase"])[0], functions, rename) if "logbase" in qual else 10
        return sp.log(args[0], base)
    if op in _UNARY:
        return _UNARY[op](args[0])
    if op in _REL:
        if len(args) == 2:
            return _REL[op](args[0], args[1])
        return sp.And(*[_REL[op](a, b) for a, b in zip(args, args[1:], strict=False)])
    if op == "quotient":
        return sp.floor(args[0] / args[1])
    if op == "rem":
        return sp.Mod(args[0], args[1])
    raise SBMLUnsupported(f"MathML operator <{op}>")


def _function_definitions(model) -> dict[str, sp.Lambda]:
    out: dict[str, sp.Lambda] = {}
    for fd in _list(model, "listOfFunctionDefinitions", "functionDefinition"):
        math_el = _child(fd, "math")
        lam = [c for c in math_el if _local(c.tag) in ("lambda", "semantics")][0]
        if _local(lam.tag) == "semantics":
            lam = list(lam)[0]
        bvars = [_s((_child(b, "ci").text or "").strip()) for b in _children(lam, "bvar")]
        body = [c for c in lam if _local(c.tag) != "bvar"][0]
        out[fd.get("id")] = sp.Lambda(tuple(bvars), mathml_to_sympy(body, out))
    return out


def _math(el, functions, rename=None):
    m = _child(el, "math")
    if m is None:
        raise SBMLError(f"<{_local(el.tag)} id={el.get('id') or el.get('variable')}> has no math")
    return mathml_to_sympy(m, functions, rename)


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None else v.strip().lower() == "true"


def _float(v: str | None) -> float | None:
    return None if v is None else float(v)


def _toposort(rules: dict[str, sp.Expr]) -> dict[str, sp.Expr]:
    names = set(rules)
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(n):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise SBMLError(f"assignment rules form a cycle through {n}")
        state[n] = 1
        for s in rules[n].free_symbols:
            if str(s) in names:
                visit(str(s))
        state[n] = 2
        order.append(n)

    for n in rules:
        visit(n)
    return {n: rules[n] for n in order}


def _validate_with_libsbml(text: str) -> None:
    try:
        import libsbml
    except ImportError:
        return
    doc = libsbml.readSBMLFromString(text)
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    doc.checkConsistency()
    errors = [doc.getError(i) for i in range(doc.getNumErrors())
              if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
    if errors:
        raise SBMLError("libsbml: " + "; ".join(e.getMessage().strip() for e in errors[:5]))


def read_sbml(source: str | Path, *, validate: bool = False, name: str | None = None) -> ODESystem:
    """Read an SBML file (path) or document (text starting with ``<``) into an :class:`ODESystem`.

    The system's ``meta`` keeps the SBML level/version, the compartment of each species, which species are in
    concentration units, and the reactions' rate expressions (``meta["reactions"]``)."""
    text = str(source)
    if not text.lstrip().startswith("<"):
        text = Path(source).read_text(encoding="utf-8")
    if validate:
        _validate_with_libsbml(text)
    try:
        root = ET.fromstring(text.encode("utf-8"))
    except ET.ParseError as exc:
        raise SBMLError(f"not XML: {exc}") from exc
    if _local(root.tag) != "sbml":
        raise SBMLError(f"root element is <{_local(root.tag)}>, not <sbml>")
    level, version = int(root.get("level", "3")), int(root.get("version", "1"))
    model = _child(root, "model")
    if model is None:
        raise SBMLError("no <model>")
    if _list(model, "listOfEvents", "event"):
        raise SBMLUnsupported(f"events ({len(_list(model, 'listOfEvents', 'event'))}) are not supported")
    for pkg in root.attrib:
        if pkg.endswith(":required") and root.attrib[pkg] == "true":
            raise SBMLUnsupported(f"required package {pkg}")

    functions = _function_definitions(model)
    params: dict[str, float] = {}
    constant: dict[str, bool] = {}
    compartments: dict[str, float] = {}
    units: dict[str, str] = {}

    for c in _list(model, "listOfCompartments", "compartment"):
        cid = c.get("id")
        size = _float(c.get("size") if c.get("size") is not None else c.get("volume"))
        dims = c.get("spatialDimensions")
        if size is None and (dims is None or float(dims) != 0):
            size = 1.0 if level == 2 else None
        compartments[cid] = size if size is not None else math.nan
        params[cid] = compartments[cid]
        constant[cid] = _bool(c.get("constant"), True)
    for p in _list(model, "listOfParameters", "parameter"):
        v = _float(p.get("value"))
        params[p.get("id")] = math.nan if v is None else v
        constant[p.get("id")] = _bool(p.get("constant"), True)
        if p.get("units"):
            units[p.get("id")] = p.get("units")

    species: dict[str, dict[str, Any]] = {}
    model_cf = model.get("conversionFactor")
    for s in _list(model, "listOfSpecies", "species"):
        sid = s.get("id")
        comp = s.get("compartment")
        only_sub = _bool(s.get("hasOnlySubstanceUnits"), False)
        amount, conc = _float(s.get("initialAmount")), _float(s.get("initialConcentration"))
        V = _s(comp)
        if only_sub:
            init = sp.Float(amount) if amount is not None else (sp.Float(conc) * V if conc is not None else None)
        else:
            init = sp.Float(conc) if conc is not None else (sp.Float(amount) / V if amount is not None else None)
        species[sid] = {"compartment": comp, "only_substance": only_sub, "initial": init,
                        "boundary": _bool(s.get("boundaryCondition"), False),
                        "constant": _bool(s.get("constant"), False),
                        "conversion": s.get("conversionFactor") or model_cf}

    # initial assignments override declared values
    init_assign: dict[str, sp.Expr] = {}
    for ia in _list(model, "listOfInitialAssignments", "initialAssignment"):
        init_assign[ia.get("symbol")] = _math(ia, functions)

    assignments: dict[str, sp.Expr] = {}
    rate_rules: dict[str, sp.Expr] = {}
    for r in _list(model, "listOfRules", "assignmentRule"):
        assignments[r.get("variable")] = _math(r, functions)
    for r in _list(model, "listOfRules", "rateRule"):
        rate_rules[r.get("variable")] = _math(r, functions)
    if _list(model, "listOfRules", "algebraicRule"):
        raise SBMLUnsupported("algebraic rules (DAE) are not supported")
    for v in rate_rules:
        if v in compartments:
            raise SBMLUnsupported(f"compartment {v} changes size (rate rule): variable compartments are not supported")

    # reactions
    flux: dict[str, sp.Expr] = {}
    reactions: dict[str, sp.Expr] = {}
    for rx in _list(model, "listOfReactions", "reaction"):
        rid = rx.get("id")
        if _bool(rx.get("fast"), False):
            raise SBMLUnsupported(f"reaction {rid} is fast (quasi-steady state) and is not supported")
        kl = _child(rx, "kineticLaw")
        if kl is None:
            raise SBMLError(f"reaction {rid} has no kinetic law")
        rename = {}
        for lst, item in (("listOfParameters", "parameter"), ("listOfLocalParameters", "localParameter")):
            for lp in _list(kl, lst, item):
                new = f"{rid}_{lp.get('id')}"
                while new in params:
                    new += "_"
                rename[lp.get("id")] = new
                v = _float(lp.get("value"))
                params[new] = math.nan if v is None else v
                constant[new] = True
        rate = _math(kl, functions, rename)
        reactions[rid] = rate
        for lst, sign in (("listOfReactants", -1), ("listOfProducts", 1)):
            for sr in _list(rx, lst, "speciesReference"):
                sid = sr.get("species")
                sm = _child(sr, "stoichiometryMath")
                if sm is not None:
                    st = _math(sm, functions)
                elif sr.get("stoichiometry") is not None:
                    st = sp.sympify(float(sr.get("stoichiometry")))
                    if st == int(st):
                        st = sp.Integer(int(st))
                elif level >= 3 and sr.get("id") and sr.get("id") in init_assign:
                    st = init_assign[sr.get("id")]
                else:
                    st = sp.Integer(1)
                if sr.get("id") and (sr.get("id") in assignments or sr.get("id") in rate_rules):
                    raise SBMLUnsupported(f"variable stoichiometry {sr.get('id')} is not supported")
                flux[sid] = flux.get(sid, sp.Integer(0)) + sign * st * rate

    # states and right-hand sides
    states: list[str] = []
    rhs: dict[str, sp.Expr] = {}
    init_values: dict[str, Any] = {}
    for sid, info in species.items():
        if sid in assignments:
            continue
        if sid in rate_rules:
            states.append(sid)
            rhs[sid] = rate_rules[sid]
        elif sid in flux and not info["boundary"] and not info["constant"]:
            d = flux[sid]
            if info["conversion"]:
                d = d * _s(info["conversion"])
            if not info["only_substance"]:
                d = d / _s(info["compartment"])
            states.append(sid)
            rhs[sid] = d
        else:
            # constant or boundary species: a parameter with its initial value
            v = init_assign.get(sid, info["initial"])
            if v is None:
                raise SBMLError(f"species {sid} has no initial value")
            if v.free_symbols:
                assignments[sid] = v
            else:
                params[sid] = float(v)
            continue
        v = init_assign.get(sid, info["initial"])
        if v is None:
            raise SBMLError(f"species {sid} has no initial value")
        init_values[sid] = v
    for v, e in rate_rules.items():
        if v in species:
            continue
        states.append(v)
        rhs[v] = e
        init_values[v] = init_assign.get(v, sp.Float(params.pop(v)))
    # parameters set by initial assignments become assignments evaluated from the others (they are constant);
    # parameters with an assignment rule are algebraic variables
    for v, e in init_assign.items():
        if v in params and v not in assignments and v not in rate_rules:
            params.pop(v)
            assignments[v] = e
    for v in assignments:
        params.pop(v, None)
    for p, val in params.items():
        if isinstance(val, float) and math.isnan(val):
            raise SBMLError(f"{p} has no value")

    assignments = _toposort(assignments)
    # SBML allows an initial value to reference other symbols' initial values; keep them symbolic
    sysname = name or model.get("id") or model.get("name") or "sbml_model"
    sys_ = ODESystem(sysname, states, rhs, init_values, params, assignments, units,
                     meta={"sbml_level": level, "sbml_version": version, "reactions": reactions,
                           "compartments": {s: species[s]["compartment"] for s in species},
                           "concentration_species": [s for s in species if not species[s]["only_substance"]],
                           "model_name": model.get("name", "")})
    return sys_


def list_biomodels() -> list[str]:
    """The BioModels entries shipped with the package (CC0; see ``data/biomodels/README.md``)."""
    return sorted(p.stem for p in BIOMODELS_DIR.glob("BIOMD*.xml"))


def biomodel(model_id: str | int) -> ODESystem:
    """A shipped BioModels entry as an :class:`ODESystem`: ``biomodel(12)`` or ``biomodel("BIOMD0000000012")``."""
    mid = f"BIOMD{int(model_id):010d}" if str(model_id).isdigit() else str(model_id)
    path = BIOMODELS_DIR / f"{mid}.xml"
    if not path.exists():
        raise KeyError(f"{mid} is not shipped; available: {list_biomodels()}")
    return read_sbml(path, name=mid)


# --------------------------------------------------------------------------- writer
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_PRINT_FUNC = {v: k for k, v in _UNARY.items() if k not in ("not",)}
_PRINT_FUNC[sp.log] = "ln"
_PRINT_REL = {sp.Eq: "eq", sp.Ne: "neq", sp.StrictGreaterThan: "gt", sp.StrictLessThan: "lt",
              sp.GreaterThan: "geq", sp.LessThan: "leq"}


def _to_mathml(e) -> str:
    """Content MathML for a SymPy expression (the subset :func:`mathml_to_sympy` reads back)."""
    if e is TIME:
        return f'<csymbol encoding="text" definitionURL="{TIME_URL}">t</csymbol>'
    if e is sp.true or e is sp.false:
        return f"<{str(e).lower()}/>"
    if isinstance(e, sp.Symbol):
        return f"<ci>{e.name}</ci>"
    if e is sp.pi:
        return "<pi/>"
    if e is sp.E:
        return "<exponentiale/>"
    if e is sp.oo:
        return "<infinity/>"
    if e is sp.nan:
        return "<notanumber/>"
    if isinstance(e, sp.Integer):
        return f'<cn type="integer">{int(e)}</cn>'
    if isinstance(e, sp.Rational):
        return f"<apply><divide/>{_to_mathml(e.p)}{_to_mathml(e.q)}</apply>"
    if isinstance(e, (int,)):
        return f'<cn type="integer">{e}</cn>'
    if e.is_Number:
        return f"<cn>{float(e)!r}</cn>"
    args = "".join(_to_mathml(a) for a in e.args)
    if isinstance(e, sp.Add):
        return f"<apply><plus/>{args}</apply>"
    if isinstance(e, sp.Mul):
        return f"<apply><times/>{args}</apply>"
    if isinstance(e, sp.Pow):
        return f"<apply><power/>{args}</apply>"
    if isinstance(e, sp.exp):
        return f"<apply><exp/>{args}</apply>"
    if isinstance(e, sp.Piecewise):
        body = ""
        for val, cond in e.args:
            if cond is sp.true:
                body += f"<otherwise>{_to_mathml(val)}</otherwise>"
            else:
                body += f"<piece>{_to_mathml(val)}{_to_mathml(cond)}</piece>"
        return f"<piecewise>{body}</piecewise>"
    for cls, tag in ((sp.And, "and"), (sp.Or, "or"), (sp.Not, "not"), (sp.Xor, "xor"), (sp.Max, "max"),
                     (sp.Min, "min")):
        if isinstance(e, cls):
            return f"<apply><{tag}/>{args}</apply>"
    for cls, tag in _PRINT_REL.items():
        if type(e) is cls:
            return f"<apply><{tag}/>{args}</apply>"
    tag = _PRINT_FUNC.get(type(e))
    if tag is not None:
        return f"<apply><{tag}/>{args}</apply>"
    raise SBMLError(f"cannot write {type(e).__name__} as MathML")


def _content_mathml(expr) -> str:
    return f'<math xmlns="{MATHML}">{_to_mathml(sp.sympify(expr))}</math>'


def write_sbml(system: ODESystem) -> str:
    """SBML Level 3 Version 2 text for ``system`` (see :meth:`ODESystem.to_sbml`)."""
    for n in [*system.states, *system.parameters, *system.assignments]:
        if not _ID.match(n) or n == "t":
            raise SBMLError(f"{n!r} is not a valid SBML identifier")
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">',
           f'  <model id="{system.name if _ID.match(system.name) else "model"}">',
           "    <listOfParameters>"]
    y0 = {}
    for s in system.states:
        v = sp.sympify(system.initial[s])
        y0[s] = v
        val = f' value="{float(v)!r}"' if not v.free_symbols else ""
        out.append(f'      <parameter id="{s}"{val} constant="false"/>')
    for p, v in system.parameters.items():
        out.append(f'      <parameter id="{p}" value="{v!r}" constant="true"/>')
    for a in system.assignments:
        out.append(f'      <parameter id="{a}" constant="false"/>')
    out.append("    </listOfParameters>")
    exprs = {s: v for s, v in y0.items() if v.free_symbols}
    if exprs:
        out.append("    <listOfInitialAssignments>")
        for s, v in exprs.items():
            out.append(f'      <initialAssignment symbol="{s}">{_content_mathml(v)}</initialAssignment>')
        out.append("    </listOfInitialAssignments>")
    out.append("    <listOfRules>")
    for a, e in system.assignments.items():
        out.append(f'      <assignmentRule variable="{a}">{_content_mathml(e)}</assignmentRule>')
    for s in system.states:
        out.append(f'      <rateRule variable="{s}">{_content_mathml(system.rhs[s])}</rateRule>')
    out += ["    </listOfRules>", "  </model>", "</sbml>", ""]
    return "\n".join(out)
