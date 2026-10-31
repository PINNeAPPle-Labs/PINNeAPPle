"""Every public class and function of the library as a tool: executable, composite or reference, with the reason.

The category is computed from the code (the inventory's parameters, annotations, bases, decorators, module path and
flags), never assigned by hand:

* ``executable`` (a): a user can call it from a form. Every required input has a type a form can hold (numbers,
  text, booleans, lists and dicts of those, ``Literal`` choices, paths); a class also needs a method that does the work
  (``run``, ``solve``, ``predict``, ...).
* ``composite`` (b): it only makes sense inside a pipeline: it takes or is an object other tools build (a tensor, an
  array, a model, a problem, a callable), or it is a network layer, a loss, a callback or an adapter.
* ``reference`` (c): infrastructure: exceptions, type definitions, entry points (CLI, servers), internal or
  deprecated modules, plain data records.

:func:`classify` returns ``(category, reason)``; the reason names the parameter, base or rule that decided it.
:func:`form` is the JSON Schema of an executable tool's inputs, with the default and, when the docstring gives them,
the description and the unit of each input.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

CATEGORIES = ("executable", "composite", "reference")

# types a form can hold
SCALARS = {"int": "integer", "float": "number", "bool": "boolean", "str": "string", "complex": None,
           "Path": "string", "PathLike": "string", "None": "null", "NoneType": "null"}
CONTAINERS = {"list", "List", "tuple", "Tuple", "Sequence", "Iterable", "set", "Set", "frozenset", "dict", "Dict",
              "Mapping", "MutableMapping", "Optional", "Union", "Literal"}
ARRAYS = {"ndarray", "ArrayLike", "NDArray"}
RUN_METHODS = ("run", "solve", "predict", "evaluate", "simulate", "compute", "analyze", "analyse", "fit", "train",
               "optimize", "optimise", "check", "validate", "build", "generate", "sample", "__call__")
# bases that make a class a building block of a pipeline
BLOCK_BASES = re.compile(r"(^|\.)(Module|Layer|Loss|_Loss|Callback|Adapter|Dataset|DataLoader|Optimizer|Sampler|"
                         r"LightningModule|Function|Hook|Trainer|Backend|Solver|Model|Operator|Transform|"
                         r"BaseModel|BaseSolver|Base\w+)$")
EXCEPTION_BASES = re.compile(r"(Error|Exception|Warning|Interrupt|Exit)$")
TYPE_BASES = re.compile(r"(^|\.)(Enum|IntEnum|StrEnum|Flag|IntFlag|Protocol|TypedDict|NamedTuple|ABC|Generic)"
                        r"(\[.*\])?$")
ENTRY_MODULES = {"cli", "__main__", "main", "server", "app", "conftest", "routes", "api_server"}
ENTRY_NAMES = {"main", "cli", "serve", "run_server", "create_app"}


def _node(ann: str) -> ast.AST | None:
    try:
        return ast.parse(ann, mode="eval").body
    except SyntaxError:
        return None


def _name(n: ast.AST) -> str:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return n.attr
    if isinstance(n, ast.Constant) and n.value is None:
        return "None"
    return ""


Resolver = Callable[[str], "dict[str, Any] | None"]


def form_type(ann: str, resolve: Resolver | None = None) -> dict[str, Any] | None:
    """The JSON Schema of an annotation, or ``None`` when a form cannot hold it (a tensor, a model, a callable).

    Numeric arrays (``np.ndarray``, ``ArrayLike``) are lists of numbers in a form (``x-convert: numpy``); another class
    is a nested form when ``resolve(name)`` returns its schema (a dataclass of form types, for instance)."""
    if not ann:
        return None
    if ann.startswith(("'", '"')):
        ann = ann.strip("'\"")
    n = _node(ann)
    return None if n is None else _schema(n, resolve)


def _schema(n: ast.AST, resolve: Resolver | None = None) -> dict[str, Any] | None:
    if isinstance(n, ast.Constant) and isinstance(n.value, str):          # forward reference "float"
        inner = _node(n.value)
        return None if inner is None else _schema(inner, resolve)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):          # X | Y
        a, b = _schema(n.left, resolve), _schema(n.right, resolve)
        if a is None or b is None:
            return None
        return {"anyOf": [a, b]}
    name = _name(n)
    if name in ARRAYS:
        return {"type": "array", "items": {"type": "number"}, "x-convert": "numpy"}
    if name in SCALARS:
        t = SCALARS[name]
        if t is None:
            return None
        out = {"type": t}
        if name in ("Path", "PathLike"):
            out["format"] = "path"
        return out
    if isinstance(n, ast.Subscript):
        base = _name(n.value)
        args = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
        if base == "Literal":
            vals = [a.value for a in args if isinstance(a, ast.Constant)]
            return {"enum": vals} if len(vals) == len(args) else None
        if base in ("Optional",):
            s = _schema(args[0], resolve)
            return None if s is None else {"anyOf": [s, {"type": "null"}]}
        if base == "Union":
            parts = [_schema(a, resolve) for a in args]
            return None if any(p is None for p in parts) else {"anyOf": parts}
        if base in ("list", "List", "Sequence", "Iterable", "set", "Set", "frozenset"):
            s = _schema(args[0], resolve)
            return None if s is None else {"type": "array", "items": s}
        if base in ("tuple", "Tuple"):
            items = [a for a in args if not (isinstance(a, ast.Constant) and a.value is Ellipsis)]
            parts = [_schema(a, resolve) for a in items]
            return None if any(p is None for p in parts) else {"type": "array", "items": parts[0] if len(
                {repr(p) for p in parts}) == 1 else {"anyOf": parts}}
        if base in ("dict", "Dict", "Mapping", "MutableMapping"):
            if len(args) == 2 and _name(args[0]) == "str":
                v = _schema(args[1], resolve) if _name(args[1]) != "Any" else {}
                return None if v is None else {"type": "object", "additionalProperties": v}
            return None
        return None
    if name in ("list", "List", "tuple", "Tuple", "Sequence"):
        return {"type": "array"}
    if name in ("dict", "Dict", "Mapping"):
        return {"type": "object"}
    if resolve is not None and name and isinstance(n, (ast.Name, ast.Attribute)):
        return resolve(name)
    return None


def _required(params: list[dict]) -> list[dict]:
    return [p for p in params if p["default"] is None and p["kind"] in ("positional", "keyword")]


def classify(sym: dict[str, Any], module: dict[str, Any], kind: str,
             resolve: Resolver | None = None) -> tuple[str, str]:
    """``(category, reason)`` of a public function (``kind="function"``) or class (``kind="class"``) of ``module``
    (an inventory module record)."""
    mod = module["module"]
    parts = mod.split(".")
    name = sym["name"]
    flags = set(module.get("flags") or [])
    if any(p.startswith("_") and p != "__init__" for p in parts[1:]):
        return "reference", f"internal module ({mod})"
    if parts[-1] in ENTRY_MODULES or name in ENTRY_NAMES:
        return "reference", "entry point (command line, server or app factory), used through its command"
    if "deprecated" in flags:
        return "reference", "the module says it is deprecated or a compatibility shim"
    if "placeholder" in flags and not sym.get("doc"):
        return "reference", "the module marks itself as a placeholder or stub"
    if kind == "class":
        bases = sym.get("bases") or []
        if any(EXCEPTION_BASES.search(b) for b in bases):
            return "reference", f"exception type (base {', '.join(bases)})"
        if any(TYPE_BASES.search(b) for b in bases):
            return "reference", f"type definition (base {', '.join(bases)})"
        block = [b for b in bases if BLOCK_BASES.search(b)]
        if block:
            return "composite", f"building block of a pipeline (base {block[0]})"
        is_dc = any("dataclass" in d for d in sym.get("decorators") or [])
        runs = [m for m in sym.get("methods") or [] if m in RUN_METHODS or m.startswith(("run_", "solve_"))]
        req = _required(sym.get("params") or [])
        bad = [p for p in req if form_type(p["ann"], resolve) is None]
        if not runs:
            if is_dc and not sym.get("methods"):
                return "reference", "data record (a dataclass with no methods): built and read by other tools"
            if bad:
                return "composite", _why_object(bad[0])
            return "composite", "an object other tools use; it has no method that runs a computation"
        if bad:
            return "composite", _why_object(bad[0])
        return "executable", f"built from a form ({len(req)} required inputs) and run with .{runs[0]}()"
    req = _required(sym.get("params") or [])
    bad = [p for p in req if form_type(p["ann"], resolve) is None]
    if bad:
        return "composite", _why_object(bad[0])
    return "executable", ("runs with its defaults" if not req else
                          f"every required input has a form type ({', '.join(p['name'] for p in req)})")


def _why_object(p: dict) -> str:
    if not p["ann"]:
        return f"input `{p['name']}` has no type annotation, so a pipeline supplies it"
    return f"input `{p['name']}: {p['ann']}` is an object built by other tools"


# ------------------------------------------------------------------------------------------------- forms
UNIT = re.compile(r"\[([^\]\s][^\]]{0,24})\]")


def param_doc(doc: str, name: str) -> tuple[str, str]:
    """``(description, unit)`` of a parameter from a docstring (Google, NumPy, reST or ``name: text [unit]``)."""
    if not doc:
        return "", ""
    m = re.search(rf"^\s*:param\s+(?:\w+\s+)?{re.escape(name)}:\s*(.+)$", doc, re.M)
    if not m:
        m = re.search(rf"^\s*(?:[-*]\s*)?(?:``|`)?{re.escape(name)}(?:``|`)?\s*(?:\([^)]*\))?\s*[:\-–]\s+(.+)$",
                      doc, re.M)
    if not m:
        m = re.search(rf"^\s*{re.escape(name)}\s*:\s*[^\n]*\n\s{{2,}}(.+)$", doc, re.M)     # NumPy style
    if not m:
        return "", ""
    text = m.group(1).strip()
    u = UNIT.search(text)
    return text[:300], (u.group(1) if u else "")


def _default(v: str | None) -> Any:
    if v is None:
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return None


def form(sym: dict[str, Any], resolve: Resolver | None = None) -> dict[str, Any]:
    """JSON Schema of a tool's inputs: one property per parameter a form can hold; the others are listed under
    ``x-not-in-form`` (optional objects keep their default)."""
    props, required, skipped = {}, [], []
    doc = sym.get("doc") or ""
    for p in sym.get("params") or []:
        if p["kind"] in ("var_positional", "var_keyword"):
            continue
        s = form_type(p["ann"], resolve)
        if s is None and p["default"] is not None:
            d = _default(p["default"])
            if d is not None or p["default"] == "None":
                s = {"type": {bool: "boolean", int: "integer", float: "number", str: "string"}.get(type(d), "null")}
                if s["type"] == "null" and d is not None:
                    s = None
        if s is None:
            skipped.append({"name": p["name"], "annotation": p["ann"], "default": p["default"]})
            continue
        s = dict(s)
        desc, unit = param_doc(doc, p["name"])
        if desc:
            s["description"] = desc
        if unit:
            s["x-unit"] = unit
        if p["ann"]:
            s["x-annotation"] = p["ann"]
        if p["default"] is None:
            required.append(p["name"])
        else:
            d = _default(p["default"])
            if d is not None or p["default"] == "None":
                s["default"] = d
            else:
                s["x-default-expression"] = p["default"]
        props[p["name"]] = s
    return {"type": "object", "properties": props, "required": required, "x-not-in-form": skipped}


def valid_form(schema: dict[str, Any]) -> list[str]:
    """Problems of a form: a required input missing from it, a default that does not match its type."""
    out = []
    for name in schema.get("required", []):
        if name not in schema["properties"]:
            out.append(f"required input {name} has no field")
    for name, s in schema["properties"].items():
        if "default" in s and s.get("type") in ("integer", "number") and not isinstance(s["default"], (int, float)) \
                and s["default"] is not None:
            out.append(f"default of {name} is not a number")
    return out
