"""What a tool returns: ``outputs`` (name, type, unit, description of each value) and where they were read.

Read from the code, never generated:

- ``"declared"``: the module's ``X_OUTPUTS`` dict, written by the author for a tool whose code says too little
  (``X_OUTPUTS = {"step": [{"name": "u", "type": "ndarray", "unit": "m/s", "description": "velocity"}]}``);
- ``"annotation"``: the return annotation; a ``NamedTuple`` or a dataclass gives one output per field;
- ``"docstring"``: the ``Returns`` section, numpy or Google style, with the unit in brackets after the type
  (``tau : float [s]``);
- ``"none"``: nothing says what it returns, and ``outputs`` is empty.

The declaration wins when there is one. A ``NamedTuple`` or dataclass annotation gives the names, and the docstring's
entries with the same names (or the class's own ``Attributes`` section) add units and descriptions. Otherwise
the docstring's entries come next, taking the annotation's types where they state none, and a plain annotation alone
gives one unnamed output. An explicit ``-> None`` gives no outputs, read from the annotation.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

SOURCES = ("declared", "annotation", "docstring", "none")
FIELDS = ("name", "type", "unit", "description")
_SECTION = re.compile(r"^(Returns?|Attributes|Fields)\s*:?\s*$")
_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_UNIT = re.compile(r"(?:^|\s)\[([^\[\],]+)\]")
_GOOGLE = re.compile(r"^(?:(?P<name>[A-Za-z_]\w*)\s*\((?P<ptype>[^)]*)\)|(?P<type>[^:]+?))\s*:\s*(?P<desc>.*)$")
_TUPLE = re.compile(r"^(?:tuple|Tuple|typing\.Tuple)\[(.*)\]$")


def _out(name: str = "", type: str = "", unit: str = "", description: str = "") -> dict[str, str]:
    return {"name": name, "type": type, "unit": unit, "description": description}


def _unit(text: str) -> tuple[str, str]:
    """``("float", "s")`` from ``"float [s]"``. Brackets glued to a name (``list[float]``) are part of the type, and
    brackets holding a comma (an interval, ``[0, 1]``) are not a unit."""
    m = _UNIT.search(text)
    if not m:
        return text.strip(), ""
    return (text[:m.start()] + text[m.end():]).strip(), m.group(1).strip()


def _sections(doc: str) -> dict[str, list[str]]:
    """The lines of each ``Returns`` / ``Attributes`` section, in either style."""
    lines = doc.splitlines()
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        m = _SECTION.match(lines[i].strip())
        numpy = bool(m) and i + 1 < len(lines) and bool(_UNDERLINE.match(lines[i + 1]))
        google = bool(m) and lines[i].strip().endswith(":")
        if not (numpy or google):
            i += 1
            continue
        head = "returns" if m.group(1).startswith("Return") else "attributes"
        indent = len(lines[i]) - len(lines[i].lstrip())
        j = i + (2 if numpy else 1)
        body = []
        while j < len(lines):
            ln = lines[j]
            if numpy:                                   # ends at the next header, or after a blank line, at text
                if j + 1 < len(lines) and _UNDERLINE.match(lines[j + 1]):     # back at the section's indentation
                    break
                if ln.strip() and body and not body[-1].strip() and len(ln) - len(ln.lstrip()) <= indent:
                    break
            elif ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
                break                                   # Google: ends where the indentation comes back
            body.append(ln)
            j += 1
        out.setdefault(head, body)
        i = j
    return out


def _entries(body: list[str]) -> list[dict[str, str]]:
    """Each value of a section: the lines at its first indentation start an entry, deeper lines describe it."""
    rows = [ln for ln in body if ln.strip()]
    if not rows:
        return []
    top = min(len(ln) - len(ln.lstrip()) for ln in rows)
    entries: list[dict[str, str]] = []
    group = 1                                                           # entries the last header line made
    for ln in rows:
        if len(ln) - len(ln.lstrip()) > top and entries:
            for e in entries[-group:]:
                e["description"] = (e["description"] + " " + ln.strip()).strip()
            continue
        group = 1
        text = ln.strip().lstrip("-*• ").replace("``", "")
        if " : " in text or text.endswith(" :"):                       # numpy: "name : type [unit]"
            name, _, typ = text.partition(":")
            typ, unit = _unit(typ)
            names = [n.strip() for n in name.split(",")] if re.fullmatch(r"\s*\w+(\s*,\s*\w+)+\s*", name) else [name.strip()]
            entries.extend(_out(n, typ, unit) for n in names)          # "x, y : float [m]": one entry per name
            group = len(names)
            continue
        g = _GOOGLE.match(text)
        if g:                                                           # Google: "name (type) [unit]: text"
            typ = (g.group("ptype") or g.group("type") or "").strip()
            typ, unit = _unit(typ)
            entries.append(_out(g.group("name") or "", typ, unit, g.group("desc").strip()))
            continue
        typ, unit = _unit(text)                                         # numpy without a name: "float [s]"
        entries.append(_out("", typ, unit))
    return entries


def from_docstring(doc: str, section: str = "returns") -> list[dict[str, str]]:
    """The entries of the docstring's ``Returns`` (or ``Attributes``) section; empty without one."""
    return _entries(_sections(doc or "").get(section, []))


def _strip_ann(ann: str) -> str:
    ann = ann.strip().strip("'\"")
    for wrap in ("Optional[", "typing.Optional["):
        if ann.startswith(wrap) and ann.endswith("]"):
            ann = ann[len(wrap):-1]
    return ann.replace(" | None", "").replace("None | ", "")


def _split_args(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def is_record_class(cls: dict[str, Any]) -> bool:
    """A ``NamedTuple`` or a dataclass: its fields are the outputs."""
    return any(b.split(".")[-1] == "NamedTuple" for b in cls.get("bases") or []) or any(
        "dataclass" in d for d in cls.get("decorators") or [])


def outputs(sym: dict[str, Any], declared: dict[str, Any] | None,
            find_class: Callable[[str], dict[str, Any] | None]) -> tuple[list[dict[str, str]], str]:
    """``(outputs, outputs_source)`` of a function record (``returns`` annotation, ``doc``) of the inventory.

    ``declared`` is the module's ``X_OUTPUTS``; ``find_class(name)`` returns the inventory record of a class the
    annotation names, or ``None``."""
    if declared and isinstance(declared.get(sym["name"]), list):
        rows = [_out(**{k: str(e.get(k, "")) for k in FIELDS}) for e in declared[sym["name"]] if isinstance(e, dict)]
        return rows, "declared"
    doc = from_docstring(sym.get("doc", ""))
    ann = _strip_ann(sym.get("returns") or "")
    if ann in ("None", "NoReturn"):
        return [], "annotation"
    cls = find_class(ann.split("[")[0].split(".")[-1]) if ann and re.match(r"^[\w.]+$", ann) else None
    if cls and is_record_class(cls):
        notes = {e["name"]: e for e in doc + from_docstring(cls.get("doc", ""), "attributes") if e["name"]}
        rows = []
        for p in cls.get("params") or []:
            n = notes.get(p["name"], {})
            rows.append(_out(p["name"], p.get("ann", ""), n.get("unit", ""), n.get("description", "")))
        if rows:
            return rows, "annotation"
    m = _TUPLE.match(ann)
    types = [t for t in _split_args(m.group(1)) if t != "..."] if m else ([ann] if ann else [])
    if doc:
        if len(types) == len(doc):
            for e, t in zip(doc, types, strict=True):
                e["type"] = e["type"] or t
        return doc, "docstring"
    if types:
        return [_out("", t) for t in types], "annotation"
    return [], "none"


def declared_outputs(tree: ast.Module) -> dict[str, Any] | None:
    """The literal ``X_OUTPUTS = {...}`` at the top level of a module, if any."""
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "X_OUTPUTS" for t in targets) and node.value is not None:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
                return value if isinstance(value, dict) else None
    return None
