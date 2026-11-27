"""A tolerant parser for OpenFOAM dictionary files (controlDict, fvSchemes, fvSolution, 0/U, ...).

Returns nested Python dicts. Scalars, words and vectors are kept as strings (``"(10 0 0)"``,
``"uniform (0 0 0)"``); helpers turn them into numbers where the caller needs them. Directives
(``#include``, ``#includeEtc``, ``#calc``, code blocks ``#{ ... #}``) are recorded, not evaluated, and
``$macro`` references are kept as text: the parser reads what the files say, it does not run them.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r'''
    (?P<ws>\s+)
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<code>\#\{.*?\#\})
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<punct>[{}();\[\]])
  | (?P<word>[^\s{}();"\[\]]+)
''', re.S | re.X)


def _tokens(text: str) -> list[str]:
    """Tokens; a word glued to an opening parenthesis (``div(phi,U)``, ``"(U|k)"``) stays one token."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _TOKEN.match(text, i)
        if not m:
            i += 1
            continue
        k, tok, i = m.lastgroup, m.group(), m.end()
        if k in ("ws", "lc", "bc"):
            continue
        if k == "code":
            out.append("#codeStream")
            continue
        if k == "word" and i < n and text[i] == "(" and not tok.isdigit():   # "100(" starts a sized list
            depth, j = 0, i
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                elif text[j] in ";{}":
                    break
                j += 1
            if depth == 0 and j < n:
                tok, i = tok + text[i:j + 1], j + 1
        out.append(tok)
    return out


def parse(text: str, max_list: int = 2000) -> dict[str, Any]:
    """Parse a FoamFile. Very long lists (mesh points, nonuniform fields) are summarised as
    ``"<list of N>"`` beyond ``max_list`` entries to keep the result small."""
    toks = _tokens(text)
    pos = 0
    directives: list[str] = []

    def skip_list(i: int) -> tuple[str, int]:
        depth, start = 0, i
        n_items = 0
        while i < len(toks):
            t = toks[i]
            if t == "(":
                depth += 1
            elif t == ")":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1:
                n_items += 1
            i += 1
        span = toks[start:i + 1]
        if len(span) > max_list:
            return f"<list of {n_items}>", i + 1
        return " ".join(span).replace("( ", "(").replace(" )", ")"), i + 1

    def block(i: int, end: str | None) -> tuple[dict[str, Any], int]:
        d: dict[str, Any] = {}
        while i < len(toks):
            t = toks[i]
            if t == end:
                return d, i + 1
            if t in (";",):
                i += 1
                continue
            if t.startswith("#"):
                # directive: rest of the "line" up to the next key is hard to know; take one argument
                arg = toks[i + 1] if i + 1 < len(toks) and toks[i + 1] not in "{};" else ""
                directives.append(f"{t} {arg}".strip())
                i += 2 if arg else 1
                continue
            key = t.strip('"')
            i += 1
            if i < len(toks) and toks[i] == "{":
                sub, i = block(i + 1, "}")
                d[key] = sub
                continue
            vals = []
            while i < len(toks) and toks[i] not in (";", "}"):
                if toks[i] == "(":
                    s, i = skip_list(i)
                    vals.append(s)
                    continue
                if toks[i] == "{":                        # dictionary value inside a list-ish entry
                    sub, i = block(i + 1, "}")
                    vals.append(sub)
                    continue
                vals.append(toks[i])
                i += 1
            if len(vals) == 1 and isinstance(vals[0], dict):
                d[key] = vals[0]
            else:
                d[key] = " ".join(v if isinstance(v, str) else "{...}" for v in vals)
            if i < len(toks) and toks[i] == ";":
                i += 1
        return d, i

    result, _ = block(pos, None)
    if directives:
        result["__directives__"] = directives
    return result


def header(text: str) -> dict[str, str]:
    """The FoamFile header (class, object, format, note) without parsing the body."""
    m = re.search(r"FoamFile\s*\{(.*?)\}", text, re.S)
    if not m:
        return {}
    return {k: v.strip('"') for k, v in re.findall(r"(\w+)\s+([^;]+);", m.group(1))}


def scalar(v: Any) -> float | None:
    """Last number in an entry: 'nu [0 2 -1 0 0 0 0] 1e-05' -> 1e-05; 'uniform 0' -> 0.0."""
    if v is None or isinstance(v, dict):
        return None
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(v).split("]")[-1])
    return float(nums[-1]) if nums else None


def vector(v: Any) -> list[float] | None:
    """'uniform (10 0 0)' -> [10, 0, 0]."""
    if v is None or isinstance(v, dict):
        return None
    m = re.search(r"\(([^()]*)\)", str(v))
    if not m:
        return None
    try:
        return [float(x) for x in m.group(1).split()]
    except ValueError:
        return None


def words(v: Any) -> list[str]:
    return [] if v is None or isinstance(v, dict) else str(v).split()


def boundary(text: str) -> list[dict[str, Any]]:
    """Patches of a polyMesh/boundary file: name, type, nFaces, startFace, inGroups."""
    body = text[text.find("}", text.find("FoamFile")) + 1:] if "FoamFile" in text else text
    out = []
    for name, inner in re.findall(r"([A-Za-z_][\w.:\-]*)\s*\{([^{}]*)\}", _strip_comments(body)):
        e = dict(re.findall(r"(\w+)\s+([^;]+);", inner))
        out.append({"name": name, "type": e.get("type"), "n_faces": int(e["nFaces"]) if "nFaces" in e else None,
                    "groups": re.findall(r"[A-Za-z_]\w*", e.get("inGroups", "").split("(")[-1]) if "inGroups" in e else []})
    return out


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
