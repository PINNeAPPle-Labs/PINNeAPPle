"""Stage C: static and structural validation of generated artifacts.

What this module actually is: an AST walk plus real introspection of the
PINNeAPPle modules. What it is not: a sandbox, a type checker, or proof
that the code does what it claims. The docstring says so because
"validated statically" is exactly the kind of phrase that gets read as
"safe and correct" when it means "we parsed it and looked things up".

Three checks are worth naming precisely, because they are the ones that
catch the failure modes ordinary syntax checks miss:

``static_api_contracts``
    Every ``pinneapple_*`` attribute the artifact touches is resolved
    against the *installed* module with :func:`getattr`. A call to
    ``pinneapple_twin3d.primitives.torus(...)`` -- a function that does
    not exist -- fails here, before execution, with the closest real
    names suggested.

``static_signatures``
    Where a called function could be resolved, its :func:`inspect.signature`
    is compared with the call node: too many positional arguments, or a
    keyword the function does not accept, is reported. This is still
    static -- ``*args``/``**kwargs`` and dynamically built calls are out
    of reach, and a passing signature says nothing about argument
    *semantics*.

``static_security``
    Imports and calls that escape the artifact's intended scope are
    refused: process spawning, raw sockets, ``eval``/``exec``,
    ``ctypes``/``marshal``/``pickle`` loads, filesystem deletion, and
    writes outside the artifact's own directory. This is a policy check
    on source text, and a determined author can evade it; it is not a
    security boundary. See :mod:`pinneapple_llm.qa.sandbox` for what the
    execution path actually enforces and where its limits are.
"""
from __future__ import annotations

import ast
import difflib
import importlib
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from .report import Finding, Status

#: Modules an artifact may import. Anything else is reported; the list is a
#: policy, not an impossibility proof.
DEFAULT_ALLOWED_ROOTS: FrozenSet[str] = frozenset({
    "pinneapple_adaptation", "pinneapple_analysis", "pinneapple_app", "pinneapple_arena",
    "pinneapple_blender", "pinneapple_catalog", "pinneapple_core", "pinneapple_data",
    "pinneapple_decision", "pinneapple_design", "pinneapple_hub", "pinneapple_lab",
    "pinneapple_llm", "pinneapple_models", "pinneapple_neural", "pinneapple_orchestration",
    "pinneapple_pdb", "pinneapple_perception", "pinneapple_physics", "pinneapple_problemdesign",
    "pinneapple_quantum", "pinneapple_registry", "pinneapple_security", "pinneapple_simulation",
    "pinneapple_solvers", "pinneapple_systems", "pinneapple_tools", "pinneapple_train",
    "pinneapple_twin3d", "pinneapple_veriphysics", "pinneapple_worldmodel",
    # standard library that cannot by itself spawn a process, open a socket,
    # or touch the filesystem destructively
    "bisect", "collections", "copy", "csv", "dataclasses", "datetime", "enum", "functools",
    "hashlib", "itertools", "json", "math", "numbers", "operator", "pathlib", "random",
    "re", "statistics", "string", "textwrap", "time", "typing", "uuid", "warnings",
    "numpy", "scipy", "pandas",
})

#: Modules refused outright, even though some are stdlib.
FORBIDDEN_ROOTS: FrozenSet[str] = frozenset({
    "subprocess", "socket", "ssl", "ctypes", "cffi", "multiprocessing", "threading",
    "http", "urllib", "urllib3", "requests", "ftplib", "smtplib", "telnetlib",
    "pickle", "marshal", "shelve", "dbm", "sqlite3", "pty", "tty", "resource",
    "shutil", "tempfile", "importlib", "builtins", "gc", "sys", "os", "signal", "mmap",
    "webbrowser", "antigravity", "pip", "setuptools", "distutils",
})

#: Dotted call targets refused regardless of how they were reached.
FORBIDDEN_CALLS: FrozenSet[str] = frozenset({
    "eval", "exec", "compile", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "__import__", "input", "breakpoint", "memoryview",
    "open",  # replaced by the narrower path check below for literal paths
})

#: Attribute names dangerous enough to refuse on sight. Deliberately short:
#: common methods (``list.remove``, ``str.replace``, ``json.loads``) are NOT
#: here, because flagging them would train users to ignore this check.
#: ``os.remove``-class calls are already unreachable -- ``os`` is a forbidden
#: import -- so only names with no benign common meaning are listed.
FORBIDDEN_ATTR_CALLS: FrozenSet[str] = frozenset({
    "system", "popen", "spawn", "spawnl", "spawnv", "execv", "execve", "fork",
    "urlopen", "rmtree", "chmod", "chown", "kill", "killpg",
})


@dataclass
class StaticPolicy:
    allowed_roots: FrozenSet[str] = DEFAULT_ALLOWED_ROOTS
    forbidden_roots: FrozenSet[str] = FORBIDDEN_ROOTS
    forbidden_calls: FrozenSet[str] = FORBIDDEN_CALLS
    forbidden_attr_calls: FrozenSet[str] = FORBIDDEN_ATTR_CALLS
    #: Root directories an artifact may write under. Empty means "the
    #: directory the artifact itself lives in", supplied by the caller.
    allowed_write_roots: Tuple[str, ...] = ()
    max_source_bytes: int = 400_000
    resolve_api: bool = True


@dataclass
class StaticResult:
    findings: List[Finding] = field(default_factory=list)
    resolved_imports: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(f.blocks for f in self.findings)

    @property
    def status(self) -> Status:
        return Status.PASS if self.ok else Status.FAIL

    def step_findings(self, prefix: str) -> List[Finding]:
        return [f for f in self.findings if f.id.startswith(prefix)]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_syntax(source: str, filename: str = "<artifact>") -> List[Finding]:
    """Compile-only check. A ``SyntaxError`` is reported, never raised."""
    try:
        compile(source, filename, "exec")
    except SyntaxError as exc:
        return [Finding(
            id="static.syntax", status=Status.FAIL, severity="error",
            message=f"SyntaxError at line {exc.lineno}: {exc.msg}",
            evidence={"line": exc.lineno, "offset": exc.offset, "text": (exc.text or "").strip()},
            justification="the artifact cannot be imported or executed at all",
        )]
    return [Finding(id="static.syntax", status=Status.PASS, message="source compiles")]


def _import_roots(node: ast.AST) -> List[Tuple[str, str, int]]:
    """``(root, full_module, lineno)`` for every import in the tree."""
    out: List[Tuple[str, str, int]] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                out.append((alias.name.split(".")[0], alias.name, n.lineno))
        elif isinstance(n, ast.ImportFrom):
            if n.level and not n.module:
                out.append(("<relative>", ".", n.lineno))
                continue
            module = n.module or ""
            out.append((module.split(".")[0], module, n.lineno))
    return out


def check_imports(source: str, policy: StaticPolicy) -> Tuple[List[Finding], Dict[str, Any]]:
    """Refuse forbidden imports; report unknown/unresolvable ones with evidence."""
    findings: List[Finding] = []
    resolved: Dict[str, Any] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings, resolved  # syntax finding already reported by check_syntax

    for root, module, lineno in _import_roots(tree):
        if root in policy.forbidden_roots:
            findings.append(Finding(
                id="static.import", status=Status.FAIL, severity="error",
                message=f"forbidden import '{module}' (line {lineno})",
                evidence={"module": module, "line": lineno, "root": root},
                justification="this module can spawn processes, open sockets, mutate the filesystem "
                            "or load arbitrary code; generated artifacts may not use it",
            ))
            continue
        if root not in policy.allowed_roots and not root.startswith("pinneapple_"):
            findings.append(Finding(
                id="static.import", status=Status.FAIL, severity="error",
                message=f"import '{module}' (line {lineno}) is outside the artifact allowlist",
                evidence={"module": module, "line": lineno},
                justification="an unlisted dependency is either unavailable in the target environment "
                            "or a deliberate escape from the declared scope",
            ))
            continue
        if policy.resolve_api:
            try:
                module_obj = importlib.import_module(module)
            except Exception as exc:  # ImportError, but also a broken optional dep
                findings.append(Finding(
                    id="static.import", status=Status.FAIL, severity="error",
                    message=f"import '{module}' (line {lineno}) does not resolve: {type(exc).__name__}: {exc}",
                    evidence={"module": module, "line": lineno, "error": str(exc)},
                    justification="the artifact would fail at import time in this environment",
                ))
            else:
                resolved[module] = module_obj
    # names imported FROM a resolvable module must exist on it
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module_obj = resolved.get(node.module)
        if module_obj is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if not hasattr(module_obj, alias.name):
                available = sorted(a for a in dir(module_obj) if not a.startswith("_"))
                close = difflib.get_close_matches(alias.name, available, n=3)
                findings.append(Finding(
                    id="static.import", status=Status.FAIL, severity="error",
                    message=f"'{node.module}' has no attribute '{alias.name}' (line {node.lineno})"
                            + (f"; did you mean {close}?" if close else ""),
                    evidence={"module": node.module, "missing": alias.name,
                              "line": node.lineno, "suggestions": close},
                    justification="the artifact imports a name this PINNeAPPle version does not expose",
                ))
    if not any(f.blocks for f in findings):
        findings.append(Finding(id="static.import", status=Status.PASS,
                                message=f"{len({m for _, m, _ in _import_roots(tree)})} import(s) allowed and resolvable"))
    return findings, resolved


@dataclass
class _Binding:
    obj: Any
    label: str
    lineno: int


def _collect_bindings(tree: ast.AST, resolved: Dict[str, Any]) -> Dict[str, _Binding]:
    """Map local names to the objects they imported, for API resolution."""
    bindings: Dict[str, _Binding] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                obj = resolved.get(alias.name)
                if obj is not None:
                    bindings[alias.asname or alias.name.split(".")[0]] = _Binding(
                        obj, alias.name, n.lineno)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mod = resolved.get(n.module)
            if mod is None:
                continue
            for alias in n.names:
                if alias.name == "*":
                    continue
                obj = getattr(mod, alias.name, _MISSING)
                if obj is not _MISSING:
                    bindings[alias.asname or alias.name] = _Binding(obj, f"{n.module}.{alias.name}", n.lineno)
    return bindings


_MISSING = object()


def _resolve_chain(node: ast.AST, bindings: Dict[str, _Binding]) -> Tuple[Any, str]:
    """Resolve ``a.b.c`` against the import bindings. Returns ``(obj, label)``."""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return _MISSING, ""
    binding = bindings.get(cur.id)
    if binding is None:
        return _MISSING, ""
    obj = binding.obj
    label = binding.label
    for attr in reversed(parts):
        nxt = getattr(obj, attr, _MISSING)
        candidate = f"{label}.{attr}"
        if nxt is _MISSING:
            # A package submodule is only an attribute of its parent after
            # something has imported it; try the import so a legitimate
            # ``pinneapple_twin3d.primitives.x`` is not a false positive.
            try:
                nxt = importlib.import_module(candidate)
            except Exception:
                return _MISSING, candidate
        obj = nxt
        label = candidate
    return obj, label


def check_api_contracts(source: str, resolved: Dict[str, Any]) -> List[Finding]:
    """Resolve every ``pinneapple_*`` attribute the artifact touches, for real.

    Uses the installed modules (``importlib`` + ``getattr``), not a regex
    or a hard-coded name list, so an invented API is caught even when it
    is syntactically valid Python.
    """
    findings: List[Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    bindings = _collect_bindings(tree, resolved)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        obj, label = _resolve_chain(node, bindings)
        if obj is not _MISSING or not label.startswith("pinneapple_"):
            continue
        # find the first missing segment and suggest real alternatives
        segments = label.split(".")
        container = resolved.get(segments[0])
        walked = segments[0]
        for seg in segments[1:]:
            if container is None:
                break
            nxt = getattr(container, seg, _MISSING)
            walked = f"{walked}.{seg}"
            if nxt is _MISSING:
                try:
                    nxt = importlib.import_module(walked)
                except Exception:
                    pass
            if nxt is _MISSING:
                available = sorted(a for a in dir(container) if not a.startswith("_"))
                close = difflib.get_close_matches(seg, available, n=3)
                findings.append(Finding(
                    id="static.api_contracts", status=Status.FAIL, severity="error",
                    message=f"'{walked}' does not exist (line {node.lineno})",
                    evidence={"missing": walked, "line": node.lineno, "suggestions": close,
                              "available_count": len(available)},
                    justification="the artifact calls an API this PINNeAPPle version does not have; "
                                "it would fail at run time even though the source parses",
                ))
                break
            container = nxt
        checked += 1
    if not any(f.blocks for f in findings):
        findings.append(Finding(
            id="static.api_contracts", status=Status.PASS,
            message=f"{checked} pinneapple_* attribute access(es) resolved against installed modules",
        ))
    return findings


def check_signatures(source: str, resolved: Dict[str, Any]) -> List[Finding]:
    """Compare literal call sites against the real ``inspect.signature``."""
    findings: List[Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    bindings = _collect_bindings(tree, resolved)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func, label = _resolve_chain(node.func, bindings)
        if func is _MISSING or not callable(func) or not label.startswith("pinneapple_"):
            continue
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue
        checked += 1
        try:
            sig.bind(*([None] * len(node.args)), **{kw.arg: None for kw in node.keywords if kw.arg})
        except TypeError as exc:
            findings.append(Finding(
                id="static.signatures", status=Status.FAIL, severity="error",
                message=f"call to '{label}' does not match its signature (line {node.lineno}): {exc}",
                evidence={"function": label, "line": node.lineno, "signature": str(sig),
                          "positional": len(node.args),
                          "keywords": [kw.arg for kw in node.keywords]},
                justification="the call would raise TypeError before doing any work",
            ))
    if not any(f.blocks for f in findings):
        findings.append(Finding(id="static.signatures", status=Status.PASS,
                                message=f"{checked} call signature(s) checked"))
    return findings


def check_security(source: str, policy: StaticPolicy, write_roots: Iterable[str] = ()) -> List[Finding]:
    """Refuse process spawning, code execution, network and out-of-scope writes."""
    findings: List[Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    roots = [os.path.abspath(r) for r in (write_roots or policy.allowed_write_roots)]

    def _literal_targets(node: ast.Call) -> List[str]:
        out = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                out.append("".join(v.value for v in arg.values if isinstance(v, ast.Constant)))
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            leaf = name.rsplit(".", 1)[-1] if name else ""
            if name in policy.forbidden_calls or leaf in policy.forbidden_calls:
                if leaf == "open":
                    targets = _literal_targets(node)
                    if not targets:
                        findings.append(Finding(
                            id="static.security", status=Status.NOT_RUN, severity="warning",
                            message=f"open() with a non-literal path cannot be checked statically (line {node.lineno})",
                            evidence={"line": node.lineno},
                            justification="path confinement for computed paths is only enforceable at "
                                        "run time, by the executor's working directory",
                        ))
                        continue
                    outside = [t for t in targets if _escapes(t, roots)]
                    if outside:
                        findings.append(Finding(
                            id="static.security", status=Status.FAIL, severity="error",
                            message=f"open() outside the artifact directory (line {node.lineno}): {outside[0]}",
                            evidence={"path": outside[0], "line": node.lineno, "allowed_roots": roots},
                            justification="generated code may only write inside its own output directory",
                        ))
                    continue
                findings.append(Finding(
                    id="static.security", status=Status.FAIL, severity="error",
                    message=f"forbidden call '{name or leaf}' (line {node.lineno})",
                    evidence={"call": name or leaf, "line": node.lineno},
                    justification="arbitrary code execution or dynamic dispatch is outside the artifact's remit",
                ))
            elif leaf in policy.forbidden_attr_calls:
                findings.append(Finding(
                    id="static.security", status=Status.FAIL, severity="error",
                    message=f"forbidden operation '{name or leaf}' (line {node.lineno})",
                    evidence={"call": name or leaf, "line": node.lineno},
                    justification="process spawning, network access, file deletion and deserialisation "
                                "of untrusted bytes are outside the artifact's remit",
                ))
    if not any(f.blocks for f in findings):
        findings.append(Finding(id="static.security", status=Status.PASS,
                                message="no process, network, dynamic-exec or out-of-scope write detected"))
    return findings


def _escapes(path: str, roots: List[str]) -> bool:
    """True if a literal path is absolute or climbs out of every allowed root."""
    if not path:
        return False
    if os.path.isabs(path) or path.startswith("~"):
        return True
    norm = os.path.normpath(path)
    if norm.startswith(".."):
        return True
    if roots:
        # relative paths are resolved against each allowed root; escape if all
        # of them land outside their own root
        return not any(
            os.path.abspath(os.path.join(r, norm)).startswith(os.path.abspath(r)) for r in roots
        )
    return False


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# ---------------------------------------------------------------------------
# JSON / schema artifacts
# ---------------------------------------------------------------------------
def check_json_artifact(data: Any, schema: Optional[Dict[str, Any]] = None,
                        name: str = "artifact") -> List[Finding]:
    """Validate a JSON artifact against a small, explicit schema subset.

    Supported keywords: ``type`` (``object``/``array``/``number``/``integer``/
    ``string``/``boolean``), ``properties``, ``required``, ``items``,
    ``enum``, ``minimum``, ``maximum``, ``minItems``, ``maxItems``. Anything
    else in the schema is ignored *and reported* -- a schema keyword this
    validator does not implement must not quietly become a no-op.
    """
    findings: List[Finding] = []
    unknown = sorted(set(schema or {}) - {"type", "properties", "required", "items", "enum",
                                          "minimum", "maximum", "minItems", "maxItems",
                                          "title", "description"})
    if unknown:
        findings.append(Finding(
            id="static.schema", status=Status.NOT_APPLICABLE, severity="warning",
            message=f"schema keywords not enforced by this validator: {unknown}",
            evidence={"keywords": unknown},
            justification="reporting the gap rather than pretending the schema was fully applied",
        ))

    errors: List[str] = []

    def walk(value: Any, sch: Dict[str, Any], path: str) -> None:
        if not sch:
            return
        expected = sch.get("type")
        if expected and not _type_ok(value, expected):
            errors.append(f"{path or '<root>'}: expected {expected}, got {type(value).__name__}")
            return
        if "enum" in sch and value not in sch["enum"]:
            errors.append(f"{path}: {value!r} not in {sch['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in sch and value < sch["minimum"]:
                errors.append(f"{path}: {value} < minimum {sch['minimum']}")
            if "maximum" in sch and value > sch["maximum"]:
                errors.append(f"{path}: {value} > maximum {sch['maximum']}")
        if isinstance(value, dict):
            for req in sch.get("required", []):
                if req not in value:
                    errors.append(f"{path}: missing required key '{req}'")
            for key, sub in (sch.get("properties") or {}).items():
                if key in value:
                    walk(value[key], sub, f"{path}.{key}" if path else key)
        if isinstance(value, list):
            if "minItems" in sch and len(value) < sch["minItems"]:
                errors.append(f"{path}: {len(value)} items < minItems {sch['minItems']}")
            if "maxItems" in sch and len(value) > sch["maxItems"]:
                errors.append(f"{path}: {len(value)} items > maxItems {sch['maxItems']}")
            if isinstance(sch.get("items"), dict):
                for i, item in enumerate(value):
                    walk(item, sch["items"], f"{path}[{i}]")

    walk(data, schema or {}, "")
    if errors:
        findings.append(Finding(
            id="static.schema", status=Status.FAIL, severity="error",
            message=f"{len(errors)} schema violation(s) in {name}",
            evidence={"errors": errors[:20], "total": len(errors)},
            justification="the artifact does not conform to the contract its consumers expect",
        ))
    else:
        findings.append(Finding(id="static.schema", status=Status.PASS,
                                message=f"{name} conforms to the schema"))
    return findings


def _type_ok(value: Any, expected: str) -> bool:
    table = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    return table.get(expected, lambda v: True)(value)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def validate_python_artifact(source: str, *, policy: Optional[StaticPolicy] = None,
                             filename: str = "<artifact>",
                             write_roots: Iterable[str] = (),
                             json_schema: Optional[Dict[str, Any]] = None,
                             ) -> StaticResult:
    """Run every static check and return one :class:`StaticResult`.

    ``json_schema`` is only meaningful when the source embeds a JSON
    literal; it is provided for artifact types whose contract is a JSON
    payload (e.g. a twin scene manifest) -- see
    :func:`validate_json_artifact`.
    """
    policy = policy or StaticPolicy()
    result = StaticResult()
    result.findings.extend(check_syntax(source, filename))
    if any(f.blocks for f in result.findings):
        return result  # nothing else is meaningful on unparsable source
    import_findings, resolved = check_imports(source, policy)
    result.findings.extend(import_findings)
    result.resolved_imports = resolved
    result.findings.extend(check_security(source, policy, write_roots))
    if policy.resolve_api:
        result.findings.extend(check_api_contracts(source, resolved))
        result.findings.extend(check_signatures(source, resolved))
    return result


def validate_json_artifact(payload: Any, schema: Optional[Dict[str, Any]] = None,
                           name: str = "artifact") -> StaticResult:
    result = StaticResult()
    result.findings.extend(check_json_artifact(payload, schema, name))
    return result


def load_json_file(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
