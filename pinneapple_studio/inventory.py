"""Static inventory of a PINNeAPPle checkout: what the library has, read from the code by ``ast`` alone.

Nothing is imported and nothing is run. :func:`build` returns a dict with

* every ``pinneapple*`` package and module: docstring, public classes and functions with their signatures and the
  first line of their docstrings, public methods, ``__all__``, lines of code, experimental / placeholder flags;
* the top-level ``pp.*`` names (imports of ``pinneapple/__init__.py`` and ``_LAZY_ATTRS``);
* tests: ``test_*`` functions per file and, per module, the tests in files that import it;
* ``pinneapple_catalog/method_status.json`` with each method's name (read from ``methods.py``);
* the extras of ``pyproject.toml``; ``apps/``, ``examples/``, ``scripts/``, ``benchmarks/``, ``docs/``,
  ``.github/workflows/``.

:func:`load` caches it per commit under :func:`cache_dir` (never committed). The Portuguese map of the library
(``docs/pt/_mapa/inventario.py``) and the PINNeAPPle Studio both read it from here.

    python -m pinneapple_studio inventory -o inventory.json
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import tomllib

SKIP_DIRS = {".git", "__pycache__", "node_modules", "vendor", ".ipynb_checkpoints", "_out", "_runs",
             "site-packages", ".pytest_cache", "build", "dist"}
# Third-party names never reach the inventory or the Studio (see exclude_pattern).
def exclude_pattern(root: Path | None = None) -> re.Pattern:
    """Names that must never reach the inventory or the Studio (third-party data). The pattern is not kept in the
    repository: ``PINNEAPPLE_MAPA_EXCLUDE`` (a regular expression), else ``docs/pt/_mapa/exclude.local.txt``
    (git-ignored; one expression per line, ``#`` comments). With neither, nothing is excluded."""
    env = os.environ.get("PINNEAPPLE_MAPA_EXCLUDE", "").strip()
    parts = [env] if env else []
    local = Path(root or Path(__file__).resolve().parents[1]) / "docs" / "pt" / "_mapa" / "exclude.local.txt"
    if not parts and local.is_file():
        parts = [ln.strip() for ln in local.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
    return re.compile("|".join(f"(?:{p})" for p in parts), re.I) if parts else re.compile(r"(?!x)x")


EXCLUDE_RE = exclude_pattern()
FLAG_RE = {
    "experimental": re.compile(r"\bexperimental\b", re.I),
    "placeholder": re.compile(r"\bplaceholder\b|\bstub\b|not (yet )?implemented|\bTODO\b", re.I),
    "deprecated": re.compile(r"\bdeprecated\b|\bcompatibility shim\b|\bshim\b", re.I),
}


def clean(text: str | None, limit: int | None = None) -> str:
    """Remove linhas com termos de clientes e corta o texto."""
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if not EXCLUDE_RE.search(ln)]
    out = "\n".join(lines).strip()
    if limit and len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + " …"
    return out


def first_line(doc: str | None) -> str:
    if not doc:
        return ""
    para = []
    for ln in doc.strip().splitlines():
        if not ln.strip():
            break
        para.append(ln.strip())
    return clean(" ".join(para), 300)


def walk_py(base: Path, root: Path):
    for p in sorted(base.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if EXCLUDE_RE.search(str(rel)):
            continue
        yield p


def parse(path: Path):
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        return src, ast.parse(src)
    except (SyntaxError, ValueError):
        return None, None


def unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool = False) -> str:
    a = fn.args
    parts = []
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for i, (arg, d) in enumerate(zip(pos, defaults, strict=True)):
        if drop_self and i == 0 and arg.arg in ("self", "cls"):
            continue
        s = arg.arg
        if d is not None:
            dv = unparse(d)
            s += "=" + (dv if len(dv) <= 30 else "…")
        parts.append(s)
        if a.posonlyargs and i == len(a.posonlyargs) - 1:
            parts.append("/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        s = arg.arg
        if d is not None:
            dv = unparse(d)
            s += "=" + (dv if len(dv) <= 30 else "…")
        parts.append(s)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = f" -> {unparse(fn.returns)}" if fn.returns is not None else ""
    if len(ret) > 50:
        ret = ""
    return f"({', '.join(parts)}){ret}"


def params(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_self: bool = False) -> list[dict]:
    """Each parameter: name, annotation (as written), default (as written, ``None`` when required), kind."""
    a = fn.args
    out = []
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for i, (arg, d) in enumerate(zip(pos, defaults, strict=True)):
        if drop_self and i == 0 and arg.arg in ("self", "cls"):
            continue
        out.append({"name": arg.arg, "ann": unparse(arg.annotation) if arg.annotation else "",
                    "default": None if d is None else unparse(d)[:80], "kind": "positional"})
    if a.vararg:
        out.append({"name": a.vararg.arg, "ann": "", "default": "()", "kind": "var_positional"})
    for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append({"name": arg.arg, "ann": unparse(arg.annotation) if arg.annotation else "",
                    "default": None if d is None else unparse(d)[:80], "kind": "keyword"})
    if a.kwarg:
        out.append({"name": a.kwarg.arg, "ann": "", "default": "{}", "kind": "var_keyword"})
    return out


def dataclass_params(node: ast.ClassDef) -> list[dict]:
    return [{"name": unparse(b.target), "ann": unparse(b.annotation),
             "default": None if b.value is None else unparse(b.value)[:80], "kind": "keyword"}
            for b in node.body if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)
            and "ClassVar" not in unparse(b.annotation)]


def declared_ranges(tree: ast.Module) -> dict[str, dict[str, list]]:
    """``fn.x_ranges = {"k": (lo, hi), ...}`` written as a literal at the top level of a module: the input box the
    author declared for ``fn``. ``None`` leaves a side open; entries that are not a pair of numbers are dropped."""
    out: dict[str, dict[str, list]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        t = node.targets[0]
        if not (isinstance(t, ast.Attribute) and t.attr == "x_ranges" and isinstance(t.value, ast.Name)):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, dict):
            continue
        box = {k: [b[0], b[1]] for k, b in value.items()
               if isinstance(k, str) and isinstance(b, (list, tuple)) and len(b) == 2
               and all(x is None or (isinstance(x, (int, float)) and not isinstance(x, bool)) for x in b)}
        if box:
            out[t.value.id] = box
    return out


def is_public(name: str) -> bool:
    return not name.startswith("_")


def module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_record(path: Path, root: Path) -> dict:
    src, tree = parse(path)
    rec = {"module": module_name(path, root), "path": str(path.relative_to(root)),
           "lines": 0, "doc": "", "doc1": "", "classes": [], "functions": [], "all": None,
           "flags": [], "imports": [], "parse_error": tree is None}
    if tree is None:
        return rec
    rec["lines"] = src.count("\n") + 1
    doc = ast.get_docstring(tree)
    rec["doc"] = clean(doc, 1500)
    rec["doc1"] = first_line(doc)
    not_impl = False
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None and "NotImplementedError" in unparse(node.exc):
            not_impl = True
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                and node.module.split(".")[0].startswith("pinneapple"):
            imports |= {f"{node.module}:{a.name}" for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            base = rec["module"].split(".")
            if not path.name == "__init__.py":
                base = base[:-1]
            base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
            mod = ".".join(base + ([node.module] if node.module else []))
            imports |= {f"{mod}:{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imports |= {a.name for a in node.names if a.name.split(".")[0].startswith("pinneapple")}
    rec["imports"] = sorted(imports)
    rec["x_ranges"] = declared_ranges(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public(node.name):
            fdoc = ast.get_docstring(node)
            rec["functions"].append({"name": node.name, "sig": signature(node), "doc1": first_line(fdoc),
                                     "line": node.lineno, "end": node.end_lineno, "doc": clean(fdoc, 4000),
                                     "params": params(node), "returns": unparse(node.returns) if node.returns else "",
                                     "decorators": [unparse(d) for d in node.decorator_list],
                                     "async": isinstance(node, ast.AsyncFunctionDef)})
        elif isinstance(node, ast.ClassDef) and is_public(node.name):
            cdoc = ast.get_docstring(node)
            init = None
            init_params = None
            methods = []
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if b.name == "__init__":
                        init = signature(b, drop_self=True)
                        init_params = params(b, drop_self=True)
                    elif is_public(b.name):
                        methods.append(b.name)
            if init is None:  # dataclass: campos anotados
                fields = [unparse(b.target) for b in node.body
                          if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)
                          and "ClassVar" not in unparse(b.annotation)]
                init = f"({', '.join(fields)})" if fields else "()"
            rec["classes"].append({"name": node.name, "bases": [unparse(b) for b in node.bases],
                                   "sig": init if len(init) <= 160 else init[:157] + "…)",
                                   "doc1": first_line(cdoc), "methods": methods, "line": node.lineno,
                                   "end": node.end_lineno, "doc": clean(cdoc, 4000),
                                   "params": init_params if init_params is not None else dataclass_params(node),
                                   "has_init": init_params is not None,
                                   "decorators": [unparse(d) for d in node.decorator_list]})
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    rec["all"] = [unparse(e).strip("'\"") for e in node.value.elts]
    for f in rec["functions"]:
        if len(f["sig"]) > 160:
            f["sig"] = f["sig"][:157] + "…)"
    text = (doc or "") + " " + " ".join(c["doc1"] for c in rec["classes"]) + " " + \
        " ".join(f["doc1"] for f in rec["functions"])
    for flag, rx in FLAG_RE.items():
        if rx.search(text):
            rec["flags"].append(flag)
    if not_impl:
        rec["flags"].append("raises_NotImplementedError")
    return rec


def imported_modules(tree, known: set[str], current: str) -> set[str]:
    """Módulos ``pinneapple*`` conhecidos importados por um arquivo de teste."""
    out = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Call) and unparse(node.func) in ("importlib.import_module", "import_module",
                                                                  "pytest.importorskip"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names = [node.args[0].value]
        for n in names:
            if n in known:
                out.add(n)
    return out


def count_tests(tree) -> list[str]:
    tests = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            tests.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name.startswith("test"):
                    tests.append(f"{node.name}::{b.name}")
    return tests


def top_api(root: Path) -> dict:
    """Nomes de ``pp.*``: imports diretos, funções de topo e ``_LAZY_ATTRS``."""
    src, tree = parse(root / "pinneapple" / "__init__.py")
    eager, lazy, subs, defs = {}, {}, {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and \
                node.module.startswith("pinneapple"):
            for a in node.names:
                eager[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_LAZY_ATTRS"
                                                  for t in node.targets) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                try:
                    key = ast.literal_eval(k)
                    val = ast.literal_eval(v)
                    lazy[key] = ".".join(x for x in val if x) if isinstance(val, (tuple, list)) else str(val)
                except Exception:
                    pass
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_SUBMODULES"
                                                  for t in node.targets) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                try:
                    subs[ast.literal_eval(k)] = ast.literal_eval(v)
                except Exception:
                    pass
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and is_public(node.name):
            defs.append(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.ClassDef)) and is_public(b.name):
                    defs.append(b.name)
    return {"eager": dict(sorted(eager.items())), "lazy": dict(sorted(lazy.items())),
            "submodules": dict(sorted(subs.items())),
            "defined_here": sorted(set(defs)),
            "version": re.search(r'__version__\s*=\s*"([^"]+)"', src).group(1)}


def method_names(root: Path) -> dict:
    src, tree = parse(root / "pinneapple_catalog" / "methods.py")
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and unparse(node.func) == "_m" and len(node.args) >= 3:
            try:
                mid = ast.literal_eval(node.args[0])
                name = ast.literal_eval(node.args[1])
            except Exception:
                continue
            code = []
            if isinstance(node.args[2], (ast.List, ast.Tuple)):
                for e in node.args[2].elts:
                    if isinstance(e, ast.Constant):
                        code.append(e.value)
                    elif isinstance(e, ast.BinOp) and isinstance(e.left, ast.Name) and isinstance(e.right, ast.Constant):
                        code.append(str(consts.get(e.left.id, "")) + e.right.value)
            out[mid] = {"name": name, "code": code}
    return out


def readme_summary(folder: Path, root: Path) -> dict:
    for name in ("README.md", "readme.md", "README.rst"):
        p = folder / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            title = ""
            para = []
            for ln in text.splitlines():
                s = ln.strip()
                if not title and s.startswith("#"):
                    title = s.lstrip("#").strip()
                    continue
                if title and s and not s.startswith(("#", "!", "<", "|", "```", "[!")):
                    para.append(s)
                elif para and not s:
                    break
            return {"readme": str(p.relative_to(root)), "title": clean(title, 200), "summary": clean(" ".join(para), 500)}
    return {"readme": None, "title": "", "summary": ""}


def folder_inventory(base: Path, test_refs: dict, root: Path) -> list[dict]:
    items = []
    for d in sorted(base.iterdir()):
        if d.name.startswith((".", "_")) and d.name not in ("_shared",):
            continue
        if EXCLUDE_RE.search(d.name) or d.name in SKIP_DIRS:
            continue
        rel = str(d.relative_to(root))
        if d.is_dir():
            pys = list(walk_py(d, root))
            rec = {"name": d.name, "path": rel, "py_files": len(pys),
                   "files": sum(1 for p in d.rglob("*") if p.is_file() and not any(s in p.parts for s in SKIP_DIRS))}
            rec.update(readme_summary(d, root))
            if not rec["summary"]:
                for p in pys:
                    _, t = parse(p)
                    if t is not None and ast.get_docstring(t):
                        rec["summary"] = first_line(ast.get_docstring(t))
                        rec["summary_from"] = str(p.relative_to(root))
                        break
            rec["children"] = sorted(c.name for c in d.iterdir() if not c.name.startswith((".", "__"))
                                     and not EXCLUDE_RE.search(c.name))[:60]
        elif d.suffix in (".py", ".sh", ".txt", ".md"):
            rec = {"name": d.name, "path": rel, "py_files": int(d.suffix == ".py"), "files": 1}
            if d.suffix == ".py":
                _, t = parse(d)
                rec["summary"] = first_line(ast.get_docstring(t)) if t is not None else ""
            else:
                rec["summary"] = clean(d.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
                                       if d.stat().st_size else "", 200)
        else:
            continue
        rec["tests"] = test_refs.get(rel, 0)
        items.append(rec)
    return items


def docs_inventory(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / "docs").rglob("*.md")):
        rel = p.relative_to(root)
        if EXCLUDE_RE.search(str(rel)) or "pt/mapa_completo" in str(rel):
            continue
        title = ""
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.startswith("#"):
                title = ln.lstrip("#").strip()
                break
        out.append({"path": str(rel), "title": clean(title, 200)})
    return out


def workflows(root: Path) -> list[dict]:
    out = []
    for p in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        text = p.read_text(encoding="utf-8")
        name = re.search(r"^name:\s*(.+)$", text, re.M)
        jobs = re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", text.split("\njobs:", 1)[-1], re.M)
        on = text.split("\non:", 1)[-1].split("\njobs:", 1)[0] if "\non:" in text else ""
        triggers = re.findall(r"^  ([a-z_]+):", on, re.M)
        crons = re.findall(r"cron:\s*['\"]([^'\"]+)", text)
        out.append({"file": str(p.relative_to(root)), "name": name.group(1).strip().strip("'\"") if name else p.stem,
                    "triggers": triggers, "cron": crons, "jobs": jobs})
    return out


def repo_root() -> Path:
    """The checkout this package lives in (``PINNEAPPLE_ROOT`` overrides it)."""
    env = os.environ.get("PINNEAPPLE_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[1]


def cache_dir() -> Path:
    """Where generated Studio files live (``PINNEAPPLE_STUDIO_CACHE``, else ``~/.cache/pinneapple/studio``)."""
    env = os.environ.get("PINNEAPPLE_STUDIO_CACHE")
    d = Path(env) if env else Path.home() / ".cache" / "pinneapple" / "studio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_key(root: Path) -> str:
    """The commit of ``root`` plus a hash of its uncommitted changes to Python files (empty outside git)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py", "*.json", "*.toml"], cwd=root,
                               capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    if not head:
        return ""
    return head[:12] + ("-" + hashlib.sha256(dirty.encode()).hexdigest()[:8] if dirty else "")


def load(root: Path | None = None, *, rebuild: bool = False) -> dict:
    """The inventory of ``root`` (default :func:`repo_root`), from the cache when its key still matches."""
    root = Path(root or repo_root()).resolve()
    key = source_key(root)
    path = cache_dir() / f"inventory-{hashlib.sha256(str(root).encode()).hexdigest()[:8]}-{key or 'nogit'}.json"
    if key and path.is_file() and not rebuild:
        return json.loads(path.read_text(encoding="utf-8"))
    inv = build(root)
    if key:
        path.write_text(json.dumps(inv, ensure_ascii=False), encoding="utf-8")
    return inv


def build(root: Path | None = None) -> dict:
    """Walk ``root`` and return the inventory (see the module docstring)."""
    root = Path(root or repo_root()).resolve()
    pkgs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("pinneapple")
                  and any(p.rglob("*.py")))
    modules = {}
    for pkg in pkgs:
        for p in walk_py(root / pkg, root):
            rec = module_record(p, root)
            modules[rec["module"]] = rec
    known = set(modules)

    # testes
    test_files = {}
    per_module = defaultdict(int)
    per_module_files = defaultdict(list)
    path_refs = defaultdict(int)
    for p in walk_py(root / "tests", root):
        if not p.name.startswith("test"):
            continue
        src, tree = parse(p)
        if tree is None:
            continue
        tests = count_tests(tree)
        imps = imported_modules(tree, known, "")
        rel = str(p.relative_to(root))
        test_files[rel] = {"tests": len(tests), "imports": sorted(imps)}
        for m in imps:
            per_module[m] += len(tests)
            per_module_files[m].append(rel)
        for ref in set(re.findall(r"(?:apps|examples|scripts)/[A-Za-z0-9_.\-]+", src)):
            path_refs[ref.rstrip(".")] += len(tests)
    for m, rec in modules.items():
        rec["tests"] = per_module.get(m, 0)
        rec["test_files"] = sorted(per_module_files.get(m, []))

    packages = {}
    for pkg in pkgs:
        mods = [r for r in modules.values() if r["module"] == pkg or r["module"].startswith(pkg + ".")]
        subs = sorted({r["module"].split(".")[1] for r in mods if r["module"].count(".") >= 1})
        tests_dir = root / "tests" / pkg
        n_tests_dir = sum(v["tests"] for k, v in test_files.items() if k.startswith(f"tests/{pkg}/"))
        files_any = {f for r in mods for f in r["test_files"]}
        init = modules.get(pkg, {})
        packages[pkg] = {
            "doc": init.get("doc", ""), "doc1": init.get("doc1", ""),
            "modules": len(mods), "lines": sum(r["lines"] for r in mods),
            "classes": sum(len(r["classes"]) for r in mods),
            "functions": sum(len(r["functions"]) for r in mods),
            "submodules": subs,
            "test_files_importing": len(files_any),
            "tests_in_files_importing": sum(test_files[f]["tests"] for f in files_any),
            "tests_dir": str(tests_dir.relative_to(root)) if tests_dir.exists() else None,
            "tests_in_tests_dir": n_tests_dir,
            "modules_without_tests": sum(1 for r in mods if r["tests"] == 0),
        }

    status = json.loads((root / "pinneapple_catalog" / "method_status.json").read_text())
    names = method_names(root)
    methods = {}
    for mid, v in status["methods"].items():
        methods[mid] = {"status": v.get("status"), "name": names.get(mid, {}).get("name", ""),
                        "code": names.get(mid, {}).get("code", []),
                        "reference_tests": len(v.get("reference_tests", [])),
                        "other_tests": len(v.get("other_tests", []))}

    py = tomllib.loads((root / "pyproject.toml").read_text())
    proj = py["project"]
    extras = {k: v for k, v in proj.get("optional-dependencies", {}).items()}

    inv = {
        "generated_from_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                                text=True).stdout.strip(),
        "project": {"name": proj.get("name"), "version": proj.get("version"),
                    "requires_python": proj.get("requires-python"),
                    "dependencies": proj.get("dependencies", []),
                    "scripts": proj.get("scripts", {})},
        "extras": extras,
        "top_api": top_api(root),
        "packages": packages,
        "modules": modules,
        "tests": {"files": test_files, "n_files": len(test_files),
                  "n_tests": sum(v["tests"] for v in test_files.values())},
        "method_status": {"generated_on": status.get("generated_on"), "rule": status.get("rule"),
                          "test_files_scanned": status.get("test_files_scanned"),
                          "test_functions_scanned": status.get("test_functions_scanned"),
                          "methods": methods},
        "apps": folder_inventory(root / "apps", path_refs, root),
        "examples": folder_inventory(root / "examples", path_refs, root),
        "scripts": folder_inventory(root / "scripts", path_refs, root),
        "benchmarks": folder_inventory(root / "benchmarks", path_refs, root),
        "docs": docs_inventory(root),
        "workflows": workflows(root),
    }
    inv["totals"] = {
        "packages": len(packages),
        "modules": len(modules),
        "lines": sum(r["lines"] for r in modules.values()),
        "public_classes": sum(len(r["classes"]) for r in modules.values()),
        "public_functions": sum(len(r["functions"]) for r in modules.values()),
        "public_methods": sum(len(c["methods"]) for r in modules.values() for c in r["classes"]),
        "modules_with_tests": sum(1 for r in modules.values() if r["tests"] > 0),
        "test_files": len(test_files),
        "test_functions": inv["tests"]["n_tests"],
        "extras": len(extras),
        "apps": len(inv["apps"]),
        "examples": len(inv["examples"]),
        "scripts": len(inv["scripts"]),
        "docs_md": len(inv["docs"]),
        "workflows": len(inv["workflows"]),
        "methods": len(methods),
        "methods_by_status": {s: sum(1 for m in methods.values() if m["status"] == s)
                              for s in sorted({m["status"] for m in methods.values()})},
        "pp_names": len(set(inv["top_api"]["eager"]) | set(inv["top_api"]["lazy"])
                        | set(inv["top_api"]["submodules"]) | set(inv["top_api"]["defined_here"])),
    }
    return inv
