"""Parameter campaigns of OpenFOAM runs, split over any number of machines.

A campaign is one verified base case and a list of runs, each a set of input parameters. Every run:
1. copies the base case (a directory or a .tar.gz), optionally with another mesh;
2. applies the parameters: dictionary edits with values computed from the parameters, an optional patch rebuilt
   to a given area (an inlet diameter as an input), and an optional Python hook;
3. runs the pre-processing commands, decomposes the case and runs the solver with MPI;
4. samples lines while the solver runs (``LiveLineSampler``, so no function objects are needed and old times can
   be deleted to bound disk use);
5. writes ``<results>/<id>.json``: parameters, solver health from the log (Courant number, phase-fraction bounds,
   phase volume, end reached) and the time-mean profiles on the lines.

The same plan file runs on one workstation or on many machines. Each machine takes its share with ``--worker`` (the
runs tagged with that worker) or ``--shard i/n`` (round robin), skips the runs whose result already exists, and
claims a run with a lock file, so machines sharing a disk never run the same case. Results can be committed and
pushed to a git branch as each run finishes (``--publish-git``), which is how machines that share nothing, such as
cloud sessions, return their results.

    python -m pinneapple_simulation.external_solvers.openfoam.campaign plan.json --commands 5   # one line per machine
    python -m pinneapple_simulation.external_solvers.openfoam.campaign plan.json --shard 2/5 --procs 4 --publish-git

Plan (JSON; relative paths are relative to the plan file)::

    {"name": "jet sweep", "base": "base_case.tar.gz", "solver": "interFoam", "procs": 4, "end_time": 30,
     "pre": ["checkMesh", "setFields"],
     "derived": {"Uy": "U*cos(radians(theta))", "Uz": "U*sin(radians(theta))"},
     "edits": [{"file": "0/U", "key": "boundaryField/inlet/value", "value": "uniform (0 {Uy} {Uz})"}],
     "area_patch": {"patch": "inlet", "area": "pi*(d/1000)**2/4", "centre": [0, -7.4, -1.3], "normal": [0, -1, 0],
                    "from": ["inlet", "wall"], "rest": "wall"},
     "sample": {"fields": ["k", "alpha.water"], "keep_every": 10, "average_from": 0,
                "lines": [{"name": "L1", "start": [0, -7.4, -2.2], "end": [0, 15.3, -2.2], "n": 101}]},
     "runs": [{"id": "T01", "params": {"U": 8.6, "theta": 5.0, "d": 109}, "worker": "1"},
              {"id": "D1", "params": {"U": 10, "theta": 10, "d": 85}, "mesh": "mesh_deflector.tar.gz"}]}

Expressions (``derived``, ``area``) may use the parameters, earlier derived values, numbers, + - * / ** and the
functions and constants of ``math``; nothing else is evaluated. Edit values are templates filled with them
(``{Uy}``, or ``{Uy:.3f}``); a list value is written as an OpenFOAM vector ``(a b c)``.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pinneapple_simulation.numerical_solvers.free_surface import (
    LiveLineSampler,
    _ascii_list,
    read_profiles,
)

__all__ = ["CampaignRun", "Campaign", "set_entry", "get_entry", "evaluate", "area_patch", "parse_solver_log",
           "write_decompose", "GitPublisher", "worker_commands"]

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")


# --------------------------------------------------------------------------------------------- expressions
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")} | {"abs": abs, "min": min, "max": max,
                                                                            "round": round}


def evaluate(expr: str | float | int | list, names: dict) -> float | list:
    """Value of an arithmetic expression over ``names`` and ``math``; only numbers, names, arithmetic, function
    calls of math and lists are allowed."""
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, list):
        return [evaluate(e, names) for e in expr]

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            if n.id in _MATH:
                return _MATH[n.id]
            raise NameError(f"unknown name {n.id!r} in {expr!r}")
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and isinstance(right, int) and abs(right) > 1024:
                left = float(left)                  # 9**9**9 in integers would stall the server; floats overflow
            return _BIN[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _MATH and not n.keywords:
            return _MATH[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"not allowed in an expression: {ast.dump(n)[:60]} in {expr!r}")
    return ev(ast.parse(str(expr), mode="eval"))


_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SPEC = re.compile(r"^[<>=^+\- #0]{0,4}(\d{1,2})?(,|_)?(\.\d{1,2})?[eEfFgGdnxXo%s]?$")


def format_template(template: str, values: dict) -> str:
    """``template.format(**values)`` restricted to ``{name}`` and ``{name:spec}`` with a numeric spec of width and
    precision below 100: user templates (case decks, dictionary values) can not reach attributes or items of the
    values (``{x.__class__}``, ``{x[0]}``) or ask for gigabyte-wide padding (``{x:>999999999}``)."""
    import string
    for _, name, spec, conv in string.Formatter().parse(template):
        if name is None:
            continue
        if not _FIELD.match(name) or conv not in (None, "s", "r"):
            raise ValueError(f"template field {{{name}}} not allowed: use {{name}} or {{name:.3g}}")
        if spec and not _SPEC.match(spec):
            raise ValueError(f"format spec {spec!r} of {{{name}}} not allowed")
    return template.format(**values)


class _V:
    """Template value: a number keeps format specs ({x:.3f}); a list is written as (a b c)."""

    def __init__(self, v):
        self.v = v

    def __format__(self, spec):
        if isinstance(self.v, (list, tuple)):
            return "(" + " ".join(format(float(x), spec or "g") for x in self.v) + ")"
        if isinstance(self.v, (int, float)):
            return format(self.v, spec) if spec else repr(float(self.v)) if isinstance(self.v, float) else str(self.v)
        return format(str(self.v), spec)


def resolve(params: dict, derived: dict | None = None) -> dict:
    """The parameters plus the derived values, in the order given."""
    names = dict(params)
    for k, e in (derived or {}).items():
        names[k] = evaluate(e, names)
    return names


# ------------------------------------------------------------------------------------- dictionary edits
_KEY = re.compile(r'"[^"]*"|[^\s{};]+')


def _skip(t: str, i: int) -> int:
    while i < len(t):
        if t[i].isspace():
            i += 1
        elif t.startswith("//", i):
            j = t.find("\n", i)
            i = len(t) if j < 0 else j + 1
        elif t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = len(t) if j < 0 else j + 2
        else:
            break
    return i


def _close(t: str, i: int, o: str = "{", c: str = "}") -> int:
    """Index just past the bracket that closes the one at ``i``."""
    depth = 0
    while i < len(t):
        if t.startswith("//", i) or t.startswith("/*", i):
            i = _skip(t, i)
            continue
        ch = t[i]
        if ch == o:
            depth += 1
        elif ch == c:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced brackets")


def _entries(t: str, a: int, b: int):
    """Top-level entries between a and b: (key, key_start, value_start, end, is_dict); end is past ';' or '}'."""
    i = _skip(t, a)
    while i < b:
        m = _KEY.match(t, i)
        if not m:
            i = _skip(t, i + 1)
            continue
        key, ks = m.group(0).strip('"'), i
        if key.startswith("#"):
            j = t.find("\n", i)
            i = _skip(t, b if j < 0 else j + 1)
            continue
        j = _skip(t, m.end())
        if j < b and t[j] == "{":
            e = _close(t, j)
            yield key, ks, j, e, True
            i = _skip(t, e)
            continue
        k = j
        while k < b and t[k] != ";":
            if t[k] == "(":
                k = _close(t, k, "(", ")")
            elif t[k] in "/" and (t.startswith("//", k) or t.startswith("/*", k)):
                k = _skip(t, k)
            else:
                k += 1
        yield key, ks, j, min(k + 1, b), False
        i = _skip(t, k + 1)


def _locate(t: str, path: Sequence[str], create: bool):
    a, b = 0, len(t)
    for depth, key in enumerate(path):
        last = depth == len(path) - 1
        hit = next((e for e in _entries(t, a, b) if e[0] == key), None)
        if hit is None:
            if not create:
                raise KeyError("/".join(path[:depth + 1]))
            ins = b - 1 if depth else b                          # inside the closing brace of the parent dict
            add = f"\n{key} ;\n" if last else f"\n{key}\n{{\n}}\n"
            t = t[:ins] + add + t[ins:]
            return _locate(t, path, create)
        if last:
            return t, hit
        a, b = hit[2] + 1, hit[3]
    raise KeyError("/".join(path))


def get_entry(text: str, key: str) -> str:
    """The value of ``a/b/c`` in OpenFOAM dictionary text (without the ';'), or the body of a sub-dictionary."""
    _, (k, ks, vs, e, is_dict) = _locate(text, key.split("/"), False)
    return text[vs + 1:e - 1].strip() if is_dict else text[vs:e].rstrip(";").strip()


def set_entry(text: str, key: str, value: str) -> str:
    """Set ``a/b/c`` (slash-separated, e.g. ``boundaryField/inlet/value``) to ``value``; missing levels are
    added."""
    t, (k, ks, vs, e, is_dict) = _locate(text, key.split("/"), True)
    if is_dict:
        raise ValueError(f"{key} is a sub-dictionary")
    return t[:vs] + f"{value};" + t[e:]


def edit_file(path: str | Path, key: str, value: str, root: str | Path | None = None) -> None:
    """Set ``key`` to ``value`` in an OpenFOAM dictionary file. With ``root``, the file must lie inside it (the edit
    list of a campaign comes from the user)."""
    p = Path(path)
    if root is not None and not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"edit target {path} is outside the case")
    p.write_text(set_entry(p.read_text(), key, value))


def reset_nonuniform(case: str | Path, n_cells: int) -> list[str]:
    """After a mesh swap, set every ``0/`` field whose internalField does not have ``n_cells`` values to its most
    common value (``setFields`` or the solver rebuilds the rest). Returns the fields changed."""
    out = []
    for f in sorted((Path(case) / "0").glob("*")):
        if not f.is_file():
            continue
        t = f.read_text()
        m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", t)
        if not m or int(m.group(2)) == n_cells:
            continue
        e = _close(t, m.end() - 1, "(", ")")
        body = t[m.end():e - 1]
        if m.group(1) == "scalar":
            vals, cnt = np.unique(np.array(body.split(), float), return_counts=True)
            val = repr(float(vals[cnt.argmax()]))
        else:
            first = re.search(r"\(([^()]*)\)", body)
            val = f"({first.group(1)})" if first else "(0 0 0)"
        t = t[:m.start()] + f"internalField   uniform {val}" + t[t.index(";", e):]
        f.write_text(t)
        out.append(f.name)
    return out


# ----------------------------------------------------------------------------------------------- mesh
def _mesh(poly: Path):
    n, b = _ascii_list(str(poly / "points"))
    pts = np.array([p.split() for p in re.findall(r"\(([^()]+)\)", b)[:n]], float)
    n, b = _ascii_list(str(poly / "faces"))
    faces = [list(map(int, f.split())) for f in re.findall(r"\(([^()]+)\)", b)[:n]]
    bd = (poly / "boundary").read_text()
    pat = {m.group(1): (int(m.group(2)), int(m.group(3)))
           for m in re.finditer(r"(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", bd)}
    return pts, faces, pat


def n_cells(case: str | Path) -> int:
    own = (Path(case) / "constant" / "polyMesh" / "owner").read_text()
    m = re.search(r"nCells:\s*(\d+)", own)
    if m:
        return int(m.group(1))
    return int(np.array(_ascii_list(str(Path(case) / "constant" / "polyMesh" / "owner"))[1].split(), int).max()) + 1


def area_patch(case: str | Path, patch: str, area: float, centre: Sequence[float], normal: Sequence[float],
               from_patches: Sequence[str] | None = None, rest: str = "wall", plane_tol: float = 0.03,
               bashrc: str | None = None) -> dict:
    """Rebuild boundary patch ``patch`` with the faces of the plane through ``centre`` (outward ``normal``) that are
    nearest to ``centre``, until their area reaches ``area``: a circular inlet of a chosen diameter cut from a
    larger opening or from the wall. Faces of the old ``patch`` left out go to ``rest``. ASCII meshes."""
    case = Path(case)
    poly = case / "constant" / "polyMesh"
    pts, faces, pat = _mesh(poly)
    c0, nrm = np.asarray(centre, float), np.asarray(normal, float) / np.linalg.norm(normal)
    cand = []
    for name in from_patches or (patch, rest):
        if name not in pat:
            continue
        nf, sf = pat[name]
        for fi in range(sf, sf + nf):
            p = pts[faces[fi]]
            c = p.mean(0)
            S = sum(0.5 * np.cross(p[k] - c, p[(k + 1) % len(p)] - c) for k in range(len(p)))
            a = float(np.linalg.norm(S))
            if abs(np.dot(c - c0, nrm)) < plane_tol and np.dot(S, nrm) / a > 0.99:
                r = c - c0 - np.dot(c - c0, nrm) * nrm
                cand.append((float(np.linalg.norm(r)), fi, a))
    if not cand:
        raise ValueError(f"no boundary face on the plane through {tuple(c0)} with normal {tuple(nrm)}")
    cand.sort()
    sel, acc = [], 0.0
    for _, fi, a in cand:
        if acc + a / 2 > area:
            break
        sel.append(fi)
        acc += a
    old = set(range(pat[patch][1], pat[patch][1] + pat[patch][0])) if patch in pat else set()
    left = sorted(old - set(sel))
    (poly / "sets").mkdir(exist_ok=True)
    for name, ids in (("_areaPatch", sel), ("_areaRest", left)):
        (poly / "sets" / name).write_text(
            f"FoamFile {{ version 2.0; format ascii; class faceSet; object {name}; }}\n{len(ids)}\n(\n"
            + "\n".join(map(str, sorted(ids))) + "\n)\n")
    rest_type = re.search(rf"\n\s*{rest}\s*\{{[^}}]*?type\s+(\w+);", (poly / "boundary").read_text())
    (case / "system" / "createPatchDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object createPatchDict; }\npointSync false;\n"
        "patches (\n { name _carved; patchInfo { type patch; } constructFrom set; set _areaPatch; }\n"
        + (f" {{ name {rest}; patchInfo {{ type {rest_type.group(1) if rest_type else 'wall'}; }} "
           "constructFrom set; set _areaRest; }\n" if left else "") + ");\n")
    _sh("createPatch -overwrite", case, "log.createPatch", bashrc)
    t = (poly / "boundary").read_text()
    t = re.sub(rf"\n\s*{patch}\s*\{{[^}}]*nFaces\s+0;[^}}]*\}}", "", t)      # the emptied old patch
    (poly / "boundary").write_text(re.sub(r"\b_carved\b", patch, t))
    shutil.rmtree(poly / "sets", ignore_errors=True)
    return {"area_target": area, "area_patch": acc, "faces": len(sel)}


def write_decompose(case: str | Path, n: int, method: str = "auto", bashrc: str | None = None) -> str:
    """decomposeParDict for ``n`` ranks. ``auto`` uses scotch when the installation has it, else ``simple``
    slices along the longest side of the mesh. Returns the method used."""
    case = Path(case)
    if method == "auto":
        r = subprocess.run(["bash", "-c", f'source "{bashrc or FOAM_BASHRC}" >/dev/null 2>&1; '
                                          'ls "$FOAM_LIBBIN"/libscotchDecomp* 2>/dev/null'],
                           capture_output=True, text=True, check=False)
        method = "scotch" if r.stdout.strip() else "simple"
    body = f"numberOfSubdomains {n};\nmethod {method};\n"
    if method == "simple":
        pts = _mesh(case / "constant" / "polyMesh")[0]
        ax = int(np.argmax(np.ptp(pts, 0)))
        nn = [1, 1, 1]
        nn[ax] = n
        body += f"simpleCoeffs {{ n ({nn[0]} {nn[1]} {nn[2]}); delta 0.001; }}\n"
    (case / "system" / "decomposeParDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n" + body)
    return method


# ----------------------------------------------------------------------------------------- solver log
def parse_solver_log(text: str, every: int = 200) -> dict:
    """Health of a transient run from its log: last time, Courant numbers, phase-fraction bounds and volume
    history (VOF solvers), whether it ended normally, and the first fatal error."""
    blocks = text.split("\nTime = ")[1:]
    times = [float(b.split("\n", 1)[0].split()[0]) for b in blocks if b.split("\n", 1)[0].split()]
    co = [float(x) for x in re.findall(r"\nCourant Number mean: \S+ max: (\S+)", text)]
    ico = [float(x) for x in re.findall(r"\nInterface Courant Number mean: \S+ max: (\S+)", text)]
    bounds: dict = {}
    for name, lo, hi in re.findall(r"Min\(([^)]+)\) = (\S+)\s+Max\(\1\) = (\S+)", text):
        b = bounds.setdefault(name, [math.inf, -math.inf])
        b[0], b[1] = min(b[0], float(lo)), max(b[1], float(hi))
    vol = [(float(b.split("\n", 1)[0].split()[0]), float(re.findall(r"volume fraction = (\S+)", b)[-1]))
           for b in blocks[::every] if "volume fraction =" in b]
    fatal = re.search(r"FOAM FATAL[^\n]*\n(?:.*\n){0,6}", text)
    return {"last_time": times[-1] if times else None, "steps": len(times), "courant_max": max(co) if co else None,
            "interface_courant_max": max(ico) if ico else None, "bounds": bounds, "phase_volume": vol,
            "ended": "\nEnd\n" in text or text.rstrip().endswith("End"), "fatal": fatal.group(0) if fatal else None}


# --------------------------------------------------------------------------------------------- running
def _sh(cmd: str, cwd: Path, log: str, bashrc: str | None = None, check: bool = True) -> int:
    with open(Path(cwd) / log, "w") as f:
        rc = subprocess.run(["bash", "-c", f'source "{bashrc or FOAM_BASHRC}" >/dev/null 2>&1; {cmd}'], cwd=cwd,
                            stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    if rc and check:
        raise RuntimeError(f"{cmd} failed (exit {rc}), see {Path(cwd) / log}")
    return rc


def _unpack(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        with tarfile.open(src) as tf:
            tf.extractall(dst, filter="data")


@dataclass
class CampaignRun:
    id: str
    params: dict
    mesh: str | None = None          # directory or .tar.gz holding constant/polyMesh, replacing the base mesh
    worker: str | None = None
    group: str = ""


@dataclass
class Campaign:
    base: str
    runs: list[CampaignRun]
    name: str = "campaign"
    solver: str = "interFoam"
    procs: int = 4
    end_time: float | None = None
    pre: Sequence[str] = ("checkMesh",)
    derived: dict = field(default_factory=dict)
    edits: list[dict] = field(default_factory=list)
    area_patch: dict | None = None
    setup: str | Callable | None = None       # "module:function" or callable(case, names, run) -> dict | None
    sample: dict | None = None                 # {"lines": [...], "fields": [...], "keep_every": s, "average_from": s}
    results: str = "results"
    workdir: str | None = None
    keep_case: bool = False
    decompose: str = "auto"
    mpirun: str = "mpirun --allow-run-as-root --oversubscribe"
    bashrc: str | None = None
    root: str = "."                            # paths in the plan are relative to this

    # ------------------------------------------------------------------ plan file
    @classmethod
    def from_json(cls, path: str | Path, **over) -> Campaign:
        path = Path(path).resolve()
        d = json.loads(path.read_text())
        d["runs"] = [r if isinstance(r, CampaignRun) else CampaignRun(**r) for r in d["runs"]]
        d.setdefault("root", str(path.parent))
        d.update({k: v for k, v in over.items() if v is not None})
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown plan keys: {sorted(extra)}")
        return cls(**d)

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d.pop("root")
        if callable(d.get("setup")):
            raise ValueError("a callable setup cannot be saved; use 'module:function'")
        Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False))

    def _p(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else Path(self.root) / q

    @property
    def results_dir(self) -> Path:
        return self._p(self.results)

    # ------------------------------------------------------------------ selection
    def pending(self, ids: Sequence[str] | None = None, worker: str | None = None,
                shard: tuple[int, int] | None = None) -> list[CampaignRun]:
        runs = [r for r in self.runs if (not ids or r.id in ids) and (worker is None or str(r.worker) == str(worker))]
        if shard is not None:
            i, n = shard
            runs = [r for k, r in enumerate(runs) if k % n == i - 1]
        return [r for r in runs if not (self.results_dir / f"{r.id}.json").exists()]

    def _claim(self, rid: str, force: bool) -> Path | None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        lock = self.results_dir / f"{rid}.running"
        if force:
            lock.unlink(missing_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        os.close(fd)
        return lock

    # ------------------------------------------------------------------ one run
    def stage(self, run: CampaignRun, case: Path) -> dict:
        """Write the case of ``run`` in ``case`` (no solver): base, mesh, edits, area patch, hook, endTime,
        decomposition. Returns what was applied."""
        shutil.rmtree(case, ignore_errors=True)
        case.mkdir(parents=True)
        _unpack(self._p(self.base), case)
        info: dict = {}
        if run.mesh:
            shutil.rmtree(case / "constant" / "polyMesh", ignore_errors=True)
            src = self._p(run.mesh)
            if src.is_dir():
                inner = src / "constant" / "polyMesh" if (src / "constant" / "polyMesh").is_dir() else src
                shutil.copytree(inner, case / "constant" / "polyMesh")
            else:
                _unpack(src, case)
            info["fields_reset"] = reset_nonuniform(case, n_cells(case))
        names = resolve(run.params, self.derived)
        if self.area_patch:
            ap = dict(self.area_patch)
            info["area_patch"] = area_patch(case, ap["patch"], float(evaluate(ap["area"], names)), ap["centre"],
                                            ap["normal"], ap.get("from"), ap.get("rest", "wall"),
                                            ap.get("plane_tol", 0.03), self.bashrc)
            names["patch_area"] = info["area_patch"]["area_patch"]
        fmt = {k: _V(v) for k, v in names.items()}
        for e in self.edits:
            edit_file(case / e["file"], e["key"], e["value"].format_map(fmt))
        if self.setup:
            fn = self.setup
            if isinstance(fn, str):
                import importlib
                mod, _, attr = fn.partition(":")
                if str(self.root) not in sys.path:
                    sys.path.insert(0, str(self.root))
                fn = getattr(importlib.import_module(mod), attr)
            info["setup"] = fn(case, names, run) or {}
        if self.end_time is not None:
            edit_file(case / "system" / "controlDict", "endTime", repr(float(self.end_time)))
        if self.procs > 1:
            info["decompose"] = write_decompose(case, self.procs, self.decompose, self.bashrc)
        info["values"] = {k: v for k, v in names.items() if isinstance(v, (int, float, list))}
        return info

    def run_one(self, run: CampaignRun, log: Callable[[str], None] = print) -> dict:
        work = Path(self.workdir) if self.workdir else Path(tempfile.gettempdir()) / "pinneapple_campaign"
        case = work / self.name.replace(" ", "_") / run.id
        t0 = time.time()
        info = self.stage(run, case)
        for cmd in self.pre:
            _sh(cmd, case, f"log.{cmd.split()[0]}", self.bashrc)
        if self.procs > 1:
            _sh("decomposePar -force", case, "log.decomposePar", self.bashrc)
        smp = dict(self.sample or {})
        sampler = None
        if smp.get("lines"):
            end = float(self.end_time) if self.end_time is not None else float(
                get_entry((case / "system" / "controlDict").read_text(), "endTime"))
            sampler = LiveLineSampler(case, smp["lines"], smp.get("fields", ["U"]), keep_every=smp.get("keep_every"),
                                      end_time=end).start()
        cmd = f"{self.mpirun} -np {self.procs} {self.solver} -parallel" if self.procs > 1 else self.solver
        log(f"{run.id}: {cmd} in {case}")
        try:
            rc = _sh(cmd, case, f"log.{self.solver}", self.bashrc, check=False)
        finally:
            if sampler is not None:
                sampler.stop()
        health = parse_solver_log((case / f"log.{self.solver}").read_text())
        res = {"id": run.id, "group": run.group, "params": run.params, "mesh": run.mesh, "rc": rc,
               "host": socket.gethostname(), "wall_s": round(time.time() - t0, 1), "procs": self.procs, **info,
               "health": health, "lines": {}}
        if sampler is not None:
            prof = read_profiles(str(case), t_start=float(smp.get("average_from", 0.0)))
            for ln, d in prof.items():
                res["lines"][ln] = {"distance": d["s"].round(6).tolist(), "xyz": d["xyz"].round(6).tolist(),
                                    "times": d["times"].tolist(),
                                    **{f"mean_{f}": d[f"mean_{f}"].tolist() for f in smp.get("fields", ["U"])}}
        if not self.keep_case:
            shutil.rmtree(case, ignore_errors=True)
        return res

    # ------------------------------------------------------------------ many runs
    def run(self, ids: Sequence[str] | None = None, worker: str | None = None, shard: tuple[int, int] | None = None,
            publish: GitPublisher | None = None, force: bool = False, log: Callable[[str], None] = print) -> list[dict]:
        """Run this machine's share of the pending runs, one after the other; each result is written (and
        published) as soon as it is done. A failed run writes its result with ``rc`` != 0 and the log health."""
        out = []
        for run in self.pending(ids, worker, shard):
            lock = self._claim(run.id, force)
            if lock is None:
                log(f"{run.id}: claimed by another worker, skipped")
                continue
            try:
                try:
                    res = self.run_one(run, log)
                except RuntimeError as e:
                    res = {"id": run.id, "params": run.params, "rc": -1, "error": str(e), "host": socket.gethostname()}
                dst = self.results_dir / f"{run.id}.json"
                tmp = dst.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(res))
                tmp.replace(dst)
            finally:
                lock.unlink(missing_ok=True)
            if publish is not None:
                publish(dst, f"{self.name}: result {run.id}")
            log(f"{run.id}: rc {res['rc']}, {res.get('wall_s', 0) / 3600:.2f} h")
            out.append(res)
        return out


class GitPublisher:
    """Commit and push one result file to the current branch, rebasing on what other workers pushed meanwhile.
    The author is ``name``/``email``, else the repository's git config."""

    def __init__(self, name: str | None = None, email: str | None = None, remote: str = "origin", tries: int = 6):
        self.name, self.email, self.remote, self.tries = name, email, remote, tries

    def __call__(self, path: Path, message: str) -> bool:
        repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.parent, capture_output=True,
                              text=True, check=True).stdout.strip()
        git = ["git"] + (["-c", f"user.name={self.name}"] if self.name else []) + (
            ["-c", f"user.email={self.email}"] if self.email else [])
        subprocess.run(git + ["add", str(path)], cwd=repo, check=True)
        subprocess.run(git + ["commit", "-qm", message], cwd=repo, check=True)
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True,
                                check=True).stdout.strip()
        for k in range(self.tries):
            subprocess.run(git + ["pull", "-q", "--rebase", self.remote, branch], cwd=repo, check=False)
            if subprocess.run(git + ["push", "-q", self.remote, f"HEAD:{branch}"], cwd=repo, check=False).returncode == 0:
                return True
            time.sleep(2 ** (k + 1))
        return False


def worker_commands(plan: str, n: int, procs: int = 4, publish_git: bool = True) -> list[str]:
    """One shell line per machine: shard i of n, in the background, logging to campaign_<i>.log."""
    mod = "pinneapple_simulation.external_solvers.openfoam.campaign"
    return [f"nohup python -m {mod} {plan} --shard {i}/{n} --procs {procs}"
            f"{' --publish-git' if publish_git else ''} > campaign_{i}.log 2>&1 &" for i in range(1, n + 1)]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run an OpenFOAM parameter campaign (this machine's share).")
    ap.add_argument("plan")
    ap.add_argument("ids", nargs="*", help="only these run ids")
    ap.add_argument("--worker", help="the runs tagged with this worker")
    ap.add_argument("--shard", help="i/n: every n-th pending run, starting at the i-th")
    ap.add_argument("--procs", type=int)
    ap.add_argument("--end-time", type=float)
    ap.add_argument("--workdir")
    ap.add_argument("--results")
    ap.add_argument("--keep-case", action="store_true", default=None)
    ap.add_argument("--publish-git", action="store_true", help="commit and push each result to the current branch")
    ap.add_argument("--git-name", default=os.environ.get("CAMPAIGN_GIT_NAME"))
    ap.add_argument("--git-email", default=os.environ.get("CAMPAIGN_GIT_EMAIL"))
    ap.add_argument("--force", action="store_true", help="ignore claim locks left by a dead worker")
    ap.add_argument("--stage-only", metavar="DIR", help="write the cases to DIR without running them")
    ap.add_argument("--commands", type=int, metavar="N", help="print the command for each of N machines")
    ap.add_argument("--status", action="store_true", help="list done, running and pending runs")
    a = ap.parse_args(argv)
    c = Campaign.from_json(a.plan, procs=a.procs, end_time=a.end_time, workdir=a.workdir, results=a.results,
                           keep_case=a.keep_case)
    if a.commands:
        print("\n".join(worker_commands(a.plan, a.commands, c.procs, publish_git=True)))
        return
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    if a.status:
        for r in c.runs:
            st = "done" if (c.results_dir / f"{r.id}.json").exists() else (
                "running" if (c.results_dir / f"{r.id}.running").exists() else "pending")
            print(f"{r.id:10s} {st:8s} {json.dumps(r.params)}")
        return
    if a.stage_only:
        for r in c.pending(a.ids, a.worker, shard):
            print(r.id, json.dumps(c.stage(r, Path(a.stage_only) / r.id)))
        return
    pub = GitPublisher(a.git_name, a.git_email) if a.publish_git else None
    c.run(a.ids, a.worker, shard, publish=pub, force=a.force, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
