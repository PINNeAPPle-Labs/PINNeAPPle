"""The FreeCAD step: a parametric document in, named boundary patches out.

A *backend* sets the document's spreadsheet aliases, recomputes it and returns its faces as triangles
(:class:`CadGeometry`); the patches are then assigned by rules over each face's centre, mean outward normal, area and
surface type, so the same rules keep working when a parameter moves a face.

* :class:`FreeCADBackend` runs ``freecadcmd`` (FreeCAD 1.0 headless, the AppImage or a system install) with
  ``freecad_scripts/fc_job.py``: no GUI, no display. It also saves the modified document and a STEP file.
* :class:`FakeFreeCAD` builds the same templates in NumPy, for tests and machines without FreeCAD. Its "documents"
  are JSON files; it writes no STEP file and says so in the result.

Rules (:func:`assign_patches`) are expressions in the document's units (mm) over ``cx cy cz`` (face centre),
``nx ny nz`` (area-weighted outward normal), ``area``, the flags ``plane cylinder torus sphere cone`` (surface type),
the triangulated bounding box ``xmin ... zmax`` and every document parameter; ``and or not``, comparisons,
arithmetic and ``math`` functions are allowed, nothing else. ``"rest"`` takes the faces no earlier rule took::

    patches = {"inlet": "plane and nx < -0.99 and cy < Rc",
               "outlet": "plane and cy > Rc",
               "walls": "rest"}
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent / "freecad_scripts"
TYPES = ("plane", "cylinder", "torus", "sphere", "cone")

__all__ = ["Face", "CadGeometry", "PatchSpec", "FreeCADBackend", "FakeFreeCAD", "find_freecadcmd", "rule_eval",
           "assign_patches", "get_backend"]


# ----------------------------------------------------------------------------------------------- rules
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod}
_CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq,
        ast.NotEq: operator.ne}
_FN = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "atan2", "radians", "degrees", "hypot", "pi")}
_FN.update(abs=abs, min=min, max=max)


def rule_eval(expr: str, names: dict[str, Any], functions: dict[str, Any] | None = None) -> Any:
    """Value of a patch rule: numbers, names, arithmetic, comparisons (chained), ``and or not`` and a few ``math``
    functions (plus ``functions``); anything else raises ``ValueError``."""
    fns = {**_FN, **(functions or {})}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            if n.id in fns:
                return fns[n.id]
            raise NameError(f"unknown name {n.id!r} in rule {expr!r}; known: {sorted(names)}")
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.BoolOp):
            vals = (ev(v) for v in n.values)
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare) and all(type(o) in _CMP for o in n.ops):
            left = ev(n.left)
            for op, c in zip(n.ops, n.comparators, strict=True):
                right = ev(c)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fns and not n.keywords:
            return fns[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"not allowed in a patch rule: {ast.dump(n)[:60]} in {expr!r}")
    return ev(ast.parse(str(expr), mode="eval"))


@dataclass
class PatchSpec:
    """One boundary patch: the rule that selects its faces and its kind (``inlet``, ``outlet``, ``wall``,
    ``symmetry``); a plain string is a rule, the kind then follows the name (inlet*, outlet*, else wall)."""
    name: str
    where: str
    kind: str = ""

    def __post_init__(self):
        if not self.kind:
            low = self.name.lower()
            self.kind = "inlet" if low.startswith("inlet") else "outlet" if low.startswith("outlet") else "wall"
        if self.kind not in ("inlet", "outlet", "wall", "symmetry"):
            raise ValueError(f"patch {self.name}: kind {self.kind!r} is not inlet, outlet, wall or symmetry")

    @classmethod
    def parse(cls, patches: dict[str, Any] | list[PatchSpec]) -> list[PatchSpec]:
        if isinstance(patches, list):
            return [p if isinstance(p, PatchSpec) else cls(**p) for p in patches]
        return [cls(k, v) if isinstance(v, str) else cls(k, **v) for k, v in patches.items()]

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "where": self.where, "kind": self.kind}


# ----------------------------------------------------------------------------------------------- geometry
@dataclass
class Face:
    index: int
    type: str                                   # Plane, Cylinder, Toroid, Sphere, Cone, BSplineSurface, ...
    area: float
    centre: np.ndarray
    points: np.ndarray                          # (n, 3), document units
    triangles: np.ndarray                       # (m, 3), outward oriented

    @property
    def normal(self) -> np.ndarray:
        """Area-weighted mean outward normal of the triangles (unit; exact for a plane)."""
        P, T = self.points, self.triangles
        n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]).sum(0)
        L = np.linalg.norm(n)
        return n / L if L > 0 else n

    def names(self) -> dict[str, Any]:
        t = self.type.lower()
        flags = {k: (k in t) or (k == "torus" and "toroid" in t) for k in TYPES}
        n = self.normal
        return {"cx": float(self.centre[0]), "cy": float(self.centre[1]), "cz": float(self.centre[2]),
                "nx": float(n[0]), "ny": float(n[1]), "nz": float(n[2]), "area": float(self.area), **flags}


@dataclass
class CadGeometry:
    """Faces of one solid in document units (mm), with what the backend reported (volume, validity, files)."""
    faces: list[Face]
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, faces: list[dict[str, Any]], info: dict[str, Any] | None = None) -> CadGeometry:
        out = [Face(int(f["index"]), str(f["type"]), float(f["area"]), np.asarray(f["centre"], float),
                    np.asarray(f["points"], float).reshape(-1, 3), np.asarray(f["triangles"], np.int64).reshape(-1, 3))
               for f in faces]
        return cls(out, dict(info or {}))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        P = np.concatenate([f.points for f in self.faces])
        return P.min(0), P.max(0)

    def mesh(self, faces: list[int] | None = None, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """One triangle mesh of the given faces (default all), points merged across faces, times ``scale``."""
        sel = [self.faces[i] for i in (faces if faces is not None else range(len(self.faces)))]
        P = np.concatenate([f.points for f in sel])
        off = np.cumsum([0] + [len(f.points) for f in sel])[:-1]
        T = np.concatenate([f.triangles + o for f, o in zip(sel, off, strict=True)])
        lo, hi = P.min(0), P.max(0)
        q = np.round((P - lo) / (1e-7 * max(float(np.max(hi - lo)), 1e-30))).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        return P[first] * scale, inv.ravel()[T]

    def volume(self) -> float:
        """Enclosed volume of the triangulation (document units³); positive when the faces point outward."""
        V, F = self.mesh()
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def open_edges(self) -> int:
        """Edges used by one triangle only (0 for a closed, watertight triangulation)."""
        _, F = self.mesh()
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return int((counts == 1).sum())

    def names(self, parameters: dict[str, float] | None = None) -> dict[str, float]:
        lo, hi = self.bounds()
        out = {f"{a}{s}": float(v[i]) for i, a in enumerate("xyz") for s, v in (("min", lo), ("max", hi))}
        out.update({k: float(v) for k, v in (parameters or self.info.get("parameters") or {}).items()})
        return out


def assign_patches(geom: CadGeometry, patches: list[PatchSpec], parameters: dict[str, float] | None = None
                   ) -> dict[str, list[int]]:
    """Face indices of each patch: the first rule a face satisfies takes it; ``rest`` takes what is left. Raises
    ``ValueError`` when a face is left without a patch or a patch gets no face."""
    base = geom.names(parameters)
    out: dict[str, list[int]] = {p.name: [] for p in patches}
    rest = [p.name for p in patches if p.where.strip() in ("rest", "*")]
    if len(rest) > 1:
        raise ValueError(f"only one patch may take the rest; {rest} do")
    for f in geom.faces:
        names = {**base, **f.names()}
        hit = next((p.name for p in patches if p.where.strip() not in ("rest", "*") and rule_eval(p.where, names)),
                   rest[0] if rest else None)
        if hit is None:
            raise ValueError(f"face {f.index} ({f.type}, centre {np.round(f.centre, 3).tolist()} mm, normal "
                             f"{np.round(f.normal, 3).tolist()}) matches no patch rule")
        out[hit].append(f.index)
    empty = [k for k, v in out.items() if not v]
    if empty:
        raise ValueError(f"patches without faces: {empty}; faces are " + "; ".join(
            f"{f.index} {f.type} centre {np.round(f.centre, 2).tolist()} normal {np.round(f.normal, 2).tolist()}"
            for f in geom.faces))
    return out


def write_stl(path: str | Path, geom: CadGeometry, assignment: dict[str, list[int]], scale: float = 1e-3) -> Path:
    """ASCII STL with one ``solid <patch>`` per patch (snappyHexMesh turns each into a region), points times
    ``scale`` (mm -> m by default), merged across faces so shared edges stay shared."""
    V, F = geom.mesh(scale=scale)
    off = np.cumsum([0] + [len(f.triangles) for f in geom.faces])
    lines = []
    for name, idx in assignment.items():
        rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx])
        tri = V[F[rows]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
        lines.append(f"solid {name}")
        for k in range(len(tri)):
            lines.append(f" facet normal {n[k, 0]:.6e} {n[k, 1]:.6e} {n[k, 2]:.6e}\n  outer loop")
            lines += [f"   vertex {p[0]:.9e} {p[1]:.9e} {p[2]:.9e}" for p in tri[k]]
            lines.append("  endloop\n endfacet")
        lines.append(f"endsolid {name}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


# ----------------------------------------------------------------------------------------------- backends
def find_freecadcmd() -> str | None:
    """``PINNEAPPLE_FREECADCMD``, the automation toolchain's FreeCAD (``/opt/freecad`` AppImage), or PATH."""
    try:
        from pinneapple_tools.automation.toolchain import Toolchain
        found = Toolchain.discover().freecadcmd
    except Exception:                                    # the toolchain module is optional here
        found = None
    for c in (os.environ.get("PINNEAPPLE_FREECADCMD"), found, "/opt/freecad/usr/bin/freecadcmd", "freecadcmd",
              "FreeCADCmd"):
        if c and (shutil.which(c) or Path(c).is_file()):
            return shutil.which(c) or c
    return None


class FreeCADBackend:
    """FreeCAD through ``freecadcmd`` (headless). ``timeout_s`` per job."""
    name = "freecad"

    def __init__(self, freecadcmd: str | None = None, timeout_s: float = 600.0):
        self.freecadcmd = freecadcmd or find_freecadcmd()
        self.timeout_s = timeout_s

    def available(self) -> tuple[bool, str]:
        return (True, self.freecadcmd) if self.freecadcmd else (False, "freecadcmd not found (set "
                                                                       "PINNEAPPLE_FREECADCMD)")

    def version(self) -> str | None:
        if not self.freecadcmd:
            return None
        try:
            r = subprocess.run([self.freecadcmd, "--version"], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return None
        import re
        m = re.search(r"FreeCAD\s+([\d.]+)", r.stdout + r.stderr)
        return m.group(1) if m else None

    def _job(self, job: dict[str, Any], work: Path, log_name: str) -> dict[str, Any]:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        work.mkdir(parents=True, exist_ok=True)
        job = {**job, "result": str(work / f"{log_name}.result.json")}
        Path(job["result"]).unlink(missing_ok=True)
        jf = work / f"{log_name}.job.json"
        jf.write_text(json.dumps(job, indent=1))
        env = {**os.environ, "PINNEAPPLE_FC_JOB": str(jf), "QT_QPA_PLATFORM": "offscreen"}
        with open(work / f"log.{log_name}", "w") as f:
            subprocess.run([self.freecadcmd, str(SCRIPTS / "fc_job.py")], cwd=work, env=env, stdout=f,
                           stderr=subprocess.STDOUT, timeout=self.timeout_s)
        if not Path(job["result"]).is_file():
            raise RuntimeError(f"freecadcmd wrote no result (log {work / f'log.{log_name}'})")
        res = json.loads(Path(job["result"]).read_text())
        if res.get("error"):
            raise RuntimeError(f"FreeCAD raised:\n{res['error'][-1500:]}")
        return res

    def build(self, template: str, parameters: dict[str, float], output: str | Path) -> dict[str, Any]:
        """Write the parametric document of a template (see :mod:`.models`) to ``output`` (.FCStd)."""
        output = Path(output).resolve()
        return self._job({"job": "build", "template": template, "parameters": parameters, "output": str(output)},
                         output.parent, f"build_{output.stem}")

    def export(self, document: str | Path, parameters: dict[str, float], solid: str, out_dir: str | Path,
               tolerance_mm: float = 0.05) -> CadGeometry:
        out_dir = Path(out_dir).resolve()
        res = self._job({"job": "export", "document": str(Path(document).resolve()), "parameters": parameters,
                         "solid": solid, "output_dir": str(out_dir), "tolerance_mm": tolerance_mm}, out_dir, "export")
        faces = json.loads(Path(res["faces_json"]).read_text())
        return CadGeometry.from_json(faces, {**res, "backend": "freecad", "version": res.get("freecad")})


class FakeFreeCAD:
    """The templates of :mod:`.models` built in NumPy: the same faces (types, centres, outward triangles) and
    spreadsheet parameters as the FreeCAD documents, without FreeCAD. Documents are JSON:
    ``{"fake_freecad": template, "parameters": {...}, "solid": name}``."""
    name = "fake"

    def available(self) -> tuple[bool, str]:
        return True, "NumPy stand-in for FreeCAD (no STEP export)"

    def version(self) -> str:
        return "fake"

    def build(self, template: str, parameters: dict[str, float], output: str | Path) -> dict[str, Any]:
        from .models import MODELS
        if template not in MODELS:
            raise KeyError(f"no template {template!r}; there are {sorted(MODELS)}")
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fake_freecad": template, "parameters": dict(parameters),
                                   "solid": MODELS[template].solid}, indent=1))
        return {"ok": True, "document": str(out), "solid": MODELS[template].solid}

    def export(self, document: str | Path, parameters: dict[str, float], solid: str, out_dir: str | Path,
               tolerance_mm: float = 0.05) -> CadGeometry:
        from .models import MODELS
        doc = json.loads(Path(document).read_text())
        model = MODELS[doc["fake_freecad"]]
        unknown = [k for k in parameters if k not in doc["parameters"]]
        if unknown:
            raise KeyError(f"no spreadsheet alias {unknown}; the document has {sorted(doc['parameters'])}")
        values = {**doc["parameters"], **{k: float(v) for k, v in parameters.items()}}
        faces = model.fake(values, tolerance_mm)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.json").write_text(json.dumps({**doc, "parameters": values}, indent=1))
        geom = CadGeometry.from_json(faces)
        lo, hi = geom.bounds()
        geom.info = {"ok": True, "backend": "fake", "version": "fake", "parameters": values, "invalid_objects": [],
                     "valid": True, "closed": True, "solids": 1, "volume_mm3": geom.volume(),
                     "faces": len(faces), "bound_box_mm": [*lo.tolist(), *hi.tolist()], "step": None,
                     "document": str(out_dir / "model.json")}
        return geom


def get_backend(name: str = "auto"):
    """``"freecad"``, ``"fake"``, or ``"auto"`` (FreeCAD when ``freecadcmd`` is found, else the fake)."""
    if name == "fake":
        return FakeFreeCAD()
    if name == "freecad":
        return FreeCADBackend()
    if name == "auto":
        b = FreeCADBackend()
        return b if b.available()[0] else FakeFreeCAD()
    raise ValueError(f"backend {name!r} is not freecad, fake or auto")
