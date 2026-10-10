"""CalculiX studies in a few lines: a mesh (structured hexahedra, gmsh tetrahedra from CAD-like geometry, or an
engineer's deck), node and face sets picked by geometry, one analysis step, ``ccx``, and the results back as arrays.

    from pinneapple_simulation.external_solvers.calculix.study import FEModel, Material, Static, Frequency, solve
    m = FEModel.from_box(1.0, 0.1, 0.1, 40, 4, 4, element="C3D8I")          # or FEModel.from_gmsh(geo, size=...)
    m.material = Material("steel", E=210e9, nu=0.3, density=7850.0)
    root, tip = m.nodes_where(lambda X: X[:, 0] < 1e-9), m.faces_where(lambda C: C[:, 0] > 1.0 - 1e-9)
    res = solve(m, Static(fix={root: (1, 2, 3)}, pressure={tip: 1e6}), "work/beam")
    res.u, res.stress, res.von_mises, res.reactions
    res = solve(m, Frequency(6, fix={root: (1, 2, 3)}), "work/modes")
    res.frequencies, res.modes                                                # Hz, one (n, 3) shape per mode

Steps: ``Static`` (point loads, pressure on faces, gravity; nonlinear geometry and plasticity with increments),
``Frequency`` (eigenfrequencies and mode shapes), ``Buckle`` (buckling load factors and shapes), ``Heat``
(steady conduction with fixed temperatures, film convection and surface flux). Keyword syntax: CalculiX User's
Manual (G. Dhondt), section 7. Faces follow its numbering (hexahedra 1-6, tetrahedra 1-4).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .frd import read_frd
from .runner import run_ccx

# corner-node faces, CalculiX numbering (0-based node positions inside the element)
FACES = {
    "hex": [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)],
    "tet": [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)],
}


def _family(etype: str) -> str:
    e = etype.upper()
    if e.startswith(("C3D8", "C3D20", "DC3D8", "DC3D20")):
        return "hex"
    if e.startswith(("C3D4", "C3D10", "DC3D4", "DC3D10")):
        return "tet"
    raise ValueError(f"unsupported element type {etype}")


@dataclass
class Material:
    name: str = "steel"
    E: float = 210e9
    nu: float = 0.3
    density: float | None = 7850.0
    conductivity: float | None = None              # W/(m K)
    expansion: float | None = None                 # 1/K
    plastic: Sequence[tuple[float, float]] | None = None    # (true stress, plastic strain) rows, isotropic hardening

    def deck(self) -> list[str]:
        L = [f"*MATERIAL, NAME={self.name.upper()}", "*ELASTIC", f"{self.E:.10g}, {self.nu:.10g}"]
        if self.density is not None:
            L += ["*DENSITY", f"{self.density:.10g}"]
        if self.conductivity is not None:
            L += ["*CONDUCTIVITY", f"{self.conductivity:.10g}"]
        if self.expansion is not None:
            L += ["*EXPANSION", f"{self.expansion:.10g}"]
        if self.plastic:
            L += ["*PLASTIC"] + [f"{s:.10g}, {e:.10g}" for s, e in self.plastic]
        return L


@dataclass
class FEModel:
    nodes: np.ndarray                              # (n, 3)
    elements: np.ndarray                           # (m, k) 0-based node indices, CalculiX node order
    element: str = "C3D8I"
    material: Material = field(default_factory=Material)
    title: str = "pinneapple"

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_box(cls, L: float, W: float, H: float, nx: int, ny: int, nz: int, element: str = "C3D8I",
                 origin=(0.0, -0.5, -0.5)) -> FEModel:
        """Structured hexahedra on [0, L] x [-W/2, W/2] x [-H/2, H/2] (``origin`` in units of W and H)."""
        from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh
        m = box_mesh(L, W, H, nx, ny, nz, origin=(0.0, (origin[1] + 0.5) * W, (origin[2] + 0.5) * H))
        return cls(m.nodes, m.elements, element)

    @classmethod
    def from_hexmesh(cls, mesh, element: str = "C3D8I") -> FEModel:
        return cls(np.asarray(mesh.nodes, float), np.asarray(mesh.elements, int), element)

    @classmethod
    def from_inp(cls, path: str) -> FEModel:
        from .inp import read_inp
        cm = read_inp(path)
        ids, X = cm.node_array()
        pos = {int(i): k for k, i in enumerate(ids)}
        el = np.array([[pos[n] for n in conn] for _, conn in sorted(cm.elements.items())], int)
        return cls(X, el, cm.element_type)

    @classmethod
    def from_gmsh(cls, geo: str, size: float, order: int = 2, workdir: str | None = None,
                  optimize: bool = True) -> FEModel:
        """Tetrahedra (C3D10 for ``order`` 2, C3D4 for 1) of a gmsh .geo script (OpenCASCADE kernel allowed); only
        the volume elements are kept. Needs the ``gmsh`` executable."""
        exe = shutil.which("gmsh")
        if not exe:
            raise RuntimeError("gmsh not found: install gmsh to mesh geometry")
        work = workdir or os.path.join(os.getcwd(), "_gmsh")
        os.makedirs(work, exist_ok=True)
        gp = os.path.join(work, "part.geo")
        with open(gp, "w") as f:
            f.write(geo)
        out = os.path.join(work, "part.inp")
        cmd = [exe, gp, "-3", "-order", str(order), "-clmax", f"{size:.6g}", "-format", "inp", "-o", out,
               "-setnumber", "Mesh.SaveAll", "0"]
        if optimize:
            cmd += ["-optimize_netgen" if order == 1 else "-optimize_ho"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if p.returncode or not os.path.exists(out):
            raise RuntimeError(f"gmsh failed:\n{(p.stdout + p.stderr)[-3000:]}")
        return cls._from_gmsh_inp(out, order)

    @classmethod
    def _from_gmsh_inp(cls, path: str, order: int) -> FEModel:
        nodes, elems, etype, block = {}, [], None, None
        pending: list[int] = []
        for raw in open(path):
            s = raw.strip()
            if not s or s.startswith("**"):
                continue
            if s.startswith("*"):
                kw = s.upper()
                if kw.startswith("*NODE"):
                    block = "node"
                elif kw.startswith("*ELEMENT"):
                    t = re.search(r"TYPE=([A-Z0-9]+)", kw)
                    t = t.group(1) if t else ""
                    block = "elem" if t in ("C3D4", "C3D10") else "skip"
                    if block == "elem":
                        etype = t
                else:
                    block = None
                continue
            vals = [v for v in s.replace(" ", "").split(",") if v]
            if block == "node":
                nodes[int(vals[0])] = tuple(float(v) for v in vals[1:4])
            elif block == "elem":
                pending += [int(v) for v in vals]
                if not s.endswith(","):
                    elems.append(pending)
                    pending = []
        ids = np.array(sorted(nodes))
        pos = {int(i): k for k, i in enumerate(ids)}
        used = sorted({n for e in elems for n in e[1:]})
        keep = {n: k for k, n in enumerate(used)}
        X = np.array([nodes[n] for n in used], float)
        el = np.array([[keep[n] for n in e[1:]] for e in elems], int)
        del pos
        return cls(X, el, etype or ("C3D10" if order == 2 else "C3D4"))

    # ---------------------------------------------------------------- selections
    def nodes_where(self, pred: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Indices of the nodes whose coordinates satisfy ``pred(X) -> bool mask``."""
        return np.nonzero(pred(self.nodes))[0]

    def boundary_faces(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """(element index, CalculiX face number 1.., corner nodes) of every boundary face."""
        fam = _family(self.element)
        fl = FACES[fam]
        keys, owners = {}, []
        for f, loc in enumerate(fl):
            corners = self.elements[:, list(loc)]
            k = np.sort(corners, axis=1)
            for e in range(len(self.elements)):
                t = tuple(k[e])
                keys[t] = keys.get(t, 0) + 1
                owners.append((t, e, f + 1, corners[e]))
        out = [(e, f, c) for t, e, f, c in owners if keys[t] == 1]
        return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), [o[2] for o in out])

    def faces_where(self, pred: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Boundary faces whose centroid satisfies ``pred(C) -> bool mask``: (element indices, face numbers)."""
        e, f, c = self.boundary_faces()
        C = np.array([self.nodes[cc].mean(0) for cc in c])
        m = pred(C)
        return e[m], f[m]

    def outward_faces(self) -> np.ndarray:
        """Corner-node polygons of the boundary, ordered so that their normal points out of the body."""
        e, _, c = self.boundary_faces()
        C = np.asarray(c)
        P = self.nodes[C]
        if C.shape[1] >= 4:
            n = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
        else:
            n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        k = min(self.elements.shape[1], 8 if C.shape[1] >= 4 else 4)
        inward = ((self.nodes[self.elements[e, :k]].mean(1) - P.mean(1)) * n).sum(1) > 0
        C[inward] = C[inward][:, ::-1]
        return C

    def surface(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Corner-node polygons of the boundary (quads or triangles) for drawing."""
        _, _, c = self.boundary_faces()
        return self.nodes, c

    # ---------------------------------------------------------------- deck
    def deck(self, step) -> str:
        n = len(self.nodes)
        heat = isinstance(step, Heat)
        etype = self.element
        if heat and not etype.upper().startswith("DC"):
            etype = {"C3D8I": "C3D8", "C3D20R": "C3D20"}.get(etype.upper(), etype)
        L = [f"** {self.title}", "*NODE, NSET=NALL"]
        L += [f"{i + 1}, {x:.12g}, {y:.12g}, {z:.12g}" for i, (x, y, z) in enumerate(self.nodes)]
        L.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
        for e, conn in enumerate(self.elements):
            row = [str(e + 1)] + [str(int(v) + 1) for v in conn]
            chunks = [row[i:i + 16] for i in range(0, len(row), 16)]
            L += [", ".join(ch) + ("," if k < len(chunks) - 1 else "") for k, ch in enumerate(chunks)]
        L += self.material.deck()
        L.append(f"*SOLID SECTION, ELSET=EALL, MATERIAL={self.material.name.upper()}")
        del n
        L += step.deck(self)
        return "\n".join(L) + "\n"


def _sel(d):
    if d is None:
        return []
    if isinstance(d, dict):
        return list(d.items())
    return list(d)


@dataclass
class Static:
    """Linear (or nonlinear) static step. ``fix``: [(nodes, dofs)] or {...}; ``loads``: [(nodes, (Fx, Fy, Fz))] a
    total force shared by the nodes; ``pressure``: [((elements, faces), p)] positive into the body; ``gravity``:
    (gx, gy, gz) in m/s^2. ``nlgeom`` and ``increments`` (initial, total) for nonlinear geometry or plasticity."""
    fix: object = None
    loads: object = None
    pressure: object = None
    gravity: tuple[float, float, float] | None = None
    nlgeom: bool = False
    increments: tuple[float, float] | None = None
    prescribed: object = None                                    # [(nodes, dof, value)]

    def deck(self, m: FEModel) -> list[str]:
        return _static_deck(self, m)


def _bc_from(fix) -> list[str]:
    L = []
    for nodes, dofs in _sel(fix):
        dofs = (dofs,) if isinstance(dofs, int) else tuple(dofs)
        for nd in np.atleast_1d(np.asarray(nodes)):
            L += [f"{int(nd) + 1}, {d}, {d}" for d in dofs]
    return L


def _static_deck(self, m):
    L = ["*BOUNDARY"] + _bc_from(self.fix)
    L += [f"{int(n) + 1}, {d}, {d}, {v:.10g}" for nodes, d, v in _sel(self.prescribed) for n in np.atleast_1d(nodes)]
    L += ["*STEP" + (", NLGEOM, INC=1000" if (self.nlgeom or self.increments) else ""), "*STATIC"]
    if self.increments:
        L.append(f"{self.increments[0]:.6g}, {self.increments[1]:.6g}")
    if self.loads:
        L.append("*CLOAD")
        for nodes, F in _sel(self.loads):
            nodes = np.atleast_1d(nodes)
            for k in range(3):
                if F[k]:
                    L += [f"{int(n) + 1}, {k + 1}, {F[k] / len(nodes):.10g}" for n in nodes]
    if self.pressure or self.gravity:
        L.append("*DLOAD")
        for (els, fcs), p in _sel(self.pressure):
            L += [f"{int(e) + 1}, P{int(f)}, {p:.10g}" for e, f in zip(els, fcs, strict=True)]
        if self.gravity:
            g = np.asarray(self.gravity, float)
            gn = float(np.linalg.norm(g))
            L.append(f"EALL, GRAV, {gn:.10g}, {g[0] / gn:.10g}, {g[1] / gn:.10g}, {g[2] / gn:.10g}")
    L += ["*NODE FILE", "U, RF", "*EL FILE", "S, E" + (", PEEQ" if m.material.plastic else ""), "*END STEP"]
    return L




@dataclass
class Frequency:
    """Eigenfrequencies and mode shapes (needs a density)."""
    n_modes: int = 6
    fix: object = None

    def deck(self, m: FEModel) -> list[str]:
        L = (["*BOUNDARY"] + _bc_from(self.fix)) if self.fix else []
        return L + ["*STEP", "*FREQUENCY", f"{self.n_modes}", "*NODE FILE", "U", "*END STEP"]


@dataclass
class Buckle:
    """Linear buckling: load factors of the given reference loads, and the buckling shapes."""
    n_modes: int = 3
    fix: object = None
    loads: object = None
    pressure: object = None

    def deck(self, m: FEModel) -> list[str]:
        L = ["*BOUNDARY"] + _bc_from(self.fix) + ["*STEP", "*BUCKLE", f"{self.n_modes}"]
        if self.loads:
            L.append("*CLOAD")
            for nodes, F in _sel(self.loads):
                nodes = np.atleast_1d(nodes)
                for k in range(3):
                    if F[k]:
                        L += [f"{int(n) + 1}, {k + 1}, {F[k] / len(nodes):.10g}" for n in nodes]
        if self.pressure:
            L.append("*DLOAD")
            for (els, fcs), p in _sel(self.pressure):
                L += [f"{int(e) + 1}, P{int(f)}, {p:.10g}" for e, f in zip(els, fcs, strict=True)]
        return L + ["*NODE FILE", "U", "*END STEP"]


@dataclass
class Heat:
    """Steady heat conduction (needs a conductivity). ``temperature``: [(nodes, T)]; ``film``: [((elements, faces),
    (T_sink, h))]; ``flux``: [((elements, faces), q)] W/m^2 into the body."""
    temperature: object = None
    film: object = None
    flux: object = None
    initial: float = 0.0

    def deck(self, m: FEModel) -> list[str]:
        L = ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", f"NALL, {self.initial:.10g}", "*STEP",
             "*HEAT TRANSFER, STEADY STATE", "1., 1."]
        if self.temperature:
            L.append("*BOUNDARY")
            for nodes, T in _sel(self.temperature):
                L += [f"{int(n) + 1}, 11, 11, {T:.10g}" for n in np.atleast_1d(nodes)]
        if self.film:
            L.append("*FILM")
            for (els, fcs), (Ts, h) in _sel(self.film):
                L += [f"{int(e) + 1}, F{int(f)}, {Ts:.10g}, {h:.10g}" for e, f in zip(els, fcs, strict=True)]
        if self.flux:
            L.append("*DFLUX")
            for (els, fcs), q in _sel(self.flux):
                L += [f"{int(e) + 1}, S{int(f)}, {q:.10g}" for e, f in zip(els, fcs, strict=True)]
        return L + ["*NODE FILE", "NT", "*EL FILE", "HFL", "*END STEP"]


@dataclass
class Result:
    model: FEModel
    u: np.ndarray | None = None                     # (n, 3) last increment
    stress: np.ndarray | None = None                # (n, 6) xx yy zz xy yz zx, nodal (CalculiX extrapolated)
    strain: np.ndarray | None = None
    reactions: np.ndarray | None = None
    peeq: np.ndarray | None = None
    temperature: np.ndarray | None = None
    frequencies: np.ndarray | None = None           # Hz
    buckling_factors: np.ndarray | None = None
    modes: list[np.ndarray] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def von_mises(self) -> np.ndarray:
        from pinneapple_simulation.numerical_solvers.solid_fem import von_mises
        return von_mises(self.stress)

    @property
    def max_displacement(self) -> float:
        return float(np.linalg.norm(self.u, axis=1).max())

    @property
    def mesh(self):
        """Duck type of solid_fem.HexMesh for the post-processor figures (boundary polygons, nodes)."""
        m = self.model

        class _M:
            nodes = m.nodes
            elements = m.elements

            @staticmethod
            def boundary_faces():
                return m.outward_faces()
        return _M()


def _read_dat_table(path: str, header: str) -> np.ndarray:
    """Rows of numbers after a CalculiX .dat section header (eigenvalues, buckling factors)."""
    if not os.path.exists(path):
        return np.zeros((0, 0))
    rows, on = [], False
    for ln in open(path, errors="replace"):
        if header in ln:
            on, rows = True, []
            continue
        if on:
            t = ln.split()
            if not t:
                if rows:
                    on = False
                continue
            try:
                rows.append([float(v) for v in t])
            except ValueError:
                if rows:
                    on = False
    return np.array(rows) if rows else np.zeros((0, 0))


def solve(model: FEModel, step, workdir: str, *, threads: int = 2, keep: bool = True) -> Result:
    """Write ``workdir/job.inp``, run ccx, read the results."""
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "job.inp"), "w") as f:
        f.write(model.deck(step))
    run = run_ccx(workdir, "job", threads=threads)
    if run.returncode != 0 or not os.path.exists(run.frd) or "ERROR" in run.stdout.upper().split("JOB FINISHED")[0]:
        raise RuntimeError(f"ccx failed (exit {run.returncode}):\n{run.stdout[-3000:]}")
    fr = read_frd(run.frd)
    order = np.argsort(fr.node_ids)
    n = len(model.nodes)
    res = Result(model, info={"element": model.element, "elements": len(model.elements), "dofs": 3 * n,
                              "solver": "CalculiX " + _version()})

    def grab(st, name, k):
        a = st.get(name)
        return None if a is None else a[order][:n, :k]
    if isinstance(step, Frequency):
        tab = _read_dat_table(run.dat, "E I G E N V A L U E   O U T P U T")
        if tab.size:
            tab = tab[tab.shape[1] >= 4] if tab.ndim == 1 else tab
            res.frequencies = tab[:, 3] if tab.shape[1] >= 4 else np.sqrt(np.abs(tab[:, 1])) / (2 * math.pi)
        res.modes = [grab(st, "DISP", 3) for st in fr.steps if "DISP" in st]
    elif isinstance(step, Buckle):
        tab = _read_dat_table(run.dat, "B U C K L I N G   F A C T O R   O U T P U T")
        res.buckling_factors = tab[:, 1] if tab.size else None
        res.modes = [grab(st, "DISP", 3) for st in fr.steps if "DISP" in st]
    elif isinstance(step, Heat):
        st = [s for s in fr.steps if "NDTEMP" in s][-1]
        res.temperature = grab(st, "NDTEMP", 1)[:, 0]
    else:
        st = [s for s in fr.steps if "DISP" in s][-1]
        res.u = grab(st, "DISP", 3)
        last = [s for s in fr.steps if "STRESS" in s]
        if last:
            res.stress = grab(last[-1], "STRESS", 6)
        last = [s for s in fr.steps if "TOSTRAIN" in s]
        if last:
            res.strain = grab(last[-1], "TOSTRAIN", 6)
        last = [s for s in fr.steps if "FORC" in s]
        if last:
            res.reactions = grab(last[-1], "FORC", 3)
        last = [s for s in fr.steps if "PE" in s]
        if last:
            res.peeq = grab(last[-1], "PE", 1)[:, 0]
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return res


def _version() -> str:
    try:
        p = subprocess.run(["ccx", "-v"], capture_output=True, text=True, timeout=30)
        m = re.search(r"Version\s+([\d.]+)", p.stdout + p.stderr)
        return m.group(1) if m else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ---------------------------------------------------------------------- geometry scripts for gmsh
def geo_plate_with_hole(L: float, W: float, t: float, d: float) -> str:
    """A plate L x W x t centred at the origin (x along the length) with a central through hole of diameter d."""
    return f"""SetFactory("OpenCASCADE");
Box(1) = {{{-L / 2}, {-W / 2}, 0, {L}, {W}, {t}}};
Cylinder(2) = {{0, 0, -1, 0, 0, {t + 2}, {d / 2}}};
BooleanDifference{{ Volume{{1}}; Delete; }}{{ Volume{{2}}; Delete; }}
Physical Volume("part") = {{1}};
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumElementsPerTwoPi = 36;
"""


def geo_l_bracket(a: float = 0.1, b: float = 0.08, w: float = 0.04, t: float = 0.01, hole: float = 0.012,
                  fillet: float = 0.01) -> str:
    """An L bracket: a base plate a x w x t on z = 0 and an upright b x w x t at x = 0, a filleted inner corner and a
    bolt hole in the upright."""
    return f"""SetFactory("OpenCASCADE");
Box(1) = {{0, 0, 0, {a}, {w}, {t}}};
Box(2) = {{0, 0, 0, {t}, {w}, {b}}};
Box(3) = {{{t}, 0, {t}, {fillet}, {w}, {fillet}}};
Cylinder(4) = {{{t + fillet}, -1, {t + fillet}, 0, {w + 2}, 0, {fillet}}};
BooleanDifference(5) = {{ Volume{{3}}; Delete; }}{{ Volume{{4}}; Delete; }};
BooleanUnion(6) = {{ Volume{{1}}; Delete; }}{{ Volume{{2, 5}}; Delete; }};
Cylinder(7) = {{-1, {w / 2}, {0.75 * b}, {t + 2}, 0, 0, {hole / 2}}};
BooleanDifference(8) = {{ Volume{{6}}; Delete; }}{{ Volume{{7}}; Delete; }};
Physical Volume("part") = {{8}};
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumElementsPerTwoPi = 24;
"""
