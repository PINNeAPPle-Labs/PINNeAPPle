"""Interoperability of the machine models: pyleecan, FEMM, MATLAB, DXF and the PINNeAPPle data contracts.

* pyleecan (Apache-2.0, https://github.com/Eomys/pyleecan): :func:`from_pyleecan` reads a machine file
  (``MachineIPMSM`` with a ``HoleM50`` rotor and a ``SlotW11`` stator, e.g. ``Toyota_Prius.json``) into a
  :class:`~.geometry.MachineGeometry` and its winding; :func:`to_pyleecan` writes the parameters back in pyleecan's
  field names (the parametrisations are the same, so the round trip is exact).
* FEMM (https://www.femm.info): :func:`femm_lua` writes a Lua script for FEMM's magnetics module that draws the full
  cross-section (every polygon of the mesher's sector geometry, copied around the circle), assigns the B-H curve,
  magnets (direction per piece), coils (circuits per phase and turns) and runs the analysis. FEMM is not available
  where this package is tested; the script is checked by re-reading its own nodes and segments
  (:func:`read_femm_lua_polygons`).
* MATLAB: :func:`to_mat` / :func:`from_mat` (``scipy.io``) for flux maps, efficiency maps and field solutions.
* DXF: :func:`to_dxf` writes the cross-section as closed polylines on one layer per region (DXF R12, ASCII);
  :func:`read_dxf_polylines` reads such files back.
* ElmerFEM: :func:`elmer_case` writes the gmsh geometry and a MagnetoDynamics2D solver input file (``.sif``)
  (not run here: ElmerFEM was not available).
* gmsh / GetDP: :mod:`.getdp` writes, meshes and solves the machine (used as the independent cross-check).
* Contracts: :func:`rotor_parameter_space` (``ParameterSpace`` of the rotor design variables) and
  :func:`fluxmap_dataset` (a flux map as a table with a ``DatasetSpec``, written by
  :func:`pinneapple_data.contracts.tables.write_table` to CSV, Parquet or HDF5).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..constants import MU0
from .geometry import MachineGeometry, RoundBottomSlots, SectorGeometry, VRotor

__all__ = ["from_pyleecan", "to_pyleecan", "femm_lua", "read_femm_lua_polygons", "to_mat", "from_mat", "to_dxf",
           "read_dxf_polylines", "rotor_parameter_space", "fluxmap_dataset", "cross_section_polygons", "elmer_case"]

TWO_PI = 2.0 * math.pi
_HOLE = ("H0", "H1", "H2", "H3", "W0", "W1", "W2", "W3", "W4")
_SLOT = ("W0", "H0", "H1", "W1", "W2", "H2", "R1")


# --------------------------------------------------------------------------------------------- pyleecan
def from_pyleecan(machine: dict | str | Path) -> tuple[MachineGeometry, dict]:
    """Geometry and winding data of a pyleecan ``MachineIPMSM`` file (or the parsed dict)."""
    d = json.loads(Path(machine).read_text()) if not isinstance(machine, dict) else machine
    rot, sta = d["rotor"], d["stator"]
    hole = rot["hole"][0]
    if hole.get("__class__") != "HoleM50" or sta["slot"].get("__class__") != "SlotW11":
        raise ValueError("only HoleM50 rotors and SlotW11 stators are supported")
    if hole.get("H4", 0.0) not in (0, 0.0):
        raise ValueError("HoleM50 with H4 > 0 is not supported")
    vr = VRotor(poles=2 * int(hole["Zh"]) // 2, r_out=rot["Rext"], r_in=rot["Rint"],
                **{k: float(hole[k]) for k in _HOLE})
    slot = sta["slot"]
    rs = RoundBottomSlots(slots=int(slot["Zs"]), r_in=sta["Rint"], r_out=sta["Rext"],
                          **{k: float(slot[k]) for k in _SLOT})
    g = MachineGeometry(vr, rs, length=float(rot["L1"]), stacking_factor=float(rot.get("Kf1", 1.0)))
    w = sta["winding"]
    winding = {"turns_per_coil": w.get("Ntcoil"), "parallel_paths": w.get("Npcp", 1), "layers": w.get("Nlayer"),
               "coil_pitch": w.get("coil_pitch"), "pole_pairs": w.get("p"), "phases": w.get("qs"),
               "end_overhang": w.get("Lewout")}
    if "wind_mat" in w and w["wind_mat"] is not None:
        wm = np.asarray(w["wind_mat"], float)  # (layers r, layers t, slots, phases)
        winding["conductors_per_slot"] = wm.sum(axis=(0, 1)).tolist()
    return g, winding


def to_pyleecan(geometry: MachineGeometry) -> dict:
    """pyleecan field names and values of the rotor hole, the stator slot and the laminations."""
    r, s = geometry.rotor, geometry.stator
    return {"rotor": {"Rext": r.r_out, "Rint": r.r_in, "L1": geometry.length, "Kf1": geometry.stacking_factor,
                      "hole": [{"__class__": "HoleM50", "Zh": r.poles, "H4": 0.0, **{k: getattr(r, k) for k in _HOLE}}]},
            "stator": {"Rint": s.r_in, "Rext": s.r_out, "L1": geometry.length, "Kf1": geometry.stacking_factor,
                       "slot": {"__class__": "SlotW11", "Zs": s.slots, **{k: getattr(s, k) for k in _SLOT}}}}


# --------------------------------------------------------------------------------------------- polygons
def cross_section_polygons(geometry: MachineGeometry, h: float = 1e-3) -> list[tuple[str, np.ndarray, float | None]]:
    """Every polygon of the full cross-section as ``(region, xy, magnetisation angle)``: pole ``k`` turned by
    ``k * pole pitch`` (magnetisation reversed on odd poles), slot ``k`` by ``k * slot pitch``."""
    out = []
    rs: SectorGeometry = geometry.rotor.sector(h)
    for k in range(geometry.rotor.poles):
        a = k * geometry.rotor.pole_pitch
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        for pg in rs.polygons:
            m = None if pg.magnetization is None else pg.magnetization + a + (math.pi if k % 2 else 0.0)
            name = "magnet" if pg.name.startswith("magnet") else "pocket"
            out.append((f"{name}{k}" if name == "magnet" else name, pg.xy @ R.T, m))
    ss = geometry.stator.sector(h)
    for k in range(geometry.stator.slots):
        a = k * geometry.stator.slot_pitch
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        for pg in ss.polygons:
            out.append((f"slot{k}" if pg.name == "slot" else pg.name, pg.xy @ R.T, None))
    return out


# --------------------------------------------------------------------------------------------- FEMM
def femm_lua(geometry: MachineGeometry, steel, magnet, layout: list, turns_per_slot: float,
             currents=(0.0, 0.0, 0.0), magnet_temperature: float = 20.0, filename: str = "machine.fem",
             h: float = 1e-3) -> str:
    """Lua script for FEMM 4.2 (magnetics, planar, millimetres) of the full machine at the rotor angle of the
    mesher's sector geometry (pole 0 on +x). ``layout``: (phase, sign) per slot; ``currents``: phase currents (A)."""
    L = []
    w = L.append
    w("newdocument(0)")
    w(f'mi_probdef(0, "millimeters", "planar", 1e-8, {geometry.length * 1e3:.6f}, 30)')
    w('mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)')
    w('mi_addmaterial("Coil", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)')
    Hc = magnet.Br(magnet_temperature) / (4e-7 * math.pi * magnet.mu_r)
    w(f'mi_addmaterial("Magnet", {magnet.mu_r}, {magnet.mu_r}, {Hc:.6g}, 0, {1e-6 / magnet.resistivity:.6g}, 0, 0, 1, 0, 0, 0)')
    w(f'mi_addmaterial("{steel.name}", 1, 1, 0, 0, {1e-6 / steel.resistivity:.6g}, {steel.thickness * 1e3:g}, 0, '
      f'{geometry.stacking_factor:g}, 0, 0, 0)')
    for b, hh in zip(steel.B, steel.H, strict=True):
        w(f'mi_addbhpoint("{steel.name}", {b:.6g}, {hh:.6g})')
    for ph, c in zip("ABC", currents, strict=True):
        w(f'mi_addcircprop("{ph}", {c:.9g}, 1)')
    w('mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0)')

    def poly(xy):
        xy = np.asarray(xy) * 1e3
        for x, y in xy:
            w(f"mi_addnode({x:.9f}, {y:.9f})")
        for i in range(len(xy)):
            (x0, y0), (x1, y1) = xy[i], xy[(i + 1) % len(xy)]
            w(f"mi_addsegment({x0:.9f}, {y0:.9f}, {x1:.9f}, {y1:.9f})")

    def circle(r, a0=0.0):
        r *= 1e3
        w(f"mi_addnode({r * math.cos(a0):.9f}, {r * math.sin(a0):.9f})")
        w(f"mi_addnode({-r * math.cos(a0):.9f}, {-r * math.sin(a0):.9f})")
        w(f"mi_addarc({r * math.cos(a0):.9f}, {r * math.sin(a0):.9f}, {-r * math.cos(a0):.9f}, {-r * math.sin(a0):.9f}, 180, 1)")
        w(f"mi_addarc({-r * math.cos(a0):.9f}, {-r * math.sin(a0):.9f}, {r * math.cos(a0):.9f}, {r * math.sin(a0):.9f}, 180, 1)")

    rt, st = geometry.rotor, geometry.stator
    for r in (rt.r_in, rt.r_out, st.r_in, st.r_out):
        circle(r)
    w(f"mi_selectarcsegment({st.r_out * 1e3:.9f}, 0)")
    w(f"mi_selectarcsegment({-st.r_out * 1e3:.9f}, 0)")
    w('mi_setarcsegmentprop(1, "A0", 0, 0)')
    w("mi_clearselected()")
    w(f"mi_selectarcsegment({rt.r_in * 1e3:.9f}, 0)")
    w(f"mi_selectarcsegment({-rt.r_in * 1e3:.9f}, 0)")
    w('mi_setarcsegmentprop(1, "A0", 0, 0)')
    w("mi_clearselected()")
    for name, xy, mag in cross_section_polygons(geometry, h):
        poly(xy)
        c = _interior_point(xy)
        w(f"mi_addblocklabel({c[0] * 1e3:.9f}, {c[1] * 1e3:.9f})")
        w(f"mi_selectlabel({c[0] * 1e3:.9f}, {c[1] * 1e3:.9f})")
        if name.startswith("magnet"):
            w(f'mi_setblockprop("Magnet", 1, 0, "<None>", {math.degrees(mag):.9f}, 0, 0)')
        elif name.startswith("slot"):
            k = int(name[4:])
            ph, sg = layout[k]
            w(f'mi_setblockprop("Coil", 1, 0, "{"ABC"[ph]}", 0, 0, {sg * turns_per_slot:g})')
        else:
            w('mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)')
        w("mi_clearselected()")
    for x, y, mat in ((0.5 * (rt.r_in + rt.r_out) * math.cos(math.pi / rt.poles) * 0.9, 0.0, steel.name),
                      (0.5 * (st.r_in + st.r_out) * 0 + st.r_out - 2e-3, 0.0, steel.name),
                      (0.5 * (rt.r_out + st.r_in), 0.0, "Air")):
        w(f"mi_addblocklabel({x * 1e3:.9f}, {y * 1e3:.9f})")
        w(f"mi_selectlabel({x * 1e3:.9f}, {y * 1e3:.9f})")
        w(f'mi_setblockprop("{mat}", 1, 0, "<None>", 0, 0, 0)')
        w("mi_clearselected()")
    w(f'mi_saveas("{filename}")')
    w("mi_analyze(1)")
    w("mi_loadsolution()")
    return "\n".join(L) + "\n"


def _interior_point(xy: np.ndarray) -> np.ndarray:
    """A point strictly inside a simple polygon: the centroid if inside, else the midpoint of a short chord."""
    from matplotlib.path import Path

    p = Path(xy)
    c = xy.mean(0)
    if p.contains_point(c):
        return c
    for i in range(len(xy)):
        for j in range(i + 2, len(xy)):
            m = 0.5 * (xy[i] + xy[j])
            if p.contains_point(m):
                return m
    raise ValueError("no interior point found")


def read_femm_lua_polygons(script: str) -> list[np.ndarray]:
    """Closed polygons (metres) traced by consecutive ``mi_addsegment`` calls of a script from :func:`femm_lua`."""
    polys, cur = [], []
    for line in script.splitlines():
        if line.startswith("mi_addsegment("):
            x0, y0, x1, y1 = (float(v) for v in line[len("mi_addsegment("):-1].split(","))
            cur.append((x0, y0))
            if cur and abs(x1 - cur[0][0]) < 1e-9 and abs(y1 - cur[0][1]) < 1e-9:
                polys.append(np.array(cur) * 1e-3)
                cur = []
    return polys


# --------------------------------------------------------------------------------------------- MATLAB
def to_mat(path, **arrays) -> Path:
    """Write named arrays (or nested dicts of arrays) to a MATLAB ``.mat`` file (``scipy.io.savemat``)."""
    from scipy.io import savemat

    savemat(str(path), {k: (v if isinstance(v, dict) else np.asarray(v)) for k, v in arrays.items()},
            do_compression=True)
    return Path(path)


def from_mat(path) -> dict:
    from scipy.io import loadmat

    return {k: v for k, v in loadmat(str(path), squeeze_me=True).items() if not k.startswith("__")}


# --------------------------------------------------------------------------------------------- DXF
def to_dxf(path, geometry: MachineGeometry, h: float = 1e-3, unit_scale: float = 1e3) -> Path:
    """Cross-section as closed POLYLINE entities (DXF R12, ASCII, millimetres by default), one layer per region kind
    (MAGNET, POCKET, SLOT, OPENING, WEDGE) plus the four lamination circles on layer LAMINATION."""
    out = ["0", "SECTION", "2", "ENTITIES"]

    def polyline(layer, xy):
        out.extend(["0", "POLYLINE", "8", layer, "66", "1", "70", "1"])
        for x, y in np.asarray(xy) * unit_scale:
            out.extend(["0", "VERTEX", "8", layer, "10", f"{x:.9f}", "20", f"{y:.9f}"])
        out.extend(["0", "SEQEND"])

    for name, xy, _ in cross_section_polygons(geometry, h):
        layer = "".join(c for c in name if c.isalpha()).upper()
        polyline(layer, xy)
    for r in (geometry.rotor.r_in, geometry.rotor.r_out, geometry.stator.r_in, geometry.stator.r_out):
        out.extend(["0", "CIRCLE", "8", "LAMINATION", "10", "0.0", "20", "0.0", "40", f"{r * unit_scale:.9f}"])
    out.extend(["0", "ENDSEC", "0", "EOF"])
    path = Path(path)
    path.write_text("\n".join(out) + "\n")
    return path


def read_dxf_polylines(path, unit_scale: float = 1e3) -> dict[str, list[np.ndarray]]:
    """Closed POLYLINEs of a DXF R12 file per layer (coordinates divided by ``unit_scale``)."""
    tok = Path(path).read_text().split("\n")
    pairs = list(zip(tok[0::2], tok[1::2], strict=False))
    out: dict[str, list[np.ndarray]] = {}
    i = 0
    while i < len(pairs):
        code, val = pairs[i]
        if code.strip() == "0" and val.strip() == "POLYLINE":
            layer, pts = "", []
            i += 1
            while not (pairs[i][0].strip() == "0" and pairs[i][1].strip() == "SEQEND"):
                c, v = pairs[i][0].strip(), pairs[i][1].strip()
                if c == "8" and not layer:
                    layer = v
                if c == "10":
                    x = float(v)
                if c == "20":
                    pts.append((x, float(v)))
                i += 1
            out.setdefault(layer, []).append(np.array(pts) / unit_scale)
        i += 1
    return out


# --------------------------------------------------------------------------------------------- contracts
def rotor_parameter_space(name: str = "v-ipm-rotor"):
    """``ParameterSpace`` of the rotor design variables (:data:`~.geometry.ROTOR_VARIABLES`, metres)."""
    from pinneapple_data.contracts.models import Parameter, ParameterSpace

    from .geometry import ROTOR_VARIABLES

    desc = {"W4": "magnet width", "H3": "magnet thickness", "H0": "depth of the V apex", "W0": "chord of the V",
            "H1": "tangential bridge"}
    return ParameterSpace(name=name, parameters=[Parameter(name=n, low=lo, high=hi, unit="m", description=desc[n])
                                                 for n, lo, hi in ROTOR_VARIABLES])


def fluxmap_dataset(fm, name: str = "ipm-flux-map"):
    """A flux map (attributes ``i_d``, ``i_q`` grids and ``psi_d``, ``psi_q``, ``T`` tables, e.g. a drive ``FluxMap``)
    as a table (pandas DataFrame: i_d, i_q, psi_d, psi_q, torque) and its ``DatasetSpec``."""
    import pandas as pd

    from pinneapple_data.contracts.models import Column, DatasetSpec

    D, Q = np.meshgrid(fm.i_d, fm.i_q, indexing="ij")
    df = pd.DataFrame({"i_d": D.ravel(), "i_q": Q.ravel(), "psi_d": fm.psi_d.ravel(), "psi_q": fm.psi_q.ravel(),
                       "torque": fm.T.ravel()})
    cols = [Column(name="i_d", role="input", unit="A"), Column(name="i_q", role="input", unit="A"),
            Column(name="psi_d", role="output", unit="Wb"), Column(name="psi_q", role="output", unit="Wb"),
            Column(name="torque", role="output", unit="N*m")]
    return df, DatasetSpec(name=name, columns=cols, n_rows=len(df),
                           description="mean torque and dq flux linkages of the 2-D FEA on a current grid")


# --------------------------------------------------------------------------------------------- Elmer
def elmer_case(directory, geometry: MachineGeometry, steel, magnet, layout: list, turns_per_slot: float,
               currents=(0.0, 0.0, 0.0), theta: float = 0.0, magnet_temperature: float = 20.0,
               n_gap: int = 960) -> dict:
    """An ElmerFEM case (MagnetoDynamics2D, steady) of the machine at rotor angle ``theta``: ``machine.geo`` (gmsh,
    the same geometry as the GetDP cross-check) and ``case.sif``. Mesh it with ``gmsh -2 machine.geo -format msh22``
    and convert with ``ElmerGrid 14 2 machine.msh -autoclean``; body numbers in the SIF are the gmsh physical surface
    numbers in order. Written to the documented SIF keywords; ElmerFEM was not available where this was tested, so
    the case itself has not been run."""
    from .getdp import write_geo

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    info = write_geo(d / "machine.geo", geometry, theta, n_gap)
    curve = steel.curve(geometry.stacking_factor)
    phys = [1, 2, 3, 4] + [100 + k for k in sorted(info["magnets"])] + [1000 + k for k in sorted(info["slot_areas"])]
    body = {p: i + 1 for i, p in enumerate(phys)}
    Br = magnet.Br(magnet_temperature)
    s = ["Header", '  Mesh DB "." "machine"', "End", "",
         "Simulation", "  Coordinate System = Cartesian 2D", "  Simulation Type = Steady state",
         "  Steady State Max Iterations = 1", "End", "",
         "Constants", "  Permeability of Vacuum = 1.25663706212e-6", "End", "",
         "Equation 1", "  Active Solvers(1) = 1", "End", "",
         "Solver 1", "  Equation = MgDyn2D", '  Procedure = "MagnetoDynamics2D" "MagnetoDynamics2D"',
         "  Variable = Potential", "  Nonlinear System Max Iterations = 60",
         "  Nonlinear System Convergence Tolerance = 1e-9", "  Nonlinear System Newton After Iterations = 3",
         "  Linear System Solver = Direct", "  Linear System Direct Method = UMFPack", "End", "",
         "Material 1  ! air", "  Relative Permeability = 1.0", "End", "",
         f"Material 2  ! {steel.name}, stacking factor {geometry.stacking_factor}",
         "  H-B Curve = Variable dummy", "    Real Cubic", "      0.0 0.0"]
    for b, h in zip(curve.B[1:], curve.H[1:], strict=True):
        s.append(f"      {b:.8g} {h:.8g}")
    s += ["    End", "End", ""]
    mats = {}
    for k, ang in sorted(info["magnets"].items()):
        mats[k] = 3 + k
        Hc = Br / (MU0 * magnet.mu_r)
        s += [f"Material {3 + k}  ! magnet {k}", f"  Relative Permeability = {magnet.mu_r}",
              f"  Magnetization 1 = Real {Hc * math.cos(ang):.10g}", f"  Magnetization 2 = Real {Hc * math.sin(ang):.10g}",
              "End", ""]
    bf = {}
    for k, area in sorted(info["slot_areas"].items()):
        ph, sg = layout[k]
        bf[k] = len(bf) + 1
        s += [f"Body Force {bf[k]}  ! slot {k}, phase {'ABC'[ph]}",
              f"  Current Density = {sg * turns_per_slot * currents[ph] / area:.10g}", "End", ""]
    for p in phys:
        if p in (1, 2):
            mat, extra = 2, []
        elif p in (3, 4):
            mat, extra = 1, []
        elif p >= 1000:
            mat, extra = 1, [f"  Body Force = {bf[p - 1000]}"]
        else:
            mat, extra = mats[p - 100], []
        s += [f"Body {body[p]}  ! gmsh physical surface {p}", "  Equation = 1", f"  Material = {mat}", *extra, "End", ""]
    s += ["Boundary Condition 1  ! shaft and outer circles (gmsh physical curve 10)", "  Target Boundaries(1) = 1",
          "  Potential = Real 0", "End"]
    (d / "case.sif").write_text("\n".join(s) + "\n")
    return {"bodies": body, "files": [str(d / "machine.geo"), str(d / "case.sif")]}

