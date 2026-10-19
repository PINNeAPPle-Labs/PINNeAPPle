"""A liquid jet falling into an open trough: 2-D centre-plane interFoam (VOF) case, run and post-processing.

    python trough_case.py CASE_DIR [--speed 2.5] [--angle 60] [--cell 0.005] [--end 8] [--procs 1] [--run]
    python trough_case.py CASE_DIR --post-only           # re-run the post-processing of a finished case

The case is written with ``pp.cfd.FreeSurfaceFlow`` (pinneapple_simulation.numerical_solvers.free_surface) on a
blockMesh of the trough's centre plane (x along the trough, z up). The jet enters through a slot in the top of the
domain at the nozzle exit; the slot width is d / sin(angle), so the jet thickness normal to its axis equals the
nozzle bore d. Water fills the trough up to the overflow sill at t = 0; liquid leaves over the sill (outlet) and,
if it splashes high enough, through the open top (atmosphere).

Post-processing (written to CASE_DIR/post/): liquid-volume balance, alpha bounds and Courant numbers from the
solver log, the jet impingement point on the floor, splash height, floor wall shear stress, time-mean turbulent
kinetic energy on sampling lines, free-surface snapshots.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if os.path.isdir(os.path.join(ROOT, "pinneapple_simulation")):
    sys.path.insert(0, ROOT)
from pinneapple_data.cae import foam  # noqa: E402
from pinneapple_simulation.numerical_solvers.free_surface import (  # noqa: E402
    FreeSurfaceFlow,
    Phase,
    SampleLine,
    cell_centres,
    read_profiles,
)

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")

# Water and air at 20 C, 1 atm (NIST Chemistry WebBook, SRD 69; surface tension: IAPWS 2014 release)
WATER = Phase("water", rho=998.2, nu=1.0034e-6)
AIR = Phase("air", rho=1.204, nu=1.516e-5)
SIGMA = 0.0728


@dataclass
class Trough:
    # geometry (m), trough interior; x = 0 at the upstream end wall, z = 0 on the floor
    length: float = 1.20          # floor length between the upstream wall and the overflow sill
    wall_height: float = 0.25     # side and upstream wall height
    sill_height: float = 0.08     # downstream overflow sill (full width)
    width: float = 0.15           # interior width (3-D CAD only; the CFD is the centre plane)
    nozzle_x: float = 0.30        # nozzle exit centre, distance from the upstream wall
    nozzle_z: float = 0.40        # nozzle exit centre, height above the floor (= top of the CFD domain)
    nozzle_bore: float = 0.020    # nozzle inner diameter d
    # jet
    speed: float = 2.5            # mean exit velocity, m/s
    angle: float = 60.0           # below the horizontal, pointing downstream, deg
    # numerics
    cell: float = 0.005           # mesh cell size, m
    turbulence: str = "kOmegaSST"
    intensity: float = 0.05       # jet inlet turbulence intensity
    end_time: float = 12.0        # s of physical time
    write_every: float = 0.02     # s between field writes (balance and line sampling use every write)
    max_co: float = 0.5

    @property
    def slot(self) -> float:
        return self.nozzle_bore / math.sin(math.radians(self.angle))

    @property
    def velocity(self) -> tuple:
        a = math.radians(self.angle)
        return (self.speed * math.cos(a), 0.0, -self.speed * math.sin(a))

    @property
    def q_in(self) -> float:
        """Liquid inflow per metre of span, m^2/s (slot width x vertical velocity = bore x speed)."""
        return self.slot * self.speed * math.sin(math.radians(self.angle))

    # ------------------------------------------------------------------ mesh
    def blockmesh(self, depth: float) -> str:
        L, H, hs, zt = self.length, self.wall_height, self.sill_height, self.nozzle_z
        x0, x1 = self.nozzle_x - self.slot / 2, self.nozzle_x + self.slot / 2
        xs, zs = [0.0, x0, x1, L], [0.0, hs, H, zt]
        pts = [(x, y, z) for y in (0.0, depth) for z in zs for x in xs]
        idx = lambda iy, iz, ix: (iy * 4 + iz) * 4 + ix                 # noqa: E731
        n = lambda a, b: max(2, round((b - a) / self.cell))              # noqa: E731
        blocks, faces = [], {k: [] for k in ("floor", "upstreamWall", "sill", "outlet", "atmosphere", "nozzle",
                                             "frontAndBack")}
        for iz in range(3):
            for ix in range(3):
                v = [idx(0, iz, ix), idx(0, iz, ix + 1), idx(1, iz, ix + 1), idx(1, iz, ix),
                     idx(0, iz + 1, ix), idx(0, iz + 1, ix + 1), idx(1, iz + 1, ix + 1), idx(1, iz + 1, ix)]
                blocks.append(f"    hex ({' '.join(map(str, v))}) ({n(xs[ix], xs[ix + 1])} 1 "
                              f"{n(zs[iz], zs[iz + 1])}) simpleGrading (1 1 1)")
                faces["frontAndBack"] += [f"({v[0]} {v[1]} {v[5]} {v[4]})", f"({v[3]} {v[7]} {v[6]} {v[2]})"]
                if iz == 0:
                    faces["floor"].append(f"({v[0]} {v[3]} {v[2]} {v[1]})")
                if iz == 2:
                    faces["nozzle" if ix == 1 else "atmosphere"].append(f"({v[4]} {v[5]} {v[6]} {v[7]})")
                if ix == 0:
                    faces["upstreamWall" if iz < 2 else "atmosphere"].append(f"({v[0]} {v[4]} {v[7]} {v[3]})")
                if ix == 2:
                    faces["sill" if iz == 0 else "outlet"].append(f"({v[1]} {v[2]} {v[6]} {v[5]})")
        types = {"floor": "wall", "upstreamWall": "wall", "sill": "wall", "outlet": "patch", "atmosphere": "patch",
                 "nozzle": "patch", "frontAndBack": "empty"}
        bnd = "\n".join(f"    {k} {{ type {types[k]}; faces ( {' '.join(f)} ); }}" for k, f in faces.items())
        vs = "\n".join(f"    ({x:.6g} {y:.6g} {z:.6g})" for x, y, z in pts)
        return ("FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
                f"convertToMeters 1;\nvertices\n(\n{vs}\n);\nblocks\n(\n" + "\n".join(blocks) +
                f"\n);\nedges ();\nboundary\n(\n{bnd}\n);\nmergePatchPairs ();\n")

    def lines(self) -> list[SampleLine]:
        L, h = self.length, self.cell
        out = [SampleLine(f"floor_z{int(round(z * 1000)):03d}mm", (h / 2, 0.0, z), (L - h / 2, 0.0, z), 241)
               for z in (0.01, 0.03, 0.06)]
        out += [SampleLine(f"vertical_x{int(round(x * 1000)):04d}mm", (x, 0.0, h / 2), (x, 0.0, self.wall_height), 51)
                for x in (0.45, 0.80, 1.10)]
        return out


def sh(case: str, cmd: str, name: str) -> None:
    r = subprocess.run(["bash", "-c", f'source "{FOAM_BASHRC}" >/dev/null 2>&1; cd "{case}" && {cmd} > log.{name} 2>&1'])
    if r.returncode:
        raise RuntimeError(f"{name} failed, see {case}/log.{name}")


def write_case(case: str, t: Trough, procs: int = 1) -> FreeSurfaceFlow:
    depth = t.cell                                       # one cell thick, empty front/back
    mesh = os.path.join(case, "mesh")
    os.makedirs(os.path.join(mesh, "system"), exist_ok=True)
    os.makedirs(os.path.join(mesh, "constant"), exist_ok=True)
    open(os.path.join(mesh, "system", "blockMeshDict"), "w").write(t.blockmesh(depth))
    open(os.path.join(mesh, "system", "controlDict"), "w").write(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\n"
        "writeControl timeStep; writeInterval 1;\n")
    sh(mesh, "blockMesh", "blockMesh")
    flow = FreeSurfaceFlow(
        mesh=mesh, two_d=True,
        roles={"nozzle": "inlet", "outlet": "outlet", "atmosphere": "atmosphere"},   # floor, walls, sill: walls
        inlet_velocity={"nozzle": t.velocity}, level=t.sill_height,
        liquid=WATER, gas=AIR, sigma=SIGMA, gravity=(0.0, 0.0, -9.81), turbulence=t.turbulence,
        intensity=t.intensity, mixing_length=0.07 * t.nozzle_bore,
        end_time=t.end_time, delta_t=1e-4, max_co=t.max_co, max_alpha_co=t.max_co, max_delta_t=0.002,
        sample_every=t.write_every, lines=t.lines())
    run_dir = os.path.join(case, "run")
    info = flow.write(run_dir, procs=procs)
    with open(os.path.join(case, "trough.json"), "w") as f:
        json.dump({"parameters": asdict(t), "derived": {"slot_width": t.slot, "q_in_m2_s": t.q_in,
                                                        "inlet_velocity": t.velocity, "span_cfd": depth},
                   "case": info}, f, indent=1)
    sh(run_dir, "checkMesh", "checkMesh")
    return flow


# ------------------------------------------------------------------------------------------ post-processing
def _boundary_values(path: str, patch: str, n_faces: int) -> np.ndarray:
    """Values of one patch (n_faces faces) in an ASCII field file (``value uniform x`` or ``nonuniform List<...>``)."""
    txt = open(path).read()
    i = txt.index("boundaryField")
    m = re.search(r"\n\s*" + re.escape(patch) + r"\s*\{", txt[i:])
    body = txt[i + m.end():]
    body = body[:body.index("\n    }")]
    m = re.search(r"\bvalue\s+uniform\s+(\([^)]*\)|[^;]+);", body)
    if m:
        v = np.array(m.group(1).strip("() ").split(), float)
        return np.full(n_faces, v[0]) if len(v) == 1 else np.tile(v, (n_faces, 1))
    m = re.search(r"\bvalue\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", body)
    if not m:
        return np.zeros(n_faces)
    n, rest = int(m.group(2)), body[m.end():]
    vals = rest[:rest.index("\n)")] if "\n)" in rest[:rest.index(";")] else rest[:rest.index(")")]
    if m.group(1) == "vector":
        return np.array(vals.replace("(", " ").replace(")", " ").split(), float).reshape(n, 3)
    return np.array(vals.split()[:n], float)


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _internal(path: str, n: int) -> np.ndarray:
    """internalField of a volume field: (n,) for scalars, (n, 3) for vectors."""
    return foam.read_field(_read(path), n)[0]


def _patch_faces(poly: str, patch: str) -> tuple[int, int]:
    """(nFaces, startFace) of a patch of the polyMesh."""
    b = next(b for b in foam.read_boundary(_read(os.path.join(poly, "boundary"))) if b["name"] == patch)
    return b["nFaces"], b["startFace"]


def _times(run: str) -> list[tuple[float, str]]:
    out = []
    for d in os.listdir(run):
        try:
            out.append((float(d), d))
        except ValueError:
            pass
    return sorted(out)


def parse_log(run: str) -> dict:
    """Every time step of log.interFoam and of the restart logs (log.interFoam.restart*), in order."""
    logs = sorted((f for f in os.listdir(run) if f.startswith("log.interFoam")), key=lambda f: (len(f), f))
    txt = "".join(open(os.path.join(run, f)).read() for f in logs)
    steps = txt.split("\nTime = ")[1:]
    rows = []
    for s in steps:
        t = float(s.split("\n", 1)[0])
        co = re.search(r"\nCourant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)", s)
        ico = re.search(r"Interface Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)", s)
        dt = re.search(r"deltaT = ([0-9.eE+-]+)", s)
        ph = re.findall(r"Phase-1 volume fraction = ([0-9.eE+-]+)\s+Min\([\w.]+\) = ([0-9.eE+-]+)\s+"
                        r"Max\([\w.]+\) = ([0-9.eE+-]+)", s)
        if ph:
            rows.append([t, float(dt.group(1)) if dt else np.nan, float(co.group(2)) if co else np.nan,
                         float(ico.group(2)) if ico else np.nan, *map(float, ph[-1])])
    return {"t": np.array([r[0] for r in rows]), "dt": np.array([r[1] for r in rows]),
            "co_max": np.array([r[2] for r in rows]), "alpha_co_max": np.array([r[3] for r in rows]),
            "volume_fraction": np.array([r[4] for r in rows]), "alpha_min": np.array([r[5] for r in rows]),
            "alpha_max": np.array([r[6] for r in rows]), "finished": "\nEnd\n" in txt[-200:]}


def postprocess(case: str, out: str | None = None, t_avg: tuple[float, float] | None = None) -> dict:
    """Balance, bounds, Courant, impingement, splash, floor shear, mean k on lines; JSON + CSV + figures."""
    meta = json.load(open(os.path.join(case, "trough.json")))
    p = meta["parameters"]
    t = Trough(**p)
    run = os.path.join(case, "run")
    out = out or os.path.join(case, "post")
    os.makedirs(out, exist_ok=True)
    poly = os.path.join(run, "constant", "polyMesh")
    cc = cell_centres(poly)
    nc = len(cc)
    depth = meta["derived"]["span_cfd"]
    dom_area = t.length * t.nozzle_z                           # domain area in the x-z plane
    log = parse_log(run)
    times = [(tv, s) for tv, s in _times(run) if tv > 0 or s == "0"]
    a_name = "alpha.water"

    # ---- liquid volume and boundary fluxes at every written time (per metre of span)
    nf_floor, s_floor = _patch_faces(poly, "floor")
    rec = []
    for tv, s in times:
        d = os.path.join(run, s)
        if not os.path.exists(os.path.join(d, a_name)):
            continue
        alpha = _internal(os.path.join(d, a_name), nc)
        row = {"t": tv, "V_liquid": float(alpha.mean() * dom_area)}       # uniform-volume cells? checked below
        if tv > 0 and os.path.exists(os.path.join(d, "phi")):
            for patch in ("outlet", "atmosphere", "nozzle"):
                nfp = _patch_faces(poly, patch)[0]
                ph = _boundary_values(os.path.join(d, "phi"), patch, nfp)
                al = _boundary_values(os.path.join(d, a_name), patch, nfp)
                row[f"Q_{patch}"] = float(np.sum(np.asarray(ph) * np.asarray(al))) / depth
        rec.append(row)
    # the cells are not all the same size (block-wise sizes): weight by the cell area from the mesh
    areas = _cell_areas(poly, nc)
    for r in rec:
        alpha = _internal(os.path.join(run, _tname(times, r["t"]), a_name), nc)
        r["V_liquid"] = float((alpha * areas).sum())
    T = np.array([r["t"] for r in rec])
    V = np.array([r["V_liquid"] for r in rec])
    q_in = t.q_in
    q_out = np.array([r.get("Q_outlet", 0.0) for r in rec])
    q_atm = np.array([r.get("Q_atmosphere", 0.0) for r in rec])
    q_noz = np.array([r.get("Q_nozzle", -q_in) for r in rec])
    q_out[0], q_atm[0] = q_out[min(1, len(q_out) - 1)] * 0, 0.0
    net = (-q_noz) - q_out - q_atm                               # into the domain (phi > 0 leaves)
    net[0] = q_in
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * np.diff(T))])
    cum_in = np.concatenate([[0.0], np.cumsum(0.5 * ((-q_noz)[1:] + (-q_noz)[:-1]) * np.diff(T))])
    cum_in[0] = 0.0
    err = (V - V[0]) - cum
    rel = err / np.maximum(cum_in, 1e-12)
    # independent: the solver's own domain-average liquid fraction at every time step
    V_log = log["volume_fraction"] * float(areas.sum())
    balance = {"V0_m2": float(V[0]), "V_end_m2": float(V[-1]), "inflow_total_m2": float(cum_in[-1]),
               "outflow_outlet_m2": float(np.trapezoid(q_out, T)), "outflow_atmosphere_m2": float(np.trapezoid(q_atm, T)),
               "imbalance_end_m2": float(err[-1]), "imbalance_end_pct_of_inflow": float(100 * rel[-1]),
               "imbalance_max_abs_pct_of_inflow_after_1s": float(100 * np.abs(rel[T >= 1.0]).max()) if (T >= 1).any() else None,
               "q_in_m2_s": q_in, "q_in_nozzle_patch_mean_m2_s": float(np.mean(-q_noz[1:])) if len(q_noz) > 1 else None,
               "note": "per metre of span; fluxes from the written phi and alpha boundary values at every write "
                       f"({t.write_every} s), trapezoidal in time"}
    # ---- solver checks
    checks = {"alpha_min": float(log["alpha_min"].min()), "alpha_max": float(log["alpha_max"].max()),
              "courant_max": float(np.nanmax(log["co_max"])), "interface_courant_max": float(np.nanmax(log["alpha_co_max"])),
              "courant_max_after_0p1s": float(np.nanmax(log["co_max"][log["t"] > 0.1])),
              "dt_min": float(np.nanmin(log["dt"])), "dt_max": float(np.nanmax(log["dt"])), "steps": int(len(log["t"])),
              "t_end": float(log["t"][-1]), "finished": bool(log["finished"]),
              "liquid_volume_solver_vs_fields_max_rel_diff": None}
    if len(V_log):
        vi = np.interp(T, log["t"], V_log)
        checks["liquid_volume_solver_vs_fields_max_rel_diff"] = float(np.max(np.abs(vi[1:] - V[1:]) / V[1:])) if len(T) > 1 else None

    # ---- time window for averages: the second half of the run, unless given
    t0, t1 = t_avg or (0.5 * T[-1], T[-1])
    sel = [(tv, s) for tv, s in times if t0 - 1e-9 <= tv <= t1 + 1e-9]
    # ---- floor: wall shear, pressure, depth; impingement point
    fl = _floor_cells(poly, nf_floor, s_floor)
    xf = cc[fl["owner"], 0]
    order = np.argsort(xf)
    tau_list, p_list, depth_list = [], [], []
    for _tv, s in sel:
        d = os.path.join(run, s)
        U = _internal(os.path.join(d, "U"), nc)
        alpha = _internal(os.path.join(d, a_name), nc)
        nut_w = _boundary_values(os.path.join(d, "nut"), "floor", nf_floor) if os.path.exists(os.path.join(d, "nut")) else 0.0
        a_c = alpha[fl["owner"]]
        rho = a_c * WATER.rho + (1 - a_c) * AIR.rho
        nu = a_c * WATER.nu + (1 - a_c) * AIR.nu
        ut = U[fl["owner"], 0]                                  # floor-parallel velocity of the first cell
        tau = rho * (nu + np.asarray(nut_w)) * ut / fl["y"]     # wall-function shear (signed, + downstream)
        tau_list.append(tau)
        p_list.append(_internal(os.path.join(d, "p"), nc)[fl["owner"]])
        depth_list.append(_column_depth(cc, alpha, areas, t))
    half = len(tau_list) // 2                                   # steadiness: first vs second half of the window
    halves = {"tau_mean_abs_max_Pa": [float(np.mean(np.abs(tau_list[:half]), 0).max()),
                                      float(np.mean(np.abs(tau_list[half:]), 0).max())]}
    tau_m = np.mean(tau_list, 0)
    tau_abs = np.mean(np.abs(tau_list), 0)
    tau_max_inst = np.max(np.abs(tau_list), 0)
    p_m = np.mean(p_list, 0)
    xs = xf[order]
    i_stag = int(np.argmax(p_m[order] - np.interp(xs, *_hydro_ref(xs, depth_list, t))))
    # ballistic estimate of where the jet axis meets the mean free surface
    h_up = float(np.mean([dl[1][(dl[0] > 0.05) & (dl[0] < t.nozzle_x - 0.05)].mean() for dl in depth_list]))
    vx, _, vz = t.velocity
    dz = t.nozzle_z - h_up
    tt = (vz + math.sqrt(vz * vz + 2 * 9.81 * dz)) / 9.81 if dz > 0 else 0.0
    x_ball = t.nozzle_x + vx * tt
    sign_change = np.where(np.diff(np.sign(tau_m[order])) != 0)[0]
    imp = {"x_floor_pressure_peak_m": float(xs[i_stag]),
           "x_floor_shear_reversal_m": [float(xs[k]) for k in sign_change[:4]],
           "x_ballistic_at_mean_free_surface_m": x_ball, "mean_depth_upstream_of_jet_m": h_up,
           "jet_speed_at_free_surface_ballistic_m_s": math.sqrt(t.speed ** 2 + 2 * 9.81 * max(dz, 0)),
           "floor_tau_mean_abs_max_Pa": float(tau_abs.max()), "x_floor_tau_mean_abs_max_m": float(xf[np.argmax(tau_abs)]),
           "floor_tau_instantaneous_max_Pa": float(tau_max_inst.max()), "averaging_window_s": [float(t0), float(t1)],
           "floor_y_plus_of_first_cell_max": float((np.sqrt(tau_abs / WATER.rho) * fl["y"] / WATER.nu).max()),
           "floor_first_cell_centre_height_m": float(fl["y"].mean()),
           "window_halves": halves}
    # ---- splash: highest liquid (alpha > 0.5) away from the jet column, per write
    spl = []
    for tv, s in times[1:]:
        alpha = _internal(os.path.join(run, s, a_name), nc)
        wet = alpha > 0.5
        xj = t.nozzle_x + (t.nozzle_z - cc[:, 2]) / math.tan(math.radians(t.angle))    # jet axis (straight)
        away = np.abs(cc[:, 0] - xj) > 3 * t.slot
        zmax = float(cc[wet & away, 2].max()) if (wet & away).any() else 0.0
        spl.append((tv, zmax))
    spl = np.array(spl)
    splash = {"max_liquid_height_away_from_jet_m": float(spl[:, 1].max()),
              "time_of_max_s": float(spl[np.argmax(spl[:, 1]), 0]),
              "mean_over_window_m": float(spl[(spl[:, 0] >= t0), 1].mean()),
              "wall_height_m": t.wall_height,
              "overtops_wall": bool(spl[:, 1].max() > t.wall_height),
              "liquid_out_through_open_top_m2": balance["outflow_atmosphere_m2"]}
    # ---- k on lines (time mean over the window)
    tm = 0.5 * (t0 + t1)
    for _nm, (a0, a1) in (("first", (t0, tm)), ("second", (tm, t1))):
        ph = read_profiles(run, None, a0, a1)
        for ln, d in ph.items():
            if ln.startswith("floor") and "k" in d:
                halves.setdefault(f"mean_k_{ln}", []).append(float(np.mean(d["mean_k"])))
    for v in halves.values():
        v.append(abs(v[1] - v[0]) / max(abs(v[0]), abs(v[1]), 1e-12) * 100)       # relative difference, %
    prof = read_profiles(run, None, t0, t1)
    kl = {}
    for name, d in prof.items():
        if "k" not in d:
            continue
        kl[name] = {"s": d["s"].tolist(), "xyz": d["xyz"].tolist(), "mean_k": d["mean_k"].tolist(),
                    "mean_alpha": d["mean_alpha.water"].tolist(), "n_times": int(len(d["times"]))}
    res = {"parameters": p, "derived": meta["derived"], "mass_balance": balance, "solver_checks": checks,
           "impingement": imp, "splash": splash,
           "k_lines_summary": {n: {"mean_k_max": float(np.max(v["mean_k"])),
                                   "mean_k_avg_in_liquid": _mean_wet(v)}
                               for n, v in kl.items()}}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out, "k_lines_mean.json"), "w") as f:
        json.dump(kl, f)
    _csv(os.path.join(out, "mass_balance.csv"), ["t_s", "V_liquid_m2", "q_nozzle_in_m2_s", "q_outlet_m2_s",
                                                 "q_atmosphere_m2_s", "cum_net_in_m2", "imbalance_m2"],
         np.c_[T, V, -q_noz, q_out, q_atm, cum, err])
    _csv(os.path.join(out, "solver_log.csv"), ["t_s", "dt_s", "courant_max", "interface_courant_max",
                                               "liquid_volume_fraction", "alpha_min", "alpha_max"],
         np.c_[log["t"], log["dt"], log["co_max"], log["alpha_co_max"], log["volume_fraction"], log["alpha_min"],
               log["alpha_max"]][::10])
    _csv(os.path.join(out, "floor.csv"), ["x_m", "tau_mean_Pa", "tau_mean_abs_Pa", "tau_max_abs_Pa", "p_mean_Pa"],
         np.c_[xs, tau_m[order], tau_abs[order], tau_max_inst[order], p_m[order]])
    _csv(os.path.join(out, "splash.csv"), ["t_s", "max_liquid_height_away_from_jet_m"], spl)
    figures(case, out, t, times, cc, areas, res, kl, T, V, cum, log, xs, tau_m[order], tau_abs[order], spl)
    return res


def _mean_wet(v: dict) -> float | None:
    k = [k for k, a in zip(v["mean_k"], v["mean_alpha"], strict=False) if a > 0.5]
    return float(np.mean(k)) if k else None


def _tname(times, tv):
    return min(times, key=lambda x: abs(x[0] - tv))[1]


def _csv(path, head, A):
    np.savetxt(path, np.asarray(A, float), delimiter=",", header=",".join(head), comments="", fmt="%.8g")


def _cell_areas(poly: str, nc: int) -> np.ndarray:
    """x-z area of each cell of the one-cell-thick 2-D mesh (front faces have y = 0)."""
    P = foam.read_points(_read(os.path.join(poly, "points")))
    flat, offsets = foam.read_faces(_read(os.path.join(poly, "faces")))
    own = foam.read_labels(_read(os.path.join(poly, "owner")))
    area = np.zeros(nc)
    for i, o in enumerate(own):
        q = P[flat[offsets[i]:offsets[i + 1]]]
        if np.allclose(q[:, 1], q[0, 1]) and abs(q[0, 1]) < 1e-12:          # a front face (y = 0)
            x, z = q[:, 0], q[:, 2]
            area[o] = 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
    return area


def _floor_cells(poly: str, nf: int, start: int) -> dict:
    own = foam.read_labels(_read(os.path.join(poly, "owner")))
    o = own[start:start + nf]
    cc = cell_centres(poly)
    return {"owner": o, "y": cc[o, 2]}                                         # floor at z = 0


def _column_depth(cc, alpha, areas, t: Trough):
    """Liquid depth (area of liquid per unit x) in columns of cells below the wall height."""
    xs = np.round(cc[:, 0], 6)
    ux = np.unique(xs)
    below = cc[:, 2] < t.wall_height
    dx = np.gradient(ux) if len(ux) > 1 else np.array([1.0])
    depth = np.array([(alpha[(xs == x) & below] * areas[(xs == x) & below]).sum() for x in ux]) / dx
    return ux, depth


def _hydro_ref(xs, depth_list, t):
    ux = depth_list[0][0]
    h = np.mean([d[1] for d in depth_list], 0)
    return ux, WATER.rho * 9.81 * h


def figures(case, out, t, times, cc, areas, res, kl, T, V, cum, log, xs, tau_m, tau_abs, spl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = os.path.join(case, "run")
    nc = len(cc)
    # free-surface snapshots
    snap = [s for s in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, t.end_time) if s <= T[-1] + 1e-9]
    fig, axs = plt.subplots(len(snap), 1, figsize=(8, 2.35 * len(snap) + 0.6), dpi=130, sharex=True)
    for ax, ts in zip(np.atleast_1d(axs), snap, strict=False):
        alpha = np.clip(_internal(os.path.join(run, _tname(times, ts), "alpha.water"), nc), 0.0, 1.0)
        ax.tricontourf(cc[:, 0], cc[:, 2], alpha, levels=np.linspace(0, 1.0001, 11), cmap="Blues")
        ax.tricontour(cc[:, 0], cc[:, 2], alpha, levels=[0.5], colors="k", linewidths=0.6)
        _outline(ax, t)
        ax.set_aspect("equal")
        ax.set_ylim(0, t.nozzle_z)
        ax.text(0.01, 0.82, f"t = {ts:g} s", transform=ax.transAxes, fontsize=8)
        ax.set_yticks([0, 0.2, 0.4])
    np.atleast_1d(axs)[-1].set_xlabel("x (m)")
    fig.suptitle("Liquid fraction (alpha = 0.5 contour in black)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(out, "free_surface_snapshots.png"))
    plt.close(fig)
    # mass balance and solver checks
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), dpi=130)
    ax[0].plot(T, V - V[0], label="liquid volume change (fields)")
    ax[0].plot(T, cum, "--", label="integrated net inflow (boundary fluxes)")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("m$^3$ per m of span")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Liquid volume balance", fontsize=9)
    ax[1].plot(log["t"], log["co_max"], lw=0.6, label="Courant max")
    ax[1].plot(log["t"], log["alpha_co_max"], lw=0.6, label="interface Courant max")
    ax[1].axhline(t.max_co, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylim(0, 1.2)
    ax[1].legend(fontsize=7)
    ax[1].set_title("Courant numbers", fontsize=9)
    ax[2].semilogy(log["t"], np.maximum(-log["alpha_min"], 1e-16), lw=0.6, label="-min(alpha)")
    ax[2].semilogy(log["t"], np.maximum(log["alpha_max"] - 1, 1e-16), lw=0.6, label="max(alpha) - 1")
    ax[2].set_xlabel("t (s)")
    ax[2].legend(fontsize=7)
    ax[2].set_title("alpha bounds (overshoot)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "checks.png"))
    plt.close(fig)
    # floor shear and k on lines
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), dpi=130, sharex=True)
    ax[0].plot(xs, tau_m, label="time-mean")
    ax[0].plot(xs, tau_abs, label="time-mean of |tau|")
    ax[0].axvline(res["impingement"]["x_floor_pressure_peak_m"], color="r", lw=0.8, ls="--", label="pressure peak")
    ax[0].axvline(res["impingement"]["x_ballistic_at_mean_free_surface_m"], color="g", lw=0.8, ls=":",
                  label="ballistic jet at free surface")
    ax[0].set_ylabel("floor shear stress (Pa)")
    ax[0].legend(fontsize=7)
    for name, d in kl.items():
        if name.startswith("floor"):
            x = np.array(d["xyz"])[:, 0]
            ax[1].plot(x, d["mean_k"], label=name)
    ax[1].set_ylabel("time-mean k (m$^2$/s$^2$)")
    ax[1].set_xlabel("x (m)")
    ax[1].legend(fontsize=7)
    w = res["impingement"]["averaging_window_s"]
    fig.suptitle(f"Floor shear and turbulent kinetic energy, averaged over {w[0]:.1f}-{w[1]:.1f} s", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "floor_shear_and_k.png"))
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=130)
    for name, d in kl.items():
        if name.startswith("vertical"):
            z = np.array(d["xyz"])[:, 2]
            ax.plot(d["mean_k"], z, label=name)
    ax.set_xlabel("time-mean k (m$^2$/s$^2$)")
    ax.set_ylabel("z (m)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "k_vertical_lines.png"))
    plt.close(fig)


def _outline(ax, t: Trough):
    ax.plot([0, 0, t.length, t.length], [t.wall_height, 0, 0, t.sill_height], color="0.3", lw=1.5)
    ax.plot([t.nozzle_x - t.slot / 2, t.nozzle_x + t.slot / 2], [t.nozzle_z, t.nozzle_z], color="r", lw=2)


def extend(case: str, end: float) -> None:
    """Continue the run from its last write to ``end`` seconds; log in log.interFoam.restart<N>."""
    from pinneapple_simulation.numerical_solvers.free_surface import LiveLineSampler
    meta_p = os.path.join(case, "trough.json")
    meta = json.load(open(meta_p))
    meta["parameters"]["end_time"] = end
    json.dump(meta, open(meta_p, "w"), indent=1)
    t = Trough(**meta["parameters"])
    run = os.path.join(case, "run")
    cd = os.path.join(run, "system", "controlDict")
    txt = open(cd).read()
    open(cd, "w").write(re.sub(r"\nendTime [^;]+;", f"\nendTime {end!r};", txt))
    n = len([f for f in os.listdir(run) if f.startswith("log.interFoam")])
    sampler = LiveLineSampler(run, t.lines(), ["alpha.water", "k"], end_time=end).start()
    try:
        sh(run, "interFoam", f"interFoam.restart{n}")
    finally:
        sampler.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--speed", type=float, default=Trough.speed)
    ap.add_argument("--angle", type=float, default=Trough.angle)
    ap.add_argument("--cell", type=float, default=Trough.cell)
    ap.add_argument("--end", type=float, default=Trough.end_time)
    ap.add_argument("--turbulence", choices=["kOmegaSST", "laminar"], default="kOmegaSST")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--post-only", action="store_true")
    ap.add_argument("--extend", type=float, default=None, help="continue a finished run to this end time (s)")
    a = ap.parse_args()
    if a.extend:
        extend(a.case, a.extend)
        a.post_only = True
    elif not a.post_only:
        t = Trough(speed=a.speed, angle=a.angle, cell=a.cell, end_time=a.end, turbulence=a.turbulence)
        flow = write_case(a.case, t, a.procs)
        if a.run:
            flow.run(os.path.join(a.case, "run"), procs=a.procs)
    if a.run or a.post_only:
        print(json.dumps({k: v for k, v in postprocess(a.case).items() if k not in ("parameters", "derived")},
                         indent=1))


if __name__ == "__main__":
    main()
