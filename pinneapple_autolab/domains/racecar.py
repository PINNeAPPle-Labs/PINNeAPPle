"""The AI Aerodynamicist's simulator: a parametric open-wheel racing car in OpenFOAM, and its flagship mission.

    sim = RaceCarAero(resolution="medium")
    sim.check(params)                     # geometry and boundary-condition checks, in words
    out = sim.run(params, "cases/run001")  # snappyHexMesh + simpleFoam (k-omega SST), forces, surface Cp, cost

Design variables (:data:`VARIABLES`): front wing main-plane and flap angles, rear wing main-plane and flap angles,
diffuser angle, ride height of the floor, on :class:`~pinneapple_design.geometry.vehicles3d.FormulaCar` (not any
team's car). The bounds keep the wing elements apart: beyond 5 deg (front) and 7.5 deg (rear) the main plane's
trailing edge cuts into the flap at low flap angles (measured with ``meshops.self_intersections``).

Outputs: drag and downforce in N at the mission's speed, CD and CL on the design's frontal area, the downforce of
each component, and the aerodynamic balance (share of the downforce on the front axle, from the moments of the
pressure and friction forces of every wall face about the rear contact patch). The surface pressure coefficient is
returned on the car's own parametric vertices (fixed count for every design), in regions ``front_wing``,
``rear_wing``, ``floor``, ``diffuser`` (the floor's ramp) and ``body``, for field surrogates.

Fixed conditions (the lab cannot change them): 70 m/s, sea-level air, road moving with the flow, half model with a
symmetry plane, steady RANS k-omega SST with wall functions, static wheels. Each case directory is deleted after its
outputs, fields and flow scene (``<case>.flow.npz``) are extracted, so a campaign fits in a few hundred MB.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer.study import Objective, Variable

from ..mission import Budget, Limit, Mission
from ..simulator import SimOutcome, Simulator

__all__ = ["VARIABLES", "BASELINE", "CONDITIONS", "MESHES", "RaceCarAero", "force_convergence", "render_run", "flagship_mission", "make_car",
           "solver_log_summary"]

VARIABLES = [
    Variable("front_wing_aoa", 0.0, 5.0, unit="deg", description="front wing main plane angle"),
    Variable("front_flap_aoa", 12.0, 30.0, unit="deg", description="front wing flap angle"),
    Variable("rear_wing_aoa", 0.0, 7.5, unit="deg", description="rear wing main plane angle"),
    Variable("rear_flap_aoa", 16.0, 40.0, unit="deg", description="rear wing flap angle"),
    Variable("diffuser_angle", 7.0, 18.0, unit="deg", description="diffuser ramp angle"),
    Variable("ride_height", 0.025, 0.060, unit="m", description="floor height above the road"),
]
BASELINE = {"front_wing_aoa": 4.0, "front_flap_aoa": 22.0, "rear_wing_aoa": 6.0, "rear_flap_aoa": 32.0,
            "diffuser_angle": 13.0, "ride_height": 0.035}
CONDITIONS = {"speed_m_s": 70.0, "rho_kg_m3": 1.225, "nu_m2_s": 1.5e-5, "ground": "moving road at the flow speed",
              "model": "half model, symmetry plane y = 0", "turbulence": "steady RANS k-omega SST, wall functions",
              "wheels": "static walls", "solver": "OpenFOAM simpleFoam"}
FIELD_BODIES = ("front_wing", "rear_wing", "floor", "body")
X_RAMP = 4.15                     # where FormulaCar's floor starts to rise into the diffuser
# Meshes, qualified on the baseline against "fine" (0.82 M cells; forces averaged over iterations 400-800, see
# examples/autolab_aero/_out/qualification.json): "medium" (0.21 M) leaves the 1-3 cm thick wing elements a cell or
# two thick and under-predicts the downforce by 44 % (drag -8.8 %); "campaign" (0.42 M) refines only the wings, the
# floor and the gap under it to the fine levels: drag -0.2 %, downforce -8.4 %, front balance +9.6 %.
MESHES: dict[str, dict[str, Any]] = {
    "coarse": {"resolution": "coarse"},
    "medium": {"resolution": "medium"},
    "fine": {"resolution": "fine"},
    "campaign": {"resolution": "medium",
                 "body_levels": {"front_wing": (5, 6), "rear_wing": (5, 6), "floor": (5, 6)},
                 "refine_boxes": [((1.7, 0.0, 0.0), (5.1, 0.8, 0.12), 5)]},
}
MIN_SLOT_GAP = 0.005              # m between a flap and its main plane
MIN_CLEARANCE = 0.02              # m between the road and any aero surface but the tyres
SYMMETRY_TOL = 1e-3               # m; the finest surface cells are ~2 cm (round-off in the loft is ~0.2 mm)


def make_car(params: Mapping[str, float]):
    from pinneapple_design.geometry.vehicles3d import FormulaCar
    return FormulaCar(**{k: float(v) for k, v in params.items()})


def force_convergence(history: list[dict[str, float]], window: int = 2) -> dict[str, float]:
    """Relative change of drag and downforce over the last ``window`` written intervals (e.g. the last 200
    iterations when forces are written every 100)."""
    if len(history) <= window:
        return {}
    a, b = history[-1 - window], history[-1]
    return {f"{k}_rel_change": abs(b[k] - a[k]) / max(abs(b[k]), 1e-12) for k in ("drag_N", "downforce_N")} | {
        "over_iterations": b["iteration"] - a["iteration"]}


def solver_log_summary(case: str) -> dict[str, Any]:
    """Last iteration, final initial residuals per field, whether checkMesh passed, snappy cells and timings."""
    out: dict[str, Any] = {}
    p = os.path.join(case, "log.simpleFoam")
    if os.path.exists(p):
        txt = open(p, errors="ignore").read()
        its = re.findall(r"^Time = (\d+)", txt, re.M)
        out["iterations"] = int(its[-1]) if its else 0
        res: dict[str, float] = {}
        for f, v in re.findall(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)", txt[-6000:]):
            res[f] = float(v)
        out["final_residuals"] = res
        out["simpleFoam_end"] = "End" in txt[-300:]
        out["fatal"] = "FOAM FATAL" in txt
    c = os.path.join(case, "log.checkMesh")
    if os.path.exists(c):
        t = open(c, errors="ignore").read()
        out["mesh_ok"] = "Mesh OK." in t
        m = re.search(r"^\s+cells:\s+(\d+)", t, re.M)
        out["cells"] = int(m.group(1)) if m else None
        out["mesh_failed_checks"] = int(re.search(r"Failed (\d+) mesh checks", t).group(1)) \
            if re.search(r"Failed (\d+) mesh checks", t) else 0
    return out


class RaceCarAero(Simulator):
    """OpenFOAM external aerodynamics of :class:`FormulaCar`. ``resolution``: a key of :data:`MESHES`
    ("coarse" 0.14 M cells, "medium" 0.21 M, "fine" 0.82 M cells at the baseline, "campaign": medium with the wings
    and the floor at the fine levels); ``iterations`` of simpleFoam. Forces are written every ``history_every``
    iterations and every output (forces, balance, surface Cp) is the mean over the written iterations from
    ``average_from`` x ``iterations`` on: the steady RANS forces of this car keep oscillating by +-2-3 % after 500
    iterations (measured on the campaign mesh, 1000 iterations), so one final snapshot would carry that noise. Their
    spread is returned as ``meta["iteration_std"]``. ``keep_cases``: keep the OpenFOAM case directories (large)."""
    name = "OpenFOAM simpleFoam (snappyHexMesh, k-omega SST), FormulaCar half model"
    fidelity = "reference"
    outputs = ("drag_N", "downforce_N", "balance_front", "CD", "CL", "downforce_front_wing_N",
               "downforce_rear_wing_N", "downforce_floor_N", "downforce_body_N", "downforce_wheels_N",
               "drag_front_wing_N", "drag_rear_wing_N", "drag_wheels_N", "frontal_area_m2")
    field_names = ("cp",)

    def __init__(self, resolution: str = "campaign", iterations: int = 800, procs: int = 4, n_lines: int = 40,
                 keep_cases: bool = False, conditions: Mapping[str, Any] | None = None, history_every: int = 100,
                 average_from: float = 0.5):
        if resolution not in MESHES:
            raise ValueError(f"unknown mesh {resolution!r}; use one of {sorted(MESHES)}")
        self.resolution, self.iterations, self.procs, self.n_lines = resolution, iterations, procs, n_lines
        self.history_every, self.average_from = history_every, average_from
        self.keep_cases = keep_cases
        self.conditions = dict(CONDITIONS if conditions is None else conditions)
        self.speed = float(self.conditions["speed_m_s"])
        self.rho, self.nu = float(self.conditions["rho_kg_m3"]), float(self.conditions["nu_m2_s"])
        V, idx = self._field_vertices(make_car(BASELINE))
        self.n_points = len(V)
        self._slices = idx
        regions = {}
        for b, (s, e) in idx.items():
            m = np.zeros(self.n_points, bool)
            m[s:e] = True
            regions[b] = m
        diff = regions["floor"] & (V[:, 0] > X_RAMP)
        regions["diffuser"] = diff
        self.regions = regions

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def _field_vertices(car) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
        B = car.cfd_bodies()
        Vs, idx, n = [], {}, 0
        for b in FIELD_BODIES:
            V = B[b][0]
            V = V[V[:, 1] >= -1e-9]
            idx[b] = (n, n + len(V))
            n += len(V)
            Vs.append(V)
        return np.vstack(Vs), idx

    def field_points(self, params):
        V, idx = self._field_vertices(make_car(params))
        if len(V) != self.n_points:
            raise RuntimeError("the parametric surface changed its vertex count")
        return V

    def check(self, params: Mapping[str, float]) -> list[str]:
        """Geometry: every CFD body closed, outward and with positive volume; wing elements apart (no
        intersection, slot gap >= 5 mm); every aero surface but the tyres >= 20 mm above the road; mirror
        symmetry about y = 0 (the half model's symmetry plane). Conditions: the simulator's are the mission's."""
        from scipy.spatial import cKDTree

        from pinneapple_design.geometry_library import meshops as mo
        issues: list[str] = []
        try:
            car = make_car(params)
            B = car.cfd_bodies()
        except Exception as e:  # noqa: BLE001
            return [f"geometry could not be built: {type(e).__name__}: {e}"]
        for k, (V, F) in B.items():
            es = mo.edge_stats(V, F)
            if es["boundary_edges"] or es["nonmanifold_edges"]:
                issues.append(f"{k}: surface not closed ({es['boundary_edges']} open, {es['nonmanifold_edges']} "
                              "non-manifold edges)")
            if mo.volume(V, F) <= 0:
                issues.append(f"{k}: encloses no positive volume (inward normals)")
            if k != "wheels" and V[:, 2].min() < MIN_CLEARANCE - 1e-9:
                issues.append(f"{k}: {1000 * V[:, 2].min():.1f} mm above the road (< {1000 * MIN_CLEARANCE:g} mm)")
            d = cKDTree(V).query(V * [1, -1, 1])[0].max()
            if d > SYMMETRY_TOL:
                issues.append(f"{k}: not symmetric about y = 0 (max mirror distance {d:.2e} m): the half model "
                              "would be wrong")
        for wing, parts in (("front wing", car._front_wing()), ("rear wing", car._rear_wing())):
            (V1, F1), (V2, F2) = parts[0], parts[1]
            n = mo.self_intersections(np.vstack([V1, V2]), np.vstack([F1, F2 + len(V1)]))
            gap = float(cKDTree(V1).query(V2)[0].min())
            if n:
                issues.append(f"{wing}: main plane and flap intersect ({n} triangle pairs)")
            elif gap < MIN_SLOT_GAP:
                issues.append(f"{wing}: slot gap {1000 * gap:.1f} mm < {1000 * MIN_SLOT_GAP:g} mm")
        if self.conditions != CONDITIONS:
            diff = {k: v for k, v in self.conditions.items() if CONDITIONS.get(k) != v}
            issues.append(f"boundary conditions differ from the mission's: {diff}")
        return issues

    # ------------------------------------------------------------------ run
    def run(self, params: Mapping[str, float], workdir: str, resolution: str | None = None,
            iterations: int | None = None) -> SimOutcome:
        from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow
        t0 = time.perf_counter()
        res = resolution or self.resolution
        its = iterations or self.iterations
        mesh = dict(MESHES[res])
        car = make_car(params)
        A = car.frontal_area()
        flow = ExternalFlow(car.cfd_bodies(), speed=self.speed, nu=self.nu, rho=self.rho, ground=0.0,
                            half_model=True, ref_area=A, iterations=its, n_lines=self.n_lines,
                            write_every=self.history_every or None, **mesh,
                            title=f"FormulaCar, {self.speed:g} m/s")
        case = os.path.abspath(workdir)
        os.makedirs(os.path.dirname(case), exist_ok=True)
        meta: dict[str, Any] = {"resolution": res, "iterations_requested": its, "frontal_area_m2": A,
                                "procs": self.procs}
        try:
            flow.write(case, self.procs)
            meta["timings_s"] = flow.run(case, self.procs, log=lambda s: None)
            meta.update(solver_log_summary(case))
            if meta.get("fatal") or meta.get("iterations", 0) < its:
                raise RuntimeError(f"simpleFoam stopped at iteration {meta.get('iterations', 0)} of {its}"
                                   + (" (FOAM FATAL ERROR)" if meta.get("fatal") else ""))
            r = flow.read(case)
            hist = flow.force_history(case, faces=True) if self.history_every else []
        except Exception as e:  # noqa: BLE001 - the failure and its logs are the outcome
            meta.update(solver_log_summary(case) if os.path.isdir(case) else {})
            self._cleanup(case)
            return SimOutcome({}, False, f"{type(e).__name__}: {e}", time.perf_counter() - t0, None, meta)
        # every written iteration in the averaging window (the last half of the run) gives one sample of each
        # output; the outputs are their means, the spread is the solver's own iteration noise
        window = [h for h in hist if h["iteration"] >= self.average_from * its] or [None]
        samples = [self._outputs(car, A, h, r) for h in window]
        out = {k: float(np.mean([smp[k] for smp in samples])) for k in samples[0]}
        meta["iteration_std"] = {k: float(np.std([smp[k] for smp in samples])) for k in samples[0]}
        meta["averaged_iterations"] = [h["iteration"] for h in window if h is not None]
        meta["force_history"] = [{"iteration": h["iteration"], "drag_N": h["drag_N"], "downforce_N": -h["lift_N"]}
                                 for h in hist]
        meta["convergence"] = force_convergence(meta["force_history"])
        cps = [self._cp_on_vertices(car, h["surface"] if h is not None else r.surface, r.bodies) for h in window]
        cp = np.mean(cps, 0)
        c = r.coefficients
        meta.update(cells=c.get("cells"), cd_pressure=c.get("CD_pressure"), cd_friction=c.get("CD_friction"))
        try:
            r.save(case + ".flow.npz")
            meta["flow_npz"] = case + ".flow.npz"
        except Exception as e:  # noqa: BLE001 - the scene is a convenience, not evidence
            meta["flow_npz_error"] = str(e)
        self._cleanup(case)
        bad = [k for k, v in out.items() if not np.isfinite(v)]
        if bad:
            return SimOutcome({}, False, f"non-finite outputs {bad}", time.perf_counter() - t0, None, meta)
        return SimOutcome(out, True, "", time.perf_counter() - t0, {"cp": cp}, meta)

    def _outputs(self, car, A: float, h: dict[str, Any] | None, r) -> dict[str, float]:
        """Outputs from one written iteration ``h`` of the force history (or the final result ``r``)."""
        q = 0.5 * self.rho * self.speed ** 2
        c = r.coefficients if h is None else h
        bodies = c["bodies"]
        out = {"drag_N": float(c["drag_N"]), "downforce_N": float(-c["lift_N"]), "CD": float(c["CD"]),
               "CL": float(c["CL"]), "frontal_area_m2": float(A)}
        for b in ("front_wing", "rear_wing", "floor", "body", "wheels"):
            bc = bodies.get(b, {"CL": np.nan, "CD": np.nan})
            out[f"downforce_{b}_N"] = float(-bc["CL"] * q * A)
            if b in ("front_wing", "rear_wing", "wheels"):
                out[f"drag_{b}_N"] = float(bc["CD"] * q * A)
        out["balance_front"] = self._balance(car, r.surface if h is None else h["surface"])
        return out

    def _cleanup(self, case: str) -> None:
        if not self.keep_cases and os.path.isdir(case):
            shutil.rmtree(case, ignore_errors=True)

    def _balance(self, car, surface: dict[str, np.ndarray]) -> float:
        """Front-axle share of the downforce: moments about the rear contact patch of every face force (pressure
        and friction, half model doubled; the moment of the drag about the road counts)."""
        F = surface["force"] * self.rho            # per face, N (half model)
        x, z = surface["xyz"][:, 0], surface["xyz"][:, 2]
        xf, xr = car.front_axle_x, car.front_axle_x + car.wheelbase
        down = -F[:, 2]
        front = (np.sum(down * (xr - x)) - np.sum(F[:, 0] * z)) / (xr - xf)
        return float(front / np.sum(down))

    def _cp_on_vertices(self, car, surface: dict[str, np.ndarray], bodies: list[str]) -> np.ndarray:
        """Cp of the nearest wall face of the same body for every field vertex (y >= 0)."""
        from scipy.spatial import cKDTree
        V, idx = self._field_vertices(car)
        cp = np.full(len(V), np.nan)
        xyz, body, cps = surface["xyz"], surface["body"], surface["cp"]
        for b, (s, e) in idx.items():
            if b not in bodies:
                continue
            m = body == bodies.index(b)
            if not m.any():
                continue
            _, j = cKDTree(xyz[m]).query(V[s:e])
            cp[s:e] = cps[m][j]
        return cp


def render_run(params: Mapping[str, float], flow_npz: str, out_prefix: str, title: str = "",
               size: tuple[int, int] = (1200, 675)) -> dict[str, str]:
    """PyVista renders of one simulated design from its saved flow scene: surface Cp (jet, from above and below)
    and the streamlines coloured by speed. Returns {name: path}."""
    from pinneapple_simulation.numerical_solvers.external_flow import FlowResult
    from pinneapple_tools.visualization.studio.pyvista_render import render_scene
    from pinneapple_tools.visualization.studio.scene import Scene
    car = make_car(params)
    sc = FlowResult.load(flow_npz).to_scene(Scene.from_parts(car.parts(), axes="z_up", title=title))
    out = {"cp_top": f"{out_prefix}_cp_top.png", "cp_bottom": f"{out_prefix}_cp_bottom.png",
           "streamlines": f"{out_prefix}_streamlines.png"}
    render_scene(sc, out["cp_top"], field="cp", clim=(-2.0, 1.0), view="iso", zoom=1.6, hide=("halo",), title=title,
                 size=size)
    render_scene(sc, out["cp_bottom"], field="cp", clim=(-2.0, 1.0), view=(-0.25, -0.35, -1.0), zoom=1.35,
                 title=title, size=size)
    render_scene(sc, out["streamlines"], lines=True, view="iso", zoom=1.4, background="#000000", title=title,
                 size=size)
    from ..report import shrink_png
    for p in out.values():
        shrink_png(p)
    return out


def flagship_mission(budget: int = 30, reserve_verify: int = 3) -> Mission:
    """"Find a lower-drag aerodynamic design at 70 m/s. Preserve the required downforce, stay within the geometry
    constraints, and validate the final design using CFD.\""""
    return Mission(
        "The AI Aerodynamicist: lower drag at 70 m/s",
        "Find a lower-drag aerodynamic design at 70 m/s. Preserve the required downforce, stay within the geometry "
        "constraints, and validate the final design using CFD.",
        list(VARIABLES),
        [Objective("drag_N", "min", "N", "drag")],
        [Limit("downforce_N", lower_frac_of_baseline=1.0, unit="N", label="downforce"),
         Limit("balance_front", lower_frac_of_baseline=0.9, upper_frac_of_baseline=1.1, unit="",
               label="aero balance (front share)")],
        dict(BASELINE), Budget(budget, reserve_verify),
        acceptance={"drag_N": 0.03, "downforce_N": 0.05, "balance_front": 0.05},
        conditions=dict(CONDITIONS),
        hypotheses=[
            "the rear flap is the largest single source of drag that can be traded for downforce elsewhere",
            "a steeper diffuser adds underfloor downforce at little drag until the flow separates from the ramp",
            "a lower floor strengthens the ground effect (more downforce) within the clearance limit",
            "front-wing angle moves the aero balance forward; rear-wing angle moves it back",
        ],
        domain="external aerodynamics")
