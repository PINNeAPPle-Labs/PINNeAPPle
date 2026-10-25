"""The Ahmed body in OpenFOAM as a reference simulator: geometry variants, checks before the run, a mesh family for
grid-convergence studies, quality checks after it, and the wind-tunnel measurements it is compared with.

    sim = AhmedAero(mesh="campaign")
    sim.check(params)                        # geometry and condition checks, in words
    out = sim.run(params, "cases/run001")    # snappyHexMesh + simpleFoam (k-omega SST), forces, drag breakdown, Cp
    report = run_trust_report(out.meta)      # checkMesh, y+, residuals, force convergence, mass balance

Design variables (:data:`VARIABLES`): rear slant angle, ground clearance and wind speed. Fixed (:data:`CONDITIONS`):
the Ahmed, Ramm & Faltin (1984) body (1.044 x 0.389 x 0.288 m, 0.222 m slant), sea-level air, road moving with the
flow, half model with a symmetry plane, steady RANS k-omega SST with wall functions, no stilts. Outputs: CD, CL, the
pressure and friction parts of CD, and the pressure drag of each region Ahmed et al. reported separately (front,
slant, base; plus roof, sides and underbody). The surface pressure coefficient is returned on the vertices of a
dense parametric surface (:func:`field_vertices`, 1262 vertices, y >= 0 half: same count and same material points
for every variant) with region masks, for field surrogates.

Reference (:data:`AHMED_1984_CD`): total drag coefficient against slant angle measured by Ahmed, Ramm & Faltin
(1984, SAE 840300) at 60 m/s (Re 4.29e6 on the body length), as transcribed in
``examples/qualitative/03_ahmed_slant.py``. Steady RANS with k-omega SST is known not to reproduce the drag peak just
below 30 degrees (the slant flow's partial reattachment); the validation reports that, it does not hide it.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer.study import Variable

from ..simulator import SimOutcome, Simulator

__all__ = ["VARIABLES", "BASELINE", "CONDITIONS", "AHMED_1984_CD", "AHMED_1984_SOURCE", "MESHES", "mesh_family",
           "AhmedAero", "make_body", "field_vertices", "region_of", "REGIONS", "solver_log_summary",
           "force_convergence", "qa_checks", "run_trust_report", "LENGTH", "WIDTH", "HEIGHT", "NOSE_RADIUS",
           "template_faces", "pressure_drag_from_cp", "prediction_checks", "unit_to_params", "params_to_unit",
           "ahmed_cad", "import_body", "read_stl"]

LENGTH, WIDTH, HEIGHT, NOSE_RADIUS, SLANT_LENGTH = 1.044, 0.389, 0.288, 0.100, 0.222

VARIABLES = [
    Variable("slant_deg", 0.0, 40.0, unit="deg", description="rear slant angle"),
    Variable("clearance", 0.03, 0.08, unit="m", description="gap between the underbody and the road"),
    Variable("speed", 20.0, 60.0, unit="m/s", description="wind and road speed"),
]
BASELINE = {"slant_deg": 25.0, "clearance": 0.05, "speed": 60.0}
CONDITIONS = {"rho_kg_m3": 1.225, "nu_m2_s": 1.5e-5, "ground": "moving road at the flow speed",
              "model": "half model, symmetry plane y = 0", "turbulence": "steady RANS k-omega SST, wall functions",
              "stilts": "left out", "solver": "OpenFOAM simpleFoam"}

# Ahmed, Ramm & Faltin (1984), SAE 840300: total drag coefficient against slant angle, 60 m/s, Re = 4.29e6.
AHMED_1984_CD = {0.0: 0.250, 12.5: 0.230, 25.0: 0.285, 30.0: 0.378, 35.0: 0.260, 40.0: 0.250}
AHMED_1984_SOURCE = ("Ahmed, S.R., Ramm, G., Faltin, G. (1984). Some salient features of the time-averaged ground "
                     "vehicle wake. SAE Technical Paper 840300. Values as transcribed in "
                     "examples/qualitative/03_ahmed_slant.py (60 m/s, Re 4.29e6 on the length).")

# Meshes: one geometric family (background cells per body length grow by sqrt(2), the surface and refinement-box
# levels stay the same, so every cell shrinks by the same ratio; five prism layers sized relative to the surface
# cell keep the first-cell y+ in the wall functions' log range: without them it was 500-700 at 60 m/s) for the grid-convergence study; "campaign" is the
# member the design-of-experiments runs on (chosen from that study, see examples/end_to_end/ahmed_body).
_FAMILY = (6.0, 6.0 * math.sqrt(2.0), 12.0, 12.0 * math.sqrt(2.0))
MESHES: dict[str, dict[str, Any]] = {f"L{k}": {"resolution": "coarse", "cells_per_length": n, "layers": 5}
                                     for k, n in enumerate(_FAMILY)}
MESHES["campaign"] = dict(MESHES["L1"])

REGIONS = ("front", "roof", "slant", "base", "side", "underbody")


def mesh_family() -> list[str]:
    """Names of the grid-convergence family, coarse to fine."""
    return [f"L{k}" for k in range(len(_FAMILY))]


def make_body(params: Mapping[str, float], dense: bool = False):
    """Closed triangulated Ahmed body for ``params`` (``slant_deg``, ``clearance``); ``dense``: the field template's
    sampling (1262 vertices) instead of the meshing one (442)."""
    from pinneapple_design.geometry.bodies import ahmed_body
    kw = dict(n_nose=12, n_body=20, n_slant=12, n_corner=6) if dense else {}
    return ahmed_body(float(params["slant_deg"]), clearance=float(params["clearance"]), **kw)


def region_of(xyz: np.ndarray, params: Mapping[str, float], tol: float = 0.012) -> np.ndarray:
    """Region name of each point on the body surface (front: the rounded nose, x < 0.1 m; base: the vertical rear
    face; slant: the inclined rear face; roof, underbody, side)."""
    x, y, z = (np.asarray(xyz, float)[:, i] for i in range(3))
    c = float(params["clearance"])
    t = math.tan(math.radians(float(params["slant_deg"])))
    x_s = LENGTH - SLANT_LENGTH
    top = c + HEIGHT - np.maximum(0.0, x - x_s) * t
    out = np.full(len(x), "side", dtype=object)
    out[z >= top - tol] = "roof"
    out[(x > x_s + 1e-6) & (z >= top - tol) & (t > 0)] = "slant"
    out[z <= c + tol] = "underbody"
    out[np.abs(y) >= WIDTH / 2 - tol / 3] = "side"
    out[x >= LENGTH - 1e-3] = "base"
    out[x < NOSE_RADIUS - 1e-6] = "front"
    return out


def field_vertices(params: Mapping[str, float]) -> np.ndarray:
    """The field template: dense-body vertices with y >= 0 (the half model's side), in a fixed order."""
    V, _ = make_body(params, dense=True)
    return V[V[:, 1] >= -1e-9]


def force_convergence(history: list[dict[str, float]], window: int = 4, average_from: float = 0.5) -> dict[str, float]:
    """How settled the forces are. ``CD_rel_change``: change of the instantaneous CD between the last write and
    ``window`` writes earlier (information: steady RANS of a bluff body keeps oscillating). ``CD_mean_drift``: the
    relative difference between the CD averaged over the first and over the second half of the averaging window
    (iterations from ``average_from`` of the run on): the reported output is that average, and it has settled
    when the two halves agree. ``CD_window_std``: the spread of CD over the window."""
    if len(history) <= window:
        return {}
    a, b = history[-1 - window], history[-1]
    out = {"CD_rel_change": abs(b["CD"] - a["CD"]) / max(abs(b["CD"]), 1e-12),
           "CL_rel_change": abs(b["CL"] - a["CL"]) / max(abs(b["CD"]), 1e-12),      # CL can cross zero: scale by CD
           "over_iterations": b["iteration"] - a["iteration"]}
    win = [h["CD"] for h in history if h["iteration"] >= average_from * history[-1]["iteration"]]
    if len(win) >= 4:
        h1, h2 = np.mean(win[:len(win) // 2]), np.mean(win[len(win) // 2:])
        out.update(CD_mean_drift=float(abs(h2 - h1) / max(abs(np.mean(win)), 1e-12)),
                   CD_window_std=float(np.std(win)), window_writes=len(win))
    return out


def solver_log_summary(case: str) -> dict[str, Any]:
    """Last iteration, final initial residuals, checkMesh verdict and cell count, snappyHexMesh status."""
    from .racecar import solver_log_summary as _base
    out = _base(case)
    c = os.path.join(case, "log.checkMesh")
    if os.path.exists(c):
        t = open(c, errors="ignore").read()
        for key, pat in (("max_non_orthogonality", r"Mesh non-orthogonality Max: ([0-9.eE+-]+)"),
                         ("max_skewness", r"Max skewness = ([0-9.eE+-]+)"),
                         ("max_aspect_ratio", r"Max aspect ratio = ([0-9.eE+-]+)")):
            m = re.search(pat, t)
            if m:
                out[key] = float(m.group(1))
    return out


class AhmedAero(Simulator):
    """OpenFOAM external aerodynamics of the Ahmed body. ``mesh``: a key of :data:`MESHES`; forces are written
    every ``history_every`` iterations and CD, CL and Cp are the means over the written iterations from
    ``average_from`` x ``iterations`` on (their spread is ``meta["iteration_std"]``). ``keep_cases``: keep the
    OpenFOAM directories; otherwise only ``<case>.flow.npz`` (scene: skin Cp/Cf, streamlines, slices) stays.
    ``geometry(params) -> (V, F)``: another body than :func:`make_body` (an imported STL, :func:`import_body`)."""
    name = "OpenFOAM simpleFoam (snappyHexMesh, k-omega SST), Ahmed body half model"
    fidelity = "reference"
    outputs = ("CD", "CL", "CD_pressure", "CD_friction", "CD_front", "CD_slant", "CD_base", "CD_roof", "CD_side",
               "CD_underbody", "drag_N", "lift_N")
    field_names = ("cp",)

    def __init__(self, mesh: str = "campaign", iterations: int = 800, procs: int = 4, keep_cases: bool = False,
                 history_every: int = 50, average_from: float = 0.5, n_lines: int = 40,
                 conditions: Mapping[str, Any] | None = None,
                 geometry: Callable[[Mapping[str, float]], tuple[np.ndarray, np.ndarray]] | None = None):
        if mesh not in MESHES:
            raise ValueError(f"unknown mesh {mesh!r}; use one of {sorted(MESHES)}")
        self.mesh, self.iterations, self.procs, self.keep_cases = mesh, iterations, procs, keep_cases
        self.history_every, self.average_from, self.n_lines = history_every, average_from, n_lines
        self.geometry = geometry                    # (V, F) of a design instead of make_body (e.g. an imported STL)
        self.conditions = dict(CONDITIONS if conditions is None else conditions)
        self.rho, self.nu = float(self.conditions["rho_kg_m3"]), float(self.conditions["nu_m2_s"])
        V = field_vertices(BASELINE)
        self.n_points = len(V)
        lab = region_of(V, BASELINE)
        self.regions = {r: lab == r for r in REGIONS}

    def field_points(self, params):
        V = field_vertices(params)
        if len(V) != self.n_points:
            raise RuntimeError("the parametric surface changed its vertex count")
        return V

    def check(self, params: Mapping[str, float]) -> list[str]:
        """Inputs inside the variables' bounds; the body closed, outward, with positive volume and mirror-symmetric
        about y = 0 (the half model's symmetry plane); conditions those of :data:`CONDITIONS`."""
        from scipy.spatial import cKDTree

        from pinneapple_design.geometry_library import meshops as mo
        issues: list[str] = []
        for v in VARIABLES:
            if v.name not in params:
                issues.append(f"missing {v.name}")
            elif not v.lower - 1e-9 <= float(params[v.name]) <= v.upper + 1e-9:
                issues.append(f"{v.name} = {params[v.name]:g} {v.unit} outside [{v.lower:g}, {v.upper:g}]")
        if issues:
            return issues
        try:
            V, F = (self.geometry or make_body)(params)
        except Exception as e:  # noqa: BLE001
            return [f"geometry could not be built: {type(e).__name__}: {e}"]
        es = mo.edge_stats(V, F)
        if es["boundary_edges"] or es["nonmanifold_edges"]:
            issues.append(f"surface not closed ({es['boundary_edges']} open, {es['nonmanifold_edges']} non-manifold)")
        if mo.volume(V, F) <= 0:
            issues.append("encloses no positive volume (inward normals)")
        if self.geometry is None and cKDTree(V).query(V * [1, -1, 1])[0].max() > 1e-6:
            issues.append("not symmetric about y = 0: the half model would be wrong")   # imported: import_body checks it
        if self.conditions != CONDITIONS:
            issues.append(f"conditions differ from the study's: "
                          f"{ {k: v for k, v in self.conditions.items() if CONDITIONS.get(k) != v} }")
        return issues

    # ------------------------------------------------------------------ run
    def flow(self, params: Mapping[str, float], mesh: str | None = None, iterations: int | None = None):
        """The :class:`ExternalFlow` of a design (not run)."""
        from pinneapple_simulation.numerical_solvers.external_flow import ExternalFlow
        V, F = (self.geometry or make_body)(params)
        U = float(params["speed"])
        return ExternalFlow({"ahmed": (V, F)}, speed=U, nu=self.nu, rho=self.rho, ground=0.0, half_model=True,
                            ref_area=WIDTH * HEIGHT, ref_length=LENGTH, iterations=iterations or self.iterations,
                            write_every=self.history_every or None, n_lines=self.n_lines,
                            title=f"Ahmed body, slant {float(params['slant_deg']):g} deg, {U:g} m/s",
                            **MESHES[mesh or self.mesh])

    def run(self, params: Mapping[str, float], workdir: str, mesh: str | None = None,
            iterations: int | None = None) -> SimOutcome:
        from pinneapple_simulation.numerical_solvers.external_flow import CaseFields
        t0 = time.perf_counter()
        its = iterations or self.iterations
        flow = self.flow(params, mesh, its)
        case = os.path.abspath(workdir)
        os.makedirs(os.path.dirname(case), exist_ok=True)
        meta: dict[str, Any] = {"mesh": mesh or self.mesh, "iterations_requested": its, "procs": self.procs,
                                "params": {k: float(v) for k, v in params.items()},
                                "Re": float(params["speed"]) * LENGTH / self.nu}
        try:
            flow.write(case, self.procs)
            meta["timings_s"] = flow.run(case, self.procs, log=lambda s: None)
            meta.update(solver_log_summary(case))
            if meta.get("fatal") or meta.get("iterations", 0) < its:
                raise RuntimeError(f"simpleFoam stopped at iteration {meta.get('iterations', 0)} of {its}"
                                   + (" (FOAM FATAL ERROR)" if meta.get("fatal") else ""))
            r = flow.read(case)
            hist = flow.force_history(case, faces=True) if self.history_every else []
            meta["mass_balance"] = _mass_balance(case)
            parts = self._breakdown(CaseFields(case), params)
        except Exception as e:  # noqa: BLE001 - the failure and its logs are the outcome
            meta.update(solver_log_summary(case) if os.path.isdir(case) else {})
            meta["error"] = f"{type(e).__name__}: {e}"
            self._cleanup(case)
            return SimOutcome({}, False, meta["error"], time.perf_counter() - t0, None, meta)
        window = [h for h in hist if h["iteration"] >= self.average_from * its] or [None]
        cds = [r.coefficients["CD"] if h is None else h["CD"] for h in window]
        cls_ = [r.coefficients["CL"] if h is None else h["CL"] for h in window]
        c = r.coefficients
        out = {"CD": float(np.mean(cds)), "CL": float(np.mean(cls_)), **parts}
        q = 0.5 * self.rho * float(params["speed"]) ** 2 * WIDTH * HEIGHT
        out["drag_N"], out["lift_N"] = out["CD"] * q, out["CL"] * q
        meta["iteration_std"] = {"CD": float(np.std(cds)), "CL": float(np.std(cls_))}
        meta["averaged_iterations"] = [h["iteration"] for h in window if h is not None]
        meta["force_history"] = [{"iteration": h["iteration"], "CD": h["CD"], "CL": h["CL"]} for h in hist]
        meta["convergence"] = force_convergence(meta["force_history"], average_from=self.average_from)
        yp = np.asarray(r.surface["yplus"])
        meta["yplus"] = {"mean": float(yp.mean()), "p05": float(np.percentile(yp, 5)),
                         "p95": float(np.percentile(yp, 95)), "max": float(yp.max()),
                         "frac_30_300": float(np.mean((yp >= 30) & (yp <= 300))),
                         "frac_below_11": float(np.mean(yp < 11.25))}
        meta.update(cells=c.get("cells"), CD_final=c["CD"], CL_final=c["CL"])
        cps = [self._cp_on_vertices(params, r.surface if h is None else h["surface"]) for h in window]
        cp = np.mean(cps, 0)
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

    def _breakdown(self, cf, params) -> dict[str, float]:
        """Pressure drag per region and the friction drag (Ahmed et al.'s C_K, C_S, C_B and C_R; plus the roof,
        sides and underbody), in drag coefficients on the frontal area, from the final iteration."""
        U = float(params["speed"])
        q = 0.5 * U * U
        S = WIDTH * HEIGHT / 2                                   # half model
        out = {f"CD_{r}": 0.0 for r in REGIONS}
        fric = press = 0.0
        for patch in cf.patches():
            if patch["name"] != "ahmed":
                continue
            w = cf.wall(patch, self.nu, q)
            lab = region_of(w["xyz"], params)
            fx = w["f_pressure"][:, 0]
            for rname in REGIONS:
                out[f"CD_{rname}"] += float(fx[lab == rname].sum() / (q * S))
            press += float(fx.sum() / (q * S))
            fric += float(w["f_viscous"][:, 0].sum() / (q * S))
        out["CD_pressure"], out["CD_friction"] = press, fric
        return out

    def _cp_on_vertices(self, params, surface: Mapping[str, np.ndarray]) -> np.ndarray:
        """Cp of the nearest wall face for every template vertex."""
        from scipy.spatial import cKDTree
        V = field_vertices(params)
        _, j = cKDTree(surface["xyz"]).query(V)
        return np.asarray(surface["cp"])[j]

    def _cleanup(self, case: str) -> None:
        if not self.keep_cases and os.path.isdir(case):
            shutil.rmtree(case, ignore_errors=True)


def _mass_balance(case: str) -> dict[str, float]:
    """Net volume flux through the boundary over the inflow (|net| / (sum |flux| / 2)), from phi."""
    from pinneapple_tools.automation.verify import patch_fluxes
    b = patch_fluxes(case)
    inflow = 0.5 * b["total"]
    return {"net_m3_s": float(b["net"]), "inflow_m3_s": float(inflow),
            "imbalance": float(abs(b["net"]) / max(inflow, 1e-30)), "source": b["source"]}


# ---------------------------------------------------------------------- quality of one run
QA_LIMITS = {"max_non_orthogonality": 70.0, "max_skewness": 4.0, "residual": 1e-3, "force_change": 0.01,
             "mass_imbalance": 1e-3, "yplus_low": 11.25, "yplus_high": 500.0, "frac_wall_function_range": 0.7}


def qa_checks(meta: Mapping[str, Any], limits: Mapping[str, float] | None = None):
    """The checks of one CFD run as :class:`~pinneapple_analysis.trust.Check` objects: checkMesh, mesh quality,
    final residuals, force convergence, mass balance and the wall-function y+ range."""
    from pinneapple_analysis.trust import Check
    L = {**QA_LIMITS, **(limits or {})}
    out = []

    def add(name, ok, value=None, limit=None, reason="", critical=False):
        out.append(Check(name, "not_run" if ok is None else ("supports" if ok else "contradicts"), critical=critical,
                         value=value, limit=limit, source="OpenFOAM logs and fields", reason=reason))
    add("checkMesh", meta.get("mesh_ok"), reason="checkMesh reports Mesh OK", critical=True)
    no = meta.get("max_non_orthogonality")
    add("non_orthogonality", None if no is None else no <= L["max_non_orthogonality"], no,
        L["max_non_orthogonality"], "max non-orthogonality (deg)")
    sk = meta.get("max_skewness")
    add("skewness", None if sk is None else sk <= L["max_skewness"], sk, L["max_skewness"], "max skewness")
    res = meta.get("final_residuals") or {}
    worst = max(res.values()) if res else None
    add("residuals", None if worst is None else worst <= L["residual"], worst, L["residual"],
        "largest final initial residual of p, U, k, omega")
    conv = meta.get("convergence") or {}
    ch = conv.get("CD_mean_drift")
    add("force_convergence", None if ch is None else ch <= L["force_change"], ch, L["force_change"],
        f"relative difference of the CD averaged over the two halves of the averaging window "
        f"({conv.get('window_writes', 0)} writes; instantaneous change over the last "
        f"{conv.get('over_iterations', 0):g} iterations {conv.get('CD_rel_change', float('nan')):.2%})")
    mb = (meta.get("mass_balance") or {}).get("imbalance")
    add("mass_balance", None if mb is None else mb <= L["mass_imbalance"], mb, L["mass_imbalance"],
        "|net boundary flux| / inflow, from phi", critical=True)
    yp = meta.get("yplus") or {}
    fr = yp.get("frac_30_300")
    add("yplus_wall_functions", None if fr is None else fr >= L["frac_wall_function_range"], fr,
        L["frac_wall_function_range"], f"share of wall faces with 30 <= y+ <= 300 (mean y+ {yp.get('mean', 0):.0f}, "
                                       f"p95 {yp.get('p95', 0):.0f})")
    return out


def run_trust_report(meta: Mapping[str, Any], limits: Mapping[str, float] | None = None):
    """:class:`~pinneapple_analysis.trust.TrustReport` of one CFD run from :func:`qa_checks`."""
    from pinneapple_analysis.trust import TrustReport
    return TrustReport(qa_checks(meta, limits), model="OpenFOAM run")


# ---------------------------------------------------------------------- checks on surrogate predictions
def unit_to_params(U: np.ndarray, bounds: Mapping[str, tuple[float, float]] | None = None) -> list[dict[str, float]]:
    """Unit-cube rows -> parameter dicts (``bounds``: name -> (low, high); default the :data:`VARIABLES`)."""
    b = bounds or {v.name: (v.lower, v.upper) for v in VARIABLES}
    names = [v.name for v in VARIABLES]
    U = np.atleast_2d(U)
    return [{k: float(b[k][0] + u[j] * (b[k][1] - b[k][0])) for j, k in enumerate(names)} for u in U]


def params_to_unit(P, bounds: Mapping[str, tuple[float, float]] | None = None) -> np.ndarray:
    b = bounds or {v.name: (v.lower, v.upper) for v in VARIABLES}
    P = [P] if isinstance(P, Mapping) else P
    return np.array([[(float(p[k]) - b[k][0]) / (b[k][1] - b[k][0]) for k in b] for p in P])


def template_faces(params: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The dense closed body ``(V, F)`` and, per vertex, the index of its point in :func:`field_vertices` (its
    mirror image for y < 0; the body is symmetric about y = 0)."""
    from scipy.spatial import cKDTree
    V, F = make_body(params, dense=True)
    half = field_vertices(params)
    d, j = cKDTree(half).query(np.c_[V[:, 0], np.abs(V[:, 1]), V[:, 2]])
    if d.max() > 1e-9:
        raise RuntimeError("the dense body is not symmetric about y = 0")
    return V, F, j


def pressure_drag_from_cp(params: Mapping[str, float], cp: np.ndarray) -> float:
    """Pressure drag coefficient (frontal area) of a Cp field on :func:`field_vertices`: Cp mirrored onto the whole
    closed body, then the sum over its triangles of Cp (vertex mean) x the x-component of the inward area vector,
    over W x H. A uniform Cp gives zero (closed surface); compare the result with the CFD's own CD_pressure on the
    same template before trusting a difference (the template is coarser than the CFD surface)."""
    V, F, j = template_faces(params)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = 0.5 * np.cross(b - a, c - a)                     # outward area vectors
    cpf = np.asarray(cp)[j][F].mean(1)
    return float(-np.sum(cpf * n[:, 0]) / (WIDTH * HEIGHT))


def prediction_checks(bounds: Mapping[str, tuple[float, float]], *, n: int = 256, seed: int = 0,
                      cp_max: float = 1.05, integral_tol: float = 0.03, cd_pressure: Callable | None = None):
    """Physics checks for surrogate predictions, as ``{target: [check(predict) -> Check]}`` for
    :func:`pinneapple_autolab.study.benchmark.benchmark`, on a scrambled Sobol sample of the training region:

    * CD and CD_pressure positive (a bluff body in a steady stream has positive drag);
    * CD_friction positive and falling with speed at fixed shape (friction coefficient falls with Reynolds number;
      >= 90 % of the sample);
    * Cp <= ``cp_max`` everywhere (incompressible stagnation: Cp = 1 at most, plus a margin for wall-function noise),
      and the nose's maximum Cp >= 0.9 (the stagnation point is there);
    * Cp integrated over the surface gives the pressure drag predicted by ``cd_pressure(U) -> mean`` (the selected
      scalar model) within ``integral_tol``: the field and the scalar model tell the same story.
    """
    from scipy.stats import qmc

    from pinneapple_analysis.trust import Check
    U = qmc.Sobol(len(VARIABLES), seed=seed).random(n)
    j_speed = [v.name for v in VARIABLES].index("speed")

    def _c(name, ok, value, limit, reason):
        return Check(name, "supports" if ok else "contradicts", value=value, limit=limit,
                     source="surrogate predictions on a Sobol sample of the training region", reason=reason)

    def positive(name):
        def chk(predict):
            m, _ = predict(U)
            return _c(f"{name}_positive", bool(np.all(m > 0)), float(np.min(m)), 0.0, f"min predicted {name}")
        return chk

    def friction_falls(predict):
        m, _ = predict(U)
        U2 = U.copy()
        U2[:, j_speed] = np.minimum(U2[:, j_speed] + 0.05, 1.0)
        m2, _ = predict(U2)
        moved = U2[:, j_speed] > U[:, j_speed]
        frac = float(np.mean((m2 < m)[moved]))
        return _c("CD_friction_falls_with_Re", bool(np.all(m > 0) and frac >= 0.9), frac, 0.9,
                  "share of the sample where CD_friction drops when the speed rises by 5 % of its range")

    lab = region_of(field_vertices(BASELINE), BASELINE)

    def cp_bounds(predict):
        m, _ = predict(U)
        mx, nose = float(np.max(m)), float(np.min(np.max(m[:, lab == "front"], axis=1)))
        return _c("cp_stagnation_bound", mx <= cp_max and nose >= 0.9, mx, cp_max,
                  f"max predicted Cp {mx:.3f} (<= {cp_max}); smallest nose maximum {nose:.3f} (>= 0.9)")

    checks = {"CD": [positive("CD")], "CD_pressure": [positive("CD_pressure")],
              "CD_friction": [friction_falls], "cp": [cp_bounds]}
    if cd_pressure is not None:
        P = unit_to_params(U[:48], bounds)

        def integral(predict):
            m, _ = predict(U[:48])
            cdp_field = np.array([pressure_drag_from_cp(p, cp) for p, cp in zip(P, m, strict=False)])
            cdp = np.asarray(cd_pressure(U[:48]))
            err = float(np.max(np.abs(cdp_field - cdp)))
            return _c("cp_integral_matches_CD_pressure", err <= integral_tol, err, integral_tol,
                      "max |pressure drag of the predicted Cp field - predicted CD_pressure|")
        checks["cp"].append(integral)
    return checks


# ---------------------------------------------------------------------- CAD and imported geometry
_GEO = """SetFactory("OpenCASCADE");
L = {L}; W = {W}; H = {H}; R = {R}; c = {c}; Ls = {Ls}; phi = {phi} * Pi / 180;
// side profile (x, z), drawn in the x-y plane and turned into the x-z plane: front corners rounded with R
Rectangle(1) = {{0, 0, 0, L, H, R}};
Rectangle(2) = {{R, 0, 0, L - R, H, 0}};
s() = BooleanUnion{{ Surface{{1}}; Delete; }}{{ Surface{{2}}; Delete; }};
Rotate {{{{1, 0, 0}}, {{0, 0, 0}}, Pi / 2}} {{ Surface{{s()}}; }}
e() = Extrude {{0, -W, 0}} {{ Surface{{s()}}; }};
Translate {{0, W / 2, c}} {{ Volume{{e(1)}}; }}
// plan profile (x, y), front corners rounded with R, extruded over the height
Rectangle(11) = {{0, -W / 2, c, L, W, R}};
Rectangle(12) = {{R, -W / 2, c, L - R, W, 0}};
p() = BooleanUnion{{ Surface{{11}}; Delete; }}{{ Surface{{12}}; Delete; }};
q() = Extrude {{0, 0, H}} {{ Surface{{p()}}; }};
b() = BooleanIntersection{{ Volume{{e(1)}}; Delete; }}{{ Volume{{q(1)}}; Delete; }};
// rear slant: remove everything above the plane through (L - Ls, top) falling at phi towards the rear
Box(100) = {{L - Ls, -1, c + H, 2, 2, 1}};
Rotate {{{{0, 1, 0}}, {{L - Ls, 0, c + H}}, phi}} {{ Volume{{100}}; }}
f() = BooleanDifference{{ Volume{{b()}}; Delete; }}{{ Volume{{100}}; Delete; }};
Mesh.CharacteristicLengthMax = {h};
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumElementsPerTwoPi = 24;
"""


def ahmed_cad(params: Mapping[str, float], out_dir: str, *, max_size: float = 0.02, gmsh: str = "gmsh") -> dict[str, str]:
    """The Ahmed body as a B-rep in OpenCASCADE (through the gmsh binary): the side and plan profiles with the
    0.1 m front radius intersected, the slant cut by a half-space; exported as STEP (the CAD) and STL (its
    triangulation, edge length <= ``max_size``). Unlike :func:`make_body` the long edges are sharp; the stilts are
    left out in both. Returns the paths of the .geo, .step and .stl files."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    geo = os.path.join(out_dir, "ahmed.geo")
    with open(geo, "w") as f:
        f.write(_GEO.format(L=LENGTH, W=WIDTH, H=HEIGHT, R=NOSE_RADIUS, c=float(params["clearance"]), Ls=SLANT_LENGTH,
                            phi=float(params["slant_deg"]), h=max_size))
    out = {"geo": geo, "step": os.path.join(out_dir, "ahmed.step"), "stl": os.path.join(out_dir, "ahmed.stl")}
    for args in (["-0", "-o", out["step"]], ["-2", "-format", "stl", "-o", out["stl"]]):
        r = subprocess.run([gmsh, geo, *args], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"gmsh failed: {r.stderr[-400:] or r.stdout[-400:]}")
    return out


def read_stl(path: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of an ASCII or binary STL, with coincident vertices merged (1e-9 m)."""
    data = open(path, "rb").read()
    tri = None
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if 84 + 50 * n == len(data):                           # binary STL
            rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            tri = rec["v"].astype(float)
    if tri is None:
        pts = [ln.split()[1:4] for ln in data.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
        tri = np.array(pts, float).reshape(-1, 3, 3)
    V = tri.reshape(-1, 3) * scale
    U, inv = np.unique(np.round(V, 9), axis=0, return_inverse=True)
    return U, inv.reshape(-1, 3)


def import_body(path: str, params: Mapping[str, float], *, scale: float = 1.0) -> dict[str, Any]:
    """An Ahmed-type body from an STL (any source: :func:`ahmed_cad`, a scan, a CAD export) checked before it is
    simulated: closed and manifold, outward (positive volume), symmetric about y = 0, nose at x = 0 and bottom at
    the ``clearance`` of ``params``; its triangles labelled with :func:`region_of` (the named patches: front, roof,
    slant, base, side, underbody). Returns ``{"V", "F", "regions", "issues", "volume", "bounds"}``."""
    from scipy.spatial import cKDTree

    from pinneapple_design.geometry_library import meshops as mo
    V, F = read_stl(path, scale)
    issues = []
    es = mo.edge_stats(V, F)
    if es["boundary_edges"] or es["nonmanifold_edges"]:
        issues.append(f"not closed ({es['boundary_edges']} open, {es['nonmanifold_edges']} non-manifold edges)")
    vol = mo.volume(V, F)
    if vol <= 0:
        F = F[:, ::-1]
        vol = -vol
        issues.append("triangles were inward: flipped")
    lo, hi = V.min(0), V.max(0)
    if abs(lo[0]) > 1e-3 or abs(lo[2] - float(params["clearance"])) > 1e-3:
        issues.append(f"placement: nose at x = {lo[0]:.4f} m, bottom at z = {lo[2]:.4f} m (expected 0 and "
                      f"{float(params['clearance']):g})")
    k = 12                                                      # points per triangle edge for the surface sample
    w = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], float) / k
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    S = (a[:, None] + w[None, :, :1] * (b - a)[:, None] + w[None, :, 1:] * (c - a)[:, None]).reshape(-1, 3)
    d_sym = float(cKDTree(S).query(V * [1, -1, 1])[0].max())     # mirrored vertices to the surface
    if d_sym > 2e-3:
        issues.append(f"not symmetric about y = 0: a mirrored vertex lies {1000 * d_sym:.1f} mm off the surface "
                      "(a half model would be wrong)")
    cent = V[F].mean(1)
    return {"V": V, "F": F, "regions": region_of(cent, params), "issues": [i for i in issues if "flipped" not in i],
            "symmetry_mm": 1000 * d_sym,
            "notes": [i for i in issues if "flipped" in i], "volume": float(vol), "bounds": [lo.tolist(), hi.tolist()]}
