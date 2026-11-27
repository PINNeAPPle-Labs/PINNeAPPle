"""NACA four-digit family in OpenFOAM as the reference simulator of the trust study.

The simulator the lab calls, in the shape every other domain uses: :meth:`AirfoilAero.check` says why a design
cannot be simulated, :meth:`AirfoilAero.run` returns what the solver actually produced and never a made-up number,
and the fields come back on a **fixed lattice** so that one geometry's output lines up with the next.

The lattice is what makes a field surrogate possible here. Geometry varies, so sampling on a Cartesian box would
give a different mask for every design; instead the samples sit at fixed chordwise stations dragged out along the
surface normal at fixed offsets. Every design then has the same number of samples, in the same order, and each one
is a material point of *that* section -- which is what a surrogate can learn across geometries.

Design variables (:data:`VARIABLES`): angle of attack, freestream speed, relative thickness, and maximum camber --
the last one is what the study opens as a new parameter dimension after the first model is trained and validated.
Fixed (:data:`CONDITIONS`): unit chord, sea-level air, steady RANS k-omega SST with wall functions. Outputs: lift,
drag and its pressure and friction parts, and the pitching moment about the quarter chord.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from pinneapple_design.design_optimizer.study import Variable
from pinneapple_simulation.numerical_solvers import airfoil2d as A2

from ..simulator import SimOutcome, Simulator

__all__ = ["VARIABLES", "CONDITIONS", "BASELINE", "STATIONS", "OFFSETS", "MESH_LEVELS", "AirfoilAero",
           "sample_lattice", "cp_from_probe", "section_loop"]

VARIABLES = [
    Variable("alpha_deg", -4.0, 8.0, unit="deg", description="angle of attack"),
    Variable("speed", 10.0, 30.0, unit="m/s", description="freestream speed"),
    Variable("thickness", 0.08, 0.16, unit="1", description="maximum thickness over chord"),
    Variable("camber", 0.0, 0.02, unit="1", description="maximum camber over chord"),
]
BASELINE = {"alpha_deg": 4.0, "speed": 20.0, "thickness": 0.12, "camber": 0.0}
CONDITIONS = {"chord_m": 1.0, "span_m": 0.01, "nu_m2_s": 1.5e-5, "rho_kg_m3": 1.225,
              "camber_position": 0.4, "turbulence": "steady RANS k-omega SST, wall functions",
              "solver": "OpenFOAM simpleFoam, one cell in z, empty front and back"}

#: chordwise stations and wall-normal offsets of the field lattice (fractions of the chord)
STATIONS = np.linspace(0.0, 1.0, 41)
#: how far off the surface the wall samples sit, as a fraction of the chord: below the first cell by two orders
_WALL_EPS = 2e-5
OFFSETS = np.array([5e-4, 2e-3, 6e-3, 1.5e-2, 4e-2, 1e-1, 2.5e-1])

#: mesh levels: (first cell height, spacing at the leading and trailing edges, maximum spacing at the far field), and
#: the refinement ratio of the level against L2. The family scales *every* spacing by 1.45 per level, so the cell
#: count grows by about 1.8 between consecutive levels and the grid-convergence study gets the refinement ratio the
#: meshes were actually built with. The eight-layer boundary layer follows from ``h_wall`` and a fixed ratio, so it
#: is self-similar across the family by construction. L2 is the mesh both campaigns run on; L3 exists only to give
#: the study a mesh finer than the production one, and L0/L1 only to give it coarser ones.
MESH_LEVELS = {
    "L0": {"h_wall": 0.0048, "s_min": 0.0048, "s_max": 0.0420, "h_far": 2.52, "r": 1.45 ** 2},
    "L1": {"h_wall": 0.0033, "s_min": 0.0033, "s_max": 0.0286, "h_far": 1.74, "r": 1.45},
    "L2": {"h_wall": 0.0023, "s_min": 0.0023, "s_max": 0.0200, "h_far": 1.20, "r": 1.0},
    "L3": {"h_wall": 0.0016, "s_min": 0.0016, "s_max": 0.0138, "h_far": 0.83, "r": 1.0 / 1.45},
}


def section_loop(params: Mapping[str, float], *, n: int = 200) -> np.ndarray:
    """The closed loop of a design's section, resampled for a mesh of the study's spacing."""
    t = float(params["thickness"])
    m = float(params["camber"])
    p = float(CONDITIONS["camber_position"])
    return A2.closed_loop(m, p, t, n)


def sample_lattice(params: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """The fixed lattice of a design: probe points (N, 3) and a mask of the samples that sit on the wall.

    ``2 * len(STATIONS)`` wall samples (upper then lower surface, leading to trailing edge) followed by
    ``2 * len(STATIONS) * len(OFFSETS)`` samples along the outward normal at the same stations, ordered
    (station, offset, side). Points are on the plane z = 0, the probes only need x and y.
    """
    xu, yu, xl, yl = A2.naca4_surfaces(float(params["camber"]), float(CONDITIONS["camber_position"]),
                                       float(params["thickness"]), 400)
    pts, on_wall = [], []
    for surface in (0, 1):                                 # 0 upper, 1 lower
        xs, ys = (xu, yu) if surface == 0 else (xl, yl)
        x = STATIONS
        y = np.interp(x, xs, ys)                          # surface ordinate at the stations
        dy = np.gradient(y, x)                            # normal of the local tangent ...
        nrm = np.column_stack([dy, -np.ones_like(dy)])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        if surface == 0:                                  # ... flipped to point *out* of the section: the upper
            nrm *= -1.0                                   # surface ordinate decreases after the maximum thickness
        for px, py, nv in zip(x, y, nrm, strict=True):
            # the wall sample is lifted off the surface by a hundredth of the first cell: a point exactly on the wall
            # is on a boundary face, and on a sharp trailing edge OpenFOAM's search does not always place it -- the
            # sample then comes back as "not found", which is the sentinel and not a velocity. At 2e-5 of the chord
            # the velocity there is still indistinguishable from the no-slip value the check is looking for.
            pts.append((px + _WALL_EPS * nv[0], py + _WALL_EPS * nv[1], 0.0))
            on_wall.append(True)
            for off in OFFSETS:
                pts.append((px + off * nv[0], py + off * nv[1], 0.0))
                on_wall.append(False)
    return np.array(pts, float), np.array(on_wall, bool)


def cp_from_probe(p: np.ndarray, speed: float) -> np.ndarray:
    """Pressure coefficient from the solver's kinematic pressure."""
    return np.asarray(p, float) / (0.5 * float(speed) ** 2)


class AirfoilAero(Simulator):
    """OpenFOAM on one section: mesh, steady RANS, force coefficients and the field lattice."""

    name = "OpenFOAM simpleFoam (gmsh hex mesh, k-omega SST), NACA four-digit section"
    fidelity = "reference"
    outputs = ("CL", "CD", "CD_pressure", "CD_friction", "CM")
    field_names = ("u", "v", "cp")

    def __init__(self, mesh: str = "L2", iterations: int = 1200, procs: int = 1, keep_cases: bool = False,
                 average_from: float = 0.6, allow_camber: bool = True, max_rel_spread: float = 0.01,
                 conditions: Mapping[str, Any] | None = None):
        if mesh not in MESH_LEVELS:
            raise ValueError(f"unknown mesh {mesh!r}; use one of {sorted(MESH_LEVELS)}")
        self.mesh, self.iterations, self.procs = mesh, iterations, procs
        self.keep_cases, self.average_from = keep_cases, average_from
        self.max_rel_spread = max_rel_spread
        self.conditions = dict(CONDITIONS if conditions is None else conditions)
        self.allow_camber = allow_camber
        pts, on_wall = sample_lattice(BASELINE)
        self.n_points = len(pts)
        self.regions = {"wall": on_wall, "off_wall": ~on_wall}

    # ------------------------------------------------------------------ interface
    def field_points(self, params: Mapping[str, float]) -> np.ndarray:
        pts, _ = sample_lattice(params)
        if len(pts) != self.n_points:
            raise RuntimeError("the sampling lattice changed size")
        return pts

    def check(self, params: Mapping[str, float]) -> list[str]:
        """Why this design cannot be simulated as asked; empty when it can."""
        issues: list[str] = []
        for v in VARIABLES:
            if v.name not in params:
                issues.append(f"missing {v.name}")
            elif not v.lower - 1e-9 <= float(params[v.name]) <= v.upper + 1e-9:
                issues.append(f"{v.name} = {params[v.name]:g} outside [{v.lower:g}, {v.upper:g}]")
        if issues:
            return issues
        if float(params["camber"]) > 0 and not self.allow_camber:
            issues.append("camber is not a parameter of this model's schema")
        if self.conditions != CONDITIONS:
            diff = {k: v for k, v in self.conditions.items() if CONDITIONS.get(k) != v}
            issues.append(f"conditions differ from the study's: {diff}")
        return issues

    def run(self, params: Mapping[str, float], workdir: str) -> SimOutcome:
        t0 = time.perf_counter()
        case = os.path.abspath(workdir)
        level = MESH_LEVELS[self.mesh]
        speed = float(params["speed"])
        meta: dict[str, Any] = {"mesh": self.mesh, "iterations_requested": self.iterations,
                                "procs": self.procs, "params": {k: float(v) for k, v in params.items()},
                                "Re": speed * float(self.conditions["chord_m"]) / self.conditions["nu_m2_s"]}
        pts, on_wall = sample_lattice(params)
        try:
            A2.write_case_2d(case, speed=speed, alpha_deg=float(params["alpha_deg"]),
                             nu=float(self.conditions["nu_m2_s"]), rho=float(self.conditions["rho_kg_m3"]),
                             iterations=self.iterations, dz=float(self.conditions["span_m"]), probes=pts,
                             procs=self.procs)
            loop = A2.arclength_resample(section_loop(params), level["s_min"], level["s_max"])
            A2.write_mesh(case, loop, h_wall=level["h_wall"], h_far=level["h_far"], radius=12.0)
            meta["mesh_timing_s"] = time.perf_counter() - t0
            timing = A2.run_case(case, procs=self.procs)
            meta.update(timing)
            meta["mesh_quality"] = A2.check_mesh(case)
            meta.update(A2.solver_summary(case))
            if not timing["solver_ok"] or meta.get("fatal"):
                raise RuntimeError(f"simpleFoam did not finish: {meta.get('fatal')}")
            coeffs = A2.read_coefficients(case, average_from=self.average_from)
            if not coeffs:
                raise RuntimeError("no force coefficients were written")
            # A relative spread is only a measurement on a coefficient that does not cross zero: lift is ~0 on a
            # symmetric section at zero incidence, and the moment is ~0 almost everywhere, so their relative spread
            # explodes while the absolute one is a few units in the fifth decimal. The gate therefore reads each
            # coefficient against its own value *or* one drag-count-per-ten floor, and leaves the moment out of it
            # (it is reported, not gated).
            gated = ("CL", "CD", "CD_pressure", "CD_friction")
            loose = {k: abs(coeffs.get(f"{k}_std", 0.0)) / max(abs(coeffs[k]), 5e-3) for k in gated}
            meta["coefficient_rel_spread"] = {k: abs(coeffs.get(f"{k}_std", 0.0)) / max(abs(coeffs[k]), 5e-3)
                                              for k in self.outputs}
            meta["settling_gate"] = {"gated": list(gated), "floor": 5e-3,
                                     "limit": self.max_rel_spread, "rejected": []}
            worst = max(loose, key=loose.get)
            if loose[worst] > self.max_rel_spread:
                meta["settling_gate"]["rejected"] = sorted(k for k, v in loose.items() if v > self.max_rel_spread)
                raise RuntimeError(f"{worst} not settled: spread {loose[worst]:.2%} over the averaged "
                                   f"iterations (limit {self.max_rel_spread:.0%})")
            probes = A2.read_probes(case, self.n_points)
            if "U" not in probes or "p" not in probes:
                raise RuntimeError("the probes object wrote no samples")
            # OpenFOAM reports a point it cannot locate as +/-1e300 on every row it could not interpolate. A field
            # with those in it trains a surrogate on a number that means "I do not know", so the run is rejected
            # rather than recorded: it is a defect of the case, not a hard design.
            missing = {k: int((np.abs(np.asarray(v, float)) > 1e100).sum()) for k, v in probes.items()}
            if any(missing.values()):
                raise RuntimeError(f"the probes were not found at {missing} of the {self.n_points} lattice points: "
                                   "the sampling plane is not inside the mesh")
        except Exception as e:                                   # noqa: BLE001 - the failure is the outcome
            meta["error"] = f"{type(e).__name__}: {e}"
            if not self.keep_cases:
                self._cleanup(case)
            return SimOutcome({}, False, meta["error"], time.perf_counter() - t0, None, meta)

        u = np.asarray(probes["U"])[:, 0]
        v = np.asarray(probes["U"])[:, 1]
        cp = cp_from_probe(np.asarray(probes["p"]).ravel(), speed)
        out = {k: float(coeffs[k]) for k in self.outputs}
        meta["coefficient_std"] = {k: float(coeffs.get(f"{k}_std", np.nan)) for k in self.outputs}
        meta["averaged_iterations"] = coeffs.get("averaged_iterations")
        bad = [k for k, val in out.items() if not np.isfinite(val)]
        if bad:
            return SimOutcome({}, False, f"non-finite outputs {bad}", time.perf_counter() - t0, None, meta)
        if not self.keep_cases:
            self._cleanup(case)
        return SimOutcome(out, True, "", time.perf_counter() - t0, {"u": u, "v": v, "cp": cp}, meta)

    def _cleanup(self, case: str) -> None:
        import shutil
        if os.path.isdir(case):
            shutil.rmtree(case, ignore_errors=True)
