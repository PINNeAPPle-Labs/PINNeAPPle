"""Angle (shelf) bracket that holds a load at the end of its arm: thickness, width, mounting-leg length, material.

Physics: linear elasticity. Models: cantilever beam theory (closed form), an in-house hexahedral finite-element model
of the L with the fillet as a staircase (:class:`SolidFEM`, C3D8I), and CalculiX on a gmsh tetrahedral mesh (C3D10)
of the exact geometry with the filleted corner and the bolt holes. Failure modes: yield at the root (peak von Mises,
the fillet's stress concentration measured by CalculiX), deflection at the load (when a limit is stated), and the
bolts in tension (moment about the toe of the mounting leg).
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from .. import materials as M
from ..physics import bracket as B
from ..spec import ObjectiveChoice, Question, RequirementSpec, Slot
from ..template import (
    CLOSED_FORM,
    FULL,
    REDUCED,
    DesignVariable,
    FailureMode,
    Fidelity,
    FullResult,
    ProblemTemplate,
    Uncertain,
)

G0 = 9.80665
BRACKET_MATERIALS = ("al6061_t6", "steel_1018", "ss304")
HOLE = 6.6e-3          # ISO 273 medium clearance hole for M6
ISO273 = "ISO 273: medium clearance hole for an M6 bolt, 6.6 mm"
# ISO 898-1 property class 8.8, M6: tensile stress area 20.1 mm^2, proof stress 580 MPa (d <= 16 mm)
BOLT_PROOF_N = 20.1e-6 * 580e6
ISO898 = "ISO 898-1, property class 8.8, M6: stress area 20.1 mm^2 x proof stress 580 MPa"
_LOAD_WORDS = ("holds", "hold", "holding", "load", "carry", "carries", "support", "supports", "force",
               "withstand", "rated")


class BracketTemplate(ProblemTemplate):
    name = "bracket"
    title = "Load-bearing angle bracket"
    domain = "structures (linear elasticity)"
    reduced_is_expensive = True         # a finite-element solve per design
    keywords = {"bracket": 2.0, "angle bracket": 1.0, "shelf bracket": 1.0, "l-bracket": 1.0, "mount": 0.5,
                "holds": 0.3, "load": 0.3, "cantilever": 0.5, "kn": 0.3}
    physics = ("linear elasticity of the bracket (small strains, isotropic material)",
               "yield by the von Mises criterion at the most stressed point (the fillet)",
               "bolts in tension from the moment about the toe of the mounting leg")
    assumptions = ("the load acts at the end of the arm, perpendicular to it (a shelf load)",
                   "the mounting leg's face is fully clamped by its two M6 bolts (the support is rigid)",
                   "inner-corner fillet radius equal to the thickness",
                   "static load; no fatigue, impact or temperature effects")
    output_units = {"sigma_peak_mpa": "MPa", "deflection_mm": "mm", "mass_g": "g", "bolt_tension_n": "N",
                    "sigma_nominal_mpa": "MPa"}
    output_notes = {"sigma_peak_mpa": "peak von Mises (nominal x stress-concentration factor)",
                    "bolt_tension_n": "per bolt, moment about the toe"}
    slots = (
        Slot("load", "load at the end of the arm", "N", "kN", _LOAD_WORDS, bounds=(1.0, 50e3), role="load"),
        Slot("load_mass", "mass held at the end of the arm", "kg", "kg", _LOAD_WORDS + ("weighs", "mass"),
             bounds=(0.1, 5000.0), role="load"),
        Slot("arm_length", "distance from the mounting face to the load", "m", "mm",
             ("arm", "reach", "from the wall", "projects", "overhang", "lever", "long", "length", "offset", "at",
              "distance", "deep"), bounds=(0.02, 0.5), role="scale", critical=True,
             question="How far from the mounting face does the load act (arm length)?"),
        Slot("max_deflection", "allowed deflection at the load", "m", "mm", ("deflect", "deflection", "sag",
                                                                             "sags", "bends"),
             bounds=(1e-5, 0.05), role="limit"),
        Slot("material", "material", "choice", keywords=("material", "made of"), choices=BRACKET_MATERIALS,
             role="material"),
        Slot("safety_factor", "safety factor on yield", "", "", ("safety factor", "factor of safety", "sf"),
             default=2.0, default_source="template default (engineering convention for a known static load; "
                                         "state yours)", bounds=(1.0, 10.0)),
    )
    objectives = (
        ObjectiveChoice("mass_g", "min", "mass", ("light", "lightest", "weight", "cheap material")),
        ObjectiveChoice("deflection_mm", "min", "deflection", ("stiff", "stiffest", "rigid")),
    )
    failure_modes = (
        FailureMode("margin_yield", "no yield at the most stressed point",
                    "yield / (safety factor x peak von Mises) - 1"),
        FailureMode("margin_bolts", "bolts stay below their proof load",
                    "proof load / (safety factor x bolt tension) - 1"),
        FailureMode("margin_deflection", "deflection within the stated limit",
                    "allowed / computed deflection - 1 (no limit stated: not constraining)", critical=False),
    )
    fidelities = (
        Fidelity(CLOSED_FORM, "cantilever beam theory", "root stress M c / I and tip deflection P L^3 / (3 E I)"),
        Fidelity(REDUCED, "hexahedral FE (SolidFEM)", "C3D8I structured mesh of the L, staircase fillet, clamped "
                                                      "mounting face; nominal stress x stress-concentration "
                                                      "factor", modules=("scipy",)),
        Fidelity(FULL, "CalculiX (gmsh C3D10)", "exact fillet and bolt holes, two meshes, peak von Mises, "
                                                "deflection, reactions", executables=("ccx",),
                 check=lambda: (True, "") if B.gmsh_exe() else (False, "program 'gmsh' is not installed")),
    )

    # ------------------------------------------------------------------ scope and questions
    def check_scope(self, spec: RequirementSpec) -> list[str]:
        low = spec.text.lower()
        for w in ("fatigue", "cyclic", "vibration", "impact", "crash", "seismic", "earthquake", "fire"):
            if w in low:
                return [f"'{w}': this template checks a static load only; {w} needs a load history and the "
                        "matching material data (S-N curves, strain-rate or high-temperature properties)"]
        return []

    def extra_questions(self, spec: RequirementSpec) -> list[Question]:
        if spec.get("load") is None and spec.get("load_mass") is None:
            return [Question("load", "What load must the bracket hold (force or mass)?", "kN",
                             why="needed by the physics and has no safe default")]
        return []

    # ------------------------------------------------------------------ variables
    def load_n(self, spec: RequirementSpec) -> float:
        return spec["load"] if spec.get("load") is not None else spec["load_mass"] * G0

    def _materials(self, spec: RequirementSpec) -> tuple[str, ...]:
        m = spec.get("material")
        return (m,) if m else BRACKET_MATERIALS

    def variables(self, spec: RequirementSpec) -> list[DesignVariable]:
        # whole millimetres (stock plate, a buildable drawing); the width leaves an edge distance and a ligament of
        # one hole diameter around the two M6 clearance holes
        vs = [DesignVariable("thickness_mm", 2, 25, "mm", "plate thickness t", integer=True),
              DesignVariable("width_mm", 36, 120, "mm", "width w", integer=True),
              DesignVariable("mount_mm", 40, 200, "mm", "mounting-leg length a", integer=True)]
        mats = self._materials(spec)
        if len(mats) > 1:
            vs.append(DesignVariable("material", 0, len(mats) - 1, "", "material", choices=mats))
        return vs

    def _unpack(self, X, spec):
        X = np.atleast_2d(X)
        vs = self.variables(spec)
        p = {v.name: X[:, j] for j, v in enumerate(vs)}
        for k in ("thickness", "width", "mount"):
            p[k] = np.round(p[f"{k}_mm"]) * 1e-3
        mats = self._materials(spec)
        idx = np.clip(np.round(p.get("material", np.zeros(len(X)))), 0, len(mats) - 1).astype(int)
        p["mat"] = [M.get(mats[i]) for i in idx]
        return p

    def full_design(self, design: dict[str, Any], spec: RequirementSpec) -> dict[str, Any]:
        d = dict(design)
        d.setdefault("material", self._materials(spec)[0])
        for k in ("thickness", "width", "mount"):
            d[k] = d[f"{k}_mm"] * 1e-3
        d["arm_length"] = spec["arm_length"]
        d["fillet"] = d["thickness"]
        d["hole"] = HOLE
        return d

    # ------------------------------------------------------------------ models
    def evaluate(self, level: str, X: np.ndarray, spec: RequirementSpec, *, kt: float = 1.0, k_delta: float = 1.0,
                 load_scale=1.0, t_delta=0.0, yield_scale=1.0, **_: Any) -> dict[str, np.ndarray]:
        p = self._unpack(X, spec)
        t = p["thickness"] + np.asarray(t_delta, float)
        w, a = p["width"], p["mount"]
        b = spec["arm_length"]
        P = self.load_n(spec) * np.asarray(load_scale, float)
        E = np.array([m["youngs_modulus"] for m in p["mat"]])
        nu = np.array([m["poisson"] for m in p["mat"]])
        sy = np.array([m["yield_strength"] for m in p["mat"]]) * np.asarray(yield_scale, float)
        rho = np.array([m["density"] for m in p["mat"]])
        n = len(t)
        if level == CLOSED_FORM:
            sig, dlt = B.beam(P, b, w, t, E)
            kt_used, kd_used = 1.0, 1.0
        elif level == REDUCED:
            sig, dlt = np.empty(n), np.empty(n)
            for i in range(n):
                dlt[i], sig[i], _, _ = B.hex_model(P, a[i], b, w[i], t[i], E[i], nu[i], n_t=2, fillet=t[i])
            kt_used, kd_used = kt, k_delta
        else:
            raise ValueError(f"level {level!r}: the full model runs through verify()")
        peak = kt_used * sig
        delta = kd_used * dlt
        sf = spec["safety_factor"]
        x_h = t + t + 0.5 * (a - 2 * t)                 # bolt line: middle of the mounting leg beyond the fillet
        tension = P * b / np.maximum(2 * (a - x_h), 1e-6)
        dmax = spec.get("max_deflection")
        vol = (a * w * t + (b - t) * w * t + (1 - math.pi / 4) * t * t * w - 2 * math.pi * (HOLE / 2) ** 2 * t)
        return {"margin_yield": sy / (sf * peak) - 1.0,
                "margin_bolts": BOLT_PROOF_N / (sf * tension) - 1.0,
                "margin_deflection": (dmax / np.abs(delta) - 1.0) if dmax else np.ones(n),
                "mass_g": rho * vol * 1e3, "deflection_mm": np.abs(delta) * 1e3, "sigma_peak_mpa": peak / 1e6,
                "sigma_nominal_mpa": sig / 1e6, "bolt_tension_n": tension}

    # ------------------------------------------------------------------ full fidelity
    def verify(self, design: dict[str, Any], spec: RequirementSpec, workdir: str | None = None) -> FullResult:
        import tempfile
        d = self.full_design(design, spec)
        mat = M.get(d["material"])
        P, b, a, w, t = self.load_n(spec), d["arm_length"], d["mount"], d["width"], d["thickness"]
        work = workdir or tempfile.mkdtemp(prefix="bracket_ccx_")
        peaks, sols = [], []
        for k, size in enumerate(B.mesh_sizes(t)):
            m, r, pk, where, delta, reac = B.calculix_model(P, a, b, w, t, d["fillet"], d["hole"],
                                                            mat["youngs_modulus"], mat["poisson"], mat["density"],
                                                            size, os.path.join(work, f"mesh{k}"))
            peaks.append(pk)
            sols.append((m, r, where, delta, reac, size))
        m, r, where, delta, reac, size = sols[-1]
        s_fem, zc, nsec = B.section_stress(m, r, b, w, t)
        s_ref = P * (b - zc) * (t / 2) / (w * t ** 3 / 12)
        x_h = 2 * t + 0.5 * (a - 2 * t)
        tension = P * b / (2 * (a - x_h))
        sf = spec["safety_factor"]
        out = {"sigma_peak_mpa": peaks[-1] / 1e6, "deflection_mm": abs(delta) * 1e3,
               "margin_yield": mat["yield_strength"] / (sf * peaks[-1]) - 1.0,
               "margin_bolts": BOLT_PROOF_N / (sf * tension) - 1.0,
               "elements": float(len(m.elements))}
        dmax = spec.get("max_deflection")
        if dmax:
            out["margin_deflection"] = dmax / abs(delta) - 1.0
        checks = [
            {"name": "equilibrium (support reaction = load)", "passed": abs(reac / P - 1) < 0.005, "critical": True,
             "value": abs(reac / P - 1), "limit": 0.005, "detail": f"reaction {reac:.1f} N for a load of {P:.1f} N"},
            {"name": "arm bending stress vs M c / I", "passed": abs(s_fem / s_ref - 1) < 0.06, "critical": True,
             "value": abs(s_fem / s_ref - 1), "limit": 0.06,
             "detail": f"outer face at z = {zc * 1e3:.0f} mm: {s_fem / 1e6:.1f} vs {s_ref / 1e6:.1f} MPa "
                       f"({nsec} nodes)"},
            {"name": "mesh convergence of the peak stress", "passed": abs(peaks[1] / peaks[0] - 1) < 0.05,
             "value": abs(peaks[1] / peaks[0] - 1), "limit": 0.05,
             "detail": f"peak von Mises {peaks[0] / 1e6:.1f} MPa (h = {B.mesh_sizes(t)[0] * 1e3:.1f} mm) vs "
                       f"{peaks[1] / 1e6:.1f} MPa (h = {B.mesh_sizes(t)[1] * 1e3:.1f} mm)"},
        ]
        cad = {}
        try:
            cad = B.export_cad(B.geo_shelf_bracket(a, b, w, t, d["fillet"], d["hole"]), os.path.join(work, "cad"))
        except (RuntimeError, OSError) as e:  # pragma: no cover - gmsh present when this runs
            cad = {"error": str(e)}
        cand_x = np.array([[design[v.name] if v.name != "material" else
                            self._materials(spec).index(design["material"]) for v in self.variables(spec)]])
        nominal = float(self.evaluate(REDUCED, cand_x, spec)["sigma_nominal_mpa"][0])
        notes = [f"CalculiX on gmsh C3D10 tetrahedra ({len(m.elements)} elements at h = {size * 1e3:.1f} mm); "
                 "peak von Mises taken away from the clamped face (its edge is a singularity of the idealised "
                 "clamp) and from the loaded face",
                 f"peak at x = {where[0] * 1e3:.1f}, y = {where[1] * 1e3:.1f}, z = {where[2] * 1e3:.1f} mm",
                 f"measured stress-concentration factor of the fillet: {peaks[-1] / 1e6 / nominal:.3f} "
                 "(CalculiX peak / hexahedral-model nominal stress)"]
        return FullResult(out, checks, FULL, self.fidelity(FULL).label, 0.0,
                          {"model": m, "result": r, "cad": cad, "kt": peaks[-1] / 1e6 / nominal}, notes,
                          compare={"sigma_peak_mpa": "sigma_peak_mpa", "deflection_mm": "deflection_mm"},
                          compare_options={"load_scale": 1.0, "t_delta": 0.0, "yield_scale": 1.0})

    robust_level = CLOSED_FORM           # Monte Carlo on beam theory, calibrated to the finite-element margins

    def uncertainties(self, spec: RequirementSpec, design: dict[str, Any]) -> list[Uncertain]:
        return [Uncertain("load_scale", 1.0, 0.10, "uniform", "", "load +-10 %: assumed (state the load's own "
                                                                   "uncertainty)"),
                Uncertain("t_delta", 0.0, 2e-4, "uniform", "m", "plate thickness +-0.2 mm: assumed stock "
                                                                "tolerance (check your supplier's)"),
                Uncertain("yield_scale", 0.95, 0.05, "uniform", "", "yield strength 90-100 % of the datasheet "
                                                                    "value: assumed lot scatter")]

    def design_options(self, spec: RequirementSpec) -> dict[str, Any]:
        """Size at the worst corner of the uncertainties (load +10 %, plate -0.2 mm, yield 90 %)."""
        return {"load_scale": 1.10, "t_delta": -2e-4, "yield_scale": 0.90}

    def calibrate(self, full: FullResult) -> dict[str, float]:
        """Evaluate options measured by the full solver: the fillet's stress-concentration factor."""
        kt = full.artifacts.get("kt")
        return {"kt": float(kt)} if kt and math.isfinite(kt) else {}

    # ------------------------------------------------------------------ package
    def geometry(self, spec: RequirementSpec, design: dict[str, Any]):
        from pinneapple_design.geometry_library.core import Body
        from pinneapple_design.geometry_library.kit import extrude

        d = self.full_design(design, spec)
        a, b, w, t, f = d["mount"], d["arm_length"], d["width"], d["thickness"], d["fillet"]
        ang = np.linspace(0, math.pi / 2, 9)
        arc = np.c_[t + f - f * np.cos(ang), t + f - f * np.sin(ang)][::-1]
        poly = np.vstack([[0, 0], [a, 0], [a, t], [t + f, t], *arc[1:-1], [t, t + f], [t, b], [0, b]])
        V, F = extrude(poly, 0.0, w, plane="xz")
        colour = "aluminium" if d["material"].startswith("al") else "steel"
        return [Body("bracket", V, F, colour)]

    def bill_of_materials(self, spec: RequirementSpec, design: dict[str, Any]) -> list[dict[str, Any]]:
        d = self.full_design(design, spec)
        mat = M.get(d["material"])
        x = np.array([[design[v.name] if v.name != "material" else self._materials(spec).index(design["material"])
                       for v in self.variables(spec)]])
        mass = float(self.evaluate(CLOSED_FORM, x, spec)["mass_g"][0])
        return [{"item": "angle bracket", "material": mat.label, "quantity": 1,
                 "size": f"t = {d['thickness'] * 1e3:.1f} mm, w = {d['width'] * 1e3:.1f} mm, mounting leg "
                         f"{d['mount'] * 1e3:.0f} mm, arm {d['arm_length'] * 1e3:.0f} mm, inner fillet "
                         f"R{d['fillet'] * 1e3:.1f}", "mass_g": round(mass, 1),
                 "properties": {k: {"value": mat.props[k].value, "unit": mat.props[k].unit, "source": mat.cite(k)}
                                for k in ("density", "youngs_modulus", "poisson", "yield_strength")}},
                {"item": "bolts M6, property class 8.8, with washers", "material": "steel", "quantity": 2,
                 "size": f"clearance holes {HOLE * 1e3:.1f} mm", "mass_g": None,
                 "properties": {"proof_load": {"value": BOLT_PROOF_N, "unit": "N", "source": ISO898},
                                "hole": {"value": HOLE, "unit": "m", "source": ISO273}}}]

    def operating_window(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]
                         ) -> list[dict[str, Any]]:
        P = self.load_n(spec)
        ver = ctx.get("verification")
        full = ver.result.outputs if ver is not None and ver.ran and ver.result else None
        cand = ctx["candidate"]
        my = full["margin_yield"] if full else cand.outputs["margin_yield"]
        mb = cand.outputs["margin_bolts"]
        fid = "CalculiX" if full else "hexahedral FE"
        sf = spec["safety_factor"]
        p_yield = P * (1 + my) * sf
        return [
            {"quantity": "load at the arm's end", "value": round(P, 1), "min": 0.0, "max": round(P * (1 + my), 1),
             "unit": "N", "fidelity": fid, "note": f"max: yield / {sf:g} at the fillet (linear: stress scales with "
                                                   "the load)"},
            {"quantity": "load at first yield (no safety factor)", "value": None, "min": None,
             "max": round(p_yield, 1), "unit": "N", "fidelity": fid, "note": "linear elastic extrapolation"},
            {"quantity": "load at bolt proof (no safety factor)", "value": None, "min": None,
             "max": round(P * (1 + mb) * sf, 1), "unit": "N", "fidelity": "closed form (moment about the toe)",
             "note": "two M6 8.8 bolts"},
            {"quantity": "bolt tightening", "value": None, "min": None, "max": None, "unit": "",
             "fidelity": "-", "note": "follow the bolt and anchor supplier's torque; the wall anchor is not checked"},
        ]

    def build_instructions(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
        d = self.full_design(design, spec)
        mat = M.get(d["material"])
        return [f"Make the bracket from {mat.label}: {d['thickness'] * 1e3:.1f} mm plate, {d['width'] * 1e3:.1f} mm "
                f"wide; mounting leg {d['mount'] * 1e3:.0f} mm, arm {d['arm_length'] * 1e3:.0f} mm (bracket.step).",
                f"Inner corner fillet R{d['fillet'] * 1e3:.1f} mm (bend radius if formed from sheet; machined "
                "otherwise). A sharp inner corner raises the peak stress above the verified value.",
                f"Two {HOLE * 1e3:.1f} mm clearance holes for M6 bolts in the mounting leg, on its centre line "
                "between the fillet and the end (positions in bracket.step).",
                "Bolt to a rigid, flat support with two M6 8.8 bolts and washers.",
                "Apply the load at the end of the arm, perpendicular to it, as assumed in the design."]

    def limits(self, spec: RequirementSpec) -> list[str]:
        return ["Static load only: no fatigue, impact, vibration or creep.",
                "The mounting face is idealised as fully clamped; a flexible wall or anchor changes the stresses "
                "and is not checked (nor is the anchor's pull-out).",
                "Bolt check: tension from the moment about the toe; prying and shear are not included.",
                "Linear elastic: valid while the peak stays below yield (the safety factor keeps it there)."]

    def figures(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any], out_dir: str
                ) -> dict[str, str]:
        from ..figures import bracket_figures
        return bracket_figures(self, spec, self.full_design(design, spec), ctx, out_dir)
