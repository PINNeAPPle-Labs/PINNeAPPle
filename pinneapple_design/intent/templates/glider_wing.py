"""Wing of a model glider: span, aspect ratio, taper and section for the lowest sink rate.

Physics: low-speed external aerodynamics. Lift from the wing's lift curve, induced drag from the span loading
(vortex lattice, :mod:`pinneapple_design.aero.vlm`), profile drag from a flat-plate skin friction times a thickness
form factor. Models: elliptic lifting line (closed form, e = 1), the vortex lattice on a coarse lattice (reduced),
and the vortex lattice on a fine lattice (full; a RANS model is the next step and is not run here).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..spec import ObjectiveChoice, RequirementSpec, Slot
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
RHO_ISA = 1.225          # ICAO Doc 7488, standard atmosphere at sea level
SECTIONS = ("NACA 2412", "NACA 4412")
SECTION_TC = {"NACA 2412": 0.12, "NACA 4412": 0.12}
RE_MIN = 7.0e4           # Lissaman (1983), Annu. Rev. Fluid Mech. 15: critical Reynolds number ~ 70 000
CF_SRC = "turbulent flat plate C_f = 0.074 Re^-0.2 (Prandtl; Schlichting, Boundary-Layer Theory)"
FF_SRC = "form factor 1 + 2 t/c + 60 (t/c)^4 (Hoerner, Fluid-Dynamic Drag, 1965)"
NU_AIR = 1.5e-5


def profile_cd0(re, tc):
    """Wing profile drag coefficient on the planform area: 2 (1 + 0.2 t/c) x C_f(Re) x FF (wetted area of a thin
    wing about twice the planform)."""
    cf = 0.074 * np.asarray(re, float) ** -0.2
    ff = 1 + 2 * tc + 60 * tc ** 4
    return 2 * (1 + 0.2 * tc) * cf * ff


class GliderWingTemplate(ProblemTemplate):
    name = "glider_wing"
    title = "Model glider wing"
    domain = "external aerodynamics (low speed)"
    keywords = {"glider": 2.5, "wing": 1.5, "sailplane": 2.0, "glide": 1.0, "soar": 1.0, "sink rate": 1.0,
                "lift": 0.5, "span": 0.5}
    physics = ("steady, incompressible, attached flow around the wing",
               "lift = weight in steady glide; induced drag from the span loading (vortex lattice / lifting line)",
               "profile drag from turbulent flat-plate friction and a thickness form factor",
               "sink rate = V C_D / C_L")
    assumptions = ("wing alone: the fuselage and tail drag are not included (the sink rate is the wing's share)",
                   f"air density {RHO_ISA} kg/m^3 (ICAO standard atmosphere, sea level)",
                   "fully turbulent skin friction (conservative for the profile drag at model Reynolds numbers)",
                   "no twist, no sweep, no dihedral in the aerodynamic model")
    output_units = {"sink_m_s": "m/s", "speed_m_s": "m/s", "cl": "-", "span_eff": "-", "ld": "-",
                    "area_m2": "m²", "re_mac": "-"}
    slots = (
        Slot("mass", "flying mass", "kg", "kg", ("glider", "weighs", "mass", "weight", "for a"), bounds=(0.05, 20),
             critical=True, role="load", question="What is the glider's flying mass?"),
        Slot("max_span", "largest span", "m", "m", ("span", "wingspan", "wide"), default=2.0,
             default_source="template default envelope (2 m); state yours", bounds=(0.3, 6.0), role="limit"),
        Slot("cl_limit", "highest design lift coefficient", "", "", ("cl max", "lift coefficient"), default=0.8,
             default_source="template design limit kept below stall; stall is not predicted by these models",
             bounds=(0.2, 1.4)),
    )
    objectives = (
        ObjectiveChoice("sink_m_s", "min", "minimum sink rate", ("sink", "thermal", "stay up", "float",
                                                                  "duration", "long")),
        ObjectiveChoice("ld", "max", "best glide ratio", ("glide ratio", "far", "distance", "l/d")),
    )
    failure_modes = (
        FailureMode("margin_cl", "lift coefficient below the design limit",
                    "C_L limit / operating C_L - 1, at the design speed"),
        FailureMode("margin_re", "Reynolds number above the critical value",
                    f"Re(mean chord) / {RE_MIN:g} - 1 (below it laminar separation can halve the lift; Lissaman "
                    "1983)"),
    )
    fidelities = (
        Fidelity(CLOSED_FORM, "elliptic lifting line", "C_Di = C_L^2 / (pi AR), lift slope a0 AR / (AR + 2)"),
        Fidelity(REDUCED, "vortex lattice, coarse", "pinneapple_design.aero.vlm, 3 x 10 panels per half span"),
        Fidelity(FULL, "vortex lattice, fine", "8 x 32 panels per half span; lattice-convergence check"),
    )

    def check_scope(self, spec: RequirementSpec) -> list[str]:
        low = spec.text.lower()
        for w in ("airliner", "passenger", "jet", "transonic", "supersonic", "drone", "propeller", "helicopter",
                  "full-size", "full size", "manned"):
            if w in low:
                return [f"'{w}': this template designs the wing of an unpowered model glider; {w} needs propulsion, "
                        "compressibility or certification models"]
        return []

    def variables(self, spec: RequirementSpec) -> list[DesignVariable]:
        return [DesignVariable("span", 0.5, max(0.51, spec["max_span"]), "m", "span"),
                DesignVariable("aspect_ratio", 5.0, 20.0, "", "aspect ratio"),
                DesignVariable("taper", 0.3, 1.0, "", "taper ratio (tip / root chord)"),
                DesignVariable("section", 0, len(SECTIONS) - 1, "", "section", choices=SECTIONS)]

    def _rows(self, X, spec):
        vs = self.variables(spec)
        for row in np.atleast_2d(X):
            yield {v.name: v.decode(x) for v, x in zip(vs, row, strict=True)}

    @staticmethod
    def _vlm(d, nc, ns):
        from pinneapple_design.aero.airframe import Airframe
        from pinneapple_design.aero.geometry import REFERENCE
        from pinneapple_design.aero.vlm import solve

        S = d["span"] ** 2 / d["aspect_ratio"]
        af = Airframe(wing_area=S, aspect_ratio=d["aspect_ratio"], taper=d["taper"],
                      section=REFERENCE[d["section"]].copy(), twist=0.0, incidence=0.0, dihedral=0.0, sweep_le=0.0)
        sol = solve(af, 0.25 * af.mac, nc=nc, ns_wing=ns, tail=False)
        cl0 = sol.coefficients(0.0, 0.0)["CL"]
        a1 = sol.coefficients(math.radians(4.0), 0.0)
        cla = (a1["CL"] - cl0) / math.radians(4.0)
        return sol, cl0, cla

    def _polar(self, d, spec, level, *, mass_scale=1.0, rho_scale=1.0, cd0_scale=1.0, nc=3, ns=10):
        S = d["span"] ** 2 / d["aspect_ratio"]
        AR = d["aspect_ratio"]
        W = spec["mass"] * mass_scale * G0
        rho = RHO_ISA * rho_scale
        tc = SECTION_TC[d["section"]]
        mac = 2 / 3 * (2 * S / (d["span"] * (1 + d["taper"]))) * (1 + d["taper"] + d["taper"] ** 2) / (1 + d["taper"])
        CL = np.linspace(0.15, 1.4, 126)
        V = np.sqrt(2 * W / (rho * S * CL))
        re = V * mac / NU_AIR
        cd0 = profile_cd0(re, tc) * cd0_scale
        if level == CLOSED_FORM:
            e = np.ones_like(CL)
        else:
            sol, cl0, cla = self._vlm(d, nc, ns)
            alpha = (CL - cl0) / cla
            cdi = np.array([sol.coefficients(float(a), 0.0)["CDi"] for a in alpha])
            e = CL ** 2 / (math.pi * AR * np.maximum(cdi, 1e-12))
        CD = cd0 + CL ** 2 / (math.pi * AR * e)
        sink = V * CD / CL
        return {"CL": CL, "V": V, "CD": CD, "sink": sink, "e": e, "re": re, "S": S, "mac": mac}

    def evaluate(self, level: str, X: np.ndarray, spec: RequirementSpec, *, v_design=None, mass_scale=None,
                 rho_scale=None, cd0_scale=None, nc=3, ns=10, **_: Any) -> dict[str, np.ndarray]:
        if level == FULL:
            raise ValueError("the full model runs through verify()")
        rows = list(self._rows(X, spec))
        n = len(rows)
        keys = ("sink_m_s", "speed_m_s", "cl", "span_eff", "ld", "area_m2", "re_mac", "margin_cl", "margin_re")
        out = {k: np.full(n, np.nan) for k in keys}
        lim = spec["cl_limit"]
        ms = np.ones(n) if mass_scale is None else np.broadcast_to(np.asarray(mass_scale, float), (n,))
        rs = np.ones(n) if rho_scale is None else np.broadcast_to(np.asarray(rho_scale, float), (n,))
        cs = np.ones(n) if cd0_scale is None else np.broadcast_to(np.asarray(cd0_scale, float), (n,))
        cache: dict[tuple, Any] = {}
        for i, d in enumerate(rows):
            key = (round(d["span"], 9), round(d["aspect_ratio"], 9), round(d["taper"], 9), d["section"])
            if v_design is None:
                pol = self._polar(d, spec, level, mass_scale=ms[i], rho_scale=rs[i], cd0_scale=cs[i], nc=nc, ns=ns)
                ok = pol["CL"] <= lim
                j = int(np.flatnonzero(ok)[np.argmin(pol["sink"][ok])])
                out["cl"][i], out["speed_m_s"][i] = pol["CL"][j], pol["V"][j]
                out["sink_m_s"][i] = pol["sink"][j]
                out["span_eff"][i], out["re_mac"][i] = pol["e"][j], pol["re"][j]
                out["ld"][i] = pol["CL"][j] / pol["CD"][j]
                out["area_m2"][i] = pol["S"]
            else:
                # fly the design at its design speed with sampled mass, density and profile drag
                if key not in cache:
                    cache[key] = self._polar(d, spec, level, nc=nc, ns=ns)
                pol = cache[key]
                S, W = pol["S"], spec["mass"] * ms[i] * G0
                rho = RHO_ISA * rs[i]
                V = float(np.broadcast_to(v_design, (n,))[i])
                cl = 2 * W / (rho * V * V * S)
                e = float(np.interp(cl, pol["CL"], pol["e"]))
                re = V * pol["mac"] / NU_AIR
                cd = float(profile_cd0(re, SECTION_TC[d["section"]])) * cs[i] + cl * cl / (math.pi *
                                                                                          d["aspect_ratio"] * e)
                out["cl"][i], out["speed_m_s"][i], out["sink_m_s"][i] = cl, V, V * cd / cl
                out["span_eff"][i], out["re_mac"][i], out["ld"][i], out["area_m2"][i] = e, re, cl / cd, S
        out["margin_cl"] = lim / out["cl"] - 1.0
        out["margin_re"] = out["re_mac"] / RE_MIN - 1.0
        return out

    def uncertainties(self, spec: RequirementSpec, design: dict[str, Any]) -> list[Uncertain]:
        return [Uncertain("mass_scale", 1.0, 0.05, "uniform", "", "flying mass within +-5 % (weigh it)"),
                Uncertain("rho_scale", 1.0, 0.06, "uniform", "", "air density +-6 % (weather, a few hundred metres "
                                                                 "of altitude)"),
                Uncertain("cd0_scale", 1.0, 0.25, "uniform", "", "profile drag +-25 % (transition location, "
                                                                 "surface finish)")]

    def robust_inputs(self, X, names, samples, spec):
        """Monte-Carlo samples fly at the design's own minimum-sink speed."""
        opts = {n: samples[:, j] for j, n in enumerate(names)}
        y = self.evaluate(REDUCED, X[:1], spec, **self.design_options(spec))
        opts["v_design"] = float(y["speed_m_s"][0])
        return opts

    def design_options(self, spec):
        """Choose the design speed at the heavy, thin-air, draggy corner of the uncertainties."""
        return {"mass_scale": 1.05, "rho_scale": 0.94, "cd0_scale": 1.25}

    def verify(self, design: dict[str, Any], spec: RequirementSpec, workdir: str | None = None) -> FullResult:
        d = dict(design)
        X = np.array([[d["span"], d["aspect_ratio"], d["taper"], SECTIONS.index(d["section"])]])
        o = self.design_options(spec)
        coarse = self.evaluate(REDUCED, X, spec, nc=3, ns=10, **o)
        mid = self.evaluate(REDUCED, X, spec, nc=5, ns=20, **o)
        fine = self.evaluate(REDUCED, X, spec, nc=8, ns=32, **o)
        e_c, e_m, e_f = (float(y["span_eff"][0]) for y in (coarse, mid, fine))
        _, cl0, cla = self._vlm(d, 8, 32)
        AR = d["aspect_ratio"]
        checks = [
            {"name": "lattice convergence of the span efficiency", "passed": abs(e_f / e_m - 1) < 0.01,
             "critical": True, "value": abs(e_f / e_m - 1), "limit": 0.01,
             "detail": f"e = {e_c:.4f} (3x10), {e_m:.4f} (5x20), {e_f:.4f} (8x32 panels per half span)"},
            {"name": "span efficiency <= 1 (Munk: the elliptic loading is optimal)", "passed": e_f <= 1.0 + 1e-3,
             "critical": True, "value": e_f, "limit": 1.0, "detail": "planar wing"},
            {"name": "lift slope below the 2-D thin-airfoil value", "passed": cla < 2 * math.pi, "critical": True,
             "value": cla, "limit": 2 * math.pi,
             "detail": f"C_L_alpha = {cla:.3f} /rad; Helmbold's estimate 2 pi AR / (2 + sqrt(AR^2 + 4)) = "
                       f"{2 * math.pi * AR / (2 + math.sqrt(AR * AR + 4)):.3f} /rad"},
        ]
        out = {"sink_m_s": float(fine["sink_m_s"][0]), "span_eff": e_f, "speed_m_s": float(fine["speed_m_s"][0]),
               "ld": float(fine["ld"][0]), "margin_cl": float(fine["margin_cl"][0]),
               "margin_re": float(fine["margin_re"][0]), "cl_alpha": cla}
        return FullResult(out, checks, FULL, self.fidelity(FULL).label, 0.0, {},
                          ["no viscous solver ran: profile drag is the flat-plate estimate at every fidelity; a "
                           "RANS check (OpenFOAM / the airfoil zoo surrogate) is the next level and is not part of "
                           "this verification"], compare={"sink_m_s": "sink_m_s", "span_eff": "span_eff"})

    def geometry(self, spec, design):
        from pinneapple_design.geometry_library.core import Body
        from pinneapple_design.geometry_library.kit import naca4, wing

        d = dict(design)
        b, AR, lam = d["span"], d["aspect_ratio"], d["taper"]
        S = b * b / AR
        cr = 2 * S / (b * (1 + lam))
        code = d["section"].split()[-1]
        sec = naca4(int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100, n=40)
        st = [{"le": (0.0, y, 0.0), "chord": cr * (1 - (1 - lam) * abs(y) / (b / 2)), "twist": 0.0, "section": sec}
              for y in np.linspace(-b / 2, b / 2, 21)]
        try:
            V, F = wing(st)
        except Exception:  # noqa: BLE001 - drawing only
            return []
        return [Body("wing", V, F, "white_paint")]

    def bill_of_materials(self, spec, design):
        from .. import materials as M
        d = dict(design)
        S = d["span"] ** 2 / d["aspect_ratio"]
        balsa = M.get("balsa")
        return [{"item": "wing (built-up or sheeted balsa)", "material": balsa.label, "quantity": 1,
                 "size": f"span {d['span']:.2f} m, area {S:.3f} m², taper {d['taper']:.2f}, {d['section']}",
                 "mass_g": None, "properties": {"density": {"value": balsa["density"], "unit": "kg/m^3",
                                                            "source": balsa.cite("density")}}}]

    def figures(self, spec, design, ctx, out_dir):
        from ..figures import wing_figures
        return wing_figures(self, spec, dict(design), ctx, out_dir)

    def operating_window(self, spec, design, ctx):
        cand = ctx["candidate"]
        o = cand.outputs
        nom = self.evaluate(REDUCED, cand.x[None, :], spec, v_design=o["speed_m_s"])
        W = spec["mass"] * G0
        S = o["area_m2"]
        v_lim = math.sqrt(2 * W / (RHO_ISA * S * spec["cl_limit"]))
        return [{"quantity": "glide speed (minimum sink)", "value": round(o["speed_m_s"], 2), "min": round(v_lim, 2),
                 "max": None, "unit": "m/s", "fidelity": "vortex lattice, coarse",
                 "note": f"min: C_L reaches the design limit {spec['cl_limit']:g}"},
                {"quantity": "flying mass", "value": spec["mass"], "min": None,
                 "max": round(spec["mass"] * (1 + float(nom["margin_cl"][0])), 3), "unit": "kg",
                 "fidelity": "vortex lattice, coarse",
                 "note": "at the design speed and sea-level density, C_L reaches the design limit"}]

    def build_instructions(self, spec, design, ctx):
        d = dict(design)
        S = d["span"] ** 2 / d["aspect_ratio"]
        cr = 2 * S / (d["span"] * (1 + d["taper"]))
        return [f"Wing: span {d['span'] * 100:.0f} cm, root chord {cr * 100:.1f} cm, tip chord "
                f"{cr * d['taper'] * 100:.1f} cm (straight taper), section {d['section']}, no twist.",
                "Size the spar for the launch loads separately: no structural check is part of this design.",
                f"Balance and trim the glider to fly at about {ctx['candidate'].outputs['speed_m_s']:.1f} m/s."]

    def limits(self, spec):
        return ["Wing only: fuselage, tail and interference drag are not included.",
                "Inviscid lift (vortex lattice): no stall prediction; the design lift coefficient is capped "
                "instead.", "Profile drag from a flat-plate correlation, not from a viscous solver.",
                "No structural sizing (spar, launch loads)."]
