"""Passive (or fan-cooled) plate-fin heat sink for a heat source: base, fins, material.

Reuses the heat-sink sizer physics: :mod:`pinneapple_physics.closed_form.plate_fin_heatsink` (Elenbaas / Bar-Cohen
natural convection, Teertstra forced convection, fin efficiency, spreading resistance) and
:func:`pinneapple_design.thermal.heatsink.evaluate_design` (3-D finite-volume conduction in the base with the fins'
convection, energy balance and grid checks, and a +-15 % band on the heat-transfer coefficient).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import materials as M
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

HS_MATERIALS = ("al6063", "al6061", "cu110")
SIZER = "pinneapple_design.thermal.heatsink sizer default"


def _proc():
    from pinneapple_design.thermal.heatsink.sizer import PROCESSES
    return PROCESSES["extruded"]


class HeatSinkTemplate(ProblemTemplate):
    name = "heatsink"
    title = "Plate-fin heat sink"
    domain = "heat transfer"
    keywords = {"heat sink": 2.5, "heatsink": 2.5, "cooler": 1.0, "cooling": 0.7, "chip": 0.7, "cpu": 0.7,
                "led": 0.5, "transistor": 0.5, "dissipate": 1.0, "thermal": 0.5, "fins": 0.7, "still air": 0.5}
    physics = ("steady conduction in the base and the fins (fin efficiency, spreading resistance)",
               "convection from the fins: Elenbaas / Bar-Cohen & Rohsenow correlation in still air, Teertstra "
               "model with a fan",
               "interface (thermal paste) resistance between the source and the base")
    assumptions = ("vertical fins with air rising along the base depth (natural convection)",
                   "radiation neglected (conservative: real parts run cooler, more so if anodised)",
                   "a fan's air is ducted between the fins (forced convection)",
                   "air at 1 atm; properties from Incropera & DeWitt Table A.4")
    output_units = {"t_source_c": "°C", "mass_g": "g", "r_total_k_w": "K/W", "fin_gap_mm": "mm", "h_w_m2k":
                    "W/(m²·K)", "fin_efficiency": "-"}
    slots = (
        Slot("power", "heat to dissipate", "W", "W", ("chip", "cpu", "dissipat", "power", "source", "led",
                                                     "transistor", "heat"), bounds=(0.5, 400.0), critical=True,
             role="load", question="How much heat must it dissipate?"),
        Slot("t_limit", "highest allowed source (case) temperature", "K", "°C",
             ("limit", "max", "maximum", "below", "under", "keep", "at most", "junction", "case"),
             bounds=(273.15 + 30, 273.15 + 200), critical=True, role="limit",
             question="What is the highest allowed temperature of the chip's case (or source)?"),
        Slot("t_ambient", "ambient air temperature", "K", "°C", ("ambient", "room", "air at", "environment"),
             default=298.15, default_source="template default: room temperature 25 °C (state yours)",
             bounds=(273.15 - 20, 273.15 + 70)),
        Slot("air_velocity", "air speed through the fins", "m/s", "m/s", ("fan", "airflow", "air speed",
                                                                        "velocity", "blowing"),
             cues=(("still air", 0.0), ("natural convection", 0.0), ("passive", 0.0), ("no fan", 0.0),
                   ("fanless", 0.0)), bounds=(0.0, 6.0), critical=True,
             question="Is there a fan (air speed through the fins, m/s), or is it still air (0)?"),
        Slot("source_size", "heat-source footprint (square side)", "m", "mm", ("footprint", "die", "package",
                                                                                "lid", "mm square"),
             default=0.030, default_source=f"{SIZER} (30 x 30 mm): affects the spreading resistance; state yours",
             bounds=(0.003, 0.15)),
        Slot("tim", "interface resistance", "K*m^2/W", "K·mm²/W", ("thermal paste", "tim", "interface"),
             default=10e-6, default_source=f"{SIZER} (thermal grease, 10 K·mm²/W)", bounds=(1e-7, 1e-3)),
        Slot("max_width", "largest base width", "m", "mm", ("wide", "width"), default=0.100,
             default_source=f"{SIZER} envelope (100 mm)", bounds=(0.02, 0.2)),
        Slot("max_depth", "largest base depth", "m", "mm", ("deep", "depth", "long"), default=0.100,
             default_source=f"{SIZER} envelope (100 mm)", bounds=(0.02, 0.2)),
        Slot("max_height", "largest height (base + fins)", "m", "mm", ("tall", "height", "high"), default=0.050,
             default_source=f"{SIZER} envelope (50 mm)", bounds=(0.01, 0.12)),
        Slot("material", "material", "choice", keywords=("material",), choices=HS_MATERIALS, role="material"),
    )
    objectives = (
        ObjectiveChoice("mass_g", "min", "mass", ("light", "lightest", "small", "compact")),
        ObjectiveChoice("t_source_c", "min", "source temperature", ("coolest", "cold", "lowest temperature")),
    )
    failure_modes = (
        FailureMode("margin_temp", "source stays below its limit",
                    "(T_limit - T_ambient) / (T_source - T_ambient) - 1"),
        FailureMode("margin_gap", "fins can be extruded (gap)", "fin gap / 1.5 mm - 1 (extrusion process limit)"),
        FailureMode("margin_aspect", "fins can be extruded (aspect)", "20 / (fin height / gap) - 1"),
    )
    fidelities = (
        Fidelity(CLOSED_FORM, "hand calculation", "convection coefficient evaluated once with the fin root at the "
                                                  "temperature limit, fin efficiency, spreading"),
        Fidelity(REDUCED, "resistance network", "the same correlations iterated to the fin-root temperature "
                                                "(plate_fin_heatsink.evaluate)"),
        Fidelity(FULL, "3-D finite volume (base)", "finite-volume conduction in the base with the fins' "
                                                   "convection; energy balance, grid convergence, +-15 % h band "
                                                   "(thermal.heatsink.evaluate_design)"),
    )

    def check_scope(self, spec: RequirementSpec) -> list[str]:
        low = spec.text.lower()
        for w in ("liquid", "water cool", "heat pipe", "vapour chamber", "vapor chamber", "peltier", "immersion"):
            if w in low:
                return [f"'{w}': this template sizes air-cooled plate-fin heat sinks; {w} needs a two-phase / "
                        "liquid-cooling model"]
        if spec.get("t_limit") is not None and spec.get("t_ambient") is not None and \
                spec["t_limit"] <= spec["t_ambient"]:
            return ["the temperature limit is not above the ambient temperature: no passive cooler can do that"]
        return []

    def _materials(self, spec):
        m = spec.get("material")
        return (m,) if m else HS_MATERIALS

    def variables(self, spec: RequirementSpec) -> list[DesignVariable]:
        proc = _proc()
        src = math.ceil(spec["source_size"] * 1e3)
        vs = [DesignVariable("width_mm", max(src, 20), max(src + 1, round(spec["max_width"] * 1e3)), "mm",
                             "base width", integer=True),
              DesignVariable("depth_mm", max(src, 20), max(src + 1, round(spec["max_depth"] * 1e3)), "mm",
                             "base depth (along the fins)", integer=True),
              DesignVariable("base_mm", proc["tb_min"], 12.0, "mm", "base thickness"),
              DesignVariable("n_fins", 2, 40, "", "number of fins", integer=True),
              DesignVariable("fin_mm", proc["t_min"], 3.0, "mm", "fin thickness"),
              DesignVariable("height_mm", 8, max(9, round(spec["max_height"] * 1e3) - 3), "mm", "fin height",
                             integer=True)]
        mats = self._materials(spec)
        if len(mats) > 1:
            vs.append(DesignVariable("material", 0, len(mats) - 1, "", "material", choices=mats))
        return vs

    def _rows(self, X, spec):
        X = np.atleast_2d(X)
        vs = self.variables(spec)
        mats = self._materials(spec)
        for row in X:
            d = {v.name: v.decode(x) for v, x in zip(vs, row, strict=True)}
            d.setdefault("material", mats[0])
            d["base_mm"] = round(d["base_mm"] * 2) / 2
            d["fin_mm"] = round(d["fin_mm"], 1)
            yield d

    def full_design(self, design: dict[str, Any], spec: RequirementSpec) -> dict[str, Any]:
        d = dict(design)
        d.setdefault("material", self._materials(spec)[0])
        d["base_mm"] = round(d["base_mm"] * 2) / 2
        d["fin_mm"] = round(d["fin_mm"], 1)
        return d

    def _geom(self, d):
        from pinneapple_physics.closed_form.plate_fin_heatsink import HeatSinkGeometry
        return HeatSinkGeometry(d["width_mm"] * 1e-3, d["depth_mm"] * 1e-3, d["base_mm"] * 1e-3, int(d["n_fins"]),
                                d["fin_mm"] * 1e-3, d["height_mm"] * 1e-3, d["material"])

    def _op(self, spec, tim=None):
        from pinneapple_physics.closed_form.plate_fin_heatsink import OperatingPoint
        s = spec["source_size"]
        return OperatingPoint(spec["power"], spec["t_ambient"] - 273.15, spec["air_velocity"], s, s,
                              spec["tim"] if tim is None else tim)

    def evaluate(self, level: str, X: np.ndarray, spec: RequirementSpec, *, h_scale=None, tim=None,
                 **_: Any) -> dict[str, np.ndarray]:
        from pinneapple_physics.closed_form.plate_fin_heatsink import evaluate as hs_eval

        if level == FULL:
            raise ValueError("the full model runs through verify()")
        rows = list(self._rows(X, spec))
        n = len(rows)
        hs = np.ones(n) if h_scale is None else np.broadcast_to(np.asarray(h_scale, float), (n,))
        tims = np.full(n, spec["tim"]) if tim is None else np.broadcast_to(np.asarray(tim, float), (n,))
        out = {k: np.full(n, np.nan) for k in ("t_source_c", "r_total_k_w", "h_w_m2k", "fin_efficiency")}
        gap = np.zeros(n)
        mass = np.zeros(n)
        ta = spec["t_ambient"] - 273.15
        for i, d in enumerate(rows):
            g = self._geom(d)
            gap[i] = g.fin_gap
            mass[i] = g.mass_kg() * 1e3
            try:
                if level == CLOSED_FORM:      # h once, with the fin root at the temperature limit (hand calculation)
                    r = hs_eval(g, self._op(spec, tims[i]), max_iter=1, h_scale=float(hs[i]),
                                t_root_guess_c=spec["t_limit"] - 273.15)
                else:
                    r = hs_eval(g, self._op(spec, tims[i]), h_scale=float(hs[i]))
            except ValueError:
                continue
            out["t_source_c"][i] = r["t_source_max_c"]
            out["r_total_k_w"][i] = r["r_total_k_w"]
            out["h_w_m2k"][i] = r["h_w_m2k"]
            out["fin_efficiency"][i] = r["fin_efficiency"]
        proc = _proc()
        H = np.array([d["height_mm"] for d in rows]) * 1e-3
        rise = out["t_source_c"] - ta
        lim = spec["t_limit"] - spec["t_ambient"]
        return {"margin_temp": lim / np.where(rise > 0, rise, np.nan) - 1.0,
                "margin_gap": gap * 1e3 / proc["gap_min"] - 1.0,
                "margin_aspect": proc["aspect_max"] / np.maximum(H / np.maximum(gap, 1e-9), 1e-9) - 1.0,
                "mass_g": mass, "fin_gap_mm": gap * 1e3, **out}

    def design_options(self, spec: RequirementSpec) -> dict[str, Any]:
        from pinneapple_design.thermal.heatsink.engine import H_BAND
        return {"h_scale": 1.0 - H_BAND, "tim": 1.5 * spec["tim"]}

    def uncertainties(self, spec: RequirementSpec, design: dict[str, Any]) -> list[Uncertain]:
        from pinneapple_design.thermal.heatsink.engine import H_BAND
        t = spec["tim"]
        return [Uncertain("h_scale", 1.0, H_BAND, "uniform", "",
                          f"+-{100 * H_BAND:.0f} % on the convection coefficient: the band the heat-sink engine "
                          "states for its correlations (not a measured uncertainty of this design)"),
                Uncertain("tim", t, 0.5 * t, "uniform", "K*m^2/W", "interface resistance +-50 % (paste thickness "
                                                                   "and coverage vary)")]

    def verify(self, design: dict[str, Any], spec: RequirementSpec, workdir: str | None = None) -> FullResult:
        from pinneapple_design.thermal.heatsink import DesignInput, OperatingInput, evaluate_design

        d = self.full_design(design, spec)
        di = DesignInput(d["width_mm"], d["depth_mm"], d["base_mm"], int(d["n_fins"]), d["fin_mm"], d["height_mm"],
                         d["material"])
        s = spec["source_size"] * 1e3
        op = OperatingInput(spec["power"], spec["t_ambient"] - 273.15, spec["air_velocity"], s, s,
                            tim_k_mm2_w=spec["tim"] * 1e6, t_limit_c=spec["t_limit"] - 273.15)
        r = evaluate_design(di, op)
        k = r["kpis"]
        ta, tl = spec["t_ambient"] - 273.15, spec["t_limit"] - 273.15
        checks = [{"name": c["name"], "passed": c["status"] == "pass", "critical": c["name"] == "energy_balance",
                   "value": c.get("value"), "limit": None, "detail": c.get("detail", "")} for c in r["checks"]]
        lo, hi = k["t_source_band_c"]
        checks.append({"name": "limit met with the convection coefficient 15 % low", "passed": hi <= tl,
                       "critical": False, "value": hi, "limit": tl,
                       "detail": f"source temperature band {lo:.1f} to {hi:.1f} °C for h +-15 %"})
        out = {"t_source_c": k["t_source_max_c"], "t_source_band_hi_c": hi, "r_total_k_w": k["r_total_k_w"],
               "margin_temp": (tl - ta) / (k["t_source_max_c"] - ta) - 1.0, "mass_g": k["mass_g"]}
        return FullResult(out, checks, FULL, self.fidelity(FULL).label, 0.0, {"engine": r},
                          [f"{r['method']}" if isinstance(r.get("method"), str) else "evaluate_design (FVM base)"]
                          + list(r.get("warnings", [])), compare={"t_source_c": "t_source_c"},
                          compare_options={"h_scale": 1.0, "tim": spec["tim"]})

    # ------------------------------------------------------------------ package
    def geometry(self, spec: RequirementSpec, design: dict[str, Any]):
        from pinneapple_design.geometry_library.core import Body
        from pinneapple_design.geometry_library.kit import box

        d = self.full_design(design, spec)
        W, D, tb, N, t, H = (d["width_mm"] * 1e-3, d["depth_mm"] * 1e-3, d["base_mm"] * 1e-3, int(d["n_fins"]),
                             d["fin_mm"] * 1e-3, d["height_mm"] * 1e-3)
        colour = "copper" if d["material"].startswith("cu") else "aluminium"
        bodies = [Body("base", *box((W, D, tb), center=(W / 2, D / 2, tb / 2)), colour)]
        pitch = (W - t) / (N - 1)
        for i in range(N):
            x = t / 2 + i * pitch
            bodies.append(Body(f"fin_{i:02d}", *box((t, D, H), center=(x, D / 2, tb + H / 2)), colour))
        return bodies

    def bill_of_materials(self, spec: RequirementSpec, design: dict[str, Any]) -> list[dict[str, Any]]:
        d = self.full_design(design, spec)
        mat = M.heat_sink_material(d["material"])
        g = self._geom(d)
        return [{"item": "plate-fin heat sink", "material": mat.label, "quantity": 1,
                 "size": f"base {d['width_mm']} x {d['depth_mm']} x {d['base_mm']} mm, {d['n_fins']} fins "
                         f"{d['fin_mm']} mm thick x {d['height_mm']} mm tall (gap {g.fin_gap * 1e3:.2f} mm)",
                 "mass_g": round(g.mass_kg() * 1e3, 1),
                 "properties": {k: {"value": mat.props[k].value, "unit": mat.props[k].unit, "source": mat.cite(k)}
                                for k in ("density", "conductivity")}},
                {"item": "thermal interface (paste or pad)", "material": "-", "quantity": 1,
                 "size": f"{spec['source_size'] * 1e3:.0f} mm square, {spec['tim'] * 1e6:.0f} K·mm²/W or better",
                 "mass_g": None, "properties": {}}]

    def operating_window(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]
                         ) -> list[dict[str, Any]]:
        from pinneapple_physics.closed_form.plate_fin_heatsink import evaluate as hs_eval
        d = self.full_design(design, spec)
        g = self._geom(d)
        ta, tl = spec["t_ambient"] - 273.15, spec["t_limit"] - 273.15
        P = spec["power"]
        ps = np.linspace(0.2 * P, 3 * P, 141)
        ok = []
        for p in ps:
            op = self._op(spec)
            op.power_w = float(p)
            ok.append(hs_eval(g, op)["t_source_max_c"] <= tl)
        ok = np.array(ok)
        p_max = float(ps[np.flatnonzero(ok)[-1]]) if ok.any() else float("nan")
        tas = np.linspace(ta - 20, tl - 1, 121)
        ok2 = []
        for x in tas:
            op = self._op(spec)
            op.t_ambient_c = float(x)
            ok2.append(hs_eval(g, op)["t_source_max_c"] <= tl)
        ok2 = np.array(ok2)
        ta_max = float(tas[np.flatnonzero(ok2)[-1]]) if ok2.any() else float("nan")
        return [{"quantity": "power", "value": P, "min": 0.0, "max": round(p_max, 1), "unit": "W",
                 "fidelity": "resistance network", "note": f"source stays below {tl:.0f} °C at {ta:.0f} °C ambient"},
                {"quantity": "ambient temperature", "value": round(ta, 1), "min": None, "max": round(ta_max, 1),
                 "unit": "°C", "fidelity": "resistance network", "note": f"at {P:g} W"},
                {"quantity": "orientation", "value": None, "min": None, "max": None, "unit": "",
                 "fidelity": "-", "note": "fins vertical, air rising along the depth (natural convection)"
                 if spec["air_velocity"] <= 0 else "fan air ducted along the fins"}]

    def build_instructions(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
        d = self.full_design(design, spec)
        mat = M.heat_sink_material(d["material"])
        return [f"Extrude or machine the heat sink in {mat.label}: base {d['width_mm']} x {d['depth_mm']} x "
                f"{d['base_mm']} mm with {d['n_fins']} fins {d['fin_mm']} mm thick and {d['height_mm']} mm tall "
                "(heatsink.step).",
                "Centre the heat source on the base; apply the thermal paste thinly over the whole footprint.",
                "Mount with the fins vertical and nothing blocking the air below and above them."
                if spec["air_velocity"] <= 0 else "Duct the fan's air between the fins.",
                "Check: at full power the case temperature should settle below the limit; the operating window "
                "gives the margin."]

    def limits(self, spec: RequirementSpec) -> list[str]:
        return ["Correlation-based convection: +-15 % on h is the stated band of the correlations, not a measured "
                "uncertainty of this part.",
                "Radiation is neglected (conservative).",
                "The full model resolves conduction in the base; the fins and the air are modelled by the "
                "correlations, not by CFD.",
                "The source temperature is the case / lid temperature; the junction is hotter by the package's own "
                "resistance (not modelled)."]

    def figures(self, spec, design, ctx, out_dir):
        from ..figures import heatsink_figures
        return heatsink_figures(self, spec, self.full_design(design, spec), ctx, out_dir)
