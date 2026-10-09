"""Qualitative preview of heat-sink variants, then the quantitative check.

A plate-fin heat sink is an ``Assembly`` of a base and fins. Before computing anything, ``preview`` says which way
each variant (more fins, taller, thicker, fins across the flow) should move the heat rejected, through which
mechanism (leading edges, boundary-layer growth, the air heating up, fin efficiency, wake) and how sure it is; the
surface map shows where the heat leaves. ``part_sensitivity`` ranks which part to change first. Only then
``quantify`` runs the HeatSink Sizer physics (channel-flow correlations + finite-volume base) on the variants worth
it and reports where the qualitative reading was right. The fins-across-the-flow design is outside that quantitative
tool: the preview still says it is a bad idea, and why.
"""
import numpy as np

from pinneapple_design.geometry.bodies import box
from pinneapple_design.qualitative import Assembly, part_sensitivity, preview
from pinneapple_design.thermal.heatsink.engine import (
    DesignInput,
    OperatingInput,
    physics_resistance,
)

W, D, TB = 0.060, 0.080, 0.004                 # base width (y), depth along the flow (x), thickness (m)


def heat_sink(n_fins=8, height=0.030, t=0.0015, across=False):
    parts = {"base": box((D, W, TB), (D / 2, 0, TB / 2))}
    if across:                                   # fins perpendicular to the flow (x): they block it
        for i, x in enumerate(np.linspace(t / 2, D - t / 2, n_fins)):
            parts[f"fin{i + 1}"] = box((t, W, height), (x, 0, TB + height / 2))
    else:
        for i, y in enumerate(np.linspace(-W / 2 + t / 2, W / 2 - t / 2, n_fins)):
            parts[f"fin{i + 1}"] = box((D, t, height), (D / 2, y, TB + height / 2))
    return Assembly(parts)


VARIANTS = {"8 fins (baseline)": dict(), "12 fins": dict(n_fins=12), "20 fins": dict(n_fins=20),
            "taller fins (45 mm)": dict(height=0.045), "thicker fins (3 mm)": dict(t=0.003),
            "fins across the flow": dict(across=True)}


def conductance(name, geom):
    """Quantitative step: base-to-air conductance (W/K) from the HeatSink Sizer physics."""
    kw = {"n_fins": 8, "height": 0.030, "t": 0.0015, **VARIANTS[name]}
    design = DesignInput(D * 1e3, W * 1e3, TB * 1e3, kw["n_fins"], kw["t"] * 1e3, kw["height"] * 1e3)
    return 1 / physics_resistance(design, OperatingInput(power_w=30, air_velocity_m_s=2.0))["r_base_k_w"]


def main(lang="pt", out="heat_sink_preview.png"):
    geoms = {k: heat_sink(**v) for k, v in VARIANTS.items()}
    p = preview(geoms, "maximizar dissipação de calor" if lang == "pt" else "maximize heat rejection",
                conditions={"U": 2.0, "dT": 40.0, "base_part": "base"}, lang=lang)
    print(p.text())
    p.figure(out, view=(38, -150))
    print()
    print(part_sensitivity(geoms["8 fins (baseline)"], "maximizar dissipação de calor", {"U": 2.0, "base_part": "base"},
                           lang=lang).text())
    check = p.quantify(conductance, variants=[k for k in VARIANTS if k != "fins across the flow"])
    print()
    print(check.text())
    return p, check


if __name__ == "__main__":
    main()
