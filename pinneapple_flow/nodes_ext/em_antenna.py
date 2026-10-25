"""Antenna measurement ranges (``examples/use_cases/antennas/compact_range_qz``): the quiet zone of a compact range
by physical optics (:mod:`pinneapple_simulation.electromagnetics.reflector`), its validation against the closed form
and geometrical optics, and the feed gain x reflector size x rolled edge sweep with its surrogate."""
from __future__ import annotations

from dataclasses import replace

from ..registry import Param, Port, node


@node("antenna.compact_range", "Compact-range quiet zone by physical optics", category="Electromagnetics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("validate", "bool", True, help="PO disk against the closed form, PO against GO"),
              Param("study", "bool", True, help="the feed gain x reflector size x rolled edge sweep"),
              Param("coarse_mesh", "bool", False, help="lambda/6 mesh and lambda/4 quiet-zone grid (the script's --quick)"),
              Param("gains_dbi", "list", [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
              Param("sides_m", "list", [0.9, 1.05, 1.2, 1.35, 1.5, 1.65]),
              Param("rolls", "list", ["none", "reference"], help="rolled edge per design: none (0) or the reference's"),
              Param("mesh_densities", "list", [8, 12, 16], help="validation: PO mesh points per wavelength"),
              Param("rolls_wl", "list", [3.0, 6.0], help="validation: rolled-edge lengths in wavelengths")],
      calls="pinneapple_simulation.electromagnetics.reflector.CompactRange / evaluate + "
            "reflector.study.validate / qz_figure / qz_study")
def compact_range(ctx, inputs, params):
    """The generic example (3 GHz, F = 1.5 m, 1.2 m section, 0.6 m quiet zone at 3 m): the reference design's quiet
    zone (ripple, taper, usable diameter, cross-polar), the validation and the design sweep."""
    from pinneapple_simulation.electromagnetics.reflector import CompactRange, evaluate
    from pinneapple_simulation.electromagnetics.reflector.study import qz_figure, qz_study, validate
    base = CompactRange()
    if params["coarse_mesh"]:
        base = replace(base, mesh_per_wavelength=6.0, qz_step_per_wavelength=4.0)
    ref = evaluate(base)
    for k in ("amp_pp_dB", "phase_pp_deg", "taper_dB", "ripple_pp_dB", "phase_ripple_pp_deg", "usable_diameter",
              "xpol_dB"):
        ctx.metric(f"reference.{k}", float(ref[k]), "PO, reference design")
    png = ctx.path("qz_reference.png")
    qz_figure(base, str(png))
    if params["validate"]:
        val = validate(base, mesh_densities=tuple(int(m) for m in params["mesh_densities"]),
                       rolls_wl=tuple(float(r) for r in params["rolls_wl"]))
        ctx.metric("validation.disk_max_rel_error", float(val["disk"][-1]["max_rel_error"]),
                   f"PO disk on axis vs closed form, lambda/{val['disk'][-1]['per_wavelength']:g} mesh")
        ctx.metric("validation.po_vs_go_rms_dB", float(val["go"][-1]["rms_dB"]), "smooth PO QZ amplitude vs GO taper")
    if params["study"]:
        rolls = [0.0 if r == "none" else base.reflector.roll for r in params["rolls"]]
        st = qz_study(base, [float(g) for g in params["gains_dbi"]], [float(s) for s in params["sides_m"]], rolls,
                      str(ctx.path("study")), mark=(base.feed_gain_dbi, 2 * base.reflector.half_x, "reference design"))
        ctx.metric("study.n_designs", int(st["summary"]["n_designs"]))
        ctx.metric("study.seconds_median", float(st["summary"]["seconds_median"]), "PO mesh + QZ field per design")
    from ..runtime import artifact
    ctx.report["figures"].append({"path": str(png), "name": "qz_reference", "caption": "quiet zone, reference design"})
    return {"metrics": ctx.report["metrics"], "figure": artifact(str(png), "png")}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="antenna.compact_range", title="Quiet zone of a compact antenna range by physical optics",
        domain="electromagnetics", example="use_cases",
        keywords={"compact range": 4, "quiet zone": 4, "physical optics": 3, "reflector": 1.5, "feed gain": 2},
        min_score=3, nodes=("antenna.compact_range", "examples.compare"),
        uncited="the example cites its references in docs/guides/compact_range.md",
        template="use_cases/compact_range_qz", quick=True,
        description="examples/use_cases/antennas/compact_range_qz.py"))


_register_capabilities()
