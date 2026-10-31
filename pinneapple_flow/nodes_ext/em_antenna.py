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


@node("antenna.compact_range_design", "Compact range: QZ vs feed gain and reflector size, smallest reflector",
      category="Electromagnetics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("study", "bool", True, help="off: skip the study (a flow's quick variant keeps only its checks)"),
              Param("quick", "bool", True, help="3 x 3 designs and a coarse mesh (minutes); off: the full generic study"),
              Param("gains_dbi", "list", [], help="feed gains (empty: the generic study's)"),
              Param("sides_m", "list", [], help="projected reflector sides (empty: the generic study's)"),
              Param("focals_m", "list", [], help="focal lengths (empty: the generic study's)"),
              Param("qz_diameter_m", "float", 1.0, minimum=0.05, help="required usable quiet-zone diameter"),
              Param("amp_tol_db", "float", 0.5, minimum=0.01, help="amplitude within +- this"),
              Param("phase_tol_deg", "float", 5.0, minimum=0.1, help="phase within +- this"),
              Param("xpol_max_db", "float", -25.0, help="cross-polar limit over the quiet zone"),
              Param("checks", "bool", True, help="facet convergence, QZ grid and power balance on the middle design"),
              Param("workers", "int", 2, minimum=1, maximum=32, help="parallel PO processes")],
      calls="pinneapple_simulation.electromagnetics.reflector.design.generic_2ghz / from_config / run_design_study")
def compact_range_design(ctx, inputs, params):
    """The generic 2 GHz range (F = 4 m, 3-lambda rolled edge): PO of every (gain x side x F) design for both
    polarisations, the checks, a GP with held-out error and 95 % coverage, the design maps, the Pareto front of QZ
    size against reflector size and the smallest reflector meeting the spec, verified by PO."""
    from pinneapple_simulation.electromagnetics.reflector.design import (
        from_config,
        generic_2ghz,
        run_design_study,
    )
    if not params["study"]:
        ctx.warn("study skipped (study = false)")
        return {"metrics": ctx.report["metrics"]}
    cfg = generic_2ghz(quick=bool(params["quick"]))
    sw = dict(cfg["sweep"])
    for key, p in (("gains_dBi", "gains_dbi"), ("sides_m", "sides_m"), ("focals_m", "focals_m")):
        if params[p]:
            sw[key] = [float(v) for v in params[p]]
    cfg.update({"sweep": sw, "checks": bool(params["checks"]), "workers": int(params["workers"]),
                "spec": {"qz_diameter": float(params["qz_diameter_m"]), "amp_tol_db": float(params["amp_tol_db"]),
                         "phase_tol_deg": float(params["phase_tol_deg"]), "xpol_max_db": float(params["xpol_max_db"])}})
    if params["gains_dbi"] or params["sides_m"]:
        cfg["optimise"] = {**cfg["optimise"], "gain_bounds": [min(sw["gains_dBi"]), max(sw["gains_dBi"])],
                           "side_bounds": [min(sw["sides_m"]), max(sw["sides_m"])]}
    c = from_config(cfg)
    out = ctx.path("design")
    st = run_design_study(c["base"], c["spec"], c["grid"]["gains"], c["grid"]["sides"], str(out),
                          freqs_hz=c["grid"]["freqs_hz"], focals=c["grid"]["focals"], checks=c["checks"],
                          densities=c["densities"], workers=c["workers"], log=lambda m: None,
                          **c["optimise"])
    s = st["summary"]
    ctx.metric("n_designs", int(s["n_designs"]), "PO solves, V and H each")
    ctx.metric("seconds_median", float(s["seconds_median"]), "PO mesh + QZ field per design")
    for m, d in s["surrogate"].items():
        if isinstance(d, dict):
            ctx.metric(f"gp.{m}.rmse", float(d["rmse"]), "5-fold held-out RMSE")
            ctx.metric(f"gp.{m}.coverage95", float(d["coverage95"]), "held-out fraction inside the 95% interval")
    if "power_balance" in s:
        ctx.metric("power_ratio", float(s["power_balance"]["ratio"]), "through the QZ plane / intercepted by PO")
    o = s["optimiser"]
    ctx.metric("optimiser.found", bool(o["found"]), "smallest reflector meeting the spec, verified by PO")
    if o["found"]:
        for k in ("side", "feed_gain_dBi", "usable_diameter", "ripple_pp_dB", "xpol_dB"):
            ctx.metric(f"optimiser.{k}", float(o["design"][k]), "PO verification of the optimiser's design")
    else:
        ctx.warn("no reflector in the range meets the spec by PO; see design_summary.json")
    ctx.warn(s["limits"])
    from ..runtime import artifact
    png = out / "pareto.png"
    for name, cap in (("pareto", "QZ size vs reflector size, Pareto front"), ("design_maps", "design maps"),
                      ("qz_cuts", "QZ cuts of the chosen design")):
        ctx.report["figures"].append({"path": str(out / f"{name}.png"), "name": name, "caption": cap})
    return {"metrics": ctx.report["metrics"], "figure": artifact(str(png), "png")}


@node("antenna.compact_range_check", "Compact range: one design against a quiet-zone spec (PO, V and H)",
      category="Electromagnetics", outputs=[Port("metrics", "metrics")],
      params=[Param("feed_gain_dbi", "float", 16.0, minimum=3.0, maximum=30.0),
              Param("side_m", "float", 4.0, minimum=0.2, help="projected reflector side"),
              Param("focal_m", "float", 4.0, minimum=0.2),
              Param("coarse_mesh", "bool", True, help="lambda/6 mesh, lambda/3 quiet-zone grid"),
              Param("qz_diameter_m", "float", 1.0, minimum=0.05),
              Param("amp_tol_db", "float", 0.5, minimum=0.01), Param("phase_tol_deg", "float", 5.0, minimum=0.1),
              Param("xpol_max_db", "float", -25.0)],
      calls="pinneapple_simulation.electromagnetics.reflector.design.qz_report / design_grid")
def compact_range_check(ctx, inputs, params):
    """One design of the generic 2 GHz range by PO for both feed polarisations: disk and principal-cut metrics and
    whether it meets the spec."""
    from pinneapple_simulation.electromagnetics.reflector.design import (
        Spec,
        design_grid,
        from_config,
        generic_2ghz,
        qz_report,
    )
    c = from_config(generic_2ghz(quick=bool(params["coarse_mesh"])))
    spec = Spec(float(params["qz_diameter_m"]), float(params["amp_tol_db"]), float(params["phase_tol_deg"]),
                float(params["xpol_max_db"]))
    (_, cr), = design_grid(c["base"], [float(params["feed_gain_dbi"])], [float(params["side_m"])],
                           focals=[float(params["focal_m"])])
    rep = qz_report(cr, spec)
    for k, v in rep.items():
        if isinstance(v, bool | int | float) and k != "seconds":
            ctx.metric(k, v)
    return {"metrics": ctx.report["metrics"]}


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
    register_capability(Capability(
        id="antenna.compact_range_design", title="Compact range design: QZ vs feed gain and reflector size",
        domain="electromagnetics", example="use_cases",
        keywords={"compact range": 3, "quiet zone": 3, "reflector size": 4, "feed gain": 3, "pareto": 2,
                  "smallest reflector": 4},
        min_score=4, nodes=("antenna.compact_range_design",),
        uncited="generic proportions; the guide states the PO limits (docs/guides/compact_range.md)",
        template="use_cases/compact_range_2ghz", quick=True,
        description="examples/use_cases/antennas/compact_range_2ghz.py"))


_register_capabilities()
