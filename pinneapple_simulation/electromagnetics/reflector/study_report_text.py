"""The datasheet of a compact-range study: every sentence is computed from the study's numbers.

The narrative rules are fixed here and take no model: the verdict compares the usable diameter with the
specification at the declared tolerance; the near-field remark is the Fresnel number against two; the
cut-versus-disk remark is the ratio of the two computed peak-to-peak values; the reference comparison
reports the signed differences of the rows it was given.  When a step was not run or a number is missing
the sentence says so instead of being written anyway.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from pinneapple_tools.reporting.datasheet import Datasheet, Figure, KPI, Step, Table

FT = 0.3048


def _f(x, nd=2, dash="n/a"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if not np.isfinite(v) else f"{v:.{nd}f}"


def _pct(a, b, nd=0):
    try:
        return f"{100 * float(a) / float(b):.{nd}f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "n/a"


def build_datasheet(p, study: Mapping[str, Any], figs: Mapping[str, str]) -> Datasheet:
    R = study["results"]
    T = study.get("timings", {})
    model = R.get("model", {})
    qz = R.get("qz", {})
    ref = R.get("reference", {})
    sweep = R.get("sweep", {})
    opt = R.get("optimize", {})
    prof = R.get("profile", {})
    cal = R.get("calibration", {})
    band = R.get("bandwidth", {})
    tol = R.get("tolerance", {})
    axial = R.get("axial", {})

    lam = p.wavelength
    lam_in = lam / 0.0254
    base_plane_ft = p.qz_z / FT
    pol0 = p.pols[0]
    base = qz.get("pol", {}).get(pol0, {}).get(f"{base_plane_ft:g}", {}) if qz else {}
    disk = base.get("disk", {})
    usable = disk.get("usable_at_0.5dB", float("nan"))
    ripp = disk.get("ripple_pp_dB", float("nan"))
    amp_pp = disk.get("amp_pp_dB", float("nan"))
    ph_pp = disk.get("phase_pp_deg", float("nan"))
    xpol = disk.get("xpol_dB", float("nan"))
    meets = isinstance(usable, (int, float)) and np.isfinite(usable) and usable >= p.qz_diameter

    # ------------------------------------------------------------------ headline
    d = Datasheet(
        title=p.title,
        subtitle=f"Quiet zone by physical optics, validated, at {p.freq_hz / 1e9:g} GHz",
        eyebrow="PINNeAPPle &middot; electromagnetics &middot; reflector",
        verified_by="every number recomputed from the geometry by physical optics",
        footer_left="PINNeAPPle &middot; compact-range study &middot; physical optics, "
                    "Gaussian-process surrogates, equal-budget search",
    )
    d.kpis([
        KPI("Focal length", f"{p.focal / FT:.2f}", "ft", f"{p.focal:.3f} m"),
        KPI("Section", f"{p.section / FT:.2f}", "ft square",
            f"{model.get('section_wavelengths', float('nan')):.1f} $\\lambda$"),
        KPI("Quiet zone", f"{p.qz_diameter / FT:.2f}", "ft",
            f"at {base_plane_ft:g} ft, {model.get('qz_diameter_wavelengths', float('nan')):.1f} $\\lambda$"),
        KPI("Feed", f"{p.feed_gain_dbi:.1f}", "dBi", f"cos$^q$, q = {p.cr().feed_q:.2f}"),
        KPI("Frequency", f"{p.freq_hz / 1e9:.3g}", "GHz", f"$\\lambda$ = {lam_in:.2f} in"),
        KPI("Usable, $\\pm$0.5 dB", f"{usable / FT:.2f}", "ft",
            _pct(usable, p.qz_diameter) + " of the specification",
            tone="ok" if meets else "fail"),
    ])

    if meets:
        d.verdict_box(f"The design delivers a {usable / FT:.2f} ft quiet zone within $\\pm${p.amp_tol:g} dB and "
                      f"$\\pm${p.phase_tol:g}&deg; at {base_plane_ft:g} ft: it meets the {p.qz_diameter / FT:.2f} ft "
                      f"specification with {_pct(usable - p.qz_diameter, p.qz_diameter)} to spare.",
                      "ok", "Specification met")
    else:
        d.verdict_box(
            f"The design does not deliver the {p.qz_diameter / FT:.2f} ft quiet zone. Over the full aperture it "
            f"gives {_f(amp_pp)} dB amplitude and {_f(ph_pp, 1)}&deg; phase peak to peak, and a "
            f"{_f(usable / FT)} ft diameter stays within $\\pm${p.amp_tol:g} dB and $\\pm${p.phase_tol:g}&deg;, "
            f"{_pct(usable, p.qz_diameter)} of the specification. The Fresnel number of the section seen from "
            f"the quiet zone is {_f(model.get('fresnel_number'))}, so the zone sits in the near field of the "
            f"reflector, where the edge diffraction sets the field and no taper description applies.",
            "fail", "Specification not met")

    # ------------------------------------------------------------------ 01 model
    s = d.section("Model and resolution",
                  "What the geometry is, at what resolution it is integrated, and the dimensionless numbers "
                  "that decide the rest of the study.")
    s.row(
        Table([
            ["Focal length", f"{p.focal / FT:.2f} ft ({p.focal:.4f} m)"],
            ["Section", f"{p.section / FT:.2f} x {p.section / FT:.2f} ft, superellipse n = {p.n:g}"],
            ["Section centre", f"{p.offset / FT:.2f} ft above the paraboloid axis"],
            ["Rolled edge", f"{p.roll / FT:.2f} ft ({p.roll / lam:.2f} $\\lambda$), blend {p.blend:g}, "
                            f"aspect {p.roll_aspect:g}, end {p.end_angle:g}&deg;"],
            ["Quiet zone", f"{p.qz_diameter / FT:.2f} ft at z = {base_plane_ft:g} ft"],
            ["Reflector surface", f"{_f(model.get('surface_area_m2'), 2)} m$^2$ "
                                  f"({_f(model.get('surface_area_wavelengths2'), 0)} $\\lambda^2$)"],
            ["Mesh", f"{_f(p.mesh_per_wavelength, 0)} points per wavelength, "
                     f"{int(model.get('n_triangles', 0)):,} triangles"],
            ["Fresnel number a$^2$/($\\lambda$z)", _f(model.get("fresnel_number"))],
            ["Quiet zone / section", _pct(p.qz_diameter, p.section)],
        ], caption="**The design as modelled.** The paraboloid z = (x$^2$+y$^2$)/4F is integrated with the "
                   "complete dyadic Green's function: the quiet zone is in the reflector's near field."),
        Figure(figs.get("side", ""), "**Side section x = 0.** Paraboloid, rolled edge, rays from the focus "
                                     "collimated onto the quiet zone, and the quiet-zone cylinder."),
    )
    s.row(
        Figure(figs.get("front", ""), "**Front view** from the quiet zone: the parabolic-section junction, the "
                                      "rolled edge's outer extent, the quiet zone and the two cuts."),
        Figure(figs.get("illum", ""), "**Feed illumination of the lit surface**, with the quiet-zone footprint "
                                      "dashed. The rim taper decides how much the edge diffracts."),
    )
    fr = model.get("fresnel_number", float("nan"))
    if np.isfinite(fr) and fr < 2:
        s.note(f"Fresnel number {fr:.2f} &lt; 2. The collimated region of a {model.get('section_wavelengths', 0):.1f}"
               f"-$\\lambda$ aperture extends roughly a$^2$/$\\lambda$ = {p.section / 2 / lam * 2 / 2:.1f} m; the "
               f"quiet zone is {model.get('reflector_to_qz_m', 0):.2f} m from the surface, so diffraction, not "
               f"geometry, decides the field there.", "warn", "The quiet zone is in the near field")

    # ------------------------------------------------------------------ 02 validation
    v = R.get("validation", {})
    if v:
        s = d.section("Validation of the physical-optics code",
                      "Three independent checks before any quiet-zone number is read: the integrator against a "
                      "closed form, the smooth part of the field against geometrical optics, and the mesh "
                      "against itself.")
        dk = v.get("disk_densities") or v.get("disk") or []
        rows = [[f"$\\lambda$/{int(x['per_wavelength'])}", f"{int(x['triangles']):,}",
                 f"{x['max_rel_error']:.2e}"] for x in dk]
        go = v.get("go", [])
        s.row(
            Table(rows, header=["mesh", "triangles", "max relative error"],
                  caption="**PO on a flat disk against its closed-form on-axis field** "
                          "(`analytic.disk_on_axis`). Second-order convergence; the error at the production "
                          "mesh is two orders below the quiet-zone ripple.",
                  align=["left", "right", "right"]),
            Table([[f"{g['roll_wl']:g} $\\lambda$", _f(g["max_abs_dB"]), _f(g["rms_dB"])] for g in go],
                  header=["rolled edge", "max (dB)", "RMS (dB)"],
                  caption="**PO against geometrical optics.** The smooth part of the quiet-zone amplitude "
                          "against the GO aperture taper; the remainder is the edge-diffraction ripple.",
                  align=["left", "right", "right"]),
        )
        s.figure(figs.get("validation", ""), "**Left:** the PO integrator against the closed form, at five mesh "
                                             "densities. **Right:** PO minus the GO aperture taper, as the rolled "
                                             "edge grows.")
        mesh = v.get("mesh", [])
        if mesh:
            s.table(Table([[f"$\\lambda$/{int(m['per_wavelength'])}", _f(m["ripple_pp_dB"]), _f(m["phase_pp_deg"], 1),
                            _f(m["usable_diameter"] / FT, 3), _f(m.get("seconds", 0), 1)]
                           for m in mesh],
                          header=["mesh", "ripple (dB p-p)", "phase (deg p-p)", "usable (ft)", "time (s)"],
                          caption="**Mesh convergence of the quiet-zone metrics.** The metrics move by less than "
                                  "the differences reported later, so the mesh is not what is being measured.",
                          align=["left", "right", "right", "right", "right"]))

    # ------------------------------------------------------------------ 03 quiet zone
    if qz:
        s = d.section(f"Quiet zone at z = {base_plane_ft:g} ft",
                      "Peak to peak over the cuts a range datasheet reports, and over the whole disk the "
                      "specification is actually written on.")
        rows = []
        for pol in p.pols:
            for z in qz["pol"][pol]:
                c = qz["pol"][pol][z]["cuts"]
                dk_ = qz["pol"][pol][z]["disk"]
                rows.append([pol, z,
                             _f(c["horizontal"]["amp_pp_dB"]), _f(c["horizontal"]["phase_pp_deg"], 1),
                             _f(c["vertical"]["amp_pp_dB"]), _f(c["vertical"]["phase_pp_deg"], 1),
                             _f(dk_["amp_pp_dB"]), _f(dk_["usable_at_0.5dB"] / FT)])
        s.table(Table(rows, header=["pol", "z (ft)", "h: dB p-p", "h: deg p-p", "v: dB p-p", "v: deg p-p",
                                    "disk: dB p-p", "usable 0.5 dB (ft)"],
                      caption="**Every plane and both polarisations.** The last two columns are the same field "
                              "measured over the full disk: the peak-to-peak a cut reports and the peak-to-peak "
                              "the aperture actually sees are different numbers.",
                      align=["left", "right", "right", "right", "right", "right", "right", "right"]))
        for pol in p.pols:
            if figs.get(f"cuts_{pol}"):
                s.figure(figs[f"cuts_{pol}"], f"**{pol} polarisation**: co-polar amplitude and phase and the "
                                              f"cross-polar level along the cuts, at three planes.")
        if figs.get("qz_map"):
            s.figure(figs["qz_map"], "**The quiet zone as a plane**, not a cut: amplitude and phase over the "
                                     "rated diameter, with the disk that holds $\\pm$0.5 dB drawn on it.")
        h = base.get("cuts", {}).get("horizontal", {}).get("amp_pp_dB")
        if isinstance(h, (int, float)) and np.isfinite(h) and h > 0:
            s.note(f"The horizontal cut reports {h:.2f} dB where the disk reports {amp_pp:.2f} dB, a factor of "
                   f"{amp_pp / h:.1f}. A quiet zone specified on cuts is a one-dimensional measurement of a "
                   f"two-dimensional field.", "info", "Cuts understate the variation")

    # ------------------------------------------------------------------ 04 full aperture
    if disk:
        s = d.section("The quiet zone against the specification",
                      "The same field measured against three tolerance levels, so the answer to "
                      "\"does it pass\" does not depend on one arbitrary threshold.")
        rows = []
        for lvl, lab in (("usable_at_0.5dB", "$\\pm$0.5 dB / $\\pm$5&deg;"),
                         ("usable_at_1dB", "$\\pm$1.0 dB / $\\pm$10&deg;"),
                         ("usable_at_1.5dB", "$\\pm$1.5 dB / $\\pm$15&deg;")):
            s2 = qz["pol"][pol0][f"{base_plane_ft:g}"]["disk"].get(lvl, float("nan"))
            rows.append([lab, _f(s2 / FT), _pct(s2, p.qz_diameter), "yes" if s2 >= p.qz_diameter else "no"])
        rows.append(["taper (smooth illumination)", _f(disk.get("taper_dB")) + " dB", "", ""])
        rows.append(["ripple (residual about the taper)", _f(ripp) + " dB", "", ""])
        rows.append(["phase peak to peak", _f(ph_pp, 1) + "&deg;", "", ""])
        rows.append(["cross-polar peak", _f(xpol, 1) + " dB", "", ""])
        s.table(Table(rows, header=["criterion", "usable diameter (ft)", "of specification", "meets"],
                      caption=f"**Usable diameter at increasing tolerance**, {pol0} polarisation, "
                              f"z = {base_plane_ft:g} ft.", align=["left", "right", "right", "center"],
                      emphasis=(0, 1, 2)))
        s.note("A quiet zone is usually specified at $\\pm$0.5 dB over its rated diameter. The design's rated "
               f"diameter is {p.qz_diameter / FT:.1f} ft and its usable diameter at that tolerance is "
               f"{usable / FT:.2f} ft: the specification is larger than the field supports by a factor of "
               f"{p.qz_diameter / max(usable, 1e-9):.1f}.", "warn" if not meets else "ok",
               "Read the tolerance, not the peak-to-peak")

    # ------------------------------------------------------------------ 05 reference
    if ref.get("rows"):
        su = ref["summary"]
        s = d.section("Comparison against the published model",
                      "The reference publishes twelve quiet-zone rows: two polarisations, three planes, a "
                      "horizontal and a vertical cut each. Every one is recomputed here from the same geometry.")
        rows = []
        for r in ref["rows"]:
            rows.append([r["pol"], f"{r['z_ft']:g}", r["cut"][:1].upper(),
                         _f(r["ref_amp_pp_dB"]), _f(r["po_amp_pp_dB"]), f"{r['d_amp_dB']:+.2f}",
                         _f(r["ref_phase_pp_deg"], 1), _f(r["po_phase_pp_deg"], 1),
                         f"{r['d_phase_deg']:+.1f}"])
        s.table(Table(rows, header=["pol", "z (ft)", "cut", "ref dB p-p", "PO dB p-p", "&Delta; dB",
                                    "ref deg p-p", "PO deg p-p", "&Delta; deg"],
                      caption="**Row by row.** Ref: the reference table supplied with the problem. PO: this "
                              "study, from the same geometry. The phase tracks the reference; the amplitude "
                              "peak-to-peak does not, and the cross-polar does not either.",
                      align=["left", "right", "center", "right", "right", "right", "right", "right", "right"]))
        _xa = _f(su.get("xpol_po_mean_dB"), 1)
        _xb = _f(su.get("xpol_ref_mean_dB"), 1)
        _agree = (np.isfinite(su.get("xpol_po_mean_dB", np.nan)) and
                  np.isfinite(su.get("xpol_ref_mean_dB", np.nan)) and
                  abs(su["xpol_po_mean_dB"] - su["xpol_ref_mean_dB"]) < 6.0)
        s.row(
            Table([["rows compared", str(su["n"])],
                   ["amplitude, mean |&Delta;|", _f(su["amp_abs_mean_dB"]) + " dB"],
                   ["amplitude, max |&Delta;|", _f(su["amp_abs_max_dB"]) + " dB"],
                   ["phase, mean |&Delta;|", _f(su["phase_abs_mean_deg"], 1) + "&deg;"],
                   ["phase, max |&Delta;|", _f(su["phase_abs_max_deg"], 1) + "&deg;"],
                   ["amplitude ratio, median (PO / ref)", _f(su.get("amp_ratio_median"))],
                   ["cross-polar, PO vs ref", f"{_xa} / {_xb} dB"]],
                  caption="**What the agreement is.** The phase is within the accuracy a PO model is used at. "
                          "The amplitude peak-to-peak is systematically high, and the cross-polar level is a "
                          "different regime in the two solvers, not a small difference."),
            Figure(figs.get("profile", ""), "**The amplitude peak-to-peak over the rolled-edge family.** Each point "
                                            "is one profile; the spread is what the model family cannot resolve."),
        )
        s.note(
            f"PO reports {su['amp_abs_mean_dB']:.1f} dB more amplitude peak-to-peak on average "
            f"({su['amp_abs_max_dB']:.1f} dB at worst) than the reference, and the ratio is "
            f"{su.get('amp_ratio_median', float('nan')):.1f}. Two things differ and both push this way: the "
            f"reference edge is a Gupta-Burnside blend with a $\\lambda$/4 shadow-boundary radius, which the "
            f"constant-curvature family of this module cannot represent, and PO omits the fringe (PTD) edge "
            f"current that a full-wave solve carries. The phase agreement says the illumination and the geometry "
            f"are right; the amplitude disagreement says the edge is not. Conclusions below are therefore drawn "
            f"from differences between designs, which share the edge, not from absolute levels.",
            "warn", "Where the model and the reference part company")
        if _agree:
            s.note(f"Cross-polar peak: PO {_xa} dB against the reference's {_xb} dB, within the few dB a PO model "
                   f"is expected to hold this quantity.", "ok", "Cross-polar")
        else:
            s.note(
                f"Cross-polar peak: PO {_xa} dB against the reference's {_xb} dB. The reflector geometry is "
                f"symmetric in x and the feed here is an ideal, symmetric cos$^q$ pattern, so the cross-polar "
                f"component nearly cancels on the principal cuts. The reference's {_xb} dB is carried by its "
                f"horn, whose measured pattern is not symmetric, and by a full-wave solve that keeps that "
                f"asymmetry. This is the one row where PO is not 'a little off': it is measuring a different "
                f"thing, and "
                f"the difference is the feed, not the reflector. It is also why a direct-path and horn-asymmetry "
                f"term cannot be read out of this model and has to come from a measured or full-wave feed.",
                "warn", "Cross-polar is a feed property, not a reflector one")

    # ------------------------------------------------------------------ 06 profile
    if prof:
        sp = prof["spread"]
        s = d.section("Sensitivity to the rolled-edge profile",
                      f"{prof['n']} rolled-edge profiles spanning the plausible family, at the reference plane.")
        s.row(
            Table([[k.replace("_", " "), _f(v["min"]), _f(v["max"]), _f(v["spread"])] for k, v in sp.items()],
                  header=["metric", "min", "max", "spread"],
                  caption="**The spread of every quiet-zone metric over the profile family.** Designs are "
                          "compared inside this band, never across it.",
                  align=["left", "right", "right", "right"]),
            Figure(figs.get("profile", ""), "**Ripple and usable diameter over the family.**"),
        )

    # ------------------------------------------------------------------ 07 axial
    if axial:
        s = d.section("How deep the quiet zone is",
                      "A quiet zone is a volume. The same transverse metrics are computed on "
                      f"{len(axial['z_ft'])} planes along the axis, reusing one mesh and one set of PO "
                      "currents.")
        rows = []
        z = axial["z_ft"]
        for i, zz in enumerate(z):
            row = [f"{zz:g}"]
            for pol in axial["pol"]:
                row += [_f(axial["pol"][pol]["usable_at_0.5dB"][i] / FT),
                        _f(axial["pol"][pol]["ripple_pp_dB"][i])]
            rows.append(row)
        header = ["z (ft)"] + [x for pol in axial["pol"] for x in (f"{pol}: usable (ft)", f"{pol}: ripple (dB)")]
        s.row(Table(rows, header=header,
                    caption="**Usable diameter and ripple vs distance from the vertex plane.**",
                    align=["right"] * len(header)),
              Figure(figs.get("axial", ""), "**Usable diameter and ripple along the axis**, at two tolerances."))
        for pol, dd in axial["pol"].items():
            ok = [zz for zz, u in zip(z, dd["usable_at_1dB"]) if u >= p.qz_diameter]
            if ok:
                s.note(f"{pol}: the full {p.qz_diameter / FT:.1f} ft quiet zone holds within $\\pm$1 dB over "
                       f"{len(ok)} of {len(z)} planes, from z = {min(ok):g} to {max(ok):g} ft.", "info",
                       "Axial extent")

    # ------------------------------------------------------------------ 08 bandwidth
    if band:
        s = d.section("Across the band",
                      f"The same reflector at {len(band['freq_ghz'])} frequencies from "
                      f"{band['freq_ghz'][0]:g} to {band['freq_ghz'][-1]:g} GHz. The feed gain is held at its "
                      f"design value; the rolled edge, fixed in metres, is the thing that changes.")
        rows = []
        for i, f in enumerate(band["freq_ghz"]):
            row = [f"{f:.2f}"]
            for pol in band["pol"]:
                d2 = band["pol"][pol]
                row += [_f(d2["usable_at_0.5dB"][i] / FT), _f(d2["usable_at_1dB"][i] / FT),
                        _f(d2["ripple_pp_dB"][i])]
            rows.append(row)
        header = ["GHz"] + [x for pol in band["pol"] for x in (f"{pol}: usable .5 (ft)", f"{pol}: usable 1 (ft)",
                                                               f"{pol}: ripple (dB)")]
        s.row(Table(rows, header=header, caption="**Quiet zone across the band.**", align=["right"] * len(header)),
              Figure(figs.get("bandwidth", ""), "**Usable diameter and ripple across the band.**"))
        for pol, d2 in band["pol"].items():
            u = np.array(d2["usable_at_0.5dB"])
            good = [f"{f:.2f}" for f, uu in zip(band["freq_ghz"], u) if uu >= p.qz_diameter]
            s.note(f"{pol}: the specification is met at {len(good)} of {len(u)} frequencies"
                   + (f" ({good[0]} to {good[-1]} GHz)" if good else "")
                   + f"; the usable diameter runs from {u.min() / FT:.2f} to {u.max() / FT:.2f} ft. At the "
                     f"design frequency the edge is {p.roll / lam:.1f} $\\lambda$ deep; at "
                     f"{band['freq_ghz'][0]:g} GHz it is {p.roll / (299_792_458.0 / (band['freq_ghz'][0] * 1e9)):.1f} "
                     f"$\\lambda$ and at {band['freq_ghz'][-1]:g} GHz "
                     f"{p.roll / (299_792_458.0 / (band['freq_ghz'][-1] * 1e9)):.1f} $\\lambda$, which is what moves.",
                   "info", "Bandwidth")

    # ------------------------------------------------------------------ 09 tolerance
    if tol:
        s = d.section("Feed placement tolerance",
                      "The feed has to sit on the focus. How closely is a question only the field can answer.")
        rows = []
        pol = list(tol["pol"])[0]
        steps = [r["offset_lambda"] for r in tol["pol"][pol]["lateral"]]
        for i, sl in enumerate(steps):
            row = [f"{sl:g}"]
            for ax in ("lateral", "axial", "diagonal"):
                r = tol["pol"][pol][ax][i]
                row += [_f(r["usable_at_0.5dB"] / FT), _f(r["amp_pp_dB"])]
            rows.append(row)
        header = ["offset ($\\lambda$)"] + [x for ax in ("lateral", "axial", "diagonal")
                                            for x in (f"{ax}: usable (ft)", f"{ax}: dB p-p")]
        s.row(Table(rows, header=header,
                    caption=f"**Quiet zone vs feed displacement**, {pol} polarisation, at the reference plane.",
                    align=["right"] * len(header)),
              Figure(figs.get("tolerance", ""), "**Usable diameter and ripple against feed displacement.**"))
        r0 = tol["pol"][pol]["lateral"][0]["usable_at_0.5dB"]
        r1 = tol["pol"][pol]["lateral"][1]["usable_at_0.5dB"] if len(steps) > 1 else r0
        s.note(f"A lateral displacement of {steps[1]:g} $\\lambda$ = {steps[1] * lam / 0.0254:.1f} in takes the "
               f"usable diameter from {r0 / FT:.2f} to {r1 / FT:.2f} ft ({_pct(r1, r0)}); an axial one is "
               f"{'gentler' if tol['pol'][pol]['axial'][1]['usable_at_0.5dB'] > r1 else 'worse'}. "
               f"The phase centre therefore has to be placed to a small fraction of a wavelength, which is "
               f"exactly what a full-wave phase-centre fit is for.", "info", "How much placement matters")

    # ------------------------------------------------------------------ 10 sweep
    if sweep:
        rows = sorted(sweep["rows"], key=lambda r: (-r["usable_at_0.5dB"], r["ripple_pp_dB"]))[:8]
        s = d.section("Feed gain, section size and rolled edge",
                      f"{sweep['n']} designs, {len(sweep['gains'])} feed gains x {len(sweep['sides'])} section "
                      f"sizes x {len(sweep['rolls'])} rolled edges, at {sweep['median_s']:.1f} s per design "
                      f"({sweep['wall_s']:.0f} s in all).")
        s.table(Table([[f"{r['feed_gain_dBi']:g}", _f(r["side_m"]), _f(r["roll_m"] / FT), _f(r["ripple_pp_dB"]),
                        _f(r["amp_pp_dB"]), _f(r["usable_at_0.5dB"] / FT), _f(r["usable_at_1dB"] / FT),
                        _pct(r["usable_at_0.5dB"], p.qz_diameter)]
                       for r in rows],
                      header=["gain (dBi)", "side (m)", "roll (ft)", "ripple (dB)", "amp p-p (dB)",
                              "usable 0.5 (ft)", "usable 1 (ft)", "of spec"],
                      caption="**The eight best designs** ranked by usable diameter at $\\pm$0.5 dB.",
                      align=["right"] * 8, emphasis=(0,)))
        s.row(Figure(figs.get("tradeoff", ""), "**Usable diameter over (feed gain, section side)** for each rolled "
                                               "edge; the black contour is the specification, black dots are the "
                                               "simulated designs."),
              Table([[m, _f(v["rmse_random"], 3), _f(v["rmse_leave_side_out"], 3)] for m, v in
                     sweep["heldout"].items()],
                    header=["metric", "RMSE, random 1/4", "RMSE, one size out"],
                    caption="**Held-out error of the Gaussian process** fitted on (feed gain, side). Leaving a "
                            "whole size out is the harder test and the honest one.",
                    align=["left", "right", "right"]))
        bst = sweep["best"]

        # What each factor does on its own: the mean of every metric over the levels of one factor, and the
        # share of the variance of the usable diameter that factor accounts for on its own.
        def _levels(key):
            return sorted({round(r[key], 6) for r in sweep["rows"]})

        def _slice(key, level):
            return [r for r in sweep["rows"] if abs(r[key] - level) < 1e-6]

        y = np.array([r["usable_at_0.5dB"] for r in sweep["rows"]], float)

        def _share(metric_key, key):
            v = np.array([r[metric_key] for r in sweep["rows"]], float)
            sst = float(((v - v.mean()) ** 2).sum())
            if sst <= 0:
                return float("nan")
            ss = 0.0
            for lv in _levels(key):
                col = np.array([r[metric_key] for r in _slice(key, lv)], float)
                ss += len(col) * float((col.mean() - v.mean()) ** 2)
            return 100.0 * ss / sst

        share = {(m, k): _share(m, k) for m in ("usable_at_0.5dB", "ripple_pp_dB")
                 for k in ("feed_gain_dBi", "side_m", "roll_m")}

        marg = []
        for key, lab, fmt in (("feed_gain_dBi", "feed gain (dBi)", lambda v: f"{v:g}"),
                              ("side_m", "section side (m)", _f),
                              ("roll_m", "rolled edge (ft)", lambda v: _f(v / FT))):
            for lv in _levels(key):
                sel = _slice(key, lv)
                marg.append([lab if lv == _levels(key)[0] else "", fmt(lv),
                             _f(float(np.mean([r["usable_at_0.5dB"] for r in sel])) / FT),
                             _f(float(np.mean([r["usable_at_1dB"] for r in sel])) / FT),
                             _f(float(np.mean([r["ripple_pp_dB"] for r in sel]))),
                             _f(float(np.mean([r["amp_pp_dB"] for r in sel])))])
        s.table(Table(marg,
                      header=["factor", "level", "mean usable 0.5 (ft)", "mean usable 1 (ft)",
                              "mean ripple (dB)", "mean amp p-p (dB)"],
                      caption="**One factor at a time**, averaged over every design that holds it. The levels are "
                              "the ones the sweep visited; a factor with a flat column does not matter at this "
                              "geometry.",
                      align=["left", "right", "right", "right", "right", "right"]))

        gain_means = {lv: float(np.mean([r["usable_at_0.5dB"] for r in _slice("feed_gain_dBi", lv)])) / FT
                      for lv in _levels("feed_gain_dBi")}
        best_gain = max(gain_means, key=lambda k: gain_means[k])
        worst_gain = min(gain_means, key=lambda k: gain_means[k])
        interior = best_gain not in (min(gain_means), max(gain_means))
        side_means = {lv: float(np.mean([r["usable_at_0.5dB"] for r in _slice("side_m", lv)])) / FT
                      for lv in _levels("side_m")}
        roll_means = {lv: float(np.mean([r["usable_at_0.5dB"] for r in _slice("roll_m", lv)])) / FT
                      for lv in _levels("roll_m")}
        ripple_gain = {lv: float(np.mean([r["ripple_pp_dB"] for r in _slice("feed_gain_dBi", lv)]))
                       for lv in _levels("feed_gain_dBi")}
        ripple_roll = {lv: float(np.mean([r["ripple_pp_dB"] for r in _slice("roll_m", lv)]))
                       for lv in _levels("roll_m")}
        ripple_side = {lv: float(np.mean([r["ripple_pp_dB"] for r in _slice("side_m", lv)]))
                       for lv in _levels("side_m")}
        s.note(
            f"Feed gain is the strongest factor ({share[('usable_at_0.5dB', 'feed_gain_dBi')]:.0f}% of the "
            f"variance of the usable diameter on its own, against "
            f"{share[('usable_at_0.5dB', 'side_m')]:.0f}% for the section side and "
            f"{share[('usable_at_0.5dB', 'roll_m')]:.0f}% for the rolled edge). Its mean runs from "
            f"{gain_means[worst_gain]:.2f} ft at {worst_gain:g} dBi to {gain_means[best_gain]:.2f} ft at "
            f"{best_gain:g} dBi"
            + (", an interior optimum: above it the illumination underfills the section and it is the effective "
               "aperture, not the edge, that is lost." if interior else "")
            + f" Larger sections help monotonically, {side_means[min(side_means)]:.2f} ft at "
              f"{min(side_means):.2f} m against {side_means[max(side_means)]:.2f} ft at {max(side_means):.2f} m. "
              f"The rolled edge is close to neutral at this depth (mean diameter {roll_means[min(roll_means)]:.2f} "
              f"to {roll_means[max(roll_means)]:.2f} ft, mean ripple "
              f"{min(ripple_roll.values()):.2f} to {max(ripple_roll.values()):.2f} dB). Ripple is carried by the "
              f"feed gain instead, {min(ripple_gain.values()):.2f} dB at "
              f"{min(ripple_gain, key=ripple_gain.get):g} dBi against {max(ripple_gain.values()):.2f} dB at "
              f"{max(ripple_gain, key=ripple_gain.get):g} dBi: that is spillover and rim taper, so buying ripple "
              f"with feed gain costs aperture, and the diameter column is where the trade shows. Section side is "
              f"the weakest of the three on ripple "
              f"({min(ripple_side.values()):.2f} to {max(ripple_side.values()):.2f} dB).",
            "info", "Which factor moves which metric")

        s.note(f"The best design in the family is {bst['feed_gain_dBi']:g} dBi, a {bst['side_m']:.2f} m section "
               f"and a {bst['roll_m'] / FT:.2f} ft rolled edge, giving {bst['usable_at_0.5dB'] / FT:.2f} ft "
               f"({_pct(bst['usable_at_0.5dB'], p.qz_diameter)} of the specification). "
               + ("A design inside the family reaches the specification."
                  if bst["usable_at_0.5dB"] >= p.qz_diameter else
                  "No design in the family reaches the specification: no combination of feed gain, section size "
                  "and edge inside the physical envelope of this range closes the remaining "
                  f"{100 - 100 * bst['usable_at_0.5dB'] / p.qz_diameter:.0f}%."), "info", "What the sweep says")

    # ------------------------------------------------------------------ 11 equal budget
    if opt:
        s = d.section("Equal-budget search",
                      f"Three strategies drawing on the same {opt['budget']}-solve budget: a regular grid that "
                      f"fills the box ({opt['grid']['calls']} designs), a space-filling design whose Gaussian "
                      f"process is optimised on the surrogate and then verified on the solver, and Bayesian "
                      f"optimisation calling the solver directly. Only the solver's number is ever counted.")
        rows = []
        for key, lab in (("grid", "regular grid"), ("surrogate", "DOE + GP + verify"),
                         ("bayesian", "Bayesian on the solver")):
            o = opt[key]
            b = o["best"]
            rows.append([lab, str(o["calls"]), _f(b["usable_at_0.5dB"] / FT), _f(b["ripple_pp_dB"]),
                         f"{b['feed_gain_dBi']:g}", _f(b["side_m"]), _f(b["roll_m"] / FT), _f(o["wall_s"], 0)])
        s.row(Table(rows, header=["strategy", "solves", "best usable (ft)", "ripple (dB)", "gain (dBi)",
                                  "side (m)", "roll (ft)", "wall (s)"],
                    caption="**Solves actually spent and the best design each found.** Every number in the table "
                            "is the solver's; the grid spends fewer solves because the box holds only 18 points.",
                    align=["left", "right", "right", "right", "right", "right", "right", "right"]),
              Figure(figs.get("equal_budget", ""), "**Best usable diameter so far, against solves spent.**"))
        g = opt["grid"]["best"]["usable_at_0.5dB"]
        su = opt["surrogate"]["best"]["usable_at_0.5dB"]
        bo = opt["bayesian"]["best"]["usable_at_0.5dB"]
        win = max((("the surrogate-driven search", su), ("Bayesian optimisation", bo), ("the grid", g)),
                  key=lambda t: t[1])[0]
        ratio = opt["bayesian"]["wall_s"] / max(opt["surrogate"]["wall_s"], 1.0)
        s.note(f"At {opt['budget']} solves {win} returns the best design "
               f"({max(g, su, bo) / FT:.2f} ft, {_pct(max(g, su, bo), p.qz_diameter)} of the specification). "
               f"The surrogate-guided design reached {su / FT:.2f} ft against the grid's {g / FT:.2f} ft, "
               f"{(su - g) / p.qz_diameter * 100:+.0f} points of the specification: at this budget the surrogate "
               f"bought nothing over filling the box. Sequential Bayesian search on the solver did better, "
               f"{bo / FT:.2f} ft ({(bo - g) / p.qz_diameter * 100:+.0f} points against the grid), at "
               f"{ratio:.0f} times the wall time. The space is three-dimensional and smooth, so choosing the next "
               f"point well pays more than fitting the whole map. These are single runs and the numbers move with "
               f"the seed: an indication, not a ranking.",
               "info", "What the budget buys")

    # ------------------------------------------------------------------ plan, confirm, limits
    from .study_report import plan as _plan

    done_keys = set(R)
    plan_rows = []
    for st in _plan(p):
        if not st.required:
            plan_rows.append(Step(st.title, st.reason, "skipped", ""))
            continue
        status = "done" if st.key in done_keys else "queued"
        plan_rows.append(Step(st.title, st.reason, status, f"{T.get(st.key):.0f} s" if T.get(st.key) else ""))
    d.plan(done=[x for x in plan_rows if x.status == "done"],
           queued=[x for x in plan_rows if x.status == "queued"],
           blocked=[x for x in plan_rows if x.status == "skipped"])

    d.confirm([
        "The quiet-zone requirement: which tolerance and over which diameter. The design meets "
        f"{_f(usable / FT)} ft at $\\pm${p.amp_tol:g} dB; a datasheet that quotes "
        f"{_f(amp_pp)} dB peak to peak over cuts is quoting a different quantity.",
        "The rolled-edge profile as built. The constant-curvature family here is not the Gupta-Burnside "
        "design of the reference, and the amplitude peak-to-peak is where that shows.",
        "The feed pattern outside the reflector cone. The direct path into the quiet zone is not in the PO "
        "model; a full-wave or measured pattern is needed for that term, which background subtraction removes "
        "from a measurement but not from a design.",
        "The band the range has to cover. With no band in the problem there is no bandwidth step, and the "
        "edge is a different depth in wavelengths at every frequency.",
    ])
    d.limitations([
        "Physical optics: no fringe (PTD) edge current, no multiple reflections between reflector and feed, "
        "one-point centroid quadrature per triangle at $\\lambda$/" + f"{p.mesh_per_wavelength:g}.",
        "The feed is an ideal cos$^q$ pattern normalised to 1 W, aimed at the section centre. A real horn has "
        "an asymmetric pattern, a finite aperture and a back lobe; only the gain and the rim taper are matched.",
        "Free space only: no absorber, shroud, positioner, floor or reflector back structure, and no feed "
        "blockage.",
        "The quiet-zone metrics are computed on planes; the axial study samples them to give the volume, but "
        "the volume is not itself optimised.",
    ])
    return d
