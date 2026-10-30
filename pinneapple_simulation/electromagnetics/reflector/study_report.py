"""A compact-range study that plans itself: problem -> which checks and sweeps to run -> results -> datasheet.

Given a :class:`Problem` (frequency, focal length, section, rolled edge, feed, quiet zone and, optionally, a
measured or simulated reference table and a band) this module

1. **plans** the study from the geometry alone (:func:`plan`) -- the steps, and the reason each one is there:
   a Fresnel number below two means the quiet zone is in the near field of the section and a size sweep is
   called for; a rolled edge shorter than three wavelengths means the edge diffraction needs its own sweep; a
   reference table means a comparison and a rolled-edge calibration; a spec the baseline misses means an
   optimisation at equal solver budget; no band means no bandwidth step, and the plan says so;
2. **runs** it (:func:`run`), writing every number to JSON so the document can be regenerated without solving;
3. **renders** it as a technical datasheet (:func:`report`), by number, table and figure.

No language model is involved anywhere: the steps come from thresholds on dimensionless numbers computed from
the geometry, and the sentences come from the numbers themselves.

    python -m pinneapple_simulation.electromagnetics.reflector.study_report --out cr_study
    python -m pinneapple_simulation.electromagnetics.reflector.study_report --freq 3 --focal 1.5 --section 1.2 \\
        --offset 0.9 --roll 0.15 --qz 0.6 --qz-distance 3 --gain 12 --band 2 4
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

FT = 0.3048
IN = 0.0254

__all__ = ["Problem", "plan", "run", "report", "from_text", "Problem.from_text"]


# ==================================================================================================
# The problem
# ==================================================================================================
@dataclass
class Problem:
    """A compact-range design and what it must deliver.

    Lengths in metres; ``band_hz`` is the frequency range the range must work over (optional);
    ``reference`` is a table of published or measured quiet-zone numbers to compare against, keyed
    ``(pol, z_ft, cut)`` -> ``(amp_p-p_dB, phase_p-p_deg, xpol_dB_or_None)``.
    """
    freq_hz: float = 3.0e9                               # the CompactRange default, so the two agree
    focal: float = 1.5
    section: float = 1.2
    offset: float = 0.9
    roll: float = 0.15
    feed_gain_dbi: float = 12.0
    qz_diameter: float = 0.6
    qz_z: float = 3.0
    n: float = 6.0
    pols: tuple[str, ...] = ("V", "H")
    band_hz: tuple[float, float] | None = None
    amp_tol: float = 0.5
    phase_tol: float = 5.0
    cut_points: int = 121
    mesh_per_wavelength: float = 8.0
    qz_step_per_wavelength: float = 6.0
    # rolled-edge profile (the family of the geometry module; calibrated when a reference is given)
    roll_aspect: float = 1.0
    blend: float = 0.3
    end_angle: float = -90.0
    # what the model was asked to match
    reference: Mapping[tuple[str, float, str], tuple[float, float, float | None]] | None = None
    reference_geometry: Mapping[str, float] | None = None
    title: str = "Compact range"
    name: str = "compact_range"

    # -- derived -------------------------------------------------------------------------------
    @property
    def wavelength(self) -> float:
        return 299_792_458.0 / self.freq_hz

    def reflector(self, **over):
        from pinneapple_simulation.electromagnetics.reflector import OffsetReflector
        kw = dict(focal=self.focal, half_x=self.section / 2, half_y=self.section / 2, centre_y=self.offset,
                  n=self.n, roll=self.roll, roll_aspect=self.roll_aspect, blend=self.blend,
                  end_angle=self.end_angle)
        kw.update(over)
        return OffsetReflector(**kw)

    def cr(self, pol: str = "V", **over):
        """The :class:`~...compact_range.CompactRange` for this problem.

        ``freq_hz``, ``feed_gain_dbi``, ``qz_z``, ``qz_diameter``, ``feed_offset``,
        ``mesh_per_wavelength`` and ``qz_step_per_wavelength`` go to the range, ``section`` resizes the
        section, ``reflector`` replaces it whole, and the remaining reflector keys are applied to it.
        """
        from pinneapple_simulation.electromagnetics.reflector import CompactRange
        given = dict(over)
        top_keys = ("freq_hz", "feed_gain_dbi", "qz_z", "qz_diameter", "feed_offset", "mesh_per_wavelength",
                    "qz_step_per_wavelength")
        top = {k: given.pop(k) for k in top_keys if k in given}
        refl_keys = ("focal", "half_x", "half_y", "centre_y", "n", "roll", "roll_aspect", "blend", "end_angle")
        refl = given.pop("reflector", None) or self.reflector(
            **{k: v for k, v in given.items() if k in refl_keys})
        if "section" in given:
            s = given.pop("section")
            refl = replace(refl, half_x=s / 2, half_y=s / 2, centre_y=refl.centre_y - refl.half_y + s / 2)
        if given:
            raise TypeError("unknown design keys: " + ", ".join(sorted(given)))
        return CompactRange(freq_hz=top.get("freq_hz", self.freq_hz), reflector=refl,
                            feed_gain_dbi=top.get("feed_gain_dbi", self.feed_gain_dbi), pol=pol,
                            qz_z=top.get("qz_z", self.qz_z), qz_diameter=top.get("qz_diameter", self.qz_diameter),
                            feed_offset=top.get("feed_offset", (0.0, 0.0, 0.0)),
                            mesh_per_wavelength=top.get("mesh_per_wavelength", self.mesh_per_wavelength),
                            qz_step_per_wavelength=top.get("qz_step_per_wavelength",
                                                           self.qz_step_per_wavelength))

    def facts(self, cr=None) -> dict[str, float]:
        """Dimensionless numbers that decide what the study has to check."""
        from pinneapple_simulation.electromagnetics.reflector import mesh_reflector
        cr = cr or self.cr()
        lam, r = cr.wavelength, cr.reflector
        z_eff = cr.qz_z - (r.centre_y ** 2) / (4 * r.focal)
        mesh = mesh_reflector(r, lam / cr.mesh_per_wavelength)
        a = max(r.half_x, r.half_y)
        return {
            "fresnel_number": float(a ** 2 / (lam * z_eff)),
            "section_wavelengths": float(2 * a / lam),
            "roll_wavelengths": float(r.roll / lam),
            "qz_diameter_wavelengths": float(cr.qz_diameter / lam),
            "qz_to_section": float(cr.qz_diameter / (2 * a)),
            "reflector_to_qz_m": float(z_eff),
            "surface_area_m2": mesh.area,
            "surface_area_wavelengths2": float(mesh.area / lam ** 2),
            "n_triangles": int(len(mesh.areas)),
            "focal_to_diameter": float(r.focal / (2 * a)),
        }

    # -- from text -----------------------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str, **defaults) -> Problem:
        """Parse a design written in plain words with explicit units, e.g.

        ``"3 GHz compact range, focal length 1.5 m, 1.2 m square reflector, 0.15 m rolled edge,
        0.6 m quiet zone at 3 m, feed 12 dBi"``

        Imperial units are accepted the same way: ``"focal length 5 ft, 4 ft square reflector"``.

        Recognises ``GHz``/``MHz`` (frequency), ``ft``/``m``/``in`` (lengths), ``dBi`` (feed gain), and the
        keywords *focal*, *reflector*/*section*, *rolled edge*/*roll*, *quiet zone*/*qz*, *at* (distance),
        *offset*.  Raises ``ValueError`` naming what it could not find -- it does not guess.
        """
        t = text.lower()
        unit = {"ft": FT, "feet": FT, "foot": FT, "m": 1.0, "in": IN, "inch": IN, "mm": 1e-3}
        tok = re.compile(r"(-?\d+(?:\.\d+)?)\s*(ghz|mhz|ft|feet|foot|inch|in|mm|m|dbi)\b")

        # The text is read clause by clause (a clause ends at a comma, a semicolon or a full stop), because
        # that is how a design is written: "4 ft square reflector, 1 ft rolled edge" puts the size of each
        # thing in its own clause.  Inside the clause that names the thing, the length written *before* the
        # keyword wins ("4 ft square reflector"), and the one after it is the fallback ("reflector, 4 ft").
        clauses = []
        cut = 0
        for sep in re.finditer(r"[,;]|\.(?=\s|$)", t):   # a decimal point is not a clause break
            clauses.append((cut, sep.start()))
            cut = sep.end()
        clauses.append((cut, len(t)))

        def length_for(pattern: str) -> float | None:
            for c0, c1 in clauses:
                seg = t[c0:c1]
                km = re.search(pattern, seg)
                if not km:
                    continue
                toks = [(m.start(), m.end(), float(m.group(1)), m.group(2)) for m in tok.finditer(seg)]
                toks = [x for x in toks if x[3] in unit]
                if not toks:
                    continue
                before = [x for x in toks if x[1] <= km.start()]
                after = [x for x in toks if x[0] >= km.end()]
                pick = before[-1] if before else (after[0] if after else None)
                if pick is not None:
                    return pick[2] * unit[pick[3]]
            return None

        kw: dict[str, Any] = dict(defaults)
        mf = re.search(r"(\d+(?:\.\d+)?)\s*(ghz|mhz)", t)
        if mf:
            kw["freq_hz"] = float(mf.group(1)) * (1e9 if mf.group(2) == "ghz" else 1e6)
        for key, pat in (("focal", r"focal(?:\s+length)?"), ("section", r"(?:reflector|section|aperture)"),
                         ("roll", r"(?:rolled[-\s]?edge|roll)"), ("qz_diameter", r"(?:quiet[-\s]?zone|qz)"),
                         ("offset", r"offset")):
            v = length_for(pat)
            if v is not None:
                kw[key] = v
        # The distance goes with the quiet zone, and it is read inside that clause too: "0.6 m quiet zone at
        # 3 m" has two lengths in one clause and a decimal point in the first, which a "no full stop" guard
        # would read as the end of the sentence.
        for c0, c1 in clauses:
            seg = t[c0:c1]
            km = re.search(r"(?:quiet[-\s]?zone|qz)", seg)
            if not km:
                continue
            md = re.search(r"(?:at|centred at)\s*(-?\d+(?:\.\d+)?)\s*(ft|feet|foot|inch|in|mm|m)\b",
                           seg[km.end():])
            if md:
                kw["qz_z"] = float(md.group(1)) * unit[md.group(2)]
                break
        mg = re.search(r"(-?\d+(?:\.\d+)?)\s*dbi", t)
        if mg:
            kw["feed_gain_dbi"] = float(mg.group(1))
        mb = re.search(r"(?:band|from)\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*ghz", t)
        if mb:
            kw["band_hz"] = (float(mb.group(1)) * 1e9, float(mb.group(2)) * 1e9)
        missing = [k for k in ("freq_hz", "focal", "section", "qz_diameter", "qz_z") if k not in kw]
        if missing:
            raise ValueError("could not read " + ", ".join(missing) + " from the text; give them with units")
        return cls(**kw)


def from_text(text: str, **defaults) -> Problem:
    """:meth:`Problem.from_text`."""
    return Problem.from_text(text, **defaults)


# ==================================================================================================
# The plan: heuristics, no model
# ==================================================================================================
@dataclass
class PlannedStep:
    key: str
    title: str
    reason: str
    required: bool = True


def plan(problem: Problem) -> list[PlannedStep]:
    """The steps this problem needs, and **why** -- from thresholds on dimensionless numbers only."""
    from pinneapple_simulation.electromagnetics.reflector.study import METRICS  # noqa: F401  (documented set)
    f = problem.facts()
    steps = [PlannedStep("model", "Model and resolution",
                         f"the section is {f['section_wavelengths']:.1f} wavelengths across and the mesher needs "
                         f"lambda/{problem.mesh_per_wavelength:g} triangles to integrate it"),
             PlannedStep("validation", "Validation of the physical-optics code",
                         "no number leaves this study before the integrator is checked against a closed form, "
                         "the quiet-zone amplitude against geometrical optics and the mesh against itself")]

    fr = f["fresnel_number"]
    steps.append(PlannedStep(
        "qz", "Quiet zone at the reference planes",
        f"Fresnel number a^2/(lambda z) = {fr:.2f} with a = {problem.section / 2 / problem.wavelength:.1f} "
        f"wavelengths: " + ("the quiet zone is in the near field of the section, so the field is computed "
                            "exactly as a near field" if fr < 2 else
                            "the quiet zone is in the far field of the section, where a taper description "
                            "would suffice")))
    steps.append(PlannedStep(
        "spec", "Full-aperture specification",
        "peak-to-peak over a cut is a one-dimensional measurement; the specification of a quiet zone is a "
        "disk, so both are computed and the difference is reported"))
    if problem.reference:
        steps.append(PlannedStep(
            "reference", "Comparison against the supplied reference",
            f"{len(problem.reference)} published quiet-zone numbers are compared row by row"))
    if problem.reference_geometry:
        steps.append(PlannedStep(
            "calibration", "Rolled-edge family against the published geometry",
            "the rolled edge of the reference is a Gupta-Burnside design, not the constant-curvature family of "
            "this module; the published extents and surface area are used to say how close the family can get, "
            "and the residual is reported rather than hidden by a fit"))
    if problem.band_hz:
        lo, hi = problem.band_hz
        steps.append(PlannedStep(
            "bandwidth", "Quiet zone across the band",
            f"the range must work from {lo / 1e9:g} to {hi / 1e9:g} GHz and the rolled edge is only "
            f"{f['roll_wavelengths']:.1f} wavelengths at the design frequency: the edge, not the section, is "
            "what sets the bandwidth"))
    else:
        steps.append(PlannedStep(
            "bandwidth_skipped", "No bandwidth step", "no band was given, so the study is monochromatic", False))
    steps.append(PlannedStep(
        "axial", "Axial extent of the quiet zone",
        "a quiet zone is a cylinder, not a plane: the same transverse metrics are computed on planes along the "
        "axis to find how deep it is"))
    steps.append(PlannedStep(
        "profile", "Sensitivity to the rolled-edge profile",
        "the edge is only "
        f"{f['roll_wavelengths']:.1f} wavelengths deep, so the quiet-zone numbers depend on a profile the "
        "module parameterises differently from the reference: the spread over the plausible family is "
        "measured, and only conclusions that survive it are reported"))
    steps.append(PlannedStep(
        "tolerance", "Feed placement tolerance",
        "the feed must sit on the focus to a fraction of a wavelength; the study measures how much a "
        "displacement costs in ripple and usable diameter"))
    needs_opt = fr < 2.0 or f["qz_to_section"] > 0.6
    if needs_opt:
        steps.append(PlannedStep(
            "sweep", "Feed gain x section size x rolled edge",
            f"the baseline gives the quiet zone {f['qz_to_section']:.0%} of the section and its Fresnel number "
            f"is {fr:.2f}; whether any (gain, size, roll) inside the family reaches the specification is a "
            "question a map answers and a single design cannot"))
        steps.append(PlannedStep(
            "optimize", "Equal-budget search",
            "a grid and a surrogate-driven search get the same number of physical-optics solves, so the "
            "comparison says what the surrogate is worth rather than what more computer time is worth"))
    else:
        steps.append(PlannedStep("sweep_skipped", "No design sweep",
                                 "the baseline already meets the geometry a quiet zone needs", False))
    return steps


# ==================================================================================================
# Physics helpers (all coupled to the library, none reimplemented)
# ==================================================================================================
def _mesh_and_currents(cr):
    from pinneapple_simulation.electromagnetics.reflector import mesh_reflector
    from pinneapple_simulation.electromagnetics.reflector.po import cosq_feed, po_currents, wavenumber

    k = wavenumber(cr.freq_hz)
    mesh = mesh_reflector(cr.reflector, cr.wavelength / cr.mesh_per_wavelength)
    feed = cr.reflector.focus + np.asarray(cr.feed_offset, float)
    axis = cr.reflector.centre - feed
    pol = (0.0, 1.0, 0.0) if cr.pol.upper() == "V" else (1.0, 0.0, 0.0)
    _, H = cosq_feed(mesh.centres, k, position=feed, axis=axis, q=cr.feed_q, pol=pol)
    J = po_currents(mesh.normals, mesh.centres, H, source=feed)
    return mesh, J, k


def _co_xp(pol: str) -> tuple[int, int]:
    return (1, 0) if pol.upper() == "V" else (0, 1)


def field_on_points(cr, pts: np.ndarray, *, mesh_J=None) -> np.ndarray:
    """PO field at absolute points ``pts`` (M, 3); the model is rebuilt unless ``mesh_J`` is given."""
    from pinneapple_simulation.electromagnetics.reflector.po import radiate

    mesh, J, k = mesh_J or _mesh_and_currents(cr)
    return radiate(mesh.centres, mesh.areas, J, pts, k)


def qz_metrics_at(cr, z: float | None = None, *, amp_tol: float | None = None,
                  phase_tol: float | None = None) -> dict[str, float]:
    """Disk metrics on the quiet-zone plane at ``z`` (default: the design plane)."""
    from pinneapple_simulation.electromagnetics.reflector.compact_range import qz_field
    from pinneapple_simulation.electromagnetics.reflector.qz import qz_metrics

    cr = replace(cr, qz_z=cr.qz_z if z is None else z)
    f = qz_field(cr)
    co, xp = _co_xp(cr.pol)
    out = {}
    for a_, p_ in ((0.5, 5.0), (1.0, 10.0), (1.5, 15.0)):
        m = qz_metrics(f["xy"], f["E"][:, co], cr.qz_diameter, E_x=f["E"][:, xp], amp_tol=a_, phase_tol=p_)
        out[f"usable_at_{a_:g}dB"] = m["usable_diameter"]
        if (a_, p_) == (0.5, 5.0):
            out |= {k: m[k] for k in ("amp_pp_dB", "phase_pp_deg", "taper_dB", "ripple_pp_dB",
                                      "phase_ripple_pp_deg", "xpol_dB", "n_points")}
    out["seconds"] = f["seconds"]
    out["n_triangles"] = int(len(f["mesh"].areas))
    return out


def cut_metrics_at(cr, z: float | None = None, *, half: float | None = None, points: int | None = None,
                   cuts: Sequence[str] = ("horizontal", "vertical")) -> dict[str, dict[str, float]]:
    """Peak-to-peak amplitude and phase and peak cross-polar over cuts through the quiet-zone centre,
    referred to the centre point -- the definitions a range datasheet uses."""
    c = cr.qz_centre
    half = half if half is not None else cr.qz_diameter / 2
    points = points or 121
    t = np.linspace(-half, half, points)
    out = {}
    for kind in cuts:
        rel = np.c_[t, np.zeros_like(t)] if kind == "horizontal" else np.c_[np.zeros_like(t), t]
        pts = np.c_[rel[:, 0] + c[0], rel[:, 1] + c[1], np.full(points, c[2] if z is None else z)]
        E = field_on_points(cr, pts)
        co, xp = _co_xp(cr.pol)
        i = int(np.argmin(np.hypot(rel[:, 0], rel[:, 1])))
        eco, exp_ = E[:, co], E[:, xp]
        A = 20 * np.log10(np.abs(eco) / np.abs(eco[i]))
        P = np.rad2deg(np.unwrap(np.angle(eco / eco[i])))
        X = 20 * np.log10(np.maximum(np.abs(exp_), 1e-30) / np.abs(eco[i]))
        out[kind] = {"amp_pp_dB": float(np.ptp(A)), "phase_pp_deg": float(np.ptp(P)),
                     "xpol_dB": float(X.max()), "t": rel[:, 0] if kind == "horizontal" else rel[:, 1],
                     "amp_dB": A, "phase_deg": P, "xpol_curve_dB": X}
    return out


# ==================================================================================================
# The studies
# ==================================================================================================
def _study_validation(p: Problem, log) -> dict[str, Any]:
    from pinneapple_simulation.electromagnetics.reflector import analytic, geometry, po
    from pinneapple_simulation.electromagnetics.reflector.compact_range import go_check
    from pinneapple_simulation.electromagnetics.reflector.study import validate

    cr = p.cr()
    t0 = time.perf_counter()
    out = validate(cr, mesh_densities=(8, 12), rolls_wl=(3.0, 6.0), log=log)
    k, lam = po.wavenumber(p.freq_hz), p.wavelength
    # a disk of 3 wavelengths, on axis, against the closed form, at five densities
    disk = []
    a = 3 * lam
    z = np.linspace(2 * lam, 40 * lam, 60)
    Ea = analytic.disk_on_axis(z, a, k)
    for mden in (4, 6, 8, 12, 16):
        d = geometry.flat_disk(a, lam / mden)
        _, H = po.plane_wave(d.centres, k, (0, 0, -1), (1, 0, 0))
        J = po.po_currents(d.normals, d.centres, H)
        Ep = po.radiate(d.centres, d.areas, J, np.c_[0 * z, 0 * z, z], k)[:, 0]
        disk.append({"per_wavelength": mden, "triangles": int(len(d.areas)),
                     "max_rel_error": float(np.max(np.abs(Ep - Ea)) / np.max(np.abs(Ea)))})
    out["disk_densities"] = disk
    out["seconds"] = round(time.perf_counter() - t0, 1)
    log(f"validation: PO disk vs closed form {disk[0]['max_rel_error']:.1e} (lambda/{disk[0]['per_wavelength']:g})"
        f" -> {disk[-1]['max_rel_error']:.1e} (lambda/{disk[-1]['per_wavelength']:g})")
    return out


def _study_qz(p: Problem, log) -> dict[str, Any]:
    """Cuts and disk metrics at three planes, both polarizations.  One model per polarisation: the mesh
    and the PO currents are built once and every plane and every cut is one radiation pass."""
    from pinneapple_simulation.electromagnetics.reflector.qz import disk_grid, qz_metrics

    plane_ft = p.qz_z / FT
    zs_ft = (plane_ft - 1.5, plane_ft, plane_ft + 1.5) if abs(plane_ft - 12) > 1e-9 else (10.5, 12.0, 13.5)
    out: dict[str, Any] = {"planes_ft": list(zs_ft), "pol": {}}
    for pol in p.pols:
        cr = p.cr(pol)
        co, xp = _co_xp(pol)
        c = cr.qz_centre
        lam = cr.wavelength
        disk_xy = disk_grid(cr.eval_factor * cr.qz_diameter, lam / cr.qz_step_per_wavelength)
        half = cr.qz_diameter / 2
        t = np.linspace(-half, half, p.cut_points)
        cuts = {"horizontal": np.c_[t, np.zeros_like(t)], "vertical": np.c_[np.zeros_like(t), t]}
        n_disk = len(disk_xy)
        pts_rel = np.vstack([disk_xy] + [cuts[k] for k in cuts])
        mesh, J, kk = _mesh_and_currents(cr)
        d: dict[str, Any] = {}
        for z_ft in zs_ft:
            z = z_ft * FT
            obs = np.c_[pts_rel[:, 0] + c[0], pts_rel[:, 1] + c[1], np.full(len(pts_rel), z)]
            E = field_on_points(cr, obs, mesh_J=(mesh, J, kk))
            Ed, rest = E[:n_disk], E[n_disk:]
            m = qz_metrics(disk_xy, Ed[:, co], cr.qz_diameter, E_x=Ed[:, xp], amp_tol=0.5, phase_tol=5.0)
            m1 = qz_metrics(disk_xy, Ed[:, co], cr.qz_diameter, E_x=Ed[:, xp], amp_tol=1.0, phase_tol=10.0)
            m15 = qz_metrics(disk_xy, Ed[:, co], cr.qz_diameter, E_x=Ed[:, xp], amp_tol=1.5, phase_tol=15.0)
            disk = {"usable_at_0.5dB": m["usable_diameter"], "usable_at_1dB": m1["usable_diameter"],
                    "usable_at_1.5dB": m15["usable_diameter"],
                    **{k: m[k] for k in ("amp_pp_dB", "phase_pp_deg", "taper_dB", "ripple_pp_dB",
                                         "phase_ripple_pp_deg", "xpol_dB", "n_points")},
                    "n_triangles": int(len(mesh.areas))}
            cut_out, i0 = {}, 0
            for name in cuts:
                seg = rest[i0:i0 + p.cut_points]
                i0 += p.cut_points
                i = int(np.argmin(np.hypot(cuts[name][:, 0], cuts[name][:, 1])))
                eco, exp_ = seg[:, co], seg[:, xp]
                A = 20 * np.log10(np.abs(eco) / np.abs(eco[i]))
                P = np.rad2deg(np.unwrap(np.angle(eco / eco[i])))
                X = 20 * np.log10(np.maximum(np.abs(exp_), 1e-30) / np.abs(eco[i]))
                cut_out[name] = {"amp_pp_dB": float(np.ptp(A)), "phase_pp_deg": float(np.ptp(P)),
                                 "xpol_dB": float(X.max()),
                                 "t": cuts[name][:, 0] if name == "horizontal" else cuts[name][:, 1],
                                 "amp_dB": A, "phase_deg": P, "xpol_curve_dB": X}
            d[f"{z_ft:g}"] = {"disk": disk, "cuts": cut_out}
            log(f"{pol} z={z_ft:g} ft: cuts {cut_out['horizontal']['amp_pp_dB']:.2f}/"
                f"{cut_out['horizontal']['phase_pp_deg']:.1f} dB/deg h, "
                f"{cut_out['vertical']['amp_pp_dB']:.2f}/{cut_out['vertical']['phase_pp_deg']:.1f} v | "
                f"disk {disk['amp_pp_dB']:.2f} dB, usable {disk['usable_at_0.5dB'] / p.qz_diameter:.0%} of QZ")
        out["pol"][pol] = d
    return out


def _study_reference(p: Problem, qz: dict[str, Any]) -> dict[str, Any]:
    """Compare the computed cuts with the supplied reference table."""
    rows = []
    for (pol, z_ft, cut), ref in p.reference.items():
        d = qz["pol"].get(pol, {}).get(f"{z_ft:g}")
        if d is None:
            continue
        m = d["cuts"][cut]
        rows.append({"pol": pol, "z_ft": z_ft, "cut": cut,
                     "po_amp_pp_dB": m["amp_pp_dB"], "po_phase_pp_deg": m["phase_pp_deg"],
                     "po_xpol_dB": m["xpol_dB"], "ref_amp_pp_dB": ref[0], "ref_phase_pp_deg": ref[1],
                     "ref_xpol_dB": ref[2],
                     "d_amp_dB": m["amp_pp_dB"] - ref[0], "d_phase_deg": m["phase_pp_deg"] - ref[1]})
    if not rows:
        return {"rows": [], "summary": {}}
    ea = np.array([abs(r["d_amp_dB"]) for r in rows])
    ep = np.array([abs(r["d_phase_deg"]) for r in rows])
    ca = np.array([r["po_amp_pp_dB"] / r["ref_amp_pp_dB"] for r in rows])
    summary = {"n": len(rows), "amp_abs_mean_dB": float(ea.mean()), "amp_abs_max_dB": float(ea.max()),
               "phase_abs_mean_deg": float(ep.mean()), "phase_abs_max_deg": float(ep.max()),
               "amp_ratio_median": float(np.median(ca)),
               "amp_over_predicts_all_rows": bool(np.all(ca > 1.0)),
               "xpol_po_mean_dB": float(np.mean([r["po_xpol_dB"] for r in rows if r["pol"] == "V"])) if any(
                   r["pol"] == "V" for r in rows) else None,
               "xpol_ref_mean_dB": float(np.mean([r["ref_xpol_dB"] for r in rows
                                                  if r["pol"] == "V" and r["ref_xpol_dB"] is not None]))
               if any(r["pol"] == "V" and r["ref_xpol_dB"] is not None for r in rows) else None}
    return {"rows": rows, "summary": summary}


def _geometry_of(p: Problem, **refl) -> dict[str, float]:
    """Extents and surface area of a rolled-edge variant.  Meshed at lambda/6: the extents are a
    geometric property and do not need the production density."""
    from pinneapple_simulation.electromagnetics.reflector import mesh_reflector
    r = p.reflector(**refl)
    m = mesh_reflector(r, p.wavelength / 6.0)
    return {"x_ft": float(np.abs(m.vertices[:, 0]).max() / FT),
            "ymax_ft": float(m.vertices[:, 1].max() / FT),
            "zmin_ft": float(m.vertices[:, 2].min() / FT),
            "area_m2": float(m.area)}


def _calib_row(p: Problem, target: Mapping[str, float], weights: Mapping[str, float],
               roll: float, asp: float, bl: float, end: float) -> dict[str, Any]:
    g = _geometry_of(p, roll=roll * FT, roll_aspect=asp, blend=bl, end_angle=end)
    e = sum(((g[k] - target[k]) / weights[k]) ** 2 for k in target if k in weights)
    return {"roll_ft": roll, "roll_aspect": asp, "blend": bl, "end_angle": end, "error": float(e), **g}


def _profile_row(args) -> dict[str, Any]:
    p, combo = args
    cr = p.cr(pol=p.pols[0])
    cr = replace(cr, reflector=replace(cr.reflector, **combo))
    m = qz_metrics_at(cr)
    return {**combo, **{k: float(v) for k, v in m.items() if isinstance(v, (int, float))}}


def _study_calibration(p: Problem, log, *, parallel: int = 0) -> dict[str, Any]:
    """How close the constant-curvature rolled-edge family can come to the published geometry.

    Only published geometry enters: the extents (x, y, z) and the surface area.  No quiet-zone number is
    fitted, so the comparison in the next step stays a test and not a calibration.
    """
    target = dict(p.reference_geometry or {})
    weights = {"x_ft": 0.08, "ymax_ft": 0.08, "zmin_ft": 0.08, "area_m2": 0.25}
    combos = [(r, a, b, e) for r in (0.4, 0.7, 1.0, 1.3) for a in (0.8, 1.2, 1.6)
              for b in (0.3, 0.6, 0.9) for e in (-60.0, -90.0, -120.0)]
    cands = [_calib_row(p, target, weights, *c) for c in combos]
    cands.sort(key=lambda c: c["error"])
    best = cands[0]
    nominal = _geometry_of(p)
    rel = {k: (best[k] - target[k]) / target[k] for k in target if k in weights and target[k]}
    log(f"edge family: best geometric fit roll {best['roll_ft']:g} ft, aspect {best['roll_aspect']:g}, "
        f"blend {best['blend']:g}, end {best['end_angle']:g} deg -> area {best['area_m2']:.2f} vs "
        f"{target.get('area_m2', float('nan')):.2f} m2 ({100 * rel.get('area_m2', 0):+.0f}%)")
    return {"target": target, "best": best, "nominal": nominal, "relative_error": rel,
            "top": cands[:6], "n_evaluated": len(cands)}


def _study_profile_sensitivity(p: Problem, log, *, parallel: int = 0) -> dict[str, Any]:
    """Quiet-zone metrics over the plausible rolled-edge family: how much of the answer is the edge.

    Runs the reference plane for every combination of (roll, blend, roll_aspect) around the design and
    reports, per metric, the spread.  The spread is what the model family cannot resolve; any conclusion
    that survives it is a conclusion about the design rather than about the profile choice.
    """
    combos = [dict(roll=f * p.roll, blend=b, roll_aspect=a)
              for f in (0.6, 1.0, 1.4) for b in (0.4, 0.7) for a in (0.7, 1.0, 1.4)]
    rows = [_profile_row((p, c)) for c in combos]
    for c, r in zip(combos, rows, strict=True):
        log(f"profile roll {c['roll'] / FT:g} ft, blend {c['blend']:g}, aspect {c['roll_aspect']:g}: "
            f"ripple {r['ripple_pp_dB']:.2f} dB, usable {r['usable_at_0.5dB'] / p.qz_diameter:.0%} of the QZ")
    keys = ("amp_pp_dB", "ripple_pp_dB", "phase_pp_deg", "usable_at_0.5dB", "usable_at_1dB", "taper_dB")
    spread = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        spread[k] = {"min": float(v.min()), "max": float(v.max()), "median": float(np.median(v)),
                     "spread": float(v.max() - v.min())}
    return {"rows": rows, "spread": spread, "n": len(rows)}


def _study_axial(p: Problem, log) -> dict[str, Any]:
    """Usable diameter and ripple along the axis: how deep the quiet zone really is."""
    from pinneapple_simulation.electromagnetics.reflector.qz import qz_metrics

    out = {"z_ft": [], "pol": {}}
    cr = p.cr()
    c = cr.qz_centre
    half_w = cr.eval_factor * cr.qz_diameter / 2
    lam = cr.wavelength
    n = int(np.floor(half_w / (lam / cr.qz_step_per_wavelength)))
    g = np.arange(-n, n + 1) * (lam / cr.qz_step_per_wavelength)
    X, Y = np.meshgrid(g, g)
    keep = X ** 2 + Y ** 2 <= half_w ** 2
    xy = np.c_[X[keep], Y[keep]]
    zs = p.qz_z + np.linspace(-1.2, 1.2, 9)          # metres either side of the design plane
    for pol in p.pols:
        crp = p.cr(pol)
        co, xp = _co_xp(pol)
        mesh, J, k = _mesh_and_currents(crp)
        d = {"usable_at_0.5dB": [], "ripple_pp_dB": [], "amp_pp_dB": [], "phase_pp_deg": [],
             "usable_at_1dB": []}
        for z in zs:
            pts = np.c_[xy[:, 0] + c[0], xy[:, 1] + c[1], np.full(len(xy), z)]
            E = field_on_points(crp, pts, mesh_J=(mesh, J, k))
            m = qz_metrics(xy, E[:, co], crp.qz_diameter, E_x=E[:, xp], amp_tol=0.5, phase_tol=5.0)
            m1 = qz_metrics(xy, E[:, co], crp.qz_diameter, E_x=E[:, xp], amp_tol=1.0, phase_tol=10.0)
            d["usable_at_0.5dB"].append(m["usable_diameter"])
            d["usable_at_1dB"].append(m1["usable_diameter"])
            d["ripple_pp_dB"].append(m["ripple_pp_dB"])
            d["amp_pp_dB"].append(m["amp_pp_dB"])
            d["phase_pp_deg"].append(m["phase_pp_deg"])
        out["pol"][pol] = d
        ok = [z / FT for z, u in zip(zs, d["usable_at_1dB"]) if u >= p.qz_diameter]
        log(f"{pol} axial: {len(zs)} planes, the 1 dB usable diameter reaches the full QZ on "
            f"{len(ok)} of them" + (f" ({min(ok):.1f} to {max(ok):.1f} ft)" if ok else ""))
    out["z_ft"] = list(zs / FT)
    return out


def _study_bandwidth(p: Problem, log) -> dict[str, Any]:
    from pinneapple_simulation.electromagnetics.reflector.qz import qz_metrics

    lo, hi = p.band_hz
    freqs = np.linspace(lo, hi, 6)
    out = {"freq_ghz": list(freqs / 1e9), "pol": {}}
    for pol in p.pols:
        d = {"usable_at_0.5dB": [], "usable_at_1dB": [], "ripple_pp_dB": [], "amp_pp_dB": [],
             "phase_pp_deg": [], "taper_dB": []}
        for f in freqs:
            cr = p.cr(pol, freq_hz=float(f))
            m = qz_metrics_at(cr)
            d["usable_at_0.5dB"].append(m["usable_at_0.5dB"])
            d["usable_at_1dB"].append(m["usable_at_1dB"])
            d["ripple_pp_dB"].append(m["ripple_pp_dB"])
            d["amp_pp_dB"].append(m["amp_pp_dB"])
            d["phase_pp_deg"].append(m["phase_pp_deg"])
            d["taper_dB"].append(m["taper_dB"])
        out["pol"][pol] = d
        log(f"{pol} band {lo / 1e9:g}-{hi / 1e9:g} GHz: usable(0.5 dB) "
            f"{min(d['usable_at_0.5dB']) / p.qz_diameter:.0%} to {max(d['usable_at_0.5dB']) / p.qz_diameter:.0%} "
            f"of the QZ")
    return out


def _study_tolerance(p: Problem, log) -> dict[str, Any]:
    """How much a feed displacement costs: lateral (x), axial (z) and diagonal, in wavelengths."""
    lam = p.wavelength
    steps = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0]) * lam
    out = {"offset_wavelengths": list(steps / lam), "pol": {}}
    for pol in p.pols:
        d = {"lateral": [], "axial": [], "diagonal": []}
        for axis in ("lateral", "axial", "diagonal"):
            for s in steps:
                off = {"lateral": (s, 0.0, 0.0), "axial": (0.0, 0.0, s), "diagonal": (s / np.sqrt(2), 0.0,
                                                                                        s / np.sqrt(2))}[axis]
                cr = p.cr(pol, feed_offset=off)
                m = qz_metrics_at(cr)
                d[axis].append({"offset_lambda": float(s / lam), "usable_at_0.5dB": m["usable_at_0.5dB"],
                                "usable_at_1dB": m["usable_at_1dB"], "ripple_pp_dB": m["ripple_pp_dB"],
                                "amp_pp_dB": m["amp_pp_dB"], "phase_pp_deg": m["phase_pp_deg"]})
            log(f"{pol} tolerance {axis}: usable(0.5 dB) {d[axis][0]['usable_at_0.5dB'] / p.qz_diameter:.0%} "
                f"-> {d[axis][-1]['usable_at_0.5dB'] / p.qz_diameter:.0%} of the QZ over 3 wavelengths")
        out["pol"][pol] = d
    return out


def _design_row(p: Problem, gain: float, side: float, roll: float, pol: str = "V") -> dict[str, Any]:
    t0 = time.perf_counter()
    cr = p.cr(pol, feed_gain_dbi=gain, section=side)
    cr = replace(cr, reflector=replace(cr.reflector, roll=roll))
    m = qz_metrics_at(cr)
    return {"feed_gain_dBi": float(gain), "side_m": float(side), "roll_m": float(roll),
            **{k: float(v) for k, v in m.items()}, "seconds": round(time.perf_counter() - t0, 2)}


def _study_sweep(p: Problem, log, *, parallel: int = 0) -> dict[str, Any]:
    """Feed gain x section side x rolled edge, with a GP per metric and held-out errors."""
    import multiprocessing as mp

    lam = p.wavelength
    gains = [8.0, 12.0, 16.0, 20.0]
    sides = [0.9, 1.4, 1.9, 2.4]                                # metres
    rolls = [p.roll, 0.5 * p.roll, 1.5 * p.roll]
    jobs = [(g, s, r) for r in rolls for s in sides for g in gains]
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if parallel and parallel > 1:
        with mp.get_context("fork").Pool(parallel, maxtasksperchild=8) as pool:
            args = [(p, g, s, r) for g, s, r in jobs]
            for i, row in enumerate(pool.starmap(_design_row, args, chunksize=1), 1):
                rows.append(row)
                if i % 10 == 0:
                    log(f"sweep {i}/{len(jobs)}")
    else:
        for i, (g, s, r) in enumerate(jobs, 1):
            rows.append(_design_row(p, g, s, r))
            if i % 10 == 0:
                log(f"sweep {i}/{len(jobs)}")
    wall = time.perf_counter() - t0
    # GP surrogates: a random quarter held out, and one side held out at a time
    metrics = ("ripple_pp_dB", "amp_pp_dB", "phase_pp_deg", "taper_dB", "usable_at_0.5dB")
    models, held = {}, {}
    rng = np.random.default_rng(0)
    for mname in metrics:
        ys = np.array([r[mname] for r in rows])
        feat = np.array([[r["feed_gain_dBi"], r["side_m"]] for r in rows])
        pred, err = _gp_holdout(feat, ys, rng, test_frac=0.25)
        lso = _gp_leave_one_group(feat, ys, np.array([r["side_m"] for r in rows]))
        held[mname] = {"rmse_random": float(err), "rmse_leave_side_out": float(lso)}
        models[mname] = (feat, ys)
    best = min(rows, key=lambda r: (_feasible_rank(r, p), r["ripple_pp_dB"]))
    log(f"sweep: {len(rows)} designs in {wall:.0f} s (median {np.median([r['seconds'] for r in rows]):.1f} s)")
    return {"rows": rows, "gains": gains, "sides": sides, "rolls": rolls, "wall_s": round(wall, 1),
            "median_s": float(np.median([r["seconds"] for r in rows])), "heldout": held, "best": best,
            "n": len(rows)}


def _feasible_rank(r: Mapping[str, Any], p: Problem):
    """Sort key: designs that reach the quiet zone first, then the least ripple."""
    reached = r["usable_at_0.5dB"] >= p.qz_diameter
    return (0 if reached else 1, -r["usable_at_0.5dB"], r["ripple_pp_dB"])


def _gp_fit_predict(Xtr, ytr, Xq):
    """A Gaussian process with an anisotropic squared-exponential kernel, fitted by marginal likelihood."""
    from scipy.optimize import minimize

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    U, Uq = (Xtr - mu) / sd, (Xq - mu) / sd
    ymu, ysd = ytr.mean(), ytr.std() + 1e-12
    t = (ytr - ymu) / ysd
    d = U.shape[1]

    def nll(th):
        ell, s2, n2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
        D = ((U[:, None, :] - U[None, :, :]) / ell) ** 2
        K = s2 * np.exp(-0.5 * D.sum(-1)) + (n2 + 1e-8) * np.eye(len(t))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, t))
        return float(0.5 * t @ a + np.log(np.diag(L)).sum())

    r = minimize(nll, np.zeros(d + 2) - 0.5, method="L-BFGS-B", bounds=[(-4, 4)] * d + [(-8, 4), (-10, 2)])
    th = r.x
    ell, s2, n2 = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
    D = ((U[:, None, :] - U[None, :, :]) / ell) ** 2
    K = s2 * np.exp(-0.5 * D.sum(-1)) + (n2 + 1e-8) * np.eye(len(t))
    Dq = ((Uq[:, None, :] - U[None, :, :]) / ell) ** 2
    Ks = s2 * np.exp(-0.5 * Dq.sum(-1))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, t))
    return Ks @ alpha * ysd + ymu


def _gp_holdout(X: np.ndarray, y: np.ndarray, rng, test_frac: float = 0.25) -> tuple[np.ndarray, float]:
    n = len(X)
    idx = rng.permutation(n)
    k = max(2, int(round(test_frac * n)))
    te, tr = idx[:k], idx[k:]
    pred = _gp_fit_predict(X[tr], y[tr], X[te])
    return pred, float(np.sqrt(np.mean((pred - y[te]) ** 2)))


def _gp_leave_one_group(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    errs = []
    for val in np.unique(g)[1:-1]:
        te = g == val
        pred = _gp_fit_predict(X[~te], y[~te], X[te])
        errs += list(pred - y[te])
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _study_optimize(p: Problem, log, *, budget: int = 30) -> dict[str, Any]:
    """Equal solver budget: a grid, a random DOE whose GP is optimised and then verified, and Bayesian
    optimisation on the solver itself.  The objective is the negative usable diameter at +-0.5 dB."""
    import multiprocessing as mp

    from pinneapple_data.parameter_sampling import sample_parameters

    ranges = {"feed_gain_dBi": (8.0, 20.0), "side_m": (0.9, 2.4), "roll_m": (0.5 * p.roll, 2.0 * p.roll)}
    names = list(ranges)
    n_verify = 4
    cache: dict[tuple, dict] = {}

    def solve(designs: list[dict[str, float]], pool=None) -> list[dict]:
        todo, keys = [], []
        for d in designs:
            k = tuple(round(float(d[n]), 6) for n in names)
            keys.append(k)
            if k not in cache:
                cache[k] = None
                todo.append((k, d))
        if todo:
            args = [(p, d["feed_gain_dBi"], d["side_m"], d["roll_m"]) for _, d in todo]
            got = pool.starmap(_design_row, args, chunksize=1) if pool else [_design_row(*a) for a in args]
            for (k, _), row in zip(todo, got, strict=True):
                cache[k] = row
        return [cache[k] for k in keys]

    t00 = time.perf_counter()
    with mp.get_context("fork").Pool(max(1, min(8, os.cpu_count() or 1)), maxtasksperchild=8) as pool:
        calls = {"n": 0}
        # (a) regular grid: 4 x 3 x 2 = 24 + the baseline = within the budget
        grid = [{"feed_gain_dBi": g, "side_m": s, "roll_m": r}
                for r in (0.6 * main_roll(p), 1.6 * main_roll(p))
                for s in (1.2, 1.7, 2.2)
                for g in (10.0, 14.0, 18.0)]
        t0 = time.perf_counter()
        g_rows = solve(grid, pool)
        calls["n"] += len(grid)
        g_best = min(g_rows, key=lambda r: _feasible_rank(r, p))
        grid_out = {"calls": len(grid), "best": g_best, "wall_s": round(time.perf_counter() - t0, 1),
                    "trace": list(np.maximum.accumulate([r["usable_at_0.5dB"] for r in g_rows]))}
        log(f"equal budget: grid {len(grid)} solves -> usable "
            f"{g_best['usable_at_0.5dB'] / p.qz_diameter:.0%} of the QZ")

        # (b) LHS DOE + GP + optimiser on the GP + verification of the best candidates
        n_doe = budget - n_verify
        doe = [{k: float(v) for k, v in d.items()} for d in
               sample_parameters(dict(ranges), n_doe, method="lhs", seed=0)]
        t0 = time.perf_counter()
        d_rows = solve(doe, pool)
        calls["n"] += len(doe)
        X = np.array([[r[n] for n in names] for r in d_rows])
        y = np.array([-r["usable_at_0.5dB"] for r in d_rows])
        cand = np.array([[d[k] for k in names] for d in
                         sample_parameters(dict(ranges), 4000, method="sobol", seed=7)])
        pred = _gp_fit_predict(X, y, cand)
        pick = cand[np.argsort(pred)[:n_verify]]
        v_rows = solve([dict(zip(names, map(float, c), strict=True)) for c in pick], pool)
        calls["n"] += len(v_rows)
        all_rows = d_rows + v_rows
        s_best = min(all_rows, key=lambda r: _feasible_rank(r, p))
        pred_sel = pred[np.argsort(pred)[:n_verify]]
        surrogate_out = {"calls": len(all_rows), "best": s_best,
                         "wall_s": round(time.perf_counter() - t0, 1),
                         "gp_rmse": float(np.sqrt(np.mean((pred[np.argsort(pred)[:n_verify]] +
                                                          np.array([r["usable_at_0.5dB"] for r in v_rows])) ** 2)))
                         if v_rows else None,
                         "verify": [{"predicted_usable": float(-pp), "solver_usable": float(r["usable_at_0.5dB"])}
                                    for pp, r in zip(pred_sel, v_rows, strict=True)],
                         "trace": list(np.maximum.accumulate([r["usable_at_0.5dB"] for r in all_rows]))}
        log(f"equal budget: LHS + GP + verify {len(all_rows)} solves -> usable "
            f"{s_best['usable_at_0.5dB'] / p.qz_diameter:.0%} of the QZ")

        # (c) Bayesian optimisation on the solver (same budget, sequential)
        t0 = time.perf_counter()
        bo_rows: list[dict] = []
        rng = np.random.default_rng(0)
        for i in range(budget):
            if i < 6:
                d = {k: float(rng.uniform(*ranges[k])) for k in names}
            else:
                Xb = np.array([[r[n] for n in names] for r in bo_rows])
                yb = np.array([-r["usable_at_0.5dB"] for r in bo_rows])
                C = np.array([[rng.uniform(*ranges[k]) for k in names] for _ in range(1500)])
                m = _gp_fit_predict(Xb, yb, C)
                d = dict(zip(names, map(float, C[int(np.argmin(m))]), strict=True))
            bo_rows.append(solve([d], pool)[0])
        calls["n"] += len(bo_rows)
        b_best = min(bo_rows, key=lambda r: _feasible_rank(r, p))
        bo_out = {"calls": len(bo_rows), "best": b_best, "wall_s": round(time.perf_counter() - t0, 1),
                  "trace": list(np.maximum.accumulate([r["usable_at_0.5dB"] for r in bo_rows]))}
        log(f"equal budget: Bayesian {len(bo_rows)} solves -> usable "
            f"{b_best['usable_at_0.5dB'] / p.qz_diameter:.0%} of the QZ")

    return {"budget": budget, "grid": grid_out, "surrogate": surrogate_out, "bayesian": bo_out,
            "wall_s": round(time.perf_counter() - t00, 1), "n_unique_solves": len(cache),
            "target_m": p.qz_diameter}


def main_roll(p: Problem) -> float:
    return p.roll


# ==================================================================================================
# Running the plan
# ==================================================================================================
def _problem_snapshot(p: Problem) -> dict[str, Any]:
    """The problem as JSON (the reference table's keys are tuples, so they are flattened first)."""
    snap = {k: v for k, v in asdict(p).items()}
    if p.reference:
        snap["reference"] = {f"{pol}|{z:g}|{cut}": list(v) for (pol, z, cut), v in p.reference.items()}
    return _jsonable(snap)


def run(p: Problem, out: str = "cr_study", *, steps: Sequence[str] | None = None, parallel: int = 0,
        log: Callable[[str], None] = print, reuse: bool = True) -> dict[str, Any]:
    """Execute the plan (or the ``steps`` named) and return the study; every number is cached in
    ``<out>/study.json`` so the document can be re-rendered without solving again."""
    os.makedirs(out, exist_ok=True)
    cache_path = os.path.join(out, "study.json")
    study: dict[str, Any] = {"problem": _problem_snapshot(p), "results": {},
                             "timings": {}, "log": []}
    if reuse and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                old = json.load(fh)
            if old.get("problem") == study["problem"]:
                study = old
        except (ValueError, OSError):
            pass
    want = set(steps) if steps else {s.key for s in plan(p) if s.required}
    R = study["results"]

    def need(key: str) -> bool:
        return key in want and key not in R

    def save():
        with open(cache_path, "w") as fh:
            json.dump(study, fh, indent=1, default=_jsonable)

    try:
        if need("model"):
            t0 = time.perf_counter()
            R["model"] = p.facts()
            study["timings"]["model"] = round(time.perf_counter() - t0, 1)
            save()
        if need("validation"):
            t0 = time.perf_counter()
            R["validation"] = _study_validation(p, log)
            study["timings"]["validation"] = round(time.perf_counter() - t0, 1)
            save()
        if need("qz"):
            t0 = time.perf_counter()
            R["qz"] = _study_qz(p, log)
            study["timings"]["qz"] = round(time.perf_counter() - t0, 1)
            save()
        if "reference" in want and p.reference and "qz" in R:
            R["reference"] = _study_reference(p, R["qz"])
            save()
        if need("calibration") and p.reference_geometry:
            t0 = time.perf_counter()
            R["calibration"] = _study_calibration(p, log, parallel=parallel)
            study["timings"]["calibration"] = round(time.perf_counter() - t0, 1)
            save()
        if need("profile"):
            t0 = time.perf_counter()
            R["profile"] = _study_profile_sensitivity(p, log, parallel=parallel)
            study["timings"]["profile"] = round(time.perf_counter() - t0, 1)
            save()
        if need("axial"):
            t0 = time.perf_counter()
            R["axial"] = _study_axial(p, log)
            study["timings"]["axial"] = round(time.perf_counter() - t0, 1)
            save()
        if need("bandwidth") and p.band_hz:
            t0 = time.perf_counter()
            R["bandwidth"] = _study_bandwidth(p, log)
            study["timings"]["bandwidth"] = round(time.perf_counter() - t0, 1)
            save()
        if need("tolerance"):
            t0 = time.perf_counter()
            R["tolerance"] = _study_tolerance(p, log)
            study["timings"]["tolerance"] = round(time.perf_counter() - t0, 1)
            save()
        if need("sweep"):
            t0 = time.perf_counter()
            R["sweep"] = _study_sweep(p, log, parallel=parallel)
            study["timings"]["sweep"] = round(time.perf_counter() - t0, 1)
            save()
        if need("optimize"):
            t0 = time.perf_counter()
            R["optimize"] = _study_optimize(p, log)
            study["timings"]["optimize"] = round(time.perf_counter() - t0, 1)
            save()
    finally:
        save()
    return study


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


# ==================================================================================================
# Rendering
# ==================================================================================================
def report(p: Problem, study: Mapping[str, Any], out: str = "report", *,
           pdf_name: str | None = None) -> dict[str, str | None]:
    """Render the study as a technical datasheet (HTML, and PDF when a browser is available)."""
    from .study_report_figures import build_figures
    from .study_report_text import build_datasheet

    os.makedirs(out, exist_ok=True)
    figs = build_figures(p, study, out)
    sheet = build_datasheet(p, study, figs)
    base = pdf_name or p.name
    res = sheet.build(os.path.join(out, f"{base}.html"), pdf=os.path.join(out, f"{base}.pdf"))
    return res


if __name__ == "__main__":                                  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freq", type=float, default=3.0, help="GHz")
    ap.add_argument("--focal", type=float, default=1.5, help="focal length (m, or ft with --feet)")
    ap.add_argument("--section", type=float, default=1.2, help="square section side")
    ap.add_argument("--offset", type=float, default=0.9, help="section centre above the axis")
    ap.add_argument("--roll", type=float, default=0.15, help="rolled-edge radius")
    ap.add_argument("--gain", type=float, default=12.0, help="feed gain [dBi]")
    ap.add_argument("--qz", type=float, default=0.6, help="quiet-zone diameter")
    ap.add_argument("--qz-distance", type=float, default=3.0, help="quiet-zone plane, from the vertex")
    ap.add_argument("--band", type=float, nargs=2, help="band edges in GHz")
    ap.add_argument("--feet", action="store_true", help="the lengths are in feet instead of metres")
    ap.add_argument("--out", default="cr_study")
    ap.add_argument("--parallel", type=int, default=0)
    a = ap.parse_args()
    u = FT if a.feet else 1.0
    prob = Problem(freq_hz=a.freq * 1e9, focal=a.focal * u, section=a.section * u, offset=a.offset * u,
                   roll=a.roll * u, feed_gain_dbi=a.gain, qz_diameter=a.qz * u, qz_z=a.qz_distance * u,
                   band_hz=tuple(x * 1e9 for x in a.band) if a.band else None)
    print("plan:", [(s.key, s.required) for s in plan(prob)])
    run(prob, a.out, parallel=a.parallel)
    print(report(prob, json.load(open(os.path.join(a.out, "study.json"))), os.path.join(a.out, "report")))
